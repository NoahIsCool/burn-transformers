//! Model-based property test of the tensor decode-state lifecycle.
//!
//! Random sequences of append, step publication, write-ahead (published,
//! abandoned, or invalidated by an interfering lifecycle operation), begin,
//! stage, commit, rollback, reorder, truncate, and reset drive a real
//! [`TensorDecodeState`] holding full, standard-ring, and mirrored-ring KV
//! groups, a full latent group with in-place staging, and an optional
//! recurrent group. Groups have one or two layers and one or two heads and
//! channels.
//!
//! A straightforward oracle keeps each row's token history as a `Vec<f32>`.
//! Every token carries a unique value, and each `(layer, head, channel)`
//! entry is a distinct exact function of it ([`code`]), so a stale,
//! misplaced, cross-layer, or lost cache entry is detected exactly (F32
//! equality, no tolerance). After every operation the test compares
//! committed and working lengths and, for every group and layer, both the
//! checked per-row read ([`TensorDecodeState::read_kv`]) and the batched
//! slot-order read ([`TensorDecodeState::kv_read_plan`] plus
//! [`NativeKvCache::read_batch`], rearranged chronologically by its key
//! positions and validity) with the oracle, plus the recurrent terminal
//! state. Write-ahead steps are also checked through
//! [`TensorDecodeState::read_kv_through`] before they finish. Rejected
//! operations must leave both the state and the oracle unchanged.

use burn::backend::Flex;
use burn::tensor::{Bool, DType, Int, Tensor, TensorData};
use core::num::NonZeroUsize;
use proptest::prelude::*;

use super::*;
use crate::cache::{KvBatchRead, KvBatchReadPlan, KvCacheStorage};

/// Deterministic host backend for exact value comparisons.
type TestBackend = Flex<f32, i32>;

/// Index of the full-storage KV group (separate staging).
const FULL: usize = 0;
/// Index of the standard sliding-ring KV group.
const RING: usize = 1;
/// Index of the mirrored sliding-ring KV group.
const MIRRORED: usize = 2;
/// Index of the full-storage latent group with in-place staging.
const LATENT: usize = 3;
/// Width of the recurrent group's convolution history.
const CONVOLUTION_WIDTH: usize = 2;
/// Value of padding columns in batched inputs; never a valid cache entry.
const SENTINEL: f32 = -7.0;

/// Geometry of one generated test case.
#[derive(Debug, Clone, Copy)]
struct Setup {
    /// Number of batch rows.
    rows: usize,
    /// Logical capacity of every row.
    capacity: usize,
    /// Window of the standard ring group.
    ring_window: usize,
    /// Window of the mirrored ring group.
    mirrored_window: usize,
    /// Proposal tokens each group can stage per row.
    staging: usize,
    /// Whether a recurrent group participates (it restricts rewinds).
    recurrent: bool,
    /// Layers of every KV and latent group (1 or 2).
    layers: usize,
    /// KV heads and channels per head (1 or 2); the latent width is one more.
    width: usize,
}

impl Setup {
    /// Returns `(heads, channels)` of a group's planes.
    const fn plane_shape(&self, group: usize) -> (usize, usize) {
        if group == LATENT {
            (1, self.width + 1)
        } else {
            (self.width, self.width)
        }
    }

    /// Returns the ring width of a group, `None` for full storage.
    const fn window(&self, group: usize) -> Option<usize> {
        match group {
            RING => Some(self.ring_window),
            MIRRORED => Some(self.mirrored_window),
            _ => None,
        }
    }
}

/// How a write-ahead step ends.
#[derive(Debug, Clone, Copy)]
enum Finish {
    /// Complete the step and publish it.
    Publish,
    /// Drop the step; its tokens stay in slots beyond the lengths.
    Abandon,
    /// Run a lifecycle operation first (selected by the value); publication
    /// must then be refused as stale.
    Interfere(u8),
}

/// One lifecycle operation; vectors may have the wrong length on purpose.
#[derive(Debug, Clone)]
enum Operation {
    /// Append (or, inside a transaction, stage) per-row token counts.
    Append(Vec<usize>),
    /// Publish one step write set, batched or per row, in or out of a transaction.
    Step(Vec<usize>, bool),
    /// Write the full-storage groups ahead, read through them, then finish.
    WriteAhead(Vec<usize>, Finish),
    /// Begin a transaction.
    Begin,
    /// Commit per-row accepted prefixes.
    Commit(Vec<usize>),
    /// Roll the transaction back.
    Rollback,
    /// Gather rows by source index.
    Reorder(Vec<usize>),
    /// Rewind committed rows.
    Truncate(Vec<usize>),
    /// Clear every row.
    Reset,
}

/// Naive reference: per-row token histories and an optional proposal suffix.
#[derive(Debug, Clone, PartialEq)]
struct Oracle {
    /// Committed token values per row.
    committed: Vec<Vec<f32>>,
    /// Proposed token values per row while a transaction is active.
    proposals: Option<Vec<Vec<f32>>>,
}

impl Oracle {
    /// Returns committed lengths.
    fn lengths(&self) -> Vec<usize> {
        self.committed.iter().map(Vec::len).collect()
    }

    /// Returns each row's full working history: committed tokens then proposals.
    fn working(&self, row: usize) -> Vec<f32> {
        let mut tokens = self.committed[row].clone();
        if let Some(proposals) = &self.proposals {
            tokens.extend(&proposals[row]);
        }
        tokens
    }
}

/// Entry `(head, channel)` of token `value` in `layer`.
///
/// Token values are positive integers, so `16 * value + offset` with offset
/// `8 * layer + 4 * head + channel < 16` (layer, head `<= 1`, channel `<= 2`)
/// is a distinct, exactly representable F32 integer for every entry.
fn code(value: f32, layer: usize, head: usize, channel: usize) -> f32 {
    let offset = u8::try_from(8 * layer + 4 * head + channel).expect("offset below 16");
    value * 16.0 + f32::from(offset)
}

/// Flat `[rows, heads, block, channels]` data of per-row tokens in `layer`.
///
/// Row `b` occupies the first `tokens[b].len()` columns (right padding);
/// other columns hold [`SENTINEL`]. `sign` negates values relative to keys.
fn plane_data(
    tokens: &[Vec<f32>],
    block: usize,
    (heads, channels): (usize, usize),
    layer: usize,
    sign: f32,
) -> Vec<f32> {
    let mut data = Vec::with_capacity(tokens.len() * heads * block * channels);
    for row in tokens {
        for head in 0..heads {
            for slot in 0..block {
                for channel in 0..channels {
                    data.push(
                        row.get(slot)
                            .map_or(SENTINEL, |&value| sign * code(value, layer, head, channel)),
                    );
                }
            }
        }
    }
    data
}

/// Builds `[rows, heads, block, channels]` keys and negated values of `layer`.
fn kv_input(
    shape: (usize, usize),
    tokens: &[Vec<f32>],
    block: usize,
    layer: usize,
) -> (Tensor<TestBackend, 4>, Tensor<TestBackend, 4>) {
    let dims = [tokens.len(), shape.0, block, shape.1];
    let tensor = |sign| {
        Tensor::from_data(
            TensorData::new(plane_data(tokens, block, shape, layer, sign), dims),
            &Default::default(),
        )
    };
    (tensor(1.0), tensor(-1.0))
}

/// Builds a latent `[rows, block, width]` of `layer`.
fn latent_input(
    width: usize,
    tokens: &[Vec<f32>],
    block: usize,
    layer: usize,
) -> Tensor<TestBackend, 3> {
    // With one head, `[rows, 1, block, width]` data is `[rows, block, width]`.
    let data = plane_data(tokens, block, (1, width), layer, 1.0);
    Tensor::from_data(
        TensorData::new(data, [tokens.len(), block, width]),
        &Default::default(),
    )
}

/// Live state, oracle, and the active transaction mark.
struct Harness {
    /// Generated geometry.
    setup: Setup,
    /// System under test.
    state: TensorDecodeState<TestBackend>,
    /// Reference model.
    oracle: Oracle,
    /// Mark of the active transaction, if any.
    mark: Option<TensorCacheMark>,
    /// Next unique token value.
    next_value: f32,
}

impl Harness {
    /// Allocates the state with three KV groups, a latent group, and an optional recurrent group.
    fn new(setup: Setup) -> Self {
        let config = CacheConfig::new(setup.rows, setup.capacity).expect("valid config");
        let mut state = TensorDecodeState::with_config(config, &Default::default());
        for (group, storage) in [
            (FULL, KvCacheStorage::Full),
            (RING, KvCacheStorage::SlidingRing),
            (MIRRORED, KvCacheStorage::MirroredSlidingRing),
        ] {
            let mut spec =
                NativeKvCacheSpec::new(setup.layers, setup.width, setup.width, storage, DType::F32)
                    .expect("geometry")
                    .with_staging_capacity(setup.staging);
            if let Some(window) = setup.window(group) {
                spec = spec.with_sliding_window(NonZeroUsize::new(window).expect("window"));
            }
            state.main_mut().add_native_kv(spec).expect("KV group");
        }
        let latent = NativeKvCacheSpec::latent(
            setup.layers,
            setup.plane_shape(LATENT).1,
            KvCacheStorage::Full,
            DType::F32,
        )
        .expect("geometry")
        .with_staging_capacity(setup.staging)
        .with_in_place_staging();
        state
            .main_mut()
            .add_native_kv(latent)
            .expect("latent group");
        if setup.recurrent {
            let spec = RecurrentStateSpec::new(1, 1, 1, 1, 1, CONVOLUTION_WIDTH, DType::F32)
                .expect("geometry")
                .with_staging_capacity(setup.staging);
            state
                .main_mut()
                .add_recurrent(spec)
                .expect("recurrent group");
        }
        Self {
            setup,
            state,
            oracle: Oracle {
                committed: vec![Vec::new(); setup.rows],
                proposals: None,
            },
            mark: None,
            next_value: 1.0,
        }
    }

    /// Allocates fresh token values for every row.
    fn fresh_tokens(&mut self, counts: &[usize]) -> Vec<Vec<f32>> {
        counts
            .iter()
            .map(|&count| {
                (0..count)
                    .map(|_| {
                        let value = self.next_value;
                        self.next_value += 1.0;
                        value
                    })
                    .collect()
            })
            .collect()
    }

    /// Applies one operation to both systems; returns whether it was accepted.
    fn apply(&mut self, operation: &Operation) -> bool {
        match operation {
            Operation::Append(counts) if self.oracle.proposals.is_some() => self.stage(counts),
            // Ordinary appends publish through the per-row step API, the path every
            // family's stateful call uses; batched pushes are `Step(_, true)`.
            Operation::Append(counts) => self.step(counts, false),
            Operation::Step(counts, batched) => self.step(counts, *batched),
            Operation::WriteAhead(counts, finish) => self.write_ahead(counts, *finish),
            Operation::Begin => self.begin(),
            Operation::Commit(accepted) => self.commit(accepted),
            Operation::Rollback => self.rollback(),
            Operation::Reorder(indices) => self.reorder(indices),
            Operation::Truncate(lengths) => self.truncate(lengths),
            Operation::Reset => self.reset(),
        }
    }

    /// Stages proposal tokens in every group and layer, then advances the working cursor.
    fn stage(&mut self, counts: &[usize]) -> bool {
        let rows = self.setup.rows;
        let proposals = self.oracle.proposals.as_ref().expect("active transaction");
        if counts.len() != rows {
            assert!(self.state.advance(counts).is_err());
            return false;
        }
        let over_capacity = (0..rows).any(|row| {
            self.oracle.committed[row].len() + proposals[row].len() + counts[row]
                > self.setup.capacity
        });
        if over_capacity {
            assert!(self.state.advance(counts).is_err());
            return false;
        }
        if let Some(row) =
            (0..rows).find(|&row| proposals[row].len() + counts[row] > self.setup.staging)
        {
            // Staging capacity is checked before the first tensor assignment.
            let shape = self.setup.plane_shape(FULL);
            let (keys, values) = kv_input(shape, &[vec![1.0; counts[row]]], counts[row], 0);
            let cache = self.state.main_mut().native_kv_mut(FULL).expect("group");
            assert!(matches!(
                cache.stage_sequence(0, row, keys, values),
                Err(TensorCacheError::StagingCapacityExceeded { .. })
            ));
            return false;
        }
        let tokens = self.fresh_tokens(counts);
        for row in (0..rows).filter(|&row| counts[row] > 0) {
            let row_tokens = [tokens[row].clone()];
            for layer in 0..self.setup.layers {
                for group in [FULL, RING, MIRRORED] {
                    let shape = self.setup.plane_shape(group);
                    let (keys, values) = kv_input(shape, &row_tokens, counts[row], layer);
                    self.state
                        .main_mut()
                        .native_kv_mut(group)
                        .expect("group")
                        .stage_sequence(layer, row, keys, values)
                        .expect("valid staging");
                }
                let width = self.setup.plane_shape(LATENT).1;
                self.state
                    .main_mut()
                    .native_kv_mut(LATENT)
                    .expect("latent group")
                    .stage_latent_sequence(
                        layer,
                        row,
                        latent_input(width, &row_tokens, counts[row], layer),
                    )
                    .expect("valid latent staging");
            }
            if self.setup.recurrent {
                let (matrices, convolutions) = recurrent_snapshots(&tokens[row]);
                self.state
                    .main_mut()
                    .recurrent_mut(0)
                    .expect("recurrent group")
                    .stage_sequence_snapshots(0, row, matrices, convolutions)
                    .expect("valid snapshots");
            }
        }
        self.state.advance(counts).expect("validated advance");
        let proposals = self.oracle.proposals.as_mut().expect("active transaction");
        for (proposal, new) in proposals.iter_mut().zip(tokens) {
            proposal.extend(new);
        }
        true
    }

    /// Returns whether a step of `counts` fits capacity and, in a transaction, staging.
    fn step_fits(&self, counts: &[usize]) -> bool {
        (0..self.setup.rows).all(|row| {
            let staged = self
                .oracle
                .proposals
                .as_ref()
                .map_or(0, |proposals| proposals[row].len());
            self.oracle.working(row).len() + counts[row] <= self.setup.capacity
                && (self.oracle.proposals.is_none() || staged + counts[row] <= self.setup.staging)
        })
    }

    /// Records one group's layers into a step, batched or per row.
    fn push_group(
        &self,
        step: &mut TensorStepWrites<TestBackend>,
        group: usize,
        tokens: &[Vec<f32>],
        batched: bool,
    ) {
        let shape = self.setup.plane_shape(group);
        let block = tokens.iter().map(Vec::len).max().unwrap_or(0).max(1);
        for layer in 0..self.setup.layers {
            if batched {
                if group == LATENT {
                    step.push_latent_batch(
                        group,
                        layer,
                        latent_input(shape.1, tokens, block, layer),
                    );
                } else {
                    let (keys, values) = kv_input(shape, tokens, block, layer);
                    step.push_kv_batch(group, layer, keys, values);
                }
                continue;
            }
            for (row, row_tokens) in tokens.iter().enumerate() {
                if row_tokens.is_empty() {
                    continue;
                }
                let single = [row_tokens.clone()];
                if group == LATENT {
                    let latent = latent_input(shape.1, &single, row_tokens.len(), layer);
                    step.push_latent_row(group, layer, row, latent);
                } else {
                    let (keys, values) = kv_input(shape, &single, row_tokens.len(), layer);
                    step.push_kv_row(group, layer, row, keys, values);
                }
            }
        }
    }

    /// Records the recurrent group's states into a step.
    fn push_recurrent(
        &self,
        step: &mut TensorStepWrites<TestBackend>,
        tokens: &[Vec<f32>],
        batched: bool,
    ) {
        if !self.setup.recurrent {
            return;
        }
        let single_tokens = tokens.iter().all(|row| row.len() <= 1);
        if batched && (self.oracle.proposals.is_none() || single_tokens) {
            let last: Vec<f32> = tokens
                .iter()
                .map(|row| row.last().copied().unwrap_or(-1.0))
                .collect();
            let (matrix, convolution) = batch_recurrent_state(&last);
            step.push_recurrent_batch(0, 0, matrix, convolution);
        } else {
            for (row, row_tokens) in tokens.iter().enumerate() {
                if !row_tokens.is_empty() {
                    let (matrices, convolutions) = recurrent_snapshots(row_tokens);
                    step.push_recurrent_row_snapshots(0, 0, row, matrices, convolutions);
                }
            }
        }
    }

    /// Adds published tokens to the committed history or the active proposals.
    fn record(&mut self, tokens: Vec<Vec<f32>>) {
        let target = match &mut self.oracle.proposals {
            Some(proposals) => proposals,
            None => &mut self.oracle.committed,
        };
        for (history, new) in target.iter_mut().zip(tokens) {
            history.extend(new);
        }
    }

    /// Publishes a step through `TensorStepWrites`; rejection must change nothing.
    fn step(&mut self, counts: &[usize], batched: bool) -> bool {
        let rows = self.setup.rows;
        let Ok(mut step) = self.state.step(counts) else {
            assert_ne!(counts.len(), rows, "step rejected valid counts {counts:?}");
            return false;
        };
        let valid = self.step_fits(counts);
        let tokens = self.fresh_tokens(counts);
        for group in [FULL, RING, MIRRORED, LATENT] {
            self.push_group(&mut step, group, &tokens, batched);
        }
        self.push_recurrent(&mut step, &tokens, batched);
        let result = self.state.publish_step(step);
        assert_eq!(result.is_ok(), valid, "step {counts:?}: {result:?}");
        if valid {
            self.record(tokens);
        }
        valid
    }

    /// Writes the full-storage groups ahead, checks reads through them, then finishes the step.
    ///
    /// Outside a transaction both full groups write ahead; inside one only
    /// the in-place latent group may (the separate-staging full group is
    /// pushed instead). Before finishing, every layer written ahead is read
    /// through the step and must equal the working history plus the step's
    /// tokens.
    fn write_ahead(&mut self, counts: &[usize], finish: Finish) -> bool {
        let rows = self.setup.rows;
        let Ok(mut step) = self.state.step(counts) else {
            assert_ne!(counts.len(), rows, "step rejected valid counts {counts:?}");
            return false;
        };
        let valid = self.step_fits(counts);
        let tokens = self.fresh_tokens(counts);
        let block = counts.iter().copied().max().unwrap_or(0).max(1);
        let ahead: &[usize] = if self.oracle.proposals.is_some() {
            &[LATENT]
        } else {
            &[FULL, LATENT]
        };
        for &group in ahead {
            let plan = valid.then(|| {
                self.state
                    .kv_read_plan_through(&step, group)
                    .expect("plan through")
            });
            let shape = self.setup.plane_shape(group);
            for layer in 0..self.setup.layers {
                let result = if group == LATENT {
                    let latent = latent_input(shape.1, &tokens, block, layer);
                    self.state
                        .write_latent_ahead(&mut step, group, layer, latent)
                } else {
                    let (keys, values) = kv_input(shape, &tokens, block, layer);
                    self.state
                        .write_kv_ahead(&mut step, group, layer, keys, values)
                };
                let Some(plan) = &plan else {
                    // The first write-ahead of an oversized step fails before writing.
                    assert!(result.is_err(), "oversized write-ahead {counts:?}");
                    return false;
                };
                result.expect("valid write-ahead");
                let read = self
                    .state
                    .read_kv_through(&step, group, layer, plan)
                    .expect("read through");
                for (row, new) in tokens.iter().enumerate() {
                    let mut expected = self.oracle.working(row);
                    expected.extend(new);
                    let label = format!("through group {group} layer {layer} row {row}");
                    check_batched_row(&read, row, &expected, 0, shape, layer, group, &label);
                }
            }
        }
        for group in [FULL, RING, MIRRORED] {
            if !ahead.contains(&group) {
                self.push_group(&mut step, group, &tokens, true);
            }
        }
        self.push_recurrent(&mut step, &tokens, true);
        match finish {
            Finish::Publish => {
                self.state.publish_step(step).expect("valid step");
                self.record(tokens);
                true
            }
            Finish::Abandon => false,
            Finish::Interfere(choice) => {
                self.interfere(choice);
                assert!(
                    matches!(
                        self.state.publish_step(step),
                        Err(TensorCacheError::StaleStep { .. })
                    ),
                    "a step outlived interference {choice}"
                );
                true
            }
        }
    }

    /// Runs a successful lifecycle operation chosen by `choice`.
    fn interfere(&mut self, choice: u8) {
        let transaction = self.oracle.proposals.is_some();
        match (choice % 3, transaction) {
            (0, false) => assert!(self.begin()),
            (1, false) => assert!(self.reset()),
            (_, true) if choice % 3 != 2 => assert!(self.rollback()),
            _ => {
                let empty = self.state.step(&vec![0; self.setup.rows]).expect("step");
                self.state.publish_step(empty).expect("empty step");
            }
        }
    }

    /// Begins a transaction when none is active.
    fn begin(&mut self) -> bool {
        let result = self.state.begin_transaction();
        if self.oracle.proposals.is_some() {
            assert!(matches!(
                result,
                Err(TensorCacheError::TransactionAlreadyActive)
            ));
            return false;
        }
        self.mark = Some(result.expect("begin"));
        self.oracle.proposals = Some(vec![Vec::new(); self.setup.rows]);
        true
    }

    /// Commits accepted prefixes when they fit inside every row's proposal.
    fn commit(&mut self, accepted: &[usize]) -> bool {
        let Some(mark) = self.mark.clone() else {
            return false;
        };
        let proposals = self.oracle.proposals.as_ref().expect("active transaction");
        let valid = accepted.len() == self.setup.rows
            && accepted
                .iter()
                .zip(proposals)
                .all(|(&count, proposal)| count <= proposal.len());
        let result = self.state.commit_prefix(mark, accepted);
        assert_eq!(result.is_ok(), valid, "commit {accepted:?}: {result:?}");
        if valid {
            let proposals = self.oracle.proposals.take().expect("active transaction");
            for ((history, proposal), &count) in self
                .oracle
                .committed
                .iter_mut()
                .zip(proposals)
                .zip(accepted)
            {
                history.extend(&proposal[..count]);
            }
            self.mark = None;
        }
        valid
    }

    /// Rolls the active transaction back.
    fn rollback(&mut self) -> bool {
        let Some(mark) = self.mark.take() else {
            return false;
        };
        let stale = mark.clone();
        self.state.rollback(mark).expect("rollback");
        assert!(matches!(
            self.state.rollback(stale),
            Err(TensorCacheError::TransactionNotActive)
        ));
        self.oracle.proposals = None;
        true
    }

    /// Reorders rows outside a transaction.
    fn reorder(&mut self, indices: &[usize]) -> bool {
        let valid = self.oracle.proposals.is_none()
            && indices.len() == self.setup.rows
            && indices.iter().all(|&index| index < self.setup.rows);
        assert_eq!(
            self.state.reorder(indices).is_ok(),
            valid,
            "reorder {indices:?}"
        );
        if valid {
            self.oracle.committed = indices
                .iter()
                .map(|&index| self.oracle.committed[index].clone())
                .collect();
        }
        valid
    }

    /// Rewinds rows when every group can reconstruct the shorter prefix.
    fn truncate(&mut self, lengths: &[usize]) -> bool {
        let rows = self.setup.rows;
        let valid = self.oracle.proposals.is_none()
            && lengths.len() == rows
            && (0..rows).all(|row| {
                let from = self.oracle.committed[row].len();
                let to = lengths[row];
                let ring_safe = |window: usize| to == from || to == 0 || from <= window;
                to <= from
                    && ring_safe(self.setup.ring_window)
                    && ring_safe(self.setup.mirrored_window)
                    && (!self.setup.recurrent || to == from || to == 0)
            });
        assert_eq!(
            self.state.truncate(lengths).is_ok(),
            valid,
            "truncate {lengths:?}"
        );
        if valid {
            for (history, &length) in self.oracle.committed.iter_mut().zip(lengths) {
                history.truncate(length);
            }
        }
        valid
    }

    /// Clears every row outside a transaction.
    fn reset(&mut self) -> bool {
        let valid = self.oracle.proposals.is_none();
        assert_eq!(self.state.reset().is_ok(), valid);
        if valid {
            self.oracle.committed.iter_mut().for_each(Vec::clear);
        }
        valid
    }

    /// Compares every observable length and cache read with the oracle.
    fn check(&self) {
        let lengths = self.oracle.lengths();
        assert_eq!(self.state.lengths(), lengths.as_slice());
        let working: Vec<usize> = (0..self.setup.rows)
            .map(|row| self.oracle.working(row).len())
            .collect();
        assert_eq!(self.state.working_lengths(), working.as_slice());
        assert_eq!(
            self.state.has_active_transaction(),
            self.oracle.proposals.is_some()
        );
        for group in [FULL, RING, MIRRORED, LATENT] {
            self.check_group(group);
        }
        if self.setup.recurrent {
            for row in 0..self.setup.rows {
                let read = self
                    .state
                    .read_recurrent(StepBank::Main, 0, 0, row)
                    .expect("recurrent read");
                let expected = self.oracle.working(row).last().copied().unwrap_or(0.0);
                assert_eq!(host(read.matrix().clone()), [expected], "row {row} matrix");
                assert_eq!(
                    host(read.convolution().clone()),
                    [expected; CONVOLUTION_WIDTH],
                    "row {row} convolution"
                );
            }
        }
    }

    /// Compares one group's per-row and batched reads of every layer with the oracle.
    fn check_group(&self, group: usize) {
        let shape = self.setup.plane_shape(group);
        let plan: KvBatchReadPlan<TestBackend> = self
            .state
            .kv_read_plan(StepBank::Main, group)
            .expect("batched plan");
        for layer in 0..self.setup.layers {
            let batched = self
                .state
                .main()
                .native_kv(group)
                .expect("group")
                .read_batch(layer, &plan)
                .expect("batched read");
            for row in 0..self.setup.rows {
                let tokens = self.oracle.working(row);
                let keep = self.setup.window(group).unwrap_or(tokens.len());
                let first = tokens.len().saturating_sub(keep);
                let visible = &tokens[first..];
                let label = format!("group {group} layer {layer} row {row}");
                assert_eq!(
                    plan.retained_positions()[row],
                    first..tokens.len(),
                    "{label}"
                );
                check_batched_row(&batched, row, visible, first, shape, layer, group, &label);

                let read = self
                    .state
                    .read_kv(StepBank::Main, group, layer, row)
                    .expect("checked read");
                assert_eq!(read.retained_positions(), first..tokens.len(), "{label}");
                let single = [visible.to_vec()];
                let (keys, values) = read
                    .materialize()
                    .map_or((Vec::new(), Vec::new()), |(keys, values)| {
                        (host(keys), host(values))
                    });
                let expected = plane_data(&single, visible.len(), shape, layer, 1.0);
                assert_eq!(keys, expected, "{label} keys");
                if group != LATENT {
                    let negated = plane_data(&single, visible.len(), shape, layer, -1.0);
                    assert_eq!(values, negated, "{label} values");
                }
            }
        }
    }
}

/// Checks one row of a batched slot-order read against the oracle's visible tokens.
///
/// Valid slots, sorted by key position, must be positions `first..` holding
/// exactly the tokens' codes; masked values must equal the raw values on
/// valid slots and be zero on invalid ones.
#[allow(clippy::too_many_arguments)]
fn check_batched_row(
    read: &KvBatchRead<TestBackend>,
    row: usize,
    visible: &[f32],
    first: usize,
    shape: (usize, usize),
    layer: usize,
    group: usize,
    label: &str,
) {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for segment in read.segments() {
        let [rows, heads, slots, channels] = segment.keys().dims();
        assert_eq!((heads, channels), shape, "{label}");
        let positions: Vec<i32> = host_int(segment.key_positions().clone());
        let valid = segment.key_valid().map_or_else(
            || vec![true; rows * slots],
            |valid| host_bool(valid.clone()),
        );
        let key_data = host(segment.keys().clone());
        let value_data = host(segment.values().clone());
        let masked = host(segment.masked_values());
        for slot in 0..slots {
            let entries = |data: &[f32]| -> Vec<f32> {
                (0..heads)
                    .flat_map(|head| {
                        (0..channels).map(move |channel| {
                            ((row * heads + head) * slots + slot) * channels + channel
                        })
                    })
                    .map(|index| data[index])
                    .collect()
            };
            let index = row * slots + slot;
            if valid[index] {
                keys.push((positions[index], entries(&key_data)));
                values.push((positions[index], entries(&value_data)));
                assert_eq!(entries(&masked), entries(&value_data), "{label} masked");
            } else {
                assert!(
                    entries(&masked).iter().all(|&value| value == 0.0),
                    "{label}"
                );
            }
        }
    }
    keys.sort_by_key(|(position, _)| *position);
    values.sort_by_key(|(position, _)| *position);
    let expected = |sign: f32| -> Vec<(i32, Vec<f32>)> {
        visible
            .iter()
            .enumerate()
            .map(|(offset, &value)| {
                let position = i32::try_from(first + offset).expect("small position");
                let entries = (0..shape.0)
                    .flat_map(|head| {
                        (0..shape.1).map(move |channel| sign * code(value, layer, head, channel))
                    })
                    .collect();
                (position, entries)
            })
            .collect()
    };
    assert_eq!(keys, expected(1.0), "{label} batched keys");
    let value_sign = if group == LATENT { 1.0 } else { -1.0 };
    assert_eq!(values, expected(value_sign), "{label} batched values");
}

/// Builds batched terminal recurrent states "last token value"; inactive rows hold `-1`.
fn batch_recurrent_state(last: &[f32]) -> (Tensor<TestBackend, 4>, Tensor<TestBackend, 3>) {
    let rows = last.len();
    let convolution: Vec<f32> = last
        .iter()
        .flat_map(|&value| [value; CONVOLUTION_WIDTH])
        .collect();
    (
        Tensor::from_data(
            TensorData::new(last.to_vec(), [rows, 1, 1, 1]),
            &Default::default(),
        ),
        Tensor::from_data(
            TensorData::new(convolution, [rows, 1, CONVOLUTION_WIDTH]),
            &Default::default(),
        ),
    )
}

/// Builds one post-token snapshot per proposal token.
fn recurrent_snapshots(tokens: &[f32]) -> (Tensor<TestBackend, 5>, Tensor<TestBackend, 4>) {
    let count = tokens.len();
    let matrices = Tensor::from_data(
        TensorData::new(tokens.to_vec(), [1, count, 1, 1, 1]),
        &Default::default(),
    );
    let convolutions = Tensor::from_data(
        TensorData::new(
            tokens
                .iter()
                .flat_map(|&value| [value; CONVOLUTION_WIDTH])
                .collect::<Vec<_>>(),
            [1, count, 1, CONVOLUTION_WIDTH],
        ),
        &Default::default(),
    );
    (matrices, convolutions)
}

/// Transfers a float tensor to the host for exact comparison.
fn host<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
    tensor.into_data().to_vec().expect("F32 tensor")
}

/// Transfers an Int tensor to the host.
fn host_int(tensor: Tensor<TestBackend, 2, Int>) -> Vec<i32> {
    tensor.into_data().to_vec().expect("I32 tensor")
}

/// Transfers a Bool tensor to the host.
fn host_bool(tensor: Tensor<TestBackend, 2, Bool>) -> Vec<bool> {
    tensor.into_data().to_vec().expect("Bool tensor")
}

/// Generates a per-row vector that is occasionally the wrong length.
fn row_vector(rows: usize, maximum: usize) -> impl Strategy<Value = Vec<usize>> {
    prop_oneof![
        9 => prop::collection::vec(0..=maximum, rows),
        1 => prop::collection::vec(0..=maximum, 0..=rows + 1),
    ]
}

/// Generates how a write-ahead step ends.
fn finish() -> impl Strategy<Value = Finish> {
    prop_oneof![
        3 => Just(Finish::Publish),
        1 => Just(Finish::Abandon),
        2 => any::<u8>().prop_map(Finish::Interfere),
    ]
}

/// Generates operations that straddle every validity boundary.
fn operation(setup: Setup) -> impl Strategy<Value = Operation> {
    let rows = setup.rows;
    let tokens = setup.capacity.min(4);
    prop_oneof![
        3 => row_vector(rows, tokens).prop_map(Operation::Append),
        3 => (row_vector(rows, tokens), any::<bool>())
            .prop_map(|(counts, batched)| Operation::Step(counts, batched)),
        3 => (row_vector(rows, tokens), finish())
            .prop_map(|(counts, finish)| Operation::WriteAhead(counts, finish)),
        2 => Just(Operation::Begin),
        2 => row_vector(rows, setup.staging + 1).prop_map(Operation::Commit),
        1 => Just(Operation::Rollback),
        1 => row_vector(rows, rows).prop_map(Operation::Reorder),
        2 => row_vector(rows, setup.capacity).prop_map(Operation::Truncate),
        1 => Just(Operation::Reset),
    ]
}

/// Generates the geometry of one case.
fn setup() -> impl Strategy<Value = Setup> {
    (
        1_usize..=3,
        2_usize..=10,
        1_usize..=4,
        any::<bool>(),
        1_usize..=2,
        1_usize..=2,
    )
        .prop_flat_map(|(rows, capacity, staging, recurrent, layers, width)| {
            (1..=capacity, 1..=capacity).prop_map(move |(ring_window, mirrored_window)| Setup {
                rows,
                capacity,
                ring_window,
                mirrored_window,
                staging,
                recurrent,
                layers,
                width,
            })
        })
}

/// Returns the case count: `PROPTEST_CASES` when set, otherwise `default`.
///
/// `ProptestConfig::with_cases` would silently override the environment, so
/// stress runs (`PROPTEST_CASES=3000`) would still execute the default count.
fn cases_or(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|cases| cases.parse().ok())
        .unwrap_or(default)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: cases_or(96), ..ProptestConfig::default() })]

    /// Every lifecycle sequence keeps lengths and cache contents equal to the oracle.
    #[test]
    fn tensor_decode_state_matches_naive_history_model(
        (setup, operations) in setup().prop_flat_map(|setup| {
            (Just(setup), prop::collection::vec(operation(setup), 0..40))
        }),
    ) {
        let mut harness = Harness::new(setup);
        harness.check();
        for operation in &operations {
            let before = harness.oracle.clone();
            if !harness.apply(operation) {
                prop_assert_eq!(&harness.oracle, &before, "rejected {:?} changed the oracle", operation);
            }
            harness.check();
        }
    }
}

/// A fixed multi-layer, multi-head sequence covering every write-ahead ending.
///
/// The proptest samples these paths randomly; this case pins them so a
/// regression is reported deterministically.
#[test]
fn write_ahead_endings_match_the_model_in_a_fixed_sequence() {
    let setup = Setup {
        rows: 2,
        capacity: 10,
        ring_window: 3,
        mirrored_window: 2,
        staging: 3,
        recurrent: true,
        layers: 2,
        width: 2,
    };
    let mut harness = Harness::new(setup);
    let operations = [
        Operation::WriteAhead(vec![2, 1], Finish::Publish),
        Operation::WriteAhead(vec![1, 0], Finish::Abandon),
        Operation::Step(vec![1, 1], true),
        Operation::WriteAhead(vec![1, 1], Finish::Interfere(0)),
        Operation::WriteAhead(vec![1, 1], Finish::Publish),
        Operation::WriteAhead(vec![1, 0], Finish::Abandon),
        Operation::Commit(vec![1, 0]),
        Operation::WriteAhead(vec![1, 1], Finish::Interfere(1)),
        Operation::WriteAhead(vec![2, 2], Finish::Publish),
        Operation::Begin,
        Operation::WriteAhead(vec![1, 1], Finish::Interfere(0)),
        Operation::WriteAhead(vec![9, 9], Finish::Publish),
    ];
    for operation in &operations {
        let before = harness.oracle.clone();
        if !harness.apply(operation) {
            assert_eq!(harness.oracle, before, "rejected {operation:?}");
        }
        harness.check();
    }
    assert_eq!(harness.state.lengths(), &[2, 2]);
}
