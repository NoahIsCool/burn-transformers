//! Differential cache state machines with independent oracles.
//!
//! [`exercise`] drives the metadata-only [`SequenceLengths`] transaction
//! machine and the ring planners against integer oracles. [`exercise_tensor_state`]
//! drives the tensor-backed [`TensorDecodeState`] that models actually use:
//! full, ring, mirrored, and latent groups with one or two layers, heads, and
//! channels plus an optional recurrent group, written through atomic step
//! write sets (batched or per row) and write-ahead (published, abandoned, or
//! invalidated by an interfering lifecycle operation), speculative
//! transactions with ragged acceptance, reorder, truncate, and reset. An
//! optional multi-token-prediction (MTP) bank with its own cursor receives
//! MTP-bank steps, joins every transaction with an accepted prefix drawn
//! independently of the main prefix (from its own selector bits), follows
//! reorder and reset, and rewinds through `truncate_mtp`. The oracle keeps each
//! row's token history (one per bank) as a `Vec<f32>` of unique values and
//! compares every visible key and value of every layer after every operation,
//! through both the checked per-row read and the batched slot-order read;
//! rejected operations must change nothing.

use std::num::NonZeroUsize;

use burn::backend::Flex;
use burn::tensor::{DType, Tensor, TensorData};
use transformers::cache::{
    CacheConfig, CacheMark, KvBatchRead, KvCacheRead, KvCacheStorage, NativeKvCacheSpec,
    RecurrentStateSpec, SequenceLengths, StepBank, TensorCacheError, TensorCacheMark,
    TensorDecodeState, TensorStepWrites, TransactionalCache, plan_ring_write,
    plan_staged_ring_transaction,
};

/// Host backend used by the tensor-state harness.
type Backend = Flex<f32, i32>;

/// Checks physical ring writes against `slot = absolute_position mod capacity`.
fn check_ring(start: usize, count: usize, capacity: usize) {
    let capacity = NonZeroUsize::new(capacity).unwrap();
    let plan = plan_ring_write(start, count, capacity);
    if start.checked_add(count).is_none() {
        assert!(plan.is_err());
        return;
    }
    let plan = plan.unwrap();
    assert_eq!(plan.appended_positions(), start..start + count);
    let retained = count.saturating_sub(capacity.get())..count;
    assert_eq!(plan.retained_input(), retained.clone());
    let observed = plan
        .segments()
        .iter()
        .flat_map(|segment| segment.input().zip(segment.physical()))
        .collect::<Vec<_>>();
    let expected = retained
        .map(|offset| (offset, (start + offset) % capacity.get()))
        .collect::<Vec<_>>();
    assert_eq!(observed, expected);
    let staged = plan_staged_ring_transaction(start, count).unwrap();
    assert_eq!(staged.commit_prefix(count, capacity).unwrap(), plan);
    if count < usize::MAX {
        assert!(staged.commit_prefix(count + 1, capacity).is_err());
    }
}

/// Runs at most 128 mutations; rejection must preserve both lengths and transaction state.
pub fn exercise(input: &[u8]) {
    let mut bytes = crate::Bytes::new(input);
    let batch = usize::from(bytes.byte() % 4) + 1;
    let capacity = usize::from(bytes.byte() % 32) + 1;
    let mut state = SequenceLengths::new(CacheConfig::new(batch, capacity).unwrap());
    let mut expected = vec![0_usize; batch];
    let mut active: Option<(CacheMark, Vec<usize>)> = None;
    let mut stale: Option<CacheMark> = None;
    for _ in 0..input.len().min(512).div_ceil(4) {
        let operation = bytes.byte() % 10;
        let sequence = usize::from(bytes.byte() % 6);
        let value = bytes.size();
        let second = bytes.size();
        let previous = expected.clone();
        let was_active = active.is_some();
        let success = match operation {
            0 => {
                let next = expected
                    .get(sequence)
                    .and_then(|old| old.checked_add(value))
                    .filter(|&next| next <= capacity);
                let result = state.advance(sequence, value);
                assert_eq!(result.is_ok(), next.is_some());
                if let Some(next) = next {
                    expected[sequence] = next;
                }
                result.is_ok()
            }
            1 => {
                let counts = (0..batch)
                    .map(|index| if index == 0 { value } else { second })
                    .collect::<Vec<_>>();
                let next = expected
                    .iter()
                    .zip(&counts)
                    .map(|(&old, &count)| old.checked_add(count).filter(|&next| next <= capacity))
                    .collect::<Option<Vec<_>>>();
                let result = state.advance_all(&counts);
                assert_eq!(result.is_ok(), next.is_some());
                if let Some(next) = next {
                    expected = next;
                }
                result.is_ok()
            }
            2 => {
                let valid =
                    active.is_none() && expected.get(sequence).is_some_and(|&old| value <= old);
                let result = state.truncate(sequence, value);
                assert_eq!(result.is_ok(), valid);
                if valid {
                    expected[sequence] = value;
                }
                valid
            }
            3 => {
                let lengths = vec![value; batch];
                let valid = active.is_none() && expected.iter().all(|&old| value <= old);
                assert_eq!(state.truncate_all(&lengths).is_ok(), valid);
                if valid {
                    expected = lengths;
                }
                valid
            }
            4 => {
                let indices = (0..batch)
                    .map(|index| {
                        if index == 0 {
                            sequence
                        } else {
                            second % (batch + 1)
                        }
                    })
                    .collect::<Vec<_>>();
                let valid = active.is_none() && indices.iter().all(|&index| index < batch);
                assert_eq!(state.reorder(&indices).is_ok(), valid);
                if valid {
                    expected = indices.iter().map(|&index| previous[index]).collect();
                }
                valid
            }
            5 => {
                let valid = active.is_none();
                assert_eq!(state.reset().is_ok(), valid);
                if valid {
                    expected.fill(0);
                }
                valid
            }
            6 => {
                let result = state.mark();
                assert_eq!(result.is_ok(), active.is_none());
                if let Ok(mark) = result {
                    active = Some((mark, expected.clone()));
                    true
                } else {
                    false
                }
            }
            7 => {
                if let Some((mark, baseline)) = &active {
                    let valid = expected
                        .iter()
                        .zip(baseline)
                        .all(|(&now, &old)| value <= now - old);
                    assert_eq!(state.commit_prefix(mark.clone(), value).is_ok(), valid);
                    if valid {
                        expected = baseline.iter().map(|&old| old + value).collect();
                        stale = Some(mark.clone());
                        active = None;
                    }
                    valid
                } else {
                    false
                }
            }
            8 => {
                if let Some((mark, baseline)) = active.take() {
                    state.rollback(mark.clone()).unwrap();
                    expected = baseline;
                    stale = Some(mark);
                    true
                } else {
                    false
                }
            }
            _ => {
                if let Some(mark) = &stale {
                    assert!(state.rollback(mark.clone()).is_err());
                }
                false
            }
        };
        // `expected` follows the working cursor; committed lengths stay at the
        // transaction baseline until a commit.
        assert_eq!(state.working_lengths(), expected);
        let committed = active.as_ref().map_or(&expected, |(_, baseline)| baseline);
        assert_eq!(state.lengths(), committed.as_slice());
        assert_eq!(state.has_active_transaction(), active.is_some());
        assert!(expected.iter().all(|&length| length <= capacity));
        if !success {
            assert_eq!(expected, previous);
            assert_eq!(state.has_active_transaction(), was_active);
        }
        check_ring(value, second, capacity);
    }
}

/// Index of the full-storage KV group (separate staging).
const FULL_GROUP: usize = 0;
/// Indices of the harness's KV groups, in creation order.
const KV_GROUPS: [usize; 3] = [FULL_GROUP, 1, 2];
/// Index of the latent group (full storage, in-place staging).
const LATENT_GROUP: usize = 3;
/// Every KV-like group: the key/value groups then the latent group.
const ALL_GROUPS: [usize; 4] = [FULL_GROUP, 1, 2, LATENT_GROUP];
/// Convolution width of the recurrent group.
const CONVOLUTION_WIDTH: usize = 2;
/// Value of padding columns in batched inputs; never a valid cache entry.
const SENTINEL: f32 = -7.0;

/// Geometry decoded from the first input bytes.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    /// Batch rows (1..=3).
    rows: usize,
    /// Logical capacity (2..=12).
    capacity: usize,
    /// Windows of the standard and mirrored ring groups.
    windows: [usize; 2],
    /// Staging capacity of every group (1..=3).
    staging: usize,
    /// Whether a recurrent group participates.
    recurrent: bool,
    /// Whether an MTP bank (one scalar full-storage KV group) participates.
    mtp: bool,
    /// Layers of every KV and latent group (1..=2).
    layers: usize,
    /// KV heads and channels per head (1..=2); the latent width is one more.
    width: usize,
}

impl Geometry {
    /// Returns `(heads, channels)` of a group's planes.
    const fn plane_shape(&self, group: usize) -> (usize, usize) {
        if group == LATENT_GROUP {
            (1, self.width + 1)
        } else {
            (self.width, self.width)
        }
    }

    /// Returns how many of a row's newest tokens a group keeps visible.
    const fn keep(&self, group: usize, length: usize) -> usize {
        match group {
            1 => self.windows[0],
            2 => self.windows[1],
            _ => length,
        }
    }
}

/// How a write-ahead step ends.
#[derive(Debug, Clone, Copy)]
enum Finish {
    /// Complete and publish the step.
    Publish,
    /// Drop the step; its tokens stay in slots beyond the lengths.
    Abandon,
    /// Run a lifecycle operation first; publication must be refused as stale.
    Interfere(u8),
}

/// Reference model: committed histories and the active proposal suffixes.
#[derive(Debug, Clone, PartialEq)]
struct History {
    /// Committed token values per row.
    committed: Vec<Vec<f32>>,
    /// Proposed token values per row during a transaction.
    proposals: Option<Vec<Vec<f32>>>,
}

impl History {
    /// Returns row `row`'s committed tokens followed by its proposals.
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
/// Token values are positive integers, so `16 * value + offset` with
/// `offset = 8 * layer + 4 * head + channel < 16` is a distinct, exactly
/// representable F32 integer for every layer, head, and channel.
fn code(value: f32, layer: usize, head: usize, channel: usize) -> f32 {
    let offset = u8::try_from(8 * layer + 4 * head + channel).expect("offset below 16");
    value * 16.0 + f32::from(offset)
}

/// Flat `[rows, heads, block, channels]` data of right-padded per-row tokens.
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

/// The tensor state, its oracle, and the active transaction mark.
struct TensorHarness {
    /// Decoded geometry.
    geometry: Geometry,
    /// System under test.
    state: TensorDecodeState<Backend>,
    /// Reference model of the main bank.
    history: History,
    /// Reference model of the MTP bank, when enabled.
    mtp: Option<History>,
    /// Mark of the active transaction.
    mark: Option<TensorCacheMark>,
    /// Last token value handed out; values start at 2.
    next_value: f32,
}

/// Runs at most 64 tensor-state operations decoded from `input`.
///
/// Byte layout: an eight-byte header (rows, capacity, two windows, staging, a
/// bank byte whose bit 0 clear adds the recurrent group and bit 1 the MTP
/// bank, layers, width), then per operation `[operation % 9, one count per
/// row, selector]`. The selector's bit 0 picks batched or per-row writes (or,
/// for a commit, whether an MTP prefix is passed); for a commit its bits
/// 1..=7 encode the MTP prefix (see [`TensorHarness::mtp_prefix`]); for
/// write-ahead and operation 8 its low two bits pick the ending or MTP action.
///
/// # Panics
///
/// Panics when the tensor state disagrees with the oracle, which is a bug.
pub fn exercise_tensor_state(input: &[u8]) {
    let mut bytes = crate::Bytes::new(input);
    let rows = usize::from(bytes.byte() % 3) + 1;
    let capacity = usize::from(bytes.byte() % 11) + 2;
    let windows = [
        usize::from(bytes.byte()) % capacity + 1,
        usize::from(bytes.byte()) % capacity + 1,
    ];
    let staging = usize::from(bytes.byte() % 3) + 1;
    // Bit 0 (clear) selects the recurrent group and bit 1 the MTP bank, so
    // seeds written before the MTP bank existed keep their meaning.
    let banks = bytes.byte();
    let geometry = Geometry {
        rows,
        capacity,
        windows,
        staging,
        recurrent: banks.is_multiple_of(2),
        mtp: banks & 2 != 0,
        layers: usize::from(bytes.byte() % 2) + 1,
        width: usize::from(bytes.byte() % 2) + 1,
    };
    let mut harness = TensorHarness::new(geometry);
    harness.check();
    for _ in 0..input.len().min(512).div_ceil(6).min(64) {
        let operation = bytes.byte() % 9;
        let counts: Vec<usize> = (0..rows).map(|_| usize::from(bytes.byte() % 4)).collect();
        let selector = bytes.byte();
        let before = (harness.history.clone(), harness.mtp.clone());
        let accepted = harness.apply(operation, &counts, selector);
        if !accepted {
            assert_eq!(
                (harness.history.clone(), harness.mtp.clone()),
                before,
                "rejected operation {operation} changed the oracle"
            );
        }
        harness.check();
    }
}

impl TensorHarness {
    /// Allocates full, ring, mirrored, and latent groups (+ recurrent).
    fn new(geometry: Geometry) -> Self {
        let config = CacheConfig::new(geometry.rows, geometry.capacity).expect("bounded config");
        let mut state = TensorDecodeState::with_config(config, &Default::default());
        let window = |width: usize| NonZeroUsize::new(width).expect("positive window");
        let specs = [
            (KvCacheStorage::Full, None),
            (
                KvCacheStorage::SlidingRing,
                Some(window(geometry.windows[0])),
            ),
            (
                KvCacheStorage::MirroredSlidingRing,
                Some(window(geometry.windows[1])),
            ),
        ];
        for (storage, group_window) in specs {
            let mut spec = NativeKvCacheSpec::new(
                geometry.layers,
                geometry.width,
                geometry.width,
                storage,
                DType::F32,
            )
            .expect("small geometry")
            .with_staging_capacity(geometry.staging);
            if let Some(group_window) = group_window {
                spec = spec.with_sliding_window(group_window);
            }
            state.main_mut().add_native_kv(spec).expect("KV group");
        }
        let latent = NativeKvCacheSpec::latent(
            geometry.layers,
            geometry.plane_shape(LATENT_GROUP).1,
            KvCacheStorage::Full,
            DType::F32,
        )
        .expect("small latent")
        .with_staging_capacity(geometry.staging)
        .with_in_place_staging();
        state
            .main_mut()
            .add_native_kv(latent)
            .expect("latent group");
        if geometry.recurrent {
            let spec = RecurrentStateSpec::new(1, 1, 1, 1, 1, CONVOLUTION_WIDTH, DType::F32)
                .expect("scalar recurrent geometry")
                .with_staging_capacity(geometry.staging);
            state
                .main_mut()
                .add_recurrent(spec)
                .expect("recurrent group");
        }
        let empty = History {
            committed: vec![Vec::new(); geometry.rows],
            proposals: None,
        };
        if geometry.mtp {
            let spec = NativeKvCacheSpec::new(1, 1, 1, KvCacheStorage::Full, DType::F32)
                .expect("scalar MTP geometry")
                .with_staging_capacity(geometry.staging);
            state
                .enable_mtp()
                .expect("MTP bank")
                .add_native_kv(spec)
                .expect("MTP group");
        }
        Self {
            geometry,
            state,
            mtp: geometry.mtp.then(|| empty.clone()),
            history: empty,
            mark: None,
            next_value: 1.0,
        }
    }

    /// Applies one decoded operation; returns whether it was accepted.
    fn apply(&mut self, operation: u8, counts: &[usize], selector: u8) -> bool {
        let rows = self.geometry.rows;
        let flag = selector.is_multiple_of(2);
        match operation {
            0 => self.publish(counts, flag),
            1 => self.begin(),
            2 => {
                // Selector 254 also passes a prefix without an MTP bank, which
                // must be refused; otherwise only an MTP bank gets one.
                let with_prefix = flag && (self.mtp.is_some() || selector == 254);
                let mtp_accepted = with_prefix.then(|| self.mtp_prefix(selector));
                self.commit(counts, mtp_accepted.as_deref())
            }
            3 => self.rollback(),
            4 => {
                let indices: Vec<usize> = counts.iter().map(|&index| index % (rows + 1)).collect();
                self.reorder(&indices)
            }
            5 => {
                let requested: Vec<usize> = (0..rows)
                    .map(|row| {
                        if flag {
                            0
                        } else {
                            self.history.committed[row]
                                .len()
                                .saturating_sub(counts[row])
                        }
                    })
                    .collect();
                self.truncate(&requested)
            }
            6 => self.reset(),
            7 => {
                let finish = match selector % 4 {
                    0 | 1 => Finish::Publish,
                    2 => Finish::Abandon,
                    _ => Finish::Interfere(selector / 4),
                };
                self.write_ahead(counts, finish)
            }
            _ => {
                if selector % 4 == 3 {
                    self.truncate_mtp(counts)
                } else {
                    self.publish_mtp(counts, flag)
                }
            }
        }
    }

    /// Hands out fresh token values for every row.
    fn fresh_tokens(&mut self, counts: &[usize]) -> Vec<Vec<f32>> {
        counts
            .iter()
            .map(|&count| {
                (0..count)
                    .map(|_| {
                        self.next_value += 1.0;
                        self.next_value
                    })
                    .collect()
            })
            .collect()
    }

    /// Returns whether a step fits capacity and, in a transaction, staging.
    fn fits(&self, counts: &[usize]) -> bool {
        (0..self.geometry.rows).all(|row| {
            let staged = self
                .history
                .proposals
                .as_ref()
                .map_or(0, |proposals| proposals[row].len());
            self.history.working(row).len() + counts[row] <= self.geometry.capacity
                && (self.history.proposals.is_none()
                    || staged + counts[row] <= self.geometry.staging)
        })
    }

    /// Records one group's layers into a step, batched or per row.
    fn push_group(
        &self,
        step: &mut TensorStepWrites<Backend>,
        group: usize,
        tokens: &[Vec<f32>],
        batched: bool,
    ) {
        let shape = self.geometry.plane_shape(group);
        let block = tokens.iter().map(Vec::len).max().unwrap_or(0).max(1);
        for layer in 0..self.geometry.layers {
            if batched {
                if group == LATENT_GROUP {
                    step.push_latent_batch(group, layer, latent(shape.1, tokens, block, layer));
                } else {
                    let (keys, values) = kv(shape, tokens, block, layer);
                    step.push_kv_batch(group, layer, keys, values);
                }
                continue;
            }
            for (row, row_tokens) in tokens.iter().enumerate() {
                if row_tokens.is_empty() {
                    continue;
                }
                let single = [row_tokens.clone()];
                if group == LATENT_GROUP {
                    let input = latent(shape.1, &single, row_tokens.len(), layer);
                    step.push_latent_row(group, layer, row, input);
                } else {
                    let (keys, values) = kv(shape, &single, row_tokens.len(), layer);
                    step.push_kv_row(group, layer, row, keys, values);
                }
            }
        }
    }

    /// Records per-token recurrent snapshots (valid in and out of transactions).
    fn push_recurrent(&self, step: &mut TensorStepWrites<Backend>, tokens: &[Vec<f32>]) {
        if !self.geometry.recurrent {
            return;
        }
        for (row, row_tokens) in tokens.iter().enumerate() {
            let count = row_tokens.len();
            if count == 0 {
                continue;
            }
            let matrices = Tensor::<Backend, 5>::from_data(
                TensorData::new(row_tokens.clone(), [1, count, 1, 1, 1]),
                &Default::default(),
            );
            let convolutions = Tensor::<Backend, 4>::from_data(
                TensorData::new(
                    row_tokens
                        .iter()
                        .flat_map(|&value| [value; CONVOLUTION_WIDTH])
                        .collect::<Vec<_>>(),
                    [1, count, 1, CONVOLUTION_WIDTH],
                ),
                &Default::default(),
            );
            step.push_recurrent_row_snapshots(0, 0, row, matrices, convolutions);
        }
    }

    /// Adds published tokens to the committed history or the active proposals.
    fn record(&mut self, tokens: Vec<Vec<f32>>) {
        let target = match &mut self.history.proposals {
            Some(proposals) => proposals,
            None => &mut self.history.committed,
        };
        for (row, new) in target.iter_mut().zip(tokens) {
            row.extend(new);
        }
    }

    /// Publishes one step of fresh tokens, batched or per row.
    fn publish(&mut self, counts: &[usize], batched: bool) -> bool {
        let valid = self.fits(counts);
        let tokens = self.fresh_tokens(counts);
        let mut step = self.state.step(counts).expect("one count per row");
        for group in ALL_GROUPS {
            self.push_group(&mut step, group, &tokens, batched);
        }
        self.push_recurrent(&mut step, &tokens);
        let result = self.state.publish_step(step);
        assert_eq!(result.is_ok(), valid, "step {counts:?}: {result:?}");
        if valid {
            self.record(tokens);
        }
        valid
    }

    /// Writes the full-storage groups ahead, reads through them, then finishes the step.
    ///
    /// Outside a transaction the full and latent groups write ahead; inside
    /// one only the in-place latent group can.
    fn write_ahead(&mut self, counts: &[usize], finish: Finish) -> bool {
        let valid = self.fits(counts);
        let tokens = self.fresh_tokens(counts);
        let block = counts.iter().copied().max().unwrap_or(0).max(1);
        let mut step = self.state.step(counts).expect("one count per row");
        let ahead: &[usize] = if self.history.proposals.is_some() {
            &[LATENT_GROUP]
        } else {
            &[FULL_GROUP, LATENT_GROUP]
        };
        for &group in ahead {
            let plan = valid.then(|| {
                self.state
                    .kv_read_plan_through(&step, group)
                    .expect("plan through")
            });
            let shape = self.geometry.plane_shape(group);
            for layer in 0..self.geometry.layers {
                let result = if group == LATENT_GROUP {
                    let input = latent(shape.1, &tokens, block, layer);
                    self.state
                        .write_latent_ahead(&mut step, group, layer, input)
                } else {
                    let (keys, values) = kv(shape, &tokens, block, layer);
                    self.state
                        .write_kv_ahead(&mut step, group, layer, keys, values)
                };
                let Some(plan) = &plan else {
                    assert!(result.is_err(), "oversized write-ahead {counts:?}");
                    return false;
                };
                result.expect("valid write-ahead");
                let read = self
                    .state
                    .read_kv_through(&step, group, layer, plan)
                    .expect("read through");
                for (row, new) in tokens.iter().enumerate() {
                    let mut expected = self.history.working(row);
                    expected.extend(new);
                    check_batched_row(&read, row, &expected, 0, shape, layer, group);
                }
            }
        }
        for group in KV_GROUPS {
            if !ahead.contains(&group) {
                self.push_group(&mut step, group, &tokens, true);
            }
        }
        self.push_recurrent(&mut step, &tokens);
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
        let transaction = self.history.proposals.is_some();
        match (choice % 3, transaction) {
            (0, false) => assert!(self.begin()),
            (1, false) => assert!(self.reset()),
            (0 | 1, true) => assert!(self.rollback()),
            _ => {
                let empty = self.state.step(&vec![0; self.geometry.rows]).expect("step");
                self.state.publish_step(empty).expect("empty step");
            }
        }
    }

    /// Begins a transaction unless one is active.
    fn begin(&mut self) -> bool {
        let result = self.state.begin_transaction();
        assert_eq!(result.is_ok(), self.mark.is_none());
        match result {
            Ok(mark) => {
                self.mark = Some(mark);
                self.history.proposals = Some(vec![Vec::new(); self.geometry.rows]);
                if let Some(mtp) = &mut self.mtp {
                    mtp.proposals = Some(vec![Vec::new(); self.geometry.rows]);
                }
                true
            }
            Err(_) => false,
        }
    }

    /// Decodes an MTP accepted prefix independent of the main prefix.
    ///
    /// Bits `1 + 2 * row` and `2 + 2 * row` of `selector` give row `row`'s
    /// count in `0..=3` (rows are at most three, so bits 1..=6 suffice; bit 0
    /// already chose whether a prefix is passed). With bit 7 set, each count
    /// is clamped to the row's staged MTP length so that most such commits
    /// are valid; with it clear, counts beyond the proposals test rejection.
    fn mtp_prefix(&self, selector: u8) -> Vec<usize> {
        let staged = self
            .mtp
            .as_ref()
            .and_then(|mtp| mtp.proposals.as_ref())
            .map_or_else(
                || vec![0; self.geometry.rows],
                |rows| rows.iter().map(Vec::len).collect(),
            );
        staged
            .iter()
            .enumerate()
            .map(|(row, &length)| {
                let raw = usize::from((selector >> (1 + 2 * row)) & 3);
                if selector & 0x80 == 0 {
                    raw
                } else {
                    raw.min(length)
                }
            })
            .collect()
    }

    /// Commits per-row accepted prefixes when they fit the proposals.
    ///
    /// With an MTP bank, `mtp_accepted` is the bank's own accepted prefix
    /// (see [`Self::mtp_prefix`]); without one (`None`), the MTP proposals
    /// must be empty for the commit to succeed.
    fn commit(&mut self, accepted: &[usize], mtp_accepted: Option<&[usize]>) -> bool {
        let Some(active) = self.mark.clone() else {
            return false;
        };
        let fits = |history: &History, accepted: &[usize]| {
            let proposals = history.proposals.clone().unwrap_or_default();
            accepted.len() == proposals.len()
                && accepted
                    .iter()
                    .zip(&proposals)
                    .all(|(&count, row)| count <= row.len())
        };
        let mtp_valid = match (&self.mtp, mtp_accepted) {
            (None, prefix) => prefix.is_none(),
            (Some(mtp), Some(prefix)) => fits(mtp, prefix),
            (Some(mtp), None) => mtp
                .proposals
                .as_ref()
                .is_none_or(|proposals| proposals.iter().all(Vec::is_empty)),
        };
        let valid = fits(&self.history, accepted) && mtp_valid;
        let result = match mtp_accepted {
            None => self.state.commit_prefix(active, accepted),
            Some(_) => self.state.commit_prefixes(active, accepted, mtp_accepted),
        };
        assert_eq!(
            result.is_ok(),
            valid,
            "commit {accepted:?} with MTP {mtp_accepted:?}: {result:?}"
        );
        if valid {
            Self::accept(&mut self.history, accepted);
            if let Some(mtp) = &mut self.mtp {
                let zeros = vec![0; accepted.len()];
                Self::accept(mtp, mtp_accepted.unwrap_or(&zeros));
            }
            self.mark = None;
        }
        valid
    }

    /// Moves each row's accepted proposal prefix into its committed history.
    fn accept(history: &mut History, accepted: &[usize]) {
        let proposals = history.proposals.take().unwrap_or_default();
        for ((row, proposal), &count) in history.committed.iter_mut().zip(proposals).zip(accepted) {
            row.extend(&proposal[..count]);
        }
    }

    /// Rolls the active transaction back.
    fn rollback(&mut self) -> bool {
        match self.mark.take() {
            Some(active) => {
                self.state.rollback(active).expect("active rollback");
                self.history.proposals = None;
                if let Some(mtp) = &mut self.mtp {
                    mtp.proposals = None;
                }
                true
            }
            None => false,
        }
    }

    /// Gathers rows outside a transaction.
    fn reorder(&mut self, indices: &[usize]) -> bool {
        let rows = self.geometry.rows;
        let valid = self.mark.is_none() && indices.iter().all(|&index| index < rows);
        assert_eq!(self.state.reorder(indices).is_ok(), valid);
        if valid {
            for history in std::iter::once(&mut self.history).chain(self.mtp.as_mut()) {
                let before = history.committed.clone();
                history.committed = indices.iter().map(|&index| before[index].clone()).collect();
            }
        }
        valid
    }

    /// Rewinds rows when every group can reconstruct the shorter prefix.
    fn truncate(&mut self, requested: &[usize]) -> bool {
        let geometry = self.geometry;
        let valid = self.mark.is_none()
            && (0..geometry.rows).all(|row| {
                let (from, to) = (self.history.committed[row].len(), requested[row]);
                let ring_safe = |window: usize| to == from || to == 0 || from <= window;
                ring_safe(geometry.windows[0])
                    && ring_safe(geometry.windows[1])
                    && (!geometry.recurrent || to == from || to == 0)
            });
        assert_eq!(
            self.state.truncate(requested).is_ok(),
            valid,
            "truncate {requested:?}"
        );
        if valid {
            for (row, &length) in self.history.committed.iter_mut().zip(requested) {
                row.truncate(length);
            }
        }
        valid
    }

    /// Clears every row outside a transaction.
    fn reset(&mut self) -> bool {
        let valid = self.mark.is_none();
        assert_eq!(self.state.reset().is_ok(), valid);
        if valid {
            for history in std::iter::once(&mut self.history).chain(self.mtp.as_mut()) {
                history.committed.iter_mut().for_each(Vec::clear);
            }
        }
        valid
    }

    /// Publishes one MTP-bank step, batched or per row, when the bank exists.
    fn publish_mtp(&mut self, counts: &[usize], batched: bool) -> bool {
        let Some(mtp) = &self.mtp else {
            assert!(matches!(
                self.state.mtp_step(counts),
                Err(TensorCacheError::MtpNotEnabled)
            ));
            return false;
        };
        let geometry = self.geometry;
        let valid = (0..geometry.rows).all(|row| {
            let staged = mtp.proposals.as_ref().map_or(0, |rows| rows[row].len());
            mtp.working(row).len() + counts[row] <= geometry.capacity
                && (mtp.proposals.is_none() || staged + counts[row] <= geometry.staging)
        });
        let tokens = self.fresh_tokens(counts);
        let mut step = self.state.mtp_step(counts).expect("one count per row");
        if batched {
            let block = counts.iter().copied().max().unwrap_or(0).max(1);
            let (keys, values) = kv((1, 1), &tokens, block, 0);
            step.push_kv_batch(0, 0, keys, values);
        } else {
            for (row, row_tokens) in tokens.iter().enumerate() {
                if !row_tokens.is_empty() {
                    let (keys, values) = kv(
                        (1, 1),
                        std::slice::from_ref(row_tokens),
                        row_tokens.len(),
                        0,
                    );
                    step.push_kv_row(0, 0, row, keys, values);
                }
            }
        }
        let result = self.state.publish_step(step);
        assert_eq!(result.is_ok(), valid, "MTP step {counts:?}: {result:?}");
        if valid {
            let mtp = self.mtp.as_mut().expect("checked above");
            let target = mtp.proposals.as_mut().unwrap_or(&mut mtp.committed);
            for (row, new) in target.iter_mut().zip(tokens) {
                row.extend(new);
            }
        }
        valid
    }

    /// Rewinds the MTP bank by `counts` tokens per row outside a transaction.
    fn truncate_mtp(&mut self, counts: &[usize]) -> bool {
        let Some(mtp) = &self.mtp else {
            assert!(self.state.truncate_mtp(counts).is_err());
            return false;
        };
        let requested: Vec<usize> = mtp
            .committed
            .iter()
            .zip(counts)
            .map(|(row, &count)| row.len().saturating_sub(count))
            .collect();
        // The single MTP group has full storage, so any shorter prefix is valid.
        let valid = self.mark.is_none();
        assert_eq!(self.state.truncate_mtp(&requested).is_ok(), valid);
        if valid {
            let mtp = self.mtp.as_mut().expect("checked above");
            for (row, &length) in mtp.committed.iter_mut().zip(&requested) {
                row.truncate(length);
            }
        }
        valid
    }

    /// Compares lengths, every group's per-row and batched reads, and recurrent state.
    fn check(&self) {
        let geometry = self.geometry;
        let committed: Vec<usize> = self.history.committed.iter().map(Vec::len).collect();
        assert_eq!(self.state.lengths(), committed.as_slice());
        let working: Vec<usize> = (0..geometry.rows)
            .map(|row| self.history.working(row).len())
            .collect();
        assert_eq!(self.state.working_lengths(), working.as_slice());
        assert_eq!(
            self.state.has_active_transaction(),
            self.history.proposals.is_some()
        );
        for group in ALL_GROUPS {
            let shape = geometry.plane_shape(group);
            let plan = self
                .state
                .kv_read_plan(StepBank::Main, group)
                .expect("batched plan");
            for layer in 0..geometry.layers {
                let batched = self
                    .state
                    .main()
                    .native_kv(group)
                    .expect("group")
                    .read_batch(layer, &plan)
                    .expect("batched read");
                for row in 0..geometry.rows {
                    let tokens = self.history.working(row);
                    let first = tokens
                        .len()
                        .saturating_sub(geometry.keep(group, tokens.len()));
                    let visible = &tokens[first..];
                    check_batched_row(&batched, row, visible, first, shape, layer, group);
                    let read = self
                        .state
                        .read_kv(StepBank::Main, group, layer, row)
                        .expect("checked read");
                    let label = format!("main row {row}");
                    check_row_read(&read, visible, first, shape, layer, group, &label);
                }
            }
        }
        self.check_mtp();
        if geometry.recurrent {
            for row in 0..geometry.rows {
                let read = self
                    .state
                    .read_recurrent(StepBank::Main, 0, 0, row)
                    .expect("recurrent read");
                let matrix = read
                    .matrix()
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .expect("F32");
                let last = self.history.working(row).last().copied().unwrap_or(0.0);
                assert_eq!(matrix, [last]);
            }
        }
    }

    /// Compares the MTP bank's cursors and reads with its history, or its absence.
    fn check_mtp(&self) {
        let Some(mtp) = &self.mtp else {
            assert!(self.state.mtp_lengths().is_none());
            return;
        };
        let rows = self.geometry.rows;
        let committed: Vec<usize> = mtp.committed.iter().map(Vec::len).collect();
        assert_eq!(self.state.mtp_lengths(), Some(committed.as_slice()));
        let working: Vec<usize> = (0..rows).map(|row| mtp.working(row).len()).collect();
        assert_eq!(self.state.mtp_working_lengths(), Some(working.as_slice()));
        let plan = self
            .state
            .kv_read_plan(StepBank::Mtp, 0)
            .expect("MTP batched plan");
        let batched = self
            .state
            .mtp()
            .expect("MTP bank")
            .native_kv(0)
            .expect("MTP group")
            .read_batch(0, &plan)
            .expect("MTP batched read");
        for row in 0..rows {
            let tokens = mtp.working(row);
            check_batched_row(&batched, row, &tokens, 0, (1, 1), 0, 0);
            let read = self
                .state
                .read_kv(StepBank::Mtp, 0, 0, row)
                .expect("checked MTP read");
            check_row_read(&read, &tokens, 0, (1, 1), 0, 0, &format!("MTP row {row}"));
        }
    }
}

/// Checks one row's checked per-row read against the oracle's visible tokens.
///
/// The read must cover positions `first..first + visible.len()`, and its
/// materialized keys and values must be exactly the tokens' codes in
/// chronological order: keys positive, values negated for key/value groups
/// and identical to the keys for the latent group (one shared plane). The MTP
/// bank's only group has index 0, a key/value group. `row_label` names the
/// bank and row in failure messages.
fn check_row_read(
    read: &KvCacheRead<Backend>,
    visible: &[f32],
    first: usize,
    shape: (usize, usize),
    layer: usize,
    group: usize,
    row_label: &str,
) {
    let label = format!("{row_label} group {group} layer {layer}");
    assert_eq!(
        read.retained_positions(),
        first..first + visible.len(),
        "{label} positions"
    );
    let host =
        |tensor: Tensor<Backend, 4>| -> Vec<f32> { tensor.into_data().to_vec().expect("F32") };
    let (keys, values) = read.materialize().map_or_else(
        || (Vec::new(), Vec::new()),
        |(keys, values)| (host(keys), host(values)),
    );
    let rows = [visible.to_vec()];
    assert_eq!(
        keys,
        plane_data(&rows, visible.len(), shape, layer, 1.0),
        "{label} keys"
    );
    let value_sign = if group == LATENT_GROUP { 1.0 } else { -1.0 };
    assert_eq!(
        values,
        plane_data(&rows, visible.len(), shape, layer, value_sign),
        "{label} values"
    );
}

/// Checks one row of a batched slot-order read against the oracle's visible tokens.
///
/// Valid slots sorted by key position must be positions `first..` holding
/// exactly the tokens' codes (values negated for key/value groups), and
/// [`KvBatchSegment::masked_values`] must zero every invalid slot.
fn check_batched_row(
    read: &KvBatchRead<Backend>,
    row: usize,
    visible: &[f32],
    first: usize,
    shape: (usize, usize),
    layer: usize,
    group: usize,
) {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for segment in read.segments() {
        let [rows, heads, slots, channels] = segment.keys().dims();
        assert_eq!((heads, channels), shape);
        let positions: Vec<i32> = segment
            .key_positions()
            .clone()
            .into_data()
            .to_vec()
            .expect("I32");
        let valid: Vec<bool> = segment.key_valid().map_or_else(
            || vec![true; rows * slots],
            |valid| valid.clone().into_data().to_vec().expect("Bool"),
        );
        let host =
            |tensor: Tensor<Backend, 4>| -> Vec<f32> { tensor.into_data().to_vec().expect("F32") };
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
                assert_eq!(entries(&masked), entries(&value_data));
            } else {
                assert!(entries(&masked).iter().all(|&value| value == 0.0));
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
    let label = format!("group {group} layer {layer} row {row}");
    assert_eq!(keys, expected(1.0), "{label} keys");
    let value_sign = if group == LATENT_GROUP { 1.0 } else { -1.0 };
    assert_eq!(values, expected(value_sign), "{label} values");
}

/// Builds `[rows, heads, block, channels]` keys and negated values of `layer`.
fn kv(
    shape: (usize, usize),
    tokens: &[Vec<f32>],
    block: usize,
    layer: usize,
) -> (Tensor<Backend, 4>, Tensor<Backend, 4>) {
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
fn latent(width: usize, tokens: &[Vec<f32>], block: usize, layer: usize) -> Tensor<Backend, 3> {
    let data = plane_data(tokens, block, (1, width), layer, 1.0);
    Tensor::from_data(
        TensorData::new(data, [tokens.len(), block, width]),
        &Default::default(),
    )
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
