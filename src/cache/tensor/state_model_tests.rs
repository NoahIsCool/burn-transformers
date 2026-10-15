//! Model-based property test of the tensor decode-state lifecycle.
//!
//! Random sequences of append, begin, stage, commit, rollback, reorder,
//! truncate, and reset drive a real [`TensorDecodeState`] holding full,
//! standard-ring, and mirrored-ring KV groups plus an optional recurrent
//! group. A straightforward oracle keeps each row's token history as a
//! `Vec<f32>`; every token carries a unique value, so a stale, misplaced, or
//! lost cache slot is detected exactly (F32 equality, no tolerance). After
//! every operation the test compares committed and working lengths, the
//! retained window and contents of every KV group, and the recurrent terminal
//! state. Rejected operations must leave both the state and the oracle
//! unchanged.

use burn::backend::Flex;
use burn::tensor::{DType, Tensor, TensorData};
use core::num::NonZeroUsize;
use proptest::prelude::*;

use super::*;
use crate::cache::KvCacheStorage;

/// Deterministic host backend for exact value comparisons.
type TestBackend = Flex<f32, i32>;

/// Index of the full-storage KV group.
const FULL: usize = 0;
/// Index of the standard sliding-ring KV group.
const RING: usize = 1;
/// Index of the mirrored sliding-ring KV group.
const MIRRORED: usize = 2;
/// Width of the recurrent group's convolution history.
const CONVOLUTION_WIDTH: usize = 2;

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
}

/// One lifecycle operation; vectors may have the wrong length on purpose.
#[derive(Debug, Clone)]
enum Operation {
    /// Append (or, inside a transaction, stage) per-row token counts.
    Append(Vec<usize>),
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
    /// Allocates the state with three KV groups and an optional recurrent group.
    fn new(setup: Setup) -> Self {
        let config = CacheConfig::new(setup.rows, setup.capacity).expect("valid config");
        let mut state = TensorDecodeState::with_config(config, &Default::default());
        let window = |width| NonZeroUsize::new(width).expect("non-zero window");
        for (storage, group_window) in [
            (KvCacheStorage::Full, None),
            (KvCacheStorage::SlidingRing, Some(window(setup.ring_window))),
            (
                KvCacheStorage::MirroredSlidingRing,
                Some(window(setup.mirrored_window)),
            ),
        ] {
            let mut spec = NativeKvCacheSpec::new(1, 1, 1, storage, DType::F32)
                .expect("geometry")
                .with_staging_capacity(setup.staging);
            if let Some(group_window) = group_window {
                spec = spec.with_sliding_window(group_window);
            }
            state.main_mut().add_native_kv(spec).expect("KV group");
        }
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

    /// Allocates `count` fresh token values.
    fn fresh_tokens(&mut self, count: usize) -> Vec<f32> {
        (0..count)
            .map(|_| {
                let value = self.next_value;
                self.next_value += 1.0;
                value
            })
            .collect()
    }

    /// Applies one operation to both systems; returns whether it was accepted.
    fn apply(&mut self, operation: &Operation) -> bool {
        match operation {
            Operation::Append(counts) if self.oracle.proposals.is_some() => self.stage(counts),
            Operation::Append(counts) => self.append(counts),
            Operation::Begin => self.begin(),
            Operation::Commit(accepted) => self.commit(accepted),
            Operation::Rollback => self.rollback(),
            Operation::Reorder(indices) => self.reorder(indices),
            Operation::Truncate(lengths) => self.truncate(lengths),
            Operation::Reset => self.reset(),
        }
    }

    /// Publishes committed tokens through prepare, advance, and infallible apply.
    fn append(&mut self, counts: &[usize]) -> bool {
        let rows = self.setup.rows;
        let valid = counts.len() == rows
            && (0..rows)
                .all(|row| self.oracle.committed[row].len() + counts[row] <= self.setup.capacity);
        if !valid {
            assert!(
                self.state.advance(counts).is_err(),
                "advance accepted {counts:?}"
            );
            return false;
        }
        let tokens: Vec<Vec<f32>> = counts
            .iter()
            .map(|&count| self.fresh_tokens(count))
            .collect();
        let bank = self.state.main();
        let mut prepared = Vec::new();
        for group in [FULL, RING, MIRRORED] {
            let cache = bank.native_kv(group).expect("group");
            let writes = (0..rows)
                .filter(|&row| counts[row] > 0)
                .map(|row| {
                    let (keys, values) = kv_tokens(&tokens[row]);
                    let start = self.oracle.committed[row].len();
                    cache
                        .prepare_sequence_write(0, row, start, keys, values)
                        .expect("valid write")
                })
                .collect();
            let recurrent = (group == FULL && self.setup.recurrent).then(|| {
                let cache = bank.recurrent(0).expect("recurrent group");
                let writes = (0..rows)
                    .filter(|&row| counts[row] > 0)
                    .map(|row| {
                        let last = *tokens[row].last().expect("non-empty");
                        let (matrix, convolution) = recurrent_state(last);
                        cache
                            .prepare_sequence_state(0, row, matrix, convolution)
                            .expect("valid state")
                    })
                    .collect();
                (0, writes)
            });
            prepared.push(
                bank.prepare_direct_writes(Some((group, writes)), recurrent)
                    .expect("ordinary writes"),
            );
        }
        self.state.advance(counts).expect("validated advance");
        for writes in prepared {
            self.state.main_mut().apply_prepared_direct_writes(writes);
        }
        for (history, new) in self.oracle.committed.iter_mut().zip(tokens) {
            history.extend(new);
        }
        true
    }

    /// Stages proposal tokens in every group and advances the working cursor.
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
            let (keys, values) = kv_tokens(&vec![0.0; counts[row]]);
            let cache = self.state.main_mut().native_kv_mut(FULL).expect("group");
            assert!(matches!(
                cache.stage_sequence(0, row, keys, values),
                Err(TensorCacheError::StagingCapacityExceeded { .. })
            ));
            return false;
        }
        let tokens: Vec<Vec<f32>> = counts
            .iter()
            .map(|&count| self.fresh_tokens(count))
            .collect();
        for row in (0..rows).filter(|&row| counts[row] > 0) {
            for group in [FULL, RING, MIRRORED] {
                let (keys, values) = kv_tokens(&tokens[row]);
                self.state
                    .main_mut()
                    .native_kv_mut(group)
                    .expect("group")
                    .stage_sequence(0, row, keys, values)
                    .expect("valid staging");
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
        let bank = self.state.main();
        for (row, &committed) in lengths.iter().enumerate() {
            let tokens = self.oracle.working(row);
            for (group, window) in [
                (FULL, tokens.len()),
                (RING, self.setup.ring_window),
                (MIRRORED, self.setup.mirrored_window),
            ] {
                let visible = &tokens[tokens.len().saturating_sub(window)..];
                let read = bank
                    .native_kv(group)
                    .expect("group")
                    .read_working_sequence(0, row, committed)
                    .expect("working read");
                assert_eq!(
                    read.retained_positions(),
                    tokens.len() - visible.len()..tokens.len(),
                    "group {group}, row {row}"
                );
                let (keys, values) = read
                    .materialize()
                    .map_or((Vec::new(), Vec::new()), |(keys, values)| {
                        (host(keys), host(values))
                    });
                assert_eq!(keys, visible, "group {group}, row {row} keys");
                let negated: Vec<f32> = visible.iter().map(|value| -value).collect();
                assert_eq!(values, negated, "group {group}, row {row} values");
            }
            if self.setup.recurrent {
                let read = bank
                    .recurrent(0)
                    .expect("recurrent group")
                    .read_working_sequence(0, row)
                    .expect("recurrent read");
                let expected = tokens.last().copied().unwrap_or(0.0);
                assert_eq!(host(read.matrix().clone()), [expected], "row {row} matrix");
                assert_eq!(
                    host(read.convolution().clone()),
                    [expected; CONVOLUTION_WIDTH],
                    "row {row} convolution"
                );
            }
        }
    }
}

/// Builds scalar-head keys and negated values for a token run.
fn kv_tokens(tokens: &[f32]) -> (Tensor<TestBackend, 4>, Tensor<TestBackend, 4>) {
    let shape = [1, 1, tokens.len(), 1];
    let keys = Tensor::from_data(TensorData::new(tokens.to_vec(), shape), &Default::default());
    let values = Tensor::from_data(
        TensorData::new(tokens.iter().map(|value| -value).collect::<Vec<_>>(), shape),
        &Default::default(),
    );
    (keys, values)
}

/// Builds the terminal recurrent state "last token value" for one row.
fn recurrent_state(value: f32) -> (Tensor<TestBackend, 4>, Tensor<TestBackend, 3>) {
    (
        Tensor::full([1, 1, 1, 1], value, &Default::default()),
        Tensor::full([1, 1, CONVOLUTION_WIDTH], value, &Default::default()),
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

/// Transfers a tensor to the host for exact comparison.
fn host<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
    tensor.into_data().to_vec().expect("F32 tensor")
}

/// Generates a per-row vector that is occasionally the wrong length.
fn row_vector(rows: usize, maximum: usize) -> impl Strategy<Value = Vec<usize>> {
    prop_oneof![
        9 => prop::collection::vec(0..=maximum, rows),
        1 => prop::collection::vec(0..=maximum, 0..=rows + 1),
    ]
}

/// Generates operations that straddle every validity boundary.
fn operation(setup: Setup) -> impl Strategy<Value = Operation> {
    let rows = setup.rows;
    prop_oneof![
        5 => row_vector(rows, setup.capacity.min(4)).prop_map(Operation::Append),
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
    (1_usize..=3, 2_usize..=10, 1_usize..=4, any::<bool>()).prop_flat_map(
        |(rows, capacity, staging, recurrent)| {
            (1..=capacity, 1..=capacity).prop_map(move |(ring_window, mirrored_window)| Setup {
                rows,
                capacity,
                ring_window,
                mirrored_window,
                staging,
                recurrent,
            })
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

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
