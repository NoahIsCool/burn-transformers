//! Differential cache state machines with independent oracles.
//!
//! [`exercise`] drives the metadata-only [`SequenceLengths`] transaction
//! machine and the ring planners against integer oracles. [`exercise_tensor_state`]
//! drives the tensor-backed [`TensorDecodeState`] that models actually use:
//! full, ring, mirrored, and latent groups plus an optional recurrent group,
//! written through atomic step write sets (batched or per row), speculative
//! transactions with ragged acceptance, reorder, truncate, and reset. Its
//! oracle keeps each row's token history as a `Vec<f32>` of unique values and
//! compares every visible cache entry after every operation; rejected
//! operations must change nothing.

use std::num::NonZeroUsize;

use burn::backend::Flex;
use burn::tensor::{DType, Tensor, TensorData};
use transformers::cache::{
    CacheConfig, CacheMark, KvCacheStorage, NativeKvCacheSpec, RecurrentStateSpec, SequenceLengths,
    TensorCacheMark, TensorDecodeState, TransactionalCache, plan_ring_write,
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

/// Indices of the harness's KV groups, in creation order.
const KV_GROUPS: [usize; 3] = [0, 1, 2];
/// Index of the latent group.
const LATENT_GROUP: usize = 3;
/// Convolution width of the recurrent group.
const CONVOLUTION_WIDTH: usize = 2;

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

/// Runs at most 64 tensor-state operations decoded from `input`.
///
/// # Panics
///
/// Panics when the tensor state disagrees with the oracle, which is a bug.
pub fn exercise_tensor_state(input: &[u8]) {
    let mut bytes = crate::Bytes::new(input);
    let rows = usize::from(bytes.byte() % 3) + 1;
    let capacity = usize::from(bytes.byte() % 11) + 2;
    let geometry = Geometry {
        rows,
        capacity,
        windows: [
            usize::from(bytes.byte()) % capacity + 1,
            usize::from(bytes.byte()) % capacity + 1,
        ],
        staging: usize::from(bytes.byte() % 3) + 1,
        recurrent: bytes.byte().is_multiple_of(2),
    };
    let mut state = tensor_state(geometry);
    let mut history = History {
        committed: vec![Vec::new(); rows],
        proposals: None,
    };
    let mut mark: Option<TensorCacheMark> = None;
    let mut next_value = 1.0_f32;
    for _ in 0..input.len().min(512).div_ceil(6).min(64) {
        let operation = bytes.byte() % 7;
        let counts: Vec<usize> = (0..rows).map(|_| usize::from(bytes.byte() % 4)).collect();
        let flag = bytes.byte().is_multiple_of(2);
        let before = history.clone();
        let accepted = match operation {
            0 => publish(
                &mut state,
                &mut history,
                geometry,
                &counts,
                flag,
                &mut next_value,
            ),
            1 => {
                let result = state.begin_transaction();
                assert_eq!(result.is_ok(), mark.is_none());
                if let Ok(new_mark) = result {
                    mark = Some(new_mark);
                    history.proposals = Some(vec![Vec::new(); rows]);
                }
                mark.is_some() && history != before
            }
            2 => match &mark {
                Some(active) => {
                    let proposals = history.proposals.clone().unwrap_or_default();
                    let valid = counts
                        .iter()
                        .zip(&proposals)
                        .all(|(&count, row)| count <= row.len());
                    let result = state.commit_prefix(active.clone(), &counts);
                    assert_eq!(result.is_ok(), valid, "commit {counts:?}: {result:?}");
                    if valid {
                        for ((row, proposal), &count) in
                            history.committed.iter_mut().zip(proposals).zip(&counts)
                        {
                            row.extend(&proposal[..count]);
                        }
                        history.proposals = None;
                        mark = None;
                    }
                    valid
                }
                None => false,
            },
            3 => match mark.take() {
                Some(active) => {
                    state.rollback(active).expect("active rollback");
                    history.proposals = None;
                    true
                }
                None => false,
            },
            4 => {
                let indices: Vec<usize> = counts.iter().map(|&index| index % (rows + 1)).collect();
                let valid = mark.is_none() && indices.iter().all(|&index| index < rows);
                assert_eq!(state.reorder(&indices).is_ok(), valid);
                if valid {
                    history.committed = indices
                        .iter()
                        .map(|&index| before.committed[index].clone())
                        .collect();
                }
                valid
            }
            5 => {
                let requested: Vec<usize> = (0..rows)
                    .map(|row| {
                        if flag {
                            0
                        } else {
                            history.committed[row].len().saturating_sub(counts[row])
                        }
                    })
                    .collect();
                let valid = mark.is_none()
                    && (0..rows).all(|row| {
                        let (from, to) = (history.committed[row].len(), requested[row]);
                        let ring_safe = |window: usize| to == from || to == 0 || from <= window;
                        ring_safe(geometry.windows[0])
                            && ring_safe(geometry.windows[1])
                            && (!geometry.recurrent || to == from || to == 0)
                    });
                assert_eq!(
                    state.truncate(&requested).is_ok(),
                    valid,
                    "truncate {requested:?}"
                );
                if valid {
                    for (row, &length) in history.committed.iter_mut().zip(&requested) {
                        row.truncate(length);
                    }
                }
                valid
            }
            _ => {
                let valid = mark.is_none();
                assert_eq!(state.reset().is_ok(), valid);
                if valid {
                    history.committed.iter_mut().for_each(Vec::clear);
                }
                valid
            }
        };
        if !accepted {
            assert_eq!(
                history, before,
                "rejected operation {operation} changed the oracle"
            );
        }
        check_tensor_state(&state, &history, geometry);
    }
}

/// Allocates the harness state: full, ring, mirrored, and latent groups (+ recurrent).
fn tensor_state(geometry: Geometry) -> TensorDecodeState<Backend> {
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
        let mut spec = NativeKvCacheSpec::new(1, 1, 1, storage, DType::F32)
            .expect("scalar geometry")
            .with_staging_capacity(geometry.staging);
        if let Some(group_window) = group_window {
            spec = spec.with_sliding_window(group_window);
        }
        state.main_mut().add_native_kv(spec).expect("KV group");
    }
    let latent = NativeKvCacheSpec::latent(1, 1, KvCacheStorage::Full, DType::F32)
        .expect("scalar latent")
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
    state
}

/// Publishes one step of fresh tokens, batched or per row; returns whether it was accepted.
fn publish(
    state: &mut TensorDecodeState<Backend>,
    history: &mut History,
    geometry: Geometry,
    counts: &[usize],
    batched: bool,
    next_value: &mut f32,
) -> bool {
    let rows = geometry.rows;
    let valid = (0..rows).all(|row| {
        let staged = history
            .proposals
            .as_ref()
            .map_or(0, |proposals| proposals[row].len());
        history.working(row).len() + counts[row] <= geometry.capacity
            && (history.proposals.is_none() || staged + counts[row] <= geometry.staging)
    });
    let tokens: Vec<Vec<f32>> = counts
        .iter()
        .map(|&count| {
            (0..count)
                .map(|_| {
                    *next_value += 1.0;
                    *next_value
                })
                .collect()
        })
        .collect();
    let mut step = state.step(counts).expect("one count per row");
    let block = counts.iter().copied().max().unwrap_or(0).max(1);
    if batched {
        let padded: Vec<f32> = tokens
            .iter()
            .flat_map(|row| {
                row.iter()
                    .copied()
                    .chain(std::iter::repeat_n(0.0, block - row.len()))
            })
            .collect();
        let keys = tensor4(&padded, [rows, 1, block, 1]);
        for group in KV_GROUPS {
            step.push_kv_batch(group, 0, keys.clone(), keys.clone().neg());
        }
        step.push_latent_batch(LATENT_GROUP, 0, keys.reshape([rows, block, 1]));
    } else {
        for row in (0..rows).filter(|&row| counts[row] > 0) {
            let keys = tensor4(&tokens[row], [1, 1, counts[row], 1]);
            for group in KV_GROUPS {
                step.push_kv_row(group, 0, row, keys.clone(), keys.clone().neg());
            }
            step.push_latent_row(LATENT_GROUP, 0, row, keys.reshape([1, counts[row], 1]));
        }
    }
    if geometry.recurrent {
        for row in (0..rows).filter(|&row| counts[row] > 0) {
            let count = counts[row];
            let matrices = Tensor::<Backend, 5>::from_data(
                TensorData::new(tokens[row].clone(), [1, count, 1, 1, 1]),
                &Default::default(),
            );
            let convolutions = Tensor::<Backend, 4>::from_data(
                TensorData::new(
                    tokens[row]
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
    let result = state.publish_step(step);
    assert_eq!(result.is_ok(), valid, "step {counts:?}: {result:?}");
    if valid {
        let target = match &mut history.proposals {
            Some(proposals) => proposals,
            None => &mut history.committed,
        };
        for (row, new) in target.iter_mut().zip(tokens) {
            row.extend(new);
        }
    }
    valid
}

/// Compares lengths, every group's visible tokens, and recurrent state with the oracle.
fn check_tensor_state(state: &TensorDecodeState<Backend>, history: &History, geometry: Geometry) {
    let committed: Vec<usize> = history.committed.iter().map(Vec::len).collect();
    assert_eq!(state.lengths(), committed.as_slice());
    let working: Vec<usize> = (0..geometry.rows)
        .map(|row| history.working(row).len())
        .collect();
    assert_eq!(state.working_lengths(), working.as_slice());
    assert_eq!(state.has_active_transaction(), history.proposals.is_some());
    let bank = state.main();
    let windows = [
        usize::MAX,
        geometry.windows[0],
        geometry.windows[1],
        usize::MAX,
    ];
    for (row, &length) in committed.iter().enumerate() {
        let tokens = history.working(row);
        for (group, window) in windows.iter().enumerate() {
            let read = bank
                .native_kv(group)
                .expect("group")
                .read_working_sequence(0, row, length)
                .expect("working read");
            let visible = &tokens[tokens.len().saturating_sub(*window)..];
            assert_eq!(
                read.retained_positions(),
                tokens.len() - visible.len()..tokens.len()
            );
            let keys = read.materialize().map_or_else(Vec::new, |(keys, _)| {
                keys.into_data().to_vec::<f32>().expect("F32")
            });
            assert_eq!(keys, visible, "group {group} row {row}");
        }
        if geometry.recurrent {
            let read = bank
                .recurrent(0)
                .expect("recurrent group")
                .read_working_sequence(0, row)
                .expect("recurrent read");
            let matrix = read
                .matrix()
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("F32");
            assert_eq!(matrix, [tokens.last().copied().unwrap_or(0.0)]);
        }
    }
}

/// Builds a rank-four F32 tensor from host values.
fn tensor4(values: &[f32], shape: [usize; 4]) -> Tensor<Backend, 4> {
    Tensor::from_data(TensorData::new(values.to_vec(), shape), &Default::default())
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
