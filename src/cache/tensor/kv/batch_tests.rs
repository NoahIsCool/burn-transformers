//! Batched KV writes and reads against the per-row reference paths.
//!
//! Every batched operation is compared with the established per-row API on
//! the same inputs: committed and staged tensors must be bit-identical, and a
//! batched slot-order read reordered by its key positions must reproduce the
//! per-row chronological read exactly. Attention computed from a batched read
//! and its exclusion mask must match per-row causal attention within F32
//! rounding (`1e-5`), since only the key summation order differs.

use burn::backend::Flex;
use burn::tensor::{DType, Int, Tensor, TensorData};
use core::num::NonZeroUsize;
use proptest::prelude::*;

use super::*;
use crate::cache::{CacheConfig, KvCacheStorage};

/// Deterministic host backend for exact comparisons.
type TestBackend = Flex<f32, i32>;
/// Key/value heads in every fixture.
const HEADS: usize = 2;
/// Channels per head in every fixture.
const DIM: usize = 3;

/// Returns the default Flex device.
fn device() -> <TestBackend as burn::tensor::backend::BackendTypes>::Device {
    Default::default()
}

/// Allocates a one-layer cache with the given storage, window, and staging policy.
fn cache(
    rows: usize,
    capacity: usize,
    storage: KvCacheStorage,
    window: usize,
    staging: usize,
    in_place: bool,
) -> NativeKvCache<TestBackend> {
    let config = CacheConfig::new(rows, capacity).expect("valid config");
    let mut spec = NativeKvCacheSpec::new(1, HEADS, DIM, storage, DType::F32)
        .expect("geometry")
        .with_staging_capacity(staging);
    if storage != KvCacheStorage::Full {
        spec = spec.with_sliding_window(NonZeroUsize::new(window).expect("non-zero window"));
    }
    if in_place {
        spec = spec.with_in_place_staging();
    }
    NativeKvCache::new(config, rows, spec, &device()).expect("allocation")
}

/// Builds a `[rows, HEADS, block, DIM]` tensor of distinct values offset by `tag`.
fn block(rows: usize, block: usize, tag: f32) -> Tensor<TestBackend, 4> {
    let values = (0..rows * HEADS * block * DIM)
        .map(|index| tag * 10_000.0 + index as f32)
        .collect::<Vec<_>>();
    Tensor::from_data(
        TensorData::new(values, [rows, HEADS, block, DIM]),
        &device(),
    )
}

/// Slices row `row`, columns `columns` of a block into a per-row `[1, HEADS, n, DIM]` input.
fn row_input(
    tensor: &Tensor<TestBackend, 4>,
    row: usize,
    columns: core::ops::Range<usize>,
) -> Tensor<TestBackend, 4> {
    tensor
        .clone()
        .slice([row..row + 1, 0..HEADS, columns, 0..DIM])
}

/// Transfers a float tensor to the host.
fn host<const D: usize>(tensor: &Tensor<TestBackend, D>) -> Vec<f32> {
    tensor.clone().into_data().to_vec().expect("F32 tensor")
}

/// Transfers an Int tensor to the host.
fn host_int(tensor: &Tensor<TestBackend, 2, Int>) -> Vec<i32> {
    tensor.clone().into_data().to_vec().expect("I32 tensor")
}

/// Transfers a Bool tensor to the host.
fn host_bool<const D: usize>(tensor: &Tensor<TestBackend, D, Bool>) -> Vec<bool> {
    tensor.clone().into_data().to_vec().expect("Bool tensor")
}

/// Writes the same append once per row through the established API.
fn write_per_row(
    cache: &mut NativeKvCache<TestBackend>,
    append: &KvBatchAppend,
    keys: &Tensor<TestBackend, 4>,
    values: &Tensor<TestBackend, 4>,
) {
    for row in 0..append.rows() {
        let count = append.counts()[row];
        if count == 0 {
            continue;
        }
        let columns = append.columns()[row]..append.columns()[row] + count;
        cache
            .write_sequence(
                0,
                row,
                append.starts()[row],
                row_input(keys, row, columns.clone()),
                row_input(values, row, columns),
            )
            .expect("per-row reference write");
    }
}

/// Every committed and staged tensor of layer zero.
fn snapshot(cache: &NativeKvCache<TestBackend>) -> Vec<Vec<f32>> {
    let layer = &cache.layers[0];
    layer
        .planes
        .iter()
        .chain(layer.staged.iter().flatten())
        .map(host)
        .collect()
}

/// Reconstructs one row's `(position, key)` pairs in chronological order from a batched read.
fn chronological_row(read: &KvBatchRead<TestBackend>, row: usize) -> Vec<(i32, Vec<f32>)> {
    let mut keys = Vec::new();
    for segment in read.segments() {
        let [_, _, slots, _] = segment.keys().dims();
        let positions = host_int(segment.key_positions());
        let valid = segment
            .key_valid()
            .map_or_else(|| vec![true; positions.len()], host_bool);
        let data = host(
            &segment
                .keys()
                .clone()
                .slice([row..row + 1, 0..HEADS, 0..slots, 0..DIM]),
        );
        for slot in (0..slots).filter(|&slot| valid[row * slots + slot]) {
            let key = (0..HEADS)
                .flat_map(|head| {
                    let base = (head * slots + slot) * DIM;
                    data[base..base + DIM].to_vec()
                })
                .collect();
            keys.push((positions[row * slots + slot], key));
        }
    }
    keys.sort_by_key(|(position, _)| *position);
    keys
}

/// Converts a per-row chronological read into `(position, key)` pairs.
fn reference_row(read: &crate::cache::KvCacheRead<TestBackend>) -> Vec<(i32, Vec<f32>)> {
    let Some((keys, _)) = read.materialize() else {
        return Vec::new();
    };
    let [_, _, time, _] = keys.dims();
    let data = host(&keys);
    read.retained_positions()
        .zip(0..time)
        .map(|(position, slot)| {
            let key = (0..HEADS)
                .flat_map(|head| {
                    let base = (head * time + slot) * DIM;
                    data[base..base + DIM].to_vec()
                })
                .collect();
            (i32::try_from(position).expect("small position"), key)
        })
        .collect()
}

/// All storage modes exercised by the parity tests, with their window.
const STORAGES: [(KvCacheStorage, usize); 3] = [
    (KvCacheStorage::Full, 1),
    (KvCacheStorage::SlidingRing, 3),
    (KvCacheStorage::MirroredSlidingRing, 3),
];

/// Constructors place tokens by padding side and reject impossible layouts.
#[test]
fn append_layouts_validate_rows_columns_and_positions() {
    let right = KvBatchAppend::right_padded(&[4, 7], &[2, 1], 3).expect("right padding");
    assert_eq!(right.columns(), &[0, 0]);
    assert_eq!(right.ends(), vec![6, 8]);
    assert_eq!(right.block(), 3);
    let left = KvBatchAppend::left_padded(&[4, 7], &[2, 1], 3).expect("left padding");
    assert_eq!(left.columns(), &[1, 2]);
    assert_eq!(left.starts(), &[4, 7]);
    assert_eq!(left.counts(), &[2, 1]);
    assert!(matches!(
        KvBatchAppend::left_padded(&[0], &[4], 3),
        Err(TensorCacheError::AppendOutsideBlock {
            count: 4,
            block: 3,
            ..
        })
    ));
    assert!(matches!(
        KvBatchAppend::with_columns(&[0, 0], &[2, 0], &[2, 1], 3),
        Err(TensorCacheError::AppendOutsideBlock {
            sequence: 0,
            column: 2,
            ..
        })
    ));
    assert!(matches!(
        KvBatchAppend::with_columns(&[0], &[0, 0], &[1, 1], 3),
        Err(TensorCacheError::Metadata(_))
    ));
    assert!(matches!(
        KvBatchAppend::right_padded(&[usize::MAX], &[1], 1),
        Err(TensorCacheError::WritePositionOverflow { .. })
    ));
    let message = TensorCacheError::AppendOutsideBlock {
        sequence: 1,
        column: 2,
        count: 3,
        block: 4,
    }
    .to_string();
    assert!(message.contains("column 2"), "{message}");
}

/// A uniform batch is one run; ragged rows split into per-row runs.
#[test]
fn uniform_batches_write_with_one_assignment_per_segment() {
    let cache = cache(4, 16, KvCacheStorage::Full, 1, 0, false);
    let uniform = KvBatchAppend::right_padded(&[5; 4], &[2; 4], 2).expect("layout");
    let write = cache
        .prepare_batch_write(0, &uniform, vec![block(4, 2, 1.0), block(4, 2, 2.0)])
        .expect("valid write");
    assert_eq!(write.runs.len(), 1);
    assert_eq!(write.runs[0].rows, 0..4);
    let ragged = KvBatchAppend::right_padded(&[5, 5, 6, 0], &[2, 2, 1, 0], 2).expect("layout");
    let write = cache
        .prepare_batch_write(0, &ragged, vec![block(4, 2, 1.0), block(4, 2, 2.0)])
        .expect("valid write");
    assert_eq!(
        write
            .runs
            .iter()
            .map(|run| run.rows.clone())
            .collect::<Vec<_>>(),
        [0..2, 2..3]
    );
}

proptest! {
    /// Batched committed writes equal per-row writes in every storage mode.
    #[test]
    fn batched_writes_match_per_row_writes(
        rows in 1_usize..5,
        width in 1_usize..5,
        seed in prop::collection::vec((0_usize..8, 0_usize..5, any::<bool>()), 4),
        storage in 0_usize..3,
    ) {
        let (storage, window) = STORAGES[storage];
        let starts: Vec<usize> = seed.iter().take(rows).map(|row| row.0).collect();
        let counts: Vec<usize> = seed.iter().take(rows).map(|row| row.1.min(width)).collect();
        let append = if seed[0].2 {
            KvBatchAppend::left_padded(&starts, &counts, width)
        } else {
            KvBatchAppend::right_padded(&starts, &counts, width)
        }
        .expect("layout");
        let keys = block(rows, width, 1.0);
        let values = block(rows, width, 2.0);
        let mut batched = cache(rows, 16, storage, window, 0, false);
        let mut reference = cache(rows, 16, storage, window, 0, false);
        batched.write_batch(0, &append, keys.clone(), values.clone()).expect("batched write");
        write_per_row(&mut reference, &append, &keys, &values);
        prop_assert_eq!(snapshot(&batched), snapshot(&reference));
    }

    /// Batched reads, reordered by key position, equal per-row chronological reads.
    #[test]
    fn batched_reads_match_per_row_reads(
        lengths in prop::collection::vec(0_usize..12, 1..5),
        storage in 0_usize..3,
    ) {
        let (storage, window) = STORAGES[storage];
        let rows = lengths.len();
        let mut cache = cache(rows, 12, storage, window, 0, false);
        let append = KvBatchAppend::right_padded(&vec![0; rows], &lengths, 12).expect("layout");
        cache
            .write_batch(0, &append, block(rows, 12, 1.0), block(rows, 12, 2.0))
            .expect("prefill");
        let plan = cache.plan_batch_read(&lengths).expect("plan");
        let read = cache.read_batch(0, &plan).expect("batched read");
        for (row, &length) in lengths.iter().enumerate() {
            let reference = cache.read_sequence(0, row, length).expect("per-row read");
            prop_assert_eq!(&read.retained_positions()[row], &reference.retained_positions());
            prop_assert_eq!(chronological_row(&read, row), reference_row(&reference));
        }
        prop_assert_eq!(plan.retained_positions(), read.retained_positions());
    }

    /// Working batched reads match per-row working reads for separate and in-place staging.
    #[test]
    fn working_batched_reads_match_per_row_reads(
        committed in prop::collection::vec(0_usize..8, 1..4),
        proposals in prop::collection::vec(0_usize..4, 3),
        storage in 0_usize..4,
    ) {
        let rows = committed.len();
        let (storage, window, in_place) = match storage {
            3 => (KvCacheStorage::Full, 1, true),
            index => (STORAGES[index].0, STORAGES[index].1, false),
        };
        let proposals: Vec<usize> = proposals.into_iter().take(rows).collect();
        let mut cache = cache(rows, 12, storage, window, 3, in_place);
        let prefill = KvBatchAppend::right_padded(&vec![0; rows], &committed, 8).expect("layout");
        cache
            .write_batch(0, &prefill, block(rows, 8, 1.0), block(rows, 8, 2.0))
            .expect("prefill");
        cache.begin_staging_from(&committed).expect("staging");
        let stage = KvBatchAppend::left_padded(&committed, &proposals, 3).expect("layout");
        cache
            .stage_batch(0, &stage, block(rows, 3, 3.0), block(rows, 3, 4.0))
            .expect("stage");
        let plan = cache.plan_working_batch_read(&committed).expect("plan");
        let read = cache.read_batch(0, &plan).expect("batched read");
        for (row, &length) in committed.iter().enumerate() {
            let reference = cache.read_working_sequence(0, row, length).expect("per-row read");
            prop_assert_eq!(&read.retained_positions()[row], &reference.retained_positions());
            prop_assert_eq!(chronological_row(&read, row), reference_row(&reference));
        }
    }
}

/// Batched staging equals per-row staging, and commits publish identical tensors.
#[test]
fn batched_staging_and_commit_match_per_row_staging() {
    for (storage, window, in_place) in [
        (KvCacheStorage::Full, 1, false),
        (KvCacheStorage::Full, 1, true),
        (KvCacheStorage::SlidingRing, 3, false),
        (KvCacheStorage::MirroredSlidingRing, 3, false),
    ] {
        let committed = [2, 5, 0];
        let counts = [2, 0, 1];
        let mut batched = cache(3, 10, storage, window, 3, in_place);
        let mut reference = cache(3, 10, storage, window, 3, in_place);
        let prefill = KvBatchAppend::right_padded(&[0; 3], &committed, 5).expect("layout");
        for cache in [&mut batched, &mut reference] {
            cache
                .write_batch(0, &prefill, block(3, 5, 1.0), block(3, 5, 2.0))
                .expect("prefill");
            cache.begin_staging_from(&committed).expect("staging");
        }
        let append = KvBatchAppend::right_padded(&committed, &counts, 2).expect("layout");
        let (keys, values) = (block(3, 2, 3.0), block(3, 2, 4.0));
        batched
            .stage_batch(0, &append, keys.clone(), values.clone())
            .expect("batched stage");
        for (row, &count) in counts.iter().enumerate().filter(|(_, count)| **count > 0) {
            reference
                .stage_sequence(
                    0,
                    row,
                    row_input(&keys, row, 0..count),
                    row_input(&values, row, 0..count),
                )
                .expect("per-row stage");
        }
        assert_eq!(
            snapshot(&batched),
            snapshot(&reference),
            "{storage:?} {in_place}"
        );
        assert_eq!(batched.staged_lengths, reference.staged_lengths);
        for cache in [&mut batched, &mut reference] {
            let plan = cache
                .prepare_staging_commit(&committed, &counts, &[1, 0, 1])
                .expect("commit plan");
            cache.apply_staging_commit(plan);
        }
        assert_eq!(
            snapshot(&batched),
            snapshot(&reference),
            "{storage:?} {in_place}"
        );
    }
}

/// In-place staging reads proposals from committed storage with no staging buffers.
#[test]
fn in_place_staging_reads_one_contiguous_view_and_commits_by_metadata() {
    let mut cache = cache(2, 8, KvCacheStorage::Full, 1, 2, true);
    assert!(cache.layers[0].staged.is_none());
    let prefill = KvBatchAppend::right_padded(&[0, 0], &[3, 1], 3).expect("layout");
    cache
        .write_batch(0, &prefill, block(2, 3, 1.0), block(2, 3, 2.0))
        .expect("prefill");
    cache.begin_staging_from(&[3, 1]).expect("staging");
    let append = KvBatchAppend::right_padded(&[3, 1], &[2, 1], 2).expect("layout");
    cache
        .stage_batch(0, &append, block(2, 2, 3.0), block(2, 2, 4.0))
        .expect("stage");
    let working = cache.read_working_sequence(0, 0, 3).expect("working read");
    assert_eq!(working.segments().len(), 1);
    assert_eq!(working.retained_positions(), 0..5);
    let plan = cache.plan_working_batch_read(&[3, 1]).expect("plan");
    let read = cache.read_batch(0, &plan).expect("read");
    assert_eq!(read.segments().len(), 1);
    assert_eq!(plan.key_slots(), 5);
    let before = snapshot(&cache);
    let commit = cache
        .prepare_staging_commit(&[3, 1], &[2, 1], &[1, 0])
        .expect("commit");
    cache.apply_staging_commit(commit);
    assert_eq!(
        snapshot(&cache),
        before,
        "an in-place commit moves no tensor data"
    );
}

/// In-place staging is refused for ring storage, whose proposals would overwrite visible slots.
#[test]
fn in_place_staging_requires_full_storage() {
    let config = CacheConfig::new(1, 8).expect("config");
    let spec = NativeKvCacheSpec::new(1, 1, 1, KvCacheStorage::SlidingRing, DType::F32)
        .expect("geometry")
        .with_sliding_window(NonZeroUsize::new(4).expect("window"))
        .with_staging_capacity(2)
        .with_in_place_staging();
    assert!(spec.in_place_staging());
    assert!(matches!(
        NativeKvCache::<TestBackend>::new(config, 1, spec, &device()),
        Err(TensorCacheError::InPlaceStagingRequiresFullStorage { .. })
    ));
    assert!(
        TensorCacheError::InPlaceStagingRequiresFullStorage {
            storage: "sliding ring"
        }
        .to_string()
        .contains("full storage")
    );
}

/// Invalid batched writes and stages fail before any tensor changes.
#[test]
fn invalid_batched_writes_are_rejected_atomically() {
    let mut cache = cache(2, 6, KvCacheStorage::Full, 1, 2, false);
    let before = snapshot(&cache);
    let fits = KvBatchAppend::right_padded(&[0, 0], &[2, 2], 2).expect("layout");
    let too_long = KvBatchAppend::right_padded(&[0, 5], &[2, 2], 2).expect("layout");
    assert!(matches!(
        cache.write_batch(0, &too_long, block(2, 2, 1.0), block(2, 2, 1.0)),
        Err(TensorCacheError::WriteExceedsCapacity { start: 5, .. })
    ));
    assert!(matches!(
        cache.write_batch(0, &fits, block(2, 3, 1.0), block(2, 3, 1.0)),
        Err(TensorCacheError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        cache.write_batch(1, &fits, block(2, 2, 1.0), block(2, 2, 1.0)),
        Err(TensorCacheError::LayerOutOfBounds { .. })
    ));
    let three_rows = KvBatchAppend::right_padded(&[0; 3], &[1; 3], 2).expect("layout");
    assert!(matches!(
        cache.write_batch(0, &three_rows, block(3, 2, 1.0), block(3, 2, 1.0)),
        Err(TensorCacheError::BatchSizeMismatch { .. })
    ));
    assert!(matches!(
        cache.stage_batch(0, &fits, block(2, 2, 1.0), block(2, 2, 1.0)),
        Err(TensorCacheError::TransactionNotActive)
    ));
    cache.begin_staging_from(&[0, 0]).expect("staging");
    assert!(matches!(
        cache.write_batch(0, &fits, block(2, 2, 1.0), block(2, 2, 1.0)),
        Err(TensorCacheError::TransactionActive { .. })
    ));
    let skipped = KvBatchAppend::right_padded(&[0, 1], &[1, 1], 1).expect("layout");
    assert!(matches!(
        cache.stage_batch(0, &skipped, block(2, 1, 1.0), block(2, 1, 1.0)),
        Err(TensorCacheError::TransactionCursorMismatch {
            sequence: 1,
            expected: 0,
            actual: 1,
            ..
        })
    ));
    let three = KvBatchAppend::right_padded(&[0, 0], &[3, 0], 3).expect("layout");
    assert!(matches!(
        cache.stage_batch(0, &three, block(2, 3, 1.0), block(2, 3, 1.0)),
        Err(TensorCacheError::StagingCapacityExceeded {
            requested: 3,
            capacity: 2
        })
    ));
    assert_eq!(snapshot(&cache), before);
    assert_eq!(cache.staged_lengths[0], [0, 0]);
    assert!(matches!(
        cache.read_working_sequence(0, 0, 1),
        Err(TensorCacheError::TransactionCursorMismatch {
            expected: 0,
            actual: 1,
            ..
        })
    ));
    assert!(
        TensorCacheError::TransactionCursorMismatch {
            sequence: 0,
            cursor: "committed length",
            expected: 2,
            actual: 3
        }
        .to_string()
        .contains("expects 2")
    );
}

/// Plans read only the group that built them, within the transaction they were built in.
#[test]
fn read_plans_are_bound_to_their_group_and_transaction() {
    let mut ring = cache(2, 8, KvCacheStorage::SlidingRing, 3, 2, false);
    let full = cache(2, 8, KvCacheStorage::Full, 1, 0, false);
    let twin = cache(2, 8, KvCacheStorage::Full, 1, 0, false);
    let plan = full.plan_batch_read(&[1, 2]).expect("plan");
    for other in [&ring, &twin] {
        // Even an identically configured group refuses another group's plan.
        assert!(matches!(
            other.read_batch(0, &plan),
            Err(TensorCacheError::BatchPlanMismatch { reason, .. }) if reason.contains("another cache group")
        ));
    }
    assert!(matches!(
        full.plan_batch_read(&[1]),
        Err(TensorCacheError::Metadata(_))
    ));
    assert!(matches!(
        full.plan_batch_read(&[1, 9]),
        Err(TensorCacheError::ReadExceedsCapacity { length: 9, .. })
    ));
    let empty = full.plan_batch_read(&[0, 0]).expect("empty plan");
    assert_eq!(empty.key_slots(), 0);
    assert!(
        full.read_batch(0, &empty)
            .expect("read")
            .materialize()
            .is_none()
    );
    ring.begin_staging_from(&[0, 0]).expect("staging");
    let stale = ring.plan_working_batch_read(&[0, 0]).expect("plan");
    let append = KvBatchAppend::right_padded(&[0, 0], &[1, 0], 1).expect("layout");
    ring.stage_batch(0, &append, block(2, 1, 1.0), block(2, 1, 1.0))
        .expect("stage");
    assert!(matches!(
        ring.read_batch(0, &stale),
        Err(TensorCacheError::BatchPlanMismatch { reason, .. }) if reason.contains("staged")
    ));
    assert!(matches!(
        ring.plan_working_batch_read(&[1, 0]),
        Err(TensorCacheError::TransactionCursorMismatch { .. })
    ));
    assert!(
        TensorCacheError::BatchPlanMismatch {
            layer: 3,
            reason: "x"
        }
        .to_string()
        .contains("layer 3")
    );
}

/// A working plan from an in-place group never reads a separate-staging group, and vice versa.
///
/// Both groups hold the same committed tokens and stage the same proposal.
/// The in-place plan is one committed segment `[0, C + P)`; applied to the
/// separate-staging group it would read committed slot `C`, which does not
/// hold the proposal.
#[test]
fn working_plans_do_not_cross_staging_modes() {
    let mut in_place = cache(1, 8, KvCacheStorage::Full, 1, 2, true);
    let mut separate = cache(1, 8, KvCacheStorage::Full, 1, 2, false);
    let prefill = KvBatchAppend::right_padded(&[0], &[2], 2).expect("layout");
    let proposal = KvBatchAppend::right_padded(&[2], &[1], 1).expect("layout");
    for group in [&mut in_place, &mut separate] {
        group
            .write_batch(0, &prefill, block(1, 2, 1.0), block(1, 2, 2.0))
            .expect("prefill");
        group.begin_staging_from(&[2]).expect("staging");
        group
            .stage_batch(0, &proposal, block(1, 1, 3.0), block(1, 1, 4.0))
            .expect("stage");
    }
    let in_place_plan = in_place.plan_working_batch_read(&[2]).expect("plan");
    let separate_plan = separate.plan_working_batch_read(&[2]).expect("plan");
    for (group, plan) in [(&separate, &in_place_plan), (&in_place, &separate_plan)] {
        assert!(matches!(
            group.read_batch(0, plan),
            Err(TensorCacheError::BatchPlanMismatch { reason, .. }) if reason.contains("another cache group")
        ));
    }
    // Each group's own plan reads the same chronological keys.
    let own = |group: &NativeKvCache<TestBackend>, plan| {
        chronological_row(&group.read_batch(0, plan).expect("read"), 0)
    };
    assert_eq!(
        own(&in_place, &in_place_plan),
        own(&separate, &separate_plan)
    );
}

/// A working plan needs every layer of the group to have staged the same proposal.
#[test]
fn working_plans_require_layers_staged_alike() {
    let config = CacheConfig::new(2, 8).expect("config");
    let spec = NativeKvCacheSpec::new(2, HEADS, DIM, KvCacheStorage::Full, DType::F32)
        .expect("geometry")
        .with_staging_capacity(2);
    let mut group = NativeKvCache::<TestBackend>::new(config, 2, spec, &device()).expect("group");
    group.begin_staging_from(&[0, 0]).expect("staging");
    let proposal = KvBatchAppend::right_padded(&[0, 0], &[1, 1], 1).expect("layout");
    group
        .stage_batch(0, &proposal, block(2, 1, 1.0), block(2, 1, 1.0))
        .expect("layer 0 only");
    assert!(matches!(
        group.plan_working_batch_read(&[0, 0]),
        Err(TensorCacheError::BatchPlanMismatch { layer: 1, reason }) if reason.contains("differ between layers")
    ));
    group
        .stage_batch(1, &proposal, block(2, 1, 2.0), block(2, 1, 2.0))
        .expect("layer 1");
    let plan = group
        .plan_working_batch_read(&[0, 0])
        .expect("aligned layers");
    assert_eq!(plan.retained_positions(), &[0..1, 0..1]);
}

/// A latent group refuses a key/value group's plan, and the reverse.
#[test]
fn plans_do_not_cross_kv_and_latent_groups() {
    let config = CacheConfig::new(2, 6).expect("config");
    let kv = NativeKvCache::<TestBackend>::new(
        config,
        2,
        NativeKvCacheSpec::new(1, 1, 3, KvCacheStorage::Full, DType::F32).expect("geometry"),
        &device(),
    )
    .expect("KV group");
    let latent = NativeKvCache::<TestBackend>::new(
        config,
        2,
        NativeKvCacheSpec::latent(1, 3, KvCacheStorage::Full, DType::F32).expect("geometry"),
        &device(),
    )
    .expect("latent group");
    let kv_plan = kv.plan_batch_read(&[1, 2]).expect("plan");
    let latent_plan = latent.plan_batch_read(&[1, 2]).expect("plan");
    assert!(matches!(
        latent.read_batch(0, &kv_plan),
        Err(TensorCacheError::BatchPlanMismatch { .. })
    ));
    assert!(matches!(
        kv.read_batch(0, &latent_plan),
        Err(TensorCacheError::BatchPlanMismatch { .. })
    ));
}

/// A working plan from an earlier transaction is refused even when the staged lengths agree.
#[test]
fn working_plans_expire_with_their_transaction() {
    for in_place in [false, true] {
        let mut group = cache(2, 8, KvCacheStorage::Full, 1, 2, in_place);
        let proposal = KvBatchAppend::right_padded(&[0, 0], &[1, 1], 1).expect("layout");
        group.begin_staging_from(&[0, 0]).expect("staging");
        group
            .stage_batch(0, &proposal, block(2, 1, 1.0), block(2, 1, 1.0))
            .expect("stage");
        let first = group.plan_working_batch_read(&[0, 0]).expect("plan");
        group.rollback_staging();
        assert!(matches!(
            group.read_batch(0, &first),
            Err(TensorCacheError::BatchPlanMismatch { reason, .. }) if reason.contains("another transaction")
        ));
        // The next transaction stages the same lengths with different values.
        group.begin_staging_from(&[0, 0]).expect("staging");
        group
            .stage_batch(0, &proposal, block(2, 1, 5.0), block(2, 1, 5.0))
            .expect("stage");
        assert!(matches!(
            group.read_batch(0, &first),
            Err(TensorCacheError::BatchPlanMismatch { reason, .. }) if reason.contains("another transaction")
        ));
        let current = group.plan_working_batch_read(&[0, 0]).expect("plan");
        assert!(
            group.read_batch(0, &current).is_ok(),
            "in place: {in_place}"
        );
    }
}

/// Non-finite bytes in an invalid slot poison raw values but not `masked_values`.
///
/// Row 1 stages a NaN proposal in place and rolls it back; the slot keeps the
/// NaN but lies beyond the row's length. A batched read over both rows
/// exposes it as an invalid key. Masked scores neutralize the NaN key, yet
/// `0 · NaN` in the value product still yields NaN unless the values are
/// masked; with masked values the row equals its per-row attention.
#[test]
fn masked_values_neutralize_non_finite_invalid_slots() {
    let mut group = cache(2, 8, KvCacheStorage::Full, 1, 2, true);
    let prefill = KvBatchAppend::right_padded(&[0, 0], &[3, 1], 3).expect("layout");
    let keys = block(2, 3, 0.0).mul_scalar(1e-3);
    let values = block(2, 3, 0.0).mul_scalar(2e-3);
    group
        .write_batch(0, &prefill, keys, values)
        .expect("prefill");
    group.begin_staging_from(&[3, 1]).expect("staging");
    let draft = KvBatchAppend::right_padded(&[3, 1], &[0, 1], 1).expect("layout");
    let nan = Tensor::<TestBackend, 4>::full([2, HEADS, 1, DIM], f32::NAN, &device());
    group
        .stage_batch(0, &draft, nan.clone(), nan)
        .expect("stage NaN draft");
    group.rollback_staging();

    let segment = group
        .read_batch(0, &group.plan_batch_read(&[3, 1]).expect("plan"))
        .expect("read")
        .materialize()
        .expect("keys");
    let positions =
        Tensor::<TestBackend, 2, Int>::from_ints(TensorData::new(vec![2, 0], [2, 1]), &device());
    let excluded = segment
        .exclusion_mask(&positions, HEADS, None)
        .expect("mask");
    let query = block(2, 1, 0.0).mul_scalar(3e-3);
    let raw = host(&attend(
        query.clone(),
        segment.keys().clone(),
        segment.values().clone(),
        excluded.clone(),
    ));
    assert!(raw.iter().any(|value| value.is_nan()), "the hazard is real");
    let masked = segment.masked_values();
    assert!(host(&masked).iter().all(|value| value.is_finite()));
    let safe = host(&attend(
        query.clone(),
        segment.keys().clone(),
        masked,
        excluded,
    ));
    assert!(safe.iter().all(|value| value.is_finite()));

    // Row 1 attends to its single committed key only: the output is that value.
    let history = group.read_sequence(0, 1, 1).expect("row read");
    let (_, row_values) = history.materialize().expect("one key");
    let per_row = host(&row_values);
    assert_eq!(&safe[HEADS * DIM..], per_row.as_slice());

    // Fully valid segments are returned without masking.
    let full = group
        .read_batch(0, &group.plan_batch_read(&[1, 1]).expect("plan"))
        .expect("read")
        .materialize()
        .expect("keys");
    assert!(full.key_valid().is_none());
    assert_eq!(host(&full.masked_values()), host(full.values()));
}

/// Windows wider than any representable distance exclude only causal and padding links.
///
/// `key + window` used to wrap on the backend's integer dtype for windows near
/// its range (for example `2^40` on 32-bit integers), excluding every key. The
/// mask now compares `query - key` with the window, so every width up to
/// `usize::MAX` matches the brute-force rule.
#[test]
fn exclusion_mask_handles_windows_beyond_the_integer_range() {
    let mut ring = cache(1, 10, KvCacheStorage::Full, 4, 0, false);
    let append = KvBatchAppend::right_padded(&[0], &[3], 3).expect("layout");
    ring.write_batch(0, &append, block(1, 3, 1.0), block(1, 3, 1.0))
        .expect("prefill");
    let segment = ring
        .read_batch(0, &ring.plan_batch_read(&[3]).expect("plan"))
        .expect("read")
        .materialize()
        .expect("keys");
    let query_positions =
        Tensor::<TestBackend, 2, Int>::from_ints(TensorData::new(vec![0, 1, 2], [1, 3]), &device());
    let unrestricted = host_bool(
        &segment
            .exclusion_mask(&query_positions, 1, None)
            .expect("mask"),
    );
    for width in [
        3_usize,
        i32::MAX as usize,
        i32::MAX as usize + 1,
        1 << 40,
        usize::MAX,
    ] {
        let mask = segment
            .exclusion_mask(&query_positions, 1, NonZeroUsize::new(width))
            .expect("mask");
        assert_eq!(host_bool(&mask), unrestricted, "window {width}");
    }
    let narrow = host_bool(
        &segment
            .exclusion_mask(&query_positions, 1, NonZeroUsize::new(1))
            .expect("mask"),
    );
    // A one-token window keeps only the diagonal: query q sees key q.
    let keys = host_int(segment.key_positions());
    for (query, query_row) in narrow.chunks(keys.len()).enumerate() {
        for (&excluded, &key) in query_row.iter().zip(&keys) {
            assert_eq!(excluded, usize::try_from(key) != Ok(query));
        }
    }
}

/// Exclusion masks equal a brute-force causal, window, and padding rule.
#[test]
fn exclusion_mask_matches_brute_force_rule() {
    let mut ring = cache(2, 10, KvCacheStorage::SlidingRing, 4, 0, false);
    let lengths = [7, 2];
    let append = KvBatchAppend::right_padded(&[0, 0], &lengths, 7).expect("layout");
    ring.write_batch(0, &append, block(2, 7, 1.0), block(2, 7, 1.0))
        .expect("prefill");
    let plan = ring.plan_batch_read(&lengths).expect("plan");
    let segment = ring
        .read_batch(0, &plan)
        .expect("read")
        .materialize()
        .expect("keys");
    let queries = [[5_i32, 6], [0, 1]];
    let query_positions = Tensor::<TestBackend, 2, Int>::from_ints(
        TensorData::new(queries.concat(), [2, 2]),
        &device(),
    );
    let mask = segment
        .exclusion_mask(&query_positions, 2, NonZeroUsize::new(3))
        .expect("mask");
    assert_eq!(mask.dims(), [2, 2, 2, 4]);
    let mask = host_bool(&mask);
    let positions = host_int(segment.key_positions());
    let valid = host_bool(segment.key_valid().expect("row 1 has empty slots"));
    for (row, row_queries) in queries.iter().enumerate() {
        for head in 0..2 {
            for (query, &query_position) in row_queries.iter().enumerate() {
                for slot in 0..4 {
                    let key = positions[row * 4 + slot];
                    let expected =
                        !valid[row * 4 + slot] || key > query_position || key + 3 <= query_position;
                    let index = ((row * 2 + head) * 2 + query) * 4 + slot;
                    assert_eq!(mask[index], expected, "row {row} query {query} slot {slot}");
                }
            }
        }
    }
    let wrong = Tensor::<TestBackend, 2, Int>::zeros([3, 1], &device());
    assert!(matches!(
        segment.exclusion_mask(&wrong, 2, None),
        Err(TensorCacheError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        segment.exclusion_mask(&query_positions, 0, None),
        Err(TensorCacheError::ZeroDimension { .. })
    ));
}

/// Naive softmax attention for one query block over keys with an exclusion mask.
fn attend(
    query: Tensor<TestBackend, 4>,
    keys: Tensor<TestBackend, 4>,
    values: Tensor<TestBackend, 4>,
    excluded: Tensor<TestBackend, 4, Bool>,
) -> Tensor<TestBackend, 4> {
    let scale = 1.0 / (DIM as f32).sqrt();
    let scores = query
        .matmul(keys.swap_dims(2, 3))
        .mul_scalar(scale)
        .mask_fill(excluded, f32::NEG_INFINITY);
    burn::tensor::activation::softmax(scores, 3).matmul(values)
}

/// Batched write-ahead plus a masked slot-order read reproduces per-row attention.
#[test]
fn batched_attention_matches_per_row_attention_within_f32_rounding() {
    for (storage, window) in STORAGES {
        let lengths = [5, 2, 7];
        let mut cache = cache(3, 10, storage, window, 0, false);
        let prefill = KvBatchAppend::right_padded(&[0; 3], &lengths, 7).expect("layout");
        cache
            .write_batch(
                0,
                &prefill,
                block(3, 7, 0.0).mul_scalar(1e-3),
                block(3, 7, 0.0).mul_scalar(2e-3),
            )
            .expect("prefill");
        // One decode token per row, written ahead of attention.
        let append = KvBatchAppend::right_padded(&lengths, &[1; 3], 1).expect("layout");
        let new_keys = block(3, 1, 0.0).mul_scalar(3e-3);
        let new_values = block(3, 1, 0.0).mul_scalar(4e-3);
        let query = block(3, 1, 0.0).mul_scalar(5e-3);
        let local = (storage != KvCacheStorage::Full).then_some(window);
        let mut reference = Vec::new();
        for (row, &length) in lengths.iter().enumerate() {
            let history = cache.read_sequence(0, row, length).expect("history");
            let (past_keys, past_values) = history.materialize().expect("non-empty");
            // A ring window of W keeps the newest W - 1 history keys plus the new token.
            let keep = local.map_or(length, |width| length.min(width - 1));
            let time = past_keys.dims()[2];
            let past_keys = past_keys.slice([0..1, 0..HEADS, time - keep..time, 0..DIM]);
            let past_values = past_values.slice([0..1, 0..HEADS, time - keep..time, 0..DIM]);
            let keys = Tensor::cat(vec![past_keys, row_input(&new_keys, row, 0..1)], 2);
            let values = Tensor::cat(vec![past_values, row_input(&new_values, row, 0..1)], 2);
            let excluded = Tensor::<TestBackend, 4, Int>::zeros([1, HEADS, 1, keep + 1], &device())
                .equal_elem(1);
            reference.extend(host(&attend(
                row_input(&query, row, 0..1),
                keys,
                values,
                excluded,
            )));
        }
        cache
            .write_batch(0, &append, new_keys, new_values)
            .expect("write ahead");
        let plan = cache.plan_batch_read(&append.ends()).expect("plan");
        let segment = cache
            .read_batch(0, &plan)
            .expect("read")
            .materialize()
            .expect("keys");
        let query_positions = Tensor::<TestBackend, 2, Int>::from_ints(
            TensorData::new(
                lengths
                    .iter()
                    .map(|&length| i32::try_from(length).expect("small"))
                    .collect::<Vec<_>>(),
                [3, 1],
            ),
            &device(),
        );
        let excluded = segment
            .exclusion_mask(&query_positions, HEADS, local.and_then(NonZeroUsize::new))
            .expect("mask");
        let batched = host(&attend(
            query,
            segment.keys().clone(),
            segment.values().clone(),
            excluded,
        ));
        assert_eq!(batched.len(), reference.len());
        for (actual, expected) in batched.iter().zip(&reference) {
            assert!(
                (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                "{storage:?}: {actual} vs {expected}"
            );
        }
    }
}

/// Times one decode step per layer along three paths.
///
/// 1. *per row*: read and materialize each row's history, concatenate its
///    new token, attend, and write the token row by row (today's families);
/// 2. *batched + cat*: one batched history read, joined with the step's
///    token segment by concatenation (`with_segment` + `materialize`), one
///    batched attention, one batched write — the `kv_read_plan` +
///    `step_token_segment` path;
/// 3. *batched write-ahead*: one batched write ahead of attention and one
///    zero-copy view `[0, L + 1)` — the `write_kv_ahead` path.
///
/// Path 2 isolates batching from removing the history concatenation, which
/// is what path 3 adds. All paths run the same single-query softmax
/// attention with an exclusion mask built from key positions. The paths run
/// one after another, so later paths see histories up to `2 * steps` tokens
/// longer (at most 6% here). Run with
/// `cargo test --release --lib batched_decode_step_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing measurement; run explicitly in release mode"]
fn batched_decode_step_timing() {
    use std::time::Instant;
    let rows = 8;
    let heads = 8;
    let width = 64;
    let length = 1024;
    let steps = 32;
    let config = CacheConfig::new(rows, length + 3 * steps).expect("config");
    let spec =
        NativeKvCacheSpec::new(1, heads, width, KvCacheStorage::Full, DType::F32).expect("spec");
    let mut cache =
        NativeKvCache::<TestBackend>::new(config, rows, spec, &device()).expect("cache");
    let fill =
        Tensor::<TestBackend, 4>::ones([rows, heads, length, width], &device()).mul_scalar(0.01);
    let prefill = KvBatchAppend::right_padded(&[0; 8], &[length; 8], length).expect("layout");
    cache
        .write_batch(0, &prefill, fill.clone(), fill)
        .expect("prefill");
    let token = Tensor::<TestBackend, 4>::ones([rows, heads, 1, width], &device()).mul_scalar(0.02);
    let query = token.clone();

    // Per-row reference: read and materialize history, concatenate the new
    // token, attend, then publish the token row by row.
    let started = Instant::now();
    for step in 0..steps {
        let mut outputs = Vec::with_capacity(rows);
        for row in 0..rows {
            let history = cache.read_sequence(0, row, length + step).expect("read");
            let (past_keys, past_values) = history.materialize().expect("history");
            let new = token
                .clone()
                .slice([row..row + 1, 0..heads, 0..1, 0..width]);
            let keys = Tensor::cat(vec![past_keys, new.clone()], 2);
            let values = Tensor::cat(vec![past_values, new.clone()], 2);
            let time = keys.dims()[2];
            let excluded =
                Tensor::<TestBackend, 4, Int>::zeros([1, heads, 1, time], &device()).equal_elem(1);
            let row_query = query
                .clone()
                .slice([row..row + 1, 0..heads, 0..1, 0..width]);
            outputs.push(attend(row_query, keys, values, excluded));
            cache
                .write_sequence(0, row, length + step, new.clone(), new)
                .expect("write");
        }
        assert!(Tensor::cat(outputs, 0).sum().into_scalar() > 0.0);
    }
    let per_row = started.elapsed() / u32::try_from(steps).expect("small");

    // Batched with concatenation: read the history, append the token segment.
    let started = Instant::now();
    for step in steps..2 * steps {
        let start = length + step;
        let plan = cache.plan_batch_read(&[start; 8]).expect("plan");
        let position = i32::try_from(start).expect("small");
        let positions = Tensor::<TestBackend, 2, Int>::from_ints(
            TensorData::new(vec![position; 8], [8, 1]),
            &device(),
        );
        let new_tokens =
            KvBatchSegment::from_parts(token.clone(), token.clone(), positions.clone(), None);
        let segment = cache
            .read_batch(0, &plan)
            .expect("read")
            .with_segment(new_tokens)
            .expect("join")
            .materialize()
            .expect("keys");
        let excluded = segment
            .exclusion_mask(&positions, heads, None)
            .expect("mask");
        let output = attend(
            query.clone(),
            segment.keys().clone(),
            segment.values().clone(),
            excluded,
        );
        assert!(output.sum().into_scalar() > 0.0);
        let append = KvBatchAppend::right_padded(&[start; 8], &[1; 8], 1).expect("layout");
        cache
            .write_batch(0, &append, token.clone(), token.clone())
            .expect("publish");
    }
    let concatenated = started.elapsed() / u32::try_from(steps).expect("small");

    // Batched write-ahead: write once, plan once, attend over one zero-copy view.
    let started = Instant::now();
    for step in 2 * steps..3 * steps {
        let append = KvBatchAppend::right_padded(&[length + step; 8], &[1; 8], 1).expect("layout");
        cache
            .write_batch(0, &append, token.clone(), token.clone())
            .expect("write ahead");
        let plan = cache.plan_batch_read(&append.ends()).expect("plan");
        let segment = cache
            .read_batch(0, &plan)
            .expect("read")
            .materialize()
            .expect("keys");
        let positions = Tensor::<TestBackend, 2, Int>::from_ints(
            TensorData::new(
                vec![i32::try_from(length + step).expect("small"); 8],
                [8, 1],
            ),
            &device(),
        );
        let excluded = segment
            .exclusion_mask(&positions, heads, None)
            .expect("mask");
        let output = attend(
            query.clone(),
            segment.keys().clone(),
            segment.values().clone(),
            excluded,
        );
        assert!(output.sum().into_scalar() > 0.0);
    }
    let written_ahead = started.elapsed() / u32::try_from(steps).expect("small");
    println!(
        "decode step with attention, 8 rows x ~1024 tokens x 8 heads x 64: \
         per-row {per_row:?}, batched + cat {concatenated:?}, batched write-ahead {written_ahead:?}"
    );
}

/// Appending a segment joins validity explicitly and requires the read's batch size.
///
/// A padded history carries an explicit validity mask while a step's new
/// tokens are all valid implicitly; concatenation must materialize `true`
/// for the implicit part so every key keeps its own flag. A segment with a
/// different row count is rejected before it is joined.
#[test]
fn appended_segments_join_validity_and_require_the_batch_size() {
    let mut cache = cache(2, 4, KvCacheStorage::Full, 0, 0, false);
    let history = block(2, 2, 1.0);
    let prefill = KvBatchAppend::right_padded(&[0, 0], &[2, 1], 2).expect("layout");
    cache
        .write_batch(0, &prefill, history.clone(), history)
        .expect("prefill");
    let plan = cache.plan_batch_read(&[2, 1]).expect("plan");
    let read = cache.read_batch(0, &plan).expect("read");
    let positions = Tensor::<TestBackend, 2, Int>::from_ints([[2], [1]], &device());
    let token = block(2, 1, 2.0);
    let new_tokens = KvBatchSegment::from_parts(token.clone(), token.clone(), positions, None);
    let joined = read
        .clone()
        .with_segment(new_tokens)
        .expect("same batch")
        .materialize()
        .expect("keys");
    assert_eq!(host_int(joined.key_positions()), [0, 1, 2, 0, 1, 1]);
    assert_eq!(
        host_bool(joined.key_valid().expect("explicit validity")),
        [true, true, true, true, false, true]
    );

    let one_row = KvBatchSegment::from_parts(
        block(1, 1, 3.0),
        block(1, 1, 3.0),
        Tensor::<TestBackend, 2, Int>::from_ints([[3]], &device()),
        None,
    );
    assert!(matches!(
        read.with_segment(one_row),
        Err(TensorCacheError::BatchSizeMismatch {
            configured: 2,
            requested: 1
        })
    ));
}

/// Key positions and slot counts beyond the portable index ranges are named errors.
#[test]
fn index_conversions_reject_out_of_range_values() {
    let beyond = i32::MAX as usize + 1;
    assert!(matches!(
        key_positions(&[0, 5, beyond]),
        Err(TensorCacheError::TensorIndexOverflow {
            axis: "key position",
            index,
        }) if index == beyond
    ));
    assert_eq!(key_positions(&[0, 5]).expect("in range"), [0, 5]);
    assert!(matches!(
        to_i64(usize::MAX),
        Err(TensorCacheError::TensorIndexOverflow {
            axis: "key slot",
            index: usize::MAX,
        })
    ));
    assert_eq!(to_i64(7).expect("in range"), 7);
}

/// Staging defenses reject states that public transitions never create.
///
/// Each case breaks one private invariant directly: in-place staging without
/// a committed baseline, a working plan whose staged lengths outgrow the
/// capacity, and a ring whose window was cleared. The checks must report a
/// contextual error rather than write or read outside the planned slots. A
/// transaction begun without a baseline (unit tests only) trusts any
/// proposal start.
#[test]
fn staging_defenses_reject_broken_private_invariants() {
    let mut in_place = cache(1, 4, KvCacheStorage::Full, 0, 2, true);
    in_place.staging_active = true;
    let append = KvBatchAppend::right_padded(&[0], &[1], 1).expect("layout");
    assert!(matches!(
        in_place.prepare_batch_stage(0, &append, vec![block(1, 1, 1.0), block(1, 1, 2.0)]),
        Err(TensorCacheError::TransactionNotActive)
    ));

    let mut overgrown = cache(1, 4, KvCacheStorage::Full, 0, 2, false);
    overgrown.begin_staging_from(&[3]).expect("begin staging");
    overgrown.staged_lengths[0][0] = 2;
    assert!(matches!(
        overgrown.plan_working_batch_read(&[3]),
        Err(TensorCacheError::WriteExceedsCapacity {
            start: 3,
            end: 5,
            capacity: 4
        })
    ));

    let mut ring = cache(1, 6, KvCacheStorage::SlidingRing, 2, 0, false);
    ring.window = None;
    let append = KvBatchAppend::right_padded(&[0], &[1], 1).expect("layout");
    assert!(matches!(
        ring.write_batch(0, &append, block(1, 1, 1.0), block(1, 1, 2.0)),
        Err(TensorCacheError::SlidingWindowRequired {
            storage: "sliding ring"
        })
    ));

    let mut unanchored = cache(2, 4, KvCacheStorage::Full, 0, 2, false);
    unanchored.begin_staging().expect("begin staging");
    unanchored
        .check_proposal_start(0, 1, 3)
        .expect("no baseline to compare against");
}
