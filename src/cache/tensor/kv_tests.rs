//! Internal staging-allocation defenses and multi-layer KV commit boundaries.

use burn::backend::Flex;

use super::*;
use core::num::NonZeroUsize;

/// Deterministic host backend for cache lifecycle checks.
type TestBackend = Flex<f32, i32>;

/// Allocates two scalar-head layers with one batch row and two proposal slots.
fn cache() -> NativeKvCache<TestBackend> {
    let config = CacheConfig::new(1, 4).expect("valid test config");
    let spec = NativeKvCacheSpec::new(2, 1, 1, KvCacheStorage::Full, DType::F32)
        .expect("valid test geometry")
        .with_staging_capacity(2);
    NativeKvCache::new(config, 1, spec, &Default::default()).expect("test cache allocation")
}

/// Creates one nonzero key or value with scalar head geometry.
fn token(value: f32) -> Tensor<TestBackend, 4> {
    Tensor::full([1, 1, 1, 1], value, &Default::default())
}

/// Reads all values without tolerance because the fixture only assigns exact integers.
fn values(tensor: &Tensor<TestBackend, 4>) -> Vec<f32> {
    tensor
        .clone()
        .into_data()
        .to_vec()
        .expect("F32 test tensor")
}

/// A missing staging plane cannot partially change the surviving plane.
#[test]
fn incomplete_staging_allocation_rejects_writes_before_mutation() {
    for remove_keys in [false, true] {
        let mut cache = cache();
        cache.begin_staging().expect("begin staging");
        // Public constructors always allocate every plane; deliberately break
        // the private invariant to exercise the recoverable defensive path.
        let planes = cache.layers[0].staged.as_mut().expect("staging planes");
        planes.remove(usize::from(!remove_keys));
        assert!(matches!(
            cache.stage_sequence(0, 0, token(7.0), token(9.0)),
            Err(TensorCacheError::StagingDisabled)
        ));
        let surviving = &cache.layers[0].staged.as_ref().expect("one plane remains")[0];
        assert_eq!(values(surviving), [0.0, 0.0]);
        assert_eq!(cache.staged_length(0, 0).expect("valid row"), 0);
    }
}

/// Invalid staging in the last layer fails planning while every committed layer remains intact.
#[test]
fn incomplete_staging_allocation_rejects_commit_before_publication() {
    for remove_keys in [false, true] {
        let mut cache = cache();
        cache.begin_staging().expect("begin staging");
        for layer in 0..2 {
            cache
                .stage_sequence(layer, 0, token(7.0), token(9.0))
                .expect("stage layer");
        }
        let planes = cache.layers[1].staged.as_mut().expect("staging planes");
        planes.remove(usize::from(!remove_keys));
        assert!(matches!(
            cache.prepare_staging_commit(&[0], &[1], &[1]),
            Err(TensorCacheError::StagingDisabled)
        ));
        for layer in 0..2 {
            assert_eq!(values(cache.keys(layer).expect("valid layer")), [0.0; 4]);
            assert_eq!(values(cache.values(layer).expect("valid layer")), [0.0; 4]);
            assert_eq!(cache.staged_length(layer, 0).expect("valid row"), 1);
        }
        assert!(cache.has_active_staging());
    }
}

/// Allocates a one-row scalar-head cache with explicit storage, group window, and staging.
fn windowed_cache(
    config: CacheConfig,
    storage: KvCacheStorage,
    window: Option<usize>,
    staging: usize,
) -> TensorCacheResult<NativeKvCache<TestBackend>> {
    let mut spec = NativeKvCacheSpec::new(1, 1, 1, storage, DType::F32)
        .expect("valid test geometry")
        .with_staging_capacity(staging);
    if let Some(window) = window {
        spec = spec.with_sliding_window(NonZeroUsize::new(window).expect("non-zero window"));
    }
    NativeKvCache::new(config, config.sequence_count(), spec, &Default::default())
}

/// Writes positions `0..length` with value `position + 1` into row zero.
fn write_positions(cache: &mut NativeKvCache<TestBackend>, length: usize) {
    let values: Vec<f32> = (0..length).map(|position| position as f32 + 1.0).collect();
    let tensor = Tensor::<TestBackend, 4>::from_data(
        burn::tensor::TensorData::new(values, [1, 1, length, 1]),
        &Default::default(),
    );
    cache
        .write_sequence(0, 0, 0, tensor.clone(), tensor)
        .expect("valid write");
}

/// Reads the retained chronological keys of row zero.
fn retained_keys(cache: &NativeKvCache<TestBackend>, length: usize) -> Vec<f32> {
    cache
        .read_sequence(0, 0, length)
        .expect("valid read")
        .materialize()
        .map_or_else(Vec::new, |(keys, _)| values(&keys))
}

/// Two ring groups with different windows coexist under one config and wrap independently.
#[test]
fn group_windows_override_the_shared_config_window() {
    let config = CacheConfig::new(1, 12)
        .expect("valid config")
        .with_sliding_window(4)
        .expect("valid shared window");
    let mut shared = windowed_cache(config, KvCacheStorage::SlidingRing, None, 0).expect("cache");
    let mut narrow =
        windowed_cache(config, KvCacheStorage::SlidingRing, Some(2), 0).expect("cache");
    let mut wide =
        windowed_cache(config, KvCacheStorage::MirroredSlidingRing, Some(6), 0).expect("cache");
    assert_eq!(shared.committed_time_slots(), 4);
    assert_eq!(narrow.committed_time_slots(), 2);
    assert_eq!(wide.committed_time_slots(), 12);
    for cache in [&mut shared, &mut narrow, &mut wide] {
        write_positions(cache, 9);
    }
    assert_eq!(retained_keys(&shared, 9), [6.0, 7.0, 8.0, 9.0]);
    assert_eq!(retained_keys(&narrow, 9), [8.0, 9.0]);
    assert_eq!(retained_keys(&wide, 9), [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_eq!(
        narrow
            .read_sequence(0, 0, 9)
            .expect("read")
            .retained_positions(),
        7..9
    );

    // Rewind safety follows each group's own width: 9 -> 5 exposes [3, 5) for
    // the narrow group (overwritten) but is rejected for all wrapped rings.
    for cache in [&shared, &narrow, &wide] {
        assert!(matches!(
            cache.validate_truncate(&[9], &[5]),
            Err(TensorCacheError::UnsafeWrappedTruncate { sequence: 0, .. })
        ));
        assert!(cache.validate_truncate(&[9], &[0]).is_ok());
    }
    // A group whose window covers the whole history can still rewind.
    let roomy = windowed_cache(config, KvCacheStorage::SlidingRing, Some(10), 0).expect("cache");
    assert!(roomy.validate_truncate(&[9], &[5]).is_ok());
}

/// Group windows drive speculative working reads and staged commits.
#[test]
fn group_window_bounds_working_reads_and_commits() {
    let config = CacheConfig::new(1, 12).expect("valid config");
    let mut cache = windowed_cache(config, KvCacheStorage::SlidingRing, Some(3), 2).expect("cache");
    write_positions(&mut cache, 4);
    cache.begin_staging().expect("staging");
    cache
        .stage_sequence(0, 0, token(10.0), token(10.0))
        .expect("first proposal");
    let working = cache.read_working_sequence(0, 0, 4).expect("working read");
    assert_eq!(working.retained_positions(), 2..5);
    let (keys, _) = working.materialize().expect("visible keys");
    assert_eq!(values(&keys), [3.0, 4.0, 10.0]);

    let plan = cache
        .prepare_staging_commit(&[4], &[1], &[1])
        .expect("valid commit");
    cache.apply_staging_commit(plan);
    assert_eq!(retained_keys(&cache, 5), [3.0, 4.0, 10.0]);
}

/// Rejects windows on full storage, windows beyond capacity, and rings without any window.
#[test]
fn invalid_group_windows_fail_before_allocation() {
    let config = CacheConfig::new(1, 8).expect("valid config");
    for (storage, window, reason) in [
        (KvCacheStorage::Full, Some(4), "full storage"),
        (KvCacheStorage::SlidingRing, Some(9), "exceeds"),
        (KvCacheStorage::MirroredSlidingRing, Some(9), "exceeds"),
    ] {
        match windowed_cache(config, storage, window, 0) {
            Err(TensorCacheError::InvalidSlidingWindow {
                window: 4 | 9,
                capacity: 8,
                reason: actual,
                ..
            }) => assert!(actual.contains(reason), "{actual}"),
            other => panic!("expected invalid window for {storage:?}, got {other:?}"),
        }
    }
    assert!(matches!(
        windowed_cache(config, KvCacheStorage::SlidingRing, None, 0),
        Err(TensorCacheError::SlidingWindowRequired { .. })
    ));
    let message = TensorCacheError::InvalidSlidingWindow {
        storage: "sliding ring",
        window: 9,
        capacity: 8,
        reason: "the window exceeds the logical capacity",
    }
    .to_string();
    assert!(message.contains("sliding window 9"), "{message}");
}

/// The spec reports exact committed slots and the effective window per storage mode.
#[test]
fn spec_reports_exact_committed_slots_per_storage_mode() {
    let config = CacheConfig::new(2, 10)
        .expect("valid config")
        .with_sliding_window(4)
        .expect("valid window");
    let spec = |storage| NativeKvCacheSpec::new(1, 1, 1, storage, DType::F32).expect("spec");
    assert_eq!(
        spec(KvCacheStorage::Full).committed_time_slots(config).ok(),
        Some(10)
    );
    assert_eq!(spec(KvCacheStorage::Full).effective_window(config), None);
    assert_eq!(
        spec(KvCacheStorage::SlidingRing)
            .committed_time_slots(config)
            .ok(),
        Some(4)
    );
    let mirrored = spec(KvCacheStorage::MirroredSlidingRing)
        .with_sliding_window(NonZeroUsize::new(3).expect("non-zero"));
    assert_eq!(mirrored.sliding_window().map(NonZeroUsize::get), Some(3));
    assert_eq!(
        mirrored.effective_window(config).map(NonZeroUsize::get),
        Some(3)
    );
    assert_eq!(mirrored.committed_time_slots(config).ok(), Some(6));
}

/// Allocates a two-row latent group of the given storage, window, and staging policy.
fn latent_cache(
    storage: KvCacheStorage,
    staging: usize,
    in_place: bool,
) -> NativeKvCache<TestBackend> {
    let config = CacheConfig::new(2, 8).expect("valid config");
    let mut spec = NativeKvCacheSpec::latent(1, 3, storage, DType::F32)
        .expect("latent geometry")
        .with_staging_capacity(staging);
    if storage != KvCacheStorage::Full {
        spec = spec.with_sliding_window(NonZeroUsize::new(2).expect("window"));
    }
    if in_place {
        spec = spec.with_in_place_staging();
    }
    NativeKvCache::new(config, 2, spec, &Default::default()).expect("latent allocation")
}

/// Builds a `[rows, tokens, 3]` latent whose entries encode their position.
fn latent(rows: usize, tokens: usize, tag: f32) -> Tensor<TestBackend, 3> {
    let data = (0..rows * tokens * 3)
        .map(|index| tag + index as f32)
        .collect::<Vec<_>>();
    Tensor::from_data(
        burn::tensor::TensorData::new(data, [rows, tokens, 3]),
        &Default::default(),
    )
}

/// A latent group stores one `[batch, 1, time, width]` plane and rejects key/value inputs.
#[test]
fn latent_groups_store_one_plane_and_reject_pairs() {
    let spec = NativeKvCacheSpec::latent(2, 5, KvCacheStorage::Full, DType::F32).expect("spec");
    assert!(spec.is_latent());
    assert_eq!(spec.plane_count(), 1);
    assert_eq!((spec.key_value_heads(), spec.head_dimension()), (1, 5));
    assert!(
        !NativeKvCacheSpec::new(1, 1, 1, KvCacheStorage::Full, DType::F32)
            .expect("spec")
            .is_latent()
    );
    assert!(matches!(
        NativeKvCacheSpec::latent(1, 0, KvCacheStorage::Full, DType::F32),
        Err(TensorCacheError::ZeroDimension { .. })
    ));

    let mut cache = latent_cache(KvCacheStorage::Full, 0, false);
    assert_eq!(cache.layers[0].planes.len(), 1);
    assert_eq!(cache.keys(0).expect("plane").dims(), [2, 1, 8, 3]);
    assert!(matches!(
        cache.write_sequence(0, 0, 0, token(1.0), token(1.0)),
        Err(TensorCacheError::PlaneKindMismatch { expected }) if expected.contains("latent")
    ));
    let mut pair = cache_kv();
    assert!(matches!(
        pair.write_latent_sequence(0, 0, 0, latent(1, 1, 0.0)),
        Err(TensorCacheError::PlaneKindMismatch { expected }) if expected.contains("key/value")
    ));

    cache
        .write_latent_sequence(0, 1, 2, latent(1, 3, 10.0))
        .expect("latent write");
    let read = cache.read_sequence(0, 1, 5).expect("read");
    let (history_keys, history_values) = read.materialize().expect("history");
    assert_eq!(values(&history_keys), values(&history_values));
    let committed = read.materialize_latent().expect("latent history");
    assert_eq!(committed.dims(), [1, 5, 3]);
    let tail: Vec<f32> = values(&committed.unsqueeze_dim::<4>(1))[6..].to_vec();
    assert_eq!(tail, values(&latent(1, 3, 10.0).unsqueeze_dim::<4>(1)));
    assert!(
        cache
            .read_sequence(0, 0, 0)
            .expect("empty")
            .materialize_latent()
            .is_none()
    );
}

/// Latent rings, batched writes, and staging behave exactly like their KV counterparts.
#[test]
fn latent_groups_share_ring_batch_and_staging_paths() {
    for (storage, in_place) in [
        (KvCacheStorage::Full, false),
        (KvCacheStorage::Full, true),
        (KvCacheStorage::SlidingRing, false),
        (KvCacheStorage::MirroredSlidingRing, false),
    ] {
        let mut batched = latent_cache(storage, 2, in_place);
        let mut per_row = latent_cache(storage, 2, in_place);
        let append = KvBatchAppend::right_padded(&[0, 0], &[3, 1], 3).expect("layout");
        batched
            .write_latent_batch(0, &append, latent(2, 3, 1.0))
            .expect("batched latent write");
        per_row
            .write_latent_sequence(0, 0, 0, latent(2, 3, 1.0).slice([0..1, 0..3, 0..3]))
            .expect("row 0");
        per_row
            .write_latent_sequence(0, 1, 0, latent(2, 3, 1.0).slice([1..2, 0..1, 0..3]))
            .expect("row 1");
        for cache in [&mut batched, &mut per_row] {
            cache.begin_staging_from(&[3, 1]).expect("staging");
        }
        let proposal = KvBatchAppend::right_padded(&[3, 1], &[1, 2], 2).expect("layout");
        batched
            .stage_latent_batch(0, &proposal, latent(2, 2, 50.0))
            .expect("batched latent stage");
        per_row
            .stage_latent_sequence(0, 0, latent(2, 2, 50.0).slice([0..1, 0..1, 0..3]))
            .expect("row 0 stage");
        per_row
            .stage_latent_sequence(0, 1, latent(2, 2, 50.0).slice([1..2, 0..2, 0..3]))
            .expect("row 1 stage");
        for row in 0..2 {
            let committed = [3, 1][row];
            let read = |cache: &NativeKvCache<TestBackend>| {
                cache
                    .read_working_sequence(0, row, committed)
                    .expect("working read")
                    .materialize_latent()
                    .map(|tensor| values(&tensor.unsqueeze_dim::<4>(1)))
            };
            assert_eq!(
                read(&batched),
                read(&per_row),
                "{storage:?} {in_place} row {row}"
            );
        }
        for cache in [&mut batched, &mut per_row] {
            let plan = cache
                .prepare_staging_commit(&[3, 1], &[1, 2], &[1, 1])
                .expect("commit");
            cache.apply_staging_commit(plan);
        }
        for row in 0..2 {
            let length = [4, 2][row];
            let read = |cache: &NativeKvCache<TestBackend>| {
                let read = cache.read_sequence(0, row, length).expect("read");
                (
                    read.retained_positions(),
                    read.materialize_latent()
                        .map(|tensor| values(&tensor.unsqueeze_dim::<4>(1))),
                )
            };
            assert_eq!(
                read(&batched),
                read(&per_row),
                "{storage:?} {in_place} row {row}"
            );
        }
        let plan = batched.plan_batch_read(&[4, 2]).expect("plan");
        let segment = batched
            .read_batch(0, &plan)
            .expect("read")
            .materialize()
            .expect("keys");
        assert_eq!(segment.latent().dims()[2], 3);
        assert_eq!(values(segment.keys()), values(segment.values()));
    }
}

/// A two-row key/value cache used to check plane-kind errors.
fn cache_kv() -> NativeKvCache<TestBackend> {
    cache()
}

/// Byte totals across every layer and plane are checked before any allocation.
///
/// Each plane here holds a handful of elements, so per-tensor checks pass;
/// only the product with `2 * layers` planes overflows. Committed storage is
/// checked first, and staging is checked on its own wider extent, so a cache
/// whose committed planes fit can still be rejected for its proposal buffers.
#[test]
fn repeated_plane_allocations_reject_byte_overflow_before_allocating() {
    let config = CacheConfig::new(1, 1).expect("config");
    let device = Default::default();
    let committed =
        NativeKvCacheSpec::new(1 << 62, 1, 1, KvCacheStorage::Full, DType::F32).expect("spec");
    assert!(matches!(
        NativeKvCache::<TestBackend>::new(config, 1, committed, &device),
        Err(TensorCacheError::AllocationSizeOverflow {
            tensor: "committed KV planes",
            elements,
            element_size: 4,
        }) if elements == 1 << 63
    ));
    // 2^60 one-slot planes take 2^62 bytes; four proposal slots each take 2^64.
    let staged = NativeKvCacheSpec::new(1 << 59, 1, 1, KvCacheStorage::Full, DType::F32)
        .expect("spec")
        .with_staging_capacity(4);
    assert!(matches!(
        NativeKvCache::<TestBackend>::new(config, 1, staged, &device),
        Err(TensorCacheError::AllocationSizeOverflow {
            tensor: "staged KV planes",
            elements,
            element_size: 4,
        }) if elements == 1 << 62
    ));
}

/// Segment-planning violations name the storage, the write end, or the ring error.
#[test]
fn segment_violations_map_to_contextual_cache_errors() {
    use crate::cache::tensor::plan::WriteEndViolation;

    assert!(matches!(
        segment_error(
            KvCacheStorage::MirroredSlidingRing,
            3,
            2,
            8,
            SegmentViolation::WindowRequired
        ),
        TensorCacheError::SlidingWindowRequired {
            storage: "mirrored sliding ring"
        }
    ));
    let write_end = segment_error(
        KvCacheStorage::Full,
        7,
        3,
        8,
        SegmentViolation::WriteEnd(WriteEndViolation::ExceedsCapacity { end: 10 }),
    );
    assert_eq!(
        write_end.to_string(),
        TensorCacheError::write_end(7, 3, 8, WriteEndViolation::ExceedsCapacity { end: 10 })
            .to_string()
    );
    let ring = crate::cache::RingPlanError::WindowExceedsCapacity {
        window: 5,
        capacity: 4,
    };
    assert!(matches!(
        segment_error(
            KvCacheStorage::SlidingRing,
            0,
            5,
            8,
            SegmentViolation::Ring(ring)
        ),
        TensorCacheError::Ring(error) if error == ring
    ));
}

/// A ring cache that lost its window reports it on reads and writes instead of panicking.
///
/// Constructors always resolve a ring's window; clearing it breaks that
/// private invariant to exercise the defensive mapping of the verified
/// planners' `WindowRequired` rejection in both the read and the write path.
#[test]
fn ring_without_a_resolved_window_is_a_contextual_error() {
    let config = CacheConfig::new(1, 4).expect("config");
    let spec = NativeKvCacheSpec::new(1, 1, 1, KvCacheStorage::SlidingRing, DType::F32)
        .expect("spec")
        .with_sliding_window(NonZeroUsize::new(2).expect("non-zero"));
    let mut cache =
        NativeKvCache::<TestBackend>::new(config, 1, spec, &Default::default()).expect("cache");
    cache
        .write_sequence(0, 0, 0, token(1.0), token(2.0))
        .expect("write");
    cache.window = None;
    assert!(matches!(
        cache.read_sequence(0, 0, 1),
        Err(TensorCacheError::SlidingWindowRequired {
            storage: "sliding ring"
        })
    ));
    assert!(matches!(
        cache.write_sequence(0, 0, 1, token(3.0), token(4.0)),
        Err(TensorCacheError::SlidingWindowRequired {
            storage: "sliding ring"
        })
    ));
}

/// Staging needs one baseline per row, and pair caches reject latent inputs by name.
#[test]
fn staging_baselines_and_plane_kinds_are_validated() {
    let mut cache = cache();
    assert!(matches!(
        cache.begin_staging_from(&[0, 0]),
        Err(TensorCacheError::Metadata(
            DecodeStateError::SequenceCountMismatch {
                expected: 1,
                actual: 2
            }
        ))
    ));
    assert!(!cache.staging_active);
    let latent = Tensor::<TestBackend, 3>::zeros([1, 1, 1], &Default::default());
    assert!(matches!(
        cache.write_latent_sequence(0, 0, 0, latent),
        Err(TensorCacheError::PlaneKindMismatch {
            expected: "a key/value pair; use the key/value methods"
        })
    ));
}

/// A working read of staged proposals needs every staging plane.
#[test]
fn working_reads_reject_incomplete_staging_planes() {
    let mut cache = cache();
    cache.begin_staging().expect("begin staging");
    cache
        .stage_sequence(0, 0, token(7.0), token(9.0))
        .expect("stage one proposal");
    cache.layers[0]
        .staged
        .as_mut()
        .expect("staging planes")
        .pop();
    assert!(matches!(
        cache.read_working_sequence(0, 0, 0),
        Err(TensorCacheError::StagingDisabled)
    ));
}
