//! Recurrent snapshot allocation defenses and ragged accepted-prefix selection.

use burn::backend::Flex;

use super::*;

/// Deterministic host backend for tensor lifecycle assertions.
type TestBackend = Flex<f32, i32>;

/// Allocates two layers and two rows with scalar recurrent geometry.
fn cache() -> RecurrentStateCache<TestBackend> {
    let config = CacheConfig::new(2, 4).expect("valid test config");
    let spec = RecurrentStateSpec::new(2, 1, 1, 1, 1, 1, DType::F32)
        .expect("valid test geometry")
        .with_staging_capacity(2);
    RecurrentStateCache::new(config, 2, spec, &Default::default()).expect("test cache allocation")
}

/// Reads assigned integer-valued tensors for exact fixture comparisons.
fn values<const D: usize>(tensor: &Tensor<TestBackend, D>) -> Vec<f32> {
    tensor
        .clone()
        .into_data()
        .to_vec()
        .expect("F32 test tensor")
}

/// Both batch and sequence staging validate the complete pair before updating either member.
#[test]
fn incomplete_staging_allocation_rejects_snapshots_atomically() {
    for whole_batch in [false, true] {
        for remove_matrix in [false, true] {
            let mut cache = cache();
            cache.begin_staging().expect("begin staging");
            // This deliberately corrupts a private invariant that constructors
            // maintain; callers cannot independently remove these allocations.
            if remove_matrix {
                cache.layers[0].staged_matrix = None;
            } else {
                cache.layers[0].staged_convolution = None;
            }
            let batch = if whole_batch { 2 } else { 1 };
            let matrices = Tensor::full([batch, 1, 1, 1, 1], 7.0, &Default::default());
            let convolutions = Tensor::full([batch, 1, 1, 1], 9.0, &Default::default());
            let result = if whole_batch {
                cache.stage_batch_snapshots(0, matrices, convolutions)
            } else {
                cache.stage_sequence_snapshots(0, 0, matrices, convolutions)
            };
            assert!(matches!(result, Err(TensorCacheError::StagingDisabled)));
            if remove_matrix {
                assert_eq!(
                    values(
                        cache.layers[0]
                            .staged_convolution
                            .as_ref()
                            .expect("surviving convolution")
                    ),
                    [0.0; 4]
                );
            } else {
                assert_eq!(
                    values(
                        cache.layers[0]
                            .staged_matrix
                            .as_ref()
                            .expect("surviving matrix")
                    ),
                    [0.0; 4]
                );
            }
            assert_eq!(cache.staged_lengths[0], [0, 0]);
        }
    }
}

/// A missing last-layer staging allocation fails planning before terminal states change.
#[test]
fn incomplete_staging_allocation_rejects_commit_before_publication() {
    for remove_matrix in [false, true] {
        let mut cache = cache();
        cache.begin_staging().expect("begin staging");
        for layer in 0..2 {
            cache
                .stage_batch_snapshots(
                    layer,
                    Tensor::full([2, 1, 1, 1, 1], 7.0, &Default::default()),
                    Tensor::full([2, 1, 1, 1], 9.0, &Default::default()),
                )
                .expect("stage layer");
        }
        if remove_matrix {
            cache.layers[1].staged_matrix = None;
        } else {
            cache.layers[1].staged_convolution = None;
        }
        assert!(matches!(
            cache.prepare_staging_commit(&[1, 1], &[0, 1]),
            Err(TensorCacheError::StagingDisabled)
        ));
        for layer in 0..2 {
            assert_eq!(values(cache.matrix(layer).expect("valid layer")), [0.0; 2]);
            assert_eq!(
                values(cache.convolution(layer).expect("valid layer")),
                [0.0; 2]
            );
            assert_eq!(cache.staged_lengths[layer], [1, 1]);
        }
        assert!(cache.has_active_staging());
    }
}

/// Zero acceptance retains each layer's baseline; positive acceptance selects its final snapshot.
#[test]
fn multi_layer_commit_selects_independent_ragged_snapshots() {
    let mut cache = cache();
    for layer in 0..2 {
        let baseline = (layer + 1) as f32;
        cache
            .set_batch_state(
                layer,
                Tensor::full([2, 1, 1, 1], baseline, &Default::default()),
                Tensor::full([2, 1, 1], baseline * 10.0, &Default::default()),
            )
            .expect("baseline state");
    }
    cache.begin_staging().expect("begin staging");
    for layer in 0..2 {
        for snapshot in 0..2 {
            let proposed = ((layer + 1) * 10 + snapshot + 1) as f32;
            cache
                .stage_batch_snapshots(
                    layer,
                    Tensor::full([2, 1, 1, 1, 1], proposed, &Default::default()),
                    Tensor::full([2, 1, 1, 1], proposed * 10.0, &Default::default()),
                )
                .expect("append snapshot");
        }
    }
    let plan = cache
        .prepare_staging_commit(&[2, 2], &[0, 2])
        .expect("ragged commit");
    cache.apply_staging_commit(plan);
    for layer in 0..2 {
        let baseline = (layer + 1) as f32;
        let accepted = ((layer + 1) * 10 + 2) as f32;
        assert_eq!(
            values(cache.matrix(layer).expect("valid layer")),
            [baseline, accepted]
        );
        assert_eq!(
            values(cache.convolution(layer).expect("valid layer")),
            [baseline * 10.0, accepted * 10.0]
        );
        assert_eq!(cache.staged_lengths[layer], [0, 0]);
    }
    assert!(!cache.has_active_staging());
}

/// Allocates a batch of `rows` with non-scalar geometry and distinct committed values.
fn batch_cache(
    rows: usize,
    heads: usize,
    size: usize,
    staging: usize,
) -> RecurrentStateCache<TestBackend> {
    let config = CacheConfig::new(rows, 16).expect("valid test config");
    let spec = RecurrentStateSpec::new(1, heads, size, size, 2, 3, DType::F32)
        .expect("valid test geometry")
        .with_staging_capacity(staging);
    let mut cache =
        RecurrentStateCache::new(config, rows, spec, &Default::default()).expect("allocation");
    let matrix = Tensor::<TestBackend, 1, Int>::arange(
        0..(rows * heads * size * size) as i64,
        &Default::default(),
    )
    .float()
    .reshape(spec.matrix_shape(rows));
    let convolution =
        Tensor::<TestBackend, 1, Int>::arange(0..(rows * 6) as i64, &Default::default())
            .float()
            .reshape(spec.convolution_shape(rows))
            .neg();
    cache
        .set_batch_state(0, matrix, convolution)
        .expect("committed state");
    cache
}

/// Stages `count` distinct snapshots for one row; snapshot values encode (row, index).
fn stage_row(cache: &mut RecurrentStateCache<TestBackend>, row: usize, count: usize) {
    let spec = cache.spec();
    let start = cache.staged_length(0, row).expect("valid row");
    for index in start..start + count {
        let value = 1000.0 * (row + 1) as f32 + index as f32;
        let matrix = Tensor::full(
            [
                1,
                1,
                spec.state_heads(),
                spec.state_rows(),
                spec.state_columns(),
            ],
            value,
            &Default::default(),
        );
        let convolution = Tensor::full(
            [1, 1, spec.convolution_channels(), spec.convolution_width()],
            -value,
            &Default::default(),
        );
        cache
            .stage_sequence_snapshots(0, row, matrix, convolution)
            .expect("snapshot fits");
    }
}

/// Reference batch read: concatenate committed state before the snapshots, then gather.
///
/// This is the previous implementation, kept here as the numerical oracle and
/// as the timing baseline for the cat-free path.
fn concatenated_reference(
    cache: &RecurrentStateCache<TestBackend>,
) -> (Tensor<TestBackend, 4>, Tensor<TestBackend, 3>) {
    let state = &cache.layers[0];
    let rows = cache.config.sequence_count();
    let coordinates: Vec<i32> = cache.staged_lengths[0]
        .iter()
        .enumerate()
        .flat_map(|(row, &staged)| [row as i32, staged as i32])
        .collect();
    let indices = Tensor::<TestBackend, 2, Int>::from_ints(
        TensorData::new(coordinates, [rows, 2]),
        &Default::default(),
    );
    let matrix = Tensor::cat(
        vec![
            state.matrix.clone().unsqueeze_dim::<5>(1),
            state.staged_matrix.clone().expect("staging"),
        ],
        1,
    );
    let convolution = Tensor::cat(
        vec![
            state.convolution.clone().unsqueeze_dim::<4>(1),
            state.staged_convolution.clone().expect("staging"),
        ],
        1,
    );
    (
        matrix.gather_nd::<2, 4>(indices.clone()),
        convolution.gather_nd::<2, 3>(indices),
    )
}

/// Uniform, ragged, partially empty, and empty staging all match per-row reads exactly.
#[test]
fn working_batch_reads_match_per_row_reads_for_every_staging_pattern() {
    for pattern in [
        vec![0, 0, 0],
        vec![2, 2, 2],
        vec![1, 3, 2],
        vec![0, 3, 1],
        vec![3, 0, 0],
    ] {
        let mut cache = batch_cache(3, 2, 2, 3);
        cache.begin_staging().expect("staging");
        for (row, &count) in pattern.iter().enumerate() {
            stage_row(&mut cache, row, count);
        }
        let batch = cache.read_working_batch(0).expect("batch read");
        assert_eq!(
            batch.staged_rows(),
            pattern.iter().map(|&count| count > 0).collect::<Vec<_>>()
        );
        let mut expected_matrix = Vec::new();
        let mut expected_convolution = Vec::new();
        for row in 0..3 {
            let read = cache.read_working_sequence(0, row).expect("row read");
            expected_matrix.extend(values(read.matrix()));
            expected_convolution.extend(values(read.convolution()));
        }
        assert_eq!(values(batch.matrix()), expected_matrix, "{pattern:?}");
        assert_eq!(
            values(batch.convolution()),
            expected_convolution,
            "{pattern:?}"
        );
        let (reference_matrix, reference_convolution) = concatenated_reference(&cache);
        assert_eq!(values(&reference_matrix), expected_matrix, "{pattern:?}");
        assert_eq!(values(&reference_convolution), expected_convolution);
    }
}

/// Times the cat-free batch read against the concatenating reference.
///
/// Run with `cargo test --release --lib working_batch_read_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing measurement; run explicitly in release mode"]
fn working_batch_read_timing() {
    use std::time::Instant;
    // A Gated-DeltaNet-sized state: 8 rows x 16 heads x 64 x 64 FP32, 8 snapshots.
    let mut cache = batch_cache(8, 16, 64, 8);
    cache.begin_staging().expect("staging");
    for row in 0..8 {
        stage_row(&mut cache, row, 1 + row % 4);
    }
    let iterations = 50;
    let started = Instant::now();
    for _ in 0..iterations {
        let read = cache.read_working_batch(0).expect("batch read");
        let _ = values(read.matrix());
    }
    let optimized = started.elapsed() / iterations;
    let started = Instant::now();
    for _ in 0..iterations {
        let (matrix, _) = concatenated_reference(&cache);
        let _ = values(&matrix);
    }
    let reference = started.elapsed() / iterations;
    println!(
        "ragged working batch read: cat-free {optimized:?}, concatenating reference {reference:?}"
    );
}

/// Every committed and staged allocation total is checked before any tensor exists.
///
/// Each case keeps the per-tensor shapes small (or exactly at one boundary)
/// and pushes one aggregate past `usize` bytes or one staged tensor past the
/// portable element limit, so the rejection names exactly that allocation.
/// Reduced-precision matrices let the matrix totals fit while the
/// convolution totals of the same layer count do not.
#[test]
fn recurrent_allocation_totals_name_the_first_unrepresentable_buffer() {
    let config = CacheConfig::new(1, 4).expect("config");
    let device = Default::default();
    let allocate = |spec: RecurrentStateSpec| {
        RecurrentStateCache::<TestBackend>::new(config, 1, spec, &device).err()
    };
    let spec = |layers, channels, staging, matrix_dtype| {
        RecurrentStateSpec::new_with_dtypes(layers, 1, 1, 1, channels, 1, matrix_dtype, DType::F32)
            .expect("geometry")
            .with_staging_capacity(staging)
    };
    let byte_overflow = |error: Option<TensorCacheError>, name: &str| {
        assert!(
            matches!(&error, Some(TensorCacheError::AllocationSizeOverflow { tensor, .. })
                if *tensor == name),
            "{name}: {error:?}"
        );
    };
    byte_overflow(
        allocate(spec(1 << 62, 1, 0, DType::F32)),
        "committed recurrent matrices",
    );
    byte_overflow(
        allocate(spec(1 << 62, 1, 0, DType::F16)),
        "committed convolution states",
    );
    byte_overflow(
        allocate(spec(1 << 60, 1, 4, DType::F32)),
        "staged recurrent matrices",
    );
    byte_overflow(
        allocate(spec(1 << 60, 1, 4, DType::F16)),
        "staged convolution states",
    );
    let element_limit = |error: Option<TensorCacheError>, name: &str| {
        assert!(
            matches!(&error, Some(TensorCacheError::TensorElementLimitExceeded { tensor, .. })
                if *tensor == name),
            "{name}: {error:?}"
        );
    };
    element_limit(
        allocate(spec(1, 1, 1 << 31, DType::F32)),
        "staged recurrent matrix",
    );
    element_limit(
        allocate(spec(1, 1 << 29, 4, DType::F32)),
        "staged convolution state",
    );
}

/// Batched state writes need one activity flag per row, and commits one count per row.
#[test]
fn recurrent_row_vectors_must_match_the_batch() {
    let cache = cache();
    let device = Default::default();
    let result = cache.prepare_batch_state(
        0,
        Tensor::<TestBackend, 4>::zeros([2, 1, 1, 1], &device),
        Tensor::<TestBackend, 3>::zeros([2, 1, 1], &device),
        &[true],
    );
    assert!(matches!(
        result,
        Err(TensorCacheError::Metadata(
            DecodeStateError::SequenceCountMismatch {
                expected: 2,
                actual: 1
            }
        ))
    ));
    for (proposed, accepted) in [(&[1][..], &[1, 1][..]), (&[1, 1], &[1])] {
        assert!(matches!(
            validate_commit_vector_lengths(2, proposed, accepted),
            Err(TensorCacheError::Metadata(
                DecodeStateError::SequenceCountMismatch {
                    expected: 2,
                    actual: 1
                }
            ))
        ));
    }
    validate_commit_vector_lengths(2, &[1, 0], &[0, 0]).expect("one count per row");
}
