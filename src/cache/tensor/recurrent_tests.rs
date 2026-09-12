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
