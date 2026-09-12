//! Internal staging-allocation defenses and multi-layer KV commit boundaries.

use burn::backend::Flex;

use super::*;

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

/// A missing member of an internal staging pair cannot partially change its surviving tensor.
#[test]
fn incomplete_staging_allocation_rejects_writes_before_mutation() {
    for remove_keys in [false, true] {
        let mut cache = cache();
        cache.begin_staging().expect("begin staging");
        // Public constructors always allocate both members; deliberately break
        // the private invariant to exercise the recoverable defensive path.
        if remove_keys {
            cache.layers[0].staged_keys = None;
        } else {
            cache.layers[0].staged_values = None;
        }
        assert!(matches!(
            cache.stage_sequence(0, 0, token(7.0), token(9.0)),
            Err(TensorCacheError::StagingDisabled)
        ));
        let surviving = if remove_keys {
            cache.layers[0].staged_values.as_ref()
        } else {
            cache.layers[0].staged_keys.as_ref()
        }
        .expect("one staging member remains");
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
        if remove_keys {
            cache.layers[1].staged_keys = None;
        } else {
            cache.layers[1].staged_values = None;
        }
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
