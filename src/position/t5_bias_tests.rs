//! Learned T5 bias lookup layout, cached offsets, and allocation-boundary tests.

use burn::{
    backend::Flex,
    module::Param,
    tensor::{Int, Tensor, TensorData, Tolerance},
};

use super::*;

/// Learned lookup values follow the exact `[head, query, key]` permutation.
#[test]
fn tensor_bias_matches_bucket_lookup_and_head_layout() {
    let device = Default::default();
    let mut bias = T5RelativePositionBiasConfig::new(2, 4, 16, true)
        .init::<Flex>(&device)
        .expect("valid bias");
    bias.weight = Param::from_tensor(Tensor::from_floats(
        [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        &device,
    ));

    let actual = bias
        .forward(2, 3, 0, &device)
        .expect("valid tensor geometry");
    assert_eq!(actual.dims(), [1, 2, 2, 3]);
    let buckets = bias.buckets();
    let mut expected = vec![0.0_f32; 12];
    for head in 0..2 {
        for query in 0..2 {
            for key in 0..3 {
                let bucket = buckets.bucket_for_positions(query, key);
                expected[head * 6 + query * 3 + key] = bucket as f32 * 10.0 + head as f32;
            }
        }
    }
    actual.into_data().assert_approx_eq::<f32>(
        &TensorData::new(expected, [1, 2, 2, 3]),
        Tolerance::absolute(1e-6),
    );
}

/// Cached-query offsets select displacements against absolute key positions.
#[test]
fn query_offset_matches_absolute_bucket_positions() {
    let device = Default::default();
    let mut bias = T5RelativePositionBiasConfig::new(1, 8, 32, false)
        .init::<Flex>(&device)
        .expect("valid bias");
    bias.weight = Param::from_tensor(
        Tensor::<Flex, 1, Int>::arange(0..8, &device)
            .float()
            .reshape([8, 1]),
    );

    let actual = bias
        .forward(1, 4, 3, &device)
        .expect("one cached query")
        .into_data();
    actual.assert_approx_eq::<f32>(
        &TensorData::from([[[[3.0, 2.0, 1.0, 0.0]]]]),
        Tolerance::absolute(1e-6),
    );
}

/// Every invalid public shape or scalar is rejected before tensor construction.
#[test]
fn validation_rejects_zero_invalid_and_overflowing_inputs() {
    let device = Default::default();
    for config in [
        T5RelativePositionBiasConfig::new(0, 4, 16, true),
        T5RelativePositionBiasConfig::new(1, 3, 16, true),
        T5RelativePositionBiasConfig::new(1, 8, 4, false),
        T5RelativePositionBiasConfig::new(usize::MAX, 2, 8, false),
        T5RelativePositionBiasConfig::new(1, 8, 32, false).with_initializer_std(f64::NAN),
    ] {
        assert!(config.validate().is_err());
    }
    let bias = T5RelativePositionBiasConfig::new(1, 8, 32, false)
        .init::<Flex>(&device)
        .expect("valid bias");
    assert!(bias.forward(0, 1, 0, &device).is_err());
    assert!(bias.forward(1, 0, 0, &device).is_err());
    assert!(bias.forward(2, 1, usize::MAX, &device).is_err());
}

/// Query/key multiplication rejects overflow even when the query end fits usize.
#[test]
fn query_key_product_overflow_is_contextual_before_allocation() {
    let device = Default::default();
    let bias = T5RelativePositionBiasConfig::new(1, 8, 32, false)
        .init::<Flex>(&device)
        .expect("valid bias");
    let error = bias
        .forward(usize::MAX, 2, 0, &device)
        .expect_err("query times key must not wrap to a smaller allocation");
    assert!(matches!(
        error,
        Error::Configuration { field, reason }
            if field == "t5_relative_bias.shape"
                && reason == "query length times key length overflows usize"
    ));
}

/// A representable element count can still exceed Vec<i64>'s byte capacity.
#[test]
fn host_index_capacity_overflow_is_contextual_before_iteration() {
    let device = Default::default();
    let bias = T5RelativePositionBiasConfig::new(1, 8, 32, false)
        .init::<Flex>(&device)
        .expect("valid bias");
    let error = bias
        .forward(1, usize::MAX, 0, &device)
        .expect_err("unrepresentable Vec<i64> capacity must not allocate or iterate");
    assert!(matches!(
        error,
        Error::BackendCapability(reason)
            if reason == format!(
                "could not reserve {} host indices for T5 relative bias",
                usize::MAX,
            )
    ));
}

/// Wire-sized bucket counts must fit signed embedding indices before allocation.
#[test]
fn bucket_count_signed_index_bound_is_checked_before_initialization() {
    let Some(too_many_buckets) = usize::try_from(i64::MAX)
        .ok()
        .and_then(|maximum| maximum.checked_add(1))
    else {
        // A 32-bit usize cannot express an out-of-range i64 bucket count.
        return;
    };
    let device = Default::default();
    let config = T5RelativePositionBiasConfig::new(1, too_many_buckets, 128, false);
    for error in [
        config
            .validate()
            .expect_err("validate rejects signed overflow"),
        config
            .init::<Flex>(&device)
            .expect_err("init validates before allocation"),
    ] {
        assert!(matches!(error, Error::Configuration { field, reason }
            if field == "t5_relative_bias.num_buckets"
                && reason == "must fit in the signed integer index representation"));
    }
    let corrected = T5RelativePositionBiasConfig::new(1, 8, 128, false)
        .with_initializer_std(0.0)
        .init::<Flex>(&device)
        .expect("a corrected bucket count initializes");
    let output = corrected.forward(2, 3, 0, &device).expect("valid lookup");
    assert_eq!(output.dims(), [1, 1, 2, 3]);
    assert_eq!(
        output.into_data().to_vec::<f32>().expect("F32 table"),
        vec![0.0; 6]
    );
}
