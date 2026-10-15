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

/// Displacement tables beyond the signed device index range are rejected before allocation.
#[test]
fn displacement_table_index_range_is_contextual_before_allocation() {
    let device = Default::default();
    let bias = T5RelativePositionBiasConfig::new(1, 8, 32, false)
        .init::<Flex>(&device)
        .expect("valid bias");
    let error = bias
        .forward(1, usize::MAX, 0, &device)
        .expect_err("an index past i64::MAX cannot address the device table");
    assert!(matches!(
        error,
        Error::Configuration { field, reason }
            if field == "t5_relative_bias.shape"
                && reason == "query length plus key length exceeds the i64 index range"
    ));
}

/// A representable displacement count can still exceed Vec<i64>'s byte capacity.
#[test]
fn host_bucket_capacity_overflow_is_contextual_before_iteration() {
    let device = Default::default();
    let bias = T5RelativePositionBiasConfig::new(1, 8, 32, false)
        .init::<Flex>(&device)
        .expect("valid bias");
    // 2^61 eight-byte buckets need 2^64 bytes, which no allocation can provide.
    let keys = 1_usize << (usize::BITS - 3);
    let error = bias
        .forward(1, keys, 0, &device)
        .expect_err("unrepresentable Vec<i64> capacity must not allocate or iterate");
    assert!(matches!(
        error,
        Error::BackendCapability(reason)
            if reason == format!("could not reserve {keys} host buckets for T5 relative bias")
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

/// Builds the bias the straightforward way: one bucket per `(query, key)` pair.
fn pairwise_reference(
    bias: &T5RelativePositionBias<Flex>,
    query_length: usize,
    key_length: usize,
    query_offset: usize,
) -> TensorData {
    let weight = bias
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .expect("F32 table");
    let heads = bias.weight.dims()[1];
    let mut expected = vec![0.0_f32; heads * query_length * key_length];
    for head in 0..heads {
        for query in 0..query_length {
            for key in 0..key_length {
                let bucket = bias
                    .buckets()
                    .bucket_for_positions(query_offset + query, key);
                expected[(head * query_length + query) * key_length + key] =
                    weight[bucket * heads + head];
            }
        }
    }
    TensorData::new(expected, [1, heads, query_length, key_length])
}

/// The displacement table holds exactly the anchor-row buckets in displacement order.
#[test]
fn bucket_table_lists_anchor_displacements() {
    let device = Default::default();
    let bias = T5RelativePositionBiasConfig::new(1, 8, 32, true)
        .init::<Flex>(&device)
        .expect("valid bias");
    let table = bias.bucket_table(3, 4, 5).expect("valid table");
    assert_eq!((table.query_length(), table.key_length()), (3, 4));
    assert_eq!(table.mapping(), bias.buckets());
    assert_eq!(table.bucket_ids().len(), 6);
    for (entry, &bucket) in table.bucket_ids().iter().enumerate() {
        // Anchor query position = offset + query_length - 1 = 7.
        let expected = bias.buckets().bucket_for_positions(7, entry);
        assert_eq!(bucket, expected as i64);
    }
    let reused = bias
        .forward_with_table(&table, &device)
        .expect("table from this bias");
    assert_eq!(
        reused.into_data(),
        bias.forward(3, 4, 5, &device).expect("forward").into_data()
    );
}

/// A table built by a bias with another bucket mapping is rejected, not misread.
///
/// More buckets would index the embedding out of range and the other
/// direction mode would silently select the wrong learned scalars, so every
/// mapping difference is a contextual configuration error.
#[test]
fn table_from_a_different_bucket_mapping_is_rejected() {
    let device = Default::default();
    let bias = T5RelativePositionBiasConfig::new(2, 8, 32, true)
        .init::<Flex>(&device)
        .expect("valid bias");
    for (buckets, max_distance, bidirectional) in [(16, 32, true), (8, 64, true), (8, 32, false)] {
        let other = T5RelativePositionBiasConfig::new(2, buckets, max_distance, bidirectional)
            .init::<Flex>(&device)
            .expect("valid bias");
        let table = other.bucket_table(3, 5, 0).expect("valid table");
        let error = bias
            .forward_with_table(&table, &device)
            .expect_err("mismatched mapping");
        assert!(
            matches!(&error, crate::Error::Configuration { field, .. }
                if field == "t5_relative_bias.table"),
            "{error}"
        );
    }
}

/// The device table index equals the Verus-verified host index at every pair.
///
/// [`super::super::relative_table_index`] is the proved specification of the
/// device formula; this links the two for square, wide, tall, and single-axis
/// geometries.
#[test]
fn device_table_index_matches_verified_host_index() {
    let device = Default::default();
    for (query_length, key_length) in [(1, 1), (1, 7), (6, 1), (4, 4), (3, 9), (9, 3)] {
        let entries = super::table_entries::<Flex>(query_length, key_length, &device)
            .into_data()
            .to_vec::<i32>()
            .expect("I32 indices");
        for query in 0..query_length {
            for key in 0..key_length {
                assert_eq!(
                    entries[query * key_length + key] as usize,
                    super::super::relative_table_index(query, key, query_length),
                    "query={query}, key={key}, shape=[{query_length}, {key_length}]"
                );
            }
        }
    }
}

/// The reference initializer scale is `factor / sqrt(d_model)` and invalid inputs fail validation.
#[test]
fn reference_initialization_matches_t5_scale() {
    let config =
        T5RelativePositionBiasConfig::new(8, 32, 128, true).with_reference_initialization(512, 1.0);
    assert!((config.initializer_std - 512.0_f64.powf(-0.5)).abs() < 1e-15);
    assert!(config.validate().is_ok());
    for (d_model, factor) in [(0, 1.0), (512, -1.0), (512, f64::NAN)] {
        let invalid = T5RelativePositionBiasConfig::new(8, 32, 128, true)
            .with_reference_initialization(d_model, factor);
        assert!(
            invalid.validate().is_err(),
            "d_model={d_model}, factor={factor}"
        );
    }
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(48))]

    /// The linear-size table reproduces the quadratic pairwise lookup exactly.
    #[test]
    fn displacement_table_matches_pairwise_lookup(
        query_length in 1usize..12,
        key_length in 1usize..24,
        query_offset in 0usize..40,
        bidirectional in proptest::bool::ANY,
        buckets in 2usize..10,
        max_distance in 6usize..64,
    ) {
        let num_buckets = if bidirectional { 2 * buckets.max(2) } else { buckets };
        let device = Default::default();
        let mut bias = T5RelativePositionBiasConfig::new(2, num_buckets, max_distance, bidirectional)
            .init::<Flex>(&device)
            .expect("valid bias");
        let values = (0..num_buckets * 2).map(|index| index as f32 * 0.5 - 3.0).collect::<Vec<_>>();
        bias.weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(values, [num_buckets, 2]),
            &device,
        ));
        let actual = bias
            .forward(query_length, key_length, query_offset, &device)
            .expect("valid geometry");
        proptest::prop_assert_eq!(
            actual.into_data(),
            pairwise_reference(&bias, query_length, key_length, query_offset)
        );
    }
}

/// Gradients reach exactly the learned buckets that the bias reads.
#[cfg(feature = "autodiff")]
#[test]
fn bias_table_receives_gradients_per_used_bucket() {
    type Autodiff = burn::backend::Autodiff<Flex>;
    let device = Default::default();
    let bias = T5RelativePositionBiasConfig::new(1, 8, 32, false)
        .init::<Autodiff>(&device)
        .expect("valid bias");
    let output = bias.forward(2, 3, 0, &device).expect("bias");
    let gradients = output.sum().backward();
    let gradient = bias
        .weight
        .grad(&gradients)
        .expect("weight gradient")
        .into_data()
        .to_vec::<f32>()
        .expect("F32 gradient");
    // Causal buckets: query 0 reads keys (0, 1, 2) -> buckets (0, 0, 0); query 1 reads
    // (1, 0, 0) -> past distance one, then the future keys clamp to bucket zero.
    let mut counts = [0.0_f32; 8];
    for query in 0..2 {
        for key in 0..3 {
            counts[bias.buckets().bucket_for_positions(query, key)] += 1.0;
        }
    }
    assert_eq!(gradient, counts.to_vec());
}
