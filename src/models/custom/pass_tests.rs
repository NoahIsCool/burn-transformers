//! Stack-pass tables against the per-layer reference algorithms they replace.
//!
//! The reference functions below are the former per-layer implementations:
//! a host loop over every `(row, query, key)` triple for T5 buckets, and a
//! `[batch, heads, query, key]` integer comparison for trailing windows. The
//! pass must reproduce them exactly wherever their semantics are defined.

use burn::tensor::{Tolerance, backend::BackendTypes};

use super::*;
use crate::{
    attention::{AttentionConnectivity, padding_attention_mask},
    models::custom::{
        plan::displacement_slot,
        test_support::{TestBackend, mask},
    },
    position::{RopeScaling, T5RelativePositionBiasConfig, rotary_angles},
};

/// Device used by every pass fixture.
type Device = <TestBackend as BackendTypes>::Device;

/// Former per-layer host evaluation: `weight[bucket(p[b, k] - p[b, q]), h]` for every pair.
fn reference_relative_bias(
    bias: &T5RelativePositionBias<TestBackend>,
    positions: &[Vec<i64>],
) -> Vec<f32> {
    let weight = bias.weight.val().into_data().to_vec::<f32>().unwrap();
    let heads = bias.weight.dims()[1];
    let buckets = bias.buckets();
    let mut values = Vec::new();
    for row in positions {
        for head in 0..heads {
            for &query in row {
                for &key in row {
                    let bucket = buckets.bucket_for_positions(query as usize, key as usize);
                    values.push(weight[bucket * heads + head]);
                }
            }
        }
    }
    values
}

/// Former trailing-window mask: padding plus `r_q - r_k >= window` over `[batch, heads, S, S]`.
///
/// `r` is the inclusive valid-token count `cumsum(valid)`; causality is left to the kernel.
fn reference_sliding_mask(
    token_mask: Option<&TokenMask<TestBackend>>,
    batch: usize,
    heads: usize,
    sequence: usize,
    window: usize,
    device: &Device,
) -> Option<Tensor<TestBackend, 4, Bool>> {
    let padding =
        padding_attention_mask(None, token_mask, batch, heads, sequence, sequence).unwrap();
    if window >= sequence {
        return padding;
    }
    let ordinals = match token_mask {
        Some(mask) => mask.as_tensor().clone().int().cumsum(1),
        None => Tensor::<TestBackend, 1, Int>::arange(0..sequence as i64, device)
            .reshape([1, sequence])
            .expand([batch, sequence]),
    };
    let query = ordinals
        .clone()
        .reshape([batch, 1, sequence, 1])
        .expand([batch, heads, sequence, sequence]);
    let key = ordinals
        .reshape([batch, 1, 1, sequence])
        .expand([batch, heads, sequence, sequence]);
    let too_old = (query - key).greater_equal_elem(window as i64);
    Some(match padding {
        Some(padding) => padding.bool_or(too_old),
        None => too_old,
    })
}

/// A `[rows, tokens]` position tensor from nested host rows.
fn positions(rows: &[Vec<i64>], device: &Device) -> Tensor<TestBackend, 2, Int> {
    let tokens = rows[0].len();
    let flat = rows
        .iter()
        .flatten()
        .map(|&value| value as i32)
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(flat, [rows.len(), tokens]), device)
}

/// A one-head relative bias whose bucket `b` stores the value `b`.
fn indexed_bias(
    buckets: usize,
    max_distance: usize,
    bidirectional: bool,
    heads: usize,
    device: &Device,
) -> T5RelativePositionBias<TestBackend> {
    let mut bias = T5RelativePositionBiasConfig::new(heads, buckets, max_distance, bidirectional)
        .init::<TestBackend>(device)
        .unwrap();
    let values = (0..buckets * heads)
        .map(|index| index as f32 * 0.25 + 1.0)
        .collect::<Vec<_>>();
    bias.weight = burn::module::Param::from_tensor(Tensor::from_data(
        TensorData::new(values, [buckets, heads]),
        device,
    ));
    bias
}

/// Builds a structural decoder pass over explicit host positions.
fn pass_over<'a>(
    rows: &[Vec<i64>],
    token_mask: Option<&'a TokenMask<TestBackend>>,
    device: &Device,
) -> StackPass<'a, TestBackend> {
    let context = rows
        .iter()
        .flatten()
        .max()
        .map_or(1, |&max| max as usize + 1);
    StackPass::new(
        positions(rows, device),
        token_mask,
        context,
        Stack::Decoder,
        DType::F32,
    )
}

/// Gapped, offset, and per-row positions select exactly the pairwise reference buckets.
#[test]
fn relative_bias_matches_the_pairwise_reference_for_arbitrary_positions() {
    let device = Default::default();
    let rows = vec![vec![5, 7, 10, 40, 41, 300], vec![0, 0, 3, 90, 2, 1]];
    for (buckets, max_distance, bidirectional) in [
        (8, 32, false),
        (8, 32, true),
        (32, 128, true),
        (32, 128, false),
        (4, 3, true),
        (16, 9, false),
    ] {
        let bias = indexed_bias(buckets, max_distance, bidirectional, 2, &device);
        let pass = pass_over(&rows, None, &device);
        let actual = pass.relative_bias(&bias).unwrap();
        assert_eq!(actual.dims(), [2, 2, 6, 6]);
        assert_eq!(
            actual.into_data().to_vec::<f32>().unwrap(),
            reference_relative_bias(&bias, &rows),
            "buckets {buckets}, max_distance {max_distance}, bidirectional {bidirectional}"
        );
    }
}

/// Device bucket IDs read the host table at the verified slot `clamp(d, -R, R) + R`.
#[test]
fn relative_bucket_ids_follow_the_verified_slot_specification() {
    let device = Default::default();
    let rows = vec![vec![2, 9, 4, 30]];
    let buckets = T5RelativePositionBuckets::new(8, 5, true).unwrap();
    let pass = pass_over(&rows, None, &device);
    let ids = pass.relative_bucket_ids(buckets).unwrap();
    let table = displacement_table(5, 31).unwrap();
    let mut expected = Vec::new();
    for &query in &rows[0] {
        for &key in &rows[0] {
            let slot = displacement_slot(key - query, table.radius);
            let bucket = buckets.bucket(slot_displacement(slot, table.radius));
            expected.push(bucket as i32);
        }
    }
    assert_eq!(ids.into_data().to_vec::<i32>().unwrap(), expected);
}

/// Shared `0..token` positions build one batch-broadcast table equal to every row's table.
#[test]
fn row_invariant_positions_share_one_relative_table() {
    let device = Default::default();
    let rows = vec![(0..7).collect::<Vec<i64>>(); 3];
    let bias = indexed_bias(8, 4, true, 2, &device);
    let shared = pass_over(&rows, None, &device)
        .with_row_invariant_positions(true)
        .relative_bias(&bias)
        .unwrap();
    assert_eq!(shared.dims(), [1, 2, 7, 7]);
    let per_row = pass_over(&rows, None, &device)
        .relative_bias(&bias)
        .unwrap();
    assert_eq!(per_row.dims(), [3, 2, 7, 7]);
    shared
        .expand([3, 2, 7, 7])
        .into_data()
        .assert_eq(&per_row.into_data(), true);
}

/// A context too large for a portable 32-bit table is rejected with a contextual error.
#[test]
fn oversized_relative_tables_are_rejected() {
    let device = Default::default();
    let bias = indexed_bias(8, usize::MAX / 4, false, 1, &device);
    let pass = StackPass::new(
        positions(&[vec![0, 1]], &device),
        None,
        usize::MAX / 4,
        Stack::Decoder,
        DType::F32,
    );
    let error = pass.relative_bias(&bias).unwrap_err();
    assert!(
        error.to_string().contains("custom.t5_relative_bias"),
        "{error}"
    );
}

/// Sliding exclusions equal the former per-head mask on every valid query row.
///
/// Padded query rows differ only in rows whose outputs are zeroed after
/// attention; the pass never excludes padding queries' keys beyond padding.
#[test]
fn sliding_exclusions_match_the_reference_on_valid_queries() {
    let device = Default::default();
    let layouts: [[bool; 7]; 4] = [
        [true; 7],
        [false, false, true, true, true, true, true],
        [true, true, true, true, true, false, false],
        [false, true, false, true, true, false, true],
    ];
    for validity in layouts {
        let token_mask = mask([validity, [true; 7]], &device);
        let rows = vec![(0..7).collect::<Vec<i64>>(); 2];
        for window in [1, 2, 3, 8] {
            let pass = pass_over(&rows, Some(&token_mask), &device);
            let actual = pass
                .self_attention(
                    Layer::decoder(0),
                    AttentionConnectivity::SlidingWindow { window },
                    3,
                )
                .unwrap();
            assert!(actual.causal);
            assert!(actual.bias.is_none());
            let actual = actual.mask.unwrap().into_data().to_vec::<bool>().unwrap();
            let expected = reference_sliding_mask(Some(&token_mask), 2, 3, 7, window, &device)
                .unwrap()
                .into_data()
                .to_vec::<bool>()
                .unwrap();
            let row_validity = [validity, [true; 7]];
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                let query = (index / 7) % 7;
                let row = index / (3 * 7 * 7);
                if row_validity[row][query] {
                    assert_eq!(actual, expected, "window {window}, layout {validity:?}");
                }
            }
        }
    }
}

/// Without padding, global attention needs no mask and sliding masks broadcast over rows.
#[test]
fn unpadded_exclusions_are_absent_or_batch_broadcast() {
    let device = Default::default();
    let rows = vec![(0..5).collect::<Vec<i64>>(); 2];
    let pass = pass_over(&rows, None, &device);
    let global = pass
        .self_attention(Layer::decoder(0), AttentionConnectivity::Global, 2)
        .unwrap();
    assert!(global.mask.is_none());
    let sliding = pass
        .self_attention(
            Layer::decoder(0),
            AttentionConnectivity::SlidingWindow { window: 2 },
            2,
        )
        .unwrap()
        .mask
        .unwrap();
    assert_eq!(sliding.dims(), [2, 2, 5, 5]);
    let expected = reference_sliding_mask(None, 2, 2, 5, 2, &device).unwrap();
    sliding.into_data().assert_eq(&expected.into_data(), true);
}

/// Every table is built once per distinct configuration and reused afterwards.
#[test]
fn tables_are_memoized_per_distinct_configuration() {
    let device = Default::default();
    let rows = vec![vec![0, 1, 2, 3]];
    let pass = pass_over(&rows, None, &device);
    let rope = RopeConfig::new(4, 4, 10_000.0, RopeScaling::None).unwrap();
    let narrow = RopeConfig::new(4, 2, 10_000.0, RopeScaling::None).unwrap();
    pass.rotary_angles(rope).unwrap();
    pass.rotary_angles(rope).unwrap();
    pass.rotary_angles(narrow).unwrap();
    let bias = indexed_bias(8, 4, true, 1, &device);
    pass.relative_bias(&bias).unwrap();
    pass.relative_bias(&bias).unwrap();
    for _ in 0..2 {
        for connectivity in [
            AttentionConnectivity::Global,
            AttentionConnectivity::SlidingWindow { window: 2 },
        ] {
            pass.self_attention(Layer::decoder(0), connectivity, 1)
                .unwrap();
        }
    }
    let tables = pass.tables.borrow();
    assert_eq!(tables.rotary.len(), 2);
    assert_eq!(tables.relative_buckets.len(), 1);
    assert_eq!(tables.exclusions.len(), 2);
}

/// Pass rotary tables equal the per-call construction for explicit positions.
#[test]
fn rotary_tables_match_direct_construction() {
    let device = Default::default();
    let rows = vec![vec![3, 9, 4, 27], vec![0, 1, 2, 3]];
    let pass = pass_over(&rows, None, &device);
    let rope = RopeConfig::new(4, 4, 10_000.0, RopeScaling::None).unwrap();
    let cached = pass.rotary_angles(rope).unwrap();
    let direct = rotary_angles::<TestBackend>(rope, positions(&rows, &device), 28).unwrap();
    cached
        .cosine
        .into_data()
        .assert_approx_eq::<f32>(&direct.cosine.into_data(), Tolerance::absolute(0.0));
    cached
        .sine
        .into_data()
        .assert_approx_eq::<f32>(&direct.sine.into_data(), Tolerance::absolute(0.0));
}

/// An explicit policy replaces causality and connectivity, keeps padding, and adds its bias.
#[test]
fn explicit_policies_replace_structure_but_keep_padding() {
    let device = Default::default();
    let token_mask = mask([[false, true, true]], &device);
    let rows = vec![vec![0, 0, 1]];
    let allowed = Tensor::<TestBackend, 4, Bool>::from_bool(
        [[[[true, true, false], [true, true, true], [false, true, true]]]].into(),
        &device,
    );
    let bias = Tensor::<TestBackend, 4>::ones([1, 1, 3, 3], &device);
    let pass = pass_over(&rows, Some(&token_mask), &device)
        .with_self_attention_policy(AttentionPolicy::custom(allowed).with_bias(bias));
    let resolved = pass
        .self_attention(
            Layer::decoder(0),
            AttentionConnectivity::SlidingWindow { window: 1 },
            2,
        )
        .unwrap();
    assert!(!resolved.causal);
    // Key 0 is padding; otherwise only the disallowed links are excluded.
    let expected = [true, false, true, true, false, false, true, false, false];
    let mask = resolved.mask.clone().unwrap();
    assert_eq!(mask.dims(), [1, 2, 3, 3]);
    assert_eq!(
        mask.into_data().to_vec::<bool>().unwrap(),
        [expected, expected].concat()
    );
    let combined = resolved.with_position_bias(
        Some(Tensor::<TestBackend, 4>::ones([1, 2, 3, 3], &device)),
        [1, 2, 3, 3],
    );
    combined.bias.unwrap().into_data().assert_eq(
        &Tensor::<TestBackend, 4>::full([1, 2, 3, 3], 2.0, &device).into_data(),
        true,
    );
}

/// Cross-attention sees every valid source token unless an explicit policy restricts it.
#[test]
fn cross_attention_defaults_to_source_padding() {
    let device = Default::default();
    let source = Tensor::<TestBackend, 3>::zeros([1, 4, 8], &device);
    let source_mask = mask([[true, false, true, true]], &device);
    let rows = vec![vec![0, 1]];
    let pass = pass_over(&rows, None, &device).with_source(&source, Some(&source_mask));
    let resolved = pass.cross_attention(Layer::decoder(0), 2).unwrap();
    assert!(!resolved.causal);
    assert_eq!(
        resolved.mask.unwrap().into_data().to_vec::<bool>().unwrap(),
        [false, true, false, false].repeat(4)
    );
    let missing = pass_over(&rows, None, &device)
        .cross_attention(Layer::decoder(0), 2)
        .unwrap_err();
    assert!(missing.to_string().contains("encoder hidden states"));
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]

    /// Random positions, bucket counts, and saturation distances reproduce the pairwise reference.
    #[test]
    fn relative_bias_parity_holds_for_random_geometries(
        values in proptest::collection::vec(0i64..48, 10),
        half_buckets in 2usize..10,
        distance_extra in 1usize..40,
        bidirectional in proptest::bool::ANY,
    ) {
        let device = Default::default();
        let buckets = 2 * half_buckets;
        let per_direction = if bidirectional { half_buckets } else { buckets };
        let max_distance = per_direction / 2 + distance_extra;
        let rows = vec![values[..5].to_vec(), values[5..].to_vec()];
        let bias = indexed_bias(buckets, max_distance, bidirectional, 1, &device);
        let actual = pass_over(&rows, None, &device)
            .relative_bias(&bias)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        proptest::prop_assert_eq!(actual, reference_relative_bias(&bias, &rows));
    }
}
