//! Exactness of the per-pass T5 relative-bias builders against a pairwise reference.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use proptest::prelude::*;

use super::*;
use crate::position::{T5RelativePositionBiasConfig, T5RelativePositionBuckets};

/// Deterministic CPU backend for exact bias comparisons.
type TestBackend = Flex<f32, i32>;

/// Builds a randomly initialized bias table with `heads` heads.
fn table(
    buckets: usize,
    max_distance: usize,
    bidirectional: bool,
    heads: usize,
) -> T5RelativePositionBias<TestBackend> {
    T5RelativePositionBiasConfig::new(heads, buckets, max_distance, bidirectional)
        .init(&Default::default())
        .expect("valid bias geometry")
}

/// Uploads row-major positions as a `[rows, columns]` Int tensor.
fn positions(values: &[i64], rows: usize) -> Tensor<TestBackend, 2, Int> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [rows, values.len() / rows]),
        &Default::default(),
    )
}

/// Evaluates `W[bucket(k - q), h]` pair by pair on the host: the reference.
fn pairwise(
    table: &T5RelativePositionBias<TestBackend>,
    queries: &[i64],
    keys: &[i64],
    rows: usize,
) -> Vec<f32> {
    let weight = table.weight.val().into_data().to_vec::<f32>().unwrap();
    let heads = table.weight.val().dims()[1];
    let (query_count, key_count) = (queries.len() / rows, keys.len() / rows);
    let mut values = Vec::new();
    for row in 0..rows {
        for head in 0..heads {
            for query in 0..query_count {
                for key in 0..key_count {
                    let bucket = table.buckets().bucket_for_positions(
                        queries[row * query_count + query] as usize,
                        keys[row * key_count + key] as usize,
                    );
                    values.push(weight[bucket * heads + head]);
                }
            }
        }
    }
    values
}

/// Reads an F32 tensor into a host vector.
fn floats<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

/// Per-row positional bias reproduces the pairwise lookup bit for bit.
#[test]
fn positional_bias_matches_pairwise_lookup_for_ragged_rows() {
    let bias_table = table(8, 6, true, 3);
    let queries = [0, 1, 2, 5, 9, 10];
    let keys = [0, 1, 2, 5, 9, 10];
    let actual =
        positional_bias(&bias_table, positions(&queries, 2), positions(&keys, 2), 10).unwrap();
    assert_eq!(actual.dims(), [2, 3, 3, 3]);
    assert_eq!(floats(actual), pairwise(&bias_table, &queries, &keys, 2));
}

/// A single position row broadcasts over the batch like the contiguous bias.
#[test]
fn uniform_rows_match_the_contiguous_bias_exactly() {
    for bidirectional in [true, false] {
        let bias_table = table(8, 16, bidirectional, 2);
        let arange: Vec<i64> = (0..7).collect();
        let positional =
            positional_bias(&bias_table, positions(&arange, 1), positions(&arange, 1), 6).unwrap();
        let contiguous = contiguous_bias(&bias_table, 7, 7, 0, &Default::default()).unwrap();
        assert_eq!(positional.dims(), [1, 2, 7, 7]);
        assert_eq!(floats(positional), floats(contiguous));
    }
}

/// Clamping beyond `max_distance` is exact because the T5 mapping saturates there.
#[test]
fn displacements_beyond_the_maximum_distance_share_the_saturated_bucket() {
    for (buckets, max_distance) in [(8, 6), (32, 128), (32, 17), (4, 3)] {
        for bidirectional in [true, false] {
            let Ok(mapping) = T5RelativePositionBuckets::new(buckets, max_distance, bidirectional)
            else {
                continue;
            };
            let limit = max_distance as i64;
            for distance in limit..limit + 600 {
                assert_eq!(mapping.bucket(distance), mapping.bucket(limit));
                assert_eq!(mapping.bucket(-distance), mapping.bucket(-limit));
            }
        }
    }
    // A long sequence clamps every displacement beyond 6 and still matches.
    let bias_table = table(8, 6, true, 1);
    let long: Vec<i64> = (0..40).collect();
    let actual =
        positional_bias(&bias_table, positions(&long, 1), positions(&long, 1), 39).unwrap();
    assert_eq!(floats(actual), pairwise(&bias_table, &long, &long, 1));
}

/// Empty or mismatched position tensors fail with a shape error before any lookup.
#[test]
fn malformed_position_tensors_are_rejected() {
    let bias_table = table(8, 6, true, 1);
    let error = positional_bias(
        &bias_table,
        positions(&[0, 1], 2),
        positions(&[0, 1, 2], 1),
        3,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("t5.relative_positions"),
        "{error}"
    );
    let empty = Tensor::<TestBackend, 2, Int>::zeros([1, 0], &Default::default());
    assert!(positional_bias(&bias_table, empty, positions(&[0], 1), 0).is_err());
}

proptest! {
    /// Random per-row positions (gapped, unsorted, repeated) always match the pairwise lookup.
    #[test]
    fn positional_bias_property(
        rows in 1_usize..3,
        queries in 1_usize..5,
        keys in 1_usize..5,
        bidirectional in any::<bool>(),
        seed in proptest::collection::vec(0_i64..30, 32),
    ) {
        let bias_table = table(8, 6, bidirectional, 2);
        let query_positions: Vec<i64> = seed.iter().copied().take(rows * queries).collect();
        let key_positions: Vec<i64> =
            seed.iter().rev().copied().take(rows * keys).collect();
        let actual = positional_bias(
            &bias_table,
            positions(&query_positions, rows),
            positions(&key_positions, rows),
            29,
        )
        .unwrap();
        prop_assert_eq!(
            floats(actual),
            pairwise(&bias_table, &query_positions, &key_positions, rows)
        );
    }
}
