//! Eager-softmax evaluation-precision tests: F64 references and historical F32 bit identity.

use burn::{
    backend::Flex,
    tensor::{
        Bool, DType, Tensor, TensorData, activation::softmax, backend::Backend,
        ops::AttentionModuleOptions,
    },
};

use super::{AttentionPrecisionMode, dispatch_attention, promoted_softmax};
use crate::precision::test_support::{
    assert_bitwise_eq, assert_f32_path_is_distinguishable, assert_matches_f64_reference,
    host_masked_softmax, near_tie_logits,
};

/// Frozen copy of the eager softmax before the evaluation-precision rule.
///
/// It promoted every score dtype to F32 unconditionally. Keeping it verbatim
/// proves that F16, BF16, and F32 scores still run the identical operation
/// sequence and shows that F64 scores used to be narrowed.
fn historical_fp32_softmax<B: Backend>(
    scores: Tensor<B, 4>,
    exclusion: Option<Tensor<B, 4, Bool>>,
) -> Tensor<B, 4> {
    let dtype = scores.dtype();
    let shape = scores.dims();
    let unsupported_row = scores
        .clone()
        .equal_elem(f32::NEG_INFINITY)
        .all_dim(3)
        .expand(shape);
    let scores = scores.mask_fill(unsupported_row.clone(), 0.0);
    let Some(exclusion) = exclusion else {
        return softmax(scores.cast(DType::F32), 3)
            .mask_fill(unsupported_row, 0.0)
            .cast(dtype);
    };
    let row_has_key = exclusion
        .clone()
        .bool_not()
        .int()
        .sum_dim(3)
        .greater_elem(0)
        .expand(shape);
    softmax(
        scores
            .cast(DType::F32)
            .mask_fill(row_has_key.bool_not(), 0.0),
        3,
    )
    .mask_fill(exclusion, 0.0)
    .mask_fill(unsupported_row, 0.0)
    .cast(dtype)
}

/// Score-plane shape `[batch, head, query, key]` shared by the softmax fixtures.
const SCORE_SHAPE: [usize; 4] = [2, 2, 3, 4];

/// Number of keys in each softmax row of [`SCORE_SHAPE`].
const KEYS: usize = SCORE_SHAPE[3];

/// Exclusions covering a fully masked row, partial rows, and unmasked rows.
fn exclusion_pattern() -> Vec<bool> {
    (0..SCORE_SHAPE.iter().product::<usize>())
        .map(|index| {
            let row = index / KEYS;
            let key = index % KEYS;
            row == 1 || (row % 3 == 2 && key % 2 == 1) || (row == 6 && key == 0)
        })
        .collect()
}

/// Creates score tensors of an explicit dtype from host doubles.
fn scores_tensor(values: Vec<f64>, dtype: DType) -> Tensor<Flex, 4> {
    Tensor::<Flex, 4>::from_data(
        TensorData::new(values, SCORE_SHAPE),
        (&Default::default(), DType::F64),
    )
    .cast(dtype)
}

/// Creates a boolean exclusion tensor with the score-plane shape.
fn exclusion_tensor(values: Vec<bool>) -> Tensor<Flex, 4, Bool> {
    Tensor::from_bool(TensorData::new(values, SCORE_SHAPE), &Default::default())
}

/// F64 scores keep F64 softmax with and without exclusions, matching a host reference.
///
/// The frozen historical graph narrows the same near-tie scores through F32 and
/// misses that reference by far more than F32 rounding of probabilities alone.
#[test]
fn f64_softmax_matches_an_independent_f64_reference() {
    for masked in [false, true] {
        let mut values = near_tie_logits(SCORE_SHAPE.iter().product());
        let exclusion = masked.then(exclusion_pattern);
        if let Some(exclusion) = &exclusion {
            for (value, excluded) in values.iter_mut().zip(exclusion) {
                if *excluded {
                    *value = f64::NEG_INFINITY;
                }
            }
        }
        // Row four loses all support through additive bias rather than a mask.
        for value in &mut values[4 * KEYS..5 * KEYS] {
            *value = f64::NEG_INFINITY;
        }
        let expected = host_masked_softmax(&values, exclusion.as_deref(), KEYS);
        let context = format!("masked={masked}");
        let scores = scores_tensor(values, DType::F64);
        let mask = exclusion.map(exclusion_tensor);

        let actual = promoted_softmax(scores.clone(), mask.clone()).into_data();
        assert_eq!(actual.dtype, DType::F64, "{context}");
        assert_matches_f64_reference(&actual, &expected, 1e-6, &context);
        assert_f32_path_is_distinguishable(
            &historical_fp32_softmax(scores, mask).into_data(),
            &expected,
            1e-6,
            &context,
        );
    }
}

/// F16, BF16, and F32 scores keep the historical FP32 softmax graph bit for bit.
#[test]
fn reduced_and_single_precision_softmax_match_the_historical_graph_bitwise() {
    let values = (0..SCORE_SHAPE.iter().product::<usize>())
        .map(|index| {
            if index / KEYS == 4 {
                f64::NEG_INFINITY
            } else {
                ((index as f64 + 1.0) * 0.917).sin() * 6.0
            }
        })
        .collect::<Vec<_>>();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        for masked in [false, true] {
            let mask = masked.then(|| exclusion_tensor(exclusion_pattern()));
            let scores = scores_tensor(values.clone(), dtype);
            let scores = match &mask {
                Some(mask) => scores.mask_fill(mask.clone(), f32::NEG_INFINITY),
                None => scores,
            };
            let actual = promoted_softmax(scores.clone(), mask.clone()).into_data();
            assert_eq!(actual.dtype, dtype);
            assert_bitwise_eq(
                &actual,
                &historical_fp32_softmax(scores, mask).into_data(),
                &format!("{dtype:?} masked={masked}"),
            );
        }
    }
}

/// Host F64 eager attention with scale, softcap, causal exclusion, and additive bias.
///
/// Shapes are `[query, head_dim]` for `query`, `[key, head_dim]` for `key`,
/// `[key, value_dim]` for `value`, and `[query, key]` for `bias`; causal
/// alignment is bottom-right, matching cached decoding.
fn host_eager_attention(
    query: &[[f64; 2]],
    key: &[[f64; 2]],
    value: &[[f64; 2]],
    bias: &[f64],
    scale: f64,
    softcap: Option<f64>,
    causal: bool,
) -> Vec<f64> {
    let keys = key.len();
    let offset = keys - query.len();
    query
        .iter()
        .enumerate()
        .flat_map(|(row, query)| {
            let scores = key
                .iter()
                .enumerate()
                .map(|(column, key_row)| {
                    if causal && column > row + offset {
                        return f64::NEG_INFINITY;
                    }
                    let raw = scale * (query[0] * key_row[0] + query[1] * key_row[1]);
                    let capped = softcap.map_or(raw, |cap| (raw / cap).tanh() * cap);
                    capped + bias[row * keys + column]
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f64>();
            (0..2).map(move |channel| {
                weights
                    .iter()
                    .zip(value)
                    .map(|(weight, value)| weight / total * value[channel])
                    .sum::<f64>()
            })
        })
        .collect()
}

/// Eager F64 attention keeps scores, softmax, and value mixing in F64 end to end.
///
/// Queries and keys produce near-tie scores around 1000. The same inputs rounded
/// through an F32 execution miss the reference by far more than F32 rounding of
/// the final output, so passing the F64 tolerance excludes any narrowed stage.
#[test]
fn eager_f64_attention_matches_an_independent_f64_reference() {
    let device = Default::default();
    let query = [[1.0, 1.1e-5], [1.0, 2.7e-5]];
    let key = [[1000.0, 0.0], [1000.0, 1.3], [1000.0, 2.9]];
    let value = [[2.0, -0.5], [4.25, 1.75], [8.5, 3.125]];
    let bias = [0.0, 3.1e-6, -1.7e-6, 2.3e-6, 0.0, -4.1e-6];
    let flat = |rows: &[[f64; 2]]| rows.iter().flatten().copied().collect::<Vec<_>>();
    for causal in [false, true] {
        for softcap in [None, Some(2000.0)] {
            let options = AttentionModuleOptions {
                scale: Some(1.0),
                softcap,
                is_causal: causal,
            };
            let expected = host_eager_attention(&query, &key, &value, &bias, 1.0, softcap, causal);
            let context = format!("causal={causal} softcap={softcap:?}");
            let run = |dtype: DType| {
                let tensor = |values: Vec<f64>, shape: [usize; 4]| {
                    Tensor::<Flex, 4>::from_data(
                        TensorData::new(values, shape),
                        (&device, DType::F64),
                    )
                    .cast(dtype)
                };
                dispatch_attention(
                    tensor(flat(&query), [1, 1, 2, 2]),
                    tensor(flat(&key), [1, 1, 3, 2]),
                    tensor(flat(&value), [1, 1, 3, 2]),
                    None,
                    Some(tensor(bias.to_vec(), [1, 1, 2, 3])),
                    options,
                    AttentionPrecisionMode::EagerFp32Softmax,
                )
                .expect("valid eager attention")
                .into_data()
            };

            let actual = run(DType::F64);
            assert_eq!(actual.dtype, DType::F64, "{context}");
            assert_matches_f64_reference(&actual, &expected, 1e-3, &context);
            assert_f32_path_is_distinguishable(&run(DType::F32), &expected, 1e-3, &context);
        }
    }
}
