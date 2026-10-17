//! Latent-attention evaluation-precision tests: F64 references and historical F32 bit identity.

use burn::{
    backend::Flex,
    tensor::{Bool, DType, Tensor, TensorData, activation::softmax, backend::Backend},
};

use super::attention_probabilities;
use crate::precision::test_support::{
    assert_bitwise_eq, assert_f32_path_is_distinguishable, assert_matches_f64_reference,
    host_masked_softmax, near_tie_logits,
};

/// Logit-plane shape `[batch, head, query, key]` shared by the softmax fixtures.
const LOGIT_SHAPE: [usize; 4] = [1, 2, 3, 4];

/// Number of keys in each softmax row of [`LOGIT_SHAPE`].
const KEYS: usize = LOGIT_SHAPE[3];

/// Frozen copy of MLA's softmax before the evaluation-precision rule.
///
/// It promoted every logit dtype to F32 unconditionally. Keeping it verbatim
/// proves that F16, BF16, and F32 logits still run the identical operation
/// sequence and shows that F64 logits used to be narrowed. Callers pass a mask
/// with the exact logit geometry; shape validation is not part of the copy.
fn historical_attention_probabilities<B: Backend>(
    logits: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
) -> Tensor<B, 4> {
    if let Some(mask) = mask {
        let dtype = logits.dtype();
        let shape = logits.dims();
        let row_has_key = mask
            .clone()
            .bool_not()
            .int()
            .sum_dim(3)
            .greater_elem(0)
            .expand(shape);
        let probabilities = softmax(
            logits
                .cast(DType::F32)
                .mask_fill(mask.clone(), f32::NEG_INFINITY)
                .mask_fill(row_has_key.bool_not(), 0.0),
            3,
        )
        .mask_fill(mask, 0.0);
        probabilities.cast(dtype)
    } else {
        let dtype = logits.dtype();
        softmax(logits.cast(DType::F32), 3).cast(dtype)
    }
}

/// Exclusions covering a fully masked row, partial rows, and unmasked rows.
fn exclusion_pattern() -> Vec<bool> {
    (0..LOGIT_SHAPE.iter().product::<usize>())
        .map(|index| {
            let row = index / KEYS;
            let key = index % KEYS;
            row == 1 || (row % 3 == 2 && key % 2 == 1) || (row == 3 && key == 0)
        })
        .collect()
}

/// Creates logits with explicit storage dtype from host doubles.
fn logits_tensor(values: Vec<f64>, dtype: DType) -> Tensor<Flex, 4> {
    Tensor::<Flex, 4>::from_data(
        TensorData::new(values, LOGIT_SHAPE),
        (&Default::default(), DType::F64),
    )
    .cast(dtype)
}

/// Creates a boolean exclusion tensor with the logit-plane shape.
fn exclusion_tensor(values: Vec<bool>) -> Tensor<Flex, 4, Bool> {
    Tensor::from_bool(TensorData::new(values, LOGIT_SHAPE), &Default::default())
}

/// F64 latent-attention logits keep F64 softmax with and without masks.
///
/// The frozen historical graph narrows the same near-tie logits through F32
/// and misses the host reference by far more than F64 rounding.
#[test]
fn f64_latent_softmax_matches_an_independent_f64_reference() {
    let values = near_tie_logits(LOGIT_SHAPE.iter().product());
    for masked in [false, true] {
        let exclusion = masked.then(exclusion_pattern);
        let expected = host_masked_softmax(&values, exclusion.as_deref(), KEYS);
        let context = format!("masked={masked}");
        let logits = logits_tensor(values.clone(), DType::F64);
        let mask = exclusion.map(exclusion_tensor);

        let actual = attention_probabilities(logits.clone(), mask.clone())
            .expect("mask matches logits")
            .into_data();
        assert_eq!(actual.dtype, DType::F64, "{context}");
        assert_matches_f64_reference(&actual, &expected, 1e-6, &context);
        assert_f32_path_is_distinguishable(
            &historical_attention_probabilities(logits, mask).into_data(),
            &expected,
            1e-6,
            &context,
        );
    }
}

/// F16, BF16, and F32 latent-attention logits keep the historical FP32 graph bit for bit.
#[test]
fn reduced_and_single_precision_latent_softmax_match_the_historical_graph_bitwise() {
    let values = (0..LOGIT_SHAPE.iter().product::<usize>())
        .map(|index| ((index as f64 + 1.0) * 0.917).sin() * 6.0)
        .collect::<Vec<_>>();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        for masked in [false, true] {
            let mask = masked.then(|| exclusion_tensor(exclusion_pattern()));
            let logits = logits_tensor(values.clone(), dtype);
            let actual = attention_probabilities(logits.clone(), mask.clone())
                .expect("mask matches logits")
                .into_data();
            assert_eq!(actual.dtype, dtype);
            assert_bitwise_eq(
                &actual,
                &historical_attention_probabilities(logits, mask).into_data(),
                &format!("{dtype:?} masked={masked}"),
            );
        }
    }
}
