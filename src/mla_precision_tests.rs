//! Latent-attention evaluation-precision tests: F64 references and historical F32 bit identity.

use burn::{
    backend::Flex,
    module::{Module, ModuleMapper, Param},
    tensor::{Bool, DType, Int, Tensor, TensorData, activation::softmax, backend::Backend},
};

use super::{MultiLatentAttention, MultiLatentAttentionConfig, attention_probabilities};
use crate::position::{RopeConfig, RopeScaling, inverse_frequencies_f64};
use crate::precision::test_support::{
    assert_bitwise_eq, assert_f32_path_is_distinguishable, assert_matches_f64_reference,
    f64_values, host_masked_softmax, near_tie_logits,
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

/// Replaces every float parameter with deterministic F64 values.
///
/// Matrices receive distinct non-dyadic sinusoidal entries by traversal order;
/// vectors such as normalization scales are one.
#[derive(Default)]
struct DeterministicF64Parameters(
    /// Number of floating parameters already replaced.
    usize,
);

impl ModuleMapper<Flex> for DeterministicF64Parameters {
    /// Materializes one deterministic F64 parameter per visit.
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<Flex, D>>,
    ) -> Param<Tensor<Flex, D>> {
        self.0 += 1;
        let ordinal = self.0;
        param.map(|tensor| {
            let shape = tensor.dims();
            let values = (0..shape.iter().product::<usize>())
                .map(|index| {
                    if D == 1 {
                        1.0
                    } else {
                        (((index + 1) * (ordinal + 2)) as f64 * 0.618).sin() * 0.3
                    }
                })
                .collect::<Vec<_>>();
            Tensor::from_data(
                TensorData::new(values, shape),
                (&tensor.device(), DType::F64),
            )
        })
    }
}

/// Builds an F64 latent-attention layer with a four-channel rotary head.
fn f64_latent_attention() -> MultiLatentAttention<Flex> {
    MultiLatentAttentionConfig::new(8, 2, 3, 4, 4, 5, None)
        .init::<Flex>(&Default::default())
        .expect("valid MLA")
        .map(&mut DeterministicF64Parameters::default())
}

/// Positions beyond one million, where F32 rotary phases are visibly wrong.
const LARGE_POSITIONS: [[i32; 3]; 1] = [[1_000_003, 1_000_517, 1_048_573]];

/// Rotary configuration matching the four-channel latent rotary head.
fn latent_rope() -> RopeConfig {
    RopeConfig::new(4, 4, 10_000.0, RopeScaling::None).expect("valid rope")
}

/// Host split-half rotation of `[.., token, 4]` rows by `p * omega_i` in F64.
fn host_rotate(values: &[f64], positions: &[i32], frequencies: &[f64]) -> Vec<f64> {
    values
        .chunks_exact(4)
        .enumerate()
        .flat_map(|(row, channels)| {
            let position = f64::from(positions[row % positions.len()]);
            (0..4).map(move |channel| {
                let pair = channel % 2;
                let angle = position * frequencies[pair];
                if channel < 2 {
                    channels[channel] * angle.cos() - channels[channel + 2] * angle.sin()
                } else {
                    channels[channel] * angle.cos() + channels[channel - 2] * angle.sin()
                }
            })
        })
        .collect()
}

/// F64 latent projections rotate their positional channels with F64 tables.
///
/// Public `with_rotary` matches a host F64 rotation of the unrotated projection,
/// and the crate's checkpoint projection path produces the identical tensors.
#[test]
fn f64_latent_rotary_channels_match_a_host_f64_rotation() {
    let device = Default::default();
    let mla = f64_latent_attention();
    let hidden = Tensor::<Flex, 3>::from_data(
        TensorData::new(
            (0..3 * 8)
                .map(|index| ((index as f64 + 1.0) * 0.431).cos())
                .collect::<Vec<_>>(),
            [1, 3, 8],
        ),
        (&device, DType::F64),
    );
    let positions = || Tensor::<Flex, 2, Int>::from_ints(LARGE_POSITIONS, &device);
    let context = 1_048_574;
    let frequencies = inverse_frequencies_f64(latent_rope(), context).expect("valid context");
    let unrotated = mla.project_self(hidden.clone()).expect("valid projection");
    let rotated = unrotated
        .clone()
        .with_rotary(positions(), positions(), latent_rope(), context)
        .expect("valid rotation");
    for (actual, raw, name) in [
        (&rotated.query_rotary, &unrotated.query_rotary, "query"),
        (&rotated.rotated_key, &unrotated.rotated_key, "key"),
    ] {
        let expected = host_rotate(
            &f64_values(&raw.to_data()),
            &LARGE_POSITIONS[0],
            &frequencies,
        );
        let actual = actual.to_data();
        assert_eq!(actual.dtype, DType::F64, "{name}");
        assert_matches_f64_reference(&actual, &expected, 1.0, name);
        let legacy = crate::position::rotary_angles(latent_rope(), positions(), context)
            .expect("F32 tables");
        let (narrowed, _) =
            crate::position::apply_rotary(raw.clone(), raw.clone(), &legacy).expect("rotation");
        assert_f32_path_is_distinguishable(&narrowed.into_data(), &expected, 1.0, name);
    }

    // The checkpoint projection path builds its own tables; it must agree exactly.
    let checkpoint_path = mla
        .project_rotary(hidden, positions(), latent_rope(), context)
        .expect("valid checkpoint projection");
    assert_bitwise_eq(
        &checkpoint_path.query_rotary.into_data(),
        &rotated.query_rotary.into_data(),
        "checkpoint query",
    );
    assert_bitwise_eq(
        &checkpoint_path.rotated_key.into_data(),
        &rotated.rotated_key.into_data(),
        "checkpoint key",
    );
}
