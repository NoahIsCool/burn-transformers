//! Tests for dtype-exact scalar multipliers and tanh soft caps.

use burn::tensor::{DType, Tensor, TensorData, Tolerance, bf16, f16};
use proptest::prelude::*;

use super::*;
use crate::models::{
    gemma::{GemmaArchitecture, GemmaConfig, GemmaVariant},
    shared::test_support::TestBackend,
};

/// Every floating dtype a multiplier is rounded to.
const FLOAT_DTYPES: [DType; 4] = [DType::F64, DType::F32, DType::F16, DType::BF16];

/// Reads a rank-1 F32 tensor back to the host.
fn values(tensor: Tensor<TestBackend, 1>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

/// Rounds a finite `f64` of magnitude at least `1e-3` directly to the nearest F16 value.
///
/// Test oracle for PyTorch's `torch.tensor(value, dtype=torch.float16)`,
/// independent of [`round_to_dtype`]: rounding through F32 errs by at most one
/// F16 ulp, so the three F16 values around that candidate contain the nearest
/// one, and each distance is an exact `f64` difference (Sterbenz's lemma).
/// Equal distances pick the even significand.
fn nearest_f16(value: f64) -> f64 {
    let magnitude = value.abs();
    let bits = f16::from_f32(magnitude as f32).to_bits();
    let nearest = [bits - 1, bits, bits + 1]
        .into_iter()
        .min_by(|&left, &right| {
            let distance = |bits| (f16::from_bits(bits).to_f64() - magnitude).abs();
            distance(left)
                .total_cmp(&distance(right))
                .then((left & 1).cmp(&(right & 1)))
        })
        .unwrap();
    f16::from_bits(nearest).to_f64().copysign(value)
}

/// Whether `value` (of magnitude at least `1e-3`) lies exactly halfway between two F16 values.
fn lies_on_f16_tie(value: f64) -> bool {
    let magnitude = value.abs();
    let bits = f16::from_f32(magnitude as f32).to_bits();
    let mut distances =
        [bits - 1, bits, bits + 1].map(|bits| (f16::from_bits(bits).to_f64() - magnitude).abs());
    distances.sort_by(f64::total_cmp);
    distances[0] == distances[1]
}

/// `sqrt(3072)` rounds to each dtype as the Gemma references' normalizer tensors do.
///
/// The BF16 value 55.5 is the one the Hugging Face Gemma code comments on;
/// the F16 value matches `torch.tensor(3072 ** 0.5, dtype=torch.float16)`.
#[test]
fn scalars_round_like_reference_tensors() {
    let sqrt_3072 = 3072f64.sqrt();
    assert_eq!(round_to_dtype(sqrt_3072, DType::BF16), 55.5);
    assert_eq!(round_to_dtype(sqrt_3072, DType::F16), 55.4375);
    assert_eq!(nearest_f16(sqrt_3072), 55.4375);
    assert_eq!(
        round_to_dtype(sqrt_3072, DType::F32),
        f64::from(sqrt_3072 as f32)
    );
    assert_eq!(round_to_dtype(sqrt_3072, DType::F64), sqrt_3072);
    // Non-floating dtypes keep the F32 value.
    assert_eq!(
        round_to_dtype(sqrt_3072, DType::I32),
        f64::from(sqrt_3072 as f32)
    );
}

/// Half-precision multipliers round through F32, which can double-round at an F16 tie.
///
/// Each value lies just above a tie of the target format, by less than F32
/// resolution, so its F32 rounding lands exactly on the tie, which rounds to
/// the even `1.0`. For BF16 this is also PyTorch 2.10's result
/// (`torch.tensor(y, dtype=torch.bfloat16) == 1.0`); for F16 PyTorch rounds
/// directly and gets `1.0009765625`, the documented divergence.
#[test]
fn half_precision_multipliers_round_through_f32() {
    let tiny = 2f64.powi(-40);
    let x = 1.0 + 2f64.powi(-11) + tiny;
    assert!(lies_on_f16_tie(f64::from(x as f32)));
    assert_eq!(round_to_dtype(x, DType::F16), 1.0);
    assert_eq!(nearest_f16(x), 1.000_976_562_5);

    let y = 1.0 + 2f64.powi(-8) + tiny;
    assert_eq!(round_to_dtype(y, DType::BF16), 1.0);
    // `y` is nearer the upper BF16 neighbour, so only rounding through F32 gives `1.0`.
    assert!(1.007_812_5 - y < y - 1.0);
}

/// The Gemma family's multiplier is the shared rounding of `sqrt(d)`'s F32 value.
///
/// For every shipped Gemma 2 and Gemma 3 width the F32, F16, and BF16 values
/// also equal the shared rounding of the exact root and, in F16, PyTorch's
/// direct rounding, so the Gemma 2 references agree too. Only F64 differs:
/// the family keeps the F32 value (Gemma 3's buffer), the shared helper the
/// exact root (Gemma 2's `torch.tensor(..., dtype=torch.float64)`).
#[test]
fn gemma_family_scalars_are_the_shared_rounding_of_an_f32_value() {
    let base = GemmaConfig::new(GemmaVariant::Gemma2)
        .architecture()
        .unwrap();
    for hidden_size in [
        8, 16, 640, 1152, 1536, 2048, 2304, 2560, 3072, 3584, 4096, 5376,
    ] {
        let architecture = GemmaArchitecture {
            hidden_size,
            ..base.clone()
        };
        let exact = (hidden_size as f64).sqrt();
        for dtype in FLOAT_DTYPES {
            let family = architecture.embedding_scale(dtype);
            let context = format!("{hidden_size} {dtype:?}");
            assert_eq!(
                round_to_dtype(f64::from(exact as f32), dtype),
                family,
                "{context}"
            );
            if dtype == DType::F64 {
                assert_eq!(round_to_dtype(exact, dtype), exact, "{context}");
            } else {
                assert_eq!(round_to_dtype(exact, dtype), family, "{context}");
            }
        }
        assert_eq!(
            round_to_dtype(exact, DType::F16),
            nearest_f16(exact),
            "{hidden_size}"
        );
    }
}

proptest! {
    /// Rounded multipliers are representable fixed points within (double-rounded) half an ulp.
    #[test]
    fn rounded_scalars_are_representable_fixed_points(
        magnitude in 1e-3f64..6.0e4,
        negative in any::<bool>(),
    ) {
        let value = if negative { -magnitude } else { magnitude };
        for dtype in FLOAT_DTYPES {
            let rounded = round_to_dtype(value, dtype);
            prop_assert_eq!(round_to_dtype(rounded, dtype), rounded);
            let (representable, relative) = match dtype {
                DType::F64 => (rounded, 0.0),
                DType::F32 => (f64::from(rounded as f32), 2f64.powi(-24)),
                // Rounding through F32 adds at most an F32 half-ulp.
                DType::F16 => (
                    f16::from_f32(rounded as f32).to_f64(),
                    2f64.powi(-11) + 2f64.powi(-24),
                ),
                _ => (
                    bf16::from_f32(rounded as f32).to_f64(),
                    2f64.powi(-8) + 2f64.powi(-24),
                ),
            };
            prop_assert_eq!(representable, rounded);
            prop_assert!((rounded - value).abs() <= relative * value.abs());
            prop_assert_eq!(rounded.is_sign_negative(), negative);
        }
        // F16 agrees with PyTorch's direct rounding unless the F32 value is an F16 tie.
        prop_assert!(
            round_to_dtype(value, DType::F16) == nearest_f16(value)
                || lies_on_f16_tie(f64::from(value as f32))
        );
    }

    /// Soft caps are bounded by the cap, odd, and never enlarge a logit beyond rounding.
    #[test]
    fn soft_caps_are_bounded_odd_and_contracting(
        logit in -1.0e4f32..1.0e4,
        cap in 1e-2f64..1.0e3,
    ) {
        let device = Default::default();
        let capped = values(soft_cap(
            Tensor::<TestBackend, 1>::from_floats([logit, -logit], &device),
            Some(cap),
        ));
        prop_assert!(capped[0].abs() <= cap as f32);
        prop_assert_eq!(capped[1], -capped[0]);
        prop_assert!(capped[0].abs() <= logit.abs() * (1.0 + 1e-6));
        prop_assert!(capped[0] == 0.0 || capped[0].signum() == logit.signum());
    }
}

/// The cap equals `c · tanh(z / c)`, saturates at `±c`, is monotone, and is the identity near zero.
#[test]
fn soft_cap_matches_the_reference_formula() {
    let device = Default::default();
    let cap = 5.0;
    let logits = [-1000.0f32, -7.5, -1e-3, 0.0, 1e-3, 2.5, 7.5, 1000.0];
    let capped = values(soft_cap(
        Tensor::<TestBackend, 1>::from_floats(logits, &device),
        Some(cap),
    ));
    for (&logit, &output) in logits.iter().zip(&capped) {
        let expected = cap * (f64::from(logit) / cap).tanh();
        assert!(
            (f64::from(output) - expected).abs() <= 1e-6,
            "{logit}: {output} != {expected}"
        );
        assert!(output.abs() <= cap as f32);
    }
    assert_eq!(capped[0], -5.0);
    assert_eq!(capped[7], 5.0);
    assert!(capped.windows(2).all(|pair| pair[0] <= pair[1]));
    assert!((capped[4] - 1e-3).abs() <= 1e-9);
}

/// An absent cap returns the logits bit for bit.
#[test]
fn absent_soft_cap_is_the_identity() {
    let device = Default::default();
    let logits = Tensor::<TestBackend, 3>::from_floats([[[-1e6, -2.5, 0.0, 3.25, 7e5]]], &device);
    soft_cap(logits.clone(), None)
        .into_data()
        .assert_eq(&logits.into_data(), true);
}

/// Half-precision logits stay F16 and keep their shape while being capped.
#[test]
fn soft_cap_keeps_the_shape_and_half_precision_dtype() {
    let device = Default::default();
    let logits =
        Tensor::<TestBackend, 2>::from_floats([[-40.0, -1.0, 0.5], [3.0, 12.0, 90.0]], &device)
            .cast(DType::F16);
    let capped = soft_cap(logits, Some(30.0));
    assert_eq!(capped.dtype(), DType::F16);
    assert_eq!(capped.dims(), [2, 3]);
    let expected =
        [-40.0f64, -1.0, 0.5, 3.0, 12.0, 90.0].map(|logit| (30.0 * (logit / 30.0).tanh()) as f32);
    capped.cast(DType::F32).into_data().assert_approx_eq::<f32>(
        &TensorData::new(expected.to_vec(), [2, 3]),
        Tolerance::absolute(0.05),
    );
}

/// Gradients of the cap are `1 - tanh²(z / c)`.
#[cfg(feature = "autodiff")]
#[test]
fn soft_cap_gradient_is_one_minus_tanh_squared() {
    use burn::backend::Autodiff;

    /// Reverse-mode wrapper around the deterministic Flex test backend.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let cap = 2.0;
    let logits = [-3.0f32, 0.0, 0.5, 4.0];
    let input = Tensor::<AdBackend, 1>::from_floats(logits, &device).require_grad();
    let gradients = soft_cap(input.clone(), Some(cap)).sum().backward();
    let gradient = input
        .grad(&gradients)
        .expect("the cap is differentiable")
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for (&logit, &actual) in logits.iter().zip(&gradient) {
        let expected = 1.0 - (f64::from(logit) / cap).tanh().powi(2);
        assert!(
            (f64::from(actual) - expected).abs() <= 1e-6,
            "{logit}: {actual} != {expected}"
        );
    }
}
