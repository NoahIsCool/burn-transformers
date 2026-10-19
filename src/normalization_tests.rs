//! Normalization tests for formula parity, finite-precision behavior, and configuration validation.

use burn::{
    backend::Flex,
    module::Param,
    tensor::{DType, Tensor, TensorData, Tolerance, f16},
};

use super::{RmsNormConfig, RmsNormRootMode, scale_free_rms_norm};

/// Conventional and zero-centered initialization both begin with unit scale.
#[test]
fn initialization_conventions_have_identical_functions() {
    let device = Default::default();
    let ordinary = RmsNormConfig::new(3)
        .init::<Flex>(&device)
        .expect("ordinary norm should initialize");
    let zero_centered = RmsNormConfig::new(3)
        .with_zero_centered(true)
        .init::<Flex>(&device)
        .expect("zero-centered norm should initialize");
    let input = Tensor::<Flex, 2>::from_floats([[1.0, 2.0, 3.0], [2.0, 0.0, -2.0]], &device);

    let ordinary = ordinary.forward(input.clone()).to_data();
    let zero_centered = zero_centered.forward(input).to_data();
    assert_eq!(ordinary, zero_centered);
}

/// Zero vectors remain finite because epsilon bounds the denominator.
#[test]
fn zero_input_has_no_nan() {
    let device = Default::default();
    let norm = RmsNormConfig::new(4)
        .init::<Flex>(&device)
        .expect("norm should initialize");
    let output = norm
        .forward(Tensor::<Flex, 3>::zeros([2, 3, 4], &device))
        .to_data();
    assert_eq!(output, burn::tensor::TensorData::zeros::<f32, _>([2, 3, 4]));
}

/// FP32 checkpoint scales are cast back to an F16 activation's dtype.
#[test]
fn mixed_precision_scale_preserves_activation_dtype() {
    use burn::tensor::DType;

    let device = Default::default();
    let norm = RmsNormConfig::new(4)
        .init::<Flex>(&device)
        .expect("norm should initialize");
    let input = Tensor::<Flex, 2>::ones([2, 4], (&device, DType::F16));
    let output = norm.forward(input);
    assert_eq!(output.dtype(), DType::F16);
}

/// Architectures can round before reduced-precision learned scaling.
#[test]
fn round_before_scale_matches_early_rounding_operation_order() {
    let device = Default::default();
    let mut early = RmsNormConfig::new(3)
        .with_round_before_scale(true)
        .init::<Flex>(&device)
        .unwrap();
    let mut late = RmsNormConfig::new(3).init::<Flex>(&device).unwrap();
    let weight = Tensor::<Flex, 1>::from_floats([0.731, 1.337, -0.419], &device).cast(DType::F16);
    early.weight = Param::from_tensor(weight.clone());
    late.weight = Param::from_tensor(weight.clone());
    let input = Tensor::<Flex, 2>::from_floats([[0.337, 1.719, -2.113]], &device).cast(DType::F16);

    let fp32 = input.clone().cast(DType::F32);
    let rms = (fp32.clone().square().mean_dim(1) + 1e-6).sqrt();
    let expected = (fp32 / rms).cast(DType::F16) * weight.unsqueeze();
    let actual = early.forward(input.clone());
    assert_eq!(actual.to_data(), expected.to_data());
    assert_ne!(actual.to_data(), late.forward(input).to_data());
}

/// Muse inverse-root graphs reproduce adversarial pinned F16 oracle bits.
#[test]
fn muse_inverse_root_modes_match_pinned_f16_oracles() {
    let device = Default::default();
    let input =
        Tensor::<Flex, 2>::from_floats([[0.096_557_62, 2.664_062_5]], &device).cast(DType::F16);

    let power = RmsNormConfig::new(2)
        .with_epsilon(1e-5)
        .with_root_mode(RmsNormRootMode::MultiplyByPower)
        .init::<Flex>(&device)
        .unwrap();
    let divided = RmsNormConfig::new(2)
        .with_epsilon(1e-5)
        .init::<Flex>(&device)
        .unwrap();
    let power_bits = power
        .forward(input.clone())
        .into_data()
        .to_vec::<f16>()
        .unwrap()
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();
    let divided_bits = divided
        .forward(input.clone())
        .into_data()
        .to_vec::<f16>()
        .unwrap()
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(power_bits, [0x2a8f, 0x3da7]);
    assert_eq!(divided_bits, [0x2a8e, 0x3da7]);

    let centered_weight =
        Tensor::<Flex, 1>::from_floats([-0.875, 2.529_296_9], &device).cast(DType::F16);
    let mut reciprocal = RmsNormConfig::new(2)
        .with_epsilon(1e-5)
        .with_zero_centered(true)
        .with_root_mode(RmsNormRootMode::MultiplyByReciprocalSqrt)
        .init::<Flex>(&device)
        .unwrap();
    reciprocal.weight = Param::from_tensor(centered_weight.clone());
    let mut divided_centered = RmsNormConfig::new(2)
        .with_epsilon(1e-5)
        .with_zero_centered(true)
        .init::<Flex>(&device)
        .unwrap();
    divided_centered.weight = Param::from_tensor(centered_weight);
    let reciprocal_bits = reciprocal
        .forward(input.clone())
        .into_data()
        .to_vec::<f16>()
        .unwrap()
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();
    let divided_centered_bits = divided_centered
        .forward(input)
        .into_data()
        .to_vec::<f16>()
        .unwrap()
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(reciprocal_bits, [0x1e8f, 0x44fd]);
    assert_eq!(divided_centered_bits, [0x1e8e, 0x44fd]);
}

/// T5 normalization divides by RMS without subtracting the feature mean.
#[test]
fn t5_scale_only_norm_matches_closed_form() {
    let device = Default::default();
    let mut norm = RmsNormConfig::new(2)
        .with_epsilon(1e-6)
        .init::<Flex>(&device)
        .expect("valid T5 norm");
    norm.weight = Param::from_tensor(Tensor::from_floats([2.0, 0.5], &device));
    let input = Tensor::<Flex, 2>::from_floats([[3.0, 4.0]], &device);
    let denominator = (12.5_f32 + 1e-6).sqrt();
    norm.forward(input).into_data().assert_approx_eq::<f32>(
        &TensorData::from([[6.0 / denominator, 2.0 / denominator]]),
        Tolerance::absolute(1e-6),
    );
}

/// Invalid dimensions fail before an initializer reaches the backend.
#[test]
fn invalid_config_is_rejected() {
    let error = RmsNormConfig::new(0)
        .validate()
        .expect_err("zero width must fail");
    assert!(error.to_string().contains("rms_norm.size"));

    // Non-finite, zero, and negative epsilons can make the denominator
    // sqrt(mean(x^2) + eps) zero, NaN, or infinite.
    for epsilon in [f64::NAN, f64::INFINITY, 0.0, -1e-6] {
        let error = RmsNormConfig::new(4)
            .with_epsilon(epsilon)
            .validate()
            .expect_err("invalid epsilon must fail");
        assert!(
            matches!(&error, crate::Error::Configuration { field, reason }
                if field == "rms_norm.epsilon" && reason == "must be finite and greater than zero"),
            "{epsilon}: {error:?}"
        );
    }
    assert!(
        RmsNormConfig::new(4)
            .with_epsilon(f64::MIN_POSITIVE)
            .validate()
            .is_ok()
    );
}

/// Scale-free normalization equals a unit-scale RmsNorm and the closed form, in every root graph.
#[test]
fn scale_free_rms_norm_matches_unit_scale_norm_and_closed_form() {
    let device = Default::default();
    let input = Tensor::<Flex, 2>::from_floats([[3.0, -4.0, 0.0], [1.0, 1.0, 1.0]], &device);
    for root_mode in [
        RmsNormRootMode::DivideBySqrt,
        RmsNormRootMode::MultiplyByPower,
        RmsNormRootMode::MultiplyByReciprocalSqrt,
    ] {
        let unit = RmsNormConfig::new(3)
            .with_epsilon(1e-6)
            .with_root_mode(root_mode)
            .init::<Flex>(&device)
            .unwrap()
            .forward(input.clone());
        let scale_free = scale_free_rms_norm(input.clone(), 1e-6, root_mode).unwrap();
        assert_eq!(scale_free.clone().into_data(), unit.into_data());
        let rms = (25.0_f32 / 3.0 + 1e-6).sqrt();
        scale_free.into_data().assert_approx_eq::<f32>(
            &TensorData::from([[3.0 / rms, -4.0 / rms, 0.0], [1.0, 1.0, 1.0]]),
            Tolerance::absolute(1e-5),
        );
    }
}

/// The power-root graph reproduces the scale-free formula used by Muse-style norms exactly.
#[test]
fn scale_free_power_root_matches_reference_operation_order() {
    let device = Default::default();
    let input =
        Tensor::<Flex, 3>::from_floats([[[0.3, -1.7, 2.9, 0.01]]], &device).cast(DType::F16);
    let reference = {
        let float = input.clone().cast(DType::F32);
        let inverse = (float.clone().square().mean_dim(2) + 1e-6).powf_scalar(-0.5);
        (float * inverse).cast(DType::F16)
    };
    let actual = scale_free_rms_norm(input, 1e-6, RmsNormRootMode::MultiplyByPower).unwrap();
    assert_eq!(actual.dtype(), DType::F16);
    assert_eq!(actual.into_data(), reference.into_data());
}

/// Non-finite and non-positive stabilizers are rejected with context.
#[test]
fn scale_free_rms_norm_rejects_invalid_epsilon() {
    let device = Default::default();
    for epsilon in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let error = scale_free_rms_norm(
            Tensor::<Flex, 1>::ones([2], &device),
            epsilon,
            RmsNormRootMode::DivideBySqrt,
        )
        .unwrap_err();
        assert!(error.to_string().contains("rms_norm.epsilon"));
    }
}

/// Rounds an F32 value to a 16-bit float `dtype`, to nearest with ties to even, as PyTorch's casts do.
fn round_half(dtype: DType, value: f32) -> f32 {
    match dtype {
        DType::BF16 => burn::tensor::bf16::from_f32(value).to_f32(),
        DType::F16 => f16::from_f32(value).to_f32(),
        other => unreachable!("no 16-bit float {other:?}"),
    }
}

/// Returns how many representable values of the 16-bit float `dtype` lie between `a` and `b` (both representable).
///
/// The bit patterns are mapped to a monotone integer line (negative values
/// mirrored below zero), so adjacent values are one apart and `+0` and `-0`
/// coincide.
fn units_apart(dtype: DType, a: f32, b: f32) -> u32 {
    let ordered = |value: f32| -> i32 {
        let bits = i32::from(match dtype {
            DType::BF16 => burn::tensor::bf16::from_f32(value).to_bits(),
            _ => f16::from_f32(value).to_bits(),
        });
        if bits & 0x8000 != 0 {
            -(bits & 0x7fff)
        } else {
            bits
        }
    };
    ordered(a).abs_diff(ordered(b))
}

/// The reference RMSNorm of one row: the unit-RMS activation computed in F64 and rounded to F32, then scaled.
///
/// Transformers' Llama-family `RMSNorm` computes `x * rsqrt(mean(x^2) + eps)`
/// in F32, casts that to the activation dtype, and multiplies by the weight
/// in the activation dtype (`round_before_scale`, one rounding of the
/// product, which is exact in F32 for two 16-bit values). Gemma's
/// zero-centered `RMSNorm` keeps `(1 + w) * x_f32 / rms` in F32 and rounds
/// once. The F64 evaluation stands in for any F32 reduction order: they agree
/// within a few F32 units, far below one unit of either 16-bit format.
fn reference_rms_norm(
    dtype: DType,
    row: &[f32],
    weight: &[f32],
    epsilon: f64,
    round_before_scale: bool,
    zero_centered: bool,
) -> Vec<f32> {
    let mean_square = row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / row.len() as f64;
    let inverse = (mean_square + epsilon).sqrt().recip();
    row.iter()
        .zip(weight)
        .map(|(&x, &w)| {
            let unit = (f64::from(x) * inverse) as f32;
            if round_before_scale {
                round_half(dtype, round_half(dtype, unit) * w)
            } else {
                let scale = if zero_centered { 1.0 + w } else { w };
                round_half(dtype, unit * scale)
            }
        })
        .collect()
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(128))]

    /// Half-precision RMSNorm lies within one unit in the last place of the references' rounding points, for both conventions.
    ///
    /// `round_before_scale` is the Llama-family graph (normalize in F32,
    /// round, scale in the activation dtype); without it, zero-centered, it is
    /// Gemma's (scale in F32, round once). One unit covers what may differ: the
    /// F32 reduction order and root graph move the unit-RMS value by a few F32
    /// units, which changes its rounding only at a near-tie, and Flex's F32
    /// cast can decide a near-tie the other way (see the cast pin in
    /// `feed_forward_tests.rs`). A norm evaluated in the activation dtype
    /// misses it by many units (`half_precision_rms_norm_arithmetic_misses_the_bound`).
    #[test]
    fn half_precision_rms_norm_follows_the_reference_rounding_points(
        width in 1_usize..96,
        seeds in proptest::collection::vec(-30.0_f32..30.0, 96),
        weights in proptest::collection::vec(-2.0_f32..2.0, 96),
        bf16 in proptest::bool::ANY,
        round_before_scale in proptest::bool::ANY,
    ) {
        let dtype = if bf16 { DType::BF16 } else { DType::F16 };
        let zero_centered = !round_before_scale;
        let device = Default::default();
        let row: Vec<f32> = seeds[..width].iter().map(|&x| round_half(dtype, x)).collect();
        let weight: Vec<f32> = weights[..width].iter().map(|&w| round_half(dtype, w)).collect();
        let mut norm = RmsNormConfig::new(width)
            .with_round_before_scale(round_before_scale)
            .with_zero_centered(zero_centered)
            .init::<Flex>(&device)
            .unwrap();
        norm.weight = Param::from_tensor(
            Tensor::<Flex, 1>::from_data(TensorData::new(weight.clone(), [width]), &device).cast(dtype),
        );
        let input = Tensor::<Flex, 2>::from_data(TensorData::new(row.clone(), [1, width]), &device).cast(dtype);
        let actual = norm.forward(input);
        proptest::prop_assert_eq!(actual.dtype(), dtype);
        let actual = actual.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let expected = reference_rms_norm(dtype, &row, &weight, norm.epsilon, round_before_scale, zero_centered);
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            proptest::prop_assert!(
                units_apart(dtype, actual, expected) <= 1,
                "{:?} entry {} of {}: {} against the reference {}", dtype, index, width, actual, expected
            );
        }
    }
}

/// A norm whose mean square accumulates in BF16 misses the one-unit bound by far, so the bound detects that defect.
///
/// Summing 4096 unit squares one at a time in BF16 stalls at `256`, where
/// adding one is a tie that rounds back to `256`; the mean square is then
/// `1/16` instead of `1`, and the unit-RMS value `4` instead of `1`.
#[test]
fn half_precision_rms_norm_arithmetic_misses_the_bound() {
    let row = vec![1.0_f32; 4096];
    let weight = vec![1.0_f32; 4096];
    let mut sum = 0.0_f32;
    for &x in &row {
        sum = round_half(DType::BF16, sum + round_half(DType::BF16, x * x));
    }
    assert_eq!(sum, 256.0);
    let defect = round_half(DType::BF16, 1.0 / (sum / 4096.0).sqrt());
    let reference = reference_rms_norm(DType::BF16, &row, &weight, 1e-6, true, false)[0];
    assert!(units_apart(DType::BF16, defect, reference) > 1);
    let device = Default::default();
    let norm = RmsNormConfig::new(4096)
        .with_round_before_scale(true)
        .init::<Flex>(&device)
        .unwrap();
    let actual = norm
        .forward(Tensor::<Flex, 2>::ones([1, 4096], &device).cast(DType::BF16))
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(
        actual
            .iter()
            .all(|&value| units_apart(DType::BF16, value, reference) <= 1)
    );
}
