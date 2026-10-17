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
