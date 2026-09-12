//! Normalization tests for formula parity, finite-precision behavior, and configuration validation.

use burn::{
    backend::Flex,
    module::Param,
    tensor::{DType, Tensor, TensorData, Tolerance, f16},
};

use super::{RmsNormConfig, RmsNormRootMode};

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

    let error = RmsNormConfig::new(4)
        .with_epsilon(f64::NAN)
        .validate()
        .expect_err("NaN epsilon must fail");
    assert!(error.to_string().contains("rms_norm.epsilon"));
}
