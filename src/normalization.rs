//! Transformer normalization variants with explicit checkpoint conventions.
//!
//! RMS normalization rescales a vector without subtracting its mean:
//! `y = x / sqrt(mean(x^2) + epsilon) * scale`. Computing the reduction in
//! FP32 is important for F16 inference because squaring and summing a wide
//! residual stream otherwise loses significant dynamic range. Some recent
//! checkpoints store a zero-centered learned parameter and define
//! `scale = 1 + weight`; older checkpoints store `scale` directly.

use burn::{
    module::{Initializer, Module, Param},
    tensor::{DType, Tensor, backend::Backend},
};

use crate::{Error, Result};

/// Configuration for checkpoint-compatible RMS normalization.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RmsNormConfig {
    /// Width of the normalized final tensor axis.
    pub size: usize,
    /// Positive stabilizer added before the reciprocal square root.
    pub epsilon: f64,
    /// Whether the checkpoint parameter represents `scale - 1`.
    pub zero_centered: bool,
    /// Whether to round the normalized activation before learned scaling.
    ///
    /// DeepSeek casts the unit-RMS activation back to the residual dtype and
    /// then multiplies by its parameter. Qwen keeps both operations in FP32
    /// and rounds only the final result, so the distinction is explicit.
    #[serde(default)]
    pub round_before_scale: bool,
}

impl RmsNormConfig {
    /// Creates a conventional RMSNorm configuration with epsilon `1e-6`.
    pub fn new(size: usize) -> Self {
        Self {
            size,
            epsilon: 1e-6,
            zero_centered: false,
            round_before_scale: false,
        }
    }

    /// Selects the numerical stabilizer used by the reciprocal root.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Selects the `1 + weight` checkpoint convention.
    pub fn with_zero_centered(mut self, zero_centered: bool) -> Self {
        self.zero_centered = zero_centered;
        self
    }

    /// Selects DeepSeek-style rounding before learned scaling.
    pub fn with_round_before_scale(mut self, round_before_scale: bool) -> Self {
        self.round_before_scale = round_before_scale;
        self
    }

    /// Validates dimensions and numerical constants without allocating tensors.
    pub fn validate(&self) -> Result<()> {
        if self.size == 0 {
            return Err(Error::configuration("rms_norm.size", "must be non-zero"));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::configuration(
                "rms_norm.epsilon",
                "must be finite and greater than zero",
            ));
        }
        Ok(())
    }

    /// Initializes a trainable scale using the selected checkpoint convention.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<RmsNorm<B>> {
        self.validate()?;
        let initializer = if self.zero_centered {
            Initializer::Zeros
        } else {
            Initializer::Ones
        };
        Ok(RmsNorm {
            weight: initializer.init([self.size], device),
            epsilon: self.epsilon,
            zero_centered: self.zero_centered,
            round_before_scale: self.round_before_scale,
        })
    }
}

/// RMS normalization whose parameter semantics match both T5 and Qwen3.8.
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Learned scale, or learned scale minus one for zero-centered checkpoints.
    pub weight: Param<Tensor<B, 1>>,
    /// Stabilizer added to the FP32 mean square.
    pub epsilon: f64,
    /// Whether to add one to the learned parameter before scaling.
    pub zero_centered: bool,
    /// Whether the normalized activation is rounded before scale multiplication.
    pub round_before_scale: bool,
}

impl<B: Backend> RmsNorm<B> {
    /// Normalizes the last axis of a tensor of arbitrary rank.
    ///
    /// The input dtype is restored after the FP32 reduction so this method
    /// remains compatible with F16 Metal execution while preserving an FP32
    /// variance estimate. By default, normalization and learned scaling remain
    /// in FP32 until the final cast; this matches zero-centered Qwen RMSNorm.
    /// [`RmsNormConfig::with_round_before_scale`] instead reproduces the
    /// DeepSeek convention of rounding the normalized activation first.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = input.dtype();
        let input = input.cast(DType::F32);
        let rms = (input.clone().square().mean_dim(D - 1) + self.epsilon).sqrt();
        let scale = if self.zero_centered {
            self.weight.val().cast(DType::F32) + 1.0
        } else {
            self.weight.val().cast(DType::F32)
        };
        let normalized = input / rms;
        if self.round_before_scale {
            normalized.cast(dtype) * scale.cast(dtype).unsqueeze()
        } else {
            (normalized * scale.unsqueeze()).cast(dtype)
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        module::Param,
        tensor::{Tensor, TensorData, Tolerance},
    };

    use super::RmsNormConfig;

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

    /// DeepSeek rounds the normalized activation before reduced-precision scaling.
    #[test]
    fn round_before_scale_matches_deepseek_operation_order() {
        use burn::tensor::DType;

        let device = Default::default();
        let mut early = RmsNormConfig::new(3)
            .with_round_before_scale(true)
            .init::<Flex>(&device)
            .unwrap();
        let mut late = RmsNormConfig::new(3).init::<Flex>(&device).unwrap();
        let weight =
            Tensor::<Flex, 1>::from_floats([0.731, 1.337, -0.419], &device).cast(DType::F16);
        early.weight = Param::from_tensor(weight.clone());
        late.weight = Param::from_tensor(weight.clone());
        let input =
            Tensor::<Flex, 2>::from_floats([[0.337, 1.719, -2.113]], &device).cast(DType::F16);

        let fp32 = input.clone().cast(DType::F32);
        let rms = (fp32.clone().square().mean_dim(1) + 1e-6).sqrt();
        let expected = (fp32 / rms).cast(DType::F16) * weight.unsqueeze();
        let actual = early.forward(input.clone());
        assert_eq!(actual.to_data(), expected.to_data());
        assert_ne!(actual.to_data(), late.forward(input).to_data());
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
}
