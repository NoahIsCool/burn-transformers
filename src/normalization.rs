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
}

impl RmsNormConfig {
    /// Creates a conventional RMSNorm configuration with epsilon `1e-6`.
    pub fn new(size: usize) -> Self {
        Self {
            size,
            epsilon: 1e-6,
            zero_centered: false,
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
}

impl<B: Backend> RmsNorm<B> {
    /// Normalizes the last axis of a tensor of arbitrary rank.
    ///
    /// The input dtype is restored after the FP32 reduction so this method
    /// remains compatible with F16 Metal execution while preserving an FP32
    /// variance estimate.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = input.dtype();
        let rms = (input.clone().cast(DType::F32).square().mean_dim(D - 1) + self.epsilon).sqrt();
        let scale = if self.zero_centered {
            self.weight.val() + 1.0
        } else {
            self.weight.val()
        };
        (input / rms.cast(dtype)) * scale.unsqueeze()
    }
}

#[cfg(test)]
mod tests {
    use burn::{backend::Flex, tensor::Tensor};

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
