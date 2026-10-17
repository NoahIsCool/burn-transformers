//! Transformer normalization variants with explicit checkpoint conventions.
//!
//! RMS normalization rescales a vector without subtracting its mean:
//! `y = x / sqrt(mean(x^2) + epsilon) * scale`. The reduction, inverse root,
//! and learned scaling evaluate in [`at_least_f32`] of the input dtype.
//! Promoting F16/BF16 to F32 is important for reduced-precision inference
//! because squaring and summing a wide residual stream otherwise loses
//! significant dynamic range; F32 inputs stay F32, and F64 inputs are never
//! narrowed, so a double-precision model normalizes in F64. Some recent
//! checkpoints store a zero-centered learned parameter and define
//! `scale = 1 + weight`; older checkpoints store `scale` directly. Although
//! `x / sqrt(m)`, `x * pow(m, -0.5)`, and `x * reciprocal(sqrt(m))` are
//! algebraically equal, their floating-point operation graphs can round
//! differently. The root mode therefore belongs to checkpoint compatibility.

use burn::{
    module::{Initializer, Module, Param},
    tensor::{Tensor, backend::Backend},
};

use crate::{Error, Result, precision::at_least_f32};

/// Floating-point operation graph used to apply an inverse RMS denominator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RmsNormRootMode {
    /// Divides the activation by `sqrt(mean_square + epsilon)`.
    #[default]
    DivideBySqrt,
    /// Multiplies by `pow(mean_square + epsilon, -0.5)`.
    ///
    /// Muse uses this form for its scale-free embedding/Q/K norms and its
    /// ordinary learned final norm to agree with its JAX reference.
    MultiplyByPower,
    /// Multiplies by `reciprocal(sqrt(mean_square + epsilon))`.
    ///
    /// This is Burn 0.21's portable expression of the `rsqrt` graph used by
    /// Muse's zero-centered sandwich norms and Nemotron-H RMSNorm.
    MultiplyByReciprocalSqrt,
}

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
    /// Some architectures, including DeepSeek V2, Qwen3-MoE, and Nemotron-H,
    /// cast the unit-RMS activation back to the residual dtype before
    /// multiplying by the learned parameter. Others, including Qwen3.5/Qwen3.8,
    /// keep both operations in the evaluation dtype (FP32 for reduced
    /// precision) and round only the final result.
    #[serde(default)]
    pub round_before_scale: bool,
    /// Inverse-root operation graph required by the checkpoint implementation.
    #[serde(default)]
    pub root_mode: RmsNormRootMode,
}

impl RmsNormConfig {
    /// Creates a conventional RMSNorm configuration with epsilon `1e-6`.
    pub fn new(size: usize) -> Self {
        Self {
            size,
            epsilon: 1e-6,
            zero_centered: false,
            round_before_scale: false,
            root_mode: RmsNormRootMode::DivideBySqrt,
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

    /// Selects rounding before learned scaling for architectures that require it.
    pub fn with_round_before_scale(mut self, round_before_scale: bool) -> Self {
        self.round_before_scale = round_before_scale;
        self
    }

    /// Selects the inverse-root floating-point operation graph.
    pub fn with_root_mode(mut self, root_mode: RmsNormRootMode) -> Self {
        self.root_mode = root_mode;
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
            root_mode: self.root_mode,
        })
    }
}

/// RMS normalization with explicit checkpoint-compatible precision ordering.
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Learned scale, or learned scale minus one for zero-centered checkpoints.
    pub weight: Param<Tensor<B, 1>>,
    /// Stabilizer added to the mean square in the evaluation dtype.
    pub epsilon: f64,
    /// Whether to add one to the learned parameter before scaling.
    pub zero_centered: bool,
    /// Whether the normalized activation is rounded before scale multiplication.
    pub round_before_scale: bool,
    /// Inverse-root operation graph excluded from checkpoint parameters.
    #[module(skip)]
    pub root_mode: RmsNormRootMode,
}

impl<B: Backend> RmsNorm<B> {
    /// Normalizes the last axis of a tensor of arbitrary rank.
    ///
    /// The reduction evaluates in [`at_least_f32`] of the input dtype: F16/BF16
    /// promote to F32, F32 stays F32, and F64 stays F64. The input dtype is
    /// restored afterwards, so this method remains compatible with F16 Metal
    /// execution while preserving an FP32 variance estimate, and an F64 model
    /// keeps every normalization operation in F64. The learned weight is cast
    /// to the same evaluation dtype. By default, normalization and learned
    /// scaling remain in the evaluation dtype until the final cast; this
    /// matches zero-centered Qwen RMSNorm.
    /// [`RmsNormConfig::with_round_before_scale`] instead reproduces
    /// architectures that round the normalized activation first.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = input.dtype();
        let evaluation = at_least_f32(dtype);
        let input = input.cast(evaluation);
        let mean_square = input.clone().square().mean_dim(D - 1) + self.epsilon;
        let scale = if self.zero_centered {
            self.weight.val().cast(evaluation) + 1.0
        } else {
            self.weight.val().cast(evaluation)
        };
        let normalized = match self.root_mode {
            RmsNormRootMode::DivideBySqrt => input / mean_square.sqrt(),
            RmsNormRootMode::MultiplyByPower => input * mean_square.powf_scalar(-0.5),
            RmsNormRootMode::MultiplyByReciprocalSqrt => input * mean_square.sqrt().recip(),
        };
        if self.round_before_scale {
            normalized.cast(dtype) * scale.cast(dtype).unsqueeze()
        } else {
            (normalized * scale.unsqueeze()).cast(dtype)
        }
    }
}

#[cfg(test)]
#[path = "normalization_tests.rs"]
mod tests;
