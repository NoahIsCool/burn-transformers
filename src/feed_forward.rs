//! Dense and gated feed-forward blocks shared by transformer families.
//!
//! A dense MLP computes `down(phi(up(x)))`, while a gated MLP computes
//! `down(phi(gate(x)) * up(x))`. The second gated input projection acts as a
//! multiplicative content path. SwiGLU chooses `phi = SiLU`; T5 checkpoints use
//! either a dense ReLU path or gated GELU/SiLU variants.
//!
//! Hidden axes are `[batch, token, channel]`; input and output projections
//! operate independently at every batch/token location. Intermediate width is an
//! explicit architecture choice. Tensor dtype follows the projections and chosen
//! activation; no implicit masking, cache state, or training-mode dropout is added.
//!
//! In F16 and BF16, SiLU and tanh-GELU are evaluated in F32 and rounded once to
//! the activation dtype, as PyTorch's fused `silu` and
//! `gelu(approximate="tanh")` kernels are; F32 and F64 run Burn's operation
//! graphs unchanged (see [`apply_activation`]).
//!
//! `forward_with_hooks` emits the two canonical MLP intervention points in one
//! order shared by every family: `mlp_pre_activation` (the dense projection, or
//! the gate branch of a gated MLP) and then `mlp_activation` (after the
//! nonlinearity, before gating and the output projection). Input, output, and
//! residual points remain family wiring.
//!
//! [`ClampedSwiGlu`] is the clamped gated unit of gpt-oss and DeepSeek V4
//! experts: `(clamp(u, -L, L) + c) * g * sigmoid(alpha * g)` with
//! `g = min(gate, L)`. Clamping bounds activation outliers in reduced
//! precision; the `alpha`-scaled sigmoid and the `+c` offset are learned
//! conventions of those checkpoints.

mod widths;

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{DType, Tensor, activation, backend::Backend},
};

use crate::intervention::{Hooks, Layer, PointInfo};
use crate::{Error, Result, architecture::Activation, operations};

/// Configuration for an ordinary two-projection transformer feed-forward block.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DenseFeedForwardConfig {
    /// Residual-stream width accepted and returned by the block.
    pub hidden_size: usize,
    /// Width of the intermediate activation.
    pub intermediate_size: usize,
    /// Nonlinearity applied between the two projections.
    pub activation: Activation,
    /// Whether both projections have learned additive biases.
    pub bias: bool,
}

impl DenseFeedForwardConfig {
    /// Creates a bias-free dense ReLU block matching original T5.
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            activation: Activation::Relu,
            bias: false,
        }
    }

    /// Selects the nonlinearity between the input and output projections.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Enables or disables projection biases.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Validates projection extents before parameter allocation.
    pub fn validate(&self) -> Result<()> {
        validate_projection_widths(self.hidden_size, self.intermediate_size)
    }

    /// Initializes checkpoint-native `[out, in]` projection records.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<DenseFeedForward<B>> {
        self.validate()?;
        Ok(DenseFeedForward {
            up_proj: operations::init_linear(
                &LinearConfig::new(self.hidden_size, self.intermediate_size)
                    .with_bias(self.bias)
                    .with_layout(LinearLayout::Col),
                device,
            ),
            down_proj: operations::init_linear(
                &LinearConfig::new(self.intermediate_size, self.hidden_size)
                    .with_bias(self.bias)
                    .with_layout(LinearLayout::Col),
                device,
            ),
            activation: self.activation,
        })
    }
}

/// Trainable ordinary transformer feed-forward block.
#[derive(Module, Debug)]
pub struct DenseFeedForward<B: Backend> {
    /// Projection from the residual stream into intermediate channels.
    pub up_proj: Linear<B>,
    /// Projection from intermediate channels back into the residual stream.
    pub down_proj: Linear<B>,
    /// Nonlinearity applied to the intermediate tensor.
    pub activation: Activation,
}

impl<B: Backend> DenseFeedForward<B> {
    /// Evaluates `down(activation(up(input)))` over an arbitrary leading shape.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Result<Tensor<B, D>> {
        let hidden = operations::linear(input, &self.up_proj)?;
        operations::linear(apply_activation(hidden, self.activation), &self.down_proj)
    }

    /// Evaluates the block while emitting the canonical MLP points of `layer`.
    ///
    /// `mlp_pre_activation` is `up(input)` and `mlp_activation` is its
    /// nonlinearity, both `[batch, token, intermediate]`. Call inside the
    /// caller's [`Hooks::run`] scope; with empty hooks this equals [`Self::forward`].
    pub fn forward_with_hooks(
        &self,
        input: Tensor<B, 3>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let pre_activation = hooks.apply(
            layer.mlp_pre_activation(),
            operations::linear(input, &self.up_proj)?,
        )?;
        let activated = hooks.apply(
            layer.mlp_activation(),
            apply_activation(pre_activation, self.activation),
        )?;
        operations::linear(activated, &self.down_proj)
    }
}

/// Configuration for a checkpoint-oriented gated feed-forward block.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GatedFeedForwardConfig {
    /// Residual-stream width accepted and returned by the block.
    pub hidden_size: usize,
    /// Width of both multiplicative intermediate projections.
    pub intermediate_size: usize,
    /// Nonlinearity applied to the gate projection.
    pub activation: Activation,
    /// Whether every projection has a learned additive bias.
    pub bias: bool,
}

impl GatedFeedForwardConfig {
    /// Creates a bias-free SwiGLU block.
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            activation: Activation::Silu,
            bias: false,
        }
    }

    /// Selects a checkpoint-specific gate activation.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Enables or disables projection biases.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Validates all projection extents before parameter allocation.
    pub fn validate(&self) -> Result<()> {
        validate_projection_widths(self.hidden_size, self.intermediate_size)
    }

    /// Initializes three Burn linear maps using checkpoint-native `[out, in]` storage.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<GatedFeedForward<B>> {
        self.validate()?;
        let input_projection = LinearConfig::new(self.hidden_size, self.intermediate_size)
            .with_bias(self.bias)
            .with_layout(LinearLayout::Col);
        let output_projection = LinearConfig::new(self.intermediate_size, self.hidden_size)
            .with_bias(self.bias)
            .with_layout(LinearLayout::Col);
        Ok(GatedFeedForward {
            gate_proj: operations::init_linear(&input_projection, device),
            up_proj: operations::init_linear(&input_projection, device),
            down_proj: operations::init_linear(&output_projection, device),
            activation: self.activation,
        })
    }
}

/// Trainable gated feed-forward block used by dense layers and experts.
#[derive(Module, Debug)]
pub struct GatedFeedForward<B: Backend> {
    /// Projection controlling the multiplicative gate.
    pub gate_proj: Linear<B>,
    /// Projection carrying intermediate content.
    pub up_proj: Linear<B>,
    /// Projection returning intermediate channels to the residual width.
    pub down_proj: Linear<B>,
    /// Gate nonlinearity selected by the architecture.
    pub activation: Activation,
}

impl<B: Backend> GatedFeedForward<B> {
    /// Evaluates the gated projection on states whose last axis is `hidden_size`.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Result<Tensor<B, D>> {
        let gate = apply_activation(
            operations::linear(input.clone(), &self.gate_proj)?,
            self.activation,
        );
        let content = operations::linear(input, &self.up_proj)?;
        operations::linear(gate * content, &self.down_proj)
    }

    /// Evaluates the block while emitting the canonical MLP points of `layer`.
    ///
    /// `mlp_pre_activation` is the gate branch `gate(input)` and
    /// `mlp_activation` its nonlinearity before multiplication by `up(input)`,
    /// both `[batch, token, intermediate]`. Call inside the caller's
    /// [`Hooks::run`] scope; with empty hooks this equals [`Self::forward`].
    pub fn forward_with_hooks(
        &self,
        input: Tensor<B, 3>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let pre_activation = hooks.apply(
            layer.mlp_pre_activation(),
            operations::linear(input.clone(), &self.gate_proj)?,
        )?;
        let gate = hooks.apply(
            layer.mlp_activation(),
            apply_activation(pre_activation, self.activation),
        )?;
        let content = operations::linear(input, &self.up_proj)?;
        operations::linear(gate * content, &self.down_proj)
    }

    /// Evaluates `down(glu(gate(x), up(x)))` with a clamped gated unit.
    ///
    /// The block's `activation` field is not used: [`ClampedSwiGlu`] defines
    /// the complete gate nonlinearity. Interleaved checkpoint `gate_up`
    /// layouts must be split into `gate_proj`/`up_proj` when loading.
    pub fn forward_clamped<const D: usize>(
        &self,
        input: Tensor<B, D>,
        glu: &ClampedSwiGlu,
    ) -> Result<Tensor<B, D>> {
        glu.validate()?;
        let gate = operations::linear(input.clone(), &self.gate_proj)?;
        let content = operations::linear(input, &self.up_proj)?;
        operations::linear(glu.combine(gate, content), &self.down_proj)
    }
}

/// Lists the points emitted by `forward_with_hooks`, in emission order.
#[must_use]
pub fn mlp_hook_points(layer: Layer) -> Vec<PointInfo> {
    vec![
        layer.mlp_pre_activation().info(),
        layer.mlp_activation().info(),
    ]
}

/// Clamped gated linear unit of gpt-oss and DeepSeek V4 experts.
///
/// Computes `(clamp(up, -limit, limit) + up_offset) * g * sigmoid(alpha * g)`
/// with `g = min(gate, limit)`. gpt-oss uses `limit = 7`, `alpha = 1.702`, and
/// `up_offset = 1`; DeepSeek V4's clamped SwiGLU is `alpha = 1`, `up_offset = 0`
/// (then `g * sigmoid(g)` is SiLU). The gate is clamped only from above.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClampedSwiGlu {
    /// Upper clamp of the gate and symmetric clamp of the up projection.
    pub limit: f64,
    /// Sigmoid sharpness: `sigmoid(alpha * g)`; one gives SiLU.
    pub alpha: f64,
    /// Constant added to the clamped up projection.
    pub up_offset: f64,
}

impl ClampedSwiGlu {
    /// Creates a clamped SiLU-gated unit (`alpha = 1`, no offset).
    #[must_use]
    pub const fn new(limit: f64) -> Self {
        Self {
            limit,
            alpha: 1.0,
            up_offset: 0.0,
        }
    }

    /// Selects the sigmoid sharpness `alpha`.
    #[must_use]
    pub const fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Selects the constant added to the clamped up projection.
    #[must_use]
    pub const fn with_up_offset(mut self, up_offset: f64) -> Self {
        self.up_offset = up_offset;
        self
    }

    /// Requires a finite positive limit and alpha and a finite offset.
    pub fn validate(&self) -> Result<()> {
        if !self.limit.is_finite() || self.limit <= 0.0 {
            return Err(Error::configuration(
                "feed_forward.glu.limit",
                "must be finite and greater than zero",
            ));
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(Error::configuration(
                "feed_forward.glu.alpha",
                "must be finite and greater than zero",
            ));
        }
        if !self.up_offset.is_finite() {
            return Err(Error::configuration(
                "feed_forward.glu.up_offset",
                "must be finite",
            ));
        }
        Ok(())
    }

    /// Combines gate and up projections of equal shape into the gated activation.
    ///
    /// The unit is composed operation by operation in the activation dtype,
    /// as gpt-oss's reference `gate * torch.sigmoid(gate * alpha)` is, so in
    /// F16 and BF16 the scaled gate, the sigmoid and each product are rounded
    /// separately. It deliberately does not use the once-rounded SiLU of
    /// [`apply_activation`], which follows PyTorch's fused `silu` instead.
    pub fn combine<B: Backend, const D: usize>(
        &self,
        gate: Tensor<B, D>,
        up: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let gate = gate.clamp_max(self.limit);
        let up = up.clamp(-self.limit, self.limit) + self.up_offset;
        let gated = gate.clone() * activation::sigmoid(gate * self.alpha);
        up * gated
    }
}

/// Applies a supported transformer activation without changing tensor shape.
///
/// `Gelu` is the exact `x * Phi(x)`; `GeluTanh` is PyTorch's
/// `0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))` via Burn's
/// `gelu_approximate`, which is the T5 `gelu_new` function.
///
/// # Precision
///
/// `GeluTanh` follows PyTorch's `gelu(approximate="tanh")` (Transformers'
/// `gelu_pytorch_tanh`, Gemma's activation), whose CPU and MPS kernels
/// evaluate the whole expression in F32 for an F16 or BF16 input and round
/// the result once. Burn's `gelu_approximate` instead composes eight tensor
/// operations, each rounded to the input dtype, with both constants rounded
/// to it first (in BF16, `0.044715` becomes `0.044677734` and `sqrt(2 / pi)`
/// becomes `0.796875`); for negative inputs `tanh(...) + 1` then cancels
/// most of the rounded tangent's bits, so BF16 `GeluTanh(-2)` came out as
/// `-0.046875` instead of `-0.0454`, twelve half-units in the last place
/// away. An F16 or BF16 input is therefore widened to F32 and the result
/// rounded once (by the private helper `evaluate_in_f32`); F32 and F64 inputs
/// run Burn's graph in their own dtype, unchanged. Transformers' `gelu_new`
/// (T5, GPT-2) composes the same operations in the activation dtype with
/// unrounded constants, so neither graph reproduces its half-precision
/// rounding exactly; the two agree in F32.
///
/// `Silu` follows PyTorch's `silu` (Transformers' `SiLUActivation`, the gate
/// of every SwiGLU family) the same way (the crate-private helper `silu`
/// documents the convention and every call site). Burn's
/// `activation::silu` composes a rounded `sigmoid(x)` with a rounded
/// product; on Flex those two roundings miss the single one on 27% of the
/// BF16 and of the F16 inputs with `2^-8 <= |x| < 16`, and an F16 sigmoid
/// that underflows to zero below about `x = -17.3` erases SiLU values near
/// `-5e-7`.
///
/// `Relu`, `ReluSquared` and the exact `Gelu` use Burn's operations as they
/// are.
pub fn apply_activation<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    activation_kind: Activation,
) -> Tensor<B, D> {
    match activation_kind {
        Activation::Relu => activation::relu(input),
        Activation::Silu => silu(input),
        Activation::Gelu => activation::gelu(input),
        Activation::GeluTanh => evaluate_in_f32(input, activation::gelu_approximate),
        Activation::ReluSquared => activation::relu(input).square(),
    }
}

/// SiLU `x * sigmoid(x) = x / (1 + exp(-x))`, rounded once in F16 and BF16.
///
/// PyTorch's `silu` kernels evaluate an F16 or BF16 element in F32 and round
/// the result once. PyTorch 2.10.0's CPU kernel equals the F32 value
/// `x * sigmoid(x)` rounded to the dtype on every finite BF16 input and on
/// all but one F16 input, a near-tie that F32 evaluation order decides; MPS
/// also flushes BF16 subnormals to zero and decides a second F16 near-tie
/// the other way. Burn's `activation::silu` is the two-operation graph
/// `x * sigmoid(x)`: in half precision the sigmoid is rounded to the
/// activation dtype before the product, whose rounding adds a second error.
/// Near zero the rounded sigmoid loses the `x^2 / 4` term of
/// `x / 2 + x^2 / 4`; elsewhere the two roundings often land one unit in the
/// last place away from the correctly rounded value; and an F16 sigmoid
/// underflows to zero below about `x = -17.3`. Gated MLPs feed every such
/// difference into the residual stream of every block.
///
/// An F16 or BF16 input is therefore widened to F32, passed through Burn's
/// graph there, and the result rounded once (the private helper
/// `evaluate_in_f32`). F32 and F64 inputs run Burn's graph in their own
/// dtype, unchanged, so full-precision numerics and gradients are
/// bit-identical to `activation::silu`. Every crate SiLU on activations
/// calls this: [`apply_activation`] (dense and gated MLPs, routed and shared
/// experts), the SiLU attention output gate, and the depthwise-convolution
/// SiLU of Gated DeltaNet and Mamba-2. The gated RMS norms of both mixers
/// already evaluate their SiLU in F32, as their references do, and
/// [`ClampedSwiGlu`] keeps gpt-oss's operation-by-operation graph.
///
/// Cost: the F32 evaluation materializes F32 copies of the input-sized
/// tensor (four bytes per element instead of two) while the call runs, plus
/// two casts; the result is returned in the input dtype.
pub(crate) fn silu<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    evaluate_in_f32(input, activation::silu)
}

/// Evaluates an elementwise `function` in F32 for an F16 or BF16 `input` and rounds the result once to the input dtype.
///
/// This is PyTorch's `opmath` convention for half-precision pointwise
/// kernels: every intermediate keeps F32's 24 significand bits and only the
/// output is rounded, so the result is the correctly rounded F32 value
/// (within one rounding of F32 evaluation error). Any other dtype is passed
/// to `function` unchanged. The F32 intermediate costs one tensor of the
/// input's shape at four bytes per element, live only during the call.
fn evaluate_in_f32<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    function: impl FnOnce(Tensor<B, D>) -> Tensor<B, D>,
) -> Tensor<B, D> {
    let dtype = input.dtype();
    match dtype {
        DType::F16 | DType::BF16 => function(input.cast(DType::F32)).cast(dtype),
        _ => function(input),
    }
}

/// Checks dimensions shared by dense and gated feed-forward layouts.
///
/// The check order is the Verus-verified `widths::width_fault`.
fn validate_projection_widths(hidden_size: usize, intermediate_size: usize) -> Result<()> {
    match widths::width_fault(hidden_size, intermediate_size) {
        None => Ok(()),
        Some(widths::WidthFault::HiddenSize) => Err(Error::configuration(
            "feed_forward.hidden_size",
            "must be non-zero",
        )),
        Some(widths::WidthFault::IntermediateSize) => Err(Error::configuration(
            "feed_forward.intermediate_size",
            "must be non-zero",
        )),
    }
}

#[cfg(test)]
#[path = "feed_forward_tests.rs"]
mod tests;
