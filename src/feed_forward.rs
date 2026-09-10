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

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{Tensor, activation, backend::Backend},
};

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
}

/// Applies a supported transformer activation without changing tensor shape.
pub fn apply_activation<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    activation_kind: Activation,
) -> Tensor<B, D> {
    match activation_kind {
        Activation::Relu => activation::relu(input),
        Activation::Silu => activation::silu(input),
        Activation::Gelu => activation::gelu(input),
        Activation::GeluTanh => {
            let cubic = input.clone().powf_scalar(3.0);
            let inner = (input.clone() + cubic * 0.044_715) * 0.797_884_560_802_865_4;
            input * (inner.tanh() + 1.0) * 0.5
        }
        Activation::ReluSquared => activation::relu(input).square(),
    }
}

/// Checks dimensions shared by dense and gated feed-forward layouts.
fn validate_projection_widths(hidden_size: usize, intermediate_size: usize) -> Result<()> {
    if hidden_size == 0 {
        return Err(Error::configuration(
            "feed_forward.hidden_size",
            "must be non-zero",
        ));
    }
    if intermediate_size == 0 {
        return Err(Error::configuration(
            "feed_forward.intermediate_size",
            "must be non-zero",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "feed_forward_tests.rs"]
mod tests;
