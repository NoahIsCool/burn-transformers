//! Dense gated feed-forward blocks shared by transformer families.
//!
//! A gated MLP computes `down(phi(gate(x)) * up(x))`. Unlike an ordinary
//! two-layer MLP, the second input projection acts as a multiplicative content
//! path. SwiGLU chooses `phi = SiLU`; T5 and Nemotron checkpoints select other
//! activations while retaining the same projection topology.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{Tensor, activation, backend::Backend},
};

use crate::{Error, Result, architecture::Activation};

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
        if self.hidden_size == 0 {
            return Err(Error::configuration(
                "feed_forward.hidden_size",
                "must be non-zero",
            ));
        }
        if self.intermediate_size == 0 {
            return Err(Error::configuration(
                "feed_forward.intermediate_size",
                "must be non-zero",
            ));
        }
        Ok(())
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
            gate_proj: input_projection.init(device),
            up_proj: input_projection.init(device),
            down_proj: output_projection.init(device),
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
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let gate = apply_activation(self.gate_proj.forward(input.clone()), self.activation);
        self.down_proj.forward(gate * self.up_proj.forward(input))
    }
}

/// Applies a supported transformer activation without changing tensor shape.
pub fn apply_activation<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    activation_kind: Activation,
) -> Tensor<B, D> {
    match activation_kind {
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

#[cfg(test)]
mod tests {
    use burn::{backend::Flex, tensor::Tensor};

    use crate::architecture::Activation;

    use super::{GatedFeedForwardConfig, apply_activation};

    /// Every activation preserves shape and finite values on representative inputs.
    #[test]
    fn activations_are_finite_and_shape_preserving() {
        let device = Default::default();
        for activation in [
            Activation::Silu,
            Activation::Gelu,
            Activation::GeluTanh,
            Activation::ReluSquared,
        ] {
            let input = Tensor::<Flex, 2>::from_floats([[-2.0, -0.0, 1.0, 3.0]], &device);
            let output = apply_activation(input, activation);
            assert_eq!(output.dims(), [1, 4]);
            let output = output.to_data();
            let values = output
                .as_slice::<f32>()
                .expect("Flex f32 data should expose an f32 slice");
            assert!(values.iter().all(|value| value.is_finite()));
        }
    }

    /// Squared ReLU exactly removes negative values before squaring.
    #[test]
    fn relu_squared_matches_definition() {
        let device = Default::default();
        let output = apply_activation(
            Tensor::<Flex, 1>::from_floats([-2.0, 0.0, 3.0], &device),
            Activation::ReluSquared,
        )
        .to_data();
        assert_eq!(
            output
                .as_slice::<f32>()
                .expect("Flex f32 data should expose an f32 slice"),
            &[0.0, 0.0, 9.0]
        );
    }

    /// Gated MLP configuration rejects zero-width projections before initialization.
    #[test]
    fn invalid_projection_width_is_rejected() {
        let error = GatedFeedForwardConfig::new(8, 0)
            .validate()
            .expect_err("zero intermediate width must fail");
        assert!(error.to_string().contains("intermediate_size"));
    }

    /// A valid block preserves the residual-stream shape.
    #[test]
    fn feed_forward_preserves_residual_shape() {
        let device = Default::default();
        let module = GatedFeedForwardConfig::new(4, 8)
            .init::<Flex>(&device)
            .expect("valid block should initialize");
        let output = module.forward(Tensor::<Flex, 3>::zeros([2, 3, 4], &device));
        assert_eq!(output.dims(), [2, 3, 4]);
    }
}
