//! Sparse mixture-of-experts routing and dispatch.
//!
//! A router assigns probabilities to `E` experts and evaluates only the largest
//! `K` assignments for each token. The output is
//! `sum_{j in TopK(p)} w_j Expert_j(x)`, optionally followed by always-active
//! shared experts. Router softmax is evaluated in FP32 because small ranking
//! changes can switch an entire expert computation in lower precision.
//!
//! The portable dispatcher groups selected tokens with Burn indexing
//! operations and never evaluates an unselected expert. It is a correctness
//! path; accelerator-specific grouped-GEMM dispatch can replace its grouping
//! mechanics without altering routing or checkpoint parameters.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{DType, IndexingUpdateOp, Int, Tensor, activation::softmax, backend::Backend},
};

use crate::{
    Error, Result,
    architecture::Activation,
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
};

/// Configuration for routed and optionally shared gated experts.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SparseMoeConfig {
    /// Residual-stream width accepted and returned by every expert.
    pub hidden_size: usize,
    /// Intermediate width of each routed and shared expert.
    pub intermediate_size: usize,
    /// Number of independently parameterized routed experts.
    pub routed_experts: usize,
    /// Number of routed experts selected for each token.
    pub experts_per_token: usize,
    /// Number of experts evaluated for every token.
    pub shared_experts: usize,
    /// Expert gate activation.
    pub activation: Activation,
    /// Whether selected router weights are normalized to sum to one.
    pub renormalize_selected: bool,
    /// Scale applied to the complete routed-expert contribution.
    pub routed_scale: f64,
    /// Whether expert projections contain additive biases.
    pub expert_bias: bool,
}

impl SparseMoeConfig {
    /// Creates a bias-free top-k SwiGLU router without shared experts.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        routed_experts: usize,
        experts_per_token: usize,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            routed_experts,
            experts_per_token,
            shared_experts: 0,
            activation: Activation::Silu,
            renormalize_selected: true,
            routed_scale: 1.0,
            expert_bias: false,
        }
    }

    /// Validates sparse cardinalities and expert numerical parameters.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 || self.intermediate_size == 0 || self.routed_experts == 0 {
            return Err(Error::configuration(
                "moe",
                "hidden_size, intermediate_size, and routed_experts must be non-zero",
            ));
        }
        if self.experts_per_token == 0 || self.experts_per_token > self.routed_experts {
            return Err(Error::configuration(
                "moe.experts_per_token",
                "must be in 1..=routed_experts",
            ));
        }
        if !self.routed_scale.is_finite() || self.routed_scale <= 0.0 {
            return Err(Error::configuration(
                "moe.routed_scale",
                "must be finite and greater than zero",
            ));
        }
        Ok(())
    }

    /// Initializes the router and all expert parameter sets.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<SparseMoe<B>> {
        self.validate()?;
        let expert_config = GatedFeedForwardConfig::new(self.hidden_size, self.intermediate_size)
            .with_activation(self.activation)
            .with_bias(self.expert_bias);
        let routed = (0..self.routed_experts)
            .map(|_| expert_config.init(device))
            .collect::<Result<Vec<_>>>()?;
        let shared = (0..self.shared_experts)
            .map(|_| expert_config.init(device))
            .collect::<Result<Vec<_>>>()?;
        let router = LinearConfig::new(self.hidden_size, self.routed_experts)
            .with_bias(false)
            .with_layout(LinearLayout::Col)
            .init(device);
        Ok(SparseMoe {
            router,
            routed,
            shared,
            experts_per_token: self.experts_per_token,
            renormalize_selected: self.renormalize_selected,
            routed_scale: self.routed_scale,
            hidden_size: self.hidden_size,
        })
    }
}

/// Sparse router output and optional inspection data.
#[derive(Clone, Debug)]
pub struct SparseMoeOutput<B: Backend> {
    /// Weighted routed and shared expert output.
    pub hidden_state: Tensor<B, 3>,
    /// Pre-softmax router scores shaped `[batch, sequence, experts]`.
    pub router_logits: Tensor<B, 3>,
    /// Selected expert IDs shaped `[batch, sequence, experts_per_token]`.
    pub selected_experts: Tensor<B, 3, Int>,
    /// Selected probabilities after optional renormalization and routed scaling.
    pub selected_weights: Tensor<B, 3>,
}

/// Trainable sparse MoE with independent routed and shared expert weights.
#[derive(Module, Debug)]
pub struct SparseMoe<B: Backend> {
    /// Bias-free token-to-expert router.
    pub router: Linear<B>,
    /// Experts invoked only when selected by top-k routing.
    pub routed: Vec<GatedFeedForward<B>>,
    /// Experts evaluated and added for every token.
    pub shared: Vec<GatedFeedForward<B>>,
    /// Number of selected routed experts per token.
    pub experts_per_token: usize,
    /// Whether top-k weights are renormalized.
    pub renormalize_selected: bool,
    /// Multiplicative routed-output scale.
    pub routed_scale: f64,
    /// Residual-stream width used when flattening tokens.
    pub hidden_size: usize,
}

impl<B: Backend> SparseMoe<B> {
    /// Routes and dispatches `[batch, sequence, hidden]` states sparsely.
    pub fn forward(&self, input: Tensor<B, 3>) -> SparseMoeOutput<B> {
        let [batch, sequence, hidden] = input.dims();
        assert_eq!(
            hidden, self.hidden_size,
            "MoE hidden width must match config"
        );
        let tokens = batch * sequence;
        let flat_input = input.clone().reshape([tokens, hidden]);
        let router_logits = self.router.forward(input);
        let router_dtype = router_logits.dtype();
        let probabilities = softmax(router_logits.clone().cast(DType::F32), 2).cast(router_dtype);
        let (selected_weights, selected_experts) =
            probabilities.topk_with_indices(self.experts_per_token, 2);
        let mut flat_weights = selected_weights
            .clone()
            .reshape([tokens, self.experts_per_token]);
        if self.renormalize_selected {
            let denominator = flat_weights.clone().sum_dim(1).clamp_min(f32::MIN_POSITIVE);
            flat_weights = flat_weights / denominator;
        }
        flat_weights = flat_weights * self.routed_scale;
        let flat_experts = selected_experts
            .clone()
            .reshape([tokens, self.experts_per_token]);
        let mut output = Tensor::<B, 2>::zeros([tokens, hidden], &flat_input.device());

        for (expert_index, expert) in self.routed.iter().enumerate() {
            let locations = flat_experts
                .clone()
                .equal_elem(expert_index as i64)
                .argwhere();
            let assignments = locations.dims()[0];
            if assignments == 0 {
                continue;
            }
            let token_indices = locations.clone().narrow(1, 0, 1).squeeze_dim::<1>(1);
            let weights = flat_weights
                .clone()
                .gather_nd::<2, 1>(locations)
                .unsqueeze_dim::<2>(1);
            let expert_input = flat_input.clone().select(0, token_indices.clone());
            let contribution = expert.forward(expert_input) * weights;
            output = output.select_assign(0, token_indices, contribution, IndexingUpdateOp::Add);
        }

        let mut shared_output = Tensor::<B, 2>::zeros([tokens, hidden], &flat_input.device());
        for expert in &self.shared {
            shared_output = shared_output + expert.forward(flat_input.clone());
        }
        let selected_weights = flat_weights.reshape([batch, sequence, self.experts_per_token]);
        SparseMoeOutput {
            hidden_state: (output + shared_output).reshape([batch, sequence, hidden]),
            router_logits,
            selected_experts,
            selected_weights,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{backend::Flex, tensor::Tensor};

    use super::SparseMoeConfig;

    /// Invalid top-k routing is rejected before allocating hundreds of experts.
    #[test]
    fn invalid_top_k_is_rejected() {
        let error = SparseMoeConfig::new(4, 8, 2, 3)
            .validate()
            .expect_err("top-k greater than expert count must fail");
        assert!(error.to_string().contains("experts_per_token"));
    }

    /// Sparse routing returns stable public shapes for diagnostics and hidden states.
    #[test]
    fn sparse_moe_preserves_shapes() {
        let device = Default::default();
        let module = SparseMoeConfig::new(4, 8, 3, 2)
            .init::<Flex>(&device)
            .expect("valid MoE should initialize");
        let output = module.forward(Tensor::<Flex, 3>::zeros([2, 3, 4], &device));
        assert_eq!(output.hidden_state.dims(), [2, 3, 4]);
        assert_eq!(output.router_logits.dims(), [2, 3, 3]);
        assert_eq!(output.selected_experts.dims(), [2, 3, 2]);
        assert_eq!(output.selected_weights.dims(), [2, 3, 2]);
    }

    /// Renormalized selected probabilities sum to the routed scale for every token.
    #[test]
    fn selected_weights_are_renormalized() {
        let device = Default::default();
        let mut config = SparseMoeConfig::new(4, 8, 4, 2);
        config.routed_scale = 2.5;
        let module = config
            .init::<Flex>(&device)
            .expect("valid MoE should initialize");
        let output = module.forward(Tensor::<Flex, 3>::ones([1, 2, 4], &device));
        let sums = output.selected_weights.sum_dim(2).to_data();
        let values = sums
            .as_slice::<f32>()
            .expect("Flex f32 data should expose an f32 slice");
        assert!(values.iter().all(|value| (*value - 2.5).abs() < 1e-5));
    }
}
