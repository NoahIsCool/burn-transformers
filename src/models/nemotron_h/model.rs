//! Executable NVIDIA Nemotron-H hybrid causal decoder.
//!
//! A Nemotron-H schedule is heterogeneous at residual-block granularity. Every
//! block computes `x + Mixer(RMSNorm(x))`, but `Mixer` may be Mamba2, causal
//! grouped-query attention, a dense ReLU² MLP, or a sparse ReLU² MoE. Unlike a
//! conventional transformer layer there is no mandatory attention/FFN pair.
//!
//! The released 3.5 Lightning checkpoint uses 52 blocks, alternating Mamba2 and
//! top-6/128 MoE blocks with periodic GQA. Its MoE router scores experts with a
//! sigmoid, applies a learned correction only to discrete selection, and uses
//! the uncorrected score as the mixture weight. A shared expert always executes;
//! routed experts execute only for tokens that selected them.
//!
//! Parameters never own generation state. [`NemotronHDecodeState`] combines a
//! native KV bank for attention blocks and recurrent matrix/convolution banks for
//! Mamba2 blocks. Stateless paths stay differentiable for training v2.

use burn::{
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{
        Bool, DType, ElementConversion, IndexingUpdateOp, Int, Tensor, TensorData, activation,
        backend::Backend,
    },
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, LogitsSelection,
    Result,
    architecture::{Activation, RouterDiagnostics, TokenMask},
    attention::{GroupedQueryAttention, GroupedQueryAttentionConfig, padding_attention_mask},
    feed_forward::apply_activation,
    moe::group_selected_experts,
    normalization::{RmsNorm, RmsNormConfig, RmsNormRootMode},
    operations,
};

use super::{NemotronHBlockType, NemotronHConfig};

mod decode;
pub mod mamba2;
mod mtp;
mod speculative;

pub use decode::NemotronHDecodeState;
use mamba2::{NemotronMamba2, NemotronMamba2Config, NemotronMamba2Mode};
pub use mtp::NemotronHMtpPrediction;
pub use speculative::{
    NemotronHSpeculativeConfig, NemotronHSpeculativeDecoder, NemotronHSpeculativeOutput,
};

/// Ordinary two-projection MLP used by dense, routed, and shared blocks.
#[derive(Module, Debug)]
pub struct NemotronHDense<B: Backend> {
    /// Projection into intermediate channels.
    pub up_proj: Linear<B>,
    /// Projection back to the residual/latent width.
    pub down_proj: Linear<B>,
    /// ReLU² for released checkpoints, with SiLU retained for legacy configs.
    pub activation: Activation,
    /// Input and output width.
    pub hidden_size: usize,
    /// Intermediate activation width.
    pub intermediate_size: usize,
}

impl<B: Backend> NemotronHDense<B> {
    /// Initializes a checkpoint-native `[out, in]` MLP.
    fn init(
        hidden_size: usize,
        intermediate_size: usize,
        activation: Activation,
        bias: bool,
        device: &B::Device,
    ) -> Result<Self> {
        if hidden_size == 0 || intermediate_size == 0 {
            return Err(Error::configuration(
                "nemotron_h.mlp",
                "hidden and intermediate widths must be non-zero",
            ));
        }
        let linear = |input, output| {
            operations::init_linear(
                &LinearConfig::new(input, output)
                    .with_bias(bias)
                    .with_layout(LinearLayout::Col),
                device,
            )
        };
        Ok(Self {
            up_proj: linear(hidden_size, intermediate_size),
            down_proj: linear(intermediate_size, hidden_size),
            activation,
            hidden_size,
            intermediate_size,
        })
    }

    /// Evaluates `down(activation(up(x)))` without a multiplicative gate.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Result<Tensor<B, D>> {
        let activated =
            apply_activation(operations::linear(input, &self.up_proj)?, self.activation);
        operations::linear(activated, &self.down_proj)
    }
}

/// Sparse routing output retained for requested diagnostics and tests.
#[derive(Clone, Debug)]
pub struct NemotronHMoeOutput<B: Backend> {
    /// Sum of routed and always-active shared expert outputs.
    pub hidden_state: Tensor<B, 3>,
    /// FP32 pre-sigmoid router logits `[batch, sequence, experts]`.
    pub router_logits: Tensor<B, 3>,
    /// Chosen expert IDs `[batch, sequence, top_k]`.
    pub selected_experts: Tensor<B, 3, Int>,
    /// Uncorrected sigmoid mixture weights after normalization/scaling.
    pub selected_weights: Tensor<B, 3>,
    /// Assignment count for every routed expert.
    pub expert_token_counts: Vec<usize>,
    /// Number of routed expert graphs actually evaluated.
    pub evaluated_experts: usize,
}

impl<B: Backend> NemotronHMoeOutput<B> {
    /// Computes a differentiable sigmoid-router load-balancing diagnostic.
    pub fn load_balancing_loss(&self, token_mask: Option<&TokenMask<B>>) -> Result<Tensor<B, 1>> {
        nemotron_load_balancing_loss(
            self.router_logits.clone(),
            self.selected_experts.clone(),
            token_mask,
        )
    }
}

/// Computes load balance from reconstructed cached router diagnostics.
pub(super) fn nemotron_load_balancing_loss<B: Backend>(
    router_logits: Tensor<B, 3>,
    selected_experts: Tensor<B, 3, Int>,
    token_mask: Option<&TokenMask<B>>,
) -> Result<Tensor<B, 1>> {
    let [batch, sequence, experts] = router_logits.dims();
    let top_k = selected_experts.dims()[2];
    if selected_experts.dims()[..2] != [batch, sequence] {
        return Err(Error::shape(
            "nemotron_h.moe.selected_experts",
            format!("[{batch}, {sequence}, top_k]"),
            format!("{:?}", selected_experts.dims()),
        ));
    }
    let valid = token_mask.map_or_else(
        || Tensor::<B, 2, Bool>::ones([batch, sequence], &router_logits.device()),
        |mask| mask.as_tensor().clone(),
    );
    if valid.dims() != [batch, sequence] {
        return Err(Error::shape(
            "nemotron_h.moe.token_mask",
            format!("[{batch}, {sequence}]"),
            format!("{:?}", valid.dims()),
        ));
    }
    let denominator = valid.clone().float().sum().clamp_min(1.0);
    let assignments = selected_experts.one_hot::<4>(experts).float()
        * valid
            .clone()
            .float()
            .reshape([batch, sequence, 1, 1])
            .expand([batch, sequence, top_k, experts]);
    let assignment_rate = assignments
        .sum_dim(0)
        .sum_dim(1)
        .sum_dim(2)
        .reshape([experts])
        / (denominator.clone() * top_k as f64);
    let score_rate = (activation::sigmoid(router_logits)
        * valid
            .float()
            .reshape([batch, sequence, 1])
            .expand([batch, sequence, experts]))
    .sum_dim(0)
    .sum_dim(1)
    .reshape([experts])
        / denominator;
    Ok((assignment_rate * score_rate).sum() * experts as f64)
}

/// Nemotron's correction-biased sigmoid router plus shared ReLU² expert.
#[derive(Module, Debug)]
pub struct NemotronHMoe<B: Backend> {
    /// Bias-free token-to-expert projection.
    pub router: Linear<B>,
    /// FP32 score correction used only for top-k selection.
    pub correction_bias: Param<Tensor<B, 1>>,
    /// Independently dispatched routed experts.
    pub routed: Vec<NemotronHDense<B>>,
    /// Always-evaluated shared expert.
    pub shared: NemotronHDense<B>,
    /// Optional projection into a lower-dimensional routed-expert space.
    pub fc1_latent_proj: Option<Linear<B>>,
    /// Optional projection from routed latent space to the residual width.
    pub fc2_latent_proj: Option<Linear<B>>,
    /// Experts selected per token.
    pub experts_per_token: usize,
    /// Number of equal expert groups.
    pub groups: usize,
    /// Groups retained before expert top-k.
    pub top_groups: usize,
    /// Whether selected sigmoid scores sum to one before scaling.
    pub renormalize_selected: bool,
    /// Multiplicative routed contribution scale.
    pub routed_scale: f64,
    /// Residual-stream width.
    pub hidden_size: usize,
}

impl<B: Backend> NemotronHMoe<B> {
    /// Initializes exact routed/shared expert shapes from family configuration.
    fn init(config: &NemotronHConfig, device: &B::Device) -> Result<Self> {
        let activation = activation_from_name(&config.mlp_hidden_act)?;
        let latent = config.moe_latent_size.unwrap_or(config.hidden_size);
        let routed = (0..config.n_routed_experts)
            .map(|_| {
                NemotronHDense::init(
                    latent,
                    config.moe_intermediate_size,
                    activation,
                    config.mlp_bias,
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let projection = |input, output| {
            operations::init_linear(
                &LinearConfig::new(input, output)
                    .with_bias(config.mlp_bias)
                    .with_layout(LinearLayout::Col),
                device,
            )
        };
        Ok(Self {
            router: operations::init_linear(
                &LinearConfig::new(config.hidden_size, config.n_routed_experts)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col),
                device,
            ),
            // NVIDIA declares this as an FP32 buffer: it changes the discrete
            // top-k decision but is not optimized by back-propagation.
            correction_bias: Param::from_tensor(Tensor::zeros(
                [config.n_routed_experts],
                (device, DType::F32),
            ))
            .set_require_grad(false),
            routed,
            shared: NemotronHDense::init(
                config.hidden_size,
                config.moe_shared_expert_intermediate_size,
                activation,
                config.mlp_bias,
                device,
            )?,
            fc1_latent_proj: config
                .moe_latent_size
                .map(|width| projection(config.hidden_size, width)),
            fc2_latent_proj: config
                .moe_latent_size
                .map(|width| projection(width, config.hidden_size)),
            experts_per_token: config.num_experts_per_tok,
            groups: config.n_group,
            top_groups: config.topk_group,
            renormalize_selected: config.norm_topk_prob,
            routed_scale: config.routed_scaling_factor,
            hidden_size: config.hidden_size,
        })
    }

    /// Selects and evaluates only routed experts receiving at least one token.
    pub fn forward(&self, input: Tensor<B, 3>) -> Result<NemotronHMoeOutput<B>> {
        let [batch, sequence, hidden] = input.dims();
        let experts = self.routed.len();
        if batch == 0 || sequence == 0 || hidden != self.hidden_size {
            return Err(Error::shape(
                "nemotron_h.moe.input",
                format!("[batch>0, sequence>0, {}]", self.hidden_size),
                format!("[{batch}, {sequence}, {hidden}]"),
            ));
        }
        if experts == 0
            || self.experts_per_token == 0
            || self.experts_per_token > experts
            || self.groups == 0
            || !experts.is_multiple_of(self.groups)
            || self.top_groups == 0
            || self.top_groups > self.groups
        {
            return Err(Error::configuration(
                "nemotron_h.moe.router",
                "instantiated routing cardinalities are inconsistent",
            ));
        }
        let tokens = batch.checked_mul(sequence).ok_or_else(|| {
            Error::configuration("nemotron_h.moe", "batch * sequence overflows usize")
        })?;
        let flat = input.clone().reshape([tokens, hidden]);
        let router_weight = self.router.weight.val().cast(DType::F32);
        let router_logits = flat
            .clone()
            .cast(DType::F32)
            .matmul(router_weight)
            .reshape([batch, sequence, experts]);
        let scores = activation::sigmoid(router_logits.clone()).reshape([tokens, experts]);
        let corrected = scores.clone()
            + self
                .correction_bias
                .val()
                .cast(DType::F32)
                .reshape([1, experts]);
        let eligible = if self.groups == 1 {
            // Lightning's `n_group = topk_group = 1` makes group pruning the
            // identity. Besides avoiding unnecessary work, this branch is
            // required because Burn intentionally rejects one-hot tensors with
            // fewer than two classes.
            corrected
        } else {
            let experts_per_group = experts / self.groups;
            let group_candidates = deterministic_top_k(
                corrected
                    .clone()
                    .reshape([tokens, self.groups, experts_per_group]),
                2.min(experts_per_group),
            )
            .0
            .sum_dim(2)
            .reshape([1, tokens, self.groups]);
            let selected_groups = deterministic_top_k(group_candidates, self.top_groups)
                .1
                .reshape([tokens, self.top_groups]);
            let group_mask = selected_groups
                .one_hot::<3>(self.groups)
                .sum_dim(1)
                .greater_elem(0)
                .reshape([tokens, self.groups, 1])
                .expand([tokens, self.groups, experts_per_group])
                .reshape([tokens, experts]);
            corrected.mask_fill(group_mask.bool_not(), f32::NEG_INFINITY)
        };
        let (ignored_corrected_weights, selected_experts) = deterministic_top_k(
            eligible.reshape([1, tokens, experts]),
            self.experts_per_token,
        );
        drop(ignored_corrected_weights);
        let selected_experts_flat = selected_experts.reshape([tokens, self.experts_per_token]);
        let mut selected_weights = scores.gather(1, selected_experts_flat.clone());
        if self.renormalize_selected {
            let denominator = selected_weights.clone().sum_dim(1).clamp_min(1e-20);
            selected_weights = selected_weights / denominator;
        }
        // The reference keeps sigmoid routing and the sparse weighted sum in
        // FP32 even when expert projections execute in BF16/F16.
        selected_weights = selected_weights * self.routed_scale;

        let routed_input = match &self.fc1_latent_proj {
            Some(projection) => operations::linear(flat.clone(), projection)?,
            None => flat.clone(),
        };
        let routed_dtype = routed_input.dtype();
        let latent = routed_input.dims()[1];
        let assignment_count = tokens.checked_mul(self.experts_per_token).ok_or_else(|| {
            Error::configuration(
                "nemotron_h.moe",
                "tokens * experts_per_token overflows usize",
            )
        })?;
        let groups = group_selected_experts(selected_experts_flat.clone(), experts)?;
        let flat_selected_weights = selected_weights.clone().reshape([assignment_count]);
        let mut routed_output =
            Tensor::<B, 2>::zeros([tokens, latent], (&flat.device(), DType::F32));
        let mut counts = Vec::with_capacity(experts);
        let mut evaluated = 0;
        for (expert, group) in self.routed.iter().zip(groups) {
            let assignments = group.token_indices.len();
            counts.push(assignments);
            if assignments == 0 {
                continue;
            }
            evaluated += 1;
            let token_indices = Tensor::<B, 1, Int>::from_data(
                TensorData::new(group.token_indices, [assignments]),
                &flat.device(),
            );
            let weight_indices = Tensor::<B, 1, Int>::from_data(
                TensorData::new(group.weight_indices, [assignments]),
                &flat.device(),
            );
            let weights = flat_selected_weights
                .clone()
                .select(0, weight_indices)
                .unsqueeze_dim::<2>(1);
            let expert_input = routed_input.clone().select(0, token_indices.clone());
            let contribution = expert.forward(expert_input)?.cast(DType::F32) * weights;
            routed_output =
                routed_output.select_assign(0, token_indices, contribution, IndexingUpdateOp::Add);
        }
        let routed_output = routed_output.cast(routed_dtype);
        let routed_output = match &self.fc2_latent_proj {
            Some(projection) => operations::linear(routed_output, projection)?,
            None => routed_output,
        };
        let shared = self.shared.forward(flat)?;
        Ok(NemotronHMoeOutput {
            hidden_state: (routed_output + shared).reshape([batch, sequence, hidden]),
            router_logits,
            selected_experts: selected_experts_flat.reshape([
                batch,
                sequence,
                self.experts_per_token,
            ]),
            selected_weights: selected_weights.reshape([batch, sequence, self.experts_per_token]),
            expert_token_counts: counts,
            evaluated_experts: evaluated,
        })
    }
}

/// One pre-normalized heterogeneous residual block.
#[derive(Module, Debug)]
pub struct NemotronHBlock<B: Backend> {
    /// RMS normalization applied before the selected mixer.
    pub norm: RmsNorm<B>,
    /// Mamba2 mixer for `linear_attention` blocks.
    pub mamba: Option<NemotronMamba2<B>>,
    /// Causal GQA mixer for `full_attention` blocks.
    pub attention: Option<GroupedQueryAttention<B>>,
    /// Dense two-projection mixer for `mlp` blocks.
    pub mlp: Option<NemotronHDense<B>>,
    /// Sparse shared-plus-routed mixer for `moe` blocks.
    pub moe: Option<NemotronHMoe<B>>,
    /// Serialized block discriminator.
    pub block_type: NemotronHBlockType,
}

impl<B: Backend> NemotronHBlock<B> {
    /// Initializes exactly one mixer selected by the authoritative schedule.
    fn init(
        config: &NemotronHConfig,
        block_type: NemotronHBlockType,
        device: &B::Device,
    ) -> Result<Self> {
        let norm = nemotron_rms_norm(config.hidden_size, config.layer_norm_epsilon, device)?;
        let activation = activation_from_name(&config.mlp_hidden_act)?;
        let (mamba, attention, mlp, moe) = match block_type {
            NemotronHBlockType::LinearAttention => {
                (Some(mamba_config(config).init(device)?), None, None, None)
            }
            NemotronHBlockType::FullAttention => {
                let mut attention = GroupedQueryAttentionConfig::new(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                );
                attention.bias = config.attention_bias;
                (None, Some(attention.init(device)?), None, None)
            }
            NemotronHBlockType::Mlp => (
                None,
                None,
                Some(NemotronHDense::init(
                    config.hidden_size,
                    config.intermediate_size,
                    activation,
                    config.mlp_bias,
                    device,
                )?),
                None,
            ),
            NemotronHBlockType::Moe => {
                (None, None, None, Some(NemotronHMoe::init(config, device)?))
            }
        };
        Ok(Self {
            norm,
            mamba,
            attention,
            mlp,
            moe,
            block_type,
        })
    }

    /// Evaluates one stateless residual block and optional router output.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        sliding_window: Option<usize>,
    ) -> Result<(Tensor<B, 3>, Option<NemotronHMoeOutput<B>>)> {
        let residual = hidden.clone();
        let normalized = self.norm.forward(hidden);
        match (
            self.block_type,
            &self.mamba,
            &self.attention,
            &self.mlp,
            &self.moe,
        ) {
            (NemotronHBlockType::LinearAttention, Some(mamba), None, None, None) => Ok((
                residual
                    + mamba.forward(normalized, token_mask, NemotronMamba2Mode::ChunkedPrefill)?,
                None,
            )),
            (NemotronHBlockType::FullAttention, None, Some(attention), None, None) => {
                let projection = attention.project_self(normalized)?;
                let [batch, _, sequence, _] = projection.query.dims();
                let mask = stateless_attention_mask(
                    token_mask,
                    batch,
                    attention.query_heads,
                    sequence,
                    sliding_window,
                    &projection.query.device(),
                )?;
                Ok((
                    residual + attention.attend(projection, mask, None, true)?,
                    None,
                ))
            }
            (NemotronHBlockType::Mlp, None, None, Some(mlp), None) => {
                Ok((residual + mlp.forward(normalized)?, None))
            }
            (NemotronHBlockType::Moe, None, None, None, Some(moe)) => {
                let output = moe.forward(normalized)?;
                Ok((residual + output.hidden_state.clone(), Some(output)))
            }
            _ => Err(Error::Checkpoint(
                "Nemotron-H block contains an invalid mixer topology".into(),
            )),
        }
    }
}

/// Token embedding, heterogeneous residual stack, and final RMSNorm.
#[derive(Module, Debug)]
pub struct NemotronHModel<B: Backend> {
    /// Token embedding table.
    pub embed_tokens: Embedding<B>,
    /// Authoritative ordered heterogeneous blocks.
    pub layers: Vec<NemotronHBlock<B>>,
    /// Final RMS normalization.
    pub norm: RmsNorm<B>,
    /// Validated family configuration excluded from parameter records.
    #[module(skip)]
    pub config: NemotronHConfig,
}

impl<B: Backend> NemotronHModel<B> {
    /// Allocates the exact schedule declared by a validated config.
    pub fn init(config: NemotronHConfig, device: &B::Device) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        if config.attention_dropout != 0.0 || config.hidden_dropout != 0.0 {
            return Err(Error::BackendCapability(
                "Nemotron-H inference currently requires released zero-dropout configs".into(),
            ));
        }
        if config.residual_in_fp32 {
            return Err(Error::BackendCapability(
                "Nemotron-H FP32 residual accumulation is not implemented".into(),
            ));
        }
        let layers = config
            .layers_block_type
            .iter()
            .copied()
            .map(|kind| NemotronHBlock::init(&config, kind, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers,
            norm: nemotron_rms_norm(config.hidden_size, config.layer_norm_epsilon, device)?,
            config,
        })
    }

    /// Evaluates the stateless backbone and retains requested diagnostics.
    fn forward(
        &self,
        input: &CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<NemotronHBackboneOutput<B>> {
        let [batch, sequence] = input.token_ids.dims();
        validate_input(input, batch, sequence, &self.config, false)?;
        if self.embed_tokens.weight.val().device() != input.token_ids.device() {
            return Err(Error::BackendCapability(
                "Nemotron-H model parameters and inputs must share a device".into(),
            ));
        }
        let mut hidden = self.embed_tokens.forward(input.token_ids.clone());
        let mut hidden_states = Vec::with_capacity(if options.hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        let mut routers = Vec::new();
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        for layer in &self.layers {
            let (next, routed) = layer.forward(
                hidden,
                input.token_mask.as_ref(),
                self.config.sliding_window,
            )?;
            hidden = next;
            if options.hidden_states {
                hidden_states.push(hidden.clone());
            }
            if let Some(routed) = routed
                && (options.router_logits || options.auxiliary_losses)
            {
                routers.push(RouterDiagnostics {
                    logits: options.router_logits.then(|| routed.router_logits.clone()),
                    load_balancing_loss: options
                        .auxiliary_losses
                        .then(|| routed.load_balancing_loss(input.token_mask.as_ref()))
                        .transpose()?,
                    z_loss: None,
                });
            }
        }
        hidden = self.norm.forward(hidden);
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(NemotronHBackboneOutput {
            hidden,
            hidden_states,
            routers,
        })
    }
}

/// Internal backbone output before vocabulary projection.
struct NemotronHBackboneOutput<B: Backend> {
    hidden: Tensor<B, 3>,
    hidden_states: Vec<Tensor<B, 3>>,
    routers: Vec<RouterDiagnostics<B>>,
}

/// Optional checkpoint-declared one-step MTP body.
///
/// Lightning contains one next-token stage. It normalizes a shifted token
/// embedding and the base final hidden state independently, concatenates them,
/// projects back to the residual width, then applies the checkpoint-declared
/// attention/MoE schedule and a final norm. Vocabulary projection is shared with
/// the base language-model head.
#[derive(Module, Debug)]
pub struct NemotronHNextTokenHead<B: Backend> {
    /// Normalization for the shifted token embedding.
    pub embedding_norm: RmsNorm<B>,
    /// Normalization for the base final hidden state.
    pub hidden_norm: RmsNorm<B>,
    /// Concatenation projection from `2H` back to `H`.
    pub eh_proj: Linear<B>,
    /// Pre-normalization before MTP attention.
    pub attention_norm: RmsNorm<B>,
    /// Causal GQA MTP mixer.
    pub attention: GroupedQueryAttention<B>,
    /// Pre-normalization before MTP MoE.
    pub moe_norm: RmsNorm<B>,
    /// MTP shared-plus-routed expert mixer.
    pub moe: NemotronHMoe<B>,
    /// Final MTP normalization before the shared LM head.
    pub final_layer_norm: RmsNorm<B>,
    /// Optional causal window applied by the MTP attention stage.
    pub sliding_window: Option<usize>,
}

impl<B: Backend> NemotronHNextTokenHead<B> {
    /// Allocates the single Lightning MTP stage.
    fn init(config: &NemotronHConfig, device: &B::Device) -> Result<Self> {
        if config.num_nextn_predict_layers != 1
            || config.mtp_layers_block_type
                != [NemotronHBlockType::FullAttention, NemotronHBlockType::Moe]
        {
            return Err(Error::BackendCapability(
                "Nemotron-H MTP currently supports Lightning's single [attention, moe] stage"
                    .into(),
            ));
        }
        let fusion_width = config.hidden_size.checked_mul(2).ok_or_else(|| {
            Error::configuration(
                "nemotron_h.next_token_head.eh_proj",
                "2 * hidden_size overflows usize",
            )
        })?;
        let norm = || nemotron_rms_norm(config.hidden_size, config.layer_norm_epsilon, device);
        let mut attention = GroupedQueryAttentionConfig::new(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        );
        attention.bias = config.attention_bias;
        Ok(Self {
            embedding_norm: norm()?,
            hidden_norm: norm()?,
            eh_proj: operations::init_linear(
                &LinearConfig::new(fusion_width, config.hidden_size)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col),
                device,
            ),
            attention_norm: norm()?,
            attention: attention.init(device)?,
            moe_norm: norm()?,
            moe: NemotronHMoe::init(config, device)?,
            final_layer_norm: norm()?,
            sliding_window: config.sliding_window,
        })
    }

    /// Evaluates the MTP body and returns its normalized hidden state.
    pub fn forward(
        &self,
        shifted_embedding: Tensor<B, 3>,
        base_hidden: Tensor<B, 3>,
    ) -> Result<(Tensor<B, 3>, NemotronHMoeOutput<B>)> {
        let mut hidden = self.fuse(shifted_embedding, base_hidden)?;
        let residual = hidden.clone();
        let projection = self
            .attention
            .project_self(self.attention_norm.forward(hidden))?;
        let [batch, _, sequence, _] = projection.query.dims();
        let mask = stateless_attention_mask(
            None,
            batch,
            self.attention.query_heads,
            sequence,
            self.sliding_window,
            &projection.query.device(),
        )?;
        hidden = residual + self.attention.attend(projection, mask, None, true)?;
        self.finish(hidden)
    }

    /// Applies the independently normalized embedding/target fusion equation.
    ///
    /// The concatenation order is checkpoint-significant: vLLM 0.27.1 forms
    /// `[enorm(embedding), hnorm(target_hidden)]` before the `2H -> H` map.
    pub(super) fn fuse(
        &self,
        shifted_embedding: Tensor<B, 3>,
        base_hidden: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>> {
        if shifted_embedding.dims() != base_hidden.dims() {
            return Err(Error::shape(
                "nemotron_h.next_token_head",
                format!(
                    "matching shifted/base tensors, base={:?}",
                    base_hidden.dims()
                ),
                format!("shifted={:?}", shifted_embedding.dims()),
            ));
        }
        if shifted_embedding.device() != base_hidden.device()
            || self.eh_proj.weight.val().device() != base_hidden.device()
        {
            return Err(Error::BackendCapability(
                "Nemotron-H MTP inputs and parameters must share a device".into(),
            ));
        }
        let fused = Tensor::cat(
            vec![
                self.embedding_norm.forward(shifted_embedding),
                self.hidden_norm.forward(base_hidden),
            ],
            2,
        );
        operations::linear(fused, &self.eh_proj)
    }

    /// Applies the MoE residual and terminal normalization after attention.
    ///
    /// This is equivalent to vLLM's fused RMSNorm residual bookkeeping: the
    /// attention residual is materialized before MoE normalization, then the
    /// MoE result is added and normalized exactly once at the stage boundary.
    pub(super) fn finish(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<(Tensor<B, 3>, NemotronHMoeOutput<B>)> {
        let residual = hidden.clone();
        let routed = self.moe.forward(self.moe_norm.forward(hidden))?;
        let hidden = self
            .final_layer_norm
            .forward(residual + routed.hidden_state.clone());
        Ok((hidden, routed))
    }
}

/// Decoder-only Nemotron-H language model and optional MTP body.
#[derive(Module, Debug)]
pub struct NemotronHForCausalLm<B: Backend> {
    /// Hybrid backbone.
    pub model: NemotronHModel<B>,
    /// Independent vocabulary projection, absent for tied embeddings.
    pub lm_head: Option<Linear<B>>,
    /// Checkpoint-declared MTP body when `num_nextn_predict_layers == 1`.
    pub next_token_head: Option<NemotronHNextTokenHead<B>>,
}

impl<B: Backend> NemotronHForCausalLm<B> {
    /// Initializes a validated full-precision inference graph.
    pub fn init(config: NemotronHConfig, device: &B::Device) -> Result<Self> {
        // Validate before allocating the optional MTP body: malformed widths
        // must return a contextual error instead of overflowing in LinearConfig.
        ArchitectureConfig::validate(&config)?;
        let hidden = config.hidden_size;
        let vocabulary = config.vocab_size;
        let tied = config.tie_word_embeddings;
        let next_token_head = (config.num_nextn_predict_layers > 0)
            .then(|| NemotronHNextTokenHead::init(&config, device))
            .transpose()?;
        Ok(Self {
            model: NemotronHModel::init(config, device)?,
            lm_head: (!tied).then(|| {
                operations::init_linear(
                    &LinearConfig::new(hidden, vocabulary)
                        .with_bias(false)
                        .with_layout(LinearLayout::Col),
                    device,
                )
            }),
            next_token_head,
        })
    }

    /// Evaluates differentiable stateless inference or training math.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Nemotron-H attention probability capture is not yet exposed".into(),
            ));
        }
        let backbone = self.model.forward(&input, options)?;
        let hidden = backbone.hidden;
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
        let logits = self.project_logits(selected)?.cast(DType::F32);
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states: backbone.hidden_states,
            attentions: Vec::new(),
            routers: backbone.routers,
            load_balancing_loss: None,
        })
    }

    /// Returns the validated architecture configuration.
    #[must_use]
    pub const fn config(&self) -> &NemotronHConfig {
        &self.model.config
    }

    /// Runs the optional Lightning MTP body against aligned base hidden states.
    pub fn predict_next_token_logits(
        &self,
        shifted_token_ids: Tensor<B, 2, Int>,
        base_hidden: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>> {
        let head = self.next_token_head.as_ref().ok_or_else(|| {
            Error::BackendCapability("this Nemotron-H config has no next-token head".into())
        })?;
        let [batch, sequence] = shifted_token_ids.dims();
        if base_hidden.dims() != [batch, sequence, self.config().hidden_size] {
            return Err(Error::shape(
                "nemotron_h.next_token_head.base_hidden",
                format!("[{batch}, {sequence}, {}]", self.config().hidden_size),
                format!("{:?}", base_hidden.dims()),
            ));
        }
        validate_token_ids(shifted_token_ids.clone(), self.config().vocab_size)?;
        let embedding = self.model.embed_tokens.forward(shifted_token_ids);
        let (hidden, _) = head.forward(embedding, base_hidden)?;
        Ok(self.project_logits(hidden)?.cast(DType::F32))
    }

    /// Applies the independent or tied vocabulary projection.
    fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let weight_device = self.lm_head.as_ref().map_or_else(
            || self.model.embed_tokens.weight.val().device(),
            |head| head.weight.val().device(),
        );
        if weight_device != hidden.device() {
            return Err(Error::BackendCapability(
                "Nemotron-H LM head and hidden states must share a device".into(),
            ));
        }
        match &self.lm_head {
            Some(head) => operations::linear(hidden, head),
            None => Ok(burn::tensor::module::linear(
                hidden,
                self.model.embed_tokens.weight.val().transpose(),
                None,
            )),
        }
    }
}

/// Converts family activation metadata into the shared mathematical enum.
fn activation_from_name(name: &str) -> Result<Activation> {
    match name {
        "relu2" => Ok(Activation::ReluSquared),
        "silu" => Ok(Activation::Silu),
        _ => Err(Error::configuration(
            "nemotron_h.mlp_hidden_act",
            format!("unsupported activation {name:?}"),
        )),
    }
}

/// Initializes the exact Nemotron-H RMSNorm floating-point operation graph.
///
/// NVIDIA's reference first computes `x * rsqrt(mean(x^2) + epsilon)` in
/// FP32, rounds that unit-RMS activation back to the residual dtype, and only
/// then multiplies by the learned scale. This differs by representable F16
/// values from both division by `sqrt` and late rounding after the scale.
fn nemotron_rms_norm<B: Backend>(
    size: usize,
    epsilon: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>> {
    RmsNormConfig::new(size)
        .with_epsilon(epsilon)
        .with_round_before_scale(true)
        .with_root_mode(RmsNormRootMode::MultiplyByReciprocalSqrt)
        .init(device)
}

/// Builds one exact Mamba2 component config from the family schema.
fn mamba_config(config: &NemotronHConfig) -> NemotronMamba2Config {
    NemotronMamba2Config {
        hidden_size: config.hidden_size,
        num_heads: config.mamba_num_heads,
        head_dim: config.mamba_head_dim,
        state_size: config.ssm_state_size,
        groups: config.n_groups,
        convolution_kernel: config.conv_kernel,
        chunk_size: config.chunk_size,
        time_step_min: config.time_step_min,
        time_step_max: config.time_step_limit.get(1).copied().flatten(),
        normalization_epsilon: config.layer_norm_epsilon,
        projection_bias: config.use_bias,
        convolution_bias: config.use_conv_bias,
    }
}

/// Deterministically selects largest values, resolving ties by smallest index.
fn deterministic_top_k<B: Backend>(
    values: Tensor<B, 3>,
    k: usize,
) -> (Tensor<B, 3>, Tensor<B, 3, Int>) {
    let [batch, rows, columns] = values.dims();
    let ids = Tensor::<B, 1, Int>::arange(0..columns as i64, &values.device())
        .reshape([1, 1, columns])
        .expand([batch, rows, columns]);
    let mut remaining = values.clone();
    let mut selected_values = Vec::with_capacity(k);
    let mut selected_indices = Vec::with_capacity(k);
    for _ in 0..k {
        let maximum = remaining.clone().max_dim(2);
        let ties = remaining.clone().equal(maximum);
        let index = ids
            .clone()
            .mask_fill(ties.bool_not(), columns as i64)
            .min_dim(2);
        selected_values.push(values.clone().gather(2, index.clone()));
        selected_indices.push(index.clone());
        remaining = remaining.mask_fill(
            ids.clone().equal(index.expand([batch, rows, columns])),
            f32::NEG_INFINITY,
        );
    }
    (
        Tensor::cat(selected_values, 2),
        Tensor::cat(selected_indices, 2),
    )
}

/// Combines true-valid padding semantics with an optional causal left window.
fn stateless_attention_mask<B: Backend>(
    token_mask: Option<&TokenMask<B>>,
    batch: usize,
    heads: usize,
    sequence: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Result<Option<Tensor<B, 4, Bool>>> {
    let padding = padding_attention_mask(token_mask, token_mask, batch, heads, sequence, sequence)?;
    let Some(window) = window else {
        return Ok(padding);
    };
    let end = i64::try_from(sequence).map_err(|_| {
        Error::configuration(
            "nemotron_h.attention_mask",
            "sequence length exceeds portable signed index range",
        )
    })?;
    let query = Tensor::<B, 1, Int>::arange(0..end, device)
        .reshape([1, 1, sequence, 1])
        .expand([batch, heads, sequence, sequence]);
    let key = Tensor::<B, 1, Int>::arange(0..end, device)
        .reshape([1, 1, 1, sequence])
        .expand([batch, heads, sequence, sequence]);
    let window = i64::try_from(window).map_err(|_| {
        Error::configuration(
            "nemotron_h.sliding_window",
            "window exceeds portable signed index range",
        )
    })?;
    let too_old = (key + window).lower_equal(query);
    Ok(Some(match padding {
        Some(padding) => padding.bool_or(too_old),
        None => too_old,
    }))
}

/// Validates token, mask, and position tensors before backend indexing.
fn validate_input<B: Backend>(
    input: &CausalLmInput<B>,
    batch: usize,
    sequence: usize,
    config: &NemotronHConfig,
    permit_empty_rows: bool,
) -> Result<()> {
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "nemotron_h.input_ids",
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    validate_token_ids(input.token_ids.clone(), config.vocab_size)?;
    if let Some(mask) = &input.token_mask {
        if mask.as_tensor().device() != input.token_ids.device() {
            return Err(Error::BackendCapability(
                "Nemotron-H token IDs and token mask must share a device".into(),
            ));
        }
        if mask.as_tensor().dims() != [batch, sequence] {
            return Err(Error::shape(
                "nemotron_h.token_mask",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", mask.as_tensor().dims()),
            ));
        }
        if !permit_empty_rows {
            let minimum = mask
                .as_tensor()
                .clone()
                .int()
                .sum_dim(1)
                .min()
                .try_into_scalar()
                .map_err(|error| {
                    Error::BackendCapability(format!("failed to validate Nemotron mask: {error}"))
                })?
                .elem::<i64>();
            if minimum == 0 {
                return Err(Error::shape(
                    "nemotron_h.token_mask",
                    "at least one valid token in each row",
                    "a fully masked row",
                ));
            }
        }
    }
    if let Some(positions) = &input.position_ids {
        if positions.device() != input.token_ids.device() {
            return Err(Error::BackendCapability(
                "Nemotron-H token IDs and position IDs must share a device".into(),
            ));
        }
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "nemotron_h.position_ids",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
        let minimum = positions
            .clone()
            .min()
            .try_into_scalar()
            .map_err(|error| {
                Error::BackendCapability(format!("position validation failed: {error}"))
            })?
            .elem::<i64>();
        let maximum = positions
            .clone()
            .max()
            .try_into_scalar()
            .map_err(|error| {
                Error::BackendCapability(format!("position validation failed: {error}"))
            })?
            .elem::<i64>();
        if minimum < 0 || maximum as usize >= config.max_position_embeddings {
            return Err(Error::shape(
                "nemotron_h.position_ids",
                format!("values in 0..{}", config.max_position_embeddings),
                format!("observed {minimum}..={maximum}"),
            ));
        }
    }
    Ok(())
}

/// Rejects out-of-vocabulary indices before embedding lookup.
fn validate_token_ids<B: Backend>(token_ids: Tensor<B, 2, Int>, vocabulary: usize) -> Result<()> {
    let minimum = token_ids
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("token validation failed: {error}")))?
        .elem::<i64>();
    let maximum = token_ids
        .max()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("token validation failed: {error}")))?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= vocabulary {
        return Err(Error::shape(
            "nemotron_h.input_ids",
            format!("values in 0..{vocabulary}"),
            format!("observed {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Applies the requested vocabulary-projection position policy.
fn select_hidden<B: Backend>(
    hidden: Tensor<B, 3>,
    mask: Option<&TokenMask<B>>,
    selection: &LogitsSelection,
) -> Result<Tensor<B, 3>> {
    let [batch, sequence, width] = hidden.dims();
    match selection {
        LogitsSelection::All => Ok(hidden),
        LogitsSelection::Last => match mask {
            None => Ok(hidden.narrow(1, sequence - 1, 1)),
            Some(mask) => {
                let positions = Tensor::<B, 1, Int>::arange(0..sequence as i64, &hidden.device())
                    .reshape([1, sequence])
                    .expand([batch, sequence]);
                let last = positions
                    .mask_fill(mask.as_tensor().clone().bool_not(), -1)
                    .argmax(1)
                    .unsqueeze_dim::<3>(2)
                    .expand([batch, 1, width]);
                Ok(hidden.gather(1, last))
            }
        },
        LogitsSelection::Positions(positions) => {
            if positions.is_empty() || positions.iter().any(|&position| position >= sequence) {
                return Err(Error::shape(
                    "nemotron_h.logits.positions",
                    format!("non-empty values in 0..{sequence}"),
                    format!("{positions:?}"),
                ));
            }
            let values = positions
                .iter()
                .map(|&value| value as i64)
                .collect::<Vec<_>>();
            let selector = Tensor::<B, 1, Int>::from_data(
                TensorData::new(values, [positions.len()]),
                &hidden.device(),
            );
            Ok(hidden.select(1, selector))
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        tensor::{Tolerance, f16},
    };

    use super::*;

    type TestBackend = Flex<f32, i32>;

    /// Converts one test projection to F16 without changing its logical shape.
    fn cast_linear_to_f16(linear: &mut Linear<TestBackend>) {
        linear.weight = Param::from_tensor(linear.weight.val().cast(DType::F16));
        if let Some(bias) = linear.bias.take() {
            linear.bias = Some(Param::from_tensor(bias.val().cast(DType::F16)));
        }
    }

    /// Converts both projections of one test expert to F16.
    fn cast_dense_to_f16(dense: &mut NemotronHDense<TestBackend>) {
        cast_linear_to_f16(&mut dense.up_proj);
        cast_linear_to_f16(&mut dense.down_proj);
    }

    /// Reduced schedule retaining every released block kind.
    pub(crate) fn tiny_config(mtp: bool) -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 32,
            hidden_size: 8,
            layers_block_type: vec![
                NemotronHBlockType::LinearAttention,
                NemotronHBlockType::FullAttention,
                NemotronHBlockType::Mlp,
                NemotronHBlockType::Moe,
            ],
            num_hidden_layers: Some(4),
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 32,
            intermediate_size: 12,
            ssm_state_size: 3,
            mamba_num_heads: 4,
            mamba_head_dim: 2,
            n_groups: 2,
            conv_kernel: 3,
            chunk_size: 2,
            n_routed_experts: 4,
            n_shared_experts: 1,
            moe_intermediate_size: 6,
            moe_shared_expert_intermediate_size: 8,
            num_experts_per_tok: 2,
            n_group: 2,
            topk_group: 1,
            num_nextn_predict_layers: usize::from(mtp),
            ..Default::default()
        }
    }

    /// The schedule instantiates exactly one matching component in every block.
    #[test]
    fn schedule_builds_each_mixer_kind() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
        assert!(model.model.layers[0].mamba.is_some());
        assert!(model.model.layers[1].attention.is_some());
        assert!(model.model.layers[2].mlp.is_some());
        assert!(model.model.layers[3].moe.is_some());
    }

    /// Nemotron's upstream `rsqrt` plus early-rounding graph is F16 observable.
    ///
    /// The expected bits come from the pinned Transformers Nemotron-H fallback
    /// expression `weight * (x.float() * torch.rsqrt(ms)).to(x.dtype)`.
    #[test]
    fn rms_norm_matches_pinned_f16_operation_order() {
        let device = Default::default();
        let mut norm = nemotron_rms_norm::<TestBackend>(2, 1e-6, &device).unwrap();
        norm.weight = Param::from_tensor(
            Tensor::from_floats([0.288_818_36, -3.437_5], &device).cast(DType::F16),
        );
        let input = Tensor::<TestBackend, 2>::from_floats([[7.457_031_25, -0.452_148_44]], &device)
            .cast(DType::F16);

        assert!(norm.round_before_scale);
        assert_eq!(norm.root_mode, RmsNormRootMode::MultiplyByReciprocalSqrt);
        let bits = norm
            .forward(input)
            .into_data()
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(bits, [0x3685, 0x34b5]);
    }

    /// Stateless inference honors output and router diagnostic selections.
    #[test]
    fn stateless_forward_returns_requested_shapes() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
        let output = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    hidden_states: true,
                    router_logits: true,
                    auxiliary_losses: true,
                    ..ForwardOptions::default()
                },
            )
            .unwrap();
        assert_eq!(output.logits.dims(), [1, 3, 32]);
        assert_eq!(output.logits.dtype(), DType::F32);
        assert_eq!(output.hidden_states.len(), 6);
        assert_eq!(output.routers.len(), 1);
        assert_eq!(output.routers[0].logits.as_ref().unwrap().dims(), [1, 3, 4]);
        assert!(output.routers[0].load_balancing_loss.is_some());
    }

    /// Sparse routing never evaluates experts without assignments.
    #[test]
    fn moe_dispatches_only_selected_experts() {
        let device = Default::default();
        let config = tiny_config(false);
        let moe = NemotronHMoe::<TestBackend>::init(&config, &device).unwrap();
        let output = moe.forward(Tensor::zeros([1, 1, 8], &device)).unwrap();
        assert_eq!(output.expert_token_counts.iter().sum::<usize>(), 2);
        assert_eq!(output.evaluated_experts, 2);
        assert_eq!(output.expert_token_counts, vec![1, 1, 0, 0]);
    }

    /// Lightning's single router group is the identity eligibility transform.
    #[test]
    fn moe_single_group_matches_all_groups_retained() {
        let device = Default::default();
        let config = tiny_config(false);
        let mut moe = NemotronHMoe::<TestBackend>::init(&config, &device).unwrap();
        let input = Tensor::from_floats([[[0.2, -0.5, 0.8, 0.1, -0.3, 0.7, 0.4, -0.9]]], &device);

        moe.groups = 1;
        moe.top_groups = 1;
        let single_group = moe.forward(input.clone()).unwrap();

        moe.groups = 2;
        moe.top_groups = 2;
        let all_groups = moe.forward(input).unwrap();

        assert_eq!(
            single_group.selected_experts.into_data(),
            all_groups.selected_experts.into_data()
        );
        single_group
            .selected_weights
            .into_data()
            .assert_approx_eq::<f32>(
                &all_groups.selected_weights.into_data(),
                Tolerance::absolute(0.0),
            );
        single_group
            .hidden_state
            .into_data()
            .assert_approx_eq::<f32>(
                &all_groups.hidden_state.into_data(),
                Tolerance::absolute(1e-6),
            );
    }

    /// Router buffers and sparse accumulation retain their reference precisions.
    #[test]
    fn moe_mixed_precision_keeps_router_and_accumulator_fp32() {
        let device = Default::default();
        let config = tiny_config(false);
        let mut moe = NemotronHMoe::<TestBackend>::init(&config, &device).unwrap();
        cast_linear_to_f16(&mut moe.router);
        for expert in &mut moe.routed {
            cast_dense_to_f16(expert);
        }
        cast_dense_to_f16(&mut moe.shared);
        if let Some(projection) = &mut moe.fc1_latent_proj {
            cast_linear_to_f16(projection);
        }
        if let Some(projection) = &mut moe.fc2_latent_proj {
            cast_linear_to_f16(projection);
        }

        assert_eq!(moe.correction_bias.val().dtype(), DType::F32);
        assert!(!moe.correction_bias.val().is_require_grad());
        let output = moe
            .forward(Tensor::ones([1, 3, 8], (&device, DType::F16)))
            .unwrap();
        assert_eq!(output.hidden_state.dtype(), DType::F16);
        assert_eq!(output.router_logits.dtype(), DType::F32);
        assert_eq!(output.selected_weights.dtype(), DType::F32);
        assert!(
            output
                .hidden_state
                .cast(DType::F32)
                .to_data()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    /// Invalid MTP widths are rejected before any parameter allocation occurs.
    #[test]
    fn mtp_width_overflow_returns_configuration_error() {
        let device = Default::default();
        let mut config = tiny_config(true);
        config.hidden_size = usize::MAX;
        let error = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("MTP concatenation width overflows")
        );
    }

    /// The optional MTP body consumes aligned base hidden states and shared head.
    #[test]
    fn checkpoint_declared_next_token_head_runs() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
        let logits = model
            .predict_next_token_logits(
                Tensor::from_ints([[2, 3]], &device),
                Tensor::zeros([1, 2, 8], &device),
            )
            .unwrap();
        assert_eq!(logits.dims(), [1, 2, 32]);
    }

    /// Invalid block diagnostics and token indices fail without panicking.
    #[test]
    fn invalid_inputs_are_contextual_errors() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
        let error = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[99]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("0..32"));
        assert!(
            model
                .forward(
                    CausalLmInput {
                        token_ids: Tensor::from_ints([[1]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    &ForwardOptions {
                        attentions: true,
                        ..ForwardOptions::default()
                    },
                )
                .is_err()
        );
    }

    /// All-logit and last-logit projections agree at the final position.
    #[test]
    fn last_logits_match_full_forward_tail() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
        let input = || CausalLmInput {
            token_ids: Tensor::from_ints([[1, 2, 3]], &device),
            token_mask: None,
            position_ids: None,
        };
        let all = model
            .forward(
                input(),
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..ForwardOptions::default()
                },
            )
            .unwrap()
            .logits
            .narrow(1, 2, 1);
        let last = model
            .forward(input(), &ForwardOptions::default())
            .unwrap()
            .logits;
        last.into_data()
            .assert_approx_eq::<f32>(&all.into_data(), Tolerance::absolute(1e-5));
    }

    /// Full-precision stateless execution retains parameter gradients.
    #[cfg(feature = "autodiff")]
    #[test]
    fn autodiff_smoke_reaches_embedding() {
        use burn::backend::Autodiff;

        type AdBackend = Autodiff<TestBackend>;
        let device = Default::default();
        let model = NemotronHForCausalLm::<AdBackend>::init(tiny_config(false), &device).unwrap();
        let loss = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
            .sum();
        let gradients = loss.backward();
        assert!(model.model.embed_tokens.weight.grad(&gradients).is_some());
    }
}
