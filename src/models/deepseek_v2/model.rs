//! Executable DeepSeek-V2 decoder with MLA and sparse shared-expert blocks.
//!
//! A decoder block is pre-normalized and evaluates
//! `x' = x + MLA(RMSNorm(x))`, then
//! `y = x' + FFN(RMSNorm(x'))`. The pinned Lite checkpoint uses one dense
//! SwiGLU block followed by 26 blocks that add a top-6 routed mixture to one
//! always-active SwiGLU whose intermediate width is exactly twice an expert's
//! width. The widened shared network is intentionally not represented as two
//! independent experts because its checkpoint contains only three matrices.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{
        Bool, DType, ElementConversion, Int, Tensor, TensorData, activation::softmax,
        backend::Backend,
    },
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, LogitsSelection,
    Result,
    architecture::{RouterDiagnostics, TokenMask},
    attention::padding_attention_mask,
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    moe::{SparseMoe, SparseMoeConfig, SparseMoeOutput, load_balancing_loss},
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling, YarnConfig},
};

use super::{
    DeepseekScoringFunction, DeepseekTopKMethod, DeepseekV2Config,
    mla::{MlaMixOutput, MultiLatentAttention, MultiLatentAttentionConfig},
};
use crate::models::config::RopeType;

mod decode;

pub use decode::{DeepseekV2DecodeState, MlaExecutionPath};

/// Routed expert output together with the always-active shared contribution.
#[derive(Clone, Debug)]
struct DeepseekMoeOutput<B: Backend> {
    /// Sum of selected routed contributions and the widened shared network.
    hidden_state: Tensor<B, 3>,
    /// Routing choices and accounting retained for optional diagnostics.
    routed: SparseMoeOutput<B>,
}

/// DeepSeek's top-k routed bank plus one widened shared SwiGLU network.
#[derive(Module, Debug)]
pub struct DeepseekV2Moe<B: Backend> {
    /// Selected-only routed expert dispatcher.
    pub routed: SparseMoe<B>,
    /// One always-active checkpoint module of width `n_shared * expert_width`.
    pub shared_experts: GatedFeedForward<B>,
}

impl<B: Backend> DeepseekV2Moe<B> {
    /// Allocates the exact single-process V2-Lite expert topology.
    fn init(config: &DeepseekV2Config, device: &B::Device) -> Result<Self> {
        if config.ep_size != 1 {
            return Err(Error::BackendCapability(
                "DeepSeek expert-parallel checkpoints require distributed execution, which is deferred from v1"
                    .into(),
            ));
        }
        if config.scoring_func != DeepseekScoringFunction::Softmax
            || config.topk_method != DeepseekTopKMethod::Greedy
        {
            return Err(Error::BackendCapability(
                "the v1 DeepSeek-V2 runtime implements the pinned softmax/greedy router only"
                    .into(),
            ));
        }
        let mut routed = SparseMoeConfig::new(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            config.num_experts_per_tok,
        );
        // The reference only normalizes a multi-expert mixture. For top-1
        // routing, normalization would erase the router's confidence and make
        // every selected route have unit weight.
        let renormalize_selected = config.num_experts_per_tok > 1 && config.norm_topk_prob;
        routed.renormalize_selected = renormalize_selected;
        // The reference applies routed_scaling_factor only on its unnormalized
        // branch. Lite uses `norm_topk_prob=false`, but retaining this condition
        // keeps other V2 configs mathematically faithful.
        routed.routed_scale = if renormalize_selected {
            1.0
        } else {
            config.routed_scaling_factor
        };
        let shared_width = config
            .moe_intermediate_size
            .checked_mul(config.n_shared_experts)
            .ok_or_else(|| {
                Error::configuration(
                    "deepseek_v2.shared_experts",
                    "intermediate width product overflows usize",
                )
            })?;
        Ok(Self {
            routed: routed.init(device)?,
            shared_experts: GatedFeedForwardConfig::new(config.hidden_size, shared_width)
                .init(device)?,
        })
    }

    /// Evaluates only selected routed experts and always evaluates the shared MLP.
    fn forward(&self, input: Tensor<B, 3>) -> Result<DeepseekMoeOutput<B>> {
        let [batch, sequence, hidden] = input.dims();
        if batch == 0 || sequence == 0 || hidden != self.routed.hidden_size {
            return Err(Error::shape(
                "deepseek_v2.moe.input",
                format!("[batch>0, sequence>0, {}]", self.routed.hidden_size),
                format!("[{batch}, {sequence}, {hidden}]"),
            ));
        }
        if self.routed.experts_per_token == 0
            || self.routed.experts_per_token > self.routed.routed.len()
        {
            return Err(Error::configuration(
                "deepseek_v2.moe.experts_per_token",
                "must fit the instantiated routed expert bank",
            ));
        }
        let tokens = batch.checked_mul(sequence).ok_or_else(|| {
            Error::shape(
                "deepseek_v2.moe.input",
                "a representable batch * sequence product",
                format!("{batch} * {sequence}"),
            )
        })?;
        let input_dtype = input.dtype();
        let flat_input = input.clone().reshape([tokens, hidden]);
        // The official router casts both operands to FP32 before the projection,
        // not merely before softmax. This avoids an F16 rounding event changing
        // the selected expert and therefore an entire nonlinear subgraph.
        let router_logits = burn::tensor::module::linear(
            input.clone().cast(DType::F32),
            self.routed.router.weight.val().cast(DType::F32),
            None,
        );
        let probabilities = softmax(router_logits.clone(), 2);
        let (selected_weights, selected_experts) =
            deterministic_top_k(probabilities, self.routed.experts_per_token);
        let mut flat_weights = selected_weights
            .clone()
            .reshape([tokens, self.routed.experts_per_token]);
        if self.routed.renormalize_selected {
            let denominator = flat_weights.clone().sum_dim(1).clamp_min(f32::MIN_POSITIVE);
            flat_weights = flat_weights / denominator;
        }
        flat_weights = flat_weights * self.routed.routed_scale;
        let flat_experts = selected_experts
            .clone()
            .reshape([tokens, self.routed.experts_per_token]);
        // The reference keeps probabilities, weighted expert contributions,
        // and index-add accumulation in FP32. Only the complete routed mixture
        // is rounded back to the residual dtype, avoiding one rounding event
        // per expert contribution in reduced precision.
        let dispatch = self.routed.dispatch_selected_experts(
            flat_input,
            flat_weights.clone(),
            flat_experts,
            DType::F32,
        )?;
        let routed = SparseMoeOutput {
            hidden_state: dispatch
                .hidden_state
                .cast(input_dtype)
                .reshape([batch, sequence, hidden]),
            router_logits,
            selected_experts,
            selected_weights: flat_weights.reshape([
                batch,
                sequence,
                self.routed.experts_per_token,
            ]),
            expert_token_counts: dispatch.expert_token_counts,
            evaluated_experts: dispatch.evaluated_experts,
        };
        Ok(DeepseekMoeOutput {
            hidden_state: routed.hidden_state.clone() + self.shared_experts.forward(input)?,
            routed,
        })
    }
}

/// Selects largest probabilities with a backend-independent smallest-ID tie rule.
fn deterministic_top_k<B: Backend>(
    probabilities: Tensor<B, 3>,
    k: usize,
) -> (Tensor<B, 3>, Tensor<B, 3, Int>) {
    let [batch, sequence, experts] = probabilities.dims();
    let device = probabilities.device();
    let expert_ids = Tensor::<B, 1, Int>::arange(0..experts as i64, &device)
        .reshape([1, 1, experts])
        .expand([batch, sequence, experts]);
    let mut remaining = probabilities.clone();
    let mut weights = Vec::with_capacity(k);
    let mut indices = Vec::with_capacity(k);
    for _ in 0..k {
        let maximum = remaining.clone().max_dim(2);
        let tied = remaining.clone().equal(maximum);
        let selected = expert_ids
            .clone()
            .mask_fill(tied.bool_not(), experts as i64)
            .min_dim(2);
        weights.push(probabilities.clone().gather(2, selected.clone()));
        indices.push(selected.clone());
        remaining = remaining.mask_fill(
            expert_ids
                .clone()
                .equal(selected.expand([batch, sequence, experts])),
            f32::NEG_INFINITY,
        );
    }
    (Tensor::cat(weights, 2), Tensor::cat(indices, 2))
}

/// One DeepSeek-V2 decoder layer.
#[derive(Module, Debug)]
pub struct DeepseekV2DecoderLayer<B: Backend> {
    /// Multi-head latent self-attention.
    pub self_attn: MultiLatentAttention<B>,
    /// Routed plus shared experts on sparse layers.
    pub sparse_mlp: Option<DeepseekV2Moe<B>>,
    /// Dense SwiGLU on the initial dense prefix.
    pub dense_mlp: Option<GatedFeedForward<B>>,
    /// RMS normalization before MLA.
    pub input_layernorm: RmsNorm<B>,
    /// RMS normalization before the dense or sparse feed-forward branch.
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> DeepseekV2DecoderLayer<B> {
    /// Allocates one validated layer at its exact dense/MoE schedule position.
    fn init(config: &DeepseekV2Config, layer_index: usize, device: &B::Device) -> Result<Self> {
        let mut attention = MultiLatentAttentionConfig::new(
            config.hidden_size,
            config.num_attention_heads,
            config.qk_nope_head_dim,
            config.qk_rope_head_dim,
            config.v_head_dim,
            config.kv_lora_rank,
            config.q_lora_rank,
        );
        attention.rms_norm_epsilon = config.rms_norm_eps;
        attention.bias = config.attention_bias;
        attention.softmax_scale = deepseek_attention_scale(config)?;
        let norm = || {
            RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_round_before_scale(true)
                .init(device)
        };
        let (sparse_mlp, dense_mlp) = if config.is_moe_layer(layer_index) {
            (Some(DeepseekV2Moe::init(config, device)?), None)
        } else {
            (
                None,
                Some(
                    GatedFeedForwardConfig::new(config.hidden_size, config.intermediate_size)
                        .init(device)?,
                ),
            )
        };
        Ok(Self {
            self_attn: attention.init(device)?,
            sparse_mlp,
            dense_mlp,
            input_layernorm: norm()?,
            post_attention_layernorm: norm()?,
        })
    }

    /// Evaluates one stateless layer with reference unabsorbed MLA prefill math.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        rope: RopeConfig,
        evaluated_context: usize,
        retain_attention: bool,
    ) -> Result<LayerOutput<B>> {
        let residual = hidden.clone();
        let projection = self.self_attn.project(
            self.input_layernorm.forward(hidden),
            positions,
            rope,
            evaluated_context,
        )?;
        let [batch, heads, sequence, _] = projection.query_content.dims();
        let mask =
            stateless_attention_mask(token_mask, batch, heads, sequence, &residual.device())?;
        let mixed = self.self_attn.mix_unabsorbed(
            &projection,
            projection.normalized_latent.clone(),
            projection.rotated_key.clone(),
            Some(mask),
            retain_attention,
        )?;
        self.finish_block(residual + mixed.hidden_state.clone(), mixed)
    }

    /// Applies the post-attention residual and exactly one feed-forward topology.
    fn finish_block(
        &self,
        hidden: Tensor<B, 3>,
        attention: MlaMixOutput<B>,
    ) -> Result<LayerOutput<B>> {
        let residual = hidden.clone();
        let normalized = self.post_attention_layernorm.forward(hidden);
        match (&self.sparse_mlp, &self.dense_mlp) {
            (Some(moe), None) => {
                let output = moe.forward(normalized)?;
                Ok(LayerOutput {
                    hidden: residual + output.hidden_state,
                    attention: attention.probabilities,
                    routed: Some(output.routed),
                })
            }
            (None, Some(mlp)) => Ok(LayerOutput {
                hidden: residual + mlp.forward(normalized)?,
                attention: attention.probabilities,
                routed: None,
            }),
            _ => Err(Error::Checkpoint(
                "DeepSeek-V2 layer has an invalid dense/MoE topology".into(),
            )),
        }
    }
}

/// Private result of one decoder layer.
struct LayerOutput<B: Backend> {
    /// Residual stream after attention and feed-forward updates.
    hidden: Tensor<B, 3>,
    /// Attention probabilities when explicitly requested.
    attention: Option<Tensor<B, 4>>,
    /// Sparse-router internals, absent on dense-prefix layers.
    routed: Option<SparseMoeOutput<B>>,
}

/// DeepSeek token embedding, decoder stack, and final RMS normalization.
#[derive(Module, Debug)]
pub struct DeepseekV2Model<B: Backend> {
    /// Token embedding table.
    pub embed_tokens: Embedding<B>,
    /// Ordered causal decoder layers.
    pub layers: Vec<DeepseekV2DecoderLayer<B>>,
    /// Final residual-stream RMS normalization.
    pub norm: RmsNorm<B>,
    /// Validated family configuration excluded from parameter records.
    #[module(skip)]
    pub config: DeepseekV2Config,
    /// Validated rotary algorithm shared by every MLA layer.
    #[module(skip)]
    pub rope: RopeConfig,
}

impl<B: Backend> DeepseekV2Model<B> {
    /// Initializes a full-precision graph from a validated local configuration.
    pub fn init(config: DeepseekV2Config, device: &B::Device) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        if !config.use_mla {
            return Err(Error::BackendCapability(
                "the DeepSeek-V2 v1 slice supports MLA checkpoints only".into(),
            ));
        }
        if config.attention_dropout != 0.0 {
            return Err(Error::BackendCapability(
                "DeepSeek-V2 inference requires the released zero attention dropout".into(),
            ));
        }
        let rope = deepseek_rope_config(&config)?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| DeepseekV2DecoderLayer::init(&config, index, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers,
            norm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_round_before_scale(true)
                .init(device)?,
            config,
            rope,
        })
    }

    /// Evaluates the stateless backbone while retaining only requested diagnostics.
    fn forward(
        &self,
        input: &CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<BackboneOutput<B>> {
        let [batch, sequence] = input.token_ids.dims();
        validate_nonempty_input(batch, sequence)?;
        let model_device = self.embed_tokens.weight.lazy_device();
        ensure_model_device(
            "deepseek_v2.input_ids",
            &input.token_ids.device(),
            &model_device,
        )?;
        if let Some(mask) = &input.token_mask {
            ensure_model_device(
                "deepseek_v2.token_mask",
                &mask.as_tensor().device(),
                &model_device,
            )?;
        }
        if let Some(positions) = &input.position_ids {
            ensure_model_device(
                "deepseek_v2.position_ids",
                &positions.device(),
                &model_device,
            )?;
        }
        validate_token_ids(input.token_ids.clone(), self.config.vocab_size)?;
        validate_mask(input.token_mask.as_ref(), batch, sequence, false)?;
        let positions = resolve_positions(input, batch, sequence)?;
        validate_position_range(positions.clone(), self.config.max_position_embeddings)?;
        let context = evaluated_context(positions.clone())?;
        let mut hidden = self.embed_tokens.forward(input.token_ids.clone());
        let mut hidden_states = Vec::with_capacity(if options.hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        let mut attentions = Vec::with_capacity(if options.attentions {
            self.layers.len()
        } else {
            0
        });
        let mut routers =
            Vec::with_capacity(if options.router_logits || options.auxiliary_losses {
                self.layers
                    .iter()
                    .filter(|layer| layer.sparse_mlp.is_some())
                    .count()
            } else {
                0
            });
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        for layer in &self.layers {
            let output = layer.forward(
                hidden,
                positions.clone(),
                input.token_mask.as_ref(),
                self.rope,
                context,
                options.attentions,
            )?;
            hidden = output.hidden;
            if options.hidden_states {
                hidden_states.push(hidden.clone());
            }
            if let Some(attention) = output.attention {
                attentions.push(attention);
            }
            if let Some(routed) = output.routed
                && (options.router_logits || options.auxiliary_losses)
            {
                routers.push(router_diagnostics(
                    &routed,
                    input.token_mask.as_ref(),
                    options,
                    &self.config,
                )?);
            }
        }
        hidden = self.norm.forward(hidden);
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(BackboneOutput {
            hidden,
            hidden_states,
            attentions,
            routers,
        })
    }
}

/// Internal backbone result.
struct BackboneOutput<B: Backend> {
    /// Final normalized residual stream.
    hidden: Tensor<B, 3>,
    /// Requested embedding and per-layer residual snapshots.
    hidden_states: Vec<Tensor<B, 3>>,
    /// Requested reference attention probabilities.
    attentions: Vec<Tensor<B, 4>>,
    /// Requested sparse-layer routing diagnostics.
    routers: Vec<RouterDiagnostics<B>>,
}

/// Decoder-only DeepSeek-V2 language model.
#[derive(Module, Debug)]
pub struct DeepseekV2ForCausalLm<B: Backend> {
    /// DeepSeek transformer backbone.
    pub model: DeepseekV2Model<B>,
    /// Independent vocabulary projection, absent for tied embeddings.
    pub lm_head: Option<Linear<B>>,
}

impl<B: Backend> DeepseekV2ForCausalLm<B> {
    /// Initializes the complete causal language model.
    pub fn init(config: DeepseekV2Config, device: &B::Device) -> Result<Self> {
        let hidden = config.hidden_size;
        let vocabulary = config.vocab_size;
        let tied = config.tie_word_embeddings;
        Ok(Self {
            model: DeepseekV2Model::init(config, device)?,
            lm_head: (!tied).then(|| {
                operations::init_linear(
                    &LinearConfig::new(hidden, vocabulary)
                        .with_bias(false)
                        .with_layout(LinearLayout::Col),
                    device,
                )
            }),
        })
    }

    /// Evaluates differentiable stateless inference or training math.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        let backbone = self.model.forward(&input, options)?;
        let hidden = backbone.hidden;
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
        let logits = self.project_logits(selected)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states: backbone.hidden_states,
            attentions: backbone.attentions,
            routers: backbone.routers,
            load_balancing_loss: None,
        })
    }

    /// Returns the validated checkpoint configuration.
    #[must_use]
    pub const fn config(&self) -> &DeepseekV2Config {
        &self.model.config
    }

    /// Applies the independent or tied vocabulary projection.
    fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        match &self.lm_head {
            Some(head) => operations::linear(hidden, head).map(|logits| logits.cast(DType::F32)),
            None => Ok(burn::tensor::module::linear(
                hidden,
                self.model.embed_tokens.weight.val().transpose(),
                None,
            )
            .cast(DType::F32)),
        }
    }
}

/// Converts serialized rotary metadata into the shared YaRN implementation.
fn deepseek_rope_config(config: &DeepseekV2Config) -> Result<RopeConfig> {
    let parameters = config.rope_scaling.clone().unwrap_or_default();
    let theta = parameters.rope_theta.unwrap_or(config.rope_theta);
    let scaling = match parameters.rope_type {
        RopeType::Default => RopeScaling::None,
        RopeType::Linear => LinearRopeScaling::new(parameters.factor.unwrap_or(1.0))
            .map(RopeScaling::from)
            .map_err(|error| Error::configuration("deepseek_v2.rope", error.to_string()))?,
        RopeType::Dynamic => DynamicNtkScaling::new(
            parameters.factor.unwrap_or(1.0),
            parameters
                .original_max_position_embeddings
                .unwrap_or(config.max_position_embeddings),
        )
        .map(RopeScaling::from)
        .map_err(|error| Error::configuration("deepseek_v2.rope", error.to_string()))?,
        RopeType::Yarn => {
            let factor = parameters.factor.unwrap_or(1.0);
            let mut yarn = YarnConfig::new(
                factor,
                parameters
                    .original_max_position_embeddings
                    .unwrap_or(config.max_position_embeddings),
            )
            .map_err(|error| Error::configuration("deepseek_v2.rope", error.to_string()))?;
            let amplitude = if parameters.mscale.is_some() || parameters.mscale_all_dim.is_some() {
                yarn_mscale(factor, parameters.mscale.unwrap_or(1.0))
                    / yarn_mscale(factor, parameters.mscale_all_dim.unwrap_or(0.0))
            } else {
                parameters
                    .attention_factor
                    .unwrap_or_else(|| yarn.attention_factor())
            };
            yarn = yarn
                .with_attention_factor(amplitude)
                .map_err(|error| Error::configuration("deepseek_v2.rope", error.to_string()))?;
            // Hugging Face resolves each optional bound independently. Using
            // the shared YaRN defaults for only the missing side preserves
            // one-sided custom checkpoint configurations.
            yarn = yarn
                .with_beta_overrides(parameters.beta_fast, parameters.beta_slow)
                .map_err(|error| Error::configuration("deepseek_v2.rope", error.to_string()))?;
            RopeScaling::Yarn(yarn)
        }
        unsupported => {
            return Err(Error::configuration(
                "deepseek_v2.rope_type",
                format!("{unsupported:?} is not implemented for DeepSeek-V2"),
            ));
        }
    };
    RopeConfig::new(
        config.qk_rope_head_dim,
        config.qk_rope_head_dim,
        theta,
        scaling,
    )
    .map_err(|error| Error::configuration("deepseek_v2.rope", error.to_string()))
}

/// Returns DeepSeek's logarithmic YaRN magnitude correction.
fn yarn_mscale(factor: f64, multiplier: f64) -> f64 {
    if factor <= 1.0 {
        1.0
    } else {
        1.0 + 0.1 * multiplier * factor.ln()
    }
}

/// Computes the checkpoint's scale for the sum of content and rotary logits.
fn deepseek_attention_scale(config: &DeepseekV2Config) -> Result<f64> {
    let mut scale = (config.qk_head_dim() as f64).sqrt().recip();
    if let Some(parameters) = &config.rope_scaling
        && parameters.rope_type == RopeType::Yarn
        && let Some(multiplier) = parameters.mscale_all_dim
        && multiplier != 0.0
    {
        let factor = parameters.factor.ok_or_else(|| {
            Error::configuration("deepseek_v2.rope.factor", "is required for YaRN")
        })?;
        scale *= yarn_mscale(factor, multiplier).powi(2);
    }
    Ok(scale)
}

/// Builds `excluded(q,k) = k > q` and combines semantic padding exclusions.
fn stateless_attention_mask<B: Backend>(
    token_mask: Option<&TokenMask<B>>,
    batch: usize,
    heads: usize,
    sequence: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4, Bool>> {
    let query = Tensor::<B, 1, Int>::arange(0..sequence as i64, device)
        .reshape([1, 1, sequence, 1])
        .expand([batch, heads, sequence, sequence]);
    let key = Tensor::<B, 1, Int>::arange(0..sequence as i64, device)
        .reshape([1, 1, 1, sequence])
        .expand([batch, heads, sequence, sequence]);
    let causal = key.greater(query);
    Ok(
        match padding_attention_mask(token_mask, token_mask, batch, heads, sequence, sequence)? {
            Some(padding) => causal.bool_or(padding),
            None => causal,
        },
    )
}

/// Produces optional router diagnostics using DeepSeek's configured aux semantics.
fn router_diagnostics<B: Backend>(
    output: &SparseMoeOutput<B>,
    mask: Option<&TokenMask<B>>,
    options: &ForwardOptions,
    config: &DeepseekV2Config,
) -> Result<RouterDiagnostics<B>> {
    let auxiliary = if options.auxiliary_losses {
        let raw = if config.seq_aux {
            sequence_auxiliary_loss(
                output.router_logits.clone(),
                output.selected_experts.clone(),
                mask,
            )?
        } else {
            load_balancing_loss(
                output.router_logits.clone(),
                output.selected_experts.clone(),
                mask,
            )?
        };
        Some(raw * config.aux_loss_alpha)
    } else {
        None
    };
    Ok(RouterDiagnostics {
        logits: options.router_logits.then(|| output.router_logits.clone()),
        load_balancing_loss: auxiliary,
        z_loss: None,
    })
}

/// Implements DeepSeek's per-sequence expert-balance auxiliary objective.
fn sequence_auxiliary_loss<B: Backend>(
    router_logits: Tensor<B, 3>,
    selected: Tensor<B, 3, Int>,
    mask: Option<&TokenMask<B>>,
) -> Result<Tensor<B, 1>> {
    let [batch, sequence, experts] = router_logits.dims();
    let [selected_batch, selected_sequence, top_k] = selected.dims();
    if batch == 0
        || sequence == 0
        || experts == 0
        || top_k == 0
        || [selected_batch, selected_sequence] != [batch, sequence]
    {
        return Err(Error::shape(
            "deepseek_v2.router_aux",
            "matching non-empty [batch, sequence, experts/top_k] tensors",
            format!(
                "logits={:?}, selected={:?}",
                router_logits.dims(),
                selected.dims()
            ),
        ));
    }
    let probabilities = burn::tensor::activation::softmax(router_logits.cast(DType::F32), 2);
    let mut assignments = selected.one_hot::<4>(experts).float();
    let valid = match mask {
        Some(mask) => {
            if mask.as_tensor().dims() != [batch, sequence] {
                return Err(Error::shape(
                    "deepseek_v2.router_aux.mask",
                    format!("[{batch}, {sequence}]"),
                    format!("{:?}", mask.as_tensor().dims()),
                ));
            }
            mask.as_tensor().clone().float().cast(DType::F32)
        }
        None => Tensor::ones([batch, sequence], &probabilities.device()),
    };
    assignments = assignments * valid.clone().reshape([batch, sequence, 1, 1]);
    let counts = assignments.sum_dim(1).sum_dim(2).reshape([batch, experts]);
    let valid_count = valid.clone().sum_dim(1).reshape([batch, 1]).clamp_min(1.0);
    let normalized_counts = counts * experts as f64 / (valid_count.clone() * top_k as f64);
    let mean_probability = (probabilities * valid.reshape([batch, sequence, 1]))
        .sum_dim(1)
        .reshape([batch, experts])
        / valid_count;
    Ok((normalized_counts * mean_probability)
        .sum_dim(1)
        .mean()
        .reshape([1]))
}

/// Rejects empty tensor extents before embedding or reduction kernels run.
fn validate_nonempty_input(batch: usize, sequence: usize) -> Result<()> {
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "deepseek_v2.input_ids",
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    Ok(())
}

/// Rejects stateless tensors that do not share the model parameter device.
///
/// Device equality is checked before token reductions, mask scans, positional
/// arithmetic, or embedding lookup so a user error is returned through the
/// crate taxonomy instead of surfacing as a backend mixed-device panic.
fn ensure_model_device<D: PartialEq>(tensor: &'static str, actual: &D, expected: &D) -> Result<()> {
    if actual != expected {
        return Err(Error::BackendCapability(format!(
            "{tensor} and DeepSeek model parameters are on different devices"
        )));
    }
    Ok(())
}

/// Validates token values before the embedding backend receives indices.
fn validate_token_ids<B: Backend>(token_ids: Tensor<B, 2, Int>, vocabulary: usize) -> Result<()> {
    let minimum = token_ids
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate token IDs: {error}"))
        })?
        .elem::<i64>();
    let maximum = token_ids
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate token IDs: {error}"))
        })?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= vocabulary {
        return Err(Error::shape(
            "deepseek_v2.input_ids",
            format!("token values in 0..{vocabulary}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Validates semantic-mask shape and optionally permits fully masked rows.
fn validate_mask<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
    permit_empty_rows: bool,
) -> Result<()> {
    let Some(mask) = mask else {
        return Ok(());
    };
    if mask.as_tensor().dims() != [batch, sequence] {
        return Err(Error::shape(
            "deepseek_v2.token_mask",
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
            .map_err(|error| Error::BackendCapability(format!("failed to validate mask: {error}")))?
            .elem::<i64>();
        if minimum == 0 {
            return Err(Error::shape(
                "deepseek_v2.token_mask",
                "at least one valid token in every prompt row",
                "a fully masked row",
            ));
        }
    }
    Ok(())
}

/// Resolves explicit positions or derives padding-independent ordinals.
fn resolve_positions<B: Backend>(
    input: &CausalLmInput<B>,
    batch: usize,
    sequence: usize,
) -> Result<Tensor<B, 2, Int>> {
    if let Some(positions) = &input.position_ids {
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "deepseek_v2.position_ids",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
        return Ok(positions.clone());
    }
    let device = input.token_ids.device();
    match &input.token_mask {
        Some(mask) => {
            let valid = mask.as_tensor().clone();
            Ok((valid.clone().int().cumsum(1) - 1).mask_fill(valid.bool_not(), 0))
        }
        None => Ok(Tensor::<B, 1, Int>::arange(0..sequence as i64, &device)
            .reshape([1, sequence])
            .expand([batch, sequence])),
    }
}

/// Validates absolute positions against the checkpoint's declared limit.
fn validate_position_range<B: Backend>(
    positions: Tensor<B, 2, Int>,
    capacity: usize,
) -> Result<()> {
    let minimum = positions
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate positions: {error}"))
        })?
        .elem::<i64>();
    let maximum = positions
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate positions: {error}"))
        })?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= capacity {
        return Err(Error::shape(
            "deepseek_v2.position_ids",
            format!("values in 0..{capacity}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Returns one plus the greatest position used by a batch.
fn evaluated_context<B: Backend>(positions: Tensor<B, 2, Int>) -> Result<usize> {
    let maximum = positions
        .max()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to inspect positions: {error}")))?
        .elem::<i64>();
    usize::try_from(maximum)
        .map_err(|_| Error::configuration("deepseek_v2.position_ids", "must be non-negative"))?
        .checked_add(1)
        .ok_or_else(|| Error::configuration("deepseek_v2.position_ids", "context overflows usize"))
}

/// Selects hidden positions before the expensive vocabulary projection.
fn select_hidden<B: Backend>(
    hidden: Tensor<B, 3>,
    mask: Option<&TokenMask<B>>,
    selection: &LogitsSelection,
) -> Result<Tensor<B, 3>> {
    let [batch, sequence, width] = hidden.dims();
    validate_logits_selection(selection, sequence)?;
    match selection {
        LogitsSelection::All => Ok(hidden),
        LogitsSelection::Last => match mask {
            None => Ok(hidden.narrow(1, sequence - 1, 1)),
            Some(mask) => {
                let device = hidden.device();
                let positions = Tensor::<B, 1, Int>::arange(0..sequence as i64, &device)
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
            let indices = positions
                .iter()
                .map(|&value| value as i64)
                .collect::<Vec<_>>();
            let device = hidden.device();
            Ok(hidden.select(
                1,
                Tensor::<B, 1, Int>::from_data(
                    TensorData::new(indices, [positions.len()]),
                    &device,
                ),
            ))
        }
    }
}

/// Validates explicit selective-logit indices.
fn validate_logits_selection(selection: &LogitsSelection, sequence: usize) -> Result<()> {
    if let LogitsSelection::Positions(positions) = selection
        && (positions.is_empty() || positions.iter().any(|&position| position >= sequence))
    {
        return Err(Error::shape(
            "forward_options.logits.positions",
            format!("a non-empty list of indices below {sequence}"),
            format!("{positions:?}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests_support {
    //! Deterministic tiny topology shared by family-local tests.

    use burn::{tensor::TensorData, tensor::backend::Backend};

    use super::*;
    use crate::{checkpoint::TensorSink, models::deepseek_v2::DeepseekV2CheckpointOptions};

    /// Returns a tiny graph with one dense and one sparse block.
    pub(crate) fn tiny_config() -> DeepseekV2Config {
        DeepseekV2Config {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 12,
            moe_intermediate_size: 6,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            n_shared_experts: 2,
            n_routed_experts: 4,
            ep_size: 1,
            routed_scaling_factor: 1.0,
            kv_lora_rank: 5,
            q_lora_rank: None,
            qk_nope_head_dim: 3,
            qk_rope_head_dim: 2,
            v_head_dim: 4,
            topk_method: DeepseekTopKMethod::Greedy,
            n_group: 1,
            topk_group: 1,
            num_experts_per_tok: 2,
            moe_layer_freq: 1,
            first_k_dense_replace: 1,
            norm_topk_prob: false,
            scoring_func: DeepseekScoringFunction::Softmax,
            aux_loss_alpha: 0.001,
            seq_aux: true,
            hidden_act: "silu".into(),
            max_position_embeddings: 32,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pretraining_tp: 1,
            tie_word_embeddings: false,
            rope_theta: 10_000.0,
            rope_scaling: None,
            attention_bias: false,
            attention_dropout: 0.0,
            use_mla: true,
            pad_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
            model_type: "deepseek_v2".into(),
        }
    }

    /// Installs deterministic parameters through the strict family sink.
    pub(crate) fn initialized_model<B: Backend>(
        config: DeepseekV2Config,
        device: &B::Device,
    ) -> DeepseekV2ForCausalLm<B> {
        let contract = config
            .checkpoint_contract(DeepseekV2CheckpointOptions::default())
            .expect("tiny contract");
        let mut model = DeepseekV2ForCausalLm::init(config, device).expect("tiny model");
        for (tensor_index, expected) in contract.expected_tensors().iter().enumerate() {
            let shape = expected
                .shape
                .iter()
                .map(|&dimension| usize::try_from(dimension).expect("tiny dimension"))
                .collect::<Vec<_>>();
            let elements = shape.iter().product();
            let values = (0..elements)
                .map(|element| {
                    let code = (element + tensor_index * 7) % 23;
                    (code as f32 - 11.0) * 0.002
                })
                .collect::<Vec<_>>();
            TensorSink::write_tensor(&mut model, &expected.name, TensorData::new(values, shape))
                .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
        }
        model
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        module::Param,
        tensor::{Bool, Int, Tensor, Tolerance, backend::BackendTypes},
    };

    use super::*;
    use crate::architecture::TokenMask;

    type TestBackend = Flex<f32, i32>;

    /// Builds deterministic tiny weights for forward tests.
    fn tiny_model() -> (
        DeepseekV2ForCausalLm<TestBackend>,
        <TestBackend as BackendTypes>::Device,
    ) {
        let device = Default::default();
        let model = tests_support::initialized_model(tests_support::tiny_config(), &device);
        (model, device)
    }

    /// Stateless execution exposes hidden, attention, and router diagnostics.
    #[test]
    fn stateless_shapes_and_diagnostics_are_exact() {
        let (model, device) = tiny_model();
        let output = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                        [[1, 2, 3], [4, 5, 6]],
                        &device,
                    ),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    hidden_states: true,
                    attentions: true,
                    router_logits: true,
                    auxiliary_losses: true,
                },
            )
            .unwrap();
        assert_eq!(output.logits.dims(), [2, 3, 32]);
        assert_eq!(output.hidden_states.len(), 4);
        assert_eq!(output.attentions.len(), 2);
        assert_eq!(output.attentions[0].dims(), [2, 2, 3, 3]);
        assert_eq!(output.routers.len(), 1);
        assert_eq!(output.routers[0].logits.as_ref().unwrap().dims(), [2, 3, 4]);
        assert!(output.routers[0].load_balancing_loss.is_some());
    }

    /// Matches an independent PyTorch transcription of the pinned official equations.
    ///
    /// The oracle uses reference revision
    /// `604d5664dddd88a0433dbae533b7fe9472482de0`, the deterministic checkpoint
    /// installed by `tests_support`, prompt `[1, 2, 3]`, and FP32 execution.
    #[test]
    fn stateless_logits_match_pinned_reference_oracle() {
        let (model, device) = tiny_model();
        let actual = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
            .into_data();
        let expected = TensorData::from([[[
            -0.000_076_392_36_f32,
            -0.000_215_352_72,
            -0.000_141_240_54,
            0.000_092_633_3,
            -0.000_206_088_66,
            -0.000_131_976_5,
            0.000_261_666_37,
            -0.000_196_824_65,
            -0.000_122_712_49,
            0.000_404_032_35,
            -0.000_187_560_65,
            -0.000_113_448_46,
            0.000_493_179_6,
            -0.000_178_296_61,
            -0.000_104_184_44,
            0.000_502_443_6,
            -0.000_169_032_59,
            -0.000_094_920_42,
            0.000_405_196_83,
            -0.000_159_768_56,
            -0.000_085_656_39,
            0.000_174_808_55,
            -0.000_150_504_55,
            -0.000_076_392_36,
            -0.000_215_352_72,
            -0.000_141_240_54,
            0.000_092_633_3,
            -0.000_206_088_66,
            -0.000_131_976_5,
            0.000_261_666_37,
            -0.000_196_824_65,
            -0.000_122_712_49,
        ]]]);
        actual.assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-6));
    }

    /// Layer zero is dense and layer one owns selected plus widened shared experts.
    #[test]
    fn dense_prefix_and_sparse_shared_topology_are_distinct() {
        let (model, _) = tiny_model();
        assert!(model.model.layers[0].dense_mlp.is_some());
        assert!(model.model.layers[0].sparse_mlp.is_none());
        let sparse = model.model.layers[1].sparse_mlp.as_ref().unwrap();
        assert_eq!(sparse.routed.routed.len(), 4);
        assert_eq!(sparse.shared_experts.gate_proj.weight.val().dims(), [8, 12]);
    }

    /// DeepSeek routing evaluates only experts selected by the FP32 top-k projection.
    #[test]
    fn router_projection_is_fp32_and_dispatch_is_selected_only() {
        let (mut model, device) = tiny_model();
        let sparse = model.model.layers[1].sparse_mlp.as_mut().unwrap();
        let mut weights = Vec::with_capacity(8 * 4);
        for _ in 0..8 {
            weights.extend([1.0_f32, 0.5, -0.5, -1.0]);
        }
        sparse.routed.router.weight =
            Param::from_tensor(Tensor::from_data(TensorData::new(weights, [8, 4]), &device));
        let output = sparse.forward(Tensor::ones([1, 3, 8], &device)).unwrap();
        assert_eq!(output.routed.router_logits.dtype(), DType::F32);
        assert_eq!(output.routed.expert_token_counts, vec![3, 3, 0, 0]);
        assert_eq!(output.routed.evaluated_experts, 2);
    }

    /// Reduced-precision experts are weighted and accumulated before one final cast.
    #[test]
    fn routed_mixture_accumulates_in_fp32_before_residual_cast() {
        let device = Default::default();
        let config = tests_support::tiny_config();
        let mut moe = DeepseekV2Moe::<TestBackend>::init(&config, &device).unwrap();
        moe.routed.router.weight = Param::from_tensor(Tensor::zeros([8, 4], &device));

        let single_path = |shape: [usize; 2], value: f32| {
            let mut values = vec![0.0_f32; shape.iter().product()];
            values[0] = value;
            Tensor::<TestBackend, 2>::from_data(TensorData::new(values, shape), &device)
                .cast(DType::F16)
        };
        for (expert, sign) in moe.routed.routed[..2].iter_mut().zip([1.0, -1.0]) {
            expert.gate_proj.weight = Param::from_tensor(single_path([8, 6], 1.0));
            expert.up_proj.weight = Param::from_tensor(single_path([8, 6], 1.0));
            expert.down_proj.weight = Param::from_tensor(single_path([6, 8], sign));
        }
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1, 8]),
            &device,
        )
        .cast(DType::F16);
        let expert_zero = moe.routed.routed[0]
            .forward(input.clone())
            .unwrap()
            .cast(DType::F32);
        let expert_one = moe.routed.routed[1]
            .forward(input.clone())
            .unwrap()
            .cast(DType::F32);
        let dispatch = moe
            .routed
            .dispatch_selected_experts(
                input.clone(),
                Tensor::from_floats([[0.5001, 0.4999]], &device),
                Tensor::<TestBackend, 2, Int>::from_ints([[0, 1]], &device),
                DType::F32,
            )
            .unwrap();
        assert_eq!(dispatch.hidden_state.dtype(), DType::F32);

        let actual = dispatch.hidden_state.cast(DType::F16).cast(DType::F32);
        let expected = (expert_zero.clone() * 0.5001 + expert_one.clone() * 0.4999)
            .cast(DType::F16)
            .cast(DType::F32);
        let prematurely_rounded = ((expert_zero * 0.5001).cast(DType::F16)
            + (expert_one * 0.4999).cast(DType::F16))
        .cast(DType::F32);
        actual
            .clone()
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(0.0));
        let actual_first = actual.into_data().to_vec::<f32>().unwrap()[0];
        let premature_first = prematurely_rounded.into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            (actual_first - premature_first).abs() > 1e-5,
            "the fixture must distinguish final-only rounding ({actual_first}) from per-expert rounding ({premature_first})"
        );

        for projection in [
            &mut moe.shared_experts.gate_proj,
            &mut moe.shared_experts.up_proj,
            &mut moe.shared_experts.down_proj,
        ] {
            projection.weight = Param::from_tensor(projection.weight.val().cast(DType::F16));
        }
        let output = moe.forward(input.reshape([1, 1, 8])).unwrap();
        assert_eq!(output.routed.selected_weights.dtype(), DType::F32);
        assert_eq!(output.routed.hidden_state.dtype(), DType::F16);
        assert_eq!(output.hidden_state.dtype(), DType::F16);
    }

    /// Top-one routing preserves confidence and the configured routed scale.
    #[test]
    fn top_one_routing_does_not_renormalize_away_confidence() {
        let device = Default::default();
        let mut config = tests_support::tiny_config();
        config.num_experts_per_tok = 1;
        config.norm_topk_prob = true;
        config.routed_scaling_factor = 2.5;
        let mut moe = DeepseekV2Moe::<TestBackend>::init(&config, &device).unwrap();
        moe.routed.router.weight = Param::from_tensor(Tensor::zeros([8, 4], &device));

        let output = moe.forward(Tensor::ones([1, 1, 8], &device)).unwrap();
        let weights = output
            .routed
            .selected_weights
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(weights, [0.625]);
    }

    /// Left padding is ignored and fully masked prompt rows fail before attention.
    #[test]
    fn semantic_padding_and_invalid_rows_are_validated() {
        let (model, device) = tiny_model();
        let input = CausalLmInput {
            token_ids: Tensor::from_ints([[0, 1, 2]], &device),
            token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                [[false, true, true]].into(),
                &device,
            ))),
            position_ids: None,
        };
        let output = model
            .forward(
                input,
                &ForwardOptions {
                    logits: LogitsSelection::Last,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(output.logits.dims(), [1, 1, 32]);

        let error = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[0, 0]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[false, false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("fully masked"));
    }

    /// Stateless device preflight names every independently supplied tensor.
    #[test]
    fn stateless_device_affinity_errors_are_contextual() {
        assert!(ensure_model_device("deepseek_v2.input_ids", &0_u8, &0_u8).is_ok());
        for tensor in [
            "deepseek_v2.input_ids",
            "deepseek_v2.token_mask",
            "deepseek_v2.position_ids",
        ] {
            let error = ensure_model_device(tensor, &1_u8, &0_u8)
                .expect_err("different device identities must fail preflight");
            assert!(error.to_string().contains(tensor));
            assert!(error.to_string().contains("model parameters"));
        }
    }

    /// DeepSeek's YaRN scale uses both table amplitude and all-dimension score scale.
    #[test]
    fn pinned_yarn_scaling_matches_closed_form() {
        let config = DeepseekV2Config::default();
        let rope = deepseek_rope_config(&config).unwrap();
        let RopeScaling::Yarn(yarn) = rope.scaling() else {
            panic!("pinned config must use YaRN")
        };
        assert!((yarn.attention_factor() - 1.0).abs() < 1e-12);
        let expected = (192.0_f64).sqrt().recip() * yarn_mscale(40.0, 0.707).powi(2);
        assert!((deepseek_attention_scale(&config).unwrap() - expected).abs() < 1e-12);
    }

    /// One-sided YaRN beta overrides retain the official default for the other bound.
    #[test]
    fn yarn_beta_overrides_are_resolved_independently() {
        let mut config = tests_support::tiny_config();
        config.rope_scaling = Some(crate::models::config::RopeParameters {
            rope_type: RopeType::Yarn,
            factor: Some(2.0),
            original_max_position_embeddings: Some(16),
            beta_fast: Some(24.0),
            ..Default::default()
        });
        let RopeScaling::Yarn(fast_only) = deepseek_rope_config(&config).unwrap().scaling() else {
            panic!("expected YaRN")
        };
        assert_eq!(fast_only.beta_fast(), 24.0);
        assert_eq!(fast_only.beta_slow(), 1.0);

        config.rope_scaling.as_mut().unwrap().beta_fast = None;
        config.rope_scaling.as_mut().unwrap().beta_slow = Some(2.0);
        let RopeScaling::Yarn(slow_only) = deepseek_rope_config(&config).unwrap().scaling() else {
            panic!("expected YaRN")
        };
        assert_eq!(slow_only.beta_fast(), 32.0);
        assert_eq!(slow_only.beta_slow(), 2.0);
    }

    /// DeepSeek returns vocabulary logits in FP32 even for reduced-precision modules.
    #[test]
    fn vocabulary_projection_always_returns_fp32() {
        let (mut model, device) = tiny_model();
        let head = model.lm_head.as_mut().unwrap();
        head.weight = Param::from_tensor(head.weight.val().cast(DType::F16));
        let hidden = Tensor::<TestBackend, 3>::zeros([1, 2, 8], (&device, DType::F16));

        let logits = model.project_logits(hidden).unwrap();
        assert_eq!(logits.dtype(), DType::F32);
    }

    /// Invalid token IDs and selective logits return errors rather than panics.
    #[test]
    fn invalid_runtime_values_are_rejected() {
        let (model, device) = tiny_model();
        assert!(
            model
                .forward(
                    CausalLmInput {
                        token_ids: Tensor::from_ints([[32]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    &ForwardOptions::default(),
                )
                .is_err()
        );
        assert!(
            model
                .forward(
                    CausalLmInput {
                        token_ids: Tensor::from_ints([[1, 2]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    &ForwardOptions {
                        logits: LogitsSelection::Positions(vec![2]),
                        ..Default::default()
                    },
                )
                .is_err()
        );
    }

    /// Unsupported distributed/router/dropout variants fail before large allocation.
    #[test]
    fn unsupported_runtime_variants_fail_during_initialization() {
        let device = Default::default();
        let mut config = tests_support::tiny_config();
        config.ep_size = 2;
        assert!(DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).is_err());

        let mut config = tests_support::tiny_config();
        config.topk_method = DeepseekTopKMethod::GroupLimitedGreedy;
        assert!(DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).is_err());

        let mut config = tests_support::tiny_config();
        config.attention_dropout = 0.1;
        assert!(DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).is_err());

        let mut config = tests_support::tiny_config();
        config.use_mla = false;
        assert!(DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).is_err());
    }

    /// Full-precision stateless execution retains an embedding gradient.
    #[cfg(feature = "autodiff")]
    #[test]
    fn autodiff_stateless_smoke_reaches_embedding() {
        use burn::backend::Autodiff;

        type AdBackend = Autodiff<TestBackend>;
        let device = Default::default();
        let model = DeepseekV2ForCausalLm::<AdBackend>::init(tests_support::tiny_config(), &device)
            .unwrap();
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
