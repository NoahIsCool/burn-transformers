//! Executable DeepSeek-V2 decoder with MLA and sparse shared-expert blocks.
//!
//! A decoder block is pre-normalized and evaluates
//! `x' = x + MLA(RMSNorm(x))`, then
//! `y = x' + FFN(RMSNorm(x'))`. The pinned Lite checkpoint uses one dense
//! SwiGLU block followed by 26 blocks that add a top-6 routed mixture to one
//! always-active SwiGLU whose intermediate width is exactly twice an expert's
//! width (see [`DeepseekV2Moe`] for the routing policies and balance losses).

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{Bool, DType, Int, Tensor, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::{RouterDiagnostics, TokenMask},
    attention::padding_attention_mask,
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    moe::SparseMoeOutput,
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling, YarnConfig},
};

use super::{
    DeepseekV2Config,
    mla::{MlaMixOutput, MultiLatentAttention, MultiLatentAttentionConfig},
};
use crate::models::{
    config::RopeType,
    shared::{InputContract, InputNames, select_hidden, validate_input},
};

/// External compressed MLA caches and transactional prefill/decode execution.
mod decode;
/// Routed-plus-shared expert block, router policies, and balance losses.
mod moe;

pub use decode::{DeepseekV2DecodeState, MlaExecutionPath};
pub use moe::DeepseekV2Moe;

use moe::{ROUTER_DTYPE, router_diagnostics};

/// Tensor names reported by DeepSeek-V2 input errors.
const INPUT_NAMES: InputNames<'static> = InputNames {
    token_ids: "deepseek_v2.input_ids",
    token_mask: "deepseek_v2.token_mask",
    position_ids: "deepseek_v2.position_ids",
};

/// Owner named when a stateless tensor is on another device than the weights.
const PARAMETERS: &str = "DeepSeek model parameters";

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
        let projection = self.self_attn.project_rotary(
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
        let validated = validate_input(
            input,
            &self.embed_tokens.weight.lazy_device(),
            &InputContract::new(INPUT_NAMES, PARAMETERS, self.config.vocab_size)
                .with_position_limit(self.config.max_position_embeddings),
        )?;
        let positions = validated.require_positions(INPUT_NAMES.position_ids)?;
        let context = validated.evaluated_context;
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
                    routed.router_logits,
                    routed.selected_experts,
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

#[cfg(test)]
#[path = "model_tests_support.rs"]
/// Deterministic tiny topology and checkpoint weights shared by family tests.
pub(crate) mod tests_support;

#[cfg(test)]
/// Hugging Face Transformers 5.2 goldens for the tiny reference topologies.
mod goldens;

#[cfg(test)]
/// Tiny-model logit and routing parity with Hugging Face DeepSeek-V2 and V3.
mod reference_tests;

#[cfg(test)]
#[path = "model_tests.rs"]
/// Forward diagnostics, reference parity, routing, and input-validation regressions.
mod tests;
