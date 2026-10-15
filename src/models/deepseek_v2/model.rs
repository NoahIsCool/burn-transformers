//! Executable DeepSeek-V2 decoder with MLA and sparse shared-expert blocks.
//!
//! A decoder block is pre-normalized and evaluates
//! `x' = x + MLA(RMSNorm(x))`, then
//! `y = x' + FFN(RMSNorm(x'))`. The pinned Lite checkpoint uses one dense
//! SwiGLU block followed by 26 blocks that add a top-6 routed mixture to one
//! always-active SwiGLU whose intermediate width is exactly twice an expert's
//! width (see [`DeepseekV2Moe`] for the routing policies and balance losses).
//!
//! # Positions and attention scale
//!
//! Rotary metadata goes through the shared translator
//! ([`RopeParameters::to_rope_config`](crate::models::config::RopeParameters::to_rope_config))
//! with DeepSeek's reference conventions: the YaRN amplitude is
//! `m(f, mscale) / m(f, mscale_all_dim)` whenever either key is present, and
//! every scaled `rope_type` multiplies the softmax scale
//! `1 / sqrt(qk_nope + qk_rope)` by `m(f, mscale_all_dim)^2`, where
//! `m(s, k) = 1 + 0.1 k ln s`. Only the `qk_rope_head_dim` sub-head rotates
//! and it always rotates completely, so a `partial_rotary_factor` other than
//! one is rejected rather than silently ignored. The cosine/sine tables are
//! built once per forward pass (and cast once to the activation dtype) and
//! shared by every layer.
//!
//! # Bounded prefill
//!
//! Reference (unabsorbed) MLA expands keys and values and forms one
//! `[batch, head, query, key]` score plane per layer. The model's
//! [`AttentionExecutionPolicy`] evaluates that plane in synchronized query
//! chunks, each restricted to its causally visible key prefix, so a long
//! prompt needs `O(chunk * key)` score workspace instead of `O(query * key)`.
//! Requesting attention probabilities (`ForwardOptions::attentions`)
//! materializes the full plane, which that output consists of.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{Bool, DType, Int, Tensor, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::RouterDiagnostics,
    attention::AttentionExecutionPolicy,
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    intervention::{Hooks, Layer, PointInfo, Stack},
    moe::SparseMoeOutput,
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{RopeConfig, RotaryAngles, RotaryFrequencies, rotary_angles_from_frequencies},
};

use super::{
    DeepseekV2Config,
    mla::{
        MlaMixOutput, MlaProjection, MultiLatentAttention, MultiLatentAttentionConfig,
        PositionalVisibility,
    },
};
use crate::models::{
    config::{PartialRotary, RopeContext, SoftmaxMscale, YarnMscaleConvention},
    shared::{InputContract, InputNames, select_hidden, validate_input},
};

/// Dependency-free host planning of batched stateful blocks (Verus-verified).
pub(crate) mod batch;
/// Device-side packing of ragged stateful rows, shared with Muse-Glimmer.
pub(crate) mod block;
/// External compressed MLA caches and batched transactional prefill/decode execution.
mod decode;
/// Activation hooks and staged execution (`HookedModel`, `StagedModel`).
mod hooks;
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

/// Builds the rotary-translation context with DeepSeek's reference conventions.
///
/// The rotary sub-head is `qk_rope_head_dim` wide and always fully rotated;
/// the YaRN amplitude and the softmax correction follow DeepSeek's
/// `modeling_deepseek.py` (module documentation).
fn rope_context(config: &DeepseekV2Config) -> RopeContext {
    RopeContext::new(
        "deepseek_v2",
        config.qk_rope_head_dim,
        config.max_position_embeddings,
    )
    .with_rope_theta(Some(config.rope_theta))
    .with_partial_rotary(PartialRotary::FullHeadOnly)
    .with_yarn_mscale_convention(YarnMscaleConvention::DeepSeekReference)
    .with_softmax_mscale(SoftmaxMscale::MscaleAllDimSquared)
}

/// Translates the configuration's rotary metadata and attention score scale.
///
/// Returns the executable [`RopeConfig`] and the softmax scale
/// `m(f, mscale_all_dim)^2 / sqrt(qk_nope_head_dim + qk_rope_head_dim)`.
///
/// # Errors
///
/// Returns the translator's contextual configuration errors at
/// `deepseek_v2.rope.<field>`.
pub(crate) fn rope_and_softmax_scale(config: &DeepseekV2Config) -> Result<(RopeConfig, f64)> {
    let context = rope_context(config);
    let parameters = config.rope_scaling.clone().unwrap_or_default();
    let rope = parameters.to_rope_config(&context)?;
    let scale = (config.qk_head_dim() as f64).sqrt().recip()
        * parameters.softmax_scale_multiplier(&context)?;
    Ok((rope, scale))
}

/// Layer-invariant attention inputs computed once per stateless forward pass.
struct AttentionInputs<'a, B: Backend> {
    /// Cosine/sine tables `[batch, 1, sequence, rotary]`, in the activation dtype.
    angles: &'a RotaryAngles<B>,
    /// Causal and padding visibility shared by every layer.
    visibility: &'a PositionalVisibility<B>,
    /// Peak score-plane policy for the reference path.
    policy: AttentionExecutionPolicy,
    /// Whether to return full attention probabilities (forces one full plane).
    retain_attention: bool,
}

/// An experiment-supplied attention policy for one staged layer.
///
/// Custom connectivity replaces the default causal visibility, so it always
/// evaluates one full `[batch, head, query, key]` plane through the hookable
/// reference graph.
struct CustomConnectivity<B: Backend> {
    /// Whether bottom-right causality applies in addition to `exclusion`.
    causal: bool,
    /// Padding and disallowed links, `[batch, head, query, key]`, `true` = excluded.
    exclusion: Option<Tensor<B, 4, Bool>>,
    /// Additive score bias broadcastable to `[batch, head, query, key]`.
    bias: Option<Tensor<B, 4>>,
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
    ///
    /// `softmax_scale` is the translated attention score scale shared by
    /// every layer (see [`rope_and_softmax_scale`]).
    fn init(
        config: &DeepseekV2Config,
        layer_index: usize,
        softmax_scale: f64,
        device: &B::Device,
    ) -> Result<Self> {
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
        attention.softmax_scale = softmax_scale;
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

    /// Evaluates one stateless layer, optionally emitting the points of `layer`.
    ///
    /// Without custom connectivity or attention-internal hooks, attention runs
    /// in bounded query chunks; probabilities, custom connectivity, and any
    /// requested MLA point (latents, Q/K/V, scores, pattern, head context)
    /// evaluate one full `[batch, head, query, key]` plane through the
    /// reference graph instead.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        inputs: &AttentionInputs<'_, B>,
        custom: Option<&CustomConnectivity<B>>,
        layer: Layer,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<LayerOutput<B>> {
        let mut hidden = hidden;
        if let Some(hooks) = hooks.as_deref_mut() {
            hidden = hooks.apply(layer.residual_pre(), hidden)?;
        }
        let residual = hidden.clone();
        let mut normalized = self.input_layernorm.forward(hidden);
        if let Some(hooks) = hooks.as_deref_mut() {
            normalized = hooks.apply(layer.attention_input(), normalized)?;
        }
        let points = layer.self_attention();
        let internal = hooks.as_deref().is_some_and(|hooks| {
            self.self_attn
                .activation_points(points)
                .into_iter()
                .any(|point| hooks.needs_point(point))
        });
        let (attended, probabilities) = match (hooks.as_deref_mut(), custom) {
            (Some(hooks), custom) if internal || custom.is_some() => {
                if inputs.retain_attention {
                    return Err(Error::BackendCapability(
                        "DeepSeek attention probabilities cannot be returned together with \
                         attention-internal hooks; capture the pattern point instead"
                            .into(),
                    ));
                }
                let projection = self.self_attn.project_with_angles_and_hooks(
                    normalized,
                    inputs.angles,
                    points,
                    hooks,
                )?;
                let (exclusion, bias, causal) = match custom {
                    Some(custom) => (custom.exclusion.clone(), custom.bias.clone(), custom.causal),
                    None => (
                        Some(full_exclusion(&projection, inputs.visibility)),
                        None,
                        false,
                    ),
                };
                let attended = self
                    .self_attn
                    .attend_with_hooks(projection, exclusion, bias, causal, points, hooks)?;
                (attended, None)
            }
            (None, Some(_)) => {
                return Err(Error::configuration(
                    "deepseek_v2.execution.attention",
                    "custom connectivity is evaluated inside a hook scope",
                ));
            }
            _ => {
                let projection = self
                    .self_attn
                    .project_with_angles(normalized, inputs.angles)?;
                if inputs.retain_attention {
                    let exclusion = full_exclusion(&projection, inputs.visibility);
                    let MlaMixOutput {
                        hidden_state,
                        probabilities,
                    } = self.self_attn.mix_unabsorbed(
                        &projection,
                        projection.normalized_latent.clone(),
                        projection.rotated_key.clone(),
                        Some(exclusion),
                        true,
                    )?;
                    (hidden_state, probabilities)
                } else {
                    let attended = self.self_attn.mix_unabsorbed_bounded(
                        &projection,
                        projection.normalized_latent.clone(),
                        projection.rotated_key.clone(),
                        inputs.visibility,
                        inputs.policy,
                    )?;
                    (attended, None)
                }
            }
        };
        self.finish_block(residual, attended, probabilities, layer, hooks)
    }

    /// Adds the attention residual, then applies exactly one feed-forward topology.
    ///
    /// `residual` is the block input and `attended` the attention output
    /// before its residual addition; `probabilities` are returned unchanged.
    fn finish_block(
        &self,
        residual: Tensor<B, 3>,
        attended: Tensor<B, 3>,
        probabilities: Option<Tensor<B, 4>>,
        layer: Layer,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<LayerOutput<B>> {
        let mut attended = attended;
        if let Some(hooks) = hooks.as_deref_mut() {
            attended = hooks.apply(layer.attention_output(), attended)?;
        }
        let hidden = residual + attended;
        let residual = hidden.clone();
        let mut normalized = self.post_attention_layernorm.forward(hidden);
        if let Some(hooks) = hooks.as_deref_mut() {
            normalized = hooks.apply(layer.mlp_input(), normalized)?;
        }
        let (mut fed, routed) = match (&self.sparse_mlp, &self.dense_mlp) {
            (Some(moe), None) => {
                let output = match hooks.as_deref_mut() {
                    Some(hooks) => moe.forward_with_hooks(normalized, layer, hooks)?,
                    None => moe.forward(normalized)?,
                };
                (output.hidden_state, Some(output.routed))
            }
            (None, Some(mlp)) => (
                match hooks.as_deref_mut() {
                    Some(hooks) => mlp.forward_with_hooks(normalized, layer, hooks)?,
                    None => mlp.forward(normalized)?,
                },
                None,
            ),
            _ => {
                return Err(Error::Checkpoint(
                    "DeepSeek-V2 layer has an invalid dense/MoE topology".into(),
                ));
            }
        };
        if let Some(hooks) = hooks.as_deref_mut() {
            fed = hooks.apply(layer.mlp_output(), fed)?;
        }
        let mut hidden = residual + fed;
        if let Some(hooks) = hooks {
            hidden = hooks.apply(layer.residual_post(), hidden)?;
        }
        Ok(LayerOutput {
            hidden,
            attention: probabilities,
            routed,
        })
    }

    /// Lists this layer's points in emission order (see [`DeepseekV2ForCausalLm::activation_points`]).
    fn activation_points(&self, layer: Layer) -> Vec<PointInfo> {
        let mut points = vec![layer.residual_pre().info(), layer.attention_input().info()];
        points.extend(self.self_attn.activation_points(layer.self_attention()));
        points.extend([layer.attention_output().info(), layer.mlp_input().info()]);
        if self.sparse_mlp.is_some() {
            points.extend(moe::activation_points(layer));
        } else {
            points.extend([
                layer.mlp_pre_activation().info(),
                layer.mlp_activation().info(),
            ]);
        }
        points.extend([layer.mlp_output().info(), layer.residual_post().info()]);
        points
    }
}

/// Materializes the complete `[batch, head, query, key]` exclusion of one projection.
fn full_exclusion<B: Backend>(
    projection: &MlaProjection<B>,
    visibility: &PositionalVisibility<B>,
) -> Tensor<B, 4, Bool> {
    let [batch, heads, queries, _] = projection.query_content.dims();
    let keys = projection.normalized_latent.dims()[1];
    visibility.exclusion(batch, heads, 0..queries, 0..keys)
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
    /// Peak score-plane policy of the reference attention path.
    #[module(skip)]
    execution_policy: AttentionExecutionPolicy,
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
        let (rope, softmax_scale) = rope_and_softmax_scale(&config)?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| DeepseekV2DecoderLayer::init(&config, index, softmax_scale, device))
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
            execution_policy: AttentionExecutionPolicy::default(),
        })
    }

    /// Builds the per-forward cosine/sine tables in the activation dtype.
    ///
    /// Frequencies depend on `evaluated_context` only for context-dependent
    /// scalings (dynamic NTK, LongRoPE); every other policy yields identical
    /// tables for any non-zero context.
    fn rotary_tables(
        &self,
        positions: Tensor<B, 2, Int>,
        evaluated_context: usize,
    ) -> Result<RotaryAngles<B>> {
        let frequencies = RotaryFrequencies::new(self.rope, evaluated_context)?;
        Ok(rotary_angles_from_frequencies(&frequencies, positions)?
            .to_dtype(self.embed_tokens.weight.val().dtype()))
    }

    /// Validates a public input once and resolves its positions and rotary extent.
    ///
    /// One device-to-host transfer covers token, mask, and position values.
    fn validate_public_input(
        &self,
        input: &CausalLmInput<B>,
    ) -> Result<(Tensor<B, 2, Int>, usize)> {
        let validated = validate_input(
            input,
            &self.embed_tokens.weight.lazy_device(),
            &InputContract::new(INPUT_NAMES, PARAMETERS, self.config.vocab_size)
                .with_position_limit(self.config.max_position_embeddings),
        )?;
        Ok((
            validated.require_positions(INPUT_NAMES.position_ids)?,
            validated.evaluated_context,
        ))
    }

    /// Evaluates the stateless backbone while retaining only requested diagnostics.
    ///
    /// With `hooks`, the embedding, every layer point, and the final norm are
    /// emitted; the caller owns the surrounding [`Hooks::run`] scope.
    fn forward(
        &self,
        input: &CausalLmInput<B>,
        options: &ForwardOptions,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<BackboneOutput<B>> {
        let (positions, context) = self.validate_public_input(input)?;
        let angles = self.rotary_tables(positions, context)?;
        let visibility = PositionalVisibility::causal_columns(
            input.token_ids.dims()[1],
            input
                .token_mask
                .as_ref()
                .map(|mask| mask.as_tensor().clone()),
            &input.token_ids.device(),
        )?;
        let inputs = AttentionInputs {
            angles: &angles,
            visibility: &visibility,
            policy: self.execution_policy,
            retain_attention: options.attentions,
        };
        let mut hidden = self.embed_tokens.forward(input.token_ids.clone());
        if let Some(hooks) = hooks.as_deref_mut() {
            hidden = hooks.apply(Stack::Decoder.embedding(), hidden)?;
        }
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
        let retain_routers = options.router_logits || options.auxiliary_losses;
        let mut routers = Vec::with_capacity(if retain_routers {
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
        for (index, layer) in self.layers.iter().enumerate() {
            let output = layer.forward(
                hidden,
                &inputs,
                None,
                Layer::decoder(index),
                hooks.as_deref_mut(),
            )?;
            hidden = output.hidden;
            if options.hidden_states {
                hidden_states.push(hidden.clone());
            }
            attentions.extend(output.attention);
            if let Some(routed) = output.routed
                && retain_routers
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
        if let Some(hooks) = hooks {
            hidden = hooks.apply(Stack::Decoder.final_norm(), hidden)?;
        }
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

    /// Applies one validated reference-attention workspace policy.
    ///
    /// The default is synchronized query chunking, which bounds the
    /// reference path's score planes for long prompts. Selecting
    /// [`AttentionExecutionPolicy::Unbounded`] evaluates each layer in one
    /// call. Autodiff backends always evaluate one call, because their tape
    /// retains every chunk anyway.
    ///
    /// # Errors
    ///
    /// Returns the policy's validation error.
    pub fn with_attention_execution_policy(
        mut self,
        policy: AttentionExecutionPolicy,
    ) -> Result<Self> {
        policy.validate()?;
        self.model.execution_policy = policy;
        Ok(self)
    }

    /// Returns the reference-attention workspace policy.
    #[must_use]
    pub const fn attention_execution_policy(&self) -> AttentionExecutionPolicy {
        self.model.execution_policy
    }

    /// Evaluates differentiable stateless inference or training math.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        let backbone = self.model.forward(&input, options, None)?;
        self.finish_forward(&input, options, backbone)
    }

    /// Selects the requested positions and projects them to F32 vocabulary logits.
    fn finish_forward(
        &self,
        input: &CausalLmInput<B>,
        options: &ForwardOptions,
        backbone: BackboneOutput<B>,
    ) -> Result<CausalLmOutput<B>> {
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
