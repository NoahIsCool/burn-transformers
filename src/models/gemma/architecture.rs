//! Translation of a Gemma configuration into an executable topology.
//!
//! Gemma 2 and the Gemma 3 text tower are the same sandwich-normalized
//! decoder,
//!
//! ```text
//! h_0     = Embed(tokens) · s,        s = round_dtype(sqrt(d))
//! x'      = x + N_post_attn(Attn(N_in(x)))
//! h_{l+1} = x' + N_post_ffw(GeGLU(N_pre_ffw(x')))
//! logits  = softcap_final(N_final(h_L) · Eᵀ)       (tied; an untied head otherwise)
//! ```
//!
//! with zero-centered RMSNorm `N(x) = (1 + w) ⊙ x · rsqrt(mean(x²) + ε)`,
//! GeGLU `down(gelu_tanh(gate u) ⊙ up u)`, the soft-cap
//! `softcap_c(z) = c · tanh(z / c)`, and grouped-query attention with
//! scores `softcap_attn(q·kᵀ · query_pre_attn_scalar^(-1/2))`.
//! [`GemmaArchitecture`] records the flags in which the two references
//! differ; the model reads nothing else.
//!
//! # Translation table
//!
//! | executable choice | gemma2 | gemma3_text |
//! | --- | --- | --- |
//! | q/k/v/o bias | `attention_bias` on all four | `attention_bias` on all four |
//! | Q/K RMSNorm | none | per head (`head_dim`), zero-centered |
//! | attention scale | `query_pre_attn_scalar^(-1/2)` | `query_pre_attn_scalar^(-1/2)` |
//! | attention soft-cap | `attn_logit_softcapping` | none; a non-null value is rejected |
//! | final soft-cap | `final_logit_softcapping` | `final_logit_softcapping` |
//! | layer schedule | `layer_types`, else alternating from sliding | `layer_types`, else every `sliding_window_pattern`-th (6) layer global |
//! | sliding layers | window `sliding_window` | window `sliding_window` |
//! | RoPE | one set: `rope_parameters` or `rope_scaling`, theta `rope_theta` or 1e4 | sliding: `rope_parameters.sliding_attention`, theta `rope_local_base_freq` or 1e4; global: `rope_parameters.full_attention` or `rope_scaling`, theta `rope_theta` or 1e6 |
//! | MLP | GeGLU, `hidden_activation` | GeGLU, `hidden_activation` |
//!
//! * **Norms.** Every RMSNorm (the four sandwich norms, the Q/K norms, and
//!   the final norm) stores `w` with scale `1 + w` and computes
//!   `cast((1 + w_f32) · x_f32 · rsqrt(mean(x_f32²) + ε))`: the reduction,
//!   the reciprocal root, and the scale all run in F32 and round once
//!   ([`RmsNormRootMode::MultiplyByReciprocalSqrt`], scale then cast).
//! * **Embedding scale.** Both references multiply the looked-up embedding
//!   by `sqrt(hidden_size)` *rounded to the activation dtype* (Gemma 3
//!   rounds through F32 first; both conversions agree for every hidden size
//!   up to 20000, checked with PyTorch 2.10), so a BF16 model with `d = 3072`
//!   scales by `55.5`, not `55.4256`. [`GemmaArchitecture::embedding_scale`]
//!   reproduces that rounding exactly.
//! * **Rotary layout and precision.** Every layer rotates the whole head
//!   with the split-half layout; cosine/sine tables are rounded to the
//!   activation dtype before rotation, as the references do.
//! * **Gemma 3 RoPE sets.** Transformers 5.2 converts legacy fields into a
//!   per-layer-type mapping: `rope_scaling` updates the global set only,
//!   `rope_theta` seeds the global theta, `rope_local_base_freq` the local
//!   one. A per-layer-type `rope_parameters` is used directly, with a
//!   missing entry defaulted; the inert `"rope_type": "default"` and
//!   `"rope_theta": null` that 5.2 writes beside per-type objects for
//!   single-kind `layer_types` are dropped at parse time. A *flat*
//!   `rope_parameters` is rejected: 5.2 would silently replace it by
//!   unscaled defaults. Combining `rope_parameters` with `rope_scaling` is
//!   rejected as ambiguous.
//!   Per-layer-type translation follows
//!   [`RopeParametersByLayerType::to_rope_config`].
//! * **Gemma 3 attention soft-cap.** `Gemma3Attention` stores
//!   `attn_logit_softcapping` but never passes it to the attention kernel,
//!   so a non-null value would be silently ignored by the reference.
//! * **Gemma 3 final soft-cap.** `Gemma3ForCausalLM` applies
//!   `final_logit_softcapping` while the multimodal
//!   `Gemma3ForConditionalGeneration` does not. The architecture applies a
//!   configured cap, as the text tower does, and the checkpoint contract
//!   refuses a non-null cap for a multimodal-namespace checkpoint, whose
//!   reference would not apply it. Every released Gemma 3 config sets `null`.
//! * **Rejected.** Non-zero `attention_dropout`, bidirectional attention
//!   (EmbeddingGemma), activations other than `gelu_pytorch_tanh`,
//!   `gelu_new`, `gelu`, `silu`/`swish`, `relu`, a zero `sliding_window` or
//!   `sliding_window_pattern`, non-positive scalars and soft-caps, layer
//!   types other than `sliding_attention`/`full_attention`, Gemma 3 fields
//!   on a Gemma 2 config (`sliding_window_pattern`, `rope_local_base_freq`,
//!   per-layer-type RoPE), a `quantization_config`, and any geometry whose
//!   matrices cannot be formed.

use std::collections::BTreeMap;

use burn::tensor::{DType, bf16, f16};
use serde::{Deserialize, Serialize};

use super::config::{FAMILY, GemmaConfig, GemmaVariant};
use super::plan::{rotary_sets, sliding_by_pattern};
use crate::{
    architecture::Activation,
    attention::{
        AttentionConnectivity, GroupedQueryAttentionConfig, ProjectionBias, QueryKeyNormScope,
    },
    feed_forward::GatedFeedForwardConfig,
    models::{
        config::{
            ConfigError, PartialRotary, RopeContext, RopeParameterSet, RopeParameters,
            RopeParametersByLayerType, divisible, nonzero, positive_finite,
        },
        llama::{RotaryPrecision, model::graph::RotarySchedule},
    },
    normalization::{RmsNormConfig, RmsNormRootMode},
    position::RopeConfig,
};

/// The serialized name of a sliding (local) attention layer.
const SLIDING_ATTENTION: &str = "sliding_attention";
/// The serialized name of a global attention layer.
const FULL_ATTENTION: &str = "full_attention";

/// The attention kind of one Gemma layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmaLayerType {
    /// A local layer: sliding window and the local RoPE set.
    SlidingAttention,
    /// A global layer: causal attention over every earlier token and the global RoPE set.
    FullAttention,
}

impl GemmaLayerType {
    /// Returns the serialized `layer_types` entry.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SlidingAttention => SLIDING_ATTENTION,
            Self::FullAttention => FULL_ATTENTION,
        }
    }
}

/// Key visibility and rotary set of one decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GemmaLayerSchedule {
    /// Local or global attention.
    pub layer_type: GemmaLayerType,
    /// Which earlier keys a query may see, on top of causality and padding.
    pub connectivity: AttentionConnectivity,
}

/// The validated, executable topology of one Gemma checkpoint.
///
/// Produced by [`GemmaConfig::architecture`]; every field is a direct input
/// of the model graph, the checkpoint contract, or the decode cache.
#[derive(Debug, Clone, PartialEq)]
pub struct GemmaArchitecture {
    /// Reference whose translation rules produced this topology.
    pub variant: GemmaVariant,
    /// Vocabulary size `V`.
    pub vocab_size: usize,
    /// Residual width `d`.
    pub hidden_size: usize,
    /// GeGLU intermediate width.
    pub intermediate_size: usize,
    /// Positions must lie in `0..max_position_embeddings`.
    pub max_position_embeddings: usize,
    /// Query heads `H_q`.
    pub query_heads: usize,
    /// Key/value heads `H_kv`, dividing `H_q`.
    pub key_value_heads: usize,
    /// Per-head width `d_h`.
    pub head_dim: usize,
    /// Whether all four attention projections carry biases.
    pub attention_bias: bool,
    /// Whether queries and keys pass through per-head zero-centered RMSNorms.
    pub query_key_norm: bool,
    /// Score multiplier `query_pre_attn_scalar^(-1/2)`.
    pub attention_scale: f64,
    /// Optional attention-score soft-cap `c` (`c · tanh(s / c)`).
    pub attention_softcap: Option<f64>,
    /// Optional final-logit soft-cap `c` (`c · tanh(z / c)`).
    pub final_logit_softcap: Option<f64>,
    /// Gate nonlinearity of the GeGLU block.
    pub activation: Activation,
    /// RMSNorm epsilon of every normalization.
    pub norm_epsilon: f64,
    /// Rotary configuration of sliding (local) layers.
    pub sliding_rope: RopeConfig,
    /// Rotary configuration of global layers.
    pub full_rope: RopeConfig,
    /// Per-layer attention kind and visibility, one entry per block.
    pub layers: Vec<GemmaLayerSchedule>,
    /// Whether the LM head reuses the embedding table.
    pub tie_word_embeddings: bool,
}

impl GemmaArchitecture {
    /// Translates and validates `config`; see the module translation table.
    pub(super) fn from_config(config: &GemmaConfig) -> Result<Self, ConfigError> {
        let variant = config.variant().ok_or_else(|| {
            ConfigError::new(
                FAMILY,
                "model_type",
                format!(
                    "`{}` is not one of {}",
                    config.model_type,
                    GemmaVariant::MODEL_TYPES.join(", ")
                ),
            )
        })?;
        let family = variant.model_type();
        validate_geometry(config, family)?;
        reject_unsupported(config, variant)?;
        let (sliding_rope, full_rope) = ropes(config, variant)?;
        let architecture = Self {
            variant,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            max_position_embeddings: config.max_position_embeddings,
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            attention_bias: config.attention_bias,
            query_key_norm: variant == GemmaVariant::Gemma3Text,
            attention_scale: config.query_pre_attn_scalar.powf(-0.5),
            attention_softcap: config.attn_logit_softcapping,
            final_logit_softcap: config.final_logit_softcapping,
            activation: activation(config, family)?,
            norm_epsilon: config.rms_norm_eps,
            sliding_rope,
            full_rope,
            layers: layer_schedule(config, variant)?,
            tie_word_embeddings: config.tie_word_embeddings,
        };
        config.validate_token_ids(family)?;
        architecture
            .attention_config()
            .validate()
            .map_err(|error| ConfigError::new(family, "attention", error.to_string()))?;
        Ok(architecture)
    }

    /// Returns the number of decoder blocks.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the shared grouped-query attention configuration of every block.
    ///
    /// The scale and soft-cap are explicit; Gemma 3's per-head Q/K norms are
    /// zero-centered. [`Self::rms_norm`]'s precision conventions are applied
    /// to those norms when a block is built.
    #[must_use]
    pub fn attention_config(&self) -> GroupedQueryAttentionConfig {
        let mut config = GroupedQueryAttentionConfig::new(
            self.hidden_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
        )
        .with_projection_bias(ProjectionBias::uniform(self.attention_bias));
        config.scale = Some(self.attention_scale);
        config.softcap = self.attention_softcap;
        if self.query_key_norm {
            config = config.with_query_key_norm(self.norm_epsilon, QueryKeyNormScope::PerHead);
            config.query_key_norm_zero_centered = true;
        }
        config
    }

    /// Returns the GeGLU configuration of every block.
    #[must_use]
    pub fn mlp_config(&self) -> GatedFeedForwardConfig {
        GatedFeedForwardConfig::new(self.hidden_size, self.intermediate_size)
            .with_activation(self.activation)
            .with_bias(false)
    }

    /// Returns the family's zero-centered RMSNorm over `size` channels.
    ///
    /// `cast((1 + w) · x · rsqrt(mean(x²) + ε))` with every step in F32, as
    /// both references compute it.
    #[must_use]
    pub fn rms_norm(&self, size: usize) -> RmsNormConfig {
        RmsNormConfig::new(size)
            .with_epsilon(self.norm_epsilon)
            .with_zero_centered(true)
            .with_round_before_scale(false)
            .with_root_mode(RmsNormRootMode::MultiplyByReciprocalSqrt)
    }

    /// Returns `sqrt(hidden_size)` rounded to `dtype`, the embedding multiplier.
    ///
    /// The value is rounded to F32 and then to `dtype`, exactly as the
    /// references' scalar tensor is, so multiplying an embedding of `dtype`
    /// by it reproduces their product bit for bit. Integer dtypes keep the
    /// F32 value (embeddings are always floating point).
    #[must_use]
    pub fn embedding_scale(&self, dtype: DType) -> f64 {
        let single = (self.hidden_size as f64).sqrt() as f32;
        match dtype {
            DType::F16 => f64::from(f16::from_f32(single).to_f32()),
            DType::BF16 => f64::from(bf16::from_f32(single).to_f32()),
            _ => f64::from(single),
        }
    }

    /// Returns the sliding-window width shared by every sliding layer, if any.
    #[must_use]
    pub fn sliding_window(&self) -> Option<usize> {
        self.layers
            .iter()
            .find_map(|layer| match layer.connectivity {
                AttentionConnectivity::SlidingWindow { window } => Some(window),
                AttentionConnectivity::Global | AttentionConnectivity::ChunkedLocal { .. } => None,
            })
    }

    /// Returns the rotary sets and the set each layer rotates with.
    ///
    /// Equal sliding and global configurations (always for Gemma 2) share
    /// one set, so one table is built per call.
    pub(crate) fn rotary_schedule(&self) -> RotarySchedule {
        let shared = self.sliding_rope == self.full_rope;
        let sliding: Vec<bool> = self
            .layers
            .iter()
            .map(|layer| layer.layer_type == GemmaLayerType::SlidingAttention)
            .collect();
        let sets = if shared {
            vec![self.sliding_rope]
        } else {
            vec![self.sliding_rope, self.full_rope]
        };
        RotarySchedule {
            sets,
            layers: rotary_sets(&sliding, shared)
                .into_iter()
                .map(Some)
                .collect(),
            precision: RotaryPrecision::ActivationDtype,
        }
    }
}

/// The largest accepted `num_hidden_layers`, `2^20`.
///
/// Validation materializes one schedule entry per layer, so an absurd count
/// in an untrusted `config.json` would exhaust memory (or overflow a vector's
/// capacity) before any error could be reported. Every layer owns at least
/// eleven checkpoint tensors, so under the default artifact limit of ten
/// million tensors no loadable checkpoint has more than about 909,000
/// layers; the bound leaves headroom while capping validation at a few
/// megabytes.
pub const MAX_LAYERS: usize = 1 << 20;

/// Checks the positive extents, the layer bound, grouped-query divisibility, and scalars.
fn validate_geometry(config: &GemmaConfig, family: &'static str) -> Result<(), ConfigError> {
    for (field, value) in [
        ("vocab_size", config.vocab_size),
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("head_dim", config.head_dim),
        ("max_position_embeddings", config.max_position_embeddings),
        ("sliding_window", config.sliding_window),
    ] {
        nonzero(family, field, value)?;
    }
    if config.num_hidden_layers > MAX_LAYERS {
        return Err(ConfigError::new(
            family,
            "num_hidden_layers",
            format!(
                "{} exceeds the supported maximum of {MAX_LAYERS} layers",
                config.num_hidden_layers
            ),
        ));
    }
    divisible(
        family,
        "num_attention_heads",
        config.num_attention_heads,
        "num_key_value_heads",
        config.num_key_value_heads,
    )?;
    positive_finite(family, "rms_norm_eps", config.rms_norm_eps)?;
    positive_finite(
        family,
        "query_pre_attn_scalar",
        config.query_pre_attn_scalar,
    )?;
    for (field, cap) in [
        ("attn_logit_softcapping", config.attn_logit_softcapping),
        ("final_logit_softcapping", config.final_logit_softcapping),
    ] {
        if let Some(cap) = cap {
            positive_finite(family, field, cap)?;
        }
    }
    Ok(())
}

/// Rejects directives the graph cannot honor and fields the variant's reference never reads.
fn reject_unsupported(config: &GemmaConfig, variant: GemmaVariant) -> Result<(), ConfigError> {
    let family = variant.model_type();
    let reject = |field, reason: &str| Err(ConfigError::new(family, field, reason));
    if config.attention_dropout != 0.0 {
        return reject(
            "attention_dropout",
            "attention dropout is not implemented; inference and training use 0.0",
        );
    }
    if config.quantization_config.is_some() {
        return reject(
            "quantization_config",
            "quantized checkpoint formats are not supported; load dense SafeTensors weights",
        );
    }
    if config.use_bidirectional_attention {
        return reject(
            "use_bidirectional_attention",
            "bidirectional (embedding) Gemma models are not causal language models",
        );
    }
    if config.rope_parameters.is_some() && config.rope_scaling.is_some() {
        return reject(
            "rope_scaling",
            "give either rope_parameters or the legacy rope_scaling, not both",
        );
    }
    match variant {
        GemmaVariant::Gemma2 => {
            if config.sliding_window_pattern.is_some() {
                return reject(
                    "sliding_window_pattern",
                    "Gemma 2 alternates sliding and global layers and never reads this field",
                );
            }
            if config.rope_local_base_freq.is_some() {
                return reject(
                    "rope_local_base_freq",
                    "Gemma 2 rotates every layer with one RoPE base and never reads this field",
                );
            }
        }
        GemmaVariant::Gemma3Text => {
            if config.attn_logit_softcapping.is_some() {
                return reject(
                    "attn_logit_softcapping",
                    "the Gemma 3 reference never applies an attention soft-cap; set it to null",
                );
            }
            if config.sliding_window_pattern == Some(0) {
                return reject("sliding_window_pattern", "must be nonzero");
            }
        }
    }
    Ok(())
}

/// Maps the Transformers `ACT2FN` name to the shared activation.
fn activation(config: &GemmaConfig, family: &'static str) -> Result<Activation, ConfigError> {
    match config.hidden_activation.as_str() {
        "gelu_pytorch_tanh" | "gelu_new" => Ok(Activation::GeluTanh),
        "gelu" => Ok(Activation::Gelu),
        "silu" | "swish" => Ok(Activation::Silu),
        "relu" => Ok(Activation::Relu),
        other => Err(ConfigError::new(
            family,
            "hidden_activation",
            format!("`{other}` is not one of gelu_pytorch_tanh, gelu_new, gelu, silu, swish, relu"),
        )),
    }
}

/// Derives the per-layer attention kind and visibility.
fn layer_schedule(
    config: &GemmaConfig,
    variant: GemmaVariant,
) -> Result<Vec<GemmaLayerSchedule>, ConfigError> {
    let family = variant.model_type();
    let layers = config.num_hidden_layers;
    let types = match &config.layer_types {
        Some(types) => parse_layer_types(types, layers, family)?,
        None => {
            let pattern = match variant {
                GemmaVariant::Gemma2 => 2,
                GemmaVariant::Gemma3Text => config.sliding_window_pattern.unwrap_or(6),
            };
            (0..layers)
                .map(|layer| {
                    if sliding_by_pattern(layer, pattern) {
                        GemmaLayerType::SlidingAttention
                    } else {
                        GemmaLayerType::FullAttention
                    }
                })
                .collect()
        }
    };
    Ok(types
        .into_iter()
        .map(|layer_type| GemmaLayerSchedule {
            layer_type,
            connectivity: match layer_type {
                GemmaLayerType::SlidingAttention => AttentionConnectivity::SlidingWindow {
                    window: config.sliding_window,
                },
                GemmaLayerType::FullAttention => AttentionConnectivity::Global,
            },
        })
        .collect())
}

/// Parses `full_attention`/`sliding_attention` entries, one per layer.
fn parse_layer_types(
    types: &[String],
    layers: usize,
    family: &'static str,
) -> Result<Vec<GemmaLayerType>, ConfigError> {
    if types.len() != layers {
        return Err(ConfigError::new(
            family,
            "layer_types",
            format!("must list {layers} entries, found {}", types.len()),
        ));
    }
    types
        .iter()
        .map(|kind| match kind.as_str() {
            SLIDING_ATTENTION => Ok(GemmaLayerType::SlidingAttention),
            FULL_ATTENTION => Ok(GemmaLayerType::FullAttention),
            other => Err(ConfigError::new(
                family,
                "layer_types",
                format!("`{other}` is not {FULL_ATTENTION} or {SLIDING_ATTENTION}"),
            )),
        })
        .collect()
}

/// Translates the serialized RoPE fields into the sliding and global sets.
fn ropes(
    config: &GemmaConfig,
    variant: GemmaVariant,
) -> Result<(RopeConfig, RopeConfig), ConfigError> {
    let family = variant.model_type();
    let context = |theta, default| {
        RopeContext::new(family, config.head_dim, config.max_position_embeddings)
            .with_rope_theta(theta)
            .with_default_rope_theta(default)
            .with_partial_rotary(PartialRotary::FullHeadOnly)
    };
    let error = |field, error: crate::Error| ConfigError::new(family, field, error.to_string());
    match variant {
        GemmaVariant::Gemma2 => {
            let parameters = match (&config.rope_parameters, &config.rope_scaling) {
                (Some(RopeParameterSet::ByLayerType(_)), _) => {
                    return Err(ConfigError::new(
                        family,
                        "rope_parameters",
                        "Gemma 2 reads one flat RoPE mapping, not a per-layer-type mapping",
                    ));
                }
                (Some(RopeParameterSet::Uniform(parameters)), _) | (None, Some(parameters)) => {
                    parameters.clone()
                }
                (None, None) => RopeParameters::default(),
            };
            let rope = parameters
                .to_rope_config(&context(config.rope_theta, 10_000.0))
                .map_err(|failure| error("rope_parameters", failure))?;
            Ok((rope, rope))
        }
        GemmaVariant::Gemma3Text => {
            let by_type = gemma3_rope_mapping(config, family)?;
            let sliding = by_type
                .to_rope_config(
                    SLIDING_ATTENTION,
                    &context(config.rope_local_base_freq, 10_000.0),
                )
                .map_err(|failure| error("rope_parameters", failure))?;
            let full = by_type
                .to_rope_config(FULL_ATTENTION, &context(config.rope_theta, 1_000_000.0))
                .map_err(|failure| error("rope_parameters", failure))?;
            Ok((sliding, full))
        }
    }
}

/// Builds Gemma 3's per-layer-type mapping as Transformers 5.2 does.
///
/// A per-layer-type `rope_parameters` is used with missing entries
/// defaulted; the legacy `rope_scaling` becomes the global entry; with
/// neither, both entries are default RoPE. Thetas are left to the per-type
/// contexts (nested values win, as `setdefault` does).
fn gemma3_rope_mapping(
    config: &GemmaConfig,
    family: &'static str,
) -> Result<RopeParametersByLayerType, ConfigError> {
    let mut mapping = BTreeMap::new();
    match (&config.rope_parameters, &config.rope_scaling) {
        (Some(RopeParameterSet::ByLayerType(by_type)), _) => {
            for (layer_type, parameters) in by_type.iter() {
                if layer_type != SLIDING_ATTENTION && layer_type != FULL_ATTENTION {
                    return Err(ConfigError::new(
                        family,
                        "rope_parameters",
                        format!("`{layer_type}` is not {FULL_ATTENTION} or {SLIDING_ATTENTION}"),
                    ));
                }
                mapping.insert(layer_type.to_owned(), parameters.clone());
            }
        }
        (Some(RopeParameterSet::Uniform(parameters)), _) => {
            if *parameters != RopeParameters::default() {
                return Err(ConfigError::new(
                    family,
                    "rope_parameters",
                    "Gemma 3 reads RoPE per layer type; use {sliding_attention, full_attention} or the legacy rope_scaling (a flat mapping is silently replaced by defaults in Transformers 5.2)",
                ));
            }
        }
        (None, Some(scaling)) => {
            mapping.insert(FULL_ATTENTION.to_owned(), scaling.clone());
        }
        (None, None) => {}
    }
    for layer_type in [SLIDING_ATTENTION, FULL_ATTENTION] {
        mapping.entry(layer_type.to_owned()).or_default();
    }
    RopeParametersByLayerType::new(mapping)
        .map_err(|error| ConfigError::new(family, "rope_parameters", error.to_string()))
}

#[cfg(test)]
#[path = "architecture_tests.rs"]
mod tests;
