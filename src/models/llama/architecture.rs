//! Translation of a Llama-family configuration into an executable topology.
//!
//! Every supported checkpoint is the same decoder,
//!
//! ```text
//! h_0     = Embed(tokens)
//! h_{l+1} = Block_l(h_l)                      for l in 0..L
//! logits  = RMSNorm(h_L) · Wᵀ                 (W = lm_head, or Embed when tied)
//! ```
//!
//! whose blocks differ only in a handful of flags. [`LlamaArchitecture`]
//! records those flags; the model reads nothing else, so shared operations
//! never branch on a model name. A block is
//!
//! ```text
//! input-normalized (SublayerNorm::Input):    x' = x + Attn(N₁(x)),   y = x' + MLP(N₂(x'))
//! output-normalized (SublayerNorm::Output):  x' = x + N₁(Attn(x)),   y = x' + N₂(MLP(x'))
//! ```
//!
//! with grouped-query attention `Attn`, a SwiGLU `MLP(x) = down(silu(gate x) ⊙ up x)`,
//! and RMSNorm `N(x) = w ⊙ x / sqrt(mean(x²) + ε)`.
//!
//! # Translation table
//!
//! | executable choice | llama | mistral | qwen2 | qwen3 | smollm3 | olmo2 |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | q/k/v/o bias | `attention_bias` on all | none | q, k, v only | `attention_bias` on all | `attention_bias` on all | `attention_bias` on all |
//! | MLP bias | `mlp_bias` | none | none | none | `mlp_bias` | none |
//! | Q/K RMSNorm | none | none | none | per head (`head_dim`) | none | full width (`heads·head_dim`) |
//! | sublayer norm | input | input | input | input | input | output |
//! | RMSNorm rounding | cast, then scale | cast, then scale | cast, then scale | cast, then scale | cast, then scale | scale, then cast |
//! | rotary arithmetic | activation dtype | activation dtype | activation dtype | activation dtype | activation dtype | F32 |
//! | NoPE layers | none | none | none | none | `no_rope_layers[l] == 0` | none |
//! | sliding layers | none | all, if `sliding_window` | `layer_types` | `layer_types` | `layer_types` | none |
//!
//! * **Bias flags.** The Mistral and Qwen2 references hard-code their biases
//!   and never read `attention_bias`; a serialized value contradicting them
//!   (`true` for Mistral, `false` for Qwen2) is rejected rather than silently
//!   ignored. `mlp_bias = true` is rejected for every variant whose reference
//!   hard-codes bias-free MLPs.
//! * **RoPE.** Every variant rotates whole heads with the split-half layout.
//!   The nested `rope_parameters` (or legacy `rope_scaling`), the top-level
//!   `rope_theta` and `partial_rotary_factor`, and the variant's
//!   `default_theta` go through the one shared translator
//!   [`RopeParameters::to_rope_config`](crate::models::config::RopeParameters::to_rope_config)
//!   with [`PartialRotary::FullHeadOnly`] and every other reference-faithful
//!   default, so `llama3`, `linear`, `dynamic`, `yarn`, and `longrope`
//!   scaling all execute, and a partial factor other than one is rejected.
//! * **Sliding windows.** A sliding layer sees the `W` newest valid tokens
//!   (`AttentionConnectivity::SlidingWindow`), matching Transformers'
//!   `kv_idx > q_idx - W`. Mistral applies `sliding_window` to every layer
//!   when it is not `null`. Qwen2/Qwen3 gate the window with
//!   `use_sliding_window` and, absent `layer_types`, slide layers
//!   `l >= max_window_layers`. SmolLM3 reads `sliding_window` directly and,
//!   absent `layer_types`, slides the NoPE layers when `use_sliding_window`.
//!   A `sliding_attention` layer without a window is rejected. Mistral
//!   `layer_types` (a Ministral-style schedule) and sliding `layer_types` on
//!   Llama/OLMo 2 (whose references ignore them) are rejected.
//! * **NoPE.** SmolLM3 layer `l` rotates iff `no_rope_layers[l] == 1`; absent
//!   the list, iff `(l + 1) % no_rope_layer_interval != 0`.
//! * **Precision.** Llama-style RMSNorm computes `w · cast(x_f32 / rms)`;
//!   OLMo 2 computes `cast(w · x_f32 / rms)`. Both use `x · rsqrt(m + ε)`.
//!   OLMo 2 also rotates queries and keys in F32 and rounds afterwards, while
//!   the others round the cosine/sine tables to the activation dtype first.
//!   All of these coincide in F32.
//! * **Rejected.** Non-zero `attention_dropout` (training dropout is not
//!   implemented), activations other than `silu`/`swish`/`gelu`/
//!   `gelu_pytorch_tanh`/`gelu_new`/`relu`, a zero `sliding_window`, a
//!   `quantization_config`, and any geometry whose matrices cannot be formed.

use burn::tensor::DType;
use serde::{Deserialize, Serialize};

use super::config::{FAMILY, LlamaFamilyConfig, LlamaVariant};
use super::plan::{rotary_by_interval, sliding_from_layer};
use crate::{
    architecture::Activation,
    attention::{
        AttentionConnectivity, GroupedQueryAttentionConfig, ProjectionBias, QueryKeyNormScope,
    },
    feed_forward::GatedFeedForwardConfig,
    models::config::{
        ConfigError, PartialRotary, RopeContext, divisible, nonzero, positive_finite,
    },
    normalization::{RmsNormConfig, RmsNormRootMode},
    position::RopeConfig,
};

/// Where each residual sublayer applies its RMSNorm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SublayerNorm {
    /// `x + F(N(x))`: normalize the sublayer input (Llama, Mistral, Qwen, SmolLM3).
    Input,
    /// `x + N(F(x))`: normalize the sublayer output (OLMo 2).
    Output,
}

/// Where RMSNorm rounds to the activation dtype relative to its learned scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormRounding {
    /// `w · cast(x_f32 / rms)`: round the unit-RMS activation, then scale.
    CastThenScale,
    /// `cast(w · x_f32 / rms)`: scale in F32, then round once.
    ScaleThenCast,
}

impl NormRounding {
    /// Returns the shared [`RmsNormConfig::round_before_scale`] flag.
    #[must_use]
    pub const fn round_before_scale(self) -> bool {
        matches!(self, Self::CastThenScale)
    }
}

/// The dtype in which queries and keys are rotated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RotaryPrecision {
    /// Round the F32 cosine/sine tables to the activation dtype, then rotate.
    ActivationDtype,
    /// Rotate in F32 and round the rotated queries and keys once.
    Float32,
}

impl RotaryPrecision {
    /// Returns the dtype rotation runs in for activations of `activation` dtype.
    #[must_use]
    pub const fn rotation_dtype(self, activation: DType) -> DType {
        match self {
            Self::ActivationDtype => activation,
            Self::Float32 => DType::F32,
        }
    }
}

/// Positional encoding and key visibility of one decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaLayerSchedule {
    /// Whether queries and keys are rotated (`false` for SmolLM3 NoPE layers).
    pub rotary: bool,
    /// Which earlier keys a query may see, on top of causality and padding.
    pub connectivity: AttentionConnectivity,
}

impl LlamaLayerSchedule {
    /// Returns the sliding-window width of this layer, if it has one.
    #[must_use]
    pub const fn sliding_window(self) -> Option<usize> {
        match self.connectivity {
            AttentionConnectivity::SlidingWindow { window } => Some(window),
            AttentionConnectivity::Global | AttentionConnectivity::ChunkedLocal { .. } => None,
        }
    }
}

/// The validated, executable topology of one Llama-family checkpoint.
///
/// Produced by [`LlamaFamilyConfig::architecture`]; every field is a direct
/// input of the model graph, the checkpoint contract, or the decode cache.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaArchitecture {
    /// Reference whose translation rules produced this topology.
    pub variant: LlamaVariant,
    /// Vocabulary size `V`.
    pub vocab_size: usize,
    /// Residual width `d`.
    pub hidden_size: usize,
    /// SwiGLU intermediate width.
    pub intermediate_size: usize,
    /// Positions must lie in `0..max_position_embeddings`.
    pub max_position_embeddings: usize,
    /// Query heads `H_q`.
    pub query_heads: usize,
    /// Key/value heads `H_kv`, dividing `H_q`.
    pub key_value_heads: usize,
    /// Per-head width `d_h`.
    pub head_dim: usize,
    /// Bias flags of the four attention projections.
    pub projection_bias: ProjectionBias,
    /// Optional query/key RMSNorm scope.
    pub query_key_norm: Option<QueryKeyNormScope>,
    /// Whether the three MLP projections carry biases.
    pub mlp_bias: bool,
    /// Gate nonlinearity of the SwiGLU block.
    pub activation: Activation,
    /// Residual normalization placement.
    pub sublayer_norm: SublayerNorm,
    /// RMSNorm rounding convention.
    pub norm_rounding: NormRounding,
    /// RMSNorm epsilon.
    pub norm_epsilon: f64,
    /// Shared rotary configuration of every rotary layer.
    pub rope: RopeConfig,
    /// Dtype in which rotation runs.
    pub rotary_precision: RotaryPrecision,
    /// Per-layer rotary and connectivity schedule, one entry per block.
    pub layers: Vec<LlamaLayerSchedule>,
    /// Whether the LM head reuses the embedding table.
    pub tie_word_embeddings: bool,
}

impl LlamaArchitecture {
    /// Translates and validates `config`; see the module translation table.
    pub(super) fn from_config(config: &LlamaFamilyConfig) -> Result<Self, ConfigError> {
        let variant = config.variant().ok_or_else(|| {
            ConfigError::new(
                FAMILY,
                "model_type",
                format!(
                    "`{}` is not one of {}",
                    config.model_type,
                    LlamaVariant::MODEL_TYPES.join(", ")
                ),
            )
        })?;
        let family = variant.model_type();
        validate_geometry(config, family)?;
        reject_unsupported(config, family)?;
        let architecture = Self {
            variant,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            max_position_embeddings: config.max_position_embeddings,
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            projection_bias: projection_bias(config, variant)?,
            query_key_norm: query_key_norm(variant),
            mlp_bias: mlp_bias(config, variant)?,
            activation: activation(config, family)?,
            sublayer_norm: match variant {
                LlamaVariant::Olmo2 => SublayerNorm::Output,
                _ => SublayerNorm::Input,
            },
            norm_rounding: match variant {
                LlamaVariant::Olmo2 => NormRounding::ScaleThenCast,
                _ => NormRounding::CastThenScale,
            },
            norm_epsilon: config.rms_norm_eps,
            rope: rope(config, family)?,
            rotary_precision: match variant {
                LlamaVariant::Olmo2 => RotaryPrecision::Float32,
                _ => RotaryPrecision::ActivationDtype,
            },
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
    #[must_use]
    pub fn attention_config(&self) -> GroupedQueryAttentionConfig {
        let config = GroupedQueryAttentionConfig::new(
            self.hidden_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
        )
        .with_projection_bias(self.projection_bias);
        match self.query_key_norm {
            Some(scope) => config.with_query_key_norm(self.norm_epsilon, scope),
            None => config,
        }
    }

    /// Returns the SwiGLU configuration of every block.
    #[must_use]
    pub fn mlp_config(&self) -> GatedFeedForwardConfig {
        GatedFeedForwardConfig::new(self.hidden_size, self.intermediate_size)
            .with_activation(self.activation)
            .with_bias(self.mlp_bias)
    }

    /// Returns the family RMSNorm configuration over `size` channels.
    ///
    /// Transformers evaluates `x · rsqrt(mean(x²) + ε)`, expressed portably
    /// as [`RmsNormRootMode::MultiplyByReciprocalSqrt`].
    #[must_use]
    pub fn rms_norm(&self, size: usize) -> RmsNormConfig {
        RmsNormConfig::new(size)
            .with_epsilon(self.norm_epsilon)
            .with_round_before_scale(self.norm_rounding.round_before_scale())
            .with_root_mode(RmsNormRootMode::MultiplyByReciprocalSqrt)
    }

    /// Returns the sliding-window width shared by every sliding layer, if any.
    ///
    /// Every supported reference uses one `sliding_window` for all of its
    /// sliding layers, which validation guarantees.
    #[must_use]
    pub fn sliding_window(&self) -> Option<usize> {
        self.layers.iter().find_map(|layer| layer.sliding_window())
    }
}

/// Checks the positive extents and grouped-query divisibility.
fn validate_geometry(config: &LlamaFamilyConfig, family: &'static str) -> Result<(), ConfigError> {
    for (field, value) in [
        ("vocab_size", config.vocab_size),
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("head_dim", config.head_dim),
        ("max_position_embeddings", config.max_position_embeddings),
    ] {
        nonzero(family, field, value)?;
    }
    divisible(
        family,
        "num_attention_heads",
        config.num_attention_heads,
        "num_key_value_heads",
        config.num_key_value_heads,
    )?;
    positive_finite(family, "rms_norm_eps", config.rms_norm_eps)
}

/// Rejects dropout and packed-weight directives the dense graph cannot honor.
fn reject_unsupported(config: &LlamaFamilyConfig, family: &'static str) -> Result<(), ConfigError> {
    if config.attention_dropout != 0.0 {
        return Err(ConfigError::new(
            family,
            "attention_dropout",
            "attention dropout is not implemented; inference and training use 0.0",
        ));
    }
    if config.quantization_config.is_some() {
        return Err(ConfigError::new(
            family,
            "quantization_config",
            "quantized checkpoint formats are not supported; load dense SafeTensors weights",
        ));
    }
    Ok(())
}

/// Maps the Transformers `ACT2FN` name to the shared activation.
fn activation(config: &LlamaFamilyConfig, family: &'static str) -> Result<Activation, ConfigError> {
    match config.hidden_act.as_str() {
        "silu" | "swish" => Ok(Activation::Silu),
        "gelu" => Ok(Activation::Gelu),
        "gelu_pytorch_tanh" | "gelu_new" => Ok(Activation::GeluTanh),
        "relu" => Ok(Activation::Relu),
        other => Err(ConfigError::new(
            family,
            "hidden_act",
            format!("`{other}` is not one of silu, swish, gelu, gelu_pytorch_tanh, gelu_new, relu"),
        )),
    }
}

/// Derives the attention projection biases of `variant`.
fn projection_bias(
    config: &LlamaFamilyConfig,
    variant: LlamaVariant,
) -> Result<ProjectionBias, ConfigError> {
    let family = variant.model_type();
    match variant {
        LlamaVariant::Mistral => {
            if config.attention_bias == Some(true) {
                return Err(ConfigError::new(
                    family,
                    "attention_bias",
                    "the Mistral reference has bias-free projections and ignores this field",
                ));
            }
            Ok(ProjectionBias::uniform(false))
        }
        LlamaVariant::Qwen2 => {
            if config.attention_bias == Some(false) {
                return Err(ConfigError::new(
                    family,
                    "attention_bias",
                    "the Qwen2 reference always biases q/k/v and ignores this field",
                ));
            }
            Ok(ProjectionBias::query_key_value_only())
        }
        LlamaVariant::Llama | LlamaVariant::Qwen3 | LlamaVariant::SmolLm3 | LlamaVariant::Olmo2 => {
            Ok(ProjectionBias::uniform(
                config.attention_bias.unwrap_or(false),
            ))
        }
    }
}

/// Derives whether the SwiGLU projections carry biases.
fn mlp_bias(config: &LlamaFamilyConfig, variant: LlamaVariant) -> Result<bool, ConfigError> {
    let serialized = config.mlp_bias.unwrap_or(false);
    match variant {
        LlamaVariant::Llama | LlamaVariant::SmolLm3 => Ok(serialized),
        _ if serialized => Err(ConfigError::new(
            variant.model_type(),
            "mlp_bias",
            "this reference has bias-free MLP projections and ignores this field",
        )),
        _ => Ok(false),
    }
}

/// Returns the query/key normalization scope of `variant`.
const fn query_key_norm(variant: LlamaVariant) -> Option<QueryKeyNormScope> {
    match variant {
        LlamaVariant::Qwen3 => Some(QueryKeyNormScope::PerHead),
        LlamaVariant::Olmo2 => Some(QueryKeyNormScope::FullWidth),
        LlamaVariant::Llama
        | LlamaVariant::Mistral
        | LlamaVariant::Qwen2
        | LlamaVariant::SmolLm3 => None,
    }
}

/// Translates the serialized RoPE fields through the shared translator.
fn rope(config: &LlamaFamilyConfig, family: &'static str) -> Result<RopeConfig, ConfigError> {
    let context = RopeContext::new(family, config.head_dim, config.max_position_embeddings)
        .with_rope_theta(config.rope_theta)
        .with_default_rope_theta(config.default_rope_theta)
        .with_partial_rotary_factor(config.partial_rotary_factor)
        .with_partial_rotary(PartialRotary::FullHeadOnly);
    config
        .rope_parameters
        .clone()
        .unwrap_or_default()
        .to_rope_config(&context)
        .map_err(|error| ConfigError::new(family, "rope_parameters", error.to_string()))
}

/// Derives the per-layer rotary and sliding-window schedule.
fn layer_schedule(
    config: &LlamaFamilyConfig,
    variant: LlamaVariant,
) -> Result<Vec<LlamaLayerSchedule>, ConfigError> {
    let family = variant.model_type();
    let layers = config.num_hidden_layers;
    let rotary = rotary_flags(config, variant)?;
    let sliding = sliding_flags(config, variant, &rotary)?;
    let window = sliding_window(config, variant);
    if window == Some(0) {
        return Err(ConfigError::new(
            family,
            "sliding_window",
            "must be nonzero",
        ));
    }
    (0..layers)
        .map(|layer| {
            let connectivity = match (sliding[layer], window) {
                (false, _) => AttentionConnectivity::Global,
                (true, Some(window)) => AttentionConnectivity::SlidingWindow { window },
                (true, None) => {
                    return Err(ConfigError::new(
                        family,
                        "layer_types",
                        format!("layer {layer} is sliding_attention but no sliding_window applies"),
                    ));
                }
            };
            Ok(LlamaLayerSchedule {
                rotary: rotary[layer],
                connectivity,
            })
        })
        .collect()
}

/// Returns the window the variant's reference applies to sliding layers.
fn sliding_window(config: &LlamaFamilyConfig, variant: LlamaVariant) -> Option<usize> {
    match variant {
        LlamaVariant::Mistral | LlamaVariant::SmolLm3 => config.sliding_window,
        LlamaVariant::Qwen2 | LlamaVariant::Qwen3 => config
            .use_sliding_window
            .then_some(config.sliding_window)
            .flatten(),
        LlamaVariant::Llama | LlamaVariant::Olmo2 => None,
    }
}

/// Returns one rotary flag per layer: SmolLM3's NoPE schedule, otherwise all `true`.
fn rotary_flags(
    config: &LlamaFamilyConfig,
    variant: LlamaVariant,
) -> Result<Vec<bool>, ConfigError> {
    let family = variant.model_type();
    let layers = config.num_hidden_layers;
    let explicit = match &config.no_rope_layers {
        Some(flags) => {
            if flags.len() != layers || flags.iter().any(|&flag| flag > 1) {
                return Err(ConfigError::new(
                    family,
                    "no_rope_layers",
                    format!("must list {layers} flags, each 0 (NoPE) or 1 (RoPE)"),
                ));
            }
            Some(flags.iter().map(|&flag| flag == 1).collect::<Vec<_>>())
        }
        None => None,
    };
    if variant != LlamaVariant::SmolLm3 {
        if explicit
            .as_ref()
            .is_some_and(|flags| flags.contains(&false))
        {
            return Err(ConfigError::new(
                family,
                "no_rope_layers",
                "NoPE layers are a SmolLM3 feature; this reference rotates every layer",
            ));
        }
        return Ok(vec![true; layers]);
    }
    if let Some(flags) = explicit {
        return Ok(flags);
    }
    let interval = config.no_rope_layer_interval.unwrap_or(0);
    nonzero(family, "no_rope_layer_interval", interval)?;
    Ok((0..layers)
        .map(|layer| rotary_by_interval(layer, interval))
        .collect())
}

/// Returns one sliding flag per layer from `layer_types` or the variant's default rule.
fn sliding_flags(
    config: &LlamaFamilyConfig,
    variant: LlamaVariant,
    rotary: &[bool],
) -> Result<Vec<bool>, ConfigError> {
    let family = variant.model_type();
    let layers = config.num_hidden_layers;
    if let Some(types) = &config.layer_types {
        if variant == LlamaVariant::Mistral {
            return Err(ConfigError::new(
                family,
                "layer_types",
                "Mistral ignores layer_types; an alternating schedule is a Ministral checkpoint",
            ));
        }
        let flags = parse_layer_types(types, layers, family)?;
        if matches!(variant, LlamaVariant::Llama | LlamaVariant::Olmo2) && flags.contains(&true) {
            return Err(ConfigError::new(
                family,
                "layer_types",
                "this reference ignores layer_types and has no sliding-window layers",
            ));
        }
        return Ok(flags);
    }
    let window = sliding_window(config, variant);
    Ok(match variant {
        LlamaVariant::Mistral => vec![window.is_some(); layers],
        LlamaVariant::Qwen2 | LlamaVariant::Qwen3 => {
            let first = config.max_window_layers.unwrap_or(layers);
            (0..layers)
                .map(|layer| window.is_some() && sliding_from_layer(layer, first))
                .collect()
        }
        LlamaVariant::SmolLm3 => rotary
            .iter()
            .map(|&rotates| config.use_sliding_window && window.is_some() && !rotates)
            .collect(),
        LlamaVariant::Llama | LlamaVariant::Olmo2 => vec![false; layers],
    })
}

/// Parses `full_attention`/`sliding_attention` into per-layer sliding flags.
fn parse_layer_types(
    types: &[String],
    layers: usize,
    family: &'static str,
) -> Result<Vec<bool>, ConfigError> {
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
            "full_attention" => Ok(false),
            "sliding_attention" => Ok(true),
            other => Err(ConfigError::new(
                family,
                "layer_types",
                format!("`{other}` is not full_attention or sliding_attention"),
            )),
        })
        .collect()
}

#[cfg(test)]
#[path = "architecture_tests.rs"]
mod tests;
