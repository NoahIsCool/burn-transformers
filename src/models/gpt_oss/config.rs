//! The serialized `config.json` schema of OpenAI gpt-oss checkpoints.
//!
//! [`GptOssConfig`] mirrors Transformers 5.2.0 `GptOssConfig`
//! (`models/gpt_oss/configuration_gpt_oss.py`): deserialization fills every
//! absent field with that constructor's default, so a parsed config states
//! every value explicitly and serializes back to an equivalent document.
//!
//! # Defaults
//!
//! | field | default | note |
//! | --- | --- | --- |
//! | `vocab_size` | 201088 | o200k Harmony vocabulary |
//! | `hidden_size`, `intermediate_size` | 2880, 2880 | |
//! | `num_hidden_layers` | 36 | gpt-oss-120b; gpt-oss-20b serializes 24 |
//! | `num_attention_heads`, `num_key_value_heads`, `head_dim` | 64, 8, 64 | `null` heads mean "equal to the query heads", `null` `head_dim` means `hidden / heads` |
//! | `num_local_experts`, `num_experts_per_tok` | 128, 4 | |
//! | `sliding_window` | 128 | applied to `sliding_attention` layers |
//! | `layer_types` | sliding for even layers, full for odd | `l` slides iff `(l + 1) mod 2 != 0` |
//! | `rms_norm_eps` | 1e-5 | |
//! | `max_position_embeddings` | 131072 | |
//! | `rope_parameters` | YaRN, factor 32, `beta_fast` 32, `beta_slow` 1, `truncate: false`, original context 4096 | legacy key `rope_scaling` |
//! | default `rope_theta` | 150000 | the class's `default_theta` |
//! | `attention_bias` | `true` | the constructor hard-codes it; a serialized value overrides it |
//! | `tie_word_embeddings` | `false` | |
//! | `router_aux_loss_coef`, `output_router_logits` | 0.9, `false` | |
//! | `pad_token_id`, `bos_token_id`, `eos_token_id` | none | |
//!
//! # Fields the reference does not read
//!
//! OpenAI's original files also serialize `swiglu_limit` (7.0),
//! `experts_per_token`, `initial_context_length`, and `hidden_act`. The pinned
//! reference hard-codes its clamped SwiGLU (`limit = 7.0`, `alpha = 1.702`)
//! and reads only `num_experts_per_tok`. This schema keeps `swiglu_limit` and
//! `hidden_act` so validation can reject values the reference would silently
//! ignore (anything but `7.0` and `silu`); every other unknown key is ignored.
//!
//! `quantization_config` names the checkpoint's weight format. The released
//! checkpoints declare `{"quant_method": "mxfp4", ...}`: routed-expert
//! projections are stored as MXFP4 blocks and scales, decoded exactly at load
//! time, while attention, router, embedding, and head weights stay dense.

use serde::{Deserialize, Serialize};

use super::architecture::GptOssArchitecture;
use crate::models::config::{
    ConfigError, ModelConfig, RopeParameters, RopeType, TokenIds, Validate,
};

/// Family name used in configuration errors.
pub(super) const FAMILY: &str = "gpt_oss";

/// The one Hugging Face `model_type` this family executes.
pub(super) const MODEL_TYPE: &str = "gpt_oss";

/// The clamp limit the pinned reference hard-codes in `GptOssExperts`.
pub(super) const SWIGLU_LIMIT: f64 = 7.0;

/// Default `rope_theta` of `GptOssConfig` (`default_theta`).
const DEFAULT_ROPE_THETA: f64 = 150_000.0;

/// A gpt-oss `config.json` with every reference default made explicit.
///
/// Construct it with [`ModelConfig::from_json_str`] (which also validates),
/// with `serde_json`, or with [`GptOssConfig::default`] and struct update
/// syntax. [`Self::architecture`] translates it into the executable
/// topology and is the single source of validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "SerializedGptOssConfig")]
pub struct GptOssConfig {
    /// Hugging Face architecture discriminator; must be `gpt_oss`.
    pub model_type: String,
    /// Vocabulary size `V`: rows of the embedding table and of the LM head.
    pub vocab_size: usize,
    /// Residual-stream width `d`.
    pub hidden_size: usize,
    /// Intermediate width `f` of every routed expert.
    pub intermediate_size: usize,
    /// Number of decoder blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads; divides `num_attention_heads`.
    pub num_key_value_heads: usize,
    /// Width of every query, key, and value head.
    pub head_dim: usize,
    /// Number of routed experts per block `E`.
    pub num_local_experts: usize,
    /// Experts selected per token `K`.
    pub num_experts_per_tok: usize,
    /// Window of `sliding_attention` layers, in tokens; `None` when absent.
    pub sliding_window: Option<usize>,
    /// Per-layer attention kinds (`sliding_attention`/`full_attention`).
    pub layer_types: Vec<String>,
    /// Whether the LM head reuses the embedding table.
    pub tie_word_embeddings: bool,
    /// Serialized gate nonlinearity; the reference ignores it, so only `silu` is accepted.
    pub hidden_act: String,
    /// Initialization standard deviation, retained for round trips only.
    pub initializer_range: f64,
    /// Trained context length; positions must lie below it.
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon shared by every normalization.
    pub rms_norm_eps: f64,
    /// Nested RoPE parameters (current `rope_parameters` or legacy `rope_scaling`).
    pub rope_parameters: RopeParameters,
    /// Legacy top-level RoPE base, a fallback for the nested value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<f64>,
    /// Attention-probability dropout; must be zero (dropout is not implemented).
    pub attention_dropout: f64,
    /// Whether the four attention projections carry biases.
    pub attention_bias: bool,
    /// Training coefficient of the load-balancing loss, retained for callers.
    pub router_aux_loss_coef: f64,
    /// Whether forward passes return router logits by default.
    pub output_router_logits: bool,
    /// OpenAI's serialized clamp limit; must equal the reference's 7.0 when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swiglu_limit: Option<f64>,
    /// Optional padding token ID.
    pub pad_token_id: Option<TokenIds>,
    /// Optional beginning-of-sequence token ID.
    pub bos_token_id: Option<TokenIds>,
    /// One or more end-of-sequence token IDs.
    pub eos_token_id: Option<TokenIds>,
    /// Weight-format directive; only `quant_method: "mxfp4"` is accepted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_config: Option<serde_json::Value>,
}

impl Default for GptOssConfig {
    /// Returns the reference constructor's defaults (gpt-oss-120b geometry).
    fn default() -> Self {
        SerializedGptOssConfig {
            model_type: MODEL_TYPE.into(),
            ..SerializedGptOssConfig::default()
        }
        .resolve()
    }
}

impl GptOssConfig {
    /// Validates the configuration and derives its executable topology.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] naming the offending field; see the
    /// `architecture` module for every rule.
    pub fn architecture(&self) -> Result<GptOssArchitecture, ConfigError> {
        GptOssArchitecture::from_config(self)
    }

    /// Returns the reference default `layer_types` for `layers` blocks.
    ///
    /// Transformers writes `sliding_attention` when `(l + 1) mod 2 != 0`,
    /// i.e. for even `l`, and `full_attention` otherwise.
    #[must_use]
    pub fn default_layer_types(layers: usize) -> Vec<String> {
        (0..layers)
            .map(|layer| {
                if super::plan::default_layer_slides(layer) {
                    "sliding_attention".into()
                } else {
                    "full_attention".into()
                }
            })
            .collect()
    }

    /// Returns the reference default RoPE parameters (YaRN, factor 32, untruncated).
    #[must_use]
    pub fn default_rope_parameters() -> RopeParameters {
        RopeParameters {
            rope_type: RopeType::Yarn,
            factor: Some(32.0),
            beta_fast: Some(32.0),
            beta_slow: Some(1.0),
            truncate: Some(false),
            original_max_position_embeddings: Some(4096),
            ..RopeParameters::default()
        }
    }

    /// Returns the RoPE base used when neither level serializes `rope_theta`.
    #[must_use]
    pub const fn default_rope_theta() -> f64 {
        DEFAULT_ROPE_THETA
    }
}

impl Validate for GptOssConfig {
    /// Validates by deriving the complete executable architecture.
    fn validate(&self) -> Result<(), ConfigError> {
        self.architecture().map(drop)
    }
}

impl ModelConfig for GptOssConfig {
    /// Diagnostic family name used by generic configuration parsing.
    const FAMILY: &'static str = FAMILY;
    /// Checkpoint `model_type` values accepted by this schema.
    const MODEL_TYPES: &'static [&'static str] = &[MODEL_TYPE];
}

/// The literal `config.json` document, before reference defaults.
///
/// Every field is optional so absence is observable; [`Self::resolve`]
/// applies the reference defaults. Unknown keys are ignored, as
/// Transformers ignores keys its modeling code never reads.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct SerializedGptOssConfig {
    /// See [`GptOssConfig::model_type`].
    model_type: String,
    /// See [`GptOssConfig::vocab_size`].
    vocab_size: Option<usize>,
    /// See [`GptOssConfig::hidden_size`].
    hidden_size: Option<usize>,
    /// See [`GptOssConfig::intermediate_size`].
    intermediate_size: Option<usize>,
    /// See [`GptOssConfig::num_hidden_layers`].
    num_hidden_layers: Option<usize>,
    /// See [`GptOssConfig::num_attention_heads`].
    num_attention_heads: Option<usize>,
    /// See [`GptOssConfig::num_key_value_heads`]; JSON `null` means "equal to the query heads".
    #[serde(deserialize_with = "present")]
    num_key_value_heads: Option<Option<usize>>,
    /// See [`GptOssConfig::head_dim`]; JSON `null` means `hidden_size / num_attention_heads`.
    #[serde(deserialize_with = "present")]
    head_dim: Option<Option<usize>>,
    /// See [`GptOssConfig::num_local_experts`].
    num_local_experts: Option<usize>,
    /// See [`GptOssConfig::num_experts_per_tok`].
    num_experts_per_tok: Option<usize>,
    /// See [`GptOssConfig::sliding_window`]; JSON `null` disables it.
    #[serde(deserialize_with = "present")]
    sliding_window: Option<Option<usize>>,
    /// See [`GptOssConfig::layer_types`]; JSON `null` means the default schedule.
    layer_types: Option<Vec<String>>,
    /// See [`GptOssConfig::tie_word_embeddings`].
    tie_word_embeddings: Option<bool>,
    /// See [`GptOssConfig::hidden_act`].
    hidden_act: Option<String>,
    /// See [`GptOssConfig::initializer_range`].
    initializer_range: Option<f64>,
    /// See [`GptOssConfig::max_position_embeddings`].
    max_position_embeddings: Option<usize>,
    /// See [`GptOssConfig::rms_norm_eps`].
    rms_norm_eps: Option<f64>,
    /// See [`GptOssConfig::rope_parameters`]; also read from `rope_scaling`.
    #[serde(alias = "rope_scaling")]
    rope_parameters: Option<RopeParameters>,
    /// See [`GptOssConfig::rope_theta`].
    rope_theta: Option<f64>,
    /// See [`GptOssConfig::attention_dropout`].
    attention_dropout: Option<f64>,
    /// See [`GptOssConfig::attention_bias`].
    attention_bias: Option<bool>,
    /// See [`GptOssConfig::router_aux_loss_coef`].
    router_aux_loss_coef: Option<f64>,
    /// See [`GptOssConfig::output_router_logits`].
    output_router_logits: Option<bool>,
    /// See [`GptOssConfig::swiglu_limit`].
    swiglu_limit: Option<f64>,
    /// See [`GptOssConfig::pad_token_id`].
    pad_token_id: Option<TokenIds>,
    /// See [`GptOssConfig::bos_token_id`].
    bos_token_id: Option<TokenIds>,
    /// See [`GptOssConfig::eos_token_id`].
    eos_token_id: Option<TokenIds>,
    /// See [`GptOssConfig::quantization_config`].
    quantization_config: Option<serde_json::Value>,
}

/// Distinguishes a present JSON `null` (`Some(None)`) from an absent key (`None`).
///
/// Serde calls a field's `deserialize_with` only when the key is present, so
/// wrapping the decoded `Option` in `Some` records presence; `#[serde(default)]`
/// supplies `None` for absent keys.
fn present<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

impl SerializedGptOssConfig {
    /// Fills absent fields from the reference constructor's defaults.
    fn resolve(self) -> GptOssConfig {
        let hidden_size = self.hidden_size.unwrap_or(2880);
        let num_attention_heads = self.num_attention_heads.unwrap_or(64);
        let num_hidden_layers = self.num_hidden_layers.unwrap_or(36);
        GptOssConfig {
            vocab_size: self.vocab_size.unwrap_or(201_088),
            hidden_size,
            intermediate_size: self.intermediate_size.unwrap_or(2880),
            num_hidden_layers,
            num_attention_heads,
            // An explicit `null` falls back to the query heads, as the constructor does.
            num_key_value_heads: self
                .num_key_value_heads
                .unwrap_or(Some(8))
                .unwrap_or(num_attention_heads),
            head_dim: self
                .head_dim
                .unwrap_or(Some(64))
                .unwrap_or_else(|| hidden_size.checked_div(num_attention_heads).unwrap_or(0)),
            num_local_experts: self.num_local_experts.unwrap_or(128),
            num_experts_per_tok: self.num_experts_per_tok.unwrap_or(4),
            sliding_window: self.sliding_window.unwrap_or(Some(128)),
            layer_types: self
                .layer_types
                .unwrap_or_else(|| GptOssConfig::default_layer_types(num_hidden_layers)),
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
            hidden_act: self.hidden_act.unwrap_or_else(|| "silu".into()),
            initializer_range: self.initializer_range.unwrap_or(0.02),
            max_position_embeddings: self.max_position_embeddings.unwrap_or(131_072),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-5),
            rope_parameters: self
                .rope_parameters
                .unwrap_or_else(GptOssConfig::default_rope_parameters),
            rope_theta: self.rope_theta,
            attention_dropout: self.attention_dropout.unwrap_or(0.0),
            attention_bias: self.attention_bias.unwrap_or(true),
            router_aux_loss_coef: self.router_aux_loss_coef.unwrap_or(0.9),
            output_router_logits: self.output_router_logits.unwrap_or(false),
            swiglu_limit: self.swiglu_limit,
            pad_token_id: self.pad_token_id,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            quantization_config: self.quantization_config.filter(|value| !value.is_null()),
            model_type: self.model_type,
        }
    }
}

impl From<SerializedGptOssConfig> for GptOssConfig {
    /// Applies the reference defaults; validation reports every semantic problem.
    fn from(serialized: SerializedGptOssConfig) -> Self {
        serialized.resolve()
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
pub(super) mod tests;
