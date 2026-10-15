//! One serialized configuration schema for every Llama-family `config.json`.
//!
//! Llama, Mistral, Qwen2, Qwen3, SmolLM3, and OLMo 2 checkpoints share one
//! Hugging Face field vocabulary but disagree on defaults and on which fields
//! their reference code reads. [`LlamaFamilyConfig`] is the superset schema:
//! deserialization fills every absent field with the default of the
//! checkpoint's own `model_type` (the pinned Transformers 5.2.0
//! `configuration_*.py` constructor), so a parsed config states every value
//! explicitly and serializes back to an equivalent document.
//!
//! The configuration describes *what the checkpoint says*. Turning it into an
//! executable topology (bias flags, normalization placement, the per-layer
//! RoPE/NoPE and sliding-window schedule, and the shared RoPE translation)
//! is [`LlamaFamilyConfig::architecture`], documented in the `architecture`
//! module together with the field translation table.
//!
//! # Defaults by `model_type`
//!
//! | field | llama | mistral | qwen2 | qwen3 | smollm3 | olmo2 |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | `vocab_size` | 32000 | 32000 | 151936 | 151936 | 128256 | 50304 |
//! | `hidden_size` | 4096 | 4096 | 4096 | 4096 | 2048 | 4096 |
//! | `intermediate_size` | 11008 | 14336 | 22016 | 22016 | 11008 | 11008 |
//! | `num_hidden_layers` | 32 | 32 | 32 | 32 | 36 | 32 |
//! | `num_attention_heads` | 32 | 32 | 32 | 32 | 16 | 32 |
//! | `num_key_value_heads` | heads | 8 | 32 | 32 | 4 | heads |
//! | `head_dim` | hidden / heads | hidden / heads | hidden / heads | 128 | hidden / heads | hidden / heads |
//! | `max_position_embeddings` | 2048 | 131072 | 32768 | 32768 | 32768 | 2048 |
//! | `rms_norm_eps` | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 1e-5 |
//! | `tie_word_embeddings` | false | false | false | false | true | false |
//! | default `rope_theta` | 1e4 | 1e4 | 1e4 | 1e4 | 2e6 | 1e4 |
//! | `sliding_window` | — | 4096 | 4096 | 4096 | none | — |
//! | `max_window_layers` | — | — | 28 | 28 | — | — |
//! | `no_rope_layer_interval` | — | — | — | — | 4 | — |
//!
//! "heads" means `num_attention_heads`; "—" marks a field the reference does
//! not define for that type. Qwen2 has no `head_dim` constructor argument, but
//! its attention reads a serialized `head_dim` attribute when present.
//!
//! # Fields outside the schema
//!
//! Keys that none of the six reference implementations read for inference
//! (`architectures`, `torch_dtype`/`dtype`, `transformers_version`,
//! `_name_or_path`, `use_cache`, `pretraining_tp`, generation kwargs, and so
//! on) are ignored, exactly as Transformers ignores them. The one loader
//! directive that changes the weight format, `quantization_config`, is kept
//! and rejected by validation, because these graphs load dense weights only.

use serde::{Deserialize, Serialize};

use super::architecture::LlamaArchitecture;
use crate::models::config::{
    ConfigError, ModelConfig, RopeParameters, TokenIds, Validate, token_ids,
};

/// Family name used for configuration errors that precede `model_type` resolution.
pub(super) const FAMILY: &str = "llama";

/// The Hugging Face `model_type` values this family executes.
///
/// Each variant names the pinned Transformers 5.2.0 reference whose defaults
/// and translation rules apply. Variants never reach the numerical graph: the
/// model reads only the flags that [`LlamaFamilyConfig::architecture`]
/// derives, so shared operations stay free of model-name branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlamaVariant {
    /// `llama`: Llama 2, Llama 3.x, SmolLM, and other Llama-layout decoders.
    Llama,
    /// `mistral`: Llama layout with an optional sliding window on every layer.
    Mistral,
    /// `qwen2`: Qwen2/Qwen2.5 with biased query, key, and value projections.
    Qwen2,
    /// `qwen3`: Qwen3 dense with per-head query/key RMSNorm.
    Qwen3,
    /// `smollm3`: SmolLM3 with periodic NoPE (no rotary) layers.
    #[serde(rename = "smollm3")]
    SmolLm3,
    /// `olmo2`: OLMo 2 with post-sublayer and full-width query/key RMSNorm.
    #[serde(rename = "olmo2")]
    Olmo2,
}

impl LlamaVariant {
    /// Every supported variant, in registry order.
    pub const ALL: [Self; 6] = [
        Self::Llama,
        Self::Mistral,
        Self::Qwen2,
        Self::Qwen3,
        Self::SmolLm3,
        Self::Olmo2,
    ];

    /// Every accepted `model_type`, in [`Self::ALL`] order.
    pub const MODEL_TYPES: [&'static str; 6] =
        ["llama", "mistral", "qwen2", "qwen3", "smollm3", "olmo2"];

    /// Returns the exact Hugging Face `model_type` discriminator.
    #[must_use]
    pub const fn model_type(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::SmolLm3 => "smollm3",
            Self::Olmo2 => "olmo2",
        }
    }

    /// Returns the variant whose `model_type` is exactly `model_type`.
    #[must_use]
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|variant| variant.model_type() == model_type)
    }

    /// Returns the pinned reference defaults for absent fields.
    const fn defaults(self) -> Defaults {
        /// Llama's constructor defaults, the base of most other columns.
        const LLAMA: Defaults = Defaults {
            vocab_size: 32_000,
            hidden_size: 4096,
            intermediate_size: 11_008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            head_dim: None,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            default_rope_theta: 10_000.0,
            sliding_window: None,
            max_window_layers: None,
            no_rope_layer_interval: None,
            pad_token_id: None,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
        };
        /// Qwen2's constructor defaults, shared with Qwen3 except `head_dim`.
        const QWEN2: Defaults = Defaults {
            vocab_size: 151_936,
            intermediate_size: 22_016,
            num_key_value_heads: Some(32),
            max_position_embeddings: 32_768,
            sliding_window: Some(4096),
            max_window_layers: Some(28),
            bos_token_id: None,
            eos_token_id: None,
            ..LLAMA
        };
        match self {
            Self::Llama => LLAMA,
            Self::Mistral => Defaults {
                intermediate_size: 14_336,
                num_key_value_heads: Some(8),
                max_position_embeddings: 131_072,
                sliding_window: Some(4096),
                ..LLAMA
            },
            Self::Qwen2 => QWEN2,
            Self::Qwen3 => Defaults {
                head_dim: Some(128),
                ..QWEN2
            },
            Self::SmolLm3 => Defaults {
                vocab_size: 128_256,
                hidden_size: 2048,
                num_hidden_layers: 36,
                num_attention_heads: 16,
                num_key_value_heads: Some(4),
                max_position_embeddings: 32_768,
                tie_word_embeddings: true,
                default_rope_theta: 2_000_000.0,
                no_rope_layer_interval: Some(4),
                pad_token_id: Some(128_004),
                bos_token_id: Some(128_000),
                eos_token_id: Some(128_001),
                ..LLAMA
            },
            Self::Olmo2 => Defaults {
                vocab_size: 50_304,
                rms_norm_eps: 1e-5,
                pad_token_id: Some(1),
                bos_token_id: None,
                eos_token_id: Some(50_279),
                ..LLAMA
            },
        }
    }
}

/// Constructor defaults of one reference configuration class.
#[derive(Debug, Clone, Copy)]
struct Defaults {
    /// Default vocabulary size.
    vocab_size: usize,
    /// Default residual width.
    hidden_size: usize,
    /// Default SwiGLU intermediate width.
    intermediate_size: usize,
    /// Default decoder depth.
    num_hidden_layers: usize,
    /// Default query-head count.
    num_attention_heads: usize,
    /// Default KV-head count; `None` means "equal to the query heads".
    num_key_value_heads: Option<usize>,
    /// Default head width; `None` means `hidden_size / num_attention_heads`.
    head_dim: Option<usize>,
    /// Default trained context.
    max_position_embeddings: usize,
    /// Default RMSNorm epsilon.
    rms_norm_eps: f64,
    /// Default embedding/head tying.
    tie_word_embeddings: bool,
    /// The class's `default_theta`, used when no `rope_theta` is serialized.
    default_rope_theta: f64,
    /// Default sliding window, before any `use_sliding_window` gate.
    sliding_window: Option<usize>,
    /// Default first sliding layer of the Qwen schedule.
    max_window_layers: Option<usize>,
    /// Default SmolLM3 NoPE period.
    no_rope_layer_interval: Option<usize>,
    /// Default padding token.
    pad_token_id: Option<usize>,
    /// Default beginning-of-sequence token.
    bos_token_id: Option<usize>,
    /// Default end-of-sequence token.
    eos_token_id: Option<usize>,
}

/// A Llama-family `config.json` with every reference default made explicit.
///
/// Construct it with [`ModelConfig::from_json_str`] (which also validates),
/// with `serde_json`, or with [`LlamaFamilyConfig::new`] and struct update
/// syntax. [`Self::architecture`] translates it into the executable topology
/// and is the single source of validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "SerializedLlamaFamilyConfig")]
pub struct LlamaFamilyConfig {
    /// Hugging Face architecture discriminator; one of [`LlamaVariant::MODEL_TYPES`].
    pub model_type: String,
    /// Vocabulary size: rows of the embedding table and of the LM head.
    pub vocab_size: usize,
    /// Residual-stream width `d`.
    pub hidden_size: usize,
    /// SwiGLU intermediate width.
    pub intermediate_size: usize,
    /// Number of decoder blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads; divides `num_attention_heads`.
    pub num_key_value_heads: usize,
    /// Width of every query, key, and value head.
    pub head_dim: usize,
    /// Gate nonlinearity name, e.g. `"silu"`.
    pub hidden_act: String,
    /// Trained context length; positions must lie below it.
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon shared by every normalization.
    pub rms_norm_eps: f64,
    /// Initialization standard deviation, retained for round trips only.
    pub initializer_range: f64,
    /// Whether the LM head reuses the embedding table.
    pub tie_word_embeddings: bool,
    /// Nested RoPE parameters (current `rope_parameters` or legacy `rope_scaling`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_parameters: Option<RopeParameters>,
    /// Legacy top-level RoPE base, a fallback for the nested value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<f64>,
    /// The `model_type`'s default RoPE base when no theta is serialized.
    pub default_rope_theta: f64,
    /// Legacy top-level partial rotary factor; these families rotate whole heads.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f64>,
    /// Serialized attention-bias flag; `None` when the checkpoint omits it.
    ///
    /// Its meaning depends on the variant (see the architecture translation
    /// table): all four projections for Llama, Qwen3, SmolLM3, and OLMo 2,
    /// never read for Mistral (no bias) and Qwen2 (query/key/value bias).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention_bias: Option<bool>,
    /// Serialized MLP-bias flag; read by Llama and SmolLM3 only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mlp_bias: Option<bool>,
    /// Attention-probability dropout; must be zero (dropout is not implemented).
    pub attention_dropout: f64,
    /// Qwen/SmolLM3 gate enabling the sliding window.
    pub use_sliding_window: bool,
    /// Sliding-window width in tokens, when the variant defines one.
    pub sliding_window: Option<usize>,
    /// First Qwen layer that uses the sliding window when `layer_types` is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_window_layers: Option<usize>,
    /// Explicit per-layer attention kinds (`full_attention`/`sliding_attention`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    /// SmolLM3 per-layer rotary flags: `1` applies RoPE, `0` is a NoPE layer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_rope_layers: Option<Vec<usize>>,
    /// SmolLM3 NoPE period used when `no_rope_layers` is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_rope_layer_interval: Option<usize>,
    /// Optional padding token ID.
    pub pad_token_id: Option<TokenIds>,
    /// Optional beginning-of-sequence token ID.
    pub bos_token_id: Option<TokenIds>,
    /// One or more end-of-sequence token IDs.
    pub eos_token_id: Option<TokenIds>,
    /// A packed-weight directive; present values are rejected by validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_config: Option<serde_json::Value>,
}

impl LlamaFamilyConfig {
    /// Returns the reference default configuration of `variant`.
    ///
    /// Tiny test and experiment configurations start here and override
    /// fields with struct update syntax.
    #[must_use]
    pub fn new(variant: LlamaVariant) -> Self {
        SerializedLlamaFamilyConfig {
            model_type: variant.model_type().into(),
            ..SerializedLlamaFamilyConfig::default()
        }
        .resolve()
    }

    /// Returns the variant named by `model_type`, if it is supported.
    #[must_use]
    pub fn variant(&self) -> Option<LlamaVariant> {
        LlamaVariant::from_model_type(&self.model_type)
    }

    /// Validates the configuration and derives its executable topology.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] naming the offending field for an unknown
    /// `model_type`, an impossible geometry, a field whose value the variant's
    /// reference contradicts, an unsupported activation, RoPE algorithm, or
    /// dropout, a malformed layer schedule, or a token ID outside the
    /// vocabulary.
    pub fn architecture(&self) -> Result<LlamaArchitecture, ConfigError> {
        LlamaArchitecture::from_config(self)
    }

    /// Checks the generation token IDs against the vocabulary.
    pub(super) fn validate_token_ids(&self, family: &'static str) -> Result<(), ConfigError> {
        for (field, ids) in [
            ("pad_token_id", &self.pad_token_id),
            ("bos_token_id", &self.bos_token_id),
            ("eos_token_id", &self.eos_token_id),
        ] {
            token_ids(family, field, ids.as_ref(), self.vocab_size)?;
        }
        Ok(())
    }
}

impl Validate for LlamaFamilyConfig {
    /// Validates by deriving the complete executable architecture.
    fn validate(&self) -> Result<(), ConfigError> {
        self.architecture().map(drop)
    }
}

impl ModelConfig for LlamaFamilyConfig {
    /// Diagnostic family name used by generic configuration parsing.
    const FAMILY: &'static str = FAMILY;
    /// Checkpoint `model_type` values accepted by this schema.
    const MODEL_TYPES: &'static [&'static str] = &LlamaVariant::MODEL_TYPES;
}

/// The literal `config.json` document, before per-`model_type` defaults.
///
/// Every field is optional so absence is observable; [`Self::resolve`]
/// applies the defaults of the document's own `model_type`. Unknown keys are
/// ignored, as Transformers ignores keys its modeling code never reads.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct SerializedLlamaFamilyConfig {
    /// See [`LlamaFamilyConfig::model_type`].
    model_type: String,
    /// See [`LlamaFamilyConfig::vocab_size`].
    vocab_size: Option<usize>,
    /// See [`LlamaFamilyConfig::hidden_size`].
    hidden_size: Option<usize>,
    /// See [`LlamaFamilyConfig::intermediate_size`].
    intermediate_size: Option<usize>,
    /// See [`LlamaFamilyConfig::num_hidden_layers`].
    num_hidden_layers: Option<usize>,
    /// See [`LlamaFamilyConfig::num_attention_heads`].
    num_attention_heads: Option<usize>,
    /// See [`LlamaFamilyConfig::num_key_value_heads`]; JSON `null` means "equal to the query heads".
    #[serde(deserialize_with = "present")]
    num_key_value_heads: Option<Option<usize>>,
    /// See [`LlamaFamilyConfig::head_dim`]; JSON `null` means `hidden_size / num_attention_heads`.
    #[serde(deserialize_with = "present")]
    head_dim: Option<Option<usize>>,
    /// See [`LlamaFamilyConfig::hidden_act`].
    hidden_act: Option<String>,
    /// See [`LlamaFamilyConfig::max_position_embeddings`].
    max_position_embeddings: Option<usize>,
    /// See [`LlamaFamilyConfig::rms_norm_eps`].
    rms_norm_eps: Option<f64>,
    /// See [`LlamaFamilyConfig::initializer_range`].
    initializer_range: Option<f64>,
    /// See [`LlamaFamilyConfig::tie_word_embeddings`].
    tie_word_embeddings: Option<bool>,
    /// See [`LlamaFamilyConfig::rope_parameters`]; also read from `rope_scaling`.
    #[serde(alias = "rope_scaling")]
    rope_parameters: Option<RopeParameters>,
    /// See [`LlamaFamilyConfig::rope_theta`].
    rope_theta: Option<f64>,
    /// See [`LlamaFamilyConfig::default_rope_theta`] (written by serialization).
    default_rope_theta: Option<f64>,
    /// See [`LlamaFamilyConfig::partial_rotary_factor`].
    partial_rotary_factor: Option<f64>,
    /// See [`LlamaFamilyConfig::attention_bias`].
    attention_bias: Option<bool>,
    /// See [`LlamaFamilyConfig::mlp_bias`].
    mlp_bias: Option<bool>,
    /// See [`LlamaFamilyConfig::attention_dropout`].
    attention_dropout: Option<f64>,
    /// See [`LlamaFamilyConfig::use_sliding_window`].
    use_sliding_window: Option<bool>,
    /// See [`LlamaFamilyConfig::sliding_window`]; JSON `null` disables it.
    #[serde(deserialize_with = "present")]
    sliding_window: Option<Option<usize>>,
    /// See [`LlamaFamilyConfig::max_window_layers`].
    max_window_layers: Option<usize>,
    /// See [`LlamaFamilyConfig::layer_types`].
    layer_types: Option<Vec<String>>,
    /// See [`LlamaFamilyConfig::no_rope_layers`].
    no_rope_layers: Option<Vec<usize>>,
    /// See [`LlamaFamilyConfig::no_rope_layer_interval`].
    no_rope_layer_interval: Option<usize>,
    /// See [`LlamaFamilyConfig::pad_token_id`]; JSON `null` means none.
    #[serde(deserialize_with = "present")]
    pad_token_id: Option<Option<TokenIds>>,
    /// See [`LlamaFamilyConfig::bos_token_id`]; JSON `null` means none.
    #[serde(deserialize_with = "present")]
    bos_token_id: Option<Option<TokenIds>>,
    /// See [`LlamaFamilyConfig::eos_token_id`]; JSON `null` means none.
    #[serde(deserialize_with = "present")]
    eos_token_id: Option<Option<TokenIds>>,
    /// See [`LlamaFamilyConfig::quantization_config`].
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

impl SerializedLlamaFamilyConfig {
    /// Fills absent fields from the reference defaults of this `model_type`.
    ///
    /// An unsupported `model_type` keeps Llama defaults; validation then
    /// rejects it with a field-specific error instead of a JSON error.
    fn resolve(self) -> LlamaFamilyConfig {
        let defaults = LlamaVariant::from_model_type(&self.model_type)
            .unwrap_or(LlamaVariant::Llama)
            .defaults();
        let hidden_size = self.hidden_size.unwrap_or(defaults.hidden_size);
        let num_attention_heads = self
            .num_attention_heads
            .unwrap_or(defaults.num_attention_heads);
        // An absent key takes the type's default; an explicit `null` (like an
        // absent key without a default) means "derive", as the constructors do.
        let head_dim = self
            .head_dim
            .unwrap_or(defaults.head_dim)
            .unwrap_or_else(|| hidden_size.checked_div(num_attention_heads).unwrap_or(0));
        let num_key_value_heads = self
            .num_key_value_heads
            .unwrap_or(defaults.num_key_value_heads)
            .unwrap_or(num_attention_heads);
        let token = |serialized: Option<Option<TokenIds>>, default: Option<usize>| {
            serialized.unwrap_or(default.map(TokenIds::One))
        };
        LlamaFamilyConfig {
            vocab_size: self.vocab_size.unwrap_or(defaults.vocab_size),
            hidden_size,
            intermediate_size: self.intermediate_size.unwrap_or(defaults.intermediate_size),
            num_hidden_layers: self.num_hidden_layers.unwrap_or(defaults.num_hidden_layers),
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_act: self.hidden_act.unwrap_or_else(|| "silu".into()),
            max_position_embeddings: self
                .max_position_embeddings
                .unwrap_or(defaults.max_position_embeddings),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(defaults.rms_norm_eps),
            initializer_range: self.initializer_range.unwrap_or(0.02),
            tie_word_embeddings: self
                .tie_word_embeddings
                .unwrap_or(defaults.tie_word_embeddings),
            rope_parameters: self.rope_parameters,
            rope_theta: self.rope_theta,
            default_rope_theta: self
                .default_rope_theta
                .unwrap_or(defaults.default_rope_theta),
            partial_rotary_factor: self.partial_rotary_factor,
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            attention_dropout: self.attention_dropout.unwrap_or(0.0),
            use_sliding_window: self.use_sliding_window.unwrap_or(false),
            sliding_window: self.sliding_window.unwrap_or(defaults.sliding_window),
            max_window_layers: self.max_window_layers.or(defaults.max_window_layers),
            layer_types: self.layer_types,
            no_rope_layers: self.no_rope_layers,
            no_rope_layer_interval: self
                .no_rope_layer_interval
                .or(defaults.no_rope_layer_interval),
            pad_token_id: token(self.pad_token_id, defaults.pad_token_id),
            bos_token_id: token(self.bos_token_id, defaults.bos_token_id),
            eos_token_id: token(self.eos_token_id, defaults.eos_token_id),
            quantization_config: self.quantization_config.filter(|value| !value.is_null()),
            model_type: self.model_type,
        }
    }
}

impl From<SerializedLlamaFamilyConfig> for LlamaFamilyConfig {
    /// Applies the per-`model_type` defaults; validation reports every semantic problem.
    fn from(serialized: SerializedLlamaFamilyConfig) -> Self {
        serialized.resolve()
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
