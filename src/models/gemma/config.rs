//! One serialized configuration schema for Gemma 2 and Gemma 3 text decoders.
//!
//! Gemma 2 (`model_type: "gemma2"`) and the Gemma 3 text tower
//! (`"gemma3_text"`) share one Hugging Face field vocabulary but disagree on
//! defaults and on which fields their reference code reads. [`GemmaConfig`]
//! is the superset schema: deserialization fills every absent field with the
//! default of the checkpoint's own `model_type` (the pinned Transformers
//! 5.2.0 `configuration_gemma2.py` / `configuration_gemma3.py` constructor),
//! so a parsed config states every value explicitly.
//!
//! A Gemma 3 *multimodal* checkpoint (`model_type: "gemma3"`) nests this
//! text configuration under `text_config`; [`ModelConfig::from_json_str`]
//! reads it from there, and the vision tower's fields are never parsed (see
//! the checkpoint contract for the explicit omission of its tensors).
//!
//! The configuration describes *what the checkpoint says*. Turning it into
//! an executable topology (layer schedule, RoPE sets, attention scale and
//! soft-caps) is [`GemmaConfig::architecture`], documented in the
//! `architecture` module together with the field translation table.
//!
//! # Defaults by `model_type`
//!
//! | field | gemma2 | gemma3_text |
//! | --- | --- | --- |
//! | `vocab_size` | 256000 | 262208 |
//! | `hidden_size` | 2304 | 2304 |
//! | `intermediate_size` | 9216 | 9216 |
//! | `num_hidden_layers` | 26 | 26 |
//! | `num_attention_heads` | 8 | 8 |
//! | `num_key_value_heads` | 4 | 4 |
//! | `head_dim` | 256 | 256 |
//! | `hidden_activation` | `gelu_pytorch_tanh` | `gelu_pytorch_tanh` |
//! | `max_position_embeddings` | 8192 | 131072 |
//! | `rms_norm_eps` | 1e-6 | 1e-6 |
//! | `query_pre_attn_scalar` | 256 | 256 |
//! | `sliding_window` | 4096 | 4096 |
//! | `attn_logit_softcapping` | 50.0 | `null` |
//! | `final_logit_softcapping` | 30.0 | `null` |
//! | layer schedule without `layer_types` | alternate, from sliding | `sliding_window_pattern` (6): every 6th layer global |
//! | default `rope_theta` | 1e4 | global 1e6, local (`rope_local_base_freq`) 1e4 |
//! | `pad`/`eos`/`bos_token_id` | 0 / 1 / 2 | 0 / 1 / 2 |
//! | `tie_word_embeddings` | true | true |
//!
//! # Null versus absent
//!
//! An absent key takes the type's default. JSON `null` is meaningful for the
//! soft-caps (disabled), the token IDs (none), `layer_types` (derive), and
//! `use_bidirectional_attention` (false). For the integer geometry
//! (`num_key_value_heads`, `head_dim`, `sliding_window`) and for
//! `query_pre_attn_scalar`, the reference stores `None` and then fails with a
//! Python `TypeError` at model construction; this schema rejects such a
//! `null` at parse time instead.
//!
//! # Per-layer-type RoPE mappings written by Transformers 5.2
//!
//! When `layer_types` uses only one attention kind (all global, or all
//! sliding, as in a Gemma 3 config with fewer layers than its pattern), 5.2's
//! `standardize_rope_params` finds mapping keys outside `layer_types`,
//! treats the mapping as flat, and `save_pretrained` writes
//! `"rope_type": "default"` and `"rope_theta": null` beside the
//! `sliding_attention`/`full_attention` objects. The rotary embedding reads
//! only the per-type objects, so these two inert artifacts are dropped
//! before the mapping is classified. Any other flat key beside per-type
//! objects (a non-default `rope_type`, a numeric `rope_theta`, a `factor`)
//! is ambiguous and still rejected.
//!
//! # Fields outside the schema
//!
//! Keys the references never read for text inference (`architectures`,
//! `dtype`/`torch_dtype`, `transformers_version`, `use_cache`,
//! `cache_implementation`, Gemma 2's legacy `hidden_act` mirror of
//! `hidden_activation`, Gemma 3's serialized `_sliding_window_pattern`, and
//! so on) are ignored, exactly as Transformers ignores them. The one loader
//! directive that changes the weight format, `quantization_config`, is kept
//! and rejected by validation.

use serde::{Deserialize, Serialize};

use super::architecture::GemmaArchitecture;
use crate::{
    checkpoint::{RangeHazard, TensorDType},
    models::config::{
        ConfigError, ModelConfig, RopeParameterSet, RopeParameters, TokenIds, Validate, token_ids,
    },
};

/// Family name used for configuration errors that precede `model_type` resolution.
pub(super) const FAMILY: &str = "gemma";

/// Gemma 3's declared F16 overflow, the hazard of [`GemmaVariant::Gemma3Text`].
///
/// One channel of the residual stream grows past F16's largest finite value
/// in the reference implementation itself. In gemma-3-4b-it that is channel
/// 443, whose post-sublayer norm gains `|1 + w|` reach 1,369 (block 33's
/// post-feed-forward norm), meeting the BOS position's massive activation.
/// Every branch tensor fits F16; the sum does not. The overflow is therefore
/// a property of the published weights, and no faithful F16 evaluation is
/// finite. The declaration is family-wide (a default policy):
/// it also covers sizes and fine-tunes nobody measured, and the evidence says
/// which were measured.
const GEMMA3_F16_OVERFLOW: RangeHazard = RangeHazard::new(
    "Gemma 3 (model_type gemma3 or gemma3_text)",
    TensorDType::F16,
    "measured on google/gemma-3-4b-it (revision 093f9f38): the residual addition after block \
     5's MLP reaches 73,509.6 at the BOS position, channel 443 (51,941.5 + 21,568.1 in F32), \
     beyond F16's largest finite value 65,504, and Transformers 5.2.0's CPU and MPS F16 graphs \
     and this crate's Metal and Flex F16 graphs all overflow there and return non-finite \
     logits; gemma-3-12b-it and gemma-3-27b-it first exceed 65,504 at the BOS position of \
     blocks 6 and 8 (83,968 and 83,456 in Transformers' CPU BF16 graphs), and their \
     Transformers CPU F16 graphs are non-finite from there. Unmeasured: other sizes (such as \
     1B and 270M) and fine-tunes, which this family-wide default refuses too. Records: \
     acceptance/measurements/half-precision-envelope-20261018.json and decision (d) of \
     acceptance/measurements/half-precision-followups-20261017.json",
);

/// The Hugging Face text `model_type` values this family executes.
///
/// Each variant names the pinned Transformers 5.2.0 reference whose defaults
/// and translation rules apply. Variants never reach the numerical graph:
/// the model reads only the flags that [`GemmaConfig::architecture`]
/// derives, so shared operations stay free of model-name branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GemmaVariant {
    /// `gemma2`: Gemma 2 with attention and final logit soft-capping.
    #[serde(rename = "gemma2")]
    Gemma2,
    /// `gemma3_text`: the Gemma 3 text decoder with Q/K norms and local/global RoPE.
    #[serde(rename = "gemma3_text")]
    Gemma3Text,
}

impl GemmaVariant {
    /// Every supported variant, in registry order.
    pub const ALL: [Self; 2] = [Self::Gemma2, Self::Gemma3Text];

    /// Every accepted `model_type`, in [`Self::ALL`] order.
    pub const MODEL_TYPES: [&'static str; 2] = ["gemma2", "gemma3_text"];

    /// Returns the exact Hugging Face `model_type` discriminator.
    #[must_use]
    pub const fn model_type(self) -> &'static str {
        match self {
            Self::Gemma2 => "gemma2",
            Self::Gemma3Text => "gemma3_text",
        }
    }

    /// Returns the variant whose `model_type` is exactly `model_type`.
    #[must_use]
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|variant| variant.model_type() == model_type)
    }

    /// Returns the numeric-range hazards declared for this variant's checkpoints.
    ///
    /// Gemma 3 (`gemma3_text`, which a multimodal `gemma3` checkpoint's
    /// `text_config` also names) declares F16: its published checkpoints
    /// overflow F16 in Transformers too (see the evidence the declaration
    /// carries). Gemma 2 declares none, because no Gemma 2 checkpoint has
    /// been measured in F16; that is an absence of evidence, not a
    /// measurement. The checkpoint contract's mapping plan carries these
    /// declarations, so the shared planner
    /// ([`crate::checkpoint::build_load_plan`]) refuses a matching plan unless
    /// [`LoadOptions::range_hazards`](crate::checkpoint::LoadOptions::range_hazards)
    /// accepts it ([`crate::checkpoint::check_range_hazards`]). The
    /// declaration certifies no packed path, because the automatic loader
    /// does not accept a packed sidecar for Gemma, so the refusal does not
    /// offer packed weights.
    #[must_use]
    pub const fn range_hazards(self) -> &'static [RangeHazard] {
        match self {
            Self::Gemma2 => &[],
            Self::Gemma3Text => &[GEMMA3_F16_OVERFLOW],
        }
    }

    /// Returns the pinned reference defaults for absent fields.
    const fn defaults(self) -> Defaults {
        /// Gemma 2's constructor defaults, the base of the Gemma 3 column.
        const GEMMA2: Defaults = Defaults {
            vocab_size: 256_000,
            max_position_embeddings: 8192,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
        };
        match self {
            Self::Gemma2 => GEMMA2,
            Self::Gemma3Text => Defaults {
                vocab_size: 262_208,
                max_position_embeddings: 131_072,
                attn_logit_softcapping: None,
                final_logit_softcapping: None,
            },
        }
    }
}

/// Constructor defaults that differ between the two references.
///
/// Every other default (widths, heads, epsilon, scalar, window, token IDs)
/// is shared and lives in [`SerializedGemmaConfig::resolve`].
#[derive(Debug, Clone, Copy)]
struct Defaults {
    /// Default vocabulary size.
    vocab_size: usize,
    /// Default trained context.
    max_position_embeddings: usize,
    /// Default attention-logit soft-cap.
    attn_logit_softcapping: Option<f64>,
    /// Default final-logit soft-cap.
    final_logit_softcapping: Option<f64>,
}

/// A Gemma 2 or Gemma 3 text `config.json` with every reference default made explicit.
///
/// Construct it with [`ModelConfig::from_json_str`] (which also validates
/// and reads a multimodal wrapper's `text_config`), with `serde_json`, or
/// with [`GemmaConfig::new`] and struct update syntax.
/// [`Self::architecture`] translates it into the executable topology and is
/// the single source of validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SerializedGemmaConfig")]
pub struct GemmaConfig {
    /// Hugging Face text discriminator; one of [`GemmaVariant::MODEL_TYPES`].
    pub model_type: String,
    /// Vocabulary size: rows of the embedding table and of the LM head.
    pub vocab_size: usize,
    /// Residual-stream width `d`; the embedding is scaled by `sqrt(d)`.
    pub hidden_size: usize,
    /// GeGLU intermediate width.
    pub intermediate_size: usize,
    /// Number of decoder blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads; divides `num_attention_heads`.
    pub num_key_value_heads: usize,
    /// Width of every query, key, and value head.
    pub head_dim: usize,
    /// Gate nonlinearity name, `"gelu_pytorch_tanh"` in every release.
    pub hidden_activation: String,
    /// Trained context length; positions must lie below it.
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon shared by every normalization, Q/K norms included.
    pub rms_norm_eps: f64,
    /// Initialization standard deviation, retained for round trips only.
    pub initializer_range: f64,
    /// Whether the LM head reuses the embedding table.
    pub tie_word_embeddings: bool,
    /// Whether all four attention projections carry additive biases.
    pub attention_bias: bool,
    /// Attention-probability dropout; must be zero (dropout is not implemented).
    pub attention_dropout: f64,
    /// Attention scores are scaled by `query_pre_attn_scalar^(-1/2)`.
    pub query_pre_attn_scalar: f64,
    /// Sliding-window width in tokens of every `sliding_attention` layer.
    pub sliding_window: usize,
    /// Explicit per-layer attention kinds (`full_attention`/`sliding_attention`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    /// Gemma 3 global-layer period used when `layer_types` is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sliding_window_pattern: Option<usize>,
    /// `c` in `c · tanh(s / c)` applied to attention scores; `None` disables it.
    pub attn_logit_softcapping: Option<f64>,
    /// `c` in `c · tanh(z / c)` applied to logits; `None` disables it.
    pub final_logit_softcapping: Option<f64>,
    /// Nested RoPE parameters: one mapping, or one per layer type (Gemma 3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_parameters: Option<RopeParameterSet>,
    /// Legacy nested RoPE scaling; Gemma 3 applies it to global layers only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<RopeParameters>,
    /// Legacy top-level RoPE base: every layer (Gemma 2) or global layers (Gemma 3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<f64>,
    /// Legacy Gemma 3 RoPE base of sliding (local) layers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_local_base_freq: Option<f64>,
    /// Whether queries attend in both directions (EmbeddingGemma); must be false.
    pub use_bidirectional_attention: bool,
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

impl GemmaConfig {
    /// Returns the reference default configuration of `variant`.
    ///
    /// Tiny test and experiment configurations start here and override
    /// fields with struct update syntax.
    #[must_use]
    pub fn new(variant: GemmaVariant) -> Self {
        SerializedGemmaConfig {
            model_type: variant.model_type().into(),
            ..SerializedGemmaConfig::default()
        }
        .resolve()
    }

    /// Returns the variant named by `model_type`, if it is supported.
    #[must_use]
    pub fn variant(&self) -> Option<GemmaVariant> {
        GemmaVariant::from_model_type(&self.model_type)
    }

    /// Validates the configuration and derives its executable topology.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] naming the offending field for an unknown
    /// `model_type`, an impossible geometry, a field the variant's reference
    /// never reads or contradicts, an unsupported activation, RoPE
    /// algorithm, bidirectional attention, or dropout, a malformed layer
    /// schedule, or a token ID outside the vocabulary.
    pub fn architecture(&self) -> Result<GemmaArchitecture, ConfigError> {
        GemmaArchitecture::from_config(self)
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

impl Validate for GemmaConfig {
    /// Validates by deriving the complete executable architecture.
    fn validate(&self) -> Result<(), ConfigError> {
        self.architecture().map(drop)
    }
}

impl ModelConfig for GemmaConfig {
    /// Diagnostic family name used by generic configuration parsing.
    const FAMILY: &'static str = FAMILY;
    /// Checkpoint text `model_type` values accepted by this schema.
    const MODEL_TYPES: &'static [&'static str] = &GemmaVariant::MODEL_TYPES;
}

/// The literal `config.json` document, before per-`model_type` defaults.
///
/// Every field is optional so absence is observable; [`Self::resolve`]
/// applies the defaults of the document's own `model_type`. Unknown keys are
/// ignored, as Transformers ignores keys its modeling code never reads.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct SerializedGemmaConfig {
    /// See [`GemmaConfig::model_type`].
    model_type: String,
    /// See [`GemmaConfig::vocab_size`].
    vocab_size: Option<usize>,
    /// See [`GemmaConfig::hidden_size`].
    hidden_size: Option<usize>,
    /// See [`GemmaConfig::intermediate_size`].
    intermediate_size: Option<usize>,
    /// See [`GemmaConfig::num_hidden_layers`].
    num_hidden_layers: Option<usize>,
    /// See [`GemmaConfig::num_attention_heads`].
    num_attention_heads: Option<usize>,
    /// See [`GemmaConfig::num_key_value_heads`]; `null` is rejected.
    #[serde(deserialize_with = "present")]
    num_key_value_heads: Option<Option<usize>>,
    /// See [`GemmaConfig::head_dim`]; `null` is rejected.
    #[serde(deserialize_with = "present")]
    head_dim: Option<Option<usize>>,
    /// See [`GemmaConfig::hidden_activation`].
    hidden_activation: Option<String>,
    /// See [`GemmaConfig::max_position_embeddings`].
    max_position_embeddings: Option<usize>,
    /// See [`GemmaConfig::rms_norm_eps`].
    rms_norm_eps: Option<f64>,
    /// See [`GemmaConfig::initializer_range`].
    initializer_range: Option<f64>,
    /// See [`GemmaConfig::tie_word_embeddings`].
    tie_word_embeddings: Option<bool>,
    /// See [`GemmaConfig::attention_bias`].
    attention_bias: Option<bool>,
    /// See [`GemmaConfig::attention_dropout`].
    attention_dropout: Option<f64>,
    /// See [`GemmaConfig::query_pre_attn_scalar`]; `null` is rejected.
    #[serde(deserialize_with = "present")]
    query_pre_attn_scalar: Option<Option<f64>>,
    /// See [`GemmaConfig::sliding_window`]; `null` is rejected.
    #[serde(deserialize_with = "present")]
    sliding_window: Option<Option<usize>>,
    /// See [`GemmaConfig::layer_types`]; `null` derives the schedule.
    layer_types: Option<Vec<String>>,
    /// See [`GemmaConfig::sliding_window_pattern`].
    sliding_window_pattern: Option<usize>,
    /// See [`GemmaConfig::attn_logit_softcapping`]; `null` disables it.
    #[serde(deserialize_with = "present")]
    attn_logit_softcapping: Option<Option<f64>>,
    /// See [`GemmaConfig::final_logit_softcapping`]; `null` disables it.
    #[serde(deserialize_with = "present")]
    final_logit_softcapping: Option<Option<f64>>,
    /// See [`GemmaConfig::rope_parameters`]; 5.2's inert flat artifacts are dropped.
    #[serde(deserialize_with = "rope_parameters")]
    rope_parameters: Option<RopeParameterSet>,
    /// See [`GemmaConfig::rope_scaling`].
    rope_scaling: Option<RopeParameters>,
    /// See [`GemmaConfig::rope_theta`].
    rope_theta: Option<f64>,
    /// See [`GemmaConfig::rope_local_base_freq`].
    rope_local_base_freq: Option<f64>,
    /// See [`GemmaConfig::use_bidirectional_attention`]; `null` means false.
    use_bidirectional_attention: Option<bool>,
    /// See [`GemmaConfig::pad_token_id`]; `null` means none.
    #[serde(deserialize_with = "present")]
    pad_token_id: Option<Option<TokenIds>>,
    /// See [`GemmaConfig::bos_token_id`]; `null` means none.
    #[serde(deserialize_with = "present")]
    bos_token_id: Option<Option<TokenIds>>,
    /// See [`GemmaConfig::eos_token_id`]; `null` means none.
    #[serde(deserialize_with = "present")]
    eos_token_id: Option<Option<TokenIds>>,
    /// See [`GemmaConfig::quantization_config`].
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

/// Reads `rope_parameters`, dropping the inert keys Transformers 5.2 adds beside per-type objects.
///
/// See the module documentation: with per-layer-type objects present,
/// exactly `"rope_type": "default"` and `"rope_theta": null` are removed
/// before [`RopeParameterSet`] classifies the mapping, so every other flat
/// key still makes it a rejected mix. A JSON `null` means absent.
fn rope_parameters<'de, D>(deserializer: D) -> Result<Option<RopeParameterSet>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(mut value) = Option::<serde_json::Value>::deserialize(deserializer)? else {
        return Ok(None);
    };
    drop_inert_rope_artifacts(&mut value);
    RopeParameterSet::deserialize(value)
        .map(Some)
        .map_err(serde::de::Error::custom)
}

/// Removes 5.2's `"rope_type": "default"` and `"rope_theta": null` from a mapping with per-type objects.
///
/// A mapping without object values is flat and keeps every key; values
/// other than exactly these two are kept so classification rejects them.
fn drop_inert_rope_artifacts(value: &mut serde_json::Value) {
    let serde_json::Value::Object(map) = value else {
        return;
    };
    if !map.values().any(serde_json::Value::is_object) {
        return;
    }
    if map.get("rope_type").and_then(serde_json::Value::as_str) == Some("default") {
        map.remove("rope_type");
    }
    if map
        .get("rope_theta")
        .is_some_and(serde_json::Value::is_null)
    {
        map.remove("rope_theta");
    }
}

impl SerializedGemmaConfig {
    /// Rejects an explicit `null` in a field the reference cannot construct a model without.
    ///
    /// The reference stores such a `null` and then fails at model
    /// construction, so the schema reports it at parse time instead.
    fn reject_required_nulls(&self) -> Result<(), String> {
        let nulls = [
            (
                "num_key_value_heads",
                self.num_key_value_heads == Some(None),
            ),
            ("head_dim", self.head_dim == Some(None)),
            (
                "query_pre_attn_scalar",
                self.query_pre_attn_scalar == Some(None),
            ),
            ("sliding_window", self.sliding_window == Some(None)),
        ];
        match nulls.into_iter().find(|(_, null)| *null) {
            Some((field, _)) => Err(format!(
                "{FAMILY}: `{field}` must not be null; the reference cannot construct a model without it"
            )),
            None => Ok(()),
        }
    }

    /// Fills absent fields from the reference defaults of this `model_type`.
    ///
    /// An unsupported `model_type` keeps Gemma 2 defaults; validation then
    /// rejects it with a field-specific error instead of a JSON error.
    /// Callers reject required `null`s first ([`Self::reject_required_nulls`]);
    /// here a `null` reads as absent.
    fn resolve(self) -> GemmaConfig {
        let defaults = GemmaVariant::from_model_type(&self.model_type)
            .unwrap_or(GemmaVariant::Gemma2)
            .defaults();
        let token = |serialized: Option<Option<TokenIds>>, default: usize| {
            serialized.unwrap_or(Some(TokenIds::One(default)))
        };
        GemmaConfig {
            vocab_size: self.vocab_size.unwrap_or(defaults.vocab_size),
            hidden_size: self.hidden_size.unwrap_or(2304),
            intermediate_size: self.intermediate_size.unwrap_or(9216),
            num_hidden_layers: self.num_hidden_layers.unwrap_or(26),
            num_attention_heads: self.num_attention_heads.unwrap_or(8),
            num_key_value_heads: self.num_key_value_heads.flatten().unwrap_or(4),
            head_dim: self.head_dim.flatten().unwrap_or(256),
            hidden_activation: self
                .hidden_activation
                .unwrap_or_else(|| "gelu_pytorch_tanh".into()),
            max_position_embeddings: self
                .max_position_embeddings
                .unwrap_or(defaults.max_position_embeddings),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-6),
            initializer_range: self.initializer_range.unwrap_or(0.02),
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(true),
            attention_bias: self.attention_bias.unwrap_or(false),
            attention_dropout: self.attention_dropout.unwrap_or(0.0),
            query_pre_attn_scalar: self.query_pre_attn_scalar.flatten().unwrap_or(256.0),
            sliding_window: self.sliding_window.flatten().unwrap_or(4096),
            layer_types: self.layer_types,
            sliding_window_pattern: self.sliding_window_pattern,
            attn_logit_softcapping: self
                .attn_logit_softcapping
                .unwrap_or(defaults.attn_logit_softcapping),
            final_logit_softcapping: self
                .final_logit_softcapping
                .unwrap_or(defaults.final_logit_softcapping),
            rope_parameters: self.rope_parameters,
            rope_scaling: self.rope_scaling,
            rope_theta: self.rope_theta,
            rope_local_base_freq: self.rope_local_base_freq,
            use_bidirectional_attention: self.use_bidirectional_attention.unwrap_or(false),
            pad_token_id: token(self.pad_token_id, 0),
            bos_token_id: token(self.bos_token_id, 2),
            eos_token_id: token(self.eos_token_id, 1),
            quantization_config: self.quantization_config.filter(|value| !value.is_null()),
            model_type: self.model_type,
        }
    }
}

impl TryFrom<SerializedGemmaConfig> for GemmaConfig {
    /// A parse-time message naming the offending field.
    type Error = String;

    /// Rejects required `null`s and applies the per-`model_type` defaults;
    /// validation reports every semantic problem.
    fn try_from(serialized: SerializedGemmaConfig) -> Result<Self, Self::Error> {
        serialized.reject_required_nulls()?;
        Ok(serialized.resolve())
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
