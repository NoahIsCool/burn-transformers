//! Shared, backend-free model configuration machinery.
//!
//! Transformer checkpoints encode matrix shapes indirectly. For example, grouped
//! query attention is valid only when the number of query heads is an integer
//! multiple of the number of key/value heads, and RoPE rotates pairs of channels,
//! so its effective dimension must be even. The helpers in this module turn those
//! mathematical assumptions into errors at load time rather than tensor panics in
//! the first forward pass.
//!
//! The [`rope`] submodule translates the serialized Hugging Face RoPE schema
//! ([`RopeParameters`], or a per-layer-type [`RopeParameterSet`]) together with
//! the surrounding config fields ([`RopeContext`]) into the executable
//! [`crate::position::RopeConfig`], so every family shares one precedence policy.

pub mod rope;

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub use rope::{
    DEFAULT_ROPE_THETA, DynamicNtkContext, MropeHandling, PartialRotary, RopeContext,
    RopeParameterSet, RopeParametersByLayerType, SoftmaxMscale, UnusedRopeFields,
    YarnMscaleConvention, yarn_mscale,
};

/// A topology error associated with one checkpoint field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    /// Stable architecture-family name used in diagnostics.
    pub architecture: &'static str,
    /// Serialized field or relationship that failed validation.
    pub field: &'static str,
    /// Human-readable explanation of the violated invariant.
    pub reason: String,
}

impl ConfigError {
    /// Constructs a field-specific topology error.
    pub fn new(architecture: &'static str, field: &'static str, reason: impl Into<String>) -> Self {
        Self {
            architecture,
            field,
            reason: reason.into(),
        }
    }
}

impl Display for ConfigError {
    /// Formats errors as `architecture.field: reason` for concise loader messages.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}.{}: {}",
            self.architecture, self.field, self.reason
        )
    }
}

impl Error for ConfigError {}

/// Failure to decode a serialized configuration or validate its topology.
#[derive(Debug)]
pub enum ConfigLoadError {
    /// The input was not compatible JSON or a field had the wrong serialized type.
    Json(
        /// Parser diagnostic preserving the malformed field or source location.
        serde_json::Error,
    ),
    /// JSON decoding succeeded, but the described network cannot be wired safely.
    Validation(
        /// Field-specific topology invariant that prevented model construction.
        ConfigError,
    ),
}

impl Display for ConfigLoadError {
    /// Preserves the detailed source diagnostic for JSON and topology failures.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "invalid model config JSON: {error}"),
            Self::Validation(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for ConfigLoadError {
    /// Returns the underlying parser or validation error.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Validation(error) => Some(error),
        }
    }
}

impl From<serde_json::Error> for ConfigLoadError {
    /// Converts `serde_json` diagnostics without discarding line and column data.
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<ConfigError> for ConfigLoadError {
    /// Converts a validated topology failure into a loading failure.
    fn from(value: ConfigError) -> Self {
        Self::Validation(value)
    }
}

impl From<ConfigError> for crate::Error {
    /// Maps a topology failure to a configuration error at `architecture.field`.
    ///
    /// This is the same mapping used by [`crate::ArchitectureConfig::validate`].
    fn from(error: ConfigError) -> Self {
        crate::Error::configuration(
            format!("{}.{}", error.architecture, error.field),
            error.reason,
        )
    }
}

impl From<ConfigLoadError> for crate::Error {
    /// Converts a configuration loading failure so loaders can use `?`.
    ///
    /// Topology failures keep their `architecture.field` path. JSON failures
    /// keep their parser diagnostic, including line and column, under the
    /// field `config`.
    fn from(error: ConfigLoadError) -> Self {
        match error {
            ConfigLoadError::Validation(error) => error.into(),
            ConfigLoadError::Json(error) => {
                crate::Error::configuration("config", format!("invalid model config JSON: {error}"))
            }
        }
    }
}

/// Validation implemented by every model topology schema.
pub trait Validate {
    /// Checks all relationships needed to construct matrices and caches safely.
    fn validate(&self) -> Result<(), ConfigError>;
}

/// Common loading API for a model configuration.
pub trait ModelConfig: Validate + DeserializeOwned + Sized {
    /// Stable family name used in diagnostics and compatibility tables.
    const FAMILY: &'static str;
    /// Hugging Face `model_type` values accepted by this topology.
    const MODEL_TYPES: &'static [&'static str];

    /// Parses either a direct text config or a multimodal wrapper and validates it.
    fn from_json_str(json: &str) -> Result<Self, ConfigLoadError> {
        parse_text_config(json)
    }
}

impl<T> crate::ArchitectureConfig for T
where
    T: ModelConfig + Clone + core::fmt::Debug + Send + Sync,
{
    /// Accepted source-family discriminators from the concrete topology schema.
    const MODEL_TYPES: &'static [&'static str] = T::MODEL_TYPES;

    /// Reuses the family schema's field-level validation in the runtime API.
    fn validate(&self) -> crate::Result<()> {
        Validate::validate(self).map_err(crate::Error::from)
    }
}

/// Parses a direct model config or extracts a nested text-transformer config.
///
/// Hugging Face multimodal repositories conventionally store the language backbone
/// under `text_config`. Historic/custom repositories also use `language_config` or
/// `llm_config`; accepting those aliases is safe because this crate intentionally
/// ignores the vision tower. NVIDIA's original Nemotron-H files contain the
/// non-standard JSON literal `Infinity`; it is normalized to `null`, the same
/// representation used by newer official checkpoints for an unbounded limit.
pub fn parse_text_config<T>(json: &str) -> Result<T, ConfigLoadError>
where
    T: Validate + DeserializeOwned,
{
    let normalized = normalize_positive_infinity(json);
    let value: serde_json::Value = serde_json::from_str(&normalized)?;
    let text_value = ["text_config", "language_config", "llm_config"]
        .iter()
        .find_map(|key| value.get(*key).filter(|nested| !nested.is_null()))
        .cloned()
        .unwrap_or(value);
    let config: T = serde_json::from_value(text_value)?;
    config.validate()?;
    Ok(config)
}

/// Rewrites bare positive `Infinity` tokens outside strings to JSON `null`.
///
/// JSON itself has no infinity literal. The legacy Nemotron-H artifact nevertheless
/// used Python's extension in `time_step_limit`. A small lexical pass is preferable
/// to adopting a permissive JSON dialect for every model. Quoted text is copied
/// byte-for-byte, including escaped quotes, so metadata strings cannot be changed.
fn normalize_positive_infinity(input: &str) -> String {
    /// Legacy non-JSON numeric literal accepted outside quoted metadata.
    const TOKEN: &[u8] = b"Infinity";
    let mut normalized = String::with_capacity(input.len());
    let mut remaining = input;
    let mut in_string = false;
    let mut escaped = false;

    while let Some(character) = remaining.chars().next() {
        if in_string {
            normalized.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            remaining = &remaining[character.len_utf8()..];
            continue;
        }

        if character == '"' {
            in_string = true;
            normalized.push('"');
            remaining = &remaining[character.len_utf8()..];
        } else if remaining.as_bytes().starts_with(TOKEN) {
            normalized.push_str("null");
            // The matched token is ASCII, so its end is a UTF-8 boundary.
            remaining = &remaining[TOKEN.len()..];
        } else {
            normalized.push(character);
            remaining = &remaining[character.len_utf8()..];
        }
    }

    normalized
}

/// A token field that may contain one ID or several accepted end-of-sequence IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TokenIds {
    /// One token ID.
    One(
        /// Sole token identifier, checked against the vocabulary by the model config.
        usize,
    ),
    /// Multiple token IDs, as used by several chat checkpoints.
    Many(
        /// Ordered token identifiers, checked for emptiness and vocabulary bounds.
        Vec<usize>,
    ),
}

impl TokenIds {
    /// Visits every ID without allocating a normalized temporary vector.
    pub fn iter(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        match self {
            Self::One(id) => Box::new(std::iter::once(*id)),
            Self::Many(ids) => Box::new(ids.iter().copied()),
        }
    }
}

/// Rotary-position-embedding algorithm encoded by Hugging Face configs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RopeType {
    /// Original geometric-frequency RoPE.
    #[default]
    #[serde(rename = "default")]
    Default,
    /// Uniform frequency scaling.
    #[serde(rename = "linear")]
    Linear,
    /// Dynamic NTK scaling.
    #[serde(rename = "dynamic")]
    Dynamic,
    /// YaRN interpolation/extrapolation.
    #[serde(rename = "yarn")]
    Yarn,
    /// Phi-style per-frequency long-context scaling.
    #[serde(rename = "longrope")]
    LongRope,
    /// Llama 3 low/high-frequency interpolation.
    #[serde(rename = "llama3")]
    Llama3,
    /// Context-proportional RoPE used by newer experimental checkpoints.
    #[serde(rename = "proportional")]
    Proportional,
}

/// Superset of RoPE fields used by the supported model families.
///
/// Deserialization accepts the current `rope_type` key, the legacy `type` key,
/// or both: Hugging Face config constructors copy `type` into `rope_type` and
/// `save_pretrained` then writes both. When both are present they must agree.
/// Serialization writes only `rope_type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(try_from = "RopeParametersRepr")]
pub struct RopeParameters {
    /// RoPE scaling algorithm; legacy configs serialize this key as `type`.
    #[serde(default)]
    pub rope_type: RopeType,
    /// Geometric base theta controlling the longest base wavelength.
    #[serde(default)]
    pub rope_theta: Option<f64>,
    /// Fraction of each attention head rotated in two-channel complex pairs.
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    /// Context-extension factor used by scaled variants.
    #[serde(default)]
    pub factor: Option<f64>,
    /// Context length on which the unscaled frequencies were trained.
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    /// Optional multiplier applied to attention after frequency scaling.
    #[serde(default)]
    pub attention_factor: Option<f64>,
    /// YaRN fast-rotation boundary.
    #[serde(default)]
    pub beta_fast: Option<f64>,
    /// YaRN slow-rotation boundary.
    #[serde(default)]
    pub beta_slow: Option<f64>,
    /// DeepSeek's YaRN magnitude scale.
    #[serde(default)]
    pub mscale: Option<f64>,
    /// DeepSeek's all-dimension YaRN magnitude scale.
    #[serde(default)]
    pub mscale_all_dim: Option<f64>,
    /// Short-context frequency factors for LongRoPE.
    #[serde(default)]
    pub short_factor: Option<Vec<f64>>,
    /// Long-context frequency factors for LongRoPE.
    #[serde(default)]
    pub long_factor: Option<Vec<f64>>,
    /// Llama 3 low-frequency multiplier.
    #[serde(default)]
    pub low_freq_factor: Option<f64>,
    /// Llama 3 high-frequency multiplier.
    #[serde(default)]
    pub high_freq_factor: Option<f64>,
    /// Qwen multimodal temporal/height/width complex-pair allocation.
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    /// Whether Qwen interleaves rather than concatenates multimodal frequencies.
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
    /// Whether YaRN rounds its correction range outward to whole pair indices.
    ///
    /// Absent means `true`, the reference default; gpt-oss checkpoints set
    /// `false` to keep fractional ramp bounds. Only YaRN reads this key.
    #[serde(default)]
    pub truncate: Option<bool>,
}

impl RopeType {
    /// Returns the serialized Hugging Face name, such as `"yarn"`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Linear => "linear",
            Self::Dynamic => "dynamic",
            Self::Yarn => "yarn",
            Self::LongRope => "longrope",
            Self::Llama3 => "llama3",
            Self::Proportional => "proportional",
        }
    }
}

/// Serialized form of [`RopeParameters`] with separate current and legacy type keys.
///
/// Serde aliases map both keys to one field slot and reject a map containing
/// both as a duplicate field, so the two keys are read independently here and
/// reconciled by [`resolve_rope_type`]. Destructuring every field in the
/// conversion makes adding a field to one struct but not the other a compile error.
#[derive(Deserialize)]
struct RopeParametersRepr {
    /// Current scaling-algorithm key.
    #[serde(default)]
    rope_type: Option<RopeType>,
    /// Legacy scaling-algorithm key retained by older and re-saved checkpoints.
    #[serde(default, rename = "type")]
    legacy_type: Option<RopeType>,
    /// See [`RopeParameters::rope_theta`].
    #[serde(default)]
    rope_theta: Option<f64>,
    /// See [`RopeParameters::partial_rotary_factor`].
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    /// See [`RopeParameters::factor`].
    #[serde(default)]
    factor: Option<f64>,
    /// See [`RopeParameters::original_max_position_embeddings`].
    #[serde(default)]
    original_max_position_embeddings: Option<usize>,
    /// See [`RopeParameters::attention_factor`].
    #[serde(default)]
    attention_factor: Option<f64>,
    /// See [`RopeParameters::beta_fast`].
    #[serde(default)]
    beta_fast: Option<f64>,
    /// See [`RopeParameters::beta_slow`].
    #[serde(default)]
    beta_slow: Option<f64>,
    /// See [`RopeParameters::mscale`].
    #[serde(default)]
    mscale: Option<f64>,
    /// See [`RopeParameters::mscale_all_dim`].
    #[serde(default)]
    mscale_all_dim: Option<f64>,
    /// See [`RopeParameters::short_factor`].
    #[serde(default)]
    short_factor: Option<Vec<f64>>,
    /// See [`RopeParameters::long_factor`].
    #[serde(default)]
    long_factor: Option<Vec<f64>>,
    /// See [`RopeParameters::low_freq_factor`].
    #[serde(default)]
    low_freq_factor: Option<f64>,
    /// See [`RopeParameters::high_freq_factor`].
    #[serde(default)]
    high_freq_factor: Option<f64>,
    /// See [`RopeParameters::mrope_section`].
    #[serde(default)]
    mrope_section: Option<Vec<usize>>,
    /// See [`RopeParameters::mrope_interleaved`].
    #[serde(default)]
    mrope_interleaved: Option<bool>,
    /// See [`RopeParameters::truncate`].
    #[serde(default)]
    truncate: Option<bool>,
}

impl TryFrom<RopeParametersRepr> for RopeParameters {
    type Error = String;

    /// Reconciles the type keys and moves every other field unchanged.
    fn try_from(repr: RopeParametersRepr) -> Result<Self, Self::Error> {
        let RopeParametersRepr {
            rope_type,
            legacy_type,
            rope_theta,
            partial_rotary_factor,
            factor,
            original_max_position_embeddings,
            attention_factor,
            beta_fast,
            beta_slow,
            mscale,
            mscale_all_dim,
            short_factor,
            long_factor,
            low_freq_factor,
            high_freq_factor,
            mrope_section,
            mrope_interleaved,
            truncate,
        } = repr;
        Ok(Self {
            rope_type: resolve_rope_type(rope_type, legacy_type)?,
            rope_theta,
            partial_rotary_factor,
            factor,
            original_max_position_embeddings,
            attention_factor,
            beta_fast,
            beta_slow,
            mscale,
            mscale_all_dim,
            short_factor,
            long_factor,
            low_freq_factor,
            high_freq_factor,
            mrope_section,
            mrope_interleaved,
            truncate,
        })
    }
}

/// Chooses the RoPE algorithm from the current and legacy keys.
///
/// Either key alone is authoritative, equal keys are accepted, absent keys mean
/// [`RopeType::Default`], and disagreeing keys are an error naming both values.
fn resolve_rope_type(
    current: Option<RopeType>,
    legacy: Option<RopeType>,
) -> Result<RopeType, String> {
    match (current, legacy) {
        (Some(current), Some(legacy)) if current != legacy => Err(format!(
            "rope_type \"{}\" conflicts with legacy type \"{}\"; a re-saved config must carry one algorithm",
            current.as_str(),
            legacy.as_str()
        )),
        (current, legacy) => Ok(current.or(legacy).unwrap_or_default()),
    }
}

impl RopeParameters {
    /// Validates RoPE dimensions and the fields required by each scaling variant.
    pub fn validate_for(
        &self,
        architecture: &'static str,
        head_dim: usize,
        max_position_embeddings: usize,
        fallback_theta: Option<f64>,
    ) -> Result<(), ConfigError> {
        nonzero(architecture, "head_dim", head_dim)?;
        let theta = self.rope_theta.or(fallback_theta).unwrap_or(10_000.0);
        positive_finite(architecture, "rope_theta", theta)?;

        let factor = self.partial_rotary_factor.unwrap_or(1.0);
        let proportional = self.rope_type == RopeType::Proportional;
        if proportional {
            // Proportional RoPE reads the factor as the fraction of pairs that
            // keep a nonzero frequency; zero (every pair unrotated) is valid,
            // as in Transformers 5.8.0 and `ProportionalRopeScaling`.
            if !(factor.is_finite() && (0.0..=1.0).contains(&factor)) {
                return Err(ConfigError::new(
                    architecture,
                    "partial_rotary_factor",
                    "must be finite and in [0, 1] for proportional RoPE",
                ));
            }
        } else if !(factor.is_finite() && 0.0 < factor && factor <= 1.0) {
            return Err(ConfigError::new(
                architecture,
                "partial_rotary_factor",
                "must be finite and in (0, 1]",
            ));
        }
        let rotary_width = if proportional {
            // Proportional RoPE rotates the whole head in split-half pairs and
            // reads the factor only as the fraction of leading pairs that keep
            // a nonzero frequency, `floor(factor * head_dim / 2)`, so the
            // product need not be an integral channel count.
            if !head_dim.is_multiple_of(2) {
                return Err(ConfigError::new(
                    architecture,
                    "head_dim",
                    "proportional RoPE rotates whole-head channel pairs, so head_dim must be even",
                ));
            }
            head_dim
        } else {
            let rotary_width = head_dim as f64 * factor;
            let rounded_rotary_width = rotary_width.round();
            if (rotary_width - rounded_rotary_width).abs() > 1e-9 || rotary_width < 2.0 {
                return Err(ConfigError::new(
                    architecture,
                    rotary_width_field(rotary_width, head_dim),
                    "head_dim * factor must be an integer of at least two channels",
                ));
            }
            rounded_rotary_width as usize
        };
        if !rotary_width.is_multiple_of(2) {
            return Err(ConfigError::new(
                architecture,
                rotary_width_field(rotary_width as f64, head_dim),
                "RoPE rotates channel pairs, so the effective width must be even",
            ));
        }

        if let Some(sections) = &self.mrope_section {
            if sections.is_empty() || sections.contains(&0) {
                return Err(ConfigError::new(
                    architecture,
                    "mrope_section",
                    "sections must be non-empty and contain only positive widths",
                ));
            }
            let section_width = sections
                .iter()
                .try_fold(0usize, |sum, width| sum.checked_add(*width))
                .and_then(|sum| sum.checked_mul(2));
            if section_width != Some(rotary_width) {
                return Err(ConfigError::new(
                    architecture,
                    "mrope_section",
                    "twice the complex-pair section sum must equal the rotary width",
                ));
            }
        }

        match self.rope_type {
            RopeType::Default => {}
            RopeType::Linear | RopeType::Dynamic => {
                scaling_factor(architecture, self.factor)?;
            }
            // Transformers 5.8.0 defaults the proportional position divisor to one.
            RopeType::Proportional => {
                if self.factor.is_some() {
                    scaling_factor(architecture, self.factor)?;
                }
            }
            RopeType::Yarn => {
                scaling_factor(architecture, self.factor)?;
                original_context(architecture, self, max_position_embeddings)?;
                // YaRN specifies each correction bound independently and uses
                // 32/1 when a side is absent. Resolve those defaults before
                // validation so malformed one-sided overrides fail during
                // architecture validation rather than delayed model startup.
                let fast = self.beta_fast.unwrap_or(32.0);
                let slow = self.beta_slow.unwrap_or(1.0);
                positive_finite(architecture, "beta_fast", fast)?;
                positive_finite(architecture, "beta_slow", slow)?;
                if fast <= slow {
                    return Err(ConfigError::new(
                        architecture,
                        "beta_fast",
                        "must be greater than beta_slow",
                    ));
                }
            }
            RopeType::LongRope => {
                original_context(architecture, self, max_position_embeddings)?;
                let expected = rotary_width / 2;
                for (field, values) in [
                    ("short_factor", self.short_factor.as_ref()),
                    ("long_factor", self.long_factor.as_ref()),
                ] {
                    let values = values.ok_or_else(|| {
                        ConfigError::new(architecture, field, "is required for longrope")
                    })?;
                    if values.len() != expected
                        || values
                            .iter()
                            .any(|value| !value.is_finite() || *value <= 0.0)
                    {
                        return Err(ConfigError::new(
                            architecture,
                            field,
                            format!("must contain {expected} positive finite factors"),
                        ));
                    }
                }
            }
            RopeType::Llama3 => {
                scaling_factor(architecture, self.factor)?;
                original_context(architecture, self, max_position_embeddings)?;
                let low = self.low_freq_factor.ok_or_else(|| {
                    ConfigError::new(
                        architecture,
                        "low_freq_factor",
                        "is required for llama3 RoPE",
                    )
                })?;
                let high = self.high_freq_factor.ok_or_else(|| {
                    ConfigError::new(
                        architecture,
                        "high_freq_factor",
                        "is required for llama3 RoPE",
                    )
                })?;
                positive_finite(architecture, "low_freq_factor", low)?;
                positive_finite(architecture, "high_freq_factor", high)?;
                if high <= low {
                    return Err(ConfigError::new(
                        architecture,
                        "high_freq_factor",
                        "must be greater than low_freq_factor",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Names the field to repair when the rotary width `head_dim * partial_rotary_factor` is invalid.
///
/// `rotary_width` is the exact product. A whole-head rotation (the product
/// equals `head_dim`, so the factor is one) is repaired through `head_dim`,
/// matching the shared translator's mapping of `RopeConfigError`; anything
/// else, including a non-integral product, through `partial_rotary_factor`.
fn rotary_width_field(rotary_width: f64, head_dim: usize) -> &'static str {
    if rotary_width == head_dim as f64 {
        "head_dim"
    } else {
        "partial_rotary_factor"
    }
}

/// Validates that a serialized dimension or count is nonzero.
pub(crate) fn nonzero(
    architecture: &'static str,
    field: &'static str,
    value: usize,
) -> Result<(), ConfigError> {
    if value == 0 {
        Err(ConfigError::new(architecture, field, "must be nonzero"))
    } else {
        Ok(())
    }
}

/// Validates a finite, strictly positive floating-point parameter.
pub(crate) fn positive_finite(
    architecture: &'static str,
    field: &'static str,
    value: f64,
) -> Result<(), ConfigError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ConfigError::new(
            architecture,
            field,
            "must be finite and greater than zero",
        ))
    }
}

/// Validates a probability or dropout rate in the closed unit interval.
pub(crate) fn probability(
    architecture: &'static str,
    field: &'static str,
    value: f64,
) -> Result<(), ConfigError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ConfigError::new(
            architecture,
            field,
            "must be finite and in [0, 1]",
        ))
    }
}

/// Validates the quotient needed when query heads share key/value heads.
pub(crate) fn divisible(
    architecture: &'static str,
    numerator_field: &'static str,
    numerator: usize,
    denominator_field: &'static str,
    denominator: usize,
) -> Result<(), ConfigError> {
    nonzero(architecture, numerator_field, numerator)?;
    nonzero(architecture, denominator_field, denominator)?;
    if numerator.is_multiple_of(denominator) {
        Ok(())
    } else {
        Err(ConfigError::new(
            architecture,
            numerator_field,
            format!("must be divisible by {denominator_field}"),
        ))
    }
}

/// Validates that token IDs index rows in the embedding table.
pub(crate) fn token_ids(
    architecture: &'static str,
    field: &'static str,
    ids: Option<&TokenIds>,
    vocab_size: usize,
) -> Result<(), ConfigError> {
    if let Some(ids) = ids {
        let values = ids.iter().collect::<Vec<_>>();
        if values.is_empty() {
            return Err(ConfigError::new(
                architecture,
                field,
                "must contain at least one token ID",
            ));
        }
        if values.iter().any(|id| *id >= vocab_size) {
            return Err(ConfigError::new(
                architecture,
                field,
                "contains an ID outside the vocabulary",
            ));
        }
    }
    Ok(())
}

/// Validates a supported model-type discriminator.
pub(crate) fn model_type(
    architecture: &'static str,
    actual: &str,
    allowed: &[&str],
) -> Result<(), ConfigError> {
    if allowed.contains(&actual) {
        Ok(())
    } else {
        Err(ConfigError::new(
            architecture,
            "model_type",
            format!("expected one of {allowed:?}, found {actual:?}"),
        ))
    }
}

/// Validates the scale shared by linear, dynamic, proportional, and YaRN RoPE.
fn scaling_factor(architecture: &'static str, factor: Option<f64>) -> Result<(), ConfigError> {
    let factor = factor.ok_or_else(|| {
        ConfigError::new(
            architecture,
            "rope.factor",
            "is required by the selected scaling algorithm",
        )
    })?;
    if factor.is_finite() && factor >= 1.0 {
        Ok(())
    } else {
        Err(ConfigError::new(
            architecture,
            "rope.factor",
            "must be finite and at least one",
        ))
    }
}

/// Validates the unscaled context length used by extrapolating RoPE variants.
fn original_context(
    architecture: &'static str,
    rope: &RopeParameters,
    max_position_embeddings: usize,
) -> Result<(), ConfigError> {
    let original = rope.original_max_position_embeddings.ok_or_else(|| {
        ConfigError::new(
            architecture,
            "rope.original_max_position_embeddings",
            "is required by the selected scaling algorithm",
        )
    })?;
    if original == 0 || original > max_position_embeddings {
        Err(ConfigError::new(
            architecture,
            "rope.original_max_position_embeddings",
            "must be nonzero and no larger than max_position_embeddings",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
