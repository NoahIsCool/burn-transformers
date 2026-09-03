//! Shared, backend-free model configuration machinery.
//!
//! Transformer checkpoints encode matrix shapes indirectly. For example, grouped
//! query attention is valid only when the number of query heads is an integer
//! multiple of the number of key/value heads, and RoPE rotates pairs of channels,
//! so its effective dimension must be even. The helpers in this module turn those
//! mathematical assumptions into errors at load time rather than tensor panics in
//! the first forward pass.

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

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
    Json(serde_json::Error),
    /// JSON decoding succeeded, but the described network cannot be wired safely.
    Validation(ConfigError),
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

/// Validation implemented by every model topology schema.
pub trait Validate {
    /// Checks all relationships needed to construct matrices and caches safely.
    fn validate(&self) -> Result<(), ConfigError>;
}

/// Common loading API for a model configuration.
pub trait ModelConfig: Validate + DeserializeOwned + Sized {
    /// Stable family name used in diagnostics and compatibility tables.
    const FAMILY: &'static str;

    /// Parses either a direct text config or a multimodal wrapper and validates it.
    fn from_json_str(json: &str) -> Result<Self, ConfigLoadError> {
        parse_text_config(json)
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
    const TOKEN: &[u8] = b"Infinity";
    let mut normalized = String::with_capacity(input.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;

    while index < input.len() {
        let remaining = &input[index..];
        let character = remaining
            .chars()
            .next()
            .expect("index is always below the UTF-8 string length");
        if in_string {
            normalized.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            index += character.len_utf8();
            continue;
        }

        if character == '"' {
            in_string = true;
            normalized.push('"');
            index += character.len_utf8();
        } else if remaining.as_bytes().starts_with(TOKEN) {
            normalized.push_str("null");
            index += TOKEN.len();
        } else {
            normalized.push(character);
            index += character.len_utf8();
        }
    }

    normalized
}

/// A token field that may contain one ID or several accepted end-of-sequence IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TokenIds {
    /// One token ID.
    One(usize),
    /// Multiple token IDs, as used by several chat checkpoints.
    Many(Vec<usize>),
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RopeParameters {
    /// RoPE scaling algorithm; legacy configs serialize this key as `type`.
    #[serde(default, alias = "type")]
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
        if !(factor.is_finite() && 0.0 < factor && factor <= 1.0) {
            return Err(ConfigError::new(
                architecture,
                "partial_rotary_factor",
                "must be finite and in (0, 1]",
            ));
        }
        let rotary_width = head_dim as f64 * factor;
        let rounded_rotary_width = rotary_width.round();
        if (rotary_width - rounded_rotary_width).abs() > 1e-9 || rotary_width < 2.0 {
            return Err(ConfigError::new(
                architecture,
                "partial_rotary_factor",
                "head_dim * factor must be an integer of at least two channels",
            ));
        }
        let rotary_width = rounded_rotary_width as usize;
        if !rotary_width.is_multiple_of(2) {
            return Err(ConfigError::new(
                architecture,
                "partial_rotary_factor",
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
            RopeType::Linear | RopeType::Dynamic | RopeType::Proportional => {
                scaling_factor(architecture, self.factor)?;
            }
            RopeType::Yarn => {
                scaling_factor(architecture, self.factor)?;
                original_context(architecture, self, max_position_embeddings)?;
                if let (Some(fast), Some(slow)) = (self.beta_fast, self.beta_slow) {
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
mod tests {
    use super::*;

    #[derive(Debug, Deserialize)]
    struct TinyConfig {
        width: usize,
    }

    impl Validate for TinyConfig {
        /// Rejects the zero-width synthetic topology used by parser tests.
        fn validate(&self) -> Result<(), ConfigError> {
            nonzero("tiny", "width", self.width)
        }
    }

    #[test]
    fn extracts_supported_text_wrapper_aliases() {
        for key in ["text_config", "language_config", "llm_config"] {
            let json = format!(r#"{{"{key}":{{"width":4}},"vision_config":{{}}}}"#);
            let config: TinyConfig = parse_text_config(&json).unwrap();
            assert_eq!(config.width, 4);
        }
    }

    #[test]
    fn positive_infinity_is_normalized_only_outside_strings() {
        let json = r#"{"width":4,"limit":[0.0,Infinity],"note":"Infinity – 無限"}"#;
        let normalized = normalize_positive_infinity(json);
        let value: serde_json::Value = serde_json::from_str(&normalized).unwrap();
        assert!(value["limit"][1].is_null());
        assert_eq!(value["note"], "Infinity – 無限");
    }

    #[test]
    fn invalid_effective_rope_width_is_rejected() {
        let rope = RopeParameters {
            partial_rotary_factor: Some(0.3),
            ..RopeParameters::default()
        };
        assert!(rope.validate_for("test", 8, 128, None).is_err());
    }

    #[test]
    fn multimodal_rope_sections_count_complex_pairs() {
        let rope = RopeParameters {
            partial_rotary_factor: Some(0.5),
            mrope_section: Some(vec![1, 1]),
            ..RopeParameters::default()
        };
        rope.validate_for("test", 8, 128, None).unwrap();
    }

    #[test]
    fn scaled_rope_variants_require_their_shape_fields() {
        let linear = RopeParameters {
            rope_type: RopeType::Linear,
            ..RopeParameters::default()
        };
        assert!(linear.validate_for("test", 8, 128, None).is_err());

        let yarn = RopeParameters {
            rope_type: RopeType::Yarn,
            factor: Some(4.0),
            original_max_position_embeddings: Some(256),
            beta_fast: Some(1.0),
            beta_slow: Some(2.0),
            ..RopeParameters::default()
        };
        assert!(yarn.validate_for("test", 8, 128, None).is_err());

        let llama3 = RopeParameters {
            rope_type: RopeType::Llama3,
            factor: Some(8.0),
            original_max_position_embeddings: Some(64),
            low_freq_factor: Some(1.0),
            high_freq_factor: Some(4.0),
            ..RopeParameters::default()
        };
        llama3.validate_for("test", 8, 128, None).unwrap();
    }

    #[test]
    fn longrope_factor_vectors_match_complex_frequency_count() {
        let valid = RopeParameters {
            rope_type: RopeType::LongRope,
            original_max_position_embeddings: Some(64),
            short_factor: Some(vec![1.0; 4]),
            long_factor: Some(vec![2.0; 4]),
            ..RopeParameters::default()
        };
        valid.validate_for("test", 8, 128, None).unwrap();

        let mut invalid = valid;
        invalid.long_factor = Some(vec![2.0; 3]);
        assert!(invalid.validate_for("test", 8, 128, None).is_err());
    }

    #[test]
    fn token_lists_must_be_nonempty_and_inside_the_vocabulary() {
        assert!(token_ids("test", "eos", Some(&TokenIds::Many(vec![])), 4).is_err());
        assert!(token_ids("test", "eos", Some(&TokenIds::Many(vec![1, 4])), 4).is_err());
        token_ids("test", "eos", Some(&TokenIds::Many(vec![1, 3])), 4).unwrap();
    }
}
