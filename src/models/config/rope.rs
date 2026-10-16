//! Translation of Hugging Face RoPE metadata into the executable [`RopeConfig`].
//!
//! # Problem
//!
//! A checkpoint describes rotary position embeddings in two places: the
//! `rope_parameters` mapping (legacy name `rope_scaling`, legacy algorithm key
//! `type`) and loose top-level fields (`rope_theta`, `partial_rotary_factor`,
//! `head_dim`, `max_position_embeddings`, and Phi-3's
//! `original_max_position_embeddings`). Transformers 5.x additionally allows a
//! *per-layer-type* mapping such as Gemma 3's
//! `{"sliding_attention": {...}, "full_attention": {...}}`. This module is the
//! single place that resolves those sources into a validated
//! [`crate::position::RopeConfig`]; families describe their surrounding fields
//! and conventions with a [`RopeContext`] and call
//! [`RopeParameters::to_rope_config`] or [`RopeParameterSet::to_rope_config`].
//!
//! # Mathematics
//!
//! For rotary width `d` (the even prefix of a head of width `h`) and base
//! `theta`, pair `i in 0..d/2` has the base inverse frequency
//! `omega_i = theta^(-2i/d)`. Each algorithm rewrites these frequencies and may
//! scale the cosine/sine tables by an amplitude `a` (Transformers calls it
//! `attention_scaling`); see [`crate::position`] for the evaluated tables.
//! With `f` the `factor` and `L` the original training context:
//!
//! | `rope_type` | frequencies | amplitude `a` | fields read |
//! | --- | --- | --- | --- |
//! | `default` | `omega_i` | 1 | — |
//! | `linear` | `omega_i / f` | 1 | `factor` |
//! | `dynamic` | base `theta (f c / L - (f - 1))^(d/(d-2))` once `c > L` | 1 | `factor`, `original_max_position_embeddings` (see [`DynamicNtkContext`]) |
//! | `yarn` | ramp between `omega_i` and `omega_i / f` | see below | `factor`, `original_max_position_embeddings`, `attention_factor`, `beta_fast`, `beta_slow`, `mscale`, `mscale_all_dim`, `truncate` |
//! | `longrope` | `omega_i / r_i`, short or long table by context | `sqrt(1 + ln f / ln L)` unless given | `short_factor`, `long_factor`, `factor`, `original_max_position_embeddings`, `attention_factor` |
//! | `llama3` | wavelength-banded interpolation | 1 | `factor`, `low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings` |
//! | `proportional` | `omega_i / f` for `floor(p h / 2)` leading pairs, zero after | 1 | `factor` (default 1), `partial_rotary_factor` as `p` |
//!
//! `c` is the evaluated context (largest position plus one). Every algorithm
//! also reads `rope_theta` and `partial_rotary_factor`; except for
//! proportional RoPE, `d = p h` must be an even integer (Transformers truncates
//! `int(p h)`, which would silently change projection shapes). `mrope_section`
//! and `mrope_interleaved` describe multimodal position axes that the caller
//! owns (see [`MropeHandling`]).
//!
//! ## YaRN amplitude
//!
//! With the logarithmic magnitude correction
//! `m(s, k) = 1 + 0.1 k ln s` for `s > 1` and `m(s, k) = 1` otherwise
//! ([`yarn_mscale`]), the two conventions ([`YarnMscaleConvention`]) are:
//!
//! * **Hugging Face** (`_compute_yarn_parameters`): an explicit
//!   `attention_factor` wins; otherwise `m(f, mscale) / m(f, mscale_all_dim)`
//!   when both are present and nonzero; otherwise `m(f, 1)`. A key this rule
//!   leaves unread is an unused field, except a `mscale_all_dim` read by the
//!   family's softmax correction (below).
//! * **DeepSeek reference** (`modeling_deepseek.py`): `m(f, mscale) /
//!   m(f, mscale_all_dim)` with defaults `mscale = 1`, `mscale_all_dim = 0`
//!   whenever either key is present; `attention_factor` only when neither is.
//!
//! Both agree when both keys are present and nonzero, which covers every
//! released DeepSeek checkpoint (`mscale = mscale_all_dim` gives `a = 1`).
//!
//! ## Softmax correction
//!
//! DeepSeek's reference attention (`modeling_deepseek.py`) and Transformers'
//! `deepseek_v3` also multiply the softmax scale `1/sqrt(d_qk)` by
//! `m(f, mscale_all_dim)^2` for *every* non-default `rope_type`, not only
//! YaRN. A family declares this with [`SoftmaxMscale::MscaleAllDimSquared`]
//! and obtains the multiplier from [`RopeParameters::softmax_scale_multiplier`];
//! `mscale_all_dim` is then read for every scaled algorithm. Under the default
//! [`SoftmaxMscale::Unscaled`] (every other Transformers family, including
//! 5.2's `deepseek_v2`), only the YaRN amplitude can read it.
//!
//! # Source precedence
//!
//! A value supplied in two places must agree under [`UnusedRopeFields::Reject`];
//! [`UnusedRopeFields::Ignore`] instead reproduces the Transformers 5.2 choice.
//!
//! | value | resolution order | Transformers 5.2 choice on conflict |
//! | --- | --- | --- |
//! | theta | layer override, nested `rope_theta`, top-level `rope_theta`, default | nested (`setdefault`) |
//! | partial factor | nested, top-level, 1 | top-level (`standardize_rope_params`) |
//! | original context, flat mapping | nested, top-level, `max_position_embeddings` | top-level (Phi-3 exception) |
//! | original context, per-layer-type mapping | nested, `max_position_embeddings` | nested; the top-level value is never read |
//!
//! The last row follows `standardize_rope_params`, whose per-layer branch only
//! `setdefault`s `max_position_embeddings`; a top-level value that would change
//! a layer's context is therefore an unused field.
//!
//! Dynamic NTK uses `max_position_embeddings` as `L` in Transformers and in
//! DeepSeek's reference; see [`DynamicNtkContext`] for the crate's earlier,
//! legacy-only alternative. With that `L` the scaling branch is unreachable
//! in the built-in families, because each refuses positions at or beyond
//! `max_position_embeddings` in both its stateless and cached paths: within
//! every accepted input a family's dynamic-NTK frequencies equal unscaled
//! RoPE, exactly what Transformers computes for passes of those lengths, and
//! a longer pass (which Transformers would rescale as a whole) is refused
//! rather than computed unscaled. Compositions with their own position limit,
//! such as `RotaryAttention` in custom specifications, do reach it.
//!
//! # Strictness
//!
//! Transformers warns about and then ignores keys an algorithm does not read.
//! By default this translator instead returns a contextual
//! [`crate::Error::Configuration`] naming the field, so a parsed value never
//! disappears silently. Values Transformers coerces are rejected too: a zero
//! YaRN `beta_fast`/`beta_slow` (read as its default through Python `or`) or a
//! required `factor` that is absent. Unknown JSON keys are dropped earlier by
//! the serde schema and are not visible here. Every [`RopeParameters`] field
//! is classified by an exhaustive destructuring, so adding a field fails to
//! compile until it is either resolved for every algorithm or listed as an
//! algorithm-specific field.
//!
//! All arithmetic is host `f64`; the position module rounds frequencies to
//! `f32` once. Formulas follow Transformers 5.2.0 `modeling_rope_utils.py`
//! (proportional RoPE: 5.8.0) and DeepSeek-V2-Lite `modeling_deepseek.py` at
//! revision `604d5664dddd88a0433dbae533b7fe9472482de0`.

use std::collections::{BTreeMap, BTreeSet};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

use super::plan::{OriginalContextFault, resolve_original_context};
use super::{RopeParameters, RopeType};
use crate::position::{
    DynamicNtkScaling, LinearRopeScaling, Llama3RopeScaling, LongRopeScaling,
    ProportionalRopeScaling, RopeConfig, RopeConfigError, RopeScaling, YarnConfig,
    rotary_dimension_from_partial_factor,
};
use crate::{Error, Result};

/// Transformers' `RotaryEmbeddingConfigMixin.default_theta`, used when no theta is given.
pub const DEFAULT_ROPE_THETA: f64 = 10_000.0;

/// Returns the YaRN logarithmic magnitude correction `m(scale, multiplier)`.
///
/// This is `1 + 0.1 * multiplier * ln(scale)` for `scale > 1` and exactly one
/// otherwise, matching both Transformers' `get_mscale` and DeepSeek's
/// `yarn_get_mscale`. A multiplier of zero therefore yields one.
#[must_use]
pub fn yarn_mscale(scale: f64, multiplier: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        // Same operation order as the private DeepSeek translator so results
        // are bit-identical: (0.1 * multiplier) * ln(scale), then + 1.
        1.0 + 0.1 * multiplier * scale.ln()
    }
}

// ---------------------------------------------------------------------------
// Translation options.
// ---------------------------------------------------------------------------

/// How YaRN derives its cosine/sine amplitude from `mscale` fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum YarnMscaleConvention {
    /// Transformers `_compute_yarn_parameters`: explicit `attention_factor`,
    /// else `m(f, mscale) / m(f, mscale_all_dim)` when both are nonzero, else
    /// `m(f, 1)`. A `mscale` this rule would not read is an unused field, and
    /// so is a `mscale_all_dim` that neither this rule nor a declared
    /// [`SoftmaxMscale::MscaleAllDimSquared`] correction reads.
    #[default]
    HuggingFace,
    /// DeepSeek's reference `DeepseekV2YarnRotaryEmbedding`:
    /// `m(f, mscale ?? 1) / m(f, mscale_all_dim ?? 0)` whenever either key is
    /// present. That reference has no `attention_factor`, so supplying it
    /// together with a `mscale` key is an unused field.
    DeepSeekReference,
}

/// Whether the family's attention multiplies its softmax scale by `m(f, mscale_all_dim)^2`.
///
/// This is an attention-side convention, but it decides whether
/// `mscale_all_dim` is a read field, so the translator needs it to keep the
/// unused-field policy exact. See the module's softmax-correction section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SoftmaxMscale {
    /// The softmax scale is the plain `1/sqrt(d_qk)`, as in every Transformers
    /// family except `deepseek_v3` (5.2's `deepseek_v2` included).
    /// `mscale_all_dim` is then read only where the YaRN amplitude reads it, and
    /// [`RopeParameters::softmax_scale_multiplier`] is always one.
    #[default]
    Unscaled,
    /// DeepSeek's reference attention and Transformers' `deepseek_v3`: for
    /// every non-default `rope_type` with a nonzero `mscale_all_dim`, the
    /// softmax scale is multiplied by `m(f, mscale_all_dim)^2`. The translator
    /// then treats `mscale_all_dim` as read by every scaled algorithm and
    /// validates the multiplier, which requires an explicit `factor`.
    MscaleAllDimSquared,
}

/// Which trained context `L` dynamic NTK compares the evaluated context against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DynamicNtkContext {
    /// Transformers 5.2 and DeepSeek's reference: `L = max_position_embeddings`.
    /// Both ignore a supplied `original_max_position_embeddings` for dynamic
    /// NTK, so one that differs from `max_position_embeddings` is an unused field.
    #[default]
    MaxPositionEmbeddings,
    /// Legacy only: `L = original_max_position_embeddings`, falling back to
    /// `max_position_embeddings`. This reproduces the crate's private family
    /// translators before this module (Qwen3.5, Qwen3-MoE, DeepSeek-V2,
    /// Muse Glimmer), but none of those families' references uses it: the
    /// Transformers `qwen3_5`, `qwen3_moe`, and `muse_glimmer` rotary
    /// embeddings and DeepSeek's `DeepseekV2DynamicNTKScalingRotaryEmbedding`
    /// all scale against `max_position_embeddings`. Select it only to
    /// reproduce results recorded with those translators.
    OriginalMaxPositionEmbeddings,
}

/// Whether the family's attention can rotate a strict prefix of each head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartialRotary {
    /// `partial_rotary_factor = p` rotates the even prefix `d = p h`; the
    /// remaining channels pass through unrotated (Qwen3.5, Phi-3, GPT-NeoX).
    #[default]
    Prefix,
    /// The family always rotates whole heads (Llama, Qwen3, Gemma, gpt-oss,
    /// DeepSeek MLA's rope sub-head). A resolved factor other than one is an
    /// unused field; proportional RoPE still reads it as its pair fraction.
    FullHeadOnly,
}

/// Whether the caller consumes multimodal RoPE section metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MropeHandling {
    /// `mrope_section` and `mrope_interleaved` are unused fields.
    #[default]
    Reject,
    /// The caller feeds identical temporal/height/width positions (text-only
    /// inputs). Every multimodal axis then carries the same position, so the
    /// sectioned rotation equals the one-dimensional tables built here and the
    /// section layout needs no translation.
    TextPositions,
}

/// What to do with a parsed field the selected algorithm does not read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnusedRopeFields {
    /// Return a contextual configuration error naming the field.
    #[default]
    Reject,
    /// Ignore it as Transformers 5.2 does (it logs a warning), and resolve
    /// duplicated sources with Transformers' precedence instead of rejecting a
    /// disagreement.
    Ignore,
}

/// Surrounding config fields and family conventions for one RoPE translation.
///
/// Build with [`RopeContext::new`] and the `with_*` methods. Every value is a
/// *fallback or convention*: the nested [`RopeParameters`] stay authoritative
/// for algorithm fields. A per-layer-type family builds one context per layer
/// type, because legacy defaults (for example Gemma 3's global and local theta)
/// differ by layer type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeContext {
    /// Family name used as the first segment of error field paths.
    architecture: &'static str,
    /// Full query/key head width `h` to which `partial_rotary_factor` applies.
    head_dim: usize,
    /// Serialized `max_position_embeddings`, the fallback trained context.
    max_position_embeddings: usize,
    /// Legacy top-level `rope_theta`, a fallback for the nested value.
    rope_theta: Option<f64>,
    /// Theta used when neither the nested nor the top-level value is present.
    default_rope_theta: f64,
    /// Per-layer theta that supersedes every other theta source.
    layer_rope_theta: Option<f64>,
    /// Legacy top-level `partial_rotary_factor`.
    partial_rotary_factor: Option<f64>,
    /// Top-level `original_max_position_embeddings` (Phi-3 style).
    original_max_position_embeddings: Option<usize>,
    /// Whether the family can rotate a strict head prefix.
    partial_rotary: PartialRotary,
    /// YaRN amplitude convention.
    mscale_convention: YarnMscaleConvention,
    /// Whether the family's softmax scale reads `mscale_all_dim`.
    softmax_mscale: SoftmaxMscale,
    /// Trained-context rule for dynamic NTK.
    dynamic_ntk_context: DynamicNtkContext,
    /// Whether multimodal section metadata is consumed by the caller.
    mrope: MropeHandling,
    /// Policy for parsed fields that the selected algorithm does not read.
    unused_fields: UnusedRopeFields,
}

impl RopeContext {
    /// Starts a context for `architecture` with Transformers defaults.
    ///
    /// Defaults: no top-level theta/partial/original values, default theta
    /// [`DEFAULT_ROPE_THETA`], [`PartialRotary::Prefix`],
    /// [`YarnMscaleConvention::HuggingFace`], [`SoftmaxMscale::Unscaled`],
    /// [`DynamicNtkContext::MaxPositionEmbeddings`], [`MropeHandling::Reject`],
    /// and [`UnusedRopeFields::Reject`].
    #[must_use]
    pub const fn new(
        architecture: &'static str,
        head_dim: usize,
        max_position_embeddings: usize,
    ) -> Self {
        Self {
            architecture,
            head_dim,
            max_position_embeddings,
            rope_theta: None,
            default_rope_theta: DEFAULT_ROPE_THETA,
            layer_rope_theta: None,
            partial_rotary_factor: None,
            original_max_position_embeddings: None,
            partial_rotary: PartialRotary::Prefix,
            mscale_convention: YarnMscaleConvention::HuggingFace,
            softmax_mscale: SoftmaxMscale::Unscaled,
            dynamic_ntk_context: DynamicNtkContext::MaxPositionEmbeddings,
            mrope: MropeHandling::Reject,
            unused_fields: UnusedRopeFields::Reject,
        }
    }

    /// Supplies the legacy top-level `rope_theta`, used when the nested value is absent.
    #[must_use]
    pub const fn with_rope_theta(mut self, rope_theta: Option<f64>) -> Self {
        self.rope_theta = rope_theta;
        self
    }

    /// Replaces the theta used when no theta is serialized (Transformers `default_theta`).
    #[must_use]
    pub const fn with_default_rope_theta(mut self, default_rope_theta: f64) -> Self {
        self.default_rope_theta = default_rope_theta;
        self
    }

    /// Supplies a per-layer theta that supersedes the nested and top-level values.
    ///
    /// Muse Glimmer serializes one theta per layer (`layer_rope_theta`); the
    /// global `rope_parameters.rope_theta` only seeds that schedule.
    #[must_use]
    pub const fn with_layer_rope_theta(mut self, theta: f64) -> Self {
        self.layer_rope_theta = Some(theta);
        self
    }

    /// Supplies the legacy top-level `partial_rotary_factor`.
    #[must_use]
    pub const fn with_partial_rotary_factor(mut self, factor: Option<f64>) -> Self {
        self.partial_rotary_factor = factor;
        self
    }

    /// Supplies a top-level `original_max_position_embeddings` (Phi-3 style).
    ///
    /// Only algorithms that read an original context (`yarn`, `longrope`,
    /// `llama3`, and `dynamic` under
    /// [`DynamicNtkContext::OriginalMaxPositionEmbeddings`]) consult it; it is a
    /// config-level attribute, so other algorithms do not treat it as unused.
    #[must_use]
    pub const fn with_original_max_position_embeddings(mut self, original: Option<usize>) -> Self {
        self.original_max_position_embeddings = original;
        self
    }

    /// Declares whether the family can rotate a strict head prefix.
    #[must_use]
    pub const fn with_partial_rotary(mut self, partial_rotary: PartialRotary) -> Self {
        self.partial_rotary = partial_rotary;
        self
    }

    /// Selects how YaRN resolves its amplitude from `mscale` fields.
    #[must_use]
    pub const fn with_yarn_mscale_convention(mut self, convention: YarnMscaleConvention) -> Self {
        self.mscale_convention = convention;
        self
    }

    /// Declares whether the family's softmax scale applies `m(f, mscale_all_dim)^2`.
    ///
    /// The family must then multiply its softmax scale by
    /// [`RopeParameters::softmax_scale_multiplier`]; translation validates
    /// that multiplier so it cannot fail later.
    #[must_use]
    pub const fn with_softmax_mscale(mut self, softmax_mscale: SoftmaxMscale) -> Self {
        self.softmax_mscale = softmax_mscale;
        self
    }

    /// Selects the trained context dynamic NTK compares against.
    #[must_use]
    pub const fn with_dynamic_ntk_context(mut self, context: DynamicNtkContext) -> Self {
        self.dynamic_ntk_context = context;
        self
    }

    /// Declares whether the caller consumes multimodal section metadata.
    #[must_use]
    pub const fn with_mrope(mut self, mrope: MropeHandling) -> Self {
        self.mrope = mrope;
        self
    }

    /// Selects the policy for parsed fields the chosen algorithm does not read.
    #[must_use]
    pub const fn with_unused_fields(mut self, policy: UnusedRopeFields) -> Self {
        self.unused_fields = policy;
        self
    }

    /// Returns the family name used in error field paths.
    #[must_use]
    pub const fn architecture(&self) -> &'static str {
        self.architecture
    }

    /// Returns the full query/key head width.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Returns the serialized maximum context length.
    #[must_use]
    pub const fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
}

// ---------------------------------------------------------------------------
// Flat translation.
// ---------------------------------------------------------------------------

/// Where a translation's fields live, for error paths and per-layer quirks.
struct Scope<'a> {
    /// Translation options and surrounding fields.
    context: &'a RopeContext,
    /// Dotted prefix below the architecture, `rope` or `rope.<layer_type>`.
    prefix: String,
    /// Whether the parameters came from a per-layer-type mapping.
    per_layer_type: bool,
}

impl Scope<'_> {
    /// Builds a configuration error at `architecture.prefix.field`.
    fn error(&self, field: &str, reason: impl Into<String>) -> Error {
        Error::configuration(
            format!("{}.{}.{field}", self.context.architecture, self.prefix),
            reason,
        )
    }

    /// Builds a configuration error at a top-level context field such as `head_dim`.
    fn context_error(&self, field: &str, reason: impl Into<String>) -> Error {
        Error::configuration(format!("{}.{field}", self.context.architecture), reason)
    }

    /// Applies the unused-field policy: an error under `Reject`, `Ok` under `Ignore`.
    fn unused(&self, field: &str, reason: &str) -> Result<()> {
        match self.context.unused_fields {
            UnusedRopeFields::Reject => Err(self.unused_error(field, reason)),
            UnusedRopeFields::Ignore => Ok(()),
        }
    }

    /// Builds the error [`Self::unused`] returns under [`UnusedRopeFields::Reject`].
    fn unused_error(&self, field: &str, reason: &str) -> Error {
        self.error(
            field,
            format!(
                "{reason}; remove it, or select UnusedRopeFields::Ignore to resolve it as Transformers 5.2 does"
            ),
        )
    }

    /// Reports a field the selected algorithm requires but the config omits.
    fn required<T>(&self, value: Option<T>, field: &str, rope_type: RopeType) -> Result<T> {
        value
            .ok_or_else(|| self.error(field, format!("is required by {} RoPE", rope_type.as_str())))
    }

    /// Maps a validated-construction failure to the field a caller can repair.
    ///
    /// A width constraint on a rotary dimension equal to the whole head (no
    /// partial factor, a whole-head family, or proportional RoPE) is reported
    /// at `head_dim`, matching [`RopeParameters::validate_for`]; on a strict
    /// prefix it is reported at `partial_rotary_factor`.
    fn rope_error(&self, error: RopeConfigError) -> Error {
        let field = match error {
            RopeConfigError::ZeroHeadDimension => {
                return self.context_error("head_dim", error.to_string());
            }
            RopeConfigError::OddRotaryDimension { rotary_dimension }
            | RopeConfigError::DynamicNtkDimensionTooSmall { rotary_dimension }
                if rotary_dimension == self.context.head_dim =>
            {
                return self.context_error("head_dim", error.to_string());
            }
            RopeConfigError::ZeroRotaryDimension
            | RopeConfigError::RotaryDimensionExceedsHead { .. }
            | RopeConfigError::OddRotaryDimension { .. }
            | RopeConfigError::InvalidPartialRotaryFactor { .. }
            | RopeConfigError::NonIntegralPartialRotaryDimension { .. }
            | RopeConfigError::DynamicNtkDimensionTooSmall { .. }
            | RopeConfigError::ProportionalRequiresFullRotation { .. }
            | RopeConfigError::InvalidProportionalFraction { .. } => "partial_rotary_factor",
            RopeConfigError::InvalidTheta { .. } | RopeConfigError::YarnThetaTooSmall { .. } => {
                "rope_theta"
            }
            RopeConfigError::InvalidScalingFactor { .. } => "factor",
            RopeConfigError::ZeroOriginalContext { .. } => "original_max_position_embeddings",
            RopeConfigError::InvalidYarnAttentionFactor { .. }
            | RopeConfigError::InvalidLongRopeAttentionFactor { .. } => "attention_factor",
            RopeConfigError::InvalidYarnBeta { name, .. } => name,
            RopeConfigError::InvalidYarnBetaOrder { .. } => "beta_fast",
            RopeConfigError::InvalidLlama3FrequencyFactors { .. } => "low_freq_factor",
            RopeConfigError::InvalidLongRopeFactor { table, .. } => table,
            RopeConfigError::LongRopeFactorLengthMismatch { .. } => "long_factor",
            RopeConfigError::LongRopeFactorCountOutOfRange { .. }
            | RopeConfigError::LongRopeFactorCount { .. } => "short_factor",
        };
        self.error(field, error.to_string())
    }
}

/// Fields of [`RopeParameters`] that only some algorithms read.
///
/// `rope_type`, `rope_theta`, and `partial_rotary_factor` apply to every
/// algorithm and are resolved separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlgorithmField {
    /// `factor`.
    Factor,
    /// `original_max_position_embeddings`.
    OriginalMaxPositionEmbeddings,
    /// `attention_factor`.
    AttentionFactor,
    /// `beta_fast`.
    BetaFast,
    /// `beta_slow`.
    BetaSlow,
    /// `mscale`.
    Mscale,
    /// `mscale_all_dim`.
    MscaleAllDim,
    /// `truncate`.
    Truncate,
    /// `short_factor`.
    ShortFactor,
    /// `long_factor`.
    LongFactor,
    /// `low_freq_factor`.
    LowFreqFactor,
    /// `high_freq_factor`.
    HighFreqFactor,
}

impl AlgorithmField {
    /// Every algorithm-specific field, in serialized-schema order.
    const ALL: [Self; 12] = [
        Self::Factor,
        Self::OriginalMaxPositionEmbeddings,
        Self::AttentionFactor,
        Self::BetaFast,
        Self::BetaSlow,
        Self::Mscale,
        Self::MscaleAllDim,
        Self::Truncate,
        Self::ShortFactor,
        Self::LongFactor,
        Self::LowFreqFactor,
        Self::HighFreqFactor,
    ];

    /// Returns the serialized key.
    const fn name(self) -> &'static str {
        match self {
            Self::Factor => "factor",
            Self::OriginalMaxPositionEmbeddings => "original_max_position_embeddings",
            Self::AttentionFactor => "attention_factor",
            Self::BetaFast => "beta_fast",
            Self::BetaSlow => "beta_slow",
            Self::Mscale => "mscale",
            Self::MscaleAllDim => "mscale_all_dim",
            Self::Truncate => "truncate",
            Self::ShortFactor => "short_factor",
            Self::LongFactor => "long_factor",
            Self::LowFreqFactor => "low_freq_factor",
            Self::HighFreqFactor => "high_freq_factor",
        }
    }

    /// Returns whether the field is present in `parameters`.
    fn is_present(self, parameters: &RopeParameters) -> bool {
        // Destructure without `..`: a new `RopeParameters` field is a compile
        // error here until it is classified, so it cannot bypass the
        // unused-field check by accident.
        let RopeParameters {
            // Resolved for every algorithm by `translate`.
            rope_type: _,
            rope_theta: _,
            partial_rotary_factor: _,
            // Governed by `MropeHandling` in `translate`.
            mrope_section: _,
            mrope_interleaved: _,
            // Algorithm-specific fields, one `AlgorithmField` each.
            factor,
            original_max_position_embeddings,
            attention_factor,
            beta_fast,
            beta_slow,
            mscale,
            mscale_all_dim,
            truncate,
            short_factor,
            long_factor,
            low_freq_factor,
            high_freq_factor,
        } = parameters;
        match self {
            Self::Factor => factor.is_some(),
            Self::OriginalMaxPositionEmbeddings => original_max_position_embeddings.is_some(),
            Self::AttentionFactor => attention_factor.is_some(),
            Self::BetaFast => beta_fast.is_some(),
            Self::BetaSlow => beta_slow.is_some(),
            Self::Mscale => mscale.is_some(),
            Self::MscaleAllDim => mscale_all_dim.is_some(),
            Self::Truncate => truncate.is_some(),
            Self::ShortFactor => short_factor.is_some(),
            Self::LongFactor => long_factor.is_some(),
            Self::LowFreqFactor => low_freq_factor.is_some(),
            Self::HighFreqFactor => high_freq_factor.is_some(),
        }
    }

    /// Returns whether `rope_type` may read this field (Transformers' required and optional keys).
    ///
    /// `softmax` adds `mscale_all_dim` to every scaled algorithm when the
    /// family's softmax scale reads it. YaRN's `mscale`/`mscale_all_dim` are
    /// read only in some combinations; `yarn_amplitude` rejects the others.
    const fn is_read_by(self, rope_type: RopeType, softmax: SoftmaxMscale) -> bool {
        if matches!(self, Self::MscaleAllDim)
            && matches!(softmax, SoftmaxMscale::MscaleAllDimSquared)
            && !matches!(rope_type, RopeType::Default)
        {
            return true;
        }
        match rope_type {
            RopeType::Default => false,
            RopeType::Linear | RopeType::Proportional => matches!(self, Self::Factor),
            // `original_max_position_embeddings` is checked against the
            // dynamic-NTK context rule separately.
            RopeType::Dynamic => matches!(self, Self::Factor | Self::OriginalMaxPositionEmbeddings),
            RopeType::Yarn => matches!(
                self,
                Self::Factor
                    | Self::OriginalMaxPositionEmbeddings
                    | Self::AttentionFactor
                    | Self::BetaFast
                    | Self::BetaSlow
                    | Self::Mscale
                    | Self::MscaleAllDim
                    | Self::Truncate
            ),
            RopeType::LongRope => matches!(
                self,
                Self::Factor
                    | Self::OriginalMaxPositionEmbeddings
                    | Self::AttentionFactor
                    | Self::ShortFactor
                    | Self::LongFactor
            ),
            RopeType::Llama3 => matches!(
                self,
                Self::Factor
                    | Self::OriginalMaxPositionEmbeddings
                    | Self::LowFreqFactor
                    | Self::HighFreqFactor
            ),
        }
    }
}

/// Resolves a value that may be serialized both nested and at the top level.
///
/// Equal values or a single source are accepted. Disagreeing values are an
/// error under [`UnusedRopeFields::Reject`]; under `Ignore`, `nested_wins`
/// selects Transformers' choice for that field.
fn resolve_duplicate<T: PartialEq + Copy + std::fmt::Display>(
    scope: &Scope<'_>,
    field: &str,
    nested: Option<T>,
    top_level: Option<T>,
    nested_wins: bool,
) -> Result<Option<T>> {
    match (nested, top_level) {
        (Some(nested), Some(top_level)) if nested != top_level => {
            scope.unused(
                field,
                &format!(
                    "rope_parameters value {nested} disagrees with top-level value {top_level}"
                ),
            )?;
            Ok(Some(if nested_wins { nested } else { top_level }))
        }
        (nested, top_level) => Ok(nested.or(top_level)),
    }
}

impl RopeParameters {
    /// Translates these parameters into a validated [`RopeConfig`].
    ///
    /// `context` supplies the surrounding config fields and family
    /// conventions; the [`rope`](crate::models::config::rope) module
    /// documentation gives the per-algorithm formulas, the precedence table,
    /// and the strictness policy. Families whose config omits
    /// `rope_parameters` translate `RopeParameters::default()`, which is
    /// default RoPE with theta from the context.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] at `architecture.rope.<field>` (or the
    /// top-level `architecture.head_dim`/`max_position_embeddings`) when a
    /// required field is missing, a value is out of range, two sources
    /// disagree, a parsed field is unused under [`UnusedRopeFields::Reject`],
    /// or the softmax correction declared by
    /// [`SoftmaxMscale::MscaleAllDimSquared`] is invalid (see
    /// [`Self::softmax_scale_multiplier`]).
    pub fn to_rope_config(&self, context: &RopeContext) -> Result<RopeConfig> {
        translate(
            self,
            &Scope {
                context,
                prefix: "rope".to_owned(),
                per_layer_type: false,
            },
        )
    }

    /// Returns the multiplier the family applies to its softmax scale `1/sqrt(d_qk)`.
    ///
    /// Under [`SoftmaxMscale::MscaleAllDimSquared`] this is
    /// `m(f, mscale_all_dim)^2` for every non-default `rope_type` with a
    /// nonzero `mscale_all_dim`, exactly as DeepSeek's reference attention
    /// (`softmax_scale` whenever `rope_scaling` is set) and Transformers'
    /// `deepseek_v3` (`scaling` whenever `rope_type != "default"`) compute it;
    /// it compensates the amplitude ratio applied to the rotary sub-head. In
    /// every other case, including [`SoftmaxMscale::Unscaled`], it is one, so a
    /// family may call it unconditionally and let its context decide.
    ///
    /// # Errors
    ///
    /// Returns a configuration error at `architecture.rope.factor` when the
    /// term applies but `factor` is missing or not finite and at least one, and
    /// at `architecture.rope.mscale_all_dim` when `m(f, mscale_all_dim)` is
    /// not finite and positive or its square, the returned multiplier, does
    /// not round to a normal positive `f32` (attention applies the scale to
    /// `f32` or narrower scores). The square is checked, not only `m`:
    /// `mscale_all_dim = 1e300` gives a finite `m` but `m^2 = inf`.
    pub fn softmax_scale_multiplier(&self, context: &RopeContext) -> Result<f64> {
        softmax_scale_multiplier(
            self,
            &Scope {
                context,
                prefix: "rope".to_owned(),
                per_layer_type: false,
            },
        )
    }
}

/// Computes [`RopeParameters::softmax_scale_multiplier`] with errors at `scope`.
fn softmax_scale_multiplier(parameters: &RopeParameters, scope: &Scope<'_>) -> Result<f64> {
    if scope.context.softmax_mscale == SoftmaxMscale::Unscaled
        || parameters.rope_type == RopeType::Default
    {
        return Ok(1.0);
    }
    // The references test `if mscale_all_dim:`, so an absent or zero value
    // leaves the softmax scale unchanged.
    let Some(multiplier) = parameters.mscale_all_dim.filter(|value| *value != 0.0) else {
        return Ok(1.0);
    };
    let factor = parameters.factor.ok_or_else(|| {
        scope.error(
            "factor",
            "is required by the softmax correction m(factor, mscale_all_dim)^2",
        )
    })?;
    if !(factor.is_finite() && factor >= 1.0) {
        return Err(scope.error(
            "factor",
            format!(
                "is {factor}, but the softmax correction needs a finite factor of at least one"
            ),
        ));
    }
    let correction = yarn_mscale(factor, multiplier);
    if !(correction.is_finite() && correction > 0.0) {
        return Err(scope.error(
            "mscale_all_dim",
            format!(
                "gives the softmax correction m(factor, mscale_all_dim) = {correction}, which must be finite and positive"
            ),
        ));
    }
    // `powi(2)` keeps the private DeepSeek translator's exact operation.
    let squared = correction.powi(2);
    if !is_normal_positive_f32(squared) {
        return Err(scope.error(
            "mscale_all_dim",
            format!(
                "gives the softmax multiplier m(factor, mscale_all_dim)^2 = {squared:e}, which must be a normal positive f32 value ({:e}..={:e})",
                f32::MIN_POSITIVE,
                f32::MAX
            ),
        ));
    }
    Ok(squared)
}

/// Whether `value` rounds to a normal positive `f32`, in `[f32::MIN_POSITIVE, f32::MAX]`.
///
/// The softmax multiplier scales attention scores computed in `f32` (or a
/// narrower dtype), so its `f32` rounding must neither overflow nor vanish.
/// The comparison rounds exactly as the conversion does; NaN fails it.
fn is_normal_positive_f32(value: f64) -> bool {
    let rounded = value as f32;
    rounded.is_finite() && rounded >= f32::MIN_POSITIVE
}

/// Resolves every source and builds the validated configuration.
fn translate(parameters: &RopeParameters, scope: &Scope<'_>) -> Result<RopeConfig> {
    let context = scope.context;
    if context.head_dim == 0 {
        return Err(scope.context_error("head_dim", "must be nonzero"));
    }
    if context.max_position_embeddings == 0 {
        return Err(scope.context_error("max_position_embeddings", "must be nonzero"));
    }
    let rope_type = parameters.rope_type;
    for field in AlgorithmField::ALL {
        if field.is_present(parameters) && !field.is_read_by(rope_type, context.softmax_mscale) {
            scope.unused(
                field.name(),
                &format!("is not read by {} RoPE", rope_type.as_str()),
            )?;
        }
    }
    if context.mrope == MropeHandling::Reject {
        for (field, present) in [
            ("mrope_section", parameters.mrope_section.is_some()),
            ("mrope_interleaved", parameters.mrope_interleaved.is_some()),
        ] {
            if present {
                scope.unused(
                    field,
                    "describes multimodal position axes the caller does not consume (see MropeHandling)",
                )?;
            }
        }
    }

    let theta = match context.layer_rope_theta {
        Some(theta) => theta,
        None => resolve_duplicate(
            scope,
            "rope_theta",
            parameters.rope_theta,
            context.rope_theta,
            true,
        )?
        .unwrap_or(context.default_rope_theta),
    };
    let partial = resolve_duplicate(
        scope,
        "partial_rotary_factor",
        parameters.partial_rotary_factor,
        context.partial_rotary_factor,
        false,
    )?
    .unwrap_or(1.0);
    // Only algorithms with a trained context read the original length, so its
    // sources are resolved (and checked for agreement) on demand. The
    // precedence is the Verus-verified `resolve_original_context`.
    let original_or_max = || -> Result<usize> {
        resolve_original_context(
            parameters.original_max_position_embeddings,
            context.original_max_position_embeddings,
            context.max_position_embeddings,
            scope.per_layer_type,
            context.unused_fields == UnusedRopeFields::Reject,
        )
        .map_err(|fault| match fault {
            OriginalContextFault::NestedZero | OriginalContextFault::Zero => {
                scope.error("original_max_position_embeddings", "must be nonzero")
            }
            OriginalContextFault::LayerConflict {
                top_level,
                resolved,
            } => scope.unused_error(
                "original_max_position_embeddings",
                &format!(
                    "top-level value {top_level} is not read for a per-layer-type mapping, whose layer uses {resolved} (Transformers 5.2 standardize_rope_params)"
                ),
            ),
            OriginalContextFault::FlatConflict { nested, top_level } => scope.unused_error(
                "original_max_position_embeddings",
                &format!(
                    "rope_parameters value {nested} disagrees with top-level value {top_level}"
                ),
            ),
        })
    };

    let rotary_dimension = if rope_type == RopeType::Proportional {
        // Proportional RoPE always spans the head; `partial` is its pair fraction.
        context.head_dim
    } else if context.partial_rotary == PartialRotary::FullHeadOnly && partial != 1.0 {
        scope.unused(
            "partial_rotary_factor",
            &format!(
                "is {partial}, but {} attention rotates whole heads",
                context.architecture
            ),
        )?;
        context.head_dim
    } else {
        rotary_dimension_from_partial_factor(context.head_dim, partial)
            .map_err(|error| scope.rope_error(error))?
    };

    let scaling = match rope_type {
        RopeType::Default => RopeScaling::None,
        RopeType::Linear => {
            LinearRopeScaling::new(scope.required(parameters.factor, "factor", rope_type)?)
                .map_err(|error| scope.rope_error(error))?
                .into()
        }
        RopeType::Dynamic => {
            let factor = scope.required(parameters.factor, "factor", rope_type)?;
            let trained = match context.dynamic_ntk_context {
                DynamicNtkContext::MaxPositionEmbeddings => {
                    // A top-level original context is a config attribute (Phi-3
                    // always has one), so only the nested key counts as unused.
                    if let Some(original) = parameters.original_max_position_embeddings
                        && original != context.max_position_embeddings
                    {
                        scope.unused(
                            "original_max_position_embeddings",
                            &format!(
                                "is {original}, but Transformers dynamic NTK scales against max_position_embeddings ({}); select DynamicNtkContext::OriginalMaxPositionEmbeddings to use it",
                                context.max_position_embeddings
                            ),
                        )?;
                    }
                    context.max_position_embeddings
                }
                DynamicNtkContext::OriginalMaxPositionEmbeddings => original_or_max()?,
            };
            DynamicNtkScaling::new(factor, trained)
                .map_err(|error| scope.rope_error(error))?
                .into()
        }
        RopeType::Yarn => yarn_scaling(parameters, scope, original_or_max()?)?.into(),
        RopeType::LongRope => {
            let short = scope.required(
                parameters.short_factor.as_deref(),
                "short_factor",
                rope_type,
            )?;
            let long =
                scope.required(parameters.long_factor.as_deref(), "long_factor", rope_type)?;
            let original = original_or_max()?;
            // Phi-3 omits `factor` and extends `max_position_embeddings` instead.
            let factor = match parameters.factor {
                Some(factor) => factor,
                None if original > context.max_position_embeddings => {
                    return Err(scope.error(
                        "original_max_position_embeddings",
                        format!(
                            "exceeds max_position_embeddings ({}), so the implied LongRoPE factor max_position_embeddings / original_max_position_embeddings is below one",
                            context.max_position_embeddings
                        ),
                    ));
                }
                None => context.max_position_embeddings as f64 / original as f64,
            };
            let mut long_rope = LongRopeScaling::new(short, long, original, factor)
                .map_err(|error| scope.rope_error(error))?;
            if let Some(attention_factor) = parameters.attention_factor {
                long_rope = long_rope
                    .with_attention_factor(attention_factor)
                    .map_err(|error| scope.rope_error(error))?;
            }
            long_rope.into()
        }
        RopeType::Llama3 => Llama3RopeScaling::new(
            scope.required(parameters.factor, "factor", rope_type)?,
            scope.required(parameters.low_freq_factor, "low_freq_factor", rope_type)?,
            scope.required(parameters.high_freq_factor, "high_freq_factor", rope_type)?,
            original_or_max()?,
        )
        .map_err(|error| scope.rope_error(error))?
        .into(),
        RopeType::Proportional => {
            ProportionalRopeScaling::new(partial, parameters.factor.unwrap_or(1.0))
                .map_err(|error| scope.rope_error(error))?
                .into()
        }
    };
    let config = RopeConfig::new(context.head_dim, rotary_dimension, theta, scaling)
        .map_err(|error| scope.rope_error(error))?;
    // Validate the family's softmax correction now, so a family that applies
    // it cannot fail after translation succeeded.
    softmax_scale_multiplier(parameters, scope)?;
    Ok(config)
}

/// Builds YaRN scaling: factor, original context, amplitude, betas, and truncation.
fn yarn_scaling(
    parameters: &RopeParameters,
    scope: &Scope<'_>,
    original: usize,
) -> Result<YarnConfig> {
    let factor = scope.required(parameters.factor, "factor", RopeType::Yarn)?;
    let mut yarn = YarnConfig::new(factor, original).map_err(|error| scope.rope_error(error))?;
    if let Some((amplitude, source)) = yarn_amplitude(parameters, scope, factor)? {
        yarn = yarn.with_attention_factor(amplitude).map_err(|error| {
            let origin = if source == "mscale" {
                "the mscale / mscale_all_dim ratio"
            } else {
                source
            };
            scope.error(source, format!("{error} (derived from {origin})"))
        })?;
    }
    yarn = yarn
        .with_beta_overrides(parameters.beta_fast, parameters.beta_slow)
        .map_err(|error| scope.rope_error(error))?;
    let truncate = match parameters.truncate {
        // Transformers 5.2 reads `truncate` from the outer mapping only, so a
        // per-layer-type value never reaches its YaRN implementation.
        Some(false) if scope.per_layer_type => {
            scope.unused(
                "truncate",
                "is inside a per-layer-type mapping, where Transformers 5.2 never reads it",
            )?;
            true
        }
        truncate => truncate.unwrap_or(true),
    };
    Ok(yarn.with_truncate(truncate))
}

/// Resolves the YaRN amplitude and the field it came from.
///
/// `None` keeps [`YarnConfig`]'s default `m(f, 1)`; see the module docs for
/// both conventions.
fn yarn_amplitude(
    parameters: &RopeParameters,
    scope: &Scope<'_>,
    factor: f64,
) -> Result<Option<(f64, &'static str)>> {
    let ratio = |mscale: f64, mscale_all_dim: f64| {
        yarn_mscale(factor, mscale) / yarn_mscale(factor, mscale_all_dim)
    };
    match scope.context.mscale_convention {
        YarnMscaleConvention::HuggingFace => {
            // `mscale_all_dim` the amplitude does not read is still read by a
            // softmax correction the family declared.
            let unread_all_dim = |reason: &str| -> Result<()> {
                if parameters.mscale_all_dim.is_some()
                    && scope.context.softmax_mscale == SoftmaxMscale::Unscaled
                {
                    scope.unused(
                        "mscale_all_dim",
                        &format!(
                            "{reason}, and this family's softmax scale does not read it (see SoftmaxMscale and YarnMscaleConvention::DeepSeekReference)"
                        ),
                    )?;
                }
                Ok(())
            };
            if let Some(attention_factor) = parameters.attention_factor {
                if parameters.mscale.is_some() {
                    scope.unused("mscale", "is overridden by attention_factor")?;
                }
                unread_all_dim("is overridden by attention_factor in the YaRN amplitude")?;
                return Ok(Some((attention_factor, "attention_factor")));
            }
            match (parameters.mscale, parameters.mscale_all_dim) {
                (Some(mscale), Some(all)) if mscale != 0.0 && all != 0.0 => {
                    Ok(Some((ratio(mscale, all), "mscale")))
                }
                (Some(_), _) => {
                    scope.unused(
                        "mscale",
                        "is read by the Hugging Face YaRN amplitude only together with a nonzero mscale_all_dim (see YarnMscaleConvention::DeepSeekReference)",
                    )?;
                    // Reached only under `Ignore`, where an unread
                    // `mscale_all_dim` is ignored as well.
                    Ok(None)
                }
                (None, _) => {
                    unread_all_dim(
                        "is read by the Hugging Face YaRN amplitude only together with a nonzero mscale",
                    )?;
                    Ok(None)
                }
            }
        }
        YarnMscaleConvention::DeepSeekReference => {
            if parameters.mscale.is_none() && parameters.mscale_all_dim.is_none() {
                return Ok(parameters
                    .attention_factor
                    .map(|attention_factor| (attention_factor, "attention_factor")));
            }
            if parameters.attention_factor.is_some() {
                scope.unused(
                    "attention_factor",
                    "is not read by the DeepSeek reference, which derives the amplitude from mscale and mscale_all_dim",
                )?;
            }
            Ok(Some((
                ratio(
                    parameters.mscale.unwrap_or(1.0),
                    parameters.mscale_all_dim.unwrap_or(0.0),
                ),
                "mscale",
            )))
        }
    }
}

// ---------------------------------------------------------------------------
// Per-layer-type mappings.
// ---------------------------------------------------------------------------

/// RoPE parameters keyed by attention layer type, as in Gemma 3 and Gemma 4.
///
/// Transformers 5.x stores `{"sliding_attention": {...}, "full_attention":
/// {...}}` when layer types use different bases or algorithms. Each value is a
/// complete [`RopeParameters`] object; entries are kept in key order.
///
/// The mapping is never empty: an empty JSON object is uniform default RoPE
/// (see [`RopeParameterSet`]), so an empty per-layer mapping could not
/// survive a serialization round trip.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct RopeParametersByLayerType(
    /// Parameters for each serialized layer-type name; never empty.
    BTreeMap<String, RopeParameters>,
);

/// Reason an empty per-layer-type mapping is rejected.
const EMPTY_LAYER_MAPPING: &str = "a per-layer-type mapping needs at least one layer type; an empty object is uniform default RoPE";

impl RopeParametersByLayerType {
    /// Wraps an explicit, non-empty layer-type map.
    ///
    /// # Errors
    ///
    /// Returns a configuration error at `rope_parameters` when `parameters`
    /// is empty, because `{}` deserializes as uniform default RoPE.
    pub fn new(parameters: BTreeMap<String, RopeParameters>) -> Result<Self> {
        if parameters.is_empty() {
            return Err(Error::configuration("rope_parameters", EMPTY_LAYER_MAPPING));
        }
        Ok(Self(parameters))
    }

    /// Returns the parameters for one layer type, if present.
    #[must_use]
    pub fn get(&self, layer_type: &str) -> Option<&RopeParameters> {
        self.0.get(layer_type)
    }

    /// Visits every layer type and its parameters in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &RopeParameters)> {
        self.0
            .iter()
            .map(|(layer_type, parameters)| (layer_type.as_str(), parameters))
    }

    /// Translates the parameters of `layer_type`.
    ///
    /// Errors are reported at `architecture.rope.<layer_type>.<field>`.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when the layer type has no entry or the
    /// entry fails [`RopeParameters::to_rope_config`]. Two Transformers 5.2
    /// quirks of per-layer mappings are unused fields: a YaRN entry with
    /// `truncate: false` (only a flat mapping's `truncate` is read), and a
    /// top-level `original_max_position_embeddings` that differs from the
    /// layer's own value or `max_position_embeddings` (never read here).
    pub fn to_rope_config(&self, layer_type: &str, context: &RopeContext) -> Result<RopeConfig> {
        let parameters = self.get(layer_type).ok_or_else(|| {
            Error::configuration(
                format!("{}.rope.{layer_type}", context.architecture),
                format!(
                    "no RoPE parameters for this layer type; mapping has {:?}",
                    self.0.keys().collect::<Vec<_>>()
                ),
            )
        })?;
        translate(
            parameters,
            &Scope {
                context,
                prefix: format!("rope.{layer_type}"),
                per_layer_type: true,
            },
        )
    }
}

impl<'de> Deserialize<'de> for RopeParametersByLayerType {
    /// Reads a layer-type map, rejecting an empty one (see the type documentation).
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let parameters = BTreeMap::<String, RopeParameters>::deserialize(deserializer)?;
        if parameters.is_empty() {
            return Err(D::Error::custom(EMPTY_LAYER_MAPPING));
        }
        Ok(Self(parameters))
    }
}

/// Either one RoPE mapping for every layer or a per-layer-type mapping.
///
/// This is the Transformers 5.x `rope_parameters` type
/// `RopeParameters | dict[layer_type, RopeParameters]`. Deserialization picks
/// [`Self::ByLayerType`] exactly when the JSON object is non-empty and every
/// value is an object (no flat RoPE key holds an object), rejects objects that
/// mix both shapes, and otherwise parses [`Self::Uniform`]. An empty object is
/// uniform default RoPE, as in Transformers. Any value that is not an object,
/// such as `[]` or `["yarn"]`, is rejected: Transformers reads a dict.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
#[allow(
    clippy::large_enum_variant,
    reason = "parsed once per model config; boxing the common uniform case would only complicate matching"
)]
pub enum RopeParameterSet {
    /// One mapping shared by every attention layer.
    Uniform(
        /// Parameters applied to every layer type.
        RopeParameters,
    ),
    /// Distinct mappings per attention layer type.
    ByLayerType(
        /// Parameters keyed by layer-type name.
        RopeParametersByLayerType,
    ),
}

impl Default for RopeParameterSet {
    /// Uniform default RoPE.
    fn default() -> Self {
        Self::Uniform(RopeParameters::default())
    }
}

impl From<RopeParameters> for RopeParameterSet {
    /// Shares one mapping across all layer types.
    fn from(parameters: RopeParameters) -> Self {
        Self::Uniform(parameters)
    }
}

impl From<RopeParametersByLayerType> for RopeParameterSet {
    /// Uses distinct mappings per layer type.
    fn from(parameters: RopeParametersByLayerType) -> Self {
        Self::ByLayerType(parameters)
    }
}

impl<'de> Deserialize<'de> for RopeParameterSet {
    /// Detects the per-layer-type shape structurally; see the type documentation.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        Self::from_json_value(value).map_err(D::Error::custom)
    }
}

impl RopeParameterSet {
    /// Parses a JSON value with the shape detection described on the type.
    ///
    /// Only objects are accepted: a non-object value reaches the flat
    /// [`RopeParameters`] deserializer, which requires a mapping, so `[]`,
    /// `["yarn"]`, strings, and numbers fail with a type error instead of
    /// being read as positional fields.
    fn from_json_value(value: serde_json::Value) -> std::result::Result<Self, String> {
        match value {
            serde_json::Value::Object(map)
                if !map.is_empty() && map.values().all(serde_json::Value::is_object) =>
            {
                map.into_iter()
                    .map(|(layer_type, entry)| {
                        serde_json::from_value(entry)
                            .map(|parameters| (layer_type.clone(), parameters))
                            .map_err(|error| format!("rope_parameters.{layer_type}: {error}"))
                    })
                    .collect::<std::result::Result<BTreeMap<_, _>, _>>()
                    .map(|map| Self::ByLayerType(RopeParametersByLayerType(map)))
            }
            serde_json::Value::Object(map) if map.values().any(serde_json::Value::is_object) => {
                let (objects, scalars): (Vec<&String>, Vec<&String>) =
                    map.keys().partition(|key| map[key.as_str()].is_object());
                Err(format!(
                    "rope_parameters mixes per-layer-type objects {objects:?} with flat RoPE fields {scalars:?}"
                ))
            }
            flat => serde_json::from_value(flat)
                .map(Self::Uniform)
                .map_err(|error| error.to_string()),
        }
    }

    /// Returns the parameters that apply to `layer_type`.
    ///
    /// A uniform mapping applies to every layer type.
    #[must_use]
    pub fn parameters_for(&self, layer_type: &str) -> Option<&RopeParameters> {
        match self {
            Self::Uniform(parameters) => Some(parameters),
            Self::ByLayerType(by_type) => by_type.get(layer_type),
        }
    }

    /// Translates the parameters that apply to `layer_type`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`RopeParameters::to_rope_config`] or
    /// [`RopeParametersByLayerType::to_rope_config`].
    pub fn to_rope_config(&self, layer_type: &str, context: &RopeContext) -> Result<RopeConfig> {
        match self {
            Self::Uniform(parameters) => parameters.to_rope_config(context),
            Self::ByLayerType(by_type) => by_type.to_rope_config(layer_type, context),
        }
    }

    /// Checks a per-layer-type mapping against the config's `layer_types`.
    ///
    /// Every used layer type needs an entry (Transformers raises `KeyError`
    /// otherwise), and every entry must belong to a used layer type: a stray
    /// key would make Transformers treat the whole mapping as flat. Uniform
    /// mappings always pass.
    ///
    /// # Errors
    ///
    /// Returns a configuration error at `architecture.rope.<layer_type>`.
    pub fn validate_layer_types<'a>(
        &self,
        architecture: &'static str,
        layer_types: impl IntoIterator<Item = &'a str>,
    ) -> Result<()> {
        let Self::ByLayerType(by_type) = self else {
            return Ok(());
        };
        let used = layer_types.into_iter().collect::<BTreeSet<_>>();
        for layer_type in &used {
            if by_type.get(layer_type).is_none() {
                return Err(Error::configuration(
                    format!("{architecture}.rope.{layer_type}"),
                    "layer_types uses this layer type, but rope_parameters has no entry for it",
                ));
            }
        }
        for (layer_type, _) in by_type.iter() {
            if !used.contains(layer_type) {
                return Err(Error::configuration(
                    format!("{architecture}.rope.{layer_type}"),
                    "no layer in layer_types has this type, so its RoPE parameters would be unused",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "rope_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "rope_golden_tests.rs"]
mod golden_tests;
