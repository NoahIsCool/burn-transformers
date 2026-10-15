//! Serializable specifications for assembling experimental transformer graphs.
//!
//! These types describe topology rather than checkpoint families. A layer may
//! select a sequence mixer, feed-forward operator, normalization placement, and
//! residual topology independently. Exact family models may bypass this surface
//! when a checkpoint quirk cannot be represented faithfully.
//!
//! # Compact and explicit forms
//!
//! The original compact variants ([`MixerSpec::Attention`],
//! [`MixerSpec::SlidingAttention`], [`FeedForwardSpec::Dense`]) keep their
//! serialized form and meaning. Mechanisms added later are spelled out through
//! explicit variants and serde-defaulted fields, so every previously serialized
//! specification still loads and builds the same graph:
//!
//! | mechanism | where it is expressed |
//! | --- | --- |
//! | score scale, logit soft cap, learned sinks, per-projection biases, Q/K-norm scope, SiLU output gate, global / sliding / chunked-local connectivity | [`MixerSpec::AttentionWithOptions`] and [`AttentionOptions`] |
//! | MLA score scale and connectivity | [`MlaSpec::score_scale`], [`MlaSpec::connectivity`] |
//! | differential attention (DIFF Transformer) | [`MixerSpec::DifferentialAttention`] |
//! | every shared RoPE scaling rule (linear, dynamic NTK, YaRN, Llama 3, LongRoPE, proportional) | [`RopeScaling`] |
//! | ungated MLP, clamped SwiGLU, attention-only blocks | [`FeedForwardSpec::Ungated`], [`FeedForwardSpec::ClampedSwiGlu`], [`FeedForwardSpec::None`] |
//! | router policy, shared-expert gate, expert biases | [`MixtureOfExpertsSpec`] |
//! | cross-attention scale, soft cap, sinks, biases | [`Seq2SeqSpec::cross_attention_options`], [`CrossAttentionSpec`] |
//!
//! # Roles of the stack types
//!
//! [`DecoderSpec`] and [`DecoderLayerSpec`] predate encoder-decoder support and
//! keep their names for compatibility, but they are role-neutral: a
//! [`Seq2SeqSpec`] uses a `DecoderSpec` for its bidirectional encoder stack
//! too. The *stack* decides the default attention direction (causal for
//! decoders, bidirectional for encoders); the block specification only
//! describes the computation.
//!
//! # One source of truth for validation and construction
//!
//! Every spec translates into the shared runtime configuration it describes:
//! [`RopeConfig`], [`T5RelativePositionBuckets`], and the grouped-query,
//! latent-attention, Gated DeltaNet, sparse-expert, and feed-forward configs.
//! Validation builds and validates those objects (using the block's real
//! residual width and mixer normalization) and maps their errors to spec field
//! paths, and the modular models build their modules from the very same
//! translations, so a spec that validates can be constructed. A property test
//! checks validation against direct construction of each component in both
//! directions.
//!
//! `validate` checks that a spec is well formed. A few well-formed variants
//! ([`ResidualTopology::Gated`], [`MixerSpec::Mamba2`]) are describable but not
//! yet executable by the modular models, whose `init` rejects them with a
//! capability error before allocating parameters.

use super::constraints::{
    ExpertCheck, HeadCheck, MlaCheck, StackCheck, attention_heads, cross_projection_fits,
    delta_net_heads, expert_counts, mamba2_dimensions, mla_dimensions, mla_qk_head_dim,
    stack_dimensions,
};
use crate::{
    Error, Result,
    attention::{
        AttentionConnectivity, AttentionOutputGate, DifferentialAttentionConfig,
        GroupedQueryAttentionConfig, ProjectionBias, QueryKeyNormScope,
    },
    delta_net::GatedDeltaNetConfig,
    feed_forward::{ClampedSwiGlu, DenseFeedForwardConfig, GatedFeedForwardConfig},
    mla::MultiLatentAttentionConfig,
    moe::{RoutingPolicy, SparseMoeConfig},
    position::{
        DynamicNtkScaling, LinearRopeScaling, Llama3RopeScaling, LongRopeScaling,
        ProportionalRopeScaling, RopeConfig, RopeConfigError, RopeScaling as RuntimeRopeScaling,
        T5BucketError, T5RelativePositionBuckets, YarnConfig,
    },
};

// ---------------------------------------------------------------------------
// Activations and normalization
// ---------------------------------------------------------------------------

/// Nonlinearity applied by a feed-forward expert or gated projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    /// Rectified linear unit `max(0, x)`, used by original T5 feed-forwards.
    Relu,
    /// The SiLU function `x * sigmoid(x)` used by SwiGLU.
    Silu,
    /// The exact Gaussian error linear unit.
    Gelu,
    /// PyTorch's tanh approximation to GELU.
    GeluTanh,
    /// Squared ReLU, `max(0, x)^2`, used by Nemotron-H experts.
    ReluSquared,
}

/// Layer-normalization operation and its numerical convention.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NormalizationSpec {
    /// Root-mean-square normalization without mean subtraction.
    Rms {
        /// Stabilizer added to the mean square before the reciprocal root.
        epsilon: f64,
        /// Whether the learned scale is interpreted as `1 + weight`.
        zero_centered: bool,
    },
    /// Mean-subtracting layer normalization.
    Layer {
        /// Stabilizer added to the variance before the reciprocal root.
        epsilon: f64,
        /// Whether a learned additive bias is present.
        bias: bool,
    },
}

impl NormalizationSpec {
    /// Validates that the variance stabilizer is finite and strictly positive.
    pub fn validate(&self, field: &str) -> Result<()> {
        let epsilon = self.epsilon();
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::configuration(
                format!("{field}.epsilon"),
                "must be finite and greater than zero",
            ));
        }
        Ok(())
    }

    /// Returns the variance stabilizer, which Q/K, latent, and DeltaNet norms reuse.
    #[must_use]
    pub const fn epsilon(&self) -> f64 {
        match self {
            Self::Rms { epsilon, .. } | Self::Layer { epsilon, .. } => *epsilon,
        }
    }

    /// Returns whether learned RMS scales are stored as `scale - 1` (Gemma, Qwen3.5).
    ///
    /// LayerNorm scales are never zero-centered.
    #[must_use]
    pub const fn is_zero_centered(&self) -> bool {
        matches!(
            self,
            Self::Rms {
                zero_centered: true,
                ..
            }
        )
    }
}

/// Placeholder normalization used when validating a mixer outside its block.
///
/// Only its epsilon reaches the shared configs; inside a block the real mixer
/// normalization is used instead and validated separately.
const VALIDATION_NORM: NormalizationSpec = NormalizationSpec::Rms {
    epsilon: 1e-6,
    zero_centered: false,
};

// ---------------------------------------------------------------------------
// Position encodings
// ---------------------------------------------------------------------------

/// Long-context scaling applied to rotary frequencies.
///
/// Each variant translates one-to-one into the shared runtime policy of the
/// same name in [`crate::position`]; see those types for the exact formulas
/// and bounds. Pretraining context lengths are named `original_max_positions`
/// here, matching the earliest serialized YaRN form.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RopeScaling {
    /// Frequencies are used without extrapolation scaling.
    None,
    /// Positions are divided by `factor` before rotation (position interpolation).
    Linear {
        /// Finite position divisor, at least one.
        factor: f64,
    },
    /// The frequency base grows once the evaluated context exceeds the original one.
    DynamicNtk {
        /// Finite context-extension multiplier, at least one.
        factor: f64,
        /// Context length used during pretraining.
        original_max_positions: usize,
    },
    /// Static YaRN interpolation and extrapolation scaling.
    Yarn {
        /// Requested context expansion factor.
        factor: f64,
        /// Maximum sequence length used during pretraining.
        original_max_positions: usize,
        /// Optional cosine/sine amplitude multiplier; `1 + 0.1 ln(factor)` when absent.
        attention_factor: Option<f64>,
        /// Low-frequency transition parameter.
        beta_slow: f64,
        /// High-frequency transition parameter.
        beta_fast: f64,
        /// Whether correction dimensions are floored/ceiled (the reference default).
        #[serde(default = "default_true")]
        truncate: bool,
    },
    /// Llama 3.1 band-wise frequency interpolation.
    Llama3 {
        /// Division applied to long-wavelength (low-frequency) pairs.
        factor: f64,
        /// Multiplier defining the long-wavelength boundary `original / low_freq_factor`.
        low_freq_factor: f64,
        /// Multiplier defining the short-wavelength boundary `original / high_freq_factor`.
        high_freq_factor: f64,
        /// Context length used during pretraining.
        original_max_positions: usize,
    },
    /// Phi-3 LongRoPE searched per-pair frequency divisors.
    LongRope {
        /// Divisors used while the evaluated context is at most the original context.
        short_factors: Vec<f64>,
        /// Divisors used beyond the original context.
        long_factors: Vec<f64>,
        /// Context length used during pretraining.
        original_max_positions: usize,
        /// Context-extension factor used for the default amplitude.
        factor: f64,
        /// Optional explicit cosine/sine amplitude.
        attention_factor: Option<f64>,
    },
    /// Full-head split-half rotation in which only a leading fraction of pairs rotates.
    Proportional {
        /// Fraction of complex pairs that rotate, in `[0, 1]`.
        rotated_fraction: f64,
        /// Position divisor applied to rotated frequencies, at least one.
        factor: f64,
    },
}

/// Serde default for flags whose reference value is `true`.
const fn default_true() -> bool {
    true
}

impl RopeScaling {
    /// Translates this spec into the validated shared runtime policy.
    fn to_runtime(&self) -> core::result::Result<RuntimeRopeScaling, RopeConfigError> {
        Ok(match self {
            Self::None => RuntimeRopeScaling::None,
            Self::Linear { factor } => LinearRopeScaling::new(*factor)?.into(),
            Self::DynamicNtk {
                factor,
                original_max_positions,
            } => DynamicNtkScaling::new(*factor, *original_max_positions)?.into(),
            Self::Yarn {
                factor,
                original_max_positions,
                attention_factor,
                beta_slow,
                beta_fast,
                truncate,
            } => {
                let mut yarn = YarnConfig::new(*factor, *original_max_positions)?
                    .with_betas(*beta_fast, *beta_slow)?
                    .with_truncate(*truncate);
                if let Some(attention_factor) = attention_factor {
                    yarn = yarn.with_attention_factor(*attention_factor)?;
                }
                yarn.into()
            }
            Self::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_positions,
            } => Llama3RopeScaling::new(
                *factor,
                *low_freq_factor,
                *high_freq_factor,
                *original_max_positions,
            )?
            .into(),
            Self::LongRope {
                short_factors,
                long_factors,
                original_max_positions,
                factor,
                attention_factor,
            } => {
                let mut long = LongRopeScaling::new(
                    short_factors,
                    long_factors,
                    *original_max_positions,
                    *factor,
                )?;
                if let Some(attention_factor) = attention_factor {
                    long = long.with_attention_factor(*attention_factor)?;
                }
                long.into()
            }
            Self::Proportional {
                rotated_fraction,
                factor,
            } => ProportionalRopeScaling::new(*rotated_fraction, *factor)?.into(),
        })
    }
}

/// Position information supplied to one mixer layer.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PositionEncodingSpec {
    /// No explicit positional transform is applied.
    None,
    /// Rotary position encoding over an even prefix of every head.
    Rope {
        /// Base period controlling the geometric frequency sequence.
        theta: f64,
        /// Number of head channels rotated in pairs.
        rotary_dim: usize,
        /// Optional long-context scaling policy.
        scaling: RopeScaling,
    },
    /// Learned T5 relative-position buckets added to attention logits.
    ///
    /// The bias does not change an explicit [`ScoreScale`]; only the legacy
    /// [`ScoreScale::Auto`] infers T5's unscaled logits for grouped-query attention.
    T5RelativeBias {
        /// Number of signed/logarithmic distance buckets.
        buckets: usize,
        /// Maximum distance distinguished by logarithmic buckets.
        max_distance: usize,
        /// Whether distances in both directions receive independent buckets.
        bidirectional: bool,
    },
}

impl PositionEncodingSpec {
    /// Validates pairwise rotary dimensions or T5 bucket boundaries.
    ///
    /// Delegates to the shared constructors used at initialization:
    /// [`Self::rope_config`] for RoPE over `head_dim` channels and
    /// [`Self::t5_buckets`] for T5 relative buckets. Errors name the spec field
    /// below `field`, such as `layers[0].mixer.position.rotary_dim`.
    pub fn validate(&self, head_dim: usize, field: &str) -> Result<()> {
        self.rope_config_at(head_dim, field)?;
        self.t5_buckets_at(field)?;
        Ok(())
    }

    /// Builds the shared RoPE policy for heads of `head_dim` channels.
    ///
    /// Returns `None` for non-rotary encodings. Every scaling variant passes
    /// through its shared runtime constructor with the same defaults and bounds
    /// as checkpoint configs (for YaRN: factor at least one,
    /// `0 < beta_slow < beta_fast`, `theta > 1`).
    pub fn rope_config(&self, head_dim: usize) -> Result<Option<RopeConfig>> {
        self.rope_config_at(head_dim, "position")
    }

    /// Builds the shared T5 bucket mapping, or `None` for non-T5 encodings.
    ///
    /// Bidirectional tables need an even bucket count of at least four, and
    /// `max_distance` must exceed the exact region `buckets_per_direction / 2`.
    pub fn t5_buckets(&self) -> Result<Option<T5RelativePositionBuckets>> {
        self.t5_buckets_at("position")
    }

    /// Whether this is the learned T5 bias, which the legacy [`ScoreScale::Auto`] leaves unscaled.
    #[must_use]
    pub const fn is_t5_relative_bias(&self) -> bool {
        matches!(self, Self::T5RelativeBias { .. })
    }

    /// [`Self::rope_config`] with errors reported below an explicit field path.
    fn rope_config_at(&self, head_dim: usize, field: &str) -> Result<Option<RopeConfig>> {
        let Self::Rope {
            theta,
            rotary_dim,
            scaling,
        } = self
        else {
            return Ok(None);
        };
        let at = |error: RopeConfigError| {
            Error::configuration(
                format!("{field}.{}", rope_error_field(&error)),
                error.to_string(),
            )
        };
        let scaling = scaling.to_runtime().map_err(at)?;
        RopeConfig::new(head_dim, *rotary_dim, *theta, scaling)
            .map(Some)
            .map_err(at)
    }

    /// [`Self::t5_buckets`] with errors reported below an explicit field path.
    fn t5_buckets_at(&self, field: &str) -> Result<Option<T5RelativePositionBuckets>> {
        let Self::T5RelativeBias {
            buckets,
            max_distance,
            bidirectional,
        } = self
        else {
            return Ok(None);
        };
        T5RelativePositionBuckets::new(*buckets, *max_distance, *bidirectional)
            .map(Some)
            .map_err(|error| {
                let name = match error {
                    T5BucketError::TooFewBuckets { .. }
                    | T5BucketError::OddBidirectionalBucketCount { .. } => "buckets",
                    T5BucketError::MaxDistanceNotBeyondExact { .. } => "max_distance",
                };
                Error::configuration(format!("{field}.{name}"), error.to_string())
            })
    }
}

/// Maps a shared RoPE construction error to the spec field a caller can repair.
fn rope_error_field(error: &RopeConfigError) -> &'static str {
    match error {
        RopeConfigError::ZeroHeadDimension => "head_dim",
        RopeConfigError::ZeroRotaryDimension
        | RopeConfigError::RotaryDimensionExceedsHead { .. }
        | RopeConfigError::OddRotaryDimension { .. }
        | RopeConfigError::InvalidPartialRotaryFactor { .. }
        | RopeConfigError::NonIntegralPartialRotaryDimension { .. }
        | RopeConfigError::DynamicNtkDimensionTooSmall { .. }
        | RopeConfigError::ProportionalRequiresFullRotation { .. } => "rotary_dim",
        RopeConfigError::InvalidTheta { .. } | RopeConfigError::YarnThetaTooSmall { .. } => "theta",
        RopeConfigError::InvalidScalingFactor { .. } => "scaling.factor",
        RopeConfigError::ZeroOriginalContext { .. } => "scaling.original_max_positions",
        RopeConfigError::InvalidYarnAttentionFactor { .. }
        | RopeConfigError::InvalidLongRopeAttentionFactor { .. } => "scaling.attention_factor",
        RopeConfigError::InvalidYarnBeta { .. } | RopeConfigError::InvalidYarnBetaOrder { .. } => {
            "scaling.beta"
        }
        RopeConfigError::InvalidLlama3FrequencyFactors { .. } => "scaling.frequency_factors",
        RopeConfigError::LongRopeFactorLengthMismatch { .. }
        | RopeConfigError::LongRopeFactorCountOutOfRange { .. }
        | RopeConfigError::InvalidLongRopeFactor { .. }
        | RopeConfigError::LongRopeFactorCount { .. } => "scaling.factors",
        RopeConfigError::InvalidProportionalFraction { .. } => "scaling.rotated_fraction",
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Multiplier `s` applied to query-key dot products before the softmax.
///
/// Attention computes `softmax(s * q.k + bias)`. The Transformer convention is
/// `s = 1/sqrt(d_qk)`; T5 uses `s = 1` and folds `1/sqrt(d)` into its query
/// initialization; Gemma uses `query_pre_attn_scalar^(-1/2)`; DeepSeek YaRN
/// multiplies `1/sqrt(d_qk)` by `mscale^2`. Making the choice explicit keeps a
/// position-encoding experiment from silently changing the softmax temperature.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScoreScale {
    /// The legacy inference, kept so earlier serialized specs build unchanged.
    ///
    /// Grouped-query attention uses `s = 1` with [`PositionEncodingSpec::T5RelativeBias`]
    /// and `1/sqrt(head_dim)` otherwise; MLA always uses `1/sqrt(d_qk)`.
    /// Prefer an explicit variant in new specs.
    #[default]
    Auto,
    /// `s = 1/sqrt(d_qk)`, where `d_qk` is the query-key width per head.
    InverseSqrtHeadDim,
    /// `s = 1`: raw dot products, as in T5 and mT5.
    Unscaled,
    /// An explicit finite positive multiplier.
    Fixed {
        /// The multiplier `s`.
        value: f64,
    },
}

impl ScoreScale {
    /// Rejects a non-positive or non-finite fixed multiplier.
    pub fn validate(self, field: &str) -> Result<()> {
        match self {
            Self::Fixed { value } if !(value.is_finite() && value > 0.0) => {
                Err(Error::configuration(
                    format!("{field}.value"),
                    "must be finite and greater than zero",
                ))
            }
            _ => Ok(()),
        }
    }

    /// Returns the grouped-query `scale` override; `None` keeps the kernel's `1/sqrt(head_dim)`.
    ///
    /// `t5_relative_bias` resolves [`Self::Auto`] to the legacy unscaled logits.
    /// Returning `None` for the conventional scale preserves fused-kernel eligibility.
    #[must_use]
    pub const fn grouped_query_scale(self, t5_relative_bias: bool) -> Option<f64> {
        match self {
            Self::Auto if t5_relative_bias => Some(1.0),
            Self::Auto | Self::InverseSqrtHeadDim => None,
            Self::Unscaled => Some(1.0),
            Self::Fixed { value } => Some(value),
        }
    }

    /// Returns the MLA softmax multiplier for a query-key width of `qk_head_dim` channels.
    #[must_use]
    pub fn latent_scale(self, qk_head_dim: usize) -> f64 {
        match self {
            Self::Auto | Self::InverseSqrtHeadDim => (qk_head_dim as f64).sqrt().recip(),
            Self::Unscaled => 1.0,
            Self::Fixed { value } => value,
        }
    }
}

/// Nonlinearity of an attention output gate requested by [`AttentionSpec::output_gate`].
///
/// The gate projection `g` is emitted alongside the query projection and
/// multiplies each head's attention output elementwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputGateActivation {
    /// `out * sigmoid(g)`, as in Qwen3-Next and Qwen3.5 gated attention.
    #[default]
    Sigmoid,
    /// `out * g * sigmoid(g)`, as in Muse Glimmer.
    Silu,
}

/// Every grouped-query attention choice beyond the compact [`AttentionSpec`].
///
/// All fields default to the compact form's behavior, so
/// `AttentionWithOptions { attention, options: AttentionOptions::default() }`
/// builds exactly the same layer as `Attention(attention)`. Missing JSON
/// fields take these defaults too.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AttentionOptions {
    /// Which keys each query may see: global, a trailing window, or local chunks.
    ///
    /// Windows and chunks count valid tokens, so padding consumes no window.
    /// Sliding windows bound only the past and are therefore rejected in
    /// bidirectional encoder stacks; chunked-local attention is symmetric and
    /// allowed there. An explicit attention policy in an experiment replaces
    /// this structural rule together with the stack's causality.
    pub connectivity: AttentionConnectivity,
    /// Softmax temperature applied to query-key dot products.
    pub score_scale: ScoreScale,
    /// Optional tanh soft cap `c`: scores become `c * tanh(scores / c)` (Gemma 2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_softcap: Option<f64>,
    /// Whether every query head learns a sink logit that absorbs softmax mass (gpt-oss).
    ///
    /// Cannot be combined with [`Self::logit_softcap`]. Modular models build
    /// backend-default attention, whose kernels would also cap the sink logit,
    /// and a serializable spec does not choose the attention precision, so the
    /// combination is rejected during validation.
    pub attention_sinks: bool,
    /// Which of the query, key, value, and output projections carry biases.
    pub projection_bias: ProjectionBias,
    /// Channels normalized by the optional Q/K RMS norm; full width requires `query_key_norm`.
    pub query_key_norm_scope: QueryKeyNormScope,
    /// Nonlinearity of the optional output gate; SiLU requires `output_gate`.
    pub output_gate_activation: OutputGateActivation,
}

impl Default for AttentionOptions {
    /// Global, legacy-scaled, uncapped, sink-free, bias-free attention with per-head Q/K norms.
    fn default() -> Self {
        Self {
            connectivity: AttentionConnectivity::Global,
            score_scale: ScoreScale::Auto,
            logit_softcap: None,
            attention_sinks: false,
            projection_bias: ProjectionBias::uniform(false),
            query_key_norm_scope: QueryKeyNormScope::PerHead,
            output_gate_activation: OutputGateActivation::Sigmoid,
        }
    }
}

impl AttentionOptions {
    /// Rejects malformed values and options that contradict the compact attention flags.
    ///
    /// A full-width Q/K-norm scope without `query_key_norm`, or a SiLU gate
    /// without `output_gate`, would be silently ignored, so both are errors.
    /// Learned sinks with a soft cap are rejected here, at the spec level:
    /// the shared kernel accepts that pair only under an eager precision mode
    /// that modular construction never selects.
    fn validate_against(&self, attention: &AttentionSpec, field: &str) -> Result<()> {
        self.connectivity.validate().map_err(|error| {
            Error::configuration(format!("{field}.connectivity"), reason_of(error))
        })?;
        self.score_scale.validate(&format!("{field}.score_scale"))?;
        if self
            .logit_softcap
            .is_some_and(|cap| !(cap.is_finite() && cap > 0.0))
        {
            return Err(Error::configuration(
                format!("{field}.logit_softcap"),
                "must be finite and greater than zero when present",
            ));
        }
        if self.attention_sinks && self.logit_softcap.is_some() {
            return Err(Error::configuration(
                format!("{field}.attention_sinks"),
                "modular attention cannot combine learned sinks with a logit soft cap: its backend-default kernels would also cap the sink logit",
            ));
        }
        if self.query_key_norm_scope == QueryKeyNormScope::FullWidth && !attention.query_key_norm {
            return Err(Error::configuration(
                format!("{field}.query_key_norm_scope"),
                "a full-width scope requires query_key_norm",
            ));
        }
        if self.output_gate_activation == OutputGateActivation::Silu && !attention.output_gate {
            return Err(Error::configuration(
                format!("{field}.output_gate_activation"),
                "a SiLU gate requires output_gate",
            ));
        }
        Ok(())
    }
}

/// Standard or grouped-query attention topology.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AttentionSpec {
    /// Number of independently projected query heads.
    pub query_heads: usize,
    /// Number of key/value heads retained in the cache.
    pub key_value_heads: usize,
    /// Width of one query/key/value head.
    pub head_dim: usize,
    /// Whether Q and K are RMS-normalized before their dot product.
    ///
    /// The norm reuses the block's mixer epsilon and zero-centering; it is per
    /// head unless [`AttentionOptions::query_key_norm_scope`] says otherwise.
    pub query_key_norm: bool,
    /// Whether the query projection also emits a per-head output gate.
    ///
    /// The gate is a sigmoid unless [`AttentionOptions::output_gate_activation`]
    /// selects SiLU.
    pub output_gate: bool,
    /// Position transformation used by this attention layer.
    pub position: PositionEncodingSpec,
}

impl AttentionSpec {
    /// Validates grouped-head divisibility and positional dimensions.
    ///
    /// Width-independent form of the check [`DecoderSpec::validate`] performs at
    /// the stack's residual width.
    pub fn validate(&self, field: &str) -> Result<()> {
        self.validate_in_block(field, 1, &AttentionOptions::default(), VALIDATION_NORM)
    }

    /// Validates spec invariants, options, the shared projection config, and the position encoding.
    ///
    /// `hidden_size` is the residual width and `norm` the block's mixer
    /// normalization, exactly as [`Self::grouped_query_config`] receives them
    /// at construction.
    fn validate_in_block(
        &self,
        field: &str,
        hidden_size: usize,
        options: &AttentionOptions,
        norm: NormalizationSpec,
    ) -> Result<()> {
        match attention_heads(self.query_heads, self.key_value_heads, self.head_dim) {
            HeadCheck::ZeroDimension => {
                return Err(Error::configuration(
                    field,
                    "query_heads, key_value_heads, and head_dim must be non-zero",
                ));
            }
            HeadCheck::IndivisibleGroups => {
                return Err(Error::configuration(
                    format!("{field}.key_value_heads"),
                    "must divide query_heads exactly",
                ));
            }
            HeadCheck::Valid => {}
        }
        options.validate_against(self, &format!("{field}.options"))?;
        self.grouped_query_config(hidden_size, options, norm)
            .validate()
            .map_err(|error| nested(field, error))?;
        self.position
            .validate(self.head_dim, &format!("{field}.position"))
    }

    /// The shared grouped-query attention configuration this spec describes.
    ///
    /// `options` supplies the score scale, soft cap, sinks, biases, gate
    /// activation, and Q/K-norm scope; `norm` is the block's mixer
    /// normalization, whose epsilon and zero-centering the optional Q/K norm
    /// reuses. Connectivity and the position encoding are applied by the
    /// caller, outside this projection config.
    pub(crate) fn grouped_query_config(
        &self,
        hidden_size: usize,
        options: &AttentionOptions,
        norm: NormalizationSpec,
    ) -> GroupedQueryAttentionConfig {
        let mut config = GroupedQueryAttentionConfig::new(
            hidden_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
        )
        .with_projection_bias(options.projection_bias);
        if self.query_key_norm {
            config = config.with_query_key_norm(norm.epsilon(), options.query_key_norm_scope);
            config.query_key_norm_zero_centered = norm.is_zero_centered();
        }
        config.output_gate = match (self.output_gate, options.output_gate_activation) {
            (false, _) => AttentionOutputGate::None,
            (true, OutputGateActivation::Sigmoid) => AttentionOutputGate::Sigmoid,
            (true, OutputGateActivation::Silu) => AttentionOutputGate::Silu,
        };
        config.scale = options
            .score_scale
            .grouped_query_scale(self.position.is_t5_relative_bias());
        config.softcap = options.logit_softcap;
        config.attention_sinks = options.attention_sinks;
        config
    }
}

/// Bounded-context attention that discards keys outside a trailing window.
///
/// Equivalent to [`MixerSpec::AttentionWithOptions`] with an
/// [`AttentionConnectivity::SlidingWindow`] connectivity and otherwise default options.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SlidingWindowSpec {
    /// Attention topology evaluated inside the window.
    pub attention: AttentionSpec,
    /// Maximum number of valid keys visible to a query, including the query's own key.
    ///
    /// Masked padding columns do not consume the window. Its age is measured in valid-token
    /// order, independently of offsets or gaps in explicit positional-encoding IDs.
    pub window: usize,
}

/// Decoder-to-encoder attention attached to every decoder block of a [`Seq2SeqSpec`].
///
/// Queries come from the decoder residual stream and keys/values from the final
/// encoder states, projected `hidden -> heads * head_dim -> hidden`. The branch
/// has as many key/value heads as query heads, no position encoding, no Q/K
/// norm, and no output gate; it reads every valid source token, so its
/// connectivity must be global.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CrossAttentionSpec {
    /// Number of query (and key/value) heads.
    pub heads: usize,
    /// Per-head projection width.
    pub head_dim: usize,
    /// Score scale, soft cap, sinks, and projection biases.
    #[serde(default)]
    pub options: AttentionOptions,
}

impl CrossAttentionSpec {
    /// Creates plain cross-attention with default options.
    #[must_use]
    pub fn new(heads: usize, head_dim: usize) -> Self {
        Self {
            heads,
            head_dim,
            options: AttentionOptions::default(),
        }
    }

    /// The position-free attention topology of this branch.
    #[must_use]
    pub const fn attention(&self) -> AttentionSpec {
        AttentionSpec {
            query_heads: self.heads,
            key_value_heads: self.heads,
            head_dim: self.head_dim,
            query_key_norm: false,
            output_gate: false,
            position: PositionEncodingSpec::None,
        }
    }

    /// Validates head counts, global connectivity, and the shared projection config.
    ///
    /// `hidden_size` is the decoder residual width. Errors name fields below
    /// `field` (`field.heads`, `field.options.connectivity`, ...).
    pub fn validate(&self, field: &str, hidden_size: usize) -> Result<()> {
        self.validate_fields(field, &format!("{field}.options"), hidden_size)
    }

    /// Validates with separate paths for the head geometry and the options.
    ///
    /// [`Seq2SeqSpec`] stores the options in a sibling field, so its errors
    /// name `cross_attention_options.*` rather than `cross_attention.options.*`.
    fn validate_fields(&self, field: &str, options_field: &str, hidden_size: usize) -> Result<()> {
        if !cross_projection_fits(self.heads, self.head_dim, hidden_size) {
            return Err(Error::configuration(
                field,
                "heads and head_dim must be non-zero, and hidden_size * heads * head_dim must fit in usize",
            ));
        }
        if self.options.connectivity != AttentionConnectivity::Global {
            return Err(Error::configuration(
                format!("{options_field}.connectivity"),
                "cross-attention reads every valid source token and must stay global",
            ));
        }
        let attention = self.attention();
        self.options.validate_against(&attention, options_field)?;
        attention
            .grouped_query_config(hidden_size, &self.options, VALIDATION_NORM)
            .validate()
            .map_err(|error| nested(field, error))
    }
}

/// Multi-head latent attention projection and cache ranks.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MlaSpec {
    /// Number of query heads.
    pub query_heads: usize,
    /// Non-rotary query/key width per head.
    pub content_head_dim: usize,
    /// Rotary query/key width per head.
    pub rotary_head_dim: usize,
    /// Value width per head.
    pub value_head_dim: usize,
    /// Rank of the compressed key/value latent cached during decode.
    pub key_value_lora_rank: usize,
    /// Optional rank of a compressed query projection.
    pub query_lora_rank: Option<usize>,
    /// Position transformation applied only to rotary channels.
    pub position: PositionEncodingSpec,
    /// Softmax temperature; [`ScoreScale::Auto`] is `1/sqrt(content_head_dim + rotary_head_dim)`.
    #[serde(default)]
    pub score_scale: ScoreScale,
    /// Which keys each query may see; sliding windows require a causal stack.
    #[serde(default)]
    pub connectivity: AttentionConnectivity,
}

impl MlaSpec {
    /// The shared latent-attention configuration this spec describes at `hidden_size`.
    ///
    /// `latent_norm_epsilon` stabilizes the compressed KV (and query) RMS norms;
    /// the modular models take it from the block's mixer normalization.
    pub(crate) fn latent_config(
        &self,
        hidden_size: usize,
        latent_norm_epsilon: f64,
    ) -> MultiLatentAttentionConfig {
        let mut config = MultiLatentAttentionConfig::new(
            hidden_size,
            self.query_heads,
            self.content_head_dim,
            self.rotary_head_dim,
            self.value_head_dim,
            self.key_value_lora_rank,
            self.query_lora_rank,
        );
        config.rms_norm_epsilon = latent_norm_epsilon;
        // Validation rejects an overflowing sum first; saturation keeps this
        // builder total for unvalidated specs.
        let qk_head_dim = mla_qk_head_dim(self.content_head_dim, self.rotary_head_dim);
        config.softmax_scale = self.score_scale.latent_scale(qk_head_dim);
        config
    }

    /// Validates dimensions, options, the shared latent config, and the rotary position encoding.
    fn validate_in_block(
        &self,
        field: &str,
        hidden_size: usize,
        norm: NormalizationSpec,
    ) -> Result<()> {
        match mla_dimensions(
            self.query_heads,
            self.content_head_dim,
            self.rotary_head_dim,
            self.value_head_dim,
            self.key_value_lora_rank,
            self.query_lora_rank,
        ) {
            MlaCheck::ZeroDimension => {
                return Err(Error::configuration(
                    field,
                    "MLA dimensions must be non-zero",
                ));
            }
            MlaCheck::HeadDimOverflow => {
                return Err(Error::configuration(
                    format!("{field}.head_dim"),
                    "MLA content and rotary dimensions overflow usize",
                ));
            }
            MlaCheck::Valid => {}
        }
        self.score_scale.validate(&format!("{field}.score_scale"))?;
        self.connectivity.validate().map_err(|error| {
            Error::configuration(format!("{field}.connectivity"), reason_of(error))
        })?;
        self.latent_config(hidden_size, norm.epsilon())
            .validate()
            .map_err(|error| nested(field, error))?;
        // MLA rotates only its decoupled rotary sub-head; the content
        // channels are never position-encoded.
        self.position
            .validate(self.rotary_head_dim, &format!("{field}.position"))
    }
}

/// Differential attention (DIFF Transformer, DiffLlama) over paired query heads.
///
/// Query head `i` and its partner `i + H/2` produce two softmax maps whose
/// difference `A_i - lambda A_(i+H/2)` mixes paired values; each pair is RMS
/// normalized (with the block's mixer epsilon) and scaled by `1 - lambda_init`
/// (see [`crate::attention::DifferentialAttention`]). Scores use the
/// conventional `1/sqrt(head_dim)` scale. The kernel has no additive-bias
/// input, so the position encoding is RoPE or none.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DifferentialAttentionSpec {
    /// Number of query heads; must be even because heads are used in pairs.
    pub query_heads: usize,
    /// Number of native key/value heads; must divide `query_heads`.
    pub key_value_heads: usize,
    /// Query/key/value width per head.
    pub head_dim: usize,
    /// Constant added to the learned lambda; DiffLlama uses
    /// [`crate::attention::depth_lambda_init`] of the zero-based layer index.
    pub lambda_init: f64,
    /// Position transformation of queries and keys: RoPE or none.
    pub position: PositionEncodingSpec,
    /// Which keys each query may see; sliding windows require a causal stack.
    #[serde(default)]
    pub connectivity: AttentionConnectivity,
    /// Whether all four projections carry additive biases.
    #[serde(default)]
    pub bias: bool,
}

impl DifferentialAttentionSpec {
    /// The shared differential-attention configuration at `hidden_size`.
    ///
    /// `norm_epsilon` stabilizes the weightless pair RMS norm; the modular
    /// models take it from the block's mixer normalization.
    pub(crate) fn differential_config(
        &self,
        hidden_size: usize,
        norm_epsilon: f64,
    ) -> DifferentialAttentionConfig {
        let mut config = DifferentialAttentionConfig::new(
            hidden_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
            0,
        );
        config.lambda_init = self.lambda_init;
        config.norm_epsilon = norm_epsilon;
        config.bias = self.bias;
        config
    }

    /// Validates dimensions, connectivity, the shared config, and the position encoding.
    fn validate_in_block(
        &self,
        field: &str,
        hidden_size: usize,
        norm: NormalizationSpec,
    ) -> Result<()> {
        if self.position.is_t5_relative_bias() {
            return Err(Error::configuration(
                format!("{field}.position"),
                "differential attention has no additive-bias input; use RoPE or no position encoding",
            ));
        }
        self.connectivity.validate().map_err(|error| {
            Error::configuration(format!("{field}.connectivity"), reason_of(error))
        })?;
        self.differential_config(hidden_size, norm.epsilon())
            .validate()
            .map_err(|error| nested(field, error))?;
        self.position
            .validate(self.head_dim, &format!("{field}.position"))
    }
}

// ---------------------------------------------------------------------------
// Sequence mixers
// ---------------------------------------------------------------------------

/// Mamba2 state-space mixer dimensions.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Mamba2Spec {
    /// Number of independent state-space heads.
    pub heads: usize,
    /// Per-head input/output width.
    pub head_dim: usize,
    /// Per-head recurrent state dimension.
    pub state_dim: usize,
    /// Width of the causal depthwise convolution history.
    pub convolution_kernel: usize,
    /// Number of heads sharing one state-space group.
    pub group_size: usize,
}

/// Sequence-mixing operation used by one block.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MixerSpec {
    /// Unbounded attention with default options (compact form).
    Attention(
        /// Head grouping, projection dimensions, and positional encoding.
        AttentionSpec,
    ),
    /// Causal attention restricted to a trailing window (compact form).
    SlidingAttention(
        /// Attention geometry and the permitted trailing key window.
        SlidingWindowSpec,
    ),
    /// Grouped-query attention with every option of the shared kernel spelled out.
    AttentionWithOptions {
        /// Head grouping, projection dimensions, and positional encoding.
        attention: AttentionSpec,
        /// Connectivity, score scale, soft cap, sinks, biases, and norm/gate variants.
        #[serde(default)]
        options: AttentionOptions,
    },
    /// DeepSeek-style multi-head latent attention.
    Mla(
        /// Latent compression ranks and query/key/value head dimensions.
        MlaSpec,
    ),
    /// Differential attention over paired query heads.
    DifferentialAttention(
        /// Head pairing, lambda constant, position encoding, and connectivity.
        DifferentialAttentionSpec,
    ),
    /// Gated DeltaNet linear attention.
    DeltaNet {
        /// Number of normalized query/key heads.
        key_heads: usize,
        /// Number of recurrent value heads.
        value_heads: usize,
        /// Query/key channel width per head.
        key_head_dim: usize,
        /// Value channel width per head.
        value_head_dim: usize,
        /// Width of the causal depthwise convolution.
        convolution_kernel: usize,
    },
    /// Mamba2 structured state-space mixing.
    ///
    /// Describable and validated, but not yet executable: the modular models
    /// reject it at `init` with a capability error.
    Mamba2(
        /// Recurrent state, head grouping, and convolution dimensions.
        Mamba2Spec,
    ),
}

impl MixerSpec {
    /// Validates head grouping and non-zero recurrent-state dimensions.
    ///
    /// Each executable mixer is also checked by its shared component
    /// configuration (grouped-query attention, MLA, or Gated DeltaNet), so a
    /// mixer that validates can be constructed. This width-independent form
    /// uses residual width one; [`DecoderSpec::validate`] repeats the component
    /// checks at the stack's real width and mixer normalization, which also
    /// bounds projection sizes.
    pub fn validate(&self, field: &str) -> Result<()> {
        self.validate_in_block(field, 1, VALIDATION_NORM)
    }

    /// Returns the topology and options of any grouped-query attention form.
    ///
    /// The compact forms normalize to explicit options: `Attention` uses the
    /// defaults and `SlidingAttention` a sliding-window connectivity. Returns
    /// `None` for MLA, differential attention, and recurrent mixers.
    #[must_use]
    pub fn grouped_query_attention(&self) -> Option<(&AttentionSpec, AttentionOptions)> {
        match self {
            Self::Attention(attention) => Some((attention, AttentionOptions::default())),
            Self::SlidingAttention(sliding) => Some((
                &sliding.attention,
                AttentionOptions {
                    connectivity: AttentionConnectivity::SlidingWindow {
                        window: sliding.window,
                    },
                    ..AttentionOptions::default()
                },
            )),
            Self::AttentionWithOptions { attention, options } => Some((attention, *options)),
            Self::Mla(_)
            | Self::DifferentialAttention(_)
            | Self::DeltaNet { .. }
            | Self::Mamba2(_) => None,
        }
    }

    /// Returns the key connectivity of an attention mixer, or `None` for recurrent mixers.
    #[must_use]
    pub fn connectivity(&self) -> Option<AttentionConnectivity> {
        match self {
            Self::Mla(mla) => Some(mla.connectivity),
            Self::DifferentialAttention(differential) => Some(differential.connectivity),
            other => other
                .grouped_query_attention()
                .map(|(_, options)| options.connectivity),
        }
    }

    /// Returns whether this mixer is intrinsically causal (a recurrent scan).
    ///
    /// Such mixers cannot execute bidirectional or custom connectivity.
    #[must_use]
    pub const fn is_recurrent(&self) -> bool {
        matches!(self, Self::DeltaNet { .. } | Self::Mamba2(_))
    }

    /// The shared Gated DeltaNet configuration of a [`Self::DeltaNet`] mixer, or `None`.
    ///
    /// `norm_epsilon` stabilizes the per-head output RMS norm, which the
    /// modular models take from the block's mixer normalization.
    pub(crate) fn delta_net_config(
        &self,
        hidden_size: usize,
        norm_epsilon: f64,
    ) -> Option<GatedDeltaNetConfig> {
        let Self::DeltaNet {
            key_heads,
            value_heads,
            key_head_dim,
            value_head_dim,
            convolution_kernel,
        } = self
        else {
            return None;
        };
        let mut config = GatedDeltaNetConfig::new(
            hidden_size,
            *key_heads,
            *value_heads,
            *key_head_dim,
            *value_head_dim,
        );
        config.convolution_kernel = *convolution_kernel;
        config.normalization_epsilon = norm_epsilon;
        Some(config)
    }

    /// Validates this mixer as built inside a block of `hidden_size` channels.
    ///
    /// `norm` is the block's mixer normalization, whose epsilon the optional
    /// Q/K, latent, and DeltaNet output norms reuse.
    fn validate_in_block(
        &self,
        field: &str,
        hidden_size: usize,
        norm: NormalizationSpec,
    ) -> Result<()> {
        match self {
            Self::Attention(spec) => {
                spec.validate_in_block(field, hidden_size, &AttentionOptions::default(), norm)
            }
            Self::SlidingAttention(spec) => {
                if spec.window == 0 {
                    return Err(Error::configuration(
                        format!("{field}.window"),
                        "must be non-zero",
                    ));
                }
                spec.attention.validate_in_block(
                    &format!("{field}.attention"),
                    hidden_size,
                    &AttentionOptions::default(),
                    norm,
                )
            }
            Self::AttentionWithOptions { attention, options } => attention.validate_in_block(
                &format!("{field}.attention"),
                hidden_size,
                options,
                norm,
            ),
            Self::Mla(spec) => spec.validate_in_block(field, hidden_size, norm),
            Self::DifferentialAttention(spec) => spec.validate_in_block(field, hidden_size, norm),
            Self::DeltaNet {
                key_heads,
                value_heads,
                key_head_dim,
                value_head_dim,
                convolution_kernel,
            } => {
                match delta_net_heads(
                    *key_heads,
                    *value_heads,
                    *key_head_dim,
                    *value_head_dim,
                    *convolution_kernel,
                ) {
                    HeadCheck::ZeroDimension => {
                        return Err(Error::configuration(
                            field,
                            "DeltaNet dimensions must be non-zero",
                        ));
                    }
                    HeadCheck::IndivisibleGroups => {
                        return Err(Error::configuration(
                            format!("{field}.value_heads"),
                            "must be an integer multiple of key_heads",
                        ));
                    }
                    HeadCheck::Valid => {}
                }
                self.delta_net_config(hidden_size, norm.epsilon())
                    .map_or(Ok(()), |config| config.validate())
                    .map_err(|error| nested(field, error))
            }
            Self::Mamba2(spec) => {
                if !mamba2_dimensions(
                    spec.heads,
                    spec.head_dim,
                    spec.state_dim,
                    spec.convolution_kernel,
                    spec.group_size,
                ) {
                    return Err(Error::configuration(
                        field,
                        "Mamba2 dimensions must be non-zero and group_size must divide heads",
                    ));
                }
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Feed-forward operators
// ---------------------------------------------------------------------------

/// Dense feed-forward widths, activation, and projection-bias policy.
///
/// Inside [`FeedForwardSpec::Dense`] this describes the gated MLP
/// `down(activation(gate(x)) * up(x))`; inside [`FeedForwardSpec::Ungated`] the
/// classic MLP `down(activation(up(x)))`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DenseFeedForwardSpec {
    /// Intermediate channel count.
    pub intermediate_size: usize,
    /// Gate activation, or the hidden activation of an ungated MLP.
    pub activation: Activation,
    /// Whether linear projections contain additive biases.
    pub bias: bool,
}

/// Clamped SwiGLU feed-forward `down(glu(gate(x), up(x)))` of gpt-oss and DeepSeek V4.
///
/// With `g = min(gate, limit)` the unit computes
/// `(clamp(up, -limit, limit) + up_offset) * g * sigmoid(alpha * g)`; see
/// [`ClampedSwiGlu`]. gpt-oss uses `limit = 7`, `alpha = 1.702`, `up_offset = 1`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClampedSwiGluSpec {
    /// Intermediate channel count of the gate and up projections.
    pub intermediate_size: usize,
    /// Whether linear projections contain additive biases.
    pub bias: bool,
    /// Upper clamp of the gate and symmetric clamp of the up projection.
    pub limit: f64,
    /// Sigmoid sharpness; one gives SiLU.
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    /// Constant added to the clamped up projection.
    #[serde(default)]
    pub up_offset: f64,
}

/// Serde default for [`ClampedSwiGluSpec::alpha`]: an ordinary SiLU gate.
const fn default_alpha() -> f64 {
    1.0
}

impl ClampedSwiGluSpec {
    /// The shared clamped gated unit this spec describes.
    #[must_use]
    pub fn glu(&self) -> ClampedSwiGlu {
        ClampedSwiGlu::new(self.limit)
            .with_alpha(self.alpha)
            .with_up_offset(self.up_offset)
    }
}

/// Sparse routed mixture-of-experts topology.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MixtureOfExpertsSpec {
    /// Number of separately parameterized routed experts.
    pub routed_experts: usize,
    /// Number of selected routed experts per token.
    pub experts_per_token: usize,
    /// Number of always-evaluated shared experts.
    pub shared_experts: usize,
    /// Width of each expert's intermediate representation.
    pub expert_intermediate_size: usize,
    /// Expert gate activation.
    pub activation: Activation,
    /// Whether selected router probabilities are renormalized to sum to one.
    pub renormalize_selected: bool,
    /// Multiplicative scale applied to routed expert outputs.
    pub routed_scale: f64,
    /// Router scoring, ordering, eligibility, biases, and precision.
    ///
    /// The default is softmax top-k over all experts with model-dtype logits.
    #[serde(default)]
    pub routing: RoutingPolicy,
    /// Whether a learned sigmoid gate scales the shared-expert sum (Qwen3.5-MoE).
    #[serde(default)]
    pub shared_expert_gate: bool,
    /// Whether expert projections carry additive biases (gpt-oss).
    #[serde(default)]
    pub expert_bias: bool,
}

impl MixtureOfExpertsSpec {
    /// The shared sparse-expert configuration this spec describes at `hidden_size`.
    pub(crate) fn sparse_config(&self, hidden_size: usize) -> SparseMoeConfig {
        let mut config = SparseMoeConfig::new(
            hidden_size,
            self.expert_intermediate_size,
            self.routed_experts,
            self.experts_per_token,
        );
        config.shared_experts = self.shared_experts;
        config.activation = self.activation;
        config.renormalize_selected = self.renormalize_selected;
        config.routed_scale = self.routed_scale;
        config.routing = self.routing;
        config.shared_expert_gate = self.shared_expert_gate;
        config.expert_bias = self.expert_bias;
        config
    }
}

/// Feed-forward operation attached to a transformer block.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FeedForwardSpec {
    /// One dense gated feed-forward network `down(activation(gate(x)) * up(x))`.
    Dense(
        /// Intermediate width, activation, and projection-bias policy.
        DenseFeedForwardSpec,
    ),
    /// One dense ungated feed-forward network `down(activation(up(x)))` (GPT-2, T5 v1.0).
    Ungated(
        /// Intermediate width, activation, and projection-bias policy.
        DenseFeedForwardSpec,
    ),
    /// One dense clamped SwiGLU feed-forward network.
    ClampedSwiGlu(
        /// Intermediate width, bias policy, and clamp parameters.
        ClampedSwiGluSpec,
    ),
    /// A sparse router plus independent expert networks.
    MixtureOfExperts(
        /// Router selection and per-expert network configuration.
        MixtureOfExpertsSpec,
    ),
    /// No feed-forward branch: the block is `x + mixer(x)` only (attention-only models).
    ///
    /// The layer's `feed_forward_norm` is still validated but allocates no parameters.
    None,
}

impl FeedForwardSpec {
    /// Validates intermediate widths and sparse-routing cardinalities.
    pub fn validate(&self, field: &str) -> Result<()> {
        self.validate_in_block(field, 1)
    }

    /// Validates this feed-forward operation as built inside a stack of `hidden_size` channels.
    fn validate_in_block(&self, field: &str, hidden_size: usize) -> Result<()> {
        match self {
            Self::Dense(spec) | Self::Ungated(spec) if spec.intermediate_size == 0 => Err(
                Error::configuration(format!("{field}.intermediate_size"), "must be non-zero"),
            ),
            Self::Dense(spec) => GatedFeedForwardConfig::new(hidden_size, spec.intermediate_size)
                .validate()
                .map_err(|error| nested(field, error)),
            Self::Ungated(spec) => DenseFeedForwardConfig::new(hidden_size, spec.intermediate_size)
                .validate()
                .map_err(|error| nested(field, error)),
            Self::ClampedSwiGlu(spec) => {
                if spec.intermediate_size == 0 {
                    return Err(Error::configuration(
                        format!("{field}.intermediate_size"),
                        "must be non-zero",
                    ));
                }
                spec.glu().validate().map_err(|error| {
                    Error::configuration(format!("{field}.glu"), reason_of(error))
                })?;
                GatedFeedForwardConfig::new(hidden_size, spec.intermediate_size)
                    .validate()
                    .map_err(|error| nested(field, error))
            }
            Self::MixtureOfExperts(spec) => {
                match expert_counts(
                    spec.routed_experts,
                    spec.experts_per_token,
                    spec.expert_intermediate_size,
                ) {
                    ExpertCheck::Empty => {
                        return Err(Error::configuration(
                            field,
                            "routed_experts and expert_intermediate_size must be non-zero",
                        ));
                    }
                    ExpertCheck::SelectionOutOfRange => {
                        return Err(Error::configuration(
                            format!("{field}.experts_per_token"),
                            "must be in 1..=routed_experts",
                        ));
                    }
                    ExpertCheck::Valid => {}
                }
                if !spec.routed_scale.is_finite() || spec.routed_scale <= 0.0 {
                    return Err(Error::configuration(
                        format!("{field}.routed_scale"),
                        "must be finite and greater than zero",
                    ));
                }
                spec.sparse_config(hidden_size)
                    .validate()
                    .map_err(|error| nested(field, error))
            }
            Self::None => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Blocks and stacks
// ---------------------------------------------------------------------------

/// Placement of block normalization relative to its sublayer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormPlacement {
    /// Normalize before the sublayer only.
    Pre,
    /// Normalize after the sublayer only.
    Post,
    /// Normalize both before and after, as in Muse sandwich normalization.
    Sandwich,
}

/// Residual-update topology used around a block sublayer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualTopology {
    /// Add the sublayer output directly to its input.
    Add,
    /// Apply a learned multiplicative gate before the residual addition.
    ///
    /// Reserved: the gate's activation, shape, and initialization are not yet
    /// specified, so the modular models reject it at `init`.
    Gated,
}

/// Complete specification for one heterogeneous block.
///
/// Despite its historical name, the same block specification describes
/// encoder layers too; the enclosing stack decides the default attention
/// direction.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DecoderLayerSpec {
    /// Sequence-mixing operation.
    pub mixer: MixerSpec,
    /// Dense or sparse feed-forward operation.
    pub feed_forward: FeedForwardSpec,
    /// Attention/mixer-side normalization.
    pub mixer_norm: NormalizationSpec,
    /// Feed-forward-side normalization.
    pub feed_forward_norm: NormalizationSpec,
    /// Relative normalization placement.
    pub norm_placement: NormPlacement,
    /// Residual-update topology.
    pub residual: ResidualTopology,
}

impl DecoderLayerSpec {
    /// Validates every independently configurable subcomponent.
    ///
    /// Width-independent form of the per-layer check in [`DecoderSpec::validate`];
    /// errors are reported below `layers[index]`.
    pub fn validate(&self, index: usize) -> Result<()> {
        self.validate_at(&format!("layers[{index}]"), 1)
    }

    /// Validates this layer as built inside a stack of `hidden_size` channels.
    ///
    /// Errors name fields below `field`, for example `layer.mixer.attention`
    /// for a standalone block or `layers[3].mixer.attention` inside a stack.
    /// The mixer is validated with this block's own mixer normalization.
    pub(crate) fn validate_at(&self, field: &str, hidden_size: usize) -> Result<()> {
        self.mixer_norm.validate(&format!("{field}.mixer_norm"))?;
        self.feed_forward_norm
            .validate(&format!("{field}.feed_forward_norm"))?;
        self.mixer
            .validate_in_block(&format!("{field}.mixer"), hidden_size, self.mixer_norm)?;
        self.feed_forward
            .validate_in_block(&format!("{field}.feed_forward"), hidden_size)
    }
}

/// Validated topology for one modular stack, used as a decoder-only language model.
///
/// [`Seq2SeqSpec`] reuses this type for its bidirectional encoder stack; the
/// vocabulary then describes the source tokens.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DecoderSpec {
    /// Token vocabulary cardinality.
    pub vocabulary_size: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Ordered heterogeneous blocks.
    pub layers: Vec<DecoderLayerSpec>,
    /// Final normalization before vocabulary projection.
    pub final_norm: NormalizationSpec,
    /// Whether the embedding matrix is also the vocabulary projection matrix.
    pub tie_word_embeddings: bool,
    /// Maximum supported logical position.
    pub max_position_embeddings: usize,
}

impl DecoderSpec {
    /// Validates global dimensions and every layer specification.
    pub fn validate(&self) -> Result<()> {
        let failure = match stack_dimensions(
            self.vocabulary_size,
            self.hidden_size,
            self.layers.len(),
            self.max_position_embeddings,
        ) {
            StackCheck::ZeroVocabulary => Some(("vocabulary_size", "must be non-zero")),
            StackCheck::ZeroHidden => Some(("hidden_size", "must be non-zero")),
            StackCheck::NoLayers => Some(("layers", "must not be empty")),
            StackCheck::ZeroPositions => Some(("max_position_embeddings", "must be non-zero")),
            StackCheck::Valid => None,
        };
        if let Some((field, reason)) = failure {
            return Err(Error::configuration(field, reason));
        }
        self.final_norm.validate("final_norm")?;
        for (index, layer) in self.layers.iter().enumerate() {
            layer.validate_at(&format!("layers[{index}]"), self.hidden_size)?;
        }
        Ok(())
    }
}

/// Modular encoder-decoder topology with independent source and target stacks.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Seq2SeqSpec {
    /// Bidirectional encoder stack, represented with the role-neutral stack type.
    pub encoder: DecoderSpec,
    /// Causal decoder stack.
    pub decoder: DecoderSpec,
    /// Number of decoder cross-attention heads.
    pub cross_attention_heads: usize,
    /// Per-head cross-attention width.
    pub cross_attention_head_dim: usize,
    /// Score scale, soft cap, sinks, and projection biases of every cross-attention branch.
    ///
    /// Cross-attention reads every valid source token, so connectivity must stay
    /// global; the branch has no position encoding, Q/K norm, or output gate.
    #[serde(default)]
    pub cross_attention_options: AttentionOptions,
}

impl Seq2SeqSpec {
    /// Validates stack compatibility and cross-attention projection width.
    ///
    /// Cross-attention projects `hidden_size -> heads * head_dim -> hidden_size`,
    /// so the inner width may differ from the residual width (as in T5-3B/11B);
    /// it must only be nonzero and its projection matrices must not overflow.
    pub fn validate(&self) -> Result<()> {
        self.encoder.validate()?;
        self.decoder.validate()?;
        if self.encoder.hidden_size != self.decoder.hidden_size {
            return Err(Error::configuration(
                "decoder.hidden_size",
                "must equal encoder.hidden_size",
            ));
        }
        self.cross_attention().validate_fields(
            "cross_attention",
            "cross_attention_options",
            self.decoder.hidden_size,
        )
    }

    /// The cross-attention branch shared by every decoder block.
    #[must_use]
    pub fn cross_attention(&self) -> CrossAttentionSpec {
        CrossAttentionSpec {
            heads: self.cross_attention_heads,
            head_dim: self.cross_attention_head_dim,
            options: self.cross_attention_options,
        }
    }
}

// ---------------------------------------------------------------------------
// Error-path helpers
// ---------------------------------------------------------------------------

/// Extracts the human-readable reason of a shared component's configuration error.
fn reason_of(error: Error) -> String {
    match error {
        Error::Configuration { reason, .. } => reason,
        other => other.to_string(),
    }
}

/// Prefixes a shared component's configuration field with the spec path that produced it.
///
/// For example, an MLA error at `mla.qk_rope_head_dim` raised while validating
/// `layers[0].mixer` becomes `layers[0].mixer.mla.qk_rope_head_dim`.
fn nested(field: &str, error: Error) -> Error {
    match error {
        Error::Configuration {
            field: component,
            reason,
        } => Error::configuration(format!("{field}.{component}"), reason),
        other => other,
    }
}

#[cfg(test)]
#[path = "spec_tests.rs"]
mod tests;
