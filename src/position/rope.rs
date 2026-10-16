//! Validated host configuration for rotary embeddings and context extension.
//!
//! A rotary prefix of even width `d <= head_dim` defines `d/2` complex coordinate
//! pairs with base frequencies `omega_i = theta^(-2i/d)`. Pairs use the
//! Hugging Face *split-half* layout: channel `i` is the real part and channel
//! `i + d/2` the imaginary part of pair `i`. Checkpoints that interleave pairs
//! (`[re_0, im_0, re_1, im_1, ...]`) must be permuted with
//! [`super::interleaved_to_split_half`] (or have their projection rows permuted
//! once at load time) before these tables apply. Channels beyond `d` pass
//! through unrotated.
//!
//! [`RopeScaling`] selects how frequencies and the cosine/sine amplitude change
//! for context extension:
//!
//! | Policy | Frequency rule | Amplitude |
//! | --- | --- | --- |
//! | none | `omega_i` | 1 |
//! | linear | `omega_i / factor` | 1 |
//! | dynamic NTK | base `theta * (f * c / L - (f - 1))^(d/(d-2))` once `c > L` | 1 |
//! | YaRN | ramp between `omega_i` and `omega_i / factor` over a rotation-count band | `attention_factor` |
//! | Llama 3 | wavelength-banded smooth interpolation | 1 |
//! | LongRoPE | `omega_i / r_i`, short or long table by context | `attention_factor` |
//! | proportional | `omega_i / factor` for leading pairs, zero for the rest | 1 |
//!
//! `c` is the evaluated context (largest position plus one) and `L` the
//! original training context. Formulas follow Hugging Face Transformers 5.2
//! `modeling_rope_utils.py`; proportional RoPE follows Transformers 5.8.0.
//! Configuration uses host F64 scalars with finite-domain checks; tensor dtype
//! and device conversion occur separately in the rotary tensor module. Dynamic
//! NTK needs `d > 2` because its exponent contains `d/(d-2)`.
//!
//! # Precision envelope
//!
//! Frequencies are rounded once to `f32`, angles are formed in `f32`, and the
//! amplitude multiplies `f32` cosine/sine tables, so validation guarantees
//! finite tables for every accepted configuration at every integer position,
//! not only finite `f64` parameters:
//!
//! - `theta >= 1` keeps every base frequency `omega_i = theta^(-2i/d)` in
//!   `(0, 1]` (a base below one makes them grow geometrically: `theta =
//!   1e-300` gives `omega_1 = 1e150` for `d = 4`, which is infinite in `f32`).
//!   Every published checkpoint uses `theta >= 10^4`.
//! - Every policy except LongRoPE divides `omega_i` by a factor of at least
//!   one or blends it with such a quotient, so it stays in `[0, 1]`.
//! - LongRoPE divides by searched per-pair factors, which must round to `f32`
//!   values of at least [`LongRopeScaling::MIN_FACTOR`] (`2^-64`); then
//!   `omega_i / r_i <= 2^64`. A subnormal factor such as `1e-45` would
//!   overflow the frequency itself, and GPU kernels may flush it to zero.
//! - The device forms each angle `p * omega_i` in `f32`. Integer positions
//!   satisfy `|p| <= 2^63`, so with every frequency at most `2^64` the angle
//!   stays below `2^127 < f32::MAX`, and its cosine and sine are finite at
//!   every position, not only at small ones (a LongRoPE factor of
//!   `f32::MIN_POSITIVE` keeps `omega_i / r_i` finite but overflows the angle
//!   from position 4 on, and `cos(inf)` is NaN).
//! - Amplitudes (YaRN and LongRoPE `attention_factor`) must be normal positive
//!   `f32` values too: an amplitude above `f32::MAX` turns `sin(0) = 0` at
//!   position zero into `0 * inf = NaN`.

use core::fmt;

use super::rope_scaling::{Llama3RopeScaling, LongRopeScaling, ProportionalRopeScaling};

/// Rotation-count threshold above which YaRN retains original frequencies.
const DEFAULT_YARN_BETA_FAST: f64 = 32.0;
/// Rotation-count threshold below which YaRN interpolates frequencies fully.
const DEFAULT_YARN_BETA_SLOW: f64 = 1.0;

/// Validated linear RoPE position scaling.
///
/// Linear scaling evaluates the original rotary frequencies at
/// `position / factor`. A factor of one is the identity; larger factors stretch
/// the trained position interval over a longer context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearRopeScaling {
    /// Finite position divisor, at least one.
    factor: f64,
}

impl LinearRopeScaling {
    /// Creates an identity-or-extension linear scaling factor.
    ///
    /// # Errors
    ///
    /// Returns [`RopeConfigError::InvalidScalingFactor`] unless `factor` is
    /// finite and at least one.
    pub fn new(factor: f64) -> Result<Self, RopeConfigError> {
        validate_scaling_factor("linear", factor)?;
        Ok(Self { factor })
    }

    /// Returns the position divisor used by linear scaling.
    #[must_use]
    pub const fn factor(self) -> f64 {
        self.factor
    }
}

/// Validated dynamic-NTK RoPE scaling parameters.
///
/// Dynamic NTK scaling increases the RoPE base after the sequence exceeds its
/// trained context. Its exponent contains `dimension / (dimension - 2)`, so
/// [`RopeConfig`] additionally rejects rotary dimensions of two or less.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicNtkScaling {
    /// Finite context-extension multiplier, at least one.
    factor: f64,
    /// Positive trained context length before dynamic base scaling begins.
    original_max_position_embeddings: usize,
}

impl DynamicNtkScaling {
    /// Creates dynamic-NTK scaling parameters.
    ///
    /// # Errors
    ///
    /// Returns an error unless `factor` is finite and at least one and the
    /// original trained context is non-zero.
    pub fn new(
        factor: f64,
        original_max_position_embeddings: usize,
    ) -> Result<Self, RopeConfigError> {
        validate_scaling_factor("dynamic NTK", factor)?;
        validate_original_context("dynamic NTK", original_max_position_embeddings)?;
        Ok(Self {
            factor,
            original_max_position_embeddings,
        })
    }

    /// Returns the requested context-extension factor.
    #[must_use]
    pub const fn factor(self) -> f64 {
        self.factor
    }

    /// Returns the maximum position used during pretraining.
    #[must_use]
    pub const fn original_max_position_embeddings(self) -> usize {
        self.original_max_position_embeddings
    }
}

/// Validated YaRN context-extension parameters.
///
/// YaRN interpolates low-frequency dimensions and extrapolates high-frequency
/// dimensions. `beta_fast` and `beta_slow` specify the rotation-count bounds of
/// the linear transition, with `beta_fast > beta_slow > 0`. Despite its
/// checkpoint-compatible name, `attention_factor` is the amplitude multiplier
/// applied to the computed cosine and sine tables—not directly to logits. If a
/// model specifies `mscale`/`mscale_all_dim`, its config adapter must resolve
/// those fields to this final amplitude before constructing [`YarnConfig`].
///
/// The ramp runs over frequency indices `i in 0..d/2` between correction
/// dimensions `low` and `high`, where a correction dimension for `beta`
/// rotations is `d * ln(L / (2 pi beta)) / (2 ln theta)`. With `truncate`
/// (the reference default) `low` is floored and `high` ceiled; then
/// `low = max(low, 0)` and `high = min(high, d - 1)`. The upper clamp uses the
/// full rotary width `d`, not `d/2`, exactly as the reference does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnConfig {
    /// Finite requested context-extension multiplier, at least one.
    factor: f64,
    /// Positive trained context length used to locate frequency transition bounds.
    original_max_position_embeddings: usize,
    /// Positive cosine/sine amplitude multiplier applied after frequency scaling.
    attention_factor: f64,
    /// Upper positive rotation-count boundary for the frequency transition.
    beta_fast: f64,
    /// Lower positive rotation-count boundary, strictly below `beta_fast`.
    beta_slow: f64,
    /// Whether correction dimensions are rounded outward to integers (`floor`/`ceil`).
    truncate: bool,
}

impl YarnConfig {
    /// Creates YaRN parameters with reference defaults.
    ///
    /// The default correction bounds are `beta_fast = 32` and `beta_slow = 1`.
    /// The default attention multiplier is `1 + 0.1 * ln(factor)`, matching the
    /// common YaRN implementation used by open-weight transformer configs.
    ///
    /// # Errors
    ///
    /// Returns an error unless `factor` is finite and at least one and the
    /// original trained context is non-zero.
    pub fn new(
        factor: f64,
        original_max_position_embeddings: usize,
    ) -> Result<Self, RopeConfigError> {
        validate_scaling_factor("YaRN", factor)?;
        validate_original_context("YaRN", original_max_position_embeddings)?;
        Ok(Self {
            factor,
            original_max_position_embeddings,
            attention_factor: 1.0 + 0.1 * factor.ln(),
            beta_fast: DEFAULT_YARN_BETA_FAST,
            beta_slow: DEFAULT_YARN_BETA_SLOW,
            truncate: true,
        })
    }

    /// Chooses whether correction dimensions are floored/ceiled before clamping.
    ///
    /// The reference default is `true`; gpt-oss checkpoints set
    /// `"truncate": false`, which keeps fractional ramp bounds.
    #[must_use]
    pub const fn with_truncate(mut self, truncate: bool) -> Self {
        self.truncate = truncate;
        self
    }

    /// Replaces the cosine/sine amplitude multiplier named `attention_factor`.
    ///
    /// # Errors
    ///
    /// Returns [`RopeConfigError::InvalidYarnAttentionFactor`] unless the value
    /// rounds to a normal positive `f32` (from [`f32::MIN_POSITIVE`] to
    /// [`f32::MAX`]), the precision of the cosine/sine tables it scales.
    pub fn with_attention_factor(mut self, attention_factor: f64) -> Result<Self, RopeConfigError> {
        if !is_normal_positive_f32(attention_factor) {
            return Err(RopeConfigError::InvalidYarnAttentionFactor { attention_factor });
        }
        self.attention_factor = attention_factor;
        Ok(self)
    }

    /// Replaces the fast and slow YaRN correction bounds.
    ///
    /// # Errors
    ///
    /// Returns an error unless both values are finite and positive and
    /// `beta_fast` is strictly greater than `beta_slow`.
    pub fn with_betas(mut self, beta_fast: f64, beta_slow: f64) -> Result<Self, RopeConfigError> {
        validate_yarn_beta("beta_fast", beta_fast)?;
        validate_yarn_beta("beta_slow", beta_slow)?;
        if beta_fast <= beta_slow {
            return Err(RopeConfigError::InvalidYarnBetaOrder {
                beta_fast,
                beta_slow,
            });
        }
        self.beta_fast = beta_fast;
        self.beta_slow = beta_slow;
        Ok(self)
    }

    /// Replaces either correction bound while retaining the current value for an absent side.
    ///
    /// Checkpoint schemas serialize `beta_fast` and `beta_slow` independently.
    /// Calling this on a freshly created configuration therefore resolves
    /// missing fields to the reference defaults of 32 and 1.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`YarnConfig::with_betas`] after
    /// resolving the two optional overrides.
    pub fn with_beta_overrides(
        self,
        beta_fast: Option<f64>,
        beta_slow: Option<f64>,
    ) -> Result<Self, RopeConfigError> {
        self.with_betas(
            beta_fast.unwrap_or(self.beta_fast),
            beta_slow.unwrap_or(self.beta_slow),
        )
    }

    /// Returns the requested context-extension factor.
    #[must_use]
    pub const fn factor(self) -> f64 {
        self.factor
    }

    /// Returns the maximum position used during pretraining.
    #[must_use]
    pub const fn original_max_position_embeddings(self) -> usize {
        self.original_max_position_embeddings
    }

    /// Returns the amplitude multiplier applied to both cosine and sine tables.
    ///
    /// For a fully rotated query and key, scaling both operands makes their
    /// rotary dot-product contribution proportional to this value squared.
    #[must_use]
    pub const fn attention_factor(self) -> f64 {
        self.attention_factor
    }

    /// Returns the upper rotation-count bound of the ramp.
    #[must_use]
    pub const fn beta_fast(self) -> f64 {
        self.beta_fast
    }

    /// Returns the lower rotation-count bound of the ramp.
    #[must_use]
    pub const fn beta_slow(self) -> f64 {
        self.beta_slow
    }

    /// Returns whether the ramp bounds are rounded outward to integers.
    #[must_use]
    pub const fn truncate(self) -> bool {
        self.truncate
    }
}

/// A validated choice of RoPE context-scaling algorithm.
///
/// The LongRoPE variant stores its factor tables inline so that this enum and
/// [`RopeConfig`] remain `Copy`; see [`LongRopeScaling`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[allow(
    clippy::large_enum_variant,
    reason = "RopeConfig is Copy and passed by value per layer; boxing LongRoPE tables would remove Copy"
)]
pub enum RopeScaling {
    /// Uses the checkpoint's original positions and frequency base.
    #[default]
    None,
    /// Divides absolute positions by a constant extension factor.
    Linear(
        /// Validated constant position-divisor policy.
        LinearRopeScaling,
    ),
    /// Changes the frequency base dynamically beyond the trained context.
    DynamicNtk(
        /// Validated extension multiplier and original context length.
        DynamicNtkScaling,
    ),
    /// Blends interpolated and extrapolated frequencies with YaRN.
    Yarn(
        /// Validated frequency-transition and amplitude parameters.
        YarnConfig,
    ),
    /// Interpolates frequencies by wavelength band as in Llama 3.1.
    Llama3(
        /// Validated factor, band multipliers, and original context.
        Llama3RopeScaling,
    ),
    /// Divides each frequency by a searched per-pair factor (Phi-3 LongRoPE).
    LongRope(
        /// Validated short/long factor tables and amplitude.
        LongRopeScaling,
    ),
    /// Rotates only a leading fraction of full-head split-half pairs.
    Proportional(
        /// Validated rotated fraction and position divisor.
        ProportionalRopeScaling,
    ),
}

impl RopeScaling {
    /// Returns whether frequencies depend on the evaluated context.
    ///
    /// Only dynamic NTK and LongRoPE do. For every other policy any non-zero
    /// `evaluated_context` yields identical tables, so callers can skip
    /// computing the largest position (often a device-to-host read).
    #[must_use]
    pub const fn is_context_dependent(&self) -> bool {
        matches!(self, Self::DynamicNtk(_) | Self::LongRope(_))
    }

    /// Returns the largest evaluated context that keeps cached keys valid.
    ///
    /// Keys stored in a decode cache were rotated with the frequencies of the
    /// context at the time they were written. A context-dependent policy changes
    /// those frequencies once the context exceeds its original training length,
    /// so a cache must not be extended past this limit without re-rotating
    /// keys. `None` means the policy never invalidates cached keys.
    #[must_use]
    pub const fn cached_context_limit(&self) -> Option<usize> {
        match self {
            Self::DynamicNtk(dynamic) => Some(dynamic.original_max_position_embeddings()),
            Self::LongRope(long) => Some(long.original_max_position_embeddings()),
            Self::None
            | Self::Linear(_)
            | Self::Yarn(_)
            | Self::Llama3(_)
            | Self::Proportional(_) => None,
        }
    }

    /// Returns the cosine/sine amplitude multiplier this policy applies.
    #[must_use]
    pub const fn amplitude(&self) -> f64 {
        match self {
            Self::Yarn(yarn) => yarn.attention_factor(),
            Self::LongRope(long) => long.attention_factor(),
            Self::None
            | Self::Linear(_)
            | Self::DynamicNtk(_)
            | Self::Llama3(_)
            | Self::Proportional(_) => 1.0,
        }
    }
}

impl From<LinearRopeScaling> for RopeScaling {
    /// Wraps already validated linear position scaling in the shared RoPE policy.
    fn from(value: LinearRopeScaling) -> Self {
        Self::Linear(value)
    }
}

impl From<DynamicNtkScaling> for RopeScaling {
    /// Wraps already validated dynamic-NTK scaling in the shared RoPE policy.
    fn from(value: DynamicNtkScaling) -> Self {
        Self::DynamicNtk(value)
    }
}

impl From<YarnConfig> for RopeScaling {
    /// Wraps already validated YaRN interpolation settings in the shared RoPE policy.
    fn from(value: YarnConfig) -> Self {
        Self::Yarn(value)
    }
}

impl From<Llama3RopeScaling> for RopeScaling {
    /// Wraps already validated Llama 3 band interpolation in the shared RoPE policy.
    fn from(value: Llama3RopeScaling) -> Self {
        Self::Llama3(value)
    }
}

impl From<LongRopeScaling> for RopeScaling {
    /// Wraps already validated LongRoPE factor tables in the shared RoPE policy.
    fn from(value: LongRopeScaling) -> Self {
        Self::LongRope(value)
    }
}

impl From<ProportionalRopeScaling> for RopeScaling {
    /// Wraps already validated proportional RoPE in the shared RoPE policy.
    fn from(value: ProportionalRopeScaling) -> Self {
        Self::Proportional(value)
    }
}

/// Validated rotary-position dimensions and scaling policy.
///
/// RoPE treats channel `i` and channel `i + rotary_dimension / 2` of the rotary
/// prefix as the real and imaginary components of complex pair `i` (split-half
/// layout), hence `rotary_dimension` must be even. A partial rotary dimension
/// may be smaller than `head_dimension`; untouched suffix features pass through
/// attention without rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeConfig {
    /// Total channels per attention head, including any unrotated suffix.
    head_dimension: usize,
    /// Positive even prefix width rotated as coordinate pairs.
    rotary_dimension: usize,
    /// Positive geometric base of the unscaled inverse-frequency sequence.
    theta: f64,
    /// Validated scaling policy compatible with the rotary dimension and base.
    scaling: RopeScaling,
}

impl RopeConfig {
    /// Validates explicit head and rotary dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error for zero dimensions, an odd rotary dimension, a rotary
    /// subspace wider than the head, a base that is not finite and at least
    /// one (see the module's precision envelope), or scaling constraints that
    /// depend on the rotary dimension and base: dynamic NTK needs a rotary
    /// width above two, YaRN a base above one, LongRoPE exactly
    /// `rotary_dimension / 2` factors, and proportional RoPE a full-head
    /// rotary width.
    pub fn new(
        head_dimension: usize,
        rotary_dimension: usize,
        theta: f64,
        scaling: RopeScaling,
    ) -> Result<Self, RopeConfigError> {
        validate_dimensions(head_dimension, rotary_dimension)?;
        // `!(theta >= 1)` also rejects NaN.
        if !(theta.is_finite() && theta >= 1.0) {
            return Err(RopeConfigError::InvalidTheta { theta });
        }
        match scaling {
            RopeScaling::DynamicNtk(_) if rotary_dimension <= 2 => {
                return Err(RopeConfigError::DynamicNtkDimensionTooSmall { rotary_dimension });
            }
            // The YaRN correction-dimension formula divides by ln(theta).
            RopeScaling::Yarn(_) if theta <= 1.0 => {
                return Err(RopeConfigError::YarnThetaTooSmall { theta });
            }
            RopeScaling::LongRope(long) if long.frequency_count() != rotary_dimension / 2 => {
                return Err(RopeConfigError::LongRopeFactorCount {
                    expected: rotary_dimension / 2,
                    actual: long.frequency_count(),
                });
            }
            RopeScaling::Proportional(_) if rotary_dimension != head_dimension => {
                return Err(RopeConfigError::ProportionalRequiresFullRotation {
                    rotary_dimension,
                    head_dimension,
                });
            }
            _ => {}
        }

        Ok(Self {
            head_dimension,
            rotary_dimension,
            theta,
            scaling,
        })
    }

    /// Validates a fractional rotary subspace and constructs the configuration.
    ///
    /// The product `head_dimension * partial_rotary_factor` must be an exact
    /// even integer. Rejecting implicit rounding is important for checkpoint
    /// compatibility because projection and frequency-table shapes must agree.
    ///
    /// # Errors
    ///
    /// Returns the errors documented by
    /// [`rotary_dimension_from_partial_factor`] or [`RopeConfig::new`].
    pub fn from_partial_factor(
        head_dimension: usize,
        partial_rotary_factor: f64,
        theta: f64,
        scaling: RopeScaling,
    ) -> Result<Self, RopeConfigError> {
        let rotary_dimension =
            rotary_dimension_from_partial_factor(head_dimension, partial_rotary_factor)?;
        Self::new(head_dimension, rotary_dimension, theta, scaling)
    }

    /// Returns the complete query/key head width.
    #[must_use]
    pub const fn head_dimension(self) -> usize {
        self.head_dimension
    }

    /// Returns the even prefix width rotated by RoPE.
    #[must_use]
    pub const fn rotary_dimension(self) -> usize {
        self.rotary_dimension
    }

    /// Returns the base used by inverse rotary frequencies.
    #[must_use]
    pub const fn theta(self) -> f64 {
        self.theta
    }

    /// Returns the validated context-scaling policy.
    #[must_use]
    pub const fn scaling(self) -> RopeScaling {
        self.scaling
    }

    /// Returns the fraction of each head that is rotated.
    #[must_use]
    pub fn partial_rotary_factor(self) -> f64 {
        self.rotary_dimension as f64 / self.head_dimension as f64
    }
}

/// Converts a fractional RoPE setting into a checked even dimension.
///
/// # Errors
///
/// Returns an error for a zero head width, a non-finite factor outside `(0, 1]`,
/// a non-integral product, or a resulting zero/odd rotary dimension.
pub fn rotary_dimension_from_partial_factor(
    head_dimension: usize,
    partial_rotary_factor: f64,
) -> Result<usize, RopeConfigError> {
    if head_dimension == 0 {
        return Err(RopeConfigError::ZeroHeadDimension);
    }
    if !partial_rotary_factor.is_finite()
        || partial_rotary_factor <= 0.0
        || partial_rotary_factor > 1.0
    {
        return Err(RopeConfigError::InvalidPartialRotaryFactor {
            factor: partial_rotary_factor,
        });
    }

    let exact = head_dimension as f64 * partial_rotary_factor;
    let rounded = exact.round();
    let tolerance = f64::EPSILON * (head_dimension as f64).max(1.0) * 2.0;
    if (exact - rounded).abs() > tolerance {
        return Err(RopeConfigError::NonIntegralPartialRotaryDimension {
            head_dimension,
            factor: partial_rotary_factor,
            computed: exact,
        });
    }
    let rotary_dimension = rounded as usize;
    validate_dimensions(head_dimension, rotary_dimension)?;
    Ok(rotary_dimension)
}

/// Requires a positive, even rotary prefix that fits inside a nonempty attention head.
fn validate_dimensions(
    head_dimension: usize,
    rotary_dimension: usize,
) -> Result<(), RopeConfigError> {
    if head_dimension == 0 {
        return Err(RopeConfigError::ZeroHeadDimension);
    }
    if rotary_dimension == 0 {
        return Err(RopeConfigError::ZeroRotaryDimension);
    }
    if rotary_dimension > head_dimension {
        return Err(RopeConfigError::RotaryDimensionExceedsHead {
            rotary_dimension,
            head_dimension,
        });
    }
    if !rotary_dimension.is_multiple_of(2) {
        return Err(RopeConfigError::OddRotaryDimension { rotary_dimension });
    }
    Ok(())
}

/// Requires a finite context-extension factor of at least one; one preserves the original scale.
pub(super) fn validate_scaling_factor(
    kind: &'static str,
    factor: f64,
) -> Result<(), RopeConfigError> {
    if factor.is_finite() && factor >= 1.0 {
        Ok(())
    } else {
        Err(RopeConfigError::InvalidScalingFactor { kind, factor })
    }
}

/// Requires a nonzero trained context length before evaluating an extension formula.
pub(super) fn validate_original_context(
    kind: &'static str,
    original_max_position_embeddings: usize,
) -> Result<(), RopeConfigError> {
    if original_max_position_embeddings == 0 {
        Err(RopeConfigError::ZeroOriginalContext { kind })
    } else {
        Ok(())
    }
}

/// Whether `value` rounds to a normal positive `f32`, in `[f32::MIN_POSITIVE, f32::MAX]`.
///
/// This is the representable range for a RoPE divisor or amplitude that is
/// applied in `f32`: rounding must neither overflow to infinity nor reach a
/// subnormal or zero, whose reciprocal overflows (and which GPU kernels may
/// flush to zero). The check rounds exactly as the tables do (`value as f32`),
/// so a value a hair above `f32::MAX` that rounds down to it is accepted.
/// NaN fails both comparisons.
pub(super) fn is_normal_positive_f32(value: f64) -> bool {
    let rounded = value as f32;
    rounded.is_finite() && rounded >= f32::MIN_POSITIVE
}

/// Requires a finite positive YaRN rotation-count bound before logarithmic interpolation.
fn validate_yarn_beta(name: &'static str, value: f64) -> Result<(), RopeConfigError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(RopeConfigError::InvalidYarnBeta { name, value })
    }
}

/// Invalid rotary dimensions or context-scaling parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeConfigError {
    /// Attention head width must be non-zero.
    ZeroHeadDimension,
    /// At least one complex coordinate pair must be rotated.
    ZeroRotaryDimension,
    /// The rotary prefix cannot be wider than its attention head.
    RotaryDimensionExceedsHead {
        /// Invalid rotary prefix width.
        rotary_dimension: usize,
        /// Available head width.
        head_dimension: usize,
    },
    /// Complex rotation requires an even number of scalar coordinates.
    OddRotaryDimension {
        /// Invalid odd prefix width.
        rotary_dimension: usize,
    },
    /// The inverse-frequency base must be finite and at least one.
    ///
    /// A base below one makes `theta^(-2i/d)` grow with `i` and can overflow
    /// the `f32` frequency table; no checkpoint uses one.
    InvalidTheta {
        /// Invalid base.
        theta: f64,
    },
    /// A fractional rotary setting must be finite and in `(0, 1]`.
    InvalidPartialRotaryFactor {
        /// Invalid fraction.
        factor: f64,
    },
    /// The fractional setting would require implicit dimension rounding.
    NonIntegralPartialRotaryDimension {
        /// Full attention head width.
        head_dimension: usize,
        /// Requested rotary fraction.
        factor: f64,
        /// Unrounded product.
        computed: f64,
    },
    /// Extension factors must be finite and at least one.
    InvalidScalingFactor {
        /// Human-readable scaling algorithm.
        kind: &'static str,
        /// Invalid factor.
        factor: f64,
    },
    /// A scaling algorithm needs a non-zero trained-context limit.
    ZeroOriginalContext {
        /// Human-readable scaling algorithm.
        kind: &'static str,
    },
    /// Dynamic NTK's exponent is singular at rotary dimension two.
    DynamicNtkDimensionTooSmall {
        /// Invalid rotary dimension.
        rotary_dimension: usize,
    },
    /// YaRN's correction-dimension formula needs `ln(theta) > 0`.
    YarnThetaTooSmall {
        /// Invalid base.
        theta: f64,
    },
    /// YaRN attention correction must be a normal positive `f32` value.
    InvalidYarnAttentionFactor {
        /// Invalid correction factor.
        attention_factor: f64,
    },
    /// A YaRN rotation bound must be finite and positive.
    InvalidYarnBeta {
        /// Name of the invalid bound.
        name: &'static str,
        /// Invalid bound value.
        value: f64,
    },
    /// YaRN's fast bound must be strictly greater than its slow bound.
    InvalidYarnBetaOrder {
        /// Invalid fast bound.
        beta_fast: f64,
        /// Invalid slow bound.
        beta_slow: f64,
    },
    /// Llama 3 band multipliers must be finite, positive, and strictly ordered.
    InvalidLlama3FrequencyFactors {
        /// Multiplier defining the long-wavelength boundary.
        low_freq_factor: f64,
        /// Multiplier defining the short-wavelength boundary.
        high_freq_factor: f64,
    },
    /// LongRoPE short and long tables must describe the same pairs.
    LongRopeFactorLengthMismatch {
        /// Short-table length.
        short: usize,
        /// Long-table length.
        long: usize,
    },
    /// LongRoPE needs between one and the inline capacity of factors.
    LongRopeFactorCountOutOfRange {
        /// Supplied factor count.
        count: usize,
        /// Inline table capacity.
        maximum: usize,
    },
    /// Every LongRoPE factor must be a normal positive `f32` value.
    InvalidLongRopeFactor {
        /// Table containing the value (`short_factor` or `long_factor`).
        table: &'static str,
        /// Zero-based pair index.
        index: usize,
        /// Invalid factor as supplied.
        value: f64,
    },
    /// LongRoPE tables must provide one factor per rotary pair.
    LongRopeFactorCount {
        /// `rotary_dimension / 2`.
        expected: usize,
        /// Supplied factor count.
        actual: usize,
    },
    /// A LongRoPE amplitude must be a normal positive `f32` value.
    InvalidLongRopeAttentionFactor {
        /// Invalid amplitude.
        attention_factor: f64,
    },
    /// The proportional rotated fraction must be finite and within `[0, 1]`.
    InvalidProportionalFraction {
        /// Invalid fraction.
        fraction: f64,
    },
    /// Proportional RoPE pairs span the whole head, so the rotary width must equal it.
    ProportionalRequiresFullRotation {
        /// Configured rotary width.
        rotary_dimension: usize,
        /// Attention head width.
        head_dimension: usize,
    },
}

impl fmt::Display for RopeConfigError {
    /// Formats invalid rotary geometry or scaling parameters with the offending values.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroHeadDimension => formatter.write_str("RoPE head dimension must be non-zero"),
            Self::ZeroRotaryDimension => {
                formatter.write_str("RoPE rotary dimension must be non-zero")
            }
            Self::RotaryDimensionExceedsHead {
                rotary_dimension,
                head_dimension,
            } => write!(
                formatter,
                "RoPE rotary dimension {rotary_dimension} exceeds head dimension {head_dimension}"
            ),
            Self::OddRotaryDimension { rotary_dimension } => write!(
                formatter,
                "RoPE rotary dimension {rotary_dimension} must be even"
            ),
            Self::InvalidTheta { theta } => {
                write!(
                    formatter,
                    "RoPE theta must be finite and at least 1 (a smaller base overflows the f32 frequencies), got {theta}"
                )
            }
            Self::InvalidPartialRotaryFactor { factor } => write!(
                formatter,
                "partial rotary factor must be finite and in (0, 1], got {factor}"
            ),
            Self::NonIntegralPartialRotaryDimension {
                head_dimension,
                factor,
                computed,
            } => write!(
                formatter,
                "head dimension {head_dimension} times partial rotary factor {factor} is non-integral ({computed})"
            ),
            Self::InvalidScalingFactor { kind, factor } => write!(
                formatter,
                "{kind} scaling factor must be finite and at least 1, got {factor}"
            ),
            Self::ZeroOriginalContext { kind } => {
                write!(formatter, "{kind} original context must be non-zero")
            }
            Self::DynamicNtkDimensionTooSmall { rotary_dimension } => write!(
                formatter,
                "dynamic NTK requires rotary dimension greater than 2, got {rotary_dimension}"
            ),
            Self::YarnThetaTooSmall { theta } => {
                write!(formatter, "YaRN requires theta greater than 1, got {theta}")
            }
            Self::InvalidYarnAttentionFactor { attention_factor } => write!(
                formatter,
                "YaRN attention factor must be a normal positive f32 value ({:e}..={:e}), got {attention_factor}",
                f32::MIN_POSITIVE,
                f32::MAX
            ),
            Self::InvalidYarnBeta { name, value } => write!(
                formatter,
                "YaRN {name} must be finite and positive, got {value}"
            ),
            Self::InvalidYarnBetaOrder {
                beta_fast,
                beta_slow,
            } => write!(
                formatter,
                "YaRN beta_fast ({beta_fast}) must be greater than beta_slow ({beta_slow})"
            ),
            Self::InvalidLlama3FrequencyFactors {
                low_freq_factor,
                high_freq_factor,
            } => write!(
                formatter,
                "Llama 3 low_freq_factor ({low_freq_factor}) and high_freq_factor ({high_freq_factor}) must be finite, positive, and strictly increasing"
            ),
            Self::LongRopeFactorLengthMismatch { short, long } => write!(
                formatter,
                "LongRoPE short_factor has {short} entries but long_factor has {long}"
            ),
            Self::LongRopeFactorCountOutOfRange { count, maximum } => write!(
                formatter,
                "LongRoPE needs 1..={maximum} factors per table, got {count}"
            ),
            Self::InvalidLongRopeFactor {
                table,
                index,
                value,
            } => write!(
                formatter,
                "LongRoPE {table}[{index}] must round to an f32 value in {:e}..={:e} (2^-64 to f32::MAX), got {value}",
                LongRopeScaling::MIN_FACTOR,
                f32::MAX
            ),
            Self::LongRopeFactorCount { expected, actual } => write!(
                formatter,
                "LongRoPE needs one factor per rotary pair ({expected}), got {actual}"
            ),
            Self::InvalidLongRopeAttentionFactor { attention_factor } => write!(
                formatter,
                "LongRoPE attention factor must be a normal positive f32 value ({:e}..={:e}), got {attention_factor}",
                f32::MIN_POSITIVE,
                f32::MAX
            ),
            Self::InvalidProportionalFraction { fraction } => write!(
                formatter,
                "proportional RoPE rotated fraction must be finite and in [0, 1], got {fraction}"
            ),
            Self::ProportionalRequiresFullRotation {
                rotary_dimension,
                head_dimension,
            } => write!(
                formatter,
                "proportional RoPE rotates the full head ({head_dimension}), got rotary dimension {rotary_dimension}"
            ),
        }
    }
}

impl std::error::Error for RopeConfigError {}

#[cfg(test)]
#[path = "rope_tests.rs"]
mod tests;
