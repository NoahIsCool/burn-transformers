//! Per-frequency RoPE context-extension policies: Llama 3, LongRoPE, and proportional RoPE.
//!
//! Every policy here rescales the base inverse frequencies
//! `omega_i = theta^(-2i/d)` of a rotary width `d` independently for each
//! complex pair `i in 0..d/2`. The equations follow Hugging Face Transformers
//! 5.2 `modeling_rope_utils.py` (`_compute_llama3_parameters`,
//! `_compute_longrope_parameters`) and, for proportional RoPE, which 5.2 lacks,
//! Transformers 5.8.0 `_compute_proportional_rope_parameters`:
//!
//! * **Llama 3** keeps short wavelengths `lambda_i = 2 pi / omega_i` below
//!   `L / high_freq_factor`, divides long wavelengths above `L / low_freq_factor`
//!   by `factor`, and blends linearly in between with
//!   `s_i = (L / lambda_i - low) / (high - low)`, where `L` is the original
//!   training context: `omega'_i = (1 - s_i) omega_i / factor + s_i omega_i`.
//! * **LongRoPE** divides each frequency by a searched per-pair factor,
//!   `omega'_i = omega_i / r_i`, using the *short* table while the evaluated
//!   context is at most `L` and the *long* table beyond it. The cosine/sine
//!   tables are scaled by `attention_factor`, by default
//!   `sqrt(1 + ln(factor) / ln(L))` for `factor > 1`.
//! * **Proportional** RoPE rotates the whole head in split-half layout but
//!   assigns non-zero frequencies only to the first
//!   `floor(fraction * head_dim / 2)` pairs; the remaining pairs have frequency
//!   zero, so their cosine is one and their sine is zero (no rotation).
//!
//! All parameters are host `f64` scalars validated on construction. LongRoPE
//! factor tables are stored as `f32`, the precision in which the reference
//! materializes them, inside a fixed-capacity inline array. The inline storage
//! keeps [`super::RopeConfig`] `Copy`—it is passed by value through every
//! attention layer—at the cost of a constant 1 KiB per configuration value.
//!
//! # Verified counts
//!
//! The integer steps are checked in place by Verus through
//! `verus/position.rs`: [`long_rope_frequency_count`] accepts exactly equal
//! table lengths in `1..=MAX_LONG_ROPE_FREQUENCIES` (the invariant that keeps
//! [`LongRopeScaling::short_factors`] and [`LongRopeScaling::long_factors`]
//! inside their inline arrays), and [`clamp_rotated_pairs`] bounds the
//! proportional pair count by `head_dimension / 2`.

use core::fmt;

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

use super::rope::{
    RopeConfigError, is_normal_positive_f32, validate_original_context, validate_scaling_factor,
};

/// Largest number of complex frequency pairs a [`LongRopeScaling`] table can hold.
///
/// This bounds the rotary width to `2 * 128 = 256` channels. Every published
/// LongRoPE checkpoint (Phi-3, Phi-3.5, Phi-4-mini, MiniCPM) uses at most 64
/// pairs; wider tables are rejected with
/// [`RopeConfigError::LongRopeFactorCountOutOfRange`].
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const MAX_LONG_ROPE_FREQUENCIES: usize = 128;

/// Validated Llama 3 smooth low/high-frequency RoPE interpolation.
///
/// Wavelengths shorter than `original / high_freq_factor` keep their trained
/// frequency, wavelengths longer than `original / low_freq_factor` are divided
/// by `factor`, and the band between them interpolates linearly in
/// `original / wavelength`. The cosine/sine amplitude stays one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Llama3RopeScaling {
    /// Finite division applied to low-frequency pairs, at least one.
    factor: f64,
    /// Positive multiplier defining the long-wavelength boundary `original / low`.
    low_freq_factor: f64,
    /// Multiplier strictly above `low_freq_factor` defining the short-wavelength boundary.
    high_freq_factor: f64,
    /// Positive context length used during pretraining.
    original_max_position_embeddings: usize,
}

impl Llama3RopeScaling {
    /// Creates Llama 3 scaling, for example `(8.0, 1.0, 4.0, 8192)` for Llama 3.1.
    ///
    /// # Errors
    ///
    /// Returns an error unless `factor` is finite and at least one, both
    /// frequency factors are finite and positive with `low < high`, and the
    /// original context is non-zero. Equal factors would make the smoothing
    /// denominator `high - low` zero.
    pub fn new(
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: usize,
    ) -> Result<Self, RopeConfigError> {
        validate_scaling_factor("Llama 3", factor)?;
        validate_original_context("Llama 3", original_max_position_embeddings)?;
        let valid_factor = |value: f64| value.is_finite() && value > 0.0;
        if !valid_factor(low_freq_factor)
            || !valid_factor(high_freq_factor)
            || low_freq_factor >= high_freq_factor
        {
            return Err(RopeConfigError::InvalidLlama3FrequencyFactors {
                low_freq_factor,
                high_freq_factor,
            });
        }
        Ok(Self {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        })
    }

    /// Returns the division applied to fully interpolated low frequencies.
    #[must_use]
    pub const fn factor(self) -> f64 {
        self.factor
    }

    /// Returns the multiplier that defines the long-wavelength boundary.
    #[must_use]
    pub const fn low_freq_factor(self) -> f64 {
        self.low_freq_factor
    }

    /// Returns the multiplier that defines the short-wavelength boundary.
    #[must_use]
    pub const fn high_freq_factor(self) -> f64 {
        self.high_freq_factor
    }

    /// Returns the maximum position used during pretraining.
    #[must_use]
    pub const fn original_max_position_embeddings(self) -> usize {
        self.original_max_position_embeddings
    }

    /// Rescales one unscaled inverse frequency `omega > 0` with the Llama 3 rule.
    ///
    /// The comparisons are strict exactly as in the reference: a wavelength
    /// equal to a boundary belongs to the interpolated middle band.
    pub(super) fn scale_frequency(self, inverse_frequency: f64) -> f64 {
        let original = self.original_max_position_embeddings as f64;
        let low_frequency_wavelength = original / self.low_freq_factor;
        let high_frequency_wavelength = original / self.high_freq_factor;
        let wavelength = core::f64::consts::TAU / inverse_frequency;
        if wavelength > low_frequency_wavelength {
            inverse_frequency / self.factor
        } else if wavelength < high_frequency_wavelength {
            inverse_frequency
        } else {
            let smooth = (original / wavelength - self.low_freq_factor)
                / (self.high_freq_factor - self.low_freq_factor);
            (1.0 - smooth) * inverse_frequency / self.factor + smooth * inverse_frequency
        }
    }
}

/// Validated LongRoPE per-frequency factors and amplitude.
///
/// Phi-3-style checkpoints search one rescaling factor per complex pair. The
/// short table applies while the evaluated context (largest position plus one)
/// is at most `original_max_position_embeddings`; the long table applies beyond
/// it. Because the selected table depends on the context, cached decoding that
/// crosses the boundary would leave earlier keys rotated with the other table;
/// see [`super::RopeScaling::cached_context_limit`].
#[derive(Clone, Copy, PartialEq)]
pub struct LongRopeScaling {
    /// Short-context factors; only the first `frequency_count` entries are meaningful.
    short_factors: [f32; MAX_LONG_ROPE_FREQUENCIES],
    /// Long-context factors; only the first `frequency_count` entries are meaningful.
    long_factors: [f32; MAX_LONG_ROPE_FREQUENCIES],
    /// Number of complex pairs described by both tables, in `1..=MAX_LONG_ROPE_FREQUENCIES`.
    frequency_count: usize,
    /// Positive pretraining context that separates the short and long regimes.
    original_max_position_embeddings: usize,
    /// Finite context-extension factor (at least one) used for the default amplitude.
    factor: f64,
    /// Positive multiplier applied to the cosine and sine tables.
    attention_factor: f64,
}

impl LongRopeScaling {
    /// Smallest accepted factor, `2^-64` (about `5.4e-20`), after rounding to `f32`.
    ///
    /// A factor `r` divides a base frequency of at most one (bases are at
    /// least one), so `omega / r <= 2^64`. The device forms the angle
    /// `p * omega / r` in `f32`, and every integer position satisfies
    /// `|p| <= 2^63`, so the angle stays below `2^127 < f32::MAX` and its
    /// cosine and sine are finite. A smaller normal factor such as
    /// `f32::MIN_POSITIVE` keeps the frequency itself finite (`8.5e37`) but
    /// overflows the angle from position 4 on, and `cos(inf)` is NaN.
    /// Published checkpoints use factors of at least one.
    pub const MIN_FACTOR: f32 = f32::from_bits(63 << 23);

    /// Creates LongRoPE scaling with the reference default amplitude.
    ///
    /// `factor` is the checkpoint's context-extension factor; configs that omit
    /// it use `max_position_embeddings / original_max_position_embeddings`,
    /// which the config adapter must resolve. The default `attention_factor` is
    /// one when `factor <= 1` and `sqrt(1 + ln(factor) / ln(original))`
    /// otherwise. Factors are rounded once to `f32`.
    ///
    /// # Errors
    ///
    /// Returns an error when the tables are empty, longer than
    /// [`MAX_LONG_ROPE_FREQUENCIES`], of unequal length, or contain a value that
    /// does not round to an `f32` from [`Self::MIN_FACTOR`] (`2^-64`) to
    /// [`f32::MAX`]; when `factor` is not finite and at least one; or when the
    /// original context is zero. That range keeps every angle
    /// `p * omega_i / r_i` finite in `f32` for bases `theta >= 1` (see
    /// [`super::RopeConfig::new`]) at every integer position `p`.
    pub fn new(
        short_factor: &[f64],
        long_factor: &[f64],
        original_max_position_embeddings: usize,
        factor: f64,
    ) -> Result<Self, RopeConfigError> {
        validate_scaling_factor("LongRoPE", factor)?;
        validate_original_context("LongRoPE", original_max_position_embeddings)?;
        let frequency_count = long_rope_frequency_count(short_factor.len(), long_factor.len())?;
        Ok(Self {
            short_factors: factor_table("short_factor", short_factor)?,
            long_factors: factor_table("long_factor", long_factor)?,
            frequency_count,
            original_max_position_embeddings,
            factor,
            attention_factor: default_long_rope_attention_factor(
                factor,
                original_max_position_embeddings,
            ),
        })
    }

    /// Replaces the cosine/sine amplitude with an explicit checkpoint value.
    ///
    /// # Errors
    ///
    /// Returns [`RopeConfigError::InvalidLongRopeAttentionFactor`] unless the
    /// value rounds to a normal positive `f32` (from [`f32::MIN_POSITIVE`] to
    /// [`f32::MAX`]), the precision of the cosine/sine tables it scales.
    pub fn with_attention_factor(mut self, attention_factor: f64) -> Result<Self, RopeConfigError> {
        if !is_normal_positive_f32(attention_factor) {
            return Err(RopeConfigError::InvalidLongRopeAttentionFactor { attention_factor });
        }
        self.attention_factor = attention_factor;
        Ok(self)
    }

    /// Returns the factors used while the evaluated context is at most the original context.
    #[must_use]
    pub fn short_factors(&self) -> &[f32] {
        &self.short_factors[..self.frequency_count]
    }

    /// Returns the factors used once the evaluated context exceeds the original context.
    #[must_use]
    pub fn long_factors(&self) -> &[f32] {
        &self.long_factors[..self.frequency_count]
    }

    /// Returns the number of complex pairs, which must equal `rotary_dimension / 2`.
    #[must_use]
    pub const fn frequency_count(&self) -> usize {
        self.frequency_count
    }

    /// Returns the pretraining context separating the short and long tables.
    #[must_use]
    pub const fn original_max_position_embeddings(&self) -> usize {
        self.original_max_position_embeddings
    }

    /// Returns the context-extension factor that determined the default amplitude.
    #[must_use]
    pub const fn factor(&self) -> f64 {
        self.factor
    }

    /// Returns the multiplier applied to both cosine and sine tables.
    #[must_use]
    pub const fn attention_factor(&self) -> f64 {
        self.attention_factor
    }

    /// Selects the factor table for an evaluated context (largest position plus one).
    ///
    /// The reference switches strictly after the original context:
    /// `context > original` selects the long table.
    #[must_use]
    pub fn factors_for_context(&self, evaluated_context: usize) -> &[f32] {
        if evaluated_context > self.original_max_position_embeddings {
            self.long_factors()
        } else {
            self.short_factors()
        }
    }
}

impl fmt::Debug for LongRopeScaling {
    /// Prints only the meaningful prefix of each inline factor table.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LongRopeScaling")
            .field("short_factors", &self.short_factors())
            .field("long_factors", &self.long_factors())
            .field(
                "original_max_position_embeddings",
                &self.original_max_position_embeddings,
            )
            .field("factor", &self.factor)
            .field("attention_factor", &self.attention_factor)
            .finish()
    }
}

/// Whether `value` is an accepted LongRoPE factor: finite in `f32` and at least [`LongRopeScaling::MIN_FACTOR`].
///
/// The check rounds exactly as the stored table does (`value as f32`), so a
/// huge `f64` cannot overflow to infinity, a tiny one can neither flush to a
/// zero divisor nor (as a subnormal such as `1e-45`) overflow the frequency,
/// and no accepted factor can overflow an angle `p * omega / r`. NaN fails
/// both comparisons.
fn is_long_rope_factor(value: f64) -> bool {
    let rounded = value as f32;
    rounded.is_finite() && rounded >= LongRopeScaling::MIN_FACTOR
}

/// Copies a checked factor list into fixed-capacity `f32` storage.
///
/// Each value is rounded to `f32` first and must pass
/// [`is_long_rope_factor`].
fn factor_table(
    table: &'static str,
    values: &[f64],
) -> Result<[f32; MAX_LONG_ROPE_FREQUENCIES], RopeConfigError> {
    let mut stored = [1.0_f32; MAX_LONG_ROPE_FREQUENCIES];
    for (index, (&value, slot)) in values.iter().zip(stored.iter_mut()).enumerate() {
        if !is_long_rope_factor(value) {
            return Err(RopeConfigError::InvalidLongRopeFactor {
                table,
                index,
                value,
            });
        }
        *slot = value as f32;
    }
    Ok(stored)
}

/// Returns LongRoPE's default amplitude `sqrt(1 + ln(factor) / ln(original))`.
///
/// The reference returns one for `factor <= 1`. An original context of one
/// has `ln(1) = 0`; the expression is then infinite, so it also falls back to
/// one rather than producing a non-finite table.
fn default_long_rope_attention_factor(factor: f64, original_max_position_embeddings: usize) -> f64 {
    if factor <= 1.0 || original_max_position_embeddings <= 1 {
        return 1.0;
    }
    (1.0 + factor.ln() / (original_max_position_embeddings as f64).ln()).sqrt()
}

/// Validated proportional RoPE: a full-head split-half rotation with a zero-frequency tail.
///
/// The first `floor(rotated_fraction * head_dimension / 2)` complex pairs use
/// `theta^(-2i/head_dimension) / factor`; the rest have frequency zero and are
/// therefore not rotated. Pairs are `(i, i + head_dimension / 2)`, so the
/// unrotated channels are interleaved with rotated ones rather than forming a
/// contiguous suffix; [`super::RopeConfig`] accordingly requires the rotary
/// width to equal the full head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProportionalRopeScaling {
    /// Fraction of complex pairs that rotate, in `[0, 1]`.
    rotated_fraction: f64,
    /// Finite position divisor applied to rotated frequencies, at least one.
    factor: f64,
}

impl ProportionalRopeScaling {
    /// Creates proportional RoPE from a checkpoint's `partial_rotary_factor` and `factor`.
    ///
    /// A fraction of zero rotates nothing, which is equivalent to NoPE.
    ///
    /// # Errors
    ///
    /// Returns an error unless `rotated_fraction` is finite and in `[0, 1]` and
    /// `factor` is finite and at least one.
    pub fn new(rotated_fraction: f64, factor: f64) -> Result<Self, RopeConfigError> {
        if !(rotated_fraction.is_finite() && (0.0..=1.0).contains(&rotated_fraction)) {
            return Err(RopeConfigError::InvalidProportionalFraction {
                fraction: rotated_fraction,
            });
        }
        validate_scaling_factor("proportional", factor)?;
        Ok(Self {
            rotated_fraction,
            factor,
        })
    }

    /// Returns the fraction of complex pairs that rotate.
    #[must_use]
    pub const fn rotated_fraction(self) -> f64 {
        self.rotated_fraction
    }

    /// Returns the position divisor applied to rotated frequencies.
    #[must_use]
    pub const fn factor(self) -> f64 {
        self.factor
    }

    /// Returns how many leading complex pairs rotate for a head width.
    ///
    /// This is the reference `int(fraction * head_dim // 2)`; because the
    /// fraction is at most one the result never exceeds `head_dimension / 2`.
    #[must_use]
    pub fn rotated_pairs(self, head_dimension: usize) -> usize {
        let pairs = (self.rotated_fraction * head_dimension as f64 / 2.0).floor() as usize;
        clamp_rotated_pairs(pairs, head_dimension)
    }
}

/// Validates the LongRoPE table lengths and returns the shared frequency count.
///
/// Verified: `Ok(count)` exactly when both tables have the same length
/// `count` in `1..=MAX_LONG_ROPE_FREQUENCIES`, so every accessor slice
/// `[..count]` of the inline arrays is in bounds; otherwise the length
/// mismatch is reported before the count range.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Ok(count) => {
                &&& count == short_length
                &&& short_length == long_length
                &&& 1 <= count <= MAX_LONG_ROPE_FREQUENCIES
            },
            Err(RopeConfigError::LongRopeFactorLengthMismatch { short, long }) => {
                short == short_length && long == long_length && short != long
            },
            Err(RopeConfigError::LongRopeFactorCountOutOfRange { count, maximum }) => {
                &&& short_length == long_length
                &&& count == short_length
                &&& maximum == MAX_LONG_ROPE_FREQUENCIES
                &&& (count == 0 || count > MAX_LONG_ROPE_FREQUENCIES)
            },
            Err(_) => false,
        }
))]
fn long_rope_frequency_count(
    short_length: usize,
    long_length: usize,
) -> Result<usize, RopeConfigError> {
    if short_length != long_length {
        return Err(RopeConfigError::LongRopeFactorLengthMismatch {
            short: short_length,
            long: long_length,
        });
    }
    if short_length == 0 || short_length > MAX_LONG_ROPE_FREQUENCIES {
        return Err(RopeConfigError::LongRopeFactorCountOutOfRange {
            count: short_length,
            maximum: MAX_LONG_ROPE_FREQUENCIES,
        });
    }
    Ok(short_length)
}

/// Clamps a proposed proportional pair count to the `head_dimension / 2` pairs of a head.
///
/// Verified: the result is exactly `min(pairs, head_dimension / 2)`, so the
/// rotated pairs never exceed the head's split-half pairs.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rotated =>
    ensures
        rotated == if pairs <= head_dimension / 2 { pairs } else { head_dimension / 2 },
        2 * rotated <= head_dimension,
))]
fn clamp_rotated_pairs(pairs: usize, head_dimension: usize) -> usize {
    pairs.min(head_dimension / 2)
}

#[cfg(test)]
#[path = "rope_scaling_tests.rs"]
mod tests;
