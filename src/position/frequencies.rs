//! Host computation of rotary inverse frequencies and table amplitude.
//!
//! For rotary width `d`, pair `i in 0..d/2` has base inverse frequency
//! `omega_i = theta^(-2i/d)` radians per position. Each [`RopeScaling`] policy
//! rewrites these `d/2` scalars (see the table in the `rope` module); the
//! tensor module then forms angles `p * omega_i` on the device.
//!
//! Frequencies are evaluated in `f64` and rounded once to `f32`. The
//! Hugging Face reference evaluates the same expressions directly in `f32`, so
//! results agree to a few `f32` ulps (golden tests use a relative tolerance of
//! `2e-6`). Computing `d/2 <= 128` powers on the host is negligible next to the
//! angle tables; [`RotaryFrequencies`] lets callers compute them once per model
//! or decode step and reuse them across layers.

use core::f64::consts::TAU;

use crate::{Error, Result};

use super::{
    LongRopeScaling, ProportionalRopeScaling, RopeConfig, RopeScaling, YarnConfig,
    rope_scaling::Llama3RopeScaling,
};

/// Host inverse frequencies and cosine/sine amplitude for one evaluated context.
///
/// This is the device-independent half of a rotary table. Build it once per
/// model (or once per decode step for context-dependent policies, see
/// [`RopeScaling::is_context_dependent`]) and pass it to
/// [`super::rotary_angles_from_frequencies`] for every layer that shares the
/// configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RotaryFrequencies {
    /// Inverse frequencies `omega'_i` in radians per position, one per complex pair.
    inverse: Vec<f32>,
    /// Multiplier applied to both cosine and sine tables.
    amplitude: f64,
    /// Rotary prefix width `2 * inverse.len()`.
    rotary_dimension: usize,
}

impl RotaryFrequencies {
    /// Evaluates a validated configuration at an evaluated context.
    ///
    /// `evaluated_context` is the largest logical position plus one, not merely
    /// the current tensor width when decoding from a cache. Only dynamic NTK
    /// and LongRoPE read it; it must still be non-zero for every policy.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when `evaluated_context` is zero.
    pub fn new(config: RopeConfig, evaluated_context: usize) -> Result<Self> {
        Ok(Self {
            inverse: inverse_frequencies(config, evaluated_context)?,
            amplitude: config.scaling().amplitude(),
            rotary_dimension: config.rotary_dimension(),
        })
    }

    /// Returns the `rotary_dimension / 2` inverse frequencies.
    #[must_use]
    pub fn inverse(&self) -> &[f32] {
        &self.inverse
    }

    /// Returns the multiplier applied to cosine and sine tables.
    #[must_use]
    pub const fn amplitude(&self) -> f64 {
        self.amplitude
    }

    /// Returns the rotated prefix width.
    #[must_use]
    pub const fn rotary_dimension(&self) -> usize {
        self.rotary_dimension
    }
}

/// Computes inverse angular frequencies for a concrete evaluated context.
///
/// `evaluated_context` is the largest logical position plus one, not merely the
/// current tensor width when decoding from a cache. It affects dynamic NTK and
/// LongRoPE only, but must be non-zero for every policy.
///
/// # Errors
///
/// Returns a configuration error when `evaluated_context` is zero.
pub fn inverse_frequencies(config: RopeConfig, evaluated_context: usize) -> Result<Vec<f32>> {
    if evaluated_context == 0 {
        return Err(Error::configuration(
            "rope.evaluated_context",
            "must be non-zero",
        ));
    }
    let dimension = config.rotary_dimension();
    let theta = config.theta();
    let frequencies = (0..dimension / 2).map(|index| base_frequency(theta, index, dimension));
    let scaled: Vec<f64> = match config.scaling() {
        RopeScaling::None => frequencies.collect(),
        RopeScaling::Linear(linear) => frequencies.map(|omega| omega / linear.factor()).collect(),
        RopeScaling::DynamicNtk(dynamic) => {
            let original = dynamic.original_max_position_embeddings();
            if evaluated_context <= original {
                frequencies.collect()
            } else {
                // Grow the base so the longest wavelength covers the new context.
                let context_ratio = evaluated_context as f64 / original as f64;
                let stretch = dynamic.factor() * context_ratio - (dynamic.factor() - 1.0);
                let exponent = dimension as f64 / (dimension as f64 - 2.0);
                let base = theta * stretch.powf(exponent);
                (0..dimension / 2)
                    .map(|index| base_frequency(base, index, dimension))
                    .collect()
            }
        }
        RopeScaling::Yarn(yarn) => yarn_frequencies(yarn, theta, dimension),
        RopeScaling::Llama3(llama) => llama3_frequencies(llama, frequencies),
        RopeScaling::LongRope(long) => long_rope_frequencies(&long, frequencies, evaluated_context),
        RopeScaling::Proportional(proportional) => {
            proportional_frequencies(proportional, frequencies, dimension)
        }
    };
    Ok(scaled.into_iter().map(|omega| omega as f32).collect())
}

/// Returns the unscaled inverse frequency `base^(-2 index / dimension)`.
fn base_frequency(base: f64, index: usize, dimension: usize) -> f64 {
    base.powf(-(2.0 * index as f64) / dimension as f64)
}

/// Blends interpolated and extrapolated frequencies over YaRN's correction band.
///
/// With `omega_i` the base frequency, `ramp_i = clamp((i - low) / (high - low), 0, 1)`
/// and extrapolation weight `w_i = 1 - ramp_i`, the result is
/// `omega_i / factor * (1 - w_i) + omega_i * w_i`. High-frequency pairs below
/// `low` keep `omega_i`; low-frequency pairs above `high` are interpolated.
fn yarn_frequencies(yarn: YarnConfig, theta: f64, dimension: usize) -> Vec<f64> {
    let (low, high) = yarn_correction_range(yarn, theta, dimension);
    (0..dimension / 2)
        .map(|index| {
            let extrapolated = base_frequency(theta, index, dimension);
            let interpolated = extrapolated / yarn.factor();
            let ramp = ((index as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let extrapolation_weight = 1.0 - ramp;
            interpolated * (1.0 - extrapolation_weight) + extrapolated * extrapolation_weight
        })
        .collect()
}

/// Returns YaRN's ramp bounds `(low, high)` exactly as the reference computes them.
///
/// Truncation floors `low` and ceils `high`. Both are then clamped to
/// `[0, dimension - 1]`—the full rotary width, not the pair count—and a
/// degenerate range receives `high += 0.001` to avoid division by zero.
pub(super) fn yarn_correction_range(yarn: YarnConfig, theta: f64, dimension: usize) -> (f64, f64) {
    let original = yarn.original_max_position_embeddings();
    let mut low = yarn_correction_dimension(yarn.beta_fast(), dimension, theta, original);
    let mut high = yarn_correction_dimension(yarn.beta_slow(), dimension, theta, original);
    if yarn.truncate() {
        low = low.floor();
        high = high.ceil();
    }
    let low = low.max(0.0);
    let mut high = high.min(dimension as f64 - 1.0);
    if low == high {
        high += 0.001;
    }
    (low, high)
}

/// Finds the YaRN frequency index associated with a rotation-count boundary.
///
/// Pair `i` completes `L * omega_i / (2 pi)` rotations over the original
/// context `L`; solving `L * theta^(-2i/d) / (2 pi) = rotations` for `i`
/// gives `d * ln(L / (2 pi rotations)) / (2 ln theta)`.
fn yarn_correction_dimension(
    rotations: f64,
    dimension: usize,
    theta: f64,
    original_context: usize,
) -> f64 {
    dimension as f64 * (original_context as f64 / (rotations * TAU)).ln() / (2.0 * theta.ln())
}

/// Applies Llama 3 wavelength-banded interpolation to every base frequency.
fn llama3_frequencies(
    llama: Llama3RopeScaling,
    frequencies: impl Iterator<Item = f64>,
) -> Vec<f64> {
    frequencies
        .map(|omega| llama.scale_frequency(omega))
        .collect()
}

/// Divides every base frequency by the LongRoPE factor selected for the context.
fn long_rope_frequencies(
    long: &LongRopeScaling,
    frequencies: impl Iterator<Item = f64>,
    evaluated_context: usize,
) -> Vec<f64> {
    frequencies
        .zip(long.factors_for_context(evaluated_context))
        .map(|(omega, &factor)| omega / f64::from(factor))
        .collect()
}

/// Keeps the leading proportional pairs (divided by `factor`) and zeroes the rest.
///
/// Proportional RoPE always spans the full head, so `dimension` is the head width.
fn proportional_frequencies(
    proportional: ProportionalRopeScaling,
    frequencies: impl Iterator<Item = f64>,
    dimension: usize,
) -> Vec<f64> {
    let rotated = proportional.rotated_pairs(dimension);
    frequencies
        .enumerate()
        .map(|(index, omega)| {
            if index < rotated {
                omega / proportional.factor()
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "frequencies_tests.rs"]
mod tests;
