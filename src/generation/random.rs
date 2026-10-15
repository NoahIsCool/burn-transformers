//! Counter-based SplitMix64 streams for reproducible generation randomness.
//!
//! Burn's backend seed is process-global on several backends, so generation
//! never uses it: an unrelated model or a concurrently running test could
//! otherwise perturb a seeded run. Every random decision instead comes from
//! SplitMix64 (Steele, Lea, and Flood, 2014) evaluated at an explicit counter:
//!
//! ```text
//! x_k = mix(seed + gamma * (k + 1))      (mod 2^64)
//! ```
//!
//! where `gamma = 0x9E3779B97F4A7C15` is odd, so `k -> x_k` is a bijection on
//! 64-bit integers, and `mix` is the fixed SplitMix64 finalizer. Counter-based
//! evaluation lets a processor reserve a disjoint block of draws for a whole
//! batch with one atomic add, and lets a snapshot of the counter restore a
//! stream exactly after a rolled-back speculative cycle.
//!
//! Two output conventions exist, each chosen for its consumer:
//!
//! - [`interior_unit_f32`] keeps the top 23 bits and centers them in their
//!   interval, `u = (b + 1/2) / 2^23`. Every value is exactly representable in
//!   F32 and lies strictly inside `(0, 1)`, so a backend inverse-CDF search
//!   never sees the boundary values zero or one.
//! - [`unit_f64`] keeps the top 53 bits, `u = b / 2^53` in `[0, 1)`, which fits
//!   the F64 mantissa exactly; host-side rejection sampling uses it.

/// Odd Weyl increment of SplitMix64, `floor(2^64 / golden_ratio)`.
pub(crate) const SPLITMIX64_INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;

/// Returns output `index` of the SplitMix64 stream seeded with `seed`.
///
/// Equals the `(index + 1)`-th call of a sequential SplitMix64 generator
/// whose state starts at `seed`; all arithmetic wraps modulo `2^64`.
pub(crate) fn splitmix64(seed: u64, index: u64) -> u64 {
    let mut value = seed.wrapping_add(SPLITMIX64_INCREMENT.wrapping_mul(index.wrapping_add(1)));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Maps mixed bits to an F32 uniform strictly inside `(0, 1)`.
///
/// The upper 23 bits select one of `2^23` equal-width intervals and the value
/// is that interval's midpoint, so the smallest output is `2^-24` and the
/// largest is `1 - 2^-24`; both are exact in F32.
pub(crate) fn interior_unit_f32(bits: u64) -> f32 {
    let bucket = bits >> 41;
    ((bucket as f64 + 0.5) / ((1_u64 << 23) as f64)) as f32
}

/// Maps mixed bits to an F64 uniform in `[0, 1)` using the upper 53 bits.
///
/// The largest output is `1 - 2^-53`, so `u * total < total` for any positive
/// finite F64 `total`: an inverse-CDF scan always stops inside the support.
pub(crate) fn unit_f64(bits: u64) -> f64 {
    ((bits >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

/// Sequential, copyable SplitMix64 stream used for host-side random decisions.
///
/// The stream is a `(seed, position)` pair: draw `k` is
/// `splitmix64(seed, k)`. Copying the value snapshots the stream, which makes
/// transactional rollback trivial. Draws are bit-identical to the classic
/// state-incrementing SplitMix64 implementation that starts at `seed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    /// Seed that selects the stream.
    seed: u64,
    /// Index of the next draw.
    position: u64,
}

impl SplitMix64 {
    /// Starts the stream selected by `seed` at its first draw.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed, position: 0 }
    }

    /// Returns the number of draws taken so far, modulo `2^64`.
    #[must_use]
    pub const fn position(&self) -> u64 {
        self.position
    }

    /// Returns the next raw 64-bit draw.
    pub fn next_u64(&mut self) -> u64 {
        let value = splitmix64(self.seed, self.position);
        self.position = self.position.wrapping_add(1);
        value
    }

    /// Returns the next F64 uniform in `[0, 1)` with 53 random bits.
    pub fn next_unit_f64(&mut self) -> f64 {
        unit_f64(self.next_u64())
    }
}

#[cfg(test)]
#[path = "random_tests.rs"]
mod tests;
