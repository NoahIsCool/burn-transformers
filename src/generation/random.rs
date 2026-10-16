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
//!
//! # Verified properties
//!
//! [`splitmix64`], the stream constructor, [`SplitMix64::position`], and
//! [`SplitMix64::next_u64`] are checked in place by Verus through
//! `verus/generation.rs` (see `verus/README.md`). The proofs show that:
//!
//! - [`splitmix64`] computes exactly `mix(seed + gamma * (index + 1) mod 2^64)`;
//! - this map is injective in `index` for every seed, so the `2^64` draws of a
//!   stream are pairwise distinct and disjoint counter blocks never repeat a
//!   raw draw;
//! - advancing the counter adds `gamma` to the Weyl state, which is the classic
//!   sequential generator;
//! - [`SplitMix64::new`] starts at position zero with the given seed,
//!   [`SplitMix64::position`] reports that counter, and
//!   [`SplitMix64::next_u64`] returns the draw at its position and advances the
//!   position by exactly one. Composed, the `k`-th draw of
//!   `SplitMix64::new(seed)` is `splitmix64(seed, k)`.
//!
//! The float conversions are not covered; the Rust tests check them.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Odd Weyl increment of SplitMix64, `floor(2^64 / golden_ratio)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const SPLITMIX64_INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;

#[cfg(verus_keep_ghost)]
verus! {

/// `2^64`, the modulus of every wrapping operation below.
pub open spec fn word() -> int {
    0x1_0000_0000_0000_0000
}

/// `(a + b) mod 2^64`, the value of `a.wrapping_add(b)`.
pub open spec fn add_mod(a: u64, b: u64) -> u64 {
    ((a + b) % word()) as u64
}

/// `(a * b) mod 2^64`, the value of `a.wrapping_mul(b)`.
pub open spec fn mul_mod(a: u64, b: u64) -> u64 {
    ((a * b) % word()) as u64
}

/// Weyl state `seed + gamma * (index + 1) mod 2^64` mixed into draw `index`.
pub open spec fn weyl_state(seed: u64, index: u64) -> u64 {
    add_mod(seed, mul_mod(0x9e37_79b9_7f4a_7c15, add_mod(index, 1)))
}

/// SplitMix64's finalizer: two xorshift-multiply rounds and a final xorshift.
pub open spec fn mix(state: u64) -> u64 {
    let first = mul_mod(state ^ (state >> 30u64), 0xbf58_476d_1ce4_e5b9);
    let second = mul_mod(first ^ (first >> 27u64), 0x94d0_49bb_1331_11eb);
    second ^ (second >> 31u64)
}

/// Draw `index` of the stream selected by `seed`.
pub open spec fn draw(seed: u64, index: u64) -> u64 {
    mix(weyl_state(seed, index))
}

/// Inverse of [`mix`], undoing each round in reverse order.
///
/// `x ^ (x >> s)` is undone by `y ^ (y >> s) ^ (y >> 2s)` when `3s >= 64`,
/// and multiplication by an odd constant by multiplication with its inverse
/// modulo `2^64`.
pub open spec fn unmix(value: u64) -> u64 {
    let second = value ^ (value >> 31u64) ^ (value >> 62u64);
    let first_shifted = mul_mod(second, 0x319642b2d24d8ec3);
    let first = first_shifted ^ (first_shifted >> 27u64) ^ (first_shifted >> 54u64);
    let state_shifted = mul_mod(first, 0x96de1b173f119089);
    state_shifted ^ (state_shifted >> 30u64) ^ (state_shifted >> 60u64)
}

/// Multiplying by `factor` and then by `inverse` is the identity modulo `2^64`.
proof fn lemma_mul_mod_inverse(value: u64, factor: u64, inverse: u64, multiple: int)
    requires
        multiple >= 0,
        factor * inverse == 1 + multiple * word(),
    ensures
        mul_mod(mul_mod(value, factor), inverse) == value,
{
    vstd::arithmetic::div_mod::lemma_mul_mod_noop_left(value * factor, inverse as int, word());
    assert(value * factor * inverse == word() * (value * multiple) + value) by (nonlinear_arith)
        requires
            factor * inverse == 1 + multiple * word(),
    ;
    vstd::arithmetic::div_mod::lemma_mod_multiples_vanish(value * multiple, value as int, word());
    vstd::arithmetic::div_mod::lemma_small_mod(value as nat, word() as nat);
}

/// [`unmix`] inverts [`mix`], so `mix` is a bijection of `u64`.
proof fn lemma_unmix_inverts_mix(state: u64)
    ensures
        unmix(mix(state)) == state,
{
    let first_shifted = state ^ (state >> 30u64);
    assert(first_shifted ^ (first_shifted >> 30u64) ^ (first_shifted >> 60u64) == state)
        by (bit_vector)
        requires
            first_shifted == state ^ (state >> 30u64),
    ;
    let first = mul_mod(first_shifted, 0xbf58_476d_1ce4_e5b9);
    lemma_mul_mod_inverse(first_shifted, 0xbf58_476d_1ce4_e5b9, 0x96de1b173f119089, 8125545580553827436);
    let second_shifted = first ^ (first >> 27u64);
    assert(second_shifted ^ (second_shifted >> 27u64) ^ (second_shifted >> 54u64) == first)
        by (bit_vector)
        requires
            second_shifted == first ^ (first >> 27u64),
    ;
    let second = mul_mod(second_shifted, 0x94d0_49bb_1331_11eb);
    lemma_mul_mod_inverse(second_shifted, 0x94d0_49bb_1331_11eb, 0x319642b2d24d8ec3, 2077064247507883091);
    let value = second ^ (second >> 31u64);
    assert(value ^ (value >> 31u64) ^ (value >> 62u64) == second) by (bit_vector)
        requires
            value == second ^ (second >> 31u64),
    ;
}

/// Adding a fixed `offset` modulo `2^64` is injective.
proof fn lemma_add_mod_injective(offset: u64, first: u64, second: u64)
    requires
        add_mod(offset, first) == add_mod(offset, second),
    ensures
        first == second,
{
}

/// Distinct counters of one stream give distinct draws.
///
/// `index -> draw(seed, index)` is therefore injective, hence a bijection of
/// `u64`: a stream repeats no raw value before all `2^64` counters are used.
pub proof fn lemma_draws_are_distinct(seed: u64, first: u64, second: u64)
    requires
        first != second,
    ensures
        draw(seed, first) != draw(seed, second),
{
    if draw(seed, first) == draw(seed, second) {
        lemma_unmix_inverts_mix(weyl_state(seed, first));
        lemma_unmix_inverts_mix(weyl_state(seed, second));
        let gamma: u64 = 0x9e37_79b9_7f4a_7c15;
        let first_step = add_mod(first, 1);
        let second_step = add_mod(second, 1);
        lemma_add_mod_injective(seed, mul_mod(gamma, first_step), mul_mod(gamma, second_step));
        // gamma is odd; its inverse modulo 2^64 recovers the counter.
        assert(mul_mod(gamma, first_step) == mul_mod(first_step, gamma));
        assert(mul_mod(gamma, second_step) == mul_mod(second_step, gamma));
        lemma_mul_mod_inverse(first_step, gamma, 0xf1de83e19937733d, 10771413168148108919);
        lemma_mul_mod_inverse(second_step, gamma, 0xf1de83e19937733d, 10771413168148108919);
        lemma_add_mod_injective(1, first, second);
    }
}

/// Advancing the counter adds `gamma` to the Weyl state.
///
/// Draw `index + 1` therefore mixes the state of draw `index` plus `gamma`,
/// exactly as the classic generator that adds `gamma` before every output.
pub proof fn lemma_counter_step_adds_gamma(seed: u64, index: u64)
    ensures
        weyl_state(seed, add_mod(index, 1)) == add_mod(weyl_state(seed, index), 0x9e37_79b9_7f4a_7c15),
{
    let gamma: int = 0x9e37_79b9_7f4a_7c15;
    let step = add_mod(index, 1) as int;
    let next = add_mod(add_mod(index, 1), 1) as int;
    // next == step + 1 (mod 2^64), so gamma * next == gamma * step + gamma (mod 2^64).
    assert((gamma * next) % word() == (gamma * step + gamma) % word()) by {
        if step + 1 < word() {
            assert(next == step + 1);
            assert(gamma * next == gamma * step + gamma) by (nonlinear_arith)
                requires
                    next == step + 1,
            ;
        } else {
            assert(next == 0);
            assert(step == word() - 1);
            assert(gamma * step + gamma == word() * gamma) by (nonlinear_arith)
                requires
                    step == word() - 1,
            ;
            vstd::arithmetic::div_mod::lemma_mod_multiples_basic(gamma, word());
        }
    }
    let product = mul_mod(0x9e37_79b9_7f4a_7c15, add_mod(index, 1)) as int;
    vstd::arithmetic::div_mod::lemma_add_mod_noop(gamma * step, gamma, word());
    vstd::arithmetic::div_mod::lemma_small_mod(gamma as nat, word() as nat);
    vstd::arithmetic::div_mod::lemma_add_mod_noop(seed as int, product + gamma, word());
    vstd::arithmetic::div_mod::lemma_add_mod_noop(seed + product, gamma, word());
    vstd::arithmetic::div_mod::lemma_small_mod(seed as nat, word() as nat);
}

} // verus!

/// Returns output `index` of the SplitMix64 stream seeded with `seed`.
///
/// Equals the `(index + 1)`-th call of a sequential SplitMix64 generator
/// whose state starts at `seed`; all arithmetic wraps modulo `2^64`. The
/// proof shows the result is exactly `draw(seed, index)`, which is injective
/// in `index` (`lemma_draws_are_distinct`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(value =>
    ensures
        value == draw(seed, index),
))]
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
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    /// Seed that selects the stream.
    seed: u64,
    /// Index of the next draw.
    position: u64,
}

#[cfg(verus_keep_ghost)]
verus! {

impl SplitMix64 {
    /// Seed of the stream, for specifications.
    pub closed spec fn stream_seed(self) -> u64 {
        self.seed
    }

    /// Index of the next draw, for specifications.
    pub closed spec fn stream_position(self) -> u64 {
        self.position
    }
}

} // verus!

/// Builds the stream selected by `seed`, positioned at its first draw.
///
/// [`SplitMix64::new`] delegates here because Verus cannot attach a contract
/// to an associated function without a `self` receiver. The proof shows the
/// stream's seed is `seed` and its position is zero, so together with the
/// [`SplitMix64::next_u64`] contract the `k`-th draw of `SplitMix64::new(seed)`
/// (counting from zero, `k < 2^64`) is exactly `splitmix64(seed, k)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(stream =>
    ensures
        stream.stream_seed() == seed,
        stream.stream_position() == 0,
))]
const fn stream_start(seed: u64) -> SplitMix64 {
    SplitMix64 { seed, position: 0 }
}

impl SplitMix64 {
    /// Starts the stream selected by `seed` at its first draw.
    ///
    /// The verified [`stream_start`] builds the value: seed `seed`, position
    /// zero.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        stream_start(seed)
    }

    /// Returns the number of draws taken so far, modulo `2^64`.
    ///
    /// The proof shows the value is the stream's position, the counter that
    /// [`Self::next_u64`] reads and advances.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(position =>
        ensures
            position == self.stream_position(),
    ))]
    #[must_use]
    pub const fn position(&self) -> u64 {
        self.position
    }

    /// Returns the next raw 64-bit draw.
    ///
    /// The proof shows the value is `draw(seed, position)`, the seed is
    /// unchanged, and the position advances by exactly one (modulo `2^64`).
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(value =>
        ensures
            value == draw(old(self).stream_seed(), old(self).stream_position()),
            final(self).stream_seed() == old(self).stream_seed(),
            final(self).stream_position() == add_mod(old(self).stream_position(), 1),
    ))]
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
