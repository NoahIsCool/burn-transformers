//! Correctly rounded narrowing of IEEE binary64 values to 16-bit floats.
//!
//! A finite, nonzero binary64 value with sign `s`, biased exponent `e` in
//! `1..=2046`, and fraction `f` is exactly `(-1)^s * m * 2^(e - 1075)` with
//! integer significand `m = 2^52 + f`. A 16-bit target format with `E` exponent
//! and `M` fraction bits stores a normal value `(2^M + t) * 2^(x - b - M)` for
//! biased exponent `x` in `1..2^E - 1` and bias `b = 2^(E - 1) - 1`; exponent
//! field zero holds subnormals `t * 2^(1 - b - M)`.
//!
//! Narrowing therefore keeps the top `M + 1` significand bits of a normal result
//! (shift `52 - M`) or fewer bits for a subnormal one (shift `52 - M + 1 - x`),
//! then applies IEEE 754 round-to-nearest, ties-to-even **once** to the exact
//! remainder: increment the kept quotient when the remainder exceeds half an
//! output unit, or equals it while the quotient is odd. Writing a normal result
//! as `((x - 1) << M) + quotient` lets a rounding carry propagate naturally into
//! the exponent field, including the carry from the largest finite value to
//! infinity and from the largest subnormal to the smallest normal.
//!
//! Binary64 subnormals (`e = 0`) lie below `2^-1022`, which is far less than half
//! of the smallest BF16 or F16 subnormal, so they round to a signed zero. NaNs stay
//! NaN: the result is quiet and keeps the top `M - 1` payload bits, matching
//! hardware conversion and the `half` crate's NaN propagation.
//!
//! This module is deliberately free of crate and Burn dependencies. Unlike the
//! `half` crate's `from_f64`, which discards the low 32 binary64 fraction bits
//! (or double-rounds through `f32` on some targets), every discarded bit
//! participates in the sticky remainder here.
//!
//! # Verification
//!
//! The rounding step itself, [`round_shifted`], is verified in place by Verus
//! through `verus/checkpoint.rs`: it returns the integer nearest to
//! `significand / 2^shift`, rounding ties to even, without overflow. The bit
//! assembly around it (sign, exponent field, NaN payload) is checked by tests at
//! every 16-bit midpoint and against independent oracles.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: `2^exponent`.
pub open spec fn power_of_two(exponent: nat) -> nat {
    vstd::arithmetic::power2::pow2(exponent)
}

/// Specification: the round-to-nearest, ties-to-even quotient of `value / 2^shift`.
pub open spec fn nearest_even_quotient(value: nat, shift: nat) -> nat {
    let unit = power_of_two(shift);
    let half = power_of_two((shift - 1) as nat);
    let quotient = value / unit;
    let remainder = value % unit;
    if remainder > half || (remainder == half && quotient % 2 == 1) {
        quotient + 1
    } else {
        quotient
    }
}

} // verus!

/// Bit layout of a 16-bit binary floating-point storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HalfWidthFormat {
    /// Exponent-field width `E`; the field value `2^E - 1` encodes infinity and NaN.
    exponent_bits: u32,
    /// Stored fraction width `M`, excluding the implicit leading significand bit.
    fraction_bits: u32,
}

/// IEEE 754 binary16: five exponent bits, ten fraction bits, bias 15.
pub(crate) const BINARY16: HalfWidthFormat = HalfWidthFormat {
    exponent_bits: 5,
    fraction_bits: 10,
};

/// Brain floating point: eight exponent bits, seven fraction bits, bias 127.
pub(crate) const BFLOAT16: HalfWidthFormat = HalfWidthFormat {
    exponent_bits: 8,
    fraction_bits: 7,
};

/// Number of stored fraction bits in IEEE binary64.
#[cfg_attr(verus_keep_ghost, verus_verify)]
const F64_FRACTION_BITS: u32 = 52;
/// Exponent bias of IEEE binary64.
const F64_EXPONENT_BIAS: i32 = 1023;
/// Biased binary64 exponent field reserved for infinity and NaN.
const F64_SPECIAL_EXPONENT: u64 = 0x7ff;

/// Rounds one binary64 bit pattern to a 16-bit format with ties to even.
///
/// The result is a raw bit pattern in `format`; for example
/// `round_f64_bits(0x3FF0_1000_0010_0000, BFLOAT16)`, the binary64 value
/// `1 + 2^-8 + 2^-40`, lies just above a BF16 midpoint and returns `0x3F81`.
pub(crate) fn round_f64_bits(bits: u64, format: HalfWidthFormat) -> u16 {
    let fraction_bits = format.fraction_bits;
    let sign = ((bits >> 63) as u16) << 15;
    let biased_exponent = (bits >> F64_FRACTION_BITS) & F64_SPECIAL_EXPONENT;
    let fraction = bits & ((1_u64 << F64_FRACTION_BITS) - 1);
    let special_exponent = (1_u32 << format.exponent_bits) - 1;
    let infinity = (special_exponent << fraction_bits) as u16;

    if biased_exponent == F64_SPECIAL_EXPONENT {
        if fraction == 0 {
            return sign | infinity;
        }
        // Quiet the NaN and keep its most significant payload bits.
        let quiet_bit = 1_u16 << (fraction_bits - 1);
        let payload = (fraction >> (F64_FRACTION_BITS - fraction_bits)) as u16;
        return sign | infinity | quiet_bit | payload;
    }
    if biased_exponent == 0 {
        // Zero or a binary64 subnormal: below half of every target subnormal.
        return sign;
    }

    let target_bias = (1_i32 << (format.exponent_bits - 1)) - 1;
    let target_exponent = biased_exponent as i32 - F64_EXPONENT_BIAS + target_bias;
    if target_exponent >= special_exponent as i32 {
        return sign | infinity;
    }

    let significand = fraction | (1_u64 << F64_FRACTION_BITS);
    let normal_shift = F64_FRACTION_BITS - fraction_bits;
    let shift = if target_exponent >= 1 {
        normal_shift
    } else {
        // Each exponent step below the normal range discards one more bit.
        normal_shift + (1 - target_exponent) as u32
    };
    if shift > F64_FRACTION_BITS + 1 {
        // `significand < 2^53 <= 2^(shift - 1)`: strictly below half the
        // smallest subnormal, so the nearest value is a signed zero.
        return sign;
    }

    let rounded = round_shifted(significand, shift);
    let magnitude = if target_exponent >= 1 {
        // `rounded` includes the implicit bit, so a carry to `2^(M + 1)` moves
        // into the exponent field (and reaches exactly `infinity` at the top).
        (((target_exponent - 1) as u64) << fraction_bits) + rounded
    } else {
        rounded
    };
    sign | magnitude as u16
}

/// Rounds `significand / 2^shift` to the nearest integer, ties to even.
///
/// The kept quotient is incremented when the discarded remainder exceeds half
/// an output unit `2^(shift - 1)`, or equals it while the quotient is odd.
///
/// Verified: the result is exactly [`nearest_even_quotient`]; it lies within
/// half a unit of the exact quotient (`|r · 2^shift - significand| <= 2^(shift
/// - 1)`), an exact tie rounds to an even result, and the result never exceeds
/// `2^(53 - shift)`, so the caller's exponent arithmetic cannot overflow.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rounded =>
    requires
        1 <= shift <= F64_FRACTION_BITS + 1,
        significand < 0x20_0000_0000_0000,
    ensures
        rounded as nat == nearest_even_quotient(significand as nat, shift as nat),
        significand as nat <= rounded as nat * power_of_two(shift as nat)
            + power_of_two((shift - 1) as nat),
        rounded as nat * power_of_two(shift as nat)
            <= significand as nat + power_of_two((shift - 1) as nat),
        rounded as nat * power_of_two(shift as nat) + power_of_two((shift - 1) as nat)
            == significand as nat
            || rounded as nat * power_of_two(shift as nat)
                == significand as nat + power_of_two((shift - 1) as nat)
            ==> rounded % 2 == 0,
        rounded as nat <= power_of_two((53 - shift) as nat),
))]
fn round_shifted(significand: u64, shift: u32) -> u64 {
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::bits::lemma_u64_shr_is_div(significand, shift as u64);
        vstd::bits::lemma_u64_pow2_no_overflow(shift as nat);
        vstd::bits::lemma_u64_pow2_no_overflow((shift - 1) as nat);
        vstd::bits::lemma_u64_shl_is_mul(1, shift as u64);
        vstd::bits::lemma_u64_shl_is_mul(1, (shift - 1) as u64);
        vstd::bits::lemma_u64_low_bits_mask_is_mod(significand, shift as nat);
        vstd::bits::lemma_low_bits_mask_values();
    }
    let quotient = significand >> shift;
    let remainder = significand & ((1_u64 << shift) - 1);
    let halfway = 1_u64 << (shift - 1);
    #[cfg(verus_keep_ghost)]
    proof! {
        let unit = power_of_two(shift as nat);
        let half = power_of_two((shift - 1) as nat);
        assert(quotient as nat == significand as nat / unit);
        assert(halfway as nat == half);
        assert((1_u64 << shift) as nat == unit);
        assert(((1_u64 << shift) - 1) as nat == vstd::bits::low_bits_mask(shift as nat));
        assert(remainder as nat == significand as nat % unit);
        assert(quotient & 1 == quotient % 2) by (bit_vector);
        lemma_round_shifted_bounds(significand as nat, shift as nat);
    }
    let round_up = remainder > halfway || (remainder == halfway && quotient & 1 == 1);
    quotient + if round_up { 1 } else { 0 }
}

#[cfg(verus_keep_ghost)]
verus! {

/// The nearest-even quotient is within half a unit, even on ties, and at most `2^(53 - shift)`.
proof fn lemma_round_shifted_bounds(value: nat, shift: nat)
    requires
        1 <= shift <= 53,
        value < 0x20_0000_0000_0000,
    ensures
        ({
            let unit = power_of_two(shift);
            let half = power_of_two((shift - 1) as nat);
            let rounded = nearest_even_quotient(value, shift);
            &&& unit == 2 * half
            &&& half >= 1
            &&& value / unit < 0x20_0000_0000_0000
            &&& value <= rounded * unit + half
            &&& rounded * unit <= value + half
            &&& (rounded * unit + half == value || rounded * unit == value + half)
                ==> rounded % 2 == 0
            &&& rounded <= power_of_two((53 - shift) as nat)
        }),
{
    let unit = power_of_two(shift);
    let half = power_of_two((shift - 1) as nat);
    vstd::arithmetic::power2::lemma_pow2_unfold(shift);
    vstd::arithmetic::power2::lemma_pow2_pos((shift - 1) as nat);
    vstd::arithmetic::power2::lemma_pow2_adds(shift, (53 - shift) as nat);
    vstd::arithmetic::power2::lemma2_to64();
    vstd::arithmetic::power2::lemma2_to64_rest();
    assert(unit == 2 * half);
    let quotient = value / unit;
    let remainder = value % unit;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(value as int, unit as int);
    assert(value == unit * quotient + remainder);
    assert(remainder < unit);
    let top = power_of_two((53 - shift) as nat);
    assert(unit * top == 0x20_0000_0000_0000);
    // value < unit * top, so quotient < top.
    assert(quotient < top) by (nonlinear_arith)
        requires
            value == unit * quotient + remainder,
            remainder >= 0,
            value < unit * top,
            unit >= 1,
    ;
    assert(quotient < 0x20_0000_0000_0000) by (nonlinear_arith)
        requires quotient < top, unit * top == 0x20_0000_0000_0000, unit >= 1;
    let rounded = nearest_even_quotient(value, shift);
    if remainder > half || (remainder == half && quotient % 2 == 1) {
        assert(rounded == quotient + 1);
        assert(rounded * unit == quotient * unit + unit) by (nonlinear_arith)
            requires rounded == quotient + 1;
        assert(quotient * unit == unit * quotient) by (nonlinear_arith);
    } else {
        assert(rounded == quotient);
        assert(quotient * unit == unit * quotient) by (nonlinear_arith);
    }
}

} // verus!

#[cfg(test)]
#[path = "rounding_tests.rs"]
mod tests;
