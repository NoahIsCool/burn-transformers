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

    let quotient = significand >> shift;
    let remainder = significand & ((1_u64 << shift) - 1);
    let halfway = 1_u64 << (shift - 1);
    let round_up = remainder > halfway || (remainder == halfway && quotient & 1 == 1);
    let rounded = quotient + u64::from(round_up);
    let magnitude = if target_exponent >= 1 {
        // `rounded` includes the implicit bit, so a carry to `2^(M + 1)` moves
        // into the exponent field (and reaches exactly `infinity` at the top).
        (((target_exponent - 1) as u64) << fraction_bits) + rounded
    } else {
        rounded
    };
    sign | magnitude as u16
}

#[cfg(test)]
#[path = "rounding_tests.rs"]
mod tests;
