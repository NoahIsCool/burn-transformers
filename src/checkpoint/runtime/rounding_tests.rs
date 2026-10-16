//! Checks correctly rounded binary64 narrowing against exact integer references.
//!
//! The references are independent of the implementation: hand-derived tie cases,
//! exhaustive midpoints between every pair of adjacent 16-bit values (computed
//! exactly in binary64), a round-to-odd composition through `f32`, and, where
//! available, the host's hardware binary64 -> binary16 conversion.

use burn::tensor::{bf16, f16};
use proptest::prelude::*;

use super::*;

/// Decodes a 16-bit pattern of `format` exactly into binary64.
fn decode(bits: u16, format: HalfWidthFormat) -> f64 {
    if format == BINARY16 {
        f64::from(f16::from_bits(bits).to_f32())
    } else {
        f64::from(bf16::from_bits(bits).to_f32())
    }
}

/// Rounds an `f64` through the implementation under test.
fn round(value: f64, format: HalfWidthFormat) -> u16 {
    round_f64_bits(value.to_bits(), format)
}

/// Returns the next representable binary64 value away from zero for positive values.
fn next_up(value: f64) -> f64 {
    f64::from_bits(value.to_bits() + 1)
}

/// Returns the next representable binary64 value toward zero for positive values.
fn next_down(value: f64) -> f64 {
    f64::from_bits(value.to_bits() - 1)
}

/// Independent reference: round to odd in `f32`, then round once to nearest even.
///
/// Rounding to odd with at least two extra bits of precision makes the second
/// rounding exact (Boldo and Melquiond); `f32` keeps 24 bits for targets of at
/// most 11, including every BF16 subnormal scale above `2^-149`.
fn reference(value: f64, format: HalfWidthFormat) -> u16 {
    let nearest = value as f32;
    let truncated = if f64::from(nearest).abs() > value.abs() {
        f32::from_bits(nearest.to_bits() - 1)
    } else {
        nearest
    };
    let odd = if f64::from(truncated) == value || truncated.is_infinite() {
        truncated
    } else {
        f32::from_bits(truncated.to_bits() | 1)
    };
    if format == BINARY16 {
        f16::from_f32(odd).to_bits()
    } else {
        bf16::from_f32(odd).to_bits()
    }
}

/// Values just above a tie round up even though `half` truncates their sticky bits.
#[test]
fn sticky_bits_below_the_upper_word_break_ties() {
    // 1 + 2^-8 + 2^-40: above the BF16 midpoint between 0x3F80 and 0x3F81.
    let value = 1.0 + 2f64.powi(-8) + 2f64.powi(-40);
    assert_eq!(round(value, BFLOAT16), 0x3F81);
    // The F32 value 0x3F808001 = 1 + 2^-8 + 2^-23.
    let value = f64::from(f32::from_bits(0x3F80_8001));
    assert_eq!(round(value, BFLOAT16), 0x3F81);
    // 1 + 2^-11 + 2^-23 lies above the F16 midpoint between 0x3C00 and 0x3C01.
    let value = 1.0 + 2f64.powi(-11) + 2f64.powi(-23);
    assert_eq!(round(value, BINARY16), 0x3C01);
    let value = 1.0 + 2f64.powi(-11) + 2f64.powi(-52);
    assert_eq!(round(value, BINARY16), 0x3C01);
}

/// Exact ties go to the even neighbor in both directions.
#[test]
fn exact_ties_round_to_even() {
    assert_eq!(round(1.0 + 2f64.powi(-8), BFLOAT16), 0x3F80);
    assert_eq!(round(1.0 + 3.0 * 2f64.powi(-8), BFLOAT16), 0x3F82);
    assert_eq!(round(1.0 + 2f64.powi(-11), BINARY16), 0x3C00);
    assert_eq!(round(1.0 + 3.0 * 2f64.powi(-11), BINARY16), 0x3C02);
    assert_eq!(round(-(1.0 + 3.0 * 2f64.powi(-11)), BINARY16), 0xBC02);
}

/// Overflow, subnormal, signed-zero, and binary64-subnormal boundaries.
#[test]
fn range_boundaries_round_like_ieee_754() {
    // F16 max is 65504; the midpoint to the next binade, 65520, ties to infinity.
    assert_eq!(round(65504.0, BINARY16), 0x7BFF);
    assert_eq!(round(next_down(65520.0), BINARY16), 0x7BFF);
    assert_eq!(round(65520.0, BINARY16), 0x7C00);
    assert_eq!(round(-1.0e300, BINARY16), 0xFC00);
    // BF16 max is 0x7F7F; the midpoint above it ties to infinity.
    let bf16_max = decode(0x7F7F, BFLOAT16);
    let bf16_overflow_tie = bf16_max + 2f64.powi(127 - 8);
    assert_eq!(round(bf16_max, BFLOAT16), 0x7F7F);
    assert_eq!(round(next_down(bf16_overflow_tie), BFLOAT16), 0x7F7F);
    assert_eq!(round(bf16_overflow_tie, BFLOAT16), 0x7F80);
    // Half the smallest F16 subnormal ties to zero; anything above rounds up.
    let half_min_subnormal = 2f64.powi(-25);
    assert_eq!(round(half_min_subnormal, BINARY16), 0x0000);
    assert_eq!(round(next_up(half_min_subnormal), BINARY16), 0x0001);
    assert_eq!(round(-next_up(half_min_subnormal), BINARY16), 0x8001);
    assert_eq!(round(1.5 * 2f64.powi(-24), BINARY16), 0x0002);
    // The tie above the largest subnormal carries into the smallest normal.
    let largest_subnormal = decode(0x03FF, BINARY16);
    let carry_tie = largest_subnormal + 2f64.powi(-25);
    assert_eq!(round(carry_tie, BINARY16), 0x0400);
    assert_eq!(round(2f64.powi(-133), BFLOAT16), 0x0001);
    assert_eq!(round(2f64.powi(-134), BFLOAT16), 0x0000);
    // Signed zeros and binary64 subnormals.
    assert_eq!(round(0.0, BINARY16), 0x0000);
    assert_eq!(round(-0.0, BFLOAT16), 0x8000);
    assert_eq!(round(f64::from_bits(1), BFLOAT16), 0x0000);
    assert_eq!(
        round(-f64::from_bits(0x000F_FFFF_FFFF_FFFF), BINARY16),
        0x8000
    );
    assert_eq!(round(f64::MIN_POSITIVE, BFLOAT16), 0x0000);
}

/// Infinities keep their sign and NaNs stay quiet NaNs with their top payload bits.
#[test]
fn infinities_and_nans_are_preserved() {
    assert_eq!(round(f64::INFINITY, BINARY16), 0x7C00);
    assert_eq!(round(f64::NEG_INFINITY, BFLOAT16), 0xFF80);
    assert_eq!(round_f64_bits(0x7FF8_0000_0000_0000, BINARY16), 0x7E00);
    assert_eq!(round_f64_bits(0x7FF8_0000_0000_0000, BFLOAT16), 0x7FC0);
    // A signaling NaN whose payload lies only in the low bits becomes quiet.
    assert_eq!(round_f64_bits(0xFFF0_0000_0000_0001, BINARY16), 0xFE00);
    assert_eq!(round_f64_bits(0x7FF0_0000_0000_0001, BFLOAT16), 0x7FC0);
    // High payload bits survive.
    assert_eq!(round_f64_bits(0x7FFF_C000_0000_0000, BINARY16), 0x7FF0);
    assert_eq!(round_f64_bits(0x7FF7_E000_0000_0000, BFLOAT16), 0x7FFF);
}

/// Every finite 16-bit value round-trips, and every adjacent midpoint rounds correctly.
#[test]
fn exhaustive_midpoints_and_round_trips_in_both_formats() {
    for format in [BINARY16, BFLOAT16] {
        let infinity = if format == BINARY16 { 0x7C00 } else { 0x7F80 };
        for magnitude in 0..infinity {
            let lower = decode(magnitude, format);
            assert_eq!(
                round(lower, format),
                magnitude,
                "{format:?} {magnitude:#06x}"
            );
            assert_eq!(
                round(-lower, format),
                magnitude | 0x8000,
                "{format:?} -{magnitude:#06x}"
            );
            let upper_bits = magnitude + 1;
            let upper = if upper_bits == infinity {
                // The overflow threshold continues the top binade's spacing.
                lower + (lower - decode(magnitude - 1, format))
            } else {
                decode(upper_bits, format)
            };
            let midpoint = (lower + upper) / 2.0;
            let even = if magnitude % 2 == 0 {
                magnitude
            } else {
                upper_bits
            };
            assert_eq!(
                round(midpoint, format),
                even,
                "{format:?} tie {magnitude:#06x}"
            );
            assert_eq!(round(next_up(midpoint), format), upper_bits);
            if midpoint > 0.0 && next_down(midpoint) > lower {
                assert_eq!(round(next_down(midpoint), format), magnitude);
            }
        }
    }
}

proptest! {
    /// Arbitrary binary64 bit patterns agree with the round-to-odd reference.
    #[test]
    fn arbitrary_values_match_round_to_odd_reference(bits in any::<u64>()) {
        let value = f64::from_bits(bits);
        for format in [BINARY16, BFLOAT16] {
            let actual = round_f64_bits(bits, format);
            if value.is_nan() {
                let exponent_mask = if format == BINARY16 { 0x7C00 } else { 0x7F80 };
                prop_assert_eq!(actual & exponent_mask, exponent_mask);
                prop_assert_ne!(actual & !(exponent_mask | 0x8000), 0);
            } else {
                prop_assert_eq!(actual, reference(value, format), "{:?} {:#018x}", format, bits);
            }
        }
    }

    /// Values near 16-bit magnitudes, where ties live, agree with the reference.
    #[test]
    fn values_near_representable_points_match_reference(
        base in any::<u16>(),
        offset in -4096_i64..4096,
        exponent_shift in -3_i32..3,
    ) {
        for format in [BINARY16, BFLOAT16] {
            let center = decode(base, format);
            if !center.is_finite() {
                continue;
            }
            let scaled = center * 2f64.powi(exponent_shift);
            let bits = scaled.to_bits().wrapping_add_signed(offset);
            let value = f64::from_bits(bits);
            if value.is_nan() {
                continue;
            }
            prop_assert_eq!(round_f64_bits(bits, format), reference(value, format));
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
mod hardware {
    //! On hosts with native binary16 conversion, the hardware is a second oracle.
    //!
    //! AArch64 with the `fp16` feature converts binary64 to binary16 with one
    //! correctly rounded `fcvt` (round to nearest, ties to even, with overflow to
    //! infinity and gradual underflow), so it checks the software rounding for
    //! every binary64 bit pattern independently of the reference decoder.

    use super::*;

    proptest! {
        /// `half` uses a single correctly rounded `fcvt` instruction on this target.
        #[test]
        fn binary16_matches_hardware_conversion(bits in any::<u64>()) {
            let value = f64::from_bits(bits);
            prop_assert_eq!(round_f64_bits(bits, BINARY16), f16::from_f64(value).to_bits());
        }
    }
}

/// The shared rounding step is nearest with ties to even at every small shift.
#[test]
fn round_shifted_is_nearest_with_ties_to_even() {
    for shift in 1..=6_u32 {
        let unit = 1_u64 << shift;
        for significand in 0..(8 * unit) {
            let rounded = round_shifted(significand, shift);
            let scaled = rounded * unit;
            let distance = scaled.abs_diff(significand);
            assert!(
                2 * distance <= unit,
                "{significand} >> {shift} -> {rounded}"
            );
            if 2 * distance == unit {
                assert_eq!(rounded % 2, 0, "tie {significand} >> {shift}");
            }
        }
    }
    assert_eq!(round_shifted((1 << 53) - 1, 53), 1);
    assert_eq!(round_shifted(1 << 52, 53), 0);
    assert_eq!(round_shifted((1 << 52) + 1, 53), 1);
}
