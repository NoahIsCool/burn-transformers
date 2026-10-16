//! Tests for Python-compatible integer rounding and decimal increments.

use proptest::prelude::*;

use super::*;

/// Straightforward reference: Python's round-half-even on a wider type.
fn reference_round(value: i128, places: u32) -> Option<i128> {
    let Some(unit) = 10_i128.checked_pow(places) else {
        return Some(0);
    };
    let quotient = value.div_euclid(unit);
    let remainder = value.rem_euclid(unit);
    // `remainder < unit <= 10^38`, so twice it fits `u128`.
    let up = match (2 * remainder as u128).cmp(&(unit as u128)) {
        core::cmp::Ordering::Less => false,
        core::cmp::Ordering::Greater => true,
        core::cmp::Ordering::Equal => quotient % 2 != 0,
    };
    let quotient = if up { quotient + 1 } else { quotient };
    quotient.checked_mul(unit)
}

/// Powers of ten are exact until they leave `i128`.
#[test]
fn powers_of_ten_match_checked_pow() {
    for places in 0..=45 {
        assert_eq!(
            checked_power_of_ten(places),
            10_i128.checked_pow(places),
            "{places}"
        );
    }
    assert_eq!(checked_power_of_ten(u32::MAX), None);
}

/// Python's `round(n, -k)` examples, ties to even.
#[test]
fn integers_round_half_to_even_like_python() {
    assert_eq!(round_integer_half_even(25, 1), Some(20));
    assert_eq!(round_integer_half_even(35, 1), Some(40));
    assert_eq!(round_integer_half_even(-25, 1), Some(-20));
    assert_eq!(round_integer_half_even(-35, 1), Some(-40));
    assert_eq!(round_integer_half_even(1234, 2), Some(1200));
    assert_eq!(round_integer_half_even(1250, 2), Some(1200));
    assert_eq!(round_integer_half_even(1350, 2), Some(1400));
    assert_eq!(round_integer_half_even(7, 0), Some(7));
    assert_eq!(round_integer_half_even(i128::MAX, 50), Some(0));
    assert_eq!(round_integer_half_even(i128::MIN, 39), Some(0));
    assert_eq!(round_integer_half_even(i128::MAX, 1), None);
}

/// Review finding: the lower multiple of a value near `i128::MIN` may not fit.
///
/// `value - remainder` was computed before choosing a direction, so
/// `i128::MIN` rounded to tens overflowed (a debug panic; in release the
/// wrapped value came back as a large positive number), and rounding to
/// hundreds, whose nearest multiple fits, was reported as an overflow.
#[test]
fn values_near_i128_min_round_without_overflow() {
    assert_eq!(round_integer_half_even(i128::MIN, 1), None);
    assert_eq!(round_integer_half_even(i128::MIN, 2), Some(i128::MIN + 28));
    // `i128::MIN` ends in ...728: MIN + 8 ends in ...720, a multiple of ten.
    assert_eq!(
        round_integer_half_even(i128::MIN + 8, 1),
        Some(i128::MIN + 8)
    );
    assert_eq!(
        round_integer_half_even(i128::MIN + 3, 1),
        Some(i128::MIN + 8)
    );
    assert_eq!(round_integer_half_even(i128::MIN + 1, 1), None);
}

/// Decimal increments carry through nines and grow only when every digit is nine.
#[test]
fn decimal_increments_carry() {
    for (digits, expected) in [
        ("0", "1"),
        ("8", "9"),
        ("9", "10"),
        ("129", "130"),
        ("1999", "2000"),
        ("999", "1000"),
        ("", "1"),
    ] {
        let mut bytes = digits.as_bytes().to_vec();
        increment_decimal_digits(&mut bytes);
        assert_eq!(bytes, expected.as_bytes(), "{digits}");
    }
}

/// Rounds `digits` to tens and returns the kept quotient as text.
fn rounded(digits: &str, places: usize, fractional: bool) -> String {
    String::from_utf8(round_digits_to_tens(digits.as_bytes(), places, fractional)).unwrap()
}

/// Digit strings round half to even, with a fraction breaking ties upward.
#[test]
fn digit_strings_round_half_to_even() {
    assert_eq!(rounded("1234", 2, false), "12");
    assert_eq!(rounded("1250", 2, false), "12");
    assert_eq!(rounded("1250", 2, true), "13");
    assert_eq!(rounded("1350", 2, false), "14");
    assert_eq!(rounded("1251", 2, false), "13");
    assert_eq!(rounded("9950", 2, false), "100");
    assert_eq!(rounded("5", 1, false), "0");
    assert_eq!(rounded("5", 1, true), "1");
    assert_eq!(rounded("15", 1, false), "2");
    assert_eq!(rounded("49", 3, true), "0");
    assert_eq!(rounded("0", 4, false), "0");
    assert_eq!(rounded("999", 1, false), "100");
}

proptest! {
    /// Digit rounding agrees with integer arithmetic.
    #[test]
    fn digit_rounding_matches_integers(value in any::<u64>(), places in 1_usize..22, fractional in any::<bool>()) {
        let unit = 10_u128.pow(u32::try_from(places).unwrap());
        let (quotient, remainder) = (u128::from(value) / unit, u128::from(value) % unit);
        let up = 2 * remainder > unit || (2 * remainder == unit && (fractional || quotient % 2 == 1));
        let expected = (quotient + u128::from(up)).to_string();
        prop_assert_eq!(rounded(&value.to_string(), places, fractional), expected);
    }

    /// Integer rounding agrees with the reference on the whole `i128` range.
    #[test]
    fn integer_rounding_matches_reference(value in any::<i128>(), places in 0_u32..42) {
        prop_assert_eq!(round_integer_half_even(value, places), reference_round(value, places));
    }

    /// Integer rounding agrees with the reference near both ends of `i128`.
    #[test]
    fn integer_rounding_matches_reference_at_the_edges(offset in 0_i128..1000, places in 0_u32..42) {
        prop_assert_eq!(
            round_integer_half_even(i128::MIN + offset, places),
            reference_round(i128::MIN + offset, places)
        );
        prop_assert_eq!(
            round_integer_half_even(i128::MAX - offset, places),
            reference_round(i128::MAX - offset, places)
        );
    }

    /// Incrementing digits adds one to the number they spell.
    #[test]
    fn decimal_increment_adds_one(value in 0_u64..u64::MAX) {
        let mut bytes = value.to_string().into_bytes();
        increment_decimal_digits(&mut bytes);
        prop_assert_eq!(String::from_utf8(bytes).unwrap(), (u128::from(value) + 1).to_string());
    }
}
