//! Dependency-free decimal integer arithmetic behind Python's `round`.
//!
//! Jinja2's `round` filter is Python's built-in `round`: an integer rounded to
//! a negative number of places becomes the nearest multiple of `10^places`,
//! ties going to the even multiple (`round(25, -1) == 20`, `round(35, -1) ==
//! 40`). Rounding a float to tens works on its exact decimal expansion and
//! needs a decimal increment when it rounds away from zero. Both are pure
//! integer logic and live here, free of MiniJinja values and floats.
//!
//! # Verified behavior
//!
//! `verus/tokenization.rs` checks these functions in place. For every input
//! the proofs establish, beyond the absence of overflow or panics:
//!
//! - [`checked_power_of_ten`] is exactly `10^places`, or `None` exactly when
//!   that exceeds `i128::MAX` (the result of `i128::checked_pow`);
//! - [`round_integer_half_even`] returns the multiple of `10^places` nearest
//!   to the value, ties to the multiple with an even quotient, or `None`
//!   exactly when that multiple lies outside `i128`;
//! - [`increment_decimal_digits`] adds one to the number its ASCII digits
//!   spell, keeping them ASCII digits and growing by one digit only when
//!   every digit was `9`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// `10^exponent`.
pub(crate) open spec fn decimal_unit(exponent: nat) -> int
    decreases exponent,
{
    if exponent == 0 {
        1
    } else {
        10 * decimal_unit((exponent - 1) as nat)
    }
}

/// Powers of ten are positive and never decrease with the exponent.
pub(crate) proof fn lemma_decimal_unit_grows(smaller: nat, larger: nat)
    requires
        smaller <= larger,
    ensures
        1 <= decimal_unit(smaller) <= decimal_unit(larger),
    decreases larger - smaller,
{
    if smaller < larger {
        lemma_decimal_unit_grows(smaller, (larger - 1) as nat);
    } else {
        lemma_decimal_unit_positive(smaller);
    }
}

/// Powers of ten are positive.
pub(crate) proof fn lemma_decimal_unit_positive(exponent: nat)
    ensures
        decimal_unit(exponent) >= 1,
    decreases exponent,
{
    if exponent > 0 {
        lemma_decimal_unit_positive((exponent - 1) as nat);
    }
}

/// The multiple of `unit` nearest to `value`, ties to the multiple with an even quotient.
///
/// With `q = floor(value / unit)` and `r = value mod unit`, the candidates
/// are `q * unit` and `(q + 1) * unit`; `2r` against `unit` picks the nearer.
pub(crate) open spec fn half_even_multiple(value: int, unit: int) -> int {
    let quotient = value / unit;
    let remainder = value % unit;
    if 2 * remainder < unit {
        quotient * unit
    } else if 2 * remainder > unit {
        (quotient + 1) * unit
    } else if quotient % 2 == 0 {
        quotient * unit
    } else {
        (quotient + 1) * unit
    }
}

/// [`half_even_multiple`] is a multiple of `unit` at most half a unit away,
/// exactly half a unit away only with an even quotient.
pub(crate) proof fn lemma_half_even_multiple_is_nearest(value: int, unit: int)
    requires
        unit >= 1,
    ensures
        half_even_multiple(value, unit) % unit == 0,
        2 * (value - half_even_multiple(value, unit)) <= unit,
        2 * (half_even_multiple(value, unit) - value) <= unit,
        2 * (value - half_even_multiple(value, unit)) == unit
            || 2 * (half_even_multiple(value, unit) - value) == unit
            ==> (half_even_multiple(value, unit) / unit) % 2 == 0,
{
    let quotient = value / unit;
    let remainder = value % unit;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(value, unit);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(quotient * unit, unit, quotient, 0);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
        (quotient + 1) * unit,
        unit,
        quotient + 1,
        0,
    );
    assert((quotient + 1) * unit == quotient * unit + unit) by (nonlinear_arith);
}

/// The value of an ASCII decimal digit string, most significant digit first.
pub(crate) open spec fn decimal_value(digits: Seq<u8>) -> int
    decreases digits.len(),
{
    if digits.len() == 0 {
        0
    } else {
        decimal_value(digits.drop_last()) * 10 + (digits.last() - 48)
    }
}

/// Every byte is an ASCII decimal digit.
pub(crate) open spec fn all_ascii_digits(digits: Seq<u8>) -> bool {
    forall|i: int| 0 <= i < digits.len() ==> 48 <= #[trigger] digits[i] <= 57
}

/// Every byte is the digit `digit`.
pub(crate) open spec fn all_equal(digits: Seq<u8>, digit: u8) -> bool {
    forall|i: int| 0 <= i < digits.len() ==> #[trigger] digits[i] == digit
}

/// Appending digits scales the prefix: `value(a ++ b) == value(a) * 10^|b| + value(b)`.
pub(crate) proof fn lemma_decimal_value_concat(prefix: Seq<u8>, suffix: Seq<u8>)
    ensures
        decimal_value(prefix + suffix) == decimal_value(prefix) * decimal_unit(suffix.len())
            + decimal_value(suffix),
    decreases suffix.len(),
{
    if suffix.len() == 0 {
        assert(prefix + suffix =~= prefix);
    } else {
        let shorter = suffix.drop_last();
        lemma_decimal_value_concat(prefix, shorter);
        assert((prefix + suffix).drop_last() =~= prefix + shorter);
        assert((prefix + suffix).last() == suffix.last());
        let scaled = decimal_value(prefix) * decimal_unit(shorter.len());
        assert(decimal_value(prefix) * decimal_unit(suffix.len()) == scaled * 10) by (nonlinear_arith)
            requires
                decimal_unit(suffix.len()) == 10 * decimal_unit(shorter.len()),
                scaled == decimal_value(prefix) * decimal_unit(shorter.len());
    }
}

/// A run of `0` digits spells zero; a run of `9` digits spells `10^k - 1`.
pub(crate) proof fn lemma_decimal_value_runs(digits: Seq<u8>)
    ensures
        all_equal(digits, 48) ==> decimal_value(digits) == 0,
        all_equal(digits, 57) ==> decimal_value(digits) == decimal_unit(digits.len()) - 1,
    decreases digits.len(),
{
    if digits.len() > 0 {
        let shorter = digits.drop_last();
        lemma_decimal_value_runs(shorter);
        if all_equal(digits, 48) {
            assert(all_equal(shorter, 48));
        }
        if all_equal(digits, 57) {
            assert(all_equal(shorter, 57));
        }
    }
}

} // verus!

/// `10^places` as an `i128`, or `None` when it exceeds `i128::MAX`.
///
/// This is `10_i128.checked_pow(places)`, computed by repeated checked
/// multiplication; it stops at the first overflow, after at most 39 steps.
/// Verified: the result is exactly `10^places` or `None` when that does not
/// fit.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(unit =>
    ensures
        decimal_unit(places as nat) <= i128::MAX ==> unit == Some(decimal_unit(places as nat) as i128),
        decimal_unit(places as nat) > i128::MAX ==> unit.is_none(),
))]
pub(crate) fn checked_power_of_ten(places: u32) -> Option<i128> {
    let mut unit: i128 = 1;
    let mut exponent: u32 = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            exponent <= places,
            unit == decimal_unit(exponent as nat),
        decreases places - exponent,
    ))]
    while exponent < places {
        let Some(next) = unit.checked_mul(10) else {
            #[cfg(verus_keep_ghost)]
            proof! {
                lemma_decimal_unit_grows((exponent + 1) as nat, places as nat);
            }
            return None;
        };
        unit = next;
        exponent += 1;
    }
    Some(unit)
}

/// Rounds an integer to a multiple of `10^places`, ties to the even multiple.
///
/// Returns `None` only when the rounded value overflows `i128`. A unit larger
/// than any `i128` rounds every value to zero. The result is reached from
/// `value` by one checked step toward the chosen multiple, so values near
/// `i128::MIN` whose lower multiple does not fit still round up correctly.
///
/// Verified: the result is the nearest multiple of `10^places`, ties to the
/// multiple with an even quotient ([`half_even_multiple`]), or `None` exactly
/// when that multiple lies outside `i128`. A returned value is a multiple of
/// the unit at most half a unit from `value`, and exactly half a unit only
/// when its quotient is even.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rounded =>
    ensures
        ({
            let nearest = half_even_multiple(value as int, decimal_unit(places as nat));
            &&& i128::MIN <= nearest <= i128::MAX ==> rounded == Some(nearest as i128)
            &&& (nearest < i128::MIN || nearest > i128::MAX) ==> rounded.is_none()
        }),
        rounded matches Some(result) ==> ({
            let unit = decimal_unit(places as nat);
            &&& result as int % unit == 0
            &&& 2 * (value - result) <= unit
            &&& 2 * (result - value) <= unit
            &&& (2 * (value - result) == unit || 2 * (result - value) == unit)
                ==> (result as int / unit) % 2 == 0
        }),
))]
pub(crate) fn round_integer_half_even(value: i128, places: u32) -> Option<i128> {
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_decimal_unit_positive(places as nat);
        lemma_half_even_multiple_is_nearest(value as int, decimal_unit(places as nat));
    }
    let Some(unit) = checked_power_of_ten(places) else {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_rounds_to_zero_beyond_i128(value as int, places as nat);
        }
        return Some(0);
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(value as int, unit as int);
    }
    // `unit >= 1`, so the Euclidean quotient and remainder always exist.
    let (Some(quotient), Some(remainder)) = (
        value.checked_div_euclid(unit),
        value.checked_rem_euclid(unit),
    ) else {
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(false);
        }
        return None;
    };
    let round_up = match remainder.cmp(&(unit - remainder)) {
        core::cmp::Ordering::Less => false,
        core::cmp::Ordering::Greater => true,
        core::cmp::Ordering::Equal => quotient % 2 != 0,
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_quotient_fits(value as int, unit as int);
        assert(quotient as int == value as int / unit as int);
        assert(remainder as int == value as int % unit as int);
        assert(quotient * unit + unit == (quotient + 1) * unit) by (nonlinear_arith);
        assert((quotient as int % 2 != 0) == rust_rem_is_odd(quotient as int));
    }
    if round_up {
        value.checked_add(unit - remainder)
    } else {
        value.checked_sub(remainder)
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// Rust's `q % 2 != 0` and Euclidean `q mod 2 != 0` agree: both say `q` is odd.
pub(crate) open spec fn rust_rem_is_odd(quotient: int) -> bool {
    vstd::arithmetic::div_mod::rust_rem(quotient, 2) != 0
}

/// The Euclidean quotient by a positive unit lies between zero and the value.
pub(crate) proof fn lemma_quotient_fits(value: int, unit: int)
    requires
        unit >= 1,
    ensures
        value >= 0 ==> 0 <= value / unit <= value,
        value < 0 ==> value <= value / unit < 0,
        0 <= value % unit < unit,
        value == (value / unit) * unit + value % unit,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(value, unit);
    let quotient = value / unit;
    let remainder = value % unit;
    assert(0 <= remainder < unit);
    assert(unit * quotient == quotient * unit) by (nonlinear_arith);
    if value >= 0 {
        assert(0 <= quotient <= value) by (nonlinear_arith)
            requires
                value == quotient * unit + remainder,
                0 <= remainder < unit,
                unit >= 1,
                value >= 0,
        ;
    } else {
        assert(value <= quotient < 0) by (nonlinear_arith)
            requires
                value == quotient * unit + remainder,
                0 <= remainder < unit,
                unit >= 1,
                value < 0,
        ;
    }
}

/// A unit beyond `i128::MAX` is at least `10^39`, which rounds every `i128` to zero.
pub(crate) proof fn lemma_rounds_to_zero_beyond_i128(value: int, places: nat)
    requires
        i128::MIN <= value <= i128::MAX,
        decimal_unit(places) > i128::MAX,
    ensures
        half_even_multiple(value, decimal_unit(places)) == 0,
{
    assert(decimal_unit(38) == 100_000_000_000_000_000_000_000_000_000_000_000_000int)
        by (compute_only);
    if places <= 38 {
        lemma_decimal_unit_grows(places, 38);
    }
    assert(decimal_unit(39) == 1_000_000_000_000_000_000_000_000_000_000_000_000_000int)
        by (compute_only);
    lemma_decimal_unit_grows(39, places);
    let unit = decimal_unit(places);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(value, unit);
    if value >= 0 {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(value, unit, 0, value);
    } else {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(value, unit, -1, value + unit);
    }
}

} // verus!

/// Adds one to the non-negative number spelled by ASCII decimal `digits`.
///
/// Trailing `9`s become `0`s and the digit before them grows by one; when
/// every digit is `9` (or there are none) a leading `1` is inserted. Verified: the digits stay ASCII
/// digits and spell exactly one more than before, and the length grows by
/// one exactly when every digit was `9`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(
    requires
        all_ascii_digits(old(digits)@),
    ensures
        all_ascii_digits(final(digits)@),
        decimal_value(final(digits)@) == decimal_value(old(digits)@) + 1,
        final(digits).len() == if all_equal(old(digits)@, 57) {
            old(digits).len() + 1
        } else {
            old(digits).len() as int
        },
))]
pub(crate) fn increment_decimal_digits(digits: &mut Vec<u8>) {
    #[cfg(verus_keep_ghost)]
    proof_decl! {
        let ghost original = digits@;
    }
    let mut index = digits.len();
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= digits.len(),
            digits.len() == original.len(),
            all_ascii_digits(original),
            digits@.subrange(0, index as int) == original.subrange(0, index as int),
            all_equal(digits@.subrange(index as int, digits.len() as int), 48),
            all_equal(original.subrange(index as int, original.len() as int), 57),
        decreases index,
    ))]
    while index > 0 && digits[index - 1] == b'9' {
        index -= 1;
        #[cfg(verus_keep_ghost)]
        proof_decl! {
            let ghost before = digits@;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(original[index as int] == before.subrange(0, index + 1)[index as int]);
        }
        digits[index] = b'0';
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(digits@.subrange(0, index as int) =~= original.subrange(0, index as int));
            assert forall|i: int| 0 <= i < digits.len() - index implies
                #[trigger] digits@.subrange(index as int, digits.len() as int)[i] == 48 by {
                if i > 0 {
                    assert(digits@.subrange(index as int, digits.len() as int)[i]
                        == before.subrange(index + 1, before.len() as int)[i - 1]);
                }
            }
            assert forall|i: int| 0 <= i < original.len() - index implies
                #[trigger] original.subrange(index as int, original.len() as int)[i] == 57 by {
                if i > 0 {
                    assert(original.subrange(index as int, original.len() as int)[i]
                        == original.subrange(index + 1, original.len() as int)[i - 1]);
                }
            }
        }
    }
    if index == 0 {
        #[cfg(verus_keep_ghost)]
        proof_decl! {
            let ghost zeroed = digits@;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(original =~= original.subrange(0, original.len() as int));
            assert(zeroed =~= zeroed.subrange(0, zeroed.len() as int));
            lemma_carry_out(original, zeroed);
        }
        digits.insert(0, b'1');
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(digits@ =~= seq![49u8] + zeroed);
        }
    } else {
        #[cfg(verus_keep_ghost)]
        proof_decl! {
            let ghost before = digits@;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(before.subrange(0, index as int)[index - 1] == before[index - 1]);
            assert(original.subrange(0, index as int)[index - 1] == original[index - 1]);
            assert(before[index - 1] == original[index - 1]);
            assert(48 <= original[index - 1] <= 57);
        }
        digits[index - 1] += 1;
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_increment_at(original, before, digits@, index - 1);
        }
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// When every digit was `9`, the zeroed digits plus a leading `1` spell one more.
pub(crate) proof fn lemma_carry_out(original: Seq<u8>, zeroed: Seq<u8>)
    requires
        all_equal(original, 57),
        all_equal(zeroed, 48),
        zeroed.len() == original.len(),
    ensures
        decimal_value(seq![49u8] + zeroed) == decimal_value(original) + 1,
        all_ascii_digits(seq![49u8] + zeroed),
{
    lemma_decimal_value_runs(original);
    lemma_decimal_value_runs(zeroed);
    lemma_decimal_value_concat(seq![49u8], zeroed);
    assert(seq![49u8].drop_last() =~= Seq::<u8>::empty());
    assert(decimal_value(Seq::<u8>::empty()) == 0);
    assert(decimal_value(seq![49u8]) == decimal_value(seq![49u8].drop_last()) * 10 + 1);
    assert(decimal_value(seq![49u8]) == 1);
    let one = decimal_value(seq![49u8]);
    let scale = decimal_unit(zeroed.len());
    assert(one * scale == scale) by (nonlinear_arith)
        requires
            one == 1,
    ;
    assert forall|i: int| 0 <= i < (seq![49u8] + zeroed).len() implies 48 <= #[trigger] (seq![49u8]
        + zeroed)[i] <= 57 by {
        if i > 0 {
            assert((seq![49u8] + zeroed)[i] == zeroed[i - 1]);
        }
    }
}

/// Incrementing the last non-`9` digit, with zeros after it, spells one more.
pub(crate) proof fn lemma_increment_at(original: Seq<u8>, before: Seq<u8>, after: Seq<u8>, index: int)
    requires
        0 <= index < original.len(),
        before.len() == original.len(),
        after.len() == original.len(),
        all_ascii_digits(original),
        before.subrange(0, index + 1) == original.subrange(0, index + 1),
        before[index] != 57,
        all_equal(before.subrange(index + 1, before.len() as int), 48),
        all_equal(original.subrange(index + 1, original.len() as int), 57),
        after[index] == before[index] + 1,
        after == before.update(index, after[index]),
    ensures
        all_ascii_digits(after),
        decimal_value(after) == decimal_value(original) + 1,
        !all_equal(original, 57),
{
    let length = original.len() as int;
    let head = original.subrange(0, index);
    let nines = original.subrange(index + 1, length);
    let zeros = after.subrange(index + 1, length);
    assert(original[index] == original.subrange(0, index + 1)[index]);
    assert(before[index] == before.subrange(0, index + 1)[index]);
    assert(original[index] == before[index]);
    assert forall|i: int| 0 <= i < index implies #[trigger] after[i] == original[i] by {
        assert(after[i] == before.update(index, after[index])[i]);
        assert(after[i] == before[i]);
        assert(before.subrange(0, index + 1)[i] == before[i]);
        assert(original.subrange(0, index + 1)[i] == original[i]);
        assert(before.subrange(0, index + 1)[i] == original.subrange(0, index + 1)[i]);
    }
    assert forall|i: int| index < i < after.len() implies #[trigger] after[i] == before[i] by {
        assert(after[i] == before.update(index, after[index])[i]);
    }
    assert(!all_equal(original, 57)) by {
        assert(original[index] != 57);
    }
    assert(zeros =~= before.subrange(index + 1, length));
    assert(original =~= (head + seq![original[index]]) + nines);
    assert(after =~= (head + seq![(original[index] + 1) as u8]) + zeros);
    lemma_decimal_value_concat(head + seq![original[index]], nines);
    lemma_decimal_value_concat(head + seq![(original[index] + 1) as u8], zeros);
    lemma_decimal_value_runs(nines);
    lemma_decimal_value_runs(zeros);
    assert((head + seq![original[index]]).drop_last() =~= head);
    assert((head + seq![(original[index] + 1) as u8]).drop_last() =~= head);
    let scale = decimal_unit(nines.len());
    let low = decimal_value(head + seq![original[index]]);
    assert(decimal_value(head + seq![(original[index] + 1) as u8]) == low + 1);
    assert((low + 1) * scale == low * scale + (scale - 1) + 1) by (nonlinear_arith);
    assert forall|i: int| 0 <= i < after.len() implies 48 <= #[trigger] after[i] <= 57 by {
        if i > index {
            assert(after[i] == zeros[i - index - 1]);
        }
    }
}

} // verus!

#[cfg(test)]
#[path = "decimal_tests.rs"]
mod tests;
