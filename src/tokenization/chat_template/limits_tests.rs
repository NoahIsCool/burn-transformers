//! Limit-arithmetic tests: exact values, saturation, and properties matching the Verus contracts.

use proptest::prelude::*;

use super::*;

/// Limits follow their formulas and saturate instead of wrapping.
#[test]
fn limits_are_exact_then_saturate() {
    assert_eq!(output_limit(0), OUTPUT_FLOOR_BYTES);
    assert_eq!(output_limit(10), OUTPUT_FLOOR_BYTES + 40);
    assert_eq!(output_limit(usize::MAX), usize::MAX);
    let largest_exact = (usize::MAX - OUTPUT_FLOOR_BYTES) / 4;
    assert_eq!(
        output_limit(largest_exact),
        OUTPUT_FLOOR_BYTES + 4 * largest_exact
    );
    assert_eq!(output_limit(largest_exact + 1), usize::MAX);
    assert_eq!(total_budget(3), 48);
    assert_eq!(total_budget(usize::MAX / 16), (usize::MAX / 16) * 16);
    assert_eq!(total_budget(usize::MAX / 16 + 1), usize::MAX);
}

/// A charge succeeds exactly when the new total fits the budget.
#[test]
fn charges_keep_the_total_within_budget() {
    assert_eq!(charged_total(0, 10, 10), Some(10));
    assert_eq!(charged_total(5, 6, 10), None);
    assert_eq!(charged_total(0, 11, 10), None);
    assert_eq!(charged_total(usize::MAX, 1, usize::MAX), None);
    assert_eq!(
        charged_total(usize::MAX - 1, 1, usize::MAX),
        Some(usize::MAX)
    );
}

/// Repetition sizes are exact products or `None` on overflow.
#[test]
fn repetition_sizes_are_checked_products() {
    assert_eq!(repeated_size(2, 3), Some(6));
    assert_eq!(repeated_size(0, usize::MAX), Some(0));
    assert_eq!(repeated_size(usize::MAX, 0), Some(0));
    assert_eq!(repeated_size(2, 1 << 62), Some(1 << 63));
    assert_eq!(repeated_size(2, 1 << 63), None);
}

/// Replacement and indentation bounds match the text the operations produce.
#[test]
fn text_bounds_dominate_the_produced_text() {
    let cases = [
        ("abc", "", "xy", None),
        ("aaaa", "a", "bb", Some(2)),
        ("aaaa", "aa", "", None),
        ("é-é", "-", "+++", None),
        ("", "", "z", None),
    ];
    for (text, from, to, count) in cases {
        let bound = replacement_bound(
            text.len(),
            text.chars().count(),
            from.len(),
            to.len(),
            count,
        );
        let produced = match count {
            Some(count) => text.replacen(from, to, count),
            None => text.replace(from, to),
        };
        assert!(bound >= produced.len(), "{text:?} {from:?} {to:?}: {bound}");
    }
    assert_eq!(replacement_bound(3, 3, 0, 2, None), 3 + 4 * 2);
    assert_eq!(
        replacement_bound(usize::MAX, usize::MAX, 0, 2, None),
        usize::MAX
    );
    assert_eq!(indentation_bound(3, 2, 4), 13);
    assert_eq!(indentation_bound(3, 2, usize::MAX), usize::MAX);
    assert_eq!(indentation_bound(usize::MAX, 1, 0), usize::MAX);
}

proptest! {
    /// Replacement bounds dominate `str::replace` and `str::replacen` on arbitrary text.
    #[test]
    fn replacement_bound_dominates_str_replace(
        text in "[a-cé]{0,24}",
        from in "[a-cé]{0,3}",
        to in "[a-cé]{0,5}",
        count in proptest::option::of(0_usize..6),
    ) {
        let bound = replacement_bound(text.len(), text.chars().count(), from.len(), to.len(), count);
        let produced = match count {
            Some(count) => text.replacen(from.as_str(), &to, count),
            None => text.replace(from.as_str(), &to),
        };
        prop_assert!(bound >= produced.len());
    }

    /// Charges agree with exact integer arithmetic.
    #[test]
    fn charges_agree_with_wide_arithmetic(produced in any::<usize>(), bytes in any::<usize>(), total in any::<usize>()) {
        let wide = produced as u128 + bytes as u128;
        let expected = (wide <= total as u128).then_some(wide as usize);
        prop_assert_eq!(charged_total(produced, bytes, total), expected);
    }
}
