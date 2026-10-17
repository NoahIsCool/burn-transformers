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

/// Render limits follow their formulas, and the budget always admits one maximal value.
#[test]
fn render_limits_derive_from_the_output_limit() {
    let empty = render_limits(0);
    assert_eq!(empty.output_bytes, OUTPUT_FLOOR_BYTES);
    assert_eq!(empty.value_bytes, OUTPUT_FLOOR_BYTES);
    assert_eq!(empty.sequence_items, MIN_SEQUENCE_ITEMS);
    assert_eq!(empty.total_bytes, 16 * OUTPUT_FLOOR_BYTES);
    let huge = render_limits(usize::MAX);
    assert_eq!(huge.output_bytes, usize::MAX);
    assert_eq!(huge.sequence_items, usize::MAX);
    assert_eq!(huge.total_bytes, usize::MAX);
}

/// The allowance is the per-value limit or the remaining budget, whichever is smaller.
#[test]
fn text_allowance_is_the_smaller_remaining_limit() {
    let limits = RenderLimits {
        output_bytes: 256,
        value_bytes: 64,
        sequence_items: 8,
        total_bytes: 100,
    };
    assert_eq!(text_allowance(limits, 0), 64);
    assert_eq!(text_allowance(limits, 40), 60);
    assert_eq!(text_allowance(limits, 100), 0);
    assert_eq!(text_allowance(limits, 150), 0);
}

/// Copy charges, buffer checks, and cleanup fuel saturate instead of wrapping.
#[test]
fn bookkeeping_saturates_at_the_type_limits() {
    assert_eq!(copy_charge(3, 2, 24), 144);
    assert_eq!(copy_charge(usize::MAX, 2, 24), usize::MAX);
    assert_eq!(copy_charge(usize::MAX, 2, 0), 0);
    assert_eq!(copy_charge(usize::MAX, 1, 1), usize::MAX);
    assert!(appended_within(3, 2, 5));
    assert!(!appended_within(3, 3, 5));
    assert!(!appended_within(usize::MAX, 1, usize::MAX - 1));
    assert!(appended_within(usize::MAX, 1, usize::MAX));
    assert_eq!(cleanup_fuel(2, 2, 32), 160);
    assert_eq!(cleanup_fuel(usize::MAX, 0, 1), usize::MAX as u64);
    assert_eq!(cleanup_fuel(usize::MAX, usize::MAX, 32), u64::MAX);
}

proptest! {
    /// The allowance admits exactly the texts a reservation accepts.
    #[test]
    fn text_allowance_admits_exactly_the_reservable_texts(
        value_bytes in 0_usize..1000,
        total_bytes in 0_usize..1000,
        produced in 0_usize..1000,
        bytes in 0_usize..1100,
    ) {
        prop_assume!(produced <= total_bytes);
        let limits = RenderLimits { output_bytes: value_bytes, value_bytes, sequence_items: 1, total_bytes };
        let reservable = bytes <= value_bytes && charged_total(produced, bytes, total_bytes).is_some();
        prop_assert_eq!(bytes <= text_allowance(limits, produced), reservable);
    }

    /// Copy charges, buffer checks, and fuel agree with wide arithmetic.
    #[test]
    fn bookkeeping_agrees_with_wide_arithmetic(
        items in any::<usize>(),
        values in 0_usize..8,
        item_bytes in 0_usize..64,
        current in any::<usize>(),
        added in any::<usize>(),
        limit in any::<usize>(),
        groups in any::<usize>(),
        per_step in any::<u64>(),
    ) {
        let product = items as u128 * values as u128 * item_bytes as u128;
        prop_assert_eq!(copy_charge(items, values, item_bytes) as u128, product.min(usize::MAX as u128));
        let grown = (current as u128 + added as u128).min(usize::MAX as u128);
        prop_assert_eq!(appended_within(current, added, limit), grown <= limit as u128);
        let steps = (items as u128 + groups as u128 + 1).min(usize::MAX as u128);
        let fuel = (steps * per_step as u128).min(u64::MAX as u128);
        prop_assert_eq!(cleanup_fuel(items, groups, per_step) as u128, fuel);
    }

    /// Replacement bounds equal their saturated formula.
    #[test]
    fn replacement_bound_is_its_formula(
        length in any::<usize>(),
        characters in any::<usize>(),
        from in 0_usize..16,
        to in 0_usize..16,
        count in proptest::option::of(any::<usize>()),
    ) {
        let possible = length
            .checked_div(from)
            .map_or((characters as u128 + 1).min(usize::MAX as u128), |quotient| quotient as u128);
        let matches = count.map_or(possible, |count| possible.min(count as u128));
        let expected = (length as u128 + matches * to.saturating_sub(from) as u128).min(usize::MAX as u128);
        prop_assert_eq!(replacement_bound(length, characters, from, to, count) as u128, expected);
    }

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
