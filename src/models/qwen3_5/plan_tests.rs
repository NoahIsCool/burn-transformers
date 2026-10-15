//! Qwen planning tests: schedule rule, cache slots, compact row plans, and MTP truncation.

use proptest::prelude::*;

use super::{cache_slots, compact_row_plan, default_full_attention, shifted_mtp_length};

/// The overflow-free schedule rule equals `(i + 1) mod interval = 0` (interval zero acts as one).
#[test]
fn default_schedule_matches_the_reference_rule() {
    for interval in 0..9 {
        for index in 0..80 {
            let reference = (index + 1) % interval.max(1) == 0;
            assert_eq!(default_full_attention(index, interval), reference);
        }
    }
    assert!(default_full_attention(usize::MAX, 1));
    // Released Qwen3.5: three linear layers, then one full-attention layer.
    let schedule = (0..8)
        .map(|index| default_full_attention(index, 4))
        .collect::<Vec<_>>();
    assert_eq!(
        schedule,
        [false, false, false, true, false, false, false, true]
    );
}

/// The documented example schedule maps to dense slots per family.
#[test]
fn cache_slots_number_each_family_densely() {
    let plan = cache_slots(&[false, false, true, false, true]);
    assert_eq!(plan.slots, [0, 1, 0, 2, 1]);
    assert_eq!((plan.full, plan.recurrent), (2, 3));
    let empty = cache_slots(&[]);
    assert!(empty.slots.is_empty() && empty.full == 0 && empty.recurrent == 0);
}

/// Compact plans gather valid columns first and restore padding from the zero slot.
#[test]
fn compact_row_plans_cover_every_padding_shape() {
    // Left padding inside a wider block.
    let plan = compact_row_plan(&[2, 3], 4, 3);
    assert_eq!(plan.sources, [2, 3, 2]);
    assert_eq!(plan.slots, [3, 3, 0, 1]);
    // Interior padding.
    let plan = compact_row_plan(&[0, 2], 3, 2);
    assert_eq!(plan.sources, [0, 2]);
    assert_eq!(plan.slots, [0, 2, 1]);
    // An inactive row reads column 0 and restores zeros everywhere.
    let plan = compact_row_plan(&[], 2, 1);
    assert_eq!(plan.sources, [0]);
    assert_eq!(plan.slots, [1, 1]);
}

/// Truncation keeps the shifted MTP cache one behind the target.
#[test]
fn shifted_mtp_length_preserves_alignment() {
    assert_eq!(shifted_mtp_length(0, 5), 0);
    assert_eq!(shifted_mtp_length(4, 5), 3);
    assert_eq!(shifted_mtp_length(4, 3), 3);
    assert_eq!(shifted_mtp_length(4, 1), 1);
    assert_eq!(shifted_mtp_length(1, 0), 0);
}

proptest! {
    /// Slots are exactly `0..size` within each family, in schedule order.
    #[test]
    fn cache_slots_are_dense_and_ordered(kinds in proptest::collection::vec(any::<bool>(), 0..40)) {
        let plan = cache_slots(&kinds);
        let mut seen = [0usize, 0usize];
        for (kind, slot) in kinds.iter().zip(&plan.slots) {
            let family = usize::from(*kind);
            prop_assert_eq!(*slot, seen[family]);
            seen[family] += 1;
        }
        prop_assert_eq!((plan.full, plan.recurrent), (seen[1], seen[0]));
    }

    /// Every compact plan round-trips the valid columns and zeroes the rest.
    #[test]
    fn compact_row_plans_round_trip(
        row in proptest::collection::vec(any::<bool>(), 1..12),
        extra in 0usize..3,
    ) {
        let columns = (0..row.len()).filter(|&column| row[column]).collect::<Vec<_>>();
        let width = columns.len() + extra;
        let plan = compact_row_plan(&columns, row.len(), width);
        prop_assert_eq!(plan.sources.len(), width);
        for (slot, &column) in columns.iter().enumerate() {
            prop_assert_eq!(plan.sources[slot], column);
            prop_assert_eq!(plan.slots[column], slot);
        }
        for (column, &slot) in plan.slots.iter().enumerate() {
            if !row[column] {
                prop_assert_eq!(slot, width);
            }
        }
        prop_assert!(plan.sources.iter().all(|&source| source < row.len()));
    }

    /// The MTP cursor never grows and realigns whenever it was aligned.
    #[test]
    fn shifted_mtp_length_properties(target in 0usize..1_000, mtp in 0usize..1_000) {
        let shifted = shifted_mtp_length(target, mtp);
        prop_assert!(shifted <= mtp);
        prop_assert_eq!(shifted, mtp.min(target.saturating_sub(1)));
        if target > 0 && mtp + 1 >= target {
            prop_assert_eq!(shifted + 1, target);
        }
    }
}
