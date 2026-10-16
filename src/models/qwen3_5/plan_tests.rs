//! Qwen planning tests: schedule rule, cache slots, compact layouts, and MTP bookkeeping.

use proptest::prelude::*;

use super::{
    AppendFault, CycleFault, StagingFault, all_equal, append_end, cache_slots, call_rotary_context,
    columns_below, compact_row_plan, default_full_attention, derived_full_attention,
    first_context_above, first_staging_overflow, first_token_at_or_above, gated_query_width,
    prefix_rows, row_counts, shifted_mtp_length, shifted_mtp_lengths, signed_index, slot_positions,
    slot_validity, speculative_cycle_plan, staging_requirement, strictly_increasing, widest,
};

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

/// The derived schedule equals the per-layer rule and counts `floor(L / interval)` full layers.
#[test]
fn derived_schedule_counts_one_full_layer_per_period() {
    assert_eq!(
        derived_full_attention(8, 4),
        [false, false, false, true, false, false, false, true]
    );
    for interval in 0..7 {
        for layers in 0..40 {
            let schedule = derived_full_attention(layers, interval);
            let reference = (0..layers)
                .map(|index| default_full_attention(index, interval))
                .collect::<Vec<_>>();
            assert_eq!(schedule, reference);
            let full = schedule.iter().filter(|&&full| full).count();
            assert_eq!(
                full,
                if interval <= 1 {
                    layers
                } else {
                    layers / interval
                }
            );
        }
    }
}

/// Appends report the cache end or the precise failure.
#[test]
fn append_end_checks_overflow_and_capacity() {
    assert_eq!(append_end(3, 2, 5), Ok(5));
    assert_eq!(
        append_end(3, 3, 5),
        Err(AppendFault::ExceedsCapacity { requested: 6 })
    );
    assert_eq!(
        append_end(usize::MAX, 1, usize::MAX),
        Err(AppendFault::Overflow)
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

/// Column validation, counts, width, and prefix detection on hand-picked layouts.
#[test]
fn compact_layout_helpers_cover_edge_layouts() {
    let rows = vec![vec![0, 2], vec![], vec![1]];
    assert!(columns_below(&rows, 3));
    assert!(!columns_below(&rows, 2));
    assert!(columns_below(&[], 0));
    assert_eq!(row_counts(&rows), [2, 0, 1]);
    assert_eq!(widest(&[2, 0, 1]), 2);
    assert_eq!(widest(&[]), 0);
    assert!(!prefix_rows(&rows));
    assert!(prefix_rows(&[vec![0, 1], vec![], vec![0]]));
    assert!(strictly_increasing(&[]));
    assert!(strictly_increasing(&[7]));
    assert!(strictly_increasing(&[0, 2, 5]));
    assert!(!strictly_increasing(&[0, 2, 2]));
    assert!(!strictly_increasing(&[3, 1]));
}

/// Validity is omitted without padding and exact per slot otherwise.
#[test]
fn slot_validity_and_positions_are_exact() {
    assert_eq!(slot_validity(&[2, 2], 2), None);
    assert_eq!(
        slot_validity(&[2, 0, 1], 2),
        Some(vec![
            vec![true, true],
            vec![false, false],
            vec![true, false]
        ])
    );
    assert_eq!(
        slot_positions(&[0, 5], 3),
        Some(vec![vec![0, 1, 2], vec![5, 6, 7]])
    );
    assert_eq!(slot_positions(&[], 3), Some(Vec::new()));
    let limit = usize::try_from(i64::MAX).expect("64-bit host");
    assert_eq!(
        slot_positions(&[limit - 1], 2),
        Some(vec![vec![i64::MAX - 1, i64::MAX]])
    );
    assert_eq!(slot_positions(&[limit - 1], 3), None);
    assert_eq!(slot_positions(&[usize::MAX], 1), None);
    assert_eq!(slot_positions(&[usize::MAX - 1, 0], 3), None);
}

/// Equality, saturating index conversion, and per-row MTP truncation.
#[test]
fn small_bookkeeping_helpers_match_their_definitions() {
    assert!(all_equal(&[]));
    assert!(all_equal(&[4]));
    assert!(all_equal(&[4, 4, 4]));
    assert!(!all_equal(&[4, 4, 5]));
    assert_eq!(signed_index(0), 0);
    assert_eq!(signed_index(17), 17);
    assert_eq!(signed_index(usize::MAX), i64::MAX);
    assert_eq!(shifted_mtp_lengths(&[0, 4, 4], &[5, 5, 1]), [0, 3, 1]);
    assert_eq!(gated_query_width(24, 256, true), Some(12_288));
    assert_eq!(gated_query_width(24, 256, false), Some(6_144));
    assert_eq!(gated_query_width(1, usize::MAX, false), Some(usize::MAX));
    assert_eq!(gated_query_width(1, usize::MAX, true), None);
    assert_eq!(gated_query_width(usize::MAX, 2, false), None);
    assert_eq!(gated_query_width(usize::MAX, 0, true), Some(0));
}

/// The staging preflight reports the first overflowing row and saturates.
#[test]
fn staging_overflow_reports_the_first_row() {
    assert_eq!(first_staging_overflow(&[1, 1], &[2, 3], &[1, 1], 3), None);
    assert_eq!(
        first_staging_overflow(&[1, 1, 0], &[2, 3, 9], &[1, 2, 0], 2),
        Some((1, 4))
    );
    assert_eq!(
        first_staging_overflow(&[0], &[1], &[usize::MAX], 7),
        Some((0, usize::MAX))
    );
    // Rows pair like `zip`: the shortest slice bounds the check.
    assert_eq!(first_staging_overflow(&[0, 0], &[0, 9], &[1], 1), None);
}

/// Staging and cycle preflights return one precise reason in order.
#[test]
fn speculative_preflights_name_the_first_failure() {
    assert_eq!(staging_requirement(2, 3), Ok(3));
    assert_eq!(
        staging_requirement(3, 3),
        Err(StagingFault::ExceedsAvailable { required: 4 })
    );
    assert_eq!(
        staging_requirement(usize::MAX, usize::MAX),
        Err(StagingFault::Overflow)
    );

    assert_eq!(speculative_cycle_plan(&[5], &[4], 5, 3, 8), Ok(8));
    assert_eq!(
        speculative_cycle_plan(&[5], &[4], 5, 3, 7),
        Err(CycleFault::ExceedsCapacity { requested: 8 })
    );
    assert_eq!(
        speculative_cycle_plan(&[5], &[4], 4, 3, 8),
        Err(CycleFault::HistoryMismatch { expected: 5 })
    );
    assert_eq!(
        speculative_cycle_plan(&[5], &[5], 5, 3, 8),
        Err(CycleFault::Misaligned)
    );
    assert_eq!(
        speculative_cycle_plan(&[5, 5], &[4, 4], 5, 3, 8),
        Err(CycleFault::Misaligned)
    );
    assert_eq!(
        speculative_cycle_plan(&[0], &[usize::MAX], 0, 1, 8),
        Err(CycleFault::Misaligned)
    );
    assert_eq!(
        speculative_cycle_plan(&[usize::MAX], &[usize::MAX - 1], usize::MAX, 1, usize::MAX),
        Err(CycleFault::Overflow)
    );
}

/// The call context is the largest active context, or one without active rows.
#[test]
fn call_rotary_context_skips_inactive_rows() {
    assert_eq!(call_rotary_context(&[0, 2, 1], &[9, 3, 5]), 5);
    assert_eq!(call_rotary_context(&[0, 0], &[9, 3]), 1);
    assert_eq!(call_rotary_context(&[], &[]), 1);
    assert_eq!(call_rotary_context(&[1, 1], &[4]), 4);
    assert_eq!(first_token_at_or_above(&[1, 7, 9], 8), Some(2));
    assert_eq!(first_token_at_or_above(&[1, 7], 8), None);
    assert_eq!(first_token_at_or_above(&[], 0), None);
    assert_eq!(first_token_at_or_above(&[0], 0), Some(0));
}

/// The cached-context preflight names the first active row beyond the limit.
#[test]
fn first_context_above_names_the_first_active_offender() {
    // Row 0 is inactive, so its larger context is ignored; row 2 is first.
    assert_eq!(
        first_context_above(&[0, 1, 2, 1], &[90, 8, 12, 30], 10),
        Some(12)
    );
    // A context equal to the limit is still cacheable.
    assert_eq!(first_context_above(&[1, 1], &[10, 10], 10), None);
    assert_eq!(first_context_above(&[0, 0], &[90, 91], 10), None);
    assert_eq!(first_context_above(&[], &[], 0), None);
    // Rows pair like `zip`: the unpaired count is ignored.
    assert_eq!(first_context_above(&[1, 1], &[4], 3), Some(4));
    assert_eq!(first_context_above(&[1], &[2, 99], 3), None);
}

proptest! {
    /// Layout helpers equal their iterator definitions on arbitrary layouts.
    #[test]
    fn compact_layout_helpers_match_iterator_definitions(
        rows in proptest::collection::vec(proptest::collection::vec(0usize..8, 0..6), 0..5),
        sequence in 0usize..9,
    ) {
        prop_assert_eq!(
            columns_below(&rows, sequence),
            !rows.iter().any(|row| row.iter().any(|&column| column >= sequence))
        );
        let counts = row_counts(&rows);
        prop_assert_eq!(&counts, &rows.iter().map(Vec::len).collect::<Vec<_>>());
        let width = widest(&counts);
        prop_assert_eq!(width, counts.iter().copied().max().unwrap_or(0));
        prop_assert_eq!(
            prefix_rows(&rows),
            rows.iter().all(|row| row.iter().enumerate().all(|(slot, &column)| slot == column))
        );
        for row in &rows {
            prop_assert_eq!(
                strictly_increasing(row),
                row.windows(2).all(|pair| pair[0] < pair[1])
            );
        }
        let flags = slot_validity(&counts, width).map(|rows| rows.concat());
        let expected = counts.iter().any(|&count| count < width).then(|| {
            counts
                .iter()
                .flat_map(|&count| (0..width).map(move |slot| slot < count))
                .collect::<Vec<_>>()
        });
        prop_assert_eq!(flags, expected);
    }

    /// Positions and staging requests equal their checked iterator definitions.
    #[test]
    fn positions_and_requests_match_iterator_definitions(
        starts in proptest::collection::vec(prop_oneof![0usize..100, Just(usize::MAX - 2)], 0..4),
        width in 0usize..5,
        committed in proptest::collection::vec(0usize..10, 0..4),
        extra in proptest::collection::vec(0usize..10, 0..4),
        counts in proptest::collection::vec(prop_oneof![0usize..10, Just(usize::MAX)], 0..4),
        capacity in 0usize..20,
    ) {
        let positions = slot_positions(&starts, width).map(|rows| rows.concat());
        let expected = starts
            .iter()
            .flat_map(|&start| (0..width).map(move |slot| {
                start.checked_add(slot).and_then(|value| i64::try_from(value).ok())
            }))
            .collect::<Option<Vec<_>>>();
        prop_assert_eq!(positions, expected);

        let working = committed.iter().zip(&extra).map(|(&c, &e)| c + e).collect::<Vec<_>>();
        let reference = committed
            .iter()
            .zip(&working)
            .zip(&counts)
            .enumerate()
            .map(|(row, ((&c, &w), &n))| (row, (w - c).saturating_add(n)))
            .find(|&(_, requested)| requested > capacity);
        prop_assert_eq!(
            first_staging_overflow(&committed, &working, &counts, capacity),
            reference
        );
    }

    /// The call context equals the iterator maximum over active rows.
    #[test]
    fn call_rotary_context_matches_the_iterator_maximum(
        counts in proptest::collection::vec(0usize..3, 0..6),
        contexts in proptest::collection::vec(0usize..50, 0..6),
    ) {
        let reference = contexts
            .iter()
            .zip(&counts)
            .filter(|(_, count)| **count > 0)
            .map(|(&context, _)| context)
            .max()
            .unwrap_or(1);
        prop_assert_eq!(call_rotary_context(&counts, &contexts), reference);
    }

    /// The cached-context preflight equals the per-row loop it replaced.
    #[test]
    fn first_context_above_matches_the_active_row_loop(
        counts in proptest::collection::vec(0usize..3, 0..6),
        contexts in proptest::collection::vec(0usize..50, 0..6),
        limit in 0usize..50,
    ) {
        let reference = contexts
            .iter()
            .zip(&counts)
            .filter(|(_, count)| **count > 0)
            .map(|(&context, _)| context)
            .find(|&context| context > limit);
        prop_assert_eq!(first_context_above(&counts, &contexts, limit), reference);
    }

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
