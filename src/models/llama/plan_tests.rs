//! Planning functions against direct reference formulations.
//!
//! The Verus harness proves these postconditions for all inputs; the tests
//! pin concrete values and cross-check each function against a naive
//! implementation of its specification.

use proptest::prelude::*;

use super::*;

/// Layers are numbered densely inside their own group.
#[test]
fn cache_slots_number_each_group_densely() {
    let plan = cache_slots(&[false, true, true, false, true]);
    assert_eq!(plan.slots, [0, 0, 1, 1, 2]);
    assert_eq!((plan.full_layers, plan.ring_layers), (2, 3));
    let empty = cache_slots(&[]);
    assert!(empty.slots.is_empty());
    assert_eq!((empty.full_layers, empty.ring_layers), (0, 0));
}

/// Slot positions continue the cache inside the run and fill elsewhere.
#[test]
fn slot_positions_continue_the_cache_and_fill_padding() {
    assert_eq!(slot_positions(5, 1, 2, 4), Some(vec![6, 5, 6, 6]));
    assert_eq!(slot_positions(5, 0, 0, 2), Some(vec![4, 4]));
    assert_eq!(slot_positions(0, 0, 0, 3), Some(vec![0, 0, 0]));
    assert_eq!(slot_positions(usize::MAX, 0, 1, 1), None);
}

/// Compaction gathers valid columns and repeats the first one as padding.
#[test]
fn compaction_repeats_the_first_valid_column() {
    assert_eq!(compaction_sources(&[1, 3], 4), [1, 3, 1, 1]);
    assert_eq!(compaction_sources(&[], 2), [0, 0]);
}

/// Scattering maps valid columns to their compact slot and padding to the zero slot.
#[test]
fn scatter_restores_valid_columns_and_zeroes_padding() {
    assert_eq!(scatter_sources(&[1, 3], 5, 2), [2, 0, 2, 1, 2]);
    assert_eq!(scatter_sources(&[], 3, 4), [4, 4, 4]);
}

/// Runs are recognized by their span; gaps and the empty row are classified exactly.
#[test]
fn runs_are_recognized_by_their_span() {
    assert!(is_one_run(&[]));
    assert!(is_one_run(&[4]));
    assert!(is_one_run(&[2, 3, 4]));
    assert!(!is_one_run(&[2, 4]));
    assert!(is_one_run(&[usize::MAX - 1, usize::MAX]));
}

/// Restoration offsets compact indices by the run's first slot.
#[test]
fn restore_slots_offset_the_compact_indices() {
    assert_eq!(restore_slots(&[1, 3], 5, 4, 2), [4, 2, 4, 3, 4]);
    assert_eq!(restore_slots(&[], 2, 0, 0), [0, 0]);
}

/// Strictly increasing columns below `bound`.
fn sorted_columns(bound: usize) -> impl Strategy<Value = Vec<usize>> {
    prop::collection::btree_set(0..bound, 0..=bound).prop_map(|set| set.into_iter().collect())
}

proptest! {
    /// Cache slots equal per-group prefix counts.
    #[test]
    fn cache_slots_equal_prefix_counts(ring in prop::collection::vec(any::<bool>(), 0..24)) {
        let plan = cache_slots(&ring);
        for (layer, &in_ring) in ring.iter().enumerate() {
            let earlier = ring[..layer].iter().filter(|&&other| other == in_ring).count();
            prop_assert_eq!(plan.slots[layer], earlier);
        }
        prop_assert_eq!(plan.ring_layers, ring.iter().filter(|&&flag| flag).count());
        prop_assert_eq!(plan.full_layers + plan.ring_layers, ring.len());
    }

    /// Slot positions equal the module's definition slot by slot.
    #[test]
    fn slot_positions_match_the_definition(
        (start, block, column, count) in (0usize..1000, 0usize..12).prop_flat_map(|(start, block)| {
            (Just(start), Just(block), 0..=block).prop_flat_map(|(start, block, column)| {
                (Just(start), Just(block), Just(column), 0..=block - column)
            })
        }),
    ) {
        let positions = slot_positions(start, column, count, block).unwrap();
        let fill = if count > 0 { start + count - 1 } else { start.saturating_sub(1) };
        for (slot, &position) in positions.iter().enumerate() {
            let expected = if (column..column + count).contains(&slot) {
                start + slot - column
            } else {
                fill
            };
            prop_assert_eq!(position, expected);
        }
    }

    /// The run test equals a direct check that consecutive columns differ by one.
    #[test]
    fn run_test_matches_consecutive_differences(columns in sorted_columns(12)) {
        let consecutive = columns.windows(2).all(|pair| pair[1] == pair[0] + 1);
        prop_assert_eq!(is_one_run(&columns), consecutive);
    }

    /// Restoration reads each valid column's own slot and the zero slot elsewhere.
    #[test]
    fn restore_slots_match_the_definition(columns in sorted_columns(10), offset in 0usize..3) {
        let block = columns.len() + offset + 1;
        let slots = restore_slots(&columns, 10, block, offset);
        for (column, &slot) in slots.iter().enumerate() {
            let expected = columns
                .iter()
                .position(|&valid| valid == column)
                .map_or(block, |index| offset + index);
            prop_assert_eq!(slot, expected);
        }
    }

    /// Compaction followed by scattering is the identity on valid columns.
    #[test]
    fn compaction_and_scatter_are_inverse_on_valid_columns(
        columns in sorted_columns(10),
        extra in 0usize..3,
    ) {
        let block = columns.len() + extra;
        let sources = compaction_sources(&columns, block);
        let back = scatter_sources(&columns, 10, block);
        for (column, &slot) in back.iter().enumerate() {
            match columns.iter().position(|&valid| valid == column) {
                Some(index) => {
                    prop_assert_eq!(slot, index);
                    prop_assert_eq!(sources[slot], column);
                }
                None => prop_assert_eq!(slot, block),
            }
        }
    }
}

/// The SmolLM3 and Qwen schedule rules match their reference formulas.
#[test]
fn schedule_rules_match_the_reference_formulas() {
    let nope: Vec<bool> = (0..8).map(|layer| rotary_by_interval(layer, 4)).collect();
    assert_eq!(nope, [true, true, true, false, true, true, true, false]);
    assert!(rotary_by_interval(0, 2) && !rotary_by_interval(1, 1));
    assert!(sliding_from_layer(28, 28) && !sliding_from_layer(27, 28));
}
