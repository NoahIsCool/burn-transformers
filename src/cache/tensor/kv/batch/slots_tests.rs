//! Batched read slot tables against hand-picked rows and naive oracles.

use proptest::prelude::*;

use super::*;

/// The largest value bounds every entry and is zero for an empty slice.
#[test]
fn max_value_matches_iterator_max() {
    assert_eq!(max_value(&[]), 0);
    assert_eq!(max_value(&[3, 9, 2]), 9);
    assert_eq!(max_value(&[0, 0]), 0);
}

/// Full-storage tables mark each row's visible window `[L - keep, L)`.
#[test]
fn full_tables_mark_visible_windows() {
    assert_eq!(
        full_slot_mask(&[3, 1], &[2, 1], 3),
        [false, true, true, true, false, false]
    );
    assert!(full_slots_all_valid(&[3, 3], &[3, 3], 3));
    assert!(!full_slots_all_valid(&[3, 2], &[3, 2], 3));
    assert!(!full_slots_all_valid(&[3, 3], &[3, 2], 3));
    // Only zipped rows are compared, whichever slice is longer.
    assert!(full_slots_all_valid(&[3, 3, 1], &[3, 3], 3));
    assert!(full_slots_all_valid(&[3], &[3, 0], 3));
}

/// Table reservations never overflow: an unrepresentable size reserves nothing.
#[test]
fn table_reservations_saturate_to_empty() {
    let table = table_with_capacity::<u8>(3, 4);
    assert!(table.is_empty() && table.capacity() >= 12);
    let unrepresentable = table_with_capacity::<u8>(usize::MAX, 2);
    assert!(unrepresentable.is_empty());
    assert_eq!(unrepresentable.capacity(), 0);
}

/// Ring tables hold each slot's newest position and its window membership.
#[test]
fn ring_tables_hold_newest_positions() {
    // Width 4: row 0 has length 6 (slots hold 4, 5, 2, 3), row 1 length 2.
    let (positions, valid) = ring_slot_table(&[6, 2], &[3, 2], 4, 4);
    assert_eq!(positions, [4, 5, 2, 3, 0, 1, 0, 0]);
    // Row 0 keeps [3, 6): positions 4, 5, and 3 are visible, 2 is not.
    assert_eq!(valid, [true, true, false, true, true, true, false, false]);
    // Huge lengths never overflow the visibility comparison.
    let (positions, valid) = ring_slot_table(&[usize::MAX], &[usize::MAX], 2, 2);
    assert_eq!(positions, [usize::MAX - 1, usize::MAX - 2]);
    assert_eq!(valid, [true, true]);
}

/// Staged tables number proposals from each row's committed length.
#[test]
fn staged_tables_number_proposals_from_committed_length() {
    assert_eq!(staged_slot_positions(&[5, 0], 2), [5, 6, 0, 1]);
    assert_eq!(
        staged_slot_positions(&[usize::MAX], 2),
        [usize::MAX, usize::MAX]
    );
    let windows = [
        WorkingWindow {
            retained_start: 0,
            committed_keep: 0,
            staged_skip: 1,
        },
        WorkingWindow {
            retained_start: 0,
            committed_keep: 0,
            staged_skip: 0,
        },
    ];
    assert_eq!(
        staged_slot_mask(&[3, 1], &windows, 3),
        [false, true, true, true, false, false]
    );
    assert!(!staged_slots_all_valid(&[3, 3], &windows, 3));
    assert!(staged_slots_all_valid(&[3], &windows[1..], 3));
    // Extra staged lengths beyond the windows are not compared.
    assert!(staged_slots_all_valid(&[3, 0], &windows[1..], 3));
}

/// Vector scans report the first layer whose lengths differ.
#[test]
fn first_unequal_vector_reports_lowest_layer() {
    let reference = [1, 2];
    assert_eq!(
        first_unequal_vector(&[vec![1, 2], vec![1, 2]], &reference),
        None
    );
    assert_eq!(
        first_unequal_vector(&[vec![1, 2], vec![1, 3], vec![0]], &reference),
        Some(1)
    );
    assert_eq!(first_unequal_vector(&[vec![1]], &reference), Some(0));
    assert_eq!(first_unequal_vector(&[], &reference), None);
}

proptest! {
    /// Ring tables agree with the direct per-slot computation they replaced.
    #[test]
    fn ring_tables_match_direct_computation(
        rows in prop::collection::vec((0_usize..20, 0_usize..8), 0..4),
        width in 1_usize..6,
    ) {
        let lengths: Vec<usize> = rows.iter().map(|row| row.0).collect();
        let keep: Vec<usize> = rows.iter().map(|row| row.1.min(row.0).min(width)).collect();
        let slots = max_value(&lengths).min(width);
        let (positions, valid) = ring_slot_table(&lengths, &keep, width, slots);
        let mut expected_positions = Vec::new();
        let mut expected_valid = Vec::new();
        for (&length, &keep) in lengths.iter().zip(&keep) {
            for slot in 0..slots {
                let position = ring_slot_position(length, slot, width);
                expected_valid.push(position.is_some_and(|position| position + keep >= length));
                expected_positions.push(position.unwrap_or(0));
            }
        }
        prop_assert_eq!(positions, expected_positions);
        prop_assert_eq!(valid, expected_valid);
    }

    /// Full-storage masks agree with window membership.
    #[test]
    fn full_masks_match_window_membership(
        rows in prop::collection::vec((0_usize..8, 0_usize..8), 0..4),
    ) {
        let lengths: Vec<usize> = rows.iter().map(|row| row.0).collect();
        let keep: Vec<usize> = rows.iter().map(|row| row.1.min(row.0)).collect();
        let slots = max_value(&lengths);
        let mask = full_slot_mask(&lengths, &keep, slots);
        for row in 0..rows.len() {
            for slot in 0..slots {
                prop_assert_eq!(
                    mask[row * slots + slot],
                    (lengths[row] - keep[row]..lengths[row]).contains(&slot)
                );
            }
        }
    }
}
