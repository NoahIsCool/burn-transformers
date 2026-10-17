//! Recurrent snapshot indexing against hand-picked batches and naive oracles.

use proptest::prelude::*;

use super::*;

/// The state after `count` tokens is snapshot `count - 1`; zero keeps committed state.
#[test]
fn last_snapshot_selects_state_after_final_token() {
    assert_eq!(last_snapshot(0), None);
    assert_eq!(last_snapshot(1), Some(0));
    assert_eq!(last_snapshot(5), Some(4));
    assert_eq!(last_snapshot(usize::MAX), Some(usize::MAX - 1));
}

/// A shared positive count selects one snapshot; ragged or empty batches do not.
#[test]
fn uniform_snapshot_requires_equal_positive_counts() {
    assert_eq!(uniform_snapshot(&[3, 3, 3]), Some(2));
    assert_eq!(uniform_snapshot(&[1]), Some(0));
    assert_eq!(uniform_snapshot(&[0, 0]), None);
    assert_eq!(uniform_snapshot(&[2, 3]), None);
    assert_eq!(uniform_snapshot(&[]), None);
}

/// Flat indices address each row's own snapshot block and report overflow.
#[test]
fn flat_snapshot_indices_address_each_rows_block() {
    // Capacity 4: row 0 reads slot 2, row 1 (no proposal) slot 0, row 2 slot 3.
    assert_eq!(flat_snapshot_indices(&[3, 0, 4], 4), Ok(vec![2, 4, 11]));
    assert_eq!(flat_snapshot_indices(&[], 4), Ok(Vec::new()));
    let capacity = 1_usize << 30;
    assert_eq!(
        flat_snapshot_indices(&[1, 1, 1], capacity),
        Err(2 * capacity)
    );
    assert_eq!(flat_snapshot_indices(&[1, 1], usize::MAX), Err(usize::MAX));
    // A representable row base plus an unrepresentable snapshot saturates.
    assert_eq!(flat_snapshot_indices(&[0, usize::MAX], 2), Err(usize::MAX));
}

/// Offsets report the first row that differs from row zero.
#[test]
fn first_nonuniform_offset_reports_lowest_differing_row() {
    assert_eq!(first_nonuniform_offset(&[2, 2, 2]), None);
    assert_eq!(first_nonuniform_offset(&[2, 2, 1, 3]), Some(2));
    assert_eq!(first_nonuniform_offset(&[]), None);
}

proptest! {
    /// Flat indices equal the closed form and decode back to `(row, snapshot)`.
    #[test]
    fn flat_snapshot_indices_match_closed_form(
        staged in prop::collection::vec(0_usize..6, 0..6),
        capacity in 1_usize..6,
    ) {
        let flat = flat_snapshot_indices(&staged, capacity).expect("small indices fit");
        for (row, (&index, &count)) in flat.iter().zip(&staged).enumerate() {
            let index = usize::try_from(index).expect("non-negative");
            prop_assert_eq!(index, row * capacity + count.max(1) - 1);
            if count <= capacity {
                prop_assert_eq!(index / capacity, row);
                prop_assert_eq!(index % capacity, count.max(1) - 1);
            }
        }
    }
}
