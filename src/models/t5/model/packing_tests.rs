//! Packing plans for batched stateful rows: layouts, round trips, and malformed inputs.

use proptest::prelude::*;

use super::*;

/// Lists the valid columns of every row of a row-major mask.
fn columns(mask: &[Vec<bool>]) -> Vec<Vec<usize>> {
    mask.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter_map(|(column, &valid)| valid.then_some(column))
                .collect()
        })
        .collect()
}

/// Unpadded, right-padded, and inactive rows pack as a prefix view without indices.
#[test]
fn right_aligned_rows_use_the_prefix_layout() {
    let rows = BatchedRows::new(
        &columns(&[
            vec![true, true, false],
            vec![true, false, false],
            vec![false, false, false],
        ]),
        3,
    )
    .unwrap();
    assert_eq!(rows.packing, Packing::Prefix);
    assert_eq!(rows.block, 2);
    assert_eq!(rows.counts, [2, 1, 0]);
    assert!(rows.has_padded_slots());
    assert_eq!(rows.padded_slots(), [false, false, false, true, true, true]);
    assert!(!rows.is_empty());
}

/// Left and interior padding need gathers whose indices address the right elements.
#[test]
fn left_and_interior_padding_use_exact_gathers() {
    let rows = BatchedRows::new(
        &columns(&[vec![false, true, true], vec![true, false, true]]),
        3,
    )
    .unwrap();
    let Packing::Gather { compact, scatter } = &rows.packing else {
        panic!("gapped rows need a gather");
    };
    // Row 0 reads columns 1, 2; row 1 reads columns 0, 2 (flat 3 + c).
    assert_eq!(compact, &[1, 2, 3, 5]);
    // Column 0 of row 0 is padding (reads slot 0, zeroed later); row 1 column 1 likewise.
    assert_eq!(scatter, &[0, 0, 1, 2, 2, 3]);
    assert!(!rows.has_padded_slots());
}

/// A call whose rows are all inactive has an empty block.
#[test]
fn all_inactive_rows_have_an_empty_block() {
    let rows = BatchedRows::new(&[Vec::new(), Vec::new()], 2).unwrap();
    assert!(rows.is_empty());
    assert_eq!(rows.packing, Packing::Prefix);
    assert_eq!(scatter_sources(&[0, 0], &[], 2, 0), Some(vec![0; 4]));
}

/// Unsorted, duplicated, missing, or out-of-range columns produce no plan.
#[test]
fn malformed_columns_are_rejected() {
    assert_eq!(BatchedRows::new(&[vec![2, 1]], 3), None);
    assert_eq!(BatchedRows::new(&[vec![1, 1]], 3), None);
    assert_eq!(BatchedRows::new(&[vec![0, 3]], 3), None);
    assert_eq!(BatchedRows::new(&[vec![0, 1, 2, 3]], 3), None);
    assert_eq!(compaction_sources(&[2], &[1], 3, 2), None);
    assert_eq!(compaction_sources(&[3], &[0, 1, 2], 3, 2), None);
    assert_eq!(scatter_sources(&[1], &[], 3, 1), None);
    assert_eq!(compaction_sources(&[1, 1], &[0, 0], usize::MAX, 1), None);
    assert_eq!(scatter_sources(&[1, 1], &[0, 0], 1, usize::MAX), None);
}

/// Group preservation requires a divisor group size and in-group indices.
#[test]
fn group_preservation_is_exact() {
    assert!(groups_preserved(&[1, 0, 3, 3], 2));
    assert!(groups_preserved(&[], 3));
    assert!(!groups_preserved(&[2, 0, 3, 3], 2));
    assert!(!groups_preserved(&[0, 1, 2], 2));
    assert!(!groups_preserved(&[0, 1], 0));
    assert!(groups_preserved(&[3, 3, 0, 1], 4));
}

proptest! {
    /// Packing then unpacking returns every valid token to its column; padding reads in-row slots.
    #[test]
    fn packing_round_trips_every_valid_token(
        mask in proptest::collection::vec(proptest::collection::vec(any::<bool>(), 1..6), 1..5)
            .prop_map(|rows| {
                let width = rows[0].len();
                rows.into_iter()
                    .map(|row| (0..width).map(|column| row.get(column).copied().unwrap_or(false)).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            }),
    ) {
        let sequence = mask[0].len();
        let valid = columns(&mask);
        let rows = BatchedRows::new(&valid, sequence).unwrap();
        let block = rows.block;
        let prefix = valid
            .iter()
            .all(|row| row.iter().enumerate().all(|(slot, &column)| slot == column));
        prop_assert_eq!(rows.packing == Packing::Prefix, prefix);
        let flat: Vec<usize> = valid.iter().flatten().copied().collect();
        let compact = compaction_sources(&rows.counts, &flat, sequence, block).unwrap();
        let scatter = scatter_sources(&rows.counts, &flat, sequence, block).unwrap();
        prop_assert_eq!(compact.len(), mask.len() * block);
        prop_assert_eq!(scatter.len(), mask.len() * sequence);
        for (row, row_columns) in valid.iter().enumerate() {
            for (slot, &column) in row_columns.iter().enumerate() {
                // Slot (row, slot) holds column `column`, and that column reads it back.
                prop_assert_eq!(compact[row * block + slot], row * sequence + column);
                prop_assert_eq!(scatter[row * sequence + column], row * block + slot);
            }
            for slot in row_columns.len()..block {
                prop_assert_eq!(compact[row * block + slot] / sequence, row);
            }
            for column in 0..sequence {
                if block > 0 {
                    prop_assert_eq!(scatter[row * sequence + column] / block, row);
                }
            }
        }
    }

    /// Group preservation agrees with its definition for arbitrary index vectors.
    #[test]
    fn group_preservation_property(
        indices in proptest::collection::vec(0_usize..8, 0..8),
        group_size in 0_usize..5,
    ) {
        let expected = group_size > 0
            && indices.len() % group_size == 0
            && indices
                .iter()
                .enumerate()
                .all(|(row, &source)| source / group_size == row / group_size);
        prop_assert_eq!(groups_preserved(&indices, group_size), expected);
    }
}

/// Row positions follow their formulas for new, padding, and inactive slots.
#[test]
fn row_positions_plan_causal_visibility_and_bias_positions() {
    // Length 4, two new tokens in a block of three: the padding slot reuses
    // the last new position 5 for visibility and continues the bias at 6.
    assert_eq!(row_positions(4, 2, 3), Some((vec![4, 5, 5], vec![4, 5, 6])));
    // An inactive row sees only its committed history.
    assert_eq!(row_positions(4, 0, 2), Some((vec![3, 3], vec![4, 5])));
    // An inactive empty row sees nothing; its outputs are discarded.
    assert_eq!(row_positions(0, 0, 1), Some((vec![-1], vec![0])));
    assert_eq!(row_positions(7, 0, 0), Some((Vec::new(), Vec::new())));
    // Malformed counts and positions beyond i64 or usize produce no plan.
    assert_eq!(row_positions(0, 3, 2), None);
    assert_eq!(row_positions(i64::MAX as usize, 1, 1), None);
    assert_eq!(row_positions(usize::MAX, 0, 1), None);
    assert!(row_positions(i64::MAX as usize - 2, 1, 2).is_some());
}
