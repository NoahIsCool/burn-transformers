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
    let rows = plan_rows(
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
    let rows = plan_rows(
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
    let rows = plan_rows(&[Vec::new(), Vec::new()], 2).unwrap();
    assert!(rows.is_empty());
    assert_eq!(rows.packing, Packing::Prefix);
    assert_eq!(scatter_sources(&[0, 0], &[], 2, 0), Some(vec![0; 4]));
}

/// Unsorted, duplicated, missing, or out-of-range columns produce no plan.
#[test]
fn malformed_columns_are_rejected() {
    assert_eq!(plan_rows(&[vec![2, 1]], 3), None);
    assert_eq!(plan_rows(&[vec![1, 1]], 3), None);
    assert_eq!(plan_rows(&[vec![0, 3]], 3), None);
    assert_eq!(plan_rows(&[vec![0, 1, 2, 3]], 3), None);
    assert_eq!(compaction_sources(&[2], &[1], 3, 2), None);
    assert_eq!(compaction_sources(&[3], &[0, 1, 2], 3, 2), None);
    assert_eq!(scatter_sources(&[1], &[], 3, 1), None);
    assert_eq!(compaction_sources(&[1, 1], &[0, 0], usize::MAX, 1), None);
    assert_eq!(scatter_sources(&[1, 1], &[0, 0], 1, usize::MAX), None);
}

/// An unrepresentable `B * S` refuses only plans that need flat gather indices.
#[test]
fn oversized_rectangles_refuse_only_gather_plans() {
    let prefix = plan_rows(&[vec![0], Vec::new()], usize::MAX).unwrap();
    assert_eq!(prefix.packing, Packing::Prefix);
    assert_eq!(prefix.block, 1);
    assert_eq!(plan_rows(&[vec![1], Vec::new()], usize::MAX), None);
}

/// Plan queries report padding exactly for slots past a row's count.
#[test]
fn plan_queries_mark_padding_past_each_count() {
    let rows = plan_rows(&[vec![0, 2], vec![1]], 3).unwrap();
    assert_eq!(rows.padded_slots(), [false, false, false, true]);
    assert!(rows.has_padded_slots());
    assert!(!rows.is_empty());
    let full = plan_rows(&[vec![0, 1], vec![0, 1]], 3).unwrap();
    assert_eq!(full.padded_slots(), [false; 4]);
    assert!(!full.has_padded_slots());
}

/// A padded slot needs an in-row column to read, so a zero-width input refuses a block.
#[test]
fn compaction_without_any_column_is_refused() {
    assert_eq!(compaction_sources(&[0], &[], 0, 1), None);
    assert_eq!(compaction_sources(&[0], &[], 0, 0), Some(Vec::new()));
    assert_eq!(compaction_sources(&[], &[], 0, 3), Some(Vec::new()));
}

/// Padded slots are exactly the slots at or beyond a row's count.
#[test]
fn padded_slots_follow_row_counts() {
    assert_eq!(
        padded_slots(&[2, 0, 3], 3),
        [false, false, true, true, true, true, false, false, false]
    );
    assert!(padded_slots(&[0, 0], 0).is_empty());
    assert!(padded_slots(&[], 4).is_empty());
    assert!(has_padded_slot(&[2, 0, 3], 3));
    assert!(!has_padded_slot(&[3, 3], 3));
    assert!(!has_padded_slot(&[], 3));
    assert!(!has_padded_slot(&[0, 0], 0));
}

/// The iterator formulation that [`plan_rows`] replaced, kept as a behavioral reference.
fn reference_rows(valid_columns: &[Vec<usize>], sequence: usize) -> Option<BatchedRows> {
    let batch = valid_columns.len();
    let counts: Vec<usize> = valid_columns.iter().map(Vec::len).collect();
    let block = counts.iter().copied().max().unwrap_or(0);
    if block > sequence {
        return None;
    }
    let prefix = valid_columns.iter().all(|columns| {
        columns
            .iter()
            .enumerate()
            .all(|(slot, &column)| slot == column)
    });
    let packing = if prefix {
        Packing::Prefix
    } else {
        let flat: Vec<usize> = valid_columns.iter().flatten().copied().collect();
        Packing::Gather {
            compact: compaction_sources(&counts, &flat, sequence, block)?,
            scatter: scatter_sources(&counts, &flat, sequence, block)?,
        }
    };
    Some(BatchedRows {
        batch,
        sequence,
        block,
        counts,
        packing,
    })
}

/// The iterator formulation of the step geometry that [`step_positions`] replaced.
fn reference_step(lengths: &[usize], counts: &[usize], block: usize) -> Option<StepPositions> {
    let mut mask_positions = Vec::new();
    let mut bias_positions = Vec::new();
    let mut shared_length = None;
    let mut uniform = true;
    let mut max_position = 0_usize;
    for (&length, &count) in lengths.iter().zip(counts) {
        if count > 0 {
            match shared_length {
                None => shared_length = Some(length),
                Some(shared) if shared != length => uniform = false,
                Some(_) => {}
            }
        }
        let (visible, query) = row_positions(length, count, block)?;
        mask_positions.extend(visible);
        bias_positions.extend(query);
        if let Some(last) = block.checked_sub(1) {
            max_position = max_position.max(length + last);
        }
    }
    Some(StepPositions {
        mask_positions,
        bias_positions,
        shared_length: shared_length.filter(|_| uniform),
        max_position,
    })
}

/// Step geometry shares a length only when every active row agrees on it.
#[test]
fn step_positions_share_only_agreeing_active_lengths() {
    let plan = step_positions(&[4, 2, 4], &[2, 0, 1], 2).unwrap();
    assert_eq!(plan.mask_positions, [4, 5, 1, 1, 4, 4]);
    assert_eq!(plan.bias_positions, [4, 5, 2, 3, 4, 5]);
    assert_eq!(plan.shared_length, Some(4));
    assert_eq!(plan.max_position, 5);
    assert_eq!(
        step_positions(&[3, 1], &[1, 1], 1).unwrap().shared_length,
        None
    );
    assert_eq!(
        step_positions(&[3, 1], &[0, 0], 1).unwrap().shared_length,
        None
    );
    assert_eq!(step_positions(&[3, 1], &[0, 0], 0).unwrap().max_position, 0);
    assert_eq!(step_positions(&[], &[], 2).unwrap().max_position, 0);
    assert_eq!(step_positions(&[0], &[2], 1), None);
    assert_eq!(step_positions(&[usize::MAX], &[0], 1), None);
}

/// The displacement bound covers the last key and the largest bias position.
#[test]
fn displacement_bound_covers_keys_and_queries() {
    assert_eq!(displacement_bound(5, 3), 5);
    assert_eq!(displacement_bound(1, 7), 6);
    assert_eq!(displacement_bound(0, 0), 0);
    assert_eq!(displacement_bound(4, 5), 4);
}

/// The first reorder index outside the batch is reported by value.
#[test]
fn first_out_of_range_reports_the_first_offender() {
    assert_eq!(first_out_of_range(&[0, 2, 5, 9], 3), Some(5));
    assert_eq!(first_out_of_range(&[0, 2, 1], 3), None);
    assert_eq!(first_out_of_range(&[], 0), None);
    assert_eq!(first_out_of_range(&[0], 0), Some(0));
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
        let rows = plan_rows(&valid, sequence).unwrap();
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
                if let Some(quotient) = scatter[row * sequence + column].checked_div(block) {
                    prop_assert_eq!(quotient, row);
                }
            }
        }
    }

    /// The verified planner equals the former iterator implementation on arbitrary rows.
    ///
    /// Rows are drawn without any ordering guarantee, so malformed (unsorted,
    /// duplicated, out-of-range, or too long) rows are exercised alongside
    /// valid ones.
    #[test]
    fn plan_rows_matches_the_iterator_reference(
        valid in proptest::collection::vec(proptest::collection::vec(0_usize..6, 0..5), 0..5),
        sequence in 0_usize..6,
    ) {
        prop_assert_eq!(plan_rows(&valid, sequence), reference_rows(&valid, sequence));
    }

    /// The verified step geometry equals the former iterator implementation.
    #[test]
    fn step_positions_match_the_iterator_reference(
        lengths in proptest::collection::vec(0_usize..9, 0..5),
        counts in proptest::collection::vec(0_usize..4, 0..5),
        block in 0_usize..4,
    ) {
        prop_assert_eq!(
            step_positions(&lengths, &counts, block),
            reference_step(&lengths, &counts, block)
        );
    }

    /// The reorder check agrees with a linear search for the first offender.
    #[test]
    fn first_out_of_range_matches_a_search(
        indices in proptest::collection::vec(0_usize..6, 0..6),
        batch in 0_usize..6,
    ) {
        prop_assert_eq!(
            first_out_of_range(&indices, batch),
            indices.iter().copied().find(|&index| index >= batch)
        );
    }

    /// Every displacement between a key slot and a bias position is within the bound.
    #[test]
    fn displacement_bound_property(max_position in 0_usize..20, key_slots in 0_usize..20) {
        let bound = displacement_bound(max_position, key_slots);
        for key in 0..key_slots {
            for query in 0..=max_position {
                prop_assert!(key.abs_diff(query) <= bound);
            }
        }
        prop_assert_eq!(bound, max_position.max(key_slots.saturating_sub(1)));
    }

    /// Padded slots and the padding test agree with their definitions.
    #[test]
    fn padded_slots_property(
        counts in proptest::collection::vec(0_usize..5, 0..5),
        block in 0_usize..5,
    ) {
        let padded = padded_slots(&counts, block);
        prop_assert_eq!(padded.len(), counts.len() * block);
        for (row, &count) in counts.iter().enumerate() {
            for slot in 0..block {
                prop_assert_eq!(padded[row * block + slot], slot >= count);
            }
        }
        prop_assert_eq!(
            has_padded_slot(&counts, block),
            counts.iter().any(|&count| count < block)
        );
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
