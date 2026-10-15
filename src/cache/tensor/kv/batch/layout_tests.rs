//! Batched write layouts and staging plans against hand-picked rows and naive oracles.

use proptest::prelude::*;

use super::*;

/// Layout validation checks vector lengths, then each row's block, then its position.
#[test]
fn append_layout_reports_first_violation_in_order() {
    assert_eq!(validate_append_layout(&[0, 5], &[0, 1], &[2, 3], 4), Ok(()));
    assert_eq!(
        validate_append_layout(&[0], &[0, 0], &[1, 1], 4),
        Err(AppendViolation::CountMismatch {
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        validate_append_layout(&[0, 0], &[0], &[1, 1], 4),
        Err(AppendViolation::CountMismatch {
            expected: 2,
            actual: 1
        })
    );
    // Row 0 overflows the position space but row 1's block check is never reached.
    assert_eq!(
        validate_append_layout(&[usize::MAX, 0], &[0, 4], &[1, 1], 4),
        Err(AppendViolation::PositionOverflow {
            start: usize::MAX,
            tokens: 1
        })
    );
    // Within one row the block is checked before the position.
    assert_eq!(
        validate_append_layout(&[usize::MAX], &[usize::MAX], &[1], 4),
        Err(AppendViolation::OutsideBlock {
            sequence: 0,
            column: usize::MAX,
            count: 1
        })
    );
}

/// Left padding places each row at `block - n_b` and rejects rows longer than the block.
#[test]
fn left_padding_aligns_rows_to_the_last_column() {
    assert_eq!(left_padded_columns(&[1, 3, 0], 3), Ok(vec![2, 0, 3]));
    assert_eq!(left_padded_columns(&[1, 4, 5], 3), Err(1));
    assert_eq!(left_padded_columns(&[], 0), Ok(Vec::new()));
}

/// Ends are exact sums over the zipped rows.
#[test]
fn append_ends_are_exact_sums() {
    assert_eq!(append_ends(&[3, 0, 7], &[1, 0, 2]), [4, 0, 9]);
    assert_eq!(append_ends(&[3, 4], &[1]), [4]);
}

/// Staging updates list active rows only and enforce every cursor and capacity.
#[test]
fn stage_updates_check_cursor_staging_and_capacity() {
    let staged = [1, 0, 2];
    let baseline = [4, 6, 8];
    assert_eq!(
        plan_stage_updates(&[5, 9, 10], &[2, 0, 1], &staged, Some(&baseline), 4, 16),
        Ok(vec![(0, 3), (2, 3)])
    );
    assert_eq!(
        plan_stage_updates(&[5, 6, 11], &[1, 1, 1], &staged, Some(&baseline), 4, 16),
        Err(StageViolation::CursorMismatch {
            sequence: 2,
            expected: 10,
            actual: 11
        })
    );
    assert_eq!(
        plan_stage_updates(&[5, 6, 10], &[4, 1, 1], &staged, Some(&baseline), 4, 16),
        Err(StageViolation::StagingCapacityExceeded { requested: 5 })
    );
    assert_eq!(
        plan_stage_updates(&[15, 6, 10], &[2, 0, 0], &staged, None, 4, 16),
        Err(StageViolation::WriteEnd {
            start: 15,
            tokens: 2,
            violation: WriteEndViolation::ExceedsCapacity { end: 17 }
        })
    );
    // Without a baseline any start is accepted; inactive rows are never checked.
    assert_eq!(
        plan_stage_updates(&[99, usize::MAX], &[1, 0], &[0, 9], None, 4, 128),
        Ok(vec![(0, 1)])
    );
    // A saturated cursor is reported rather than overflowing.
    assert_eq!(
        plan_stage_updates(&[0], &[1], &[1], Some(&[usize::MAX]), 4, 16),
        Err(StageViolation::CursorMismatch {
            sequence: 0,
            expected: usize::MAX,
            actual: 0
        })
    );
}

/// Linear staging runs group equal rows and shift inputs by the source offset.
#[test]
fn linear_runs_group_rows_and_shift_sources() {
    let runs = linear_stage_runs(&[2, 2, 2, 5], &[0, 0, 0, 0], &[1, 1, 0, 2], 0);
    assert_eq!(
        runs,
        [
            LinearRun {
                rows: 0..2,
                source_rows: 0..2,
                input: 0..1,
                destination: 2..3,
            },
            LinearRun {
                rows: 3..4,
                source_rows: 3..4,
                input: 0..2,
                destination: 5..7,
            },
        ]
    );
    // A single-row input feeds destination row 2 from input row 0.
    let single = linear_stage_runs(&[0, 0, 3], &[0, 0, 0], &[0, 0, 2], 2);
    assert_eq!(
        single,
        [LinearRun {
            rows: 2..3,
            source_rows: 0..1,
            input: 0..2,
            destination: 3..5,
        }]
    );
    assert!(linear_stage_runs(&[1, 1], &[0, 0], &[0, 0], 0).is_empty());
}

/// Token tables mark each row's own columns and number them from its start.
#[test]
fn token_tables_mark_each_rows_columns() {
    let (positions, valid) = token_segment_table(&[10, 4], &[1, 0], &[2, 1], 3);
    assert_eq!(positions, [0, 10, 11, 4, 0, 0]);
    assert_eq!(valid, [false, true, true, true, false, false]);
    let (positions, valid) = token_segment_table(&[], &[], &[], 3);
    assert!(positions.is_empty() && valid.is_empty());
}

proptest! {
    /// Staging updates agree with a naive per-row scan.
    #[test]
    fn stage_updates_match_naive_scan(
        rows in prop::collection::vec((0_usize..6, 0_usize..3, 0_usize..3, 0_usize..4), 0..5),
        with_baseline in any::<bool>(),
        staging_capacity in 0_usize..5,
        capacity in 0_usize..10,
    ) {
        let starts: Vec<usize> = rows.iter().map(|row| row.0).collect();
        let counts: Vec<usize> = rows.iter().map(|row| row.1).collect();
        let staged: Vec<usize> = rows.iter().map(|row| row.2).collect();
        let baseline: Vec<usize> = rows.iter().map(|row| row.3).collect();
        let base = with_baseline.then_some(baseline.as_slice());
        let mut expected = Ok(Vec::new());
        for row in 0..rows.len() {
            if counts[row] == 0 {
                continue;
            }
            if let Some(base) = base
                && starts[row] != base[row] + staged[row]
            {
                expected = Err(StageViolation::CursorMismatch {
                    sequence: row,
                    expected: base[row] + staged[row],
                    actual: starts[row],
                });
                break;
            }
            let end = staged[row] + counts[row];
            if end > staging_capacity {
                expected = Err(StageViolation::StagingCapacityExceeded { requested: end });
                break;
            }
            if starts[row] + counts[row] > capacity {
                expected = Err(StageViolation::WriteEnd {
                    start: starts[row],
                    tokens: counts[row],
                    violation: WriteEndViolation::ExceedsCapacity {
                        end: starts[row] + counts[row],
                    },
                });
                break;
            }
            if let Ok(updates) = &mut expected {
                updates.push((row, end));
            }
        }
        prop_assert_eq!(
            plan_stage_updates(&starts, &counts, &staged, base, staging_capacity, capacity),
            expected
        );
    }

    /// Token tables agree with membership in each row's column range.
    #[test]
    fn token_tables_match_column_ranges(
        rows in prop::collection::vec((0_usize..20, 0_usize..4, 0_usize..4), 0..4),
        block in 0_usize..5,
    ) {
        let starts: Vec<usize> = rows.iter().map(|row| row.0).collect();
        let columns: Vec<usize> = rows.iter().map(|row| row.1.min(block)).collect();
        let counts: Vec<usize> = rows
            .iter()
            .zip(&columns)
            .map(|(row, &column)| row.2.min(block - column))
            .collect();
        let (positions, valid) = token_segment_table(&starts, &columns, &counts, block);
        prop_assert_eq!(positions.len(), rows.len() * block);
        for row in 0..rows.len() {
            for slot in 0..block {
                let inside = (columns[row]..columns[row] + counts[row]).contains(&slot);
                prop_assert_eq!(valid[row * block + slot], inside);
                let expected = if inside { starts[row] + slot - columns[row] } else { 0 };
                prop_assert_eq!(positions[row * block + slot], expected);
            }
        }
    }
}
