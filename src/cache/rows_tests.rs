//! Row-run grouping against hand-picked batches and a naive per-row oracle.

use proptest::prelude::*;

use super::*;

/// Builds a run from its five fields.
const fn run(
    first_row: usize,
    end_row: usize,
    start: usize,
    column: usize,
    count: usize,
) -> RowRun {
    RowRun {
        first_row,
        end_row,
        start,
        column,
        count,
    }
}

/// Uniform batches collapse to one run; ragged and inactive rows split runs.
#[test]
fn runs_group_uniform_rows_and_skip_inactive_rows() {
    assert_eq!(
        row_runs(&[4, 4, 4], &[0, 0, 0], &[1, 1, 1]),
        [run(0, 3, 4, 0, 1)]
    );
    assert_eq!(
        row_runs(&[4, 4, 4, 9], &[0, 0, 0, 0], &[1, 0, 1, 1]),
        [run(0, 1, 4, 0, 1), run(2, 3, 4, 0, 1), run(3, 4, 9, 0, 1)]
    );
    // Same start and count but different input columns cannot share a slice.
    assert_eq!(
        row_runs(&[0, 0], &[1, 0], &[2, 2]),
        [run(0, 1, 0, 1, 2), run(1, 2, 0, 0, 2)]
    );
    assert_eq!(row_runs(&[1, 2], &[0, 0], &[0, 0]), []);
    assert_eq!(row_runs(&[], &[], &[]), []);
}

proptest! {
    /// Expanding the runs reproduces exactly the active rows and their parameters.
    #[test]
    fn runs_partition_active_rows(
        rows in prop::collection::vec((0_usize..3, 0_usize..2, 0_usize..3), 0..12),
    ) {
        let starts: Vec<usize> = rows.iter().map(|row| row.0).collect();
        let columns: Vec<usize> = rows.iter().map(|row| row.1).collect();
        let counts: Vec<usize> = rows.iter().map(|row| row.2).collect();
        let runs = row_runs(&starts, &columns, &counts);
        let mut expanded = Vec::new();
        for (index, run) in runs.iter().enumerate() {
            prop_assert!(run.first_row < run.end_row && run.count > 0);
            if let Some(next) = runs.get(index + 1) {
                prop_assert!(run.end_row <= next.first_row);
                // Maximality: adjacent runs differ or are separated.
                prop_assert!(
                    run.end_row < next.first_row
                        || (run.start, run.column, run.count) != (next.start, next.column, next.count)
                );
            }
            for row in run.first_row..run.end_row {
                expanded.push((row, run.start, run.column, run.count));
            }
        }
        let active: Vec<_> = (0..rows.len())
            .filter(|&row| counts[row] > 0)
            .map(|row| (row, starts[row], columns[row], counts[row]))
            .collect();
        prop_assert_eq!(expanded, active);
    }
}
