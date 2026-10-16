//! Step coverage and version bookkeeping against hand-picked write sets.

use proptest::prelude::*;

use super::*;

/// Single-row marks reject out-of-batch, inactive, and duplicate rows.
#[test]
fn row_marks_reject_invalid_and_duplicate_rows() {
    let counts = [2, 0, 1];
    let mut covered = vec![false; 3];
    assert_eq!(
        mark_covered(&mut covered, Some(3), &counts),
        Err(CoverageViolation::RowOutOfBounds { row: 3 })
    );
    assert_eq!(
        mark_covered(&mut covered, Some(1), &counts),
        Err(CoverageViolation::InactiveRow { row: 1 })
    );
    assert_eq!(covered, [false, false, false]);
    assert_eq!(mark_covered(&mut covered, Some(2), &counts), Ok(()));
    assert_eq!(covered, [false, false, true]);
    assert_eq!(
        mark_covered(&mut covered, Some(2), &counts),
        Err(CoverageViolation::Duplicate { row: 2 })
    );
}

/// Batched marks set every active row and name the first already-marked one.
#[test]
fn batch_marks_cover_active_rows_once() {
    let counts = [2, 0, 1, 3];
    let mut covered = vec![false; 4];
    assert_eq!(mark_covered(&mut covered, None, &counts), Ok(()));
    assert_eq!(covered, [true, false, true, true]);
    assert_eq!(first_uncovered_row(&covered, &counts), None);

    let mut partial = vec![false, false, true, false];
    assert_eq!(
        mark_covered(&mut partial, None, &counts),
        Err(CoverageViolation::Duplicate { row: 2 })
    );
    // The scan marked row 0 before reaching the duplicate.
    assert_eq!(partial, [true, false, true, false]);
    assert_eq!(first_uncovered_row(&partial, &counts), Some(3));
}

/// Row scans find the first active row and the set positions of a mask.
#[test]
fn row_scans_report_active_and_set_rows() {
    assert_eq!(first_active_row(&[0, 0, 4, 1]), 2);
    assert_eq!(first_active_row(&[0, 0]), 0);
    assert_eq!(first_active_row(&[]), 0);
    assert_eq!(set_positions(&[false, true, true, false, true]), [1, 2, 4]);
    assert!(set_positions(&[]).is_empty());
    assert!(all_set(&[true, true]));
    assert!(all_set(&[]));
    assert!(!all_set(&[true, false]));
}

/// Proposal checks report the position of the first row whose staging lags.
#[test]
fn proposal_checks_report_first_lagging_row() {
    let staged = [1, 2, 3];
    let proposals = [1, 5, 3];
    assert_eq!(first_unstaged_proposal(&[0, 2], &staged, &proposals), None);
    assert_eq!(
        first_unstaged_proposal(&[2, 1, 0], &staged, &proposals),
        Some(1)
    );
}

/// Bumping one bank changes only its version, and wraps at `u64::MAX`.
#[test]
fn bumps_change_exactly_one_bank() {
    let mut versions = initial_versions();
    assert_eq!(versions.current(StepBank::Main), 0);
    versions.bump(StepBank::Main);
    assert_eq!(versions.current(StepBank::Main), 1);
    assert_eq!(versions.current(StepBank::Mtp), 0);
    versions.bump(StepBank::Mtp);
    assert_eq!(versions.current(StepBank::Mtp), 1);
    let mut wrapped = BankVersions {
        main: u64::MAX,
        mtp: 7,
    };
    wrapped.bump(StepBank::Main);
    assert_eq!(wrapped.current(StepBank::Main), 0);
    assert_eq!(wrapped.current(StepBank::Mtp), 7);
}

proptest! {
    /// Any accepted sequence of marks that finishes covers each active row exactly once.
    #[test]
    fn accepted_mark_sequences_cover_active_rows_exactly_once(
        counts in prop::collection::vec(0_usize..3, 0..5),
        writes in prop::collection::vec(prop::option::of(0_usize..6), 0..8),
    ) {
        let mut covered = vec![false; counts.len()];
        let mut writes_per_row = vec![0_usize; counts.len()];
        let mut accepted = true;
        for &write in &writes {
            if mark_covered(&mut covered, write, &counts).is_err() {
                accepted = false;
                break;
            }
            match write {
                Some(row) => writes_per_row[row] += 1,
                None => {
                    for (row, &count) in counts.iter().enumerate() {
                        if count > 0 {
                            writes_per_row[row] += 1;
                        }
                    }
                }
            }
        }
        if accepted && first_uncovered_row(&covered, &counts).is_none() {
            for (row, &count) in counts.iter().enumerate() {
                prop_assert_eq!(writes_per_row[row], usize::from(count > 0));
            }
        }
    }
}
