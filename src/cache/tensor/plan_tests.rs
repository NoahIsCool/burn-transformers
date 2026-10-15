//! Shared tensor-cache planning against hand-picked boundaries and naive oracles.

use proptest::prelude::*;

use super::*;

/// Shapes report exact products, prefix overflow, the portable limit, and byte overflow.
#[test]
fn tensor_elements_reports_each_violation_in_order() {
    assert_eq!(tensor_elements(&[], 4), Ok(1));
    assert_eq!(tensor_elements(&[3, 0], 4), Ok(0));
    assert_eq!(tensor_elements(&[2, 3, 7], 2), Ok(42));
    // A running product overflows before the trailing zero is reached.
    assert_eq!(
        tensor_elements(&[usize::MAX, 2, 0], 1),
        Err(AllocationViolation::ShapeOverflow)
    );
    assert_eq!(
        tensor_elements(&[PORTABLE_TENSOR_ELEMENT_LIMIT + 1], 1),
        Err(AllocationViolation::ElementLimitExceeded {
            elements: PORTABLE_TENSOR_ELEMENT_LIMIT + 1,
        })
    );
    assert_eq!(
        tensor_elements(&[PORTABLE_TENSOR_ELEMENT_LIMIT], usize::MAX),
        Err(AllocationViolation::ByteOverflow {
            elements: PORTABLE_TENSOR_ELEMENT_LIMIT,
        })
    );
    assert_eq!(
        tensor_elements(&[PORTABLE_TENSOR_ELEMENT_LIMIT], 8),
        Ok(PORTABLE_TENSOR_ELEMENT_LIMIT)
    );
}

/// Repeated allocations distinguish count overflow from byte overflow.
#[test]
fn repeated_elements_distinguishes_count_and_byte_overflow() {
    assert_eq!(repeated_elements(34, 2, 8), Ok(68));
    assert_eq!(repeated_elements(34, 0, 8), Ok(0));
    assert_eq!(
        repeated_elements(usize::MAX, 2, 1),
        Err(AllocationViolation::ShapeOverflow)
    );
    assert_eq!(
        repeated_elements(usize::MAX, 1, 2),
        Err(AllocationViolation::ByteOverflow {
            elements: usize::MAX,
        })
    );
}

/// Write ends are exact, and overflow is reported before the capacity check.
#[test]
fn write_end_checks_overflow_then_capacity() {
    assert_eq!(write_end(3, 2, 5), Ok(5));
    assert_eq!(write_end(0, 0, 0), Ok(0));
    assert_eq!(
        write_end(3, 3, 5),
        Err(WriteEndViolation::ExceedsCapacity { end: 6 })
    );
    assert_eq!(
        write_end(usize::MAX, 1, usize::MAX),
        Err(WriteEndViolation::Overflow)
    );
}

/// The staged-commit scan reports the first failing check in layer-then-row order.
#[test]
fn staged_commit_scan_reports_first_failure() {
    let proposed = [2, 1, 0];
    assert_eq!(
        check_staged_commit(&[vec![2, 1, 0], vec![2, 1, 0]], &proposed, &[2, 0, 0]),
        Ok(())
    );
    // Row 0 over-accepts before row 1's staging mismatch is reached.
    assert_eq!(
        check_staged_commit(&[vec![2, 3, 0]], &proposed, &[3, 0, 0]),
        Err(StagedCommitViolation::PrefixExceedsProposal {
            sequence: 0,
            accepted: 3,
            proposed: 2,
        })
    );
    // A staging mismatch at row 0 precedes row 0's over-acceptance.
    assert_eq!(
        check_staged_commit(&[vec![1, 1, 0]], &proposed, &[3, 0, 0]),
        Err(StagedCommitViolation::Incomplete {
            layer: 0,
            sequence: 0,
            expected: 2,
            actual: 1,
        })
    );
    // A later layer's mismatch is found only after layer zero passes.
    assert_eq!(
        check_staged_commit(&[vec![2, 1, 0], vec![2, 1, 4]], &proposed, &[1, 1, 0]),
        Err(StagedCommitViolation::Incomplete {
            layer: 1,
            sequence: 2,
            expected: 0,
            actual: 4,
        })
    );
    // An empty group stages nothing and constrains no prefix.
    assert_eq!(check_staged_commit(&[], &proposed, &[9, 9, 9]), Ok(()));
}

/// Truncation clears exactly the rows that held tokens and return to zero.
#[test]
fn truncate_clears_only_rows_reset_from_history() {
    assert_eq!(
        rows_cleared_by_truncate(&[5, 0, 3, 4], &[0, 0, 3, 1]),
        [true, false, false, false]
    );
    // A zipped scan stops at the shorter vector.
    assert_eq!(rows_cleared_by_truncate(&[5, 5], &[0]), [true]);
    assert!(rows_cleared_by_truncate(&[], &[0]).is_empty());
}

/// Index scans report the lowest out-of-range position.
#[test]
fn index_scans_report_first_out_of_range_position() {
    assert_eq!(first_index_out_of_range(&[0, 2, 1], 3), None);
    assert_eq!(first_index_out_of_range(&[0, 3, 4], 3), Some(1));
    assert_eq!(first_index_out_of_range(&[], 0), None);
    assert_eq!(first_index_out_of_range(&[0], 0), Some(0));
}

/// Portable conversion keeps values and names the first value beyond `i32::MAX`.
#[test]
fn portable_indices_preserve_values_and_report_overflow() {
    let limit = PORTABLE_TENSOR_ELEMENT_LIMIT;
    assert_eq!(portable_indices(&[0, 7, limit]), Ok(vec![0, 7, i32::MAX]));
    assert_eq!(portable_indices(&[1, limit + 1, usize::MAX]), Err(1));
    assert_eq!(portable_indices(&[]), Ok(Vec::new()));
}

proptest! {
    /// Shape counts agree with a checked left fold of the dimensions.
    #[test]
    fn tensor_elements_matches_checked_fold(
        dimensions in prop::collection::vec(
            prop_oneof![0_usize..5, Just(usize::MAX), Just(1 << 31), Just(1 << 16)],
            0..5,
        ),
        element_size in prop_oneof![1_usize..9, Just(usize::MAX)],
    ) {
        let product = dimensions
            .iter()
            .try_fold(1_usize, |product, &dimension| product.checked_mul(dimension));
        let expected = match product {
            None => Err(AllocationViolation::ShapeOverflow),
            Some(elements) if elements > PORTABLE_TENSOR_ELEMENT_LIMIT => {
                Err(AllocationViolation::ElementLimitExceeded { elements })
            }
            Some(elements) if elements.checked_mul(element_size).is_none() => {
                Err(AllocationViolation::ByteOverflow { elements })
            }
            Some(elements) => Ok(elements),
        };
        prop_assert_eq!(tensor_elements(&dimensions, element_size), expected);
    }

    /// The staged-commit scan agrees with a naive nested scan.
    #[test]
    fn staged_commit_matches_naive_scan(
        layers in 0_usize..3,
        rows in prop::collection::vec((0_usize..3, 0_usize..4), 0..4),
        staged_noise in prop::collection::vec(0_usize..3, 12),
    ) {
        let proposed: Vec<usize> = rows.iter().map(|row| row.0).collect();
        let accepted: Vec<usize> = rows.iter().map(|row| row.1).collect();
        let staged: Vec<Vec<usize>> = (0..layers)
            .map(|layer| {
                (0..rows.len())
                    .map(|row| {
                        // Mostly agree with the proposal; occasionally differ.
                        if staged_noise[(layer * 4 + row) % 12] == 0 {
                            proposed[row] + 1
                        } else {
                            proposed[row]
                        }
                    })
                    .collect()
            })
            .collect();
        let mut expected = Ok(());
        'scan: for (layer, layer_staged) in staged.iter().enumerate() {
            for row in 0..rows.len() {
                if layer_staged[row] != proposed[row] {
                    expected = Err(StagedCommitViolation::Incomplete {
                        layer,
                        sequence: row,
                        expected: proposed[row],
                        actual: layer_staged[row],
                    });
                    break 'scan;
                }
                if accepted[row] > proposed[row] {
                    expected = Err(StagedCommitViolation::PrefixExceedsProposal {
                        sequence: row,
                        accepted: accepted[row],
                        proposed: proposed[row],
                    });
                    break 'scan;
                }
            }
        }
        prop_assert_eq!(check_staged_commit(&staged, &proposed, &accepted), expected);
    }
}
