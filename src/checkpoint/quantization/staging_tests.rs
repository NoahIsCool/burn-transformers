//! Boundary checks of Burnpack staging admission.

use proptest::prelude::*;

use super::*;

/// Bounded sums admit the limit itself and refuse one more.
#[test]
fn bounded_sums_admit_exactly_up_to_the_limit() {
    assert_eq!(bounded_add(3, 4, 7), Ok(7));
    assert_eq!(
        bounded_add(3, 5, 7),
        Err(BoundBreach::Exceeded { attempted: 8 })
    );
    assert_eq!(
        bounded_add(u64::MAX, 1, u64::MAX),
        Err(BoundBreach::Overflow)
    );
    assert_eq!(bounded_add(u64::MAX, 0, u64::MAX), Ok(u64::MAX));
}

/// Tensor admission checks the count, then the payload, then the staged total.
#[test]
fn tensor_admission_names_the_first_breach() {
    assert_eq!(
        admit_tensor(1, 10, 5, 2, 8, 15),
        Ok(AdmittedTensor {
            count: 2,
            staged: 15
        })
    );
    assert_eq!(
        admit_tensor(2, 10, 50, 2, 8, 15),
        Err(TensorBreach::TooManyTensors { attempted: 3 })
    );
    assert_eq!(
        admit_tensor(1, 10, 9, 2, 8, 15),
        Err(TensorBreach::PayloadTooLarge)
    );
    assert_eq!(
        admit_tensor(1, 10, 6, 2, 8, 15),
        Err(TensorBreach::Staged(BoundBreach::Exceeded {
            attempted: 16
        }))
    );
    assert_eq!(
        admit_tensor(1, u64::MAX, 1, 2, 8, u64::MAX),
        Err(TensorBreach::Staged(BoundBreach::Overflow))
    );
    assert_eq!(
        admit_tensor(usize::MAX, 0, 0, usize::MAX, 0, 0),
        Err(TensorBreach::CountOverflow)
    );
}

proptest! {
    /// Admission matches the original checked-add-then-compare sequence.
    #[test]
    fn admission_matches_checked_arithmetic(
        count in 0_usize..8,
        staged in any::<u64>(),
        payload in any::<u64>(),
        max_tensors in 0_usize..8,
        max_payload in any::<u64>(),
        max_staged in any::<u64>(),
    ) {
        let expected = if count + 1 > max_tensors {
            Err(TensorBreach::TooManyTensors { attempted: count + 1 })
        } else if payload > max_payload {
            Err(TensorBreach::PayloadTooLarge)
        } else {
            match staged.checked_add(payload) {
                None => Err(TensorBreach::Staged(BoundBreach::Overflow)),
                Some(sum) if sum > max_staged => {
                    Err(TensorBreach::Staged(BoundBreach::Exceeded { attempted: sum }))
                }
                Some(sum) => Ok(AdmittedTensor { count: count + 1, staged: sum }),
            }
        };
        prop_assert_eq!(
            admit_tensor(count, staged, payload, max_tensors, max_payload, max_staged),
            expected
        );
    }

    /// The limit scan names the first zero or over-cap limit, like a direct check.
    #[test]
    fn limit_scan_names_the_first_violation(
        limits in proptest::collection::vec((0_u64..4, 0_u64..4), 0..6),
    ) {
        let values = limits.iter().map(|limit| limit.0).collect::<Vec<_>>();
        let caps = limits.iter().map(|limit| limit.1).collect::<Vec<_>>();
        let expected = values.iter().zip(&caps).enumerate().find_map(|(index, (&value, &cap))| {
            if value == 0 {
                Some((index, LimitViolation::Zero))
            } else if value > cap {
                Some((index, LimitViolation::AboveCap))
            } else {
                None
            }
        });
        prop_assert_eq!(first_limit_violation(&values, &caps), expected);
    }
}
