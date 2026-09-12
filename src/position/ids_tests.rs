//! Checked absolute positions for ragged batches, padding, and integer boundaries.

use super::*;

/// Padding at either end or between tokens never advances the valid-position count.
#[test]
fn derives_unpadded_left_padded_right_padded_and_gapped_rows() {
    assert_eq!(
        derive_position_ids(&[true, true, true], 0, 0),
        Ok(vec![0, 1, 2])
    );
    assert_eq!(
        derive_position_ids(&[false, false, true, true], 5, 99),
        Ok(vec![99, 99, 5, 6])
    );
    assert_eq!(
        derive_position_ids(&[true, true, false, false], 2, 0),
        Ok(vec![2, 3, 0, 0])
    );
    assert_eq!(
        derive_position_ids(&[true, false, true, false, true], 7, 42),
        Ok(vec![7, 42, 8, 42, 9])
    );
}

/// Each ragged row starts at its own cached-prefix offset without cross-row leakage.
#[test]
fn derives_ragged_batch_with_independent_cache_offsets() {
    let mask = [true, true, false, false, true, false, true, true];
    assert_eq!(
        derive_batched_position_ids(&mask, 2, 4, &[10, 3], 0),
        Ok(vec![10, 11, 0, 0, 3, 0, 4, 5])
    );
}

/// Zero rows and zero columns both produce empty position storage.
#[test]
fn accepts_empty_shapes_without_inventing_positions() {
    assert_eq!(
        derive_batched_position_ids(&[], 0, 7, &[], 0),
        Ok(Vec::new())
    );
    assert_eq!(
        derive_batched_position_ids(&[], 3, 0, &[2, 4, 6], 0),
        Ok(Vec::new())
    );
}

/// Product overflow, flat mask length, and row-offset count report distinct errors.
#[test]
fn reports_every_shape_error() {
    assert_eq!(
        derive_batched_position_ids(&[], usize::MAX, 2, &[], 0),
        Err(PositionIdError::ShapeOverflow {
            batch_size: usize::MAX,
            sequence_length: 2,
        })
    );
    assert_eq!(
        derive_batched_position_ids(&[true], 1, 2, &[0], 0),
        Err(PositionIdError::MaskLengthMismatch {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        derive_batched_position_ids(&[true, true], 1, 2, &[], 0),
        Err(PositionIdError::StartCountMismatch {
            expected: 1,
            actual: 0,
        })
    );
}

/// The largest position is valid once; a subsequent real token reports its location.
#[test]
fn permits_max_position_once_and_rejects_the_next_valid_token() {
    assert_eq!(
        derive_position_ids(&[true, false], usize::MAX, 0),
        Ok(vec![usize::MAX, 0])
    );
    assert_eq!(
        derive_position_ids(&[true, false, true], usize::MAX, 0),
        Err(PositionIdError::PositionOverflow {
            sequence: 0,
            token: 2,
            start_position: usize::MAX,
            valid_tokens_before: 1,
        })
    );
}

/// Shape and overflow diagnostics identify the actual and expected input geometry.
#[test]
fn errors_describe_shape_and_token_context() {
    let messages = [
        PositionIdError::ShapeOverflow {
            batch_size: usize::MAX,
            sequence_length: 2,
        }
        .to_string(),
        PositionIdError::MaskLengthMismatch {
            expected: 4,
            actual: 3,
        }
        .to_string(),
        PositionIdError::StartCountMismatch {
            expected: 2,
            actual: 1,
        }
        .to_string(),
        PositionIdError::PositionOverflow {
            sequence: 1,
            token: 3,
            start_position: usize::MAX,
            valid_tokens_before: 1,
        }
        .to_string(),
    ];
    assert!(messages[0].contains("overflows usize"));
    assert_eq!(messages[1], "position mask requires 4 elements, received 3");
    assert_eq!(messages[2], "expected 2 position starts, received 1");
    assert!(messages[3].contains("sequence 1, token 3"));
}

/// Every short Boolean mask agrees with a simple contiguous valid-token counter.
#[test]
fn exhaustive_masks_preserve_contiguous_valid_positions() {
    for length in 0..=10 {
        for bits in 0usize..(1usize << length) {
            let mask = (0..length)
                .map(|index| bits & (1 << index) != 0)
                .collect::<Vec<_>>();
            let positions =
                derive_position_ids(&mask, 17, 999).expect("small positions cannot overflow");
            let mut next = 17;
            for (&valid, &position) in mask.iter().zip(&positions) {
                if valid {
                    assert_eq!(position, next);
                    next += 1;
                } else {
                    assert_eq!(position, 999);
                }
            }
        }
    }
}
