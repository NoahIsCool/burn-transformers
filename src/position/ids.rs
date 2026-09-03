use core::fmt;

/// Derives numeric positions for one token-validity row.
///
/// A `true` mask element receives the next contiguous position beginning at
/// `start_position`; a `false` element receives `masked_position` and does not
/// advance the counter. Formally, for a valid token `i`,
/// `position[i] = start + sum(mask[0..=i]) - 1`.
///
/// # Errors
///
/// Returns [`PositionIdError::PositionOverflow`] if the contiguous valid-token
/// index cannot be represented by `usize`.
pub fn derive_position_ids(
    valid_mask: &[bool],
    start_position: usize,
    masked_position: usize,
) -> Result<Vec<usize>, PositionIdError> {
    derive_batched_position_ids(
        valid_mask,
        1,
        valid_mask.len(),
        &[start_position],
        masked_position,
    )
}

/// Derives row-major numeric positions from a batched token-validity mask.
///
/// Each sequence has an independent starting position. This supports ragged
/// decode batches whose caches contain different prefix lengths. `valid_mask`
/// must contain exactly `batch_size * sequence_length` row-major elements and
/// `start_positions` must contain one offset per sequence.
///
/// Padding tokens receive `masked_position`; valid tokens are numbered
/// contiguously even when padding occurs on the left, right, or between tokens.
///
/// # Errors
///
/// Returns an error for shape multiplication overflow, mismatched flat storage,
/// a wrong number of row offsets, or absolute position overflow.
pub fn derive_batched_position_ids(
    valid_mask: &[bool],
    batch_size: usize,
    sequence_length: usize,
    start_positions: &[usize],
    masked_position: usize,
) -> Result<Vec<usize>, PositionIdError> {
    let expected =
        batch_size
            .checked_mul(sequence_length)
            .ok_or(PositionIdError::ShapeOverflow {
                batch_size,
                sequence_length,
            })?;
    if valid_mask.len() != expected {
        return Err(PositionIdError::MaskLengthMismatch {
            expected,
            actual: valid_mask.len(),
        });
    }
    if start_positions.len() != batch_size {
        return Err(PositionIdError::StartCountMismatch {
            expected: batch_size,
            actual: start_positions.len(),
        });
    }

    let mut positions = Vec::with_capacity(expected);
    for (sequence, (&start, row)) in start_positions
        .iter()
        .zip(valid_mask.chunks_exact(sequence_length.max(1)))
        .enumerate()
    {
        // `chunks_exact(1)` is only used when sequence_length is zero, and the
        // expected flat mask is empty, so this loop correctly executes no rows.
        let mut valid_before = 0usize;
        for (token, &valid) in row.iter().take(sequence_length).enumerate() {
            if valid {
                let position =
                    start
                        .checked_add(valid_before)
                        .ok_or(PositionIdError::PositionOverflow {
                            sequence,
                            token,
                            start_position: start,
                            valid_tokens_before: valid_before,
                        })?;
                positions.push(position);
                valid_before += 1;
            } else {
                positions.push(masked_position);
            }
        }
    }

    // Empty rows cannot be recovered from slice chunking. They contribute no
    // positions, and their offsets were already shape-checked above.
    Ok(positions)
}

/// Invalid shape or arithmetic while deriving absolute position IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionIdError {
    /// `batch_size * sequence_length` overflowed `usize`.
    ShapeOverflow {
        /// Number of rows.
        batch_size: usize,
        /// Tokens per row.
        sequence_length: usize,
    },
    /// Flat mask storage did not match its declared shape.
    MaskLengthMismatch {
        /// Required number of mask elements.
        expected: usize,
        /// Supplied number of mask elements.
        actual: usize,
    },
    /// The number of starting offsets differed from the batch size.
    StartCountMismatch {
        /// Required number of offsets.
        expected: usize,
        /// Supplied number of offsets.
        actual: usize,
    },
    /// Adding the valid-token ordinal to a starting offset overflowed.
    PositionOverflow {
        /// Affected sequence index.
        sequence: usize,
        /// Affected token index within the sequence.
        token: usize,
        /// Initial absolute position for the sequence.
        start_position: usize,
        /// Number of preceding valid tokens.
        valid_tokens_before: usize,
    },
}

impl fmt::Display for PositionIdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow {
                batch_size,
                sequence_length,
            } => write!(
                formatter,
                "position-mask shape [{batch_size}, {sequence_length}] overflows usize"
            ),
            Self::MaskLengthMismatch { expected, actual } => write!(
                formatter,
                "position mask requires {expected} elements, received {actual}"
            ),
            Self::StartCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} position starts, received {actual}"
            ),
            Self::PositionOverflow {
                sequence,
                token,
                start_position,
                valid_tokens_before,
            } => write!(
                formatter,
                "position for sequence {sequence}, token {token} overflows: {start_position} + {valid_tokens_before}"
            ),
        }
    }
}

impl std::error::Error for PositionIdError {}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn derives_ragged_batch_with_independent_cache_offsets() {
        let mask = [true, true, false, false, true, false, true, true];
        assert_eq!(
            derive_batched_position_ids(&mask, 2, 4, &[10, 3], 0),
            Ok(vec![10, 11, 0, 0, 3, 0, 4, 5])
        );
    }

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
}
