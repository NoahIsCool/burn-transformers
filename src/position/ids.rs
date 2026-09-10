//! Overflow-safe absolute positions from padded token-validity masks.
//!
//! For valid token `i` in row `b`, `position[b, i] = start[b] + valid_before[b, i]`.
//! Invalid tokens receive a caller-selected sentinel and do not advance the count.
//! The row-major mask has axes `[batch, token]`; each row has its own cache offset.
//! All arithmetic is integral and checked before conversion to backend tensor IDs,
//! so left padding and ragged cached prefixes do not change valid-token phases.

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
    /// Formats invalid position geometry or absolute-index overflow with row/token context.
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
#[path = "ids_tests.rs"]
mod tests;
