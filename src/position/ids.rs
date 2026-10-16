//! Overflow-safe absolute positions from padded token-validity masks.
//!
//! For valid token `i` in row `b`, `position[b, i] = start[b] + valid_before[b, i]`.
//! Invalid tokens receive a caller-selected sentinel and do not advance the count.
//! The row-major mask has axes `[batch, token]`; each row has its own cache offset.
//! All arithmetic is integral and checked before conversion to backend tensor IDs,
//! so left padding and ragged cached prefixes do not change valid-token phases.
//!
//! # Verified construction
//!
//! [`derive_batched_position_ids`] and [`derive_position_ids`] are checked in
//! place by Verus through `verus/position.rs`. For every input, the proof
//! establishes that no arithmetic overflows or panics, that a successful result
//! has exactly `batch * sequence` entries, and that entry `row * sequence +
//! token` equals `start[row] + valid_count(row, 0..token)` for a valid token and
//! the masked sentinel otherwise. The result is fully determined: the errors
//! follow a fixed precedence (shape overflow, mask length, start count), and a
//! position overflow names the *first* valid token in row-major order whose
//! position exceeds `usize`, with its exact row start and valid-token count.

use core::fmt;

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Number of `true` entries in `mask[start..end]`.
pub open spec fn valid_count(mask: Seq<bool>, start: int, end: int) -> int
    decreases end - start,
{
    if end <= start {
        0
    } else {
        valid_count(mask, start, end - 1) + if mask[end - 1] { 1int } else { 0int }
    }
}

/// Expected position of flat token `index` in a row-major `[batch, sequence]` mask.
///
/// The row is `index / sequence`; a valid token receives the row's start plus
/// the number of valid tokens before it in that row.
pub open spec fn expected_position(
    mask: Seq<bool>,
    starts: Seq<usize>,
    sequence: int,
    masked: usize,
    index: int,
) -> int {
    if mask[index] {
        starts[index / sequence] + valid_count(mask, (index / sequence) * sequence, index)
    } else {
        masked as int
    }
}

/// The shape checks of [`derive_batched_position_ids`] all pass.
pub open spec fn shape_consistent(
    mask_length: int,
    batch_size: int,
    sequence_length: int,
    start_count: int,
) -> bool {
    &&& batch_size * sequence_length <= usize::MAX
    &&& mask_length == batch_size * sequence_length
    &&& start_count == batch_size
}

/// Every valid flat token before `end` has a representable position.
pub open spec fn positions_fit_before(
    mask: Seq<bool>,
    starts: Seq<usize>,
    sequence: int,
    end: int,
) -> bool {
    forall|index: int|
        0 <= index < end && #[trigger] mask[index] ==> starts[index / sequence]
            + valid_count(mask, (index / sequence) * sequence, index) <= usize::MAX
}

/// A prefix count never exceeds the prefix width.
proof fn lemma_valid_count_bounded(mask: Seq<bool>, start: int, end: int)
    requires
        start <= end,
    ensures
        0 <= valid_count(mask, start, end) <= end - start,
    decreases end - start,
{
    if end > start {
        lemma_valid_count_bounded(mask, start, end - 1);
    }
}

} // verus!

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
///
/// Verified: the result is `Ok` exactly when no valid token's position
/// overflows, and then entry `i` is `start_position + valid_count(0..i)` for a
/// valid token and `masked_position` otherwise; the only possible error is a
/// position overflow at the first overflowing valid token.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Ok(positions) => {
                &&& positions.len() == valid_mask.len()
                &&& forall|index: int| 0 <= index < positions.len() ==>
                    #[trigger] positions[index] == if valid_mask[index] {
                        start_position + valid_count(valid_mask@, 0, index)
                    } else {
                        masked_position as int
                    }
            },
            Err(PositionIdError::PositionOverflow {
                sequence,
                token,
                start_position: start,
                valid_tokens_before,
            }) => {
                &&& sequence == 0
                &&& token < valid_mask.len()
                &&& valid_mask[token as int]
                &&& start == start_position
                &&& valid_tokens_before == valid_count(valid_mask@, 0, token as int)
                &&& start_position + valid_tokens_before > usize::MAX
                &&& forall|index: int| 0 <= index < token && #[trigger] valid_mask[index]
                    ==> start_position + valid_count(valid_mask@, 0, index) <= usize::MAX
            },
            Err(_) => false,
        }
))]
pub fn derive_position_ids(
    valid_mask: &[bool],
    start_position: usize,
    masked_position: usize,
) -> Result<Vec<usize>, PositionIdError> {
    let starts = [start_position];
    let result =
        derive_batched_position_ids(valid_mask, 1, valid_mask.len(), &starts, masked_position);
    #[cfg(verus_keep_ghost)]
    proof! {
        // One row of `len` tokens: every flat index lies in row zero.
        assert forall|index: int| 0 <= index < valid_mask.len() implies
            #[trigger] (index / (valid_mask.len() as int)) == 0 by {
            vstd::arithmetic::div_mod::lemma_basic_div(index, valid_mask.len() as int);
        }
    }
    result
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
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Ok(positions) => {
                &&& batch_size * sequence_length <= usize::MAX
                &&& valid_mask.len() == batch_size * sequence_length
                &&& start_positions.len() == batch_size
                &&& positions.len() == batch_size * sequence_length
                &&& forall|index: int| 0 <= index < positions.len() ==>
                    #[trigger] positions[index] == expected_position(
                        valid_mask@,
                        start_positions@,
                        sequence_length as int,
                        masked_position,
                        index,
                    )
            },
            Err(PositionIdError::ShapeOverflow { batch_size: rows, sequence_length: width }) => {
                &&& rows == batch_size
                &&& width == sequence_length
                &&& batch_size * sequence_length > usize::MAX
            },
            Err(PositionIdError::MaskLengthMismatch { expected, actual }) => {
                &&& expected == batch_size * sequence_length
                &&& actual == valid_mask.len()
                &&& expected != actual
            },
            Err(PositionIdError::StartCountMismatch { expected, actual }) => {
                &&& batch_size * sequence_length <= usize::MAX
                &&& valid_mask.len() == batch_size * sequence_length
                &&& expected == batch_size
                &&& actual == start_positions.len()
                &&& expected != actual
            },
            Err(PositionIdError::PositionOverflow {
                sequence,
                token,
                start_position,
                valid_tokens_before,
            }) => {
                &&& shape_consistent(
                    valid_mask.len() as int,
                    batch_size as int,
                    sequence_length as int,
                    start_positions.len() as int,
                )
                &&& sequence < batch_size
                &&& token < sequence_length
                &&& valid_mask[sequence * sequence_length + token]
                &&& start_position == start_positions[sequence as int]
                &&& valid_tokens_before == valid_count(
                    valid_mask@,
                    sequence * sequence_length,
                    sequence * sequence_length + token,
                )
                &&& start_position + valid_tokens_before > usize::MAX
                // No earlier valid token (in row-major order) overflows.
                &&& positions_fit_before(
                    valid_mask@,
                    start_positions@,
                    sequence_length as int,
                    sequence * sequence_length + token,
                )
            },
        }
))]
pub fn derive_batched_position_ids(
    valid_mask: &[bool],
    batch_size: usize,
    sequence_length: usize,
    start_positions: &[usize],
    masked_position: usize,
) -> Result<Vec<usize>, PositionIdError> {
    let Some(expected) = batch_size.checked_mul(sequence_length) else {
        return Err(PositionIdError::ShapeOverflow {
            batch_size,
            sequence_length,
        });
    };
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
    // `row_start == sequence * sequence_length` is the flat index of the row's
    // first token; maintaining it by addition avoids a multiplication per row.
    let mut sequence = 0usize;
    let mut row_start = 0usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            sequence <= batch_size,
            row_start == sequence * sequence_length,
            expected == batch_size * sequence_length,
            valid_mask.len() == expected,
            start_positions.len() == batch_size,
            positions.len() == row_start,
            forall|index: int| 0 <= index < positions.len() ==>
                #[trigger] positions[index] == expected_position(
                    valid_mask@,
                    start_positions@,
                    sequence_length as int,
                    masked_position,
                    index,
                ),
        decreases batch_size - sequence,
    ))]
    while sequence < batch_size {
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(row_start + sequence_length <= expected) by (nonlinear_arith)
                requires
                    row_start == sequence * sequence_length,
                    expected == batch_size * sequence_length,
                    sequence < batch_size,
            ;
        }
        let start = start_positions[sequence];
        let mut valid_before = 0usize;
        let mut token = 0usize;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                sequence < batch_size,
                row_start == sequence * sequence_length,
                row_start + sequence_length <= expected,
                expected == batch_size * sequence_length,
                valid_mask.len() == expected,
                start_positions.len() == batch_size,
                start == start_positions[sequence as int],
                token <= sequence_length,
                positions.len() == row_start + token,
                valid_before == valid_count(valid_mask@, row_start as int, row_start + token),
                valid_before <= token,
                forall|index: int| 0 <= index < positions.len() ==>
                    #[trigger] positions[index] == expected_position(
                        valid_mask@,
                        start_positions@,
                        sequence_length as int,
                        masked_position,
                        index,
                    ),
            decreases sequence_length - token,
        ))]
        while token < sequence_length {
            let index = row_start + token;
            #[cfg(verus_keep_ghost)]
            proof! {
                // `index = sequence * sequence_length + token` with `token < sequence_length`.
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
                    index as int,
                    sequence_length as int,
                    sequence as int,
                    token as int,
                );
            }
            if valid_mask[index] {
                let Some(position) = start.checked_add(valid_before) else {
                    #[cfg(verus_keep_ghost)]
                    proof! {
                        // Every earlier entry is a stored `usize` equal to its
                        // expected position, so it fits.
                        assert forall|earlier: int| 0 <= earlier < index && #[trigger] valid_mask@[earlier]
                            implies start_positions@[earlier / (sequence_length as int)]
                                + valid_count(
                                    valid_mask@,
                                    (earlier / (sequence_length as int)) * sequence_length,
                                    earlier,
                                ) <= usize::MAX by {
                            assert(positions[earlier] == expected_position(
                                valid_mask@,
                                start_positions@,
                                sequence_length as int,
                                masked_position,
                                earlier,
                            ));
                        }
                    }
                    return Err(PositionIdError::PositionOverflow {
                        sequence,
                        token,
                        start_position: start,
                        valid_tokens_before: valid_before,
                    });
                };
                positions.push(position);
                valid_before += 1;
            } else {
                positions.push(masked_position);
            }
            token += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((sequence + 1) * sequence_length == row_start + sequence_length)
                by (nonlinear_arith)
                requires
                    row_start == sequence * sequence_length,
            ;
        }
        row_start += sequence_length;
        sequence += 1;
    }

    Ok(positions)
}

/// Invalid shape or arithmetic while deriving absolute position IDs.
#[cfg_attr(verus_keep_ghost, verus_verify)]
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
