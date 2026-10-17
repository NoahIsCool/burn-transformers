//! Dependency-free extent and alignment checks of MTP speculation.
//!
//! # Extents
//!
//! Every write of a speculative cycle appends `count` positions after a
//! cursor `start`: the target verifier's suffix, the MTP stage's pairs, the
//! positions an MTP call is validated against, and the `K + 1` staging slots
//! a state reserves. Each append must end at `start + count <= limit` (a
//! cache capacity or `max_position_embeddings`), and the end must be
//! representable. [`bounded_end`] performs that check once for all of them
//! and reports which condition failed, so each caller keeps its own error.
//!
//! # Shifted alignment
//!
//! MTP cache position `i` holds the pair `(x_{i+1}, h_i)`, so a stable cycle
//! boundary satisfies `L_target = L_mtp + 1` for the single speculative row;
//! [`shifted_alignment`] checks exactly that.
//!
//! # Query positions
//!
//! An MTP step writes its `w` pairs to slots `[W, W + w)` after the working
//! length `W` and attends with its last `q <= w` pairs, which therefore sit
//! at the final `q` positions before `end = W + w`. [`trailing_positions`]
//! lists them as device position IDs.
//!
//! # Verified checks
//!
//! Every function here is checked in place by Verus through
//! `verus/nemotron.rs` (see `verus/README.md`). For every input they never
//! overflow, and:
//!
//! - an accepted extent is exactly `start + count <= limit`, and every
//!   rejection names its exact cause;
//! - alignment holds exactly for one row with `L_target = L_mtp + 1`;
//! - the query positions are exactly `end - q, …, end - 1`, refused exactly
//!   when the last one exceeds `i64::MAX`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Why an append `start + count` is not allowed.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExtentError {
    /// `start + count` does not fit `usize`.
    Overflow,
    /// The end `start + count` exceeds the limit.
    Exceeds(
        /// The representable end that exceeds the limit.
        usize,
    ),
}

/// Returns the end `start + count` of an append when it is at most `limit`.
///
/// The proof establishes that `Ok(end)` means `end = start + count <= limit`,
/// `Exceeds(end)` means `end = start + count > limit`, and `Overflow` means
/// the sum exceeds `usize::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(end =>
    ensures
        match end {
            Ok(end) => end == start + count && end <= limit,
            Err(ExtentError::Exceeds(end)) => end == start + count && end > limit,
            Err(ExtentError::Overflow) => start + count > usize::MAX,
        },
))]
pub(crate) const fn bounded_end(
    start: usize,
    count: usize,
    limit: usize,
) -> Result<usize, ExtentError> {
    match start.checked_add(count) {
        None => Err(ExtentError::Overflow),
        Some(end) if end > limit => Err(ExtentError::Exceeds(end)),
        Some(end) => Ok(end),
    }
}

/// Whether one target row and one MTP row satisfy `L_target = L_mtp + 1`.
///
/// The proof establishes the exact rule; the comparison never overflows.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(aligned =>
    ensures
        aligned == (target.len() == 1 && mtp.len() == 1 && mtp@[0] + 1 == target@[0]),
))]
pub(crate) fn shifted_alignment(target: &[usize], mtp: &[usize]) -> bool {
    target.len() == 1 && mtp.len() == 1 && mtp[0] < target[0] && target[0] - mtp[0] == 1
}

/// Absolute positions `end - queries, …, end - 1` of a step's last `queries` queries.
///
/// The caller guarantees `queries <= end` (its `q <= w` queries end at
/// `end = W + w`). Returns `None` exactly when the last position exceeds
/// `i64::MAX`, the range of device position IDs; the proof establishes
/// every returned entry.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(positions =>
    requires
        queries <= end,
    ensures
        match positions {
            Some(positions) => {
                &&& positions.len() == queries
                &&& forall|i: int| 0 <= i < queries
                    ==> #[trigger] positions@[i] == end - queries + i
            },
            None => queries > 0 && end - 1 > i64::MAX,
        },
))]
pub(crate) fn trailing_positions(end: usize, queries: usize) -> Option<Vec<i64>> {
    let first = end - queries;
    let mut positions: Vec<i64> = Vec::with_capacity(queries);
    let mut position = first;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            first == end - queries,
            first <= position <= end,
            positions.len() == position - first,
            forall|i: int| 0 <= i < positions.len() ==> #[trigger] positions@[i] == first + i,
        decreases end - position,
    ))]
    while position < end {
        // `usize` has at most 64 bits, so widening to `u64` is lossless.
        if position as u64 > i64::MAX as u64 {
            return None;
        }
        positions.push(position as i64);
        position += 1;
    }
    Some(positions)
}

#[cfg(test)]
#[path = "cycle_tests.rs"]
mod tests;
