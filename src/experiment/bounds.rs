//! Pure integer bounds shared by execution contexts, hooks, and training losses.
//!
//! These functions hold the index and range arithmetic behind tensor checks:
//! positional extents, value ranges read back from the device, broadcast
//! compatibility, staged block cursors, and the pairwise reduction schedule.
//! They use no Burn types, floating point, or allocation, so Verus checks these
//! exact bodies through `verus/experiment.rs` (see `verus/README.md`).
//!
//! # Verified properties
//!
//! - [`extent_after`] returns `high + 1` exactly when that fits in `usize`.
//! - [`range_within`] accepts exactly the ranges `0 <= low <= high < extent`.
//! - [`broadcasts_to`] accepts exactly the shapes whose every axis is one or equal.
//! - [`follows_cursor`] accepts exactly the ranges that resume at the cursor and
//!   stay within the stack, so consecutive accepted ranges tile `0..total`.
//! - [`fold_widths`] splits a width into `floor(w/2)` pairs and a carried odd
//!   column, and strictly shrinks the width, so the reduction terminates.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Returns the positional extent `high + 1` for a nonnegative maximum position.
///
/// `None` means `high` is negative or `high + 1` does not fit in `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(extent =>
    ensures
        match extent {
            Some(extent) => 0 <= high && extent == high + 1,
            None => high < 0 || high + 1 > usize::MAX,
        },
))]
pub(crate) fn extent_after(high: i64) -> Option<usize> {
    if high < 0 {
        return None;
    }
    let high = high as u64;
    if high >= usize::MAX as u64 {
        return None;
    }
    Some(high as usize + 1)
}

/// Whether every value in `low..=high` lies in `0..extent`.
///
/// For device-read extrema (`low = min`, `high = max`) this is exactly the
/// check that every position or label indexes an `extent`-sized table.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(within =>
    ensures
        within == (0 <= low && low <= high && high < extent),
))]
pub(crate) fn range_within(low: i64, high: i64, extent: usize) -> bool {
    low >= 0 && low <= high && (high as u64) < (extent as u64)
}

/// Whether `actual` broadcasts to `expected`: every axis is one or equal.
///
/// Shapes of different rank never broadcast here; callers compare equal ranks.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(broadcasts =>
    ensures
        broadcasts == (actual.len() == expected.len() && forall|axis: int| 0 <= axis < actual.len() ==>
            actual@[axis] == 1 || actual@[axis] == expected@[axis]),
))]
pub(crate) fn broadcasts_to(actual: &[usize], expected: &[usize]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    let mut axis = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            0 <= axis <= actual.len(),
            actual.len() == expected.len(),
            forall|earlier: int| 0 <= earlier < axis ==>
                actual@[earlier] == 1 || actual@[earlier] == expected@[earlier],
        decreases actual.len() - axis,
    ))]
    while axis < actual.len() {
        if actual[axis] != 1 && actual[axis] != expected[axis] {
            return false;
        }
        axis += 1;
    }
    true
}

/// Whether blocks `start..end` resume a staged stack at `cursor` and stay within `total`.
///
/// Accepting a range and then setting `cursor = end` makes successive accepted
/// ranges contiguous and nondecreasing, so every block of `0..total` runs at
/// most once and in order. Empty ranges at the cursor are accepted.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(follows =>
    ensures
        follows == (start == cursor && start <= end && end <= total),
))]
pub(crate) fn follows_cursor(cursor: usize, start: usize, end: usize, total: usize) -> bool {
    start == cursor && start <= end && end <= total
}

/// One folding level of a pairwise row reduction: the pair count and the next width.
///
/// `floor(width / 2)` column pairs are added and an odd middle column is
/// carried, so the next width is `ceil(width / 2)`: at least one and strictly
/// less than `width` for every `width >= 2`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(widths =>
    requires
        width >= 2,
    ensures
        2 * widths.0 + width % 2 == width,
        widths.1 == widths.0 + width % 2,
        1 <= widths.1 < width,
))]
pub(crate) fn fold_widths(width: usize) -> (usize, usize) {
    let half = width / 2;
    (half, half + width % 2)
}

#[cfg(test)]
#[path = "bounds_tests.rs"]
mod tests;
