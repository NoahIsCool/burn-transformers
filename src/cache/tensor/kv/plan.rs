//! Verified integer planning for native key/value cache groups.
//!
//! A KV group stores committed planes `[batch, heads, time, dim]` whose time
//! extent depends on the storage mode: full storage keeps every logical
//! position (`time = capacity`), a standard ring keeps `time = W` slots with
//! `slot(p) = p mod W`, and a mirrored ring keeps `time = 2W` so that slot `s`
//! is duplicated at `s + W` and any run of at most `W` positions is one
//! contiguous view. `W` is the group's own window or else the bank's.
//!
//! This module answers the integer questions every KV read and write asks:
//!
//! * **Allocation extent.** [`committed_time_slots`] validates the storage
//!   mode against the windows and returns the exact time extent.
//! * **Visible windows.** A committed read of a row of length `L` sees its
//!   newest `min(L, W)` positions (all `L` for full storage); a working read
//!   during speculation sees the window of `C + P` tokens split into a
//!   committed and a staged part (the verified `ring_working_window`).
//! * **Mirrored slices.** The newest `k <= W` positions before `L` start at
//!   physical slot `(L - k) mod W` and stay below `2W`.
//!
//! The module is burn-free so Verus checks it in place through
//! `verus/cache.rs`; the tensor slicing driven by these plans remains covered
//! by the KV cache tests.

use core::ops::Range;

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
use crate::cache::retention::working_window_correct;
use crate::cache::retention::{WorkingWindow, ring_working_window};
use crate::cache::tensor::plan::{WriteEndViolation, write_end};

/// Physical time-axis strategy used by a native key/value cache.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvCacheStorage {
    /// Stores every logical position at its absolute time index.
    Full,
    /// Stores `position mod window`, producing at most two chronological views.
    SlidingRing,
    /// Duplicates each ring slot at `slot + window` for one contiguous view.
    MirroredSlidingRing,
}

/// Integer reason a group's storage mode and windows are inconsistent.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimeSlotViolation {
    /// In-place staging was requested for ring storage.
    InPlaceStagingRequiresFullStorage,
    /// A group-specific window was set on full storage.
    WindowOnFullStorage {
        /// The group's window.
        window: usize,
    },
    /// A group-specific window exceeds the logical capacity.
    WindowExceedsCapacity {
        /// The group's window.
        window: usize,
    },
    /// Ring storage has neither a group nor a bank window.
    WindowRequired,
    /// The mirrored extent `2W` overflows `usize`.
    MirroredOverflow {
        /// The effective ring window.
        window: usize,
    },
}

/// Row-wise visible windows of a committed batched read.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommittedRows {
    /// Newest committed positions visible in each row.
    pub(crate) keep: Vec<usize>,
    /// Visible absolute interval `[L_b - keep_b, L_b)` of each row.
    pub(crate) retained: Vec<Range<usize>>,
}

/// Row-wise windows of a working batched read during speculation.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkingRows {
    /// Working end `C_b + P_b` of each row.
    pub(crate) ends: Vec<usize>,
    /// Committed and staged split of each row's visible window.
    pub(crate) windows: Vec<WorkingWindow>,
    /// Newest committed positions visible in each row.
    pub(crate) keep: Vec<usize>,
    /// Visible absolute interval `[retained_start_b, C_b + P_b)` of each row.
    pub(crate) retained: Vec<Range<usize>>,
}

#[cfg(verus_keep_ghost)]
verus! {

/// The ring width a group uses: its own window, else the bank's; none for full storage.
pub open spec fn effective_window(
    storage: KvCacheStorage,
    group_window: Option<usize>,
    config_window: Option<usize>,
) -> Option<usize> {
    match storage {
        KvCacheStorage::Full => None,
        _ => if group_window is Some { group_window } else { config_window },
    }
}

/// The committed time extent of a group, or why its configuration is invalid.
///
/// Checks run in order: in-place staging needs full storage; a group window
/// needs ring storage and must fit the capacity; then full storage spans the
/// capacity, a ring its effective window `W`, and a mirrored ring `2W`.
pub(crate) open spec fn time_slots(
    storage: KvCacheStorage,
    in_place: bool,
    group_window: Option<usize>,
    config_window: Option<usize>,
    capacity: usize,
) -> Result<usize, TimeSlotViolation> {
    if in_place && storage != KvCacheStorage::Full {
        Err(TimeSlotViolation::InPlaceStagingRequiresFullStorage)
    } else if group_window is Some && storage == KvCacheStorage::Full {
        Err(TimeSlotViolation::WindowOnFullStorage { window: group_window->0 })
    } else if group_window is Some && group_window->0 > capacity {
        Err(TimeSlotViolation::WindowExceedsCapacity { window: group_window->0 })
    } else {
        match storage {
            KvCacheStorage::Full => Ok(capacity),
            KvCacheStorage::SlidingRing => match effective_window(storage, group_window,
                config_window) {
                Some(window) => Ok(window),
                None => Err(TimeSlotViolation::WindowRequired),
            },
            KvCacheStorage::MirroredSlidingRing => match effective_window(storage,
                group_window, config_window) {
                Some(window) => if 2 * window <= usize::MAX {
                    Ok((2 * window) as usize)
                } else {
                    Err(TimeSlotViolation::MirroredOverflow { window })
                },
                None => Err(TimeSlotViolation::WindowRequired),
            },
        }
    }
}

/// Visible committed positions of a row of length `length`: `min(length, W)`, or all.
pub open spec fn visible_count_spec(length: usize, window: Option<usize>) -> usize {
    match window {
        Some(width) => if length <= width { length } else { width },
        None => length,
    }
}

/// `plan` is the working window of `committed + staged` tokens under `window`.
pub(crate) open spec fn working_window_for(
    plan: WorkingWindow,
    committed: usize,
    staged: usize,
    window: Option<usize>,
) -> bool {
    match window {
        Some(width) => working_window_correct(plan, committed as int, staged as int, width as int),
        None => plan.retained_start == 0 && plan.committed_keep == committed
            && plan.staged_skip == 0,
    }
}

} // verus!

/// Returns the exact committed time extent of a group, validating its windows.
///
/// `group_window` is the group's own window and `config_window` the bank's.
/// Proved: the result equals `time_slots` above for every input, so the
/// extent is exactly `capacity`, `W`, or `2W` and every inconsistent
/// configuration is rejected with the first violated rule.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result == time_slots(storage, in_place, group_window, config_window, capacity),
))]
pub(crate) fn committed_time_slots(
    storage: KvCacheStorage,
    in_place: bool,
    group_window: Option<usize>,
    config_window: Option<usize>,
    capacity: usize,
) -> Result<usize, TimeSlotViolation> {
    let full = match storage {
        KvCacheStorage::Full => true,
        KvCacheStorage::SlidingRing | KvCacheStorage::MirroredSlidingRing => false,
    };
    if in_place && !full {
        return Err(TimeSlotViolation::InPlaceStagingRequiresFullStorage);
    }
    if let Some(window) = group_window {
        if full {
            return Err(TimeSlotViolation::WindowOnFullStorage { window });
        }
        if window > capacity {
            return Err(TimeSlotViolation::WindowExceedsCapacity { window });
        }
    }
    let effective = match group_window {
        Some(window) => Some(window),
        None => config_window,
    };
    match storage {
        KvCacheStorage::Full => Ok(capacity),
        KvCacheStorage::SlidingRing => match effective {
            Some(window) => Ok(window),
            None => Err(TimeSlotViolation::WindowRequired),
        },
        KvCacheStorage::MirroredSlidingRing => match effective {
            Some(window) => match window.checked_mul(2) {
                Some(slots) => Ok(slots),
                None => Err(TimeSlotViolation::MirroredOverflow { window }),
            },
            None => Err(TimeSlotViolation::WindowRequired),
        },
    }
}

/// Returns the newest positions a committed read of length `length` sees.
///
/// Proved: exactly `min(length, W)` for a ring of width `W`, and `length` for
/// full storage, hence never more than `length`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(count =>
    ensures
        count == visible_count_spec(length, window),
        count <= length,
))]
pub(crate) fn visible_count(length: usize, window: Option<usize>) -> usize {
    match window {
        Some(width) => {
            if length <= width {
                length
            } else {
                width
            }
        }
        None => length,
    }
}

/// Plans the visible committed window of every row of a batched read.
///
/// Proved: row `b` keeps exactly `visible_count(L_b)` positions and retains
/// the absolute interval `[L_b - keep_b, L_b)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rows =>
    ensures
        rows.keep.len() == lengths.len(),
        rows.retained.len() == lengths.len(),
        forall|b: int| 0 <= b < lengths.len() ==> {
            &&& #[trigger] rows.keep[b] == visible_count_spec(lengths[b], window)
            &&& rows.retained[b].start == lengths[b] - rows.keep[b]
            &&& rows.retained[b].end == lengths[b]
        },
))]
pub(crate) fn committed_read_rows(lengths: &[usize], window: Option<usize>) -> CommittedRows {
    let mut keep: Vec<usize> = Vec::with_capacity(lengths.len());
    let mut retained: Vec<Range<usize>> = Vec::with_capacity(lengths.len());
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= lengths.len(),
            keep.len() == row,
            retained.len() == row,
            forall|b: int| 0 <= b < row ==> {
                &&& #[trigger] keep[b] == visible_count_spec(lengths[b], window)
                &&& retained[b].start == lengths[b] - keep[b]
                &&& retained[b].end == lengths[b]
            },
        decreases lengths.len() - row,
    ))]
    while row < lengths.len() {
        let length = lengths[row];
        let visible = visible_count(length, window);
        keep.push(visible);
        retained.push(length - visible..length);
        row += 1;
    }
    CommittedRows { keep, retained }
}

/// Splits one row's working window into its committed and staged parts.
///
/// A ring of width `W` uses the verified `ring_working_window`; full storage
/// keeps every committed token and every proposal.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    requires
        committed + staged <= usize::MAX,
        window matches Some(width) ==> width > 0,
    ensures
        working_window_for(plan, committed, staged, window),
))]
pub(crate) fn working_window(
    committed: usize,
    staged: usize,
    window: Option<usize>,
) -> WorkingWindow {
    match window {
        Some(width) => ring_working_window(committed, staged, width),
        None => WorkingWindow {
            retained_start: 0,
            committed_keep: committed,
            staged_skip: 0,
        },
    }
}

/// Plans every row's working window during speculation.
///
/// Row `b` has committed length `C_b` and `P_b` staged proposals; rows beyond
/// the shorter input are not planned, matching a zipped scan.
///
/// Proved: the plan exists exactly when every `C_b + P_b` is representable
/// and within `capacity`; it then records each working end exactly, splits
/// each row's window as `working_window` does, and retains
/// `[retained_start_b, C_b + P_b)`. A rejection names the first failing row
/// with its write-end violation.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        window matches Some(width) ==> width > 0,
    ensures
        ({
            let rows = if committed.len() <= staged.len() { committed.len() } else { staged.len() };
            &&& result is Ok <==> forall|b: int| 0 <= b < rows ==>
                #[trigger] committed[b] + staged[b] <= capacity
            &&& result matches Ok(plan) ==> {
                &&& plan.ends.len() == rows
                &&& plan.windows.len() == rows
                &&& plan.keep.len() == rows
                &&& plan.retained.len() == rows
                &&& forall|b: int| 0 <= b < rows ==> {
                    &&& #[trigger] plan.ends[b] == committed[b] + staged[b]
                    &&& working_window_for(plan.windows[b], committed[b], staged[b], window)
                    &&& plan.keep[b] == plan.windows[b].committed_keep
                    &&& plan.retained[b].start == plan.windows[b].retained_start
                    &&& plan.retained[b].end == plan.ends[b]
                }
            }
            &&& result matches Err((row, violation)) ==> {
                &&& row < rows
                &&& forall|b: int| 0 <= b < row ==> #[trigger] committed[b] + staged[b] <= capacity
                &&& match violation {
                    WriteEndViolation::Overflow =>
                        committed[row as int] + staged[row as int] > usize::MAX,
                    WriteEndViolation::ExceedsCapacity { end } =>
                        end == committed[row as int] + staged[row as int] && end > capacity,
                }
            }
        }),
))]
pub(crate) fn working_read_rows(
    committed: &[usize],
    staged: &[usize],
    window: Option<usize>,
    capacity: usize,
) -> Result<WorkingRows, (usize, WriteEndViolation)> {
    let rows = if committed.len() <= staged.len() {
        committed.len()
    } else {
        staged.len()
    };
    let mut ends: Vec<usize> = Vec::with_capacity(rows);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows <= committed.len(),
            rows <= staged.len(),
            ends.len() == row,
            forall|b: int| 0 <= b < row ==> #[trigger] committed[b] + staged[b] <= capacity,
            forall|b: int| 0 <= b < row ==> #[trigger] ends[b] == committed[b] + staged[b],
        decreases rows - row,
    ))]
    while row < rows {
        match write_end(committed[row], staged[row], capacity) {
            Ok(end) => ends.push(end),
            Err(violation) => return Err((row, violation)),
        }
        row += 1;
    }
    let mut windows: Vec<WorkingWindow> = Vec::with_capacity(rows);
    let mut keep: Vec<usize> = Vec::with_capacity(rows);
    let mut retained: Vec<Range<usize>> = Vec::with_capacity(rows);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows <= committed.len(),
            rows <= staged.len(),
            window matches Some(width) ==> width > 0,
            ends.len() == rows,
            forall|b: int| 0 <= b < rows ==> #[trigger] committed[b] + staged[b] <= capacity,
            forall|b: int| 0 <= b < rows ==> #[trigger] ends[b] == committed[b] + staged[b],
            windows.len() == row,
            keep.len() == row,
            retained.len() == row,
            forall|b: int| 0 <= b < row ==> {
                &&& working_window_for(#[trigger] windows[b], committed[b], staged[b], window)
                &&& keep[b] == windows[b].committed_keep
                &&& retained[b].start == windows[b].retained_start
                &&& retained[b].end == ends[b]
            },
        decreases rows - row,
    ))]
    while row < rows {
        let plan = working_window(committed[row], staged[row], window);
        windows.push(plan);
        keep.push(plan.committed_keep);
        retained.push(plan.retained_start..ends[row]);
        row += 1;
    }
    Ok(WorkingRows {
        ends,
        windows,
        keep,
        retained,
    })
}

/// Returns the physical slots of the newest `keep` positions starting at `start`
/// in a mirrored ring of width `window`.
///
/// Physical index `q < 2W` of a mirrored ring holds slot `q mod W`, so the run
/// begins at `start mod W` and stays contiguous. The caller guarantees
/// `keep <= W` (a read never sees more than the window) and that `2W` is
/// representable, which [`committed_time_slots`] established when the group
/// was allocated.
///
/// Proved: the range has exactly `keep` slots, ends at or below `2W`, and its
/// offset `j` holds the slot of absolute position `start + j`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(slots =>
    requires
        window > 0,
        keep <= window,
        2 * window <= usize::MAX,
    ensures
        slots.start == start % window,
        slots.end == slots.start + keep,
        slots.end <= 2 * window,
        forall|j: int| 0 <= j < keep ==>
            #[trigger] ((slots.start + j) % (window as int)) == (start + j) % (window as int),
))]
pub(crate) fn mirrored_window_slots(start: usize, keep: usize, window: usize) -> Range<usize> {
    let physical_start = start % window;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|j: int| 0 <= j < keep implies
            #[trigger] ((physical_start + j) % (window as int)) == (start + j) % (window as int)
        by {
            vstd::arithmetic::div_mod::lemma_add_mod_noop(start as int, j, window as int);
            vstd::arithmetic::div_mod::lemma_add_mod_noop(physical_start as int, j, window as int);
            vstd::arithmetic::div_mod::lemma_mod_twice(start as int, window as int);
        }
    }
    physical_start..physical_start + keep
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
