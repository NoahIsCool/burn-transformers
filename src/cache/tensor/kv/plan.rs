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
//!   mode against the windows and returns the exact time extent together
//!   with the effective width `W` that every later read and write uses.
//! * **Visible windows.** A committed read of a row of length `L` sees its
//!   newest `min(L, W)` positions (all `L` for full storage); a working read
//!   during speculation sees the window of `C + P` tokens split into a
//!   committed and a staged part (the verified `ring_working_window`).
//! * **Mirrored slices.** The newest `k <= W` positions before `L` start at
//!   physical slot `(L - k) mod W` and stay below `2W`.
//! * **Committed writes.** A direct append of `n` tokens at `s`, read from
//!   input columns `[c, c + n)`, writes slots `[s, s + n)` under full
//!   storage; a ring of width `W` keeps the newest `min(n, W)` tokens and
//!   writes column `j` to slot `(s + j - c) mod W` (and again at `+ W` when
//!   mirrored). Committing an accepted prefix of staged proposals plans the
//!   same writes from staging offset zero after its own admission checks.
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
use crate::cache::ring::{RingPlanError, mirrored_write_slots, plan_ring_write_slots};
#[cfg(verus_keep_ghost)]
use crate::cache::ring::{ring_slot, write_plan_correct};
use crate::cache::rows::RowRun;
#[cfg(verus_keep_ghost)]
use crate::cache::tensor::plan::write_end_violation_exact;
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

/// Returns the ring width a group uses: its own window, else the bank's.
///
/// Proved: exactly `effective_window` above, so full storage has no ring
/// width and a group window always takes precedence over the bank's.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(window =>
    ensures
        window == effective_window(storage, group_window, config_window),
))]
pub(crate) fn effective_window_of(
    storage: KvCacheStorage,
    group_window: Option<usize>,
    config_window: Option<usize>,
) -> Option<usize> {
    match storage {
        KvCacheStorage::Full => None,
        KvCacheStorage::SlidingRing | KvCacheStorage::MirroredSlidingRing => match group_window {
            Some(window) => Some(window),
            None => config_window,
        },
    }
}

/// Validated committed time extent and ring width of a KV group.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TimeExtent {
    /// Committed time-axis slots: the capacity, `W`, or `2W`.
    pub(crate) slots: usize,
    /// Effective ring width `W`; `None` for full storage.
    pub(crate) window: Option<usize>,
}

/// Returns the exact committed time extent and ring width of a group.
///
/// `group_window` is the group's own window and `config_window` the bank's.
/// Proved: the extent equals `time_slots` above for every input, so it is
/// exactly `capacity`, `W`, or `2W` and every inconsistent configuration is
/// rejected with the first violated rule. The returned width is the same
/// `effective_window` that sized the extent, so a mirrored group's `2W` is
/// representable for exactly the width its reads and writes later use.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Ok(extent) => {
                &&& time_slots(storage, in_place, group_window, config_window, capacity)
                    == Ok::<usize, TimeSlotViolation>(extent.slots)
                &&& extent.window == effective_window(storage, group_window, config_window)
                &&& match storage {
                    KvCacheStorage::Full => extent.window is None && extent.slots == capacity,
                    KvCacheStorage::SlidingRing => extent.window == Some(extent.slots),
                    KvCacheStorage::MirroredSlidingRing => extent.window matches Some(width)
                        && extent.slots == 2 * width,
                }
            },
            Err(violation) => time_slots(storage, in_place, group_window, config_window, capacity)
                == Err::<usize, TimeSlotViolation>(violation),
        },
))]
pub(crate) fn committed_time_slots(
    storage: KvCacheStorage,
    in_place: bool,
    group_window: Option<usize>,
    config_window: Option<usize>,
    capacity: usize,
) -> Result<TimeExtent, TimeSlotViolation> {
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
    let effective = effective_window_of(storage, group_window, config_window);
    let slots = match storage {
        KvCacheStorage::Full => capacity,
        KvCacheStorage::SlidingRing => match effective {
            Some(window) => window,
            None => return Err(TimeSlotViolation::WindowRequired),
        },
        KvCacheStorage::MirroredSlidingRing => match effective {
            Some(window) => match window.checked_mul(2) {
                Some(slots) => slots,
                None => return Err(TimeSlotViolation::MirroredOverflow { window }),
            },
            None => return Err(TimeSlotViolation::WindowRequired),
        },
    };
    Ok(TimeExtent {
        slots,
        window: effective,
    })
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
/// representable, which [`committed_time_slots`] proved for the very width it
/// returned and the group stored when it was allocated.
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

/// Returns the absolute and linear-staging ranges of a working read's staged part.
///
/// With committed length `C`, `P` staged proposals, and a working window
/// that skips the oldest `skip` proposals, the staged segment covers
/// absolute positions `[C + skip, C + P)` read from staging offsets
/// `[skip, P)`.
///
/// Proved: both ranges have exactly `P - skip` entries and offset `j` of the
/// staging range holds absolute position `C + j`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(ranges =>
    requires
        committed + staged <= usize::MAX,
        plan.staged_skip <= staged,
    ensures
        ranges.0.start == committed + plan.staged_skip,
        ranges.0.end == committed + staged,
        ranges.1.start == plan.staged_skip,
        ranges.1.end == staged,
        ranges.0.end - ranges.0.start == ranges.1.end - ranges.1.start,
))]
pub(crate) fn staged_read_ranges(
    committed: usize,
    staged: usize,
    plan: WorkingWindow,
) -> (Range<usize>, Range<usize>) {
    (
        committed + plan.staged_skip..committed + staged,
        plan.staged_skip..staged,
    )
}

/// Integer reason a write-ahead would expose unpublished tokens.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriteAheadViolation {
    /// Ring storage would overwrite a slot that is still visible.
    RingStorage,
    /// Inside a transaction only in-place staging writes final slots.
    SeparateStaging,
}

/// Decides whether a group may write a step's tokens ahead of attention.
///
/// Proved: allowed exactly for full storage outside a transaction, or for
/// full storage with in-place staging inside one; ring storage is rejected
/// first.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result is Ok <==> (storage == KvCacheStorage::Full && (!staging || in_place)),
        storage != KvCacheStorage::Full ==> result == Err::<(), WriteAheadViolation>(
            WriteAheadViolation::RingStorage),
        storage == KvCacheStorage::Full && staging && !in_place ==>
            result == Err::<(), WriteAheadViolation>(WriteAheadViolation::SeparateStaging),
))]
pub(crate) fn write_ahead_allowed(
    storage: KvCacheStorage,
    in_place: bool,
    staging: bool,
) -> Result<(), WriteAheadViolation> {
    match storage {
        KvCacheStorage::Full => {}
        KvCacheStorage::SlidingRing | KvCacheStorage::MirroredSlidingRing => {
            return Err(WriteAheadViolation::RingStorage);
        }
    }
    if staging && !in_place {
        return Err(WriteAheadViolation::SeparateStaging);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Committed writes: direct appends and staged-prefix commits
// ---------------------------------------------------------------------------

/// One input interval of a committed KV write and its one or two destinations.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DirectSegment {
    /// Input columns read (staging offsets for a commit).
    pub(crate) input: Range<usize>,
    /// Committed time slots written.
    pub(crate) primary: Range<usize>,
    /// The copy `primary + W` written by a mirrored ring; `None` otherwise.
    pub(crate) mirror: Option<Range<usize>>,
}

/// Integer reason a committed KV write cannot be planned.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectWriteViolation {
    /// Ring storage has no resolved width.
    WindowRequired,
    /// A full-storage commit leaves the logical capacity.
    WriteEnd(WriteEndViolation),
    /// Ring planning rejected the write: an unrepresentable end, a doubled
    /// mirrored extent, or a commit beyond the staged proposals.
    Ring(RingPlanError),
}

/// One run of rows written by the same committed segments.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DirectRun {
    /// Destination rows written together.
    pub(crate) rows: Range<usize>,
    /// Matching input rows, `rows` shifted down by the source offset.
    pub(crate) source_rows: Range<usize>,
    /// Input columns and destinations shared by every row of the run.
    pub(crate) segments: Vec<DirectSegment>,
}

#[cfg(verus_keep_ghost)]
verus! {

/// `segments` is the exact ring write of `count` tokens at `start`, read from columns `[column, column + count)`.
///
/// Only the newest `min(count, width)` tokens survive. The at most two
/// segments are non-empty, ordered, contiguous, and cover exactly input
/// columns `[column + count - min(count, width), column + count)`; column `j`
/// lands in slot `(start + j - column) mod width`; primary ranges never
/// overlap and stay below `width`. A mirrored ring writes each segment again
/// at `primary + width`; a standard ring writes no copy.
pub(crate) open spec fn ring_segments_correct(
    segments: Seq<DirectSegment>,
    mirrored: bool,
    start: usize,
    count: usize,
    column: usize,
    width: usize,
) -> bool {
    let retained = if count <= width { count } else { width };
    &&& segments.len() <= 2
    &&& (segments.len() == 0) == (count == 0)
    &&& segments.len() >= 1 ==> segments[0].input.start == column + count - retained
    &&& segments.len() >= 1 ==> segments[segments.len() - 1].input.end == column + count
    &&& segments.len() == 2 ==> segments[1].input.start == segments[0].input.end
    &&& segments.len() == 2 ==> segments[1].primary.end <= segments[0].primary.start
    &&& forall|i: int| #![auto] 0 <= i < segments.len() ==> {
        &&& segments[i].input.start < segments[i].input.end
        &&& segments[i].primary.start < segments[i].primary.end
        &&& segments[i].primary.end <= width
        &&& segments[i].input.end - segments[i].input.start
            == segments[i].primary.end - segments[i].primary.start
        &&& forall|j: int| segments[i].input.start <= j < segments[i].input.end ==>
            #[trigger] ring_slot(start as int, j - column, width as int)
                == segments[i].primary.start + (j - segments[i].input.start)
        &&& if mirrored {
            &&& segments[i].mirror is Some
            &&& segments[i].mirror->0.start == segments[i].primary.start + width
            &&& segments[i].mirror->0.end == segments[i].primary.end + width
        } else {
            segments[i].mirror is None
        }
    }
}

/// `segments` is the exact committed write of `count` tokens at `start`, read from columns `[column, column + count)`.
///
/// An empty write has no segments. Full storage writes one segment from
/// columns `[column, column + count)` to slots `[start, start + count)`; ring
/// storage writes the `ring_segments_correct` plan of its width.
pub(crate) open spec fn direct_segments_correct(
    segments: Seq<DirectSegment>,
    storage: KvCacheStorage,
    start: usize,
    count: usize,
    column: usize,
    window: Option<usize>,
) -> bool {
    if count == 0 {
        segments.len() == 0
    } else {
        match storage {
            KvCacheStorage::Full => {
                &&& segments.len() == 1
                &&& segments[0].input.start == column
                &&& segments[0].input.end == column + count
                &&& segments[0].primary.start == start
                &&& segments[0].primary.end == start + count
                &&& segments[0].mirror is None
            },
            KvCacheStorage::SlidingRing => window is Some && ring_segments_correct(segments,
                false, start, count, column, window->0),
            KvCacheStorage::MirroredSlidingRing => window is Some && ring_segments_correct(
                segments, true, start, count, column, window->0),
        }
    }
}

/// A direct append of `count` tokens at `start` can be planned.
pub(crate) open spec fn direct_write_accepted(
    storage: KvCacheStorage,
    start: usize,
    count: usize,
    window: Option<usize>,
) -> bool {
    count == 0 || match storage {
        KvCacheStorage::Full => start + count <= usize::MAX,
        KvCacheStorage::SlidingRing => window is Some && start + count <= usize::MAX,
        KvCacheStorage::MirroredSlidingRing => window is Some && 2 * window->0 <= usize::MAX
            && start + count <= usize::MAX,
    }
}

/// `violation` is the first failed check of a direct append.
///
/// The checks run in order: ring storage needs a width, a mirrored ring
/// needs a representable `2W`, and the end `start + count` must be
/// representable.
pub(crate) open spec fn direct_violation_exact(
    storage: KvCacheStorage,
    start: usize,
    count: usize,
    window: Option<usize>,
    violation: DirectWriteViolation,
) -> bool {
    &&& count > 0
    &&& if storage != KvCacheStorage::Full && window is None {
        violation == DirectWriteViolation::WindowRequired
    } else if storage == KvCacheStorage::MirroredSlidingRing && 2 * window->0 > usize::MAX {
        violation == DirectWriteViolation::Ring(
            RingPlanError::DoubledCapacityOverflow { capacity: window->0 })
    } else {
        &&& start + count > usize::MAX
        &&& violation == DirectWriteViolation::Ring(
            RingPlanError::PositionOverflow { start_position: start, token_count: count })
    }
}

/// Committing `accepted` of `proposed` staged tokens after `baseline` can be planned.
pub(crate) open spec fn commit_write_accepted(
    storage: KvCacheStorage,
    baseline: usize,
    proposed: usize,
    accepted: usize,
    window: Option<usize>,
    capacity: usize,
) -> bool {
    &&& baseline + proposed <= usize::MAX
    &&& match storage {
        KvCacheStorage::Full => baseline + accepted <= capacity,
        KvCacheStorage::SlidingRing => window is Some && accepted <= proposed,
        KvCacheStorage::MirroredSlidingRing => window is Some && accepted <= proposed
            && 2 * window->0 <= usize::MAX,
    }
}

/// `violation` is the first failed check of a staged-prefix commit.
///
/// The checks run in order: the proposal end `baseline + proposed` must be
/// representable; full storage then needs the accepted end within the
/// capacity; ring storage needs a width, an accepted prefix within the
/// proposal, and, when mirrored, a representable `2W`.
pub(crate) open spec fn commit_violation_exact(
    storage: KvCacheStorage,
    baseline: usize,
    proposed: usize,
    accepted: usize,
    window: Option<usize>,
    capacity: usize,
    violation: DirectWriteViolation,
) -> bool {
    if baseline + proposed > usize::MAX {
        violation == DirectWriteViolation::Ring(
            RingPlanError::PositionOverflow { start_position: baseline, token_count: proposed })
    } else if storage == KvCacheStorage::Full {
        match violation {
            DirectWriteViolation::WriteEnd(end) =>
                write_end_violation_exact(baseline, accepted, capacity, end),
            _ => false,
        }
    } else if window is None {
        violation == DirectWriteViolation::WindowRequired
    } else if accepted > proposed {
        violation == DirectWriteViolation::Ring(
            RingPlanError::CommitPrefixExceedsStaged { accepted, staged: proposed })
    } else {
        &&& storage == KvCacheStorage::MirroredSlidingRing
        &&& 2 * window->0 > usize::MAX
        &&& violation == DirectWriteViolation::Ring(
            RingPlanError::DoubledCapacityOverflow { capacity: window->0 })
    }
}

} // verus!

/// Plans a standard ring write read from input columns starting at `column`.
///
/// Proved: fails exactly when `start + count` is unrepresentable; otherwise
/// returns the verified ring plan with every input interval shifted by
/// `column`, satisfying `ring_segments_correct`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        width > 0,
        column + count <= usize::MAX,
    ensures
        match result {
            Ok(segments) => {
                &&& start + count <= usize::MAX
                &&& ring_segments_correct(segments@, false, start, count, column, width)
            },
            Err(error) => {
                &&& start + count > usize::MAX
                &&& error == (RingPlanError::PositionOverflow {
                    start_position: start,
                    token_count: count,
                })
            },
        },
))]
fn ring_direct_segments(
    start: usize,
    count: usize,
    column: usize,
    width: usize,
) -> Result<Vec<DirectSegment>, RingPlanError> {
    let plan = plan_ring_write_slots(start, count, width)?;
    let mut segments: Vec<DirectSegment> = Vec::with_capacity(plan.segments.len());
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= plan.segments.len(),
            width > 0,
            column + count <= usize::MAX,
            write_plan_correct(plan, start, count, width),
            segments.len() == index,
            forall|i: int| #![auto] 0 <= i < index ==> {
                &&& segments[i].input.start == plan.segments[i].input.start + column
                &&& segments[i].input.end == plan.segments[i].input.end + column
                &&& segments[i].primary == plan.segments[i].physical
                &&& segments[i].mirror is None
            },
        decreases plan.segments.len() - index,
    ))]
    while index < plan.segments.len() {
        let segment = &plan.segments[index];
        segments.push(DirectSegment {
            input: segment.input.start + column..segment.input.end + column,
            primary: segment.physical.start..segment.physical.end,
            mirror: None,
        });
        index += 1;
    }
    Ok(segments)
}

/// Plans a mirrored ring write read from input columns starting at `column`.
///
/// Proved: fails exactly as the verified mirrored planner does (a doubled
/// extent `2 * width` first, then the end `start + count`); otherwise returns
/// its segments with every input interval shifted by `column` and the copy
/// at `primary + width`, satisfying `ring_segments_correct`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        width > 0,
        column + count <= usize::MAX,
    ensures
        match result {
            Ok(segments) => {
                &&& 2 * width <= usize::MAX
                &&& start + count <= usize::MAX
                &&& ring_segments_correct(segments@, true, start, count, column, width)
            },
            Err(error) => if 2 * width > usize::MAX {
                error == (RingPlanError::DoubledCapacityOverflow { capacity: width })
            } else {
                &&& start + count > usize::MAX
                &&& error == (RingPlanError::PositionOverflow {
                    start_position: start,
                    token_count: count,
                })
            },
        },
))]
fn mirrored_direct_segments(
    start: usize,
    count: usize,
    column: usize,
    width: usize,
) -> Result<Vec<DirectSegment>, RingPlanError> {
    let plan = mirrored_write_slots(start, count, width)?;
    let mut segments: Vec<DirectSegment> = Vec::with_capacity(plan.len());
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= plan.len(),
            width > 0,
            2 * width <= usize::MAX,
            column + count <= usize::MAX,
            plan.len() <= 2,
            (plan.len() == 0) == (count == 0),
            plan.len() >= 1 ==> plan[0].input.start
                == count - (if count <= width { count } else { width }),
            plan.len() >= 1 ==> plan[plan.len() - 1].input.end == count,
            plan.len() == 2 ==> plan[1].input.start == plan[0].input.end,
            plan.len() == 2 ==> plan[1].primary.end <= plan[0].primary.start,
            forall|i: int| #![auto] 0 <= i < plan.len() ==> {
                &&& plan[i].input.start < plan[i].input.end <= count
                &&& plan[i].primary.start < plan[i].primary.end
                &&& plan[i].primary.end <= width
                &&& plan[i].input.end - plan[i].input.start
                    == plan[i].primary.end - plan[i].primary.start
                &&& plan[i].mirror.start == plan[i].primary.start + width
                &&& plan[i].mirror.end == plan[i].primary.end + width
                &&& forall|j: int| plan[i].input.start <= j < plan[i].input.end ==>
                    #[trigger] ring_slot(start as int, j, width as int)
                        == plan[i].primary.start + (j - plan[i].input.start)
            },
            segments.len() == index,
            forall|i: int| #![auto] 0 <= i < index ==> {
                &&& segments[i].input.start == plan[i].input.start + column
                &&& segments[i].input.end == plan[i].input.end + column
                &&& segments[i].primary == plan[i].primary
                &&& segments[i].mirror == Some(plan[i].mirror)
            },
        decreases plan.len() - index,
    ))]
    while index < plan.len() {
        let segment = &plan[index];
        segments.push(DirectSegment {
            input: segment.input.start + column..segment.input.end + column,
            primary: segment.primary.start..segment.primary.end,
            mirror: Some(segment.mirror.start..segment.mirror.end),
        });
        index += 1;
    }
    Ok(segments)
}

/// Plans the committed segments of a direct append of `count` tokens at `start`.
///
/// The tokens are read from input columns `[column, column + count)`; the
/// caller's layout guarantees that range is representable, and a ring width
/// is positive. An empty append writes nothing and checks nothing, matching
/// the runtime's early return.
///
/// Proved: the write is planned exactly when `direct_write_accepted` holds,
/// the plan is exactly `direct_segments_correct` (full storage writes slots
/// `[start, start + count)`, a ring writes column `j` to slot
/// `(start + j - column) mod W`), and a rejection is the first failed check.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        window matches Some(width) ==> width > 0,
        column + count <= usize::MAX,
    ensures
        result is Ok <==> direct_write_accepted(storage, start, count, window),
        result matches Ok(segments) ==>
            direct_segments_correct(segments@, storage, start, count, column, window),
        result matches Err(violation) ==>
            direct_violation_exact(storage, start, count, window, violation),
))]
pub(crate) fn direct_write_segments(
    storage: KvCacheStorage,
    start: usize,
    count: usize,
    column: usize,
    window: Option<usize>,
) -> Result<Vec<DirectSegment>, DirectWriteViolation> {
    let mut segments: Vec<DirectSegment> = Vec::new();
    if count == 0 {
        return Ok(segments);
    }
    match storage {
        KvCacheStorage::Full => {
            let Some(end) = start.checked_add(count) else {
                return Err(DirectWriteViolation::Ring(
                    RingPlanError::PositionOverflow {
                        start_position: start,
                        token_count: count,
                    },
                ));
            };
            segments.push(DirectSegment {
                input: column..column + count,
                primary: start..end,
                mirror: None,
            });
            Ok(segments)
        }
        KvCacheStorage::SlidingRing => {
            let Some(width) = window else {
                return Err(DirectWriteViolation::WindowRequired);
            };
            match ring_direct_segments(start, count, column, width) {
                Ok(segments) => Ok(segments),
                Err(error) => Err(DirectWriteViolation::Ring(error)),
            }
        }
        KvCacheStorage::MirroredSlidingRing => {
            let Some(width) = window else {
                return Err(DirectWriteViolation::WindowRequired);
            };
            match mirrored_direct_segments(start, count, column, width) {
                Ok(segments) => Ok(segments),
                Err(error) => Err(DirectWriteViolation::Ring(error)),
            }
        }
    }
}

/// Plans one run of a batched committed write.
///
/// Every row of `run` shares its start, column, and count, so one set of
/// segments serves the whole run. Input row `b - source_offset` feeds
/// destination row `b`; the caller passes runs of active rows at or after
/// `source_offset` (a batched input uses offset zero, a single-row input
/// its row, the only active one) whose columns fit the input block.
///
/// Proved: the run is planned exactly when `direct_write_accepted` holds;
/// it then writes rows `[first_row, end_row)` from input rows shifted down by
/// `source_offset`, with exactly the `direct_segments_correct` segments, and
/// a rejection is the first failed check.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        source_offset <= run.first_row,
        run.first_row <= run.end_row,
        run.column + run.count <= usize::MAX,
        window matches Some(width) ==> width > 0,
    ensures
        result is Ok <==> direct_write_accepted(storage, run.start, run.count, window),
        result matches Ok(plan) ==> {
            &&& plan.rows.start == run.first_row
            &&& plan.rows.end == run.end_row
            &&& plan.source_rows.start == run.first_row - source_offset
            &&& plan.source_rows.end == run.end_row - source_offset
            &&& direct_segments_correct(plan.segments@, storage, run.start, run.count,
                run.column, window)
        },
        result matches Err(violation) ==>
            direct_violation_exact(storage, run.start, run.count, window, violation),
))]
pub(crate) fn plan_direct_run(
    run: RowRun,
    source_offset: usize,
    storage: KvCacheStorage,
    window: Option<usize>,
) -> Result<DirectRun, DirectWriteViolation> {
    let segments = direct_write_segments(storage, run.start, run.count, run.column, window)?;
    Ok(DirectRun {
        rows: run.first_row..run.end_row,
        source_rows: run.first_row - source_offset..run.end_row - source_offset,
        segments,
    })
}

/// Plans the committed writes of one row's accepted staging prefix.
///
/// Row `b` staged `proposed` tokens after its committed `baseline` and
/// accepts the first `accepted`; they are read from staging offsets
/// `[0, accepted)`. A ring width is positive.
///
/// Proved: the commit is planned exactly when `commit_write_accepted`
/// holds, the plan is exactly the direct write of the accepted prefix at
/// `baseline` from column zero (`direct_segments_correct`), and a rejection
/// is the first failed check (`commit_violation_exact`) in the order the
/// runtime always used.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        window matches Some(width) ==> width > 0,
    ensures
        result is Ok <==>
            commit_write_accepted(storage, baseline, proposed, accepted, window, capacity),
        result matches Ok(segments) ==>
            direct_segments_correct(segments@, storage, baseline, accepted, 0, window),
        result matches Err(violation) ==> commit_violation_exact(storage, baseline, proposed,
            accepted, window, capacity, violation),
))]
pub(crate) fn commit_write_segments(
    storage: KvCacheStorage,
    baseline: usize,
    proposed: usize,
    accepted: usize,
    window: Option<usize>,
    capacity: usize,
) -> Result<Vec<DirectSegment>, DirectWriteViolation> {
    if baseline.checked_add(proposed).is_none() {
        return Err(DirectWriteViolation::Ring(
            RingPlanError::PositionOverflow {
                start_position: baseline,
                token_count: proposed,
            },
        ));
    }
    let width = match storage {
        KvCacheStorage::Full => {
            let end = match write_end(baseline, accepted, capacity) {
                Ok(end) => end,
                Err(violation) => return Err(DirectWriteViolation::WriteEnd(violation)),
            };
            let mut segments: Vec<DirectSegment> = Vec::new();
            if accepted > 0 {
                segments.push(DirectSegment {
                    input: 0..accepted,
                    primary: baseline..end,
                    mirror: None,
                });
            }
            return Ok(segments);
        }
        KvCacheStorage::SlidingRing | KvCacheStorage::MirroredSlidingRing => match window {
            Some(width) => width,
            None => return Err(DirectWriteViolation::WindowRequired),
        },
    };
    if accepted > proposed {
        return Err(DirectWriteViolation::Ring(
            RingPlanError::CommitPrefixExceedsStaged {
                accepted,
                staged: proposed,
            },
        ));
    }
    let planned = match storage {
        KvCacheStorage::MirroredSlidingRing => {
            mirrored_direct_segments(baseline, accepted, 0, width)
        }
        KvCacheStorage::Full | KvCacheStorage::SlidingRing => {
            ring_direct_segments(baseline, accepted, 0, width)
        }
    };
    match planned {
        Ok(segments) => Ok(segments),
        Err(error) => Err(DirectWriteViolation::Ring(error)),
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
