//! Verified retention arithmetic for sliding-window and recurrent caches.
//!
//! A sliding ring of width `W` that has received absolute positions
//! `[0, L)` still stores exactly the positions `p` with `p + W >= L`; older
//! positions were overwritten by `p + W`, `p + 2W`, …. Attention at the end of
//! a sequence of length `L` sees the window `[L - min(L, W), L)`.
//!
//! This module answers two questions with pure integer functions:
//!
//! * **Rewind safety.** After a ring has received `[0, from)`, rewinding to
//!   `to <= from` exposes the window `[to - min(to, W), to)`. The rewind is
//!   safe iff every exposed position is still stored, which holds iff
//!   `to == from`, `to == 0` (nothing is exposed), or `from <= W` (nothing was
//!   ever overwritten). Recurrent terminal states are not invertible, so they
//!   can only "rewind" to `to == from` or restart from `to == 0`.
//! * **Working windows.** During speculation, `C` committed tokens live in the
//!   ring and `P` proposal tokens live in separate linear staging. The visible
//!   window of the working sequence `[C + P - min(C + P, W), C + P)` keeps the
//!   newest `min(P, W)` proposals and then the newest committed tokens that
//!   still fit.
//!
//! # Verified behavior
//!
//! `verus/cache.rs` checks these functions in place. The rewind predicates are
//! proved sound *and* complete against the eviction model above, and the
//! working window is proved to partition the visible suffix contiguously. All
//! arithmetic is proved free of overflow and underflow.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Position `p` is still stored after a ring of width `window` received `[0, written)`.
pub open spec fn ring_stores(p: int, written: int, window: int) -> bool {
    p + window >= written
}

/// First absolute position visible to attention at the end of a length-`length` sequence.
pub open spec fn visible_start(length: int, window: int) -> int {
    if length >= window { length - window } else { 0 }
}

/// Every position visible after rewinding to `to` is still stored in the ring.
pub open spec fn ring_rewind_safe(from: int, to: int, window: int) -> bool {
    forall|p: int| visible_start(to, window) <= p < to ==> #[trigger] ring_stores(p, from, window)
}

} // verus!

/// Returns whether rewinding a sliding ring from `from` to `to` keeps every
/// newly visible position available.
///
/// The caller has already rejected extensions (`to > from`). The result is
/// `to == from || to == 0 || from <= window`; the proof shows this closed form
/// is equivalent to "every position in the new window survived eviction".
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(safe =>
    requires
        window > 0,
        to <= from,
    ensures
        safe == ring_rewind_safe(from as int, to as int, window as int),
))]
pub(crate) fn ring_rewind_is_safe(from: usize, to: usize, window: usize) -> bool {
    let safe = to == from || to == 0 || from <= window;
    #[cfg(verus_keep_ghost)]
    proof! {
        if !safe {
            // The oldest newly visible position was overwritten.
            let oldest = visible_start(to as int, window as int);
            assert(oldest < to);
            assert(!ring_stores(oldest, from as int, window as int));
        }
    }
    safe
}

/// Returns whether a recurrent terminal state can represent length `to` after `from`.
///
/// Only the unchanged state (`to == from`) or the exact initial state
/// (`to == 0`, restored by zeroing) is reconstructible; intermediate states
/// would require inverting the recurrence.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(safe =>
    ensures
        safe == (to == from || to == 0),
))]
pub(crate) fn recurrent_rewind_is_exact(from: usize, to: usize) -> bool {
    to == from || to == 0
}

/// Returns the first row whose ring rewind would expose overwritten history.
///
/// Rows are checked in order so diagnostics name the lowest offending row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(first =>
    requires
        window > 0,
        current.len() == requested.len(),
        forall|i: int| 0 <= i < current.len() ==> #[trigger] requested[i] <= current[i],
    ensures
        first is None <==> forall|i: int| 0 <= i < current.len() ==>
            ring_rewind_safe(#[trigger] current[i] as int, requested[i] as int, window as int),
        first matches Some(row) ==> {
            &&& row < current.len()
            &&& !ring_rewind_safe(current[row as int] as int, requested[row as int] as int,
                window as int)
        },
))]
pub(crate) fn first_unsafe_ring_rewind(
    current: &[usize],
    requested: &[usize],
    window: usize,
) -> Option<usize> {
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= current.len(),
            current.len() == requested.len(),
            window > 0,
            forall|i: int| 0 <= i < current.len() ==> #[trigger] requested[i] <= current[i],
            forall|i: int| 0 <= i < row ==>
                ring_rewind_safe(#[trigger] current[i] as int, requested[i] as int,
                    window as int),
        decreases current.len() - row,
    ))]
    while row < current.len() {
        if !ring_rewind_is_safe(current[row], requested[row], window) {
            return Some(row);
        }
        row += 1;
    }
    None
}

/// Returns the first row whose recurrent state cannot represent its requested length.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(first =>
    requires
        current.len() == requested.len(),
    ensures
        first is None <==> forall|i: int| 0 <= i < current.len() ==>
            (#[trigger] requested[i] == current[i] || requested[i] == 0),
        first matches Some(row) ==> {
            &&& row < current.len()
            &&& requested[row as int] != current[row as int]
            &&& requested[row as int] != 0
        },
))]
pub(crate) fn first_inexact_recurrent_rewind(
    current: &[usize],
    requested: &[usize],
) -> Option<usize> {
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= current.len(),
            current.len() == requested.len(),
            forall|i: int| 0 <= i < row ==>
                (#[trigger] requested[i] == current[i] || requested[i] == 0),
        decreases current.len() - row,
    ))]
    while row < current.len() {
        if !recurrent_rewind_is_exact(current[row], requested[row]) {
            return Some(row);
        }
        row += 1;
    }
    None
}

/// Visible suffix of a speculative working sequence over ring storage.
///
/// With committed length `C`, staged proposal length `P`, and window `W`, the
/// visible absolute interval is `[retained_start, C + P)`. It consists of the
/// committed positions `[C - committed_keep, C)` (read from the ring) followed
/// by the proposal offsets `[staged_skip, P)` (read from linear staging).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkingWindow {
    /// First visible absolute position.
    pub(crate) retained_start: usize,
    /// Newest committed tokens kept, at most `min(C, W)`.
    pub(crate) committed_keep: usize,
    /// Oldest proposal offsets dropped because `P > W`.
    pub(crate) staged_skip: usize,
}

/// Plans the visible committed and staged parts of a ring working sequence.
///
/// `staged_keep = min(P, W)` proposals stay visible, then
/// `committed_keep = min(C, W - staged_keep)` committed tokens fill the rest of
/// the window. The committed part always lies inside the ring's own retained
/// suffix `[C - min(C, W), C)`, so it is readable without reconstruction.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    requires
        window > 0,
        committed + staged <= usize::MAX,
    ensures
        ({
            let working = committed + staged;
            let visible = if working <= window { working } else { window as int };
            let staged_keep = staged - plan.staged_skip;
            &&& plan.staged_skip <= staged
            &&& plan.committed_keep <= committed
            &&& plan.committed_keep <= window
            &&& plan.committed_keep + staged_keep == visible
            &&& plan.retained_start == working - visible
            &&& plan.committed_keep > 0 ==> plan.staged_skip == 0
                && plan.retained_start == committed - plan.committed_keep
            &&& plan.committed_keep == 0 ==> plan.retained_start == committed + plan.staged_skip
        }),
))]
pub(crate) fn ring_working_window(committed: usize, staged: usize, window: usize) -> WorkingWindow {
    let staged_keep = staged.min(window);
    let committed_keep = committed.min(window - staged_keep);
    WorkingWindow {
        retained_start: committed + staged - committed_keep - staged_keep,
        committed_keep,
        staged_skip: staged - staged_keep,
    }
}

#[cfg(test)]
#[path = "retention_tests.rs"]
mod tests;
