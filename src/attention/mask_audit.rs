//! Test-only per-thread record of the largest materialized exclusion plane.
//!
//! [`super::audit_plane`] reports every Boolean plane a mask builder is about
//! to materialize. Tests reset the record, run an operation, and read the
//! largest plane, which proves bounds such as "bounded attention never builds
//! a full `[query, key]` plane" without measuring process memory. The record
//! is thread-local because the test harness runs tests on separate threads,
//! and builders record on the calling thread at graph-construction time.

use core::cell::Cell;

std::thread_local! {
    /// Elements in the largest plane recorded on this thread since the last reset.
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

/// Records one materialized plane of `elements` elements.
pub(crate) fn record(elements: usize) {
    LARGEST.with(|largest| largest.set(largest.get().max(elements)));
}

/// Clears the record before a measured operation.
pub(crate) fn reset() {
    LARGEST.with(|largest| largest.set(0));
}

/// Elements in the largest plane recorded since the last reset.
pub(crate) fn largest() -> usize {
    LARGEST.with(Cell::get)
}
