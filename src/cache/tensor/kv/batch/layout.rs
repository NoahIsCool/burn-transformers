//! Verified layouts of batched KV writes and proposal staging.
//!
//! A batched step appends `n_b` tokens to row `b` at absolute positions
//! `[s_b, s_b + n_b)`, reading them from columns `[c_b, c_b + n_b)` of a
//! `[batch, heads, block, dim]` input. A layout is valid when every column
//! range fits the block (`c_b + n_b <= block`) and every position range is
//! representable (`s_b + n_b <= usize::MAX`); right padding uses `c_b = 0`
//! and left padding `c_b = block - n_b`.
//!
//! During speculation, row `b` with committed baseline `C_b` and `q_b`
//! proposals already staged must start its next proposal at `C_b + q_b`,
//! may stage at most the staging capacity, and must stay within the logical
//! capacity. Separate staging is linear: proposal offset `q_b + j` holds the
//! row's `j`-th new token, so adjacent rows with equal `(q_b, c_b, n_b)` form
//! one run written by a single tensor assignment.
//!
//! Every function here is checked in place by Verus through
//! `verus/cache.rs`; the tensor assignments driven by these plans remain
//! covered by the batched KV tests.

use core::ops::Range;

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

use crate::cache::rows::row_runs;
#[cfg(verus_keep_ghost)]
use crate::cache::rows::{consistent, covered};
use crate::cache::tensor::plan::{WriteEndViolation, staging_end, write_end};

#[cfg(verus_keep_ghost)]
verus! {

/// Row `b` of a batched append fits its input block and the position space.
pub open spec fn append_row_fits(start: usize, column: usize, count: usize, block: usize) -> bool {
    column + count <= block && start + count <= usize::MAX
}

/// Every row of a batched append fits its block and the position space.
pub open spec fn append_layout_fits(
    starts: Seq<usize>,
    columns: Seq<usize>,
    counts: Seq<usize>,
    block: usize,
) -> bool {
    &&& starts.len() == counts.len()
    &&& columns.len() == counts.len()
    &&& forall|r: int| 0 <= r < counts.len() ==>
        append_row_fits(#[trigger] starts[r], columns[r], counts[r], block)
}

/// The proposal cursor `baseline + staged`, saturated at `usize::MAX`.
pub open spec fn proposal_cursor(baseline: usize, staged: usize) -> int {
    if baseline + staged <= usize::MAX { baseline + staged } else { usize::MAX as int }
}

/// Row `row` of a staging write passes the cursor, staging, and capacity checks.
pub open spec fn stage_row_ok(
    starts: Seq<usize>,
    counts: Seq<usize>,
    staged: Seq<usize>,
    baseline: Option<&[usize]>,
    staging_capacity: usize,
    capacity: usize,
    row: int,
) -> bool {
    counts[row] == 0 || {
        &&& match baseline {
            Some(base) => starts[row] as int == proposal_cursor(base@[row], staged[row]),
            None => true,
        }
        &&& staged[row] + counts[row] <= staging_capacity
        &&& starts[row] + counts[row] <= capacity
    }
}

/// `updates` lists exactly the active rows below `rows`, in order, with their new staged lengths.
pub open spec fn stage_updates_exact(
    updates: Seq<(usize, usize)>,
    counts: Seq<usize>,
    staged: Seq<usize>,
    rows: int,
) -> bool {
    &&& forall|k: int| 0 <= k < updates.len() ==> {
        &&& (#[trigger] updates[k]).0 < rows
        &&& counts[updates[k].0 as int] > 0
        &&& updates[k].1 == staged[updates[k].0 as int] + counts[updates[k].0 as int]
    }
    &&& forall|j: int, k: int| 0 <= j < k < updates.len() ==>
        (#[trigger] updates[j]).0 < (#[trigger] updates[k]).0
    &&& forall|r: int| 0 <= r < rows && #[trigger] counts[r] > 0 ==>
        exists|k: int| 0 <= k < updates.len() && (#[trigger] updates[k]).0 == r
}

/// Flat entry `index` of a `[rows, block]` table addresses row `index / block`, column `index % block`.
pub open spec fn token_entry_correct(
    positions: Seq<usize>,
    valid: Seq<bool>,
    starts: Seq<usize>,
    columns: Seq<usize>,
    counts: Seq<usize>,
    block: int,
    index: int,
) -> bool {
    let row = index / block;
    let slot = index % block;
    let inside = columns[row] <= slot < columns[row] + counts[row];
    &&& valid[index] == inside
    &&& positions[index] == if inside { starts[row] + (slot - columns[row]) } else { 0 }
}

} // verus!

/// Integer reason a batched append layout is rejected.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppendViolation {
    /// A per-row vector does not have one entry per counted row.
    CountMismatch {
        /// Number of token counts.
        expected: usize,
        /// Length of the mismatched vector.
        actual: usize,
    },
    /// A row's column range leaves the input block.
    OutsideBlock {
        /// Affected row.
        sequence: usize,
        /// First input column of the row.
        column: usize,
        /// New tokens of the row.
        count: usize,
    },
    /// A row's absolute end `start + count` overflows `usize`.
    PositionOverflow {
        /// First absolute position of the row.
        start: usize,
        /// New tokens of the row.
        tokens: usize,
    },
}

/// Validates a batched append layout before any tensor is sliced.
///
/// Proved: the layout is accepted exactly when all three vectors have one
/// entry per row and every row fits its block and the position space; a
/// rejection reports the first mismatched vector (`starts` before `columns`)
/// or the first offending row, checking the block before the position.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result is Ok <==> append_layout_fits(starts@, columns@, counts@, block),
        result matches Err(violation) ==> match violation {
            AppendViolation::CountMismatch { expected, actual } => {
                &&& expected == counts.len()
                &&& starts.len() != counts.len() ==> actual == starts.len()
                &&& starts.len() == counts.len() ==>
                    columns.len() != counts.len() && actual == columns.len()
            },
            AppendViolation::OutsideBlock { sequence, column, count } => {
                &&& starts.len() == counts.len() && columns.len() == counts.len()
                &&& sequence < counts.len()
                &&& column == columns[sequence as int]
                &&& count == counts[sequence as int]
                &&& column + count > block
                &&& forall|r: int| 0 <= r < sequence ==>
                    append_row_fits(#[trigger] starts[r], columns[r], counts[r], block)
            },
            AppendViolation::PositionOverflow { start, tokens } => exists|sequence: int| {
                &&& starts.len() == counts.len() && columns.len() == counts.len()
                &&& 0 <= sequence < counts.len()
                &&& start == #[trigger] starts[sequence]
                &&& tokens == counts[sequence]
                &&& columns[sequence] + counts[sequence] <= block
                &&& start + tokens > usize::MAX
                &&& forall|r: int| 0 <= r < sequence ==>
                    append_row_fits(#[trigger] starts[r], columns[r], counts[r], block)
            },
        },
))]
pub(crate) fn validate_append_layout(
    starts: &[usize],
    columns: &[usize],
    counts: &[usize],
    block: usize,
) -> Result<(), AppendViolation> {
    if starts.len() != counts.len() {
        return Err(AppendViolation::CountMismatch {
            expected: counts.len(),
            actual: starts.len(),
        });
    }
    if columns.len() != counts.len() {
        return Err(AppendViolation::CountMismatch {
            expected: counts.len(),
            actual: columns.len(),
        });
    }
    let mut sequence = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            sequence <= counts.len(),
            starts.len() == counts.len(),
            columns.len() == counts.len(),
            forall|r: int| 0 <= r < sequence ==>
                append_row_fits(#[trigger] starts[r], columns[r], counts[r], block),
        decreases counts.len() - sequence,
    ))]
    while sequence < counts.len() {
        let column = columns[sequence];
        let count = counts[sequence];
        let fits_block = match column.checked_add(count) {
            Some(end) => end <= block,
            None => false,
        };
        if !fits_block {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(!append_row_fits(starts@[sequence as int], columns@[sequence as int],
                    counts@[sequence as int], block));
            }
            return Err(AppendViolation::OutsideBlock {
                sequence,
                column,
                count,
            });
        }
        let start = starts[sequence];
        if start.checked_add(count).is_none() {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(!append_row_fits(starts@[sequence as int], columns@[sequence as int],
                    counts@[sequence as int], block));
            }
            return Err(AppendViolation::PositionOverflow {
                start,
                tokens: count,
            });
        }
        sequence += 1;
    }
    Ok(())
}

/// Places each row's tokens in the last `counts[b]` columns of a `block`-wide input.
///
/// Proved: succeeds exactly when every count fits the block, then returns
/// `block - counts[b]` for every row; otherwise names the first row longer
/// than the block.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result is Ok <==> forall|r: int| 0 <= r < counts.len() ==> #[trigger] counts[r] <= block,
        result matches Ok(columns) ==> {
            &&& columns.len() == counts.len()
            &&& forall|r: int| 0 <= r < counts.len() ==>
                #[trigger] columns[r] == block - counts[r]
        },
        result matches Err(row) ==> {
            &&& row < counts.len()
            &&& counts[row as int] > block
            &&& forall|r: int| 0 <= r < row ==> #[trigger] counts[r] <= block
        },
))]
pub(crate) fn left_padded_columns(counts: &[usize], block: usize) -> Result<Vec<usize>, usize> {
    let mut columns: Vec<usize> = Vec::with_capacity(counts.len());
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            columns.len() == row,
            forall|r: int| 0 <= r < row ==> #[trigger] counts[r] <= block,
            forall|r: int| 0 <= r < row ==> #[trigger] columns[r] == block - counts[r],
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] > block {
            return Err(row);
        }
        columns.push(block - counts[row]);
        row += 1;
    }
    Ok(columns)
}

/// Returns each row's exclusive absolute end `start + count`.
///
/// The caller passes a layout accepted by [`validate_append_layout`], so
/// every sum is representable. Rows beyond the shorter input are omitted,
/// matching a zipped scan.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(ends =>
    requires
        forall|r: int| 0 <= r < starts.len() && r < counts.len() ==>
            #[trigger] starts[r] + counts[r] <= usize::MAX,
    ensures
        ends.len() == if starts.len() <= counts.len() { starts.len() } else { counts.len() },
        forall|r: int| 0 <= r < ends.len() ==> #[trigger] ends[r] == starts[r] + counts[r],
))]
pub(crate) fn append_ends(starts: &[usize], counts: &[usize]) -> Vec<usize> {
    let rows = if starts.len() <= counts.len() {
        starts.len()
    } else {
        counts.len()
    };
    let mut ends: Vec<usize> = Vec::with_capacity(rows);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows <= starts.len(),
            rows <= counts.len(),
            forall|r: int| 0 <= r < starts.len() && r < counts.len() ==>
                #[trigger] starts[r] + counts[r] <= usize::MAX,
            ends.len() == row,
            forall|r: int| 0 <= r < row ==> #[trigger] ends[r] == starts[r] + counts[r],
        decreases rows - row,
    ))]
    while row < rows {
        ends.push(starts[row] + counts[row]);
        row += 1;
    }
    ends
}

/// Returns the proposal cursor `baseline + staged`, saturated at `usize::MAX`.
///
/// Proved: exactly the saturated sum.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(cursor =>
    ensures
        cursor as int == proposal_cursor(baseline, staged),
))]
pub(crate) fn proposal_cursor_of(baseline: usize, staged: usize) -> usize {
    baseline.saturating_add(staged)
}

/// Returns every row's next proposal start: its baseline (zero if unknown) plus its staged length.
///
/// The caller passes one staged length (and baseline) per row.
///
/// Proved: row `b` starts exactly at the saturated `baseline_b + staged_b`,
/// or at `staged_b` when the baseline is unknown.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(starts =>
    requires
        rows <= staged.len(),
        baseline matches Some(base) ==> rows <= base.len(),
    ensures
        starts.len() == rows,
        forall|r: int| 0 <= r < rows ==> #[trigger] starts[r] as int == match baseline {
            Some(base) => proposal_cursor(base@[r], staged[r]),
            None => staged[r] as int,
        },
))]
pub(crate) fn proposal_starts(
    staged: &[usize],
    baseline: Option<&[usize]>,
    rows: usize,
) -> Vec<usize> {
    let mut starts: Vec<usize> = Vec::with_capacity(rows);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows <= staged.len(),
            baseline matches Some(base) ==> rows <= base.len(),
            starts.len() == row,
            forall|r: int| 0 <= r < row ==> #[trigger] starts[r] as int == match baseline {
                Some(base) => proposal_cursor(base@[r], staged[r]),
                None => staged[r] as int,
            },
        decreases rows - row,
    ))]
    while row < rows {
        let start = match baseline {
            Some(base) => proposal_cursor_of(base[row], staged[row]),
            None => staged[row],
        };
        starts.push(start);
        row += 1;
    }
    starts
}

/// Returns `rows` token counts that are zero except `count` at row `sequence`.
///
/// Proved: exactly that one-hot vector, so the only possibly active row is
/// `sequence` and every run of a single-row staging write starts there.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(counts =>
    requires
        sequence < rows,
    ensures
        counts.len() == rows,
        counts[sequence as int] == count,
        forall|r: int| 0 <= r < rows && r != sequence ==> #[trigger] counts[r] == 0,
))]
pub(crate) fn one_hot_counts(rows: usize, sequence: usize, count: usize) -> Vec<usize> {
    let mut counts: Vec<usize> = Vec::with_capacity(rows);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            counts.len() == row,
            forall|r: int| 0 <= r < row ==>
                #[trigger] counts[r] == if r == sequence { count } else { 0 },
        decreases rows - row,
    ))]
    while row < rows {
        counts.push(if row == sequence { count } else { 0 });
        row += 1;
    }
    counts
}

/// Integer reason a proposal staging write is rejected.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StageViolation {
    /// A row's proposal does not start at its working cursor.
    CursorMismatch {
        /// Affected row.
        sequence: usize,
        /// Working cursor `baseline + staged`, saturated at `usize::MAX`.
        expected: usize,
        /// Requested proposal start.
        actual: usize,
    },
    /// The row's staged length would exceed the staging capacity.
    StagingCapacityExceeded {
        /// Requested staged length, saturated at `usize::MAX`.
        requested: usize,
    },
    /// The row's absolute write leaves the logical capacity.
    WriteEnd {
        /// First absolute position of the write.
        start: usize,
        /// New tokens of the row.
        tokens: usize,
        /// Overflow or capacity violation of the write end.
        violation: WriteEndViolation,
    },
}

/// Plans the staged-length updates of one proposal staging write.
///
/// Row `b` appends `counts[b]` proposals at absolute `starts[b]` after
/// `staged[b]` already staged ones; rows with no new tokens are skipped. When
/// the transaction baseline is known, each proposal must start at
/// `baseline[b] + staged[b]`. Rows beyond the shorter of `starts` and `counts`
/// are not examined, matching a zipped scan; the caller passes one staged
/// length (and baseline) per row.
///
/// Proved: the write is accepted exactly when every active row passes the
/// cursor, staging-capacity, and logical-capacity checks; the updates then
/// list exactly the active rows in increasing order with their new staged
/// lengths `staged[b] + counts[b]`. A rejection names the first failing
/// check of the first failing active row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        starts.len() <= staged.len() || counts.len() <= staged.len(),
        baseline matches Some(base) ==>
            starts.len() <= base.len() || counts.len() <= base.len(),
    ensures
        ({
            let rows = if starts.len() <= counts.len() { starts.len() } else { counts.len() };
            &&& result is Ok <==> forall|r: int| 0 <= r < rows ==>
                #[trigger] stage_row_ok(starts@, counts@, staged@, baseline, staging_capacity,
                    capacity, r)
            &&& result matches Ok(updates) ==>
                stage_updates_exact(updates@, counts@, staged@, rows as int)
            &&& result matches Err(violation) ==> exists|row: int| {
                &&& 0 <= row < rows
                &&& row < staged.len()
                &&& counts[row] > 0
                &&& forall|r: int| 0 <= r < row ==>
                    #[trigger] stage_row_ok(starts@, counts@, staged@, baseline,
                        staging_capacity, capacity, r)
                &&& !stage_row_ok(starts@, counts@, staged@, baseline, staging_capacity,
                    capacity, row)
                &&& match violation {
                    StageViolation::CursorMismatch { sequence, expected, actual } => {
                        &&& sequence == row
                        &&& baseline matches Some(base)
                            && expected as int == proposal_cursor(base@[row], staged[row])
                        &&& actual == starts[row]
                        &&& actual != expected
                    },
                    StageViolation::StagingCapacityExceeded { requested } => {
                        &&& requested as int == proposal_cursor(staged[row], counts[row])
                        &&& staged[row] + counts[row] > staging_capacity
                    },
                    StageViolation::WriteEnd { start, tokens, violation } => {
                        &&& start == starts[row]
                        &&& tokens == counts[row]
                        &&& staged[row] + counts[row] <= staging_capacity
                        &&& start + tokens > capacity
                    },
                }
            }
        }),
))]
pub(crate) fn plan_stage_updates(
    starts: &[usize],
    counts: &[usize],
    staged: &[usize],
    baseline: Option<&[usize]>,
    staging_capacity: usize,
    capacity: usize,
) -> Result<Vec<(usize, usize)>, StageViolation> {
    let rows = if starts.len() <= counts.len() {
        starts.len()
    } else {
        counts.len()
    };
    let mut updates: Vec<(usize, usize)> = Vec::new();
    let mut sequence = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            sequence <= rows,
            rows <= starts.len(),
            rows <= counts.len(),
            rows <= staged.len(),
            baseline matches Some(base) ==> rows <= base.len(),
            forall|r: int| 0 <= r < sequence ==>
                #[trigger] stage_row_ok(starts@, counts@, staged@, baseline, staging_capacity,
                    capacity, r),
            stage_updates_exact(updates@, counts@, staged@, sequence as int),
        decreases rows - sequence,
    ))]
    while sequence < rows {
        let start = starts[sequence];
        let count = counts[sequence];
        if count == 0 {
            sequence += 1;
            continue;
        }
        if let Some(base) = baseline {
            let expected = proposal_cursor_of(base[sequence], staged[sequence]);
            if start != expected {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(!stage_row_ok(starts@, counts@, staged@, baseline, staging_capacity,
                        capacity, sequence as int));
                }
                return Err(StageViolation::CursorMismatch {
                    sequence,
                    expected,
                    actual: start,
                });
            }
        }
        let end = match staging_end(staged[sequence], count, staging_capacity) {
            Ok(end) => end,
            Err(requested) => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(!stage_row_ok(starts@, counts@, staged@, baseline, staging_capacity,
                        capacity, sequence as int));
                }
                return Err(StageViolation::StagingCapacityExceeded { requested });
            }
        };
        if let Err(violation) = write_end(start, count, capacity) {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(!stage_row_ok(starts@, counts@, staged@, baseline, staging_capacity,
                    capacity, sequence as int));
            }
            return Err(StageViolation::WriteEnd {
                start,
                tokens: count,
                violation,
            });
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            let old_updates = updates@;
            let new_updates = old_updates.push((sequence, end));
            assert(new_updates[old_updates.len() as int] == (sequence, end));
            assert forall|r: int| 0 <= r < sequence + 1 && #[trigger] counts[r] > 0 implies
                exists|k: int| 0 <= k < new_updates.len() && (#[trigger] new_updates[k]).0 == r
            by {
                if r == sequence {
                    assert(new_updates[old_updates.len() as int].0 == r);
                } else {
                    let k = choose|k: int| 0 <= k < old_updates.len()
                        && (#[trigger] old_updates[k]).0 == r;
                    assert(new_updates[k] == old_updates[k]);
                }
            }
        }
        updates.push((sequence, end));
        sequence += 1;
    }
    Ok(updates)
}

/// One run of rows staged linearly by a single tensor assignment.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LinearRun {
    /// Destination rows written together.
    pub(crate) rows: Range<usize>,
    /// Matching rows of the input tensor (`rows` shifted by the source offset).
    pub(crate) source_rows: Range<usize>,
    /// Input columns read from every row of the run.
    pub(crate) input: Range<usize>,
    /// Linear staging offsets written in every row of the run.
    pub(crate) destination: Range<usize>,
}

/// Groups a separate-staging write into runs over linear proposal offsets.
///
/// Row `b` writes its `n_b` new tokens from columns `[c_b, c_b + n_b)` to
/// proposal offsets `[q_b, q_b + n_b)`, where `q_b = staged[b]`; input row
/// `b - source_offset` feeds destination row `b`. The caller passes a layout
/// whose active rows fit the block and the staging capacity and lie at or
/// after `source_offset`, which [`validate_append_layout`] and
/// [`plan_stage_updates`] established (a single-row input has exactly one
/// active row, at `source_offset`).
///
/// Proved: the runs are non-empty, ordered, and disjoint, cover exactly the
/// rows with new tokens, and every row of a run reads exactly its own column
/// range and writes exactly its own proposal offsets.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(runs =>
    requires
        staged.len() == counts.len(),
        columns.len() == counts.len(),
        forall|r: int| 0 <= r < counts.len() && #[trigger] counts[r] > 0 ==> {
            &&& staged[r] + counts[r] <= usize::MAX
            &&& columns[r] + counts[r] <= usize::MAX
            &&& r >= source_offset
        },
    ensures
        forall|i: int| 0 <= i < runs.len() ==> {
            &&& (#[trigger] runs[i]).rows.start < runs[i].rows.end <= counts.len()
            &&& runs[i].source_rows.start == runs[i].rows.start - source_offset
            &&& runs[i].source_rows.end == runs[i].rows.end - source_offset
            &&& forall|r: int| runs[i].rows.start <= r < runs[i].rows.end ==> {
                &&& #[trigger] counts[r] > 0
                &&& runs[i].input.start == columns[r]
                &&& runs[i].input.end == columns[r] + counts[r]
                &&& runs[i].destination.start == staged[r]
                &&& runs[i].destination.end == staged[r] + counts[r]
            }
        },
        forall|i: int, j: int| 0 <= i < j < runs.len() ==>
            (#[trigger] runs[i]).rows.end <= (#[trigger] runs[j]).rows.start,
        forall|r: int| 0 <= r < counts.len() ==>
            (#[trigger] counts[r] > 0 <==> exists|i: int| 0 <= i < runs.len()
                && runs[i].rows.start <= r < #[trigger] runs[i].rows.end),
))]
pub(crate) fn linear_stage_runs(
    staged: &[usize],
    columns: &[usize],
    counts: &[usize],
    source_offset: usize,
) -> Vec<LinearRun> {
    let groups = row_runs(staged, columns, counts);
    let mut runs: Vec<LinearRun> = Vec::with_capacity(groups.len());
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= groups.len(),
            runs.len() == index,
            staged.len() == counts.len(),
            columns.len() == counts.len(),
            forall|r: int| 0 <= r < counts.len() && #[trigger] counts[r] > 0 ==> {
                &&& staged[r] + counts[r] <= usize::MAX
                &&& columns[r] + counts[r] <= usize::MAX
                &&& r >= source_offset
            },
            forall|i: int| 0 <= i < groups.len() ==>
                consistent(#[trigger] groups[i], staged@, columns@, counts@),
            forall|i: int, j: int| 0 <= i < j < groups.len() ==>
                #[trigger] groups[i].end_row <= #[trigger] groups[j].first_row,
            forall|r: int| 0 <= r < counts.len() ==>
                (#[trigger] counts[r] > 0 <==> covered(groups@, r)),
            forall|i: int| 0 <= i < index ==> {
                &&& (#[trigger] runs[i]).rows.start == groups[i].first_row
                &&& runs[i].rows.end == groups[i].end_row
                &&& runs[i].source_rows.start == groups[i].first_row - source_offset
                &&& runs[i].source_rows.end == groups[i].end_row - source_offset
                &&& runs[i].input.start == groups[i].column
                &&& runs[i].input.end == groups[i].column + groups[i].count
                &&& runs[i].destination.start == groups[i].start
                &&& runs[i].destination.end == groups[i].start + groups[i].count
            },
        decreases groups.len() - index,
    ))]
    while index < groups.len() {
        let group = groups[index];
        #[cfg(verus_keep_ghost)]
        proof! {
            // The run's first row is active, so its parameters fit and it
            // lies at or after the source offset.
            assert(counts[group.first_row as int] > 0);
        }
        runs.push(LinearRun {
            rows: group.first_row..group.end_row,
            source_rows: group.first_row - source_offset..group.end_row - source_offset,
            input: group.column..group.column + group.count,
            destination: group.start..group.start + group.count,
        });
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|r: int| 0 <= r < counts.len() implies
            (#[trigger] counts[r] > 0 <==> exists|i: int| 0 <= i < runs.len()
                && runs[i].rows.start <= r < #[trigger] runs[i].rows.end)
        by {
            if counts[r] > 0 {
                let i = choose|i: int| 0 <= i < groups.len()
                    && groups[i].first_row <= r < #[trigger] groups[i].end_row;
                assert(runs[i].rows.start <= r < runs[i].rows.end);
            }
            if exists|i: int| 0 <= i < runs.len()
                && runs[i].rows.start <= r < #[trigger] runs[i].rows.end {
                let i = choose|i: int| 0 <= i < runs.len()
                    && runs[i].rows.start <= r < #[trigger] runs[i].rows.end;
                assert(groups[i].first_row <= r < groups[i].end_row);
                assert(covered(groups@, r));
            }
        }
    }
    runs
}

/// Describes a batched step's new tokens as `[rows, block]` positions and validity.
///
/// Column `c` of row `b` is valid exactly when it lies in the row's token
/// range `[c_b, c_b + n_b)` and then holds absolute position
/// `s_b + (c - c_b)`; invalid entries hold position zero. The caller passes a
/// layout accepted by [`validate_append_layout`]. Entries are row-major.
///
/// Proved: both tables have `rows * block` entries and entry `b * block + c`
/// is exactly as described.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(table =>
    requires
        append_layout_fits(starts@, columns@, counts@, block),
    ensures
        table.0.len() == counts.len() * block,
        table.1.len() == counts.len() * block,
        forall|index: int| 0 <= index < counts.len() * block ==>
            #[trigger] token_entry_correct(table.0@, table.1@, starts@, columns@, counts@,
                block as int, index),
))]
pub(crate) fn token_segment_table(
    starts: &[usize],
    columns: &[usize],
    counts: &[usize],
    block: usize,
) -> (Vec<usize>, Vec<bool>) {
    let mut positions: Vec<usize> = Vec::new();
    let mut valid: Vec<bool> = Vec::new();
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            append_layout_fits(starts@, columns@, counts@, block),
            positions.len() == row * block,
            valid.len() == row * block,
            forall|index: int| 0 <= index < row * block ==>
                #[trigger] token_entry_correct(positions@, valid@, starts@, columns@, counts@,
                    block as int, index),
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        let start = starts[row];
        let column = columns[row];
        let count = counts[row];
        let mut slot = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                row < counts.len(),
                slot <= block,
                append_layout_fits(starts@, columns@, counts@, block),
                start == starts[row as int],
                column == columns[row as int],
                count == counts[row as int],
                positions.len() == row * block + slot,
                valid.len() == row * block + slot,
                forall|index: int| 0 <= index < row * block + slot ==>
                    #[trigger] token_entry_correct(positions@, valid@, starts@, columns@,
                        counts@, block as int, index),
            decreases block - slot,
        ))]
        while slot < block {
            let inside = column <= slot && slot - column < count;
            #[cfg(verus_keep_ghost)]
            proof! {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
                    row * block + slot,
                    block as int,
                    row as int,
                    slot as int,
                );
                assert(append_row_fits(starts@[row as int], columns@[row as int],
                    counts@[row as int], block));
            }
            let position = if inside { start + (slot - column) } else { 0 };
            #[cfg(verus_keep_ghost)]
            proof! {
                let index = row * block + slot;
                let old_positions = positions@;
                let old_valid = valid@;
                assert forall|i: int| 0 <= i < index + 1 implies
                    #[trigger] token_entry_correct(old_positions.push(position),
                        old_valid.push(inside), starts@, columns@, counts@, block as int, i)
                by {
                    if i < index {
                        assert(token_entry_correct(old_positions, old_valid, starts@, columns@,
                            counts@, block as int, i));
                    }
                }
            }
            positions.push(position);
            valid.push(inside);
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row + 1) * block == row * block + block) by (nonlinear_arith);
        }
        row += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(row * block == counts.len() * block);
    }
    (positions, valid)
}

#[cfg(test)]
#[path = "layout_tests.rs"]
mod tests;
