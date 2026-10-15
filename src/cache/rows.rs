//! Verified grouping of batch rows into runs that share one tensor write.
//!
//! A batched cache write appends `n_b` tokens to row `b` at absolute position
//! `s_b`, reading them from columns `[c_b, c_b + n_b)` of a
//! `[batch, heads, block, dim]` input. Adjacent rows with identical
//! `(s_b, c_b, n_b)` map to identical physical slots, so one `slice_assign`
//! over the row range `[first_row, end_row)` writes all of them. A uniform
//! batch, such as single-token decode at equal lengths or an unpadded
//! prefill, becomes a single run; fully ragged rows become one run each,
//! which is exactly the per-row cost.
//!
//! # Verified behavior
//!
//! `verus/cache.rs` proves that the runs are non-empty, ordered, and
//! disjoint, that every row in a run carries the run's parameters, and that
//! the runs cover exactly the rows with `n_b > 0`; rows with no new tokens are
//! never written.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Adjacent rows written by the same assignments.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RowRun {
    /// First row of the run.
    pub(crate) first_row: usize,
    /// One past the last row of the run.
    pub(crate) end_row: usize,
    /// Absolute position of the first new token of every row in the run.
    pub(crate) start: usize,
    /// Input column of the first new token of every row in the run.
    pub(crate) column: usize,
    /// Number of new tokens of every row in the run; always positive.
    pub(crate) count: usize,
}

#[cfg(verus_keep_ghost)]
verus! {

/// Row `row` lies inside some run of `runs`.
pub(crate) open spec fn covered(runs: Seq<RowRun>, row: int) -> bool {
    exists|i: int| 0 <= i < runs.len() && runs[i].first_row <= row < #[trigger] runs[i].end_row
}

/// Run `run` is non-empty, inside the batch, and every row carries its parameters.
pub(crate) open spec fn consistent(
    run: RowRun,
    starts: Seq<usize>,
    columns: Seq<usize>,
    counts: Seq<usize>,
) -> bool {
    &&& run.first_row < run.end_row <= counts.len()
    &&& run.count > 0
    &&& forall|row: int| run.first_row <= row < run.end_row ==> #[trigger] starts[row] == run.start
    &&& forall|row: int| run.first_row <= row < run.end_row ==> #[trigger] columns[row] == run.column
    &&& forall|row: int| run.first_row <= row < run.end_row ==> #[trigger] counts[row] == run.count
}

} // verus!

/// Returns one past the last row of the run that begins at active row `first_row`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(end =>
    requires
        starts.len() == counts.len(),
        columns.len() == counts.len(),
        first_row < counts.len(),
    ensures
        first_row < end <= counts.len(),
        forall|r: int| first_row <= r < end ==> #[trigger] starts[r] == starts[first_row as int],
        forall|r: int| first_row <= r < end ==> #[trigger] columns[r] == columns[first_row as int],
        forall|r: int| first_row <= r < end ==> #[trigger] counts[r] == counts[first_row as int],
))]
fn run_end(starts: &[usize], columns: &[usize], counts: &[usize], first_row: usize) -> usize {
    let mut end = first_row + 1;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            first_row < end <= counts.len(),
            starts.len() == counts.len(),
            columns.len() == counts.len(),
            forall|r: int| first_row <= r < end ==> #[trigger] starts[r] == starts[first_row as int],
            forall|r: int| first_row <= r < end ==>
                #[trigger] columns[r] == columns[first_row as int],
            forall|r: int| first_row <= r < end ==> #[trigger] counts[r] == counts[first_row as int],
        decreases counts.len() - end,
    ))]
    while end < counts.len() {
        if counts[end] != counts[first_row]
            || starts[end] != starts[first_row]
            || columns[end] != columns[first_row]
        {
            return end;
        }
        end += 1;
    }
    end
}

/// Groups adjacent active rows with identical start, column, and count.
///
/// Rows with `counts[b] == 0` are skipped and break runs, so an inactive
/// row is never touched by a batched assignment.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(runs =>
    requires
        starts.len() == counts.len(),
        columns.len() == counts.len(),
    ensures
        forall|i: int| 0 <= i < runs.len() ==>
            consistent(#[trigger] runs[i], starts@, columns@, counts@),
        forall|i: int, j: int| 0 <= i < j < runs.len() ==>
            #[trigger] runs[i].end_row <= #[trigger] runs[j].first_row,
        forall|row: int| 0 <= row < counts.len() ==>
            (#[trigger] counts[row] > 0 <==> covered(runs@, row)),
))]
pub(crate) fn row_runs(starts: &[usize], columns: &[usize], counts: &[usize]) -> Vec<RowRun> {
    let mut runs: Vec<RowRun> = Vec::new();
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            starts.len() == counts.len(),
            columns.len() == counts.len(),
            forall|i: int| 0 <= i < runs.len() ==>
                consistent(#[trigger] runs[i], starts@, columns@, counts@),
            forall|i: int| 0 <= i < runs.len() ==> #[trigger] runs[i].end_row <= row,
            forall|i: int, j: int| 0 <= i < j < runs.len() ==>
                #[trigger] runs[i].end_row <= #[trigger] runs[j].first_row,
            forall|r: int| 0 <= r < row ==> (#[trigger] counts[r] > 0 <==> covered(runs@, r)),
            forall|r: int| row <= r < counts.len() ==> !covered(runs@, r),
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] == 0 {
            row += 1;
            continue;
        }
        let first_row = row;
        let start = starts[row];
        let column = columns[row];
        let count = counts[row];
        row = run_end(starts, columns, counts, first_row);
        let run = RowRun {
            first_row,
            end_row: row,
            start,
            column,
            count,
        };
        #[cfg(verus_keep_ghost)]
        proof! {
            let old_runs = runs@;
            let new_runs = old_runs.push(run);
            assert(new_runs[old_runs.len() as int] == run);
            assert forall|i: int| 0 <= i < old_runs.len() implies new_runs[i] == old_runs[i] by {}
            // Rows of the new run are covered by it.
            assert forall|r: int| first_row <= r < row implies covered(new_runs, r) by {
                assert(new_runs[old_runs.len() as int].end_row > r);
            }
            // Rows not yet visited lie after every run, including the new one.
            assert forall|r: int| row <= r < counts.len() implies !covered(new_runs, r) by {
                if covered(new_runs, r) {
                    let i = choose|i: int| 0 <= i < new_runs.len()
                        && new_runs[i].first_row <= r < #[trigger] new_runs[i].end_row;
                    if i < old_runs.len() {
                        assert(new_runs[i].end_row <= first_row);
                    }
                }
            }
            // Earlier rows keep their coverage: the new run starts after them.
            assert forall|r: int| 0 <= r < first_row implies
                (counts[r] > 0 <==> covered(new_runs, r)) by {
                if covered(new_runs, r) {
                    let i = choose|i: int| 0 <= i < new_runs.len()
                        && new_runs[i].first_row <= r < #[trigger] new_runs[i].end_row;
                    assert(i < old_runs.len());
                    assert(covered(old_runs, r));
                }
                if covered(old_runs, r) {
                    let i = choose|i: int| 0 <= i < old_runs.len()
                        && old_runs[i].first_row <= r < #[trigger] old_runs[i].end_row;
                    assert(new_runs[i].end_row == old_runs[i].end_row);
                    assert(covered(new_runs, r));
                }
            }
        }
        runs.push(run);
    }
    runs
}

#[cfg(test)]
#[path = "rows_tests.rs"]
mod tests;
