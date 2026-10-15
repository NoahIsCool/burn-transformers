//! Pure index planning for modular models: relative-position tables and history appends.
//!
//! Two pieces of host-side integer logic decide which device indices the
//! modular graphs read and write. Both are dependency-free and checked in place
//! by Verus through `verus/custom.rs` (see `verus/README.md`).
//!
//! # Clamped displacement tables
//!
//! A T5 relative-position bucket depends only on the signed displacement
//! `d = p_key - p_query`. Every bucket with `|d| >= max_distance` is the
//! saturated last bucket of its direction, and positions validated against an
//! evaluated context `C` satisfy `|d| <= C - 1`. Clamping `d` to
//! `[-R, R]` with `R = min(max_distance, C - 1)` therefore changes no bucket,
//! so a lookup table of `2R + 1` host-evaluated buckets indexed by
//! `clamp(d, -R, R) + R` reproduces the pairwise reference exactly for any
//! (gapped, offset, or per-row) positions. [`displacement_table`] sizes that
//! table, [`portable_displacement_table`] additionally bounds it by the
//! portable 32-bit index range, `displacement_slot` (compiled for tests and
//! proofs) is the scalar specification of the device index, and
//! [`slot_displacement`] is its inverse used to fill the table.
//!
//! # Compact history appends
//!
//! The reference decode state stores each row's valid tokens contiguously.
//! For an append step of `S` columns with row-major validity `m`, row `r`
//! appends `n_r = Σ_c m[r, c]` tokens after its committed length `L_r`; valid
//! column `c` lands in slot `L_r + Σ_{c' < c} m[r, c']` (its rank among the
//! row's valid columns). [`plan_history_append`] computes those counts, new
//! lengths, and slots while enforcing the capacity, the empty-step, the
//! prefill empty-row, and the never-started-row policies.
//! [`plan_history_writes`] turns the slots into one rectangular gather window
//! `[start, end)` so every row's history is written by a constant number of
//! tensor kernels, and [`plan_history`] composes the two.
//!
//! # Verified properties
//!
//! For every admissible input the proofs establish, without overflow:
//!
//! - the append planner returns exactly the documented first error in policy
//!   order, or per-row counts `n_r`, new lengths `L_r + n_r <= capacity`, and
//!   slots `L_r + rank(c)` for valid cells and zero for masked cells;
//! - the write window is the tightest one: `start` is the smallest committed
//!   length and `end` the largest new length among advancing rows (both zero
//!   when no row advances);
//! - window cell `(r, j)` is written exactly when `L_r <= start + j < L_r + n_r`,
//!   and then gathers the unique valid input column whose rank is
//!   `start + j - L_r`, so valid cells and written window cells correspond
//!   one to one.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Number of `true` cells in `cells[start..end)`.
pub(crate) open spec fn count_valid(cells: Seq<bool>, start: int, end: int) -> int
    decreases end - start,
{
    if end <= start {
        0
    } else {
        count_valid(cells, start, end - 1) + if cells[end - 1] { 1int } else { 0int }
    }
}

/// Counting a range never exceeds its width and is never negative.
pub(crate) proof fn lemma_count_bounds(cells: Seq<bool>, start: int, end: int)
    requires
        start <= end,
    ensures
        0 <= count_valid(cells, start, end) <= end - start,
    decreases end - start,
{
    if end > start {
        lemma_count_bounds(cells, start, end - 1);
    }
}

/// Extending a counted prefix never decreases the count.
pub(crate) proof fn lemma_count_monotone(cells: Seq<bool>, middle: int, end: int)
    requires
        0 <= middle <= end,
    ensures
        count_valid(cells, 0, middle) <= count_valid(cells, 0, end),
    decreases end - middle,
{
    if end > middle {
        lemma_count_monotone(cells, middle, end - 1);
    }
}

/// Radius `R = min(max_distance, C - 1)` of the clamped displacement table.
pub(crate) open spec fn clamped_radius(max_distance: int, evaluated_context: int) -> int {
    if max_distance < evaluated_context - 1 {
        max_distance
    } else {
        evaluated_context - 1
    }
}

/// Row `row` of a row-major grid with `step` columns: `cells[row·step .. row·step + step)`.
pub(crate) open spec fn grid_row<T>(cells: Seq<T>, step: int, row: int) -> Seq<T> {
    cells.subrange(row * step, row * step + step)
}

/// Valid tokens `n_r` in row `row` of a `[rows, step]` validity grid.
pub(crate) open spec fn row_count(validity: Seq<bool>, step: int, row: int) -> int {
    count_valid(grid_row(validity, step, row), 0, step)
}

/// `slots` gives valid column `c` the slot `committed + rank(c)` and masked columns zero.
///
/// `rank(c)` counts the valid columns before `c`, so valid columns fill
/// `committed..committed + n` in order without gaps.
pub(crate) open spec fn slots_follow_ranks(
    validity: Seq<bool>,
    committed: int,
    slots: Seq<usize>,
) -> bool {
    &&& slots.len() == validity.len()
    &&& forall|c: int| 0 <= c < validity.len() ==> {
        &&& validity[c] ==> #[trigger] slots[c] as int == committed + count_valid(validity, 0, c)
        &&& !validity[c] ==> slots[c] == 0
    }
}

/// When `require_every_row` holds, every one of the first `rows` rows has a valid token.
pub(crate) open spec fn rows_nonempty_as_required(
    validity: Seq<bool>,
    step: int,
    rows: int,
    require_every_row: bool,
) -> bool {
    require_every_row ==> forall|r: int|
        0 <= r < rows ==> #[trigger] row_count(validity, step, r) > 0
}

/// Some row among the first `rows` appends at least one token.
pub(crate) open spec fn some_row_advances(validity: Seq<bool>, step: int, rows: int) -> bool {
    exists|r: int| 0 <= r < rows && #[trigger] row_count(validity, step, r) > 0
}

/// Rows `0..end` keep `L_r + n_r <= capacity`.
pub(crate) open spec fn rows_fit(
    validity: Seq<bool>,
    step: int,
    lengths: Seq<usize>,
    capacity: int,
    end: int,
) -> bool {
    forall|r: int| 0 <= r < end ==> lengths[r] + #[trigger] row_count(validity, step, r) <= capacity
}

/// Rows `0..end` hold at least one token after the append: `L_r + n_r > 0`.
pub(crate) open spec fn rows_started(
    validity: Seq<bool>,
    step: int,
    lengths: Seq<usize>,
    end: int,
) -> bool {
    forall|r: int| 0 <= r < end ==> lengths[r] + #[trigger] row_count(validity, step, r) > 0
}

/// Exact outcome of planning one append, including the order of policy errors.
///
/// Each error names the first offending row and implies that every earlier
/// policy passed: an empty prompt row (only when `require_every_row`), an
/// entirely masked step, a capacity overflow (with the requested length when
/// it fits `usize`), and a row that holds no token after the append.
pub(crate) open spec fn history_append_outcome(
    validity: Seq<bool>,
    step: int,
    lengths: Seq<usize>,
    capacity: int,
    require_every_row: bool,
    planned: Result<HistoryAppend, HistoryAppendError>,
) -> bool {
    let rows = lengths.len() as int;
    match planned {
        Err(HistoryAppendError::EmptyRow { row }) => {
            &&& require_every_row
            &&& row < rows
            &&& row_count(validity, step, row as int) == 0
            &&& forall|r: int| 0 <= r < row ==> #[trigger] row_count(validity, step, r) > 0
        },
        Err(HistoryAppendError::EmptyStep) => {
            &&& rows_nonempty_as_required(validity, step, rows, require_every_row)
            &&& !some_row_advances(validity, step, rows)
        },
        Err(HistoryAppendError::Capacity { row, requested }) => {
            &&& rows_nonempty_as_required(validity, step, rows, require_every_row)
            &&& some_row_advances(validity, step, rows)
            &&& row < rows
            &&& rows_fit(validity, step, lengths, capacity, row as int)
            &&& lengths[row as int] + row_count(validity, step, row as int) > capacity
            &&& requested == if lengths[row as int] + row_count(validity, step, row as int)
                <= usize::MAX {
                Some((lengths[row as int] + row_count(validity, step, row as int)) as usize)
            } else {
                None::<usize>
            }
        },
        Err(HistoryAppendError::NeverStarted { row }) => {
            &&& rows_nonempty_as_required(validity, step, rows, require_every_row)
            &&& some_row_advances(validity, step, rows)
            &&& rows_fit(validity, step, lengths, capacity, rows)
            &&& row < rows
            &&& rows_started(validity, step, lengths, row as int)
            &&& lengths[row as int] + row_count(validity, step, row as int) == 0
        },
        Ok(plan) => {
            &&& rows_nonempty_as_required(validity, step, rows, require_every_row)
            &&& some_row_advances(validity, step, rows)
            &&& rows_fit(validity, step, lengths, capacity, rows)
            &&& rows_started(validity, step, lengths, rows)
            &&& plan.counts.len() == rows
            &&& plan.lengths.len() == rows
            &&& plan.slots.len() == validity.len()
            &&& forall|r: int| 0 <= r < rows ==> {
                &&& #[trigger] plan.counts@[r] as int == row_count(validity, step, r)
                &&& plan.lengths@[r] == lengths[r] + plan.counts@[r]
                &&& slots_follow_ranks(
                    grid_row(validity, step, r),
                    lengths[r] as int,
                    grid_row(plan.slots@, step, r),
                )
            }
        },
    }
}

/// `plan` is a successful append of `validity`, as [`plan_history_writes`] requires.
///
/// Committed lengths are recovered as `L_r = lengths[r] - counts[r]`.
pub(crate) open spec fn planned_append(
    plan: HistoryAppend,
    validity: Seq<bool>,
    step: int,
) -> bool {
    let rows = plan.counts.len() as int;
    &&& plan.lengths.len() == rows
    &&& validity.len() == rows * step
    &&& plan.slots.len() == rows * step
    &&& forall|r: int| 0 <= r < rows ==> {
        &&& #[trigger] plan.counts@[r] as int == row_count(validity, step, r)
        &&& plan.counts@[r] <= plan.lengths@[r]
        &&& slots_follow_ranks(
            grid_row(validity, step, r),
            plan.lengths@[r] - plan.counts@[r],
            grid_row(plan.slots@, step, r),
        )
    }
}

/// `[start, end)` is the tightest window containing every appended slot.
///
/// It spans from the smallest committed length to the largest new length of
/// the advancing rows (`n_r > 0`), and is `[0, 0)` when no row advances.
pub(crate) open spec fn tight_window(plan: HistoryAppend, start: int, end: int) -> bool {
    let rows = plan.counts.len() as int;
    if exists|r: int| 0 <= r < rows && #[trigger] plan.counts@[r] > 0 {
        &&& forall|r: int| 0 <= r < rows && #[trigger] plan.counts@[r] > 0
            ==> start <= plan.lengths@[r] - plan.counts@[r] && plan.lengths@[r] <= end
        &&& exists|r: int| 0 <= r < rows && #[trigger] plan.counts@[r] > 0
            && start == plan.lengths@[r] - plan.counts@[r]
        &&& exists|r: int| 0 <= r < rows && #[trigger] plan.counts@[r] > 0
            && end == plan.lengths@[r]
    } else {
        start == 0 && end == 0
    }
}

/// One row of the write window: cell `j` holds history slot `start + j`.
///
/// The cell is written exactly when `committed <= start + j < length`, and
/// then gathers the valid column of `cells` whose rank is
/// `start + j - committed`; unwritten cells hold source zero.
pub(crate) open spec fn window_row(
    cells: Seq<bool>,
    committed: int,
    length: int,
    start: int,
    written: Seq<bool>,
    sources: Seq<usize>,
) -> bool {
    &&& sources.len() == written.len()
    &&& forall|j: int| 0 <= j < written.len() ==> {
        &&& #[trigger] written[j] == (committed <= start + j < length)
        &&& written[j] ==> {
            &&& sources[j] < cells.len()
            &&& cells[sources[j] as int]
            &&& committed + count_valid(cells, 0, sources[j] as int) == start + j
        }
        &&& !written[j] ==> sources[j] == 0
    }
}

/// `writes` is the exact write window of the successful append `plan`.
pub(crate) open spec fn history_writes_outcome(
    plan: HistoryAppend,
    validity: Seq<bool>,
    step: int,
    writes: HistoryWrites,
) -> bool {
    let rows = plan.counts.len() as int;
    let width = writes.end - writes.start;
    &&& writes.start <= writes.end
    &&& tight_window(plan, writes.start as int, writes.end as int)
    &&& writes.written.len() == rows * width
    &&& writes.sources.len() == rows * width
    &&& forall|r: int| 0 <= r < rows ==> window_row(
        grid_row(validity, step, r),
        plan.lengths@[r] - plan.counts@[r],
        plan.lengths@[r] as int,
        writes.start as int,
        #[trigger] grid_row(writes.written@, width, r),
        grid_row(writes.sources@, width, r),
    )
}

/// Appending a whole row to a row-major prefix keeps earlier rows and adds the new one.
pub(crate) proof fn lemma_append_grid_row<T>(prefix: Seq<T>, tail: Seq<T>, step: int, rows: int)
    requires
        0 <= rows,
        0 <= step,
        prefix.len() == rows * step,
        tail.len() == step,
    ensures
        (prefix + tail).len() == (rows + 1) * step,
        grid_row(prefix + tail, step, rows) == tail,
        forall|r: int| 0 <= r < rows ==> #[trigger] grid_row(prefix + tail, step, r)
            == grid_row(prefix, step, r),
{
    assert((rows + 1) * step == rows * step + step) by (nonlinear_arith);
    assert(grid_row(prefix + tail, step, rows) =~= tail);
    assert forall|r: int| 0 <= r < rows implies #[trigger] grid_row(prefix + tail, step, r)
        == grid_row(prefix, step, r) by {
        assert(r * step + step <= rows * step) by (nonlinear_arith)
            requires
                r < rows,
                0 <= step,
        ;
        assert(0 <= r * step) by (nonlinear_arith)
            requires
                0 <= r,
                0 <= step,
        ;
        assert(grid_row(prefix + tail, step, r) =~= grid_row(prefix, step, r));
    }
}

/// Row `row` of a `[rows, step]` grid lies inside it, so its offsets are representable.
pub(crate) proof fn lemma_grid_row_in_bounds(row: int, rows: int, step: int)
    requires
        0 <= row < rows,
        0 <= step,
    ensures
        0 <= row * step,
        row * step + step <= rows * step,
{
    assert(0 <= row * step) by (nonlinear_arith)
        requires
            0 <= row,
            0 <= step,
    ;
    assert(row * step + step <= rows * step) by (nonlinear_arith)
        requires
            row < rows,
            0 <= step,
    ;
}

} // verus!

// ---------------------------------------------------------------------------
// Clamped displacement tables
// ---------------------------------------------------------------------------

/// Extent of a clamped relative-displacement lookup table.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DisplacementTable {
    /// Largest represented displacement magnitude `R`.
    pub(crate) radius: usize,
    /// Number of table entries, `2R + 1`.
    pub(crate) length: usize,
}

/// Sizes the clamped table for buckets saturating at `max_distance` over a context `C`.
///
/// Returns `R = min(max_distance, C - 1)` and `2R + 1`. The result is `None`
/// exactly for an empty context or a length that does not fit `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(table =>
    ensures
        match table {
            Some(t) => {
                &&& evaluated_context > 0
                &&& t.radius == clamped_radius(max_distance as int, evaluated_context as int)
                &&& t.length == 2 * t.radius + 1
            },
            None => {
                let radius = clamped_radius(max_distance as int, evaluated_context as int);
                evaluated_context == 0 || 2 * radius + 1 > usize::MAX
            },
        },
        table matches Some(t) ==> {
            &&& t.radius <= max_distance
            &&& t.radius < evaluated_context
        },
))]
pub(crate) fn displacement_table(
    max_distance: usize,
    evaluated_context: usize,
) -> Option<DisplacementTable> {
    if evaluated_context == 0 {
        return None;
    }
    let span = evaluated_context - 1;
    let radius = if max_distance < span {
        max_distance
    } else {
        span
    };
    let doubled = radius.checked_mul(2)?;
    let length = doubled.checked_add(1)?;
    Some(DisplacementTable { radius, length })
}

/// Largest displacement table, so every slot fits a signed 32-bit integer dtype.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const PORTABLE_INDEX_LIMIT: usize = i32::MAX as usize;

/// Sizes a displacement table whose slots also fit the portable 32-bit index range.
///
/// Returns [`displacement_table`]'s table when its length is at most
/// `i32::MAX`, and `None` otherwise. A returned radius then satisfies
/// `2R + 1 <= i32::MAX`, so it converts to `i64` (and every slot to `i32`)
/// without loss.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(table =>
    ensures
        match table {
            Some(t) => {
                &&& evaluated_context > 0
                &&& t.radius == clamped_radius(max_distance as int, evaluated_context as int)
                &&& t.length == 2 * t.radius + 1
                &&& t.length <= i32::MAX
                &&& t.radius <= i32::MAX / 2
            },
            None => evaluated_context == 0
                || 2 * clamped_radius(max_distance as int, evaluated_context as int) + 1 > i32::MAX,
        },
))]
pub(crate) fn portable_displacement_table(
    max_distance: usize,
    evaluated_context: usize,
) -> Option<DisplacementTable> {
    match displacement_table(max_distance, evaluated_context) {
        Some(table) if table.length <= PORTABLE_INDEX_LIMIT => Some(table),
        _ => None,
    }
}

/// Device index of displacement `d` in a table of radius `R`: `clamp(d, -R, R) + R`.
///
/// This is the scalar specification of the tensor expression the modular
/// models evaluate; a unit test compares every device entry with it, so it is
/// compiled only for tests and proofs.
#[cfg(any(test, verus_keep_ghost))]
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(slot =>
    requires
        radius <= i64::MAX as usize / 2,
    ensures
        slot <= 2 * radius,
        -(radius as int) <= displacement <= radius as int
            ==> slot == displacement + radius as int,
        displacement < -(radius as int) ==> slot == 0,
        displacement > radius as int ==> slot == 2 * radius,
))]
pub(crate) fn displacement_slot(displacement: i64, radius: usize) -> usize {
    let bound = radius as i64;
    if displacement <= -bound {
        0
    } else if displacement >= bound {
        2 * radius
    } else {
        (displacement + bound) as usize
    }
}

/// Displacement `k - R` stored at table slot `k`; the inverse of `displacement_slot`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(displacement =>
    requires
        radius <= i64::MAX as usize / 2,
        slot <= 2 * radius,
    ensures
        displacement == slot as int - radius as int,
        -(radius as int) <= displacement <= radius as int,
))]
pub(crate) fn slot_displacement(slot: usize, radius: usize) -> i64 {
    slot as i64 - radius as i64
}

// ---------------------------------------------------------------------------
// Compact history appends
// ---------------------------------------------------------------------------

/// Why an append to the reference history cannot be planned.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HistoryAppendError {
    /// A prompt row has no valid token.
    EmptyRow {
        /// First offending batch row.
        row: usize,
    },
    /// Every cell of a decode step is masked, so no row would advance.
    EmptyStep,
    /// A row's new length would exceed the capacity.
    Capacity {
        /// First offending batch row.
        row: usize,
        /// Requested length `L_r + n_r`, or `None` when it overflows `usize`.
        requested: Option<usize>,
    },
    /// A row that never received a token appends nothing in a mixed step.
    ///
    /// The reference recomputation would have to evaluate an empty row.
    NeverStarted {
        /// First offending batch row.
        row: usize,
    },
}

/// One row's append: its count, new length, and the slot of every input column.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RowAppend {
    /// Valid tokens appended to the row, `n`.
    pub(crate) count: usize,
    /// Row length after the append, `L + n`.
    pub(crate) length: usize,
    /// History slot of each input column; masked columns hold zero.
    pub(crate) slots: Vec<usize>,
}

/// Plans one row's append of `validity` after `committed` stored tokens.
///
/// Valid column `c` receives slot `committed + rank(c)`, where `rank(c)` counts
/// the valid columns before it, so valid columns fill
/// `committed..committed + n` in order without gaps. Returns `None` exactly
/// when the new length would exceed `capacity` (or `usize`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(planned =>
    ensures
        planned matches Some(plan) ==> {
            &&& plan.count as int == count_valid(validity@, 0, validity.len() as int)
            &&& plan.length == committed + plan.count
            &&& plan.length <= capacity
            &&& slots_follow_ranks(validity@, committed as int, plan.slots@)
            &&& forall|c: int| 0 <= c < validity.len() && validity[c]
                ==> committed <= #[trigger] plan.slots[c] < plan.length
            &&& forall|a: int, b: int| 0 <= a < b < validity.len()
                && validity[a] && validity[b] ==> #[trigger] plan.slots[a] < #[trigger] plan.slots[b]
        },
        planned is None ==> committed + count_valid(validity@, 0, validity.len() as int) > capacity,
))]
pub(crate) fn plan_row_append(
    validity: &[bool],
    committed: usize,
    capacity: usize,
) -> Option<RowAppend> {
    if committed > capacity {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_bounds(validity@, 0, validity.len() as int);
        }
        return None;
    }
    let mut slots: Vec<usize> = Vec::with_capacity(validity.len());
    let mut count: usize = 0;
    let mut column: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            column <= validity.len(),
            count as int == count_valid(validity@, 0, column as int),
            count <= column,
            committed + count <= capacity,
            slots.len() == column,
            forall|c: int| 0 <= c < column ==> {
                &&& validity[c] ==> #[trigger] slots[c] as int
                    == committed + count_valid(validity@, 0, c)
                &&& !validity[c] ==> slots[c] == 0
            },
        decreases validity.len() - column,
    ))]
    while column < validity.len() {
        if validity[column] {
            // `committed + count <= capacity` holds, so the slot cannot overflow.
            if count >= capacity - committed {
                #[cfg(verus_keep_ghost)]
                proof! {
                    lemma_count_monotone(validity@, column as int + 1, validity.len() as int);
                }
                return None;
            }
            slots.push(committed + count);
            count += 1;
        } else {
            slots.push(0);
        }
        column += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|c: int| 0 <= c < validity.len() && validity[c]
            implies committed <= #[trigger] slots[c] < committed + count by {
            lemma_count_bounds(validity@, 0, c);
            lemma_count_monotone(validity@, c + 1, validity.len() as int);
        }
        assert forall|a: int, b: int| 0 <= a < b < validity.len() && validity[a] && validity[b]
            implies #[trigger] slots[a] < #[trigger] slots[b] by {
            lemma_count_monotone(validity@, a + 1, b);
        }
    }
    Some(RowAppend {
        count,
        length: committed + count,
        slots,
    })
}

/// Counts, new lengths, and per-cell history slots of one append step.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct HistoryAppend {
    /// Valid tokens appended to each row, `n_r`.
    pub(crate) counts: Vec<usize>,
    /// Row lengths after the append, `L_r + n_r`.
    pub(crate) lengths: Vec<usize>,
    /// History slot of every input cell, row-major `[batch, step]`.
    ///
    /// A valid cell's slot is `L_r` plus its rank among the row's valid cells
    /// ([`plan_row_append`]); masked cells hold zero (any in-range slot works,
    /// as their outputs are masked after the gather).
    pub(crate) slots: Vec<usize>,
}

/// Counts the `true` cells of one row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(count =>
    ensures
        count as int == count_valid(cells@, 0, cells.len() as int),
))]
fn count_valid_cells(cells: &[bool]) -> usize {
    let mut count: usize = 0;
    let mut column: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            column <= cells.len(),
            count <= column,
            count as int == count_valid(cells@, 0, column as int),
        decreases cells.len() - column,
    ))]
    while column < cells.len() {
        if cells[column] {
            count += 1;
        }
        column += 1;
    }
    count
}

/// Counts the valid cells `n_r` of every row of a row-major `[rows, step]` grid.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(counts =>
    requires
        validity.len() == rows * step,
    ensures
        counts.len() == rows,
        forall|r: int|
            #![trigger counts@[r]]
            #![trigger row_count(validity@, step as int, r)]
            0 <= r < rows ==> counts@[r] as int == row_count(validity@, step as int, r),
))]
fn grid_row_counts(validity: &[bool], step: usize, rows: usize) -> Vec<usize> {
    let mut counts: Vec<usize> = Vec::with_capacity(rows);
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            validity.len() == rows * step,
            counts.len() == row,
            forall|r: int| 0 <= r < row
                ==> #[trigger] counts@[r] as int == row_count(validity@, step as int, r),
        decreases rows - row,
    ))]
    while row < rows {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_grid_row_in_bounds(row as int, rows as int, step as int);
        }
        let offset = row * step;
        counts.push(count_valid_cells(&validity[offset..offset + step]));
        row += 1;
    }
    counts
}

/// Returns the first zero entry of `values`, if any.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
        match found {
            Some(index) => {
                &&& index < values.len()
                &&& values[index as int] == 0
                &&& forall|j: int| 0 <= j < index ==> #[trigger] values[j] != 0
            },
            None => forall|j: int| 0 <= j < values.len() ==> #[trigger] values[j] != 0,
        },
))]
fn first_zero(values: &[usize]) -> Option<usize> {
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= values.len(),
            forall|j: int| 0 <= j < index ==> #[trigger] values[j] != 0,
        decreases values.len() - index,
    ))]
    while index < values.len() {
        if values[index] == 0 {
            return Some(index);
        }
        index += 1;
    }
    None
}

/// Plans one append of a row-major `[batch, step]` validity grid to rows of `lengths`.
///
/// `validity` must hold `lengths.len() * step` cells. Every row is planned by
/// the verified [`plan_row_append`]. Errors are checked in a fixed order, each
/// reporting the first offending row: an empty prompt row (when
/// `require_every_row`), an entirely masked step, a capacity overflow, and
/// finally a never-started row that appends nothing. The proof shows the
/// result is exactly `history_append_outcome`: the first applicable error, or
/// counts, new lengths, and rank slots for every row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(planned =>
    requires
        validity.len() == lengths.len() * step,
    ensures
        history_append_outcome(
            validity@,
            step as int,
            lengths@,
            capacity as int,
            require_every_row,
            planned,
        ),
))]
pub(crate) fn plan_history_append(
    validity: &[bool],
    step: usize,
    lengths: &[usize],
    capacity: usize,
    require_every_row: bool,
) -> Result<HistoryAppend, HistoryAppendError> {
    let rows = lengths.len();
    let counts = grid_row_counts(validity, step, rows);
    // Verus compiles verified bodies as Rust 2021, so this avoids a `let` chain.
    let empty_row = if require_every_row {
        first_zero(counts.as_slice())
    } else {
        None
    };
    if let Some(row) = empty_row {
        return Err(HistoryAppendError::EmptyRow { row });
    }
    let Some(_) = first_nonzero(counts.as_slice()) else {
        return Err(HistoryAppendError::EmptyStep);
    };
    let mut new_lengths: Vec<usize> = Vec::with_capacity(rows);
    let mut slots: Vec<usize> = Vec::with_capacity(validity.len());
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows == lengths.len(),
            validity.len() == rows * step,
            counts.len() == rows,
            forall|r: int|
                #![trigger counts@[r]]
                #![trigger row_count(validity@, step as int, r)]
                0 <= r < rows ==> counts@[r] as int == row_count(validity@, step as int, r),
            rows_nonempty_as_required(validity@, step as int, rows as int, require_every_row),
            some_row_advances(validity@, step as int, rows as int),
            rows_fit(validity@, step as int, lengths@, capacity as int, row as int),
            new_lengths.len() == row,
            forall|r: int|
                #![trigger new_lengths@[r]]
                #![trigger counts@[r]]
                0 <= r < row ==> new_lengths@[r] == lengths[r] + counts@[r],
            slots.len() == row * step,
            forall|r: int| 0 <= r < row ==> slots_follow_ranks(
                grid_row(validity@, step as int, r),
                lengths[r] as int,
                #[trigger] grid_row(slots@, step as int, r),
            ),
        decreases rows - row,
    ))]
    while row < rows {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_grid_row_in_bounds(row as int, rows as int, step as int);
        }
        let offset = row * step;
        let cells = &validity[offset..offset + step];
        let committed = lengths[row];
        let Some(appended) = plan_row_append(cells, committed, capacity) else {
            return Err(HistoryAppendError::Capacity {
                row,
                requested: committed.checked_add(counts[row]),
            });
        };
        new_lengths.push(appended.length);
        let mut row_slots = appended.slots;
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_append_grid_row(slots@, row_slots@, step as int, row as int);
        }
        slots.append(&mut row_slots);
        row += 1;
    }
    if let Some(row) = first_zero(new_lengths.as_slice()) {
        return Err(HistoryAppendError::NeverStarted { row });
    }
    Ok(HistoryAppend {
        counts,
        lengths: new_lengths,
        slots,
    })
}

/// Returns the first nonzero entry of `values`, if any.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
        match found {
            Some(index) => {
                &&& index < values.len()
                &&& values[index as int] != 0
                &&& forall|j: int| 0 <= j < index ==> #[trigger] values[j] == 0
            },
            None => forall|j: int| 0 <= j < values.len() ==> #[trigger] values[j] == 0,
        },
))]
fn first_nonzero(values: &[usize]) -> Option<usize> {
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= values.len(),
            forall|j: int| 0 <= j < index ==> #[trigger] values[j] == 0,
        decreases values.len() - index,
    ))]
    while index < values.len() {
        if values[index] != 0 {
            return Some(index);
        }
        index += 1;
    }
    None
}

/// One rectangular write window covering every slot appended in a step.
///
/// Columns `start..end` of the `[batch, capacity]` history are rewritten as
/// `written ? input[r, sources] : history`, where `sources` and `written` are
/// row-major `[batch, end - start]`. Slot `j` of row `r` is written exactly
/// when `L_r <= j < L_r + n_r`, and then reads the input column whose rank is
/// `j - L_r`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct HistoryWrites {
    /// First rewritten history column.
    pub(crate) start: usize,
    /// One past the last rewritten history column.
    pub(crate) end: usize,
    /// Input column gathered into each window cell (zero where not written).
    pub(crate) sources: Vec<usize>,
    /// Whether each window cell receives a new token.
    pub(crate) written: Vec<bool>,
}

/// Converts a planned append into its rectangular write window.
///
/// `start` is the smallest committed length of an advancing row and `end` the
/// largest new length, so the window is empty only when no row advances
/// (which [`plan_history_append`] rejects). The proof shows the window is
/// tight and that each row's written cells are exactly its appended slots,
/// each gathering the valid column of matching rank (`history_writes_outcome`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(writes =>
    requires
        planned_append(*plan, validity@, step as int),
    ensures
        history_writes_outcome(*plan, validity@, step as int, writes),
))]
pub(crate) fn plan_history_writes(
    plan: &HistoryAppend,
    validity: &[bool],
    step: usize,
) -> HistoryWrites {
    let rows = plan.counts.len();
    // The tightest window over advancing rows; `[0, 0)` when none advances.
    let mut advancing = false;
    let mut start: usize = 0;
    let mut end: usize = 0;
    #[cfg(verus_keep_ghost)]
    proof_decl! {
        let ghost mut first_row: int = 0;
        let ghost mut last_row: int = 0;
    }
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows == plan.counts.len(),
            planned_append(*plan, validity@, step as int),
            advancing == exists|r: int| 0 <= r < row && #[trigger] plan.counts@[r] > 0,
            !advancing ==> start == 0 && end == 0,
            advancing ==> {
                &&& 0 <= first_row < row
                &&& plan.counts@[first_row] > 0
                &&& start == plan.lengths@[first_row] - plan.counts@[first_row]
                &&& 0 <= last_row < row
                &&& plan.counts@[last_row] > 0
                &&& end == plan.lengths@[last_row]
            },
            forall|r: int| 0 <= r < row && #[trigger] plan.counts@[r] > 0
                ==> start <= plan.lengths@[r] - plan.counts@[r] && plan.lengths@[r] <= end,
        decreases rows - row,
    ))]
    while row < rows {
        let count = plan.counts[row];
        if count > 0 {
            let length = plan.lengths[row];
            let first = length - count;
            if !advancing || first < start {
                start = first;
                #[cfg(verus_keep_ghost)]
                proof! {
                    first_row = row as int;
                }
            }
            if !advancing || length > end {
                end = length;
                #[cfg(verus_keep_ghost)]
                proof! {
                    last_row = row as int;
                }
            }
            advancing = true;
        }
        row += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        if advancing {
            assert(start <= end);
        }
    }
    let width = end - start;
    let mut sources: Vec<usize> = Vec::new();
    let mut written: Vec<bool> = Vec::new();
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows == plan.counts.len(),
            planned_append(*plan, validity@, step as int),
            start <= end,
            width == end - start,
            forall|r: int| 0 <= r < rows && #[trigger] plan.counts@[r] > 0
                ==> start <= plan.lengths@[r] - plan.counts@[r] && plan.lengths@[r] <= end,
            written.len() == row * width,
            sources.len() == row * width,
            forall|r: int| 0 <= r < row ==> window_row(
                grid_row(validity@, step as int, r),
                plan.lengths@[r] - plan.counts@[r],
                plan.lengths@[r] as int,
                start as int,
                #[trigger] grid_row(written@, width as int, r),
                grid_row(sources@, width as int, r),
            ),
        decreases rows - row,
    ))]
    while row < rows {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_grid_row_in_bounds(row as int, rows as int, step as int);
        }
        let offset = row * step;
        let cells = &validity[offset..offset + step];
        let slots = &plan.slots[offset..offset + step];
        #[cfg(verus_keep_ghost)]
        proof_decl! {
            let ghost count = plan.counts@[row as int] as int;
            let ghost committed = plan.lengths@[row as int] - count;
        }
        let mut row_sources: Vec<usize> = vec![0; width];
        let mut row_written: Vec<bool> = vec![false; width];
        let mut column: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                column <= step,
                cells@ == grid_row(validity@, step as int, row as int),
                slots@ == grid_row(plan.slots@, step as int, row as int),
                cells.len() == step,
                count == count_valid(cells@, 0, step as int),
                committed == plan.lengths@[row as int] - count,
                slots_follow_ranks(cells@, committed, slots@),
                count > 0 ==> start <= committed && committed + count <= end,
                width == end - start,
                row_written.len() == width,
                row_sources.len() == width,
                forall|j: int| 0 <= j < width ==> {
                    &&& #[trigger] row_written@[j]
                        == (committed <= start + j
                            < committed + count_valid(cells@, 0, column as int))
                    &&& row_written@[j] ==> {
                        &&& row_sources@[j] < step
                        &&& cells@[row_sources@[j] as int]
                        &&& committed + count_valid(cells@, 0, row_sources@[j] as int) == start + j
                    }
                    &&& !row_written@[j] ==> row_sources@[j] == 0
                },
            decreases step - column,
        ))]
        while column < step {
            if cells[column] {
                #[cfg(verus_keep_ghost)]
                proof! {
                    lemma_count_monotone(cells@, column as int + 1, step as int);
                    lemma_count_bounds(cells@, 0, column as int);
                }
                let index = slots[column] - start;
                row_sources[index] = column;
                row_written[index] = true;
            }
            column += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_append_grid_row(written@, row_written@, width as int, row as int);
            lemma_append_grid_row(sources@, row_sources@, width as int, row as int);
        }
        sources.append(&mut row_sources);
        written.append(&mut row_written);
        row += 1;
    }
    HistoryWrites {
        start,
        end,
        sources,
        written,
    }
}

/// Plans an append and its write window in one verified step.
///
/// `validity` must hold `lengths.len() * step` cells. The result is exactly
/// [`plan_history_append`]'s outcome, and on success the window of
/// [`plan_history_writes`] for that plan, so the window's precondition is
/// discharged by the proof rather than by the caller.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(planned =>
    requires
        validity.len() == lengths.len() * step,
    ensures
        match planned {
            Ok((plan, writes)) => {
                &&& history_append_outcome(
                    validity@,
                    step as int,
                    lengths@,
                    capacity as int,
                    require_every_row,
                    Ok(plan),
                )
                &&& history_writes_outcome(plan, validity@, step as int, writes)
            },
            Err(error) => history_append_outcome(
                validity@,
                step as int,
                lengths@,
                capacity as int,
                require_every_row,
                Err(error),
            ),
        },
))]
pub(crate) fn plan_history(
    validity: &[bool],
    step: usize,
    lengths: &[usize],
    capacity: usize,
    require_every_row: bool,
) -> Result<(HistoryAppend, HistoryWrites), HistoryAppendError> {
    match plan_history_append(validity, step, lengths, capacity, require_every_row) {
        Ok(plan) => {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert forall|r: int| 0 <= r < plan.counts.len() implies
                    #[trigger] plan.counts@[r] <= plan.lengths@[r] by {
                    assert(plan.lengths@[r] == lengths[r] + plan.counts@[r]);
                }
            }
            let writes = plan_history_writes(&plan, validity, step);
            Ok((plan, writes))
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
/// Planning examples, edge cases, and properties against naive references.
mod tests;
