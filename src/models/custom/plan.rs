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
//! table, `displacement_slot` (compiled for tests and proofs) is the scalar
//! specification of the device index, and [`slot_displacement`] is its inverse
//! used to fill the table.
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
//! so every row's history is written by a constant number of tensor kernels.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Number of `true` cells in `cells[start..end)`.
pub open spec fn count_valid(cells: Seq<bool>, start: int, end: int) -> int
    decreases end - start,
{
    if end <= start {
        0
    } else {
        count_valid(cells, start, end - 1) + if cells[end - 1] { 1int } else { 0int }
    }
}

/// Counting a range never exceeds its width and is never negative.
pub proof fn lemma_count_bounds(cells: Seq<bool>, start: int, end: int)
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
pub proof fn lemma_count_monotone(cells: Seq<bool>, middle: int, end: int)
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
/// Returns `R = min(max_distance, C - 1)` and `2R + 1`, or `None` for an empty
/// context or a length that does not fit `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(table =>
    ensures
        evaluated_context == 0 ==> table.is_none(),
        table matches Some(t) ==> {
            &&& t.radius <= max_distance
            &&& t.radius < evaluated_context
            &&& (t.radius == max_distance || t.radius + 1 == evaluated_context)
            &&& t.length == 2 * t.radius + 1
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
/// `committed..committed + n` in order without gaps. Returns `None` when the
/// new length would exceed `capacity` (or `usize`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(planned =>
    ensures
        planned matches Some(plan) ==> {
            &&& plan.count as int == count_valid(validity@, 0, validity.len() as int)
            &&& plan.length == committed + plan.count
            &&& plan.length <= capacity
            &&& plan.slots.len() == validity.len()
            &&& forall|c: int| 0 <= c < validity.len() ==> {
                &&& validity[c] ==> #[trigger] plan.slots[c] as int
                    == committed + count_valid(validity@, 0, c)
                &&& validity[c] ==> committed <= plan.slots[c] < plan.length
                &&& !validity[c] ==> plan.slots[c] == 0
            }
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
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// Plans one append of a row-major `[batch, step]` validity grid to rows of `lengths`.
///
/// Every row is planned by the verified [`plan_row_append`]. Errors are
/// checked in a fixed order, each reporting the first offending row: an empty
/// prompt row (when `require_every_row`), an entirely masked step, a capacity
/// overflow, and finally a never-started row that appends nothing.
pub(crate) fn plan_history_append(
    validity: &[bool],
    step: usize,
    lengths: &[usize],
    capacity: usize,
    require_every_row: bool,
) -> Result<HistoryAppend, HistoryAppendError> {
    debug_assert!(step > 0 && validity.len() == lengths.len() * step);
    let counts = validity
        .chunks_exact(step)
        .map(|cells| cells.iter().filter(|&&valid| valid).count())
        .collect::<Vec<_>>();
    // Harnesses compile as Rust 2021 under Verus, so this avoids `let` chains.
    if let Some(row) = counts
        .iter()
        .position(|&count| require_every_row && count == 0)
    {
        return Err(HistoryAppendError::EmptyRow { row });
    }
    if counts.iter().all(|&count| count == 0) {
        return Err(HistoryAppendError::EmptyStep);
    }
    let mut plan = HistoryAppend {
        counts,
        lengths: Vec::with_capacity(lengths.len()),
        slots: Vec::with_capacity(validity.len()),
    };
    for (row, (cells, &committed)) in validity.chunks_exact(step).zip(lengths).enumerate() {
        let appended =
            plan_row_append(cells, committed, capacity).ok_or(HistoryAppendError::Capacity {
                row,
                requested: committed.checked_add(plan.counts[row]),
            })?;
        debug_assert_eq!(appended.count, plan.counts[row]);
        plan.lengths.push(appended.length);
        plan.slots.extend(appended.slots);
    }
    if let Some(row) = plan.lengths.iter().position(|&length| length == 0) {
        return Err(HistoryAppendError::NeverStarted { row });
    }
    Ok(plan)
}

/// One rectangular write window covering every slot appended in a step.
///
/// Columns `start..end` of the `[batch, capacity]` history are rewritten as
/// `written ? input[r, sources] : history`, where `sources` and `written` are
/// row-major `[batch, end - start]`. Slot `j` of row `r` is written exactly
/// when `L_r <= j < L_r + n_r`, and then reads the input column whose rank is
/// `j - L_r`.
#[derive(Clone, Debug, PartialEq, Eq)]
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
/// (which [`plan_history_append`] rejects).
pub(crate) fn plan_history_writes(
    plan: &HistoryAppend,
    validity: &[bool],
    step: usize,
) -> HistoryWrites {
    let advancing = plan
        .counts
        .iter()
        .zip(&plan.lengths)
        .filter(|(count, _)| **count > 0);
    let start = advancing
        .clone()
        .map(|(count, length)| length - count)
        .min()
        .unwrap_or(0);
    let end = advancing.map(|(_, length)| *length).max().unwrap_or(start);
    let width = end - start;
    let mut sources = vec![0; plan.counts.len() * width];
    let mut written = vec![false; plan.counts.len() * width];
    for (cell, (&valid, &slot)) in validity.iter().zip(&plan.slots).enumerate() {
        if valid {
            let row = cell / step;
            let index = row * width + (slot - start);
            sources[index] = cell % step;
            written[index] = true;
        }
    }
    HistoryWrites {
        start,
        end,
        sources,
        written,
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
/// Planning examples, edge cases, and properties against naive references.
mod tests;
