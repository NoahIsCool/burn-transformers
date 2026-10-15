//! Dependency-free planning for batched Nemotron-H execution.
//!
//! # Compact batches
//!
//! A stateful call receives a rectangular `[B, S]` batch in which row `b`
//! holds `n_b` valid tokens at columns `c_{b,0} < … < c_{b,n_b-1}`. The model
//! evaluates every row in one pass over a *right-padded compact* batch
//! `[B, N]` with `N = max_b n_b`: slot `(b, j)` holds column `c_{b,j}` for
//! `j < n_b` and is padding otherwise. Row-major flat indices make each
//! rearrangement a single index-select:
//!
//! ```text
//! gather[b N + j]  = b S + c_{b,j}            compact slot <- rectangular column
//! scatter[b S + s] = b N + j   if s = c_{b,j} rectangular column <- compact slot
//!                  = B N       otherwise      (an appended zero row)
//! packed[o_b + j]  = b N + j                  packed token <- compact slot
//! unpack[b N + j]  = o_b + j   if j < n_b     compact slot <- packed token
//!                  = T         otherwise      (an appended zero row)
//! last[b]          = b N + n_b - 1, or B N for an inactive row
//! ```
//!
//! with row offsets `o_b = sum_{r<b} n_r` and `T = sum_b n_b` valid tokens.
//! Sparse MoE blocks route only the `T` packed tokens; padded compact slots
//! never reach a router or an expert. Padded `gather` entries point at the
//! row's first rectangular column, which is always in range; callers replace
//! those tokens before embedding them.
//!
//! Attention queries take absolute positions from [`query_positions`]: slot
//! `(b, j)` sits at `W_b + j` after the row's `W_b` cached tokens, and a
//! padded slot repeats the row's last readable position, so causality and a
//! sliding window never exclude every key of a padded query in a row with
//! history.
//!
//! # Execution policies
//!
//! [`mamba_step`] chooses the Mamba-2 path of one row exactly as the pinned
//! reference does: the single-step update only for one token appended to an
//! existing state, the chunk scan otherwise. [`transactional_mamba`] decides
//! how a speculative step records recurrent state: one terminal state when
//! every row appends at most one token, per-token snapshots otherwise (which
//! requires existing state in every active row). [`mtp_length_after_truncate`]
//! keeps the shifted MTP cursor at most one position behind the target, and
//! [`verification_tokens`] sizes the one-pass target verification of a draft.
//!
//! # Verified planning
//!
//! Every function here is dependency-free integer logic, checked in place by
//! Verus through `verus/nemotron.rs` (see `verus/README.md`). For all
//! admissible inputs the proofs establish freedom from overflow and the exact
//! contents of every compact-batch map: `N` is the largest row count, each
//! `gather`, `unpack`, `packed`, and `last` entry equals its equation above,
//! `scatter` sends valid column `c_{b,k}` to slot `(b, k)` and every other
//! column to the zero row, and `unpack` addresses the zero row exactly at
//! padded slots. A plan is refused exactly for malformed columns or
//! unrepresentable extents. They also prove every slot's query position (a
//! query never lies beyond its row's last readable key), the policies' exact
//! decision rules, and the MTP cursor invariant. Tensor programs driven by
//! the plans remain covered by Rust tests.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

use crate::models::shared::plan::is_sorted_unique_below;
#[cfg(verus_keep_ghost)]
use crate::models::shared::plan::sorted_unique_below;

#[cfg(verus_keep_ghost)]
verus! {

/// Total number of valid tokens in the first `rows` rows.
pub open spec fn total_tokens(columns: Seq<Vec<usize>>, rows: int) -> int
    decreases rows,
{
    if rows <= 0 {
        0
    } else {
        total_tokens(columns, rows - 1) + columns[rows - 1]@.len()
    }
}

/// Every entry of `values` is strictly below `bound`.
pub open spec fn all_below(values: Seq<usize>, bound: int) -> bool {
    forall|i: int| 0 <= i < values.len() ==> values[i] < bound
}

/// Every entry of `values` is at most `bound`.
pub open spec fn all_at_most(values: Seq<usize>, bound: int) -> bool {
    forall|i: int| 0 <= i < values.len() ==> values[i] <= bound
}

/// Row `r`'s slot range `[r * width, (r + 1) * width)` lies inside `batch * width`.
proof fn lemma_row_block(row: int, batch: int, width: int)
    requires
        0 <= row < batch,
        width >= 0,
    ensures
        row * width + width <= batch * width,
        row * width >= 0,
{
    assert(row * width + width <= batch * width) by (nonlinear_arith)
        requires 0 <= row < batch, width >= 0;
    assert(row * width >= 0) by (nonlinear_arith)
        requires row >= 0, width >= 0;
}

/// Adding one row's tokens keeps the running total within one more row block.
proof fn lemma_total_step(columns: Seq<Vec<usize>>, row: int)
    requires
        0 <= row < columns.len(),
    ensures
        total_tokens(columns, row + 1) == total_tokens(columns, row) + columns[row]@.len(),
{
}

/// Running totals never decrease: `total(rows) <= total(more)` for `rows <= more`.
proof fn lemma_total_monotone(columns: Seq<Vec<usize>>, rows: int, more: int)
    requires
        0 <= rows <= more <= columns.len(),
    ensures
        total_tokens(columns, rows) <= total_tokens(columns, more),
    decreases more - rows,
{
    if rows < more {
        lemma_total_monotone(columns, rows, more - 1);
    }
}

/// Running totals are never negative.
proof fn lemma_total_nonnegative(columns: Seq<Vec<usize>>, rows: int)
    ensures
        total_tokens(columns, rows) >= 0,
    decreases rows,
{
    if rows > 0 {
        lemma_total_nonnegative(columns, rows - 1);
    }
}

/// Row-major flat index of entry `(row, column)` in a matrix `width` columns wide.
pub open spec fn flat(row: int, column: int, width: int) -> int {
    row * width + column
}

/// An entry of an earlier row lies before the first entry of row `rows`.
proof fn lemma_flat_below(row: int, column: int, rows: int, width: int)
    requires
        0 <= row < rows,
        0 <= column < width,
    ensures
        0 <= flat(row, column, width) < rows * width,
{
    assert(row * width + width <= rows * width) by (nonlinear_arith)
        requires 0 <= row < rows, width >= 0;
    assert(row * width >= 0) by (nonlinear_arith)
        requires row >= 0, width >= 0;
}

/// Valid token `j` of row `row` precedes every token of later rows in packed order.
proof fn lemma_packed_below(columns: Seq<Vec<usize>>, row: int, rows: int, j: int)
    requires
        0 <= row < rows <= columns.len(),
        0 <= j < columns[row]@.len(),
    ensures
        0 <= total_tokens(columns, row) + j < total_tokens(columns, rows),
{
    lemma_total_step(columns, row);
    lemma_total_monotone(columns, row + 1, rows);
    lemma_total_nonnegative(columns, row);
}

/// Some row among the first `rows` has exactly `width` valid columns.
pub open spec fn attains(columns: Seq<Vec<usize>>, rows: int, width: int) -> bool {
    exists|b: int| 0 <= b < rows && #[trigger] columns[b]@.len() == width
}

/// The position [`slot_position`] assigns to slot `slot` of a row (see its documentation).
pub open spec fn slot_position_spec(baseline: int, count: int, slot: int) -> int {
    if slot < count {
        baseline + slot
    } else if baseline + count > 0 {
        baseline + count - 1
    } else {
        0
    }
}

/// Packed (row-major valid-token) index `o_b + j` of valid token `j` of row `b`.
pub open spec fn packed_index(columns: Seq<Vec<usize>>, row: int, j: int) -> int {
    total_tokens(columns, row) + j
}

/// Every flat entry of rows `0..rows` lies before the first entry of row `rows`.
pub open spec fn rows_below(rows: int, width: int) -> bool {
    forall|b: int, j: int| 0 <= b < rows && 0 <= j < width
        ==> 0 <= #[trigger] flat(b, j, width) < rows * width
}

/// Every valid token of rows `0..rows` precedes the packed tokens of row `rows`.
pub open spec fn packed_rows_below(columns: Seq<Vec<usize>>, rows: int) -> bool {
    forall|b: int, j: int| 0 <= b < rows && 0 <= j < columns[b]@.len()
        ==> 0 <= #[trigger] packed_index(columns, b, j) < total_tokens(columns, rows)
}

/// `gather[b N + j] = b S + c_{b,j}` for valid slots and `b S` for padded slots of rows `0..rows`.
pub open spec fn gathers_rows(
    gather: Seq<usize>,
    columns: Seq<Vec<usize>>,
    rows: int,
    width: int,
    sequence: int,
) -> bool {
    forall|b: int, j: int| 0 <= b < rows && 0 <= j < width
        ==> #[trigger] gather[flat(b, j, width)]
            == flat(b, if j < columns[b]@.len() { columns[b]@[j] as int } else { 0 }, sequence)
}

/// `unpack[b N + j] = o_b + j` for valid slots and `T` for padded slots of rows `0..rows`.
pub open spec fn unpacks_rows(
    unpack: Seq<usize>,
    columns: Seq<Vec<usize>>,
    rows: int,
    width: int,
    tokens: int,
) -> bool {
    forall|b: int, j: int| 0 <= b < rows && 0 <= j < width
        ==> #[trigger] unpack[flat(b, j, width)]
            == if j < columns[b]@.len() { packed_index(columns, b, j) } else { tokens }
}

/// `packed[o_b + j] = b N + j` for every valid token of rows `0..rows`.
pub open spec fn packs_rows(
    packed: Seq<usize>,
    columns: Seq<Vec<usize>>,
    rows: int,
    width: int,
) -> bool {
    forall|b: int, j: int| 0 <= b < rows && 0 <= j < columns[b]@.len()
        ==> #[trigger] packed[packed_index(columns, b, j)] == flat(b, j, width)
}

/// `scatter[b S + c_{b,k}] = b N + k`, and every other column of rows `0..rows` reads `padding`.
pub open spec fn scatters_rows(
    scatter: Seq<usize>,
    columns: Seq<Vec<usize>>,
    rows: int,
    width: int,
    sequence: int,
    padding: int,
) -> bool {
    &&& forall|b: int, k: int| 0 <= b < rows && 0 <= k < columns[b]@.len()
        ==> #[trigger] scatter[flat(b, columns[b]@[k] as int, sequence)] == flat(b, k, width)
    &&& forall|b: int, s: int| 0 <= b < rows && 0 <= s < sequence
        && !columns[b]@.contains(s as usize)
        ==> #[trigger] scatter[flat(b, s, sequence)] == padding
}

/// `last[b] = b N + n_b - 1`, or `padding` for an inactive row, for rows `0..rows`.
pub open spec fn lasts_rows(
    last: Seq<usize>,
    columns: Seq<Vec<usize>>,
    rows: int,
    width: int,
    padding: int,
) -> bool {
    forall|b: int| 0 <= b < rows
        ==> #[trigger] last[b] == if columns[b]@.len() == 0 {
            padding
        } else {
            flat(b, columns[b]@.len() - 1, width)
        }
}

} // verus!

/// Right-padded compact layout of one ragged batch (see the module documentation).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verifier::allow(autoderive_clone_without_spec))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompactPlan {
    /// Rectangular width `S`.
    pub(crate) sequence: usize,
    /// Compact width `N = max_b n_b`; zero when every row is inactive.
    pub(crate) width: usize,
    /// Valid tokens `n_b` of every row.
    pub(crate) counts: Vec<usize>,
    /// Flat rectangular source of every compact slot, `[B * N]`.
    pub(crate) gather: Vec<usize>,
    /// Flat compact source of every rectangular slot, `[B * S]`, `B * N` for padding.
    pub(crate) scatter: Vec<usize>,
    /// Flat compact slot of every valid token in row-major order, `[T]`.
    pub(crate) packed: Vec<usize>,
    /// Packed index of every compact slot, `[B * N]`, `T` for padding.
    pub(crate) unpack: Vec<usize>,
    /// Flat compact slot of each row's last token, `[B]`, `B * N` for inactive rows.
    pub(crate) last: Vec<usize>,
}

impl CompactPlan {
    /// Returns the number of rows `B`.
    pub(crate) fn batch(&self) -> usize {
        self.counts.len()
    }

    /// Returns the number of valid tokens `T`.
    pub(crate) fn tokens(&self) -> usize {
        self.packed.len()
    }

    /// Returns whether every compact slot holds a valid token (`T = B * N`).
    pub(crate) fn is_dense(&self) -> bool {
        self.packed.len() == self.unpack.len()
    }

    /// Returns whether the compact batch equals the rectangular one (`N = S`, all valid).
    pub(crate) fn is_identity(&self) -> bool {
        self.width == self.sequence && self.is_dense()
    }
}

/// Plans the compact layout of rows whose valid columns are `columns[b]`.
///
/// Returns `None` exactly when a row's columns are not strictly increasing
/// below `sequence`, or when a flat extent (`T`, `B * N`, or `B * S`) would
/// overflow `usize`. On success the proof establishes every equation of the
/// module documentation: `N` is the largest row count, `gather`, `unpack`,
/// `packed`, and `last` hold exactly their documented entries, `scatter`
/// sends valid column `c_{b,k}` to compact slot `(b, k)` and every other
/// column to the zero row `B * N`, and an `unpack` entry equals `T` exactly
/// at padded slots. Each map therefore stays inside its source.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(planned =>
    ensures
        match planned {
            Some(plan) => {
                let batch = columns.len() as int;
                let width = plan.width as int;
                let tokens = total_tokens(columns@, batch);
                &&& plan.sequence == sequence
                &&& plan.counts.len() == batch
                &&& forall|b: int| 0 <= b < batch ==> {
                    &&& #[trigger] plan.counts@[b] == columns@[b]@.len()
                    &&& plan.counts@[b] <= plan.width
                    &&& sorted_unique_below(columns@[b]@, sequence as int)
                }
                &&& plan.width > 0 ==> attains(columns@, batch, width)
                &&& plan.gather.len() == batch * width
                &&& plan.scatter.len() == batch * sequence
                &&& plan.unpack.len() == batch * width
                &&& plan.packed.len() == tokens
                &&& plan.last.len() == batch
                &&& gathers_rows(plan.gather@, columns@, batch, width, sequence as int)
                &&& unpacks_rows(plan.unpack@, columns@, batch, width, tokens)
                &&& packs_rows(plan.packed@, columns@, batch, width)
                &&& scatters_rows(plan.scatter@, columns@, batch, width, sequence as int, batch * width)
                &&& lasts_rows(plan.last@, columns@, batch, width, batch * width)
                &&& forall|b: int, j: int| 0 <= b < batch && 0 <= j < width
                    ==> (#[trigger] plan.unpack@[flat(b, j, width)] == tokens)
                        == (j >= columns@[b]@.len())
                &&& all_below(plan.gather@, batch * sequence)
                &&& all_at_most(plan.scatter@, batch * width)
                &&& all_below(plan.packed@, batch * width)
                &&& all_at_most(plan.unpack@, tokens)
                &&& all_at_most(plan.last@, batch * width)
            },
            None => {
                ||| exists|b: int| 0 <= b < columns.len()
                    && !sorted_unique_below(#[trigger] columns@[b]@, sequence as int)
                ||| total_tokens(columns@, columns.len() as int) > usize::MAX
                ||| exists|b: int| 0 <= b < columns.len()
                    && columns.len() * (#[trigger] columns@[b]@.len()) > usize::MAX
                ||| columns.len() * sequence > usize::MAX
            },
        },
))]
pub(crate) fn plan_compaction(columns: &[Vec<usize>], sequence: usize) -> Option<CompactPlan> {
    let batch = columns.len();
    let mut width: usize = 0;
    let mut tokens: usize = 0;
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            batch == columns.len(),
            row <= batch,
            tokens == total_tokens(columns@, row as int),
            width > 0 ==> sequence > 0,
            width > 0 ==> attains(columns@, row as int, width as int),
            forall|b: int| 0 <= b < row ==> {
                &&& #[trigger] columns@[b]@.len() <= width
                &&& sorted_unique_below(columns@[b]@, sequence as int)
            },
        decreases batch - row,
    ))]
    while row < batch {
        if !is_sorted_unique_below(columns[row].as_slice(), sequence) {
            return None;
        }
        let count = columns[row].len();
        // A non-empty row has a column below `sequence`, so `sequence > 0`.
        #[cfg(verus_keep_ghost)]
        proof! {
            if count > 0 {
                assert(columns@[row as int]@[0] < sequence);
            }
            lemma_total_step(columns@, row as int);
            lemma_total_monotone(columns@, row as int + 1, batch as int);
        }
        if count > width {
            width = count;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            if width > 0 && width == count {
                assert(columns@[row as int]@.len() == width);
            }
        }
        tokens = tokens.checked_add(count)?;
        row += 1;
    }
    // An overflowing `B * N` is witnessed by a row whose count is `N`.
    #[cfg(verus_keep_ghost)]
    proof! {
        if batch * width > usize::MAX {
            if width == 0 {
                assert(batch * width == 0) by (nonlinear_arith)
                    requires width == 0;
            }
            let b = choose|b: int| 0 <= b < batch && #[trigger] columns@[b]@.len() == width;
            assert(columns@[b]@.len() == width);
        }
    }
    let compact_slots = batch.checked_mul(width)?;
    let rectangular_slots = batch.checked_mul(sequence)?;
    let mut counts: Vec<usize> = Vec::with_capacity(batch);
    let mut gather: Vec<usize> = Vec::with_capacity(compact_slots);
    let mut scatter: Vec<usize> = Vec::with_capacity(rectangular_slots);
    let mut packed: Vec<usize> = Vec::with_capacity(tokens);
    let mut unpack: Vec<usize> = Vec::with_capacity(compact_slots);
    let mut last: Vec<usize> = Vec::with_capacity(batch);
    // `compact_start = row * width` and `rectangular_start = row * sequence`
    // advance by addition, so no product is formed inside the loop.
    let mut compact_start: usize = 0;
    let mut rectangular_start: usize = 0;
    row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            batch == columns.len(),
            row <= batch,
            compact_slots == batch * width,
            rectangular_slots == batch * sequence,
            compact_start == row * width,
            rectangular_start == row * sequence,
            tokens == total_tokens(columns@, batch as int),
            width > 0 ==> sequence > 0,
            forall|b: int| 0 <= b < batch ==> {
                &&& #[trigger] columns@[b]@.len() <= width
                &&& sorted_unique_below(columns@[b]@, sequence as int)
            },
            counts.len() == row,
            forall|b: int| 0 <= b < row ==> #[trigger] counts@[b] == columns@[b]@.len(),
            gather.len() == row * width,
            unpack.len() == row * width,
            scatter.len() == row * sequence,
            last.len() == row,
            packed.len() == total_tokens(columns@, row as int),
            gathers_rows(gather@, columns@, row as int, width as int, sequence as int),
            unpacks_rows(unpack@, columns@, row as int, width as int, tokens as int),
            packs_rows(packed@, columns@, row as int, width as int),
            scatters_rows(scatter@, columns@, row as int, width as int, sequence as int, compact_slots as int),
            lasts_rows(last@, columns@, row as int, width as int, compact_slots as int),
            all_below(gather@, batch * sequence),
            all_at_most(scatter@, batch * width),
            all_below(packed@, batch * width),
            all_at_most(unpack@, tokens as int),
            all_at_most(last@, batch * width),
        decreases batch - row,
    ))]
    while row < batch {
        let valid = &columns[row];
        let count = valid.len();
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_row_block(row as int, batch as int, width as int);
            lemma_row_block(row as int, batch as int, sequence as int);
            lemma_total_step(columns@, row as int);
            assert(total_tokens(columns@, row as int + 1) <= total_tokens(columns@, batch as int))
                by {
                    lemma_total_monotone(columns@, row as int + 1, batch as int);
                };
            lemma_total_nonnegative(columns@, row as int);
            // Entries of earlier rows lie before everything this row appends.
            assert forall|b: int, j: int| 0 <= b < row && 0 <= j < width
                implies 0 <= #[trigger] flat(b, j, width as int) < row * width by {
                lemma_flat_below(b, j, row as int, width as int);
            }
            assert forall|b: int, j: int| 0 <= b < row && 0 <= j < sequence
                implies 0 <= #[trigger] flat(b, j, sequence as int) < row * sequence by {
                lemma_flat_below(b, j, row as int, sequence as int);
            }
            assert forall|b: int, j: int| 0 <= b < row && 0 <= j < columns@[b]@.len()
                implies 0 <= #[trigger] packed_index(columns@, b, j)
                    < total_tokens(columns@, row as int) by {
                lemma_packed_below(columns@, b, row as int, j);
            }
        }
        counts.push(count);
        let mut slot: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                batch == columns.len(),
                row < batch,
                valid@ == columns@[row as int]@,
                count == columns@[row as int]@.len(),
                count <= width,
                sorted_unique_below(columns@[row as int]@, sequence as int),
                rectangular_slots == batch * sequence,
                width > 0 ==> sequence > 0,
                compact_start == row * width,
                rectangular_start == row * sequence,
                compact_start + width <= batch * width,
                rectangular_start + sequence <= batch * sequence,
                compact_slots == batch * width,
                tokens == total_tokens(columns@, batch as int),
                total_tokens(columns@, row as int) >= 0,
                total_tokens(columns@, row as int + 1) <= tokens,
                total_tokens(columns@, row as int + 1)
                    == total_tokens(columns@, row as int) + count,
                slot <= width,
                gather.len() == row * width + slot,
                unpack.len() == row * width + slot,
                packed.len() == total_tokens(columns@, row as int)
                    + if slot < count { slot as int } else { count as int },
                rows_below(row as int, width as int),
                packed_rows_below(columns@, row as int),
                gathers_rows(gather@, columns@, row as int, width as int, sequence as int),
                unpacks_rows(unpack@, columns@, row as int, width as int, tokens as int),
                packs_rows(packed@, columns@, row as int, width as int),
                forall|j: int| 0 <= j < slot ==> #[trigger] gather@[flat(row as int, j, width as int)]
                    == flat(row as int, if j < count { valid@[j] as int } else { 0 }, sequence as int),
                forall|j: int| 0 <= j < slot ==> #[trigger] unpack@[flat(row as int, j, width as int)]
                    == if j < count { packed_index(columns@, row as int, j) } else { tokens as int },
                forall|j: int| 0 <= j < slot && j < count
                    ==> #[trigger] packed@[packed_index(columns@, row as int, j)]
                        == flat(row as int, j, width as int),
                all_below(gather@, batch * sequence),
                all_below(packed@, batch * width),
                all_at_most(unpack@, tokens as int),
            decreases width - slot,
        ))]
        while slot < width {
            if slot < count {
                gather.push(rectangular_start + valid[slot]);
                unpack.push(packed.len());
                packed.push(compact_start + slot);
            } else {
                // Padding addresses the zero row appended after the packed tokens.
                gather.push(rectangular_start);
                unpack.push(tokens);
            }
            slot += 1;
        }
        // Valid columns increase, so one merge pass places compact slot `k`
        // at column `c_k` and the zero row everywhere else.
        let mut next: usize = 0;
        let mut column: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                batch == columns.len(),
                row < batch,
                valid@ == columns@[row as int]@,
                count == columns@[row as int]@.len(),
                count <= width,
                sorted_unique_below(columns@[row as int]@, sequence as int),
                forall|b: int| 0 <= b < batch ==> {
                    &&& #[trigger] columns@[b]@.len() <= width
                    &&& sorted_unique_below(columns@[b]@, sequence as int)
                },
                compact_start == row * width,
                rectangular_start == row * sequence,
                compact_start + width <= batch * width,
                compact_slots == batch * width,
                column <= sequence,
                next <= count,
                scatter.len() == row * sequence + column,
                rows_below(row as int, sequence as int),
                scatters_rows(scatter@, columns@, row as int, width as int, sequence as int, compact_slots as int),
                // Matched columns lie before the cursor; unmatched ones at or after it.
                forall|k: int| 0 <= k < next ==> #[trigger] valid@[k] < column,
                forall|k: int| next <= k < count ==> #[trigger] valid@[k] >= column,
                forall|k: int| 0 <= k < next
                    ==> #[trigger] scatter@[flat(row as int, valid@[k] as int, sequence as int)]
                        == flat(row as int, k, width as int),
                forall|s: int| 0 <= s < column && !valid@.contains(s as usize)
                    ==> #[trigger] scatter@[flat(row as int, s, sequence as int)] == compact_slots,
                all_at_most(scatter@, batch * width),
            decreases sequence - column,
        ))]
        while column < sequence {
            if next < count && valid[next] == column {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(valid@.contains(column));
                }
                scatter.push(compact_start + next);
                next += 1;
            } else {
                scatter.push(compact_slots);
            }
            column += 1;
        }
        // Every valid column lies below `sequence`, so the merge matched all of them.
        #[cfg(verus_keep_ghost)]
        proof! {
            if next < count {
                assert(valid@[next as int] >= column);
                assert(valid@[next as int] < sequence);
            }
        }
        last.push(if count == 0 {
            compact_slots
        } else {
            compact_start + count - 1
        });
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row as int + 1) * width as int == row as int * width as int + width as int)
                by (nonlinear_arith);
            assert((row as int + 1) * sequence as int == row as int * sequence as int + sequence as int)
                by (nonlinear_arith);
            let next_row = row as int + 1;
            assert forall|b: int, j: int| 0 <= b < next_row && 0 <= j < width
                implies #[trigger] gather@[flat(b, j, width as int)]
                    == flat(b, if j < columns@[b]@.len() { columns@[b]@[j] as int } else { 0 }, sequence as int)
            by {}
            assert forall|b: int, j: int| 0 <= b < next_row && 0 <= j < width
                implies #[trigger] unpack@[flat(b, j, width as int)]
                    == if j < columns@[b]@.len() { packed_index(columns@, b, j) } else { tokens as int }
            by {}
            assert forall|b: int, j: int| 0 <= b < next_row && 0 <= j < columns@[b]@.len()
                implies #[trigger] packed@[packed_index(columns@, b, j)] == flat(b, j, width as int)
            by {}
            assert forall|b: int, k: int| 0 <= b < next_row && 0 <= k < columns@[b]@.len()
                implies #[trigger] scatter@[flat(b, columns@[b]@[k] as int, sequence as int)]
                    == flat(b, k, width as int)
            by {}
            assert forall|b: int, s: int|
                0 <= b < next_row && 0 <= s < sequence && !columns@[b]@.contains(s as usize)
                implies #[trigger] scatter@[flat(b, s, sequence as int)] == compact_slots
            by {}
            assert forall|b: int| 0 <= b < next_row
                implies #[trigger] last@[b] == if columns@[b]@.len() == 0 {
                    compact_slots as int
                } else {
                    flat(b, columns@[b]@.len() - 1, width as int)
                }
            by {}
        }
        compact_start += width;
        rectangular_start += sequence;
        row += 1;
    }
    // An `unpack` entry equals `T` exactly at padded slots.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|b: int, j: int| 0 <= b < batch && 0 <= j < width
            implies (#[trigger] unpack@[flat(b, j, width as int)] == tokens)
                == (j >= columns@[b]@.len()) by {
            if j < columns@[b]@.len() {
                lemma_packed_below(columns@, b, batch as int, j);
            }
        }
    }
    Some(CompactPlan {
        sequence,
        width,
        counts,
        gather,
        scatter,
        packed,
        unpack,
        last,
    })
}

/// Absolute query position of compact slot `slot` in a row with `count` valid tokens after `baseline`.
///
/// A valid slot (`slot < count`) sits at `W + slot`. A padded slot repeats the
/// row's last readable position `max(W + n, 1) - 1`, the position of its last
/// valid (or, for an inactive row, last cached) key. Every key at or before
/// that position is readable and the key at that position lies inside any
/// sliding window, so a padded query of a row with history is never fully
/// excluded; its output is discarded, but it stays a finite, ordinary softmax
/// on every backend. Numbering padded slots `W + slot` instead could push a
/// padded query beyond a sliding window of every valid key. Returns `None`
/// when `W + n` overflows.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(position =>
    ensures
        match position {
            Some(position) => {
                &&& baseline + count <= usize::MAX
                &&& position == slot_position_spec(baseline as int, count as int, slot as int)
                &&& slot < count ==> position == baseline + slot
                &&& slot >= count && baseline + count > 0 ==> position + 1 == baseline + count
                &&& slot >= count && baseline + count == 0 ==> position == 0
                &&& position <= baseline + slot
                &&& baseline + count > 0 ==> position < baseline + count
            },
            None => baseline + count > usize::MAX,
        },
))]
pub(crate) const fn slot_position(baseline: usize, count: usize, slot: usize) -> Option<usize> {
    match baseline.checked_add(count) {
        None => None,
        Some(_) if slot < count => Some(baseline + slot),
        Some(end) if end > 0 => Some(end - 1),
        Some(_) => Some(0),
    }
}

/// Absolute query positions of every compact slot `(b, j)`, `[B * N]`.
///
/// Slot `(b, j)` holds [`slot_position`]`(W_b, n_b, j)`: valid slots continue
/// the row's numbering and padded slots repeat its last readable position, so
/// no query of a row with history is ever fully excluded by causality or a
/// sliding window. Returns `None` exactly when `baselines` and `counts`
/// differ in length, `B * N` overflows, or some slot's position exceeds
/// `usize` or `i64`; the proof establishes the exact value of every slot.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(positions =>
    ensures
        match positions {
            Some(positions) => {
                &&& counts.len() == baselines.len()
                &&& positions.len() == baselines.len() * width
                &&& forall|b: int, j: int| 0 <= b < baselines.len() && 0 <= j < width
                    ==> #[trigger] positions@[flat(b, j, width as int)]
                        == slot_position_spec(baselines@[b] as int, counts@[b] as int, j)
                &&& forall|i: int| 0 <= i < positions.len() ==> #[trigger] positions@[i] >= 0
            },
            None => {
                ||| counts.len() != baselines.len()
                ||| baselines.len() * width > usize::MAX
                ||| exists|b: int, j: int| 0 <= b < baselines.len() && 0 <= j < width && (
                    baselines@[b] + counts@[b] > usize::MAX
                    || #[trigger] slot_position_spec(baselines@[b] as int, counts@[b] as int, j)
                        > i64::MAX)
            },
        },
))]
pub(crate) fn query_positions(
    baselines: &[usize],
    counts: &[usize],
    width: usize,
) -> Option<Vec<i64>> {
    if counts.len() != baselines.len() {
        return None;
    }
    let total = baselines.len().checked_mul(width)?;
    let mut positions: Vec<i64> = Vec::with_capacity(total);
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            counts.len() == baselines.len(),
            row <= baselines.len(),
            positions.len() == row * width,
            forall|b: int, j: int| 0 <= b < row && 0 <= j < width
                ==> #[trigger] positions@[flat(b, j, width as int)]
                    == slot_position_spec(baselines@[b] as int, counts@[b] as int, j),
            forall|i: int| 0 <= i < positions.len() ==> #[trigger] positions@[i] >= 0,
        decreases baselines.len() - row,
    ))]
    while row < baselines.len() {
        // Entries of earlier rows lie before everything this row appends.
        #[cfg(verus_keep_ghost)]
        proof! {
            assert forall|b: int, j: int| 0 <= b < row && 0 <= j < width
                implies 0 <= #[trigger] flat(b, j, width as int) < row * width by {
                lemma_flat_below(b, j, row as int, width as int);
            }
        }
        let mut slot: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                counts.len() == baselines.len(),
                row < baselines.len(),
                slot <= width,
                positions.len() == row * width + slot,
                rows_below(row as int, width as int),
                forall|b: int, j: int| 0 <= b < row && 0 <= j < width
                    ==> #[trigger] positions@[flat(b, j, width as int)]
                        == slot_position_spec(baselines@[b] as int, counts@[b] as int, j),
                forall|j: int| 0 <= j < slot
                    ==> #[trigger] positions@[flat(row as int, j, width as int)]
                        == slot_position_spec(baselines@[row as int] as int, counts@[row as int] as int, j),
                forall|i: int| 0 <= i < positions.len() ==> #[trigger] positions@[i] >= 0,
            decreases width - slot,
        ))]
        while slot < width {
            // Slot `(row, slot)` witnesses either failure below.
            #[cfg(verus_keep_ghost)]
            proof! {
                let ghost spec_position = slot_position_spec(
                    baselines@[row as int] as int,
                    counts@[row as int] as int,
                    slot as int,
                );
                assert(baselines@[row as int] + counts@[row as int] > usize::MAX
                    || spec_position > i64::MAX
                    ==> (0 <= row < baselines.len() && 0 <= slot < width && (
                        baselines@[row as int] + counts@[row as int] > usize::MAX
                        || slot_position_spec(
                            baselines@[row as int] as int,
                            counts@[row as int] as int,
                            slot as int,
                        ) > i64::MAX)));
            }
            let position = slot_position(baselines[row], counts[row], slot)?;
            // `usize` has at most 64 bits, so widening to `u64` is lossless.
            if position as u64 > i64::MAX as u64 {
                return None;
            }
            positions.push(position as i64);
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row as int + 1) * width as int == row as int * width as int + width as int)
                by (nonlinear_arith);
            assert forall|b: int, j: int| 0 <= b < row + 1 && 0 <= j < width
                implies #[trigger] positions@[flat(b, j, width as int)]
                    == slot_position_spec(baselines@[b] as int, counts@[b] as int, j) by {}
        }
        row += 1;
    }
    Some(positions)
}

/// Mamba-2 execution path of one stateful row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MambaStep {
    /// The reference single-step selective-state update.
    Recurrent,
    /// The reference chunked SSD scan (with its prefill dt clamp).
    Chunked,
}

/// Chooses the reference Mamba-2 path for `tokens` new tokens after `baseline` cached ones.
///
/// The pinned reference takes its single-step update only when
/// `use_precomputed_states and seq_len == 1`; a first prompt token, and every
/// multi-token segment, runs the chunk scan.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(step =>
    ensures
        (step == MambaStep::Recurrent) == (tokens == 1 && baseline > 0),
))]
pub(crate) const fn mamba_step(tokens: usize, baseline: usize) -> MambaStep {
    if tokens == 1 && baseline > 0 {
        MambaStep::Recurrent
    } else {
        MambaStep::Chunked
    }
}

/// How a transactional step records Mamba-2 state.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransactionalMamba {
    /// Every active row appends one token: its terminal state is its only snapshot.
    Terminal,
    /// Some row appends several tokens: every active row runs the recurrent
    /// update and records its state after each token.
    Snapshots,
}

/// Plans recurrent staging for a step inside a speculative transaction.
///
/// A rejected suffix cannot be removed from a recurrence, so a multi-token
/// proposal must keep the state after every token. Only the recurrent update
/// produces per-token states, and the reference uses it only for rows that
/// already have state; a row that would start its prompt inside a
/// transaction is therefore reported as `Err(row)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(policy =>
    ensures
        match policy {
            Ok(TransactionalMamba::Terminal) =>
                forall|r: int| 0 <= r < counts.len() ==> #[trigger] counts@[r] <= 1,
            Ok(TransactionalMamba::Snapshots) => {
                &&& exists|r: int| 0 <= r < counts.len() && #[trigger] counts@[r] > 1
                &&& forall|r: int| 0 <= r < counts.len() && #[trigger] counts@[r] > 0
                    ==> r < baselines.len() && baselines@[r] > 0
            },
            Err(row) => {
                &&& exists|r: int| 0 <= r < counts.len() && #[trigger] counts@[r] > 1
                &&& row < counts.len()
                &&& counts@[row as int] > 0
                &&& (row >= baselines.len() || baselines@[row as int] == 0)
            },
        },
))]
pub(crate) fn transactional_mamba(
    counts: &[usize],
    baselines: &[usize],
) -> Result<TransactionalMamba, usize> {
    let mut row: usize = 0;
    let mut multi_token = false;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            multi_token == exists|r: int| 0 <= r < row && #[trigger] counts@[r] > 1,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] > 1 {
            multi_token = true;
        }
        row += 1;
    }
    if !multi_token {
        return Ok(TransactionalMamba::Terminal);
    }
    row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            exists|r: int| 0 <= r < counts.len() && #[trigger] counts@[r] > 1,
            forall|r: int| 0 <= r < row && #[trigger] counts@[r] > 0
                ==> r < baselines.len() && baselines@[r] > 0,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] > 0 && (row >= baselines.len() || baselines[row] == 0) {
            return Err(row);
        }
        row += 1;
    }
    Ok(TransactionalMamba::Snapshots)
}

/// Maps a target truncation to the shifted MTP cursor.
///
/// MTP position `i` consumes `(x_{i+1}, h_i)`, so it can never be ahead of
/// `target - 1`. The result only rewinds (`<= mtp`) and keeps
/// `result + 1 <= max(target, 1)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(length =>
    ensures
        length <= mtp,
        length + 1 <= if target > 0 { target as int } else { 1 },
        length == mtp || length + 1 == target || (target == 0 && length == 0),
))]
pub(crate) const fn mtp_length_after_truncate(target: usize, mtp: usize) -> usize {
    let limit = target.saturating_sub(1);
    if mtp < limit { mtp } else { limit }
}

/// Number of target tokens one verification pass must evaluate.
///
/// The pass consumes the lead token and the drafts `y_1..y_d`. When the last
/// draft is end-of-sequence its successor distribution is never needed: an
/// accepted EOS ends the cycle and a rejection stops earlier, so `y_d` itself
/// is not evaluated. Returns `None` on overflow.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(tokens =>
    ensures
        match tokens {
            Some(tokens) => {
                &&& last_draft_is_eos && drafted > 0 ==> tokens == drafted
                &&& !(last_draft_is_eos && drafted > 0) ==> tokens == drafted + 1
                &&& tokens >= 1
            },
            None => !(last_draft_is_eos && drafted > 0) && drafted == usize::MAX,
        },
))]
pub(crate) const fn verification_tokens(drafted: usize, last_draft_is_eos: bool) -> Option<usize> {
    if last_draft_is_eos && drafted > 0 {
        Some(drafted)
    } else {
        drafted.checked_add(1)
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
