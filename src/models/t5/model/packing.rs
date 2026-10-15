//! Host plans that pack ragged stateful rows into one batched block and back.
//!
//! A stateful call supplies a `[batch, S]` rectangle whose row `b` has valid
//! columns `c_{b,0} < … < c_{b,n_b-1}` (`n_b = 0` marks an inactive decode
//! row). The batched decoder evaluates a left-aligned `[batch, Q]` block with
//! `Q = max_b n_b`: slot `(b, i)` holds row `b`'s `i`-th valid token for
//! `i < n_b` and padding for `n_b <= i < Q`, so every row's new tokens occupy
//! columns `0..n_b` with padding after them, exactly the layout the cache's
//! batched append expects (slot `i` is written to cache position `L_b + i`).
//!
//! Packing and unpacking are gathers over flattened row-major indices:
//!
//! - compaction source of slot `(b, i)`: `b * S + c_{b,i}` for `i < n_b`, and
//!   `b * S` (any in-row column; the slot is overwritten) otherwise;
//! - scatter source of column `(b, s)`: `b * Q + k` when `s = c_{b,k}`, and
//!   `b * Q` otherwise (the column is zeroed after the gather).
//!
//! When every row's valid columns are a prefix `0..n_b` (no padding, right
//! padding, or inactive rows), both maps are views — a narrow and a zero
//! extension — and no index tensor is uploaded.
//!
//! # Verified planning
//!
//! Every function here is dependency-free integer logic, verified in place by
//! Verus (`verus/t5_gpt.rs`). [`plan_rows`] (behind [`BatchedRows::new`]) is
//! proved end to end: the block width is the largest row count, the prefix
//! layout is chosen exactly when every row is `0..n_b`, otherwise every
//! compaction and scatter entry equals its formula above in terms of the
//! caller's columns, and a plan is refused only for a row that is not
//! strictly increasing below `S` or an unrepresentable `B * S`.
//! [`compaction_sources`] and [`scatter_sources`] state every returned entry
//! and characterize each refusal; [`padded_slots`] and [`has_padded_slot`]
//! mark exactly the slots `i >= n_b`. [`row_positions`], which plans every
//! slot's causal visibility and relative-bias position, computes exactly its
//! documented formulas, never letting a slot see past its row's appended
//! history, and [`groups_preserved`] exactly characterizes group-preserving
//! beam reorders.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Valid columns in rows `0..rows`: the offset `o_rows` of row `rows` in the flat column list.
pub open spec fn prefix_count(counts: Seq<usize>, rows: int) -> int
    decreases rows,
{
    if rows <= 0 {
        0
    } else {
        prefix_count(counts, rows - 1) + counts[rows - 1]
    }
}

/// Appending one row's count extends the offset: `o_{b+1} = o_b + n_b`.
proof fn lemma_prefix_step(counts: Seq<usize>, row: int)
    requires
        0 <= row,
    ensures
        prefix_count(counts, row + 1) == prefix_count(counts, row) + counts[row],
{
}

/// Row-major flat index of entry `(row, column)` in a matrix `width` columns wide.
pub open spec fn flat(row: int, column: int, width: int) -> int {
    row * width + column
}

/// Running offsets never decrease: `o_rows <= o_more` for `rows <= more`.
proof fn lemma_prefix_monotone(counts: Seq<usize>, rows: int, more: int)
    requires
        0 <= rows <= more,
    ensures
        prefix_count(counts, rows) <= prefix_count(counts, more),
    decreases more - rows,
{
    if rows < more {
        lemma_prefix_monotone(counts, rows, more - 1);
    }
}

/// Running offsets are never negative.
proof fn lemma_prefix_nonnegative(counts: Seq<usize>, rows: int)
    ensures
        prefix_count(counts, rows) >= 0,
    decreases rows,
{
    if rows > 0 {
        lemma_prefix_nonnegative(counts, rows - 1);
    }
}

/// Flat-list index `o_row + i` of valid column `i` of row `row`.
pub open spec fn column_index(counts: Seq<usize>, row: int, i: int) -> int {
    prefix_count(counts, row) + i
}

/// Valid column `i` of row `row`, `columns[o_row + i]`.
pub open spec fn row_column(columns: Seq<usize>, counts: Seq<usize>, row: int, i: int) -> int {
    columns[column_index(counts, row, i)] as int
}

/// `s` is strictly increasing with every entry below `bound`.
pub open spec fn increasing_below(s: Seq<usize>, bound: int) -> bool {
    &&& forall|i: int, j: int| 0 <= i < j < s.len() ==> s[i] < s[j]
    &&& forall|i: int| 0 <= i < s.len() ==> s[i] < bound
}

/// A row's valid columns are exactly `0..n`: no leading or interior padding.
pub open spec fn is_prefix_row(s: Seq<usize>) -> bool {
    forall|i: int| 0 <= i < s.len() ==> s[i] == i
}

/// Entry `i` of a strictly increasing sequence of naturals is at least `i`.
proof fn lemma_increasing_at_least_index(s: Seq<usize>, i: int)
    requires
        forall|a: int, b: int| 0 <= a < b < s.len() ==> s[a] < s[b],
        0 <= i < s.len(),
    ensures
        s[i] >= i,
    decreases i,
{
    if i > 0 {
        lemma_increasing_at_least_index(s, i - 1);
    }
}

/// A strictly increasing sequence below `bound` has at most `bound` entries.
proof fn lemma_increasing_length(s: Seq<usize>, bound: int)
    requires
        bound >= 0,
        increasing_below(s, bound),
    ensures
        s.len() <= bound,
{
    if s.len() > 0 {
        lemma_increasing_at_least_index(s, s.len() - 1);
        assert(s[s.len() - 1] < bound);
    }
}

/// A row stored in a flat column list is sorted exactly when the caller's row is.
proof fn lemma_row_matches(
    valid: Seq<usize>,
    columns: Seq<usize>,
    counts: Seq<usize>,
    row: int,
    bound: int,
)
    requires
        counts[row] == valid.len(),
        forall|i: int| 0 <= i < counts[row]
            ==> #[trigger] row_column(columns, counts, row, i) == valid[i] as int,
    ensures
        row_sorted_below(columns, counts, row, bound) == increasing_below(valid, bound),
        forall|s: int| 0 <= s <= usize::MAX ==> (in_row(columns, counts, row, s)
            == #[trigger] valid.contains(s as usize)),
{
    if increasing_below(valid, bound) {
        assert forall|i: int, j: int| 0 <= i < j < counts[row]
            implies row_column(columns, counts, row, i) < row_column(columns, counts, row, j) by {
            assert(row_column(columns, counts, row, i) == valid[i] as int);
            assert(row_column(columns, counts, row, j) == valid[j] as int);
        }
    }
    if row_sorted_below(columns, counts, row, bound) {
        assert forall|i: int, j: int| 0 <= i < j < valid.len() implies valid[i] < valid[j] by {
            assert(row_column(columns, counts, row, i) == valid[i] as int);
            assert(row_column(columns, counts, row, j) == valid[j] as int);
        }
        assert forall|i: int| 0 <= i < valid.len() implies valid[i] < bound by {
            assert(row_column(columns, counts, row, i) == valid[i] as int);
        }
    }
    assert forall|s: int| 0 <= s <= usize::MAX implies (in_row(columns, counts, row, s)
        == #[trigger] valid.contains(s as usize)) by {
        if in_row(columns, counts, row, s) {
            let k = choose|k: int| 0 <= k < counts[row]
                && #[trigger] row_column(columns, counts, row, k) == s;
            assert(valid[k] == s as usize);
        }
        if valid.contains(s as usize) {
            let k = choose|k: int| 0 <= k < valid.len() && #[trigger] valid[k] == s as usize;
            assert(row_column(columns, counts, row, k) == s);
        }
    }
}

/// Whether column `s` is one of row `row`'s valid columns.
pub open spec fn in_row(columns: Seq<usize>, counts: Seq<usize>, row: int, s: int) -> bool {
    exists|k: int| 0 <= k < counts[row] && #[trigger] row_column(columns, counts, row, k) == s
}

/// Row `row`'s valid columns are strictly increasing and below `bound`.
pub open spec fn row_sorted_below(
    columns: Seq<usize>,
    counts: Seq<usize>,
    row: int,
    bound: int,
) -> bool {
    &&& forall|i: int, j: int| 0 <= i < j < counts[row]
        ==> row_column(columns, counts, row, i) < row_column(columns, counts, row, j)
    &&& forall|i: int| 0 <= i < counts[row] ==> #[trigger] row_column(columns, counts, row, i) < bound
}

/// Every flat entry of rows `0..rows` lies before the first entry of row `rows`.
pub open spec fn rows_below(rows: int, width: int) -> bool {
    forall|b: int, j: int| 0 <= b < rows && 0 <= j < width
        ==> 0 <= #[trigger] flat(b, j, width) < rows * width
}

/// Proves [`rows_below`] for any row count.
proof fn lemma_rows_below(rows: int, width: int)
    requires
        rows >= 0,
    ensures
        rows_below(rows, width),
{
    assert forall|b: int, j: int| 0 <= b < rows && 0 <= j < width
        implies 0 <= #[trigger] flat(b, j, width) < rows * width by {
        assert(b * width + width <= rows * width) by (nonlinear_arith)
            requires 0 <= b < rows, 0 <= j < width;
        assert(b * width >= 0) by (nonlinear_arith)
            requires 0 <= b, 0 <= j < width;
    }
}

/// Row `row`'s block `[row * width, (row + 1) * width)` lies inside `batch * width`.
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

} // verus!

/// How a call's valid tokens map onto the batched `[batch, Q]` block.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verifier::allow(autoderive_clone_without_spec))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Packing {
    /// Every row's valid columns are `0..n_b`: the block is the first `Q` columns.
    Prefix,
    /// Some row has leading or interior padding: flat gather indices.
    Gather {
        /// Compaction source of every block slot, `batch * Q` entries.
        compact: Vec<usize>,
        /// Scatter source of every input column, `batch * S` entries.
        scatter: Vec<usize>,
    },
}

/// Packing plan for one validated stateful call.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verifier::allow(autoderive_clone_without_spec))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BatchedRows {
    /// Rows `B`.
    pub(super) batch: usize,
    /// Input columns `S`, including padding.
    pub(super) sequence: usize,
    /// Block width `Q = max_b n_b`; zero when every row is inactive.
    pub(super) block: usize,
    /// New tokens `n_b` of every row.
    pub(super) counts: Vec<usize>,
    /// Kernel strategy for packing and unpacking.
    pub(super) packing: Packing,
}

impl BatchedRows {
    /// Plans a call from each row's sorted valid columns below `sequence`.
    ///
    /// Returns `None` when the columns are malformed (unsorted, duplicated,
    /// out of range) or a flat index would overflow `usize`; validated
    /// stateful inputs never are. See [`plan_rows`] for the verified plan.
    pub(super) fn new(valid_columns: &[Vec<usize>], sequence: usize) -> Option<Self> {
        plan_rows(valid_columns, sequence)
    }

    /// Whether block slot `(b, i)` is padding (`i >= n_b`), row-major over `[batch, Q]`.
    pub(super) fn padded_slots(&self) -> Vec<bool> {
        padded_slots(&self.counts, self.block)
    }

    /// Whether any block slot is padding.
    pub(super) fn has_padded_slots(&self) -> bool {
        has_padded_slot(&self.counts, self.block)
    }

    /// Whether every row is inactive (no new token anywhere).
    pub(super) fn is_empty(&self) -> bool {
        self.block == 0
    }
}

/// Plans a call from each row's valid columns `valid_columns[b]` below `sequence`.
///
/// The block width is the largest row count `Q = max_b n_b`. When every
/// row's valid columns are exactly `0..n_b` the block is a prefix view of
/// the input; otherwise the plan holds the flat [`compaction_sources`] and
/// [`scatter_sources`] of the rows. The proof establishes every entry of
/// both gathers in terms of the caller's columns (slot `(b, i)` reads column
/// `c_{b,i}` of row `b`, and column `c_{b,k}` reads slot `(b, k)`), that a
/// plan exists only when every row is strictly increasing below `sequence`,
/// and that a refusal means a malformed row or an unrepresentable `B * S`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(planned =>
    ensures
        match planned {
            Some(rows) => {
                let batch = valid_columns.len() as int;
                let block = rows.block as int;
                &&& rows.batch == valid_columns.len()
                &&& rows.sequence == sequence
                &&& rows.counts.len() == batch
                &&& rows.block <= sequence
                &&& forall|b: int| 0 <= b < batch ==> {
                    &&& #[trigger] rows.counts@[b] == valid_columns@[b]@.len()
                    &&& rows.counts@[b] <= rows.block
                    &&& increasing_below(valid_columns@[b]@, sequence as int)
                }
                &&& rows.block > 0
                    ==> exists|b: int| 0 <= b < batch && #[trigger] valid_columns@[b]@.len() == rows.block
                &&& (rows.packing is Prefix)
                    == forall|b: int| 0 <= b < batch ==> is_prefix_row(#[trigger] valid_columns@[b]@)
                &&& match rows.packing {
                    Packing::Prefix => true,
                    Packing::Gather { compact, scatter } => {
                        &&& compact.len() == batch * block
                        &&& scatter.len() == batch * sequence
                        &&& forall|b: int, i: int| 0 <= b < batch && 0 <= i < block
                            ==> #[trigger] compact@[flat(b, i, block)] == flat(
                                b,
                                if i < valid_columns@[b]@.len() { valid_columns@[b]@[i] as int } else { 0 },
                                sequence as int,
                            )
                        &&& forall|b: int, k: int| 0 <= b < batch && 0 <= k < valid_columns@[b]@.len()
                            ==> #[trigger] scatter@[flat(b, valid_columns@[b]@[k] as int, sequence as int)]
                                == flat(b, k, block)
                        &&& forall|b: int, s: int| 0 <= b < batch && 0 <= s < sequence
                            && !valid_columns@[b]@.contains(s as usize)
                            ==> #[trigger] scatter@[flat(b, s, sequence as int)] == flat(b, 0, block)
                    },
                }
            },
            None => {
                ||| exists|b: int| 0 <= b < valid_columns.len()
                    && !increasing_below(#[trigger] valid_columns@[b]@, sequence as int)
                ||| valid_columns.len() * sequence > usize::MAX
            },
        },
))]
pub(super) fn plan_rows(valid_columns: &[Vec<usize>], sequence: usize) -> Option<BatchedRows> {
    let batch = valid_columns.len();
    let mut counts: Vec<usize> = Vec::with_capacity(batch);
    let mut block: usize = 0;
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            batch == valid_columns.len(),
            row <= batch,
            counts.len() == row,
            forall|b: int| 0 <= b < row ==> {
                &&& #[trigger] counts@[b] == valid_columns@[b]@.len()
                &&& counts@[b] <= block
            },
            block > 0 ==> exists|b: int| 0 <= b < row && #[trigger] valid_columns@[b]@.len() == block,
        decreases batch - row,
    ))]
    while row < batch {
        let count = valid_columns[row].len();
        counts.push(count);
        if count > block {
            block = count;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(counts@[row as int] == count);
            assert(valid_columns@[row as int]@.len() == count);
        }
        row += 1;
    }
    if block > sequence {
        // A row longer than `sequence` cannot be strictly increasing below it.
        #[cfg(verus_keep_ghost)]
        proof! {
            let b = choose|b: int| 0 <= b < batch && #[trigger] valid_columns@[b]@.len() == block;
            if increasing_below(valid_columns@[b]@, sequence as int) {
                lemma_increasing_length(valid_columns@[b]@, sequence as int);
            }
        }
        return None;
    }
    // Every row's valid columns are `0..n_b` exactly when no row has leading
    // or interior padding; the scan stops at the first displaced column.
    let mut prefix = true;
    row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            batch == valid_columns.len(),
            row <= batch,
            prefix ==> forall|b: int| 0 <= b < row ==> is_prefix_row(#[trigger] valid_columns@[b]@),
            !prefix ==> exists|b: int| 0 <= b < batch && !is_prefix_row(#[trigger] valid_columns@[b]@),
        decreases batch - row,
    ))]
    while prefix && row < batch {
        let columns = &valid_columns[row];
        let mut slot: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                batch == valid_columns.len(),
                row < batch,
                columns@ == valid_columns@[row as int]@,
                slot <= columns.len(),
                prefix ==> forall|b: int| 0 <= b < row ==> is_prefix_row(#[trigger] valid_columns@[b]@),
                prefix ==> forall|i: int| 0 <= i < slot ==> #[trigger] columns@[i] == i,
                !prefix ==> exists|b: int| 0 <= b < batch && !is_prefix_row(#[trigger] valid_columns@[b]@),
            decreases columns.len() - slot,
        ))]
        while prefix && slot < columns.len() {
            if columns[slot] != slot {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(!is_prefix_row(valid_columns@[row as int]@));
                }
                prefix = false;
            }
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            if prefix {
                assert(is_prefix_row(valid_columns@[row as int]@));
            }
        }
        row += 1;
    }
    let packing = if prefix {
        // Prefix rows `0..n_b` with `n_b <= Q <= S` are strictly increasing below `S`.
        #[cfg(verus_keep_ghost)]
        proof! {
            assert forall|b: int| 0 <= b < batch
                implies increasing_below(#[trigger] valid_columns@[b]@, sequence as int) by {
                assert(is_prefix_row(valid_columns@[b]@));
                assert(counts@[b] <= block);
            }
        }
        Packing::Prefix
    } else {
        let mut stored: Vec<usize> = Vec::new();
        row = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                batch == valid_columns.len(),
                row <= batch,
                counts.len() == batch,
                forall|b: int| 0 <= b < batch ==> #[trigger] counts@[b] == valid_columns@[b]@.len(),
                stored.len() == prefix_count(counts@, row as int),
                forall|b: int, i: int| 0 <= b < row && 0 <= i < counts@[b]
                    ==> #[trigger] stored@[column_index(counts@, b, i)] == valid_columns@[b]@[i],
            decreases batch - row,
        ))]
        while row < batch {
            let columns = &valid_columns[row];
            let mut slot: usize = 0;
            #[cfg(verus_keep_ghost)]
            proof! {
                lemma_prefix_step(counts@, row as int);
                lemma_prefix_nonnegative(counts@, row as int);
                assert forall|b: int, i: int| 0 <= b < row && 0 <= i < counts@[b]
                    implies 0 <= #[trigger] column_index(counts@, b, i) < prefix_count(counts@, row as int) by {
                    lemma_prefix_step(counts@, b);
                    lemma_prefix_monotone(counts@, b + 1, row as int);
                    lemma_prefix_nonnegative(counts@, b);
                }
            }
            #[cfg_attr(verus_keep_ghost, verus_spec(
                invariant
                    batch == valid_columns.len(),
                    row < batch,
                    counts.len() == batch,
                    columns@ == valid_columns@[row as int]@,
                    counts@[row as int] == columns.len(),
                    slot <= columns.len(),
                    stored.len() == prefix_count(counts@, row as int) + slot,
                    prefix_count(counts@, row as int) >= 0,
                    forall|b: int, i: int| 0 <= b < row && 0 <= i < counts@[b]
                        ==> 0 <= #[trigger] column_index(counts@, b, i) < prefix_count(counts@, row as int),
                    forall|b: int, i: int| 0 <= b < row && 0 <= i < counts@[b]
                        ==> #[trigger] stored@[column_index(counts@, b, i)] == valid_columns@[b]@[i],
                    forall|i: int| 0 <= i < slot
                        ==> #[trigger] stored@[column_index(counts@, row as int, i)] == columns@[i],
                decreases columns.len() - slot,
            ))]
            while slot < columns.len() {
                stored.push(columns[slot]);
                slot += 1;
            }
            row += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_prefix_nonnegative(counts@, batch as int);
            assert forall|b: int, i: int| 0 <= b < batch && 0 <= i < counts@[b]
                implies 0 <= #[trigger] column_index(counts@, b, i) < stored.len() by {
                lemma_prefix_step(counts@, b);
                lemma_prefix_monotone(counts@, b + 1, batch as int);
                lemma_prefix_nonnegative(counts@, b);
            }
            if batch * block > usize::MAX {
                assert(batch * block <= batch * sequence) by (nonlinear_arith)
                    requires block <= sequence, batch >= 0;
            }
            // Each stored row is sorted exactly when the caller's row is, and fits the list.
            assert forall|b: int| 0 <= b < batch implies {
                &&& row_sorted_below(stored@, counts@, b, sequence as int)
                    == increasing_below(#[trigger] valid_columns@[b]@, sequence as int)
                &&& prefix_count(counts@, b) + counts@[b] <= stored.len()
                &&& forall|s: int| 0 <= s <= usize::MAX ==> (in_row(stored@, counts@, b, s)
                    == #[trigger] valid_columns@[b]@.contains(s as usize))
            } by {
                lemma_row_matches(valid_columns@[b]@, stored@, counts@, b, sequence as int);
                lemma_prefix_step(counts@, b);
                lemma_prefix_monotone(counts@, b + 1, batch as int);
            }
        }
        let compact = compaction_sources(&counts, &stored, sequence, block)?;
        let scatter = scatter_sources(&counts, &stored, sequence, block)?;
        #[cfg(verus_keep_ghost)]
        proof! {
            assert forall|b: int, i: int| 0 <= b < batch && 0 <= i < block
                implies #[trigger] compact@[flat(b, i, block as int)] == flat(
                    b,
                    if i < valid_columns@[b]@.len() { valid_columns@[b]@[i] as int } else { 0 },
                    sequence as int,
                ) by {
                if i < valid_columns@[b]@.len() {
                    assert(row_column(stored@, counts@, b, i) == valid_columns@[b]@[i]);
                }
            }
            assert forall|b: int, k: int| 0 <= b < batch && 0 <= k < valid_columns@[b]@.len()
                implies #[trigger] scatter@[flat(b, valid_columns@[b]@[k] as int, sequence as int)]
                    == flat(b, k, block as int) by {
                assert(row_column(stored@, counts@, b, k) == valid_columns@[b]@[k]);
            }
            assert forall|b: int, s: int| 0 <= b < batch && 0 <= s < sequence
                && !valid_columns@[b]@.contains(s as usize)
                implies #[trigger] scatter@[flat(b, s, sequence as int)] == flat(b, 0, block as int) by {
                if in_row(stored@, counts@, b, s) {
                    let k = choose|k: int| 0 <= k < counts@[b]
                        && #[trigger] row_column(stored@, counts@, b, k) == s;
                    assert(valid_columns@[b]@[k] == s);
                }
            }
        }
        Packing::Gather { compact, scatter }
    };
    Some(BatchedRows {
        batch,
        sequence,
        block,
        counts,
        packing,
    })
}

/// Whether each block slot `(b, i)` is padding, row-major over `[batch, Q]`.
///
/// The proof establishes `padded[b * Q + i] = (i >= counts[b])` for every slot.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(padded =>
    ensures
        padded.len() == counts.len() * block,
        forall|b: int, i: int| 0 <= b < counts.len() && 0 <= i < block
            ==> #[trigger] padded@[flat(b, i, block as int)] == (i >= counts@[b]),
))]
pub(super) fn padded_slots(counts: &[usize], block: usize) -> Vec<bool> {
    let mut padded: Vec<bool> = Vec::with_capacity(counts.len().saturating_mul(block));
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            padded.len() == row * block,
            forall|b: int, i: int| 0 <= b < row && 0 <= i < block
                ==> #[trigger] padded@[flat(b, i, block as int)] == (i >= counts@[b]),
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        let count = counts[row];
        let mut slot: usize = 0;
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_rows_below(row as int, block as int);
        }
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                row < counts.len(),
                count == counts@[row as int],
                slot <= block,
                padded.len() == row * block + slot,
                rows_below(row as int, block as int),
                forall|b: int, i: int| 0 <= b < row && 0 <= i < block
                    ==> #[trigger] padded@[flat(b, i, block as int)] == (i >= counts@[b]),
                forall|i: int| 0 <= i < slot
                    ==> #[trigger] padded@[flat(row as int, i, block as int)] == (i >= count),
            decreases block - slot,
        ))]
        while slot < block {
            padded.push(slot >= count);
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(row * block + block == (row + 1) * block) by (nonlinear_arith);
        }
        row += 1;
    }
    padded
}

/// Whether any block slot is padding: some row has fewer than `block` new tokens.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(padded =>
    ensures
        padded == exists|b: int| 0 <= b < counts.len() && #[trigger] counts@[b] < block,
))]
pub(super) fn has_padded_slot(counts: &[usize], block: usize) -> bool {
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            forall|b: int| 0 <= b < row ==> #[trigger] counts@[b] >= block,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] < block {
            return true;
        }
        row += 1;
    }
    false
}

/// Lists the flat compaction source of every block slot.
///
/// `counts[b]` valid columns of row `b` are stored consecutively in
/// `columns`, starting at offset `o_b = counts[0] + … + counts[b - 1]`. Slot
/// `(b, i)` reads `b * sequence + columns[o_b + i]` for `i < counts[b]` and
/// `b * sequence` otherwise. Returns `None` exactly for a count above
/// `block`, too few columns, a column at or beyond `sequence`, a padded slot
/// without any column to read (`sequence = 0`), or an unrepresentable
/// extent; the proof establishes every returned entry.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sources =>
    ensures
        match sources {
            Some(sources) => {
                &&& sources.len() == counts.len() * block
                &&& forall|b: int| 0 <= b < counts.len() ==> #[trigger] counts@[b] <= block
                &&& forall|b: int, i: int| 0 <= b < counts.len() && 0 <= i < counts@[b]
                    ==> #[trigger] row_column(columns@, counts@, b, i) < sequence
                        && prefix_count(counts@, b) + i < columns.len()
                &&& forall|b: int, i: int| 0 <= b < counts.len() && 0 <= i < block
                    ==> #[trigger] sources@[flat(b, i, block as int)] == flat(
                        b,
                        if i < counts@[b] { row_column(columns@, counts@, b, i) } else { 0 },
                        sequence as int,
                    )
                &&& forall|j: int| 0 <= j < sources.len()
                    ==> #[trigger] sources[j] < counts.len() * sequence
            },
            None => {
                ||| counts.len() * sequence > usize::MAX
                ||| counts.len() * block > usize::MAX
                ||| sequence == 0 && block > 0 && counts.len() > 0
                ||| exists|b: int| 0 <= b < counts.len() && #[trigger] counts@[b] > block
                ||| exists|b: int, i: int| 0 <= b < counts.len() && 0 <= i < counts@[b] && (
                    prefix_count(counts@, b) + i >= columns.len()
                    || #[trigger] row_column(columns@, counts@, b, i) >= sequence)
            },
        },
))]
pub(super) fn compaction_sources(
    counts: &[usize],
    columns: &[usize],
    sequence: usize,
    block: usize,
) -> Option<Vec<usize>> {
    let batch = counts.len();
    let (limit, total) = match (batch.checked_mul(sequence), batch.checked_mul(block)) {
        (Some(limit), Some(total)) => (limit, total),
        _ => return None,
    };
    let mut sources: Vec<usize> = Vec::with_capacity(total);
    let mut offset: usize = 0;
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= batch,
            batch == counts.len(),
            limit == batch * sequence,
            total == batch * block,
            offset == prefix_count(counts@, row as int),
            sources.len() == row * block,
            forall|b: int| 0 <= b < row ==> #[trigger] counts@[b] <= block,
            forall|b: int, i: int| 0 <= b < row && 0 <= i < counts@[b]
                ==> #[trigger] row_column(columns@, counts@, b, i) < sequence
                    && prefix_count(counts@, b) + i < columns.len(),
            forall|b: int, i: int| 0 <= b < row && 0 <= i < block
                ==> #[trigger] sources@[flat(b, i, block as int)] == flat(
                    b,
                    if i < counts@[b] { row_column(columns@, counts@, b, i) } else { 0 },
                    sequence as int,
                ),
            forall|j: int| 0 <= j < sources.len() ==> #[trigger] sources[j] < limit,
        decreases batch - row,
    ))]
    while row < batch {
        let count = counts[row];
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_row_block(row as int, batch as int, sequence as int);
            lemma_rows_below(row as int, block as int);
            lemma_prefix_step(counts@, row as int);
        }
        let base = match row.checked_mul(sequence) {
            Some(base) if count <= block => base,
            _ => return None,
        };
        let mut slot: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                slot <= block,
                row < batch,
                batch == counts.len(),
                limit == batch * sequence,
                count == counts@[row as int],
                count <= block,
                base == row * sequence,
                row * sequence + sequence <= batch * sequence,
                offset == prefix_count(counts@, row as int),
                sources.len() == row * block + slot,
                rows_below(row as int, block as int),
                forall|b: int, i: int| 0 <= b < row && 0 <= i < block
                    ==> #[trigger] sources@[flat(b, i, block as int)] == flat(
                        b,
                        if i < counts@[b] { row_column(columns@, counts@, b, i) } else { 0 },
                        sequence as int,
                    ),
                forall|i: int| 0 <= i < slot && i < count
                    ==> #[trigger] row_column(columns@, counts@, row as int, i) < sequence
                        && prefix_count(counts@, row as int) + i < columns.len(),
                forall|i: int| 0 <= i < slot
                    ==> #[trigger] sources@[flat(row as int, i, block as int)] == flat(
                        row as int,
                        if i < count { row_column(columns@, counts@, row as int, i) } else { 0 },
                        sequence as int,
                    ),
                forall|j: int| 0 <= j < sources.len() ==> #[trigger] sources[j] < limit,
            decreases block - slot,
        ))]
        while slot < block {
            // Slot `(row, slot)` witnesses every refusal inside this loop.
            #[cfg(verus_keep_ghost)]
            proof! {
                let ghost witness = row_column(columns@, counts@, row as int, slot as int);
                assert(witness == row_column(columns@, counts@, row as int, slot as int));
            }
            let column = if slot < count {
                let index = match offset.checked_add(slot) {
                    Some(index) if index < columns.len() => index,
                    _ => return None,
                };
                columns[index]
            } else {
                0
            };
            let source = match base.checked_add(column) {
                Some(source) if column < sequence && source < limit => source,
                _ => return None,
            };
            sources.push(source);
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(row * block + block == (row + 1) * block) by (nonlinear_arith);
            // The last valid column was read, so the next offset fits in `columns`.
            if count > 0 {
                assert(row_column(columns@, counts@, row as int, count - 1) < sequence);
                assert(prefix_count(counts@, row as int) + (count - 1) < columns.len());
            }
            assert forall|b: int, i: int| 0 <= b < row + 1 && 0 <= i < block
                implies #[trigger] sources@[flat(b, i, block as int)] == flat(
                    b,
                    if i < counts@[b] { row_column(columns@, counts@, b, i) } else { 0 },
                    sequence as int,
                ) by {}
            assert forall|b: int, i: int| 0 <= b < row + 1 && 0 <= i < counts@[b]
                implies #[trigger] row_column(columns@, counts@, b, i) < sequence
                    && prefix_count(counts@, b) + i < columns.len() by {}
        }
        offset = offset.checked_add(count)?;
        row += 1;
    }
    Some(sources)
}

/// Lists the flat scatter source of every input column.
///
/// Column `(b, s)` reads block slot `b * block + k` when `s` is row `b`'s
/// `k`-th valid column and `b * block` otherwise. Returns `None` exactly for
/// a count above `block`, too few columns, a row whose columns are not
/// strictly increasing below `sequence`, or an unrepresentable extent; the
/// proof establishes every returned entry. With `block = 0` every row is
/// inactive and every source is `0` (callers zero the whole output).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sources =>
    ensures
        match sources {
            Some(sources) => {
                &&& sources.len() == counts.len() * sequence
                &&& forall|b: int| 0 <= b < counts.len() ==> {
                    &&& #[trigger] counts@[b] <= block
                    &&& prefix_count(counts@, b) + counts@[b] <= columns.len()
                    &&& row_sorted_below(columns@, counts@, b, sequence as int)
                }
                &&& forall|b: int, k: int| 0 <= b < counts.len() && 0 <= k < counts@[b]
                    ==> #[trigger] sources@[flat(b, row_column(columns@, counts@, b, k), sequence as int)]
                        == flat(b, k, block as int)
                &&& forall|b: int, s: int| 0 <= b < counts.len() && 0 <= s < sequence
                    && !in_row(columns@, counts@, b, s)
                    ==> #[trigger] sources@[flat(b, s, sequence as int)] == flat(b, 0, block as int)
                &&& block > 0 ==> forall|j: int| 0 <= j < sources.len()
                    ==> #[trigger] sources[j] < counts.len() * block
            },
            None => {
                ||| counts.len() * block > usize::MAX
                ||| counts.len() * sequence > usize::MAX
                ||| exists|b: int| 0 <= b < counts.len() && {
                    ||| #[trigger] counts@[b] > block
                    ||| prefix_count(counts@, b) + counts@[b] > columns.len()
                    ||| !row_sorted_below(columns@, counts@, b, sequence as int)
                }
            },
        },
))]
pub(super) fn scatter_sources(
    counts: &[usize],
    columns: &[usize],
    sequence: usize,
    block: usize,
) -> Option<Vec<usize>> {
    let batch = counts.len();
    let (limit, total) = match (batch.checked_mul(block), batch.checked_mul(sequence)) {
        (Some(limit), Some(total)) => (limit, total),
        _ => return None,
    };
    let mut sources: Vec<usize> = Vec::with_capacity(total);
    let mut offset: usize = 0;
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= batch,
            batch == counts.len(),
            limit == batch * block,
            total == batch * sequence,
            offset == prefix_count(counts@, row as int),
            sources.len() == row * sequence,
            forall|b: int| 0 <= b < row ==> {
                &&& #[trigger] counts@[b] <= block
                &&& prefix_count(counts@, b) + counts@[b] <= columns.len()
                &&& row_sorted_below(columns@, counts@, b, sequence as int)
            },
            forall|b: int, k: int| 0 <= b < row && 0 <= k < counts@[b]
                ==> #[trigger] sources@[flat(b, row_column(columns@, counts@, b, k), sequence as int)]
                    == flat(b, k, block as int),
            forall|b: int, s: int| 0 <= b < row && 0 <= s < sequence
                && !in_row(columns@, counts@, b, s)
                ==> #[trigger] sources@[flat(b, s, sequence as int)] == flat(b, 0, block as int),
            block > 0 ==> forall|j: int| 0 <= j < sources.len() ==> #[trigger] sources[j] < limit,
        decreases batch - row,
    ))]
    while row < batch {
        let count = counts[row];
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_row_block(row as int, batch as int, block as int);
            lemma_rows_below(row as int, sequence as int);
            lemma_prefix_step(counts@, row as int);
            assert forall|b: int, k: int| 0 <= b < row && 0 <= k < counts@[b]
                implies 0 <= #[trigger] row_column(columns@, counts@, b, k) < sequence by {
                assert(row_sorted_below(columns@, counts@, b, sequence as int));
            }
        }
        let base = match row.checked_mul(block) {
            Some(base) if count <= block && (block == 0 || base < limit) => base,
            _ => return None,
        };
        // `rank` walks the row's sorted columns while `column` scans 0..sequence.
        let mut rank: usize = 0;
        let mut column: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                column <= sequence,
                rank <= count,
                row < batch,
                batch == counts.len(),
                count == counts@[row as int],
                count <= block,
                limit == batch * block,
                base == row * block,
                row * block + block <= batch * block,
                block == 0 || base < limit,
                offset == prefix_count(counts@, row as int),
                sources.len() == row * sequence + column,
                rows_below(row as int, sequence as int),
                forall|b: int, k: int| 0 <= b < row && 0 <= k < counts@[b]
                    ==> 0 <= #[trigger] row_column(columns@, counts@, b, k) < sequence,
                forall|b: int, k: int| 0 <= b < row && 0 <= k < counts@[b]
                    ==> #[trigger] sources@[flat(b, row_column(columns@, counts@, b, k), sequence as int)]
                        == flat(b, k, block as int),
                forall|b: int, s: int| 0 <= b < row && 0 <= s < sequence
                    && !in_row(columns@, counts@, b, s)
                    ==> #[trigger] sources@[flat(b, s, sequence as int)] == flat(b, 0, block as int),
                // Matched columns were read, increase, and lie before the cursor.
                forall|k: int| 0 <= k < rank
                    ==> #[trigger] row_column(columns@, counts@, row as int, k) < column
                        && offset + k < columns.len(),
                forall|i: int, j: int| 0 <= i < j < rank
                    ==> row_column(columns@, counts@, row as int, i)
                        < row_column(columns@, counts@, row as int, j),
                // The next column to match is ahead of the cursor or already out of order.
                rank < count && offset + rank < columns.len() ==> (
                    row_column(columns@, counts@, row as int, rank as int) >= column
                    || (rank > 0 && row_column(columns@, counts@, row as int, rank as int)
                        <= row_column(columns@, counts@, row as int, rank - 1))),
                forall|k: int| 0 <= k < rank
                    ==> #[trigger] sources@[flat(row as int, row_column(columns@, counts@, row as int, k), sequence as int)]
                        == flat(row as int, k, block as int),
                forall|s: int| 0 <= s < column && !in_row(columns@, counts@, row as int, s)
                    ==> #[trigger] sources@[flat(row as int, s, sequence as int)] == base,
                block > 0 ==> forall|j: int| 0 <= j < sources.len() ==> #[trigger] sources[j] < limit,
            decreases sequence - column,
        ))]
        while column < sequence {
            let mut source = base;
            if rank < count {
                let index = match offset.checked_add(rank) {
                    Some(index) if index < columns.len() => index,
                    _ => return None,
                };
                if columns[index] == column {
                    source = match base.checked_add(rank) {
                        Some(source) if source < limit => source,
                        _ => return None,
                    };
                    #[cfg(verus_keep_ghost)]
                    proof! {
                        assert(in_row(columns@, counts@, row as int, column as int)) by {
                            assert(row_column(columns@, counts@, row as int, rank as int) == column);
                        }
                    }
                    rank += 1;
                }
            }
            sources.push(source);
            column += 1;
        }
        if rank != count {
            // Some valid column was unsorted, duplicated, or out of range.
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(!row_sorted_below(columns@, counts@, row as int, sequence as int)
                    || prefix_count(counts@, row as int) + counts@[row as int] > columns.len());
            }
            return None;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(row * sequence + sequence == (row + 1) * sequence) by (nonlinear_arith);
            assert(row_sorted_below(columns@, counts@, row as int, sequence as int));
            if count > 0 {
                assert(offset + (count - 1) < columns.len());
            }
            assert forall|s: int| 0 <= s < sequence && !in_row(columns@, counts@, row as int, s)
                implies #[trigger] sources@[flat(row as int, s, sequence as int)]
                    == flat(row as int, 0, block as int) by {}
        }
        offset = offset.checked_add(count)?;
        row += 1;
    }
    Some(sources)
}

/// Query positions of one row's block slots: `(visible, query)`.
///
/// Row `b` has committed length `L = length` and `n = count` new tokens in a
/// block of `Q = block` slots. Slot `i` is the row's `i`-th new token when
/// `i < n` and padding otherwise. Two positions are planned per slot:
///
/// - `query[i] = L + i`, the absolute position whose relative bias the slot
///   uses (padding slots continue the sequence; their outputs are discarded);
/// - `visible[i]`, the last key position the slot may attend to under the
///   causal rule `key <= visible[i]`: `L + i` for a new token, the row's last
///   new position `L + n - 1` for a padding slot (so no padded query row is
///   ever empty), and `L - 1` (only the committed history) when the row is
///   inactive (`n = 0`).
///
/// Consequently no slot of an active row sees a key beyond `L + n - 1`, the
/// row's own appended history. Returns `None` when `count > block` or a
/// position would not fit `i64` or `usize` (`length + block` overflows).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(positions =>
    ensures
        match positions {
            Some((visible, query)) => {
                &&& count <= block
                &&& visible.len() == block
                &&& query.len() == block
                &&& forall|i: int| 0 <= i < block ==> #[trigger] query[i] == length + i
                &&& forall|i: int| 0 <= i < block ==> #[trigger] visible[i] == (
                    if count == 0 {
                        length - 1
                    } else if i < count {
                        length + i
                    } else {
                        length + count - 1
                    })
                &&& count > 0 ==> forall|i: int| 0 <= i < block
                    ==> #[trigger] visible[i] <= length + count - 1
            },
            None => count > block || length + block > usize::MAX || length + block > i64::MAX,
        },
))]
pub(super) fn row_positions(
    length: usize,
    count: usize,
    block: usize,
) -> Option<(Vec<i64>, Vec<i64>)> {
    if count > block {
        return None;
    }
    let end = length.checked_add(block)?;
    // `usize` has at most 64 bits, so widening to `u64` is lossless.
    if end as u64 > i64::MAX as u64 {
        return None;
    }
    let start = length as i64;
    let mut visible: Vec<i64> = Vec::with_capacity(block);
    let mut query: Vec<i64> = Vec::with_capacity(block);
    let mut slot: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            slot <= block,
            count <= block,
            length + block <= i64::MAX,
            start == length,
            visible.len() == slot,
            query.len() == slot,
            forall|i: int| 0 <= i < slot ==> #[trigger] query[i] == length + i,
            forall|i: int| 0 <= i < slot ==> #[trigger] visible[i] == (
                if count == 0 {
                    length - 1
                } else if i < count {
                    length + i
                } else {
                    length + count - 1
                }),
        decreases block - slot,
    ))]
    while slot < block {
        let offset = slot as i64;
        query.push(start + offset);
        let shown = if count == 0 {
            start - 1
        } else if slot < count {
            start + offset
        } else {
            start + count as i64 - 1
        };
        visible.push(shown);
        slot += 1;
    }
    Some((visible, query))
}

/// Whether `indices` keeps every row inside its group of `group_size` consecutive rows.
///
/// True exactly when `group_size > 0`, `group_size` divides the batch
/// `indices.len()`, and `indices[d] / group_size == d / group_size` for every
/// row `d`. Beam search reorders only within a source's beams, which share
/// one encoder binding, so such a reorder can leave that binding in place.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(preserved =>
    ensures
        preserved == (group_size > 0
            && indices.len() % group_size == 0
            && forall|d: int| 0 <= d < indices.len()
                ==> #[trigger] indices[d] / group_size == d / group_size as int),
))]
pub(super) fn groups_preserved(indices: &[usize], group_size: usize) -> bool {
    if group_size == 0 || !indices.len().is_multiple_of(group_size) {
        return false;
    }
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= indices.len(),
            group_size > 0,
            forall|d: int| 0 <= d < row ==> #[trigger] indices[d] / group_size == d / group_size as int,
        decreases indices.len() - row,
    ))]
    while row < indices.len() {
        if indices[row] / group_size != row / group_size {
            return false;
        }
        row += 1;
    }
    true
}

#[cfg(test)]
#[path = "packing_tests.rs"]
mod tests;
