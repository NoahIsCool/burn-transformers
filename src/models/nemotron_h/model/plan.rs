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
//! admissible inputs the proofs establish freedom from overflow and that every
//! compact-batch map has its stated length and addresses only valid entries
//! of its source (a flat index below `B * S`, `B * N`, or the appended zero
//! row), so tensor programs built from these plans cannot index out of range.
//! They also prove the per-slot query-position rule (a query never lies
//! beyond its row's last readable key), the policies' exact decision rules,
//! and the MTP cursor invariant. Tensor programs driven by the plans remain
//! covered by Rust tests.

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
/// Returns `None` when a row's columns are not strictly increasing below
/// `sequence`, or when a flat index would overflow `usize`. On success every
/// map has its documented length and stays inside its source: `gather` below
/// `B * S`, `scatter` and `last` at most `B * N` (the zero row), `packed`
/// below `B * N`, and `unpack` at most `T` (the zero row).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(planned =>
    ensures
        match planned {
            Some(plan) => {
                let batch = columns.len() as int;
                &&& plan.sequence == sequence
                &&& plan.counts.len() == batch
                &&& forall|b: int| 0 <= b < batch ==> {
                    &&& #[trigger] plan.counts@[b] == columns@[b]@.len()
                    &&& plan.counts@[b] <= plan.width
                    &&& sorted_unique_below(columns@[b]@, sequence as int)
                }
                &&& plan.gather.len() == batch * plan.width
                &&& plan.scatter.len() == batch * sequence
                &&& plan.unpack.len() == batch * plan.width
                &&& plan.packed.len() == total_tokens(columns@, batch)
                &&& plan.last.len() == batch
                &&& all_below(plan.gather@, batch * sequence)
                &&& all_at_most(plan.scatter@, batch * plan.width)
                &&& all_below(plan.packed@, batch * plan.width)
                &&& all_at_most(plan.unpack@, plan.packed.len() as int)
                &&& all_at_most(plan.last@, batch * plan.width)
            },
            None => true,
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
        }
        if count > width {
            width = count;
        }
        tokens = tokens.checked_add(count)?;
        row += 1;
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
                total_tokens(columns@, row as int + 1) <= tokens,
                total_tokens(columns@, row as int + 1)
                    == total_tokens(columns@, row as int) + count,
                slot <= width,
                gather.len() == row * width + slot,
                unpack.len() == row * width + slot,
                packed.len() == total_tokens(columns@, row as int)
                    + if slot < count { slot as int } else { count as int },
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
                compact_start + width <= batch * width,
                compact_slots == batch * width,
                column <= sequence,
                next <= count,
                scatter.len() == row * sequence + column,
                all_at_most(scatter@, batch * width),
            decreases sequence - column,
        ))]
        while column < sequence {
            if next < count && valid[next] == column {
                scatter.push(compact_start + next);
                next += 1;
            } else {
                scatter.push(compact_slots);
            }
            column += 1;
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
        }
        compact_start += width;
        rectangular_start += sequence;
        row += 1;
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
/// sliding window. Returns `None` when `baselines` and `counts` differ in
/// length or a position exceeds `usize` or `i64`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(positions =>
    ensures
        match positions {
            Some(positions) => {
                &&& counts.len() == baselines.len()
                &&& positions.len() == baselines.len() * width
                &&& forall|i: int| 0 <= i < positions.len() ==> #[trigger] positions@[i] >= 0
            },
            None => true,
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
            forall|i: int| 0 <= i < positions.len() ==> #[trigger] positions@[i] >= 0,
        decreases baselines.len() - row,
    ))]
    while row < baselines.len() {
        let mut slot: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                counts.len() == baselines.len(),
                row < baselines.len(),
                slot <= width,
                positions.len() == row * width + slot,
                forall|i: int| 0 <= i < positions.len() ==> #[trigger] positions@[i] >= 0,
            decreases width - slot,
        ))]
        while slot < width {
            let position = slot_position(baselines[row], counts[row], slot)?;
            if position > i64::MAX as usize {
                return None;
            }
            positions.push(position as i64);
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row as int + 1) * width as int == row as int * width as int + width as int)
                by (nonlinear_arith);
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
