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
//! # Execution policies
//!
//! [`mamba_step`] chooses the Mamba-2 path of one row exactly as the pinned
//! reference does: the single-step update only for one token appended to an
//! existing state, the chunk scan otherwise. [`transactional_mamba`] decides
//! how a speculative step records recurrent state: one terminal state when
//! every row appends at most one token, per-token snapshots otherwise (which
//! requires existing state in every active row). [`mtp_length_after_truncate`]
//! keeps the shifted MTP cursor at most one position behind the target.
//!
//! Every function is pure integer logic with checked arithmetic, so tensor
//! programs built from these plans cannot index out of range.

/// Right-padded compact layout of one ragged batch (see the module documentation).
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
/// `sequence`, or when a flat index would overflow `usize`.
pub(crate) fn plan_compaction(columns: &[Vec<usize>], sequence: usize) -> Option<CompactPlan> {
    let batch = columns.len();
    let mut width = 0;
    let mut tokens: usize = 0;
    let mut row = 0;
    while row < batch {
        if !strictly_increasing_below(&columns[row], sequence) {
            return None;
        }
        width = width.max(columns[row].len());
        tokens = tokens.checked_add(columns[row].len())?;
        row += 1;
    }
    let compact_slots = batch.checked_mul(width)?;
    let rectangular_slots = batch.checked_mul(sequence)?;
    let mut counts = Vec::with_capacity(batch);
    let mut gather = Vec::with_capacity(compact_slots);
    let mut scatter = Vec::with_capacity(rectangular_slots);
    let mut packed = Vec::with_capacity(tokens);
    let mut unpack = Vec::with_capacity(compact_slots);
    let mut last = Vec::with_capacity(batch);
    row = 0;
    while row < batch {
        let valid = &columns[row];
        let count = valid.len();
        // `row < batch`, so every product below is at most `batch * width`
        // or `batch * sequence`, both already proven representable.
        let compact_start = row * width;
        let rectangular_start = row * sequence;
        counts.push(count);
        let mut slot = 0;
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
        let mut next = 0;
        let mut column = 0;
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

/// Returns whether `values` is strictly increasing with every entry below `bound`.
fn strictly_increasing_below(values: &[usize], bound: usize) -> bool {
    let mut index = 0;
    while index < values.len() {
        if values[index] >= bound || (index > 0 && values[index - 1] >= values[index]) {
            return false;
        }
        index += 1;
    }
    true
}

/// Absolute positions `W_b + j` of every compact slot `(b, j)`, `[B * N]`.
///
/// Padded slots continue the row's numbering, so a padded query sees every
/// valid key of its row and attention rows are never fully excluded for an
/// active row. Returns `None` when a position exceeds `usize` or `i64`.
pub(crate) fn query_positions(baselines: &[usize], width: usize) -> Option<Vec<i64>> {
    let mut positions = Vec::with_capacity(baselines.len().checked_mul(width)?);
    for &baseline in baselines {
        let mut slot = 0;
        while slot < width {
            positions.push(i64::try_from(baseline.checked_add(slot)?).ok()?);
            slot += 1;
        }
    }
    Some(positions)
}

/// Mamba-2 execution path of one stateful row.
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
pub(crate) const fn mamba_step(tokens: usize, baseline: usize) -> MambaStep {
    if tokens == 1 && baseline > 0 {
        MambaStep::Recurrent
    } else {
        MambaStep::Chunked
    }
}

/// How a transactional step records Mamba-2 state.
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
pub(crate) fn transactional_mamba(
    counts: &[usize],
    baselines: &[usize],
) -> Result<TransactionalMamba, usize> {
    let mut row = 0;
    let mut multi_token = false;
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
pub(crate) const fn mtp_length_after_truncate(target: usize, mtp: usize) -> usize {
    let limit = target.saturating_sub(1);
    if mtp < limit { mtp } else { limit }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
