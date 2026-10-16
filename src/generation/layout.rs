//! Verified host-side layouts of generation metadata.
//!
//! The coordinators turn small host vectors into backend tensors and back:
//! row-major `[batch, sequence]` validity masks are scanned for per-row counts,
//! last positions, and per-chunk activity; sparse scatters (logit bias, hard
//! constraints, history counts) are laid out as padded `[batch, width]` index
//! rectangles; and one packed readback is split into token IDs and scores. A
//! misplaced element in any of these silently corrupts another row, so the
//! layouts live here as pure integer functions, free of Burn, floating-point
//! arithmetic, and crate errors, and the Burn-facing callers stay thin.
//!
//! # Verified properties
//!
//! Every function below is checked in place by Verus through
//! `verus/generation.rs` (see `verus/README.md`). For all admissible inputs
//! the proofs establish that indexing stays in bounds, arithmetic cannot
//! overflow, and:
//!
//! - mask scans return exactly the number of valid entries of a span, the
//!   last valid index, or whether any entry is valid; per-row scans visit row
//!   `r`'s slots `r * sequence..(r + 1) * sequence` and report the first row
//!   without a valid token;
//! - a prefill chunk's row activity is exactly "some slot of this row inside
//!   the chunk is valid", and the chunk is skipped, prefilled, or decoded
//!   exactly when no row, every row, or only some rows are active;
//! - scatter layouts put every row's entries in that row's segment, in order,
//!   padded with no-op `(0, 0)` entries to the common width; repeated rows are
//!   exact copies; history increments are one exactly for active or present
//!   tokens (and generated ones for the generated view);
//! - a decode step adds one to the generated length of exactly the active
//!   rows;
//! - a packed `[batch, 2]` readback splits into its even and odd entries;
//! - special-token validation reports the first configured token outside the
//!   vocabulary, so accepted tokens are valid scatter indices;
//! - a history value becomes a token ID exactly when it is a non-negative
//!   `u32` below the vocabulary size.
//!
//! The proofs do not cover the tensor uploads and scatters that consume these
//! layouts; the Rust test suite covers those.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
use super::bookkeeping::count_matching;
use super::bookkeeping::row_span;

#[cfg(verus_keep_ghost)]
verus! {

/// Some entry of `flags[start..end]` is `true`.
pub open spec fn any_true(flags: Seq<bool>, start: int, end: int) -> bool {
    exists|index: int| start <= index < end && #[trigger] flags[index]
}

/// Slot `k` of a padded hard-constraint row: suppressed EOS IDs, then banned
/// tokens, then no-op padding at token zero.
pub open spec fn constraint_slot(eos: Seq<u32>, suppress: bool, banned: Seq<u32>, k: int) -> i64 {
    let leading = if suppress { eos.len() as int } else { 0 };
    if k < leading {
        eos[k] as i64
    } else if k < leading + banned.len() {
        banned[k - leading] as i64
    } else {
        0
    }
}

} // verus!

// ---------------------------------------------------------------------------
// Mask scans
// ---------------------------------------------------------------------------

/// Counts the `true` entries of `valid[start..end]`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(count =>
    requires
        start <= end,
        end <= valid.len(),
    ensures
        count == count_matching(valid@, true, start as int, end as int),
        count <= end - start,
))]
pub(crate) fn count_valid(valid: &[bool], start: usize, end: usize) -> usize {
    let mut count = 0;
    let mut index = start;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            start <= index <= end,
            end <= valid.len(),
            count == count_matching(valid@, true, start as int, index as int),
            count <= index - start,
        decreases end - index,
    ))]
    while index < end {
        if valid[index] {
            count += 1;
        }
        index += 1;
    }
    count
}

/// Returns the last index of `valid[start..end]` holding `true`.
///
/// `None` means every entry of the span is `false`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        start <= end,
        end <= valid.len(),
    ensures
        match found {
            Some(index) => {
                &&& start <= index < end
                &&& valid@[index as int]
                &&& forall|later: int| index < later < end ==> !#[trigger] valid@[later]
            },
            None => forall|index: int| start <= index < end ==> !#[trigger] valid@[index],
        },
))]
pub(crate) fn last_valid(valid: &[bool], start: usize, end: usize) -> Option<usize> {
    let mut index = end;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            start <= index <= end,
            end <= valid.len(),
            forall|later: int| index <= later < end ==> !#[trigger] valid@[later],
        decreases index - start,
    ))]
    while index > start {
        index -= 1;
        if valid[index] {
            return Some(index);
        }
    }
    None
}

/// Returns whether any entry of `valid[start..end]` is `true`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        start <= end,
        end <= valid.len(),
    ensures
        found == any_true(valid@, start as int, end as int),
))]
pub(crate) fn any_valid(valid: &[bool], start: usize, end: usize) -> bool {
    let mut index = start;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            start <= index <= end,
            end <= valid.len(),
            forall|earlier: int| start <= earlier < index ==> !#[trigger] valid@[earlier],
        decreases end - index,
    ))]
    while index < end {
        if valid[index] {
            return true;
        }
        index += 1;
    }
    false
}

/// Appends the valid-token count of every row of a `[batch, sequence]` mask.
///
/// The proof shows `lengths[r]` counts exactly row `r`'s slots
/// `r * sequence..(r + 1) * sequence`, every appended count is positive, and
/// `Err(row)` names the first row without a valid token (rows before it are
/// counted, nothing after it is).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(filled =>
    requires
        old(lengths).len() == 0,
        valid.len() == batch * sequence,
    ensures
        forall|r: int| 0 <= r < final(lengths).len() ==> {
            &&& #[trigger] final(lengths)@[r] == count_matching(
                valid@,
                true,
                r * sequence,
                r * sequence + sequence,
            )
            &&& final(lengths)@[r] > 0
        },
        match filled {
            Ok(()) => final(lengths).len() == batch,
            Err(row) => {
                &&& row < batch
                &&& final(lengths).len() == row
                &&& count_matching(valid@, true, row * sequence, row * sequence + sequence) == 0
            },
        },
))]
pub(crate) fn fill_row_lengths(
    lengths: &mut Vec<usize>,
    valid: &[bool],
    batch: usize,
    sequence: usize,
) -> Result<(), usize> {
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= batch,
            valid.len() == batch * sequence,
            lengths.len() == row,
            forall|r: int| 0 <= r < lengths.len() ==> {
                &&& #[trigger] lengths@[r] == count_matching(
                    valid@,
                    true,
                    r * sequence,
                    r * sequence + sequence,
                )
                &&& lengths@[r] > 0
            },
        decreases batch - row,
    ))]
    while row < batch {
        let (start, end) = row_span(row, batch, sequence, 0, sequence);
        let length = count_valid(valid, start, end);
        if length == 0 {
            return Err(row);
        }
        lengths.push(length);
        row += 1;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Prefill chunks
// ---------------------------------------------------------------------------

/// Whether each row has a valid prompt token in columns `start..end`.
///
/// The proof shows `active[r]` holds exactly when some slot of row `r`
/// between columns `start` and `end` of the flattened `[batch, sequence]`
/// mask is valid.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(active =>
    requires
        valid.len() == batch * sequence,
        start <= end,
        end <= sequence,
    ensures
        active.len() == batch,
        forall|r: int| 0 <= r < batch ==> #[trigger] active@[r] == any_true(
            valid@,
            r * sequence + start,
            r * sequence + end,
        ),
))]
pub(crate) fn chunk_activity(
    valid: &[bool],
    batch: usize,
    sequence: usize,
    start: usize,
    end: usize,
) -> Vec<bool> {
    let mut active = Vec::with_capacity(batch);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= batch,
            valid.len() == batch * sequence,
            start <= end,
            end <= sequence,
            active.len() == row,
            forall|r: int| 0 <= r < row ==> #[trigger] active@[r] == any_true(
                valid@,
                r * sequence + start,
                r * sequence + end,
            ),
        decreases batch - row,
    ))]
    while row < batch {
        let (row_start, row_end) = row_span(row, batch, sequence, start, end);
        active.push(any_valid(valid, row_start, row_end));
        row += 1;
    }
    active
}

/// How a prefill chunk reaches the model.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChunkCall {
    /// No row has a valid token: the chunk is a no-op and is skipped.
    Skip,
    /// Every row is active: the architecture's dense `prefill` runs.
    Prefill,
    /// Only some rows are active: `decode` treats the others as no-ops.
    Decode,
}

/// Chooses how a prefill chunk with per-row `active` flags is evaluated.
///
/// The proof shows `Skip` exactly when no row is active, `Prefill` exactly
/// when at least one row is active and every row is, and `Decode` otherwise.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(call =>
    ensures
        call == ChunkCall::Skip <==> !any_true(active@, 0, active.len() as int),
        call == ChunkCall::Prefill <==> {
            &&& any_true(active@, 0, active.len() as int)
            &&& forall|r: int| 0 <= r < active.len() ==> #[trigger] active@[r]
        },
        call == ChunkCall::Decode <==> {
            &&& any_true(active@, 0, active.len() as int)
            &&& exists|r: int| 0 <= r < active.len() && !#[trigger] active@[r]
        },
))]
pub(crate) fn chunk_call(active: &[bool]) -> ChunkCall {
    let mut any = false;
    let mut all = true;
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= active.len(),
            any == any_true(active@, 0, row as int),
            all == forall|r: int| 0 <= r < row ==> #[trigger] active@[r],
        decreases active.len() - row,
    ))]
    while row < active.len() {
        if active[row] {
            any = true;
        } else {
            all = false;
        }
        row += 1;
    }
    if !any {
        ChunkCall::Skip
    } else if all {
        ChunkCall::Prefill
    } else {
        ChunkCall::Decode
    }
}

// ---------------------------------------------------------------------------
// Sparse scatter layouts
// ---------------------------------------------------------------------------

/// Repeats `row` for `batch` rows of a row-major `[batch, row.len()]` buffer.
///
/// The proof shows element `i` is `row[i mod row.len()]`, so row `b` is an
/// exact copy of `row`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(repeated =>
    requires
        row.len() > 0,
        batch * row.len() <= usize::MAX,
    ensures
        repeated.len() == batch * row.len(),
        forall|i: int| 0 <= i < repeated.len() ==> #[trigger] repeated@[i] == row@[i % (row.len() as int)],
))]
pub(crate) fn repeat_row<T: Copy>(row: &[T], batch: usize) -> Vec<T> {
    let width = row.len();
    let mut repeated = Vec::with_capacity(batch * width);
    let mut copy = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            width == row.len(),
            width > 0,
            copy <= batch,
            batch * width <= usize::MAX,
            repeated.len() == copy * width,
            forall|i: int| 0 <= i < repeated.len() ==> #[trigger] repeated@[i] == row@[i % (width as int)],
        decreases batch - copy,
    ))]
    while copy < batch {
        let mut column = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                width == row.len(),
                width > 0,
                copy < batch,
                column <= width,
                repeated.len() == copy * width + column,
                forall|i: int| 0 <= i < repeated.len() ==> #[trigger] repeated@[i] == row@[i % (width as int)],
            decreases width - column,
        ))]
        while column < width {
            #[cfg(verus_keep_ghost)]
            proof! {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
                    (copy * width + column) as int,
                    width as int,
                    copy as int,
                    column as int,
                );
            }
            repeated.push(row[column]);
            column += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((copy + 1) * width == copy * width + width) by (nonlinear_arith);
        }
        copy += 1;
    }
    repeated
}

/// Appends one `width`-wide row of hard-constraint scatter indices.
///
/// The row holds the EOS IDs (when `suppress`), then the banned tokens, then
/// no-op padding at token zero. Returns the number of real (non-padding)
/// entries; the caller pairs exactly those with `-inf`. The proof shows the
/// buffer grows by exactly `width`, earlier rows are untouched, and slot `k`
/// of the new row is `constraint_slot(eos, suppress, banned, k)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(entries =>
    requires
        (if suppress { eos.len() as int } else { 0 }) + banned.len() <= width,
    ensures
        entries == (if suppress { eos.len() as int } else { 0 }) + banned.len(),
        final(indices).len() == old(indices).len() + width,
        forall|i: int| 0 <= i < old(indices).len() ==> #[trigger] final(indices)@[i] == old(indices)@[i],
        forall|i: int| old(indices).len() <= i < final(indices).len() ==> #[trigger] final(indices)@[i]
            == constraint_slot(eos@, suppress, banned@, i - old(indices).len()),
))]
pub(crate) fn push_constraint_row(
    indices: &mut Vec<i64>,
    eos: &[u32],
    suppress: bool,
    banned: &[u32],
    width: usize,
) -> usize {
    let leading = if suppress { eos.len() } else { 0 };
    let entries = leading + banned.len();
    let mut column = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            leading == if suppress { eos.len() } else { 0 },
            entries == leading + banned.len(),
            entries <= width,
            column <= width,
            indices.len() == old(indices).len() + column,
            forall|i: int| 0 <= i < old(indices).len() ==> #[trigger] indices@[i] == old(indices)@[i],
            forall|i: int| old(indices).len() <= i < indices.len() ==> #[trigger] indices@[i]
                == constraint_slot(eos@, suppress, banned@, i - old(indices).len()),
        decreases width - column,
    ))]
    while column < width {
        let token = if column < leading {
            eos[column] as i64
        } else if column < entries {
            banned[column - leading] as i64
        } else {
            0
        };
        indices.push(token);
        column += 1;
    }
    entries
}

/// Appends one `width`-wide row of the history-count scatter for `tokens`.
///
/// Column `c` carries `tokens[c]` with observed increment one and generated
/// increment one exactly when `c >= prompt_length` (a generated token); past
/// the row's end it carries the no-op `(0, 0, 0)`. The proof shows all three
/// buffers grow by exactly `width`, earlier rows are untouched, and every new
/// slot holds exactly those values.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(
    requires
        old(ids).len() == old(observed).len(),
        old(observed).len() == old(generated).len(),
    ensures
        final(ids).len() == old(ids).len() + width,
        final(observed).len() == old(ids).len() + width,
        final(generated).len() == old(ids).len() + width,
        forall|i: int| 0 <= i < old(ids).len() ==> #[trigger] final(ids)@[i] == old(ids)@[i],
        forall|i: int| 0 <= i < old(ids).len() ==> #[trigger] final(observed)@[i] == old(observed)@[i],
        forall|i: int| 0 <= i < old(ids).len() ==> #[trigger] final(generated)@[i] == old(generated)@[i],
        forall|i: int| old(ids).len() <= i < final(ids).len() ==> #[trigger] final(ids)@[i] == if i
            - old(ids).len() < tokens.len() {
            tokens@[i - old(ids).len()] as i64
        } else {
            0
        },
        forall|i: int| old(ids).len() <= i < final(observed).len() ==> #[trigger] final(observed)@[i]
            == if i - old(ids).len() < tokens.len() {
            1i64
        } else {
            0i64
        },
        forall|i: int| old(ids).len() <= i < final(generated).len() ==> #[trigger] final(generated)@[i]
            == if prompt_length <= i - old(ids).len() < tokens.len() {
            1i64
        } else {
            0i64
        },
))]
pub(crate) fn push_history_row(
    ids: &mut Vec<i64>,
    observed: &mut Vec<i64>,
    generated: &mut Vec<i64>,
    tokens: &[u32],
    prompt_length: usize,
    width: usize,
) {
    let mut column = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            column <= width,
            ids.len() == old(ids).len() + column,
            observed.len() == old(ids).len() + column,
            generated.len() == old(ids).len() + column,
            forall|i: int| 0 <= i < old(ids).len() ==> #[trigger] ids@[i] == old(ids)@[i],
            forall|i: int| 0 <= i < old(ids).len() ==> #[trigger] observed@[i] == old(observed)@[i],
            forall|i: int| 0 <= i < old(ids).len() ==> #[trigger] generated@[i] == old(generated)@[i],
            forall|i: int| old(ids).len() <= i < ids.len() ==> #[trigger] ids@[i] == if i
                - old(ids).len() < tokens.len() {
                tokens@[i - old(ids).len()] as i64
            } else {
                0
            },
            forall|i: int| old(ids).len() <= i < observed.len() ==> #[trigger] observed@[i] == if i
                - old(ids).len() < tokens.len() {
                1i64
            } else {
                0i64
            },
            forall|i: int| old(ids).len() <= i < generated.len() ==> #[trigger] generated@[i] == if prompt_length
                <= i - old(ids).len() < tokens.len() {
                1i64
            } else {
                0i64
            },
        decreases width - column,
    ))]
    while column < width {
        if column < tokens.len() {
            ids.push(tokens[column] as i64);
            observed.push(1);
            generated.push(if column >= prompt_length { 1 } else { 0 });
        } else {
            ids.push(0);
            observed.push(0);
            generated.push(0);
        }
        column += 1;
    }
}

/// Scatter IDs and increments recording one decode step's tokens.
///
/// The proof shows row `i` carries `tokens[i]` with increment one when
/// `active[i]`, and the no-op `(0, 0)` otherwise.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(scatter =>
    requires
        tokens.len() == active.len(),
    ensures
        scatter.0.len() == tokens.len(),
        scatter.1.len() == tokens.len(),
        forall|i: int| 0 <= i < tokens.len() ==> #[trigger] scatter.0@[i] == if active@[i] {
            tokens@[i] as i64
        } else {
            0
        },
        forall|i: int| 0 <= i < tokens.len() ==> #[trigger] scatter.1@[i] == if active@[i] {
            1i64
        } else {
            0i64
        },
))]
pub(crate) fn record_scatter(tokens: &[u32], active: &[bool]) -> (Vec<i64>, Vec<i64>) {
    let mut ids = Vec::with_capacity(tokens.len());
    let mut increments = Vec::with_capacity(tokens.len());
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            tokens.len() == active.len(),
            row <= tokens.len(),
            ids.len() == row,
            increments.len() == row,
            forall|i: int| 0 <= i < row ==> #[trigger] ids@[i] == if active@[i] {
                tokens@[i] as i64
            } else {
                0
            },
            forall|i: int| 0 <= i < row ==> #[trigger] increments@[i] == if active@[i] {
                1i64
            } else {
                0i64
            },
        decreases tokens.len() - row,
    ))]
    while row < tokens.len() {
        if active[row] {
            ids.push(tokens[row] as i64);
            increments.push(1);
        } else {
            ids.push(0);
            increments.push(0);
        }
        row += 1;
    }
    (ids, increments)
}

/// Adds one to the generated length of every active row.
///
/// The proof shows each active row's length grows by exactly one and every
/// inactive row's length is unchanged. The precondition holds for every
/// reachable history: a length counts recorded steps, far below `usize::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(
    requires
        old(lengths).len() == active.len(),
        forall|i: int| 0 <= i < active.len() && #[trigger] active@[i] ==> old(lengths)@[i] < usize::MAX,
    ensures
        final(lengths).len() == active.len(),
        forall|i: int| 0 <= i < active.len() ==> #[trigger] final(lengths)@[i] == old(lengths)@[i]
            + if active@[i] { 1int } else { 0int },
))]
pub(crate) fn advance_generated_lengths(lengths: &mut [usize], active: &[bool]) {
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= active.len(),
            lengths.len() == active.len(),
            old(lengths).len() == active.len(),
            forall|i: int| 0 <= i < active.len() && #[trigger] active@[i] ==> old(lengths)@[i] < usize::MAX,
            forall|i: int| 0 <= i < row ==> #[trigger] lengths@[i] == old(lengths)@[i]
                + if active@[i] { 1int } else { 0int },
            forall|i: int| row <= i < active.len() ==> #[trigger] lengths@[i] == old(lengths)@[i],
        decreases active.len() - row,
    ))]
    while row < active.len() {
        if active[row] {
            lengths[row] += 1;
        }
        row += 1;
    }
}

// ---------------------------------------------------------------------------
// Readback and token IDs
// ---------------------------------------------------------------------------

/// Splits a packed buffer into its even-index and odd-index entries.
///
/// A `[batch, 2]` selection readback holds `(id, score)` pairs row by row, so
/// the proof's `evens[i] == values[2 i]` and `odds[i] == values[2 i + 1]` are
/// row `i`'s ID and score. An odd-length buffer leaves one more even entry.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(split =>
    ensures
        split.0.len() == (values.len() + 1) / 2,
        split.1.len() == values.len() / 2,
        forall|i: int| 0 <= i < split.0.len() ==> #[trigger] split.0@[i] == values@[2 * i],
        forall|i: int| 0 <= i < split.1.len() ==> #[trigger] split.1@[i] == values@[2 * i + 1],
))]
pub(crate) fn deinterleave<T: Copy>(values: &[T]) -> (Vec<T>, Vec<T>) {
    let mut evens = Vec::with_capacity(values.len() - values.len() / 2);
    let mut odds = Vec::with_capacity(values.len() / 2);
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= values.len(),
            evens.len() == (index + 1) / 2,
            odds.len() == index / 2,
            forall|i: int| 0 <= i < evens.len() ==> #[trigger] evens@[i] == values@[2 * i],
            forall|i: int| 0 <= i < odds.len() ==> #[trigger] odds@[i] == values@[2 * i + 1],
        decreases values.len() - index,
    ))]
    while index < values.len() {
        if index % 2 == 0 {
            evens.push(values[index]);
        } else {
            odds.push(values[index]);
        }
        index += 1;
    }
    (evens, odds)
}

/// Returns the first token of `tokens` outside a vocabulary of `vocabulary_size` IDs.
///
/// The proof shows `Some(token)` is `tokens[i]` for the smallest `i` with
/// `tokens[i] >= vocabulary_size`, and `None` means every token is a valid
/// scatter index into the vocabulary.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(outside =>
    ensures
        match outside {
            Some(token) => exists|i: int| {
                &&& 0 <= i < tokens.len()
                &&& #[trigger] tokens@[i] == token
                &&& token >= vocabulary_size
                &&& forall|j: int| 0 <= j < i ==> #[trigger] tokens@[j] < vocabulary_size
            },
            None => forall|i: int| 0 <= i < tokens.len() ==> #[trigger] tokens@[i] < vocabulary_size,
        },
))]
pub(crate) fn first_outside_vocabulary(tokens: &[u32], vocabulary_size: usize) -> Option<u32> {
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= tokens.len(),
            forall|j: int| 0 <= j < index ==> #[trigger] tokens@[j] < vocabulary_size,
        decreases tokens.len() - index,
    ))]
    while index < tokens.len() {
        if tokens[index] as usize >= vocabulary_size {
            return Some(tokens[index]);
        }
        index += 1;
    }
    None
}

/// Converts one history value to a token ID inside the vocabulary.
///
/// The proof shows `Some(value)` exactly when `0 <= value <= u32::MAX` and
/// `value < vocabulary_size`, and `None` otherwise.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(token =>
    ensures
        match token {
            Some(token) => {
                &&& 0 <= value <= u32::MAX
                &&& value < vocabulary_size
                &&& token == value
            },
            None => value < 0 || value > u32::MAX || value >= vocabulary_size,
        },
))]
pub(crate) fn history_token_id(value: i64, vocabulary_size: usize) -> Option<u32> {
    if value < 0 || value > u32::MAX as i64 {
        return None;
    }
    let token = value as u32;
    if (token as usize) < vocabulary_size {
        Some(token)
    } else {
        None
    }
}

#[cfg(test)]
#[path = "layout_tests.rs"]
mod tests;
