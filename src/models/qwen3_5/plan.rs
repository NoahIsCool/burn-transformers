//! Dependency-free planning for the Qwen families, verified in place by Verus.
//!
//! Every function here is integer and index logic with no Burn types, so
//! `verus/qwen.rs` checks the exact bodies that ship (see `verus/README.md`).
//!
//! # Layer schedule and cache slots
//!
//! * [`default_full_attention`] — the legacy hybrid schedule: layer `i` is
//!   full attention exactly when `(i + 1)` is a multiple of the interval;
//! * [`derived_full_attention`] — the whole derived schedule, which holds
//!   exactly `floor(L / interval)` full-attention layers;
//! * [`cache_slots`] — the dense slot of every layer inside its cache family:
//!   the number of earlier layers of the same family.
//!
//! # Compact right-padded batches
//!
//! A cached call packs row `b`'s valid caller columns `c_{b,0} < … <
//! c_{b,n_b-1}` into slots `0..n_b` of an `N = max_b n_b` wide block, slot
//! `i` sitting at cache position `W_b + i`:
//!
//! * [`columns_below`] and [`strictly_increasing`] decide the column
//!   validation that [`compact_row_plan`] requires;
//! * [`row_counts`], [`widest`], and [`prefix_rows`] give `n_b`, `N`, and
//!   whether every row is already the prefix `0..n_b` (no gather needed);
//! * [`compact_row_plan`] — the gather and restore indices of one row;
//! * [`slot_validity`] and [`slot_positions`] — the per-slot validity
//!   `i < n_b` and cache position `W_b + i`;
//! * [`all_equal`] — whether every row shares one working length;
//! * [`signed_index`] — the saturating `usize -> i64` index conversion.
//!
//! # MTP and speculative bookkeeping
//!
//! * [`shifted_mtp_length`] and [`shifted_mtp_lengths`] — the MTP cursor
//!   after a target truncation, which keeps the invariant `target = MTP + 1`;
//! * [`first_staging_overflow`] — the transactional staging preflight;
//! * [`append_end`] — the cache end of an MTP append and its capacity check;
//! * [`staging_requirement`] and [`speculative_cycle_plan`] — the staging
//!   capacity and cache-alignment preflights of a speculative cycle;
//! * [`call_rotary_context`] and [`first_token_at_or_above`] — the rotary
//!   context of a batched call and the host vocabulary check;
//! * [`gated_query_width`] — the gated full-attention query width.
//!
//! The proofs establish freedom from overflow and out-of-bounds indexing and
//! the stated postconditions for all admissible inputs. Tensor programs
//! driven by these plans remain covered by the Rust tests.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// `s` is strictly increasing and every entry lies below `bound`.
pub open spec fn increasing_below(s: Seq<usize>, bound: int) -> bool {
    &&& forall|i: int, j: int| 0 <= i < j < s.len() ==> s[i] < s[j]
    &&& forall|i: int| 0 <= i < s.len() ==> s[i] < bound
}

/// Number of entries of `kinds` equal to `kind`.
pub open spec fn count_kind(kinds: Seq<bool>, kind: bool) -> nat
    decreases kinds.len(),
{
    if kinds.len() == 0 {
        0
    } else {
        count_kind(kinds.drop_last(), kind) + if kinds.last() == kind { 1nat } else { 0nat }
    }
}

/// Extending a prefix by one entry adds one exactly when the entry matches.
pub proof fn lemma_count_extend(kinds: Seq<bool>, end: int, kind: bool)
    requires
        0 <= end < kinds.len(),
    ensures
        count_kind(kinds.subrange(0, end + 1), kind) == count_kind(kinds.subrange(0, end), kind)
            + if kinds[end] == kind { 1nat } else { 0nat },
{
    assert(kinds.subrange(0, end + 1).drop_last() =~= kinds.subrange(0, end));
}

/// Appending an entry adds one to its kind's count.
pub proof fn lemma_count_push(kinds: Seq<bool>, entry: bool, kind: bool)
    ensures
        count_kind(kinds.push(entry), kind) == count_kind(kinds, kind)
            + if entry == kind { 1nat } else { 0nat },
{
    assert(kinds.push(entry).drop_last() =~= kinds);
}

/// One more layer adds one multiple of `interval` exactly when it closes a period.
///
/// `(i + 1) / m = i / m + [i mod m = m - 1]` for `m >= 2`.
pub proof fn lemma_period_step(index: int, interval: int)
    requires
        interval >= 2,
        index >= 0,
    ensures
        (index + 1) / interval == index / interval
            + if index % interval == interval - 1 { 1int } else { 0int },
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(index, interval);
    let quotient = index / interval;
    let remainder = index % interval;
    if remainder == interval - 1 {
        assert(index + 1 == (quotient + 1) * interval + 0) by (nonlinear_arith)
            requires
                index == interval * quotient + remainder,
                remainder == interval - 1;
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
            index + 1,
            interval,
            quotient + 1,
            0,
        );
    } else {
        assert(index + 1 == quotient * interval + (remainder + 1)) by (nonlinear_arith)
            requires
                index == interval * quotient + remainder;
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
            index + 1,
            interval,
            quotient,
            remainder + 1,
        );
    }
}

/// Every column of every row lies below `sequence`.
pub open spec fn rows_below(rows: Seq<Vec<usize>>, sequence: int) -> bool {
    forall|b: int, k: int| 0 <= b < rows.len() && 0 <= k < rows[b]@.len() ==> rows[b]@[k] < sequence
}

/// The MTP cursor kept after truncating the target to `target` tokens.
///
/// `min(mtp, target - 1)`, and zero for an empty target.
pub open spec fn shifted_spec(target: int, mtp: int) -> int {
    if target == 0 {
        0
    } else if mtp < target - 1 {
        mtp
    } else {
        target - 1
    }
}

/// Tokens a transactional row stages after a call: `(W - C) + n`, saturated at `usize::MAX`.
pub open spec fn staged_request(committed: int, working: int, count: int) -> int {
    if working - committed + count > usize::MAX {
        usize::MAX as int
    } else {
        working - committed + count
    }
}

/// The smaller of two integers.
pub open spec fn min_int(a: int, b: int) -> int {
    if a < b {
        a
    } else {
        b
    }
}

} // verus!

// ---------------------------------------------------------------------------
// Layer schedule and cache slots
// ---------------------------------------------------------------------------

/// Returns whether layer `index` uses full attention in the derived hybrid schedule.
///
/// Configurations that omit an explicit schedule place one full-attention
/// layer after every `interval - 1` linear-attention layers: layer `i` is full
/// attention exactly when `(i + 1) mod interval = 0`, which for `interval >= 1`
/// is `i mod interval = interval - 1` (the form evaluated here, which cannot
/// overflow). An interval of zero behaves as one: every layer is full
/// attention.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(full =>
    ensures
        full == (interval <= 1 || index % interval == interval - 1),
))]
pub(crate) fn default_full_attention(index: usize, interval: usize) -> bool {
    interval <= 1 || index % interval == interval - 1
}

/// Returns the derived hybrid schedule of `layers` layers (`true` = full attention).
///
/// Entry `i` is [`default_full_attention`]`(i, interval)`. Proved: every
/// entry is exactly that rule, and the schedule holds exactly
/// `floor(layers / interval)` full-attention layers (every layer for an
/// interval of zero or one): one full-attention layer closes each complete
/// period of `interval` layers.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(schedule =>
    ensures
        schedule.len() == layers,
        forall|i: int| 0 <= i < layers
            ==> #[trigger] schedule[i] == (interval <= 1 || i % (interval as int) == interval - 1),
        count_kind(schedule@, true) == if interval <= 1 { layers as int } else { (layers as int) / (interval as int) },
))]
pub(crate) fn derived_full_attention(layers: usize, interval: usize) -> Vec<bool> {
    let mut schedule: Vec<bool> = Vec::with_capacity(layers);
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= layers,
            schedule.len() == index,
            forall|i: int| 0 <= i < index
                ==> #[trigger] schedule[i] == (interval <= 1 || i % (interval as int) == interval - 1),
            count_kind(schedule@, true) == if interval <= 1 { index as int } else { (index as int) / (interval as int) },
        decreases layers - index,
    ))]
    while index < layers {
        let full = default_full_attention(index, interval);
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_push(schedule@, full, true);
            if interval > 1 {
                lemma_period_step(index as int, interval as int);
            }
        }
        schedule.push(full);
        index += 1;
    }
    schedule
}

/// Dense per-family cache slots of a hybrid schedule.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CacheSlots {
    /// Slot of every layer inside its own family (full attention or recurrent).
    pub(crate) slots: Vec<usize>,
    /// Number of full-attention layers.
    pub(crate) full: usize,
    /// Number of recurrent (DeltaNet) layers.
    pub(crate) recurrent: usize,
}

/// Assigns each layer the next free slot of its cache family.
///
/// `full_attention[i]` names layer `i`'s family. Proved: the slot of layer
/// `i` is exactly the number of earlier layers of its family, `full` and
/// `recurrent` are exactly the two family sizes, and within each family the
/// slots strictly increase in schedule order and stay below the family's
/// size. So the `k`th full-attention layer owns native-KV slot `k` and the
/// `k`th DeltaNet layer recurrent slot `k`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    ensures
        plan.slots.len() == full_attention.len(),
        plan.full == count_kind(full_attention@, true),
        plan.recurrent == count_kind(full_attention@, false),
        plan.full + plan.recurrent == full_attention.len(),
        forall|i: int| 0 <= i < full_attention.len()
            ==> plan.slots[i] == count_kind(full_attention@.subrange(0, i), full_attention[i]),
        forall|i: int| 0 <= i < full_attention.len() && full_attention[i]
            ==> plan.slots[i] < plan.full,
        forall|i: int| 0 <= i < full_attention.len() && !full_attention[i]
            ==> plan.slots[i] < plan.recurrent,
        forall|i: int, j: int| 0 <= i < j < full_attention.len()
            && full_attention[i] == full_attention[j] ==> plan.slots[i] < plan.slots[j],
))]
pub(crate) fn cache_slots(full_attention: &[bool]) -> CacheSlots {
    let mut slots: Vec<usize> = Vec::with_capacity(full_attention.len());
    let mut full: usize = 0;
    let mut recurrent: usize = 0;
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= full_attention.len(),
            slots.len() == index,
            full + recurrent == index,
            full == count_kind(full_attention@.subrange(0, index as int), true),
            recurrent == count_kind(full_attention@.subrange(0, index as int), false),
            forall|i: int| 0 <= i < index
                ==> slots[i] == count_kind(full_attention@.subrange(0, i), full_attention[i]),
            forall|i: int| 0 <= i < index
                ==> slots[i] < count_kind(full_attention@.subrange(0, index as int), full_attention[i]),
            forall|i: int, j: int| 0 <= i < j < index
                && full_attention[i] == full_attention[j] ==> slots[i] < slots[j],
        decreases full_attention.len() - index,
    ))]
    while index < full_attention.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_extend(full_attention@, index as int, true);
            lemma_count_extend(full_attention@, index as int, false);
        }
        if full_attention[index] {
            slots.push(full);
            full += 1;
        } else {
            slots.push(recurrent);
            recurrent += 1;
        }
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(full_attention@.subrange(0, full_attention.len() as int) =~= full_attention@);
    }
    CacheSlots {
        slots,
        full,
        recurrent,
    }
}

// ---------------------------------------------------------------------------
// Compact right-padded batches
// ---------------------------------------------------------------------------

/// Returns whether every column of every row lies below `sequence`.
///
/// Proved: the result is exactly that predicate over all rows and columns.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(below =>
    ensures
        below == rows_below(valid_columns@, sequence as int),
))]
pub(crate) fn columns_below(valid_columns: &[Vec<usize>], sequence: usize) -> bool {
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= valid_columns.len(),
            forall|b: int, k: int| 0 <= b < row && 0 <= k < valid_columns@[b]@.len()
                ==> valid_columns@[b]@[k] < sequence,
        decreases valid_columns.len() - row,
    ))]
    while row < valid_columns.len() {
        let columns = &valid_columns[row];
        let mut k: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                row < valid_columns.len(),
                k <= columns.len(),
                columns@ == valid_columns@[row as int]@,
                forall|b: int, j: int| 0 <= b < row && 0 <= j < valid_columns@[b]@.len()
                    ==> valid_columns@[b]@[j] < sequence,
                forall|j: int| 0 <= j < k ==> columns@[j] < sequence,
            decreases columns.len() - k,
        ))]
        while k < columns.len() {
            if columns[k] >= sequence {
                return false;
            }
            k += 1;
        }
        row += 1;
    }
    true
}

/// Returns whether `columns` strictly increases.
///
/// Comparing adjacent pairs suffices; the proof establishes that the result
/// is exactly `c_i < c_j` for every `i < j`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(increasing =>
    ensures
        increasing == (forall|i: int, j: int| 0 <= i < j < columns.len() ==> columns@[i] < columns@[j]),
))]
pub(crate) fn strictly_increasing(columns: &[usize]) -> bool {
    let mut k: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            k <= columns.len(),
            forall|i: int, j: int| 0 <= i < j < k ==> columns@[i] < columns@[j],
        decreases columns.len() - k,
    ))]
    while k < columns.len() {
        if k > 0 && columns[k - 1] >= columns[k] {
            return false;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert forall|i: int| 0 <= i < k implies columns@[i] < columns@[k as int] by {
                if i < k - 1 {
                    assert(columns@[i] < columns@[k - 1]);
                }
            }
        }
        k += 1;
    }
    true
}

/// Returns every row's valid-token count `n_b`.
///
/// Proved: entry `b` is exactly the length of row `b`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(counts =>
    ensures
        counts.len() == valid_columns.len(),
        forall|b: int| 0 <= b < valid_columns.len() ==> counts[b] == valid_columns@[b]@.len(),
))]
pub(crate) fn row_counts(valid_columns: &[Vec<usize>]) -> Vec<usize> {
    let mut counts: Vec<usize> = Vec::with_capacity(valid_columns.len());
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= valid_columns.len(),
            counts.len() == row,
            forall|b: int| 0 <= b < row ==> counts[b] == valid_columns@[b]@.len(),
        decreases valid_columns.len() - row,
    ))]
    while row < valid_columns.len() {
        counts.push(valid_columns[row].len());
        row += 1;
    }
    counts
}

/// Returns the compact width `N = max_b n_b`, or zero for no rows.
///
/// Proved: the result bounds every count and, for a non-empty batch, is
/// attained by some row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    ensures
        forall|b: int| 0 <= b < counts.len() ==> counts[b] <= width,
        counts.len() > 0 ==> exists|b: int| 0 <= b < counts.len() && counts[b] == width,
        counts.len() == 0 ==> width == 0,
))]
pub(crate) fn widest(counts: &[usize]) -> usize {
    let mut width: usize = 0;
    let mut row: usize = 0;
    #[cfg(verus_keep_ghost)]
    proof_decl! { let ghost mut witness: int = 0; }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            forall|b: int| 0 <= b < row ==> counts[b] <= width,
            row > 0 ==> 0 <= witness < row && counts[witness] == width,
            row == 0 ==> width == 0,
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if row == 0 || counts[row] > width {
            width = counts[row];
            #[cfg(verus_keep_ghost)]
            proof! {
                witness = row as int;
            }
        }
        row += 1;
    }
    width
}

/// Returns whether every row's valid columns are the prefix `0..n_b`.
///
/// Proved: exactly when column `k` of every row equals `k`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(prefix =>
    ensures
        prefix == (forall|b: int, k: int| 0 <= b < valid_columns.len() && 0 <= k < valid_columns@[b]@.len()
            ==> valid_columns@[b]@[k] == k),
))]
pub(crate) fn prefix_rows(valid_columns: &[Vec<usize>]) -> bool {
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= valid_columns.len(),
            forall|b: int, k: int| 0 <= b < row && 0 <= k < valid_columns@[b]@.len()
                ==> valid_columns@[b]@[k] == k,
        decreases valid_columns.len() - row,
    ))]
    while row < valid_columns.len() {
        let columns = &valid_columns[row];
        let mut k: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                row < valid_columns.len(),
                k <= columns.len(),
                columns@ == valid_columns@[row as int]@,
                forall|b: int, j: int| 0 <= b < row && 0 <= j < valid_columns@[b]@.len()
                    ==> valid_columns@[b]@[j] == j,
                forall|j: int| 0 <= j < k ==> columns@[j] == j,
            decreases columns.len() - k,
        ))]
        while k < columns.len() {
            if columns[k] != k {
                return false;
            }
            k += 1;
        }
        row += 1;
    }
    true
}

/// Gather and restore indices of one row of the compact batch layout.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CompactRowPlan {
    /// Caller column read by each of the `width` compact slots.
    pub(crate) sources: Vec<usize>,
    /// Compact slot restored to each of the `sequence` caller columns
    /// (`width` addresses an appended zero slot).
    pub(crate) slots: Vec<usize>,
}

/// Plans one row of the compact right-padded layout.
///
/// The row's valid caller columns `c_0 < ... < c_{n-1} < sequence` move to
/// slots `0..n` of a `width`-slot block (`n <= width`); slots `n..width` are
/// padding and read the row's first valid column (column 0 for an empty row)
/// so every gather index stays in range. Restoring reads slot `k` into column
/// `c_k` and the zero slot `width` into every other column:
///
/// ```text
/// sources[k] = c_k            (k < n)
/// sources[k] = c_0 (or 0)     (n <= k < width)
/// slots[c_k] = k              (k < n)
/// slots[j]   = width          (j not a valid column)
/// ```
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    requires
        increasing_below(columns@, sequence as int),
        columns.len() <= width,
        sequence > 0,
    ensures
        plan.sources.len() == width,
        plan.slots.len() == sequence,
        forall|k: int| 0 <= k < columns.len() ==> plan.sources[k] == columns[k],
        forall|k: int| columns.len() <= k < width
            ==> plan.sources[k] == (if columns.len() > 0 { columns[0] } else { 0 }),
        forall|k: int| 0 <= k < width ==> plan.sources[k] < sequence,
        forall|k: int| 0 <= k < columns.len() ==> plan.slots[columns[k] as int] == k,
        forall|j: int| 0 <= j < sequence ==> plan.slots[j] == width
            || (plan.slots[j] < columns.len() && columns[plan.slots[j] as int] == j),
))]
pub(crate) fn compact_row_plan(columns: &[usize], sequence: usize, width: usize) -> CompactRowPlan {
    let count = columns.len();
    let filler = if count > 0 { columns[0] } else { 0 };
    let mut sources: Vec<usize> = Vec::with_capacity(width);
    let mut slot: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            slot <= width,
            count == columns.len(),
            count <= width,
            filler < sequence,
            filler == (if count > 0 { columns[0] } else { 0 }),
            increasing_below(columns@, sequence as int),
            sources.len() == slot,
            forall|k: int| 0 <= k < slot && k < count ==> sources[k] == columns[k],
            forall|k: int| count <= k < slot ==> sources[k] == filler,
            forall|k: int| 0 <= k < slot ==> sources[k] < sequence,
        decreases width - slot,
    ))]
    while slot < width {
        if slot < count {
            sources.push(columns[slot]);
        } else {
            sources.push(filler);
        }
        slot += 1;
    }
    let mut slots: Vec<usize> = Vec::with_capacity(sequence);
    let mut next: usize = 0;
    let mut column: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            column <= sequence,
            next <= count,
            count == columns.len(),
            count <= width,
            increasing_below(columns@, sequence as int),
            slots.len() == column,
            next < count ==> columns[next as int] >= column,
            forall|k: int| 0 <= k < next ==> columns[k] < column,
            forall|k: int| 0 <= k < next ==> slots[columns[k] as int] == k,
            forall|j: int| 0 <= j < column ==> slots[j] == width
                || (slots[j] < count && columns[slots[j] as int] == j),
        decreases sequence - column,
    ))]
    while column < sequence {
        if next < count && columns[next] == column {
            slots.push(next);
            // Strict increase moves the next unmatched column beyond `column`.
            #[cfg(verus_keep_ghost)]
            proof! {
                if next + 1 < count {
                    assert(columns[next as int] < columns[next + 1]);
                }
            }
            next += 1;
        } else {
            slots.push(width);
        }
        column += 1;
    }
    // Every column is below `sequence`, so the merge matched all of them.
    #[cfg(verus_keep_ghost)]
    proof! {
        if next < count {
            assert(columns[next as int] < sequence);
        }
    }
    CompactRowPlan { sources, slots }
}

/// Returns each row's slot validity `i < n_b`, or `None` when no slot is padding.
///
/// Proved: `None` exactly when every count reaches `width`; otherwise row
/// `b` holds `width` flags and flag `i` is `i < n_b`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(validity =>
    ensures
        match validity {
            None => forall|b: int| 0 <= b < counts.len() ==> counts[b] >= width,
            Some(rows) => {
                &&& exists|b: int| 0 <= b < counts.len() && counts[b] < width
                &&& rows.len() == counts.len()
                &&& forall|b: int| 0 <= b < counts.len() ==> (#[trigger] rows[b])@.len() == width
                &&& forall|b: int, i: int| 0 <= b < counts.len() && 0 <= i < width
                    ==> #[trigger] rows[b]@[i] == (i < counts[b])
            },
        },
))]
pub(crate) fn slot_validity(counts: &[usize], width: usize) -> Option<Vec<Vec<bool>>> {
    let mut padded = false;
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= counts.len(),
            padded == (exists|b: int| 0 <= b < row && counts[b] < width),
        decreases counts.len() - row,
    ))]
    while row < counts.len() {
        if counts[row] < width {
            padded = true;
        }
        row += 1;
    }
    if !padded {
        return None;
    }
    let mut rows: Vec<Vec<bool>> = Vec::with_capacity(counts.len());
    let mut b: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            b <= counts.len(),
            rows.len() == b,
            forall|r: int| 0 <= r < b ==> (#[trigger] rows[r])@.len() == width,
            forall|r: int, i: int| 0 <= r < b && 0 <= i < width
                ==> #[trigger] rows[r]@[i] == (i < counts[r]),
        decreases counts.len() - b,
    ))]
    while b < counts.len() {
        let mut flags: Vec<bool> = Vec::with_capacity(width);
        let mut slot: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                b < counts.len(),
                slot <= width,
                flags.len() == slot,
                forall|i: int| 0 <= i < slot ==> flags[i] == (i < counts[b as int]),
            decreases width - slot,
        ))]
        while slot < width {
            flags.push(slot < counts[b]);
            slot += 1;
        }
        rows.push(flags);
        b += 1;
    }
    Some(rows)
}

/// Returns each row's slot cache positions `W_b + i`, or `None` when one exceeds the index range.
///
/// Proved: `Some` exactly when every `W_b + i` (`i < width`) is representable
/// in both `usize` and `i64` (`None` names a row whose last slot is not), and
/// then row `b`, slot `i` is `W_b + i`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(positions =>
    ensures
        match positions {
            Some(rows) => {
                &&& rows.len() == starts.len()
                &&& forall|b: int| 0 <= b < starts.len() ==> (#[trigger] rows[b])@.len() == width
                &&& forall|b: int, i: int| 0 <= b < starts.len() && 0 <= i < width
                    ==> #[trigger] rows[b]@[i] == starts[b] + i
            },
            None => width > 0 && exists|b: int| 0 <= b < starts.len()
                && starts[b] + width - 1 > min_int(usize::MAX as int, i64::MAX as int),
        },
))]
pub(crate) fn slot_positions(starts: &[usize], width: usize) -> Option<Vec<Vec<i64>>> {
    let mut rows: Vec<Vec<i64>> = Vec::with_capacity(starts.len());
    let mut b: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            b <= starts.len(),
            rows.len() == b,
            forall|r: int| 0 <= r < b ==> (#[trigger] rows[r])@.len() == width,
            forall|r: int, i: int| 0 <= r < b && 0 <= i < width
                ==> #[trigger] rows[r]@[i] == starts[r] + i,
        decreases starts.len() - b,
    ))]
    while b < starts.len() {
        let mut positions: Vec<i64> = Vec::with_capacity(width);
        let mut slot: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                b < starts.len(),
                slot <= width,
                positions.len() == slot,
                forall|i: int| 0 <= i < slot ==> positions[i] == starts[b as int] + i,
            decreases width - slot,
        ))]
        while slot < width {
            let position = starts[b].checked_add(slot)?;
            if position as u64 > i64::MAX as u64 {
                return None;
            }
            positions.push(position as i64);
            slot += 1;
        }
        rows.push(positions);
        b += 1;
    }
    Some(rows)
}

/// Returns whether every entry equals every other (vacuously for at most one entry).
///
/// Proved: exactly when every entry equals the first.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(equal =>
    ensures
        equal == (forall|i: int| 0 <= i < values.len() ==> #[trigger] values@[i] == values@[0]),
))]
pub(crate) fn all_equal(values: &[usize]) -> bool {
    let mut k: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            k <= values.len(),
            forall|i: int| 0 <= i < k ==> #[trigger] values@[i] == values@[0],
        decreases values.len() - k,
    ))]
    while k < values.len() {
        if k > 0 && values[k - 1] != values[k] {
            return false;
        }
        k += 1;
    }
    true
}

/// Converts a host index to `i64`, saturating at `i64::MAX`.
///
/// Proved: the result is `min(value, i64::MAX)`; in-memory indices fit, so
/// the saturation never applies to a real index.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(signed =>
    ensures
        signed == min_int(value as int, i64::MAX as int),
))]
pub(crate) fn signed_index(value: usize) -> i64 {
    if value as u64 <= i64::MAX as u64 {
        value as i64
    } else {
        i64::MAX
    }
}

// ---------------------------------------------------------------------------
// MTP and speculative bookkeeping
// ---------------------------------------------------------------------------

/// Returns the MTP cursor after truncating the target cache to `target` tokens.
///
/// The shifted MTP cache holds one entry fewer than the target, so after a
/// truncation it keeps `min(mtp, target - 1)` entries (zero for an empty
/// target). Proved: the result is exactly that minimum, never grows, and
/// restores `target = MTP + 1` whenever the caches were aligned
/// (`mtp + 1 >= target`) and the target keeps at least one token.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(shifted =>
    ensures
        shifted == shifted_spec(target as int, mtp as int),
        shifted <= mtp,
        target == 0 ==> shifted == 0,
        target > 0 ==> shifted < target,
        target > 0 && mtp + 1 >= target ==> shifted + 1 == target,
))]
pub(crate) fn shifted_mtp_length(target: usize, mtp: usize) -> usize {
    if target == 0 {
        0
    } else if mtp < target - 1 {
        mtp
    } else {
        target - 1
    }
}

/// Applies [`shifted_mtp_length`] to every row of a truncation.
///
/// Proved: row `b` is exactly `min(mtp_b, target_b - 1)` (zero for an empty
/// target), so every aligned row stays aligned.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(shifted =>
    requires
        targets.len() == mtp.len(),
    ensures
        shifted.len() == targets.len(),
        forall|b: int| 0 <= b < targets.len()
            ==> shifted[b] == shifted_spec(targets[b] as int, mtp[b] as int),
))]
pub(crate) fn shifted_mtp_lengths(targets: &[usize], mtp: &[usize]) -> Vec<usize> {
    let mut shifted: Vec<usize> = Vec::with_capacity(targets.len());
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= targets.len(),
            targets.len() == mtp.len(),
            shifted.len() == row,
            forall|b: int| 0 <= b < row
                ==> shifted[b] == shifted_spec(targets[b] as int, mtp[b] as int),
        decreases targets.len() - row,
    ))]
    while row < targets.len() {
        shifted.push(shifted_mtp_length(targets[row], mtp[row]));
        row += 1;
    }
    shifted
}

/// Returns the first row whose staged suffix would exceed `capacity`, with its request.
///
/// Inside a transaction row `b` already stages `W_b - C_b` tokens and the
/// call adds `n_b`; the request saturates at `usize::MAX` instead of
/// wrapping. Rows are paired like `zip`: only the first
/// `min(|C|, |W|, |n|)` rows are checked. Proved: `Some((b, r))` names the
/// first row whose request `r` exceeds the capacity; `None` means every
/// checked request fits.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(overflow =>
    requires
        forall|b: int| 0 <= b < committed.len() && b < working.len() ==> committed[b] <= working[b],
    ensures
        match overflow {
            Some((row, requested)) => {
                &&& row < committed.len() && row < working.len() && row < counts.len()
                &&& requested == staged_request(committed[row as int] as int,
                    working[row as int] as int, counts[row as int] as int)
                &&& requested > capacity
                &&& forall|b: int| 0 <= b < row ==> staged_request(committed[b] as int,
                    working[b] as int, counts[b] as int) <= capacity
            },
            None => forall|b: int| 0 <= b < committed.len() && b < working.len() && b < counts.len()
                ==> staged_request(committed[b] as int, working[b] as int, counts[b] as int) <= capacity,
        },
))]
pub(crate) fn first_staging_overflow(
    committed: &[usize],
    working: &[usize],
    counts: &[usize],
    capacity: usize,
) -> Option<(usize, usize)> {
    let rows = committed.len().min(working.len()).min(counts.len());
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows <= committed.len() && rows <= working.len() && rows <= counts.len(),
            rows == committed.len() || rows == working.len() || rows == counts.len(),
            forall|b: int| 0 <= b < committed.len() && b < working.len() ==> committed[b] <= working[b],
            forall|b: int| 0 <= b < row ==> staged_request(committed[b] as int,
                working[b] as int, counts[b] as int) <= capacity,
        decreases rows - row,
    ))]
    while row < rows {
        let requested = (working[row] - committed[row]).saturating_add(counts[row]);
        if requested > capacity {
            return Some((row, requested));
        }
        row += 1;
    }
    None
}

/// Why `count` tokens cannot be appended after `baseline` cached tokens.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppendFault {
    /// `baseline + count` overflows `usize`.
    Overflow,
    /// The cache end `baseline + count` exceeds the capacity.
    ExceedsCapacity {
        /// The cache end the append would reach.
        requested: usize,
    },
}

/// Returns the cache end `baseline + count` of an append that fits within `capacity`.
///
/// Proved: `Ok(end)` exactly when the sum is representable and at most
/// the capacity, with `end = baseline + count`; each error names its reason.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(end =>
    ensures
        match end {
            Ok(end) => end == baseline + count && end <= capacity,
            Err(AppendFault::Overflow) => baseline + count > usize::MAX,
            Err(AppendFault::ExceedsCapacity { requested }) =>
                requested == baseline + count && requested > capacity,
        },
))]
pub(crate) fn append_end(
    baseline: usize,
    count: usize,
    capacity: usize,
) -> Result<usize, AppendFault> {
    let end = match baseline.checked_add(count) {
        Some(end) => end,
        None => return Err(AppendFault::Overflow),
    };
    if end > capacity {
        return Err(AppendFault::ExceedsCapacity { requested: end });
    }
    Ok(end)
}

/// Why a speculative proposal suffix cannot be staged.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StagingFault {
    /// `max_draft_tokens + 1` overflows `usize`.
    Overflow,
    /// The required staging `max_draft_tokens + 1` exceeds what is available.
    ExceedsAvailable {
        /// The required staging capacity `max_draft_tokens + 1`.
        required: usize,
    },
}

/// Returns the staging capacity `K + 1` one MTP cycle needs, if `available` holds it.
///
/// A cycle stages the lead token plus up to `K = max_draft_tokens` drafts in
/// the target cache and at most `K + 1` accepted pairs in the MTP cache.
/// Proved: `Ok(r)` exactly when `K + 1` is representable and at most
/// `available`, with `r = K + 1`; each error names its precise reason.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(staging =>
    ensures
        match staging {
            Ok(required) => required == max_draft_tokens + 1 && required <= available,
            Err(StagingFault::Overflow) => max_draft_tokens + 1 > usize::MAX,
            Err(StagingFault::ExceedsAvailable { required }) =>
                required == max_draft_tokens + 1 && required > available,
        },
))]
pub(crate) fn staging_requirement(
    max_draft_tokens: usize,
    available: usize,
) -> Result<usize, StagingFault> {
    let required = match max_draft_tokens.checked_add(1) {
        Some(required) => required,
        None => return Err(StagingFault::Overflow),
    };
    if required > available {
        return Err(StagingFault::ExceedsAvailable { required });
    }
    Ok(required)
}

/// Why a speculative cycle cannot start from the current cache cursors.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CycleFault {
    /// The state is not batch one with `target = MTP + 1`.
    Misaligned,
    /// The supplied history length differs from the target cursor.
    HistoryMismatch {
        /// The target cursor the history must match.
        expected: usize,
    },
    /// `target + required` overflows `usize`.
    Overflow,
    /// The staged target suffix would end beyond the cache capacity.
    ExceedsCapacity {
        /// The target length after staging the whole suffix.
        requested: usize,
    },
}

/// Checks a speculative cycle's cursors and returns the target length after staging.
///
/// `target` and `mtp` are the committed per-row lengths, `history` the
/// number of tokens before the lead, and `required` the staging capacity of
/// [`staging_requirement`]. Proved: `Ok(t)` exactly when the state is batch
/// one, `target = MTP + 1`, the history matches the target cursor, and
/// `t = target + required <= capacity`; each error names the first failing
/// condition in that order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    ensures
        ({
            let aligned = target.len() == 1 && mtp.len() == 1 && mtp[0] + 1 == target[0];
            match plan {
                Ok(requested) => {
                    &&& aligned
                    &&& history == target[0]
                    &&& requested == target[0] + required
                    &&& requested <= capacity
                },
                Err(CycleFault::Misaligned) => !aligned,
                Err(CycleFault::HistoryMismatch { expected }) =>
                    aligned && expected == target[0] && history != target[0],
                Err(CycleFault::Overflow) =>
                    aligned && history == target[0] && target[0] + required > usize::MAX,
                Err(CycleFault::ExceedsCapacity { requested }) => {
                    &&& aligned
                    &&& history == target[0]
                    &&& requested == target[0] + required
                    &&& requested > capacity
                },
            }
        }),
))]
pub(crate) fn speculative_cycle_plan(
    target: &[usize],
    mtp: &[usize],
    history: usize,
    required: usize,
    capacity: usize,
) -> Result<usize, CycleFault> {
    if target.len() != 1 || mtp.len() != 1 || mtp[0].checked_add(1) != Some(target[0]) {
        return Err(CycleFault::Misaligned);
    }
    if history != target[0] {
        return Err(CycleFault::HistoryMismatch {
            expected: target[0],
        });
    }
    let requested = match target[0].checked_add(required) {
        Some(requested) => requested,
        None => return Err(CycleFault::Overflow),
    };
    if requested > capacity {
        return Err(CycleFault::ExceedsCapacity { requested });
    }
    Ok(requested)
}

/// Returns the rotary context of a batched call: the largest active row's, or one.
///
/// Rows pair `contexts` with `counts` like `zip`; a row is active when its
/// count is positive. Proved: with an active row the result is the maximum
/// active context (attained by an active row and bounding all of them);
/// without one it is `1`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(context =>
    ensures
        ({
            let rows = min_int(counts.len() as int, contexts.len() as int);
            if exists|b: int| 0 <= b < rows && counts[b] > 0 {
                &&& exists|b: int| 0 <= b < rows && counts[b] > 0 && contexts[b] == context
                &&& forall|b: int| 0 <= b < rows && counts[b] > 0 ==> contexts[b] <= context
            } else {
                context == 1
            }
        }),
))]
pub(crate) fn call_rotary_context(counts: &[usize], contexts: &[usize]) -> usize {
    let rows = counts.len().min(contexts.len());
    let mut found = false;
    let mut best: usize = 0;
    let mut row: usize = 0;
    #[cfg(verus_keep_ghost)]
    proof_decl! { let ghost mut witness: int = 0; }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows as int == min_int(counts.len() as int, contexts.len() as int),
            found == (exists|b: int| 0 <= b < row && counts[b] > 0),
            found ==> 0 <= witness < row && counts[witness] > 0 && contexts[witness] == best,
            forall|b: int| 0 <= b < row && counts[b] > 0 ==> contexts[b] <= best,
        decreases rows - row,
    ))]
    while row < rows {
        if counts[row] > 0 && (!found || contexts[row] >= best) {
            best = contexts[row];
            found = true;
            #[cfg(verus_keep_ghost)]
            proof! {
                witness = row as int;
            }
        }
        row += 1;
    }
    if found { best } else { 1 }
}

/// Returns the index of the first token at or above `bound`.
///
/// Proved: `Some(i)` names the first such token; `None` means every token
/// lies below the bound (a valid vocabulary range).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(first =>
    ensures
        match first {
            Some(index) => {
                &&& index < tokens.len()
                &&& tokens[index as int] >= bound
                &&& forall|j: int| 0 <= j < index ==> tokens[j] < bound
            },
            None => forall|j: int| 0 <= j < tokens.len() ==> tokens[j] < bound,
        },
))]
pub(crate) fn first_token_at_or_above(tokens: &[u32], bound: usize) -> Option<usize> {
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= tokens.len(),
            forall|j: int| 0 <= j < index ==> tokens[j] < bound,
        decreases tokens.len() - index,
    ))]
    while index < tokens.len() {
        if tokens[index] as usize >= bound {
            return Some(index);
        }
        index += 1;
    }
    None
}

/// Returns the full-attention query projection width `heads * head_dim * (2 if gated)`.
///
/// Output-gated Qwen packs an equally sized gate beside Q. Proved: the
/// result is exactly that product whenever it is representable (the caller
/// validates the configuration's widths).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    requires
        heads * head_dim * (if gated { 2int } else { 1int }) <= usize::MAX,
    ensures
        width == heads * head_dim * (if gated { 2int } else { 1int }),
))]
pub(crate) fn gated_query_width(heads: usize, head_dim: usize, gated: bool) -> usize {
    let gate_factor: usize = if gated { 2 } else { 1 };
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(heads * head_dim <= heads * head_dim * gate_factor) by (nonlinear_arith)
            requires
                gate_factor >= 1,
                heads >= 0,
                head_dim >= 0;
    }
    heads * head_dim * gate_factor
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
