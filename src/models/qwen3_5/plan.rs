//! Dependency-free planning for the Qwen families, verified in place by Verus.
//!
//! Every function here is integer and index logic with no Burn types, so
//! `verus/qwen.rs` checks the exact bodies that ship (see `verus/README.md`):
//!
//! * [`default_full_attention`] — the legacy hybrid schedule: layer `i` is
//!   full attention exactly when `(i + 1)` is a multiple of the interval;
//! * [`cache_slots`] — the dense slot of every layer inside its cache family;
//! * [`compact_row_plan`] — the gather and restore indices of one row of the
//!   compact right-padded batch layout used by cached execution;
//! * [`shifted_mtp_length`] — the MTP cursor after a target truncation, which
//!   keeps the speculative invariant `target = MTP + 1`.
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

} // verus!

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
/// `full_attention[i]` names layer `i`'s family. Within each family the slots
/// strictly increase in schedule order and stay below the family's size, so
/// (by counting) they are exactly `0..size`: the `k`th full-attention layer
/// owns native-KV slot `k` and the `k`th DeltaNet layer recurrent slot `k`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    ensures
        plan.slots.len() == full_attention.len(),
        plan.full + plan.recurrent == full_attention.len(),
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
            forall|i: int| 0 <= i < index && full_attention[i] ==> slots[i] < full,
            forall|i: int| 0 <= i < index && !full_attention[i] ==> slots[i] < recurrent,
            forall|i: int, j: int| 0 <= i < j < index
                && full_attention[i] == full_attention[j] ==> slots[i] < slots[j],
        decreases full_attention.len() - index,
    ))]
    while index < full_attention.len() {
        if full_attention[index] {
            slots.push(full);
            full += 1;
        } else {
            slots.push(recurrent);
            recurrent += 1;
        }
        index += 1;
    }
    CacheSlots {
        slots,
        full,
        recurrent,
    }
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
            increasing_below(columns@, sequence as int),
            sources.len() == slot,
            forall|k: int| 0 <= k < slot && k < count ==> sources[k] == columns[k],
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

/// Returns the MTP cursor after truncating the target cache to `target` tokens.
///
/// The shifted MTP cache holds one entry fewer than the target, so after a
/// truncation it keeps `min(mtp, target - 1)` entries (zero for an empty
/// target). When the caches were aligned (`mtp + 1 >= target`) and the target
/// keeps at least one token, the result restores `target = MTP + 1` exactly.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(shifted =>
    ensures
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

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
