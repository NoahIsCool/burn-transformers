//! Verified integer planning shared by the dense cached decoders.
//!
//! The Llama family, Gemma, and gpt-oss keep their KV caches in at most two
//! groups, chronological full storage and one sliding ring, and read their
//! checkpoints through the same `u64` extent contract. This module owns that
//! bookkeeping once:
//!
//! * the cache layout: [`cache_slots`] numbers every layer inside its group
//!   (layer `l` of group `g` gets slot `#{l' < l : ring[l'] = g}`),
//!   [`group_indices`] numbers the allocated groups densely (full storage
//!   first), and [`truncation_leaves_bytes`] decides whether a truncation
//!   keeps the decode state's clean tails;
//! * [`widest_context`]: the rotary context `max_b ctx_b` (one when no row is
//!   active) a cached step evaluates and checks against cached limits;
//! * the checkpoint contract: [`checkpoint_extent`] and [`projection_extent`]
//!   convert widths to checked `u64` extents, and [`derived_head_dim`] is the
//!   head width `⌊hidden / heads⌋` of a configuration that omits it.
//!
//! The counting vocabulary of the slot proofs (`count_equal`, `numbered`,
//! and their lemmas) is public so the families' own schedule proofs reuse
//! it instead of restating it.
//!
//! # Verification
//!
//! Every function is dependency-free and checked in place by Verus through
//! `verus/models.rs` (and re-checked by the Llama, Gemma, and gpt-oss
//! harnesses, which include it at its crate path): for all admissible inputs
//! the proofs establish freedom from integer overflow and the stated
//! postconditions. The tensor programs driven by these plans are covered by
//! the families' decode parity tests.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Number of entries of `flags[0..end]` equal to `flag`.
pub open spec fn count_equal(flags: Seq<bool>, end: int, flag: bool) -> int
    decreases end,
{
    if end <= 0 {
        0
    } else {
        count_equal(flags, end - 1, flag) + if flags[end - 1] == flag { 1int } else { 0int }
    }
}

/// Some layer of group `flag` (sliding ring or full storage) holds slot `s`.
pub open spec fn numbered(slots: Seq<usize>, ring: Seq<bool>, flag: bool, s: int) -> bool {
    exists|l: int| 0 <= l < slots.len() && l < ring.len() && ring[l] == flag && slots[l] == s
}

/// `count_equal` reads only the prefix it counts.
pub proof fn lemma_count_prefix_stable(a: Seq<bool>, b: Seq<bool>, end: int, flag: bool)
    requires
        0 <= end <= a.len(),
        end <= b.len(),
        forall|i: int| 0 <= i < end ==> a[i] == b[i],
    ensures
        count_equal(a, end, flag) == count_equal(b, end, flag),
    decreases end,
{
    if end > 0 {
        lemma_count_prefix_stable(a, b, end - 1, flag);
    }
}

/// Proves `0 <= count_equal(flags, end, flag) <= end`.
pub proof fn count_equal_bounds(flags: Seq<bool>, end: int, flag: bool)
    requires
        0 <= end <= flags.len(),
    ensures
        0 <= count_equal(flags, end, flag) <= end,
    decreases end,
{
    if end > 0 {
        count_equal_bounds(flags, end - 1, flag);
    }
}

/// Proves that the two counts of a prefix add up to its length.
pub proof fn count_equal_partition(flags: Seq<bool>, end: int)
    requires
        0 <= end <= flags.len(),
    ensures
        count_equal(flags, end, true) + count_equal(flags, end, false) == end,
    decreases end,
{
    if end > 0 {
        count_equal_partition(flags, end - 1);
    }
}

/// A flag at `l` makes every longer prefix count strictly more of it.
pub proof fn count_prefix_grows(flags: Seq<bool>, l: int, end: int, flag: bool)
    requires
        0 <= l < end <= flags.len(),
        flags[l] == flag,
    ensures
        count_equal(flags, l, flag) < count_equal(flags, end, flag),
    decreases end - l,
{
    if end > l + 1 {
        count_prefix_grows(flags, l, end - 1, flag);
    }
}

} // verus!

// Cache layout.

/// Layer numbering inside the two KV cache groups of a decode state.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verifier::allow(autoderive_clone_without_spec))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheSlots {
    /// Index of each decoder layer within its own group.
    pub(crate) slots: Vec<usize>,
    /// Layers stored in full (chronological) storage.
    pub(crate) full_layers: usize,
    /// Layers stored in the sliding ring.
    pub(crate) ring_layers: usize,
}

/// Numbers every layer within its cache group.
///
/// `ring[l]` says whether layer `l` lives in the sliding ring group; its slot
/// is the number of earlier layers in the same group, so each group's layers
/// are numbered `0, 1, …` in model order and the two counts partition the
/// layers.
///
/// Proved: within each group the slots strictly increase in model order and
/// every slot below the group's size belongs to exactly one of its layers,
/// so each group's layers map bijectively, in order, onto `0..size` and no
/// two layers of one group ever share cache storage.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    ensures
        plan.slots.len() == ring.len(),
        plan.full_layers + plan.ring_layers == ring.len(),
        plan.full_layers == count_equal(ring@, ring.len() as int, false),
        plan.ring_layers == count_equal(ring@, ring.len() as int, true),
        forall|l: int| 0 <= l < ring.len() ==>
            #[trigger] plan.slots[l] == count_equal(ring@, l, ring[l]),
        forall|l: int| 0 <= l < ring.len() && !ring[l] ==> #[trigger] plan.slots[l] < plan.full_layers,
        forall|l: int| 0 <= l < ring.len() && ring[l] ==> #[trigger] plan.slots[l] < plan.ring_layers,
        forall|i: int, j: int| 0 <= i < j < ring.len() && ring[i] == ring[j] ==>
            plan.slots[i] < plan.slots[j],
        forall|s: int| 0 <= s < plan.full_layers ==> #[trigger] numbered(plan.slots@, ring@, false, s),
        forall|s: int| 0 <= s < plan.ring_layers ==> #[trigger] numbered(plan.slots@, ring@, true, s),
))]
pub(crate) fn cache_slots(ring: &[bool]) -> CacheSlots {
    let mut slots: Vec<usize> = Vec::with_capacity(ring.len());
    let mut full_layers: usize = 0;
    let mut ring_layers: usize = 0;
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= ring.len(),
            slots.len() == layer,
            full_layers == count_equal(ring@, layer as int, false),
            ring_layers == count_equal(ring@, layer as int, true),
            full_layers + ring_layers == layer,
            forall|l: int| 0 <= l < layer ==>
                #[trigger] slots[l] == count_equal(ring@, l, ring[l]),
            forall|s: int| 0 <= s < full_layers ==> #[trigger] numbered(slots@, ring@, false, s),
            forall|s: int| 0 <= s < ring_layers ==> #[trigger] numbered(slots@, ring@, true, s),
        decreases ring.len() - layer,
    ))]
    while layer < ring.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            count_equal_partition(ring@, layer as int);
        }
        #[cfg(verus_keep_ghost)]
        proof_decl! { let ghost before = slots@; }
        if ring[layer] {
            slots.push(ring_layers);
            ring_layers += 1;
        } else {
            slots.push(full_layers);
            full_layers += 1;
        }
        // Earlier slots keep their witnesses; the new slot is its group's next number.
        #[cfg(verus_keep_ghost)]
        proof! {
            assert forall|s: int| 0 <= s < full_layers implies
                #[trigger] numbered(slots@, ring@, false, s) by {
                if s < full_layers - 1 || ring[layer as int] {
                    assert(numbered(before, ring@, false, s));
                    let l = choose|l: int| 0 <= l < before.len() && l < ring.len() && ring[l] == false && before[l] == s;
                    assert(slots@[l] == s);
                } else {
                    assert(slots@[layer as int] == s);
                }
            }
            assert forall|s: int| 0 <= s < ring_layers implies
                #[trigger] numbered(slots@, ring@, true, s) by {
                if s < ring_layers - 1 || !ring[layer as int] {
                    assert(numbered(before, ring@, true, s));
                    let l = choose|l: int| 0 <= l < before.len() && l < ring.len() && ring[l] == true && before[l] == s;
                    assert(slots@[l] == s);
                } else {
                    assert(slots@[layer as int] == s);
                }
            }
        }
        layer += 1;
    }
    // Monotone prefix counts bound every slot by its group's final size and
    // order the slots of one group.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|l: int| 0 <= l < ring.len() && !ring[l]
            implies #[trigger] slots[l] < full_layers by {
            count_prefix_grows(ring@, l, ring.len() as int, false);
        }
        assert forall|l: int| 0 <= l < ring.len() && ring[l]
            implies #[trigger] slots[l] < ring_layers by {
            count_prefix_grows(ring@, l, ring.len() as int, true);
        }
        assert forall|i: int, j: int| 0 <= i < j < ring.len() && ring[i] == ring[j]
            implies slots[i] < slots[j] by {
            count_prefix_grows(ring@, i, j, ring[i]);
        }
    }
    CacheSlots {
        slots,
        full_layers,
        ring_layers,
    }
}

/// Bank indices of a layout's two KV groups.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GroupIndices {
    /// Index of the full-storage group, when any layer uses it.
    pub(crate) full: Option<usize>,
    /// Index of the sliding-ring group, when any layer uses it.
    pub(crate) ring: Option<usize>,
}

/// Numbers the allocated KV groups: full storage first, then the ring.
///
/// Proved: a group exists exactly when it has layers, the full group (if
/// any) is bank group 0, and the ring follows it, so the allocated groups
/// are numbered densely `0..groups` in allocation order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(groups =>
    ensures
        groups.full == if full_layers > 0 { Some(0usize) } else { None },
        groups.ring == if ring_layers > 0 {
            Some(if full_layers > 0 { 1usize } else { 0usize })
        } else {
            None
        },
))]
pub(crate) fn group_indices(full_layers: usize, ring_layers: usize) -> GroupIndices {
    let full = if full_layers > 0 { Some(0) } else { None };
    let ring = if ring_layers > 0 {
        Some(if full_layers > 0 { 1 } else { 0 })
    } else {
        None
    };
    GroupIndices { full, ring }
}

/// Returns whether truncating rows from `from` to `to` tokens leaves stale bytes.
///
/// A row shortened to a nonzero length keeps its later keys and values
/// beyond the new end; truncation to zero zeroes the row, and an unchanged
/// or longer target writes nothing. Proved: the result is exactly whether
/// some row (of the common prefix of both lists) is shortened to a nonzero
/// length, so the decode state's clean-tail flag survives a truncation
/// exactly when no row keeps stale bytes.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(stale =>
    ensures
        stale == exists|r: int| 0 <= r < from.len() && r < to.len()
            && #[trigger] to@[r] != 0 && to@[r] < from@[r],
))]
pub(crate) fn truncation_leaves_bytes(from: &[usize], to: &[usize]) -> bool {
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= from.len(),
            forall|r: int| 0 <= r < row && r < to.len() ==>
                !(#[trigger] to@[r] != 0 && to@[r] < from@[r]),
        decreases from.len() - row,
    ))]
    while row < from.len() && row < to.len() {
        if to[row] != 0 && to[row] < from[row] {
            return true;
        }
        row += 1;
    }
    false
}

// Cached steps.

/// Returns the widest rotary context of a step's active rows, or 1 when none is active.
///
/// Proved: with no context the result is 1; otherwise it is one of the
/// contexts and at least every other, so the step evaluates RoPE (and
/// checks its cached limit) at the largest context any row reaches.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(widest =>
    ensures
        contexts.len() == 0 ==> widest == 1,
        contexts.len() > 0 ==> exists|i: int| 0 <= i < contexts.len() && #[trigger] contexts@[i] == widest,
        forall|i: int| 0 <= i < contexts.len() ==> #[trigger] contexts@[i] <= widest,
))]
pub(crate) fn widest_context(contexts: &[usize]) -> usize {
    if contexts.is_empty() {
        return 1;
    }
    let mut widest = contexts[0];
    let mut index: usize = 1;
    #[cfg(verus_keep_ghost)]
    proof_decl! { let ghost mut witness: int = 0; }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            1 <= index <= contexts.len(),
            0 <= witness < index,
            contexts@[witness] == widest,
            forall|i: int| 0 <= i < index ==> #[trigger] contexts@[i] <= widest,
        decreases contexts.len() - index,
    ))]
    while index < contexts.len() {
        if contexts[index] > widest {
            widest = contexts[index];
            #[cfg(verus_keep_ghost)]
            proof! {
                witness = index as int;
            }
        }
        index += 1;
    }
    widest
}

// Checkpoint extents.

/// Why a projection width cannot be a checkpoint extent.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExtentError {
    /// `heads · head_dim` overflows `usize`.
    Overflow,
    /// The width exceeds the checkpoint format's `u64` extents.
    Format,
}

/// Returns `value` as a checkpoint-format `u64` extent, or `None` beyond `u64`.
///
/// Proved: `Some(value)` exactly when `value <= u64::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(extent =>
    ensures
        match extent {
            Some(extent) => extent == value,
            None => value > u64::MAX,
        },
))]
pub(crate) fn checkpoint_extent(value: usize) -> Option<u64> {
    u64::try_from(value).ok()
}

/// Returns the projection width `heads · head_dim` as a checkpoint-format extent.
///
/// Proved: exactly `heads · head_dim`, or `Overflow` when the product
/// exceeds `usize` and `Format` when it fits `usize` but not `u64`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    ensures
        match width {
            Ok(width) => width == heads * head_dim,
            Err(ExtentError::Overflow) => heads * head_dim > usize::MAX,
            Err(ExtentError::Format) => heads * head_dim <= usize::MAX && heads * head_dim > u64::MAX,
        },
))]
pub(crate) fn projection_extent(heads: usize, head_dim: usize) -> Result<u64, ExtentError> {
    let Some(width) = heads.checked_mul(head_dim) else {
        return Err(ExtentError::Overflow);
    };
    match checkpoint_extent(width) {
        Some(extent) => Ok(extent),
        None => Err(ExtentError::Format),
    }
}

/// Returns the head width derived when `head_dim` is absent or `null`.
///
/// Proved: `hidden / heads`, rounded down, or zero when there are no heads
/// (validation then rejects the configuration with a field error).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    ensures
        width == if heads == 0 { 0 } else { hidden / heads },
))]
pub(crate) fn derived_head_dim(hidden: usize, heads: usize) -> usize {
    hidden.checked_div(heads).unwrap_or(0)
}

#[cfg(test)]
#[path = "decoder_plan_tests.rs"]
/// Cache-slot numbering, group indices, truncation, context, and extent contracts.
mod tests;
