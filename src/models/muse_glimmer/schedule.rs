//! Dependency-free Muse-Glimmer layer schedules and cached key geometry, verified in place by Verus.
//!
//! # Layer schedule
//!
//! Muse counts its hybrid schedule *backward* from the last layer: of `n`
//! layers, layer `i` is a global (full-attention, NoPE) layer exactly when
//! `(n - 1 - i) mod 4 = 0`, so the final layer is always global and three
//! local (sliding, RoPE) layers precede each global one.
//!
//! Decode caches store each kind in one homogeneous group, so every layer
//! needs its dense *slot* inside its group: the number of earlier layers of
//! the same kind. [`group_slots`](crate::models::muse_glimmer::schedule::group_slots) computes that map; its proof establishes
//! that each group's slots are exactly `0, 1, …` in layer order, hence
//! injective and dense.
//!
//! # Ring key slices
//!
//! A batched cached step evaluates a `[B, T]` block whose query `k` of row
//! `b` sits at position `L_b + k`. Sliding layers read their ring history in
//! physical slot order (not chronologically) and append the step's fresh
//! block, so the keys are `[history H | block T]` and fresh entry `j` of
//! *every* row is key column `H + j`. Under a window of `W` positions, query
//! `k` sees `L_b + k - W < p <= L_b + k`: fresh entries `k - W < j <= k`
//! and, only while `k + 1 < W`, some history slots. A query chunk `s..e`
//! therefore needs exactly
//!
//! ```text
//! keys = H + s + 1 - W .. H + e     if s + 1 >= W   (no history)
//!      = 0             .. H + e     otherwise       (the whole ring)
//! ```
//!
//! which is row-independent and at most `H + W - 1 + (e - s)` keys wide
//! ([`ring_chunk_keys`](crate::models::muse_glimmer::schedule::ring_chunk_keys)),
//! instead of every key of the step.
//!
//! All three functions are checked through `verus/deepseek_muse.rs`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

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

/// A prefix holds at most as many matching entries as it has entries.
pub proof fn lemma_count_bounded(kinds: Seq<bool>, kind: bool)
    ensures
        count_kind(kinds, kind) <= kinds.len(),
    decreases kinds.len(),
{
    if kinds.len() > 0 {
        lemma_count_bounded(kinds.drop_last(), kind);
    }
}

/// Fresh block entry `j` is visible to block query `k` under a `window`-position window.
///
/// Both sit in the same row at positions `L_b + j` and `L_b + k`, so the
/// causal and window rules reduce to `k - window < j <= k`.
pub open spec fn fresh_visible(k: int, j: int, window: int) -> bool {
    0 <= j <= k && k < j + window
}

} // verus!

/// Returns whether layer `index` of `layers` is a global (full-attention) layer.
///
/// Global layers sit at `(layers - 1 - index) mod 4 = 0`, counted backward
/// so the last layer is global.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(global =>
    requires
        index < layers,
    ensures
        global == ((layers - 1 - index) % 4 == 0),
        index + 1 == layers ==> global,
))]
#[allow(
    clippy::manual_is_multiple_of,
    reason = "Verus specifies `%` but has no specification for `usize::is_multiple_of`"
)]
pub(crate) fn is_global_layer(layers: usize, index: usize) -> bool {
    (layers - 1 - index) % 4 == 0
}

/// Returns each layer's dense slot within its kind's cache group.
///
/// `global[i]` marks layer `i` as global; the slot of layer `i` is the number
/// of earlier layers of the same kind, so each group's slots run `0, 1, …`
/// in layer order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(slots =>
    ensures
        slots.len() == global.len(),
        forall|i: int| 0 <= i < global.len()
            ==> slots@[i] == count_kind(global@.subrange(0, i), global@[i]),
        forall|i: int, j: int| 0 <= i < j < global.len() && global@[i] == global@[j]
            ==> slots@[i] < slots@[j],
))]
pub(crate) fn group_slots(global: &[bool]) -> Vec<usize> {
    let mut slots: Vec<usize> = Vec::with_capacity(global.len());
    let mut next_global: usize = 0;
    let mut next_local: usize = 0;
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= global.len(),
            slots.len() == index,
            next_global == count_kind(global@.subrange(0, index as int), true),
            next_local == count_kind(global@.subrange(0, index as int), false),
            forall|i: int| 0 <= i < index
                ==> slots@[i] == count_kind(global@.subrange(0, i), global@[i]),
            forall|i: int, j: int| 0 <= i < j < index && global@[i] == global@[j]
                ==> slots@[i] < slots@[j],
            forall|i: int| 0 <= i < index
                ==> slots@[i] < count_kind(global@.subrange(0, index as int), global@[i]),
        decreases global.len() - index,
    ))]
    while index < global.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_extend(global@, index as int, true);
            lemma_count_extend(global@, index as int, false);
            lemma_count_bounded(global@.subrange(0, index as int), true);
            lemma_count_bounded(global@.subrange(0, index as int), false);
        }
        if global[index] {
            slots.push(next_global);
            next_global += 1;
        } else {
            slots.push(next_local);
            next_local += 1;
        }
        index += 1;
    }
    slots
}

/// Returns the key slice `(lo, hi)` that block query rows `start..end` need over `[history | block]`.
///
/// See the module documentation: rows at or past `window - 1` skip the ring
/// history and read exactly the union of their windows; earlier rows read
/// the whole history (its slots are in ring order, so no narrower contiguous
/// slice is exact); and every slice ends at the chunk's last fresh entry.
/// Requires `start < end` and `window > 0`; returns `None` only when
/// `history + end` overflows `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(range =>
    requires
        start < end,
        window > 0,
    ensures
        match range {
            Some((lo, hi)) => {
                &&& hi == history + end
                &&& lo <= hi
                &&& forall|k: int, j: int| start <= k < end && #[trigger] fresh_visible(k, j, window as int)
                        ==> lo <= history + j && history + j < hi
                &&& (start + 1 < window ==> lo == 0)
                &&& (start + 1 >= window ==> history <= lo && lo == history + start + 1 - window)
                &&& hi - lo <= history + window - 1 + (end - start)
            },
            None => history + end > usize::MAX,
        },
))]
pub(crate) fn ring_chunk_keys(
    history: usize,
    start: usize,
    end: usize,
    window: usize,
) -> Option<(usize, usize)> {
    let hi = history.checked_add(end)?;
    // `start < end` keeps `start + 1` in range, and `lo <= history + start < hi`.
    let lo = if start >= window - 1 {
        history + (start + 1 - window)
    } else {
        0
    };
    Some((lo, hi))
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
/// Schedule and slot properties against brute-force oracles.
mod tests;
