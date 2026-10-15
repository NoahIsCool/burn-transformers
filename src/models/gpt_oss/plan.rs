//! Dependency-free gpt-oss planning, verified in place by Verus.
//!
//! Everything here is integer bookkeeping consumed by the configuration,
//! the loader, and the decode cache:
//!
//! * `default_layer_slides`: the reference's default attention schedule;
//! * `cache_slots`: the numbering of layers inside the two KV cache groups
//!   (full storage and sliding ring);
//! * `interleaved_offset`: where the loader finds expert `e`'s gate and up
//!   weights inside the stacked, column-interleaved `gate_up_proj`.
//!
//! # Interleaved expert projections
//!
//! The reference stores every expert's fused gate/up projection input-major,
//! `gate_up_proj: [E, d, 2f]`, and splits it as `gate = gate_up[..., ::2]`,
//! `up = gate_up[..., 1::2]`: column `2j` of row `i` is gate output `j` and
//! column `2j + 1` is up output `j`. The loader therefore copies, for every
//! expert `e`, input feature `i`, and output `j`,
//!
//! ```text
//! gate_e[i, j] = gate_up[(e · d + i) · 2f + 2j]
//! up_e[i, j]   = gate_up[(e · d + i) · 2f + 2j + 1]
//! ```
//!
//! and the same rule splits the stacked bias `[E, 2f]` with `d = 1`.
//!
//! # Verification
//!
//! Each function is checked by Verus through `verus/gpt_oss.rs` (see
//! `verus/README.md`): for all admissible inputs the proofs establish
//! freedom from integer overflow and the stated postconditions. The tensor
//! programs driven by these plans are covered by `load_tests.rs` and the
//! decode parity tests.

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

/// Returns whether layer `layer` slides under the reference's default schedule.
///
/// `GptOssConfig` writes `sliding_attention` when `bool((l + 1) % 2)`; the
/// equivalent `l % 2 == 0` cannot overflow.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(slides =>
    ensures
        slides == ((layer as int + 1) % 2 != 0),
))]
// `usize::is_multiple_of` has no Verus specification; the remainder is checked directly.
#[allow(clippy::manual_is_multiple_of)]
pub(crate) const fn default_layer_slides(layer: usize) -> bool {
    layer % 2 == 0
}

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
/// `ring[l]` says whether layer `l` lives in the sliding-ring group; its slot
/// is the number of earlier layers of the same group, so each group's layers
/// are numbered `0, 1, …` in model order and the two counts partition the
/// layers.
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
        decreases ring.len() - layer,
    ))]
    while layer < ring.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            count_equal_partition(ring@, layer as int);
        }
        if ring[layer] {
            slots.push(ring_layers);
            ring_layers += 1;
        } else {
            slots.push(full_layers);
            full_layers += 1;
        }
        layer += 1;
    }
    // Monotone prefix counts bound every slot by its group's final size.
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
    }
    CacheSlots {
        slots,
        full_layers,
        ring_layers,
    }
}

/// Which half of an interleaved gate/up column pair a weight belongs to.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GluHalf {
    /// The gate projection: even columns `2j`.
    Gate,
    /// The up (content) projection: odd columns `2j + 1`.
    Up,
}

/// Returns the element offset of `half`'s output `j` of row `i` of expert `e`.
///
/// The stacked tensor is `[experts, rows, 2 · width]` row-major with gate and
/// up outputs interleaved along its last axis (see the module
/// documentation): gate output `j` is column `2j`, up output `j` column
/// `2j + 1`. The offset lies inside the tensor.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(offset =>
    requires
        expert < experts,
        row < rows,
        column < width,
        experts * rows * (2 * width) <= usize::MAX,
    ensures
        offset == (expert * rows + row) * (2 * width) + 2 * column
            + if half == GluHalf::Up { 1int } else { 0int },
        offset < experts * rows * (2 * width),
))]
// `experts` appears only in the contract: it bounds the offset Verus proves.
#[allow(unused_variables)]
pub(crate) fn interleaved_offset(
    experts: usize,
    rows: usize,
    width: usize,
    expert: usize,
    row: usize,
    column: usize,
    half: GluHalf,
) -> usize {
    #[cfg(verus_keep_ghost)]
    proof! {
        let pair = 2 * width as int;
        assert(expert * rows + row < experts * rows) by (nonlinear_arith)
            requires expert < experts, row < rows;
        assert((expert * rows + row) * pair + pair <= experts * rows * pair) by (nonlinear_arith)
            requires expert * rows + row < experts * rows, pair >= 0;
        assert(0 <= expert * rows) by (nonlinear_arith) requires 0 <= expert, 0 <= rows;
        assert(expert * rows <= expert * rows + row);
        assert(expert * rows + row <= (expert * rows + row) * pair) by (nonlinear_arith)
            requires pair >= 1, expert * rows + row >= 0;
    }
    let base = (expert * rows + row) * (2 * width) + 2 * column;
    match half {
        GluHalf::Gate => base,
        GluHalf::Up => base + 1,
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
