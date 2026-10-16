//! Pure index planning for the Llama-family schedule and batched cache steps.
//!
//! Everything here is integer bookkeeping that the tensor code consumes:
//!
//! * the SmolLM3 NoPE rule (`rotary_by_interval`) and the Qwen sliding
//!   rule (`sliding_from_layer`) that derive per-layer schedules, and the
//!   whole-model schedules built from them (`interval_rotary_flags`,
//!   `listed_rotary_flags`, `suffix_sliding_flags`, `nope_sliding_flags`,
//!   and `first_set` for the first sliding layer without a window);
//! * `used_rotary_sets`, the rotary parameter sets some layer rotates with,
//!   and `tightest_limit`, the used set whose cached-context limit binds;
//! * the cache layout: `plan_ring`, which assigns sliding layers below the
//!   capacity to one ring group of one window;
//! * the batched-step layout of one row of a `[batch, block]` token block:
//!   whether its valid columns form one run (`is_one_run`), the cache
//!   positions of its slots (`slot_positions`), the gather that compacts
//!   interior padding into a right-padded block (`compaction_sources`), and
//!   the gathers that restore the rectangular layout with exact zeros
//!   (`scatter_sources` from a compact row, `restore_slots` from any run
//!   offset), composed for one row by `plan_row_step` and for the whole
//!   batch by `plan_step`, which chooses between slicing and gathering.
//!
//! The cache-group numbering (`cache_slots`, `group_indices`,
//! `truncation_leaves_bytes`), the step context (`widest_context`), and the
//! checkpoint extents (`checkpoint_extent`, `projection_extent`,
//! `derived_head_dim`) are shared with Gemma and gpt-oss and live in
//! [`crate::models::shared::decoder_plan`]; `signed_indices` is
//! [`crate::models::shared::plan::signed_indices`]. This module re-exports
//! them for the family's callers.
//!
//! A batched step places row `b`'s `n_b` new tokens in block slots
//! `[c_b, c_b + n_b)`. Slot `t` in that run is the token at absolute cache
//! position `L_b + (t - c_b)`, where `L_b` is the row's committed length:
//! these are both the implicit RoPE positions and the query positions of the
//! causal and sliding-window masks. Slots outside the run are padding; they
//! receive a *fill* position (the row's last valid position, the last cached
//! one for an inactive row, or zero) so that, wherever possible, a padding
//! query still sees at least one key and its discarded output stays finite.
//!
//! # Verified planning
//!
//! Each function is dependency-free and checked in place by Verus through
//! `verus/llama.rs` (see `verus/README.md`): for all admissible inputs the
//! proofs establish freedom from integer overflow and the stated
//! postconditions. The tensor programs driven by these plans are covered by
//! the Rust tests in `model/step_tests.rs` and the parity tests in
//! `model/decode_tests.rs`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

// The cache layout, step context, and checkpoint extents are shared with
// Gemma and gpt-oss; the counting vocabulary also underpins this module's
// schedule proofs.
#[cfg(verus_keep_ghost)]
use crate::models::shared::decoder_plan::{count_equal, lemma_count_prefix_stable};
pub(crate) use crate::models::shared::{
    decoder_plan::{
        ExtentError, cache_slots, checkpoint_extent, derived_head_dim, group_indices,
        projection_extent, truncation_leaves_bytes, widest_context,
    },
    plan::signed_indices,
};

#[cfg(verus_keep_ghost)]
verus! {

/// `s` is strictly increasing and every entry lies below `bound`.
pub open spec fn sorted_unique_below(s: Seq<usize>, bound: int) -> bool {
    &&& forall|i: int, j: int| 0 <= i < j < s.len() ==> s[i] < s[j]
    &&& forall|i: int| 0 <= i < s.len() ==> s[i] < bound
}

/// The fill position of a row with cache cursor `start` and `count` new tokens.
pub open spec fn fill_position(start: int, count: int) -> int {
    if count > 0 {
        start + count - 1
    } else if start > 0 {
        start - 1
    } else {
        0
    }
}

/// The successor of `a` is a multiple of `c` exactly when `a mod c = c - 1`.
///
/// With `a = c·q + r` and `0 <= r < c`: if `r + 1 < c` then
/// `(a + 1) mod c = r + 1 != 0`; otherwise `a + 1 = c·(q + 1)`.
pub proof fn lemma_successor_mod(a: int, c: int)
    requires
        c > 0,
        a >= 0,
    ensures
        ((a + 1) % c == 0) == (a % c == c - 1),
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(a, c);
    let q = a / c;
    let r = a % c;
    assert(a + 1 == c * q + (r + 1));
    if r + 1 < c {
        vstd::arithmetic::div_mod::lemma_mod_multiples_vanish(q, r + 1, c);
        vstd::arithmetic::div_mod::lemma_small_mod((r + 1) as nat, c as nat);
    } else {
        assert(a + 1 == c * (q + 1) + 0) by (nonlinear_arith)
            requires
                a + 1 == c * q + (r + 1),
                r + 1 == c,
        ;
        vstd::arithmetic::div_mod::lemma_mod_multiples_vanish(q + 1, 0, c);
        vstd::arithmetic::div_mod::lemma_small_mod(0nat, c as nat);
    }
}

/// Stepping `n` to `n + 1` raises `⌊n / k⌋` exactly when `k` divides `n + 1`.
///
/// With `n = k·q + r` and `0 <= r < k`: if `r + 1 < k` the quotient stays
/// `q` and the remainder `r + 1` is nonzero; otherwise `n + 1 = k·(q + 1)`.
pub proof fn lemma_div_successor(n: int, k: int)
    requires
        n >= 0,
        k > 0,
    ensures
        (n + 1) / k == n / k + if (n + 1) % k == 0 { 1int } else { 0int },
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(n, k);
    let q = n / k;
    let r = n % k;
    if r + 1 < k {
        assert(n + 1 == q * k + (r + 1)) by (nonlinear_arith)
            requires
                n == k * q + r,
        ;
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(n + 1, k, q, r + 1);
    } else {
        assert(n + 1 == (q + 1) * k + 0) by (nonlinear_arith)
            requires
                n == k * q + r,
                r + 1 == k,
        ;
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(n + 1, k, q + 1, 0);
    }
}

} // verus!

/// Returns whether SmolLM3 layer `layer` rotates under a NoPE period `interval`.
///
/// Transformers derives `no_rope_layers[l] = int((l + 1) % interval != 0)`:
/// every `interval`-th layer (1-based) is a NoPE layer. The equivalent test
/// `l % interval != interval - 1` cannot overflow.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rotary =>
    requires
        interval > 0,
    ensures
        rotary == ((layer as int + 1) % (interval as int) != 0),
))]
pub(crate) fn rotary_by_interval(layer: usize, interval: usize) -> bool {
    // (l + 1) mod k == 0 exactly when l mod k == k - 1.
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_successor_mod(layer as int, interval as int);
    }
    layer % interval != interval - 1
}

/// Returns whether Qwen layer `layer` slides when sliding starts at `first`.
///
/// Transformers derives `layer_types[l] = sliding_attention` for
/// `l >= max_window_layers` when a window is enabled.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sliding =>
    ensures
        sliding == (layer >= first),
))]
pub(crate) const fn sliding_from_layer(layer: usize, first: usize) -> bool {
    layer >= first
}

/// Returns SmolLM3's default rotary flag of each of `layers` layers.
///
/// Proved: layer `l` rotates exactly when `(l + 1) mod interval != 0`, as
/// Transformers derives `no_rope_layers`, so exactly `⌊layers / interval⌋`
/// layers (the 1-based multiples of the interval) are NoPE layers.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(flags =>
    requires
        interval > 0,
    ensures
        flags.len() == layers,
        forall|l: int| 0 <= l < layers ==>
            #[trigger] flags@[l] == ((l + 1) % (interval as int) != 0),
        count_equal(flags@, layers as int, false) == layers / interval,
))]
pub(crate) fn interval_rotary_flags(layers: usize, interval: usize) -> Vec<bool> {
    let mut flags: Vec<bool> = Vec::with_capacity(layers);
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= layers,
            interval > 0,
            flags.len() == layer,
            forall|l: int| 0 <= l < layer ==>
                #[trigger] flags@[l] == ((l + 1) % (interval as int) != 0),
            count_equal(flags@, layer as int, false) == layer / interval,
        decreases layers - layer,
    ))]
    while layer < layers {
        #[cfg(verus_keep_ghost)]
        proof_decl! { let ghost before = flags@; }
        flags.push(rotary_by_interval(layer, interval));
        // The new flag counts as NoPE exactly when the quotient steps.
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_prefix_stable(before, flags@, layer as int, false);
            lemma_div_successor(layer as int, interval as int);
        }
        layer += 1;
    }
    flags
}

/// Returns the rotary flags of an explicit `no_rope_layers` list, or `None`.
///
/// Proved: the list is accepted exactly when every entry is `0` (NoPE) or
/// `1` (RoPE), and then layer `l` rotates exactly when its entry is `1`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(flags =>
    ensures
        match flags {
            Some(flags) => {
                &&& flags.len() == listed.len()
                &&& forall|l: int| 0 <= l < listed.len() ==> #[trigger] listed@[l] <= 1
                &&& forall|l: int| 0 <= l < listed.len() ==>
                    #[trigger] flags@[l] == (listed@[l] == 1)
            },
            None => exists|l: int| 0 <= l < listed.len() && #[trigger] listed@[l] > 1,
        },
))]
pub(crate) fn listed_rotary_flags(listed: &[usize]) -> Option<Vec<bool>> {
    let mut flags: Vec<bool> = Vec::with_capacity(listed.len());
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= listed.len(),
            flags.len() == layer,
            forall|l: int| 0 <= l < layer ==> #[trigger] listed@[l] <= 1,
            forall|l: int| 0 <= l < layer ==> #[trigger] flags@[l] == (listed@[l] == 1),
        decreases listed.len() - layer,
    ))]
    while layer < listed.len() {
        if listed[layer] > 1 {
            return None;
        }
        flags.push(listed[layer] == 1);
        layer += 1;
    }
    Some(flags)
}

/// Returns Qwen's default sliding flag of each of `layers` layers.
///
/// Proved: layer `l` slides exactly when a window applies (`windowed`) and
/// `l >= first` (`max_window_layers`), so the sliding layers form a suffix.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(flags =>
    ensures
        flags.len() == layers,
        forall|l: int| 0 <= l < layers ==> #[trigger] flags@[l] == (windowed && l >= first),
))]
pub(crate) fn suffix_sliding_flags(layers: usize, first: usize, windowed: bool) -> Vec<bool> {
    let mut flags: Vec<bool> = Vec::with_capacity(layers);
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= layers,
            flags.len() == layer,
            forall|l: int| 0 <= l < layer ==> #[trigger] flags@[l] == (windowed && l >= first),
        decreases layers - layer,
    ))]
    while layer < layers {
        flags.push(windowed && sliding_from_layer(layer, first));
        layer += 1;
    }
    flags
}

/// Returns SmolLM3's default sliding flags: its NoPE layers, when enabled.
///
/// Proved: layer `l` slides exactly when sliding is `enabled` (the config's
/// `use_sliding_window` with a window present) and the layer does not rotate.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(flags =>
    ensures
        flags.len() == rotary.len(),
        forall|l: int| 0 <= l < rotary.len() ==> #[trigger] flags@[l] == (enabled && !rotary@[l]),
))]
pub(crate) fn nope_sliding_flags(rotary: &[bool], enabled: bool) -> Vec<bool> {
    let mut flags: Vec<bool> = Vec::with_capacity(rotary.len());
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= rotary.len(),
            flags.len() == layer,
            forall|l: int| 0 <= l < layer ==> #[trigger] flags@[l] == (enabled && !rotary@[l]),
        decreases rotary.len() - layer,
    ))]
    while layer < rotary.len() {
        flags.push(enabled && !rotary[layer]);
        layer += 1;
    }
    flags
}

/// Returns the first set flag's index, or `None` when no flag is set.
///
/// Proved: a returned index holds a set flag and every earlier flag is
/// clear; `None` means every flag is clear. The schedule reports the first
/// sliding layer without a window this way.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(first =>
    ensures
        match first {
            Some(index) => {
                &&& index < flags.len()
                &&& flags@[index as int]
                &&& forall|l: int| 0 <= l < index ==> !#[trigger] flags@[l]
            },
            None => forall|l: int| 0 <= l < flags.len() ==> !#[trigger] flags@[l],
        },
))]
pub(crate) fn first_set(flags: &[bool]) -> Option<usize> {
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= flags.len(),
            forall|l: int| 0 <= l < index ==> !#[trigger] flags@[l],
        decreases flags.len() - index,
    ))]
    while index < flags.len() {
        if flags[index] {
            return Some(index);
        }
        index += 1;
    }
    None
}

// Cache layout: which layers share the sliding ring.

/// How far back one layer's queries see, as the cache layout needs it.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerReach {
    /// Every earlier key (global attention).
    Global,
    /// The `window` newest keys (sliding-window attention).
    Sliding(
        /// Window width `W`.
        usize,
    ),
    /// Chunked-local attention, which this cache cannot store.
    Chunked,
}

/// The sliding-ring assignment of a cache layout.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verifier::allow(autoderive_clone_without_spec))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RingPlan {
    /// Whether each layer lives in the sliding ring (else in full storage).
    pub(crate) in_ring: Vec<bool>,
    /// The ring's window, or zero when no layer lives in the ring.
    pub(crate) window: usize,
}

/// Why no cache layout holds a schedule, at the first offending layer.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RingPlanError {
    /// The layer uses chunked-local attention.
    Chunked {
        /// The offending layer.
        layer: usize,
    },
    /// The layer needs a ring window other than an earlier ring layer's.
    Conflict {
        /// The offending layer.
        layer: usize,
        /// Its window.
        window: usize,
        /// The window of the earlier ring layers.
        earlier: usize,
    },
}

#[cfg(verus_keep_ghost)]
verus! {

/// A layer lives in the ring exactly when it slides over `0 < W < capacity` keys.
pub(crate) open spec fn ring_member(reach: LayerReach, capacity: int) -> bool {
    match reach {
        LayerReach::Sliding(window) => 0 < window && window < capacity,
        _ => false,
    }
}

/// The window of a sliding layer, zero otherwise.
pub(crate) open spec fn reach_window(reach: LayerReach) -> int {
    match reach {
        LayerReach::Sliding(window) => window as int,
        _ => 0,
    }
}

/// Layers `0..end` are storable: none is chunked and all ring layers share one window.
pub(crate) open spec fn storable_prefix(reach: Seq<LayerReach>, end: int, capacity: int) -> bool {
    &&& forall|l: int| 0 <= l < end ==> #[trigger] reach[l] != LayerReach::Chunked
    &&& forall|i: int, j: int|
        0 <= i < end && 0 <= j < end && #[trigger] ring_member(reach[i], capacity)
            && #[trigger] ring_member(reach[j], capacity)
            ==> reach_window(reach[i]) == reach_window(reach[j])
}

} // verus!

/// Assigns every layer to full storage or the sliding ring, in model order.
///
/// A sliding layer whose window `W` is nonzero and below the capacity lives
/// in the ring; a wider window can never exclude a cached key, so such a
/// layer lives in full storage like a global one. One ring holds one
/// window. Proved: `Ok` exactly when no layer is chunked and every ring
/// layer has the same window, which is then the plan's window (zero exactly
/// when no layer lives in the ring), and each layer's flag is exactly its
/// ring membership; an error names the first offending layer, every earlier
/// layer being storable, and a conflict reports a genuinely different
/// window of an earlier ring layer.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    ensures
        plan.is_ok() == storable_prefix(reach@, reach.len() as int, capacity as int),
        match plan {
            Ok(plan) => {
                &&& plan.in_ring.len() == reach.len()
                &&& forall|l: int| 0 <= l < reach.len() ==>
                    #[trigger] plan.in_ring@[l] == ring_member(reach@[l], capacity as int)
                &&& forall|l: int| 0 <= l < reach.len() && #[trigger] ring_member(reach@[l], capacity as int)
                    ==> reach_window(reach@[l]) == plan.window
                &&& (plan.window > 0) == exists|l: int| 0 <= l < reach.len()
                    && #[trigger] ring_member(reach@[l], capacity as int)
            },
            Err(RingPlanError::Chunked { layer }) => {
                &&& layer < reach.len()
                &&& reach@[layer as int] == LayerReach::Chunked
                &&& storable_prefix(reach@, layer as int, capacity as int)
            },
            Err(RingPlanError::Conflict { layer, window, earlier }) => {
                &&& layer < reach.len()
                &&& reach@[layer as int] == LayerReach::Sliding(window)
                &&& ring_member(reach@[layer as int], capacity as int)
                &&& earlier != window
                &&& exists|l: int| 0 <= l < layer && #[trigger] ring_member(reach@[l], capacity as int)
                    && reach_window(reach@[l]) == earlier
                &&& storable_prefix(reach@, layer as int, capacity as int)
            },
        },
))]
pub(crate) fn plan_ring(reach: &[LayerReach], capacity: usize) -> Result<RingPlan, RingPlanError> {
    let mut in_ring: Vec<bool> = Vec::with_capacity(reach.len());
    let mut window: usize = 0;
    let mut layer: usize = 0;
    #[cfg(verus_keep_ghost)]
    proof_decl! { let ghost mut witness: int = 0; }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= reach.len(),
            in_ring.len() == layer,
            storable_prefix(reach@, layer as int, capacity as int),
            forall|l: int| 0 <= l < layer ==>
                #[trigger] in_ring@[l] == ring_member(reach@[l], capacity as int),
            forall|l: int| 0 <= l < layer && #[trigger] ring_member(reach@[l], capacity as int)
                ==> reach_window(reach@[l]) == window,
            window > 0 ==> 0 <= witness < layer && ring_member(reach@[witness], capacity as int),
            window == 0 ==> forall|l: int| 0 <= l < layer ==>
                !#[trigger] ring_member(reach@[l], capacity as int),
        decreases reach.len() - layer,
    ))]
    while layer < reach.len() {
        match reach[layer] {
            LayerReach::Chunked => {
                return Err(RingPlanError::Chunked { layer });
            }
            LayerReach::Global => in_ring.push(false),
            LayerReach::Sliding(width) => {
                if width > 0 && width < capacity {
                    if window != 0 && window != width {
                        // The earlier ring layer and this one disagree, so no layout exists.
                        #[cfg(verus_keep_ghost)]
                        proof! {
                            assert(ring_member(reach@[witness], capacity as int));
                            assert(ring_member(reach@[layer as int], capacity as int));
                            assert(reach_window(reach@[witness]) != reach_window(reach@[layer as int]));
                        }
                        return Err(RingPlanError::Conflict {
                            layer,
                            window: width,
                            earlier: window,
                        });
                    }
                    window = width;
                    #[cfg(verus_keep_ghost)]
                    proof! {
                        witness = layer as int;
                    }
                    in_ring.push(true);
                } else {
                    in_ring.push(false);
                }
            }
        }
        layer += 1;
    }
    Ok(RingPlan { in_ring, window })
}

/// Returns the cache position of every slot of one row's block.
///
/// The row appends `count` tokens at cache positions `start, start + 1, …`
/// from block slots `[column, column + count)`. Slot `t` inside that run gets
/// `start + (t - column)`; every other slot gets the row's fill position
/// (see the module documentation). Returns `None` when `start + count`
/// overflows, so every returned value is at most `start + count`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(positions =>
    requires
        column + count <= block,
    ensures
        match positions {
            Some(positions) => {
                &&& start + count <= usize::MAX
                &&& positions.len() == block
                &&& forall|t: int| column <= t < column + count ==>
                    #[trigger] positions[t] == start + (t - column)
                &&& forall|t: int| 0 <= t < block && !(column <= t < column + count) ==>
                    #[trigger] positions[t] == fill_position(start as int, count as int)
                &&& forall|t: int| 0 <= t < block ==> #[trigger] positions[t] <= start + count
            },
            None => start + count > usize::MAX,
        },
))]
pub(crate) fn slot_positions(
    start: usize,
    column: usize,
    count: usize,
    block: usize,
) -> Option<Vec<usize>> {
    let end = start.checked_add(count)?;
    let fill = if count > 0 {
        end - 1
    } else if start > 0 {
        start - 1
    } else {
        0
    };
    let mut positions: Vec<usize> = Vec::with_capacity(block);
    let mut slot: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            slot <= block,
            column + count <= block,
            end == start + count,
            fill == fill_position(start as int, count as int),
            positions.len() == slot,
            forall|t: int| column <= t < column + count && t < slot ==>
                #[trigger] positions[t] == start + (t - column),
            forall|t: int| 0 <= t < slot && !(column <= t < column + count) ==>
                #[trigger] positions[t] == fill,
            forall|t: int| 0 <= t < slot ==> #[trigger] positions[t] <= end,
        decreases block - slot,
    ))]
    while slot < block {
        if slot >= column && slot - column < count {
            positions.push(start + (slot - column));
        } else {
            positions.push(fill);
        }
        slot += 1;
    }
    Some(positions)
}

/// Returns the source column of every slot of one compacted row.
///
/// Compaction gathers a row's valid columns `c_0 < … < c_{n-1}` into slots
/// `0..n` of a right-padded block of width `block >= n`. Padding slots repeat
/// the first valid column (or column zero for an empty row), so the gather
/// stays in range and padding tokens are real, finite inputs.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sources =>
    requires
        columns.len() <= block,
    ensures
        sources.len() == block,
        forall|k: int| 0 <= k < columns.len() ==> #[trigger] sources[k] == columns[k],
        forall|k: int| columns.len() <= k < block ==>
            #[trigger] sources[k] == if columns.len() > 0 { columns[0] } else { 0usize },
))]
pub(crate) fn compaction_sources(columns: &[usize], block: usize) -> Vec<usize> {
    let pad = if columns.is_empty() { 0 } else { columns[0] };
    let mut sources: Vec<usize> = Vec::with_capacity(block);
    let mut slot: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            slot <= block,
            columns.len() <= block,
            pad == if columns.len() > 0 { columns[0] } else { 0usize },
            sources.len() == slot,
            forall|k: int| 0 <= k < slot && k < columns.len() ==>
                #[trigger] sources[k] == columns[k],
            forall|k: int| columns.len() <= k < slot ==> #[trigger] sources[k] == pad,
        decreases block - slot,
    ))]
    while slot < block {
        if slot < columns.len() {
            sources.push(columns[slot]);
        } else {
            sources.push(pad);
        }
        slot += 1;
    }
    sources
}

/// Returns, for each rectangular column, the compact slot that restores it.
///
/// Column `j = c_k` reads compact slot `k`; every other column reads slot
/// `block`, which the caller appends as an exact zero row. Both index
/// sequences increase, so one merge pass suffices.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sources =>
    requires
        sorted_unique_below(columns@, sequence as int),
        columns.len() <= block,
    ensures
        sources.len() == sequence,
        forall|k: int| 0 <= k < columns.len() ==> sources[columns[k] as int] == k,
        forall|j: int| 0 <= j < sequence ==> #[trigger] sources[j] <= block,
        forall|j: int| 0 <= j < sequence && sources[j] < block ==>
            sources[j] < columns.len() && columns[sources[j] as int] == j,
))]
pub(crate) fn scatter_sources(columns: &[usize], sequence: usize, block: usize) -> Vec<usize> {
    let count = columns.len();
    let mut sources: Vec<usize> = Vec::with_capacity(sequence);
    let mut next: usize = 0;
    let mut column: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            column <= sequence,
            next <= count,
            count == columns.len(),
            count <= block,
            sorted_unique_below(columns@, sequence as int),
            sources.len() == column,
            next < count ==> columns[next as int] >= column,
            forall|k: int| 0 <= k < next ==> columns[k] < column,
            forall|k: int| 0 <= k < next ==> sources[columns[k] as int] == k,
            forall|j: int| 0 <= j < column ==> #[trigger] sources[j] <= block,
            forall|j: int| 0 <= j < column && sources[j] < block ==>
                sources[j] < count && columns[sources[j] as int] == j,
        decreases sequence - column,
    ))]
    while column < sequence {
        if next < count && columns[next] == column {
            sources.push(next);
            // Strict increase moves the next unmatched column beyond `column`.
            #[cfg(verus_keep_ghost)]
            proof! {
                if next + 1 < count {
                    assert(columns[next as int] < columns[next + 1]);
                }
            }
            next += 1;
        } else {
            sources.push(block);
        }
        column += 1;
    }
    // Every column is below `sequence`, so the merge consumed all of them.
    #[cfg(verus_keep_ghost)]
    proof! {
        if next < count {
            assert(columns[next as int] < sequence);
        }
    }
    sources
}

/// Returns whether strictly increasing columns form one contiguous run.
///
/// `n` strictly increasing integers lie in `[first, last]`, which holds
/// `last - first + 1` integers, so they are consecutive exactly when
/// `last - first = n - 1`. The empty row is a (trivial) run. Comparing with
/// `n - 1` rather than adding one cannot overflow.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(run =>
    requires
        forall|i: int, j: int| 0 <= i < j < columns.len() ==> columns[i] < columns[j],
    ensures
        run == (columns.len() == 0
            || columns[columns.len() - 1] - columns[0] == columns.len() - 1),
))]
pub(crate) fn is_one_run(columns: &[usize]) -> bool {
    let count = columns.len();
    if count == 0 {
        return true;
    }
    // Strict increase orders the last column after (or at) the first.
    #[cfg(verus_keep_ghost)]
    proof! {
        if count > 1 {
            assert(columns[0] < columns[count - 1]);
        }
    }
    columns[count - 1] - columns[0] == count - 1
}

/// Returns, for each of `sequence` input columns, the block slot restoring it.
///
/// A row whose `n` valid columns `v_0 < … < v_{n-1}` occupy block slots
/// `[column, column + n)` restores column `v_k` from slot `column + k`;
/// every other column reads slot `block`, the zero slot appended to the
/// block. One gather with these indices puts every valid result back in place
/// and writes exact zeros at padding.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(slots =>
    requires
        sorted_unique_below(columns@, sequence as int),
        column + columns.len() <= block,
    ensures
        slots.len() == sequence,
        forall|k: int| 0 <= k < columns.len() ==> slots[columns[k] as int] == column + k,
        forall|j: int| 0 <= j < sequence ==> #[trigger] slots[j] <= block,
        forall|j: int| 0 <= j < sequence && slots[j] < block ==>
            column <= slots[j] && columns[slots[j] - column] == j,
))]
pub(crate) fn restore_slots(
    columns: &[usize],
    sequence: usize,
    block: usize,
    column: usize,
) -> Vec<usize> {
    let sources = scatter_sources(columns, sequence, block);
    let mut slots: Vec<usize> = Vec::with_capacity(sequence);
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= sequence,
            sources.len() == sequence,
            column + columns.len() <= block,
            slots.len() == index,
            forall|k: int| 0 <= k < columns.len() ==> sources[columns[k] as int] == k,
            forall|j: int| 0 <= j < sequence ==> #[trigger] sources[j] <= block,
            forall|j: int| 0 <= j < sequence && sources[j] < block ==>
                sources[j] < columns.len() && columns[sources[j] as int] == j,
            forall|j: int| 0 <= j < index ==>
                #[trigger] slots[j] == if sources[j] < block { column + sources[j] } else { block as int },
        decreases sequence - index,
    ))]
    while index < sequence {
        let source = sources[index];
        if source < block {
            slots.push(column + source);
        } else {
            slots.push(block);
        }
        index += 1;
    }
    // A valid column's compact index is below n, so its slot is below the block.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|k: int| 0 <= k < columns.len()
            implies slots[columns[k] as int] == column + k by {
            assert(columns[k] < sequence);
            assert(sources[columns[k] as int] == k);
        }
    }
    slots
}

// Rotary sets.

/// Returns, for each of `set_count` rotary sets, whether some layer rotates with it.
///
/// `layers[l]` is the set layer `l` rotates with, or `None` for a NoPE
/// layer. Proved: set `s` is used exactly when some layer names it, so a
/// call builds the tables of exactly the sets it needs; out-of-range set
/// indices mark nothing.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(used =>
    ensures
        used.len() == set_count,
        forall|s: int| 0 <= s < set_count ==> #[trigger] used@[s]
            == exists|l: int| 0 <= l < layers.len() && #[trigger] layers@[l] == Some(s as usize),
))]
pub(crate) fn used_rotary_sets(set_count: usize, layers: &[Option<usize>]) -> Vec<bool> {
    let mut used: Vec<bool> = Vec::with_capacity(set_count);
    let mut set: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            set <= set_count,
            used.len() == set,
            forall|s: int| 0 <= s < set ==> !#[trigger] used@[s],
        decreases set_count - set,
    ))]
    while set < set_count {
        used.push(false);
        set += 1;
    }
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= layers.len(),
            used.len() == set_count,
            forall|s: int| 0 <= s < set_count ==> #[trigger] used@[s]
                == exists|l: int| 0 <= l < layer && #[trigger] layers@[l] == Some(s as usize),
        decreases layers.len() - layer,
    ))]
    while layer < layers.len() {
        let named = match layers[layer] {
            Some(named) => named,
            None => set_count,
        };
        if named < set_count {
            used[named] = true;
        }
        // Only the named set gains a witness; every other set keeps its own.
        #[cfg(verus_keep_ghost)]
        proof! {
            assert forall|s: int| 0 <= s < set_count implies #[trigger] used@[s]
                == exists|l: int| 0 <= l < layer + 1 && #[trigger] layers@[l] == Some(s as usize) by {
                if layers@[layer as int] == Some(s as usize) {
                    assert(0 <= layer < layer + 1);
                } else if used@[s] {
                    let l = choose|l: int| 0 <= l < layer && layers@[l] == Some(s as usize);
                    assert(0 <= l < layer + 1);
                }
            }
        }
        layer += 1;
    }
    used
}

// The whole step.

/// One row's share of a batched step block.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verifier::allow(autoderive_clone_without_spec))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RowStep {
    /// `[block]` cache position of each slot, in Burn's signed index type.
    pub(crate) positions: Vec<i64>,
    /// `[block]` flags: the slot holds one of the row's new tokens.
    pub(crate) valid: Vec<bool>,
    /// `[sequence]` block slot restoring each input column (`block` is the zero slot).
    pub(crate) restore: Vec<usize>,
    /// `[block]` input column gathered into each slot; empty for a sliced block.
    pub(crate) sources: Vec<usize>,
    /// Slot of the row's last new token (slot zero for an inactive row).
    pub(crate) last: usize,
}

/// The integer layout of one batched step.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verifier::allow(autoderive_clone_without_spec))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StepPlan {
    /// First input column of a sliced block; `None` when interior padding forces a gather.
    pub(crate) offset: Option<usize>,
    /// Block width `Q`.
    pub(crate) block: usize,
    /// First block slot `c_b` of each row's new tokens.
    pub(crate) columns: Vec<usize>,
    /// Each row's slots.
    pub(crate) rows: Vec<RowStep>,
}

/// Why a batched step cannot be planned.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepPlanError {
    /// No row appends a token.
    NoActiveRow,
    /// A row's end position `L_b + n_b` overflows `usize`.
    PositionOverflow,
    /// A cache position exceeds `i64::MAX`, Burn's index range.
    PositionBeyondI64,
}

#[cfg(verus_keep_ghost)]
verus! {

/// Each later entry of a strictly increasing sequence exceeds an earlier one by the index gap.
pub proof fn lemma_increasing_gaps(s: Seq<usize>, i: int, j: int)
    requires
        forall|a: int, b: int| 0 <= a < b < s.len() ==> s[a] < s[b],
        0 <= i <= j < s.len(),
    ensures
        s[j] >= s[i] + (j - i),
    decreases j - i,
{
    if i < j {
        lemma_increasing_gaps(s, i, j - 1);
    }
}

/// Strictly increasing entries spanning `n - 1` are consecutive.
pub proof fn lemma_run_is_consecutive(s: Seq<usize>)
    requires
        forall|a: int, b: int| 0 <= a < b < s.len() ==> s[a] < s[b],
        s.len() > 0,
        s[s.len() - 1] - s[0] == s.len() - 1,
    ensures
        forall|k: int| 0 <= k < s.len() ==> #[trigger] s[k] == s[0] + k,
{
    assert forall|k: int| 0 <= k < s.len() implies #[trigger] s[k] == s[0] + k by {
        lemma_increasing_gaps(s, 0, k);
        lemma_increasing_gaps(s, k, s.len() - 1);
    }
}

/// A row's valid columns form one run (the empty row trivially).
pub open spec fn is_run(s: Seq<usize>) -> bool {
    s.len() == 0 || s[s.len() - 1] - s[0] == s.len() - 1
}

/// Every row's valid columns are strictly increasing and below `sequence`.
pub open spec fn rows_sorted_below(rows: Seq<Vec<usize>>, sequence: int) -> bool {
    forall|b: int| 0 <= b < rows.len() ==> #[trigger] sorted_unique_below(rows[b]@, sequence)
}

/// Cache position of slot `t` for a row appending `count` tokens at `start` from slot `column`.
pub open spec fn slot_position(start: int, column: int, count: int, t: int) -> int {
    if column <= t < column + count {
        start + (t - column)
    } else {
        fill_position(start, count)
    }
}

/// First block slot of a row: its offset into a sliced window, else zero.
pub open spec fn first_slot(columns: Seq<usize>, lo: int, sliced: bool) -> int {
    if sliced && columns.len() > 0 {
        columns[0] - lo
    } else {
        0
    }
}

/// A row appending `count` tokens at cursor `start` has representable positions.
///
/// Its end `start + count` fits `usize` and its largest position, the fill
/// position, fits `i64` (Burn's index range).
pub open spec fn row_fits(start: int, count: int) -> bool {
    &&& start + count <= usize::MAX
    &&& fill_position(start, count) <= i64::MAX
}

/// Rows `0..end` all have representable positions.
pub open spec fn rows_fit(starts: Seq<usize>, rows: Seq<Vec<usize>>, end: int) -> bool {
    forall|b: int| 0 <= b < end ==> #[trigger] row_fits(starts[b] as int, rows[b].len() as int)
}

/// Some row has a valid column.
pub open spec fn some_active(rows: Seq<Vec<usize>>) -> bool {
    exists|b: int| 0 <= b < rows.len() && #[trigger] rows[b].len() > 0
}

/// Row `b` is the first row whose positions are not representable.
pub open spec fn first_unfit(starts: Seq<usize>, rows: Seq<Vec<usize>>, b: int) -> bool {
    &&& 0 <= b < rows.len()
    &&& rows_fit(starts, rows, b)
    &&& !row_fits(starts[b] as int, rows[b].len() as int)
}

/// `step` is exactly the share of the block of a row with valid `columns`.
pub(crate) open spec fn row_step_matches(
    step: RowStep,
    columns: Seq<usize>,
    start: int,
    column: int,
    block: int,
    sequence: int,
    gathered: bool,
) -> bool {
    let count = columns.len() as int;
    &&& step.positions.len() == block
    &&& forall|t: int| 0 <= t < block ==>
        #[trigger] step.positions@[t] as int == slot_position(start, column, count, t)
    &&& step.valid.len() == block
    &&& forall|t: int| 0 <= t < block ==> #[trigger] step.valid@[t] == (column <= t < column + count)
    &&& step.restore.len() == sequence
    &&& forall|k: int| 0 <= k < count ==> step.restore@[columns[k] as int] == column + k
    &&& forall|j: int| 0 <= j < sequence ==> #[trigger] step.restore@[j] <= block
    &&& forall|j: int| 0 <= j < sequence && step.restore@[j] < block ==>
        column <= step.restore@[j] && columns[step.restore@[j] - column] == j
    &&& if gathered {
        &&& step.sources.len() == block
        &&& forall|k: int| 0 <= k < count ==> #[trigger] step.sources@[k] == columns[k]
        &&& forall|k: int| count <= k < block ==>
            #[trigger] step.sources@[k] == if count > 0 { columns[0] } else { 0usize }
    } else {
        step.sources.len() == 0
    }
    &&& step.last == if column + count > 0 { column + count - 1 } else { 0 }
    &&& step.last < block
}

} // verus!

/// Plans one row's slots of a step block of width `block`.
///
/// The row's `n` valid `columns` occupy block slots `[column, column + n)`
/// and append at cache position `start`. Proved: the positions, slot
/// flags, restoring gather, compaction gather (only when `gathered`), and
/// last-token slot are exactly those of `row_step_matches`. The result is
/// `Ok` exactly when the positions are representable (`row_fits`): it is
/// `PositionOverflow` exactly when `start + n` overflows `usize`, and
/// `PositionBeyondI64` exactly when it does not but the largest position,
/// the fill position, exceeds `i64::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(step =>
    requires
        sorted_unique_below(columns@, sequence as int),
        column + columns.len() <= block,
        block > 0,
    ensures
        match step {
            Ok(step) => {
                &&& row_fits(start as int, columns.len() as int)
                &&& row_step_matches(
                    step,
                    columns@,
                    start as int,
                    column as int,
                    block as int,
                    sequence as int,
                    gathered,
                )
            },
            Err(StepPlanError::PositionOverflow) => start + columns.len() > usize::MAX,
            Err(StepPlanError::PositionBeyondI64) => {
                &&& start + columns.len() <= usize::MAX
                &&& fill_position(start as int, columns.len() as int) > i64::MAX
            },
            Err(StepPlanError::NoActiveRow) => false,
        },
))]
pub(crate) fn plan_row_step(
    columns: &[usize],
    sequence: usize,
    start: usize,
    column: usize,
    block: usize,
    gathered: bool,
) -> Result<RowStep, StepPlanError> {
    let count = columns.len();
    let Some(cache) = slot_positions(start, column, count, block) else {
        return Err(StepPlanError::PositionOverflow);
    };
    let mut positions: Vec<i64> = Vec::with_capacity(block);
    let mut slot: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            slot <= block,
            cache.len() == block,
            column + count <= block,
            count == columns.len(),
            start + count <= usize::MAX,
            positions.len() == slot,
            forall|t: int| 0 <= t < block ==> #[trigger] cache@[t] <= start + count,
            forall|t: int| 0 <= t < block ==>
                #[trigger] cache@[t] as int == slot_position(start as int, column as int, count as int, t),
            forall|t: int| 0 <= t < slot ==> #[trigger] positions@[t] as int == cache@[t] as int,
        decreases block - slot,
    ))]
    while slot < block {
        match i64::try_from(cache[slot]) {
            Ok(position) => positions.push(position),
            Err(_) => {
                // No slot lies past the fill position, so it exceeds `i64::MAX` too.
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(cache@[slot as int] <= fill_position(start as int, count as int));
                }
                return Err(StepPlanError::PositionBeyondI64);
            }
        }
        slot += 1;
    }
    // The fill position is some slot's position: the run's last slot, or
    // every slot of an empty row.
    #[cfg(verus_keep_ghost)]
    proof! {
        let witness: int = if count > 0 { (column + count - 1) as int } else { 0 };
        assert(positions@[witness] as int == fill_position(start as int, count as int));
    }
    let mut valid: Vec<bool> = Vec::with_capacity(block);
    let mut slot: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            slot <= block,
            column + count <= block,
            valid.len() == slot,
            forall|t: int| 0 <= t < slot ==> #[trigger] valid@[t] == (column <= t < column + count),
        decreases block - slot,
    ))]
    while slot < block {
        valid.push(slot >= column && slot - column < count);
        slot += 1;
    }
    let restore = restore_slots(columns, sequence, block, column);
    let sources = if gathered {
        compaction_sources(columns, block)
    } else {
        Vec::new()
    };
    let last = (column + count).saturating_sub(1);
    Ok(RowStep {
        positions,
        valid,
        restore,
        sources,
        last,
    })
}

/// Plans a batched step of rows with valid `valid_columns` appending at cursors `starts`.
///
/// When every row's valid columns form one run, the block is the narrowest
/// input window `[lo, hi)` holding every valid column (one slice); otherwise
/// every row is compacted into a right-padded block as wide as the longest
/// row (one gather). Proved: `NoActiveRow` exactly when no row has a valid
/// column; otherwise the first row (in batch order) whose positions are not
/// representable determines the error (`PositionOverflow` when its end
/// overflows `usize`, else `PositionBeyondI64`), and the plan is `Ok`
/// exactly when every row's positions are representable. For a slice, every row's `k`-th valid column is input column
/// `lo + c_b + k` (so slots `[c_b, c_b + n_b)` hold exactly its tokens), the
/// window lies inside the input and is minimal (some row starts at `lo`,
/// some row ends at `hi`); for a gather, some row is not a run, every
/// `c_b = 0`, and the block is the longest row; in both, every row fits its
/// block (`c_b + n_b <= Q`, so the per-row plans' preconditions hold) and
/// each row's slots are exactly `row_step_matches`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    requires
        starts.len() == valid_columns.len(),
        rows_sorted_below(valid_columns@, sequence as int),
    ensures
        match plan {
            Err(StepPlanError::NoActiveRow) => !some_active(valid_columns@),
            Err(StepPlanError::PositionOverflow) => {
                &&& some_active(valid_columns@)
                &&& exists|b: int| #[trigger] first_unfit(starts@, valid_columns@, b)
                    && starts@[b] + valid_columns@[b].len() > usize::MAX
            },
            Err(StepPlanError::PositionBeyondI64) => {
                &&& some_active(valid_columns@)
                &&& exists|b: int| #[trigger] first_unfit(starts@, valid_columns@, b)
                    && starts@[b] + valid_columns@[b].len() <= usize::MAX
            },
            Ok(plan) => {
                &&& some_active(valid_columns@)
                &&& rows_fit(starts@, valid_columns@, valid_columns.len() as int)
                &&& plan.block > 0
                &&& plan.columns.len() == valid_columns.len()
                &&& plan.rows.len() == valid_columns.len()
                &&& forall|b: int| 0 <= b < valid_columns.len() ==>
                    plan.columns@[b] + #[trigger] valid_columns@[b].len() <= plan.block
                &&& forall|b: int| 0 <= b < valid_columns.len() ==> #[trigger] row_step_matches(
                    plan.rows@[b],
                    valid_columns@[b]@,
                    starts@[b] as int,
                    plan.columns@[b] as int,
                    plan.block as int,
                    sequence as int,
                    plan.offset.is_none(),
                )
                &&& match plan.offset {
                    Some(lo) => {
                        &&& forall|b: int| 0 <= b < valid_columns.len() ==>
                            #[trigger] is_run(valid_columns@[b]@)
                        &&& lo + plan.block <= sequence
                        &&& forall|b: int, k: int|
                            0 <= b < valid_columns.len() && 0 <= k < valid_columns@[b].len()
                            ==> #[trigger] valid_columns@[b]@[k] == lo + plan.columns@[b] + k
                        &&& forall|b: int| 0 <= b < valid_columns.len()
                            && #[trigger] valid_columns@[b].len() == 0 ==> plan.columns@[b] == 0
                        &&& exists|b: int| 0 <= b < valid_columns.len()
                            && #[trigger] valid_columns@[b].len() > 0 && plan.columns@[b] == 0
                        &&& exists|b: int| 0 <= b < valid_columns.len()
                            && plan.columns@[b] + #[trigger] valid_columns@[b].len() == plan.block
                    },
                    None => {
                        &&& exists|b: int| 0 <= b < valid_columns.len()
                            && !#[trigger] is_run(valid_columns@[b]@)
                        &&& forall|b: int| 0 <= b < valid_columns.len() ==> #[trigger] plan.columns@[b] == 0
                        &&& exists|b: int| 0 <= b < valid_columns.len()
                            && #[trigger] valid_columns@[b].len() == plan.block
                    },
                }
            },
        },
))]
pub(crate) fn plan_step(
    valid_columns: &[Vec<usize>],
    starts: &[usize],
    sequence: usize,
) -> Result<StepPlan, StepPlanError> {
    let batch = valid_columns.len();
    // Pass 1: the common window, whether every row is a run, and the longest row.
    let mut lo: usize = usize::MAX;
    let mut hi: usize = 0;
    let mut active = false;
    let mut contiguous = true;
    let mut widest: usize = 0;
    let mut row: usize = 0;
    #[cfg(verus_keep_ghost)]
    proof_decl! {
        let ghost mut lo_row: int = 0;
        let ghost mut hi_row: int = 0;
        let ghost mut wide_row: int = 0;
        let ghost mut broken_row: int = 0;
    }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= batch,
            batch == valid_columns.len(),
            rows_sorted_below(valid_columns@, sequence as int),
            active == exists|b: int| 0 <= b < row && #[trigger] valid_columns@[b].len() > 0,
            active ==> lo < hi && hi <= sequence,
            forall|b: int| 0 <= b < row && #[trigger] valid_columns@[b].len() > 0 ==> {
                &&& lo <= valid_columns@[b]@[0]
                &&& valid_columns@[b]@[valid_columns@[b].len() - 1] + 1 <= hi
            },
            active ==> 0 <= lo_row < row && valid_columns@[lo_row].len() > 0
                && valid_columns@[lo_row]@[0] == lo,
            active ==> 0 <= hi_row < row && valid_columns@[hi_row].len() > 0
                && valid_columns@[hi_row]@[valid_columns@[hi_row].len() - 1] + 1 == hi,
            contiguous ==> forall|b: int| 0 <= b < row ==> #[trigger] is_run(valid_columns@[b]@),
            !contiguous ==> 0 <= broken_row < row && !is_run(valid_columns@[broken_row]@),
            forall|b: int| 0 <= b < row ==> #[trigger] valid_columns@[b].len() <= widest,
            widest > 0 ==> 0 <= wide_row < row && valid_columns@[wide_row].len() == widest,
        decreases batch - row,
    ))]
    while row < batch {
        let columns = valid_columns[row].as_slice();
        let count = columns.len();
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(sorted_unique_below(valid_columns@[row as int]@, sequence as int));
        }
        if count > 0 {
            // Strict increase orders the row's last column after its first.
            #[cfg(verus_keep_ghost)]
            proof! {
                if count > 1 {
                    assert(columns@[0] < columns@[count - 1]);
                }
            }
            if !active || columns[0] < lo {
                lo = columns[0];
                #[cfg(verus_keep_ghost)]
                proof! {
                    lo_row = row as int;
                }
            }
            if !active || columns[count - 1] + 1 > hi {
                hi = columns[count - 1] + 1;
                #[cfg(verus_keep_ghost)]
                proof! {
                    hi_row = row as int;
                }
            }
            active = true;
        }
        if !is_one_run(columns) {
            contiguous = false;
            #[cfg(verus_keep_ghost)]
            proof! {
                broken_row = row as int;
            }
        }
        if count > widest {
            widest = count;
            #[cfg(verus_keep_ghost)]
            proof! {
                wide_row = row as int;
            }
        }
        row += 1;
    }
    if !active {
        return Err(StepPlanError::NoActiveRow);
    }
    let block = if contiguous { hi - lo } else { widest };
    // Every row fits the block: a run ends by `hi`, and no row is longer than the widest.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(valid_columns@[lo_row].len() <= widest);
        assert forall|b: int| 0 <= b < batch implies
            #[trigger] first_slot(valid_columns@[b]@, lo as int, contiguous)
                + valid_columns@[b].len() <= block by {
            if contiguous && valid_columns@[b].len() > 0 {
                assert(is_run(valid_columns@[b]@));
            }
        }
    }
    let gathered = !contiguous;
    let mut columns_out: Vec<usize> = Vec::with_capacity(batch);
    let mut rows_out: Vec<RowStep> = Vec::with_capacity(batch);
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= batch,
            batch == valid_columns.len(),
            starts.len() == batch,
            block > 0,
            gathered == !contiguous,
            some_active(valid_columns@),
            rows_fit(starts@, valid_columns@, row as int),
            rows_sorted_below(valid_columns@, sequence as int),
            contiguous ==> forall|b: int| 0 <= b < batch ==> #[trigger] is_run(valid_columns@[b]@),
            forall|b: int| 0 <= b < batch && #[trigger] valid_columns@[b].len() > 0 ==>
                lo <= valid_columns@[b]@[0],
            forall|b: int| 0 <= b < batch ==>
                #[trigger] first_slot(valid_columns@[b]@, lo as int, contiguous)
                    + valid_columns@[b].len() <= block,
            columns_out.len() == row,
            rows_out.len() == row,
            forall|b: int| 0 <= b < row ==>
                #[trigger] columns_out@[b] as int == first_slot(valid_columns@[b]@, lo as int, contiguous),
            forall|b: int| 0 <= b < row ==> #[trigger] row_step_matches(
                rows_out@[b],
                valid_columns@[b]@,
                starts@[b] as int,
                columns_out@[b] as int,
                block as int,
                sequence as int,
                gathered,
            ),
        decreases batch - row,
    ))]
    while row < batch {
        let columns = valid_columns[row].as_slice();
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(columns@ == valid_columns@[row as int]@);
            assert(sorted_unique_below(valid_columns@[row as int]@, sequence as int));
            assert(first_slot(valid_columns@[row as int]@, lo as int, contiguous)
                + valid_columns@[row as int].len() <= block);
        }
        let column = if contiguous && !columns.is_empty() {
            columns[0] - lo
        } else {
            0
        };
        let step = match plan_row_step(columns, sequence, starts[row], column, block, gathered) {
            Ok(step) => step,
            Err(error) => {
                // Every earlier row fits, so this row is the first that does not.
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(first_unfit(starts@, valid_columns@, row as int));
                }
                return Err(error);
            }
        };
        columns_out.push(column);
        rows_out.push(step);
        row += 1;
    }
    // A run's k-th column is its first plus k, which places it at slot c_b + k.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|b: int, k: int|
            contiguous && 0 <= b < batch && 0 <= k < valid_columns@[b].len()
            implies #[trigger] valid_columns@[b]@[k] == lo + columns_out@[b] + k by {
            assert(is_run(valid_columns@[b]@));
            assert(sorted_unique_below(valid_columns@[b]@, sequence as int));
            lemma_run_is_consecutive(valid_columns@[b]@);
        }
    }
    if contiguous {
        // The rows attaining `lo` and `hi` start and end the sliced window.
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(columns_out@[lo_row] == 0);
            assert(is_run(valid_columns@[hi_row]@));
            assert(columns_out@[hi_row] + valid_columns@[hi_row].len() == block);
        }
        Ok(StepPlan {
            offset: Some(lo),
            block,
            columns: columns_out,
            rows: rows_out,
        })
    } else {
        // The widest row sets the block, and some row is not a run.
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(valid_columns@[wide_row].len() == block);
            assert(!is_run(valid_columns@[broken_row]@));
        }
        Ok(StepPlan {
            offset: None,
            block,
            columns: columns_out,
            rows: rows_out,
        })
    }
}

// Checkpoint extents.

/// Returns the first used set whose cached-context limit is the smallest.
///
/// `limits[s]` is set `s`'s original context when its frequencies change
/// beyond it (dynamic NTK, LongRoPE), `None` otherwise; `used[s]` says
/// whether some layer rotates with set `s`; sets beyond either list are
/// ignored. Proved: `Some(s)` names a used set with a limit that no used
/// set's limit undercuts, and every earlier used set with a limit has a
/// strictly larger one (the first minimum, as `Iterator::min_by_key`
/// picks); `None` exactly when no used set has a limit.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(tightest =>
    ensures
        match tightest {
            Some(set) => {
                &&& set < limits.len() && set < used.len()
                &&& used@[set as int]
                &&& limits@[set as int] is Some
                &&& forall|s: int| 0 <= s < limits.len() && s < used.len() && #[trigger] used@[s]
                    && limits@[s] is Some ==> limits@[set as int].unwrap() <= limits@[s].unwrap()
                &&& forall|s: int| 0 <= s < set && #[trigger] used@[s] && limits@[s] is Some
                    ==> limits@[set as int].unwrap() < limits@[s].unwrap()
            },
            None => forall|s: int| 0 <= s < limits.len() && s < used.len() && #[trigger] used@[s]
                ==> limits@[s] is None,
        },
))]
pub(crate) fn tightest_limit(limits: &[Option<usize>], used: &[bool]) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut set: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            set <= limits.len(),
            match best {
                Some(chosen) => {
                    &&& chosen < set && chosen < used.len()
                    &&& used@[chosen as int]
                    &&& limits@[chosen as int] is Some
                    &&& forall|s: int| 0 <= s < set && s < used.len() && #[trigger] used@[s]
                        && limits@[s] is Some ==> limits@[chosen as int].unwrap() <= limits@[s].unwrap()
                    &&& forall|s: int| 0 <= s < chosen && #[trigger] used@[s] && limits@[s] is Some
                        ==> limits@[chosen as int].unwrap() < limits@[s].unwrap()
                },
                None => forall|s: int| 0 <= s < set && s < used.len() && #[trigger] used@[s]
                    ==> limits@[s] is None,
            },
        decreases limits.len() - set,
    ))]
    while set < limits.len() && set < used.len() {
        if let (true, Some(limit)) = (used[set], limits[set]) {
            let tighter = match best {
                Some(chosen) => match limits[chosen] {
                    Some(current) => limit < current,
                    None => true,
                },
                None => true,
            };
            if tighter {
                best = Some(set);
            }
        }
        set += 1;
    }
    best
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
