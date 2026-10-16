//! Dependency-free gpt-oss planning, verified in place by Verus.
//!
//! Everything here is integer bookkeeping consumed by the configuration,
//! the loader, and the decode cache:
//!
//! * `default_layer_slides` and `default_schedule`: the reference's default
//!   attention schedule, per layer and for the whole model;
//! * the cache layout: `ring_assignment` (which sliding layers share the
//!   ring), `cache_slots` (the numbering of layers inside the two KV cache
//!   groups, full storage and sliding ring), `group_indices` (the bank index
//!   of each allocated group), and `truncation_leaves_bytes` (whether a
//!   truncation keeps the state's clean tails);
//! * `fused_expert_elements`: the overflow check of the stacked expert
//!   projections; `checkpoint_extent`/`projection_extent`: the checked
//!   `u64` extents of the checkpoint contract; `derived_head_dim`: the head
//!   width of a config that omits it; `widest_context`: the rotary context a
//!   cached step evaluates;
//! * `interleaved_offset`: where the loader finds expert `e`'s gate and up
//!   weights inside the stacked, column-interleaved `gate_up_proj`, and the
//!   byte-level splitters built on it: `stacked_layout` (one expert's and
//!   the whole tensor's bytes), `expert_slab` (one expert's contiguous
//!   slab), and `deinterleave` (one expert's gate or up half).
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

/// Element index of output `column` of `row` of `expert` in the interleaved `[E, rows, 2·width]` tensor.
///
/// Gate outputs (`up == false`) are the even columns `2j`, up outputs the odd columns `2j + 1`.
pub open spec fn interleaved_index(expert: int, rows: int, row: int, width: int, column: int, up: bool) -> int {
    (expert * rows + row) * (2 * width) + 2 * column + if up { 1int } else { 0int }
}

/// Source byte of output byte `t` of one expert's de-interleaved `[rows, width]` half.
///
/// Byte `t` is byte `t mod element` of output value `v = ⌊t / element⌋`,
/// which is output `v mod width` of row `⌊v / width⌋`.
pub open spec fn deinterleaved_source(
    t: int,
    expert: int,
    rows: int,
    width: int,
    element: int,
    up: bool,
) -> int {
    let value = t / element;
    interleaved_index(expert, rows, value / width, width, value % width, up) * element + t % element
}

/// Locates output byte `(row · width + column) · element + k` of a de-interleaved half.
pub proof fn lemma_deinterleaved_source(
    row: int,
    column: int,
    k: int,
    expert: int,
    rows: int,
    width: int,
    element: int,
    up: bool,
)
    requires
        0 <= row,
        0 <= column < width,
        0 <= k < element,
    ensures
        deinterleaved_source((row * width + column) * element + k, expert, rows, width, element, up)
            == interleaved_index(expert, rows, row, width, column, up) * element + k,
{
    let value = row * width + column;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(value * element + k, element, value, k);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(value, width, row, column);
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

/// Returns the reference's default sliding flag of each of `layers` layers.
///
/// Proved: layer `l` slides exactly when `(l + 1) mod 2 != 0` (the even
/// layers), consecutive layers alternate, and exactly `⌊layers / 2⌋` layers
/// are full-attention layers.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(flags =>
    ensures
        flags.len() == layers,
        forall|l: int| 0 <= l < layers ==> #[trigger] flags@[l] == ((l + 1) % 2 != 0),
        forall|l: int| 0 <= l && l + 1 < layers ==> #[trigger] flags@[l] != flags@[l + 1],
        count_equal(flags@, layers as int, false) == layers / 2,
))]
pub(crate) fn default_schedule(layers: usize) -> Vec<bool> {
    let mut flags: Vec<bool> = Vec::with_capacity(layers);
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= layers,
            flags.len() == layer,
            forall|l: int| 0 <= l < layer ==> #[trigger] flags@[l] == ((l + 1) % 2 != 0),
            count_equal(flags@, layer as int, false) == layer / 2,
        decreases layers - layer,
    ))]
    while layer < layers {
        #[cfg(verus_keep_ghost)]
        proof_decl! { let ghost before = flags@; }
        flags.push(default_layer_slides(layer));
        // A full layer completes a pair, which steps the halved count.
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_prefix_stable(before, flags@, layer as int, false);
        }
        layer += 1;
    }
    flags
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

/// The sliding-ring assignment of gpt-oss's cache layout.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verifier::allow(autoderive_clone_without_spec))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RingAssignment {
    /// Whether each layer lives in the sliding ring (else in full storage).
    pub(crate) in_ring: Vec<bool>,
    /// The ring's window, or zero when the ring is unused.
    pub(crate) window: usize,
}

/// Assigns the sliding layers to the ring when their shared window is below the capacity.
///
/// `sliding[l]` says whether layer `l` slides; `window` is the window every
/// sliding layer shares. A window of at least the capacity can never
/// exclude a cached key, so its layers live in full storage. Proved: the
/// ring's window is `W` exactly when `0 < W < capacity` (zero otherwise),
/// and then the ring holds exactly the sliding layers; otherwise it holds
/// none.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    ensures
        plan.window == match window {
            Some(width) => if 0 < width && width < capacity { width } else { 0usize },
            None => 0usize,
        },
        plan.in_ring.len() == sliding.len(),
        forall|l: int| 0 <= l < sliding.len() ==>
            #[trigger] plan.in_ring@[l] == (plan.window > 0 && sliding@[l]),
))]
pub(crate) fn ring_assignment(
    sliding: &[bool],
    window: Option<usize>,
    capacity: usize,
) -> RingAssignment {
    let ring_window = match window {
        Some(width) if width > 0 && width < capacity => width,
        _ => 0,
    };
    let mut in_ring: Vec<bool> = Vec::with_capacity(sliding.len());
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= sliding.len(),
            in_ring.len() == layer,
            forall|l: int| 0 <= l < layer ==>
                #[trigger] in_ring@[l] == (ring_window > 0 && sliding@[l]),
        decreases sliding.len() - layer,
    ))]
    while layer < sliding.len() {
        in_ring.push(ring_window > 0 && sliding[layer]);
        layer += 1;
    }
    RingAssignment {
        in_ring,
        window: ring_window,
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
/// `2j + 1`. The offset lies inside the tensor, and `interleaved_index` is a
/// bijection onto its elements (`lemma_interleaved_injective`,
/// `lemma_interleaved_surjective`), so the two halves of every expert
/// partition the stacked weights.
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

// Stacked expert tensors.

/// Returns the element count `2f · d · E` of the stacked `gate_up_proj`, or `None`.
///
/// Proved: `Some(2f · d · E)` exactly when the products `2f`, `2f · d`, and
/// `2f · d · E` (formed in that order) all fit `usize`; `None` names an
/// overflowing one. Validation rejects such a configuration, which also
/// bounds every interleaved offset of the loader.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(elements =>
    ensures
        match elements {
            Some(count) => {
                &&& intermediate * 2 <= usize::MAX
                &&& intermediate * 2 * hidden <= usize::MAX
                &&& count == intermediate * 2 * hidden * experts
            },
            None => intermediate * 2 > usize::MAX
                || intermediate * 2 * hidden > usize::MAX
                || intermediate * 2 * hidden * experts > usize::MAX,
        },
))]
pub(crate) fn fused_expert_elements(
    intermediate: usize,
    hidden: usize,
    experts: usize,
) -> Option<usize> {
    intermediate
        .checked_mul(2)?
        .checked_mul(hidden)?
        .checked_mul(experts)
}

/// Returns `(span, total)`: one expert's bytes and the stacked tensor's bytes.
///
/// A stacked `[E, rows, columns]` tensor of `element`-byte scalars holds
/// `E` contiguous slabs of `span = rows · columns · element` bytes. Proved:
/// `Some` exactly when `rows · columns`, `span`, and `E · span` fit
/// `usize`, with those exact values.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(layout =>
    ensures
        match layout {
            Some((span, total)) => {
                &&& span == rows * columns * element
                &&& total == experts * span
            },
            None => rows * columns > usize::MAX
                || rows * columns * element > usize::MAX
                || experts * (rows * columns * element) > usize::MAX,
        },
))]
pub(crate) fn stacked_layout(
    experts: usize,
    rows: usize,
    columns: usize,
    element: usize,
) -> Option<(usize, usize)> {
    let span = rows.checked_mul(columns)?.checked_mul(element)?;
    let total = experts.checked_mul(span)?;
    Some((span, total))
}

/// Copies expert `expert`'s slab out of a stacked byte buffer.
///
/// Proved: exactly bytes `[expert · span, (expert + 1) · span)`, which lie
/// inside the buffer.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(slab =>
    requires
        expert < experts,
        bytes.len() == experts * span,
    ensures
        slab@ == bytes@.subrange(expert * span, (expert + 1) * span),
))]
// `experts` appears only in the contract: it bounds the slab Verus proves.
#[allow(unused_variables)]
pub(crate) fn expert_slab(bytes: &[u8], experts: usize, span: usize, expert: usize) -> Vec<u8> {
    #[cfg(verus_keep_ghost)]
    proof! {
        assert((expert + 1) * span <= experts * span) by (nonlinear_arith)
            requires expert + 1 <= experts, span >= 0;
        assert(expert * span + span == (expert + 1) * span) by (nonlinear_arith);
    }
    let start = expert * span;
    let mut slab: Vec<u8> = Vec::with_capacity(span);
    slab.extend_from_slice(&bytes[start..start + span]);
    slab
}

/// Copies `half`'s `[rows, width]` outputs of expert `expert` out of the interleaved `gate_up_proj`.
///
/// `bytes` is the stacked `[experts, rows, 2 · width]` tensor of
/// `element`-byte scalars (see the module documentation). Proved: the
/// result has `rows · width · element` bytes and output byte `t` (byte
/// `t mod element` of output `v mod width` of row `⌊v / width⌋`, where
/// `v = ⌊t / element⌋`) is that byte of column `2j` (gate) or `2j + 1` (up)
/// of the stacked row, so the split is exactly the reference's
/// `gate_up[e, :, ::2]` and `gate_up[e, :, 1::2]`, and every read is in bounds.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(out =>
    requires
        expert < experts,
        bytes.len() == experts * (rows * (2 * width) * element),
    ensures
        out.len() == rows * width * element,
        forall|t: int| 0 <= t < out.len() ==> 0 <= #[trigger] deinterleaved_source(
            t,
            expert as int,
            rows as int,
            width as int,
            element as int,
            is_up(half),
        ) < bytes.len(),
        forall|t: int| 0 <= t < out.len() ==> #[trigger] out@[t] == bytes@[deinterleaved_source(
            t,
            expert as int,
            rows as int,
            width as int,
            element as int,
            is_up(half),
        )],
))]
pub(crate) fn deinterleave(
    bytes: &[u8],
    experts: usize,
    rows: usize,
    width: usize,
    element: usize,
    expert: usize,
    half: GluHalf,
) -> Vec<u8> {
    if element == 0 || width == 0 {
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(rows * width * element == 0) by (nonlinear_arith)
                requires element == 0 || width == 0;
        }
        return Vec::new();
    }
    #[cfg(verus_keep_ghost)]
    proof_decl! { let ghost up: bool = is_up(half); }
    // The stacked tensor fits `usize`, and so does one half of one expert.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(experts * (rows * (2 * width) * element) == experts * rows * (2 * width) * element)
            by (nonlinear_arith);
        assert(experts * rows * (2 * width) <= experts * rows * (2 * width) * element)
            by (nonlinear_arith) requires element >= 1, experts >= 0, rows >= 0, width >= 0;
        assert(rows * width * element <= experts * (rows * (2 * width) * element))
            by (nonlinear_arith) requires experts >= 1, width >= 1, rows >= 0, element >= 0;
        assert(rows * width <= rows * width * element)
            by (nonlinear_arith) requires element >= 1, rows >= 0, width >= 0;
    }
    let mut out: Vec<u8> = Vec::with_capacity(rows * width * element);
    let mut row: usize = 0;
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(row * width * element == 0) by (nonlinear_arith) requires row == 0;
    }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            element > 0,
            width > 0,
            row <= rows,
            expert < experts,
            up == is_up(half),
            bytes.len() == experts * (rows * (2 * width) * element),
            experts * rows * (2 * width) <= usize::MAX,
            bytes.len() == experts * rows * (2 * width) * element,
            out.len() == row * width * element,
            forall|t: int| 0 <= t < out.len() ==> 0 <= #[trigger] deinterleaved_source(
                t,
                expert as int,
                rows as int,
                width as int,
                element as int,
                up,
            ) < bytes.len(),
            forall|t: int| 0 <= t < out.len() ==> #[trigger] out@[t] == bytes@[deinterleaved_source(
                t,
                expert as int,
                rows as int,
                width as int,
                element as int,
                up,
            )],
        decreases rows - row,
    ))]
    while row < rows {
        // The row's last pair ends inside the expert's slab, so the scaled start fits.
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_row_start(expert as int, experts as int, rows as int, row as int, width as int, element as int, up);
        }
        let start = interleaved_offset(experts, rows, width, expert, row, 0, half) * element;
        let mut column: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                element > 0,
                width > 0,
                row < rows,
                column <= width,
                expert < experts,
                up == is_up(half),
                bytes.len() == experts * rows * (2 * width) * element,
                start == interleaved_index(expert as int, rows as int, row as int, width as int, 0, up) * element,
                out.len() == (row * width + column) * element,
                forall|t: int| 0 <= t < out.len() ==> 0 <= #[trigger] deinterleaved_source(
                    t,
                    expert as int,
                    rows as int,
                    width as int,
                    element as int,
                    up,
                ) < bytes.len(),
                forall|t: int| 0 <= t < out.len() ==> #[trigger] out@[t] == bytes@[deinterleaved_source(
                    t,
                    expert as int,
                    rows as int,
                    width as int,
                    element as int,
                    up,
                )],
            decreases width - column,
        ))]
        while column < width {
            #[cfg(verus_keep_ghost)]
            proof! {
                lemma_pair_in_bounds(expert as int, experts as int, rows as int, row as int, width as int, column as int, element as int, up);
                assert(2 * column <= 2 * column * element)
                    by (nonlinear_arith) requires element >= 1, column >= 0;
                assert(0 <= 2 * column * element)
                    by (nonlinear_arith) requires element >= 1, column >= 0;
            }
            let source = start + 2 * column * element;
            #[cfg(verus_keep_ghost)]
            proof_decl! { let ghost before = out@; }
            out.extend_from_slice(&bytes[source..source + element]);
            // Each copied byte is its output position's source byte.
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(before.len() == (row * width + column) * element);
                assert forall|t: int| 0 <= t < out.len() implies 0 <= #[trigger] deinterleaved_source(
                    t,
                    expert as int,
                    rows as int,
                    width as int,
                    element as int,
                    up,
                ) < bytes.len() by {
                    if t >= before.len() {
                        let k = t - before.len();
                        lemma_deinterleaved_source(row as int, column as int, k, expert as int, rows as int, width as int, element as int, up);
                    }
                }
                assert forall|t: int| 0 <= t < out.len() implies #[trigger] out@[t] == bytes@[deinterleaved_source(
                    t,
                    expert as int,
                    rows as int,
                    width as int,
                    element as int,
                    up,
                )] by {
                    if t >= before.len() {
                        let k = t - before.len();
                        lemma_deinterleaved_source(row as int, column as int, k, expert as int, rows as int, width as int, element as int, up);
                        assert(out@[t] == bytes@[source + k]);
                    } else {
                        assert(out@[t] == before[t]);
                    }
                }
                assert((row * width + column) * element + element == (row * width + (column + 1)) * element)
                    by (nonlinear_arith);
            }
            column += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((row * width + width) * element == (row + 1) * width * element) by (nonlinear_arith);
        }
        row += 1;
    }
    out
}

#[cfg(verus_keep_ghost)]
verus! {

/// Distinct in-range coordinates have distinct interleaved indices.
///
/// With `o = (e·R + i)·2W + 2j + h` and `0 <= 2j + h < 2W`, the quotient
/// and remainder by `2W` recover `e·R + i` and `2j + h`, which in turn
/// recover `(e, i)` and `(j, h)`.
pub proof fn lemma_interleaved_injective(
    rows: int,
    width: int,
    e1: int,
    i1: int,
    j1: int,
    up1: bool,
    e2: int,
    i2: int,
    j2: int,
    up2: bool,
)
    requires
        rows > 0,
        width > 0,
        0 <= e1,
        0 <= i1 < rows,
        0 <= j1 < width,
        0 <= e2,
        0 <= i2 < rows,
        0 <= j2 < width,
        interleaved_index(e1, rows, i1, width, j1, up1) == interleaved_index(e2, rows, i2, width, j2, up2),
    ensures
        e1 == e2 && i1 == i2 && j1 == j2 && up1 == up2,
{
    let pair = 2 * width;
    let r1 = 2 * j1 + if up1 { 1int } else { 0int };
    let r2 = 2 * j2 + if up2 { 1int } else { 0int };
    let o = interleaved_index(e1, rows, i1, width, j1, up1);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(o, pair, e1 * rows + i1, r1);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(o, pair, e2 * rows + i2, r2);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(e1 * rows + i1, rows, e1, i1);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(e2 * rows + i2, rows, e2, i2);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(r1, 2, j1, if up1 { 1int } else { 0int });
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(r2, 2, j2, if up2 { 1int } else { 0int });
}

/// Every element of the stacked `[E, R, 2W]` tensor is some coordinate's interleaved index.
///
/// Together with [`lemma_interleaved_injective`], the gate and up halves of
/// all experts partition the stacked tensor: the loader installs every
/// stacked weight exactly once.
pub proof fn lemma_interleaved_surjective(experts: int, rows: int, width: int, o: int)
    requires
        rows > 0,
        width > 0,
        0 <= o < experts * rows * (2 * width),
    ensures
        exists|e: int, i: int, j: int, up: bool|
            0 <= e < experts && 0 <= i < rows && 0 <= j < width
                && #[trigger] interleaved_index(e, rows, i, width, j, up) == o,
{
    let pair = 2 * width;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(o, pair);
    let q = o / pair;
    let r = o % pair;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(q, rows);
    let e = q / rows;
    let i = q % rows;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(r, 2);
    let j = r / 2;
    let up = r % 2 == 1;
    assert(0 <= q) by {
        vstd::arithmetic::div_mod::lemma_div_pos_is_pos(o, pair);
    }
    assert(q < experts * rows) by (nonlinear_arith)
        requires
            o == pair * q + r,
            0 <= r,
            o < experts * rows * pair,
            pair > 0,
    ;
    assert(0 <= e) by {
        vstd::arithmetic::div_mod::lemma_div_pos_is_pos(q, rows);
    }
    assert(e < experts) by (nonlinear_arith)
        requires
            q == rows * e + i,
            0 <= i,
            q < experts * rows,
            rows > 0,
    ;
    assert(0 <= j < width);
    assert(interleaved_index(e, rows, i, width, j, up) == o) by {
        assert(r == 2 * j + if up { 1int } else { 0int });
        assert(q == rows * e + i);
        assert(o == pair * q + r);
        assert((e * rows + i) * pair == pair * q) by (nonlinear_arith)
            requires q == rows * e + i;
    }
}

/// Whether `half` is the up projection (odd columns).
pub(crate) open spec fn is_up(half: GluHalf) -> bool {
    half == GluHalf::Up
}

/// The interleaved index of row `row`'s last pair stays below the stacked tensor's size.
pub open spec fn interleaved_offset_bound(expert: int, experts: int, rows: int, row: int, width: int) -> bool {
    (expert * rows + row) * (2 * width) + 2 * width <= experts * rows * (2 * width)
}

/// Scaling a row's first offset by `element` stays inside the byte buffer.
pub proof fn lemma_row_start(expert: int, experts: int, rows: int, row: int, width: int, element: int, up: bool)
    requires
        0 <= expert < experts,
        0 <= row < rows,
        width > 0,
        element > 0,
    ensures
        interleaved_offset_bound(expert, experts, rows, row, width),
        0 <= interleaved_index(expert, rows, row, width, 0, up) * element,
        interleaved_index(expert, rows, row, width, 0, up) * element + element
            <= experts * rows * (2 * width) * element,
{
    let pair = 2 * width;
    assert(expert * rows + row < experts * rows) by (nonlinear_arith)
        requires expert < experts, row < rows, 0 <= expert, 0 <= row;
    assert((expert * rows + row) * pair + pair <= experts * rows * pair) by (nonlinear_arith)
        requires expert * rows + row < experts * rows, pair >= 0;
    let index = interleaved_index(expert, rows, row, width, 0, up);
    assert(0 <= expert * rows) by (nonlinear_arith) requires 0 <= expert, 0 <= rows;
    assert(0 <= (expert * rows + row) * pair) by (nonlinear_arith)
        requires 0 <= expert * rows + row, pair >= 0;
    assert(index + 1 <= experts * rows * pair);
    assert(index * element + element <= experts * rows * pair * element) by (nonlinear_arith)
        requires index + 1 <= experts * rows * pair, element > 0;
    assert(0 <= index * element) by (nonlinear_arith) requires 0 <= index, element > 0;
}

/// Output `column`'s scalar of row `row` lies inside the buffer, `2·column·element` past the row start.
pub proof fn lemma_pair_in_bounds(
    expert: int,
    experts: int,
    rows: int,
    row: int,
    width: int,
    column: int,
    element: int,
    up: bool,
)
    requires
        0 <= expert < experts,
        0 <= row < rows,
        0 <= column < width,
        element > 0,
    ensures
        interleaved_index(expert, rows, row, width, column, up) * element
            == interleaved_index(expert, rows, row, width, 0, up) * element + 2 * column * element,
        interleaved_index(expert, rows, row, width, column, up) * element + element
            <= experts * rows * (2 * width) * element,
        0 <= interleaved_index(expert, rows, row, width, 0, up) * element,
{
    let pair = 2 * width;
    let base = interleaved_index(expert, rows, row, width, 0, up);
    let index = interleaved_index(expert, rows, row, width, column, up);
    assert(0 <= expert * rows) by (nonlinear_arith) requires 0 <= expert, 0 <= rows;
    assert(0 <= (expert * rows + row) * pair) by (nonlinear_arith)
        requires 0 <= expert * rows + row, pair >= 0;
    assert(index == base + 2 * column);
    assert(index * element == base * element + 2 * column * element) by (nonlinear_arith)
        requires index == base + 2 * column;
    assert(expert * rows + row < experts * rows) by (nonlinear_arith)
        requires expert < experts, row < rows, 0 <= expert, 0 <= row;
    assert((expert * rows + row) * pair + pair <= experts * rows * pair) by (nonlinear_arith)
        requires expert * rows + row < experts * rows, pair >= 0;
    assert(index + 1 <= experts * rows * pair);
    assert(index * element + element <= experts * rows * pair * element) by (nonlinear_arith)
        requires index + 1 <= experts * rows * pair, element > 0;
    assert(0 <= base * element) by (nonlinear_arith) requires 0 <= base, element > 0;
}

} // verus!

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
    match u64::try_from(value) {
        Ok(extent) => Some(extent),
        Err(_) => None,
    }
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

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
