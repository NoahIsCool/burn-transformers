//! Pure index planning for the Llama-family schedule and batched cache steps.
//!
//! Everything here is integer bookkeeping that the tensor code consumes:
//!
//! * the SmolLM3 NoPE rule (`rotary_by_interval`) and the Qwen sliding
//!   rule (`sliding_from_layer`) that derive per-layer schedules;
//! * `cache_slots`, which numbers layers inside their KV cache group (full
//!   storage or sliding ring);
//! * the batched-step layout of one row of a `[batch, block]` token block:
//!   whether its valid columns form one run (`is_one_run`), the cache
//!   positions of its slots (`slot_positions`), the gather that compacts
//!   interior padding into a right-padded block (`compaction_sources`), and
//!   the gathers that restore the rectangular layout with exact zeros
//!   (`scatter_sources` from a compact row, `restore_slots` from any run
//!   offset).
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

#[cfg(verus_keep_ghost)]
verus! {

/// `s` is strictly increasing and every entry lies below `bound`.
pub open spec fn sorted_unique_below(s: Seq<usize>, bound: int) -> bool {
    &&& forall|i: int, j: int| 0 <= i < j < s.len() ==> s[i] < s[j]
    &&& forall|i: int| 0 <= i < s.len() ==> s[i] < bound
}

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

#[cfg(verus_keep_ghost)]
verus! {

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

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
