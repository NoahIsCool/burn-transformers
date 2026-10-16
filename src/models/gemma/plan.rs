//! Pure integer planning for the Gemma layer schedule and its rotary sets.
//!
//! * [`sliding_by_pattern`] is the default local/global schedule of both
//!   references when a config omits `layer_types`: Gemma 2 alternates
//!   (`pattern = 2`), Gemma 3 makes every `sliding_window_pattern`-th layer
//!   (1-based) global. Transformers writes it as
//!   `"sliding_attention" if bool((i + 1) % pattern) else "full_attention"`.
//!   [`default_period`] picks that period and [`sliding_schedule`] applies it
//!   to every layer.
//! * [`rotary_sets`] assigns every layer the rotary parameter set it rotates
//!   with: sliding layers set `0`, global layers set `1`, or set `0` for
//!   every layer when both layer types share one set (Gemma 2, or a Gemma 3
//!   config whose two sets coincide); [`rotary_set_count`] is the number of
//!   sets, and [`rotary_set_table`] builds the parameter table with exactly
//!   that many entries. The tables of one call are indexed by these numbers,
//!   so an out-of-range index would leave a layer unrotated.
//! * [`unique_flag`] decides which single published namespace a checkpoint
//!   uses; the checked `u64` extents of its contract, `checkpoint_extent`
//!   and `projection_extent`, are the shared ones in
//!   [`crate::models::shared::decoder_plan`].
//!
//! # Verified planning
//!
//! Each function is dependency-free and checked in place by Verus through
//! `verus/gemma.rs` (see `verus/README.md`): for all admissible inputs the
//! proofs establish freedom from integer overflow and the stated
//! postconditions. The tensor code driven by these plans is covered by the
//! architecture and model tests.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

// The shared decoder plan's counting vocabulary underpins the schedule proof
// below.
#[cfg(verus_keep_ghost)]
use crate::models::shared::decoder_plan::{count_equal, lemma_count_prefix_stable};

#[cfg(verus_keep_ghost)]
verus! {

/// The successor of `a` is a multiple of `c` exactly when `a mod c = c - 1`.
///
/// With `a = c·q + r` and `0 <= r < c`: if `r + 1 < c` then
/// `(a + 1) mod c = r + 1 != 0`; otherwise `a + 1 = c·(q + 1)`.
pub proof fn lemma_successor_multiple(a: int, c: int)
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

/// The rotary set a layer reads: `0` when shared, else `0` sliding and `1` global.
pub open spec fn set_of(sliding: bool, shared: bool) -> usize {
    if shared || sliding {
        0
    } else {
        1
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

/// Returns whether layer `layer` slides under a global-layer period `pattern`.
///
/// Every `pattern`-th layer, counting from one, is global; all others slide.
/// The equivalent test `l mod p != p - 1` cannot overflow, unlike the
/// reference's `(l + 1) mod p != 0` at `l = usize::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sliding =>
    requires
        pattern > 0,
    ensures
        sliding == ((layer as int + 1) % (pattern as int) != 0),
))]
pub(crate) fn sliding_by_pattern(layer: usize, pattern: usize) -> bool {
    // (l + 1) mod p == 0 exactly when l mod p == p - 1.
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_successor_multiple(layer as int, pattern as int);
    }
    layer % pattern != pattern - 1
}

/// Returns the global-layer period of the default schedule.
///
/// Gemma 2 alternates (`alternating`, period 2); Gemma 3 reads
/// `sliding_window_pattern`, 6 when absent. Proved: exactly that period,
/// which is positive unless a Gemma 3 config sets zero (rejected before the
/// schedule is derived).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(period =>
    ensures
        period == if alternating {
            2
        } else {
            match configured {
                Some(pattern) => pattern,
                None => 6,
            }
        },
        alternating || configured != Some(0usize) ==> period > 0,
))]
pub(crate) fn default_period(alternating: bool, configured: Option<usize>) -> usize {
    if alternating {
        2
    } else {
        configured.unwrap_or(6)
    }
}

/// Returns the default sliding flag of each of `layers` layers under period `pattern`.
///
/// Proved: layer `l` slides exactly when `(l + 1) mod pattern != 0`, so the
/// global layers are exactly the 1-based multiples of the period and there
/// are exactly `⌊layers / pattern⌋` of them (Gemma 3's 26 layers with period
/// 6 have 4 global layers).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(flags =>
    requires
        pattern > 0,
    ensures
        flags.len() == layers,
        forall|l: int| 0 <= l < layers ==>
            #[trigger] flags@[l] == ((l + 1) % (pattern as int) != 0),
        count_equal(flags@, layers as int, false) == layers / pattern,
))]
pub(crate) fn sliding_schedule(layers: usize, pattern: usize) -> Vec<bool> {
    let mut flags: Vec<bool> = Vec::with_capacity(layers);
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= layers,
            pattern > 0,
            flags.len() == layer,
            forall|l: int| 0 <= l < layer ==>
                #[trigger] flags@[l] == ((l + 1) % (pattern as int) != 0),
            count_equal(flags@, layer as int, false) == layer / pattern,
        decreases layers - layer,
    ))]
    while layer < layers {
        #[cfg(verus_keep_ghost)]
        proof_decl! { let ghost before = flags@; }
        flags.push(sliding_by_pattern(layer, pattern));
        // The new flag is global exactly when the quotient steps.
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_prefix_stable(before, flags@, layer as int, false);
            lemma_div_successor(layer as int, pattern as int);
        }
        layer += 1;
    }
    flags
}

/// Returns the number of rotary parameter sets: one when shared, else two.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(count =>
    ensures
        count == if shared { 1usize } else { 2usize },
))]
pub(crate) const fn rotary_set_count(shared: bool) -> usize {
    if shared { 1 } else { 2 }
}

/// Returns the rotary parameter table that [`rotary_sets`] indexes.
///
/// Set `0` is the sliding configuration and, unless both layer types share
/// one set, set `1` is the global configuration. Proved: the table has
/// exactly [`rotary_set_count`] entries, and the set `set_of(type, shared)`
/// that [`rotary_sets`] assigns a layer is in range and holds that layer
/// type's configuration (the sliding one for a shared table, which the
/// caller builds only when both configurations are equal).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(table =>
    ensures
        table.len() == if shared { 1usize } else { 2usize },
        table@[0] == sliding,
        !shared ==> table@[1] == global,
        forall|layer_sliding: bool| #[trigger] set_of(layer_sliding, shared) < table.len()
            && table@[set_of(layer_sliding, shared) as int]
                == if layer_sliding || shared { sliding } else { global },
))]
pub(crate) fn rotary_set_table<T: Copy>(sliding: T, global: T, shared: bool) -> Vec<T> {
    let count = rotary_set_count(shared);
    let mut table: Vec<T> = Vec::with_capacity(count);
    table.push(sliding);
    if count > 1 {
        table.push(global);
    }
    table
}

/// Returns the rotary set index of every layer.
///
/// `sliding[l]` says whether layer `l` is a sliding (local) layer; `shared`
/// says whether both layer types rotate with one parameter set. Proved:
/// every layer rotates (no NoPE layer) with set `set_of(type, shared)`,
/// below [`rotary_set_count`]; with shared sets every layer reads set 0, and
/// otherwise two layers read the same set exactly when they have the same
/// type, so no layer ever rotates with the other type's parameters.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sets =>
    ensures
        sets.len() == sliding.len(),
        forall|l: int| 0 <= l < sliding.len() ==>
            #[trigger] sets@[l] == Some(set_of(sliding@[l], shared)),
        forall|l: int| 0 <= l < sliding.len() ==> match #[trigger] sets@[l] {
            Some(set) => set < if shared { 1usize } else { 2usize },
            None => false,
        },
        forall|i: int, j: int| 0 <= i < sliding.len() && 0 <= j < sliding.len() ==>
            (#[trigger] sets@[i] == #[trigger] sets@[j]) == (shared || sliding@[i] == sliding@[j]),
))]
pub(crate) fn rotary_sets(sliding: &[bool], shared: bool) -> Vec<Option<usize>> {
    let mut sets: Vec<Option<usize>> = Vec::with_capacity(sliding.len());
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= sliding.len(),
            sets.len() == layer,
            forall|l: int| 0 <= l < layer ==>
                #[trigger] sets@[l] == Some(set_of(sliding@[l], shared)),
        decreases sliding.len() - layer,
    ))]
    while layer < sliding.len() {
        if shared || sliding[layer] {
            sets.push(Some(0));
        } else {
            sets.push(Some(1));
        }
        layer += 1;
    }
    sets
}

/// Which of several flags is set: none, exactly one, or more than one.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UniqueFlag {
    /// No flag is set.
    Absent,
    /// Exactly this flag is set.
    Unique(
        /// The set flag's index.
        usize,
    ),
    /// At least two flags are set.
    Ambiguous,
}

/// Classifies `found` by how many of its flags are set.
///
/// Proved: `Unique(i)` exactly when flag `i` is the only set flag, `Absent`
/// exactly when none is set, and `Ambiguous` exactly when two distinct
/// flags are set, so a checkpoint is assigned a namespace only when exactly
/// one namespace's embedding is present.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(selection =>
    ensures
        match selection {
            UniqueFlag::Absent => forall|i: int| 0 <= i < found.len() ==> !#[trigger] found@[i],
            UniqueFlag::Unique(index) => {
                &&& index < found.len()
                &&& found@[index as int]
                &&& forall|i: int| 0 <= i < found.len() && i != index ==> !#[trigger] found@[i]
            },
            UniqueFlag::Ambiguous => exists|i: int, j: int|
                0 <= i < j < found.len() && #[trigger] found@[i] && #[trigger] found@[j],
        },
))]
pub(crate) fn unique_flag(found: &[bool]) -> UniqueFlag {
    let mut selection = UniqueFlag::Absent;
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= found.len(),
            match selection {
                UniqueFlag::Absent => forall|i: int| 0 <= i < index ==> !#[trigger] found@[i],
                UniqueFlag::Unique(first) => {
                    &&& first < index
                    &&& found@[first as int]
                    &&& forall|i: int| 0 <= i < index && i != first ==> !#[trigger] found@[i]
                },
                UniqueFlag::Ambiguous => false,
            },
        decreases found.len() - index,
    ))]
    while index < found.len() {
        if found[index] {
            match selection {
                UniqueFlag::Absent => selection = UniqueFlag::Unique(index),
                _ => {
                    // The earlier unique flag and this one are two set flags.
                    #[cfg(verus_keep_ghost)]
                    proof! {
                        if let UniqueFlag::Unique(first) = selection {
                            assert(found@[first as int] && found@[index as int]);
                        }
                    }
                    return UniqueFlag::Ambiguous;
                }
            }
        }
        index += 1;
    }
    selection
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
