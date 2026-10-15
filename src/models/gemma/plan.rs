//! Pure integer planning for the Gemma layer schedule and its rotary sets.
//!
//! * [`sliding_by_pattern`] is the default local/global schedule of both
//!   references when a config omits `layer_types`: Gemma 2 alternates
//!   (`pattern = 2`), Gemma 3 makes every `sliding_window_pattern`-th layer
//!   (1-based) global. Transformers writes it as
//!   `"sliding_attention" if bool((i + 1) % pattern) else "full_attention"`.
//! * [`rotary_sets`] assigns every layer the rotary parameter set it rotates
//!   with: sliding layers set `0`, global layers set `1`, or set `0` for
//!   every layer when both layer types share one set (Gemma 2, or a Gemma 3
//!   config whose two sets coincide). The tables of one call are indexed by
//!   these numbers, so an out-of-range index would leave a layer unrotated.
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

/// Returns the rotary set index of every layer.
///
/// `sliding[l]` says whether layer `l` is a sliding (local) layer; `shared`
/// says whether both layer types rotate with one parameter set. Every index
/// is below the number of sets: one when shared, two otherwise.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sets =>
    ensures
        sets.len() == sliding.len(),
        forall|l: int| 0 <= l < sliding.len() ==> #[trigger] sets[l] == set_of(sliding[l], shared),
        forall|l: int| 0 <= l < sliding.len() ==> #[trigger] sets[l] < if shared { 1usize } else { 2usize },
))]
pub(crate) fn rotary_sets(sliding: &[bool], shared: bool) -> Vec<usize> {
    let mut sets: Vec<usize> = Vec::with_capacity(sliding.len());
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= sliding.len(),
            sets.len() == layer,
            forall|l: int| 0 <= l < layer ==> #[trigger] sets[l] == set_of(sliding[l], shared),
        decreases sliding.len() - layer,
    ))]
    while layer < sliding.len() {
        if shared || sliding[layer] {
            sets.push(0);
        } else {
            sets.push(1);
        }
        layer += 1;
    }
    sets
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
