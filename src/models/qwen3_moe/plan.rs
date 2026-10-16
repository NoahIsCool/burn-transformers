//! Dependency-free Qwen3-MoE layer schedule, verified in place by Verus.
//!
//! The schedule decides, for every decoder block, whether its feed-forward
//! branch is a routed expert bank: a pure integer predicate over the
//! configuration ([`sparse_layer`]) and the `mlp_only_layers` membership test
//! ([`listed`]), both checked by `verus/qwen.rs` for every input (see
//! `verus/README.md`). [`any_sparse_layer`], the verified statement that the
//! schedule contains a sparse block at all, is the test oracle for the
//! `O(L + M)` check in `Qwen3MoeConfig::validate`, which marks the overrides
//! once instead of scanning them for every block. The sliding window needs no
//! schedule: the pinned reference applies an enabled window to every block.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Block `index` owns routed experts (the reference's `(i + 1) mod step = 0` rule).
pub open spec fn sparse_spec(index: int, sparse_step: int, experts: int, dense_override: bool) -> bool {
    sparse_step != 0 && !dense_override && experts > 0 && index % sparse_step == sparse_step - 1
}

} // verus!

/// Returns whether block `index` owns routed experts.
///
/// Transformers' Qwen3-MoE makes block `i` sparse when it is not listed in
/// `mlp_only_layers` (`dense_override`), the model has experts, and
/// `(i + 1) mod decoder_sparse_step = 0`; a zero step disables sparsity. For
/// a non-zero step the modular condition equals `i mod step = step - 1`, the
/// overflow-free form evaluated here.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sparse =>
    ensures
        sparse == (sparse_step != 0 && !dense_override && experts > 0
            && index % sparse_step == sparse_step - 1),
        sparse == sparse_spec(index as int, sparse_step as int, experts as int, dense_override),
))]
pub(crate) fn sparse_layer(
    index: usize,
    sparse_step: usize,
    experts: usize,
    dense_override: bool,
) -> bool {
    sparse_step != 0 && !dense_override && experts > 0 && index % sparse_step == sparse_step - 1
}

/// Returns whether `index` appears in `indices` (the `mlp_only_layers` override).
///
/// Proved: exactly sequence membership.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
        found == indices@.contains(index),
))]
pub(crate) fn listed(indices: &[usize], index: usize) -> bool {
    let mut k: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            k <= indices.len(),
            forall|j: int| 0 <= j < k ==> indices@[j] != index,
        decreases indices.len() - k,
    ))]
    while k < indices.len() {
        if indices[k] == index {
            return true;
        }
        k += 1;
    }
    false
}

/// Returns whether any of the first `layers` blocks owns routed experts.
///
/// Evaluates blocks in order and stops at the first sparse one, as the
/// configuration validation always has. Proved: exactly when some block
/// `i < layers` satisfies [`sparse_layer`] with its `mlp_only_layers`
/// membership.
#[cfg_attr(
    all(not(test), not(verus_keep_ghost)),
    expect(
        dead_code,
        reason = "verified oracle for the O(L + M) schedule check of validate; only tests call it"
    )
)]
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(any =>
    ensures
        any == (exists|i: int| 0 <= i < layers
            && #[trigger] sparse_spec(i, sparse_step as int, experts as int, dense_layers@.contains(i as usize))),
))]
pub(crate) fn any_sparse_layer(
    layers: usize,
    sparse_step: usize,
    experts: usize,
    dense_layers: &[usize],
) -> bool {
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= layers,
            forall|i: int| 0 <= i < index ==> !#[trigger] sparse_spec(i, sparse_step as int,
                experts as int, dense_layers@.contains(i as usize)),
        decreases layers - index,
    ))]
    while index < layers {
        if sparse_layer(index, sparse_step, experts, listed(dense_layers, index)) {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(sparse_spec(index as int, sparse_step as int, experts as int,
                    dense_layers@.contains((index as int) as usize)));
            }
            return true;
        }
        index += 1;
    }
    false
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
