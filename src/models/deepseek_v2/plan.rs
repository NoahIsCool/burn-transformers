//! Dependency-free DeepSeek-V2 schedule, routing, and latent-cache arithmetic, verified in place by Verus.
//!
//! # Sparse schedule
//!
//! Layer `i` of a DeepSeek-V2 checkpoint owns routed experts exactly when
//! `moe_layer_freq != 0`, `i >= first_k_dense_replace`, and
//! `i mod moe_layer_freq = 0` ([`sparse_layer`]); configuration validation
//! requires that some layer qualifies ([`any_sparse_layer`]).
//!
//! # Group-limited routing
//!
//! Group-limited selection keeps `topk_group` of the `n_group` expert groups
//! of `n_routed_experts / n_group` experts each, so at most
//! `(n_routed_experts / n_group) * topk_group` experts are eligible per token
//! ([`eligible_experts`]).
//!
//! # Latent cache geometry
//!
//! One MLA cache layer is a `[batch, capacity, kv_lora_rank +
//! qk_rope_head_dim]` latent plane; Burn's portable tensor indices are
//! signed 32-bit, and the whole cache must have a representable byte count
//! ([`latent_cache_bytes`]).
//!
//! Every function is checked through `verus/deepseek_muse.rs`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Layer `index` owns routed experts.
pub open spec fn sparse_spec(index: int, first_dense: int, frequency: int) -> bool {
    frequency != 0 && index >= first_dense && index % frequency == 0
}

} // verus!

/// Returns whether layer `index` owns routed (and shared) experts.
///
/// Proved: exactly `frequency != 0 && index >= first_dense && index mod
/// frequency = 0`, the reference `moe_layer_freq` and
/// `first_k_dense_replace` rule.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sparse =>
    ensures
        sparse == sparse_spec(index as int, first_dense as int, frequency as int),
))]
#[allow(
    clippy::manual_is_multiple_of,
    reason = "Verus specifies `%` but has no specification for `usize::is_multiple_of`"
)]
pub(crate) fn sparse_layer(index: usize, first_dense: usize, frequency: usize) -> bool {
    frequency != 0 && index >= first_dense && index % frequency == 0
}

/// Returns whether any of the first `layers` layers owns routed experts.
///
/// Evaluates layers in order and stops at the first sparse one. Proved:
/// exactly when some layer `i < layers` satisfies [`sparse_layer`].
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(any =>
    ensures
        any == (exists|i: int| 0 <= i < layers
            && #[trigger] sparse_spec(i, first_dense as int, frequency as int)),
))]
pub(crate) fn any_sparse_layer(layers: usize, first_dense: usize, frequency: usize) -> bool {
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= layers,
            forall|i: int| 0 <= i < index
                ==> !#[trigger] sparse_spec(i, first_dense as int, frequency as int),
        decreases layers - index,
    ))]
    while index < layers {
        if sparse_layer(index, first_dense, frequency) {
            return true;
        }
        index += 1;
    }
    false
}

/// Returns the experts eligible under group-limited routing, `(experts / groups) * top_groups`.
///
/// Proved: exactly that product when it is representable, `None` otherwise.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(eligible =>
    requires
        groups > 0,
    ensures
        match eligible {
            Some(eligible) => eligible == (experts / groups) * top_groups,
            None => (experts / groups) * top_groups > usize::MAX,
        },
))]
pub(crate) fn eligible_experts(experts: usize, groups: usize, top_groups: usize) -> Option<usize> {
    (experts / groups).checked_mul(top_groups)
}

/// Why a latent cache cannot be allocated.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LatentFault {
    /// One layer's `batch * capacity * width` elements exceed `i32::MAX` (or `usize`).
    LayerTooLarge,
    /// The whole cache's byte count overflows `usize`.
    BytesOverflow,
}

/// Returns the byte count of `layers` latent planes `[batch, capacity, width]` of `scalar_bytes`-byte scalars.
///
/// Proved: `Ok(total)` exactly when one layer's element count
/// `batch * capacity * width` fits `i32::MAX` and the products with `layers`
/// and `scalar_bytes` are representable, with
/// `total = batch * capacity * width * layers * scalar_bytes`; each error
/// names the first failing product, in the order the checks run. A cache
/// with at least one row and slot also bounds the per-token byte count
/// `width * layers * scalar_bytes <= total` and splits as
/// `total = batch * capacity * (width * layers * scalar_bytes)`, so the
/// state's byte accessors cannot overflow.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bytes =>
    ensures
        match bytes {
            Ok(total) => {
                &&& batch * capacity * width <= i32::MAX
                &&& total == batch * capacity * width * layers * scalar_bytes
                &&& (batch >= 1 && capacity >= 1 ==> width * layers * scalar_bytes <= total)
                &&& total == batch * capacity * (width * layers * scalar_bytes)
            },
            Err(LatentFault::LayerTooLarge) =>
                batch * capacity > usize::MAX || batch * capacity * width > i32::MAX,
            Err(LatentFault::BytesOverflow) => {
                &&& batch * capacity * width <= i32::MAX
                &&& (batch * capacity * width * layers > usize::MAX
                    || batch * capacity * width * layers * scalar_bytes > usize::MAX)
            },
        },
))]
pub(crate) fn latent_cache_bytes(
    batch: usize,
    capacity: usize,
    width: usize,
    layers: usize,
    scalar_bytes: usize,
) -> Result<usize, LatentFault> {
    let per_layer = match batch.checked_mul(capacity) {
        Some(rows) => match rows.checked_mul(width) {
            Some(elements) if elements <= i32::MAX as usize => elements,
            _ => return Err(LatentFault::LayerTooLarge),
        },
        None => return Err(LatentFault::LayerTooLarge),
    };
    let total = match per_layer.checked_mul(layers) {
        Some(elements) => match elements.checked_mul(scalar_bytes) {
            Some(total) => total,
            None => return Err(LatentFault::BytesOverflow),
        },
        None => return Err(LatentFault::BytesOverflow),
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(batch * capacity * width * layers * scalar_bytes
            == batch * capacity * (width * layers * scalar_bytes)) by (nonlinear_arith);
        if batch >= 1 && capacity >= 1 {
            assert(width * layers * scalar_bytes
                <= batch * capacity * (width * layers * scalar_bytes)) by (nonlinear_arith)
                requires
                    batch >= 1,
                    capacity >= 1,
                    width * layers * scalar_bytes >= 0;
        }
    }
    Ok(total)
}

#[cfg(test)]
#[path = "plan_tests.rs"]
/// Schedule, routing, and latent-geometry arithmetic against reference definitions.
mod tests;
