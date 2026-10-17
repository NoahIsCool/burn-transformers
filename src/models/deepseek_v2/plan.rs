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
//! signed 32-bit, and the whole cache must have a representable byte count.
//! [`latent_cache_bytes`] checks both and returns the total and per-token
//! bytes the decode state reports ([`LatentBytes`]).
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
pub(crate) fn sparse_layer(index: usize, first_dense: usize, frequency: usize) -> bool {
    frequency != 0 && index >= first_dense && index.is_multiple_of(frequency)
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
    /// `batch * capacity` overflows `usize`, or one layer's `batch * capacity * width` elements exceed `i32::MAX`.
    LayerTooLarge,
    /// `elements * layers` or `elements * layers * scalar_bytes` overflows `usize`.
    BytesOverflow,
}

/// Byte accounting of an accepted latent cache, fixed for the state's lifetime.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LatentBytes {
    /// Bytes allocated by every layer's `[batch, capacity, width]` plane.
    pub(crate) total: usize,
    /// Bytes one more token occupies across all layers, `width * layers * scalar_bytes`.
    pub(crate) per_token: usize,
}

/// Returns the byte accounting of `layers` latent planes `[batch, capacity, width]` of `scalar_bytes`-byte scalars.
///
/// The checks run in order: `batch * capacity` and then one layer's element
/// count `elements = batch * capacity * width` must be representable, with
/// `elements <= i32::MAX` (Burn's portable index range), else
/// [`LatentFault::LayerTooLarge`]; then `elements * layers` and
/// `elements * layers * scalar_bytes` must be representable, else
/// [`LatentFault::BytesOverflow`]. Proved: the cache is accepted exactly when
/// every one of those intermediate products is representable and
/// `elements <= i32::MAX`, each error names the first failing check, and an
/// accepted cache reports the exact `total = elements * layers *
/// scalar_bytes` and the per-token bytes `width * layers * scalar_bytes`
/// saturated at `usize::MAX`. Every cache with at least one row and slot (all
/// a decode state can hold) has the exact, unsaturated per-token count, and
/// `total = batch * capacity * per_token`, so the stored values are the
/// state's byte accessors.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bytes =>
    ensures
        match bytes {
            Ok(bytes) => {
                &&& batch * capacity * width <= i32::MAX
                &&& bytes.total == batch * capacity * width * layers * scalar_bytes
                &&& bytes.per_token == vstd::math::min(
                    width * layers * scalar_bytes, usize::MAX as int)
                &&& (batch >= 1 && capacity >= 1 ==> {
                    &&& bytes.per_token == width * layers * scalar_bytes
                    &&& bytes.total == batch * capacity * bytes.per_token
                })
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
) -> Result<LatentBytes, LatentFault> {
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
    // Saturating `width * layers * scalar_bytes`: only a cache without rows
    // or slots, which no decode state holds, can saturate (see below).
    let per_token = match width.checked_mul(layers) {
        Some(scalars) => match scalars.checked_mul(scalar_bytes) {
            Some(bytes) => bytes,
            None => usize::MAX,
        },
        None => {
            if scalar_bytes == 0 {
                0
            } else {
                usize::MAX
            }
        }
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(batch * capacity * width * layers * scalar_bytes
            == batch * capacity * (width * layers * scalar_bytes)) by (nonlinear_arith);
        if scalar_bytes >= 1 {
            assert(width * layers <= width * layers * scalar_bytes) by (nonlinear_arith)
                requires
                    scalar_bytes >= 1,
                    width * layers >= 0;
        }
        if batch >= 1 && capacity >= 1 {
            assert(width * layers * scalar_bytes
                <= batch * capacity * (width * layers * scalar_bytes)) by (nonlinear_arith)
                requires
                    batch >= 1,
                    capacity >= 1,
                    width * layers * scalar_bytes >= 0;
        }
    }
    Ok(LatentBytes { total, per_token })
}

#[cfg(test)]
#[path = "plan_tests.rs"]
/// Schedule, routing, and latent-geometry arithmetic against reference definitions.
mod tests;
