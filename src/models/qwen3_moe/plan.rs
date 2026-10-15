//! Dependency-free Qwen3-MoE layer schedule, verified in place by Verus.
//!
//! The schedule decides, for every decoder block, whether its feed-forward
//! branch is a routed expert bank and whether its attention is limited to a
//! sliding window. Both are pure integer predicates over the configuration,
//! checked by `verus/qwen.rs` for every input (see `verus/README.md`).

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

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
))]
pub(crate) fn sparse_layer(
    index: usize,
    sparse_step: usize,
    experts: usize,
    dense_override: bool,
) -> bool {
    sparse_step != 0 && !dense_override && experts > 0 && index % sparse_step == sparse_step - 1
}

/// Returns the sliding window of block `index`, when it has one.
///
/// Transformers 4.51 keeps blocks `0..max_window_layers` on full attention
/// and applies the configured window from block `max_window_layers` on, and
/// only when the feature is enabled and a width is configured.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(limited =>
    ensures
        limited.is_some() <==> (enabled && index >= max_window_layers && window.is_some()),
        limited.is_some() ==> limited == window,
))]
pub(crate) fn layer_window(
    index: usize,
    enabled: bool,
    max_window_layers: usize,
    window: Option<usize>,
) -> Option<usize> {
    if enabled && index >= max_window_layers {
        window
    } else {
        None
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
