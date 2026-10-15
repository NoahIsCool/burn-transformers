//! Spec-derived random initialization for modular models.
//!
//! Burn's defaults (`N(0, 1)` embeddings, Kaiming-uniform projections) make a
//! tied readout start with logit standard deviation `~ sqrt(hidden_size)` and
//! an initial loss far above `ln V` (see [`crate::initialization`]). The
//! modular models therefore offer a recommended [`InitPolicy`] derived from
//! the specification:
//!
//! * embeddings and every projection draw from `N(0, std^2)`, biases are zero;
//! * projections that write into the residual stream are additionally scaled
//!   by `1/sqrt(2 L)` (GPT-2), so the residual variance at initialization does
//!   not grow with depth `L`.
//!
//! [`InitPolicy::apply`] rejects residual-projection names that match no
//! `Linear` field, so the names are derived from the blocks the spec actually
//! contains: attention, MLA, and differential-attention output projections are
//! `o_proj`, Gated DeltaNet and Mamba-2 write through `out_proj`, and every
//! dense or expert MLP through `down_proj`. Attention-only stacks have no
//! `down_proj`. Mamba-2's fused input projection is an ordinary projection;
//! its convolution, decay (`A_log`), step bias, skip `D`, and norm scale are
//! [`crate::initialization::ParameterRole::Other`] and keep their reference
//! initialization.

use std::collections::BTreeSet;

use crate::{
    architecture::{DecoderSpec, FeedForwardSpec, MixerSpec},
    initialization::InitPolicy,
};

/// Builds the recommended policy for the blocks of `stacks`.
///
/// `depth` is the residual-block count `L` of the depth scaling;
/// `cross_attention` adds the decoder cross-attention output projection.
pub(super) fn scaled_init_policy<'a>(
    stacks: impl IntoIterator<Item = &'a DecoderSpec>,
    depth: usize,
    cross_attention: bool,
    std: f64,
) -> InitPolicy {
    let mut residual_projections = BTreeSet::new();
    if cross_attention {
        residual_projections.insert("o_proj");
    }
    for layer in stacks.into_iter().flat_map(|spec| &spec.layers) {
        match &layer.mixer {
            MixerSpec::DeltaNet { .. } | MixerSpec::Mamba2(_) => {
                residual_projections.insert("out_proj");
            }
            MixerSpec::Attention(_)
            | MixerSpec::SlidingAttention(_)
            | MixerSpec::AttentionWithOptions { .. }
            | MixerSpec::Mla(_)
            | MixerSpec::DifferentialAttention(_) => {
                residual_projections.insert("o_proj");
            }
        }
        if layer.feed_forward != FeedForwardSpec::None {
            residual_projections.insert("down_proj");
        }
    }
    let policy = InitPolicy::normal(std);
    if depth == 0 || residual_projections.is_empty() {
        policy
    } else {
        policy.with_depth_scaling(depth, residual_projections)
    }
}

#[cfg(test)]
#[path = "init_tests.rs"]
/// Residual-projection names and initial logit scale of the derived policies.
mod tests;
