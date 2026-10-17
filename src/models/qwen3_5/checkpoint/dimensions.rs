//! Checked Qwen3.5 checkpoint tensor widths.
//!
//! Every shape in the Qwen3.5 tensor contract is a configuration extent or one
//! of these fused widths, computed in the portable `u64` checkpoint
//! representation:
//!
//! - query `q d` and the gated query projection `2 q d`, whose rows alternate
//!   one query block and one output-gate block per head;
//! - native key/value `kv d`, before head repetition;
//! - the DeltaNet value width `Vh Vd` and convolution channels
//!   `2 Kh Kd + Vh Vd`, one per channel of the concatenated query, key, and
//!   value streams;
//! - the MTP fusion input `2 h`, the concatenated embedding and hidden state.
//!
//! # Verified properties
//!
//! Verus checks [`checkpoint_dimensions`] and [`query_projection`] in place
//! through `verus/loaders.rs` (see `verus/README.md`): every width is exact,
//! every plain extent is copied unchanged, the dimensions are returned
//! exactly when every fused width fits `u64`, and otherwise the error is the
//! first term that exceeds `u64::MAX`, in the order the contract has always
//! checked them. A `usize` extent always fits `u64`, so the conversions
//! cannot fail.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Configuration extents that determine every checkpoint tensor width.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy)]
pub(super) struct DimensionInputs {
    /// `vocab_size`.
    pub(super) vocab: usize,
    /// `hidden_size`.
    pub(super) hidden: usize,
    /// `intermediate_size`.
    pub(super) intermediate: usize,
    /// `num_attention_heads`.
    pub(super) attention_heads: usize,
    /// `num_key_value_heads`.
    pub(super) key_value_heads: usize,
    /// `head_dim`.
    pub(super) head_dim: usize,
    /// `linear_num_key_heads`.
    pub(super) delta_key_heads: usize,
    /// `linear_key_head_dim`.
    pub(super) delta_key_head_dim: usize,
    /// `linear_num_value_heads`.
    pub(super) delta_value_heads: usize,
    /// `linear_value_head_dim`.
    pub(super) delta_value_head_dim: usize,
    /// `linear_conv_kernel_dim`.
    pub(super) conv_kernel: usize,
}

/// Portable checkpoint shape terms computed with checked arithmetic.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy)]
pub(super) struct Dimensions {
    /// Vocabulary rows in the embedding and output projection.
    pub(super) vocab: u64,
    /// Residual-stream width shared by decoder blocks.
    pub(super) hidden: u64,
    /// Intermediate width of the gated feed-forward network.
    pub(super) intermediate: u64,
    /// Query width before gating: query heads times head dimension.
    pub(super) query: u64,
    /// Twice the query width for the combined query and output-gate projection.
    pub(super) gated_query: u64,
    /// Native key/value heads times head dimension, before head repetition.
    pub(super) key_value: u64,
    /// DeltaNet value heads times value-head dimension.
    pub(super) delta_value: u64,
    /// DeltaNet convolution channels: twice the key width plus the value width.
    pub(super) delta_conv: u64,
    /// Concatenated embedding and hidden-state width at the MTP input.
    pub(super) double_hidden: u64,
    /// Per-head query/key width used by headwise normalization.
    pub(super) head_dim: u64,
    /// Number of DeltaNet value heads and scalar gate outputs.
    pub(super) delta_value_heads: u64,
    /// Value width normalized independently within each DeltaNet head.
    pub(super) delta_value_head_dim: u64,
    /// Number of causal-convolution taps in each DeltaNet channel.
    pub(super) conv_kernel: u64,
}

/// The first fused width that does not fit `u64`, in checking order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DimensionError {
    /// `num_attention_heads * head_dim`.
    Query,
    /// `num_key_value_heads * head_dim`.
    KeyValue,
    /// `linear_num_key_heads * linear_key_head_dim`.
    DeltaKey,
    /// `linear_num_value_heads * linear_value_head_dim`.
    DeltaValue,
    /// `2 * delta key width + delta value width`.
    DeltaConv,
    /// `2 * query width`.
    GatedQuery,
    /// `2 * hidden_size`.
    DoubleHidden,
}

impl DimensionError {
    /// Configuration expression named in the contract's overflow diagnostic.
    pub(super) const fn field(self) -> &'static str {
        match self {
            Self::Query => "num_attention_heads * head_dim",
            Self::KeyValue => "num_key_value_heads * head_dim",
            Self::DeltaKey => "linear_num_key_heads * linear_key_head_dim",
            Self::DeltaValue => "linear_num_value_heads * linear_value_head_dim",
            Self::DeltaConv => "2 * delta key width + delta value width",
            Self::GatedQuery => "2 * query width",
            Self::DoubleHidden => "2 * hidden_size",
        }
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// `value` is representable in the portable checkpoint dimension type.
pub open spec fn fits(value: int) -> bool {
    value <= u64::MAX
}

/// The first fused width that does not fit `u64`, in the contract's checking
/// order (the order of [`DimensionError`]), or `None` exactly when every
/// product and sum fits.
pub(super) open spec fn first_overflow(inputs: DimensionInputs) -> Option<DimensionError> {
    let query = inputs.attention_heads * inputs.head_dim;
    let delta_key = inputs.delta_key_heads * inputs.delta_key_head_dim;
    let delta_value = inputs.delta_value_heads * inputs.delta_value_head_dim;
    if !fits(query) {
        Some(DimensionError::Query)
    } else if !fits(inputs.key_value_heads * inputs.head_dim) {
        Some(DimensionError::KeyValue)
    } else if !fits(delta_key) {
        Some(DimensionError::DeltaKey)
    } else if !fits(delta_value) {
        Some(DimensionError::DeltaValue)
    } else if !fits(2 * delta_key + delta_value) {
        Some(DimensionError::DeltaConv)
    } else if !fits(2 * query) {
        Some(DimensionError::GatedQuery)
    } else if !fits(2 * inputs.hidden) {
        Some(DimensionError::DoubleHidden)
    } else {
        None
    }
}

} // verus!

/// Computes every Qwen3.5 checkpoint width from configuration extents.
///
/// Succeeds exactly when every fused width fits `u64`, and then returns the
/// exact widths listed in the module documentation; otherwise reports the
/// first fused width (in the order of [`DimensionError`]) that exceeds
/// `u64::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(dimensions =>
    ensures
        ({
            let query = inputs.attention_heads * inputs.head_dim;
            let key_value = inputs.key_value_heads * inputs.head_dim;
            let delta_key = inputs.delta_key_heads * inputs.delta_key_head_dim;
            let delta_value = inputs.delta_value_heads * inputs.delta_value_head_dim;
            match dimensions {
                Ok(dimensions) => {
                    &&& first_overflow(inputs) is None
                    &&& dimensions.vocab == inputs.vocab
                    &&& dimensions.hidden == inputs.hidden
                    &&& dimensions.intermediate == inputs.intermediate
                    &&& dimensions.head_dim == inputs.head_dim
                    &&& dimensions.delta_value_heads == inputs.delta_value_heads
                    &&& dimensions.delta_value_head_dim == inputs.delta_value_head_dim
                    &&& dimensions.conv_kernel == inputs.conv_kernel
                    &&& dimensions.query == query
                    &&& dimensions.gated_query == 2 * query
                    &&& dimensions.key_value == key_value
                    &&& dimensions.delta_value == delta_value
                    &&& dimensions.delta_conv == 2 * delta_key + delta_value
                    &&& dimensions.double_hidden == 2 * inputs.hidden
                },
                Err(error) => first_overflow(inputs) == Some(error),
            }
        }),
))]
pub(super) fn checkpoint_dimensions(inputs: DimensionInputs) -> Result<Dimensions, DimensionError> {
    let hidden = inputs.hidden as u64;
    let Some(query) = (inputs.attention_heads as u64).checked_mul(inputs.head_dim as u64) else {
        return Err(DimensionError::Query);
    };
    let Some(key_value) = (inputs.key_value_heads as u64).checked_mul(inputs.head_dim as u64)
    else {
        return Err(DimensionError::KeyValue);
    };
    let Some(delta_key) =
        (inputs.delta_key_heads as u64).checked_mul(inputs.delta_key_head_dim as u64)
    else {
        return Err(DimensionError::DeltaKey);
    };
    let Some(delta_value) =
        (inputs.delta_value_heads as u64).checked_mul(inputs.delta_value_head_dim as u64)
    else {
        return Err(DimensionError::DeltaValue);
    };
    let delta_conv = match delta_key.checked_add(delta_key) {
        Some(keys) => match keys.checked_add(delta_value) {
            Some(channels) => channels,
            None => return Err(DimensionError::DeltaConv),
        },
        None => return Err(DimensionError::DeltaConv),
    };
    let Some(gated_query) = query.checked_add(query) else {
        return Err(DimensionError::GatedQuery);
    };
    let Some(double_hidden) = hidden.checked_add(hidden) else {
        return Err(DimensionError::DoubleHidden);
    };
    Ok(Dimensions {
        vocab: inputs.vocab as u64,
        hidden,
        intermediate: inputs.intermediate as u64,
        query,
        gated_query,
        key_value,
        delta_value,
        delta_conv,
        double_hidden,
        head_dim: inputs.head_dim as u64,
        delta_value_heads: inputs.delta_value_heads as u64,
        delta_value_head_dim: inputs.delta_value_head_dim as u64,
        conv_kernel: inputs.conv_kernel as u64,
    })
}

/// Query projection rows: `2 q d` with a fused output gate, otherwise `q d`.
///
/// The gated rows keep the official head-local `[head, query_or_gate,
/// head_dim]` order, so the width doubles without changing the head count.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    ensures
        width == if gated { dimensions.gated_query } else { dimensions.query },
))]
pub(super) fn query_projection(dimensions: &Dimensions, gated: bool) -> u64 {
    if gated {
        dimensions.gated_query
    } else {
        dimensions.query
    }
}

#[cfg(test)]
#[path = "dimensions_tests.rs"]
mod tests;
