//! Checked Qwen3-MoE checkpoint tensor widths.
//!
//! The Qwen3-MoE contract uses plain configuration extents (vocabulary,
//! residual, dense and expert intermediate widths, expert count, head width)
//! and two fused attention widths: the concatenated query heads `q d` and the
//! native key/value heads `kv d`, each multiplied in `usize` as the attention
//! layers do.
//!
//! # Verified properties
//!
//! Verus checks [`checkpoint_dimensions`] in place through `verus/loaders.rs`
//! (see `verus/README.md`): every width is exact, every extent is copied
//! unchanged, and an overflowing width is reported by the first product that
//! exceeds `usize::MAX` (query before key/value). A `usize` extent always
//! fits `u64`, so the conversions cannot fail.

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
    /// `moe_intermediate_size`.
    pub(super) moe_intermediate: usize,
    /// `num_experts`.
    pub(super) experts: usize,
    /// `num_attention_heads`.
    pub(super) attention_heads: usize,
    /// `num_key_value_heads`.
    pub(super) key_value_heads: usize,
    /// `head_dim`.
    pub(super) head_dim: usize,
}

/// Checked shape terms used by the contract builder.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy)]
pub(super) struct Dimensions {
    /// Number of vocabulary rows in embedding and output matrices.
    pub(super) vocab: u64,
    /// Residual-stream width shared by decoder blocks.
    pub(super) hidden: u64,
    /// Intermediate width of dense feed-forward layers.
    pub(super) intermediate: u64,
    /// Intermediate width of each routed expert.
    pub(super) moe_intermediate: u64,
    /// Number of routed experts and router output channels.
    pub(super) experts: u64,
    /// Concatenated query width: query heads times head dimension.
    pub(super) query: u64,
    /// Concatenated native key/value width before head repetition.
    pub(super) key_value: u64,
    /// Per-head query/key width used by headwise RMS normalization.
    pub(super) head_dim: u64,
}

/// The first attention width that does not fit `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DimensionError {
    /// `num_attention_heads * head_dim`.
    Query,
    /// `num_key_value_heads * head_dim`.
    KeyValue,
}

impl DimensionError {
    /// Configuration expression named in the contract's overflow diagnostic.
    pub(super) const fn field(self) -> &'static str {
        match self {
            Self::Query => "num_attention_heads * head_dim",
            Self::KeyValue => "num_key_value_heads * head_dim",
        }
    }
}

/// Computes every Qwen3-MoE checkpoint width from configuration extents.
///
/// Returns `q d` and `kv d` exactly with every extent unchanged, or the first
/// product that exceeds `usize::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(dimensions =>
    ensures
        ({
            let query = inputs.attention_heads * inputs.head_dim;
            let key_value = inputs.key_value_heads * inputs.head_dim;
            match dimensions {
                Ok(dimensions) => {
                    &&& dimensions.vocab == inputs.vocab
                    &&& dimensions.hidden == inputs.hidden
                    &&& dimensions.intermediate == inputs.intermediate
                    &&& dimensions.moe_intermediate == inputs.moe_intermediate
                    &&& dimensions.experts == inputs.experts
                    &&& dimensions.head_dim == inputs.head_dim
                    &&& dimensions.query == query
                    &&& dimensions.key_value == key_value
                },
                Err(DimensionError::Query) => query > usize::MAX,
                Err(DimensionError::KeyValue) => query <= usize::MAX && key_value > usize::MAX,
            }
        }),
))]
pub(super) fn checkpoint_dimensions(inputs: DimensionInputs) -> Result<Dimensions, DimensionError> {
    let Some(query) = inputs.attention_heads.checked_mul(inputs.head_dim) else {
        return Err(DimensionError::Query);
    };
    let Some(key_value) = inputs.key_value_heads.checked_mul(inputs.head_dim) else {
        return Err(DimensionError::KeyValue);
    };
    Ok(Dimensions {
        vocab: inputs.vocab as u64,
        hidden: inputs.hidden as u64,
        intermediate: inputs.intermediate as u64,
        moe_intermediate: inputs.moe_intermediate as u64,
        experts: inputs.experts as u64,
        query: query as u64,
        key_value: key_value as u64,
        head_dim: inputs.head_dim as u64,
    })
}

#[cfg(test)]
#[path = "dimensions_tests.rs"]
mod tests;
