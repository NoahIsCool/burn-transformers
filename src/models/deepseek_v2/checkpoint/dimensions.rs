//! Checked DeepSeek-V2 checkpoint tensor widths.
//!
//! Multi-head latent attention fuses several projections, and the contract
//! derives each fused width from configuration extents, in `usize`:
//!
//! - query heads `H (n + r)`, with `n = qk_nope_head_dim` content and
//!   `r = qk_rope_head_dim` rotary channels per head (directly or after the
//!   optional low-rank query bottleneck);
//! - the joint KV-A output `c + r`: the cacheable latent `c = kv_lora_rank`
//!   followed by one shared rotary key;
//! - the KV-B expansion `H (n + v)`: content keys and values of every head;
//! - the output projection input `H v`;
//! - the widened shared expert `m s`, `s` shared experts of width `m`.
//!
//! # Verified properties
//!
//! Verus checks [`checkpoint_dimensions`] in place through `verus/loaders.rs`
//! (see `verus/README.md`): every width is exact, every extent (including the
//! optional query rank) is copied unchanged, and an overflowing width is
//! reported by the first sum or product that exceeds `usize::MAX`, in the
//! order the contract has always checked them. A `usize` extent always fits
//! `u64`, so the conversions cannot fail.

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
    /// `n_shared_experts`.
    pub(super) shared_experts: usize,
    /// `n_routed_experts`.
    pub(super) routed_experts: usize,
    /// `num_attention_heads`.
    pub(super) attention_heads: usize,
    /// `qk_nope_head_dim`.
    pub(super) qk_nope: usize,
    /// `qk_rope_head_dim`.
    pub(super) qk_rope: usize,
    /// `v_head_dim`.
    pub(super) value_head_dim: usize,
    /// `q_lora_rank`.
    pub(super) query_rank: Option<usize>,
    /// `kv_lora_rank`.
    pub(super) kv_rank: usize,
}

/// Checked dimensions used by the contract builder.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy)]
pub(super) struct Dimensions {
    /// Token embedding and LM-head row count.
    pub(super) vocab: u64,
    /// Residual-stream width.
    pub(super) hidden: u64,
    /// Dense-prefix SwiGLU width.
    pub(super) intermediate: u64,
    /// One routed expert's SwiGLU width.
    pub(super) moe_intermediate: u64,
    /// Concatenated width of the checkpoint's shared-expert network.
    pub(super) shared_intermediate: u64,
    /// Routed expert count.
    pub(super) experts: u64,
    /// Concatenated per-head query width.
    pub(super) query: u64,
    /// Optional low-rank query bottleneck width.
    pub(super) query_rank: Option<u64>,
    /// Cacheable normalized-latent width.
    pub(super) kv_rank: u64,
    /// Joint KV-A latent-plus-rotary output width.
    pub(super) compressed: u64,
    /// Concatenated content-key and value expansion width.
    pub(super) expanded_kv: u64,
    /// Concatenated value-head width consumed by the output projection.
    pub(super) output_heads: u64,
}

/// The first fused width that does not fit `usize`, in checking order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DimensionError {
    /// `qk_nope_head_dim + qk_rope_head_dim`.
    QueryHead,
    /// `qk_nope_head_dim + v_head_dim`.
    ExpandedHead,
    /// `moe_intermediate_size * n_shared_experts`.
    SharedIntermediate,
    /// `num_attention_heads * qk_head_dim`.
    Query,
    /// `kv_lora_rank + qk_rope_head_dim`.
    Compressed,
    /// `num_attention_heads * expanded_kv_head_dim`.
    ExpandedKeyValue,
    /// `num_attention_heads * v_head_dim`.
    OutputHeads,
}

impl DimensionError {
    /// Configuration expression named in the contract's overflow diagnostic.
    pub(super) const fn field(self) -> &'static str {
        match self {
            Self::QueryHead => "qk_nope_head_dim + qk_rope_head_dim",
            Self::ExpandedHead => "qk_nope_head_dim + v_head_dim",
            Self::SharedIntermediate => "moe_intermediate_size * n_shared_experts",
            Self::Query => "num_attention_heads * qk_head_dim",
            Self::Compressed => "kv_lora_rank + qk_rope_head_dim",
            Self::ExpandedKeyValue => "num_attention_heads * expanded_kv_head_dim",
            Self::OutputHeads => "num_attention_heads * v_head_dim",
        }
    }

    /// Diagnostic naming the overflowing operation.
    pub(super) const fn reason(self) -> &'static str {
        match self {
            Self::QueryHead | Self::ExpandedHead | Self::Compressed => {
                "dimension sum overflows usize"
            }
            Self::SharedIntermediate | Self::Query | Self::ExpandedKeyValue | Self::OutputHeads => {
                "dimension product overflows usize"
            }
        }
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// `value` is representable as a `usize` extent.
pub open spec fn fits(value: int) -> bool {
    value <= usize::MAX
}

} // verus!

/// Computes every DeepSeek-V2 checkpoint width from configuration extents.
///
/// Returns the exact widths listed in the module documentation, or the first
/// sum or product (in the order of [`DimensionError`]) that exceeds
/// `usize::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(dimensions =>
    ensures
        ({
            let query_head = inputs.qk_nope + inputs.qk_rope;
            let expanded_head = inputs.qk_nope + inputs.value_head_dim;
            let shared = inputs.moe_intermediate * inputs.shared_experts;
            let query = inputs.attention_heads * query_head;
            let compressed = inputs.kv_rank + inputs.qk_rope;
            let expanded = inputs.attention_heads * expanded_head;
            let output = inputs.attention_heads * inputs.value_head_dim;
            match dimensions {
                Ok(dimensions) => {
                    &&& dimensions.vocab == inputs.vocab
                    &&& dimensions.hidden == inputs.hidden
                    &&& dimensions.intermediate == inputs.intermediate
                    &&& dimensions.moe_intermediate == inputs.moe_intermediate
                    &&& dimensions.experts == inputs.routed_experts
                    &&& dimensions.kv_rank == inputs.kv_rank
                    &&& match (dimensions.query_rank, inputs.query_rank) {
                        (Some(rank), Some(declared)) => rank == declared,
                        (None, None) => true,
                        _ => false,
                    }
                    &&& dimensions.shared_intermediate == shared
                    &&& dimensions.query == query
                    &&& dimensions.compressed == compressed
                    &&& dimensions.expanded_kv == expanded
                    &&& dimensions.output_heads == output
                },
                Err(DimensionError::QueryHead) => !fits(query_head),
                Err(DimensionError::ExpandedHead) => fits(query_head) && !fits(expanded_head),
                Err(DimensionError::SharedIntermediate) => fits(query_head)
                    && fits(expanded_head) && !fits(shared),
                Err(DimensionError::Query) => fits(query_head) && fits(expanded_head)
                    && fits(shared) && !fits(query),
                Err(DimensionError::Compressed) => fits(query_head) && fits(expanded_head)
                    && fits(shared) && fits(query) && !fits(compressed),
                Err(DimensionError::ExpandedKeyValue) => fits(query_head) && fits(expanded_head)
                    && fits(shared) && fits(query) && fits(compressed) && !fits(expanded),
                Err(DimensionError::OutputHeads) => fits(query_head) && fits(expanded_head)
                    && fits(shared) && fits(query) && fits(compressed) && fits(expanded)
                    && !fits(output),
            }
        }),
))]
pub(super) fn checkpoint_dimensions(inputs: DimensionInputs) -> Result<Dimensions, DimensionError> {
    let Some(query_head) = inputs.qk_nope.checked_add(inputs.qk_rope) else {
        return Err(DimensionError::QueryHead);
    };
    let Some(expanded_head) = inputs.qk_nope.checked_add(inputs.value_head_dim) else {
        return Err(DimensionError::ExpandedHead);
    };
    let Some(shared_intermediate) = inputs.moe_intermediate.checked_mul(inputs.shared_experts)
    else {
        return Err(DimensionError::SharedIntermediate);
    };
    let Some(query) = inputs.attention_heads.checked_mul(query_head) else {
        return Err(DimensionError::Query);
    };
    let Some(compressed) = inputs.kv_rank.checked_add(inputs.qk_rope) else {
        return Err(DimensionError::Compressed);
    };
    let Some(expanded_kv) = inputs.attention_heads.checked_mul(expanded_head) else {
        return Err(DimensionError::ExpandedKeyValue);
    };
    let Some(output_heads) = inputs.attention_heads.checked_mul(inputs.value_head_dim) else {
        return Err(DimensionError::OutputHeads);
    };
    // Verus cannot see through a closure passed to `Option::map`.
    #[allow(clippy::manual_map)]
    let query_rank = match inputs.query_rank {
        Some(rank) => Some(rank as u64),
        None => None,
    };
    Ok(Dimensions {
        vocab: inputs.vocab as u64,
        hidden: inputs.hidden as u64,
        intermediate: inputs.intermediate as u64,
        moe_intermediate: inputs.moe_intermediate as u64,
        shared_intermediate: shared_intermediate as u64,
        experts: inputs.routed_experts as u64,
        query: query as u64,
        query_rank,
        kv_rank: inputs.kv_rank as u64,
        compressed: compressed as u64,
        expanded_kv: expanded_kv as u64,
        output_heads: output_heads as u64,
    })
}

#[cfg(test)]
#[path = "dimensions_tests.rs"]
mod tests;
