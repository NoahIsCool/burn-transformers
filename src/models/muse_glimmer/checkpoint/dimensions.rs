//! Checked Muse-Glimmer checkpoint tensor widths.
//!
//! The Muse-Glimmer language-tower contract uses plain configuration extents
//! (vocabulary, residual, and SwiGLU widths) and two fused attention widths:
//! the concatenated query heads `q d`, shared by the query and output-gate
//! projections, and the native key/value heads `kv d`, each multiplied in
//! `usize` as the attention layers do.
//!
//! # Verified properties
//!
//! Verus checks [`checkpoint_dimensions`] in place through `verus/loaders.rs`
//! (see `verus/README.md`): every width is exact, every extent is copied
//! unchanged, the dimensions are returned exactly when both products fit
//! `usize`, and otherwise the error is the first product that exceeds
//! `usize::MAX` (query before key/value). A `usize` extent always fits
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
    /// `num_attention_heads`.
    pub(super) attention_heads: usize,
    /// `num_key_value_heads`.
    pub(super) key_value_heads: usize,
    /// `head_dim`.
    pub(super) head_dim: usize,
}

/// Checked shape terms shared across every layer mapping.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy)]
pub(super) struct Dimensions {
    /// Number of embedding rows and vocabulary logits.
    pub(super) vocab: u64,
    /// Residual width used by embeddings and normalization scales.
    pub(super) hidden: u64,
    /// Expanded SwiGLU width.
    pub(super) intermediate: u64,
    /// Concatenated query-head width after checked multiplication.
    pub(super) query: u64,
    /// Concatenated native key/value-head width after checked multiplication.
    pub(super) key_value: u64,
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

#[cfg(verus_keep_ghost)]
verus! {

/// `value` is representable as a `usize` extent.
pub open spec fn fits(value: int) -> bool {
    value <= usize::MAX
}

/// The first attention product that does not fit `usize` (query before
/// key/value), or `None` exactly when both `q d` and `kv d` fit.
pub(super) open spec fn first_overflow(inputs: DimensionInputs) -> Option<DimensionError> {
    if !fits(inputs.attention_heads * inputs.head_dim) {
        Some(DimensionError::Query)
    } else if !fits(inputs.key_value_heads * inputs.head_dim) {
        Some(DimensionError::KeyValue)
    } else {
        None
    }
}

} // verus!

/// Computes every Muse-Glimmer checkpoint width from configuration extents.
///
/// Succeeds exactly when both `q d` and `kv d` fit `usize`, and then returns
/// them exactly with every extent unchanged; otherwise reports the first
/// product that exceeds `usize::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(dimensions =>
    ensures
        ({
            let query = inputs.attention_heads * inputs.head_dim;
            let key_value = inputs.key_value_heads * inputs.head_dim;
            match dimensions {
                Ok(dimensions) => {
                    &&& first_overflow(inputs) is None
                    &&& dimensions.vocab == inputs.vocab
                    &&& dimensions.hidden == inputs.hidden
                    &&& dimensions.intermediate == inputs.intermediate
                    &&& dimensions.query == query
                    &&& dimensions.key_value == key_value
                },
                Err(error) => first_overflow(inputs) == Some(error),
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
        query: query as u64,
        key_value: key_value as u64,
    })
}

#[cfg(test)]
#[path = "dimensions_tests.rs"]
mod tests;
