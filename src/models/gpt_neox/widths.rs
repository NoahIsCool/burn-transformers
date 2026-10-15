//! Dependency-free projection widths of an edited GPT-NeoX attention layer.
//!
//! Public modules may be replaced between forwards, so every pass checks
//! each layer's attention geometry before reshaping: the attention must keep
//! the residual width, every head count and head width must be nonzero, the
//! key/value heads must divide the query heads, and the query width `q d`,
//! key/value width `kv d`, and query projection width (`q d`, doubled with a
//! fused output gate) must be representable.
//!
//! # Verified check
//!
//! [`layer_widths`] is checked in place by Verus through `verus/t5_gpt.rs`
//! (see `verus/README.md`): it never overflows or divides by zero, accepts
//! exactly the valid geometries with their exact widths, and names each
//! rejection's cause in the order the parameter check reports them.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Projection widths of one valid attention layer.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerWidths {
    /// Concatenated query-head width `q d`, the output projection's input.
    pub(crate) query: usize,
    /// Concatenated key (or value) head width `kv d`.
    pub(crate) key_value: usize,
    /// Query projection output, `q d` or `2 q d` with a fused output gate.
    pub(crate) query_projection: usize,
}

/// Why an attention layer's geometry is invalid.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerWidthError {
    /// The residual width changed, an extent is zero, or the heads do not group.
    Geometry,
    /// `q d` or `kv d` does not fit `usize`.
    Overflow,
    /// The gated query projection `2 q d` does not fit `usize`.
    GatedOverflow,
}

/// Checks one layer's attention geometry and returns its projection widths.
///
/// `hidden_size` is the attention's own residual width and `residual` the
/// model's; `gated` selects a query projection fused with an output gate.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(widths =>
    ensures
        ({
            let geometry = hidden_size == residual && query_heads > 0 && key_value_heads > 0
                && head_dim > 0 && query_heads % key_value_heads == 0;
            let gate = if gated { 2int } else { 1int };
            match widths {
                Ok(widths) => {
                    &&& geometry
                    &&& widths.query == query_heads * head_dim
                    &&& widths.key_value == key_value_heads * head_dim
                    &&& widths.query_projection == query_heads * head_dim * gate
                },
                Err(LayerWidthError::Geometry) => !geometry,
                Err(LayerWidthError::Overflow) => {
                    &&& geometry
                    &&& (query_heads * head_dim > usize::MAX
                        || key_value_heads * head_dim > usize::MAX)
                },
                Err(LayerWidthError::GatedOverflow) => {
                    &&& geometry
                    &&& query_heads * head_dim <= usize::MAX
                    &&& key_value_heads * head_dim <= usize::MAX
                    &&& query_heads * head_dim * gate > usize::MAX
                },
            }
        }),
))]
pub(crate) fn layer_widths(
    hidden_size: usize,
    residual: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    gated: bool,
) -> Result<LayerWidths, LayerWidthError> {
    if hidden_size != residual
        || query_heads == 0
        || key_value_heads == 0
        || head_dim == 0
        || !query_heads.is_multiple_of(key_value_heads)
    {
        return Err(LayerWidthError::Geometry);
    }
    let (query, key_value) = match (
        query_heads.checked_mul(head_dim),
        key_value_heads.checked_mul(head_dim),
    ) {
        (Some(query), Some(key_value)) => (query, key_value),
        _ => return Err(LayerWidthError::Overflow),
    };
    let query_projection = match query.checked_mul(if gated { 2 } else { 1 }) {
        Some(query_projection) => query_projection,
        None => return Err(LayerWidthError::GatedOverflow),
    };
    Ok(LayerWidths {
        query,
        key_value,
        query_projection,
    })
}

#[cfg(test)]
#[path = "widths_tests.rs"]
mod tests;
