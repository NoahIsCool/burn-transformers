//! Pure projection-width arithmetic for grouped-query attention.
//!
//! A layer with hidden width `H`, `q` query heads, `kv` key/value heads, and
//! head width `d` has a query projection `H -> q * d` (doubled to `2 * q * d`
//! when a per-head output gate is fused into it), key and value projections
//! `H -> kv * d`, and an output projection `q * d -> H`. Grouping requires
//! `kv` to divide `q`. Burn allocates parameters lazily, so an overflowing
//! matrix element count would otherwise surface only at first use.
//!
//! # Verified arithmetic
//!
//! [`projection_widths`] is checked in place by Verus through
//! `verus/attention.rs`: for every input it never overflows, a successful
//! result reports the exact widths with every projection matrix's element
//! count inside `usize`, and each error states exactly which condition failed.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Validated widths of one grouped-query attention layer's projections.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionWidths {
    /// Concatenated query-head width `q * d`, also the output projection's input.
    pub query_width: usize,
    /// Concatenated key (or value) head width `kv * d`.
    pub key_value_width: usize,
    /// Query projection output, `q * d` or `2 * q * d` with a fused gate.
    pub query_output_width: usize,
}

/// Why projection widths are not representable.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WidthError {
    /// A width or head count is zero.
    EmptyAxis,
    /// Key/value heads do not divide query heads.
    UngroupedHeads,
    /// A head width or projection element count overflows `usize`.
    Overflow,
}

/// Computes projection widths, rejecting empty, ungrouped, or overflowing layouts.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Ok(widths) => {
                let query_output = if gated { query_heads * head_dim * 2 } else { query_heads * head_dim };
                &&& hidden_size >= 1 && query_heads >= 1 && key_value_heads >= 1 && head_dim >= 1
                &&& query_heads % key_value_heads == 0
                &&& widths.query_width == query_heads * head_dim
                &&& widths.key_value_width == key_value_heads * head_dim
                &&& widths.query_output_width == query_output
                &&& widths.query_width <= widths.query_output_width
                &&& hidden_size * widths.query_output_width <= usize::MAX
                &&& hidden_size * widths.key_value_width <= usize::MAX
                &&& widths.query_width * hidden_size <= usize::MAX
            },
            Err(WidthError::EmptyAxis) =>
                hidden_size == 0 || query_heads == 0 || key_value_heads == 0 || head_dim == 0,
            Err(WidthError::UngroupedHeads) => {
                &&& hidden_size >= 1 && query_heads >= 1 && key_value_heads >= 1 && head_dim >= 1
                &&& query_heads % key_value_heads != 0
            },
            Err(WidthError::Overflow) => {
                let query_output = if gated { query_heads * head_dim * 2 } else { query_heads * head_dim };
                &&& hidden_size >= 1 && query_heads >= 1 && key_value_heads >= 1 && head_dim >= 1
                &&& query_heads % key_value_heads == 0
                &&& !(query_output <= usize::MAX
                    && key_value_heads * head_dim <= usize::MAX
                    && hidden_size * query_output <= usize::MAX
                    && hidden_size * (key_value_heads * head_dim) <= usize::MAX)
            },
        }
))]
pub fn projection_widths(
    hidden_size: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    gated: bool,
) -> Result<ProjectionWidths, WidthError> {
    if hidden_size == 0 || query_heads == 0 || key_value_heads == 0 || head_dim == 0 {
        return Err(WidthError::EmptyAxis);
    }
    let ungrouped = query_heads % key_value_heads;
    if ungrouped != 0 {
        return Err(WidthError::UngroupedHeads);
    }
    let Some(query_width) = query_heads.checked_mul(head_dim) else {
        return Err(WidthError::Overflow);
    };
    let Some(key_value_width) = key_value_heads.checked_mul(head_dim) else {
        return Err(WidthError::Overflow);
    };
    let query_output_width = if gated {
        let Some(doubled) = query_width.checked_mul(2) else {
            return Err(WidthError::Overflow);
        };
        doubled
    } else {
        query_width
    };
    // Matrix element counts: the query and output projections share
    // `hidden * query_width <= hidden * query_output_width`.
    if hidden_size.checked_mul(query_output_width).is_none()
        || hidden_size.checked_mul(key_value_width).is_none()
    {
        return Err(WidthError::Overflow);
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(query_width as int * hidden_size as int
            <= hidden_size as int * query_output_width as int) by (nonlinear_arith)
            requires
                query_width <= query_output_width,
                hidden_size >= 1;
    }
    Ok(ProjectionWidths {
        query_width,
        key_value_width,
        query_output_width,
    })
}

#[cfg(test)]
#[path = "widths_tests.rs"]
mod tests;
