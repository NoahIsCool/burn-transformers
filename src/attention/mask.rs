//! Low-level exclusion planes shared by attention, MLA, and connectivity rules.
//!
//! Every mask here uses Burn's convention: `true` excludes a `[query, key]`
//! link. Builders materialize at most one Boolean plane per distinct
//! `[batch or 1, 1, query, key]` pattern and *expand* (a zero-copy view) it
//! across heads; integer positions are only ever vectors that broadcast in a
//! comparison, never `[batch, head, query, key]` integer planes.
//!
//! Causality is bottom-right aligned: with `Q` queries and `K` keys, query row
//! `i` sits at key position `i + K - Q`, so it may see keys `j <= i + K - Q`.
//! This is the only alignment that is correct when the queries follow a cached
//! prefix. With `Q > K`, the leading `Q - K` rows see no key at all.

use burn::tensor::{Bool, Int, Tensor, backend::Backend};

use crate::{Error, Result, architecture::TokenMask};

/// Builds the bottom-right causal exclusion `key > query + (K - Q)` as `[1, 1, Q, K]`.
///
/// Only one `Q x K` Boolean plane is materialized; callers expand it across
/// batch and heads.
pub(crate) fn bottom_right_causal_plane<B: Backend>(
    query_sequence: usize,
    key_sequence: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4, Bool>> {
    let queries = aligned_query_positions::<B>(query_sequence, key_sequence, device)?;
    let keys = key_positions::<B>(key_sequence, device)?;
    // [1, 1, 1, K] against [1, 1, Q, 1] broadcasts to one [1, 1, Q, K] plane.
    Ok(keys
        .reshape([1, 1, 1, key_sequence])
        .greater(queries.reshape([1, 1, query_sequence, 1])))
}

/// Builds the bottom-right causal exclusion expanded to `[batch, heads, Q, K]`.
///
/// The result is a broadcast view of one [`bottom_right_causal_plane`].
pub(crate) fn bottom_right_causal_mask<B: Backend>(
    batch: usize,
    heads: usize,
    query_sequence: usize,
    key_sequence: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4, Bool>> {
    Ok(
        bottom_right_causal_plane::<B>(query_sequence, key_sequence, device)?.expand([
            batch,
            heads,
            query_sequence,
            key_sequence,
        ]),
    )
}

/// Unions two optional exclusion masks; `None` means nothing is excluded.
///
/// Operands may differ in size-one axes; both are expanded to the common
/// broadcast shape before the elementwise OR.
pub(crate) fn union_exclusions<B: Backend>(
    first: Option<Tensor<B, 4, Bool>>,
    second: Option<Tensor<B, 4, Bool>>,
) -> Option<Tensor<B, 4, Bool>> {
    match (first, second) {
        (Some(first), Some(second)) => {
            let shape = broadcast_shape(first.dims(), second.dims());
            Some(expand_to(first, shape).bool_or(expand_to(second, shape)))
        }
        (first, None) => first,
        (None, second) => second,
    }
}

/// Builds the `[batch, 1, Q, K]` plane excluding invalid query rows and key columns.
///
/// Token masks use `true == valid` and must be `[batch, Q]` and `[batch, K]`.
pub(crate) fn padding_plane<B: Backend>(
    query_mask: Option<&TokenMask<B>>,
    key_mask: Option<&TokenMask<B>>,
    batch: usize,
    query_sequence: usize,
    key_sequence: usize,
) -> Result<Option<Tensor<B, 4, Bool>>> {
    let plane = [batch, 1, query_sequence, key_sequence];
    let query_invalid = query_mask
        .map(|mask| {
            invalid_tokens(mask, "query_mask", batch, query_sequence)
                .map(|invalid| invalid.reshape([batch, 1, query_sequence, 1]).expand(plane))
        })
        .transpose()?;
    let key_invalid = key_mask
        .map(|mask| {
            invalid_tokens(mask, "key_mask", batch, key_sequence)
                .map(|invalid| invalid.reshape([batch, 1, 1, key_sequence]).expand(plane))
        })
        .transpose()?;
    Ok(union_exclusions(query_invalid, key_invalid))
}

/// Bottom-right aligned query positions `K - Q .. K` as `[1, Q]` (may be negative).
pub(crate) fn aligned_query_positions<B: Backend>(
    query_sequence: usize,
    key_sequence: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
    let queries = signed_length(query_sequence, "query length exceeds i64")?;
    let keys = signed_length(key_sequence, "key length exceeds i64")?;
    let offset = keys - queries;
    Ok((Tensor::<B, 1, Int>::arange(0..queries, device) + offset).reshape([1, query_sequence]))
}

/// Key column positions `0 .. K` as `[1, K]`.
pub(crate) fn key_positions<B: Backend>(
    key_sequence: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
    let keys = signed_length(key_sequence, "key length exceeds i64")?;
    Ok(Tensor::<B, 1, Int>::arange(0..keys, device).reshape([1, key_sequence]))
}

/// Expands a mask to `shape` unless it already has exactly that shape.
pub(crate) fn expand_to<B: Backend>(
    mask: Tensor<B, 4, Bool>,
    shape: [usize; 4],
) -> Tensor<B, 4, Bool> {
    if mask.dims() == shape {
        mask
    } else {
        mask.expand(shape)
    }
}

/// Converts a length to `i64` for integer position arithmetic, with context.
pub(crate) fn signed_length(length: usize, reason: &'static str) -> Result<i64> {
    i64::try_from(length).map_err(|_| Error::configuration("attention.causal_mask", reason))
}

/// Checks one token mask's extent and returns its `true == invalid` complement.
fn invalid_tokens<B: Backend>(
    mask: &TokenMask<B>,
    name: &str,
    batch: usize,
    sequence: usize,
) -> Result<Tensor<B, 2, Bool>> {
    let [mask_batch, mask_sequence] = mask.as_tensor().dims();
    if [mask_batch, mask_sequence] != [batch, sequence] {
        return Err(Error::shape(
            name,
            format!("[{batch}, {sequence}]"),
            format!("[{mask_batch}, {mask_sequence}]"),
        ));
    }
    Ok(mask.as_tensor().clone().bool_not())
}

/// Returns the elementwise broadcast of two rank-4 shapes (size-one axes stretch).
fn broadcast_shape(first: [usize; 4], second: [usize; 4]) -> [usize; 4] {
    core::array::from_fn(|axis| first[axis].max(second[axis]))
}

#[cfg(test)]
#[path = "mask_tests.rs"]
mod tests;
