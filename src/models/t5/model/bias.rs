//! T5 relative-position bias for every execution path, built once per pass.
//!
//! T5 adds a learned scalar `W[bucket(k - q), h]` to every attention score of
//! head `h`, where `q` and `k` are the query's and key's positions and
//! `bucket` is the stack's logarithmic displacement mapping (bidirectional for
//! the encoder, causal for the decoder). Only block zero of a stack owns the
//! `[buckets, heads]` table `W`; every later block reuses the same bias, so a
//! pass builds it exactly once and shares the tensor across blocks.
//!
//! # Two geometries
//!
//! - **Contiguous positions** (`q = offset + i`, `k = j`): the bias depends on
//!   `j - i` only and is a `[1, heads, query, key]` plane that broadcasts over
//!   the batch. [`T5RelativePositionBias::forward`] builds it from
//!   `query + key - 1` host buckets in `O(query + key)`. Unpadded stateless
//!   passes, right- or left-padded rows, and cached decoding whose active rows
//!   share one history length all use it.
//! - **Explicit per-row positions** `p[b, t]` (logical decoder positions
//!   around interior padding, edited experiment positions, ragged cache
//!   lengths): the bias is `[batch, heads, query, key]` with
//!   `B[b, h, i, j] = W[bucket(p_k[b, j] - p_q[b, i]), h]`.
//!   [`positional_bias`] forms the displacement on the device and gathers
//!   bucket IDs from a host table of only `2r + 1` entries, so neither a host
//!   loop over `batch * query * key` pairs nor a quadratic upload remains.
//!
//! # Exact displacement clamping
//!
//! The host table covers displacements `-r..=r` with
//! `r = min(bound, max_distance)`, where `bound` is an upper bound on every
//! `|p_k - p_q|` the caller can guarantee (for positions in `0..E`, `E - 1`).
//! Device displacements are clamped into that range before the lookup. This
//! is exact: within `bound` nothing is clamped, and beyond `max_distance` the
//! T5 mapping saturates (`log(d / E') / log(M / E')` reaches one at `d = M`,
//! so every farther past key shares the final past bucket and every farther
//! future key the final future bucket; causal future keys are bucket zero).
//! A unit test checks the saturation against the pairwise mapping.
//!
//! # Precision and gradients
//!
//! Bucket IDs are integers, so both geometries select exactly the same table
//! entries as a pairwise evaluation and produce bit-identical biases. The
//! lookup is an ordinary embedding gather, so gradients reach the single
//! owner table in autodiff runs.

use burn::tensor::{Int, Tensor, TensorData, backend::Backend, module::embedding};

use crate::{Error, Result, position::T5RelativePositionBias};

/// Builds the `[1, heads, query_length, key_length]` bias for contiguous positions.
///
/// Queries sit at `query_offset + i` and keys at `j`; this is the geometry of
/// unpadded stateless passes (`query_offset = 0`) and of cached decoding
/// whose active rows share one committed length (`query_offset = L`).
pub(super) fn contiguous_bias<B: Backend>(
    table: &T5RelativePositionBias<B>,
    query_length: usize,
    key_length: usize,
    query_offset: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    table.forward(query_length, key_length, query_offset, device)
}

/// Builds `[batch, heads, query, key]` bias from explicit per-row positions.
///
/// `query_positions` is `[batch, query]` and `key_positions` is
/// `[batch, key]` (Int). `displacement_bound` must bound every
/// `|key_position - query_position|` (see the module documentation); the
/// result then equals a pairwise evaluation exactly. A batch of one
/// broadcasts like the contiguous bias.
///
/// # Errors
///
/// Returns a shape error for mismatched or empty extents and the errors of
/// [`T5RelativePositionBias::bucket_table`] for an unrepresentable table.
pub(super) fn positional_bias<B: Backend>(
    table: &T5RelativePositionBias<B>,
    query_positions: Tensor<B, 2, Int>,
    key_positions: Tensor<B, 2, Int>,
    displacement_bound: usize,
) -> Result<Tensor<B, 4>> {
    let [batch, queries] = query_positions.dims();
    let [key_batch, keys] = key_positions.dims();
    if batch == 0 || queries == 0 || keys == 0 || key_batch != batch {
        return Err(Error::shape(
            "t5.relative_positions",
            "query positions [batch>0, query>0] and key positions [batch, key>0]",
            format!(
                "query={:?}, key={:?}",
                query_positions.dims(),
                key_positions.dims()
            ),
        ));
    }
    let radius = displacement_bound.min(table.buckets().max_distance());
    let width = radius.checked_add(1).ok_or_else(|| {
        Error::configuration(
            "t5.relative_positions",
            "displacement bound overflows usize",
        )
    })?;
    // Entry t of a (r + 1) x (r + 1) table anchored at query position r holds
    // the bucket of displacement t - r, so it spans exactly -r..=r.
    let host = table.bucket_table(width, width, 0)?;
    let radius = i64::try_from(radius).map_err(|_| {
        Error::configuration("t5.relative_positions", "displacement bound exceeds i64")
    })?;
    let device = query_positions.device();
    let ids = Tensor::<B, 1, Int>::from_data(
        TensorData::new(host.bucket_ids().to_vec(), [host.bucket_ids().len()]),
        &device,
    );
    let plane = [batch, queries, keys];
    let displacement = key_positions.reshape([batch, 1, keys]).expand(plane)
        - query_positions.reshape([batch, queries, 1]).expand(plane);
    let index = displacement.clamp(-radius, radius).add_scalar(radius);
    let bucket_ids = ids
        .select(0, index.reshape([batch * queries * keys]))
        .reshape([batch * queries, keys]);
    let heads = table.weight.val().dims()[1];
    Ok(embedding(table.weight.val(), bucket_ids)
        .reshape([batch, queries, keys, heads])
        .permute([0, 3, 1, 2]))
}

#[cfg(test)]
#[path = "bias_tests.rs"]
mod tests;
