//! Checked T5 checkpoint tensor widths and block sublayer numbering.
//!
//! Every T5 tensor shape is a configuration extent or the concatenated
//! attention width `H d_kv`, formed in the portable `u64` checkpoint
//! representation. A T5 block numbers its sublayers from zero: self-attention
//! is sublayer 0, a decoder block inserts encoder-decoder attention as
//! sublayer 1, and the feed-forward network follows, so it is sublayer 1 in
//! the encoder and 2 in the decoder. The contract builder and the tensor sink
//! both take that number from [`feed_forward_sublayer`].
//!
//! # Verified properties
//!
//! Verus checks [`checkpoint_dimensions`] and [`feed_forward_sublayer`] in
//! place through `verus/loaders.rs` (see `verus/README.md`): the attention
//! width is exactly `H d_kv` (or reported when it exceeds `u64::MAX`), every
//! extent is copied unchanged, and the feed-forward sublayer is the one after
//! the block's attention sublayers. A `usize` extent always fits `u64`, so
//! the conversions cannot fail.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Configuration extents that determine every checkpoint tensor width.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy)]
pub(super) struct DimensionInputs {
    /// `vocab_size`.
    pub(super) vocab: usize,
    /// `d_model`.
    pub(super) model: usize,
    /// `num_heads`.
    pub(super) heads: usize,
    /// `d_kv`.
    pub(super) head_dim: usize,
    /// `d_ff`.
    pub(super) feed_forward: usize,
    /// `relative_attention_num_buckets`.
    pub(super) buckets: usize,
}

/// Portable checkpoint dimensions converted once after topology validation.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy)]
pub(super) struct Dimensions {
    /// Number of rows in the shared token embedding table.
    pub(super) vocab: u64,
    /// Residual-stream width, equal to the embedding column count.
    pub(super) model: u64,
    /// Combined attention projection width, `num_heads * d_kv`.
    pub(super) attention: u64,
    /// Intermediate width of each feed-forward branch.
    pub(super) feed_forward: u64,
    /// Number of learned relative-position buckets.
    pub(super) buckets: u64,
    /// Number of attention heads, also the relative-bias table width.
    pub(super) heads: u64,
}

/// The attention width `num_heads * d_kv` does not fit `u64`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AttentionWidthOverflow;

/// Computes every T5 checkpoint width from configuration extents.
///
/// Returns the exact attention width `H d_kv` with every extent unchanged, or
/// [`AttentionWidthOverflow`] exactly when `H d_kv` exceeds `u64::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(dimensions =>
    ensures
        match dimensions {
            Ok(dimensions) => {
                &&& dimensions.vocab == inputs.vocab
                &&& dimensions.model == inputs.model
                &&& dimensions.feed_forward == inputs.feed_forward
                &&& dimensions.buckets == inputs.buckets
                &&& dimensions.heads == inputs.heads
                &&& dimensions.attention == inputs.heads * inputs.head_dim
            },
            Err(_) => inputs.heads * inputs.head_dim > u64::MAX,
        },
))]
pub(super) fn checkpoint_dimensions(
    inputs: DimensionInputs,
) -> Result<Dimensions, AttentionWidthOverflow> {
    let Some(attention) = (inputs.heads as u64).checked_mul(inputs.head_dim as u64) else {
        return Err(AttentionWidthOverflow);
    };
    Ok(Dimensions {
        vocab: inputs.vocab as u64,
        model: inputs.model as u64,
        attention,
        feed_forward: inputs.feed_forward as u64,
        buckets: inputs.buckets as u64,
        heads: inputs.heads as u64,
    })
}

/// Sublayer index of a block's feed-forward network: 1 in the encoder, 2 in
/// the decoder, immediately after self-attention and, in the decoder,
/// encoder-decoder attention.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sublayer =>
    ensures
        sublayer == 1 + if decoder { 1int } else { 0int },
))]
pub(crate) fn feed_forward_sublayer(decoder: bool) -> usize {
    if decoder { 2 } else { 1 }
}

#[cfg(test)]
#[path = "dimensions_tests.rs"]
mod tests;
