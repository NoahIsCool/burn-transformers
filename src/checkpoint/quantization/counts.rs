//! Exact group, packed-value, and scale counts of a quantized tensor.
//!
//! A tensor of `n` values quantized in contiguous groups of `G` values has
//! `⌈n / G⌉` groups (the last one possibly partial), one four-byte F32 scale
//! per group, and `n` packed Q8 bytes or `⌈n / 2⌉` packed Q4 bytes. For a
//! rank-two `[rows, columns]` matrix whose width is a multiple of `G`, those
//! flat row-major groups are exactly Burn's `[1, G]` blocks: there are
//! `rows · (columns / G)` of them, which equals `⌈rows · columns / G⌉`.
//!
//! The module depends only on the packed code widths and the shared
//! [`ceil_div_u64`] core, with no
//! Burn or allocator dependencies, so that Verus can check it in place through
//! `verus/checkpoint.rs`; callers map its `None` results to contextual
//! quantization errors.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

use super::packing::CodeWidth;
use crate::division::ceil_div_u64;
#[cfg(verus_keep_ghost)]
use crate::division::{ceiling, lemma_ceiling_bounds};

/// Persisted counts derived from a tensor's element count and quantization layout.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WireCounts {
    /// Contiguous groups, each sharing one scale: `⌈n / G⌉`.
    pub(crate) group_count: u64,
    /// Packed code bytes: `n` (Q8) or `⌈n / 2⌉` (Q4).
    pub(crate) packed_value_bytes: u64,
    /// Scale bytes `4 · groups`, or `None` when that overflows `u64`.
    pub(crate) scale_bytes: Option<u64>,
}

/// Derives the group, packed-byte, and scale-byte counts of `elements` values.
///
/// Verified: every count is exact; the scale length is `None` only when
/// `4 · groups` exceeds `u64::MAX`, which cannot happen for the supported
/// group sizes (`G >= 32`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(counts =>
    requires
        group_size > 0,
    ensures
        counts.group_count == ceiling(elements as int, group_size as int),
        width == CodeWidth::Byte ==> counts.packed_value_bytes == elements,
        width == CodeWidth::Nibble ==>
            counts.packed_value_bytes == ceiling(elements as int, 2),
        counts.packed_value_bytes <= elements,
        match counts.scale_bytes {
            Some(bytes) => bytes == 4 * counts.group_count,
            None => 4 * counts.group_count > u64::MAX,
        },
        group_size >= 32 ==> counts.scale_bytes is Some,
))]
pub(crate) fn wire_counts(elements: u64, group_size: u64, width: CodeWidth) -> WireCounts {
    let group_count = ceil_div_u64(elements, group_size);
    let packed_value_bytes = match width {
        CodeWidth::Byte => elements,
        CodeWidth::Nibble => ceil_div_u64(elements, 2),
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        if group_size >= 32 {
            lemma_small_group_count(elements as int, group_size as int);
        }
    }
    WireCounts {
        group_count,
        packed_value_bytes,
        scale_bytes: group_count.checked_mul(4),
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// With groups of at least 32 values, four bytes per group never overflow `u64`.
proof fn lemma_small_group_count(elements: int, group_size: int)
    requires
        0 <= elements <= u64::MAX,
        group_size >= 32,
    ensures
        4 * ceiling(elements, group_size) <= u64::MAX,
{
    lemma_ceiling_bounds(elements, group_size);
    let groups = ceiling(elements, group_size);
    // (groups - 1) * G < elements <= u64::MAX, so groups <= u64::MAX / 32 + 1.
    assert(groups == 0 || (groups - 1) * 32 < elements) by (nonlinear_arith)
        requires
            groups == 0 || (groups - 1) * group_size < elements,
            group_size >= 32,
            groups >= 0,
    ;
}

} // verus!

/// Returns the Burn `[1, G]` block count `rows · (columns / G)` of a matrix, or `None` on overflow.
///
/// Verified: the exact product; and when `G` divides the width, the blocks
/// hold exactly `rows · columns` values and their count equals the flat group
/// count `⌈rows · columns / G⌉` that the persisted metadata records, so a
/// group-aligned artifact's scales map one-to-one onto Burn's blocks.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(blocks =>
    requires
        group_size > 0,
    ensures
        match blocks {
            Some(count) => {
                &&& count == rows * (columns / group_size)
                &&& columns % group_size == 0 ==> count * group_size == rows * columns
                &&& columns % group_size == 0
                    ==> count == ceiling(rows * columns, group_size as int)
            },
            None => rows * (columns / group_size) > usize::MAX,
        }
))]
pub(crate) fn block_scale_count(rows: usize, columns: usize, group_size: usize) -> Option<usize> {
    let count = rows.checked_mul(columns / group_size);
    #[cfg(verus_keep_ghost)]
    proof! {
        if columns % group_size == 0 {
            let per_row = columns / group_size;
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod(columns as int, group_size as int);
            assert(columns == group_size * per_row);
            assert(rows * per_row * group_size == rows * columns) by (nonlinear_arith)
                requires columns == group_size * per_row;
            vstd::arithmetic::div_mod::lemma_div_multiples_vanish(
                rows * per_row,
                group_size as int,
            );
            vstd::arithmetic::div_mod::lemma_mod_multiples_basic(
                rows * per_row,
                group_size as int,
            );
            assert((rows * per_row) * group_size == group_size * (rows * per_row)) by (nonlinear_arith);
        }
    }
    count
}

/// Returns the group, and so the scale index, of flattened element `element`: `element / G`.
///
/// Verified: the group's values `[g · G, (g + 1) · G)` contain the element,
/// and for an element of an `elements`-value tensor the index is below the
/// group count `⌈elements / G⌉`, so a validated scale vector is never indexed
/// out of bounds.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(group =>
    requires
        group_size > 0,
        element < elements,
    ensures
        group == element / group_size,
        group * group_size <= element < (group + 1) * group_size,
        group < ceiling(elements as int, group_size as int),
))]
// `elements` appears only in the verified specification.
#[cfg_attr(not(verus_keep_ghost), allow(unused_variables))]
pub(crate) fn group_index(element: usize, elements: usize, group_size: usize) -> usize {
    #[cfg(verus_keep_ghost)]
    proof! {
        let group = element / group_size;
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(element as int, group_size as int);
        assert(element == group_size * group + element % group_size);
        assert(group * group_size <= element) by (nonlinear_arith)
            requires
                element == group_size * group + element % group_size,
                element % group_size >= 0,
        ;
        assert(element < (group + 1) * group_size) by (nonlinear_arith)
            requires
                element == group_size * group + element % group_size,
                element % group_size < group_size,
        ;
        lemma_ceiling_bounds(elements as int, group_size as int);
        // `element < elements <= ceiling · G`, so `group · G <= element` forces `group < ceiling`.
        assert(group < ceiling(elements as int, group_size as int)) by (nonlinear_arith)
            requires
                group * group_size <= element,
                element < elements,
                elements <= ceiling(elements as int, group_size as int) * group_size,
                group_size >= 1,
        ;
    }
    element / group_size
}

/// Host lengths of a stored payload: packed codes followed by little-endian scales.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PayloadSplit {
    /// Packed code bytes, the payload prefix `[0, packed)`.
    pub(crate) packed: usize,
    /// Scale bytes, the payload suffix `[packed, packed + scales)`.
    pub(crate) scales: usize,
}

/// First reason a stored payload cannot be split into its packed codes and scales.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PayloadSplitError {
    /// The packed byte count does not fit `usize`.
    PackedTooLarge,
    /// The scale byte count does not fit `usize`.
    ScalesTooLarge,
    /// The combined length does not fit `usize`.
    CombinedOverflow,
    /// The payload length differs from the combined length.
    LengthMismatch {
        /// The combined length the metadata requires.
        expected: usize,
    },
}

/// Splits a `payload_len`-byte payload into `packed_bytes` codes followed by `scale_bytes` scales.
///
/// The checks run in order (packed host length, scale host length, combined
/// length, payload length) and the first failure is returned.
///
/// Verified: on success the split keeps both metadata counts exactly and
/// `packed + scales == payload_len`, so the prefix `[0, packed)` and the
/// suffix `[packed, payload_len)` tile the payload and the scale suffix holds
/// exactly `scales` bytes; each error names the first failed check, with
/// every earlier check proved to have passed.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(split =>
    ensures
        match split {
            Ok(lengths) => {
                &&& lengths.packed == packed_bytes
                &&& lengths.scales == scale_bytes
                &&& lengths.packed + lengths.scales == payload_len
            },
            Err(PayloadSplitError::PackedTooLarge) => packed_bytes > usize::MAX,
            Err(PayloadSplitError::ScalesTooLarge) => packed_bytes <= usize::MAX
                && scale_bytes > usize::MAX,
            Err(PayloadSplitError::CombinedOverflow) => packed_bytes <= usize::MAX
                && scale_bytes <= usize::MAX
                && packed_bytes + scale_bytes > usize::MAX,
            Err(PayloadSplitError::LengthMismatch { expected }) => {
                &&& expected == packed_bytes + scale_bytes
                &&& expected != payload_len
            },
        }
))]
pub(crate) fn payload_split(
    packed_bytes: u64,
    scale_bytes: u64,
    payload_len: usize,
) -> Result<PayloadSplit, PayloadSplitError> {
    if packed_bytes > usize::MAX as u64 {
        return Err(PayloadSplitError::PackedTooLarge);
    }
    if scale_bytes > usize::MAX as u64 {
        return Err(PayloadSplitError::ScalesTooLarge);
    }
    let packed = packed_bytes as usize;
    let scales = scale_bytes as usize;
    let Some(expected) = packed.checked_add(scales) else {
        return Err(PayloadSplitError::CombinedOverflow);
    };
    if payload_len != expected {
        return Err(PayloadSplitError::LengthMismatch { expected });
    }
    Ok(PayloadSplit { packed, scales })
}

#[cfg(test)]
#[path = "counts_tests.rs"]
mod tests;
