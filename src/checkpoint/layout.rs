//! Pure integer arithmetic for checkpoint tensor layouts.
//!
//! A tensor with dimensions `d[0..rank]` holds `N = product(d)` scalars (one for
//! rank zero) and occupies `N * bits / 8` bytes, which must be a whole number.
//! A SafeTensors data section of `D` bytes is valid only when the tensors' byte
//! intervals `[begin, end)`, sorted by `(begin, end)`, tile it exactly: the first
//! begins at zero, each begins where the previous ended, and the last ends at
//! `D`. Tiling implies the intervals are pairwise disjoint (apart from empty
//! ones), lie inside `[0, D)`, and account for every data byte.
//!
//! Row-major slicing and concatenation along an axis view a tensor as
//! `[outer, axis, inner]`, where `outer` and `inner` are the products of the
//! dimensions before and after that axis. Outer row `o` of a slice
//! `[start, start + length)` reads source elements
//! `[(o * axis + start) * inner, (o * axis + start + length) * inner)` and writes
//! target elements `[o * length * inner, (o + 1) * length * inner)`.
//!
//! The module has no crate, Burn, or allocator dependencies so that Verus can
//! check its arithmetic in place through `verus/checkpoint.rs`. Callers map the
//! small error enums below to contextual [`super::CheckpointError`] values.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

use core::ops::Range;

#[cfg(verus_keep_ghost)]
verus! {

/// Mathematical product of `shape[0..len]`, the scalar count of that prefix.
pub open spec fn shape_product(shape: Seq<u64>, len: int) -> int
    decreases len,
{
    if len <= 0 {
        1
    } else {
        shape_product(shape, len - 1) * shape[len - 1] as int
    }
}

/// Total bytes accounted for by the first `count` intervals: `sum(end - begin)`.
pub open spec fn covered_bytes(intervals: Seq<(u64, u64)>, count: int) -> int
    decreases count,
{
    if count <= 0 {
        0
    } else {
        covered_bytes(intervals, count - 1) + intervals[count - 1].1 - intervals[count - 1].0
    }
}

} // verus!

/// Computes the scalar count of a shape, or `None` when a running product overflows.
///
/// Rank zero is one scalar. The product accumulates left to right with checked
/// multiplication, exactly as the pinned SafeTensors 0.7.0 validator
/// (`try_fold(1, checked_mul)`) and Burn's `Shape::num_elements` do, so a shape
/// is accepted only when every prefix product fits in `u64`. `[u64::MAX, 2, 0]`
/// is therefore rejected although its mathematical product is zero: accepting it
/// would admit headers the reference rejects and shapes whose element count
/// overflows inside Burn.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(count =>
    ensures
        match count {
            Some(n) => n as int == shape_product(shape@, shape@.len() as int)
                && forall|k: int| 0 <= k <= shape@.len()
                    ==> #[trigger] shape_product(shape@, k) <= u64::MAX,
            None => exists|k: int| 0 <= k <= shape@.len()
                && #[trigger] shape_product(shape@, k) > u64::MAX,
        }
))]
pub(crate) fn element_count(shape: &[u64]) -> Option<u64> {
    let mut product: u64 = 1;
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= shape@.len(),
            product as int == shape_product(shape@, index as int),
            forall|k: int| 0 <= k <= index ==> #[trigger] shape_product(shape@, k) <= u64::MAX,
        decreases shape@.len() - index,
    ))]
    while index < shape.len() {
        match product.checked_mul(shape[index]) {
            Some(next) => product = next,
            None => {
                // The prefix through `index` is the overflow witness.
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(shape_product(shape@, index as int + 1) > u64::MAX);
                }
                return None;
            }
        }
        index += 1;
    }
    Some(product)
}

/// Why a scalar count cannot be stored as a whole number of bytes.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StorageSizeError {
    /// `elements * bits` exceeds `u64::MAX`.
    Overflow,
    /// `elements * bits` is not a multiple of eight.
    PartialByte,
}

/// Computes `elements * bits / 8`, rejecting overflow and a trailing partial byte.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bytes =>
    ensures
        match bytes {
            Ok(n) => elements * bits <= u64::MAX
                && (elements * bits) % 8 == 0
                && 8 * n == elements * bits,
            Err(StorageSizeError::Overflow) => elements * bits > u64::MAX,
            Err(StorageSizeError::PartialByte) => elements * bits <= u64::MAX
                && (elements * bits) % 8 != 0,
        }
))]
pub(crate) fn storage_bytes(elements: u64, bits: u64) -> Result<u64, StorageSizeError> {
    let Some(total_bits) = elements.checked_mul(bits) else {
        return Err(StorageSizeError::Overflow);
    };
    if total_bits % 8 != 0 {
        return Err(StorageSizeError::PartialByte);
    }
    Ok(total_bits / 8)
}

/// First violation of an exact contiguous tiling of a data section.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TilingError {
    /// Interval `index` ends before it begins.
    Reversed {
        /// Position of the interval in the sorted input.
        index: usize,
    },
    /// Interval `index` begins away from the next uncovered byte (a gap or overlap).
    Discontiguous {
        /// Position of the interval in the sorted input.
        index: usize,
        /// First uncovered byte when the interval was reached.
        expected: u64,
    },
    /// Interval `index` ends beyond the data section.
    OutOfBounds {
        /// Position of the interval in the sorted input.
        index: usize,
    },
    /// The intervals stop before the end of the data section.
    Uncovered {
        /// Number of bytes covered by the complete tiling.
        covered: u64,
    },
}

/// Checks that sorted byte intervals tile `[0, data_bytes)` exactly.
///
/// `intervals` must be sorted by `(begin, end)`; the caller sorts them because
/// SafeTensors headers list tensors by name. On success, interval `k` begins at
/// the end of interval `k - 1` (zero for the first), and the last ends at
/// `data_bytes`, so the intervals are disjoint and cover every data byte: their
/// lengths sum exactly to `data_bytes` (the byte accounting is exact). Empty
/// intervals are allowed anywhere inside the section.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result is Ok ==> {
            &&& intervals@.len() == 0 ==> data_bytes == 0
            &&& intervals@.len() > 0 ==> intervals@[0].0 == 0
            &&& intervals@.len() > 0 ==> intervals@[intervals@.len() - 1].1 == data_bytes
            &&& forall|k: int| 0 <= k < intervals@.len() ==>
                #[trigger] intervals@[k].0 <= intervals@[k].1 && intervals@[k].1 <= data_bytes
            &&& forall|k: int| 0 < k < intervals@.len() ==>
                #[trigger] intervals@[k].0 == intervals@[k - 1].1
            &&& covered_bytes(intervals@, intervals@.len() as int) == data_bytes
        }
))]
pub(crate) fn validate_contiguous_tiling(
    intervals: &[(u64, u64)],
    data_bytes: u64,
) -> Result<(), TilingError> {
    let mut cursor: u64 = 0;
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= intervals@.len(),
            cursor <= data_bytes,
            index == 0 ==> cursor == 0,
            index > 0 ==> cursor == intervals@[index - 1].1,
            forall|k: int| 0 <= k < index ==>
                #[trigger] intervals@[k].0 <= intervals@[k].1 && intervals@[k].1 <= data_bytes,
            forall|k: int| 0 < k < index ==> #[trigger] intervals@[k].0 == intervals@[k - 1].1,
            index > 0 ==> intervals@[0].0 == 0,
            cursor as int == covered_bytes(intervals@, index as int),
        decreases intervals@.len() - index,
    ))]
    while index < intervals.len() {
        let (begin, end) = intervals[index];
        if end < begin {
            return Err(TilingError::Reversed { index });
        }
        if begin != cursor {
            return Err(TilingError::Discontiguous {
                index,
                expected: cursor,
            });
        }
        if end > data_bytes {
            return Err(TilingError::OutOfBounds { index });
        }
        cursor = end;
        index += 1;
    }
    if cursor != data_bytes {
        return Err(TilingError::Uncovered { covered: cursor });
    }
    Ok(())
}

/// Returns the absolute file offset `8 + header_bytes + begin` of a tensor payload.
///
/// SafeTensors files start with an eight-byte little-endian header length, then
/// the JSON header, then the data section addressed by tensor `data_offsets`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(offset =>
    ensures
        match offset {
            Some(n) => n == 8 + header_bytes + begin,
            None => 8 + header_bytes + begin > u64::MAX,
        }
))]
pub(crate) fn tensor_file_offset(header_bytes: u64, begin: u64) -> Option<u64> {
    match 8_u64.checked_add(header_bytes) {
        Some(data_start) => data_start.checked_add(begin),
        None => None,
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// Sum of the first `count` section lengths: the start of section `count`.
pub open spec fn prefix_sum(lengths: Seq<u64>, count: int) -> int
    decreases count,
{
    if count <= 0 {
        0
    } else {
        prefix_sum(lengths, count - 1) + lengths[count - 1] as int
    }
}

/// Prefix sums of unsigned lengths never decrease.
proof fn lemma_prefix_sum_monotone(lengths: Seq<u64>, shorter: int, longer: int)
    requires
        0 <= shorter <= longer <= lengths.len(),
    ensures
        prefix_sum(lengths, shorter) <= prefix_sum(lengths, longer),
    decreases longer - shorter,
{
    if longer > shorter {
        lemma_prefix_sum_monotone(lengths, shorter, longer - 1);
    }
}

} // verus!

/// Why consecutive sections cannot partition an axis.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SectionError {
    /// The section lengths sum past `u64::MAX`.
    Overflow,
    /// The section lengths sum to `total`, not the axis length.
    LengthMismatch {
        /// Exact sum of all section lengths.
        total: u64,
    },
}

/// Returns the start of each consecutive section of an axis of length `axis_len`.
///
/// Section `k` covers `[starts[k], starts[k] + lengths[k])` with
/// `starts[k] = lengths[0] + ... + lengths[k - 1]`. Success requires the lengths
/// to sum exactly to `axis_len`, so the sections partition the axis in order and
/// none extends past it; zero-length sections are allowed.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Ok(starts) => {
                &&& starts@.len() == lengths@.len()
                &&& prefix_sum(lengths@, lengths@.len() as int) == axis_len
                &&& forall|k: int| 0 <= k < starts@.len() ==>
                    #[trigger] starts@[k] == prefix_sum(lengths@, k)
                &&& forall|k: int| 0 <= k < starts@.len() ==>
                    #[trigger] starts@[k] + lengths@[k] <= axis_len
            },
            Err(SectionError::Overflow) => prefix_sum(lengths@, lengths@.len() as int) > u64::MAX,
            Err(SectionError::LengthMismatch { total }) => {
                &&& total == prefix_sum(lengths@, lengths@.len() as int)
                &&& total != axis_len
            },
        }
))]
pub(crate) fn section_starts(lengths: &[u64], axis_len: u64) -> Result<Vec<u64>, SectionError> {
    let mut starts = Vec::new();
    let mut cursor: u64 = 0;
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= lengths@.len(),
            cursor as int == prefix_sum(lengths@, index as int),
            starts@.len() == index,
            forall|k: int| 0 <= k < index ==> #[trigger] starts@[k] == prefix_sum(lengths@, k),
        decreases lengths@.len() - index,
    ))]
    while index < lengths.len() {
        starts.push(cursor);
        match cursor.checked_add(lengths[index]) {
            Some(next) => cursor = next,
            None => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    lemma_prefix_sum_monotone(
                        lengths@,
                        index as int + 1,
                        lengths@.len() as int,
                    );
                }
                return Err(SectionError::Overflow);
            }
        }
        index += 1;
    }
    if cursor != axis_len {
        return Err(SectionError::LengthMismatch { total: cursor });
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|k: int| 0 <= k < starts@.len() implies
            #[trigger] starts@[k] + lengths@[k] <= axis_len
        by {
            lemma_prefix_sum_monotone(lengths@, k + 1, lengths@.len() as int);
        }
    }
    Ok(starts)
}

/// Element intervals for one outer row of an axis slice.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BlockCopy {
    /// Source element interval read for this outer row.
    pub(crate) source: Range<usize>,
    /// Target element interval written for this outer row.
    pub(crate) target: Range<usize>,
}

/// Locates outer row `outer_index` of `[outer, axis, inner]` sliced to `[start, start + length)`.
///
/// The caller proves `start + length <= axis` and that `outer * axis` and the
/// whole source, `outer * axis * inner` elements, are addressable; every
/// intermediate value is then bounded by those totals, so none of the products
/// below can overflow. The rows' target intervals are consecutive and partition
/// `[0, outer * length * inner)`, which lets a gather write the output strictly
/// in order.
///
/// Concatenation is the same map with roles exchanged: a source with `length`
/// rows along the axis, placed at `start` in a target of `axis` rows, reads the
/// returned `target` interval and writes the returned `source` interval.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(copy =>
    requires
        outer_index < outer,
        start + length <= axis,
        outer * axis <= usize::MAX,
        outer * axis * inner <= usize::MAX,
    ensures
        copy.source.start == (outer_index * axis + start) * inner,
        copy.source.end == copy.source.start + length * inner,
        copy.source.end <= outer * axis * inner,
        copy.target.start == outer_index * length * inner,
        copy.target.end == (outer_index + 1) * length * inner,
        copy.target.end <= outer * length * inner,
))]
// `outer` appears only in the verified precondition.
#[cfg_attr(not(verus_keep_ghost), allow(unused_variables))]
pub(crate) fn slice_block(
    outer_index: usize,
    outer: usize,
    axis: usize,
    inner: usize,
    start: usize,
    length: usize,
) -> BlockCopy {
    #[cfg(verus_keep_ghost)]
    proof! {
        slice_block_bounds(outer_index, outer, axis, inner, start, length);
    }
    let source_start = (outer_index * axis + start) * inner;
    let block = length * inner;
    let target_start = outer_index * block;
    BlockCopy {
        source: source_start..source_start + block,
        target: target_start..target_start + block,
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// Bounds every intermediate product of [`slice_block`] by the source size.
proof fn slice_block_bounds(
    outer_index: usize,
    outer: usize,
    axis: usize,
    inner: usize,
    start: usize,
    length: usize,
)
    requires
        outer_index < outer,
        start + length <= axis,
        outer * axis <= usize::MAX,
        outer * axis * inner <= usize::MAX,
    ensures
        outer_index * axis + start + length <= outer * axis,
        (outer_index * axis + start) * inner + length * inner <= outer * axis * inner,
        outer_index * length * inner + length * inner <= outer * length * inner,
        outer * length * inner <= outer * axis * inner,
        (outer_index * axis + start) * inner == outer_index * axis * inner + start * inner,
        outer_index * (length * inner) == outer_index * length * inner,
        (outer_index + 1) * length * inner == outer_index * length * inner + length * inner,
{
    assert(outer_index * axis + start + length <= outer * axis) by (nonlinear_arith)
        requires
            outer_index < outer,
            start + length <= axis,
    ;
    assert((outer_index * axis + start) * inner + length * inner <= outer * axis * inner)
        by (nonlinear_arith)
        requires
            outer_index * axis + start + length <= outer * axis,
    ;
    assert(outer_index * length * inner + length * inner <= outer * length * inner)
        by (nonlinear_arith)
        requires
            outer_index < outer,
    ;
    assert(outer * length * inner <= outer * axis * inner) by (nonlinear_arith)
        requires
            length <= axis,
    ;
    assert((outer_index * axis + start) * inner == outer_index * axis * inner + start * inner)
        by (nonlinear_arith);
    assert(outer_index * (length * inner) == outer_index * length * inner) by (nonlinear_arith);
    assert((outer_index + 1) * length * inner == outer_index * length * inner + length * inner)
        by (nonlinear_arith);
}

} // verus!

#[cfg(test)]
#[path = "layout_tests.rs"]
mod tests;
