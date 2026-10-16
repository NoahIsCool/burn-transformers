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
//! The same view splits a shape around an axis for the runtime transforms
//! ([`axis_view`]), concatenation sums one axis while every other axis agrees
//! ([`extend_concatenation`]), dense host byte counts multiply the shape by the
//! scalar width ([`dense_byte_len`]), and a file's data section starts after the
//! eight-byte prefix and JSON header and runs to the end of the file
//! ([`data_section`]).
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

/// First byte not covered by a contiguous chain of the first `count` intervals.
///
/// Zero for the empty chain, otherwise the end of interval `count - 1`.
pub open spec fn chain_end(intervals: Seq<(u64, u64)>, count: int) -> int {
    if count <= 0 {
        0
    } else {
        intervals[count - 1].1 as int
    }
}

/// The first `count` intervals chain from zero inside `[0, data_bytes)`.
///
/// Interval `j` begins exactly where the chain before it ends, is not
/// reversed, and ends inside the data section.
pub open spec fn chains(intervals: Seq<(u64, u64)>, count: int, data_bytes: u64) -> bool {
    forall|j: int| 0 <= j < count ==> {
        &&& #[trigger] intervals[j].0 as int == chain_end(intervals, j)
        &&& intervals[j].0 <= intervals[j].1
        &&& intervals[j].1 <= data_bytes
    }
}

/// The sorted intervals tile `[0, data_bytes)` exactly.
pub open spec fn tiles(intervals: Seq<(u64, u64)>, data_bytes: u64) -> bool {
    &&& chains(intervals, intervals.len() as int, data_bytes)
    &&& chain_end(intervals, intervals.len() as int) == data_bytes
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
///
/// Verified: the result is `Ok` exactly when the intervals tile the section
/// (soundness and completeness), and each error names the first violation:
/// every interval before `index` chains correctly, and interval `index` is
/// reversed, begins away from the next uncovered byte `expected`, or ends past
/// the section; `Uncovered` means the whole chain is valid but ends at
/// `covered < data_bytes`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result is Ok <==> tiles(intervals@, data_bytes),
        match result {
            Ok(()) => true,
            Err(TilingError::Reversed { index }) => {
                &&& index < intervals@.len()
                &&& chains(intervals@, index as int, data_bytes)
                &&& intervals@[index as int].1 < intervals@[index as int].0
            },
            Err(TilingError::Discontiguous { index, expected }) => {
                &&& index < intervals@.len()
                &&& chains(intervals@, index as int, data_bytes)
                &&& expected as int == chain_end(intervals@, index as int)
                &&& intervals@[index as int].0 <= intervals@[index as int].1
                &&& intervals@[index as int].0 != expected
            },
            Err(TilingError::OutOfBounds { index }) => {
                &&& index < intervals@.len()
                &&& chains(intervals@, index as int, data_bytes)
                &&& intervals@[index as int].0 as int == chain_end(intervals@, index as int)
                &&& intervals@[index as int].0 <= intervals@[index as int].1
                &&& intervals@[index as int].1 > data_bytes
            },
            Err(TilingError::Uncovered { covered }) => {
                &&& chains(intervals@, intervals@.len() as int, data_bytes)
                &&& covered as int == chain_end(intervals@, intervals@.len() as int)
                &&& covered < data_bytes
            },
        },
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
            cursor as int == chain_end(intervals@, index as int),
            chains(intervals@, index as int, data_bytes),
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

// Axis views, slice ranges, concatenation shapes, host sizes, and data sections.

#[cfg(verus_keep_ghost)]
verus! {

/// Mathematical product of `shape[start..end]`, one for an empty range.
pub open spec fn range_product(shape: Seq<u64>, start: int, end: int) -> int
    decreases end - start,
{
    if end <= start {
        1
    } else {
        range_product(shape, start, end - 1) * shape[end - 1] as int
    }
}

/// Every left-to-right partial product of `shape[start..end]` fits `u64`.
pub open spec fn range_fits(shape: Seq<u64>, start: int, end: int) -> bool {
    forall|k: int| start <= k <= end ==> #[trigger] range_product(shape, start, k) <= u64::MAX
}

/// A product over `[a, c)` factors at any `b` in between.
proof fn lemma_range_product_split(shape: Seq<u64>, a: int, b: int, c: int)
    requires
        a <= b <= c,
    ensures
        range_product(shape, a, c) == range_product(shape, a, b) * range_product(shape, b, c),
    decreases c - b,
{
    if c > b {
        lemma_range_product_split(shape, a, b, c - 1);
        let left = range_product(shape, a, b);
        let middle = range_product(shape, b, c - 1);
        let last = shape[c - 1] as int;
        assert(left * middle * last == left * (middle * last)) by (nonlinear_arith);
    }
}

/// The whole-shape product used by [`element_count`] is the product over `[0, len)`.
proof fn lemma_shape_product_is_range(shape: Seq<u64>, len: int)
    ensures
        shape_product(shape, len) == range_product(shape, 0, len),
    decreases len,
{
    if len > 0 {
        lemma_shape_product_is_range(shape, len - 1);
    }
}

/// Mathematical product of `shape[0..len]` over host dimensions.
pub open spec fn host_product(shape: Seq<usize>, len: int) -> int
    decreases len,
{
    if len <= 0 {
        1
    } else {
        host_product(shape, len - 1) * shape[len - 1] as int
    }
}

/// The first `count` dimensions of `old` and `new` agree everywhere except `axis`.
pub open spec fn other_axes_agree(old: Seq<u64>, new: Seq<u64>, axis: int, count: int) -> bool {
    forall|i: int| 0 <= i < count && i != axis ==> #[trigger] new[i] == old[i]
}

} // verus!

/// Computes `shape[start] * ... * shape[end - 1]`, or `None` when a running product overflows.
///
/// This is [`element_count`] applied to the subslice `shape[start..end]`: the
/// product accumulates left to right with checked multiplication, so it is
/// `None` exactly when some partial product of the range exceeds `u64::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(product =>
    requires
        start <= end <= shape@.len(),
    ensures
        match product {
            Some(n) => n as int == range_product(shape@, start as int, end as int)
                && range_fits(shape@, start as int, end as int),
            None => exists|k: int| start <= k <= end
                && #[trigger] range_product(shape@, start as int, k) > u64::MAX,
        }
))]
pub(crate) fn checked_range_product(shape: &[u64], start: usize, end: usize) -> Option<u64> {
    let mut product: u64 = 1;
    let mut index = start;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            start <= index <= end,
            end <= shape@.len(),
            product as int == range_product(shape@, start as int, index as int),
            forall|k: int| start <= k <= index
                ==> #[trigger] range_product(shape@, start as int, k) <= u64::MAX,
        decreases end - index,
    ))]
    while index < end {
        match product.checked_mul(shape[index]) {
            Some(next) => product = next,
            None => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(range_product(shape@, start as int, index as int + 1) > u64::MAX);
                }
                return None;
            }
        }
        index += 1;
    }
    Some(product)
}

/// A row-major shape viewed as `[outer, axis, inner]` around one axis, in host units.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AxisExtents {
    /// Product of the dimensions before the axis.
    pub(crate) outer: usize,
    /// Length of the axis itself.
    pub(crate) axis: usize,
    /// Product of the dimensions after the axis.
    pub(crate) inner: usize,
}

/// First reason a shape cannot be viewed as `[outer, axis, inner]` on this host.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AxisViewError {
    /// The axis is not below the rank.
    AxisOutOfBounds,
    /// A running product of the dimensions before the axis overflows `u64`.
    OuterOverflow,
    /// The product of the dimensions before the axis does not fit `usize`.
    OuterTooLarge {
        /// The exact product.
        value: u64,
    },
    /// The axis length does not fit `usize`.
    AxisTooLarge {
        /// The axis length.
        value: u64,
    },
    /// A running product of the dimensions after the axis overflows `u64`.
    InnerOverflow,
    /// The product of the dimensions after the axis does not fit `usize`.
    InnerTooLarge {
        /// The exact product.
        value: u64,
    },
    /// `outer * axis` or `outer * axis * inner` does not fit `usize`.
    ElementsTooLarge,
}

/// Splits `shape` around `axis` into host-sized `[outer, axis, inner]` extents.
///
/// The checks run in a fixed order (axis bound, outer product, outer host
/// size, axis host size, inner product, inner host size, element count), and
/// the first failure is returned.
///
/// Verified, as a complete specification (exactly one outcome holds for every
/// input): on success every running product before and after the axis fits
/// `u64` (so a zero dimension cannot hide an overflowing prefix), the extents
/// are the exact products of the dimensions before and after the axis and the
/// axis length itself, `outer * axis` and `outer * axis * inner` fit `usize`
/// (exactly [`slice_block`]'s preconditions for this view), and
/// `outer * axis * inner` is the shape's full product. Each error names the
/// first failed check, with every earlier check proved to have passed.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(view =>
    ensures
        match view {
            Ok(extents) => {
                &&& axis < shape@.len()
                &&& range_fits(shape@, 0, axis as int)
                &&& range_fits(shape@, axis as int + 1, shape@.len() as int)
                &&& extents.outer as int == range_product(shape@, 0, axis as int)
                &&& extents.axis == shape@[axis as int]
                &&& extents.inner as int
                    == range_product(shape@, axis as int + 1, shape@.len() as int)
                &&& extents.outer * extents.axis <= usize::MAX
                &&& extents.outer * extents.axis * extents.inner <= usize::MAX
                &&& extents.outer * extents.axis * extents.inner
                    == range_product(shape@, 0, shape@.len() as int)
                &&& extents.outer * extents.axis * extents.inner
                    == shape_product(shape@, shape@.len() as int)
            },
            Err(AxisViewError::AxisOutOfBounds) => axis >= shape@.len(),
            Err(AxisViewError::OuterOverflow) => axis < shape@.len()
                && !range_fits(shape@, 0, axis as int),
            Err(AxisViewError::OuterTooLarge { value }) => axis < shape@.len()
                && range_fits(shape@, 0, axis as int)
                && value as int == range_product(shape@, 0, axis as int)
                && value > usize::MAX,
            Err(AxisViewError::AxisTooLarge { value }) => axis < shape@.len()
                && range_fits(shape@, 0, axis as int)
                && range_product(shape@, 0, axis as int) <= usize::MAX
                && value == shape@[axis as int]
                && value > usize::MAX,
            Err(AxisViewError::InnerOverflow) => axis < shape@.len()
                && range_fits(shape@, 0, axis as int)
                && range_product(shape@, 0, axis as int) <= usize::MAX
                && shape@[axis as int] <= usize::MAX
                && !range_fits(shape@, axis as int + 1, shape@.len() as int),
            Err(AxisViewError::InnerTooLarge { value }) => axis < shape@.len()
                && range_fits(shape@, 0, axis as int)
                && range_product(shape@, 0, axis as int) <= usize::MAX
                && shape@[axis as int] <= usize::MAX
                && range_fits(shape@, axis as int + 1, shape@.len() as int)
                && value as int == range_product(shape@, axis as int + 1, shape@.len() as int)
                && value > usize::MAX,
            Err(AxisViewError::ElementsTooLarge) => axis < shape@.len()
                && range_fits(shape@, 0, axis as int)
                && range_fits(shape@, axis as int + 1, shape@.len() as int)
                && {
                    let outer = range_product(shape@, 0, axis as int);
                    let inner = range_product(shape@, axis as int + 1, shape@.len() as int);
                    &&& outer <= usize::MAX
                    &&& shape@[axis as int] <= usize::MAX
                    &&& inner <= usize::MAX
                    &&& (outer * shape@[axis as int] > usize::MAX
                        || outer * shape@[axis as int] * inner > usize::MAX)
                },
        }
))]
pub(crate) fn axis_view(shape: &[u64], axis: usize) -> Result<AxisExtents, AxisViewError> {
    if axis >= shape.len() {
        return Err(AxisViewError::AxisOutOfBounds);
    }
    let Some(outer) = checked_range_product(shape, 0, axis) else {
        return Err(AxisViewError::OuterOverflow);
    };
    if outer > usize::MAX as u64 {
        return Err(AxisViewError::OuterTooLarge { value: outer });
    }
    let axis_len = shape[axis];
    if axis_len > usize::MAX as u64 {
        return Err(AxisViewError::AxisTooLarge { value: axis_len });
    }
    let Some(inner) = checked_range_product(shape, axis + 1, shape.len()) else {
        return Err(AxisViewError::InnerOverflow);
    };
    if inner > usize::MAX as u64 {
        return Err(AxisViewError::InnerTooLarge { value: inner });
    }
    let extents = AxisExtents {
        outer: outer as usize,
        axis: axis_len as usize,
        inner: inner as usize,
    };
    let Some(_) = axis_elements(extents.outer, extents.axis, extents.inner) else {
        return Err(AxisViewError::ElementsTooLarge);
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_range_product_split(shape@, 0, axis as int, axis as int + 1);
        lemma_range_product_split(shape@, 0, axis as int + 1, shape@.len() as int);
        lemma_shape_product_is_range(shape@, shape@.len() as int);
        assert(range_product(shape@, axis as int, axis as int) == 1);
        assert(range_product(shape@, axis as int, axis as int + 1) == shape@[axis as int]);
    }
    Ok(extents)
}

/// Returns `outer * axis_len * inner`, or `None` when `outer * axis_len` or the whole product overflows.
///
/// Verified: `Some` exactly when both products fit `usize`, with the exact product.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(elements =>
    ensures
        match elements {
            Some(n) => outer * axis_len <= usize::MAX
                && outer * axis_len * inner <= usize::MAX
                && n == outer * axis_len * inner,
            None => outer * axis_len > usize::MAX || outer * axis_len * inner > usize::MAX,
        }
))]
pub(crate) fn axis_elements(outer: usize, axis_len: usize, inner: usize) -> Option<usize> {
    match outer.checked_mul(axis_len) {
        Some(rows) => rows.checked_mul(inner),
        None => None,
    }
}

/// Returns whether `[start, start + length)` lies inside `[0, limit)`, without overflow.
///
/// Verified: true exactly when `start + length <= limit` mathematically, so a
/// `true` answer supplies [`slice_block`]'s range precondition.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fits =>
    ensures
        fits == (start + length <= limit),
))]
pub(crate) fn range_within(start: usize, length: usize, limit: usize) -> bool {
    match start.checked_add(length) {
        Some(end) => end <= limit,
        None => false,
    }
}

/// Why a concatenation source cannot extend the running target shape.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConcatError {
    /// The source rank differs, or a non-axis dimension disagrees.
    Incompatible,
    /// The concatenated axis length exceeds `u64::MAX`.
    Overflow,
}

/// Adds one concatenation source to the running target `shape` along `axis`.
///
/// Verified: success exactly when the source has the target's rank, agrees on
/// every other axis, and the summed axis length fits `u64`; it then adds the
/// source's axis length to `shape[axis]` and changes nothing else. On either
/// error `shape` is unchanged, and `Incompatible` takes precedence.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        axis < old(shape)@.len(),
    ensures
        final(shape)@.len() == old(shape)@.len(),
        match result {
            Ok(()) => {
                &&& source@.len() == old(shape)@.len()
                &&& other_axes_agree(old(shape)@, source@, axis as int, source@.len() as int)
                &&& old(shape)@[axis as int] + source@[axis as int] <= u64::MAX
                &&& final(shape)@[axis as int]
                    == old(shape)@[axis as int] + source@[axis as int]
                &&& other_axes_agree(
                    old(shape)@,
                    final(shape)@,
                    axis as int,
                    old(shape)@.len() as int,
                )
            },
            Err(ConcatError::Incompatible) => {
                &&& !(source@.len() == old(shape)@.len()
                    && other_axes_agree(old(shape)@, source@, axis as int, source@.len() as int))
                &&& final(shape)@ == old(shape)@
            },
            Err(ConcatError::Overflow) => {
                &&& source@.len() == old(shape)@.len()
                &&& other_axes_agree(old(shape)@, source@, axis as int, source@.len() as int)
                &&& old(shape)@[axis as int] + source@[axis as int] > u64::MAX
                &&& final(shape)@ == old(shape)@
            },
        }
))]
pub(crate) fn extend_concatenation(
    shape: &mut [u64],
    source: &[u64],
    axis: usize,
) -> Result<(), ConcatError> {
    if source.len() != shape.len() {
        return Err(ConcatError::Incompatible);
    }
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= source@.len(),
            source@.len() == shape@.len(),
            axis < shape@.len(),
            other_axes_agree(shape@, source@, axis as int, index as int),
        decreases source@.len() - index,
    ))]
    while index < source.len() {
        if index != axis && source[index] != shape[index] {
            return Err(ConcatError::Incompatible);
        }
        index += 1;
    }
    let Some(total) = shape[axis].checked_add(source[axis]) else {
        return Err(ConcatError::Overflow);
    };
    shape[axis] = total;
    Ok(())
}

/// Computes the dense byte length `product(shape) * width`, or `None` on overflow.
///
/// The product accumulates left to right with checked multiplication (rank
/// zero is one scalar), then multiplies by the scalar width.
///
/// Verified: `Some` exactly when every running product and the final byte
/// count fit `usize`, with the exact byte count.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bytes =>
    ensures
        match bytes {
            Some(n) => {
                &&& n == host_product(shape@, shape@.len() as int) * width
                &&& forall|k: int| 0 <= k <= shape@.len()
                    ==> #[trigger] host_product(shape@, k) <= usize::MAX
            },
            None => {
                ||| exists|k: int| 0 <= k <= shape@.len()
                    && #[trigger] host_product(shape@, k) > usize::MAX
                ||| host_product(shape@, shape@.len() as int) * width > usize::MAX
            },
        }
))]
pub(crate) fn dense_byte_len(shape: &[usize], width: usize) -> Option<usize> {
    match checked_host_product(shape) {
        Some(elements) => elements.checked_mul(width),
        None => None,
    }
}

/// Computes `product(shape)` over host dimensions, or `None` when a running product overflows.
///
/// Verified: `Some` exactly when every left-to-right partial product fits
/// `usize`, with the exact product; rank zero is one scalar.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(count =>
    ensures
        match count {
            Some(n) => n as int == host_product(shape@, shape@.len() as int)
                && forall|k: int| 0 <= k <= shape@.len()
                    ==> #[trigger] host_product(shape@, k) <= usize::MAX,
            None => exists|k: int| 0 <= k <= shape@.len()
                && #[trigger] host_product(shape@, k) > usize::MAX,
        }
))]
pub(crate) fn checked_host_product(shape: &[usize]) -> Option<usize> {
    let mut product: usize = 1;
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= shape@.len(),
            product as int == host_product(shape@, index as int),
            forall|k: int| 0 <= k <= index ==> #[trigger] host_product(shape@, k) <= usize::MAX,
        decreases shape@.len() - index,
    ))]
    while index < shape.len() {
        match product.checked_mul(shape[index]) {
            Some(next) => product = next,
            None => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(host_product(shape@, index as int + 1) > usize::MAX);
                }
                return None;
            }
        }
        index += 1;
    }
    Some(product)
}

/// The data section of a SafeTensors file: its absolute start and byte length.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DataSection {
    /// Absolute offset of the first data byte, `8 + header_bytes`.
    pub(crate) start: u64,
    /// Bytes from `start` to the end of the file.
    pub(crate) bytes: u64,
}

/// Why a header length cannot delimit a data section.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DataSectionError {
    /// `8 + header_bytes` exceeds `u64::MAX`.
    Overflow,
    /// The header is empty or extends past the end of the file.
    InvalidHeaderLength,
}

/// Locates the data section of a `file_bytes`-byte file with a `header_bytes` JSON header.
///
/// Verified: success exactly when the header is nonempty and the eight-byte
/// prefix plus header fit in the file; the section then starts at
/// `8 + header_bytes` and its length reaches exactly the end of the file.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(section =>
    ensures
        match section {
            Ok(found) => {
                &&& header_bytes > 0
                &&& 8 + header_bytes <= file_bytes
                &&& found.start == 8 + header_bytes
                &&& found.start + found.bytes == file_bytes
            },
            Err(DataSectionError::Overflow) => 8 + header_bytes > u64::MAX,
            Err(DataSectionError::InvalidHeaderLength) => 8 + header_bytes <= u64::MAX
                && (header_bytes == 0 || 8 + header_bytes > file_bytes),
        }
))]
pub(crate) fn data_section(
    header_bytes: u64,
    file_bytes: u64,
) -> Result<DataSection, DataSectionError> {
    let Some(start) = 8_u64.checked_add(header_bytes) else {
        return Err(DataSectionError::Overflow);
    };
    if header_bytes == 0 || start > file_bytes {
        return Err(DataSectionError::InvalidHeaderLength);
    }
    Ok(DataSection {
        start,
        bytes: file_bytes - start,
    })
}

#[cfg(test)]
#[path = "layout_tests.rs"]
mod tests;
