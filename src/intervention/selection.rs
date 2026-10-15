//! Pure index arithmetic behind semantic activation selections.
//!
//! A selection names semantic axes (`Token`, `Head`, ...) and half-open ranges.
//! Before execution, each requested axis is resolved to its position in the
//! point's declared axis list; at execution, the ranges are checked against the
//! emitted tensor's extents and expanded to one range per tensor dimension,
//! with unselected dimensions kept whole. This module holds only integer logic,
//! free of Burn types and allocation-dependent behavior, so Verus checks these
//! exact bodies through `verus/experiment.rs` (see `verus/README.md`).
//!
//! # Verified properties
//!
//! - [`unit_range`] selects exactly `index..index + 1`, or an empty range at
//!   `usize::MAX` that every later check rejects, without overflow.
//! - [`first_ill_formed_slice`] returns `None` exactly when every slice names an
//!   axis that occurs once in the point, no axis is selected twice, and every
//!   range is nonempty; otherwise it returns the first offending slice.
//! - [`selection_ranges`] returns, for every dimension `d`, exactly the range
//!   of the last slice naming `d` (nonempty, with `end <= extents[d]`), or the
//!   whole `0..extents[d]` when no slice names `d`; otherwise it names the
//!   first slice outside its extent.

use core::ops::Range;

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// One requested slice after its semantic axis was resolved to a tensor dimension.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResolvedSlice {
    /// Tensor dimension holding the requested semantic axis.
    pub(crate) dimension: usize,
    /// Number of times the requested semantic axis occurs in the point's axes.
    pub(crate) occurrences: usize,
    /// Requested half-open range along that dimension.
    pub(crate) range: Range<usize>,
}

/// Selects the length-one range `index..index + 1` without overflowing.
///
/// At `usize::MAX` the result is the empty range `index..index`, which
/// [`first_ill_formed_slice`] rejects during scope validation instead of
/// panicking during registration.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(range =>
    ensures
        range.start == index,
        index < usize::MAX ==> range.end == index + 1,
        index == usize::MAX ==> range.end == index,
))]
pub(crate) fn unit_range(index: usize) -> Range<usize> {
    if index < usize::MAX {
        index..index + 1
    } else {
        index..index
    }
}

/// Finds the first resolved slice that is ambiguous, empty, or repeats an axis.
///
/// `slices[i].occurrences` counts how often slice `i`'s semantic axis appears
/// in the point; anything but one is ambiguous or absent. Two slices resolving
/// to the same dimension select one axis twice. `None` means every slice is
/// well formed.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(first =>
    ensures
        match first {
            None => forall|i: int| 0 <= i < slices.len() ==> slice_well_formed(slices@, i),
            Some(index) => {
                &&& index < slices.len()
                &&& !slice_well_formed(slices@, index as int)
                &&& forall|i: int| 0 <= i < index ==> slice_well_formed(slices@, i)
            },
        },
))]
pub(crate) fn first_ill_formed_slice(slices: &[ResolvedSlice]) -> Option<usize> {
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            0 <= index <= slices.len(),
            forall|i: int| 0 <= i < index ==> slice_well_formed(slices@, i),
        decreases slices.len() - index,
    ))]
    while index < slices.len() {
        let dimension = slices[index].dimension;
        if slices[index].occurrences != 1 || slices[index].range.start >= slices[index].range.end {
            return Some(index);
        }
        let mut previous = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                0 <= previous <= index,
                index < slices.len(),
                dimension == slices@[index as int].dimension,
                forall|i: int| 0 <= i < index ==> slice_well_formed(slices@, i),
                forall|j: int| 0 <= j < previous ==> slices@[j].dimension != dimension,
            decreases index - previous,
        ))]
        while previous < index {
            if slices[previous].dimension == dimension {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(slices@[previous as int].dimension == slices@[index as int].dimension);
                    assert(!slice_well_formed(slices@, index as int));
                }
                return Some(index);
            }
            previous += 1;
        }
        index += 1;
    }
    None
}

/// Expands resolved slices to one range per tensor dimension, checked against `extents`.
///
/// Each selected dimension receives the range of the last slice naming it,
/// and unselected dimensions keep `0..extents[d]`. `Err(i)` names the first
/// slice whose dimension lies outside `extents` or whose range is empty or
/// extends past its extent; well-formedness is checked separately by
/// [`first_ill_formed_slice`], so in practice each dimension is named at most
/// once. Every returned range lies inside its extent.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(ranges =>
    ensures
        match ranges {
            Err(index) => {
                &&& index < slices.len()
                &&& !slice_fits(extents@, slices@[index as int])
                &&& forall|i: int| 0 <= i < index ==> slice_fits(extents@, slices@[i])
            },
            Ok(ranges) => {
                &&& slices_fit(extents@, slices@)
                &&& ranges.len() == extents.len()
                &&& forall|d: int| 0 <= d < extents.len() ==> range_inside(ranges@[d], extents@[d])
                &&& forall|d: int| 0 <= d < extents.len()
                    ==> range_selected(extents@[d], slices@, d, slices.len() as int, ranges@[d])
            },
        },
))]
pub(crate) fn selection_ranges(
    extents: &[usize],
    slices: &[ResolvedSlice],
) -> Result<Vec<Range<usize>>, usize> {
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            0 <= index <= slices.len(),
            forall|i: int| 0 <= i < index ==> slice_fits(extents@, slices@[i]),
        decreases slices.len() - index,
    ))]
    while index < slices.len() {
        let slice = &slices[index];
        if slice.dimension >= extents.len()
            || slice.range.start >= slice.range.end
            || slice.range.end > extents[slice.dimension]
        {
            return Err(index);
        }
        index += 1;
    }
    let mut ranges: Vec<Range<usize>> = Vec::new();
    let mut dimension = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            0 <= dimension <= extents.len(),
            ranges.len() == dimension,
            slices_fit(extents@, slices@),
            forall|d: int| 0 <= d < dimension ==> range_inside(ranges@[d], extents@[d]),
            forall|d: int| 0 <= d < dimension
                ==> range_selected(extents@[d], slices@, d, slices.len() as int, ranges@[d]),
        decreases extents.len() - dimension,
    ))]
    while dimension < extents.len() {
        let extent = extents[dimension];
        let mut range = 0..extent;
        let mut candidate = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                0 <= candidate <= slices.len(),
                dimension < extents.len(),
                extent == extents@[dimension as int],
                slices_fit(extents@, slices@),
                range_inside(range, extent),
                range_selected(extent, slices@, dimension as int, candidate as int, range),
            decreases slices.len() - candidate,
        ))]
        while candidate < slices.len() {
            if slices[candidate].dimension == dimension {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(slice_fits(extents@, slices@[candidate as int]));
                }
                range = slices[candidate].range.start..slices[candidate].range.end;
            }
            candidate += 1;
        }
        ranges.push(range);
        dimension += 1;
    }
    Ok(ranges)
}

#[cfg(verus_keep_ghost)]
verus! {

/// Whether one resolved slice names a dimension of `extents` and a nonempty range inside it.
pub(crate) open spec fn slice_fits(extents: Seq<usize>, slice: ResolvedSlice) -> bool {
    &&& slice.dimension < extents.len()
    &&& slice.range.start < slice.range.end
    &&& slice.range.end <= extents[slice.dimension as int]
}

/// Index of the last of the first `count` slices naming `dimension`, or `-1` if none does.
pub(crate) open spec fn last_naming(slices: Seq<ResolvedSlice>, dimension: int, count: int) -> int
    decreases count,
{
    if count <= 0 {
        -1
    } else if slices[count - 1].dimension as int == dimension {
        count - 1
    } else {
        last_naming(slices, dimension, count - 1)
    }
}

/// Whether `range` is what the first `count` slices select along `dimension`.
///
/// That is the range of the last slice naming `dimension`, or the whole
/// `0..extent` when none of them names it.
pub(crate) open spec fn range_selected(
    extent: usize,
    slices: Seq<ResolvedSlice>,
    dimension: int,
    count: int,
    range: Range<usize>,
) -> bool {
    let last = last_naming(slices, dimension, count);
    if last < 0 {
        range.start == 0 && range.end == extent
    } else {
        range.start == slices[last].range.start && range.end == slices[last].range.end
    }
}

/// Whether `range` is the whole extent or a nonempty subrange of it.
pub(crate) open spec fn range_inside(range: Range<usize>, extent: usize) -> bool {
    ||| (range.start == 0 && range.end == extent)
    ||| (range.start < range.end && range.end <= extent)
}

/// Whether slice `i` names a singly declared axis, a nonempty range, and no earlier dimension.
pub(crate) open spec fn slice_well_formed(slices: Seq<ResolvedSlice>, i: int) -> bool {
    &&& slices[i].occurrences == 1
    &&& slices[i].range.start < slices[i].range.end
    &&& forall|j: int| 0 <= j < i ==> slices[j].dimension != slices[i].dimension
}

/// Whether every resolved slice fits the emitted tensor's extents.
pub(crate) open spec fn slices_fit(extents: Seq<usize>, slices: Seq<ResolvedSlice>) -> bool {
    forall|i: int| 0 <= i < slices.len() ==> slice_fits(extents, slices[i])
}

} // verus!

#[cfg(test)]
#[path = "selection_tests.rs"]
mod tests;
