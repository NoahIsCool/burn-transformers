//! Verified integer planning shared by the tensor-backed caches.
//!
//! Every tensor cache validates the same few integer facts before it touches
//! a Burn tensor:
//!
//! * **Allocation sizes.** A shape `d_0, …, d_{r-1}` holds
//!   `N = d_0 · … · d_{r-1}` scalars (one for rank zero) and `N · s` bytes for
//!   an element size `s`. The running product is checked left to right, so a
//!   shape is accepted only when every prefix product fits in `usize`; `N`
//!   must also stay within the portable signed 32-bit index range that Burn
//!   backends and selector tensors use. `k` repeated tensors hold `N · k`
//!   scalars and `N · k · s` bytes.
//! * **Write ends.** Appending `t` tokens at absolute `s` ends at `s + t`,
//!   which must be representable and at most the logical capacity; staging
//!   `t` more entries after `s` staged ones likewise needs `s + t` within
//!   the staging capacity.
//! * **Staged commits.** Every layer must hold exactly the proposed number of
//!   staged tokens or snapshots per row, and every accepted prefix must lie
//!   within its row's proposal.
//! * **Rewind clearing, beam sources, and portable indices.** A truncate to
//!   zero clears exactly the rows that held tokens; a reorder names only
//!   in-batch sources; host index vectors fit Burn's `i32` selector type.
//!
//! The module has no Burn or allocator-sensitive dependencies so that Verus
//! checks it in place through `verus/cache.rs` (see `verus/README.md`).
//! Callers map the small violation enums below to contextual
//! [`super::TensorCacheError`] values, so the proofs cover the decision and
//! its diagnostic operands while the error vocabulary stays with the caller.
//!
//! # Verified behavior
//!
//! For every input, each function returns exactly the documented value, fails
//! exactly when its condition holds, and reports the first violation in the
//! order the original scans used. No machine arithmetic overflows.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Largest per-tensor element count accepted before calling a Burn backend.
///
/// Burn backends and selector tensors commonly use signed 32-bit indices. The
/// conservative bound avoids backend-specific truncation even on 64-bit hosts.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub const PORTABLE_TENSOR_ELEMENT_LIMIT: usize = i32::MAX as usize;

#[cfg(verus_keep_ghost)]
verus! {

/// Mathematical product of `dimensions[0..len]`, the scalar count of that prefix.
pub open spec fn dimension_product(dimensions: Seq<usize>, len: int) -> int
    decreases len,
{
    if len <= 0 {
        1
    } else {
        dimension_product(dimensions, len - 1) * dimensions[len - 1] as int
    }
}

/// Every prefix product of `dimensions` fits in `usize`.
pub open spec fn prefixes_fit(dimensions: Seq<usize>) -> bool {
    forall|k: int| 0 <= k <= dimensions.len() ==> #[trigger] dimension_product(dimensions, k)
        <= usize::MAX
}

/// Every check of a staged commit scan before `(layer, sequence)` passed.
///
/// The scan visits layers in order and, within a layer, rows in order; at
/// each row it first compares the staged count with the proposal, then the
/// accepted prefix with the proposal.
pub open spec fn staged_commit_passes_before(
    staged: Seq<Vec<usize>>,
    proposed: Seq<usize>,
    accepted: Seq<usize>,
    layer: int,
    sequence: int,
) -> bool {
    &&& forall|l: int, s: int| 0 <= l < layer && 0 <= s < proposed.len()
        ==> #[trigger] staged[l]@[s] == proposed[s]
    &&& forall|s: int| 0 <= s < sequence ==> #[trigger] staged[layer]@[s] == proposed[s]
    &&& forall|s: int| 0 <= s < proposed.len() && (layer > 0 || s < sequence)
        ==> #[trigger] accepted[s] <= proposed[s]
}

} // verus!

/// Integer reason an allocation geometry is rejected.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocationViolation {
    /// A running scalar-count product overflowed `usize`.
    ShapeOverflow,
    /// The scalar count exceeds [`PORTABLE_TENSOR_ELEMENT_LIMIT`].
    ElementLimitExceeded {
        /// Scalar count of the rejected shape.
        elements: usize,
    },
    /// `elements * element_size` bytes overflow `usize`.
    ByteOverflow {
        /// Scalar count whose byte size overflowed.
        elements: usize,
    },
}

/// Returns the scalar count of one tensor shape after checking its byte size.
///
/// Proved: the count is the exact product of `dimensions`; the shape is
/// rejected with `ShapeOverflow` exactly when some running product overflows,
/// otherwise with `ElementLimitExceeded` exactly when the product exceeds the
/// portable limit, otherwise with `ByteOverflow` exactly when the byte size
/// `product * element_size` overflows.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Ok(elements) => {
                &&& prefixes_fit(dimensions@)
                &&& elements as int == dimension_product(dimensions@, dimensions@.len() as int)
                &&& elements <= PORTABLE_TENSOR_ELEMENT_LIMIT
                &&& elements * element_size <= usize::MAX
            },
            Err(AllocationViolation::ShapeOverflow) => !prefixes_fit(dimensions@),
            Err(AllocationViolation::ElementLimitExceeded { elements }) => {
                &&& prefixes_fit(dimensions@)
                &&& elements as int == dimension_product(dimensions@, dimensions@.len() as int)
                &&& elements > PORTABLE_TENSOR_ELEMENT_LIMIT
            },
            Err(AllocationViolation::ByteOverflow { elements }) => {
                &&& prefixes_fit(dimensions@)
                &&& elements as int == dimension_product(dimensions@, dimensions@.len() as int)
                &&& elements <= PORTABLE_TENSOR_ELEMENT_LIMIT
                &&& elements * element_size > usize::MAX
            },
        }
))]
pub(crate) fn tensor_elements(
    dimensions: &[usize],
    element_size: usize,
) -> Result<usize, AllocationViolation> {
    let mut elements: usize = 1;
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= dimensions@.len(),
            elements as int == dimension_product(dimensions@, index as int),
            forall|k: int| 0 <= k <= index ==> #[trigger] dimension_product(dimensions@, k)
                <= usize::MAX,
        decreases dimensions@.len() - index,
    ))]
    while index < dimensions.len() {
        match elements.checked_mul(dimensions[index]) {
            Some(next) => elements = next,
            None => {
                // The prefix through `index` is the overflow witness.
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(dimension_product(dimensions@, index as int + 1) > usize::MAX);
                }
                return Err(AllocationViolation::ShapeOverflow);
            }
        }
        index += 1;
    }
    if elements > PORTABLE_TENSOR_ELEMENT_LIMIT {
        return Err(AllocationViolation::ElementLimitExceeded { elements });
    }
    if elements.checked_mul(element_size).is_none() {
        return Err(AllocationViolation::ByteOverflow { elements });
    }
    Ok(elements)
}

/// Returns the scalar count of `tensor_count` equal allocations after checking bytes.
///
/// Proved: the count is exactly `elements_per_tensor * tensor_count`, rejected
/// with `ShapeOverflow` exactly when that product overflows and otherwise with
/// `ByteOverflow` exactly when its byte size overflows. The portable element
/// limit applies per tensor, so it is not rechecked for the aggregate.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Ok(elements) => {
                &&& elements == elements_per_tensor * tensor_count
                &&& elements * element_size <= usize::MAX
            },
            Err(AllocationViolation::ShapeOverflow) =>
                elements_per_tensor * tensor_count > usize::MAX,
            Err(AllocationViolation::ByteOverflow { elements }) => {
                &&& elements == elements_per_tensor * tensor_count
                &&& elements * element_size > usize::MAX
            },
            Err(AllocationViolation::ElementLimitExceeded { .. }) => false,
        }
))]
pub(crate) fn repeated_elements(
    elements_per_tensor: usize,
    tensor_count: usize,
    element_size: usize,
) -> Result<usize, AllocationViolation> {
    let Some(elements) = elements_per_tensor.checked_mul(tensor_count) else {
        return Err(AllocationViolation::ShapeOverflow);
    };
    if elements.checked_mul(element_size).is_none() {
        return Err(AllocationViolation::ByteOverflow { elements });
    }
    Ok(elements)
}

/// Integer reason an absolute append is rejected.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriteEndViolation {
    /// `start + tokens` overflows `usize`.
    Overflow,
    /// The exclusive end exceeds the logical capacity.
    ExceedsCapacity {
        /// Representable exclusive end `start + tokens`.
        end: usize,
    },
}

/// Returns the exclusive end `start + tokens` of an append within `capacity`.
///
/// Proved: the end is exact, and the append is rejected with `Overflow`
/// exactly when the sum is unrepresentable and otherwise with
/// `ExceedsCapacity` exactly when the end passes `capacity`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Ok(end) => end == start + tokens && end <= capacity,
            Err(WriteEndViolation::Overflow) => start + tokens > usize::MAX,
            Err(WriteEndViolation::ExceedsCapacity { end }) =>
                end == start + tokens && end > capacity,
        }
))]
pub(crate) fn write_end(
    start: usize,
    tokens: usize,
    capacity: usize,
) -> Result<usize, WriteEndViolation> {
    let Some(end) = start.checked_add(tokens) else {
        return Err(WriteEndViolation::Overflow);
    };
    if end > capacity {
        Err(WriteEndViolation::ExceedsCapacity { end })
    } else {
        Ok(end)
    }
}

/// Returns each row's exclusive end `start + count`, rejecting the first overflow.
///
/// Rows beyond the shorter input are omitted, matching a zipped scan.
///
/// Proved: succeeds exactly when every sum is representable, then returns
/// every sum exactly; otherwise names the first overflowing row.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        ({
            let rows = if starts.len() <= counts.len() { starts.len() } else { counts.len() };
            &&& result is Ok <==> forall|r: int| 0 <= r < rows ==>
                #[trigger] starts[r] + counts[r] <= usize::MAX
            &&& result matches Ok(ends) ==> {
                &&& ends.len() == rows
                &&& forall|r: int| 0 <= r < rows ==> #[trigger] ends[r] == starts[r] + counts[r]
            }
            &&& result matches Err(row) ==> {
                &&& row < rows
                &&& starts[row as int] + counts[row as int] > usize::MAX
                &&& forall|r: int| 0 <= r < row ==> #[trigger] starts[r] + counts[r] <= usize::MAX
            }
        }),
))]
pub(crate) fn checked_ends(starts: &[usize], counts: &[usize]) -> Result<Vec<usize>, usize> {
    let rows = if starts.len() <= counts.len() {
        starts.len()
    } else {
        counts.len()
    };
    let mut ends: Vec<usize> = Vec::with_capacity(rows);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows <= starts.len(),
            rows <= counts.len(),
            ends.len() == row,
            forall|r: int| 0 <= r < row ==> #[trigger] starts[r] + counts[r] <= usize::MAX,
            forall|r: int| 0 <= r < row ==> #[trigger] ends[r] == starts[r] + counts[r],
        decreases rows - row,
    ))]
    while row < rows {
        match starts[row].checked_add(counts[row]) {
            Some(end) => ends.push(end),
            None => return Err(row),
        }
        row += 1;
    }
    Ok(ends)
}

/// Returns the staged length `staged + count` after appending to a staging buffer.
///
/// Proved: the new length is exact and accepted exactly when it is
/// representable and at most `staging_capacity`; a rejection reports the
/// requested length saturated at `usize::MAX`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result is Ok <==> staged + count <= staging_capacity,
        result matches Ok(end) ==> end == staged + count,
        result matches Err(requested) ==>
            requested as int == if staged + count <= usize::MAX {
                staged + count
            } else {
                usize::MAX as int
            },
))]
pub(crate) fn staging_end(
    staged: usize,
    count: usize,
    staging_capacity: usize,
) -> Result<usize, usize> {
    match staged.checked_add(count) {
        Some(end) if end <= staging_capacity => Ok(end),
        Some(end) => Err(end),
        None => Err(usize::MAX),
    }
}

/// First failed check of a staged commit scan.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StagedCommitViolation {
    /// A layer staged a different number of entries than the row proposed.
    Incomplete {
        /// Layer whose staging disagrees.
        layer: usize,
        /// Affected row.
        sequence: usize,
        /// Logical proposal length of the row.
        expected: usize,
        /// Entries the layer actually staged.
        actual: usize,
    },
    /// A row accepts more entries than it proposed.
    PrefixExceedsProposal {
        /// Affected row.
        sequence: usize,
        /// Requested accepted prefix.
        accepted: usize,
        /// Logical proposal length of the row.
        proposed: usize,
    },
}

/// Checks that every layer staged each row's proposal and every prefix fits it.
///
/// `staged[l][s]` is the number of entries layer `l` staged for row `s`. The
/// scan runs over layers, then rows, checking at each row the staged count
/// and then the accepted prefix, and stops at the first failure.
///
/// Proved: the commit is accepted exactly when every layer staged exactly
/// `proposed[s]` entries for every row and (for a non-empty group) every
/// `accepted[s] <= proposed[s]`; a rejection reports the first failing check
/// of that scan with its exact operands.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        accepted.len() == proposed.len(),
        forall|l: int| 0 <= l < staged.len() ==> #[trigger] staged@[l]@.len() == proposed.len(),
    ensures
        result is Ok <==> {
            &&& forall|l: int, s: int| 0 <= l < staged.len() && 0 <= s < proposed.len()
                ==> #[trigger] staged@[l]@[s] == proposed[s]
            &&& staged.len() > 0 ==> forall|s: int| 0 <= s < proposed.len()
                ==> #[trigger] accepted[s] <= proposed[s]
        },
        result matches Err(violation) ==> match violation {
            StagedCommitViolation::Incomplete { layer, sequence, expected, actual } => {
                &&& layer < staged.len()
                &&& sequence < proposed.len()
                &&& expected == proposed[sequence as int]
                &&& actual == staged@[layer as int]@[sequence as int]
                &&& actual != expected
                &&& staged_commit_passes_before(staged@, proposed@, accepted@, layer as int,
                    sequence as int)
            },
            StagedCommitViolation::PrefixExceedsProposal { sequence, accepted: taken, proposed: limit } => {
                &&& staged.len() > 0
                &&& sequence < proposed.len()
                &&& taken == accepted[sequence as int]
                &&& limit == proposed[sequence as int]
                &&& taken > limit
                &&& staged@[0]@[sequence as int] == proposed[sequence as int]
                &&& staged_commit_passes_before(staged@, proposed@, accepted@, 0, sequence as int)
            },
        },
))]
pub(crate) fn check_staged_commit(
    staged: &[Vec<usize>],
    proposed: &[usize],
    accepted: &[usize],
) -> Result<(), StagedCommitViolation> {
    let mut layer = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= staged.len(),
            accepted.len() == proposed.len(),
            forall|l: int| 0 <= l < staged.len() ==> #[trigger] staged@[l]@.len()
                == proposed.len(),
            forall|l: int, s: int| 0 <= l < layer && 0 <= s < proposed.len()
                ==> #[trigger] staged@[l]@[s] == proposed[s],
            layer > 0 ==> forall|s: int| 0 <= s < proposed.len()
                ==> #[trigger] accepted[s] <= proposed[s],
        decreases staged.len() - layer,
    ))]
    while layer < staged.len() {
        let mut sequence = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                layer < staged.len(),
                sequence <= proposed.len(),
                accepted.len() == proposed.len(),
                forall|l: int| 0 <= l < staged.len() ==> #[trigger] staged@[l]@.len()
                    == proposed.len(),
                staged_commit_passes_before(staged@, proposed@, accepted@, layer as int,
                    sequence as int),
            decreases proposed.len() - sequence,
        ))]
        while sequence < proposed.len() {
            let actual = staged[layer][sequence];
            if actual != proposed[sequence] {
                return Err(StagedCommitViolation::Incomplete {
                    layer,
                    sequence,
                    expected: proposed[sequence],
                    actual,
                });
            }
            if accepted[sequence] > proposed[sequence] {
                // Rows before this one passed every check in every earlier layer
                // too, so an over-acceptance can only surface in layer zero.
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(layer == 0);
                }
                return Err(StagedCommitViolation::PrefixExceedsProposal {
                    sequence,
                    accepted: accepted[sequence],
                    proposed: proposed[sequence],
                });
            }
            sequence += 1;
        }
        layer += 1;
    }
    Ok(())
}

/// Marks the rows a validated truncate must physically clear.
///
/// Row `r` is cleared exactly when it held tokens (`current[r] != 0`) and is
/// truncated to zero (`requested[r] == 0`); every other permitted rewind
/// leaves the stored prefix in place. Rows beyond the shorter input are not
/// examined, matching a zipped scan.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(cleared =>
    ensures
        cleared.len() == if current.len() <= requested.len() {
            current.len()
        } else {
            requested.len()
        },
        forall|r: int| 0 <= r < cleared.len() ==>
            #[trigger] cleared[r] == (current[r] != 0 && requested[r] == 0),
))]
pub(crate) fn rows_cleared_by_truncate(current: &[usize], requested: &[usize]) -> Vec<bool> {
    let rows = if current.len() <= requested.len() {
        current.len()
    } else {
        requested.len()
    };
    let mut cleared: Vec<bool> = Vec::with_capacity(rows);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= rows,
            rows <= current.len(),
            rows <= requested.len(),
            cleared.len() == row,
            forall|r: int| 0 <= r < row ==>
                #[trigger] cleared[r] == (current[r] != 0 && requested[r] == 0),
        decreases rows - row,
    ))]
    while row < rows {
        cleared.push(current[row] != 0 && requested[row] == 0);
        row += 1;
    }
    cleared
}

/// Returns the first position whose index is not below `bound`.
///
/// Proved: `None` exactly when every index lies in `[0, bound)`; otherwise
/// the lowest offending position.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(first =>
    ensures
        first is None <==> forall|i: int| 0 <= i < indices.len() ==> #[trigger] indices[i] < bound,
        first matches Some(position) ==> {
            &&& position < indices.len()
            &&& indices[position as int] >= bound
            &&& forall|i: int| 0 <= i < position ==> #[trigger] indices[i] < bound
        },
))]
pub(crate) fn first_index_out_of_range(indices: &[usize], bound: usize) -> Option<usize> {
    let mut position = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            position <= indices.len(),
            forall|i: int| 0 <= i < position ==> #[trigger] indices[i] < bound,
        decreases indices.len() - position,
    ))]
    while position < indices.len() {
        if indices[position] >= bound {
            return Some(position);
        }
        position += 1;
    }
    None
}

/// Converts host indices or positions to Burn's portable `i32` index type.
///
/// Proved: the conversion succeeds exactly when every value is at most
/// `i32::MAX`, preserves every value exactly, and otherwise reports the
/// first position whose value does not fit.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        result is Ok <==> forall|i: int| 0 <= i < values.len() ==>
            #[trigger] values[i] <= i32::MAX,
        result matches Ok(converted) ==> {
            &&& converted.len() == values.len()
            &&& forall|i: int| 0 <= i < values.len() ==>
                #[trigger] converted[i] as int == values[i] as int
        },
        result matches Err(position) ==> {
            &&& position < values.len()
            &&& values[position as int] > i32::MAX
            &&& forall|i: int| 0 <= i < position ==> #[trigger] values[i] <= i32::MAX
        },
))]
pub(crate) fn portable_indices(values: &[usize]) -> Result<Vec<i32>, usize> {
    let mut converted: Vec<i32> = Vec::with_capacity(values.len());
    let mut position = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            position <= values.len(),
            converted.len() == position,
            forall|i: int| 0 <= i < position ==> #[trigger] values[i] <= i32::MAX,
            forall|i: int| 0 <= i < position ==>
                #[trigger] converted[i] as int == values[i] as int,
        decreases values.len() - position,
    ))]
    while position < values.len() {
        let value = values[position];
        if value > PORTABLE_TENSOR_ELEMENT_LIMIT {
            return Err(position);
        }
        converted.push(value as i32);
        position += 1;
    }
    Ok(converted)
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
