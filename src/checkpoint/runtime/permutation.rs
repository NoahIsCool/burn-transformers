//! Probe dimensions and axis-permutation recovery for classifying load mappers.
//!
//! A packed weight's load mapper is classified by loading a probe whose axes
//! have pairwise-distinct lengths ([`probe_dimension`]). Distinct lengths make
//! the mapped shape determine the axis permutation uniquely: output axis `k`
//! must come from the only input axis with the same length
//! ([`permutation_of`]), and a mapper whose output shape is not such a
//! rearrangement is not a pure permutation.
//!
//! The module has no crate, Burn, or allocator dependencies so that Verus can
//! check it in place through `verus/checkpoint.rs`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: the probe length of axis `axis`.
pub open spec fn probe(axis: int) -> int {
    if axis == 0 {
        2
    } else if axis == 1 {
        3
    } else if axis == 2 {
        5
    } else if axis == 3 {
        7
    } else if axis == 4 {
        11
    } else if axis == 5 {
        13
    } else if axis == 6 {
        17
    } else if axis == 7 {
        19
    } else {
        axis + 23
    }
}

/// Specification: the first `count` entries of `values` are pairwise distinct.
pub open spec fn distinct(values: Seq<usize>, count: int) -> bool {
    forall|i: int, j: int| 0 <= i < j < count ==> #[trigger] values[i] != #[trigger] values[j]
}

/// Specification: `value` occurs in `values`.
pub open spec fn occurs(values: Seq<usize>, value: usize) -> bool {
    exists|j: int| 0 <= j < values.len() && values[j] == value
}

/// Specification: `output` is a rearrangement of the distinct `input`:
/// equal length, every entry occurs in `input`, and no entry repeats.
pub open spec fn rearrangement(input: Seq<usize>, output: Seq<usize>) -> bool {
    &&& output.len() == input.len()
    &&& forall|k: int| 0 <= k < output.len() ==> occurs(input, #[trigger] output[k])
    &&& distinct(output, output.len() as int)
}

} // verus!

/// Returns the probe length of axis `axis`: the primes `2..=19`, then `axis + 23`.
///
/// Verified: lengths strictly increase with the axis, so the lengths of any
/// probe are pairwise distinct (and at least two, so a transpose is visible).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(length =>
    requires
        axis <= usize::MAX - 23,
    ensures
        length == probe(axis as int),
        length >= 2,
        forall|earlier: int| 0 <= earlier < axis ==> #[trigger] probe(earlier) < length,
))]
pub(crate) const fn probe_dimension(axis: usize) -> usize {
    match axis {
        0 => 2,
        1 => 3,
        2 => 5,
        3 => 7,
        4 => 11,
        5 => 13,
        6 => 17,
        7 => 19,
        _ => axis + 23,
    }
}

/// Writes into `axes` the permutation mapping `input` lengths onto `output`.
///
/// For each output axis `k`, `axes[k]` is the first input axis of equal
/// length. Returns `false`, leaving `axes` partially written, when the ranks
/// differ, an output length occurs in no input axis, or two output axes would
/// take the same input axis.
///
/// Verified, for pairwise-distinct `input`: `true` exactly when `output` is a
/// rearrangement of `input`, and then `axes` is a bijection on the axes with
/// `output[k] == input[axes[k]]` for every `k` (Burn's `permute` convention).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        old(axes)@.len() == input@.len(),
        distinct(input@, input@.len() as int),
    ensures
        final(axes)@.len() == input@.len(),
        found == rearrangement(input@, output@),
        found ==> forall|k: int| 0 <= k < input@.len() ==> {
            &&& #[trigger] final(axes)@[k] < input@.len()
            &&& output@[k] == input@[final(axes)@[k] as int]
        },
        found ==> distinct(final(axes)@, input@.len() as int),
))]
pub(crate) fn permutation_of(input: &[usize], output: &[usize], axes: &mut [usize]) -> bool {
    let rank = input.len();
    if output.len() != rank {
        return false;
    }
    let mut axis = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            axis <= rank,
            rank == input@.len(),
            output@.len() == rank,
            axes@.len() == rank,
            distinct(input@, rank as int),
            forall|k: int| 0 <= k < axis ==> {
                &&& #[trigger] axes@[k] < rank
                &&& output@[k] == input@[axes@[k] as int]
            },
            distinct(axes@, axis as int),
            distinct(output@, axis as int),
            forall|k: int| 0 <= k < axis ==> occurs(input@, #[trigger] output@[k]),
        decreases rank - axis,
    ))]
    while axis < rank {
        let Some(source) = position(input, output[axis]) else {
            return false;
        };
        let mut earlier = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                earlier <= axis,
                axis < rank,
                rank == input@.len(),
                output@.len() == rank,
                axes@.len() == rank,
                source < rank,
                input@[source as int] == output@[axis as int],
                forall|k: int| 0 <= k < axis ==> {
                    &&& #[trigger] axes@[k] < rank
                    &&& output@[k] == input@[axes@[k] as int]
                },
                forall|k: int| 0 <= k < earlier ==> axes@[k] != source,
            decreases axis - earlier,
        ))]
        while earlier < axis {
            if axes[earlier] == source {
                // Output axes `earlier` and `axis` both have length `input[source]`.
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(output@[earlier as int] == output@[axis as int]);
                    assert(!distinct(output@, output@.len() as int));
                }
                return false;
            }
            earlier += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            // Distinct input lengths: a repeated output length would have reused `source`.
            assert forall|k: int| 0 <= k < axis implies output@[k] != output@[axis as int] by {
                if output@[k] == output@[axis as int] {
                    assert(input@[axes@[k] as int] == input@[source as int]);
                }
            }
        }
        axes[axis] = source;
        axis += 1;
    }
    true
}

/// Returns the first index of `value` in `values`, or `None` when it does not occur.
///
/// Verified: `Some(j)` holds `values[j] == value` with no earlier match;
/// `None` exactly when `value` does not occur.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
        match found {
            Some(index) => index < values@.len() && values@[index as int] == value
                && forall|j: int| 0 <= j < index ==> values@[j] != value,
            None => !occurs(values@, value),
        }
))]
fn position(values: &[usize], value: usize) -> Option<usize> {
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= values@.len(),
            forall|j: int| 0 <= j < index ==> values@[j] != value,
        decreases values@.len() - index,
    ))]
    while index < values.len() {
        if values[index] == value {
            return Some(index);
        }
        index += 1;
    }
    None
}

#[cfg(test)]
#[path = "permutation_tests.rs"]
mod tests;
