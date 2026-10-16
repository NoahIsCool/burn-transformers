//! Dependency-free geometry of T5 relative-position buckets.
//!
//! # Bucket layout
//!
//! A T5 stack maps a key-minus-query displacement to one of `B` learned
//! buckets. The bidirectional encoder spends `B / 2` buckets per direction,
//! of which the first `B / 4` hold exact distances; the causal decoder
//! spends all `B` buckets on past keys, of which `B / 2` are exact. The
//! remaining buckets of each direction cover distances logarithmically up to
//! `max_distance`, which therefore must exceed the exact region.
//! [`bucket_geometry`] checks these conditions for the stacks a
//! configuration builds.
//!
//! # Displacement tables
//!
//! The explicit-position bias gathers bucket IDs from a host table anchored
//! at radius `r = min(bound, max_distance)`: a `(r + 1) x (r + 1)` table
//! anchored at query `r` lists the bucket of every displacement `-r..=r` in
//! its `2r + 1` entries, and the device clamps each displacement into
//! `[-r, r]` before adding `r`. [`displacement_table`] sizes that table, and
//! a lemma proves every clamped index lies inside it.
//!
//! # Table owners
//!
//! An edited encoder may mix head counts. A `[B, heads]` table serves only
//! layers with its head count, so [`bias_owners`] assigns every layer the
//! first layer with the same head count, which registers the shared table.
//!
//! # Verified geometry
//!
//! Every function here is checked in place by Verus through `verus/t5_gpt.rs`
//! (see `verus/README.md`): the geometry check accepts exactly the valid
//! layouts and names each rejection's cause, an accepted layout has a
//! non-empty exact region below `max_distance` in every built stack, the
//! table radius and width are exact, every clamped displacement indexes the
//! table (exactly, without clamping, for displacements within `r`), and two
//! layers share a bias owner exactly when their head counts agree.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// `value` clamped into `[low, high]`, as the device's `clamp` does.
pub open spec fn clamp(value: int, low: int, high: int) -> int {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

/// Every clamped displacement indexes the `2r + 1` entries of a radius-`r` table.
///
/// The index `clamp(d, -r, r) + r` lies in `0..2r + 1`, the bucket-table
/// length `2 (r + 1) - 1` of a `(r + 1) x (r + 1)` table, and a displacement
/// within the radius is not altered by the clamp.
pub proof fn lemma_clamped_index_in_table(displacement: int, radius: int)
    requires
        radius >= 0,
    ensures
        0 <= clamp(displacement, -radius, radius) + radius < 2 * (radius + 1) - 1,
        -radius <= displacement <= radius
            ==> clamp(displacement, -radius, radius) == displacement,
{
}

} // verus!

/// Why a relative-bucket layout cannot serve the stacks being built.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BucketGeometryError {
    /// Fewer than four buckets cannot split into two directions with an exact region.
    TooFewBuckets,
    /// An odd bucket count cannot split evenly by displacement sign.
    OddBuckets,
    /// `max_distance` does not exceed the encoder's exact region `B / 4`.
    EncoderDistance,
    /// `max_distance` does not exceed the causal decoder's exact region `B / 2`.
    DecoderDistance,
}

/// Checks a bucket count `B` and maximum distance for the encoder and, when `decoder`, the decoder.
///
/// Conditions are checked in order: `B >= 4`, `B` even,
/// `max_distance > B / 4`, and (for a decoder) `max_distance > B / 2`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(geometry =>
    ensures
        match geometry {
            Ok(()) => {
                &&& buckets >= 4
                &&& buckets % 2 == 0
                &&& 1 <= buckets / 4 < max_distance
                &&& buckets / 4 < buckets / 2
                &&& decoder ==> buckets / 2 < max_distance
            },
            Err(BucketGeometryError::TooFewBuckets) => buckets < 4,
            Err(BucketGeometryError::OddBuckets) => buckets >= 4 && buckets % 2 != 0,
            Err(BucketGeometryError::EncoderDistance) => {
                &&& buckets >= 4
                &&& buckets % 2 == 0
                &&& max_distance <= buckets / 4
            },
            Err(BucketGeometryError::DecoderDistance) => {
                &&& buckets >= 4
                &&& buckets % 2 == 0
                &&& buckets / 4 < max_distance <= buckets / 2
                &&& decoder
            },
        },
))]
pub(crate) fn bucket_geometry(
    buckets: usize,
    max_distance: usize,
    decoder: bool,
) -> Result<(), BucketGeometryError> {
    if buckets < 4 {
        return Err(BucketGeometryError::TooFewBuckets);
    }
    if !buckets.is_multiple_of(2) {
        return Err(BucketGeometryError::OddBuckets);
    }
    if max_distance <= buckets / 4 {
        return Err(BucketGeometryError::EncoderDistance);
    }
    if decoder && max_distance <= buckets / 2 {
        return Err(BucketGeometryError::DecoderDistance);
    }
    Ok(())
}

/// Radius `r = min(bound, max_distance)` and width `r + 1` of a displacement table.
///
/// Returns `None` when `r + 1` overflows. See the module documentation for
/// the table's layout and the clamped lookup.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(table =>
    ensures
        ({
            let radius = if bound <= max_distance { bound as int } else { max_distance as int };
            match table {
                Some((r, width)) => r == radius && width == radius + 1,
                None => radius == usize::MAX,
            }
        }),
))]
pub(crate) fn displacement_table(bound: usize, max_distance: usize) -> Option<(usize, usize)> {
    let radius = if bound <= max_distance {
        bound
    } else {
        max_distance
    };
    let width = radius.checked_add(1)?;
    Some((radius, width))
}

/// Learned relative-bias owner of every layer of an edited encoder.
///
/// A table `[B, heads]` serves only layers with its head count, so layer `i`
/// uses the table of the first layer with `heads[i]` heads. The proof
/// establishes that every owner precedes (or is) its layer, has the layer's
/// head count, and is the first such layer, so two layers share a table
/// exactly when their head counts agree and every owner owns itself.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(owners =>
    ensures
        owners.len() == heads.len(),
        forall|i: int| 0 <= i < heads.len() ==> {
            &&& #[trigger] owners@[i] <= i
            &&& heads@[owners@[i] as int] == heads@[i]
            &&& forall|j: int| 0 <= j < owners@[i] ==> #[trigger] heads@[j] != heads@[i]
        },
        forall|i: int, j: int| 0 <= i < heads.len() && 0 <= j < heads.len()
            ==> (#[trigger] owners@[i] == #[trigger] owners@[j]) == (heads@[i] == heads@[j]),
        forall|i: int| 0 <= i < heads.len() ==> owners@[owners@[i] as int] == #[trigger] owners@[i],
))]
pub(crate) fn bias_owners(heads: &[usize]) -> Vec<usize> {
    let mut owners: Vec<usize> = Vec::with_capacity(heads.len());
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= heads.len(),
            owners.len() == layer,
            forall|i: int| 0 <= i < layer ==> {
                &&& #[trigger] owners@[i] <= i
                &&& heads@[owners@[i] as int] == heads@[i]
                &&& forall|j: int| 0 <= j < owners@[i] ==> #[trigger] heads@[j] != heads@[i]
            },
        decreases heads.len() - layer,
    ))]
    while layer < heads.len() {
        // The first layer with this head count; the layer itself at the latest.
        let mut owner: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                layer < heads.len(),
                owner <= layer,
                forall|j: int| 0 <= j < owner ==> #[trigger] heads@[j] != heads@[layer as int],
            decreases layer - owner,
        ))]
        while owner < layer && heads[owner] != heads[layer] {
            owner += 1;
        }
        owners.push(owner);
        layer += 1;
    }
    // First occurrences coincide exactly for equal head counts.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|i: int, j: int| 0 <= i < heads.len() && 0 <= j < heads.len()
            implies (#[trigger] owners@[i] == #[trigger] owners@[j]) == (heads@[i] == heads@[j]) by {
            if heads@[i] == heads@[j] {
                if owners@[i] < owners@[j] {
                    assert(heads@[owners@[i] as int] == heads@[j]);
                } else if owners@[j] < owners@[i] {
                    assert(heads@[owners@[j] as int] == heads@[i]);
                }
            }
        }
        assert forall|i: int| 0 <= i < heads.len()
            implies owners@[owners@[i] as int] == #[trigger] owners@[i] by {
            let o = owners@[i] as int;
            assert(heads@[o] == heads@[i]);
            assert((owners@[o] == owners@[i]) == (heads@[o] == heads@[i]));
        }
    }
    owners
}

#[cfg(test)]
#[path = "buckets_tests.rs"]
mod tests;
