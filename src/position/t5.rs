//! Exact and logarithmic relative-position buckets for T5 attention.
//!
//! The displacement is `key_position - query_position`. Within each direction,
//! distances below `E = buckets_per_direction / 2` receive exact buckets; larger
//! distances use `E + floor(log(distance/E) / log(max_distance/E) * (B-E))`,
//! clipped to `B-1`, where `B` is the number of buckets in that direction.
//!
//! Unsigned distance arithmetic handles extreme absolute or signed positions;
//! logarithms use host F64. Bidirectional tables separate past and future halves.
//! Causal tables map future displacements to zero, but the attention mask remains
//! responsible for excluding those future keys.
//!
//! # Verified bucket assembly
//!
//! The floating-point logarithm only proposes a logarithmic offset. Every
//! integer step around it is checked in place by Verus through
//! `verus/position.rs`:
//!
//! * [`checked_bucket_geometry`] (behind [`T5RelativePositionBuckets::new`])
//!   accepts exactly the geometries satisfying `valid_bucket_geometry` and
//!   otherwise returns the first failed check in order;
//! * [`signed_direction_distance`] and [`position_direction_distance`] split a
//!   signed displacement or a pair of absolute positions into the direction
//!   `key > query` and the exact distance `|key - query|`, including
//!   `i64::MIN` and the full `usize` range;
//! * [`combine_bucket`] equals the reference formula `spec_bucket` for every
//!   validated geometry and *any* proposed offset, so the result is a valid
//!   table index, bidirectional future keys use the upper half and past keys
//!   the lower half, causal future keys map to bucket zero, exact-region
//!   distances map to `direction_offset + distance`, and the logarithmic
//!   region is `min(E + offset, B - 1)`.

use core::fmt;

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Buckets per direction `B`: half the table when bidirectional, all of it otherwise.
pub open spec fn spec_buckets_per_direction(num_buckets: int, bidirectional: bool) -> int {
    if bidirectional {
        num_buckets / 2
    } else {
        num_buckets
    }
}

/// The geometry [`T5RelativePositionBuckets::new`] accepts.
///
/// A causal table needs two buckets, a bidirectional one an even count of at
/// least four, and the saturation distance must exceed the exact-region size
/// `E = B / 2`.
pub open spec fn valid_bucket_geometry(
    num_buckets: int,
    max_distance: int,
    bidirectional: bool,
) -> bool {
    &&& if bidirectional {
        num_buckets >= 4 && num_buckets % 2 == 0
    } else {
        num_buckets >= 2
    }
    &&& max_distance > spec_buckets_per_direction(num_buckets, bidirectional) / 2
}

/// Reference T5 bucket for a direction, distance, and proposed logarithmic offset.
///
/// With `B` buckets per direction and `E = B / 2`, the within-direction index
/// is the distance below `E` and `min(E + offset, B - 1)` from `E` on;
/// bidirectional future keys add `B`, and causal future keys use distance zero.
pub open spec fn spec_bucket(
    num_buckets: int,
    bidirectional: bool,
    key_is_future: bool,
    distance: int,
    logarithmic_offset: int,
) -> int {
    let per_direction = spec_buckets_per_direction(num_buckets, bidirectional);
    let max_exact = per_direction / 2;
    let direction_offset = if bidirectional && key_is_future {
        per_direction
    } else {
        0
    };
    let distance = if !bidirectional && key_is_future {
        0
    } else {
        distance
    };
    let within_direction = if distance < max_exact {
        distance
    } else if max_exact + logarithmic_offset < per_direction - 1 {
        max_exact + logarithmic_offset
    } else {
        per_direction - 1
    };
    direction_offset + within_direction
}

/// Absolute value of a signed displacement.
pub open spec fn spec_abs(value: int) -> int {
    if value >= 0 {
        value
    } else {
        -value
    }
}

} // verus!

/// Validated T5 relative-position bucketing parameters.
///
/// T5 maps `relative_position = key_position - query_position` into a learned
/// bias-table index. For each direction, the first half of the available
/// buckets represent exact distances. The remaining half grows logarithmically
/// until all distances at or beyond `max_distance` share the final bucket.
/// Bidirectional attention splits the table evenly between past and future;
/// causal attention maps every future key to distance zero.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct T5RelativePositionBuckets {
    /// Total learned bias-table entries across both directions, when bidirectional.
    num_buckets: usize,
    /// Positive distance at which the logarithmic region reaches its final bucket.
    max_distance: usize,
    /// Whether future keys receive a separate half of the learned bucket table.
    bidirectional: bool,
}

impl T5RelativePositionBuckets {
    /// Creates a checked T5 bucket mapping.
    ///
    /// # Errors
    ///
    /// Unidirectional mappings require at least two buckets. Bidirectional
    /// mappings require an even count of at least four. `max_distance` must be
    /// greater than the largest exact-region boundary so the logarithmic
    /// denominator is finite and positive.
    pub fn new(
        num_buckets: usize,
        max_distance: usize,
        bidirectional: bool,
    ) -> Result<Self, T5BucketError> {
        checked_bucket_geometry(num_buckets, max_distance, bidirectional)
    }

    /// Returns the total learned relative-bias bucket count.
    #[must_use]
    pub const fn num_buckets(self) -> usize {
        self.num_buckets
    }

    /// Returns the distance at which logarithmic buckets saturate.
    #[must_use]
    pub const fn max_distance(self) -> usize {
        self.max_distance
    }

    /// Returns whether past and future use separate table halves.
    #[must_use]
    pub const fn is_bidirectional(self) -> bool {
        self.bidirectional
    }

    /// Maps a signed key-minus-query displacement to a table bucket.
    ///
    /// Positive values denote future keys, negative values denote past keys,
    /// and zero denotes the query token itself. The implementation handles
    /// `i64::MIN` without signed-absolute-value overflow.
    #[must_use]
    pub fn bucket(self, relative_position: i64) -> usize {
        let (key_is_future, distance) = signed_direction_distance(relative_position);
        self.bucket_parts(key_is_future, distance)
    }

    /// Maps explicit absolute query and key positions to a table bucket.
    ///
    /// This overload avoids signed subtraction and therefore supports the full
    /// `usize` position range.
    #[must_use]
    pub fn bucket_for_positions(self, query_position: usize, key_position: usize) -> usize {
        let (key_is_future, distance) = position_direction_distance(query_position, key_position);
        self.bucket_parts(key_is_future, distance)
    }

    /// Maps a direction and unsigned distance to exact or logarithmic T5 buckets.
    ///
    /// Unsigned distances avoid signed-minimum overflow. Causal future keys map to
    /// zero; distant keys saturate at the last bucket after host F64 logarithms.
    fn bucket_parts(self, key_is_future: bool, distance: u64) -> usize {
        let buckets_per_direction = if self.bidirectional {
            self.num_buckets / 2
        } else {
            self.num_buckets
        };
        // Causal future keys use distance zero, so they never need a logarithm.
        let effective_distance = if !self.bidirectional && key_is_future {
            0
        } else {
            distance
        };
        let logarithmic_offset = self.logarithmic_offset(effective_distance, buckets_per_direction);
        combine_bucket(
            self.num_buckets,
            self.bidirectional,
            key_is_future,
            distance,
            logarithmic_offset,
        )
    }

    /// Proposes the logarithmic-region offset `floor(log(d/E) / log(M/E) * (B-E))`.
    ///
    /// `E = B/2` is the exact-region size, `B` the per-direction bucket count,
    /// and `M` the saturation distance. Only distances `d >= E` consult this
    /// value; the cast saturates on overflow and [`combine_bucket`] clips it.
    fn logarithmic_offset(self, distance: u64, buckets_per_direction: usize) -> usize {
        let max_exact = buckets_per_direction / 2;
        if distance < max_exact as u64 {
            return 0;
        }
        let distance_ratio = distance as f64 / max_exact as f64;
        let max_ratio = self.max_distance as f64 / max_exact as f64;
        let logarithmic_fraction = distance_ratio.ln() / max_ratio.ln();
        let logarithmic_buckets = buckets_per_direction - max_exact;
        (logarithmic_fraction * logarithmic_buckets as f64) as usize
    }
}

/// Validates a bucket geometry and builds the mapping, or reports the first failed check.
///
/// Verified: the result is `Ok` exactly for geometries satisfying
/// `valid_bucket_geometry`, and then stores the inputs unchanged. Otherwise it
/// is the first failure in the order too few buckets, odd bidirectional count,
/// saturation distance not beyond the exact region, each carrying the
/// offending values.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Ok(buckets) => {
                &&& buckets.num_buckets == num_buckets
                &&& buckets.max_distance == max_distance
                &&& buckets.bidirectional == bidirectional
                &&& valid_bucket_geometry(num_buckets as int, max_distance as int, bidirectional)
            },
            Err(T5BucketError::TooFewBuckets { num_buckets: count, bidirectional: mode, minimum }) => {
                &&& count == num_buckets
                &&& mode == bidirectional
                &&& minimum == if bidirectional { 4usize } else { 2usize }
                &&& num_buckets < minimum
            },
            Err(T5BucketError::OddBidirectionalBucketCount { num_buckets: count }) => {
                &&& count == num_buckets
                &&& bidirectional
                &&& num_buckets >= 4
                &&& num_buckets % 2 == 1
            },
            Err(T5BucketError::MaxDistanceNotBeyondExact { max_distance: distance, max_exact }) => {
                &&& distance == max_distance
                &&& max_exact == spec_buckets_per_direction(num_buckets as int, bidirectional) / 2
                &&& if bidirectional {
                    num_buckets >= 4 && num_buckets % 2 == 0
                } else {
                    num_buckets >= 2
                }
                &&& max_distance <= max_exact
            },
        }
))]
fn checked_bucket_geometry(
    num_buckets: usize,
    max_distance: usize,
    bidirectional: bool,
) -> Result<T5RelativePositionBuckets, T5BucketError> {
    let minimum = if bidirectional { 4 } else { 2 };
    if num_buckets < minimum {
        return Err(T5BucketError::TooFewBuckets {
            num_buckets,
            bidirectional,
            minimum,
        });
    }
    if bidirectional && !num_buckets.is_multiple_of(2) {
        return Err(T5BucketError::OddBidirectionalBucketCount { num_buckets });
    }
    let buckets_per_direction = if bidirectional {
        num_buckets / 2
    } else {
        num_buckets
    };
    let max_exact = buckets_per_direction / 2;
    if max_distance <= max_exact {
        return Err(T5BucketError::MaxDistanceNotBeyondExact {
            max_distance,
            max_exact,
        });
    }

    Ok(T5RelativePositionBuckets {
        num_buckets,
        max_distance,
        bidirectional,
    })
}

/// Splits a signed key-minus-query displacement into direction and distance.
///
/// Verified: the direction is `relative_position > 0` and the distance is
/// exactly `|relative_position|`, including `2^63` for `i64::MIN`, without
/// signed overflow (the magnitude of a negative value is formed as
/// `-(value + 1) + 1`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(parts =>
    ensures
        parts.0 == (relative_position > 0),
        parts.1 as int == spec_abs(relative_position as int),
))]
fn signed_direction_distance(relative_position: i64) -> (bool, u64) {
    if relative_position >= 0 {
        (relative_position > 0, relative_position as u64)
    } else {
        // `-(relative_position + 1)` is at most `i64::MAX`, so it cannot overflow.
        (false, (-(relative_position + 1)) as u64 + 1)
    }
}

/// Splits absolute query and key positions into direction and distance.
///
/// Verified: the direction is `key_position > query_position` and the
/// distance is exactly `|key_position - query_position|` for every pair of
/// `usize` positions, so it agrees with [`signed_direction_distance`] of the
/// signed displacement whenever that displacement fits in `i64`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(parts =>
    ensures
        parts.0 == (key_position > query_position),
        parts.1 as int == spec_abs(key_position as int - query_position as int),
))]
fn position_direction_distance(query_position: usize, key_position: usize) -> (bool, u64) {
    if key_position >= query_position {
        (
            key_position > query_position,
            (key_position - query_position) as u64,
        )
    } else {
        (false, (query_position - key_position) as u64)
    }
}

/// Assembles a table index from direction, distance, and a proposed logarithmic offset.
///
/// With `B` buckets per direction and `E = B/2` exact buckets, the
/// within-direction index is `distance` when `distance < E` and
/// `min(E + logarithmic_offset, B - 1)` otherwise; bidirectional future keys
/// add `B`. Causal future keys use distance zero, matching the reference,
/// because the causal mask—not the bias—excludes them. The minimum is computed
/// without forming `E + logarithmic_offset`, so arbitrary offsets cannot overflow.
///
/// Verified: the result equals the reference formula `spec_bucket` exactly
/// for every validated bucket count and every distance and offset, which
/// implies the listed range, direction, and exact-region properties.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bucket =>
    requires
        bidirectional ==> num_buckets >= 4 && num_buckets % 2 == 0,
        !bidirectional ==> num_buckets >= 2,
    ensures
        bucket == spec_bucket(
            num_buckets as int,
            bidirectional,
            key_is_future,
            distance as int,
            logarithmic_offset as int,
        ),
        bucket < num_buckets,
        bidirectional && key_is_future ==> bucket >= num_buckets / 2,
        bidirectional && !key_is_future ==> bucket < num_buckets / 2,
        !bidirectional && key_is_future ==> bucket == 0,
        bidirectional && key_is_future && distance < num_buckets / 4
            ==> bucket == num_buckets / 2 + distance,
        !key_is_future
            && distance < (if bidirectional { num_buckets / 4 } else { num_buckets / 2 })
            ==> bucket == distance,
))]
fn combine_bucket(
    num_buckets: usize,
    bidirectional: bool,
    key_is_future: bool,
    distance: u64,
    logarithmic_offset: usize,
) -> usize {
    let buckets_per_direction = if bidirectional {
        num_buckets / 2
    } else {
        num_buckets
    };
    let direction_offset = if bidirectional && key_is_future {
        buckets_per_direction
    } else {
        0
    };
    // Causal attention cannot observe future keys. Matching T5's reference
    // implementation, clamp their relative distance to zero rather than
    // assigning a past-distance bucket.
    let distance = if !bidirectional && key_is_future {
        0
    } else {
        distance
    };
    let max_exact = buckets_per_direction / 2;
    let within_direction = if distance < max_exact as u64 {
        distance as usize
    } else if logarithmic_offset >= buckets_per_direction - 1 - max_exact {
        buckets_per_direction - 1
    } else {
        max_exact + logarithmic_offset
    };
    direction_offset + within_direction
}

/// Invalid parameters for [`T5RelativePositionBuckets`].
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum T5BucketError {
    /// There are not enough buckets to divide exact and logarithmic regions.
    TooFewBuckets {
        /// Invalid bucket count.
        num_buckets: usize,
        /// Whether the count would be split by direction.
        bidirectional: bool,
        /// Minimum valid count for that mode.
        minimum: usize,
    },
    /// A bidirectional bias table must split evenly by sign.
    OddBidirectionalBucketCount {
        /// Invalid odd bucket count.
        num_buckets: usize,
    },
    /// Logarithmic bucketing needs a range beyond the exact boundary.
    MaxDistanceNotBeyondExact {
        /// Invalid saturation distance.
        max_distance: usize,
        /// First logarithmic distance.
        max_exact: usize,
    },
}

impl fmt::Display for T5BucketError {
    /// Formats invalid bucket counts or logarithmic-distance bounds with their values.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewBuckets {
                num_buckets,
                bidirectional,
                minimum,
            } => write!(
                formatter,
                "T5 {mode} bucketing requires at least {minimum} buckets, got {num_buckets}",
                mode = if *bidirectional {
                    "bidirectional"
                } else {
                    "causal"
                }
            ),
            Self::OddBidirectionalBucketCount { num_buckets } => write!(
                formatter,
                "T5 bidirectional bucket count must be even, got {num_buckets}"
            ),
            Self::MaxDistanceNotBeyondExact {
                max_distance,
                max_exact,
            } => write!(
                formatter,
                "T5 max distance {max_distance} must exceed exact-region boundary {max_exact}"
            ),
        }
    }
}

impl std::error::Error for T5BucketError {}

#[cfg(test)]
#[path = "t5_tests.rs"]
mod tests;
