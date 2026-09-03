use core::fmt;

/// Validated T5 relative-position bucketing parameters.
///
/// T5 maps `relative_position = key_position - query_position` into a learned
/// bias-table index. For each direction, the first half of the available
/// buckets represent exact distances. The remaining half grows logarithmically
/// until all distances at or beyond `max_distance` share the final bucket.
/// Bidirectional attention splits the table evenly between past and future;
/// causal attention maps every future key to distance zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct T5RelativePositionBuckets {
    num_buckets: usize,
    max_distance: usize,
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

        Ok(Self {
            num_buckets,
            max_distance,
            bidirectional,
        })
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
        let key_is_future = relative_position > 0;
        let distance = relative_position.unsigned_abs();
        self.bucket_parts(key_is_future, distance)
    }

    /// Maps explicit absolute query and key positions to a table bucket.
    ///
    /// This overload avoids signed subtraction and therefore supports the full
    /// `usize` position range.
    #[must_use]
    pub fn bucket_for_positions(self, query_position: usize, key_position: usize) -> usize {
        if key_position >= query_position {
            self.bucket_parts(
                key_position > query_position,
                (key_position - query_position) as u64,
            )
        } else {
            self.bucket_parts(false, (query_position - key_position) as u64)
        }
    }

    fn bucket_parts(self, key_is_future: bool, distance: u64) -> usize {
        let buckets_per_direction = if self.bidirectional {
            self.num_buckets / 2
        } else {
            self.num_buckets
        };
        let direction_offset =
            usize::from(self.bidirectional && key_is_future) * buckets_per_direction;

        // Causal attention cannot observe future keys. Matching T5's reference
        // implementation, clamp their relative distance to zero rather than
        // assigning a past-distance bucket.
        let distance = if !self.bidirectional && key_is_future {
            0
        } else {
            distance
        };
        let max_exact = buckets_per_direction / 2;
        let within_direction = if distance < max_exact as u64 {
            distance as usize
        } else {
            let distance_ratio = distance as f64 / max_exact as f64;
            let max_ratio = self.max_distance as f64 / max_exact as f64;
            let logarithmic_fraction = distance_ratio.ln() / max_ratio.ln();
            let logarithmic_buckets = buckets_per_direction - max_exact;
            let candidate =
                max_exact + (logarithmic_fraction * logarithmic_buckets as f64) as usize;
            candidate.min(buckets_per_direction - 1)
        };

        direction_offset + within_direction
    }
}

/// Invalid parameters for [`T5RelativePositionBuckets`].
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
mod tests {
    use super::*;

    #[test]
    fn validates_bucket_geometry() {
        assert_eq!(
            T5RelativePositionBuckets::new(1, 128, false),
            Err(T5BucketError::TooFewBuckets {
                num_buckets: 1,
                bidirectional: false,
                minimum: 2,
            })
        );
        assert_eq!(
            T5RelativePositionBuckets::new(2, 128, true),
            Err(T5BucketError::TooFewBuckets {
                num_buckets: 2,
                bidirectional: true,
                minimum: 4,
            })
        );
        assert_eq!(
            T5RelativePositionBuckets::new(5, 128, true),
            Err(T5BucketError::OddBidirectionalBucketCount { num_buckets: 5 })
        );
        assert_eq!(
            T5RelativePositionBuckets::new(32, 16, false),
            Err(T5BucketError::MaxDistanceNotBeyondExact {
                max_distance: 16,
                max_exact: 16,
            })
        );
        assert_eq!(
            T5RelativePositionBuckets::new(32, 8, true),
            Err(T5BucketError::MaxDistanceNotBeyondExact {
                max_distance: 8,
                max_exact: 8,
            })
        );
    }

    #[test]
    fn matches_reference_causal_buckets() {
        let buckets = T5RelativePositionBuckets::new(32, 128, false).expect("valid config");
        let cases = [
            (0, 0),
            (-1, 1),
            (-2, 2),
            (-15, 15),
            (-16, 16),
            (-32, 21),
            (-64, 26),
            (-127, 31),
            (-128, 31),
            (-1024, 31),
            (1, 0),
            (100, 0),
        ];
        for (relative_position, expected) in cases {
            assert_eq!(
                buckets.bucket(relative_position),
                expected,
                "relative_position={relative_position}"
            );
        }
    }

    #[test]
    fn matches_reference_bidirectional_buckets() {
        let buckets = T5RelativePositionBuckets::new(32, 128, true).expect("valid config");
        let cases = [
            (0, 0),
            (-1, 1),
            (-7, 7),
            (-8, 8),
            (-16, 10),
            (-32, 12),
            (-64, 14),
            (-127, 15),
            (-128, 15),
            (1, 17),
            (7, 23),
            (8, 24),
            (16, 26),
            (32, 28),
            (64, 30),
            (127, 31),
            (128, 31),
        ];
        for (relative_position, expected) in cases {
            assert_eq!(
                buckets.bucket(relative_position),
                expected,
                "relative_position={relative_position}"
            );
        }
    }

    #[test]
    fn explicit_positions_match_signed_displacements() {
        let buckets = T5RelativePositionBuckets::new(32, 128, true).expect("valid config");
        for query in 0..=100usize {
            for key in 0..=100usize {
                let relative = key as i64 - query as i64;
                assert_eq!(
                    buckets.bucket_for_positions(query, key),
                    buckets.bucket(relative)
                );
            }
        }
    }

    #[test]
    fn signed_and_unsigned_extremes_saturate_without_panicking() {
        let bidirectional = T5RelativePositionBuckets::new(32, 128, true).expect("valid config");
        assert_eq!(bidirectional.bucket(i64::MIN), 15);
        assert_eq!(bidirectional.bucket(i64::MAX), 31);
        assert_eq!(bidirectional.bucket_for_positions(usize::MAX, 0), 15);
        assert_eq!(bidirectional.bucket_for_positions(0, usize::MAX), 31);

        let causal = T5RelativePositionBuckets::new(32, 128, false).expect("valid config");
        assert_eq!(causal.bucket(i64::MIN), 31);
        assert_eq!(causal.bucket(i64::MAX), 0);
    }

    #[test]
    fn buckets_are_monotonic_and_saturating_in_each_direction() {
        for num_buckets in [4, 8, 16, 32, 64] {
            let max_distance = num_buckets * 8;
            let bidirectional = T5RelativePositionBuckets::new(num_buckets, max_distance, true)
                .expect("valid geometry");
            let half = num_buckets / 2;
            let mut previous_past = 0;
            let mut previous_future = half;
            for distance in 0..=(max_distance * 4) {
                let past = bidirectional.bucket(-(distance as i64));
                let future = bidirectional.bucket(distance as i64);
                assert!(past >= previous_past && past < half);
                if distance == 0 {
                    assert_eq!(future, 0);
                } else {
                    assert!(future >= previous_future && future < num_buckets);
                    previous_future = future;
                }
                previous_past = past;
            }
            assert_eq!(bidirectional.bucket(-((max_distance * 4) as i64)), half - 1);
            assert_eq!(
                bidirectional.bucket((max_distance * 4) as i64),
                num_buckets - 1
            );
        }
    }

    #[test]
    fn accessors_preserve_validated_configuration() {
        let buckets = T5RelativePositionBuckets::new(32, 128, true).expect("valid config");
        assert_eq!(buckets.num_buckets(), 32);
        assert_eq!(buckets.max_distance(), 128);
        assert!(buckets.is_bidirectional());
    }

    #[test]
    fn validation_errors_explain_invalid_geometry() {
        let messages = [
            T5BucketError::TooFewBuckets {
                num_buckets: 1,
                bidirectional: false,
                minimum: 2,
            }
            .to_string(),
            T5BucketError::TooFewBuckets {
                num_buckets: 2,
                bidirectional: true,
                minimum: 4,
            }
            .to_string(),
            T5BucketError::OddBidirectionalBucketCount { num_buckets: 5 }.to_string(),
            T5BucketError::MaxDistanceNotBeyondExact {
                max_distance: 8,
                max_exact: 8,
            }
            .to_string(),
        ];
        assert_eq!(
            messages[0],
            "T5 causal bucketing requires at least 2 buckets, got 1"
        );
        assert_eq!(
            messages[1],
            "T5 bidirectional bucketing requires at least 4 buckets, got 2"
        );
        assert!(messages[2].contains("must be even"));
        assert!(messages[3].contains("must exceed exact-region boundary"));
    }
}
