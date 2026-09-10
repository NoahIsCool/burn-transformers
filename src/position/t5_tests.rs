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
