//! Relative-bucket geometry and displacement tables.

use proptest::prelude::*;

use super::*;

/// Released T5 layouts pass; each defect is reported in validation order.
#[test]
fn bucket_geometry_reports_each_defect_in_order() {
    assert_eq!(bucket_geometry(32, 128, true), Ok(()));
    assert_eq!(bucket_geometry(4, 3, true), Ok(()));
    assert_eq!(
        bucket_geometry(3, 128, true),
        Err(BucketGeometryError::TooFewBuckets)
    );
    assert_eq!(
        bucket_geometry(33, 128, true),
        Err(BucketGeometryError::OddBuckets)
    );
    assert_eq!(
        bucket_geometry(32, 8, false),
        Err(BucketGeometryError::EncoderDistance)
    );
    assert_eq!(
        bucket_geometry(32, 16, true),
        Err(BucketGeometryError::DecoderDistance)
    );
    // An encoder-only stack needs only the encoder bound.
    assert_eq!(bucket_geometry(32, 16, false), Ok(()));
}

/// The radius is the smaller bound and the width one more, unless it overflows.
#[test]
fn displacement_table_takes_the_smaller_radius() {
    assert_eq!(displacement_table(5, 128), Some((5, 6)));
    assert_eq!(displacement_table(500, 128), Some((128, 129)));
    assert_eq!(displacement_table(0, 0), Some((0, 1)));
    assert_eq!(displacement_table(usize::MAX, usize::MAX), None);
    assert_eq!(displacement_table(usize::MAX, 7), Some((7, 8)));
}

/// Layers share the learned table of the first layer with their head count.
#[test]
fn bias_owners_share_tables_by_head_count() {
    assert_eq!(bias_owners(&[8, 8, 4, 8, 4, 2]), [0, 0, 2, 0, 2, 5]);
    assert!(bias_owners(&[]).is_empty());
    assert_eq!(bias_owners(&[6]), [0]);
}

proptest! {
    /// Owners equal the map-based first occurrence they replaced.
    #[test]
    fn bias_owners_match_first_occurrences(heads in proptest::collection::vec(1_usize..4, 0..10)) {
        let mut first = std::collections::BTreeMap::new();
        let expected = heads
            .iter()
            .enumerate()
            .map(|(index, &count)| *first.entry(count).or_insert(index))
            .collect::<Vec<_>>();
        prop_assert_eq!(bias_owners(&heads), expected);
    }

    /// The geometry check equals the configuration validator's former inline checks.
    #[test]
    fn bucket_geometry_matches_the_inline_checks(
        buckets in 0_usize..80,
        max_distance in 0_usize..60,
        decoder in any::<bool>(),
    ) {
        let expected = if buckets < 4 {
            Err(BucketGeometryError::TooFewBuckets)
        } else if buckets % 2 != 0 {
            Err(BucketGeometryError::OddBuckets)
        } else if max_distance <= buckets / 4 {
            Err(BucketGeometryError::EncoderDistance)
        } else if decoder && max_distance <= buckets / 2 {
            Err(BucketGeometryError::DecoderDistance)
        } else {
            Ok(())
        };
        prop_assert_eq!(bucket_geometry(buckets, max_distance, decoder), expected);
    }

    /// Every clamped displacement indexes the `2r + 1`-entry table.
    #[test]
    fn clamped_displacements_index_the_table(
        bound in 0_usize..40,
        max_distance in 0_usize..40,
        displacement in -100_i64..100,
    ) {
        let (radius, width) = displacement_table(bound, max_distance).unwrap();
        prop_assert_eq!(radius, bound.min(max_distance));
        let radius = i64::try_from(radius).unwrap();
        let index = displacement.clamp(-radius, radius) + radius;
        prop_assert!(index >= 0);
        prop_assert!(usize::try_from(index).unwrap() < 2 * width - 1);
    }
}
