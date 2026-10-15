//! Selection arithmetic tests against direct reference definitions.

use super::*;

/// Builds a resolved slice from its dimension, axis multiplicity, and range.
fn slice(dimension: usize, occurrences: usize, range: Range<usize>) -> ResolvedSlice {
    ResolvedSlice {
        dimension,
        occurrences,
        range,
    }
}

/// Length-one ranges never overflow and saturate to an empty range at the maximum index.
#[test]
fn unit_ranges_select_one_index_or_saturate_empty() {
    assert_eq!(unit_range(0), 0..1);
    assert_eq!(unit_range(41), 41..42);
    assert_eq!(unit_range(usize::MAX), usize::MAX..usize::MAX);
    assert_eq!(
        first_ill_formed_slice(&[slice(0, 1, unit_range(usize::MAX))]),
        Some(0)
    );
}

/// The first ambiguous, empty, or duplicate slice is reported by index.
#[test]
fn ill_formed_slices_are_reported_in_order() {
    assert_eq!(first_ill_formed_slice(&[]), None);
    assert_eq!(
        first_ill_formed_slice(&[slice(0, 1, 0..1), slice(2, 1, 1..3)]),
        None
    );
    assert_eq!(first_ill_formed_slice(&[slice(0, 0, 0..1)]), Some(0));
    assert_eq!(first_ill_formed_slice(&[slice(0, 2, 0..1)]), Some(0));
    assert_eq!(
        first_ill_formed_slice(&[slice(0, 1, 0..1), slice(1, 1, 2..2)]),
        Some(1)
    );
    assert_eq!(
        first_ill_formed_slice(&[slice(1, 1, 0..1), slice(0, 1, 0..1), slice(1, 1, 1..2)]),
        Some(2)
    );
}

/// Unselected dimensions stay whole; selected ones take their range; overflows name the slice.
#[test]
fn selection_ranges_expand_and_bound_each_dimension() {
    assert_eq!(
        selection_ranges(&[2, 3, 4], &[]),
        Ok(vec![0..2, 0..3, 0..4])
    );
    assert_eq!(
        selection_ranges(&[2, 3, 4], &[slice(1, 1, 2..3), slice(2, 1, 0..2)]),
        Ok(vec![0..2, 2..3, 0..2])
    );
    assert_eq!(selection_ranges(&[2, 3], &[slice(1, 1, 3..4)]), Err(0));
    assert_eq!(
        selection_ranges(&[2, 3], &[slice(0, 1, 0..1), slice(2, 1, 0..1)]),
        Err(1)
    );
    assert_eq!(selection_ranges(&[2, 3], &[slice(0, 1, 1..1)]), Err(0));
    assert_eq!(selection_ranges(&[], &[]), Ok(Vec::new()));
}

proptest::proptest! {
    /// Well-formedness agrees with its definition: one occurrence, nonempty, distinct dimensions.
    #[test]
    fn first_ill_formed_slice_matches_its_definition(
        raw in proptest::collection::vec((0usize..4, 0usize..3, 0usize..5, 0usize..5), 0..5)
    ) {
        let slices = raw
            .iter()
            .map(|&(dimension, occurrences, start, end)| slice(dimension, occurrences, start..end))
            .collect::<Vec<_>>();
        let expected = slices.iter().enumerate().position(|(index, candidate)| {
            candidate.occurrences != 1
                || candidate.range.start >= candidate.range.end
                || slices[..index]
                    .iter()
                    .any(|earlier| earlier.dimension == candidate.dimension)
        });
        proptest::prop_assert_eq!(first_ill_formed_slice(&slices), expected);
    }

    /// Expanded ranges equal a direct per-dimension reference and always fit their extents.
    #[test]
    fn selection_ranges_match_a_direct_reference(
        extents in proptest::collection::vec(0usize..5, 0..4),
        raw in proptest::collection::vec((0usize..5, 0usize..6, 0usize..6), 0..4),
    ) {
        let slices = raw
            .iter()
            .map(|&(dimension, start, end)| slice(dimension, 1, start..end))
            .collect::<Vec<_>>();
        let first_bad = slices.iter().position(|candidate| {
            candidate.dimension >= extents.len()
                || candidate.range.start >= candidate.range.end
                || candidate.range.end > extents[candidate.dimension]
        });
        match (selection_ranges(&extents, &slices), first_bad) {
            (Err(index), Some(expected)) => proptest::prop_assert_eq!(index, expected),
            (Ok(ranges), None) => {
                proptest::prop_assert_eq!(ranges.len(), extents.len());
                for (dimension, range) in ranges.iter().enumerate() {
                    let expected = slices
                        .iter()
                        .rev()
                        .find(|candidate| candidate.dimension == dimension)
                        .map_or(0..extents[dimension], |candidate| candidate.range.clone());
                    proptest::prop_assert_eq!(range, &expected);
                    proptest::prop_assert!(range.end <= extents[dimension]);
                }
            }
            (actual, expected) => {
                proptest::prop_assert!(false, "{actual:?} disagrees with {expected:?}");
            }
        }
    }
}
