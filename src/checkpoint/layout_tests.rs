//! Exhaustive edge cases and property tests for pure layout arithmetic.

use proptest::prelude::*;

use super::*;

/// Element counts are exact products whose every left-to-right prefix fits in `u64`.
#[test]
fn element_count_is_the_exact_product() {
    assert_eq!(element_count(&[]), Some(1));
    assert_eq!(element_count(&[7]), Some(7));
    assert_eq!(element_count(&[2, 3, 4]), Some(24));
    assert_eq!(element_count(&[u64::MAX]), Some(u64::MAX));
    assert_eq!(element_count(&[u64::MAX, 2]), None);
    assert_eq!(element_count(&[1 << 32, 1 << 32]), None);
    // An overflowing prefix is rejected even when a later zero would make the
    // mathematical product zero, as the pinned SafeTensors validator does.
    assert_eq!(element_count(&[u64::MAX, u64::MAX, 0]), None);
    assert_eq!(element_count(&[u64::MAX, 2, 0]), None);
    // A zero before any overflow keeps every later prefix at zero.
    assert_eq!(element_count(&[0, u64::MAX]), Some(0));
    assert_eq!(element_count(&[u64::MAX, 0, u64::MAX]), Some(0));
}

/// The pinned SafeTensors 0.7.0 validator's element count, `try_fold(1, checked_mul)`.
fn reference_element_count(shape: &[u64]) -> Option<u64> {
    shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
}

/// Storage sizes reject overflow and trailing partial bytes.
#[test]
fn storage_bytes_requires_whole_bytes_without_overflow() {
    assert_eq!(storage_bytes(0, 4), Ok(0));
    assert_eq!(storage_bytes(3, 16), Ok(6));
    assert_eq!(storage_bytes(2, 4), Ok(1));
    assert_eq!(storage_bytes(3, 4), Err(StorageSizeError::PartialByte));
    assert_eq!(storage_bytes(4, 6), Ok(3));
    assert_eq!(storage_bytes(u64::MAX, 8), Err(StorageSizeError::Overflow));
    assert_eq!(storage_bytes(u64::MAX / 8, 8), Ok(u64::MAX / 8));
}

/// Exact tilings are accepted and each violation is located precisely.
#[test]
fn tiling_accepts_exact_coverage_and_locates_violations() {
    assert_eq!(validate_contiguous_tiling(&[], 0), Ok(()));
    assert_eq!(
        validate_contiguous_tiling(&[], 4),
        Err(TilingError::Uncovered { covered: 0 })
    );
    assert_eq!(
        validate_contiguous_tiling(&[(0, 0), (0, 4), (4, 4), (4, 10)], 10),
        Ok(())
    );
    assert_eq!(
        validate_contiguous_tiling(&[(0, 4), (5, 10)], 10),
        Err(TilingError::Discontiguous {
            index: 1,
            expected: 4
        })
    );
    assert_eq!(
        validate_contiguous_tiling(&[(0, 6), (4, 10)], 10),
        Err(TilingError::Discontiguous {
            index: 1,
            expected: 6
        })
    );
    assert_eq!(
        validate_contiguous_tiling(&[(1, 4)], 4),
        Err(TilingError::Discontiguous {
            index: 0,
            expected: 0
        })
    );
    assert_eq!(
        validate_contiguous_tiling(&[(0, 4), (4, 2)], 4),
        Err(TilingError::Reversed { index: 1 })
    );
    assert_eq!(
        validate_contiguous_tiling(&[(0, 12)], 10),
        Err(TilingError::OutOfBounds { index: 0 })
    );
    assert_eq!(
        validate_contiguous_tiling(&[(0, 4), (4, 8)], 10),
        Err(TilingError::Uncovered { covered: 8 })
    );
    assert_eq!(
        validate_contiguous_tiling(&[(0, u64::MAX)], u64::MAX),
        Ok(())
    );
}

/// File offsets add the eight-byte prefix and header without wrapping.
#[test]
fn tensor_file_offset_adds_prefix_and_header() {
    assert_eq!(tensor_file_offset(0, 0), Some(8));
    assert_eq!(tensor_file_offset(100, 24), Some(132));
    assert_eq!(tensor_file_offset(u64::MAX - 8, 0), Some(u64::MAX));
    assert_eq!(tensor_file_offset(u64::MAX - 8, 1), None);
    assert_eq!(tensor_file_offset(u64::MAX, 0), None);
}

/// Slice blocks for a small explicit shape match hand-derived offsets.
#[test]
fn slice_blocks_match_hand_derived_offsets() {
    // Shape [2, 5, 3] sliced to [1, 3) on the middle axis.
    let blocks = (0..2)
        .map(|row| slice_block(row, 2, 5, 3, 1, 2))
        .collect::<Vec<_>>();
    assert_eq!(
        blocks,
        [
            BlockCopy {
                source: 3..9,
                target: 0..6
            },
            BlockCopy {
                source: 18..24,
                target: 6..12
            },
        ]
    );
}

proptest! {
    /// Element counts agree with a widened `u128` product when no prefix overflows.
    #[test]
    fn element_count_matches_wide_product(shape in proptest::collection::vec(any::<u64>().prop_map(|value| value % 70_000), 0..6)) {
        let prefixes_fit = (0..=shape.len()).all(|end| {
            shape[..end]
                .iter()
                .try_fold(1_u128, |product, &dimension| product.checked_mul(u128::from(dimension)))
                .is_some_and(|product| product <= u128::from(u64::MAX))
        });
        let wide = shape.iter().map(|&dimension| u128::from(dimension)).product::<u128>();
        let expected = prefixes_fit.then(|| u64::try_from(wide).unwrap());
        prop_assert_eq!(element_count(&shape), expected);
    }

    /// Element counts match the pinned SafeTensors validator, including zeros
    /// placed after overflowing prefixes.
    #[test]
    fn element_count_matches_the_safetensors_reference(
        shape in proptest::collection::vec(
            prop_oneof![Just(0_u64), 1_u64..5, any::<u64>()],
            0..6,
        )
    ) {
        prop_assert_eq!(element_count(&shape), reference_element_count(&shape));
    }

    /// Successful storage sizes satisfy `8 * bytes == elements * bits` exactly.
    #[test]
    fn storage_bytes_is_exact(elements in any::<u64>(), bits in prop::sample::select(vec![4_u64, 6, 8, 16, 32, 64])) {
        let wide = u128::from(elements) * u128::from(bits);
        match storage_bytes(elements, bits) {
            Ok(bytes) => prop_assert_eq!(u128::from(bytes) * 8, wide),
            Err(StorageSizeError::Overflow) => prop_assert!(wide > u128::from(u64::MAX)),
            Err(StorageSizeError::PartialByte) => {
                prop_assert!(wide <= u128::from(u64::MAX));
                prop_assert_ne!(wide % 8, 0);
            }
        }
    }

    /// Any partition is accepted; moving one boundary or truncating is rejected.
    #[test]
    fn tiling_accepts_partitions_and_rejects_perturbations(
        lengths in proptest::collection::vec(0_u64..50, 1..12),
        victim in any::<prop::sample::Index>(),
        delta in 1_u64..5,
    ) {
        let mut intervals = Vec::new();
        let mut cursor = 0;
        for length in &lengths {
            intervals.push((cursor, cursor + length));
            cursor += length;
        }
        prop_assert_eq!(validate_contiguous_tiling(&intervals, cursor), Ok(()));
        prop_assert!(validate_contiguous_tiling(&intervals, cursor + delta).is_err());
        let index = victim.index(intervals.len());
        let mut shifted = intervals.clone();
        shifted[index].0 += delta;
        shifted[index].1 += delta;
        prop_assert!(validate_contiguous_tiling(&shifted, cursor + delta).is_err());
    }

    /// Slice blocks stay in bounds, partition the target in order, and select the right rows.
    #[test]
    fn slice_blocks_partition_target_and_read_selected_rows(
        outer in 1_usize..6,
        axis in 1_usize..7,
        inner in 0_usize..5,
        start_seed in any::<prop::sample::Index>(),
        length_seed in any::<prop::sample::Index>(),
    ) {
        let start = start_seed.index(axis + 1);
        let length = length_seed.index(axis - start + 1);
        let mut expected_target = 0;
        for row in 0..outer {
            let block = slice_block(row, outer, axis, inner, start, length);
            prop_assert_eq!(block.target.start, expected_target);
            prop_assert_eq!(block.source.len(), length * inner);
            prop_assert_eq!(block.target.len(), length * inner);
            prop_assert!(block.source.end <= outer * axis * inner);
            // Every source element in the block has its axis coordinate in the slice.
            for element in block.source.clone() {
                let axis_coordinate = if inner == 0 { start } else { (element / inner) % axis };
                prop_assert!(axis_coordinate >= start && axis_coordinate < start + length);
                prop_assert_eq!(element / (axis * inner.max(1)), row);
            }
            expected_target = block.target.end;
        }
        prop_assert_eq!(expected_target, outer * length * inner);
    }
}
