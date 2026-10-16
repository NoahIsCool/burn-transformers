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

/// Range products match `element_count` on the corresponding subslice.
#[test]
fn range_products_match_subslice_element_counts() {
    let shapes: [&[u64]; 5] = [
        &[],
        &[2, 3, 4],
        &[u64::MAX, 2, 0],
        &[0, u64::MAX, 3],
        &[1 << 40, 1 << 30, 5],
    ];
    for shape in shapes {
        for start in 0..=shape.len() {
            for end in start..=shape.len() {
                assert_eq!(
                    checked_range_product(shape, start, end),
                    element_count(&shape[start..end]),
                    "{shape:?}[{start}..{end}]"
                );
            }
        }
    }
}

/// The axis-view check sequence written directly with `try_fold` and `try_from`.
fn reference_axis_view(shape: &[u64], axis: usize) -> Result<AxisExtents, AxisViewError> {
    if axis >= shape.len() {
        return Err(AxisViewError::AxisOutOfBounds);
    }
    let fold = |dimensions: &[u64]| {
        dimensions
            .iter()
            .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
    };
    let outer = fold(&shape[..axis]).ok_or(AxisViewError::OuterOverflow)?;
    let outer =
        usize::try_from(outer).map_err(|_| AxisViewError::OuterTooLarge { value: outer })?;
    let axis_len = usize::try_from(shape[axis])
        .map_err(|_| AxisViewError::AxisTooLarge { value: shape[axis] })?;
    let inner = fold(&shape[axis + 1..]).ok_or(AxisViewError::InnerOverflow)?;
    let inner =
        usize::try_from(inner).map_err(|_| AxisViewError::InnerTooLarge { value: inner })?;
    outer
        .checked_mul(axis_len)
        .and_then(|rows| rows.checked_mul(inner))
        .ok_or(AxisViewError::ElementsTooLarge)?;
    Ok(AxisExtents {
        outer,
        axis: axis_len,
        inner,
    })
}

/// The original runtime slice-plan checks, written directly.
fn reference_slice_plan(
    source: &[u64],
    target: &[u64],
    axis: usize,
    start: u64,
    length: u64,
) -> Result<(), SlicePlanError> {
    let axis_len = *source.get(axis).ok_or(SlicePlanError::AxisOutOfBounds)?;
    let end = start
        .checked_add(length)
        .ok_or(SlicePlanError::EndOverflow)?;
    if end > axis_len {
        return Err(SlicePlanError::PastAxis { end });
    }
    let mut expected = source.to_vec();
    expected[axis] = length;
    if target != expected {
        return Err(SlicePlanError::ShapeMismatch);
    }
    Ok(())
}

/// Slice plans accept in-range slices with the replaced shape and name the first failure.
#[test]
fn slice_plans_check_range_then_shape() {
    assert_eq!(check_slice_plan(&[4, 6], &[4, 2], 1, 3, 2), Ok(()));
    assert_eq!(check_slice_plan(&[4, 6], &[4, 0], 1, 6, 0), Ok(()));
    assert_eq!(
        check_slice_plan(&[4, 6], &[4, 2], 2, 0, 2),
        Err(SlicePlanError::AxisOutOfBounds)
    );
    assert_eq!(
        check_slice_plan(&[4, u64::MAX], &[4, 2], 1, u64::MAX, 2),
        Err(SlicePlanError::EndOverflow)
    );
    assert_eq!(
        check_slice_plan(&[4, 6], &[4, 2], 1, 5, 2),
        Err(SlicePlanError::PastAxis { end: 7 })
    );
    assert_eq!(
        check_slice_plan(&[4, 6], &[2, 2], 1, 0, 2),
        Err(SlicePlanError::ShapeMismatch)
    );
    assert_eq!(
        check_slice_plan(&[4, 6], &[4, 2, 1], 1, 0, 2),
        Err(SlicePlanError::ShapeMismatch)
    );
}

/// Axis views report exact extents and name the first failed check.
#[test]
fn axis_views_split_shapes_and_name_the_first_failure() {
    assert_eq!(
        axis_view(&[2, 5, 3], 1),
        Ok(AxisExtents {
            outer: 2,
            axis: 5,
            inner: 3
        })
    );
    assert_eq!(
        axis_view(&[7], 0),
        Ok(AxisExtents {
            outer: 1,
            axis: 7,
            inner: 1
        })
    );
    assert_eq!(axis_view(&[2, 3], 2), Err(AxisViewError::AxisOutOfBounds));
    assert_eq!(
        axis_view(&[u64::MAX, 2, 1], 2),
        Err(AxisViewError::OuterOverflow)
    );
    assert_eq!(
        axis_view(&[1, 2, u64::MAX, 2], 1),
        Err(AxisViewError::InnerOverflow)
    );
    // A later zero does not hide an overflowing running product on either side.
    assert_eq!(
        axis_view(&[1 << 40, 1 << 40, 0, 5], 3),
        Err(AxisViewError::OuterOverflow)
    );
    assert_eq!(
        axis_view(&[5, 1 << 40, 1 << 40, 0], 0),
        Err(AxisViewError::InnerOverflow)
    );
    // A zero inner extent keeps the element count at zero.
    assert_eq!(
        axis_view(&[3, 4, 0, 9], 1),
        Ok(AxisExtents {
            outer: 3,
            axis: 4,
            inner: 0
        })
    );
    if usize::BITS < u64::BITS {
        let huge = u64::from(u32::MAX) + 1;
        assert_eq!(
            axis_view(&[huge, 1], 1),
            Err(AxisViewError::OuterTooLarge { value: huge })
        );
        assert_eq!(
            axis_view(&[1, huge], 0),
            Err(AxisViewError::InnerTooLarge { value: huge })
        );
        assert_eq!(
            axis_view(&[huge], 0),
            Err(AxisViewError::AxisTooLarge { value: huge })
        );
    } else {
        // `outer * axis` overflows `usize` even though the inner extent is zero.
        assert_eq!(
            axis_view(&[1 << 40, 1 << 40, 0], 1),
            Err(AxisViewError::ElementsTooLarge)
        );
        assert_eq!(
            axis_view(&[1 << 30, 1 << 30, 1 << 10], 1),
            Err(AxisViewError::ElementsTooLarge)
        );
    }
}

/// Axis element counts check the partial and full products separately.
#[test]
fn axis_elements_reject_either_overflowing_product() {
    assert_eq!(axis_elements(2, 5, 3), Some(30));
    assert_eq!(axis_elements(usize::MAX, 2, 0), None);
    assert_eq!(axis_elements(usize::MAX, 1, 1), Some(usize::MAX));
    assert_eq!(axis_elements(usize::MAX, 1, 2), None);
    assert_eq!(axis_elements(0, usize::MAX, usize::MAX), Some(0));
}

/// Range containment is exact and never overflows.
#[test]
fn range_within_checks_the_mathematical_end() {
    assert!(range_within(0, 0, 0));
    assert!(range_within(2, 3, 5));
    assert!(!range_within(3, 3, 5));
    assert!(!range_within(usize::MAX, 1, usize::MAX));
    assert!(range_within(usize::MAX, 0, usize::MAX));
}

/// Concatenation extends the axis only for compatible sources and leaves failures untouched.
#[test]
fn concatenation_extends_the_axis_or_leaves_the_shape_unchanged() {
    let mut shape = vec![2, 0, 4];
    assert_eq!(extend_concatenation(&mut shape, &[2, 3, 4], 1), Ok(()));
    assert_eq!(extend_concatenation(&mut shape, &[2, 5, 4], 1), Ok(()));
    assert_eq!(shape, [2, 8, 4]);
    for incompatible in [&[2, 1][..], &[2, 1, 4, 1], &[3, 1, 4], &[2, 1, 5]] {
        assert_eq!(
            extend_concatenation(&mut shape, incompatible, 1),
            Err(ConcatError::Incompatible)
        );
        assert_eq!(shape, [2, 8, 4]);
    }
    assert_eq!(
        extend_concatenation(&mut shape, &[2, u64::MAX, 4], 1),
        Err(ConcatError::Overflow)
    );
    assert_eq!(shape, [2, 8, 4]);
    // An incompatible shape is reported even when the axis would also overflow.
    assert_eq!(
        extend_concatenation(&mut shape, &[3, u64::MAX, 4], 1),
        Err(ConcatError::Incompatible)
    );
}

/// Dense byte lengths reject overflowing running products and byte counts.
#[test]
fn dense_byte_lengths_are_exact_or_rejected() {
    assert_eq!(dense_byte_len(&[], 4), Some(4));
    assert_eq!(dense_byte_len(&[2, 3], 2), Some(12));
    assert_eq!(dense_byte_len(&[usize::MAX, 2, 0], 4), None);
    assert_eq!(dense_byte_len(&[0, usize::MAX], 8), Some(0));
    assert_eq!(dense_byte_len(&[usize::MAX], 1), Some(usize::MAX));
    assert_eq!(dense_byte_len(&[usize::MAX], 2), None);
    assert_eq!(checked_host_product(&[usize::MAX, 2, 0]), None);
    assert_eq!(checked_host_product(&[3, 0, usize::MAX]), Some(0));
}

/// The data section follows the prefix and header and reaches the end of the file.
#[test]
fn data_sections_start_after_the_header_and_reach_the_file_end() {
    assert_eq!(
        data_section(10, 30),
        Ok(DataSection {
            start: 18,
            bytes: 12
        })
    );
    assert_eq!(
        data_section(10, 18),
        Ok(DataSection {
            start: 18,
            bytes: 0
        })
    );
    assert_eq!(
        data_section(0, 30),
        Err(DataSectionError::InvalidHeaderLength)
    );
    assert_eq!(
        data_section(23, 30),
        Err(DataSectionError::InvalidHeaderLength)
    );
    assert_eq!(
        data_section(u64::MAX - 7, u64::MAX),
        Err(DataSectionError::Overflow)
    );
    assert_eq!(
        data_section(u64::MAX - 8, u64::MAX),
        Ok(DataSection {
            start: u64::MAX,
            bytes: 0
        })
    );
}

proptest! {
    /// Axis views agree with independently computed prefix and suffix products.
    #[test]
    fn axis_views_match_independent_products(
        shape in proptest::collection::vec(0_u64..9, 1..6),
        axis_seed in any::<prop::sample::Index>(),
    ) {
        let axis = axis_seed.index(shape.len());
        let view = axis_view(&shape, axis).expect("small shapes fit");
        let product = |dimensions: &[u64]| dimensions.iter().product::<u64>();
        prop_assert_eq!(view.outer as u64, product(&shape[..axis]));
        prop_assert_eq!(view.axis as u64, shape[axis]);
        prop_assert_eq!(view.inner as u64, product(&shape[axis + 1..]));
        prop_assert_eq!((view.outer * view.axis * view.inner) as u64, product(&shape));
    }

    /// Slice plans give the original verdict on small and overflowing plans.
    #[test]
    fn slice_plans_match_the_original_checks(
        source in proptest::collection::vec(0_u64..5, 0..4),
        target in proptest::collection::vec(0_u64..5, 0..4),
        axis in 0_usize..4,
        start in prop_oneof![0_u64..6, Just(u64::MAX)],
        length in 0_u64..6,
    ) {
        prop_assert_eq!(
            check_slice_plan(&source, &target, axis, start, length),
            reference_slice_plan(&source, &target, axis, start, length)
        );
    }

    /// Axis views classify overflowing and zero-bearing shapes like a direct reference.
    #[test]
    fn axis_views_classify_like_the_reference(
        shape in proptest::collection::vec(
            prop::sample::select(vec![0_u64, 1, 3, 1 << 20, 1 << 32, 1 << 40, u64::MAX]),
            0..6,
        ),
        axis in 0_usize..7,
    ) {
        prop_assert_eq!(axis_view(&shape, axis), reference_axis_view(&shape, axis));
    }

    /// Dense byte lengths match the original running-product fold.
    #[test]
    fn dense_byte_lengths_match_the_fold(
        shape in proptest::collection::vec(any::<usize>().prop_map(|value| value % 100_000), 0..6),
        width in 1_usize..9,
    ) {
        let reference = shape
            .iter()
            .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
            .and_then(|elements| elements.checked_mul(width));
        prop_assert_eq!(dense_byte_len(&shape, width), reference);
    }
}
