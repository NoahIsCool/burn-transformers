//! Exhaustive and property checks for quantized tensor counts.

use proptest::prelude::*;

use super::*;

/// Wire counts follow the group, packing, and scale formulas.
#[test]
fn wire_counts_follow_the_layout_formulas() {
    assert_eq!(
        wire_counts(65, 32, CodeWidth::Nibble),
        WireCounts {
            group_count: 3,
            packed_value_bytes: 33,
            scale_bytes: Some(12),
        }
    );
    assert_eq!(
        wire_counts(0, 64, CodeWidth::Byte),
        WireCounts {
            group_count: 0,
            packed_value_bytes: 0,
            scale_bytes: Some(0),
        }
    );
    // The supported group sizes never overflow the scale length.
    for group_size in [32, 64, 128] {
        let counts = wire_counts(u64::MAX, group_size, CodeWidth::Byte);
        assert_eq!(counts.group_count, u64::MAX.div_ceil(group_size));
        assert_eq!(counts.scale_bytes, Some(4 * counts.group_count));
    }
    // A unit group size can overflow four bytes per group.
    assert_eq!(wire_counts(u64::MAX, 1, CodeWidth::Byte).scale_bytes, None);
}

/// Burn block counts equal flat group counts exactly for group-aligned widths.
#[test]
fn block_counts_match_flat_groups_for_aligned_matrices() {
    for rows in 0..5_usize {
        for columns in [0_usize, 32, 64, 96, 128, 256] {
            for group_size in [32_usize, 64, 128] {
                let blocks = block_scale_count(rows, columns, group_size).unwrap();
                assert_eq!(blocks, rows * (columns / group_size));
                if columns % group_size == 0 {
                    assert_eq!(blocks, (rows * columns).div_ceil(group_size));
                }
            }
        }
    }
    assert_eq!(block_scale_count(usize::MAX, 64, 32), None);
    assert_eq!(block_scale_count(usize::MAX, 31, 32), Some(0));
}

/// Group indices stay below the group count and contain their element.
#[test]
fn group_indices_stay_below_the_group_count() {
    for elements in 1..200_usize {
        for group_size in [1_usize, 32, 64, 128] {
            for element in 0..elements {
                let group = group_index(element, elements, group_size);
                assert_eq!(group, element / group_size);
                assert!(group < elements.div_ceil(group_size));
            }
        }
    }
}

proptest! {
    /// Wire counts agree with `div_ceil` for arbitrary element counts.
    #[test]
    fn wire_counts_agree_with_div_ceil(
        elements in any::<u64>(),
        group_size in prop::sample::select(vec![32_u64, 64, 128]),
        nibble in any::<bool>(),
    ) {
        let width = if nibble { CodeWidth::Nibble } else { CodeWidth::Byte };
        let counts = wire_counts(elements, group_size, width);
        prop_assert_eq!(counts.group_count, elements.div_ceil(group_size));
        prop_assert_eq!(
            counts.packed_value_bytes,
            if nibble { elements.div_ceil(2) } else { elements }
        );
        prop_assert_eq!(counts.scale_bytes, counts.group_count.checked_mul(4));
    }

    /// Payload splits match the original conversion, checked-add, and length sequence.
    #[test]
    fn payload_splits_match_the_checked_sequence(
        packed in prop_oneof![0_u64..64, Just(u64::MAX)],
        scales in prop_oneof![0_u64..64, Just(u64::MAX)],
        payload_len in 0_usize..130,
    ) {
        let expected = (|| {
            let packed = usize::try_from(packed).map_err(|_| PayloadSplitError::PackedTooLarge)?;
            let scales = usize::try_from(scales).map_err(|_| PayloadSplitError::ScalesTooLarge)?;
            let total = packed.checked_add(scales).ok_or(PayloadSplitError::CombinedOverflow)?;
            if payload_len != total {
                return Err(PayloadSplitError::LengthMismatch { expected: total });
            }
            Ok(PayloadSplit { packed, scales })
        })();
        prop_assert_eq!(payload_split(packed, scales, payload_len), expected);
    }
}

/// Payload splits tile the payload and report the first failed check.
#[test]
fn payload_splits_tile_the_payload() {
    assert_eq!(
        payload_split(6, 8, 14),
        Ok(PayloadSplit {
            packed: 6,
            scales: 8
        })
    );
    assert_eq!(
        payload_split(6, 8, 13),
        Err(PayloadSplitError::LengthMismatch { expected: 14 })
    );
    if usize::BITS == u64::BITS {
        assert_eq!(
            payload_split(u64::MAX, 1, 0),
            Err(PayloadSplitError::CombinedOverflow)
        );
    } else {
        assert_eq!(
            payload_split(u64::MAX, 1, 0),
            Err(PayloadSplitError::PackedTooLarge)
        );
        assert_eq!(
            payload_split(1, u64::MAX, 0),
            Err(PayloadSplitError::ScalesTooLarge)
        );
    }
}
