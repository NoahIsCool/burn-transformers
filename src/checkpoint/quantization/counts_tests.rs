//! Exhaustive and property checks for quantized tensor counts.

use proptest::prelude::*;

use super::*;

/// Ceilings match the overflow-free `div_ceil` on edge values.
#[test]
fn ceilings_match_div_ceil_at_the_edges() {
    for value in [0_u64, 1, 2, 31, 32, 33, u64::MAX - 1, u64::MAX] {
        for divisor in [1_u64, 2, 32, 64, 128, u64::MAX] {
            assert_eq!(ceil_div_u64(value, divisor), value.div_ceil(divisor));
            if let (Ok(value), Ok(divisor)) = (usize::try_from(value), usize::try_from(divisor)) {
                assert_eq!(ceil_div_usize(value, divisor), value.div_ceil(divisor));
            }
        }
    }
}

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
}
