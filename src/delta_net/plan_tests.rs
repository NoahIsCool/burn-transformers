//! DeltaNet prefill planning: chunk plans, inverse schedules, widths, and geometry checks.

use super::*;

/// Short prompts are one unpadded chunk; long ones pad less than one chunk.
#[test]
fn chunk_plans_cover_the_tokens_with_minimal_padding() {
    assert_eq!(
        chunk_plan(10, 64),
        ChunkPlan {
            chunk: 10,
            chunks: 1,
            padded: 10
        }
    );
    assert_eq!(
        chunk_plan(10, 4),
        ChunkPlan {
            chunk: 4,
            chunks: 3,
            padded: 12
        }
    );
    assert_eq!(
        chunk_plan(128, 64),
        ChunkPlan {
            chunk: 64,
            chunks: 2,
            padded: 128
        }
    );
    assert_eq!(
        chunk_plan(1, usize::MAX),
        ChunkPlan {
            chunk: 1,
            chunks: 1,
            padded: 1
        }
    );
}

/// The doubling schedule halves from the least power of two down to two.
#[test]
fn inverse_schedules_halve_from_the_padded_power_of_two() {
    assert_eq!(inverse_block_sizes(2), [2]);
    assert_eq!(inverse_block_sizes(3), [4, 2]);
    assert_eq!(inverse_block_sizes(4), [4, 2]);
    assert_eq!(inverse_block_sizes(5), [8, 4, 2]);
    assert_eq!(inverse_block_sizes(64), [64, 32, 16, 8, 4, 2]);
    assert_eq!(inverse_block_sizes(65)[0], 128);
}

/// The fused convolution width is exact or rejected on overflow.
#[test]
fn convolution_channels_are_exact_or_rejected() {
    assert_eq!(
        convolution_channels(16, 128, 32, 128),
        Some(2 * 16 * 128 + 32 * 128)
    );
    assert_eq!(convolution_channels(usize::MAX, 2, 1, 1), None);
    assert_eq!(convolution_channels(usize::MAX / 2 + 1, 1, 0, 0), None);
    assert_eq!(convolution_channels(1, 1, usize::MAX - 1, 1), None);
    assert_eq!(
        convolution_channels(1, 1, usize::MAX - 2, 1),
        Some(usize::MAX)
    );
}

/// Zero widths are reported before uneven head grouping.
#[test]
fn delta_geometry_reports_the_first_violation() {
    assert_eq!(check_delta_geometry(8, 2, 4, 4, 4, 4), Ok(()));
    assert_eq!(check_delta_geometry(8, 2, 4, 4, 4, 1), Ok(()));
    for fields in [
        [0, 2, 4, 4, 4, 4],
        [8, 0, 4, 4, 4, 4],
        [8, 2, 0, 4, 4, 4],
        [8, 2, 4, 0, 4, 4],
        [8, 2, 4, 4, 0, 4],
        [8, 2, 4, 4, 4, 0],
        [8, 3, 4, 0, 4, 4],
    ] {
        let [hidden, key_heads, value_heads, key_dim, value_dim, kernel] = fields;
        assert_eq!(
            check_delta_geometry(hidden, key_heads, value_heads, key_dim, value_dim, kernel),
            Err(DeltaGeometryViolation::ZeroWidth)
        );
    }
    assert_eq!(
        check_delta_geometry(8, 3, 4, 4, 4, 4),
        Err(DeltaGeometryViolation::ValueHeads)
    );
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(256))]

    /// Chunk plans match `div_ceil` and pad by less than one chunk.
    #[test]
    fn chunk_plans_match_div_ceil(sequence in 1usize..5000, chunk_size in 1usize..300) {
        let plan = chunk_plan(sequence, chunk_size);
        proptest::prop_assert_eq!(plan.chunk, chunk_size.min(sequence));
        proptest::prop_assert_eq!(plan.chunks, sequence.div_ceil(plan.chunk));
        proptest::prop_assert_eq!(plan.padded, plan.chunks * plan.chunk);
        proptest::prop_assert!(sequence <= plan.padded && plan.padded < sequence + plan.chunk);
    }

    /// The schedule starts at `next_power_of_two` and halves exactly to two.
    #[test]
    fn inverse_schedule_matches_next_power_of_two(width in 2usize..5000) {
        let sizes = inverse_block_sizes(width);
        proptest::prop_assert_eq!(sizes[0], width.next_power_of_two());
        proptest::prop_assert_eq!(*sizes.last().unwrap(), 2);
        for pair in sizes.windows(2) {
            proptest::prop_assert_eq!(pair[0], 2 * pair[1]);
        }
    }

    /// The width check matches wide arithmetic near the overflow boundary.
    #[test]
    fn convolution_channels_match_wide_arithmetic(
        key_heads in proptest::prop_oneof![0usize..64, (usize::MAX / 8)..=usize::MAX],
        key_head_dim in 0usize..4,
        value_heads in proptest::prop_oneof![0usize..64, (usize::MAX / 8)..=usize::MAX],
        value_head_dim in 0usize..4,
    ) {
        let wide = 2 * key_heads as u128 * key_head_dim as u128
            + value_heads as u128 * value_head_dim as u128;
        let expected = (wide <= usize::MAX as u128).then_some(wide as usize);
        proptest::prop_assert_eq!(
            convolution_channels(key_heads, key_head_dim, value_heads, value_head_dim),
            expected
        );
    }
}
