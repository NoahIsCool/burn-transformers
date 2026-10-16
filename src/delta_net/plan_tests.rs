//! DeltaNet prefill planning: chunk plans, inverse schedules and levels, widths, mixer layouts,
//! and geometry checks.

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

/// Each inverse level tiles the padded tile with twice the blocks of the previous one.
#[test]
fn inverse_levels_double_the_block_count() {
    let level = |block, count| InverseLevel {
        block,
        half: block / 2,
        count,
    };
    assert_eq!(inverse_levels(2), [level(2, 1)]);
    assert_eq!(inverse_levels(3), [level(4, 1), level(2, 2)]);
    assert_eq!(inverse_levels(5), [level(8, 1), level(4, 2), level(2, 4)]);
    assert_eq!(inverse_levels(64).len(), 6);
    assert_eq!(*inverse_levels(64).last().unwrap(), level(2, 32));
}

/// The fused projection widths are exact or rejected on overflow.
#[test]
fn projection_widths_are_exact_or_rejected() {
    assert_eq!(
        projection_widths(16, 128, 32, 128),
        Some(ProjectionWidths {
            key_width: 16 * 128,
            value_width: 32 * 128,
            channels: 2 * 16 * 128 + 32 * 128,
        })
    );
    assert_eq!(projection_widths(usize::MAX, 2, 1, 1), None);
    assert_eq!(projection_widths(usize::MAX / 2 + 1, 1, 0, 0), None);
    assert_eq!(projection_widths(1, 1, usize::MAX - 1, 1), None);
    assert_eq!(
        projection_widths(1, 1, usize::MAX - 2, 1),
        Some(ProjectionWidths {
            key_width: 1,
            value_width: usize::MAX - 2,
            channels: usize::MAX,
        })
    );
}

/// Query, key, and value slices tile the convolved channels; the state size is exact.
#[test]
fn mixer_layout_tiles_the_fused_channels() {
    let layout = mixer_layout(2, 4, 6, 3);
    assert_eq!(
        layout,
        MixerLayout {
            key_width: 8,
            value_start: 16,
            value_width: 18,
        }
    );
    assert_eq!(state_elements(6, 4, 3), 72);
    assert_eq!(state_elements(0, 4, 3), 0);
    let widths = projection_widths(2, 4, 6, 3).expect("small widths");
    assert_eq!(layout.value_start + layout.value_width, widths.channels);
}

/// Each key head repeats into an equal, contiguous run of value heads.
#[test]
fn head_repetitions_tile_the_value_heads() {
    assert_eq!(head_repetitions(32, 16), 2);
    assert_eq!(head_repetitions(16, 16), 1);
    assert_eq!(head_repetitions(0, 4), 0);
    for key_heads in 1..9 {
        assert_eq!(
            head_repetitions(key_heads * 6, key_heads) * key_heads,
            key_heads * 6
        );
    }
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
    fn projection_widths_match_wide_arithmetic(
        key_heads in proptest::prop_oneof![0usize..64, (usize::MAX / 8)..=usize::MAX],
        key_head_dim in 0usize..4,
        value_heads in proptest::prop_oneof![0usize..64, (usize::MAX / 8)..=usize::MAX],
        value_head_dim in 0usize..4,
    ) {
        let wide = 2 * key_heads as u128 * key_head_dim as u128
            + value_heads as u128 * value_head_dim as u128;
        let expected = (wide <= usize::MAX as u128).then_some(wide as usize);
        proptest::prop_assert_eq!(
            projection_widths(key_heads, key_head_dim, value_heads, value_head_dim)
                .map(|widths| widths.channels),
            expected
        );
        if let Some(widths) = projection_widths(key_heads, key_head_dim, value_heads, value_head_dim) {
            proptest::prop_assert_eq!(widths.key_width, key_heads * key_head_dim);
            proptest::prop_assert_eq!(widths.value_width, value_heads * value_head_dim);
        }
    }

    /// Levels match the schedule: `2^k` blocks of the `k`-th size, each the next level's parent.
    #[test]
    fn inverse_levels_match_the_schedule(width in 2usize..5000) {
        let sizes = inverse_block_sizes(width);
        let levels = inverse_levels(width);
        proptest::prop_assert_eq!(levels.len(), sizes.len());
        for (k, (level, &size)) in levels.iter().zip(&sizes).enumerate() {
            proptest::prop_assert_eq!(level.block, size);
            proptest::prop_assert_eq!(level.half * 2, size);
            proptest::prop_assert_eq!(level.count, 1 << k);
            proptest::prop_assert_eq!(level.count * level.block, sizes[0]);
        }
    }

    /// The mixer layout ends at the fused channel count for any small positive geometry.
    #[test]
    fn mixer_layout_matches_projection_widths(
        key_heads in 1usize..9,
        key_head_dim in 1usize..9,
        value_heads in 1usize..9,
        value_head_dim in 1usize..9,
    ) {
        let layout = mixer_layout(key_heads, key_head_dim, value_heads, value_head_dim);
        let widths = projection_widths(key_heads, key_head_dim, value_heads, value_head_dim)
            .expect("small widths");
        proptest::prop_assert_eq!(layout.key_width, widths.key_width);
        proptest::prop_assert_eq!(layout.value_start, 2 * widths.key_width);
        proptest::prop_assert_eq!(layout.value_width, widths.value_width);
        proptest::prop_assert_eq!(layout.value_start + layout.value_width, widths.channels);
        proptest::prop_assert_eq!(
            state_elements(value_heads, key_head_dim, value_head_dim),
            value_heads * key_head_dim * value_head_dim
        );
    }
}
