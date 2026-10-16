//! Exhaustive and property checks for multimodal-RoPE axes and T5 displacement indices.

use super::*;

/// Reproduces the original loop formulation: H at `1, 4, ...` below `3 * h`, W at `2, 5, ...`.
fn loop_reference_axes(half: usize, sections: [usize; 3]) -> Vec<usize> {
    let mut axes = vec![0; half];
    for index in (1..sections[1].saturating_mul(3).min(half)).step_by(3) {
        axes[index] = 1;
    }
    for index in (2..sections[2].saturating_mul(3).min(half)).step_by(3) {
        axes[index] = 2;
    }
    axes
}

/// Qwen3-VL's interleaved `[24, 20, 20]` split over 64 frequencies gives each axis its count.
#[test]
fn qwen_sections_interleave_height_and_width() {
    let axes = (0..64)
        .map(|index| mrope_frequency_axis(index, 20, 20))
        .collect::<Vec<_>>();
    assert_eq!(&axes[..6], &[0, 1, 2, 0, 1, 2]);
    assert_eq!(axes, loop_reference_axes(64, [24, 20, 20]));
    for (axis, count) in [(0, 24), (1, 20), (2, 20)] {
        assert_eq!(axes.iter().filter(|&&value| value == axis).count(), count);
    }
}

/// Saturating section counts never overflow and still follow the period-three pattern.
#[test]
fn huge_section_counts_do_not_overflow() {
    for index in 0..30 {
        let axis = mrope_frequency_axis(index, usize::MAX, usize::MAX);
        assert_eq!(axis, index % 3);
    }
    assert_eq!(mrope_frequency_axis(usize::MAX, 0, 0), 0);
}

/// Empty and one-sided tables are rejected, while representable sums are exact.
#[test]
fn relative_table_length_rejects_empty_and_overflowing_geometry() {
    assert_eq!(relative_table_length(0, 4), None);
    assert_eq!(relative_table_length(4, 0), None);
    assert_eq!(relative_table_length(1, 1), Some(1));
    assert_eq!(relative_table_length(3, 5), Some(7));
    assert_eq!(relative_table_length(1, usize::MAX), Some(usize::MAX));
    assert_eq!(relative_table_length(2, usize::MAX), None);
}

/// Section totals are exact when representable and absent on overflow.
#[test]
fn mrope_section_totals_are_exact_or_absent() {
    assert_eq!(mrope_section_total([24, 20, 20]), Some(64));
    assert_eq!(mrope_section_total([0, 0, 0]), Some(0));
    assert_eq!(mrope_section_total([usize::MAX, 0, 0]), Some(usize::MAX));
    assert_eq!(mrope_section_total([usize::MAX, 1, 0]), None);
    assert_eq!(mrope_section_total([1, usize::MAX, 0]), None);
    assert_eq!(mrope_section_total([usize::MAX - 1, 0, 2]), None);
}

/// Rotary tables fit an operand only with matching batch, sequence, unit heads, and an even prefix.
#[test]
fn rotary_operand_check_requires_every_shape_relation() {
    let operand = [2, 4, 5, 8];
    assert!(rotary_operand_fits(operand, [2, 1, 5, 6], [2, 1, 5, 6], 6));
    assert!(rotary_operand_fits(operand, [2, 1, 5, 8], [2, 1, 5, 8], 8));
    for (cosine, sine, rotary) in [
        ([3, 1, 5, 6], [3, 1, 5, 6], 6),
        ([2, 2, 5, 6], [2, 2, 5, 6], 6),
        ([2, 1, 4, 6], [2, 1, 4, 6], 6),
        ([2, 1, 5, 6], [2, 1, 5, 6], 4),
        ([2, 1, 5, 6], [3, 1, 5, 6], 6),
        ([2, 1, 5, 6], [2, 2, 5, 6], 6),
        ([2, 1, 5, 6], [2, 1, 4, 6], 6),
        ([2, 1, 5, 6], [2, 1, 5, 4], 6),
        ([2, 1, 5, 10], [2, 1, 5, 10], 10),
        ([2, 1, 5, 5], [2, 1, 5, 5], 5),
    ] {
        assert!(!rotary_operand_fits(operand, cosine, sine, rotary));
    }
    assert!(split_half_width_fits(2));
    assert!(!split_half_width_fits(0));
    assert!(!split_half_width_fits(7));
}

/// Relative-table faults follow the documented order and plans carry the exact anchor.
#[test]
fn relative_table_plan_orders_faults_and_reports_anchor() {
    assert_eq!(
        plan_relative_table(0, usize::MAX, usize::MAX),
        Err(RelativeTableFault::EmptyAxis)
    );
    assert_eq!(
        plan_relative_table(2, usize::MAX, usize::MAX),
        Err(RelativeTableFault::OutputOverflow)
    );
    // `q + k - 1` fits but the `q * k` output plane does not.
    assert_eq!(
        plan_relative_table(1 << 33, 1 << 32, 0),
        Err(RelativeTableFault::OutputOverflow)
    );
    assert_eq!(
        plan_relative_table(2, 3, usize::MAX - 1),
        Err(RelativeTableFault::OffsetOverflow)
    );
    assert_eq!(
        plan_relative_table(1, usize::MAX, 0),
        Err(RelativeTableFault::IndexRange)
    );
    assert_eq!(
        plan_relative_table(3, 5, 10),
        Ok(RelativeTablePlan {
            length: 7,
            anchor: 12
        })
    );
    assert_eq!(
        plan_relative_table(1, 1 << 62, usize::MAX - 1),
        Ok(RelativeTablePlan {
            length: 1 << 62,
            anchor: usize::MAX - 1
        })
    );
}

/// Bias table shapes report the first failed check.
#[test]
fn relative_bias_shape_reports_first_fault() {
    assert_eq!(relative_bias_shape_fault(8, 32), None);
    assert_eq!(
        relative_bias_shape_fault(0, usize::MAX),
        Some(RelativeBiasShapeFault::ZeroHeads)
    );
    assert_eq!(
        relative_bias_shape_fault(2, usize::MAX),
        Some(RelativeBiasShapeFault::BucketIndexRange)
    );
    assert_eq!(
        relative_bias_shape_fault(4, i64::MAX as usize),
        Some(RelativeBiasShapeFault::TableOverflow)
    );
    assert_eq!(relative_bias_shape_fault(1, i64::MAX as usize), None);
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(128))]

    /// Axis counts follow the verified closed form `min(h, (n + 1) / 3)` and `min(w, n / 3)`.
    #[test]
    fn axis_counts_match_closed_form(
        half in 0usize..200,
        height in 0usize..80,
        width in 0usize..80,
    ) {
        let mut counts = [0usize; 3];
        for index in 0..half {
            counts[mrope_frequency_axis(index, height, width)] += 1;
        }
        let height_count = height.min((half + 1) / 3);
        let width_count = width.min(half / 3);
        proptest::prop_assert_eq!(counts, [half - height_count - width_count, height_count, width_count]);
    }

    /// Offset query blocks read the entry whose anchor displacement is the true displacement.
    #[test]
    fn offset_table_entries_carry_true_displacements(
        query_length in 1usize..48,
        key_length in 1usize..48,
        query_offset in 0usize..1000,
    ) {
        let plan = plan_relative_table(query_length, key_length, query_offset).expect("small geometry");
        let mut read = vec![false; plan.length];
        for query in 0..query_length {
            for key in 0..key_length {
                let entry = relative_table_index(query, key, query_length);
                proptest::prop_assert!(entry < plan.length);
                proptest::prop_assert_eq!(
                    entry as i64 - plan.anchor as i64,
                    key as i64 - (query_offset + query) as i64
                );
                read[entry] = true;
            }
        }
        proptest::prop_assert!(read.into_iter().all(|entry_read| entry_read));
    }

    /// The closed-form axis selector equals the original masking loops for any sections.
    #[test]
    fn axis_selector_matches_loop_reference(
        half in 0usize..96,
        height in 0usize..40,
        width in 0usize..40,
    ) {
        let expected = loop_reference_axes(half, [0, height, width]);
        for (index, expected) in expected.into_iter().enumerate() {
            proptest::prop_assert_eq!(mrope_frequency_axis(index, height, width), expected);
        }
    }

    /// Every lookup lies in the table and preserves the key-minus-query displacement.
    #[test]
    fn table_index_preserves_displacement_and_bounds(
        query_length in 1usize..64,
        key_length in 1usize..64,
        query_seed in 0usize..1000,
        key_seed in 0usize..1000,
    ) {
        let query = query_seed % query_length;
        let key = key_seed % key_length;
        let entry = relative_table_index(query, key, query_length);
        let length = relative_table_length(query_length, key_length).expect("small geometry");
        proptest::prop_assert!(entry < length);
        let anchor = query_length - 1;
        proptest::prop_assert_eq!(
            entry as i64 - anchor as i64,
            key as i64 - query as i64
        );
    }
}
