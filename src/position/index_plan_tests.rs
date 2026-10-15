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

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(128))]

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
