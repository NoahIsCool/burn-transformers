//! Executable checks of the Verus-verified projection-width arithmetic.

use super::*;

/// Ordinary and gated layouts report exact widths.
#[test]
fn widths_are_exact_for_plain_and_gated_queries() {
    assert_eq!(
        projection_widths(16, 8, 2, 4, false),
        Ok(ProjectionWidths {
            query_width: 32,
            key_value_width: 8,
            query_output_width: 32,
        })
    );
    assert_eq!(
        projection_widths(16, 8, 2, 4, true).map(|widths| widths.query_output_width),
        Ok(64)
    );
}

/// Each failure names its own reason, in check order.
#[test]
fn width_errors_are_precise() {
    for shape in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
        assert_eq!(
            projection_widths(shape.0, shape.1, shape.2, shape.3, false),
            Err(WidthError::EmptyAxis)
        );
    }
    assert_eq!(
        projection_widths(4, 3, 2, 2, false),
        Err(WidthError::UngroupedHeads)
    );
    for (hidden, queries, keys, width, gated) in [
        (8, usize::MAX, 1, 2, false),
        (1, usize::MAX / 2 + 1, 1, 1, true),
        (usize::MAX, 2, 1, 1, false),
        (2, 1, 1, usize::MAX / 2 + 1, false),
    ] {
        assert_eq!(
            projection_widths(hidden, queries, keys, width, gated),
            Err(WidthError::Overflow),
            "{hidden} {queries} {keys} {width} {gated}"
        );
    }
}

proptest::proptest! {
    /// Success implies grouping and representable matrices; failure implies the stated cause.
    #[test]
    fn widths_agree_with_wide_arithmetic(
        hidden in 0usize..=64,
        queries in 0usize..=16,
        keys in 0usize..=16,
        width in 0usize..=64,
        gated in proptest::bool::ANY,
    ) {
        let result = projection_widths(hidden, queries, keys, width, gated);
        if hidden == 0 || queries == 0 || keys == 0 || width == 0 {
            proptest::prop_assert_eq!(result, Err(WidthError::EmptyAxis));
        } else if queries % keys != 0 {
            proptest::prop_assert_eq!(result, Err(WidthError::UngroupedHeads));
        } else {
            let widths = result.unwrap();
            let factor = if gated { 2 } else { 1 };
            proptest::prop_assert_eq!(widths.query_output_width, queries * width * factor);
            proptest::prop_assert_eq!(widths.key_value_width, keys * width);
        }
    }
}
