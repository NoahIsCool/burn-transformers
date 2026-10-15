//! Edited GPT-NeoX attention widths.

use proptest::prelude::*;

use super::*;

/// Valid layers report their widths; each defect is reported in order.
#[test]
fn layer_widths_report_each_defect() {
    assert_eq!(
        layer_widths(768, 768, 12, 12, 64, false),
        Ok(LayerWidths {
            query: 768,
            key_value: 768,
            query_projection: 768,
        })
    );
    assert_eq!(
        layer_widths(768, 768, 12, 4, 64, true).map(|widths| widths.query_projection),
        Ok(1536)
    );
    assert_eq!(
        layer_widths(512, 768, 12, 12, 64, false),
        Err(LayerWidthError::Geometry)
    );
    assert_eq!(
        layer_widths(768, 768, 12, 5, 64, false),
        Err(LayerWidthError::Geometry)
    );
    assert_eq!(
        layer_widths(768, 768, 0, 1, 64, false),
        Err(LayerWidthError::Geometry)
    );
    assert_eq!(
        layer_widths(8, 8, 2, 1, usize::MAX, false),
        Err(LayerWidthError::Overflow)
    );
    assert_eq!(
        layer_widths(8, 8, 1, 1, usize::MAX, true),
        Err(LayerWidthError::GatedOverflow)
    );
}

proptest! {
    /// The check equals the parameter validation's former inline arithmetic.
    #[test]
    fn layer_widths_match_the_inline_checks(
        hidden_size in 0_usize..4,
        residual in 0_usize..4,
        query_heads in prop_oneof![0_usize..9, Just(usize::MAX)],
        key_value_heads in prop_oneof![0_usize..9, Just(1_usize)],
        head_dim in prop_oneof![0_usize..9, Just(usize::MAX / 2), Just(usize::MAX)],
        gated in any::<bool>(),
    ) {
        let expected = if hidden_size != residual
            || query_heads == 0
            || key_value_heads == 0
            || head_dim == 0
            || !query_heads.is_multiple_of(key_value_heads)
        {
            Err(LayerWidthError::Geometry)
        } else {
            match query_heads
                .checked_mul(head_dim)
                .zip(key_value_heads.checked_mul(head_dim))
            {
                None => Err(LayerWidthError::Overflow),
                Some((query, key_value)) => match query.checked_mul(if gated { 2 } else { 1 }) {
                    None => Err(LayerWidthError::GatedOverflow),
                    Some(query_projection) => Ok(LayerWidths {
                        query,
                        key_value,
                        query_projection,
                    }),
                },
            }
        };
        prop_assert_eq!(
            layer_widths(hidden_size, residual, query_heads, key_value_heads, head_dim, gated),
            expected
        );
    }
}
