//! Muse-Glimmer checkpoint widths and their overflow order.

use proptest::prelude::*;

use super::*;

/// Extents of a small language tower with distinct values in every field.
fn inputs() -> DimensionInputs {
    DimensionInputs {
        vocab: 11,
        hidden: 8,
        intermediate: 12,
        attention_heads: 4,
        key_value_heads: 2,
        head_dim: 3,
    }
}

/// Attention widths are exact and plain extents are copied unchanged.
#[test]
fn checkpoint_dimensions_are_exact() {
    let dimensions = checkpoint_dimensions(inputs()).unwrap();
    assert_eq!(
        [
            dimensions.vocab,
            dimensions.hidden,
            dimensions.intermediate,
            dimensions.query,
            dimensions.key_value,
        ],
        [11, 8, 12, 12, 6]
    );
}

proptest! {
    /// The verified widths equal the former checked `usize` products, query first.
    #[test]
    fn checkpoint_dimensions_match_the_former_products(
        attention_heads in prop_oneof![0_usize..9, Just(usize::MAX)],
        key_value_heads in prop_oneof![0_usize..9, Just(usize::MAX)],
        head_dim in prop_oneof![0_usize..9, Just(usize::MAX / 2)],
    ) {
        let inputs = DimensionInputs { attention_heads, key_value_heads, head_dim, ..inputs() };
        let expected = attention_heads
            .checked_mul(head_dim)
            .ok_or(DimensionError::Query)
            .and_then(|query| {
                key_value_heads
                    .checked_mul(head_dim)
                    .map(|key_value| (query as u64, key_value as u64))
                    .ok_or(DimensionError::KeyValue)
            });
        prop_assert_eq!(
            checkpoint_dimensions(inputs).map(|dimensions| (dimensions.query, dimensions.key_value)),
            expected
        );
    }
}

/// Each overflow names the configuration expression reported by the contract.
#[test]
fn dimension_errors_name_their_expressions() {
    assert_eq!(
        DimensionError::Query.field(),
        "num_attention_heads * head_dim"
    );
    assert_eq!(
        DimensionError::KeyValue.field(),
        "num_key_value_heads * head_dim"
    );
}
