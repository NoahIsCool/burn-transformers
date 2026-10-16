//! Qwen3-MoE checkpoint widths and their overflow order.

use proptest::prelude::*;

use super::*;

/// Extents of a small routed model with distinct values in every field.
fn inputs() -> DimensionInputs {
    DimensionInputs {
        vocab: 11,
        hidden: 8,
        intermediate: 12,
        moe_intermediate: 6,
        experts: 5,
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
            dimensions.moe_intermediate,
            dimensions.experts,
            dimensions.query,
            dimensions.key_value,
            dimensions.head_dim,
        ],
        [11, 8, 12, 6, 5, 12, 6, 3]
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

/// Each overflow names the configuration product reported by the contract diagnostic.
#[test]
fn dimension_errors_name_their_configuration_products() {
    assert_eq!(
        DimensionError::Query.field(),
        "num_attention_heads * head_dim"
    );
    assert_eq!(
        DimensionError::KeyValue.field(),
        "num_key_value_heads * head_dim"
    );
    let key_value_overflow = DimensionInputs {
        attention_heads: 1,
        key_value_heads: usize::MAX,
        head_dim: 2,
        ..inputs()
    };
    assert_eq!(
        checkpoint_dimensions(key_value_overflow).err(),
        Some(DimensionError::KeyValue)
    );
}
