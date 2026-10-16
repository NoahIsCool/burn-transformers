//! T5 checkpoint widths and block sublayer numbering.

use proptest::prelude::*;

use super::*;

/// Extents of a small T5 model with distinct values in every field.
fn inputs() -> DimensionInputs {
    DimensionInputs {
        vocab: 11,
        model: 8,
        heads: 4,
        head_dim: 3,
        feed_forward: 12,
        buckets: 32,
    }
}

/// The attention width is exact and plain extents are copied unchanged.
#[test]
fn checkpoint_dimensions_are_exact() {
    let dimensions = checkpoint_dimensions(inputs()).unwrap();
    assert_eq!(
        [
            dimensions.vocab,
            dimensions.model,
            dimensions.attention,
            dimensions.feed_forward,
            dimensions.buckets,
            dimensions.heads,
        ],
        [11, 8, 12, 12, 32, 4]
    );
}

/// The feed-forward network follows the block's attention sublayers.
#[test]
fn feed_forward_follows_the_attention_sublayers() {
    assert_eq!(feed_forward_sublayer(false), 1);
    assert_eq!(feed_forward_sublayer(true), 2);
}

proptest! {
    /// The verified width equals the former checked `u64` product.
    #[test]
    fn attention_width_matches_the_former_product(
        heads in prop_oneof![0_usize..9, Just(usize::MAX)],
        head_dim in prop_oneof![0_usize..9, Just(usize::MAX), Just(2_usize)],
    ) {
        let inputs = DimensionInputs { heads, head_dim, ..inputs() };
        let expected = [heads, head_dim]
            .iter()
            .try_fold(1_u64, |product, &value| product.checked_mul(value as u64))
            .ok_or(AttentionWidthOverflow);
        prop_assert_eq!(
            checkpoint_dimensions(inputs).map(|dimensions| dimensions.attention),
            expected
        );
    }
}
