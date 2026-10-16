//! DeepSeek-V2 checkpoint widths and their overflow order.

use proptest::prelude::*;

use super::*;

/// Extents of a small latent-attention model with distinct values in every field.
fn inputs() -> DimensionInputs {
    DimensionInputs {
        vocab: 13,
        hidden: 8,
        intermediate: 12,
        moe_intermediate: 6,
        shared_experts: 2,
        routed_experts: 4,
        attention_heads: 3,
        qk_nope: 5,
        qk_rope: 2,
        value_head_dim: 7,
        query_rank: Some(9),
        kv_rank: 10,
    }
}

/// Fused widths of one configuration in contract order.
type Widths = [u64; 6];

/// The former contract-builder arithmetic, reporting the first failing term.
fn reference(inputs: DimensionInputs) -> Result<Widths, DimensionError> {
    let query_head = inputs
        .qk_nope
        .checked_add(inputs.qk_rope)
        .ok_or(DimensionError::QueryHead)?;
    let expanded_head = inputs
        .qk_nope
        .checked_add(inputs.value_head_dim)
        .ok_or(DimensionError::ExpandedHead)?;
    let shared = inputs
        .moe_intermediate
        .checked_mul(inputs.shared_experts)
        .ok_or(DimensionError::SharedIntermediate)?;
    let query = inputs
        .attention_heads
        .checked_mul(query_head)
        .ok_or(DimensionError::Query)?;
    let compressed = inputs
        .kv_rank
        .checked_add(inputs.qk_rope)
        .ok_or(DimensionError::Compressed)?;
    let expanded = inputs
        .attention_heads
        .checked_mul(expanded_head)
        .ok_or(DimensionError::ExpandedKeyValue)?;
    let output = inputs
        .attention_heads
        .checked_mul(inputs.value_head_dim)
        .ok_or(DimensionError::OutputHeads)?;
    Ok([shared, query, compressed, expanded, output, inputs.kv_rank].map(|width| width as u64))
}

/// Projects the fused widths of a verified result.
fn widths(dimensions: Dimensions) -> Widths {
    [
        dimensions.shared_intermediate,
        dimensions.query,
        dimensions.compressed,
        dimensions.expanded_kv,
        dimensions.output_heads,
        dimensions.kv_rank,
    ]
}

/// Fused widths are exact and plain extents are copied unchanged.
#[test]
fn checkpoint_dimensions_are_exact() {
    let dimensions = checkpoint_dimensions(inputs()).unwrap();
    assert_eq!(widths(dimensions), [12, 21, 12, 36, 21, 10]);
    assert_eq!(
        [
            dimensions.vocab,
            dimensions.hidden,
            dimensions.intermediate,
            dimensions.moe_intermediate,
            dimensions.experts,
        ],
        [13, 8, 12, 6, 4]
    );
    assert_eq!(dimensions.query_rank, Some(9));
    let direct = DimensionInputs {
        query_rank: None,
        ..inputs()
    };
    assert_eq!(checkpoint_dimensions(direct).unwrap().query_rank, None);
}

/// Each overflowing term is reported with its configuration expression and operation.
#[test]
fn checkpoint_dimensions_report_the_first_overflow() {
    let huge = usize::MAX;
    let cases = [
        (
            DimensionInputs {
                qk_nope: huge,
                ..inputs()
            },
            DimensionError::QueryHead,
        ),
        (
            DimensionInputs {
                value_head_dim: huge,
                ..inputs()
            },
            DimensionError::ExpandedHead,
        ),
        (
            DimensionInputs {
                shared_experts: huge,
                ..inputs()
            },
            DimensionError::SharedIntermediate,
        ),
        (
            DimensionInputs {
                attention_heads: huge / 4,
                qk_nope: 0,
                qk_rope: 8,
                value_head_dim: 1,
                ..inputs()
            },
            DimensionError::Query,
        ),
        (
            DimensionInputs {
                kv_rank: huge,
                ..inputs()
            },
            DimensionError::Compressed,
        ),
        (
            DimensionInputs {
                attention_heads: huge / 4,
                qk_nope: 0,
                qk_rope: 1,
                value_head_dim: 8,
                ..inputs()
            },
            DimensionError::ExpandedKeyValue,
        ),
    ];
    for (inputs, error) in cases {
        assert_eq!(checkpoint_dimensions(inputs).err(), Some(error));
        assert_eq!(reference(inputs).err(), Some(error));
    }
    assert_eq!(
        DimensionError::Compressed.field(),
        "kv_lora_rank + qk_rope_head_dim"
    );
    assert_eq!(
        DimensionError::Compressed.reason(),
        "dimension sum overflows usize"
    );
    assert_eq!(
        DimensionError::OutputHeads.reason(),
        "dimension product overflows usize"
    );
}

proptest! {
    /// The verified widths equal the former contract-builder arithmetic.
    #[test]
    fn checkpoint_dimensions_match_the_former_arithmetic(
        attention_heads in prop_oneof![0_usize..5, Just(usize::MAX / 4)],
        qk_nope in prop_oneof![0_usize..9, Just(usize::MAX)],
        qk_rope in prop_oneof![0_usize..9, Just(usize::MAX)],
        value_head_dim in prop_oneof![0_usize..9, Just(usize::MAX)],
        moe_intermediate in prop_oneof![0_usize..9, Just(usize::MAX)],
        shared_experts in 0_usize..4,
        kv_rank in prop_oneof![0_usize..9, Just(usize::MAX)],
    ) {
        let inputs = DimensionInputs {
            attention_heads,
            qk_nope,
            qk_rope,
            value_head_dim,
            moe_intermediate,
            shared_experts,
            kv_rank,
            ..inputs()
        };
        prop_assert_eq!(checkpoint_dimensions(inputs).map(widths), reference(inputs));
    }
}
