//! Qwen3.5 checkpoint widths and their overflow order.

use proptest::prelude::*;

use super::*;

/// Extents of a small hybrid model with distinct values in every field.
fn inputs() -> DimensionInputs {
    DimensionInputs {
        vocab: 11,
        hidden: 8,
        intermediate: 12,
        attention_heads: 4,
        key_value_heads: 2,
        head_dim: 3,
        delta_key_heads: 2,
        delta_key_head_dim: 5,
        delta_value_heads: 3,
        delta_value_head_dim: 7,
        conv_kernel: 4,
    }
}

/// The former checked product of the contract builder.
fn reference_product(left: usize, right: usize) -> Option<u64> {
    [left, right]
        .iter()
        .try_fold(1_u64, |product, &value| product.checked_mul(value as u64))
}

/// The former checked sum of the contract builder.
fn reference_sum(values: &[u64]) -> Option<u64> {
    values
        .iter()
        .try_fold(0_u64, |sum, &value| sum.checked_add(value))
}

/// The former contract-builder arithmetic, reporting the first failing term.
fn reference(inputs: DimensionInputs) -> Result<[u64; 7], DimensionError> {
    let query =
        reference_product(inputs.attention_heads, inputs.head_dim).ok_or(DimensionError::Query)?;
    let key_value = reference_product(inputs.key_value_heads, inputs.head_dim)
        .ok_or(DimensionError::KeyValue)?;
    let delta_key = reference_product(inputs.delta_key_heads, inputs.delta_key_head_dim)
        .ok_or(DimensionError::DeltaKey)?;
    let delta_value = reference_product(inputs.delta_value_heads, inputs.delta_value_head_dim)
        .ok_or(DimensionError::DeltaValue)?;
    let delta_conv =
        reference_sum(&[delta_key, delta_key, delta_value]).ok_or(DimensionError::DeltaConv)?;
    let gated_query = reference_sum(&[query, query]).ok_or(DimensionError::GatedQuery)?;
    let hidden = inputs.hidden as u64;
    let double_hidden = reference_sum(&[hidden, hidden]).ok_or(DimensionError::DoubleHidden)?;
    Ok([
        query,
        gated_query,
        key_value,
        delta_value,
        delta_conv,
        double_hidden,
        hidden,
    ])
}

/// Fused widths are exact and plain extents are copied unchanged.
#[test]
fn checkpoint_dimensions_are_exact() {
    let dimensions = checkpoint_dimensions(inputs()).unwrap();
    assert_eq!(
        [
            dimensions.vocab,
            dimensions.hidden,
            dimensions.intermediate,
            dimensions.head_dim,
            dimensions.delta_value_heads,
            dimensions.delta_value_head_dim,
            dimensions.conv_kernel,
        ],
        [11, 8, 12, 3, 3, 7, 4]
    );
    assert_eq!(
        [
            dimensions.query,
            dimensions.gated_query,
            dimensions.key_value,
            dimensions.delta_value,
            dimensions.delta_conv,
            dimensions.double_hidden,
        ],
        [12, 24, 6, 21, 41, 16]
    );
    assert_eq!(query_projection(&dimensions, false), 12);
    assert_eq!(query_projection(&dimensions, true), 24);
}

/// Each overflowing term is reported, earliest first.
#[test]
fn checkpoint_dimensions_report_the_first_overflow() {
    let huge = usize::MAX;
    let half = usize::MAX / 2 + 1;
    let cases = [
        (
            DimensionInputs {
                attention_heads: huge,
                key_value_heads: huge,
                ..inputs()
            },
            DimensionError::Query,
        ),
        (
            DimensionInputs {
                key_value_heads: huge,
                ..inputs()
            },
            DimensionError::KeyValue,
        ),
        (
            DimensionInputs {
                delta_key_heads: huge,
                ..inputs()
            },
            DimensionError::DeltaKey,
        ),
        (
            DimensionInputs {
                delta_value_heads: huge,
                ..inputs()
            },
            DimensionError::DeltaValue,
        ),
        (
            DimensionInputs {
                delta_key_heads: half,
                delta_key_head_dim: 1,
                ..inputs()
            },
            DimensionError::DeltaConv,
        ),
        (
            DimensionInputs {
                attention_heads: half,
                head_dim: 1,
                key_value_heads: 1,
                ..inputs()
            },
            DimensionError::GatedQuery,
        ),
        (
            DimensionInputs {
                hidden: half,
                ..inputs()
            },
            DimensionError::DoubleHidden,
        ),
    ];
    for (inputs, error) in cases {
        if usize::BITS == 64 {
            assert_eq!(checkpoint_dimensions(inputs).unwrap_err(), error);
        }
        assert_eq!(reference(inputs).err(), checkpoint_dimensions(inputs).err());
    }
}

proptest! {
    /// The verified widths equal the former contract-builder arithmetic.
    #[test]
    fn checkpoint_dimensions_match_the_former_arithmetic(
        attention_heads in prop_oneof![0_usize..9, Just(usize::MAX), Just(usize::MAX / 2 + 1)],
        key_value_heads in prop_oneof![0_usize..9, Just(usize::MAX)],
        head_dim in prop_oneof![0_usize..9, Just(usize::MAX / 3)],
        delta_key_heads in prop_oneof![0_usize..9, Just(usize::MAX / 2 + 1)],
        delta_key_head_dim in prop_oneof![0_usize..9, Just(usize::MAX)],
        delta_value_heads in prop_oneof![0_usize..9, Just(usize::MAX)],
        delta_value_head_dim in 0_usize..9,
        hidden in prop_oneof![0_usize..9, Just(usize::MAX), Just(usize::MAX / 2 + 1)],
    ) {
        let inputs = DimensionInputs {
            attention_heads,
            key_value_heads,
            head_dim,
            delta_key_heads,
            delta_key_head_dim,
            delta_value_heads,
            delta_value_head_dim,
            hidden,
            ..inputs()
        };
        let verified = checkpoint_dimensions(inputs).map(|dimensions| {
            [
                dimensions.query,
                dimensions.gated_query,
                dimensions.key_value,
                dimensions.delta_value,
                dimensions.delta_conv,
                dimensions.double_hidden,
                dimensions.hidden,
            ]
        });
        prop_assert_eq!(verified, reference(inputs));
    }
}

/// Every overflow names its configuration expression in the contract diagnostic.
#[test]
fn every_dimension_error_names_its_expression() {
    let cases = [
        (DimensionError::Query, "num_attention_heads * head_dim"),
        (DimensionError::KeyValue, "num_key_value_heads * head_dim"),
        (
            DimensionError::DeltaKey,
            "linear_num_key_heads * linear_key_head_dim",
        ),
        (
            DimensionError::DeltaValue,
            "linear_num_value_heads * linear_value_head_dim",
        ),
        (
            DimensionError::DeltaConv,
            "2 * delta key width + delta value width",
        ),
        (DimensionError::GatedQuery, "2 * query width"),
        (DimensionError::DoubleHidden, "2 * hidden_size"),
    ];
    for (error, field) in cases {
        assert_eq!(error.field(), field, "{error:?}");
    }
}

/// The convolution width overflows when the doubled key width fits but adding values does not.
#[test]
fn convolution_width_overflow_at_the_value_term() {
    if usize::BITS != 64 {
        return;
    }
    let inputs = DimensionInputs {
        delta_key_heads: 1 << 62,
        delta_key_head_dim: 1,
        delta_value_heads: 1 << 63,
        delta_value_head_dim: 1,
        ..inputs()
    };
    assert_eq!(
        checkpoint_dimensions(inputs).err(),
        Some(DimensionError::DeltaConv)
    );
    assert_eq!(reference(inputs).err(), Some(DimensionError::DeltaConv));
}
