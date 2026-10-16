//! Nemotron-H checkpoint widths and their overflow order.

use proptest::prelude::*;

use super::*;

/// Extents of a small hybrid model with distinct values in every field.
fn inputs() -> DimensionInputs {
    DimensionInputs {
        vocab: 17,
        hidden: 8,
        intermediate: 12,
        attention_heads: 4,
        key_value_heads: 2,
        head_dim: 3,
        mamba_heads: 4,
        mamba_head_dim: 5,
        groups: 2,
        state_size: 6,
        conv_kernel: 4,
        routed_experts: 3,
        moe_intermediate: 7,
        shared_intermediate: 9,
        latent: None,
    }
}

/// Fused widths of one configuration in contract order.
type Widths = [u64; 5];

/// The former contract-builder arithmetic, reporting the first failing term.
fn reference(inputs: DimensionInputs) -> Result<Widths, DimensionError> {
    let values = inputs
        .mamba_heads
        .checked_mul(inputs.mamba_head_dim)
        .ok_or(DimensionError::MambaIntermediate)? as u64;
    let states = inputs
        .groups
        .checked_mul(inputs.state_size)
        .ok_or(DimensionError::StateGroups)? as u64;
    let convolution = values
        .checked_add(
            states
                .checked_mul(2)
                .ok_or(DimensionError::ConvolutionStates)?,
        )
        .ok_or(DimensionError::Convolution)?;
    let projection = values
        .checked_add(convolution)
        .and_then(|value| value.checked_add(inputs.mamba_heads as u64))
        .ok_or(DimensionError::Projection)?;
    let query = inputs
        .attention_heads
        .checked_mul(inputs.head_dim)
        .ok_or(DimensionError::AttentionQuery)? as u64;
    let key_value = inputs
        .key_value_heads
        .checked_mul(inputs.head_dim)
        .ok_or(DimensionError::AttentionKeyValue)? as u64;
    Ok([values, convolution, projection, query, key_value])
}

/// Projects the fused widths of a verified result.
fn widths(dimensions: Dimensions) -> Widths {
    [
        dimensions.mamba_intermediate,
        dimensions.mamba_convolution,
        dimensions.mamba_projection,
        dimensions.attention_query,
        dimensions.attention_kv,
    ]
}

/// The Mamba split widths, attention widths, and expert input are exact.
#[test]
fn checkpoint_dimensions_are_exact() {
    let dimensions = checkpoint_dimensions(inputs()).unwrap();
    // I = 20, 2 G N = 24, C_conv = 44, in_proj = 20 + 44 + 4 = 68.
    assert_eq!(widths(dimensions), [20, 44, 68, 12, 6]);
    assert_eq!(dimensions.expert_input, 8);
    assert_eq!(dimensions.double_hidden, Some(16));
    assert_eq!(
        [
            dimensions.vocab,
            dimensions.hidden,
            dimensions.intermediate,
            dimensions.mamba_heads,
            dimensions.mamba_kernel,
            dimensions.routed_experts,
            dimensions.moe_intermediate,
            dimensions.shared_intermediate,
        ],
        [17, 8, 12, 4, 4, 3, 7, 9]
    );
    let latent = DimensionInputs {
        latent: Some(5),
        ..inputs()
    };
    assert_eq!(checkpoint_dimensions(latent).unwrap().expert_input, 5);
}

/// Each overflowing term is reported with its expression and operation.
#[test]
fn checkpoint_dimensions_report_the_first_overflow() {
    let huge = usize::MAX;
    let cases = [
        (
            DimensionInputs {
                mamba_heads: huge,
                ..inputs()
            },
            DimensionError::MambaIntermediate,
        ),
        (
            DimensionInputs {
                groups: huge,
                ..inputs()
            },
            DimensionError::StateGroups,
        ),
        (
            DimensionInputs {
                groups: huge,
                state_size: 1,
                ..inputs()
            },
            DimensionError::ConvolutionStates,
        ),
        (
            DimensionInputs {
                attention_heads: huge,
                ..inputs()
            },
            DimensionError::AttentionQuery,
        ),
        (
            DimensionInputs {
                key_value_heads: huge,
                ..inputs()
            },
            DimensionError::AttentionKeyValue,
        ),
    ];
    for (inputs, error) in cases {
        if usize::BITS == 64 {
            assert_eq!(checkpoint_dimensions(inputs).err(), Some(error));
        }
        assert_eq!(checkpoint_dimensions(inputs).err(), reference(inputs).err());
    }
    assert_eq!(
        DimensionError::ConvolutionStates.reason(),
        "dimension product overflows u64"
    );
    assert_eq!(DimensionError::Projection.field(), "mamba projection");
    let wide = DimensionInputs {
        hidden: huge,
        ..inputs()
    };
    if usize::BITS == 64 {
        assert_eq!(checkpoint_dimensions(wide).unwrap().double_hidden, None);
    }
}

proptest! {
    /// The verified widths equal the former contract-builder arithmetic.
    #[test]
    fn checkpoint_dimensions_match_the_former_arithmetic(
        mamba_heads in prop_oneof![0_usize..6, Just(usize::MAX / 2), Just(usize::MAX)],
        mamba_head_dim in prop_oneof![0_usize..6, Just(1_usize)],
        groups in prop_oneof![0_usize..4, Just(usize::MAX / 4), Just(usize::MAX)],
        state_size in prop_oneof![0_usize..4, Just(1_usize)],
        attention_heads in prop_oneof![0_usize..6, Just(usize::MAX)],
        key_value_heads in prop_oneof![0_usize..6, Just(usize::MAX)],
        head_dim in 0_usize..4,
        hidden in prop_oneof![0_usize..9, Just(usize::MAX)],
    ) {
        let inputs = DimensionInputs {
            mamba_heads,
            mamba_head_dim,
            groups,
            state_size,
            attention_heads,
            key_value_heads,
            head_dim,
            hidden,
            ..inputs()
        };
        prop_assert_eq!(checkpoint_dimensions(inputs).map(widths), reference(inputs));
        if let Ok(dimensions) = checkpoint_dimensions(inputs) {
            prop_assert_eq!(dimensions.double_hidden, (hidden as u64).checked_mul(2));
        }
    }
}

/// Every overflow names its configuration expression and the representation it exceeded.
#[test]
fn every_dimension_error_names_its_expression_and_representation() {
    let product_usize = "dimension product overflows usize";
    for (error, field, reason) in [
        (
            DimensionError::MambaIntermediate,
            "mamba_num_heads * mamba_head_dim",
            product_usize,
        ),
        (
            DimensionError::StateGroups,
            "n_groups * ssm_state_size",
            product_usize,
        ),
        (
            DimensionError::ConvolutionStates,
            "mamba convolution",
            "dimension product overflows u64",
        ),
        (
            DimensionError::Convolution,
            "mamba convolution",
            "dimension sum overflows u64",
        ),
        (
            DimensionError::Projection,
            "mamba projection",
            "dimension sum overflows u64",
        ),
        (
            DimensionError::AttentionQuery,
            "num_attention_heads * head_dim",
            product_usize,
        ),
        (
            DimensionError::AttentionKeyValue,
            "num_key_value_heads * head_dim",
            product_usize,
        ),
    ] {
        assert_eq!(
            (error.field(), error.reason()),
            (field, reason),
            "{error:?}"
        );
    }
}
