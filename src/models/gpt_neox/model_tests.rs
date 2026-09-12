//! Edited GPT-NeoX attention geometry fails before backend tensor operations.

use burn::backend::Flex;

use super::*;

/// Small CPU backend used to test metadata validation without large allocations.
type B = Flex<f32, i32>;

/// Initializes ordinary tiny parameters before tests edit only attention metadata.
fn model() -> GptNeoXModel<B> {
    GptNeoXModel::init(
        GptNeoXConfig {
            vocab_size: 13,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 12,
            rotary_pct: 0.5,
            max_position_embeddings: 16,
            ..Default::default()
        },
        &Default::default(),
    )
    .expect("the tiny topology must initialize")
}

/// Zero, incompatible, or residual-changing heads return contextual configuration errors.
#[test]
fn invalid_attention_geometry_is_rejected_before_projection_validation() {
    for (hidden_size, query_heads, key_value_heads, head_dim) in [
        (7, 2, 2, 4),
        (8, 0, 2, 4),
        (8, 2, 0, 4),
        (8, 2, 2, 0),
        (8, 3, 2, 4),
    ] {
        let mut model = model();
        let attention = &mut model.layers[0].attention;
        attention.hidden_size = hidden_size;
        attention.query_heads = query_heads;
        attention.key_value_heads = key_value_heads;
        attention.head_dim = head_dim;
        let error = model.validate_parameters().expect_err("invalid geometry");
        assert!(error.to_string().contains("gpt_neox.layers.0.attention"));
        assert!(error.to_string().contains("nonzero, compatible"));
    }
}

/// Multiplication overflow in query or key/value geometry returns an error without allocation.
#[test]
fn projection_width_overflow_is_reported_from_edited_head_counts() {
    for key_value_heads in [1, usize::MAX] {
        let mut model = model();
        model.layers[0].attention.query_heads = usize::MAX;
        model.layers[0].attention.key_value_heads = key_value_heads;
        let error = model.validate_parameters().expect_err("overflowing width");
        assert!(error.to_string().contains("gpt_neox.layers.0.attention"));
        assert!(error.to_string().contains("projection widths overflow"));
    }
}

/// A fitting ungated query width can still overflow when the output gate doubles it.
#[test]
fn gated_projection_width_overflow_is_reported_separately() {
    let mut model = model();
    let attention = &mut model.layers[0].attention;
    attention.query_heads = usize::MAX / 2 + 1;
    attention.key_value_heads = 1;
    attention.head_dim = 1;
    attention.output_gate = AttentionOutputGate::Sigmoid;
    let error = model
        .validate_parameters()
        .expect_err("gated width overflow");
    assert!(error.to_string().contains("gpt_neox.layers.0"));
    assert!(error.to_string().contains("gated query width overflows"));
}

/// An unchanged initialized model satisfies all projection and normalization contracts.
#[test]
fn initialized_attention_parameters_pass_validation() {
    model()
        .validate_parameters()
        .expect("valid tiny parameters");
}
