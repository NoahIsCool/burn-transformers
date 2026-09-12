//! Configuration checks for sparse-layer cadence, dense overrides, and router bounds.

use super::*;

/// Sparse cadence selects every second block, while an explicit dense override takes precedence.
#[test]
fn sparse_step_and_dense_overrides_define_layer_kind() {
    let config = Qwen3MoeConfig {
        num_hidden_layers: 4,
        decoder_sparse_step: 2,
        mlp_only_layers: vec![3],
        ..Qwen3MoeConfig::default()
    };
    assert!(!config.is_moe_layer(0));
    assert!(config.is_moe_layer(1));
    assert!(!config.is_moe_layer(3));
}

/// Selecting five experts from a four-expert bank must fail configuration validation.
#[test]
fn rejects_more_selected_experts_than_exist() {
    let config = Qwen3MoeConfig {
        num_experts: 4,
        num_experts_per_tok: 5,
        ..Qwen3MoeConfig::default()
    };
    assert!(config.validate().is_err());
}
