//! Nemotron-H schedule normalization, configuration rejection, and pinned schema tests.

use super::*;

/// Published 52-block Nemotron 3/3.5 topology: 23 Mamba, 23 MoE, 6 GQA.
const OFFICIAL_30B_PATTERN: &str = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";

/// Legacy characters deserialize to ordered block types and reconstruct unchanged.
#[test]
fn legacy_pattern_round_trips_to_normalized_blocks() {
    let json = r#"{
            "model_type":"nemotron_h",
            "hybrid_override_pattern":"M-*E",
            "num_hidden_layers":4
        }"#;
    let config: NemotronHConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.hybrid_override_pattern(), "M-*E");
    config.validate().unwrap();
}

/// A declared depth that disagrees with the configured block schedule is rejected.
#[test]
fn rejects_declared_depth_that_conflicts_with_pattern() {
    let config = NemotronHConfig {
        num_hidden_layers: Some(99),
        ..NemotronHConfig::default()
    };
    assert!(config.validate().is_err());
}

/// Runtime-specific recurrence and routing constraints fail during config validation.
#[test]
fn rejects_unsupported_recurrent_and_shared_expert_edges() {
    let config = NemotronHConfig {
        conv_kernel: 1,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    let config = NemotronHConfig {
        n_shared_experts: 2,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    let config = NemotronHConfig {
        n_routed_experts: 2,
        n_group: 2,
        topk_group: 2,
        num_experts_per_tok: 2,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

/// The pinned Nemotron 3 Nano legacy-pattern config remains schema compatible.
#[test]
fn nemotron_3_nano_official_shape_config_is_compatible() {
    let json = format!(
        r#"{{
                "model_type":"nemotron_h",
                "vocab_size":131072,
                "hidden_size":2688,
                "hybrid_override_pattern":"{OFFICIAL_30B_PATTERN}",
                "num_hidden_layers":52,
                "num_attention_heads":32,
                "num_key_value_heads":2,
                "head_dim":128,
                "max_position_embeddings":262144,
                "intermediate_size":1856,
                "ssm_state_size":128,
                "mamba_num_heads":64,
                "mamba_head_dim":64,
                "n_groups":8,
                "conv_kernel":4,
                "chunk_size":128,
                "n_routed_experts":128,
                "n_shared_experts":1,
                "moe_intermediate_size":1856,
                "moe_shared_expert_intermediate_size":3712,
                "num_experts_per_tok":6,
                "routed_scaling_factor":2.5,
                "n_group":1,
                "topk_group":1,
                "norm_eps":0.00001
            }}"#
    );
    let config: NemotronHConfig = serde_json::from_str(&json).unwrap();
    config.validate().unwrap();
    assert_eq!(config.block_count(), 52);
    assert_eq!(
        config
            .layers_block_type
            .iter()
            .filter(|&&kind| kind == NemotronHBlockType::LinearAttention)
            .count(),
        23
    );
    assert_eq!(
        config
            .layers_block_type
            .iter()
            .filter(|&&kind| kind == NemotronHBlockType::Moe)
            .count(),
        23
    );
}

/// The 3.5 Lightning list-form schema and checkpoint-declared MTP schedule parse.
#[test]
fn nemotron_3_5_lightning_list_config_is_compatible() {
    let layers = OFFICIAL_30B_PATTERN
        .chars()
        .map(|character| match character {
            'M' => "mamba",
            'E' => "moe",
            '*' => "attention",
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    let config: NemotronHConfig = serde_json::from_value(serde_json::json!({
        "model_type": "nemotron_h",
        "vocab_size": 131072,
        "hidden_size": 2688,
        "layers_block_type": layers,
        "num_hidden_layers": 52,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "max_position_embeddings": 262144,
        "intermediate_size": 1856,
        "ssm_state_size": 128,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "n_groups": 8,
        "conv_kernel": 4,
        "chunk_size": 128,
        "n_routed_experts": 128,
        "n_shared_experts": 1,
        "moe_intermediate_size": 1856,
        "moe_shared_expert_intermediate_size": 3712,
        "num_experts_per_tok": 6,
        "routed_scaling_factor": 2.5,
        "n_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
        "norm_eps": 0.00001
    }))
    .unwrap();
    config.validate().unwrap();
    assert_eq!(config.block_count(), 52);
    assert_eq!(config.num_nextn_predict_layers, 1);
    assert_eq!(
        config.mtp_layers_block_type,
        [NemotronHBlockType::FullAttention, NemotronHBlockType::Moe]
    );
}
