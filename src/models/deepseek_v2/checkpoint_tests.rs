//! DeepSeek-V2 checkpoint inventory, expert mappings, and strict shape-validation regressions.

use crate::checkpoint::TensorOperation;

use super::*;

/// The pinned Lite topology has exactly the official 5,291 tensor entries.
#[test]
fn pinned_lite_inventory_count_and_shapes_are_exact() {
    let config = DeepseekV2Config::default();
    let contract = config
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .unwrap();
    assert_eq!(contract.expected_tensors().len(), 5_291);
    let shared = contract
        .expected_tensors()
        .iter()
        .find(|tensor| tensor.name == "model.layers.1.sparse_mlp.shared_experts.gate_proj.weight")
        .unwrap();
    assert_eq!(shared.shape, vec![2816, 2048]);
    let kv = contract
        .expected_tensors()
        .iter()
        .find(|tensor| tensor.name == "model.layers.0.self_attn.kv_b_proj.weight")
        .unwrap();
    assert_eq!(kv.shape, vec![4096, 512]);
}

/// Query-LoRA adds its three checkpoint components while direct Q uses one.
#[test]
fn optional_query_lora_changes_only_its_projection_contract() {
    let mut direct = DeepseekV2Config {
        num_hidden_layers: 2,
        n_routed_experts: 4,
        num_experts_per_tok: 2,
        first_k_dense_replace: 1,
        max_position_embeddings: 32,
        rope_scaling: None,
        ..Default::default()
    };
    let direct_count = direct
        .checkpoint_contract(Default::default())
        .unwrap()
        .expected_tensors()
        .len();
    direct.q_lora_rank = Some(6);
    let lora = direct.checkpoint_contract(Default::default()).unwrap();
    assert_eq!(lora.expected_tensors().len(), direct_count + 4);
    assert!(
        lora.mappings()
            .operations()
            .iter()
            .any(|operation| matches!(
                operation,
                TensorOperation::Map { source, target, .. }
                    if source.ends_with("q_a_layernorm.weight") && target == source
            ))
    );
}

/// `noaux_tc` requires exactly one F32-shaped selection bias per sparse layer.
#[test]
fn noaux_routing_maps_one_correction_bias_per_sparse_layer() {
    let mut config = DeepseekV2Config::default();
    let baseline = config
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .unwrap()
        .expected_tensors()
        .len();
    config.scoring_func = super::super::DeepseekScoringFunction::Sigmoid;
    config.topk_method = DeepseekTopKMethod::NoAuxTc;
    let contract = config
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .unwrap();
    let sparse_layers = (0..config.num_hidden_layers)
        .filter(|&layer| config.is_moe_layer(layer))
        .count();
    assert_eq!(contract.expected_tensors().len(), baseline + sparse_layers);
    let bias = contract
        .expected_tensors()
        .iter()
        .find(|tensor| tensor.name == "model.layers.1.sparse_mlp.routed.correction_bias")
        .unwrap();
    assert_eq!(bias.shape, vec![64]);
    assert!(
        contract
            .mappings()
            .operations()
            .iter()
            .any(|operation| matches!(
                operation,
                TensorOperation::Map { source, target, .. }
                    if source == "model.layers.1.mlp.gate.e_score_correction_bias"
                        && target == "model.layers.1.sparse_mlp.routed.correction_bias"
            ))
    );
    assert!(
        !contract
            .expected_tensors()
            .iter()
            .any(|tensor| tensor.name == "model.layers.0.sparse_mlp.routed.correction_bias")
    );
}

/// Biased attention adds the query-down, joint KV-A, and output biases per layer.
///
/// With a low-rank query, each layer gains `q_a_proj.bias` `[q_lora_rank]`,
/// `kv_a_proj_with_mqa.bias` `[kv_lora_rank + qk_rope_head_dim]`, and
/// `o_proj.bias` `[hidden_size]`; a direct query projection has no `q_a_proj`.
#[test]
fn attention_bias_adds_one_bias_per_biased_projection() {
    let unbiased = DeepseekV2Config {
        num_hidden_layers: 2,
        n_routed_experts: 4,
        num_experts_per_tok: 2,
        first_k_dense_replace: 1,
        max_position_embeddings: 32,
        rope_scaling: None,
        q_lora_rank: Some(6),
        ..Default::default()
    };
    let biased = DeepseekV2Config {
        attention_bias: true,
        ..unbiased.clone()
    };
    let count = |config: &DeepseekV2Config| {
        config
            .checkpoint_contract(DeepseekV2CheckpointOptions::default())
            .unwrap()
            .expected_tensors()
            .len()
    };
    assert_eq!(count(&biased), count(&unbiased) + 3 * 2);
    let contract = biased.checkpoint_contract(Default::default()).unwrap();
    let shape = |name: &str| {
        contract
            .expected_tensors()
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("{name}"))
            .shape
            .clone()
    };
    assert_eq!(shape("model.layers.1.self_attn.q_a_proj.bias"), vec![6]);
    assert_eq!(
        shape("model.layers.1.self_attn.kv_a_proj_with_mqa.bias"),
        vec![512 + 64]
    );
    assert_eq!(shape("model.layers.0.self_attn.o_proj.bias"), vec![2048]);

    let direct = DeepseekV2Config {
        q_lora_rank: None,
        ..biased
    };
    // Per layer: the three low-rank query tensors become one direct `q_proj`,
    // and only the joint KV-A and output biases remain.
    let layers = 2;
    assert_eq!(
        count(&direct),
        count(&unbiased) - 3 * layers + layers + 2 * layers
    );
}

/// A validated topology whose fused query width overflows is a contract error.
///
/// Validation bounds each extent and the per-head sum, but not the products
/// `heads * width`; the checked dimension conversion names the first one.
#[test]
fn overflowing_fused_widths_are_contract_errors() {
    let config = DeepseekV2Config {
        num_attention_heads: 1 << (usize::BITS - 4),
        qk_nope_head_dim: 64,
        ..Default::default()
    };
    config
        .validate()
        .expect("each extent is individually valid");
    let error = config
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .expect_err("the fused query width does not fit usize");
    assert_eq!(error.architecture, "deepseek_v2");
    assert_eq!(error.field, "num_attention_heads * qk_head_dim");
    assert_eq!(error.reason, "dimension product overflows usize");
}

/// The validation-time count equals the contract's length for every optional branch.
///
/// Validation bounds the inventory with [`DeepseekV2Config::checkpoint_tensor_count`]
/// before building it, so the formula must track the builder exactly.
#[test]
fn tensor_count_matches_the_contract_for_every_branch() {
    let base = DeepseekV2Config {
        num_hidden_layers: 3,
        n_routed_experts: 4,
        num_experts_per_tok: 2,
        first_k_dense_replace: 1,
        max_position_embeddings: 32,
        rope_scaling: None,
        ..Default::default()
    };
    let mut noaux = base.clone();
    noaux.scoring_func = super::super::DeepseekScoringFunction::Sigmoid;
    noaux.topk_method = DeepseekTopKMethod::NoAuxTc;
    let configs = [
        base.clone(),
        DeepseekV2Config {
            q_lora_rank: Some(6),
            ..base.clone()
        },
        DeepseekV2Config {
            attention_bias: true,
            q_lora_rank: Some(6),
            ..base.clone()
        },
        DeepseekV2Config {
            attention_bias: true,
            moe_layer_freq: 2,
            ..base.clone()
        },
        noaux,
        DeepseekV2Config::default(),
    ];
    for config in configs {
        let Ok(contract) = config.checkpoint_contract(Default::default()) else {
            panic!("{config:?}");
        };
        assert_eq!(
            config.checkpoint_tensor_count(),
            Some(contract.expected_tensors().len())
        );
    }
}

/// Layers times experts beyond the artifact limit fail validation before any contract exists.
///
/// Regression for CFG-COUNT-1: `n_routed_experts = 2^40` and
/// `num_hidden_layers = 2^40` both passed validation.
#[test]
fn expert_inventories_beyond_the_artifact_limit_are_rejected_before_listing() {
    use crate::models::config::{MAX_CHECKPOINT_TENSORS, MAX_EXPERTS, MAX_LAYERS};

    let config = DeepseekV2Config {
        n_routed_experts: MAX_EXPERTS,
        n_group: 1,
        topk_group: 1,
        ..Default::default()
    };
    assert!(config.checkpoint_tensor_count().unwrap() > MAX_CHECKPOINT_TENSORS);
    let error = config.checkpoint_contract(Default::default()).unwrap_err();
    assert_eq!(error.field, "n_routed_experts");
    for (layers, experts, field) in [
        (1 << 40, 64, "num_hidden_layers"),
        (MAX_LAYERS + 1, 64, "num_hidden_layers"),
        (27, 1 << 40, "n_routed_experts"),
        (27, usize::MAX, "n_routed_experts"),
    ] {
        let error = DeepseekV2Config {
            num_hidden_layers: layers,
            n_routed_experts: experts,
            n_group: 1,
            topk_group: 1,
            ..Default::default()
        }
        .validate()
        .unwrap_err();
        assert_eq!(error.field, field, "{layers} {experts}");
    }
}
