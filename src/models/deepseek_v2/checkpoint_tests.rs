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
