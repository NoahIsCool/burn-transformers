//! Checkpoint contracts: every tensor name and shape per variant, renames,
//! optional biases and Q/K norms, the tied-head alias, and overflow.

use super::*;
use crate::checkpoint::TensorOperation;
use crate::models::llama::LlamaVariant;

/// A small configuration of `variant`: `d = 8`, `f = 12`, two layers, `H = 4`, `K = 2`, `h = 2`.
fn small(variant: LlamaVariant) -> LlamaFamilyConfig {
    LlamaFamilyConfig {
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 2,
        max_position_embeddings: 32,
        pad_token_id: None,
        bos_token_id: None,
        eos_token_id: None,
        ..LlamaFamilyConfig::new(variant)
    }
}

/// Returns `(target, shape)` pairs of a contract.
fn shapes(contract: &LlamaCheckpointContract) -> Vec<(String, Vec<u64>)> {
    contract
        .expected_tensors()
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.shape.clone()))
        .collect()
}

/// Returns `source -> target` renames and aliases of a contract.
fn sources(contract: &LlamaCheckpointContract) -> Vec<(String, String)> {
    contract
        .mappings()
        .operations()
        .iter()
        .map(|operation| match operation {
            TensorOperation::Map { source, target, .. } => (source.clone(), target.clone()),
            TensorOperation::Alias {
                source_target,
                target,
            } => (format!("alias:{source_target}"), target.clone()),
            other => panic!("unexpected operation {other:?}"),
        })
        .collect()
}

/// A plain Llama layer has seven projections and two norms; the head is independent.
#[test]
fn llama_contract_lists_every_tensor() {
    let contract = small(LlamaVariant::Llama)
        .checkpoint_contract(LlamaCheckpointOptions::default())
        .unwrap();
    let expected = shapes(&contract);
    assert_eq!(expected.len(), 1 + 2 * 9 + 2);
    let layer: Vec<(String, Vec<u64>)> = [
        ("model.layers.0.self_attn.q_proj.weight", vec![8, 8]),
        ("model.layers.0.self_attn.k_proj.weight", vec![4, 8]),
        ("model.layers.0.self_attn.v_proj.weight", vec![4, 8]),
        ("model.layers.0.self_attn.o_proj.weight", vec![8, 8]),
        ("model.layers.0.mlp.gate_proj.weight", vec![12, 8]),
        ("model.layers.0.mlp.up_proj.weight", vec![12, 8]),
        ("model.layers.0.mlp.down_proj.weight", vec![8, 12]),
        ("model.layers.0.attention_norm.weight", vec![8]),
        ("model.layers.0.mlp_norm.weight", vec![8]),
    ]
    .into_iter()
    .map(|(name, shape)| (name.to_owned(), shape))
    .collect();
    assert_eq!(expected[1..10], layer);
    assert_eq!(
        expected[0],
        ("model.embed_tokens.weight".into(), vec![16, 8])
    );
    assert_eq!(expected[19], ("model.norm.weight".into(), vec![8]));
    assert_eq!(expected[20], ("lm_head.weight".into(), vec![16, 8]));
    let renames = sources(&contract);
    assert!(renames.contains(&(
        "model.layers.1.input_layernorm.weight".into(),
        "model.layers.1.attention_norm.weight".into()
    )));
    assert!(renames.contains(&(
        "model.layers.1.post_attention_layernorm.weight".into(),
        "model.layers.1.mlp_norm.weight".into()
    )));
    assert!(renames.contains(&("lm_head.weight".into(), "lm_head.weight".into())));
}

/// Biases appear exactly where the variant's projections carry them.
#[test]
fn bias_tensors_follow_the_projection_flags() {
    let names = |config: LlamaFamilyConfig| -> Vec<String> {
        shapes(
            &config
                .checkpoint_contract(LlamaCheckpointOptions::default())
                .unwrap(),
        )
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| name.ends_with(".bias"))
        .collect()
    };
    let qwen2 = names(small(LlamaVariant::Qwen2));
    assert_eq!(qwen2.len(), 2 * 3);
    assert!(qwen2.iter().all(|name| !name.contains("o_proj")));
    let llama = names(LlamaFamilyConfig {
        attention_bias: Some(true),
        mlp_bias: Some(true),
        ..small(LlamaVariant::Llama)
    });
    assert_eq!(llama.len(), 2 * 7);
    assert!(names(small(LlamaVariant::Mistral)).is_empty());
}

/// Per-head and full-width Q/K norms have their own widths and renames.
#[test]
fn query_key_norms_have_scope_dependent_shapes() {
    let qwen3 = shapes(
        &small(LlamaVariant::Qwen3)
            .checkpoint_contract(LlamaCheckpointOptions::default())
            .unwrap(),
    );
    assert!(qwen3.contains(&("model.layers.0.self_attn.query_norm.weight".into(), vec![2])));
    assert!(qwen3.contains(&("model.layers.0.self_attn.key_norm.weight".into(), vec![2])));
    let olmo2 = small(LlamaVariant::Olmo2)
        .checkpoint_contract(LlamaCheckpointOptions::default())
        .unwrap();
    let olmo_shapes = shapes(&olmo2);
    assert!(olmo_shapes.contains(&("model.layers.0.self_attn.query_norm.weight".into(), vec![8])));
    assert!(olmo_shapes.contains(&("model.layers.0.self_attn.key_norm.weight".into(), vec![4])));
    let renames = sources(&olmo2);
    for (source, target) in [
        ("self_attn.q_norm.weight", "self_attn.query_norm.weight"),
        ("post_attention_layernorm.weight", "attention_norm.weight"),
        ("post_feedforward_layernorm.weight", "mlp_norm.weight"),
    ] {
        assert!(renames.contains(&(
            format!("model.layers.0.{source}"),
            format!("model.layers.0.{target}")
        )));
    }
    assert!(
        !renames
            .iter()
            .any(|(source, _)| source.contains("input_layernorm"))
    );
}

/// A tied head is an alias of the embedding, never a stored tensor.
#[test]
fn tied_heads_alias_the_embedding() {
    let contract = small(LlamaVariant::SmolLm3)
        .checkpoint_contract(LlamaCheckpointOptions::default())
        .unwrap();
    let renames = sources(&contract);
    assert!(renames.contains(&(
        "alias:model.embed_tokens.weight".into(),
        "lm_head.weight".into()
    )));
    assert!(!renames.contains(&("lm_head.weight".into(), "lm_head.weight".into())));
}

/// Invalid configurations fail contract construction with their validation error.
#[test]
fn invalid_configurations_have_no_contract() {
    let error = LlamaFamilyConfig {
        num_key_value_heads: 3,
        ..small(LlamaVariant::Llama)
    }
    .checkpoint_contract(LlamaCheckpointOptions::default())
    .unwrap_err();
    assert_eq!(error.field, "num_attention_heads");
}
