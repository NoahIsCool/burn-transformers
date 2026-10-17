//! The tensor contract: every target with its shape, dense and MXFP4 expert
//! sources, optional biases, and the tied head.

use super::super::config::tests::GPT_OSS_20B;
use super::*;
use crate::checkpoint::TensorOperation;
use crate::models::config::ModelConfig;

/// Returns the released config without its quantization directive (a dense export).
fn dense_20b() -> GptOssConfig {
    GptOssConfig {
        quantization_config: None,
        ..GptOssConfig::from_json_str(GPT_OSS_20B).unwrap()
    }
}

/// Returns the expected shape of `name`.
fn shape_of(contract: &GptOssCheckpointContract, name: &str) -> Vec<u64> {
    contract
        .expected_tensors()
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("`{name}` is not expected"))
        .shape
        .clone()
}

/// The dense contract lists every tensor once with its reference shape.
#[test]
fn dense_contract_lists_every_tensor() {
    let contract = dense_20b()
        .checkpoint_contract(GptOssCheckpointOptions::default())
        .unwrap();
    // Per layer: 4 weights + 4 biases + sinks + router (2) + experts (4) + 2 norms.
    assert_eq!(contract.expected_tensors().len(), 1 + 24 * 17 + 2);
    for (name, shape) in [
        ("model.embed_tokens.weight", vec![201_088, 2880]),
        ("model.layers.0.self_attn.q_proj.weight", vec![4096, 2880]),
        ("model.layers.0.self_attn.k_proj.weight", vec![512, 2880]),
        ("model.layers.0.self_attn.v_proj.bias", vec![512]),
        ("model.layers.0.self_attn.o_proj.weight", vec![2880, 4096]),
        ("model.layers.0.self_attn.o_proj.bias", vec![2880]),
        ("model.layers.0.self_attn.sinks", vec![64]),
        ("model.layers.0.mlp.router.weight", vec![32, 2880]),
        ("model.layers.0.mlp.router.bias", vec![32]),
        (
            "model.layers.0.mlp.experts.gate_up_proj",
            vec![32, 2880, 5760],
        ),
        (
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            vec![32, 5760],
        ),
        ("model.layers.0.mlp.experts.down_proj", vec![32, 2880, 2880]),
        ("model.layers.0.mlp.experts.down_proj_bias", vec![32, 2880]),
        ("model.layers.23.input_layernorm.weight", vec![2880]),
        (
            "model.layers.23.post_attention_layernorm.weight",
            vec![2880],
        ),
        ("model.norm.weight", vec![2880]),
        ("lm_head.weight", vec![201_088, 2880]),
    ] {
        assert_eq!(shape_of(&contract, name), shape, "{name}");
    }
    // Every dense target maps from the checkpoint tensor of the same name.
    for operation in contract.mappings().operations() {
        let TensorOperation::Map { source, target, .. } = operation else {
            panic!("dense contracts only rename: {operation:?}");
        };
        assert_eq!(source, target);
    }
}

/// MXFP4 checkpoints decode both stacked expert projections from blocks and scales.
#[test]
fn mxfp4_contract_decodes_expert_projections() {
    let config = GptOssConfig::from_json_str(GPT_OSS_20B).unwrap();
    let contract = config
        .checkpoint_contract(GptOssCheckpointOptions::default())
        .unwrap();
    let decodings: Vec<_> = contract
        .mappings()
        .operations()
        .iter()
        .filter_map(|operation| match operation {
            TensorOperation::DequantizeMxfp4 {
                blocks,
                scales,
                target,
            } => Some((blocks.clone(), scales.clone(), target.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(decodings.len(), 2 * 24);
    assert!(decodings.contains(&(
        "model.layers.7.mlp.experts.gate_up_proj_blocks".into(),
        "model.layers.7.mlp.experts.gate_up_proj_scales".into(),
        "model.layers.7.mlp.experts.gate_up_proj".into(),
    )));
    assert!(decodings.contains(&(
        "model.layers.7.mlp.experts.down_proj_blocks".into(),
        "model.layers.7.mlp.experts.down_proj_scales".into(),
        "model.layers.7.mlp.experts.down_proj".into(),
    )));
    // The decoded targets keep the dense shapes; biases stay dense.
    assert_eq!(
        shape_of(&contract, "model.layers.7.mlp.experts.gate_up_proj"),
        [32, 2880, 5760]
    );
    assert_eq!(
        contract.expected_tensors().len(),
        dense_20b()
            .checkpoint_contract(GptOssCheckpointOptions::default())
            .unwrap()
            .expected_tensors()
            .len()
    );
}

/// Bias-free attention drops the four projection biases; sinks remain.
#[test]
fn bias_free_attention_drops_projection_biases() {
    let config = GptOssConfig {
        attention_bias: false,
        ..dense_20b()
    };
    let contract = config
        .checkpoint_contract(GptOssCheckpointOptions::default())
        .unwrap();
    assert_eq!(contract.expected_tensors().len(), 1 + 24 * 13 + 2);
    assert!(
        contract
            .expected_tensors()
            .iter()
            .all(|tensor| !tensor.name.contains("_proj.bias"))
    );
    assert_eq!(shape_of(&contract, "model.layers.0.self_attn.sinks"), [64]);
}

/// A tied head is an alias of the embedding rather than a checkpoint tensor.
#[test]
fn tied_heads_alias_the_embedding() {
    let contract = GptOssConfig {
        tie_word_embeddings: true,
        ..dense_20b()
    }
    .checkpoint_contract(GptOssCheckpointOptions::default())
    .unwrap();
    assert!(
        contract
            .mappings()
            .operations()
            .contains(&TensorOperation::Alias {
                source_target: "model.embed_tokens.weight".into(),
                target: "lm_head.weight".into(),
            })
    );
}

/// An invalid configuration has no contract.
#[test]
fn invalid_configurations_have_no_contract() {
    let error = GptOssConfig {
        hidden_size: 0,
        ..dense_20b()
    }
    .checkpoint_contract(GptOssCheckpointOptions::default())
    .unwrap_err();
    assert_eq!(error.field, "hidden_size");
}
