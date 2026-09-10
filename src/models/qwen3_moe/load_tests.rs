use std::{fs::File, io::Write, path::Path};

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use serde_json::{Map, Value, json};
use tempfile::TempDir;

use crate::{
    CausalLmInput, ForwardOptions,
    checkpoint::{CheckpointError, TensorOperation},
};

use super::*;
use crate::models::qwen3_moe::Qwen3MoeConfig;

type TestBackend = Flex<f32, i32>;

/// Minimal routed topology used by sink and streaming tests.
fn tiny_config(tied: bool) -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        vocab_size: 16,
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 2,
        max_position_embeddings: 16,
        max_window_layers: 1,
        decoder_sparse_step: 1,
        moe_intermediate_size: 3,
        num_experts: 3,
        num_experts_per_tok: 2,
        norm_topk_prob: true,
        tie_word_embeddings: tied,
        ..Default::default()
    }
}

/// Writes a complete zero-valued F32 SafeTensors artifact.
fn write_checkpoint(
    root: &Path,
    contract: &super::super::Qwen3MoeCheckpointContract,
    mutate: impl FnOnce(&mut Vec<(String, Vec<u64>)>),
) {
    let mut tensors = contract
        .mappings()
        .operations()
        .iter()
        .filter_map(|operation| match operation {
            TensorOperation::Map { source, target, .. } => {
                let shape = contract
                    .expected_tensors()
                    .iter()
                    .find(|expected| &expected.name == target)
                    .unwrap()
                    .shape
                    .clone();
                Some((source.clone(), shape))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    mutate(&mut tensors);

    let mut offset = 0_u64;
    let mut header = Map::new();
    for (name, shape) in tensors {
        let elements = shape.iter().copied().product::<u64>();
        let end = offset + elements * 4;
        header.insert(
            name,
            json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&vec![0; offset as usize]).unwrap();
}

/// Every official sparse target dispatches to its exact Burn field.
#[test]
fn complete_contract_installs_and_runs() {
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .unwrap();
    let device = Default::default();
    let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(config, &device).unwrap();
    for expected in contract.expected_tensors() {
        let shape = expected
            .shape
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        model
            .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert!(
        output
            .logits
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0)
    );
}

/// Streamed local loading opens one shard and applies every parameter once.
#[test]
fn local_loader_streams_tied_and_untied_checkpoints() {
    let device = Default::default();
    for tied in [false, true] {
        let config = tiny_config(tied);
        let contract = config
            .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
            .unwrap();
        let temp = TempDir::new().unwrap();
        write_checkpoint(temp.path(), &contract, |_| {});
        let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(config, &device).unwrap();
        let outcome = model
            .load_checkpoint(temp.path(), &LoadOptions::default())
            .unwrap();
        assert_eq!(
            outcome.report.applied.len(),
            contract.expected_tensors().len()
        );
        assert_eq!(outcome.stats.source.shard_opens, 1);
        assert_eq!(outcome.stats.source.max_open_shards, 1);
        assert_eq!(outcome.stats.aliases_applied, usize::from(tied));
    }
}

/// Planning rejects malformed inventories before initializing a parameter.
#[test]
fn missing_shape_and_unexpected_fail_before_writes() {
    let device = Default::default();
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .unwrap();

    let missing = TempDir::new().unwrap();
    write_checkpoint(missing.path(), &contract, |tensors| {
        tensors.retain(|(name, _)| name != "model.layers.0.mlp.gate.weight");
    });
    let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(missing.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        Qwen3MoeCheckpointLoadError::Checkpoint(CheckpointError::MissingMappingSource { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let unexpected = TempDir::new().unwrap();
    write_checkpoint(unexpected.path(), &contract, |tensors| {
        tensors.push(("unexpected.weight".into(), vec![1]));
    });
    let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(unexpected.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        Qwen3MoeCheckpointLoadError::Checkpoint(CheckpointError::StrictValidation {
            unexpected: 1,
            ..
        })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let wrong = TempDir::new().unwrap();
    write_checkpoint(wrong.path(), &contract, |tensors| {
        tensors
            .iter_mut()
            .find(|(name, _)| name == "model.embed_tokens.weight")
            .unwrap()
            .1 = vec![15, 4];
    });
    let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let error = model
        .load_checkpoint(wrong.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        Qwen3MoeCheckpointLoadError::Checkpoint(CheckpointError::ShapeMismatch { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());
}

/// Wrong dtype, malformed names, expert bounds, and topology mismatch are typed.
#[test]
fn sink_rejects_invalid_targets_and_payloads() {
    let device = Default::default();
    let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    assert!(matches!(
        model.write_tensor(
            "model.embed_tokens.weight",
            TensorData::zeros::<i32, _>([16, 4])
        ),
        Err(Qwen3MoeLoadError::Parameter { .. })
    ));
    assert!(matches!(
        model.write_tensor("model.layers.bad.weight", TensorData::from([0.0_f32])),
        Err(Qwen3MoeLoadError::InvalidIndex { .. })
    ));
    assert!(matches!(
        model.write_tensor(
            "model.layers.0.sparse_mlp.routed.9.gate_proj.weight",
            TensorData::zeros::<f32, _>([3, 4])
        ),
        Err(Qwen3MoeLoadError::InvalidIndex { .. })
    ));
    model.model.layers[0].sparse_mlp = None;
    assert!(matches!(
        model.write_tensor(
            "model.layers.0.sparse_mlp.router.weight",
            TensorData::zeros::<f32, _>([3, 4])
        ),
        Err(Qwen3MoeLoadError::MissingComponent { .. })
    ));
}
