//! DeepSeek-V2 streamed parameter installation and malformed-checkpoint rejection.

use std::{fs::File, io::Write, path::Path};

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use serde_json::{Map, Value, json};
use tempfile::TempDir;

use super::*;
use crate::{
    CausalLmInput, ForwardOptions,
    checkpoint::{CheckpointError, TensorOperation, TensorSink},
    models::deepseek_v2::model::tests_support::tiny_config,
};

/// CPU backend with F32 tensor values and I32 token indices for deterministic fixtures.
type TestBackend = Flex<f32, i32>;

/// Writes a complete zero-valued F32 SafeTensors artifact.
fn write_checkpoint(
    root: &Path,
    contract: &super::super::DeepseekV2CheckpointContract,
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

/// Invalid target syntax is rejected before model indexing.
#[test]
fn target_parser_rejects_unknown_and_malformed_indices() {
    assert!(matches!(
        parse_layer_target("not.a.layer"),
        Err(DeepseekV2LoadError::UnknownTarget { .. })
    ));
    assert!(matches!(
        parse_layer_target("model.layers.x.self_attn.q_proj.weight"),
        Err(DeepseekV2LoadError::InvalidIndex { .. })
    ));
}

/// A strict target cannot install a tensor with the wrong checkpoint shape.
#[test]
fn sink_rejects_wrong_parameter_shape() {
    let device = Default::default();
    let mut config = super::super::model::tests_support::tiny_config();
    config.tie_word_embeddings = false;
    let mut model = DeepseekV2ForCausalLm::<Flex>::init(config, &device).unwrap();
    let error = TensorSink::write_tensor(
        &mut model,
        "model.embed_tokens.weight",
        TensorData::zeros::<f32, _>([1, 1]),
    )
    .unwrap_err();
    assert!(matches!(error, DeepseekV2LoadError::Parameter { .. }));
}

/// Every dense, MLA, routed, and shared target installs and executes.
#[test]
fn complete_contract_installs_and_runs() {
    let config = tiny_config();
    let contract = config
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .unwrap();
    let device = Default::default();
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
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

/// Streamed local loading opens one shard and applies every owner once.
#[test]
fn local_loader_streams_tied_and_untied_checkpoints() {
    let device = Default::default();
    for tied in [false, true] {
        let mut config = tiny_config();
        config.tie_word_embeddings = tied;
        let contract = config
            .checkpoint_contract(DeepseekV2CheckpointOptions::default())
            .unwrap();
        let temp = TempDir::new().unwrap();
        write_checkpoint(temp.path(), &contract, |_| {});
        let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
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

/// Missing, unexpected, and wrong-shaped inventory fails before first write.
#[test]
fn strict_planning_rejects_inventory_mismatches_before_loading() {
    let device = Default::default();
    let config = tiny_config();
    let contract = config
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .unwrap();

    let missing = TempDir::new().unwrap();
    write_checkpoint(missing.path(), &contract, |tensors| {
        tensors.retain(|(name, _)| name != "model.layers.1.mlp.gate.weight");
    });
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(missing.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        DeepseekV2CheckpointLoadError::Checkpoint(CheckpointError::MissingMappingSource { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let unexpected = TempDir::new().unwrap();
    write_checkpoint(unexpected.path(), &contract, |tensors| {
        tensors.push(("unexpected.weight".into(), vec![1]));
    });
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(unexpected.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        DeepseekV2CheckpointLoadError::Checkpoint(CheckpointError::StrictValidation {
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
            .1 = vec![31, 8];
    });
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let error = model
        .load_checkpoint(wrong.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        DeepseekV2CheckpointLoadError::Checkpoint(CheckpointError::ShapeMismatch { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());
}

/// Query-LoRA targets and expert bounds dispatch to typed optional fields.
#[test]
fn query_lora_and_invalid_expert_targets_are_typed() {
    let device = Default::default();
    let mut config = tiny_config();
    config.q_lora_rank = Some(4);
    let contract = config.checkpoint_contract(Default::default()).unwrap();
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    for expected in contract.expected_tensors() {
        let shape = expected
            .shape
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>();
        model
            .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    assert!(matches!(
        model.write_tensor(
            "model.layers.1.sparse_mlp.routed.routed.9.gate_proj.weight",
            TensorData::zeros::<f32, _>([6, 8]),
        ),
        Err(DeepseekV2LoadError::InvalidIndex { .. })
    ));
    assert!(matches!(
        model.write_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            TensorData::zeros::<f32, _>([10, 8]),
        ),
        Err(DeepseekV2LoadError::MissingComponent { .. })
    ));
}
