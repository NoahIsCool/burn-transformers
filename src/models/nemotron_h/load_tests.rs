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
use crate::models::nemotron_h::{NemotronHBlockType, NemotronHCheckpointContract, NemotronHConfig};

type TestBackend = Flex<f32, i32>;

/// Reduced hybrid topology retaining every sink-dispatch branch and MTP.
fn tiny_config(tied: bool) -> NemotronHConfig {
    NemotronHConfig {
        vocab_size: 16,
        hidden_size: 8,
        layers_block_type: vec![
            NemotronHBlockType::LinearAttention,
            NemotronHBlockType::FullAttention,
            NemotronHBlockType::Mlp,
            NemotronHBlockType::Moe,
        ],
        num_hidden_layers: Some(4),
        tie_word_embeddings: tied,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 16,
        intermediate_size: 12,
        ssm_state_size: 3,
        mamba_num_heads: 4,
        mamba_head_dim: 2,
        n_groups: 2,
        conv_kernel: 3,
        chunk_size: 2,
        n_routed_experts: 4,
        n_shared_experts: 1,
        moe_intermediate_size: 6,
        moe_shared_expert_intermediate_size: 8,
        num_experts_per_tok: 2,
        n_group: 2,
        topk_group: 1,
        num_nextn_predict_layers: 1,
        ..Default::default()
    }
}

/// Writes one complete, zero-valued F32 SafeTensors fixture.
fn write_checkpoint(
    root: &Path,
    contract: &NemotronHCheckpointContract,
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
            TensorOperation::Alias { .. }
            | TensorOperation::Split { .. }
            | TensorOperation::Concatenate { .. } => None,
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

/// Every base and MTP contract target reaches the exact ranked Burn field.
#[test]
fn complete_contract_installs_and_runs() {
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    let device = Default::default();
    let mut model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
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

/// Pass two opens one local shard and applies tied and untied contracts.
#[test]
fn local_loader_streams_tied_and_untied_checkpoints() {
    let device = Default::default();
    for tied in [false, true] {
        let config = tiny_config(tied);
        let contract = config
            .checkpoint_contract(NemotronHCheckpointOptions::default())
            .unwrap();
        let temp = TempDir::new().unwrap();
        write_checkpoint(temp.path(), &contract, |_| {});
        let mut model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
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

/// Metadata mismatches fail before the first lazy parameter is initialized.
#[test]
fn strict_mismatches_fail_before_writes() {
    let device = Default::default();
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();

    let missing = TempDir::new().unwrap();
    write_checkpoint(missing.path(), &contract, |tensors| {
        tensors.retain(|(name, _)| name != "backbone.layers.0.mixer.A_log");
    });
    let mut model = NemotronHForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(missing.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        NemotronHCheckpointLoadError::Checkpoint(CheckpointError::MissingMappingSource { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let unexpected = TempDir::new().unwrap();
    write_checkpoint(unexpected.path(), &contract, |tensors| {
        tensors.push(("unexpected.weight".into(), vec![1]));
    });
    let mut model = NemotronHForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(unexpected.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        NemotronHCheckpointLoadError::Checkpoint(CheckpointError::StrictValidation {
            unexpected: 1,
            ..
        })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let wrong_shape = TempDir::new().unwrap();
    write_checkpoint(wrong_shape.path(), &contract, |tensors| {
        tensors
            .iter_mut()
            .find(|(name, _)| name == "backbone.embeddings.weight")
            .unwrap()
            .1 = vec![15, 8];
    });
    let mut model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let error = model
        .load_checkpoint(wrong_shape.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        NemotronHCheckpointLoadError::Checkpoint(CheckpointError::ShapeMismatch { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());
}

/// Malformed ranks, indexes, aliases, and absent mixers return typed errors.
#[test]
fn sink_rejects_invalid_targets_and_payloads() {
    let device = Default::default();
    let mut model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    assert!(matches!(
        model.write_tensor(
            "model.embed_tokens.weight",
            TensorData::zeros::<i32, _>([16, 8])
        ),
        Err(NemotronHLoadError::Parameter { .. })
    ));
    assert!(matches!(
        model.write_tensor("model.layers.bad.weight", TensorData::from([0.0_f32])),
        Err(NemotronHLoadError::InvalidIndex { .. })
    ));
    assert!(matches!(
        model.write_tensor(
            "model.layers.3.moe.routed.9.up_proj.weight",
            TensorData::zeros::<f32, _>([6, 8])
        ),
        Err(NemotronHLoadError::InvalidIndex { .. })
    ));
    assert!(matches!(
        model.write_tensor(
            "model.layers.0.attention.q_proj.weight",
            TensorData::zeros::<f32, _>([8, 8])
        ),
        Err(NemotronHLoadError::MissingComponent { .. })
    ));
    assert!(matches!(
        model.alias_tensor("other.weight", "model.embed_tokens.weight"),
        Err(NemotronHLoadError::InvalidAlias { .. })
    ));
}
