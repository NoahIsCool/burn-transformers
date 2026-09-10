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
use crate::models::muse_glimmer::{MuseGlimmerCheckpointSource, MuseGlimmerTextConfig};

type TestBackend = Flex<f32, i32>;

/// Builds a one-layer miniature suitable for sink and I/O tests.
fn tiny_config(tied: bool) -> MuseGlimmerTextConfig {
    MuseGlimmerTextConfig {
        vocab_size: 16,
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 2,
        max_position_embeddings: 16,
        sliding_window: Some(4),
        layer_types: vec![super::super::MuseGlimmerLayerType::FullAttention],
        layer_rope_theta: vec![0.0],
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: tied,
        ..Default::default()
    }
}

/// Writes a complete zero-valued F32 SafeTensors file from the mapping plan.
fn write_checkpoint(
    root: &Path,
    contract: &super::super::MuseGlimmerCheckpointContract,
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

/// Every family target installs into its exact field and the graph runs.
#[test]
fn complete_contract_installs_and_runs() {
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .unwrap();
    let device = Default::default();
    let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config, &device).unwrap();
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

/// Local pass two streams all text tensors while preserving strict accounting.
#[test]
fn local_loader_streams_tied_and_untied_text_checkpoints() {
    let device = Default::default();
    for tied in [false, true] {
        let config = tiny_config(tied);
        let family_options = MuseGlimmerCheckpointOptions {
            source: MuseGlimmerCheckpointSource::TextOnly,
        };
        let contract = config.checkpoint_contract(family_options).unwrap();
        let temp = TempDir::new().unwrap();
        write_checkpoint(temp.path(), &contract, |_| {});
        let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config, &device).unwrap();
        let outcome = model
            .load_checkpoint_with_family_options(
                temp.path(),
                &LoadOptions::default(),
                family_options,
            )
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

/// Missing, unknown, and wrong-shaped tensors fail before any sink write.
#[test]
fn malformed_inventory_fails_before_parameter_initialization() {
    let device = Default::default();
    let config = tiny_config(false);
    let family_options = MuseGlimmerCheckpointOptions {
        source: MuseGlimmerCheckpointSource::TextOnly,
    };
    let contract = config.checkpoint_contract(family_options).unwrap();

    let missing = TempDir::new().unwrap();
    write_checkpoint(missing.path(), &contract, |tensors| {
        tensors.retain(|(name, _)| !name.ends_with("self_attn.gate_proj.weight"));
    });
    let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint_with_family_options(
            missing.path(),
            &LoadOptions::default(),
            family_options,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        MuseGlimmerCheckpointLoadError::Checkpoint(CheckpointError::MissingMappingSource { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let unexpected = TempDir::new().unwrap();
    write_checkpoint(unexpected.path(), &contract, |tensors| {
        tensors.push(("unknown.weight".into(), vec![1]));
    });
    let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint_with_family_options(
            unexpected.path(),
            &LoadOptions::default(),
            family_options,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        MuseGlimmerCheckpointLoadError::Checkpoint(CheckpointError::StrictValidation {
            unexpected: 1,
            ..
        })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let wrong = TempDir::new().unwrap();
    write_checkpoint(wrong.path(), &contract, |tensors| {
        tensors
            .iter_mut()
            .find(|(name, _)| name.ends_with("embed_tokens.weight"))
            .unwrap()
            .1 = vec![15, 4];
    });
    let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let error = model
        .load_checkpoint_with_family_options(wrong.path(), &LoadOptions::default(), family_options)
        .unwrap_err();
    assert!(matches!(
        error,
        MuseGlimmerCheckpointLoadError::Checkpoint(CheckpointError::ShapeMismatch { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());
}

/// Sink dispatch errors are typed rather than falling through to tensor panics.
#[test]
fn sink_rejects_bad_targets_payloads_and_aliases() {
    let device = Default::default();
    let mut model =
        MuseGlimmerForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    assert!(matches!(
        model.write_tensor(
            "model.embed_tokens.weight",
            TensorData::zeros::<i32, _>([16, 4])
        ),
        Err(MuseGlimmerLoadError::Parameter { .. })
    ));
    assert!(matches!(
        model.write_tensor("model.layers.bad.weight", TensorData::from([0.0_f32])),
        Err(MuseGlimmerLoadError::InvalidIndex { .. })
    ));
    assert!(matches!(
        model.write_tensor(
            "model.layers.9.input_layernorm.weight",
            TensorData::zeros::<f32, _>([4])
        ),
        Err(MuseGlimmerLoadError::InvalidIndex { .. })
    ));
    assert!(matches!(
        model.alias_tensor("lm_head.weight", "model.embed_tokens.weight"),
        Err(MuseGlimmerLoadError::InvalidAlias { .. })
    ));
}
