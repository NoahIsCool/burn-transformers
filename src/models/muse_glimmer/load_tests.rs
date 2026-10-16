//! Synthetic SafeTensors tests for Muse parameter installation and recoverable loader failures.

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

/// Deterministic CPU backend using F32 values and I32 token indices.
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
            .load_checkpoint_with(temp.path(), &LoadOptions::default(), family_options)
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
        .load_checkpoint_with(missing.path(), &LoadOptions::default(), family_options)
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
        .load_checkpoint_with(unexpected.path(), &LoadOptions::default(), family_options)
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
        .load_checkpoint_with(wrong.path(), &LoadOptions::default(), family_options)
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

/// Writes `target` with a one-element payload and returns the sink's typed error.
fn sink_error(
    model: &mut MuseGlimmerForCausalLm<TestBackend>,
    target: &str,
) -> MuseGlimmerLoadError {
    model
        .write_tensor(target, TensorData::from([0.0_f32]))
        .expect_err(target)
}

/// Malformed, out-of-range, unknown, and unconfigured targets name their cause.
#[test]
fn sink_dispatch_rejects_every_malformed_or_absent_target_with_its_cause() {
    let device = Default::default();
    let mut model =
        MuseGlimmerForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    for target in [
        "model.unknown.weight",
        "lm_head.bias",
        "model.layers.0.unknown.weight",
        "model.layers.0.mlp.gate_proj.bias",
        "model.layers.0.self_attn.gate_proj.bias",
    ] {
        assert_eq!(
            sink_error(&mut model, target),
            MuseGlimmerLoadError::UnknownTarget {
                target: target.into()
            }
        );
    }
    for (target, reason) in [
        (
            "model.layers.1.input_layernorm.weight",
            "layer 1 is outside 0..1",
        ),
        (
            "model.layers.0",
            "expected `model.layers.<index>.<parameter>`",
        ),
        ("model.layers.0.", "parameter suffix must not be empty"),
        (
            "model.layers.+.input_layernorm.weight",
            "`+` is not a decimal usize: invalid digit found in string",
        ),
    ] {
        assert_eq!(
            sink_error(&mut model, target),
            MuseGlimmerLoadError::InvalidIndex {
                target: target.into(),
                reason: reason.into()
            }
        );
    }
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        let target = format!("model.layers.0.self_attn.{projection}.bias");
        assert_eq!(
            sink_error(&mut model, &target),
            MuseGlimmerLoadError::MissingComponent {
                target,
                component: "configured attention projection bias"
            }
        );
    }
    let mut tied = MuseGlimmerForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let error = sink_error(&mut tied, "lm_head.weight");
    assert_eq!(
        error,
        MuseGlimmerLoadError::MissingComponent {
            target: "lm_head.weight".into(),
            component: "independent LM head for an untied checkpoint"
        }
    );
    assert_eq!(
        error.to_string(),
        "Muse-Glimmer target `lm_head.weight` requires missing component \
         `independent LM head for an untied checkpoint`"
    );
    // No rejected target initialized a parameter as a side effect.
    assert!(!model.model.embed_tokens.weight.is_initialized());
    assert!(
        !model.model.layers[0]
            .input_layernorm
            .weight
            .is_initialized()
    );
}

/// Every enabled attention bias installs exactly; rendered messages name target and cause.
#[test]
fn enabled_attention_biases_install_and_errors_render_their_cause() {
    let device = Default::default();
    let config = MuseGlimmerTextConfig {
        attention_bias: true,
        ..tiny_config(false)
    };
    let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config, &device).unwrap();
    for (projection, width) in [("q_proj", 4), ("k_proj", 2), ("v_proj", 2), ("o_proj", 4)] {
        let target = format!("model.layers.0.self_attn.{projection}.bias");
        let values = (0..width)
            .map(|index| index as f32 + 0.5)
            .collect::<Vec<_>>();
        model
            .write_tensor(&target, TensorData::new(values.clone(), [width]))
            .unwrap();
        let attention = &model.model.layers[0].self_attn;
        let linear = match projection {
            "q_proj" => &attention.q_proj,
            "k_proj" => &attention.k_proj,
            "v_proj" => &attention.v_proj,
            _ => &attention.o_proj,
        };
        assert_eq!(
            linear
                .bias
                .as_ref()
                .unwrap()
                .val()
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            values
        );
    }
    assert_eq!(
        sink_error(&mut model, "model.other").to_string(),
        "unknown Muse-Glimmer parameter target `model.other`"
    );
    assert_eq!(
        sink_error(&mut model, "model.layers.4.input_layernorm.weight").to_string(),
        "invalid Muse-Glimmer target `model.layers.4.input_layernorm.weight`: \
         layer 4 is outside 0..1"
    );
    let error = sink_error(&mut model, "model.norm.weight");
    assert!(
        matches!(&error, MuseGlimmerLoadError::Parameter { target, .. } if target == "model.norm.weight")
    );
    assert!(
        error
            .to_string()
            .starts_with("cannot install Muse-Glimmer parameter `model.norm.weight`: "),
        "{error}"
    );
    assert!(std::error::Error::source(&error).is_some());
}

/// Only an installed embedding may alias into a tied head; each refusal names its cause.
#[test]
fn tied_alias_requires_the_tied_topology_and_an_installed_owner() {
    let device = Default::default();
    let (head, embedding) = ("lm_head.weight", "model.embed_tokens.weight");
    let alias_error = |source: &str, reason: &str| MuseGlimmerLoadError::InvalidAlias {
        target: head.into(),
        source_target: source.into(),
        reason: reason.into(),
    };
    let mut untied =
        MuseGlimmerForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    assert_eq!(
        untied.alias_tensor(head, "model.norm.weight").unwrap_err(),
        alias_error(
            "model.norm.weight",
            "only the tied embedding-to-LM-head alias is supported"
        )
    );
    // The embedding may only alias into the head, never into another parameter.
    assert_eq!(
        untied
            .alias_tensor("model.norm.weight", embedding)
            .unwrap_err(),
        MuseGlimmerLoadError::InvalidAlias {
            target: "model.norm.weight".into(),
            source_target: embedding.into(),
            reason: "only the tied embedding-to-LM-head alias is supported".into(),
        }
    );
    assert_eq!(
        untied.alias_tensor(head, embedding).unwrap_err(),
        alias_error(
            embedding,
            "the configured model declares an independent LM head"
        )
    );
    let mut tied = MuseGlimmerForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let error = tied.alias_tensor(head, embedding).unwrap_err();
    assert_eq!(
        error,
        alias_error(
            embedding,
            "the embedding storage owner has not been installed"
        )
    );
    assert_eq!(
        error.to_string(),
        "invalid Muse-Glimmer alias `lm_head.weight` -> `model.embed_tokens.weight`: \
         the embedding storage owner has not been installed"
    );
    tied.write_tensor(embedding, TensorData::zeros::<f32, _>([16, 4]))
        .unwrap();
    tied.alias_tensor(head, embedding).unwrap();
    // A tied configuration whose graph regained an independent head is refused.
    let mut regrown = untied.clone();
    regrown.model.config.tie_word_embeddings = true;
    assert_eq!(
        regrown.alias_tensor(head, embedding).unwrap_err(),
        alias_error(
            embedding,
            "the configured model declares an independent LM head"
        )
    );
}

/// The default loader reads the released multimodal layout and audits visual keys.
#[test]
fn default_loader_reads_the_multimodal_layout_and_allowlists_visual_keys() {
    let device = Default::default();
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .unwrap();
    let temp = TempDir::new().unwrap();
    write_checkpoint(temp.path(), &contract, |tensors| {
        assert!(
            tensors
                .iter()
                .any(|(name, _)| name == "model.language_model.embed_tokens.weight")
        );
        tensors.push(("model.vision_tower.patch.weight".into(), vec![2]));
        tensors.push(("model.vision_projection.weight".into(), vec![2]));
    });
    let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let outcome = model
        .load_checkpoint(temp.path(), &LoadOptions::default())
        .unwrap();
    assert_eq!(
        outcome.report.applied.len(),
        contract.expected_tensors().len()
    );
    let mut allowed = outcome.report.allowlisted_ignored.clone();
    allowed.sort();
    assert_eq!(
        allowed,
        [
            "model.vision_projection.weight",
            "model.vision_tower.patch.weight"
        ]
    );
    assert!(outcome.report.unexpected.is_empty());
    assert!(model.model.embed_tokens.weight.is_initialized());
}

/// Configuration, discovery, and pass-two failures are distinct; only pass two reports progress.
#[test]
fn checkpoint_load_errors_distinguish_configuration_planning_and_execution() {
    let device = Default::default();
    let config = tiny_config(false);
    let options = MuseGlimmerCheckpointOptions {
        source: MuseGlimmerCheckpointSource::TextOnly,
    };
    let contract = config.checkpoint_contract(options).unwrap();
    let temp = TempDir::new().unwrap();
    write_checkpoint(temp.path(), &contract, |_| {});

    // A configuration edited after initialization cannot describe a checkpoint.
    let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    model.model.config.hidden_activation = "gelu".into();
    let error = model
        .load_checkpoint_with(temp.path(), &LoadOptions::default(), options)
        .unwrap_err();
    assert!(
        matches!(&error, MuseGlimmerCheckpointLoadError::Config(config)
        if config.field == "hidden_activation")
    );
    assert!(error.execution_failure().is_none());

    let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint_with(temp.path().join("absent"), &LoadOptions::default(), options)
        .unwrap_err();
    assert!(
        matches!(&error, MuseGlimmerCheckpointLoadError::Checkpoint(CheckpointError::Io {
            operation, path, ..
        }) if *operation == "inspect checkpoint root" && path.ends_with("absent")),
        "{error}"
    );
    assert!(error.execution_failure().is_none());

    // A graph whose head was removed after planning fails in pass two with progress.
    let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config, &device).unwrap();
    model.lm_head = None;
    let error = model
        .load_checkpoint_with(temp.path(), &LoadOptions::default(), options)
        .unwrap_err();
    let failure = error
        .execution_failure()
        .expect("pass two reached the sink");
    assert_eq!(failure.failed_target.as_deref(), Some("lm_head.weight"));
    assert!(matches!(
        &failure.cause,
        crate::checkpoint::ExecutionFailureCause::Sink(MuseGlimmerLoadError::MissingComponent {
            component: "independent LM head for an untied checkpoint",
            ..
        })
    ));
    assert!(error.to_string().contains("lm_head.weight"), "{error}");
}

/// Deterministic `[rows, columns]` payload whose row magnitudes grow by `1 + 0.37 * row`.
///
/// Every `[1, G]` quantization block of such a matrix has its own scale, so a
/// backend that requantizes the transposed buffer over runs of the *input*
/// axis (Burn Flex 0.21) cannot reproduce the artifact's blocks.
fn row_scaled_values(name: &str, shape: &[usize]) -> Vec<f32> {
    let columns = shape.last().copied().unwrap_or(1).max(1);
    let elements = shape.iter().product::<usize>();
    let seed = name
        .bytes()
        .fold(0_usize, |state, byte| state.wrapping_add(byte as usize));
    (0..elements)
        .map(|index| {
            let row = index / columns;
            (((index * 7 + seed) % 31) as f32 - 15.0) * 0.002 * (1.0 + 0.37 * row as f32)
        })
        .collect()
}

/// Asserts a column-layout `[in, out]` linear weight holds exactly the dequantized `[out, in]` artifact.
///
/// The installed weight must be packed QFloat exactly when the backend keeps
/// the artifact's blocks through the transpose, and plain float otherwise.
fn assert_exact_artifact(target: &str, weight: Tensor<TestBackend, 2>, artifact: &TensorData) {
    let packed = matches!(weight.dtype(), burn::tensor::DType::QFloat(_));
    let exact_layout = crate::checkpoint::packed_layout_is_exact::<TestBackend>(&weight.device());
    let installed = if packed { weight.dequantize() } else { weight };
    let [inputs, outputs] = installed.dims();
    assert_eq!(artifact.shape.to_vec(), vec![outputs, inputs], "{target}");
    let values = installed.into_data().to_vec::<f32>().unwrap();
    let expected = artifact.to_vec::<f32>().unwrap();
    let mismatches = (0..inputs * outputs)
        .filter(|&flat| {
            let (input, output) = (flat / outputs, flat % outputs);
            values[flat].to_bits() != expected[output * inputs + input].to_bits()
        })
        .count();
    assert_eq!(
        mismatches, 0,
        "{target}: installed weight differs from the artifact at {mismatches} entries"
    );
    assert_eq!(
        packed, exact_layout,
        "{target}: packed storage must be kept exactly when the backend's layout is exact"
    );
}

/// Packed sidecar weights install exactly the artifact, never a requantized approximation.
///
/// Every quantization-eligible projection (five attention and three FFN
/// matrices per layer, plus the LM head) is written as Q8 `[1, 32]`-block QFloat
/// data, the form the Burnpack sidecar streams into the sink. On a backend
/// whose quantized transpose requantizes (Burn Flex 0.21), the column-layout
/// weights must hold the exact dequantized artifact in float storage; where the
/// transpose is exact they stay packed. The fully packed graph must still execute.
#[test]
fn packed_projections_install_the_artifact_exactly() {
    use crate::checkpoint::quantization::{
        QuantizationEligibility, QuantizationGroupSize, WeightQuantizationConfig,
        classify_weight_for_quantization, quantize_tensor_data,
    };

    let config = MuseGlimmerTextConfig {
        vocab_size: 64,
        hidden_size: 64,
        intermediate_size: 64,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        max_position_embeddings: 16,
        sliding_window: Some(4),
        layer_types: vec![super::super::MuseGlimmerLayerType::FullAttention],
        layer_rope_theta: vec![0.0],
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: false,
        attention_bias: true,
        ..Default::default()
    };
    let contract = config
        .checkpoint_contract(MuseGlimmerCheckpointOptions {
            source: MuseGlimmerCheckpointSource::TextOnly,
        })
        .unwrap();
    let device = Default::default();
    let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let quantization = WeightQuantizationConfig {
        group_size: QuantizationGroupSize::new(32).unwrap(),
        ..WeightQuantizationConfig::default()
    };
    let mut artifacts = std::collections::BTreeMap::new();
    for expected in contract.expected_tensors() {
        let shape = expected
            .shape
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        let source = TensorData::new(row_scaled_values(&expected.name, &shape), shape.clone());
        let data = match classify_weight_for_quantization(&expected.name, &shape) {
            QuantizationEligibility::Eligible => {
                let quantized = quantize_tensor_data(&source, quantization).unwrap();
                artifacts.insert(expected.name.clone(), quantized.dequantize_f32().unwrap());
                quantized.into_burn_qfloat_data().unwrap()
            }
            QuantizationEligibility::Excluded(_) => source,
        };
        model
            .write_tensor(&expected.name, data)
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    // Five attention and three FFN projections in the layer, plus the head.
    assert_eq!(artifacts.len(), 9);

    let layer = &model.model.layers[0];
    for (target, weight) in [
        (
            "lm_head.weight",
            model.lm_head.as_ref().unwrap().weight.val(),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            layer.self_attn.q_proj.weight.val(),
        ),
        (
            "model.layers.0.self_attn.gate_proj.weight",
            layer.self_attn.gate_proj.weight.val(),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            layer.self_attn.o_proj.weight.val(),
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            layer.mlp.down_proj.weight.val(),
        ),
    ] {
        assert_exact_artifact(target, weight, &artifacts[target]);
    }
    // Excluded tensors (embedding, norms, biases) keep their float source.
    assert_eq!(
        model.model.embed_tokens.weight.val().dtype(),
        burn::tensor::DType::F32
    );
    assert_eq!(
        layer.self_attn.q_proj.bias.as_ref().unwrap().val().dtype(),
        burn::tensor::DType::F32
    );

    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device),
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
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}

/// The default loader detects a text-only checkpoint, as automatic loading does.
///
/// Before namespace detection, `load_checkpoint` assumed the multimodal layout
/// and rejected every standalone text checkpoint as missing and unexpected
/// tensors.
#[test]
fn default_loader_detects_a_text_only_checkpoint() {
    let device = Default::default();
    for tied in [false, true] {
        let config = tiny_config(tied);
        let contract = config
            .checkpoint_contract(MuseGlimmerCheckpointOptions {
                source: MuseGlimmerCheckpointSource::TextOnly,
            })
            .unwrap();
        let temp = TempDir::new().unwrap();
        write_checkpoint(temp.path(), &contract, |tensors| {
            assert!(
                tensors
                    .iter()
                    .any(|(name, _)| name == "model.embed_tokens.weight")
            );
        });
        let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config, &device).unwrap();
        let outcome = model
            .load_checkpoint(temp.path(), &LoadOptions::default())
            .unwrap();
        assert!(outcome.report.is_strictly_valid());
        assert_eq!(
            outcome.report.applied.len(),
            contract.expected_tensors().len()
        );
        assert!(outcome.report.allowlisted_ignored.is_empty());
    }
}

/// Neither or both language namespaces fail detection with a typed, family-named error.
#[test]
fn default_loader_rejects_missing_and_ambiguous_namespaces() {
    let device = Default::default();
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(MuseGlimmerCheckpointOptions {
            source: MuseGlimmerCheckpointSource::TextOnly,
        })
        .unwrap();

    let ambiguous = TempDir::new().unwrap();
    write_checkpoint(ambiguous.path(), &contract, |tensors| {
        tensors.push(("model.language_model.norm.weight".into(), vec![4]));
    });
    let mut model = MuseGlimmerForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(ambiguous.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(
        matches!(
            error,
            MuseGlimmerCheckpointLoadError::Checkpoint(
                CheckpointError::AmbiguousLanguageNamespace { family: "Muse" }
            )
        ),
        "{error}"
    );
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let neither = TempDir::new().unwrap();
    write_checkpoint(neither.path(), &contract, |tensors| {
        tensors.retain(|(name, _)| name == "lm_head.weight");
    });
    let error = model
        .load_checkpoint(neither.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(
        matches!(
            error,
            MuseGlimmerCheckpointLoadError::Checkpoint(
                CheckpointError::LanguageNamespaceNotFound { family: "Muse" }
            )
        ),
        "{error}"
    );
    assert!(error.execution_failure().is_none());
}

/// The 0.1.0 options method forwards unchanged to `load_checkpoint_with`.
#[test]
#[allow(deprecated)]
fn deprecated_family_options_method_forwards_to_load_checkpoint_with() {
    let device = Default::default();
    let config = tiny_config(false);
    let options = MuseGlimmerCheckpointOptions {
        source: MuseGlimmerCheckpointSource::TextOnly,
    };
    let contract = config.checkpoint_contract(options).unwrap();
    let temp = TempDir::new().unwrap();
    write_checkpoint(temp.path(), &contract, |_| {});
    let mut current = MuseGlimmerForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let mut former = MuseGlimmerForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let expected = current
        .load_checkpoint_with(temp.path(), &LoadOptions::default(), options)
        .unwrap();
    let actual = former
        .load_checkpoint_with_family_options(temp.path(), &LoadOptions::default(), options)
        .unwrap();
    assert_eq!(actual.report, expected.report);
    // Forcing the other namespace through either name is a strict planning error.
    let error = former
        .load_checkpoint_with_family_options(
            temp.path(),
            &LoadOptions::default(),
            MuseGlimmerCheckpointOptions::default(),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        MuseGlimmerCheckpointLoadError::Checkpoint(_)
    ));
}
