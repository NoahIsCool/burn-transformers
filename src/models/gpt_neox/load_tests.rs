//! GPT-NeoX checkpoint preflight, packed-QKV installation, and failure-atomicity tests.

use std::{fs, path::Path};

use burn::{
    backend::Flex,
    tensor::{DType, TensorData},
};
use serde_json::{Map, Value, json};
use tempfile::TempDir;

use crate::checkpoint::{DTypePolicy, TensorDType, TensorOperation};

use super::*;

/// Deterministic CPU backend using F32 parameters and I32 tensor indices.
type TestBackend = Flex<f32, i32>;
/// Serialized source tensor name paired with its checkpoint dimensions.
type FixtureTensor = (String, Vec<u64>);

/// Builds a one-layer, two-head fixture with full-head RoPE and an independent vocabulary head.
fn tiny() -> GptNeoXConfig {
    GptNeoXConfig {
        hidden_size: 4,
        intermediate_size: 6,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        vocab_size: 7,
        max_position_embeddings: 8,
        rotary_pct: 1.0,
        ..GptNeoXConfig::default()
    }
}

/// Writes a synthetic F32 SafeTensors inventory with ascending values and distinct bias offsets.
fn write_checkpoint(root: &Path, mutate: impl FnOnce(&mut Vec<FixtureTensor>)) {
    let contract = tiny().checkpoint_contract().unwrap();
    let mut tensors = contract
        .mappings()
        .operations()
        .iter()
        .map(|mapping| {
            let TensorOperation::Map { source, target, .. } = mapping else {
                panic!("fixture expects direct mappings before architecture expansion");
            };
            let expected = contract
                .expected_tensors()
                .iter()
                .find(|tensor| tensor.name == *target)
                .unwrap();
            (source.clone(), expected.shape.clone())
        })
        .collect::<Vec<_>>();
    mutate(&mut tensors);
    let mut payload = Vec::new();
    let mut header = Map::new();
    for (name, shape) in tensors {
        let start = payload.len();
        let count: u64 = shape.iter().product();
        let offset = if name.ends_with(".bias") { 100.0 } else { 0.0 };
        for index in 0..count {
            payload.extend_from_slice(&(index as f32 + offset).to_le_bytes());
        }
        header.insert(
            name,
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    fs::write(root.join("model.safetensors"), bytes).unwrap();
}

/// Reads a parameter in checkpoint layout so tests can inspect packed-QKV row ordering.
fn saved<const D: usize>(parameter: &Param<Tensor<TestBackend, D>>) -> Vec<f32> {
    parameter
        .transform_for_save()
        .val()
        .into_data()
        .to_vec()
        .unwrap()
}

/// Head-interleaved QKV data splits into the expected weights/biases and reports six component transforms.
#[test]
fn local_import_reorders_qkv_heads_and_preserves_projection_layout() {
    let directory = TempDir::new().unwrap();
    write_checkpoint(directory.path(), |_| {});
    let mut model = GptNeoXModel::<TestBackend>::init(tiny(), &Default::default()).unwrap();
    let outcome = model
        .load_checkpoint(directory.path(), &LoadOptions::default())
        .unwrap();
    let attention = &model.layers[0].attention;
    assert_eq!(
        saved(&attention.q_proj.weight),
        (0..8)
            .chain(24..32)
            .map(|value| value as f32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        saved(&attention.k_proj.weight),
        (8..16)
            .chain(32..40)
            .map(|value| value as f32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        saved(&attention.v_proj.weight),
        (16..24)
            .chain(40..48)
            .map(|value| value as f32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        saved(attention.q_proj.bias.as_ref().unwrap()),
        [100.0, 101.0, 106.0, 107.0]
    );
    assert_eq!(
        saved(attention.k_proj.bias.as_ref().unwrap()),
        [102.0, 103.0, 108.0, 109.0]
    );
    assert_eq!(
        saved(attention.v_proj.bias.as_ref().unwrap()),
        [104.0, 105.0, 110.0, 111.0]
    );
    assert_eq!(model.layers[0].mlp.up_proj.weight.val().dims(), [4, 6]);
    assert_eq!(saved(&model.final_layer_norm.gamma), [0.0, 1.0, 2.0, 3.0]);
    assert_eq!(
        saved(model.final_layer_norm.beta.as_ref().unwrap()),
        [100.0, 101.0, 102.0, 103.0]
    );

    let input = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0, 0.0, 0.0]], &Default::default());
    assert_eq!(
        attention
            .q_proj
            .forward(input)
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        [101.0, 106.0, 131.0, 136.0]
    );
    assert_eq!(outcome.report.applied.len(), 20);
    assert!(
        !outcome
            .report
            .applied
            .iter()
            .any(|name| name.contains("query_key_value"))
    );
    assert_eq!(
        outcome
            .report
            .transformed
            .iter()
            .filter(|record| record.operation.starts_with("gpt_neox_qkv_component"))
            .count(),
        6
    );
}

/// Invalid packed-QKV inventories fail during preflight and preserve the embedding bytes.
#[test]
fn missing_wrong_shape_and_unknown_tensors_fail_before_parameter_writes() {
    for case in 0..3 {
        let directory = TempDir::new().unwrap();
        write_checkpoint(directory.path(), |tensors| match case {
            0 => tensors
                .retain(|(name, _)| name != "gpt_neox.layers.0.attention.query_key_value.weight"),
            1 => {
                tensors
                    .iter_mut()
                    .find(|(name, _)| name.ends_with("query_key_value.weight"))
                    .unwrap()
                    .1 = vec![11, 4]
            }
            _ => tensors.push(("gpt_neox.layers.0.attention.unexpected".into(), vec![1])),
        });
        let mut model = GptNeoXModel::<TestBackend>::init(tiny(), &Default::default()).unwrap();
        let original = saved(&model.embed_in.weight);
        let error = model
            .load_checkpoint(directory.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(matches!(error, GptNeoXCheckpointLoadError::Checkpoint(_)));
        assert_eq!(saved(&model.embed_in.weight), original);
    }
}

/// The exact causal-mask, masked-bias, and rotary-frequency buffers are audited as three exceptions.
#[test]
fn exact_legacy_buffer_exceptions_are_reported() {
    let directory = TempDir::new().unwrap();
    write_checkpoint(directory.path(), |tensors| {
        for (suffix, shape) in [
            ("bias", vec![1, 1, 8, 8]),
            ("masked_bias", vec![]),
            ("rotary_emb.inv_freq", vec![1]),
        ] {
            tensors.push((format!("gpt_neox.layers.0.attention.{suffix}"), shape));
        }
    });
    let mut model = GptNeoXModel::<TestBackend>::init(tiny(), &Default::default()).unwrap();
    let outcome = model
        .load_checkpoint(directory.path(), &LoadOptions::default())
        .unwrap();
    assert_eq!(outcome.report.allowlisted_ignored.len(), 3);
    assert!(outcome.report.is_strictly_valid());
}

/// A missing value bias rejects the whole packed-bias write while preserving query/key biases.
#[test]
fn packed_component_failure_is_atomic_and_retains_context() {
    let mut model = GptNeoXModel::<TestBackend>::init(tiny(), &Default::default()).unwrap();
    model.layers[0].attention.v_proj.bias = None;
    let original_query = saved(model.layers[0].attention.q_proj.bias.as_ref().unwrap());
    let original_key = saved(model.layers[0].attention.k_proj.bias.as_ref().unwrap());
    let error = model
        .write_tensor(
            "layers.0.attention.query_key_value.bias",
            TensorData::new((0..12).map(|value| value as f32).collect(), [12]),
        )
        .unwrap_err();
    assert!(
        matches!(error, GptNeoXLoadError::MissingBias { ref target } if target == "layers.0.attention.v_proj.bias")
    );
    assert_eq!(
        saved(model.layers[0].attention.q_proj.bias.as_ref().unwrap()),
        original_query
    );
    assert_eq!(
        saved(model.layers[0].attention.k_proj.bias.as_ref().unwrap()),
        original_key
    );
}

/// Bad payloads, target syntax, aliases, and mutated head geometry return recoverable errors.
#[test]
fn malformed_packed_payload_and_indices_return_errors() {
    let mut model = GptNeoXModel::<TestBackend>::init(tiny(), &Default::default()).unwrap();
    for data in [
        TensorData::new(vec![0.0_f32; 11], [11]),
        TensorData::from_bytes_vec(vec![0; 4], [12], DType::F32),
        TensorData::new(vec![0_i32; 12], [12]),
    ] {
        assert!(
            model
                .write_tensor("layers.0.attention.query_key_value.bias", data)
                .is_err()
        );
    }
    for target in [
        "layers.1.input_layernorm.gamma",
        "layers.-1.input_layernorm.gamma",
        "layers.+0.input_layernorm.gamma",
        "layers.0.",
        "not_a_parameter",
    ] {
        assert!(
            model
                .write_tensor(target, TensorData::new(vec![0.0_f32; 4], [4]))
                .is_err()
        );
    }
    assert!(
        model
            .alias_tensor("embed_out.weight", "embed_in.weight")
            .is_err()
    );
    model.layers[0].attention.head_dim = 0;
    assert!(matches!(
        model.write_tensor(
            "layers.0.attention.query_key_value.bias",
            TensorData::new(Vec::<f32>::new(), [0]),
        ),
        Err(GptNeoXLoadError::InvalidProjection { .. })
    ));
    // Equal total width alone cannot authorize reinterpreting packed head rows.
    model.layers[0].attention.query_heads = 1;
    model.layers[0].attention.key_value_heads = 1;
    model.layers[0].attention.head_dim = 4;
    assert!(matches!(
        model.write_tensor(
            "layers.0.attention.query_key_value.bias",
            TensorData::new(vec![0.0_f32; 12], [12]),
        ),
        Err(GptNeoXLoadError::InvalidProjection { .. })
    ));
}

/// F16 conversion reports each split component; integer targets and integer source storage are rejected.
#[test]
fn explicit_dtype_conversion_attributes_all_qkv_components() {
    let directory = TempDir::new().unwrap();
    write_checkpoint(directory.path(), |_| {});
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        ..LoadOptions::default()
    };
    let manifest = CheckpointManifest::discover(directory.path(), &options).unwrap();
    let plan = tiny().build_load_plan(&manifest, &options).unwrap();
    assert_eq!(plan.report.dtype_converted.len(), 20);
    assert!(
        !plan
            .report
            .dtype_converted
            .iter()
            .any(|record| record.target.contains("query_key_value"))
    );
    assert!(
        plan.report
            .transformed
            .iter()
            .any(|record| record.target == "layers.0.attention.k_proj.weight"
                && record.sources == ["gpt_neox.layers.0.attention.query_key_value.weight"])
    );
    let integer_options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::I32),
        ..LoadOptions::default()
    };
    assert!(tiny().build_load_plan(&manifest, &integer_options).is_err());
    let mut integer_manifest = manifest;
    integer_manifest
        .tensors
        .get_mut("embed_out.weight")
        .unwrap()
        .dtype = TensorDType::I32;
    for shard in &mut integer_manifest.shards {
        if let Some(tensor) = shard.tensors.get_mut("embed_out.weight") {
            tensor.dtype = TensorDType::I32;
        }
    }
    assert!(
        tiny()
            .build_load_plan(&integer_manifest, &LoadOptions::default())
            .unwrap_err()
            .to_string()
            .contains("floating-point storage")
    );
}

/// A tied-head configuration cannot produce the independent-head checkpoint contract.
#[test]
fn tied_checkpoint_topology_is_rejected_explicitly() {
    let mut config = tiny();
    config.tie_word_embeddings = true;
    assert!(
        config
            .checkpoint_contract()
            .unwrap_err()
            .to_string()
            .contains("tie_word_embeddings")
    );
}
