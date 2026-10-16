//! Metadata-only tests for strict Muse text inventories, visual exceptions, and tied storage.

use std::{collections::BTreeMap, path::PathBuf};

use crate::checkpoint::{
    ArtifactSource, CheckpointError, CheckpointManifest, DTypePolicy, TensorDType, TensorInventory,
};

use super::*;

/// Produces a small four-layer topology retaining the released schedule.
fn tiny_config() -> MuseGlimmerTextConfig {
    MuseGlimmerTextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        sliding_window: Some(3),
        bos_token_id: None,
        eos_token_id: None,
        ..Default::default()
    }
}

/// Constructs a metadata-only manifest from mapped source keys.
fn manifest(
    contract: &MuseGlimmerCheckpointContract,
    dtype: TensorDType,
    include_visual: bool,
) -> CheckpointManifest {
    let mut offset = 0_u64;
    let mut tensors = BTreeMap::new();
    for operation in contract.mappings().operations() {
        let TensorOperation::Map { source, target, .. } = operation else {
            continue;
        };
        let expected = contract
            .expected_tensors()
            .iter()
            .find(|expected| &expected.name == target)
            .unwrap();
        let elements = expected.shape.iter().copied().product();
        let bytes = dtype.storage_bytes(elements).unwrap();
        tensors.insert(
            source.clone(),
            TensorInventory {
                name: source.clone(),
                shard: "metadata.safetensors".into(),
                shape: expected.shape.clone(),
                dtype,
                data_offsets: [offset, offset + bytes],
                byte_len: bytes,
            },
        );
        offset += bytes;
    }
    if include_visual {
        for name in [
            "model.vision_tower.layers.0.attn.q_proj.weight",
            "model.vision_adapter.fc1.weight",
            "model.vision_projection.weight",
        ] {
            tensors.insert(
                name.into(),
                TensorInventory {
                    name: name.into(),
                    shard: "metadata.safetensors".into(),
                    shape: vec![1],
                    dtype,
                    data_offsets: [offset, offset + 4],
                    byte_len: 4,
                },
            );
            offset += 4;
        }
    }
    CheckpointManifest {
        root: PathBuf::from("metadata"),
        source: ArtifactSource::Single {
            relative_path: "metadata.safetensors".into(),
        },
        shards: Vec::new(),
        tensors,
        index_aliases: Vec::new(),
        index_alias_size_accounting: None,
        indexed_tensor_bytes: offset,
        tensor_bytes: offset,
        artifact_bytes: offset,
    }
}

/// Released dimensions produce 627 text-side parameters and exact projection shapes.
#[test]
fn official_contract_cardinality_and_shapes_are_exact() {
    let contract = MuseGlimmerTextConfig::default()
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .unwrap();
    assert_eq!(contract.expected_tensors().len(), 627);
    let shape = |name: &str| {
        contract
            .expected_tensors()
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap()
            .shape
            .clone()
    };
    assert_eq!(
        shape("model.layers.0.self_attn.q_proj.weight"),
        [4096, 6656]
    );
    assert_eq!(shape("model.layers.0.self_attn.k_proj.weight"), [256, 6656]);
    assert_eq!(
        shape("model.layers.51.self_attn.gate_proj.weight"),
        [4096, 6656]
    );
    assert!(
        contract
            .expected_tensors()
            .iter()
            .all(|tensor| { !tensor.name.contains("q_norm") && !tensor.name.contains("k_norm") })
    );
}

/// Multimodal planning consumes language tensors and audits visual exceptions.
#[test]
fn strict_multimodal_plan_allowlists_only_visual_namespaces() {
    let contract = tiny_config()
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .unwrap();
    let plan = contract
        .build_load_plan(
            &manifest(&contract, TensorDType::F32, true),
            &LoadOptions::default(),
        )
        .unwrap();
    assert_eq!(plan.parameter_count(), 51);
    assert_eq!(plan.report.allowlisted_ignored.len(), 3);
    assert!(plan.report.unexpected.is_empty());
}

/// Missing, unknown, and wrong-shaped text tensors fail before pass two.
#[test]
fn strict_plan_rejects_text_inventory_mismatches() {
    let contract = tiny_config()
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .unwrap();
    let mut missing = manifest(&contract, TensorDType::F32, false);
    missing
        .tensors
        .remove("model.language_model.layers.0.self_attn.gate_proj.weight");
    assert!(matches!(
        contract
            .build_load_plan(&missing, &LoadOptions::default())
            .unwrap_err(),
        CheckpointError::MissingMappingSource { .. }
    ));

    let mut unexpected = manifest(&contract, TensorDType::F32, false);
    let sample = unexpected.tensors.values().next().unwrap().clone();
    unexpected.tensors.insert(
        "model.audio_tower.weight".into(),
        TensorInventory {
            name: "model.audio_tower.weight".into(),
            ..sample
        },
    );
    assert!(matches!(
        contract
            .build_load_plan(&unexpected, &LoadOptions::default())
            .unwrap_err(),
        CheckpointError::StrictValidation { unexpected: 1, .. }
    ));

    let mut wrong = manifest(&contract, TensorDType::F32, false);
    wrong
        .tensors
        .get_mut("model.language_model.embed_tokens.weight")
        .unwrap()
        .shape = vec![31, 8];
    assert!(matches!(
        contract
            .build_load_plan(&wrong, &LoadOptions::default())
            .unwrap_err(),
        CheckpointError::ShapeMismatch { .. }
    ));
}

/// Integer parameter payloads are rejected during strict planning.
#[test]
fn strict_plan_rejects_non_float_parameter_dtypes() {
    let contract = tiny_config()
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .unwrap();
    let error = contract
        .build_load_plan(
            &manifest(&contract, TensorDType::I32, false),
            &LoadOptions::default(),
        )
        .unwrap_err();
    assert!(matches!(error, CheckpointError::InvalidMapping { .. }));
}

/// BF16-to-F16 conversion remains visible in the load report.
#[test]
fn dtype_conversion_is_explicit() {
    let contract = tiny_config()
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .unwrap();
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        ..Default::default()
    };
    let plan = contract
        .build_load_plan(&manifest(&contract, TensorDType::BF16, false), &options)
        .unwrap();
    assert_eq!(plan.report.dtype_converted.len(), 51);
}

/// A tied experimental model aliases the embedding storage owner.
#[test]
fn tied_head_uses_one_storage_owner() {
    let mut config = tiny_config();
    config.tie_word_embeddings = true;
    let contract = config
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .unwrap();
    let plan = contract
        .build_load_plan(
            &manifest(&contract, TensorDType::F32, false),
            &LoadOptions::default(),
        )
        .unwrap();
    assert_eq!(plan.aliases.len(), 1);
    assert_eq!(plan.aliases[0].target, "lm_head.weight");
}

/// Each source layout exposes exactly its visual exceptions, and configuring options is idempotent.
#[test]
fn source_layouts_expose_their_visual_exceptions() {
    let multimodal = tiny_config()
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .unwrap();
    assert_eq!(
        multimodal.allow_unexpected(),
        [
            KeyMatcher::prefix("model.vision_tower."),
            KeyMatcher::prefix("model.vision_adapter."),
            KeyMatcher::prefix("model.vision_projection."),
        ]
    );
    let mut options = LoadOptions {
        allow_unexpected: vec![KeyMatcher::exact("caller.extra")],
        ..LoadOptions::default()
    };
    multimodal.configure_load_options(&mut options);
    multimodal.configure_load_options(&mut options);
    assert_eq!(options.allow_unexpected.len(), 4);
    assert_eq!(
        options.allow_unexpected[0],
        KeyMatcher::exact("caller.extra")
    );
    let text_only = tiny_config()
        .checkpoint_contract(MuseGlimmerCheckpointOptions {
            source: MuseGlimmerCheckpointSource::TextOnly,
        })
        .unwrap();
    assert!(text_only.allow_unexpected().is_empty());
}

/// An unrepresentable query width is a configuration error naming its expression.
///
/// Validation accepts any head count divisible by the key/value heads, so the
/// contract's checked width product is the guard that reports `q d` overflow.
#[test]
fn overflowing_attention_width_is_a_configuration_error() {
    let config = MuseGlimmerTextConfig {
        num_attention_heads: usize::MAX / 2 + 1,
        num_key_value_heads: 1,
        head_dim: 2,
        ..tiny_config()
    };
    config.validate().unwrap();
    let error = config
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .unwrap_err();
    assert_eq!(
        (error.architecture, error.field, error.reason.as_str()),
        (
            FAMILY,
            "num_attention_heads * head_dim",
            "dimension product overflows usize"
        )
    );
}
