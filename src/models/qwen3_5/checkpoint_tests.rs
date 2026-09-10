use std::{collections::BTreeMap, path::PathBuf, str::FromStr};

use serde::Deserialize;
use serde_json::Value;

use crate::{
    checkpoint::{
        ArtifactSource, CheckpointError, CheckpointManifest, KeyMatcher, LoadOptions,
        PlannedTransform, TensorDType, TensorInventory, TensorOperation, TensorTransform,
    },
    models::config::ModelConfig,
};

use super::*;

const QWEN_08_CONFIG: &str = include_str!("../../../tests/fixtures/qwen3_5_0_8b.json");
const QWEN_27_CONFIG: &str = include_str!("../../../tests/fixtures/qwen3_8_27b.json");
const QWEN_08_INVENTORY: &str =
    include_str!("../../../tests/fixtures/qwen3_5_checkpoint/qwen3_5_0_8b_inventory.json");
const QWEN_27_INVENTORY: &str =
    include_str!("../../../tests/fixtures/qwen3_5_checkpoint/qwen3_8_27b_inventory.json");
const TINY_INVENTORY: &str =
    include_str!("../../../tests/fixtures/qwen3_5_checkpoint/tiny_text_inventory.json");

#[derive(Debug, Deserialize)]
struct GoldenInventory {
    checkpoint: String,
    revision: String,
    #[serde(default)]
    config: Option<Value>,
    tensors: Vec<GoldenTensor>,
}

#[derive(Debug, Clone, Deserialize)]
struct GoldenTensor {
    name: String,
    shape: Vec<u64>,
    dtype: String,
}

fn golden(json: &str) -> GoldenInventory {
    serde_json::from_str(json).unwrap()
}

fn config(json: &str) -> Qwen3_5TextConfig {
    <Qwen3_5TextConfig as ModelConfig>::from_json_str(json).unwrap()
}

fn multimodal_with_mtp() -> Qwen3_5CheckpointOptions {
    Qwen3_5CheckpointOptions {
        source: Qwen3_5CheckpointSource::Multimodal,
        include_mtp: true,
    }
}

fn manifest(golden: &GoldenInventory) -> CheckpointManifest {
    let mut offset = 0_u64;
    let mut tensors = BTreeMap::new();
    for tensor in &golden.tensors {
        let dtype = TensorDType::from_str(&tensor.dtype).unwrap();
        let elements = tensor.shape.iter().copied().product();
        let byte_len = dtype.storage_bytes(elements).unwrap();
        let end = offset.checked_add(byte_len).unwrap();
        tensors.insert(
            tensor.name.clone(),
            TensorInventory {
                name: tensor.name.clone(),
                shard: "metadata-only-fixture.safetensors".into(),
                shape: tensor.shape.clone(),
                dtype,
                data_offsets: [offset, end],
                byte_len,
            },
        );
        offset = end;
    }
    CheckpointManifest {
        root: PathBuf::from("metadata-only-fixture"),
        source: ArtifactSource::Single {
            relative_path: "metadata-only-fixture.safetensors".into(),
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

fn expected_shape<'a>(contract: &'a Qwen3_5CheckpointContract, name: &str) -> &'a [u64] {
    &contract
        .expected_tensors()
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap()
        .shape
}

fn planned<'a>(plan: &'a LoadPlan, target: &str) -> &'a crate::checkpoint::PlannedTensor {
    plan.tensors
        .iter()
        .find(|tensor| tensor.target == target)
        .unwrap()
}

#[test]
fn tiny_text_fixture_is_an_exact_strict_contract() {
    let fixture = golden(TINY_INVENTORY);
    assert_eq!(fixture.checkpoint, "synthetic/qwen3_5-tiny");
    assert_eq!(fixture.revision, "test-fixture-v1");
    let config = config(&fixture.config.as_ref().unwrap().to_string());
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: true,
        })
        .unwrap();
    let plan = contract
        .build_load_plan(&manifest(&fixture), &LoadOptions::default())
        .unwrap();

    assert_eq!(fixture.tensors.len(), 42);
    assert_eq!(contract.expected_tensors().len(), 43);
    assert_eq!(plan.parameter_count(), 43);
    assert!(plan.report.missing.is_empty());
    assert!(plan.report.unexpected.is_empty());
    assert!(plan.report.allowlisted_ignored.is_empty());
    assert!(matches!(
        plan.aliases.as_slice(),
        [alias]
            if alias.target == "lm_head.weight"
                && matches!(
                    &alias.transform,
                    PlannedTransform::Alias { source_target }
                        if source_target == "model.embed_tokens.weight"
                )
    ));
    assert_eq!(
        expected_shape(&contract, "model.layers.0.linear_attn.in_proj_qkv.weight"),
        [12, 8]
    );
    assert_eq!(expected_shape(&contract, "mtp.fc.weight"), [8, 16]);
}

#[test]
fn official_qwen35_point8_metadata_is_consumed_without_unknown_keys() {
    let fixture = golden(QWEN_08_INVENTORY);
    assert_eq!(fixture.checkpoint, "Qwen/Qwen3.5-0.8B");
    assert_eq!(fixture.revision, "2fc06364715b967f1860aea9cf38778875588b17");
    assert_eq!(fixture.tensors.len(), 488);
    let contract = config(QWEN_08_CONFIG)
        .checkpoint_contract(multimodal_with_mtp())
        .unwrap();
    let plan = contract
        .build_load_plan(&manifest(&fixture), &LoadOptions::default())
        .unwrap();

    assert_eq!(contract.expected_tensors().len(), 336);
    assert_eq!(plan.tensors.len(), 335);
    assert_eq!(plan.aliases.len(), 1);
    assert_eq!(plan.report.allowlisted_ignored.len(), 153);
    assert!(
        plan.report
            .allowlisted_ignored
            .iter()
            .all(|name| name.starts_with("model.visual."))
    );
    assert!(
        plan.report
            .allowlisted_ignored
            .iter()
            .all(|name| !name.starts_with("mtp."))
    );

    let delta = planned(&plan, "model.layers.0.linear_attn.in_proj_qkv.weight");
    assert_eq!(delta.shape, [6_144, 1_024]);
    assert_eq!(
        delta.sources[0].name,
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    );
    let query = planned(&plan, "model.layers.3.self_attn.q_proj.weight");
    assert_eq!(query.shape, [4_096, 1_024]);
    assert!(matches!(query.transform, PlannedTransform::Identity));
    assert!(matches!(
        planned(&plan, "mtp.layers.0.self_attn.q_proj.weight").transform,
        PlannedTransform::Identity
    ));
}

#[test]
fn official_qwen38_27b_metadata_is_consumed_without_unknown_keys() {
    let fixture = golden(QWEN_27_INVENTORY);
    assert_eq!(fixture.checkpoint, "Qwen/Qwen3.8-27B");
    assert_eq!(fixture.revision, "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0");
    assert_eq!(fixture.tensors.len(), 1_199);
    let contract = config(QWEN_27_CONFIG)
        .checkpoint_contract(multimodal_with_mtp())
        .unwrap();
    let plan = contract
        .build_load_plan(&manifest(&fixture), &LoadOptions::default())
        .unwrap();

    assert_eq!(contract.expected_tensors().len(), 866);
    assert_eq!(plan.tensors.len(), 866);
    assert!(plan.aliases.is_empty());
    assert_eq!(plan.report.allowlisted_ignored.len(), 333);
    assert_eq!(
        expected_shape(&contract, "model.layers.0.linear_attn.conv1d.weight"),
        [10_240, 1, 4]
    );
    assert_eq!(
        expected_shape(&contract, "model.layers.0.linear_attn.in_proj_a.weight"),
        [48, 5_120]
    );
    assert_eq!(
        expected_shape(&contract, "model.layers.0.linear_attn.in_proj_z.weight"),
        [6_144, 5_120]
    );
    assert_eq!(
        expected_shape(&contract, "model.layers.3.self_attn.q_proj.weight"),
        [12_288, 5_120]
    );
    assert_eq!(
        expected_shape(&contract, "model.layers.3.self_attn.o_proj.weight"),
        [5_120, 6_144]
    );
    assert_eq!(expected_shape(&contract, "mtp.fc.weight"), [5_120, 10_240]);
}

#[test]
fn safe_base_only_default_classifies_optional_mtp_tensors_separately() {
    let fixture = golden(QWEN_08_INVENTORY);
    let contract = config(QWEN_08_CONFIG)
        .checkpoint_contract(Qwen3_5CheckpointOptions::default())
        .unwrap();
    assert_eq!(contract.expected_tensors().len(), 321);
    assert!(
        contract
            .allow_unexpected()
            .contains(&KeyMatcher::prefix("mtp."))
    );

    let plan = contract
        .build_load_plan(&manifest(&fixture), &LoadOptions::default())
        .unwrap();
    assert_eq!(plan.report.allowlisted_ignored.len(), 168);
    assert_eq!(
        plan.report
            .allowlisted_ignored
            .iter()
            .filter(|name| name.starts_with("mtp."))
            .count(),
        15
    );
}

#[test]
fn multimodal_exceptions_are_additive_and_deduplicated() {
    let contract = config(QWEN_08_CONFIG)
        .checkpoint_contract(Qwen3_5CheckpointOptions::default())
        .unwrap();
    let custom = KeyMatcher::exact("local.adapter.weight");
    let mut options = LoadOptions {
        allow_unexpected: vec![KeyMatcher::prefix("model.visual."), custom.clone()],
        ..LoadOptions::default()
    };
    contract.configure_load_options(&mut options);
    assert_eq!(
        options
            .allow_unexpected
            .iter()
            .filter(|matcher| **matcher == KeyMatcher::prefix("model.visual."))
            .count(),
        1
    );
    assert!(options.allow_unexpected.contains(&custom));
}

#[test]
fn query_weight_and_bias_preserve_official_head_local_layout() {
    let fixture = golden(TINY_INVENTORY);
    let mut config = config(&fixture.config.unwrap().to_string());
    config.attention_bias = true;
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: true,
        })
        .unwrap();
    for target in [
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_proj.bias",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.q_proj.bias",
    ] {
        let operation = contract
            .mappings()
            .operations()
            .iter()
            .find(|operation| {
                matches!(operation, TensorOperation::Map { target: found, .. } if found == target)
            })
            .unwrap();
        assert!(matches!(
            operation,
            TensorOperation::Map {
                transform: TensorTransform::Identity,
                ..
            }
        ));
    }
    assert_eq!(
        expected_shape(&contract, "model.layers.1.self_attn.q_proj.bias"),
        [16]
    );
}

#[test]
fn malformed_gated_query_shape_fails_inventory_validation() {
    let mut fixture = golden(TINY_INVENTORY);
    fixture
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "model.layers.1.self_attn.q_proj.weight")
        .unwrap()
        .shape = vec![15, 8];
    let config = config(&fixture.config.as_ref().unwrap().to_string());
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: true,
        })
        .unwrap();
    assert!(matches!(
        contract.build_load_plan(&manifest(&fixture), &LoadOptions::default()),
        Err(CheckpointError::ShapeMismatch { .. })
    ));
}

#[test]
fn dedicated_mtp_embeddings_are_rejected_until_a_named_contract_exists() {
    let fixture = golden(TINY_INVENTORY);
    let mut config = config(&fixture.config.unwrap().to_string());
    config.mtp_use_dedicated_embeddings = true;
    let error = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: true,
        })
        .unwrap_err();
    assert_eq!(error.field, "mtp_use_dedicated_embeddings");
}

#[test]
fn dimension_products_are_checked() {
    let fixture = golden(TINY_INVENTORY);
    let mut config = config(&fixture.config.unwrap().to_string());
    config.num_attention_heads = usize::MAX;
    config.num_key_value_heads = 1;
    let error = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: false,
        })
        .unwrap_err();
    assert!(error.reason.contains("overflows"));
}
