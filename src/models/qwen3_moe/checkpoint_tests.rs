//! Metadata-only tests of Qwen3 MoE tensor geometry, mappings, and strict load plans.

use std::{collections::BTreeMap, path::PathBuf};

use crate::{
    checkpoint::{
        ArtifactSource, CheckpointError, CheckpointManifest, DTypePolicy, LoadOptions, TensorDType,
        TensorInventory,
    },
    models::config::ModelConfig,
};

use super::*;

/// Tiny topology used to exercise both sparse and dense contract branches.
fn tiny_config() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        max_window_layers: 2,
        decoder_sparse_step: 1,
        moe_intermediate_size: 6,
        num_experts: 4,
        num_experts_per_tok: 2,
        norm_topk_prob: true,
        ..Default::default()
    }
}

/// Constructs a metadata-only manifest matching identity mappings.
fn manifest(contract: &Qwen3MoeCheckpointContract, dtype: TensorDType) -> CheckpointManifest {
    let mut offset = 0;
    let mut tensors = BTreeMap::new();
    for operation in contract.mappings().operations() {
        let crate::checkpoint::TensorOperation::Map { source, target, .. } = operation else {
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

/// Official pinned configuration yields the exact 128-by-eight expert inventory.
#[test]
fn official_a3b_contract_has_expected_cardinality_and_shapes() {
    let config =
        Qwen3MoeConfig::from_json_str(include_str!("../../../tests/fixtures/qwen3_30b_a3b.json"))
            .unwrap();
    let contract = config
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .unwrap();
    assert_eq!(contract.expected_tensors().len(), 18_867);
    let find = |name: &str| {
        contract
            .expected_tensors()
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap()
            .shape
            .clone()
    };
    assert_eq!(find("model.layers.0.self_attn.q_proj.weight"), [4096, 2048]);
    assert_eq!(find("model.layers.0.self_attn.k_proj.weight"), [512, 2048]);
    assert_eq!(find("model.layers.0.sparse_mlp.router.weight"), [128, 2048]);
    assert_eq!(
        find("model.layers.47.sparse_mlp.routed.127.down_proj.weight"),
        [2048, 768]
    );
    assert!(contract.mappings().operations().iter().any(|operation| {
        matches!(
            operation,
            crate::checkpoint::TensorOperation::Map { source, target, .. }
                if source == "model.layers.0.mlp.gate.weight"
                    && target == "model.layers.0.sparse_mlp.router.weight"
        )
    }));
}

/// Strict planning accepts exactly the synthetic tiny inventory.
#[test]
fn exact_inventory_builds_identity_plan() {
    let contract = tiny_config()
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .unwrap();
    let plan = contract
        .build_load_plan(
            &manifest(&contract, TensorDType::F32),
            &LoadOptions::default(),
        )
        .unwrap();
    assert_eq!(plan.parameter_count(), 45);
    assert!(plan.report.missing.is_empty());
    assert!(plan.report.unexpected.is_empty());
    assert!(plan.aliases.is_empty());
}

/// Missing, unexpected, and wrong-shaped tensors all fail closed.
#[test]
fn strict_contract_rejects_inventory_mismatches() {
    let contract = tiny_config()
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .unwrap();

    let mut missing = manifest(&contract, TensorDType::F32);
    missing.tensors.remove("model.layers.0.mlp.gate.weight");
    let error = contract
        .build_load_plan(&missing, &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        CheckpointError::MissingMappingSource { .. }
    ));

    let mut unexpected = manifest(&contract, TensorDType::F32);
    let sample = unexpected.tensors.values().next().unwrap().clone();
    unexpected.tensors.insert(
        "model.layers.0.mlp.shared_expert.weight".into(),
        TensorInventory {
            name: "model.layers.0.mlp.shared_expert.weight".into(),
            ..sample
        },
    );
    let error = contract
        .build_load_plan(&unexpected, &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        CheckpointError::StrictValidation {
            missing: 0,
            unexpected: 1,
            ..
        }
    ));

    let mut wrong_shape = manifest(&contract, TensorDType::F32);
    wrong_shape
        .tensors
        .get_mut("model.layers.0.mlp.gate.weight")
        .unwrap()
        .shape = vec![3, 8];
    let error = contract
        .build_load_plan(&wrong_shape, &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(error, CheckpointError::ShapeMismatch { .. }));
}

/// BF16 checkpoint tensors can be streamed into an F16 Metal-oriented plan.
#[test]
fn dtype_policy_records_bf16_to_f16_conversion() {
    let contract = tiny_config()
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .unwrap();
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        ..Default::default()
    };
    let plan = contract
        .build_load_plan(&manifest(&contract, TensorDType::BF16), &options)
        .unwrap();
    assert!(
        plan.tensors
            .iter()
            .all(|tensor| tensor.dtype == TensorDType::F16)
    );
    assert_eq!(plan.report.dtype_converted.len(), plan.tensors.len());
}

/// A tied head is represented as an alias with no second source tensor.
#[test]
fn tied_head_contract_uses_storage_alias() {
    let mut config = tiny_config();
    config.tie_word_embeddings = true;
    let contract = config
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .unwrap();
    let plan = contract
        .build_load_plan(
            &manifest(&contract, TensorDType::F32),
            &LoadOptions::default(),
        )
        .unwrap();
    assert_eq!(plan.aliases.len(), 1);
    assert_eq!(plan.aliases[0].target, "lm_head.weight");
}

/// A validated topology whose query width `heads * head_dim` overflows fails contract derivation.
///
/// Validation checks each extent but not the fused projection product, so
/// the checked dimension conversion is the gate that keeps an unrepresentable
/// `[heads * head_dim, hidden]` shape out of the tensor inventory.
#[test]
fn overflowing_attention_width_is_a_contract_error() {
    let config = Qwen3MoeConfig {
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 1 << (usize::BITS - 2),
        max_position_embeddings: 32,
        ..tiny_config()
    };
    config
        .validate()
        .expect("each extent is individually valid");
    let error = config
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .expect_err("the query width does not fit usize");
    assert_eq!(error.architecture, "qwen3_moe");
    assert_eq!(error.field, "num_attention_heads * head_dim");
    assert_eq!(error.reason, "dimension product overflows usize");
}
