//! Nemotron-H tensor-inventory, mapping, dtype, and pinned index-accounting tests.

use std::{collections::BTreeMap, path::PathBuf};

use crate::checkpoint::{
    ArtifactSource, CheckpointError, CheckpointManifest, DTypePolicy, TensorDType, TensorInventory,
    TensorOperation, accounting::IndexTotalSizeEvidence,
};

use super::*;

/// Reduced topology covering every mapping branch including MTP.
fn tiny_config() -> NemotronHConfig {
    NemotronHConfig {
        vocab_size: 32,
        hidden_size: 8,
        layers_block_type: vec![
            NemotronHBlockType::LinearAttention,
            NemotronHBlockType::FullAttention,
            NemotronHBlockType::Mlp,
            NemotronHBlockType::Moe,
        ],
        num_hidden_layers: Some(4),
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
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

/// Constructs a complete metadata-only manifest for strict plan checks.
fn manifest(contract: &NemotronHCheckpointContract) -> CheckpointManifest {
    manifest_with_dtype(contract, TensorDType::F32)
}

/// Constructs a manifest with one source dtype except for fixed-F32 buffers.
fn manifest_with_dtype(
    contract: &NemotronHCheckpointContract,
    source_dtype: TensorDType,
) -> CheckpointManifest {
    let mut offset = 0;
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
        let elements = expected.shape.iter().product::<u64>();
        let dtype = if target.ends_with("correction_bias") {
            TensorDType::F32
        } else {
            expected.dtype.unwrap_or(source_dtype)
        };
        let bytes = dtype.storage_bytes(elements).unwrap();
        tensors.insert(
            source.clone(),
            TensorInventory {
                name: source.clone(),
                shard: "synthetic.safetensors".into(),
                shape: expected.shape.clone(),
                dtype,
                data_offsets: [offset, offset + bytes],
                byte_len: bytes,
            },
        );
        offset += bytes;
    }
    CheckpointManifest {
        root: PathBuf::from("synthetic"),
        source: ArtifactSource::Single {
            relative_path: "synthetic.safetensors".into(),
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

/// Builds one physical tensor descriptor for policy-only evidence tests.
fn policy_tensor(
    name: impl Into<String>,
    dtype: TensorDType,
    shape: impl Into<Vec<u64>>,
) -> TensorInventory {
    let name = name.into();
    let shape = shape.into();
    let elements = shape.iter().product();
    let byte_len = dtype.storage_bytes(elements).unwrap();
    TensorInventory {
        name: name.clone(),
        shard: "model-00001-of-00014.safetensors".into(),
        shape,
        dtype,
        data_offsets: [0, byte_len],
        byte_len,
    }
}

/// Recreates the exact 23-base-router set plus the excluded MTP router.
fn lightning_router_inventory() -> BTreeMap<String, TensorInventory> {
    LIGHTNING_BASE_ROUTER_LAYERS
        .iter()
        .map(|layer| format!("backbone.layers.{layer}.mixer.gate.weight"))
        .chain(std::iter::once(LIGHTNING_MTP_ROUTER.into()))
        .map(|name| {
            let tensor = policy_tensor(&name, TensorDType::BF16, [128, 2_688]);
            (name, tensor)
        })
        .collect()
}

/// Returns exact official policy evidence over a caller-owned inventory.
fn lightning_evidence(tensors: &BTreeMap<String, TensorInventory>) -> IndexTotalSizeEvidence<'_> {
    IndexTotalSizeEvidence {
        index_relative_path: "model.safetensors.index.json",
        index_bytes: LIGHTNING_INDEX_BYTES,
        index_sha256: LIGHTNING_INDEX_SHA256,
        config_relative_path: "config.json",
        config_bytes: LIGHTNING_CONFIG_BYTES,
        config_sha256: LIGHTNING_CONFIG_SHA256,
        declared_total_size: LIGHTNING_DECLARED_TOTAL_SIZE,
        physical_tensor_bytes: LIGHTNING_PHYSICAL_TENSOR_BYTES,
        tensors,
        index_alias_count: 0,
    }
}

/// Requires a near miss to fail through the dedicated policy taxonomy.
fn assert_policy_mismatch(result: crate::checkpoint::Result<()>) {
    assert!(matches!(
        result,
        Err(CheckpointError::IndexTotalSizePolicyMismatch { .. })
    ));
}

/// The strict plan consumes base and all 270 MTP tensor sources exactly once.
#[test]
fn complete_hybrid_contract_builds_strict_plan() {
    let contract = tiny_config()
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    let plan = contract
        .build_load_plan(
            &manifest(&contract),
            &LoadOptions {
                dtype_policy: DTypePolicy::Preserve,
                ..LoadOptions::default()
            },
        )
        .unwrap();
    assert_eq!(plan.tensors.len(), contract.expected_tensors().len());
    assert!(plan.report.unexpected.is_empty());
    assert!(plan.report.missing.is_empty());
    assert!(contract.mappings().operations().iter().any(
        |operation| matches!(operation, TensorOperation::Map { source, target, .. }
                if source == "backbone.layers.0.mixer.A_log" && target.ends_with("mamba.a_log"))
    ));
}

/// Lightning's published 23/23/6 schedule yields its exact 6,513-key index.
///
/// The fingerprint covers the sorted tensor names from NVIDIA's checkpoint
/// index at revision `a9904d24bcc1d289a1950fa9d2b978c47cf903b9`.
/// Keeping the compact digest here detects spelling and hierarchy drift
/// without checking a multi-thousand-line generated fixture into Git.
#[test]
fn lightning_shape_contract_matches_pinned_inventory_count() {
    let mut config = tiny_config();
    config.vocab_size = 131_072;
    config.hidden_size = 2688;
    config.layers_block_type = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
        .chars()
        .map(|character| NemotronHBlockType::from_pattern(character).unwrap())
        .collect();
    config.num_hidden_layers = Some(52);
    config.num_attention_heads = 32;
    config.num_key_value_heads = 2;
    config.head_dim = 128;
    config.intermediate_size = 1856;
    config.mamba_num_heads = 64;
    config.mamba_head_dim = 64;
    config.ssm_state_size = 128;
    config.n_groups = 8;
    config.conv_kernel = 4;
    config.n_routed_experts = 128;
    config.num_experts_per_tok = 6;
    config.n_group = 1;
    config.topk_group = 1;
    config.moe_intermediate_size = 1856;
    config.moe_shared_expert_intermediate_size = 3712;
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    assert_eq!(contract.expected_tensors().len(), 6_513);
    let mut source_names = Vec::new();
    for operation in contract.mappings().operations() {
        match operation {
            TensorOperation::Map { source, .. } | TensorOperation::Split { source, .. } => {
                source_names.push(source.as_str());
            }
            TensorOperation::Concatenate { sources, .. } => {
                source_names.extend(sources.iter().map(String::as_str));
            }
            TensorOperation::DequantizeMxfp4 { blocks, scales, .. } => {
                source_names.extend([blocks.as_str(), scales.as_str()]);
            }
            TensorOperation::Alias { .. } => {}
        }
    }
    source_names.sort_unstable();
    let fingerprint =
        source_names
            .into_iter()
            .fold(14_695_981_039_346_656_037_u64, |mut fingerprint, source| {
                for byte in source.bytes().chain(std::iter::once(0)) {
                    fingerprint ^= u64::from(byte);
                    fingerprint = fingerprint.wrapping_mul(1_099_511_628_211);
                }
                fingerprint
            });
    assert_eq!(fingerprint, 0x81d1_b113_e57a_5fbb);
    let expected = |name: &str| {
        contract
            .expected_tensors()
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap()
    };
    assert_eq!(
        expected("model.layers.0.mamba.in_proj.weight").shape,
        [10_304, 2_688]
    );
    assert_eq!(
        expected("model.layers.0.mamba.conv1d.weight").shape,
        [6_144, 1, 4]
    );
    assert_eq!(
        expected("model.layers.1.moe.router.weight").shape,
        [128, 2_688]
    );
    assert_eq!(
        expected("model.layers.1.moe.correction_bias").dtype,
        Some(TensorDType::F32)
    );
    assert_eq!(
        expected("next_token_head.eh_proj.weight").shape,
        [2_688, 5_376]
    );
}

/// The pinned defect is exactly 23 base BF16 routers and excludes MTP.
#[test]
fn lightning_total_size_policy_accepts_only_the_reviewed_router_delta() {
    let tensors = lightning_router_inventory();
    let policy = lightning_total_size_policy(&NemotronHCheckpointProvenance::new(
        LIGHTNING_REPOSITORY,
        LIGHTNING_REVISION,
    ));
    policy.validate(lightning_evidence(&tensors)).unwrap();
    assert_eq!(policy.duplicated_tensors.len(), 23);
    assert_eq!(policy.excluded_tensors.len(), 1);
    assert_eq!(policy.excluded_tensors[0].name, LIGHTNING_MTP_ROUTER);
    let router_bytes = TensorDType::BF16.storage_bytes(128 * 2_688).unwrap();
    assert_eq!(router_bytes, 688_128);
    assert_eq!(
        LIGHTNING_DECLARED_TOTAL_SIZE - LIGHTNING_PHYSICAL_TENSOR_BYTES,
        router_bytes * 23
    );
}

/// Repository and immutable revision are both mandatory provenance inputs.
#[test]
fn lightning_total_size_policy_rejects_wrong_source_labels_and_other_families() {
    let tensors = lightning_router_inventory();
    for provenance in [
        NemotronHCheckpointProvenance::new("nvidia/wrong", LIGHTNING_REVISION),
        NemotronHCheckpointProvenance::new(LIGHTNING_REPOSITORY, "wrong-revision"),
        NemotronHCheckpointProvenance::new("Qwen/Qwen3-30B-A3B", LIGHTNING_REVISION),
    ] {
        assert_policy_mismatch(
            lightning_total_size_policy(&provenance).validate(lightning_evidence(&tensors)),
        );
    }
}

/// Both raw control files require exact paths, byte lengths, and SHA-256 hashes.
#[test]
fn lightning_total_size_policy_rejects_every_control_identity_near_miss() {
    let tensors = lightning_router_inventory();
    let policy = lightning_total_size_policy(&NemotronHCheckpointProvenance::new(
        LIGHTNING_REPOSITORY,
        LIGHTNING_REVISION,
    ));
    let exact = lightning_evidence(&tensors);

    let mut wrong = exact;
    wrong.config_relative_path = "nested/config.json";
    assert_policy_mismatch(policy.validate(wrong));
    let mut wrong = exact;
    wrong.config_bytes -= 1;
    assert_policy_mismatch(policy.validate(wrong));
    let mut wrong = exact;
    wrong.config_sha256 = "03827a0f5e311547b40943dc081e3ff2f8a277466e8c1a3df2291e8db8a7617c";
    assert_policy_mismatch(policy.validate(wrong));
    let mut wrong = exact;
    wrong.index_relative_path = "weights.safetensors.index.json";
    assert_policy_mismatch(policy.validate(wrong));
    let mut wrong = exact;
    wrong.index_bytes += 1;
    assert_policy_mismatch(policy.validate(wrong));
    let mut wrong = exact;
    wrong.index_sha256 = "07f21da80ce245a3e24967f54eef3fa10e67a63eba16ef19a2d0569f06103f50";
    assert_policy_mismatch(policy.validate(wrong));
}

/// Missing, renamed, and extra base routers cannot preserve policy validity.
#[test]
fn lightning_total_size_policy_rejects_router_name_and_count_drift() {
    let policy = lightning_total_size_policy(&NemotronHCheckpointProvenance::new(
        LIGHTNING_REPOSITORY,
        LIGHTNING_REVISION,
    ));
    let first = "backbone.layers.1.mixer.gate.weight";

    let mut missing = lightning_router_inventory();
    missing.remove(first);
    assert_policy_mismatch(policy.validate(lightning_evidence(&missing)));

    let mut renamed = lightning_router_inventory();
    let tensor = renamed.remove(first).unwrap();
    let wrong = "backbone.layers.2.mixer.gate.weight";
    renamed.insert(
        wrong.into(),
        TensorInventory {
            name: wrong.into(),
            ..tensor
        },
    );
    assert_policy_mismatch(policy.validate(lightning_evidence(&renamed)));

    let mut extra = lightning_router_inventory();
    let wrong = "backbone.layers.52.mixer.gate.weight";
    extra.insert(
        wrong.into(),
        policy_tensor(wrong, TensorDType::BF16, [128, 2_688]),
    );
    assert_policy_mismatch(policy.validate(lightning_evidence(&extra)));

    let mut duplicate_policy = policy.clone();
    duplicate_policy
        .duplicated_tensors
        .push(duplicate_policy.duplicated_tensors[0].clone());
    assert_policy_mismatch(
        duplicate_policy.validate(lightning_evidence(&lightning_router_inventory())),
    );
}

/// Router descriptor changes fail even when their byte width happens to match.
#[test]
fn lightning_total_size_policy_rejects_dtype_shape_and_byte_length_drift() {
    let policy = lightning_total_size_policy(&NemotronHCheckpointProvenance::new(
        LIGHTNING_REPOSITORY,
        LIGHTNING_REVISION,
    ));
    let first = "backbone.layers.1.mixer.gate.weight";

    let mut wrong_dtype = lightning_router_inventory();
    wrong_dtype.get_mut(first).unwrap().dtype = TensorDType::F16;
    assert_policy_mismatch(policy.validate(lightning_evidence(&wrong_dtype)));

    let mut wrong_shape = lightning_router_inventory();
    wrong_shape.get_mut(first).unwrap().shape = vec![64, 5_376];
    assert_policy_mismatch(policy.validate(lightning_evidence(&wrong_shape)));

    let mut wrong_bytes = lightning_router_inventory();
    wrong_bytes.get_mut(first).unwrap().byte_len += 1;
    assert_policy_mismatch(policy.validate(lightning_evidence(&wrong_bytes)));
}

/// Totals, aliases, and the non-accounted MTP router are independent gates.
#[test]
fn lightning_total_size_policy_rejects_delta_alias_and_mtp_drift() {
    let tensors = lightning_router_inventory();
    let policy = lightning_total_size_policy(&NemotronHCheckpointProvenance::new(
        LIGHTNING_REPOSITORY,
        LIGHTNING_REVISION,
    ));
    let exact = lightning_evidence(&tensors);

    let mut wrong = exact;
    wrong.declared_total_size += 1;
    assert_policy_mismatch(policy.validate(wrong));
    let mut wrong = exact;
    wrong.physical_tensor_bytes -= 1;
    assert_policy_mismatch(policy.validate(wrong));
    let mut wrong = exact;
    wrong.index_alias_count = 1;
    assert_policy_mismatch(policy.validate(wrong));

    let mut missing_mtp = tensors.clone();
    missing_mtp.remove(LIGHTNING_MTP_ROUTER);
    assert_policy_mismatch(policy.validate(lightning_evidence(&missing_mtp)));

    let mut counted_mtp = policy.clone();
    counted_mtp
        .duplicated_tensors
        .push(counted_mtp.excluded_tensors[0].clone());
    assert_policy_mismatch(counted_mtp.validate(exact));

    let mut mtp_sized_delta = exact;
    mtp_sized_delta.declared_total_size += TensorDType::BF16.storage_bytes(128 * 2_688).unwrap();
    assert_policy_mismatch(policy.validate(mtp_sized_delta));
}

/// A Metal policy casts generic weights while preserving router buffers in FP32.
#[test]
fn bf16_plan_preserves_fixed_fp32_router_buffers() {
    let contract = tiny_config()
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    let plan = contract
        .build_load_plan(
            &manifest_with_dtype(&contract, TensorDType::BF16),
            &LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
        )
        .unwrap();
    for tensor in &plan.tensors {
        let expected = if tensor.target.ends_with("correction_bias") {
            TensorDType::F32
        } else {
            TensorDType::F16
        };
        assert_eq!(tensor.dtype, expected, "{}", tensor.target);
    }
    assert_eq!(
        plan.report
            .dtype_converted
            .iter()
            .filter(|conversion| conversion.target_dtype == TensorDType::F16)
            .count(),
        plan.tensors
            .iter()
            .filter(|tensor| !tensor.target.ends_with("correction_bias"))
            .count()
    );
}

/// Unsupported multi-stage MTP fails before constructing ambiguous mappings.
#[test]
fn unsupported_mtp_schedule_is_rejected() {
    let mut config = tiny_config();
    config.num_nextn_predict_layers = 2;
    assert!(
        config
            .checkpoint_contract(NemotronHCheckpointOptions::default())
            .is_err()
    );
}

/// An unrepresentable MTP fusion width `2 h` is a configuration error, not an overflow.
///
/// Validation rejects it first; the contract's checked width is a second guard.
#[test]
fn overflowing_mtp_fusion_width_is_rejected() {
    let mut config = tiny_config();
    config.hidden_size = usize::MAX / 2 + 1;
    let error = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap_err();
    assert_eq!(error.field, "hidden_size");
    config.num_nextn_predict_layers = 0;
    assert!(
        config
            .checkpoint_contract(NemotronHCheckpointOptions::default())
            .is_ok()
    );
}
