//! Test-only fixtures for checkpoint.rs; excluded from non-test compiler inputs.

use std::{fs::File, io::Write, path::Path};

use serde_json::{Map, Value, json};

use super::super::SymmetricQuantization;
use super::*;
use crate::checkpoint::{
    ExpectedTensor, LoadOptions, MappingPlan, build_load_plan, discover_checkpoint,
};

/// Minimal input tensor for a standards-compliant SafeTensors fixture.
struct FixtureTensor {
    /// Name stored in the fixture's header and expected load inventory.
    name: &'static str,
    /// Logical F32 tensor dimensions.
    shape: Vec<u64>,
    /// Row-major values serialized in little-endian order.
    values: Vec<f32>,
}

/// Writes one deterministic unsharded F32 SafeTensors artifact.
fn write_checkpoint(root: &Path, tensors: &[FixtureTensor]) {
    let mut offset = 0_u64;
    let mut header = Map::new();
    for tensor in tensors {
        let bytes = u64::try_from(tensor.values.len() * size_of::<f32>()).unwrap();
        let end = offset + bytes;
        header.insert(
            tensor.name.into(),
            json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [offset, end],
            }),
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
    for tensor in tensors {
        for value in &tensor.values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }
}

/// Builds a strict plan including a renamed weight, exclusions, and an alias.
fn fixture_plan(root: &Path, tensors: &[FixtureTensor]) -> (CheckpointManifest, LoadPlan) {
    write_checkpoint(root, tensors);
    let manifest = discover_checkpoint(root, &LoadOptions::default()).unwrap();
    let expected = [
        ExpectedTensor::new("model.embed_tokens.weight", [4, 8]),
        ExpectedTensor::new("model.layers.0.input_layernorm.weight", [8]),
        ExpectedTensor::new("model.layers.0.self_attn.q_proj.weight", [4, 8]),
        ExpectedTensor::new("lm_head.weight", [4, 8]),
    ];
    let mut mappings = MappingPlan::new();
    mappings
        .map("external.embedding", "model.embed_tokens.weight")
        .map("external.norm", "model.layers.0.input_layernorm.weight")
        .map(
            "external.projection",
            "model.layers.0.self_attn.q_proj.weight",
        )
        .alias("model.embed_tokens.weight", "lm_head.weight");
    let plan = build_load_plan(&manifest, &expected, &mappings, &LoadOptions::default()).unwrap();
    (manifest, plan)
}

/// Architecture mappings run before policy and every outcome is auditable.
#[test]
fn local_plan_streams_quantized_targets_and_reports_exclusions() {
    let root = tempfile::tempdir().unwrap();
    let tensors = [
        FixtureTensor {
            name: "external.embedding",
            shape: vec![4, 8],
            values: vec![0.25; 32],
        },
        FixtureTensor {
            name: "external.norm",
            shape: vec![8],
            values: vec![1.0; 8],
        },
        FixtureTensor {
            name: "external.projection",
            shape: vec![4, 8],
            values: (0..32).map(|index| index as f32 - 15.5).collect(),
        },
    ];
    let (manifest, plan) = fixture_plan(root.path(), &tensors);
    let output = root.path().join("weights-q4.bpk");
    let options = QuantizedCheckpointOptions {
        weights: WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q4,
            ..WeightQuantizationConfig::default()
        },
        ..QuantizedCheckpointOptions::default()
    };

    let outcome = quantize_local_checkpoint_to_burnpack(&manifest, &plan, &output, options)
        .expect("valid conversion");
    assert_eq!(outcome.artifact.tensor_count, 1);
    assert_eq!(outcome.report.quantized.len(), 1);
    assert_eq!(
        outcome.report.quantized[0].target,
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        outcome.report.quantized[0].scheme,
        "symmetric-q4-g64-flat-row-major"
    );
    assert_eq!(outcome.report.applied.len(), 4);
    assert_eq!(outcome.report.aliased.len(), 1);
    assert_eq!(
        outcome.excluded,
        [
            QuantizationExclusionRecord {
                target: "model.embed_tokens.weight".into(),
                reason: QuantizationExclusion::Embedding,
            },
            QuantizationExclusionRecord {
                target: "model.layers.0.input_layernorm.weight".into(),
                reason: QuantizationExclusion::Normalization,
            },
        ]
    );
    assert_eq!(outcome.execution.source.tensor_reads, 3);
    assert_eq!(outcome.staging.tensor_count, 1);
    assert!(outcome.staging.estimated_manifest_bytes >= outcome.artifact.manifest_bytes as u64);
    let pre_hash = outcome.provenance.pre_execution.unwrap();
    let post_hash = outcome.provenance.post_execution.unwrap();
    for stats in [pre_hash, post_hash] {
        assert_eq!(stats.passes_completed, 1);
        assert_eq!(stats.shards_hashed, 1);
        assert_eq!(stats.bytes_hashed, manifest.artifact_bytes);
        assert_eq!(
            stats.peak_buffer_bytes,
            crate::checkpoint::quantization::PROVENANCE_HASH_BUFFER_BYTES
        );
    }
    assert!(output.is_file());

    let mut artifact = super::super::QuantizedBurnpackArtifact::open(&output).unwrap();
    assert!(
        artifact
            .manifest()
            .tensors
            .values()
            .all(|metadata| metadata.payload_sha256.is_some())
    );
    artifact.validate_all().unwrap();
    assert_eq!(
        artifact
            .read_weight("model.layers.0.self_attn.q_proj.weight")
            .unwrap()
            .metadata()
            .shape,
        [4, 8]
    );
}

/// Numerical failure preserves partial diagnostics and publishes no artifact.
#[test]
fn conversion_failure_is_structured_and_non_publishing() {
    let root = tempfile::tempdir().unwrap();
    let tensors = [
        FixtureTensor {
            name: "a.weight",
            shape: vec![4, 8],
            values: vec![1.0; 32],
        },
        FixtureTensor {
            name: "z.weight",
            shape: vec![4, 8],
            values: vec![f32::NAN; 32],
        },
    ];
    write_checkpoint(root.path(), &tensors);
    let manifest = discover_checkpoint(root.path(), &LoadOptions::default()).unwrap();
    let expected = [
        ExpectedTensor::new("a.weight", [4, 8]),
        ExpectedTensor::new("z.weight", [4, 8]),
    ];
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap();
    let output = root.path().join("failed.bpk");

    let failure = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions::default(),
    )
    .unwrap_err();
    assert_eq!(failure.failed_target.as_deref(), Some("z.weight"));
    assert!(matches!(
        failure.cause,
        CheckpointQuantizationFailureCause::Quantization(QuantizationError::NonFiniteValue { .. })
    ));
    assert_eq!(failure.report.applied, ["a.weight"]);
    assert_eq!(failure.report.quantized.len(), 1);
    assert_eq!(failure.staging.tensor_count, 1);
    let pre_hash = failure.provenance.pre_execution.unwrap();
    assert_eq!(pre_hash.passes_completed, 1);
    assert_eq!(pre_hash.shards_hashed, 1);
    assert_eq!(pre_hash.bytes_hashed, manifest.artifact_bytes);
    assert_eq!(failure.provenance.post_execution, None);
    assert!(!output.exists());
}

/// Existing outputs fail before the first source tensor is read or replaced.
#[test]
fn existing_output_is_preserved_before_checkpoint_io() {
    let root = tempfile::tempdir().unwrap();
    let tensors = [
        FixtureTensor {
            name: "external.embedding",
            shape: vec![4, 8],
            values: vec![0.25; 32],
        },
        FixtureTensor {
            name: "external.norm",
            shape: vec![8],
            values: vec![1.0; 8],
        },
        FixtureTensor {
            name: "external.projection",
            shape: vec![4, 8],
            values: vec![2.0; 32],
        },
    ];
    let (manifest, plan) = fixture_plan(root.path(), &tensors);
    let output = root.path().join("occupied.bpk");
    std::fs::write(&output, b"original").unwrap();

    let failure = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        failure.cause,
        CheckpointQuantizationFailureCause::Quantization(
            QuantizationError::OutputAlreadyExists { .. }
        )
    ));
    assert_eq!(failure.execution.source.tensor_reads, 0);
    assert_eq!(failure.provenance, Default::default());
    assert_eq!(std::fs::read(&output).unwrap(), b"original");
}

/// Creates a strict identity plan for arbitrary small conversion targets.
fn identity_fixture(root: &Path, names: &[&'static str]) -> (CheckpointManifest, LoadPlan) {
    let tensors = names
        .iter()
        .map(|&name| FixtureTensor {
            name,
            shape: vec![4, 8],
            values: vec![1.0; 32],
        })
        .collect::<Vec<_>>();
    write_checkpoint(root, &tensors);
    let manifest = discover_checkpoint(root, &LoadOptions::default()).unwrap();
    let expected = names
        .iter()
        .map(|&name| ExpectedTensor::new(name, [4, 8]))
        .collect::<Vec<_>>();
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap();
    (manifest, plan)
}

/// Seeds stale runtime categories to ensure failed conversions do not inherit them.
fn stale_report(plan: &mut LoadPlan) {
    plan.report.applied.push("old".into());
    plan.report.quantized.push(QuantizationRecord {
        target: "old".into(),
        scheme: "old".into(),
        bytes: 1,
    });
    plan.report
        .equivalence_verified
        .push(crate::checkpoint::CheckpointEquivalenceRecord::new(
            "old", "owner",
        ));
}

/// Setup errors avoid hashing and preserve the source error's inspectable chain.
#[test]
fn invalid_conversion_limits_reset_stale_progress_before_checkpoint_access() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, mut plan) = identity_fixture(root.path(), &["weight"]);
    stale_report(&mut plan);
    std::fs::remove_file(&manifest.shards[0].path).unwrap();
    let output = root.path().join("invalid-limit.bpk");
    let failure = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions {
            limits: BurnpackSessionLimits {
                max_tensors: 0,
                ..BurnpackSessionLimits::default()
            },
            ..QuantizedCheckpointOptions::default()
        },
    )
    .unwrap_err();
    assert!(failure.failed_target.is_none());
    assert_eq!(failure.provenance, ProvenanceExecutionStats::default());
    assert_eq!(failure.execution, ExecutionStats::default());
    assert!(failure.report.applied.is_empty());
    assert!(failure.report.quantized.is_empty());
    assert!(failure.report.equivalence_verified.is_empty());
    assert!(
        failure
            .to_string()
            .starts_with("checkpoint quantization failed for")
    );
    let cause = StdError::source(failure.as_ref()).unwrap();
    assert!(
        cause
            .source()
            .unwrap()
            .downcast_ref::<QuantizationError>()
            .is_some()
    );
    assert!(matches!(
        failure.cause,
        CheckpointQuantizationFailureCause::Quantization(QuantizationError::InvalidSessionLimit {
            field: "max_tensors",
            ..
        })
    ));
    assert!(!output.exists());
}

/// A failed initial hash reports its completed prefix without staging a tensor.
#[test]
fn initial_provenance_failure_records_partial_hashing_without_publication() {
    let root = tempfile::tempdir().unwrap();
    let (mut manifest, plan) = identity_fixture(root.path(), &["weight"]);
    let mut absent = manifest.shards[0].clone();
    absent.relative_path = "missing.safetensors".into();
    absent.path = manifest.root.join(&absent.relative_path);
    manifest.shards.push(absent);
    let output = root.path().join("missing-source.bpk");
    let failure = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        failure.cause,
        CheckpointQuantizationFailureCause::Quantization(QuantizationError::ArtifactIo {
            operation: "canonicalize checkpoint shard for provenance",
            ..
        })
    ));
    let pre = failure.provenance.pre_execution.unwrap();
    assert_eq!(pre.shards_hashed, 1);
    assert_eq!(pre.bytes_hashed, manifest.shards[0].file_bytes);
    assert_eq!(pre.passes_completed, 0);
    assert_eq!(failure.provenance.post_execution, None);
    assert_eq!(failure.execution, ExecutionStats::default());
    assert_eq!(failure.staging.tensor_count, 0);
    assert!(failure.report.applied.is_empty());
    assert!(!output.exists());
}

/// Invalid public plans remain checkpoint errors after the initial source hash.
#[test]
fn malformed_conversion_plan_never_reaches_the_quantization_sink() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, mut plan) = identity_fixture(root.path(), &["weight"]);
    plan.tensors[0].sources[0].data_offsets[1] -= 1;
    let output = root.path().join("bad-plan.bpk");
    let failure = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions::default(),
    )
    .unwrap_err();
    assert!(
        matches!(failure.cause, CheckpointQuantizationFailureCause::Checkpoint(
        CheckpointError::InvalidMapping { ref message }) if message.contains("offsets describe"))
    );
    assert!(failure.failed_target.is_none());
    assert!(failure.report.applied.is_empty());
    assert!(failure.report.quantized.is_empty());
    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.staging.tensor_count, 0);
    assert_eq!(
        failure.provenance.pre_execution.unwrap().bytes_hashed,
        manifest.artifact_bytes
    );
    assert_eq!(failure.provenance.post_execution, None);
    let cause = StdError::source(failure.as_ref()).unwrap();
    assert!(
        cause
            .source()
            .unwrap()
            .downcast_ref::<CheckpointError>()
            .is_some()
    );
    assert!(!output.exists());
}

/// A late checkpoint failure retains quantized and excluded successful targets.
#[test]
fn checkpoint_failure_keeps_exact_partial_quantized_and_exclusion_reports() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, mut plan) = identity_fixture(
        root.path(),
        &["a.weight", "b.embed_tokens.weight", "z.weight"],
    );
    plan.tensors
        .iter_mut()
        .find(|tensor| tensor.target == "z.weight")
        .unwrap()
        .sources[0]
        .name = "missing".into();
    let output = root.path().join("partial.bpk");
    let failure = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions::default(),
    )
    .unwrap_err();
    assert!(
        matches!(failure.cause, CheckpointQuantizationFailureCause::Checkpoint(
        CheckpointError::PlanManifestMismatch { ref tensor, .. }) if tensor == "missing")
    );
    assert_eq!(failure.failed_target.as_deref(), Some("z.weight"));
    assert_eq!(
        failure.report.applied,
        ["a.weight", "b.embed_tokens.weight"]
    );
    assert_eq!(failure.report.quantized.len(), 1);
    assert_eq!(failure.report.quantized[0].target, "a.weight");
    assert_eq!(
        failure.excluded,
        [QuantizationExclusionRecord {
            target: "b.embed_tokens.weight".into(),
            reason: QuantizationExclusion::Embedding
        }]
    );
    assert_eq!(failure.execution.source_reads, 2);
    assert_eq!(failure.execution.tensors_applied, 2);
    assert_eq!(failure.staging.tensor_count, 1);
    assert_eq!(
        failure.staging.staged_payload_bytes,
        failure.report.quantized[0].bytes
    );
    assert_eq!(failure.provenance.post_execution, None);
    assert!(failure.to_string().contains("failed at `z.weight`"));
    assert!(failure.cause.to_string().contains("missing"));
    assert!(!output.exists());
}

/// A manifest limit the planned sidecar cannot meet fails before hashing or reading.
#[test]
fn infeasible_manifest_limit_fails_before_any_source_read() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, plan) = identity_fixture(root.path(), &["embed_tokens.weight"]);
    let output = root.path().join("bound-limit.bpk");
    let failure = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions {
            limits: BurnpackSessionLimits {
                max_manifest_bytes: 1,
                ..BurnpackSessionLimits::default()
            },
            ..QuantizedCheckpointOptions::default()
        },
    )
    .unwrap_err();
    assert!(matches!(
        failure.cause,
        CheckpointQuantizationFailureCause::Quantization(QuantizationError::SessionLimitExceeded {
            kind: "estimated manifest",
            limit: 1,
            ..
        })
    ));
    assert!(failure.failed_target.is_none());
    assert!(failure.report.applied.is_empty());
    assert!(failure.excluded.is_empty());
    assert_eq!(failure.execution.source_reads, 0);
    assert_eq!(failure.staging.tensor_count, 0);
    assert!(failure.provenance.pre_execution.is_none());
    assert!(failure.provenance.post_execution.is_none());
    assert!(!output.exists());
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
}

/// Staged-payload and tensor-count limits are also checked against the whole plan.
#[test]
fn planned_payload_and_count_limits_fail_before_any_source_read() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, plan) = identity_fixture(root.path(), &["a.weight", "b.weight"]);
    for (kind, limits) in [
        (
            "staged payload",
            BurnpackSessionLimits {
                max_staged_bytes: 1,
                ..BurnpackSessionLimits::default()
            },
        ),
        (
            "tensor count",
            BurnpackSessionLimits {
                max_tensors: 1,
                ..BurnpackSessionLimits::default()
            },
        ),
    ] {
        let failure = quantize_local_checkpoint_to_burnpack(
            &manifest,
            &plan,
            root.path().join(format!("{kind}.bpk")),
            QuantizedCheckpointOptions {
                limits,
                ..QuantizedCheckpointOptions::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(&failure.cause,
            CheckpointQuantizationFailureCause::Quantization(QuantizationError::SessionLimitExceeded { kind: actual, .. })
            if *actual == kind),
            "{kind}: {}",
            failure.cause
        );
        assert_eq!(failure.execution.source_reads, 0);
        assert!(failure.provenance.pre_execution.is_none());
    }
}

/// Change diagnostics distinguish removed, added, modified, and plan-only state.
#[test]
fn changed_shard_diagnostic_identifies_membership_digest_and_plan_changes() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, plan) = identity_fixture(root.path(), &["weight"]);
    let (before, _) = super::super::hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
    for case in ["remove", "add", "digest", "plan"] {
        let mut after = before.clone();
        match case {
            "remove" => after.checkpoint_shards.clear(),
            "add" => {
                after.checkpoint_shards.insert(
                    "new.safetensors".into(),
                    before.checkpoint_shards["model.safetensors"].clone(),
                );
            }
            "digest" => {
                after
                    .checkpoint_shards
                    .get_mut("model.safetensors")
                    .unwrap()
                    .sha256 = "0".repeat(64)
            }
            "plan" => after.load_plan_sha256 = "0".repeat(64),
            _ => unreachable!(),
        }
        assert_eq!(
            first_changed_shard(&before, &after),
            match case {
                "add" => "new.safetensors",
                "plan" => "<load-plan>",
                _ => "model.safetensors",
            }
        );
    }
}

/// A publication-stage limit preserves both completed hash passes and every staged target.
#[test]
fn checkpoint_standards_finalize_limit_keeps_complete_audit_without_output() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, plan) = identity_fixture(root.path(), &["a.weight", "b.embed_tokens.weight"]);
    let output = root.path().join("manifest-does-not-fit.bpk");
    // The 32 Q8 values and one F32 scale fit exactly; the final JSON cannot fit.
    let payload_bytes = 32 + 4;
    let failure = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions {
            limits: BurnpackSessionLimits {
                max_staged_bytes: payload_bytes,
                ..BurnpackSessionLimits::default()
            },
            ..QuantizedCheckpointOptions::default()
        },
    )
    .unwrap_err();
    assert!(matches!(
        failure.cause,
        CheckpointQuantizationFailureCause::Quantization(QuantizationError::SessionLimitExceeded {
            kind: "staged payload", attempted, limit
        }) if attempted > payload_bytes && limit == payload_bytes
    ));
    assert!(failure.failed_target.is_none());
    assert_eq!(
        failure.report.applied,
        ["a.weight", "b.embed_tokens.weight"]
    );
    assert_eq!(failure.report.quantized.len(), 1);
    assert_eq!(failure.report.quantized[0].target, "a.weight");
    assert_eq!(failure.report.quantized[0].bytes, payload_bytes);
    assert_eq!(failure.excluded.len(), 1);
    assert_eq!(failure.excluded[0].reason, QuantizationExclusion::Embedding);
    assert_eq!(failure.execution.tensors_applied, 2);
    assert_eq!(failure.staging.tensor_count, 1);
    assert_eq!(failure.staging.staged_payload_bytes, payload_bytes);
    for stats in [
        failure.provenance.pre_execution.unwrap(),
        failure.provenance.post_execution.unwrap(),
    ] {
        assert_eq!(stats.passes_completed, 1);
        assert_eq!(stats.shards_hashed, 1);
        assert_eq!(stats.bytes_hashed, manifest.artifact_bytes);
    }
    assert!(!output.exists());
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
}

/// Writes one aligned and one group-unaligned projection with an identity plan.
fn mixed_alignment_plan(root: &Path) -> (CheckpointManifest, LoadPlan) {
    let tensors = [
        FixtureTensor {
            name: "model.layers.0.self_attn.k_proj.weight",
            shape: vec![2, 64],
            values: (0..128).map(|index| index as f32 / 8.0 - 8.0).collect(),
        },
        FixtureTensor {
            name: "model.layers.0.self_attn.q_proj.weight",
            shape: vec![4, 8],
            values: (0..32).map(|index| index as f32 - 15.5).collect(),
        },
    ];
    write_checkpoint(root, &tensors);
    let manifest = discover_checkpoint(root, &LoadOptions::default()).unwrap();
    let expected = tensors
        .iter()
        .map(|tensor| ExpectedTensor::new(tensor.name, tensor.shape.clone()))
        .collect::<Vec<_>>();
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap();
    (manifest, plan)
}

/// Group-unaligned matrices follow the configured policy; the summary reports packability.
#[test]
fn unaligned_matrices_follow_the_configured_policy() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, plan) = mixed_alignment_plan(root.path());
    let unaligned = "model.layers.0.self_attn.q_proj.weight";
    let options = |unaligned| QuantizedCheckpointOptions {
        unaligned,
        ..QuantizedCheckpointOptions::default()
    };

    let reference = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        root.path().join("reference.bpk"),
        options(UnalignedMatrixPolicy::ReferenceOnly),
    )
    .unwrap();
    assert_eq!(reference.artifact.tensor_count, 2);
    assert_eq!(
        reference.artifact.packed_execution,
        super::super::PackedExecutionSupport::Unavailable
    );

    let excluded = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        root.path().join("excluded.bpk"),
        options(UnalignedMatrixPolicy::Exclude),
    )
    .unwrap();
    assert_eq!(excluded.artifact.tensor_count, 1);
    assert_eq!(
        excluded.artifact.packed_execution,
        super::super::PackedExecutionSupport::BurnQFloatMatmul
    );
    assert_eq!(
        excluded.excluded,
        [QuantizationExclusionRecord {
            target: unaligned.into(),
            reason: QuantizationExclusion::GroupUnaligned,
        }]
    );
    assert_eq!(excluded.report.applied.len(), 2);

    let output = root.path().join("rejected.bpk");
    let failure = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        options(UnalignedMatrixPolicy::Reject),
    )
    .unwrap_err();
    assert_eq!(failure.failed_target.as_deref(), Some(unaligned));
    assert!(matches!(
        &failure.cause,
        CheckpointQuantizationFailureCause::Quantization(QuantizationError::BurnQFloatIncompatible { reason })
        if reason.contains("group size 64")
    ));
    assert_eq!(failure.execution.source_reads, 0);
    assert!(failure.provenance.pre_execution.is_none());
    assert!(!output.exists());
}

/// Flips one payload bit of the only shard while keeping its length and header.
fn flip_last_payload_bit(manifest: &CheckpointManifest) {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&manifest.shards[0].path)
        .unwrap();
    let mut byte = [0_u8; 1];
    file.seek(SeekFrom::End(-1)).unwrap();
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::End(-1)).unwrap();
    file.write_all(&[byte[0] ^ 1]).unwrap();
}

/// Asserts the complete staged-but-unpublished state shared by late failures.
fn assert_unpublished_after_staging(
    failure: &CheckpointQuantizationFailure,
    manifest: &CheckpointManifest,
    output: &Path,
) {
    assert!(failure.failed_target.is_none());
    assert_eq!(failure.report.applied, ["weight"]);
    assert_eq!(failure.report.quantized.len(), 1);
    assert_eq!(failure.execution.tensors_applied, 1);
    assert_eq!(failure.staging.tensor_count, 1);
    let pre = failure.provenance.pre_execution.unwrap();
    assert_eq!(pre.passes_completed, 1);
    assert_eq!(pre.bytes_hashed, manifest.artifact_bytes);
    assert!(!output.exists());
}

/// A checkpoint edited during execution is detected and blocks publication.
///
/// The edit keeps the shard's length and header, so only the second complete
/// hash pass can observe it; the error names the changed shard.
#[test]
fn checkpoint_change_between_hash_passes_blocks_publication() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, plan) = identity_fixture(root.path(), &["weight"]);
    let output = root.path().join("changed.bpk");
    let failure = quantize_with_post_execution_hook(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions::default(),
        || flip_last_payload_bit(&manifest),
    )
    .unwrap_err();
    assert!(matches!(
        &failure.cause,
        CheckpointQuantizationFailureCause::Quantization(
            QuantizationError::CheckpointChangedDuringQuantization { shard }
        ) if shard == "model.safetensors"
    ));
    assert_eq!(
        failure.to_string(),
        format!(
            "checkpoint quantization failed for `{}`: checkpoint changed while quantizing shard `model.safetensors`",
            output.display()
        )
    );
    assert_unpublished_after_staging(&failure, &manifest, &output);
    let post = failure.provenance.post_execution.unwrap();
    assert_eq!(post.passes_completed, 1);
    assert_eq!(post.bytes_hashed, manifest.artifact_bytes);
    // Only the checkpoint remains: the private staging file was removed.
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
}

/// A shard that disappears during execution fails the post-execution hash pass.
#[test]
fn post_execution_hash_failure_records_its_partial_pass() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, plan) = identity_fixture(root.path(), &["weight"]);
    let output = root.path().join("vanished.bpk");
    let failure = quantize_with_post_execution_hook(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions::default(),
        || std::fs::remove_file(&manifest.shards[0].path).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(
        failure.cause,
        CheckpointQuantizationFailureCause::Quantization(QuantizationError::ArtifactIo {
            operation: "canonicalize checkpoint shard for provenance",
            ..
        })
    ));
    assert_unpublished_after_staging(&failure, &manifest, &output);
    let post = failure.provenance.post_execution.unwrap();
    assert_eq!(post, super::super::ProvenanceHashStats::default());
}

/// Staging counters of the one-tensor identity conversion around provenance binding.
///
/// Returns `(unbound, bound)`. `unbound` comes from a conversion that a
/// checkpoint edit stops after execution but before provenance binding;
/// `bound` comes from a conversion that binds provenance and publishes. Each
/// runs on its own fresh fixture, so neither touches the caller's directory.
fn one_tensor_reference_staging() -> (BurnpackStagingStats, BurnpackStagingStats) {
    let unbound_root = tempfile::tempdir().unwrap();
    let (manifest, plan) = identity_fixture(unbound_root.path(), &["weight"]);
    let unbound = quantize_with_post_execution_hook(
        &manifest,
        &plan,
        &unbound_root.path().join("changed.bpk"),
        QuantizedCheckpointOptions::default(),
        || flip_last_payload_bit(&manifest),
    )
    .unwrap_err()
    .staging;

    let bound_root = tempfile::tempdir().unwrap();
    let (manifest, plan) = identity_fixture(bound_root.path(), &["weight"]);
    let bound = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        bound_root.path().join("published.bpk"),
        QuantizedCheckpointOptions::default(),
    )
    .unwrap()
    .staging;
    (unbound, bound)
}

/// Publication stays create-only when the output appears during execution.
///
/// Both hash passes agree and provenance binds, but the final create-only
/// publication refuses to replace the path and leaves its bytes untouched.
#[test]
fn output_created_during_execution_is_never_replaced() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, plan) = identity_fixture(root.path(), &["weight"]);
    let output = root.path().join("claimed.bpk");
    let failure = quantize_with_post_execution_hook(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions::default(),
        || std::fs::write(&output, b"someone else's artifact").unwrap(),
    )
    .unwrap_err();
    let canonical = std::fs::canonicalize(root.path())
        .unwrap()
        .join("claimed.bpk");
    assert!(matches!(
        &failure.cause,
        CheckpointQuantizationFailureCause::Quantization(QuantizationError::OutputAlreadyExists { path })
            if *path == canonical
    ));
    assert!(failure.failed_target.is_none());
    assert_eq!(failure.report.applied, ["weight"]);
    // Publication fails after provenance binding, so the reported staging
    // counters are the bound ones: equal to a published conversion's, and
    // with a larger manifest estimate than a conversion stopped before binding.
    let (unbound, bound) = one_tensor_reference_staging();
    assert_eq!(failure.staging, bound);
    assert_eq!(failure.staging.tensor_count, 1);
    assert_eq!(
        failure.staging.staged_payload_bytes,
        unbound.staged_payload_bytes
    );
    assert!(
        failure.staging.estimated_manifest_bytes > unbound.estimated_manifest_bytes,
        "{:?} vs {unbound:?}",
        failure.staging
    );
    for stats in [
        failure.provenance.pre_execution.unwrap(),
        failure.provenance.post_execution.unwrap(),
    ] {
        assert_eq!(stats.passes_completed, 1);
        assert_eq!(stats.bytes_hashed, manifest.artifact_bytes);
    }
    assert_eq!(std::fs::read(&output).unwrap(), b"someone else's artifact");
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 2);
}

/// The quantizer records each post-transform float dtype as its source dtype.
///
/// The F32 checkpoint is cast by the load plan, so the sidecar must describe
/// the F16, BF16, or F64 tensor the quantizer actually received.
#[test]
fn post_transform_float_dtypes_are_recorded_as_quantizer_inputs() {
    for (dtype, recorded) in [
        (TensorDType::F16, SourceFloatDType::F16),
        (TensorDType::BF16, SourceFloatDType::Bf16),
        (TensorDType::F32, SourceFloatDType::F32),
        (TensorDType::F64, SourceFloatDType::F64),
    ] {
        let root = tempfile::tempdir().unwrap();
        write_checkpoint(
            root.path(),
            &[FixtureTensor {
                name: "weight",
                shape: vec![4, 64],
                values: (0..256).map(|index| index as f32 / 64.0 - 2.0).collect(),
            }],
        );
        let options = LoadOptions {
            dtype_policy: crate::checkpoint::DTypePolicy::Cast(dtype),
            ..LoadOptions::default()
        };
        let manifest = discover_checkpoint(root.path(), &options).unwrap();
        let expected = [ExpectedTensor::new("weight", [4, 64])];
        let plan = build_load_plan(
            &manifest,
            &expected,
            &MappingPlan::identity(&expected),
            &options,
        )
        .unwrap();
        assert_eq!(plan.tensors[0].dtype, dtype);
        let output = root.path().join("cast.bpk");
        let outcome = quantize_local_checkpoint_to_burnpack(
            &manifest,
            &plan,
            &output,
            QuantizedCheckpointOptions::default(),
        )
        .unwrap();
        assert_eq!(outcome.artifact.tensor_count, 1);
        let artifact = super::super::QuantizedBurnpackArtifact::open(&output).unwrap();
        let metadata = &artifact.manifest().tensors["weight"];
        assert_eq!(metadata.source_dtype, recorded, "{dtype}");
        assert_eq!(metadata.shape, [4, 64]);
    }
}

/// An eligible matrix of integers is skipped by preflight and refused when staged.
///
/// Preflight sizes only float targets; the integer target therefore reaches
/// the quantizer, which names the rejected dtype at that target.
#[test]
fn integer_matrix_targets_fail_at_the_quantizer_with_their_dtype() {
    let root = tempfile::tempdir().unwrap();
    let header = serde_json::to_vec(&json!({
        "weight": {"dtype": "I32", "shape": [4, 8], "data_offsets": [0, 128]}
    }))
    .unwrap();
    let mut file = File::create(root.path().join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&[0_u8; 128]).unwrap();
    drop(file);
    let manifest = discover_checkpoint(root.path(), &LoadOptions::default()).unwrap();
    let expected = [ExpectedTensor::new("weight", [4, 8]).with_dtype(TensorDType::I32)];
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap();
    let output = root.path().join("integers.bpk");
    let failure = quantize_local_checkpoint_to_burnpack(
        &manifest,
        &plan,
        &output,
        QuantizedCheckpointOptions::default(),
    )
    .unwrap_err();
    assert_eq!(failure.failed_target.as_deref(), Some("weight"));
    assert!(matches!(
        failure.cause,
        CheckpointQuantizationFailureCause::Quantization(
            QuantizationError::UnsupportedInputDType {
                dtype: burn::tensor::DType::I32
            }
        )
    ));
    assert_eq!(failure.staging.tensor_count, 0);
    assert!(failure.report.applied.is_empty());
    assert!(!output.exists());
}
