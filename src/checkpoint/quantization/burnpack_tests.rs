//! Tests official Burnpack interoperability, staging atomicity, limits, and payload integrity.

use std::fs::{self, OpenOptions};
use std::mem::size_of;
use std::path::Path;

use burn::module::ParamId;
use burn::tensor::TensorData;
use burn_store::{BurnpackStore, BurnpackWriter, ModuleStore, TensorSnapshot};
use tempfile::tempdir;

use super::provenance::packed_payload_sha256;
use super::*;

/// Builds a fixture quantizer with the requested scheme and 32-value groups.
fn config(scheme: SymmetricQuantization) -> WeightQuantizationConfig {
    WeightQuantizationConfig {
        scheme,
        group_size: QuantizationGroupSize::new(32).unwrap(),
    }
}

/// Serializes packed codes followed by little-endian scales for byte comparisons.
fn payload(weight: &QuantizedWeight) -> Vec<u8> {
    let mut bytes = Vec::new();
    weight.write_burnpack_payload(&mut bytes).unwrap();
    bytes
}

/// Builds a named upstream Burnpack snapshot with a deterministic parameter identity.
fn official_snapshot(name: &str, data: TensorData, id: u64) -> TensorSnapshot {
    TensorSnapshot::from_data(
        data,
        vec![name.to_owned()],
        vec!["Fixture".into()],
        ParamId::from(id),
    )
}

/// Writes fixture tensors through Burn's own Burnpack writer.
fn write_official(path: &Path, tensors: Vec<(&str, TensorData)>) {
    let snapshots = tensors
        .into_iter()
        .enumerate()
        .map(|(index, (name, data))| official_snapshot(name, data, index as u64))
        .collect();
    BurnpackWriter::new(snapshots).write_to_file(path).unwrap();
}

/// Encodes the reserved JSON manifest as a one-dimensional U8 tensor.
fn manifest_tensor(manifest: &BurnpackQuantizationManifest) -> TensorData {
    let bytes = manifest.to_json_pretty().unwrap().into_bytes();
    let length = bytes.len();
    TensorData::new(bytes, [length])
}

/// Locates known fixture bytes before a deliberate corruption; panics if absent.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> usize {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("fixture payload must occur in Burnpack bytes")
}

/// Builds structurally valid synthetic digests for provenance-contract tests.
fn fixture_provenance() -> BurnpackProvenance {
    BurnpackProvenance {
        load_plan_sha256: "1".repeat(64),
        checkpoint_shards: std::collections::BTreeMap::from([(
            "model.safetensors".into(),
            BurnpackShardProvenance {
                file_bytes: 128,
                sha256: "2".repeat(64),
            },
        )]),
    }
}

/// Mapped v2 payload access verifies its digest and reports absent logical names precisely.
#[test]
fn artifact_contract_mapped_payload_is_read_only_and_name_checked() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("mapped-v2.bpk");
    let source = TensorData::new(vec![1.0_f32; 32], [1, 32]);
    let weight = quantize_tensor_data(&source, config(SymmetricQuantization::Q8)).unwrap();
    let expected = payload(&weight);
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append("q.weight", &source, config(SymmetricQuantization::Q8))
        .unwrap();
    session.bind_provenance(fixture_provenance()).unwrap();
    session.finalize().unwrap();
    let before = fs::read(&output).unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    for _ in 0..2 {
        let data = artifact.tensor_payload("q.weight").unwrap();
        assert_eq!(data.dtype, burn::tensor::DType::U8);
        assert_eq!(data.as_bytes(), expected);
    }
    assert!(
        matches!(artifact.tensor_payload("missing.weight"), Err(QuantizationError::QuantizedTensorNotFound { name }) if name == "missing.weight")
    );
    assert_eq!(fs::read(output).unwrap(), before);
}

/// A well-formed but false digest must fail lazy access without modifying the artifact.
#[test]
fn artifact_contract_mapped_payload_rejects_false_provenance_digest() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("false-digest.bpk");
    let weight = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 32], [1, 32]),
        config(SymmetricQuantization::Q8),
    )
    .unwrap();
    let bytes = payload(&weight);
    let actual_digest = packed_payload_sha256(&bytes);
    let expected_digest = "0".repeat(64);
    let mut metadata = weight.metadata().clone();
    metadata.payload_sha256 = Some(expected_digest.clone());
    let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
    manifest.insert("q.weight", metadata).unwrap();
    manifest.bind_provenance(fixture_provenance()).unwrap();
    let length = bytes.len();
    write_official(
        &output,
        vec![
            (BURNPACK_MANIFEST_TENSOR, manifest_tensor(&manifest)),
            ("q.weight", TensorData::new(bytes, [length])),
        ],
    );
    let before = fs::read(&output).unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    // Opening inspects only descriptors. Both subsequent reads must independently fail.
    for _ in 0..2 {
        assert!(
            matches!(artifact.tensor_payload("q.weight"), Err(QuantizationError::PackedPayloadDigestMismatch { name, expected, actual })
            if name == "q.weight" && expected == expected_digest && actual == actual_digest)
        );
    }
    assert_eq!(fs::read(output).unwrap(), before);
}

/// Unsupported or provenance-free version tags fail before model payload lookup.
#[test]
fn artifact_contract_open_rejects_invalid_manifest_version_provenance_pairs() {
    let directory = tempdir().unwrap();
    for version in [2, 3] {
        let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
        manifest.format_version = version;
        // Serialize deliberately invalid metadata directly, not through its validated writer.
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let length = bytes.len();
        let output = directory.path().join(format!("invalid-v{version}.bpk"));
        write_official(
            &output,
            vec![(BURNPACK_MANIFEST_TENSOR, TensorData::new(bytes, [length]))],
        );
        let before = fs::read(&output).unwrap();
        let error = QuantizedBurnpackArtifact::open(&output)
            .err()
            .expect("invalid manifest version/provenance pair");
        match version {
            2 => assert!(
                matches!(error, QuantizationError::InvalidProvenance { reason }
                if reason == "format version two requires checkpoint and load-plan provenance")
            ),
            3 => assert!(matches!(
                error,
                QuantizationError::UnsupportedFormatVersion { version: 3 }
            )),
            _ => unreachable!("fixed fixture versions"),
        }
        assert_eq!(fs::read(output).unwrap(), before);
    }
}

#[test]
/// Official writer/reader round trips preserve quantized payload bytes and manifest metadata.
fn official_burnpack_round_trip_preserves_exact_payloads_and_metadata() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("weights.bpk");
    let q8_source = TensorData::new((0..64).map(|index| index as f32 - 31.0).collect(), [8, 8]);
    let q4_source = TensorData::new(
        (0..35).map(|index| index as f32 * 0.25 - 4.0).collect(),
        [5, 7],
    );
    let expected_q8 = quantize_tensor_data(&q8_source, config(SymmetricQuantization::Q8)).unwrap();
    let expected_q4 = quantize_tensor_data(&q4_source, config(SymmetricQuantization::Q4)).unwrap();

    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    let first = session
        .append(
            "model.layers.0.q_proj.weight",
            &q8_source,
            config(SymmetricQuantization::Q8),
        )
        .unwrap();
    let second = session
        .append(
            "model.layers.0.down_proj.weight",
            &q4_source,
            config(SymmetricQuantization::Q4),
        )
        .unwrap();
    assert!(matches!(first, BurnpackAppendOutcome::Staged(_)));
    assert!(matches!(second, BurnpackAppendOutcome::Staged(_)));
    assert_eq!(session.stats().tensor_count, 2);
    assert_eq!(
        session.stats().staged_payload_bytes,
        payload(&expected_q8).len() as u64 + payload(&expected_q4).len() as u64
    );
    let expected_manifest = session.manifest().clone();

    let summary = session.finalize().unwrap();
    assert_eq!(summary.tensor_count, 2);
    let canonical_output = fs::canonicalize(directory.path())
        .unwrap()
        .join("weights.bpk");
    assert_eq!(summary.path, canonical_output);
    assert_eq!(summary.artifact_bytes, fs::metadata(&output).unwrap().len());
    #[cfg(unix)]
    assert!(summary.parent_directory_synchronized);
    assert_eq!(
        summary.largest_materialized_tensor_bytes,
        session
            .stats()
            .largest_payload_bytes
            .max(summary.manifest_bytes as u64)
    );
    assert!(output.exists());
    assert_eq!(&fs::read(&output).unwrap()[..4], b"NRUB");

    let mut official = BurnpackStore::from_file(&output)
        .auto_extension(false)
        .zero_copy(true);
    let official_keys = official.keys().unwrap();
    assert_eq!(official_keys.len(), 3);
    assert!(
        official_keys
            .iter()
            .any(|name| name == BURNPACK_MANIFEST_TENSOR)
    );

    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    assert_eq!(artifact.manifest(), &expected_manifest);
    assert_eq!(artifact.path(), canonical_output);
    assert_eq!(
        artifact
            .tensor_payload("model.layers.0.q_proj.weight")
            .unwrap()
            .as_bytes(),
        payload(&expected_q8)
    );
    let restored_q4 = artifact
        .read_weight("model.layers.0.down_proj.weight")
        .unwrap();
    assert_eq!(restored_q4.metadata(), expected_q4.metadata());
    assert_eq!(restored_q4.packed_values(), expected_q4.packed_values());
    assert_eq!(restored_q4.scales(), expected_q4.scales());
    let validation = artifact.validate_all().unwrap();
    assert_eq!(validation.tensor_count, 2);
    assert_eq!(validation.payload_bytes, summary.quantized_payload_bytes);
    let audit = artifact.validate_all_audited().unwrap();
    assert_eq!(audit.tensor_reads, 2);
    assert_eq!(audit.tensors_validated, 2);
    assert_eq!(
        audit.largest_payload_bytes,
        payload(&expected_q8).len().max(payload(&expected_q4).len()) as u64
    );
    assert!(audit.peak_owned_payload_bytes >= audit.largest_payload_bytes);
}

#[test]
/// Staging keeps range metadata in memory and removes its private file when the session drops.
fn staging_retains_only_ranges_and_is_removed_with_the_session() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("bounded.bpk");
    let staging_path;
    {
        let source = TensorData::new(vec![1.0_f32; 64], [8, 8]);
        let expected = quantize_tensor_data(&source, config(SymmetricQuantization::Q8)).unwrap();
        let mut session = QuantizedBurnpackSession::new(&output).unwrap();
        staging_path = session.staging_path_for_test().to_owned();
        let summary = session.append_quantized("layer.weight", expected).unwrap();
        assert_eq!(summary.staging_offset, 0);
        assert_eq!(session.staging_len_for_test(), summary.payload_bytes);
        assert_eq!(session.manifest().tensors.len(), 1);
        #[cfg(not(unix))]
        assert!(staging_path.exists());
        #[cfg(unix)]
        assert!(!staging_path.exists());
    }
    assert!(!staging_path.exists());
    assert!(!output.exists());
}

#[test]
/// Excluded targets require neither floating storage nor a staging write.
fn exclusions_never_reach_staging_or_require_a_float_dtype() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("excluded.bpk");
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    let source = TensorData::new(vec![1_i8; 64], [8, 8]);
    assert_eq!(
        session
            .append(
                "model.embed_tokens.weight",
                &source,
                WeightQuantizationConfig::default(),
            )
            .unwrap(),
        BurnpackAppendOutcome::Excluded(QuantizationExclusion::Embedding)
    );
    assert_eq!(session.stats().tensor_count, 0);
    assert_eq!(session.stats().staged_payload_bytes, 0);
    assert!(session.manifest().tensors.is_empty());
}

#[test]
/// Invalid and duplicate names fail before any payload is appended.
fn duplicate_empty_reserved_and_long_names_are_rejected_before_writing() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("names.bpk");
    let source = TensorData::new(vec![1.0_f32; 64], [8, 8]);
    let limits = BurnpackSessionLimits {
        max_tensor_name_bytes: 16,
        ..BurnpackSessionLimits::default()
    };
    let mut session = QuantizedBurnpackSession::with_limits(&output, limits).unwrap();

    assert!(matches!(
        session.append("", &source, WeightQuantizationConfig::default()),
        Err(QuantizationError::EmptyTensorName)
    ));
    assert!(matches!(
        session.append(
            BURNPACK_MANIFEST_TENSOR,
            &source,
            WeightQuantizationConfig::default()
        ),
        Err(QuantizationError::ReservedTensorCollision { .. })
    ));
    assert!(matches!(
        session.append(
            "a_name_far_too_long.weight",
            &source,
            WeightQuantizationConfig::default()
        ),
        Err(QuantizationError::SessionLimitExceeded {
            kind: "tensor name",
            ..
        })
    ));

    session
        .append("q.weight", &source, WeightQuantizationConfig::default())
        .unwrap();
    let before = session.stats();
    assert!(matches!(
        session.append("q.weight", &source, WeightQuantizationConfig::default()),
        Err(QuantizationError::DuplicateManifestTensor { .. })
    ));
    assert_eq!(session.stats(), before);
}

#[test]
/// Configured count, byte, manifest, and parameter limits reject excess work before publication.
fn limits_bound_counts_payloads_manifests_and_configuration() {
    let invalid = BurnpackSessionLimits {
        max_tensors: 0,
        ..BurnpackSessionLimits::default()
    };
    assert!(matches!(
        invalid.validate(),
        Err(QuantizationError::InvalidSessionLimit {
            field: "max_tensors",
            ..
        })
    ));
    let invalid = BurnpackSessionLimits {
        max_tensors: 100_000,
        ..BurnpackSessionLimits::default()
    };
    assert!(invalid.validate().is_err());
    let defaults = BurnpackSessionLimits::default();
    let invalid = BurnpackSessionLimits {
        max_staged_bytes: defaults.max_staged_bytes + 1,
        ..defaults
    };
    assert!(invalid.validate().is_err());
    let defaults = BurnpackSessionLimits::default();
    let invalid = BurnpackSessionLimits {
        max_tensor_payload_bytes: defaults.max_tensor_payload_bytes + 1,
        ..defaults
    };
    assert!(invalid.validate().is_err());
    let defaults = BurnpackSessionLimits::default();
    let invalid = BurnpackSessionLimits {
        max_manifest_bytes: defaults.max_manifest_bytes + 1,
        ..defaults
    };
    assert!(invalid.validate().is_err());
    let defaults = BurnpackSessionLimits::default();
    let invalid = BurnpackSessionLimits {
        max_tensor_name_bytes: defaults.max_tensor_name_bytes + 1,
        ..defaults
    };
    assert!(invalid.validate().is_err());

    let directory = tempdir().unwrap();
    let source = TensorData::new(vec![1.0_f32; 64], [8, 8]);
    let count_limits = BurnpackSessionLimits {
        max_tensors: 1,
        ..BurnpackSessionLimits::default()
    };
    let mut count_session =
        QuantizedBurnpackSession::with_limits(directory.path().join("count.bpk"), count_limits)
            .unwrap();
    count_session
        .append("a.weight", &source, WeightQuantizationConfig::default())
        .unwrap();
    assert!(matches!(
        count_session.append("b.weight", &source, WeightQuantizationConfig::default()),
        Err(QuantizationError::SessionLimitExceeded {
            kind: "tensor count",
            ..
        })
    ));

    let payload_limits = BurnpackSessionLimits {
        max_tensor_payload_bytes: 16,
        ..BurnpackSessionLimits::default()
    };
    let mut payload_session =
        QuantizedBurnpackSession::with_limits(directory.path().join("payload.bpk"), payload_limits)
            .unwrap();
    assert!(matches!(
        payload_session.append("a.weight", &source, WeightQuantizationConfig::default()),
        Err(QuantizationError::SessionLimitExceeded {
            kind: "tensor payload",
            ..
        })
    ));

    let staging_limits = BurnpackSessionLimits {
        max_staged_bytes: 35,
        ..BurnpackSessionLimits::default()
    };
    let mut staging_session =
        QuantizedBurnpackSession::with_limits(directory.path().join("staging.bpk"), staging_limits)
            .unwrap();
    assert!(matches!(
        staging_session.append("a.weight", &source, WeightQuantizationConfig::default()),
        Err(QuantizationError::SessionLimitExceeded {
            kind: "staged payload",
            ..
        })
    ));

    // The empty-manifest envelope is estimated at 512 bytes and one entry at
    // several hundred more, so 600 bytes admits the envelope but not the entry.
    let estimate_limits = BurnpackSessionLimits {
        max_manifest_bytes: 600,
        ..BurnpackSessionLimits::default()
    };
    let mut estimate_session = QuantizedBurnpackSession::with_limits(
        directory.path().join("estimate.bpk"),
        estimate_limits,
    )
    .unwrap();
    assert!(matches!(
        estimate_session.append("a.weight", &source, WeightQuantizationConfig::default()),
        Err(QuantizationError::SessionLimitExceeded {
            kind: "estimated manifest",
            ..
        })
    ));

    let manifest_limits = BurnpackSessionLimits {
        max_manifest_bytes: 1,
        ..BurnpackSessionLimits::default()
    };
    let mut empty = QuantizedBurnpackSession::with_limits(
        directory.path().join("manifest.bpk"),
        manifest_limits,
    )
    .unwrap();
    assert!(matches!(
        empty.finalize(),
        Err(QuantizationError::SessionLimitExceeded {
            kind: "manifest",
            ..
        })
    ));
}

#[test]
/// An existing output survives a publication collision and the session remains retryable.
fn output_collision_is_non_destructive_and_session_can_retry() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("collision.bpk");
    fs::write(&output, b"existing").unwrap();
    assert!(matches!(
        QuantizedBurnpackSession::new(&output),
        Err(QuantizationError::OutputAlreadyExists { .. })
    ));
    fs::remove_file(&output).unwrap();

    let source = TensorData::new(vec![1.0_f32; 64], [8, 8]);
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append("q.weight", &source, WeightQuantizationConfig::default())
        .unwrap();
    fs::write(&output, b"racing writer").unwrap();
    assert!(matches!(
        session.finalize(),
        Err(QuantizationError::OutputAlreadyExists { .. })
    ));
    assert_eq!(fs::read(&output).unwrap(), b"racing writer");
    fs::remove_file(&output).unwrap();
    session.finalize().unwrap();
    assert!(QuantizedBurnpackArtifact::open(output).is_ok());
}

/// Create-only publication preserves a dangling destination link and rolls back for a retry.
#[cfg(unix)]
#[test]
fn atomicity_dangling_output_symlink_survives_publication_and_retry() {
    use std::os::unix::fs::symlink;
    let directory = tempdir().unwrap();
    let output = directory.path().join("create-only.bpk");
    let missing_target = directory.path().join("absent-owner.bpk");
    let source = TensorData::new(vec![1.0_f32; 32], [1, 32]);
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append("q.weight", &source, config(SymmetricQuantization::Q8))
        .unwrap();
    session.bind_provenance(fixture_provenance()).unwrap();
    let before_stats = session.stats();
    let before_manifest = session.manifest().clone();
    symlink(&missing_target, &output).unwrap();
    assert!(!output.exists());
    assert!(
        fs::symlink_metadata(&output)
            .unwrap()
            .file_type()
            .is_symlink()
    );

    // The fast exists() guard cannot see this link. The create-only hard link is authoritative.
    for _ in 0..2 {
        assert!(
            matches!(session.finalize(), Err(QuantizationError::OutputAlreadyExists { path }) if path == session.output_path())
        );
        assert_eq!(fs::read_link(&output).unwrap(), missing_target);
        assert!(!missing_target.exists());
        assert_eq!(session.stats(), before_stats);
        assert_eq!(session.manifest(), &before_manifest);
        assert_eq!(
            session.staging_len_for_test(),
            before_stats.staged_payload_bytes
        );
        // On Unix the private staging inode is unlinked; no .partial directory may survive.
        let entries: Vec<_> = fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(entries, std::slice::from_ref(&output));
    }

    // Only the fixture's symlink is removed; its nonexistent target is never touched.
    fs::remove_file(&output).unwrap();
    let summary = session.finalize().unwrap();
    assert_eq!(summary.tensor_count, 1);
    assert!(
        !fs::symlink_metadata(&output)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(!missing_target.exists());
    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    assert_eq!(artifact.manifest(), &before_manifest);
    assert_eq!(artifact.validate_all().unwrap().tensor_count, 1);
    assert_eq!(
        artifact
            .read_weight("q.weight")
            .unwrap()
            .dequantize_f32()
            .unwrap(),
        source
    );
}

#[test]
/// Finalization closes further mutation and removes temporary publication state.
fn successful_finalization_closes_mutation_and_leaves_no_partial_file() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("final.bpk");
    let source = TensorData::new(vec![1.0_f32; 64], [8, 8]);
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    let staging_path = session.staging_path_for_test().to_owned();
    session
        .append("q.weight", &source, WeightQuantizationConfig::default())
        .unwrap();
    session.finalize().unwrap();
    assert!(!staging_path.exists());
    assert!(matches!(
        session.finalize(),
        Err(QuantizationError::SessionFinalized)
    ));
    assert!(matches!(
        session.append("k.weight", &source, WeightQuantizationConfig::default()),
        Err(QuantizationError::SessionFinalized)
    ));
    let filenames: Vec<_> = fs::read_dir(directory.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(filenames.iter().all(|name| !name.ends_with(".partial")));
}

#[test]
/// Provenance binding adds payload digests and freezes the staged inventory.
fn provenance_binding_adds_payload_digests_updates_size_and_freezes_staging() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("bound.bpk");
    let source = TensorData::new(vec![1.0_f32; 64], [8, 8]);
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append("q.weight", &source, WeightQuantizationConfig::default())
        .unwrap();
    let estimate_before = session.stats().estimated_manifest_bytes;

    session.bind_provenance(fixture_provenance()).unwrap();

    let manifest_bytes = session.manifest().to_json_pretty().unwrap().len() as u64;
    assert_eq!(session.manifest().format_version, 2);
    assert!(
        session.manifest().tensors["q.weight"]
            .payload_sha256
            .is_some()
    );
    assert!(session.stats().estimated_manifest_bytes >= estimate_before);
    assert!(session.stats().estimated_manifest_bytes >= manifest_bytes);
    assert!(matches!(
        session.append("k.weight", &source, WeightQuantizationConfig::default()),
        Err(QuantizationError::ProvenanceAlreadyBound)
    ));
    assert!(matches!(
        session.bind_provenance(fixture_provenance()),
        Err(QuantizationError::ProvenanceAlreadyBound)
    ));

    session.finalize().unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    artifact.validate_all().unwrap();
}

#[test]
/// Version-two publication detects changed staged bytes even when their length is unchanged.
fn v2_same_size_staging_corruption_is_rejected_before_publication() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("staging-content-corrupt.bpk");
    let source = TensorData::new(vec![1.0_f32; 64], [8, 8]);
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append("q.weight", &source, WeightQuantizationConfig::default())
        .unwrap();
    session.bind_provenance(fixture_provenance()).unwrap();
    let before_stats = session.stats();
    let before_manifest = session.manifest().clone();
    session.xor_staging_byte_for_test(0, 1);

    assert!(matches!(
        session.finalize(),
        Err(QuantizationError::PackedPayloadDigestMismatch { .. })
    ));
    assert!(!output.exists());
    assert_eq!(session.stats(), before_stats);
    assert_eq!(session.manifest(), &before_manifest);
    assert_eq!(
        session.staging_len_for_test(),
        before_stats.staged_payload_bytes
    );
    #[cfg(unix)]
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);

    // Restoring the deliberately flipped byte makes the same transaction retryable;
    // a stale appended manifest or partial Burnpack must not survive the failure.
    session.xor_staging_byte_for_test(0, 1);
    session.finalize().unwrap();
    assert_eq!(
        QuantizedBurnpackArtifact::open(output)
            .unwrap()
            .validate_all()
            .unwrap()
            .tensor_count,
        1
    );
}

#[test]
/// An externally changed staging length prevents a subsequent append.
fn unexpected_staging_mutation_is_detected_before_another_append() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("stage-corrupt.bpk");
    let source = TensorData::new(vec![1.0_f32; 64], [8, 8]);
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append("q.weight", &source, WeightQuantizationConfig::default())
        .unwrap();
    let corrupted_length = session.stats().staged_payload_bytes + 1;
    session.truncate_staging_for_test(corrupted_length);
    assert!(matches!(
        session.append("k.weight", &source, WeightQuantizationConfig::default()),
        Err(QuantizationError::StagingCorrupted { .. })
    ));
}

#[test]
/// An empty session still produces a valid upstream-readable artifact.
fn empty_session_is_still_a_valid_official_burnpack() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("empty.bpk");
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    let summary = session.finalize().unwrap();
    assert_eq!(summary.tensor_count, 0);
    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    assert!(artifact.manifest().tensors.is_empty());
    assert_eq!(
        artifact.manifest().packed_execution,
        PackedExecutionSupport::Unavailable
    );
    assert_eq!(
        artifact.validate_all().unwrap(),
        BurnpackValidationReport {
            tensor_count: 0,
            payload_bytes: 0,
        }
    );
}

#[test]
/// The manifest and physical payload inventory must agree exactly.
fn missing_manifest_and_unexpected_or_missing_payloads_are_rejected() {
    let directory = tempdir().unwrap();
    let missing_manifest = directory.path().join("missing-manifest.bpk");
    write_official(
        &missing_manifest,
        vec![("q.weight", TensorData::new(vec![0_u8; 36], [36]))],
    );
    assert!(matches!(
        QuantizedBurnpackArtifact::open(missing_manifest),
        Err(QuantizationError::MissingManifestTensor { .. })
    ));

    let unexpected = directory.path().join("unexpected.bpk");
    let empty_manifest = BurnpackQuantizationManifest::new_legacy_v1();
    write_official(
        &unexpected,
        vec![
            (BURNPACK_MANIFEST_TENSOR, manifest_tensor(&empty_manifest)),
            ("surprise.weight", TensorData::new(vec![0_u8; 1], [1])),
        ],
    );
    assert!(matches!(
        QuantizedBurnpackArtifact::open(unexpected),
        Err(QuantizationError::InvalidArtifactTensor { .. })
    ));

    let missing_payload = directory.path().join("missing-payload.bpk");
    let metadata = QuantizedTensorMetadata::new(
        &[8, 8],
        SourceFloatDType::F32,
        WeightQuantizationConfig::default(),
    )
    .unwrap();
    let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
    manifest.insert("q.weight", metadata).unwrap();
    write_official(
        &missing_payload,
        vec![(BURNPACK_MANIFEST_TENSOR, manifest_tensor(&manifest))],
    );
    assert!(matches!(
        QuantizedBurnpackArtifact::open(missing_payload),
        Err(QuantizationError::InvalidArtifactTensor { .. })
    ));
}

#[test]
/// Reserved payloads must use the required one-dimensional U8 representation.
fn physical_dtype_and_shape_must_match_the_reserved_u8_contract() {
    let directory = tempdir().unwrap();
    let metadata = QuantizedTensorMetadata::new(
        &[8, 8],
        SourceFloatDType::F32,
        WeightQuantizationConfig::default(),
    )
    .unwrap();
    let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
    manifest.insert("q.weight", metadata).unwrap();

    let wrong_shape = directory.path().join("wrong-shape.bpk");
    write_official(
        &wrong_shape,
        vec![
            (BURNPACK_MANIFEST_TENSOR, manifest_tensor(&manifest)),
            ("q.weight", TensorData::new(vec![0_u8; 35], [35])),
        ],
    );
    assert!(matches!(
        QuantizedBurnpackArtifact::open(wrong_shape),
        Err(QuantizationError::InvalidArtifactTensor { .. })
    ));

    let wrong_manifest_dtype = directory.path().join("wrong-manifest-dtype.bpk");
    write_official(
        &wrong_manifest_dtype,
        vec![(
            BURNPACK_MANIFEST_TENSOR,
            TensorData::new(vec![0.0_f32; 4], [4]),
        )],
    );
    assert!(matches!(
        QuantizedBurnpackArtifact::open(wrong_manifest_dtype),
        Err(QuantizationError::InvalidArtifactTensor { .. })
    ));

    let invalid_utf8 = directory.path().join("invalid-utf8.bpk");
    write_official(
        &invalid_utf8,
        vec![(
            BURNPACK_MANIFEST_TENSOR,
            TensorData::new(vec![0xff_u8], [1]),
        )],
    );
    assert!(matches!(
        QuantizedBurnpackArtifact::open(invalid_utf8),
        Err(QuantizationError::ManifestUtf8(_))
    ));
}

#[test]
/// Truncated and corrupted headers or manifests produce contextual artifact errors.
fn truncation_header_corruption_and_manifest_corruption_are_contextual() {
    let directory = tempdir().unwrap();
    let source = TensorData::new(vec![127.0_f32; 32], [4, 8]);

    let make_artifact = |filename: &str| {
        let output = directory.path().join(filename);
        let mut session = QuantizedBurnpackSession::new(&output).unwrap();
        session
            .append("q.weight", &source, config(SymmetricQuantization::Q8))
            .unwrap();
        let manifest = session.manifest().to_json_pretty().unwrap();
        session.finalize().unwrap();
        (output, manifest)
    };

    let (truncated, _) = make_artifact("truncated.bpk");
    let length = fs::metadata(&truncated).unwrap().len();
    OpenOptions::new()
        .write(true)
        .open(&truncated)
        .unwrap()
        .set_len(length - 1)
        .unwrap();
    assert!(matches!(
        QuantizedBurnpackArtifact::open(truncated),
        Err(QuantizationError::BurnStore { .. })
    ));

    let (bad_header, _) = make_artifact("header.bpk");
    let mut bytes = fs::read(&bad_header).unwrap();
    bytes[0] ^= 0xff;
    fs::write(&bad_header, bytes).unwrap();
    assert!(matches!(
        QuantizedBurnpackArtifact::open(bad_header),
        Err(QuantizationError::BurnStore { .. })
    ));

    let (bad_manifest, manifest) = make_artifact("manifest-corrupt.bpk");
    let mut bytes = fs::read(&bad_manifest).unwrap();
    let location = find_subslice(&bytes, manifest.as_bytes());
    bytes[location] = b'!';
    fs::write(&bad_manifest, bytes).unwrap();
    assert!(matches!(
        QuantizedBurnpackArtifact::open(bad_manifest),
        Err(QuantizationError::ManifestJson(_))
    ));
}

#[test]
/// Sequential validation rejects reserved quantized codes in persisted payloads.
fn invalid_packed_codes_are_found_by_sequential_validation() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("payload-corrupt.bpk");
    let source = TensorData::new(vec![127.0_f32; 32], [4, 8]);
    let expected = quantize_tensor_data(&source, config(SymmetricQuantization::Q8)).unwrap();
    let expected_payload = payload(&expected);
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append("q.weight", &source, config(SymmetricQuantization::Q8))
        .unwrap();
    session.finalize().unwrap();
    drop(session);

    let mut bytes = fs::read(&output).unwrap();
    let location = find_subslice(&bytes, &expected_payload);
    bytes[location] = 0x80;
    fs::write(&output, bytes).unwrap();

    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    let failure = artifact.validate_all_audited().unwrap_err();
    assert!(matches!(
        failure.error,
        QuantizationError::InvalidQuantizedValue {
            index: 0,
            value: -128
        }
    ));
    assert_eq!(failure.audit.tensor_reads, 1);
    assert_eq!(failure.audit.tensors_validated, 0);
    assert_eq!(failure.audit.payload_bytes, expected_payload.len() as u64);
    assert!(failure.audit.peak_owned_payload_bytes >= expected_payload.len() as u64);
    assert!(matches!(
        artifact.read_weight("missing.weight"),
        Err(QuantizationError::QuantizedTensorNotFound { .. })
    ));
}

#[test]
/// Sequential validation rejects malformed persisted quantization scales.
fn invalid_scales_are_found_by_sequential_validation() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("scale-corrupt.bpk");
    let source = TensorData::new(vec![127.0_f32; 32], [4, 8]);
    let expected = quantize_tensor_data(&source, config(SymmetricQuantization::Q8)).unwrap();
    let expected_payload = payload(&expected);
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append("q.weight", &source, config(SymmetricQuantization::Q8))
        .unwrap();
    session.finalize().unwrap();
    drop(session);

    let mut bytes = fs::read(&output).unwrap();
    let location = find_subslice(&bytes, &expected_payload);
    let scale_offset = location + expected.packed_values().len();
    bytes[scale_offset..scale_offset + size_of::<f32>()].copy_from_slice(&f32::NAN.to_le_bytes());
    fs::write(&output, bytes).unwrap();

    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    assert!(matches!(
        artifact.validate_all(),
        Err(QuantizationError::InvalidScale { group: 0, scale }) if scale.is_nan()
    ));
}

#[test]
/// Path and manifest size checks complete before payload access.
fn path_and_open_manifest_limits_are_enforced_before_payload_use() {
    let directory = tempdir().unwrap();
    assert!(matches!(
        QuantizedBurnpackSession::new(directory.path().join("wrong.bin")),
        Err(QuantizationError::InvalidArtifactPath { .. })
    ));
    assert!(matches!(
        QuantizedBurnpackSession::new(directory.path().join("missing/weights.bpk")),
        Err(QuantizationError::InvalidArtifactPath { .. })
    ));

    let output = directory.path().join("limit.bpk");
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    session.finalize().unwrap();
    assert!(matches!(
        QuantizedBurnpackArtifact::open_with_manifest_limit(&output, 0),
        Err(QuantizationError::InvalidSessionLimit { .. })
    ));
    assert!(matches!(
        QuantizedBurnpackArtifact::open_with_manifest_limit(&output, 1),
        Err(QuantizationError::SessionLimitExceeded {
            kind: "manifest",
            ..
        })
    ));
}

/// Failed source conversion cannot reserve a name, grow staging, or alter accounting.
#[test]
fn rejected_source_conversions_leave_staging_retryable() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("conversion-retry.bpk");
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    assert_eq!(
        session.output_path(),
        fs::canonicalize(directory.path())
            .unwrap()
            .join("conversion-retry.bpk")
    );
    let before_stats = session.stats();
    let before_manifest = session.manifest().clone();
    let mut short = TensorData::new(vec![1.0_f32; 31], [1, 31]);
    short.shape = burn::tensor::Shape::new([1, 32]);
    for (source, expected) in [
        (TensorData::new(vec![1_i32; 32], [1, 32]), "dtype"),
        (short, "bytes"),
        (TensorData::new(vec![f32::NAN; 32], [1, 32]), "nonfinite"),
    ] {
        let error = session
            .append("q.weight", &source, config(SymmetricQuantization::Q4))
            .unwrap_err();
        assert!(matches!(
            (expected, error),
            ("dtype", QuantizationError::UnsupportedInputDType { .. })
                | (
                    "bytes",
                    QuantizationError::SourceByteLengthMismatch {
                        expected: 128,
                        actual: 124
                    }
                )
                | ("nonfinite", QuantizationError::NonFiniteValue { index: 0 })
        ));
        assert_eq!(session.stats(), before_stats);
        assert_eq!(session.manifest(), &before_manifest);
        assert_eq!(session.staging_len_for_test(), 0);
        assert!(!output.exists());
    }
    let valid = TensorData::new(vec![1.0_f32; 32], [1, 32]);
    session
        .append("q.weight", &valid, config(SymmetricQuantization::Q4))
        .unwrap();
    session.finalize().unwrap();
    assert_eq!(
        QuantizedBurnpackArtifact::open(output)
            .unwrap()
            .validate_all()
            .unwrap()
            .tensor_count,
        1
    );
}

/// Invalid or oversized late provenance leaves the original v1 manifest publishable.
#[test]
fn provenance_validation_and_both_size_limits_are_transactional() {
    let directory = tempdir().unwrap();
    let source = TensorData::new(vec![1.0_f32; 32], [1, 32]);
    let weight = quantize_tensor_data(&source, config(SymmetricQuantization::Q4)).unwrap();
    let mut legacy = BurnpackQuantizationManifest::new_legacy_v1();
    legacy
        .insert("q.weight", weight.metadata().clone())
        .unwrap();
    legacy.packed_execution = PackedExecutionSupport::BurnQFloatMatmul;
    let exact_v1_staging =
        payload(&weight).len() as u64 + legacy.to_json_pretty().unwrap().len() as u64;
    // Admit exactly the appended entry's estimate, leaving no room for provenance.
    let mut probe = QuantizedBurnpackSession::new(directory.path().join("probe.bpk")).unwrap();
    probe.append_quantized("q.weight", weight.clone()).unwrap();
    let appended_estimate = usize::try_from(probe.stats().estimated_manifest_bytes).unwrap();
    drop(probe);
    for (case, limits) in [
        ("invalid", BurnpackSessionLimits::default()),
        (
            "estimated manifest",
            BurnpackSessionLimits {
                max_manifest_bytes: appended_estimate,
                ..BurnpackSessionLimits::default()
            },
        ),
        (
            "staged payload",
            BurnpackSessionLimits {
                max_staged_bytes: exact_v1_staging,
                ..BurnpackSessionLimits::default()
            },
        ),
    ] {
        let output = directory.path().join(format!("{case}.bpk"));
        let mut session = QuantizedBurnpackSession::with_limits(&output, limits).unwrap();
        session
            .append_quantized("q.weight", weight.clone())
            .unwrap();
        let before_stats = session.stats();
        let before_manifest = session.manifest().clone();
        let before_length = session.staging_len_for_test();
        let mut provenance = fixture_provenance();
        if case == "invalid" {
            provenance.load_plan_sha256 = "not-a-digest".into();
        }
        let error = session.bind_provenance(provenance).unwrap_err();
        match error {
            QuantizationError::InvalidProvenance { .. } => assert_eq!(case, "invalid"),
            QuantizationError::SessionLimitExceeded {
                kind,
                attempted,
                limit,
            } => {
                assert_eq!(kind, case);
                assert!(attempted > limit);
            }
            other => panic!("unexpected provenance failure: {other}"),
        }
        assert_eq!(session.stats(), before_stats);
        assert_eq!(session.manifest(), &before_manifest);
        assert_eq!(session.staging_len_for_test(), before_length);
        assert!(!output.exists());
        session.finalize().unwrap();
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        assert_eq!(artifact.manifest().format_version, 1);
        assert!(artifact.manifest().provenance.is_none());
        assert_eq!(artifact.read_weight("q.weight").unwrap(), weight);
    }
}

/// A directory disappearing after staging fails publication but permits a clean retry.
#[test]
fn missing_output_parent_rolls_back_finalization_manifest_for_retry() {
    let directory = tempdir().unwrap();
    let parent = directory.path().join("destination");
    fs::create_dir(&parent).unwrap();
    let output = parent.join("retry.bpk");
    let source = TensorData::new(vec![1.0_f32; 32], [1, 32]);
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append("q.weight", &source, config(SymmetricQuantization::Q4))
        .unwrap();
    session.bind_provenance(fixture_provenance()).unwrap();
    let before_stats = session.stats();
    let before_manifest = session.manifest().clone();
    // Unix sessions unlink their private staging name immediately, so moving the
    // empty directory invalidates the destination without invalidating its handle.
    let moved = directory.path().join("moved-destination");
    fs::rename(&parent, &moved).unwrap();
    assert!(matches!(
        session.finalize(),
        Err(QuantizationError::ArtifactIo { .. })
    ));
    assert_eq!(session.stats(), before_stats);
    assert_eq!(session.manifest(), &before_manifest);
    assert_eq!(
        session.staging_len_for_test(),
        before_stats.staged_payload_bytes
    );
    assert!(!output.exists());
    fs::rename(&moved, &parent).unwrap();
    let summary = session.finalize().unwrap();
    assert_eq!(summary.tensor_count, 1);
    assert_eq!(
        QuantizedBurnpackArtifact::open(output)
            .unwrap()
            .validate_all()
            .unwrap()
            .tensor_count,
        1
    );
    assert_eq!(fs::read_dir(&parent).unwrap().count(), 1);
}

/// Empty v2 artifacts have no packed execution or validation reads, even at the exact manifest limit.
#[test]
fn empty_bound_artifact_has_zero_audit_and_exact_manifest_boundary() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("empty-v2.bpk");
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    let before = session.stats();
    let excluded = TensorData::new(vec![1_i8; 8], [8]);
    for _ in 0..2 {
        assert!(matches!(
            session
                .append("layer.bias", &excluded, WeightQuantizationConfig::default())
                .unwrap(),
            BurnpackAppendOutcome::Excluded(_)
        ));
        assert_eq!(session.stats(), before);
        assert_eq!(session.staging_len_for_test(), 0);
    }
    session.bind_provenance(fixture_provenance()).unwrap();
    let summary = session.finalize().unwrap();
    let mut artifact =
        QuantizedBurnpackArtifact::open_with_manifest_limit(&output, summary.manifest_bytes)
            .unwrap();
    assert_eq!(artifact.manifest().format_version, 2);
    assert_eq!(
        artifact.manifest().packed_execution,
        PackedExecutionSupport::Unavailable
    );
    assert_eq!(
        artifact.validate_all_audited().unwrap(),
        burnpack::BurnpackValidationAudit::default()
    );
    assert!(
        matches!(artifact.tensor_payload("missing.weight"), Err(QuantizationError::QuantizedTensorNotFound { ref name }) if name == "missing.weight")
    );
    let failure = artifact
        .read_weight_audited("missing.weight")
        .err()
        .unwrap();
    assert_eq!(failure.stats, burnpack::BurnpackPayloadReadStats::default());
    assert!(matches!(
        failure.error,
        QuantizationError::QuantizedTensorNotFound { .. }
    ));
    assert!(
        matches!(QuantizedBurnpackArtifact::open_with_manifest_limit(&output, summary.manifest_bytes - 1), Err(QuantizationError::SessionLimitExceeded { kind: "manifest", attempted, limit }) if attempted == summary.manifest_bytes as u64 && limit + 1 == attempted)
    );
}

/// Reserved logical names, matrix-shaped manifests, and non-U8 model storage are distinct errors.
#[test]
fn reserved_manifest_and_wrong_rank_or_model_dtype_are_rejected() {
    let directory = tempdir().unwrap();
    let weight = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 32], [1, 32]),
        config(SymmetricQuantization::Q8),
    )
    .unwrap();
    let mut reserved = BurnpackQuantizationManifest::new_legacy_v1();
    reserved
        .insert(BURNPACK_MANIFEST_TENSOR, weight.metadata().clone())
        .unwrap();
    let output = directory.path().join("reserved.bpk");
    write_official(
        &output,
        vec![(BURNPACK_MANIFEST_TENSOR, manifest_tensor(&reserved))],
    );
    assert!(
        matches!(QuantizedBurnpackArtifact::open(output), Err(QuantizationError::ReservedTensorCollision { ref name }) if name == BURNPACK_MANIFEST_TENSOR)
    );

    let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
    manifest
        .insert("q.weight", weight.metadata().clone())
        .unwrap();
    let mut wrong_rank = manifest_tensor(&manifest);
    wrong_rank.shape = burn::tensor::Shape::new([1, wrong_rank.as_bytes().len()]);
    let output = directory.path().join("manifest-rank.bpk");
    write_official(&output, vec![(BURNPACK_MANIFEST_TENSOR, wrong_rank)]);
    assert!(
        matches!(QuantizedBurnpackArtifact::open(output), Err(QuantizationError::InvalidArtifactTensor { ref name, ref reason }) if name == BURNPACK_MANIFEST_TENSOR && reason.contains("expected one dimension"))
    );

    let output = directory.path().join("model-dtype.bpk");
    write_official(
        &output,
        vec![
            (BURNPACK_MANIFEST_TENSOR, manifest_tensor(&manifest)),
            ("q.weight", TensorData::new(vec![0_i8; 36], [36])),
        ],
    );
    assert!(
        matches!(QuantizedBurnpackArtifact::open(output), Err(QuantizationError::InvalidArtifactTensor { ref name, ref reason }) if name == "q.weight" && reason.contains("physical dtype"))
    );
}

/// Validation stops at the first bad payload and counts only its completely copied prefix.
#[test]
fn sequential_failure_audit_preserves_prior_success_and_skips_later_tensors() {
    let directory = tempdir().unwrap();
    let weight = quantize_tensor_data(
        &TensorData::new(vec![127.0_f32; 32], [1, 32]),
        config(SymmetricQuantization::Q8),
    )
    .unwrap();
    for failure_kind in ["code", "scale", "digest"] {
        let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
        let mut tensors = Vec::new();
        for name in ["a.weight", "b.weight", "c.weight"] {
            let mut metadata = weight.metadata().clone();
            let mut bytes = payload(&weight);
            if failure_kind == "digest" {
                metadata.payload_sha256 = Some(provenance::packed_payload_sha256(&bytes));
            }
            if name == "b.weight" {
                match failure_kind {
                    "code" => bytes[0] = 0x80,
                    "scale" => bytes[32..36].copy_from_slice(&0.0_f32.to_le_bytes()),
                    "digest" => bytes[0] ^= 1,
                    _ => unreachable!(),
                }
            }
            manifest.insert(name, metadata).unwrap();
            let length = bytes.len();
            tensors.push((name, TensorData::new(bytes, [length])));
        }
        if failure_kind == "digest" {
            manifest.packed_execution = PackedExecutionSupport::BurnQFloatMatmul;
            manifest.bind_provenance(fixture_provenance()).unwrap();
        }
        tensors.push((BURNPACK_MANIFEST_TENSOR, manifest_tensor(&manifest)));
        let output = directory.path().join(format!("prefix-{failure_kind}.bpk"));
        write_official(&output, tensors);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let failure = artifact.validate_all_audited().unwrap_err();
        assert_eq!(failure.audit.tensor_reads, 2);
        assert_eq!(failure.audit.tensors_validated, 1);
        assert_eq!(failure.audit.payload_bytes, 72);
        assert_eq!(failure.audit.largest_payload_bytes, 36);
        assert!(failure.audit.peak_owned_payload_bytes >= 40);
        assert!(matches!(
            (failure_kind, failure.error),
            (
                "code",
                QuantizationError::InvalidQuantizedValue {
                    index: 0,
                    value: -128
                }
            ) | (
                "scale",
                QuantizationError::InvalidScale {
                    group: 0,
                    scale: 0.0
                }
            ) | (
                "digest",
                QuantizationError::PackedPayloadDigestMismatch { .. }
            )
        ));
        if failure_kind == "digest" {
            assert!(
                matches!(artifact.tensor_payload("b.weight"), Err(QuantizationError::PackedPayloadDigestMismatch { ref name, .. }) if name == "b.weight")
            );
            assert_eq!(
                artifact.tensor_payload("a.weight").unwrap().as_bytes(),
                payload(&weight)
            );
        }
        // A sequential failure does not poison unrelated later lazy reads.
        assert_eq!(
            artifact.read_weight("c.weight").unwrap().packed_values(),
            weight.packed_values()
        );
    }
}

/// Reopening preserves each supported source dtype and every persisted group size.
#[test]
fn burnpack_boundary_source_dtypes_and_groups_survive_v2_reopen() {
    use burn::tensor::{DType, bf16, f16};

    let directory = tempdir().unwrap();
    for scheme in [SymmetricQuantization::Q4, SymmetricQuantization::Q8] {
        for group_size in [32, 64, 128] {
            let shape = [2, group_size];
            let values: Vec<_> = (0..2 * group_size)
                .map(|index| (index % 17) as f32 - 8.0)
                .collect();
            let f32_source = TensorData::new(values.clone(), shape);
            // Flex32 has the same F32 byte representation, but a distinct source tag.
            let mut flex32_source = f32_source.clone();
            flex32_source.dtype = DType::Flex32;
            for (source, dtype) in [
                (f32_source, SourceFloatDType::F32),
                (flex32_source, SourceFloatDType::Flex32),
                (
                    TensorData::new(values.iter().copied().map(f16::from_f32).collect(), shape),
                    SourceFloatDType::F16,
                ),
                (
                    TensorData::new(values.iter().copied().map(bf16::from_f32).collect(), shape),
                    SourceFloatDType::Bf16,
                ),
                (
                    TensorData::new(values.iter().copied().map(f64::from).collect(), shape),
                    SourceFloatDType::F64,
                ),
            ] {
                let configuration = WeightQuantizationConfig {
                    scheme,
                    group_size: QuantizationGroupSize::new(group_size).unwrap(),
                };
                let expected = quantize_tensor_data(&source, configuration).unwrap();
                let bytes = payload(&expected);
                let output = directory
                    .path()
                    .join(format!("{scheme:?}-{group_size}-{dtype:?}.bpk"));
                let mut session = QuantizedBurnpackSession::new(&output).unwrap();
                session.append("q.weight", &source, configuration).unwrap();
                session.bind_provenance(fixture_provenance()).unwrap();
                session.finalize().unwrap();
                drop(session);

                for _ in 0..2 {
                    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
                    assert_eq!(
                        artifact.manifest().packed_execution,
                        PackedExecutionSupport::BurnQFloatMatmul
                    );
                    let restored = artifact.read_weight("q.weight").unwrap();
                    let mut metadata = expected.metadata().clone();
                    metadata.payload_sha256 = Some(packed_payload_sha256(&bytes));
                    assert_eq!(restored.metadata(), &metadata);
                    assert_eq!(restored.metadata().source_dtype, dtype);
                    assert_eq!(restored.packed_values(), expected.packed_values());
                    assert_eq!(restored.scales(), expected.scales());
                    assert_eq!(
                        restored.dequantize_f32().unwrap(),
                        expected.dequantize_f32().unwrap()
                    );
                    assert_eq!(
                        artifact.tensor_payload("q.weight").unwrap().as_bytes(),
                        bytes
                    );
                    assert_eq!(
                        artifact.validate_all().unwrap().payload_bytes,
                        bytes.len() as u64
                    );
                }
            }
        }
    }
}

/// Row alignment, not the flattened element count, controls packed execution eligibility.
#[test]
fn burnpack_boundary_row_axes_and_partial_groups_keep_their_wire_layout() {
    let directory = tempdir().unwrap();
    for scheme in [SymmetricQuantization::Q4, SymmetricQuantization::Q8] {
        for (shape, packed) in [([2, 32], true), ([32, 2], false), ([1, 65], false)] {
            let source = TensorData::new(
                (0..shape[0] * shape[1])
                    .map(|index| index as f32 - 13.0)
                    .collect(),
                shape,
            );
            let expected = quantize_tensor_data(&source, config(scheme)).unwrap();
            let output = directory
                .path()
                .join(format!("{scheme:?}-{}-{}.bpk", shape[0], shape[1]));
            let mut session = QuantizedBurnpackSession::new(&output).unwrap();
            session
                .append_quantized("q.weight", expected.clone())
                .unwrap();
            session.bind_provenance(fixture_provenance()).unwrap();
            let manifest = session.manifest().clone();
            session.finalize().unwrap();
            let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
            assert_eq!(artifact.manifest(), &manifest);
            assert_eq!(
                manifest.packed_execution,
                if packed {
                    PackedExecutionSupport::BurnQFloatMatmul
                } else {
                    PackedExecutionSupport::Unavailable
                }
            );
            let restored = artifact.read_weight("q.weight").unwrap();
            assert_eq!(restored.packed_values(), expected.packed_values());
            assert_eq!(restored.scales(), expected.scales());
            assert_eq!(restored.metadata().host_shape().unwrap(), shape);
            assert_eq!(restored.clone().into_burn_qfloat_data().is_ok(), packed);
            assert_eq!(artifact.validate_all().unwrap().tensor_count, 1);
        }
    }
}

/// A mapped payload retains its original inode across publication of a new path owner.
#[cfg(unix)]
#[test]
fn burnpack_boundary_mapped_payload_outlives_reader_and_path_replacement() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("current.bpk");
    let publish = |value| {
        let source = TensorData::new(vec![value; 32], [1, 32]);
        let expected = quantize_tensor_data(&source, config(SymmetricQuantization::Q4)).unwrap();
        let mut session = QuantizedBurnpackSession::new(&output).unwrap();
        session
            .append_quantized("q.weight", expected.clone())
            .unwrap();
        session.bind_provenance(fixture_provenance()).unwrap();
        session.finalize().unwrap();
        payload(&expected)
    };
    let original = publish(1.0_f32);
    let mut old_reader = QuantizedBurnpackArtifact::open(&output).unwrap();
    let retained = old_reader.tensor_payload("q.weight").unwrap();
    let archived = directory.path().join("old.bpk");
    fs::rename(&output, &archived).unwrap();
    let replacement = publish(-3.0_f32);
    assert_ne!(original, replacement);
    // Unlinking an owned fixture is safe: the old mapping holds the original inode.
    fs::remove_file(&archived).unwrap();
    assert_eq!(
        old_reader.tensor_payload("q.weight").unwrap().as_bytes(),
        original
    );
    assert_eq!(
        old_reader
            .read_weight("q.weight")
            .unwrap()
            .dequantize_f32()
            .unwrap(),
        TensorData::new(vec![1.0_f32; 32], [1, 32])
    );
    drop(old_reader);
    assert_eq!(retained.as_bytes(), original);
    let mut new_reader = QuantizedBurnpackArtifact::open(&output).unwrap();
    assert_eq!(
        new_reader.tensor_payload("q.weight").unwrap().as_bytes(),
        replacement
    );
    assert_eq!(new_reader.validate_all().unwrap().tensor_count, 1);
    assert_eq!(retained.as_bytes(), original);
}

/// Valid SHA-256 binds bytes but cannot make an invalid quantized code semantically valid.
#[test]
fn burnpack_boundary_valid_digest_does_not_approve_bad_codes_or_q4_padding() {
    let directory = tempdir().unwrap();
    for case in ["q8-code", "q4-code", "q4-padding"] {
        let scheme = if case == "q8-code" {
            SymmetricQuantization::Q8
        } else {
            SymmetricQuantization::Q4
        };
        let weight =
            quantize_tensor_data(&TensorData::new(vec![1.0_f32; 33], [1, 33]), config(scheme))
                .unwrap();
        let mut bytes = payload(&weight);
        match case {
            "q8-code" => bytes[32] = 0x80,
            "q4-code" => bytes[16] = 0x08,
            "q4-padding" => bytes[16] |= 0x10,
            _ => unreachable!("fixed corruption cases"),
        }
        let mut metadata = weight.metadata().clone();
        metadata.payload_sha256 = Some(packed_payload_sha256(&bytes));
        let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
        manifest.insert("q.weight", metadata).unwrap();
        manifest.bind_provenance(fixture_provenance()).unwrap();
        let output = directory.path().join(format!("{case}.bpk"));
        let length = bytes.len();
        write_official(
            &output,
            vec![
                (BURNPACK_MANIFEST_TENSOR, manifest_tensor(&manifest)),
                ("q.weight", TensorData::new(bytes.clone(), [length])),
            ],
        );
        let before = fs::read(&output).unwrap();
        let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
        assert_eq!(
            artifact.tensor_payload("q.weight").unwrap().as_bytes(),
            bytes
        );
        for _ in 0..2 {
            let failure = artifact.read_weight_audited("q.weight").err().unwrap();
            match (case, failure.error) {
                (
                    "q8-code",
                    QuantizationError::InvalidQuantizedValue {
                        index: 32,
                        value: -128,
                    },
                )
                | (
                    "q4-code",
                    QuantizationError::InvalidQuantizedValue {
                        index: 32,
                        value: -8,
                    },
                ) => {}
                ("q4-padding", QuantizationError::InvalidMetadata { reason }) => {
                    assert_eq!(reason, "unused final Q4 high nibble must be zero")
                }
                (_, other) => panic!("unexpected {case} error: {other}"),
            }
            assert_eq!(failure.stats.tensor_reads, 1);
            assert_eq!(failure.stats.payload_bytes, length as u64);
            assert!(
                failure.stats.peak_owned_payload_bytes
                    >= length as u64 + weight.metadata().scale_bytes
            );
        }
        assert_eq!(fs::read(output).unwrap(), before);
    }
}

/// Integrity-valid payloads still reject a non-positive or non-finite second-group scale.
#[test]
fn burnpack_boundary_valid_digest_does_not_approve_invalid_group_scales() {
    let directory = tempdir().unwrap();
    let weight = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 64], [2, 32]),
        config(SymmetricQuantization::Q8),
    )
    .unwrap();
    for (index, scale) in [
        0.0_f32,
        -0.0,
        -1.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ]
    .into_iter()
    .enumerate()
    {
        let mut bytes = payload(&weight);
        let second_scale = weight.packed_values().len() + size_of::<f32>();
        bytes[second_scale..second_scale + size_of::<f32>()].copy_from_slice(&scale.to_le_bytes());
        let mut metadata = weight.metadata().clone();
        metadata.payload_sha256 = Some(packed_payload_sha256(&bytes));
        let mut manifest = BurnpackQuantizationManifest::new_legacy_v1();
        manifest.insert("q.weight", metadata).unwrap();
        manifest.bind_provenance(fixture_provenance()).unwrap();
        let output = directory.path().join(format!("scale-{index}.bpk"));
        let length = bytes.len();
        write_official(
            &output,
            vec![
                (BURNPACK_MANIFEST_TENSOR, manifest_tensor(&manifest)),
                ("q.weight", TensorData::new(bytes.clone(), [length])),
            ],
        );
        let before = fs::read(&output).unwrap();
        let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
        assert_eq!(
            artifact.tensor_payload("q.weight").unwrap().as_bytes(),
            bytes
        );
        let failure = artifact.validate_all_audited().unwrap_err();
        match failure.error {
            QuantizationError::InvalidScale {
                group: 1,
                scale: actual,
            } => assert_eq!(actual.to_bits(), scale.to_bits()),
            other => panic!("unexpected scale {scale:?} error: {other}"),
        }
        assert_eq!(failure.audit.tensor_reads, 1);
        assert_eq!(failure.audit.tensors_validated, 0);
        assert_eq!(failure.audit.payload_bytes, length as u64);
        assert_eq!(fs::read(output).unwrap(), before);
    }
}

/// A legacy artifact cannot publish malformed staging; restoring its code permits a retry.
#[test]
fn burnpack_boundary_legacy_semantic_publication_failure_is_retryable() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("legacy-retry.bpk");
    let source = TensorData::new(vec![1.0_f32; 32], [1, 32]);
    let expected = quantize_tensor_data(&source, config(SymmetricQuantization::Q4)).unwrap();
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append_quantized("q.weight", expected.clone())
        .unwrap();
    let stats = session.stats();
    // Q4's first 0x7 code becomes the forbidden -8 without changing payload length.
    session.xor_staging_byte_for_test(0, 0x0f);
    for _ in 0..2 {
        assert!(matches!(
            session.finalize(),
            Err(QuantizationError::InvalidQuantizedValue {
                index: 0,
                value: -8
            })
        ));
        assert_eq!(session.stats(), stats);
        assert_eq!(session.staging_len_for_test(), stats.staged_payload_bytes);
        assert_eq!(session.manifest().format_version, 1);
        assert!(!output.exists());
        assert!(fs::read_dir(directory.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".partial")
        }));
    }
    session.xor_staging_byte_for_test(0, 0x0f);
    session.finalize().unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
    assert_eq!(artifact.read_weight("q.weight").unwrap(), expected);
    assert_eq!(artifact.validate_all().unwrap().tensor_count, 1);
}

/// Invalid file-path spellings fail before creation, while an OS name error preserves neighbors.
#[test]
fn burnpack_boundary_invalid_paths_and_long_name_errors_are_contextual() {
    for path in [Path::new(""), Path::new("/"), Path::new(".")] {
        for error in [
            QuantizedBurnpackSession::new(path).err().unwrap(),
            QuantizedBurnpackArtifact::open(path).err().unwrap(),
        ] {
            assert!(
                matches!(error, QuantizationError::InvalidArtifactPath { path: actual, reason: "expected a file path" } if actual == path)
            );
        }
    }
    let directory = tempdir().unwrap();
    let sentinel = directory.path().join("existing.bpk");
    fs::write(&sentinel, b"unrelated owner").unwrap();
    let output = directory.path().join(format!("{}.bpk", "x".repeat(300)));
    let error = QuantizedBurnpackSession::new(&output).err().unwrap();
    let canonical_parent = fs::canonicalize(directory.path()).unwrap();
    assert!(
        matches!(error, QuantizationError::ArtifactIo { operation: "create private Burnpack temporary file", path, .. } if path.parent() == Some(canonical_parent.as_path()) && path.file_name().unwrap().to_string_lossy().contains("quant-stage"))
    );
    assert_eq!(fs::read(&sentinel).unwrap(), b"unrelated owner");
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}

/// Canonicalizing the destination parent pins publication before an alias is retargeted.
#[cfg(unix)]
#[test]
fn burnpack_boundary_parent_alias_retargeting_cannot_redirect_publication() {
    use std::os::unix::fs::symlink;

    let directory = tempdir().unwrap();
    let canonical_parent = fs::canonicalize(directory.path()).unwrap();
    let original = canonical_parent.join("original");
    let other = canonical_parent.join("other");
    fs::create_dir(&original).unwrap();
    fs::create_dir(&other).unwrap();
    let alias = directory.path().join("alias");
    symlink(&original, &alias).unwrap();
    let mut session = QuantizedBurnpackSession::new(alias.join("weights.bpk")).unwrap();
    assert_eq!(session.output_path(), original.join("weights.bpk"));
    fs::remove_file(&alias).unwrap();
    symlink(&other, &alias).unwrap();
    let source = TensorData::new(vec![1.0_f32; 32], [1, 32]);
    session
        .append("q.weight", &source, config(SymmetricQuantization::Q8))
        .unwrap();
    session.bind_provenance(fixture_provenance()).unwrap();
    let summary = session.finalize().unwrap();
    assert_eq!(summary.path, original.join("weights.bpk"));
    assert_eq!(fs::read_link(alias).unwrap(), other);
    assert_eq!(fs::read_dir(&other).unwrap().count(), 0);
    let mut artifact = QuantizedBurnpackArtifact::open(summary.path).unwrap();
    assert_eq!(
        artifact
            .read_weight("q.weight")
            .unwrap()
            .dequantize_f32()
            .unwrap(),
        source
    );
}

/// A weight read from a v2 artifact re-stages into a v1 session that still finalizes.
#[test]
fn v2_read_weights_restage_into_a_publishable_v1_session() {
    let directory = tempdir().unwrap();
    let bound = directory.path().join("bound.bpk");
    let source = TensorData::new(
        (0..64).map(|value| value as f32).collect::<Vec<_>>(),
        [2, 32],
    );
    let mut session = QuantizedBurnpackSession::new(&bound).unwrap();
    session
        .append("q.weight", &source, config(SymmetricQuantization::Q4))
        .unwrap();
    session.bind_provenance(fixture_provenance()).unwrap();
    session.finalize().unwrap();
    let carried = QuantizedBurnpackArtifact::open(&bound)
        .unwrap()
        .read_weight("q.weight")
        .unwrap();
    assert!(carried.metadata().payload_sha256.is_some());

    let restaged = directory.path().join("restaged.bpk");
    let mut legacy = QuantizedBurnpackSession::new(&restaged).unwrap();
    let summary = legacy
        .append_quantized("q.weight", carried.clone())
        .unwrap();
    assert!(summary.metadata.payload_sha256.is_none());
    legacy.finalize().unwrap();
    let mut artifact = QuantizedBurnpackArtifact::open(&restaged).unwrap();
    assert_eq!(artifact.manifest().format_version, 1);
    let reread = artifact.read_weight("q.weight").unwrap();
    assert_eq!(reread.packed_values(), carried.packed_values());
    assert_eq!(reread.scales(), carried.scales());

    // Binding provenance later attaches the session's own staged digest.
    let rebound = directory.path().join("rebound.bpk");
    let mut session = QuantizedBurnpackSession::new(&rebound).unwrap();
    session
        .append_quantized("q.weight", carried.clone())
        .unwrap();
    session.bind_provenance(fixture_provenance()).unwrap();
    assert_eq!(
        session.manifest().tensors["q.weight"].payload_sha256,
        carried.metadata().payload_sha256
    );
}

/// A carried digest that disagrees with the staged bytes rolls the append back.
#[test]
fn carried_digest_mismatch_rolls_back_the_append() {
    let directory = tempdir().unwrap();
    let weight = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 32], [1, 32]),
        config(SymmetricQuantization::Q8),
    )
    .unwrap();
    let (mut metadata, packed, scales) = weight.into_parts();
    metadata.payload_sha256 = Some("0".repeat(64));
    // The digest-verified constructor is the crate-internal path that skips hashing.
    let forged =
        QuantizedWeight::from_payload_digest_verified_parts(metadata, packed, scales).unwrap();
    let mut session = QuantizedBurnpackSession::new(directory.path().join("forged.bpk")).unwrap();
    let before = session.stats();
    assert!(matches!(
        session.append_quantized("q.weight", forged),
        Err(QuantizationError::PackedPayloadDigestMismatch { name, .. }) if name == "q.weight"
    ));
    assert_eq!(session.stats(), before);
    assert_eq!(session.staging_len_for_test(), 0);
    assert!(session.manifest().tensors.is_empty());
}

/// Manifest estimates bound the real pretty JSON tightly enough for large MoE inventories.
#[test]
fn manifest_estimate_is_a_tight_upper_bound() {
    let directory = tempdir().unwrap();
    let source = TensorData::new(vec![0.5_f32; 64], [2, 32]);
    let mut session = QuantizedBurnpackSession::new(directory.path().join("moe.bpk")).unwrap();
    let mut names = Vec::new();
    for layer in [0, 7, 93] {
        for expert in [0, 31, 127] {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                names.push(format!(
                    "model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                ));
            }
        }
    }
    names.push("model.layers.0.self_attn.\"quoted\\\\name\".weight".into());
    for name in &names {
        session
            .append(name, &source, config(SymmetricQuantization::Q4))
            .unwrap();
    }
    let unbound_actual = session.manifest().to_json_pretty().unwrap().len() as u64;
    let unbound_estimate = session.stats().estimated_manifest_bytes;
    assert!(unbound_estimate >= unbound_actual);
    session.bind_provenance(fixture_provenance()).unwrap();
    let bound_actual = session.manifest().to_json_pretty().unwrap().len() as u64;
    assert!(session.stats().estimated_manifest_bytes >= bound_actual);
    // Each entry is charged at most a small fixed slack over its bound JSON size.
    let slack_per_entry = unbound_estimate.saturating_sub(bound_actual) / names.len() as u64;
    assert!(
        slack_per_entry < 128,
        "slack {slack_per_entry} bytes per entry"
    );
    // At this per-entry size, Burn Store's 99,999-tensor limit binds before 64 MiB.
    let per_entry = unbound_estimate / names.len() as u64;
    assert!(
        per_entry * 99_999 < 64 * 1024 * 1024,
        "{per_entry} bytes per entry"
    );
}

/// An unreadable output directory still publishes atomically, reporting no sync.
///
/// Creating, linking, and removing entries needs only write and search
/// permission, but synchronizing the directory needs a readable handle. The
/// published artifact must therefore be complete and valid while the summary
/// truthfully reports that its directory entry was not synchronized. A
/// privileged process can open the directory anyway and then reports a sync.
#[cfg(unix)]
#[test]
fn publication_without_a_readable_parent_reports_unsynchronized_durability() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempdir().unwrap();
    let parent = directory.path().join("write-only");
    fs::create_dir(&parent).unwrap();
    let output = parent.join("weights.bpk");
    let source = TensorData::new(vec![0.5_f32; 64], [2, 32]);
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    session
        .append("q.weight", &source, config(SymmetricQuantization::Q8))
        .unwrap();
    fs::set_permissions(&parent, fs::Permissions::from_mode(0o300)).unwrap();
    let privileged = fs::File::open(&parent).is_ok();
    let summary = session.finalize();
    fs::set_permissions(&parent, fs::Permissions::from_mode(0o700)).unwrap();
    let summary = summary.unwrap();
    assert_eq!(summary.parent_directory_synchronized, privileged);
    assert_eq!(summary.tensor_count, 1);
    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    assert_eq!(artifact.validate_all().unwrap().tensor_count, 1);
    let names: Vec<_> = fs::read_dir(&parent)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["weights.bpk"]);
}
