use std::fs::{self, OpenOptions};
use std::mem::size_of;
use std::path::Path;

use burn::module::ParamId;
use burn::tensor::TensorData;
use burn_store::{BurnpackStore, BurnpackWriter, ModuleStore, TensorSnapshot};
use tempfile::tempdir;

use super::*;

fn config(scheme: SymmetricQuantization) -> WeightQuantizationConfig {
    WeightQuantizationConfig {
        scheme,
        group_size: QuantizationGroupSize::new(32).unwrap(),
    }
}

fn payload(weight: &QuantizedWeight) -> Vec<u8> {
    let mut bytes = Vec::new();
    weight.write_burnpack_payload(&mut bytes).unwrap();
    bytes
}

fn official_snapshot(name: &str, data: TensorData, id: u64) -> TensorSnapshot {
    TensorSnapshot::from_data(
        data,
        vec![name.to_owned()],
        vec!["Fixture".into()],
        ParamId::from(id),
    )
}

fn write_official(path: &Path, tensors: Vec<(&str, TensorData)>) {
    let snapshots = tensors
        .into_iter()
        .enumerate()
        .map(|(index, (name, data))| official_snapshot(name, data, index as u64))
        .collect();
    BurnpackWriter::new(snapshots).write_to_file(path).unwrap();
}

fn manifest_tensor(manifest: &BurnpackQuantizationManifest) -> TensorData {
    let bytes = manifest.to_json_pretty().unwrap().into_bytes();
    let length = bytes.len();
    TensorData::new(bytes, [length])
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> usize {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("fixture payload must occur in Burnpack bytes")
}

#[test]
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
    assert_eq!(
        artifact.validate_all().unwrap(),
        BurnpackValidationReport {
            tensor_count: 2,
            payload_bytes: summary.quantized_payload_bytes,
        }
    );
}

#[test]
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

    let estimate_limits = BurnpackSessionLimits {
        max_manifest_bytes: 3_000,
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

#[test]
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
fn empty_session_is_still_a_valid_official_burnpack() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("empty.bpk");
    let mut session = QuantizedBurnpackSession::new(&output).unwrap();
    let summary = session.finalize().unwrap();
    assert_eq!(summary.tensor_count, 0);
    let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
    assert!(artifact.manifest().tensors.is_empty());
    assert_eq!(
        artifact.validate_all().unwrap(),
        BurnpackValidationReport {
            tensor_count: 0,
            payload_bytes: 0,
        }
    );
}

#[test]
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
    let empty_manifest = BurnpackQuantizationManifest::new();
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
    let mut manifest = BurnpackQuantizationManifest::new();
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
fn physical_dtype_and_shape_must_match_the_reserved_u8_contract() {
    let directory = tempdir().unwrap();
    let metadata = QuantizedTensorMetadata::new(
        &[8, 8],
        SourceFloatDType::F32,
        WeightQuantizationConfig::default(),
    )
    .unwrap();
    let mut manifest = BurnpackQuantizationManifest::new();
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
    assert!(matches!(
        artifact.validate_all(),
        Err(QuantizationError::InvalidQuantizedValue {
            index: 0,
            value: -128
        })
    ));
    assert!(matches!(
        artifact.read_weight("missing.weight"),
        Err(QuantizationError::QuantizedTensorNotFound { .. })
    ));
}

#[test]
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
