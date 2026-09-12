//! Test-only fixtures for burnpack.rs; excluded from non-test compiler inputs.

use super::*;
use crate::checkpoint::quantization::{QuantizationGroupSize, SymmetricQuantization};

/// Path validation resolves parent aliases without creating the requested artifact.
#[test]
fn artifact_path_retains_filename_and_canonicalizes_parent_without_writing() {
    let directory = tempfile::tempdir().expect("temporary artifact directory");
    let requested = directory.path().join(".").join("weights.bpk");
    let validated = validate_bpk_path(&requested).expect("existing parent and valid extension");
    assert_eq!(
        validated,
        fs::canonicalize(directory.path())
            .expect("canonical temporary directory")
            .join("weights.bpk")
    );
    assert!(
        !validated.exists(),
        "validation must not create an artifact"
    );

    let missing_parent = directory.path().join("missing").join("weights.bpk");
    assert!(matches!(
        validate_bpk_path(&missing_parent),
        Err(QuantizationError::InvalidArtifactPath {
            path,
            reason: "parent directory does not exist",
        }) if path == missing_parent
    ));
}

/// Native Unix filenames remain byte-preserving even when their stem is not UTF-8.
#[cfg(unix)]
#[test]
fn artifact_path_preserves_non_utf8_filename_bytes() {
    use std::os::unix::ffi::OsStringExt;

    let directory = tempfile::tempdir().expect("temporary artifact directory");
    let filename = std::ffi::OsString::from_vec(b"weights-\xff.bpk".to_vec());
    let validated = validate_bpk_path(&directory.path().join(&filename))
        .expect("native filename with a valid ASCII extension");
    assert_eq!(validated.file_name(), Some(filename.as_os_str()));
    assert!(
        !validated.exists(),
        "validation must not create an artifact"
    );
}

/// Produces one small row-aligned payload with valid codes and positive scales.
fn packed_fixture() -> (QuantizedTensorMetadata, Vec<u8>) {
    let weight = quantize_tensor_data(
        &TensorData::new((0..64).map(|index| index as f32 - 31.0).collect(), [2, 32]),
        WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q4,
            group_size: QuantizationGroupSize::new(32).unwrap(),
        },
    )
    .unwrap();
    let mut payload = Vec::new();
    weight.write_burnpack_payload(&mut payload).unwrap();
    (weight.metadata().clone(), payload)
}

/// Rejects each upstream storage limit before any file or model allocation.
#[test]
fn independent_session_maxima_and_open_limit_are_enforced() {
    let defaults = BurnpackSessionLimits::default();
    for (field, limits) in [
        (
            "max_tensors",
            BurnpackSessionLimits {
                max_tensors: UPSTREAM_MAX_TENSORS,
                ..defaults
            },
        ),
        (
            "max_staged_bytes",
            BurnpackSessionLimits {
                max_staged_bytes: DEFAULT_MAX_STAGED_BYTES + 1,
                ..defaults
            },
        ),
        (
            "max_tensor_payload_bytes",
            BurnpackSessionLimits {
                max_tensor_payload_bytes: UPSTREAM_MAX_TENSOR_BYTES + 1,
                ..defaults
            },
        ),
        (
            "max_manifest_bytes",
            BurnpackSessionLimits {
                max_manifest_bytes: DEFAULT_MAX_MANIFEST_BYTES + 1,
                ..defaults
            },
        ),
        (
            "max_tensor_name_bytes",
            BurnpackSessionLimits {
                max_tensor_name_bytes: DEFAULT_MAX_TENSOR_NAME_BYTES + 1,
                ..defaults
            },
        ),
    ] {
        assert!(
            matches!(limits.validate(), Err(QuantizationError::InvalidSessionLimit { field: actual, .. }) if actual == field)
        );
    }
    assert!(matches!(
        QuantizedBurnpackArtifact::open_with_manifest_limit(
            "never-opened.bpk",
            DEFAULT_MAX_MANIFEST_BYTES + 1
        ),
        Err(QuantizationError::InvalidSessionLimit {
            field: "max_manifest_bytes",
            ..
        })
    ));
}

/// Dtype, shape and byte count remain separate errors, including mutated TensorData metadata.
#[test]
fn payload_descriptor_rejects_each_physical_inconsistency() {
    let mut wrong_storage_length = TensorData::new(vec![1_u8, 2], [2]);
    wrong_storage_length.shape = Shape::new([4]);
    for (data, expected, fragment) in [
        (TensorData::new(vec![1_f32], [1]), 1, "physical dtype"),
        (
            TensorData::new(vec![1_u8, 2, 3, 4], [2, 2]),
            4,
            "physical shape",
        ),
        (wrong_storage_length, 4, "physical payload has 2 bytes"),
    ] {
        match validate_u8_payload_descriptor("layer.weight", &data, expected).unwrap_err() {
            QuantizationError::InvalidArtifactTensor { name, reason } => {
                assert_eq!(name, "layer.weight");
                assert!(reason.contains(fragment), "{reason}");
            }
            other => panic!("unexpected descriptor error: {other}"),
        }
    }
    validate_u8_payload_descriptor("empty", &TensorData::new(Vec::<u8>::new(), [0]), 0).unwrap();
}

/// Missing/truncated input fails before ownership and does not invent read progress.
#[test]
fn decoder_rejects_length_and_arithmetic_errors_before_copying() {
    let (metadata, bytes) = packed_fixture();
    let failure = decode_payload("layer.weight", metadata.clone(), &bytes[..bytes.len() - 1])
        .expect_err("short payload");
    assert!(matches!(
        failure.error,
        QuantizationError::InvalidArtifactTensor { .. }
    ));
    assert_eq!(failure.stats.tensor_reads, 0);
    assert_eq!(failure.stats.payload_bytes, 0);
    assert_eq!(failure.stats.peak_owned_payload_bytes, 0);

    let impossible = QuantizedTensorMetadata {
        packed_value_bytes: u64::MAX,
        scale_bytes: 4,
        ..metadata
    };
    let failure = decode_payload("overflow", impossible, &[]).expect_err("length overflow");
    assert!(matches!(
        failure.error,
        QuantizationError::ByteLengthOverflow { .. }
    ));
    assert_eq!(failure.stats.tensor_reads, 0);
}

/// Digest and scale-layout failures retain the exact completed owned-byte prefix.
#[test]
fn decoder_audit_distinguishes_copied_digest_and_scale_failures() {
    let (mut metadata, bytes) = packed_fixture();
    metadata.payload_sha256 = Some("0".repeat(64));
    let failure = decode_payload("digest", metadata.clone(), &bytes).expect_err("digest mismatch");
    assert!(matches!(
        failure.error,
        QuantizationError::PackedPayloadDigestMismatch { .. }
    ));
    assert_eq!(failure.stats.tensor_reads, 1);
    assert_eq!(failure.stats.payload_bytes, bytes.len() as u64);
    assert!(failure.stats.peak_owned_payload_bytes >= bytes.len() as u64);

    metadata.payload_sha256 = None;
    metadata.scale_bytes = 3;
    let invalid_bytes = &bytes[..metadata.packed_value_bytes as usize + 3];
    let failure = decode_payload("scales", metadata, invalid_bytes).expect_err("unaligned scales");
    assert!(
        matches!(failure.error, QuantizationError::InvalidMetadata { ref reason } if reason.contains("divisible by four"))
    );
    assert_eq!(failure.stats.tensor_reads, 1);
    assert_eq!(failure.stats.payload_bytes, invalid_bytes.len() as u64);
}

/// Owned-byte accounting measures concurrent buffers and rejects both overflow factors.
#[test]
fn owned_memory_accounting_is_checked() {
    assert_eq!(owned_payload_bytes(40, 2).unwrap(), 48);
    assert_eq!(owned_payload_bytes(0, 0).unwrap(), 0);
    assert!(matches!(
        owned_payload_bytes(0, usize::MAX),
        Err(QuantizationError::ByteLengthOverflow {
            context: "owned packed-scale capacity"
        })
    ));
    #[cfg(target_pointer_width = "64")]
    assert!(matches!(
        owned_payload_bytes(usize::MAX, 1),
        Err(QuantizationError::ByteLengthOverflow {
            context: "owned packed decoder high-water mark"
        })
    ));
}

/// Aggregation rejects either overflowing counter without changing any audit field.
#[test]
fn validation_read_accumulation_is_atomic_and_keeps_high_water() {
    let read = BurnpackPayloadReadStats {
        tensor_reads: 1,
        payload_bytes: 40,
        peak_owned_payload_bytes: 48,
    };
    let snapshot = |audit: &BurnpackValidationAudit| {
        [
            audit.tensor_reads,
            audit.tensors_validated as u64,
            audit.payload_bytes,
            audit.largest_payload_bytes,
            audit.peak_owned_payload_bytes,
        ]
    };
    for (reads, bytes, context) in [
        (u64::MAX, 0, "Burnpack validation tensor reads"),
        (0, u64::MAX, "Burnpack validation payload bytes"),
    ] {
        let mut audit = BurnpackValidationAudit {
            tensor_reads: reads,
            tensors_validated: 7,
            payload_bytes: bytes,
            largest_payload_bytes: 20,
            peak_owned_payload_bytes: 32,
        };
        let before = snapshot(&audit);
        assert!(
            matches!(record_validation_read(&mut audit, read), Err(QuantizationError::ByteLengthOverflow { context: actual }) if actual == context)
        );
        assert_eq!(snapshot(&audit), before);
    }
    let mut audit = BurnpackValidationAudit::default();
    record_validation_read(&mut audit, read).unwrap();
    record_validation_read(
        &mut audit,
        BurnpackPayloadReadStats {
            payload_bytes: 8,
            peak_owned_payload_bytes: 12,
            ..read
        },
    )
    .unwrap();
    assert_eq!(snapshot(&audit), [2, 0, 48, 40, 48]);
}

/// Even an empty session budgets its manifest before extending the staging file.
#[test]
fn empty_finalize_staging_budget_failure_does_not_close_session() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("too-small.bpk");
    let mut session = QuantizedBurnpackSession::with_limits(
        &output,
        BurnpackSessionLimits {
            max_staged_bytes: 1,
            ..BurnpackSessionLimits::default()
        },
    )
    .unwrap();
    let before = session.stats();
    for _ in 0..2 {
        assert!(
            matches!(session.finalize(), Err(QuantizationError::SessionLimitExceeded { kind: "staged payload", attempted, limit: 1 }) if attempted > 1)
        );
        assert_eq!(session.stats(), before);
        assert_eq!(session.staging_len_for_test(), 0);
        assert!(!session.finalized);
        assert!(!output.exists());
    }
}

/// Lazy range snapshots re-seek their shared handle and surface truncation at materialization.
#[test]
fn staged_snapshot_bounds_and_allocation_errors_are_lazy_and_contextual() {
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(&[1_u8, 2, 3]).unwrap();
    let file = Rc::new(file);
    let path = Path::new("private-test-stage");
    let snapshot =
        staged_snapshot(Rc::clone(&file), path, "q.weight", 1, 2, ParamId::from(1)).unwrap();
    assert_eq!(snapshot.to_data().unwrap().as_bytes(), &[2, 3]);
    assert_eq!(snapshot.to_data().unwrap().as_bytes(), &[2, 3]);

    file.set_len(2).unwrap();
    assert!(
        matches!(snapshot.to_data(), Err(TensorSnapshotError::IoError(message)) if message.contains("failed to read 2 bytes") && message.contains("private-test-stage"))
    );
    let empty = staged_snapshot(Rc::clone(&file), path, "empty", 2, 0, ParamId::from(2)).unwrap();
    assert!(empty.to_data().unwrap().as_bytes().is_empty());

    // Vec rejects a capacity above isize::MAX before attempting allocation.
    let impossible = staged_snapshot(
        file,
        path,
        "overflow",
        0,
        usize::MAX as u64,
        ParamId::from(3),
    )
    .unwrap();
    assert!(
        matches!(impossible.to_data(), Err(TensorSnapshotError::DataError(message)) if message.contains("failed to reserve"))
    );
}

/// Read-only staging reports failed rollback and never commits metadata or publishes output.
#[test]
fn checkpoint_standards_read_only_staging_preserves_uncommitted_state() {
    for append_weight in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("failed-write.bpk");
        let mut session = QuantizedBurnpackSession::new(&output).unwrap();
        let read_only = tempfile::NamedTempFile::new().unwrap();
        session.staging_file = Some(File::open(read_only.path()).unwrap());
        let before = session.stats();
        let result = if append_weight {
            let (metadata, bytes) = packed_fixture();
            let (weight, _) = decode_payload("q.weight", metadata, &bytes)
                .unwrap_or_else(|failure| panic!("{}", failure.error));
            session.append_quantized("q.weight", weight).map(|_| ())
        } else {
            session.finalize().map(|_| ())
        };
        assert!(matches!(
            result,
            Err(QuantizationError::ArtifactIo {
                operation: "roll back quantization staging file",
                ..
            })
        ));
        assert_eq!(session.stats(), before);
        assert!(session.entries.is_empty());
        assert!(session.manifest.tensors.is_empty());
        assert!(!session.finalized);
        assert_eq!(read_only.as_file().metadata().unwrap().len(), 0);
        assert!(!output.exists());

        // A repaired descriptor can finalize the still-empty session normally.
        session.staging_file = Some(tempfile::tempfile().unwrap());
        let summary = session.finalize().unwrap();
        assert_eq!(summary.tensor_count, 0);
        assert!(output.is_file());
    }
}

/// Parent path failures stay contextual and cannot overwrite an unrelated ordinary file.
#[test]
fn checkpoint_standards_temporary_creation_rejects_non_directory_parent() {
    let root = tempfile::tempdir().unwrap();
    let parent = root.path().join("occupied-parent");
    fs::write(&parent, b"preserve me").unwrap();
    let output = parent.join("weights.bpk");
    for (error, operation) in [
        (
            create_unique_file(&output, "test-stage").unwrap_err(),
            "create private Burnpack temporary file",
        ),
        (
            create_unique_directory(&output, "test-output").unwrap_err(),
            "create private Burnpack temporary directory",
        ),
    ] {
        assert!(matches!(error, QuantizationError::ArtifactIo {
            operation: actual, ref path, ..
        } if actual == operation && path.parent() == Some(parent.as_path())));
    }
    assert_eq!(fs::read(&parent).unwrap(), b"preserve me");
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
}

/// A digest-valid payload with invalid numerical semantics is still rejected after copying.
#[test]
fn checkpoint_standards_decoder_rejects_authenticated_invalid_scales_and_codes() {
    let (base_metadata, base_bytes) = packed_fixture();
    for corrupt_scale in [false, true] {
        let mut metadata = base_metadata.clone();
        let mut bytes = base_bytes.clone();
        if corrupt_scale {
            let offset = metadata.packed_value_bytes as usize;
            bytes[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        } else {
            bytes[0] = (bytes[0] & 0xf0) | 0x08; // Reserved Q4 code -8.
        }
        metadata.payload_sha256 = Some(packed_payload_sha256(&bytes));
        let failure = decode_payload("q.weight", metadata, &bytes).unwrap_err();
        if corrupt_scale {
            assert!(
                matches!(failure.error, QuantizationError::InvalidScale { group: 0, scale }
                if scale.is_nan())
            );
        } else {
            assert!(matches!(
                failure.error,
                QuantizationError::InvalidQuantizedValue {
                    index: 0,
                    value: -8
                }
            ));
        }
        assert_eq!(failure.stats.tensor_reads, 1);
        assert_eq!(failure.stats.payload_bytes, bytes.len() as u64);
        assert!(failure.stats.peak_owned_payload_bytes >= bytes.len() as u64 + 8);
    }
}
