//! Tests artifact discovery, strict checkpoint mapping, alias topology, and memory accounting.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::Path,
};

use proptest::prelude::*;
use serde_json::{Map, Value, json};
use tempfile::TempDir;

use super::*;

/// Borrowed tensor descriptor whose payload bytes are synthesized by the writer.
#[derive(Clone)]
struct FixtureTensor<'a> {
    /// Fixture tensor name stored in the shard header.
    name: &'a str,
    /// Wire scalar format used to calculate payload bytes.
    dtype: TensorDType,
    /// Logical dimensions in row-major axis order.
    shape: Vec<u64>,
}

impl<'a> FixtureTensor<'a> {
    /// Creates a named tensor descriptor for a discovery or planning fixture.
    fn new(name: &'a str, dtype: TensorDType, shape: impl Into<Vec<u64>>) -> Self {
        Self {
            name,
            dtype,
            shape: shape.into(),
        }
    }

    /// Calculates the exact wire payload size for the fixture's shape and dtype.
    fn bytes(&self) -> u64 {
        let elements = self.shape.iter().product();
        self.dtype.storage_bytes(elements).unwrap()
    }
}

/// Writes a zero-filled SafeTensors fixture without user metadata.
fn write_shard(root: &Path, relative: &str, tensors: &[FixtureTensor<'_>]) {
    write_shard_with_metadata(root, relative, tensors, BTreeMap::new());
}

/// Writes deterministic aligned SafeTensors headers and zero-filled payloads.
fn write_shard_with_metadata(
    root: &Path,
    relative: &str,
    tensors: &[FixtureTensor<'_>],
    metadata: BTreeMap<String, String>,
) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let mut offset = 0_u64;
    let mut object = Map::new();
    if !metadata.is_empty() {
        object.insert(
            "__metadata__".into(),
            serde_json::to_value(metadata).unwrap(),
        );
    }
    for tensor in tensors {
        let end = offset + tensor.bytes();
        object.insert(
            tensor.name.into(),
            json!({
                "dtype": tensor.dtype.as_str(),
                "shape": tensor.shape,
                "data_offsets": [offset, end]
            }),
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&Value::Object(object)).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&vec![0_u8; offset as usize]).unwrap();
}

/// Writes supplied header bytes unchanged for malformed-artifact tests.
fn write_raw_shard(root: &Path, relative: &str, header: &[u8], data_bytes: usize) {
    let path = root.join(relative);
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header).unwrap();
    file.write_all(&vec![0_u8; data_bytes]).unwrap();
}

/// Writes tensor-to-shard assignments with an optional declared byte total.
fn write_index(root: &Path, map: &[(&str, &str)], total_size: Option<u64>) {
    let weight_map: BTreeMap<_, _> = map.iter().copied().collect();
    let value = json!({
        "metadata": { "total_size": total_size },
        "weight_map": weight_map,
    });
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
}

/// Creates and discovers a temporary single-shard fixture while retaining its directory.
fn fixture_manifest(tensors: &[FixtureTensor<'_>]) -> (TempDir, CheckpointManifest) {
    let temp = TempDir::new().unwrap();
    write_shard(temp.path(), "model.safetensors", tensors);
    let manifest = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap();
    (temp, manifest)
}

#[test]
/// Unsharded discovery validates descriptors and metadata without interpreting payload values.
fn discovers_unsharded_checkpoint_without_reading_payload_semantics() {
    let temp = TempDir::new().unwrap();
    let metadata = BTreeMap::from([("format".into(), "pt".into())]);
    let tensors = [
        FixtureTensor::new("empty", TensorDType::F16, [0, 7]),
        FixtureTensor::new("weight", TensorDType::F32, [2, 3]),
    ];
    write_shard_with_metadata(temp.path(), "model.safetensors", &tensors, metadata);

    let manifest = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap();
    assert!(matches!(manifest.source, ArtifactSource::Single { .. }));
    assert_eq!(manifest.tensors.len(), 2);
    assert_eq!(manifest.tensor_bytes, 24);
    assert_eq!(manifest.tensor("weight").unwrap().elements().unwrap(), 6);
    let shard = manifest.shard("model.safetensors").unwrap();
    assert_eq!(shard.metadata["format"], "pt");
    assert_eq!(
        shard
            .tensor_file_offset(manifest.tensor("weight").unwrap())
            .unwrap(),
        8 + shard.header_bytes
    );
}

#[test]
/// Sharded discovery verifies tensor assignments and the index's declared byte total.
fn discovers_shards_and_verifies_index_total_size() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "model-00001-of-00002.safetensors",
        &[FixtureTensor::new("a", TensorDType::F16, [2, 2])],
    );
    write_shard(
        temp.path(),
        "model-00002-of-00002.safetensors",
        &[FixtureTensor::new("b", TensorDType::BF16, [3])],
    );
    write_index(
        temp.path(),
        &[
            ("b", "model-00002-of-00002.safetensors"),
            ("a", "model-00001-of-00002.safetensors"),
        ],
        Some(14),
    );

    let manifest = CheckpointManifest::discover(temp.path(), &LoadOptions::default()).unwrap();
    assert_eq!(manifest.shards.len(), 2);
    assert_eq!(manifest.tensor_bytes, 14);
    assert_eq!(manifest.tensors.keys().collect::<Vec<_>>(), [&"a", &"b"]);
    assert!(matches!(
        manifest.source,
        ArtifactSource::Index {
            declared_total_size: Some(14),
            ..
        }
    ));
}

#[test]
/// Exact integer-valued decimal byte totals are accepted without floating-point rounding.
fn index_total_size_accepts_exact_decimal_compatibility_form() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "part.safetensors",
        &[FixtureTensor::new("a", TensorDType::F16, [1])],
    );
    fs::write(
        temp.path().join("model.safetensors.index.json"),
        br#"{"metadata":{"total_size":2.0},"weight_map":{"a":"part.safetensors"}}"#,
    )
    .unwrap();

    let manifest = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap();
    assert!(matches!(
        manifest.source,
        ArtifactSource::Index {
            declared_total_size: Some(2),
            ..
        }
    ));
}

#[test]
/// Ambiguous and nonintegral decimal byte totals are rejected.
fn index_total_size_rejects_ambiguous_or_non_integral_decimals() {
    for invalid in [
        "2.5",
        "-2.0",
        "2e0",
        "55562855904.000001",
        "18446744073709551616.0",
    ] {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("model.safetensors.index.json"),
            format!(
                r#"{{"metadata":{{"total_size":{invalid}}},"weight_map":{{"a":"missing.safetensors"}}}}"#
            ),
        )
        .unwrap();
        assert!(matches!(
            discover_checkpoint(temp.path(), &LoadOptions::default()),
            Err(CheckpointError::InvalidJson {
                kind: "checkpoint index",
                ..
            })
        ));
    }
}

#[test]
/// Explicit nested shard paths are accepted when canonical containment is preserved.
fn explicit_nested_shard_path_is_allowed_when_contained() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "weights/part.safetensors",
        &[FixtureTensor::new("a", TensorDType::F16, [1])],
    );
    write_index(temp.path(), &[("a", "weights/part.safetensors")], Some(2));
    let manifest = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap();
    assert_eq!(manifest.shards[0].relative_path, "weights/part.safetensors");
}

#[test]
/// Automatic discovery accepts a unique nonstandard artifact and rejects ambiguous candidates.
fn auto_discovery_accepts_one_nonstandard_artifact_and_rejects_ambiguity() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "weights.safetensors",
        &[FixtureTensor::new("a", TensorDType::F16, [1])],
    );
    assert!(discover_checkpoint(temp.path(), &LoadOptions::default()).is_ok());
    write_shard(
        temp.path(),
        "other.safetensors",
        &[FixtureTensor::new("b", TensorDType::F16, [1])],
    );
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::AmbiguousArtifacts { .. })
    ));
}

#[test]
/// Missing artifacts and roots that are not directories produce contextual errors.
fn missing_artifact_and_nondirectory_roots_are_contextual() {
    let temp = TempDir::new().unwrap();
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::ArtifactNotFound { .. })
    ));
    let file = temp.path().join("ordinary-file");
    fs::write(&file, b"x").unwrap();
    assert!(matches!(
        discover_checkpoint(file, &LoadOptions::default()),
        Err(CheckpointError::RootNotDirectory { .. })
    ));
}

#[test]
/// Portable path validation rejects traversal, absolute paths, and other hostile spellings.
fn hostile_relative_paths_are_rejected_portably() {
    for path in [
        "../outside.safetensors",
        "./model.safetensors",
        "/tmp/model.safetensors",
        "C:/model.safetensors",
        "dir\\model.safetensors",
        "nul\0model.safetensors",
        "",
    ] {
        let temp = TempDir::new().unwrap();
        let options = LoadOptions {
            artifact: ArtifactSelection::Single(path.into()),
            ..LoadOptions::default()
        };
        assert!(
            matches!(
                discover_checkpoint(temp.path(), &options),
                Err(CheckpointError::UnsafeRelativePath { .. })
            ),
            "path unexpectedly accepted: {path:?}"
        );
    }
}

#[cfg(unix)]
#[test]
/// A shard symlink cannot escape the canonical artifact root.
fn shard_symlink_cannot_escape_canonical_root() {
    use std::os::unix::fs::symlink;

    let root = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();
    write_shard(
        outside.path(),
        "outside.safetensors",
        &[FixtureTensor::new("a", TensorDType::F16, [1])],
    );
    symlink(
        outside.path().join("outside.safetensors"),
        root.path().join("model.safetensors"),
    )
    .unwrap();
    assert!(matches!(
        discover_checkpoint(root.path(), &LoadOptions::default()),
        Err(CheckpointError::PathEscapesRoot { .. })
    ));
}

#[test]
/// Header and index resource limits reject oversized inputs before allocation.
fn header_and_index_resource_limits_are_enforced_before_allocation() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "model.safetensors",
        &[FixtureTensor::new("a", TensorDType::F16, [1])],
    );
    let mut options = LoadOptions::default();
    options.limits.max_header_bytes = 1;
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::LimitExceeded {
            kind: "SafeTensors header byte",
            ..
        })
    ));

    fs::remove_file(temp.path().join("model.safetensors")).unwrap();
    fs::write(temp.path().join("model.safetensors.index.json"), b"{}").unwrap();
    options = LoadOptions::default();
    options.limits.max_index_bytes = 1;
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::LimitExceeded {
            kind: "checkpoint index byte",
            ..
        })
    ));
}

#[test]
/// Truncated files and impossible declared header lengths fail discovery.
fn truncated_and_impossible_header_lengths_are_rejected() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("model.safetensors"), b"tiny").unwrap();
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::TruncatedFile { .. })
    ));

    let mut bytes = 100_u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    fs::write(temp.path().join("model.safetensors"), bytes).unwrap();
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::InvalidHeaderLength { .. })
    ));
}

#[test]
/// Malformed tensor dtypes, dimensions, and payload ranges fail validation.
fn malformed_dtype_shape_and_range_metadata_are_rejected() {
    let cases = [
        (
            br#"{"a":{"dtype":"FP16","shape":[1],"data_offsets":[0,2]}}"#.as_slice(),
            2,
            "unsupported dtype",
        ),
        (
            br#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#.as_slice(),
            4,
            "byte mismatch",
        ),
        (
            br#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[1,3]}}"#.as_slice(),
            3,
            "leading hole",
        ),
        (
            br#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#.as_slice(),
            3,
            "trailing byte",
        ),
    ];
    for (header, data_bytes, label) in cases {
        let temp = TempDir::new().unwrap();
        write_raw_shard(temp.path(), "model.safetensors", header, data_bytes);
        assert!(
            discover_checkpoint(temp.path(), &LoadOptions::default()).is_err(),
            "malformed case accepted: {label}"
        );
    }
}

#[test]
/// SafeTensors headers must begin with the JSON object marker.
fn safetensors_header_must_start_with_an_object_byte() {
    let temp = TempDir::new().unwrap();
    write_raw_shard(temp.path(), "model.safetensors", b" {}", 0);
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::InvalidJson {
            kind: "SafeTensors header",
            ..
        })
    ));
}

#[test]
/// Duplicate decoded tensor names cannot overwrite an earlier header entry.
fn duplicate_header_tensor_names_are_rejected() {
    let temp = TempDir::new().unwrap();
    let header = br#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
    write_raw_shard(temp.path(), "model.safetensors", header, 2);
    let error = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap_err();
    assert!(matches!(error, CheckpointError::DuplicateJsonKey { .. }));
    assert!(error.to_string().contains("duplicate"));
}

#[test]
/// Every index entry and every shard tensor must have a consistent counterpart.
fn index_must_match_headers_in_both_directions() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "part.safetensors",
        &[FixtureTensor::new("actual", TensorDType::F16, [1])],
    );
    write_index(temp.path(), &[("declared", "part.safetensors")], Some(2));
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::IndexMismatch { .. })
    ));
}

#[test]
/// Generic index totals must equal the validated tensor payload byte sum.
fn index_total_size_must_equal_validated_tensor_bytes() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "part.safetensors",
        &[FixtureTensor::new("a", TensorDType::F16, [1])],
    );
    write_index(temp.path(), &[("a", "part.safetensors")], Some(99));
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::TotalSizeMismatch {
            declared: 99,
            observed: 2
        })
    ));
}

#[test]
/// Pinned byte-accounting exceptions are explicit and do not weaken generic discovery.
fn pinned_total_size_policy_is_explicit_and_generic_paths_remain_strict() {
    use sha2::{Digest, Sha256};

    use super::accounting::{
        DecimalLayerTensorSelector, PinnedAccountingTensor, PinnedControlFile,
        PinnedIndexTotalSizePolicy,
    };

    let temp = TempDir::new().unwrap();
    let base_router = "backbone.layers.0.mixer.gate.weight";
    let mtp_router = "mtp.layers.1.mixer.gate.weight";
    write_shard(
        temp.path(),
        "part.safetensors",
        &[
            FixtureTensor::new(base_router, TensorDType::BF16, [1]),
            FixtureTensor::new(mtp_router, TensorDType::BF16, [1]),
        ],
    );
    write_index(
        temp.path(),
        &[
            (base_router, "part.safetensors"),
            (mtp_router, "part.safetensors"),
        ],
        Some(6),
    );
    let config = b"{}";
    fs::write(temp.path().join("config.json"), config).unwrap();
    let index = fs::read(temp.path().join("model.safetensors.index.json")).unwrap();
    let policy = PinnedIndexTotalSizePolicy {
        supplied_repository: "test/pinned".into(),
        supplied_revision: "0123456789abcdef".into(),
        expected_repository: "test/pinned".into(),
        expected_revision: "0123456789abcdef".into(),
        config: PinnedControlFile::new(
            "config.json",
            config.len() as u64,
            format!("{:x}", Sha256::digest(config)),
        ),
        index: PinnedControlFile::new(
            "model.safetensors.index.json",
            index.len() as u64,
            format!("{:x}", Sha256::digest(&index)),
        ),
        expected_physical_tensor_bytes: 4,
        expected_declared_total_size: 6,
        duplicated_tensor_selector: DecimalLayerTensorSelector::new(
            "backbone.layers.",
            ".mixer.gate.weight",
        ),
        duplicated_tensors: vec![PinnedAccountingTensor::new(
            base_router,
            TensorDType::BF16,
            vec![1],
        )],
        excluded_tensors: vec![PinnedAccountingTensor::new(
            mtp_router,
            TensorDType::BF16,
            vec![1],
        )],
    };

    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::TotalSizeMismatch {
            declared: 6,
            observed: 4
        })
    ));
    let manifest = super::artifact::discover_checkpoint_with_index_total_size_policy(
        temp.path(),
        &LoadOptions::default(),
        &policy,
    )
    .unwrap();
    assert_eq!(manifest.tensor_bytes, 4);
    assert_eq!(manifest.indexed_tensor_bytes, 4);

    // Policy application must reject an edited artifact kind before reading controls.
    for (source, reason) in [
        (
            ArtifactSource::Single {
                relative_path: "part.safetensors".into(),
            },
            "pinned index accounting cannot apply to an unsharded artifact",
        ),
        (
            ArtifactSource::Index {
                relative_path: "model.safetensors.index.json".into(),
                declared_total_size: None,
            },
            "pinned accounting requires index metadata.total_size",
        ),
    ] {
        let mut edited = manifest.clone();
        edited.source = source;
        assert!(matches!(
            super::artifact::validate_manifest_index_total_size_policy(&edited, &policy),
            Err(CheckpointError::IndexTotalSizePolicyMismatch { reason: actual })
                if actual == reason
        ));
    }

    // Authentic control files cannot excuse an internally inconsistent policy.
    let mut empty_policy = policy.clone();
    empty_policy.duplicated_tensors.clear();
    let mut overlapping_policy = policy.clone();
    overlapping_policy.excluded_tensors = overlapping_policy.duplicated_tensors.clone();
    for (edited_policy, fragment) in [
        (empty_policy, "duplicate-counted tensor set is empty"),
        (overlapping_policy, "also listed as duplicate-counted"),
    ] {
        assert!(matches!(
            super::artifact::validate_manifest_index_total_size_policy(&manifest, &edited_policy),
            Err(CheckpointError::IndexTotalSizePolicyMismatch { reason })
                if reason.contains(fragment)
        ));
    }
    let mut wrong_total_policy = policy.clone();
    wrong_total_policy.expected_declared_total_size = 8;
    let mut wrong_total_manifest = manifest.clone();
    wrong_total_manifest.source = ArtifactSource::Index {
        relative_path: "model.safetensors.index.json".into(),
        declared_total_size: Some(8),
    };
    assert!(matches!(
        super::artifact::validate_manifest_index_total_size_policy(
            &wrong_total_manifest, &wrong_total_policy
        ),
        Err(CheckpointError::IndexTotalSizePolicyMismatch { reason })
            if reason.contains("equal 6, not declared bytes 8")
    ));

    let expected = [
        ExpectedTensor::new(base_router, [1]),
        ExpectedTensor::new(mtp_router, [1]),
    ];
    assert!(matches!(
        build_load_plan(
            &manifest,
            &expected,
            &MappingPlan::identity(&expected),
            &LoadOptions::default(),
        ),
        Err(CheckpointError::TotalSizeMismatch {
            declared: 6,
            observed: 4
        })
    ));
    let plan = super::mapping::build_load_plan_with_index_total_size_policy(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
        &policy,
    )
    .unwrap();
    assert_eq!(plan.tensors.len(), 2);
    assert!(plan.report.is_strictly_valid());
}

#[test]
/// Explicit and unique nonstandard indexes can omit optional total-size metadata.
fn explicit_index_and_nonstandard_index_auto_discovery_work_without_total_size() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "part.safetensors",
        &[FixtureTensor::new("a", TensorDType::F16, [1])],
    );
    let custom_index = temp.path().join("weights.safetensors.index.json");
    fs::write(&custom_index, br#"{"weight_map":{"a":"part.safetensors"}}"#).unwrap();
    let auto = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap();
    assert!(matches!(
        auto.source,
        ArtifactSource::Index {
            declared_total_size: None,
            ..
        }
    ));

    let explicit = LoadOptions {
        artifact: ArtifactSelection::Index("weights.safetensors.index.json".into()),
        ..LoadOptions::default()
    };
    assert!(discover_checkpoint(temp.path(), &explicit).is_ok());
}

#[test]
/// Index size and entry limits are enforced before opening referenced shards.
fn index_limits_apply_before_any_shard_is_opened() {
    let temp = TempDir::new().unwrap();
    write_index(
        temp.path(),
        &[("a", "a.safetensors"), ("b", "b.safetensors")],
        None,
    );
    let mut options = LoadOptions::default();
    options.limits.max_shards = 1;
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::LimitExceeded { kind: "shard", .. })
    ));

    options.limits.max_shards = 2;
    options.limits.max_tensors = 1;
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::LimitExceeded { kind: "tensor", .. })
    ));
}

#[test]
/// Selected artifacts require supported extensions and regular-file semantics.
fn artifact_extensions_and_regular_file_types_are_enforced() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("weights.bin"), b"not tensors").unwrap();
    let options = LoadOptions {
        artifact: ArtifactSelection::Single("weights.bin".into()),
        ..LoadOptions::default()
    };
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::UnsafeRelativePath { .. })
    ));

    fs::create_dir(temp.path().join("directory.safetensors")).unwrap();
    let options = LoadOptions {
        artifact: ArtifactSelection::Single("directory.safetensors".into()),
        ..LoadOptions::default()
    };
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::UnsafeRelativePath { .. })
    ));

    write_index(temp.path(), &[("a", "not-a-shard.bin")], None);
    let options = LoadOptions {
        artifact: ArtifactSelection::Index("model.safetensors.index.json".into()),
        ..LoadOptions::default()
    };
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::UnsafeRelativePath { .. })
    ));

    fs::create_dir(temp.path().join("directory-shard.safetensors")).unwrap();
    write_index(temp.path(), &[("a", "directory-shard.safetensors")], None);
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::UnsafeRelativePath { .. })
    ));
}

#[test]
/// Multiple nonstandard indexes produce an ambiguity error.
fn multiple_nonstandard_indexes_are_ambiguous() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("a.safetensors.index.json"), b"{}").unwrap();
    fs::write(temp.path().join("b.safetensors.index.json"), b"{}").unwrap();
    let error = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap_err();
    let CheckpointError::AmbiguousArtifacts { candidates } = error else {
        panic!("expected ambiguous indexes")
    };
    assert_eq!(
        candidates,
        ["a.safetensors.index.json", "b.safetensors.index.json"]
    );
}

#[test]
/// Incorrect shard assignments and declared-but-absent tensors are rejected.
fn index_assignment_mismatch_and_declared_but_absent_tensor_are_rejected() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "one.safetensors",
        &[
            FixtureTensor::new("a", TensorDType::F16, [1]),
            FixtureTensor::new("one", TensorDType::F16, [1]),
        ],
    );
    write_shard(
        temp.path(),
        "two.safetensors",
        &[FixtureTensor::new("two", TensorDType::F16, [1])],
    );
    write_index(
        temp.path(),
        &[
            ("a", "two.safetensors"),
            ("one", "one.safetensors"),
            ("two", "two.safetensors"),
        ],
        None,
    );
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::IndexMismatch { .. })
    ));

    let temp = TempDir::new().unwrap();
    write_shard(temp.path(), "empty.safetensors", &[]);
    write_index(temp.path(), &[("ghost", "empty.safetensors")], None);
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::IndexMismatch { tensor, .. }) if tensor == "ghost"
    ));
}

#[test]
/// An exact same-shard index alias resolves to its physical owner without inventing payload storage.
fn exact_same_shard_index_alias_resolves_without_inventing_a_tensor() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "part.safetensors",
        &[FixtureTensor::new("shared.weight", TensorDType::F16, [1])],
    );
    write_index(
        temp.path(),
        &[
            ("shared.weight", "part.safetensors"),
            ("encoder.embed_tokens.weight", "part.safetensors"),
        ],
        Some(4),
    );
    let options = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new(
            "encoder.embed_tokens.weight",
            "shared.weight",
        )],
        ..LoadOptions::default()
    };

    let manifest = discover_checkpoint(temp.path(), &options).unwrap();

    assert_eq!(
        manifest.tensors.keys().collect::<Vec<_>>(),
        [&"shared.weight"]
    );
    assert_eq!(manifest.tensor_bytes, 2);
    assert_eq!(manifest.indexed_tensor_bytes, 4);
    assert_eq!(
        manifest.index_alias_size_accounting,
        Some(IndexAliasSizeAccounting::Logical)
    );
    assert_eq!(
        manifest.index_aliases,
        [ResolvedIndexTensorAlias::new(
            "encoder.embed_tokens.weight",
            "shared.weight",
            "part.safetensors"
        )]
    );
    let expected = [ExpectedTensor::new("shared.weight", [1])];
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions {
            allow_unexpected: vec![KeyMatcher::exact("encoder.embed_tokens.weight")],
            index_aliases: options.index_aliases.clone(),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    assert!(plan.report.allowlisted_ignored.is_empty());
    assert_eq!(plan.report.checkpoint_aliases, manifest.index_aliases);
    assert!(plan.report.equivalence_planned.is_empty());

    assert!(
        build_load_plan(
            &manifest,
            &expected,
            &MappingPlan::identity(&expected),
            &LoadOptions {
                index_aliases: options.index_aliases.clone(),
                ..LoadOptions::default()
            }
        )
        .is_ok()
    );

    let mut unauthorized = manifest.clone();
    unauthorized.index_aliases[0].storage_key = "other.weight".into();
    assert!(matches!(
        build_load_plan(
            &unauthorized,
            &expected,
            &MappingPlan::identity(&expected),
            &LoadOptions {
                allow_unexpected: vec![KeyMatcher::exact("encoder.embed_tokens.weight")],
                index_aliases: options.index_aliases.clone(),
                ..LoadOptions::default()
            }
        ),
        Err(CheckpointError::InvalidIndexAlias { alias, .. })
            if alias == "encoder.embed_tokens.weight"
    ));

    let planning_options = LoadOptions {
        allow_unexpected: vec![KeyMatcher::exact("encoder.embed_tokens.weight")],
        index_aliases: options.index_aliases.clone(),
        ..LoadOptions::default()
    };
    let mut missing_owner = manifest.clone();
    missing_owner.tensors.remove("shared.weight");
    assert!(matches!(
        build_load_plan(
            &missing_owner,
            &expected,
            &MappingPlan::identity(&expected),
            &planning_options,
        ),
        Err(CheckpointError::InvalidIndexAlias { alias, .. })
            if alias == "encoder.embed_tokens.weight"
    ));

    let mut physical_alias = manifest.clone();
    let mut duplicate = physical_alias.tensors["shared.weight"].clone();
    duplicate.name = "encoder.embed_tokens.weight".into();
    physical_alias
        .tensors
        .insert(duplicate.name.clone(), duplicate);
    assert!(matches!(
        build_load_plan(
            &physical_alias,
            &expected,
            &MappingPlan::identity(&expected),
            &planning_options,
        ),
        Err(CheckpointError::InvalidIndexAlias { alias, .. })
            if alias == "encoder.embed_tokens.weight"
    ));

    let mut wrong_shard = manifest.clone();
    wrong_shard.index_aliases[0].shard = "other.safetensors".into();
    assert!(matches!(
        build_load_plan(
            &wrong_shard,
            &expected,
            &MappingPlan::identity(&expected),
            &planning_options,
        ),
        Err(CheckpointError::InvalidIndexAlias { alias, .. })
            if alias == "encoder.embed_tokens.weight"
    ));

    let mut wrong_indexed_bytes = manifest.clone();
    wrong_indexed_bytes.indexed_tensor_bytes = 2;
    assert!(matches!(
        build_load_plan(
            &wrong_indexed_bytes,
            &expected,
            &MappingPlan::identity(&expected),
            &planning_options,
        ),
        Err(CheckpointError::ManifestAccountingMismatch { .. })
    ));

    let contradictory_accounting = LoadOptions {
        index_alias_size_accounting: IndexAliasSizeAccounting::Physical,
        ..planning_options.clone()
    };
    assert!(matches!(
        build_load_plan(
            &manifest,
            &expected,
            &MappingPlan::identity(&expected),
            &contradictory_accounting,
        ),
        Err(CheckpointError::ManifestAccountingMismatch { .. })
    ));

    write_index(
        temp.path(),
        &[
            ("shared.weight", "part.safetensors"),
            ("encoder.embed_tokens.weight", "part.safetensors"),
        ],
        Some(2),
    );
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::TotalSizeMismatch {
            declared: 2,
            observed: 4
        })
    ));
    let physical_options = LoadOptions {
        index_aliases: options.index_aliases.clone(),
        index_alias_size_accounting: IndexAliasSizeAccounting::Physical,
        ..LoadOptions::default()
    };
    let physical = discover_checkpoint(temp.path(), &physical_options).unwrap();
    assert_eq!(physical.indexed_tensor_bytes, 2);
    assert_eq!(
        physical.index_alias_size_accounting,
        Some(IndexAliasSizeAccounting::Physical)
    );
}

#[test]
/// Index-alias options round-trip and preserve compatibility defaults when omitted.
fn index_alias_options_round_trip_and_default_when_absent() {
    let options = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new("alias.weight", "owner.weight")],
        ..LoadOptions::default()
    };
    let value = serde_json::to_value(&options).unwrap();
    assert_eq!(
        serde_json::from_value::<LoadOptions>(value).unwrap(),
        options
    );

    let mut legacy = serde_json::to_value(LoadOptions::default()).unwrap();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("index_aliases")
        .unwrap();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("index_alias_size_accounting")
        .unwrap();
    let decoded = serde_json::from_value::<LoadOptions>(legacy).unwrap();
    assert!(decoded.index_aliases.is_empty());
    assert_eq!(
        decoded.index_alias_size_accounting,
        IndexAliasSizeAccounting::Logical
    );
}

#[test]
/// Index aliases require a unique exact owner in the same physical shard.
fn index_alias_requires_exact_unique_same_shard_physical_storage() {
    let make = || {
        let temp = TempDir::new().unwrap();
        write_shard(
            temp.path(),
            "one.safetensors",
            &[FixtureTensor::new("shared.weight", TensorDType::F16, [1])],
        );
        write_shard(
            temp.path(),
            "two.safetensors",
            &[FixtureTensor::new("sentinel", TensorDType::F16, [1])],
        );
        temp
    };

    let temp = make();
    write_index(
        temp.path(),
        &[
            ("shared.weight", "one.safetensors"),
            ("sentinel", "two.safetensors"),
            ("alias.weight", "two.safetensors"),
        ],
        Some(4),
    );
    let options = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new("alias.weight", "shared.weight")],
        ..LoadOptions::default()
    };
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::InvalidIndexAlias { alias, .. }) if alias == "alias.weight"
    ));

    let temp = make();
    write_index(
        temp.path(),
        &[
            ("shared.weight", "one.safetensors"),
            ("sentinel", "two.safetensors"),
            ("alias.weight", "one.safetensors"),
        ],
        Some(4),
    );
    let options = LoadOptions {
        index_aliases: vec![
            IndexTensorAlias::new("alias.weight", "shared.weight"),
            IndexTensorAlias::new("alias.weight", "sentinel"),
        ],
        ..LoadOptions::default()
    };
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::InvalidIndexAlias { alias, .. }) if alias == "alias.weight"
    ));

    for invalid in [
        IndexTensorAlias::new("", "shared.weight"),
        IndexTensorAlias::new("alias.weight", ""),
        IndexTensorAlias::new("shared.weight", "shared.weight"),
    ] {
        let temp = make();
        write_index(
            temp.path(),
            &[
                ("shared.weight", "one.safetensors"),
                ("sentinel", "two.safetensors"),
            ],
            Some(4),
        );
        let options = LoadOptions {
            index_aliases: vec![invalid.clone()],
            ..LoadOptions::default()
        };
        assert!(matches!(
            discover_checkpoint(temp.path(), &options),
            Err(CheckpointError::InvalidIndexAlias { .. })
        ));
    }
}

#[test]
/// A physical tensor name cannot occur in two shards.
fn tensor_cannot_appear_in_two_shards() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "one.safetensors",
        &[
            FixtureTensor::new("duplicate", TensorDType::F16, [1]),
            FixtureTensor::new("one", TensorDType::F16, [1]),
        ],
    );
    write_shard(
        temp.path(),
        "two.safetensors",
        &[
            FixtureTensor::new("duplicate", TensorDType::F16, [1]),
            FixtureTensor::new("two", TensorDType::F16, [1]),
        ],
    );
    write_index(
        temp.path(),
        &[
            ("duplicate", "one.safetensors"),
            ("one", "one.safetensors"),
            ("two", "two.safetensors"),
        ],
        None,
    );
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::DuplicateTensor { tensor, .. }) if tensor == "duplicate"
    ));
}

#[test]
/// Malformed index objects and duplicate decoded weight keys fail deterministically.
fn malformed_index_json_and_duplicate_weight_keys_are_rejected() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("model.safetensors.index.json"), b"{").unwrap();
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::InvalidJson {
            kind: "checkpoint index",
            ..
        })
    ));

    fs::write(
        temp.path().join("model.safetensors.index.json"),
        br#"{"weight_map":{"a":"one.safetensors","a":"two.safetensors"}}"#,
    )
    .unwrap();
    let error = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap_err();
    assert!(matches!(error, CheckpointError::DuplicateJsonKey { .. }));
    assert!(error.to_string().contains("duplicate"));
}

/// Checks the exact map-type diagnostic while retaining a bounded parser location.
fn assert_unique_map_type_error(
    error: CheckpointError,
    kind: &'static str,
    path: &Path,
    actual_type: &str,
    json_bytes: usize,
) {
    let CheckpointError::InvalidJson {
        kind: actual_kind,
        path: actual_path,
        message,
    } = error
    else {
        panic!("expected contextual JSON type error, found {error}");
    };
    assert_eq!(actual_kind, kind);
    assert_eq!(actual_path, fs::canonicalize(path).unwrap());
    let (detail, location) = message
        .split_once(" at line ")
        .expect("retained parser coordinates");
    assert_eq!(
        detail,
        format!("invalid type: {actual_type}, expected an object with unique string keys")
    );
    let column = location
        .strip_prefix("1 column ")
        .expect("single-line fixture location")
        .parse::<usize>()
        .unwrap();
    assert!((1..=json_bytes).contains(&column), "{message}");
}

/// Header metadata must be a string map, never a null/scalar/sequence masquerading as one.
#[test]
fn artifact_entrypoint_rejects_non_map_header_metadata_without_writes() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("model.safetensors");
    write_shard_with_metadata(
        temp.path(),
        "model.safetensors",
        &[FixtureTensor::new("weight", TensorDType::F16, [1])],
        BTreeMap::from([("format".into(), "pt".into())]),
    );
    let valid = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap();
    assert_eq!(valid.tensor_bytes, 2);
    assert_eq!(
        valid.shards[0].metadata.get("format").map(String::as_str),
        Some("pt")
    );
    for (value, actual_type) in [
        (Value::Null, "null"),
        (json!([]), "sequence"),
        (json!(true), "boolean `true`"),
        (json!(42), "integer `42`"),
    ] {
        let header = serde_json::to_vec(&json!({
            "__metadata__": value,
            "weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        }))
        .unwrap();
        write_raw_shard(temp.path(), "model.safetensors", &header, 2);
        let before = fs::read(&path).unwrap();
        let error = CheckpointManifest::discover(temp.path(), &LoadOptions::default()).unwrap_err();
        assert_unique_map_type_error(
            error,
            "SafeTensors header",
            &path,
            actual_type,
            header.len(),
        );
        assert_eq!(fs::read(&path).unwrap(), before);
    }
}

/// Index storage ownership must be an object; rejected map types cannot touch shard payloads.
#[test]
fn artifact_entrypoint_rejects_non_map_index_inventory_without_writes() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "part.safetensors",
        &[FixtureTensor::new("weight", TensorDType::F16, [1])],
    );
    write_index(temp.path(), &[("weight", "part.safetensors")], Some(2));
    let path = temp.path().join("model.safetensors.index.json");
    let shard = temp.path().join("part.safetensors");
    let before_shard = fs::read(&shard).unwrap();
    let valid = CheckpointManifest::discover(temp.path(), &LoadOptions::default()).unwrap();
    assert_eq!(valid.tensor_bytes, 2);
    assert_eq!(valid.shards[0].relative_path, "part.safetensors");
    for (value, actual_type) in [
        (Value::Null, "null"),
        (json!([]), "sequence"),
        (json!(true), "boolean `true`"),
        (json!(42), "integer `42`"),
    ] {
        let index =
            serde_json::to_vec(&json!({"metadata": {"total_size": 2}, "weight_map": value}))
                .unwrap();
        fs::write(&path, &index).unwrap();
        let error = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap_err();
        assert_unique_map_type_error(error, "checkpoint index", &path, actual_type, index.len());
        assert_eq!(fs::read(&path).unwrap(), index);
        assert_eq!(fs::read(&shard).unwrap(), before_shard);
    }
}

/// An alias cannot introduce storage that is absent from the authoritative index.
#[test]
fn artifact_contract_alias_storage_must_be_indexed() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "one.safetensors",
        &[FixtureTensor::new("sentinel", TensorDType::F16, [1])],
    );
    write_index(
        temp.path(),
        &[
            ("a.alias", "one.safetensors"),
            ("sentinel", "one.safetensors"),
        ],
        None,
    );
    let shard = temp.path().join("one.safetensors");
    let index = temp.path().join("model.safetensors.index.json");
    let before = (fs::read(&shard).unwrap(), fs::read(&index).unwrap());
    let options = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new("a.alias", "z.storage")],
        ..LoadOptions::default()
    };
    assert!(
        matches!(discover_checkpoint(temp.path(), &options), Err(CheckpointError::InvalidIndexAlias { alias, reason })
        if alias == "a.alias" && reason == "storage key `z.storage` is absent from weight_map")
    );
    assert_eq!((fs::read(shard).unwrap(), fs::read(index).unwrap()), before);
}

/// Declaring another alias as storage must not make an absent physical tensor real.
#[test]
fn artifact_contract_alias_storage_must_exist_in_the_shard_header() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "one.safetensors",
        &[FixtureTensor::new("sentinel", TensorDType::F16, [1])],
    );
    write_index(
        temp.path(),
        &[
            ("a.alias", "one.safetensors"),
            ("z.storage", "one.safetensors"),
            ("sentinel", "one.safetensors"),
        ],
        None,
    );
    let shard = temp.path().join("one.safetensors");
    let before = fs::read(&shard).unwrap();
    // An alias chain is not a substitute for a direct physical-storage binding.
    for aliases in [
        vec![IndexTensorAlias::new("a.alias", "z.storage")],
        vec![
            IndexTensorAlias::new("a.alias", "z.storage"),
            IndexTensorAlias::new("z.storage", "sentinel"),
        ],
    ] {
        let options = LoadOptions {
            index_aliases: aliases,
            ..LoadOptions::default()
        };
        assert!(
            matches!(discover_checkpoint(temp.path(), &options), Err(CheckpointError::InvalidIndexAlias { alias, reason })
            if alias == "a.alias" && reason == "storage key `z.storage` is not present in its assigned shard header")
        );
        assert_eq!(fs::read(&shard).unwrap(), before);
    }
}

/// A forbidden shard extension is rejected before any attempt to open that path.
#[test]
fn artifact_contract_index_rejects_non_safetensors_storage() {
    let temp = TempDir::new().unwrap();
    write_index(temp.path(), &[("weight", "absent.bin")], None);
    let index = temp.path().join("model.safetensors.index.json");
    let before = fs::read(&index).unwrap();
    assert!(
        matches!(discover_checkpoint(temp.path(), &LoadOptions::default()), Err(CheckpointError::UnsafeRelativePath { path, reason })
        if path == "absent.bin" && reason == "tensor shard must end in .safetensors")
    );
    assert!(!temp.path().join("absent.bin").exists());
    assert_eq!(fs::read(index).unwrap(), before);
}

/// JSON escapes cannot disguise duplicate index-map identities.
#[test]
fn artifact_contract_index_rejects_escape_equivalent_tensor_names() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("model.safetensors.index.json");
    let bytes = br#"{"weight_map":{"weight":"one.safetensors","\u0077eight":"two.safetensors"}}"#;
    fs::write(&path, bytes).unwrap();
    assert!(
        matches!(discover_checkpoint(temp.path(), &LoadOptions::default()), Err(CheckpointError::DuplicateJsonKey { kind, path: actual, key })
        if kind == "checkpoint index" && actual == fs::canonicalize(&path).unwrap() && key == "weight")
    );
    assert_eq!(fs::read(path).unwrap(), bytes);
}

/// Duplicate metadata identities are rejected after escape decoding in either map.
#[test]
fn artifact_contract_header_rejects_escape_equivalent_metadata_names() {
    let temp = TempDir::new().unwrap();
    for (header, key) in [
        (
            br#"{"__metadata__":{},"\u005f_metadata__":{}}"#.as_slice(),
            "__metadata__",
        ),
        (
            br#"{"__metadata__":{"format":"pt","\u0066ormat":"other"}}"#.as_slice(),
            "format",
        ),
    ] {
        write_raw_shard(temp.path(), "model.safetensors", header, 0);
        let path = temp.path().join("model.safetensors");
        let before = fs::read(&path).unwrap();
        assert!(
            matches!(discover_checkpoint(temp.path(), &LoadOptions::default()), Err(CheckpointError::DuplicateJsonKey { kind, path: actual, key: actual_key })
            if kind == "SafeTensors header" && actual == fs::canonicalize(&path).unwrap() && actual_key == key)
        );
        assert_eq!(fs::read(path).unwrap(), before);
    }
}

#[test]
/// Tensor count, rank, name length, and checked-arithmetic limits are enforced.
fn all_header_count_rank_name_and_arithmetic_bounds_are_enforced() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "model.safetensors",
        &[
            FixtureTensor::new("a", TensorDType::F16, [1]),
            FixtureTensor::new("b", TensorDType::F16, [1]),
        ],
    );
    let mut options = LoadOptions::default();
    options.limits.max_tensors = 1;
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::LimitExceeded { kind: "tensor", .. })
    ));

    let temp = TempDir::new().unwrap();
    write_shard_with_metadata(
        temp.path(),
        "model.safetensors",
        &[],
        BTreeMap::from([("a".into(), "1".into()), ("b".into(), "2".into())]),
    );
    options = LoadOptions::default();
    options.limits.max_metadata_entries = 1;
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::LimitExceeded {
            kind: "SafeTensors metadata entry",
            ..
        })
    ));

    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "model.safetensors",
        &[FixtureTensor::new("ranked", TensorDType::F16, [1, 1])],
    );
    options = LoadOptions::default();
    options.limits.max_rank = 1;
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::LimitExceeded {
            kind: "tensor rank",
            ..
        })
    ));

    let temp = TempDir::new().unwrap();
    write_raw_shard(
        temp.path(),
        "model.safetensors",
        br#"{"":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#,
        2,
    );
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::InvalidTensorName { shard, tensor, .. })
        if shard == "model.safetensors" && tensor.is_empty()
    ));

    let temp = TempDir::new().unwrap();
    write_raw_shard(
        temp.path(),
        "model.safetensors",
        br#"{"huge":{"dtype":"F16","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#,
        0,
    );
    let error = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap_err();
    assert!(matches!(
        &error,
        CheckpointError::InvalidTensorHeader { shard, tensor, source }
        if shard == "model.safetensors" && tensor == "huge"
            && matches!(**source, CheckpointError::ArithmeticOverflow { .. })
    ));
    assert!(error.to_string().contains("`huge` in `model.safetensors`"));
}

/// A malformed header, its data size, and a predicate on the wrapped inner error.
type HeaderCase = (&'static [u8], usize, fn(&CheckpointError) -> bool);

/// Header-entry failures name their shard and tensor around the violated rule.
#[test]
fn header_entry_errors_name_their_shard_and_tensor() {
    let cases: [HeaderCase; 3] = [
        (
            br#"{"w":{"dtype":"F8_E4M3FNUZ","shape":[1],"data_offsets":[0,1]}}"#,
            1,
            |error| matches!(error, CheckpointError::UnsupportedDType { dtype } if dtype == "F8_E4M3FNUZ"),
        ),
        (
            br#"{"w":{"dtype":"F4","shape":[3],"data_offsets":[0,2]}}"#,
            2,
            |error| {
                matches!(
                    error,
                    CheckpointError::MisalignedTensorStorage { elements: 3, .. }
                )
            },
        ),
        (
            br#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
            4,
            |error| {
                matches!(
                    error,
                    CheckpointError::TensorByteLengthMismatch {
                        expected: 8,
                        actual: 4,
                        ..
                    }
                )
            },
        ),
    ];
    for (header, data_bytes, inner) in cases {
        let temp = TempDir::new().unwrap();
        write_raw_shard(temp.path(), "model.safetensors", header, data_bytes);
        let error = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap_err();
        let CheckpointError::InvalidTensorHeader {
            shard,
            tensor,
            source,
        } = &error
        else {
            panic!("expected contextual header error, found {error}");
        };
        assert_eq!(
            (shard.as_str(), tensor.as_str()),
            ("model.safetensors", "w")
        );
        assert!(inner(source), "unexpected inner error {source}");
        assert!(std::error::Error::source(&error).is_some());
    }
}

#[test]
/// Reversed payload ranges and ranges outside the file produce distinct diagnostics.
fn reversed_and_out_of_file_ranges_are_distinguished() {
    let temp = TempDir::new().unwrap();
    write_raw_shard(
        temp.path(),
        "model.safetensors",
        br#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[2,0]}}"#,
        2,
    );
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::InvalidDataOffsets { .. })
    ));

    let temp = TempDir::new().unwrap();
    write_raw_shard(
        temp.path(),
        "model.safetensors",
        br#"{"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#,
        2,
    );
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::InvalidDataOffsets { .. })
    ));
}

#[test]
/// Duplicate metadata keys and absent roots fail without ambiguous interpretation.
fn duplicate_metadata_keys_and_missing_roots_fail_cleanly() {
    let temp = TempDir::new().unwrap();
    write_raw_shard(
        temp.path(),
        "model.safetensors",
        br#"{"__metadata__":{"format":"pt","format":"rust"}}"#,
        0,
    );
    let error = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap_err();
    assert!(matches!(error, CheckpointError::DuplicateJsonKey { .. }));
    assert!(error.to_string().contains("duplicate"));

    let absent = temp.path().join("does-not-exist");
    assert!(matches!(
        discover_checkpoint(&absent, &LoadOptions::default()),
        Err(CheckpointError::Io { operation: "inspect checkpoint root", path, source })
            if path == absent && source.kind() == std::io::ErrorKind::NotFound
    ));
    assert!(!absent.exists());
}

#[test]
/// An identity plan records exact tensor transformations and memory estimates.
fn strict_identity_plan_and_memory_estimate_are_exact() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("a", TensorDType::F16, [2, 3]),
        FixtureTensor::new("b", TensorDType::F32, [2]),
    ]);
    let expected = [
        ExpectedTensor::new("a", [2, 3]).with_dtype(TensorDType::F16),
        ExpectedTensor::new("b", [2]).with_dtype(TensorDType::F32),
    ];
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap();
    assert_eq!(plan.parameter_count(), 2);
    assert_eq!(plan.memory.checkpoint_tensor_bytes, 20);
    assert_eq!(plan.memory.target_parameter_bytes, 20);
    assert_eq!(plan.memory.logical_model_bytes, 20);
    assert_eq!(plan.report.memory, Some(plan.memory));
    assert!(plan.report.transformed.is_empty());
}

#[test]
/// Each supported mapping operation retains its explicit source/target transformation.
fn rename_transpose_split_concatenate_and_alias_are_explicit() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("matrix", TensorDType::F16, [2, 3]),
        FixtureTensor::new("fused", TensorDType::F16, [6, 2]),
        FixtureTensor::new("left", TensorDType::F16, [2, 2]),
        FixtureTensor::new("right", TensorDType::F16, [1, 2]),
    ]);
    let expected = [
        ExpectedTensor::new("renamed_transposed", [3, 2]),
        ExpectedTensor::new("q", [2, 2]),
        ExpectedTensor::new("k", [1, 2]),
        ExpectedTensor::new("v", [3, 2]),
        ExpectedTensor::new("joined", [3, 2]),
        ExpectedTensor::new("joined_tied", [3, 2]),
    ];
    let mut mappings = MappingPlan::new();
    mappings
        .transpose_2d("matrix", "renamed_transposed")
        .split(
            "fused",
            0,
            [("q".into(), 2), ("k".into(), 1), ("v".into(), 3)],
        )
        .concatenate(["left".into(), "right".into()], "joined", 0)
        .alias("joined", "joined_tied");
    let plan = build_load_plan(&manifest, &expected, &mappings, &LoadOptions::default()).unwrap();

    assert_eq!(plan.tensors.len(), 5);
    assert_eq!(plan.aliases.len(), 1);
    assert_eq!(plan.report.transformed.len(), 5);
    assert_eq!(plan.report.aliased.len(), 1);
    assert!(matches!(
        plan.tensors
            .iter()
            .find(|item| item.target == "v")
            .unwrap()
            .transform,
        PlannedTransform::Slice {
            start: 3,
            length: 3,
            ..
        }
    ));
    assert!(plan.memory.logical_model_bytes > plan.memory.target_parameter_bytes);
}

#[test]
/// Alias chains resolve to a single physical storage owner.
fn chained_aliases_resolve_to_the_storage_owner() {
    let (_temp, manifest) =
        fixture_manifest(&[FixtureTensor::new("shared", TensorDType::F16, [4, 2])]);
    let expected = [
        ExpectedTensor::new("embedding", [4, 2]),
        ExpectedTensor::new("head", [4, 2]),
        ExpectedTensor::new("decoder_head", [4, 2]),
    ];
    let mut mappings = MappingPlan::new();
    mappings
        .map("shared", "embedding")
        .alias("embedding", "head")
        .alias("head", "decoder_head");
    let plan = build_load_plan(&manifest, &expected, &mappings, &LoadOptions::default()).unwrap();
    for alias in &plan.aliases {
        assert!(matches!(
            &alias.transform,
            PlannedTransform::Alias { source_target } if source_target == "embedding"
        ));
    }
}

#[test]
/// Automatic dtype policy records the required BF16-to-F16 conversion.
fn auto_dtype_policy_plans_bf16_to_f16_conversion() {
    let (_temp, manifest) =
        fixture_manifest(&[FixtureTensor::new("weight", TensorDType::BF16, [2, 2])]);
    let expected = [ExpectedTensor::new("weight", [2, 2]).with_dtype(TensorDType::F16)];
    let mapping = MappingPlan::identity(&expected);
    let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
    assert_eq!(plan.tensors[0].dtype, TensorDType::F16);
    assert_eq!(plan.report.dtype_converted.len(), 1);
}

#[test]
/// Preserve and explicit-cast policies honor architecture dtype constraints.
fn preserve_and_cast_dtype_policies_are_strict() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("float", TensorDType::BF16, [2]),
        FixtureTensor::new("integer", TensorDType::I64, [1]),
    ]);
    let expected = [
        ExpectedTensor::new("float", [2]).with_dtype(TensorDType::F16),
        ExpectedTensor::new("integer", [1]).with_dtype(TensorDType::I64),
    ];
    let mappings = MappingPlan::identity(&expected);
    let preserve = LoadOptions {
        dtype_policy: DTypePolicy::Preserve,
        ..LoadOptions::default()
    };
    assert!(matches!(
        build_load_plan(&manifest, &expected, &mappings, &preserve),
        Err(CheckpointError::DTypeMismatch { .. })
    ));

    let generic = [
        ExpectedTensor::new("float", [2]),
        ExpectedTensor::new("integer", [1]),
    ];
    let cast = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F32),
        ..LoadOptions::default()
    };
    let plan =
        build_load_plan(&manifest, &generic, &MappingPlan::identity(&generic), &cast).unwrap();
    assert_eq!(
        plan.tensors
            .iter()
            .find(|t| t.target == "float")
            .unwrap()
            .dtype,
        TensorDType::F32
    );
    assert_eq!(
        plan.tensors
            .iter()
            .find(|t| t.target == "integer")
            .unwrap()
            .dtype,
        TensorDType::I64
    );

    let invalid_cast = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::I8),
        ..LoadOptions::default()
    };
    assert!(matches!(
        build_load_plan(
            &manifest,
            &generic,
            &MappingPlan::identity(&generic),
            &invalid_cast
        ),
        Err(CheckpointError::InvalidMapping { .. })
    ));
}

/// Concrete architecture dtypes take precedence over a generic float cast.
#[test]
fn concrete_expected_dtype_overrides_blanket_float_cast() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("fixed", TensorDType::F32, [2]),
        FixtureTensor::new("generic", TensorDType::BF16, [2]),
    ]);
    let expected = [
        ExpectedTensor::new("fixed", [2]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("generic", [2]),
    ];
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        ..LoadOptions::default()
    };
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    assert_eq!(
        plan.tensors
            .iter()
            .find(|tensor| tensor.target == "fixed")
            .unwrap()
            .dtype,
        TensorDType::F32
    );
    assert_eq!(
        plan.tensors
            .iter()
            .find(|tensor| tensor.target == "generic")
            .unwrap()
            .dtype,
        TensorDType::F16
    );
}

#[test]
/// Strict mismatch reports retain all missing and unexpected tensor keys.
fn strict_mismatch_report_preserves_all_keys() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("used", TensorDType::F16, [1]),
        FixtureTensor::new("vision.block", TensorDType::F16, [1]),
        FixtureTensor::new("surprise", TensorDType::F16, [1]),
    ]);
    let expected = [
        ExpectedTensor::new("used", [1]),
        ExpectedTensor::new("optional", [1]),
        ExpectedTensor::new("missing", [1]),
    ];
    let mut mappings = MappingPlan::new();
    mappings.map("used", "used");
    let options = LoadOptions {
        allow_missing: vec![KeyMatcher::exact("optional")],
        allow_unexpected: vec![KeyMatcher::prefix("vision.")],
        ..LoadOptions::default()
    };
    let error = build_load_plan(&manifest, &expected, &mappings, &options).unwrap_err();
    let CheckpointError::StrictValidation { report, .. } = error else {
        panic!("unexpected error")
    };
    assert_eq!(report.allowlisted_missing, ["optional"]);
    assert_eq!(report.allowlisted_ignored, ["vision.block"]);
    assert_eq!(report.missing, ["missing"]);
    assert_eq!(report.unexpected, ["surprise"]);
}

#[test]
/// Explicit allowlists permit an intentional partial load to satisfy strict validation.
fn allowlists_can_make_an_intentionally_partial_text_load_strict() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("text.weight", TensorDType::F16, [1]),
        FixtureTensor::new("visual.weight", TensorDType::F16, [1]),
    ]);
    let expected = [
        ExpectedTensor::new("text.weight", [1]),
        ExpectedTensor::new("optional.bias", [1]),
    ];
    let mut mappings = MappingPlan::new();
    mappings.map("text.weight", "text.weight");
    let options = LoadOptions {
        allow_missing: vec![KeyMatcher::prefix("optional.")],
        allow_unexpected: vec![KeyMatcher::prefix("visual.")],
        ..LoadOptions::default()
    };
    let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
    assert!(plan.report.is_strictly_valid());
}

#[test]
/// Empty allowlist patterns cannot silently permit every checkpoint key.
fn empty_allowlist_patterns_are_forbidden() {
    let (_temp, manifest) = fixture_manifest(&[]);
    let options = LoadOptions {
        allow_unexpected: vec![KeyMatcher::prefix("")],
        ..LoadOptions::default()
    };
    assert!(matches!(
        build_load_plan(&manifest, &[], &MappingPlan::new(), &options),
        Err(CheckpointError::InvalidMapping { .. })
    ));
}

#[test]
/// Mapping validation rejects incompatible shapes, axes, and source consumption.
fn invalid_mapping_shapes_axes_and_consumption_are_rejected() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("rank1", TensorDType::F16, [6]),
        FixtureTensor::new("matrix", TensorDType::F16, [2, 3]),
    ]);

    let expected = [ExpectedTensor::new("target", [3, 2])];
    let mut wrong_rank = MappingPlan::new();
    wrong_rank.transpose_2d("rank1", "target");
    assert!(matches!(
        build_load_plan(&manifest, &expected, &wrong_rank, &LoadOptions::default()),
        Err(CheckpointError::InvalidMapping { .. })
    ));

    let expected = [ExpectedTensor::new("target", [1, 1])];
    let mut mismatch = MappingPlan::new();
    mismatch.map("matrix", "target");
    assert!(matches!(
        build_load_plan(&manifest, &expected, &mismatch, &LoadOptions::default()),
        Err(CheckpointError::ShapeMismatch { .. })
    ));

    let expected = [ExpectedTensor::new("a", [6]), ExpectedTensor::new("b", [6])];
    let mut reused = MappingPlan::new();
    reused.map("rank1", "a").map("rank1", "b");
    assert!(matches!(
        build_load_plan(&manifest, &expected, &reused, &LoadOptions::default()),
        Err(CheckpointError::MappingSourceCollision { .. })
    ));
}

#[test]
/// Malformed split and concatenation requests fail planning.
fn malformed_split_and_concatenate_operations_are_rejected() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("a", TensorDType::F16, [2, 3]),
        FixtureTensor::new("b", TensorDType::F32, [2, 3]),
        FixtureTensor::new("c", TensorDType::F16, [3, 4]),
    ]);
    let expected = [
        ExpectedTensor::new("x", [1, 3]),
        ExpectedTensor::new("y", [1, 3]),
    ];
    let mut bad_split = MappingPlan::new();
    bad_split.split("a", 0, [("x".into(), 1), ("y".into(), 2)]);
    assert!(matches!(
        build_load_plan(&manifest, &expected, &bad_split, &LoadOptions::default()),
        Err(CheckpointError::InvalidMapping { .. })
    ));

    let expected = [ExpectedTensor::new("joined", [4, 3])];
    let mut mixed = MappingPlan::new();
    mixed.concatenate(["a".into(), "b".into()], "joined", 0);
    assert!(matches!(
        build_load_plan(&manifest, &expected, &mixed, &LoadOptions::default()),
        Err(CheckpointError::InvalidMapping { .. })
    ));

    let mut incompatible = MappingPlan::new();
    incompatible.concatenate(["a".into(), "c".into()], "joined", 0);
    assert!(matches!(
        build_load_plan(&manifest, &expected, &incompatible, &LoadOptions::default()),
        Err(CheckpointError::InvalidMapping { .. })
    ));
}

#[test]
/// Cyclic aliases and aliases without known owners fail planning.
fn alias_cycles_and_unknown_alias_sources_are_rejected() {
    let (_temp, manifest) = fixture_manifest(&[]);
    let expected = [ExpectedTensor::new("a", [1]), ExpectedTensor::new("b", [1])];
    let mut cycle = MappingPlan::new();
    cycle.alias("b", "a").alias("a", "b");
    assert!(matches!(
        build_load_plan(&manifest, &expected, &cycle, &LoadOptions::default()),
        Err(CheckpointError::InvalidMapping { .. })
    ));

    let mut dangling = MappingPlan::new();
    dangling.alias("nowhere", "a");
    assert!(matches!(
        build_load_plan(&manifest, &expected, &dangling, &LoadOptions::default()),
        Err(CheckpointError::InvalidMapping { .. })
    ));
}

#[test]
/// Unknown sources/targets and duplicate expected tensors fail before execution.
fn missing_sources_unknown_targets_and_duplicate_expectations_fail_early() {
    let (_temp, manifest) = fixture_manifest(&[]);
    let expected = [ExpectedTensor::new("known", [1])];
    let mut missing = MappingPlan::new();
    missing.map("absent", "known");
    assert!(matches!(
        build_load_plan(&manifest, &expected, &missing, &LoadOptions::default()),
        Err(CheckpointError::MissingMappingSource { .. })
    ));

    let (_temp, manifest) =
        fixture_manifest(&[FixtureTensor::new("source", TensorDType::F16, [1])]);
    let mut unknown = MappingPlan::new();
    unknown.map("source", "unknown");
    assert!(matches!(
        build_load_plan(&manifest, &expected, &unknown, &LoadOptions::default()),
        Err(CheckpointError::UnknownMappingTarget { .. })
    ));

    let duplicate = [
        ExpectedTensor::new("same", [1]),
        ExpectedTensor::new("same", [1]),
    ];
    assert!(matches!(
        build_load_plan(
            &manifest,
            &duplicate,
            &MappingPlan::new(),
            &LoadOptions::default()
        ),
        Err(CheckpointError::MappingTargetCollision { .. })
    ));
}

#[test]
/// Splits and concatenations validate empty input sets and axis bounds.
fn split_and_concatenate_validate_empty_inputs_and_axes() {
    let (_temp, manifest) =
        fixture_manifest(&[FixtureTensor::new("source", TensorDType::F16, [2, 3])]);
    let expected = [ExpectedTensor::new("target", [2, 3])];

    let mut split_axis = MappingPlan::new();
    split_axis.split("source", 2, [("target".into(), 1)]);
    assert!(matches!(
        build_load_plan(&manifest, &expected, &split_axis, &LoadOptions::default()),
        Err(CheckpointError::InvalidMapping { .. })
    ));

    let mut empty_split = MappingPlan::new();
    empty_split.split("source", 0, Vec::<(String, u64)>::new());
    assert!(matches!(
        build_load_plan(&manifest, &expected, &empty_split, &LoadOptions::default()),
        Err(CheckpointError::InvalidMapping { .. })
    ));

    let mut empty_concat = MappingPlan::new();
    empty_concat.concatenate(Vec::<String>::new(), "target", 0);
    assert!(matches!(
        build_load_plan(&manifest, &expected, &empty_concat, &LoadOptions::default()),
        Err(CheckpointError::InvalidMapping { .. })
    ));

    let mut concat_axis = MappingPlan::new();
    concat_axis.concatenate(["source".into()], "target", 2);
    assert!(matches!(
        build_load_plan(&manifest, &expected, &concat_axis, &LoadOptions::default()),
        Err(CheckpointError::InvalidMapping { .. })
    ));
}

#[test]
/// Every mapping operation rejects collisions with an existing target.
fn all_mapping_target_collision_paths_are_rejected() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("first", TensorDType::F16, [1]),
        FixtureTensor::new("second", TensorDType::F16, [1]),
    ]);
    let expected = [ExpectedTensor::new("target", [1])];
    let mut direct_collision = MappingPlan::new();
    direct_collision
        .map("first", "target")
        .map("second", "target");
    assert!(matches!(
        build_load_plan(
            &manifest,
            &expected,
            &direct_collision,
            &LoadOptions::default()
        ),
        Err(CheckpointError::MappingTargetCollision { .. })
    ));

    let mut alias_collision = MappingPlan::new();
    alias_collision
        .map("first", "target")
        .alias("target", "target");
    assert!(matches!(
        build_load_plan(
            &manifest,
            &expected,
            &alias_collision,
            &LoadOptions::default()
        ),
        Err(CheckpointError::MappingTargetCollision { .. })
    ));

    let mut unknown_alias = MappingPlan::new();
    unknown_alias.alias("target", "not_expected");
    assert!(matches!(
        build_load_plan(
            &manifest,
            &expected,
            &unknown_alias,
            &LoadOptions::default()
        ),
        Err(CheckpointError::UnknownMappingTarget { .. })
    ));
}

#[test]
/// Aliases must match their storage owner's logical shape and dtype.
fn aliases_must_match_owner_shape_and_dtype() {
    let (_temp, manifest) =
        fixture_manifest(&[FixtureTensor::new("source", TensorDType::F16, [2])]);
    let mut mappings = MappingPlan::new();
    mappings.map("source", "owner").alias("owner", "alias");

    let shape_mismatch = [
        ExpectedTensor::new("owner", [2]),
        ExpectedTensor::new("alias", [3]),
    ];
    assert!(matches!(
        build_load_plan(
            &manifest,
            &shape_mismatch,
            &mappings,
            &LoadOptions::default()
        ),
        Err(CheckpointError::ShapeMismatch { .. })
    ));

    let dtype_mismatch = [
        ExpectedTensor::new("owner", [2]).with_dtype(TensorDType::F16),
        ExpectedTensor::new("alias", [2]).with_dtype(TensorDType::F32),
    ];
    let preserve = LoadOptions {
        dtype_policy: DTypePolicy::Preserve,
        ..LoadOptions::default()
    };
    assert!(matches!(
        build_load_plan(&manifest, &dtype_mismatch, &mappings, &preserve),
        Err(CheckpointError::DTypeMismatch { .. })
    ));
}

#[test]
/// Expected inventories reject unnamed tensors and overflowing shape products.
fn expected_inventory_rejects_empty_names_and_shape_overflow() {
    let (_temp, manifest) = fixture_manifest(&[]);
    let empty = [ExpectedTensor::new("", [1])];
    assert!(matches!(
        build_load_plan(
            &manifest,
            &empty,
            &MappingPlan::new(),
            &LoadOptions::default()
        ),
        Err(CheckpointError::InvalidMapping { .. })
    ));

    let overflowing = [ExpectedTensor::new("huge", [u64::MAX, 2])];
    assert!(matches!(
        build_load_plan(
            &manifest,
            &overflowing,
            &MappingPlan::new(),
            &LoadOptions::default()
        ),
        Err(CheckpointError::ArithmeticOverflow { .. })
    ));
}

#[test]
/// Explicit operation insertion and runtime report recording preserve their public contracts.
fn explicit_operation_push_and_report_runtime_records_are_available() {
    let (_temp, manifest) =
        fixture_manifest(&[FixtureTensor::new("source", TensorDType::F16, [1])]);
    let expected = [ExpectedTensor::new("target", [1])];
    let mut mappings = MappingPlan::new();
    mappings.push(TensorOperation::Map {
        source: "source".into(),
        target: "target".into(),
        transform: TensorTransform::Identity,
    });
    assert_eq!(mappings.operations().len(), 1);
    let mut plan =
        build_load_plan(&manifest, &expected, &mappings, &LoadOptions::default()).unwrap();
    plan.report.record_applied("target");
    plan.report.record_applied("target");
    plan.report.record_quantized(QuantizationRecord {
        target: "target".into(),
        scheme: "symmetric-q8".into(),
        bytes: 1,
    });
    plan.report.finalize();
    assert_eq!(plan.report.applied, ["target"]);
    assert_eq!(plan.report.quantized.len(), 1);
}

/// Requires the exact planner error variant and complete diagnostic context.
#[track_caller]
fn assert_mapping_contract_error(actual: CheckpointError, expected: CheckpointError) {
    assert_eq!(
        core::mem::discriminant(&actual),
        core::mem::discriminant(&expected)
    );
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

/// Discovers a real indexed alias before exercising public-manifest revalidation.
fn indexed_mapping_contract_fixture() -> (TempDir, CheckpointManifest, LoadOptions) {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "part.safetensors",
        &[FixtureTensor::new("shared.weight", TensorDType::F16, [1])],
    );
    write_index(
        temp.path(),
        &[
            ("shared.weight", "part.safetensors"),
            ("alias.weight", "part.safetensors"),
        ],
        Some(4),
    );
    let options = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new("alias.weight", "shared.weight")],
        ..LoadOptions::default()
    };
    let manifest = discover_checkpoint(temp.path(), &options).unwrap();
    assert!(plan_mapping_contract_fixture(&manifest, &options).is_ok());
    (temp, manifest, options)
}

/// Plans one owner while consuming its authorized index-only alias.
fn plan_mapping_contract_fixture(
    manifest: &CheckpointManifest,
    options: &LoadOptions,
) -> Result<LoadPlan> {
    let expected = [ExpectedTensor::new("shared.weight", [1])];
    build_load_plan(
        manifest,
        &expected,
        &MappingPlan::identity(&expected),
        options,
    )
}

/// Public manifest edits cannot detach an alias from its proven shard inventory.
#[test]
fn mapping_contract_revalidates_alias_shard_provenance() {
    let (_temp, manifest, options) = indexed_mapping_contract_fixture();
    let mut empty_shard = manifest.clone();
    empty_shard.index_aliases[0].shard.clear();
    let mut absent_shard = manifest.clone();
    absent_shard.shards.clear();
    let mut missing_inventory = manifest.clone();
    missing_inventory.shards[0].tensors.clear();
    let mut mismatched_inventory = manifest.clone();
    mismatched_inventory.shards[0]
        .tensors
        .get_mut("shared.weight")
        .unwrap()
        .shape = vec![2];
    for (edited, reason) in [
        (empty_shard, "resolved shard provenance must not be empty"),
        (
            absent_shard,
            "resolved shard `part.safetensors` is absent from the manifest",
        ),
        (
            missing_inventory,
            "storage key `shared.weight` does not match its resolved shard inventory",
        ),
        (
            mismatched_inventory,
            "storage key `shared.weight` does not match its resolved shard inventory",
        ),
    ] {
        let before = edited.clone();
        assert_mapping_contract_error(
            plan_mapping_contract_fixture(&edited, &options).unwrap_err(),
            CheckpointError::InvalidIndexAlias {
                alias: "alias.weight".into(),
                reason: reason.into(),
            },
        );
        assert_eq!(edited, before);
    }
    assert!(
        plan_mapping_contract_fixture(&manifest, &options)
            .unwrap()
            .report
            .is_strictly_valid()
    );
}

/// Index-only metadata cannot survive a public change to single-file selection.
#[test]
fn mapping_contract_rejects_index_metadata_on_unsharded_sources() {
    let (_temp, mut manifest, options) = indexed_mapping_contract_fixture();
    manifest.source = ArtifactSource::Single {
        relative_path: "part.safetensors".into(),
    };
    assert_mapping_contract_error(
        plan_mapping_contract_fixture(&manifest, &options).unwrap_err(),
        CheckpointError::ManifestAccountingMismatch {
            reason: "an unsharded manifest cannot contain index-only aliases".into(),
        },
    );
    manifest.index_aliases.clear();
    assert_mapping_contract_error(
        plan_mapping_contract_fixture(&manifest, &options).unwrap_err(),
        CheckpointError::ManifestAccountingMismatch {
            reason: "an unsharded manifest cannot record index alias accounting".into(),
        },
    );
}

/// Accounting provenance is rechecked before malformed architecture expectations.
#[test]
fn mapping_contract_revalidates_accounting_before_target_shapes() {
    let (_temp, manifest, options) = indexed_mapping_contract_fixture();
    let mut wrong_total = manifest.clone();
    wrong_total.tensor_bytes = 3;
    let mut absent_convention = manifest.clone();
    absent_convention.index_alias_size_accounting = None;
    for (edited, reason) in [
        (
            wrong_total,
            "manifest records 3 physical tensor bytes, inventory sums to 2",
        ),
        (
            absent_convention,
            "an indexed manifest is missing its alias-size convention",
        ),
    ] {
        let error = build_load_plan(
            &edited,
            &[ExpectedTensor::new("", [u64::MAX, 2])],
            &MappingPlan::new(),
            &options,
        )
        .unwrap_err();
        assert_mapping_contract_error(
            error,
            CheckpointError::ManifestAccountingMismatch {
                reason: reason.into(),
            },
        );
    }
}

/// Mutable physical-alias inventories cannot conceal a changed payload length.
#[test]
fn mapping_contract_equivalence_rechecks_physical_alias_lengths() {
    let (_temp, mut manifest) = fixture_manifest(&[
        FixtureTensor::new("shared.weight", TensorDType::F16, [1]),
        FixtureTensor::new("alias.weight", TensorDType::F16, [1]),
    ]);
    let options = LoadOptions {
        index_aliases: vec![IndexTensorAlias::new("alias.weight", "shared.weight")],
        ..LoadOptions::default()
    };
    assert_eq!(
        plan_mapping_contract_fixture(&manifest, &options)
            .unwrap()
            .equivalences
            .len(),
        1
    );
    manifest.tensors.get_mut("alias.weight").unwrap().byte_len = 1;
    assert_mapping_contract_error(
        plan_mapping_contract_fixture(&manifest, &options).unwrap_err(),
        CheckpointError::TensorEquivalenceMismatch {
            alias: "alias.weight".into(),
            owner: "shared.weight".into(),
            reason: "payload length 1 differs from owner length 2".into(),
        },
    );
}

/// Later split destinations preserve shape, dtype, and name failure context.
#[test]
fn mapping_contract_split_validates_each_target_before_publishing_a_plan() {
    let (_temp, manifest) =
        fixture_manifest(&[FixtureTensor::new("fused", TensorDType::BF16, [2, 3])]);
    let mut mapping = MappingPlan::new();
    mapping.split("fused", 0, [("left".into(), 1), ("right".into(), 1)]);
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Preserve,
        ..LoadOptions::default()
    };
    let left = ExpectedTensor::new("left", [1, 3]);
    for (expected, error) in [
        (
            vec![left.clone()],
            CheckpointError::UnknownMappingTarget {
                tensor: "right".into(),
            },
        ),
        (
            vec![left.clone(), ExpectedTensor::new("right", [1, 2])],
            CheckpointError::ShapeMismatch {
                tensor: "right".into(),
                expected: vec![1, 2],
                observed: vec![1, 3],
            },
        ),
        (
            vec![
                left.clone(),
                ExpectedTensor::new("right", [1, 3]).with_dtype(TensorDType::F16),
            ],
            CheckpointError::DTypeMismatch {
                tensor: "right".into(),
                observed: TensorDType::BF16,
                expected: TensorDType::F16,
            },
        ),
    ] {
        assert_mapping_contract_error(
            build_load_plan(&manifest, &expected, &mapping, &options).unwrap_err(),
            error,
        );
    }
    let mut collision = MappingPlan::new();
    collision.split("fused", 0, [("left".into(), 1), ("left".into(), 1)]);
    assert_mapping_contract_error(
        build_load_plan(&manifest, std::slice::from_ref(&left), &collision, &options).unwrap_err(),
        CheckpointError::MappingTargetCollision {
            tensor: "left".into(),
        },
    );
    let plan = build_load_plan(
        &manifest,
        &[left, ExpectedTensor::new("right", [1, 3])],
        &mapping,
        &options,
    )
    .unwrap();
    assert_eq!(
        plan.tensors[1].transform,
        PlannedTransform::Slice {
            axis: 0,
            start: 1,
            length: 1
        }
    );
    assert_eq!(plan.memory.target_parameter_bytes, 12);
}

/// Concatenation checks its produced shape and dtype without losing source order.
#[test]
fn mapping_contract_concatenation_validates_destination_and_preserves_order() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("a", TensorDType::F16, [1, 3]),
        FixtureTensor::new("z", TensorDType::F16, [1, 3]),
    ]);
    let mut mapping = MappingPlan::new();
    mapping.concatenate(["z".into(), "a".into()], "joined", 0);
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Preserve,
        ..LoadOptions::default()
    };
    for (expected, error) in [
        (
            vec![],
            CheckpointError::UnknownMappingTarget {
                tensor: "joined".into(),
            },
        ),
        (
            vec![ExpectedTensor::new("joined", [3, 2])],
            CheckpointError::ShapeMismatch {
                tensor: "joined".into(),
                expected: vec![3, 2],
                observed: vec![2, 3],
            },
        ),
        (
            vec![ExpectedTensor::new("joined", [2, 3]).with_dtype(TensorDType::F32)],
            CheckpointError::DTypeMismatch {
                tensor: "joined".into(),
                observed: TensorDType::F16,
                expected: TensorDType::F32,
            },
        ),
    ] {
        assert_mapping_contract_error(
            build_load_plan(&manifest, &expected, &mapping, &options).unwrap_err(),
            error,
        );
    }
    let plan = build_load_plan(
        &manifest,
        &[ExpectedTensor::new("joined", [2, 3])],
        &mapping,
        &options,
    )
    .unwrap();
    assert_eq!(
        plan.tensors[0]
            .sources
            .iter()
            .map(|source| source.name.as_str())
            .collect::<Vec<_>>(),
        ["z", "a"]
    );
    assert_eq!(
        plan.tensors[0].transform,
        PlannedTransform::Concatenate { axis: 0 }
    );
}

/// Empty SafeTensors axes expose dimension-sum overflow without large payloads.
#[cfg(target_pointer_width = "64")]
#[test]
fn mapping_contract_empty_tensor_concatenation_checks_dimension_overflow() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("wide", TensorDType::F16, [0, u64::MAX]),
        FixtureTensor::new("one", TensorDType::F16, [0, 1]),
    ]);
    assert_eq!(manifest.tensor_bytes, 0);
    let mut mapping = MappingPlan::new();
    mapping.concatenate(["wide".into(), "one".into()], "joined", 1);
    assert_mapping_contract_error(
        build_load_plan(
            &manifest,
            &[ExpectedTensor::new("joined", [0, 0])],
            &mapping,
            &LoadOptions::default(),
        )
        .unwrap_err(),
        CheckpointError::ArithmeticOverflow {
            context: "summing concatenated tensor dimensions",
        },
    );
}

/// Per-target precision overrides survive splitting and subsequent tied aliases.
#[test]
fn mapping_contract_split_casts_respect_mixed_precision_and_alias_storage() {
    let (_temp, manifest) =
        fixture_manifest(&[FixtureTensor::new("fused", TensorDType::BF16, [2, 2])]);
    let expected = [
        ExpectedTensor::new("half", [1, 2]).with_dtype(TensorDType::F16),
        ExpectedTensor::new("precise", [1, 2]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("tied", [1, 2]).with_dtype(TensorDType::F32),
    ];
    let mut mapping = MappingPlan::new();
    mapping
        .split("fused", 0, [("half".into(), 1), ("precise".into(), 1)])
        .alias("precise", "tied");
    let plan = build_load_plan(
        &manifest,
        &expected,
        &mapping,
        &LoadOptions {
            dtype_policy: DTypePolicy::Cast(TensorDType::F16),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        plan.tensors
            .iter()
            .map(|tensor| tensor.dtype)
            .collect::<Vec<_>>(),
        [TensorDType::F16, TensorDType::F32]
    );
    assert_eq!(plan.aliases[0].dtype, TensorDType::F32);
    assert_eq!(
        plan.aliases[0].transform,
        PlannedTransform::Alias {
            source_target: "precise".into()
        }
    );
    assert_eq!(plan.memory.target_parameter_bytes, 12);
    assert_eq!(plan.memory.logical_model_bytes, 20);
    assert_eq!(plan.report.dtype_converted.len(), 2);
    assert_eq!(plan.report.aliased.len(), 1);
}

/// A small real destination retains writes and aliases in publication order.
#[derive(Default)]
struct CheckpointContractSink {
    /// Only successfully installed targets are represented here.
    tensors: BTreeMap<String, std::sync::Arc<burn::tensor::TensorData>>,
    /// Distinguishes writes from tied aliases and detects premature publication.
    events: Vec<String>,
}

impl TensorSink for CheckpointContractSink {
    /// I/O-style failure used when an alias cannot resolve its installed owner.
    type Error = std::io::Error;

    /// Stores a completed target; a retry may replace an earlier complete value.
    fn write_tensor(
        &mut self,
        target: &str,
        data: burn::tensor::TensorData,
    ) -> std::io::Result<()> {
        self.tensors
            .insert(target.into(), std::sync::Arc::new(data));
        self.events.push(format!("write:{target}"));
        Ok(())
    }

    /// Shares the owner's allocation only after it exists in the destination.
    fn alias_tensor(&mut self, target: &str, owner: &str) -> std::io::Result<()> {
        let data = self
            .tensors
            .get(owner)
            .cloned()
            .ok_or_else(|| std::io::Error::other("missing alias owner"))?;
        self.tensors.insert(target.into(), data);
        self.events.push(format!("alias:{target}:{owner}"));
        Ok(())
    }
}

/// A lost later shard preserves exact strict diagnostics and supports disk repair.
#[test]
fn checkpoint_contract_missing_second_shard_keeps_partial_report_then_retries() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "a.safetensors",
        &[FixtureTensor::new("first", TensorDType::F32, [1])],
    );
    write_shard(
        temp.path(),
        "z.safetensors",
        &[
            FixtureTensor::new("second", TensorDType::F32, [1]),
            FixtureTensor::new("vision.weight", TensorDType::F32, [1]),
        ],
    );
    write_index(
        temp.path(),
        &[
            ("first", "a.safetensors"),
            ("second", "z.safetensors"),
            ("vision.weight", "z.safetensors"),
        ],
        Some(12),
    );
    let options = LoadOptions {
        allow_missing: vec![KeyMatcher::exact("optional")],
        allow_unexpected: vec![KeyMatcher::prefix("vision.")],
        ..LoadOptions::default()
    };
    let manifest = discover_checkpoint(temp.path(), &options).unwrap();
    let expected =
        ["first", "second", "tied", "optional"].map(|name| ExpectedTensor::new(name, [1]));
    let mut mapping = MappingPlan::new();
    mapping
        .map("first", "first")
        .map("second", "second")
        .alias("first", "tied");
    let plan = build_load_plan(&manifest, &expected, &mapping, &options).unwrap();
    assert!(plan.report.is_strictly_valid());
    let original_report = plan.report.clone();
    let missing_path = &manifest.shard("z.safetensors").unwrap().path;
    let bytes = fs::read(missing_path).unwrap();
    fs::remove_file(missing_path).unwrap();

    let mut sink = CheckpointContractSink::default();
    let failure = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap_err();
    assert_eq!(failure.failed_target.as_deref(), Some("second"));
    assert!(matches!(&failure.cause, ExecutionFailureCause::Checkpoint(
        CheckpointError::ArtifactChanged { path, reason })
        if path == missing_path && reason.contains("cannot canonicalize")));
    assert_eq!(failure.report.applied, ["first"]);
    assert_eq!(failure.report.allowlisted_missing, ["optional"]);
    assert_eq!(failure.report.allowlisted_ignored, ["vision.weight"]);
    assert_eq!(failure.report.aliased, original_report.aliased);
    assert!(failure.report.is_strictly_valid());
    assert_eq!(failure.stats.tensors_applied, 1);
    assert_eq!(failure.stats.aliases_applied, 0);
    assert_eq!(failure.stats.source_reads, 1);
    assert_eq!(failure.stats.source_payload_bytes, 4);
    assert_eq!(failure.stats.source.shard_open_order, ["a.safetensors"]);
    assert_eq!(sink.events, ["write:first"]);
    assert_eq!(sink.tensors.len(), 1);
    assert_eq!(sink.tensors["first"].to_vec::<f32>().unwrap(), [0.0]);

    fs::write(missing_path, bytes).unwrap();
    let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();
    let mut expected_report = original_report.clone();
    expected_report.applied = vec!["first".into(), "second".into(), "tied".into()];
    assert_eq!(outcome.report, expected_report);
    assert_eq!(outcome.stats.tensors_applied, 2);
    assert_eq!(outcome.stats.aliases_applied, 1);
    assert_eq!(outcome.stats.source_reads, 2);
    assert_eq!(outcome.stats.source_payload_bytes, 8);
    assert_eq!(
        outcome.stats.source.shard_open_order,
        ["a.safetensors", "z.safetensors"]
    );
    assert_eq!(outcome.stats.source.max_open_shards, 1);
    assert_eq!(
        sink.events,
        [
            "write:first",
            "write:first",
            "write:second",
            "alias:tied:first"
        ]
    );
    assert_eq!(sink.tensors.len(), 3);
    assert_eq!(sink.tensors["tied"], sink.tensors["first"]);
    assert!(std::sync::Arc::ptr_eq(
        &sink.tensors["tied"],
        &sink.tensors["first"]
    ));
    assert_eq!(plan.report, original_report);
}

/// Two standard artifacts are ambiguous; neither a malformed nor a repaired index is preferred silently.
#[test]
fn checkpoint_contract_bad_preferred_index_requires_explicit_selection_or_repair() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "model.safetensors",
        &[FixtureTensor::new("single", TensorDType::F32, [1])],
    );
    write_shard(
        temp.path(),
        "part.safetensors",
        &[FixtureTensor::new("indexed", TensorDType::F32, [1])],
    );
    let index = temp.path().join("model.safetensors.index.json");
    fs::write(&index, br#"{"weight_map": ["not-an-object"]}"#).unwrap();
    let before = fs::read(&index).unwrap();
    let ambiguous = discover_checkpoint(temp.path(), &LoadOptions::default()).unwrap_err();
    assert!(matches!(
        &ambiguous,
        CheckpointError::AmbiguousArtifacts { candidates }
        if candidates == &["model.safetensors", "model.safetensors.index.json"]
    ));
    let index_only = LoadOptions {
        artifact: ArtifactSelection::Index("model.safetensors.index.json".into()),
        ..LoadOptions::default()
    };
    let error = discover_checkpoint(temp.path(), &index_only).unwrap_err();
    assert_unique_map_type_error(error, "checkpoint index", &index, "sequence", before.len());
    let single = discover_checkpoint(
        temp.path(),
        &LoadOptions {
            artifact: ArtifactSelection::Single("model.safetensors".into()),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    assert!(single.tensor("single").is_some());
    assert!(single.tensor("indexed").is_none());
    assert_eq!(fs::read(&index).unwrap(), before);

    write_index(temp.path(), &[("indexed", "part.safetensors")], Some(4));
    assert!(matches!(
        discover_checkpoint(temp.path(), &LoadOptions::default()),
        Err(CheckpointError::AmbiguousArtifacts { .. })
    ));
    let repaired = discover_checkpoint(temp.path(), &index_only).unwrap();
    assert!(matches!(repaired.source, ArtifactSource::Index { .. }));
    assert!(repaired.tensor("indexed").is_some());
    assert!(repaired.tensor("single").is_none());
    let expected = [ExpectedTensor::new("indexed", [1])];
    let plan = build_load_plan(
        &repaired,
        &expected,
        &MappingPlan::identity(&expected),
        &index_only,
    )
    .unwrap();
    let mut sink = CheckpointContractSink::default();
    let outcome = execute_local_load_plan(&repaired, &plan, &mut sink).unwrap();
    assert_eq!(outcome.report.applied, ["indexed"]);
    assert!(outcome.report.is_strictly_valid());
    assert_eq!(sink.events, ["write:indexed"]);
}

proptest! {
    #[test]
    /// Generated dtype storage calculations scale linearly for byte-aligned element counts.
    fn dtype_storage_bytes_scale_linearly(elements in 0_u64..1_000_000) {
        prop_assert_eq!(TensorDType::F16.storage_bytes(elements).unwrap(), elements * 2);
        prop_assert_eq!(TensorDType::F32.storage_bytes(elements).unwrap(), elements * 4);
        prop_assert_eq!(TensorDType::I64.storage_bytes(elements).unwrap(), elements * 8);
    }

    #[test]
    /// Generated exhaustive splits preserve the original shape extent and payload byte count.
    fn arbitrary_exhaustive_splits_preserve_shape_and_bytes(
        left in 1_u64..32,
        middle in 1_u64..32,
        right in 1_u64..32,
        width in 1_u64..32,
    ) {
        let total = left + middle + right;
        let (_temp, manifest) = fixture_manifest(&[FixtureTensor::new(
            "fused",
            TensorDType::F16,
            vec![total, width],
        )]);
        let expected = [
            ExpectedTensor::new("left", vec![left, width]),
            ExpectedTensor::new("middle", vec![middle, width]),
            ExpectedTensor::new("right", vec![right, width]),
        ];
        let mut mappings = MappingPlan::new();
        mappings.split(
            "fused",
            0,
            [
                ("left".into(), left),
                ("middle".into(), middle),
                ("right".into(), right),
            ],
        );
        let plan = build_load_plan(&manifest, &expected, &mappings, &LoadOptions::default()).unwrap();
        prop_assert_eq!(plan.memory.target_parameter_bytes, total * width * 2);
        let starts: Vec<_> = plan
            .tensors
            .iter()
            .map(|tensor| match tensor.transform {
                PlannedTransform::Slice { start, .. } => start,
                _ => unreachable!(),
            })
            .collect();
        // Targets are sorted alphabetically after validation; inspect as a set.
        prop_assert!(starts.contains(&0));
        prop_assert!(starts.contains(&left));
        prop_assert!(starts.contains(&(left + middle)));
    }

    #[test]
    /// Generated transpose plans swap only the two selected dimensions.
    fn transpose_plans_swap_exactly_two_dimensions(rows in 1_u64..64, cols in 1_u64..64) {
        let (_temp, manifest) = fixture_manifest(&[FixtureTensor::new(
            "source",
            TensorDType::F32,
            vec![rows, cols],
        )]);
        let expected = [ExpectedTensor::new("target", vec![cols, rows])];
        let mut mapping = MappingPlan::new();
        mapping.transpose_2d("source", "target");
        let plan = build_load_plan(&manifest, &expected, &mapping, &LoadOptions::default()).unwrap();
        prop_assert_eq!(&plan.tensors[0].shape, &vec![cols, rows]);
        prop_assert_eq!(plan.memory.target_parameter_bytes, rows * cols * 4);
    }
}

/// Repeating a resolved alias must not double count or silently overwrite its identity.
#[test]
fn checkpoint_standards_duplicate_resolved_alias_is_rejected_before_planning() {
    let (_directory, mut manifest, options) = indexed_mapping_contract_fixture();
    let before = manifest.clone();
    manifest
        .index_aliases
        .push(manifest.index_aliases[0].clone());
    let error = plan_mapping_contract_fixture(&manifest, &options).unwrap_err();
    assert!(
        matches!(error, CheckpointError::InvalidIndexAlias { alias, reason }
        if alias == "<duplicate>" && reason == "resolved manifest aliases must have unique alias keys")
    );
    assert_eq!(manifest.tensors, before.tensors);
    assert_eq!(manifest.tensor_bytes, before.tensor_bytes);
}

/// An alias declared before a primary operation on the same target still collides.
#[test]
fn alias_before_primary_target_is_a_collision_for_every_operation() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("embed", TensorDType::F32, [2, 2]),
        FixtureTensor::new("head_source", TensorDType::F32, [2, 2]),
        FixtureTensor::new("fused", TensorDType::F32, [4, 2]),
        FixtureTensor::new("left", TensorDType::F32, [1, 2]),
        FixtureTensor::new("right", TensorDType::F32, [1, 2]),
    ]);
    let expected = [
        ExpectedTensor::new("embedding", [2, 2]),
        ExpectedTensor::new("head", [2, 2]),
        ExpectedTensor::new("other", [2, 2]),
    ];
    let operations: [fn(&mut MappingPlan); 3] = [
        |plan| {
            plan.map("head_source", "head");
        },
        |plan| {
            plan.split(
                "fused",
                0,
                [("head".to_owned(), 2), ("other".to_owned(), 2)],
            );
        },
        |plan| {
            plan.concatenate(["left".to_owned(), "right".to_owned()], "head", 0);
        },
    ];
    for primary in operations {
        let mut mappings = MappingPlan::new();
        mappings
            .map("embed", "embedding")
            .alias("embedding", "head");
        primary(&mut mappings);
        assert!(matches!(
            build_load_plan(&manifest, &expected, &mappings, &LoadOptions::default()),
            Err(CheckpointError::MappingTargetCollision { tensor }) if tensor == "head"
        ));
    }
}

/// Casts the executor cannot perform are rejected while planning, not at execution.
#[test]
fn planner_rejects_float_casts_outside_the_runtime_formats() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("fp8", TensorDType::F8_E4M3, [4]),
        FixtureTensor::new("half", TensorDType::BF16, [4]),
    ]);
    let generic = [
        ExpectedTensor::new("fp8", [4]),
        ExpectedTensor::new("half", [4]),
    ];
    let identity = MappingPlan::identity(&generic);
    let cast_to_fp8 = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F8_E4M3),
        ..LoadOptions::default()
    };
    assert!(matches!(
        build_load_plan(&manifest, &generic, &identity, &cast_to_fp8),
        Err(CheckpointError::InvalidMapping { message }) if message.contains("castable")
    ));
    let cast_fp8_source = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::BF16),
        ..LoadOptions::default()
    };
    assert!(matches!(
        build_load_plan(&manifest, &generic, &identity, &cast_fp8_source),
        Err(CheckpointError::UnsupportedDTypeConversion {
            source_dtype: TensorDType::F8_E4M3,
            target_dtype: TensorDType::BF16
        })
    ));
    let typed = [
        ExpectedTensor::new("fp8", [4]).with_dtype(TensorDType::BF16),
        ExpectedTensor::new("half", [4]),
    ];
    assert!(matches!(
        build_load_plan(
            &manifest,
            &typed,
            &MappingPlan::identity(&typed),
            &LoadOptions::default()
        ),
        Err(CheckpointError::UnsupportedDTypeConversion {
            source_dtype: TensorDType::F8_E4M3,
            target_dtype: TensorDType::BF16
        })
    ));
    // FP8 storage loads unchanged when no conversion is requested.
    let plan = build_load_plan(&manifest, &generic, &identity, &LoadOptions::default()).unwrap();
    assert!(plan.report.dtype_converted.is_empty());
}

/// A nested index resolves co-located shards and records root-relative shard paths.
#[test]
fn nested_index_resolves_shards_beside_it() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "text_encoder/part-1.safetensors",
        &[FixtureTensor::new("a", TensorDType::F32, [2])],
    );
    write_shard(
        temp.path(),
        "text_encoder/part-2.safetensors",
        &[FixtureTensor::new("b", TensorDType::F16, [3])],
    );
    let index = json!({
        "metadata": {"total_size": 14},
        "weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"},
    });
    fs::write(
        temp.path()
            .join("text_encoder/model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    let options = LoadOptions {
        artifact: ArtifactSelection::Index("text_encoder/model.safetensors.index.json".into()),
        ..LoadOptions::default()
    };
    let manifest = discover_checkpoint(temp.path(), &options).unwrap();
    assert_eq!(
        manifest
            .shards
            .iter()
            .map(|shard| shard.relative_path.as_str())
            .collect::<Vec<_>>(),
        [
            "text_encoder/part-1.safetensors",
            "text_encoder/part-2.safetensors"
        ]
    );
    assert_eq!(
        manifest.tensor("b").unwrap().shard,
        "text_encoder/part-2.safetensors"
    );
    let expected = [ExpectedTensor::new("a", [2]), ExpectedTensor::new("b", [3])];
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &options,
    )
    .unwrap();
    let mut sink = CheckpointContractSink::default();
    let outcome = execute_local_load_plan(&manifest, &plan, &mut sink).unwrap();
    assert_eq!(outcome.report.applied, ["a", "b"]);
    assert_eq!(outcome.stats.source.shard_opens, 2);
}

/// Shard names in a nested index cannot climb out of the index directory.
#[test]
fn nested_index_shard_names_must_be_normal_relative_paths() {
    let temp = TempDir::new().unwrap();
    write_shard(
        temp.path(),
        "outside.safetensors",
        &[FixtureTensor::new("a", TensorDType::F32, [1])],
    );
    fs::create_dir_all(temp.path().join("nested")).unwrap();
    for shard in [
        "../outside.safetensors",
        "/outside.safetensors",
        "a\\b.safetensors",
    ] {
        let index = json!({"weight_map": {"a": shard}});
        fs::write(
            temp.path().join("nested/model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        let options = LoadOptions {
            artifact: ArtifactSelection::Index("nested/model.safetensors.index.json".into()),
            ..LoadOptions::default()
        };
        assert!(
            matches!(
                discover_checkpoint(temp.path(), &options),
                Err(CheckpointError::UnsafeRelativePath { path, .. }) if path == shard
            ),
            "{shard} escaped the index directory"
        );
    }
}

/// The tight staging figure never exceeds the conservative planned bound.
#[test]
fn required_staging_is_the_largest_operation_and_within_the_planned_bound() {
    let (_temp, manifest) = fixture_manifest(&[
        FixtureTensor::new("cast", TensorDType::BF16, [8]),
        FixtureTensor::new("widen", TensorDType::BF16, [4]),
        FixtureTensor::new("matrix", TensorDType::F32, [2, 3]),
    ]);
    let expected = [
        ExpectedTensor::new("cast", [8]).with_dtype(TensorDType::F16),
        ExpectedTensor::new("widen", [4]).with_dtype(TensorDType::F32),
        ExpectedTensor::new("matrix", [3, 2]),
    ];
    let mut mappings = MappingPlan::new();
    mappings
        .map("cast", "cast")
        .map("widen", "widen")
        .transpose_2d("matrix", "matrix");
    let plan = build_load_plan(&manifest, &expected, &mappings, &LoadOptions::default()).unwrap();
    let staging = |target: &str| {
        plan.tensors
            .iter()
            .find(|tensor| tensor.target == target)
            .unwrap()
            .staging_bytes()
            .unwrap()
    };
    // BF16 -> F16 converts in place: only the 16-byte source is staged.
    assert_eq!(staging("cast"), 16);
    // BF16 -> F32 needs the 8-byte source plus a 16-byte output.
    assert_eq!(staging("widen"), 24);
    // The transpose holds the 24-byte source and a 24-byte output.
    assert_eq!(staging("matrix"), 48);
    assert_eq!(plan.required_host_staging_bytes().unwrap(), 48);
    assert!(plan.required_host_staging_bytes().unwrap() <= plan.memory.peak_host_staging_bytes);
}
