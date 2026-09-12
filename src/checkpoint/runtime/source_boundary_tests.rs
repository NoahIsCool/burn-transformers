//! Checks descriptor integrity, shard replacement, and bounded source reads.

use std::io::Write;

use tempfile::TempDir;

use super::*;
use crate::checkpoint::{LoadOptions, discover_checkpoint};

/// Writes a real two-element shard so mutation tests start with audited bytes.
fn fixture() -> (TempDir, CheckpointManifest, PlannedSource) {
    let directory = TempDir::new().unwrap();
    let header = serde_json::to_vec(&serde_json::json!({
        "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    }))
    .unwrap();
    let mut file = File::create(directory.path().join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&1.0_f32.to_le_bytes()).unwrap();
    file.write_all(&2.0_f32.to_le_bytes()).unwrap();
    let manifest = discover_checkpoint(directory.path(), &LoadOptions::default()).unwrap();
    let inventory = manifest.tensor("weight").unwrap();
    let descriptor = PlannedSource {
        name: inventory.name.clone(),
        shard: inventory.shard.clone(),
        shape: inventory.shape.clone(),
        dtype: inventory.dtype,
        data_offsets: inventory.data_offsets,
    };
    (directory, manifest, descriptor)
}

/// Checks that descriptor rejection reports the exact field before any I/O.
#[test]
fn every_descriptor_field_is_revalidated_without_opening_a_shard() {
    let (_directory, manifest, descriptor) = fixture();
    let mut source = LocalShardSource::new(&manifest);
    assert!(std::ptr::eq(source.manifest(), &manifest));
    assert!(!source.validate_tensor_descriptor(&descriptor).unwrap());
    for (field, reason) in [
        ("name", "tensor is absent from the manifest"),
        ("shard", "shard name differs"),
        ("shape", "shape differs"),
        ("dtype", "dtype differs"),
        ("offsets", "data offsets differ"),
    ] {
        let mut invalid = descriptor.clone();
        match field {
            "name" => invalid.name = "missing".into(),
            "shard" => invalid.shard = "missing.safetensors".into(),
            "shape" => invalid.shape = vec![1, 2],
            "dtype" => invalid.dtype = TensorDType::F16,
            "offsets" => invalid.data_offsets = [1, 9],
            _ => unreachable!(),
        }
        let error = source.read_tensor(&invalid).unwrap_err();
        assert!(
            matches!(error, CheckpointError::PlanManifestMismatch { tensor, reason: actual }
            if tensor == invalid.name && actual == reason),
            "field {field}"
        );
        assert_eq!(source.stats(), SourceStats::default());
    }

    // The public inventory map can itself be edited after discovery. Its key
    // must not conceal a mismatched name stored in the corresponding value.
    let mut inconsistent = manifest.clone();
    inconsistent.tensors.get_mut("weight").unwrap().name = "other".into();
    assert!(matches!(
        LocalShardSource::new(&inconsistent).read_tensor(&descriptor),
        Err(CheckpointError::PlanManifestMismatch { reason, .. }) if reason == "source name differs"
    ));
}

/// Ensures descriptor certification never substitutes for payload provenance.
#[test]
fn removed_shard_is_not_certified_and_fails_when_payload_is_requested() {
    let (_directory, manifest, descriptor) = fixture();
    fs::remove_file(&manifest.shards[0].path).unwrap();
    let mut source = LocalShardSource::new(&manifest);
    assert!(!source.validate_tensor_descriptor(&descriptor).unwrap());
    assert!(matches!(source.read_tensor(&descriptor),
        Err(CheckpointError::ArtifactChanged { reason, .. }) if reason.contains("cannot canonicalize")));
    assert_eq!(source.stats(), SourceStats::default());
}

/// Rejects a now-directory shard without attempting to read a payload.
#[test]
fn directory_replacement_is_rejected_before_open() {
    let (_directory, manifest, descriptor) = fixture();
    let path = &manifest.shards[0].path;
    fs::remove_file(path).unwrap();
    fs::create_dir(path).unwrap();
    let mut source = LocalShardSource::new(&manifest);
    assert!(matches!(source.read_tensor(&descriptor),
        Err(CheckpointError::ArtifactChanged { reason, .. }) if reason.contains("regular non-symlink")));
    assert_eq!(source.stats(), SourceStats::default());
}

/// Rejects symlink substitution both inside and outside the original root.
#[cfg(unix)]
#[test]
fn symlink_replacement_cannot_redirect_a_discovered_shard() {
    for outside_root in [false, true] {
        let (_directory, manifest, descriptor) = fixture();
        let outside = TempDir::new().unwrap();
        let path = &manifest.shards[0].path;
        let replacement = if outside_root {
            outside.path().join("replacement.safetensors")
        } else {
            manifest.root.join("replacement.safetensors")
        };
        fs::rename(path, &replacement).unwrap();
        std::os::unix::fs::symlink(&replacement, path).unwrap();
        let mut source = LocalShardSource::new(&manifest);
        assert!(matches!(source.read_tensor(&descriptor),
            Err(CheckpointError::ArtifactChanged { reason, .. }) if reason.contains("canonical path is now")));
        assert_eq!(source.stats(), SourceStats::default());
    }
}

/// Treats a mutated root or removed shard inventory entry as untrusted input.
#[test]
fn public_manifest_mutations_cannot_bypass_shard_membership() {
    let (_directory, mut manifest, descriptor) = fixture();
    let other = TempDir::new().unwrap();
    let original_root = manifest.root.clone();
    manifest.root = fs::canonicalize(other.path()).unwrap();
    assert!(
        matches!(LocalShardSource::new(&manifest).read_tensor(&descriptor),
        Err(CheckpointError::ArtifactChanged { reason, .. }) if reason.contains("canonical path is now"))
    );
    manifest.root = original_root;
    manifest.shards.clear();
    assert!(
        matches!(LocalShardSource::new(&manifest).read_tensor(&descriptor),
        Err(CheckpointError::PlanManifestMismatch { reason, .. }) if reason.contains("shard absent"))
    );
}

/// Exercises checked shard-boundary arithmetic without allocating large files.
#[test]
fn forged_manifest_lengths_fail_closed_at_each_arithmetic_boundary() {
    for case in [
        "header_overflow",
        "data_overflow",
        "data_end",
        "short_prefix",
    ] {
        let (_directory, mut manifest, descriptor) = fixture();
        let shard = &mut manifest.shards[0];
        match case {
            "header_overflow" => {
                shard.header_bytes = u64::MAX;
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .open(&shard.path)
                    .unwrap();
                file.write_all(&u64::MAX.to_le_bytes()).unwrap();
            }
            "data_overflow" => shard.data_bytes = u64::MAX,
            "data_end" => shard.data_bytes -= 1,
            "short_prefix" => {
                fs::OpenOptions::new()
                    .write(true)
                    .open(&shard.path)
                    .unwrap()
                    .set_len(4)
                    .unwrap();
                shard.file_bytes = 4;
            }
            _ => unreachable!(),
        }
        let mut source = LocalShardSource::new(&manifest);
        let error = source.read_tensor(&descriptor).unwrap_err();
        match case {
            "header_overflow" => assert!(matches!(
                error,
                CheckpointError::ArithmeticOverflow {
                    context: "computing runtime shard data offset"
                }
            )),
            "data_overflow" => assert!(matches!(
                error,
                CheckpointError::ArithmeticOverflow {
                    context: "computing runtime shard data end"
                }
            )),
            "data_end" => assert!(matches!(error,
                CheckpointError::ArtifactChanged { reason, .. } if reason.contains("data section ends"))),
            "short_prefix" => assert!(matches!(error,
                CheckpointError::Io { operation: "re-read shard header prefix", source, .. }
                if source.kind() == std::io::ErrorKind::UnexpectedEof)),
            _ => unreachable!(),
        }
        assert_eq!(source.stats(), SourceStats::default());
    }
}

/// Checks offsets and impossible allocation sizes after public metadata edits.
#[test]
fn forged_tensor_offsets_are_checked_before_reading() {
    for offsets in [[8, 0], [u64::MAX - 8, u64::MAX], [0, u64::MAX]] {
        let (_directory, mut manifest, mut descriptor) = fixture();
        descriptor.data_offsets = offsets;
        manifest.tensors.get_mut("weight").unwrap().data_offsets = offsets;
        let mut source = LocalShardSource::new(&manifest);
        let error = source.read_tensor(&descriptor).unwrap_err();
        if offsets[0] > offsets[1] {
            assert!(
                matches!(error, CheckpointError::PlanManifestMismatch { reason, .. }
                if reason == "data offsets are reversed")
            );
        } else if offsets[0] == 0 {
            // `u64::MAX` cannot fit a Vec allocation on supported 32/64-bit
            // hosts: this exercises reservation failure without requesting
            // physical memory or depending on the machine's free space.
            assert!(matches!(
                error,
                CheckpointError::AllocationFailed {
                    bytes: u64::MAX,
                    context: "allocating one checkpoint tensor payload"
                }
            ));
        } else {
            assert!(matches!(
                error,
                CheckpointError::ArithmeticOverflow {
                    context: "computing runtime tensor file offset"
                }
            ));
        }
        assert_eq!(source.stats().tensor_reads, 0);
        assert_eq!(source.stats().payload_bytes_read, 0);
    }
}

/// Verifies handle reuse, explicit release, and successful-read-only counters.
#[test]
fn reused_handle_detects_truncated_payload_and_close_allows_reopen() {
    let (_directory, manifest, descriptor) = fixture();
    let path = &manifest.shards[0].path;
    let original = fs::read(path).unwrap();
    let mut source = LocalShardSource::new(&manifest);
    let first = source.read_tensor(&descriptor).unwrap();
    assert_eq!(source.read_tensor(&descriptor).unwrap(), first);
    assert_eq!(source.stats().shard_opens, 1);
    assert_eq!(source.stats().tensor_reads, 2);
    assert_eq!(source.stats().payload_bytes_read, 16);
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(8 + manifest.shards[0].header_bytes + 4)
        .unwrap();
    assert!(matches!(source.read_tensor(&descriptor),
        Err(CheckpointError::Io { operation: "read checkpoint tensor", source, .. })
        if source.kind() == std::io::ErrorKind::UnexpectedEof));
    assert_eq!(source.stats().tensor_reads, 2);
    assert_eq!(source.stats().payload_bytes_read, 16);
    source.close().unwrap();
    source.close().unwrap();
    assert!(source.open.is_none());
    fs::write(path, original).unwrap();
    assert_eq!(source.read_tensor(&descriptor).unwrap(), first);
    assert_eq!(source.stats().shard_opens, 2);
    assert_eq!(source.stats().max_open_shards, 1);
    assert_eq!(
        source.stats().shard_open_order,
        vec![descriptor.shard.clone(), descriptor.shard]
    );
}

/// Preserves scalar/empty shapes while rejecting storage-size overflow.
#[test]
fn source_payload_scalar_empty_and_storage_overflow_boundaries() {
    let scalar = SourceTensor::new(
        "scalar",
        [],
        TensorDType::F32,
        1.0_f32.to_le_bytes().to_vec(),
    )
    .unwrap();
    assert!(scalar.shape().is_empty());
    assert_eq!(scalar.byte_len(), 4);
    let empty = SourceTensor::new("empty", [0, u64::MAX], TensorDType::F64, vec![]).unwrap();
    assert_eq!(empty.byte_len(), 0);
    assert!(matches!(
        SourceTensor::new("large", [u64::MAX], TensorDType::F64, vec![]),
        Err(CheckpointError::ArithmeticOverflow { .. })
    ));
}
