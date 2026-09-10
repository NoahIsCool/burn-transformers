use std::io::Write;

use super::*;
use crate::checkpoint::{
    ExpectedTensor, LoadOptions, LoadReport, MappingPlan, MemoryEstimate, PlannedTensorEquivalence,
    TensorDType, build_load_plan, discover_checkpoint,
};

/// Constructs a plan exercising source, transform, equivalence, and alias domains.
fn fixture_plan() -> LoadPlan {
    let owner = PlannedSource {
        name: "external.weight".into(),
        shard: "model-00001-of-00002.safetensors".into(),
        shape: vec![2, 4],
        dtype: TensorDType::BF16,
        data_offsets: [128, 144],
    };
    let redundant = PlannedSource {
        name: "external.tied_weight".into(),
        data_offsets: [144, 160],
        ..owner.clone()
    };
    LoadPlan {
        equivalences: vec![PlannedTensorEquivalence {
            alias: redundant,
            owner: owner.clone(),
        }],
        tensors: vec![PlannedTensor {
            target: "model.projection.weight".into(),
            sources: vec![owner],
            transform: PlannedTransform::Transpose2d,
            shape: vec![4, 2],
            dtype: TensorDType::F16,
        }],
        aliases: vec![PlannedTensor {
            target: "lm_head.weight".into(),
            sources: Vec::new(),
            transform: PlannedTransform::Alias {
                source_target: "model.projection.weight".into(),
            },
            shape: vec![4, 2],
            dtype: TensorDType::F16,
        }],
        report: LoadReport::default(),
        memory: MemoryEstimate::default(),
    }
}

/// Canonical plan hashing is stable and covers mapping semantics, not reports.
#[test]
fn canonical_plan_digest_is_stable_and_semantically_sensitive() {
    let plan = fixture_plan();
    let digest = load_plan_sha256(&plan);
    assert_eq!(
        digest,
        "2ff631fc5de762e8499277fe3cfd817611d5091b8d601dddd87e74689426c126"
    );
    assert_eq!(load_plan_sha256(&plan.clone()), digest);

    let mut changed = plan.clone();
    changed.tensors[0].sources[0].data_offsets[0] += 1;
    assert_ne!(load_plan_sha256(&changed), digest);

    let mut changed = plan.clone();
    changed.tensors[0].transform = PlannedTransform::Identity;
    assert_ne!(load_plan_sha256(&changed), digest);

    let mut changed = plan.clone();
    changed.aliases[0].target = "different.weight".into();
    assert_ne!(load_plan_sha256(&changed), digest);

    let mut report_only = plan.clone();
    report_only.report.applied.push("diagnostic-only".into());
    assert_eq!(load_plan_sha256(&report_only), digest);
}

/// V2 digest spelling is canonical and rejects missing or malformed source identity.
#[test]
fn provenance_metadata_rejects_noncanonical_digests_and_empty_shards() {
    let mut provenance = BurnpackProvenance {
        load_plan_sha256: "0".repeat(64),
        checkpoint_shards: BTreeMap::from([(
            "model.safetensors".into(),
            BurnpackShardProvenance {
                file_bytes: 128,
                sha256: "a".repeat(64),
            },
        )]),
    };
    provenance.validate().unwrap();

    provenance.load_plan_sha256 = "A".repeat(64);
    assert!(matches!(
        provenance.validate(),
        Err(QuantizationError::InvalidProvenance { .. })
    ));
    provenance.load_plan_sha256 = "0".repeat(64);
    provenance.checkpoint_shards.clear();
    assert!(matches!(
        provenance.validate(),
        Err(QuantizationError::InvalidProvenance { .. })
    ));

    provenance.checkpoint_shards.insert(
        "../model.safetensors".into(),
        BurnpackShardProvenance {
            file_bytes: 128,
            sha256: "a".repeat(64),
        },
    );
    assert!(matches!(
        provenance.validate(),
        Err(QuantizationError::InvalidProvenance { .. })
    ));

    for invalid_path in [
        "nested//model.safetensors",
        "nested\\model.safetensors",
        "C:model.safetensors",
        "model\0.safetensors",
    ] {
        provenance.checkpoint_shards.clear();
        provenance.checkpoint_shards.insert(
            invalid_path.into(),
            BurnpackShardProvenance {
                file_bytes: 128,
                sha256: "a".repeat(64),
            },
        );
        assert!(matches!(
            provenance.validate(),
            Err(QuantizationError::InvalidProvenance { .. })
        ));
    }
}

/// Writes a real shard large enough, when requested, to span hash-buffer reads.
fn local_fixture(elements: usize) -> (tempfile::TempDir, CheckpointManifest, LoadPlan) {
    let directory = tempfile::tempdir().unwrap();
    let header = serde_json::to_vec(&serde_json::json!({
        "weight": {"dtype": "F32", "shape": [elements], "data_offsets": [0, elements * 4]}
    }))
    .unwrap();
    let mut file = File::create(directory.path().join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    let bytes = (0..elements)
        .flat_map(|_| 1.0_f32.to_le_bytes())
        .collect::<Vec<_>>();
    file.write_all(&bytes).unwrap();
    let manifest = discover_checkpoint(directory.path(), &LoadOptions::default()).unwrap();
    let expected = [ExpectedTensor::new("weight", [elements as u64])];
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap();
    (directory, manifest, plan)
}

/// Covers every plan field and the slice/concatenation tags absent in the golden.
#[test]
fn canonical_digest_binds_every_executable_field_but_not_memory_estimates() {
    let plan = fixture_plan();
    let base = load_plan_sha256(&plan);
    for field in [
        "target",
        "source_name",
        "source_shard",
        "source_shape",
        "source_dtype",
        "source_start",
        "source_end",
        "shape",
        "dtype",
        "source_count",
        "tensor_count",
        "alias_owner",
        "equivalence_owner",
        "equivalence_order",
    ] {
        let mut changed = plan.clone();
        match field {
            "target" => changed.tensors[0].target.push('x'),
            "source_name" => changed.tensors[0].sources[0].name.push('x'),
            "source_shard" => changed.tensors[0].sources[0].shard.push('x'),
            "source_shape" => changed.tensors[0].sources[0].shape.reverse(),
            "source_dtype" => changed.tensors[0].sources[0].dtype = TensorDType::F16,
            "source_start" => changed.tensors[0].sources[0].data_offsets[0] += 1,
            "source_end" => changed.tensors[0].sources[0].data_offsets[1] += 1,
            "shape" => changed.tensors[0].shape.reverse(),
            "dtype" => changed.tensors[0].dtype = TensorDType::BF16,
            "source_count" => {
                let source = changed.tensors[0].sources[0].clone();
                changed.tensors[0].sources.push(source);
            }
            "tensor_count" => changed.tensors.push(changed.tensors[0].clone()),
            "alias_owner" => {
                changed.aliases[0].transform = PlannedTransform::Alias {
                    source_target: "different".into(),
                }
            }
            "equivalence_owner" => changed.equivalences[0].owner.name.push('x'),
            "equivalence_order" => {
                let item = &mut changed.equivalences[0];
                std::mem::swap(&mut item.owner, &mut item.alias);
            }
            _ => unreachable!(),
        }
        assert_ne!(load_plan_sha256(&changed), base, "field {field}");
    }
    let mut digests = std::collections::BTreeSet::new();
    for transform in [
        PlannedTransform::Identity,
        PlannedTransform::Transpose2d,
        PlannedTransform::Slice {
            axis: 0,
            start: 0,
            length: 1,
        },
        PlannedTransform::Slice {
            axis: 1,
            start: 0,
            length: 1,
        },
        PlannedTransform::Slice {
            axis: 0,
            start: 1,
            length: 1,
        },
        PlannedTransform::Slice {
            axis: 0,
            start: 0,
            length: 2,
        },
        PlannedTransform::Concatenate { axis: 0 },
        PlannedTransform::Concatenate { axis: 1 },
    ] {
        let mut changed = plan.clone();
        changed.tensors[0].transform = transform;
        assert!(digests.insert(load_plan_sha256(&changed)));
    }
    let mut diagnostic = plan.clone();
    diagnostic.memory.peak_host_staging_bytes = u64::MAX;
    diagnostic.report.unexpected.push("diagnostic".into());
    assert_eq!(load_plan_sha256(&diagnostic), base);

    // Length prefixes separate fields that would collide under concatenation.
    let mut left = plan.clone();
    left.tensors[0].sources[0].name = "ab".into();
    left.tensors[0].sources[0].shard = "c".into();
    let mut right = left.clone();
    right.tensors[0].sources[0].name = "a".into();
    right.tensors[0].sources[0].shard = "bc".into();
    assert_ne!(load_plan_sha256(&left), load_plan_sha256(&right));
}

/// Rejects lexical dot segments even where Path::components normalizes them.
#[test]
fn provenance_requires_lexically_normalized_paths_nonzero_lengths_and_shard_hashes() {
    let mut provenance = BurnpackProvenance {
        load_plan_sha256: "a".repeat(64),
        checkpoint_shards: BTreeMap::from([(
            "nested/model.safetensors".into(),
            BurnpackShardProvenance {
                file_bytes: 8,
                sha256: "0".repeat(64),
            },
        )]),
    };
    provenance.validate().unwrap();
    let shard = provenance
        .checkpoint_shards
        .values()
        .next()
        .unwrap()
        .clone();
    for path in [
        "",
        "/model.safetensors",
        "./model.safetensors",
        "nested/./model.safetensors",
        "nested/../model.safetensors",
        "nested/model.safetensors/",
        "model.bin",
    ] {
        provenance.checkpoint_shards = BTreeMap::from([(path.into(), shard.clone())]);
        assert!(
            matches!(provenance.validate(), Err(QuantizationError::InvalidProvenance { reason })
            if reason.contains("normalized relative")),
            "accepted {path:?}"
        );
    }
    provenance.checkpoint_shards = BTreeMap::from([("model.safetensors".into(), shard)]);
    provenance
        .checkpoint_shards
        .get_mut("model.safetensors")
        .unwrap()
        .file_bytes = 0;
    assert!(
        matches!(provenance.validate(), Err(QuantizationError::InvalidProvenance { reason }) if reason.contains("zero length"))
    );
    provenance
        .checkpoint_shards
        .get_mut("model.safetensors")
        .unwrap()
        .file_bytes = 8;
    provenance
        .checkpoint_shards
        .get_mut("model.safetensors")
        .unwrap()
        .sha256 = "z".repeat(64);
    assert!(
        matches!(provenance.validate(), Err(QuantizationError::InvalidProvenance { reason }) if reason.starts_with("checkpoint shard SHA-256"))
    );
}

/// Compares a multi-buffer streaming hash with a one-shot digest and re-verifies.
#[test]
fn streamed_shard_hash_matches_complete_file_bytes_and_exact_plan() {
    let (_directory, manifest, plan) = local_fixture(PROVENANCE_HASH_BUFFER_BYTES / 4 + 7);
    let (provenance, stats) = hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
    let bytes = fs::read(&manifest.shards[0].path).unwrap();
    assert_eq!(
        provenance.checkpoint_shards["model.safetensors"].sha256,
        hex_digest(Sha256::digest(&bytes))
    );
    assert_eq!(stats.bytes_hashed, bytes.len() as u64);
    assert_eq!(stats.shards_hashed, 1);
    assert_eq!(stats.passes_completed, 1);
    assert_eq!(stats.peak_buffer_bytes, PROVENANCE_HASH_BUFFER_BYTES);
    assert_eq!(
        verify_local_checkpoint_provenance(&manifest, &plan, &provenance).unwrap(),
        stats
    );
    let mut changed = plan.clone();
    changed.tensors[0].target.push('x');
    assert!(matches!(verify_load_plan_provenance(&changed, &provenance),
        Err(QuantizationError::LoadPlanProvenanceMismatch { expected, actual })
        if expected == provenance.load_plan_sha256 && actual == load_plan_sha256(&changed)));
    let mut changed_bytes = bytes;
    *changed_bytes.last_mut().unwrap() ^= 1;
    fs::write(&manifest.shards[0].path, changed_bytes).unwrap();
    assert!(
        matches!(verify_local_checkpoint_provenance(&manifest, &plan, &provenance),
        Err(QuantizationError::CheckpointProvenanceDigestMismatch { shard, .. }) if shard == "model.safetensors")
    );
}

/// Preserves completed-shard counters when a later file cannot be opened.
#[test]
fn audited_hash_failure_preserves_prefix_and_rejects_duplicate_shards() {
    let (_directory, mut manifest, plan) = local_fixture(2);
    let mut absent = manifest.shards[0].clone();
    absent.relative_path = "missing.safetensors".into();
    absent.path = manifest.root.join(&absent.relative_path);
    manifest.shards.push(absent);
    let failure = hash_local_checkpoint_provenance_audited(&manifest, &plan).unwrap_err();
    assert!(matches!(
        failure.error,
        QuantizationError::ArtifactIo {
            operation: "canonicalize checkpoint shard for provenance",
            ..
        }
    ));
    assert_eq!(failure.stats.shards_hashed, 1);
    assert_eq!(failure.stats.bytes_hashed, manifest.shards[0].file_bytes);
    assert_eq!(failure.stats.passes_completed, 0);
    assert_eq!(
        failure.stats.peak_buffer_bytes,
        PROVENANCE_HASH_BUFFER_BYTES
    );

    manifest.shards[1] = manifest.shards[0].clone();
    let failure = hash_local_checkpoint_provenance_audited(&manifest, &plan).unwrap_err();
    assert!(
        matches!(failure.error, QuantizationError::InvalidProvenance { reason } if reason.contains("repeats shard"))
    );
    assert_eq!(failure.stats.shards_hashed, 2);
    assert_eq!(
        failure.stats.bytes_hashed,
        2 * manifest.shards[0].file_bytes
    );
    assert_eq!(failure.stats.passes_completed, 0);
    manifest.shards.clear();
    let failure = hash_local_checkpoint_provenance_audited(&manifest, &plan).unwrap_err();
    assert!(
        matches!(failure.error, QuantizationError::InvalidProvenance { reason } if reason.contains("digest map is empty"))
    );
    assert_eq!(failure.stats.bytes_hashed, 0);
    assert_eq!(failure.stats.peak_buffer_bytes, 0);
}

/// Rejects inventory and filesystem changes before allocating a hash buffer.
#[test]
fn shard_preflight_and_hashing_fail_closed_on_changed_files() {
    for case in ["missing", "directory", "length", "root"] {
        let (_directory, mut manifest, plan) = local_fixture(2);
        let (expected, _) = hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
        let other = tempfile::tempdir().unwrap();
        let path = manifest.shards[0].path.clone();
        match case {
            "missing" => fs::remove_file(&path).unwrap(),
            "directory" => {
                fs::remove_file(&path).unwrap();
                fs::create_dir(&path).unwrap();
            }
            "length" => {
                let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
                file.set_len(manifest.shards[0].file_bytes + 1).unwrap();
            }
            "root" => manifest.root = fs::canonicalize(other.path()).unwrap(),
            _ => unreachable!(),
        }
        let failure = hash_local_checkpoint_provenance_audited(&manifest, &plan).unwrap_err();
        assert_eq!(failure.stats, ProvenanceHashStats::default());
        match case {
            "missing" => assert!(matches!(
                failure.error,
                QuantizationError::ArtifactIo { .. }
            )),
            "length" => assert!(matches!(
                failure.error,
                QuantizationError::CheckpointProvenanceLengthMismatch { .. }
            )),
            _ => assert!(matches!(
                failure.error,
                QuantizationError::InvalidProvenance { .. }
            )),
        }
        if case != "root" {
            assert!(preflight_local_checkpoint_provenance(&manifest, &expected).is_err());
        }
    }
}

/// Both preflight and hashing refuse replacement symlinks, even within the root.
#[cfg(unix)]
#[test]
fn provenance_rejects_symlink_replacement_before_hashing() {
    let (_directory, manifest, plan) = local_fixture(2);
    let (expected, _) = hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
    let shard = &manifest.shards[0];
    let renamed = manifest.root.join("replacement.safetensors");
    fs::rename(&shard.path, &renamed).unwrap();
    std::os::unix::fs::symlink(&renamed, &shard.path).unwrap();
    assert!(
        matches!(preflight_local_checkpoint_provenance(&manifest, &expected),
        Err(QuantizationError::InvalidProvenance { reason }) if reason.contains("regular non-symlink"))
    );
    let failure = hash_local_checkpoint_provenance_audited(&manifest, &plan).unwrap_err();
    assert!(
        matches!(failure.error, QuantizationError::InvalidProvenance { reason } if reason.contains("discovered canonical path"))
    );
    assert_eq!(failure.stats, ProvenanceHashStats::default());
}

/// Distinguishes inventory cardinality, membership, and byte-length errors.
#[test]
fn provenance_inventory_comparisons_validate_counts_keys_and_lengths() {
    let (_directory, manifest, plan) = local_fixture(2);
    let (expected, _) = hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
    for case in ["count", "key", "length", "duplicate"] {
        let mut manifest = manifest.clone();
        let mut expected = expected.clone();
        match case {
            "count" => manifest.shards.clear(),
            "key" => manifest.shards[0].relative_path = "unknown.safetensors".into(),
            "length" => manifest.shards[0].file_bytes += 1,
            "duplicate" => {
                manifest.shards.push(manifest.shards[0].clone());
                expected.checkpoint_shards.insert(
                    "extra.safetensors".into(),
                    expected.checkpoint_shards["model.safetensors"].clone(),
                );
            }
            _ => unreachable!(),
        }
        let error = preflight_local_checkpoint_provenance(&manifest, &expected).unwrap_err();
        if case == "length" {
            assert!(matches!(
                error,
                QuantizationError::CheckpointProvenanceLengthMismatch { .. }
            ));
        } else {
            let expected = match case {
                "count" => "shard count",
                "key" => "absent from sidecar",
                _ => "repeats shard",
            };
            assert!(
                matches!(error, QuantizationError::InvalidProvenance { reason } if reason.contains(expected))
            );
        }
    }
    for case in ["count", "key", "length", "digest"] {
        let mut actual = expected.clone();
        match case {
            "count" => actual.checkpoint_shards.clear(),
            "key" => {
                let shard = actual
                    .checkpoint_shards
                    .remove("model.safetensors")
                    .unwrap();
                actual
                    .checkpoint_shards
                    .insert("other.safetensors".into(), shard);
            }
            "length" => {
                actual
                    .checkpoint_shards
                    .get_mut("model.safetensors")
                    .unwrap()
                    .file_bytes += 1
            }
            "digest" => {
                actual
                    .checkpoint_shards
                    .get_mut("model.safetensors")
                    .unwrap()
                    .sha256 = "0".repeat(64)
            }
            _ => unreachable!(),
        }
        let error = compare_checkpoint_provenance(&actual, &expected).unwrap_err();
        match case {
            "length" => assert!(matches!(
                error,
                QuantizationError::CheckpointProvenanceLengthMismatch { .. }
            )),
            "digest" => assert!(matches!(
                error,
                QuantizationError::CheckpointProvenanceDigestMismatch { .. }
            )),
            _ => assert!(matches!(error, QuantizationError::InvalidProvenance { .. })),
        }
    }
}

/// Hashes scale wire bits identically without allocating the combined payload.
#[test]
fn packed_scale_hashing_and_two_pass_statistics_preserve_exact_boundaries() {
    assert_eq!(
        packed_payload_sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let values = [0x12, 0xff, 0];
    let scales = [1.0_f32, -0.0, f32::from_bits(0x7fc0_0012)];
    let combined = values
        .into_iter()
        .chain(scales.into_iter().flat_map(f32::to_le_bytes))
        .collect::<Vec<_>>();
    assert_eq!(
        quantized_payload_sha256(&values, &scales),
        packed_payload_sha256(&combined)
    );
    let stats = ProvenanceExecutionStats {
        pre_execution: Some(ProvenanceHashStats {
            bytes_hashed: u64::MAX,
            peak_buffer_bytes: 16,
            ..ProvenanceHashStats::default()
        }),
        post_execution: Some(ProvenanceHashStats {
            bytes_hashed: u64::MAX,
            peak_buffer_bytes: 32,
            ..ProvenanceHashStats::default()
        }),
    };
    assert_eq!(stats.bytes_hashed(), 2 * u128::from(u64::MAX));
    assert_eq!(stats.peak_buffer_bytes(), 32);
    assert_eq!(ProvenanceExecutionStats::default().bytes_hashed(), 0);
    assert_eq!(ProvenanceExecutionStats::default().peak_buffer_bytes(), 0);
}
