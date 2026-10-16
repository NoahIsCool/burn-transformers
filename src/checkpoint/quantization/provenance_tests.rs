//! Tests exact source, plan, payload, and manifest provenance binding.

use std::{io::Write, sync::atomic::AtomicUsize};

use super::super::hash_workers::MAX_HASH_WORKERS;
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
    // Only the worker that read shard 0 allocated its share of the budget.
    let workers = hash_worker_count(
        2,
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get),
    );
    assert_eq!(
        failure.stats.peak_buffer_bytes,
        PROVENANCE_HASH_BUFFER_BYTES / workers
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

/// Counter overflow cannot falsely certify a completed shard or wrap accumulated bytes.
#[test]
fn checkpoint_standards_provenance_counter_overflow_retains_precise_progress() {
    let (_root, manifest, _plan) = local_fixture(4);
    let shard = &manifest.shards[0];
    for byte_overflow in [true, false] {
        let mut stats = ProvenanceHashStats {
            bytes_hashed: if byte_overflow { u64::MAX } else { 0 },
            shards_hashed: if byte_overflow { 0 } else { usize::MAX },
            ..Default::default()
        };
        let error = hash_shard(
            &manifest,
            shard,
            &mut HashBuffer::new(PROVENANCE_HASH_BUFFER_BYTES),
            never_cancelled(&AtomicUsize::new(usize::MAX)),
            &mut stats,
        )
        .unwrap_err();
        assert!(
            matches!(error, QuantizationError::ByteLengthOverflow { context }
            if context == if byte_overflow {
                "provenance checkpoint byte count"
            } else {
                "provenance shard count"
            })
        );
        assert_eq!(stats.passes_completed, 0);
        assert_eq!(stats.peak_buffer_bytes, PROVENANCE_HASH_BUFFER_BYTES);
        if byte_overflow {
            assert_eq!(stats.bytes_hashed, u64::MAX);
            assert_eq!(stats.shards_hashed, 0);
        } else {
            assert_eq!(stats.bytes_hashed, shard.file_bytes);
            assert_eq!(stats.shards_hashed, usize::MAX);
        }
    }
}

/// Writes `shards` single-tensor shards plus an index and returns its manifest and plan.
fn sharded_fixture(
    shards: usize,
    elements: usize,
) -> (tempfile::TempDir, CheckpointManifest, LoadPlan) {
    let directory = tempfile::tempdir().unwrap();
    let mut weight_map = serde_json::Map::new();
    let mut expected = Vec::new();
    for shard in 0..shards {
        let name = format!("layer{shard}.weight");
        let file_name = format!("model-{shard:05}.safetensors");
        let header = serde_json::to_vec(&serde_json::json!({
            name.clone(): {"dtype": "F32", "shape": [elements], "data_offsets": [0, elements * 4]}
        }))
        .unwrap();
        let mut file = File::create(directory.path().join(&file_name)).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        let bytes = (0..elements)
            .flat_map(|index| ((index * (shard + 1)) as f32).to_le_bytes())
            .collect::<Vec<_>>();
        file.write_all(&bytes).unwrap();
        weight_map.insert(name.clone(), file_name.into());
        expected.push(ExpectedTensor::new(name, [elements as u64]));
    }
    std::fs::write(
        directory.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    let manifest = discover_checkpoint(directory.path(), &LoadOptions::default()).unwrap();
    let plan = build_load_plan(
        &manifest,
        &expected,
        &MappingPlan::identity(&expected),
        &LoadOptions::default(),
    )
    .unwrap();
    (directory, manifest, plan)
}

/// A cancellation for shard zero, which no earlier failure can ever cancel.
fn never_cancelled(first_failure: &AtomicUsize) -> Cancellation<'_> {
    Cancellation {
        first_failure,
        index: 0,
    }
}

/// Hashes one shard alone with a full-budget buffer and no cancellation.
fn hash_one_shard(
    manifest: &CheckpointManifest,
    shard: &crate::checkpoint::ShardInventory,
    stats: &mut ProvenanceHashStats,
) -> Result<Option<BurnpackShardProvenance>> {
    hash_shard(
        manifest,
        shard,
        &mut HashBuffer::new(PROVENANCE_HASH_BUFFER_BYTES),
        never_cancelled(&AtomicUsize::new(usize::MAX)),
        stats,
    )
}

/// Concurrent hashing yields the digests of a sequential per-shard pass.
#[test]
fn concurrent_shard_hashing_matches_sequential_digests() {
    let (_directory, manifest, plan) = sharded_fixture(11, 1_000);
    let (provenance, stats) = hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
    assert_eq!(stats.passes_completed, 1);
    assert_eq!(stats.shards_hashed, 11);
    assert_eq!(stats.bytes_hashed, manifest.artifact_bytes);
    // Workers that never claim a tiny shard never allocate their share.
    assert!(stats.peak_buffer_bytes > 0);
    assert!(stats.peak_buffer_bytes <= PROVENANCE_HASH_BUFFER_BYTES);
    for shard in &manifest.shards {
        let mut sequential = ProvenanceHashStats::default();
        let digest = hash_one_shard(&manifest, shard, &mut sequential)
            .unwrap()
            .unwrap();
        assert_eq!(provenance.checkpoint_shards[&shard.relative_path], digest);
        let bytes = std::fs::read(&shard.path).unwrap();
        assert_eq!(digest.sha256, packed_payload_sha256(&bytes));
    }
}

/// Every worker count returns the same digests and stays within the buffer budget.
#[test]
fn every_worker_count_shares_one_buffer_budget() {
    let (_directory, manifest, _plan) = sharded_fixture(8, 1_000);
    let mut reference = None;
    for workers in [1, 2, 4, 8] {
        let outcomes = hash_shards_concurrently(&manifest, workers)
            .into_iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(outcomes.len(), 8, "{workers} workers");
        let share = PROVENANCE_HASH_BUFFER_BYTES / workers;
        let mut allocated = 0;
        for (_, stats) in &outcomes {
            assert!(
                stats.peak_buffer_bytes == 0 || stats.peak_buffer_bytes == share,
                "{workers} workers charged {}",
                stats.peak_buffer_bytes
            );
            allocated += stats.peak_buffer_bytes;
        }
        assert!(allocated > 0 && allocated <= PROVENANCE_HASH_BUFFER_BYTES);
        if workers == 1 {
            assert_eq!(allocated, PROVENANCE_HASH_BUFFER_BYTES);
        }
        let digests = outcomes
            .into_iter()
            .map(|(digest, _)| digest.unwrap().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(reference.get_or_insert_with(|| digests.clone()), &digests);
    }
}

/// The first failing shard in manifest order is reported, whatever the worker count.
#[test]
fn concurrent_hash_failure_reports_the_first_shard_in_order() {
    let (_directory, mut manifest, plan) = sharded_fixture(5, 64);
    let removed = manifest.shards[1].path.clone();
    let length_changed = manifest.shards[3].path.clone();
    std::fs::remove_file(&removed).unwrap();
    std::fs::write(&length_changed, b"short").unwrap();
    let is_removed_shard_error = |error: &QuantizationError| {
        matches!(
            error,
            QuantizationError::ArtifactIo {
                operation: "canonicalize checkpoint shard for provenance",
                path,
                ..
            } if *path == removed
        )
    };
    for workers in [1, 2, 4] {
        let outcomes = hash_shards_concurrently(&manifest, workers);
        // Shards before the first failure are never cancelled.
        assert!(
            matches!(&outcomes[0], Some((Ok(Some(_)), stats)) if stats.shards_hashed == 1),
            "{workers} workers"
        );
        let first_error = outcomes
            .iter()
            .flatten()
            .find_map(|(digest, _)| digest.as_ref().err())
            .unwrap();
        assert!(is_removed_shard_error(first_error), "{workers} workers");
    }
    let failure = hash_local_checkpoint_provenance_audited(&manifest, &plan).unwrap_err();
    assert!(is_removed_shard_error(&failure.error));
    let first = manifest.shards[0].file_bytes;
    let intact = [0, 2, 4]
        .iter()
        .map(|&index| manifest.shards[index].file_bytes)
        .sum::<u64>();
    // The prefix before the failure is complete; later shards may have run.
    assert!((1..=3).contains(&failure.stats.shards_hashed));
    assert!((first..=intact).contains(&failure.stats.bytes_hashed));
    assert_eq!(failure.stats.passes_completed, 0);
    manifest.shards.truncate(1);
    assert!(hash_local_checkpoint_provenance_audited(&manifest, &plan).is_ok());
}

/// One worker stops at the first failure and never opens a later shard.
#[test]
fn sequential_hashing_fails_fast_at_the_first_broken_shard() {
    let (_directory, manifest, _plan) = sharded_fixture(5, 64);
    std::fs::remove_file(&manifest.shards[1].path).unwrap();
    let outcomes = hash_shards_concurrently(&manifest, 1);
    assert!(matches!(&outcomes[0], Some((Ok(Some(_)), stats)) if stats.shards_hashed == 1));
    assert!(matches!(&outcomes[1], Some((Err(_), stats)) if stats.bytes_hashed == 0));
    assert!(
        outcomes[2..].iter().all(Option::is_none),
        "a shard after the failure was attempted"
    );
}

/// Cancellation stops before the next read but still reports file faults.
#[test]
fn cancelled_shard_stops_reading_but_still_reports_file_faults() {
    let (_directory, manifest, _plan) = sharded_fixture(2, 64);
    // Shard 0 has failed, so shard 1 is cancelled; shard 0 itself never is.
    let first_failure = AtomicUsize::new(0);
    assert!(!never_cancelled(&first_failure).is_cancelled());
    let cancelled = Cancellation {
        first_failure: &first_failure,
        index: 1,
    };
    assert!(cancelled.is_cancelled());
    let mut buffer = HashBuffer::new(PROVENANCE_HASH_BUFFER_BYTES / 2);
    let mut stats = ProvenanceHashStats::default();
    let outcome = hash_shard(
        &manifest,
        &manifest.shards[0],
        &mut buffer,
        cancelled,
        &mut stats,
    );
    assert!(matches!(outcome, Ok(None)));
    assert_eq!(stats.bytes_hashed, 0);
    assert_eq!(stats.shards_hashed, 0);
    assert_eq!(stats.peak_buffer_bytes, PROVENANCE_HASH_BUFFER_BYTES / 2);

    // The buffer is reused, so a second shard charges nothing more.
    std::fs::write(&manifest.shards[1].path, b"short").unwrap();
    let mut stats = ProvenanceHashStats::default();
    let outcome = hash_shard(
        &manifest,
        &manifest.shards[1],
        &mut buffer,
        cancelled,
        &mut stats,
    );
    assert!(matches!(
        outcome,
        Err(QuantizationError::CheckpointProvenanceLengthMismatch { .. })
    ));
    assert_eq!(stats, ProvenanceHashStats::default());
}

/// Worker counts are powers of two that divide the buffer budget exactly.
#[test]
fn hash_worker_count_is_a_bounded_power_of_two() {
    for (shards, parallelism, expected) in [
        (0, 8, 1),
        (1, 8, 1),
        (3, 8, 2),
        (5, 14, 4),
        (11, 14, 8),
        (100, 6, 4),
        (100, 1, 1),
        (100, 0, 1),
    ] {
        assert_eq!(
            hash_worker_count(shards, parallelism),
            expected,
            "{shards} shards on {parallelism} threads"
        );
    }
}

proptest::proptest! {
    /// The worker count never exceeds its limits and loses at most half of them.
    #[test]
    fn hash_worker_count_respects_every_limit(shards in 0_usize..1_000, parallelism in 0_usize..256) {
        let workers = hash_worker_count(shards, parallelism);
        let usable = parallelism.min(MAX_HASH_WORKERS).min(shards).max(1);
        proptest::prop_assert!(workers.is_power_of_two());
        proptest::prop_assert!(workers <= usable && 2 * workers > usable);
        proptest::prop_assert_eq!(PROVENANCE_HASH_BUFFER_BYTES % workers, 0);
    }
}

/// Times concurrent against one-shard-at-a-time hashing of a page-cached checkpoint.
///
/// Run with `cargo test --release --lib concurrent_hash_throughput -- --ignored --nocapture`.
#[test]
#[ignore = "timing benchmark; run explicitly in release mode"]
fn concurrent_hash_throughput_against_sequential_hashing() {
    use std::time::Instant;

    let (_directory, manifest, plan) = sharded_fixture(8, 32 << 20);
    hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
    let started = Instant::now();
    hash_local_checkpoint_provenance(&manifest, &plan).unwrap();
    let concurrent = started.elapsed().as_secs_f64();
    let started = Instant::now();
    hash_shards_concurrently(&manifest, 1);
    let sequential = started.elapsed().as_secs_f64();
    let gib = manifest.artifact_bytes as f64 / f64::from(1_u32 << 30);
    println!(
        "provenance hashing of {gib:.2} GiB: concurrent {:.2} GiB/s, sequential {:.2} GiB/s",
        gib / concurrent,
        gib / sequential
    );
}
