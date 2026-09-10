use crate::checkpoint::{
    CheckpointEquivalenceRecord, ExecutionStats, LoadReport, MemoryEstimate, QuantizationRecord,
    SourceStats,
    quantization::{
        AuditedQuantizedExecutionOutcome, BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2, PackedLoadStats, PackedPreflightStats,
        ProvenanceExecutionStats, ProvenanceHashStats, QuantizedExecutionOutcome,
    },
};

use super::{LoadedModel, ModelSourceMetadata};

/// Source metadata canonicalizes roots and preserves pinned provenance.
#[test]
fn local_source_is_canonical_and_pinned() {
    let directory = tempfile::tempdir().expect("temporary source root");
    let source = ModelSourceMetadata::local(directory.path(), "qwen3_5_text")
        .expect("valid local source")
        .with_repository("Qwen/Qwen3.5-0.8B")
        .expect("valid repository")
        .with_revision("2fc06364715b967f1860aea9cf38778875588b17")
        .expect("valid revision");
    assert!(source.root().is_absolute());
    assert_eq!(source.model_type(), "qwen3_5_text");
    assert_eq!(source.repository(), Some("Qwen/Qwen3.5-0.8B"));
    assert_eq!(
        source.revision(),
        Some("2fc06364715b967f1860aea9cf38778875588b17")
    );
}

/// Invalid roots and ambiguous textual identifiers produce contextual errors.
#[test]
fn local_source_rejects_invalid_roots_and_labels() {
    let directory = tempfile::tempdir().expect("temporary source root");
    let file = directory.path().join("config.json");
    std::fs::write(&file, b"{}").expect("fixture file");
    assert!(ModelSourceMetadata::local(&file, "t5").is_err());
    assert!(ModelSourceMetadata::local(directory.path().join("missing"), "t5").is_err());
    assert!(ModelSourceMetadata::local(directory.path(), "").is_err());

    let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
    assert!(
        source
            .clone()
            .with_repository(" google/flan-t5-small")
            .is_err()
    );
    assert!(source.with_revision("bad\nrevision").is_err());
}

/// A successful model bundle cannot retain a tokenizer with out-of-range IDs.
#[cfg(feature = "tokenizers")]
#[test]
fn loaded_model_cross_validates_tokenizer_vocabulary() {
    use std::collections::HashMap;

    use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

    let directory = tempfile::tempdir().expect("temporary tokenizer root");
    let vocabulary = HashMap::from([("zero".to_owned(), 0_u32), ("one".to_owned(), 1_u32)])
        .into_iter()
        .collect();
    let model = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("zero".into())
        .build()
        .expect("word-level tokenizer");
    Tokenizer::new(model)
        .save(directory.path().join("tokenizer.json"), false)
        .expect("save tokenizer fixture");
    let tokenizer =
        crate::LocalTokenizer::from_directory(directory.path()).expect("load tokenizer fixture");
    let source =
        ModelSourceMetadata::local(directory.path(), "qwen3_5_text").expect("valid source root");

    let error = LoadedModel::new(
        (),
        Some(tokenizer),
        Default::default(),
        source,
        LoadReport::new(),
        1,
    )
    .expect_err("tokenizer ID one cannot index a one-row model vocabulary");
    assert!(matches!(error, crate::Error::Tokenizer(_)));
    assert!(error.to_string().contains("ID 1"));
}

/// Construction finalizes a strict report and round-trips every component.
#[test]
fn loaded_model_validates_and_preserves_components() {
    let directory = tempfile::tempdir().expect("temporary source root");
    let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
    let mut report = LoadReport::new();
    report.applied = vec!["z.weight".into(), "a.weight".into(), "a.weight".into()];
    let loaded = LoadedModel::new(
        17_u8,
        None,
        Default::default(),
        source.clone(),
        report,
        32_128,
    )
    .expect("strict load should bundle");
    assert_eq!(*loaded.model(), 17);
    assert!(loaded.tokenizer().is_none());
    assert_eq!(loaded.source(), &source);
    assert_eq!(loaded.load_report().applied, ["a.weight", "z.weight"]);
    assert_eq!(loaded.generation_defaults().max_new_tokens, 256);
    assert!(loaded.execution_stats().is_none());
    assert!(loaded.quantized_load_audit().is_none());

    let (model, tokenizer, _, unpacked_source, unpacked_report) = loaded.into_parts();
    assert_eq!(model, 17);
    assert!(tokenizer.is_none());
    assert_eq!(unpacked_source, source);
    assert_eq!(unpacked_report.applied, ["a.weight", "z.weight"]);
}

/// Execution-aware construction and consumption retain exact audit counters.
#[test]
fn loaded_model_preserves_execution_stats_without_changing_legacy_parts() {
    let directory = tempfile::tempdir().expect("temporary source root");
    let source =
        ModelSourceMetadata::local(directory.path(), "qwen3_moe").expect("valid local source");
    let stats = ExecutionStats {
        tensors_applied: 2,
        aliases_applied: 1,
        equivalences_verified: 0,
        equivalence_source_reads: 0,
        equivalence_source_payload_bytes: 0,
        source_reads: 2,
        source_payload_bytes: 96,
        delivered_tensor_bytes: 96,
        largest_source_payload_bytes: 64,
        largest_output_tensor_bytes: 64,
        largest_equivalence_pair_bytes: 0,
        peak_host_staging_bytes: 128,
        source: SourceStats {
            shard_opens: 1,
            tensor_reads: 2,
            payload_bytes_read: 96,
            max_open_shards: 1,
            shard_open_order: vec!["model.safetensors".into()],
        },
    };
    let mut report = LoadReport::new();
    report.applied = vec![
        "model.embed_tokens.weight".into(),
        "model.norm.weight".into(),
        "lm_head.weight".into(),
    ];
    let loaded = LoadedModel::new_with_execution_stats(
        23_u8,
        None,
        Default::default(),
        source.clone(),
        report,
        stats.clone(),
        32_128,
    )
    .expect("strict execution-aware bundle");

    assert_eq!(loaded.execution_stats(), Some(&stats));
    assert!(loaded.quantized_load_audit().is_none());
    let (model, tokenizer, _, unpacked_source, unpacked_report, unpacked_stats) =
        loaded.into_parts_with_execution_stats();
    assert_eq!(model, 23);
    assert!(tokenizer.is_none());
    assert_eq!(unpacked_source, source);
    assert!(unpacked_report.is_strictly_valid());
    assert_eq!(unpacked_stats, Some(stats));
}

/// Quantized construction retains all three execution counter domains.
#[test]
fn loaded_model_preserves_quantized_load_audit() {
    let directory = tempfile::tempdir().expect("temporary source root");
    let source = ModelSourceMetadata::local(directory.path(), "qwen3_moe").expect("valid source");
    let mut report = LoadReport::new();
    report
        .applied
        .push("model.layers.0.mlp.down_proj.weight".into());
    report.quantized.push(QuantizationRecord {
        target: "model.layers.0.mlp.down_proj.weight".into(),
        scheme: "burn-qfloat-symmetric-q8-g64-block-1xg".into(),
        bytes: 72,
    });
    report.memory = Some(MemoryEstimate {
        artifact_bytes: 4_096,
        peak_host_staging_bytes: 256,
        ..MemoryEstimate::default()
    });
    let execution = ExecutionStats {
        tensors_applied: 1,
        delivered_tensor_bytes: 256,
        largest_output_tensor_bytes: 256,
        peak_host_staging_bytes: 256,
        ..ExecutionStats::default()
    };
    let hash_pass = ProvenanceHashStats {
        passes_completed: 1,
        shards_hashed: 2,
        bytes_hashed: 4_096,
        peak_buffer_bytes: 1_048_576,
    };
    let packed = PackedLoadStats {
        tensor_reads: 1,
        payload_bytes: 72,
        largest_payload_bytes: 72,
        tensors_applied: 1,
    };
    let preflight = PackedPreflightStats {
        passes_started: 1,
        passes_completed: 1,
        tensor_reads: 1,
        payload_bytes: 72,
        largest_payload_bytes: 72,
        peak_owned_payload_bytes: 144,
    };
    let provenance = ProvenanceExecutionStats {
        pre_execution: Some(hash_pass),
        post_execution: Some(hash_pass),
    };
    let loaded = LoadedModel::new_with_quantized_execution(
        29_u8,
        None,
        Default::default(),
        source.clone(),
        AuditedQuantizedExecutionOutcome {
            format_version: BURNPACK_QUANTIZATION_FORMAT_VERSION_V2,
            preflight,
            execution: QuantizedExecutionOutcome {
                report,
                execution: execution.clone(),
                packed,
                provenance,
            },
        },
        32_128,
    )
    .expect("consistent quantized outcome must bundle");

    assert_eq!(loaded.execution_stats(), Some(&execution));
    let audit = loaded
        .quantized_load_audit()
        .expect("quantized outcome must retain its sidecar audit");
    assert_eq!(
        audit.format_version(),
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    assert_eq!(audit.preflight(), &preflight);
    assert_eq!(audit.packed(), &packed);
    assert_eq!(audit.provenance(), &provenance);

    let (model, tokenizer, _, unpacked_source, unpacked_report, unpacked_stats, unpacked_audit) =
        loaded.into_parts_with_quantized_load_audit();
    assert_eq!(model, 29);
    assert!(tokenizer.is_none());
    assert_eq!(unpacked_source, source);
    assert_eq!(unpacked_report.quantized.len(), 1);
    assert_eq!(unpacked_stats, Some(execution));
    let unpacked_audit = unpacked_audit.expect("retained audit");
    assert_eq!(unpacked_audit.preflight(), &preflight);
    assert_eq!(unpacked_audit.packed(), &packed);
}

/// Contradictory packed or provenance counters cannot escape as success.
#[test]
fn loaded_model_rejects_inconsistent_quantized_load_audit() {
    let directory = tempfile::tempdir().expect("temporary source root");
    let source = ModelSourceMetadata::local(directory.path(), "qwen3_moe").expect("valid source");
    let mut report = LoadReport::new();
    report
        .applied
        .push("model.layers.0.mlp.down_proj.weight".into());
    report.quantized.push(QuantizationRecord {
        target: "model.layers.0.mlp.down_proj.weight".into(),
        scheme: "burn-qfloat-symmetric-q8-g64-block-1xg".into(),
        bytes: 72,
    });
    report.memory = Some(MemoryEstimate {
        artifact_bytes: 4_096,
        ..MemoryEstimate::default()
    });
    let execution = ExecutionStats {
        tensors_applied: 1,
        ..ExecutionStats::default()
    };
    let packed = PackedLoadStats {
        tensor_reads: 1,
        payload_bytes: 72,
        largest_payload_bytes: 72,
        tensors_applied: 1,
    };
    let preflight = PackedPreflightStats {
        passes_started: 1,
        passes_completed: 1,
        tensor_reads: 1,
        payload_bytes: 72,
        largest_payload_bytes: 72,
        peak_owned_payload_bytes: 144,
    };
    let hash_pass = ProvenanceHashStats {
        passes_completed: 1,
        shards_hashed: 1,
        bytes_hashed: 4_096,
        peak_buffer_bytes: 1_048_576,
    };
    let construct = |format_version, report, preflight, packed, provenance| {
        LoadedModel::new_with_quantized_execution(
            (),
            None,
            Default::default(),
            source.clone(),
            AuditedQuantizedExecutionOutcome {
                format_version,
                preflight,
                execution: QuantizedExecutionOutcome {
                    report,
                    execution: execution.clone(),
                    packed,
                    provenance,
                },
            },
            32,
        )
    };

    for inconsistent in [
        PackedPreflightStats {
            passes_started: 0,
            ..preflight
        },
        PackedPreflightStats {
            passes_completed: 0,
            ..preflight
        },
        PackedPreflightStats {
            tensor_reads: 2,
            ..preflight
        },
        PackedPreflightStats {
            payload_bytes: 73,
            ..preflight
        },
        PackedPreflightStats {
            largest_payload_bytes: 73,
            ..preflight
        },
        PackedPreflightStats {
            peak_owned_payload_bytes: 0,
            ..preflight
        },
        PackedPreflightStats {
            peak_owned_payload_bytes: 71,
            ..preflight
        },
    ] {
        assert!(
            construct(
                BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
                report.clone(),
                inconsistent,
                packed,
                ProvenanceExecutionStats::default(),
            )
            .is_err()
        );
    }

    for inconsistent in [
        PackedLoadStats {
            tensor_reads: 2,
            ..packed
        },
        PackedLoadStats {
            payload_bytes: 73,
            ..packed
        },
        PackedLoadStats {
            tensors_applied: 0,
            ..packed
        },
    ] {
        assert!(
            construct(
                BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
                report.clone(),
                preflight,
                inconsistent,
                ProvenanceExecutionStats::default(),
            )
            .is_err()
        );
    }

    let zero_largest = PackedLoadStats {
        largest_payload_bytes: 0,
        ..packed
    };
    assert!(
        construct(
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
            report.clone(),
            PackedPreflightStats {
                largest_payload_bytes: 0,
                ..preflight
            },
            zero_largest,
            ProvenanceExecutionStats::default(),
        )
        .is_err()
    );
    let largest_above_total = PackedLoadStats {
        payload_bytes: 71,
        ..packed
    };
    assert!(
        construct(
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
            report.clone(),
            PackedPreflightStats {
                payload_bytes: 71,
                ..preflight
            },
            largest_above_total,
            ProvenanceExecutionStats::default(),
        )
        .is_err()
    );

    assert!(
        construct(
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2,
            report.clone(),
            preflight,
            packed,
            ProvenanceExecutionStats::default(),
        )
        .is_err()
    );
    assert!(
        construct(
            99,
            report.clone(),
            preflight,
            packed,
            ProvenanceExecutionStats::default(),
        )
        .is_err()
    );
    construct(
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
        report.clone(),
        preflight,
        packed,
        ProvenanceExecutionStats::default(),
    )
    .expect("v1 explicitly explains the absence of provenance hashes");

    assert!(
        construct(
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2,
            report.clone(),
            preflight,
            packed,
            ProvenanceExecutionStats {
                pre_execution: Some(hash_pass),
                post_execution: None,
            }
        )
        .is_err()
    );
    assert!(
        construct(
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2,
            report.clone(),
            preflight,
            packed,
            ProvenanceExecutionStats {
                pre_execution: Some(ProvenanceHashStats {
                    passes_completed: 0,
                    ..hash_pass
                }),
                post_execution: Some(hash_pass),
            }
        )
        .is_err()
    );
    assert!(
        construct(
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2,
            report.clone(),
            preflight,
            packed,
            ProvenanceExecutionStats {
                pre_execution: Some(hash_pass),
                post_execution: Some(ProvenanceHashStats {
                    bytes_hashed: hash_pass.bytes_hashed + 1,
                    ..hash_pass
                }),
            }
        )
        .is_err()
    );

    let after_wrong_bytes = construct(
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2,
        report.clone(),
        preflight,
        packed,
        ProvenanceExecutionStats {
            pre_execution: Some(hash_pass),
            post_execution: Some(ProvenanceHashStats {
                bytes_hashed: 4_095,
                ..hash_pass
            }),
        },
    )
    .expect_err("the post-execution pass must match planned artifact bytes");
    assert!(after_wrong_bytes.to_string().contains("post-execution"));

    let mut overflowing_report = report;
    overflowing_report.applied.push("second.weight".into());
    overflowing_report.quantized[0].bytes = u64::MAX;
    overflowing_report.quantized.push(QuantizationRecord {
        target: "second.weight".into(),
        scheme: "burn-qfloat-symmetric-q8-g64-block-1xg".into(),
        bytes: 1,
    });
    let overflowing_execution = ExecutionStats {
        tensors_applied: 2,
        ..ExecutionStats::default()
    };
    let overflowing = LoadedModel::new_with_quantized_execution(
        (),
        None,
        Default::default(),
        source,
        AuditedQuantizedExecutionOutcome {
            format_version: BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
            preflight: PackedPreflightStats {
                passes_started: 1,
                passes_completed: 1,
                tensor_reads: 2,
                payload_bytes: u64::MAX,
                largest_payload_bytes: u64::MAX,
                peak_owned_payload_bytes: u64::MAX,
            },
            execution: QuantizedExecutionOutcome {
                report: overflowing_report,
                execution: overflowing_execution,
                packed: PackedLoadStats {
                    tensor_reads: 2,
                    payload_bytes: u64::MAX,
                    largest_payload_bytes: u64::MAX,
                    tensors_applied: 2,
                },
                provenance: ProvenanceExecutionStats::default(),
            },
        },
        32,
    )
    .expect_err("overflowing report byte sums must fail closed");
    assert!(overflowing.to_string().contains("overflowed"));
}

/// Invalid generation controls and non-strict reports cannot escape loaders.
#[test]
fn loaded_model_rejects_invalid_runtime_metadata() {
    let directory = tempfile::tempdir().expect("temporary source root");
    let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
    let invalid_generation = crate::GenerationConfig {
        max_new_tokens: 0,
        ..Default::default()
    };
    assert!(
        LoadedModel::new(
            (),
            None,
            invalid_generation,
            source.clone(),
            LoadReport::new(),
            32
        )
        .is_err()
    );

    let mut report = LoadReport::new();
    report.missing = vec!["decoder.weight".into(), "decoder.weight".into()];
    report.unexpected.push("foreign.weight".into());
    let error = LoadedModel::new((), None, Default::default(), source, report, 32)
        .expect_err("non-strict report must fail");
    assert!(error.to_string().contains("1 missing and 1 unexpected"));
}

/// Contradictory retained counters cannot be presented as a successful audit.
#[test]
fn loaded_model_rejects_inconsistent_execution_stats() {
    let directory = tempfile::tempdir().expect("temporary source root");
    let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
    let construct = |stats| {
        LoadedModel::new_with_execution_stats(
            (),
            None,
            Default::default(),
            source.clone(),
            LoadReport::new(),
            stats,
            32,
        )
    };

    let write_overflow = ExecutionStats {
        tensors_applied: usize::MAX,
        aliases_applied: 1,
        ..ExecutionStats::default()
    };
    assert!(
        construct(write_overflow)
            .unwrap_err()
            .to_string()
            .contains("overflowed")
    );

    let write_mismatch = ExecutionStats {
        tensors_applied: 1,
        ..ExecutionStats::default()
    };
    assert!(
        construct(write_mismatch)
            .unwrap_err()
            .to_string()
            .contains("1 successful writes")
    );

    let read_mismatch = ExecutionStats {
        source_reads: 1,
        ..ExecutionStats::default()
    };
    assert!(
        construct(read_mismatch)
            .unwrap_err()
            .to_string()
            .contains("1 source reads")
    );

    let byte_mismatch = ExecutionStats {
        source_payload_bytes: 1,
        ..ExecutionStats::default()
    };
    assert!(
        construct(byte_mismatch)
            .unwrap_err()
            .to_string()
            .contains("1 source bytes")
    );

    let equivalence = CheckpointEquivalenceRecord::new("copy", "owner");
    let equivalence_report = LoadReport {
        equivalence_planned: vec![equivalence.clone()],
        equivalence_verified: vec![equivalence],
        memory: Some(MemoryEstimate {
            largest_equivalence_pair_bytes: 8,
            peak_host_staging_bytes: 8,
            ..MemoryEstimate::default()
        }),
        ..LoadReport::default()
    };
    let equivalence_stats = ExecutionStats {
        equivalences_verified: 1,
        equivalence_source_reads: 2,
        equivalence_source_payload_bytes: 8,
        source_reads: 2,
        source_payload_bytes: 8,
        largest_source_payload_bytes: 4,
        largest_equivalence_pair_bytes: 8,
        peak_host_staging_bytes: 8,
        source: SourceStats {
            tensor_reads: 2,
            payload_bytes_read: 8,
            ..SourceStats::default()
        },
        ..ExecutionStats::default()
    };
    LoadedModel::new_with_execution_stats(
        (),
        None,
        Default::default(),
        source.clone(),
        equivalence_report.clone(),
        equivalence_stats.clone(),
        32,
    )
    .expect("consistent equivalence accounting must remain loadable");

    for stats in [
        ExecutionStats {
            equivalences_verified: 0,
            ..equivalence_stats.clone()
        },
        ExecutionStats {
            equivalence_source_reads: 1,
            ..equivalence_stats.clone()
        },
        ExecutionStats {
            equivalence_source_payload_bytes: 7,
            ..equivalence_stats.clone()
        },
        ExecutionStats {
            largest_equivalence_pair_bytes: 7,
            ..equivalence_stats.clone()
        },
        ExecutionStats {
            peak_host_staging_bytes: 9,
            ..equivalence_stats.clone()
        },
    ] {
        assert!(
            LoadedModel::new_with_execution_stats(
                (),
                None,
                Default::default(),
                source.clone(),
                equivalence_report.clone(),
                stats,
                32,
            )
            .is_err()
        );
    }

    let incomplete_report = LoadReport {
        equivalence_verified: Vec::new(),
        ..equivalence_report
    };
    assert!(
        LoadedModel::new_with_execution_stats(
            (),
            None,
            Default::default(),
            source,
            incomplete_report,
            equivalence_stats,
            32,
        )
        .unwrap_err()
        .to_string()
        .contains("verified 0 of 1")
    );
}

/// The compatibility constructor cannot bless an unexecuted equivalence plan.
#[test]
fn loaded_model_rejects_unverified_equivalence_without_execution_stats() {
    let directory = tempfile::tempdir().expect("temporary source root");
    let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
    let report = LoadReport {
        equivalence_planned: vec![CheckpointEquivalenceRecord::new(
            "decoder.embed_tokens.weight",
            "shared.weight",
        )],
        ..LoadReport::default()
    };

    let error = LoadedModel::new((), None, Default::default(), source, report, 32)
        .expect_err("unverified physical equivalence must not be presented as loaded");

    assert!(error.to_string().contains("0 of 1"));
}

/// Creates a consistent two-matrix audit with independently checkable byte sums.
fn two_matrix_audit() -> AuditedQuantizedExecutionOutcome {
    AuditedQuantizedExecutionOutcome {
        format_version: BURNPACK_QUANTIZATION_FORMAT_VERSION_V1,
        preflight: PackedPreflightStats {
            passes_started: 1,
            passes_completed: 1,
            tensor_reads: 2,
            payload_bytes: 72,
            largest_payload_bytes: 40,
            peak_owned_payload_bytes: 80,
        },
        execution: QuantizedExecutionOutcome {
            report: LoadReport {
                applied: vec!["a.weight".into(), "b.weight".into()],
                quantized: [("a.weight", 32), ("b.weight", 40)]
                    .map(|(target, bytes)| QuantizationRecord {
                        target: target.into(),
                        scheme: "burn-qfloat-symmetric-q8-g64-block-1xg".into(),
                        bytes,
                    })
                    .into(),
                ..LoadReport::default()
            },
            execution: ExecutionStats {
                tensors_applied: 2,
                ..ExecutionStats::default()
            },
            packed: PackedLoadStats {
                tensor_reads: 2,
                payload_bytes: 72,
                largest_payload_bytes: 40,
                tensors_applied: 2,
            },
            provenance: ProvenanceExecutionStats::default(),
        },
    }
}

/// Exercises the public bundle boundary without allocating backend parameters.
fn bundle_audit(
    source: &ModelSourceMetadata,
    outcome: AuditedQuantizedExecutionOutcome,
) -> crate::Result<LoadedModel<()>> {
    LoadedModel::new_with_quantized_execution(
        (),
        None,
        Default::default(),
        source.clone(),
        outcome,
        32,
    )
}

/// Returns a completed non-empty checkpoint hash pass without a memory plan.
fn completed_hash_pass() -> ProvenanceHashStats {
    ProvenanceHashStats {
        passes_completed: 1,
        shards_hashed: 1,
        bytes_hashed: 4_096,
        peak_buffer_bytes: 1_024,
    }
}

/// Mutable and consuming model access leave source and audit metadata intact.
#[test]
fn mutable_model_access_and_consumption_preserve_bundle_ownership() {
    let directory = tempfile::tempdir().unwrap();
    let source = ModelSourceMetadata::local(directory.path(), "t5").unwrap();
    let mut loaded = LoadedModel::new(
        vec![1_u32, 2],
        None,
        Default::default(),
        source.clone(),
        LoadReport::new(),
        32,
    )
    .unwrap();
    loaded.model_mut().push(3);
    assert_eq!(loaded.model(), &[1, 2, 3]);
    assert_eq!(loaded.source(), &source);
    assert_eq!(loaded.load_report(), &LoadReport::new());
    assert_eq!(
        loaded.generation_defaults(),
        &crate::GenerationConfig::default()
    );
    assert!(loaded.execution_stats().is_none());
    assert!(loaded.quantized_load_audit().is_none());
    assert_eq!(loaded.into_model(), [1, 2, 3]);
}

/// Matching preflight/execution counters cannot conceal a missing report entry.
#[test]
fn quantized_bundle_rejects_report_install_count_mismatch() {
    let directory = tempfile::tempdir().unwrap();
    let source = ModelSourceMetadata::local(directory.path(), "t5").unwrap();
    bundle_audit(&source, two_matrix_audit()).unwrap();
    let mut outcome = two_matrix_audit();
    outcome.execution.report.quantized.pop();
    let error = bundle_audit(&source, outcome).unwrap_err();
    assert!(matches!(&error, crate::Error::Checkpoint(_)));
    assert_eq!(
        error.to_string(),
        "checkpoint error: Burnpack installed 2 tensors but the load report contains 1 quantized records"
    );
}

/// Payload totals are checked against report entries, not only against preflight.
#[test]
fn quantized_bundle_rejects_report_payload_total_mismatch() {
    let directory = tempfile::tempdir().unwrap();
    let source = ModelSourceMetadata::local(directory.path(), "t5").unwrap();
    let mut outcome = two_matrix_audit();
    outcome.execution.report.quantized[1].bytes = 39;
    let error = bundle_audit(&source, outcome).unwrap_err();
    assert!(matches!(&error, crate::Error::Checkpoint(_)));
    assert_eq!(
        error.to_string(),
        "checkpoint error: Burnpack decoded 72 payload bytes but the load report accounts for 71"
    );
}

/// Equal aggregate bytes do not excuse an inconsistent largest-payload record.
#[test]
fn quantized_bundle_rejects_report_largest_payload_mismatch() {
    let directory = tempfile::tempdir().unwrap();
    let source = ModelSourceMetadata::local(directory.path(), "t5").unwrap();
    let mut outcome = two_matrix_audit();
    for record in &mut outcome.execution.report.quantized {
        record.bytes = 36;
    }
    let error = bundle_audit(&source, outcome).unwrap_err();
    assert!(matches!(&error, crate::Error::Checkpoint(_)));
    assert_eq!(
        error.to_string(),
        "checkpoint error: largest Burnpack payload is 40, but the load report records 36"
    );
}

/// A legacy sidecar may never claim even one provenance-bound hash pass.
#[test]
fn legacy_quantized_bundle_rejects_each_nonempty_provenance_combination() {
    let directory = tempfile::tempdir().unwrap();
    let source = ModelSourceMetadata::local(directory.path(), "t5").unwrap();
    let pass = Some(completed_hash_pass());
    for (pre_execution, post_execution) in [(pass, None), (None, pass), (pass, pass)] {
        let mut outcome = two_matrix_audit();
        outcome.execution.provenance = ProvenanceExecutionStats {
            pre_execution,
            post_execution,
        };
        let error = bundle_audit(&source, outcome).unwrap_err();
        assert!(matches!(&error, crate::Error::Checkpoint(_)));
        assert_eq!(
            error.to_string(),
            "checkpoint error: legacy Burnpack v1 execution cannot claim checkpoint provenance hash passes"
        );
    }
}

/// Each v2 pass must independently record non-empty checkpoint content and work.
#[test]
fn provenance_bundle_rejects_each_empty_pre_and_post_hash_counter() {
    let directory = tempfile::tempdir().unwrap();
    let source = ModelSourceMetadata::local(directory.path(), "t5").unwrap();
    let valid = completed_hash_pass();
    let empty = [
        ProvenanceHashStats {
            shards_hashed: 0,
            ..valid
        },
        ProvenanceHashStats {
            bytes_hashed: 0,
            ..valid
        },
        ProvenanceHashStats {
            peak_buffer_bytes: 0,
            ..valid
        },
    ];
    for phase in ["pre-execution", "post-execution"] {
        for invalid in empty {
            let mut outcome = two_matrix_audit();
            outcome.format_version = BURNPACK_QUANTIZATION_FORMAT_VERSION_V2;
            outcome.execution.provenance = ProvenanceExecutionStats {
                pre_execution: Some(if phase == "pre-execution" {
                    invalid
                } else {
                    valid
                }),
                post_execution: Some(if phase == "post-execution" {
                    invalid
                } else {
                    valid
                }),
            };
            let error = bundle_audit(&source, outcome).unwrap_err();
            assert!(matches!(&error, crate::Error::Checkpoint(_)));
            assert_eq!(
                error.to_string(),
                format!(
                    "checkpoint error: successful {phase} provenance pass must hash non-empty checkpoint content"
                )
            );
        }
    }
}

/// Hash-pass agreement is required even when the caller supplies no memory plan.
#[test]
fn provenance_bundle_checks_pair_coverage_without_a_memory_estimate() {
    let directory = tempfile::tempdir().unwrap();
    let source = ModelSourceMetadata::local(directory.path(), "t5").unwrap();
    let before = completed_hash_pass();
    for after in [
        before,
        ProvenanceHashStats {
            shards_hashed: 2,
            ..before
        },
        ProvenanceHashStats {
            bytes_hashed: 4_095,
            ..before
        },
    ] {
        let mut outcome = two_matrix_audit();
        outcome.format_version = BURNPACK_QUANTIZATION_FORMAT_VERSION_V2;
        outcome.execution.provenance = ProvenanceExecutionStats {
            pre_execution: Some(before),
            post_execution: Some(after),
        };
        assert!(outcome.execution.report.memory.is_none());
        let result = bundle_audit(&source, outcome);
        if after == before {
            let loaded = result.unwrap();
            assert_eq!(
                loaded
                    .quantized_load_audit()
                    .unwrap()
                    .provenance()
                    .post_execution,
                Some(after)
            );
        } else {
            let error = result.unwrap_err();
            assert!(matches!(&error, crate::Error::Checkpoint(_)));
            assert_eq!(
                error.to_string(),
                format!(
                    "checkpoint error: pre/post provenance coverage differs: 1 shards and 4096 bytes before, {} shards and {} bytes after",
                    after.shards_hashed, after.bytes_hashed
                )
            );
        }
    }
}

/// Equivalence reads form a subset of all source reads, including their bytes.
#[test]
fn bundle_rejects_equivalence_bytes_above_total_source_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let source = ModelSourceMetadata::local(directory.path(), "t5").unwrap();
    let pair = CheckpointEquivalenceRecord::new("copy", "owner");
    let report = LoadReport {
        equivalence_planned: vec![pair.clone()],
        equivalence_verified: vec![pair],
        ..LoadReport::default()
    };
    let stats = ExecutionStats {
        equivalences_verified: 1,
        equivalence_source_reads: 2,
        equivalence_source_payload_bytes: 9,
        source_reads: 2,
        source_payload_bytes: 8,
        source: SourceStats {
            tensor_reads: 2,
            payload_bytes_read: 8,
            ..SourceStats::default()
        },
        ..ExecutionStats::default()
    };
    let error = LoadedModel::new_with_execution_stats(
        (),
        None,
        Default::default(),
        source,
        report,
        stats,
        32,
    )
    .unwrap_err();
    assert!(matches!(&error, crate::Error::Checkpoint(_)));
    assert_eq!(
        error.to_string(),
        "checkpoint error: equivalence source bytes 9 exceed all checkpoint source bytes 8"
    );
}
