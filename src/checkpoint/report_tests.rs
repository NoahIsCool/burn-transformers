use super::*;

#[test]
fn report_finalization_is_deterministic_and_idempotent() {
    let mut report = LoadReport {
        applied: vec!["z".into(), "a".into(), "a".into()],
        unexpected: vec!["b".into(), "a".into(), "b".into()],
        checkpoint_aliases: vec![
            ResolvedIndexTensorAlias::new("z.alias", "z.owner", "z.safetensors"),
            ResolvedIndexTensorAlias::new("a.alias", "a.owner", "a.safetensors"),
            ResolvedIndexTensorAlias::new("a.alias", "a.owner", "a.safetensors"),
        ],
        equivalence_planned: vec![
            CheckpointEquivalenceRecord::new("z.copy", "z.owner"),
            CheckpointEquivalenceRecord::new("a.copy", "a.owner"),
            CheckpointEquivalenceRecord::new("a.copy", "a.owner"),
        ],
        ..LoadReport::default()
    };
    report.finalize();
    let once = report.clone();
    report.finalize();
    assert_eq!(report, once);
    assert_eq!(report.applied, ["a", "z"]);
    assert_eq!(report.unexpected, ["a", "b"]);
    assert_eq!(
        report.checkpoint_aliases,
        [
            ResolvedIndexTensorAlias::new("a.alias", "a.owner", "a.safetensors"),
            ResolvedIndexTensorAlias::new("z.alias", "z.owner", "z.safetensors")
        ]
    );
    assert_eq!(
        report.equivalence_planned,
        [
            CheckpointEquivalenceRecord::new("a.copy", "a.owner"),
            CheckpointEquivalenceRecord::new("z.copy", "z.owner")
        ]
    );
}

#[test]
fn legacy_report_without_alias_equivalence_fields_deserializes_with_defaults() {
    let mut value = serde_json::to_value(LoadReport {
        memory: Some(MemoryEstimate {
            artifact_bytes: 11,
            checkpoint_tensor_bytes: 7,
            target_parameter_bytes: 5,
            logical_model_bytes: 5,
            largest_shard_bytes: 7,
            largest_source_tensor_bytes: 4,
            largest_target_tensor_bytes: 4,
            largest_equivalence_pair_bytes: 0,
            peak_host_staging_bytes: 11,
        }),
        ..LoadReport::default()
    })
    .unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("checkpoint_aliases")
        .unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("equivalence_planned")
        .unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("equivalence_verified")
        .unwrap();
    value["memory"]
        .as_object_mut()
        .unwrap()
        .remove("largest_equivalence_pair_bytes")
        .unwrap();

    let decoded: LoadReport = serde_json::from_value(value).unwrap();

    assert!(decoded.checkpoint_aliases.is_empty());
    assert!(decoded.equivalence_planned.is_empty());
    assert!(decoded.equivalence_verified.is_empty());
    assert_eq!(
        decoded
            .memory
            .expect("legacy memory estimate should remain present")
            .largest_equivalence_pair_bytes,
        0
    );
}

#[test]
fn strict_validity_ignores_explicit_exceptions() {
    let mut report = LoadReport {
        allowlisted_ignored: vec!["vision.weight".into()],
        allowlisted_missing: vec!["optional.bias".into()],
        ..LoadReport::default()
    };
    assert!(report.is_strictly_valid());
    report.missing.push("required.weight".into());
    assert!(!report.is_strictly_valid());
}
