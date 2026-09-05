//! Auditable checkpoint load reports and memory estimates.

use serde::{Deserialize, Serialize};

use super::{ResolvedIndexTensorAlias, TensorDType};

/// Conservative memory figures computed before parameter allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MemoryEstimate {
    /// Sum of complete SafeTensors shard file sizes.
    pub artifact_bytes: u64,
    /// Sum of source tensor payload bytes.
    pub checkpoint_tensor_bytes: u64,
    /// Bytes allocated for unique, non-aliased target parameters.
    pub target_parameter_bytes: u64,
    /// Logical model bytes when tied aliases are counted independently.
    pub logical_model_bytes: u64,
    /// Largest complete shard mapping/open-file window.
    pub largest_shard_bytes: u64,
    /// Largest source tensor payload.
    pub largest_source_tensor_bytes: u64,
    /// Largest materialized target tensor produced by one operation.
    pub largest_target_tensor_bytes: u64,
    /// Largest pair of raw checkpoint payloads compared for exact equivalence.
    #[serde(default)]
    pub largest_equivalence_pair_bytes: u64,
    /// Conservative pass-two host bound for transforms and equivalence preflight.
    ///
    /// Memory mapping can make physical RSS lower than this value, while backend
    /// transfer queues can make process-wide usage higher. The bound is the
    /// greater of largest-shard-plus-output and the largest simultaneously held
    /// physical alias/owner pair.
    pub peak_host_staging_bytes: u64,
}

/// Records a tensor whose key or layout changes during loading.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformRecord {
    /// Target model parameter.
    pub target: String,
    /// Source checkpoint tensors.
    pub sources: Vec<String>,
    /// Human-readable, stable operation name.
    pub operation: String,
}

/// Records one tied parameter alias.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AliasRecord {
    /// Model parameter receiving the alias.
    pub target: String,
    /// Already loaded target parameter that owns storage.
    pub source_target: String,
}

/// Records a floating-point dtype conversion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DTypeConversion {
    /// Target model parameter.
    pub target: String,
    /// Checkpoint dtype.
    pub source: TensorDType,
    /// Materialized model dtype.
    pub target_dtype: TensorDType,
}

/// Records runtime quantization applied while streaming a parameter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationRecord {
    /// Target model parameter.
    pub target: String,
    /// Stable scheme name, such as `symmetric-q4-g64-axis0`.
    pub scheme: String,
    /// Quantized byte count written to the target store.
    pub bytes: u64,
}

/// Records one physical checkpoint tensor required to equal a storage owner.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CheckpointEquivalenceRecord {
    /// Redundant physical checkpoint key.
    pub alias: String,
    /// Canonical physical checkpoint key owning the model storage.
    pub owner: String,
}

impl CheckpointEquivalenceRecord {
    /// Creates one exact physical alias-to-owner equivalence record.
    #[must_use]
    pub fn new(alias: impl Into<String>, owner: impl Into<String>) -> Self {
        Self {
            alias: alias.into(),
            owner: owner.into(),
        }
    }
}

/// Exhaustive accounting of a checkpoint load attempt.
///
/// Vectors are sorted and deduplicated by [`LoadReport::finalize`] so reports are
/// reproducible across filesystems and shard-map iteration orders.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct LoadReport {
    /// Parameters successfully applied to the model.
    pub applied: Vec<String>,
    /// Key/layout transformations planned or performed.
    pub transformed: Vec<TransformRecord>,
    /// Tied parameter aliases planned or performed.
    pub aliased: Vec<AliasRecord>,
    /// Index-only checkpoint keys resolved to physical storage-owning keys.
    ///
    /// These describe artifact serialization, whereas [`Self::aliased`]
    /// describes logical parameters that share storage in the executable model.
    #[serde(default)]
    pub checkpoint_aliases: Vec<ResolvedIndexTensorAlias>,
    /// Physical checkpoint equivalences that pass two must verify before writes.
    #[serde(default)]
    pub equivalence_planned: Vec<CheckpointEquivalenceRecord>,
    /// Physical checkpoint equivalences already verified byte-for-byte.
    #[serde(default)]
    pub equivalence_verified: Vec<CheckpointEquivalenceRecord>,
    /// Dtype conversions planned or performed.
    pub dtype_converted: Vec<DTypeConversion>,
    /// Weight-only quantization performed by the runtime loader.
    pub quantized: Vec<QuantizationRecord>,
    /// Unexpected source tensors accepted by an explicit allowlist.
    pub allowlisted_ignored: Vec<String>,
    /// Missing model parameters accepted by an explicit allowlist.
    pub allowlisted_missing: Vec<String>,
    /// Expected model parameters with no mapping.
    pub missing: Vec<String>,
    /// Checkpoint tensors not consumed or allowlisted.
    pub unexpected: Vec<String>,
    /// Pre-allocation memory estimate, when planning reached that stage.
    pub memory: Option<MemoryEstimate>,
}

impl LoadReport {
    /// Creates an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true when strict validation found no unexplained mismatch.
    #[must_use]
    pub fn is_strictly_valid(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty()
    }

    /// Records a parameter after the runtime has applied it.
    pub fn record_applied(&mut self, target: impl Into<String>) {
        self.applied.push(target.into());
    }

    /// Records a quantized parameter after the runtime has persisted it.
    pub fn record_quantized(&mut self, record: QuantizationRecord) {
        self.quantized.push(record);
    }

    /// Records one successful raw-payload equivalence comparison.
    pub fn record_equivalence_verified(&mut self, record: CheckpointEquivalenceRecord) {
        self.equivalence_verified.push(record);
    }

    /// Sorts and deduplicates all set-like report categories.
    pub fn finalize(&mut self) {
        self.applied.sort();
        self.applied.dedup();
        self.allowlisted_ignored.sort();
        self.allowlisted_ignored.dedup();
        self.allowlisted_missing.sort();
        self.allowlisted_missing.dedup();
        self.missing.sort();
        self.missing.dedup();
        self.unexpected.sort();
        self.unexpected.dedup();
        self.transformed.sort_by(|a, b| a.target.cmp(&b.target));
        self.transformed.dedup();
        self.aliased.sort_by(|a, b| a.target.cmp(&b.target));
        self.aliased.dedup();
        self.checkpoint_aliases.sort();
        self.checkpoint_aliases.dedup();
        self.equivalence_planned.sort();
        self.equivalence_planned.dedup();
        self.equivalence_verified.sort();
        self.equivalence_verified.dedup();
        self.dtype_converted.sort_by(|a, b| a.target.cmp(&b.target));
        self.dtype_converted.dedup();
        self.quantized.sort_by(|a, b| a.target.cmp(&b.target));
        self.quantized.dedup();
    }
}

#[cfg(test)]
mod tests {
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
}
