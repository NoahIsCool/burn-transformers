//! Auditable checkpoint load reports and memory estimates.

use serde::{Deserialize, Serialize};

use super::TensorDType;

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
    /// Conservative pass-two host bound: largest shard plus largest output buffer.
    ///
    /// Memory mapping can make physical RSS lower than this value, while backend
    /// transfer queues can make process-wide usage higher. This figure captures
    /// the memory controlled directly by a one-shard-at-a-time loader.
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
            ..LoadReport::default()
        };
        report.finalize();
        let once = report.clone();
        report.finalize();
        assert_eq!(report, once);
        assert_eq!(report.applied, ["a", "z"]);
        assert_eq!(report.unexpected, ["a", "b"]);
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
