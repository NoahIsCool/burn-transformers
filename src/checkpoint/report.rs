//! Auditable checkpoint load reports and memory estimates.
//!
//! Reports distinguish planned transformations from successful parameter writes
//! and verified physical equivalences. A partial failure retains the exact
//! successful write set; sorting the report does not imply execution order.
//! Storage estimates distinguish logical aliases from unique parameter owners,
//! and loader staging bounds exclude activations and backend transfer queues.
//! A strictly valid report establishes loading consistency, not reference parity
//! or advertised checkpoint support.

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
    /// Largest complete shard file size.
    ///
    /// This is an artifact metric, not loader staging: the local source reads
    /// one tensor interval at a time and never maps or buffers a whole shard.
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
    /// [`crate::checkpoint::build_load_plan`] sets it to the maximum of three
    /// terms: the largest shard plus the largest materialized output, the
    /// largest alias/owner payload pair compared for exact equivalence, and the
    /// largest MXFP4 decode (`blocks + scales + output`). It over-approximates
    /// what the executor stages for any single operation: one source and its
    /// output, only the source for an in-place equal-width cast, both MXFP4
    /// sources and the output, or one equivalence pair.
    /// [`crate::checkpoint::LoadPlan::required_host_staging_bytes`] computes
    /// that per-operation figure exactly. The runtime checks every planned
    /// operation against this field before reading any payload and fails with
    /// [`crate::checkpoint::CheckpointError::MemoryEstimateExceeded`] when an
    /// operation needs more. Architecture loaders may raise it to budget for
    /// staging inside their sinks, such as splitting a packed projection into
    /// components. Backend transfer queues and activations are excluded, so
    /// process-wide usage can be higher.
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

    /// Returns a planned report with every execution-progress category cleared.
    ///
    /// Execution starts from this state: `applied`, `quantized`, and
    /// `equivalence_verified` record only work that actually completes, while the
    /// planned categories are retained. The result is finalized.
    #[must_use]
    pub fn without_progress(&self) -> Self {
        let mut report = self.clone();
        report.applied.clear();
        report.quantized.clear();
        report.equivalence_verified.clear();
        report.finalize();
        report
    }

    /// Records quantized targets and restores deterministic ordering.
    pub fn extend_quantized(&mut self, records: impl IntoIterator<Item = QuantizationRecord>) {
        self.quantized.extend(records);
        self.finalize();
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
#[path = "report_tests.rs"]
mod tests;
