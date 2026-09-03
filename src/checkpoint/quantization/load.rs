//! Transactional overlay of packed Burnpack weights onto strict load plans.
//!
//! A quantized Burnpack is a weight-only sidecar. Embeddings, normalization
//! scales, biases, recurrent parameters, and MTP tensors still come from the
//! original SafeTensors checkpoint. [`execute_quantized_load_plan`] therefore
//! runs the ordinary architecture plan and interposes at its final tensor-sink
//! boundary: a matching sidecar payload replaces the already transformed source
//! tensor, while an absent payload passes through unchanged.
//!
//! Preflight happens before the first source read. Every packed name must identify
//! a storage-owning plan target with the same post-transform shape and dtype, must
//! remain eligible under the conservative policy, and must map exactly to Burn's
//! `[1, group]` block geometry. This prevents a same-shaped but differently wired
//! artifact from silently bypassing the architecture contract. Runtime ownership
//! remains bounded by one ordinary source tensor plus one decoded packed target.

use std::{error::Error as StdError, fmt};

use burn::tensor::TensorData;

use super::{
    PackedExecutionSupport, QuantizationEligibility, QuantizationError, QuantizedBurnpackArtifact,
    QuantizedTensorMetadata, SourceFloatDType, SymmetricQuantization,
    classify_weight_for_quantization,
};
use crate::checkpoint::{
    CheckpointManifest, ExecutionFailureCause, ExecutionOutcome, ExecutionStats, LoadPlan,
    LoadReport, LocalShardSource, QuantizationRecord, TensorDType, TensorSink, TensorSource,
    execute_load_plan,
};

/// Sidecar I/O counters kept separate from ordinary checkpoint source statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PackedLoadStats {
    /// Packed payloads decoded successfully, including a later sink rejection.
    pub tensor_reads: u64,
    /// Sum of packed-value and scale bytes read from the mapped artifact.
    pub payload_bytes: u64,
    /// Largest individual packed-value-plus-scale payload materialized.
    pub largest_payload_bytes: u64,
    /// Packed parameters accepted by the architecture sink.
    pub tensors_applied: usize,
}

/// Successful mixed source-precision and packed checkpoint execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedExecutionOutcome {
    /// Strict checkpoint report augmented with packed parameters actually installed.
    pub report: LoadReport,
    /// SafeTensors and generic transformation counters.
    pub execution: ExecutionStats,
    /// Mapped Burnpack payload counters.
    pub packed: PackedLoadStats,
}

/// Error raised by the packed overlay or its architecture-owned inner sink.
#[derive(Debug)]
#[non_exhaustive]
pub enum QuantizedSinkError<E> {
    /// Burnpack metadata, payload, or QFloat conversion failed.
    Quantization(QuantizationError),
    /// The architecture rejected the final source-precision or packed tensor.
    Architecture(E),
}

impl<E: fmt::Display> fmt::Display for QuantizedSinkError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quantization(error) => error.fmt(formatter),
            Self::Architecture(error) => error.fmt(formatter),
        }
    }
}

impl<E: StdError + 'static> StdError for QuantizedSinkError<E> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Quantization(error) => Some(error),
            Self::Architecture(error) => Some(error),
        }
    }
}

impl<E> From<QuantizationError> for QuantizedSinkError<E> {
    fn from(error: QuantizationError) -> Self {
        Self::Quantization(error)
    }
}

/// Audited packed-load failure including both source and sidecar progress.
#[derive(Debug)]
pub struct QuantizedExecutionFailure<E> {
    /// Target being installed, or `None` for manifest/source lifecycle failures.
    pub failed_target: Option<String>,
    /// Checkpoint, quantization, or architecture failure category.
    pub cause: ExecutionFailureCause<QuantizedSinkError<E>>,
    /// Exact successfully applied prefix, augmented with packed records.
    pub report: LoadReport,
    /// SafeTensors and generic transformation counters.
    pub execution: ExecutionStats,
    /// Sidecar reads and successfully installed packed tensor counts.
    pub packed: PackedLoadStats,
}

impl<E: fmt::Display> fmt::Display for QuantizedExecutionFailure<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(target) = &self.failed_target {
            write!(
                formatter,
                "quantized checkpoint execution failed at `{target}`: {}",
                self.cause
            )
        } else {
            write!(
                formatter,
                "quantized checkpoint execution failed: {}",
                self.cause
            )
        }
    }
}

impl<E: StdError + 'static> StdError for QuantizedExecutionFailure<E> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.cause)
    }
}

/// Rich execution result retaining partial strict reports on failure.
pub type QuantizedExecutionResult<E> =
    std::result::Result<QuantizedExecutionOutcome, Box<QuantizedExecutionFailure<E>>>;

/// Executes a strict plan while replacing matching targets with packed weights.
///
/// The source remains authoritative for checkpoint discovery, allowlists, aliases,
/// and all policy-excluded tensors. The sidecar is accepted only when its complete
/// manifest passes [`validate_sidecar_contract`] before `source.read_tensor` can be
/// called. On a later failure, the boxed [`QuantizedExecutionFailure`] contains
/// both the exact applied prefix and the underlying quantization or architecture
/// error.
pub fn execute_quantized_load_plan<S, K>(
    plan: &LoadPlan,
    source: &mut S,
    artifact: &mut QuantizedBurnpackArtifact,
    sink: &mut K,
) -> QuantizedExecutionResult<K::Error>
where
    S: TensorSource,
    K: TensorSink,
{
    if let Err(error) = validate_sidecar_contract(plan, artifact) {
        let mut report = unstarted_report(plan);
        let mut stats = ExecutionStats::default();
        if let Err(close_error) = source.close() {
            stats.source = source.stats();
            report.finalize();
            return Err(Box::new(QuantizedExecutionFailure {
                failed_target: None,
                cause: ExecutionFailureCause::Checkpoint(close_error),
                report,
                execution: stats,
                packed: PackedLoadStats::default(),
            }));
        }
        stats.source = source.stats();
        report.finalize();
        return Err(Box::new(QuantizedExecutionFailure {
            failed_target: None,
            cause: ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(error)),
            report,
            execution: stats,
            packed: PackedLoadStats::default(),
        }));
    }

    let mut overlay = QuantizedOverlaySink::new(artifact, sink);
    let execution = execute_load_plan(plan, source, &mut overlay);
    let packed = overlay.stats;
    let records = std::mem::take(&mut overlay.records);
    drop(overlay);

    match execution {
        Ok(ExecutionOutcome { mut report, stats }) => {
            extend_report(&mut report, records);
            Ok(QuantizedExecutionOutcome {
                report,
                execution: stats,
                packed,
            })
        }
        Err(failure) => {
            let failure = *failure;
            let mut report = failure.report;
            extend_report(&mut report, records);
            Err(Box::new(QuantizedExecutionFailure {
                failed_target: failure.failed_target,
                cause: failure.cause,
                report,
                execution: failure.stats,
                packed,
            }))
        }
    }
}

/// Executes a packed overlay against the ordinary bounded local shard source.
pub fn execute_local_quantized_load_plan<K: TensorSink>(
    manifest: &CheckpointManifest,
    plan: &LoadPlan,
    artifact: &mut QuantizedBurnpackArtifact,
    sink: &mut K,
) -> QuantizedExecutionResult<K::Error> {
    let mut source = LocalShardSource::new(manifest);
    execute_quantized_load_plan(plan, &mut source, artifact, sink)
}

/// Validates a complete sidecar against post-transform plan targets.
///
/// Partial sidecars are permitted: every absent target simply remains in source
/// precision. Present entries, however, must be an exact subset of eligible
/// storage owners. This supports deliberate mixed-precision experiments while
/// preventing unknown, aliased, or policy-excluded tensors from entering a model.
pub fn validate_sidecar_contract(
    plan: &LoadPlan,
    artifact: &QuantizedBurnpackArtifact,
) -> Result<(), QuantizationError> {
    let manifest = artifact.manifest();
    manifest.packed_execution.require_available()?;
    if manifest.tensors.is_empty() {
        return Err(QuantizationError::SidecarContract {
            name: "<manifest>".into(),
            reason: "packed execution requires at least one model tensor".into(),
        });
    }

    for (name, metadata) in &manifest.tensors {
        let planned = plan
            .tensors
            .iter()
            .find(|tensor| tensor.target == *name)
            .ok_or_else(|| QuantizationError::SidecarContract {
                name: name.clone(),
                reason: "name is not a storage-owning target in the strict load plan".into(),
            })?;
        metadata.validate_burn_qfloat_layout()?;
        if metadata.shape != planned.shape {
            return Err(QuantizationError::SidecarContract {
                name: name.clone(),
                reason: format!(
                    "logical shape {:?} does not match planned post-transform shape {:?}",
                    metadata.shape, planned.shape
                ),
            });
        }
        let host_shape = metadata.host_shape()?;
        if !matches!(
            classify_weight_for_quantization(name, &host_shape),
            QuantizationEligibility::Eligible
        ) {
            return Err(QuantizationError::SidecarContract {
                name: name.clone(),
                reason: "tensor is excluded by the conservative weight-only policy".into(),
            });
        }
        let metadata_dtype = source_tensor_dtype(metadata.source_dtype);
        let matches_target = metadata_dtype == Some(planned.dtype);
        let matches_checkpoint_source = metadata_dtype.is_some_and(|dtype| {
            !planned.sources.is_empty()
                && planned.sources.iter().all(|source| source.dtype == dtype)
        });
        if !matches_target && !matches_checkpoint_source {
            return Err(QuantizationError::SidecarContract {
                name: name.clone(),
                reason: format!(
                    "quantized source dtype {:?} matches neither planned target dtype {} nor its checkpoint source dtypes",
                    metadata.source_dtype, planned.dtype,
                ),
            });
        }
    }
    Ok(())
}

/// Sink wrapper that chooses packed or original data at the final target boundary.
struct QuantizedOverlaySink<'a, K> {
    artifact: &'a mut QuantizedBurnpackArtifact,
    inner: &'a mut K,
    records: Vec<QuantizationRecord>,
    stats: PackedLoadStats,
}

impl<'a, K> QuantizedOverlaySink<'a, K> {
    /// Creates an empty overlay after its caller completes manifest preflight.
    fn new(artifact: &'a mut QuantizedBurnpackArtifact, inner: &'a mut K) -> Self {
        Self {
            artifact,
            inner,
            records: Vec::new(),
            stats: PackedLoadStats::default(),
        }
    }
}

impl<K: TensorSink> TensorSink for QuantizedOverlaySink<'_, K> {
    type Error = QuantizedSinkError<K::Error>;

    fn write_tensor(&mut self, target: &str, source_data: TensorData) -> Result<(), Self::Error> {
        let Some(metadata) = self.artifact.manifest().tensors.get(target).cloned() else {
            return self
                .inner
                .write_tensor(target, source_data)
                .map_err(QuantizedSinkError::Architecture);
        };
        let payload_bytes = metadata
            .packed_value_bytes
            .checked_add(metadata.scale_bytes)
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "packed load payload accounting",
            })?;
        let tensor_reads = self.stats.tensor_reads.checked_add(1).ok_or(
            QuantizationError::ByteLengthOverflow {
                context: "packed load tensor count",
            },
        )?;
        let total_payload = self.stats.payload_bytes.checked_add(payload_bytes).ok_or(
            QuantizationError::ByteLengthOverflow {
                context: "packed load payload sum",
            },
        )?;
        let packed = self
            .artifact
            .read_weight(target)
            .map_err(QuantizedSinkError::Quantization)?
            .into_burn_qfloat_data()
            .map_err(QuantizedSinkError::Quantization)?;
        self.stats.tensor_reads = tensor_reads;
        self.stats.payload_bytes = total_payload;
        self.stats.largest_payload_bytes = self.stats.largest_payload_bytes.max(payload_bytes);

        self.inner
            .write_tensor(target, packed)
            .map_err(QuantizedSinkError::Architecture)?;
        self.stats.tensors_applied = self.stats.tensors_applied.checked_add(1).ok_or(
            QuantizationError::ByteLengthOverflow {
                context: "packed applied tensor count",
            },
        )?;
        self.records.push(QuantizationRecord {
            target: target.into(),
            scheme: scheme_name(&metadata),
            bytes: payload_bytes,
        });
        Ok(())
    }

    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        self.inner
            .alias_tensor(target, source_target)
            .map_err(QuantizedSinkError::Architecture)
    }
}

/// Converts persisted source dtype metadata back to the strict plan vocabulary.
const fn source_tensor_dtype(dtype: SourceFloatDType) -> Option<TensorDType> {
    match dtype {
        SourceFloatDType::F16 => Some(TensorDType::F16),
        SourceFloatDType::Bf16 => Some(TensorDType::BF16),
        SourceFloatDType::F32 => Some(TensorDType::F32),
        SourceFloatDType::F64 => Some(TensorDType::F64),
        SourceFloatDType::Flex32 => None,
    }
}

/// Stable report spelling for one persisted tensor's actual configuration.
fn scheme_name(metadata: &QuantizedTensorMetadata) -> String {
    let precision = match metadata.scheme {
        SymmetricQuantization::Q8 => "q8",
        SymmetricQuantization::Q4 => "q4",
    };
    format!(
        "burn-qfloat-symmetric-{precision}-g{}-block-1xg",
        metadata.group_size.get()
    )
}

/// Clears progress categories for a failure that precedes pass-two execution.
fn unstarted_report(plan: &LoadPlan) -> LoadReport {
    let mut report = plan.report.clone();
    report.applied.clear();
    report.quantized.clear();
    report.finalize();
    report
}

/// Adds successfully installed packed targets without losing deterministic order.
fn extend_report(report: &mut LoadReport, records: Vec<QuantizationRecord>) {
    for record in records {
        report.record_quantized(record);
    }
    report.finalize();
}

/// Exposes the stable executable discriminator without requiring enum matching.
#[must_use]
pub const fn burn_qfloat_execution_support() -> PackedExecutionSupport {
    PackedExecutionSupport::BurnQFloatMatmul
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, fs::File, io::Write, path::Path};

    use burn::tensor::{DType, TensorData};
    use serde_json::{Map, Value, json};
    use tempfile::TempDir;
    use thiserror::Error;

    use super::*;
    use crate::checkpoint::quantization::{
        QuantizationGroupSize, QuantizedCheckpointOptions, WeightQuantizationConfig,
        quantize_local_checkpoint_to_burnpack,
    };
    use crate::checkpoint::{
        DTypePolicy, ExpectedTensor, LoadOptions, MappingPlan, build_load_plan, discover_checkpoint,
    };

    /// Minimal architecture sink that retains delivered metadata and payloads.
    #[derive(Default)]
    struct RecordingSink {
        tensors: BTreeMap<String, TensorData>,
    }

    /// Deterministic architecture rejection used to inspect error wrapping.
    #[derive(Debug, Error)]
    #[error("fixture architecture rejected `{0}`")]
    struct SinkError(String);

    impl TensorSink for RecordingSink {
        type Error = SinkError;

        fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
            self.tensors.insert(target.into(), data);
            Ok(())
        }

        fn alias_tensor(&mut self, target: &str, _source_target: &str) -> Result<(), Self::Error> {
            Err(SinkError(target.into()))
        }
    }

    /// Writes a standards-compliant unsharded F32 SafeTensors fixture.
    fn write_checkpoint(root: &Path, tensors: &[(&str, Vec<usize>)]) {
        let mut header = Map::new();
        let mut offset = 0_u64;
        for (name, shape) in tensors {
            let elements = shape.iter().product::<usize>();
            let bytes = u64::try_from(elements * size_of::<f32>()).unwrap();
            header.insert(
                (*name).into(),
                json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        for (_, shape) in tensors {
            let elements = shape.iter().product::<usize>();
            for index in 0..elements {
                file.write_all(&((index as f32 - 31.0) / 8.0).to_le_bytes())
                    .unwrap();
            }
        }
    }

    /// Builds a strict plan with exactly one eligible projection.
    fn fixture_plan(root: &Path, cast: Option<TensorDType>) -> (CheckpointManifest, LoadPlan) {
        let options = LoadOptions {
            dtype_policy: cast.map_or(DTypePolicy::Preserve, DTypePolicy::Cast),
            ..LoadOptions::default()
        };
        let manifest = discover_checkpoint(root, &options).unwrap();
        let expected = [
            ExpectedTensor::new("model.embed_tokens.weight", [2, 32]),
            ExpectedTensor::new("model.layers.0.input_layernorm.weight", [32]),
            ExpectedTensor::new("model.layers.0.self_attn.q_proj.weight", [32, 32]),
        ];
        let mappings = MappingPlan::identity(&expected);
        let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
        (manifest, plan)
    }

    /// Produces one executable Q8 sidecar from the fixture's F32 plan.
    fn executable_sidecar(
        directory: &TempDir,
    ) -> (CheckpointManifest, LoadPlan, std::path::PathBuf) {
        write_checkpoint(
            directory.path(),
            &[
                ("model.embed_tokens.weight", vec![2, 32]),
                ("model.layers.0.input_layernorm.weight", vec![32]),
                ("model.layers.0.self_attn.q_proj.weight", vec![32, 32]),
            ],
        );
        let (manifest, plan) = fixture_plan(directory.path(), None);
        let output = directory.path().join("weights-q8.bpk");
        quantize_local_checkpoint_to_burnpack(
            &manifest,
            &plan,
            &output,
            QuantizedCheckpointOptions {
                weights: WeightQuantizationConfig {
                    group_size: QuantizationGroupSize::new(32).unwrap(),
                    ..WeightQuantizationConfig::default()
                },
                ..QuantizedCheckpointOptions::default()
            },
        )
        .unwrap();
        (manifest, plan, output)
    }

    /// Packed projections replace source tensors while excluded weights pass through.
    #[test]
    fn local_overlay_installs_mixed_precision_and_reports_exact_io() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, plan, output) = executable_sidecar(&directory);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        assert_eq!(
            artifact.manifest().packed_execution,
            PackedExecutionSupport::BurnQFloatMatmul
        );
        let mut sink = RecordingSink::default();

        let outcome =
            execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

        assert_eq!(outcome.execution.source.tensor_reads, 3);
        assert_eq!(outcome.packed.tensor_reads, 1);
        assert_eq!(outcome.packed.tensors_applied, 1);
        assert_eq!(outcome.report.quantized.len(), 1);
        assert_eq!(
            outcome.report.quantized[0].target,
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert!(matches!(
            sink.tensors["model.layers.0.self_attn.q_proj.weight"].dtype,
            DType::QFloat(_)
        ));
        assert_eq!(sink.tensors["model.embed_tokens.weight"].dtype, DType::F32);
        assert_eq!(
            sink.tensors["model.layers.0.input_layernorm.weight"].dtype,
            DType::F32
        );
    }

    /// Packed weights remain reusable when excluded source tensors change dtype.
    #[test]
    fn preserved_sidecar_composes_with_a_cast_inference_plan() {
        let directory = tempfile::tempdir().unwrap();
        let (_, _, output) = executable_sidecar(&directory);
        let (manifest, cast_plan) = fixture_plan(directory.path(), Some(TensorDType::F16));
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let outcome =
            execute_local_quantized_load_plan(&manifest, &cast_plan, &mut artifact, &mut sink)
                .unwrap();

        assert_eq!(outcome.report.quantized.len(), 1);
        assert_eq!(sink.tensors["model.embed_tokens.weight"].dtype, DType::F16);
        assert!(matches!(
            sink.tensors["model.layers.0.self_attn.q_proj.weight"].dtype,
            DType::QFloat(_)
        ));
    }

    /// An unknown packed target fails before either source or sink observes a tensor.
    #[test]
    fn sidecar_target_mismatch_is_preflighted_before_source_io() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, plan, _) = executable_sidecar(&directory);
        let output = directory.path().join("unknown-target.bpk");
        let mut session =
            crate::checkpoint::quantization::QuantizedBurnpackSession::new(&output).unwrap();
        session
            .append(
                "model.layers.0.self_attn.k_proj.weight",
                &TensorData::new(vec![1.0_f32; 32 * 32], [32, 32]),
                WeightQuantizationConfig {
                    group_size: QuantizationGroupSize::new(32).unwrap(),
                    ..WeightQuantizationConfig::default()
                },
            )
            .unwrap();
        session.finalize().unwrap();
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let failure = execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();

        assert_eq!(failure.execution.source.tensor_reads, 0);
        assert!(failure.report.applied.is_empty());
        assert!(sink.tensors.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::SidecarContract { .. }
            ))
        ));
    }

    /// Reference-only flat groups can be stored but cannot enter packed execution.
    #[test]
    fn incompatible_artifact_is_marked_and_rejected_before_loading() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("incompatible.bpk");
        let source = TensorData::new(vec![1.0_f32; 64], [8, 8]);
        let mut session =
            crate::checkpoint::quantization::QuantizedBurnpackSession::new(&output).unwrap();
        session
            .append(
                "model.layers.0.self_attn.q_proj.weight",
                &source,
                WeightQuantizationConfig {
                    group_size: QuantizationGroupSize::new(32).unwrap(),
                    ..WeightQuantizationConfig::default()
                },
            )
            .unwrap();
        session.finalize().unwrap();
        let artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        assert_eq!(
            artifact.manifest().packed_execution,
            PackedExecutionSupport::Unavailable
        );
        assert!(matches!(
            validate_sidecar_contract(
                &LoadPlan {
                    tensors: Vec::new(),
                    aliases: Vec::new(),
                    report: LoadReport::default(),
                    memory: Default::default(),
                },
                &artifact,
            ),
            Err(QuantizationError::PackedExecutionUnavailable)
        ));
    }

    /// An architecture rejection remains distinguishable from sidecar corruption.
    #[test]
    fn packed_sink_failure_preserves_partial_read_accounting() {
        #[derive(Default)]
        struct RejectingSink;
        impl TensorSink for RejectingSink {
            type Error = SinkError;

            fn write_tensor(&mut self, target: &str, _data: TensorData) -> Result<(), Self::Error> {
                if target.ends_with("q_proj.weight") {
                    Err(SinkError(target.into()))
                } else {
                    Ok(())
                }
            }

            fn alias_tensor(
                &mut self,
                _target: &str,
                _source_target: &str,
            ) -> Result<(), Self::Error> {
                Ok(())
            }
        }

        let directory = tempfile::tempdir().unwrap();
        let (manifest, plan, output) = executable_sidecar(&directory);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RejectingSink;
        let failure = execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();

        assert_eq!(
            failure.failed_target.as_deref(),
            Some("model.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(failure.packed.tensor_reads, 1);
        assert_eq!(failure.packed.tensors_applied, 0);
        assert!(failure.report.quantized.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Architecture(_))
        ));
    }
}
