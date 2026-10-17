//! Load-plan-aware SafeTensors-to-Burnpack conversion.
//!
//! Architecture checkpoint contracts may rename, transpose, split, concatenate,
//! or alias external tensors. Quantizing raw SafeTensors names would silently
//! bypass those semantics. This module instead reuses the audited pass-two load
//! executor and replaces its model sink with [`BurnpackQuantizationSink`]. Each
//! final architecture-owned tensor is materialized exactly as it would be for a
//! full-precision model, quantized if policy permits, appended to disk staging,
//! and dropped before the executor advances.
//!
//! Peak host ownership is therefore bounded by one load-plan source/output plus
//! one packed target. Excluded embeddings, norms, biases, recurrent scalars, and
//! MTP weights are validated and reported but remain in the source checkpoint.
//! Aliases consume no duplicate payload; their topology remains in the strict
//! checkpoint report.
//!
//! Architecture-aware conversion emits provenance-bearing format v2. It hashes
//! every complete source shard before and after pass two and refuses publication
//! if the identities differ, then binds the stable source inventory to a canonical
//! SHA-256 of the exact load-plan transformations. Direct low-level Burnpack write
//! sessions remain available for backward-compatible v1 artifacts.

use std::{
    error::Error as StdError,
    fmt,
    path::{Path, PathBuf},
};

use burn::tensor::TensorData;

use super::{
    BurnpackAppendOutcome, BurnpackArtifactSummary, BurnpackProvenance, BurnpackSessionLimits,
    BurnpackStagingStats, QuantizationEligibility, QuantizationError, QuantizationExclusion,
    QuantizedBurnpackSession, QuantizedTensorMetadata, SourceFloatDType, WeightQuantizationConfig,
    classify_weight_for_quantization,
    provenance::{ProvenanceExecutionStats, hash_local_checkpoint_provenance_audited},
};
use crate::checkpoint::{
    CheckpointError, CheckpointManifest, ExecutionFailureCause, ExecutionStats, LoadPlan,
    LoadReport, LocalShardSource, QuantizationRecord, TensorDType, TensorSink, execute_load_plan,
};

/// Controls one strict, architecture-aware checkpoint conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QuantizedCheckpointOptions {
    /// Symmetric integer precision and contiguous group size.
    pub weights: WeightQuantizationConfig,
    /// Bounded disk, tensor-count, and metadata resources for the writer.
    pub limits: BurnpackSessionLimits,
    /// Treatment of eligible matrices whose input width is not group-aligned.
    pub unaligned: UnalignedMatrixPolicy,
}

/// Treatment of an eligible `[output, input]` matrix whose `input` width is not a
/// multiple of the group size `G`.
///
/// Packed Burn block-QFloat needs every group to stay within one row, which
/// holds exactly when `input % G == 0`. A single unaligned matrix marks the
/// whole sidecar [`super::PackedExecutionSupport::Unavailable`], and every packed
/// loader then rejects it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum UnalignedMatrixPolicy {
    /// Quantize it anyway; the artifact is valid reference-only storage.
    #[default]
    ReferenceOnly,
    /// Keep it in source precision, recording
    /// [`QuantizationExclusion::GroupUnaligned`], so the sidecar stays loadable.
    Exclude,
    /// Fail before reading any source payload, naming the first such target.
    Reject,
}

/// Returns whether an eligible matrix's groups stay within its rows.
///
/// Supported group sizes are multiples of eight, so `input % G == 0` also
/// aligns rows to whole packed `u32` words for both Q4 and Q8.
fn group_aligned_matrix(shape: &[usize], config: WeightQuantizationConfig) -> bool {
    matches!(shape, [_, input] if input.is_multiple_of(config.group_size.get()))
}

/// Predicts every packed entry and the provenance of a conversion, then checks limits.
///
/// Targets are classified with their post-transform shapes exactly as the sink
/// will classify them; metadata sizes depend only on shape and configuration.
/// Provenance digests are placeholders of the correct length.
fn preflight_planned_sidecar(
    session: &QuantizedBurnpackSession,
    manifest: &CheckpointManifest,
    plan: &LoadPlan,
    options: QuantizedCheckpointOptions,
) -> Result<(), QuantizationError> {
    let mut planned = Vec::new();
    for tensor in &plan.tensors {
        let Ok(shape) = tensor
            .shape
            .iter()
            .map(|&dimension| usize::try_from(dimension))
            .collect::<Result<Vec<_>, _>>()
        else {
            continue;
        };
        if classify_weight_for_quantization(&tensor.target, &shape)
            != QuantizationEligibility::Eligible
            || (options.unaligned == UnalignedMatrixPolicy::Exclude
                && !group_aligned_matrix(&shape, options.weights))
        {
            continue;
        }
        // Non-float targets fail at quantization time with their own error.
        let Some(source_dtype) = source_float_dtype(tensor.dtype) else {
            continue;
        };
        let metadata = QuantizedTensorMetadata::new(&shape, source_dtype, options.weights)?;
        planned.push((tensor.target.as_str(), metadata));
    }
    let provenance = BurnpackProvenance {
        load_plan_sha256: "0".repeat(64),
        checkpoint_shards: manifest
            .shards
            .iter()
            .map(|shard| {
                (
                    shard.relative_path.clone(),
                    super::BurnpackShardProvenance {
                        file_bytes: shard.file_bytes,
                        sha256: "0".repeat(64),
                    },
                )
            })
            .collect(),
    };
    session.preflight_inventory(
        planned.iter().map(|(name, metadata)| (*name, metadata)),
        &provenance,
    )
}

/// Maps a planned float dtype to the quantizer's source-dtype vocabulary.
fn source_float_dtype(dtype: TensorDType) -> Option<SourceFloatDType> {
    match dtype {
        TensorDType::F16 => Some(SourceFloatDType::F16),
        TensorDType::BF16 => Some(SourceFloatDType::Bf16),
        TensorDType::F32 => Some(SourceFloatDType::F32),
        TensorDType::F64 => Some(SourceFloatDType::F64),
        _ => None,
    }
}

/// Finds the first plan target that `Reject` refuses, before any source read.
fn first_unaligned_target(plan: &LoadPlan, config: WeightQuantizationConfig) -> Option<&str> {
    plan.tensors.iter().find_map(|tensor| {
        let shape = tensor
            .shape
            .iter()
            .map(|&dimension| usize::try_from(dimension).unwrap_or(usize::MAX))
            .collect::<Vec<_>>();
        let eligible = classify_weight_for_quantization(&tensor.target, &shape)
            == QuantizationEligibility::Eligible;
        (eligible && !group_aligned_matrix(&shape, config)).then_some(tensor.target.as_str())
    })
}

/// One load-plan target deliberately retained in source precision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizationExclusionRecord {
    /// Architecture-owned parameter name after checkpoint transformations.
    pub target: String,
    /// Conservative policy rule that kept the tensor unpacked.
    pub reason: QuantizationExclusion,
}

/// Successful result of streaming one checkpoint plan to an official Burnpack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointQuantizationOutcome {
    /// Atomically published artifact sizes and durability information.
    pub artifact: BurnpackArtifactSummary,
    /// Strict mapping report, augmented with every persisted quantized tensor.
    ///
    /// `applied` means the load-plan target was successfully consumed by the
    /// conversion sink. Consult `quantized` and [`Self::excluded`] to distinguish
    /// sidecar payloads from source-precision parameters.
    pub report: LoadReport,
    /// Architecture parameters intentionally absent from the weight-only sidecar.
    pub excluded: Vec<QuantizationExclusionRecord>,
    /// Source I/O and one-tensor transformation high-water counters.
    pub execution: ExecutionStats,
    /// Final writer staging counters before the private staging file was removed.
    pub staging: BurnpackStagingStats,
    /// Full-shard hashing performed before and after conversion execution.
    pub provenance: ProvenanceExecutionStats,
}

/// Root cause of a failed checkpoint conversion.
#[derive(Debug)]
#[non_exhaustive]
pub enum CheckpointQuantizationFailureCause {
    /// Strict plan validation, SafeTensors I/O, or transformation failed.
    Checkpoint(
        /// Original discovery, planning, or execution diagnostic.
        CheckpointError,
    ),
    /// Eligibility, numerical conversion, staging, or Burnpack publication failed.
    Quantization(
        /// Original quantizer or artifact-publication diagnostic.
        QuantizationError,
    ),
}

impl fmt::Display for CheckpointQuantizationFailureCause {
    /// Formats the original checkpoint or quantization failure without hiding its detail.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Quantization(error) => error.fmt(formatter),
        }
    }
}

impl StdError for CheckpointQuantizationFailureCause {
    /// Exposes the underlying failure for standard chained error reporting.
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Checkpoint(error) => Some(error),
            Self::Quantization(error) => Some(error),
        }
    }
}

/// Auditable failure with exact partial progress and no published output.
#[derive(Debug)]
pub struct CheckpointQuantizationFailure {
    /// Requested final artifact path.
    pub output_path: PathBuf,
    /// Target being transformed or staged, or `None` for setup/finalization.
    pub failed_target: Option<String>,
    /// Underlying checkpoint or quantization error.
    pub cause: CheckpointQuantizationFailureCause,
    /// Finalized strict report through the last successfully consumed target.
    pub report: LoadReport,
    /// Runtime source/transformation counters at failure.
    pub execution: ExecutionStats,
    /// Quantizer staging counters at failure.
    pub staging: BurnpackStagingStats,
    /// Targets intentionally excluded before the failure.
    pub excluded: Vec<QuantizationExclusionRecord>,
    /// Full-shard hashing attempted before and after conversion execution.
    pub provenance: ProvenanceExecutionStats,
}

/// Progress fields shared by every conversion failure boundary.
struct CheckpointQuantizationProgress {
    /// Strict load report through the last accepted target.
    report: LoadReport,
    /// Source reads and transformations completed before the failure.
    execution: ExecutionStats,
    /// Private staging activity completed before the failure.
    staging: BurnpackStagingStats,
    /// Targets skipped by the declared quantization policy.
    excluded: Vec<QuantizationExclusionRecord>,
    /// Source hashing attempted around conversion execution.
    provenance: ProvenanceExecutionStats,
}

impl fmt::Display for CheckpointQuantizationFailure {
    /// Formats the requested output and, when available, the target where conversion stopped.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(target) = &self.failed_target {
            write!(
                formatter,
                "checkpoint quantization failed at `{target}` for `{}`: {}",
                self.output_path.display(),
                self.cause
            )
        } else {
            write!(
                formatter,
                "checkpoint quantization failed for `{}`: {}",
                self.output_path.display(),
                self.cause
            )
        }
    }
}

impl StdError for CheckpointQuantizationFailure {
    /// Exposes the categorized conversion cause while keeping progress in the outer error.
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.cause)
    }
}

/// Result type retaining structured partial reports off the success-path stack.
pub type CheckpointQuantizationResult<T> =
    std::result::Result<T, Box<CheckpointQuantizationFailure>>;

/// Streams a validated architecture load plan into a quantized Burnpack sidecar.
///
/// The function is local-only and never contacts a model hub. The passed
/// [`CheckpointManifest`] remains the authority for containment and shard byte
/// ranges, while `plan` supplies the exact architecture mapping. Publication is
/// create-only and atomic; every error leaves `output_path` absent or preserves a
/// file that already existed before this call.
///
/// Provenance requires two complete sequential reads of every source shard. This
/// adds bounded-I/O work to offline conversion and prevents publication when an
/// ordinary persistent checkpoint modification spans execution. It is not a
/// hostile-filesystem authenticity guarantee: an adversarial ABA replacement
/// restored before the second observation remains outside the contract.
pub fn quantize_local_checkpoint_to_burnpack(
    manifest: &CheckpointManifest,
    plan: &LoadPlan,
    output_path: impl AsRef<Path>,
    options: QuantizedCheckpointOptions,
) -> CheckpointQuantizationResult<CheckpointQuantizationOutcome> {
    quantize_with_post_execution_hook(manifest, plan, output_path.as_ref(), options, || {})
}

/// Runs the conversion, calling `after_execution` once between its two hash passes.
///
/// The hook runs after every target has been staged and the source closed, and
/// before the post-execution provenance pass that guards publication; it is not
/// called when an earlier phase fails. The public entry point passes a no-op,
/// so the hook exists only to let tests change the checkpoint or claim the
/// output inside exactly the window the second hash pass and create-only
/// publication must defend.
fn quantize_with_post_execution_hook(
    manifest: &CheckpointManifest,
    plan: &LoadPlan,
    output_path: &Path,
    options: QuantizedCheckpointOptions,
    after_execution: impl FnOnce(),
) -> CheckpointQuantizationResult<CheckpointQuantizationOutcome> {
    let output_path = output_path.to_path_buf();
    let mut report = plan.report.without_progress();
    let mut provenance_stats = ProvenanceExecutionStats::default();
    if options.unaligned == UnalignedMatrixPolicy::Reject
        && let Some(target) = first_unaligned_target(plan, options.weights)
    {
        let cause = QuantizationError::BurnQFloatIncompatible {
            reason: format!(
                "`{target}` has an input width that is not a multiple of group size {}",
                options.weights.group_size.get()
            ),
        };
        return Err(conversion_failure(
            output_path,
            Some(target.to_owned()),
            CheckpointQuantizationFailureCause::Quantization(cause),
            CheckpointQuantizationProgress {
                report,
                execution: ExecutionStats::default(),
                staging: BurnpackStagingStats::default(),
                excluded: Vec::new(),
                provenance: provenance_stats,
            },
        ));
    }
    let mut session =
        QuantizedBurnpackSession::with_limits(&output_path, options.limits).map_err(|cause| {
            conversion_failure(
                output_path.clone(),
                None,
                CheckpointQuantizationFailureCause::Quantization(cause),
                CheckpointQuantizationProgress {
                    report: report.clone(),
                    execution: ExecutionStats::default(),
                    staging: BurnpackStagingStats::default(),
                    excluded: Vec::new(),
                    provenance: provenance_stats,
                },
            )
        })?;

    // Fail before any hashing or source read when the planned sidecar cannot fit
    // the session limits: count, per-tensor and total payload, and manifest.
    if let Err(cause) = preflight_planned_sidecar(&session, manifest, plan, options) {
        return Err(conversion_failure(
            output_path,
            None,
            CheckpointQuantizationFailureCause::Quantization(cause),
            CheckpointQuantizationProgress {
                report,
                execution: ExecutionStats::default(),
                staging: session.stats(),
                excluded: Vec::new(),
                provenance: provenance_stats,
            },
        ));
    }

    // Hashing on both sides of pass two prevents a source mutation from spanning
    // conversion unnoticed. Each hash worker's buffer is independent of model size.
    provenance_stats.pre_execution = Some(Default::default());
    let (initial_provenance, initial_hash_stats) =
        match hash_local_checkpoint_provenance_audited(manifest, plan) {
            Ok(result) => result,
            Err(failure) => {
                provenance_stats.pre_execution = Some(failure.stats);
                return Err(conversion_failure(
                    output_path,
                    None,
                    CheckpointQuantizationFailureCause::Quantization(failure.error),
                    CheckpointQuantizationProgress {
                        report,
                        execution: ExecutionStats::default(),
                        staging: BurnpackStagingStats::default(),
                        excluded: Vec::new(),
                        provenance: provenance_stats,
                    },
                ));
            }
        };
    provenance_stats.pre_execution = Some(initial_hash_stats);

    let mut source = LocalShardSource::new(manifest);
    let mut sink = BurnpackQuantizationSink::new(&mut session, options.weights, options.unaligned);
    let execution_result = execute_load_plan(plan, &mut source, &mut sink);
    let quantized = std::mem::take(&mut sink.quantized);
    let mut excluded = std::mem::take(&mut sink.excluded);
    drop(sink);
    let mut staging = session.stats();

    let execution = match execution_result {
        Ok(execution) => execution,
        Err(failure) => {
            let failure = *failure;
            report = failure.report;
            report.extend_quantized(quantized);
            normalize_exclusions(&mut excluded);
            let cause = match failure.cause {
                ExecutionFailureCause::Checkpoint(cause) => {
                    CheckpointQuantizationFailureCause::Checkpoint(cause)
                }
                ExecutionFailureCause::Sink(cause) => {
                    CheckpointQuantizationFailureCause::Quantization(cause)
                }
            };
            return Err(conversion_failure(
                output_path,
                failure.failed_target,
                cause,
                CheckpointQuantizationProgress {
                    report,
                    execution: failure.stats,
                    staging,
                    excluded,
                    provenance: provenance_stats,
                },
            ));
        }
    };

    report = execution.report;
    report.extend_quantized(quantized);
    normalize_exclusions(&mut excluded);
    after_execution();
    provenance_stats.post_execution = Some(Default::default());
    let (final_provenance, final_hash_stats) =
        match hash_local_checkpoint_provenance_audited(manifest, plan) {
            Ok(result) => result,
            Err(failure) => {
                provenance_stats.post_execution = Some(failure.stats);
                return Err(conversion_failure(
                    output_path,
                    None,
                    CheckpointQuantizationFailureCause::Quantization(failure.error),
                    CheckpointQuantizationProgress {
                        report,
                        execution: execution.stats,
                        staging,
                        excluded,
                        provenance: provenance_stats,
                    },
                ));
            }
        };
    provenance_stats.post_execution = Some(final_hash_stats);
    if initial_provenance != final_provenance {
        let shard = first_changed_shard(&initial_provenance, &final_provenance);
        return Err(conversion_failure(
            output_path,
            None,
            CheckpointQuantizationFailureCause::Quantization(
                QuantizationError::CheckpointChangedDuringQuantization { shard },
            ),
            CheckpointQuantizationProgress {
                report,
                execution: execution.stats,
                staging,
                excluded,
                provenance: provenance_stats,
            },
        ));
    }
    session.bind_provenance(final_provenance).map_err(|cause| {
        conversion_failure(
            output_path.clone(),
            None,
            CheckpointQuantizationFailureCause::Quantization(cause),
            CheckpointQuantizationProgress {
                report: report.clone(),
                execution: execution.stats.clone(),
                staging,
                excluded: excluded.clone(),
                provenance: provenance_stats,
            },
        )
    })?;
    staging = session.stats();
    let artifact = session.finalize().map_err(|cause| {
        conversion_failure(
            output_path.clone(),
            None,
            CheckpointQuantizationFailureCause::Quantization(cause),
            CheckpointQuantizationProgress {
                report: report.clone(),
                execution: execution.stats.clone(),
                staging,
                excluded: excluded.clone(),
                provenance: provenance_stats,
            },
        )
    })?;

    Ok(CheckpointQuantizationOutcome {
        artifact,
        report,
        excluded,
        execution: execution.stats,
        staging,
        provenance: provenance_stats,
    })
}

/// Identifies the first source shard changed across conversion's two hash passes.
fn first_changed_shard(before: &BurnpackProvenance, after: &BurnpackProvenance) -> String {
    before
        .checkpoint_shards
        .keys()
        .chain(after.checkpoint_shards.keys())
        .find(|name| before.checkpoint_shards.get(*name) != after.checkpoint_shards.get(*name))
        .cloned()
        .unwrap_or_else(|| "<load-plan>".into())
}

/// Tensor sink that applies quantization policy after load-plan transformations.
struct BurnpackQuantizationSink<'a> {
    /// Borrowed writer that owns unpublished staging bytes.
    session: &'a mut QuantizedBurnpackSession,
    /// Quantization scheme and group size applied to eligible targets.
    config: WeightQuantizationConfig,
    /// Whether group-unaligned eligible matrices are quantized or kept unpacked.
    unaligned: UnalignedMatrixPolicy,
    /// Successful staged target conversions, in execution order.
    quantized: Vec<QuantizationRecord>,
    /// Ineligible targets and their explicit exclusion reasons.
    excluded: Vec<QuantizationExclusionRecord>,
}

impl<'a> BurnpackQuantizationSink<'a> {
    /// Creates an empty reporting sink over one private write session.
    fn new(
        session: &'a mut QuantizedBurnpackSession,
        config: WeightQuantizationConfig,
        unaligned: UnalignedMatrixPolicy,
    ) -> Self {
        Self {
            session,
            config,
            unaligned,
            quantized: Vec::new(),
            excluded: Vec::new(),
        }
    }
}

impl TensorSink for BurnpackQuantizationSink<'_> {
    /// Quantizer or staging failure returned to the checkpoint executor.
    type Error = QuantizationError;

    /// Quantizes or explicitly excludes one final architecture parameter.
    fn write_tensor(
        &mut self,
        target: &str,
        data: TensorData,
    ) -> std::result::Result<(), Self::Error> {
        let shape = data.shape.as_slice();
        if self.unaligned == UnalignedMatrixPolicy::Exclude
            && classify_weight_for_quantization(target, shape) == QuantizationEligibility::Eligible
            && !group_aligned_matrix(shape, self.config)
        {
            self.excluded.push(QuantizationExclusionRecord {
                target: target.to_owned(),
                reason: QuantizationExclusion::GroupUnaligned,
            });
            return Ok(());
        }
        match self.session.append(target, &data, self.config)? {
            BurnpackAppendOutcome::Staged(staged) => {
                self.quantized.push(QuantizationRecord {
                    target: target.to_owned(),
                    scheme: self.config.storage_label(),
                    bytes: staged.payload_bytes,
                });
            }
            BurnpackAppendOutcome::Excluded(reason) => {
                self.excluded.push(QuantizationExclusionRecord {
                    target: target.to_owned(),
                    reason,
                });
            }
        }
        Ok(())
    }

    /// Records no duplicate payload for a tied architecture parameter.
    fn alias_tensor(
        &mut self,
        _target: &str,
        _source_target: &str,
    ) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

/// Sorts exclusion records and removes exact duplicates defensively.
fn normalize_exclusions(excluded: &mut Vec<QuantizationExclusionRecord>) {
    excluded.sort_by(|left, right| left.target.cmp(&right.target));
    excluded.dedup();
}

/// Boxes a rich failure consistently across setup, execution, and publication.
fn conversion_failure(
    output_path: PathBuf,
    failed_target: Option<String>,
    cause: CheckpointQuantizationFailureCause,
    progress: CheckpointQuantizationProgress,
) -> Box<CheckpointQuantizationFailure> {
    Box::new(CheckpointQuantizationFailure {
        output_path,
        failed_target,
        cause,
        report: progress.report,
        execution: progress.execution,
        staging: progress.staging,
        excluded: progress.excluded,
        provenance: progress.provenance,
    })
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
