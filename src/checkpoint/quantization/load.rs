//! Audited overlay of packed Burnpack weights onto strict load plans.
//!
//! A quantized Burnpack is a weight-only sidecar. Embeddings, normalization
//! scales, biases, recurrent parameters, and MTP tensors still come from the
//! original SafeTensors checkpoint. [`execute_quantized_load_plan`] therefore
//! runs the ordinary architecture plan with a pre-materialization override. A
//! matching, locally verified source may let a sidecar payload replace its
//! checkpoint source directly. Version-two local artifacts earn that privilege
//! only after every complete shard and the exact load plan match persisted
//! SHA-256 provenance. Legacy v1 artifacts read and validate the ordinary value
//! before replacement. An absent payload always follows the ordinary path.
//!
//! Preflight happens before the first source read. Every packed name must identify
//! a storage-owning plan target with the same post-transform shape and dtype, must
//! remain eligible under the conservative policy, and must map exactly to Burn's
//! `[1, group]` block geometry. This prevents a same-shaped but differently wired
//! artifact from silently bypassing the architecture contract. Runtime ownership
//! is normally one source tensor or one decoded packed target. A partial fused
//! split may hold its one shared source while installing a packed sibling, still
//! within the plan's conservative source-plus-output memory bound.
//!
//! Every packed payload in a stable artifact is verified and decoded sequentially
//! before the first model-sink call, so existing corruption in a later payload
//! cannot create a partial model write. [`PackedPreflightStats`] accounts for that
//! mandatory pass separately from the compatibility-stable [`PackedLoadStats`]
//! execution counters and records the decoder's owned-allocation high-water mark.
//! Burnpack decoding and conversion remain bounded to one tensor at a time; their
//! memory stays separate from generic checkpoint staging in
//! [`ExecutionStats::peak_host_staging_bytes`].

use std::{collections::BTreeMap, error::Error as StdError, fmt};

use burn::tensor::TensorData;

use super::{
    BURNPACK_QUANTIZATION_FORMAT_VERSION_V2, PackedExecutionSupport, QuantizationEligibility,
    QuantizationError, QuantizedBurnpackArtifact, QuantizedTensorMetadata, SourceFloatDType,
    SymmetricQuantization,
    burnpack::BurnpackValidationAudit,
    classify_weight_for_quantization,
    provenance::{
        ProvenanceExecutionStats, compare_checkpoint_provenance,
        hash_local_checkpoint_provenance_audited, preflight_local_checkpoint_provenance,
    },
    verify_load_plan_provenance,
};
use crate::checkpoint::{
    CheckpointManifest, ExecutionFailureCause, ExecutionOutcome, ExecutionStats, LoadPlan,
    LoadReport, LocalShardSource, QuantizationRecord, TensorDType, TensorSink, TensorSource,
    runtime::execute_load_plan_with_overrides,
};

/// Sidecar I/O counters kept separate from ordinary checkpoint source statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PackedLoadStats {
    /// Packed payloads decoded successfully, including a later sink rejection.
    pub tensor_reads: u64,
    /// Sum of packed-value and scale bytes decoded successfully.
    pub payload_bytes: u64,
    /// Largest individual packed-value-plus-scale payload decoded successfully.
    pub largest_payload_bytes: u64,
    /// Packed parameters accepted by the architecture sink.
    pub tensors_applied: usize,
}

/// Mandatory Burnpack validation work completed before model mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct PackedPreflightStats {
    /// Sequential validation passes entered; a load attempt starts at most one.
    pub passes_started: u64,
    /// Sequential validation passes that reached every sidecar tensor.
    pub passes_completed: u64,
    /// Payloads copied completely, including a failing payload.
    pub tensor_reads: u64,
    /// Packed-value and scale bytes copied, including a failing complete copy.
    pub payload_bytes: u64,
    /// Largest individual payload copied completely from mapped storage.
    pub largest_payload_bytes: u64,
    /// Maximum simultaneous Rust-owned bytes used to decode one payload.
    ///
    /// This reports actual `Vec` capacities. It excludes memory mappings,
    /// persistent manifests, generic checkpoint staging, backend allocations,
    /// and buffers retained by the model sink.
    pub peak_owned_payload_bytes: u64,
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
    /// Full-shard hashing performed around local v2 execution.
    pub provenance: ProvenanceExecutionStats,
}

/// Successful local packed execution plus preflight and format-version evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AuditedQuantizedExecutionOutcome {
    /// Exact Burnpack manifest format used by this load.
    pub format_version: u32,
    /// Mandatory packed-payload validation performed before model mutation.
    pub preflight: PackedPreflightStats,
    /// Legacy-compatible checkpoint, packed execution, and provenance outcome.
    pub execution: QuantizedExecutionOutcome,
}

impl AuditedQuantizedExecutionOutcome {
    /// Discards additive preflight/version evidence for a compatibility caller.
    #[must_use]
    pub fn into_execution(self) -> QuantizedExecutionOutcome {
        self.execution
    }
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
    /// Full-shard hashing attempted before and after local v2 execution.
    pub provenance: ProvenanceExecutionStats,
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

/// Failed local packed execution with its additive preflight/version evidence.
#[derive(Debug)]
#[non_exhaustive]
pub struct AuditedQuantizedExecutionFailure<E> {
    /// Exact Burnpack manifest format used by this load attempt.
    pub format_version: u32,
    /// Mandatory payload-validation progress before the failure.
    pub preflight: PackedPreflightStats,
    /// Legacy-compatible execution failure and its applied-prefix audit.
    pub execution: QuantizedExecutionFailure<E>,
}

impl<E> AuditedQuantizedExecutionFailure<E> {
    /// Discards additive preflight/version evidence for a compatibility caller.
    #[must_use]
    pub fn into_execution(self) -> QuantizedExecutionFailure<E> {
        self.execution
    }
}

impl<E: fmt::Display> fmt::Display for AuditedQuantizedExecutionFailure<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.execution.fmt(formatter)
    }
}

impl<E: StdError + 'static> StdError for AuditedQuantizedExecutionFailure<E> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.execution)
    }
}

/// Rich execution result retaining partial strict reports on failure.
pub type QuantizedExecutionResult<E> =
    std::result::Result<QuantizedExecutionOutcome, Box<QuantizedExecutionFailure<E>>>;

/// Local packed execution result retaining additive preflight/version evidence.
pub type AuditedQuantizedExecutionResult<E> =
    std::result::Result<AuditedQuantizedExecutionOutcome, Box<AuditedQuantizedExecutionFailure<E>>>;

/// Executes a strict plan while replacing matching targets with packed weights.
///
/// The source remains authoritative for checkpoint discovery, allowlists, aliases,
/// and all policy-excluded tensors. The sidecar is accepted only when its complete
/// manifest passes [`validate_sidecar_contract`] before `source.read_tensor` can be
/// called. This generic API supports legacy v1 artifacts only: its abstract
/// [`TensorSource`] cannot prove the complete checkpoint identity recorded by a
/// v2 sidecar. Use [`execute_local_quantized_load_plan`] for v2 local artifacts.
/// A generic source's bare positive descriptor boolean is also ignored because
/// it cannot mechanically bind that claim to an artifact. V1 execution remains
/// fail-closed to ordinary source validation. On a later failure, the boxed
/// [`QuantizedExecutionFailure`] contains both the exact applied prefix and the
/// underlying quantization or architecture error.
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
    if artifact.manifest().format_version == BURNPACK_QUANTIZATION_FORMAT_VERSION_V2 {
        return Err(source_preflight_failure(
            plan,
            source,
            QuantizationError::ArtifactBoundSourceRequired,
            ProvenanceExecutionStats::default(),
        ));
    }
    if let Err(error) = validate_sidecar_contract(plan, artifact) {
        return Err(source_preflight_failure(
            plan,
            source,
            error,
            ProvenanceExecutionStats::default(),
        ));
    }
    match artifact.validate_all_audited() {
        Ok(_) => {}
        Err(failure) => {
            return Err(source_preflight_failure(
                plan,
                source,
                failure.error,
                ProvenanceExecutionStats::default(),
            ));
        }
    }

    execute_prevalidated_quantized_load_plan(plan, source, artifact, sink, false)
}

/// Runs an already validated sidecar under an explicit certificate policy.
fn execute_prevalidated_quantized_load_plan<S, K>(
    plan: &LoadPlan,
    source: &mut S,
    artifact: &mut QuantizedBurnpackArtifact,
    sink: &mut K,
    allow_descriptor_bypass: bool,
) -> QuantizedExecutionResult<K::Error>
where
    S: TensorSource,
    K: TensorSink,
{
    let mut policy_source = DescriptorBypassPolicySource {
        inner: source,
        allow_descriptor_bypass,
    };

    let mut overlay = QuantizedOverlaySink::new(artifact, sink);
    let execution = execute_load_plan_with_overrides(
        plan,
        &mut policy_source,
        &mut overlay,
        |overlay, tensor| overlay.has_override(&tensor.target),
        |overlay, tensor| overlay.read_override(&tensor.target),
    );
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
                provenance: ProvenanceExecutionStats::default(),
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
                provenance: ProvenanceExecutionStats::default(),
            }))
        }
    }
}

/// Executes a local packed overlay through the compatibility outcome API.
///
/// This entry point performs the same mandatory validation and provenance gates
/// as [`execute_local_quantized_load_plan_audited`], then deliberately discards
/// only the additive preflight and format-version fields. Existing packed
/// execution and provenance counters retain their historical representation.
pub fn execute_local_quantized_load_plan<K: TensorSink>(
    manifest: &CheckpointManifest,
    plan: &LoadPlan,
    artifact: &mut QuantizedBurnpackArtifact,
    sink: &mut K,
) -> QuantizedExecutionResult<K::Error> {
    execute_local_quantized_load_plan_audited(manifest, plan, artifact, sink)
        .map(AuditedQuantizedExecutionOutcome::into_execution)
        .map_err(|failure| Box::new((*failure).into_execution()))
}

/// Executes a packed overlay and retains validation, format, and provenance audits.
///
/// V2 first streams every complete shard through SHA-256 before any sink mutation,
/// then permits matching packed descriptors to skip their full-precision buffers.
/// After successful execution it hashes every shard again and refuses to return
/// success if source content changed. The sink is not transactional: callers must
/// discard it when this post-execution gate fails. V1 retains the ordinary
/// one-tensor-at-a-time validation path and explicitly reports format version one
/// with no provenance hash passes.
///
/// Pre/post hashing detects persistent concurrent changes but does not establish
/// hostile-file authenticity. It assumes non-adversarial concurrency and cannot
/// detect an ABA replacement restored to the expected bytes before the
/// post-execution read. The mapped Burnpack must likewise remain stable throughout
/// execution; eliminating same-user concurrent artifact races requires an external
/// immutable snapshot or operating-system sealing.
pub fn execute_local_quantized_load_plan_audited<K: TensorSink>(
    manifest: &CheckpointManifest,
    plan: &LoadPlan,
    artifact: &mut QuantizedBurnpackArtifact,
    sink: &mut K,
) -> AuditedQuantizedExecutionResult<K::Error> {
    let format_version = artifact.manifest().format_version;
    if let Err(error) = validate_sidecar_contract(plan, artifact) {
        return Err(local_audited_preflight_failure(
            plan,
            error,
            format_version,
            PackedPreflightStats::default(),
            ProvenanceExecutionStats::default(),
        ));
    }

    let expected_provenance = artifact.manifest().provenance.clone();
    let mut provenance_stats = ProvenanceExecutionStats::default();
    if let Some(expected) = &expected_provenance
        && let Err(error) = preflight_local_checkpoint_provenance(manifest, expected)
    {
        return Err(local_audited_preflight_failure(
            plan,
            error,
            format_version,
            PackedPreflightStats::default(),
            provenance_stats,
        ));
    }

    // Validate every payload before the first model sink mutation. This is a
    // sequential, one-payload-at-a-time pass for both v1 semantic validation and
    // v2 content-digest verification.
    let preflight = match artifact.validate_all_audited() {
        Ok(audit) => packed_preflight_stats(audit, true),
        Err(failure) => {
            return Err(local_audited_preflight_failure(
                plan,
                failure.error,
                format_version,
                packed_preflight_stats(failure.audit, false),
                provenance_stats,
            ));
        }
    };

    let content_verified = if let Some(expected) = &expected_provenance {
        provenance_stats.pre_execution = Some(Default::default());
        let (actual, stats) = match hash_local_checkpoint_provenance_audited(manifest, plan) {
            Ok(result) => result,
            Err(failure) => {
                provenance_stats.pre_execution = Some(failure.stats);
                return Err(local_audited_preflight_failure(
                    plan,
                    failure.error,
                    format_version,
                    preflight,
                    provenance_stats,
                ));
            }
        };
        provenance_stats.pre_execution = Some(stats);
        if let Err(error) = compare_checkpoint_provenance(&actual, expected) {
            return Err(local_audited_preflight_failure(
                plan,
                error,
                format_version,
                preflight,
                provenance_stats,
            ));
        }
        true
    } else {
        false
    };
    let mut source = ProvenanceLocalSource {
        inner: LocalShardSource::new(manifest),
        content_verified,
    };
    let execution = execute_prevalidated_quantized_load_plan(
        plan,
        &mut source,
        artifact,
        sink,
        content_verified,
    );
    let mut outcome = match execution {
        Ok(outcome) => outcome,
        Err(failure) => {
            let mut failure = *failure;
            failure.provenance = provenance_stats;
            return Err(Box::new(AuditedQuantizedExecutionFailure {
                format_version,
                preflight,
                execution: failure,
            }));
        }
    };

    let Some(expected) = expected_provenance else {
        outcome.provenance = provenance_stats;
        return Ok(AuditedQuantizedExecutionOutcome {
            format_version,
            preflight,
            execution: outcome,
        });
    };

    if let Err(error) = verify_load_plan_provenance(plan, &expected)
        .and_then(|()| preflight_local_checkpoint_provenance(manifest, &expected))
    {
        return Err(post_audited_execution_failure(
            outcome,
            error,
            format_version,
            preflight,
            provenance_stats,
        ));
    }
    provenance_stats.post_execution = Some(Default::default());
    let (actual, stats) = match hash_local_checkpoint_provenance_audited(manifest, plan) {
        Ok(result) => result,
        Err(failure) => {
            provenance_stats.post_execution = Some(failure.stats);
            return Err(post_audited_execution_failure(
                outcome,
                failure.error,
                format_version,
                preflight,
                provenance_stats,
            ));
        }
    };
    provenance_stats.post_execution = Some(stats);
    if let Err(error) = compare_checkpoint_provenance(&actual, &expected) {
        return Err(post_audited_execution_failure(
            outcome,
            error,
            format_version,
            preflight,
            provenance_stats,
        ));
    }
    outcome.provenance = provenance_stats;
    Ok(AuditedQuantizedExecutionOutcome {
        format_version,
        preflight,
        execution: outcome,
    })
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
    if let Some(provenance) = &manifest.provenance {
        verify_load_plan_provenance(plan, provenance)?;
    }
    if manifest.tensors.is_empty() {
        return Err(QuantizationError::SidecarContract {
            name: "<manifest>".into(),
            reason: "packed execution requires at least one model tensor".into(),
        });
    }

    // Index once rather than scanning all model targets for every packed tensor.
    // Qwen3-MoE has roughly nineteen thousand entries on both sides, so the
    // previous nested search performed hundreds of millions of string compares.
    let mut planned_by_target = BTreeMap::new();
    for planned in &plan.tensors {
        if planned_by_target
            .insert(planned.target.as_str(), planned)
            .is_some()
        {
            return Err(QuantizationError::SidecarContract {
                name: planned.target.clone(),
                reason: "strict load plan contains a duplicate storage target".into(),
            });
        }
    }

    for (name, metadata) in &manifest.tensors {
        let planned = planned_by_target.get(name.as_str()).ok_or_else(|| {
            QuantizationError::SidecarContract {
                name: name.clone(),
                reason: "name is not a storage-owning target in the strict load plan".into(),
            }
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
        let matches_format_contract =
            if manifest.format_version == BURNPACK_QUANTIZATION_FORMAT_VERSION_V2 {
                matches_target
            } else {
                matches_target || matches_checkpoint_source
            };
        if !matches_format_contract {
            return Err(QuantizationError::SidecarContract {
                name: name.clone(),
                reason: if manifest.format_version == BURNPACK_QUANTIZATION_FORMAT_VERSION_V2 {
                    format!(
                        "v2 quantizer-input dtype {:?} does not match exact planned target dtype {}",
                        metadata.source_dtype, planned.dtype,
                    )
                } else {
                    format!(
                        "legacy quantizer-input dtype {:?} matches neither planned target dtype {} nor its checkpoint source dtypes",
                        metadata.source_dtype, planned.dtype,
                    )
                },
            });
        }
        let payload_bytes = metadata
            .packed_value_bytes
            .checked_add(metadata.scale_bytes)
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "packed preflight payload size",
            })?;
        if payload_bytes > plan.memory.peak_host_staging_bytes {
            return Err(QuantizationError::SidecarContract {
                name: name.clone(),
                reason: format!(
                    "packed payload requires {payload_bytes} bytes before backend transfer, exceeding the plan's {}-byte host-staging bound",
                    plan.memory.peak_host_staging_bytes
                ),
            });
        }
    }
    Ok(())
}

/// Local source whose positive descriptor result follows a complete SHA-256 gate.
struct ProvenanceLocalSource<'a> {
    inner: LocalShardSource<'a>,
    content_verified: bool,
}

/// Masks generic descriptor booleans unless the caller established artifact binding.
struct DescriptorBypassPolicySource<'a, S> {
    inner: &'a mut S,
    allow_descriptor_bypass: bool,
}

impl<S: TensorSource> TensorSource for DescriptorBypassPolicySource<'_, S> {
    fn validate_tensor_descriptor(
        &self,
        source: &crate::checkpoint::PlannedSource,
    ) -> crate::checkpoint::Result<bool> {
        let descriptor_matches = self.inner.validate_tensor_descriptor(source)?;
        Ok(self.allow_descriptor_bypass && descriptor_matches)
    }

    fn read_tensor(
        &mut self,
        source: &crate::checkpoint::PlannedSource,
    ) -> crate::checkpoint::Result<crate::checkpoint::SourceTensor> {
        self.inner.read_tensor(source)
    }

    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.inner.close()
    }

    fn stats(&self) -> crate::checkpoint::SourceStats {
        self.inner.stats()
    }
}

impl TensorSource for ProvenanceLocalSource<'_> {
    fn validate_tensor_descriptor(
        &self,
        source: &crate::checkpoint::PlannedSource,
    ) -> crate::checkpoint::Result<bool> {
        let _ = self.inner.validate_tensor_descriptor(source)?;
        Ok(self.content_verified)
    }

    fn read_tensor(
        &mut self,
        source: &crate::checkpoint::PlannedSource,
    ) -> crate::checkpoint::Result<crate::checkpoint::SourceTensor> {
        self.inner.read_tensor(source)
    }

    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.inner.close()
    }

    fn stats(&self) -> crate::checkpoint::SourceStats {
        self.inner.stats()
    }
}

/// Converts one mandatory validation attempt into public packed-load counters.
fn packed_preflight_stats(audit: BurnpackValidationAudit, completed: bool) -> PackedPreflightStats {
    PackedPreflightStats {
        passes_started: 1,
        passes_completed: u64::from(completed),
        tensor_reads: audit.tensor_reads,
        payload_bytes: audit.payload_bytes,
        largest_payload_bytes: audit.largest_payload_bytes,
        peak_owned_payload_bytes: audit.peak_owned_payload_bytes,
    }
}

/// Builds an audited pre-execution failure for local provenance or contract gates.
fn local_audited_preflight_failure<E>(
    plan: &LoadPlan,
    error: QuantizationError,
    format_version: u32,
    preflight: PackedPreflightStats,
    provenance: ProvenanceExecutionStats,
) -> Box<AuditedQuantizedExecutionFailure<E>> {
    Box::new(AuditedQuantizedExecutionFailure {
        format_version,
        preflight,
        execution: QuantizedExecutionFailure {
            failed_target: None,
            cause: ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(error)),
            report: unstarted_report(plan),
            execution: ExecutionStats::default(),
            packed: PackedLoadStats::default(),
            provenance,
        },
    })
}

/// Closes a generic source and retains lifecycle counters after sidecar preflight.
fn source_preflight_failure<S, E>(
    plan: &LoadPlan,
    source: &mut S,
    error: QuantizationError,
    provenance: ProvenanceExecutionStats,
) -> Box<QuantizedExecutionFailure<E>>
where
    S: TensorSource,
{
    let mut report = unstarted_report(plan);
    let mut execution = ExecutionStats::default();
    let cause = match source.close() {
        Ok(()) => ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(error)),
        Err(close_error) => ExecutionFailureCause::Checkpoint(close_error),
    };
    execution.source = source.stats();
    report.finalize();
    Box::new(QuantizedExecutionFailure {
        failed_target: None,
        cause,
        report,
        execution,
        packed: PackedLoadStats::default(),
        provenance,
    })
}

/// Converts completed execution into an audited fail-closed post-verification error.
fn post_audited_execution_failure<E>(
    outcome: QuantizedExecutionOutcome,
    error: QuantizationError,
    format_version: u32,
    preflight: PackedPreflightStats,
    provenance: ProvenanceExecutionStats,
) -> Box<AuditedQuantizedExecutionFailure<E>> {
    Box::new(AuditedQuantizedExecutionFailure {
        format_version,
        preflight,
        execution: QuantizedExecutionFailure {
            failed_target: None,
            cause: ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(error)),
            report: outcome.report,
            execution: outcome.execution,
            packed: outcome.packed,
            provenance,
        },
    })
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

    /// Returns whether the sidecar owns one post-transform storage target.
    fn has_override(&self, target: &str) -> bool {
        self.artifact.manifest().tensors.contains_key(target)
    }
}

impl<K: TensorSink> TensorSink for QuantizedOverlaySink<'_, K> {
    type Error = QuantizedSinkError<K::Error>;

    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
        let Some(metadata) = self.artifact.manifest().tensors.get(target).cloned() else {
            return self
                .inner
                .write_tensor(target, data)
                .map_err(QuantizedSinkError::Architecture);
        };
        let payload_bytes = metadata
            .packed_value_bytes
            .checked_add(metadata.scale_bytes)
            .ok_or(QuantizationError::ByteLengthOverflow {
                context: "packed load payload accounting",
            })?;
        let tensors_applied = self.stats.tensors_applied.checked_add(1).ok_or(
            QuantizationError::ByteLengthOverflow {
                context: "packed applied tensor count",
            },
        )?;
        let record = QuantizationRecord {
            target: target.into(),
            scheme: scheme_name(&metadata),
            bytes: payload_bytes,
        };

        self.inner
            .write_tensor(target, data)
            .map_err(QuantizedSinkError::Architecture)?;
        self.stats.tensors_applied = tensors_applied;
        self.records.push(record);
        Ok(())
    }

    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
        self.inner
            .alias_tensor(target, source_target)
            .map_err(QuantizedSinkError::Architecture)
    }
}

impl<K: TensorSink> QuantizedOverlaySink<'_, K> {
    /// Decodes one selected packed target and advances read-side counters only.
    fn read_override(
        &mut self,
        target: &str,
    ) -> Result<Option<TensorData>, QuantizedSinkError<K::Error>> {
        let Some(metadata) = self.artifact.manifest().tensors.get(target).cloned() else {
            return Ok(None);
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
        let weight = self
            .artifact
            .read_weight(target)
            .map_err(QuantizedSinkError::Quantization)?;
        self.stats.tensor_reads = tensor_reads;
        self.stats.payload_bytes = total_payload;
        self.stats.largest_payload_bytes = self.stats.largest_payload_bytes.max(payload_bytes);
        let packed = weight
            .into_burn_qfloat_data()
            .map_err(QuantizedSinkError::Quantization)?;
        Ok(Some(packed))
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
    report.equivalence_verified.clear();
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
    use std::{
        cell::Cell,
        collections::BTreeMap,
        fs::{self, File, OpenOptions},
        io::{Read, Seek, SeekFrom, Write},
        path::{Path, PathBuf},
        rc::Rc,
    };

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
        CheckpointEquivalenceRecord, CheckpointError, DTypePolicy, ExpectedTensor,
        IndexTensorAlias, LoadOptions, LocalShardSource, MappingPlan, PlannedSource, SourceStats,
        SourceTensor, build_load_plan, discover_checkpoint, execute_local_load_plan,
    };

    /// Minimal architecture sink that retains delivered metadata and payloads.
    #[derive(Default)]
    struct RecordingSink {
        tensors: BTreeMap<String, TensorData>,
        aliases: Vec<(String, String)>,
        operations: Vec<String>,
    }

    /// Deterministic architecture rejection used to inspect error wrapping.
    #[derive(Debug, Error)]
    #[error("fixture architecture rejected `{0}`")]
    struct SinkError(String);

    impl TensorSink for RecordingSink {
        type Error = SinkError;

        fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
            self.operations.push(format!("write:{target}"));
            self.tensors.insert(target.into(), data);
            Ok(())
        }

        fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
            self.operations.push(format!("alias:{target}"));
            self.aliases
                .push((target.to_owned(), source_target.to_owned()));
            Ok(())
        }
    }

    /// Sink that persistently changes one checkpoint byte after its first write.
    struct SameSizeMutatingSink {
        inner: RecordingSink,
        checkpoint: PathBuf,
        mutated: bool,
    }

    impl TensorSink for SameSizeMutatingSink {
        type Error = SinkError;

        fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error> {
            self.inner.write_tensor(target, data)?;
            if !self.mutated {
                let mut file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&self.checkpoint)
                    .map_err(|error| SinkError(error.to_string()))?;
                let length = file
                    .metadata()
                    .map_err(|error| SinkError(error.to_string()))?
                    .len();
                file.seek(SeekFrom::Start(length - 1))
                    .map_err(|error| SinkError(error.to_string()))?;
                let mut byte = [0_u8; 1];
                file.read_exact(&mut byte)
                    .map_err(|error| SinkError(error.to_string()))?;
                byte[0] ^= 1;
                file.seek(SeekFrom::Start(length - 1))
                    .map_err(|error| SinkError(error.to_string()))?;
                file.write_all(&byte)
                    .and_then(|()| file.sync_data())
                    .map_err(|error| SinkError(error.to_string()))?;
                self.mutated = true;
            }
            Ok(())
        }

        fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error> {
            self.inner.alias_tensor(target, source_target)
        }
    }

    /// Local source wrapper that deliberately cannot certify descriptors without I/O.
    struct NonCertifyingLocalSource<'a> {
        inner: LocalShardSource<'a>,
    }

    impl TensorSource for NonCertifyingLocalSource<'_> {
        fn read_tensor(
            &mut self,
            source: &PlannedSource,
        ) -> crate::checkpoint::Result<SourceTensor> {
            self.inner.read_tensor(source)
        }

        fn close(&mut self) -> crate::checkpoint::Result<()> {
            self.inner.close()
        }

        fn stats(&self) -> SourceStats {
            self.inner.stats()
        }
    }

    /// Local source that records closure while retaining fail-closed descriptor validation.
    struct CloseAwareLocalSource<'a> {
        inner: LocalShardSource<'a>,
        closed: Rc<Cell<bool>>,
        fail_close: bool,
    }

    impl TensorSource for CloseAwareLocalSource<'_> {
        fn validate_tensor_descriptor(
            &self,
            source: &PlannedSource,
        ) -> crate::checkpoint::Result<bool> {
            self.inner.validate_tensor_descriptor(source)
        }

        fn read_tensor(
            &mut self,
            source: &PlannedSource,
        ) -> crate::checkpoint::Result<SourceTensor> {
            self.inner.read_tensor(source)
        }

        fn close(&mut self) -> crate::checkpoint::Result<()> {
            self.closed.set(true);
            self.inner.close()?;
            if self.fail_close {
                Err(CheckpointError::InvalidMapping {
                    message: "synthetic close failure".into(),
                })
            } else {
                Ok(())
            }
        }

        fn stats(&self) -> SourceStats {
            self.inner.stats()
        }
    }

    /// Sink whose alias boundary proves that source closure precedes tied storage.
    struct CloseAwareSink {
        closed: Rc<Cell<bool>>,
        operations: Vec<String>,
    }

    impl TensorSink for CloseAwareSink {
        type Error = SinkError;

        fn write_tensor(&mut self, target: &str, _data: TensorData) -> Result<(), Self::Error> {
            self.operations.push(format!("write:{target}"));
            Ok(())
        }

        fn alias_tensor(&mut self, target: &str, _source_target: &str) -> Result<(), Self::Error> {
            if !self.closed.get() {
                return Err(SinkError(format!("source-open:{target}")));
            }
            self.operations.push(format!("alias:{target}"));
            Ok(())
        }
    }

    /// Source with a configurable bare descriptor boolean that rejects payload reads.
    #[derive(Default)]
    struct FailingSource {
        certify: bool,
        closed: bool,
        reads: usize,
    }

    impl TensorSource for FailingSource {
        fn validate_tensor_descriptor(
            &self,
            _source: &PlannedSource,
        ) -> crate::checkpoint::Result<bool> {
            Ok(self.certify)
        }

        fn read_tensor(
            &mut self,
            source: &PlannedSource,
        ) -> crate::checkpoint::Result<SourceTensor> {
            self.reads += 1;
            Err(CheckpointError::InvalidTensorPayload {
                tensor: source.name.clone(),
                reason: "synthetic read failure".into(),
            })
        }

        fn close(&mut self) -> crate::checkpoint::Result<()> {
            self.closed = true;
            Ok(())
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

    /// Writes an executable group-32 Q8 sidecar for explicit post-transform targets.
    fn write_sidecar(path: &Path, tensors: &[(&str, Vec<usize>)]) {
        let mut session =
            crate::checkpoint::quantization::QuantizedBurnpackSession::new(path).unwrap();
        for (name, shape) in tensors {
            let elements = shape.iter().product::<usize>();
            let values = (0..elements)
                .map(|index| (index as f32 - 17.0) / 11.0)
                .collect::<Vec<_>>();
            session
                .append(
                    name,
                    &TensorData::new(values, shape.clone()),
                    WeightQuantizationConfig {
                        group_size: QuantizationGroupSize::new(32).unwrap(),
                        ..WeightQuantizationConfig::default()
                    },
                )
                .unwrap();
        }
        session.finalize().unwrap();
    }

    /// Returns the unique location of one complete byte sequence in a fixture.
    fn unique_subslice(haystack: &[u8], needle: &[u8]) -> usize {
        let locations = haystack
            .windows(needle.len())
            .enumerate()
            .filter_map(|(index, candidate)| (candidate == needle).then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(locations.len(), 1, "fixture byte sequence must be unique");
        locations[0]
    }

    /// Checks one complete bounded-memory hash pass over a fixture manifest.
    fn assert_complete_hash_pass(
        stats: Option<crate::checkpoint::quantization::ProvenanceHashStats>,
        manifest: &CheckpointManifest,
    ) {
        let stats = stats.expect("v2 local execution must expose this hash pass");
        assert_eq!(stats.passes_completed, 1);
        assert_eq!(stats.shards_hashed, manifest.shards.len());
        assert_eq!(stats.bytes_hashed, manifest.artifact_bytes);
        assert_eq!(
            stats.peak_buffer_bytes,
            crate::checkpoint::quantization::PROVENANCE_HASH_BUFFER_BYTES
        );
    }

    /// Content-verified local overlays bypass packed sources after whole-shard hashing.
    #[test]
    fn local_overlay_installs_mixed_precision_and_reports_exact_io() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, plan, output) = executable_sidecar(&directory);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        assert_eq!(
            artifact.manifest().packed_execution,
            PackedExecutionSupport::BurnQFloatMatmul
        );
        assert!(artifact.manifest().provenance.is_some());
        let mut sink = RecordingSink::default();

        let audited =
            execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
                .unwrap();
        assert_eq!(
            audited.format_version,
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
        );
        assert_eq!(audited.preflight.passes_started, 1);
        assert_eq!(audited.preflight.passes_completed, 1);
        assert_eq!(audited.preflight.tensor_reads, 1);
        assert_eq!(audited.preflight.payload_bytes, 1_152);
        assert_eq!(audited.preflight.largest_payload_bytes, 1_152);
        assert!(audited.preflight.peak_owned_payload_bytes >= 1_280);
        let outcome = audited.execution;

        assert_eq!(outcome.execution.tensors_applied, 3);
        assert_eq!(outcome.execution.source_reads, 2);
        assert_eq!(outcome.execution.source_payload_bytes, 384);
        assert_eq!(outcome.execution.delivered_tensor_bytes, 1_536);
        assert_eq!(outcome.execution.largest_source_payload_bytes, 256);
        assert_eq!(outcome.execution.largest_output_tensor_bytes, 1_152);
        assert_eq!(outcome.execution.peak_host_staging_bytes, 1_152);
        assert_eq!(outcome.execution.source.tensor_reads, 2);
        assert_eq!(outcome.execution.source.payload_bytes_read, 384);
        assert_eq!(outcome.packed.tensor_reads, 1);
        assert_eq!(outcome.packed.payload_bytes, 1_152);
        assert_eq!(outcome.packed.largest_payload_bytes, 1_152);
        assert_eq!(outcome.packed.tensors_applied, 1);
        assert_complete_hash_pass(outcome.provenance.pre_execution, &manifest);
        assert_complete_hash_pass(outcome.provenance.post_execution, &manifest);
        assert_eq!(
            outcome.provenance.bytes_hashed(),
            u128::from(manifest.artifact_bytes) * 2
        );
        assert_eq!(
            outcome.provenance.peak_buffer_bytes(),
            crate::checkpoint::quantization::PROVENANCE_HASH_BUFFER_BYTES
        );
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

    /// Packed overlays cannot bypass physical checkpoint equivalence preflight.
    #[test]
    fn local_overlay_verifies_redundant_physical_payloads_before_installation() {
        let directory = tempfile::tempdir().unwrap();
        write_checkpoint(
            directory.path(),
            &[
                ("model.embed_tokens.weight", vec![2, 32]),
                ("decoder.embed_tokens.weight", vec![2, 32]),
                ("model.layers.0.input_layernorm.weight", vec![32]),
                ("model.layers.0.self_attn.q_proj.weight", vec![32, 32]),
            ],
        );
        let options = LoadOptions {
            index_aliases: vec![IndexTensorAlias::new(
                "decoder.embed_tokens.weight",
                "model.embed_tokens.weight",
            )],
            ..LoadOptions::default()
        };
        let manifest = discover_checkpoint(directory.path(), &options).unwrap();
        let expected = [
            ExpectedTensor::new("model.embed_tokens.weight", [2, 32]),
            ExpectedTensor::new("model.layers.0.input_layernorm.weight", [32]),
            ExpectedTensor::new("model.layers.0.self_attn.q_proj.weight", [32, 32]),
        ];
        let plan = build_load_plan(
            &manifest,
            &expected,
            &MappingPlan::identity(&expected),
            &options,
        )
        .unwrap();
        let record = CheckpointEquivalenceRecord::new(
            "decoder.embed_tokens.weight",
            "model.embed_tokens.weight",
        );
        assert_eq!(
            plan.report.equivalence_planned,
            std::slice::from_ref(&record)
        );

        let output = directory.path().join("weights-equivalence-q8.bpk");
        let conversion = quantize_local_checkpoint_to_burnpack(
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
        assert_eq!(
            conversion.report.equivalence_verified,
            std::slice::from_ref(&record)
        );
        assert_eq!(conversion.execution.equivalences_verified, 1);

        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();
        let outcome =
            execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

        assert_eq!(outcome.report.equivalence_verified, [record]);
        assert_eq!(outcome.execution.equivalences_verified, 1);
        assert_eq!(outcome.execution.equivalence_source_reads, 2);
        assert_eq!(outcome.execution.equivalence_source_payload_bytes, 512);
        assert_eq!(outcome.execution.source_reads, 4);
        assert_eq!(outcome.execution.source_payload_bytes, 896);
        assert_eq!(outcome.execution.largest_equivalence_pair_bytes, 512);
        assert_eq!(sink.operations.len(), 3);
    }

    /// A sidecar cannot conceal a local source removed after header discovery.
    #[test]
    fn local_overlay_fails_closed_when_an_overridden_source_disappears() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, plan, output) = executable_sidecar(&directory);
        fs::remove_file(directory.path().join("model.safetensors")).unwrap();
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let audited =
            execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
                .unwrap_err();
        assert_eq!(audited.preflight, PackedPreflightStats::default());
        let failure = &audited.execution;

        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.execution.source.shard_opens, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert!(failure.report.applied.is_empty());
        assert!(sink.operations.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::ArtifactIo { .. }
            ))
        ));
    }

    /// Same-size source mutation is detected by its content digest before any model write.
    #[test]
    fn local_overlay_rejects_checkpoint_payload_digest_mismatch() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, plan, output) = executable_sidecar(&directory);
        let checkpoint = directory.path().join("model.safetensors");
        let mut bytes = fs::read(&checkpoint).unwrap();
        let last = bytes.last_mut().unwrap();
        *last ^= 1;
        fs::write(&checkpoint, bytes).unwrap();
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let audited =
            execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
                .unwrap_err();
        assert_eq!(audited.preflight.passes_started, 1);
        assert_eq!(audited.preflight.passes_completed, 1);
        assert_eq!(audited.preflight.tensor_reads, 1);
        assert_eq!(audited.preflight.payload_bytes, 1_152);
        assert!(audited.preflight.peak_owned_payload_bytes >= 1_280);
        let failure = &audited.execution;

        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.execution.source.shard_opens, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert!(failure.report.applied.is_empty());
        assert!(sink.operations.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::CheckpointProvenanceDigestMismatch { .. }
            ))
        ));
    }

    /// Shard-count and declared-length mismatches fail before allocating a hash buffer.
    #[test]
    fn local_overlay_preflights_shard_inventory_before_hashing() {
        for remove_shard in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let (mut manifest, plan, output) = executable_sidecar(&directory);
            if remove_shard {
                manifest.shards.clear();
            } else {
                manifest.shards[0].file_bytes += 1;
            }
            let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
            let mut sink = RecordingSink::default();

            let failure =
                execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink)
                    .unwrap_err();

            assert_eq!(failure.provenance.pre_execution, None);
            assert_eq!(failure.provenance.post_execution, None);
            assert_eq!(failure.execution.source_reads, 0);
            assert_eq!(failure.packed.tensor_reads, 0);
            assert!(sink.operations.is_empty());
            assert!(matches!(
                failure.cause,
                ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                    QuantizationError::InvalidProvenance { .. }
                        | QuantizationError::CheckpointProvenanceLengthMismatch { .. }
                ))
            ));
        }
    }

    /// Persistent same-size mutation during execution is rejected after full progress.
    #[test]
    fn local_overlay_posthash_rejects_sink_driven_source_mutation() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, plan, output) = executable_sidecar(&directory);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = SameSizeMutatingSink {
            inner: RecordingSink::default(),
            checkpoint: directory.path().join("model.safetensors"),
            mutated: false,
        };

        let audited =
            execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
                .unwrap_err();
        assert_eq!(audited.preflight.passes_started, 1);
        assert_eq!(audited.preflight.passes_completed, 1);
        assert_eq!(audited.preflight.tensor_reads, 1);
        assert_eq!(audited.preflight.payload_bytes, 1_152);
        assert!(audited.preflight.peak_owned_payload_bytes >= 1_280);
        let failure = &audited.execution;

        assert!(sink.mutated);
        assert_eq!(failure.failed_target, None);
        assert_eq!(failure.execution.tensors_applied, 3);
        assert_eq!(failure.execution.source_reads, 2);
        assert_eq!(failure.execution.source_payload_bytes, 384);
        assert_eq!(failure.packed.tensor_reads, 1);
        assert_eq!(failure.packed.payload_bytes, 1_152);
        assert_eq!(failure.packed.tensors_applied, 1);
        assert_eq!(failure.report.applied.len(), 3);
        assert_eq!(failure.report.quantized.len(), 1);
        assert_eq!(sink.inner.operations.len(), 3);
        assert_complete_hash_pass(failure.provenance.pre_execution, &manifest);
        assert_complete_hash_pass(failure.provenance.post_execution, &manifest);
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::CheckpointProvenanceDigestMismatch { .. }
            ))
        ));
    }

    /// Packed size is rejected from metadata before source or sidecar payload I/O.
    #[test]
    fn packed_payload_exceeding_the_plan_bound_fails_in_preflight() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, mut plan, output) = executable_sidecar(&directory);
        plan.memory.peak_host_staging_bytes = 1_151;
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let audited =
            execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
                .unwrap_err();
        assert_eq!(audited.preflight, PackedPreflightStats::default());
        let failure = &audited.execution;

        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert!(failure.report.applied.is_empty());
        assert!(sink.operations.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::SidecarContract { .. }
            ))
        ));
    }

    /// The deterministic preflight index rejects duplicate mutable plan targets.
    #[test]
    fn sidecar_preflight_rejects_duplicate_plan_targets() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, mut plan, _) = executable_sidecar(&directory);
        let output = directory.path().join("duplicate-plan-v1.bpk");
        write_sidecar(
            &output,
            &[("model.layers.0.self_attn.q_proj.weight", vec![32, 32])],
        );
        let duplicate = plan
            .tensors
            .iter()
            .find(|tensor| tensor.target.ends_with("q_proj.weight"))
            .unwrap()
            .clone();
        plan.tensors.push(duplicate);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let audited =
            execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
                .unwrap_err();
        assert_eq!(audited.format_version, 1);
        assert_eq!(audited.preflight, PackedPreflightStats::default());
        let failure = &audited.execution;

        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert!(failure.report.applied.is_empty());
        assert!(sink.operations.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::SidecarContract { .. }
            ))
        ));
    }

    /// A source without artifact-bound evidence retains strict source reads.
    #[test]
    fn non_certifying_source_falls_back_to_full_materialization() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, plan, converted_output) = executable_sidecar(&directory);
        fs::remove_file(converted_output).unwrap();
        let output = directory.path().join("legacy-generic-fallback.bpk");
        write_sidecar(
            &output,
            &[("model.layers.0.self_attn.q_proj.weight", vec![32, 32])],
        );
        let mut source = NonCertifyingLocalSource {
            inner: LocalShardSource::new(&manifest),
        };
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let outcome =
            execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap();

        assert_eq!(outcome.execution.tensors_applied, 3);
        assert_eq!(outcome.execution.source_reads, 3);
        assert_eq!(outcome.execution.source_payload_bytes, 4_480);
        assert_eq!(outcome.execution.source.tensor_reads, 3);
        assert_eq!(outcome.execution.source.payload_bytes_read, 4_480);
        assert_eq!(outcome.execution.delivered_tensor_bytes, 1_536);
        assert_eq!(outcome.execution.largest_source_payload_bytes, 4_096);
        assert_eq!(outcome.execution.largest_output_tensor_bytes, 1_152);
        assert_eq!(outcome.execution.peak_host_staging_bytes, 4_096);
        assert_eq!(outcome.packed.tensor_reads, 1);
        assert_eq!(outcome.packed.payload_bytes, 1_152);
        assert_eq!(outcome.packed.tensors_applied, 1);
        assert!(matches!(
            sink.tensors["model.layers.0.self_attn.q_proj.weight"].dtype,
            DType::QFloat(_)
        ));
    }

    /// Strict fallback must validate a failing source before decoding its override.
    #[test]
    fn non_certifying_override_source_failure_precedes_packed_read() {
        let directory = tempfile::tempdir().unwrap();
        let target = "model.layers.0.self_attn.q_proj.weight";
        write_checkpoint(directory.path(), &[(target, vec![32, 32])]);
        let options = LoadOptions::default();
        let manifest = discover_checkpoint(directory.path(), &options).unwrap();
        let expected = [ExpectedTensor::new(target, [32, 32])];
        let plan = build_load_plan(
            &manifest,
            &expected,
            &MappingPlan::identity(&expected),
            &options,
        )
        .unwrap();
        let output = directory.path().join("strict-fallback.bpk");
        write_sidecar(&output, &[(target, vec![32, 32])]);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut source = FailingSource::default();
        let mut sink = RecordingSink::default();

        let failure =
            execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();

        assert_eq!(failure.failed_target.as_deref(), Some(target));
        assert_eq!(source.reads, 1);
        assert!(source.closed);
        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.execution.tensors_applied, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert_eq!(failure.packed.payload_bytes, 0);
        assert_eq!(failure.packed.tensors_applied, 0);
        assert!(failure.report.applied.is_empty());
        assert!(failure.report.quantized.is_empty());
        assert!(sink.operations.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Checkpoint(CheckpointError::InvalidTensorPayload { .. })
        ));
    }

    /// Generic sources reject v2 before a bare descriptor claim or payload read.
    #[test]
    fn generic_source_certificate_is_never_treated_as_v2_artifact_binding() {
        let directory = tempfile::tempdir().unwrap();
        let target = "model.layers.0.self_attn.q_proj.weight";
        write_checkpoint(directory.path(), &[(target, vec![32, 32])]);
        let options = LoadOptions::default();
        let manifest = discover_checkpoint(directory.path(), &options).unwrap();
        let expected = [ExpectedTensor::new(target, [32, 32])];
        let plan = build_load_plan(
            &manifest,
            &expected,
            &MappingPlan::identity(&expected),
            &options,
        )
        .unwrap();
        let output = directory.path().join("generic-v2.bpk");
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
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        assert!(artifact.manifest().provenance.is_some());
        let mut source = FailingSource {
            certify: true,
            ..FailingSource::default()
        };
        let mut sink = RecordingSink::default();

        let failure =
            execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();

        assert_eq!(failure.failed_target, None);
        assert_eq!(source.reads, 0);
        assert!(source.closed);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert_eq!(failure.provenance, Default::default());
        assert!(sink.operations.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::ArtifactBoundSourceRequired
            ))
        ));
    }

    /// A same-layout checkpoint with different values cannot mix with a v2 sidecar.
    #[test]
    fn generic_source_rejects_same_plan_different_checkpoint_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, plan, output) = executable_sidecar(&directory);
        let checkpoint = directory.path().join("model.safetensors");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(checkpoint)
            .unwrap();
        let final_offset = file.metadata().unwrap().len() - 1;
        file.seek(SeekFrom::Start(final_offset)).unwrap();
        let mut byte = [0_u8; 1];
        file.read_exact(&mut byte).unwrap();
        byte[0] ^= 1;
        file.seek(SeekFrom::Start(final_offset)).unwrap();
        file.write_all(&byte).unwrap();
        file.sync_data().unwrap();

        let mut source = LocalShardSource::new(&manifest);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();
        let failure =
            execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();

        assert_eq!(failure.failed_target, None);
        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.execution.source.shard_opens, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert_eq!(failure.provenance, Default::default());
        assert!(failure.report.applied.is_empty());
        assert!(sink.operations.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::ArtifactBoundSourceRequired
            ))
        ));
    }

    /// Legacy packed weights remain reusable when excluded source tensors change dtype.
    #[test]
    fn preserved_sidecar_composes_with_a_cast_inference_plan() {
        let directory = tempfile::tempdir().unwrap();
        let (_, _, converted_output) = executable_sidecar(&directory);
        fs::remove_file(converted_output).unwrap();
        let output = directory.path().join("legacy-cast-compatible.bpk");
        write_sidecar(
            &output,
            &[("model.layers.0.self_attn.q_proj.weight", vec![32, 32])],
        );
        let (manifest, cast_plan) = fixture_plan(directory.path(), Some(TensorDType::F16));
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let outcome =
            execute_local_quantized_load_plan(&manifest, &cast_plan, &mut artifact, &mut sink)
                .unwrap();

        assert_eq!(outcome.execution.tensors_applied, 3);
        assert_eq!(outcome.execution.source_reads, 3);
        assert_eq!(outcome.execution.source_payload_bytes, 4_480);
        assert_eq!(outcome.execution.delivered_tensor_bytes, 1_344);
        assert_eq!(outcome.execution.largest_source_payload_bytes, 4_096);
        assert_eq!(outcome.execution.largest_output_tensor_bytes, 2_048);
        assert_eq!(outcome.execution.peak_host_staging_bytes, 6_144);
        assert_eq!(outcome.report.quantized.len(), 1);
        assert_eq!(sink.tensors["model.embed_tokens.weight"].dtype, DType::F16);
        assert!(matches!(
            sink.tensors["model.layers.0.self_attn.q_proj.weight"].dtype,
            DType::QFloat(_)
        ));
    }

    /// V2 cannot use legacy source-dtype compatibility under a matching plan hash.
    #[test]
    fn v2_rejects_source_dtype_metadata_that_differs_from_the_planned_target() {
        let directory = tempfile::tempdir().unwrap();
        let target = "model.layers.0.self_attn.q_proj.weight";
        write_checkpoint(
            directory.path(),
            &[
                ("model.embed_tokens.weight", vec![2, 32]),
                ("model.layers.0.input_layernorm.weight", vec![32]),
                (target, vec![32, 32]),
            ],
        );
        let (manifest, cast_plan) = fixture_plan(directory.path(), Some(TensorDType::F16));
        let output = directory.path().join("crafted-v2-source-dtype.bpk");
        let mut session =
            crate::checkpoint::quantization::QuantizedBurnpackSession::new(&output).unwrap();
        session
            .append(
                target,
                &TensorData::new(vec![1.0_f32; 32 * 32], [32, 32]),
                WeightQuantizationConfig {
                    group_size: QuantizationGroupSize::new(32).unwrap(),
                    ..WeightQuantizationConfig::default()
                },
            )
            .unwrap();
        let (provenance, _) = crate::checkpoint::quantization::hash_local_checkpoint_provenance(
            &manifest, &cast_plan,
        )
        .unwrap();
        session.bind_provenance(provenance).unwrap();
        session.finalize().unwrap();

        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        assert_eq!(
            artifact.manifest().format_version,
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
        );
        assert_eq!(
            artifact.manifest().tensors[target].source_dtype,
            SourceFloatDType::F32
        );
        let mut sink = RecordingSink::default();

        let failure =
            execute_local_quantized_load_plan(&manifest, &cast_plan, &mut artifact, &mut sink)
                .unwrap_err();

        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert_eq!(failure.provenance, Default::default());
        assert!(failure.report.applied.is_empty());
        assert!(sink.operations.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::SidecarContract { .. }
            ))
        ));
    }

    /// V2 rejects even shape-compatible plans whose destination dtype changed.
    #[test]
    fn provenance_rejects_a_different_cast_plan_before_checkpoint_io() {
        let directory = tempfile::tempdir().unwrap();
        let (_, _, output) = executable_sidecar(&directory);
        let (manifest, cast_plan) = fixture_plan(directory.path(), Some(TensorDType::F16));
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let failure =
            execute_local_quantized_load_plan(&manifest, &cast_plan, &mut artifact, &mut sink)
                .unwrap_err();

        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert!(failure.report.applied.is_empty());
        assert!(sink.operations.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::LoadPlanProvenanceMismatch { .. }
            ))
        ));
    }

    /// A deliberately partial sidecar still validates all local source tensors.
    #[test]
    fn partial_sidecar_revalidates_every_local_source() {
        let directory = tempfile::tempdir().unwrap();
        write_checkpoint(
            directory.path(),
            &[
                ("model.embed_tokens.weight", vec![2, 32]),
                ("model.layers.0.input_layernorm.weight", vec![32]),
                ("model.layers.0.self_attn.q_proj.weight", vec![32, 32]),
                ("model.layers.0.self_attn.k_proj.weight", vec![32, 32]),
            ],
        );
        let options = LoadOptions::default();
        let manifest = discover_checkpoint(directory.path(), &options).unwrap();
        let expected = [
            ExpectedTensor::new("model.embed_tokens.weight", [2, 32]),
            ExpectedTensor::new("model.layers.0.input_layernorm.weight", [32]),
            ExpectedTensor::new("model.layers.0.self_attn.q_proj.weight", [32, 32]),
            ExpectedTensor::new("model.layers.0.self_attn.k_proj.weight", [32, 32]),
        ];
        let plan = build_load_plan(
            &manifest,
            &expected,
            &MappingPlan::identity(&expected),
            &options,
        )
        .unwrap();
        let output = directory.path().join("partial.bpk");
        write_sidecar(
            &output,
            &[("model.layers.0.self_attn.q_proj.weight", vec![32, 32])],
        );
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let outcome =
            execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

        assert_eq!(outcome.execution.tensors_applied, 4);
        assert_eq!(outcome.execution.source_reads, 4);
        assert_eq!(outcome.execution.source_payload_bytes, 8_576);
        assert_eq!(outcome.execution.source.tensor_reads, 4);
        assert_eq!(outcome.execution.source.payload_bytes_read, 8_576);
        assert_eq!(outcome.execution.delivered_tensor_bytes, 5_632);
        assert_eq!(outcome.execution.largest_source_payload_bytes, 4_096);
        assert_eq!(outcome.execution.largest_output_tensor_bytes, 4_096);
        assert_eq!(outcome.execution.peak_host_staging_bytes, 4_096);
        assert_eq!(outcome.packed.tensor_reads, 1);
        assert_eq!(outcome.packed.payload_bytes, 1_152);
        assert_eq!(outcome.packed.tensors_applied, 1);
        assert!(matches!(
            sink.tensors["model.layers.0.self_attn.q_proj.weight"].dtype,
            DType::QFloat(_)
        ));
        assert_eq!(
            sink.tensors["model.layers.0.self_attn.k_proj.weight"].dtype,
            DType::F32
        );
    }

    /// A partial fused split reads its shared source once and preserves alias order.
    #[test]
    fn partial_fused_split_reuses_one_source_around_a_packed_sibling() {
        let directory = tempfile::tempdir().unwrap();
        write_checkpoint(directory.path(), &[("fused.weight", vec![64, 32])]);
        let options = LoadOptions::default();
        let manifest = discover_checkpoint(directory.path(), &options).unwrap();
        let q = "model.layers.0.self_attn.q_proj.weight";
        let k = "model.layers.0.self_attn.k_proj.weight";
        let alias = "lm_head.weight";
        let expected = [
            ExpectedTensor::new(q, [32, 32]),
            ExpectedTensor::new(k, [32, 32]),
            ExpectedTensor::new(alias, [32, 32]),
        ];
        let mut mappings = MappingPlan::new();
        mappings
            .split("fused.weight", 0, [(q.to_owned(), 32), (k.to_owned(), 32)])
            .alias(q, alias);
        let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
        let output = directory.path().join("partial-fused.bpk");
        write_sidecar(&output, &[(q, vec![32, 32])]);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let outcome =
            execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

        assert_eq!(outcome.execution.tensors_applied, 2);
        assert_eq!(outcome.execution.aliases_applied, 1);
        assert_eq!(outcome.execution.source_reads, 1);
        assert_eq!(outcome.execution.source_payload_bytes, 8_192);
        assert_eq!(outcome.execution.source.tensor_reads, 1);
        assert_eq!(outcome.execution.delivered_tensor_bytes, 5_248);
        assert_eq!(outcome.execution.largest_source_payload_bytes, 8_192);
        assert_eq!(outcome.execution.largest_output_tensor_bytes, 4_096);
        assert_eq!(outcome.execution.peak_host_staging_bytes, 12_288);
        assert_eq!(outcome.packed.tensor_reads, 1);
        assert_eq!(outcome.packed.payload_bytes, 1_152);
        assert_eq!(outcome.packed.tensors_applied, 1);
        assert_eq!(outcome.report.applied.len(), 3);
        assert_eq!(sink.aliases, [(alias.to_owned(), q.to_owned())]);
        assert_eq!(
            sink.operations,
            [
                format!("write:{k}"),
                format!("write:{q}"),
                format!("alias:{alias}"),
            ]
        );
        assert!(matches!(sink.tensors[q].dtype, DType::QFloat(_)));
        assert_eq!(sink.tensors[k].dtype, DType::F32);
    }

    /// Packed output contributes to the high-water mark while a fused source is live.
    #[test]
    fn asymmetric_partial_fused_split_accounts_for_concurrent_packed_output() {
        let directory = tempfile::tempdir().unwrap();
        write_checkpoint(directory.path(), &[("fused.weight", vec![129, 32])]);
        let options = LoadOptions::default();
        let manifest = discover_checkpoint(directory.path(), &options).unwrap();
        let packed = "model.layers.0.self_attn.k_proj.weight";
        let retained = "model.layers.0.self_attn.q_proj.weight";
        let expected = [
            ExpectedTensor::new(packed, [128, 32]),
            ExpectedTensor::new(retained, [1, 32]),
        ];
        let mut mappings = MappingPlan::new();
        mappings.split(
            "fused.weight",
            0,
            [(packed.to_owned(), 128), (retained.to_owned(), 1)],
        );
        let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
        let output = directory.path().join("asymmetric-fused.bpk");
        write_sidecar(&output, &[(packed, vec![128, 32])]);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let outcome =
            execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

        assert_eq!(outcome.execution.source_reads, 1);
        assert_eq!(outcome.execution.source_payload_bytes, 16_512);
        assert_eq!(outcome.execution.delivered_tensor_bytes, 4_736);
        assert_eq!(outcome.execution.largest_source_payload_bytes, 16_512);
        assert_eq!(outcome.execution.largest_output_tensor_bytes, 16_384);
        assert_eq!(outcome.execution.peak_host_staging_bytes, 32_896);
        assert_eq!(outcome.packed.tensor_reads, 1);
        assert_eq!(outcome.packed.payload_bytes, 4_608);
        assert_eq!(outcome.packed.tensors_applied, 1);
        assert_eq!(
            sink.operations,
            [format!("write:{packed}"), format!("write:{retained}")]
        );
    }

    /// A fully packed fused split validates its local source and applies aliases last.
    #[test]
    fn complete_fused_split_sidecar_reads_the_shared_local_source_once() {
        let directory = tempfile::tempdir().unwrap();
        write_checkpoint(directory.path(), &[("fused.weight", vec![64, 32])]);
        let options = LoadOptions::default();
        let manifest = discover_checkpoint(directory.path(), &options).unwrap();
        let q = "model.layers.0.self_attn.q_proj.weight";
        let k = "model.layers.0.self_attn.k_proj.weight";
        let alias = "lm_head.weight";
        let expected = [
            ExpectedTensor::new(q, [32, 32]),
            ExpectedTensor::new(k, [32, 32]),
            ExpectedTensor::new(alias, [32, 32]),
        ];
        let mut mappings = MappingPlan::new();
        mappings
            .split("fused.weight", 0, [(q.to_owned(), 32), (k.to_owned(), 32)])
            .alias(q, alias);
        let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
        let output = directory.path().join("complete-fused.bpk");
        write_sidecar(&output, &[(q, vec![32, 32]), (k, vec![32, 32])]);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let outcome =
            execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

        assert_eq!(outcome.execution.tensors_applied, 2);
        assert_eq!(outcome.execution.aliases_applied, 1);
        assert_eq!(outcome.execution.source_reads, 1);
        assert_eq!(outcome.execution.source_payload_bytes, 8_192);
        assert_eq!(outcome.execution.source.shard_opens, 1);
        assert_eq!(outcome.execution.source.tensor_reads, 1);
        assert_eq!(outcome.execution.source.payload_bytes_read, 8_192);
        assert_eq!(outcome.execution.delivered_tensor_bytes, 2_304);
        assert_eq!(outcome.execution.largest_source_payload_bytes, 8_192);
        assert_eq!(outcome.execution.largest_output_tensor_bytes, 4_096);
        assert_eq!(outcome.execution.peak_host_staging_bytes, 12_288);
        assert_eq!(outcome.packed.tensor_reads, 2);
        assert_eq!(outcome.packed.payload_bytes, 2_304);
        assert_eq!(outcome.packed.tensors_applied, 2);
        assert_eq!(outcome.report.applied.len(), 3);
        assert_eq!(sink.aliases, [(alias.to_owned(), q.to_owned())]);
        assert_eq!(
            sink.operations,
            [
                format!("write:{k}"),
                format!("write:{q}"),
                format!("alias:{alias}"),
            ]
        );
        assert!(matches!(sink.tensors[q].dtype, DType::QFloat(_)));
        assert!(matches!(sink.tensors[k].dtype, DType::QFloat(_)));
    }

    /// Source-validated packed execution closes its source before applying aliases.
    #[test]
    fn fully_packed_owner_closes_source_before_alias_installation() {
        let directory = tempfile::tempdir().unwrap();
        let owner = "model.layers.0.self_attn.q_proj.weight";
        let alias = "lm_head.weight";
        write_checkpoint(directory.path(), &[(owner, vec![32, 32])]);
        let options = LoadOptions::default();
        let manifest = discover_checkpoint(directory.path(), &options).unwrap();
        let expected = [
            ExpectedTensor::new(owner, [32, 32]),
            ExpectedTensor::new(alias, [32, 32]),
        ];
        let mut mappings = MappingPlan::new();
        mappings.map(owner, owner).alias(owner, alias);
        let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
        let output = directory.path().join("owner-alias.bpk");
        write_sidecar(&output, &[(owner, vec![32, 32])]);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let closed = Rc::new(Cell::new(false));
        let mut source = CloseAwareLocalSource {
            inner: LocalShardSource::new(&manifest),
            closed: closed.clone(),
            fail_close: false,
        };
        let mut sink = CloseAwareSink {
            closed,
            operations: Vec::new(),
        };

        let outcome =
            execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap();

        assert_eq!(outcome.execution.source_reads, 1);
        assert_eq!(outcome.execution.source.shard_opens, 1);
        assert_eq!(outcome.execution.tensors_applied, 1);
        assert_eq!(outcome.execution.aliases_applied, 1);
        assert_eq!(outcome.packed.tensor_reads, 1);
        assert_eq!(outcome.packed.tensors_applied, 1);
        assert_eq!(
            sink.operations,
            [format!("write:{owner}"), format!("alias:{alias}")]
        );
    }

    /// A close failure preserves its installed packed owner and suppresses aliases.
    #[test]
    fn source_close_failure_reports_packed_owner_without_pending_alias() {
        let directory = tempfile::tempdir().unwrap();
        let owner = "model.layers.0.self_attn.q_proj.weight";
        let alias = "lm_head.weight";
        write_checkpoint(directory.path(), &[(owner, vec![32, 32])]);
        let options = LoadOptions::default();
        let manifest = discover_checkpoint(directory.path(), &options).unwrap();
        let expected = [
            ExpectedTensor::new(owner, [32, 32]),
            ExpectedTensor::new(alias, [32, 32]),
        ];
        let mut mappings = MappingPlan::new();
        mappings.map(owner, owner).alias(owner, alias);
        let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
        let output = directory.path().join("owner-close-failure.bpk");
        write_sidecar(&output, &[(owner, vec![32, 32])]);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let closed = Rc::new(Cell::new(false));
        let mut source = CloseAwareLocalSource {
            inner: LocalShardSource::new(&manifest),
            closed: closed.clone(),
            fail_close: true,
        };
        let mut sink = CloseAwareSink {
            closed,
            operations: Vec::new(),
        };

        let failure =
            execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();

        assert_eq!(failure.failed_target, None);
        assert_eq!(failure.execution.source_reads, 1);
        assert_eq!(failure.execution.tensors_applied, 1);
        assert_eq!(failure.execution.aliases_applied, 0);
        assert_eq!(failure.execution.delivered_tensor_bytes, 1_152);
        assert_eq!(failure.packed.tensor_reads, 1);
        assert_eq!(failure.packed.payload_bytes, 1_152);
        assert_eq!(failure.packed.tensors_applied, 1);
        assert_eq!(failure.report.applied, [owner]);
        assert_eq!(failure.report.quantized.len(), 1);
        assert_eq!(failure.report.quantized[0].target, owner);
        assert_eq!(sink.operations, [format!("write:{owner}")]);
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Checkpoint(CheckpointError::InvalidMapping { .. })
        ));
    }

    /// A generic source's bare positive certificate cannot authorize packed bypass.
    #[test]
    fn fused_source_failure_precedes_packed_first_target_installation() {
        let directory = tempfile::tempdir().unwrap();
        write_checkpoint(directory.path(), &[("fused.weight", vec![64, 32])]);
        let options = LoadOptions::default();
        let manifest = discover_checkpoint(directory.path(), &options).unwrap();
        let packed_first = "model.layers.0.self_attn.k_proj.weight";
        let uncovered_second = "model.layers.0.self_attn.q_proj.weight";
        let expected = [
            ExpectedTensor::new(packed_first, [32, 32]),
            ExpectedTensor::new(uncovered_second, [32, 32]),
        ];
        let mut mappings = MappingPlan::new();
        mappings.split(
            "fused.weight",
            0,
            [
                (packed_first.to_owned(), 32),
                (uncovered_second.to_owned(), 32),
            ],
        );
        let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
        assert_eq!(plan.tensors[0].target, packed_first);
        let output = directory.path().join("packed-first.bpk");
        write_sidecar(&output, &[(packed_first, vec![32, 32])]);
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut source = FailingSource {
            certify: true,
            ..FailingSource::default()
        };
        let mut sink = RecordingSink::default();

        let failure =
            execute_quantized_load_plan(&plan, &mut source, &mut artifact, &mut sink).unwrap_err();

        assert_eq!(failure.failed_target.as_deref(), Some(packed_first));
        assert_eq!(source.reads, 1);
        assert!(source.closed);
        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.execution.tensors_applied, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert_eq!(failure.packed.tensors_applied, 0);
        assert!(failure.report.applied.is_empty());
        assert!(failure.report.quantized.is_empty());
        assert!(sink.operations.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Checkpoint(CheckpointError::InvalidTensorPayload { .. })
        ));
    }

    /// Packed post-transform targets still validate transpose and concatenation sources.
    #[test]
    fn packed_transpose_and_concatenation_targets_validate_every_local_input() {
        let directory = tempfile::tempdir().unwrap();
        write_checkpoint(
            directory.path(),
            &[
                ("transpose.weight", vec![32, 64]),
                ("concat.a.weight", vec![16, 32]),
                ("concat.b.weight", vec![16, 32]),
            ],
        );
        let options = LoadOptions::default();
        let manifest = discover_checkpoint(directory.path(), &options).unwrap();
        let transpose = "model.layers.0.self_attn.o_proj.weight";
        let concatenate = "model.layers.0.mlp.down_proj.weight";
        let alias = "lm_head.weight";
        let expected = [
            ExpectedTensor::new(transpose, [64, 32]),
            ExpectedTensor::new(concatenate, [32, 32]),
            ExpectedTensor::new(alias, [64, 32]),
        ];
        let mut mappings = MappingPlan::new();
        mappings
            .transpose_2d("transpose.weight", transpose)
            .concatenate(
                ["concat.a.weight".to_owned(), "concat.b.weight".to_owned()],
                concatenate,
                0,
            )
            .alias(transpose, alias);
        let plan = build_load_plan(&manifest, &expected, &mappings, &options).unwrap();
        let output = directory.path().join("transformed.bpk");
        let conversion = quantize_local_checkpoint_to_burnpack(
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
        assert_eq!(conversion.execution.source_reads, 3);
        assert_eq!(conversion.execution.source_payload_bytes, 12_288);
        assert_eq!(conversion.staging.tensor_count, 2);

        let mut reference_sink = RecordingSink::default();
        execute_local_load_plan(&manifest, &plan, &mut reference_sink).unwrap();
        let mut parity_artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
        for target in [transpose, concatenate] {
            let packed = parity_artifact.read_weight(target).unwrap();
            let tolerance = packed.scales().iter().copied().fold(0.0_f32, f32::max) / 2.0 + 1.0e-5;
            let restored = packed.dequantize_f32().unwrap();
            let reference = reference_sink.tensors[target].as_slice::<f32>().unwrap();
            let restored = restored.as_slice::<f32>().unwrap();
            assert_eq!(restored.len(), reference.len());
            assert!(
                restored
                    .iter()
                    .zip(reference)
                    .all(|(restored, reference)| (restored - reference).abs() <= tolerance),
                "{target} quantized transform exceeded error tolerance {tolerance}"
            );
        }
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let outcome =
            execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink).unwrap();

        assert_eq!(outcome.execution.tensors_applied, 2);
        assert_eq!(outcome.execution.aliases_applied, 1);
        assert_eq!(outcome.execution.source_reads, 0);
        assert_eq!(outcome.execution.source_payload_bytes, 0);
        assert_eq!(outcome.execution.source.shard_opens, 0);
        assert_eq!(outcome.execution.delivered_tensor_bytes, 3_456);
        assert_eq!(outcome.execution.largest_source_payload_bytes, 0);
        assert_eq!(outcome.execution.largest_output_tensor_bytes, 2_304);
        assert_eq!(outcome.execution.peak_host_staging_bytes, 2_304);
        assert_eq!(outcome.packed.tensor_reads, 2);
        assert_eq!(outcome.packed.payload_bytes, 3_456);
        assert_eq!(outcome.packed.largest_payload_bytes, 2_304);
        assert_eq!(outcome.packed.tensors_applied, 2);
        assert_eq!(outcome.report.quantized.len(), 2);
        assert_eq!(sink.aliases, [(alias.to_owned(), transpose.to_owned())]);
        assert!(matches!(sink.tensors[transpose].dtype, DType::QFloat(_)));
        assert!(matches!(sink.tensors[concatenate].dtype, DType::QFloat(_)));
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

    /// A packed target cannot conceal a source descriptor mutated after planning.
    #[test]
    fn skipped_source_descriptor_is_still_checked_against_the_manifest() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, mut plan, converted_output) = executable_sidecar(&directory);
        fs::remove_file(converted_output).unwrap();
        let target = "model.layers.0.self_attn.q_proj.weight";
        let output = directory.path().join("descriptor-v1.bpk");
        write_sidecar(&output, &[(target, vec![32, 32])]);
        let tensor = plan
            .tensors
            .iter_mut()
            .find(|tensor| tensor.target == target)
            .unwrap();
        tensor.sources[0].data_offsets[0] += 4;
        tensor.sources[0].data_offsets[1] += 4;
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let failure = execute_local_quantized_load_plan(&manifest, &plan, &mut artifact, &mut sink)
            .unwrap_err();

        assert_eq!(failure.failed_target.as_deref(), Some(target));
        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.execution.source.shard_opens, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert!(failure.report.applied.is_empty());
        assert!(sink.tensors.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Checkpoint(
                crate::checkpoint::CheckpointError::PlanManifestMismatch { .. }
            )
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
                    equivalences: Vec::new(),
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
        let audited =
            execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
                .unwrap_err();
        assert_eq!(
            audited.format_version,
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
        );
        assert_eq!(audited.preflight.passes_started, 1);
        assert_eq!(audited.preflight.passes_completed, 1);
        assert_eq!(audited.preflight.tensor_reads, 1);
        assert_eq!(audited.preflight.payload_bytes, 1_152);
        assert_eq!(audited.preflight.largest_payload_bytes, 1_152);
        assert!(audited.preflight.peak_owned_payload_bytes >= 1_280);
        let failure = &audited.execution;

        assert_eq!(
            failure.failed_target.as_deref(),
            Some("model.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(failure.execution.source_reads, 2);
        assert_eq!(failure.execution.source_payload_bytes, 384);
        assert_eq!(failure.execution.source.tensor_reads, 2);
        assert_eq!(failure.execution.source.payload_bytes_read, 384);
        assert_eq!(failure.execution.tensors_applied, 2);
        assert_eq!(failure.execution.delivered_tensor_bytes, 384);
        assert_eq!(failure.execution.largest_source_payload_bytes, 256);
        assert_eq!(failure.execution.largest_output_tensor_bytes, 1_152);
        assert_eq!(failure.execution.peak_host_staging_bytes, 1_152);
        assert_eq!(failure.packed.tensor_reads, 1);
        assert_eq!(failure.packed.payload_bytes, 1_152);
        assert_eq!(failure.packed.largest_payload_bytes, 1_152);
        assert_eq!(failure.packed.tensors_applied, 0);
        assert_eq!(
            failure.report.applied,
            [
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
            ]
        );
        assert!(failure.report.quantized.is_empty());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Architecture(_))
        ));
    }

    /// A corrupt v2 packed payload fails before the first model sink mutation.
    #[test]
    fn corrupt_v2_packed_payload_cannot_create_partial_model_writes() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest, plan, output) = executable_sidecar(&directory);
        let target = "model.layers.0.self_attn.q_proj.weight";
        let payload = {
            let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
            artifact.tensor_payload(target).unwrap().as_bytes().to_vec()
        };
        let mut artifact_bytes = fs::read(&output).unwrap();
        let payload_offset = unique_subslice(&artifact_bytes, &payload);
        artifact_bytes[payload_offset] ^= 1;
        fs::write(&output, artifact_bytes).unwrap();
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let audited =
            execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
                .unwrap_err();
        assert_eq!(
            audited.format_version,
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
        );
        assert_eq!(audited.preflight.passes_started, 1);
        assert_eq!(audited.preflight.passes_completed, 0);
        assert_eq!(audited.preflight.tensor_reads, 1);
        assert_eq!(audited.preflight.payload_bytes, 1_152);
        assert_eq!(audited.preflight.largest_payload_bytes, 1_152);
        assert!(audited.preflight.peak_owned_payload_bytes >= 1_152);
        let failure = &audited.execution;

        assert_eq!(failure.failed_target, None);
        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.execution.source_payload_bytes, 0);
        assert_eq!(failure.execution.tensors_applied, 0);
        assert_eq!(failure.execution.delivered_tensor_bytes, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert_eq!(failure.packed.payload_bytes, 0);
        assert_eq!(failure.packed.tensors_applied, 0);
        assert!(failure.report.applied.is_empty());
        assert!(failure.report.quantized.is_empty());
        assert!(sink.tensors.is_empty());
        assert_eq!(failure.provenance.pre_execution, None);
        assert_eq!(failure.provenance.post_execution, None);
        assert!(matches!(
            &failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::PackedPayloadDigestMismatch { name, .. }
            )) if name == target
        ));
    }

    /// Legacy v1 semantic validation also completes before any model sink call.
    #[test]
    fn corrupt_v1_packed_payload_cannot_create_partial_model_writes() {
        let directory = tempfile::tempdir().unwrap();
        write_checkpoint(
            directory.path(),
            &[
                ("model.embed_tokens.weight", vec![2, 32]),
                ("model.layers.0.input_layernorm.weight", vec![32]),
                ("model.layers.0.self_attn.q_proj.weight", vec![32, 32]),
            ],
        );
        let (manifest, plan) = fixture_plan(directory.path(), None);
        let target = "model.layers.0.self_attn.q_proj.weight";
        let output = directory.path().join("legacy-corrupt.bpk");
        write_sidecar(&output, &[(target, vec![32, 32])]);
        let payload = {
            let mut artifact = QuantizedBurnpackArtifact::open(&output).unwrap();
            assert_eq!(artifact.manifest().format_version, 1);
            assert!(artifact.manifest().tensors[target].payload_sha256.is_none());
            artifact.tensor_payload(target).unwrap().as_bytes().to_vec()
        };
        let mut artifact_bytes = fs::read(&output).unwrap();
        let payload_offset = unique_subslice(&artifact_bytes, &payload);
        artifact_bytes[payload_offset] = 0x80;
        fs::write(&output, artifact_bytes).unwrap();
        let mut artifact = QuantizedBurnpackArtifact::open(output).unwrap();
        let mut sink = RecordingSink::default();

        let audited =
            execute_local_quantized_load_plan_audited(&manifest, &plan, &mut artifact, &mut sink)
                .unwrap_err();
        assert_eq!(audited.format_version, 1);
        assert_eq!(audited.preflight.passes_started, 1);
        assert_eq!(audited.preflight.passes_completed, 0);
        assert_eq!(audited.preflight.tensor_reads, 1);
        assert_eq!(audited.preflight.payload_bytes, 1_152);
        assert!(audited.preflight.peak_owned_payload_bytes >= 1_280);
        let failure = &audited.execution;

        assert_eq!(failure.failed_target, None);
        assert_eq!(failure.execution.source_reads, 0);
        assert_eq!(failure.packed.tensor_reads, 0);
        assert!(failure.report.applied.is_empty());
        assert!(sink.operations.is_empty());
        assert_eq!(failure.provenance, Default::default());
        assert!(matches!(
            failure.cause,
            ExecutionFailureCause::Sink(QuantizedSinkError::Quantization(
                QuantizationError::InvalidQuantizedValue {
                    index: 0,
                    value: -128,
                }
            ))
        ));
    }
}
