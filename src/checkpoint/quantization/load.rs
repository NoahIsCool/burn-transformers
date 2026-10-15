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
    QuantizationError, QuantizedBurnpackArtifact, SourceFloatDType,
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
    Quantization(
        /// Original packed artifact validation or conversion failure.
        QuantizationError,
    ),
    /// The architecture rejected the final source-precision or packed tensor.
    Architecture(
        /// Error returned by the architecture's parameter sink.
        E,
    ),
}

impl<E: fmt::Display> fmt::Display for QuantizedSinkError<E> {
    /// Formats either the packed-payload failure or the architecture sink rejection.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quantization(error) => error.fmt(formatter),
            Self::Architecture(error) => error.fmt(formatter),
        }
    }
}

impl<E: StdError + 'static> StdError for QuantizedSinkError<E> {
    /// Exposes the quantization or architecture error as the next diagnostic cause.
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Quantization(error) => Some(error),
            Self::Architecture(error) => Some(error),
        }
    }
}

impl<E> From<QuantizationError> for QuantizedSinkError<E> {
    /// Wraps a packed-format error without changing its structured information.
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
    /// Formats the target-specific packed-load failure when execution reached a target.
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
    /// Exposes the common executor cause while preserving packed-load audit counters.
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
    /// Delegates display to the execution failure; preflight evidence remains structured.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.execution.fmt(formatter)
    }
}

impl<E: StdError + 'static> StdError for AuditedQuantizedExecutionFailure<E> {
    /// Exposes the execution failure, including its own checkpoint or sink cause.
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
            report.extend_quantized(records);
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
            report.extend_quantized(records);
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

// Source wrappers separate structural validation from content-authenticated bypass.

/// Local source whose positive descriptor result follows a complete SHA-256 gate.
struct ProvenanceLocalSource<'a> {
    /// Source retained for descriptors and ordinary payload reads.
    inner: LocalShardSource<'a>,
    /// Whether complete source content has passed the provenance digest gate.
    content_verified: bool,
}

/// Masks generic descriptor booleans unless the caller established artifact binding.
struct DescriptorBypassPolicySource<'a, S> {
    /// Borrowed source whose descriptor decisions are restricted by this wrapper.
    inner: &'a mut S,
    /// Whether artifact binding authorizes descriptor-only payload bypass.
    allow_descriptor_bypass: bool,
}

impl<S: TensorSource> TensorSource for DescriptorBypassPolicySource<'_, S> {
    /// Permits descriptor-only bypass only when both the source and binding policy authorize it.
    fn validate_tensor_descriptor(
        &self,
        source: &crate::checkpoint::PlannedSource,
    ) -> crate::checkpoint::Result<bool> {
        let descriptor_matches = self.inner.validate_tensor_descriptor(source)?;
        Ok(self.allow_descriptor_bypass && descriptor_matches)
    }

    /// Forwards a normal payload read without changing source ownership or accounting.
    fn read_tensor(
        &mut self,
        source: &crate::checkpoint::PlannedSource,
    ) -> crate::checkpoint::Result<crate::checkpoint::SourceTensor> {
        self.inner.read_tensor(source)
    }

    /// Closes the wrapped source even when descriptor bypass was disabled.
    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.inner.close()
    }

    /// Returns the wrapped source's unmodified I/O audit counters.
    fn stats(&self) -> crate::checkpoint::SourceStats {
        self.inner.stats()
    }
}

impl TensorSource for ProvenanceLocalSource<'_> {
    /// Revalidates structure and permits bypass only after complete content verification.
    fn validate_tensor_descriptor(
        &self,
        source: &crate::checkpoint::PlannedSource,
    ) -> crate::checkpoint::Result<bool> {
        let _ = self.inner.validate_tensor_descriptor(source)?;
        Ok(self.content_verified)
    }

    /// Reads an ordinary tensor through the same contained local shard source.
    fn read_tensor(
        &mut self,
        source: &crate::checkpoint::PlannedSource,
    ) -> crate::checkpoint::Result<crate::checkpoint::SourceTensor> {
        self.inner.read_tensor(source)
    }

    /// Releases the verified source's active shard handle.
    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.inner.close()
    }

    /// Returns local read counters independently of provenance-hashing counters.
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
            report: plan.report.without_progress(),
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
    let mut report = plan.report.without_progress();
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

// The overlay records successful installs separately from sidecar payload reads.

/// Sink wrapper that chooses packed or original data at the final target boundary.
struct QuantizedOverlaySink<'a, K> {
    /// Validated sidecar supplying packed payloads for overridden targets.
    artifact: &'a mut QuantizedBurnpackArtifact,
    /// Architecture sink receiving final packed or source-precision tensors.
    inner: &'a mut K,
    /// Packed targets accepted by the architecture sink.
    records: Vec<QuantizationRecord>,
    /// Sidecar reads and packed application counters.
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
    /// Distinguishes packed-payload failures from architecture rejections.
    type Error = QuantizedSinkError<K::Error>;

    /// Forwards a final tensor and records packed application only after sink success.
    ///
    /// Targets absent from the sidecar retain the ordinary source-precision path.
    /// Packed read statistics are maintained separately from successful installation.
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
            scheme: metadata.runtime_label(),
            bytes: payload_bytes,
        };

        self.inner
            .write_tensor(target, data)
            .map_err(QuantizedSinkError::Architecture)?;
        self.stats.tensors_applied = tensors_applied;
        self.records.push(record);
        Ok(())
    }

    /// Delegates a model alias after its storage owner has been installed by the executor.
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

/// Exposes the stable executable discriminator without requiring enum matching.
#[must_use]
pub const fn burn_qfloat_execution_support() -> PackedExecutionSupport {
    PackedExecutionSupport::BurnQFloatMatmul
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
