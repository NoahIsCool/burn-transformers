//! Burn-facing execution of validated checkpoint load plans.
//!
//! The executor is deliberately split at an ownership boundary. A [`TensorSource`]
//! returns one SafeTensors payload, transformation code constructs one final Burn
//! [`TensorData`], and a [`TensorSink`] takes ownership of it. Architecture modules
//! implement the sink because Burn parameters encode tensor rank in their Rust
//! types: only the owning model knows whether a dynamic key names `Param<Tensor<B,
//! 1>>`, `Param<Tensor<B, 2>>`, or another field. Keeping that knowledge out of the
//! generic loader prevents unchecked rank casts and permits the sink to transfer a
//! tensor to its backend immediately.
//!
//! Physical checkpoint equivalences are compared before the first sink mutation.
//! Model alias operations then run only after every storage-owning parameter
//! succeeds. These boundaries ensure a mismatched redundant payload cannot leak a
//! partially initialized model and no model alias can reference a missing owner.

mod source;
mod transform;

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use burn::tensor::{DType, TensorData};
use burn::{
    module::{Module, Param, ParamId},
    tensor::{
        Tensor,
        backend::{Backend, DTypeUsage},
    },
};

pub use source::{LocalShardSource, SourceStats, SourceTensor, TensorSource};

use super::{
    CheckpointEquivalenceRecord, CheckpointError, CheckpointManifest, LoadPlan, LoadReport,
    PlannedSource, PlannedTensor, PlannedTransform, TensorDType,
};

/// Destination for streamed, transformed Burn tensor data.
///
/// `write_tensor` receives ownership so an implementation can synchronously create
/// a backend tensor and replace the correctly ranked model parameter without
/// retaining a second host copy. `alias_tensor` must make `target` share the
/// storage-owning parameter named by `source_target`; copying the tensor would break
/// tied-weight semantics and inflate the model memory estimate.
///
/// Implementations should perform their own key-to-field dispatch and return an
/// error for unknown keys or ranks. The executor records a target as applied only
/// after the corresponding method returns `Ok(())`.
pub trait TensorSink {
    /// Sink-specific error type, such as an architecture key or backend failure.
    type Error: Error + Send + Sync + 'static;

    /// Installs one storage-owning model parameter.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> Result<(), Self::Error>;

    /// Installs one tied parameter alias to an already written storage owner.
    fn alias_tensor(&mut self, target: &str, source_target: &str) -> Result<(), Self::Error>;
}

/// Failure to install checkpoint data into a statically ranked Burn parameter.
///
/// The generic checkpoint executor reasons about dynamic tensor metadata, while
/// Burn model fields encode rank in `Tensor<B, D>`. This error reports a rank,
/// shape, or element-kind disagreement before invoking a backend operation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParameterInstallError {
    /// The dynamic checkpoint tensor has a different rank from the model field.
    Rank {
        /// Parameter rank encoded by its Rust field.
        expected: usize,
        /// Rank carried by the checkpoint tensor.
        actual: usize,
    },
    /// The checkpoint shape differs from the parameter's saved shape.
    Shape {
        /// Saved/checkpoint-facing shape declared by the Burn parameter.
        expected: Vec<usize>,
        /// Shape carried by the checkpoint tensor.
        actual: Vec<usize>,
    },
    /// A floating-point parameter was given integer or Boolean data.
    NonFloat {
        /// Human-readable Burn dtype name.
        dtype: String,
    },
    /// The target backend cannot store the checkpoint dtype on this device.
    UnsupportedStorage {
        /// Human-readable Burn dtype name.
        dtype: String,
    },
    /// Burn 0.21 cannot construct or differentiate packed tensors on autodiff.
    QuantizedAutodiff,
    /// Reloading existing QFloat storage cannot inspect its saved shape cheaply.
    ///
    /// The save mapper may dequantize the complete weight, and Burn exposes no
    /// shape-only mapper inverse. Initial loading into a lazy parameter remains
    /// supported; this rejects only replacement of an initialized QFloat value.
    InitializedQuantized,
    /// The shape and ordinary float dtype cannot describe a host-sized payload.
    PayloadSizeOverflow,
    /// Ordinary float bytes disagree with the declared shape and dtype.
    PayloadLength {
        /// Required bytes, computed without allocating or initializing the parameter.
        expected: usize,
        /// Bytes actually owned by the supplied tensor data.
        actual: usize,
    },
}

impl fmt::Display for ParameterInstallError {
    /// Describes the rejected storage contract without inspecting tensor values.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rank { expected, actual } => {
                write!(
                    formatter,
                    "expected rank {expected}, received rank {actual}"
                )
            }
            Self::Shape { expected, actual } => {
                write!(
                    formatter,
                    "expected shape {expected:?}, received {actual:?}"
                )
            }
            Self::NonFloat { dtype } => {
                write!(formatter, "expected floating-point data, received {dtype}")
            }
            Self::UnsupportedStorage { dtype } => {
                write!(formatter, "backend cannot store checkpoint dtype {dtype}")
            }
            Self::QuantizedAutodiff => write!(
                formatter,
                "Burn 0.21 autodiff cannot load or differentiate packed Q4/Q8 weights, including frozen weights; explicitly dequantize on the inner backend before loading floating data"
            ),
            Self::InitializedQuantized => write!(
                formatter,
                "reloading an initialized QFloat parameter is unsupported without saved-shape metadata; load checkpoint data into a fresh lazy parameter"
            ),
            Self::PayloadSizeOverflow => {
                write!(formatter, "checkpoint float payload size overflows usize")
            }
            Self::PayloadLength { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} checkpoint float bytes, received {actual}"
                )
            }
        }
    }
}

impl Error for ParameterInstallError {}

/// Installs checkpoint-shaped data into a Burn floating-point parameter.
///
/// Burn's parameter mapper is semantically important: for example, a
/// `LinearLayout::Col` weight is serialized as `[out, in]`, then transposed to
/// `[in, out]` when loaded for matrix multiplication. Replacing the field with
/// `Param::from_tensor` would discard that mapper and produce a model with
/// incorrectly oriented projections. For an uninitialized parameter this helper
/// validates against [`Param::lazy_shape`]. Once initialized, Burn reports the
/// runtime shape instead, so the helper obtains the checkpoint-facing shape by
/// applying [`Param::transform_for_save`] to a shared tensor handle. Burn 0.21
/// exposes no shape-only mapper inverse. Its ordinary column-layout mapper is a
/// transpose, but a caller-defined save mapper may allocate or perform backend
/// work; that is the cost of validating an already initialized mapped parameter.
/// The lazy path never invokes the initializer or save mapper. Shape inspection
/// itself requests no host tensor readback or dtype conversion and does not
/// replace the original parameter before validation succeeds.
///
/// Reloading an already initialized QFloat parameter returns
/// [`ParameterInstallError::InitializedQuantized`] before invoking any save
/// mapper. Burn Flex may implement a block-quantized transpose through complete
/// dequantization and requantization, and Burn provides neither a shape-only
/// inverse nor a way to identify unmapped parameters. The guard therefore also
/// covers unmapped initialized QFloat values. This narrow unsupported reload
/// boundary does not restrict initial lazy packed loading or quantized inference.
/// Packed loading into an autodiff backend is rejected separately with
/// [`ParameterInstallError::QuantizedAutodiff`]. Burn 0.21 does not implement its
/// quantized tensor constructor or input gradients through frozen packed
/// weights. Explicitly dequantize before loading floating data for training.
///
/// The helper constructs the tensor at its checkpoint dtype and
/// delegates to Burn's ordinary record-loading path so device, gradient, and
/// layout policies remain intact. Incoming checkpoint data is untracked until
/// the layout mapper finishes: marking it trainable before a transpose would
/// turn a column-layout weight into a non-leaf and make Burn's subsequent
/// gradient-policy restoration panic on an autodiff backend.
///
/// Ordinary float storage is checked against `product(shape) * dtype_bytes`
/// before constructing the incoming backend tensor. Packed inputs must already
/// have passed their format-specific payload validation, as in the crate's QFloat adapter; their
/// codes and scale blocks do not follow that ordinary byte-count equation.
///
/// The temporary parameter clone shares the incoming backend allocation; it is
/// used solely to move the existing parameter out of `&mut` safely and is
/// released after the mapped parameter is installed.
pub fn install_float_parameter<B: Backend, const D: usize>(
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> std::result::Result<(), ParameterInstallError> {
    let actual = data.shape.to_vec();
    if actual.len() != D {
        return Err(ParameterInstallError::Rank {
            expected: D,
            actual: actual.len(),
        });
    }
    if matches!(data.dtype, DType::QFloat(_)) && B::ad_enabled(&parameter.lazy_device()) {
        return Err(ParameterInstallError::QuantizedAutodiff);
    }
    let initialized = parameter.is_initialized();
    if initialized && matches!(parameter.val().dtype(), DType::QFloat(_)) {
        return Err(ParameterInstallError::InitializedQuantized);
    }
    let expected = if initialized {
        parameter.transform_for_save().val().shape().to_vec()
    } else {
        parameter.lazy_shape().to_vec()
    };
    if actual != expected {
        return Err(ParameterInstallError::Shape { expected, actual });
    }
    let is_quantized_float = matches!(data.dtype, DType::QFloat(_));
    if !data.dtype.is_float() && !is_quantized_float {
        return Err(ParameterInstallError::NonFloat {
            dtype: format!("{:?}", data.dtype),
        });
    }
    if !is_quantized_float {
        validate_float_payload_length(&actual, data.dtype, data.bytes.len())?;
    }

    let device = parameter.lazy_device();
    let dtype = data.dtype;
    if !B::dtype_usage(&device, dtype).contains(DTypeUsage::Storage) {
        return Err(ParameterInstallError::UnsupportedStorage {
            dtype: format!("{dtype:?}"),
        });
    }
    let tensor = Tensor::<B, D>::from_data(data, (&device, dtype));
    let record = Param::initialized(ParamId::new(), tensor);
    let existing = std::mem::replace(parameter, record.clone());
    *parameter = existing.load_record(record);
    Ok(())
}

/// Checks dense floating storage before a backend can retain malformed metadata.
///
/// This equation deliberately excludes packed QFloat, whose scale and code
/// regions are validated by the quantization layer. Overflow is an error even
/// for tiny owned buffers; no allocation is attempted to test representability.
fn validate_float_payload_length(
    shape: &[usize],
    dtype: DType,
    actual: usize,
) -> std::result::Result<(), ParameterInstallError> {
    let elements = shape
        .iter()
        .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or(ParameterInstallError::PayloadSizeOverflow)?;
    let expected = elements
        .checked_mul(dtype.size())
        .ok_or(ParameterInstallError::PayloadSizeOverflow)?;
    if expected != actual {
        return Err(ParameterInstallError::PayloadLength { expected, actual });
    }
    Ok(())
}

/// Runtime counters for one pass-two execution attempt.
///
/// Host staging counts memory owned by the generic loader before `TensorData` is
/// handed to the sink. Backend allocations and any copies retained by a sink are
/// intentionally outside this number.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExecutionStats {
    /// Storage-owning target tensors accepted by the sink.
    pub tensors_applied: usize,
    /// Alias targets accepted by the sink.
    pub aliases_applied: usize,
    /// Physical checkpoint alias/owner pairs verified before sink mutation.
    pub equivalences_verified: usize,
    /// Source reads performed specifically for equivalence preflight.
    pub equivalence_source_reads: u64,
    /// Raw source bytes read specifically for equivalence preflight.
    pub equivalence_source_payload_bytes: u64,
    /// Successful source payload reads; fused-split groups count as one read.
    pub source_reads: u64,
    /// Total source payload bytes returned to transformation code.
    pub source_payload_bytes: u64,
    /// Sum of target bytes transferred into the sink.
    pub delivered_tensor_bytes: u64,
    /// Largest individual source payload held by transformation code.
    pub largest_source_payload_bytes: u64,
    /// Largest individual target buffer materialized by the executor.
    ///
    /// For a non-certifying packed override this can include the validated
    /// full-precision value discarded immediately before the packed replacement
    /// is decoded, even though that value never reaches the sink.
    pub largest_output_tensor_bytes: u64,
    /// Largest simultaneously held alias-plus-owner comparison payload.
    pub largest_equivalence_pair_bytes: u64,
    /// Maximum simultaneous source-plus-output bytes controlled by the executor.
    pub peak_host_staging_bytes: u64,
    /// Source-specific file/resource counters at completion or failure.
    pub source: SourceStats,
}

/// Successful result of executing a complete load plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionOutcome {
    /// Finalized report whose `applied` list covers every model parameter.
    pub report: LoadReport,
    /// Runtime I/O and staging counters.
    pub stats: ExecutionStats,
}

/// Underlying category of an [`ExecutionFailure`].
#[derive(Debug)]
#[non_exhaustive]
pub enum ExecutionFailureCause<E> {
    /// Source, plan, payload, transform, conversion, or staging failure.
    Checkpoint(CheckpointError),
    /// Architecture/backend sink rejected a tensor or alias.
    Sink(E),
}

impl<E: fmt::Display> fmt::Display for ExecutionFailureCause<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Sink(error) => write!(formatter, "tensor sink failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for ExecutionFailureCause<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Checkpoint(error) => Some(error),
            Self::Sink(error) => Some(error),
        }
    }
}

/// Audited failure from a partially executed load plan.
///
/// A sink is permitted to be non-transactional. Consequently, `report.applied`
/// lists the exact set of writes that returned successfully; callers can discard
/// the partially initialized model or implement rollback at the sink boundary.
#[derive(Debug)]
pub struct ExecutionFailure<E> {
    /// Target being materialized or written, or `None` for plan/source-close errors.
    pub failed_target: Option<String>,
    /// Checkpoint-side or sink-side failure.
    pub cause: ExecutionFailureCause<E>,
    /// Finalized partial load report.
    pub report: LoadReport,
    /// Counters observed before failure.
    pub stats: ExecutionStats,
}

/// Result returned by pass-two execution.
///
/// Failures are boxed because they intentionally carry a complete partial report
/// and runtime audit counters; keeping that rich diagnostic off the success-path
/// stack also makes the result inexpensive to move.
pub type ExecutionResult<T, E> = Result<T, Box<ExecutionFailure<E>>>;

impl<E: fmt::Display> fmt::Display for ExecutionFailure<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(target) = &self.failed_target {
            write!(
                formatter,
                "checkpoint execution failed at `{target}`: {}",
                self.cause
            )
        } else {
            write!(formatter, "checkpoint execution failed: {}", self.cause)
        }
    }
}

impl<E: Error + 'static> Error for ExecutionFailure<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.cause)
    }
}

/// Executes a validated load plan from an arbitrary audited source into a sink.
///
/// The complete mutable plan is preflighted before the first source read or sink
/// call. Primary operations are then scheduled by source shard and byte offset,
/// independently of the target-name order retained by [`LoadPlan`]. Consequently,
/// a local source opens each shard once for the usual unary and fused-split model
/// mappings even when architecture target names are interleaved across shards.
/// Physical alias/owner pairs are first compared as exact raw payloads, before
/// `TensorSink` receives any call. Targets slicing the same fused source are grouped at that source's physical
/// position: the payload remains resident while each output is materialized and
/// handed off, avoiding rereads without retaining multiple target buffers. A
/// concatenation remains one indivisible operation and keeps the final output while
/// replacing one source payload at a time. Source resources are closed before
/// aliases are installed.
pub fn execute_load_plan<S, K>(
    plan: &LoadPlan,
    source: &mut S,
    sink: &mut K,
) -> ExecutionResult<ExecutionOutcome, K::Error>
where
    S: TensorSource,
    K: TensorSink,
{
    execute_load_plan_with_overrides(plan, source, sink, |_, _| false, |_, _| Ok(None))
}

/// Executes a validated plan with optional post-transform target overrides.
///
/// The immutable predicate is evaluated before a fused-slice group acquires its
/// shared source. An overridden target bypasses that source only when the source
/// independently certifies every planned descriptor through
/// [`TensorSource::validate_tensor_descriptor`]. Certification must bind the
/// descriptor to authenticated checkpoint content; structural header validation
/// alone is insufficient. Sources without such a certificate take the strict
/// fallback: the ordinary source payload is read and transformed for validation,
/// discarded, and only then replaced. When
/// every target in a fused group is both overridden and certified, its physical
/// payload is never requested. A partial group still reads the shared source
/// exactly once, while overridden targets retain the group's established
/// installation order. The materializer returns `Some(data)` only for targets
/// selected by the predicate; an inconsistent `None` is reported as a checkpoint
/// error rather than panicking.
///
/// This hook is crate-internal because overrides must already have passed an
/// architecture-specific contract proving their post-transform shapes and
/// storage semantics. The ordinary public executor above supplies a predicate
/// that is always false and therefore preserves its existing behavior.
pub(super) fn execute_load_plan_with_overrides<S, K, H, R>(
    plan: &LoadPlan,
    source: &mut S,
    sink: &mut K,
    has_override: H,
    mut read_override: R,
) -> ExecutionResult<ExecutionOutcome, K::Error>
where
    S: TensorSource,
    K: TensorSink,
    H: Fn(&K, &PlannedTensor) -> bool,
    R: FnMut(&mut K, &PlannedTensor) -> Result<Option<TensorData>, K::Error>,
{
    let mut report = plan.report.clone();
    report.applied.clear();
    report.quantized.clear();
    report.equivalence_verified.clear();
    let mut stats = ExecutionStats::default();

    if let Err(error) = transform::validate_plan(plan) {
        let _ = source.close();
        return Err(checkpoint_failure(None, error, report, stats, source));
    }

    if let Err((failed_alias, error)) =
        verify_checkpoint_equivalences(plan, source, &mut report, &mut stats)
    {
        let _ = source.close();
        return Err(checkpoint_failure(
            Some(failed_alias),
            error,
            report,
            stats,
            source,
        ));
    }

    // Freeze ownership before any source or sink mutation. Besides making the
    // scheduling decision deterministic, this prevents a stateful predicate
    // from selecting a target during group preflight and changing its answer at
    // installation time.
    let overridden = plan
        .tensors
        .iter()
        .map(|tensor| has_override(sink, tensor))
        .collect::<Vec<_>>();
    let mut can_skip_source = vec![false; plan.tensors.len()];
    for (index, tensor) in plan.tensors.iter().enumerate() {
        if !overridden[index] {
            continue;
        }
        let mut descriptors_certified = true;
        for planned_source in &tensor.sources {
            match source.validate_tensor_descriptor(planned_source) {
                Ok(certified) => descriptors_certified &= certified,
                Err(error) => {
                    let _ = source.close();
                    return Err(checkpoint_failure(
                        Some(tensor.target.clone()),
                        error,
                        report,
                        stats,
                        source,
                    ));
                }
            }
        }
        can_skip_source[index] = descriptors_certified;
    }

    for group in execution_groups(&plan.tensors) {
        let source_target_index = group.iter().copied().find(|&index| !can_skip_source[index]);
        let shared_payload =
            if let (true, Some(source_target_index)) = (group.len() > 1, source_target_index) {
                let tensor = &plan.tensors[source_target_index];
                let Some(planned_source) = tensor.sources.first() else {
                    let _ = source.close();
                    return Err(checkpoint_failure(
                        Some(tensor.target.clone()),
                        CheckpointError::InvalidMapping {
                            message: format!(
                                "fused-split target `{}` has no source descriptor",
                                tensor.target
                            ),
                        },
                        report,
                        stats,
                        source,
                    ));
                };
                match transform::read_verified(source, planned_source, &mut stats) {
                    Ok(payload) => Some(payload),
                    Err(error) => {
                        let _ = source.close();
                        return Err(checkpoint_failure(
                            Some(tensor.target.clone()),
                            error,
                            report,
                            stats,
                            source,
                        ));
                    }
                }
            } else {
                None
            };

        for index in group {
            let tensor = &plan.tensors[index];
            let source_data = if can_skip_source[index] {
                None
            } else {
                let materialized = match &shared_payload {
                    Some(payload) => {
                        transform::materialize_shared_slice(tensor, payload, &mut stats)
                    }
                    None => transform::materialize(tensor, source, &mut stats),
                };
                match materialized {
                    Ok(data) => Some(data),
                    Err(error) => {
                        let _ = source.close();
                        return Err(checkpoint_failure(
                            Some(tensor.target.clone()),
                            error,
                            report,
                            stats,
                            source,
                        ));
                    }
                }
            };
            let data = if overridden[index] {
                // A non-certifying source reaches this branch only after its
                // ordinary materialized value has been fully validated. Drop it
                // before decoding the override so the fallback does not retain
                // two output buffers at once.
                drop(source_data);
                let override_data = match read_override(sink, tensor) {
                    Ok(Some(data)) => data,
                    Ok(None) => {
                        let _ = source.close();
                        return Err(checkpoint_failure(
                            Some(tensor.target.clone()),
                            CheckpointError::InvalidMapping {
                                message: format!(
                                    "override predicate selected `{}` but its materializer returned no tensor",
                                    tensor.target
                                ),
                            },
                            report,
                            stats,
                            source,
                        ));
                    }
                    Err(error) => {
                        let _ = source.close();
                        return Err(sink_failure(
                            Some(tensor.target.clone()),
                            error,
                            report,
                            stats,
                            source,
                        ));
                    }
                };
                if let Err(error) = observe_override_peak(
                    tensor,
                    &override_data,
                    shared_payload.as_ref(),
                    plan.memory.peak_host_staging_bytes,
                    &mut stats,
                ) {
                    let _ = source.close();
                    return Err(checkpoint_failure(
                        Some(tensor.target.clone()),
                        error,
                        report,
                        stats,
                        source,
                    ));
                }
                override_data
            } else {
                let Some(data) = source_data else {
                    let _ = source.close();
                    return Err(checkpoint_failure(
                        Some(tensor.target.clone()),
                        CheckpointError::InvalidMapping {
                            message: format!(
                                "non-overridden target `{}` unexpectedly bypassed its source",
                                tensor.target
                            ),
                        },
                        report,
                        stats,
                        source,
                    ));
                };
                data
            };
            match install_tensor(tensor, data, sink, &mut report, &mut stats) {
                Ok(()) => {}
                Err(InstallError::Checkpoint(error)) => {
                    let _ = source.close();
                    return Err(checkpoint_failure(
                        Some(tensor.target.clone()),
                        error,
                        report,
                        stats,
                        source,
                    ));
                }
                Err(InstallError::Sink(error)) => {
                    let _ = source.close();
                    return Err(sink_failure(
                        Some(tensor.target.clone()),
                        error,
                        report,
                        stats,
                        source,
                    ));
                }
            }
        }
    }

    if let Err(error) = source.close() {
        return Err(checkpoint_failure(None, error, report, stats, source));
    }

    for alias in &plan.aliases {
        let super::PlannedTransform::Alias { source_target } = &alias.transform else {
            return Err(checkpoint_failure(
                Some(alias.target.clone()),
                CheckpointError::InvalidMapping {
                    message: format!("entry `{}` in alias list is not an alias", alias.target),
                },
                report,
                stats,
                source,
            ));
        };
        let aliases_applied = match stats.aliases_applied.checked_add(1) {
            Some(count) => count,
            None => {
                return Err(checkpoint_failure(
                    Some(alias.target.clone()),
                    CheckpointError::ArithmeticOverflow {
                        context: "counting applied aliases",
                    },
                    report,
                    stats,
                    source,
                ));
            }
        };
        if let Err(error) = sink.alias_tensor(&alias.target, source_target) {
            return Err(sink_failure(
                Some(alias.target.clone()),
                error,
                report,
                stats,
                source,
            ));
        }
        stats.aliases_applied = aliases_applied;
        report.record_applied(alias.target.clone());
    }

    report.finalize();
    stats.source = source.stats();
    Ok(ExecutionOutcome { report, stats })
}

/// Verifies every redundant physical checkpoint tensor before the first write.
fn verify_checkpoint_equivalences<S: TensorSource>(
    plan: &LoadPlan,
    source: &mut S,
    report: &mut LoadReport,
    stats: &mut ExecutionStats,
) -> std::result::Result<(), (String, CheckpointError)> {
    for equivalence in &plan.equivalences {
        let failed_alias = equivalence.alias.name.clone();
        let owner = transform::read_verified(source, &equivalence.owner, stats)
            .map_err(|error| (failed_alias.clone(), error))?;
        observe_equivalence_read(&failed_alias, owner.byte_len(), stats)
            .map_err(|error| (failed_alias.clone(), error))?;
        let alias = transform::read_verified(source, &equivalence.alias, stats)
            .map_err(|error| (failed_alias.clone(), error))?;
        observe_equivalence_read(&failed_alias, alias.byte_len(), stats)
            .map_err(|error| (failed_alias.clone(), error))?;

        let pair_bytes = u64::try_from(owner.byte_len())
            .ok()
            .and_then(|owner| {
                u64::try_from(alias.byte_len())
                    .ok()
                    .and_then(|alias| owner.checked_add(alias))
            })
            .ok_or_else(|| {
                (
                    failed_alias.clone(),
                    CheckpointError::ArithmeticOverflow {
                        context: "accounting checkpoint equivalence payloads",
                    },
                )
            })?;
        stats.largest_equivalence_pair_bytes = stats.largest_equivalence_pair_bytes.max(pair_bytes);
        stats.peak_host_staging_bytes = stats.peak_host_staging_bytes.max(pair_bytes);
        if pair_bytes > plan.memory.peak_host_staging_bytes {
            return Err((
                failed_alias.clone(),
                CheckpointError::MemoryEstimateExceeded {
                    tensor: failed_alias,
                    required: pair_bytes,
                    planned: plan.memory.peak_host_staging_bytes,
                },
            ));
        }
        if owner.bytes() != alias.bytes() {
            let offset = owner
                .bytes()
                .iter()
                .zip(alias.bytes())
                .position(|(owner, alias)| owner != alias)
                .unwrap_or(0);
            return Err((
                failed_alias.clone(),
                CheckpointError::TensorEquivalenceMismatch {
                    alias: failed_alias,
                    owner: equivalence.owner.name.clone(),
                    reason: format!("raw payload differs at byte offset {offset}"),
                },
            ));
        }
        stats.equivalences_verified =
            stats.equivalences_verified.checked_add(1).ok_or_else(|| {
                (
                    failed_alias.clone(),
                    CheckpointError::ArithmeticOverflow {
                        context: "counting verified checkpoint equivalences",
                    },
                )
            })?;
        report.record_equivalence_verified(CheckpointEquivalenceRecord::new(
            &equivalence.alias.name,
            &equivalence.owner.name,
        ));
    }
    Ok(())
}

/// Accounts for one successful source read used by equivalence preflight.
fn observe_equivalence_read(
    tensor: &str,
    payload_bytes: usize,
    stats: &mut ExecutionStats,
) -> Result<(), CheckpointError> {
    let payload_bytes =
        u64::try_from(payload_bytes).map_err(|_| CheckpointError::InvalidTensorPayload {
            tensor: tensor.into(),
            reason: "equivalence payload length does not fit in u64".into(),
        })?;
    stats.equivalence_source_reads = stats.equivalence_source_reads.checked_add(1).ok_or(
        CheckpointError::ArithmeticOverflow {
            context: "counting checkpoint equivalence source reads",
        },
    )?;
    stats.equivalence_source_payload_bytes = stats
        .equivalence_source_payload_bytes
        .checked_add(payload_bytes)
        .ok_or(CheckpointError::ArithmeticOverflow {
            context: "summing checkpoint equivalence source bytes",
        })?;
    stats.peak_host_staging_bytes = stats.peak_host_staging_bytes.max(payload_bytes);
    Ok(())
}

/// Error raised immediately before or while handing one tensor to its sink.
enum InstallError<E> {
    /// Executor accounting rejected the materialized tensor.
    Checkpoint(CheckpointError),
    /// The architecture-specific destination rejected the tensor.
    Sink(E),
}

/// Accounts for one override buffer and any fused source retained beside it.
///
/// Override dtypes may intentionally differ from the plan's floating-point
/// target dtype—for example, Burn's packed `QFloat` storage—so this boundary
/// validates the logical shape and physical byte count without imposing the
/// ordinary transform dtype. The pass-one bound remains conservative because a
/// valid packed override cannot be allowed to exceed it.
fn observe_override_peak(
    tensor: &PlannedTensor,
    data: &TensorData,
    shared_source: Option<&SourceTensor>,
    planned_peak: u64,
    stats: &mut ExecutionStats,
) -> Result<(), CheckpointError> {
    let expected_shape = tensor
        .shape
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension).map_err(|_| CheckpointError::InvalidTensorPayload {
                tensor: tensor.target.clone(),
                reason: format!("dimension {dimension} does not fit in usize"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if data.shape.as_slice() != expected_shape {
        return Err(CheckpointError::InvalidTensorPayload {
            tensor: tensor.target.clone(),
            reason: format!(
                "override shape {:?} differs from planned shape {:?}",
                data.shape, expected_shape
            ),
        });
    }
    let output =
        u64::try_from(data.bytes.len()).map_err(|_| CheckpointError::InvalidTensorPayload {
            tensor: tensor.target.clone(),
            reason: "override byte length does not fit in u64".into(),
        })?;
    let source = shared_source.map_or(Ok(0_u64), |payload| {
        u64::try_from(payload.byte_len()).map_err(|_| CheckpointError::InvalidTensorPayload {
            tensor: tensor.target.clone(),
            reason: "shared source byte length does not fit in u64".into(),
        })
    })?;
    let peak = source
        .checked_add(output)
        .ok_or(CheckpointError::ArithmeticOverflow {
            context: "recording override staging high-water mark",
        })?;
    stats.peak_host_staging_bytes = stats.peak_host_staging_bytes.max(peak);
    stats.largest_output_tensor_bytes = stats.largest_output_tensor_bytes.max(output);
    if peak > planned_peak {
        return Err(CheckpointError::MemoryEstimateExceeded {
            tensor: tensor.target.clone(),
            required: peak,
            planned: planned_peak,
        });
    }
    Ok(())
}

/// Transfers one final tensor and atomically advances its report counters.
fn install_tensor<K: TensorSink>(
    tensor: &PlannedTensor,
    data: TensorData,
    sink: &mut K,
    report: &mut LoadReport,
    stats: &mut ExecutionStats,
) -> Result<(), InstallError<K::Error>> {
    let delivered = u64::try_from(data.bytes.len()).map_err(|_| {
        InstallError::Checkpoint(CheckpointError::InvalidTensorPayload {
            tensor: tensor.target.clone(),
            reason: "materialized byte length does not fit in u64".into(),
        })
    })?;
    let tensors_applied = stats.tensors_applied.checked_add(1).ok_or_else(|| {
        InstallError::Checkpoint(CheckpointError::ArithmeticOverflow {
            context: "counting applied target tensors",
        })
    })?;
    let delivered_tensor_bytes = stats
        .delivered_tensor_bytes
        .checked_add(delivered)
        .ok_or_else(|| {
            InstallError::Checkpoint(CheckpointError::ArithmeticOverflow {
                context: "summing target bytes delivered to the sink",
            })
        })?;

    sink.write_tensor(&tensor.target, data)
        .map_err(InstallError::Sink)?;
    stats.tensors_applied = tensors_applied;
    stats.delivered_tensor_bytes = delivered_tensor_bytes;
    stats.largest_output_tensor_bytes = stats.largest_output_tensor_bytes.max(delivered);
    report.record_applied(tensor.target.clone());
    Ok(())
}

/// Immutable descriptor key used to identify outputs of one fused split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SharedSourceKey<'a> {
    name: &'a str,
    shard: &'a str,
    shape: &'a [u64],
    dtype: TensorDType,
    data_offsets: [u64; 2],
}

impl<'a> From<&'a PlannedSource> for SharedSourceKey<'a> {
    fn from(source: &'a PlannedSource) -> Self {
        Self {
            name: &source.name,
            shard: &source.shard,
            shape: &source.shape,
            dtype: source.dtype,
            data_offsets: source.data_offsets,
        }
    }
}

/// Returns the shared descriptor only for a well-formed unary slice.
fn shared_slice_key(tensor: &PlannedTensor) -> Option<SharedSourceKey<'_>> {
    if !matches!(&tensor.transform, PlannedTransform::Slice { .. }) {
        return None;
    }
    let [source] = tensor.sources.as_slice() else {
        return None;
    };
    Some(source.into())
}

/// Groups fused slices and orders primary work by physical checkpoint location.
///
/// [`LoadPlan::tensors`] stays sorted by target name because that ordering is part
/// of the deterministic mapping contract. Execution has a different concern: the
/// bounded local source owns only one shard handle, so alternating target names
/// from different shards causes needless close/open cycles. Sorting the independent
/// primary groups by their first source `(shard, byte offset)` makes unary model
/// loads monotonic in checkpoint storage while leaving aliases and report ordering
/// untouched.
///
/// A multi-source concatenation cannot be interleaved with another target without
/// retaining several incomplete output allocations. It is therefore scheduled by
/// its first logical source and remains atomic, preserving the pass-one memory
/// bound. Fused slices are a single group because all outputs depend on one shared
/// payload. Target name and original plan index are total-order tie breakers for
/// hand-authored plans whose source descriptors happen to compare equally.
fn execution_groups(tensors: &[PlannedTensor]) -> Vec<Vec<usize>> {
    let mut slices_by_source = BTreeMap::<SharedSourceKey<'_>, Vec<usize>>::new();
    for (index, tensor) in tensors.iter().enumerate() {
        if let Some(key) = shared_slice_key(tensor) {
            slices_by_source.entry(key).or_default().push(index);
        }
    }

    let mut emitted = BTreeSet::new();
    let mut groups = Vec::with_capacity(tensors.len());
    for (index, tensor) in tensors.iter().enumerate() {
        let Some(key) = shared_slice_key(tensor) else {
            groups.push(vec![index]);
            continue;
        };
        let Some(indices) = slices_by_source.get(&key) else {
            // `key` was inserted during the immutable collection pass above. Keep
            // this internal logic invariant non-panicking if the grouping code is
            // changed independently in the future.
            groups.push(vec![index]);
            continue;
        };
        if indices.len() == 1 {
            groups.push(vec![index]);
        } else if emitted.insert(key) {
            groups.push(indices.clone());
        }
    }
    groups.sort_by(|left, right| execution_group_order(tensors, left, right));
    groups
}

/// Compares two indivisible execution groups by their first physical source.
fn execution_group_order(
    tensors: &[PlannedTensor],
    left: &[usize],
    right: &[usize],
) -> std::cmp::Ordering {
    let left_index = left.first().copied();
    let right_index = right.first().copied();
    let left_tensor = left_index.and_then(|index| tensors.get(index));
    let right_tensor = right_index.and_then(|index| tensors.get(index));

    match (left_tensor, right_tensor) {
        (Some(left_tensor), Some(right_tensor)) => {
            physical_source_order(left_tensor.sources.first(), right_tensor.sources.first())
                .then_with(|| left_tensor.target.cmp(&right_tensor.target))
                .then_with(|| left_index.cmp(&right_index))
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => left_index.cmp(&right_index),
    }
}

/// Compares optional source descriptors in stable shard-and-offset order.
///
/// Missing descriptors sort after real sources. Runtime validation rejects them
/// before this scheduler is called, but retaining a total comparison here keeps
/// the helper defensive and avoids making scheduling itself a fallible phase.
fn physical_source_order(
    left: Option<&PlannedSource>,
    right: Option<&PlannedSource>,
) -> std::cmp::Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left
            .shard
            .cmp(&right.shard)
            .then_with(|| left.data_offsets[0].cmp(&right.data_offsets[0]))
            .then_with(|| left.data_offsets[1].cmp(&right.data_offsets[1]))
            .then_with(|| left.name.cmp(&right.name)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

/// Executes a load plan against its discovered local checkpoint manifest.
///
/// This is the production convenience entry point. Use [`execute_load_plan`] with a
/// custom source for tests or alternative content-addressed artifact stores.
pub fn execute_local_load_plan<K: TensorSink>(
    manifest: &CheckpointManifest,
    plan: &LoadPlan,
    sink: &mut K,
) -> ExecutionResult<ExecutionOutcome, K::Error> {
    let mut source = LocalShardSource::new(manifest);
    execute_load_plan(plan, &mut source, sink)
}

/// Builds a checkpoint-side failure with finalized partial diagnostics.
fn checkpoint_failure<S, E>(
    failed_target: Option<String>,
    error: CheckpointError,
    mut report: LoadReport,
    mut stats: ExecutionStats,
    source: &S,
) -> Box<ExecutionFailure<E>>
where
    S: TensorSource,
{
    report.finalize();
    stats.source = source.stats();
    Box::new(ExecutionFailure {
        failed_target,
        cause: ExecutionFailureCause::Checkpoint(error),
        report,
        stats,
    })
}

/// Builds a sink-side failure with finalized partial diagnostics.
fn sink_failure<S, E>(
    failed_target: Option<String>,
    error: E,
    mut report: LoadReport,
    mut stats: ExecutionStats,
    source: &S,
) -> Box<ExecutionFailure<E>>
where
    S: TensorSource,
{
    report.finalize();
    stats.source = source.stats();
    Box::new(ExecutionFailure {
        failed_target,
        cause: ExecutionFailureCause::Sink(error),
        report,
        stats,
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "boundary_tests.rs"]
mod boundary_tests;
