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

mod convert;
mod packed_install;
#[cfg(test)]
pub(crate) mod packed_sink_support;
mod permutation;
mod rounding;
mod schedule;
mod source;
mod storage_order;
mod tiles;
mod transform;

use std::{collections::BTreeMap, error::Error, fmt};

use burn::tensor::{DType, TensorData};
use burn::{
    module::{Module, Param, ParamId},
    tensor::{
        Tensor,
        backend::{Backend, DTypeUsage},
    },
};

pub use packed_install::{
    PackedInstall, PackedLayoutPolicy, install_packed_parameter, packed_layout_is_exact,
};
use schedule::execution_group_indices;
pub use source::{LocalShardSource, SourceStats, SourceTensor, TensorSource};
#[cfg(test)]
pub(super) use transform::dequantize_mxfp4;

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
    /// Packed data would not keep its exact blocks through the load mapper.
    ///
    /// Returned by [`install_packed_parameter`] under
    /// [`PackedLayoutPolicy::RequireExact`]: either the parameter's load mapper
    /// permutes axes and [`packed_layout_is_exact`] reports that the backend
    /// re-derives block scales after such a permutation, or the mapper changes
    /// values, dtype, or shape beyond an axis permutation.
    InexactPackedLayout,
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
            Self::InexactPackedLayout => write!(
                formatter,
                "the parameter's load mapper would not preserve the packed weight's exact blocks on this backend (a requantizing axis permutation, or value or shape arithmetic); install dequantized values or use a backend with exact packed layout operations"
            ),
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
///
/// Packed data passes through the parameter's mapper unchanged, so on a backend
/// whose quantized transpose requantizes (Burn Flex 0.21) a column-layout weight
/// no longer equals the artifact. [`install_packed_parameter`] detects that case
/// and can install the exact dequantized values instead.
///
/// # Storage order
///
/// The installed *values* always equal `mapper(data)`. Their *memory layout* is
/// chosen for the backend's matrix products: on Burn Flex without autodiff, a
/// dense rank-2 float matrix whose load mapper transposes (a column-layout
/// projection weight) is transposed once on the host so the parameter ends up
/// contiguous `[in, out]` instead of a strided view of the `[out, in]` buffer.
/// Flex's GEMM ran single-row (decode) projections about twice as fast from
/// contiguous storage; other backends keep the zero-copy view, which measured as
/// fast or faster there. Detecting a transposing mapper loads a `[2, 3]` probe
/// into a clone of the parameter; a mapper that panics on that shape is treated
/// as non-transposing (the default panic hook still prints its message, and a
/// `panic = "abort"` build aborts instead). The host transpose briefly holds a
/// second host copy of this one tensor.
pub fn install_float_parameter<B: Backend, const D: usize>(
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) -> std::result::Result<(), ParameterInstallError> {
    validate_parameter_install(parameter, &data)?;
    install_validated_parameter(parameter, data);
    Ok(())
}

/// Checks every [`install_float_parameter`] precondition without mutating anything.
///
/// The order is part of the contract: rank, autodiff packed rejection, initialized
/// QFloat rejection, checkpoint-facing shape, element kind, dense payload length,
/// and finally backend storage support. Packed installers call this first so a
/// rejected tensor never reaches a backend constructor; Burn 0.21 autodiff, for
/// example, panics instead of returning an error when asked to build QFloat data.
fn validate_parameter_install<B: Backend, const D: usize>(
    parameter: &Param<Tensor<B, D>>,
    data: &TensorData,
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

    let dtype = data.dtype;
    if !B::dtype_usage(&parameter.lazy_device(), dtype).contains(DTypeUsage::Storage) {
        return Err(ParameterInstallError::UnsupportedStorage {
            dtype: format!("{dtype:?}"),
        });
    }
    Ok(())
}

/// Installs data that already passed [`validate_parameter_install`].
///
/// The tensor is built at its checkpoint dtype on the parameter's device and loaded
/// through Burn's record path, so the parameter's mapper, device, and gradient
/// policy apply exactly as for any other record. Its physical storage order is
/// chosen by [`storage_order::stored_for_mapper`], which never changes values.
fn install_validated_parameter<B: Backend, const D: usize>(
    parameter: &mut Param<Tensor<B, D>>,
    data: TensorData,
) {
    let device = parameter.lazy_device();
    let tensor = storage_order::stored_for_mapper(parameter, data, &device);
    load_parameter_tensor(parameter, tensor);
}

/// Loads a checkpoint-facing tensor through the parameter's record path.
///
/// The parameter keeps its mapper, device, and gradient policy. The temporary
/// record clone only moves the existing parameter out of `&mut`; it shares the
/// incoming backend allocation and is dropped once the mapped value replaces it.
fn load_parameter_tensor<B: Backend, const D: usize>(
    parameter: &mut Param<Tensor<B, D>>,
    tensor: Tensor<B, D>,
) {
    let record = Param::initialized(ParamId::new(), tensor);
    let existing = std::mem::replace(parameter, record.clone());
    *parameter = existing.load_record(record);
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
    let expected = super::layout::dense_byte_len(shape, dtype.size())
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
    Checkpoint(
        /// Original checkpoint executor diagnostic.
        CheckpointError,
    ),
    /// Architecture/backend sink rejected a tensor or alias.
    Sink(
        /// Original architecture-owned destination error.
        E,
    ),
}

impl<E: fmt::Display> fmt::Display for ExecutionFailureCause<E> {
    /// Formats the checkpoint cause or identifies an architecture-owned sink failure.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Sink(error) => write!(formatter, "tensor sink failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for ExecutionFailureCause<E> {
    /// Exposes the underlying checkpoint or sink error for chained diagnostics.
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
    /// Formats the failed target when known while retaining the underlying cause.
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
    /// Exposes the categorized cause without discarding the partial execution report.
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
    read_override: R,
) -> ExecutionResult<ExecutionOutcome, K::Error>
where
    S: TensorSource,
    K: TensorSink,
    H: Fn(&K, &PlannedTensor) -> bool,
    R: FnMut(&mut K, &PlannedTensor) -> Result<Option<TensorData>, K::Error>,
{
    let mut report = plan.report.without_progress();
    let mut stats = ExecutionStats::default();

    // Phase 1: preflight, equivalence checks, and every storage-owning write.
    // Any failure releases the source exactly once, here.
    let primary = install_primary_tensors(
        plan,
        source,
        sink,
        has_override,
        read_override,
        &mut report,
        &mut stats,
    );
    if let Err((failed_target, cause)) = primary {
        let _ = source.close();
        return Err(execution_failure(
            failed_target,
            cause,
            report,
            stats,
            source,
        ));
    }
    // Phase 2: release source resources before any alias references an owner.
    if let Err(error) = source.close() {
        let cause = ExecutionFailureCause::Checkpoint(error);
        return Err(execution_failure(None, cause, report, stats, source));
    }
    // Phase 3: tie aliases to their now-installed storage owners.
    if let Err((failed_target, cause)) = install_aliases(plan, sink, &mut report, &mut stats) {
        return Err(execution_failure(
            failed_target,
            cause,
            report,
            stats,
            source,
        ));
    }

    report.finalize();
    stats.source = source.stats();
    Ok(ExecutionOutcome { report, stats })
}

/// A failed execution step: the target being processed, if any, and its cause.
type StepFailure<E> = (Option<String>, ExecutionFailureCause<E>);

/// Returns a mapper that attributes a checkpoint error to `target`.
fn checkpoint_step<E>(target: &str) -> impl FnOnce(CheckpointError) -> StepFailure<E> + '_ {
    move |error| {
        (
            Some(target.to_owned()),
            ExecutionFailureCause::Checkpoint(error),
        )
    }
}

/// Validates the plan, verifies equivalences, and installs every primary target.
///
/// Override ownership is frozen before any source or sink mutation, so a stateful
/// predicate cannot select a target during group preflight and change its answer
/// at installation time. Groups run in physical source order.
#[allow(clippy::too_many_arguments)]
fn install_primary_tensors<S, K, H, R>(
    plan: &LoadPlan,
    source: &mut S,
    sink: &mut K,
    has_override: H,
    mut read_override: R,
    report: &mut LoadReport,
    stats: &mut ExecutionStats,
) -> Result<(), StepFailure<K::Error>>
where
    S: TensorSource,
    K: TensorSink,
    H: Fn(&K, &PlannedTensor) -> bool,
    R: FnMut(&mut K, &PlannedTensor) -> Result<Option<TensorData>, K::Error>,
{
    transform::validate_plan(plan)
        .map_err(|error| (None, ExecutionFailureCause::Checkpoint(error)))?;
    verify_checkpoint_equivalences(plan, source, report, stats)
        .map_err(|(alias, error)| (Some(alias), ExecutionFailureCause::Checkpoint(error)))?;

    let overridden = plan
        .tensors
        .iter()
        .map(|tensor| has_override(sink, tensor))
        .collect::<Vec<_>>();
    let can_skip_source = certified_source_bypasses(plan, source, &overridden)?;
    let targets = TargetDisposition {
        overridden: &overridden,
        can_skip_source: &can_skip_source,
    };
    for group in execution_groups(&plan.tensors) {
        install_group(
            plan,
            &group,
            source,
            sink,
            &mut read_override,
            &targets,
            report,
            stats,
        )?;
    }
    Ok(())
}

/// Per-target override decisions frozen before the first sink mutation.
struct TargetDisposition<'a> {
    /// Whether the sink supplies the target's final data itself.
    overridden: &'a [bool],
    /// Whether the source certified every descriptor, so its payload is never read.
    can_skip_source: &'a [bool],
}

/// Decides, per target, whether an override may bypass its source payload.
///
/// Only overridden targets whose every source descriptor the source certifies
/// (see [`TensorSource::validate_tensor_descriptor`]) skip their reads; all
/// others take the strict read-validate-discard fallback.
fn certified_source_bypasses<S: TensorSource, E>(
    plan: &LoadPlan,
    source: &S,
    overridden: &[bool],
) -> Result<Vec<bool>, StepFailure<E>> {
    let mut can_skip_source = vec![false; plan.tensors.len()];
    for (index, tensor) in plan.tensors.iter().enumerate() {
        if !overridden[index] {
            continue;
        }
        let mut descriptors_certified = true;
        for planned_source in &tensor.sources {
            let certified = source
                .validate_tensor_descriptor(planned_source)
                .map_err(checkpoint_step(&tensor.target))?;
            descriptors_certified &= certified;
        }
        can_skip_source[index] = descriptors_certified;
    }
    Ok(can_skip_source)
}

/// Materializes and installs one indivisible execution group.
///
/// A multi-target group shares one fused source payload, read once by its
/// first target that still needs the source and held while every output is
/// materialized and handed off.
#[allow(clippy::too_many_arguments)]
fn install_group<S, K, R>(
    plan: &LoadPlan,
    group: &[usize],
    source: &mut S,
    sink: &mut K,
    read_override: &mut R,
    targets: &TargetDisposition<'_>,
    report: &mut LoadReport,
    stats: &mut ExecutionStats,
) -> Result<(), StepFailure<K::Error>>
where
    S: TensorSource,
    K: TensorSink,
    R: FnMut(&mut K, &PlannedTensor) -> Result<Option<TensorData>, K::Error>,
{
    let reader = group
        .iter()
        .copied()
        .find(|&index| !targets.can_skip_source[index]);
    let shared_payload = match (group.len() > 1, reader) {
        (true, Some(index)) => {
            let tensor = &plan.tensors[index];
            let planned_source = tensor.sources.first().ok_or_else(|| {
                checkpoint_step(&tensor.target)(CheckpointError::InvalidMapping {
                    message: format!(
                        "fused-split target `{}` has no source descriptor",
                        tensor.target
                    ),
                })
            })?;
            Some(
                transform::read_verified(source, planned_source, stats)
                    .map_err(checkpoint_step(&tensor.target))?,
            )
        }
        _ => None,
    };

    for &index in group {
        let tensor = &plan.tensors[index];
        let source_data = if targets.can_skip_source[index] {
            None
        } else {
            let materialized = match &shared_payload {
                Some(payload) => transform::materialize_shared_slice(tensor, payload, stats),
                None => transform::materialize(tensor, source, stats),
            };
            Some(materialized.map_err(checkpoint_step(&tensor.target))?)
        };
        let data = if targets.overridden[index] {
            // A non-certifying source reaches this branch only after its
            // ordinary materialized value has been fully validated. Drop it
            // before decoding the override so the fallback does not retain
            // two output buffers at once.
            drop(source_data);
            read_validated_override(
                plan,
                tensor,
                sink,
                read_override,
                shared_payload.as_ref(),
                stats,
            )?
        } else {
            source_data.ok_or_else(|| {
                checkpoint_step(&tensor.target)(CheckpointError::InvalidMapping {
                    message: format!(
                        "non-overridden target `{}` unexpectedly bypassed its source",
                        tensor.target
                    ),
                })
            })?
        };
        install_tensor(tensor, data, sink, report, stats).map_err(|error| {
            let cause = match error {
                InstallError::Checkpoint(error) => ExecutionFailureCause::Checkpoint(error),
                InstallError::Sink(error) => ExecutionFailureCause::Sink(error),
            };
            (Some(tensor.target.clone()), cause)
        })?;
    }
    Ok(())
}

/// Obtains an override's data and checks it against the plan and staging bound.
fn read_validated_override<K, R>(
    plan: &LoadPlan,
    tensor: &PlannedTensor,
    sink: &mut K,
    read_override: &mut R,
    shared_payload: Option<&SourceTensor>,
    stats: &mut ExecutionStats,
) -> Result<TensorData, StepFailure<K::Error>>
where
    K: TensorSink,
    R: FnMut(&mut K, &PlannedTensor) -> Result<Option<TensorData>, K::Error>,
{
    let data = match read_override(sink, tensor) {
        Ok(Some(data)) => data,
        Ok(None) => {
            return Err(checkpoint_step(&tensor.target)(
                CheckpointError::InvalidMapping {
                    message: format!(
                        "override predicate selected `{}` but its materializer returned no tensor",
                        tensor.target
                    ),
                },
            ));
        }
        Err(error) => {
            return Err((
                Some(tensor.target.clone()),
                ExecutionFailureCause::Sink(error),
            ));
        }
    };
    observe_override_peak(
        tensor,
        &data,
        shared_payload,
        plan.memory.peak_host_staging_bytes,
        stats,
    )
    .map_err(checkpoint_step(&tensor.target))?;
    Ok(data)
}

/// Ties every alias to its installed storage owner after sources are closed.
fn install_aliases<K: TensorSink>(
    plan: &LoadPlan,
    sink: &mut K,
    report: &mut LoadReport,
    stats: &mut ExecutionStats,
) -> Result<(), StepFailure<K::Error>> {
    for alias in &plan.aliases {
        let PlannedTransform::Alias { source_target } = &alias.transform else {
            return Err(checkpoint_step(&alias.target)(
                CheckpointError::InvalidMapping {
                    message: format!("entry `{}` in alias list is not an alias", alias.target),
                },
            ));
        };
        let aliases_applied = stats.aliases_applied.checked_add(1).ok_or_else(|| {
            checkpoint_step(&alias.target)(CheckpointError::ArithmeticOverflow {
                context: "counting applied aliases",
            })
        })?;
        sink.alias_tensor(&alias.target, source_target)
            .map_err(|error| {
                (
                    Some(alias.target.clone()),
                    ExecutionFailureCause::Sink(error),
                )
            })?;
        stats.aliases_applied = aliases_applied;
        report.record_applied(alias.target.clone());
    }
    Ok(())
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
    Checkpoint(
        /// Shape, dtype, byte-count, or staging-accounting diagnostic.
        CheckpointError,
    ),
    /// The architecture-specific destination rejected the tensor.
    Sink(
        /// Original parameter-installation error.
        E,
    ),
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
    /// Physical checkpoint tensor name shared by the split outputs.
    name: &'a str,
    /// Root-relative shard containing the source payload.
    shard: &'a str,
    /// Logical source dimensions before splitting.
    shape: &'a [u64],
    /// Source wire scalar representation.
    dtype: TensorDType,
    /// Half-open source payload range within the shard's data section.
    data_offsets: [u64; 2],
}

impl<'a> From<&'a PlannedSource> for SharedSourceKey<'a> {
    /// Borrows the complete physical descriptor used to group slices of one fused source.
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
///
/// One exception precedes the physical order: groups whose outputs total at
/// least [`LARGE_GROUP_BYTES`] are installed first, in plan order. Installation
/// order is also device allocation order, and Burn's CubeCL backends pack
/// tensors first-fit into sliced memory pages. Allocating the outsized
/// vocabulary matrices last, after per-layer tensors have filled the pages,
/// stranded several gigabytes: on Metal the strict F16 loads of Qwen3.8-27B and
/// Muse-Glimmer-30B settled 7.4 and 5.4 GB above the hoisted schedule (62.35
/// against 54.92 GB, 64.78 against 59.40 GB), with peaks 5.2 and 8.1 GB higher.
/// Each hoisted group costs at most one extra shard open, and checkpoints
/// without such a group keep the pure physical schedule; Nemotron-H,
/// Qwen3-30B-A3B, and FLAN-T5-XXL footprints were unchanged by either order.
///
/// Each distinct shared-source descriptor is numbered by first appearance, and
/// the verified [`execution_group_indices`] forms the groups: they partition
/// the plan, so every primary target is installed exactly once.
fn execution_groups(tensors: &[PlannedTensor]) -> Vec<Vec<usize>> {
    let mut class_of_source = BTreeMap::<SharedSourceKey<'_>, usize>::new();
    let classes = tensors
        .iter()
        .map(|tensor| {
            shared_slice_key(tensor).map(|key| {
                let next = class_of_source.len();
                *class_of_source.entry(key).or_insert(next)
            })
        })
        .collect::<Vec<_>>();
    let mut groups = execution_group_indices(&classes, class_of_source.len());
    groups.sort_by(|left, right| {
        let large = |group: &[usize]| group_output_bytes(tensors, group) >= LARGE_GROUP_BYTES;
        match (large(left), large(right)) {
            // Large groups keep plan order among themselves; the verified groups
            // are already emitted in plan order, so their leaders compare it.
            (true, true) => left.first().cmp(&right.first()),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => execution_group_order(tensors, left, right),
        }
    });
    groups
}

/// Total output bytes from which an execution group is installed before physical order.
///
/// The threshold separates vocabulary-sized matrices (2.5 to 2.8 GB in the
/// measured Qwen3.8 and Muse checkpoints) from per-layer tensors, none of
/// which reached 1 GB in any measured checkpoint (see [`execution_groups`]).
pub(crate) const LARGE_GROUP_BYTES: u64 = 1 << 30;

/// Total target storage of one execution group, saturating on overflow.
///
/// Shapes and dtypes were validated before scheduling; an unrepresentable
/// size cannot occur there and is treated as large, which is conservative.
fn group_output_bytes(tensors: &[PlannedTensor], group: &[usize]) -> u64 {
    group
        .iter()
        .filter_map(|&index| tensors.get(index))
        .map(|tensor| {
            tensor
                .dtype
                .tensor_storage_bytes(&tensor.shape)
                .unwrap_or(u64::MAX)
        })
        .fold(0_u64, u64::saturating_add)
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

/// Builds a failure with finalized partial diagnostics and source counters.
fn execution_failure<S, E>(
    failed_target: Option<String>,
    cause: ExecutionFailureCause<E>,
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
        cause,
        report,
        stats,
    })
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "runtime/boundary_tests.rs"]
mod boundary_tests;
