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
//! Alias operations run only after every storage-owning parameter succeeds. This
//! provides a simple transaction boundary: a failure report lists exactly the
//! target writes accepted by the sink, and no alias can reference a missing owner.

mod source;
mod transform;

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use burn::tensor::{DType, TensorData};
use burn::{
    module::{Module, Param},
    tensor::{
        Tensor,
        backend::{Backend, DTypeUsage},
    },
};

pub use source::{LocalShardSource, SourceStats, SourceTensor, TensorSource};

use super::{
    CheckpointError, CheckpointManifest, LoadPlan, LoadReport, PlannedSource, PlannedTensor,
    PlannedTransform, TensorDType,
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
    /// The checkpoint shape differs from the parameter's lazy saved shape.
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
}

impl fmt::Display for ParameterInstallError {
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
        }
    }
}

impl Error for ParameterInstallError {}

/// Installs checkpoint-shaped data into a lazy Burn floating-point parameter.
///
/// Burn's parameter mapper is semantically important: for example, a
/// `LinearLayout::Col` weight is serialized as `[out, in]`, then transposed to
/// `[in, out]` when loaded for matrix multiplication. Replacing the field with
/// `Param::from_tensor` would discard that mapper and produce a model with
/// incorrectly oriented projections. This helper validates against
/// [`Param::lazy_shape`], constructs the tensor at its checkpoint dtype, and
/// delegates to Burn's ordinary record-loading path so device, gradient, and
/// layout policies remain intact.
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
    let expected = parameter.lazy_shape().to_vec();
    if actual != expected {
        return Err(ParameterInstallError::Shape { expected, actual });
    }
    let is_quantized_float = matches!(data.dtype, DType::QFloat(_));
    if !data.dtype.is_float() && !is_quantized_float {
        return Err(ParameterInstallError::NonFloat {
            dtype: format!("{:?}", data.dtype),
        });
    }

    let device = parameter.lazy_device();
    let dtype = data.dtype;
    if is_quantized_float && !B::dtype_usage(&device, dtype).contains(DTypeUsage::Storage) {
        return Err(ParameterInstallError::UnsupportedStorage {
            dtype: format!("{dtype:?}"),
        });
    }
    let tensor = Tensor::<B, D>::from_data(data, (&device, dtype));
    let record = Param::from_tensor(tensor);
    let existing = std::mem::replace(parameter, record.clone());
    *parameter = existing.load_record(record);
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
    /// Successful source payload reads; fused-split groups count as one read.
    pub source_reads: u64,
    /// Total source payload bytes returned to transformation code.
    pub source_payload_bytes: u64,
    /// Sum of target bytes transferred into the sink.
    pub delivered_tensor_bytes: u64,
    /// Largest individual source payload held by transformation code.
    pub largest_source_payload_bytes: u64,
    /// Largest individual target buffer delivered to the sink.
    pub largest_output_tensor_bytes: u64,
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
/// Targets slicing the same fused source are grouped at that source's physical
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
    let mut report = plan.report.clone();
    report.applied.clear();
    let mut stats = ExecutionStats::default();

    if let Err(error) = transform::validate_plan(plan) {
        let _ = source.close();
        return Err(checkpoint_failure(None, error, report, stats, source));
    }

    for group in execution_groups(&plan.tensors) {
        let shared_payload = if group.len() > 1 {
            let tensor = &plan.tensors[group[0]];
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
            let materialized = match &shared_payload {
                Some(payload) => transform::materialize_shared_slice(tensor, payload, &mut stats),
                None => transform::materialize(tensor, source, &mut stats),
            };
            let data = match materialized {
                Ok(data) => data,
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

/// Error raised immediately before or while handing one tensor to its sink.
enum InstallError<E> {
    /// Executor accounting rejected the materialized tensor.
    Checkpoint(CheckpointError),
    /// The architecture-specific destination rejected the tensor.
    Sink(E),
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
