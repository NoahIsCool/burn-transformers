//! Row-major tensor transformations for the streaming checkpoint executor.
//!
//! A rank-`n` row-major index has linear offset
//! `sum(i[k] * product(shape[k + 1..]))`. Slicing and concatenation therefore copy
//! contiguous `product(shape[axis + 1..])`-element inner blocks, while a matrix
//! transpose maps `(row, column)` to `(column, row)`. Transformations write directly
//! into their final output allocation; they never construct a second full-model or
//! full-shard representation.
//!
//! SafeTensors scalars are little-endian. Burn [`TensorData`] bytes use native
//! scalar representation, so direct copies are byte-swapped on big-endian hosts.
//! Floating-point casts decode the exact source scalar, round once to the target
//! format with ties to even, and write native-endian bits. Integer
//! reinterpretation is forbidden. The dtype pair is resolved once per tensor and
//! the element loop runs in a monomorphic bulk kernel (see the `convert`
//! module); outputs produced in order (identity casts and slices) are never
//! zero-filled, while transposes and concatenations write into one zeroed
//! allocation because their writes are not sequential.

use std::collections::{BTreeMap, BTreeSet};

use burn::tensor::{BoolStore, DType, TensorData};

use super::{
    ExecutionStats, TensorSource,
    convert::{
        ConvertInPlace, GatherIntervals, ScalarKernel, ScatterIntervals, TransposeInto,
        validate_cast, with_scalar_conversion,
    },
};
use crate::checkpoint::layout::{
    AxisViewError, ConcatError, SlicePlanError, axis_elements, axis_view, check_slice_plan,
    element_count, extend_concatenation, range_within, slice_block,
};
use crate::checkpoint::mxfp4::{
    Mxfp4Geometry, Mxfp4Table, decode_transposed, decoded_shape, mxfp4_geometry,
};
use crate::checkpoint::{
    CheckpointEquivalenceRecord, CheckpointError, LoadPlan, PlannedSource, PlannedTensor,
    PlannedTensorEquivalence, PlannedTransform, Result, SourceTensor, TensorDType,
};

/// Validates all mutable public plan fields before the first sink mutation.
pub(super) fn validate_plan(plan: &LoadPlan) -> Result<()> {
    validate_equivalences(plan)?;
    let mut targets = BTreeSet::new();
    let mut primary = BTreeMap::new();
    for tensor in &plan.tensors {
        if tensor.target.is_empty() {
            return invalid_plan("primary target name cannot be empty");
        }
        if !targets.insert(tensor.target.as_str()) {
            return invalid_plan(format!("duplicate runtime target `{}`", tensor.target));
        }
        validate_primary(tensor)?;
        validate_memory_bound(tensor, plan.memory.peak_host_staging_bytes)?;
        primary.insert(tensor.target.as_str(), tensor);
    }

    for alias in &plan.aliases {
        if alias.target.is_empty() {
            return invalid_plan("alias target name cannot be empty");
        }
        if !targets.insert(alias.target.as_str()) {
            return invalid_plan(format!("duplicate runtime target `{}`", alias.target));
        }
        if !alias.sources.is_empty() {
            return invalid_plan(format!(
                "alias `{}` unexpectedly contains source payloads",
                alias.target
            ));
        }
        let PlannedTransform::Alias { source_target } = &alias.transform else {
            return invalid_plan(format!(
                "entry `{}` in alias list is not an alias",
                alias.target
            ));
        };
        let source = primary.get(source_target.as_str()).ok_or_else(|| {
            CheckpointError::InvalidMapping {
                message: format!(
                    "alias `{}` must reference a storage-owning primary target, got `{source_target}`",
                    alias.target
                ),
            }
        })?;
        if alias.shape != source.shape || alias.dtype != source.dtype {
            return invalid_plan(format!(
                "alias `{}` metadata differs from storage owner `{source_target}`",
                alias.target
            ));
        }
    }
    Ok(())
}

/// Materializes one primary target and updates executor-owned memory counters.
pub(super) fn materialize<S: TensorSource>(
    tensor: &PlannedTensor,
    source: &mut S,
    stats: &mut ExecutionStats,
) -> Result<TensorData> {
    match &tensor.transform {
        PlannedTransform::Identity => materialize_identity(tensor, source, stats),
        PlannedTransform::Transpose2d => materialize_transpose(tensor, source, stats),
        PlannedTransform::Slice {
            axis,
            start,
            length,
        } => materialize_slice(tensor, *axis, *start, *length, source, stats),
        PlannedTransform::Concatenate { axis } => {
            materialize_concatenate(tensor, *axis, source, stats)
        }
        PlannedTransform::DequantizeMxfp4 => materialize_mxfp4(tensor, source, stats),
        PlannedTransform::Alias { .. } => invalid_plan(format!(
            "alias `{}` occurred in the primary tensor list",
            tensor.target
        )),
    }
}

/// Checks source/target ranks, dimensions, dtypes, and transform arity.
fn validate_primary(tensor: &PlannedTensor) -> Result<()> {
    let _ = burn_dtype(tensor.dtype)?;
    let _ = byte_len(&tensor.shape, tensor.dtype, &tensor.target)?;
    let decoded = matches!(tensor.transform, PlannedTransform::DequantizeMxfp4);
    for source in &tensor.sources {
        validate_source_descriptor(source)?;
        if !decoded {
            validate_cast(source.dtype, tensor.dtype)?;
        }
    }

    match &tensor.transform {
        PlannedTransform::Identity => {
            let source = exactly_one_source(tensor)?;
            if tensor.shape != source.shape {
                return invalid_plan(format!(
                    "identity target `{}` shape differs from source `{}`",
                    tensor.target, source.name
                ));
            }
        }
        PlannedTransform::Transpose2d => {
            let source = exactly_one_source(tensor)?;
            if source.shape.len() != 2
                || tensor.shape.as_slice() != [source.shape[1], source.shape[0]]
            {
                return invalid_plan(format!(
                    "transpose target `{}` does not reverse a rank-two source shape",
                    tensor.target
                ));
            }
        }
        PlannedTransform::Slice {
            axis,
            start,
            length,
        } => {
            let source = exactly_one_source(tensor)?;
            // Verified: `Ok` exactly for an in-range slice whose target shape
            // is the source shape with the axis replaced by the length.
            match check_slice_plan(&source.shape, &tensor.shape, *axis, *start, *length) {
                Ok(()) => {}
                Err(SlicePlanError::AxisOutOfBounds) => {
                    return Err(CheckpointError::InvalidMapping {
                        message: format!(
                            "slice axis {axis} is out of bounds for source `{}`",
                            source.name
                        ),
                    });
                }
                Err(SlicePlanError::EndOverflow) => {
                    return Err(CheckpointError::ArithmeticOverflow {
                        context: "computing runtime slice end",
                    });
                }
                Err(SlicePlanError::PastAxis { end }) => {
                    return invalid_plan(format!(
                        "slice target `{}` ends at {end}, source axis has length {}",
                        tensor.target, source.shape[*axis]
                    ));
                }
                Err(SlicePlanError::ShapeMismatch) => {
                    let mut expected = source.shape.clone();
                    expected[*axis] = *length;
                    return invalid_plan(format!(
                        "slice target `{}` shape {:?} should be {expected:?}",
                        tensor.target, tensor.shape
                    ));
                }
            }
        }
        PlannedTransform::Concatenate { axis } => {
            validate_concatenation(tensor, *axis)?;
        }
        PlannedTransform::DequantizeMxfp4 => validate_mxfp4(tensor)?,
        PlannedTransform::Alias { .. } => {
            return invalid_plan(format!(
                "alias `{}` occurred in the primary tensor list",
                tensor.target
            ));
        }
    }
    Ok(())
}

/// Validates an MXFP4 work item: dtypes, the shape pair, and the decoded target.
fn validate_mxfp4(tensor: &PlannedTensor) -> Result<()> {
    let [blocks, scales] = tensor.sources.as_slice() else {
        return invalid_plan(format!(
            "MXFP4 target `{}` requires blocks and scales sources, found {}",
            tensor.target,
            tensor.sources.len()
        ));
    };
    if blocks.dtype != TensorDType::U8
        || !matches!(scales.dtype, TensorDType::U8 | TensorDType::F8_E8M0)
        || !matches!(
            tensor.dtype,
            TensorDType::F16 | TensorDType::BF16 | TensorDType::F32 | TensorDType::F64
        )
    {
        return invalid_plan(format!(
            "MXFP4 target `{}` decodes U8 blocks and U8/F8_E8M0 scales into F16, BF16, F32, or F64",
            tensor.target
        ));
    }
    let shape = decoded_shape(&blocks.shape, &scales.shape).map_err(|reason| {
        CheckpointError::InvalidMapping {
            message: format!("MXFP4 target `{}` sources {reason}", tensor.target),
        }
    })?;
    if tensor.shape != shape {
        return invalid_plan(format!(
            "MXFP4 target `{}` shape {:?} should be {shape:?}",
            tensor.target, tensor.shape
        ));
    }
    Ok(())
}

/// Reads an MXFP4 blocks/scales pair and decodes it into the planned dtype.
///
/// Both payloads and the output are resident together, which is the
/// operation's planned staging bound ([`PlannedTensor::staging_bytes`]).
fn materialize_mxfp4<S: TensorSource>(
    tensor: &PlannedTensor,
    source: &mut S,
    stats: &mut ExecutionStats,
) -> Result<TensorData> {
    let [blocks_source, scales_source] = tensor.sources.as_slice() else {
        return invalid_plan(format!(
            "MXFP4 target `{}` requires blocks and scales sources",
            tensor.target
        ));
    };
    let blocks = read_verified(source, blocks_source, stats)?;
    let scales = read_verified(source, scales_source, stats)?;
    let output = dequantize_mxfp4(
        &tensor.target,
        (blocks.bytes(), blocks.shape()),
        (scales.bytes(), scales.shape()),
        tensor.dtype,
    )?;
    let sources = blocks
        .byte_len()
        .checked_add(scales.byte_len())
        .ok_or_else(|| payload_bounds(&tensor.target))?;
    observe_peak(stats, sources, output.len(), &tensor.target)?;
    make_tensor_data(&tensor.target, tensor.dtype, &tensor.shape, output)
}

/// Decodes MXFP4 `(bytes, shape)` blocks and scales into native `dtype` bytes.
///
/// The result is `[experts, 32 · groups, rows]` in Transformers' layout (see
/// the `mxfp4` module). The exact binary64 value of each of the 4096
/// `(scale, code)` pairs is converted once to `dtype` with the loader's own
/// correctly rounded F64 conversion, then the blocks decode by table lookup.
///
/// # Errors
///
/// Returns [`CheckpointError::InvalidMapping`] for a malformed shape pair or
/// payload length, an unsupported target dtype, or an allocation failure.
pub(crate) fn dequantize_mxfp4(
    target: &str,
    blocks: (&[u8], &[u64]),
    scales: (&[u8], &[u64]),
    dtype: TensorDType,
) -> Result<Vec<u8>> {
    let geometry =
        mxfp4_geometry(blocks.1, scales.1).map_err(|reason| CheckpointError::InvalidMapping {
            message: format!("MXFP4 target `{target}` sources {reason}"),
        })?;
    if !matches!(
        dtype,
        TensorDType::F16 | TensorDType::BF16 | TensorDType::F32 | TensorDType::F64
    ) {
        return Err(CheckpointError::UnsupportedDTypeConversion {
            source_dtype: TensorDType::F64,
            target_dtype: dtype,
        });
    }
    with_scalar_conversion(
        TensorDType::F64,
        dtype,
        Mxfp4Kernel {
            target,
            blocks: blocks.0,
            scales: scales.0,
            geometry,
        },
    )?
}

/// Builds one dtype's decode table and runs the MXFP4 kernel with it.
struct Mxfp4Kernel<'a> {
    /// Target name for diagnostics.
    target: &'a str,
    /// Packed block bytes.
    blocks: &'a [u8],
    /// E8M0 scale codes.
    scales: &'a [u8],
    /// Validated extents.
    geometry: Mxfp4Geometry,
}

impl ScalarKernel for Mxfp4Kernel<'_> {
    /// Native target bytes of the decoded tensor.
    type Output = Result<Vec<u8>>;

    /// Encodes the 4096 exact values with `convert`, then decodes by lookup.
    ///
    /// `convert` maps little-endian binary64 scalars (`SOURCE = 8`) to the
    /// target dtype's native bytes.
    fn run<const SOURCE: usize, const TARGET: usize>(
        self,
        convert: impl Fn([u8; SOURCE]) -> [u8; TARGET] + Copy,
    ) -> Self::Output {
        let mut scalar = [0_u8; SOURCE];
        if SOURCE != core::mem::size_of::<u64>() {
            return invalid_plan(format!(
                "MXFP4 target `{}` encodes binary64 values, not {SOURCE}-byte scalars",
                self.target
            ));
        }
        let table = Mxfp4Table::new(|bits| {
            scalar.copy_from_slice(&bits.to_le_bytes());
            convert(scalar)
        });
        let bytes = self
            .geometry
            .values()
            .checked_mul(TARGET)
            .ok_or_else(|| payload_bounds(self.target))?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(bytes)
            .map_err(|_| CheckpointError::AllocationFailed {
                bytes: bytes as u64,
                context: "allocating a decoded MXFP4 tensor",
            })?;
        output.resize(bytes, 0);
        decode_transposed(self.blocks, self.scales, self.geometry, &table, &mut output).map_err(
            |reason| CheckpointError::InvalidTensorPayload {
                tensor: self.target.to_owned(),
                reason: format!("MXFP4 payloads {reason}"),
            },
        )?;
        Ok(output)
    }
}

/// Validates exact physical-equivalence work and its report/memory accounting.
fn validate_equivalences(plan: &LoadPlan) -> Result<()> {
    let mut aliases = BTreeSet::new();
    let mut records = Vec::with_capacity(plan.equivalences.len());
    let mut largest_pair = 0_u64;
    for equivalence in &plan.equivalences {
        validate_source_descriptor(&equivalence.alias)?;
        validate_source_descriptor(&equivalence.owner)?;
        if equivalence.alias.name == equivalence.owner.name {
            return equivalence_error(equivalence, "alias and owner names must differ");
        }
        if !aliases.insert(equivalence.alias.name.as_str()) {
            return equivalence_error(
                equivalence,
                "alias occurs in multiple equivalence work items",
            );
        }
        if equivalence.alias.shape != equivalence.owner.shape {
            return equivalence_error(equivalence, "alias and owner shapes differ");
        }
        if equivalence.alias.dtype != equivalence.owner.dtype {
            return equivalence_error(equivalence, "alias and owner dtypes differ");
        }
        let alias_bytes = source_byte_len(&equivalence.alias)?;
        let owner_bytes = source_byte_len(&equivalence.owner)?;
        if alias_bytes != owner_bytes {
            return equivalence_error(equivalence, "alias and owner payload lengths differ");
        }
        let pair =
            alias_bytes
                .checked_add(owner_bytes)
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "validating checkpoint equivalence staging bytes",
                })?;
        if pair > plan.memory.peak_host_staging_bytes {
            return Err(CheckpointError::MemoryEstimateExceeded {
                tensor: equivalence.alias.name.clone(),
                required: pair,
                planned: plan.memory.peak_host_staging_bytes,
            });
        }
        largest_pair = largest_pair.max(pair);
        records.push(CheckpointEquivalenceRecord::new(
            &equivalence.alias.name,
            &equivalence.owner.name,
        ));
    }
    records.sort();
    if plan.report.equivalence_planned != records {
        return invalid_plan(
            "planned checkpoint equivalence report does not match executable work items",
        );
    }
    if !plan.report.equivalence_verified.is_empty() {
        return invalid_plan("an unexecuted load plan cannot contain verified equivalences");
    }
    if plan.memory.largest_equivalence_pair_bytes != largest_pair {
        return invalid_plan(format!(
            "largest equivalence pair is {largest_pair} bytes, memory plan records {}",
            plan.memory.largest_equivalence_pair_bytes
        ));
    }
    Ok(())
}

/// Validates one immutable source descriptor carried by a public load plan.
fn validate_source_descriptor(source: &PlannedSource) -> Result<()> {
    if source.name.is_empty() || source.shard.is_empty() {
        return invalid_plan("source descriptor has an empty tensor name or shard");
    }
    let _ = burn_dtype(source.dtype)?;
    let described = source_byte_len(source)?;
    let expected = byte_len(&source.shape, source.dtype, &source.name)?;
    if described != expected {
        return invalid_plan(format!(
            "source `{}` offsets describe {described} bytes, expected {expected}",
            source.name
        ));
    }
    Ok(())
}

/// Returns the checked payload extent carried by one planned source.
fn source_byte_len(source: &PlannedSource) -> Result<u64> {
    source.data_offsets[1]
        .checked_sub(source.data_offsets[0])
        .ok_or_else(|| CheckpointError::InvalidMapping {
            message: format!("source `{}` has reversed data offsets", source.name),
        })
}

/// Builds one contextual equivalence-plan validation failure.
fn equivalence_error<T>(
    equivalence: &PlannedTensorEquivalence,
    reason: impl Into<String>,
) -> Result<T> {
    Err(CheckpointError::TensorEquivalenceMismatch {
        alias: equivalence.alias.name.clone(),
        owner: equivalence.owner.name.clone(),
        reason: reason.into(),
    })
}

/// Validates the shape equation for a concatenation.
fn validate_concatenation(tensor: &PlannedTensor, axis: usize) -> Result<()> {
    let first = tensor
        .sources
        .first()
        .ok_or_else(|| CheckpointError::InvalidMapping {
            message: format!("concatenation `{}` has no sources", tensor.target),
        })?;
    if axis >= first.shape.len() {
        return invalid_plan(format!(
            "concatenation axis {axis} is out of bounds for `{}`",
            tensor.target
        ));
    }
    let mut expected = first.shape.clone();
    expected[axis] = 0;
    for source in &tensor.sources {
        if source.dtype != first.dtype {
            return invalid_plan(format!(
                "concatenation `{}` mixes source dtypes",
                tensor.target
            ));
        }
        match extend_concatenation(&mut expected, &source.shape, axis) {
            Ok(()) => {}
            Err(ConcatError::Incompatible) => {
                return invalid_plan(format!(
                    "concatenation source `{}` has incompatible shape",
                    source.name
                ));
            }
            Err(ConcatError::Overflow) => {
                return Err(CheckpointError::ArithmeticOverflow {
                    context: "summing runtime concatenation dimensions",
                });
            }
        }
    }
    if tensor.shape != expected {
        return invalid_plan(format!(
            "concatenation target `{}` shape {:?} should be {expected:?}",
            tensor.target, tensor.shape
        ));
    }
    Ok(())
}

/// Ensures loader staging for one operation fits the pass-one conservative bound.
///
/// The requirement is [`PlannedTensor::staging_bytes`], the same formula the
/// planner exposes through [`LoadPlan::required_host_staging_bytes`].
fn validate_memory_bound(tensor: &PlannedTensor, planned: u64) -> Result<()> {
    let required = tensor.staging_bytes()?;
    if required > planned {
        return Err(CheckpointError::MemoryEstimateExceeded {
            tensor: tensor.target.clone(),
            required,
            planned,
        });
    }
    Ok(())
}

/// Directly transfers a tensor, optionally casting its floating-point dtype.
///
/// Equal-width transfers (equal dtypes and BF16 <-> F16) convert the source
/// allocation in place. Other casts gather the whole payload through one bulk
/// kernel into a reserved, never zero-filled output.
fn materialize_identity<S: TensorSource>(
    tensor: &PlannedTensor,
    source: &mut S,
    stats: &mut ExecutionStats,
) -> Result<TensorData> {
    let planned = exactly_one_source(tensor)?;
    let payload = read_verified(source, planned, stats)?;
    let (_, shape, source_dtype, mut bytes) = payload.into_parts();
    // Equal dtypes and BF16 <-> F16 keep the scalar width, including the
    // BF16 -> F16 conversion that dominates F16 Metal model loading.
    if source_dtype.converts_in_place_to(tensor.dtype) {
        observe_peak(stats, bytes.len(), 0, &tensor.target)?;
        convert_in_place(&tensor.target, &mut bytes, source_dtype, tensor.dtype)?;
        return make_tensor_data(&tensor.target, tensor.dtype, &shape, bytes);
    }

    let output_bytes = host_len(byte_len(&tensor.shape, tensor.dtype, &tensor.target)?)?;
    observe_peak(stats, bytes.len(), output_bytes, &tensor.target)?;
    let elements = host_len(checked_elements(&shape, &tensor.target)?)?;
    let output = with_scalar_conversion(
        source_dtype,
        tensor.dtype,
        GatherIntervals {
            tensor: &tensor.target,
            source: &bytes,
            intervals: core::iter::once(0..elements),
            elements,
        },
    )??;
    make_tensor_data(&tensor.target, tensor.dtype, &tensor.shape, output)
}

/// Converts equal-width scalars in place; a no-op for equal dtypes on little-endian hosts.
fn convert_in_place(
    tensor: &str,
    bytes: &mut [u8],
    source: TensorDType,
    target: TensorDType,
) -> Result<()> {
    if source == target && cfg!(target_endian = "little") {
        // SafeTensors and this host share byte order: already native storage.
        return Ok(());
    }
    with_scalar_conversion(source, target, ConvertInPlace { tensor, bytes })?
}

/// Transposes a row-major matrix while casting directly into its final buffer.
///
/// Every dtype pair, including equal dtypes, uses the same cache-tiled kernel;
/// the conversion is chosen once for the whole matrix.
fn materialize_transpose<S: TensorSource>(
    tensor: &PlannedTensor,
    source: &mut S,
    stats: &mut ExecutionStats,
) -> Result<TensorData> {
    let planned = exactly_one_source(tensor)?;
    let payload = read_verified(source, planned, stats)?;
    let (_, shape, source_dtype, bytes) = payload.into_parts();
    let [rows, columns] = shape.as_slice() else {
        return invalid_plan(format!(
            "transpose target `{}` requires a rank-two source",
            tensor.target
        ));
    };
    let rows = host_dimension(*rows, &tensor.target, "transpose row count")?;
    let columns = host_dimension(*columns, &tensor.target, "transpose column count")?;
    let mut output = allocate_tensor_bytes(&tensor.target, &tensor.shape, tensor.dtype)?;
    observe_peak(stats, bytes.len(), output.len(), &tensor.target)?;
    with_scalar_conversion(
        source_dtype,
        tensor.dtype,
        TransposeInto {
            tensor: &tensor.target,
            source: &bytes,
            target: &mut output,
            rows,
            columns,
        },
    )??;
    make_tensor_data(&tensor.target, tensor.dtype, &tensor.shape, output)
}

/// Extracts a consecutive section along an arbitrary row-major axis.
fn materialize_slice<S: TensorSource>(
    tensor: &PlannedTensor,
    axis: usize,
    start: u64,
    length: u64,
    source: &mut S,
    stats: &mut ExecutionStats,
) -> Result<TensorData> {
    let planned = exactly_one_source(tensor)?;
    let payload = read_verified(source, planned, stats)?;
    materialize_slice_payload(tensor, axis, start, length, &payload, stats)
}

/// Extracts one planned slice from a payload shared by a fused-split group.
pub(super) fn materialize_shared_slice(
    tensor: &PlannedTensor,
    payload: &SourceTensor,
    stats: &mut ExecutionStats,
) -> Result<TensorData> {
    let PlannedTransform::Slice {
        axis,
        start,
        length,
    } = tensor.transform
    else {
        return invalid_plan(format!(
            "shared source group contains non-slice target `{}`",
            tensor.target
        ));
    };
    let planned = exactly_one_source(tensor)?;
    validate_payload_descriptor(planned, payload)?;
    materialize_slice_payload(tensor, axis, start, length, payload, stats)
}

/// Copies an arbitrary-axis slice from one already verified source payload.
///
/// The source is viewed as `[outer, axis, inner]`; outer row `o` contributes one
/// contiguous block (see [`slice_block`]). Blocks arrive in output order, so the
/// gather appends them to a reserved buffer without zero-filling it first.
fn materialize_slice_payload(
    tensor: &PlannedTensor,
    axis: usize,
    start: u64,
    length: u64,
    payload: &SourceTensor,
    stats: &mut ExecutionStats,
) -> Result<TensorData> {
    let shape = payload.shape();
    let axis_view = AxisView::new(shape, axis, &tensor.target)?;
    let start = host_dimension(start, &tensor.target, "slice start")?;
    let length = host_dimension(length, &tensor.target, "slice length")?;
    if !range_within(start, length, axis_view.axis) {
        return invalid_plan(format!(
            "slice target `{}` exceeds source axis {axis} of length {}",
            tensor.target, axis_view.axis
        ));
    }
    let elements = axis_view.elements_with_axis(length, &tensor.target)?;
    let output_bytes = host_len(byte_len(&tensor.shape, tensor.dtype, &tensor.target)?)?;
    observe_peak(stats, payload.byte_len(), output_bytes, &tensor.target)?;
    let AxisView { outer, axis, inner } = axis_view;
    let output = with_scalar_conversion(
        payload.dtype(),
        tensor.dtype,
        GatherIntervals {
            tensor: &tensor.target,
            source: payload.bytes(),
            intervals: (0..outer)
                .map(|row| slice_block(row, outer, axis, inner, start, length).source),
            elements,
        },
    )??;
    make_tensor_data(&tensor.target, tensor.dtype, &tensor.shape, output)
}

/// Concatenates sources directly into one final row-major target allocation.
///
/// Sources are read one at a time, so the output is preallocated and each
/// source's outer rows are scattered to their strided target blocks: a source
/// with `a` rows along the axis, placed at offset `p` in a target of `T` rows,
/// writes outer row `o` to elements `[(o * T + p) * inner, (o * T + p + a) * inner)`.
fn materialize_concatenate<S: TensorSource>(
    tensor: &PlannedTensor,
    axis: usize,
    source: &mut S,
    stats: &mut ExecutionStats,
) -> Result<TensorData> {
    let mut output = allocate_tensor_bytes(&tensor.target, &tensor.shape, tensor.dtype)?;
    let output_len = output.len();
    let target_view = AxisView::new(&tensor.shape, axis, &tensor.target)?;
    let mut axis_offset = 0_usize;

    for planned in &tensor.sources {
        let payload = read_verified(source, planned, stats)?;
        observe_peak(stats, payload.byte_len(), output_len, &tensor.target)?;
        let source_axis = AxisView::new(payload.shape(), axis, &tensor.target)?.axis;
        if !range_within(axis_offset, source_axis, target_view.axis) {
            return invalid_plan(format!(
                "concatenation sources exceed target `{}` axis {axis}",
                tensor.target
            ));
        }
        let AxisView { outer, axis, inner } = target_view;
        // `slice_block` with exchanged roles: its `source` interval is this
        // input's destination and its `target` interval is the input itself.
        let blocks = (0..outer).map(|row| {
            let block = slice_block(row, outer, axis, inner, axis_offset, source_axis);
            (block.target, block.source)
        });
        with_scalar_conversion(
            payload.dtype(),
            tensor.dtype,
            ScatterIntervals {
                tensor: &tensor.target,
                source: payload.bytes(),
                target: &mut output,
                blocks,
            },
        )??;
        axis_offset += source_axis;
    }
    make_tensor_data(&tensor.target, tensor.dtype, &tensor.shape, output)
}

/// A row-major shape viewed as `[outer, axis, inner]` around one axis.
#[derive(Debug, Clone, Copy)]
struct AxisView {
    /// Product of the dimensions before the axis.
    outer: usize,
    /// Length of the axis itself.
    axis: usize,
    /// Product of the dimensions after the axis.
    inner: usize,
}

impl AxisView {
    /// Splits `shape` around `axis`, requiring `outer * axis * inner` to be host-addressable.
    ///
    /// These are exactly the preconditions of [`slice_block`]: every block
    /// offset is then bounded by the element count and cannot overflow. The
    /// verified [`axis_view`] computes the extents and proves them exact.
    fn new(shape: &[u64], axis: usize, tensor: &str) -> Result<Self> {
        let extents = axis_view(shape, axis).map_err(|error| match error {
            AxisViewError::AxisOutOfBounds => CheckpointError::InvalidMapping {
                message: format!("axis {axis} is out of bounds for `{tensor}` shape {shape:?}"),
            },
            AxisViewError::OuterOverflow | AxisViewError::InnerOverflow => {
                CheckpointError::InvalidTensorPayload {
                    tensor: tensor.to_owned(),
                    reason: "shape element count overflows u64".into(),
                }
            }
            AxisViewError::OuterTooLarge { value } | AxisViewError::InnerTooLarge { value } => {
                CheckpointError::AllocationFailed {
                    bytes: value,
                    context: "addressing a transformed tensor on this host",
                }
            }
            AxisViewError::AxisTooLarge { value } => CheckpointError::InvalidTensorPayload {
                tensor: tensor.to_owned(),
                reason: format!("axis length {value} does not fit in usize"),
            },
            AxisViewError::ElementsTooLarge => axis_elements_overflow(tensor),
        })?;
        Ok(Self {
            outer: extents.outer,
            axis: extents.axis,
            inner: extents.inner,
        })
    }

    /// Returns `outer * axis_len * inner`, checking `outer * axis_len` separately.
    fn elements_with_axis(self, axis_len: usize, tensor: &str) -> Result<usize> {
        axis_elements(self.outer, axis_len, self.inner)
            .ok_or_else(|| axis_elements_overflow(tensor))
    }
}

/// Reports an axis view whose element count does not fit `usize`.
fn axis_elements_overflow(tensor: &str) -> CheckpointError {
    CheckpointError::InvalidTensorPayload {
        tensor: tensor.to_owned(),
        reason: "axis view element count does not fit in usize".into(),
    }
}

/// Reads one payload and checks the source implementation's response.
pub(super) fn read_verified<S: TensorSource>(
    source: &mut S,
    planned: &PlannedSource,
    stats: &mut ExecutionStats,
) -> Result<SourceTensor> {
    let payload = source.read_tensor(planned)?;
    validate_payload_descriptor(planned, &payload)?;
    let payload_bytes =
        u64::try_from(payload.byte_len()).map_err(|_| CheckpointError::InvalidTensorPayload {
            tensor: planned.name.clone(),
            reason: "payload length does not fit in u64".into(),
        })?;
    stats.source_reads =
        stats
            .source_reads
            .checked_add(1)
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "counting source reads",
            })?;
    stats.source_payload_bytes = stats
        .source_payload_bytes
        .checked_add(payload_bytes)
        .ok_or(CheckpointError::ArithmeticOverflow {
            context: "summing source payload reads",
        })?;
    stats.largest_source_payload_bytes = stats.largest_source_payload_bytes.max(payload_bytes);
    Ok(payload)
}

/// Revalidates a source response before transformation or payload reuse.
fn validate_payload_descriptor(planned: &PlannedSource, payload: &SourceTensor) -> Result<()> {
    if payload.name() != planned.name
        || payload.shape() != planned.shape
        || payload.dtype() != planned.dtype
    {
        return Err(CheckpointError::InvalidTensorPayload {
            tensor: planned.name.clone(),
            reason: format!(
                "source returned name `{}`, shape {:?}, dtype {}; expected name `{}`, shape {:?}, dtype {}",
                payload.name(),
                payload.shape(),
                payload.dtype(),
                planned.name,
                planned.shape,
                planned.dtype
            ),
        });
    }
    Ok(())
}

/// Converts a tensor dtype into Burn's unquantized runtime dtype.
fn burn_dtype(dtype: TensorDType) -> Result<DType> {
    let dtype = match dtype {
        TensorDType::BOOL => DType::Bool(BoolStore::Native),
        TensorDType::U8 => DType::U8,
        TensorDType::I8 => DType::I8,
        TensorDType::U16 => DType::U16,
        TensorDType::I16 => DType::I16,
        TensorDType::F16 => DType::F16,
        TensorDType::BF16 => DType::BF16,
        TensorDType::U32 => DType::U32,
        TensorDType::I32 => DType::I32,
        TensorDType::F32 => DType::F32,
        TensorDType::F64 => DType::F64,
        TensorDType::U64 => DType::U64,
        TensorDType::I64 => DType::I64,
        TensorDType::F4
        | TensorDType::F6_E2M3
        | TensorDType::F6_E3M2
        | TensorDType::F8_E5M2
        | TensorDType::F8_E4M3
        | TensorDType::F8_E8M0
        | TensorDType::C64 => {
            return Err(CheckpointError::UnsupportedRuntimeDType { dtype });
        }
    };
    Ok(dtype)
}

/// Validates boolean values before Burn treats bytes as native `bool` storage.
fn validate_bool_bytes(tensor: &str, dtype: TensorDType, bytes: &[u8]) -> Result<()> {
    if dtype == TensorDType::BOOL
        && let Some((index, value)) = bytes
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value > 1)
    {
        return Err(CheckpointError::InvalidTensorPayload {
            tensor: tensor.to_owned(),
            reason: format!("boolean element {index} has invalid byte value {value}"),
        });
    }
    Ok(())
}

/// Constructs Burn tensor data only after all byte and boolean invariants hold.
fn make_tensor_data(
    tensor: &str,
    dtype: TensorDType,
    shape: &[u64],
    bytes: Vec<u8>,
) -> Result<TensorData> {
    let expected = byte_len(shape, dtype, tensor)?;
    let actual = u64::try_from(bytes.len()).map_err(|_| payload_bounds(tensor))?;
    if actual != expected {
        return Err(CheckpointError::InvalidTensorPayload {
            tensor: tensor.to_owned(),
            reason: format!("materialized {actual} bytes, expected {expected}"),
        });
    }
    validate_bool_bytes(tensor, dtype, &bytes)?;
    let burn_shape = shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| CheckpointError::InvalidTensorPayload {
                tensor: tensor.to_owned(),
                reason: format!("dimension {dimension} does not fit in usize"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TensorData::from_bytes_vec(
        bytes,
        burn_shape,
        burn_dtype(dtype)?,
    ))
}

/// Reserves and initializes exactly one target tensor output buffer.
fn allocate_tensor_bytes(tensor: &str, shape: &[u64], dtype: TensorDType) -> Result<Vec<u8>> {
    let bytes = byte_len(shape, dtype, tensor)?;
    let capacity = usize::try_from(bytes).map_err(|_| CheckpointError::AllocationFailed {
        bytes,
        context: "allocating a transformed target tensor",
    })?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| CheckpointError::AllocationFailed {
            bytes,
            context: "allocating a transformed target tensor",
        })?;
    output.resize(capacity, 0);
    Ok(output)
}

/// Records the source-plus-output staging high-water mark.
fn observe_peak(
    stats: &mut ExecutionStats,
    source_bytes: usize,
    output_bytes: usize,
    tensor: &str,
) -> Result<()> {
    let source = u64::try_from(source_bytes).map_err(|_| payload_bounds(tensor))?;
    let output = u64::try_from(output_bytes).map_err(|_| payload_bounds(tensor))?;
    let peak = source
        .checked_add(output)
        .ok_or(CheckpointError::ArithmeticOverflow {
            context: "recording loader staging high-water mark",
        })?;
    stats.peak_host_staging_bytes = stats.peak_host_staging_bytes.max(peak);
    stats.largest_output_tensor_bytes = stats.largest_output_tensor_bytes.max(output);
    Ok(())
}

/// Returns the one source required by unary transformations.
fn exactly_one_source(tensor: &PlannedTensor) -> Result<&PlannedSource> {
    if let [source] = tensor.sources.as_slice() {
        Ok(source)
    } else {
        invalid_plan(format!(
            "target `{}` requires exactly one source, found {}",
            tensor.target,
            tensor.sources.len()
        ))
    }
}

/// Computes `product(shape) * bits / 8` with checked shape multiplication.
///
/// A shape product that overflows `u64` is reported against `tensor`;
/// [`TensorDType::storage_bytes`] then fails only with
/// [`CheckpointError::ArithmeticOverflow`] (too many bits) or
/// [`CheckpointError::MisalignedTensorStorage`] (a partial final byte), which
/// are returned unchanged.
fn byte_len(shape: &[u64], dtype: TensorDType, tensor: &str) -> Result<u64> {
    dtype.storage_bytes(checked_elements(shape, tensor)?)
}

/// Computes a shape product, treating rank zero as one scalar.
fn checked_elements(shape: &[u64], tensor: &str) -> Result<u64> {
    element_count(shape).ok_or_else(|| CheckpointError::InvalidTensorPayload {
        tensor: tensor.to_owned(),
        reason: "shape element count overflows u64".into(),
    })
}

/// Converts a validated byte or element count to a host index.
fn host_len(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| CheckpointError::AllocationFailed {
        bytes: value,
        context: "addressing a transformed tensor on this host",
    })
}

/// Converts one planned dimension or offset to a host index with context.
fn host_dimension(value: u64, tensor: &str, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| CheckpointError::InvalidTensorPayload {
        tensor: tensor.to_owned(),
        reason: format!("{what} {value} does not fit in usize"),
    })
}

/// Creates a contextual out-of-bounds payload error.
fn payload_bounds(tensor: &str) -> CheckpointError {
    CheckpointError::InvalidTensorPayload {
        tensor: tensor.to_owned(),
        reason: "computed scalar range is outside the payload".into(),
    }
}

/// Creates a malformed-plan error without a temporary allocation at call sites.
fn invalid_plan<T>(message: impl Into<String>) -> Result<T> {
    Err(CheckpointError::InvalidMapping {
        message: message.into(),
    })
}

#[cfg(test)]
#[path = "transform_boundary_tests.rs"]
mod boundary_tests;
