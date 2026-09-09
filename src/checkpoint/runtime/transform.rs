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
//! format, and write native-endian bits. Integer reinterpretation is forbidden.

use std::collections::{BTreeMap, BTreeSet};

use burn::tensor::{BoolStore, DType, TensorData, bf16, f16};

use super::{ExecutionStats, TensorSource};
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
    for source in &tensor.sources {
        validate_source_descriptor(source)?;
        validate_conversion(source.dtype, tensor.dtype)?;
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
            let source_axis =
                source
                    .shape
                    .get(*axis)
                    .ok_or_else(|| CheckpointError::InvalidMapping {
                        message: format!(
                            "slice axis {axis} is out of bounds for source `{}`",
                            source.name
                        ),
                    })?;
            let end = start
                .checked_add(*length)
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "computing runtime slice end",
                })?;
            if end > *source_axis {
                return invalid_plan(format!(
                    "slice target `{}` ends at {end}, source axis has length {source_axis}",
                    tensor.target
                ));
            }
            let mut expected = source.shape.clone();
            expected[*axis] = *length;
            if tensor.shape != expected {
                return invalid_plan(format!(
                    "slice target `{}` shape {:?} should be {expected:?}",
                    tensor.target, tensor.shape
                ));
            }
        }
        PlannedTransform::Concatenate { axis } => {
            validate_concatenation(tensor, *axis)?;
        }
        PlannedTransform::Alias { .. } => {
            return invalid_plan(format!(
                "alias `{}` occurred in the primary tensor list",
                tensor.target
            ));
        }
    }
    Ok(())
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
        if source.shape.len() != expected.len()
            || source
                .shape
                .iter()
                .enumerate()
                .any(|(index, value)| index != axis && *value != expected[index])
        {
            return invalid_plan(format!(
                "concatenation source `{}` has incompatible shape",
                source.name
            ));
        }
        expected[axis] = expected[axis].checked_add(source.shape[axis]).ok_or(
            CheckpointError::ArithmeticOverflow {
                context: "summing runtime concatenation dimensions",
            },
        )?;
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
fn validate_memory_bound(tensor: &PlannedTensor, planned: u64) -> Result<()> {
    let output = byte_len(&tensor.shape, tensor.dtype, &tensor.target)?;
    let largest_source = tensor.sources.iter().try_fold(0_u64, |largest, source| {
        byte_len(&source.shape, source.dtype, &source.name).map(|bytes| largest.max(bytes))
    })?;
    let required = if matches!(tensor.transform, PlannedTransform::Identity)
        && tensor
            .sources
            .first()
            .is_some_and(|source| identity_reuses_source_buffer(source.dtype, tensor.dtype))
    {
        largest_source
    } else {
        largest_source
            .checked_add(output)
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "computing runtime staging bound",
            })?
    };
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
fn materialize_identity<S: TensorSource>(
    tensor: &PlannedTensor,
    source: &mut S,
    stats: &mut ExecutionStats,
) -> Result<TensorData> {
    let planned = exactly_one_source(tensor)?;
    let payload = read_verified(source, planned, stats)?;
    let (_, shape, source_dtype, mut bytes) = payload.into_parts();
    if source_dtype == tensor.dtype {
        normalize_little_endian(&mut bytes, source_dtype)?;
        observe_peak(stats, bytes.len(), 0, &tensor.target)?;
        return make_tensor_data(&tensor.target, tensor.dtype, &shape, bytes);
    }

    // Burn 0.21 converts equal-width TensorData storage in place. A BF16 value
    // is exactly representable as F32, so Burn's BF16 -> F32 -> F16 conversion
    // performs the same single F16 rounding as the scalar reference path. This
    // avoids allocating and zeroing a second multi-gigabyte buffer for the
    // checkpoint conversion that dominates F16 Metal model loading.
    if identity_reuses_source_buffer(source_dtype, tensor.dtype) {
        normalize_little_endian(&mut bytes, source_dtype)?;
        observe_peak(stats, bytes.len(), 0, &tensor.target)?;
        let data = make_tensor_data(&tensor.target, source_dtype, &shape, bytes)?;
        return Ok(data.convert_dtype(burn_dtype(tensor.dtype)?));
    }

    let mut output = allocate_tensor_bytes(&tensor.target, &tensor.shape, tensor.dtype)?;
    observe_peak(stats, bytes.len(), output.len(), &tensor.target)?;
    let elements = checked_elements(&shape, &tensor.target)?;
    copy_elements(
        &tensor.target,
        &bytes,
        source_dtype,
        0,
        &mut output,
        tensor.dtype,
        0,
        elements,
    )?;
    make_tensor_data(&tensor.target, tensor.dtype, &tensor.shape, output)
}

/// Returns whether one identity operation can reuse its source allocation.
///
/// Equal dtypes are already a byte-preserving transfer. Burn 0.21 additionally
/// guarantees an in-place `TensorData` conversion when source and target scalar
/// widths match. The optimized cast is intentionally restricted to BF16 -> F16,
/// the production Metal checkpoint path whose rounding semantics are covered
/// exhaustively; every other dtype pair retains the generic reference transform.
fn identity_reuses_source_buffer(source: TensorDType, target: TensorDType) -> bool {
    source == target || matches!((source, target), (TensorDType::BF16, TensorDType::F16))
}

/// Transposes a row-major matrix while casting directly into its final buffer.
fn materialize_transpose<S: TensorSource>(
    tensor: &PlannedTensor,
    source: &mut S,
    stats: &mut ExecutionStats,
) -> Result<TensorData> {
    let planned = exactly_one_source(tensor)?;
    let payload = read_verified(source, planned, stats)?;
    let (_, shape, source_dtype, bytes) = payload.into_parts();
    let rows = shape[0];
    let columns = shape[1];
    let mut output = allocate_tensor_bytes(&tensor.target, &tensor.shape, tensor.dtype)?;
    observe_peak(stats, bytes.len(), output.len(), &tensor.target)?;

    // SafeTensors and this host use the same scalar byte order, so an equal-dtype
    // transpose is a pure permutation of fixed-width byte strings. The tiled path
    // validates the complete layout once, then keeps a small source tile and the
    // corresponding target row segments hot in cache. Casts retain the scalar
    // reference loop below because they must round each value exactly once.
    #[cfg(target_endian = "little")]
    if source_dtype == tensor.dtype {
        transpose_little_endian_bytes(
            &tensor.target,
            &bytes,
            &mut output,
            rows,
            columns,
            source_dtype,
        )?;
        return make_tensor_data(&tensor.target, tensor.dtype, &tensor.shape, output);
    }

    for row in 0..rows {
        for column in 0..columns {
            let source_index = row
                .checked_mul(columns)
                .and_then(|base| base.checked_add(column))
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "computing transpose source index",
                })?;
            let target_index = column
                .checked_mul(rows)
                .and_then(|base| base.checked_add(row))
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "computing transpose target index",
                })?;
            copy_elements(
                &tensor.target,
                &bytes,
                source_dtype,
                source_index,
                &mut output,
                tensor.dtype,
                target_index,
                1,
            )?;
        }
    }
    if source_dtype == tensor.dtype {
        normalize_little_endian(&mut output, tensor.dtype)?;
    }
    make_tensor_data(&tensor.target, tensor.dtype, &tensor.shape, output)
}

/// Transposes equal-dtype little-endian scalars as opaque, fixed-width bytes.
///
/// A scalar at source coordinate `(r, c)` occupies byte interval
/// `((r * columns + c) * width)..`; its destination begins at
/// `((c * rows + r) * width)`. Blocking both axes bounds the active source and
/// destination cache footprint without allocating a scratch tile. Selecting a
/// constant-width worker also lets optimized builds reduce each scalar copy to
/// one fixed-size load and store while preserving every floating-point bit
/// pattern, including NaN payloads and signed zero.
#[cfg(target_endian = "little")]
fn transpose_little_endian_bytes(
    tensor: &str,
    source: &[u8],
    target: &mut [u8],
    rows: u64,
    columns: u64,
    dtype: TensorDType,
) -> Result<()> {
    let width =
        usize::try_from(scalar_bytes(dtype)?).map_err(|_| CheckpointError::ArithmeticOverflow {
            context: "converting transpose scalar width to usize",
        })?;
    let rows = usize::try_from(rows).map_err(|_| CheckpointError::InvalidTensorPayload {
        tensor: tensor.to_owned(),
        reason: "transpose row count does not fit in usize".into(),
    })?;
    let columns = usize::try_from(columns).map_err(|_| CheckpointError::InvalidTensorPayload {
        tensor: tensor.to_owned(),
        reason: "transpose column count does not fit in usize".into(),
    })?;
    let expected = rows
        .checked_mul(columns)
        .and_then(|elements| elements.checked_mul(width))
        .ok_or(CheckpointError::ArithmeticOverflow {
            context: "computing tiled transpose byte length",
        })?;
    if source.len() != expected || target.len() != expected {
        return Err(payload_bounds(tensor));
    }
    if expected == 0 {
        return Ok(());
    }

    match width {
        1 => transpose_scalar_bytes::<1>(source, target, rows, columns),
        2 => transpose_scalar_bytes::<2>(source, target, rows, columns),
        4 => transpose_scalar_bytes::<4>(source, target, rows, columns),
        8 => transpose_scalar_bytes::<8>(source, target, rows, columns),
        _ => {
            return Err(CheckpointError::InvalidTensorPayload {
                tensor: tensor.to_owned(),
                reason: format!("runtime scalar width {width} cannot be transposed"),
            });
        }
    }
    Ok(())
}

/// Applies one cache-tiled transpose for a compile-time scalar byte width.
///
/// The function is infallible after [`transpose_little_endian_bytes`] proves that
/// both buffers contain exactly `rows * columns` complete scalars. It uses only
/// disjoint mutable slices and adds no staging allocation, so the executor's
/// source-plus-output peak-memory bound remains exact.
#[cfg(target_endian = "little")]
fn transpose_scalar_bytes<const WIDTH: usize>(
    source: &[u8],
    target: &mut [u8],
    rows: usize,
    columns: usize,
) {
    const TILE_EDGE: usize = 32;

    let source_row_bytes = columns * WIDTH;
    let target_row_bytes = rows * WIDTH;
    let columns_per_tile = columns.min(TILE_EDGE);
    let target_tile_bytes = target_row_bytes * columns_per_tile;

    for (column_block, target_block) in target.chunks_mut(target_tile_bytes).enumerate() {
        let column_start = column_block * columns_per_tile;
        for row_start in (0..rows).step_by(TILE_EDGE) {
            let row_end = row_start.saturating_add(TILE_EDGE).min(rows);
            let source_start = row_start * source_row_bytes;
            let source_end = row_end * source_row_bytes;
            let source_rows = &source[source_start..source_end];
            let target_start = row_start * WIDTH;
            let target_end = row_end * WIDTH;

            for (local_column, target_row) in
                target_block.chunks_exact_mut(target_row_bytes).enumerate()
            {
                let source_column = (column_start + local_column) * WIDTH;
                let target_scalars = &mut target_row[target_start..target_end];
                for (source_row, target_scalar) in source_rows
                    .chunks_exact(source_row_bytes)
                    .zip(target_scalars.chunks_exact_mut(WIDTH))
                {
                    target_scalar
                        .copy_from_slice(&source_row[source_column..source_column + WIDTH]);
                }
            }
        }
    }
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
fn materialize_slice_payload(
    tensor: &PlannedTensor,
    axis: usize,
    start: u64,
    length: u64,
    payload: &SourceTensor,
    stats: &mut ExecutionStats,
) -> Result<TensorData> {
    let shape = payload.shape();
    let source_dtype = payload.dtype();
    let bytes = payload.bytes();
    let outer = checked_elements(&shape[..axis], &tensor.target)?;
    let inner = checked_elements(&shape[axis + 1..], &tensor.target)?;
    let source_axis = shape[axis];
    let block_elements = length
        .checked_mul(inner)
        .ok_or(CheckpointError::ArithmeticOverflow {
            context: "computing slice block length",
        })?;
    let mut output = allocate_tensor_bytes(&tensor.target, &tensor.shape, tensor.dtype)?;
    observe_peak(stats, bytes.len(), output.len(), &tensor.target)?;
    for outer_index in 0..outer {
        let source_element = outer_index
            .checked_mul(source_axis)
            .and_then(|base| base.checked_add(start))
            .and_then(|base| base.checked_mul(inner))
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "computing slice source offset",
            })?;
        let target_element =
            outer_index
                .checked_mul(block_elements)
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "computing slice target offset",
                })?;
        copy_elements(
            &tensor.target,
            bytes,
            source_dtype,
            source_element,
            &mut output,
            tensor.dtype,
            target_element,
            block_elements,
        )?;
    }
    if source_dtype == tensor.dtype {
        normalize_little_endian(&mut output, tensor.dtype)?;
    }
    make_tensor_data(&tensor.target, tensor.dtype, &tensor.shape, output)
}

/// Concatenates sources directly into one final row-major target allocation.
fn materialize_concatenate<S: TensorSource>(
    tensor: &PlannedTensor,
    axis: usize,
    source: &mut S,
    stats: &mut ExecutionStats,
) -> Result<TensorData> {
    let mut output = allocate_tensor_bytes(&tensor.target, &tensor.shape, tensor.dtype)?;
    let output_len = output.len();
    let outer = checked_elements(&tensor.shape[..axis], &tensor.target)?;
    let inner = checked_elements(&tensor.shape[axis + 1..], &tensor.target)?;
    let target_axis = tensor.shape[axis];
    let source_dtype = tensor.sources[0].dtype;
    let mut axis_offset = 0_u64;

    for planned in &tensor.sources {
        let payload = read_verified(source, planned, stats)?;
        let (_, shape, payload_dtype, bytes) = payload.into_parts();
        observe_peak(stats, bytes.len(), output_len, &tensor.target)?;
        let source_axis = shape[axis];
        let block_elements =
            source_axis
                .checked_mul(inner)
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "computing concatenation block length",
                })?;
        for outer_index in 0..outer {
            let source_element = outer_index.checked_mul(block_elements).ok_or(
                CheckpointError::ArithmeticOverflow {
                    context: "computing concatenation source offset",
                },
            )?;
            let target_element = outer_index
                .checked_mul(target_axis)
                .and_then(|base| base.checked_add(axis_offset))
                .and_then(|base| base.checked_mul(inner))
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "computing concatenation target offset",
                })?;
            copy_elements(
                &tensor.target,
                &bytes,
                payload_dtype,
                source_element,
                &mut output,
                tensor.dtype,
                target_element,
                block_elements,
            )?;
        }
        axis_offset =
            axis_offset
                .checked_add(source_axis)
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "advancing concatenation axis offset",
                })?;
    }
    if source_dtype == tensor.dtype {
        normalize_little_endian(&mut output, tensor.dtype)?;
    }
    make_tensor_data(&tensor.target, tensor.dtype, &tensor.shape, output)
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

/// Copies or numerically converts a contiguous scalar interval.
#[allow(clippy::too_many_arguments)]
fn copy_elements(
    tensor: &str,
    source: &[u8],
    source_dtype: TensorDType,
    source_element: u64,
    target: &mut [u8],
    target_dtype: TensorDType,
    target_element: u64,
    elements: u64,
) -> Result<()> {
    let source_width = scalar_bytes(source_dtype)?;
    let target_width = scalar_bytes(target_dtype)?;
    if source_dtype == target_dtype {
        let source_range = byte_range(source_element, elements, source_width, tensor)?;
        let target_range = byte_range(target_element, elements, target_width, tensor)?;
        let source_bytes = source
            .get(source_range)
            .ok_or_else(|| payload_bounds(tensor))?;
        let target_bytes = target
            .get_mut(target_range)
            .ok_or_else(|| payload_bounds(tensor))?;
        if source_bytes.len() != target_bytes.len() {
            return Err(payload_bounds(tensor));
        }
        target_bytes.copy_from_slice(source_bytes);
        return Ok(());
    }

    validate_conversion(source_dtype, target_dtype)?;
    for index in 0..elements {
        let source_index =
            source_element
                .checked_add(index)
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "advancing conversion source index",
                })?;
        let target_index =
            target_element
                .checked_add(index)
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "advancing conversion target index",
                })?;
        let source_range = byte_range(source_index, 1, source_width, tensor)?;
        let target_range = byte_range(target_index, 1, target_width, tensor)?;
        let source_bytes = source
            .get(source_range)
            .ok_or_else(|| payload_bounds(tensor))?;
        let target_bytes = target
            .get_mut(target_range)
            .ok_or_else(|| payload_bounds(tensor))?;
        convert_float_scalar(
            tensor,
            source_dtype,
            source_bytes,
            target_dtype,
            target_bytes,
        )?;
    }
    Ok(())
}

/// Decodes one real scalar and encodes it in the target format.
fn convert_float_scalar(
    tensor: &str,
    source_dtype: TensorDType,
    source: &[u8],
    target_dtype: TensorDType,
    target: &mut [u8],
) -> Result<()> {
    let value = match source_dtype {
        TensorDType::F16 => f16::from_bits(read_u16(tensor, source)?).to_f64(),
        TensorDType::BF16 => bf16::from_bits(read_u16(tensor, source)?).to_f64(),
        TensorDType::F32 => f32::from_bits(read_u32(tensor, source)?) as f64,
        TensorDType::F64 => f64::from_bits(read_u64(tensor, source)?),
        _ => {
            return Err(CheckpointError::UnsupportedDTypeConversion {
                source_dtype,
                target_dtype,
            });
        }
    };
    match target_dtype {
        TensorDType::F16 => {
            copy_scalar_bytes(tensor, target, f16::from_f64(value).to_bits().to_ne_bytes())
        }
        TensorDType::BF16 => copy_scalar_bytes(
            tensor,
            target,
            bf16::from_f64(value).to_bits().to_ne_bytes(),
        ),
        TensorDType::F32 => {
            copy_scalar_bytes(tensor, target, (value as f32).to_bits().to_ne_bytes())
        }
        TensorDType::F64 => copy_scalar_bytes(tensor, target, value.to_bits().to_ne_bytes()),
        _ => Err(CheckpointError::UnsupportedDTypeConversion {
            source_dtype,
            target_dtype,
        }),
    }
}

/// Copies one native-endian scalar from a fixed stack array into its output slot.
fn copy_scalar_bytes<const N: usize>(
    tensor: &str,
    target: &mut [u8],
    encoded: [u8; N],
) -> Result<()> {
    if target.len() != N {
        return Err(payload_bounds(tensor));
    }
    target.copy_from_slice(&encoded);
    Ok(())
}

/// Reads a little-endian sixteen-bit scalar without alignment assumptions.
fn read_u16(tensor: &str, bytes: &[u8]) -> Result<u16> {
    let array: [u8; 2] = bytes.try_into().map_err(|_| payload_bounds(tensor))?;
    Ok(u16::from_le_bytes(array))
}

/// Reads a little-endian thirty-two-bit scalar without alignment assumptions.
fn read_u32(tensor: &str, bytes: &[u8]) -> Result<u32> {
    let array: [u8; 4] = bytes.try_into().map_err(|_| payload_bounds(tensor))?;
    Ok(u32::from_le_bytes(array))
}

/// Reads a little-endian sixty-four-bit scalar without alignment assumptions.
fn read_u64(tensor: &str, bytes: &[u8]) -> Result<u64> {
    let array: [u8; 8] = bytes.try_into().map_err(|_| payload_bounds(tensor))?;
    Ok(u64::from_le_bytes(array))
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

/// Rejects reinterpretation and permits casts only among conventional real floats.
fn validate_conversion(source: TensorDType, target: TensorDType) -> Result<()> {
    if source == target {
        return burn_dtype(source).map(|_| ());
    }
    if matches!(
        source,
        TensorDType::F16 | TensorDType::BF16 | TensorDType::F32 | TensorDType::F64
    ) && matches!(
        target,
        TensorDType::F16 | TensorDType::BF16 | TensorDType::F32 | TensorDType::F64
    ) {
        Ok(())
    } else {
        Err(CheckpointError::UnsupportedDTypeConversion {
            source_dtype: source,
            target_dtype: target,
        })
    }
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

/// Converts a complete little-endian direct-copy buffer to native endian.
fn normalize_little_endian(bytes: &mut [u8], dtype: TensorDType) -> Result<()> {
    let width = scalar_bytes(dtype)?;
    let width = usize::try_from(width).map_err(|_| CheckpointError::ArithmeticOverflow {
        context: "converting scalar width to usize",
    })?;
    if !bytes.len().is_multiple_of(width) {
        return Err(CheckpointError::InvalidTensorPayload {
            tensor: "<native-endian conversion>".into(),
            reason: "payload is not a whole number of scalars".into(),
        });
    }
    #[cfg(target_endian = "big")]
    if width > 1 {
        for scalar in bytes.chunks_exact_mut(width) {
            scalar.reverse();
        }
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

/// Computes tensor bytes with checked shape multiplication.
fn byte_len(shape: &[u64], dtype: TensorDType, tensor: &str) -> Result<u64> {
    dtype
        .storage_bytes(checked_elements(shape, tensor)?)
        .map_err(|error| match error {
            CheckpointError::ArithmeticOverflow { .. }
            | CheckpointError::MisalignedTensorStorage { .. } => error,
            _ => CheckpointError::InvalidTensorPayload {
                tensor: tensor.to_owned(),
                reason: error.to_string(),
            },
        })
}

/// Computes a shape product, treating rank zero as one scalar.
fn checked_elements(shape: &[u64], tensor: &str) -> Result<u64> {
    shape.iter().try_fold(1_u64, |elements, dimension| {
        elements
            .checked_mul(*dimension)
            .ok_or_else(|| CheckpointError::InvalidTensorPayload {
                tensor: tensor.to_owned(),
                reason: "shape element count overflows u64".into(),
            })
    })
}

/// Returns the whole-byte scalar width supported by Burn.
fn scalar_bytes(dtype: TensorDType) -> Result<u64> {
    let _ = burn_dtype(dtype)?;
    Ok(dtype.bits() / 8)
}

/// Computes a checked byte range for a scalar interval.
fn byte_range(
    start: u64,
    elements: u64,
    width: u64,
    tensor: &str,
) -> Result<std::ops::Range<usize>> {
    let start = start
        .checked_mul(width)
        .ok_or(CheckpointError::ArithmeticOverflow {
            context: "computing scalar byte offset",
        })?;
    let length = elements
        .checked_mul(width)
        .ok_or(CheckpointError::ArithmeticOverflow {
            context: "computing scalar byte length",
        })?;
    let end = start
        .checked_add(length)
        .ok_or(CheckpointError::ArithmeticOverflow {
            context: "computing scalar byte range",
        })?;
    let start = usize::try_from(start).map_err(|_| payload_bounds(tensor))?;
    let end = usize::try_from(end).map_err(|_| payload_bounds(tensor))?;
    Ok(start..end)
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
mod boundary_tests {
    use super::*;
    use crate::checkpoint::{LoadReport, MemoryEstimate};

    /// Supplies immutable synthetic payloads while counting actual materializations.
    struct FixtureSource {
        payloads: Vec<SourceTensor>,
        reads: usize,
    }

    impl TensorSource for FixtureSource {
        /// Clones the exact requested wire payload without performing conversion.
        fn read_tensor(&mut self, planned: &PlannedSource) -> Result<SourceTensor> {
            self.reads += 1;
            self.payloads
                .iter()
                .find(|payload| payload.name() == planned.name)
                .cloned()
                .ok_or_else(|| CheckpointError::InvalidTensorPayload {
                    tensor: planned.name.clone(),
                    reason: "fixture payload absent".into(),
                })
        }
    }

    /// Describes one standalone tensor with a precisely sized byte interval.
    fn descriptor(name: &str, shape: &[u64], dtype: TensorDType) -> PlannedSource {
        PlannedSource {
            name: name.into(),
            shard: format!("{name}.safetensors"),
            shape: shape.to_vec(),
            dtype,
            data_offsets: [0, dtype.storage_bytes(shape.iter().product()).unwrap()],
        }
    }

    /// Creates one identity operation that individual tests mutate deliberately.
    fn identity() -> PlannedTensor {
        PlannedTensor {
            target: "target".into(),
            sources: vec![descriptor("weight", &[2, 3], TensorDType::F32)],
            transform: PlannedTransform::Identity,
            shape: vec![2, 3],
            dtype: TensorDType::F32,
        }
    }

    /// Wraps a primary operation without hiding validation behind fixture setup.
    fn plan(tensor: PlannedTensor) -> LoadPlan {
        LoadPlan {
            tensors: vec![tensor],
            aliases: Vec::new(),
            equivalences: Vec::new(),
            report: LoadReport::default(),
            memory: MemoryEstimate {
                peak_host_staging_bytes: 1024,
                ..MemoryEstimate::default()
            },
        }
    }

    /// Requires a stable contextual message, not merely any unrelated failure.
    fn assert_mapping_error(plan: &LoadPlan, expected: &str) {
        let error = validate_plan(plan).unwrap_err();
        assert!(
            matches!(error, CheckpointError::InvalidMapping { ref message }
            if message.contains(expected)),
            "expected {expected:?}, got {error}"
        );
    }

    /// Rejects every malformed descriptor before layout processing can trust it.
    #[test]
    fn public_source_descriptors_require_names_extents_and_supported_dtypes() {
        for (case, message) in [
            ("name", "empty tensor name or shard"),
            ("shard", "empty tensor name or shard"),
            ("reversed", "reversed data offsets"),
            ("length", "offsets describe 23 bytes, expected 24"),
        ] {
            let mut tensor = identity();
            let source = &mut tensor.sources[0];
            match case {
                "name" => source.name.clear(),
                "shard" => source.shard.clear(),
                "reversed" => source.data_offsets = [24, 0],
                "length" => source.data_offsets[1] = 23,
                _ => unreachable!(),
            }
            assert_mapping_error(&plan(tensor), message);
        }
        for dtype in [
            TensorDType::F4,
            TensorDType::F6_E2M3,
            TensorDType::F6_E3M2,
            TensorDType::F8_E5M2,
            TensorDType::F8_E4M3,
            TensorDType::F8_E8M0,
            TensorDType::C64,
        ] {
            let mut tensor = identity();
            tensor.sources[0].dtype = dtype;
            assert!(matches!(validate_plan(&plan(tensor)),
                Err(CheckpointError::UnsupportedRuntimeDType { dtype: actual }) if actual == dtype));
        }
        let mut tensor = identity();
        tensor.sources[0].shape = vec![u64::MAX, 2];
        assert!(
            matches!(validate_plan(&plan(tensor)), Err(CheckpointError::InvalidTensorPayload { tensor, reason })
            if tensor == "weight" && reason.contains("shape element count overflows"))
        );
    }

    /// Keeps primary uniqueness, unary arity, and matrix orientation independent.
    #[test]
    fn primary_operations_reject_duplicate_targets_and_invalid_unary_contracts() {
        for (case, message) in [
            ("duplicate", "duplicate runtime target `target`"),
            ("no_source", "requires exactly one source, found 0"),
            ("two_sources", "requires exactly one source, found 2"),
            ("identity_shape", "identity target `target` shape differs"),
            ("transpose_rank", "does not reverse a rank-two source shape"),
            (
                "transpose_shape",
                "does not reverse a rank-two source shape",
            ),
            (
                "alias",
                "alias `target` occurred in the primary tensor list",
            ),
        ] {
            let mut plan = plan(identity());
            let tensor = &mut plan.tensors[0];
            match case {
                "no_source" => tensor.sources.clear(),
                "two_sources" => tensor.sources.push(tensor.sources[0].clone()),
                "identity_shape" => tensor.shape = vec![6],
                "transpose_rank" => {
                    tensor.transform = PlannedTransform::Transpose2d;
                    tensor.sources[0].shape = vec![6];
                }
                "transpose_shape" => tensor.transform = PlannedTransform::Transpose2d,
                "alias" => {
                    tensor.transform = PlannedTransform::Alias {
                        source_target: "owner".into(),
                    }
                }
                "duplicate" => plan.tensors.push(plan.tensors[0].clone()),
                _ => unreachable!(),
            }
            assert_mapping_error(&plan, message);
        }
    }

    /// Requires aliases to share a primary owner's exact logical storage contract.
    #[test]
    fn public_alias_entries_cannot_smuggle_sources_or_incompatible_owners() {
        for (case, message) in [
            ("empty", "alias target name cannot be empty"),
            ("duplicate", "duplicate runtime target `target`"),
            ("payload", "unexpectedly contains source payloads"),
            ("operation", "entry `alias` in alias list is not an alias"),
            ("owner", "must reference a storage-owning primary target"),
            ("shape", "metadata differs from storage owner"),
            ("dtype", "metadata differs from storage owner"),
        ] {
            let mut plan = plan(identity());
            let mut alias = plan.tensors[0].clone();
            alias.target = "alias".into();
            alias.sources.clear();
            alias.transform = PlannedTransform::Alias {
                source_target: "target".into(),
            };
            match case {
                "empty" => alias.target.clear(),
                "duplicate" => alias.target = "target".into(),
                "payload" => alias.sources = plan.tensors[0].sources.clone(),
                "operation" => alias.transform = PlannedTransform::Identity,
                "owner" => {
                    alias.transform = PlannedTransform::Alias {
                        source_target: "absent".into(),
                    }
                }
                "shape" => alias.shape = vec![6],
                "dtype" => alias.dtype = TensorDType::F16,
                _ => unreachable!(),
            }
            plan.aliases.push(alias);
            assert_mapping_error(&plan, message);
        }
    }

    /// Validates slice intervals using checked addition and the full output shape.
    #[test]
    fn slice_contract_rejects_axis_extent_shape_and_end_overflow() {
        for (axis, start, length, shape, message) in [
            (2, 0, 1, vec![2, 1], "slice axis 2 is out of bounds"),
            (1, 2, 2, vec![2, 2], "ends at 4, source axis has length 3"),
            (1, 0, 1, vec![1, 2], "shape [1, 2] should be [2, 1]"),
        ] {
            let mut tensor = identity();
            tensor.transform = PlannedTransform::Slice {
                axis,
                start,
                length,
            };
            tensor.shape = shape;
            assert_mapping_error(&plan(tensor), message);
        }
        let mut tensor = identity();
        tensor.transform = PlannedTransform::Slice {
            axis: 1,
            start: u64::MAX,
            length: 1,
        };
        tensor.shape = vec![2, 1];
        assert!(matches!(
            validate_plan(&plan(tensor)),
            Err(CheckpointError::ArithmeticOverflow {
                context: "computing runtime slice end"
            })
        ));
    }

    /// Requires concatenation to conserve rank, non-axis dimensions, and dtype.
    #[test]
    fn concatenation_contract_rejects_each_distinct_shape_violation() {
        for (case, message) in [
            ("empty", "concatenation `target` has no sources"),
            ("axis", "concatenation axis 2 is out of bounds"),
            ("dtype", "mixes source dtypes"),
            ("rank", "source `second` has incompatible shape"),
            ("non_axis", "source `second` has incompatible shape"),
            ("target", "shape [6] should be [2, 3]"),
        ] {
            let mut tensor = identity();
            tensor.transform = PlannedTransform::Concatenate { axis: 1 };
            tensor.sources = vec![
                descriptor("first", &[2, 2], TensorDType::F32),
                descriptor("second", &[2, 1], TensorDType::F32),
            ];
            match case {
                "empty" => tensor.sources.clear(),
                "axis" => tensor.transform = PlannedTransform::Concatenate { axis: 2 },
                "dtype" => tensor.sources[1] = descriptor("second", &[2, 1], TensorDType::F16),
                "rank" => tensor.sources[1] = descriptor("second", &[2, 1, 1], TensorDType::F32),
                "non_axis" => tensor.sources[1] = descriptor("second", &[3, 1], TensorDType::F32),
                "target" => tensor.shape = vec![6],
                _ => unreachable!(),
            }
            assert_mapping_error(&plan(tensor), message);
        }

        // Empty leading axes make each payload legitimately zero bytes, while
        // their declared concatenation dimensions must still add without wrap.
        let mut tensor = identity();
        tensor.sources = vec![
            descriptor("first", &[0, u64::MAX], TensorDType::F32),
            descriptor("second", &[0, 1], TensorDType::F32),
        ];
        tensor.shape = vec![0, 0];
        tensor.transform = PlannedTransform::Concatenate { axis: 1 };
        assert!(matches!(
            validate_plan(&plan(tensor)),
            Err(CheckpointError::ArithmeticOverflow {
                context: "summing runtime concatenation dimensions"
            })
        ));
    }

    /// Separates equal-byte-count aliases from actual shape and identity equality.
    #[test]
    fn physical_equivalence_rejects_self_alias_and_equal_size_shape_changes() {
        for self_alias in [false, true] {
            let mut plan = plan(identity());
            let owner = plan.tensors[0].sources[0].clone();
            let alias = if self_alias {
                owner.clone()
            } else {
                descriptor("copy", &[3, 2], TensorDType::F32)
            };
            plan.equivalences
                .push(PlannedTensorEquivalence { alias, owner });
            let error = validate_plan(&plan).unwrap_err();
            assert!(
                matches!(error, CheckpointError::TensorEquivalenceMismatch { alias, owner, reason }
                if owner == "weight" && alias == if self_alias { "weight" } else { "copy" }
                && reason == if self_alias { "alias and owner names must differ" } else { "alias and owner shapes differ" })
            );
        }
    }

    /// Encodes exactly representable fixture values in a chosen source byte order.
    fn encode(values: &[f64], dtype: TensorDType, little_endian: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &value in values {
            let mut scalar = match dtype {
                TensorDType::F16 => f16::from_f64(value).to_bits().to_ne_bytes().to_vec(),
                TensorDType::BF16 => bf16::from_f64(value).to_bits().to_ne_bytes().to_vec(),
                TensorDType::F32 => (value as f32).to_ne_bytes().to_vec(),
                TensorDType::F64 => value.to_ne_bytes().to_vec(),
                _ => unreachable!(),
            };
            if little_endian && cfg!(target_endian = "big") {
                scalar.reverse();
            }
            bytes.extend(scalar);
        }
        bytes
    }

    /// Builds a wire payload independently of the production conversion helper.
    fn float_payload(
        name: &str,
        shape: &[u64],
        dtype: TensorDType,
        values: &[f64],
    ) -> SourceTensor {
        SourceTensor::new(name, shape.to_vec(), dtype, encode(values, dtype, true)).unwrap()
    }

    /// Checks every conventional float cast through four distinct layout paths.
    #[test]
    fn all_real_dtype_pairs_preserve_row_major_layout_and_signed_zero() {
        let dtypes = [
            TensorDType::F16,
            TensorDType::BF16,
            TensorDType::F32,
            TensorDType::F64,
        ];
        let values = [-2.0, -0.0, 0.0, 1.5, 7.0, f64::INFINITY];
        for source_dtype in dtypes {
            for target_dtype in dtypes {
                for operation in ["identity", "transpose", "slice", "concat"] {
                    let mut tensor = identity();
                    tensor.sources[0] = descriptor("weight", &[2, 3], source_dtype);
                    tensor.dtype = target_dtype;
                    let mut payloads =
                        vec![float_payload("weight", &[2, 3], source_dtype, &values)];
                    let expected = match operation {
                        "identity" => values.to_vec(),
                        "transpose" => {
                            tensor.transform = PlannedTransform::Transpose2d;
                            tensor.shape = vec![3, 2];
                            vec![-2.0, 1.5, -0.0, 7.0, 0.0, f64::INFINITY]
                        }
                        "slice" => {
                            tensor.transform = PlannedTransform::Slice {
                                axis: 1,
                                start: 1,
                                length: 1,
                            };
                            tensor.shape = vec![2, 1];
                            vec![-0.0, 7.0]
                        }
                        "concat" => {
                            tensor.transform = PlannedTransform::Concatenate { axis: 1 };
                            tensor.shape = vec![2, 4];
                            tensor
                                .sources
                                .push(descriptor("second", &[2, 1], source_dtype));
                            payloads.push(float_payload(
                                "second",
                                &[2, 1],
                                source_dtype,
                                &[100.0, 200.0],
                            ));
                            vec![-2.0, -0.0, 0.0, 100.0, 1.5, 7.0, f64::INFINITY, 200.0]
                        }
                        _ => unreachable!(),
                    };
                    validate_plan(&plan(tensor.clone())).unwrap();
                    let mut source = FixtureSource { payloads, reads: 0 };
                    let mut stats = ExecutionStats::default();
                    let data = materialize(&tensor, &mut source, &mut stats).unwrap();
                    assert_eq!(
                        data.as_bytes(),
                        encode(&expected, target_dtype, false),
                        "{source_dtype:?}->{target_dtype:?} {operation}"
                    );
                    assert_eq!(
                        data.shape.as_slice(),
                        tensor.shape.iter().map(|&n| n as usize).collect::<Vec<_>>()
                    );
                    assert_eq!(data.dtype, burn_dtype(target_dtype).unwrap());
                    assert_eq!(source.reads, if operation == "concat" { 2 } else { 1 });
                    assert_eq!(stats.source_reads as usize, source.reads);
                }
            }
        }
    }

    /// Validates shared source identity and proves reuse agrees with a fresh read.
    #[test]
    fn shared_slices_require_matching_payload_descriptors_and_slice_operations() {
        let mut tensor = identity();
        let payload = float_payload("weight", &[2, 3], TensorDType::F32, &[0.0; 6]);
        assert!(
            matches!(materialize_shared_slice(&tensor, &payload, &mut ExecutionStats::default()),
            Err(CheckpointError::InvalidMapping { message }) if message.contains("non-slice target `target`"))
        );
        tensor.transform = PlannedTransform::Slice {
            axis: 1,
            start: 1,
            length: 2,
        };
        tensor.shape = vec![2, 2];
        for payload in [
            float_payload("other", &[2, 3], TensorDType::F32, &[0.0; 6]),
            float_payload("weight", &[3, 2], TensorDType::F32, &[0.0; 6]),
            float_payload("weight", &[2, 3], TensorDType::F64, &[0.0; 6]),
        ] {
            assert!(
                matches!(materialize_shared_slice(&tensor, &payload, &mut ExecutionStats::default()),
                Err(CheckpointError::InvalidTensorPayload { tensor, reason })
                if tensor == "weight" && reason.contains("source returned"))
            );
        }
        let payload = float_payload(
            "weight",
            &[2, 3],
            TensorDType::F32,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let mut shared_stats = ExecutionStats::default();
        let shared = materialize_shared_slice(&tensor, &payload, &mut shared_stats).unwrap();
        let mut source = FixtureSource {
            payloads: vec![payload],
            reads: 0,
        };
        let mut read_stats = ExecutionStats::default();
        let fresh = materialize(&tensor, &mut source, &mut read_stats).unwrap();
        assert_eq!(shared, fresh);
        assert_eq!(shared.to_vec::<f32>().unwrap(), vec![2.0, 3.0, 5.0, 6.0]);
        assert_eq!(shared_stats.source_reads, 0);
        assert_eq!(read_stats.source_reads, 1);
        assert_eq!(shared_stats.peak_host_staging_bytes, 40);
        assert_eq!(read_stats.peak_host_staging_bytes, 40);
    }

    /// Rejects truncated contiguous spans before any destination byte is changed.
    #[test]
    fn copy_ranges_fail_contextually_for_equal_dtype_and_cast_paths() {
        for target_dtype in [TensorDType::F32, TensorDType::F16] {
            for source_short in [false, true] {
                let source = if source_short { vec![0; 3] } else { vec![0; 4] };
                let width = scalar_bytes(target_dtype).unwrap() as usize;
                let mut target = vec![0xa5; if source_short { width } else { width - 1 }];
                let before = target.clone();
                assert!(
                    matches!(copy_elements("bounded", &source, TensorDType::F32, 0,
                    &mut target, target_dtype, 0, 1), Err(CheckpointError::InvalidTensorPayload { tensor, reason })
                    if tensor == "bounded" && reason.contains("outside the payload"))
                );
                assert_eq!(target, before);
            }
        }
        let mut target = [0xa5; 4];
        copy_elements(
            "empty",
            &[],
            TensorDType::F32,
            0,
            &mut target,
            TensorDType::F32,
            1,
            0,
        )
        .unwrap();
        assert_eq!(target, [0xa5; 4]);
    }

    /// Proves each fixed-width decoder and target encoder checks exact slot sizes.
    #[test]
    fn scalar_conversion_rejects_truncated_slots_and_non_real_dtypes() {
        for (dtype, bytes) in [
            (TensorDType::F16, 1),
            (TensorDType::BF16, 3),
            (TensorDType::F32, 3),
            (TensorDType::F64, 7),
        ] {
            assert!(
                matches!(convert_float_scalar("scalar", dtype, &vec![0; bytes],
                TensorDType::F32, &mut [0; 4]), Err(CheckpointError::InvalidTensorPayload { tensor, .. }) if tensor == "scalar")
            );
        }
        assert!(matches!(
            convert_float_scalar(
                "scalar",
                TensorDType::F32,
                &1.0_f32.to_le_bytes(),
                TensorDType::F64,
                &mut [0; 7]
            ),
            Err(CheckpointError::InvalidTensorPayload { .. })
        ));
        for (source_dtype, target_dtype) in [
            (TensorDType::I32, TensorDType::F32),
            (TensorDType::F32, TensorDType::I32),
            (TensorDType::U32, TensorDType::I32),
            (TensorDType::BOOL, TensorDType::U8),
        ] {
            assert!(matches!(validate_conversion(source_dtype, target_dtype),
                Err(CheckpointError::UnsupportedDTypeConversion { source_dtype: source, target_dtype: target })
                if source == source_dtype && target == target_dtype));
        }
        for (source_dtype, target_dtype) in [
            (TensorDType::I32, TensorDType::F32),
            (TensorDType::F32, TensorDType::I32),
        ] {
            assert!(matches!(
                convert_float_scalar("scalar", source_dtype, &[0; 4], target_dtype, &mut [0; 4]),
                Err(CheckpointError::UnsupportedDTypeConversion { .. })
            ));
        }
    }

    /// Rejects incomplete scalar storage before Burn receives a byte-backed tensor.
    #[test]
    fn final_data_and_endian_boundaries_reject_malformed_storage() {
        assert!(
            matches!(make_tensor_data("short", TensorDType::F32, &[1], vec![0; 3]),
            Err(CheckpointError::InvalidTensorPayload { tensor, reason })
            if tensor == "short" && reason == "materialized 3 bytes, expected 4")
        );
        assert!(
            matches!(make_tensor_data("mask", TensorDType::BOOL, &[3], vec![0, 1, 255]),
            Err(CheckpointError::InvalidTensorPayload { tensor, reason })
            if tensor == "mask" && reason.contains("element 2 has invalid byte value 255"))
        );
        assert_eq!(
            make_tensor_data("bytes", TensorDType::U8, &[3], vec![0, 1, 255])
                .unwrap()
                .to_vec::<u8>()
                .unwrap(),
            vec![0, 1, 255]
        );
        assert!(
            matches!(normalize_little_endian(&mut [0; 3], TensorDType::F16),
            Err(CheckpointError::InvalidTensorPayload { tensor, reason })
            if tensor == "<native-endian conversion>" && reason.contains("whole number of scalars"))
        );
        assert_eq!(
            make_tensor_data(
                "scalar",
                TensorDType::F32,
                &[],
                1.0_f32.to_ne_bytes().to_vec()
            )
            .unwrap()
            .shape
            .as_slice(),
            Vec::<usize>::new()
        );
        assert!(
            make_tensor_data("empty", TensorDType::F32, &[0, 3], vec![])
                .unwrap()
                .as_bytes()
                .is_empty()
        );
    }

    /// Exercises offset, extent, and end addition without allocating large buffers.
    #[test]
    fn checked_byte_ranges_and_staging_counters_reject_wraparound() {
        for (start, elements, width, context) in [
            (u64::MAX, 0, 2, "computing scalar byte offset"),
            (0, u64::MAX, 2, "computing scalar byte length"),
            (u64::MAX, 1, 1, "computing scalar byte range"),
        ] {
            assert!(matches!(byte_range(start, elements, width, "range"),
                Err(CheckpointError::ArithmeticOverflow { context: actual }) if actual == context));
        }
        assert_eq!(byte_range(2, 3, 4, "range").unwrap(), 8..20);
        assert_eq!(checked_elements(&[], "scalar").unwrap(), 1);
        assert_eq!(checked_elements(&[0, u64::MAX], "empty").unwrap(), 0);
        assert!(matches!(
            byte_len(&[u64::MAX], TensorDType::F64, "huge"),
            Err(CheckpointError::ArithmeticOverflow {
                context: "computing tensor storage bits"
            })
        ));
        assert!(matches!(
            byte_len(&[1], TensorDType::F4, "partial"),
            Err(CheckpointError::MisalignedTensorStorage { .. })
        ));
        #[cfg(target_pointer_width = "64")]
        {
            let mut stats = ExecutionStats::default();
            assert!(matches!(
                observe_peak(&mut stats, usize::MAX, 1, "peak"),
                Err(CheckpointError::ArithmeticOverflow {
                    context: "recording loader staging high-water mark"
                })
            ));
            assert_eq!(stats, ExecutionStats::default());
        }
    }

    /// Rejects tiled transpose layouts before fixed-width workers index buffers.
    #[cfg(target_endian = "little")]
    #[test]
    fn tiled_transpose_rejects_wrong_buffer_lengths_and_dimension_overflow() {
        for source_short in [false, true] {
            let source = vec![0; if source_short { 23 } else { 24 }];
            let mut target = vec![0xa5; if source_short { 24 } else { 23 }];
            let before = target.clone();
            assert!(
                matches!(transpose_little_endian_bytes("matrix", &source, &mut target, 2, 3, TensorDType::F32),
                Err(CheckpointError::InvalidTensorPayload { tensor, .. }) if tensor == "matrix")
            );
            assert_eq!(target, before);
        }
        #[cfg(target_pointer_width = "64")]
        assert!(matches!(
            transpose_little_endian_bytes("matrix", &[], &mut [], u64::MAX, 2, TensorDType::U8),
            Err(CheckpointError::ArithmeticOverflow {
                context: "computing tiled transpose byte length"
            })
        ));
    }
}
