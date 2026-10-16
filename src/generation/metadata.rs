//! Host-side metadata helpers shared by the generation coordinators.
//!
//! Coordinators read a few small tensors back to the host: a prompt's mask and
//! positions (or a history's mask and token IDs) once per request, and the
//! selected token IDs with their log probabilities once per decode step. Each
//! of those is a single device-to-host transfer: [`read_grid`] concatenates a
//! mask with its aligned integer tensor, and [`read_token_selection`] packs IDs
//! with their scores. Everything here is `O(batch * sequence)` or smaller;
//! vocabulary-sized tensors never cross the device boundary. Allocation goes
//! through `try_reserve_exact`, so a request whose metadata cannot be allocated
//! fails with an error instead of aborting.
//!
//! Integer index arithmetic delegates to the Verus-verified functions in
//! `bookkeeping`; this module adds tensor readback and error context.

use burn::tensor::{
    Bool, DType, Element, ElementConversion, Int, Tensor, TensorData, backend::Backend,
};
use burn::tensor::{bf16, f16};

use crate::{
    Error, Result, architecture::tensor_bool_values, models::shared::checks::read_int_tensors,
};

use super::TokenSelection;
use super::bookkeeping::{PositionError, next_position_after, row_span};

/// Returns an empty vector with exactly `capacity` reserved slots.
pub(super) fn vec_with_capacity<T>(capacity: usize, label: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Generation(format!("failed to allocate {capacity} {label}")))?;
    Ok(values)
}

/// Returns `batch` empty per-row vectors.
pub(super) fn empty_rows<T>(batch: usize, label: &str) -> Result<Vec<Vec<T>>> {
    let mut rows = vec_with_capacity(batch, label)?;
    rows.resize_with(batch, Vec::new);
    Ok(rows)
}

/// Returns `batch * sequence`, the element count of a rectangular buffer.
pub(super) fn rectangle_size(batch: usize, sequence: usize, label: &str) -> Result<usize> {
    batch
        .checked_mul(sequence)
        .ok_or_else(|| Error::Generation(format!("{label} element count overflows usize")))
}

/// Copies a small integer tensor to the host as `i64` values.
pub(super) fn tensor_i64_values<B: Backend, const D: usize>(
    tensor: Tensor<B, D, Int>,
    label: &str,
) -> Result<Vec<i64>> {
    tensor
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to read {label} from backend: {error}"))
        })?
        .to_vec::<B::IntElem>()
        .map(|values| {
            values
                .into_iter()
                .map(|value| value.elem::<i64>())
                .collect()
        })
        .map_err(|error| {
            Error::BackendCapability(format!("failed to read {label} from backend: {error}"))
        })
}

/// Copies a small float tensor to the host as `f64`, whatever its runtime dtype.
///
/// Burn backends may execute F16/BF16 tensors even when their default
/// `FloatElem` is F32, so reading with that associated type would reject valid
/// mixed-precision output.
pub(super) fn tensor_f64_values<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    label: &str,
) -> Result<Vec<f64>> {
    let data = tensor.try_into_data().map_err(|error| {
        Error::BackendCapability(format!("failed to read {label} from backend: {error}"))
    })?;
    host_float_data_f64(data, label)
}

/// Converts floating host storage directly to F64 without a lower-precision stage.
///
/// F16, BF16, F32, and Flex32 storage have exact F64 representations, and F64
/// values keep their precision. Typed borrowed slices validate byte alignment
/// and length without allocating a second source buffer. Quantized or integer
/// payloads are not probabilities and fail contextually rather than being
/// reinterpreted or dequantized.
pub(super) fn host_float_data_f64(mut data: TensorData, label: &str) -> Result<Vec<f64>> {
    // Burn's Flex32 is F32 host storage with a reduced-precision compute policy.
    // Normalizing its tag is lossless and moves ownership without cloning bytes;
    // unlike `iter`, the checked typed slice below reports malformed storage.
    if data.dtype == DType::Flex32 {
        data.dtype = DType::F32;
    }
    match data.dtype {
        DType::F16 => float_slice_f64::<f16>(&data, label),
        DType::BF16 => float_slice_f64::<bf16>(&data, label),
        DType::F32 => float_slice_f64::<f32>(&data, label),
        DType::F64 => float_slice_f64::<f64>(&data, label),
        dtype => Err(Error::BackendCapability(format!(
            "failed to read {label} from backend: unsupported floating dtype {dtype:?}"
        ))),
    }
}

/// Borrows a validated concrete float representation and promotes its values.
fn float_slice_f64<E: Element>(data: &TensorData, label: &str) -> Result<Vec<f64>> {
    data.as_slice::<E>()
        .map(|values| values.iter().map(|&value| value.elem::<f64>()).collect())
        .map_err(|error| {
            Error::BackendCapability(format!("failed to read {label} from backend: {error}"))
        })
}

/// Host copy of a `[batch, sequence]` validity mask and an aligned integer tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HostGrid {
    /// Row-major validity flags; all `true` when no mask was supplied.
    pub(super) valid: Vec<bool>,
    /// Row-major integer values (positions or token IDs), when supplied.
    pub(super) values: Option<Vec<i64>>,
}

/// Reads a `[batch, sequence]` mask and an aligned integer tensor in one transfer.
///
/// The mask (as 0/1 integers) and the values are flattened and concatenated
/// into one integer readback, so a padded prompt with explicit positions, or
/// a masked history with token IDs, costs one device-to-host synchronization
/// instead of two. A missing mask means every slot is valid; with neither
/// tensor nothing is read. Both tensors must already have shape
/// `[batch, sequence]`.
///
/// # Errors
///
/// Returns [`Error::Generation`] when a supplied tensor does not hold
/// `batch * sequence` elements or the host buffers cannot be allocated, and
/// [`Error::BackendCapability`] when the backend cannot read the values.
pub(super) fn read_grid<B: Backend>(
    mask: Option<Tensor<B, 2, Bool>>,
    values: Option<Tensor<B, 2, Int>>,
    batch: usize,
    sequence: usize,
    label: &str,
) -> Result<HostGrid> {
    let count = rectangle_size(batch, sequence, label)?;
    for (name, dims) in [
        ("mask", mask.as_ref().map(Tensor::dims)),
        ("values", values.as_ref().map(Tensor::dims)),
    ] {
        if let Some(dims) = dims
            && dims != [batch, sequence]
        {
            return Err(Error::Generation(format!(
                "{label} {name} has shape {dims:?}, expected [{batch}, {sequence}]"
            )));
        }
    }
    let (mask, values) = match (mask, values) {
        // A mask alone is read directly as booleans.
        (Some(mask), None) => {
            return Ok(HostGrid {
                valid: tensor_bool_values(mask, label)?,
                values: None,
            });
        }
        pair => pair,
    };
    let has_mask = mask.is_some();
    let has_values = values.is_some();
    let flat = mask
        .map(|mask| mask.int().reshape([count]))
        .into_iter()
        .chain(values.map(|values| values.reshape([count])))
        .collect::<Vec<_>>();
    let read = read_int_tensors(flat, label)?;
    let (mask_slots, value_slots) = read.split_at(if has_mask { count } else { 0 });
    let mut valid = vec_with_capacity(count, label)?;
    if has_mask {
        valid.extend(mask_slots.iter().map(|&flag| flag != 0));
    } else {
        valid.resize(count, true);
    }
    Ok(HostGrid {
        valid,
        values: has_values.then(|| value_slots.to_vec()),
    })
}

/// Counts valid entries per row of a flattened `[batch, sequence]` mask.
///
/// Every row must contain at least one valid token: an empty prompt row has
/// no last position from which to continue.
pub(super) fn nonempty_row_lengths(
    valid: &[bool],
    batch: usize,
    sequence: usize,
    label: &str,
) -> Result<Vec<usize>> {
    let expected = rectangle_size(batch, sequence, label)?;
    if valid.len() != expected {
        return Err(Error::Generation(format!(
            "{label} mask contains {} entries, expected {expected}",
            valid.len()
        )));
    }
    let mut lengths = vec_with_capacity(batch, label)?;
    for row in 0..batch {
        let (start, end) = row_span(row, batch, sequence, 0, sequence);
        let length = valid[start..end].iter().filter(|&&value| value).count();
        if length == 0 {
            return Err(Error::Generation(format!(
                "{label} row {row} has no valid tokens"
            )));
        }
        lengths.push(length);
    }
    Ok(lengths)
}

/// Finds the explicit position that follows each row's last valid prompt position.
///
/// A coordinator feeds back at most `max_new_tokens - 1` generated tokens, so
/// the request is accepted when every position it will send fits Burn's
/// portable signed 32-bit index range (see `bookkeeping::next_position_after`).
pub(super) fn next_positions(
    positions: &[i64],
    valid: &[bool],
    batch: usize,
    sequence: usize,
    max_new_tokens: usize,
    label: &str,
) -> Result<Vec<i64>> {
    let expected = rectangle_size(batch, sequence, label)?;
    if positions.len() != expected || valid.len() != expected {
        return Err(Error::Generation(format!(
            "{label} metadata is not rectangular: {} positions and {} mask entries for {expected} slots",
            positions.len(),
            valid.len()
        )));
    }
    let mut next = vec_with_capacity(batch, label)?;
    for row in 0..batch {
        let (start, end) = row_span(row, batch, sequence, 0, sequence);
        let last = positions[start..end]
            .iter()
            .zip(&valid[start..end])
            .rev()
            .find_map(|(&position, &is_valid)| is_valid.then_some(position))
            .ok_or_else(|| Error::Generation(format!("{label} row {row} has no valid position")))?;
        let position = next_position_after(last, max_new_tokens).map_err(|error| match error {
            PositionError::Negative => Error::Generation(format!(
                "{label} row {row} has negative last valid position {last}"
            )),
            PositionError::ExceedsSignedIndex => Error::Generation(format!(
                "{label} row {row}: continuing position {last} for {max_new_tokens} new tokens exceeds Burn's portable signed index range"
            )),
        })?;
        next.push(position);
    }
    Ok(next)
}

/// Converts last-position model output `[batch, 1, vocabulary]` into `[batch, vocabulary]`.
pub(super) fn extract_last_logits<B: Backend>(
    logits: Tensor<B, 3>,
    batch: usize,
    vocabulary_size: usize,
    label: &str,
) -> Result<Tensor<B, 2>> {
    let dimensions = logits.dims();
    if dimensions != [batch, 1, vocabulary_size] {
        return Err(Error::shape(
            label,
            format!("[{batch}, 1, {vocabulary_size}]"),
            format!("{dimensions:?}"),
        ));
    }
    Ok(logits.squeeze_dim::<2>(1))
}

/// Largest integer `n` such that every integer in `0..=n` is exact in `dtype`.
fn exact_integer_limit(dtype: DType) -> u64 {
    match dtype {
        DType::F32 => 1 << f32::MANTISSA_DIGITS,
        DType::F64 => 1 << f64::MANTISSA_DIGITS,
        _ => 0,
    }
}

/// Whether token IDs below `vocabulary_size` survive a round trip through the
/// float dtypes involved in a packed selection readback.
fn can_pack_token_ids(id_dtype: DType, packed_dtype: DType, vocabulary_size: usize) -> bool {
    let limit = exact_integer_limit(id_dtype).min(exact_integer_limit(packed_dtype));
    (vocabulary_size as u64) <= limit
}

/// Reads one step's selected token IDs and log probabilities to the host.
///
/// When every token ID is exactly representable in the log-probability dtype
/// (vocabularies up to `2^24` for F32), the IDs are converted to that dtype
/// and concatenated with the log probabilities into one `[batch, 2]` tensor,
/// so a decode step needs a single device-to-host synchronization. Otherwise
/// the two vectors are read separately.
pub(super) fn read_token_selection<B: Backend>(
    selection: TokenSelection<B>,
    vocabulary_size: usize,
) -> Result<(Vec<u32>, Vec<f64>)> {
    let TokenSelection {
        token_ids,
        log_probabilities,
    } = selection;
    let [batch] = token_ids.dims();
    if log_probabilities.dims() != [batch] {
        return Err(Error::shape(
            "generation.selected_log_probabilities",
            format!("[{batch}]"),
            format!("{:?}", log_probabilities.dims()),
        ));
    }
    let packed_dtype = log_probabilities.dtype();
    let float_ids = token_ids.clone().float();
    let (ids, log_probabilities) =
        if can_pack_token_ids(float_ids.dtype(), packed_dtype, vocabulary_size) {
            let packed = Tensor::cat(
                vec![
                    float_ids.cast(packed_dtype).reshape([batch, 1]),
                    log_probabilities.reshape([batch, 1]),
                ],
                1,
            );
            let values = tensor_f64_values(packed, "selected tokens and log probabilities")?;
            let ids = values.iter().step_by(2).copied().collect::<Vec<_>>();
            let scores = values
                .iter()
                .skip(1)
                .step_by(2)
                .copied()
                .collect::<Vec<_>>();
            (ids, scores)
        } else {
            let ids = tensor_i64_values(token_ids, "selected token IDs")?
                .into_iter()
                .map(|id| id as f64)
                .collect::<Vec<_>>();
            (
                ids,
                tensor_f64_values(log_probabilities, "selected log probabilities")?,
            )
        };
    if ids.len() != batch || log_probabilities.len() != batch {
        return Err(Error::Generation(format!(
            "backend returned {} token IDs and {} log probabilities for batch size {batch}",
            ids.len(),
            log_probabilities.len()
        )));
    }
    let ids = ids
        .into_iter()
        .map(|id| checked_token_id(id, vocabulary_size))
        .collect::<Result<Vec<_>>>()?;
    Ok((ids, log_probabilities))
}

/// Converts a host token value to a vocabulary ID, rejecting fractions and out-of-range values.
fn checked_token_id(value: f64, vocabulary_size: usize) -> Result<u32> {
    if value.fract() != 0.0 || value < 0.0 || value >= vocabulary_size as f64 {
        return Err(Error::Generation(format!(
            "backend selected token {value} outside vocabulary size {vocabulary_size}"
        )));
    }
    u32::try_from(value as u64).map_err(|_| {
        Error::Generation(format!(
            "backend selected token {value} outside the u32 token-ID domain"
        ))
    })
}

#[cfg(test)]
#[path = "metadata_tests.rs"]
mod tests;
