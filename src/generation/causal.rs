//! End-to-end cached generation for decoder-only language models.
//!
//! A causal model factorizes a continuation as
//! `p(y_1, ..., y_T | x) = product_t p(y_t | x, y_<t)`. The coordinator
//! performs the expensive prompt pass once or in caller-bounded cache-backed
//! chunks, then feeds only the previously selected token through
//! [`CausalLanguageModel::decode`]. Model caches remain preallocated and
//! architecture-owned; the only dynamically concatenated tensor is the compact
//! integer history used by repetition penalties.
//!
//! Batched rows may have different prompt and completion lengths. Every mask in
//! this module follows the crate-wide `true == valid` rule. Once a row reaches
//! EOS, a stop sequence, or its length limit, subsequent rectangular decode
//! calls carry a padding ID with a false mask entry, so that row neither changes
//! its cache nor influences token penalties.

use burn::tensor::{
    Bool, DType, Element, ElementConversion, Int, Tensor, TensorData, backend::Backend, bf16, f16,
};

use crate::{
    CausalLanguageModel, CausalLmInput, DecodeInput, Error, ForwardOptions, LogitsSelection,
    Result, TokenMask, architecture::bool_tensor_from_values, cache::CacheConfig,
};

use super::{FinishReason, GenerationConfig, GenerationTracker, LogitsProcessor, TokenSelection};

/// Complete controls and tensors for one cached causal-generation request.
#[derive(Clone, Debug)]
pub struct CausalGenerationRequest<B: Backend> {
    /// Prompt IDs, semantic mask, and optional explicit positions.
    pub prompt: CausalLmInput<B>,
    /// Preallocated architecture-cache policy used for this request.
    pub cache: CacheConfig,
    /// Sampling, penalty, and stopping controls.
    pub generation: GenerationConfig,
    /// Optional maximum number of physical prompt columns per prefill call.
    ///
    /// Chunking bounds temporary attention masks and score tensors for long
    /// prompts while preserving one shared decode state. `None` evaluates the
    /// complete prompt in one call. A value of zero is invalid.
    pub prefill_chunk_size: Option<usize>,
    /// Number of output categories in the model language head.
    pub vocabulary_size: usize,
}

/// One generated token delivered synchronously to a streaming callback.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationToken {
    /// Zero-based batch row that produced the token.
    pub sequence: usize,
    /// Zero-based position within this row's generated continuation.
    pub step: usize,
    /// Selected vocabulary ID.
    pub token_id: u32,
    /// Natural-log probability after penalties and sampling truncation.
    pub log_probability: f64,
    /// Terminal reason when this token completes the row.
    pub finish_reason: Option<FinishReason>,
}

/// Materialized continuation and terminal metadata for one batch row.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedSequence {
    /// Count of valid prompt tokens consumed for this row.
    pub prompt_length: usize,
    /// Generated vocabulary IDs, including a terminal EOS or stop suffix.
    pub token_ids: Vec<u32>,
    /// Processed log probability corresponding to each generated token.
    pub log_probabilities: Vec<f64>,
    /// Condition that terminated this row.
    pub finish_reason: FinishReason,
}

/// Completed batched causal-generation result.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalGenerationOutput {
    /// Results in the same row order as the prompt batch.
    pub sequences: Vec<GeneratedSequence>,
}

/// Generates a complete continuation without a streaming callback.
pub fn generate_causal<B, M>(
    model: &M,
    request: CausalGenerationRequest<B>,
) -> Result<CausalGenerationOutput>
where
    B: Backend,
    M: CausalLanguageModel<B>,
{
    generate_causal_streaming(model, request, |_| Ok(()))
}

/// Generates continuations and invokes `on_token` exactly once per emitted ID.
///
/// The callback runs after stopping state has been updated, so its event carries
/// a finish reason on the terminal token. Returning an error cancels the local
/// session immediately. Cache state is internal to this function and is dropped
/// on either success or failure; callers needing resumable sessions should use
/// the model's explicit `prefill` and `decode` methods directly.
pub fn generate_causal_streaming<B, M, F>(
    model: &M,
    request: CausalGenerationRequest<B>,
    mut on_token: F,
) -> Result<CausalGenerationOutput>
where
    B: Backend,
    M: CausalLanguageModel<B>,
    F: FnMut(GenerationToken) -> Result<()>,
{
    let PreparedRequest {
        batch,
        prompt_lengths,
        prompt_valid,
        mut next_positions,
    } = prepare_request(&request)?;
    let device = request.prompt.token_ids.device();
    let mut history = request.prompt.token_ids.clone();
    let mut history_mask = bool_tensor_from_values(prompt_valid.clone(), history.dims(), &device);
    let mut state = model.new_decode_state(request.cache, batch, &device)?;
    let options = ForwardOptions {
        logits: LogitsSelection::Last,
        ..Default::default()
    };
    let prompt_logits = prefill_prompt(
        model,
        request.prompt,
        &mut state,
        &options,
        &prompt_valid,
        request.prefill_chunk_size,
        request.vocabulary_size,
    )?;
    let mut logits = extract_last_logits(prompt_logits, batch, request.vocabulary_size)?;
    let processor =
        LogitsProcessor::<B>::new(request.generation.clone(), request.vocabulary_size, &device)?;
    let mut tracker =
        GenerationTracker::new(request.generation.clone(), request.vocabulary_size, batch)?;
    let mut log_probabilities = empty_rows::<f64>(batch, "generation log-probability")?;
    let padding = i64::from(request.generation.pad_token_id.unwrap_or(0));

    loop {
        let processed = processor.process_with_mask(
            logits,
            Some(history.clone()),
            Some(TokenMask::new(history_mask.clone())),
        )?;
        let TokenSelection {
            token_ids,
            log_probabilities: selected_log_probabilities,
        } = processor.select_with_log_probs(processed)?;
        let selected_ids = tensor_i64_values(token_ids.clone(), "selected token IDs")?;
        let selected_log_probabilities =
            tensor_f64_values(selected_log_probabilities, "selected log probabilities")?;
        if selected_ids.len() != batch || selected_log_probabilities.len() != batch {
            return Err(Error::Generation(format!(
                "backend returned {} token IDs and {} log probabilities for batch size {batch}",
                selected_ids.len(),
                selected_log_probabilities.len()
            )));
        }

        let mut still_active = Vec::new();
        still_active.try_reserve_exact(batch).map_err(|_| {
            Error::Generation(format!(
                "failed to allocate active-row mask for batch size {batch}"
            ))
        })?;
        for sequence in 0..batch {
            if tracker.finish_reason(sequence)?.is_some() {
                still_active.push(false);
                continue;
            }
            let token = selected_ids
                .get(sequence)
                .copied()
                .ok_or_else(|| Error::Generation("selected token batch became ragged".into()))?;
            let token = u32::try_from(token).map_err(|_| {
                Error::Generation(format!(
                    "backend selected token {token} outside the u32 token-ID domain"
                ))
            })?;
            let log_probability = selected_log_probabilities
                .get(sequence)
                .copied()
                .ok_or_else(|| {
                    Error::Generation("selected log-probability batch became ragged".into())
                })?;
            if !log_probability.is_finite() {
                return Err(Error::Generation(format!(
                    "sequence {sequence} produced non-finite selected log probability {log_probability}"
                )));
            }
            let step = tracker.generated(sequence)?.len();
            let finish_reason = tracker.push(sequence, token)?.cloned();
            log_probabilities
                .get_mut(sequence)
                .ok_or_else(|| Error::Generation("log-probability batch became ragged".into()))?
                .push(log_probability);
            on_token(GenerationToken {
                sequence,
                step,
                token_id: token,
                log_probability,
                finish_reason: finish_reason.clone(),
            })?;
            still_active.push(finish_reason.is_none());
        }

        if tracker.all_finished() {
            break;
        }

        let decode_mask = bool_tensor_from_values(still_active.clone(), [batch, 1], &device);
        let decode_ids = token_ids
            .reshape([batch, 1])
            .mask_fill(decode_mask.clone().bool_not(), padding);
        history = Tensor::cat(vec![history, decode_ids.clone()], 1);
        history_mask = Tensor::cat(vec![history_mask, decode_mask.clone()], 1);
        let position_ids = next_positions
            .as_ref()
            .map(|positions| position_column::<B>(positions, &device));
        let output = model.decode(
            DecodeInput {
                token_ids: decode_ids,
                position_ids,
                token_mask: Some(TokenMask::new(decode_mask)),
            },
            &mut state,
            &options,
        )?;
        if let Some(positions) = &mut next_positions {
            advance_positions(positions, &still_active)?;
        }
        logits = neutralize_inactive_logits(
            extract_last_logits(output.logits, batch, request.vocabulary_size)?,
            &still_active,
        )?;
    }

    finish_output(tracker, prompt_lengths, log_probabilities)
}

/// Prefills a prompt in bounded physical-column chunks and retains each row's latest logits.
///
/// Rectangular batches can place the final valid token in different chunks.
/// Every chunk is therefore evaluated for validation and cache semantics, but
/// an inactive row cannot overwrite logits produced by its most recent active
/// chunk. The generation coordinator owns and drops `state` on any later error,
/// so a multi-call prefill cannot expose a partially initialized session.
#[allow(clippy::too_many_arguments)]
fn prefill_prompt<B, M>(
    model: &M,
    prompt: CausalLmInput<B>,
    state: &mut M::State,
    options: &ForwardOptions,
    prompt_valid: &[bool],
    chunk_size: Option<usize>,
    vocabulary_size: usize,
) -> Result<Tensor<B, 3>>
where
    B: Backend,
    M: CausalLanguageModel<B>,
{
    let [batch, sequence] = prompt.token_ids.dims();
    let Some(chunk_size) = chunk_size else {
        return Ok(model.prefill(prompt, state, options)?.logits);
    };
    if chunk_size == 0 {
        return Err(Error::Generation(
            "prefill_chunk_size must be greater than zero when present".into(),
        ));
    }
    if chunk_size >= sequence {
        return Ok(model.prefill(prompt, state, options)?.logits);
    }
    let expected_mask_values = batch
        .checked_mul(sequence)
        .ok_or_else(|| Error::Generation("prompt mask element count overflows usize".into()))?;
    if prompt_valid.len() != expected_mask_values {
        return Err(Error::Generation(format!(
            "prompt mask contains {} entries, expected {expected_mask_values}",
            prompt_valid.len()
        )));
    }

    let device = prompt.token_ids.device();
    let mut latest_logits: Option<Tensor<B, 3>> = None;
    let mut start = 0;
    while start < sequence {
        let end = start.saturating_add(chunk_size).min(sequence);
        let token_ids = prompt.token_ids.clone().slice([0..batch, start..end]);
        let token_mask = prompt
            .token_mask
            .as_ref()
            .map(|mask| TokenMask::new(mask.as_tensor().clone().slice([0..batch, start..end])));
        let position_ids = prompt
            .position_ids
            .as_ref()
            .map(|positions| positions.clone().slice([0..batch, start..end]));
        let mut active = Vec::new();
        active.try_reserve_exact(batch).map_err(|_| {
            Error::Generation(format!(
                "failed to allocate active-row metadata for batch size {batch}"
            ))
        })?;
        for row in 0..batch {
            let row_start = row
                .checked_mul(sequence)
                .and_then(|offset| offset.checked_add(start))
                .ok_or_else(|| Error::Generation("prompt row offset overflows usize".into()))?;
            let row_end = row_start
                .checked_add(end - start)
                .ok_or_else(|| Error::Generation("prompt row end overflows usize".into()))?;
            let row_mask = prompt_valid.get(row_start..row_end).ok_or_else(|| {
                Error::Generation("prompt chunk mask falls outside its rectangular input".into())
            })?;
            active.push(row_mask.iter().any(|&valid| valid));
        }
        // Architecture prefill contracts reject a completely inactive row,
        // whereas decode explicitly treats it as a cache no-op. A ragged
        // physical chunk therefore uses the decode entry point; dense chunks
        // retain each architecture's optimized prefill implementation.
        let output = if active.iter().all(|&row_active| row_active) {
            model.prefill(
                CausalLmInput {
                    token_ids,
                    token_mask,
                    position_ids,
                },
                state,
                options,
            )?
        } else {
            model.decode(
                DecodeInput {
                    token_ids,
                    token_mask,
                    position_ids,
                },
                state,
                options,
            )?
        };
        if output.logits.dims() != [batch, 1, vocabulary_size] {
            return Err(Error::shape(
                "generation.model_logits",
                format!("[{batch}, 1, {vocabulary_size}] for prefill chunk {start}..{end}"),
                format!("{:?}", output.logits.dims()),
            ));
        }
        let active = bool_tensor_from_values(active, [batch], &device)
            .reshape([batch, 1, 1])
            .expand([batch, 1, vocabulary_size]);
        latest_logits = Some(match latest_logits {
            Some(previous) => previous.mask_where(active, output.logits),
            None => output.logits,
        });
        start = end;
    }

    latest_logits
        .ok_or_else(|| Error::Generation("a non-empty prompt produced no prefill chunks".into()))
}

/// Host metadata derived before any model/cache mutation.
struct PreparedRequest {
    /// Prompt batch width.
    batch: usize,
    /// Valid prompt-token count for each row.
    prompt_lengths: Vec<usize>,
    /// Flattened true-valid prompt mask.
    prompt_valid: Vec<bool>,
    /// Next explicit position per row, when the prompt supplies positions.
    next_positions: Option<Vec<i64>>,
}

/// Validates all coordinator-owned boundaries and derives compact host metadata.
fn prepare_request<B: Backend>(request: &CausalGenerationRequest<B>) -> Result<PreparedRequest> {
    request.generation.validate(request.vocabulary_size)?;
    if request.prefill_chunk_size == Some(0) {
        return Err(Error::Generation(
            "prefill_chunk_size must be greater than zero when present".into(),
        ));
    }
    let [batch, sequence] = request.prompt.token_ids.dims();
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "generation.prompt.token_ids",
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    if request.cache.sequence_count() != batch {
        return Err(Error::Generation(format!(
            "cache sequence_count {} does not match prompt batch size {batch}",
            request.cache.sequence_count()
        )));
    }

    let prompt_valid = match &request.prompt.token_mask {
        Some(mask) => {
            if mask.as_tensor().dims() != [batch, sequence] {
                return Err(Error::shape(
                    "generation.prompt.token_mask",
                    format!("[{batch}, {sequence}]"),
                    format!("{:?}", mask.as_tensor().dims()),
                ));
            }
            tensor_bool_values(mask.as_tensor().clone(), "prompt token mask")?
        }
        None => {
            let count = batch.checked_mul(sequence).ok_or_else(|| {
                Error::Generation("prompt mask element count overflows usize".into())
            })?;
            let mut valid = Vec::new();
            valid.try_reserve_exact(count).map_err(|_| {
                Error::Generation(format!(
                    "failed to allocate implicit prompt mask with {count} entries"
                ))
            })?;
            valid.resize(count, true);
            valid
        }
    };
    let prompt_lengths = prompt_lengths(&prompt_valid, batch, sequence)?;
    let maximum_prompt = prompt_lengths.iter().copied().max().ok_or_else(|| {
        Error::Generation("non-empty prompt unexpectedly had no batch rows".into())
    })?;
    let required_capacity = maximum_prompt
        .checked_add(request.generation.max_new_tokens.saturating_sub(1))
        .ok_or_else(|| Error::Generation("required cache capacity overflows usize".into()))?;
    if request.cache.capacity() < required_capacity {
        return Err(Error::Generation(format!(
            "cache capacity {} is smaller than the required {required_capacity} for a {maximum_prompt}-token prompt and {} generated tokens",
            request.cache.capacity(),
            request.generation.max_new_tokens
        )));
    }

    let next_positions = match &request.prompt.position_ids {
        Some(positions) => {
            if positions.dims() != [batch, sequence] {
                return Err(Error::shape(
                    "generation.prompt.position_ids",
                    format!("[{batch}, {sequence}]"),
                    format!("{:?}", positions.dims()),
                ));
            }
            let values = tensor_i64_values(positions.clone(), "prompt position IDs")?;
            Some(next_explicit_positions(
                &values,
                &prompt_valid,
                batch,
                sequence,
                request.generation.max_new_tokens,
            )?)
        }
        None => None,
    };

    Ok(PreparedRequest {
        batch,
        prompt_lengths,
        prompt_valid,
        next_positions,
    })
}

/// Counts true entries in each rectangular prompt row and rejects empty rows.
fn prompt_lengths(valid: &[bool], batch: usize, sequence: usize) -> Result<Vec<usize>> {
    let expected = batch
        .checked_mul(sequence)
        .ok_or_else(|| Error::Generation("prompt element count overflows usize".into()))?;
    if valid.len() != expected {
        return Err(Error::Generation(format!(
            "prompt mask contains {} entries, expected {expected}",
            valid.len()
        )));
    }
    let mut lengths = Vec::new();
    lengths.try_reserve_exact(batch).map_err(|_| {
        Error::Generation(format!(
            "failed to allocate prompt lengths for batch size {batch}"
        ))
    })?;
    for (row, values) in valid.chunks_exact(sequence).enumerate() {
        let length = values.iter().filter(|&&value| value).count();
        if length == 0 {
            return Err(Error::Generation(format!(
                "prompt row {row} has no valid tokens"
            )));
        }
        lengths.push(length);
    }
    Ok(lengths)
}

/// Finds the next position after each row's last valid explicit position.
fn next_explicit_positions(
    positions: &[i64],
    valid: &[bool],
    batch: usize,
    sequence: usize,
    max_new_tokens: usize,
) -> Result<Vec<i64>> {
    let expected = batch
        .checked_mul(sequence)
        .ok_or_else(|| Error::Generation("position element count overflows usize".into()))?;
    if positions.len() != expected || valid.len() != expected {
        return Err(Error::Generation(format!(
            "position metadata is not rectangular: {} positions and {} mask entries for {expected} prompt slots",
            positions.len(),
            valid.len()
        )));
    }
    let additional = i64::try_from(max_new_tokens.saturating_sub(1)).map_err(|_| {
        Error::Generation("max_new_tokens exceeds the signed position domain".into())
    })?;
    let mut next = Vec::new();
    next.try_reserve_exact(batch).map_err(|_| {
        Error::Generation(format!(
            "failed to allocate explicit positions for batch size {batch}"
        ))
    })?;
    for row in 0..batch {
        let offset = row
            .checked_mul(sequence)
            .ok_or_else(|| Error::Generation("position row offset overflows usize".into()))?;
        let row_positions = positions
            .get(offset..offset + sequence)
            .ok_or_else(|| Error::Generation("position rows became ragged".into()))?;
        let row_valid = valid
            .get(offset..offset + sequence)
            .ok_or_else(|| Error::Generation("position-mask rows became ragged".into()))?;
        let last = row_positions
            .iter()
            .zip(row_valid)
            .rev()
            .find_map(|(&position, &is_valid)| is_valid.then_some(position))
            .ok_or_else(|| Error::Generation(format!("prompt row {row} has no valid position")))?;
        if last < 0 {
            return Err(Error::Generation(format!(
                "prompt row {row} has negative last valid position {last}"
            )));
        }
        let maximum = last.checked_add(additional).ok_or_else(|| {
            Error::Generation(format!("explicit positions overflow for prompt row {row}"))
        })?;
        if maximum > i64::from(i32::MAX) {
            return Err(Error::Generation(format!(
                "explicit position {maximum} for prompt row {row} exceeds Burn's portable signed index range"
            )));
        }
        next.push(last + 1);
    }
    Ok(next)
}

/// Converts model output `[batch, 1, vocabulary]` into processor input.
fn extract_last_logits<B: Backend>(
    logits: Tensor<B, 3>,
    batch: usize,
    vocabulary_size: usize,
) -> Result<Tensor<B, 2>> {
    let dimensions = logits.dims();
    if dimensions != [batch, 1, vocabulary_size] {
        return Err(Error::shape(
            "generation.model_logits",
            format!("[{batch}, 1, {vocabulary_size}]"),
            format!("{dimensions:?}"),
        ));
    }
    Ok(logits.squeeze_dim::<2>(1))
}

/// Replaces ignored rows with finite scores before backend-wide sampling math.
///
/// Architecture implementations define inactive decode outputs as ignorable,
/// not necessarily finite. Neutralizing those rows prevents a NaN produced by
/// one finished sequence from contaminating fused batched reductions.
fn neutralize_inactive_logits<B: Backend>(
    logits: Tensor<B, 2>,
    active: &[bool],
) -> Result<Tensor<B, 2>> {
    let [batch, vocabulary] = logits.dims();
    if active.len() != batch {
        return Err(Error::Generation(format!(
            "active-row count {} does not match logits batch {batch}",
            active.len()
        )));
    }
    let device = logits.device();
    let valid =
        bool_tensor_from_values(active.to_vec(), [batch, 1], &device).expand([batch, vocabulary]);
    Ok(logits.mask_fill(valid.bool_not(), 0.0))
}

/// Copies a small Boolean metadata tensor to the host with error context.
fn tensor_bool_values<B: Backend>(tensor: Tensor<B, 2, Bool>, label: &str) -> Result<Vec<bool>> {
    crate::architecture::tensor_bool_values(tensor, label)
}

/// Copies a small integer metadata tensor to the host with error context.
fn tensor_i64_values<B: Backend, const D: usize>(
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

/// Copies selected log probabilities using their runtime storage dtype.
///
/// Burn backends may execute F16/BF16 tensors even when their default
/// `FloatElem` is F32. Reading with that associated type would reject valid
/// mixed-precision output. Only the already-selected `[batch]` values cross
/// the device boundary here; the vocabulary logits remain on the backend.
fn tensor_f64_values<B: Backend>(tensor: Tensor<B, 1>, label: &str) -> Result<Vec<f64>> {
    let data = tensor.try_into_data().map_err(|error| {
        Error::BackendCapability(format!("failed to read {label} from backend: {error}"))
    })?;
    host_float_data_f64(data, label)
}

/// Converts floating host storage directly to F64 without a lower-precision stage.
///
/// F16, BF16, F32, and Flex32 storage have exact F64 representations; existing
/// F64 values therefore retain their precision too. Typed borrowed slices validate byte
/// alignment and length without allocating a second source buffer. Quantized
/// or integer payloads are not log probabilities and fail contextually rather
/// than being reinterpreted or dequantized.
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
            "failed to read {label} from backend: unsupported probability dtype {dtype:?}"
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

/// Creates one explicit decode-position column on the selected backend.
fn position_column<B: Backend>(positions: &[i64], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(
        TensorData::new(positions.to_vec(), [positions.len(), 1]),
        device,
    )
}

/// Advances positions only for rows that committed a decode token.
fn advance_positions(positions: &mut [i64], active: &[bool]) -> Result<()> {
    if positions.len() != active.len() {
        return Err(Error::Generation(format!(
            "position batch {} does not match active-row batch {}",
            positions.len(),
            active.len()
        )));
    }
    for (row, (position, is_active)) in positions.iter_mut().zip(active).enumerate() {
        if *is_active {
            *position = position.checked_add(1).ok_or_else(|| {
                Error::Generation(format!("explicit position overflow for row {row}"))
            })?;
        }
    }
    Ok(())
}

/// Allocates a fallible vector of empty per-sequence rows.
fn empty_rows<T>(batch: usize, label: &str) -> Result<Vec<Vec<T>>> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(batch).map_err(|_| {
        Error::Generation(format!(
            "failed to allocate {label} rows for batch size {batch}"
        ))
    })?;
    rows.resize_with(batch, Vec::new);
    Ok(rows)
}

/// Combines tracker tokens, probabilities, and prompt lengths into public output.
fn finish_output(
    tracker: GenerationTracker,
    prompt_lengths: Vec<usize>,
    log_probabilities: Vec<Vec<f64>>,
) -> Result<CausalGenerationOutput> {
    let batch = tracker.batch_size();
    if prompt_lengths.len() != batch || log_probabilities.len() != batch {
        return Err(Error::Generation(
            "generation result metadata has inconsistent batch dimensions".into(),
        ));
    }
    let mut sequences = Vec::new();
    sequences.try_reserve_exact(batch).map_err(|_| {
        Error::Generation(format!(
            "failed to allocate completed output for batch size {batch}"
        ))
    })?;
    for sequence in 0..batch {
        let token_ids = tracker.generated(sequence)?.to_vec();
        let probabilities = log_probabilities
            .get(sequence)
            .cloned()
            .ok_or_else(|| Error::Generation("missing generated probabilities".into()))?;
        if token_ids.len() != probabilities.len() {
            return Err(Error::Generation(format!(
                "sequence {sequence} has {} tokens but {} log probabilities",
                token_ids.len(),
                probabilities.len()
            )));
        }
        let finish_reason = tracker
            .finish_reason(sequence)?
            .cloned()
            .ok_or_else(|| Error::Generation(format!("sequence {sequence} did not finish")))?;
        sequences.push(GeneratedSequence {
            prompt_length: *prompt_lengths
                .get(sequence)
                .ok_or_else(|| Error::Generation("missing prompt length".into()))?,
            token_ids,
            log_probabilities: probabilities,
            finish_reason,
        });
    }
    Ok(CausalGenerationOutput { sequences })
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        tensor::{
            Bool, Int, Tensor,
            backend::{Backend, BackendTypes},
        },
    };

    use super::{
        CausalGenerationRequest, GenerationToken, generate_causal, generate_causal_streaming,
    };
    use crate::{
        CausalLmInput, Error, FinishReason, GenerationConfig, SamplingStrategy, TokenMask,
        cache::CacheConfig,
        models::qwen3_5::{Qwen3_5ForCausalLm, Qwen3_5LayerType, Qwen3_5TextConfig},
    };

    type TestBackend = Flex<f32, i32>;

    /// Replaces one real model output with an exact nonfinite score fixture.
    ///
    /// A constant NaN or infinite row makes normalized log probabilities
    /// undefined even if argmax returns an in-range token. Retaining the real
    /// Qwen cache makes this a coordinator boundary test, not a tracker-only test.
    #[derive(burn::module::Module, Debug)]
    struct NonfiniteModel<B: Backend> {
        inner: Qwen3_5ForCausalLm<B>,
        #[module(skip)]
        fail_on_decode: bool,
        #[module(skip)]
        corrupt_value: f32,
        #[module(skip)]
        prefill_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        #[module(skip)]
        decode_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl<B: Backend> crate::CausalLanguageModel<B> for NonfiniteModel<B> {
        type Config = Qwen3_5TextConfig;
        type State = <Qwen3_5ForCausalLm<B> as crate::CausalLanguageModel<B>>::State;

        /// Leaves stateless evaluation untouched; only cached outputs inject faults.
        fn forward(
            &self,
            input: CausalLmInput<B>,
            options: &crate::ForwardOptions,
        ) -> crate::Result<crate::CausalLmOutput<B>> {
            self.inner.forward(input, options)
        }

        /// Allocates the real architecture's external cache without changing policy.
        fn new_decode_state(
            &self,
            config: CacheConfig,
            batch_size: usize,
            device: &B::Device,
        ) -> crate::Result<Self::State> {
            self.inner.new_decode_state(config, batch_size, device)
        }

        /// Counts prompt calls and optionally corrupts the first selectable distribution.
        fn prefill(
            &self,
            input: CausalLmInput<B>,
            state: &mut Self::State,
            options: &crate::ForwardOptions,
        ) -> crate::Result<crate::CausalLmOutput<B>> {
            self.prefill_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut output = self.inner.prefill(input, state, options)?;
            if !self.fail_on_decode {
                output.logits = Tensor::full(
                    output.logits.dims(),
                    self.corrupt_value,
                    &output.logits.device(),
                );
            }
            Ok(output)
        }

        /// Counts feedback calls and injects the fault only after one valid token.
        fn decode(
            &self,
            input: crate::DecodeInput<B>,
            state: &mut Self::State,
            options: &crate::ForwardOptions,
        ) -> crate::Result<crate::CausalLmOutput<B>> {
            self.decode_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut output = self.inner.decode(input, state, options)?;
            output.logits = Tensor::full(
                output.logits.dims(),
                self.corrupt_value,
                &output.logits.device(),
            );
            Ok(output)
        }
    }

    /// Active nonfinite probabilities abort before emission or any later model call.
    #[test]
    fn active_nonfinite_probabilities_cancel_before_the_invalid_token() {
        use std::sync::{Arc, atomic::AtomicUsize};

        for corrupt_value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for fail_on_decode in [false, true] {
                let (inner, device) = tiny_model();
                let model = NonfiniteModel {
                    inner,
                    fail_on_decode,
                    corrupt_value,
                    prefill_calls: Arc::new(AtomicUsize::new(0)),
                    decode_calls: Arc::new(AtomicUsize::new(0)),
                };
                let mut events = Vec::new();
                let error = generate_causal_streaming(
                    &model,
                    request(
                        &device,
                        GenerationConfig {
                            max_new_tokens: 3,
                            ..Default::default()
                        },
                    ),
                    |event| {
                        events.push(event);
                        Ok(())
                    },
                )
                .expect_err("an undefined active probability cannot be emitted");
                assert!(matches!(error, Error::Generation(_)));
                assert!(
                    error
                        .to_string()
                        .contains("sequence 0 produced non-finite selected log probability"),
                    "{error}"
                );
                let valid_tokens = usize::from(fail_on_decode);
                assert_eq!(events.len(), valid_tokens);
                for event in events {
                    assert_eq!((event.sequence, event.step), (0, 0));
                    assert!(event.log_probability.is_finite());
                    assert_eq!(event.finish_reason, None);
                }
                assert_eq!(
                    model
                        .prefill_calls
                        .load(std::sync::atomic::Ordering::Relaxed),
                    1
                );
                assert_eq!(
                    model
                        .decode_calls
                        .load(std::sync::atomic::Ordering::Relaxed),
                    valid_tokens,
                    "the coordinator must not decode after rejecting a probability"
                );
            }
        }
    }

    /// Runtime half-precision selected values cannot be decoded as the backend's default F32.
    #[test]
    fn selected_log_probabilities_respect_runtime_float_dtype() {
        use burn::tensor::DType;

        let device = Default::default();
        for strategy in [SamplingStrategy::Greedy, SamplingStrategy::Sample] {
            let processor = crate::LogitsProcessor::<TestBackend>::new(
                GenerationConfig {
                    strategy,
                    ..GenerationConfig::default()
                },
                2,
                &device,
            )
            .expect("token processor");
            for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
                let logits =
                    Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0], [0.0, 0.0]], &device)
                        .cast(dtype);
                let selected = processor
                    .select_with_log_probs(logits)
                    .expect("selected probabilities");
                assert_eq!(selected.log_probabilities.dtype(), dtype);
                assert_eq!(selected.log_probabilities.dims(), [2]);
                let values = super::tensor_f64_values(
                    selected.log_probabilities,
                    "selected probability fixture",
                )
                .expect("runtime dtype-aware host conversion");
                assert_eq!(values.len(), 2);
                assert!(
                    values
                        .iter()
                        .all(|value| (*value + core::f64::consts::LN_2).abs() < 0.01)
                );
            }
        }
    }

    /// F64 values keep sub-F32 precision, and malformed or non-floating storage is recoverable.
    #[test]
    fn host_probability_conversion_preserves_f64_and_rejects_invalid_storage() {
        use burn::tensor::{DType, TensorData};

        let exact = -1.0_f64 - 2.0_f64.powi(-40);
        let data = TensorData::new(vec![exact], [1]);
        assert_eq!(
            super::host_float_data_f64(data, "F64 fixture").expect("native F64"),
            [exact]
        );
        let integer = TensorData::new(vec![1_i32], [1]);
        let error = super::host_float_data_f64(integer, "integer fixture")
            .expect_err("probabilities must use float storage");
        assert!(matches!(error, Error::BackendCapability(_)));
        assert!(error.to_string().contains("integer fixture"));
        let mut malformed = TensorData::new(vec![0_u8; 3], [3]);
        malformed.dtype = DType::F32;
        let error = super::host_float_data_f64(malformed, "truncated fixture")
            .expect_err("bad element size must not panic");
        assert!(error.to_string().contains("truncated fixture"));
    }

    /// Flex32 uses exact F32 host bytes even though its dtype tag is distinct.
    #[test]
    fn flex32_host_probability_storage_is_lossless_and_checked() {
        use burn::tensor::{DType, TensorData};

        let values = [-1.0_f32 - 2.0_f32.powi(-20), -0.0, f32::MIN_POSITIVE];
        let mut data = TensorData::new(values.to_vec(), [values.len()]);
        // This is CubeCL's host-data contract; Flex CPU cannot execute Flex32
        // casts, so ordinary CPU CI covers storage independently of GPU kernels.
        data.dtype = DType::Flex32;
        let actual = super::host_float_data_f64(data, "Flex32 fixture")
            .expect("Flex32 is supported floating storage");
        assert_eq!(actual, values.map(f64::from));
        assert!(actual[1].is_sign_negative());
        let mut malformed = TensorData::new(vec![0_u8; 3], [3]);
        malformed.dtype = DType::Flex32;
        let error = super::host_float_data_f64(malformed, "truncated Flex32 fixture")
            .expect_err("invalid Flex32 element size is recoverable");
        assert!(error.to_string().contains("truncated Flex32 fixture"));
    }

    /// The native Flex32 Metal backend supports selection, penalties, and host readback.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    #[ignore = "requires serialized Metal execution"]
    fn native_flex32_generation_readback() {
        use burn::{
            backend::{Metal, wgpu::flex32},
            tensor::DType,
        };

        type NativeFlex32 = Metal<flex32>;
        let device = Default::default();
        for strategy in [SamplingStrategy::Greedy, SamplingStrategy::Sample] {
            let processor = crate::LogitsProcessor::<NativeFlex32>::new(
                GenerationConfig {
                    strategy,
                    repetition_penalty: 2.0,
                    presence_penalty: 0.5,
                    temperature: 0.5,
                    ..GenerationConfig::default()
                },
                4,
                &device,
            )
            .expect("native Flex32 processor");
            // Keep sorting out of this readback regression: Burn 0.21's generic
            // Flex32 sort fallback currently reinterprets its tag as F32.
            let logits = Tensor::<NativeFlex32, 2>::from_floats([[4.0, -3.0, 32.0, 1.0]], &device)
                // CubeCL aliases a direct F32->Flex32 cast without changing
                // its tag; F16->Flex32 allocates genuinely Flex32-typed storage.
                .cast(DType::F16)
                .cast(DType::Flex32);
            assert_eq!(logits.dtype(), DType::Flex32);
            assert_eq!(logits.clone().cast(DType::F32).dtype(), DType::Flex32);
            let history = Tensor::<NativeFlex32, 2, Int>::from_ints([[0, 1]], &device);
            let selected = processor
                .process_and_select_with_log_probs(logits, Some(history))
                .expect("native Flex32 probability processing");
            assert_eq!(selected.log_probabilities.dtype(), DType::Flex32);
            assert_eq!(
                super::tensor_i64_values(selected.token_ids, "Flex32 selection")
                    .expect("selected ID"),
                [2]
            );
            let values =
                super::tensor_f64_values(selected.log_probabilities, "Flex32 log probability")
                    .expect("native Flex32 host readback");
            assert_eq!(values.len(), 1);
            assert!(values[0].is_finite() && values[0] <= 0.0);
        }
    }

    /// Builds a deterministic tiny model with a real external KV cache.
    fn tiny_model() -> (
        Qwen3_5ForCausalLm<TestBackend>,
        <TestBackend as BackendTypes>::Device,
    ) {
        let device = Default::default();
        TestBackend::seed(&device, 991);
        let config = Qwen3_5TextConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 64,
            layer_types: vec![Qwen3_5LayerType::FullAttention],
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 1,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 3,
            partial_rotary_factor: Some(1.0),
            rope_theta: Some(10_000.0),
            mtp_num_hidden_layers: 0,
            ..Default::default()
        };
        let model = Qwen3_5ForCausalLm::init(config, &device)
            .expect("tiny generation model must initialize");
        (model, device)
    }

    /// Constructs a single-row prompt and a cache large enough for generation.
    fn request(
        device: &<TestBackend as BackendTypes>::Device,
        generation: GenerationConfig,
    ) -> CausalGenerationRequest<TestBackend> {
        CausalGenerationRequest {
            prompt: CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], device),
                token_mask: None,
                position_ids: None,
            },
            cache: CacheConfig::new(1, 8).expect("test cache config"),
            generation,
            prefill_chunk_size: None,
            vocabulary_size: 32,
        }
    }

    /// Greedy generation emits ordered events and aligned finite probabilities.
    #[test]
    fn greedy_generation_streams_complete_length_limited_output() {
        let (model, device) = tiny_model();
        let mut events = Vec::<GenerationToken>::new();
        let output = generate_causal_streaming(
            &model,
            request(
                &device,
                GenerationConfig {
                    max_new_tokens: 3,
                    ..Default::default()
                },
            ),
            |event| {
                events.push(event);
                Ok(())
            },
        )
        .expect("cached generation should succeed");
        let sequence = &output.sequences[0];
        assert_eq!(sequence.prompt_length, 2);
        assert_eq!(sequence.token_ids.len(), 3);
        assert_eq!(sequence.log_probabilities.len(), 3);
        assert!(
            sequence
                .log_probabilities
                .iter()
                .all(|value| value.is_finite())
        );
        assert_eq!(sequence.finish_reason, FinishReason::Length);
        assert_eq!(events.len(), 3);
        assert_eq!(
            events.iter().map(|event| event.step).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            events.last().and_then(|event| event.finish_reason.clone()),
            Some(FinishReason::Length)
        );
    }

    /// EOS and multi-token suffixes terminate at the first matching step.
    #[test]
    fn eos_and_stop_sequences_terminate_generation() {
        let (model, device) = tiny_model();
        let baseline = generate_causal(
            &model,
            request(
                &device,
                GenerationConfig {
                    max_new_tokens: 3,
                    ..Default::default()
                },
            ),
        )
        .expect("baseline generation");
        let tokens = baseline.sequences[0].token_ids.clone();

        let eos = generate_causal(
            &model,
            request(
                &device,
                GenerationConfig {
                    max_new_tokens: 3,
                    eos_token_ids: vec![tokens[0]],
                    ..Default::default()
                },
            ),
        )
        .expect("EOS generation");
        assert_eq!(eos.sequences[0].token_ids, vec![tokens[0]]);
        assert_eq!(
            eos.sequences[0].finish_reason,
            FinishReason::EndOfSequence {
                token_id: tokens[0]
            }
        );

        let stopped = generate_causal(
            &model,
            request(
                &device,
                GenerationConfig {
                    max_new_tokens: 3,
                    stop_sequences: vec![tokens[..2].to_vec()],
                    ..Default::default()
                },
            ),
        )
        .expect("stop-sequence generation");
        assert_eq!(stopped.sequences[0].token_ids, tokens[..2]);
        assert_eq!(
            stopped.sequences[0].finish_reason,
            FinishReason::StopSequence {
                index: 0,
                length: 2
            }
        );
    }

    /// Independent seeded processors reproduce a complete stochastic sequence.
    #[test]
    fn stochastic_generation_is_reproducible() {
        let (model, device) = tiny_model();
        let generation = GenerationConfig {
            max_new_tokens: 4,
            strategy: SamplingStrategy::Sample,
            temperature: 0.8,
            top_k: Some(8),
            top_p: Some(0.9),
            seed: 17,
            ..Default::default()
        };
        let first = generate_causal(&model, request(&device, generation.clone()))
            .expect("first seeded generation");
        let second = generate_causal(&model, request(&device, generation))
            .expect("second seeded generation");
        assert_eq!(first, second);
    }

    /// Ragged prompt rows preserve their semantic lengths throughout decoding.
    #[test]
    fn ragged_batch_uses_true_valid_masks() {
        let (model, device) = tiny_model();
        let output = generate_causal(
            &model,
            CausalGenerationRequest {
                prompt: CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                        [[1, 31, 2], [3, 31, 31]],
                        &device,
                    ),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[true, false, true], [true, false, false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                cache: CacheConfig::new(2, 4).expect("test cache"),
                generation: GenerationConfig {
                    max_new_tokens: 2,
                    repetition_penalty: 1.1,
                    presence_penalty: 0.2,
                    ..Default::default()
                },
                prefill_chunk_size: None,
                vocabulary_size: 32,
            },
        )
        .expect("ragged generation");
        assert_eq!(output.sequences.len(), 2);
        assert_eq!(output.sequences[0].prompt_length, 2);
        assert_eq!(output.sequences[1].prompt_length, 1);
        assert!(
            output
                .sequences
                .iter()
                .all(|sequence| sequence.token_ids.len() == 2)
        );
    }

    /// Chunked prefill retains each ragged row's last active logits across padding-only chunks.
    #[test]
    fn chunked_prefill_matches_one_shot_for_ragged_batches() {
        let (model, device) = tiny_model();
        let request = CausalGenerationRequest {
            prompt: CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                    [[1, 2, 3, 31, 31, 31], [31, 31, 4, 5, 31, 6]],
                    &device,
                ),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [
                        [true, true, true, false, false, false],
                        [false, false, true, true, false, true],
                    ]
                    .into(),
                    &device,
                ))),
                position_ids: None,
            },
            cache: CacheConfig::new(2, 8).expect("test cache"),
            generation: GenerationConfig {
                max_new_tokens: 3,
                ..Default::default()
            },
            prefill_chunk_size: None,
            vocabulary_size: 32,
        };
        let one_shot = generate_causal(&model, request.clone()).expect("one-shot prefill");
        let mut chunked_request = request;
        chunked_request.prefill_chunk_size = Some(2);
        let chunked = generate_causal(&model, chunked_request).expect("chunked prefill");

        assert_eq!(chunked.sequences.len(), one_shot.sequences.len());
        for (chunked, one_shot) in chunked.sequences.iter().zip(&one_shot.sequences) {
            assert_eq!(chunked.prompt_length, one_shot.prompt_length);
            assert_eq!(chunked.token_ids, one_shot.token_ids);
            assert_eq!(chunked.finish_reason, one_shot.finish_reason);
            assert_eq!(
                chunked.log_probabilities.len(),
                one_shot.log_probabilities.len()
            );
            for (&actual, &expected) in chunked
                .log_probabilities
                .iter()
                .zip(&one_shot.log_probabilities)
            {
                assert!(
                    (actual - expected).abs() <= 1.0e-5,
                    "chunked log probability {actual} differs from one-shot {expected}"
                );
            }
        }
    }

    /// A zero prefill bound is rejected before any model operation.
    #[test]
    fn zero_prefill_chunk_size_is_invalid() {
        let (model, device) = tiny_model();
        let mut invalid = request(
            &device,
            GenerationConfig {
                max_new_tokens: 1,
                ..Default::default()
            },
        );
        invalid.prefill_chunk_size = Some(0);

        let error = generate_causal(&model, invalid).expect_err("zero cannot advance chunks");
        assert!(error.to_string().contains("prefill_chunk_size"));
    }

    /// Explicit prompt positions advance once per committed continuation token.
    #[test]
    fn explicit_positions_continue_from_last_valid_token() {
        let (model, device) = tiny_model();
        let mut generation_request = request(
            &device,
            GenerationConfig {
                max_new_tokens: 2,
                ..Default::default()
            },
        );
        generation_request.prompt.position_ids = Some(Tensor::from_ints([[7, 9]], &device));
        let output = generate_causal(&model, generation_request)
            .expect("generation with explicit positions should succeed");
        assert_eq!(output.sequences[0].token_ids.len(), 2);
    }

    /// Coordinator errors are raised before entering model inference when possible.
    #[test]
    fn request_validation_rejects_bad_masks_and_cache_geometry() {
        let (model, device) = tiny_model();
        let generation = GenerationConfig {
            max_new_tokens: 3,
            ..Default::default()
        };
        let mut wrong_batch = request(&device, generation.clone());
        wrong_batch.cache = CacheConfig::new(2, 8).expect("valid but mismatched cache");
        assert!(generate_causal(&model, wrong_batch).is_err());

        let mut too_small = request(&device, generation.clone());
        too_small.cache = CacheConfig::new(1, 3).expect("valid but undersized cache");
        assert!(generate_causal(&model, too_small).is_err());

        let mut wrong_mask = request(&device, generation.clone());
        wrong_mask.prompt.token_mask =
            Some(TokenMask::new(Tensor::from_bool([[true]].into(), &device)));
        assert!(generate_causal(&model, wrong_mask).is_err());

        let mut empty_row = request(&device, generation);
        empty_row.prompt.token_mask = Some(TokenMask::new(Tensor::from_bool(
            [[false, false]].into(),
            &device,
        )));
        assert!(generate_causal(&model, empty_row).is_err());
    }

    /// A callback can cancel generation without panicking or emitting later tokens.
    #[test]
    fn streaming_callback_errors_cancel_generation() {
        let (model, device) = tiny_model();
        let mut calls = 0;
        let result = generate_causal_streaming(
            &model,
            request(
                &device,
                GenerationConfig {
                    max_new_tokens: 3,
                    ..Default::default()
                },
            ),
            |_| {
                calls += 1;
                Err(Error::Generation("consumer disconnected".into()))
            },
        );
        assert!(
            matches!(result, Err(Error::Generation(message)) if message == "consumer disconnected")
        );
        assert_eq!(calls, 1);
    }

    /// Malformed positions and impossible logical capacity fail before any token event.
    #[test]
    fn request_position_and_capacity_boundaries_are_contextual() {
        let (model, device) = tiny_model();
        let generation = GenerationConfig {
            max_new_tokens: 3,
            ..Default::default()
        };
        for (positions, expected) in [
            (vec![7], "generation.prompt.position_ids"),
            (vec![7, -1], "negative last valid position"),
            (vec![7, i32::MAX], "portable signed index range"),
        ] {
            let length = positions.len();
            let mut invalid = request(&device, generation.clone());
            invalid.prompt.position_ids = Some(Tensor::from_data(
                burn::tensor::TensorData::new(positions, [1, length]),
                &device,
            ));
            let error = generate_causal_streaming(&model, invalid, |_| {
                panic!("invalid request must not emit a token")
            })
            .expect_err("invalid explicit positions");
            assert!(error.to_string().contains(expected), "{error}");
        }
        let mut overflow = request(&device, generation);
        overflow.generation.max_new_tokens = usize::MAX;
        let error = generate_causal(&model, overflow).expect_err("capacity sum overflows");
        assert!(
            error
                .to_string()
                .contains("required cache capacity overflows")
        );
    }

    /// Last-logit selection enforces every axis, including a wrong vocabulary declaration.
    #[test]
    fn model_output_shape_errors_are_contextual_in_every_prefill_mode() {
        let (model, device) = tiny_model();
        for chunk_size in [None, Some(1), Some(2), Some(usize::MAX)] {
            let mut invalid = request(
                &device,
                GenerationConfig {
                    max_new_tokens: 1,
                    ..Default::default()
                },
            );
            invalid.vocabulary_size = 33;
            invalid.prefill_chunk_size = chunk_size;
            let error = generate_causal(&model, invalid).expect_err("model has 32 logits");
            assert!(error.to_string().contains("generation.model_logits"));
            assert!(error.to_string().contains("33"));
        }
        for shape in [[2, 1, 32], [1, 2, 32], [1, 1, 31]] {
            let error =
                super::extract_last_logits(Tensor::<TestBackend, 3>::zeros(shape, &device), 1, 32)
                    .expect_err("one wrong output axis");
            assert!(error.to_string().contains("generation.model_logits"));
        }
    }

    /// Chunk-size fast paths preserve exact greedy results and full prompt position IDs.
    #[test]
    fn dense_chunking_and_unbounded_fast_paths_preserve_positions() {
        let (model, device) = tiny_model();
        let mut input = request(
            &device,
            GenerationConfig {
                max_new_tokens: 3,
                ..Default::default()
            },
        );
        input.prompt.position_ids = Some(Tensor::from_ints([[4, 7]], &device));
        let baseline = generate_causal(&model, input.clone()).expect("one-shot positioned prompt");
        for chunk in [1, 2, usize::MAX] {
            let mut chunked = input.clone();
            chunked.prefill_chunk_size = Some(chunk);
            let output = generate_causal(&model, chunked).expect("bounded positioned prompt");
            assert_eq!(
                output.sequences[0].token_ids,
                baseline.sequences[0].token_ids
            );
            for (actual, expected) in output.sequences[0]
                .log_probabilities
                .iter()
                .zip(&baseline.sequences[0].log_probabilities)
            {
                assert!((actual - expected).abs() < 1.0e-5);
            }
        }
    }

    /// Helper preflight rejects rectangular metadata errors before invoking a model.
    #[test]
    fn chunked_prefill_rejects_invalid_compact_metadata() {
        use crate::{CausalLanguageModel, DecodeState, ForwardOptions};

        let (model, device) = tiny_model();
        let request = request(
            &device,
            GenerationConfig {
                max_new_tokens: 1,
                ..Default::default()
            },
        );
        let mut state = model
            .new_decode_state(request.cache, 1, &device)
            .expect("state");
        for (mask, size, expected) in [
            (vec![true, true], 0, "prefill_chunk_size"),
            (vec![true], 1, "prompt mask contains 1 entries, expected 2"),
        ] {
            let error = super::prefill_prompt(
                &model,
                request.prompt.clone(),
                &mut state,
                &ForwardOptions::default(),
                &mask,
                Some(size),
                32,
            )
            .expect_err("bad prefill metadata");
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(state.lengths(), [0]);
        }
    }

    /// Position planning validates overflow, signed bounds, ragged inputs, and padding.
    #[test]
    fn compact_position_metadata_has_checked_boundaries() {
        let error = super::prompt_lengths(&[], usize::MAX, 2).expect_err("product overflow");
        assert!(error.to_string().contains("element count overflows"));
        let error = super::prompt_lengths(&[true], 1, 2).expect_err("ragged mask");
        assert!(error.to_string().contains("expected 2"));
        assert_eq!(
            super::prompt_lengths(&[true, false, true, false], 2, 2).expect("valid rows"),
            [1, 1]
        );
        for (positions, valid, batch, sequence, tokens, message) in [
            (vec![], vec![], usize::MAX, 2, 1, "element count overflows"),
            (vec![0], vec![true, true], 1, 2, 1, "not rectangular"),
            (
                vec![0],
                vec![true],
                1,
                1,
                usize::MAX,
                "signed position domain",
            ),
            (vec![0], vec![false], 1, 1, 1, "no valid position"),
            (
                vec![-1],
                vec![true],
                1,
                1,
                1,
                "negative last valid position",
            ),
            (
                vec![i64::MAX],
                vec![true],
                1,
                1,
                2,
                "explicit positions overflow",
            ),
            (
                vec![i64::from(i32::MAX)],
                vec![true],
                1,
                1,
                2,
                "portable signed index range",
            ),
        ] {
            let error = super::next_explicit_positions(&positions, &valid, batch, sequence, tokens)
                .expect_err("invalid compact position metadata");
            assert!(error.to_string().contains(message), "{error}");
        }
        assert_eq!(
            super::next_explicit_positions(
                &[i64::MAX, 7, -1, i64::from(i32::MAX)],
                &[false, true, false, true],
                2,
                2,
                1,
            )
            .expect("masked sentinels are ignored and last sampled token is not decoded"),
            [8, i64::from(i32::MAX) + 1]
        );
        let mut positions = [5, i64::MAX];
        super::advance_positions(&mut positions, &[true, false])
            .expect("inactive row is untouched");
        assert_eq!(positions, [6, i64::MAX]);
        let error = super::advance_positions(&mut positions, &[true]).expect_err("ragged activity");
        assert!(error.to_string().contains("position batch"));
        assert_eq!(positions, [6, i64::MAX]);
        let error =
            super::advance_positions(&mut [i64::MAX], &[true]).expect_err("signed overflow");
        assert!(
            error
                .to_string()
                .contains("explicit position overflow for row 0")
        );
    }

    /// Inactive nonfinite logits are zeroed without modifying still-active rows.
    #[test]
    fn neutralization_is_row_local_and_checks_batch_metadata() {
        let device = Default::default();
        let logits = Tensor::<TestBackend, 2>::from_floats(
            [[f32::NAN, f32::INFINITY], [1.0, -2.0]],
            &device,
        );
        let values = super::neutralize_inactive_logits(logits.clone(), &[false, true])
            .expect("inactive nonfinite row")
            .into_data()
            .to_vec::<f32>()
            .expect("F32 values");
        assert_eq!(values, [0.0, 0.0, 1.0, -2.0]);
        let error = super::neutralize_inactive_logits(logits, &[false])
            .expect_err("activity must cover the batch");
        assert!(error.to_string().contains("active-row count 1"));
    }

    /// Public output assembly rejects incomplete tracking or misaligned log probabilities.
    #[test]
    fn output_assembly_never_fabricates_terminal_metadata() {
        let config = GenerationConfig {
            max_new_tokens: 1,
            ..Default::default()
        };
        let tracker = super::GenerationTracker::new(config, 32, 1).expect("tracker");
        for (lengths, probabilities, expected) in [
            (vec![], vec![vec![]], "inconsistent batch dimensions"),
            (vec![2], vec![], "inconsistent batch dimensions"),
            (
                vec![2],
                vec![vec![-1.0]],
                "0 tokens but 1 log probabilities",
            ),
            (vec![2], vec![vec![]], "sequence 0 did not finish"),
        ] {
            let error = super::finish_output(tracker.clone(), lengths, probabilities)
                .expect_err("inconsistent output must fail");
            assert!(error.to_string().contains(expected), "{error}");
        }
        let mut completed = tracker;
        completed.push(0, 7).expect("terminal token");
        let output = super::finish_output(completed, vec![2], vec![vec![-0.5]])
            .expect("aligned completed metadata");
        assert_eq!(output.sequences[0].token_ids, [7]);
        assert_eq!(output.sequences[0].log_probabilities, [-0.5]);
        assert_eq!(output.sequences[0].finish_reason, FinishReason::Length);
    }
}
