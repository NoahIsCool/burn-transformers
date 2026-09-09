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

use burn::tensor::{Bool, ElementConversion, Int, Tensor, TensorData, backend::Backend};

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

/// Copies selected log probabilities to the host without transferring logits.
fn tensor_f64_values<B: Backend>(tensor: Tensor<B, 1>, label: &str) -> Result<Vec<f64>> {
    tensor
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to read {label} from backend: {error}"))
        })?
        .to_vec::<B::FloatElem>()
        .map(|values| {
            values
                .into_iter()
                .map(|value| value.elem::<f64>())
                .collect()
        })
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
}
