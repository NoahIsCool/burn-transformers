//! End-to-end cached generation for decoder-only language models.
//!
//! A causal model factorizes a continuation as
//! `p(y_1, ..., y_T | x) = product_t p(y_t | x, y_<t)`. The coordinator
//! performs the expensive prompt pass once or in caller-bounded cache-backed
//! chunks, then feeds only the previously selected token through
//! [`CausalLanguageModel::decode`]. Model caches remain preallocated and
//! architecture-owned.
//!
//! Each decode step costs one device-to-host synchronization: the selected
//! IDs and their log probabilities are read back together. Penalty state is a
//! [`TokenHistory`] updated with the new `[batch]` IDs, so no history tensor
//! grows and no vocabulary-sized data leaves the device. Like Hugging Face,
//! the coordinator promotes F16/BF16 logits to F32 before any processing.
//!
//! Batched rows may have different prompt and completion lengths. Every mask in
//! this module follows the crate-wide `true == valid` rule. Once a row reaches
//! EOS, a stop sequence, or its length limit, subsequent rectangular decode
//! calls carry a padding ID with a false mask entry, so that row neither changes
//! its cache nor influences token penalties.

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, DecodeInput, Error, ForwardOptions, LogitsSelection,
    Result, TokenMask, architecture::bool_tensor_from_values, cache::CacheConfig,
};

use super::bookkeeping::{prefill_chunk_end, row_span};
use super::metadata::{self, extract_last_logits};
use super::{
    FinishReason, GenerationConfig, GenerationTracker, LogitsProcessor, TokenHistory,
    upcast_half_precision,
};

/// Label used for model-logit shape errors raised by this coordinator.
const MODEL_LOGITS: &str = "generation.model_logits";

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
    let vocabulary_size = request.vocabulary_size;
    let processor =
        LogitsProcessor::<B>::new(request.generation.clone(), vocabulary_size, &device)?;
    // Built before any model call: it validates prompt IDs when penalties or
    // n-gram constraints need them, so a bad prompt fails without a session.
    let mut history = TokenHistory::from_prompt_values(
        &request.generation,
        vocabulary_size,
        request.prompt.token_ids.clone(),
        request.prompt.token_mask.clone(),
        &prompt_valid,
    )?;
    let mut tracker = GenerationTracker::new(request.generation.clone(), vocabulary_size, batch)?;
    let mut log_probabilities = metadata::empty_rows::<f64>(batch, "log-probability rows")?;
    let padding = i64::from(request.generation.pad_token_id.unwrap_or(0));
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
        vocabulary_size,
    )?;
    let mut logits = step_logits(prompt_logits, batch, vocabulary_size)?;

    loop {
        let processed = processor.process_with_history(logits, &history)?;
        let selection = processor.select_with_log_probs(processed)?;
        let token_ids = selection.token_ids.clone();
        // The step's only host synchronization.
        let (selected_ids, selected_log_probabilities) =
            metadata::read_token_selection(selection, vocabulary_size)?;
        let rows = emit_step(
            &mut tracker,
            &selected_ids,
            &selected_log_probabilities,
            &mut log_probabilities,
            &mut on_token,
        )?;
        history.record(&selected_ids, &rows.emitted)?;
        if tracker.all_finished() {
            break;
        }

        let decode_mask = bool_tensor_from_values(rows.still_active.clone(), [batch, 1], &device);
        let decode_ids = token_ids
            .reshape([batch, 1])
            .mask_fill(decode_mask.clone().bool_not(), padding);
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
            advance_positions(positions, &rows.still_active)?;
        }
        logits = neutralize_inactive_logits(
            step_logits(output.logits, batch, vocabulary_size)?,
            &rows.still_active,
        )?;
    }

    finish_output(tracker, prompt_lengths, log_probabilities)
}

/// Extracts one step's `[batch, vocabulary]` logits and promotes half precision to F32.
///
/// See [`upcast_half_precision`] for why processing runs in at least F32.
fn step_logits<B: Backend>(
    logits: Tensor<B, 3>,
    batch: usize,
    vocabulary_size: usize,
) -> Result<Tensor<B, 2>> {
    let logits = extract_last_logits(logits, batch, vocabulary_size, MODEL_LOGITS)?;
    Ok(upcast_half_precision(logits))
}

/// Rows that emitted a token in one step, and rows that remain active after it.
struct StepRows {
    /// Rows that were active before the step and therefore emitted a token.
    emitted: Vec<bool>,
    /// Rows still unfinished after the step; only these are decoded next.
    still_active: Vec<bool>,
}

/// Records one step's selected tokens for every unfinished row and streams them.
///
/// Rows that finished earlier are skipped: their selections come from
/// neutralized logits and carry no meaning. An active row with a non-finite
/// selected log probability aborts generation before that token is emitted.
fn emit_step<F>(
    tracker: &mut GenerationTracker,
    selected_ids: &[u32],
    selected_log_probabilities: &[f64],
    log_probabilities: &mut [Vec<f64>],
    on_token: &mut F,
) -> Result<StepRows>
where
    F: FnMut(GenerationToken) -> Result<()>,
{
    let batch = tracker.batch_size();
    if selected_ids.len() != batch
        || selected_log_probabilities.len() != batch
        || log_probabilities.len() != batch
    {
        return Err(Error::Generation(format!(
            "step metadata has {} token IDs, {} log probabilities, and {} output rows for batch size {batch}",
            selected_ids.len(),
            selected_log_probabilities.len(),
            log_probabilities.len()
        )));
    }
    let mut rows = StepRows {
        emitted: metadata::vec_with_capacity(batch, "emitted-row flags")?,
        still_active: metadata::vec_with_capacity(batch, "active-row flags")?,
    };
    for sequence in 0..batch {
        if tracker.finish_reason(sequence)?.is_some() {
            rows.emitted.push(false);
            rows.still_active.push(false);
            continue;
        }
        let token = selected_ids[sequence];
        let log_probability = selected_log_probabilities[sequence];
        if !log_probability.is_finite() {
            return Err(Error::Generation(format!(
                "sequence {sequence} produced non-finite selected log probability {log_probability}"
            )));
        }
        let step = tracker.generated(sequence)?.len();
        let finish_reason = tracker.push(sequence, token)?.cloned();
        log_probabilities[sequence].push(log_probability);
        on_token(GenerationToken {
            sequence,
            step,
            token_id: token,
            log_probability,
            finish_reason: finish_reason.clone(),
        })?;
        rows.emitted.push(true);
        rows.still_active.push(finish_reason.is_none());
    }
    Ok(rows)
}

/// Prefills a prompt in bounded physical-column chunks and retains each row's latest logits.
///
/// Rectangular batches can place the final valid token in different chunks.
/// A chunk in which every row is padding contributes nothing and is skipped:
/// its logits would be discarded and its cache effect is a no-op. A chunk in
/// which only some rows are active goes through `decode`, whose contract
/// treats inactive rows as cache no-ops, while fully dense chunks keep each
/// architecture's optimized `prefill`. An inactive row never overwrites the
/// logits of its most recent active chunk. The coordinator owns and drops
/// `state` on any later error, so a multi-call prefill cannot expose a
/// partially initialized session.
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
    let expected_mask_values = metadata::rectangle_size(batch, sequence, "prompt mask")?;
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
        let end = prefill_chunk_end(start, chunk_size, sequence);
        let active = (0..batch)
            .map(|row| {
                let (row_start, row_end) = row_span(row, batch, sequence, start, end);
                prompt_valid[row_start..row_end].iter().any(|&valid| valid)
            })
            .collect::<Vec<_>>();
        if active.iter().all(|&row_active| !row_active) {
            start = end;
            continue;
        }
        let token_ids = prompt.token_ids.clone().slice([0..batch, start..end]);
        let token_mask = prompt
            .token_mask
            .as_ref()
            .map(|mask| TokenMask::new(mask.as_tensor().clone().slice([0..batch, start..end])));
        let position_ids = prompt
            .position_ids
            .as_ref()
            .map(|positions| positions.clone().slice([0..batch, start..end]));
        // Architecture prefill contracts reject a completely inactive row,
        // whereas decode explicitly treats it as a cache no-op.
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
                MODEL_LOGITS,
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
    if let Some(mask) = &request.prompt.token_mask
        && mask.as_tensor().dims() != [batch, sequence]
    {
        return Err(Error::shape(
            "generation.prompt.token_mask",
            format!("[{batch}, {sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    if let Some(positions) = &request.prompt.position_ids
        && positions.dims() != [batch, sequence]
    {
        return Err(Error::shape(
            "generation.prompt.position_ids",
            format!("[{batch}, {sequence}]"),
            format!("{:?}", positions.dims()),
        ));
    }
    // One transfer reads the mask together with any explicit positions.
    let metadata::HostGrid {
        valid: prompt_valid,
        values: prompt_positions,
    } = metadata::read_grid(
        request
            .prompt
            .token_mask
            .as_ref()
            .map(|mask| mask.as_tensor().clone()),
        request.prompt.position_ids.clone(),
        batch,
        sequence,
        "prompt mask and positions",
    )?;
    let prompt_lengths = metadata::nonempty_row_lengths(&prompt_valid, batch, sequence, "prompt")?;
    let maximum_prompt = prompt_lengths.iter().copied().max().unwrap_or(0);
    let required_capacity =
        metadata::required_cache_capacity(maximum_prompt, request.generation.max_new_tokens)
            .ok_or_else(|| Error::Generation("required cache capacity overflows usize".into()))?;
    if request.cache.capacity() < required_capacity {
        return Err(Error::Generation(format!(
            "cache capacity {} is smaller than the required {required_capacity} for a {maximum_prompt}-token prompt and {} generated tokens",
            request.cache.capacity(),
            request.generation.max_new_tokens
        )));
    }

    let next_positions = prompt_positions
        .map(|values| {
            metadata::next_positions(
                &values,
                &prompt_valid,
                batch,
                sequence,
                request.generation.max_new_tokens,
                "prompt",
            )
        })
        .transpose()?;

    Ok(PreparedRequest {
        batch,
        prompt_lengths,
        prompt_valid,
        next_positions,
    })
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

/// Creates one explicit decode-position column on the selected backend.
fn position_column<B: Backend>(positions: &[i64], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(
        TensorData::new(positions.to_vec(), [positions.len(), 1]),
        device,
    )
}

/// Advances positions only for rows that committed a decode token.
///
/// The request preflight (`metadata::next_positions`) guarantees every
/// position sent during `max_new_tokens - 1` decode steps fits `i32`, so the
/// checked increment cannot fail for a validated request.
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
    let mut sequences = metadata::vec_with_capacity(batch, "completed output rows")?;
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
#[path = "causal_tests.rs"]
mod tests;
