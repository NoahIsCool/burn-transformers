//! Deterministic cached beam search for encoder-decoder language models.
//!
//! Beam search approximates
//! `argmax_y sum_t log p(y_t | y_<t, x)` by retaining the `K` greatest
//! unfinished prefixes after every decoder step. Finished hypotheses are kept
//! separately and ranked by
//! `score(y) = log p(y | x) / |y|^length_penalty`, where `|y|` counts only
//! tokens generated after the caller-supplied decoder prompt. This definition
//! makes the length convention explicit and avoids silently treating T5's
//! decoder-start token as model output.
//!
//! Encoder computation runs once. Its output and the external decoder cache are
//! expanded to a fixed `[source_batch * beams]` batch. At each later step,
//! [`DecodeState::reorder`] gathers or duplicates parent rows before selected
//! tokens enter [`Seq2SeqLanguageModel::decode`]. Finished source rows remain in
//! that rectangle as false-masked PAD no-ops, preserving other rows' progress
//! without mutating their cache lengths.
//!
//! This portable reference coordinator transfers log probabilities to the host
//! to perform an exact stable ordering over every beam-vocabulary pair. Model
//! tensors and caches stay on the backend. Backend-specific fused top-k search
//! can implement the same public semantics later without changing model code.

use burn::tensor::{
    ElementConversion, Int, Tensor, TensorData, activation::log_softmax, backend::Backend,
};
use serde::{Deserialize, Serialize};

use crate::{
    CausalLmInput, DecodeInput, DecodeState, EncoderOutput, Error, FinishReason, ForwardOptions,
    LogitsSelection, Result, Seq2SeqLanguageModel, TokenMask,
    architecture::{bool_tensor_from_values, select_bool_tensor},
    cache::CacheConfig,
};

/// Deterministic beam width, termination, and ranking controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BeamSearchConfig {
    /// Maximum number of tokens generated after the decoder prompt.
    pub max_new_tokens: usize,
    /// Number of live decoder prefixes retained for each source row.
    pub num_beams: usize,
    /// Number of ranked hypotheses returned for each source row.
    pub num_return_sequences: usize,
    /// Exponent in `cumulative_log_probability / generated_length^penalty`.
    pub length_penalty: f64,
    /// Stop a source row as soon as `num_beams` finished hypotheses exist.
    ///
    /// Following Transformers 5.2 eager semantics, only finished candidates
    /// ranked within the current top `num_beams` may enter that pool; the rest
    /// of the wider candidate frontier exists solely to find live successors.
    /// When false, search continues to the configured length bound so an
    /// unfinished prefix may still overtake an early EOS after length
    /// normalization.
    pub early_stopping: bool,
    /// Token IDs that finish a hypothesis after being appended.
    pub eos_token_ids: Vec<u32>,
    /// Token supplied with a false mask for inactive slots in rectangular calls.
    pub pad_token_id: u32,
}

impl Default for BeamSearchConfig {
    /// Returns conventional four-beam, single-result search controls.
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            num_beams: 4,
            num_return_sequences: 1,
            length_penalty: 1.0,
            early_stopping: false,
            eos_token_ids: Vec::new(),
            pad_token_id: 0,
        }
    }
}

impl BeamSearchConfig {
    /// Validates beam geometry and token IDs against a vocabulary.
    pub fn validate(&self, vocabulary_size: usize) -> Result<()> {
        if vocabulary_size == 0 {
            return Err(Error::Generation(
                "beam-search vocabulary_size must be greater than zero".into(),
            ));
        }
        if self.max_new_tokens == 0 {
            return Err(Error::Generation(
                "beam-search max_new_tokens must be greater than zero".into(),
            ));
        }
        if self.num_beams == 0 {
            return Err(Error::Generation(
                "beam-search num_beams must be greater than zero".into(),
            ));
        }
        if self.num_beams > vocabulary_size {
            return Err(Error::Generation(format!(
                "beam-search num_beams {} exceeds vocabulary size {vocabulary_size}",
                self.num_beams
            )));
        }
        if self.num_return_sequences == 0 || self.num_return_sequences > self.num_beams {
            return Err(Error::Generation(format!(
                "beam-search num_return_sequences must be in 1..={}, found {}",
                self.num_beams, self.num_return_sequences
            )));
        }
        if !self.length_penalty.is_finite() {
            return Err(Error::Generation(format!(
                "beam-search length_penalty must be finite, found {}",
                self.length_penalty
            )));
        }
        let maximum_normalizer = (self.max_new_tokens as f64).powf(self.length_penalty);
        if !maximum_normalizer.is_finite() || maximum_normalizer == 0.0 {
            return Err(Error::Generation(format!(
                "beam-search length_penalty {} is not numerically representable through length {}",
                self.length_penalty, self.max_new_tokens
            )));
        }
        if u32::try_from(vocabulary_size - 1).is_err() {
            return Err(Error::Generation(format!(
                "beam-search vocabulary size {vocabulary_size} exceeds the u32 token-ID domain"
            )));
        }
        if self.pad_token_id as usize >= vocabulary_size {
            return Err(Error::Generation(format!(
                "beam-search PAD token {} is outside vocabulary size {vocabulary_size}",
                self.pad_token_id
            )));
        }
        for &token in &self.eos_token_ids {
            if token as usize >= vocabulary_size {
                return Err(Error::Generation(format!(
                    "beam-search EOS token {token} is outside vocabulary size {vocabulary_size}"
                )));
            }
        }
        Ok(())
    }
}

/// Complete tensors and policies for one encoder-decoder beam search.
#[derive(Clone, Debug)]
pub struct Seq2SeqBeamSearchRequest<B: Backend> {
    /// Source tokens encoded exactly once before beam expansion.
    pub source: CausalLmInput<B>,
    /// Initial decoder prefix, commonly T5's one-token decoder start ID.
    pub decoder_prompt: CausalLmInput<B>,
    /// Fixed-size cache for `source_batch * num_beams` decoder rows.
    pub cache: CacheConfig,
    /// Beam width, length normalization, and stopping controls.
    pub search: BeamSearchConfig,
    /// Number of output categories in the decoder language head.
    pub vocabulary_size: usize,
}

/// One returned decoder hypothesis and its probability accounting.
#[derive(Clone, Debug, PartialEq)]
pub struct BeamSearchSequence {
    /// Count of valid tokens in the caller-supplied decoder prompt.
    pub decoder_prompt_length: usize,
    /// Generated token IDs, including a terminal EOS when selected.
    pub token_ids: Vec<u32>,
    /// Conditional log probability for each generated token.
    pub log_probabilities: Vec<f64>,
    /// Sum of all entries in [`Self::log_probabilities`].
    pub cumulative_log_probability: f64,
    /// Length-normalized score used to rank returned hypotheses.
    pub score: f64,
    /// Condition that finalized this hypothesis.
    pub finish_reason: FinishReason,
}

/// Ranked beam-search results grouped in original source-batch order.
#[derive(Clone, Debug, PartialEq)]
pub struct Seq2SeqBeamSearchOutput {
    /// Per-source hypotheses ordered from greatest to least beam score.
    pub sequences: Vec<Vec<BeamSearchSequence>>,
}

/// Runs deterministic cached beam search for any encoder-decoder architecture.
///
/// Equal candidate scores are ordered by parent beam and then token ID. Equal
/// finished scores are ordered by cumulative score, lexicographic token IDs,
/// and finally creation order. These rules make results reproducible even when
/// a backend's vocabulary sort does not define ties.
pub fn generate_seq2seq_beam_search<B, M>(
    model: &M,
    request: Seq2SeqBeamSearchRequest<B>,
) -> Result<Seq2SeqBeamSearchOutput>
where
    B: Backend,
    M: Seq2SeqLanguageModel<B>,
{
    let prepared = PreparedRequest::new(&request)?;
    let device = request.source.token_ids.device();
    let options = ForwardOptions {
        logits: LogitsSelection::Last,
        ..ForwardOptions::default()
    };

    let encoded = model.encode(request.source, &options)?;
    validate_encoder_batch(&encoded, prepared.source_batch)?;
    let expansion = beam_expansion_indices(prepared.source_batch, request.search.num_beams)?;
    let mut encoder = select_encoder(encoded, &expansion)?;
    let decoder_prompt = select_input(request.decoder_prompt, &expansion)?;
    let mut state = model.new_decode_state(request.cache, prepared.expanded_batch, &device)?;
    let prefill = model.prefill_decoder(&encoder, decoder_prompt, &mut state, &options)?;
    encoder = prefill.encoder;
    validate_encoder_batch(&encoder, prepared.expanded_batch)?;
    let mut logits = extract_last_logits(
        prefill.decoder.logits,
        prepared.expanded_batch,
        request.vocabulary_size,
    )?;

    let mut active = initial_beams(&prepared)?;
    let mut completed = empty_completed(prepared.source_batch)?;
    let mut done = vec![false; prepared.source_batch];
    let mut ordinal = 0_usize;

    for step in 0..request.search.max_new_tokens {
        let active_rows = active
            .iter()
            .enumerate()
            .map(|(row, beam)| {
                let source = row / request.search.num_beams;
                beam.is_some() && !done[source]
            })
            .collect::<Vec<_>>();
        let probabilities = host_log_probabilities(
            logits,
            &active_rows,
            prepared.expanded_batch,
            request.vocabulary_size,
        )?;
        let at_length_limit = step + 1 == request.search.max_new_tokens;
        let mut next_active = vec![None; prepared.expanded_batch];
        let mut parent_indices = (0..prepared.expanded_batch).collect::<Vec<_>>();
        let mut next_tokens = vec![i64::from(request.search.pad_token_id); prepared.expanded_batch];
        let mut next_valid = vec![false; prepared.expanded_batch];
        let mut decode_positions = prepared
            .has_explicit_decoder_positions
            .then(|| vec![0_i64; prepared.expanded_batch]);

        for (source, source_done) in done.iter_mut().enumerate() {
            if *source_done {
                continue;
            }
            let candidates = candidates_for_source(
                source,
                request.search.num_beams,
                request.vocabulary_size,
                &active,
                &probabilities,
            )?;
            if candidates.is_empty() {
                return Err(Error::Generation(format!(
                    "beam-search source row {source} has no finite candidate"
                )));
            }

            let mut selected = Vec::with_capacity(request.search.num_beams);
            let frontier = candidate_frontier_size(
                request.search.num_beams,
                request.search.eos_token_ids.len(),
                candidates.len(),
            )?;
            for (candidate_rank, candidate) in candidates.into_iter().take(frontier).enumerate() {
                let is_eos = request.search.eos_token_ids.contains(&candidate.token_id);
                if is_eos || at_length_limit {
                    // Transformers 5.2 masks finished candidates below the
                    // current beam width. Those extra frontier rows are only
                    // backups for selecting enough non-finished successors.
                    if candidate_rank >= request.search.num_beams {
                        continue;
                    }
                    let reason = if is_eos {
                        FinishReason::EndOfSequence {
                            token_id: candidate.token_id,
                        }
                    } else {
                        FinishReason::Length
                    };
                    let hypothesis = extend_candidate(&active, &candidate, Some(reason), ordinal)?;
                    ordinal = ordinal.checked_add(1).ok_or_else(|| {
                        Error::Generation("beam-search hypothesis ordinal overflow".into())
                    })?;
                    completed
                        .get_mut(source)
                        .ok_or_else(|| {
                            Error::Generation("completed beam batch became ragged".into())
                        })?
                        .push(hypothesis);
                } else if selected.len() < request.search.num_beams {
                    selected.push(candidate);
                }
            }
            rank_and_prune(
                completed.get_mut(source).ok_or_else(|| {
                    Error::Generation("completed beam batch became ragged".into())
                })?,
                request.search.num_beams,
                request.search.length_penalty,
            );

            if at_length_limit
                || (request.search.early_stopping
                    && completed
                        .get(source)
                        .ok_or_else(|| {
                            Error::Generation("completed beam batch became ragged".into())
                        })?
                        .len()
                        >= request.search.num_beams)
                || selected.is_empty()
            {
                *source_done = true;
                continue;
            }

            for (slot, candidate) in selected.into_iter().enumerate() {
                let destination = source
                    .checked_mul(request.search.num_beams)
                    .and_then(|offset| offset.checked_add(slot))
                    .ok_or_else(|| Error::Generation("beam destination index overflow".into()))?;
                let parent = active
                    .get(candidate.parent_global)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| Error::Generation("selected beam parent disappeared".into()))?;
                if let Some(positions) = &mut decode_positions {
                    positions[destination] = parent.next_position.ok_or_else(|| {
                        Error::Generation("explicit beam position disappeared".into())
                    })?;
                }
                parent_indices[destination] = candidate.parent_global;
                next_tokens[destination] = i64::from(candidate.token_id);
                next_valid[destination] = true;
                next_active[destination] =
                    Some(extend_candidate(&active, &candidate, None, ordinal)?);
                ordinal = ordinal.checked_add(1).ok_or_else(|| {
                    Error::Generation("beam-search hypothesis ordinal overflow".into())
                })?;
            }
        }

        if done.iter().all(|finished| *finished) {
            break;
        }

        state.reorder(&parent_indices)?;
        encoder = select_encoder(encoder, &parent_indices)?;
        let token_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(next_tokens, [prepared.expanded_batch, 1]),
            &device,
        );
        let valid = bool_tensor_from_values(next_valid, [prepared.expanded_batch, 1], &device);
        let position_ids = decode_positions.map(|positions| {
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(positions, [prepared.expanded_batch, 1]),
                &device,
            )
        });
        active = next_active;
        let decoded = model.decode(
            &encoder,
            DecodeInput {
                token_ids,
                position_ids,
                token_mask: Some(TokenMask::new(valid)),
            },
            &mut state,
            &options,
        )?;
        encoder = decoded.encoder;
        validate_encoder_batch(&encoder, prepared.expanded_batch)?;
        logits = extract_last_logits(
            decoded.decoder.logits,
            prepared.expanded_batch,
            request.vocabulary_size,
        )?;
    }

    finish_output(completed, &prepared.decoder_prompt_lengths, &request.search)
}

/// Validated dimensions and position metadata derived before model mutation.
struct PreparedRequest {
    source_batch: usize,
    expanded_batch: usize,
    decoder_prompt_lengths: Vec<usize>,
    initial_next_positions: Option<Vec<i64>>,
    has_explicit_decoder_positions: bool,
}

impl PreparedRequest {
    /// Checks public request geometry and derives initial per-source beam state.
    fn new<B: Backend>(request: &Seq2SeqBeamSearchRequest<B>) -> Result<Self> {
        request.search.validate(request.vocabulary_size)?;
        let [source_batch, source_sequence] = request.source.token_ids.dims();
        let [decoder_batch, decoder_sequence] = request.decoder_prompt.token_ids.dims();
        if source_batch == 0 || source_sequence == 0 {
            return Err(Error::shape(
                "beam_search.source",
                "non-empty [batch, sequence]",
                format!("[{source_batch}, {source_sequence}]"),
            ));
        }
        if decoder_batch != source_batch || decoder_sequence == 0 {
            return Err(Error::shape(
                "beam_search.decoder_prompt",
                format!("[{source_batch}, nonzero_sequence]"),
                format!("[{decoder_batch}, {decoder_sequence}]"),
            ));
        }
        validate_input_metadata(&request.source, source_batch, source_sequence, "source")?;
        validate_input_metadata(
            &request.decoder_prompt,
            decoder_batch,
            decoder_sequence,
            "decoder_prompt",
        )?;
        let decoder_valid = mask_values(
            request.decoder_prompt.token_mask.as_ref(),
            decoder_batch,
            decoder_sequence,
            "decoder prompt mask",
        )?;
        let decoder_prompt_lengths = nonempty_row_lengths(
            &decoder_valid,
            decoder_batch,
            decoder_sequence,
            "decoder prompt",
        )?;
        let expanded_batch = source_batch
            .checked_mul(request.search.num_beams)
            .ok_or_else(|| Error::Generation("expanded beam batch overflows usize".into()))?;
        if request.cache.sequence_count() != expanded_batch {
            return Err(Error::Generation(format!(
                "beam-search cache tracks {} rows, expected source batch {source_batch} * {} beams = {expanded_batch}",
                request.cache.sequence_count(),
                request.search.num_beams
            )));
        }
        let maximum_prompt = decoder_prompt_lengths
            .iter()
            .copied()
            .max()
            .ok_or_else(|| {
                Error::Generation("decoder prompt lengths unexpectedly became empty".into())
            })?;
        let required_capacity = maximum_prompt
            .checked_add(request.search.max_new_tokens)
            .ok_or_else(|| {
                Error::Generation("beam-search cache capacity overflows usize".into())
            })?;
        if request.cache.capacity() < required_capacity {
            return Err(Error::Generation(format!(
                "beam-search cache capacity {} is smaller than required decoder prompt {maximum_prompt} + {} generated tokens = {required_capacity}",
                request.cache.capacity(),
                request.search.max_new_tokens
            )));
        }

        let initial_next_positions = request
            .decoder_prompt
            .position_ids
            .clone()
            .map(|positions| {
                tensor_i64_values(positions, "decoder prompt positions").and_then(|values| {
                    next_positions(
                        &values,
                        &decoder_valid,
                        decoder_batch,
                        decoder_sequence,
                        request.search.max_new_tokens,
                    )
                })
            })
            .transpose()?;
        Ok(Self {
            source_batch,
            expanded_batch,
            decoder_prompt_lengths,
            has_explicit_decoder_positions: initial_next_positions.is_some(),
            initial_next_positions,
        })
    }
}

/// Internal unfinished or finalized hypothesis.
#[derive(Clone, Debug)]
struct BeamHypothesis {
    token_ids: Vec<u32>,
    log_probabilities: Vec<f64>,
    cumulative_log_probability: f64,
    next_position: Option<i64>,
    finish_reason: Option<FinishReason>,
    ordinal: usize,
}

/// One not-yet-materialized child from a model probability row.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    parent_global: usize,
    parent_slot: usize,
    token_id: u32,
    log_probability: f64,
    cumulative_log_probability: f64,
}

/// Creates one score-zero live prefix per source and dormant spare slots.
fn initial_beams(prepared: &PreparedRequest) -> Result<Vec<Option<BeamHypothesis>>> {
    let beams_per_source = prepared.expanded_batch / prepared.source_batch;
    let mut active = vec![None; prepared.expanded_batch];
    for source in 0..prepared.source_batch {
        let row = source
            .checked_mul(beams_per_source)
            .ok_or_else(|| Error::Generation("initial beam index overflow".into()))?;
        active[row] = Some(BeamHypothesis {
            token_ids: Vec::new(),
            log_probabilities: Vec::new(),
            cumulative_log_probability: 0.0,
            next_position: prepared
                .initial_next_positions
                .as_ref()
                .and_then(|positions| positions.get(source).copied()),
            finish_reason: None,
            ordinal: source,
        });
    }
    Ok(active)
}

/// Allocates one empty finished-hypothesis pool per source row.
fn empty_completed(batch: usize) -> Result<Vec<Vec<BeamHypothesis>>> {
    let mut completed = Vec::new();
    completed.try_reserve_exact(batch).map_err(|_| {
        Error::Generation(format!(
            "failed to allocate completed beam pools for batch size {batch}"
        ))
    })?;
    completed.resize_with(batch, Vec::new);
    Ok(completed)
}

/// Enumerates and stably ranks every finite beam-vocabulary continuation.
fn candidates_for_source(
    source: usize,
    beams: usize,
    vocabulary: usize,
    active: &[Option<BeamHypothesis>],
    probabilities: &[f64],
) -> Result<Vec<Candidate>> {
    let start = source
        .checked_mul(beams)
        .ok_or_else(|| Error::Generation("beam source offset overflow".into()))?;
    let mut candidates = Vec::new();
    for parent_slot in 0..beams {
        let parent_global = start
            .checked_add(parent_slot)
            .ok_or_else(|| Error::Generation("beam parent index overflow".into()))?;
        let Some(parent) = active.get(parent_global).and_then(Option::as_ref) else {
            continue;
        };
        let probability_offset = parent_global
            .checked_mul(vocabulary)
            .ok_or_else(|| Error::Generation("beam probability offset overflow".into()))?;
        let probability_end = probability_offset
            .checked_add(vocabulary)
            .ok_or_else(|| Error::Generation("beam probability row end overflow".into()))?;
        let row = probabilities
            .get(probability_offset..probability_end)
            .ok_or_else(|| Error::Generation("beam probability matrix became ragged".into()))?;
        candidates.try_reserve(vocabulary).map_err(|_| {
            Error::Generation(format!("failed to allocate {vocabulary} beam candidates"))
        })?;
        for (token, &log_probability) in row.iter().enumerate() {
            if log_probability == f64::NEG_INFINITY {
                continue;
            }
            if !log_probability.is_finite() {
                return Err(Error::Generation(format!(
                    "beam {parent_global} token {token} produced non-finite log probability {log_probability}"
                )));
            }
            let token_id = u32::try_from(token).map_err(|_| {
                Error::Generation(format!(
                    "vocabulary token {token} exceeds the u32 token-ID domain"
                ))
            })?;
            let cumulative_log_probability = parent.cumulative_log_probability + log_probability;
            if !cumulative_log_probability.is_finite() {
                return Err(Error::Generation(format!(
                    "beam {parent_global} token {token} overflowed cumulative log probability"
                )));
            }
            candidates.push(Candidate {
                parent_global,
                parent_slot,
                token_id,
                log_probability,
                cumulative_log_probability,
            });
        }
    }
    candidates.sort_by(candidate_order);
    Ok(candidates)
}

/// Defines deterministic candidate order independently of backend sort rules.
fn candidate_order(left: &Candidate, right: &Candidate) -> core::cmp::Ordering {
    right
        .cumulative_log_probability
        .total_cmp(&left.cumulative_log_probability)
        .then_with(|| left.parent_slot.cmp(&right.parent_slot))
        .then_with(|| left.token_id.cmp(&right.token_id))
}

/// Sizes the conventional top frontier examined for EOS and live prefixes.
///
/// One beam width supplies live children and each distinct EOS ID can consume
/// another width. A minimum factor of two matches ordinary single-EOS search.
fn candidate_frontier_size(beams: usize, eos_count: usize, available: usize) -> Result<usize> {
    let multiplier = eos_count
        .checked_add(1)
        .ok_or_else(|| Error::Generation("beam-search EOS count overflows usize".into()))?
        .max(2);
    beams
        .checked_mul(multiplier)
        .map(|frontier| frontier.min(available))
        .ok_or_else(|| Error::Generation("beam-search candidate frontier overflows usize".into()))
}

/// Materializes a candidate by extending its parent's token and score vectors.
fn extend_candidate(
    active: &[Option<BeamHypothesis>],
    candidate: &Candidate,
    finish_reason: Option<FinishReason>,
    ordinal: usize,
) -> Result<BeamHypothesis> {
    let parent = active
        .get(candidate.parent_global)
        .and_then(Option::as_ref)
        .ok_or_else(|| Error::Generation("beam candidate parent is unavailable".into()))?;
    let mut token_ids = parent.token_ids.clone();
    token_ids
        .try_reserve(1)
        .map_err(|_| Error::Generation("failed to extend beam token history".into()))?;
    token_ids.push(candidate.token_id);
    let mut log_probabilities = parent.log_probabilities.clone();
    log_probabilities
        .try_reserve(1)
        .map_err(|_| Error::Generation("failed to extend beam probability history".into()))?;
    log_probabilities.push(candidate.log_probability);
    let next_position = parent
        .next_position
        .map(|position| {
            position
                .checked_add(1)
                .ok_or_else(|| Error::Generation("explicit decoder position overflow".into()))
        })
        .transpose()?;
    Ok(BeamHypothesis {
        token_ids,
        log_probabilities,
        cumulative_log_probability: candidate.cumulative_log_probability,
        next_position,
        finish_reason,
        ordinal,
    })
}

/// Sorts finished hypotheses by public score and retains only the beam width.
fn rank_and_prune(hypotheses: &mut Vec<BeamHypothesis>, beams: usize, length_penalty: f64) {
    hypotheses.sort_by(|left, right| completed_order(left, right, length_penalty));
    hypotheses.truncate(beams);
}

/// Compares finished hypotheses using documented deterministic tie breakers.
fn completed_order(
    left: &BeamHypothesis,
    right: &BeamHypothesis,
    length_penalty: f64,
) -> core::cmp::Ordering {
    normalized_score(right, length_penalty)
        .total_cmp(&normalized_score(left, length_penalty))
        .then_with(|| {
            right
                .cumulative_log_probability
                .total_cmp(&left.cumulative_log_probability)
        })
        .then_with(|| left.token_ids.cmp(&right.token_ids))
        .then_with(|| left.ordinal.cmp(&right.ordinal))
}

/// Computes a finite score for a non-empty generated hypothesis.
fn normalized_score(hypothesis: &BeamHypothesis, length_penalty: f64) -> f64 {
    hypothesis.cumulative_log_probability / (hypothesis.token_ids.len() as f64).powf(length_penalty)
}

/// Converts internal finished pools into stable public output.
fn finish_output(
    mut completed: Vec<Vec<BeamHypothesis>>,
    prompt_lengths: &[usize],
    search: &BeamSearchConfig,
) -> Result<Seq2SeqBeamSearchOutput> {
    if completed.len() != prompt_lengths.len() {
        return Err(Error::Generation(
            "beam-search result batch does not match decoder prompt batch".into(),
        ));
    }
    let mut sequences = Vec::new();
    sequences
        .try_reserve_exact(completed.len())
        .map_err(|_| Error::Generation("failed to allocate beam-search output batch".into()))?;
    for (source, hypotheses) in completed.iter_mut().enumerate() {
        rank_and_prune(hypotheses, search.num_beams, search.length_penalty);
        if hypotheses.len() < search.num_return_sequences {
            return Err(Error::Generation(format!(
                "beam-search source row {source} produced {} finished hypotheses, fewer than requested {}",
                hypotheses.len(),
                search.num_return_sequences
            )));
        }
        let prompt_length = *prompt_lengths
            .get(source)
            .ok_or_else(|| Error::Generation("decoder prompt lengths became ragged".into()))?;
        let mut returned = Vec::new();
        returned
            .try_reserve_exact(search.num_return_sequences)
            .map_err(|_| Error::Generation("failed to allocate returned beam hypotheses".into()))?;
        for hypothesis in hypotheses.drain(..search.num_return_sequences) {
            let score = normalized_score(&hypothesis, search.length_penalty);
            let finish_reason = hypothesis.finish_reason.ok_or_else(|| {
                Error::Generation("unfinished hypothesis reached beam output".into())
            })?;
            returned.push(BeamSearchSequence {
                decoder_prompt_length: prompt_length,
                token_ids: hypothesis.token_ids,
                log_probabilities: hypothesis.log_probabilities,
                cumulative_log_probability: hypothesis.cumulative_log_probability,
                score,
                finish_reason,
            });
        }
        sequences.push(returned);
    }
    Ok(Seq2SeqBeamSearchOutput { sequences })
}

/// Applies a batch gather to every tensor carried by an encoder output.
fn select_encoder<B: Backend>(
    encoder: EncoderOutput<B>,
    indices: &[usize],
) -> Result<EncoderOutput<B>> {
    let selector = batch_selector::<B>(indices, &encoder.hidden_state.device())?;
    Ok(EncoderOutput {
        hidden_state: encoder.hidden_state.select(0, selector.clone()),
        token_mask: encoder.token_mask.map(|mask| {
            TokenMask::new(select_bool_tensor(mask.into_tensor(), 0, selector.clone()))
        }),
        hidden_states: encoder
            .hidden_states
            .into_iter()
            .map(|hidden| hidden.select(0, selector.clone()))
            .collect(),
        attentions: encoder
            .attentions
            .into_iter()
            .map(|attention| attention.select(0, selector.clone()))
            .collect(),
    })
}

/// Applies a batch gather to a token input and all optional metadata.
fn select_input<B: Backend>(
    input: CausalLmInput<B>,
    indices: &[usize],
) -> Result<CausalLmInput<B>> {
    let selector = batch_selector::<B>(indices, &input.token_ids.device())?;
    Ok(CausalLmInput {
        token_ids: input.token_ids.select(0, selector.clone()),
        token_mask: input.token_mask.map(|mask| {
            TokenMask::new(select_bool_tensor(mask.into_tensor(), 0, selector.clone()))
        }),
        position_ids: input
            .position_ids
            .map(|positions| positions.select(0, selector)),
    })
}

/// Builds the initial repeated source-row selector in group-major order.
fn beam_expansion_indices(batch: usize, beams: usize) -> Result<Vec<usize>> {
    let count = batch
        .checked_mul(beams)
        .ok_or_else(|| Error::Generation("beam expansion count overflows usize".into()))?;
    let mut indices = Vec::new();
    indices.try_reserve_exact(count).map_err(|_| {
        Error::Generation(format!("failed to allocate {count} beam expansion indices"))
    })?;
    for source in 0..batch {
        indices.extend(std::iter::repeat_n(source, beams));
    }
    Ok(indices)
}

/// Converts host indices to Burn's portable signed batch selector.
fn batch_selector<B: Backend>(indices: &[usize], device: &B::Device) -> Result<Tensor<B, 1, Int>> {
    let values = indices
        .iter()
        .map(|&index| {
            i64::try_from(index).map_err(|_| {
                Error::Generation(format!(
                    "beam batch index {index} exceeds the signed tensor-index domain"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::from_data(
        TensorData::new(values, [indices.len()]),
        device,
    ))
}

/// Validates optional mask and position rectangles before model execution.
fn validate_input_metadata<B: Backend>(
    input: &CausalLmInput<B>,
    batch: usize,
    sequence: usize,
    label: &str,
) -> Result<()> {
    if let Some(mask) = &input.token_mask
        && mask.as_tensor().dims() != [batch, sequence]
    {
        return Err(Error::shape(
            format!("beam_search.{label}.token_mask"),
            format!("[{batch}, {sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    if let Some(positions) = &input.position_ids
        && positions.dims() != [batch, sequence]
    {
        return Err(Error::shape(
            format!("beam_search.{label}.position_ids"),
            format!("[{batch}, {sequence}]"),
            format!("{:?}", positions.dims()),
        ));
    }
    Ok(())
}

/// Verifies model-produced encoder batch geometry.
fn validate_encoder_batch<B: Backend>(encoder: &EncoderOutput<B>, batch: usize) -> Result<()> {
    let [observed_batch, source, hidden] = encoder.hidden_state.dims();
    if observed_batch != batch || source == 0 || hidden == 0 {
        return Err(Error::shape(
            "beam_search.encoder_output",
            format!("[{batch}, nonzero_source, nonzero_hidden]"),
            format!("{:?}", encoder.hidden_state.dims()),
        ));
    }
    if let Some(mask) = &encoder.token_mask
        && mask.as_tensor().dims() != [batch, source]
    {
        return Err(Error::shape(
            "beam_search.encoder_output.token_mask",
            format!("[{batch}, {source}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    for (index, state) in encoder.hidden_states.iter().enumerate() {
        if state.dims() != [batch, source, hidden] {
            return Err(Error::shape(
                format!("beam_search.encoder_output.hidden_states[{index}]"),
                format!("[{batch}, {source}, {hidden}]"),
                format!("{:?}", state.dims()),
            ));
        }
    }
    for (index, attention) in encoder.attentions.iter().enumerate() {
        let [attention_batch, heads, query, key] = attention.dims();
        if attention_batch != batch || heads == 0 || query != source || key != source {
            return Err(Error::shape(
                format!("beam_search.encoder_output.attentions[{index}]"),
                format!("[{batch}, nonzero_heads, {source}, {source}]"),
                format!("{:?}", attention.dims()),
            ));
        }
    }
    Ok(())
}

/// Extracts `[beam_batch, vocabulary]` scores from last-logit model output.
fn extract_last_logits<B: Backend>(
    logits: Tensor<B, 3>,
    batch: usize,
    vocabulary: usize,
) -> Result<Tensor<B, 2>> {
    if logits.dims() != [batch, 1, vocabulary] {
        return Err(Error::shape(
            "beam_search.model_logits",
            format!("[{batch}, 1, {vocabulary}]"),
            format!("{:?}", logits.dims()),
        ));
    }
    Ok(logits.squeeze_dim::<2>(1))
}

/// Computes row-wise log-softmax and transfers the exact candidate matrix.
fn host_log_probabilities<B: Backend>(
    logits: Tensor<B, 2>,
    active: &[bool],
    batch: usize,
    vocabulary: usize,
) -> Result<Vec<f64>> {
    if logits.dims() != [batch, vocabulary] || active.len() != batch {
        return Err(Error::Generation(
            "beam-search active metadata does not match model logits".into(),
        ));
    }
    let device = logits.device();
    let active_mask =
        bool_tensor_from_values(active.to_vec(), [batch, 1], &device).expand([batch, vocabulary]);
    log_softmax(logits.mask_fill(active_mask.bool_not(), 0.0), 1)
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to read beam-search log probabilities from backend: {error}"
            ))
        })?
        .to_vec::<B::FloatElem>()
        .map(|values| {
            values
                .into_iter()
                .map(|value| value.elem::<f64>())
                .collect()
        })
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to read beam-search log probabilities from backend: {error}"
            ))
        })
}

/// Materializes a token mask or an all-valid rectangle on the host.
fn mask_values<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
    label: &str,
) -> Result<Vec<bool>> {
    match mask {
        Some(mask) => mask
            .as_tensor()
            .clone()
            .int()
            .try_into_data()
            .map_err(|error| Error::BackendCapability(format!("failed to read {label}: {error}")))?
            .to_vec::<B::IntElem>()
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| value.elem::<i64>() != 0)
                    .collect()
            })
            .map_err(|error| Error::BackendCapability(format!("failed to read {label}: {error}"))),
        None => batch
            .checked_mul(sequence)
            .map(|count| vec![true; count])
            .ok_or_else(|| Error::Generation(format!("{label} size overflows usize"))),
    }
}

/// Counts valid entries and rejects fully masked prompt rows.
fn nonempty_row_lengths(
    valid: &[bool],
    batch: usize,
    sequence: usize,
    label: &str,
) -> Result<Vec<usize>> {
    let expected = batch
        .checked_mul(sequence)
        .ok_or_else(|| Error::Generation(format!("{label} size overflows usize")))?;
    if valid.len() != expected {
        return Err(Error::Generation(format!(
            "{label} mask contains {} entries, expected {expected}",
            valid.len()
        )));
    }
    let mut lengths = Vec::with_capacity(batch);
    for (row, values) in valid.chunks_exact(sequence).enumerate() {
        let length = values.iter().filter(|&&value| value).count();
        if length == 0 {
            return Err(Error::Generation(format!(
                "{label} row {row} has no valid tokens"
            )));
        }
        lengths.push(length);
    }
    Ok(lengths)
}

/// Finds the next explicit decoder position and preflights signed overflow.
fn next_positions(
    positions: &[i64],
    valid: &[bool],
    batch: usize,
    sequence: usize,
    max_new_tokens: usize,
) -> Result<Vec<i64>> {
    let expected = batch
        .checked_mul(sequence)
        .ok_or_else(|| Error::Generation("decoder position size overflows usize".into()))?;
    if positions.len() != expected || valid.len() != expected {
        return Err(Error::Generation(
            "decoder positions and token mask are not rectangular".into(),
        ));
    }
    let additional = i64::try_from(max_new_tokens).map_err(|_| {
        Error::Generation("beam-search length exceeds signed position domain".into())
    })?;
    let mut next = Vec::with_capacity(batch);
    for row in 0..batch {
        let offset = row
            .checked_mul(sequence)
            .ok_or_else(|| Error::Generation("decoder position offset overflow".into()))?;
        let end = offset
            .checked_add(sequence)
            .ok_or_else(|| Error::Generation("decoder position row end overflow".into()))?;
        let row_positions = positions
            .get(offset..end)
            .ok_or_else(|| Error::Generation("decoder position rows became ragged".into()))?;
        let row_valid = valid
            .get(offset..end)
            .ok_or_else(|| Error::Generation("decoder mask rows became ragged".into()))?;
        let last = row_positions
            .iter()
            .zip(row_valid)
            .rev()
            .find_map(|(&position, &is_valid)| is_valid.then_some(position))
            .ok_or_else(|| Error::Generation(format!("decoder prompt row {row} is empty")))?;
        if last < 0 {
            return Err(Error::Generation(format!(
                "decoder prompt row {row} has negative last position {last}"
            )));
        }
        let maximum = last.checked_add(additional).ok_or_else(|| {
            Error::Generation(format!("decoder prompt row {row} position overflow"))
        })?;
        if maximum > i64::from(i32::MAX) {
            return Err(Error::Generation(format!(
                "decoder prompt row {row} position {maximum} exceeds Burn's portable signed index range"
            )));
        }
        next.push(last + 1);
    }
    Ok(next)
}

/// Copies an integer tensor into portable host values with context.
fn tensor_i64_values<B: Backend, const D: usize>(
    tensor: Tensor<B, D, Int>,
    label: &str,
) -> Result<Vec<i64>> {
    tensor
        .try_into_data()
        .map_err(|error| Error::BackendCapability(format!("failed to read {label}: {error}")))?
        .to_vec::<B::IntElem>()
        .map(|values| {
            values
                .into_iter()
                .map(|value| value.elem::<i64>())
                .collect()
        })
        .map_err(|error| Error::BackendCapability(format!("failed to read {label}: {error}")))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use burn::{
        backend::Flex,
        module::{Module, Param},
        tensor::{
            Bool, ElementConversion, Int, Tensor, TensorData,
            backend::{Backend, BackendTypes},
        },
    };

    use super::{
        BeamHypothesis, BeamSearchConfig, Seq2SeqBeamSearchRequest, completed_order,
        generate_seq2seq_beam_search,
    };
    use crate::{
        ArchitectureConfig, CausalLmInput, CausalLmOutput, DecodeInput, DecodeState, EncoderOutput,
        Error, FinishReason, ForwardOptions, Result, Seq2SeqInput, Seq2SeqLanguageModel,
        Seq2SeqOutput, TokenMask, cache::CacheConfig,
    };

    type TestBackend = Flex<f32, i32>;
    type TestDevice = <TestBackend as BackendTypes>::Device;

    /// Minimal configuration for the coordinator's scripted model.
    #[derive(Clone, Debug)]
    struct ScriptedConfig {
        vocabulary_size: usize,
        emit_nonfinite: bool,
    }

    impl ArchitectureConfig for ScriptedConfig {
        const MODEL_TYPES: &'static [&'static str] = &["beam_search_script"];

        /// The test configuration only requires a non-empty vocabulary.
        fn validate(&self) -> Result<()> {
            if self.vocabulary_size == 0 {
                Err(Error::configuration(
                    "beam_search_script.vocabulary_size",
                    "must be nonzero",
                ))
            } else {
                Ok(())
            }
        }
    }

    /// Observable lifecycle events emitted by the scripted cache and model.
    #[derive(Debug, Default)]
    struct ScriptTrace {
        reorders: Vec<Vec<usize>>,
        decode_tokens: Vec<Vec<i64>>,
        decode_valid: Vec<Vec<bool>>,
        decode_positions: Vec<Option<Vec<i64>>>,
    }

    /// Fixed-batch logical state used to prove coordinator reorder behavior.
    #[derive(Debug)]
    struct ScriptedState<B: Backend> {
        batch: usize,
        capacity: usize,
        lengths: Vec<usize>,
        device: B::Device,
        trace: Arc<Mutex<ScriptTrace>>,
    }

    impl<B: Backend> DecodeState<B> for ScriptedState<B> {
        /// Returns the configured fixed beam batch.
        fn batch_size(&self) -> usize {
            self.batch
        }

        /// Returns the configured logical target capacity.
        fn capacity(&self) -> usize {
            self.capacity
        }

        /// Returns committed decoder lengths after false-mask no-ops.
        fn lengths(&self) -> &[usize] {
            &self.lengths
        }

        /// Clears all scripted logical lengths.
        fn reset(&mut self) -> Result<()> {
            self.lengths.fill(0);
            Ok(())
        }

        /// Reorders or duplicates lengths and records exact parent indices.
        fn reorder(&mut self, indices: &[usize]) -> Result<()> {
            if indices.len() != self.batch {
                return Err(Error::Cache(format!(
                    "scripted reorder expected {} indices, found {}",
                    self.batch,
                    indices.len()
                )));
            }
            let next = indices
                .iter()
                .map(|&index| {
                    self.lengths.get(index).copied().ok_or_else(|| {
                        Error::Cache(format!("scripted reorder index {index} is out of bounds"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            self.lengths = next;
            self.trace
                .lock()
                .expect("script trace mutex")
                .reorders
                .push(indices.to_vec());
            Ok(())
        }

        /// Rewinds every scripted logical length.
        fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
            if lengths.len() != self.batch
                || lengths
                    .iter()
                    .zip(&self.lengths)
                    .any(|(next, current)| next > current)
            {
                return Err(Error::Cache("invalid scripted truncation".into()));
            }
            self.lengths.clone_from_slice(lengths);
            Ok(())
        }

        /// Returns the backend allocation device.
        fn device(&self) -> &B::Device {
            &self.device
        }
    }

    /// Tiny deterministic seq2seq graph whose logits expose beam decisions.
    #[derive(Module, Debug)]
    struct ScriptedModel<B: Backend> {
        dummy: Param<Tensor<B, 1>>,
        #[module(skip)]
        config: ScriptedConfig,
        #[module(skip)]
        trace: Arc<Mutex<ScriptTrace>>,
    }

    impl<B: Backend> ScriptedModel<B> {
        /// Initializes one parameter so Burn can derive the full Module contract.
        fn new(config: ScriptedConfig, device: &B::Device) -> Self {
            Self {
                dummy: Param::from_tensor(Tensor::zeros([1], device)),
                config,
                trace: Arc::new(Mutex::new(ScriptTrace::default())),
            }
        }

        /// Creates a model output from one flat row-major logits buffer.
        fn output(
            &self,
            logits: Vec<f32>,
            batch: usize,
            encoder: EncoderOutput<B>,
            device: &B::Device,
        ) -> Seq2SeqOutput<B> {
            Seq2SeqOutput {
                decoder: CausalLmOutput {
                    logits: Tensor::from_data(
                        TensorData::new(logits, [batch, 1, self.config.vocabulary_size]),
                        device,
                    ),
                    final_hidden_state: None,
                    hidden_states: Vec::new(),
                    attentions: Vec::new(),
                    routers: Vec::new(),
                    load_balancing_loss: None,
                },
                encoder,
                cross_attentions: Vec::new(),
            }
        }

        /// Returns first-step logits keyed by the first encoded source token.
        fn prefill_logits(&self, encoder: &EncoderOutput<B>) -> Result<Vec<f32>> {
            let [batch, source, _] = encoder.hidden_state.dims();
            let markers = encoder
                .hidden_state
                .clone()
                .slice([0..batch, 0..1.min(source), 0..1])
                .reshape([batch])
                .into_data()
                .to_vec::<B::FloatElem>()
                .map_err(|error| Error::BackendCapability(error.to_string()))?;
            let mut logits = Vec::with_capacity(batch * self.config.vocabulary_size);
            for marker in markers {
                if self.config.emit_nonfinite {
                    logits.extend([f32::NAN, 0.0, 0.0, 0.0, 0.0]);
                } else if marker.elem::<f64>() < 3.5 {
                    // A second source marker favors a non-EOS continuation.
                    logits.extend([0.0, -1.0, -1.0, 5.0, 1.0]);
                } else {
                    // IDs one and two are an intentional stable tie.
                    logits.extend([0.0, 4.0, 4.0, 1.0, 3.0]);
                }
            }
            Ok(logits)
        }

        /// Returns continuation logits keyed by each selected decoder token.
        fn decode_logits(&self, tokens: &[i64], valid: &[bool]) -> Vec<f32> {
            let mut logits = Vec::with_capacity(tokens.len() * self.config.vocabulary_size);
            for (&token, &is_valid) in tokens.iter().zip(valid) {
                if !is_valid {
                    logits.extend([0.0; 5]);
                } else {
                    match token {
                        1 => logits.extend([0.0, 0.0, 1.0, 4.0, 5.0]),
                        2 => logits.extend([0.0, 0.0, 1.0, 5.0, 4.0]),
                        _ => logits.extend([0.0, 0.0, 1.0, 5.0, 3.0]),
                    }
                }
            }
            logits
        }
    }

    impl<B: Backend> Seq2SeqLanguageModel<B> for ScriptedModel<B> {
        type Config = ScriptedConfig;
        type State = ScriptedState<B>;

        /// Encodes source IDs as one-channel floating markers.
        fn encode(
            &self,
            source: CausalLmInput<B>,
            _options: &ForwardOptions,
        ) -> Result<EncoderOutput<B>> {
            Ok(EncoderOutput {
                hidden_state: source.token_ids.float().unsqueeze_dim(2),
                token_mask: source.token_mask,
                hidden_states: Vec::new(),
                attentions: Vec::new(),
            })
        }

        /// Provides a complete stateless graph for the trait, unused by search.
        fn forward(
            &self,
            input: Seq2SeqInput<B>,
            options: &ForwardOptions,
        ) -> Result<Seq2SeqOutput<B>> {
            let encoder = self.encode(input.encoder, options)?;
            let batch = input.decoder.token_ids.dims()[0];
            let logits = self.prefill_logits(&encoder)?;
            let device = input.decoder.token_ids.device();
            Ok(self.output(logits, batch, encoder, &device))
        }

        /// Allocates the fixed-batch scripted lifecycle state.
        fn new_decode_state(
            &self,
            config: CacheConfig,
            batch_size: usize,
            device: &B::Device,
        ) -> Result<Self::State> {
            if config.sequence_count() != batch_size {
                return Err(Error::Cache("scripted cache batch mismatch".into()));
            }
            Ok(ScriptedState {
                batch: batch_size,
                capacity: config.capacity(),
                lengths: vec![0; batch_size],
                device: device.clone(),
                trace: Arc::clone(&self.trace),
            })
        }

        /// Commits initial decoder-prefix lengths and returns scripted logits.
        fn prefill_decoder(
            &self,
            encoder: &EncoderOutput<B>,
            decoder: CausalLmInput<B>,
            state: &mut Self::State,
            _options: &ForwardOptions,
        ) -> Result<Seq2SeqOutput<B>> {
            let [batch, sequence] = decoder.token_ids.dims();
            let valid = super::mask_values(
                decoder.token_mask.as_ref(),
                batch,
                sequence,
                "scripted prefill mask",
            )?;
            state.lengths =
                super::nonempty_row_lengths(&valid, batch, sequence, "scripted prefill")?;
            let logits = self.prefill_logits(encoder)?;
            let device = decoder.token_ids.device();
            Ok(self.output(logits, batch, encoder.clone(), &device))
        }

        /// Records one reordered cached step and returns token-keyed logits.
        fn decode(
            &self,
            encoder: &EncoderOutput<B>,
            input: DecodeInput<B>,
            state: &mut Self::State,
            _options: &ForwardOptions,
        ) -> Result<Seq2SeqOutput<B>> {
            let [batch, sequence] = input.token_ids.dims();
            if sequence != 1 || batch != state.batch {
                return Err(Error::Cache("scripted decode shape mismatch".into()));
            }
            let tokens = super::tensor_i64_values(input.token_ids.clone(), "scripted tokens")?;
            let valid = super::mask_values(
                input.token_mask.as_ref(),
                batch,
                sequence,
                "scripted decode mask",
            )?;
            for (length, &is_valid) in state.lengths.iter_mut().zip(&valid) {
                if is_valid {
                    *length += 1;
                }
            }
            let positions = input
                .position_ids
                .map(|positions| super::tensor_i64_values(positions, "scripted positions"))
                .transpose()?;
            {
                let mut trace = self.trace.lock().expect("script trace mutex");
                trace.decode_tokens.push(tokens.clone());
                trace.decode_valid.push(valid.clone());
                trace.decode_positions.push(positions);
            }
            let logits = self.decode_logits(&tokens, &valid);
            let device = input.token_ids.device();
            Ok(self.output(logits, batch, encoder.clone(), &device))
        }
    }

    /// Constructs a valid one-source scripted request.
    fn request(
        device: &TestDevice,
        search: BeamSearchConfig,
    ) -> Seq2SeqBeamSearchRequest<TestBackend> {
        let beam_batch = search.num_beams;
        Seq2SeqBeamSearchRequest {
            source: CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[4, 0]], device),
                token_mask: None,
                position_ids: None,
            },
            decoder_prompt: CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0]], device),
                token_mask: None,
                position_ids: None,
            },
            cache: CacheConfig::new(beam_batch, search.max_new_tokens + 1).expect("script cache"),
            search,
            vocabulary_size: 5,
        }
    }

    /// Returns standard scripted search controls with no EOS termination.
    fn search(max_new_tokens: usize, beams: usize) -> BeamSearchConfig {
        BeamSearchConfig {
            max_new_tokens,
            num_beams: beams,
            num_return_sequences: beams,
            length_penalty: 1.0,
            early_stopping: false,
            eos_token_ids: Vec::new(),
            pad_token_id: 0,
        }
    }

    /// Every scalar range and cross-field relationship fails preflight.
    #[test]
    fn beam_configuration_rejects_invalid_values() {
        let invalid = [
            BeamSearchConfig {
                max_new_tokens: 0,
                ..BeamSearchConfig::default()
            },
            BeamSearchConfig {
                num_beams: 0,
                ..BeamSearchConfig::default()
            },
            BeamSearchConfig {
                num_beams: 6,
                ..BeamSearchConfig::default()
            },
            BeamSearchConfig {
                num_beams: 2,
                num_return_sequences: 0,
                ..BeamSearchConfig::default()
            },
            BeamSearchConfig {
                num_beams: 2,
                num_return_sequences: 3,
                ..BeamSearchConfig::default()
            },
            BeamSearchConfig {
                length_penalty: f64::NAN,
                ..BeamSearchConfig::default()
            },
            BeamSearchConfig {
                length_penalty: f64::MAX,
                ..BeamSearchConfig::default()
            },
            BeamSearchConfig {
                pad_token_id: 5,
                ..BeamSearchConfig::default()
            },
            BeamSearchConfig {
                eos_token_ids: vec![5],
                ..BeamSearchConfig::default()
            },
        ];
        for config in invalid {
            assert!(config.validate(5).is_err(), "accepted {config:?}");
        }
        assert!(BeamSearchConfig::default().validate(0).is_err());
        assert!(
            BeamSearchConfig {
                length_penalty: -0.1,
                ..BeamSearchConfig::default()
            }
            .validate(5)
            .is_ok()
        );
        assert!(
            BeamSearchConfig {
                num_beams: 1,
                num_return_sequences: 1,
                ..BeamSearchConfig::default()
            }
            .validate(5)
            .is_ok()
        );
    }

    /// Score ties are resolved by ascending token ID and remain reproducible.
    #[test]
    fn one_step_stable_ties_return_tokens_in_id_order() {
        let device = Default::default();
        let model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        let output = generate_seq2seq_beam_search(&model, request(&device, search(1, 2)))
            .expect("one-step beam search");
        assert_eq!(output.sequences.len(), 1);
        assert_eq!(output.sequences[0].len(), 2);
        assert_eq!(output.sequences[0][0].token_ids, vec![1]);
        assert_eq!(output.sequences[0][1].token_ids, vec![2]);
        assert_eq!(output.sequences[0][0].score, output.sequences[0][1].score);
        for sequence in &output.sequences[0] {
            assert_eq!(sequence.decoder_prompt_length, 1);
            assert_eq!(
                sequence.log_probabilities.iter().sum::<f64>(),
                sequence.cumulative_log_probability
            );
        }
        assert!(
            output.sequences[0]
                .iter()
                .all(|sequence| sequence.finish_reason == FinishReason::Length)
        );
    }

    /// Parent cache rows are duplicated first and gathered at every later step.
    #[test]
    fn cached_multistep_search_reorders_external_state() {
        let device = Default::default();
        let model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        let trace = Arc::clone(&model.trace);
        let output = generate_seq2seq_beam_search(&model, request(&device, search(3, 2)))
            .expect("multistep beam search");
        assert_eq!(output.sequences[0].len(), 2);
        assert!(
            output.sequences[0]
                .iter()
                .all(|sequence| sequence.token_ids.len() == 3)
        );
        let trace = trace.lock().expect("script trace mutex");
        assert_eq!(trace.reorders.len(), 2);
        assert_eq!(trace.reorders[0], vec![0, 0]);
        assert_eq!(trace.decode_tokens.len(), 2);
        assert!(trace.decode_valid.iter().flatten().all(|valid| *valid));
    }

    /// Eager early stopping halts once the beam-width EOS pool is complete.
    #[test]
    fn eager_early_stopping_avoids_unnecessary_decode() {
        let device = Default::default();
        let model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        let trace = Arc::clone(&model.trace);
        let mut config = search(4, 2);
        config.eos_token_ids = vec![1, 2];
        config.early_stopping = true;
        let output = generate_seq2seq_beam_search(&model, request(&device, config))
            .expect("early beam search");
        assert_eq!(output.sequences[0][0].token_ids, vec![1]);
        assert_eq!(output.sequences[0][1].token_ids, vec![2]);
        assert!(output.sequences[0].iter().all(|sequence| matches!(
            sequence.finish_reason,
            FinishReason::EndOfSequence { token_id: 1 | 2 }
        )));
        let trace = trace.lock().expect("script trace mutex");
        assert!(trace.reorders.is_empty());
        assert!(trace.decode_tokens.is_empty());
    }

    /// Lower-frontier EOS candidates are backups, not admissible completions.
    #[test]
    fn eager_stopping_ignores_eos_below_top_beam_ranks() {
        let device = Default::default();
        let model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        let trace = Arc::clone(&model.trace);
        let mut config = search(4, 2);
        // The prefill ranks live IDs 1 and 2 first, followed by EOS IDs 4
        // and 3. Transformers 5.2 therefore ignores both initial EOS rows;
        // they must not fill the eager completion pool before one decode.
        config.eos_token_ids = vec![3, 4];
        config.early_stopping = true;
        let output = generate_seq2seq_beam_search(&model, request(&device, config))
            .expect("lower-frontier EOS beam search");
        assert_eq!(output.sequences[0].len(), 2);
        assert!(output.sequences[0].iter().all(|sequence| {
            sequence.token_ids.len() == 2
                && matches!(
                    sequence.finish_reason,
                    FinishReason::EndOfSequence { token_id: 3 | 4 }
                )
        }));
        let trace = trace.lock().expect("script trace mutex");
        assert_eq!(trace.reorders, vec![vec![0, 0]]);
        assert_eq!(trace.decode_tokens, vec![vec![1, 2]]);
        assert_eq!(trace.decode_valid, vec![vec![true, true]]);
    }

    /// Disabled eager stopping continues live non-EOS prefixes to the bound.
    #[test]
    fn non_eager_search_continues_after_enough_eos_hypotheses() {
        let device = Default::default();
        let model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        let trace = Arc::clone(&model.trace);
        let mut config = search(3, 2);
        config.eos_token_ids = vec![1, 2];
        let _ = generate_seq2seq_beam_search(&model, request(&device, config))
            .expect("non-eager beam search");
        let trace = trace.lock().expect("script trace mutex");
        assert_eq!(trace.decode_tokens.len(), 2);
        assert_eq!(trace.reorders.len(), 2);
    }

    /// Finished source groups contribute false-masked PAD rows while peers run.
    #[test]
    fn batched_finished_rows_use_pad_noops_during_peer_decode() {
        let device = Default::default();
        let model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        let trace = Arc::clone(&model.trace);
        let mut config = search(3, 1);
        config.num_return_sequences = 1;
        config.eos_token_ids = vec![1];
        config.early_stopping = true;
        let request = Seq2SeqBeamSearchRequest {
            source: CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[4, 0], [3, 0]], &device),
                token_mask: None,
                position_ids: None,
            },
            decoder_prompt: CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0], [0]], &device),
                token_mask: None,
                position_ids: None,
            },
            cache: CacheConfig::new(2, 4).unwrap(),
            search: config,
            vocabulary_size: 5,
        };
        let output =
            generate_seq2seq_beam_search(&model, request).expect("ragged batch beam search");
        assert_eq!(output.sequences.len(), 2);
        let trace = trace.lock().expect("script trace mutex");
        assert!(!trace.decode_valid.is_empty());
        assert_eq!(trace.decode_valid[0], vec![false, true]);
        assert_eq!(trace.decode_tokens[0], vec![0, 3]);
    }

    /// Explicit decoder positions follow the selected parent through reorder.
    #[test]
    fn explicit_positions_advance_after_beam_duplication() {
        let device = Default::default();
        let model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        let trace = Arc::clone(&model.trace);
        let mut request = request(&device, search(2, 2));
        request.decoder_prompt.position_ids = Some(Tensor::from_ints([[7]], &device));
        let _ = generate_seq2seq_beam_search(&model, request).expect("positioned beam search");
        let trace = trace.lock().expect("script trace mutex");
        assert_eq!(trace.decode_positions, vec![Some(vec![8, 8])]);
    }

    /// Positive length penalty can prefer a longer, lower-probability sequence.
    #[test]
    fn length_penalty_changes_finished_ranking() {
        let short = BeamHypothesis {
            token_ids: vec![1],
            log_probabilities: vec![-1.0],
            cumulative_log_probability: -1.0,
            next_position: None,
            finish_reason: Some(FinishReason::Length),
            ordinal: 0,
        };
        let long = BeamHypothesis {
            token_ids: vec![2, 3],
            log_probabilities: vec![-0.75, -0.75],
            cumulative_log_probability: -1.5,
            next_position: None,
            finish_reason: Some(FinishReason::Length),
            ordinal: 1,
        };
        assert!(completed_order(&short, &long, 0.0).is_lt());
        assert!(completed_order(&long, &short, 1.0).is_lt());
    }

    /// Invalid request rectangles and cache bounds fail before state allocation.
    #[test]
    fn request_preflight_rejects_shape_mask_and_cache_errors() {
        let device = Default::default();
        let model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        let mut wrong_batch = request(&device, search(2, 2));
        wrong_batch.decoder_prompt.token_ids = Tensor::from_ints([[0], [0]], &device);
        assert!(generate_seq2seq_beam_search(&model, wrong_batch).is_err());

        let mut wrong_mask = request(&device, search(2, 2));
        wrong_mask.decoder_prompt.token_mask = Some(TokenMask::new(
            Tensor::<TestBackend, 2, Bool>::from_bool([[true, false]].into(), &device),
        ));
        assert!(generate_seq2seq_beam_search(&model, wrong_mask).is_err());

        let mut empty_prompt = request(&device, search(2, 2));
        empty_prompt.decoder_prompt.token_mask = Some(TokenMask::new(
            Tensor::<TestBackend, 2, Bool>::from_bool([[false]].into(), &device),
        ));
        assert!(generate_seq2seq_beam_search(&model, empty_prompt).is_err());

        let mut wrong_cache_batch = request(&device, search(2, 2));
        wrong_cache_batch.cache = CacheConfig::new(1, 3).unwrap();
        assert!(generate_seq2seq_beam_search(&model, wrong_cache_batch).is_err());

        let mut short_cache = request(&device, search(2, 2));
        short_cache.cache = CacheConfig::new(2, 2).unwrap();
        assert!(generate_seq2seq_beam_search(&model, short_cache).is_err());
    }

    /// Gapped true-valid decoder masks report semantic rather than padded length.
    #[test]
    fn decoder_prompt_length_counts_only_valid_mask_entries() {
        let device = Default::default();
        let model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        let mut request = request(&device, search(1, 2));
        request.decoder_prompt = CausalLmInput {
            token_ids: Tensor::from_ints([[0, 4, 0]], &device),
            token_mask: Some(TokenMask::new(Tensor::from_bool(
                [[true, false, true]].into(),
                &device,
            ))),
            position_ids: None,
        };
        request.cache = CacheConfig::new(2, 3).unwrap();
        let output = generate_seq2seq_beam_search(&model, request)
            .expect("masked decoder prompt beam search");
        assert!(
            output.sequences[0]
                .iter()
                .all(|sequence| sequence.decoder_prompt_length == 2)
        );
    }

    /// NaN model scores are rejected instead of entering an unstable sort.
    #[test]
    fn nonfinite_model_probabilities_are_contextual_errors() {
        let device = Default::default();
        let model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: true,
            },
            &device,
        );
        let error = generate_seq2seq_beam_search(&model, request(&device, search(1, 2)))
            .expect_err("NaN scores must fail");
        assert!(error.to_string().contains("non-finite"));
    }

    /// A one-beam request is the deterministic greedy boundary case.
    #[test]
    fn one_beam_returns_one_length_limited_sequence() {
        let device = Default::default();
        let model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        let output = generate_seq2seq_beam_search(&model, request(&device, search(2, 1)))
            .expect("one-beam search");
        assert_eq!(output.sequences[0].len(), 1);
        assert_eq!(output.sequences[0][0].token_ids.len(), 2);
        assert_eq!(output.sequences[0][0].finish_reason, FinishReason::Length);
    }
}
