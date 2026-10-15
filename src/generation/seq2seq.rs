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
//! expanded to a fixed `[source_batch * beams]` batch, with beam `k` of source
//! `s` in row `s * beams + k`. At each later step,
//! [`DecodeState::reorder_within_groups`] gathers or duplicates parent rows
//! inside each source's group of `beams` rows before selected tokens enter
//! [`Seq2SeqLanguageModel::decode`]. A reorder never moves a hypothesis to
//! another source (the row arithmetic is verified with Verus), and the rows of
//! one source start as identical copies of its encoder output, so the encoder
//! output is expanded once and never re-gathered; a model may likewise keep
//! per-source state such as projected cross-attention keys and values in
//! place (T5 does). Finished source rows remain
//! in the rectangle as false-masked PAD no-ops, preserving other rows'
//! progress without mutating their cache lengths.
//!
//! Each step reads the active rows' log probabilities to the host and keeps,
//! per source, the best `beams * max(2, eos_count + 1)` candidates with a
//! bounded heap: `O(beams * vocabulary * log frontier)` time and
//! `O(frontier)` memory instead of materializing and sorting every
//! beam-vocabulary pair. The order is total (score, then parent slot, then
//! token ID), so the result is exactly the prefix a full stable sort would
//! produce. Log probabilities are computed in at least F32 like Hugging Face.
//!
//! # Stopping rules
//!
//! [`EarlyStopping`] mirrors Transformers' `early_stopping` setting. Every
//! rule first needs a full finished pool of `num_beams` hypotheses; let `c` be
//! the best live beam's cumulative log probability, `lp` the length penalty,
//! and `w` the worst finished score.
//!
//! - [`EarlyStopping::Eager`] (`true`) stops as soon as the pool is full.
//! - [`EarlyStopping::Heuristic`] (`false`, the Transformers default) stops
//!   once `c / n^lp <= w`, scoring the live beam at its current generated
//!   length `n`. With a positive penalty a longer continuation can still
//!   overtake `w`, so this is a heuristic, kept for parity.
//! - [`EarlyStopping::Never`] (`"never"`) stops only when no live beam can
//!   enter the pool at any reachable length: log probabilities are
//!   non-positive, so a descendant scores at most
//!   `max(c / L_min^lp, c / L_max^lp)` over the lengths `L_min..=L_max` it can
//!   still reach. The result is identical to searching every source to the
//!   length bound; only wasted steps disappear. The same exact bound also
//!   ends heuristic searches early when it is tighter, without changing them.

use std::collections::BinaryHeap;

use burn::tensor::{Int, Tensor, TensorData, activation::log_softmax, backend::Backend};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    CausalLmInput, DecodeInput, DecodeState, EncoderOutput, Error, FinishReason, ForwardOptions,
    LogitsSelection, Result, Seq2SeqLanguageModel, TokenMask,
    architecture::{bool_tensor_from_values, select_bool_tensor},
    cache::CacheConfig,
};

use super::bookkeeping::{beam_row, fill_beam_expansion, frontier_capacity};
use super::metadata::{self, extract_last_logits};
use super::probability_accumulation_dtype;

/// Label used for model-logit shape errors raised by beam search.
const MODEL_LOGITS: &str = "beam_search.model_logits";

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
    /// When a source row stops searching (see [`EarlyStopping`]).
    ///
    /// Serialized like Transformers' `early_stopping`: `true`, `false`, or
    /// `"never"`. Under every rule, following Transformers 5.2, only finished
    /// candidates ranked within the current top `num_beams` may enter the
    /// finished pool; the rest of the wider candidate frontier exists solely
    /// to find live successors.
    pub early_stopping: EarlyStopping,
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
            early_stopping: EarlyStopping::Heuristic,
            eos_token_ids: Vec::new(),
            pad_token_id: 0,
        }
    }
}

/// When beam search stops a source row, mirroring Transformers' `early_stopping`.
///
/// All three rules wait for a full finished pool of `num_beams` hypotheses;
/// they differ in how they decide that no live beam can still improve it (see
/// the [module documentation](self) for the bounds). Serializes exactly like
/// Transformers' `GenerationConfig.early_stopping`: [`Self::Eager`] as `true`,
/// [`Self::Heuristic`] as `false`, and [`Self::Never`] as `"never"`.
///
/// ```
/// use transformers::generation::EarlyStopping;
///
/// let never: EarlyStopping = serde_json::from_str("\"never\"").unwrap();
/// assert_eq!(never, EarlyStopping::Never);
/// assert_eq!(EarlyStopping::from(false), EarlyStopping::Heuristic);
/// assert_eq!(serde_json::to_string(&EarlyStopping::Eager).unwrap(), "true");
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum EarlyStopping {
    /// `true`: stop as soon as `num_beams` finished hypotheses exist.
    Eager,
    /// `false` (the Transformers default): stop once the best live beam, scored
    /// at its current generated length, cannot beat the worst finished score.
    #[default]
    Heuristic,
    /// `"never"`: stop only when no live beam can beat the worst finished score
    /// at any reachable length; identical to searching to the length bound.
    Never,
}

impl From<bool> for EarlyStopping {
    /// Maps Transformers' boolean setting: `true` is eager, `false` heuristic.
    fn from(eager: bool) -> Self {
        if eager { Self::Eager } else { Self::Heuristic }
    }
}

impl Serialize for EarlyStopping {
    /// Writes `true`, `false`, or `"never"`, as Transformers does.
    fn serialize<S: Serializer>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error> {
        match self {
            Self::Eager => serializer.serialize_bool(true),
            Self::Heuristic => serializer.serialize_bool(false),
            Self::Never => serializer.serialize_str("never"),
        }
    }
}

impl<'de> Deserialize<'de> for EarlyStopping {
    /// Reads `true`, `false`, or `"never"`; any other value is an error.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> core::result::Result<Self, D::Error> {
        /// The two JSON shapes Transformers uses for the setting.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Representation {
            /// `true` or `false`.
            Flag(bool),
            /// The only accepted string, `"never"`.
            Text(String),
        }
        match Representation::deserialize(deserializer)? {
            Representation::Flag(eager) => Ok(eager.into()),
            Representation::Text(text) if text == "never" => Ok(Self::Never),
            Representation::Text(text) => Err(serde::de::Error::custom(format!(
                "early_stopping must be true, false, or \"never\", found \"{text}\""
            ))),
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
    run_beam_search(model, request, true)
}

/// Runs beam search, optionally stopping sources whose finished pool is final.
///
/// `prune` only removes decode steps that cannot change the result; tests run
/// both settings to check that claim.
fn run_beam_search<B, M>(
    model: &M,
    request: Seq2SeqBeamSearchRequest<B>,
    prune: bool,
) -> Result<Seq2SeqBeamSearchOutput>
where
    B: Backend,
    M: Seq2SeqLanguageModel<B>,
{
    let prepared = PreparedRequest::new(&request)?;
    let device = request.source.token_ids.device();
    let beams = request.search.num_beams;
    let vocabulary_size = request.vocabulary_size;
    let options = ForwardOptions {
        logits: LogitsSelection::Last,
        ..ForwardOptions::default()
    };

    let encoded = model.encode(request.source, &options)?;
    validate_encoder_batch(&encoded, prepared.source_batch)?;
    let expansion = beam_expansion_indices(prepared.source_batch, beams)?;
    let mut encoder = select_encoder(encoded, &expansion)?;
    let decoder_prompt = select_input(request.decoder_prompt, &expansion)?;
    let mut state = model.new_decode_state(request.cache, prepared.expanded_batch, &device)?;
    let prefill = model.prefill_decoder(&encoder, decoder_prompt, &mut state, &options)?;
    encoder = prefill.encoder;
    validate_encoder_batch(&encoder, prepared.expanded_batch)?;
    let mut logits = extract_last_logits(
        prefill.decoder.logits,
        prepared.expanded_batch,
        vocabulary_size,
        MODEL_LOGITS,
    )?;

    let mut search = BeamSearch::new(&prepared, &request.search, vocabulary_size, prune)?;
    for step in 0..request.search.max_new_tokens {
        let probabilities = host_log_probabilities(
            logits,
            &search.active_rows(),
            prepared.expanded_batch,
            vocabulary_size,
        )?;
        let plan = search.advance(step, &probabilities)?;
        if search.all_done() {
            break;
        }

        // Parents and destinations both come from `beam_row(source, ..)`, so
        // the reorder stays inside each source group, whose encoder rows are
        // identical: neither the encoder output nor any per-source state a
        // model binds needs a gather.
        state.reorder_within_groups(&plan.parent_indices, beams)?;
        let token_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(plan.tokens, [prepared.expanded_batch, 1]),
            &device,
        );
        let valid = bool_tensor_from_values(plan.valid, [prepared.expanded_batch, 1], &device);
        let position_ids = plan.positions.map(|positions| {
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(positions, [prepared.expanded_batch, 1]),
                &device,
            )
        });
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
            vocabulary_size,
            MODEL_LOGITS,
        )?;
    }

    finish_output(
        search.completed,
        &prepared.decoder_prompt_lengths,
        &request.search,
    )
}

/// Validated dimensions and position metadata derived before model mutation.
struct PreparedRequest {
    /// Independent encoder rows before expansion into beam slots.
    source_batch: usize,
    /// Decoder batch after allocating a fixed number of beams per source.
    expanded_batch: usize,
    /// Number of valid decoder prompt tokens for each source row.
    decoder_prompt_lengths: Vec<usize>,
    /// Optional absolute position to assign after each decoder prompt.
    initial_next_positions: Option<Vec<i64>>,
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
        let decoder_valid = metadata::mask_values(
            request.decoder_prompt.token_mask.as_ref(),
            decoder_batch,
            decoder_sequence,
            "decoder prompt mask",
        )?;
        let decoder_prompt_lengths = metadata::nonempty_row_lengths(
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
        let maximum_prompt = decoder_prompt_lengths.iter().copied().max().unwrap_or(0);
        let required_capacity =
            metadata::required_cache_capacity(maximum_prompt, request.search.max_new_tokens)
                .ok_or_else(|| {
                    Error::Generation("beam-search cache capacity overflows usize".into())
                })?;
        if request.cache.capacity() < required_capacity {
            return Err(Error::Generation(format!(
                "beam-search cache capacity {} is smaller than required decoder prompt {maximum_prompt} + {} - 1 fed-back tokens = {required_capacity}",
                request.cache.capacity(),
                request.search.max_new_tokens
            )));
        }

        let initial_next_positions = request
            .decoder_prompt
            .position_ids
            .clone()
            .map(|positions| {
                metadata::tensor_i64_values(positions, "decoder prompt positions").and_then(
                    |values| {
                        metadata::next_positions(
                            &values,
                            &decoder_valid,
                            decoder_batch,
                            decoder_sequence,
                            request.search.max_new_tokens,
                            "decoder prompt",
                        )
                    },
                )
            })
            .transpose()?;
        Ok(Self {
            source_batch,
            expanded_batch,
            decoder_prompt_lengths,
            initial_next_positions,
        })
    }
}

/// Internal unfinished or finalized hypothesis.
#[derive(Clone, Debug)]
struct BeamHypothesis {
    /// Generated suffix, excluding the decoder prompt.
    token_ids: Vec<u32>,
    /// Processed conditional log probability of each suffix token.
    log_probabilities: Vec<f64>,
    /// Sum of token log probabilities before length normalization.
    cumulative_log_probability: f64,
    /// Explicit absolute position for the next token, when supplied by the request.
    next_position: Option<i64>,
    /// Stopping condition for a finalized hypothesis; absent for a live beam.
    finish_reason: Option<FinishReason>,
    /// Stable insertion order used to break otherwise equal ranking ties.
    ordinal: usize,
}

/// One not-yet-materialized child from a model probability row.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    /// Parent row in the complete expanded decoder batch.
    parent_global: usize,
    /// Parent beam slot within its independent source row.
    parent_slot: usize,
    /// Proposed next vocabulary token.
    token_id: u32,
    /// Conditional log probability of the proposed token.
    log_probability: f64,
    /// Parent cumulative log probability plus the candidate token contribution.
    cumulative_log_probability: f64,
}

/// Decoder inputs for the next step, built while selecting successors.
struct StepPlan {
    /// Parent row gathered into each destination row by the cache reorder.
    parent_indices: Vec<usize>,
    /// Next token for each row; PAD for inactive rows.
    tokens: Vec<i64>,
    /// Whether each row carries a real token this step.
    valid: Vec<bool>,
    /// Explicit next positions when the request supplies decoder positions.
    positions: Option<Vec<i64>>,
}

/// Mutable search state: live beams, finished pools, and per-source status.
struct BeamSearch<'a> {
    /// Validated search controls.
    config: &'a BeamSearchConfig,
    /// Vocabulary extent of every probability row.
    vocabulary_size: usize,
    /// Live hypothesis for each expanded row, or `None` for an unused slot.
    active: Vec<Option<BeamHypothesis>>,
    /// Ranked finished hypotheses for each source, at most `num_beams` long.
    completed: Vec<Vec<BeamHypothesis>>,
    /// Whether each source has stopped searching.
    done: Vec<bool>,
    /// Next creation ordinal, the last tie breaker between equal hypotheses.
    ordinal: usize,
    /// Whether decode calls carry explicit positions.
    has_positions: bool,
    /// Whether sources stop once no live beam can enter their finished pool.
    prune: bool,
}

impl<'a> BeamSearch<'a> {
    /// Starts one score-zero live prefix per source, with dormant spare slots.
    fn new(
        prepared: &PreparedRequest,
        config: &'a BeamSearchConfig,
        vocabulary_size: usize,
        prune: bool,
    ) -> Result<Self> {
        let mut active = vec![None; prepared.expanded_batch];
        for source in 0..prepared.source_batch {
            let row = beam_row(source, config.num_beams, 0)
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
        Ok(Self {
            config,
            vocabulary_size,
            active,
            completed: empty_completed(prepared.source_batch)?,
            done: vec![false; prepared.source_batch],
            ordinal: prepared.source_batch,
            has_positions: prepared.initial_next_positions.is_some(),
            prune,
        })
    }

    /// Rows whose live hypothesis belongs to an unfinished source.
    fn active_rows(&self) -> Vec<bool> {
        self.active
            .iter()
            .enumerate()
            .map(|(row, beam)| beam.is_some() && !self.done[row / self.config.num_beams])
            .collect()
    }

    /// Whether every source has stopped.
    fn all_done(&self) -> bool {
        self.done.iter().all(|&done| done)
    }

    /// Returns a fresh hypothesis ordinal.
    fn next_ordinal(&mut self) -> Result<usize> {
        let ordinal = self.ordinal;
        self.ordinal = ordinal
            .checked_add(1)
            .ok_or_else(|| Error::Generation("beam-search hypothesis ordinal overflow".into()))?;
        Ok(ordinal)
    }

    /// Ranks every unfinished source's candidates and plans the next decode.
    fn advance(&mut self, step: usize, probabilities: &[f64]) -> Result<StepPlan> {
        let expanded_batch = self.active.len();
        let mut plan = StepPlan {
            parent_indices: (0..expanded_batch).collect(),
            tokens: vec![i64::from(self.config.pad_token_id); expanded_batch],
            valid: vec![false; expanded_batch],
            positions: self.has_positions.then(|| vec![0; expanded_batch]),
        };
        let mut next_active = vec![None; expanded_batch];
        for source in 0..self.done.len() {
            if !self.done[source] {
                self.advance_source(source, step, probabilities, &mut plan, &mut next_active)?;
            }
        }
        self.active = next_active;
        Ok(plan)
    }

    /// Finalizes EOS/length candidates of one source and selects its live successors.
    fn advance_source(
        &mut self,
        source: usize,
        step: usize,
        probabilities: &[f64],
        plan: &mut StepPlan,
        next_active: &mut [Option<BeamHypothesis>],
    ) -> Result<()> {
        let beams = self.config.num_beams;
        let capacity =
            frontier_capacity(beams, self.config.eos_token_ids.len()).ok_or_else(|| {
                Error::Generation("beam-search candidate frontier overflows usize".into())
            })?;
        let candidates = candidates_for_source(
            source,
            beams,
            self.vocabulary_size,
            capacity,
            &self.active,
            probabilities,
        )?;
        if candidates.is_empty() {
            return Err(Error::Generation(format!(
                "beam-search source row {source} has no finite candidate"
            )));
        }

        let at_length_limit = step + 1 == self.config.max_new_tokens;
        let mut selected = Vec::with_capacity(beams);
        for (rank, candidate) in candidates.into_iter().enumerate() {
            let is_eos = self.config.eos_token_ids.contains(&candidate.token_id);
            if is_eos || at_length_limit {
                // Transformers 5.2 masks finished candidates below the
                // current beam width. Those extra frontier rows are only
                // backups for selecting enough non-finished successors.
                if rank >= beams {
                    continue;
                }
                let reason = if is_eos {
                    FinishReason::EndOfSequence {
                        token_id: candidate.token_id,
                    }
                } else {
                    FinishReason::Length
                };
                let ordinal = self.next_ordinal()?;
                let hypothesis = extend_candidate(&self.active, &candidate, Some(reason), ordinal)?;
                self.completed[source].push(hypothesis);
            } else if selected.len() < beams {
                selected.push(candidate);
            }
        }
        rank_and_prune(
            &mut self.completed[source],
            beams,
            self.config.length_penalty,
        );

        let pool_full = self.completed[source].len() >= beams;
        let best_live = selected.first().map(|best| best.cumulative_log_probability);
        // Exact: no live descendant can enter the pool at any reachable length.
        // Every rule may stop here without changing its result.
        let no_live_beam_can_improve = self.prune
            && best_live.is_some_and(|best| {
                finished_pool_is_final(
                    &self.completed[source],
                    beams,
                    best,
                    step + 2,
                    self.config.max_new_tokens,
                    self.config.length_penalty,
                )
            });
        let rule_stops = match self.config.early_stopping {
            EarlyStopping::Eager => pool_full,
            // Live beams now hold `step + 1` generated tokens.
            EarlyStopping::Heuristic => best_live.is_some_and(|best| {
                heuristic_pool_is_final(
                    &self.completed[source],
                    beams,
                    best,
                    step + 1,
                    self.config.length_penalty,
                )
            }),
            EarlyStopping::Never => false,
        };
        if at_length_limit || selected.is_empty() || rule_stops || no_live_beam_can_improve {
            self.done[source] = true;
            return Ok(());
        }

        for (slot, candidate) in selected.into_iter().enumerate() {
            let destination = beam_row(source, beams, slot)
                .ok_or_else(|| Error::Generation("beam destination index overflow".into()))?;
            let parent = self.active[candidate.parent_global]
                .as_ref()
                .ok_or_else(|| Error::Generation("selected beam parent disappeared".into()))?;
            if let Some(positions) = &mut plan.positions {
                positions[destination] = parent.next_position.ok_or_else(|| {
                    Error::Generation("explicit beam position disappeared".into())
                })?;
            }
            plan.parent_indices[destination] = candidate.parent_global;
            plan.tokens[destination] = i64::from(candidate.token_id);
            plan.valid[destination] = true;
            let ordinal = self.next_ordinal()?;
            next_active[destination] =
                Some(extend_candidate(&self.active, &candidate, None, ordinal)?);
        }
        Ok(())
    }
}

/// Whether a full finished pool is final: no live descendant can enter it.
///
/// A live beam with cumulative log probability `c` finishes later with some
/// length `L` in `shortest..=longest` and cumulative `c' <= c`, because every
/// log probability is non-positive. Its score `c' / L^lp` is then at most
/// `c / L^lp`, which is monotone in `L`, so the bound is the larger of the
/// two endpoint values. The comparison is strict: a descendant that could
/// merely tie the worst finished score might still win a tie breaker.
fn finished_pool_is_final(
    completed: &[BeamHypothesis],
    beams: usize,
    best_live_cumulative: f64,
    shortest: usize,
    longest: usize,
    length_penalty: f64,
) -> bool {
    if completed.len() < beams || shortest > longest {
        return false;
    }
    let Some(worst) = completed.last() else {
        return false;
    };
    let score_at = |length: usize| best_live_cumulative / (length as f64).powf(length_penalty);
    let bound = score_at(shortest).max(score_at(longest));
    bound < normalized_score(worst, length_penalty)
}

/// Transformers' default stopping heuristic (`early_stopping=False`).
///
/// A full pool is final once the best live beam, scored at its *current*
/// generated length `n`, does not exceed the worst finished score:
/// `c / n^lp <= w` (Transformers' `_check_early_stop_heuristic` with
/// `best_hypothetical_length = cur_len - decoder_prompt_len`). With a positive
/// penalty a longer descendant could still score higher, so unlike
/// [`finished_pool_is_final`] this may stop before the exhaustive optimum; it
/// is kept for parity with Transformers' default results.
fn heuristic_pool_is_final(
    completed: &[BeamHypothesis],
    beams: usize,
    best_live_cumulative: f64,
    generated: usize,
    length_penalty: f64,
) -> bool {
    if completed.len() < beams || generated == 0 {
        return false;
    }
    let Some(worst) = completed.last() else {
        return false;
    };
    best_live_cumulative / (generated as f64).powf(length_penalty)
        <= normalized_score(worst, length_penalty)
}

/// Allocates one empty finished-hypothesis pool per source row.
fn empty_completed(batch: usize) -> Result<Vec<Vec<BeamHypothesis>>> {
    metadata::empty_rows(batch, "completed beam pools")
}

/// Candidate wrapper whose heap order puts the *worst* retained candidate on top.
///
/// `candidate_order` ranks better candidates first (`Less`), so a max-heap of
/// this wrapper exposes the candidate that the next better one should evict.
#[derive(Clone, Copy, Debug)]
struct WorstFirst(Candidate);

impl PartialEq for WorstFirst {
    /// Candidates are equal only when the total order ties, i.e. never for distinct pairs.
    fn eq(&self, other: &Self) -> bool {
        candidate_order(&self.0, &other.0).is_eq()
    }
}

impl Eq for WorstFirst {}

impl PartialOrd for WorstFirst {
    /// Delegates to the total candidate order.
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorstFirst {
    /// Orders by rank, so the greatest element is the lowest-ranked candidate.
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        candidate_order(&self.0, &other.0)
    }
}

/// Returns the best `capacity` finite continuations of one source, best first.
///
/// Scans every `(parent beam, token)` pair once and keeps a bounded heap, so
/// memory is `O(capacity)` rather than `O(beams * vocabulary)`. Because
/// `candidate_order` is a strict total order, the result equals the first
/// `capacity` entries of a full stable sort. `-inf` scores (masked tokens) are
/// skipped; NaN, `+inf`, or an overflowing cumulative score is an error.
fn candidates_for_source(
    source: usize,
    beams: usize,
    vocabulary: usize,
    capacity: usize,
    active: &[Option<BeamHypothesis>],
    probabilities: &[f64],
) -> Result<Vec<Candidate>> {
    let mut frontier = BinaryHeap::new();
    frontier
        .try_reserve(capacity.saturating_add(1))
        .map_err(|_| Error::Generation(format!("failed to allocate {capacity} beam candidates")))?;
    for parent_slot in 0..beams {
        let parent_global = beam_row(source, beams, parent_slot)
            .ok_or_else(|| Error::Generation("beam parent index overflow".into()))?;
        let Some(parent) = active.get(parent_global).and_then(Option::as_ref) else {
            continue;
        };
        let row = probability_row(probabilities, parent_global, vocabulary)?;
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
            let candidate = WorstFirst(Candidate {
                parent_global,
                parent_slot,
                token_id,
                log_probability,
                cumulative_log_probability,
            });
            if frontier.len() < capacity {
                frontier.push(candidate);
            } else if let Some(mut worst) = frontier.peek_mut()
                && candidate < *worst
            {
                *worst = candidate;
            }
        }
    }
    Ok(frontier
        .into_sorted_vec()
        .into_iter()
        .map(|ranked| ranked.0)
        .collect())
}

/// Borrows the log-probability row of one expanded beam.
fn probability_row(probabilities: &[f64], row: usize, vocabulary: usize) -> Result<&[f64]> {
    let start = row
        .checked_mul(vocabulary)
        .ok_or_else(|| Error::Generation("beam probability offset overflow".into()))?;
    let end = start
        .checked_add(vocabulary)
        .ok_or_else(|| Error::Generation("beam probability row end overflow".into()))?;
    probabilities
        .get(start..end)
        .ok_or_else(|| Error::Generation("beam probability matrix became ragged".into()))
}

/// Defines deterministic candidate order independently of backend sort rules.
fn candidate_order(left: &Candidate, right: &Candidate) -> core::cmp::Ordering {
    right
        .cumulative_log_probability
        .total_cmp(&left.cumulative_log_probability)
        .then_with(|| left.parent_slot.cmp(&right.parent_slot))
        .then_with(|| left.token_id.cmp(&right.token_id))
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

/// Computes a length-normalized score, checked for overflow before public output.
fn normalized_score(hypothesis: &BeamHypothesis, length_penalty: f64) -> f64 {
    hypothesis.cumulative_log_probability / (hypothesis.token_ids.len() as f64).powf(length_penalty)
}

/// Converts internal finished pools into stable public output.
///
/// A finite cumulative log probability divided by a positive subnormal length
/// normalizer can still overflow F64. Such scores are rejected contextually,
/// without changing the requested penalty or clamping the mathematical result.
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
    let mut sequences = metadata::vec_with_capacity(completed.len(), "beam-search output rows")?;
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
        let mut returned =
            metadata::vec_with_capacity(search.num_return_sequences, "returned beam hypotheses")?;
        for hypothesis in hypotheses.drain(..search.num_return_sequences) {
            let score = normalized_score(&hypothesis, search.length_penalty);
            if !score.is_finite() {
                return Err(Error::Generation(format!(
                    "beam-search source row {source} produced a non-finite normalized score for {} generated tokens",
                    hypothesis.token_ids.len(),
                )));
            }
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

/// Builds the initial group-major selector: row `i` copies source `i / beams`.
fn beam_expansion_indices(batch: usize, beams: usize) -> Result<Vec<usize>> {
    if beams == 0 {
        return Err(Error::Generation(
            "beam expansion requires at least one beam".into(),
        ));
    }
    let count = metadata::rectangle_size(batch, beams, "beam expansion")?;
    let mut indices = metadata::vec_with_capacity(count, "beam expansion indices")?;
    fill_beam_expansion(&mut indices, count, beams);
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

/// Computes row-wise log-softmax in at least F32 and copies it to the host.
///
/// Inactive rows are zeroed first so their arbitrary (possibly NaN) scores
/// cannot reach the host ranking. F16/BF16 logits are promoted to F32 before
/// the log-sum-exp, as Hugging Face does, so near-tied beams are not ranked
/// by half-precision rounding; F64 stays F64.
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
    let dtype = probability_accumulation_dtype(logits.dtype());
    let log_probabilities =
        log_softmax(logits.cast(dtype).mask_fill(active_mask.bool_not(), 0.0), 1);
    metadata::tensor_f64_values(log_probabilities, "beam-search log probabilities")
}

#[cfg(test)]
#[path = "seq2seq_tests.rs"]
mod tests;
