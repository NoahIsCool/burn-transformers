//! Backend-generic token sampling, logits processing, and host-side stopping state.
//!
//! Generation is split into small pieces with one job each:
//!
//! - [`LogitsProcessor`] turns raw `[batch, vocabulary]` model scores into the
//!   distribution that is actually sampled, entirely on the selected Burn
//!   backend, and selects one token per row.
//! - [`TokenHistory`] keeps the per-row statistics that history-dependent
//!   processing needs (token counts on the device, n-grams and generated
//!   lengths on the host). It is updated incrementally with each step's new
//!   tokens instead of being rebuilt from the whole history every step.
//! - [`GenerationTracker`] keeps emitted IDs on the host, where stop-suffix
//!   matching and finish reasons are cheap and deterministic.
//! - The coordinators [`generate_causal`] and [`generate_seq2seq_beam_search`]
//!   drive models and caches.
//!
//! # Processing order
//!
//! For logits `z` of one row, [`LogitsProcessor::process_with_history`] applies,
//! in this order (matching the Hugging Face Transformers 5.x processor order
//! and vLLM's sampler):
//!
//! 1. **Logit bias**: `z_i += bias_i` for configured token IDs.
//! 2. **Hard constraints**: tokens that would repeat an n-gram of
//!    `no_repeat_ngram_size`, and EOS IDs before `min_new_tokens`, get `-inf`.
//! 3. **Penalties**: repetition `z_i / r` (or `z_i * r` when `z_i < 0`) for
//!    every token seen in the prompt or output; then frequency and presence,
//!    `z_i -= f * c_i + p * [c_i > 0]`, where `c_i` counts occurrences in the
//!    [`PenaltyScope`] (generated tokens only by default, like OpenAI and vLLM).
//! 4. **Sampling warpers** (only for [`SamplingStrategy::Sample`]):
//!    temperature `z / T`, top-k, top-p, then min-p.
//!
//! Greedy decoding therefore ignores temperature and truncation, as in Hugging
//! Face, and its reported log probabilities come from the penalized,
//! untruncated distribution. Steps 1 and 2 commute with step 3 because `-inf`
//! is a fixed point of every penalty.
//!
//! Temperature sampling draws from `p(i) = exp(z_i / T) / sum_j exp(z_j / T)`.
//! Top-k keeps exactly the `k` greatest scores. Nucleus (top-p) keeps the
//! shortest sorted prefix whose probability mass reaches `p`; equivalently, a
//! sorted item survives when the mass *before* it is below `p`. Min-p keeps
//! `p_i >= min_p * max_j p_j`, evaluated as `z_i >= max_j z_j + ln(min_p)`
//! without a sort. Rejected scores become negative infinity.
//!
//! # Precision
//!
//! Bias, constraints, penalties, and warpers keep the input runtime dtype,
//! which need not equal the backend's default float type. Probability sums and
//! returned log probabilities promote F16/BF16 to F32 (Hugging Face upcasts
//! logits to F32 for the same reason); native F64 stays F64. Flex32 requests
//! F32 arithmetic, but a backend may keep its F32 storage alias. Sampling with
//! active top-k or top-p on Flex32 fails contextually because Burn 0.21's
//! fallback sort cannot read that storage tag.

mod bookkeeping;
mod causal;
mod history;
mod metadata;
mod random;
mod seq2seq;

pub use causal::{
    CausalGenerationOutput, CausalGenerationRequest, GeneratedSequence, GenerationToken,
    generate_causal, generate_causal_streaming,
};
pub use history::TokenHistory;
pub use random::SplitMix64;
pub use seq2seq::{
    BeamSearchConfig, BeamSearchSequence, Seq2SeqBeamSearchOutput, Seq2SeqBeamSearchRequest,
    generate_seq2seq_beam_search,
};

use core::marker::PhantomData;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use burn::tensor::{
    Bool, DType, IndexingUpdateOp, Int, Shape, Tensor, TensorData,
    activation::{log_softmax, softmax},
    backend::Backend,
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result, TokenMask};

use bookkeeping::{StopDecision, stop_decision};

/// Selects deterministic greedy decoding or stochastic categorical sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategy {
    /// Select the greatest processed logit for each sequence.
    #[default]
    Greedy,
    /// Sample from the processed categorical distribution.
    Sample,
}

/// Which earlier tokens the additive presence and frequency penalties count.
///
/// The multiplicative repetition penalty always counts prompt and generated
/// tokens, as in Hugging Face Transformers and vLLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PenaltyScope {
    /// Count only tokens generated after the prompt (OpenAI and vLLM semantics).
    ///
    /// A long document prompt then does not penalize copying from it.
    #[default]
    Generated,
    /// Count prompt tokens as well as generated tokens.
    PromptAndGenerated,
}

/// Serializable controls shared by ordinary and speculative generation.
///
/// Every field has a serde default, so older JSON without the newer controls
/// still deserializes to the same behavior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerationConfig {
    /// Maximum number of new tokens generated for each sequence.
    pub max_new_tokens: usize,
    /// Minimum number of new tokens before a sequence may finish early.
    ///
    /// Until a row has this many tokens, its EOS logits are set to `-inf` and
    /// stop sequences are not checked (vLLM `min_tokens`; Hugging Face
    /// `min_new_tokens`). Zero disables the constraint. Must not exceed
    /// `max_new_tokens`.
    pub min_new_tokens: usize,
    /// Greedy or stochastic token selection.
    pub strategy: SamplingStrategy,
    /// Positive softmax temperature used for stochastic sampling.
    pub temperature: f64,
    /// Optional exact count of greatest logits retained before sampling.
    pub top_k: Option<usize>,
    /// Optional nucleus probability in the interval `(0, 1]`.
    pub top_p: Option<f64>,
    /// Optional min-p cutoff in `[0, 1]` applied when sampling.
    ///
    /// Keeps tokens whose probability is at least `min_p` times the most
    /// likely token's probability, after temperature, top-k, and top-p
    /// (Hugging Face `MinPLogitsWarper`). `Some(0.0)` keeps everything.
    pub min_p: Option<f64>,
    /// Multiplicative repetition penalty; one disables it.
    ///
    /// Applies to every token seen in the prompt or the output: positive
    /// logits are divided by the penalty and negative logits multiplied by it.
    pub repetition_penalty: f64,
    /// Additive penalty subtracted once from every token counted by
    /// [`Self::additive_penalty_scope`].
    pub presence_penalty: f64,
    /// Additive penalty multiplied by each token's occurrence count in
    /// [`Self::additive_penalty_scope`] (OpenAI `frequency_penalty`).
    pub frequency_penalty: f64,
    /// Which tokens the presence and frequency penalties count.
    pub additive_penalty_scope: PenaltyScope,
    /// Forbid generating any n-gram of this size that already occurs in the
    /// prompt or output (Hugging Face `no_repeat_ngram_size`); zero disables.
    pub no_repeat_ngram_size: usize,
    /// Finite additive logit offsets keyed by token ID, applied before
    /// penalties (OpenAI and vLLM `logit_bias`).
    pub logit_bias: BTreeMap<u32, f64>,
    /// Per-processor random seed used to make sampling reproducible.
    pub seed: u64,
    /// Token IDs that terminate a sequence after being emitted.
    pub eos_token_ids: Vec<u32>,
    /// Optional padding token used by a batched generation coordinator.
    pub pad_token_id: Option<u32>,
    /// Token sequences that terminate generation when matched as a suffix.
    pub stop_sequences: Vec<Vec<u32>>,
}

impl Default for GenerationConfig {
    /// Returns conservative deterministic generation defaults.
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            min_new_tokens: 0,
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            additive_penalty_scope: PenaltyScope::Generated,
            no_repeat_ngram_size: 0,
            logit_bias: BTreeMap::new(),
            seed: 0,
            eos_token_ids: Vec::new(),
            pad_token_id: None,
            stop_sequences: Vec::new(),
        }
    }
}

impl GenerationConfig {
    /// Validates generation parameters against a model vocabulary size.
    pub fn validate(&self, vocabulary_size: usize) -> Result<()> {
        if vocabulary_size == 0 {
            return Err(Error::Generation(
                "vocabulary_size must be greater than zero".into(),
            ));
        }
        if self.max_new_tokens == 0 {
            return Err(Error::Generation(
                "max_new_tokens must be greater than zero".into(),
            ));
        }
        if self.min_new_tokens > self.max_new_tokens {
            return Err(Error::Generation(format!(
                "min_new_tokens {} exceeds max_new_tokens {}",
                self.min_new_tokens, self.max_new_tokens
            )));
        }
        self.validate_sampling(vocabulary_size)?;
        self.validate_penalties()?;
        self.validate_token_ids(vocabulary_size)
    }

    /// Checks temperature and the truncation warpers.
    fn validate_sampling(&self, vocabulary_size: usize) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(Error::Generation(
                "temperature must be finite and greater than zero".into(),
            ));
        }
        if let Some(top_k) = self.top_k
            && !(1..=vocabulary_size).contains(&top_k)
        {
            return Err(Error::Generation(format!(
                "top_k must be in 1..={vocabulary_size}, found {top_k}"
            )));
        }
        if let Some(top_p) = self.top_p
            && !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0)
        {
            return Err(Error::Generation(format!(
                "top_p must be finite and in (0, 1], found {top_p}"
            )));
        }
        if let Some(min_p) = self.min_p
            && !(0.0..=1.0).contains(&min_p)
        {
            return Err(Error::Generation(format!(
                "min_p must be in [0, 1], found {min_p}"
            )));
        }
        Ok(())
    }

    /// Checks the repetition, presence, and frequency penalties.
    fn validate_penalties(&self) -> Result<()> {
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(Error::Generation(
                "repetition_penalty must be finite and greater than zero".into(),
            ));
        }
        if !self.presence_penalty.is_finite() {
            return Err(Error::Generation("presence_penalty must be finite".into()));
        }
        if !self.frequency_penalty.is_finite() {
            return Err(Error::Generation("frequency_penalty must be finite".into()));
        }
        Ok(())
    }

    /// Checks that every configured token ID and bias lies inside the vocabulary.
    fn validate_token_ids(&self, vocabulary_size: usize) -> Result<()> {
        for token in self
            .eos_token_ids
            .iter()
            .copied()
            .chain(self.pad_token_id)
            .chain(self.stop_sequences.iter().flatten().copied())
        {
            if token as usize >= vocabulary_size {
                return Err(Error::Generation(format!(
                    "special token {token} is outside vocabulary size {vocabulary_size}"
                )));
            }
        }
        if self.stop_sequences.iter().any(Vec::is_empty) {
            return Err(Error::Generation("stop sequences must not be empty".into()));
        }
        for (&token, &bias) in &self.logit_bias {
            if token as usize >= vocabulary_size {
                return Err(Error::Generation(format!(
                    "logit_bias token {token} is outside vocabulary size {vocabulary_size}"
                )));
            }
            if !bias.is_finite() {
                return Err(Error::Generation(format!(
                    "logit_bias for token {token} must be finite, found {bias}"
                )));
            }
        }
        Ok(())
    }

    /// Whether any count-based penalty is active.
    pub(crate) fn uses_token_counts(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
    }

    /// Whether the sampling warpers include a vocabulary sort.
    fn sorts_vocabulary(&self, vocabulary_size: usize) -> bool {
        self.strategy == SamplingStrategy::Sample
            && (self.top_k.is_some_and(|k| k < vocabulary_size)
                || self.top_p.is_some_and(|p| p < 1.0))
    }
}

/// Applies bias, constraints, penalties, and sampling warpers on the backend.
#[derive(Debug)]
pub struct LogitsProcessor<B: Backend> {
    /// Validated penalties, sampling controls, and deterministic seed policy.
    config: GenerationConfig,
    /// Expected final-axis extent of every processed logit tensor.
    vocabulary_size: usize,
    /// Position in the processor's reproducible sampling stream.
    sample_counter: AtomicU64,
    /// Backend association for device-side processing without owned tensors.
    _backend: PhantomData<B>,
}

impl<B: Backend> Clone for LogitsProcessor<B> {
    /// Clones the configuration and the current position in its random stream.
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            vocabulary_size: self.vocabulary_size,
            sample_counter: AtomicU64::new(self.sample_counter.load(Ordering::Relaxed)),
            _backend: PhantomData,
        }
    }
}

/// Tokens selected for one generation step and their processed log probabilities.
#[derive(Debug)]
pub struct TokenSelection<B: Backend> {
    /// One selected vocabulary ID per batch row.
    pub token_ids: Tensor<B, 1, Int>,
    /// Natural-log probability of each selected ID under the processed
    /// distribution, in F32 for half-precision logits and F64 for F64 logits.
    pub log_probabilities: Tensor<B, 1>,
}

impl<B: Backend> LogitsProcessor<B> {
    /// Creates a validated processor with an independent random stream.
    ///
    /// Each categorical draw consumes one counter of a SplitMix64 stream (see
    /// [`SplitMix64`]) and transfers only one uniform value per batch row to
    /// the backend for inverse-CDF selection.
    pub fn new(
        config: GenerationConfig,
        vocabulary_size: usize,
        _device: &B::Device,
    ) -> Result<Self> {
        config.validate(vocabulary_size)?;
        Ok(Self {
            config,
            vocabulary_size,
            sample_counter: AtomicU64::new(0),
            _backend: PhantomData,
        })
    }

    /// Returns the immutable normalized generation controls.
    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }

    /// Returns the vocabulary extent this processor accepts.
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// Processes `[batch, vocabulary]` logits using optional token history.
    ///
    /// History has shape `[batch, prior_sequence]`. Every supplied token counts
    /// as a previously *generated* token: it is repetition-, presence-, and
    /// frequency-penalized, forms n-grams, and counts toward
    /// `min_new_tokens`. Callers that distinguish prompt from output should
    /// keep a [`TokenHistory`] and call [`Self::process_with_history`].
    pub fn process(
        &self,
        logits: Tensor<B, 2>,
        history: Option<Tensor<B, 2, Int>>,
    ) -> Result<Tensor<B, 2>> {
        self.process_with_mask(logits, history, None)
    }

    /// Processes logits while excluding padded history tokens.
    ///
    /// `history_mask` follows the crate-wide `true == valid` convention. Masked
    /// token slots may contain arbitrary sentinel values: they are replaced
    /// before any indexing and affect no penalty, n-gram, or length count.
    /// This convenience path rebuilds a [`TokenHistory`] from the tensor on
    /// every call (one host read when any history-dependent control is
    /// active); incremental callers should keep a [`TokenHistory`] instead.
    pub fn process_with_mask(
        &self,
        logits: Tensor<B, 2>,
        history: Option<Tensor<B, 2, Int>>,
        history_mask: Option<TokenMask<B>>,
    ) -> Result<Tensor<B, 2>> {
        let batch = self.validate_logits_shape(&logits)?;
        let history = match history {
            Some(history) => TokenHistory::from_generated_tensor(
                &self.config,
                self.vocabulary_size,
                batch,
                history,
                history_mask,
            )?,
            None if history_mask.is_some() => {
                return Err(Error::Generation(
                    "history_mask requires a corresponding history tensor".into(),
                ));
            }
            None => {
                TokenHistory::empty(&self.config, self.vocabulary_size, batch, &logits.device())?
            }
        };
        self.process_with_history(logits, &history)
    }

    /// Processes logits using incrementally maintained per-row history.
    ///
    /// This is the efficient path used by the coordinators: penalties read
    /// `[batch, vocabulary]` count tensors already on the device, and no host
    /// synchronization occurs. `history` must have been created from a
    /// configuration with the same history-dependent controls.
    pub fn process_with_history(
        &self,
        logits: Tensor<B, 2>,
        history: &TokenHistory<B>,
    ) -> Result<Tensor<B, 2>> {
        let batch = self.validate_logits_shape(&logits)?;
        history.check_compatible(&self.config, self.vocabulary_size, batch)?;
        if logits.dtype() == DType::Flex32 && self.config.sorts_vocabulary(self.vocabulary_size) {
            return Err(Error::BackendCapability(
                "Burn 0.21 does not support Flex32 top-k/top-p sorting; use F32 logits or disable those filters explicitly".into(),
            ));
        }
        let logits = self.apply_sparse_adjustments(logits, history)?;
        let logits = self.apply_token_penalties(logits, history);
        Ok(self.apply_sampling_warpers(logits))
    }

    /// Selects one token per row from already processed logits.
    pub fn select(&self, logits: Tensor<B, 2>) -> Result<Tensor<B, 1, Int>> {
        let batch = self.validate_logits_shape(&logits)?;
        match self.config.strategy {
            SamplingStrategy::Greedy => Ok(logits.argmax(1).squeeze_dim::<1>(1)),
            SamplingStrategy::Sample => {
                let uniforms = self.next_uniforms(batch)?;
                Ok(sample_with_uniforms(logits, uniforms))
            }
        }
    }

    /// Reserves and evaluates one deterministic random value per batch row.
    fn next_uniforms(&self, batch: usize) -> Result<Vec<f32>> {
        let count = u64::try_from(batch).map_err(|_| {
            Error::Generation(format!(
                "sampling batch size {batch} does not fit the u64 random stream"
            ))
        })?;
        // One atomic reservation gives this call the disjoint counter block
        // `start..start + count`, so concurrent callers never share a draw.
        let start = self
            .sample_counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |position| {
                position.checked_add(count)
            })
            .map_err(|position| {
                Error::Generation(format!(
                    "sampling random stream exhausted at position {position}"
                ))
            })?;
        let mut uniforms = metadata::vec_with_capacity(batch, "categorical sampling thresholds")?;
        for offset in 0..count {
            uniforms.push(splitmix64_uniform(self.config.seed, start + offset));
        }
        Ok(uniforms)
    }

    /// Processes logits and immediately selects the next token for each row.
    pub fn process_and_select(
        &self,
        logits: Tensor<B, 2>,
        history: Option<Tensor<B, 2, Int>>,
    ) -> Result<Tensor<B, 1, Int>> {
        let logits = self.process(logits, history)?;
        self.select(logits)
    }

    /// Processes one step and returns both selected IDs and their log probabilities.
    ///
    /// Log probabilities describe the distribution actually used: after bias,
    /// constraints, and penalties for both strategies, plus temperature and
    /// truncation when sampling. Greedy log probabilities therefore match
    /// Hugging Face `compute_transition_scores` for `do_sample=False`.
    pub fn process_and_select_with_log_probs(
        &self,
        logits: Tensor<B, 2>,
        history: Option<Tensor<B, 2, Int>>,
    ) -> Result<TokenSelection<B>> {
        let logits = self.process(logits, history)?;
        self.select_with_log_probs(logits)
    }

    /// Selects from processed logits and returns the selected log probability.
    ///
    /// The log-softmax runs in F32 for half-precision logits (F64 stays F64),
    /// so reported scores do not carry F16 log-sum-exp rounding.
    pub fn select_with_log_probs(&self, logits: Tensor<B, 2>) -> Result<TokenSelection<B>> {
        let token_ids = self.select(logits.clone())?;
        let batch = token_ids.dims()[0];
        let dtype = probability_accumulation_dtype(logits.dtype());
        let selected = log_softmax(logits.cast(dtype), 1)
            .gather(1, token_ids.clone().reshape([batch, 1]))
            .squeeze_dim::<1>(1);
        Ok(TokenSelection {
            token_ids,
            log_probabilities: selected,
        })
    }

    /// Validates a public logits boundary before backend reductions or sampling.
    fn validate_logits_shape(&self, logits: &Tensor<B, 2>) -> Result<usize> {
        let [batch, vocabulary_size] = logits.dims();
        if batch == 0 || vocabulary_size != self.vocabulary_size {
            return Err(Error::shape(
                "generation.logits",
                format!("[nonzero_batch, {}]", self.vocabulary_size),
                format!("[{batch}, {vocabulary_size}]"),
            ));
        }
        Ok(batch)
    }

    /// Whether `row` must not select an EOS token yet.
    fn suppresses_eos(&self, history: &TokenHistory<B>, row: usize) -> bool {
        !self.config.eos_token_ids.is_empty()
            && history.generated_lengths()[row] < self.config.min_new_tokens
    }

    /// Adds logit bias and `-inf` constraints with one sparse scatter.
    ///
    /// Each row receives `(token, delta)` pairs: the configured bias for every
    /// row, `-inf` for EOS IDs while the row is below `min_new_tokens`, and
    /// `-inf` for tokens banned by the n-gram constraint. Rows are padded to a
    /// common width with `(0, 0.0)`, which adds nothing. Only the small
    /// `[batch, width]` index and value tensors cross to the device.
    fn apply_sparse_adjustments(
        &self,
        logits: Tensor<B, 2>,
        history: &TokenHistory<B>,
    ) -> Result<Tensor<B, 2>> {
        let batch = logits.dims()[0];
        let mut width = 0;
        for row in 0..batch {
            let mut count = self.config.logit_bias.len() + history.banned_tokens(row).len();
            if self.suppresses_eos(history, row) {
                count += self.config.eos_token_ids.len();
            }
            width = width.max(count);
        }
        if width == 0 {
            return Ok(logits);
        }
        let elements = batch
            .checked_mul(width)
            .ok_or_else(|| Error::Generation("sparse logit adjustment size overflows".into()))?;
        let mut indices = metadata::vec_with_capacity::<i64>(elements, "logit adjustment indices")?;
        let mut deltas = metadata::vec_with_capacity::<f64>(elements, "logit adjustment values")?;
        for row in 0..batch {
            let row_end = indices.len() + width;
            for (&token, &bias) in &self.config.logit_bias {
                indices.push(i64::from(token));
                deltas.push(bias);
            }
            if self.suppresses_eos(history, row) {
                for &token in &self.config.eos_token_ids {
                    indices.push(i64::from(token));
                    deltas.push(f64::NEG_INFINITY);
                }
            }
            for &token in history.banned_tokens(row) {
                indices.push(i64::from(token));
                deltas.push(f64::NEG_INFINITY);
            }
            // Pad to the common width with a no-op addition at token zero.
            indices.resize(row_end, 0);
            deltas.resize(row_end, 0.0);
        }
        let device = logits.device();
        let dtype = logits.dtype();
        let indices =
            Tensor::<B, 2, Int>::from_data(TensorData::new(indices, [batch, width]), &device);
        let deltas = float_tensor_in_dtype::<B, 2>(deltas, [batch, width], dtype, &device);
        Ok(logits.scatter(1, indices, deltas, IndexingUpdateOp::Add))
    }

    /// Applies sign-aware repetition and additive frequency/presence penalties.
    ///
    /// For scope counts `c` and the seen-anywhere indicator `s`:
    /// `z_i <- s_i ? (z_i < 0 ? z_i * r : z_i / r) : z_i`, then
    /// `z_i <- z_i - f * c_i - p * [c_i > 0]`. All work is `[batch,
    /// vocabulary]` device arithmetic on counts the history already holds.
    fn apply_token_penalties(
        &self,
        logits: Tensor<B, 2>,
        history: &TokenHistory<B>,
    ) -> Tensor<B, 2> {
        let Some(counts) = history.counts() else {
            return logits;
        };
        let dtype = logits.dtype();
        let mut logits = logits;
        if self.config.repetition_penalty != 1.0 {
            let seen = counts.all.clone().greater_elem(0);
            let penalized = logits
                .clone()
                .div_scalar(self.config.repetition_penalty)
                .mask_where(
                    logits.clone().lower_elem(0.0),
                    logits.clone().mul_scalar(self.config.repetition_penalty),
                );
            logits = logits.mask_where(seen, penalized);
        }
        let additive = match self.config.additive_penalty_scope {
            PenaltyScope::Generated => &counts.generated,
            PenaltyScope::PromptAndGenerated => &counts.all,
        };
        if self.config.frequency_penalty != 0.0 {
            let frequency = additive
                .clone()
                .float()
                .cast(dtype)
                .mul_scalar(self.config.frequency_penalty);
            logits = logits - frequency;
        }
        if self.config.presence_penalty != 0.0 {
            let presence = additive
                .clone()
                .greater_elem(0)
                .float()
                .cast(dtype)
                .mul_scalar(self.config.presence_penalty);
            logits = logits - presence;
        }
        logits
    }

    /// Applies temperature, top-k, top-p, and min-p when sampling.
    fn apply_sampling_warpers(&self, logits: Tensor<B, 2>) -> Tensor<B, 2> {
        if self.config.strategy != SamplingStrategy::Sample {
            return logits;
        }
        let mut logits = logits.div_scalar(self.config.temperature);
        // `k == vocabulary` keeps every token, so it needs no sort.
        if let Some(top_k) = self.config.top_k
            && top_k < self.vocabulary_size
        {
            logits = filter_top_k(logits, top_k);
        }
        if let Some(top_p) = self.config.top_p
            && top_p < 1.0
        {
            logits = filter_top_p(logits, top_p);
        }
        if let Some(min_p) = self.config.min_p
            && min_p > 0.0
        {
            logits = filter_min_p(logits, min_p);
        }
        logits
    }
}

/// Requests at least F32 precision for probability arithmetic.
///
/// Half-precision cumulative sums can lose small masses or reach one early.
/// F32 also exactly represents every seeded uniform; F64 scores are never
/// narrowed, so backends with native double precision keep that precision.
/// CubeCL may retain the Flex32 storage tag when casting its F32-backed alias.
fn probability_accumulation_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 | DType::Flex32 => DType::F32,
        _ => dtype,
    }
}

/// Creates a float tensor from F64 host values in the requested runtime dtype.
///
/// F64 targets keep every value exactly; other targets are built as F32 and
/// cast, which keeps `-inf` and rounds finite values once.
fn float_tensor_in_dtype<B: Backend, const D: usize>(
    values: Vec<f64>,
    shape: [usize; D],
    dtype: DType,
    device: &B::Device,
) -> Tensor<B, D> {
    if dtype == DType::F64 {
        Tensor::from_data(TensorData::new(values, shape), (device, DType::F64))
    } else {
        let values = values
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        Tensor::<B, D>::from_data(TensorData::new(values, shape), (device, DType::F32)).cast(dtype)
    }
}

/// Returns uniform draw `index` of the processor stream seeded with `seed`.
///
/// Values lie strictly inside `(0, 1)` and are exact in F32 (see
/// `random::interior_unit_f32`).
fn splitmix64_uniform(seed: u64, index: u64) -> f32 {
    random::interior_unit_f32(random::splitmix64(seed, index))
}

/// Selects one category per row by inverse-CDF search with seeded uniforms.
///
/// With probabilities `p` and running sums `c_k = p_0 + ... + p_k`, the
/// selected index is `#{k : c_k < u * c_(V-1)}`. Scaling the uniform by the
/// row's *realized* total `c_(V-1)` matters: the F32 softmax and running sum
/// of a 150k-token vocabulary end measurably away from one (about `1e-5` on
/// Flex, up to `3e-4` when the normalizer is summed pairwise), and an unscaled
/// `u` above that total would select index `V`, clamped to the last (often
/// padded) vocabulary row. Because `u <= 1 - 2^-24`, the scaled threshold rounds strictly below
/// `c_(V-1)` in F32 and F64, so the count is at most `V - 1`; and the chosen
/// index satisfies `c_(k-1) < threshold <= c_k`, so it has positive mass
/// whenever the backend's cumulative sum is monotone. Tail probabilities
/// smaller than half an ulp of the running sum are absorbed by F32 addition
/// and cannot be drawn; F64 logits avoid that residual bias.
fn sample_with_uniforms<B: Backend>(logits: Tensor<B, 2>, uniforms: Vec<f32>) -> Tensor<B, 1, Int> {
    let [batch, vocabulary] = logits.dims();
    let device = logits.device();
    let probability_dtype = probability_accumulation_dtype(logits.dtype());
    let cumulative = softmax(logits.cast(probability_dtype), 1).cumsum(1);
    let total = cumulative.clone().narrow(1, vocabulary - 1, 1);
    // Create the uniforms explicitly as F32 so a device whose default float
    // type is F16 cannot round an interior draw to zero or one.
    let uniforms =
        Tensor::<B, 2>::from_data(TensorData::new(uniforms, [batch, 1]), (&device, DType::F32))
            .cast(probability_dtype);
    let thresholds = (uniforms * total).expand([batch, vocabulary]);
    cumulative
        .lower(thresholds)
        .int()
        .sum_dim(1)
        .squeeze_dim::<1>(1)
}

/// Keeps exactly the requested greatest values in every batch row.
fn filter_top_k<B: Backend>(logits: Tensor<B, 2>, top_k: usize) -> Tensor<B, 2> {
    let [batch, vocabulary] = logits.dims();
    let (_, indices) = logits.clone().sort_descending_with_indices(1);
    let indices = indices.narrow(1, 0, top_k);
    let keep = scatter_selection_mask(indices, batch, vocabulary);
    logits.mask_fill(keep.bool_not(), f64::NEG_INFINITY)
}

/// Keeps the minimal descending-probability prefix whose mass reaches `top_p`.
fn filter_top_p<B: Backend>(logits: Tensor<B, 2>, top_p: f64) -> Tensor<B, 2> {
    let [batch, vocabulary] = logits.dims();
    let probabilities = softmax(
        logits
            .clone()
            .cast(probability_accumulation_dtype(logits.dtype())),
        1,
    );
    let (sorted, indices) = probabilities.sort_descending_with_indices(1);
    let mass_before = sorted.clone().cumsum(1) - sorted;
    let keep_sorted = mass_before.lower_elem(top_p);
    let keep = scatter_mask(indices, keep_sorted, batch, vocabulary);
    logits.mask_fill(keep.bool_not(), f64::NEG_INFINITY)
}

/// Keeps tokens with `p_i >= min_p * max_j p_j`, evaluated in log space.
///
/// Since `p_i / p_max = exp(z_i - z_max)`, the condition is
/// `z_i >= z_max + ln(min_p)`: one row maximum and one comparison, with no
/// softmax or sort. The comparison runs in at least F32. The most likely
/// token always survives because `ln(min_p) <= 0`.
fn filter_min_p<B: Backend>(logits: Tensor<B, 2>, min_p: f64) -> Tensor<B, 2> {
    let [batch, vocabulary] = logits.dims();
    let scores = logits
        .clone()
        .cast(probability_accumulation_dtype(logits.dtype()));
    let threshold = scores
        .clone()
        .max_dim(1)
        .add_scalar(min_p.ln())
        .expand([batch, vocabulary]);
    logits.mask_fill(scores.lower(threshold), f64::NEG_INFINITY)
}

/// Converts per-row selected vocabulary indices into a dense Boolean mask.
fn scatter_selection_mask<B: Backend>(
    indices: Tensor<B, 2, Int>,
    batch: usize,
    vocabulary: usize,
) -> Tensor<B, 2, Bool> {
    let width = indices.dims()[1];
    let device = indices.device();
    let selected = Tensor::<B, 2, Bool>::ones([batch, width], &device);
    scatter_mask(indices, selected, batch, vocabulary)
}

/// Scatters a sorted Boolean selection back to original vocabulary order.
fn scatter_mask<B: Backend>(
    vocabulary_indices: Tensor<B, 2, Int>,
    values: Tensor<B, 2, Bool>,
    batch: usize,
    vocabulary: usize,
) -> Tensor<B, 2, Bool> {
    let width = vocabulary_indices.dims()[1];
    let device = vocabulary_indices.device();
    let rows = Tensor::<B, 1, Int>::arange(0..batch as i64, &device)
        .reshape([batch, 1])
        .expand([batch, width]);
    let coordinates = Tensor::cat(
        vec![
            rows.unsqueeze_dim::<3>(2),
            vocabulary_indices.unsqueeze_dim::<3>(2),
        ],
        2,
    );
    Tensor::<B, 2, Int>::zeros(Shape::new([batch, vocabulary]), &device)
        .scatter_nd(coordinates, values.int(), IndexingUpdateOp::Assign)
        .greater_elem(0)
}

/// Describes why generation stopped for one sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// An end-of-sequence token was emitted.
    EndOfSequence {
        /// The matching end-of-sequence token.
        token_id: u32,
    },
    /// A configured multi-token stop sequence was emitted.
    StopSequence {
        /// Index in [`GenerationConfig::stop_sequences`].
        index: usize,
        /// Number of tokens in the matching sequence.
        length: usize,
    },
    /// The configured generation length was exhausted.
    Length,
}

/// Tracks emitted token suffixes and terminal state for a batch.
#[derive(Debug, Clone)]
pub struct GenerationTracker {
    /// Validated stop conditions and generation length limits.
    config: GenerationConfig,
    /// Upper bound used to validate every emitted token identifier.
    vocabulary_size: usize,
    /// Generated suffix for each independent batch row, excluding the prompt.
    generated: Vec<Vec<u32>>,
    /// Terminal reason for each row, or `None` while that row remains active.
    finished: Vec<Option<FinishReason>>,
}

impl GenerationTracker {
    /// Creates empty host-side state for `batch_size` independent sequences.
    pub fn new(
        config: GenerationConfig,
        vocabulary_size: usize,
        batch_size: usize,
    ) -> Result<Self> {
        config.validate(vocabulary_size)?;
        if batch_size == 0 {
            return Err(Error::Generation(
                "generation batch_size must be greater than zero".into(),
            ));
        }
        let generated = metadata::empty_rows(batch_size, "generation state")?;
        let mut finished =
            metadata::vec_with_capacity::<Option<FinishReason>>(batch_size, "finish state")?;
        finished.resize(batch_size, None);
        Ok(Self {
            config,
            vocabulary_size,
            generated,
            finished,
        })
    }

    /// Returns the fixed number of independently tracked sequences.
    pub fn batch_size(&self) -> usize {
        self.generated.len()
    }

    /// Returns the vocabulary bound enforced for every emitted token.
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// Records a token and returns the sequence's finish reason, if any.
    ///
    /// Precedence is EOS, then the longest stop sequence (ties resolve to the
    /// earliest configured one, and stops only count after `min_new_tokens`),
    /// then the length limit. Calling this after the sequence has finished is
    /// idempotent: the token is ignored and the original reason is returned.
    pub fn push(&mut self, sequence: usize, token: u32) -> Result<Option<&FinishReason>> {
        if sequence >= self.generated.len() {
            return Err(Error::Generation(format!(
                "sequence index {sequence} is outside batch size {}",
                self.generated.len()
            )));
        }
        if self.finished[sequence].is_some() {
            return Ok(self.finished[sequence].as_ref());
        }
        if token as usize >= self.vocabulary_size {
            return Err(Error::Generation(format!(
                "generated token {token} is outside vocabulary size {}",
                self.vocabulary_size
            )));
        }

        self.generated[sequence].try_reserve(1).map_err(|_| {
            Error::Generation(format!(
                "failed to extend generated tokens for sequence {sequence}"
            ))
        })?;
        self.generated[sequence].push(token);
        let decision = stop_decision(
            self.config.eos_token_ids.contains(&token),
            &self.generated[sequence],
            &self.config.stop_sequences,
            self.config.min_new_tokens,
            self.config.max_new_tokens,
        );
        self.finished[sequence] = match decision {
            StopDecision::Continue => None,
            StopDecision::EndOfSequence => Some(FinishReason::EndOfSequence { token_id: token }),
            StopDecision::StopSequence { index, length } => {
                Some(FinishReason::StopSequence { index, length })
            }
            StopDecision::Length => Some(FinishReason::Length),
        };
        Ok(self.finished[sequence].as_ref())
    }

    /// Returns emitted tokens for one sequence.
    pub fn generated(&self, sequence: usize) -> Result<&[u32]> {
        self.generated
            .get(sequence)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                Error::Generation(format!(
                    "sequence index {sequence} is outside batch size {}",
                    self.generated.len()
                ))
            })
    }

    /// Returns the current finish reason for one sequence.
    pub fn finish_reason(&self, sequence: usize) -> Result<Option<&FinishReason>> {
        self.finished
            .get(sequence)
            .map(Option::as_ref)
            .ok_or_else(|| {
                Error::Generation(format!(
                    "sequence index {sequence} is outside batch size {}",
                    self.finished.len()
                ))
            })
    }

    /// Returns whether every sequence in the batch has finished.
    pub fn all_finished(&self) -> bool {
        self.finished.iter().all(Option::is_some)
    }
}

#[cfg(test)]
#[path = "generation_tests.rs"]
mod tests;
