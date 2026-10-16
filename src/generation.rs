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
//!   drive models and caches; [`speculative`] holds the family-independent
//!   acceptance rule of speculative decoding.
//!
//! # Processing order
//!
//! For logits `z` of one row, [`LogitsProcessor::process_with_history`] applies,
//! in this order (matching the Hugging Face Transformers 5.x processor order
//! and vLLM's sampler):
//!
//! 1. **Logit bias**: `z_i += bias_i` for configured token IDs.
//! 2. **Penalties**: repetition `z_i / r` (or `z_i * r` when `z_i < 0`) for
//!    every token seen in the prompt or output; then frequency and presence,
//!    `z_i -= f * c_i + p * [c_i > 0]`, where `c_i` counts occurrences in the
//!    [`PenaltyScope`] (generated tokens only by default, like OpenAI and vLLM).
//! 3. **Temperature** (only for [`SamplingStrategy::Sample`]): `z / T`.
//! 4. **Hard constraints**: tokens that would repeat an n-gram of
//!    `no_repeat_ngram_size`, and EOS IDs before `min_new_tokens`, become
//!    exactly `-inf`.
//! 5. **Truncation warpers** (only when sampling): top-k, top-p, then min-p.
//!
//! Greedy decoding therefore ignores temperature and truncation, as in Hugging
//! Face, and its reported log probabilities come from the penalized,
//! untruncated distribution. Step 4 commutes with steps 2 and 3 because `-inf`
//! is a fixed point of every penalty and of `/ T`; it comes after them, as in
//! Hugging Face (`RepetitionPenalty` before `NoRepeatNGram` and
//! `MinNewTokensLength`), so a constrained token is `-inf` whatever arithmetic
//! preceded it. Step 1 does not commute with step 2
//! (`(z + b) / r != z / r + b`); bias comes before penalties because Hugging
//! Face (`SequenceBias` before `RepetitionPenalty`) and vLLM apply it in that
//! order.
//!
//! Temperature sampling draws from `p(i) = exp(z_i / T) / sum_j exp(z_j / T)`.
//! Top-k keeps every token whose score is at least the `k`-th greatest score
//! `z_(k)`: `z_i >= z_(k)`. Tokens tied with the boundary score all survive, so
//! more than `k` may remain, exactly as Hugging Face's `TopKLogitsWarper`
//! (`scores < topk(scores, k)[..., -1]` is removed). Nucleus (top-p) keeps the
//! shortest sorted prefix whose probability mass reaches `p`; equivalently, a
//! sorted item survives when the mass *before* it is below `p`. Min-p keeps
//! `p_i >= min_p * max_j p_j`, evaluated as `z_i >= max_j z_j + ln(min_p)`
//! without a sort. Rejected scores become negative infinity.
//!
//! # Entry points
//!
//! - [`LogitsProcessor::process_with_history`] reads a [`TokenHistory`] kept
//!   across steps; [`generate_causal`] uses it.
//! - [`LogitsProcessor::process_with_prompt_lengths`] rebuilds that history
//!   from host token rows with an explicit prompt boundary and promotes
//!   half-precision logits like [`generate_causal`], so it reproduces the
//!   coordinator's processing exactly; it is the form speculative decoders need.
//! - [`LogitsProcessor::process`] and [`LogitsProcessor::process_with_mask`]
//!   keep their original meaning: every supplied history token counts as
//!   generated, and the logits dtype is kept.
//!
//! # Precision
//!
//! Bias, constraints, penalties, and warpers keep the input runtime dtype,
//! which need not equal the backend's default float type. Probability sums and
//! returned log probabilities promote F16/BF16 to F32 (Hugging Face upcasts
//! logits to F32 for the same reason); native F64 stays F64. The coordinators
//! and [`LogitsProcessor::process_with_prompt_lengths`] additionally promote
//! half-precision logits before any processing ([`upcast_half_precision`]). Flex32 requests
//! F32 arithmetic, but a backend may keep its F32 storage alias. Sampling with
//! active top-k or top-p on Flex32 fails contextually because Burn 0.21's
//! fallback sort cannot read that storage tag.
//!
//! ## Finite arithmetic
//!
//! A backend rounds each scalar control to the logits dtype before using it,
//! so a finite `f64` can still become zero or infinite: `temperature = 1e-46`
//! is `0.0_f32`, and `0 / 0` is NaN. Validation therefore happens in the
//! precision processing actually uses. [`GenerationConfig::validate`] checks
//! F32, the narrowest dtype the coordinators process in (F64 is wider):
//!
//! - the divisors `temperature` and `repetition_penalty` must round to
//!   *normal* positive values (`f32::MIN_POSITIVE..=f32::MAX`), since zero,
//!   infinity, and subnormals (which GPU kernels may flush to zero) break
//!   `z / T` and `z / r`;
//! - the additive `presence_penalty`, `frequency_penalty`, and every
//!   `logit_bias` must round to finite values (`|x| <= f32::MAX`).
//!
//! Logits processed in F16 or BF16 by [`LogitsProcessor::process`],
//! [`LogitsProcessor::process_with_mask`], or
//! [`LogitsProcessor::process_with_history`] are checked again against that
//! dtype's range (F16: `6.1e-5..=65504`). Every error names the field, the
//! dtype, and its range.
//!
//! Valid scalars still let the *results* overflow, for example a huge bias
//! plus a large logit, or `count * frequency_penalty`. For input logits that
//! are finite or `-inf`, processing is nevertheless closed over "finite or
//! `-inf`": `count * frequency_penalty` saturates at the dtype's largest
//! finite value before it is subtracted (so `-inf - (-inf)` cannot occur),
//! every other overflow is a `+-inf` that is never added to an infinity of
//! the opposite sign, positive overflow saturates at the largest finite value
//! before the constraints and truncation warpers run, and constraints *add*
//! `-inf` to values that can no longer be `+inf`. Processed logits are
//! therefore never NaN and never `+inf`, so the softmax of selection and of
//! top-p stays defined whenever a row keeps a finite logit. Saturation is a
//! no-op for values that did not overflow, so ordinary configurations are
//! bit-for-bit unchanged; it costs one elementwise clamp per step and only
//! runs when bias, a penalty, or a temperature below one can overflow.

mod bookkeeping;
mod causal;
mod history;
mod layout;
mod metadata;
mod ngram;
mod random;
mod seq2seq;
pub mod speculative;

pub use causal::{
    CausalGenerationOutput, CausalGenerationRequest, GeneratedSequence, GenerationToken,
    generate_causal, generate_causal_streaming,
};
pub use history::TokenHistory;
pub use random::SplitMix64;
pub use seq2seq::{
    BeamSearchConfig, BeamSearchSequence, EarlyStopping, Seq2SeqBeamSearchOutput,
    Seq2SeqBeamSearchRequest, generate_seq2seq_beam_search,
};

use core::marker::PhantomData;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use burn::tensor::{
    Bool, DType, IndexingUpdateOp, Int, Shape, Tensor, TensorData,
    activation::{log_softmax, softmax},
    backend::Backend,
    bf16, f16,
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result, TokenMask};

use bookkeeping::{StopDecision, stop_decision};
use history::CountView;
use layout::{first_outside_vocabulary, push_constraint_row, repeat_row};

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
    ///
    /// Must round to a normal positive F32 value (see the module's finite
    /// arithmetic section); values below one sharpen the distribution.
    pub temperature: f64,
    /// Optional top-k cutoff applied when sampling.
    ///
    /// Keeps every token scoring at least the `k`-th greatest logit, so tokens
    /// tied with that boundary score all survive (Hugging Face
    /// `TopKLogitsWarper` semantics). Must lie in `1..=vocabulary_size`.
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
    /// Must round to a normal positive F32 value.
    pub repetition_penalty: f64,
    /// Additive penalty subtracted once from every token counted by
    /// [`Self::additive_penalty_scope`]; must round to a finite F32 value.
    pub presence_penalty: f64,
    /// Additive penalty multiplied by each token's occurrence count in
    /// [`Self::additive_penalty_scope`] (OpenAI `frequency_penalty`); must
    /// round to a finite F32 value.
    pub frequency_penalty: f64,
    /// Which tokens the presence and frequency penalties count.
    pub additive_penalty_scope: PenaltyScope,
    /// Forbid generating any n-gram of this size that already occurs in the
    /// prompt or output (Hugging Face `no_repeat_ngram_size`); zero disables.
    pub no_repeat_ngram_size: usize,
    /// Additive logit offsets keyed by token ID, applied before penalties
    /// (OpenAI and vLLM `logit_bias`); each must round to a finite F32 value.
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
    ///
    /// Float controls are checked in F32, the narrowest precision the
    /// coordinators process in; see the module's finite-arithmetic section.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Generation`] naming the first invalid field (and, for
    /// float controls, the representable range).
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
        ScalarPrecision::F32.check_divisor("temperature", self.temperature)?;
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

    /// Checks the repetition, presence, and frequency penalties in F32.
    fn validate_penalties(&self) -> Result<()> {
        let precision = ScalarPrecision::F32;
        precision.check_divisor("repetition_penalty", self.repetition_penalty)?;
        precision.check_finite("presence_penalty", self.presence_penalty)?;
        precision.check_finite("frequency_penalty", self.frequency_penalty)
    }

    /// Checks every float control against the range of `precision`.
    ///
    /// [`Self::validate`] covers F32 through [`Self::validate_sampling`],
    /// [`Self::validate_penalties`], and [`Self::validate_token_ids`]; the
    /// processor calls this again for F16/BF16 logits, whose range is
    /// narrower (F16) or rounds its largest values differently (BF16).
    fn validate_scalars(&self, precision: ScalarPrecision) -> Result<()> {
        precision.check_divisor("temperature", self.temperature)?;
        precision.check_divisor("repetition_penalty", self.repetition_penalty)?;
        precision.check_finite("presence_penalty", self.presence_penalty)?;
        precision.check_finite("frequency_penalty", self.frequency_penalty)?;
        for (&token, &bias) in &self.logit_bias {
            precision.check_finite(&format!("logit_bias for token {token}"), bias)?;
        }
        Ok(())
    }

    /// Whether bias or a penalty changes logits arithmetically (and so can overflow).
    fn adjusts_logits(&self) -> bool {
        !self.logit_bias.is_empty()
            || self.repetition_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
    }

    /// Checks that every configured token ID and bias lies inside the vocabulary.
    ///
    /// Special tokens are checked in configuration order (EOS IDs, PAD, then
    /// each stop sequence) with the Verus-verified [`first_outside_vocabulary`],
    /// so the first offending token is reported. The same scan finds the first
    /// logit-bias key outside the vocabulary; entries before it are only
    /// checked for a finite F32 bias, so the error reported is the one the
    /// per-entry check (range, then finiteness) meets first.
    fn validate_token_ids(&self, vocabulary_size: usize) -> Result<()> {
        let pad = self.pad_token_id.as_slice();
        let special = [self.eos_token_ids.as_slice(), pad]
            .into_iter()
            .chain(self.stop_sequences.iter().map(Vec::as_slice));
        for tokens in special {
            if let Some(index) = first_outside_vocabulary(tokens, vocabulary_size) {
                let token = tokens[index];
                return Err(Error::Generation(format!(
                    "special token {token} is outside vocabulary size {vocabulary_size}"
                )));
            }
        }
        if self.stop_sequences.iter().any(Vec::is_empty) {
            return Err(Error::Generation("stop sequences must not be empty".into()));
        }
        let bias_tokens = self.logit_bias.keys().copied().collect::<Vec<_>>();
        let outside = first_outside_vocabulary(&bias_tokens, vocabulary_size);
        for (index, (&token, &bias)) in self.logit_bias.iter().enumerate() {
            if outside == Some(index) {
                return Err(Error::Generation(format!(
                    "logit_bias token {token} is outside vocabulary size {vocabulary_size}"
                )));
            }
            ScalarPrecision::F32.check_finite(&format!("logit_bias for token {token}"), bias)?;
        }
        Ok(())
    }

    /// Which occurrence counts the additive presence and frequency penalties read.
    ///
    /// `None` when both are zero, so no generated-only counts are tracked.
    fn additive_count_view(&self) -> Option<CountView> {
        let active = self.presence_penalty != 0.0 || self.frequency_penalty != 0.0;
        active.then_some(match self.additive_penalty_scope {
            PenaltyScope::Generated => CountView::Generated,
            PenaltyScope::PromptAndGenerated => CountView::All,
        })
    }

    /// Whether any configured penalty reads the occurrence counts of `view`.
    ///
    /// The repetition penalty reads [`CountView::All`]; the additive penalties
    /// read the view their [`PenaltyScope`] selects.
    fn reads_count_view(&self, view: CountView) -> bool {
        (view == CountView::All && self.repetition_penalty != 1.0)
            || self.additive_count_view() == Some(view)
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
    /// as a previously *generated* token: it is repetition-penalized, forms
    /// n-grams, is counted by presence and frequency whatever
    /// [`GenerationConfig::additive_penalty_scope`] says, and counts toward
    /// `min_new_tokens`. A history that includes the prompt therefore matches
    /// [`generate_causal`] only with [`PenaltyScope::PromptAndGenerated`] and
    /// `min_new_tokens == 0`: once it holds `min_new_tokens` prompt tokens,
    /// EOS is no longer suppressed. Logits keep their runtime dtype.
    ///
    /// To reproduce [`generate_causal`] exactly, pass the prompt boundary with
    /// [`Self::process_with_prompt_lengths`] or keep a [`TokenHistory`] for
    /// [`Self::process_with_history`].
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

    /// Processes one step exactly as [`generate_causal`] does, from host token rows.
    ///
    /// `histories[b]` holds row `b`'s prompt followed by the tokens generated
    /// so far, and `prompt_lengths[b]` is where its prompt ends. Prompt tokens
    /// count for the repetition penalty and n-grams; under the default
    /// [`PenaltyScope::Generated`] only the generated part is counted by
    /// presence and frequency, and only it counts toward `min_new_tokens`.
    /// F16/BF16 logits are promoted to F32 first (see
    /// [`upcast_half_precision`]), so the result is F32 for half-precision
    /// input.
    ///
    /// This is the stateless entry point for decoders that keep token rows
    /// on the host and may discard a suffix, such as speculative decoders
    /// rolling back rejected drafts. Each call uploads the rows once and runs
    /// at most one scatter per tracked count view; nothing is read back. A
    /// loop that only appends tokens should keep a [`TokenHistory`] instead.
    pub fn process_with_prompt_lengths(
        &self,
        logits: Tensor<B, 2>,
        histories: &[&[u32]],
        prompt_lengths: &[usize],
    ) -> Result<Tensor<B, 2>> {
        self.validate_logits_shape(&logits)?;
        let history = TokenHistory::from_token_rows(
            &self.config,
            self.vocabulary_size,
            histories,
            prompt_lengths,
            &logits.device(),
        )?;
        self.process_with_history(upcast_half_precision(logits), &history)
    }

    /// Processes logits using incrementally maintained per-row history.
    ///
    /// This is the efficient path used by the coordinators: penalties read
    /// `[batch, vocabulary]` count tensors already on the device, and no host
    /// synchronization occurs. `history` must have been created from a
    /// configuration with the same history-dependent controls.
    ///
    /// Logits must be finite or `-inf`; the result is then finite or `-inf`
    /// as well (see the module's finite-arithmetic section).
    ///
    /// # Errors
    ///
    /// Fails for a wrong logits shape, an incompatible history, Flex32 logits
    /// with an active sort, or F16/BF16 logits whose dtype cannot represent a
    /// configured float control.
    pub fn process_with_history(
        &self,
        logits: Tensor<B, 2>,
        history: &TokenHistory<B>,
    ) -> Result<Tensor<B, 2>> {
        let batch = self.validate_logits_shape(&logits)?;
        history.check_compatible(&self.config, self.vocabulary_size, batch)?;
        let dtype = logits.dtype();
        if dtype == DType::Flex32 && self.config.sorts_vocabulary(self.vocabulary_size) {
            return Err(Error::BackendCapability(
                "Burn 0.21 does not support Flex32 top-k/top-p sorting; use F32 logits or disable those filters explicitly".into(),
            ));
        }
        let precision = ScalarPrecision::of(dtype);
        // F32 and F64 were validated at construction; half precision is narrower.
        if precision.narrower_than_f32() {
            self.config
                .validate_scalars(precision)
                .map_err(|error| match error {
                    Error::Generation(reason) => Error::Generation(format!(
                        "{reason}; these {dtype:?} logits cannot represent it, so promote them with upcast_half_precision"
                    )),
                    other => other,
                })?;
        }
        let logits = self.apply_logit_bias(logits);
        let logits = self.apply_token_penalties(logits, history, precision)?;
        let sampling = self.config.strategy == SamplingStrategy::Sample;
        let logits = if sampling {
            logits.div_scalar(self.config.temperature)
        } else {
            logits
        };
        // Bias and penalties can overflow upward, and so can `z / T` for
        // `T < 1`; every other step keeps values bounded.
        let logits = if self.config.adjusts_logits() || (sampling && self.config.temperature < 1.0)
        {
            logits.clamp_max(precision.largest())
        } else {
            logits
        };
        let logits = self.apply_hard_constraints(logits, history)?;
        Ok(if sampling {
            self.apply_truncation_warpers(logits)
        } else {
            logits
        })
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
    /// Hugging Face `compute_transition_scores` for `do_sample=False` with
    /// `normalize_logits=True` (its greedy `scores` are processed logits, not
    /// log probabilities).
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

    /// Adds the configured logit bias with one sparse scatter.
    ///
    /// Every row receives the same `(token, bias)` pairs, so the `[batch,
    /// entries]` index and value tensors need no padding; only they cross to
    /// the device. Biases are finite in the logits dtype (validated), so the
    /// sum is finite or `+-inf` from overflow, never NaN. Validation keeps
    /// every biased token inside the vocabulary, so `entries <= vocabulary`
    /// and `batch * entries` never exceeds the logits' own element count; the
    /// Verus-verified [`repeat_row`] then copies the pairs into every row.
    fn apply_logit_bias(&self, logits: Tensor<B, 2>) -> Tensor<B, 2> {
        let entries = self.config.logit_bias.len();
        if entries == 0 {
            return logits;
        }
        let batch = logits.dims()[0];
        let row_indices = self
            .config
            .logit_bias
            .keys()
            .map(|&token| i64::from(token))
            .collect::<Vec<_>>();
        let row_biases = self.config.logit_bias.values().copied().collect::<Vec<_>>();
        let indices = repeat_row(&row_indices, batch);
        let biases = repeat_row(&row_biases, batch);
        let device = logits.device();
        let dtype = logits.dtype();
        let indices =
            Tensor::<B, 2, Int>::from_data(TensorData::new(indices, [batch, entries]), &device);
        let biases = float_tensor_in_dtype::<B, 2>(biases, [batch, entries], dtype, &device);
        logits.scatter(1, indices, biases, IndexingUpdateOp::Add)
    }

    /// Adds `-inf` to every hard-constrained token with one sparse scatter.
    ///
    /// Each row receives `(token, -inf)` pairs for EOS IDs while the row is
    /// below `min_new_tokens` and for tokens banned by the n-gram constraint.
    /// Rows are padded to a common width with `(0, 0.0)`, which adds nothing.
    /// The caller has already saturated positive overflow, so no `+inf`
    /// remains to meet the `-inf` and every constrained token becomes exactly
    /// `-inf` (adding is idempotent, so a token banned twice is fine). Only the
    /// small `[batch, width]` index and value tensors cross to the device. The
    /// Verus-verified [`push_constraint_row`] lays out each row's indices.
    fn apply_hard_constraints(
        &self,
        logits: Tensor<B, 2>,
        history: &TokenHistory<B>,
    ) -> Result<Tensor<B, 2>> {
        let batch = logits.dims()[0];
        let mut width = 0;
        for row in 0..batch {
            let mut count = history.banned_tokens(row).len();
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
            .ok_or_else(|| Error::Generation("hard-constraint scatter size overflows".into()))?;
        let mut indices = metadata::vec_with_capacity::<i64>(elements, "constraint indices")?;
        let mut deltas = metadata::vec_with_capacity::<f64>(elements, "constraint values")?;
        for row in 0..batch {
            let row_end = deltas.len() + width;
            // `width` is the maximum of these per-row counts, so every row fits.
            let constrained = push_constraint_row(
                &mut indices,
                &self.config.eos_token_ids,
                self.suppresses_eos(history, row),
                history.banned_tokens(row),
                width,
            );
            // `-inf` for every real entry, then a no-op `0.0` for the padding.
            deltas.resize(deltas.len() + constrained, f64::NEG_INFINITY);
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
    ///
    /// The product `f * c_i` saturates at `+-largest` of `precision` when the
    /// history's count bound says it could overflow (or the counts themselves
    /// exceed the dtype's range), so subtracting it never forms
    /// `inf - inf`. Below that bound the clamp is skipped; it would be a
    /// no-op anyway. Counts that exceed the dtype's range (only F16's 65504
    /// is reachable) saturate as integers *before* the float cast: an `inf`
    /// count times a penalty that rounds to zero in the dtype would be NaN.
    fn apply_token_penalties(
        &self,
        logits: Tensor<B, 2>,
        history: &TokenHistory<B>,
        precision: ScalarPrecision,
    ) -> Result<Tensor<B, 2>> {
        let dtype = logits.dtype();
        let mut logits = logits;
        if self.config.repetition_penalty != 1.0 {
            let seen = history.counts(CountView::All)?.clone().greater_elem(0);
            let penalized = logits
                .clone()
                .div_scalar(self.config.repetition_penalty)
                .mask_where(
                    logits.clone().lower_elem(0.0),
                    logits.clone().mul_scalar(self.config.repetition_penalty),
                );
            logits = logits.mask_where(seen, penalized);
        }
        let Some(view) = self.config.additive_count_view() else {
            return Ok(logits);
        };
        let additive = history.counts(view)?;
        if self.config.frequency_penalty != 0.0 {
            let largest = precision.largest();
            let bound = history.count_bound() as f64;
            let counts = if bound > largest {
                // Every integer element type holds `i32`; a dtype whose range
                // exceeds it cannot be exceeded by a count anyway.
                additive
                    .clone()
                    .clamp_max(largest.min(f64::from(i32::MAX)) as i32)
            } else {
                additive.clone()
            };
            let frequency = counts
                .float()
                .cast(dtype)
                .mul_scalar(self.config.frequency_penalty);
            // Half of `largest` leaves room for rounding `c`, `f`, and their product.
            let may_overflow =
                bound > largest || self.config.frequency_penalty.abs() * bound > largest / 2.0;
            let frequency = if may_overflow {
                frequency.clamp(-largest, largest)
            } else {
                frequency
            };
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
        Ok(logits)
    }

    /// Applies top-k, top-p, and min-p to temperature-scaled logits when sampling.
    fn apply_truncation_warpers(&self, logits: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut logits = logits;
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

/// The floating-point format a backend rounds scalar controls into.
///
/// Burn passes scalars as `f64`, and each backend rounds them to the logits
/// dtype before the arithmetic (Flex: `value as f32`, then `f16::from_f32` or
/// `bf16::from_f32` for half precision). Validation rounds the same way, so a
/// control accepted for a precision is exactly the value the kernel uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarPrecision {
    /// IEEE binary16: normal magnitudes `6.1035156e-5..=65504`.
    F16,
    /// bfloat16: F32's exponent range with an 8-bit significand.
    Bf16,
    /// IEEE binary32, also used for the Flex32 compute tag.
    F32,
    /// IEEE binary64.
    F64,
}

impl ScalarPrecision {
    /// The precision scalars are rounded into for logits of `dtype`.
    ///
    /// Logits are float tensors; any other tag (only Flex32 in practice)
    /// computes in F32.
    fn of(dtype: DType) -> Self {
        match dtype {
            DType::F16 => Self::F16,
            DType::BF16 => Self::Bf16,
            DType::F64 => Self::F64,
            _ => Self::F32,
        }
    }

    /// Dtype name used in error messages.
    fn name(self) -> &'static str {
        match self {
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
        }
    }

    /// Whether this precision can lose a control F32 represents (F16, BF16).
    fn narrower_than_f32(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16)
    }

    /// Rounds `value` exactly as a backend does before applying it to this precision.
    fn round(self, value: f64) -> f64 {
        match self {
            Self::F16 => f16::from_f32(value as f32).to_f64(),
            Self::Bf16 => bf16::from_f32(value as f32).to_f64(),
            Self::F32 => f64::from(value as f32),
            Self::F64 => value,
        }
    }

    /// Largest finite value; positive overflow saturates here.
    fn largest(self) -> f64 {
        match self {
            Self::F16 => f16::MAX.to_f64(),
            Self::Bf16 => bf16::MAX.to_f64(),
            Self::F32 => f64::from(f32::MAX),
            Self::F64 => f64::MAX,
        }
    }

    /// Smallest positive normal value.
    fn smallest_normal(self) -> f64 {
        match self {
            Self::F16 => f16::MIN_POSITIVE.to_f64(),
            Self::Bf16 => bf16::MIN_POSITIVE.to_f64(),
            Self::F32 => f64::from(f32::MIN_POSITIVE),
            Self::F64 => f64::MIN_POSITIVE,
        }
    }

    /// Formats a range bound at this precision's own digits, such as `3.4028235e38`.
    fn bound(self, value: f64) -> String {
        match self {
            Self::F64 => format!("{value:e}"),
            _ => format!("{:e}", value as f32),
        }
    }

    /// Requires a divisor (temperature, repetition penalty) that rounds to a normal positive value.
    ///
    /// Zero and infinity make `z / x` NaN for some `z`; a subnormal divisor
    /// overflows for ordinary logits and may be flushed to zero by GPU kernels.
    /// A value that is not even finite and positive keeps the plain message.
    fn check_divisor(self, field: &str, value: f64) -> Result<()> {
        if !(value.is_finite() && value > 0.0) {
            return Err(Error::Generation(format!(
                "{field} must be finite and greater than zero"
            )));
        }
        let rounded = self.round(value);
        if rounded.is_finite() && rounded >= self.smallest_normal() {
            Ok(())
        } else {
            Err(Error::Generation(format!(
                "{field} {value:?} rounds to {rounded:?} in {}; it must round to a normal positive {} value in [{}, {}]",
                self.name(),
                self.name(),
                self.bound(self.smallest_normal()),
                self.bound(self.largest()),
            )))
        }
    }

    /// Requires an additive control (penalty or bias) that rounds to a finite value.
    fn check_finite(self, field: &str, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(Error::Generation(format!(
                "{field} must be finite, found {value:?}"
            )));
        }
        if self.round(value).is_finite() {
            Ok(())
        } else {
            Err(Error::Generation(format!(
                "{field} {value:?} overflows {}; it must round to a finite {} value in [-{}, {}]",
                self.name(),
                self.name(),
                self.bound(self.largest()),
                self.bound(self.largest()),
            )))
        }
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

/// Promotes half-precision logits to F32 and leaves F32 and F64 unchanged.
///
/// Hugging Face upcasts next-token logits to F32 before any processor runs,
/// and [`generate_causal`] does the same: bias, penalties, and warpers are
/// then free of F16/BF16 rounding (for example a 1.05 repetition penalty on a
/// logit near 20, where F16 spacing is 0.016). The Flex32 compute tag also
/// maps to F32 storage. Casting one `[batch, vocabulary]` step is negligible
/// next to a model forward pass.
pub fn upcast_half_precision<B: Backend, const D: usize>(logits: Tensor<B, D>) -> Tensor<B, D> {
    let dtype = probability_accumulation_dtype(logits.dtype());
    logits.cast(dtype)
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

/// Keeps every score at least the `k`-th greatest of its row, ties included.
///
/// With row scores sorted descending `z_(1) >= ... >= z_(V)`, a token survives
/// when `z_i >= z_(k)`; every other score becomes `-inf`. This is Hugging
/// Face's `TopKLogitsWarper` rule, so tokens tied with the boundary score are
/// never dropped by sort order. One value sort and one comparison per row: no
/// index sort or scatter is needed, and the comparison runs in the logits
/// dtype, so the boundary token itself always survives. Requires
/// `1 <= top_k <= vocabulary`.
fn filter_top_k<B: Backend>(logits: Tensor<B, 2>, top_k: usize) -> Tensor<B, 2> {
    let [batch, vocabulary] = logits.dims();
    let boundary = logits
        .clone()
        .sort_descending(1)
        .narrow(1, top_k - 1, 1)
        .expand([batch, vocabulary]);
    let below = logits.clone().lower(boundary);
    logits.mask_fill(below, f64::NEG_INFINITY)
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
