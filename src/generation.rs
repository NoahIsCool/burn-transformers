//! Backend-generic token sampling and host-side stopping state.
//!
//! Generation is split into two deliberately small pieces. [`LogitsProcessor`]
//! transforms model scores entirely on the selected Burn backend, avoiding a
//! device-to-host copy of the full vocabulary. [`GenerationTracker`] retains
//! only emitted token IDs on the host, where suffix matching and finish-reason
//! bookkeeping are inexpensive and deterministic.
//!
//! Temperature sampling draws from
//! `p(i) = exp(z_i / T) / sum_j exp(z_j / T)`. Top-k truncation keeps exactly
//! the `k` greatest scores. Nucleus (top-p) truncation keeps the shortest sorted
//! prefix whose probability mass reaches `p`; equivalently, a sorted item is
//! retained when the cumulative mass *before* it is less than `p`. All filtering
//! happens before the final softmax, with rejected scores set to negative
//! infinity.

mod causal;

pub use causal::{
    CausalGenerationOutput, CausalGenerationRequest, GeneratedSequence, GenerationToken,
    generate_causal, generate_causal_streaming,
};

use core::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use burn::tensor::{
    Bool, ElementConversion, IndexingUpdateOp, Int, Shape, Tensor, TensorData,
    activation::{log_softmax, softmax},
    backend::Backend,
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result, TokenMask};

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

/// Serializable controls shared by ordinary and speculative generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerationConfig {
    /// Maximum number of new tokens generated for each sequence.
    pub max_new_tokens: usize,
    /// Greedy or stochastic token selection.
    pub strategy: SamplingStrategy,
    /// Positive softmax temperature used for stochastic sampling.
    pub temperature: f64,
    /// Optional exact count of greatest logits retained before sampling.
    pub top_k: Option<usize>,
    /// Optional nucleus probability in the interval `(0, 1]`.
    pub top_p: Option<f64>,
    /// Multiplicative repetition penalty; one disables it.
    pub repetition_penalty: f64,
    /// Additive penalty applied once to every previously observed token.
    pub presence_penalty: f64,
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
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
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
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(Error::Generation(
                "repetition_penalty must be finite and greater than zero".into(),
            ));
        }
        if !self.presence_penalty.is_finite() {
            return Err(Error::Generation("presence_penalty must be finite".into()));
        }

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
        Ok(())
    }
}

/// Applies penalties and probability truncation without leaving the backend.
#[derive(Debug)]
pub struct LogitsProcessor<B: Backend> {
    config: GenerationConfig,
    vocabulary_size: usize,
    sample_counter: AtomicU64,
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
    /// Log probability of each selected ID after penalties and truncation.
    pub log_probabilities: Tensor<B, 1>,
}

impl<B: Backend> LogitsProcessor<B> {
    /// Creates a validated processor with an independent random stream.
    ///
    /// Burn's backend seed is process-global on several backends, so using it
    /// here would let an unrelated model or concurrently running test perturb
    /// this processor. Instead, each categorical draw consumes a counter from
    /// a SplitMix64 stream and transfers only one uniform value per batch row
    /// to the backend for inverse-CDF selection.
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

    /// Processes `[batch, vocabulary]` logits using optional token history.
    ///
    /// History has shape `[batch, prior_sequence]`. Repetition and presence
    /// penalties count prompt tokens as well as already generated tokens when
    /// the caller supplies the complete history.
    pub fn process(
        &self,
        logits: Tensor<B, 2>,
        history: Option<Tensor<B, 2, Int>>,
    ) -> Result<Tensor<B, 2>> {
        self.process_with_mask(logits, history, None)
    }

    /// Processes logits while excluding padded history tokens from penalties.
    ///
    /// `history_mask` follows the crate-wide `true == valid` convention. Masked
    /// token slots may contain arbitrary sentinel values: they are replaced
    /// before one-hot indexing and cannot affect repetition or presence counts.
    pub fn process_with_mask(
        &self,
        logits: Tensor<B, 2>,
        history: Option<Tensor<B, 2, Int>>,
        history_mask: Option<TokenMask<B>>,
    ) -> Result<Tensor<B, 2>> {
        let batch = self.validate_logits_shape(&logits)?;

        let mut logits = if let Some(history) = history {
            let [history_batch, history_length] = history.dims();
            if history_batch != batch {
                return Err(Error::shape(
                    "generation.history",
                    format!("[{batch}, sequence]"),
                    format!("[{history_batch}, sequence]"),
                ));
            }
            let history_mask = history_mask
                .map(TokenMask::into_tensor)
                .map(|mask| {
                    let [mask_batch, mask_length] = mask.dims();
                    if [mask_batch, mask_length] != [history_batch, history_length] {
                        return Err(Error::shape(
                            "generation.history_mask",
                            format!("[{history_batch}, {history_length}]"),
                            format!("[{mask_batch}, {mask_length}]"),
                        ));
                    }
                    Ok(mask)
                })
                .transpose()?;
            if history_length == 0
                || (self.config.repetition_penalty == 1.0 && self.config.presence_penalty == 0.0)
            {
                logits
            } else {
                let history = match &history_mask {
                    Some(mask) => history.mask_fill(mask.clone().bool_not(), 0),
                    None => history,
                };
                self.validate_history_values(&history)?;
                self.apply_token_penalties(logits, history, history_mask)
            }
        } else if history_mask.is_some() {
            return Err(Error::Generation(
                "history_mask requires a corresponding history tensor".into(),
            ));
        } else {
            logits
        };

        if self.config.strategy == SamplingStrategy::Sample {
            logits = logits / self.config.temperature;
        }
        if let Some(top_k) = self.config.top_k {
            logits = filter_top_k(logits, top_k);
        }
        if let Some(top_p) = self.config.top_p
            && top_p < 1.0
        {
            logits = filter_top_p(logits, top_p);
        }
        Ok(logits)
    }

    /// Selects one token per row from already processed logits.
    pub fn select(&self, logits: Tensor<B, 2>) -> Result<Tensor<B, 1, Int>> {
        let batch = self.validate_logits_shape(&logits)?;
        match self.config.strategy {
            SamplingStrategy::Greedy => Ok(logits.argmax(1).squeeze_dim::<1>(1)),
            SamplingStrategy::Sample => self.sample_inverse_cdf(logits, batch),
        }
    }

    /// Samples processed logits with per-instance uniforms and a backend CDF.
    fn sample_inverse_cdf(&self, logits: Tensor<B, 2>, batch: usize) -> Result<Tensor<B, 1, Int>> {
        let maximum_index = i64::try_from(self.vocabulary_size - 1).map_err(|_| {
            Error::Generation(format!(
                "vocabulary size {} cannot be represented by the backend index API",
                self.vocabulary_size
            ))
        })?;
        let uniforms = self.next_uniforms(batch)?;
        let device = logits.device();
        let thresholds = Tensor::<B, 2>::from_data(TensorData::new(uniforms, [batch, 1]), &device)
            .expand([batch, self.vocabulary_size]);
        let cumulative = softmax(logits, 1).cumsum(1);

        // The number of CDF entries strictly below u is the first category
        // whose inclusive cumulative mass reaches u. Clamp only protects
        // against the final sum rounding microscopically below one.
        Ok(cumulative
            .lower(thresholds)
            .int()
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .clamp_max(maximum_index))
    }

    /// Reserves and evaluates one deterministic random value per batch row.
    fn next_uniforms(&self, batch: usize) -> Result<Vec<f32>> {
        let count = u64::try_from(batch).map_err(|_| {
            Error::Generation(format!(
                "sampling batch size {batch} does not fit the u64 random stream"
            ))
        })?;
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
        let mut uniforms = Vec::new();
        uniforms.try_reserve_exact(batch).map_err(|_| {
            Error::Generation(format!(
                "failed to allocate {batch} categorical sampling thresholds"
            ))
        })?;
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
    /// Log probabilities use the renormalized distribution after temperature,
    /// top-k/top-p filtering, repetition, and presence penalties. This is the
    /// probability actually sampled, rather than the unprocessed model score.
    pub fn process_and_select_with_log_probs(
        &self,
        logits: Tensor<B, 2>,
        history: Option<Tensor<B, 2, Int>>,
    ) -> Result<TokenSelection<B>> {
        let logits = self.process(logits, history)?;
        self.select_with_log_probs(logits)
    }

    /// Selects from processed logits and returns the sampled log probability.
    ///
    /// Callers using [`Self::process_with_mask`] can use this method to retain
    /// the same probability semantics as [`Self::process_and_select_with_log_probs`].
    pub fn select_with_log_probs(&self, logits: Tensor<B, 2>) -> Result<TokenSelection<B>> {
        let token_ids = self.select(logits.clone())?;
        let batch = token_ids.dims()[0];
        let selected = log_softmax(logits, 1)
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

    /// Prevents invalid history IDs from reaching backend one-hot indexing.
    fn validate_history_values(&self, history: &Tensor<B, 2, Int>) -> Result<()> {
        let minimum = history
            .clone()
            .min()
            .try_into_scalar()
            .map_err(|error| {
                Error::BackendCapability(format!(
                    "failed to validate generation history minimum: {error}"
                ))
            })?
            .elem::<i64>();
        let maximum = history
            .clone()
            .max()
            .try_into_scalar()
            .map_err(|error| {
                Error::BackendCapability(format!(
                    "failed to validate generation history maximum: {error}"
                ))
            })?
            .elem::<i64>();
        if minimum < 0 || maximum as usize >= self.vocabulary_size {
            return Err(Error::shape(
                "generation.history",
                format!("token values in 0..{}", self.vocabulary_size),
                format!("observed range {minimum}..={maximum}"),
            ));
        }
        Ok(())
    }

    /// Applies sign-aware repetition and additive presence penalties.
    fn apply_token_penalties(
        &self,
        logits: Tensor<B, 2>,
        history: Tensor<B, 2, Int>,
        history_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 2> {
        let mut occurrences = history.one_hot::<3>(self.vocabulary_size);
        if let Some(mask) = history_mask {
            occurrences = occurrences * mask.unsqueeze_dim::<3>(2).int();
        }
        let observed = occurrences.sum_dim(1).squeeze_dim::<2>(1).greater_elem(0);

        let mut logits = logits;
        if self.config.repetition_penalty != 1.0 {
            let penalized = logits
                .clone()
                .div_scalar(self.config.repetition_penalty)
                .mask_where(
                    logits.clone().lower_elem(0.0),
                    logits.clone().mul_scalar(self.config.repetition_penalty),
                );
            logits = logits.mask_where(observed.clone(), penalized);
        }
        if self.config.presence_penalty != 0.0 {
            let penalty = observed
                .int()
                .float()
                .mul_scalar(self.config.presence_penalty);
            logits = logits - penalty;
        }
        logits
    }
}

/// Maps one SplitMix64 output to a strictly interior 23-bit uniform value.
///
/// SplitMix64 advances in the ring of integers modulo `2^64`; multiplying the
/// stream index by its odd Weyl increment therefore visits every state exactly
/// once. The upper 23 result bits are centered within equal-width intervals of
/// `(0, 1)`, avoiding the CDF boundary values zero and one in F32.
fn splitmix64_uniform(seed: u64, index: u64) -> f32 {
    let mut value =
        seed.wrapping_add(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(index.wrapping_add(1)));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let bucket = value >> 41;
    ((bucket as f64 + 0.5) / ((1_u64 << 23) as f64)) as f32
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
    let probabilities = softmax(logits.clone(), 1);
    let (sorted, indices) = probabilities.sort_descending_with_indices(1);
    let mass_before = sorted.clone().cumsum(1) - sorted;
    let keep_sorted = mass_before.lower_elem(top_p);
    let keep = scatter_mask(indices, keep_sorted, batch, vocabulary);
    logits.mask_fill(keep.bool_not(), f64::NEG_INFINITY)
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
    config: GenerationConfig,
    vocabulary_size: usize,
    generated: Vec<Vec<u32>>,
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
        let mut generated = Vec::new();
        generated.try_reserve_exact(batch_size).map_err(|_| {
            Error::Generation(format!(
                "failed to allocate generation state for batch size {batch_size}"
            ))
        })?;
        generated.resize_with(batch_size, Vec::new);
        let mut finished = Vec::new();
        finished.try_reserve_exact(batch_size).map_err(|_| {
            Error::Generation(format!(
                "failed to allocate finish state for batch size {batch_size}"
            ))
        })?;
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
    /// Calling this after the sequence has finished is idempotent: the token is
    /// ignored and the original finish reason is returned.
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
        let reason = if self.config.eos_token_ids.contains(&token) {
            Some(FinishReason::EndOfSequence { token_id: token })
        } else {
            self.matching_stop(sequence)
                .map(|(index, length)| FinishReason::StopSequence { index, length })
                .or_else(|| {
                    (self.generated[sequence].len() >= self.config.max_new_tokens)
                        .then_some(FinishReason::Length)
                })
        };
        self.finished[sequence] = reason;
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

    /// Finds the longest matching stop suffix, breaking equal lengths by order.
    fn matching_stop(&self, sequence: usize) -> Option<(usize, usize)> {
        self.config
            .stop_sequences
            .iter()
            .enumerate()
            .filter(|(_, stop)| self.generated[sequence].ends_with(stop))
            .map(|(index, stop)| (index, stop.len()))
            .max_by_key(|(index, length)| (*length, usize::MAX - *index))
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Flex;
    use burn::tensor::{Int, Tensor, TensorData, Tolerance};

    use super::{
        FinishReason, GenerationConfig, GenerationTracker, LogitsProcessor, SamplingStrategy,
        splitmix64_uniform,
    };
    use crate::TokenMask;

    type TestBackend = Flex<f32, i32>;

    /// Invalid sampling controls fail before any tensor operation can panic.
    #[test]
    fn configuration_validation_rejects_invalid_ranges() {
        let invalid = [
            GenerationConfig {
                max_new_tokens: 0,
                ..Default::default()
            },
            GenerationConfig {
                temperature: 0.0,
                ..Default::default()
            },
            GenerationConfig {
                top_k: Some(5),
                ..Default::default()
            },
            GenerationConfig {
                top_p: Some(0.0),
                ..Default::default()
            },
            GenerationConfig {
                repetition_penalty: -1.0,
                ..Default::default()
            },
            GenerationConfig {
                stop_sequences: vec![Vec::new()],
                ..Default::default()
            },
            GenerationConfig {
                top_p: Some(f64::NAN),
                ..Default::default()
            },
            GenerationConfig {
                repetition_penalty: f64::INFINITY,
                ..Default::default()
            },
            GenerationConfig {
                presence_penalty: f64::NAN,
                ..Default::default()
            },
            GenerationConfig {
                eos_token_ids: vec![4],
                ..Default::default()
            },
            GenerationConfig {
                pad_token_id: Some(4),
                ..Default::default()
            },
            GenerationConfig {
                stop_sequences: vec![vec![4]],
                ..Default::default()
            },
        ];
        for config in invalid {
            assert!(config.validate(4).is_err());
        }
        assert!(GenerationConfig::default().validate(0).is_err());
    }

    /// Greedy selection preserves row boundaries and returns each maximum.
    #[test]
    fn greedy_selects_each_rows_maximum() {
        let device = Default::default();
        let processor =
            LogitsProcessor::<TestBackend>::new(GenerationConfig::default(), 4, &device)
                .expect("valid processor");
        let logits = Tensor::from_floats([[0.0, 4.0, 1.0, -1.0], [3.0, 2.0, 7.0, 1.0]], &device);
        processor
            .process_and_select(logits, None)
            .expect("valid scores")
            .into_data()
            .assert_eq(&TensorData::from([1_i32, 2]), false);
    }

    /// Selected-token log probabilities describe the processed distribution.
    #[test]
    fn selection_reports_processed_log_probability() {
        let device = Default::default();
        let processor =
            LogitsProcessor::<TestBackend>::new(GenerationConfig::default(), 2, &device)
                .expect("valid processor");
        let selection = processor
            .process_and_select_with_log_probs(Tensor::from_floats([[0.0, 1.0]], &device), None)
            .expect("valid scores");
        selection
            .token_ids
            .into_data()
            .assert_eq(&TensorData::from([1_i32]), false);
        selection
            .log_probabilities
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([-0.313_261_7_f32]),
                Tolerance::absolute(1e-6),
            );
    }

    /// Every public processor boundary rejects empty or mismatched score tensors.
    #[test]
    fn processor_rejects_invalid_shapes_before_backend_sampling() {
        let device = Default::default();
        let processor =
            LogitsProcessor::<TestBackend>::new(GenerationConfig::default(), 4, &device)
                .expect("valid processor");
        assert!(
            processor
                .process(Tensor::zeros([0, 4], &device), None)
                .is_err()
        );
        assert!(processor.select(Tensor::zeros([1, 3], &device)).is_err());
        assert!(
            processor
                .process(
                    Tensor::zeros([2, 4], &device),
                    Some(Tensor::<TestBackend, 2, Int>::zeros([1, 1], &device)),
                )
                .is_err()
        );
    }

    /// Penalty history validates token ranges before constructing one-hot tensors.
    #[test]
    fn penalty_history_rejects_invalid_ids_and_accepts_empty_history() {
        let device = Default::default();
        let config = GenerationConfig {
            presence_penalty: 0.5,
            ..Default::default()
        };
        let processor =
            LogitsProcessor::<TestBackend>::new(config, 4, &device).expect("valid processor");
        let logits = Tensor::from_floats([[3.0, 2.0, 1.0, 0.0]], &device);
        for history in [
            Tensor::<TestBackend, 2, Int>::from_ints([[-1]], &device),
            Tensor::<TestBackend, 2, Int>::from_ints([[4]], &device),
        ] {
            assert!(processor.process(logits.clone(), Some(history)).is_err());
        }
        processor
            .process(
                logits.clone(),
                Some(Tensor::<TestBackend, 2, Int>::zeros([1, 0], &device)),
            )
            .expect("an empty history applies no penalties")
            .into_data()
            .assert_eq(&logits.into_data(), false);
    }

    /// Padded sentinels neither reach one-hot indexing nor accrue token penalties.
    #[test]
    fn masked_history_ignores_sentinels_and_validates_mask_shape() {
        let device = Default::default();
        let config = GenerationConfig {
            presence_penalty: 2.0,
            ..Default::default()
        };
        let processor =
            LogitsProcessor::<TestBackend>::new(config, 4, &device).expect("valid processor");
        let logits = Tensor::from_floats([[0.0, 4.0, 3.0, 2.0]], &device);
        let history = Tensor::<TestBackend, 2, Int>::from_ints([[99, 1]], &device);
        let mask = TokenMask::new(Tensor::from_bool([[false, true]].into(), &device));
        let processed = processor
            .process_with_mask(logits, Some(history), Some(mask))
            .expect("masked sentinel is never indexed")
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(processed, [0.0, 2.0, 3.0, 2.0]);

        let history = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1]], &device);
        let wrong = TokenMask::new(Tensor::from_bool([[true]].into(), &device));
        assert!(
            processor
                .process_with_mask(Tensor::zeros([1, 4], &device), Some(history), Some(wrong))
                .is_err()
        );
        assert!(
            processor
                .process_with_mask(
                    Tensor::zeros([1, 4], &device),
                    None,
                    Some(TokenMask::new(Tensor::from_bool([[true]].into(), &device)))
                )
                .is_err()
        );
    }

    /// Independent processors reproduce the same multi-step stochastic stream.
    #[test]
    fn sampling_seed_is_deterministic() {
        let device = Default::default();
        let config = GenerationConfig {
            strategy: SamplingStrategy::Sample,
            seed: 91,
            ..Default::default()
        };
        let logits = Tensor::<TestBackend, 2>::zeros([64, 4], &device);
        let first = LogitsProcessor::<TestBackend>::new(config.clone(), 4, &device)
            .expect("valid processor");
        let second =
            LogitsProcessor::<TestBackend>::new(config, 4, &device).expect("valid processor");
        for _ in 0..4 {
            let first_ids = first
                .process_and_select(logits.clone(), None)
                .expect("valid scores")
                .into_data();
            let second_ids = second
                .process_and_select(logits.clone(), None)
                .expect("valid scores")
                .into_data();
            assert_eq!(first_ids, second_ids);
        }
    }

    /// Stream cloning snapshots state, uniforms stay interior, and exhaustion is safe.
    #[test]
    fn sampling_stream_boundaries_are_explicit() {
        for (seed, index) in [(0, 0), (u64::MAX, 0), (0, u64::MAX), (u64::MAX, u64::MAX)] {
            let uniform = splitmix64_uniform(seed, index);
            assert!(uniform > 0.0 && uniform < 1.0);
        }

        let device = Default::default();
        let config = GenerationConfig {
            strategy: SamplingStrategy::Sample,
            seed: 17,
            ..Default::default()
        };
        let processor =
            LogitsProcessor::<TestBackend>::new(config, 3, &device).expect("valid processor");
        processor
            .process_and_select(Tensor::zeros([1, 3], &device), None)
            .expect("first draw");
        let clone = processor.clone();
        assert_eq!(
            processor
                .process_and_select(Tensor::zeros([8, 3], &device), None)
                .expect("continued draw")
                .into_data(),
            clone
                .process_and_select(Tensor::zeros([8, 3], &device), None)
                .expect("cloned draw")
                .into_data()
        );

        processor
            .sample_counter
            .store(u64::MAX - 1, std::sync::atomic::Ordering::Relaxed);
        assert!(processor.select(Tensor::zeros([2, 3], &device)).is_err());
    }

    /// Top-k retains exactly k finite entries, including tied boundary values.
    #[test]
    fn top_k_is_exact_even_at_ties() {
        let device = Default::default();
        let config = GenerationConfig {
            strategy: SamplingStrategy::Sample,
            top_k: Some(2),
            ..Default::default()
        };
        let processor =
            LogitsProcessor::<TestBackend>::new(config, 4, &device).expect("valid processor");
        let output = processor
            .process(Tensor::from_floats([[3.0, 2.0, 2.0, 1.0]], &device), None)
            .expect("valid scores")
            .into_data()
            .to_vec::<f32>()
            .expect("float output");
        assert_eq!(output.iter().filter(|value| value.is_finite()).count(), 2);
        assert_eq!(output[0], 3.0);
    }

    /// Nucleus filtering includes the item that crosses the target probability.
    #[test]
    fn top_p_keeps_crossing_token() {
        let device = Default::default();
        let config = GenerationConfig {
            strategy: SamplingStrategy::Sample,
            top_p: Some(0.8),
            ..Default::default()
        };
        let processor =
            LogitsProcessor::<TestBackend>::new(config, 3, &device).expect("valid processor");
        let output = processor
            .process(Tensor::from_floats([[2.0, 1.0, 0.0]], &device), None)
            .expect("valid scores")
            .into_data()
            .to_vec::<f32>()
            .expect("float output");
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
        assert!(output[2].is_infinite());
    }

    /// Repetition is sign-aware and presence applies only to observed IDs.
    #[test]
    fn token_penalties_match_standard_logit_rules() {
        let device = Default::default();
        let config = GenerationConfig {
            repetition_penalty: 2.0,
            presence_penalty: 0.5,
            ..Default::default()
        };
        let processor =
            LogitsProcessor::<TestBackend>::new(config, 4, &device).expect("valid processor");
        let logits = Tensor::from_floats([[4.0, -3.0, 2.0, 1.0]], &device);
        let history = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 1]], &device);
        processor
            .process(logits, Some(history))
            .expect("valid scores")
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[1.5, -6.5, 2.0, 1.0]]),
                Tolerance::absolute(1e-5),
            );
    }

    /// Tracker gives EOS precedence and never mutates completed sequences.
    #[test]
    fn tracker_handles_eos_and_is_idempotent() {
        let config = GenerationConfig {
            eos_token_ids: vec![2],
            stop_sequences: vec![vec![1, 2]],
            ..Default::default()
        };
        let mut tracker = GenerationTracker::new(config, 8, 1).expect("valid tracker");
        assert_eq!(tracker.push(0, 1).expect("in range"), None);
        assert_eq!(
            tracker.push(0, 2).expect("in range"),
            Some(&FinishReason::EndOfSequence { token_id: 2 })
        );
        tracker
            .push(0, u32::MAX)
            .expect("completed push is ignored before token validation");
        assert_eq!(tracker.generated(0).expect("in range"), &[1, 2]);
        assert!(tracker.all_finished());
    }

    /// Longest suffix wins and length termination remains a fallback.
    #[test]
    fn tracker_resolves_stops_then_length() {
        let config = GenerationConfig {
            max_new_tokens: 3,
            stop_sequences: vec![vec![2], vec![1, 2]],
            ..Default::default()
        };
        let mut tracker = GenerationTracker::new(config, 8, 2).expect("valid tracker");
        tracker.push(0, 1).expect("in range");
        assert_eq!(
            tracker.push(0, 2).expect("in range"),
            Some(&FinishReason::StopSequence {
                index: 1,
                length: 2,
            })
        );
        tracker.push(1, 4).expect("in range");
        tracker.push(1, 5).expect("in range");
        assert_eq!(
            tracker.push(1, 6).expect("in range"),
            Some(&FinishReason::Length)
        );
        assert!(tracker.finish_reason(2).is_err());
        assert!(tracker.generated(2).is_err());
        assert!(tracker.push(2, 0).is_err());
    }

    /// Tracker bounds and allocation failures are recoverable and mutation-free.
    #[test]
    fn tracker_rejects_invalid_tokens_and_impossible_batch_capacity() {
        let mut tracker =
            GenerationTracker::new(GenerationConfig::default(), 4, 2).expect("valid tracker");
        assert_eq!(tracker.batch_size(), 2);
        assert_eq!(tracker.vocabulary_size(), 4);
        assert!(tracker.push(0, 4).is_err());
        assert!(tracker.generated(0).expect("in range").is_empty());
        assert_eq!(tracker.finish_reason(0).expect("in range"), None);
        assert!(
            GenerationTracker::new(GenerationConfig::default(), 4, usize::MAX).is_err(),
            "capacity overflow must not reach an infallible allocation"
        );
    }
}
