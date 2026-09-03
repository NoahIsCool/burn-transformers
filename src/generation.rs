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

use core::marker::PhantomData;

use burn::tensor::{
    Bool, IndexingUpdateOp, Int, Shape, Tensor, activation::softmax, backend::Backend,
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

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
    /// Backend random seed used to make sampling reproducible.
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
#[derive(Debug, Clone)]
pub struct LogitsProcessor<B: Backend> {
    config: GenerationConfig,
    vocabulary_size: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> LogitsProcessor<B> {
    /// Creates a validated processor and seeds the backend random generator.
    pub fn new(
        config: GenerationConfig,
        vocabulary_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate(vocabulary_size)?;
        B::seed(device, config.seed);
        Ok(Self {
            config,
            vocabulary_size,
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
        let [batch, vocabulary_size] = logits.dims();
        if vocabulary_size != self.vocabulary_size {
            return Err(Error::shape(
                "generation.logits",
                format!("[batch, {}]", self.vocabulary_size),
                format!("[{batch}, {vocabulary_size}]"),
            ));
        }

        let mut logits = if let Some(history) = history {
            let [history_batch, _] = history.dims();
            if history_batch != batch {
                return Err(Error::shape(
                    "generation.history",
                    format!("[{batch}, sequence]"),
                    format!("[{history_batch}, sequence]"),
                ));
            }
            self.apply_token_penalties(logits, history)
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
    pub fn select(&self, logits: Tensor<B, 2>) -> Tensor<B, 1, Int> {
        match self.config.strategy {
            SamplingStrategy::Greedy => logits.argmax(1).squeeze_dim::<1>(1),
            SamplingStrategy::Sample => softmax(logits, 1).categorical(1).squeeze_dim::<1>(1),
        }
    }

    /// Processes logits and immediately selects the next token for each row.
    pub fn process_and_select(
        &self,
        logits: Tensor<B, 2>,
        history: Option<Tensor<B, 2, Int>>,
    ) -> Result<Tensor<B, 1, Int>> {
        self.process(logits, history)
            .map(|logits| self.select(logits))
    }

    /// Applies sign-aware repetition and additive presence penalties.
    fn apply_token_penalties(
        &self,
        logits: Tensor<B, 2>,
        history: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let observed = history
            .one_hot::<3>(self.vocabulary_size)
            .sum_dim(1)
            .squeeze_dim::<2>(1)
            .greater_elem(0);

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
        Ok(Self {
            config,
            generated: vec![Vec::new(); batch_size],
            finished: vec![None; batch_size],
        })
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
    };

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
        tracker.push(0, 7).expect("completed push is accepted");
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
}
