//! Incrementally maintained per-row token history for logits processing.
//!
//! History-dependent controls need different views of the tokens a row has
//! seen:
//!
//! - repetition, presence, and frequency penalties need per-token occurrence
//!   counts `c[b, v]`, which live on the device as two `[batch, vocabulary]`
//!   integer tensors (all observed tokens, and generated tokens only);
//! - `no_repeat_ngram_size` needs the token sequence itself, kept on the host
//!   with an index from each `(n - 1)`-gram to the tokens that followed it;
//! - `min_new_tokens` needs each row's generated length, also on the host.
//!
//! Rebuilding counts from a `[batch, history]` tensor every step costs a
//! `[batch, history, vocabulary]` one-hot (gigabytes for a 4k-token prompt and
//! a 250k vocabulary) plus host synchronizations. [`TokenHistory`] instead
//! scatter-adds the prompt once, `O(batch * prompt)`, and then one token per
//! row per step, `O(batch)`, uploading only the new IDs. A coordinator that
//! already reads selected IDs back to the host therefore adds no extra
//! synchronization.
//!
//! Only state that the configuration actually uses is kept: without penalties
//! there are no count tensors, and without an n-gram constraint no host index.

use std::collections::HashMap;

use burn::tensor::{Bool, IndexingUpdateOp, Int, Tensor, TensorData, backend::Backend};

use crate::{Error, Result, TokenMask, architecture::tensor_bool_values};

use super::{GenerationConfig, metadata};

/// Device-resident occurrence counts used by the token penalties.
#[derive(Debug, Clone)]
pub(super) struct TokenCounts<B: Backend> {
    /// `[batch, vocabulary]` counts of every observed token, prompt and generated.
    pub(super) all: Tensor<B, 2, Int>,
    /// `[batch, vocabulary]` counts of generated tokens only.
    pub(super) generated: Tensor<B, 2, Int>,
}

/// Host n-gram index of one row for the no-repeat constraint.
#[derive(Debug, Clone, Default)]
struct NgramIndex {
    /// Every valid token of the row so far, prompt first.
    tokens: Vec<u32>,
    /// Sorted, deduplicated tokens that followed each `(n - 1)`-token prefix.
    continuations: HashMap<Vec<u32>, Vec<u32>>,
}

impl NgramIndex {
    /// Appends `token` and records the n-gram of size `size` it completes.
    fn push(&mut self, token: u32, size: usize) {
        self.tokens.push(token);
        let length = self.tokens.len();
        if length >= size {
            let prefix = self.tokens[length - size..length - 1].to_vec();
            let followers = self.continuations.entry(prefix).or_default();
            if let Err(position) = followers.binary_search(&token) {
                followers.insert(position, token);
            }
        }
    }

    /// Tokens that would complete an n-gram of `size` already present.
    ///
    /// Matches Hugging Face `NoRepeatNGramLogitsProcessor`: nothing is banned
    /// until the next token could complete a full n-gram, and the lookup key
    /// is the last `n - 1` tokens.
    fn banned(&self, size: usize) -> &[u32] {
        let length = self.tokens.len();
        if length + 1 < size {
            return &[];
        }
        let prefix = &self.tokens[length + 1 - size..];
        self.continuations.get(prefix).map_or(&[], Vec::as_slice)
    }
}

/// Per-row token statistics maintained incrementally across decode steps.
///
/// Create one with [`TokenHistory::from_prompt`] (or [`TokenHistory::empty`])
/// using the same [`GenerationConfig`] as the [`super::LogitsProcessor`], pass
/// it to [`super::LogitsProcessor::process_with_history`] each step, and call
/// [`TokenHistory::record`] with the tokens each row emitted. Prompt tokens
/// count for the repetition penalty and n-grams but not as generated tokens.
#[derive(Debug, Clone)]
pub struct TokenHistory<B: Backend> {
    /// Exclusive upper bound of every recorded token ID.
    vocabulary_size: usize,
    /// Occurrence counts, present only when a count-based penalty is enabled.
    counts: Option<TokenCounts<B>>,
    /// Number of tokens recorded as generated for each row.
    generated_lengths: Vec<usize>,
    /// N-gram size of the no-repeat constraint; zero disables the index.
    ngram_size: usize,
    /// One n-gram index per row, empty when `ngram_size == 0`.
    ngrams: Vec<NgramIndex>,
    /// Device holding the count tensors.
    device: B::Device,
}

impl<B: Backend> TokenHistory<B> {
    /// Creates history for `batch` rows that have seen no tokens yet.
    pub fn empty(
        config: &GenerationConfig,
        vocabulary_size: usize,
        batch: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let tokens = Tensor::<B, 2, Int>::zeros([batch, 0], device);
        Self::build(config, vocabulary_size, tokens, None, None, false)
    }

    /// Creates history from a `[batch, prompt]` prompt and optional validity mask.
    ///
    /// Reads the prompt to the host once, and only when a history-dependent
    /// control is enabled, to validate token IDs and build n-grams.
    pub fn from_prompt(
        config: &GenerationConfig,
        vocabulary_size: usize,
        prompt: Tensor<B, 2, Int>,
        mask: Option<TokenMask<B>>,
    ) -> Result<Self> {
        Self::build(
            config,
            vocabulary_size,
            prompt,
            mask.map(TokenMask::into_tensor),
            None,
            false,
        )
    }

    /// Creates prompt history when the caller already holds the host mask values.
    pub(super) fn from_prompt_values(
        config: &GenerationConfig,
        vocabulary_size: usize,
        prompt: Tensor<B, 2, Int>,
        mask: Option<TokenMask<B>>,
        valid: &[bool],
    ) -> Result<Self> {
        Self::build(
            config,
            vocabulary_size,
            prompt,
            mask.map(TokenMask::into_tensor),
            Some(valid),
            false,
        )
    }

    /// Creates history in which every valid token counts as already generated.
    ///
    /// This is the meaning of the `history` argument of
    /// [`super::LogitsProcessor::process`], kept for source compatibility.
    pub(super) fn from_generated_tensor(
        config: &GenerationConfig,
        vocabulary_size: usize,
        batch: usize,
        history: Tensor<B, 2, Int>,
        mask: Option<TokenMask<B>>,
    ) -> Result<Self> {
        let [history_batch, width] = history.dims();
        if history_batch != batch {
            return Err(Error::shape(
                "generation.history",
                format!("[{batch}, sequence]"),
                format!("[{history_batch}, sequence]"),
            ));
        }
        let mask = mask.map(TokenMask::into_tensor);
        if let Some(mask) = &mask
            && mask.dims() != [batch, width]
        {
            return Err(Error::shape(
                "generation.history_mask",
                format!("[{batch}, {width}]"),
                format!("{:?}", mask.dims()),
            ));
        }
        Self::build(config, vocabulary_size, history, mask, None, true)
    }

    /// Builds the state the configuration needs from observed `[batch, width]` tokens.
    ///
    /// `host_valid` avoids re-reading a mask the caller already holds. When
    /// `as_generated` is true the tokens also count as generated output.
    fn build(
        config: &GenerationConfig,
        vocabulary_size: usize,
        tokens: Tensor<B, 2, Int>,
        mask: Option<Tensor<B, 2, Bool>>,
        host_valid: Option<&[bool]>,
        as_generated: bool,
    ) -> Result<Self> {
        let [batch, width] = tokens.dims();
        if batch == 0 {
            return Err(Error::shape(
                "generation.history",
                "[nonzero_batch, sequence]",
                format!("[{batch}, {width}]"),
            ));
        }
        if let Some(mask) = &mask
            && mask.dims() != [batch, width]
        {
            return Err(Error::shape(
                "generation.history_mask",
                format!("[{batch}, {width}]"),
                format!("{:?}", mask.dims()),
            ));
        }
        let device = tokens.device();
        let needs_counts = config.uses_token_counts();
        let ngram_size = config.no_repeat_ngram_size;
        let needs_lengths = as_generated && config.min_new_tokens > 0;
        let mut history = Self {
            vocabulary_size,
            counts: None,
            generated_lengths: vec![0; batch],
            ngram_size,
            ngrams: if ngram_size > 0 {
                vec![NgramIndex::default(); batch]
            } else {
                Vec::new()
            },
            device: device.clone(),
        };
        if width == 0 {
            history.counts = needs_counts.then(|| zero_counts(batch, vocabulary_size, &device));
            return Ok(history);
        }
        if !(needs_counts || ngram_size > 0 || needs_lengths) {
            return Ok(history);
        }

        // One host read validates IDs and feeds the n-gram index and lengths.
        let expected = metadata::rectangle_size(batch, width, "generation history")?;
        let valid = match (host_valid, &mask) {
            (Some(valid), _) => valid.to_vec(),
            (None, Some(mask)) => tensor_bool_values(mask.clone(), "generation history mask")?,
            (None, None) => vec![true; expected],
        };
        if valid.len() != expected {
            return Err(Error::Generation(format!(
                "history mask contains {} entries, expected {expected}",
                valid.len()
            )));
        }
        let needs_values = needs_counts || ngram_size > 0;
        let values = if needs_values {
            metadata::tensor_i64_values(tokens.clone(), "generation history")?
        } else {
            Vec::new()
        };
        for row in 0..batch {
            for column in 0..width {
                let offset = row * width + column;
                if !valid[offset] {
                    continue;
                }
                if as_generated {
                    history.generated_lengths[row] += 1;
                }
                if !needs_values {
                    continue;
                }
                let token = checked_history_token(values[offset], vocabulary_size, row, column)?;
                if ngram_size > 0 {
                    history.ngrams[row].push(token, ngram_size);
                }
            }
        }
        if needs_counts {
            history.counts = Some(observed_counts(
                tokens,
                mask,
                vocabulary_size,
                as_generated,
                &device,
            ));
        }
        Ok(history)
    }

    /// Records the token each active row emitted in the latest step.
    ///
    /// `tokens` and `active` have one entry per row; inactive rows (finished
    /// or padded) are ignored. Count tensors receive one scatter-add of the
    /// `[batch, 1]` new IDs; no data is read back from the device.
    pub fn record(&mut self, tokens: &[u32], active: &[bool]) -> Result<()> {
        let batch = self.batch_size();
        if tokens.len() != batch || active.len() != batch {
            return Err(Error::Generation(format!(
                "history update has {} tokens and {} activity flags for batch size {batch}",
                tokens.len(),
                active.len()
            )));
        }
        for (row, (&token, &is_active)) in tokens.iter().zip(active).enumerate() {
            if is_active && token as usize >= self.vocabulary_size {
                return Err(Error::Generation(format!(
                    "history update for row {row} has token {token} outside vocabulary size {}",
                    self.vocabulary_size
                )));
            }
        }
        if !active.iter().any(|&is_active| is_active) {
            return Ok(());
        }
        if let Some(counts) = &mut self.counts {
            // Inactive rows scatter a zero increment at token zero.
            let ids = tokens
                .iter()
                .zip(active)
                .map(|(&token, &is_active)| if is_active { i64::from(token) } else { 0 })
                .collect::<Vec<_>>();
            let increments = active
                .iter()
                .map(|&is_active| i64::from(is_active))
                .collect::<Vec<_>>();
            let ids =
                Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [batch, 1]), &self.device);
            let increments = Tensor::<B, 2, Int>::from_data(
                TensorData::new(increments, [batch, 1]),
                &self.device,
            );
            counts.all = counts.all.clone().scatter(
                1,
                ids.clone(),
                increments.clone(),
                IndexingUpdateOp::Add,
            );
            counts.generated =
                counts
                    .generated
                    .clone()
                    .scatter(1, ids, increments, IndexingUpdateOp::Add);
        }
        for (row, (&token, &is_active)) in tokens.iter().zip(active).enumerate() {
            if is_active {
                self.generated_lengths[row] += 1;
                if self.ngram_size > 0 {
                    self.ngrams[row].push(token, self.ngram_size);
                }
            }
        }
        Ok(())
    }

    /// Returns the number of independent rows.
    pub fn batch_size(&self) -> usize {
        self.generated_lengths.len()
    }

    /// Returns the exclusive upper bound of recorded token IDs.
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// Returns each row's generated length as far as the configuration tracks it.
    ///
    /// Histories built from a prompt or by [`Self::record`] always count
    /// exactly; the stateless `history` tensor path counts only when
    /// `min_new_tokens` needs it.
    pub(super) fn generated_lengths(&self) -> &[usize] {
        &self.generated_lengths
    }

    /// Returns the device-resident occurrence counts, when penalties need them.
    pub(super) fn counts(&self) -> Option<&TokenCounts<B>> {
        self.counts.as_ref()
    }

    /// Returns the tokens the n-gram constraint forbids for `row` next.
    pub(super) fn banned_tokens(&self, row: usize) -> &[u32] {
        if self.ngram_size == 0 {
            return &[];
        }
        self.ngrams[row].banned(self.ngram_size)
    }

    /// Rejects a history built for a different batch, vocabulary, or set of controls.
    pub(super) fn check_compatible(
        &self,
        config: &GenerationConfig,
        vocabulary_size: usize,
        batch: usize,
    ) -> Result<()> {
        if self.batch_size() != batch || self.vocabulary_size != vocabulary_size {
            return Err(Error::shape(
                "generation.token_history",
                format!("batch {batch} with vocabulary {vocabulary_size}"),
                format!(
                    "batch {} with vocabulary {}",
                    self.batch_size(),
                    self.vocabulary_size
                ),
            ));
        }
        if config.uses_token_counts() && self.counts.is_none() {
            return Err(Error::Generation(
                "token history tracks no counts, but the processor applies penalties; build the history from the processor's GenerationConfig".into(),
            ));
        }
        if config.no_repeat_ngram_size != self.ngram_size {
            return Err(Error::Generation(format!(
                "token history indexes {}-grams, but the processor forbids repeated {}-grams",
                self.ngram_size, config.no_repeat_ngram_size
            )));
        }
        Ok(())
    }
}

/// Returns zero-initialized count tensors for `batch` rows.
fn zero_counts<B: Backend>(
    batch: usize,
    vocabulary_size: usize,
    device: &B::Device,
) -> TokenCounts<B> {
    let zeros = Tensor::<B, 2, Int>::zeros([batch, vocabulary_size], device);
    TokenCounts {
        all: zeros.clone(),
        generated: zeros,
    }
}

/// Counts validated observed tokens with one `[batch, width]` scatter-add.
///
/// Masked slots are redirected to token zero with a zero increment, so their
/// sentinel values are never used as indices.
fn observed_counts<B: Backend>(
    tokens: Tensor<B, 2, Int>,
    mask: Option<Tensor<B, 2, Bool>>,
    vocabulary_size: usize,
    as_generated: bool,
    device: &B::Device,
) -> TokenCounts<B> {
    let [batch, _] = tokens.dims();
    let (indices, increments) = match mask {
        Some(mask) => (tokens.mask_fill(mask.clone().bool_not(), 0), mask.int()),
        None => {
            let increments = Tensor::<B, 2, Int>::ones(tokens.dims(), device);
            (tokens, increments)
        }
    };
    let observed = Tensor::<B, 2, Int>::zeros([batch, vocabulary_size], device).scatter(
        1,
        indices,
        increments,
        IndexingUpdateOp::Add,
    );
    let generated = if as_generated {
        observed.clone()
    } else {
        Tensor::<B, 2, Int>::zeros([batch, vocabulary_size], device)
    };
    TokenCounts {
        all: observed,
        generated,
    }
}

/// Converts one valid history value to a token ID inside the vocabulary.
fn checked_history_token(
    value: i64,
    vocabulary_size: usize,
    row: usize,
    column: usize,
) -> Result<u32> {
    u32::try_from(value)
        .ok()
        .filter(|&token| (token as usize) < vocabulary_size)
        .ok_or_else(|| {
            Error::shape(
                "generation.history",
                format!("token values in 0..{vocabulary_size}"),
                format!("token {value} at row {row}, column {column}"),
            )
        })
}

#[cfg(test)]
#[path = "history_tests.rs"]
mod tests;
