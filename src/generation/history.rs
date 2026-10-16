//! Incrementally maintained per-row token history for logits processing.
//!
//! History-dependent controls need different views of the tokens a row has
//! seen:
//!
//! - repetition, presence, and frequency penalties need per-token occurrence
//!   counts `c[b, v]`, which live on the device as `[batch, vocabulary]`
//!   integer tensors: one counting every observed token (prompt and output)
//!   and one counting generated tokens only. Each tensor exists only when a
//!   configured penalty reads it (see [`CountView`]);
//! - `no_repeat_ngram_size` needs the token sequence itself, kept on the host
//!   with an index from each `(n - 1)`-gram to the tokens that followed it;
//! - `min_new_tokens` needs each row's generated length, also on the host.
//!
//! Rebuilding counts from a `[batch, history]` tensor every step costs a
//! `[batch, history, vocabulary]` one-hot (gigabytes for a 4k-token prompt and
//! a 250k vocabulary) plus host synchronizations. [`TokenHistory`] instead
//! scatter-adds the prompt once, `O(batch * prompt)`, and then one token per
//! row per step, uploading only the new `[batch, 1]` IDs. A coordinator that
//! already reads selected IDs back to the host therefore adds no extra
//! synchronization.
//!
//! # Update cost
//!
//! [`TokenHistory::record`] moves each count tensor out of the history before
//! scattering into it, so the tensor is uniquely owned during the update.
//! Backends with in-place scatter (CubeCL, used by Metal, CUDA, and ROCm)
//! then add `batch` increments without touching the rest of the
//! `[batch, vocabulary]` buffer. Burn 0.21's Flex CPU scatter always writes a
//! fresh output, so there each update is one `[batch, vocabulary]` copy per
//! tracked view; tracking only the views a configuration reads keeps that to
//! at most two. A cloned history shares its buffers until the next update,
//! which then copies (copy-on-write), so clones never observe each other.

use std::collections::HashMap;

use burn::tensor::{Bool, IndexingUpdateOp, Int, Tensor, TensorData, backend::Backend};

use crate::{Error, Result, TokenMask};

use super::bookkeeping::row_span;
use super::layout::{count_valid, history_token_id, push_history_row, record_scatter};
use super::ngram::{completed_ngram_prefix, insert_follower, ngram_lookup_start};
use super::{GenerationConfig, metadata};

/// Which occurrence counts a penalty reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CountView {
    /// Every observed token, prompt and generated (repetition penalty, and
    /// additive penalties scoped to [`super::PenaltyScope::PromptAndGenerated`]).
    All,
    /// Generated tokens only (additive penalties with the default
    /// [`super::PenaltyScope::Generated`] scope).
    Generated,
}

impl CountView {
    /// Human-readable name used in compatibility errors.
    fn label(self) -> &'static str {
        match self {
            Self::All => "prompt-and-generated",
            Self::Generated => "generated-only",
        }
    }
}

/// Device-resident occurrence counts; each view exists only when a penalty reads it.
#[derive(Debug, Clone)]
struct TokenCounts<B: Backend> {
    /// `[batch, vocabulary]` counts of every observed token, prompt and generated.
    all: Option<Tensor<B, 2, Int>>,
    /// `[batch, vocabulary]` counts of generated tokens only.
    generated: Option<Tensor<B, 2, Int>>,
}

impl<B: Backend> TokenCounts<B> {
    /// Allocates separate zero tensors for the views `config` reads.
    ///
    /// Separate buffers (rather than one shared zero tensor) keep each view
    /// uniquely owned, so the first [`TokenHistory::record`] needs no copy.
    fn zeros(
        config: &GenerationConfig,
        batch: usize,
        vocabulary_size: usize,
        device: &B::Device,
    ) -> Self {
        let zeros = || Tensor::<B, 2, Int>::zeros([batch, vocabulary_size], device);
        Self {
            all: config.reads_count_view(CountView::All).then(zeros),
            generated: config.reads_count_view(CountView::Generated).then(zeros),
        }
    }

    /// Whether any view is tracked.
    fn is_tracking(&self) -> bool {
        self.all.is_some() || self.generated.is_some()
    }

    /// Adds `increments` at `[batch, width]` token `ids` to every tracked view.
    ///
    /// `Option::take` moves each tensor out before the scatter so it is
    /// uniquely owned; see the module documentation for the cost model.
    fn add(&mut self, ids: Tensor<B, 2, Int>, increments: Tensor<B, 2, Int>) {
        self.all = self
            .all
            .take()
            .map(|counts| scatter_add(counts, ids.clone(), increments.clone()));
        self.generated = self
            .generated
            .take()
            .map(|counts| scatter_add(counts, ids, increments));
    }
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
    ///
    /// `size` is positive. The prefix range and the sorted, deduplicated
    /// follower insertion are the Verus-verified [`completed_ngram_prefix`]
    /// and [`insert_follower`].
    fn push(&mut self, token: u32, size: usize) {
        self.tokens.push(token);
        if let Some((start, end)) = completed_ngram_prefix(self.tokens.len(), size) {
            let prefix = self.tokens[start..end].to_vec();
            insert_follower(self.continuations.entry(prefix).or_default(), token);
        }
    }

    /// Tokens that would complete an n-gram of `size` already present.
    ///
    /// Matches Hugging Face `NoRepeatNGramLogitsProcessor`: nothing is banned
    /// until the next token could complete a full n-gram, and the lookup key
    /// is the last `n - 1` tokens. `size` is positive; the verified
    /// [`ngram_lookup_start`] proves this key is the prefix [`Self::push`]
    /// records the next token under.
    fn banned(&self, size: usize) -> &[u32] {
        let Some(start) = ngram_lookup_start(self.tokens.len(), size) else {
            return &[];
        };
        self.continuations
            .get(&self.tokens[start..])
            .map_or(&[], Vec::as_slice)
    }
}

/// Per-row token statistics maintained incrementally across decode steps.
///
/// Create one with [`TokenHistory::from_prompt`] (or [`TokenHistory::empty`])
/// using the same [`GenerationConfig`] as the [`super::LogitsProcessor`], pass
/// it to [`super::LogitsProcessor::process_with_history`] each step, and call
/// [`TokenHistory::record`] with the tokens each row emitted. Prompt tokens
/// count for the repetition penalty and n-grams but not as generated tokens.
/// Callers that hold whole token rows on the host (for example speculative
/// decoders that roll drafts back) can rebuild the same state in one call
/// with [`TokenHistory::from_token_rows`].
#[derive(Debug, Clone)]
pub struct TokenHistory<B: Backend> {
    /// Exclusive upper bound of every recorded token ID.
    vocabulary_size: usize,
    /// Occurrence counts for the views the configured penalties read.
    counts: TokenCounts<B>,
    /// Number of tokens recorded as generated for each row.
    generated_lengths: Vec<usize>,
    /// N-gram size of the no-repeat constraint; zero disables the index.
    ngram_size: usize,
    /// One n-gram index per row, empty when `ngram_size == 0`.
    ngrams: Vec<NgramIndex>,
    /// Host upper bound on every occurrence count `c[b, v]`.
    ///
    /// The widest observed `[batch, width]` prompt (or longest token row)
    /// plus one per [`Self::record`] call that recorded a token. Penalties use
    /// it to decide, without reading the counts, whether `count * penalty`
    /// can overflow the logits dtype.
    count_bound: usize,
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

    /// Creates history from host token rows with an explicit prompt boundary.
    ///
    /// Row `b` holds `rows[b]`: its first `prompt_lengths[b]` tokens are the
    /// prompt and the remainder was generated. The result equals
    /// [`Self::from_prompt`] on the prompt followed by one [`Self::record`]
    /// per generated token, built with at most one scatter per tracked view
    /// and no device read. Rows may have different lengths; every token must
    /// lie inside the vocabulary.
    pub fn from_token_rows(
        config: &GenerationConfig,
        vocabulary_size: usize,
        rows: &[&[u32]],
        prompt_lengths: &[usize],
        device: &B::Device,
    ) -> Result<Self> {
        let batch = rows.len();
        if batch == 0 || prompt_lengths.len() != batch {
            return Err(Error::shape(
                "generation.history_rows",
                "nonzero rows with one prompt length each",
                format!("{batch} rows and {} prompt lengths", prompt_lengths.len()),
            ));
        }
        let mut history = Self::new_unfilled(config, vocabulary_size, batch, device);
        let mut width = 0;
        for (row, (&tokens, &prompt_length)) in rows.iter().zip(prompt_lengths).enumerate() {
            let generated = tokens.len().checked_sub(prompt_length).ok_or_else(|| {
                Error::Generation(format!(
                    "history row {row} has {} tokens, fewer than its prompt length {prompt_length}",
                    tokens.len()
                ))
            })?;
            for (column, &token) in tokens.iter().enumerate() {
                checked_history_token(i64::from(token), vocabulary_size, row, column)?;
                if history.ngram_size > 0 {
                    history.ngrams[row].push(token, history.ngram_size);
                }
            }
            history.generated_lengths[row] = generated;
            width = width.max(tokens.len());
        }
        history.counts = row_counts(config, vocabulary_size, rows, prompt_lengths, width, device)?;
        history.count_bound = width;
        Ok(history)
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

    /// Returns history with no observed tokens and no count tensors yet.
    fn new_unfilled(
        config: &GenerationConfig,
        vocabulary_size: usize,
        batch: usize,
        device: &B::Device,
    ) -> Self {
        let ngram_size = config.no_repeat_ngram_size;
        Self {
            vocabulary_size,
            counts: TokenCounts {
                all: None,
                generated: None,
            },
            generated_lengths: vec![0; batch],
            ngram_size,
            ngrams: if ngram_size > 0 {
                vec![NgramIndex::default(); batch]
            } else {
                Vec::new()
            },
            count_bound: 0,
            device: device.clone(),
        }
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
        let mut history = Self::new_unfilled(config, vocabulary_size, batch, &device);
        // A row holds at most `width` valid tokens, so no count exceeds it.
        history.count_bound = width;
        let needs_counts = config.reads_count_view(CountView::All)
            || config.reads_count_view(CountView::Generated);
        let ngram_size = history.ngram_size;
        let needs_lengths = as_generated && config.min_new_tokens > 0;
        if width == 0 || !(needs_counts || ngram_size > 0 || needs_lengths) {
            history.counts = TokenCounts::zeros(config, batch, vocabulary_size, &device);
            return Ok(history);
        }

        // One host read (mask and IDs together) validates IDs and feeds the
        // n-gram index and lengths; a mask the caller already holds is not re-read.
        let expected = metadata::rectangle_size(batch, width, "generation history")?;
        let needs_values = needs_counts || ngram_size > 0;
        let device_mask = match host_valid {
            Some(_) => None,
            None => mask.clone(),
        };
        let grid = metadata::read_grid(
            device_mask,
            needs_values.then(|| tokens.clone()),
            batch,
            width,
            "generation history",
        )?;
        let valid = match host_valid {
            Some(valid) => valid.to_vec(),
            None => grid.valid,
        };
        if valid.len() != expected {
            return Err(Error::Generation(format!(
                "history mask contains {} entries, expected {expected}",
                valid.len()
            )));
        }
        let values = grid.values.unwrap_or_default();
        for row in 0..batch {
            let (start, end) = row_span(row, batch, width, 0, width);
            if as_generated {
                history.generated_lengths[row] = count_valid(&valid, start, end);
            }
            if !needs_values {
                continue;
            }
            for column in 0..width {
                let offset = start + column;
                if !valid[offset] {
                    continue;
                }
                let token = checked_history_token(values[offset], vocabulary_size, row, column)?;
                if ngram_size > 0 {
                    history.ngrams[row].push(token, ngram_size);
                }
            }
        }
        history.counts = observed_counts(config, tokens, mask, vocabulary_size, as_generated);
        Ok(history)
    }

    /// Records the token each active row emitted in the latest step.
    ///
    /// `tokens` and `active` have one entry per row; inactive rows (finished
    /// or padded) are ignored. Each tracked count tensor receives one
    /// scatter-add of the `[batch, 1]` new IDs; no data is read back from the
    /// device. See the module documentation for the per-backend cost.
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
        // Each active row gains one token, so no count grows by more than one.
        self.count_bound = self.count_bound.saturating_add(1);
        if self.counts.is_tracking() {
            // Inactive rows scatter a zero increment at token zero.
            let (ids, increments) = record_scatter(tokens, active);
            let ids =
                Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [batch, 1]), &self.device);
            let increments = Tensor::<B, 2, Int>::from_data(
                TensorData::new(increments, [batch, 1]),
                &self.device,
            );
            self.counts.add(ids, increments);
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
    /// Histories built from a prompt, from token rows, or by [`Self::record`]
    /// always count exactly; the stateless `history` tensor path counts only
    /// when `min_new_tokens` needs it.
    pub(super) fn generated_lengths(&self) -> &[usize] {
        &self.generated_lengths
    }

    /// Returns a host upper bound on every occurrence count, without a device read.
    pub(super) fn count_bound(&self) -> usize {
        self.count_bound
    }

    /// Returns the `[batch, vocabulary]` counts of `view`.
    ///
    /// Fails contextually when the history was built for a configuration
    /// whose penalties do not read that view.
    pub(super) fn counts(&self, view: CountView) -> Result<&Tensor<B, 2, Int>> {
        let counts = match view {
            CountView::All => self.counts.all.as_ref(),
            CountView::Generated => self.counts.generated.as_ref(),
        };
        counts.ok_or_else(|| {
            Error::Generation(format!(
                "token history tracks no {} token counts, but the processor's penalties read them; build the history from the processor's GenerationConfig",
                view.label()
            ))
        })
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
        for view in [CountView::All, CountView::Generated] {
            if config.reads_count_view(view) {
                self.counts(view)?;
            }
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

/// Adds `increments` to `counts` at token `ids` along the vocabulary axis.
fn scatter_add<B: Backend>(
    counts: Tensor<B, 2, Int>,
    ids: Tensor<B, 2, Int>,
    increments: Tensor<B, 2, Int>,
) -> Tensor<B, 2, Int> {
    counts.scatter(1, ids, increments, IndexingUpdateOp::Add)
}

/// Counts validated observed tokens with one `[batch, width]` scatter-add.
///
/// Masked slots are redirected to token zero with a zero increment, so their
/// sentinel values are never used as indices. The generated view of a prompt
/// (`as_generated == false`) starts at zero.
fn observed_counts<B: Backend>(
    config: &GenerationConfig,
    tokens: Tensor<B, 2, Int>,
    mask: Option<Tensor<B, 2, Bool>>,
    vocabulary_size: usize,
    as_generated: bool,
) -> TokenCounts<B> {
    let [batch, _] = tokens.dims();
    let device = tokens.device();
    let mut counts = TokenCounts::zeros(config, batch, vocabulary_size, &device);
    if counts.all.is_none() && !(as_generated && counts.generated.is_some()) {
        return counts;
    }
    let (indices, increments) = match mask {
        Some(mask) => (tokens.mask_fill(mask.clone().bool_not(), 0), mask.int()),
        None => {
            let increments = Tensor::<B, 2, Int>::ones(tokens.dims(), &device);
            (tokens, increments)
        }
    };
    counts.all = counts
        .all
        .take()
        .map(|zeros| scatter_add(zeros, indices.clone(), increments.clone()));
    if as_generated {
        counts.generated = counts
            .generated
            .take()
            .map(|zeros| scatter_add(zeros, indices, increments));
    }
    counts
}

/// Counts host token rows with one padded `[batch, width]` upload per tracked view.
///
/// Padding slots (beyond a row's length) and, for the generated view, prompt
/// slots carry a zero increment at token zero, so they change nothing.
fn row_counts<B: Backend>(
    config: &GenerationConfig,
    vocabulary_size: usize,
    rows: &[&[u32]],
    prompt_lengths: &[usize],
    width: usize,
    device: &B::Device,
) -> Result<TokenCounts<B>> {
    let batch = rows.len();
    let mut counts = TokenCounts::zeros(config, batch, vocabulary_size, device);
    if width == 0 || !counts.is_tracking() {
        return Ok(counts);
    }
    let elements = metadata::rectangle_size(batch, width, "history rows")?;
    let mut ids = metadata::vec_with_capacity::<i64>(elements, "history row IDs")?;
    let mut observed = metadata::vec_with_capacity::<i64>(elements, "history row increments")?;
    let mut generated = metadata::vec_with_capacity::<i64>(elements, "generated increments")?;
    for (&tokens, &prompt_length) in rows.iter().zip(prompt_lengths) {
        push_history_row(
            &mut ids,
            &mut observed,
            &mut generated,
            tokens,
            prompt_length,
            width,
        );
    }
    let upload = |values: Vec<i64>| {
        Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch, width]), device)
    };
    let ids = upload(ids);
    counts.all = counts
        .all
        .take()
        .map(|zeros| scatter_add(zeros, ids.clone(), upload(observed)));
    counts.generated = counts
        .generated
        .take()
        .map(|zeros| scatter_add(zeros, ids, upload(generated)));
    Ok(counts)
}

/// Converts one valid history value to a token ID inside the vocabulary.
fn checked_history_token(
    value: i64,
    vocabulary_size: usize,
    row: usize,
    column: usize,
) -> Result<u32> {
    history_token_id(value, vocabulary_size).ok_or_else(|| {
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
