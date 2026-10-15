//! Batch-one lossless speculation with Nemotron-H Lightning's MTP stage.
//!
//! # One cycle
//!
//! Let `y_0` be a token already selected by the target (the *lead*) and `h`
//! the target's final hidden state immediately before it. A cycle
//!
//! 1. **drafts** `y_1..y_d` (`d <= K`): the MTP stage maps `(y_j, m_j)` to a
//!    draft distribution `q_{j+1}` and a feedback state `m_{j+1}`, starting
//!    from `m_0 = h`; drafting stops early at an end-of-sequence proposal;
//! 2. **verifies** every draft in **one** target pass over
//!    `[y_0, y_1, …, y_d]`, which yields the target distributions `p_0..p_d`
//!    (`p_i` predicts the token after `y_i`). Greedy decoding accepts `y_{i+1}`
//!    exactly when it is the target's choice under `p_i`; sampling accepts
//!    with probability `min(1, p_i(y) / q_{i+1}(y))` and replaces the first
//!    rejection by a draw from the normalized residual `(p_i - q_{i+1})_+`
//!    (the shared [`SpeculativeVerifier`]), which preserves the target
//!    marginal exactly. If every draft is accepted, `p_d` supplies a bonus;
//! 3. **publishes** the accepted prefix chosen by [`speculative_commit`]: `t`
//!    target tokens (the lead and the accepted non-EOS drafts) and the `t`
//!    MTP pairs `(y_i, h^target_{i-1})` rebuilt from the verification pass's
//!    hidden states, so the auxiliary cache matches a fresh
//!    [`NemotronHForCausalLm::prefill_mtp_cache`] over the same prefix.
//!
//! A cycle therefore costs one target pass and `d` MTP passes for `t + 1`
//! emitted tokens, instead of one target pass per token.
//!
//! # Determinism and atomicity
//!
//! All model work happens inside one target-plus-MTP cache transaction, and
//! target sampling, draft sampling, and acceptance each own an independent
//! seeded stream. Errors roll back the transaction and restore all three
//! streams, so a retry is bit-for-bit identical.
//!
//! # Token history
//!
//! Logits are processed exactly as [`crate::generate_causal`] would process
//! them: the caller passes the sequence's tokens before the lead and how many
//! of them form the prompt, so presence and frequency penalties count only
//! generated tokens (by default) and `min_new_tokens` counts generated tokens.
//!
//! Like vLLM's model-side speculator, this object executes one cycle rather
//! than owning a generation loop: the caller enforces the global
//! `max_new_tokens` budget, stop sequences, and padding. EOS tokens are
//! handled within a cycle because they determine the publishable prefix.

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::{
    Error, GenerationConfig, LogitsProcessor, Result, SamplingStrategy,
    generation::speculative::{
        SpeculativeOutcome, SpeculativeVerifier, Verdict, normalized_probabilities,
        speculative_commit,
    },
};

use super::{
    NemotronHDecodeState, NemotronHForCausalLm, mtp::DraftPairs, plan::verification_tokens,
};

/// XOR seed domain separating MTP proposal sampling from target sampling.
const DRAFT_SEED_DOMAIN: u64 = 0x4452_4146_545f_4e48;
/// XOR seed domain separating acceptance/residual draws from token sampling.
const VERIFY_SEED_DOMAIN: u64 = 0x5645_5249_4659_4e48;

/// Validated controls for one Nemotron-H MTP speculative decoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NemotronHSpeculativeConfig {
    /// Maximum number `K` of auxiliary candidates generated per cycle.
    pub max_draft_tokens: usize,
    /// Penalties, truncation, sampling policy, seed, and terminal token IDs.
    pub generation: GenerationConfig,
}

impl Default for NemotronHSpeculativeConfig {
    /// Uses the library-wide two-candidate production default.
    fn default() -> Self {
        Self {
            max_draft_tokens: 2,
            generation: GenerationConfig::default(),
        }
    }
}

impl NemotronHSpeculativeConfig {
    /// Validates proposal capacity and ordinary generation controls.
    ///
    /// # Errors
    ///
    /// Returns a generation error for zero drafts or an overflowing staging
    /// capacity, and the errors of [`GenerationConfig::validate`].
    pub fn validate(&self, vocabulary_size: usize) -> Result<()> {
        if self.max_draft_tokens == 0 {
            return Err(Error::Generation(
                "Nemotron-H MTP max_draft_tokens must be greater than zero".into(),
            ));
        }
        self.max_draft_tokens.checked_add(1).ok_or_else(|| {
            Error::Generation("Nemotron-H MTP proposal capacity overflows usize".into())
        })?;
        self.generation.validate(vocabulary_size)
    }
}

/// Observable result of one atomic speculative cycle.
#[derive(Debug)]
pub struct NemotronHSpeculativeOutput<B: Backend> {
    /// Emitted tokens, beginning with the guaranteed target lead token.
    pub token_ids: Vec<u32>,
    /// Number of candidates produced before verification stopped.
    pub drafted_tokens: usize,
    /// Number of MTP candidates accepted, including an accepted EOS.
    pub accepted_draft_tokens: usize,
    /// Whether every configured non-EOS draft was accepted and a bonus emitted.
    pub all_drafts_accepted: bool,
    /// Whether the final emitted token is a configured EOS token.
    pub terminated: bool,
    /// Target hidden state immediately before the final emitted token.
    ///
    /// Feed this state and the last token into the next cycle. It is absent
    /// after EOS because no continuation is valid.
    pub continuation_hidden_state: Option<Tensor<B, 3>>,
}

/// Stateful auxiliary proposer and exact target verifier.
///
/// Target sampling, draft sampling, and acceptance each own independent seeded
/// streams. A recoverable model/cache failure restores all three streams, so a
/// retry is bit-for-bit deterministic.
#[derive(Debug)]
pub struct NemotronHSpeculativeDecoder<B: Backend> {
    /// Validated draft capacity and generation policy shared by every cycle.
    config: NemotronHSpeculativeConfig,
    /// Fixed vocabulary extent checked against the target model before each cycle.
    vocabulary_size: usize,
    /// Target-logit processing and target token sampling stream.
    target: LogitsProcessor<B>,
    /// MTP-logit processing with an independent proposal sampling stream.
    draft: LogitsProcessor<B>,
    /// Acceptance tests and residual corrections with their own stream.
    verifier: SpeculativeVerifier,
}

impl<B: Backend> NemotronHSpeculativeDecoder<B> {
    /// Creates a lossless decoder for one fixed vocabulary and device.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`NemotronHSpeculativeConfig::validate`].
    pub fn new(
        config: NemotronHSpeculativeConfig,
        vocabulary_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate(vocabulary_size)?;
        let target = LogitsProcessor::new(config.generation.clone(), vocabulary_size, device)?;
        let mut draft_config = config.generation.clone();
        draft_config.seed ^= DRAFT_SEED_DOMAIN;
        let draft = LogitsProcessor::new(draft_config, vocabulary_size, device)?;
        let verifier = SpeculativeVerifier::new(
            config.generation.strategy,
            config.generation.seed ^ VERIFY_SEED_DOMAIN,
        );
        Ok(Self {
            config,
            vocabulary_size,
            target,
            draft,
            verifier,
        })
    }

    /// Returns immutable speculation and generation controls.
    #[must_use]
    pub const fn config(&self) -> &NemotronHSpeculativeConfig {
        &self.config
    }

    /// Selects the guaranteed lead token from one target-logit row `[1, vocabulary]`.
    ///
    /// `history` holds every token before the lead, of which the first
    /// `prompt_length` are the prompt.
    ///
    /// # Errors
    ///
    /// Returns a shape error for mis-sized logits or a prompt longer than the
    /// history, and the processing and sampling errors of [`LogitsProcessor`].
    pub fn select_lead(
        &self,
        target_logits: Tensor<B, 2>,
        history: &[u32],
        prompt_length: usize,
    ) -> Result<u32> {
        check_prompt_length(history, prompt_length)?;
        let processed =
            self.target
                .process_with_prompt_lengths(target_logits, &[history], &[prompt_length])?;
        first_token(self.target.select(processed)?)
    }

    /// Runs one active speculative cycle.
    ///
    /// `history_before_lead` holds every token before `lead_token` (its length
    /// must equal the target cache length) and its first `prompt_length`
    /// tokens are the prompt. The caller remains responsible for global token
    /// budgets and stop sequences; see the module-level scheduler boundary.
    ///
    /// # Errors
    ///
    /// Returns validation errors before any model work, and model, cache, or
    /// sampling errors after restoring the cache and every random stream.
    pub fn speculate(
        &mut self,
        model: &NemotronHForCausalLm<B>,
        lead_token: u32,
        target_hidden_before_lead: Tensor<B, 3>,
        history_before_lead: &[u32],
        prompt_length: usize,
        state: &mut NemotronHDecodeState<B>,
    ) -> Result<NemotronHSpeculativeOutput<B>> {
        self.speculate_if_active(
            model,
            lead_token,
            target_hidden_before_lead,
            history_before_lead,
            prompt_length,
            true,
            state,
        )
    }

    /// Runs a cycle only when its scheduler row remains active.
    ///
    /// An inactive cycle performs no model work, draws no randomness, and does
    /// not open a cache transaction.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::speculate`] for an active cycle.
    #[allow(clippy::too_many_arguments)]
    pub fn speculate_if_active(
        &mut self,
        model: &NemotronHForCausalLm<B>,
        lead_token: u32,
        target_hidden_before_lead: Tensor<B, 3>,
        history_before_lead: &[u32],
        prompt_length: usize,
        active: bool,
        state: &mut NemotronHDecodeState<B>,
    ) -> Result<NemotronHSpeculativeOutput<B>> {
        if !active {
            return Ok(NemotronHSpeculativeOutput {
                token_ids: Vec::new(),
                drafted_tokens: 0,
                accepted_draft_tokens: 0,
                all_drafts_accepted: false,
                terminated: false,
                continuation_hidden_state: Some(target_hidden_before_lead),
            });
        }
        self.validate_cycle(
            model,
            lead_token,
            &target_hidden_before_lead,
            history_before_lead,
            prompt_length,
            state,
        )?;
        if self.is_eos(lead_token) {
            return Ok(NemotronHSpeculativeOutput {
                token_ids: vec![lead_token],
                drafted_tokens: 0,
                accepted_draft_tokens: 0,
                all_drafts_accepted: false,
                terminated: true,
                continuation_hidden_state: None,
            });
        }

        let mark = state.mark()?;
        let recovery_mark = mark.clone();
        let target_snapshot = self.target.clone();
        let draft_snapshot = self.draft.clone();
        let verifier_snapshot = self.verifier;
        let cycle = Cycle {
            lead_token,
            hidden_before_lead: target_hidden_before_lead,
            history: history_before_lead,
            prompt_length,
        };
        match self.run_transaction(model, cycle, state) {
            Ok(pending) => {
                if let Err(error) = state.commit_prefixes(
                    mark,
                    pending.target_cache_tokens,
                    pending.mtp_cache_tokens,
                ) {
                    self.target = target_snapshot;
                    self.draft = draft_snapshot;
                    self.verifier = verifier_snapshot;
                    state.rollback(recovery_mark).map_err(|rollback| {
                        Error::Cache(format!(
                            "Nemotron-H speculative commit failed ({error}); rollback also failed ({rollback})"
                        ))
                    })?;
                    return Err(error);
                }
                Ok(pending.output)
            }
            Err(error) => {
                self.target = target_snapshot;
                self.draft = draft_snapshot;
                self.verifier = verifier_snapshot;
                state.rollback(mark).map_err(|rollback| {
                    Error::Cache(format!(
                        "Nemotron-H speculative step failed ({error}); rollback also failed ({rollback})"
                    ))
                })?;
                Err(error)
            }
        }
    }

    /// Drafts, verifies, and stages one cycle while both banks remain in a transaction.
    fn run_transaction(
        &mut self,
        model: &NemotronHForCausalLm<B>,
        cycle: Cycle<'_, B>,
        state: &mut NemotronHDecodeState<B>,
    ) -> Result<PendingCycle<B>> {
        // `sequence` grows to `history, y_0, y_1, …, y_d`; the lead sits at `base`.
        let base = cycle.history.len();
        let mut sequence = Vec::with_capacity(base + self.config.max_draft_tokens + 1);
        sequence.extend_from_slice(cycle.history);
        sequence.push(cycle.lead_token);
        let probabilities = self.draft(model, &cycle, &mut sequence, state)?;
        let drafted = probabilities.len();
        let last_is_eos = self.is_eos(sequence[base + drafted]);
        let evaluated = verification_tokens(drafted, last_is_eos).ok_or_else(|| {
            Error::Generation("Nemotron-H verification length overflows usize".into())
        })?;
        let (logits, hidden) = model.verify_tokens(&sequence[base..base + evaluated], state)?;
        // Row `i` predicts the token after `y_i` from the history `sequence[..=base + i]`.
        let histories = (0..evaluated)
            .map(|row| &sequence[..base + row + 1])
            .collect::<Vec<_>>();
        let processed = self.target.process_with_prompt_lengths(
            logits,
            &histories,
            &vec![cycle.prompt_length; evaluated],
        )?;
        let (outcome, last) = self.judge(
            processed,
            &sequence[base + 1..base + 1 + drafted],
            &probabilities,
        )?;
        let commit = speculative_commit(drafted, outcome).ok_or_else(|| {
            Error::Generation(
                "Nemotron-H speculative outcome is inconsistent with its draft".into(),
            )
        })?;
        let kept = commit.target_cache_tokens;

        // MTP pair `i` of the kept prefix is `(y_i, h^target before y_i)`.
        let pair_hidden = if kept > 1 {
            Tensor::cat(
                vec![
                    cycle.hidden_before_lead.clone(),
                    hidden.clone().narrow(1, 0, kept - 1),
                ],
                1,
            )
        } else {
            cycle.hidden_before_lead.clone()
        };
        let kept_tokens = sequence[base..base + kept]
            .iter()
            .map(|&token| i64::from(token))
            .collect::<Vec<_>>();
        model.append_mtp_pairs(
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(kept_tokens, [1, kept]),
                state.tensor_state().device(),
            ),
            pair_hidden,
            state,
        )?;

        let mut token_ids = sequence[base..base + kept].to_vec();
        token_ids.push(last);
        let terminated = self.is_eos(last);
        Ok(PendingCycle {
            output: NemotronHSpeculativeOutput {
                token_ids,
                drafted_tokens: drafted,
                accepted_draft_tokens: match outcome {
                    SpeculativeOutcome::Rejected { accepted }
                    | SpeculativeOutcome::AcceptedEndOfSequence { accepted } => accepted,
                    SpeculativeOutcome::AllAccepted => drafted,
                },
                all_drafts_accepted: outcome == SpeculativeOutcome::AllAccepted,
                terminated,
                continuation_hidden_state: (!terminated).then(|| hidden.narrow(1, kept - 1, 1)),
            },
            target_cache_tokens: kept,
            mtp_cache_tokens: commit.draft_cache_tokens,
        })
    }

    /// Proposes up to `K` drafts with the MTP stage and appends them to `sequence`.
    ///
    /// Returns each draft's processed probability vector when sampling (for
    /// the acceptance test) and `None` entries for greedy decoding.
    fn draft(
        &mut self,
        model: &NemotronHForCausalLm<B>,
        cycle: &Cycle<'_, B>,
        sequence: &mut Vec<u32>,
        state: &mut NemotronHDecodeState<B>,
    ) -> Result<Vec<Option<Vec<f64>>>> {
        let sampling = self.config.generation.strategy == SamplingStrategy::Sample;
        let mut pairs = DraftPairs::new();
        let mut probabilities = Vec::with_capacity(self.config.max_draft_tokens);
        let mut feedback = cycle.hidden_before_lead.clone();
        let mut input = cycle.lead_token;
        for _ in 0..self.config.max_draft_tokens {
            let prediction = model.draft_mtp(input, feedback, &mut pairs, state)?;
            feedback = prediction.hidden_state;
            let processed = self.draft.process_with_prompt_lengths(
                prediction.logits.squeeze_dim::<2>(1),
                &[sequence.as_slice()],
                &[cycle.prompt_length],
            )?;
            probabilities.push(
                sampling
                    .then(|| normalized_probabilities(processed.clone()))
                    .transpose()?,
            );
            let proposal = first_token(self.draft.select(processed)?)?;
            sequence.push(proposal);
            input = proposal;
            if self.is_eos(proposal) {
                break;
            }
        }
        Ok(probabilities)
    }

    /// Applies the acceptance rule to every draft in order.
    ///
    /// `processed` holds the target's processed logits after the lead and each
    /// evaluated draft. Returns the outcome and the final emitted token: the
    /// correction of the first rejection, an accepted EOS draft, or the bonus
    /// drawn from the last row when every draft is accepted.
    fn judge(
        &mut self,
        processed: Tensor<B, 2>,
        proposals: &[u32],
        probabilities: &[Option<Vec<f64>>],
    ) -> Result<(SpeculativeOutcome, u32)> {
        let greedy_choices = match self.config.generation.strategy {
            // One device argmax and one readback cover every verified row.
            SamplingStrategy::Greedy => Some(tokens(self.target.select(processed.clone())?)?),
            SamplingStrategy::Sample => None,
        };
        for (index, (&proposal, draft)) in proposals.iter().zip(probabilities).enumerate() {
            let verdict = match &greedy_choices {
                Some(choices) => Verdict::greedy(proposal, choices[index]),
                None => {
                    let target = normalized_probabilities(processed.clone().narrow(0, index, 1))?;
                    let draft = draft.as_deref().ok_or_else(|| {
                        Error::Generation(
                            "Nemotron-H stochastic verification lost its draft distribution".into(),
                        )
                    })?;
                    self.verifier.verify(proposal, &target, draft)?
                }
            };
            match verdict {
                Verdict::Rejected { correction } => {
                    return Ok((SpeculativeOutcome::Rejected { accepted: index }, correction));
                }
                Verdict::Accepted if self.is_eos(proposal) => {
                    return Ok((
                        SpeculativeOutcome::AcceptedEndOfSequence {
                            accepted: index + 1,
                        },
                        proposal,
                    ));
                }
                Verdict::Accepted => {}
            }
        }
        let drafted = proposals.len();
        let bonus = match greedy_choices {
            Some(choices) => *choices.get(drafted).ok_or_else(|| {
                Error::Generation("Nemotron-H verification omitted the bonus row".into())
            })?,
            None => first_token(self.target.select(processed.narrow(0, drafted, 1))?)?,
        };
        Ok((SpeculativeOutcome::AllAccepted, bonus))
    }

    /// Checks batch, shift alignment, vocabulary, history, shape, and free capacity.
    fn validate_cycle(
        &self,
        model: &NemotronHForCausalLm<B>,
        lead_token: u32,
        hidden: &Tensor<B, 3>,
        history_before_lead: &[u32],
        prompt_length: usize,
        state: &NemotronHDecodeState<B>,
    ) -> Result<()> {
        state.validate_for_model(model)?;
        if state.tensor_state().has_active_transaction() {
            return Err(crate::cache::TensorCacheError::TransactionAlreadyActive.into());
        }
        if self.vocabulary_size != model.config().vocab_size {
            return Err(Error::Generation(format!(
                "speculative decoder vocabulary {} does not match Nemotron-H vocabulary {}",
                self.vocabulary_size,
                model.config().vocab_size
            )));
        }
        if &hidden.device() != state.tensor_state().device() {
            return Err(Error::BackendCapability(
                "Nemotron-H speculative hidden state must use the decode-state device".into(),
            ));
        }
        check_prompt_length(history_before_lead, prompt_length)?;
        if history_before_lead
            .iter()
            .any(|&token| token as usize >= model.config().vocab_size)
        {
            return Err(Error::shape(
                "nemotron_h.speculative.history",
                format!("token values in 0..{}", model.config().vocab_size),
                "an out-of-vocabulary token",
            ));
        }
        if lead_token as usize >= model.config().vocab_size {
            return Err(Error::shape(
                "nemotron_h.mtp.lead_token",
                format!("a token in 0..{}", model.config().vocab_size),
                lead_token.to_string(),
            ));
        }
        if hidden.dims() != [1, 1, model.config().hidden_size] {
            return Err(Error::shape(
                "nemotron_h.mtp.target_hidden_before_lead",
                format!("[1, 1, {}]", model.config().hidden_size),
                format!("{:?}", hidden.dims()),
            ));
        }
        let required_staging = self.config.max_draft_tokens.checked_add(1).ok_or_else(|| {
            Error::Generation("Nemotron-H MTP proposal capacity overflows usize".into())
        })?;
        if state.staging_capacity() < required_staging {
            return Err(Error::Generation(format!(
                "Nemotron-H decode-state staging capacity {} is below required {required_staging}",
                state.staging_capacity()
            )));
        }
        let target = state.tensor_state().lengths();
        let mtp = state.tensor_state().mtp_lengths().ok_or_else(|| {
            Error::Cache("Nemotron-H speculative state has no independent MTP cursor".into())
        })?;
        if target.len() != 1 || mtp.len() != 1 || mtp[0].checked_add(1) != Some(target[0]) {
            return Err(Error::Cache(format!(
                "Nemotron-H speculative alignment requires target = MTP + 1, found target={target:?}, MTP={mtp:?}"
            )));
        }
        if history_before_lead.len() != target[0] {
            return Err(Error::shape(
                "nemotron_h.speculative.history",
                format!("{} tokens matching the target-cache cursor", target[0]),
                format!("{} tokens", history_before_lead.len()),
            ));
        }
        let requested = target[0].checked_add(required_staging).ok_or_else(|| {
            Error::Generation("Nemotron-H speculative target length overflows usize".into())
        })?;
        if requested > state.tensor_state().capacity() {
            return Err(Error::Generation(format!(
                "Nemotron-H speculative suffix reaches {requested}, exceeding cache capacity {}",
                state.tensor_state().capacity()
            )));
        }
        Ok(())
    }

    /// Returns whether a token terminates ordinary generation.
    fn is_eos(&self, token: u32) -> bool {
        self.config.generation.eos_token_ids.contains(&token)
    }
}

/// Inputs of one active cycle.
struct Cycle<'a, B: Backend> {
    /// Target-selected token that starts the cycle.
    lead_token: u32,
    /// Target final hidden state `[1, 1, hidden]` immediately before the lead.
    hidden_before_lead: Tensor<B, 3>,
    /// Every token before the lead, prompt first.
    history: &'a [u32],
    /// Number of leading prompt tokens in `history`.
    prompt_length: usize,
}

/// A verified cycle waiting for atomic publication.
struct PendingCycle<B: Backend> {
    /// The caller-visible result.
    output: NemotronHSpeculativeOutput<B>,
    /// Number of staged target tokens to publish from this transaction.
    target_cache_tokens: usize,
    /// Number of staged MTP pairs to publish (rebuilt from target hidden states).
    mtp_cache_tokens: usize,
}

/// Requires the prompt to be a prefix of the history.
fn check_prompt_length(history: &[u32], prompt_length: usize) -> Result<()> {
    if prompt_length > history.len() {
        return Err(Error::shape(
            "nemotron_h.speculative.prompt_length",
            format!("at most the {} history tokens", history.len()),
            prompt_length.to_string(),
        ));
    }
    Ok(())
}

/// Reads selected token IDs `[n]` with checked signed-to-unsigned conversion.
fn tokens<B: Backend>(selected: Tensor<B, 1, Int>) -> Result<Vec<u32>> {
    selected
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to read Nemotron-H selected tokens from the backend: {error}"
            ))
        })?
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to decode Nemotron-H tokens: {error}"))
        })?
        .into_iter()
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                Error::Generation(format!("Nemotron-H selected invalid token {value}"))
            })
        })
        .collect()
}

/// Reads the first selected token ID.
fn first_token<B: Backend>(selected: Tensor<B, 1, Int>) -> Result<u32> {
    tokens(selected)?
        .first()
        .copied()
        .ok_or_else(|| Error::Generation("Nemotron-H token selector returned no row".into()))
}

#[cfg(test)]
#[path = "speculative_tests.rs"]
mod tests;
