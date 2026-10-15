//! Batch-one lossless speculative decoding with Qwen's MTP head.
//!
//! # One cycle
//!
//! A cycle starts at target length `L` with a target-selected lead token
//! `y_0 = x_L` and the target hidden state `h_(L-1)` that produced it. Then:
//!
//! 1. **Draft.** The MTP head proposes `d_1..d_K` autoregressively: step 0
//!    reads `(y_0, h_(L-1))`, step `j` reads `(d_j, m_(j-1))` with its own
//!    previous output `m`, as the released runtime does. These MTP cache
//!    entries live in a scratch transaction that is rolled back afterwards.
//! 2. **Verify.** One target pass over `[y_0, d_1, ..., d_K]` yields every
//!    verification distribution `p_0..p_K` and hidden state `h_L..h_(L+K)` at
//!    once, staging all `K + 1` tokens. Greedy acceptance is equality with
//!    the target's choice; stochastic acceptance applies
//!    `a_i = min(1, p_i(y_i) / q_i(y_i))` and samples a rejection's correction
//!    from the normalized residual `(p_i - q_i)_+`
//!    ([`SpeculativeVerifier`]), so the output distribution is exactly the
//!    target's. If every draft is accepted, `p_K` yields one bonus token.
//! 3. **Commit.** [`speculative_commit`] gives the accepted target prefix `c`.
//!    The MTP layer is re-run over the `c` committed pairs
//!    `(x_(L+j), h_(L-1+j))`, `j < c`, all built from *target* hidden states,
//!    and both prefixes are committed atomically, keeping the invariant
//!    `target length = MTP length + 1`.
//!
//! Step 3 follows the pinned vLLM speculator, whose drafter is always
//! prefilled over verified tokens with the target's hidden states; the MTP
//! history of accepted drafts therefore never contains the approximations
//! `m_j` the head was not trained on. Any error restores both caches and every
//! random stream.
//!
//! # Logits processing
//!
//! Target and draft logits are processed exactly as
//! [`generate_causal`](crate::generate_causal) processes them, through
//! [`LogitsProcessor::process_with_prompt_lengths`]: the caller supplies the
//! prompt length, so presence and frequency penalties and `min_new_tokens`
//! count generated tokens only, and half-precision logits are promoted to
//! F32. Stochastic drafts are sampled on the host from the same F64 vector
//! `q` that verification uses, so the proposal is drawn from exactly the
//! distribution the acceptance rule assumes.

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::{
    CausalLmInput, Error, ForwardOptions, GenerationConfig, LogitsProcessor, LogitsSelection,
    Result, SamplingStrategy,
    generation::{
        SplitMix64,
        speculative::{
            SpeculativeOutcome, SpeculativeVerifier, Verdict, normalized_probabilities,
            sample_weights, speculative_commit,
        },
    },
};

use super::{Qwen3_5DecodeState, Qwen3_5ForCausalLm, decode::StatefulPhase};

/// XOR domain separating the draft sampling stream from target token sampling.
const DRAFT_SEED_DOMAIN: u64 = 0x4452_4146_545f_5157;
/// XOR domain separating acceptance and residual draws from token sampling.
const VERIFY_SEED_DOMAIN: u64 = 0x5645_5249_4659_5157;

/// Validated controls for one Qwen MTP speculative decoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Qwen3_5SpeculativeConfig {
    /// Maximum number `K` of MTP proposals per cycle.
    pub max_draft_tokens: usize,
    /// Penalties, truncation, sampling strategy, seed, and EOS IDs.
    pub generation: GenerationConfig,
}

impl Default for Qwen3_5SpeculativeConfig {
    /// Uses two draft tokens per cycle and the ordinary generation defaults.
    fn default() -> Self {
        Self {
            max_draft_tokens: 2,
            generation: GenerationConfig::default(),
        }
    }
}

impl Qwen3_5SpeculativeConfig {
    /// Validates proposal count and all ordinary generation controls.
    pub fn validate(&self, vocabulary_size: usize) -> Result<()> {
        if self.max_draft_tokens == 0 {
            return Err(Error::Generation(
                "Qwen MTP max_draft_tokens must be greater than zero".into(),
            ));
        }
        self.max_draft_tokens.checked_add(1).ok_or_else(|| {
            Error::Generation("Qwen MTP proposal capacity overflows usize".into())
        })?;
        self.generation.validate(vocabulary_size)
    }
}

/// Observable result of one atomic speculative cycle.
#[derive(Debug)]
pub struct Qwen3_5SpeculativeOutput<B: Backend> {
    /// Tokens of the cycle, beginning with the guaranteed target lead token.
    ///
    /// The last token is the next cycle's lead. A caller that streams every
    /// cycle's tokens therefore emits `token_ids[0]` only for the first cycle
    /// (whose lead came from [`Qwen3_5SpeculativeDecoder::select_lead`]) and
    /// skips it afterwards, because the previous cycle already emitted it.
    pub token_ids: Vec<u32>,
    /// Number of MTP candidates generated before target verification stopped.
    pub drafted_tokens: usize,
    /// Number of MTP candidates accepted by the target, including an accepted EOS.
    pub accepted_draft_tokens: usize,
    /// Whether every configured non-EOS draft was accepted and a bonus was emitted.
    pub all_drafts_accepted: bool,
    /// Whether the emitted suffix ended in a configured EOS token.
    pub terminated: bool,
    /// Target hidden state immediately before the last emitted token.
    ///
    /// Feed this state and the last emitted token into the next cycle. It is
    /// absent after EOS because no continuation is valid.
    pub continuation_hidden_state: Option<Tensor<B, 3>>,
}

/// Stateful Qwen MTP proposer and exact target verifier.
///
/// Randomness comes from independent deterministic streams: the target
/// processor's (lead and bonus tokens), the host draft stream (stochastic
/// proposals), and the verifier's (acceptance and residual draws). Unrelated
/// model instances therefore cannot perturb reproducibility, and a failed
/// cycle restores all three.
#[derive(Debug)]
pub struct Qwen3_5SpeculativeDecoder<B: Backend> {
    /// Validated proposal-count and ordinary generation controls.
    config: Qwen3_5SpeculativeConfig,
    /// Expected probability-vector width and exclusive token-ID bound.
    vocabulary_size: usize,
    /// Target penalties, truncation, and the lead/bonus sampling stream.
    target: LogitsProcessor<B>,
    /// Draft penalties and truncation; greedy drafts select through it.
    draft: LogitsProcessor<B>,
    /// Host uniforms for stochastic proposals drawn from the draft distribution.
    draft_stream: SplitMix64,
    /// Acceptance tests and residual sampling.
    verifier: SpeculativeVerifier,
}

impl<B: Backend> Qwen3_5SpeculativeDecoder<B> {
    /// Creates a decoder whose default proposal count is two.
    pub fn new(
        config: Qwen3_5SpeculativeConfig,
        vocabulary_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate(vocabulary_size)?;
        let target = LogitsProcessor::new(config.generation.clone(), vocabulary_size, device)?;
        let mut draft_config = config.generation.clone();
        draft_config.seed ^= DRAFT_SEED_DOMAIN;
        let draft = LogitsProcessor::new(draft_config, vocabulary_size, device)?;
        let seed = config.generation.seed;
        Ok(Self {
            draft_stream: SplitMix64::new(seed ^ DRAFT_SEED_DOMAIN),
            verifier: SpeculativeVerifier::new(
                config.generation.strategy,
                seed ^ VERIFY_SEED_DOMAIN,
            ),
            config,
            vocabulary_size,
            target,
            draft,
        })
    }

    /// Returns immutable speculative and ordinary generation controls.
    #[must_use]
    pub const fn config(&self) -> &Qwen3_5SpeculativeConfig {
        &self.config
    }

    /// Selects the guaranteed lead token from `[1, vocabulary]` target logits.
    ///
    /// `history` is the prompt followed by every generated token and
    /// `prompt_length` marks where the prompt ends, as for
    /// [`Self::speculate`]. The token comes from the target processor's
    /// stream, the one that later samples all-accepted bonus tokens.
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
        let tokens = read_tokens(self.target.select(processed)?)?;
        tokens
            .first()
            .copied()
            .ok_or_else(|| Error::Generation("target processor returned no lead token".into()))
    }

    /// Runs one active speculative cycle.
    ///
    /// `history_before_lead` is the prompt followed by every token generated
    /// before `lead_token`; its length must equal the committed target length.
    /// The first `prompt_length` of those tokens are the prompt.
    pub fn speculate(
        &mut self,
        model: &Qwen3_5ForCausalLm<B>,
        lead_token: u32,
        target_hidden_before_lead: Tensor<B, 3>,
        history_before_lead: &[u32],
        prompt_length: usize,
        state: &mut Qwen3_5DecodeState<B>,
    ) -> Result<Qwen3_5SpeculativeOutput<B>> {
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

    /// Runs a cycle only when a scheduler row remains active.
    ///
    /// `active = false` is a strict no-op: it draws no randomness, performs no
    /// model computation, and leaves both caches unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn speculate_if_active(
        &mut self,
        model: &Qwen3_5ForCausalLm<B>,
        lead_token: u32,
        target_hidden_before_lead: Tensor<B, 3>,
        history_before_lead: &[u32],
        prompt_length: usize,
        active: bool,
        state: &mut Qwen3_5DecodeState<B>,
    ) -> Result<Qwen3_5SpeculativeOutput<B>> {
        if !active {
            return Ok(Qwen3_5SpeculativeOutput {
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
            return Ok(Qwen3_5SpeculativeOutput {
                token_ids: vec![lead_token],
                drafted_tokens: 0,
                accepted_draft_tokens: 0,
                all_drafts_accepted: false,
                terminated: true,
                continuation_hidden_state: None,
            });
        }
        let snapshot = (
            self.target.clone(),
            self.draft.clone(),
            self.draft_stream,
            self.verifier,
        );
        let cycle = Cycle {
            lead_token,
            target_hidden_before_lead,
            history_before_lead,
            prompt_length,
        };
        let result = self.run_cycle(model, &cycle, state);
        if result.is_err() {
            (self.target, self.draft, self.draft_stream, self.verifier) = snapshot;
        }
        result
    }

    /// Drafts, verifies, and commits one cycle; every exit leaves no open transaction.
    fn run_cycle(
        &mut self,
        model: &Qwen3_5ForCausalLm<B>,
        cycle: &Cycle<'_, B>,
        state: &mut Qwen3_5DecodeState<B>,
    ) -> Result<Qwen3_5SpeculativeOutput<B>> {
        let mark = state.mark()?;
        let drafts = self.draft(model, cycle, state);
        // Draft KV entries are scratch: committed MTP history is rebuilt
        // from target states below, so the drafting suffix is always dropped.
        state.rollback(mark)?;
        let drafts = drafts?;
        let mark = state.mark()?;
        match self.verify_and_stage(model, cycle, &drafts, state) {
            Ok((output, commit)) => {
                state.commit_prefixes(
                    mark,
                    commit.target_cache_tokens,
                    commit.draft_cache_tokens,
                )?;
                Ok(output)
            }
            Err(error) => {
                state.rollback(mark).map_err(|rollback| {
                    Error::Cache(format!(
                        "Qwen speculative step failed ({error}); rollback also failed ({rollback})"
                    ))
                })?;
                Err(error)
            }
        }
    }

    /// Proposes up to `K` tokens with the MTP head, stopping after an EOS proposal.
    ///
    /// Returns the proposals and, for stochastic decoding, the host draft
    /// distribution `q_i` each was sampled from.
    fn draft(
        &mut self,
        model: &Qwen3_5ForCausalLm<B>,
        cycle: &Cycle<'_, B>,
        state: &mut Qwen3_5DecodeState<B>,
    ) -> Result<Drafts> {
        let device = cycle.target_hidden_before_lead.device();
        let mut history = cycle.history_before_lead.to_vec();
        history.push(cycle.lead_token);
        let mut drafts = Drafts {
            tokens: Vec::with_capacity(self.config.max_draft_tokens),
            distributions: Vec::with_capacity(self.config.max_draft_tokens),
        };
        let mut hidden = cycle.target_hidden_before_lead.clone();
        let mut input = cycle.lead_token;
        for step in 0..self.config.max_draft_tokens {
            let output = model.forward_mtp_cached(
                token_tensor(&[input], &device),
                hidden,
                None,
                step,
                state,
            )?;
            hidden = output.hidden_state;
            let processed = self.draft.process_with_prompt_lengths(
                output.logits.squeeze_dim::<2>(1),
                &[&history],
                &[cycle.prompt_length],
            )?;
            let proposal = match self.config.generation.strategy {
                SamplingStrategy::Greedy => single_token(self.draft.select(processed)?)?,
                SamplingStrategy::Sample => {
                    let distribution = normalized_probabilities(processed)?;
                    let proposal =
                        sample_weights(&distribution, self.draft_stream.next_unit_f64())?;
                    drafts.distributions.push(distribution);
                    proposal
                }
            };
            drafts.tokens.push(proposal);
            history.push(proposal);
            input = proposal;
            if self.is_eos(proposal) {
                break;
            }
        }
        Ok(drafts)
    }

    /// Verifies every draft with one target pass and stages the committed MTP pairs.
    ///
    /// Must run inside a fresh transaction; returns the cycle output and the
    /// prefix lengths to commit.
    fn verify_and_stage(
        &mut self,
        model: &Qwen3_5ForCausalLm<B>,
        cycle: &Cycle<'_, B>,
        drafts: &Drafts,
        state: &mut Qwen3_5DecodeState<B>,
    ) -> Result<(
        Qwen3_5SpeculativeOutput<B>,
        crate::generation::speculative::SpeculativeCommit,
    )> {
        let device = cycle.target_hidden_before_lead.device();
        let drafted = drafts.tokens.len();
        if drafted == 0 {
            return Err(Error::Generation("MTP produced no proposal".into()));
        }
        // [y_0, d_1, ..., d_K]: position i verifies d_(i+1) or yields the bonus.
        let mut verified = Vec::with_capacity(drafted + 1);
        verified.push(cycle.lead_token);
        verified.extend(&drafts.tokens);
        let target = model.cached_forward(
            CausalLmInput {
                token_ids: token_tensor(&verified, &device),
                token_mask: None,
                position_ids: None,
            },
            state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
            StatefulPhase::Decode,
            true,
        )?;
        let target_hidden = target.final_hidden_state.ok_or_else(|| {
            Error::Generation("target verification omitted its final hidden states".into())
        })?;
        // Row i is processed with the history up to and including verified[i].
        let mut histories = Vec::with_capacity(drafted + 1);
        let mut history = cycle.history_before_lead.to_vec();
        for &token in &verified {
            history.push(token);
            histories.push(history.clone());
        }
        let history_rows = histories.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let processed = self.target.process_with_prompt_lengths(
            target.logits.squeeze_dim::<2>(0),
            &history_rows,
            &vec![cycle.prompt_length; drafted + 1],
        )?;
        // Greedy choices for every position come back in one readback.
        let greedy_choices = match self.config.generation.strategy {
            SamplingStrategy::Greedy => Some(read_tokens(self.target.select(processed.clone())?)?),
            SamplingStrategy::Sample => None,
        };
        let row = |index: usize| processed.clone().narrow(0, index, 1);

        let mut emitted = vec![cycle.lead_token];
        let mut outcome = SpeculativeOutcome::AllAccepted;
        for (index, &proposal) in drafts.tokens.iter().enumerate() {
            let verdict = match &greedy_choices {
                Some(choices) => Verdict::greedy(proposal, choices[index]),
                None => {
                    let target = normalized_probabilities(row(index))?;
                    let draft = drafts.distributions.get(index).ok_or_else(|| {
                        Error::Generation(
                            "Qwen stochastic verification lost its draft distribution".into(),
                        )
                    })?;
                    self.verifier.verify(proposal, &target, draft)?
                }
            };
            match verdict {
                Verdict::Accepted => {
                    emitted.push(proposal);
                    if self.is_eos(proposal) {
                        outcome = SpeculativeOutcome::AcceptedEndOfSequence {
                            accepted: index + 1,
                        };
                        break;
                    }
                }
                Verdict::Rejected { correction } => {
                    emitted.push(correction);
                    outcome = SpeculativeOutcome::Rejected { accepted: index };
                    break;
                }
            }
        }
        if outcome == SpeculativeOutcome::AllAccepted {
            let bonus = match &greedy_choices {
                Some(choices) => choices[drafted],
                None => single_token(self.target.select(row(drafted))?)?,
            };
            emitted.push(bonus);
        }
        let commit = speculative_commit(drafted, outcome).ok_or_else(|| {
            Error::Generation("speculative outcome is inconsistent with the drafts".into())
        })?;
        if emitted.len() != commit.emitted_tokens {
            return Err(Error::Generation(format!(
                "speculative cycle emitted {} tokens, expected {}",
                emitted.len(),
                commit.emitted_tokens
            )));
        }
        // Re-run the MTP layer over the committed pairs (x_(L+j), h_(L-1+j)),
        // whose hidden states all come from the target.
        let accepted_pairs = commit.draft_cache_tokens;
        let pair_hidden = if accepted_pairs == 1 {
            cycle.target_hidden_before_lead.clone()
        } else {
            Tensor::cat(
                vec![
                    cycle.target_hidden_before_lead.clone(),
                    target_hidden.clone().narrow(1, 0, accepted_pairs - 1),
                ],
                1,
            )
        };
        model.forward_mtp_cached(
            token_tensor(&verified[..accepted_pairs], &device),
            pair_hidden,
            None,
            0,
            state,
        )?;
        let terminated = emitted.last().is_some_and(|&token| self.is_eos(token));
        let accepted_draft_tokens = match outcome {
            SpeculativeOutcome::Rejected { accepted }
            | SpeculativeOutcome::AcceptedEndOfSequence { accepted } => accepted,
            SpeculativeOutcome::AllAccepted => drafted,
        };
        let output = Qwen3_5SpeculativeOutput {
            token_ids: emitted,
            drafted_tokens: drafted,
            accepted_draft_tokens,
            all_drafts_accepted: outcome == SpeculativeOutcome::AllAccepted,
            terminated,
            // The last committed target position precedes the last emitted token.
            continuation_hidden_state: (!terminated)
                .then(|| target_hidden.narrow(1, commit.target_cache_tokens - 1, 1)),
        };
        Ok((output, commit))
    }

    /// Checks batch, cache alignment, shape, vocabulary, history, and remaining capacity.
    fn validate_cycle(
        &self,
        model: &Qwen3_5ForCausalLm<B>,
        lead_token: u32,
        hidden: &Tensor<B, 3>,
        history_before_lead: &[u32],
        prompt_length: usize,
        state: &Qwen3_5DecodeState<B>,
    ) -> Result<()> {
        state.validate_for_model(model)?;
        if state.tensor_state().has_active_transaction() {
            return Err(crate::cache::TensorCacheError::TransactionAlreadyActive.into());
        }
        if self.vocabulary_size != model.config().vocab_size {
            return Err(Error::Generation(format!(
                "speculative decoder vocabulary {} does not match model vocabulary {}",
                self.vocabulary_size,
                model.config().vocab_size
            )));
        }
        if &hidden.device() != state.tensor_state().device() {
            return Err(Error::BackendCapability(
                "Qwen speculative hidden state must use the decode-state device".into(),
            ));
        }
        check_prompt_length(history_before_lead, prompt_length)?;
        if history_before_lead
            .iter()
            .any(|&token| token as usize >= model.config().vocab_size)
        {
            return Err(Error::shape(
                "qwen.speculative.history",
                format!("token values in 0..{}", model.config().vocab_size),
                "an out-of-vocabulary token",
            ));
        }
        if lead_token as usize >= model.config().vocab_size {
            return Err(Error::shape(
                "qwen.mtp.lead_token",
                format!("a token in 0..{}", model.config().vocab_size),
                lead_token.to_string(),
            ));
        }
        if hidden.dims() != [1, 1, model.config().hidden_size] {
            return Err(Error::shape(
                "qwen.mtp.target_hidden_before_lead",
                format!("[1, 1, {}]", model.config().hidden_size),
                format!("{:?}", hidden.dims()),
            ));
        }
        let required_staging = self.config.max_draft_tokens.checked_add(1).ok_or_else(|| {
            Error::Generation("Qwen MTP proposal capacity overflows usize".into())
        })?;
        if state.staging_capacity() < required_staging {
            return Err(Error::Generation(format!(
                "decode state staging capacity {} is below required {}",
                state.staging_capacity(),
                required_staging
            )));
        }
        let target = state.tensor_state().lengths();
        let mtp = state.tensor_state().mtp_lengths().ok_or_else(|| {
            Error::Cache("Qwen speculative state has no independent MTP cursor".into())
        })?;
        if target.len() != 1 || mtp.len() != 1 || mtp[0].checked_add(1) != Some(target[0]) {
            return Err(Error::Cache(format!(
                "Qwen speculative cache alignment requires target = MTP + 1, found target={target:?}, MTP={mtp:?}"
            )));
        }
        if history_before_lead.len() != target[0] {
            return Err(Error::shape(
                "qwen.speculative.history",
                format!("{} tokens matching the target-cache cursor", target[0]),
                format!("{} tokens", history_before_lead.len()),
            ));
        }
        let requested = target[0]
            .checked_add(required_staging)
            .ok_or_else(|| Error::Generation("target speculative length overflows usize".into()))?;
        if requested > state.tensor_state().capacity() {
            return Err(Error::Generation(format!(
                "speculative target suffix reaches {requested}, exceeding cache capacity {}",
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

/// Inputs shared by the drafting and verification phases of one cycle.
struct Cycle<'a, B: Backend> {
    /// Target-selected first token of the cycle, `y_0 = x_L`.
    lead_token: u32,
    /// Target hidden state `h_(L-1)` that selected the lead, `[1, 1, hidden]`.
    target_hidden_before_lead: Tensor<B, 3>,
    /// Prompt plus generated tokens before the lead (`L` tokens).
    history_before_lead: &'a [u32],
    /// Number of leading history tokens that are the prompt.
    prompt_length: usize,
}

/// Proposals of one drafting phase.
struct Drafts {
    /// Proposed tokens `d_1..d_K'`, `K' <= K` (an EOS proposal ends drafting).
    tokens: Vec<u32>,
    /// Host draft distributions `q_i` (stochastic decoding only, else empty).
    distributions: Vec<Vec<f64>>,
}

/// Rejects a prompt length beyond the supplied history.
fn check_prompt_length(history: &[u32], prompt_length: usize) -> Result<()> {
    if prompt_length > history.len() {
        return Err(Error::shape(
            "qwen.speculative.prompt_length",
            format!("at most the {} history tokens", history.len()),
            prompt_length.to_string(),
        ));
    }
    Ok(())
}

/// Creates a `[1, n]` token tensor after portable integer conversion.
fn token_tensor<B: Backend>(tokens: &[u32], device: &B::Device) -> Tensor<B, 2, Int> {
    let values = tokens.iter().copied().map(i64::from).collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [1, tokens.len()]), device)
}

/// Reads selected backend tokens with checked unsigned conversion.
fn read_tokens<B: Backend>(tokens: Tensor<B, 1, Int>) -> Result<Vec<u32>> {
    tokens
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to read selected tokens from the backend: {error}"
            ))
        })?
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to read selected tokens: {error}"))
        })?
        .into_iter()
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                Error::Generation(format!("token selector returned invalid ID {value}"))
            })
        })
        .collect()
}

/// Reads exactly one selected backend token.
fn single_token<B: Backend>(token: Tensor<B, 1, Int>) -> Result<u32> {
    match read_tokens(token)?.as_slice() {
        [token] => Ok(*token),
        tokens => Err(Error::Generation(format!(
            "token selector returned {} tokens for one row",
            tokens.len()
        ))),
    }
}

#[cfg(test)]
#[path = "speculative_tests.rs"]
mod tests;
