//! Batch-one lossless speculation with Nemotron-H Lightning's MTP stage.
//!
//! Let `y_0` be a token already selected by the target and let `h` be the
//! target state immediately before `y_0`. The auxiliary stage repeatedly maps
//! `(y_i, h_i)` to a draft distribution `q_i` and feedback state `h_(i+1)`.
//! The target consumes `y_0` and verifies candidates under distributions `p_i`.
//! Greedy verification accepts only exact target argmaxes. Stochastic
//! verification accepts with `min(1, p_i(y_i) / q_i(y_i))`; a rejection draws
//! from normalized `(p_i - q_i)_+`, preserving the target marginal exactly.
//!
//! Model work occurs inside a target-plus-MTP cache transaction. Verification
//! publishes only the accepted physical prefix; errors restore model cursors,
//! KV/Mamba tensors, and all pseudo-random streams to the pre-cycle state.
//! Like vLLM's model-side speculator, this object executes one cycle rather
//! than owning a complete generation loop: the caller enforces the global
//! `max_new_tokens` budget, stop-sequence matching, and padding. EOS tokens are
//! handled within a cycle because they determine the publishable cache prefix.

use burn::tensor::{Int, Tensor, TensorData, activation::softmax, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::{
    CausalLanguageModel, DecodeInput, Error, ForwardOptions, GenerationConfig, LogitsProcessor,
    LogitsSelection, Result, SamplingStrategy,
};

use super::{NemotronHDecodeState, NemotronHForCausalLm};

const DRAFT_SEED_DOMAIN: u64 = 0x4452_4146_545f_4e48;
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
    config: NemotronHSpeculativeConfig,
    vocabulary_size: usize,
    target: LogitsProcessor<B>,
    draft: LogitsProcessor<B>,
    verifier: SplitMix64,
}

impl<B: Backend> NemotronHSpeculativeDecoder<B> {
    /// Creates a lossless decoder for one fixed vocabulary and device.
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
        let verifier = SplitMix64::new(config.generation.seed ^ VERIFY_SEED_DOMAIN);
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

    /// Selects the guaranteed lead token from one target-logit row.
    pub fn select_lead(&self, target_logits: Tensor<B, 2>, history: &[u32]) -> Result<u32> {
        let history = history_tensor(history, &target_logits.device())?;
        selected_token(self.target.process_and_select(target_logits, history)?)
    }

    /// Runs one active speculative cycle.
    ///
    /// The caller remains responsible for global token budgets and stop
    /// sequences; see the module-level scheduler boundary.
    pub fn speculate(
        &mut self,
        model: &NemotronHForCausalLm<B>,
        lead_token: u32,
        target_hidden_before_lead: Tensor<B, 3>,
        history_before_lead: &[u32],
        state: &mut NemotronHDecodeState<B>,
    ) -> Result<NemotronHSpeculativeOutput<B>> {
        self.speculate_if_active(
            model,
            lead_token,
            target_hidden_before_lead,
            history_before_lead,
            true,
            state,
        )
    }

    /// Runs a cycle only when its scheduler row remains active.
    ///
    /// An inactive cycle performs no model work, draws no randomness, and does
    /// not open a cache transaction.
    #[allow(clippy::too_many_arguments)]
    pub fn speculate_if_active(
        &mut self,
        model: &NemotronHForCausalLm<B>,
        lead_token: u32,
        target_hidden_before_lead: Tensor<B, 3>,
        history_before_lead: &[u32],
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
        match self.run_transaction(
            model,
            lead_token,
            target_hidden_before_lead,
            history_before_lead,
            state,
        ) {
            Ok(pending) => {
                let output = NemotronHSpeculativeOutput {
                    token_ids: pending.token_ids,
                    drafted_tokens: pending.drafted_tokens,
                    accepted_draft_tokens: pending.accepted_draft_tokens,
                    all_drafts_accepted: pending.all_drafts_accepted,
                    terminated: pending.terminated,
                    continuation_hidden_state: pending.continuation_hidden_state,
                };
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
                Ok(output)
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

    /// Executes all model work while both physical banks remain staged.
    fn run_transaction(
        &mut self,
        model: &NemotronHForCausalLm<B>,
        lead_token: u32,
        target_hidden_before_lead: Tensor<B, 3>,
        history_before_lead: &[u32],
        state: &mut NemotronHDecodeState<B>,
    ) -> Result<PendingCycle<B>> {
        let device = target_hidden_before_lead.device();
        let mut generated_history = history_before_lead.to_vec();
        generated_history.push(lead_token);
        let mut proposals = Vec::with_capacity(self.config.max_draft_tokens);
        let mut draft_probabilities = Vec::with_capacity(self.config.max_draft_tokens);
        let mut feedback_hidden = target_hidden_before_lead;
        let mut draft_input = lead_token;

        for _ in 0..self.config.max_draft_tokens {
            let output = model.forward_mtp_cached(
                token_tensor::<B>(draft_input, &device),
                feedback_hidden,
                None,
                state,
            )?;
            feedback_hidden = output.hidden_state;
            let processed = self.draft.process(
                output.logits.squeeze_dim::<2>(1),
                history_tensor(&generated_history, &device)?,
            )?;
            let probabilities = (self.config.generation.strategy == SamplingStrategy::Sample)
                .then(|| normalized_probabilities(processed.clone()))
                .transpose()?;
            let proposal = selected_token(self.draft.select(processed)?)?;
            proposals.push(proposal);
            draft_probabilities.push(probabilities);
            generated_history.push(proposal);
            draft_input = proposal;
            if self.is_eos(proposal) {
                break;
            }
        }

        let drafted_tokens = proposals.len();
        let mut emitted = vec![lead_token];
        let mut target_history = history_before_lead.to_vec();
        target_history.push(lead_token);
        let mut target_output = decode_one(model, lead_token, state, &device)?;
        let mut target_cache_tokens = 1usize;
        let mut accepted = 0usize;

        for (proposal, q) in proposals.iter().copied().zip(&draft_probabilities) {
            let processed = self
                .target
                .process(target_output.0, history_tensor(&target_history, &device)?)?;
            let correction = match self.config.generation.strategy {
                SamplingStrategy::Greedy => {
                    let target = selected_token(self.target.select(processed)?)?;
                    (target != proposal).then_some(target)
                }
                SamplingStrategy::Sample => {
                    let p = normalized_probabilities(processed)?;
                    let q = q.as_deref().ok_or_else(|| {
                        Error::Generation(
                            "Nemotron-H stochastic verification lost its draft distribution".into(),
                        )
                    })?;
                    (!self.accept(proposal, &p, q)?)
                        .then(|| self.sample_residual(&p, q))
                        .transpose()?
                }
            };
            if let Some(correction) = correction {
                emitted.push(correction);
                let terminated = self.is_eos(correction);
                return Ok(PendingCycle {
                    token_ids: emitted,
                    drafted_tokens,
                    accepted_draft_tokens: accepted,
                    all_drafts_accepted: false,
                    terminated,
                    continuation_hidden_state: (!terminated).then_some(target_output.1),
                    target_cache_tokens,
                    mtp_cache_tokens: (accepted + 1).min(drafted_tokens),
                });
            }

            emitted.push(proposal);
            accepted += 1;
            if self.is_eos(proposal) {
                return Ok(PendingCycle {
                    token_ids: emitted,
                    drafted_tokens,
                    accepted_draft_tokens: accepted,
                    all_drafts_accepted: false,
                    terminated: true,
                    continuation_hidden_state: None,
                    target_cache_tokens,
                    mtp_cache_tokens: accepted.min(drafted_tokens),
                });
            }
            target_history.push(proposal);
            target_output = decode_one(model, proposal, state, &device)?;
            target_cache_tokens += 1;
        }

        // Consume the final accepted token once in the shifted auxiliary stream.
        // The resulting logits are unused; its KV restores target=MTP+1 at the
        // next stable boundary after the unconsumed target bonus token.
        let last = *proposals
            .last()
            .ok_or_else(|| Error::Generation("Nemotron-H MTP produced no proposal".into()))?;
        model.forward_mtp_cached(
            token_tensor::<B>(last, &device),
            feedback_hidden,
            None,
            state,
        )?;
        let processed = self
            .target
            .process(target_output.0, history_tensor(&target_history, &device)?)?;
        let bonus = selected_token(self.target.select(processed)?)?;
        emitted.push(bonus);
        let terminated = self.is_eos(bonus);
        Ok(PendingCycle {
            token_ids: emitted,
            drafted_tokens,
            accepted_draft_tokens: accepted,
            all_drafts_accepted: true,
            terminated,
            continuation_hidden_state: (!terminated).then_some(target_output.1),
            target_cache_tokens,
            mtp_cache_tokens: drafted_tokens + 1,
        })
    }

    /// Checks batch, shift alignment, vocabulary, shape, and free capacity.
    fn validate_cycle(
        &self,
        model: &NemotronHForCausalLm<B>,
        lead_token: u32,
        hidden: &Tensor<B, 3>,
        history_before_lead: &[u32],
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

    /// Applies exact greedy or stochastic acceptance to one proposal.
    fn accept(&mut self, proposal: u32, p: &[f64], q: &[f64]) -> Result<bool> {
        let index = proposal as usize;
        if index >= p.len() || p.len() != q.len() {
            return Err(Error::Generation(
                "Nemotron-H proposal probabilities have incompatible vocabulary shapes".into(),
            ));
        }
        if self.config.generation.strategy == SamplingStrategy::Greedy {
            return Ok(argmax(p)? == proposal);
        }
        let q_value = q[index];
        if !(q_value.is_finite() && q_value > 0.0) {
            return Err(Error::Generation(
                "Nemotron-H draft selected a token with zero or invalid probability".into(),
            ));
        }
        Ok(self.verifier.next_unit() < (p[index] / q_value).min(1.0))
    }

    /// Draws a correction from normalized positive residual `(p-q)_+`.
    fn sample_residual(&mut self, p: &[f64], q: &[f64]) -> Result<u32> {
        if p.len() != q.len() {
            return Err(Error::Generation(
                "Nemotron-H target and draft distributions have different vocabularies".into(),
            ));
        }
        let residual = p
            .iter()
            .zip(q)
            .map(|(&target, &draft)| (target - draft).max(0.0))
            .collect::<Vec<_>>();
        sample_weights(&residual, self.verifier.next_unit())
    }

    /// Returns whether a token terminates ordinary generation.
    fn is_eos(&self, token: u32) -> bool {
        self.config.generation.eos_token_ids.contains(&token)
    }
}

/// Model outputs and cache-prefix lengths waiting for atomic publication.
struct PendingCycle<B: Backend> {
    token_ids: Vec<u32>,
    drafted_tokens: usize,
    accepted_draft_tokens: usize,
    all_drafts_accepted: bool,
    terminated: bool,
    continuation_hidden_state: Option<Tensor<B, 3>>,
    target_cache_tokens: usize,
    mtp_cache_tokens: usize,
}

/// Consumes one target token and retains the hidden state needed by a correction.
fn decode_one<B: Backend>(
    model: &NemotronHForCausalLm<B>,
    token: u32,
    state: &mut NemotronHDecodeState<B>,
    device: &B::Device,
) -> Result<(Tensor<B, 2>, Tensor<B, 3>)> {
    let output = model.decode(
        DecodeInput {
            token_ids: token_tensor(token, device),
            token_mask: None,
            position_ids: None,
        },
        state,
        &ForwardOptions {
            logits: LogitsSelection::Last,
            hidden_states: true,
            ..Default::default()
        },
    )?;
    let hidden = output.final_hidden_state.ok_or_else(|| {
        Error::Generation("Nemotron-H target decode omitted its requested hidden state".into())
    })?;
    Ok((output.logits.squeeze_dim::<2>(1), hidden))
}

/// Creates one checked batch-one token tensor.
fn token_tensor<B: Backend>(token: u32, device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(vec![i64::from(token)], [1, 1]), device)
}

/// Creates optional batch-one history for the public logits processor.
fn history_tensor<B: Backend>(
    history: &[u32],
    device: &B::Device,
) -> Result<Option<Tensor<B, 2, Int>>> {
    if history.is_empty() {
        return Ok(None);
    }
    let values = history.iter().copied().map(i64::from).collect::<Vec<_>>();
    Ok(Some(Tensor::from_data(
        TensorData::new(values, [1, history.len()]),
        device,
    )))
}

/// Reads a selected token with checked signed-to-unsigned conversion.
fn selected_token<B: Backend>(token: Tensor<B, 1, Int>) -> Result<u32> {
    let values = token
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to read Nemotron-H selected token from the backend: {error}"
            ))
        })?
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to decode Nemotron-H token: {error}"))
        })?;
    let value = *values
        .first()
        .ok_or_else(|| Error::Generation("Nemotron-H token selector returned no row".into()))?;
    u32::try_from(value)
        .map_err(|_| Error::Generation(format!("Nemotron-H selected invalid token {value}")))
}

/// Reproduces backend softmax and returns one finite host probability row.
fn normalized_probabilities<B: Backend>(logits: Tensor<B, 2>) -> Result<Vec<f64>> {
    let [batch, vocabulary] = logits.dims();
    if batch != 1 || vocabulary == 0 {
        return Err(Error::shape(
            "nemotron_h.speculative.logits",
            "[1, nonzero_vocabulary]",
            format!("[{batch}, {vocabulary}]"),
        ));
    }
    let values = logits
        .clone()
        .try_into_data()
        .map_err(|error| Error::BackendCapability(error.to_string()))?
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|error| Error::BackendCapability(error.to_string()))?;
    if values.iter().any(|value| value.is_nan()) {
        return Err(Error::Generation(
            "Nemotron-H processed logits contain NaN".into(),
        ));
    }
    if values
        .iter()
        .any(|value| value.is_infinite() && value.is_sign_positive())
    {
        return Err(Error::Generation(
            "Nemotron-H processed logits contain positive infinity".into(),
        ));
    }
    if !values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .is_finite()
    {
        return Err(Error::Generation(
            "Nemotron-H processed logits assign zero mass to every token".into(),
        ));
    }
    let probabilities = softmax(logits, 1)
        .try_into_data()
        .map_err(|error| Error::BackendCapability(error.to_string()))?
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|error| Error::BackendCapability(error.to_string()))?;
    if probabilities
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(Error::Generation(
            "Nemotron-H softmax produced an invalid probability".into(),
        ));
    }
    let mut probabilities = probabilities.into_iter().map(f64::from).collect::<Vec<_>>();
    let total = probabilities.iter().sum::<f64>();
    if !(total.is_finite() && total > 0.0) {
        return Err(Error::Generation(
            "Nemotron-H processed logits cannot be normalized".into(),
        ));
    }
    probabilities.iter_mut().for_each(|value| *value /= total);
    Ok(probabilities)
}

/// Returns the lowest-ID greatest-probability token.
fn argmax(probabilities: &[f64]) -> Result<u32> {
    let (index, _) = probabilities
        .iter()
        .copied()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .ok_or_else(|| Error::Generation("cannot select from an empty distribution".into()))?;
    u32::try_from(index).map_err(|_| Error::Generation("selected token does not fit u32".into()))
}

/// Samples finite non-negative weights using inverse CDF.
fn sample_weights(weights: &[f64], uniform: f64) -> Result<u32> {
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(Error::Generation(
            "Nemotron-H rejection residual contains an invalid weight".into(),
        ));
    }
    let total = weights.iter().sum::<f64>();
    if !(total.is_finite() && total > 0.0) {
        return Err(Error::Generation(
            "Nemotron-H rejection residual has no positive mass".into(),
        ));
    }
    let threshold = uniform * total;
    let mut cumulative = 0.0;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if threshold < cumulative {
            return u32::try_from(index)
                .map_err(|_| Error::Generation("sampled token does not fit u32".into()));
        }
    }
    u32::try_from(weights.len() - 1)
        .map_err(|_| Error::Generation("sampled token does not fit u32".into()))
}

/// Small deterministic stream dedicated to acceptance decisions.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a stream in one architecture-specific seed domain.
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a uniformly distributed IEEE-754 value in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        ((value >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        tensor::{Int, Tensor, TensorData, backend::BackendTypes},
    };

    use crate::{
        CausalLmInput,
        cache::CacheConfig,
        checkpoint::TensorSink,
        models::nemotron_h::{
            NemotronHBlockType, NemotronHCheckpointOptions, NemotronHConfig,
            model::tests::tiny_config,
        },
    };

    use super::*;

    type TestBackend = Flex<f32, i32>;

    /// Uses a cheap base MLP while retaining Lightning's exact MTP stage.
    fn speculative_config() -> NemotronHConfig {
        let mut config = tiny_config(true);
        config.layers_block_type = vec![NemotronHBlockType::Mlp];
        config.num_hidden_layers = Some(1);
        config.tie_word_embeddings = false;
        config
    }

    /// Installs deterministic parameters, optionally adding a two-token target cycle.
    ///
    /// Each parameter is installed exactly once in checkpoint layout. Burn stores
    /// linear weights transposed relative to SafeTensors, so overwriting an already
    /// mapped parameter would make this fixture depend on the sink's runtime-layout
    /// compatibility path rather than exercising the checkpoint mapping itself.
    fn install_parameters(model: &mut NemotronHForCausalLm<TestBackend>, scripted_target: bool) {
        let contract = model
            .config()
            .checkpoint_contract(NemotronHCheckpointOptions::default())
            .unwrap();
        for expected in contract.expected_tensors() {
            let shape = expected
                .shape
                .iter()
                .map(|dimension| usize::try_from(*dimension).unwrap())
                .collect::<Vec<_>>();
            let elements = shape.iter().product();
            let mut values = vec![0.0_f32; elements];
            if expected.name.contains("norm.weight") {
                values.fill(1.0);
            }
            if scripted_target && expected.name == "model.embed_tokens.weight" {
                let hidden = model.config().hidden_size;
                values[0] = 1.0;
                values[hidden + 1] = 1.0;
            }
            if scripted_target && expected.name == "lm_head.weight" {
                let hidden = model.config().hidden_size;
                // SafeTensors layout is [vocabulary, hidden]: hidden coordinate
                // one predicts token zero and coordinate zero predicts token one.
                values[1] = 1.0;
                values[hidden] = 1.0;
            }
            model
                .write_tensor(&expected.name, TensorData::new(values, shape))
                .unwrap();
        }
    }

    /// Creates a graph whose target and auxiliary logits tie at token zero.
    fn zero_model(
        device: &<TestBackend as BackendTypes>::Device,
    ) -> NemotronHForCausalLm<TestBackend> {
        let mut model = NemotronHForCausalLm::init(speculative_config(), device).unwrap();
        install_parameters(&mut model, false);
        model
    }

    /// Creates target transitions `1 -> 0` and `0 -> 1`, while MTP proposes zero.
    fn scripted_rejection_model(
        device: &<TestBackend as BackendTypes>::Device,
    ) -> NemotronHForCausalLm<TestBackend> {
        let mut model = NemotronHForCausalLm::init(speculative_config(), device).unwrap();
        install_parameters(&mut model, true);
        model
    }

    /// Prefills target and shifted MTP streams at the stable one-token offset.
    fn aligned_state(
        model: &NemotronHForCausalLm<TestBackend>,
        device: &<TestBackend as BackendTypes>::Device,
    ) -> (NemotronHDecodeState<TestBackend>, Tensor<TestBackend, 3>) {
        let mut state = model
            .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, device)
            .unwrap();
        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[2, 3]], device);
        let hidden = model
            .prefill(
                CausalLmInput {
                    token_ids: prompt.clone(),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    hidden_states: true,
                    ..Default::default()
                },
            )
            .unwrap()
            .final_hidden_state
            .unwrap();
        model
            .prefill_mtp_cache(prompt, hidden.clone(), &mut state)
            .unwrap();
        (state, hidden.narrow(1, 1, 1))
    }

    /// Default capacity is two and zero-draft configurations fail closed.
    #[test]
    fn config_defaults_and_validation_are_strict() {
        assert_eq!(NemotronHSpeculativeConfig::default().max_draft_tokens, 2);
        assert!(
            NemotronHSpeculativeConfig {
                max_draft_tokens: 0,
                ..Default::default()
            }
            .validate(8)
            .is_err()
        );
        assert!(
            NemotronHSpeculativeConfig {
                max_draft_tokens: usize::MAX,
                ..Default::default()
            }
            .validate(8)
            .is_err()
        );
    }

    /// Probability conversion accepts negative infinity but rejects invalid mass.
    #[test]
    fn probability_validation_rejects_non_finite_distributions() {
        let device = Default::default();
        assert_eq!(
            normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
                [[0.0, 0.0, f32::NEG_INFINITY]],
                &device,
            ))
            .unwrap(),
            [0.5, 0.5, 0.0]
        );
        assert!(
            normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
                [[f32::NAN, 0.0]],
                &device,
            ))
            .is_err()
        );
        assert!(
            normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
                [[f32::INFINITY, 0.0]],
                &device,
            ))
            .is_err()
        );
        assert!(
            normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
                [[f32::NEG_INFINITY, f32::NEG_INFINITY]],
                &device,
            ))
            .is_err()
        );
    }

    /// Greedy verification covers rejected, partial, and fully accepted prefixes.
    #[test]
    fn greedy_acceptance_covers_every_prefix_length() {
        let device = Default::default();
        for accepted_prefix in 0..=2 {
            let mut verifier = NemotronHSpeculativeDecoder::<TestBackend>::new(
                NemotronHSpeculativeConfig::default(),
                3,
                &device,
            )
            .unwrap();
            for index in 0..2 {
                let proposal = 1;
                let p = if index < accepted_prefix {
                    [0.1, 0.8, 0.1]
                } else {
                    [0.8, 0.1, 0.1]
                };
                assert_eq!(
                    verifier.accept(proposal, &p, &p).unwrap(),
                    index < accepted_prefix
                );
                if index == accepted_prefix {
                    break;
                }
            }
        }
    }

    /// Inverse-CDF sampling never leaves positive residual support.
    #[test]
    fn residual_sampling_respects_support_and_validates_weights() {
        for uniform in [0.0, 0.1, 0.499, 0.999_999] {
            let token = sample_weights(&[0.0, 0.4, 0.0, 0.6], uniform).unwrap();
            assert!(token == 1 || token == 3);
        }
        assert!(sample_weights(&[0.0, 0.0], 0.5).is_err());
        assert!(sample_weights(&[f64::NAN, 1.0], 0.5).is_err());
        assert!(sample_weights(&[1.0, -0.1], 0.0).is_err());
    }

    /// The verifier stream is reproducible and strictly inside the unit interval.
    #[test]
    fn verifier_rng_is_reproducible() {
        let mut left = SplitMix64::new(19);
        let mut right = SplitMix64::new(19);
        for _ in 0..128 {
            let value = left.next_unit();
            assert_eq!(value, right.next_unit());
            assert!((0.0..1.0).contains(&value));
        }
    }

    /// Fully accepted drafts commit catch-up and preserve target=MTP+1.
    #[test]
    fn all_accepted_cycle_commits_catch_up() {
        let device = Default::default();
        let model = zero_model(&device);
        let (mut state, hidden) = aligned_state(&model, &device);
        let mut decoder = NemotronHSpeculativeDecoder::new(
            NemotronHSpeculativeConfig::default(),
            model.config().vocab_size,
            &device,
        )
        .unwrap();
        let output = decoder
            .speculate(&model, 1, hidden, &[2, 3], &mut state)
            .unwrap();
        assert_eq!(output.token_ids, [1, 0, 0, 0]);
        assert_eq!(output.drafted_tokens, 2);
        assert_eq!(output.accepted_draft_tokens, 2);
        assert!(output.all_drafts_accepted);
        assert!(!output.terminated);
        assert!(output.continuation_hidden_state.is_some());
        assert_eq!(state.tensor_state().lengths(), &[5]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([4].as_slice()));
    }

    /// Real model cycles commit zero, one, and two accepted draft prefixes.
    #[test]
    fn complete_cycles_cover_every_acceptance_prefix() {
        let device = Default::default();
        let model = scripted_rejection_model(&device);
        for (lead, accepted, emitted) in [(0, 0, vec![0, 1]), (1, 1, vec![1, 0, 1])] {
            let (mut state, hidden) = aligned_state(&model, &device);
            let mut decoder = NemotronHSpeculativeDecoder::new(
                NemotronHSpeculativeConfig::default(),
                model.config().vocab_size,
                &device,
            )
            .unwrap();
            let output = decoder
                .speculate(&model, lead, hidden, &[2, 3], &mut state)
                .unwrap();
            assert_eq!(output.token_ids, emitted);
            assert_eq!(output.accepted_draft_tokens, accepted);
            assert!(!output.all_drafts_accepted);
            assert_eq!(state.tensor_state().lengths(), &[3 + accepted]);
            assert_eq!(
                state.tensor_state().mtp_lengths(),
                Some([2 + accepted].as_slice())
            );
        }
    }

    /// EOS at the lead or either draft boundary terminates without cursor drift.
    #[test]
    fn eos_placement_preserves_shifted_cache_alignment() {
        let device = Default::default();
        let model = zero_model(&device);
        let (mut state, hidden) = aligned_state(&model, &device);
        let mut lead_decoder = NemotronHSpeculativeDecoder::new(
            NemotronHSpeculativeConfig {
                generation: GenerationConfig {
                    eos_token_ids: vec![1],
                    ..Default::default()
                },
                ..Default::default()
            },
            model.config().vocab_size,
            &device,
        )
        .unwrap();
        let lead = lead_decoder
            .speculate(&model, 1, hidden.clone(), &[2, 3], &mut state)
            .unwrap();
        assert_eq!(lead.token_ids, [1]);
        assert!(lead.terminated);
        assert_eq!(state.tensor_state().lengths(), &[2]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));

        let mut first_decoder = NemotronHSpeculativeDecoder::new(
            NemotronHSpeculativeConfig {
                generation: GenerationConfig {
                    eos_token_ids: vec![0],
                    ..Default::default()
                },
                ..Default::default()
            },
            model.config().vocab_size,
            &device,
        )
        .unwrap();
        let first = first_decoder
            .speculate(&model, 1, hidden, &[2, 3], &mut state)
            .unwrap();
        assert_eq!(first.token_ids, [1, 0]);
        assert_eq!(first.accepted_draft_tokens, 1);
        assert!(first.terminated);
        assert_eq!(state.tensor_state().lengths(), &[3]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));

        let (mut state, hidden) = aligned_state(&model, &device);
        let mut second_decoder = NemotronHSpeculativeDecoder::new(
            NemotronHSpeculativeConfig {
                generation: GenerationConfig {
                    presence_penalty: 1.0,
                    eos_token_ids: vec![4],
                    ..Default::default()
                },
                ..Default::default()
            },
            model.config().vocab_size,
            &device,
        )
        .unwrap();
        let second = second_decoder
            .speculate(&model, 1, hidden, &[2, 3], &mut state)
            .unwrap();
        assert_eq!(second.token_ids, [1, 0, 4]);
        assert_eq!(second.accepted_draft_tokens, 2);
        assert!(second.terminated);
        assert_eq!(state.tensor_state().lengths(), &[4]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([3].as_slice()));
    }

    /// EOS cannot bypass the ownership check for a caller-open transaction.
    #[test]
    fn eos_lead_rejects_a_caller_owned_active_transaction() {
        let device = Default::default();
        let model = zero_model(&device);
        let (mut state, hidden) = aligned_state(&model, &device);
        let mark = state.mark().unwrap();
        let mut decoder = NemotronHSpeculativeDecoder::new(
            NemotronHSpeculativeConfig {
                generation: GenerationConfig {
                    eos_token_ids: vec![1],
                    ..Default::default()
                },
                ..Default::default()
            },
            model.config().vocab_size,
            &device,
        )
        .unwrap();
        assert!(matches!(
            decoder.speculate(&model, 1, hidden, &[2, 3], &mut state),
            Err(Error::TensorCache(
                crate::cache::TensorCacheError::TransactionAlreadyActive
            ))
        ));
        assert!(state.tensor_state().has_active_transaction());
        state.rollback(mark).unwrap();
    }

    /// Inactive rows and post-stage failures are exact retryable no-ops.
    #[test]
    fn inactive_and_recoverable_failure_restore_state_and_rng() {
        let device = Default::default();
        let model = zero_model(&device);
        let (mut state, hidden) = aligned_state(&model, &device);
        let mut decoder = NemotronHSpeculativeDecoder::new(
            NemotronHSpeculativeConfig::default(),
            model.config().vocab_size,
            &device,
        )
        .unwrap();
        let inactive = decoder
            .speculate_if_active(&model, 1, hidden.clone(), &[2, 3], false, &mut state)
            .unwrap();
        assert!(inactive.token_ids.is_empty());
        assert_eq!(state.tensor_state().lengths(), &[2]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));

        decoder.draft = LogitsProcessor::new(
            decoder.config.generation.clone(),
            model.config().vocab_size - 1,
            &device,
        )
        .unwrap();
        let first = decoder
            .speculate(&model, 1, hidden.clone(), &[2, 3], &mut state)
            .unwrap_err();
        assert!(matches!(first, Error::Shape { .. }));
        assert_eq!(state.tensor_state().lengths(), &[2]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));
        let second = decoder
            .speculate(&model, 1, hidden, &[2, 3], &mut state)
            .unwrap_err();
        assert_eq!(first.to_string(), second.to_string());
    }

    /// Cycle preflight rejects vocabulary, history, token, hidden, and capacity drift.
    #[test]
    fn cycle_preflight_rejects_all_alignment_mismatches() {
        let device = Default::default();
        let model = zero_model(&device);
        let (mut state, hidden) = aligned_state(&model, &device);
        let mut wrong_vocabulary = NemotronHSpeculativeDecoder::new(
            NemotronHSpeculativeConfig::default(),
            model.config().vocab_size - 1,
            &device,
        )
        .unwrap();
        assert!(
            wrong_vocabulary
                .speculate(&model, 1, hidden.clone(), &[2, 3], &mut state)
                .is_err()
        );

        let mut decoder = NemotronHSpeculativeDecoder::new(
            NemotronHSpeculativeConfig::default(),
            model.config().vocab_size,
            &device,
        )
        .unwrap();
        assert!(
            decoder
                .speculate(&model, 1, hidden.clone(), &[2], &mut state)
                .is_err()
        );
        assert!(
            decoder
                .speculate(
                    &model,
                    model.config().vocab_size as u32,
                    hidden.clone(),
                    &[2, 3],
                    &mut state,
                )
                .is_err()
        );
        assert!(
            decoder
                .speculate(
                    &model,
                    1,
                    Tensor::zeros([1, 2, model.config().hidden_size], &device),
                    &[2, 3],
                    &mut state,
                )
                .is_err()
        );
        assert_eq!(state.tensor_state().lengths(), &[2]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));
    }

    /// A physically rewindable graph clips MTP to the surviving shifted prefix.
    #[test]
    fn trait_truncate_rewinds_target_and_mtp_to_independent_lengths() {
        let device = Default::default();
        let model = zero_model(&device);
        let (mut state, _) = aligned_state(&model, &device);
        crate::architecture::DecodeState::<TestBackend>::truncate(&mut state, &[1]).unwrap();
        assert_eq!(state.tensor_state().lengths(), &[1]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([0].as_slice()));
    }

    /// Monte Carlo rejection sampling recovers the target distribution.
    #[test]
    fn stochastic_rejection_recovers_target_marginal() {
        let p = [0.1, 0.2, 0.7];
        let q = [0.6, 0.3, 0.1];
        let mut proposal_rng = SplitMix64::new(71);
        let mut verifier_rng = SplitMix64::new(72);
        let mut counts = [0usize; 3];
        let trials = 100_000usize;
        for _ in 0..trials {
            let proposal = sample_weights(&q, proposal_rng.next_unit()).unwrap();
            let ratio = (p[proposal as usize] / q[proposal as usize]).min(1.0);
            let token = if verifier_rng.next_unit() < ratio {
                proposal
            } else {
                let residual = p
                    .iter()
                    .zip(q)
                    .map(|(&target, draft)| (target - draft).max(0.0))
                    .collect::<Vec<_>>();
                sample_weights(&residual, verifier_rng.next_unit()).unwrap()
            };
            counts[token as usize] += 1;
        }
        for (observed, expected) in counts.into_iter().zip(p) {
            assert!((observed as f64 / trials as f64 - expected).abs() < 0.01);
        }
    }

    /// Equal seeds produce equal sampled cycles and cache cursors.
    #[test]
    fn sampled_cycles_are_reproducible() {
        let device = Default::default();
        let model = zero_model(&device);
        let (mut left_state, left_hidden) = aligned_state(&model, &device);
        let (mut right_state, right_hidden) = aligned_state(&model, &device);
        let config = NemotronHSpeculativeConfig {
            generation: GenerationConfig {
                strategy: SamplingStrategy::Sample,
                seed: 0x5eed,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut left =
            NemotronHSpeculativeDecoder::new(config.clone(), model.config().vocab_size, &device)
                .unwrap();
        let mut right =
            NemotronHSpeculativeDecoder::new(config, model.config().vocab_size, &device).unwrap();
        let left = left
            .speculate(&model, 1, left_hidden, &[2, 3], &mut left_state)
            .unwrap();
        let right = right
            .speculate(&model, 1, right_hidden, &[2, 3], &mut right_state)
            .unwrap();
        assert_eq!(left.token_ids, right.token_ids);
        assert_eq!(left.accepted_draft_tokens, right.accepted_draft_tokens);
        assert_eq!(
            left_state.tensor_state().lengths(),
            right_state.tensor_state().lengths()
        );
        assert_eq!(
            left_state.tensor_state().mtp_lengths(),
            right_state.tensor_state().mtp_lengths()
        );
    }

    /// A failure after draft sampling restores the proposal stream for retry.
    #[test]
    fn sampled_failure_after_rng_draw_retries_bit_for_bit() {
        let device = Default::default();
        let model = zero_model(&device);
        let config = NemotronHSpeculativeConfig {
            generation: GenerationConfig {
                strategy: SamplingStrategy::Sample,
                seed: 0x5eed_cafe,
                ..Default::default()
            },
            ..Default::default()
        };

        let (mut expected_state, expected_hidden) = aligned_state(&model, &device);
        let mut expected_decoder =
            NemotronHSpeculativeDecoder::new(config.clone(), model.config().vocab_size, &device)
                .unwrap();
        let expected = expected_decoder
            .speculate(&model, 1, expected_hidden, &[2, 3], &mut expected_state)
            .unwrap();

        let (mut retry_state, retry_hidden) = aligned_state(&model, &device);
        let mut retry_decoder =
            NemotronHSpeculativeDecoder::new(config.clone(), model.config().vocab_size, &device)
                .unwrap();
        retry_decoder.target = LogitsProcessor::new(
            config.generation.clone(),
            model.config().vocab_size - 1,
            &device,
        )
        .unwrap();
        assert!(matches!(
            retry_decoder.speculate(&model, 1, retry_hidden.clone(), &[2, 3], &mut retry_state,),
            Err(Error::Shape { .. })
        ));
        assert_eq!(retry_state.tensor_state().lengths(), &[2]);
        assert_eq!(
            retry_state.tensor_state().mtp_lengths(),
            Some([1].as_slice())
        );

        retry_decoder.target =
            LogitsProcessor::new(config.generation, model.config().vocab_size, &device).unwrap();
        let retried = retry_decoder
            .speculate(&model, 1, retry_hidden, &[2, 3], &mut retry_state)
            .unwrap();
        assert_eq!(retried.token_ids, expected.token_ids);
        assert_eq!(
            retried.accepted_draft_tokens,
            expected.accepted_draft_tokens
        );
        assert_eq!(retried.all_drafts_accepted, expected.all_drafts_accepted);
        assert_eq!(
            retry_state.tensor_state().lengths(),
            expected_state.tensor_state().lengths()
        );
        assert_eq!(
            retry_state.tensor_state().mtp_lengths(),
            expected_state.tensor_state().mtp_lengths()
        );
    }
}
