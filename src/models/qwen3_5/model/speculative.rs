//! Batch-one lossless speculative decoding with Qwen's MTP head.
//!
//! A cycle starts with a target-selected lead token `y_0` and the target hidden
//! state immediately before it. The MTP graph proposes `K` further tokens from
//! distributions `q_i`. The target consumes `y_0`, verifies each proposal under
//! `p_i`, and either emits a correction or (when all proposals are accepted) one
//! bonus target token. Greedy acceptance is exact equality with `argmax p_i`.
//! Stochastic acceptance uses
//! `a_i = min(1, p_i(y_i) / q_i(y_i))`; rejection samples from the normalized
//! residual `(p_i - q_i)_+`. This is the standard lossless rejection rule, so
//! marginal output remains exactly the target distribution.

use burn::tensor::{Int, Tensor, TensorData, activation::softmax, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::{
    CausalLanguageModel, DecodeInput, Error, ForwardOptions, GenerationConfig, LogitsProcessor,
    LogitsSelection, Result, SamplingStrategy,
};

use super::{Qwen3_5DecodeState, Qwen3_5ForCausalLm};

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
    /// Tokens to emit, beginning with the guaranteed target lead token.
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
/// Each logits processor owns an independent deterministic random stream. A
/// third SplitMix64 stream drives accept/reject uniforms and residual sampling;
/// unrelated model instances therefore cannot perturb reproducibility.
#[derive(Debug)]
pub struct Qwen3_5SpeculativeDecoder<B: Backend> {
    /// Validated proposal-count and ordinary generation controls.
    config: Qwen3_5SpeculativeConfig,
    /// Expected probability-vector width and exclusive token-ID bound.
    vocabulary_size: usize,
    /// Target penalties, truncation, and independent lead/bonus sampling stream.
    target: LogitsProcessor<B>,
    /// Draft penalties, truncation, and independently seeded proposal stream.
    draft: LogitsProcessor<B>,
    /// Independent uniforms used for acceptance and positive-residual sampling.
    verifier: SplitMix64,
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
        let verifier = SplitMix64::new(config.generation.seed ^ VERIFY_SEED_DOMAIN);
        Ok(Self {
            config,
            vocabulary_size,
            target,
            draft,
            verifier,
        })
    }

    /// Returns immutable speculative and ordinary generation controls.
    #[must_use]
    pub const fn config(&self) -> &Qwen3_5SpeculativeConfig {
        &self.config
    }

    /// Selects the guaranteed lead token from target logits.
    ///
    /// This convenience method uses the same target processor stream later used
    /// for all-accepted bonus tokens. The returned token is the `lead_token`
    /// argument for [`speculate`](Self::speculate).
    pub fn select_lead(&self, target_logits: Tensor<B, 2>, history: &[u32]) -> Result<u32> {
        let history = history_tensor(history, &target_logits.device())?;
        let token = self
            .target
            .process_and_select(target_logits, history)?
            .try_into_data()
            .map_err(|error| {
                Error::BackendCapability(format!(
                    "failed to read lead token from the backend: {error}"
                ))
            })?
            .convert::<i64>()
            .into_vec::<i64>()
            .map_err(|error| {
                Error::BackendCapability(format!("failed to read lead token: {error}"))
            })?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Generation("target processor returned no lead token".into()))?;
        u32::try_from(token)
            .map_err(|_| Error::Generation(format!("selected invalid lead token {token}")))
    }

    /// Runs one active speculative cycle.
    pub fn speculate(
        &mut self,
        model: &Qwen3_5ForCausalLm<B>,
        lead_token: u32,
        target_hidden_before_lead: Tensor<B, 3>,
        history_before_lead: &[u32],
        state: &mut Qwen3_5DecodeState<B>,
    ) -> Result<Qwen3_5SpeculativeOutput<B>> {
        self.speculate_if_active(
            model,
            lead_token,
            target_hidden_before_lead,
            history_before_lead,
            true,
            state,
        )
    }

    /// Runs a cycle only when a scheduler row remains active.
    ///
    /// `active = false` is a strict no-op: it draws no randomness, performs no
    /// model computation, and leaves both caches byte-for-byte unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn speculate_if_active(
        &mut self,
        model: &Qwen3_5ForCausalLm<B>,
        lead_token: u32,
        target_hidden_before_lead: Tensor<B, 3>,
        history_before_lead: &[u32],
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
                let result = Qwen3_5SpeculativeOutput {
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
                            "Qwen speculative commit failed ({error}); rollback also failed ({rollback})"
                        ))
                    })?;
                    return Err(error);
                }
                Ok(result)
            }
            Err(error) => {
                self.target = target_snapshot;
                self.draft = draft_snapshot;
                self.verifier = verifier_snapshot;
                state.rollback(mark).map_err(|rollback| {
                    Error::Cache(format!(
                        "Qwen speculative step failed ({error}); rollback also failed ({rollback})"
                    ))
                })?;
                Err(error)
            }
        }
    }

    /// Executes model work inside an already-open transaction.
    fn run_transaction(
        &mut self,
        model: &Qwen3_5ForCausalLm<B>,
        lead_token: u32,
        target_hidden_before_lead: Tensor<B, 3>,
        history_before_lead: &[u32],
        state: &mut Qwen3_5DecodeState<B>,
    ) -> Result<PendingCycle<B>> {
        let device = target_hidden_before_lead.device();
        let mut generated_history = history_before_lead.to_vec();
        generated_history.push(lead_token);
        let mut proposals = Vec::with_capacity(self.config.max_draft_tokens);
        let mut draft_probabilities = Vec::with_capacity(self.config.max_draft_tokens);
        let mut draft_hidden = target_hidden_before_lead;
        let mut draft_input = lead_token;

        for step in 0..self.config.max_draft_tokens {
            let output = model.forward_mtp_cached(
                token_tensor::<B>(draft_input, &device),
                draft_hidden,
                None,
                step,
                state,
            )?;
            draft_hidden = output.hidden_state;
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
                            "Qwen stochastic verification lost its draft distribution".into(),
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

        // Every draft was accepted. Consume the last accepted token once more
        // in the shifted MTP stream so its cache remains exactly one position
        // behind the target at the next cycle boundary. Its logits are unused.
        let last = *proposals
            .last()
            .ok_or_else(|| Error::Generation("MTP produced no proposal".into()))?;
        model.forward_mtp_cached(
            token_tensor::<B>(last, &device),
            draft_hidden,
            None,
            drafted_tokens,
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

    /// Checks batch, cache alignment, shape, vocabulary, and remaining capacity.
    fn validate_cycle(
        &self,
        model: &Qwen3_5ForCausalLm<B>,
        lead_token: u32,
        hidden: &Tensor<B, 3>,
        history_before_lead: &[u32],
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

    /// Applies exact greedy or stochastic acceptance to one proposal.
    fn accept(&mut self, proposal: u32, p: &[f64], q: &[f64]) -> Result<bool> {
        let index = proposal as usize;
        if index >= p.len() || p.len() != q.len() {
            return Err(Error::Generation(
                "proposal probability vectors have incompatible vocabulary shapes".into(),
            ));
        }
        if self.config.generation.strategy == SamplingStrategy::Greedy {
            return Ok(argmax(p)? == proposal);
        }
        let q_value = q[index];
        if !(q_value.is_finite() && q_value > 0.0) {
            return Err(Error::Generation(
                "draft selected a token with zero or invalid probability".into(),
            ));
        }
        Ok(self.verifier.next_unit() < (p[index] / q_value).min(1.0))
    }

    /// Samples the normalized positive residual `(p-q)_+` after rejection.
    fn sample_residual(&mut self, p: &[f64], q: &[f64]) -> Result<u32> {
        if p.len() != q.len() {
            return Err(Error::Generation(
                "target and draft distributions have different vocabularies".into(),
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

/// Target tensors and cache-prefix lengths awaiting atomic publication.
struct PendingCycle<B: Backend> {
    /// Suffix awaiting emission, including the guaranteed target lead token.
    token_ids: Vec<u32>,
    /// Number of MTP proposals produced before verification.
    drafted_tokens: usize,
    /// Number of proposals accepted by the target, including an accepted EOS.
    accepted_draft_tokens: usize,
    /// Whether verification accepted every proposal and selected a bonus token.
    all_drafts_accepted: bool,
    /// Whether the emitted suffix ends in a configured EOS token.
    terminated: bool,
    /// Target state immediately before the last emitted token, absent after EOS.
    continuation_hidden_state: Option<Tensor<B, 3>>,
    /// Number of staged target tokens to retain relative to the transaction mark.
    target_cache_tokens: usize,
    /// Number of staged MTP tokens to retain relative to the transaction mark.
    mtp_cache_tokens: usize,
}

/// Runs one target token and returns `[1,V]` logits plus its normalized hidden state.
fn decode_one<B: Backend>(
    model: &Qwen3_5ForCausalLm<B>,
    token: u32,
    state: &mut Qwen3_5DecodeState<B>,
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
        Error::Generation("target decode omitted requested final hidden state".into())
    })?;
    Ok((output.logits.squeeze_dim::<2>(1), hidden))
}

/// Creates one batch-one token tensor after portable integer conversion.
fn token_tensor<B: Backend>(token: u32, device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(vec![i64::from(token)], [1, 1]), device)
}

/// Creates optional batch-one history used by public logits processing.
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

/// Reads one selected backend token with checked unsigned conversion.
fn selected_token<B: Backend>(token: Tensor<B, 1, Int>) -> Result<u32> {
    let values = token
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to read selected token from the backend: {error}"
            ))
        })?
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to read selected token: {error}"))
        })?;
    let value = *values
        .first()
        .ok_or_else(|| Error::Generation("token selector returned an empty batch".into()))?;
    u32::try_from(value)
        .map_err(|_| Error::Generation(format!("token selector returned invalid ID {value}")))
}

/// Converts processed logits to the same backend-softmax probabilities sampled
/// by [`LogitsProcessor`], then copies one batch-one vocabulary row to the host.
fn normalized_probabilities<B: Backend>(logits: Tensor<B, 2>) -> Result<Vec<f64>> {
    let [batch, vocabulary] = logits.dims();
    if batch != 1 || vocabulary == 0 {
        return Err(Error::shape(
            "qwen.speculative.logits",
            "[1, nonzero_vocabulary]",
            format!("[{batch}, {vocabulary}]"),
        ));
    }
    let values = logits
        .clone()
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to read processed logits from the backend: {error}"
            ))
        })?
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to read processed logits: {error}"))
        })?;
    if values.iter().any(|value| value.is_nan()) {
        return Err(Error::Generation("processed logits contain NaN".into()));
    }
    if values
        .iter()
        .any(|value| value.is_infinite() && value.is_sign_positive())
    {
        return Err(Error::Generation(
            "processed logits contain positive infinity".into(),
        ));
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite() {
        return Err(Error::Generation(
            "processed logits assign zero mass to every token".into(),
        ));
    }
    let probabilities = softmax(logits, 1)
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to read processed probabilities from the backend: {error}"
            ))
        })?
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to read processed probabilities: {error}"))
        })?;
    if probabilities
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(Error::Generation(
            "processed logits produced invalid softmax probabilities".into(),
        ));
    }
    let mut probabilities = probabilities.into_iter().map(f64::from).collect::<Vec<_>>();
    let total = probabilities.iter().sum::<f64>();
    if !(total.is_finite() && total > 0.0) {
        return Err(Error::Generation(
            "processed logits cannot be normalized".into(),
        ));
    }
    probabilities.iter_mut().for_each(|value| *value /= total);
    Ok(probabilities)
}

/// Returns the first greatest-probability token, matching tensor argmax ties.
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

/// Samples non-negative weights by inverse CDF with deterministic tie behavior.
fn sample_weights(weights: &[f64], uniform: f64) -> Result<u32> {
    let total = weights.iter().sum::<f64>();
    if !(total.is_finite() && total > 0.0) {
        return Err(Error::Generation(
            "rejection residual has no positive probability mass".into(),
        ));
    }
    let threshold = uniform * total;
    let mut cumulative = 0.0;
    for (index, &weight) in weights.iter().enumerate() {
        if !(weight.is_finite() && weight >= 0.0) {
            return Err(Error::Generation(
                "rejection residual contains an invalid weight".into(),
            ));
        }
        cumulative += weight;
        if threshold < cumulative {
            return u32::try_from(index)
                .map_err(|_| Error::Generation("sampled token does not fit u32".into()));
        }
    }
    u32::try_from(weights.len() - 1)
        .map_err(|_| Error::Generation("sampled token does not fit u32".into()))
}

/// Small deterministic stream dedicated to verification decisions.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    /// Wrapping 64-bit counter mixed to produce the next verification uniform.
    state: u64,
}

impl SplitMix64 {
    /// Initializes the independent deterministic stream used for verification decisions.
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a deterministic uniform variate in `[0, 1)` from the upper 53 mixed bits.
    ///
    /// The conversion fits the F64 mantissa exactly and excludes one, preventing
    /// out-of-range inverse-CDF selections at the top of the sampling interval.
    fn next_unit(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        // The greatest 53-bit value remains strictly below one.
        ((value >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[cfg(test)]
#[path = "speculative_tests.rs"]
mod tests;
