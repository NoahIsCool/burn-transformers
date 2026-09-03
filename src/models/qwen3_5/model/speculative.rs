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

const DRAFT_SEED_DOMAIN: u64 = 0x4452_4146_545f_5157;
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
    config: Qwen3_5SpeculativeConfig,
    vocabulary_size: usize,
    target: LogitsProcessor<B>,
    draft: LogitsProcessor<B>,
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
            let probabilities = normalized_probabilities(processed.clone())?;
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
            let p = normalized_probabilities(processed)?;
            if !self.accept(proposal, &p, q)? {
                let correction = match self.config.generation.strategy {
                    SamplingStrategy::Greedy => argmax(&p)?,
                    SamplingStrategy::Sample => self.sample_residual(&p, q)?,
                };
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
    token_ids: Vec<u32>,
    drafted_tokens: usize,
    accepted_draft_tokens: usize,
    all_drafts_accepted: bool,
    terminated: bool,
    continuation_hidden_state: Option<Tensor<B, 3>>,
    target_cache_tokens: usize,
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
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

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
mod tests {
    use burn::{
        backend::Flex,
        tensor::{Int, Tensor, TensorData, backend::BackendTypes},
    };

    use super::*;
    use crate::{
        CausalLmInput,
        cache::CacheConfig,
        checkpoint::TensorSink,
        models::qwen3_5::{
            Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5LayerType, Qwen3_5TextConfig,
        },
    };

    type TestBackend = Flex<f32, i32>;

    /// Creates a zero-logit graph whose greedy target and draft both select ID zero.
    fn zero_model(
        device: &<TestBackend as BackendTypes>::Device,
    ) -> Qwen3_5ForCausalLm<TestBackend> {
        let config = Qwen3_5TextConfig {
            vocab_size: 8,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 32,
            layer_types: vec![Qwen3_5LayerType::FullAttention],
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 1,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 3,
            partial_rotary_factor: Some(1.0),
            rope_theta: Some(10_000.0),
            mtp_num_hidden_layers: 1,
            tie_word_embeddings: true,
            ..Default::default()
        };
        let contract = config
            .checkpoint_contract(Qwen3_5CheckpointOptions {
                source: Qwen3_5CheckpointSource::TextOnly,
                include_mtp: true,
            })
            .unwrap();
        let mut model = Qwen3_5ForCausalLm::init(config, device).unwrap();
        for expected in contract.expected_tensors() {
            if expected.name == "lm_head.weight" {
                continue;
            }
            let shape = expected
                .shape
                .iter()
                .map(|dimension| usize::try_from(*dimension).unwrap())
                .collect::<Vec<_>>();
            model
                .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
                .unwrap();
        }
        model
            .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
            .unwrap();
        model
    }

    /// Creates a graph whose MTP head proposes zero while target transitions
    /// `1 -> 0` and `0 -> 1`, producing controlled acceptance prefixes.
    fn scripted_rejection_model(
        device: &<TestBackend as BackendTypes>::Device,
    ) -> Qwen3_5ForCausalLm<TestBackend> {
        let config = Qwen3_5TextConfig {
            vocab_size: 8,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 32,
            layer_types: vec![Qwen3_5LayerType::FullAttention],
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 1,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 3,
            partial_rotary_factor: Some(1.0),
            rope_theta: Some(10_000.0),
            mtp_num_hidden_layers: 1,
            tie_word_embeddings: false,
            ..Default::default()
        };
        let contract = config
            .checkpoint_contract(Qwen3_5CheckpointOptions {
                source: Qwen3_5CheckpointSource::TextOnly,
                include_mtp: true,
            })
            .unwrap();
        let mut model = Qwen3_5ForCausalLm::init(config, device).unwrap();
        for expected in contract.expected_tensors() {
            let shape = expected
                .shape
                .iter()
                .map(|dimension| usize::try_from(*dimension).unwrap())
                .collect::<Vec<_>>();
            model
                .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
                .unwrap();
        }

        let mut embeddings = vec![0.0_f32; 8 * 8];
        embeddings[0] = 1.0;
        embeddings[8 + 1] = 1.0;
        model
            .write_tensor(
                "model.embed_tokens.weight",
                TensorData::new(embeddings, [8, 8]),
            )
            .unwrap();
        let mut head = vec![0.0_f32; 8 * 8];
        head[1] = 1.0; // hidden for token 1 predicts token 0
        head[8] = 1.0; // hidden for token 0 predicts token 1
        model
            .write_tensor("lm_head.weight", TensorData::new(head, [8, 8]))
            .unwrap();
        model
    }

    /// Prefills target and shifted MTP caches and returns the last target hidden state.
    fn aligned_state(
        model: &Qwen3_5ForCausalLm<TestBackend>,
        device: &<TestBackend as BackendTypes>::Device,
    ) -> (Qwen3_5DecodeState<TestBackend>, Tensor<TestBackend, 3>) {
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

    #[test]
    fn defaults_to_two_drafts_and_rejects_zero() {
        assert_eq!(Qwen3_5SpeculativeConfig::default().max_draft_tokens, 2);
        let config = Qwen3_5SpeculativeConfig {
            max_draft_tokens: 0,
            ..Default::default()
        };
        assert!(config.validate(8).is_err());
    }

    #[test]
    fn probability_normalization_matches_backend_softmax_and_rejects_non_finite_logits() {
        let device = Default::default();
        let finite = normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 0.0, f32::NEG_INFINITY]],
            &device,
        ))
        .unwrap();
        assert_eq!(finite, vec![0.5, 0.5, 0.0]);
        let infinite = normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
            [[f32::INFINITY, 1.0, f32::INFINITY]],
            &device,
        ));
        assert!(infinite.is_err());
        let empty_mass = normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
            [[f32::NEG_INFINITY, f32::NEG_INFINITY]],
            &device,
        ));
        assert!(empty_mass.is_err());
    }

    #[test]
    fn greedy_acceptance_covers_every_prefix_length() {
        for accepted_prefix in 0..=2 {
            let device = Default::default();
            let mut verifier = Qwen3_5SpeculativeDecoder::<TestBackend>::new(
                Qwen3_5SpeculativeConfig::default(),
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

    #[test]
    fn residual_sampling_never_returns_non_positive_support() {
        for uniform in [0.0, 0.1, 0.499, 0.999_999] {
            let token = sample_weights(&[0.0, 0.4, 0.0, 0.6], uniform).unwrap();
            assert!(token == 1 || token == 3);
        }
        assert!(sample_weights(&[0.0, 0.0], 0.5).is_err());
    }

    #[test]
    fn splitmix_stream_is_reproducible_and_strictly_unit_interval() {
        let mut left = SplitMix64::new(19);
        let mut right = SplitMix64::new(19);
        for _ in 0..128 {
            let lhs = left.next_unit();
            assert_eq!(lhs, right.next_unit());
            assert!((0.0..1.0).contains(&lhs));
        }
    }

    #[test]
    fn all_accepted_cycle_commits_catch_up_and_preserves_shift() {
        let device = Default::default();
        let model = zero_model(&device);
        let (mut state, hidden) = aligned_state(&model, &device);
        let mut decoder = Qwen3_5SpeculativeDecoder::new(
            Qwen3_5SpeculativeConfig::default(),
            model.config().vocab_size,
            &device,
        )
        .unwrap();
        let output = decoder
            .speculate(&model, 1, hidden, &[2, 3], &mut state)
            .unwrap();
        assert_eq!(output.token_ids, vec![1, 0, 0, 0]);
        assert_eq!(output.drafted_tokens, 2);
        assert_eq!(output.accepted_draft_tokens, 2);
        assert!(output.all_drafts_accepted);
        assert!(!output.terminated);
        assert!(output.continuation_hidden_state.is_some());
        assert_eq!(state.tensor_state().lengths(), &[5]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([4].as_slice()));
    }

    #[test]
    fn complete_cycles_commit_every_acceptance_prefix() {
        let device = Default::default();
        let model = scripted_rejection_model(&device);
        for (lead, accepted, emitted) in [(0, 0, vec![0, 1]), (1, 1, vec![1, 0, 1])] {
            let (mut state, hidden) = aligned_state(&model, &device);
            let mut decoder = Qwen3_5SpeculativeDecoder::new(
                Qwen3_5SpeculativeConfig::default(),
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

    #[test]
    fn eos_lead_and_eos_draft_stop_without_cache_drift() {
        let device = Default::default();
        let model = zero_model(&device);
        let (mut state, hidden) = aligned_state(&model, &device);
        let config = Qwen3_5SpeculativeConfig {
            generation: GenerationConfig {
                eos_token_ids: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut decoder =
            Qwen3_5SpeculativeDecoder::new(config, model.config().vocab_size, &device).unwrap();
        let lead = decoder
            .speculate(&model, 1, hidden.clone(), &[2, 3], &mut state)
            .unwrap();
        assert_eq!(lead.token_ids, vec![1]);
        assert!(lead.terminated);
        assert_eq!(state.tensor_state().lengths(), &[2]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));

        let config = Qwen3_5SpeculativeConfig {
            generation: GenerationConfig {
                eos_token_ids: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut decoder =
            Qwen3_5SpeculativeDecoder::new(config, model.config().vocab_size, &device).unwrap();
        let draft = decoder
            .speculate(&model, 1, hidden, &[2, 3], &mut state)
            .unwrap();
        assert_eq!(draft.token_ids, vec![1, 0]);
        assert_eq!(draft.accepted_draft_tokens, 1);
        assert!(draft.terminated);
        assert_eq!(state.tensor_state().lengths(), &[3]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));

        let (mut state, hidden) = aligned_state(&model, &device);
        let config = Qwen3_5SpeculativeConfig {
            generation: GenerationConfig {
                presence_penalty: 1.0,
                eos_token_ids: vec![4],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut decoder =
            Qwen3_5SpeculativeDecoder::new(config, model.config().vocab_size, &device).unwrap();
        let second_draft = decoder
            .speculate(&model, 1, hidden, &[2, 3], &mut state)
            .unwrap();
        assert_eq!(second_draft.token_ids, vec![1, 0, 4]);
        assert_eq!(second_draft.accepted_draft_tokens, 2);
        assert!(second_draft.terminated);
        assert_eq!(state.tensor_state().lengths(), &[4]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([3].as_slice()));
    }

    #[test]
    fn inactive_and_recoverable_failure_leave_both_cursors_unchanged() {
        let device = Default::default();
        let model = zero_model(&device);
        let (mut state, hidden) = aligned_state(&model, &device);
        let mut decoder = Qwen3_5SpeculativeDecoder::new(
            Qwen3_5SpeculativeConfig::default(),
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

        // Deliberately replace the internal draft processor with one for a
        // smaller vocabulary. Public construction cannot create this mismatch;
        // the test-only corruption forces an error after an MTP KV proposal was
        // staged and therefore exercises the recovery path itself.
        let mut wrong = Qwen3_5SpeculativeDecoder::new(
            Qwen3_5SpeculativeConfig::default(),
            model.config().vocab_size,
            &device,
        )
        .unwrap();
        wrong.draft = LogitsProcessor::new(
            wrong.config.generation.clone(),
            model.config().vocab_size - 1,
            &device,
        )
        .unwrap();
        let first = wrong
            .speculate(&model, 1, hidden.clone(), &[2, 3], &mut state)
            .unwrap_err();
        assert!(matches!(first, Error::Shape { .. }));
        assert_eq!(state.tensor_state().lengths(), &[2]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([1].as_slice()));
        let second = wrong
            .speculate(&model, 1, hidden, &[2, 3], &mut state)
            .unwrap_err();
        assert_eq!(first.to_string(), second.to_string());
    }

    #[test]
    fn speculative_state_rejects_batch_capacity_and_missing_mtp() {
        let device = Default::default();
        let model = zero_model(&device);
        assert!(
            model
                .new_speculative_decode_state(CacheConfig::new(2, 8).unwrap(), 2, 2, &device)
                .is_err()
        );
        assert!(
            model
                .new_speculative_decode_state(CacheConfig::new(1, 2).unwrap(), 1, 2, &device)
                .is_err()
        );
        let mut config = model.config().clone();
        config.mtp_num_hidden_layers = 0;
        let base = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();
        assert!(
            base.new_speculative_decode_state(CacheConfig::new(1, 8).unwrap(), 1, 2, &device)
                .is_err()
        );
    }

    #[test]
    fn cycle_preflight_rejects_vocabulary_history_token_and_hidden_mismatches() {
        let device = Default::default();
        let model = zero_model(&device);
        let (mut state, hidden) = aligned_state(&model, &device);
        let mut wrong_vocabulary = Qwen3_5SpeculativeDecoder::new(
            Qwen3_5SpeculativeConfig::default(),
            model.config().vocab_size - 1,
            &device,
        )
        .unwrap();
        assert!(
            wrong_vocabulary
                .speculate(&model, 1, hidden.clone(), &[2, 3], &mut state)
                .is_err()
        );

        let mut decoder = Qwen3_5SpeculativeDecoder::new(
            Qwen3_5SpeculativeConfig::default(),
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

    #[test]
    fn stochastic_rejection_rule_recovers_target_marginal() {
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
            let frequency = observed as f64 / trials as f64;
            assert!(
                (frequency - expected).abs() < 0.01,
                "{frequency} != {expected}"
            );
        }
    }

    #[test]
    fn stochastic_cycles_with_equal_seeds_are_reproducible() {
        let device = Default::default();
        let model = zero_model(&device);
        let (mut first_state, first_hidden) = aligned_state(&model, &device);
        let (mut second_state, second_hidden) = aligned_state(&model, &device);
        let config = Qwen3_5SpeculativeConfig {
            generation: GenerationConfig {
                strategy: SamplingStrategy::Sample,
                seed: 0x5eed,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut first =
            Qwen3_5SpeculativeDecoder::new(config.clone(), model.config().vocab_size, &device)
                .unwrap();
        let mut second =
            Qwen3_5SpeculativeDecoder::new(config, model.config().vocab_size, &device).unwrap();
        let first = first
            .speculate(&model, 1, first_hidden, &[2, 3], &mut first_state)
            .unwrap();
        let second = second
            .speculate(&model, 1, second_hidden, &[2, 3], &mut second_state)
            .unwrap();
        assert_eq!(first.token_ids, second.token_ids);
        assert_eq!(first.accepted_draft_tokens, 2);
        assert_eq!(
            first_state.tensor_state().lengths(),
            second_state.tensor_state().lengths()
        );
        assert_eq!(
            first_state.tensor_state().mtp_lengths(),
            second_state.tensor_state().mtp_lengths()
        );
    }
}
