//! Stateful execution for the checkpoint-declared Nemotron-H MTP stage.
//!
//! # Shifted pairs
//!
//! Lightning's auxiliary stream is shifted by one token relative to the
//! target: MTP cache position `i` holds the attention keys and values of the
//! pair `(x_{i+1}, h_i)`, the embedding of the next token fused with the
//! target's final hidden state at position `i`. A stable cycle boundary
//! therefore satisfies `L_target = L_mtp + 1`. The MTP attention keys and
//! values live in a physically independent bank; only the token embedding
//! and the language-model head share target parameters.
//!
//! # Cache writes
//!
//! A pair's keys and values depend only on
//! `fuse -> attention_norm -> k_proj/v_proj`, not on attention, the MoE, or
//! the vocabulary projection. Prompt alignment
//! ([`NemotronHForCausalLm::prefill_mtp_cache`]) and the pairs a speculative
//! cycle keeps therefore run that key/value projection alone, so a long
//! prompt never materializes MTP logits.
//!
//! Every write is one step of the MTP bank: committed outside a transaction
//! and staged in place inside one. Attention writes the step's keys ahead
//! into the full cache and attends over one zero-copy view of the history.
//!
//! # Drafting
//!
//! While drafting, pair `j` of a cycle is `(y_j, m_{j-1})` with the MTP's own
//! hidden state `m`, which approximates the target's. Draft pairs are only
//! written ahead into unreachable slots and are never published: after
//! verification the cycle stages the accepted pairs rebuilt from the
//! target's hidden states, exactly as vLLM's speculator rewrites the draft
//! KV of every verified position.

use core::num::NonZeroUsize;

use burn::tensor::{DType, ElementConversion, Int, Tensor, TensorData, backend::Backend};

use crate::{
    CausalLmInput, Error, Result,
    attention::AttentionProjection,
    cache::{CacheConfig, DecodeStateError, TensorCacheError, TensorStepWrites},
    operations,
};

use super::{
    INPUT_NAMES, NemotronHDecodeState, NemotronHForCausalLm, NemotronHNextTokenHead,
    batch::CompactBatch,
    decode::{KV_GROUP, Retain, StatefulPhase, prepare_stateful_input},
};
use crate::models::shared::validate_token_values;

/// One cached MTP prediction after projection through the shared target head.
#[derive(Debug)]
pub struct NemotronHMtpPrediction<B: Backend> {
    /// F32 vocabulary logits `[1, sequence, vocabulary]`.
    pub logits: Tensor<B, 3>,
    /// Final normalized auxiliary state recurrently fed to the next draft step.
    pub hidden_state: Tensor<B, 3>,
}

/// What the first speculative cycle needs after a prompt.
#[derive(Debug)]
pub struct NemotronHSpeculativePrefill<B: Backend> {
    /// F32 logits `[1, vocabulary]` after the last prompt token; select the lead from them.
    pub logits: Tensor<B, 2>,
    /// Target final hidden state `[1, 1, hidden]` of the last prompt token, the state before the lead.
    pub hidden_state: Tensor<B, 3>,
}

/// Keys and values of the draft pairs evaluated so far in one speculative cycle.
///
/// Draft pairs never enter the cache's readable prefix; each draft step
/// writes all of them ahead at the MTP working cursor and attends over the
/// resulting view.
#[derive(Debug)]
pub(super) struct DraftPairs<B: Backend> {
    /// Keys `[1, kv_heads, drafted, head_dim]`, or `None` before the first draft.
    keys: Option<Tensor<B, 4>>,
    /// Values with the same geometry as `keys`.
    values: Option<Tensor<B, 4>>,
}

impl<B: Backend> DraftPairs<B> {
    /// Starts an empty draft.
    pub(super) const fn new() -> Self {
        Self {
            keys: None,
            values: None,
        }
    }

    /// Appends one pair's keys and values and returns every pair so far.
    fn push(&mut self, keys: Tensor<B, 4>, values: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let join = |earlier: Option<Tensor<B, 4>>, next: Tensor<B, 4>| match earlier {
            Some(earlier) => Tensor::cat(vec![earlier, next], 2),
            None => next,
        };
        let keys = join(self.keys.take(), keys);
        let values = join(self.values.take(), values);
        self.keys = Some(keys.clone());
        self.values = Some(values.clone());
        (keys, values)
    }

    /// Returns the number of draft pairs evaluated so far.
    fn len(&self) -> usize {
        self.keys.as_ref().map_or(0, |keys| keys.dims()[2])
    }
}

impl<B: Backend> NemotronHForCausalLm<B> {
    /// Allocates the batch-one transactional state required by MTP speculation.
    ///
    /// The target and auxiliary banks each reserve `K + 1` tentative slots. A
    /// target cycle may consume the lead token plus all `K` accepted proposals,
    /// and the auxiliary stream keeps the same number of rebuilt pairs.
    ///
    /// # Errors
    ///
    /// Returns a generation error for a batch other than one, zero drafts, a
    /// model without an MTP stage, or staging beyond the cache capacity.
    pub fn new_speculative_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        max_draft_tokens: usize,
        device: &B::Device,
    ) -> Result<NemotronHDecodeState<B>> {
        if batch_size != 1 {
            return Err(Error::Generation(format!(
                "Nemotron-H MTP speculative decoding supports batch size one, found {batch_size}"
            )));
        }
        if max_draft_tokens == 0 {
            return Err(Error::Generation(
                "Nemotron-H MTP max_draft_tokens must be greater than zero".into(),
            ));
        }
        if self.next_token_head.is_none() {
            return Err(Error::Generation(
                "Nemotron-H model has no configured next-token prediction stage".into(),
            ));
        }
        let staging_capacity = max_draft_tokens.checked_add(1).ok_or_else(|| {
            Error::Generation("Nemotron-H MTP staging capacity overflows usize".into())
        })?;
        if staging_capacity > config.capacity() {
            return Err(Error::Generation(format!(
                "Nemotron-H MTP needs staging capacity {staging_capacity}, exceeding cache capacity {}",
                config.capacity()
            )));
        }
        NemotronHDecodeState::new_with_staging(self, config, batch_size, staging_capacity, device)
    }

    /// Prefills a fresh speculative state's target and MTP banks from one prompt.
    ///
    /// This is [`CausalLanguageModel::prefill`] followed by
    /// [`Self::prefill_mtp_cache`], without retaining any per-layer hidden
    /// state: the target pass keeps only its final normalized states, which
    /// the MTP pairs `(x_{i+1}, h_i)` need, and projects logits only for the
    /// last token. For long prompts this avoids holding every block's
    /// `[1, L, hidden]` activations just to recover the final ones.
    ///
    /// Both banks are validated and computed before either is published, so
    /// an error leaves the state unchanged.
    ///
    /// # Errors
    ///
    /// Returns shape errors for a batch other than one, an empty prompt, or
    /// out-of-vocabulary tokens; cache errors for a non-fresh state or an
    /// active transaction; and the errors of the cached pass.
    ///
    /// [`CausalLanguageModel::prefill`]: crate::CausalLanguageModel::prefill
    pub fn prefill_speculative(
        &self,
        prompt_token_ids: Tensor<B, 2, Int>,
        state: &mut NemotronHDecodeState<B>,
    ) -> Result<NemotronHSpeculativePrefill<B>> {
        state.validate_for_model(self)?;
        let head = self.next_token_head.as_ref().ok_or_else(|| {
            Error::Generation(
                "Nemotron-H model has no configured next-token prediction stage".into(),
            )
        })?;
        if state.tensors.has_active_transaction() {
            return Err(TensorCacheError::TransactionActive {
                operation: "prefill a Nemotron-H speculative state",
            }
            .into());
        }
        if state.tensors.lengths() != [0] || state.tensors.mtp_lengths() != Some([0].as_slice()) {
            return Err(Error::Cache(
                "Nemotron-H speculative prefill requires a fresh batch-one state".into(),
            ));
        }
        let input = CausalLmInput {
            token_ids: prompt_token_ids.clone(),
            token_mask: None,
            position_ids: None,
        };
        let prepared = prepare_stateful_input(self, &input, state, StatefulPhase::Prefill)?;
        let sequence = prepared.sequence;
        let batch = CompactBatch::new(&prepared.valid_indices, sequence, state.tensors.device())?;
        let tokens = batch.compact_tokens(prompt_token_ids.clone())?;
        let (pass, step) = self.cached_pass(tokens, &batch, state, Retain::default())?;
        let hidden_state = pass.hidden.clone().narrow(1, sequence - 1, 1);
        let logits = self
            .project_logits(hidden_state.clone())?
            .cast(DType::F32)
            .squeeze_dim::<2>(1);
        // Shifted pairs (x_{i+1}, h_i) for i < L - 1, computed before any publication.
        let pairs = (sequence > 1)
            .then(|| {
                let fused = head.fuse(
                    self.model
                        .embed_tokens
                        .forward(prompt_token_ids.narrow(1, 1, sequence - 1)),
                    pass.hidden.narrow(1, 0, sequence - 1),
                )?;
                pair_key_values(head, head.attention_norm.forward(fused))
            })
            .transpose()?;
        let mtp_step = pairs
            .map(|(keys, values)| {
                let mut step = state.tensors.mtp_step(&[sequence - 1])?;
                step.push_kv_batch(KV_GROUP, 0, keys, values);
                Ok::<_, Error>(step)
            })
            .transpose()?;
        // The MTP step appends L - 1 < L pairs to an empty bank, which the
        // target step's capacity check already covers; publishing banks in
        // this order cannot leave a prefilled target with an unaligned MTP.
        state.tensors.publish_step(step)?;
        if let Some(step) = mtp_step {
            state.tensors.publish_step(step)?;
        }
        Ok(NemotronHSpeculativePrefill {
            logits,
            hidden_state,
        })
    }

    /// Evaluates shifted MTP pairs against the independent incremental KV bank.
    ///
    /// `next_token_ids[t]` is paired with the target or previous-MTP hidden
    /// state immediately preceding it. The pairs' keys and values are
    /// committed outside a transaction and staged inside one, so a later
    /// [`NemotronHDecodeState::commit_prefixes`] may keep any prefix of them.
    ///
    /// # Errors
    ///
    /// Returns shape errors for mismatched inputs, explicit positions that are
    /// not the contiguous cache positions, or out-of-vocabulary tokens; a
    /// capability error for mixed devices; and cache errors for a missing MTP
    /// bank or exhausted capacity. Nothing is published on error.
    pub fn forward_mtp_cached(
        &self,
        next_token_ids: Tensor<B, 2, Int>,
        target_or_mtp_hidden: Tensor<B, 3>,
        position_ids: Option<Tensor<B, 2, Int>>,
        state: &mut NemotronHDecodeState<B>,
    ) -> Result<NemotronHMtpPrediction<B>> {
        let head = self.validate_mtp_pairs(&next_token_ids, &target_or_mtp_hidden, state)?;
        let sequence = next_token_ids.dims()[1];
        validate_mtp_positions(
            position_ids,
            mtp_working_length(state)?,
            sequence,
            self.config().max_position_embeddings,
            state.tensors.device(),
        )?;
        let fused = head.fuse(
            self.model.embed_tokens.forward(next_token_ids),
            target_or_mtp_hidden,
        )?;
        let projection = head
            .attention
            .project_self(head.attention_norm.forward(fused.clone()))?;
        let keys = projection.key.clone();
        let values = projection.value.clone();
        let mut step = state.tensors.mtp_step(&[sequence])?;
        let attention = mtp_attention(head, projection, keys, values, &mut step, state)?;
        let (hidden_state, _) = head.finish(fused + attention)?;
        let logits = self.project_logits(hidden_state.clone())?.cast(DType::F32);
        state.tensors.publish_step(step)?;
        Ok(NemotronHMtpPrediction {
            logits,
            hidden_state,
        })
    }

    /// Aligns an empty auxiliary bank with an already-prefilled target prompt.
    ///
    /// For target tokens `x_0..x_(L-1)` and states `h_0..h_(L-1)`, the MTP bank
    /// receives the keys and values of the `L-1` pairs `(x_(i+1), h_i)`. It
    /// therefore remains one token behind the target until a speculative
    /// cycle supplies its lead-token pair. Only the key/value projection runs:
    /// no MTP attention, MoE, or vocabulary projection is evaluated for the
    /// prompt, whose predictions nobody reads.
    ///
    /// # Errors
    ///
    /// Returns shape errors for mismatched prompt and hidden states or
    /// out-of-vocabulary tokens, and cache errors for a target cache whose
    /// length differs from the prompt, a non-fresh MTP bank, or an active
    /// transaction.
    pub fn prefill_mtp_cache(
        &self,
        prompt_token_ids: Tensor<B, 2, Int>,
        target_final_hidden_states: Tensor<B, 3>,
        state: &mut NemotronHDecodeState<B>,
    ) -> Result<()> {
        state.validate_for_model(self)?;
        if state.tensors.has_active_transaction() {
            return Err(TensorCacheError::TransactionActive {
                operation: "prefill Nemotron-H MTP cache",
            }
            .into());
        }
        let [batch, sequence] = prompt_token_ids.dims();
        if batch != 1
            || sequence == 0
            || target_final_hidden_states.dims() != [1, sequence, self.config().hidden_size]
        {
            return Err(Error::shape(
                "nemotron_h.mtp.prefill",
                format!(
                    "token IDs [1, L] and target hidden states [1, L, {}] for L > 0",
                    self.config().hidden_size
                ),
                format!(
                    "tokens={:?}, hidden={:?}",
                    prompt_token_ids.dims(),
                    target_final_hidden_states.dims()
                ),
            ));
        }
        // Validate every prompt token, including one that forms no pair.
        self.validate_mtp_pairs(&prompt_token_ids, &target_final_hidden_states, state)?;
        if state.tensors.lengths() != [sequence] {
            return Err(Error::Cache(format!(
                "Nemotron-H target cache length must equal MTP prompt length {sequence}, found {:?}",
                state.tensors.lengths()
            )));
        }
        if state.tensors.mtp_lengths() != Some([0].as_slice()) {
            return Err(Error::Cache(
                "Nemotron-H MTP prefill requires a fresh auxiliary cache".into(),
            ));
        }
        if sequence == 1 {
            return Ok(());
        }
        self.append_mtp_pairs(
            prompt_token_ids.narrow(1, 1, sequence - 1),
            target_final_hidden_states.narrow(1, 0, sequence - 1),
            state,
        )
    }

    /// Appends the keys and values of shifted pairs as one MTP step, without attention.
    ///
    /// The step is committed outside a transaction and staged inside one. A
    /// speculative cycle uses this to keep exactly the verified pairs, built
    /// from target hidden states.
    pub(super) fn append_mtp_pairs(
        &self,
        next_token_ids: Tensor<B, 2, Int>,
        target_hidden: Tensor<B, 3>,
        state: &mut NemotronHDecodeState<B>,
    ) -> Result<()> {
        let head = self.validate_mtp_pairs(&next_token_ids, &target_hidden, state)?;
        let sequence = next_token_ids.dims()[1];
        validate_mtp_positions::<B>(
            None,
            mtp_working_length(state)?,
            sequence,
            self.config().max_position_embeddings,
            state.tensors.device(),
        )?;
        let fused = head.fuse(
            self.model.embed_tokens.forward(next_token_ids),
            target_hidden,
        )?;
        let (keys, values) = pair_key_values(head, head.attention_norm.forward(fused))?;
        let mut step = state.tensors.mtp_step(&[sequence])?;
        step.push_kv_batch(KV_GROUP, 0, keys, values);
        state.tensors.publish_step(step).map_err(Into::into)
    }

    /// Evaluates one draft pair `(token, hidden)` without publishing anything.
    ///
    /// The pair joins `drafts`; every draft pair of the cycle is written ahead
    /// at the MTP working cursor and the new pair attends over the committed
    /// history plus all of them. The cache's readable state is unchanged.
    pub(super) fn draft_mtp(
        &self,
        token: u32,
        hidden: Tensor<B, 3>,
        drafts: &mut DraftPairs<B>,
        state: &mut NemotronHDecodeState<B>,
    ) -> Result<NemotronHMtpPrediction<B>> {
        let token_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![i64::from(token)], [1, 1]),
            state.tensors.device(),
        );
        let head = self.validate_mtp_pairs(&token_ids, &hidden, state)?;
        let fused = head.fuse(self.model.embed_tokens.forward(token_ids), hidden)?;
        let projection = head
            .attention
            .project_self(head.attention_norm.forward(fused.clone()))?;
        let (keys, values) = drafts.push(projection.key.clone(), projection.value.clone());
        let mut step = state.tensors.mtp_step(&[drafts.len()])?;
        let attention = mtp_attention(head, projection, keys, values, &mut step, state)?;
        let (hidden_state, _) = head.finish(fused + attention)?;
        let logits = self.project_logits(hidden_state.clone())?.cast(DType::F32);
        Ok(NemotronHMtpPrediction {
            logits,
            hidden_state,
        })
    }

    /// Validates batch-one shifted pairs and returns the MTP head.
    fn validate_mtp_pairs(
        &self,
        next_token_ids: &Tensor<B, 2, Int>,
        hidden: &Tensor<B, 3>,
        state: &NemotronHDecodeState<B>,
    ) -> Result<&NemotronHNextTokenHead<B>> {
        state.validate_for_model(self)?;
        let head = self.next_token_head.as_ref().ok_or_else(|| {
            Error::Generation(
                "Nemotron-H model has no configured next-token prediction stage".into(),
            )
        })?;
        let [batch, sequence] = next_token_ids.dims();
        if batch != 1 || batch != state.tensors.batch_size() || sequence == 0 {
            return Err(Error::shape(
                "nemotron_h.mtp.next_token_ids",
                "[1, nonzero_sequence] matching a batch-one decode state",
                format!("[{batch}, {sequence}]"),
            ));
        }
        if hidden.dims() != [1, sequence, self.config().hidden_size] {
            return Err(Error::shape(
                "nemotron_h.mtp.hidden_state",
                format!("[1, {sequence}, {}]", self.config().hidden_size),
                format!("{:?}", hidden.dims()),
            ));
        }
        ensure_mtp_device(
            "next-token IDs",
            &next_token_ids.device(),
            state.tensors.device(),
        )?;
        ensure_mtp_device("hidden state", &hidden.device(), state.tensors.device())?;
        ensure_mtp_device(
            "parameters",
            &head.eh_proj.weight.val().device(),
            state.tensors.device(),
        )?;
        validate_token_values(
            next_token_ids,
            self.config().vocab_size,
            INPUT_NAMES.token_ids,
        )?;
        Ok(head)
    }
}

/// Writes a step's keys ahead and attends its last queries over the MTP history.
///
/// `projection` holds the queries of the step's last `q` pairs; `keys` and
/// `values` `[1, kv_heads, w, head_dim]` hold all `w >= q` pairs of the step,
/// which land at slots `[W, W + w)` after the working length `W`. The queries
/// sit at positions `W + w - q, …, W + w - 1`, so attention is bottom-right
/// causal over the view `[0, W + w)`; an active sliding window adds
/// position-based exclusions.
fn mtp_attention<B: Backend>(
    head: &NemotronHNextTokenHead<B>,
    mut projection: AttentionProjection<B>,
    keys: Tensor<B, 4>,
    values: Tensor<B, 4>,
    step: &mut TensorStepWrites<B>,
    state: &mut NemotronHDecodeState<B>,
) -> Result<Tensor<B, 3>> {
    let queries = projection.query.dims()[2];
    let end = mtp_working_length(state)?
        .checked_add(keys.dims()[2])
        .ok_or_else(|| Error::Generation("Nemotron-H MTP position overflows usize".into()))?;
    if end > state.tensors.capacity() {
        return Err(
            TensorCacheError::Metadata(DecodeStateError::CapacityExceeded {
                sequence: 0,
                requested: end,
                capacity: state.tensors.capacity(),
            })
            .into(),
        );
    }
    let plan = state.tensors.kv_read_plan_through(step, KV_GROUP)?;
    state
        .tensors
        .write_kv_ahead(step, KV_GROUP, 0, keys, values)?;
    let segment = state
        .tensors
        .read_kv_through(step, KV_GROUP, 0, &plan)?
        .materialize()
        .ok_or_else(|| Error::Cache("Nemotron-H MTP attention read returned no keys".into()))?;
    projection.key = segment.keys().clone();
    projection.value = segment.masked_values();
    // A window wider than the whole history can exclude nothing.
    let window = head
        .sliding_window
        .filter(|&window| window < end)
        .and_then(NonZeroUsize::new);
    let Some(window) = window else {
        return head.attention.attend(projection, None, None, true);
    };
    let positions = (end - queries..end)
        .map(|position| {
            i64::try_from(position).map_err(|_| {
                Error::configuration(
                    "nemotron_h.mtp.position_ids",
                    "position exceeds the signed index range",
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let positions = Tensor::<B, 2, Int>::from_data(
        TensorData::new(positions, [1, queries]),
        state.tensors.device(),
    );
    let mask = segment.exclusion_mask(&positions, head.attention.query_heads, Some(window))?;
    head.attention.attend(projection, Some(mask), None, false)
}

/// Projects the MTP attention keys and values `[1, kv_heads, n, head_dim]` of normalized pairs.
///
/// This is the key/value half of `GroupedQueryAttention::project_self`
/// without the much wider query projection, which only attention needs
/// (Lightning's queries are eight times wider than its keys and values).
/// Nemotron-H's MTP attention has no key normalization; a head that had one
/// is rejected rather than silently projected differently.
fn pair_key_values<B: Backend>(
    head: &NemotronHNextTokenHead<B>,
    normalized: Tensor<B, 3>,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
    let attention = &head.attention;
    if attention.key_norm.is_some() {
        return Err(Error::BackendCapability(
            "Nemotron-H MTP key/value projection does not support key normalization".into(),
        ));
    }
    let [batch, sequence, _] = normalized.dims();
    let heads = |projected: Tensor<B, 3>| {
        projected
            .reshape([
                batch,
                sequence,
                attention.key_value_heads,
                attention.head_dim,
            ])
            .swap_dims(1, 2)
    };
    Ok((
        heads(operations::linear(normalized.clone(), &attention.k_proj)?),
        heads(operations::linear(normalized, &attention.v_proj)?),
    ))
}

/// Returns the MTP bank's working length (committed plus staged pairs) of row zero.
fn mtp_working_length<B: Backend>(state: &NemotronHDecodeState<B>) -> Result<usize> {
    state
        .tensors
        .mtp_working_lengths()
        .and_then(|lengths| lengths.first().copied())
        .ok_or_else(|| Error::Cache("Nemotron-H MTP working cursor is not enabled".into()))
}

/// Validates vLLM's monotonically advanced MTP position contract.
///
/// Nemotron attention is NoPE, so positions do not rotate vectors. They still
/// identify cache slots in the reference scheduler; accepting a different
/// explicit sequence here would silently disagree with our contiguous bank.
fn validate_mtp_positions<B: Backend>(
    positions: Option<Tensor<B, 2, Int>>,
    baseline: usize,
    sequence: usize,
    maximum: usize,
    device: &B::Device,
) -> Result<()> {
    let end = baseline.checked_add(sequence).ok_or_else(|| {
        Error::configuration(
            "nemotron_h.mtp.position_ids",
            "position range overflows usize",
        )
    })?;
    if end > maximum {
        return Err(Error::shape(
            "nemotron_h.mtp.position_ids",
            format!("positions below {maximum}"),
            format!("range {baseline}..{end}"),
        ));
    }
    let Some(positions) = positions else {
        return Ok(());
    };
    if positions.device() != *device {
        return Err(Error::BackendCapability(
            "Nemotron-H MTP positions and cache must share a device".into(),
        ));
    }
    if positions.dims() != [1, sequence] {
        return Err(Error::shape(
            "nemotron_h.mtp.position_ids",
            format!("[1, {sequence}]"),
            format!("{:?}", positions.dims()),
        ));
    }
    let expected_start = i64::try_from(baseline)
        .map_err(|_| Error::configuration("nemotron_h.mtp.position_ids", "baseline exceeds i64"))?;
    let expected_end = i64::try_from(end)
        .map_err(|_| Error::configuration("nemotron_h.mtp.position_ids", "end exceeds i64"))?;
    let expected =
        Tensor::<B, 1, Int>::arange(expected_start..expected_end, device).reshape([1, sequence]);
    let matches = positions
        .equal(expected)
        .all()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to validate Nemotron-H MTP positions: {error}"
            ))
        })?
        .elem::<bool>();
    if !matches {
        return Err(Error::shape(
            "nemotron_h.mtp.position_ids",
            format!("contiguous cache positions {baseline}..{end}"),
            "a different explicit position sequence",
        ));
    }
    Ok(())
}

/// Rejects mixed-device MTP inputs before backend dispatch.
fn ensure_mtp_device<D: PartialEq>(label: &str, actual: &D, expected: &D) -> Result<()> {
    if actual != expected {
        return Err(Error::BackendCapability(format!(
            "Nemotron-H MTP {label} must use the decode-state device"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "mtp_tests.rs"]
mod tests;
