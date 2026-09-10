//! Stateful execution for the checkpoint-declared Nemotron-H MTP stage.
//!
//! Lightning's auxiliary stream is shifted by one token relative to the target:
//! cache position `i` consumes `(x_(i+1), h_i)`. Consequently a stable cycle
//! boundary satisfies `L_target = L_mtp + 1`. The MTP attention keys and values
//! live in a physically independent bank; only the token embedding and language
//! model head share target parameters.
//!
//! During speculation, both banks append into bounded staging tensors. Their
//! committed prefixes remain unchanged until verification chooses independent
//! target and MTP prefix lengths. This is essential for the base Mamba2 blocks:
//! recurrent terminal states cannot be reconstructed by deleting a rejected
//! token after the fact.

use burn::tensor::{DType, ElementConversion, Int, Tensor, backend::Backend};

use crate::{
    Error, Result,
    cache::{CacheConfig, DecodeStateError, TensorCacheError, TensorDecodeState},
};

use super::{
    NemotronHDecodeState, NemotronHForCausalLm,
    decode::{KV_GROUP, cached_attention_mask},
    validate_token_ids,
};

/// One cached MTP prediction after projection through the shared target head.
#[derive(Debug)]
pub struct NemotronHMtpPrediction<B: Backend> {
    /// F32 vocabulary logits `[1, sequence, vocabulary]`.
    pub logits: Tensor<B, 3>,
    /// Final normalized auxiliary state recurrently fed to the next draft step.
    pub hidden_state: Tensor<B, 3>,
}

impl<B: Backend> NemotronHForCausalLm<B> {
    /// Allocates the batch-one transactional state required by MTP speculation.
    ///
    /// The target and auxiliary banks each reserve `K + 1` tentative slots. A
    /// target cycle may consume the lead token plus all `K` accepted proposals;
    /// the auxiliary stream may need the same number after all-accepted catch-up.
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

    /// Evaluates shifted MTP pairs against the independent incremental KV bank.
    ///
    /// `next_token_ids[t]` is paired with the target or previous-MTP hidden state
    /// immediately preceding it. Outside a transaction this method also accepts
    /// a non-empty prompt chunk; inside one, each call must contain exactly one
    /// pair so every accepted prefix has a physical cache boundary.
    pub fn forward_mtp_cached(
        &self,
        next_token_ids: Tensor<B, 2, Int>,
        target_or_mtp_hidden: Tensor<B, 3>,
        position_ids: Option<Tensor<B, 2, Int>>,
        state: &mut NemotronHDecodeState<B>,
    ) -> Result<NemotronHMtpPrediction<B>> {
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
        if state.tensors.has_active_transaction() && sequence != 1 {
            return Err(Error::Generation(
                "transactional Nemotron-H MTP decoding accepts exactly one token per call".into(),
            ));
        }
        if target_or_mtp_hidden.dims() != [1, sequence, self.config().hidden_size] {
            return Err(Error::shape(
                "nemotron_h.mtp.hidden_state",
                format!("[1, {sequence}, {}]", self.config().hidden_size),
                format!("{:?}", target_or_mtp_hidden.dims()),
            ));
        }
        ensure_mtp_device(
            "next-token IDs",
            &next_token_ids.device(),
            state.tensors.device(),
        )?;
        ensure_mtp_device(
            "hidden state",
            &target_or_mtp_hidden.device(),
            state.tensors.device(),
        )?;
        ensure_mtp_device(
            "parameters",
            &head.eh_proj.weight.val().device(),
            state.tensors.device(),
        )?;
        validate_token_ids(next_token_ids.clone(), self.config().vocab_size)?;

        let committed = state
            .tensors
            .mtp_lengths()
            .and_then(|lengths| lengths.first().copied())
            .ok_or_else(|| Error::Cache("Nemotron-H MTP metadata is not enabled".into()))?;
        let baseline = state
            .tensors
            .mtp_working_lengths()
            .and_then(|lengths| lengths.first().copied())
            .ok_or_else(|| Error::Cache("Nemotron-H MTP working cursor is not enabled".into()))?;
        let end = baseline.checked_add(sequence).ok_or_else(|| {
            Error::Generation("Nemotron-H MTP logical position overflows usize".into())
        })?;
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
        validate_mtp_positions(
            position_ids,
            baseline,
            sequence,
            self.config().max_position_embeddings,
            state.tensors.device(),
        )?;

        let embedding = self.model.embed_tokens.forward(next_token_ids);
        let fused = head.fuse(embedding, target_or_mtp_hidden)?;
        let residual = fused.clone();
        let mut projection = head
            .attention
            .project_self(head.attention_norm.forward(fused))?;
        let new_keys = projection.key.clone();
        let new_values = projection.value.clone();
        let bank = state
            .tensors
            .mtp()
            .ok_or_else(|| Error::Cache("Nemotron-H MTP cache bank is not enabled".into()))?;
        let cache = bank.native_kv(KV_GROUP)?;
        let history = if cache.has_active_staging() {
            cache.read_working_sequence(0, 0, committed)?
        } else {
            cache.read_sequence(0, 0, baseline)?
        };
        if history.retained_positions() != (0..baseline) {
            return Err(Error::Cache(
                "Nemotron-H MTP cache unexpectedly evicted attention history".into(),
            ));
        }
        if let Some((past_keys, past_values)) = history.materialize() {
            projection.key = Tensor::cat(vec![past_keys, projection.key], 2);
            projection.value = Tensor::cat(vec![past_values, projection.value], 2);
        }
        let mask = cached_attention_mask::<B>(
            baseline,
            sequence,
            head.attention.query_heads,
            head.sliding_window,
            &projection.query.device(),
        )?;
        let attention = head.attention.attend(projection, Some(mask), None, false)?;
        let (hidden_state, _) = head.finish(residual + attention)?;
        let logits = self.project_logits(hidden_state.clone())?.cast(DType::F32);
        commit_mtp_kv(&mut state.tensors, baseline, new_keys, new_values, sequence)?;
        Ok(NemotronHMtpPrediction {
            logits,
            hidden_state,
        })
    }

    /// Aligns an empty auxiliary bank with an already-prefilled target prompt.
    ///
    /// For target tokens `x_0..x_(L-1)` and states `h_0..h_(L-1)`, the MTP bank
    /// receives the `L-1` pairs `(x_(i+1), h_i)`. It therefore remains one token
    /// behind the target until a speculative cycle supplies its lead-token pair.
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
        if prompt_token_ids.device() != target_final_hidden_states.device() {
            return Err(Error::BackendCapability(
                "Nemotron-H MTP prompt tokens and target hidden states must share a device".into(),
            ));
        }
        ensure_mtp_device(
            "prompt token IDs",
            &prompt_token_ids.device(),
            state.tensors.device(),
        )?;
        validate_token_ids(prompt_token_ids.clone(), self.config().vocab_size)?;
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
        self.forward_mtp_cached(
            prompt_token_ids.narrow(1, 1, sequence - 1),
            target_final_hidden_states.narrow(1, 0, sequence - 1),
            None,
            state,
        )?;
        Ok(())
    }
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

/// Atomically publishes or stages one complete MTP attention suffix.
fn commit_mtp_kv<B: Backend>(
    state: &mut TensorDecodeState<B>,
    baseline: usize,
    keys: Tensor<B, 4>,
    values: Tensor<B, 4>,
    token_count: usize,
) -> Result<()> {
    let working = state
        .mtp_working_lengths()
        .and_then(|lengths| lengths.first().copied())
        .ok_or_else(|| Error::Cache("Nemotron-H MTP metadata is not enabled".into()))?;
    if baseline != working || keys.dims()[2] != token_count || values.dims()[2] != token_count {
        return Err(Error::Cache(
            "Nemotron-H MTP cache write does not cover its complete logical suffix".into(),
        ));
    }
    if state.has_active_transaction() {
        state
            .mtp_mut()
            .ok_or_else(|| Error::Cache("Nemotron-H MTP cache bank is not enabled".into()))?
            .native_kv_mut(KV_GROUP)?
            .stage_sequence(0, 0, keys, values)?;
        state.advance_mtp(&[token_count])?;
        return Ok(());
    }

    let bank = state
        .mtp()
        .ok_or_else(|| Error::Cache("Nemotron-H MTP cache bank is not enabled".into()))?;
    let cache = bank.native_kv(KV_GROUP)?;
    let write = cache.prepare_sequence_write(0, 0, baseline, keys, values)?;
    let writes = bank.prepare_direct_writes(Some((KV_GROUP, vec![write])), None)?;
    state.advance_mtp(&[token_count])?;
    state
        .mtp_mut()
        .ok_or_else(|| Error::Cache("Nemotron-H MTP cache bank is not enabled".into()))?
        .apply_prepared_direct_writes(writes);
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
