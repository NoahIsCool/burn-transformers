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
mod tests {
    use burn::{
        backend::Flex,
        tensor::{Tensor, TensorData, Tolerance, backend::BackendTypes},
    };

    use crate::{
        CausalLanguageModel, CausalLmInput, DecodeInput, ForwardOptions, LogitsSelection,
        cache::CacheConfig, models::nemotron_h::model::tests::tiny_config,
    };

    use super::*;

    type TestBackend = Flex<f32, i32>;

    /// Prefills the target and returns its complete final hidden sequence.
    fn prefill_target(
        model: &NemotronHForCausalLm<TestBackend>,
        state: &mut NemotronHDecodeState<TestBackend>,
        device: &<TestBackend as BackendTypes>::Device,
    ) -> Tensor<TestBackend, 3> {
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3]], device),
                    token_mask: None,
                    position_ids: None,
                },
                state,
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    hidden_states: true,
                    ..Default::default()
                },
            )
            .unwrap()
            .final_hidden_state
            .unwrap()
    }

    /// Captures every committed physical state kind in the tiny hybrid graph.
    fn physical_cache_snapshot(state: &NemotronHDecodeState<TestBackend>) -> Vec<TensorData> {
        let main = state.tensors.main();
        let target_kv = main.native_kv(KV_GROUP).unwrap();
        let recurrent = main
            .recurrent(super::super::decode::RECURRENT_GROUP)
            .unwrap();
        let mtp_kv = state.tensors.mtp().unwrap().native_kv(KV_GROUP).unwrap();
        vec![
            target_kv.keys(0).unwrap().clone().into_data(),
            target_kv.values(0).unwrap().clone().into_data(),
            recurrent.matrix(0).unwrap().clone().into_data(),
            recurrent.convolution(0).unwrap().clone().into_data(),
            mtp_kv.keys(0).unwrap().clone().into_data(),
            mtp_kv.values(0).unwrap().clone().into_data(),
        ]
    }

    /// Full, chunk-cached, and token-incremental MTP paths are equivalent.
    #[test]
    fn cached_mtp_matches_stateless_and_incremental() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
        let mut chunk_state = model
            .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
            .unwrap();
        let target = prefill_target(&model, &mut chunk_state, &device);
        let shifted_tokens = Tensor::from_ints([[2, 3]], &device);
        let shifted_hidden = target.clone().narrow(1, 0, 2);
        let stateless = model
            .predict_next_token_logits(shifted_tokens.clone(), shifted_hidden.clone())
            .unwrap();
        let chunk = model
            .forward_mtp_cached(
                shifted_tokens.clone(),
                shifted_hidden.clone(),
                None,
                &mut chunk_state,
            )
            .unwrap();
        assert_eq!(chunk_state.tensor_state().lengths(), &[3]);
        assert_eq!(
            chunk_state.tensor_state().mtp_lengths(),
            Some([2].as_slice())
        );
        assert_eq!(chunk_state.mtp_layer_count(), 1);
        chunk
            .logits
            .clone()
            .into_data()
            .assert_approx_eq::<f32>(&stateless.into_data(), Tolerance::absolute(3e-5));

        let mut incremental_state = model
            .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
            .unwrap();
        prefill_target(&model, &mut incremental_state, &device);
        let first = model
            .forward_mtp_cached(
                shifted_tokens.clone().narrow(1, 0, 1),
                shifted_hidden.clone().narrow(1, 0, 1),
                Some(Tensor::from_ints([[0]], &device)),
                &mut incremental_state,
            )
            .unwrap();
        let second = model
            .forward_mtp_cached(
                shifted_tokens.narrow(1, 1, 1),
                shifted_hidden.narrow(1, 1, 1),
                Some(Tensor::from_ints([[1]], &device)),
                &mut incremental_state,
            )
            .unwrap();
        Tensor::cat(vec![first.logits, second.logits], 1)
            .into_data()
            .assert_approx_eq::<f32>(&chunk.logits.into_data(), Tolerance::absolute(3e-5));
        assert_eq!(
            incremental_state.tensor_state().mtp_lengths(),
            Some([2].as_slice())
        );
    }

    /// Prompt alignment installs exactly the shifted `L-1` auxiliary pairs.
    #[test]
    fn prefill_mtp_cache_establishes_one_token_lag() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
        let mut state = model
            .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
            .unwrap();
        let target = prefill_target(&model, &mut state, &device);
        model
            .prefill_mtp_cache(Tensor::from_ints([[1, 2, 3]], &device), target, &mut state)
            .unwrap();
        assert_eq!(state.tensor_state().lengths(), &[3]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));
        let history = state
            .tensor_state()
            .mtp()
            .unwrap()
            .native_kv(KV_GROUP)
            .unwrap()
            .read_sequence(0, 0, 2)
            .unwrap();
        assert_eq!(history.retained_positions(), 0..2);
        assert_eq!(history.materialize().unwrap().0.dims()[2], 2);
    }

    /// Rollback restores target KV, Mamba, convolution, and MTP KV bytes.
    #[test]
    fn transaction_rollback_restores_every_physical_bank() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
        let mut state = model
            .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
            .unwrap();
        let target = prefill_target(&model, &mut state, &device);
        model
            .prefill_mtp_cache(
                Tensor::from_ints([[1, 2, 3]], &device),
                target.clone(),
                &mut state,
            )
            .unwrap();
        let target_keys = state
            .tensors
            .main()
            .native_kv(KV_GROUP)
            .unwrap()
            .keys(0)
            .unwrap()
            .clone()
            .into_data();
        let recurrent = state
            .tensors
            .main()
            .recurrent(super::super::decode::RECURRENT_GROUP)
            .unwrap()
            .matrix(0)
            .unwrap()
            .clone()
            .into_data();
        let convolution = state
            .tensors
            .main()
            .recurrent(super::super::decode::RECURRENT_GROUP)
            .unwrap()
            .convolution(0)
            .unwrap()
            .clone()
            .into_data();
        let mtp_keys = state
            .tensors
            .mtp()
            .unwrap()
            .native_kv(KV_GROUP)
            .unwrap()
            .keys(0)
            .unwrap()
            .clone()
            .into_data();

        let mark = state.mark().unwrap();
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        model
            .forward_mtp_cached(
                Tensor::from_ints([[4]], &device),
                target.narrow(1, 2, 1),
                None,
                &mut state,
            )
            .unwrap();
        assert_eq!(state.tensor_state().lengths(), &[3]);
        assert_eq!(state.tensor_state().working_lengths(), &[4]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));
        assert_eq!(
            state.tensor_state().mtp_working_lengths(),
            Some([3].as_slice())
        );
        state.rollback(mark).unwrap();
        assert_eq!(state.tensor_state().lengths(), &[3]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));
        assert_eq!(
            state
                .tensors
                .main()
                .native_kv(KV_GROUP)
                .unwrap()
                .keys(0)
                .unwrap()
                .clone()
                .into_data(),
            target_keys
        );
        assert_eq!(
            state
                .tensors
                .main()
                .recurrent(super::super::decode::RECURRENT_GROUP)
                .unwrap()
                .matrix(0)
                .unwrap()
                .clone()
                .into_data(),
            recurrent
        );
        assert_eq!(
            state
                .tensors
                .main()
                .recurrent(super::super::decode::RECURRENT_GROUP)
                .unwrap()
                .convolution(0)
                .unwrap()
                .clone()
                .into_data(),
            convolution
        );
        assert_eq!(
            state
                .tensors
                .mtp()
                .unwrap()
                .native_kv(KV_GROUP)
                .unwrap()
                .keys(0)
                .unwrap()
                .clone()
                .into_data(),
            mtp_keys
        );
    }

    /// A partial commit publishes one target/MTP step and drops later snapshots.
    #[test]
    fn transaction_partial_commit_discards_rejected_hybrid_suffix() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
        let mut speculative = model
            .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
            .unwrap();
        let target = prefill_target(&model, &mut speculative, &device);
        model
            .prefill_mtp_cache(
                Tensor::from_ints([[1, 2, 3]], &device),
                target.clone(),
                &mut speculative,
            )
            .unwrap();
        let mark = speculative.mark().unwrap();
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut speculative,
                &ForwardOptions {
                    hidden_states: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let first_mtp = model
            .forward_mtp_cached(
                Tensor::from_ints([[4]], &device),
                target.clone().narrow(1, 2, 1),
                None,
                &mut speculative,
            )
            .unwrap();
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[5]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut speculative,
                &ForwardOptions::default(),
            )
            .unwrap();
        model
            .forward_mtp_cached(
                Tensor::from_ints([[5]], &device),
                first_mtp.hidden_state,
                None,
                &mut speculative,
            )
            .unwrap();
        assert_eq!(speculative.tensor_state().working_lengths(), &[5]);
        assert_eq!(
            speculative.tensor_state().mtp_working_lengths(),
            Some([4].as_slice())
        );
        speculative.commit_prefixes(mark, 1, 1).unwrap();

        let mut reference = model
            .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
            .unwrap();
        let reference_target = prefill_target(&model, &mut reference, &device);
        model
            .prefill_mtp_cache(
                Tensor::from_ints([[1, 2, 3]], &device),
                reference_target.clone(),
                &mut reference,
            )
            .unwrap();
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut reference,
                &ForwardOptions::default(),
            )
            .unwrap();
        model
            .forward_mtp_cached(
                Tensor::from_ints([[4]], &device),
                reference_target.narrow(1, 2, 1),
                None,
                &mut reference,
            )
            .unwrap();

        assert_eq!(speculative.tensor_state().lengths(), &[4]);
        assert_eq!(
            speculative.tensor_state().mtp_lengths(),
            Some([3].as_slice())
        );
        assert_eq!(
            physical_cache_snapshot(&speculative),
            physical_cache_snapshot(&reference)
        );
    }

    /// Constructor and runtime reject unsupported batches, shapes, and positions.
    #[test]
    fn mtp_cache_validation_fails_before_mutation() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
        let ordinary = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        assert_eq!(ordinary.mtp_layer_count(), 0);
        assert!(ordinary.tensor_state().mtp().is_none());
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
        let mut state = model
            .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
            .unwrap();
        prefill_target(&model, &mut state, &device);
        let before = state.tensor_state().mtp_lengths().unwrap().to_vec();
        assert!(
            model
                .forward_mtp_cached(
                    Tensor::from_ints([[2, 3]], &device),
                    Tensor::zeros([1, 1, 8], &device),
                    None,
                    &mut state,
                )
                .is_err()
        );
        assert!(
            model
                .forward_mtp_cached(
                    Tensor::from_ints([[2]], &device),
                    Tensor::zeros([1, 1, 8], &device),
                    Some(Tensor::from_ints([[1]], &device)),
                    &mut state,
                )
                .is_err()
        );
        assert_eq!(state.tensor_state().mtp_lengths(), Some(before.as_slice()));
    }

    /// Trait truncation never leaves the shifted target and MTP cursors divergent.
    #[test]
    fn speculative_state_truncate_is_atomic_and_reusable() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
        let mut state = model
            .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
            .unwrap();
        let target = prefill_target(&model, &mut state, &device);
        model
            .prefill_mtp_cache(Tensor::from_ints([[1, 2, 3]], &device), target, &mut state)
            .unwrap();

        crate::architecture::DecodeState::<TestBackend>::truncate(&mut state, &[3]).unwrap();
        assert_eq!(state.tensor_state().lengths(), &[3]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));
        assert!(
            crate::architecture::DecodeState::<TestBackend>::truncate(&mut state, &[2]).is_err()
        );
        assert_eq!(state.tensor_state().lengths(), &[3]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));

        crate::architecture::DecodeState::<TestBackend>::truncate(&mut state, &[0]).unwrap();
        assert_eq!(state.tensor_state().lengths(), &[0]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([0].as_slice()));
        let target = prefill_target(&model, &mut state, &device);
        model
            .prefill_mtp_cache(Tensor::from_ints([[1, 2, 3]], &device), target, &mut state)
            .unwrap();
        assert_eq!(state.tensor_state().lengths(), &[3]);
        assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));
    }
}
