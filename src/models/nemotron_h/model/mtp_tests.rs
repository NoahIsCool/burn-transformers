//! Nemotron-H shifted MTP parity, transaction rollback, and prefix-publication tests.

use burn::{
    backend::Flex,
    tensor::{Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use crate::{
    CausalLanguageModel, CausalLmInput, DecodeInput, ForwardOptions, LogitsSelection,
    cache::CacheConfig, models::nemotron_h::model::tests::tiny_config,
};

use super::*;

/// CPU Flex backend with F32 activations and I32 indices for MTP cache fixtures.
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

/// Captures every *readable* state of the tiny hybrid graph.
///
/// Attention caches contribute each bank's committed prefix `[0, L)`;
/// slots beyond it are unreachable (in-place proposals or abandoned
/// write-ahead keys may remain there), so equality of these snapshots is the
/// observable cache state. Recurrent matrices and convolution histories are
/// committed terminal states and are compared in full.
fn logical_cache_snapshot(state: &NemotronHDecodeState<TestBackend>) -> Vec<TensorData> {
    let main = state.tensors.main();
    let target_kv = main.native_kv(KV_GROUP).unwrap();
    let recurrent = main
        .recurrent(super::super::decode::RECURRENT_GROUP)
        .unwrap();
    let mtp_kv = state.tensors.mtp().unwrap().native_kv(KV_GROUP).unwrap();
    let target_length = state.tensors.lengths()[0];
    let mtp_length = state.tensors.mtp_lengths().unwrap()[0];
    let prefix = |tensor: &Tensor<TestBackend, 4>, length: usize| {
        tensor.clone().narrow(2, 0, length).into_data()
    };
    vec![
        prefix(target_kv.keys(0).unwrap(), target_length),
        prefix(target_kv.values(0).unwrap(), target_length),
        recurrent.matrix(0).unwrap().clone().into_data(),
        recurrent.convolution(0).unwrap().clone().into_data(),
        prefix(mtp_kv.keys(0).unwrap(), mtp_length),
        prefix(mtp_kv.values(0).unwrap(), mtp_length),
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

/// Prompt alignment projects keys and values only (NEM-2).
///
/// No vocabulary projection runs, and the cached pairs are bit-identical to
/// those of the full MTP forward over the same shifted pairs.
#[test]
fn prefill_mtp_cache_projects_only_keys_and_values() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let mut aligned = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    let target = prefill_target(&model, &mut aligned, &device);
    let projections = || super::super::LOGIT_PROJECTIONS.with(std::cell::Cell::get);
    let before = projections();
    model
        .prefill_mtp_cache(
            Tensor::from_ints([[1, 2, 3]], &device),
            target.clone(),
            &mut aligned,
        )
        .unwrap();
    assert_eq!(projections(), before);

    let mut full = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    prefill_target(&model, &mut full, &device);
    model
        .forward_mtp_cached(
            Tensor::from_ints([[2, 3]], &device),
            target.narrow(1, 0, 2),
            None,
            &mut full,
        )
        .unwrap();
    assert_eq!(
        logical_cache_snapshot(&aligned),
        logical_cache_snapshot(&full)
    );
}

/// One-call speculative prefill equals target prefill plus MTP alignment.
///
/// The logits, the hidden state before the lead, and every readable cache
/// value agree with the two-call path, and invalid states are rejected
/// without publishing anything.
#[test]
fn prefill_speculative_matches_the_two_call_path() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let mut manual = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    let options = ForwardOptions {
        hidden_states: true,
        ..Default::default()
    };
    let prefill = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut manual,
            &options,
        )
        .unwrap();
    let hidden = prefill.final_hidden_state.unwrap();
    model
        .prefill_mtp_cache(
            Tensor::from_ints([[1, 2, 3]], &device),
            hidden.clone(),
            &mut manual,
        )
        .unwrap();

    let mut fused = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    let result = model
        .prefill_speculative(Tensor::from_ints([[1, 2, 3]], &device), &mut fused)
        .unwrap();
    assert_eq!(fused.tensor_state().lengths(), &[3]);
    assert_eq!(fused.tensor_state().mtp_lengths(), Some([2].as_slice()));
    assert_eq!(
        result.logits.into_data(),
        prefill.logits.squeeze_dim::<2>(1).into_data()
    );
    assert_eq!(
        result.hidden_state.into_data(),
        hidden.narrow(1, 2, 1).into_data()
    );
    assert_eq!(
        logical_cache_snapshot(&fused),
        logical_cache_snapshot(&manual)
    );

    // A prefilled state, an open transaction, and a model without MTP are rejected.
    assert!(
        model
            .prefill_speculative(Tensor::from_ints([[4]], &device), &mut fused)
            .is_err()
    );
    let mut transactional = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    let mark = transactional.mark().unwrap();
    assert!(
        model
            .prefill_speculative(Tensor::from_ints([[4]], &device), &mut transactional)
            .is_err()
    );
    transactional.rollback(mark).unwrap();
    let single = model
        .prefill_speculative(Tensor::from_ints([[4]], &device), &mut transactional)
        .unwrap();
    assert_eq!(single.hidden_state.dims(), [1, 1, 8]);
    assert_eq!(
        transactional.tensor_state().mtp_lengths(),
        Some([0].as_slice())
    );
    let plain = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let mut plain_state = plain
        .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, &device)
        .unwrap();
    assert!(
        plain
            .prefill_speculative(Tensor::from_ints([[4]], &device), &mut plain_state)
            .is_err()
    );
    assert_eq!(plain_state.tensor_state().lengths(), &[0]);
}

/// Rollback restores every readable target KV, Mamba, convolution, and MTP KV value.
#[test]
fn transaction_rollback_restores_every_readable_bank() {
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
    let baseline = logical_cache_snapshot(&state);

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
    assert_eq!(logical_cache_snapshot(&state), baseline);
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
        logical_cache_snapshot(&speculative),
        logical_cache_snapshot(&reference)
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
    assert!(crate::architecture::DecodeState::<TestBackend>::truncate(&mut state, &[2]).is_err());
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

/// The speculative constructor names zero drafts, a missing MTP stage, and staging overflow.
#[test]
fn speculative_state_constructor_rejects_unusable_requests() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let base = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let cache = || CacheConfig::new(1, 8).unwrap();
    for (error, expected) in [
        (
            model
                .new_speculative_decode_state(cache(), 2, 2, &device)
                .unwrap_err(),
            "Nemotron-H MTP speculative decoding supports batch size one, found 2",
        ),
        (
            model
                .new_speculative_decode_state(cache(), 1, 0, &device)
                .unwrap_err(),
            "Nemotron-H MTP max_draft_tokens must be greater than zero",
        ),
        (
            base.new_speculative_decode_state(cache(), 1, 2, &device)
                .unwrap_err(),
            "Nemotron-H model has no configured next-token prediction stage",
        ),
        (
            model
                .new_speculative_decode_state(cache(), 1, usize::MAX, &device)
                .unwrap_err(),
            "Nemotron-H MTP staging capacity overflows usize",
        ),
        (
            model
                .new_speculative_decode_state(cache(), 1, 8, &device)
                .unwrap_err(),
            "Nemotron-H MTP needs staging capacity 9, exceeding cache capacity 8",
        ),
    ] {
        assert!(
            matches!(&error, Error::Generation(reason) if reason == expected),
            "{error}"
        );
    }
    // The largest draft that fits leaves exactly `K + 1` staging slots.
    let state = model
        .new_speculative_decode_state(cache(), 1, 7, &device)
        .unwrap();
    assert_eq!(state.staging_capacity(), 8);
}

/// Prompt alignment rejects transactions, shapes, tokens, and misaligned or used banks.
#[test]
fn prefill_mtp_cache_rejects_invalid_requests_without_mutation() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    let target = prefill_target(&model, &mut state, &device);
    let prompt = || Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device);

    let mark = state.mark().unwrap();
    let error = model
        .prefill_mtp_cache(prompt(), target.clone(), &mut state)
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::TensorCache(TensorCacheError::TransactionActive {
                operation: "prefill Nemotron-H MTP cache"
            })
        ),
        "{error}"
    );
    state.rollback(mark).unwrap();

    let error = model
        .prefill_mtp_cache(prompt(), target.clone().narrow(1, 0, 2), &mut state)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, expected, actual }
            if tensor == "nemotron_h.mtp.prefill"
            && expected == "token IDs [1, L] and target hidden states [1, L, 8] for L > 0"
            && actual == "tokens=[1, 3], hidden=[1, 2, 8]"),
        "{error}"
    );
    let error = model
        .prefill_mtp_cache(
            Tensor::from_ints([[1, 2, 32]], &device),
            target.clone(),
            &mut state,
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, .. } if tensor == "nemotron_h.input_ids"),
        "{error}"
    );
    let error = model
        .prefill_mtp_cache(
            Tensor::from_ints([[1, 2]], &device),
            target.clone().narrow(1, 0, 2),
            &mut state,
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Cache(reason)
            if reason == "Nemotron-H target cache length must equal MTP prompt length 2, found [3]"),
        "{error}"
    );
    assert_eq!(state.tensor_state().mtp_lengths(), Some([0].as_slice()));

    model
        .prefill_mtp_cache(prompt(), target.clone(), &mut state)
        .unwrap();
    let error = model
        .prefill_mtp_cache(prompt(), target, &mut state)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Cache(reason)
            if reason == "Nemotron-H MTP prefill requires a fresh auxiliary cache"),
        "{error}"
    );
    assert_eq!(state.tensor_state().mtp_lengths(), Some([2].as_slice()));
}

/// A one-token prompt forms no shifted pair, so the auxiliary bank stays empty.
#[test]
fn one_token_prompt_aligns_without_mtp_pairs() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    let hidden = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap()
        .final_hidden_state
        .unwrap();
    model
        .prefill_mtp_cache(Tensor::from_ints([[4]], &device), hidden, &mut state)
        .unwrap();
    assert_eq!(state.tensor_state().lengths(), &[1]);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([0].as_slice()));
}

/// Cached MTP evaluation names a missing stage, bad shapes, tokens, and position tensors.
#[test]
fn forward_mtp_cached_rejects_invalid_requests_without_mutation() {
    let device = Default::default();
    let base = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let mut base_state = base
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let error = base
        .forward_mtp_cached(
            Tensor::from_ints([[1]], &device),
            Tensor::zeros([1, 1, 8], &device),
            None,
            &mut base_state,
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Generation(reason)
            if reason == "Nemotron-H model has no configured next-token prediction stage"),
        "{error}"
    );

    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    prefill_target(&model, &mut state, &device);
    let error = model
        .forward_mtp_cached(
            Tensor::from_ints([[1], [2]], &device),
            Tensor::zeros([2, 1, 8], &device),
            None,
            &mut state,
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, expected, actual }
            if tensor == "nemotron_h.mtp.next_token_ids"
            && expected == "[1, nonzero_sequence] matching a batch-one decode state"
            && actual == "[2, 1]"),
        "{error}"
    );
    let error = model
        .forward_mtp_cached(
            Tensor::from_ints([[1]], &device),
            Tensor::zeros([1, 1, 7], &device),
            None,
            &mut state,
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, expected, actual }
            if tensor == "nemotron_h.mtp.hidden_state"
            && expected == "[1, 1, 8]"
            && actual == "[1, 1, 7]"),
        "{error}"
    );
    let error = model
        .forward_mtp_cached(
            Tensor::from_ints([[32]], &device),
            Tensor::zeros([1, 1, 8], &device),
            None,
            &mut state,
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, .. } if tensor == "nemotron_h.input_ids"),
        "{error}"
    );
    let error = model
        .forward_mtp_cached(
            Tensor::from_ints([[1]], &device),
            Tensor::zeros([1, 1, 8], &device),
            Some(Tensor::from_ints([[0, 1]], &device)),
            &mut state,
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, expected, actual }
            if tensor == "nemotron_h.mtp.position_ids" && expected == "[1, 1]" && actual == "[1, 2]"),
        "{error}"
    );
    assert_eq!(state.tensor_state().mtp_lengths(), Some([0].as_slice()));
}

/// The contiguous-position contract reports overflow, the context limit, and drift.
#[test]
fn mtp_position_contract_reports_overflow_limit_and_drift() {
    let device = Default::default();
    let error =
        validate_mtp_positions::<TestBackend>(None, usize::MAX, 1, 32, &device).unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, reason }
            if field == "nemotron_h.mtp.position_ids" && reason == "position range overflows usize"),
        "{error}"
    );
    let error = validate_mtp_positions::<TestBackend>(None, 30, 3, 32, &device).unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, expected, actual }
            if tensor == "nemotron_h.mtp.position_ids"
            && expected == "positions below 32"
            && actual == "range 30..33"),
        "{error}"
    );
    validate_mtp_positions::<TestBackend>(None, 30, 2, 32, &device).unwrap();
    validate_mtp_positions::<TestBackend>(
        Some(Tensor::from_ints([[4, 5]], &device)),
        4,
        2,
        32,
        &device,
    )
    .unwrap();
    let error = validate_mtp_positions::<TestBackend>(
        Some(Tensor::from_ints([[4, 6]], &device)),
        4,
        2,
        32,
        &device,
    )
    .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, expected, actual }
            if tensor == "nemotron_h.mtp.position_ids"
            && expected == "contiguous cache positions 4..6"
            && actual == "a different explicit position sequence"),
        "{error}"
    );
}

/// Device checks name the mismatched MTP input.
#[test]
fn mtp_device_check_names_the_mismatched_input() {
    ensure_mtp_device("hidden state", &1, &1).unwrap();
    let error = ensure_mtp_device("hidden state", &1, &2).unwrap_err();
    assert!(
        matches!(&error, Error::BackendCapability(reason)
            if reason == "Nemotron-H MTP hidden state must use the decode-state device"),
        "{error}"
    );
}

/// Draft pairs accept only in-vocabulary tokens and never publish.
#[test]
fn draft_pairs_reject_out_of_vocabulary_tokens_and_publish_nothing() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    let target = prefill_target(&model, &mut state, &device);
    let mut drafts = DraftPairs::new();
    let error = model
        .draft_mtp(32, target.clone().narrow(1, 2, 1), &mut drafts, &mut state)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, expected, actual }
            if tensor == "nemotron_h.input_ids" && expected == "a token in 0..32" && actual == "32"),
        "{error}"
    );
    assert_eq!(drafts.len(), 0);
    let first = model
        .draft_mtp(4, target.narrow(1, 2, 1), &mut drafts, &mut state)
        .unwrap();
    assert_eq!(first.logits.dims(), [1, 1, 32]);
    model
        .draft_mtp(5, first.hidden_state, &mut drafts, &mut state)
        .unwrap();
    assert_eq!(drafts.len(), 2);
    assert_eq!(state.tensor_state().mtp_lengths(), Some([0].as_slice()));
}

/// A step longer than the remaining MTP capacity fails before any publication.
#[test]
fn mtp_step_beyond_capacity_is_rejected() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 8).unwrap(), 1, 2, &device)
        .unwrap();
    prefill_target(&model, &mut state, &device);
    let error = model
        .forward_mtp_cached(
            Tensor::from_ints([[1, 2, 3, 4, 5, 6, 7, 8, 9]], &device),
            Tensor::zeros([1, 9, 8], &device),
            None,
            &mut state,
        )
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::TensorCache(TensorCacheError::Metadata(
                DecodeStateError::CapacityExceeded {
                    sequence: 0,
                    requested: 9,
                    capacity: 8
                }
            ))
        ),
        "{error}"
    );
    assert_eq!(state.tensor_state().mtp_lengths(), Some([0].as_slice()));
}

/// With an active window, cached MTP attention equals the windowed stateless head.
#[test]
fn windowed_cached_mtp_matches_the_stateless_head() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(
        crate::models::nemotron_h::NemotronHConfig {
            sliding_window: Some(2),
            ..tiny_config(true)
        },
        &device,
    )
    .unwrap();
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    let target = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4, 5]], &device),
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
    let shifted = Tensor::<TestBackend, 2, Int>::from_ints([[2, 3, 4, 5]], &device);
    let hidden = target.narrow(1, 0, 4);
    let stateless = model
        .predict_next_token_logits(shifted.clone(), hidden.clone())
        .unwrap();
    // Three pairs at once (window active at the step's end), then one more.
    let chunk = model
        .forward_mtp_cached(
            shifted.clone().narrow(1, 0, 3),
            hidden.clone().narrow(1, 0, 3),
            None,
            &mut state,
        )
        .unwrap();
    let last = model
        .forward_mtp_cached(
            shifted.narrow(1, 3, 1),
            hidden.narrow(1, 3, 1),
            None,
            &mut state,
        )
        .unwrap();
    assert_eq!(state.tensor_state().mtp_lengths(), Some([4].as_slice()));
    Tensor::cat(vec![chunk.logits, last.logits], 1)
        .into_data()
        .assert_approx_eq::<f32>(&stateless.into_data(), Tolerance::absolute(3e-5));
}

/// The key/value-only projection refuses a head with key normalization.
#[test]
fn pair_projection_refuses_key_normalization() {
    let device = Default::default();
    let mut model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    model.next_token_head.as_mut().unwrap().attention.key_norm = Some(
        crate::normalization::RmsNormConfig::new(4)
            .init(&device)
            .unwrap(),
    );
    let mut state = model
        .new_speculative_decode_state(CacheConfig::new(1, 16).unwrap(), 1, 2, &device)
        .unwrap();
    let target = prefill_target(&model, &mut state, &device);
    let error = model
        .prefill_mtp_cache(Tensor::from_ints([[1, 2, 3]], &device), target, &mut state)
        .unwrap_err();
    assert!(
        matches!(&error, Error::BackendCapability(reason)
            if reason == "Nemotron-H MTP key/value projection does not support key normalization"),
        "{error}"
    );
    assert_eq!(state.tensor_state().mtp_lengths(), Some([0].as_slice()));
}
