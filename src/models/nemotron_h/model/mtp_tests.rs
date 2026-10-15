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
