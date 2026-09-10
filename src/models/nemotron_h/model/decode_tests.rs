use burn::{
    backend::Flex,
    tensor::{Bool, Tensor, Tolerance},
};

use crate::{DecodeState, architecture::TokenMask, models::nemotron_h::model::tests::tiny_config};

use super::*;

type TestBackend = Flex<f32, i32>;

/// Cached chunk/decode execution matches the full stateless continuation.
#[test]
fn cached_prefill_decode_matches_stateless() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    assert_eq!(state.attention_layer_count(), 1);
    assert_eq!(state.mamba_layer_count(), 1);
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cached = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    assert_eq!(state.lengths(), &[4]);
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(3e-5));
}

/// Sliding attention keeps full storage while stateless and cached math agree.
#[test]
fn sliding_window_cached_path_matches_stateless() {
    let device = Default::default();
    let mut config = tiny_config(false);
    config.sliding_window = Some(2);
    let model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4, 5]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cached = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[4, 5]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(3e-5));
}

/// Left padding does not advance recurrent or KV state for semantic tokens.
#[test]
fn left_padded_prefill_matches_stateless_last_token() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let mask = || {
        TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[false, false, true, true, true]].into(),
            &device,
        ))
    };
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 0, 1, 2, 3]], &device),
                token_mask: Some(mask()),
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let cached = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 0, 1, 2, 3]], &device),
                token_mask: Some(mask()),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    assert_eq!(state.lengths(), &[3]);
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(3e-5));
}

/// A failed decode leaves logical, KV, recurrent, and convolution state intact.
#[test]
fn failed_decode_is_atomic_across_cache_kinds() {
    let device = Default::default();
    let mut model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let kv = state
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
        .recurrent(RECURRENT_GROUP)
        .unwrap()
        .matrix(0)
        .unwrap()
        .clone()
        .into_data();
    let convolution = state
        .tensors
        .main()
        .recurrent(RECURRENT_GROUP)
        .unwrap()
        .convolution(0)
        .unwrap()
        .clone()
        .into_data();
    // Fail in the final MoE block, after both earlier Mamba and attention
    // blocks have produced operation-local candidate cache updates.
    model.model.layers[3].moe = None;
    assert!(
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert_eq!(state.lengths(), &[2]);
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
        kv
    );
    assert_eq!(
        state
            .tensors
            .main()
            .recurrent(RECURRENT_GROUP)
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
            .recurrent(RECURRENT_GROUP)
            .unwrap()
            .convolution(0)
            .unwrap()
            .clone()
            .into_data(),
        convolution
    );
}

/// Inactive decode rows are exact no-ops while active peers advance.
#[test]
fn ragged_decode_and_lifecycle_operations_are_safe() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2], [3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[0], [5]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false], [true]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(state.lengths(), &[2, 3]);
    state.reorder(&[1, 0]).unwrap();
    assert_eq!(state.lengths(), &[3, 2]);
    assert!(state.truncate(&[2, 2]).is_err());
    state.truncate(&[0, 0]).unwrap();
    assert_eq!(state.lengths(), &[0, 0]);
    state.reset().unwrap();
}

/// Capacity failures happen before any terminal-state publication.
#[test]
fn capacity_error_preserves_lengths() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 2).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert!(
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert_eq!(state.lengths(), &[2]);
}

/// Cached mask construction rejects windows that cannot enter signed indices.
#[cfg(target_pointer_width = "64")]
#[test]
fn oversized_cached_attention_window_is_rejected() {
    let device = Default::default();
    assert!(cached_attention_mask::<TestBackend>(0, 1, 1, Some(usize::MAX), &device).is_err());
}
