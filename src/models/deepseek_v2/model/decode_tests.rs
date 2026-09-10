use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::{
    DecodeState, LogitsSelection,
    models::deepseek_v2::model::tests_support::{initialized_model, tiny_config},
};

type TestBackend = Flex<f32, i32>;

/// Returns a deterministic tiny model and device.
fn tiny_model() -> (
    DeepseekV2ForCausalLm<TestBackend>,
    <TestBackend as BackendTypes>::Device,
) {
    let device = Default::default();
    (initialized_model(tiny_config(), &device), device)
}

/// Builds the standard input wrapper for one dense batch row.
fn input(tokens: Tensor<TestBackend, 2, Int>) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: tokens,
        token_mask: None,
        position_ids: None,
    }
}

/// Single-token decode uses absorbed MLA and matches full stateless logits.
#[test]
fn prefill_then_absorbed_decode_matches_full_forward() {
    let (model, device) = tiny_model();
    let full = model
        .forward(
            input(Tensor::from_ints([[1, 2, 3, 4]], &device)),
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            input(Tensor::from_ints([[1, 2, 3]], &device)),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(state.lengths(), &[3]);
    assert_eq!(
        state.last_execution_path(),
        Some(MlaExecutionPath::ReferenceUnabsorbed)
    );
    let cached = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    assert_eq!(state.lengths(), &[4]);
    assert_eq!(
        state.last_execution_path(),
        Some(MlaExecutionPath::AbsorbedDecode)
    );
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
}

/// Multi-token chunks stay on reference MLA and preserve causal parity.
#[test]
fn chunked_decode_matches_full_forward() {
    let (model, device) = tiny_model();
    let full = model
        .forward(
            input(Tensor::from_ints([[1, 2, 3, 4, 5]], &device)),
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap()
        .logits
        .slice([0..1, 2..5, 0..32]);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            input(Tensor::from_ints([[1, 2]], &device)),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cached = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3, 4, 5]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap()
        .logits;
    assert_eq!(
        state.last_execution_path(),
        Some(MlaExecutionPath::ReferenceUnabsorbed)
    );
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
}

/// Physical storage contains only normalized latent plus one rotary key.
#[test]
fn compressed_cache_geometry_and_byte_count_are_exact() {
    let (model, device) = tiny_model();
    let state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    assert_eq!(state.scalars_per_layer_token(), 7);
    assert_eq!(state.layer_storage(0).unwrap().dims(), [2, 8, 7]);
    assert_eq!(state.bytes_per_token(), 7 * 2 * 4);
    assert_eq!(state.allocated_bytes(), 2 * 8 * 7 * 2 * 4);
    let conventional_kv_scalars = 2 * (5 + 4);
    assert!(state.scalars_per_layer_token() < conventional_kv_scalars);
}

/// Ragged prompts and inactive decode rows preserve independent lengths.
#[test]
fn ragged_batch_and_inactive_rows_are_supported() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3], [4, 5, 0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true, true], [true, true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(state.lengths(), &[3, 2]);
    let before_inactive = state.layer_storage(0).unwrap().into_data();
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[6], [0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true], [false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                router_logits: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(state.lengths(), &[4, 2]);
    assert!(
        output
            .logits
            .slice([1..2, 0..1, 0..32])
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0)
    );
    let after = state.layer_storage(0).unwrap();
    let before = Tensor::<TestBackend, 3>::from_data(before_inactive, &device);
    assert_eq!(
        after.slice([1..2, 0..8, 0..7]).into_data(),
        before.slice([1..2, 0..8, 0..7]).into_data()
    );
}

/// Reorder permits duplication, truncate clears suffixes, and reset zeros all bytes.
#[test]
fn lifecycle_operations_update_metadata_and_storage() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3], [4, 5, 0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true, true], [true, true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    state.reorder(&[1, 1]).unwrap();
    assert_eq!(state.lengths(), &[2, 2]);
    let storage = state.layer_storage(0).unwrap();
    assert_eq!(
        storage.clone().slice([0..1, 0..8, 0..7]).into_data(),
        storage.slice([1..2, 0..8, 0..7]).into_data()
    );
    state.truncate(&[1, 0]).unwrap();
    assert_eq!(state.lengths(), &[1, 0]);
    let values = state.layer_storage(0).unwrap().into_data();
    let values = values.as_slice::<f32>().unwrap();
    assert!(values[7..].iter().all(|&value| value == 0.0));
    state.reset().unwrap();
    assert_eq!(state.lengths(), &[0, 0]);
    assert!(
        state
            .layer_storage(0)
            .unwrap()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0)
    );
}

/// Capacity, token, and diagnostic errors leave lengths and bytes unchanged.
#[test]
fn failures_are_byte_and_metadata_atomic() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 3).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            input(Tensor::from_ints([[1, 2]], &device)),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let before = state.layer_storage(0).unwrap().into_data();
    let before_path = state.last_execution_path();
    assert!(
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert!(
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[32]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert!(
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions {
                    attentions: true,
                    ..Default::default()
                },
            )
            .is_err()
    );
    assert_eq!(state.lengths(), &[2]);
    assert_eq!(state.last_execution_path(), before_path);
    assert_eq!(state.layer_storage(0).unwrap().into_data(), before);
}

/// Explicit positions are range-checked before any cache write.
#[test]
fn explicit_position_errors_are_atomic() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let error = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[32]], &device)),
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("position_ids"));
    assert_eq!(state.lengths(), &[0]);
}

/// Query-LoRA uses the same compressed cache and cached parity contract.
#[test]
fn query_lora_cached_path_matches_stateless() {
    let device = Default::default();
    let mut config = tiny_config();
    config.q_lora_rank = Some(4);
    let model = initialized_model(config, &device);
    let full = model
        .forward(
            input(Tensor::from_ints([[1, 2, 3]], &device)),
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            input(Tensor::from_ints([[1, 2]], &device)),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cached = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3]], &device),
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
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
}

/// Cache construction and lifecycle reject incompatible dimensions safely.
#[test]
fn cache_configuration_and_lifecycle_errors_are_contextual() {
    let (model, device) = tiny_model();
    assert!(
        model
            .new_decode_state(CacheConfig::new(2, 8).unwrap(), 1, &device)
            .is_err()
    );
    assert!(
        model
            .new_decode_state(
                CacheConfig::new(1, 8)
                    .unwrap()
                    .with_sliding_window(4)
                    .unwrap(),
                1,
                &device,
            )
            .is_err()
    );
    assert!(
        model
            .new_decode_state(CacheConfig::new(1, 33).unwrap(), 1, &device)
            .is_err()
    );
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    assert!(state.reorder(&[0]).is_err());
    assert!(state.reorder(&[0, 2]).is_err());
    assert!(state.truncate(&[0]).is_err());
    assert!(state.truncate(&[1, 0]).is_err());
    assert!(state.layer_storage(2).is_err());
}

/// A state created for different MLA geometry is rejected without mutation.
#[test]
fn decode_state_model_mismatch_is_atomic() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let mut other_config = tiny_config();
    other_config.kv_lora_rank = 6;
    let other = initialized_model(other_config, &device);
    let before = state.layer_storage(0).unwrap().into_data();
    assert!(
        other
            .prefill(
                input(Tensor::from_ints([[1]], &device)),
                &mut state,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert_eq!(state.lengths(), &[0]);
    assert_eq!(state.layer_storage(0).unwrap().into_data(), before);
}
