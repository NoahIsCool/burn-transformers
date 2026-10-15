//! Muse cache lifecycle, failure atomicity, and stateless-versus-cached parity tests.

use burn::{
    backend::Flex,
    tensor::{Bool, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::{
    DecodeState, TokenMask, attention::AttentionExecutionPolicy, checkpoint::TensorSink,
    models::muse_glimmer::MuseGlimmerCheckpointOptions,
};

/// Deterministic CPU backend using F32 values and I32 token indices.
type TestBackend = Flex<f32, i32>;

/// Materializes every public KV buffer so failures are compared against host bytes.
fn contract_cache_snapshot(state: &MuseGlimmerDecodeState<TestBackend>) -> Vec<TensorData> {
    let bank = state.tensor_state().main();
    let mut data = Vec::new();
    for group in 0..bank.native_kv_count() {
        let cache = bank.native_kv(group).unwrap();
        for layer in 0..cache.spec().layers() {
            data.push(cache.keys(layer).unwrap().to_data());
            data.push(cache.values(layer).unwrap().to_data());
        }
    }
    data
}

/// Compares finite public outputs with an absolute F32 tolerance of `1e-4`.
fn contract_close(actual: Tensor<TestBackend, 3>, expected: Tensor<TestBackend, 3>) {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.to_data().to_vec::<f32>().unwrap();
    let expected = expected.to_data().to_vec::<f32>().unwrap();
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
    }
}

/// Public mask, position and output failures preserve both sliding and full caches.
#[test]
fn experimental_contract_public_rejections_preserve_muse_history() {
    let (model, device) = tiny_model();
    let error = model
        .new_decode_state(CacheConfig::new(1, 33).unwrap(), 1, &device)
        .unwrap_err();
    assert!(matches!(error, Error::Configuration { field, reason }
        if field == "muse_glimmer.decode_state.capacity"
        && reason == "33 exceeds max_position_embeddings 32"));
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
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
    let baseline = contract_cache_snapshot(&state);
    assert!(baseline.iter().all(|tensor| {
        tensor
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .any(|value| value != 0.0)
    }));
    for case in 0..7 {
        let mut input = CausalLmInput {
            token_ids: Tensor::from_ints([[3]], &device),
            token_mask: None,
            position_ids: None,
        };
        let mut options = ForwardOptions::default();
        let (tensor, expected, actual) = match case {
            0 => {
                input.token_mask = Some(TokenMask::new(Tensor::from_bool(
                    [[true, true]].into(),
                    &device,
                )));
                ("muse_glimmer.token_mask", "[1, 1]", "[1, 2]")
            }
            1 => {
                input.position_ids = Some(Tensor::from_ints([[2, 3]], &device));
                ("muse_glimmer.position_ids", "[1, 1]", "[1, 2]")
            }
            2 | 3 => {
                input.position_ids = Some(Tensor::from_ints(
                    [[if case == 2 { -1 } else { 32 }]],
                    &device,
                ));
                (
                    "muse_glimmer.position_ids",
                    "position values in 0..32",
                    if case == 2 {
                        "observed range -1..=-1"
                    } else {
                        "observed range 32..=32"
                    },
                )
            }
            4 | 5 => {
                options.logits =
                    crate::LogitsSelection::Positions(if case == 4 { vec![] } else { vec![1] });
                (
                    "forward_options.logits.positions",
                    "a non-empty list of indices below 1",
                    if case == 4 { "[]" } else { "[1]" },
                )
            }
            _ => {
                input.token_mask =
                    Some(TokenMask::new(Tensor::from_bool([[false]].into(), &device)));
                (
                    "muse_glimmer.token_mask",
                    "at least one valid token in every prefill row",
                    "a fully masked prompt row",
                )
            }
        };
        let error = if case == 6 {
            model.prefill(input, &mut state, &options)
        } else {
            model.decode(
                DecodeInput {
                    token_ids: input.token_ids,
                    token_mask: input.token_mask,
                    position_ids: input.position_ids,
                },
                &mut state,
                &options,
            )
        }
        .unwrap_err();
        assert!(
            matches!(error, Error::Shape { tensor: found, expected: expected_found, actual: actual_found }
            if found == tensor && expected_found == expected && actual_found == actual)
        );
        assert_eq!(state.lengths(), &[2]);
        assert_eq!(contract_cache_snapshot(&state), baseline);
    }
    let error = model
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
        .unwrap_err();
    assert!(matches!(error, Error::BackendCapability(reason)
        if reason == "Muse-Glimmer cached attention-probability capture is unavailable"));
    assert_eq!(state.lengths(), &[2]);
    assert_eq!(contract_cache_snapshot(&state), baseline);
    let actual = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let expected = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    contract_close(actual.logits, expected.logits);
    assert_eq!(state.lengths(), &[3]);
}

/// Explicit positions are compacted with tokens; hidden diagnostics scatter only valid columns.
#[test]
fn experimental_contract_muse_hidden_positions_and_inactive_rows() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1], [2]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[2], [4]], &device)),
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let options = ForwardOptions {
        logits: crate::LogitsSelection::All,
        hidden_states: true,
        ..Default::default()
    };
    let actual = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[31, 3, 31], [31, 31, 31]], &device),
                token_mask: Some(TokenMask::new(Tensor::from_bool(
                    [[false, true, false], [false, false, false]].into(),
                    &device,
                ))),
                position_ids: Some(Tensor::from_ints([[-1, 8, 99], [-1, 99, 100]], &device)),
            },
            &mut state,
            &options,
        )
        .unwrap();
    let expected = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 3]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[2, 8]], &device)),
            },
            &options,
        )
        .unwrap();
    assert_eq!(state.lengths(), &[2, 1]);
    assert_eq!(actual.hidden_states.len(), 6);
    for (actual, expected) in actual.hidden_states.into_iter().zip(expected.hidden_states) {
        assert_eq!(actual.dims(), [2, 3, 8]);
        contract_close(
            actual.clone().slice([0..1, 1..2, 0..8]),
            expected.narrow(1, 1, 1),
        );
        let values = actual.to_data().to_vec::<f32>().unwrap();
        for (index, value) in values.into_iter().enumerate() {
            if !(8..16).contains(&index) {
                assert_eq!(value, 0.0);
            }
        }
    }
    contract_close(
        actual.logits.slice([0..1, 1..2, 0..32]),
        expected.logits.narrow(1, 1, 1),
    );
    state.reorder(&[1, 0]).unwrap();
    let actual = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[4], [5]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[9], [10]], &device)),
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    for (row, tokens, positions) in [
        (0, vec![2, 4], vec![4, 9]),
        (1, vec![1, 3, 5], vec![2, 8, 10]),
    ] {
        let length = tokens.len();
        let expected = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_data(TensorData::new(tokens, [1, length]), &device),
                    token_mask: None,
                    position_ids: Some(Tensor::from_data(
                        TensorData::new(positions, [1, length]),
                        &device,
                    )),
                },
                &ForwardOptions::default(),
            )
            .unwrap();
        contract_close(actual.logits.clone().narrow(0, row, 1), expected.logits);
    }
    assert_eq!(state.lengths(), &[2, 3]);
}

/// Four-layer miniature preserving local/local/local/global scheduling.
fn tiny_config() -> MuseGlimmerTextConfig {
    MuseGlimmerTextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        sliding_window: Some(3),
        bos_token_id: None,
        eos_token_id: None,
        ..Default::default()
    }
}

/// One full-attention layer whose explicit head width differs from residual width.
fn full_only_config() -> MuseGlimmerTextConfig {
    MuseGlimmerTextConfig {
        hidden_size: 10,
        num_hidden_layers: 1,
        num_attention_heads: 3,
        num_key_value_heads: 1,
        head_dim: 4,
        sliding_window: None,
        layer_types: vec![MuseGlimmerLayerType::FullAttention],
        layer_rope_theta: vec![0.0],
        ..tiny_config()
    }
}

/// Installs stable tiny tensors for cached-versus-stateless comparisons.
fn initialized_model<B: Backend>(
    config: MuseGlimmerTextConfig,
    device: &B::Device,
) -> MuseGlimmerForCausalLm<B> {
    let contract = config
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .unwrap();
    let mut model = MuseGlimmerForCausalLm::init(config, device).unwrap();
    for (tensor_index, expected) in contract.expected_tensors().iter().enumerate() {
        let shape = expected
            .shape
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        let elements = shape.iter().product();
        let values = (0..elements)
            .map(|element| {
                let code = (element + tensor_index * 11) % 31;
                (code as f32 - 15.0) * 0.002
            })
            .collect::<Vec<_>>();
        TensorSink::write_tensor(&mut model, &expected.name, TensorData::new(values, shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    model
}

/// Returns one initialized miniature and device.
fn tiny_model() -> (
    MuseGlimmerForCausalLm<TestBackend>,
    <TestBackend as BackendTypes>::Device,
) {
    let device = Default::default();
    (initialized_model(tiny_config(), &device), device)
}

/// Local layers use one mirrored group while the final NoPE layer stays full.
#[test]
fn cache_layout_separates_ring_and_full_storage() {
    let (model, device) = tiny_model();
    let state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    assert_eq!(state.tensor_state().main().native_kv_count(), 2);
    assert_eq!(state.sliding_window(), Some(3));
    assert_eq!(
        state.layer_cache(0),
        Some(MuseGlimmerLayerCache::Sliding { group: 0, layer: 0 })
    );
    assert_eq!(
        state.layer_cache(3),
        Some(MuseGlimmerLayerCache::Full { group: 1, layer: 0 })
    );
    assert_eq!(
        state
            .tensor_state()
            .main()
            .native_kv(0)
            .unwrap()
            .committed_time_slots(),
        6
    );
    assert_eq!(
        state
            .tensor_state()
            .main()
            .native_kv(1)
            .unwrap()
            .committed_time_slots(),
        8
    );
}

/// Full-only models allocate no ring metadata and retain cached/stateless parity.
#[test]
fn full_only_cache_needs_no_sliding_window() {
    let device = <TestBackend as BackendTypes>::Device::default();
    let model = initialized_model::<TestBackend>(full_only_config(), &device);
    let stateless = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
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
    assert_eq!(state.sliding_window(), None);
    assert!(
        state
            .tensor_state()
            .main()
            .config()
            .sliding_window()
            .is_none()
    );
    assert_eq!(state.tensor_state().main().native_kv_count(), 1);
    assert_eq!(
        state.layer_cache(0),
        Some(MuseGlimmerLayerCache::Full { group: 0, layer: 0 })
    );
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
        .assert_approx_eq::<f32>(&stateless.into_data(), Tolerance::absolute(3e-4));
}

/// A prompt followed by one token agrees with one-shot execution.
#[test]
fn stateless_prefill_decode_parity() {
    let (model, device) = tiny_model();
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
                token_ids: Tensor::from_ints([[4]], &device),
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
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(3e-4));
    assert_eq!(state.lengths(), &[4]);
}

/// Multi-query cached prefill uses the same logits under bounded chunking.
#[test]
fn bounded_cached_prefill_matches_unbounded_execution() {
    let (model, device) = tiny_model();
    let unbounded = model
        .clone()
        .with_attention_execution_policy(AttentionExecutionPolicy::Unbounded)
        .expect("unbounded fixture");
    let bounded = model
        .with_attention_execution_policy(AttentionExecutionPolicy::query_chunked(2, 20))
        .expect("two-query chunks");
    let mut unbounded_state = unbounded
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let mut bounded_state = bounded
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let input = || CausalLmInput {
        token_ids: Tensor::from_ints([[1, 2, 3, 4, 5]], &device),
        token_mask: None,
        position_ids: None,
    };
    let options = ForwardOptions {
        logits: crate::LogitsSelection::All,
        ..Default::default()
    };
    let expected = unbounded
        .prefill(input(), &mut unbounded_state, &options)
        .expect("unbounded prefill")
        .logits;
    let actual = bounded
        .prefill(input(), &mut bounded_state, &options)
        .expect("bounded prefill")
        .logits;
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(2e-6));
    assert_eq!(bounded_state.lengths(), unbounded_state.lengths());
}

/// Interior holes and ordinary padding retain semantic cached/stateless parity.
#[test]
fn semantic_padding_patterns_match_stateless_prefill_and_decode() {
    let (model, device) = tiny_model();
    let prompt_tokens = Tensor::from_ints(
        [[1, 2, 0, 3, 4], [0, 0, 5, 6, 7], [8, 9, 10, 0, 0]],
        &device,
    );
    let prompt_mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [
            [true, true, false, true, true],
            [false, false, true, true, true],
            [true, true, true, false, false],
        ]
        .into(),
        &device,
    ));
    let stateless_prompt = model
        .forward(
            CausalLmInput {
                token_ids: prompt_tokens.clone(),
                token_mask: Some(prompt_mask.clone()),
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(3, 8).unwrap(), 3, &device)
        .unwrap();
    let cached_prompt = model
        .prefill(
            CausalLmInput {
                token_ids: prompt_tokens,
                token_mask: Some(prompt_mask),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    cached_prompt
        .into_data()
        .assert_approx_eq::<f32>(&stateless_prompt.into_data(), Tolerance::absolute(3e-4));
    assert_eq!(state.lengths(), &[4, 3, 3]);

    let cached_suffix = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[11], [12], [13]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let stateless_suffix = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints(
                    [
                        [1, 2, 0, 3, 4, 11],
                        [0, 0, 5, 6, 7, 12],
                        [8, 9, 10, 0, 0, 13],
                    ],
                    &device,
                ),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [
                        [true, true, false, true, true, true],
                        [false, false, true, true, true, true],
                        [true, true, true, false, false, true],
                    ]
                    .into(),
                    &device,
                ))),
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    cached_suffix
        .into_data()
        .assert_approx_eq::<f32>(&stateless_suffix.into_data(), Tolerance::absolute(3e-4));
    assert_eq!(state.lengths(), &[5, 4, 4]);
}

/// Multi-token cached chunks preserve causality within the appended suffix.
#[test]
fn chunked_decode_matches_one_shot_suffix() {
    let (model, device) = tiny_model();
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: crate::LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap()
        .logits
        .slice([0..1, 2..4, 0..32]);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
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
    let chunk = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: crate::LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap()
        .logits;
    chunk
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(3e-4));
}

/// Cached masks use absolute time and exclude exactly the left window boundary.
#[test]
fn cached_mask_tracks_absolute_time_and_local_boundary() {
    let device = Default::default();
    let local = cached_attention_mask::<TestBackend>(4, 2, 1, 5, 1, Some(3), &device)
        .unwrap()
        .reshape([2, 5])
        .into_data()
        .to_vec::<bool>()
        .unwrap();
    assert_eq!(
        local,
        [
            true, false, false, false, true, true, true, false, false, false
        ]
    );

    let full = cached_attention_mask::<TestBackend>(4, 2, 0, 6, 1, None, &device)
        .unwrap()
        .reshape([2, 6])
        .into_data()
        .to_vec::<bool>()
        .unwrap();
    assert_eq!(
        full,
        [
            false, false, false, false, false, true, false, false, false, false, false, false,
        ]
    );
}

/// Mirrored rings wrap while cached logits retain stateless local-window parity.
#[test]
fn sliding_ring_wrap_matches_stateless_attention() {
    let (model, device) = tiny_model();
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4, 5, 6]], &device),
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
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
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
                token_ids: Tensor::from_ints([[5, 6]], &device),
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
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(3e-4));
    let retained = state
        .tensor_state()
        .main()
        .native_kv(0)
        .unwrap()
        .read_sequence(0, 0, 6)
        .unwrap()
        .retained_positions();
    assert_eq!(retained, 3..6);
    assert_eq!(
        state
            .tensor_state()
            .main()
            .native_kv(1)
            .unwrap()
            .read_sequence(0, 0, 6)
            .unwrap()
            .retained_positions(),
        0..6
    );
}

/// Ragged prefill and inactive decode rows preserve independent lengths.
#[test]
fn semantic_masks_compact_rows_and_allow_inactive_decode() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2], [3, 0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true], [true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(state.lengths(), &[2, 1]);
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[4, 5], [0, 0]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true], [false, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: crate::LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(state.lengths(), &[4, 1]);
    assert!(
        output
            .logits
            .slice([1..2, 0..2, 0..32])
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0)
    );
}

/// A later-layer error leaves earlier pending writes and logical lengths untouched.
#[test]
fn recoverable_layer_failure_is_cache_atomic() {
    let (mut model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let before = state
        .tensor_state()
        .main()
        .native_kv(0)
        .unwrap()
        .keys(0)
        .unwrap()
        .clone()
        .into_data();
    model.model.layers[3].self_attn.hidden_size = 7;
    let error = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("later layer must fail");
    assert!(error.to_string().contains("attention_input"));
    assert_eq!(state.lengths(), &[0]);
    assert_eq!(
        before,
        state
            .tensor_state()
            .main()
            .native_kv(0)
            .unwrap()
            .keys(0)
            .unwrap()
            .clone()
            .into_data()
    );
}

/// Reorder, safe truncate, reset, and capacity failures use shared lifecycle rules.
#[test]
fn decode_state_lifecycle_and_boundaries_are_checked() {
    let (model, device) = tiny_model();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 4).unwrap(), 2, &device)
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
    state.reorder(&[1, 1]).unwrap();
    assert_eq!(state.lengths(), &[2, 2]);
    state.truncate(&[1, 2]).unwrap();
    assert_eq!(state.lengths(), &[1, 2]);
    let error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[5, 6, 7], [5, 6, 7]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("capacity overflow must be preflighted");
    assert!(error.to_string().contains("capacity"));
    assert_eq!(state.lengths(), &[1, 2]);
    state.reset().unwrap();
    assert_eq!(state.lengths(), &[0, 0]);
}

/// Builds the complete cached mask only for compact unit-test fixtures.
///
/// Production cached execution calls the range-aware builder in the parent
/// module so long prefills never allocate this full query-by-key plane.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn cached_attention_mask<B: Backend>(
    baseline: usize,
    query_sequence: usize,
    key_start: usize,
    key_sequence: usize,
    heads: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Result<Tensor<B, 4, Bool>> {
    let query_end = baseline.checked_add(query_sequence).ok_or_else(|| {
        Error::configuration(
            "muse_glimmer.attention_mask",
            "query length overflows usize",
        )
    })?;
    let key_end = key_start.checked_add(key_sequence).ok_or_else(|| {
        Error::configuration("muse_glimmer.attention_mask", "key length overflows usize")
    })?;
    if key_end != query_end {
        return Err(Error::shape(
            "muse_glimmer.cached_keys",
            format!("chronological end {query_end}"),
            format!("end {key_end}"),
        ));
    }
    let positions = |start: usize, end: usize| {
        Tensor::<B, 1, Int>::arange(start as i64..end as i64, device).reshape([1, end - start])
    };
    Ok(crate::mla::PositionalVisibility::cached(
        positions(baseline, query_end),
        None,
        positions(key_start, key_end),
        None,
        vec![key_sequence; query_sequence],
    )
    .with_window(window)
    .exclusion(1, heads, 0..query_sequence, 0..key_sequence))
}

/// Installs the tiny weights scaled by `scale`, so logits are large enough to compare.
fn scaled_model(scale: f32) -> MuseGlimmerForCausalLm<TestBackend> {
    use burn::module::{Module, ModuleMapper, Param};

    /// Multiplies every floating parameter by a constant.
    struct Scale(
        /// Factor applied to every floating parameter.
        f32,
    );

    impl ModuleMapper<TestBackend> for Scale {
        /// Scales one parameter, leaving norms (whose scale is `1 + w`) near one.
        fn map_float<const D: usize>(
            &mut self,
            param: Param<Tensor<TestBackend, D>>,
        ) -> Param<Tensor<TestBackend, D>> {
            let factor = self.0;
            param.map(|tensor| tensor * factor)
        }
    }
    let (model, _) = tiny_model();
    model.map(&mut Scale(scale))
}

/// One stateful call: rectangular tokens and validity per row.
struct Step {
    /// Token IDs `[B][S]`.
    tokens: Vec<Vec<i64>>,
    /// Validity `[B][S]`, or `None` when every token is valid.
    mask: Option<Vec<Vec<bool>>>,
}

/// Requests every logit and every hidden state.
fn all_outputs() -> ForwardOptions {
    ForwardOptions {
        logits: crate::LogitsSelection::All,
        hidden_states: true,
        ..Default::default()
    }
}

/// Runs one step on `state`, as prefill when the state is empty.
fn run_step(
    model: &MuseGlimmerForCausalLm<TestBackend>,
    state: &mut MuseGlimmerDecodeState<TestBackend>,
    tokens: Vec<i64>,
    mask: Option<Vec<bool>>,
    shape: [usize; 2],
) -> CausalLmOutput<TestBackend> {
    let device = Default::default();
    let token_ids = Tensor::from_data(TensorData::new(tokens, shape), &device);
    let token_mask = mask.map(|mask| {
        TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_data(
            TensorData::new(mask, shape),
            &device,
        ))
    });
    if state.lengths().iter().all(|&length| length == 0) {
        model.prefill(
            CausalLmInput {
                token_ids,
                token_mask,
                position_ids: None,
            },
            state,
            &all_outputs(),
        )
    } else {
        model.decode(
            DecodeInput {
                token_ids,
                token_mask,
                position_ids: None,
            },
            state,
            &all_outputs(),
        )
    }
    .unwrap()
}

/// Ragged batched execution equals independent batch-one rows, across ring wraparound.
///
/// The window is 3 and rows grow to 10, 5, and 7 tokens, so every local
/// layer's ring wraps while rows keep different lengths; the scenario covers
/// left padding, an interior hole, unequal multi-token steps, and a finished
/// row. Padding and finished rows must be exact zeros.
#[test]
fn batched_rows_match_independent_rows_across_ring_wraparound() {
    let model = scaled_model(20.0);
    let device = Default::default();
    let steps = [
        Step {
            tokens: vec![vec![0, 3, 17, 5], vec![9, 30, 0, 0], vec![7, 0, 1, 22]],
            mask: Some(vec![
                vec![false, true, true, true],
                vec![true, true, false, false],
                vec![true, false, true, true],
            ]),
        },
        Step {
            tokens: vec![vec![4, 4, 2], vec![19, 0, 0], vec![0, 6, 0]],
            mask: Some(vec![
                vec![true, true, true],
                vec![true, false, false],
                vec![false, true, false],
            ]),
        },
        Step {
            tokens: vec![vec![11], vec![12], vec![13]],
            mask: None,
        },
        Step {
            tokens: vec![vec![14, 8], vec![0, 0], vec![15, 16]],
            mask: Some(vec![vec![true, true], vec![false, false], vec![true, true]]),
        },
        Step {
            tokens: vec![vec![21], vec![23], vec![0]],
            mask: Some(vec![vec![true], vec![true], vec![false]]),
        },
    ];
    let mut batched = model
        .new_decode_state(CacheConfig::new(3, 16).unwrap(), 3, &device)
        .unwrap();
    let mut rows = (0..3)
        .map(|_| {
            model
                .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, &device)
                .unwrap()
        })
        .collect::<Vec<_>>();
    for (index, step) in steps.iter().enumerate() {
        let sequence = step.tokens[0].len();
        let output = run_step(
            &model,
            &mut batched,
            step.tokens.concat(),
            step.mask.as_ref().map(|mask| mask.concat()),
            [3, sequence],
        );
        let logits = output.logits.into_data().to_vec::<f32>().unwrap();
        let hidden = output
            .final_hidden_state
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (row, state) in rows.iter_mut().enumerate() {
            let valid = (0..sequence)
                .filter(|&column| step.mask.as_ref().is_none_or(|mask| mask[row][column]))
                .collect::<Vec<_>>();
            let at = |values: &[f32], column: usize, width: usize| {
                values[(row * sequence + column) * width..(row * sequence + column + 1) * width]
                    .to_vec()
            };
            if valid.is_empty() {
                for column in 0..sequence {
                    assert!(at(&logits, column, 32).iter().all(|&value| value == 0.0));
                }
                continue;
            }
            let tokens = valid
                .iter()
                .map(|&column| step.tokens[row][column])
                .collect::<Vec<_>>();
            let count = tokens.len();
            let oracle = run_step(&model, state, tokens, None, [1, count]);
            let oracle_logits = oracle.logits.into_data().to_vec::<f32>().unwrap();
            let oracle_hidden = oracle
                .final_hidden_state
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            for column in 0..sequence {
                match valid.iter().position(|&c| c == column) {
                    Some(k) => {
                        for (width, actual, expected) in
                            [(32, &logits, &oracle_logits), (8, &hidden, &oracle_hidden)]
                        {
                            let actual = at(actual, column, width);
                            let expected = &expected[k * width..(k + 1) * width];
                            for (left, right) in actual.iter().zip(expected) {
                                assert!(
                                    (left - right).abs() <= 2e-5,
                                    "step {index} row {row} column {column}: {left} vs {right}"
                                );
                            }
                        }
                    }
                    None => {
                        assert!(at(&logits, column, 32).iter().all(|&value| value == 0.0));
                        assert!(at(&hidden, column, 8).iter().all(|&value| value == 0.0));
                    }
                }
            }
        }
    }
    assert_eq!(batched.lengths(), &[10, 5, 7]);
    for (row, state) in rows.iter().enumerate() {
        assert_eq!(state.lengths()[0], batched.lengths()[row]);
    }
}

/// Cached decoding after ring wraparound matches the stateless forward of each row.
#[test]
fn batched_cached_logits_match_stateless_rows_after_wraparound() {
    let model = scaled_model(20.0);
    let device = Default::default();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 16).unwrap(), 2, &device)
        .unwrap();
    let prompt = [[3_i64, 17, 5, 9, 30, 12], [0, 0, 7, 1, 22, 4]];
    run_step(
        &model,
        &mut state,
        prompt.concat(),
        Some([[true; 6], [false, false, true, true, true, true]].concat()),
        [2, 6],
    );
    let decoded = run_step(&model, &mut state, vec![2, 19], None, [2, 1]).logits;
    for (row, tokens) in [vec![3, 17, 5, 9, 30, 12, 2], vec![7, 1, 22, 4, 19]]
        .into_iter()
        .enumerate()
    {
        let width = tokens.len();
        let expected = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_data(TensorData::new(tokens, [1, width]), &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        decoded
            .clone()
            .narrow(0, row, 1)
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(2e-5));
    }
}

/// Masked sentinels of a finished or padded row never reach the embedding or RoPE.
///
/// Masked columns are not validated, so a finished row may carry an
/// out-of-vocabulary pad ID, `-1`, `i32::MIN`, or `i32::MAX`, with an equally
/// invalid explicit position. Regression (review DSM-R1): batched packing
/// gathered a finished row's column 0 into the block and Flex panicked in the
/// embedding gather. Every sentinel must produce bit-identical logits to
/// padding with token and position 0, for one- and two-token steps, implicit
/// and explicit positions, and finished and left-padded rows.
#[test]
fn masked_sentinel_tokens_and_positions_are_inert() {
    let (model, device) = tiny_model();
    /// Builds row 1's tokens, mask, and positions around a sentinel token `pad` and position `at`.
    type MaskedRow = fn(i32, i32) -> (Vec<i32>, Vec<bool>, Vec<i32>);
    let cases: [(usize, MaskedRow); 3] = [
        (1, |pad, at| (vec![pad], vec![false], vec![at])),
        (2, |pad, at| {
            (vec![pad, pad], vec![false, false], vec![at, at])
        }),
        (2, |pad, at| (vec![pad, 8], vec![false, true], vec![at, 3])),
    ];
    for (width, row) in cases {
        for explicit in [false, true] {
            let run = |pad: i32, at: i32| {
                let (tokens, mask, positions) = row(pad, at);
                let active_tokens = [7, 9][..width].to_vec();
                let active_positions = [3, 4][..width].to_vec();
                let mut state = model
                    .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
                    .unwrap();
                model
                    .prefill(
                        CausalLmInput {
                            token_ids: Tensor::from_ints([[1, 2, 3], [4, 5, 6]], &device),
                            token_mask: None,
                            position_ids: None,
                        },
                        &mut state,
                        &ForwardOptions::default(),
                    )
                    .unwrap();
                let output = model
                    .decode(
                        DecodeInput {
                            token_ids: Tensor::from_data(
                                TensorData::new([active_tokens, tokens].concat(), [2, width]),
                                &device,
                            ),
                            token_mask: Some(TokenMask::new(
                                Tensor::<TestBackend, 2, Bool>::from_data(
                                    TensorData::new([vec![true; width], mask].concat(), [2, width]),
                                    &device,
                                ),
                            )),
                            position_ids: explicit.then(|| {
                                Tensor::from_data(
                                    TensorData::new(
                                        [active_positions, positions].concat(),
                                        [2, width],
                                    ),
                                    &device,
                                )
                            }),
                        },
                        &mut state,
                        &ForwardOptions {
                            logits: crate::LogitsSelection::All,
                            ..Default::default()
                        },
                    )
                    .unwrap_or_else(|error| panic!("pad {pad} at {at}: {error}"));
                (output.logits.into_data(), state.lengths().to_vec())
            };
            let baseline = run(0, 0);
            for (pad, at) in [
                (32, 32),
                (-1, -1),
                (i32::MAX, i32::MAX),
                (i32::MIN, 1 << 20),
            ] {
                assert_eq!(
                    run(pad, at),
                    baseline,
                    "width {width}, explicit {explicit}, pad {pad}, position {at}"
                );
            }
        }
    }
}

/// Measures batched decode against one batch-one call per row (the previous algorithm).
///
/// Run with `cargo test --release --lib muse_batched_decode_benchmark -- --ignored --nocapture`.
/// Five alternating samples per variant; minimum and median wall times of a
/// 128-token prefill and 16 one-token decode steps at batch 8 are reported.
#[test]
#[ignore = "release-mode timing benchmark"]
fn muse_batched_decode_benchmark() {
    use std::time::Instant;

    let device = Default::default();
    let config = MuseGlimmerTextConfig {
        vocab_size: 64,
        hidden_size: 256,
        intermediate_size: 512,
        num_hidden_layers: 4,
        num_attention_heads: 8,
        num_key_value_heads: 2,
        head_dim: 32,
        max_position_embeddings: 1024,
        sliding_window: Some(64),
        ..tiny_config()
    };
    let model = initialized_model::<TestBackend>(config, &device);
    let (batch, prompt, steps) = (8, 128, 16);
    let prompts = (0..batch * prompt)
        .map(|index| ((index * 7) % 61) as i64)
        .collect::<Vec<_>>();
    let run = |first: usize, rows: usize| {
        let mut state = model
            .new_decode_state(CacheConfig::new(rows, 512).unwrap(), rows, &device)
            .unwrap();
        let tokens = prompts[first * prompt..(first + rows) * prompt].to_vec();
        let started = Instant::now();
        let _ = model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_data(TensorData::new(tokens, [rows, prompt]), &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
            .into_data();
        let prefill = started.elapsed().as_secs_f64();
        let started = Instant::now();
        for step in 0..steps {
            let _ = model
                .decode(
                    DecodeInput {
                        token_ids: Tensor::from_data(
                            TensorData::new(vec![(step % 61) as i64; rows], [rows, 1]),
                            &device,
                        ),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap()
                .logits
                .into_data();
        }
        (prefill, started.elapsed().as_secs_f64())
    };
    let per_row = || {
        (0..batch)
            .map(|row| run(row, 1))
            .fold((0.0, 0.0), |sum, time| (sum.0 + time.0, sum.1 + time.1))
    };
    let _ = run(0, batch);
    let (mut batched, mut separate) = (Vec::new(), Vec::new());
    for _ in 0..5 {
        batched.push(run(0, batch));
        separate.push(per_row());
    }
    let summary = |samples: &[(f64, f64)], pick: fn(&(f64, f64)) -> f64| {
        let mut values = samples.iter().map(pick).collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        (values[0], values[values.len() / 2])
    };
    for (label, pick) in [
        (
            "prefill",
            (|time: &(f64, f64)| time.0) as fn(&(f64, f64)) -> f64,
        ),
        ("decode x16", |time: &(f64, f64)| time.1),
    ] {
        let (batched_min, batched_median) = summary(&batched, pick);
        let (separate_min, separate_median) = summary(&separate, pick);
        println!(
            "{label}: per-row min {separate_min:.4} s median {separate_median:.4} s; \
             batched min {batched_min:.4} s median {batched_median:.4} s; \
             speedup min {:.2}x median {:.2}x",
            separate_min / batched_min,
            separate_median / batched_median
        );
    }
}
