use burn::{
    backend::Flex,
    tensor::{Bool, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::{
    DecodeState, attention::AttentionExecutionPolicy, checkpoint::TensorSink,
    models::muse_glimmer::MuseGlimmerCheckpointOptions,
};

type TestBackend = Flex<f32, i32>;

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
    super::super::aligned_attention_mask(
        1,
        heads,
        query_sequence,
        key_sequence,
        0..query_sequence,
        0..key_sequence,
        true,
        window,
        device,
    )?
    .ok_or_else(|| {
        Error::configuration(
            "muse_glimmer.attention_mask",
            "cached attention must produce a causal exclusion",
        )
    })
}
