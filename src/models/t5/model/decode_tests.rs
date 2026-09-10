use burn::{
    backend::Flex,
    module::Initializer,
    tensor::{Bool, Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};
use proptest::prelude::*;

use super::*;
use crate::{
    DecodeState, LogitsSelection,
    models::t5::{T5FeedForwardProjection, T5LayerCrossAttention},
};

/// Deterministic CPU backend used by cache parity tests.
type TestBackend = Flex<f32, i32>;

/// Device selected by the deterministic test backend.
type TestDevice = <TestBackend as BackendTypes>::Device;

/// Builds a small topology with configurable FFN and vocabulary-head layouts.
fn tiny_config(projection: T5FeedForwardProjection, tied: bool, layers: usize) -> T5Config {
    T5Config {
        vocab_size: 24,
        d_model: 8,
        d_kv: 4,
        d_ff: 16,
        num_layers: layers,
        num_decoder_layers: Some(layers),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        dropout_rate: 0.0,
        feed_forward_proj: projection,
        tie_word_embeddings: tied,
        ..T5Config::default()
    }
}

/// Encodes a fixed source prompt for one decoder row.
fn encoded_source(
    model: &T5ForConditionalGeneration<TestBackend>,
    device: &TestDevice,
) -> EncoderOutput<TestBackend> {
    model
        .encode(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[8, 9, 10]], device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect("source encoding")
}

/// Creates an unmasked target segment from one row of IDs.
fn target(ids: &[i32], device: &TestDevice) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device),
        token_mask: None,
        position_ids: None,
    }
}

/// Compares two floating tensors at the crate's F32 golden tolerance.
fn assert_close<const D: usize>(left: Tensor<TestBackend, D>, right: Tensor<TestBackend, D>) {
    left.into_data()
        .assert_approx_eq::<f32>(&right.into_data(), Tolerance::absolute(1e-4));
}

/// Materializes all physical self-KV slots, including uncommitted zero suffixes.
fn self_cache_snapshot(state: &T5DecodeState<TestBackend>) -> Vec<Vec<f32>> {
    let cache = state
        .tensors
        .main()
        .native_kv(SELF_KV_GROUP)
        .expect("self cache");
    let mut snapshot = Vec::new();
    for layer in 0..state.self_spec.layers() {
        for row in 0..state.tensors.batch_size() {
            let (keys, values) = cache
                .read_sequence(layer, row, state.tensors.capacity())
                .expect("physical cache read")
                .materialize()
                .expect("capacity is nonzero");
            snapshot.push(keys.into_data().to_vec::<f32>().expect("F32 keys"));
            snapshot.push(values.into_data().to_vec::<f32>().expect("F32 values"));
        }
    }
    snapshot
}

/// Cached prefill exactly reproduces stateless logits and diagnostics.
#[test]
fn prefill_matches_stateless_for_dense_gated_tied_and_untied_models() {
    let device = Default::default();
    for (projection, tied) in [
        (T5FeedForwardProjection::Relu, true),
        (T5FeedForwardProjection::GatedGelu, false),
        (T5FeedForwardProjection::GatedSilu, false),
    ] {
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(projection, tied, 2),
            &device,
        )
        .expect("tiny model");
        let encoder = encoded_source(&model, &device);
        let decoder = target(&[0, 4, 5], &device);
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            hidden_states: true,
            ..ForwardOptions::default()
        };
        let stateless = model
            .decode_stateless(&encoder, decoder.clone(), &options)
            .expect("stateless decoder");
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .expect("decode state");
        let cached = model
            .prefill_decoder(&encoder, decoder, &mut state, &options)
            .expect("cached prefill")
            .decoder;

        assert_eq!(state.lengths(), [3]);
        assert!(state.has_encoder_cache());
        assert_eq!(state.encoder_sequence_length(), Some(3));
        assert_close(cached.logits, stateless.logits);
        assert_close(
            cached.final_hidden_state.expect("cached final"),
            stateless.final_hidden_state.expect("stateless final"),
        );
        assert_eq!(cached.hidden_states.len(), stateless.hidden_states.len());
        for (cached, stateless) in cached
            .hidden_states
            .into_iter()
            .zip(stateless.hidden_states)
        {
            assert_close(cached, stateless);
        }
    }
}

/// Multi-token and recurrent decode agree with the matching stateless suffix.
#[test]
fn chunked_and_recurrent_decode_match_full_forward() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 2),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(&encoder, target(&[0, 3], &device), &mut state, &options)
        .unwrap();
    let chunk = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[4, 5], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .expect("two-token decode")
        .decoder;
    let full = model
        .decode_stateless(&encoder, target(&[0, 3, 4, 5], &device), &options)
        .unwrap();
    assert_close(chunk.logits, full.logits.narrow(1, 2, 2));
    assert_close(
        chunk.final_hidden_state.unwrap(),
        full.final_hidden_state.unwrap().narrow(1, 2, 2),
    );
    assert_eq!(state.lengths(), [4]);

    let step = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[6], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .expect("single-token decode")
        .decoder;
    let full = model
        .decode_stateless(&encoder, target(&[0, 3, 4, 5, 6], &device), &options)
        .unwrap();
    assert_close(step.logits, full.logits.narrow(1, 4, 1));
    assert_eq!(state.lengths(), [5]);
}

/// Ragged and gapped semantic masks have identical stateless/prefill meaning.
#[test]
fn ragged_gapped_prefill_matches_stateless_valid_positions() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 2),
        &device,
    )
    .unwrap();
    let encoder = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::from_ints([[8, 9, 10], [11, 12, 13]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, false, true], [true, true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    let decoder = CausalLmInput {
        token_ids: Tensor::from_ints([[0, 1, 4], [0, 5, 1]], &device),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, false, true], [true, true, false]].into(),
            &device,
        ))),
        position_ids: None,
    };
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let stateless = model
        .decode_stateless(&encoder, decoder.clone(), &options)
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 6).unwrap(), 2, &device)
        .unwrap();
    let cached = model
        .prefill_decoder(&encoder, decoder, &mut state, &options)
        .unwrap()
        .decoder;
    assert_eq!(state.lengths(), [2, 2]);
    assert_close(cached.logits, stateless.logits);
    assert_close(
        cached.final_hidden_state.unwrap(),
        stateless.final_hidden_state.unwrap(),
    );
    assert!(
        cached
            .hidden_states
            .iter()
            .all(|hidden| hidden.dims() == [2, 3, 8])
    );
}

/// Finished rows ignore masked sentinel IDs and preserve every physical slot.
#[test]
fn mixed_and_all_inactive_decode_rows_are_exact_no_ops() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 2),
        &device,
    )
    .unwrap();
    let encoder = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::from_ints([[8, 9], [10, 11]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 6).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 3], [0, 99]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true], [true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    assert_eq!(state.lengths(), [2, 1]);
    let before = self_cache_snapshot(&state);
    let output = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: Tensor::from_ints([[4, 5], [-1, 99]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true], [false, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .expect("mixed decode");
    assert_eq!(state.lengths(), [4, 1]);
    let after = self_cache_snapshot(&state);
    let per_row_layer_tensors = 2;
    for layer in 0..2 {
        let row_one_start = (layer * 2 + 1) * per_row_layer_tensors;
        assert_eq!(after[row_one_start], before[row_one_start]);
        assert_eq!(after[row_one_start + 1], before[row_one_start + 1]);
    }
    assert!(
        output
            .decoder
            .logits
            .slice([1..2, 0..2, 0..24])
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );
    assert!(
        output
            .decoder
            .final_hidden_state
            .unwrap()
            .slice([1..2, 0..2, 0..8])
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );
    assert!(output.decoder.hidden_states.iter().all(|hidden| {
        hidden
            .clone()
            .slice([1..2, 0..2, 0..8])
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    }));

    let before = self_cache_snapshot(&state);
    let lengths = state.lengths().to_vec();
    let output = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: Tensor::from_ints([[99], [-7]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false], [false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("all-inactive decode");
    assert_eq!(state.lengths(), lengths);
    assert_eq!(self_cache_snapshot(&state), before);
    assert!(
        output
            .decoder
            .logits
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );
}

/// A later-layer error leaves lengths, cross state, and all physical KV unchanged.
#[test]
fn later_layer_failure_is_result_atomic() {
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 2),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 6).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            target(&[0, 3], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let before = self_cache_snapshot(&state);
    let lengths = state.lengths().to_vec();
    let cross_before = state.cross.as_ref().unwrap().keys[0].clone().into_data();
    model.decoder.block[1].cross_attention = None;
    let error = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[4], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("corrupt later layer");
    assert!(error.to_string().contains("cross_attention"));
    assert_eq!(state.lengths(), lengths);
    assert_eq!(self_cache_snapshot(&state), before);
    assert_eq!(
        state.cross.as_ref().unwrap().keys[0].clone().into_data(),
        cross_before
    );
}

/// Capacity failure occurs before model execution and preserves the complete state.
#[test]
fn capacity_boundary_failure_preserves_state() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 1),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 2).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            target(&[0, 3], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let before = self_cache_snapshot(&state);
    let error = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[4], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("full cache");
    assert!(error.to_string().contains("capacity"));
    assert_eq!(state.lengths(), [2]);
    assert_eq!(self_cache_snapshot(&state), before);
}

/// Decoder steps consume cached encoder K/V even if projection weights change.
#[test]
fn cross_attention_key_values_are_projected_only_during_prefill() {
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 2),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 5).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            target(&[0], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    for block in &mut model.decoder.block {
        let cross = block.cross_attention.as_mut().unwrap();
        cross.enc_dec_attention.attention.k_proj.weight = Initializer::Constant { value: f64::NAN }
            .init(
                cross.enc_dec_attention.attention.k_proj.weight.shape(),
                &device,
            );
        cross.enc_dec_attention.attention.v_proj.weight = Initializer::Constant { value: f64::NAN }
            .init(
                cross.enc_dec_attention.attention.v_proj.weight.shape(),
                &device,
            );
    }
    let output = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[4], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("decode must not execute changed K/V projections");
    assert!(
        output
            .decoder
            .logits
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}

/// Gapped encoder masks exclude cached source columns exactly as stateless attention does.
#[test]
fn cached_cross_attention_preserves_gapped_encoder_masks() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedSilu, false, 1),
        &device,
    )
    .unwrap();
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, false, true]].into(),
        &device,
    ));
    let encoder = EncoderOutput {
        hidden_state: Tensor::from_data(
            TensorData::new(
                vec![
                    0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 10_000.0, -10_000.0, 9_000.0,
                    -9_000.0, 8_000.0, -8_000.0, 7_000.0, -7_000.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
                    0.2, 0.1,
                ],
                [1, 3, 8],
            ),
            &device,
        ),
        token_mask: Some(mask),
        hidden_states: Vec::new(),
        attentions: Vec::new(),
    };
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let stateless = model
        .decode_stateless(&encoder, target(&[0, 4], &device), &options)
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    let cached = model
        .prefill_decoder(&encoder, target(&[0, 4], &device), &mut state, &options)
        .unwrap();
    assert_close(cached.decoder.logits, stateless.logits);
}

/// Reorder, truncate, and reset coordinate both self and cross state.
#[test]
fn lifecycle_operations_coordinate_self_and_cross_caches() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let encoder = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::from_ints([[8, 9], [10, 11]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 5).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 3], [0, 99]], &device),
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
    let cross_before = state.cross.as_ref().unwrap().keys[0].clone();
    state.reorder(&[1, 0]).expect("beam reorder");
    assert_eq!(state.lengths(), [1, 2]);
    assert_close(
        state.cross.as_ref().unwrap().keys[0]
            .clone()
            .slice([0..1, 0..2, 0..2, 0..4]),
        cross_before.slice([1..2, 0..2, 0..2, 0..4]),
    );
    let first_reordered_cross = state.cross.as_ref().unwrap().keys[0].clone();
    state.reorder(&[0, 0]).expect("beam duplication");
    assert_eq!(state.lengths(), [1, 1]);
    assert_close(
        state.cross.as_ref().unwrap().keys[0]
            .clone()
            .slice([1..2, 0..2, 0..2, 0..4]),
        first_reordered_cross.slice([0..1, 0..2, 0..2, 0..4]),
    );
    state.truncate(&[0, 1]).expect("full-cache rewind");
    assert_eq!(state.lengths(), [0, 1]);
    assert!(state.has_encoder_cache());
    state.reset().expect("state reset");
    assert_eq!(state.lengths(), [0, 0]);
    assert!(!state.has_encoder_cache());
    assert!(
        self_cache_snapshot(&state)
            .into_iter()
            .flatten()
            .all(|value| value == 0.0)
    );
}

/// A full-KV rewind hides and overwrites the stale suffix on the next decode.
#[test]
fn truncate_then_decode_matches_the_shortened_prefix() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 2),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 6).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            target(&[0, 3, 4], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    state.truncate(&[1]).expect("full cache supports rewind");
    assert!(state.has_encoder_cache());
    let cached = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[5], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let stateless = model
        .decode_stateless(
            &encoder,
            target(&[0, 5], &device),
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_close(cached.decoder.logits, stateless.logits);
    assert_eq!(state.lengths(), [2]);
    let snapshot = self_cache_snapshot(&state);
    assert!(state.truncate(&[3]).is_err());
    assert_eq!(state.lengths(), [2]);
    assert_eq!(self_cache_snapshot(&state), snapshot);
}

/// Invalid lifecycle/input operations are contextual and do not initialize cross state.
#[test]
fn invalid_cached_inputs_fail_before_state_mutation() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    let error = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[1], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("decode before prefill");
    assert!(error.to_string().contains("successful decoder prefill"));
    assert!(!state.has_encoder_cache());

    let error = model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: target(&[0], &device).token_ids,
                token_mask: None,
                position_ids: Some(Tensor::zeros([1, 1], &device)),
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("explicit positions are unsupported");
    assert!(matches!(error, Error::BackendCapability(_)));
    assert_eq!(state.lengths(), [0]);
    assert!(!state.has_encoder_cache());

    let error = model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[99]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("active invalid ID");
    assert!(error.to_string().contains("0..24"));
    assert_eq!(state.lengths(), [0]);
    assert!(!state.has_encoder_cache());

    let error = model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[99]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("prefill cannot contain an inactive row");
    assert!(error.to_string().contains("fully masked prompt row"));
    assert_eq!(state.lengths(), [0]);
    assert!(!state.has_encoder_cache());

    assert!(state.reorder(&[]).is_err());
    assert!(state.reorder(&[1]).is_err());
    assert_eq!(state.lengths(), [0]);
}

/// Initial cross projection is also deferred when the output head fails.
#[test]
fn initial_prefill_output_failure_does_not_bind_encoder() {
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    model.lm_head = Some(super::super::t5_linear(8, 24, 1.0, &device));
    model
        .prefill_decoder(
            &encoder,
            target(&[0], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("tied topology with an independent head");
    assert_eq!(state.lengths(), [0]);
    assert!(!state.has_encoder_cache());
    assert!(
        self_cache_snapshot(&state)
            .into_iter()
            .flatten()
            .all(|value| value == 0.0)
    );
}

/// Corrupt cross-attention topology is rejected instead of indexing cache vectors.
#[test]
fn incompatible_model_and_state_layout_is_rejected() {
    let device = Default::default();
    let one = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let two = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 2),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&two, &device);
    let mut state = one
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    let error = two
        .prefill_decoder(
            &encoder,
            target(&[0], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("state belongs to another topology");
    assert!(error.to_string().contains("incompatible model layout"));
}

/// A semantic causal mask exposes history and only the current query prefix.
#[test]
fn cached_causal_mask_matches_absolute_query_positions() {
    let device = Default::default();
    let mask = cached_causal_mask::<TestBackend>(2, 3, 1, &device)
        .unwrap()
        .into_data();
    assert_eq!(
        mask.to_vec::<bool>().unwrap(),
        vec![
            false, false, false, true, true, false, false, false, false, true, false, false, false,
            false, false,
        ]
    );
}

proptest! {
    /// Every cached causal mask agrees with the absolute-position inequality.
    #[test]
    fn cached_causal_mask_property(
        baseline in 0_usize..8,
        query_sequence in 1_usize..6,
        heads in 1_usize..4,
    ) {
        let device = Default::default();
        let actual = cached_causal_mask::<TestBackend>(
            baseline,
            query_sequence,
            heads,
            &device,
        )
        .unwrap()
        .into_data()
        .to_vec::<bool>()
        .unwrap();
        let key_sequence = baseline + query_sequence;
        let mut expected = Vec::with_capacity(heads * query_sequence * key_sequence);
        for _head in 0..heads {
            for query in 0..query_sequence {
                for key in 0..key_sequence {
                    expected.push(key > baseline + query);
                }
            }
        }
        prop_assert_eq!(actual, expected);
    }
}

/// Test-only construction ensures a decoder block without cross attention errors cleanly.
#[test]
fn cross_projection_rejects_missing_decoder_component() {
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    model.decoder.block[0].cross_attention = None::<T5LayerCrossAttention<TestBackend>>;
    let error = T5CrossAttentionCache::project(&model, &encoder)
        .expect_err("missing cross layer must fail");
    assert!(error.to_string().contains("cross_attention"));
}

/// Full-precision stateless execution retains gradients for training v2.
#[cfg(feature = "autodiff")]
#[test]
fn stateless_encoder_decoder_graph_has_parameter_gradients() {
    use burn::backend::Autodiff;

    type AdBackend = Autodiff<TestBackend>;

    let device = Default::default();
    let model = T5ForConditionalGeneration::<AdBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 1),
        &device,
    )
    .unwrap();
    let output = model
        .forward(
            Seq2SeqInput {
                encoder: CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[8, 9]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                decoder: CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[0, 1, 4]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<AdBackend, 2, Bool>::from_bool(
                        [[true, false, true]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .expect("autodiff forward");
    let gradients =
        burn::optim::GradientsParams::from_grads(output.decoder.logits.sum().backward(), &model);
    assert!(
        gradients
            .get::<TestBackend, 2>(model.lm_head.as_ref().unwrap().weight.id)
            .is_some()
    );
    assert!(
        gradients
            .get::<TestBackend, 1>(model.decoder.final_layer_norm.weight.id)
            .is_some()
    );
    assert!(
        gradients
            .get::<TestBackend, 2>(
                model.decoder.block[0]
                    .feed_forward
                    .gated
                    .as_ref()
                    .unwrap()
                    .down_proj
                    .weight
                    .id
            )
            .is_some()
    );
    assert!(
        gradients
            .get::<TestBackend, 2>(
                model.encoder.block[0]
                    .self_attention
                    .self_attention
                    .attention
                    .q_proj
                    .weight
                    .id
            )
            .is_some()
    );
    assert!(
        gradients
            .get::<TestBackend, 2>(
                model.decoder.block[0]
                    .cross_attention
                    .as_ref()
                    .unwrap()
                    .enc_dec_attention
                    .attention
                    .q_proj
                    .weight
                    .id
            )
            .is_some()
    );
    assert!(
        gradients
            .get::<TestBackend, 2>(
                model.decoder.block[0]
                    .cross_attention
                    .as_ref()
                    .unwrap()
                    .enc_dec_attention
                    .attention
                    .k_proj
                    .weight
                    .id
            )
            .is_some()
    );
    assert!(
        gradients
            .get::<TestBackend, 2>(model.shared.weight.id)
            .is_some()
    );
}
