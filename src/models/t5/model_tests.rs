//! Tests T5 attention, feed-forward layers, masks, initialization, and stateless execution.

use burn::{
    backend::Flex,
    module::Param,
    tensor::{Bool, Int, Tensor, TensorData, Tolerance},
};

use super::*;
use crate::models::config::ModelConfig;

/// CPU backend used by T5 component and stateless execution tests.
type TestBackend = Flex<f32, i32>;

/// Returns a tiny topology that still exercises every T5 attention path.
fn tiny_config(projection: T5FeedForwardProjection) -> T5Config {
    T5Config {
        vocab_size: 16,
        d_model: 4,
        d_kv: 2,
        d_ff: 8,
        num_layers: 1,
        num_decoder_layers: Some(1),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 32,
        dropout_rate: 0.0,
        feed_forward_proj: projection,
        dense_act_fn: None,
        is_gated_act: None,
        tie_word_embeddings: true,
        ..T5Config::default()
    }
}

/// Replaces all four runtime projection matrices with two-dimensional identity.
fn make_attention_identity(attention: &mut T5Attention<TestBackend>) {
    let device = attention.attention.q_proj.weight.device();
    let identity = Tensor::eye(2, &device);
    attention.attention.q_proj.weight = Param::from_tensor(identity.clone());
    attention.attention.k_proj.weight = Param::from_tensor(identity.clone());
    attention.attention.v_proj.weight = Param::from_tensor(identity.clone());
    attention.attention.o_proj.weight = Param::from_tensor(identity);
}

/// Reads the first position from a `[1, sequence, width]` tensor.
fn first_position(tensor: Tensor<TestBackend, 3>) -> Vec<f32> {
    let width = tensor.dims()[2];
    tensor
        .narrow(1, 0, 1)
        .reshape([width])
        .into_data()
        .to_vec::<f32>()
        .expect("Flex exposes f32 values")
}

/// Decoder causality hides future values while encoder attention observes them.
#[test]
fn encoder_is_bidirectional_and_decoder_self_attention_is_causal() {
    let device = Default::default();
    let config = T5Config {
        d_model: 2,
        d_kv: 2,
        num_heads: 1,
        d_ff: 4,
        vocab_size: 8,
        num_layers: 1,
        num_decoder_layers: Some(1),
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 32,
        dropout_rate: 0.0,
        ..T5Config::default()
    };
    let mut encoder = T5LayerSelfAttention::init(&config, false, false, &device)
        .expect("valid encoder attention");
    let mut decoder =
        T5LayerSelfAttention::init(&config, true, false, &device).expect("valid decoder attention");
    make_attention_identity(&mut encoder.self_attention);
    make_attention_identity(&mut decoder.self_attention);
    let first = Tensor::from_floats([[[1.0, 0.0], [0.0, 1.0]]], &device);
    let changed = Tensor::from_floats([[[1.0, 0.0], [1.0, 1.0]]], &device);
    let zero_bias = Tensor::zeros([1, 1, 2, 2], &device);

    let encoder_first = first_position(
        encoder
            .forward(first.clone(), None, zero_bias.clone())
            .expect("encoder forward"),
    );
    let encoder_changed = first_position(
        encoder
            .forward(changed.clone(), None, zero_bias.clone())
            .expect("encoder changed forward"),
    );
    assert!(
        encoder_first
            .iter()
            .zip(&encoder_changed)
            .any(|(left, right)| (left - right).abs() > 1e-4)
    );

    let decoder_first = first_position(
        decoder
            .forward(first, None, zero_bias.clone())
            .expect("decoder forward"),
    );
    let decoder_changed = first_position(
        decoder
            .forward(changed, None, zero_bias)
            .expect("decoder changed forward"),
    );
    assert_eq!(decoder_first, decoder_changed);
}

/// Stateless preflight names token, mask, position, encoder, and head tensors.
#[test]
fn stateless_device_affinity_errors_are_contextual() {
    assert!(ensure_model_device("t5.encoder.input_ids", &0_u8, &0_u8).is_ok());
    for tensor in [
        "t5.encoder.input_ids",
        "t5.encoder.token_mask",
        "t5.encoder.position_ids",
        "t5.decoder.input_ids",
        "t5.decoder.token_mask",
        "t5.decoder.position_ids",
        "t5.encoder_output.hidden_state",
        "t5.encoder_output.token_mask",
        "t5.lm_head",
    ] {
        let error = ensure_model_device(tensor, &1_u8, &0_u8)
            .expect_err("different device identities must fail preflight");
        assert!(error.to_string().contains(tensor));
        assert!(error.to_string().contains("model parameters"));
    }
}

/// A false key mask prevents padded source content from affecting real queries.
#[test]
fn semantic_padding_mask_excludes_encoder_keys() {
    let device = Default::default();
    let config = T5Config {
        d_model: 2,
        d_kv: 2,
        num_heads: 1,
        d_ff: 4,
        vocab_size: 8,
        num_layers: 1,
        num_decoder_layers: Some(1),
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 32,
        dropout_rate: 0.0,
        ..T5Config::default()
    };
    let mut layer =
        T5LayerSelfAttention::init(&config, false, false, &device).expect("valid attention");
    make_attention_identity(&mut layer.self_attention);
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, false]].into(),
        &device,
    ));
    let first = Tensor::from_floats([[[1.0, 0.0], [0.0, 1.0]]], &device);
    let changed = Tensor::from_floats([[[1.0, 0.0], [1.0, 1.0]]], &device);
    let bias = Tensor::zeros([1, 1, 2, 2], &device);
    let first = first_position(
        layer
            .forward(first, Some(&mask), bias.clone())
            .expect("masked forward"),
    );
    let changed = first_position(
        layer
            .forward(changed, Some(&mask), bias)
            .expect("masked changed forward"),
    );
    assert_eq!(first, changed);
}

/// Decoder cross-attention consumes encoder key/value states.
#[test]
fn cross_attention_changes_with_encoder_content() {
    let device = Default::default();
    let config = T5Config {
        d_model: 2,
        d_kv: 2,
        num_heads: 1,
        d_ff: 4,
        vocab_size: 8,
        num_layers: 1,
        num_decoder_layers: Some(1),
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 32,
        dropout_rate: 0.0,
        ..T5Config::default()
    };
    let mut layer = T5LayerCrossAttention::init(&config, &device).expect("valid cross attention");
    make_attention_identity(&mut layer.enc_dec_attention);
    let query = Tensor::from_floats([[[1.0, 0.0]]], &device);
    let first_encoder = Tensor::from_floats([[[1.0, 0.0], [0.0, 1.0]]], &device);
    let changed_encoder = Tensor::from_floats([[[0.0, 1.0], [0.0, 1.0]]], &device);
    let first = layer
        .forward(query.clone(), first_encoder, None)
        .expect("first cross attention")
        .into_data();
    let changed = layer
        .forward(query, changed_encoder, None)
        .expect("changed cross attention")
        .into_data();
    assert_ne!(first, changed);
}

/// Every serialized FFN spelling selects exactly one matching shared module.
#[test]
fn feed_forward_variants_select_exact_topology_and_activation() {
    use crate::architecture::Activation;

    let device = Default::default();
    for (projection, activation, gated) in [
        (T5FeedForwardProjection::Relu, Activation::Relu, false),
        (T5FeedForwardProjection::Gelu, Activation::Gelu, false),
        (
            T5FeedForwardProjection::GatedGelu,
            Activation::GeluTanh,
            true,
        ),
        (
            T5FeedForwardProjection::GatedGeluNew,
            Activation::GeluTanh,
            true,
        ),
        (T5FeedForwardProjection::GatedSilu, Activation::Silu, true),
    ] {
        let layer = T5LayerFeedForward::<TestBackend>::init(&tiny_config(projection), &device)
            .expect("valid feed-forward variant");
        if gated {
            assert!(layer.dense.is_none());
            assert_eq!(
                layer.gated.expect("gated projection").activation,
                activation
            );
        } else {
            assert!(layer.gated.is_none());
            assert_eq!(
                layer.dense.expect("dense projection").activation,
                activation
            );
        }
    }
}

/// T5's global initializer factor also scales every RMSNorm weight.
#[test]
fn t5_norm_initializer_honors_global_factor() {
    let device = Default::default();
    let config = T5Config {
        d_model: 3,
        initializer_factor: 2.5,
        ..tiny_config(T5FeedForwardProjection::Relu)
    };
    let values = t5_norm::<TestBackend>(&config, &device)
        .expect("valid norm")
        .weight
        .val()
        .into_data();
    assert_eq!(values, TensorData::from([2.5_f32, 2.5, 2.5]));
}

/// Tied logits reuse the sole embedding table and apply `d_model^-1/2`.
#[test]
fn tied_lm_head_matches_shared_weight_scaling_equation() {
    let device = Default::default();
    let mut config = tiny_config(T5FeedForwardProjection::Relu);
    config.vocab_size = 3;
    config.d_model = 2;
    config.d_kv = 2;
    config.num_heads = 1;
    config.d_ff = 4;
    let mut model =
        T5ForConditionalGeneration::<TestBackend>::init(config, &device).expect("valid model");
    model.shared.weight = Param::from_tensor(Tensor::from_floats(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        &device,
    ));
    let hidden = Tensor::from_floats([[[2.0_f32.sqrt(), 0.0]]], &device);
    model
        .project_logits(hidden)
        .expect("tied projection")
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::from([[[1.0, 0.0, 1.0]]]),
            Tolerance::absolute(1e-5),
        );
}

/// T5 v1.1/FLAN disable scaling and use their checkpoint's separate head.
#[test]
fn untied_flag_uses_separate_head_without_decoder_scaling() {
    let device = Default::default();
    let mut config = tiny_config(T5FeedForwardProjection::GatedGelu);
    config.vocab_size = 2;
    config.d_model = 2;
    config.d_kv = 2;
    config.num_heads = 1;
    config.d_ff = 4;
    config.tie_word_embeddings = false;
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(config, &device)
        .expect("valid T5 v1.1 model");
    model.shared.weight =
        Param::from_tensor(Tensor::from_floats([[100.0, 0.0], [0.0, 100.0]], &device));
    model
        .lm_head
        .as_mut()
        .expect("untied checkpoint head")
        .weight = Param::from_tensor(Tensor::eye(2, &device));

    model
        .project_logits(Tensor::from_floats([[[2.0, 4.0]]], &device))
        .expect("untied unscaled projection")
        .into_data()
        .assert_approx_eq::<f32>(&TensorData::from([[[2.0, 4.0]]]), Tolerance::absolute(1e-6));
}

/// A tiny complete graph returns requested encoder and decoder diagnostics.
#[test]
fn stateless_forward_covers_encoder_decoder_and_cross_attention() {
    let device = Default::default();
    TestBackend::seed(&device, 19);
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu),
        &device,
    )
    .expect("valid tiny model");
    let encoder_mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, true, false], [true, true, true]].into(),
        &device,
    ));
    let decoder_mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, true], [true, false]].into(),
        &device,
    ));
    let output = model
        .forward(
            Seq2SeqInput {
                encoder: CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                        [[1, 2, 0], [3, 4, 5]],
                        &device,
                    ),
                    token_mask: Some(encoder_mask),
                    position_ids: None,
                },
                decoder: CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0, 6], [0, 7]], &device),
                    token_mask: Some(decoder_mask),
                    position_ids: None,
                },
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .expect("stateless T5 forward");
    assert_eq!(output.encoder.hidden_state.dims(), [2, 3, 4]);
    assert_eq!(output.encoder.hidden_states.len(), 3);
    assert_eq!(output.decoder.logits.dims(), [2, 2, 16]);
    assert_eq!(output.decoder.hidden_states.len(), 3);
    assert_eq!(
        output
            .decoder
            .final_hidden_state
            .expect("requested final state")
            .dims(),
        [2, 2, 4]
    );
    assert!(output.cross_attentions.is_empty());
}

/// Every public malformed shape is rejected before indexing or reshape panics.
#[test]
fn stateless_input_validation_is_contextual() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu),
        &device,
    )
    .expect("valid model");
    let empty_mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[false, false]].into(),
        &device,
    ));
    let error = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: Some(empty_mask),
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect_err("fully masked rows are invalid");
    assert!(error.to_string().contains("fully masked row"));

    let error = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[16]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect_err("out-of-range IDs are invalid");
    assert!(error.to_string().contains("0..16"));

    let error = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1]], &device),
                token_mask: None,
                position_ids: Some(Tensor::zeros([1, 1], &device)),
            },
            &ForwardOptions::default(),
        )
        .expect_err("absolute positions are unsupported");
    assert!(matches!(error, Error::BackendCapability(_)));
}

/// Last-valid and explicit-position selection retain order for every batch row.
#[test]
fn logit_selection_handles_gapped_masks_and_ordered_positions() {
    let device = Default::default();
    let hidden = Tensor::<TestBackend, 3>::from_floats(
        [[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]]],
        &device,
    );
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, false, true], [false, true, false]].into(),
        &device,
    ));
    select_hidden(hidden.clone(), Some(&mask), &LogitsSelection::Last)
        .expect("semantic last position")
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::from([[[2.0]], [[4.0]]]),
            Tolerance::absolute(1e-6),
        );
    select_hidden(
        hidden.clone(),
        None,
        &LogitsSelection::Positions(vec![2, 0]),
    )
    .expect("ordered explicit positions")
    .into_data()
    .assert_approx_eq::<f32>(
        &TensorData::from([[[2.0], [0.0]], [[5.0], [3.0]]]),
        Tolerance::absolute(1e-6),
    );
    assert!(
        select_hidden(
            hidden.clone(),
            None,
            &LogitsSelection::Positions(Vec::new())
        )
        .is_err()
    );
    assert!(select_hidden(hidden, None, &LogitsSelection::Positions(vec![3])).is_err());
}

/// Official FLAN and UL2 schemas preserve their distinct gated activations.
#[test]
fn official_flan_and_ul2_fixtures_initialize_when_reduced() {
    let device = Default::default();
    let mut flan = T5Config::from_json_str(include_str!("../../../tests/fixtures/flan_t5_xl.json"))
        .expect("official FLAN fixture");
    let mut ul2 = T5Config::from_json_str(include_str!("../../../tests/fixtures/ul2.json"))
        .expect("official UL2 fixture");
    assert_eq!(flan.feed_forward_proj, T5FeedForwardProjection::GatedGelu);
    assert_eq!(ul2.feed_forward_proj, T5FeedForwardProjection::GatedSilu);
    reduce_official_config(&mut flan);
    reduce_official_config(&mut ul2);

    for config in [flan, ul2] {
        let expects_separate_head = !config.tie_word_embeddings;
        let model = T5ForConditionalGeneration::<TestBackend>::init(config, &device)
            .expect("reduced official topology should initialize");
        assert_eq!(model.encoder.block.len(), 1);
        assert_eq!(model.decoder.block.len(), 1);
        assert!(model.encoder.block[0].feed_forward.gated.is_some());
        assert_eq!(model.lm_head.is_some(), expects_separate_head);
        assert!(
            model.encoder.block[0]
                .self_attention
                .self_attention
                .relative_attention_bias
                .is_some()
        );
        assert!(
            model.decoder.block[0]
                .self_attention
                .self_attention
                .relative_attention_bias
                .is_some()
        );
    }
}

/// Shrinks only allocation dimensions while retaining a fixture's family choices.
fn reduce_official_config(config: &mut T5Config) {
    config.vocab_size = 32;
    config.d_model = 8;
    config.d_kv = 4;
    config.d_ff = 16;
    config.num_heads = 2;
    config.num_layers = 1;
    config.num_decoder_layers = Some(1);
    config.dropout_rate = 0.0;
}

/// T5 attention scale remains one even when `d_kv` is not one.
#[test]
fn attention_uses_unscaled_dot_products() {
    let device = Default::default();
    let config = tiny_config(T5FeedForwardProjection::Relu);
    let attention =
        T5Attention::<TestBackend>::init(&config, None, &device).expect("valid attention");
    assert_eq!(attention.attention.scale, Some(1.0));
    assert_eq!(config.attention_inner_dim(), 4);
}
