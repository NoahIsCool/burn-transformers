//! Tests T5 attention, feed-forward layers, masks, initialization, and stateless execution.

use burn::{
    backend::Flex,
    module::Param,
    tensor::{Bool, Int, Tensor, TensorData, Tolerance},
};

use super::*;
use crate::LogitsSelection;
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

/// Runs one self-attention residual sublayer through the shared block graph.
///
/// Padded keys are excluded and `bias` is added; causality follows the sublayer.
fn self_attend(
    layer: &T5LayerSelfAttention<TestBackend>,
    hidden: Tensor<TestBackend, 3>,
    mask: Option<&TokenMask<TestBackend>>,
    bias: Tensor<TestBackend, 4>,
) -> Result<Tensor<TestBackend, 3>> {
    layer.forward_with(
        hidden,
        block::SelfAttentionPlan::padded(mask, Some(bias), layer.causal),
        &mut block::FreshKeyValues,
        Layer::encoder(0),
        &mut Hooks::new(),
    )
}

/// Runs one cross-attention residual sublayer over fresh encoder states.
fn cross_attend(
    layer: &T5LayerCrossAttention<TestBackend>,
    query: Tensor<TestBackend, 3>,
    encoder: Tensor<TestBackend, 3>,
    mask: Option<&TokenMask<TestBackend>>,
) -> Result<Tensor<TestBackend, 3>> {
    layer.forward_with(
        query,
        &block::CrossAttentionPlan::new(block::CrossSource::Encoder(&encoder), mask),
        Layer::decoder(0),
        &mut Hooks::new(),
    )
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
        self_attend(&encoder, first.clone(), None, zero_bias.clone()).expect("encoder forward"),
    );
    let encoder_changed = first_position(
        self_attend(&encoder, changed.clone(), None, zero_bias.clone())
            .expect("encoder changed forward"),
    );
    assert!(
        encoder_first
            .iter()
            .zip(&encoder_changed)
            .any(|(left, right)| (left - right).abs() > 1e-4)
    );

    let decoder_first = first_position(
        self_attend(&decoder, first, None, zero_bias.clone()).expect("decoder forward"),
    );
    let decoder_changed = first_position(
        self_attend(&decoder, changed, None, zero_bias).expect("decoder changed forward"),
    );
    assert_eq!(decoder_first, decoder_changed);
}

/// Stateless preflight names token, mask, position, encoder, and head tensors.
#[test]
fn stateless_device_affinity_errors_are_contextual() {
    assert!(ensure_same_device("t5.encoder.input_ids", &0_u8, &0_u8, PARAMETERS).is_ok());
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
        let error = ensure_same_device(tensor, &1_u8, &0_u8, PARAMETERS)
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
        self_attend(&layer, first, Some(&mask), bias.clone()).expect("masked forward"),
    );
    let changed = first_position(
        self_attend(&layer, changed, Some(&mask), bias).expect("masked changed forward"),
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
    let first = cross_attend(&layer, query.clone(), first_encoder, None)
        .expect("first cross attention")
        .into_data();
    let changed = cross_attend(&layer, query, changed_encoder, None)
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

/// Builds a `[rows, columns]` token input with an optional validity mask.
fn masked_input(ids: &[&[i64]], mask: Option<&[&[bool]]>) -> CausalLmInput<TestBackend> {
    let device = Default::default();
    let (rows, columns) = (ids.len(), ids[0].len());
    CausalLmInput {
        token_ids: Tensor::from_data(TensorData::new(ids.concat(), [rows, columns]), &device),
        token_mask: mask.map(|mask| {
            TokenMask::new(Tensor::from_data(
                TensorData::new(mask.concat(), [rows, columns]),
                &device,
            ))
        }),
        position_ids: None,
    }
}

/// Padded targets run as one batch whose valid positions equal each row's unpadded forward.
///
/// Right, left, and interior padding (plus an all-valid row) are combined in
/// one batch, against a padded source. Padded positions of logits and every
/// diagnostic are exact zeros.
#[test]
fn padded_targets_match_each_rows_unpadded_forward() {
    let device = Default::default();
    let ids: [&[i64]; 4] = [&[0, 3, 4, 9], &[9, 0, 5, 6], &[0, 9, 7, 8], &[0, 1, 2, 3]];
    let mask: [&[bool]; 4] = [
        &[true, true, true, false],
        &[false, true, true, true],
        &[true, false, true, true],
        &[true, true, true, true],
    ];
    let source_ids: [&[i64]; 4] = [&[8, 9, 10], &[11, 12, 0], &[13, 14, 15], &[1, 2, 3]];
    let source_mask: [&[bool]; 4] = [
        &[true, true, true],
        &[true, true, false],
        &[true, true, true],
        &[true, true, true],
    ];
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..ForwardOptions::default()
    };
    for projection in [
        T5FeedForwardProjection::Relu,
        T5FeedForwardProjection::GatedGelu,
    ] {
        let model =
            T5ForConditionalGeneration::<TestBackend>::init(tiny_config(projection), &device)
                .unwrap();
        let batched = model
            .forward(
                Seq2SeqInput {
                    encoder: masked_input(&source_ids, Some(&source_mask)),
                    decoder: masked_input(&ids, Some(&mask)),
                },
                &options,
            )
            .unwrap();
        for row in 0..ids.len() {
            let valid: Vec<usize> = (0..4).filter(|&column| mask[row][column]).collect();
            let compact: Vec<i64> = valid.iter().map(|&column| ids[row][column]).collect();
            let source_valid: Vec<i64> = (0..3)
                .filter(|&column| source_mask[row][column])
                .map(|column| source_ids[row][column])
                .collect();
            let alone = model
                .forward(
                    Seq2SeqInput {
                        encoder: masked_input(&[&source_valid], None),
                        decoder: masked_input(&[&compact], None),
                    },
                    &options,
                )
                .unwrap();
            let pairs =
                std::iter::once((batched.decoder.logits.clone(), alone.decoder.logits.clone()))
                    .chain(
                        batched
                            .decoder
                            .hidden_states
                            .iter()
                            .cloned()
                            .zip(alone.decoder.hidden_states.iter().cloned()),
                    );
            for (padded, unpadded) in pairs {
                let width = padded.dims()[2];
                for (slot, &column) in valid.iter().enumerate() {
                    padded
                        .clone()
                        .slice([row..row + 1, column..column + 1, 0..width])
                        .into_data()
                        .assert_approx_eq::<f32>(
                            &unpadded
                                .clone()
                                .slice([0..1, slot..slot + 1, 0..width])
                                .into_data(),
                            Tolerance::absolute(2e-5),
                        );
                }
                for column in (0..4).filter(|&column| !mask[row][column]) {
                    let values = padded
                        .clone()
                        .slice([row..row + 1, column..column + 1, 0..width])
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();
                    assert!(values.iter().all(|&value| value == 0.0));
                }
            }
        }
    }
}

/// A fully valid mask takes the unpadded path and reproduces the unmasked outputs bit for bit.
#[test]
fn all_valid_target_masks_equal_unmasked_targets() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu),
        &device,
    )
    .unwrap();
    let ids: [&[i64]; 2] = [&[0, 3, 4], &[0, 5, 6]];
    let run = |mask: Option<&[&[bool]]>| {
        model
            .forward(
                Seq2SeqInput {
                    encoder: masked_input(&ids, None),
                    decoder: masked_input(&ids, mask),
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..ForwardOptions::default()
                },
            )
            .unwrap()
            .decoder
            .logits
            .into_data()
    };
    assert_eq!(run(Some(&[&[true; 3], &[true; 3]])), run(None));
}

/// Left padding (fully masked leading queries) keeps every parameter gradient finite.
#[cfg(feature = "autodiff")]
#[test]
fn left_padded_targets_have_finite_gradients() {
    use burn::backend::Autodiff;
    use burn::module::{ModuleVisitor, ParamId};

    /// Autodiff wrapper used to differentiate the batched padded decoder.
    type AdBackend = Autodiff<TestBackend>;

    /// Collects every floating parameter ID of a module.
    struct Ids(
        /// `(parameter ID, tensor rank)` in visitation order.
        Vec<(ParamId, usize)>,
    );

    impl ModuleVisitor<AdBackend> for Ids {
        /// Records the parameter ID and rank of each floating tensor.
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<AdBackend, D>>) {
            self.0.push((param.id, D));
        }
    }

    let device = Default::default();
    let model = T5ForConditionalGeneration::<AdBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu),
        &device,
    )
    .unwrap();
    let ids = Tensor::<AdBackend, 2, Int>::from_ints([[9, 9, 0, 3], [0, 4, 5, 9]], &device);
    let mask = Tensor::<AdBackend, 2, Bool>::from_bool(
        [[false, false, true, true], [true, true, true, false]].into(),
        &device,
    );
    let output = model
        .forward(
            Seq2SeqInput {
                encoder: CausalLmInput {
                    token_ids: Tensor::from_ints([[8, 9, 10], [11, 12, 13]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                decoder: CausalLmInput {
                    token_ids: ids,
                    token_mask: Some(TokenMask::new(mask)),
                    position_ids: None,
                },
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    let gradients =
        burn::optim::GradientsParams::from_grads(output.decoder.logits.sum().backward(), &model);
    let mut ids = Ids(Vec::new());
    model.visit(&mut ids);
    let mut checked = 0;
    for (id, rank) in ids.0 {
        let values = match rank {
            1 => gradients
                .get::<TestBackend, 1>(id)
                .map(|tensor| tensor.into_data().to_vec::<f32>().unwrap()),
            2 => gradients
                .get::<TestBackend, 2>(id)
                .map(|tensor| tensor.into_data().to_vec::<f32>().unwrap()),
            _ => None,
        };
        if let Some(values) = values {
            assert!(
                values.iter().all(|value| value.is_finite()),
                "parameter {id}"
            );
            checked += 1;
        }
    }
    assert!(checked > 10, "gradients reach most parameters");
}

/// Times a padded stateless forward (release builds only).
///
/// A right-padded batch of 16 targets with lengths 33..=64 against 64-token
/// sources, as in seq2seq training. The same test body runs unchanged on
/// earlier sources, so the printed time compares implementations. Run with
/// `cargo test --release --lib padded_forward_benchmark -- --ignored --nocapture`.
#[test]
#[ignore = "timing benchmark; run explicitly in release mode"]
fn padded_forward_benchmark() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        T5Config {
            vocab_size: 512,
            d_model: 256,
            d_kv: 64,
            d_ff: 1024,
            num_layers: 4,
            num_decoder_layers: Some(4),
            num_heads: 4,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            feed_forward_proj: T5FeedForwardProjection::GatedGelu,
            tie_word_embeddings: false,
            ..T5Config::default()
        },
        &device,
    )
    .unwrap();
    let (batch, sequence) = (16_usize, 64_usize);
    let ids: Vec<i64> = (0..batch * sequence)
        .map(|index| (index * 7 % 500) as i64)
        .collect();
    let valid: Vec<bool> = (0..batch * sequence)
        .map(|index| index % sequence < sequence - 2 * (index / sequence))
        .collect();
    let input = || Seq2SeqInput {
        encoder: CausalLmInput {
            token_ids: Tensor::from_data(TensorData::new(ids.clone(), [batch, sequence]), &device),
            token_mask: None,
            position_ids: None,
        },
        decoder: CausalLmInput {
            token_ids: Tensor::from_data(TensorData::new(ids.clone(), [batch, sequence]), &device),
            token_mask: Some(TokenMask::new(Tensor::from_data(
                TensorData::new(valid.clone(), [batch, sequence]),
                &device,
            ))),
            position_ids: None,
        },
    };
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let started = std::time::Instant::now();
        model
            .forward(input(), &options)
            .unwrap()
            .decoder
            .logits
            .into_data();
        best = best.min(started.elapsed().as_secs_f64());
    }
    eprintln!("T5 padded forward batch={batch} sequence={sequence}: best={best:.4}s");
}

/// Deterministic `[2, 3, width]` hidden states with irregular F32 values.
fn readout_hidden(
    width: usize,
    device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<TestBackend, 3> {
    Tensor::<TestBackend, 1, burn::tensor::Int>::arange(0..(6 * width) as i64, device)
        .float()
        .mul_scalar(0.731)
        .sin()
        .reshape([2, 3, width])
}

/// The scaled tied readout equals Burn's reshaping `linear(h / sqrt(d), Eᵀ)` within F32 rounding.
///
/// It multiplies by the transposed table view without reshaping it (no
/// per-call table copy on Flex); the logits must not change beyond rounding.
#[test]
fn tied_readout_matches_reshaping_linear_oracle() {
    let device = Default::default();
    let config = tiny_config(T5FeedForwardProjection::Relu);
    let width = config.d_model;
    let model =
        T5ForConditionalGeneration::<TestBackend>::init(config, &device).expect("valid model");
    let hidden = readout_hidden(width, &device);
    let oracle = burn::tensor::module::linear(
        hidden.clone().mul_scalar(1.0 / (width as f64).sqrt()),
        model.shared.weight.val().transpose(),
        None,
    )
    .into_data();
    model
        .project_logits(hidden)
        .expect("tied projection")
        .into_data()
        .assert_approx_eq::<f32>(&oracle, Tolerance::absolute(1e-6));
}

/// Selects eager attention in every self- and cross-attention module, exposing score points.
fn eager(
    mut model: T5ForConditionalGeneration<TestBackend>,
) -> T5ForConditionalGeneration<TestBackend> {
    for block in model
        .encoder
        .block
        .iter_mut()
        .chain(&mut model.decoder.block)
    {
        block.self_attention.self_attention.attention.precision_mode =
            AttentionPrecisionMode::EagerFp32Softmax;
        if let Some(cross) = &mut block.cross_attention {
            cross.enc_dec_attention.attention.precision_mode =
                AttentionPrecisionMode::EagerFp32Softmax;
        }
    }
    model
}

/// A two-row source and a target with interior padding, so padded hooks run.
fn padded_pair() -> Seq2SeqInput<TestBackend> {
    Seq2SeqInput {
        encoder: masked_input(
            &[&[1, 2, 3], &[4, 5, 0]],
            Some(&[&[true, true, true], &[true, true, false]]),
        ),
        decoder: masked_input(
            &[&[0, 6, 7], &[0, 8, 9]],
            Some(&[&[true, true, true], &[true, false, true]]),
        ),
    }
}

/// A failing edit at any encoder, decoder, or cross-attention point aborts the pass.
#[test]
fn every_point_propagates_edit_failures() {
    use crate::models::shared::test_support::assert_edit_failures_propagate;
    let device = Default::default();
    for projection in [
        T5FeedForwardProjection::Relu,
        T5FeedForwardProjection::GatedGelu,
    ] {
        let model = eager(
            T5ForConditionalGeneration::<TestBackend>::init(tiny_config(projection), &device)
                .unwrap(),
        );
        assert_edit_failures_propagate(&model.activation_points(), |hooks| {
            model.forward_with_hooks(padded_pair(), &ForwardOptions::default(), hooks)
        });
        assert_edit_failures_propagate(&model.encoder.activation_points(), |hooks| {
            model.encode_with_hooks(padded_pair().encoder, &ForwardOptions::default(), hooks)
        });
    }
}

/// Empty hook collections run the plain passes; hooked passes validate inputs before callbacks.
#[test]
fn hooked_passes_delegate_or_validate_before_callbacks() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    let options = ForwardOptions::default();
    let plain = model.encode(padded_pair().encoder, &options).unwrap();
    let delegated = model
        .encode_with_hooks(padded_pair().encoder, &options, &mut Hooks::new())
        .unwrap();
    assert_eq!(
        plain.hidden_state.into_data(),
        delegated.hidden_state.into_data()
    );

    let mut hooks = Hooks::new();
    let embedding = hooks.capture(crate::intervention::Stack::Encoder.embedding());
    let bad_source = masked_input(&[&[1, 2, 16]], None);
    let error = model
        .encode_with_hooks(bad_source, &options, &mut hooks)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, .. } if tensor == "t5.encoder.input_ids"),
        "{error:?}"
    );
    assert!(hooks.get(&embedding).is_err(), "no callback ran");

    let mut misshapen = padded_pair();
    misshapen.encoder.token_mask = Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, true], [true, true]].into(),
        &device,
    )));
    let error = model
        .forward_with_hooks(misshapen, &options, &mut hooks)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, .. } if tensor == "t5.encoder.token_mask"),
        "{error:?}"
    );
}

/// Hooked target padding without host columns uses logical positions and matches the plain pass.
#[test]
fn hooked_logical_padding_matches_the_plain_pass() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu),
        &device,
    )
    .unwrap();
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let plain = model.forward(padded_pair(), &options).unwrap();
    let mut hooks = Hooks::new();
    let residual = hooks.capture(Layer::decoder(0).residual_post());
    let hooked = model
        .forward_with_hooks(padded_pair(), &options, &mut hooks)
        .unwrap();
    plain.decoder.logits.into_data().assert_approx_eq::<f32>(
        &hooked.decoder.logits.into_data(),
        Tolerance::absolute(1e-6),
    );
    assert_eq!(hooks.get(&residual).unwrap().dims(), [2, 3, 4]);
}

/// The vocabulary head must agree with tying; attention probabilities are unavailable.
#[test]
fn head_ownership_and_unsupported_diagnostics_are_rejected() {
    let device = Default::default();
    let mut tied = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    let error = tied
        .forward(
            padded_pair(),
            &ForwardOptions {
                attentions: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::BackendCapability(reason) if reason.contains("reference attention path")),
        "{error:?}"
    );
    tied.lm_head = Some(crate::operations::init_linear(
        &burn::nn::LinearConfig::new(4, 16).with_bias(false),
        &device,
    ));
    let error = tied
        .forward(padded_pair(), &ForwardOptions::default())
        .unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, reason }
            if field == "t5.lm_head" && reason.contains("must not own")),
        "{error:?}"
    );
    let mut untied = T5ForConditionalGeneration::<TestBackend>::init(
        T5Config {
            tie_word_embeddings: false,
            ..tiny_config(T5FeedForwardProjection::Relu)
        },
        &device,
    )
    .unwrap();
    untied.lm_head = None;
    let error = untied
        .forward(padded_pair(), &ForwardOptions::default())
        .unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, reason }
            if field == "t5.lm_head" && reason.contains("require a separate")),
        "{error:?}"
    );
}

/// Block zero must own each stack's shared relative-bias table.
#[test]
fn removing_the_shared_bias_owner_is_reported() {
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    model.encoder.block[0]
        .self_attention
        .self_attention
        .relative_attention_bias = None;
    let error = model
        .encode(padded_pair().encoder, &ForwardOptions::default())
        .unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, reason }
            if field == "t5.relative_attention_bias" && reason.contains("block zero")),
        "{error:?}"
    );
}

/// One-head attention receives exclusion planes of exactly the score shape.
///
/// With one head the `[batch, 1, query, key]` source and logical-target
/// exclusion planes already equal the score shape. The padded pair (a padded
/// source row and an interior-padded target row) must therefore reproduce,
/// at every valid target column, the logits of each row run alone without
/// padding; padded target columns are exact zeros.
#[test]
fn single_head_exclusion_planes_need_no_broadcast() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        T5Config {
            num_heads: 1,
            d_kv: 4,
            ..tiny_config(T5FeedForwardProjection::Relu)
        },
        &device,
    )
    .unwrap();
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let batched = model
        .forward(padded_pair(), &options)
        .unwrap()
        .decoder
        .logits;
    assert_eq!(batched.dims(), [2, 3, 16]);
    // Each row's unpadded source and target, and its valid target columns.
    let rows: [(&[i64], &[i64], &[usize]); 2] = [
        (&[1, 2, 3], &[0, 6, 7], &[0, 1, 2]),
        (&[4, 5], &[0, 9], &[0, 2]),
    ];
    for (row, (source, target, valid)) in rows.into_iter().enumerate() {
        let alone = model
            .forward(
                Seq2SeqInput {
                    encoder: masked_input(&[source], None),
                    decoder: masked_input(&[target], None),
                },
                &options,
            )
            .unwrap()
            .decoder
            .logits;
        for (slot, &column) in valid.iter().enumerate() {
            batched
                .clone()
                .slice([row..row + 1, column..column + 1, 0..16])
                .into_data()
                .assert_approx_eq::<f32>(
                    &alone
                        .clone()
                        .slice([0..1, slot..slot + 1, 0..16])
                        .into_data(),
                    Tolerance::absolute(1e-5),
                );
        }
        for column in (0..3).filter(|column| !valid.contains(column)) {
            let padded = batched
                .clone()
                .slice([row..row + 1, column..column + 1, 0..16])
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert!(padded.iter().all(|&value| value == 0.0), "{padded:?}");
        }
    }
}
