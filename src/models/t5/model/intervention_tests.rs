//! Tiny public-interface checks for T5 representation experiments.

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, Tolerance},
};

use super::*;

/// CPU backend used for hook parity and padding regressions.
type TestBackend = Flex<f32, i32>;

/// Builds one small encoder and decoder block with the selected feed-forward topology.
fn config(projection: T5FeedForwardProjection) -> T5Config {
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

/// Supplies one unpadded three-token sequence with implicit positions.
fn tokens<B: Backend>(device: &B::Device) -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_ints([[1, 2, 3]], device),
        token_mask: None,
        position_ids: None,
    }
}

/// Uses the same small token sequence for the encoder source and decoder target.
fn input<B: Backend>(device: &B::Device) -> Seq2SeqInput<B> {
    Seq2SeqInput {
        encoder: tokens(device),
        decoder: tokens(device),
    }
}

/// Requests logits at every decoder position and all hidden-state diagnostics.
fn options() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..ForwardOptions::default()
    }
}

/// Identity encoder and cross-attention edits preserve logits and hidden states for both FFN layouts.
#[test]
fn t5_noop_hooks_preserve_dense_and_gated_outputs() {
    let device = Default::default();
    for projection in [
        T5FeedForwardProjection::Relu,
        T5FeedForwardProjection::GatedGelu,
    ] {
        let model = T5ForConditionalGeneration::<TestBackend>::init(config(projection), &device)
            .expect("tiny T5");
        let ordinary = model.forward(input(&device), &options()).unwrap();
        let mut hooks = Hooks::new();
        hooks.edit(Layer::encoder(0).mlp_activation(), |activation| activation);
        hooks.edit(Layer::decoder(0).cross_attention_output(), |activation| {
            activation
        });
        let hooked = model
            .forward_with_hooks(input(&device), &options(), &mut hooks)
            .unwrap();
        hooked
            .decoder
            .logits
            .into_data()
            .assert_approx_eq::<f32>(&ordinary.decoder.logits.into_data(), Tolerance::default());
        hooked
            .encoder
            .hidden_state
            .into_data()
            .assert_approx_eq::<f32>(
                &ordinary.encoder.hidden_state.into_data(),
                Tolerance::default(),
            );
        assert_eq!(
            hooked.decoder.hidden_states.len(),
            ordinary.decoder.hidden_states.len()
        );
        for (actual, expected) in hooked
            .decoder
            .hidden_states
            .into_iter()
            .zip(ordinary.decoder.hidden_states)
        {
            actual
                .into_data()
                .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
        }
    }
}

/// Capturing encoder attention leaves masked-source outputs equal to ordinary execution.
#[test]
fn t5_encoder_hook_path_preserves_padding_semantics() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    let source = || CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 0]], &device),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, true, false]].into(),
            &device,
        ))),
        position_ids: None,
    };
    let ordinary = model.encode(source(), &options()).unwrap();
    let mut hooks = Hooks::new();
    hooks.capture(Layer::encoder(0).attention_output());
    let hooked = model
        .encode_with_hooks(source(), &options(), &mut hooks)
        .unwrap();
    hooked
        .hidden_state
        .into_data()
        .assert_approx_eq::<f32>(&ordinary.hidden_state.into_data(), Tolerance::default());
}

/// Left, right, and gapped decoder padding preserve outputs and zero captured invalid positions.
#[test]
fn t5_padded_decoder_hooks_match_compacted_outputs_in_original_coordinates() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::GatedGelu),
        &device,
    )
    .unwrap();
    for padding in [
        [true, true, false],
        [false, true, true],
        [true, false, true],
    ] {
        let padded = || {
            let mut input = input(&device);
            input.decoder.token_mask = Some(TokenMask::new(
                Tensor::<TestBackend, 2, Bool>::from_bool([padding].into(), &device),
            ));
            input
        };
        let ordinary = model.forward(padded(), &options()).unwrap();
        let mut hooks = Hooks::new();
        let residual = hooks.capture(Layer::decoder(0).residual_post());
        let actual = model
            .forward_with_hooks(padded(), &options(), &mut hooks)
            .unwrap();
        actual
            .decoder
            .logits
            .into_data()
            .assert_approx_eq::<f32>(&ordinary.decoder.logits.into_data(), Tolerance::default());
        let captured = hooks.get(&residual).unwrap();
        assert_eq!(captured.dims(), [1, 3, 4]);
        for (position, valid) in padding.into_iter().enumerate() {
            if !valid {
                assert_eq!(
                    captured
                        .clone()
                        .narrow(1, position, 1)
                        .abs()
                        .sum()
                        .into_scalar(),
                    0.0
                );
            }
        }
        for (actual, expected) in actual
            .decoder
            .hidden_states
            .into_iter()
            .zip(ordinary.decoder.hidden_states)
        {
            actual
                .into_data()
                .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
        }
    }
}

/// Cross-attention ablation changes logits and restoring a clean capture recovers the baseline.
#[test]
fn t5_selective_capture_and_cross_attention_patch_are_consumed() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::GatedGelu),
        &device,
    )
    .unwrap();
    let point = Layer::decoder(0).cross_attention_output();
    let mut clean = Hooks::new();
    let saved = clean.capture(point.clone());
    let mlp = clean.capture(Layer::encoder(0).mlp_activation().token(1));
    let baseline = model
        .forward_with_hooks(input(&device), &options(), &mut clean)
        .unwrap();
    assert_eq!(clean.get(&mlp).unwrap().dims(), [1, 1, 8]);

    let mut ablated = Hooks::new();
    ablated.edit(point.clone(), |tensor| tensor.zeros_like());
    let after_ablation = ablated.capture(point.clone());
    let altered = model
        .forward_with_hooks(input(&device), &options(), &mut ablated)
        .unwrap();
    assert_eq!(
        ablated
            .get(&after_ablation)
            .unwrap()
            .abs()
            .sum()
            .into_scalar(),
        0.0
    );
    let difference = (altered.decoder.logits - baseline.decoder.logits.clone())
        .abs()
        .max()
        .into_scalar();
    assert!(
        difference > 1e-6,
        "cross-attention ablation must affect decoder logits"
    );

    let mut restored = Hooks::new();
    restored.edit(point.clone(), |tensor| tensor.zeros_like());
    restored.replace(point, clean.get(&saved).unwrap());
    let patched = model
        .forward_with_hooks(input(&device), &options(), &mut restored)
        .unwrap();
    patched
        .decoder
        .logits
        .into_data()
        .assert_approx_eq::<f32>(&baseline.decoder.logits.into_data(), Tolerance::default());
}

/// Single-head captures retain `[batch, head, token, width]` axes without changing logits.
#[test]
fn t5_attention_head_capture_uses_native_encoder_and_cross_attention_shapes() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    let mut hooks = Hooks::new();
    let query = hooks.capture(Layer::encoder(0).self_attention().query().head(1));
    let context = hooks.capture(Layer::decoder(0).cross_attention().head_context().head(0));
    let ordinary = model.forward(input(&device), &options()).unwrap();
    let output = model
        .forward_with_hooks(input(&device), &options(), &mut hooks)
        .unwrap();
    assert_eq!(hooks.get(&query).unwrap().dims(), [1, 1, 3, 2]);
    assert_eq!(hooks.get(&context).unwrap().dims(), [1, 1, 3, 2]);
    output
        .decoder
        .logits
        .into_data()
        .assert_approx_eq::<f32>(&ordinary.decoder.logits.into_data(), Tolerance::default());
}

/// Unavailable activation points and shape-changing replacements return contextual hook errors.
#[test]
fn t5_hook_errors_name_unavailable_points_and_invalid_replacements() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    let mut decoder_on_encoder = Hooks::new();
    decoder_on_encoder.capture(Layer::decoder(0).mlp_output());
    let error = model
        .encode_with_hooks(tokens(&device), &options(), &mut decoder_on_encoder)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains(&Layer::decoder(0).mlp_output().info().name)
    );

    let mut wrong_shape = Hooks::new();
    let point = Layer::encoder(0).mlp_activation();
    wrong_shape.edit(point, |tensor| tensor.narrow(2, 0, 1));
    let error = model
        .encode_with_hooks(tokens(&device), &options(), &mut wrong_shape)
        .unwrap_err();
    assert!(error.to_string().contains("replacement must preserve"));

    let mut unavailable_pattern = Hooks::new();
    unavailable_pattern.capture(Layer::encoder(0).self_attention().pattern());
    let error = model
        .encode_with_hooks(tokens(&device), &options(), &mut unavailable_pattern)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("activation point is unavailable")
    );
}

/// An empty hook registry preserves ordinary logits for a gapped decoder mask.
#[test]
fn t5_empty_hooks_keep_ordinary_masked_decoder_support() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    let masked = || {
        let mut input = input(&device);
        input.decoder.token_mask = Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, false, true]].into(),
            &device,
        )));
        input
    };
    let ordinary = model.forward(masked(), &options()).unwrap();
    let hooked = model
        .forward_with_hooks(masked(), &options(), &mut Hooks::new())
        .unwrap();
    hooked
        .decoder
        .logits
        .into_data()
        .assert_approx_eq::<f32>(&ordinary.decoder.logits.into_data(), Tolerance::default());
}

/// An encoder residual edit preserves gradients for its inserted linear module and shared embeddings.
#[cfg(feature = "autodiff")]
#[test]
fn t5_inserted_module_and_upstream_weights_receive_gradients() {
    use burn::{backend::Autodiff, nn::LinearConfig, optim::GradientsParams};

    /// Autodiff wrapper used to inspect inserted-module and upstream parameter gradients.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let model = T5ForConditionalGeneration::<AdBackend>::init(
        config(T5FeedForwardProjection::GatedGelu),
        &device,
    )
    .unwrap();
    let inserted = LinearConfig::new(4, 4).init::<AdBackend>(&device);
    let mut hooks = Hooks::new();
    hooks.edit(Layer::encoder(0).residual_post(), |hidden| {
        inserted.forward(hidden)
    });
    let output = model
        .forward_with_hooks(input(&device), &options(), &mut hooks)
        .unwrap();
    let gradients = output.decoder.logits.sum().backward();
    assert!(
        inserted.weight.val().grad(&gradients).is_some(),
        "inserted module remains in autodiff graph"
    );
    let parameters = GradientsParams::from_grads(gradients, &model);
    assert!(
        parameters
            .get::<TestBackend, 2>(model.shared.weight.id)
            .is_some(),
        "upstream shared embeddings remain connected"
    );
}

/// Eager attention exposes zero future-token probabilities while capture preserves its logits.
#[test]
fn t5_eager_attention_exposes_causal_patterns_without_changing_logits() {
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    model.decoder.block[0]
        .self_attention
        .self_attention
        .attention
        .precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
    let point = Layer::decoder(0).self_attention().pattern();
    assert!(model.activation_points().contains(&point.info()));
    let ordinary = model.forward(input(&device), &options()).unwrap();
    let mut hooks = Hooks::new();
    let pattern = hooks.capture(point);
    let output = model
        .forward_with_hooks(input(&device), &options(), &mut hooks)
        .unwrap();
    let pattern = hooks.get(&pattern).unwrap();
    assert_eq!(pattern.dims(), [1, 2, 3, 3]);
    let future = pattern.slice([0..1, 0..2, 0..1, 1..3]);
    assert_eq!(future.abs().sum().into_scalar(), 0.0);
    output
        .decoder
        .logits
        .into_data()
        .assert_approx_eq::<f32>(&ordinary.decoder.logits.into_data(), Tolerance::default());
}

/// An all-valid decoder mask preserves exact logits and diagnostics through identity hooks.
#[test]
fn t5_all_valid_decoder_masks_preserve_exact_noop_hook_results() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    let masked_input = || {
        let mut input = input(&device);
        input.decoder.token_mask = Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, true, true]].into(),
            &device,
        )));
        input
    };
    let ordinary = model.forward(masked_input(), &options()).unwrap();
    let mut hooks = Hooks::new();
    hooks.edit(Layer::decoder(0).mlp_activation(), |activation| activation);
    let activation = hooks.capture(Layer::decoder(0).mlp_activation());
    let hooked = model
        .forward_with_hooks(masked_input(), &options(), &mut hooks)
        .unwrap();
    assert_eq!(hooks.get(&activation).unwrap().dims(), [1, 3, 8]);
    assert_eq!(
        hooked.decoder.logits.into_data().to_vec::<f32>().unwrap(),
        ordinary.decoder.logits.into_data().to_vec::<f32>().unwrap(),
    );
    assert_eq!(
        hooked.decoder.hidden_states.len(),
        ordinary.decoder.hidden_states.len()
    );
    for (actual, expected) in hooked
        .decoder
        .hidden_states
        .into_iter()
        .zip(ordinary.decoder.hidden_states)
    {
        assert_eq!(
            actual.into_data().to_vec::<f32>().unwrap(),
            expected.into_data().to_vec::<f32>().unwrap()
        );
    }
}

/// A short all-valid mask reports the decoder field and both expected and actual shapes.
#[test]
fn t5_malformed_all_valid_decoder_mask_fails_shape_validation() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    let mut input = input(&device);
    input.decoder.token_mask = Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, true]].into(),
        &device,
    )));
    let mut hooks = Hooks::new();
    hooks.capture(Layer::decoder(0).mlp_activation());
    let error = model
        .forward_with_hooks(input, &options(), &mut hooks)
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("t5.decoder.token_mask"), "{message}");
    assert!(message.contains("[1, 3]"), "{message}");
    assert!(message.contains("[1, 2]"), "{message}");
}

/// Fully masked or malformed decoder inputs fail before encoder callbacks can run.
#[test]
fn t5_decoder_mask_preflight_runs_before_encoder_callbacks() {
    use std::cell::Cell;

    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    for mask in [
        Tensor::<TestBackend, 2, Bool>::from_bool([[false, false, false]].into(), &device),
        Tensor::<TestBackend, 2, Bool>::from_bool([[true, true]].into(), &device),
    ] {
        let called = Cell::new(0);
        let mut hooks = Hooks::new();
        hooks.edit(Layer::encoder(0).residual_pre(), |activation| {
            called.set(called.get() + 1);
            activation
        });
        let mut input = input(&device);
        input.decoder.token_mask = Some(TokenMask::new(mask));
        assert!(
            model
                .forward_with_hooks(input, &options(), &mut hooks)
                .is_err()
        );
        assert_eq!(
            called.get(),
            0,
            "invalid targets must fail before encoder callbacks"
        );
    }
}

/// Value and pattern edits preserve padding and causal zeros; cross-attention
/// captures retain padding zeros.
#[test]
fn t5_padded_rank_four_points_stay_zero_after_edits() {
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    model.decoder.block[0]
        .self_attention
        .self_attention
        .attention
        .precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
    model.decoder.block[0]
        .cross_attention
        .as_mut()
        .unwrap()
        .enc_dec_attention
        .attention
        .precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
    let mut input = input(&device);
    input.decoder.token_mask = Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, false, true]].into(),
        &device,
    )));
    let mut hooks = Hooks::new();
    let self_attention = Layer::decoder(0).self_attention();
    hooks.edit(self_attention.value().token(1), |tensor| tensor.ones_like());
    let value = hooks.capture(self_attention.value());
    hooks.edit(self_attention.pattern(), |tensor| tensor.ones_like());
    let pattern = hooks.capture(self_attention.pattern());
    let cross_query = hooks.capture(Layer::decoder(0).cross_attention().query());
    let cross_context = hooks.capture(Layer::decoder(0).cross_attention().head_context());
    model
        .forward_with_hooks(input, &options(), &mut hooks)
        .unwrap();
    assert_eq!(
        hooks
            .get(&value)
            .unwrap()
            .slice([0..1, 0..2, 1..2, 0..2])
            .abs()
            .sum()
            .into_scalar(),
        0.0
    );
    let pattern = hooks.get(&pattern).unwrap();
    assert_eq!(
        pattern
            .clone()
            .slice([0..1, 0..2, 1..2, 0..3])
            .abs()
            .sum()
            .into_scalar(),
        0.0
    );
    assert_eq!(
        pattern
            .clone()
            .slice([0..1, 0..2, 0..3, 1..2])
            .abs()
            .sum()
            .into_scalar(),
        0.0
    );
    assert_eq!(
        pattern
            .slice([0..1, 0..2, 0..1, 2..3])
            .abs()
            .sum()
            .into_scalar(),
        0.0
    );
    for tensor in [
        hooks.get(&cross_query).unwrap(),
        hooks.get(&cross_context).unwrap(),
    ] {
        assert_eq!(
            tensor
                .slice([0..1, 0..2, 1..2, 0..2])
                .abs()
                .sum()
                .into_scalar(),
            0.0
        );
    }
}

/// Different source and target padding across batch rows preserves ordinary logits and diagnostics.
#[test]
fn t5_mixed_padded_batch_retains_valid_rows_and_zero_diagnostics() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::GatedGelu),
        &device,
    )
    .unwrap();
    let input = || Seq2SeqInput {
        encoder: CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3], [4, 5, 0]], &device),
            token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                [[true, true, true], [true, true, false]].into(),
                &device,
            ))),
            position_ids: None,
        },
        decoder: CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                [[0, 6, 0, 7], [8, 9, 0, 0]],
                &device,
            ),
            token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                [[false, true, false, true], [true, true, false, false]].into(),
                &device,
            ))),
            position_ids: None,
        },
    };
    let expected = model.forward(input(), &options()).unwrap();
    let mut hooks = Hooks::new();
    let mlp = hooks.capture(Layer::decoder(0).mlp_activation());
    let actual = model
        .forward_with_hooks(input(), &options(), &mut hooks)
        .unwrap();
    assert_eq!(hooks.get(&mlp).unwrap().dims(), [2, 4, 8]);
    actual
        .decoder
        .logits
        .into_data()
        .assert_approx_eq::<f32>(&expected.decoder.logits.into_data(), Tolerance::default());
    for (actual, expected) in actual
        .decoder
        .hidden_states
        .into_iter()
        .zip(expected.decoder.hidden_states)
    {
        actual
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }
}

/// Editing an original valid token matches compact execution and editing padding remains a no-op.
#[test]
fn t5_padded_interventions_address_original_token_positions() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    let padded = Seq2SeqInput {
        encoder: tokens(&device),
        decoder: CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 0, 2]], &device),
            token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                [[false, true, false, true]].into(),
                &device,
            ))),
            position_ids: None,
        },
    };
    let compact = Seq2SeqInput {
        encoder: tokens(&device),
        decoder: CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
            token_mask: None,
            position_ids: None,
        },
    };
    let mut padded_hooks = Hooks::new();
    padded_hooks.edit(Layer::decoder(0).mlp_activation().token(3), |tensor| {
        tensor.zeros_like()
    });
    padded_hooks.edit(Layer::decoder(0).mlp_activation().token(2), |tensor| {
        tensor.ones_like()
    });
    let padded_capture = padded_hooks.capture(Layer::decoder(0).mlp_activation().token(2));
    let mut compact_hooks = Hooks::new();
    compact_hooks.edit(Layer::decoder(0).mlp_activation().token(1), |tensor| {
        tensor.zeros_like()
    });
    let actual = model
        .forward_with_hooks(
            padded,
            &ForwardOptions {
                logits: LogitsSelection::Positions(vec![3]),
                ..options()
            },
            &mut padded_hooks,
        )
        .unwrap();
    let expected = model
        .forward_with_hooks(
            compact,
            &ForwardOptions {
                logits: LogitsSelection::Positions(vec![1]),
                ..options()
            },
            &mut compact_hooks,
        )
        .unwrap();
    assert_eq!(
        padded_hooks
            .get(&padded_capture)
            .unwrap()
            .abs()
            .sum()
            .into_scalar(),
        0.0
    );
    actual
        .decoder
        .logits
        .into_data()
        .assert_approx_eq::<f32>(&expected.decoder.logits.into_data(), Tolerance::default());
}

/// Identity hooks preserve embedding, attention, and gated-MLP gradients with gapped padding.
#[cfg(feature = "autodiff")]
#[test]
fn t5_padded_noop_hooks_preserve_gradients() {
    use burn::backend::Autodiff;
    /// Autodiff wrapper used to compare ordinary and hooked parameter gradients.
    type Ad = Autodiff<TestBackend>;
    let device = Default::default();
    let model =
        T5ForConditionalGeneration::<Ad>::init(config(T5FeedForwardProjection::GatedGelu), &device)
            .unwrap();
    let padded = || {
        let mut input = input(&device);
        input.decoder.token_mask = Some(TokenMask::new(Tensor::<Ad, 2, Bool>::from_bool(
            [[true, false, true]].into(),
            &device,
        )));
        input
    };
    let ordinary = model
        .forward(padded(), &options())
        .unwrap()
        .decoder
        .logits
        .sum()
        .backward();
    let mut hooks = Hooks::new();
    hooks.edit(Layer::decoder(0).mlp_activation(), |tensor| tensor);
    let hooked = model
        .forward_with_hooks(padded(), &options(), &mut hooks)
        .unwrap()
        .decoder
        .logits
        .sum()
        .backward();
    for parameter in [
        model.shared.weight.val(),
        model.decoder.block[0]
            .self_attention
            .self_attention
            .attention
            .q_proj
            .weight
            .val(),
        model.decoder.block[0]
            .feed_forward
            .gated
            .as_ref()
            .unwrap()
            .gate_proj
            .weight
            .val(),
    ] {
        parameter
            .grad(&hooked)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(
                &parameter.grad(&ordinary).unwrap().into_data(),
                Tolerance::default(),
            );
    }
}

/// Invalid decoder-mask validation clears captures from a preceding successful forward pass.
#[test]
fn t5_invalid_decoder_mask_clears_previous_forward_captures() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu),
        &device,
    )
    .unwrap();
    let mut hooks = Hooks::new();
    let saved = hooks.capture(Layer::decoder(0).mlp_activation());
    model
        .forward_with_hooks(input(&device), &options(), &mut hooks)
        .unwrap();
    assert!(hooks.get(&saved).is_ok());
    let mut malformed = input(&device);
    malformed.decoder.token_mask = Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, true]].into(),
        &device,
    )));
    assert!(
        model
            .forward_with_hooks(malformed, &options(), &mut hooks)
            .is_err()
    );
    assert!(hooks.get(&saved).is_err());
}
