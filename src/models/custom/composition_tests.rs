//! Public staged execution, explicit connectivity, and open registered-module composition.

use super::*;
use crate::{
    attention::AttentionPrecisionMode,
    experiment::{
        AttentionPolicy, EncoderContext, ExecutionContext, HiddenTransform, ResidualBlock,
        WidthAdapter,
    },
    feed_forward::DenseFeedForwardConfig,
    models::custom::ModularLayer,
};
use burn::{
    nn::{Linear, LinearConfig},
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::Int,
};

/// Resuming a decoder at the next layer matches full execution and rejects replay or repeated final normalization.
#[test]
fn staged_decoder_splits_resume_exactly_and_normalizes_only_when_requested() {
    let mut config = spec(NormPlacement::Pre);
    config.layers.push(config.layers[0].clone());
    let model = ModularCausalLm::<B>::init(config, &Default::default())
        .unwrap()
        .map(&mut FixedParameters::default());
    let expected = model.forward_hidden(input([[1, 2, 3]])).unwrap();
    let state = model.embed(input([[1, 2, 3]])).unwrap();
    let prefix = model.run_layers(state, 0..1).unwrap();
    assert_eq!(prefix.next_layer, 1);
    assert!(!prefix.normalized);
    assert!(model.run_layers(prefix.clone(), 0..1).is_err());
    let mut hooks = Hooks::new();
    let saved = hooks.capture(Layer::decoder(1).residual_pre());
    let raw = prefix.hidden.clone();
    let completed = model
        .run_layers_with_hooks(prefix.clone(), 1..2, &mut hooks)
        .unwrap();
    assert_close(hooks.get(&saved).unwrap(), raw);
    let completed = model.normalize(completed).unwrap();
    assert_close(completed.hidden.clone(), expected);
    assert!(model.normalize(completed).is_err());
    assert!(model.normalize(prefix).unwrap().normalized);
}

/// A custom first-layer policy admits future keys, zeros excluded rows and padding, and leaves the next layer causal.
#[test]
fn per_layer_connectivity_preserves_padding_and_allows_fully_excluded_rows() {
    let mut config = spec(NormPlacement::Pre);
    config.layers.push(config.layers[0].clone());
    let model = ModularCausalLm::<B>::init(config, &Default::default())
        .unwrap()
        .map(&mut FixedParameters::default())
        .with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    let mut tokens = input([[1, 2, 3]]);
    tokens.token_mask = Some(TokenMask::new(Tensor::from_bool(
        [[true, true, false]].into(),
        &Default::default(),
    )));
    let mut state = model.embed(tokens).unwrap();
    // This explicit noncausal matrix allows row zero to read the future, but row one
    // has no permitted key. Padding independently excludes column two.
    let allowed = Tensor::from_bool(
        [[[
            [true, true, true],
            [false, false, false],
            [true, true, true],
        ]]]
        .into(),
        &Default::default(),
    );
    let bias = Tensor::zeros([1, 1, 1, 1], &Default::default());
    state.context = state.context.with_layer_attention(
        Layer::decoder(0),
        AttentionPolicy::custom(allowed).with_bias(bias),
    );
    let mut hooks = Hooks::new();
    let first = hooks.capture(Layer::decoder(0).self_attention().pattern());
    let second = hooks.capture(Layer::decoder(1).self_attention().pattern());
    let result = model
        .run_layers_with_hooks(state, 0..2, &mut hooks)
        .unwrap();
    let first = hooks.get(&first).unwrap();
    assert!(
        first
            .clone()
            .slice([0..1, 0..2, 0..1, 1..2])
            .min()
            .into_scalar()
            > 0.0
    );
    assert_eq!(
        first
            .clone()
            .slice([0..1, 0..2, 1..2, 0..3])
            .abs()
            .max()
            .into_scalar(),
        0.0
    );
    assert_eq!(
        first
            .slice([0..1, 0..2, 0..3, 2..3])
            .abs()
            .max()
            .into_scalar(),
        0.0
    );
    assert_eq!(
        hooks
            .get(&second)
            .unwrap()
            .slice([0..1, 0..2, 0..1, 1..3])
            .abs()
            .max()
            .into_scalar(),
        0.0
    );
    assert!(
        result
            .hidden
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}

/// An invalid bidirectional policy on a later DeltaNet block is rejected before an earlier-layer hook can run.
#[test]
fn unsupported_later_recurrent_policy_is_rejected_before_any_callback() {
    let mut config = spec(NormPlacement::Pre);
    let mut recurrent = config.layers[0].clone();
    recurrent.mixer = MixerSpec::DeltaNet {
        key_heads: 1,
        value_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        convolution_kernel: 2,
    };
    config.layers.push(recurrent);
    let model = ModularCausalLm::<B>::init(config, &Default::default()).unwrap();
    let mut state = model.embed(input([[1, 2, 3]])).unwrap();
    state.context = state
        .context
        .with_layer_attention(Layer::decoder(1), AttentionPolicy::bidirectional());
    let called = std::cell::Cell::new(false);
    let mut hooks = Hooks::new();
    hooks.edit(Layer::decoder(0).residual_pre(), |hidden| {
        called.set(true);
        hidden
    });
    assert!(
        model
            .run_layers_with_hooks(state, 0..2, &mut hooks)
            .is_err()
    );
    assert!(!called.get());
}

/// Staged source/target execution matches full states, while a separate cross policy restricts attention to one source key.
#[test]
fn staged_encoder_decoder_matches_full_graph_and_accepts_separate_cross_policy() {
    let model = seq2seq().with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    let expected = model
        .forward(
            seq_input(),
            &ForwardOptions {
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap();
    let source = model
        .normalize_encoder(
            model
                .run_encoder_layers(model.embed_encoder(input([[1, 2, 3]])).unwrap(), 0..1)
                .unwrap(),
        )
        .unwrap();
    assert_close(source.hidden.clone(), expected.encoder.hidden_state);
    let source = EncoderContext {
        hidden: source.hidden,
        positions: source.context.positions,
        token_mask: source.context.token_mask,
    };
    let target = model.embed_decoder(input([[4, 5, 6]]), source).unwrap();
    let completed = model
        .normalize_decoder(model.run_decoder_layers(target.clone(), 0..1).unwrap())
        .unwrap();
    assert_close(
        completed.hidden,
        expected.decoder.final_hidden_state.unwrap(),
    );

    let mut target = target;
    let allowed = Tensor::from_bool([[[[true, false, false]]]].into(), &Default::default());
    target.context = target
        .context
        .with_cross_attention(AttentionPolicy::custom(allowed));
    let mut hooks = Hooks::new();
    let cross = hooks.capture(Layer::decoder(0).cross_attention().pattern());
    model
        .run_decoder_layers_with_hooks(target, 0..1, &mut hooks)
        .unwrap();
    let pattern = hooks.get(&cross).unwrap();
    assert_eq!(
        pattern
            .clone()
            .slice([0..1, 0..2, 0..3, 0..1])
            .min()
            .into_scalar(),
        1.0
    );
    assert_eq!(
        pattern
            .slice([0..1, 0..2, 0..3, 1..3])
            .abs()
            .max()
            .into_scalar(),
        0.0
    );
}

/// Encoder MLA exposes compressed query/KV captures; ablation affects decoder logits and restoring latents restores them.
#[test]
fn shared_mla_runs_in_encoder_decoder_and_exposes_compressed_activations() {
    let mut encoder = spec(NormPlacement::Pre);
    encoder.layers[0].mixer = MixerSpec::Mla(crate::architecture::MlaSpec {
        query_heads: 2,
        content_head_dim: 2,
        rotary_head_dim: 2,
        value_head_dim: 3,
        key_value_lora_rank: 4,
        query_lora_rank: Some(3),
        position: PositionEncodingSpec::None,
    });
    let model = ModularSeq2SeqLm::<B>::init(
        Seq2SeqSpec {
            encoder,
            decoder: spec(NormPlacement::Pre),
            cross_attention_heads: 2,
            cross_attention_head_dim: 4,
        },
        &Default::default(),
    )
    .unwrap()
    .map(&mut FixedParameters::default());
    let mut hooks = Hooks::new();
    let points = Layer::encoder(0).self_attention();
    let latent = hooks.capture(points.kv_latent());
    let query = hooks.capture(points.query_latent());
    let pattern = hooks.capture(points.pattern());
    let reference = model
        .forward_with_hooks(seq_input(), &ForwardOptions::default(), &mut hooks)
        .unwrap();
    let saved = hooks.get(&latent).unwrap();
    assert_eq!(saved.dims(), [1, 3, 4]);
    assert_eq!(hooks.get(&query).unwrap().dims(), [1, 3, 3]);
    assert!(
        hooks
            .get(&pattern)
            .unwrap()
            .slice([0..1, 0..2, 0..1, 1..3])
            .min()
            .into_scalar()
            > 0.0
    );
    hooks.edit(points.kv_latent(), |hidden| hidden.zeros_like());
    let removed = model
        .forward_with_hooks(seq_input(), &ForwardOptions::default(), &mut hooks)
        .unwrap();
    assert!(difference(removed.decoder.logits, reference.decoder.logits.clone()) > 1e-5);
    hooks.replace(points.kv_latent(), saved);
    let restored = model
        .forward_with_hooks(seq_input(), &ForwardOptions::default(), &mut hooks)
        .unwrap();
    assert_close(restored.decoder.logits, reference.decoder.logits);
}

/// A consumer-defined mixer with registered parameters and completely different sequence math.
#[derive(Module, Debug)]
struct MeanMixer<B: burn::tensor::backend::Backend> {
    /// Learned projection applied to the broadcast sequence mean.
    projection: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> HiddenTransform<B> for MeanMixer<B> {
    /// Projects the mean over the token axis after broadcasting it to every original token position.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        _context: &ExecutionContext<B>,
        _layer: Layer,
        _hooks: &mut Hooks<'_, B>,
    ) -> crate::Result<Tensor<B, 3>> {
        let shape = hidden.dims();
        crate::operations::linear(hidden.mean_dim(1).expand(shape), &self.projection)
    }
}

/// Wraps a learned mean mixer and dense MLP at width 12 with input/output adapters of widths 8 and 6.
fn open_composition()
-> WidthAdapter<B, ResidualBlock<B, MeanMixer<B>, crate::feed_forward::DenseFeedForward<B>>> {
    let device = Default::default();
    let mixer = MeanMixer {
        projection: LinearConfig::new(12, 12).init(&device),
    };
    let ffn = DenseFeedForwardConfig::new(12, 18).init(&device).unwrap();
    let block = ResidualBlock::without_normalization(12, mixer, ffn).unwrap();
    WidthAdapter::new(block, 8, 12, 6, &device)
        .unwrap()
        .map(&mut FixedParameters::default())
}

/// Custom mixer and adapters contribute all parameters, survive a Burn record roundtrip, and reject invalid context or width.
#[test]
fn open_mixer_and_width_adapters_are_registered_and_survive_record_roundtrip() {
    let module = open_composition();
    let context = ExecutionContext::new(
        Tensor::<B, 2, Int>::from_data([[0, 1, 2]], &Default::default()),
        None,
    )
    .unwrap();
    let input = Tensor::<B, 3>::ones([1, 3, 8], &Default::default());
    let mut hooks = Hooks::new();
    let activation = hooks.capture(Layer::encoder(0).mlp_activation());
    let output = module
        .run_with_hooks(input.clone(), &context, Layer::encoder(0), &mut hooks)
        .unwrap();
    assert_eq!(output.dims(), [1, 3, 6]);
    assert_eq!(hooks.get(&activation).unwrap().dims(), [1, 3, 18]);
    // 8*12 input adapter + (12*12+12) custom mixer + 12*18*2 MLP + 12*6 output adapter.
    assert_eq!(module.num_params(), 96 + 156 + 432 + 72);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = recorder.record(module.clone().into_record(), ()).unwrap();
    let restored =
        open_composition().load_record(recorder.load(bytes, &Default::default()).unwrap());
    assert_close(
        restored.run(input, &context, Layer::encoder(0)).unwrap(),
        output,
    );
    let mut invalid_context = context.clone();
    invalid_context.positions = invalid_context.positions.slice([0..1, 0..2]);
    assert!(
        module
            .run(
                Tensor::ones([1, 3, 8], &Default::default()),
                &invalid_context,
                Layer::encoder(0)
            )
            .is_err()
    );
    assert!(
        module
            .run(
                Tensor::ones([1, 3, 7], &Default::default()),
                &context,
                Layer::encoder(0)
            )
            .is_err()
    );
}

/// An 8-wide block composes with a 12-wide block through adapters while preserving the external residual shape.
#[test]
fn public_modular_blocks_can_be_initialized_with_independent_widths() {
    let layer = &spec(NormPlacement::Pre).layers[0];
    let narrow = ModularLayer::<B>::init(layer, 8, None, &Default::default()).unwrap();
    let wide = ModularLayer::<B>::init(layer, 12, None, &Default::default()).unwrap();
    let wide = WidthAdapter::new(wide, 8, 12, 8, &Default::default()).unwrap();
    let context = ExecutionContext::new(
        Tensor::<B, 2, Int>::from_data([[0, 1, 2]], &Default::default()),
        None,
    )
    .unwrap();
    let hidden = narrow
        .run(
            Tensor::ones([1, 3, 8], &Default::default()),
            &context,
            Layer::encoder(0),
        )
        .unwrap();
    assert_eq!(
        wide.run(hidden, &context, Layer::encoder(1))
            .unwrap()
            .dims(),
        [1, 3, 8]
    );
}

/// Backpropagation reaches the input, custom mixer projection, and both learned width adapters.
#[cfg(feature = "autodiff")]
#[test]
fn open_custom_module_and_both_width_adapters_receive_nonzero_gradients() {
    /// Reverse-mode backend used to check the custom mixer and both width adapters.
    type AD = burn::backend::Autodiff<B>;
    let device = Default::default();
    let mixer = MeanMixer::<AD> {
        projection: LinearConfig::new(12, 12).init(&device),
    };
    let ffn = DenseFeedForwardConfig::new(12, 18).init(&device).unwrap();
    let block = ResidualBlock::without_normalization(12, mixer, ffn).unwrap();
    let module = WidthAdapter::new(block, 8, 12, 6, &device).unwrap();
    let context =
        ExecutionContext::new(Tensor::<AD, 2, Int>::from_data([[0, 1, 2]], &device), None).unwrap();
    let input = Tensor::<AD, 3>::ones([1, 3, 8], &device).require_grad();
    let output = module
        .run(input.clone(), &context, Layer::encoder(0))
        .unwrap();
    let gradients = output.square().sum().backward();
    assert!(input.grad(&gradients).unwrap().abs().max().into_scalar() > 0.0);
    for weight in [
        module.input_projection.as_ref().unwrap().weight.val(),
        module.transform.mixer.projection.weight.val(),
        module.output_projection.as_ref().unwrap().weight.val(),
    ] {
        assert!(weight.grad(&gradients).unwrap().abs().max().into_scalar() > 0.0);
    }
}

/// Wrong-stack and out-of-range attention policies fail staged validation before callbacks are invoked.
#[test]
fn staged_models_reject_invalid_attention_addresses_before_callbacks() {
    let model = causal(NormPlacement::Pre);
    for invalid in [Layer::encoder(0), Layer::decoder(1)] {
        let mut state = model.embed(input([[1, 2, 3]])).unwrap();
        state.context = state
            .context
            .with_layer_attention(invalid, AttentionPolicy::bidirectional());
        let called = std::cell::Cell::new(false);
        let mut hooks = Hooks::new();
        hooks.edit(Layer::decoder(0).residual_pre(), |hidden| {
            called.set(true);
            hidden
        });
        assert!(
            model
                .run_layers_with_hooks(state, 0..1, &mut hooks)
                .is_err()
        );
        assert!(!called.get());
    }
    let model = seq2seq();
    let mut source = model.embed_encoder(input([[1, 2, 3]])).unwrap();
    source.context = source
        .context
        .with_layer_cross_attention(Layer::encoder(0), AttentionPolicy::bidirectional());
    assert!(model.run_encoder_layers(source, 0..1).is_err());
    let mut source = model.embed_encoder(input([[1, 2, 3]])).unwrap();
    source.context = source
        .context
        .with_layer_attention(Layer::decoder(1), AttentionPolicy::bidirectional());
    assert!(model.run_encoder_layers(source, 0..1).is_err());
}

/// Forged layer cursors beyond either stack are rejected by final normalization.
#[test]
fn staged_models_reject_forged_normalization_boundaries() {
    let model = causal(NormPlacement::Pre);
    let mut state = model.embed(input([[1, 2, 3]])).unwrap();
    state.next_layer = 2;
    assert!(model.normalize(state).is_err());
    let model = seq2seq();
    let mut source = model.embed_encoder(input([[1, 2, 3]])).unwrap();
    source.next_layer = 2;
    assert!(model.normalize_encoder(source).is_err());
}

/// F64 residuals against F32 parameters fail with dtype context before staged hooks or matrix multiplication.
#[test]
fn staged_dtype_errors_are_contextual_before_matmul_or_callbacks() {
    let model = causal(NormPlacement::Pre);
    let mut state = model.embed(input([[1, 2, 3]])).unwrap();
    state.hidden = state.hidden.cast(burn::tensor::DType::F64);
    let called = std::cell::Cell::new(false);
    let mut hooks = Hooks::new();
    hooks.edit(Layer::decoder(0).residual_pre(), |hidden| {
        called.set(true);
        hidden
    });
    let error = model
        .run_layers_with_hooks(state.clone(), 0..1, &mut hooks)
        .unwrap_err();
    assert!(error.to_string().contains("dtype"));
    assert!(!called.get());
    assert!(model.normalize(state.clone()).is_err());
    assert!(
        model.layers()[0]
            .run(state.hidden, &state.context, Layer::decoder(0))
            .is_err()
    );
    let model = seq2seq();
    let mut source = model.embed_encoder(input([[1, 2, 3]])).unwrap();
    source.hidden = source.hidden.cast(burn::tensor::DType::F64);
    assert!(model.run_encoder_layers(source.clone(), 0..1).is_err());
    assert!(model.normalize_encoder(source).is_err());
}

/// Explicit bidirectional attention removes a trailing-window restriction consistently in direct and staged execution.
#[test]
fn explicit_bidirectional_policy_replaces_a_trailing_window_consistently() {
    let mut config = spec(NormPlacement::Pre);
    let MixerSpec::Attention(attention) = config.layers[0].mixer.clone() else {
        unreachable!()
    };
    config.layers[0].mixer = MixerSpec::SlidingAttention(crate::architecture::SlidingWindowSpec {
        attention,
        window: 1,
    });
    let model = ModularCausalLm::<B>::init(config, &Default::default())
        .unwrap()
        .map(&mut FixedParameters::default())
        .with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    let mut hooks = Hooks::new();
    let saved = hooks.capture(Layer::decoder(0).self_attention().pattern());
    model
        .forward_with_hooks(input([[1, 2, 3]]), &ForwardOptions::default(), &mut hooks)
        .unwrap();
    assert_eq!(
        hooks
            .get(&saved)
            .unwrap()
            .slice([0..1, 0..2, 2..3, 0..1])
            .max()
            .into_scalar(),
        0.0
    );
    let expected = model
        .forward_hidden_bidirectional_with_hooks(input([[1, 2, 3]]), &mut hooks)
        .unwrap();
    assert!(
        hooks
            .get(&saved)
            .unwrap()
            .slice([0..1, 0..2, 2..3, 0..1])
            .min()
            .into_scalar()
            > 0.0
    );
    let mut state = model.embed(input([[1, 2, 3]])).unwrap();
    state.context = state
        .context
        .with_attention(AttentionPolicy::bidirectional());
    let observed = model
        .normalize(model.run_layers(state, 0..1).unwrap())
        .unwrap();
    assert_close(observed.hidden, expected);
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config {
        cases: 16,
        rng_seed: proptest::test_runner::RngSeed::Fixed(0x4355_5354_4f4d_5354),
        ..Default::default()
    })]

    /// Generated depths, split points, and norm placements preserve both full-stack results and reject replay or renormalization.
    #[test]
    fn modular_partitions_preserve_both_stacks_and_reject_replayed_blocks(
        depth in 1usize..4,
        encoder_cut_seed in 0usize..4,
        decoder_cut_seed in 0usize..4,
        placement_index in 0usize..3,
        first_token in 1i32..16,
    ) {
        let placement = [NormPlacement::Pre, NormPlacement::Post, NormPlacement::Sandwich][placement_index];
        let mut stack = spec(placement);
        stack.layers = vec![stack.layers[0].clone(); depth];
        let model = ModularSeq2SeqLm::<B>::init(Seq2SeqSpec {
            encoder: stack.clone(),
            decoder: stack,
            cross_attention_heads: 2,
            cross_attention_head_dim: 4,
        }, &Default::default()).unwrap().map(&mut FixedParameters::default());
        let source_tokens = input([[first_token, 2, 3]]);
        let target_tokens = input([[4, 5, 6]]);
        let expected = model.forward(Seq2SeqInput {
            encoder: source_tokens.clone(),
            decoder: target_tokens.clone(),
        }, &ForwardOptions { hidden_states: true, ..Default::default() }).unwrap();
        let encoder_cut = encoder_cut_seed % (depth + 1);
        let source = model.run_encoder_layers(model.embed_encoder(source_tokens).unwrap(), 0..encoder_cut).unwrap();
        proptest::prop_assert_eq!(source.next_layer, encoder_cut);
        if encoder_cut > 0 {
            proptest::prop_assert!(model.run_encoder_layers(source.clone(), 0..encoder_cut).is_err());
        }
        let source = model.normalize_encoder(model.run_encoder_layers(source, encoder_cut..depth).unwrap()).unwrap();
        proptest::prop_assert_eq!(source.hidden.to_data(), expected.encoder.hidden_state.to_data());
        let source = EncoderContext {
            hidden: source.hidden,
            positions: source.context.positions,
            token_mask: source.context.token_mask,
        };
        let decoder_cut = decoder_cut_seed % (depth + 1);
        let target = model.run_decoder_layers(model.embed_decoder(target_tokens, source).unwrap(), 0..decoder_cut).unwrap();
        proptest::prop_assert_eq!(target.next_layer, decoder_cut);
        if decoder_cut > 0 {
            proptest::prop_assert!(model.run_decoder_layers(target.clone(), 0..decoder_cut).is_err());
        }
        let completed = model.normalize_decoder(model.run_decoder_layers(target, decoder_cut..depth).unwrap()).unwrap();
        proptest::prop_assert_eq!(completed.hidden.to_data(), expected.decoder.final_hidden_state.unwrap().to_data());
        proptest::prop_assert!(model.normalize_decoder(completed).is_err());
    }
}
