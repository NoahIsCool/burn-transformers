//! Public experiment API regressions using deterministic CPU-sized architectures.

use burn::{
    backend::Flex,
    module::{Module, ModuleMapper, Param},
    tensor::{Tensor, TensorData, Tolerance},
};

use crate::{
    CausalLmInput, ForwardOptions, LogitsSelection, Seq2SeqInput, TokenMask,
    architecture::{
        Activation, AttentionOptions, AttentionSpec, DecoderLayerSpec, DecoderSpec,
        DenseFeedForwardSpec, FeedForwardSpec, MixerSpec, MixtureOfExpertsSpec, NormPlacement,
        NormalizationSpec, PositionEncodingSpec, ResidualTopology, Seq2SeqSpec,
    },
    intervention::{Hooks, Layer, Stack},
    models::custom::{ModularCausalLm, ModularSeq2SeqLm},
    moe::RoutingPolicy,
};

/// CPU backend with F32 tensor values and I32 indices for deterministic intervention fixtures.
type B = Flex<f32, i32>;

/// Builds one attention/MLP block with the requested normalization placement and a 16-token vocabulary.
fn spec(placement: NormPlacement) -> DecoderSpec {
    let norm = NormalizationSpec::Rms {
        epsilon: 1e-6,
        zero_centered: false,
    };
    DecoderSpec {
        vocabulary_size: 16,
        hidden_size: 8,
        layers: vec![DecoderLayerSpec {
            mixer: MixerSpec::Attention(AttentionSpec {
                query_heads: 2,
                key_value_heads: 1,
                head_dim: 4,
                query_key_norm: false,
                output_gate: false,
                position: PositionEncodingSpec::None,
            }),
            feed_forward: FeedForwardSpec::Dense(DenseFeedForwardSpec {
                intermediate_size: 12,
                activation: Activation::Relu,
                bias: false,
            }),
            mixer_norm: norm,
            feed_forward_norm: norm,
            norm_placement: placement,
            residual: ResidualTopology::Add,
        }],
        final_norm: norm,
        tie_word_embeddings: false,
        max_position_embeddings: 16,
    }
}

/// Tracks parameter traversal so every matrix receives a reproducible distinct sinusoidal pattern.
#[derive(Default)]
struct FixedParameters(
    /// Number of floating-point parameters already assigned fixture values.
    usize,
);

impl ModuleMapper<B> for FixedParameters {
    /// Assigns unit normalization scales and deterministic sinusoidal matrix weights per parameter ordinal.
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        self.0 += 1;
        let ordinal = self.0;
        param.map(|tensor| {
            let values = (0..tensor.dims().iter().product())
                .map(|index| {
                    if D == 1 {
                        1.0
                    } else {
                        (((index + 1) * (ordinal + 3)) as f32 * 0.17).sin() * 0.2
                    }
                })
                .collect::<Vec<_>>();
            Tensor::from_data(TensorData::new(values, tensor.dims()), &tensor.device())
        })
    }
}

/// Creates one unmasked three-token row on the default CPU device.
fn input(ids: [[i32; 3]; 1]) -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: Tensor::from_data(ids, &Default::default()),
        token_mask: None,
        position_ids: None,
    }
}

/// Pairs fixed three-token source and target rows for encoder-decoder hook tests.
fn seq_input() -> Seq2SeqInput<B> {
    Seq2SeqInput {
        encoder: input([[1, 2, 3]]),
        decoder: input([[4, 5, 6]]),
    }
}

/// Initializes a one-block decoder and materializes reproducible fixture weights.
fn causal(placement: NormPlacement) -> ModularCausalLm<B> {
    ModularCausalLm::init(spec(placement), &Default::default())
        .unwrap()
        .map(&mut FixedParameters::default())
}

/// Initializes a pre-normalized encoder-decoder graph with deterministic weights in both stacks.
fn seq2seq() -> ModularSeq2SeqLm<B> {
    ModularSeq2SeqLm::init(
        Seq2SeqSpec {
            encoder: spec(NormPlacement::Pre),
            decoder: spec(NormPlacement::Pre),
            cross_attention_heads: 2,
            cross_attention_head_dim: 4,
            cross_attention_options: AttentionOptions::default(),
        },
        &Default::default(),
    )
    .unwrap()
    .map(&mut FixedParameters::default())
}

/// Requires F32 tensors to agree within an absolute tolerance of one millionth.
fn assert_close(actual: Tensor<B, 3>, expected: Tensor<B, 3>) {
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));
}

/// Returns the largest absolute tensor difference to detect consequential interventions.
fn difference(actual: Tensor<B, 3>, expected: Tensor<B, 3>) -> f32 {
    (actual - expected).abs().max().into_scalar()
}

/// Identity edits preserve logits for pre, post, and sandwich norms; final-state captures match hidden-only output.
#[test]
fn identity_hooks_preserve_each_normalization_topology_and_hidden_only_output() {
    for placement in [
        NormPlacement::Pre,
        NormPlacement::Post,
        NormPlacement::Sandwich,
    ] {
        let model = causal(placement);
        let options = ForwardOptions::default();
        let reference = model.forward(input([[1, 2, 3]]), &options).unwrap();
        let mut hooks = Hooks::new();
        hooks.try_edit(Layer::decoder(0).attention_output(), Ok);
        hooks.try_edit(Layer::decoder(0).mlp_activation(), Ok);
        let observed = model
            .forward_with_hooks(input([[1, 2, 3]]), &options, &mut hooks)
            .unwrap();
        assert_close(observed.logits, reference.logits);

        let mut capture = Hooks::new();
        let saved = capture.capture(Stack::Decoder.final_norm());
        let hidden = model
            .forward_hidden_with_hooks(input([[1, 2, 3]]), &mut capture)
            .unwrap();
        assert_eq!(hidden.dims(), [1, 3, 8]);
        assert_close(hidden.clone(), capture.get(&saved).unwrap());
        assert_close(hidden, model.forward_hidden(input([[1, 2, 3]])).unwrap());
    }
}

/// Captured MLP activations equal ReLU of the pre-activation; ablation changes logits and a clean patch restores them.
#[test]
fn gate_activation_capture_ablation_and_clean_patch_change_downstream_computation() {
    let model = causal(NormPlacement::Pre);
    let options = ForwardOptions::default();
    let point = Layer::decoder(0).mlp_activation();
    let mut clean = Hooks::new();
    let saved_pre = clean.capture(Layer::decoder(0).mlp_pre_activation());
    let saved_activation = clean.capture(point.clone());
    let reference = model
        .forward_with_hooks(input([[1, 2, 3]]), &options, &mut clean)
        .unwrap();
    let pre = clean.get(&saved_pre).unwrap();
    let activation = clean.get(&saved_activation).unwrap();
    assert_eq!(activation.dims(), [1, 3, 12]);
    assert_close(activation.clone(), burn::tensor::activation::relu(pre));

    let mut ablate = Hooks::new();
    ablate.try_edit(point.clone(), |hidden| Ok(hidden.zeros_like()));
    let removed = model
        .forward_with_hooks(input([[1, 2, 3]]), &options, &mut ablate)
        .unwrap();
    assert!(difference(removed.logits, reference.logits.clone()) > 1e-5);

    let mut patch = Hooks::new();
    patch.try_edit(point.clone(), |hidden| Ok(hidden.zeros_like()));
    patch.try_edit(point.clone(), move |_| Ok(activation.clone()));
    let restored = model
        .forward_with_hooks(input([[1, 2, 3]]), &options, &mut patch)
        .unwrap();
    assert_close(restored.logits, reference.logits);
}

/// Encoder final-state edits affect decoder logits through cross-attention, and restoring the source state restores logits.
#[test]
fn seq2seq_hooks_keep_both_stacks_and_encoder_edits_reach_cross_attention() {
    let model = seq2seq();
    let options = ForwardOptions::default();
    let encoder_point = Stack::Encoder.final_norm();
    let decoder_point = Layer::decoder(0).cross_attention_output();
    let mut clean = Hooks::new();
    let saved_encoder = clean.capture(encoder_point.clone());
    let saved_decoder = clean.capture(decoder_point);
    let reference = model
        .forward_with_hooks(seq_input(), &options, &mut clean)
        .unwrap();
    let encoded = clean.get(&saved_encoder).unwrap();
    assert_close(encoded.clone(), reference.encoder.hidden_state);
    assert_eq!(clean.get(&saved_decoder).unwrap().dims(), [1, 3, 8]);

    let mut ablate = Hooks::new();
    ablate.try_edit(encoder_point.clone(), |hidden| Ok(hidden.zeros_like()));
    let removed = model
        .forward_with_hooks(seq_input(), &options, &mut ablate)
        .unwrap();
    assert!(difference(removed.decoder.logits, reference.decoder.logits.clone()) > 1e-5);

    let mut patch = Hooks::new();
    patch.try_edit(encoder_point, move |_| Ok(encoded.clone()));
    let restored = model
        .forward_with_hooks(seq_input(), &options, &mut patch)
        .unwrap();
    assert_close(restored.decoder.logits, reference.decoder.logits);
}

/// Encoding with a decoder-only activation request fails before returning an encoder result.
#[test]
fn encoder_only_execution_rejects_decoder_requests() {
    let model = seq2seq();
    let mut hooks = Hooks::new();
    hooks.capture(Layer::decoder(0).mlp_activation());
    assert!(
        model
            .encode_with_hooks(input([[1, 2, 3]]), &ForwardOptions::default(), &mut hooks)
            .is_err()
    );
}

/// Changing the last token preserves earlier causal logits but changes them under bidirectional attention.
#[test]
fn future_token_changes_only_affect_earlier_outputs_with_bidirectional_attention() {
    let model = causal(NormPlacement::Pre);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let causal_clean = model.forward(input([[1, 2, 3]]), &options).unwrap().logits;
    let causal_changed = model.forward(input([[1, 2, 7]]), &options).unwrap().logits;
    assert_close(
        causal_clean.slice([0..1, 0..2, 0..16]),
        causal_changed.slice([0..1, 0..2, 0..16]),
    );
    let full_clean = model
        .forward_bidirectional(input([[1, 2, 3]]), &options)
        .unwrap()
        .logits;
    let full_changed = model
        .forward_bidirectional(input([[1, 2, 7]]), &options)
        .unwrap()
        .logits;
    assert!(
        difference(
            full_clean.slice([0..1, 0..2, 0..16]),
            full_changed.slice([0..1, 0..2, 0..16])
        ) > 1e-5
    );
}

/// A DeltaNet mixer rejects bidirectional execution because its recurrence is causal.
#[test]
fn bidirectional_execution_rejects_causal_recurrent_mixers() {
    let mut spec = spec(NormPlacement::Pre);
    spec.layers[0].mixer = MixerSpec::DeltaNet {
        key_heads: 1,
        value_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        convolution_kernel: 2,
    };
    let model = ModularCausalLm::<B>::init(spec, &Default::default()).unwrap();
    assert!(
        model
            .forward_bidirectional(input([[1, 2, 3]]), &ForwardOptions::default())
            .is_err()
    );
}

/// Head-scoped context and native KV captures have expected shapes; removing and restoring one head changes and restores logits.
#[test]
fn selected_attention_head_can_be_removed_and_patched_with_native_kv_capture() {
    let model = causal(NormPlacement::Pre);
    let point = Layer::decoder(0).self_attention().head_context().head(1);
    let mut hooks = Hooks::new();
    let original_head = hooks.capture(point.clone());
    let native_keys = hooks.capture(Layer::decoder(0).self_attention().key());
    let options = ForwardOptions::default();
    let reference = model
        .forward_with_hooks(input([[1, 2, 3]]), &options, &mut hooks)
        .unwrap();
    assert_eq!(hooks.get(&native_keys).unwrap().dims(), [1, 1, 3, 4]);
    let saved = hooks.get(&original_head).unwrap();
    assert_eq!(saved.dims(), [1, 1, 3, 4]);
    let mut remove = Hooks::new();
    remove.edit(point.clone(), |hidden| hidden.zeros_like());
    let ablated = model
        .forward_with_hooks(input([[1, 2, 3]]), &options, &mut remove)
        .unwrap();
    assert!(difference(ablated.logits, reference.logits.clone()) > 1e-5);
    remove.replace(point, saved);
    let restored = model
        .forward_with_hooks(input([[1, 2, 3]]), &options, &mut remove)
        .unwrap();
    assert_close(restored.logits, reference.logits);
}

/// Changing a masked token cannot alter valid hidden states even when attention is bidirectional.
#[test]
fn bidirectional_attention_still_excludes_padded_keys() {
    let model = causal(NormPlacement::Pre);
    let padded = |last| {
        let mut input = input([[1, 2, last]]);
        input.token_mask = Some(TokenMask::new(Tensor::from_bool(
            [[true, true, false]].into(),
            &Default::default(),
        )));
        input
    };
    let clean = model.forward_hidden_bidirectional(padded(3)).unwrap();
    let changed = model.forward_hidden_bidirectional(padded(7)).unwrap();
    assert_close(
        clean.slice([0..1, 0..2, 0..8]),
        changed.slice([0..1, 0..2, 0..8]),
    );
}

/// Unsupported diagnostic options clear captures saved by an earlier successful hooked forward.
#[test]
fn failed_validation_discards_previous_captures() {
    let model = causal(NormPlacement::Pre);
    let mut hooks = Hooks::new();
    let saved = hooks.capture(Stack::Decoder.embedding());
    model
        .forward_with_hooks(input([[1, 2, 3]]), &ForwardOptions::default(), &mut hooks)
        .unwrap();
    assert!(hooks.get(&saved).is_ok());
    let options = ForwardOptions {
        attentions: true,
        ..Default::default()
    };
    assert!(
        model
            .forward_with_hooks(input([[1, 2, 3]]), &options, &mut hooks)
            .is_err()
    );
    assert!(hooks.get(&saved).is_err());
}

/// DeltaNet and sparse MLP layers expose output captures while omitting unavailable query and dense-activation points.
#[test]
fn recurrent_and_moe_layers_advertise_only_executable_points() {
    let mut config = spec(NormPlacement::Pre);
    config.layers[0].mixer = MixerSpec::DeltaNet {
        key_heads: 1,
        value_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        convolution_kernel: 2,
    };
    config.layers[0].feed_forward = FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
        routed_experts: 2,
        experts_per_token: 1,
        shared_experts: 0,
        expert_intermediate_size: 6,
        activation: Activation::Silu,
        renormalize_selected: true,
        routed_scale: 1.0,
        routing: RoutingPolicy::default(),
        shared_expert_gate: false,
        expert_bias: false,
    });
    let model = ModularCausalLm::<B>::init(config, &Default::default())
        .unwrap()
        .map(&mut FixedParameters::default());
    let layer = Layer::decoder(0);
    let points = model.activation_points();
    assert!(points.contains(&layer.attention_output().info()));
    assert!(points.contains(&layer.mlp_output().info()));
    assert!(!points.contains(&layer.self_attention().query().info()));
    assert!(!points.contains(&layer.mlp_activation().info()));
    let mut hooks = Hooks::new();
    let mixer = hooks.capture(layer.attention_output());
    let mlp = hooks.capture(layer.mlp_output());
    model
        .forward_with_hooks(input([[1, 2, 3]]), &ForwardOptions::default(), &mut hooks)
        .unwrap();
    assert_eq!(hooks.get(&mixer).unwrap().dims(), [1, 3, 8]);
    assert_eq!(hooks.get(&mlp).unwrap().dims(), [1, 3, 8]);
}

/// Eager precision exposes score/pattern hooks with zero future causal mass and positive bidirectional mass.
#[test]
fn explicit_eager_attention_exposes_causal_and_bidirectional_patterns() {
    use crate::attention::AttentionPrecisionMode;

    let model = causal(NormPlacement::Pre);
    let points = Layer::decoder(0).self_attention();
    assert!(!model.activation_points().contains(&points.pattern().info()));
    let model = model.with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    assert!(model.activation_points().contains(&points.scores().info()));
    assert!(model.activation_points().contains(&points.pattern().info()));

    let mut hooks = Hooks::new();
    let saved = hooks.capture(points.pattern());
    model
        .forward_with_hooks(input([[1, 2, 3]]), &ForwardOptions::default(), &mut hooks)
        .unwrap();
    let causal_pattern = hooks.get(&saved).unwrap();
    assert_eq!(causal_pattern.dims(), [1, 2, 3, 3]);
    assert_eq!(
        causal_pattern
            .slice([0..1, 0..2, 0..1, 1..3])
            .abs()
            .max()
            .into_scalar(),
        0.0
    );

    model
        .forward_bidirectional_with_hooks(
            input([[1, 2, 3]]),
            &ForwardOptions::default(),
            &mut hooks,
        )
        .unwrap();
    let full_pattern = hooks.get(&saved).unwrap();
    assert!(
        full_pattern
            .slice([0..1, 0..2, 0..1, 1..3])
            .min()
            .into_scalar()
            > 0.0
    );

    let model = model.with_attention_precision(AttentionPrecisionMode::BackendDefault);
    assert!(!model.activation_points().contains(&points.pattern().info()));
}

/// Removing one attention-pattern head changes logits in either direction, and restoring it reproduces the clean result.
#[test]
fn eager_pattern_ablation_and_clean_patch_work_with_each_attention_direction() {
    use crate::attention::AttentionPrecisionMode;

    let model = causal(NormPlacement::Pre)
        .with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    for bidirectional in [false, true] {
        let run = |hooks: &mut Hooks<'_, B>| {
            if bidirectional {
                model.forward_bidirectional_with_hooks(
                    input([[1, 2, 3]]),
                    &ForwardOptions::default(),
                    hooks,
                )
            } else {
                model.forward_with_hooks(input([[1, 2, 3]]), &ForwardOptions::default(), hooks)
            }
            .unwrap()
            .logits
        };
        let selection = Layer::decoder(0).self_attention().pattern().head(0);
        let mut hooks = Hooks::new();
        let saved = hooks.capture(selection.clone());
        let reference = run(&mut hooks);
        let pattern = hooks.get(&saved).unwrap();
        hooks.edit(selection.clone(), |probabilities| {
            probabilities.zeros_like()
        });
        let removed = run(&mut hooks);
        assert!(difference(removed, reference.clone()) > 1e-5);
        hooks.replace(selection, pattern);
        assert_close(run(&mut hooks), reference);
    }
}

/// Eager precision exposes patterns in both stacks and cross-attention; cross-pattern patching restores decoder logits.
#[test]
fn seq2seq_eager_precision_reaches_both_stacks_and_cross_attention_patterns() {
    use crate::attention::AttentionPrecisionMode;

    let model = seq2seq().with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    let encoder_point = Layer::encoder(0).self_attention().pattern();
    let decoder_point = Layer::decoder(0).self_attention().pattern();
    let cross_point = Layer::decoder(0).cross_attention().pattern();
    let catalog = model.activation_points();
    for point in [&encoder_point, &decoder_point, &cross_point] {
        assert!(catalog.contains(&point.info()));
    }
    let mut hooks = Hooks::new();
    let encoder = hooks.capture(encoder_point);
    let decoder = hooks.capture(decoder_point);
    let cross = hooks.capture(cross_point.clone());
    let options = ForwardOptions::default();
    let reference = model
        .forward_with_hooks(seq_input(), &options, &mut hooks)
        .unwrap()
        .decoder
        .logits;
    assert_eq!(hooks.get(&encoder).unwrap().dims(), [1, 2, 3, 3]);
    assert_eq!(hooks.get(&decoder).unwrap().dims(), [1, 2, 3, 3]);
    let saved_cross = hooks.get(&cross).unwrap();
    hooks.edit(cross_point.clone(), |probabilities| {
        probabilities.zeros_like()
    });
    let removed = model
        .forward_with_hooks(seq_input(), &options, &mut hooks)
        .unwrap()
        .decoder
        .logits;
    assert!(difference(removed, reference.clone()) > 1e-5);
    hooks.replace(cross_point, saved_cross);
    let restored = model
        .forward_with_hooks(seq_input(), &options, &mut hooks)
        .unwrap()
        .decoder
        .logits;
    assert_close(restored, reference);
}

#[path = "composition_tests.rs"]
/// Staged execution, explicit attention connectivity, and consumer-defined module tests.
mod composition;
