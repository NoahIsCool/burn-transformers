//! Standalone block construction, option wiring, connectivity, and encoder-direction contracts.

use burn::{
    module::Module,
    tensor::{Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::{
    Error,
    architecture::{
        AttentionOptions, AttentionSpec, FeedForwardSpec, Mamba2Spec, MixerSpec, MlaSpec,
        NormalizationSpec, OutputGateActivation, PositionEncodingSpec, RopeScaling, ScoreScale,
        SlidingWindowSpec,
    },
    attention::{AttentionOutputGate, ProjectionBias, QueryKeyNormScope},
    experiment::{AttentionPolicy, EncoderContext},
    models::custom::test_support::{
        DeterministicParameters, TestBackend, attention, layer as block, rms,
    },
};

/// Device used by every block fixture.
type Device = <TestBackend as BackendTypes>::Device;

/// Deterministic `[1, tokens, 8]` hidden states.
fn hidden(tokens: usize, device: &Device) -> Tensor<TestBackend, 3> {
    let values = (0..tokens * 8)
        .map(|index| ((index * 7 + 3) as f32 * 0.31).sin())
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [1, tokens, 8]), device)
}

/// Builds a deterministic standalone block of width eight.
fn built(spec: &DecoderLayerSpec, device: &Device) -> ModularLayer<TestBackend> {
    ModularLayer::<TestBackend>::init(spec, 8, None, device)
        .unwrap()
        .map(&mut DeterministicParameters::default())
}

/// Attention with every option, so tests can toggle one choice at a time.
fn options_mixer(attention: AttentionSpec, options: AttentionOptions) -> MixerSpec {
    MixerSpec::AttentionWithOptions { attention, options }
}

/// A two-head GQA topology with RoPE over the full head.
fn rope_attention() -> AttentionSpec {
    AttentionSpec {
        query_heads: 2,
        key_value_heads: 1,
        head_dim: 4,
        query_key_norm: true,
        output_gate: true,
        position: PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 4,
            scaling: RopeScaling::None,
        },
    }
}

/// Returns the grouped-query module of an attention block.
fn grouped(layer: &ModularLayer<TestBackend>) -> &ModularAttention<TestBackend> {
    match &layer.mixer {
        ModularMixer::Attention(attention) => attention,
        other => panic!("expected grouped-query attention, found {other:?}"),
    }
}

/// CUS-6: an encoder-addressed sliding block errors instead of seeing future keys.
///
/// The past-only window would otherwise combine with bidirectional encoder
/// attention into an asymmetric "future-unbounded" window. An explicit
/// bidirectional policy replaces the structural window and is accepted.
#[test]
fn encoder_sliding_blocks_require_an_explicit_policy() {
    let device = Default::default();
    let spec = block(MixerSpec::SlidingAttention(SlidingWindowSpec {
        attention: rope_attention(),
        window: 2,
    }));
    let layer = built(&spec, &device);
    let context = ExecutionContext::contiguous(1, 6, &device).unwrap();
    let mut hooks = Hooks::new();
    let probe = hooks.capture(Layer::encoder(0).residual_pre());
    let error = layer
        .run_with_hooks(hidden(6, &device), &context, Layer::encoder(0), &mut hooks)
        .unwrap_err();
    assert!(
        matches!(&error, Error::BackendCapability(reason) if reason.contains("past-only")),
        "{error:?}"
    );
    assert!(
        hooks.get(&probe).is_err(),
        "validation precedes every callback"
    );
    let explicit = context
        .clone()
        .with_attention(AttentionPolicy::bidirectional());
    layer
        .run(hidden(6, &device), &explicit, Layer::encoder(0))
        .unwrap();
    // The same block is ordinary causal sliding attention in a decoder stack.
    layer
        .run(hidden(6, &device), &context, Layer::decoder(0))
        .unwrap();
}

/// Chunked-local attention is symmetric, so encoder blocks accept it and see future keys in-chunk.
#[test]
fn chunked_local_encoder_blocks_see_only_their_chunk() {
    let device = Default::default();
    let spec = block(options_mixer(
        rope_attention(),
        AttentionOptions {
            connectivity: AttentionConnectivity::ChunkedLocal { chunk: 2 },
            ..AttentionOptions::default()
        },
    ));
    let layer = built(&spec, &device);
    let context = ExecutionContext::contiguous(1, 4, &device).unwrap();
    let base = hidden(4, &device);
    let clean = layer
        .run(base.clone(), &context, Layer::encoder(0))
        .unwrap();
    // Changing token 1 (chunk 0) changes token 0 (same chunk, future key) but not tokens 2, 3.
    let edited = base.clone().slice_assign(
        [0..1, 1..2, 0..8],
        Tensor::<TestBackend, 3>::ones([1, 1, 8], &device),
    );
    let changed = layer.run(edited, &context, Layer::encoder(0)).unwrap();
    let delta = (changed - clean)
        .abs()
        .sum_dim(2)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(
        delta[0] > 1e-6,
        "future key inside the chunk is visible: {delta:?}"
    );
    assert!(
        delta[2] < 1e-6 && delta[3] < 1e-6,
        "other chunk is unaffected: {delta:?}"
    );
}

/// Every attention option reaches the shared grouped-query module.
#[test]
fn attention_options_configure_the_shared_module() {
    let device = Default::default();
    let options = AttentionOptions {
        score_scale: ScoreScale::Fixed { value: 0.25 },
        logit_softcap: Some(30.0),
        projection_bias: ProjectionBias::query_key_value_only(),
        query_key_norm_scope: QueryKeyNormScope::FullWidth,
        output_gate_activation: OutputGateActivation::Silu,
        ..AttentionOptions::default()
    };
    let layer = built(&block(options_mixer(rope_attention(), options)), &device);
    let module = &grouped(&layer).attention;
    assert_eq!(module.scale, Some(0.25));
    assert_eq!(module.softcap, Some(30.0));
    assert_eq!(module.output_gate, AttentionOutputGate::Silu);
    assert_eq!(module.query_key_norm_scope, QueryKeyNormScope::FullWidth);
    assert!(module.q_proj.bias.is_some() && module.k_proj.bias.is_some());
    assert!(module.v_proj.bias.is_some() && module.o_proj.bias.is_none());
    assert_eq!(module.query_norm.as_ref().unwrap().weight.dims(), [8]);
    assert!(module.sinks.is_none());

    let sinks = AttentionOptions {
        attention_sinks: true,
        ..AttentionOptions::default()
    };
    let layer = built(&block(options_mixer(rope_attention(), sinks)), &device);
    assert_eq!(
        grouped(&layer).attention.sinks.as_ref().unwrap().dims(),
        [2]
    );
    let output = layer
        .run(
            hidden(3, &device),
            &ExecutionContext::contiguous(1, 3, &device).unwrap(),
            Layer::decoder(0),
        )
        .unwrap();
    assert_eq!(output.dims(), [1, 3, 8]);
}

/// CUS-7: the score scale is explicit, and the legacy `Auto` keeps old specs unchanged.
#[test]
fn score_scales_resolve_per_mixer_and_position() {
    let device = Default::default();
    let t5 = PositionEncodingSpec::T5RelativeBias {
        buckets: 8,
        max_distance: 16,
        bidirectional: false,
    };
    let with_t5 = AttentionSpec {
        position: t5.clone(),
        ..rope_attention()
    };
    let scale_of = |mixer: MixerSpec| grouped(&built(&block(mixer), &device)).attention.scale;
    // Compact T5 specs keep their historical unscaled logits.
    assert_eq!(scale_of(MixerSpec::Attention(with_t5.clone())), Some(1.0));
    // Explicit choices are independent of the position encoding.
    for (scale, expected) in [
        (ScoreScale::InverseSqrtHeadDim, None),
        (ScoreScale::Unscaled, Some(1.0)),
        (ScoreScale::Fixed { value: 0.5 }, Some(0.5)),
    ] {
        let options = AttentionOptions {
            score_scale: scale,
            ..AttentionOptions::default()
        };
        assert_eq!(scale_of(options_mixer(with_t5.clone(), options)), expected);
        assert_eq!(scale_of(options_mixer(rope_attention(), options)), expected);
    }
    assert_eq!(scale_of(MixerSpec::Attention(rope_attention())), None);

    let mla = |score_scale| MlaSpec {
        query_heads: 2,
        content_head_dim: 4,
        rotary_head_dim: 4,
        value_head_dim: 4,
        key_value_lora_rank: 4,
        query_lora_rank: None,
        position: t5.clone(),
        score_scale,
        connectivity: AttentionConnectivity::Global,
    };
    let latent_scale =
        |score_scale| match &built(&block(MixerSpec::Mla(mla(score_scale))), &device).mixer {
            ModularMixer::Mla(mla) => mla.attention.config.softmax_scale,
            other => panic!("expected MLA, found {other:?}"),
        };
    assert!((latent_scale(ScoreScale::Auto) - 8f64.sqrt().recip()).abs() < 1e-15);
    assert_eq!(latent_scale(ScoreScale::Unscaled), 1.0);
    assert_eq!(latent_scale(ScoreScale::Fixed { value: 0.2 }), 0.2);
}

/// Attention-only blocks compute `x + mixer(x)` and expose no MLP point or parameter.
#[test]
fn attention_only_blocks_skip_the_feed_forward_branch() {
    let device = Default::default();
    let mut spec = block(attention(PositionEncodingSpec::None));
    spec.feed_forward = FeedForwardSpec::None;
    let layer = built(&spec, &device);
    assert!(layer.feed_forward.is_none());
    let points = layer.activation_points(Layer::decoder(0));
    assert!(!points.contains(&Layer::decoder(0).mlp_input().info()));
    assert!(points.contains(&Layer::decoder(0).residual_post().info()));
    let mut with_mlp = block(attention(PositionEncodingSpec::None));
    with_mlp.feed_forward = FeedForwardSpec::Dense(crate::architecture::DenseFeedForwardSpec {
        intermediate_size: 4,
        activation: crate::architecture::Activation::Silu,
        bias: false,
    });
    assert!(built(&with_mlp, &device).num_params() > layer.num_params());

    let context = ExecutionContext::contiguous(1, 3, &device).unwrap();
    let input = hidden(3, &device);
    let mut hooks = Hooks::new();
    let mixed = hooks.capture(Layer::decoder(0).attention_output());
    let output = layer
        .run_with_hooks(input.clone(), &context, Layer::decoder(0), &mut hooks)
        .unwrap();
    output.into_data().assert_approx_eq::<f32>(
        &(input + hooks.get(&mixed).unwrap()).into_data(),
        Tolerance::absolute(1e-6),
    );
}

/// Standalone errors name the block's own fields, not a fictitious `layers[0]`.
#[test]
fn standalone_block_errors_name_block_fields() {
    let device: Device = Default::default();
    let mut invalid = block(attention(PositionEncodingSpec::None));
    invalid.mixer = MixerSpec::Attention(AttentionSpec {
        key_value_heads: 3,
        ..rope_attention()
    });
    let error = ModularLayer::<TestBackend>::init(&invalid, 8, None, &device).unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, .. } if field == "layer.mixer.key_value_heads"),
        "{error:?}"
    );
    let error = ModularLayer::<TestBackend>::init(
        &block(attention(PositionEncodingSpec::None)),
        0,
        None,
        &device,
    )
    .unwrap_err();
    assert!(matches!(&error, Error::Configuration { field, .. } if field == "layer.hidden_size"));
    let cross = CrossAttentionSpec::new(0, 4);
    let error = ModularLayer::<TestBackend>::init(
        &block(attention(PositionEncodingSpec::None)),
        8,
        Some(&cross),
        &device,
    )
    .unwrap_err();
    assert!(matches!(&error, Error::Configuration { field, .. } if field == "cross_attention"));
}

/// Well-formed but unexecutable mechanisms fail with capability errors before allocation.
#[test]
fn unexecutable_blocks_report_capabilities() {
    let device: Device = Default::default();
    let mut gated = block(attention(PositionEncodingSpec::None));
    gated.residual = ResidualTopology::Gated;
    let mamba = block(MixerSpec::Mamba2(Mamba2Spec {
        heads: 2,
        head_dim: 4,
        state_dim: 4,
        convolution_kernel: 4,
        group_size: 1,
    }));
    for spec in [gated, mamba] {
        let error = ModularLayer::<TestBackend>::init(&spec, 8, None, &device).unwrap_err();
        assert!(matches!(error, Error::BackendCapability(_)), "{error:?}");
    }
}

/// A standalone decoder block with cross-attention reads the context's encoder source.
#[test]
fn standalone_cross_attention_reads_the_context_source() {
    let device = Default::default();
    let spec = block(attention(PositionEncodingSpec::None));
    let cross = CrossAttentionSpec {
        options: AttentionOptions {
            projection_bias: ProjectionBias::uniform(true),
            ..AttentionOptions::default()
        },
        ..CrossAttentionSpec::new(2, 4)
    };
    let layer = ModularLayer::<TestBackend>::init(&spec, 8, Some(&cross), &device)
        .unwrap()
        .map(&mut DeterministicParameters::default());
    assert!(
        layer
            .cross_attention
            .as_ref()
            .unwrap()
            .attention
            .o_proj
            .bias
            .is_some()
    );
    let context = ExecutionContext::contiguous(1, 2, &device).unwrap();
    let missing = layer
        .run(hidden(2, &device), &context, Layer::decoder(0))
        .unwrap_err();
    assert!(missing.to_string().contains("context.encoder"));
    let source = EncoderContext {
        hidden: hidden(3, &device),
        positions: Tensor::<TestBackend, 1, Int>::arange(0..3, &device).reshape([1, 3]),
        token_mask: None,
    };
    let with_source = context.with_encoder(source).unwrap();
    let mut hooks = Hooks::new();
    let read = hooks.capture(Layer::decoder(0).cross_attention_output());
    layer
        .run_with_hooks(
            hidden(2, &device),
            &with_source,
            Layer::decoder(0),
            &mut hooks,
        )
        .unwrap();
    assert_eq!(hooks.get(&read).unwrap().dims(), [1, 2, 8]);
}

/// Zero-centered RMS norms reach both the block norm and the Q/K norm.
#[test]
fn zero_centered_norms_reach_query_key_normalization() {
    let device = Default::default();
    let mut spec = block(MixerSpec::Attention(rope_attention()));
    spec.mixer_norm = NormalizationSpec::Rms {
        epsilon: 1e-5,
        zero_centered: true,
    };
    spec.feed_forward_norm = rms();
    let layer = ModularLayer::<TestBackend>::init(&spec, 8, None, &device).unwrap();
    let norm = grouped(&layer).attention.query_norm.as_ref().unwrap();
    assert!(norm.zero_centered);
    assert_eq!(norm.epsilon, 1e-5);
}

/// A two-pair differential-attention topology with full-head RoPE.
fn differential() -> crate::architecture::DifferentialAttentionSpec {
    crate::architecture::DifferentialAttentionSpec {
        query_heads: 4,
        key_value_heads: 2,
        head_dim: 2,
        lambda_init: crate::attention::depth_lambda_init(1),
        position: PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 2,
            scaling: RopeScaling::None,
        },
        connectivity: AttentionConnectivity::Global,
        bias: false,
    }
}

/// The differential mixer equals the shared kernel applied to the rotated normalized input.
#[test]
fn differential_attention_blocks_run_the_shared_kernel() {
    use crate::position::{RopeConfig, apply_rotary, rotary_angles};

    let device = Default::default();
    let layer = built(
        &block(MixerSpec::DifferentialAttention(differential())),
        &device,
    );
    let ModularMixer::Differential(mixer) = &layer.mixer else {
        panic!("expected differential attention");
    };
    assert_eq!(
        mixer.attention.lambda_init,
        crate::attention::depth_lambda_init(1)
    );
    assert_eq!(mixer.attention.norm_epsilon, 1e-6);
    let context = ExecutionContext::contiguous(1, 5, &device).unwrap();
    let mut hooks = Hooks::new();
    let normalized = hooks.capture(Layer::decoder(0).attention_input());
    let mixed = hooks.capture(Layer::decoder(0).attention_output());
    layer
        .run_with_hooks(hidden(5, &device), &context, Layer::decoder(0), &mut hooks)
        .unwrap();
    let mut projection = mixer
        .attention
        .project_self(hooks.get(&normalized).unwrap())
        .unwrap();
    let rope = RopeConfig::new(2, 2, 10_000.0, crate::position::RopeScaling::None).unwrap();
    let angles = rotary_angles(rope, context.positions.clone(), 5).unwrap();
    (projection.query, projection.key) =
        apply_rotary(projection.query, projection.key, &angles).unwrap();
    let expected = mixer.attention.attend(projection, None, true).unwrap();
    hooks
        .get(&mixed)
        .unwrap()
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));
    assert!(
        !layer
            .activation_points(Layer::decoder(0))
            .contains(&Layer::decoder(0).self_attention().scores().info())
    );
}

/// Differential attention rejects additive biases and encoder sliding windows, but accepts chunks.
#[test]
fn differential_attention_reports_unsupported_connectivity() {
    let device = Default::default();
    let layer = built(
        &block(MixerSpec::DifferentialAttention(differential())),
        &device,
    );
    let context = ExecutionContext::contiguous(1, 4, &device).unwrap();
    let biased = context
        .clone()
        .with_attention(AttentionPolicy::causal().with_bias(Tensor::zeros([1, 1, 4, 4], &device)));
    let error = layer
        .run(hidden(4, &device), &biased, Layer::decoder(0))
        .unwrap_err();
    assert!(matches!(error, Error::BackendCapability(_)), "{error:?}");
    let sliding = built(
        &block(MixerSpec::DifferentialAttention(
            crate::architecture::DifferentialAttentionSpec {
                connectivity: AttentionConnectivity::SlidingWindow { window: 2 },
                ..differential()
            },
        )),
        &device,
    );
    assert!(
        sliding
            .run(hidden(4, &device), &context, Layer::encoder(0))
            .is_err()
    );
    sliding
        .run(hidden(4, &device), &context, Layer::decoder(0))
        .unwrap();
    let chunked = built(
        &block(MixerSpec::DifferentialAttention(
            crate::architecture::DifferentialAttentionSpec {
                connectivity: AttentionConnectivity::ChunkedLocal { chunk: 2 },
                ..differential()
            },
        )),
        &device,
    );
    chunked
        .run(hidden(4, &device), &context, Layer::encoder(0))
        .unwrap();
}
