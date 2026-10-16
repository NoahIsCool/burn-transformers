//! Hook-failure propagation across every mixer and feed-forward kind, staged-state
//! validation, and the architecture-trait entry points of [`ModularCausalLm`].

use burn::{
    module::Module,
    nn::LinearConfig,
    tensor::{Int, Tensor, backend::BackendTypes},
};

use super::*;
use crate::{
    architecture::{
        Activation, ClampedSwiGluSpec, DenseFeedForwardSpec, DifferentialAttentionSpec,
        FeedForwardSpec, Mamba2Spec, MixerSpec, MixtureOfExpertsSpec, MlaSpec,
        PositionEncodingSpec, RopeScaling, ScoreScale,
    },
    attention::AttentionConnectivity,
    experiment::EncoderContext,
    models::{
        custom::test_support::{
            DeterministicParameters, TestBackend, attention, causal_spec, decoder, layer, mask,
        },
        shared::test_support::{
            assert_edit_failures_propagate, assert_staged_edit_failures_propagate,
        },
    },
    moe::RoutingPolicy,
};

/// Device of every fixture.
type Device = <TestBackend as BackendTypes>::Device;

/// A heterogeneous decoder exercising every mixer and every feed-forward kind.
///
/// Layers: RoPE attention with a dense MLP, latent attention (with a low-rank
/// query bottleneck, so its `query_latent` point exists) with a clamped
/// SwiGLU, differential attention with an ungated MLP, DeltaNet with routed
/// experts, and Mamba-2 with the dense MLP. Attention runs eagerly, so score
/// and pattern points exist.
fn every_kind(device: &Device) -> ModularCausalLm<TestBackend> {
    let rope = PositionEncodingSpec::Rope {
        theta: 10_000.0,
        rotary_dim: 4,
        scaling: RopeScaling::None,
    };
    let mut latent = layer(MixerSpec::Mla(MlaSpec {
        query_heads: 2,
        content_head_dim: 4,
        rotary_head_dim: 4,
        value_head_dim: 4,
        key_value_lora_rank: 4,
        query_lora_rank: Some(4),
        position: rope.clone(),
        score_scale: ScoreScale::Auto,
        connectivity: AttentionConnectivity::Global,
    }));
    latent.feed_forward = FeedForwardSpec::ClampedSwiGlu(ClampedSwiGluSpec {
        intermediate_size: 6,
        bias: true,
        limit: 0.5,
        alpha: 1.702,
        up_offset: 1.0,
    });
    let mut differential = layer(MixerSpec::DifferentialAttention(
        DifferentialAttentionSpec {
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
        },
    ));
    differential.feed_forward = FeedForwardSpec::Ungated(DenseFeedForwardSpec {
        intermediate_size: 6,
        activation: Activation::Gelu,
        bias: true,
    });
    let mut delta = layer(MixerSpec::DeltaNet {
        key_heads: 1,
        value_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        convolution_kernel: 2,
    });
    delta.feed_forward = FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
        routed_experts: 3,
        experts_per_token: 2,
        shared_experts: 1,
        expert_intermediate_size: 4,
        activation: Activation::Silu,
        renormalize_selected: true,
        routed_scale: 1.0,
        routing: RoutingPolicy::default(),
        shared_expert_gate: false,
        expert_bias: false,
    });
    let mamba = layer(MixerSpec::Mamba2(Mamba2Spec {
        heads: 2,
        head_dim: 4,
        state_dim: 8,
        convolution_kernel: 2,
        group_size: 1,
    }));
    ModularCausalLm::init(
        decoder(vec![
            layer(attention(rope)),
            latent,
            differential,
            delta,
            mamba,
        ]),
        device,
    )
    .unwrap()
    .with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax)
    .map(&mut DeterministicParameters::default())
}

/// A right-padded two-row batch.
fn input(device: &Device) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3], [4, 5, 0]], device),
        token_mask: Some(mask([[true, true, true], [true, true, false]], device)),
        position_ids: None,
    }
}

/// A failing edit at any point of any mixer or feed-forward kind aborts the whole pass.
#[test]
fn every_mixer_and_feed_forward_point_propagates_edit_failures() {
    let device = Default::default();
    let model = every_kind(&device);
    let points = model.activation_points();
    assert_edit_failures_propagate(&points, |hooks| {
        model.forward_with_hooks(input(&device), &ForwardOptions::default(), hooks)
    });
    assert_staged_edit_failures_propagate(&model, &input(&device), &points);
}

/// The trait entry points delegate to the inherent model and validate the decode capacity.
#[test]
fn architecture_trait_entry_points_match_the_inherent_model() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(causal_spec(), &device)
        .unwrap()
        .map(&mut DeterministicParameters::default());
    let inherent = model
        .forward(input(&device), &ForwardOptions::default())
        .unwrap();
    let generic =
        CausalLanguageModel::forward(&model, input(&device), &ForwardOptions::default()).unwrap();
    assert_eq!(inherent.logits.into_data(), generic.logits.into_data());
    ArchitectureConfig::validate(model.spec()).unwrap();
    let error = model
        .new_decode_state(CacheConfig::new(1, 33).unwrap(), 1, &device)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, reason }
            if field == "custom.causal.decode_state.capacity"
                && reason == "33 exceeds max_position_embeddings 32"),
        "{error:?}"
    );
    model
        .new_decode_state(CacheConfig::new(1, 32).unwrap(), 1, &device)
        .expect("the full position range is a valid capacity");
}

/// Position tables beyond Burn's signed 32-bit index range are rejected at construction.
#[test]
fn position_ranges_beyond_i32_are_not_executable() {
    let mut spec = causal_spec();
    spec.max_position_embeddings = i32::MAX as usize + 1;
    for error in [
        ArchitectureConfig::validate(&spec).unwrap_err(),
        ModularCausalLm::<TestBackend>::init(spec, &Default::default()).unwrap_err(),
    ] {
        assert!(
            matches!(&error, Error::Configuration { field, reason }
                if field == "custom.causal.max_position_embeddings"
                    && reason.contains("portable signed position-index range")),
            "{error:?}"
        );
    }
}

/// Staged states are checked for width, cross-attention, and position range before any block.
#[test]
fn staged_states_reject_foreign_widths_sources_and_positions() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(causal_spec(), &device)
        .unwrap()
        .map(&mut DeterministicParameters::default());
    let state = model.embed(input(&device)).unwrap();

    let mut narrow = state.clone();
    narrow.hidden = Tensor::zeros([2, 3, 7], &device);
    match model.run_layers(narrow, 0..2).unwrap_err() {
        Error::Shape {
            tensor,
            expected,
            actual,
        } => assert_eq!(
            (tensor.as_str(), expected.as_str(), actual.as_str()),
            (
                "execution.hidden",
                "nonempty [batch, token, 8]",
                "[2, 3, 7]"
            )
        ),
        other => panic!("expected a shape error, found {other:?}"),
    }

    let mut sourced = state.clone();
    sourced.context = sourced
        .context
        .with_encoder(EncoderContext {
            hidden: Tensor::zeros([2, 2, 8], &device),
            positions: Tensor::<TestBackend, 1, Int>::arange(0..2, &device)
                .reshape([1, 2])
                .expand([2, 2]),
            token_mask: None,
        })
        .unwrap();
    let error = model.run_layers(sourced, 0..2).unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, .. } if field == "custom.execution.cross_attention"),
        "{error:?}"
    );
    let crossed = state
        .clone()
        .context
        .with_cross_attention(AttentionPolicy::bidirectional());
    let error = model
        .normalize(HiddenState {
            context: crossed,
            ..state.clone()
        })
        .unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, .. } if field == "custom.execution.cross_attention"),
        "{error:?}"
    );

    let mut moved = state;
    moved.context.positions =
        Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 2], [0, 1, 40]], &device);
    moved.context.evaluated_context = 41;
    let error = model.run_layers(moved, 0..2).unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, .. } if tensor == "custom.run_layers.positions.position_ids"),
        "{error:?}"
    );
}

/// A replaced vocabulary head of the wrong width fails at projection, after the blocks.
#[test]
fn replaced_vocabulary_head_is_checked_at_projection() {
    let device = Default::default();
    let mut model = ModularCausalLm::<TestBackend>::init(causal_spec(), &device).unwrap();
    model.lm_head = Some(LinearConfig::new(7, 32).init(&device));
    match model
        .forward(input(&device), &ForwardOptions::default())
        .unwrap_err()
    {
        Error::Shape {
            tensor,
            expected,
            actual,
        } => assert_eq!(
            (tensor.as_str(), expected.as_str(), actual.as_str()),
            ("linear.weight", "[8, nonzero_output]", "[7, 32]")
        ),
        other => panic!("expected a projection shape error, found {other:?}"),
    }
}

/// Staged embedding validates token IDs, and the complete pass validates logits positions.
#[test]
fn staged_tokens_and_logits_positions_are_validated() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(causal_spec(), &device).unwrap();
    let mut bad = input(&device);
    bad.token_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 32], [4, 5, 0]], &device);
    let error = model.embed(bad).unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, actual, .. }
            if tensor == "custom.embed.token_ids" && actual == "observed range 0..=32"),
        "{error:?}"
    );
    let error = model
        .forward(
            input(&device),
            &ForwardOptions {
                logits: crate::LogitsSelection::Positions(vec![3]),
                ..ForwardOptions::default()
            },
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, .. } if tensor == "forward_options.logits.positions"),
        "{error:?}"
    );
}
