//! Tiny executable, validation, serialization, state-lifecycle, and autodiff tests.

use burn::{
    backend::Flex,
    module::Param,
    tensor::{Bool, Int, Tensor, Tolerance, backend::BackendTypes},
};

use crate::{
    CausalLanguageModel, CausalLmInput, DecodeInput, DecodeState, ForwardOptions, LogitsSelection,
    Seq2SeqInput, Seq2SeqLanguageModel, TokenMask,
    architecture::{
        Activation, AttentionSpec, DecoderLayerSpec, DecoderSpec, DenseFeedForwardSpec,
        FeedForwardSpec, Mamba2Spec, MixerSpec, MixtureOfExpertsSpec, MlaSpec, NormPlacement,
        NormalizationSpec, PositionEncodingSpec, ResidualTopology, RopeScaling, Seq2SeqSpec,
        SlidingWindowSpec,
    },
    cache::CacheConfig,
    position::T5RelativePositionBiasConfig,
};

use super::{ModularCausalLm, ModularSeq2SeqLm, common::relative_bias_from_positions};

type TestBackend = Flex<f32, i32>;

/// Standard scale-only normalization for compact fixtures.
fn rms() -> NormalizationSpec {
    NormalizationSpec::Rms {
        epsilon: 1e-6,
        zero_centered: false,
    }
}

/// Full two-head attention without an explicit position transform.
fn attention(position: PositionEncodingSpec) -> MixerSpec {
    MixerSpec::Attention(AttentionSpec {
        query_heads: 2,
        key_value_heads: 1,
        head_dim: 4,
        query_key_norm: true,
        output_gate: false,
        position,
    })
}

/// Bias-free gated feed-forward branch used by most fixtures.
fn dense_ffn() -> FeedForwardSpec {
    FeedForwardSpec::Dense(DenseFeedForwardSpec {
        intermediate_size: 12,
        activation: Activation::Silu,
        bias: false,
    })
}

/// One pre-normalized additive residual layer.
fn layer(mixer: MixerSpec) -> DecoderLayerSpec {
    DecoderLayerSpec {
        mixer,
        feed_forward: dense_ffn(),
        mixer_norm: rms(),
        feed_forward_norm: rms(),
        norm_placement: NormPlacement::Pre,
        residual: ResidualTopology::Add,
    }
}

/// Small decoder whose layer schedule is supplied by the test.
fn decoder(layers: Vec<DecoderLayerSpec>) -> DecoderSpec {
    DecoderSpec {
        vocabulary_size: 32,
        hidden_size: 8,
        layers,
        final_norm: rms(),
        tie_word_embeddings: false,
        max_position_embeddings: 32,
    }
}

/// Small causal decoder with RoPE attention followed by DeltaNet.
fn causal_spec() -> DecoderSpec {
    decoder(vec![
        layer(attention(PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 4,
            scaling: RopeScaling::None,
        })),
        layer(MixerSpec::DeltaNet {
            key_heads: 1,
            value_heads: 2,
            key_head_dim: 4,
            value_head_dim: 4,
            convolution_kernel: 2,
        }),
    ])
}

/// One-layer encoder-decoder fixture with plain absolute-free attention.
fn seq2seq_spec() -> Seq2SeqSpec {
    Seq2SeqSpec {
        encoder: decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        decoder: decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        cross_attention_heads: 2,
        cross_attention_head_dim: 4,
    }
}

/// Constructs a token-validity mask from a nested boolean array.
fn mask<const B: usize, const S: usize>(
    values: [[bool; S]; B],
    device: &<TestBackend as BackendTypes>::Device,
) -> TokenMask<TestBackend> {
    TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        values.into(),
        device,
    ))
}

/// Heterogeneous attention and recurrent layers execute as one decoder graph.
#[test]
fn causal_heterogeneous_forward_returns_requested_diagnostics() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(causal_spec(), &device).unwrap();
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3], [4, 5, 0]], &device),
                token_mask: Some(mask([[true, true, true], [true, true, false]], &device)),
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..Default::default()
            },
        )
        .expect("heterogeneous forward");
    assert_eq!(output.logits.dims(), [2, 3, 32]);
    assert_eq!(output.hidden_states.len(), 4);
    assert_eq!(output.final_hidden_state.unwrap().dims(), [2, 3, 8]);
    assert!(output.attentions.is_empty());
}

/// Sparse custom layers expose raw router scores and a differentiable balancing loss.
#[test]
fn causal_moe_exposes_opt_in_router_diagnostics() {
    let device = Default::default();
    let mut moe_layer = layer(attention(PositionEncodingSpec::None));
    moe_layer.feed_forward = FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
        routed_experts: 4,
        experts_per_token: 2,
        shared_experts: 1,
        expert_intermediate_size: 6,
        activation: Activation::ReluSquared,
        renormalize_selected: true,
        routed_scale: 0.75,
    });
    let model = ModularCausalLm::<TestBackend>::init(decoder(vec![moe_layer]), &device).unwrap();
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                router_logits: true,
                auxiliary_losses: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(output.routers.len(), 1);
    assert_eq!(output.routers[0].logits.as_ref().unwrap().dims(), [1, 2, 4]);
    assert_eq!(
        output.routers[0]
            .load_balancing_loss
            .as_ref()
            .unwrap()
            .dims(),
        [1]
    );
}

/// Sliding attention, a sigmoid output gate, LayerNorm, and sandwich placement compose.
#[test]
fn sliding_sandwich_layernorm_graph_is_executable() {
    let device = Default::default();
    let attention = AttentionSpec {
        query_heads: 2,
        key_value_heads: 2,
        head_dim: 4,
        query_key_norm: false,
        output_gate: true,
        position: PositionEncodingSpec::None,
    };
    let mut block = layer(MixerSpec::SlidingAttention(SlidingWindowSpec {
        attention: attention.clone(),
        window: 2,
    }));
    block.norm_placement = NormPlacement::Sandwich;
    block.mixer_norm = NormalizationSpec::Layer {
        epsilon: 1e-5,
        bias: true,
    };
    block.feed_forward_norm = NormalizationSpec::Layer {
        epsilon: 1e-5,
        bias: false,
    };
    let model = ModularCausalLm::<TestBackend>::init(decoder(vec![block]), &device).unwrap();
    let data = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits
        .into_data();
    assert!(
        data.as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    );
}

/// T5 relative buckets select unscaled learned-bias attention in a custom decoder.
#[test]
fn t5_relative_bias_is_executable_outside_t5_family_wiring() {
    let device = Default::default();
    let spec = decoder(vec![layer(attention(
        PositionEncodingSpec::T5RelativeBias {
            buckets: 8,
            max_distance: 32,
            bidirectional: false,
        },
    ))]);
    let model = ModularCausalLm::<TestBackend>::init(spec, &device).unwrap();
    assert_eq!(
        model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
            .dims(),
        [1, 1, 32]
    );
}

/// Explicit, gapped T5 positions select buckets from actual displacement rather than indices.
#[test]
fn t5_relative_bias_honors_explicit_position_gaps() {
    let device = Default::default();
    let mut bias = T5RelativePositionBiasConfig::new(1, 8, 32, false)
        .init::<TestBackend>(&device)
        .unwrap();
    bias.weight = Param::from_tensor(
        Tensor::<TestBackend, 1, Int>::arange(0..8, &device)
            .float()
            .reshape([8, 1]),
    );

    let actual =
        relative_bias_from_positions(&bias, Tensor::from_ints([[5, 7, 10]], &device)).unwrap();
    actual.into_data().assert_approx_eq::<f32>(
        &Tensor::<TestBackend, 4>::from_floats(
            [[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 3.0, 0.0]]]],
            &device,
        )
        .into_data(),
        Tolerance::absolute(1e-6),
    );
}

/// YaRN rotary scaling and conventional post-normalization remain independently selectable.
#[test]
fn yarn_and_post_norm_graph_is_executable() {
    let device = Default::default();
    let mut block = layer(attention(PositionEncodingSpec::Rope {
        theta: 10_000.0,
        rotary_dim: 4,
        scaling: RopeScaling::Yarn {
            factor: 2.0,
            original_max_positions: 16,
            attention_factor: Some(1.1),
            beta_slow: 1.0,
            beta_fast: 32.0,
        },
    }));
    block.norm_placement = NormPlacement::Post;
    let model = ModularCausalLm::<TestBackend>::init(decoder(vec![block]), &device).unwrap();
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[16, 17]], &device)),
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(output.logits.dims(), [1, 1, 32]);
}

/// Tied output projection owns no duplicate linear parameter matrix.
#[test]
fn tied_causal_head_reuses_embedding_weight() {
    let device = Default::default();
    let mut spec = decoder(vec![layer(attention(PositionEncodingSpec::None))]);
    spec.tie_word_embeddings = true;
    let model = ModularCausalLm::<TestBackend>::init(spec, &device).unwrap();
    assert!(model.lm_head.is_none());
    assert_eq!(
        model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
            .dims(),
        [1, 1, 32]
    );
}

/// Serialized heterogeneous topology is lossless across JSON round trips.
#[test]
fn modular_spec_round_trips_through_json() {
    let spec = causal_spec();
    let json = serde_json::to_string_pretty(&spec).unwrap();
    let decoded: DecoderSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, spec);
}

/// MLA, Mamba2, and underspecified learned residual gates fail explicitly before allocation.
#[test]
fn unavailable_or_ambiguous_components_are_rejected() {
    let device = Default::default();
    let mla = MixerSpec::Mla(MlaSpec {
        query_heads: 2,
        content_head_dim: 2,
        rotary_head_dim: 2,
        value_head_dim: 4,
        key_value_lora_rank: 4,
        query_lora_rank: None,
        position: PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 4,
            scaling: RopeScaling::None,
        },
    });
    let error = ModularCausalLm::<TestBackend>::init(decoder(vec![layer(mla)]), &device)
        .expect_err("MLA kernel is not executable yet");
    assert!(error.to_string().contains("requests MLA"));

    let mamba = MixerSpec::Mamba2(Mamba2Spec {
        heads: 2,
        head_dim: 4,
        state_dim: 8,
        convolution_kernel: 4,
        group_size: 1,
    });
    let error = ModularCausalLm::<TestBackend>::init(decoder(vec![layer(mamba)]), &device)
        .expect_err("Mamba2 kernel is not executable yet");
    assert!(error.to_string().contains("requests Mamba2"));

    let mut gated = layer(attention(PositionEncodingSpec::None));
    gated.residual = ResidualTopology::Gated;
    let error = ModularCausalLm::<TestBackend>::init(decoder(vec![gated]), &device)
        .expect_err("gate semantics are underspecified");
    assert!(error.to_string().contains("does not define its activation"));
}

/// Stateless and prefill-plus-decode paths agree exactly at the appended token.
#[test]
fn causal_reference_decode_matches_stateless_forward() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(causal_spec(), &device).unwrap();
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
    assert_eq!(state.allocated_integer_elements(), 16);
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
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
    assert_eq!(state.lengths(), &[4]);
}

/// Explicit absolute positions and selective logits retain their meaning in cached execution.
#[test]
fn cached_explicit_positions_and_selection_match_stateless() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 4,
            scaling: RopeScaling::None,
        }))]),
        &device,
    )
    .unwrap();
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[7, 9, 12, 13]], &device)),
            },
            &ForwardOptions {
                logits: LogitsSelection::Positions(vec![2, 3]),
                ..Default::default()
            },
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[7, 9]], &device)),
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cached = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3, 4]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[12, 13]], &device)),
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap()
        .logits;
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
}

/// Ragged cached steps compact valid tokens, zero inactive outputs, and preserve row lengths.
#[test]
fn causal_decode_compacts_ragged_steps() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
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
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[5, 6], [0, 0]], &device),
                token_mask: Some(mask([[true, true], [false, false]], &device)),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(state.lengths(), &[4, 2]);
    assert_eq!(output.logits.dims(), [2, 2, 32]);
    let inactive = output.logits.slice([1..2, 0..2, 0..32]).into_data();
    assert!(
        inactive
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&x| x == 0.0)
    );
}

/// Reset, duplicate reorder, and truncation update external history lifecycle consistently.
#[test]
fn causal_state_lifecycle_is_checked() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 6).unwrap(), 2, &device)
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
    state.truncate(&[1, 0]).unwrap();
    assert_eq!(state.lengths(), &[1, 0]);
    state.reset().unwrap();
    assert_eq!(state.lengths(), &[0, 0]);
    let error = state
        .truncate(&[1, 0])
        .expect_err("truncation cannot extend state");
    assert!(error.to_string().contains("cannot truncate"));
}

/// A failed over-capacity append leaves committed lengths unchanged.
#[test]
fn decode_capacity_failure_is_atomic() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 2).unwrap(), 1, &device)
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
    let error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("capacity must be checked before mutation");
    assert!(error.to_string().contains("exceeds capacity"));
    assert_eq!(state.lengths(), &[2]);
}

/// State construction rejects disagreement between cache sequence count and requested batch.
#[test]
fn decode_state_rejects_batch_configuration_mismatch() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let error = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 1, &device)
        .expect_err("batch and cache geometry must agree");
    assert!(error.to_string().contains("expected 2 sequences"));
}

/// State topology identity prevents passing a valid state to a different custom graph.
#[test]
fn decode_rejects_state_from_another_topology() {
    let device = Default::default();
    let first = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let second = ModularCausalLm::<TestBackend>::init(causal_spec(), &device).unwrap();
    let mut state = first
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let error = second
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("topologies must not share state");
    assert!(error.to_string().contains("different model topology"));
}

/// Public input validation reports token, mask, position, and diagnostic errors without panics.
#[test]
fn forward_validation_is_contextual() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let error = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[32]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect_err("out-of-range token");
    assert!(error.to_string().contains("values in 0..32"));

    let error = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: Some(mask([[false, false]], &device)),
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect_err("fully masked rows are invalid");
    assert!(error.to_string().contains("fully masked row"));

    let error = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[32]], &device)),
            },
            &ForwardOptions::default(),
        )
        .expect_err("out-of-range position");
    assert!(error.to_string().contains("position_ids"));

    let error = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                attentions: true,
                ..Default::default()
            },
        )
        .expect_err("attention capture is unavailable");
    assert!(error.to_string().contains("probability capture"));
}

/// Bidirectional encoding, causal decoding, and cross attention produce the public shapes.
#[test]
fn seq2seq_forward_executes_both_stacks() {
    let device = Default::default();
    let model = ModularSeq2SeqLm::<TestBackend>::init(seq2seq_spec(), &device).unwrap();
    let output = model
        .forward(
            Seq2SeqInput {
                encoder: CausalLmInput {
                    token_ids: Tensor::from_ints([[4, 5, 6]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                decoder: CausalLmInput {
                    token_ids: Tensor::from_ints([[0, 1]], &device),
                    token_mask: None,
                    position_ids: None,
                },
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(output.encoder.hidden_state.dims(), [1, 3, 8]);
    assert_eq!(output.encoder.hidden_states.len(), 3);
    assert_eq!(output.decoder.logits.dims(), [1, 2, 32]);
    assert_eq!(output.decoder.hidden_states.len(), 3);
    assert!(output.cross_attentions.is_empty());
}

/// Seq2seq prefill and reference decode agree with a complete target forward.
#[test]
fn seq2seq_cached_decoder_matches_stateless_forward() {
    let device = Default::default();
    let model = ModularSeq2SeqLm::<TestBackend>::init(seq2seq_spec(), &device).unwrap();
    let source = CausalLmInput {
        token_ids: Tensor::from_ints([[4, 5, 6]], &device),
        token_mask: None,
        position_ids: None,
    };
    let encoder = model
        .encode(source.clone(), &ForwardOptions::default())
        .unwrap();
    let full = model
        .forward(
            Seq2SeqInput {
                encoder: source,
                decoder: CausalLmInput {
                    token_ids: Tensor::from_ints([[0, 1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .decoder
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 1]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cached = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: Tensor::from_ints([[2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .decoder
        .logits;
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
}

/// Causal recurrent and trailing-window mixers cannot be placed in a bidirectional encoder.
#[test]
fn seq2seq_rejects_directionally_invalid_encoder_mixers() {
    let device = Default::default();
    let mut spec = seq2seq_spec();
    spec.encoder.layers[0].mixer = MixerSpec::DeltaNet {
        key_heads: 1,
        value_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        convolution_kernel: 2,
    };
    let error = ModularSeq2SeqLm::<TestBackend>::init(spec, &device)
        .expect_err("causal recurrence is invalid in encoder");
    assert!(error.to_string().contains("bidirectional encoder"));

    let mut spec = seq2seq_spec();
    let attention = match attention(PositionEncodingSpec::None) {
        MixerSpec::Attention(attention) => attention,
        _ => unreachable!(),
    };
    spec.encoder.layers[0].mixer = MixerSpec::SlidingAttention(SlidingWindowSpec {
        attention,
        window: 2,
    });
    let error = ModularSeq2SeqLm::<TestBackend>::init(spec, &device)
        .expect_err("trailing window is invalid in encoder");
    assert!(error.to_string().contains("symmetric bidirectional"));
}

/// Encoder MoE math is allowed, but unavailable encoder router output is rejected explicitly.
#[test]
fn seq2seq_encoder_router_diagnostics_are_not_silently_dropped() {
    let device = Default::default();
    let mut spec = seq2seq_spec();
    spec.encoder.layers[0].feed_forward = FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
        routed_experts: 2,
        experts_per_token: 1,
        shared_experts: 0,
        expert_intermediate_size: 4,
        activation: Activation::Silu,
        renormalize_selected: true,
        routed_scale: 1.0,
    });
    let model = ModularSeq2SeqLm::<TestBackend>::init(spec, &device).unwrap();
    let error = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                router_logits: true,
                ..Default::default()
            },
        )
        .expect_err("EncoderOutput cannot carry router diagnostics");
    assert!(error.to_string().contains("no router-diagnostic field"));
}

/// Causal stateless execution retains a gradient path to embeddings and the LM head.
#[cfg(feature = "autodiff")]
#[test]
fn causal_stateless_graph_is_autodiff_safe() {
    use burn::backend::Autodiff;

    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let model = ModularCausalLm::<AdBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let loss = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits
        .sum();
    let gradients = loss.backward();
    assert!(model.embed_tokens.weight.grad(&gradients).is_some());
}

/// Encoder-decoder stateless execution retains gradients through both embedding tables.
#[cfg(feature = "autodiff")]
#[test]
fn seq2seq_stateless_graph_is_autodiff_safe() {
    use burn::backend::Autodiff;

    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let model = ModularSeq2SeqLm::<AdBackend>::init(seq2seq_spec(), &device).unwrap();
    let loss = model
        .forward(
            Seq2SeqInput {
                encoder: CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[4, 5]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                decoder: CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[0, 1]], &device),
                    token_mask: None,
                    position_ids: None,
                },
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .decoder
        .logits
        .sum();
    let gradients = loss.backward();
    assert!(model.encoder_embed_tokens.weight.grad(&gradients).is_some());
    assert!(model.decoder_embed_tokens.weight.grad(&gradients).is_some());
}
