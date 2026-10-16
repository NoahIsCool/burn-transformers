//! Deterministic regression seeds for the modular-specification fuzz harness.

use super::*;

/// Pre-norm RMS normalization.
fn rms() -> NormalizationSpec {
    NormalizationSpec::Rms {
        epsilon: 1e-6,
        zero_centered: false,
    }
}

/// One pre-norm block around `mixer` and `feed_forward`.
fn block(mixer: MixerSpec, feed_forward: FeedForwardSpec) -> DecoderLayerSpec {
    DecoderLayerSpec {
        mixer,
        feed_forward,
        mixer_norm: rms(),
        feed_forward_norm: rms(),
        norm_placement: NormPlacement::Pre,
        residual: ResidualTopology::Add,
    }
}

/// A one-to-three-layer causal stack of width 8 over 16 tokens.
fn stack_of(layers: Vec<DecoderLayerSpec>) -> DecoderSpec {
    DecoderSpec {
        vocabulary_size: 16,
        hidden_size: 8,
        layers,
        final_norm: rms(),
        tie_word_embeddings: false,
        max_position_embeddings: 16,
        embedding_scale: None,
        final_logit_softcap: None,
    }
}

/// Grouped-query attention with RoPE and every option.
fn rope_attention(options: AttentionOptions) -> MixerSpec {
    MixerSpec::AttentionWithOptions {
        attention: AttentionSpec {
            query_heads: 4,
            key_value_heads: 2,
            head_dim: 4,
            query_key_norm: true,
            output_gate: true,
            position: PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: 4,
                scaling: RopeScaling::Yarn {
                    factor: 4.0,
                    original_max_positions: 4,
                    attention_factor: None,
                    beta_slow: 1.0,
                    beta_fast: 32.0,
                    truncate: false,
                },
            },
        },
        options,
    }
}

/// Fixed tokens for the execution checks.
const TOKENS: [i32; 16] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3];

/// Every executable mixer family, alone and stacked, matches its cached decode.
#[test]
fn executable_mixers_construct_run_and_match_their_cache() {
    let dense = || {
        FeedForwardSpec::Dense(DenseFeedForwardSpec {
            intermediate_size: 8,
            activation: Activation::Silu,
            bias: false,
        })
    };
    let options = AttentionOptions {
        connectivity: AttentionConnectivity::SlidingWindow { window: 2 },
        attention_sinks: true,
        query_key_norm_scope: QueryKeyNormScope::FullWidth,
        output_gate_activation: OutputGateActivation::Silu,
        ..AttentionOptions::default()
    };
    let mixers = [
        rope_attention(options),
        rope_attention(AttentionOptions {
            connectivity: AttentionConnectivity::ChunkedLocal { chunk: 2 },
            logit_softcap: Some(30.0),
            ..AttentionOptions::default()
        }),
        MixerSpec::Mla(MlaSpec {
            query_heads: 2,
            content_head_dim: 2,
            rotary_head_dim: 2,
            value_head_dim: 2,
            key_value_lora_rank: 4,
            query_lora_rank: Some(4),
            position: PositionEncodingSpec::None,
            score_scale: ScoreScale::Auto,
            connectivity: AttentionConnectivity::Global,
        }),
        MixerSpec::DifferentialAttention(DifferentialAttentionSpec {
            query_heads: 2,
            key_value_heads: 1,
            head_dim: 4,
            lambda_init: 0.2,
            position: PositionEncodingSpec::None,
            connectivity: AttentionConnectivity::Global,
            bias: true,
        }),
        MixerSpec::DeltaNet {
            key_heads: 1,
            value_heads: 2,
            key_head_dim: 4,
            value_head_dim: 2,
            convolution_kernel: 2,
        },
    ];
    for mixer in mixers.clone() {
        let spec = stack_of(vec![block(mixer, dense())]);
        if let Err(error) = spec.validate() {
            panic!("{error}");
        }
        assert_eq!(check_causal(&spec, true, &TOKENS), Outcome::Executed);
    }
    let stacked: Vec<_> = mixers
        .into_iter()
        .map(|mixer| block(mixer, FeedForwardSpec::None))
        .collect();
    assert_eq!(
        check_causal(&stack_of(stacked), true, &TOKENS),
        Outcome::Executed
    );
}

/// Routed experts construct and run (cache parity is skipped for them).
#[test]
fn routed_experts_construct_and_run() {
    let experts = FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
        routed_experts: 4,
        experts_per_token: 2,
        shared_experts: 1,
        expert_intermediate_size: 4,
        activation: Activation::Silu,
        renormalize_selected: true,
        routed_scale: 2.5,
        routing: RoutingPolicy {
            scoring: RouterScoring::Sigmoid,
            selection: ExpertSelection::GroupLimited(GroupLimitedSelection {
                groups: 2,
                top_groups: 1,
                group_score: GroupScore::TopTwoSum,
                pruned_score: PrunedExpertScore::NegativeInfinity,
            }),
            correction_bias: true,
            ..RoutingPolicy::default()
        },
        shared_expert_gate: true,
        expert_bias: true,
    });
    let spec = stack_of(vec![block(
        rope_attention(AttentionOptions::default()),
        experts,
    )]);
    assert!(has_routed_experts(&spec));
    assert_eq!(check_causal(&spec, true, &TOKENS), Outcome::Executed);
}

/// Clamped SwiGLU experts (gpt-oss) construct and run, and skip cache parity like routed experts.
#[test]
fn clamped_routed_experts_construct_and_run() {
    let experts = FeedForwardSpec::ClampedMixtureOfExperts {
        experts: MixtureOfExpertsSpec {
            routed_experts: 4,
            experts_per_token: 2,
            shared_experts: 1,
            expert_intermediate_size: 4,
            activation: Activation::Silu,
            renormalize_selected: true,
            routed_scale: 1.0,
            routing: RoutingPolicy::default(),
            shared_expert_gate: false,
            expert_bias: true,
        },
        limit: 7.0,
        alpha: 1.702,
        up_offset: 1.0,
    };
    let spec = stack_of(vec![block(
        rope_attention(AttentionOptions::default()),
        experts,
    )]);
    assert!(has_routed_experts(&spec));
    assert_eq!(check_causal(&spec, true, &TOKENS), Outcome::Executed);
}

/// A scaled embedding and a final logit cap execute and keep cache parity.
#[test]
fn embedding_scale_and_logit_cap_execute_with_cache_parity() {
    let mut spec = stack_of(vec![block(
        rope_attention(AttentionOptions::default()),
        FeedForwardSpec::None,
    )]);
    spec.embedding_scale = Some(8.0_f64.sqrt());
    spec.final_logit_softcap = Some(30.0);
    assert!(!has_routed_experts(&spec));
    assert_eq!(check_causal(&spec, true, &TOKENS), Outcome::Executed);
    spec.final_logit_softcap = Some(0.0);
    assert_eq!(check_causal(&spec, false, &TOKENS), Outcome::Invalid);
}

/// Capability gaps fail at construction, after validation succeeds.
#[test]
fn capability_gaps_are_reported_at_construction() {
    let mut gated = block(
        rope_attention(AttentionOptions::default()),
        FeedForwardSpec::None,
    );
    gated.residual = ResidualTopology::Gated;
    assert_eq!(
        check_causal(&stack_of(vec![gated]), true, &TOKENS),
        Outcome::Unsupported
    );
    let mamba = |convolution_kernel| {
        block(
            MixerSpec::Mamba2(Mamba2Spec {
                heads: 2,
                head_dim: 2,
                state_dim: 2,
                convolution_kernel,
                group_size: 1,
            }),
            FeedForwardSpec::None,
        )
    };
    // Mamba-2 mixers execute; only a width-one convolution is refused.
    assert_eq!(
        check_causal(&stack_of(vec![mamba(2)]), true, &TOKENS),
        Outcome::Executed
    );
    assert_eq!(
        check_causal(&stack_of(vec![mamba(1)]), true, &TOKENS),
        Outcome::Unsupported
    );
    let mut far = stack_of(vec![block(
        rope_attention(AttentionOptions::default()),
        FeedForwardSpec::None,
    )]);
    far.max_position_embeddings = 1 << 31;
    assert_eq!(check_causal(&far, true, &TOKENS), Outcome::Unsupported);
}

/// Encoder-decoders reject recurrent and sliding-window encoders and run otherwise.
#[test]
fn encoder_decoders_construct_and_run() {
    let encoder_layer = |mixer| block(mixer, FeedForwardSpec::None);
    let plain = MixerSpec::Attention(AttentionSpec {
        query_heads: 2,
        key_value_heads: 2,
        head_dim: 4,
        query_key_norm: false,
        output_gate: false,
        position: PositionEncodingSpec::T5RelativeBias {
            buckets: 8,
            max_distance: 16,
            bidirectional: true,
        },
    });
    let seq2seq = |encoder_mixer: MixerSpec| Seq2SeqSpec {
        encoder: stack_of(vec![encoder_layer(encoder_mixer)]),
        decoder: stack_of(vec![encoder_layer(plain.clone())]),
        cross_attention_heads: 2,
        cross_attention_head_dim: 4,
        cross_attention_options: AttentionOptions::default(),
    };
    assert_eq!(
        check_seq2seq(&seq2seq(plain.clone()), true, &TOKENS),
        Outcome::Executed
    );
    let recurrent = seq2seq(MixerSpec::DeltaNet {
        key_heads: 1,
        value_heads: 1,
        key_head_dim: 2,
        value_head_dim: 2,
        convolution_kernel: 2,
    });
    assert_eq!(
        check_seq2seq(&recurrent, true, &TOKENS),
        Outcome::Unsupported
    );
    let sliding = seq2seq(rope_attention(AttentionOptions {
        connectivity: AttentionConnectivity::SlidingWindow { window: 2 },
        ..AttentionOptions::default()
    }));
    assert_eq!(check_seq2seq(&sliding, true, &TOKENS), Outcome::Unsupported);
}

/// Committed seeds keep their meaning.
#[test]
fn committed_seeds_pass() {
    for seed in [
        &include_bytes!("../seeds/modular_specs/causal-stack")[..],
        &include_bytes!("../seeds/modular_specs/encoder-decoder")[..],
        &include_bytes!("../seeds/modular_specs/extreme-floats")[..],
    ] {
        exercise(seed);
    }
    exercise(&[]);
    exercise(&[255; 256]);
}

/// Pseudo-random inputs cover many architectures, reaching every outcome.
#[test]
fn pseudo_random_specs_validate_construct_and_run() {
    let mut outcomes = std::collections::BTreeMap::new();
    for input in crate::lcg_inputs(0x7370_6563, 256, 192) {
        *outcomes.entry(exercise_outcome(&input)).or_insert(0_usize) += 1;
    }
    assert_eq!(outcomes.len(), 4, "{outcomes:?}");
}
