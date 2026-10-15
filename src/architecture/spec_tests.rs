//! Independent architecture validation boundaries and field-specific error checks.

use super::*;

/// Builds the smallest valid dense attention layer used by validation tests.
fn dense_layer() -> DecoderLayerSpec {
    DecoderLayerSpec {
        mixer: MixerSpec::Attention(AttentionSpec {
            query_heads: 4,
            key_value_heads: 2,
            head_dim: 8,
            query_key_norm: true,
            output_gate: false,
            position: PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: 8,
                scaling: RopeScaling::None,
            },
        }),
        feed_forward: FeedForwardSpec::Dense(DenseFeedForwardSpec {
            intermediate_size: 64,
            activation: Activation::Silu,
            bias: false,
        }),
        mixer_norm: NormalizationSpec::Rms {
            epsilon: 1e-6,
            zero_centered: false,
        },
        feed_forward_norm: NormalizationSpec::Rms {
            epsilon: 1e-6,
            zero_centered: false,
        },
        norm_placement: NormPlacement::Pre,
        residual: ResidualTopology::Add,
    }
}

/// Builds the smallest valid modular decoder.
fn decoder() -> DecoderSpec {
    DecoderSpec {
        vocabulary_size: 128,
        hidden_size: 32,
        layers: vec![dense_layer()],
        final_norm: NormalizationSpec::Rms {
            epsilon: 1e-6,
            zero_centered: false,
        },
        tie_word_embeddings: false,
        max_position_embeddings: 256,
    }
}

/// Accepts a valid grouped-query decoder whose query width equals the residual stream.
#[test]
fn decoder_validation_accepts_valid_spec() {
    decoder().validate().expect("valid decoder must pass");
}

/// Configuration errors retain the exact nested field a caller can repair.
fn assert_configuration_field(result: Result<()>, expected: &str) {
    match result.expect_err("invalid configuration must return an error") {
        Error::Configuration { field, reason } => {
            assert_eq!(field, expected);
            assert!(!reason.is_empty());
        }
        error => panic!("expected configuration error for {expected}, got {error}"),
    }
}

/// Both normalization variants reject non-positive and non-finite stabilizers.
#[test]
fn normalization_validates_each_epsilon_boundary() {
    for epsilon in [0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        for norm in [
            NormalizationSpec::Rms {
                epsilon,
                zero_centered: true,
            },
            NormalizationSpec::Layer {
                epsilon,
                bias: true,
            },
        ] {
            assert_configuration_field(norm.validate("norm"), "norm.epsilon");
        }
    }
    NormalizationSpec::Layer {
        epsilon: f64::MIN_POSITIVE,
        bias: false,
    }
    .validate("norm")
    .unwrap();
}

/// Rotary pairs and their geometric base are validated independently of head grouping.
#[test]
fn position_encoding_rejects_invalid_base_width_and_bucket_distance() {
    for theta in [0.0, -10.0, f64::NAN, f64::INFINITY] {
        assert_configuration_field(
            PositionEncodingSpec::Rope {
                theta,
                rotary_dim: 2,
                scaling: RopeScaling::None,
            }
            .validate(4, "position"),
            "position.theta",
        );
    }
    for rotary_dim in [0, 3, 6] {
        assert_configuration_field(
            PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim,
                scaling: RopeScaling::None,
            }
            .validate(4, "position"),
            "position.rotary_dim",
        );
    }
    for buckets in [0, 1] {
        assert_configuration_field(
            PositionEncodingSpec::T5RelativeBias {
                buckets,
                max_distance: 16,
                bidirectional: false,
            }
            .validate(4, "position"),
            "position.buckets",
        );
    }
    for max_distance in [0, 15, 16] {
        assert_configuration_field(
            PositionEncodingSpec::T5RelativeBias {
                buckets: 32,
                max_distance,
                bidirectional: false,
            }
            .validate(4, "position"),
            "position.max_distance",
        );
    }
    PositionEncodingSpec::None.validate(0, "position").unwrap();
    PositionEncodingSpec::T5RelativeBias {
        buckets: 32,
        max_distance: 17,
        bidirectional: false,
    }
    .validate(4, "position")
    .unwrap();
}

/// YaRN rejects each singular interpolation/temperature boundary with its own field.
#[test]
fn yarn_validation_reports_expansion_temperature_and_transition_errors() {
    let validate = |factor, original_max_positions, attention_factor, beta_slow, beta_fast| {
        PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 4,
            scaling: RopeScaling::Yarn {
                factor,
                original_max_positions,
                attention_factor,
                beta_slow,
                beta_fast,
            },
        }
        .validate(4, "position")
    };
    validate(2.0, 128, None, 1.0, 32.0).unwrap();
    validate(2.0, 128, Some(1.0), 1.0, 32.0).unwrap();
    // Like Hugging Face and the shared YarnConfig, a factor of one (no extension) is valid.
    validate(1.0, 128, None, 1.0, 32.0).unwrap();
    for factor in [0.999, 0.0, f64::NAN, f64::INFINITY] {
        assert_configuration_field(
            validate(factor, 128, None, 1.0, 32.0),
            "position.scaling.factor",
        );
    }
    assert_configuration_field(
        validate(2.0, 0, None, 1.0, 32.0),
        "position.scaling.original_max_positions",
    );
    for factor in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_configuration_field(
            validate(2.0, 128, Some(factor), 1.0, 32.0),
            "position.scaling.attention_factor",
        );
    }
    for (slow, fast) in [
        (0.0, 32.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (f64::NAN, 32.0),
        (1.0, f64::INFINITY),
    ] {
        assert_configuration_field(
            validate(2.0, 128, None, slow, fast),
            "position.scaling.beta",
        );
    }
}

/// Every attention and recurrent-state axis must be nonzero before shape arithmetic.
#[test]
fn mixers_validate_zero_dimensions_and_grouping() {
    for index in 0..3 {
        let mut axes = [2, 1, 4];
        axes[index] = 0;
        assert_configuration_field(
            AttentionSpec {
                query_heads: axes[0],
                key_value_heads: axes[1],
                head_dim: axes[2],
                query_key_norm: false,
                output_gate: false,
                position: PositionEncodingSpec::None,
            }
            .validate("attention"),
            "attention",
        );
    }
    let attention = AttentionSpec {
        query_heads: 2,
        key_value_heads: 1,
        head_dim: 4,
        query_key_norm: false,
        output_gate: false,
        position: PositionEncodingSpec::None,
    };
    assert_configuration_field(
        MixerSpec::SlidingAttention(SlidingWindowSpec {
            attention: attention.clone(),
            window: 0,
        })
        .validate("mixer"),
        "mixer.window",
    );
    MixerSpec::SlidingAttention(SlidingWindowSpec {
        attention,
        window: 1,
    })
    .validate("mixer")
    .unwrap();
    for index in 0..5 {
        let mut axes = [1, 2, 4, 4, 3];
        axes[index] = 0;
        assert_configuration_field(
            MixerSpec::DeltaNet {
                key_heads: axes[0],
                value_heads: axes[1],
                key_head_dim: axes[2],
                value_head_dim: axes[3],
                convolution_kernel: axes[4],
            }
            .validate("delta"),
            "delta",
        );
    }
    assert_configuration_field(
        MixerSpec::DeltaNet {
            key_heads: 2,
            value_heads: 3,
            key_head_dim: 4,
            value_head_dim: 4,
            convolution_kernel: 3,
        }
        .validate("delta"),
        "delta.value_heads",
    );
    for index in 0..5 {
        let mut axes = [4, 4, 8, 3, 2];
        axes[index] = 0;
        assert_configuration_field(
            MixerSpec::Mamba2(Mamba2Spec {
                heads: axes[0],
                head_dim: axes[1],
                state_dim: axes[2],
                convolution_kernel: axes[3],
                group_size: axes[4],
            })
            .validate("mamba"),
            "mamba",
        );
    }
    for group_size in [1, 2, 3, 4, 5] {
        let result = MixerSpec::Mamba2(Mamba2Spec {
            heads: 4,
            head_dim: 4,
            state_dim: 8,
            convolution_kernel: 3,
            group_size,
        })
        .validate("mamba");
        if [1, 2, 4].contains(&group_size) {
            result.unwrap();
        } else {
            assert_configuration_field(result, "mamba");
        }
    }
}

/// Serialized MLA dimensions cannot overflow even before a runtime rejects the mixer.
#[test]
fn mla_validation_rejects_zero_ranks_and_overflowing_head_width() {
    let valid = MlaSpec {
        query_heads: 2,
        content_head_dim: 4,
        rotary_head_dim: 2,
        value_head_dim: 4,
        key_value_lora_rank: 2,
        query_lora_rank: None,
        position: PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 2,
            scaling: RopeScaling::None,
        },
    };
    MixerSpec::Mla(valid.clone()).validate("mla").unwrap();
    for index in 0..6 {
        let mut invalid = valid.clone();
        match index {
            0 => invalid.query_heads = 0,
            1 => invalid.content_head_dim = 0,
            2 => invalid.rotary_head_dim = 0,
            3 => invalid.value_head_dim = 0,
            4 => invalid.key_value_lora_rank = 0,
            _ => invalid.query_lora_rank = Some(0),
        }
        assert_configuration_field(MixerSpec::Mla(invalid).validate("mla"), "mla");
    }
    let mut invalid = valid;
    invalid.content_head_dim = usize::MAX;
    assert_configuration_field(MixerSpec::Mla(invalid).validate("mla"), "mla.head_dim");
}

/// Dense widths, routed cardinalities and scaling each have recoverable diagnostics.
#[test]
fn feed_forward_validation_covers_empty_experts_and_invalid_scales() {
    assert_configuration_field(
        FeedForwardSpec::Dense(DenseFeedForwardSpec {
            intermediate_size: 0,
            activation: Activation::Silu,
            bias: false,
        })
        .validate("ffn"),
        "ffn.intermediate_size",
    );
    let valid = MixtureOfExpertsSpec {
        routed_experts: 4,
        experts_per_token: 2,
        shared_experts: 1,
        expert_intermediate_size: 8,
        activation: Activation::ReluSquared,
        renormalize_selected: true,
        routed_scale: 1.0,
    };
    FeedForwardSpec::MixtureOfExperts(valid.clone())
        .validate("ffn")
        .unwrap();
    for index in 0..3 {
        let mut invalid = valid.clone();
        match index {
            0 => invalid.routed_experts = 0,
            1 => invalid.expert_intermediate_size = 0,
            _ => invalid.experts_per_token = 0,
        }
        assert_configuration_field(
            FeedForwardSpec::MixtureOfExperts(invalid).validate("ffn"),
            if index == 2 {
                "ffn.experts_per_token"
            } else {
                "ffn"
            },
        );
    }
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut invalid = valid.clone();
        invalid.routed_scale = scale;
        assert_configuration_field(
            FeedForwardSpec::MixtureOfExperts(invalid).validate("ffn"),
            "ffn.routed_scale",
        );
    }
}

/// Global and later-layer validation errors retain precise field paths.
#[test]
fn decoder_reports_global_and_nested_normalization_fields() {
    for (index, expected) in [
        "vocabulary_size",
        "hidden_size",
        "layers",
        "max_position_embeddings",
        "final_norm.epsilon",
    ]
    .into_iter()
    .enumerate()
    {
        let mut invalid = decoder();
        match index {
            0 => invalid.vocabulary_size = 0,
            1 => invalid.hidden_size = 0,
            2 => invalid.layers.clear(),
            3 => invalid.max_position_embeddings = 0,
            _ => {
                invalid.final_norm = NormalizationSpec::Layer {
                    epsilon: 0.0,
                    bias: false,
                }
            }
        }
        assert_configuration_field(invalid.validate(), expected);
    }
    for mixer_side in [true, false] {
        let mut invalid = decoder();
        invalid.layers.push(dense_layer());
        let invalid_norm = NormalizationSpec::Rms {
            epsilon: 0.0,
            zero_centered: false,
        };
        if mixer_side {
            invalid.layers[1].mixer_norm = invalid_norm;
        } else {
            invalid.layers[1].feed_forward_norm = invalid_norm;
        }
        assert_configuration_field(
            invalid.validate(),
            if mixer_side {
                "layers[1].mixer_norm.epsilon"
            } else {
                "layers[1].feed_forward_norm.epsilon"
            },
        );
    }
}

/// Rejects KV head counts that cannot be repeated evenly across query heads.
#[test]
fn attention_rejects_non_divisible_grouping() {
    let mut spec = decoder();
    let MixerSpec::Attention(attention) = &mut spec.layers[0].mixer else {
        unreachable!("fixture uses attention")
    };
    attention.key_value_heads = 3;
    let error = spec.validate().expect_err("invalid GQA ratio must fail");
    assert!(error.to_string().contains("must divide query_heads"));
}

/// Rejects odd rotary dimensions because channel pairs would be incomplete.
#[test]
fn rope_rejects_unpaired_channel() {
    let mut spec = decoder();
    let MixerSpec::Attention(attention) = &mut spec.layers[0].mixer else {
        unreachable!("fixture uses attention")
    };
    let PositionEncodingSpec::Rope { rotary_dim, .. } = &mut attention.position else {
        unreachable!("fixture uses RoPE")
    };
    *rotary_dim = 7;
    let error = spec.validate().expect_err("odd rotary width must fail");
    let message = error.to_string();
    assert!(
        message.contains("layers[0].mixer.position.rotary_dim"),
        "{message}"
    );
    assert!(message.contains("must be even"), "{message}");
}

/// Rejects routers that select more experts than the checkpoint owns.
#[test]
fn moe_rejects_impossible_top_k() {
    let spec = FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
        routed_experts: 4,
        experts_per_token: 5,
        shared_experts: 0,
        expert_intermediate_size: 16,
        activation: Activation::Silu,
        renormalize_selected: true,
        routed_scale: 1.0,
    });
    let error = spec
        .validate("moe")
        .expect_err("top-k exceeds expert count");
    assert!(error.to_string().contains("1..=routed_experts"));
}

/// Validates matching encoder and decoder widths for cross attention.
#[test]
fn seq2seq_requires_matching_residual_widths() {
    let encoder = decoder();
    let mut decoder = decoder();
    decoder.hidden_size = 64;
    let MixerSpec::Attention(attention) = &mut decoder.layers[0].mixer else {
        unreachable!("fixture uses attention")
    };
    attention.head_dim = 16;
    let spec = Seq2SeqSpec {
        encoder,
        decoder,
        cross_attention_heads: 4,
        cross_attention_head_dim: 8,
    };
    let error = spec.validate().expect_err("mismatched stacks must fail");
    assert!(error.to_string().contains("encoder.hidden_size"));
}

/// Rejects overflow before it can panic or wrap into a valid residual width.
#[test]
fn seq2seq_rejects_overflowing_cross_attention_width() {
    let spec = Seq2SeqSpec {
        encoder: decoder(),
        decoder: decoder(),
        // On every supported usize width this product wraps to 32, matching
        // the fixture's hidden size and bypassing unchecked release builds.
        cross_attention_heads: usize::MAX / 2 + 17,
        cross_attention_head_dim: 2,
    };
    let error = spec.validate().expect_err("overflow must be rejected");
    assert!(error.to_string().contains("cross_attention"));
}

/// Accepts any nonzero cross-attention inner width, including widths unequal to the residual.
#[test]
fn seq2seq_checks_cross_attention_width() {
    for (heads, head_dim, valid) in [
        (4, 8, true),
        (0, 8, false),
        (4, 0, false),
        (3, 8, true),
        (32, 128, true),
    ] {
        let spec = Seq2SeqSpec {
            encoder: decoder(),
            decoder: decoder(),
            cross_attention_heads: heads,
            cross_attention_head_dim: head_dim,
        };
        assert_eq!(spec.validate().is_ok(), valid);
    }
}

proptest::proptest! {
    /// Checks every valid GQA divisor accepted by the arithmetic invariant.
    #[test]
    fn gqa_accepts_exact_divisors(query_heads in 1usize..64, divisor in 1usize..64) {
        let key_value_heads = divisor.min(query_heads);
        let spec = AttentionSpec {
            query_heads,
            key_value_heads,
            head_dim: 2,
            query_key_norm: false,
            output_gate: false,
            position: PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: 2,
                scaling: RopeScaling::None,
            },
        };
        proptest::prop_assert_eq!(
            spec.validate("attention").is_ok(),
            query_heads.is_multiple_of(key_value_heads)
        );
    }
}

// Delegation to the shared position components.

/// T5 bucket validation follows the shared, direction-aware exact region.
///
/// Bidirectional tables halve the buckets before `max_exact = per_direction / 2`,
/// as Hugging Face's `_relative_position_bucket` does.
#[test]
fn t5_bucket_validation_matches_the_direction_aware_component() {
    let t5 = |buckets, max_distance, bidirectional| PositionEncodingSpec::T5RelativeBias {
        buckets,
        max_distance,
        bidirectional,
    };
    // Previously rejected although valid: max_exact is 8 for 32 bidirectional buckets.
    t5(32, 12, true).validate(4, "position").unwrap();
    t5(32, 9, true).validate(4, "position").unwrap();
    assert_configuration_field(
        t5(32, 8, true).validate(4, "position"),
        "position.max_distance",
    );
    // Previously accepted although the runtime table cannot be built.
    assert_configuration_field(
        t5(33, 128, true).validate(4, "position"),
        "position.buckets",
    );
    for buckets in [2, 3] {
        assert_configuration_field(
            t5(buckets, 128, true).validate(4, "position"),
            "position.buckets",
        );
    }
    t5(3, 128, false).validate(4, "position").unwrap();
    let buckets = t5(32, 12, true).t5_buckets().unwrap().unwrap();
    assert_eq!(
        (
            buckets.num_buckets(),
            buckets.max_distance(),
            buckets.is_bidirectional()
        ),
        (32, 12, true)
    );
    assert!(PositionEncodingSpec::None.t5_buckets().unwrap().is_none());
}

/// MLA RoPE is checked against the rotary sub-head, not the combined head width.
#[test]
fn mla_rope_width_is_bounded_by_the_rotary_sub_head() {
    let mla = |rotary_dim| {
        MixerSpec::Mla(MlaSpec {
            query_heads: 2,
            content_head_dim: 4,
            rotary_head_dim: 2,
            value_head_dim: 4,
            key_value_lora_rank: 2,
            query_lora_rank: None,
            position: PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim,
                scaling: RopeScaling::None,
            },
        })
    };
    mla(2).validate("mla").unwrap();
    for rotary_dim in [4, 6] {
        assert_configuration_field(mla(rotary_dim).validate("mla"), "mla.position.rotary_dim");
    }
    // Found by the initialization-agreement property: the shared MLA component
    // rotates complete channel pairs, so an odd rotary sub-head is invalid even
    // when the RoPE prefix itself is even.
    let MixerSpec::Mla(mut odd) = mla(2) else {
        unreachable!("fixture is MLA")
    };
    odd.rotary_head_dim = 5;
    assert_configuration_field(
        MixerSpec::Mla(odd).validate("mla"),
        "mla.mla.qk_rope_head_dim",
    );
}

/// Component checks run at the stack's real residual width, bounding projection sizes.
#[cfg(target_pointer_width = "64")]
#[test]
fn decoder_validation_checks_projection_sizes_at_the_residual_width() {
    let mut spec = decoder();
    // 2^33 * 2^32 query-projection elements overflow 64-bit usize; one layer of
    // width one does not, so only the stack-level check can catch it.
    spec.hidden_size = 1 << 33;
    let MixerSpec::Attention(attention) = &mut spec.layers[0].mixer else {
        unreachable!("fixture uses attention")
    };
    attention.query_heads = 1 << 24;
    attention.key_value_heads = 1;
    attention.head_dim = 1 << 8;
    let PositionEncodingSpec::Rope { rotary_dim, .. } = &mut attention.position else {
        unreachable!("fixture uses RoPE")
    };
    *rotary_dim = 2;
    spec.layers[0].validate(0).unwrap();
    assert_configuration_field(spec.validate(), "layers[0].mixer.attention");
}

/// YaRN's logarithmic correction needs `theta > 1`, which plain RoPE does not.
#[test]
fn yarn_requires_a_base_above_one() {
    let rope = |theta, scaling| PositionEncodingSpec::Rope {
        theta,
        rotary_dim: 4,
        scaling,
    };
    let yarn = RopeScaling::Yarn {
        factor: 2.0,
        original_max_positions: 64,
        attention_factor: None,
        beta_slow: 1.0,
        beta_fast: 32.0,
    };
    rope(1.0, RopeScaling::None)
        .validate(4, "position")
        .unwrap();
    for theta in [1.0, 0.5] {
        assert_configuration_field(rope(theta, yarn).validate(4, "position"), "position.theta");
    }
    let config = rope(10_000.0, yarn).rope_config(4).unwrap().unwrap();
    assert_eq!(config.rotary_dimension(), 4);
    match config.scaling() {
        RuntimeRopeScaling::Yarn(runtime) => {
            assert_eq!(runtime.factor(), 2.0);
            assert_eq!(runtime.original_max_position_embeddings(), 64);
            // The default amplitude follows YarnConfig: 1 + 0.1 ln(factor).
            assert!((runtime.attention_factor() - (1.0 + 0.1 * 2.0_f64.ln())).abs() < 1e-12);
        }
        other => panic!("expected YaRN, found {other:?}"),
    }
    assert!(PositionEncodingSpec::None.rope_config(4).unwrap().is_none());
}

/// A cross-attention inner width unequal to the residual width builds and runs.
#[cfg(feature = "flex")]
#[test]
fn seq2seq_with_a_narrower_cross_attention_width_builds_and_runs() {
    use crate::{CausalLmInput, ForwardOptions, Seq2SeqInput, models::custom::ModularSeq2SeqLm};
    use burn::{backend::Flex, tensor::Tensor};
    let device = Default::default();
    let spec = Seq2SeqSpec {
        encoder: decoder(),
        decoder: decoder(),
        cross_attention_heads: 3,
        cross_attention_head_dim: 8,
    };
    spec.validate().unwrap();
    let model = ModularSeq2SeqLm::<Flex>::init(spec, &device).unwrap();
    let input = |ids: [[i32; 3]; 1]| CausalLmInput {
        token_ids: Tensor::from_ints(ids, &device),
        token_mask: None,
        position_ids: None,
    };
    let output = model
        .forward(
            Seq2SeqInput {
                encoder: input([[4, 5, 6]]),
                decoder: input([[0, 1, 2]]),
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(output.decoder.logits.dims(), [1, 1, 128]);
}

#[cfg(feature = "flex")]
proptest::proptest! {
    /// A layer spec validates exactly when the modular layer it describes can be built.
    ///
    /// Covers RoPE (plain and YaRN), T5 buckets in both directions, and MLA,
    /// across zero, odd, oversized, and boundary dimensions.
    #[test]
    fn layer_validation_agrees_with_modular_initialization(
        query_heads in 1usize..4,
        head_dim in 1usize..7,
        rotary_dim in 0usize..8,
        theta_choice in 0usize..3,
        yarn in proptest::bool::ANY,
        buckets in 0usize..12,
        max_distance in 0usize..12,
        bidirectional in proptest::bool::ANY,
        encoding in 0usize..4,
    ) {
        use crate::models::custom::ModularLayer;
        let theta = [0.5, 1.0, 10_000.0][theta_choice];
        let scaling = if yarn {
            RopeScaling::Yarn {
                factor: 2.0,
                original_max_positions: 16,
                attention_factor: None,
                beta_slow: 1.0,
                beta_fast: 32.0,
            }
        } else {
            RopeScaling::None
        };
        let rope = PositionEncodingSpec::Rope { theta, rotary_dim, scaling };
        let mixer = match encoding {
            0 | 1 => MixerSpec::Attention(AttentionSpec {
                query_heads,
                key_value_heads: 1,
                head_dim,
                query_key_norm: false,
                output_gate: false,
                position: if encoding == 0 {
                    rope
                } else {
                    PositionEncodingSpec::T5RelativeBias { buckets, max_distance, bidirectional }
                },
            }),
            2 => MixerSpec::Mla(MlaSpec {
                query_heads,
                content_head_dim: head_dim,
                rotary_head_dim: rotary_dim.max(1),
                value_head_dim: head_dim,
                key_value_lora_rank: 2,
                query_lora_rank: None,
                position: PositionEncodingSpec::Rope { theta, rotary_dim: head_dim, scaling },
            }),
            _ => MixerSpec::Attention(AttentionSpec {
                query_heads,
                key_value_heads: 1,
                head_dim,
                query_key_norm: true,
                output_gate: true,
                position: PositionEncodingSpec::None,
            }),
        };
        let mut layer = dense_layer();
        layer.mixer = mixer;
        let validated = layer.validate(0);
        let built = ModularLayer::<burn::backend::Flex>::init(&layer, 8, None, &Default::default());
        proptest::prop_assert_eq!(
            validated.is_ok(),
            built.is_ok(),
            "validate: {:?}, init: {:?}",
            validated.err(),
            built.err().map(|error| error.to_string())
        );
    }
}
