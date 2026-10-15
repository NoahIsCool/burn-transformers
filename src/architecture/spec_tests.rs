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
                truncate: true,
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
        score_scale: ScoreScale::Auto,
        connectivity: AttentionConnectivity::Global,
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
        routing: RoutingPolicy::default(),
        shared_expert_gate: false,
        expert_bias: false,
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
        routing: RoutingPolicy::default(),
        shared_expert_gate: false,
        expert_bias: false,
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
        cross_attention_options: AttentionOptions::default(),
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
        cross_attention_options: AttentionOptions::default(),
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
            cross_attention_options: AttentionOptions::default(),
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
            score_scale: ScoreScale::Auto,
            connectivity: AttentionConnectivity::Global,
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
        truncate: true,
    };
    rope(1.0, RopeScaling::None)
        .validate(4, "position")
        .unwrap();
    for theta in [1.0, 0.5] {
        assert_configuration_field(
            rope(theta, yarn.clone()).validate(4, "position"),
            "position.theta",
        );
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
        cross_attention_options: AttentionOptions::default(),
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

/// Builds a position encoding's runtime object directly, mirroring the modular models.
///
/// RoPE goes through each runtime scaling constructor and `RopeConfig` over
/// `head_dim` channels, and T5 relative bias through
/// `T5RelativePositionBiasConfig::init` with one table column per head. No
/// spec validation or spec translation is consulted.
#[cfg(feature = "flex")]
fn build_position_directly(
    position: &PositionEncodingSpec,
    head_dim: usize,
    heads: usize,
) -> Result<()> {
    let rope_error = |error: RopeConfigError| Error::configuration("rope", error.to_string());
    match position {
        PositionEncodingSpec::None => Ok(()),
        PositionEncodingSpec::Rope {
            theta,
            rotary_dim,
            scaling,
        } => {
            let scaling: RuntimeRopeScaling = match scaling {
                RopeScaling::None => RuntimeRopeScaling::None,
                RopeScaling::Linear { factor } => {
                    LinearRopeScaling::new(*factor).map_err(rope_error)?.into()
                }
                RopeScaling::DynamicNtk {
                    factor,
                    original_max_positions,
                } => DynamicNtkScaling::new(*factor, *original_max_positions)
                    .map_err(rope_error)?
                    .into(),
                RopeScaling::Yarn {
                    factor,
                    original_max_positions,
                    attention_factor,
                    beta_slow,
                    beta_fast,
                    truncate,
                } => {
                    let mut yarn = YarnConfig::new(*factor, *original_max_positions)
                        .and_then(|yarn| yarn.with_betas(*beta_fast, *beta_slow))
                        .map_err(rope_error)?
                        .with_truncate(*truncate);
                    if let Some(attention_factor) = attention_factor {
                        yarn = yarn
                            .with_attention_factor(*attention_factor)
                            .map_err(rope_error)?;
                    }
                    yarn.into()
                }
                RopeScaling::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_positions,
                } => Llama3RopeScaling::new(
                    *factor,
                    *low_freq_factor,
                    *high_freq_factor,
                    *original_max_positions,
                )
                .map_err(rope_error)?
                .into(),
                RopeScaling::LongRope {
                    short_factors,
                    long_factors,
                    original_max_positions,
                    factor,
                    attention_factor,
                } => {
                    let mut long = LongRopeScaling::new(
                        short_factors,
                        long_factors,
                        *original_max_positions,
                        *factor,
                    )
                    .map_err(rope_error)?;
                    if let Some(attention_factor) = attention_factor {
                        long = long
                            .with_attention_factor(*attention_factor)
                            .map_err(rope_error)?;
                    }
                    long.into()
                }
                RopeScaling::Proportional {
                    rotated_fraction,
                    factor,
                } => ProportionalRopeScaling::new(*rotated_fraction, *factor)
                    .map_err(rope_error)?
                    .into(),
            };
            RopeConfig::new(head_dim, *rotary_dim, *theta, scaling)
                .map(drop)
                .map_err(rope_error)
        }
        PositionEncodingSpec::T5RelativeBias {
            buckets,
            max_distance,
            bidirectional,
        } => crate::position::T5RelativePositionBiasConfig::new(
            heads,
            *buckets,
            *max_distance,
            *bidirectional,
        )
        .init::<burn::backend::Flex>(&Default::default())
        .map(drop),
    }
}

/// Builds the runtime modules a mixer spec describes at `hidden_size`, bypassing spec validation.
///
/// Mirrors the modular models' construction (grouped-query attention with the
/// options' gate, scale, soft cap, sinks, biases, and Q/K-norm scope; legacy
/// unit T5 score scale; MLA with RoPE over its rotary sub-head; Gated DeltaNet
/// with an explicit kernel) using only the components' own `init`, which
/// validates and allocates. Connectivity is checked by its own validator.
#[cfg(feature = "flex")]
fn build_mixer_directly(mixer: &MixerSpec, hidden_size: usize) -> Result<()> {
    /// Deterministic CPU backend used only to allocate the probe modules.
    type F = burn::backend::Flex;
    let device = Default::default();
    let grouped = |spec: &AttentionSpec, options: &AttentionOptions| -> Result<()> {
        options.connectivity.validate()?;
        let mut config = GroupedQueryAttentionConfig::new(
            hidden_size,
            spec.query_heads,
            spec.key_value_heads,
            spec.head_dim,
        );
        config.query_key_norm_epsilon = spec.query_key_norm.then_some(1e-6);
        config.query_key_norm_scope = options.query_key_norm_scope;
        if options.query_key_norm_scope == QueryKeyNormScope::FullWidth && !spec.query_key_norm {
            return Err(Error::configuration("scope", "requires query_key_norm"));
        }
        config.output_gate = match (spec.output_gate, options.output_gate_activation) {
            (false, OutputGateActivation::Sigmoid) => AttentionOutputGate::None,
            (false, OutputGateActivation::Silu) => {
                return Err(Error::configuration("gate", "requires output_gate"));
            }
            (true, OutputGateActivation::Sigmoid) => AttentionOutputGate::Sigmoid,
            (true, OutputGateActivation::Silu) => AttentionOutputGate::Silu,
        };
        config.scale = match options.score_scale {
            ScoreScale::Auto
                if matches!(spec.position, PositionEncodingSpec::T5RelativeBias { .. }) =>
            {
                Some(1.0)
            }
            ScoreScale::Auto | ScoreScale::InverseSqrtHeadDim => None,
            ScoreScale::Unscaled => Some(1.0),
            ScoreScale::Fixed { value } => Some(value),
        };
        config.softcap = options.logit_softcap;
        config.attention_sinks = options.attention_sinks;
        config.projection_bias = Some(options.projection_bias);
        config.init::<F>(&device)?;
        build_position_directly(&spec.position, spec.head_dim, spec.query_heads)
    };
    match mixer {
        MixerSpec::Attention(spec) => grouped(spec, &AttentionOptions::default()),
        MixerSpec::AttentionWithOptions { attention, options } => grouped(attention, options),
        MixerSpec::Mla(spec) => {
            MultiLatentAttentionConfig::new(
                hidden_size,
                spec.query_heads,
                spec.content_head_dim,
                spec.rotary_head_dim,
                spec.value_head_dim,
                spec.key_value_lora_rank,
                spec.query_lora_rank,
            )
            .init::<F>(&device)?;
            build_position_directly(&spec.position, spec.rotary_head_dim, spec.query_heads)
        }
        MixerSpec::DeltaNet {
            key_heads,
            value_heads,
            key_head_dim,
            value_head_dim,
            convolution_kernel,
        } => {
            let mut config = GatedDeltaNetConfig::new(
                hidden_size,
                *key_heads,
                *value_heads,
                *key_head_dim,
                *value_head_dim,
            );
            config.convolution_kernel = *convolution_kernel;
            config.init::<F>(&device).map(drop)
        }
        other => panic!("no direct construction for unsampled mixer {other:?}"),
    }
}

#[cfg(feature = "flex")]
proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(2048))]

    /// Spec validation accepts exactly the mixers whose components build directly.
    ///
    /// The oracle never calls spec validation, so this catches both drift
    /// directions: an over-strict spec check (rejecting a mixer every
    /// component accepts, as the old T5 `max_distance` and YaRN factor-one
    /// rules did) and an under-strict one (accepting a mixer some component
    /// rejects). Sampled at the stack's real width of 8 over grouped-query
    /// attention with RoPE, YaRN, or T5 bias; MLA with RoPE or T5 bias; and
    /// Gated DeltaNet. Each dimension is drawn from a short list that mixes
    /// valid values with zero, odd, oversized, and boundary values, so both
    /// outcomes are common.
    #[test]
    fn mixer_validation_agrees_with_direct_component_construction(
        kind in 0usize..5,
        heads in proptest::prop_oneof![1 => proptest::strategy::Just(0usize), 6 => 1usize..5],
        grouping in 0usize..4,
        head_dim_choice in 0usize..6,
        rotary_choice in 0usize..5,
        theta_choice in 0usize..4,
        yarn_choice in 0usize..9,
        bucket_choice in 0usize..6,
        distance_choice in 0usize..6,
        bidirectional in proptest::bool::ANY,
        flags in 0usize..4,
        kernel_choice in 0usize..3,
    ) {
        // Grouping: equal heads, one shared head, none, or a non-divisor.
        let paired = [heads, 1, 0, heads + 1][grouping];
        let head_dim = [2, 4, 6, 8, 3, 0][head_dim_choice];
        let rotary = |width: usize| [width, width / 2, 0, width + 2, 1][rotary_choice];
        let theta = [10_000.0, 10_000.0, 1.0, 0.5][theta_choice];
        let scaling = match yarn_choice {
            0 => RopeScaling::None,
            1 | 2 => RopeScaling::Yarn {
                // Factor one is the boundary the spec once rejected.
                factor: [1.0, 2.0][yarn_choice - 1],
                original_max_positions: 16,
                attention_factor: None,
                beta_slow: 1.0,
                beta_fast: 32.0,
                truncate: yarn_choice == 1,
            },
            3 => RopeScaling::Linear { factor: [2.0, 0.5][kernel_choice % 2] },
            4 => RopeScaling::DynamicNtk {
                factor: 2.0,
                original_max_positions: [16, 0][kernel_choice % 2],
            },
            5 => RopeScaling::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                // Equal frequency factors make the smoothing denominator zero.
                high_freq_factor: [4.0, 1.0][kernel_choice % 2],
                original_max_positions: 16,
            },
            6 => RopeScaling::LongRope {
                // One factor per rotated complex pair; mismatched lengths are invalid.
                short_factors: vec![1.0; 1 + kernel_choice],
                long_factors: vec![2.0; 1 + kernel_choice],
                original_max_positions: 16,
                factor: 4.0,
                attention_factor: None,
            },
            7 => RopeScaling::Proportional {
                rotated_fraction: [0.5, 1.5][kernel_choice % 2],
                factor: 1.0,
            },
            _ => RopeScaling::Yarn {
                factor: 2.0,
                original_max_positions: 16,
                attention_factor: Some([1.1, 0.0][kernel_choice % 2]),
                beta_slow: 1.0,
                beta_fast: 32.0,
                truncate: true,
            },
        };
        let position = |width: usize| {
            if flags & 1 == 0 {
                PositionEncodingSpec::Rope { theta, rotary_dim: rotary(width), scaling }
            } else {
                PositionEncodingSpec::T5RelativeBias {
                    buckets: [32, 8, 4, 3, 2, 0][bucket_choice],
                    max_distance: [128, 8, 4, 2, 1, 0][distance_choice],
                    bidirectional,
                }
            }
        };
        let mixer = match kind {
            0 => MixerSpec::Attention(AttentionSpec {
                query_heads: heads,
                key_value_heads: paired,
                head_dim,
                query_key_norm: flags & 2 != 0,
                output_gate: flags & 2 != 0,
                position: position(head_dim),
            }),
            1 => MixerSpec::Attention(AttentionSpec {
                query_heads: heads,
                key_value_heads: paired,
                head_dim,
                query_key_norm: false,
                output_gate: false,
                position: PositionEncodingSpec::None,
            }),
            2 => {
                // The rotary sub-head is rotated; content channels never are.
                let rotary_head_dim = [2, 4, 3, 0][grouping];
                MixerSpec::Mla(MlaSpec {
                    query_heads: heads,
                    content_head_dim: head_dim,
                    rotary_head_dim,
                    value_head_dim: head_dim,
                    key_value_lora_rank: 2,
                    query_lora_rank: (flags & 2 != 0).then_some(rotary_choice),
                    position: position(rotary_head_dim),
                    score_scale: ScoreScale::Auto,
                    connectivity: AttentionConnectivity::Global,
                })
            }
            3 => MixerSpec::AttentionWithOptions {
                attention: AttentionSpec {
                    query_heads: heads,
                    key_value_heads: paired,
                    head_dim,
                    query_key_norm: flags & 2 != 0,
                    output_gate: flags & 2 == 0,
                    position: position(head_dim),
                },
                options: AttentionOptions {
                    connectivity: [
                        AttentionConnectivity::Global,
                        AttentionConnectivity::SlidingWindow { window: 4 },
                        AttentionConnectivity::ChunkedLocal { chunk: 0 },
                    ][kernel_choice],
                    score_scale: [
                        ScoreScale::Auto,
                        ScoreScale::Unscaled,
                        ScoreScale::Fixed { value: 0.5 },
                        ScoreScale::Fixed { value: -1.0 },
                    ][grouping],
                    logit_softcap: [None, Some(30.0), Some(0.0)][kernel_choice],
                    attention_sinks: flags & 1 == 0,
                    projection_bias: ProjectionBias::query_key_value_only(),
                    query_key_norm_scope: [QueryKeyNormScope::PerHead, QueryKeyNormScope::FullWidth]
                        [rotary_choice % 2],
                    output_gate_activation: [OutputGateActivation::Sigmoid, OutputGateActivation::Silu]
                        [theta_choice % 2],
                },
            },
            _ => MixerSpec::DeltaNet {
                key_heads: heads,
                // Value heads must be a positive multiple of key heads.
                value_heads: [heads, 2 * heads, 0, heads + 1][grouping],
                key_head_dim: head_dim,
                value_head_dim: rotary(head_dim),
                convolution_kernel: [4, 1, 0][kernel_choice],
            },
        };
        let mut layer = dense_layer();
        layer.mixer = mixer.clone();
        let spec = DecoderSpec {
            hidden_size: 8,
            layers: vec![layer],
            ..decoder()
        };
        let validated = spec.validate();
        let built = build_mixer_directly(&mixer, spec.hidden_size);
        proptest::prop_assert_eq!(
            validated.is_ok(),
            built.is_ok(),
            "mixer {:?}\nvalidate: {:?}\ndirect: {:?}",
            mixer,
            validated.err(),
            built.err()
        );
    }
}

#[cfg(feature = "flex")]
proptest::proptest! {
    /// A layer that validates can be built by the modular models.
    ///
    /// `ModularLayer::init` runs `validate` first, so only this direction is
    /// informative: it catches initialization-time requirements that
    /// validation misses. Over-strict validation is caught by
    /// `mixer_validation_agrees_with_direct_component_construction`.
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
                truncate: true,
            }
        } else {
            RopeScaling::None
        };
        let rope = PositionEncodingSpec::Rope { theta, rotary_dim, scaling: scaling.clone() };
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
                score_scale: ScoreScale::Auto,
                connectivity: AttentionConnectivity::Global,
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

/// Specs serialized before the explicit options existed load unchanged with legacy defaults.
#[test]
fn legacy_serialized_specs_load_with_default_options() {
    let json = r#"{
        "encoder": {
            "vocabulary_size": 16, "hidden_size": 8, "max_position_embeddings": 32,
            "tie_word_embeddings": false,
            "final_norm": {"kind": "rms", "epsilon": 1e-6, "zero_centered": false},
            "layers": [{
                "mixer": {"kind": "mla", "query_heads": 2, "content_head_dim": 4,
                    "rotary_head_dim": 4, "value_head_dim": 4, "key_value_lora_rank": 4,
                    "query_lora_rank": null,
                    "position": {"kind": "rope", "theta": 10000.0, "rotary_dim": 4,
                        "scaling": {"kind": "yarn", "factor": 2.0, "original_max_positions": 16,
                            "attention_factor": null, "beta_slow": 1.0, "beta_fast": 32.0}}},
                "feed_forward": {"kind": "mixture_of_experts", "routed_experts": 4,
                    "experts_per_token": 2, "shared_experts": 0, "expert_intermediate_size": 4,
                    "activation": "silu", "renormalize_selected": true, "routed_scale": 1.0},
                "mixer_norm": {"kind": "rms", "epsilon": 1e-6, "zero_centered": false},
                "feed_forward_norm": {"kind": "rms", "epsilon": 1e-6, "zero_centered": false},
                "norm_placement": "pre", "residual": "add"
            }]
        },
        "decoder": {
            "vocabulary_size": 16, "hidden_size": 8, "max_position_embeddings": 32,
            "tie_word_embeddings": true,
            "final_norm": {"kind": "layer", "epsilon": 1e-5, "bias": true},
            "layers": [{
                "mixer": {"kind": "sliding_attention", "window": 3, "attention": {
                    "query_heads": 2, "key_value_heads": 1, "head_dim": 4,
                    "query_key_norm": true, "output_gate": false,
                    "position": {"kind": "t5_relative_bias", "buckets": 8, "max_distance": 16,
                        "bidirectional": false}}},
                "feed_forward": {"kind": "dense", "intermediate_size": 12,
                    "activation": "gelu_tanh", "bias": false},
                "mixer_norm": {"kind": "rms", "epsilon": 1e-6, "zero_centered": true},
                "feed_forward_norm": {"kind": "rms", "epsilon": 1e-6, "zero_centered": true},
                "norm_placement": "sandwich", "residual": "add"
            }]
        },
        "cross_attention_heads": 2,
        "cross_attention_head_dim": 4
    }"#;
    let spec: Seq2SeqSpec = serde_json::from_str(json).unwrap();
    spec.validate().unwrap();
    assert_eq!(spec.cross_attention_options, AttentionOptions::default());
    let MixerSpec::Mla(mla) = &spec.encoder.layers[0].mixer else {
        panic!("expected MLA");
    };
    assert_eq!(mla.score_scale, ScoreScale::Auto);
    assert_eq!(mla.connectivity, AttentionConnectivity::Global);
    let PositionEncodingSpec::Rope {
        scaling: RopeScaling::Yarn { truncate, .. },
        ..
    } = &mla.position
    else {
        panic!("expected YaRN");
    };
    assert!(*truncate);
    let FeedForwardSpec::MixtureOfExperts(moe) = &spec.encoder.layers[0].feed_forward else {
        panic!("expected experts");
    };
    assert_eq!(moe.routing, RoutingPolicy::default());
    assert!(!moe.shared_expert_gate && !moe.expert_bias);
    // The compact sliding form normalizes to explicit sliding connectivity.
    let (_, options) = spec.decoder.layers[0]
        .mixer
        .grouped_query_attention()
        .unwrap();
    assert_eq!(
        options.connectivity,
        AttentionConnectivity::SlidingWindow { window: 3 }
    );
    let reloaded: Seq2SeqSpec =
        serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
    assert_eq!(reloaded, spec);
}

/// Every new variant and option round-trips through JSON.
#[test]
fn explicit_options_and_variants_round_trip() {
    let mut layer = dense_layer();
    layer.mixer = MixerSpec::AttentionWithOptions {
        attention: AttentionSpec {
            output_gate: true,
            ..match dense_layer().mixer {
                MixerSpec::Attention(attention) => attention,
                _ => unreachable!(),
            }
        },
        options: AttentionOptions {
            connectivity: AttentionConnectivity::ChunkedLocal { chunk: 8 },
            score_scale: ScoreScale::Fixed { value: 0.125 },
            logit_softcap: Some(50.0),
            attention_sinks: false,
            projection_bias: ProjectionBias::query_key_value_only(),
            query_key_norm_scope: QueryKeyNormScope::FullWidth,
            output_gate_activation: OutputGateActivation::Silu,
        },
    };
    layer.feed_forward = FeedForwardSpec::ClampedSwiGlu(ClampedSwiGluSpec {
        intermediate_size: 16,
        bias: true,
        limit: 7.0,
        alpha: 1.702,
        up_offset: 1.0,
    });
    let mut spec = decoder();
    spec.layers = vec![layer.clone()];
    for scaling in [
        RopeScaling::Linear { factor: 2.0 },
        RopeScaling::DynamicNtk {
            factor: 2.0,
            original_max_positions: 16,
        },
        RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_positions: 16,
        },
        RopeScaling::LongRope {
            short_factors: vec![1.0; 4],
            long_factors: vec![2.0; 4],
            original_max_positions: 16,
            factor: 4.0,
            attention_factor: Some(1.2),
        },
    ] {
        let mut scaled = spec.clone();
        if let MixerSpec::AttentionWithOptions { attention, .. } = &mut scaled.layers[0].mixer {
            attention.position = PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: 8,
                scaling,
            };
        }
        scaled.validate().unwrap();
        let json = serde_json::to_string(&scaled).unwrap();
        assert_eq!(serde_json::from_str::<DecoderSpec>(&json).unwrap(), scaled);
    }
    let mut ungated = spec.clone();
    ungated.layers[0].feed_forward = FeedForwardSpec::Ungated(DenseFeedForwardSpec {
        intermediate_size: 4,
        activation: Activation::Gelu,
        bias: true,
    });
    let mut absent = spec;
    absent.layers[0].feed_forward = FeedForwardSpec::None;
    for spec in [ungated, absent] {
        spec.validate().unwrap();
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(serde_json::from_str::<DecoderSpec>(&json).unwrap(), spec);
    }
    // Missing option fields take their defaults.
    let partial: MixerSpec = serde_json::from_str(
        r#"{"kind": "attention_with_options", "attention": {"query_heads": 2,
            "key_value_heads": 2, "head_dim": 4, "query_key_norm": false,
            "output_gate": false, "position": {"kind": "none"}},
            "options": {"logit_softcap": 20.0}}"#,
    )
    .unwrap();
    let (_, options) = partial.grouped_query_attention().unwrap();
    assert_eq!(
        options,
        AttentionOptions {
            logit_softcap: Some(20.0),
            ..AttentionOptions::default()
        }
    );
}

/// Attention options reject malformed values and choices their compact flags would ignore.
#[test]
fn attention_options_report_each_invalid_field() {
    let attention = match dense_layer().mixer {
        MixerSpec::Attention(attention) => attention,
        _ => unreachable!(),
    };
    let check = |attention: &AttentionSpec, options: AttentionOptions, field: &str| {
        assert_configuration_field(
            MixerSpec::AttentionWithOptions {
                attention: attention.clone(),
                options,
            }
            .validate("mixer"),
            field,
        );
    };
    check(
        &attention,
        AttentionOptions {
            connectivity: AttentionConnectivity::SlidingWindow { window: 0 },
            ..AttentionOptions::default()
        },
        "mixer.attention.options.connectivity",
    );
    for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        check(
            &attention,
            AttentionOptions {
                score_scale: ScoreScale::Fixed { value },
                ..AttentionOptions::default()
            },
            "mixer.attention.options.score_scale.value",
        );
        check(
            &attention,
            AttentionOptions {
                logit_softcap: Some(value),
                ..AttentionOptions::default()
            },
            "mixer.attention.options.logit_softcap",
        );
    }
    let plain = AttentionSpec {
        query_key_norm: false,
        output_gate: false,
        ..attention.clone()
    };
    check(
        &plain,
        AttentionOptions {
            query_key_norm_scope: QueryKeyNormScope::FullWidth,
            ..AttentionOptions::default()
        },
        "mixer.attention.options.query_key_norm_scope",
    );
    check(
        &plain,
        AttentionOptions {
            output_gate_activation: OutputGateActivation::Silu,
            ..AttentionOptions::default()
        },
        "mixer.attention.options.output_gate_activation",
    );
}

/// Sinks with a soft cap fail at the options field with a spec-level reason (review API-1).
///
/// The shared kernel accepts the pair only under `EagerFp32Softmax`, which a
/// spec cannot select, so the error must name `options.attention_sinks` (not
/// the derived projection config's field) and must not suggest that
/// unreachable remedy. Grouped-query self-attention, standalone
/// cross-attention, and a sequence-to-sequence spec's sibling options field
/// all report it.
#[test]
fn sinks_with_a_soft_cap_are_rejected_at_the_options_field() {
    let attention = match dense_layer().mixer {
        MixerSpec::Attention(attention) => attention,
        _ => unreachable!(),
    };
    let options = AttentionOptions {
        attention_sinks: true,
        logit_softcap: Some(30.0),
        ..AttentionOptions::default()
    };
    let assert_spec_level = |result: Result<()>, expected: &str| match result
        .expect_err("sinks with a soft cap must be rejected")
    {
        Error::Configuration { field, reason } => {
            assert_eq!(field, expected);
            assert!(reason.contains("soft cap"), "{reason}");
            assert!(!reason.contains("EagerFp32Softmax"), "{reason}");
        }
        error => panic!("expected a configuration error at {expected}, got {error}"),
    };
    assert_spec_level(
        MixerSpec::AttentionWithOptions { attention, options }.validate("mixer"),
        "mixer.attention.options.attention_sinks",
    );
    assert_spec_level(
        CrossAttentionSpec {
            options,
            ..CrossAttentionSpec::new(4, 8)
        }
        .validate("cross_attention", 32),
        "cross_attention.options.attention_sinks",
    );
    assert_spec_level(
        Seq2SeqSpec {
            encoder: decoder(),
            decoder: decoder(),
            cross_attention_heads: 4,
            cross_attention_head_dim: 8,
            cross_attention_options: options,
        }
        .validate(),
        "cross_attention_options.attention_sinks",
    );
    for (sinks, softcap) in [(true, None), (false, Some(30.0))] {
        MixerSpec::AttentionWithOptions {
            attention: match dense_layer().mixer {
                MixerSpec::Attention(attention) => attention,
                _ => unreachable!(),
            },
            options: AttentionOptions {
                attention_sinks: sinks,
                logit_softcap: softcap,
                ..AttentionOptions::default()
            },
        }
        .validate("mixer")
        .expect("sinks alone and a soft cap alone stay valid");
    }
}

/// Feed-forward variants validate their own widths and clamp parameters.
#[test]
fn feed_forward_variants_report_each_invalid_field() {
    let clamped = |limit, alpha, intermediate_size| {
        FeedForwardSpec::ClampedSwiGlu(ClampedSwiGluSpec {
            intermediate_size,
            bias: false,
            limit,
            alpha,
            up_offset: 0.0,
        })
    };
    clamped(7.0, 1.702, 4).validate("ffn").unwrap();
    assert_configuration_field(
        clamped(7.0, 1.0, 0).validate("ffn"),
        "ffn.intermediate_size",
    );
    assert_configuration_field(clamped(0.0, 1.0, 4).validate("ffn"), "ffn.glu");
    assert_configuration_field(clamped(7.0, f64::NAN, 4).validate("ffn"), "ffn.glu");
    assert_configuration_field(
        FeedForwardSpec::Ungated(DenseFeedForwardSpec {
            intermediate_size: 0,
            activation: Activation::Relu,
            bias: false,
        })
        .validate("ffn"),
        "ffn.intermediate_size",
    );
    FeedForwardSpec::None.validate("ffn").unwrap();
    let json =
        r#"{"kind": "clamped_swi_glu", "intermediate_size": 4, "bias": false, "limit": 7.0}"#;
    let FeedForwardSpec::ClampedSwiGlu(defaults) = serde_json::from_str(json).unwrap() else {
        panic!("expected clamped SwiGLU");
    };
    assert_eq!((defaults.alpha, defaults.up_offset), (1.0, 0.0));
}

/// Cross-attention options must keep global connectivity and name their own fields.
#[test]
fn cross_attention_options_are_validated() {
    let seq2seq = |options| Seq2SeqSpec {
        encoder: decoder(),
        decoder: decoder(),
        cross_attention_heads: 4,
        cross_attention_head_dim: 8,
        cross_attention_options: options,
    };
    seq2seq(AttentionOptions {
        score_scale: ScoreScale::Unscaled,
        projection_bias: ProjectionBias::uniform(true),
        ..AttentionOptions::default()
    })
    .validate()
    .unwrap();
    assert_configuration_field(
        seq2seq(AttentionOptions {
            connectivity: AttentionConnectivity::SlidingWindow { window: 2 },
            ..AttentionOptions::default()
        })
        .validate(),
        "cross_attention_options.connectivity",
    );
    assert_configuration_field(
        seq2seq(AttentionOptions {
            query_key_norm_scope: QueryKeyNormScope::FullWidth,
            ..AttentionOptions::default()
        })
        .validate(),
        "cross_attention_options.query_key_norm_scope",
    );
    let standalone = CrossAttentionSpec::new(4, 8);
    assert_eq!(standalone.attention().key_value_heads, 4);
    standalone.validate("cross", 32).unwrap();
    assert_configuration_field(CrossAttentionSpec::new(4, 0).validate("cross", 32), "cross");
    assert_eq!(
        seq2seq(AttentionOptions::default()).cross_attention(),
        standalone
    );
}

/// Every RoPE scaling variant reaches its runtime policy, and errors name its fields.
#[test]
fn every_rope_scaling_variant_maps_to_its_runtime_policy() {
    let config = |scaling| {
        PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 8,
            scaling,
        }
        .rope_config(8)
    };
    let scaling_of = |scaling| config(scaling).unwrap().unwrap().scaling();
    assert!(matches!(
        scaling_of(RopeScaling::Linear { factor: 2.0 }),
        RuntimeRopeScaling::Linear(_)
    ));
    assert!(matches!(
        scaling_of(RopeScaling::DynamicNtk {
            factor: 2.0,
            original_max_positions: 16
        }),
        RuntimeRopeScaling::DynamicNtk(_)
    ));
    assert!(matches!(
        scaling_of(RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_positions: 16
        }),
        RuntimeRopeScaling::Llama3(_)
    ));
    assert!(matches!(
        scaling_of(RopeScaling::LongRope {
            short_factors: vec![1.0; 4],
            long_factors: vec![2.0; 4],
            original_max_positions: 16,
            factor: 4.0,
            attention_factor: None
        }),
        RuntimeRopeScaling::LongRope(_)
    ));
    assert!(matches!(
        scaling_of(RopeScaling::Proportional {
            rotated_fraction: 0.5,
            factor: 1.0
        }),
        RuntimeRopeScaling::Proportional(_)
    ));
    let RuntimeRopeScaling::Yarn(yarn) = scaling_of(RopeScaling::Yarn {
        factor: 2.0,
        original_max_positions: 16,
        attention_factor: None,
        beta_slow: 1.0,
        beta_fast: 32.0,
        truncate: false,
    }) else {
        panic!("expected YaRN");
    };
    assert!(!yarn.truncate());
    let field = |scaling| match config(scaling).unwrap_err() {
        Error::Configuration { field, .. } => field,
        other => panic!("unexpected {other}"),
    };
    assert_eq!(
        field(RopeScaling::Linear { factor: 0.5 }),
        "position.scaling.factor"
    );
    assert_eq!(
        field(RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 2.0,
            high_freq_factor: 2.0,
            original_max_positions: 16
        }),
        "position.scaling.frequency_factors"
    );
    assert_eq!(
        field(RopeScaling::LongRope {
            short_factors: vec![1.0; 3],
            long_factors: vec![1.0; 4],
            original_max_positions: 16,
            factor: 4.0,
            attention_factor: None
        }),
        "position.scaling.factors"
    );
    assert_eq!(
        field(RopeScaling::Proportional {
            rotated_fraction: 2.0,
            factor: 1.0
        }),
        "position.scaling.rotated_fraction"
    );
}

/// The block's mixer normalization reaches validation, so its epsilon error names that norm.
#[test]
fn mixer_validation_uses_the_block_normalization() {
    let mut spec = decoder();
    spec.layers[0].mixer_norm = NormalizationSpec::Rms {
        epsilon: f64::NAN,
        zero_centered: true,
    };
    assert_configuration_field(spec.validate(), "layers[0].mixer_norm.epsilon");
    let mut zero_centered = decoder();
    zero_centered.layers[0].mixer_norm = NormalizationSpec::Rms {
        epsilon: 1e-5,
        zero_centered: true,
    };
    zero_centered.validate().unwrap();
    let (attention, options) = zero_centered.layers[0]
        .mixer
        .grouped_query_attention()
        .unwrap();
    let config = attention.grouped_query_config(32, &options, zero_centered.layers[0].mixer_norm);
    assert_eq!(config.query_key_norm_epsilon, Some(1e-5));
    assert!(config.query_key_norm_zero_centered);
    assert!(
        MixerSpec::DeltaNet {
            key_heads: 1,
            value_heads: 1,
            key_head_dim: 4,
            value_head_dim: 4,
            convolution_kernel: 2,
        }
        .is_recurrent()
    );
    assert_eq!(
        dense_layer().mixer.connectivity(),
        Some(AttentionConnectivity::Global)
    );
}

/// Differential attention validates head pairing, its shared config, and a bias-free position encoding.
#[test]
fn differential_attention_specs_are_validated() {
    let valid = DifferentialAttentionSpec {
        query_heads: 4,
        key_value_heads: 2,
        head_dim: 8,
        lambda_init: 0.2,
        position: PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 8,
            scaling: RopeScaling::None,
        },
        connectivity: AttentionConnectivity::Global,
        bias: false,
    };
    let mut spec = decoder();
    spec.layers[0].mixer = MixerSpec::DifferentialAttention(valid.clone());
    spec.validate().unwrap();
    let json = serde_json::to_string(&spec).unwrap();
    assert_eq!(serde_json::from_str::<DecoderSpec>(&json).unwrap(), spec);
    assert_eq!(
        spec.layers[0].mixer.connectivity(),
        Some(AttentionConnectivity::Global)
    );
    let check = |differential: DifferentialAttentionSpec, field: &str| {
        assert_configuration_field(
            MixerSpec::DifferentialAttention(differential).validate("mixer"),
            field,
        );
    };
    check(
        DifferentialAttentionSpec {
            query_heads: 3,
            key_value_heads: 1,
            ..valid.clone()
        },
        "mixer.differential_attention.query_heads",
    );
    check(
        DifferentialAttentionSpec {
            position: PositionEncodingSpec::T5RelativeBias {
                buckets: 8,
                max_distance: 16,
                bidirectional: false,
            },
            ..valid.clone()
        },
        "mixer.position",
    );
    check(
        DifferentialAttentionSpec {
            connectivity: AttentionConnectivity::ChunkedLocal { chunk: 0 },
            ..valid.clone()
        },
        "mixer.connectivity",
    );
    check(
        DifferentialAttentionSpec {
            position: PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: 10,
                scaling: RopeScaling::None,
            },
            ..valid
        },
        "mixer.position.rotary_dim",
    );
}
