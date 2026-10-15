//! Configuration parsing, token validation, and rotary shape boundary tests.

use super::*;

/// Minimal topology that isolates wrapper parsing from model-family validation.
#[derive(Debug, Deserialize)]
struct TinyConfig {
    /// Positive residual width required by the synthetic topology.
    width: usize,
}

impl Validate for TinyConfig {
    /// Rejects the zero-width synthetic topology used by parser tests.
    fn validate(&self) -> Result<(), ConfigError> {
        nonzero("tiny", "width", self.width)
    }
}

/// Each supported wrapper alias selects its nested text topology.
#[test]
fn extracts_supported_text_wrapper_aliases() {
    for key in ["text_config", "language_config", "llm_config"] {
        let json = format!(r#"{{"{key}":{{"width":4}},"vision_config":{{}}}}"#);
        let config: TinyConfig = parse_text_config(&json).unwrap();
        assert_eq!(config.width, 4);
    }
}

/// Bare legacy infinity becomes null while quoted Unicode metadata stays exact.
#[test]
fn positive_infinity_is_normalized_only_outside_strings() {
    let json = r#"{"width":4,"limit":[0.0,Infinity],"note":"Infinity – 無限"}"#;
    let normalized = normalize_positive_infinity(json);
    let value: serde_json::Value = serde_json::from_str(&normalized).unwrap();
    assert!(value["limit"][1].is_null());
    assert_eq!(value["note"], "Infinity – 無限");
}

/// Empty inputs, escaped quotes, backslashes, and Unicode preserve scanner boundaries.
#[test]
fn infinity_normalization_preserves_utf8_and_escaped_string_boundaries() {
    for (input, expected) in [
        ("", ""),
        ("Infinity", "null"),
        ("[Infinity, Infinity]", "[null, null]"),
        (
            r#"["escaped \" Infinity",Infinity]"#,
            r#"["escaped \" Infinity",null]"#,
        ),
        (r#"["slash \\",Infinity]"#, r#"["slash \\",null]"#),
        (r#"["無限 🦀",Infinity,"é"]"#, r#"["無限 🦀",null,"é"]"#),
        (r#""unfinished \" Infinity"#, r#""unfinished \" Infinity"#),
    ] {
        assert_eq!(normalize_positive_infinity(input), expected, "{input:?}");
    }
}

/// Normalization does not conceal malformed JSON or invalid topology fields.
#[test]
fn configuration_loading_preserves_parser_and_validation_failures() {
    for json in [
        "",
        "{",
        r#"{"width":InfinitySuffix}"#,
        r#"{"width":-Infinity}"#,
    ] {
        assert!(
            matches!(
                parse_text_config::<TinyConfig>(json),
                Err(ConfigLoadError::Json(_))
            ),
            "malformed JSON must remain invalid: {json:?}"
        );
    }
    assert!(matches!(
        parse_text_config::<TinyConfig>(r#"{"width":0}"#),
        Err(ConfigLoadError::Validation(_))
    ));
    let direct = parse_text_config::<TinyConfig>(r#"{"width":7,"text_config":null}"#)
        .expect("null wrappers must preserve direct config parsing");
    assert_eq!(direct.width, 7);
}

/// A partial rotary factor producing an odd effective width is rejected.
#[test]
fn invalid_effective_rope_width_is_rejected() {
    let rope = RopeParameters {
        partial_rotary_factor: Some(0.3),
        ..RopeParameters::default()
    };
    assert!(rope.validate_for("test", 8, 128, None).is_err());
}

/// Multimodal rotary sections sum to the rotated complex-pair count.
#[test]
fn multimodal_rope_sections_count_complex_pairs() {
    let rope = RopeParameters {
        partial_rotary_factor: Some(0.5),
        mrope_section: Some(vec![1, 1]),
        ..RopeParameters::default()
    };
    rope.validate_for("test", 8, 128, None).unwrap();
}

/// Scaling variants reject missing or inconsistent algorithm-specific parameters.
#[test]
fn scaled_rope_variants_require_their_shape_fields() {
    let linear = RopeParameters {
        rope_type: RopeType::Linear,
        ..RopeParameters::default()
    };
    assert!(linear.validate_for("test", 8, 128, None).is_err());

    let yarn = RopeParameters {
        rope_type: RopeType::Yarn,
        factor: Some(4.0),
        original_max_position_embeddings: Some(256),
        beta_fast: Some(1.0),
        beta_slow: Some(2.0),
        ..RopeParameters::default()
    };
    assert!(yarn.validate_for("test", 8, 128, None).is_err());

    let llama3 = RopeParameters {
        rope_type: RopeType::Llama3,
        factor: Some(8.0),
        original_max_position_embeddings: Some(64),
        low_freq_factor: Some(1.0),
        high_freq_factor: Some(4.0),
        ..RopeParameters::default()
    };
    llama3.validate_for("test", 8, 128, None).unwrap();
}

/// A single YaRN beta override is validated together with the other beta's default.
#[test]
fn yarn_validates_one_sided_beta_overrides_against_defaults() {
    let base = RopeParameters {
        rope_type: RopeType::Yarn,
        factor: Some(4.0),
        original_max_position_embeddings: Some(64),
        ..RopeParameters::default()
    };

    let mut invalid_fast = base.clone();
    invalid_fast.beta_fast = Some(-1.0);
    assert!(invalid_fast.validate_for("test", 8, 128, None).is_err());

    let mut invalid_slow = base.clone();
    invalid_slow.beta_slow = Some(32.0);
    assert!(invalid_slow.validate_for("test", 8, 128, None).is_err());

    let mut valid_fast = base.clone();
    valid_fast.beta_fast = Some(24.0);
    valid_fast.validate_for("test", 8, 128, None).unwrap();

    let mut valid_slow = base;
    valid_slow.beta_slow = Some(2.0);
    valid_slow.validate_for("test", 8, 128, None).unwrap();
}

/// LongRoPE factors have one value per rotated complex pair.
#[test]
fn longrope_factor_vectors_match_complex_frequency_count() {
    let valid = RopeParameters {
        rope_type: RopeType::LongRope,
        original_max_position_embeddings: Some(64),
        short_factor: Some(vec![1.0; 4]),
        long_factor: Some(vec![2.0; 4]),
        ..RopeParameters::default()
    };
    valid.validate_for("test", 8, 128, None).unwrap();

    let mut invalid = valid.clone();
    invalid.long_factor = Some(vec![2.0; 3]);
    assert!(invalid.validate_for("test", 8, 128, None).is_err());

    // Factors divide `f32` frequencies: a subnormal or overflowing value is
    // rejected (ROPE-F32-2), the smallest normal one is accepted.
    for factor in [1e-45, 1e-39, 1e39, 0.0, f64::NAN] {
        let mut tiny = valid.clone();
        tiny.short_factor = Some(vec![factor, 1.0, 1.0, 1.0]);
        let error = tiny.validate_for("test", 8, 128, None).unwrap_err();
        assert_eq!(error.field, "short_factor", "{factor}");
        assert!(error.reason.contains("normal positive f32"), "{error}");
    }
    let mut smallest = valid;
    smallest.short_factor = Some(vec![f64::from(f32::MIN_POSITIVE), 1.0, 1.0, 1.0]);
    smallest.validate_for("test", 8, 128, None).unwrap();
}

/// Bases below one are rejected wherever the base comes from (ROPE-F32-1).
///
/// `theta^(-2i/d)` grows with `i` for a base below one and can overflow the
/// `f32` frequency table; the shared translator rejects the same bases.
#[test]
fn rope_bases_below_one_are_rejected() {
    let nested = |theta: f64| RopeParameters {
        rope_theta: Some(theta),
        ..RopeParameters::default()
    };
    for theta in [
        1e-300,
        0.5,
        1.0 - f64::EPSILON,
        0.0,
        -1.0,
        f64::INFINITY,
        f64::NAN,
    ] {
        let error = nested(theta)
            .validate_for("test", 8, 128, None)
            .unwrap_err();
        assert_eq!(error.field, "rope_theta", "{theta}");
        assert!(error.reason.contains("at least one"), "{error}");
        let fallback = RopeParameters::default()
            .validate_for("test", 8, 128, Some(theta))
            .unwrap_err();
        assert_eq!(fallback.field, "rope_theta", "{theta}");
    }
    for theta in [1.0, 10_000.0, f64::MAX] {
        nested(theta).validate_for("test", 8, 128, None).unwrap();
    }
}

/// Token lists reject emptiness and out-of-vocabulary IDs while accepting valid IDs.
#[test]
fn token_lists_must_be_nonempty_and_inside_the_vocabulary() {
    assert!(token_ids("test", "eos", Some(&TokenIds::Many(vec![])), 4).is_err());
    assert!(token_ids("test", "eos", Some(&TokenIds::Many(vec![1, 4])), 4).is_err());
    token_ids("test", "eos", Some(&TokenIds::Many(vec![1, 3])), 4).unwrap();
}

// Current and legacy RoPE type keys.

/// Parses a `rope_scaling`-style object into the shared RoPE parameter schema.
fn rope_from(json: &str) -> Result<RopeParameters, serde_json::Error> {
    serde_json::from_str(json)
}

/// Either key alone, both equal keys, or neither key select the expected algorithm.
#[test]
fn rope_type_accepts_current_legacy_or_both_agreeing_keys() {
    let cases = [
        (r#"{"type": "yarn", "factor": 4.0}"#, RopeType::Yarn),
        (r#"{"rope_type": "yarn", "factor": 4.0}"#, RopeType::Yarn),
        (
            r#"{"type": "yarn", "rope_type": "yarn", "factor": 4.0}"#,
            RopeType::Yarn,
        ),
        (
            r#"{"rope_type": "linear", "type": "linear", "factor": 2.0}"#,
            RopeType::Linear,
        ),
        (r#"{"factor": 4.0}"#, RopeType::Default),
    ];
    for (json, expected) in cases {
        let rope = rope_from(json).unwrap_or_else(|error| panic!("{json}: {error}"));
        assert_eq!(rope.rope_type, expected, "{json}");
        assert_eq!(
            rope.factor,
            Some(if expected == RopeType::Linear {
                2.0
            } else {
                4.0
            })
        );
    }
}

/// Disagreeing keys are rejected with both serialized names in the message.
#[test]
fn conflicting_rope_type_keys_are_rejected_with_both_values() {
    let error = rope_from(r#"{"type": "linear", "rope_type": "yarn", "factor": 2.0}"#)
        .expect_err("ambiguous algorithms must not be resolved silently");
    let message = error.to_string();
    assert!(message.contains(r#"rope_type "yarn""#), "{message}");
    assert!(message.contains(r#"legacy type "linear""#), "{message}");
}

/// Serialization emits only the current key and round-trips every algorithm.
#[test]
fn rope_parameters_serialize_only_the_current_type_key() {
    for rope_type in [
        RopeType::Default,
        RopeType::Linear,
        RopeType::Dynamic,
        RopeType::Yarn,
        RopeType::LongRope,
        RopeType::Llama3,
        RopeType::Proportional,
    ] {
        assert_eq!(
            serde_json::to_string(&rope_type).unwrap(),
            format!("\"{}\"", rope_type.as_str())
        );
        let rope = RopeParameters {
            rope_type,
            factor: Some(3.0),
            mrope_section: Some(vec![1, 2, 3]),
            ..RopeParameters::default()
        };
        let value = serde_json::to_value(&rope).unwrap();
        assert!(value.get("type").is_none());
        assert_eq!(value["rope_type"], rope_type.as_str());
        assert_eq!(
            serde_json::from_value::<RopeParameters>(value).unwrap(),
            rope
        );
    }
}

/// A DeepSeek-V2 config re-saved by Hugging Face, carrying both keys, still loads.
#[test]
fn resaved_deepseek_config_with_both_rope_keys_loads() {
    use crate::models::deepseek_v2::DeepseekV2Config;
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/deepseek_v2_lite.json")).unwrap();
    let legacy = value["rope_scaling"]["type"].clone();
    assert_eq!(legacy, "yarn");
    value["rope_scaling"]["rope_type"] = legacy;
    let resaved = DeepseekV2Config::from_json_str(&value.to_string())
        .expect("transformers save_pretrained output must parse");
    let original =
        DeepseekV2Config::from_json_str(include_str!("../../tests/fixtures/deepseek_v2_lite.json"))
            .unwrap();
    assert_eq!(
        resaved.rope_scaling.as_ref().unwrap().rope_type,
        RopeType::Yarn
    );
    assert_eq!(resaved.rope_scaling, original.rope_scaling);

    value["rope_scaling"]["rope_type"] = "linear".into();
    let error = DeepseekV2Config::from_json_str(&value.to_string()).unwrap_err();
    assert!(error.to_string().contains("conflicts"), "{error}");
}

// Conversion into the crate error type.

/// Loader failures convert into crate errors that keep their field path or parser location.
#[test]
fn configuration_load_errors_convert_with_field_paths_and_parser_locations() {
    let validation: crate::Error =
        ConfigLoadError::from(ConfigError::new("tiny", "width", "must be nonzero")).into();
    assert!(matches!(
        &validation,
        crate::Error::Configuration { field, reason }
            if field == "tiny.width" && reason == "must be nonzero"
    ));

    let parse = parse_text_config::<TinyConfig>("{\n  \"width\": }").unwrap_err();
    let converted: crate::Error = parse.into();
    match converted {
        crate::Error::Configuration { field, reason } => {
            assert_eq!(field, "config");
            assert!(reason.starts_with("invalid model config JSON"), "{reason}");
            assert!(reason.contains("line 2"), "{reason}");
        }
        other => panic!("unexpected error {other:?}"),
    }

    let loaded = (|| -> crate::Result<TinyConfig> { Ok(parse_text_config(r#"{"width": 0}"#)?) })();
    assert!(loaded.unwrap_err().to_string().contains("tiny.width"));
}

/// Exhaustively checks key reconciliation over every pair of present or absent algorithms.
#[test]
fn rope_type_resolution_is_exhaustively_consistent() {
    let algorithms = [
        RopeType::Default,
        RopeType::Linear,
        RopeType::Dynamic,
        RopeType::Yarn,
        RopeType::LongRope,
        RopeType::Llama3,
        RopeType::Proportional,
    ];
    let options = std::iter::once(None)
        .chain(algorithms.into_iter().map(Some))
        .collect::<Vec<_>>();
    for &current in &options {
        for &legacy in &options {
            let resolved = resolve_rope_type(current, legacy);
            match (current, legacy) {
                (Some(current), Some(legacy)) if current != legacy => {
                    assert!(resolved.is_err(), "{current:?}/{legacy:?}")
                }
                _ => assert_eq!(
                    resolved.unwrap(),
                    current.or(legacy).unwrap_or(RopeType::Default)
                ),
            }
        }
    }
}

/// Proportional RoPE validates a pair fraction over the whole head, not a prefix width.
///
/// Transformers 5.8.0 rotates `floor(fraction * head_dim / 2)` leading pairs
/// and defaults the optional `factor` to one, so a non-integral product and an
/// absent factor are valid; an odd head width or an extension factor below one
/// is not.
#[test]
fn proportional_rope_validates_a_whole_head_pair_fraction() {
    let proportional = |partial_rotary_factor, factor| RopeParameters {
        rope_type: RopeType::Proportional,
        partial_rotary_factor,
        factor,
        ..RopeParameters::default()
    };
    proportional(Some(0.25), None)
        .validate_for("test", 16, 128, Some(1.0e6))
        .unwrap();
    proportional(Some(0.6), Some(2.0))
        .validate_for("test", 16, 128, None)
        .unwrap();
    proportional(None, None)
        .validate_for("test", 16, 128, None)
        .unwrap();
    let odd = proportional(Some(0.5), None)
        .validate_for("test", 15, 128, None)
        .unwrap_err();
    assert_eq!(odd.field, "head_dim");
    let small = proportional(Some(0.5), Some(0.5))
        .validate_for("test", 16, 128, None)
        .unwrap_err();
    assert_eq!(small.field, "rope.factor");
    let wide = proportional(Some(1.5), None)
        .validate_for("test", 16, 128, None)
        .unwrap_err();
    assert_eq!(wide.field, "partial_rotary_factor");
    let sections = RopeParameters {
        mrope_section: Some(vec![4, 4]),
        ..proportional(Some(0.25), None)
    };
    sections.validate_for("test", 16, 128, None).unwrap();
}

/// Every validated proportional configuration also translates through the shared path.
#[test]
fn validated_proportional_parameters_translate() {
    let rope = RopeParameters {
        rope_type: RopeType::Proportional,
        partial_rotary_factor: Some(0.6),
        ..RopeParameters::default()
    };
    rope.validate_for("test", 16, 128, None).unwrap();
    let config = rope
        .to_rope_config(&RopeContext::new("test", 16, 128))
        .unwrap();
    assert_eq!(config.rotary_dimension(), 16);
}

/// `validate_for` and the shared translator accept exactly the same proportional configs.
///
/// Both follow Transformers 5.8.0, which reads `partial_rotary_factor` as a
/// pair fraction in `[0, 1]` (zero leaves every pair unrotated), defaults
/// `factor` to one, and needs an even head width. Only acceptance is
/// compared: the validators check fields in different orders.
#[test]
fn proportional_validators_accept_the_same_configs() {
    let fractions = [
        None,
        Some(-0.1),
        Some(0.0),
        Some(0.25),
        Some(0.3),
        Some(0.6),
        Some(1.0),
        Some(1.5),
        Some(f64::NAN),
        Some(f64::INFINITY),
    ];
    let factors = [None, Some(0.5), Some(1.0), Some(2.0), Some(f64::NAN)];
    let thetas = [None, Some(-1.0), Some(10_000.0)];
    let (mut accepted, mut rejected) = (0, 0);
    for head_dim in [1, 7, 8, 16] {
        for fraction in fractions {
            for factor in factors {
                for theta in thetas {
                    let rope = RopeParameters {
                        rope_type: RopeType::Proportional,
                        partial_rotary_factor: fraction,
                        factor,
                        rope_theta: theta,
                        ..RopeParameters::default()
                    };
                    let validated = rope.validate_for("test", head_dim, 128, None);
                    let translated = rope.to_rope_config(&RopeContext::new("test", head_dim, 128));
                    assert_eq!(
                        validated.is_ok(),
                        translated.is_ok(),
                        "head {head_dim}, fraction {fraction:?}, factor {factor:?}, theta {theta:?}: {validated:?} vs {translated:?}"
                    );
                    if validated.is_ok() {
                        accepted += 1;
                    } else {
                        rejected += 1;
                    }
                }
            }
        }
    }
    // Even heads x {None, 0, 0.25, 0.3, 0.6, 1} x {None, 1, 2} x {None, 1e4}.
    assert_eq!(accepted, 2 * 6 * 3 * 2);
    assert_eq!(accepted + rejected, 4 * 10 * 5 * 3);

    // A zero fraction is all-NoPE in both validators.
    let nope = RopeParameters {
        rope_type: RopeType::Proportional,
        partial_rotary_factor: Some(0.0),
        ..RopeParameters::default()
    };
    nope.validate_for("test", 16, 128, None).unwrap();
    let config = nope
        .to_rope_config(&RopeContext::new("test", 16, 128))
        .unwrap();
    assert_eq!(config.scaling().amplitude(), 1.0);
    // Other algorithms still need a nonzero rotated prefix.
    let error = RopeParameters {
        partial_rotary_factor: Some(0.0),
        ..RopeParameters::default()
    }
    .validate_for("test", 16, 128, None)
    .unwrap_err();
    assert_eq!(error.field, "partial_rotary_factor");
    assert!(error.reason.contains("(0, 1]"), "{error}");
}

/// Every `validate_for` rejection names the field that violates its invariant.
#[test]
fn rope_validation_rejections_name_their_fields() {
    let field_of = |rope: RopeParameters, head_dim: usize| {
        rope.validate_for("test", head_dim, 128, None)
            .expect_err("validation must fail")
            .field
    };
    // An integral but odd prefix cannot be split into channel pairs.
    let odd = RopeParameters {
        partial_rotary_factor: Some(0.5),
        ..RopeParameters::default()
    };
    assert_eq!(field_of(odd, 6), "partial_rotary_factor");
    for sections in [vec![], vec![2, 0, 2], vec![1, 1, 1]] {
        let rope = RopeParameters {
            mrope_section: Some(sections),
            ..RopeParameters::default()
        };
        assert_eq!(field_of(rope, 8), "mrope_section");
    }
    let longrope = RopeParameters {
        rope_type: RopeType::LongRope,
        original_max_position_embeddings: Some(64),
        long_factor: Some(vec![1.0; 4]),
        ..RopeParameters::default()
    };
    assert_eq!(field_of(longrope, 8), "short_factor");
    let llama3 = RopeParameters {
        rope_type: RopeType::Llama3,
        factor: Some(8.0),
        original_max_position_embeddings: Some(64),
        ..RopeParameters::default()
    };
    assert_eq!(field_of(llama3.clone(), 8), "low_freq_factor");
    let low_only = RopeParameters {
        low_freq_factor: Some(1.0),
        ..llama3.clone()
    };
    assert_eq!(field_of(low_only, 8), "high_freq_factor");
    let inverted = RopeParameters {
        low_freq_factor: Some(4.0),
        high_freq_factor: Some(1.0),
        ..llama3.clone()
    };
    assert_eq!(field_of(inverted, 8), "high_freq_factor");
    let missing_original = RopeParameters {
        original_max_position_embeddings: None,
        ..llama3.clone()
    };
    assert_eq!(
        field_of(missing_original, 8),
        "rope.original_max_position_embeddings"
    );
    let too_long = RopeParameters {
        original_max_position_embeddings: Some(256),
        ..llama3.clone()
    };
    assert_eq!(
        field_of(too_long, 8),
        "rope.original_max_position_embeddings"
    );
    let zero_original = RopeParameters {
        original_max_position_embeddings: Some(0),
        ..llama3
    };
    assert_eq!(
        field_of(zero_original, 8),
        "rope.original_max_position_embeddings"
    );
    // A rotated prefix needs a finite factor in (0, 1].
    for invalid in [f64::NAN, f64::INFINITY, 1.5, -0.25] {
        let rope = RopeParameters {
            partial_rotary_factor: Some(invalid),
            ..RopeParameters::default()
        };
        assert_eq!(field_of(rope, 8), "partial_rotary_factor", "{invalid}");
    }
    // LongRoPE tables need one positive finite factor per rotated pair.
    let tables = |short_factor: Vec<f64>| RopeParameters {
        rope_type: RopeType::LongRope,
        original_max_position_embeddings: Some(64),
        short_factor: Some(short_factor),
        long_factor: Some(vec![1.0; 4]),
        ..RopeParameters::default()
    };
    tables(vec![1.0; 4])
        .validate_for("test", 8, 128, None)
        .unwrap();
    for short_factor in [
        vec![1.0; 3],
        vec![1.0, 1.0, f64::NAN, 1.0],
        vec![1.0, 0.0, 1.0, 1.0],
    ] {
        let label = format!("{short_factor:?}");
        assert_eq!(field_of(tables(short_factor), 8), "short_factor", "{label}");
    }
}

/// Shared scalar validators reject out-of-range values with field-specific messages.
#[test]
fn scalar_validators_reject_out_of_range_values() {
    assert_eq!(
        probability("test", "dropout", 1.5).unwrap_err().reason,
        "must be finite and in [0, 1]"
    );
    probability("test", "dropout", 0.0).unwrap();
    for invalid in [f64::NAN, f64::INFINITY, -0.1] {
        assert!(
            probability("test", "dropout", invalid).is_err(),
            "{invalid}"
        );
    }
    let error = divisible("test", "heads", 6, "groups", 4).unwrap_err();
    assert_eq!(
        (error.field, error.reason.as_str()),
        ("heads", "must be divisible by groups")
    );
    divisible("test", "heads", 8, "groups", 4).unwrap();
    let error = model_type("test", "llama", &["qwen3"]).unwrap_err();
    assert_eq!(error.field, "model_type");
    assert!(error.reason.contains("\"llama\""), "{}", error.reason);
    assert!(positive_finite("test", "theta", f64::INFINITY).is_err());
}

/// Loader errors expose their source and format; topology checks back the runtime trait.
#[test]
fn load_errors_display_sources_and_back_architecture_validation() {
    let topology = ConfigError::new("tiny", "width", "must be nonzero");
    assert_eq!(topology.to_string(), "tiny.width: must be nonzero");
    let validation = ConfigLoadError::from(topology.clone());
    assert_eq!(validation.to_string(), "tiny.width: must be nonzero");
    assert_eq!(
        validation.source().map(ToString::to_string),
        Some(topology.to_string())
    );
    let json = ConfigLoadError::from(serde_json::from_str::<serde_json::Value>("{").unwrap_err());
    assert!(json.to_string().starts_with("invalid model config JSON"));
    assert!(json.source().is_some());

    use crate::ArchitectureConfig;
    let invalid = crate::models::qwen3_moe::Qwen3MoeConfig {
        head_dim: 0,
        ..crate::models::qwen3_moe::Qwen3MoeConfig::default()
    };
    let error = ArchitectureConfig::validate(&invalid).unwrap_err();
    assert!(error.to_string().contains("qwen3_moe.head_dim"), "{error}");
    ArchitectureConfig::validate(&crate::models::qwen3_moe::Qwen3MoeConfig::default()).unwrap();
}
