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

    let mut invalid = valid;
    invalid.long_factor = Some(vec![2.0; 3]);
    assert!(invalid.validate_for("test", 8, 128, None).is_err());
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
