//! Unit, family-parity, and property tests for the shared RoPE translator.
//!
//! Golden comparisons against Transformers live in `rope_golden_tests.rs`;
//! this file covers precedence, strictness, error paths, per-layer mappings,
//! and parity with the private family translators the shared one replaces.

use proptest::prelude::*;

use super::*;
use crate::models::config::ModelConfig;
use crate::models::deepseek_v2::DeepseekV2Config;
use crate::models::muse_glimmer::MuseGlimmerTextConfig;
use crate::models::qwen3_5::Qwen3_5TextConfig;
use crate::models::qwen3_moe::Qwen3MoeConfig;

/// Head width used by the synthetic contexts.
const HEAD: usize = 16;
/// Maximum context used by the synthetic contexts.
const MAX: usize = 128;

/// Returns a strict synthetic context with a 16-channel head and 128 positions.
fn context() -> RopeContext {
    RopeContext::new("test", HEAD, MAX)
}

/// Parses a serialized `rope_parameters` object.
fn params(json: &str) -> RopeParameters {
    serde_json::from_str(json).unwrap_or_else(|error| panic!("{json}: {error}"))
}

/// Returns the dotted field path of a configuration error.
fn field(error: &Error) -> &str {
    match error {
        Error::Configuration { field, .. } => field,
        other => panic!("expected a configuration error, got {other:?}"),
    }
}

/// Translates and expects a configuration error at `expected_field`.
fn rejects(parameters: &RopeParameters, context: &RopeContext, expected_field: &str) -> Error {
    let error = parameters
        .to_rope_config(context)
        .expect_err("translation must fail");
    assert_eq!(field(&error), expected_field, "{error}");
    error
}

/// Translates one layer type and expects an error at `expected_field`.
fn rejects_layer(
    set: &RopeParameterSet,
    layer_type: &str,
    context: &RopeContext,
    expected_field: &str,
) {
    let error = set
        .to_rope_config(layer_type, context)
        .expect_err("translation must fail");
    assert_eq!(field(&error), expected_field, "{error}");
}

/// Returns the YaRN policy of a configuration.
fn yarn_of(config: RopeConfig) -> YarnConfig {
    match config.scaling() {
        RopeScaling::Yarn(yarn) => yarn,
        other => panic!("expected YaRN, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Context and parsing.
// ---------------------------------------------------------------------------

/// The builder records every option and starts from Transformers defaults.
#[test]
fn context_builder_records_every_option() {
    let base = context();
    assert_eq!(base.architecture(), "test");
    assert_eq!(base.head_dim(), HEAD);
    assert_eq!(base.max_position_embeddings(), MAX);
    assert_eq!(base.rope_theta, None);
    assert_eq!(base.default_rope_theta, DEFAULT_ROPE_THETA);
    assert_eq!(base.layer_rope_theta, None);
    assert_eq!(base.partial_rotary_factor, None);
    assert_eq!(base.original_max_position_embeddings, None);
    assert_eq!(base.partial_rotary, PartialRotary::Prefix);
    assert_eq!(base.mscale_convention, YarnMscaleConvention::HuggingFace);
    assert_eq!(base.softmax_mscale, SoftmaxMscale::Unscaled);
    assert_eq!(
        base.dynamic_ntk_context,
        DynamicNtkContext::MaxPositionEmbeddings
    );
    assert_eq!(base.mrope, MropeHandling::Reject);
    assert_eq!(base.unused_fields, UnusedRopeFields::Reject);

    let full = base
        .with_rope_theta(Some(5.0))
        .with_default_rope_theta(6.0)
        .with_layer_rope_theta(7.0)
        .with_partial_rotary_factor(Some(0.5))
        .with_original_max_position_embeddings(Some(64))
        .with_partial_rotary(PartialRotary::FullHeadOnly)
        .with_yarn_mscale_convention(YarnMscaleConvention::DeepSeekReference)
        .with_softmax_mscale(SoftmaxMscale::MscaleAllDimSquared)
        .with_dynamic_ntk_context(DynamicNtkContext::OriginalMaxPositionEmbeddings)
        .with_mrope(MropeHandling::TextPositions)
        .with_unused_fields(UnusedRopeFields::Ignore);
    assert_eq!(full.rope_theta, Some(5.0));
    assert_eq!(full.default_rope_theta, 6.0);
    assert_eq!(full.layer_rope_theta, Some(7.0));
    assert_eq!(full.partial_rotary_factor, Some(0.5));
    assert_eq!(full.original_max_position_embeddings, Some(64));
    assert_eq!(full.partial_rotary, PartialRotary::FullHeadOnly);
    assert_eq!(
        full.mscale_convention,
        YarnMscaleConvention::DeepSeekReference
    );
    assert_eq!(full.softmax_mscale, SoftmaxMscale::MscaleAllDimSquared);
    assert_eq!(
        full.dynamic_ntk_context,
        DynamicNtkContext::OriginalMaxPositionEmbeddings
    );
    assert_eq!(full.mrope, MropeHandling::TextPositions);
    assert_eq!(full.unused_fields, UnusedRopeFields::Ignore);
}

/// `truncate` is parsed, defaults to absent, and round-trips through serialization.
#[test]
fn truncate_is_parsed_and_round_trips() {
    let rope = params(r#"{"rope_type": "yarn", "factor": 4.0, "truncate": false}"#);
    assert_eq!(rope.truncate, Some(false));
    assert_eq!(params(r#"{"rope_type": "yarn"}"#).truncate, None);
    let value = serde_json::to_value(&rope).unwrap();
    assert_eq!(value["truncate"], false);
    assert_eq!(
        serde_json::from_value::<RopeParameters>(value).unwrap(),
        rope
    );
}

/// Flat, empty, and per-layer-type objects select the matching shape.
#[test]
fn parameter_sets_detect_the_mapping_shape() {
    let flat: RopeParameterSet =
        serde_json::from_str(r#"{"type": "linear", "factor": 2.0}"#).unwrap();
    assert_eq!(
        flat,
        RopeParameterSet::Uniform(params(r#"{"rope_type": "linear", "factor": 2.0}"#))
    );
    let empty: RopeParameterSet = serde_json::from_str("{}").unwrap();
    assert_eq!(empty, RopeParameterSet::default());

    let layered: RopeParameterSet = serde_json::from_str(
        r#"{"sliding_attention": {"rope_theta": 10000.0},
            "full_attention": {"type": "linear", "factor": 8.0, "rope_theta": 1e6}}"#,
    )
    .unwrap();
    let RopeParameterSet::ByLayerType(by_type) = &layered else {
        panic!("expected a per-layer mapping, got {layered:?}");
    };
    assert_eq!(
        by_type.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        ["full_attention", "sliding_attention"]
    );
    assert_eq!(
        by_type.get("full_attention").unwrap().rope_type,
        RopeType::Linear
    );
    assert_eq!(
        layered
            .parameters_for("sliding_attention")
            .unwrap()
            .rope_theta,
        Some(10_000.0)
    );
    assert!(layered.parameters_for("chunked_attention").is_none());
    assert_eq!(
        flat.parameters_for("anything").unwrap().rope_type,
        RopeType::Linear
    );
}

/// Mixed shapes and malformed layer entries fail with the offending keys.
#[test]
fn malformed_parameter_sets_name_the_offending_keys() {
    let mixed = serde_json::from_str::<RopeParameterSet>(
        r#"{"rope_type": "default", "full_attention": {"rope_theta": 1.0}}"#,
    )
    .unwrap_err()
    .to_string();
    assert!(mixed.contains("mixes per-layer-type objects"), "{mixed}");
    assert!(mixed.contains("full_attention"), "{mixed}");
    assert!(mixed.contains("rope_type"), "{mixed}");

    let malformed =
        serde_json::from_str::<RopeParameterSet>(r#"{"full_attention": {"rope_type": "unknown"}}"#)
            .unwrap_err()
            .to_string();
    assert!(
        malformed.contains("rope_parameters.full_attention"),
        "{malformed}"
    );

    let conflicting = serde_json::from_str::<RopeParameterSet>(
        r#"{"full_attention": {"type": "yarn", "rope_type": "linear"}}"#,
    )
    .unwrap_err()
    .to_string();
    assert!(conflicting.contains("conflicts"), "{conflicting}");

    for json in ["null", "3", "{", r#"{"factor": "two"}"#] {
        assert!(
            serde_json::from_str::<RopeParameterSet>(json).is_err(),
            "{json} must not parse"
        );
    }
}

/// Non-object JSON is rejected instead of parsing as positional fields.
///
/// Regression for fuzz finding ROPE-PARSE-1: serde's derived struct
/// deserializer also accepts sequences, so `[]` parsed as default RoPE and
/// `["yarn"]` set `rope_type` positionally. Transformers reads a dict.
#[test]
fn non_object_json_is_not_rope_parameters() {
    for json in [
        "[]",
        r#"["yarn"]"#,
        r#"["linear", 10000.0, 1.0, 2.0]"#,
        r#""yarn""#,
        "1.5",
        "true",
        "null",
    ] {
        let set = serde_json::from_str::<RopeParameterSet>(json)
            .expect_err("a non-object set must not parse")
            .to_string();
        assert!(
            set.contains("expected a RoPE parameter object"),
            "{json}: {set}"
        );
        let flat = serde_json::from_str::<RopeParameters>(json)
            .expect_err("non-object parameters must not parse")
            .to_string();
        assert!(
            flat.contains("expected a RoPE parameter object"),
            "{json}: {flat}"
        );
    }
    // Objects, including the empty one, still parse; unknown keys are dropped.
    assert_eq!(params("{}"), RopeParameters::default());
    assert_eq!(
        params(r#"{"rope_type": "yarn", "factor": 4.0, "unknown": [1]}"#),
        RopeParameters {
            rope_type: RopeType::Yarn,
            factor: Some(4.0),
            ..RopeParameters::default()
        }
    );
    // An absent mapping is still `None` in an optional config field.
    let optional: Option<RopeParameters> = serde_json::from_str("null").unwrap();
    assert!(optional.is_none());
}

/// Both shapes serialize back to the form they were parsed from.
#[test]
fn parameter_sets_round_trip_through_serialization() {
    let mut layers = std::collections::BTreeMap::new();
    layers.insert(
        "full_attention".to_owned(),
        params(r#"{"rope_type": "yarn", "factor": 4.0, "truncate": false}"#),
    );
    layers.insert("sliding_attention".to_owned(), RopeParameters::default());
    for set in [
        RopeParameterSet::default(),
        RopeParameterSet::from(params(r#"{"rope_type": "linear", "factor": 2.0}"#)),
        RopeParameterSet::from(RopeParametersByLayerType::new(layers).unwrap()),
    ] {
        let value = serde_json::to_value(&set).unwrap();
        assert_eq!(
            serde_json::from_value::<RopeParameterSet>(value).unwrap(),
            set
        );
    }
}

/// An empty per-layer mapping is unrepresentable, because `{}` means uniform default RoPE.
#[test]
fn empty_per_layer_mappings_are_rejected() {
    let error = RopeParametersByLayerType::new(std::collections::BTreeMap::new()).unwrap_err();
    assert_eq!(field(&error), "rope_parameters");
    assert!(
        error.to_string().contains("uniform default RoPE"),
        "{error}"
    );
    let parsed = serde_json::from_str::<RopeParametersByLayerType>("{}")
        .unwrap_err()
        .to_string();
    assert!(parsed.contains("at least one layer type"), "{parsed}");
    let one: RopeParametersByLayerType =
        serde_json::from_str(r#"{"full_attention": {"rope_type": "default"}}"#).unwrap();
    assert_eq!(
        one.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        ["full_attention"]
    );
    assert_eq!(
        serde_json::from_str::<RopeParameterSet>("{}").unwrap(),
        RopeParameterSet::default()
    );
}

/// Per-layer mappings must cover exactly the layer types the schedule uses.
#[test]
fn layer_types_must_match_per_layer_entries() {
    let layered: RopeParameterSet =
        serde_json::from_str(r#"{"sliding_attention": {}, "full_attention": {}}"#).unwrap();
    layered
        .validate_layer_types(
            "test",
            ["sliding_attention", "full_attention", "sliding_attention"],
        )
        .unwrap();
    let missing = layered
        .validate_layer_types("test", ["sliding_attention", "chunked_attention"])
        .unwrap_err();
    assert_eq!(field(&missing), "test.rope.chunked_attention");
    let stray = layered
        .validate_layer_types("test", ["sliding_attention"])
        .unwrap_err();
    assert_eq!(field(&stray), "test.rope.full_attention");
    RopeParameterSet::default()
        .validate_layer_types("test", ["anything"])
        .unwrap();

    let error = layered
        .to_rope_config("chunked_attention", &context())
        .unwrap_err();
    assert_eq!(field(&error), "test.rope.chunked_attention");
    assert!(error.to_string().contains("full_attention"), "{error}");
}

/// Per-layer errors and translations are scoped to the layer type.
#[test]
fn per_layer_translation_uses_layer_scoped_error_paths() {
    let layered: RopeParameterSet = serde_json::from_str(
        r#"{"sliding_attention": {"rope_theta": 10000.0},
            "full_attention": {"rope_type": "linear", "rope_theta": 1e6}}"#,
    )
    .unwrap();
    let sliding = layered
        .to_rope_config("sliding_attention", &context())
        .unwrap();
    assert_eq!(sliding.theta(), 10_000.0);
    rejects_layer(
        &layered,
        "full_attention",
        &context(),
        "test.rope.full_attention.factor",
    );
}

// ---------------------------------------------------------------------------
// Source precedence.
// ---------------------------------------------------------------------------

/// Theta resolves layer override, nested, top-level, then default.
#[test]
fn theta_resolves_sources_in_precedence_order() {
    let nested = params(r#"{"rope_theta": 500.0}"#);
    let empty = RopeParameters::default();
    let theta = |rope: &RopeParameters, context: RopeContext| {
        rope.to_rope_config(&context).unwrap().theta()
    };
    assert_eq!(theta(&empty, context()), DEFAULT_ROPE_THETA);
    assert_eq!(
        theta(&empty, context().with_default_rope_theta(150_000.0)),
        150_000.0
    );
    assert_eq!(theta(&empty, context().with_rope_theta(Some(20.0))), 20.0);
    assert_eq!(theta(&nested, context()), 500.0);
    assert_eq!(
        theta(&nested, context().with_rope_theta(Some(500.0))),
        500.0
    );
    assert_eq!(theta(&nested, context().with_layer_rope_theta(7.0)), 7.0);

    let disagree = context().with_rope_theta(Some(20.0));
    let error = rejects(&nested, &disagree, "test.rope.rope_theta");
    assert!(error.to_string().contains("disagrees"), "{error}");
    // Transformers keeps the nested value (`setdefault`).
    assert_eq!(
        theta(
            &nested,
            disagree.with_unused_fields(UnusedRopeFields::Ignore)
        ),
        500.0
    );
}

/// Partial factors resolve nested, then top-level, and must agree.
#[test]
fn partial_rotary_factor_resolves_and_validates_the_rotary_width() {
    let nested = params(r#"{"partial_rotary_factor": 0.5}"#);
    let empty = RopeParameters::default();
    let width = |rope: &RopeParameters, context: RopeContext| {
        rope.to_rope_config(&context).unwrap().rotary_dimension()
    };
    assert_eq!(width(&empty, context()), HEAD);
    assert_eq!(width(&nested, context()), 8);
    assert_eq!(
        width(&empty, context().with_partial_rotary_factor(Some(0.25))),
        4
    );
    assert_eq!(
        width(&nested, context().with_partial_rotary_factor(Some(0.5))),
        8
    );

    let disagree = context().with_partial_rotary_factor(Some(0.25));
    rejects(&nested, &disagree, "test.rope.partial_rotary_factor");
    // Transformers' `standardize_rope_params` writes the top-level value last.
    assert_eq!(
        width(
            &nested,
            disagree.with_unused_fields(UnusedRopeFields::Ignore)
        ),
        4
    );

    for invalid in [0.3, 0.0, 1.5, f64::NAN] {
        let rope = RopeParameters {
            partial_rotary_factor: Some(invalid),
            ..RopeParameters::default()
        };
        rejects(&rope, &context(), "test.rope.partial_rotary_factor");
    }
}

/// Whole-head families reject a strict prefix unless told to ignore it.
#[test]
fn full_head_families_reject_or_ignore_partial_rotation() {
    let whole = context().with_partial_rotary(PartialRotary::FullHeadOnly);
    let nested = params(r#"{"partial_rotary_factor": 0.5}"#);
    let error = rejects(&nested, &whole, "test.rope.partial_rotary_factor");
    assert!(error.to_string().contains("whole heads"), "{error}");
    rejects(
        &RopeParameters::default(),
        &whole.with_partial_rotary_factor(Some(0.25)),
        "test.rope.partial_rotary_factor",
    );
    let ignored = nested
        .to_rope_config(&whole.with_unused_fields(UnusedRopeFields::Ignore))
        .unwrap();
    assert_eq!(ignored.rotary_dimension(), HEAD);
    let explicit_one = params(r#"{"partial_rotary_factor": 1.0}"#);
    assert_eq!(
        explicit_one
            .to_rope_config(&whole)
            .unwrap()
            .rotary_dimension(),
        HEAD
    );
}

/// The context's sizes and an explicit zero original context are rejected early.
#[test]
fn context_dimensions_are_validated_before_translation() {
    let rope = RopeParameters::default();
    rejects(&rope, &RopeContext::new("test", 0, MAX), "test.head_dim");
    rejects(
        &rope,
        &RopeContext::new("test", HEAD, 0),
        "test.max_position_embeddings",
    );
    let yarn =
        params(r#"{"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 0}"#);
    rejects(
        &yarn,
        &context(),
        "test.rope.original_max_position_embeddings",
    );
    let llama = params(
        r#"{"rope_type": "llama3", "factor": 2.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0}"#,
    );
    rejects(
        &llama,
        &context().with_original_max_position_embeddings(Some(0)),
        "test.rope.original_max_position_embeddings",
    );
    rejects(
        &params(r#"{"rope_theta": -1.0}"#),
        &context(),
        "test.rope.rope_theta",
    );
}

// ---------------------------------------------------------------------------
// Algorithms.
// ---------------------------------------------------------------------------

/// Default RoPE maps to no scaling with the resolved geometry.
#[test]
fn default_rope_has_no_scaling() {
    let config = RopeParameters::default()
        .to_rope_config(&context())
        .unwrap();
    assert_eq!(
        config,
        RopeConfig::new(HEAD, HEAD, DEFAULT_ROPE_THETA, RopeScaling::None).unwrap()
    );
}

/// Linear scaling needs a factor of at least one.
#[test]
fn linear_requires_a_valid_factor() {
    rejects(
        &params(r#"{"rope_type": "linear"}"#),
        &context(),
        "test.rope.factor",
    );
    rejects(
        &params(r#"{"rope_type": "linear", "factor": 0.5}"#),
        &context(),
        "test.rope.factor",
    );
    let config = params(r#"{"rope_type": "linear", "factor": 4.0}"#)
        .to_rope_config(&context())
        .unwrap();
    assert_eq!(
        config.scaling(),
        RopeScaling::Linear(LinearRopeScaling::new(4.0).unwrap())
    );
}

/// Dynamic NTK compares against `max_position_embeddings` unless told otherwise.
#[test]
fn dynamic_ntk_selects_its_trained_context() {
    let trained = |rope: &RopeParameters, context: RopeContext| match rope
        .to_rope_config(&context)
        .unwrap()
        .scaling()
    {
        RopeScaling::DynamicNtk(dynamic) => dynamic.original_max_position_embeddings(),
        other => panic!("expected dynamic NTK, got {other:?}"),
    };
    let plain = params(r#"{"rope_type": "dynamic", "factor": 2.0}"#);
    let nested_max = params(
        r#"{"rope_type": "dynamic", "factor": 2.0, "original_max_position_embeddings": 128}"#,
    );
    let nested_short = params(
        r#"{"rope_type": "dynamic", "factor": 2.0, "original_max_position_embeddings": 32}"#,
    );
    assert_eq!(trained(&plain, context()), MAX);
    assert_eq!(trained(&nested_max, context()), MAX);
    // A top-level original context is a config attribute, not an unused key.
    assert_eq!(
        trained(
            &plain,
            context().with_original_max_position_embeddings(Some(32))
        ),
        MAX
    );
    let error = rejects(
        &nested_short,
        &context(),
        "test.rope.original_max_position_embeddings",
    );
    assert!(
        error.to_string().contains("OriginalMaxPositionEmbeddings"),
        "{error}"
    );
    assert_eq!(
        trained(
            &nested_short,
            context().with_unused_fields(UnusedRopeFields::Ignore)
        ),
        MAX
    );

    let legacy =
        context().with_dynamic_ntk_context(DynamicNtkContext::OriginalMaxPositionEmbeddings);
    assert_eq!(trained(&nested_short, legacy), 32);
    assert_eq!(trained(&plain, legacy), MAX);
    assert_eq!(
        trained(
            &plain,
            legacy.with_original_max_position_embeddings(Some(64))
        ),
        64
    );

    rejects(
        &params(r#"{"rope_type": "dynamic"}"#),
        &context(),
        "test.rope.factor",
    );
    rejects(
        &params(r#"{"rope_type": "dynamic", "factor": 0.5}"#),
        &context(),
        "test.rope.factor",
    );
    rejects(
        &params(
            r#"{"rope_type": "dynamic", "factor": 2.0, "original_max_position_embeddings": 0}"#,
        ),
        &legacy,
        "test.rope.original_max_position_embeddings",
    );
    // A two-channel whole head is too narrow for dynamic NTK; the head is at fault.
    rejects(&plain, &RopeContext::new("test", 2, MAX), "test.head_dim");
    // The same width as a strict prefix is the partial factor's fault.
    rejects(
        &plain,
        &RopeContext::new("test", 8, MAX).with_partial_rotary_factor(Some(0.25)),
        "test.rope.partial_rotary_factor",
    );
}

/// YaRN resolves its factor, original context, betas, and truncation.
#[test]
fn yarn_resolves_factor_context_betas_and_truncation() {
    rejects(
        &params(r#"{"rope_type": "yarn"}"#),
        &context(),
        "test.rope.factor",
    );
    let defaults = yarn_of(
        params(r#"{"rope_type": "yarn", "factor": 4.0}"#)
            .to_rope_config(&context())
            .unwrap(),
    );
    assert_eq!(defaults.original_max_position_embeddings(), MAX);
    assert_eq!((defaults.beta_fast(), defaults.beta_slow()), (32.0, 1.0));
    assert!(defaults.truncate());
    assert_eq!(defaults.attention_factor(), 1.0 + 0.1 * 4.0_f64.ln());

    let top_level = yarn_of(
        params(r#"{"rope_type": "yarn", "factor": 4.0}"#)
            .to_rope_config(&context().with_original_max_position_embeddings(Some(32)))
            .unwrap(),
    );
    assert_eq!(top_level.original_max_position_embeddings(), 32);
    let conflicting =
        params(r#"{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16}"#);
    let phi = context().with_original_max_position_embeddings(Some(32));
    rejects(
        &conflicting,
        &phi,
        "test.rope.original_max_position_embeddings",
    );
    // Agreeing nested and top-level values are one source.
    let agreeing =
        params(r#"{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32}"#);
    assert_eq!(
        yarn_of(agreeing.to_rope_config(&phi).unwrap()).original_max_position_embeddings(),
        32
    );
    // Transformers gives the top-level attribute priority (Phi-3 exception).
    let lenient = conflicting
        .to_rope_config(&phi.with_unused_fields(UnusedRopeFields::Ignore))
        .unwrap();
    assert_eq!(yarn_of(lenient).original_max_position_embeddings(), 32);

    let custom = yarn_of(
        params(r#"{"rope_type": "yarn", "factor": 4.0, "beta_slow": 2.0, "truncate": false}"#)
            .to_rope_config(&context())
            .unwrap(),
    );
    assert_eq!((custom.beta_fast(), custom.beta_slow()), (32.0, 2.0));
    assert!(!custom.truncate());
    for (json, expected) in [
        (
            r#"{"rope_type": "yarn", "factor": 4.0, "beta_fast": -1.0}"#,
            "test.rope.beta_fast",
        ),
        (
            r#"{"rope_type": "yarn", "factor": 4.0, "beta_slow": 0.0}"#,
            "test.rope.beta_slow",
        ),
        (
            r#"{"rope_type": "yarn", "factor": 4.0, "beta_fast": 1.0, "beta_slow": 2.0}"#,
            "test.rope.beta_fast",
        ),
        (
            r#"{"rope_type": "yarn", "factor": 4.0, "rope_theta": 1.0}"#,
            "test.rope.rope_theta",
        ),
        (
            r#"{"rope_type": "yarn", "factor": 0.5}"#,
            "test.rope.factor",
        ),
    ] {
        rejects(&params(json), &context(), expected);
    }
}

/// A per-layer `truncate: false` is never read by Transformers 5.2, so it is unused.
#[test]
fn per_layer_yarn_truncate_follows_the_reference_reader() {
    let layered: RopeParameterSet = serde_json::from_str(
        r#"{"full_attention": {"rope_type": "yarn", "factor": 4.0, "truncate": false},
            "sliding_attention": {"rope_type": "yarn", "factor": 4.0, "truncate": true}}"#,
    )
    .unwrap();
    rejects_layer(
        &layered,
        "full_attention",
        &context(),
        "test.rope.full_attention.truncate",
    );
    let ignored = layered
        .to_rope_config(
            "full_attention",
            &context().with_unused_fields(UnusedRopeFields::Ignore),
        )
        .unwrap();
    assert!(yarn_of(ignored).truncate(), "Transformers 5.2 truncates");
    let explicit_true = layered
        .to_rope_config("sliding_attention", &context())
        .unwrap();
    assert!(yarn_of(explicit_true).truncate());
}

/// Per-layer mappings never read a top-level original context, as in Transformers 5.2.
#[test]
fn per_layer_original_context_ignores_the_top_level_value() {
    let layered: RopeParameterSet = serde_json::from_str(
        r#"{"full_attention": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32},
            "sliding_attention": {"rope_type": "yarn", "factor": 4.0}}"#,
    )
    .unwrap();
    let original = |layer_type: &str, context: RopeContext| {
        yarn_of(layered.to_rope_config(layer_type, &context).unwrap())
            .original_max_position_embeddings()
    };
    // Without a top-level value: nested, else `max_position_embeddings`.
    assert_eq!(original("full_attention", context()), 32);
    assert_eq!(original("sliding_attention", context()), MAX);
    // A top-level value equal to the layer's resolved context changes nothing.
    assert_eq!(
        original(
            "full_attention",
            context().with_original_max_position_embeddings(Some(32))
        ),
        32
    );
    assert_eq!(
        original(
            "sliding_attention",
            context().with_original_max_position_embeddings(Some(MAX))
        ),
        MAX
    );
    // A differing top-level value is unused: strictly an error, leniently ignored.
    let top_level = context().with_original_max_position_embeddings(Some(64));
    for (layer_type, resolved) in [("full_attention", 32), ("sliding_attention", MAX)] {
        let error = layered
            .to_rope_config(layer_type, &top_level)
            .expect_err(layer_type);
        assert_eq!(
            field(&error),
            format!("test.rope.{layer_type}.original_max_position_embeddings")
        );
        assert!(error.to_string().contains("per-layer-type"), "{error}");
        assert_eq!(
            original(
                layer_type,
                top_level.with_unused_fields(UnusedRopeFields::Ignore)
            ),
            resolved,
            "Transformers keeps the layer's own context"
        );
    }
    // A flat mapping still gives the top-level value priority (Phi-3 exception).
    let flat = params(r#"{"rope_type": "yarn", "factor": 4.0}"#);
    assert_eq!(
        yarn_of(flat.to_rope_config(&top_level).unwrap()).original_max_position_embeddings(),
        64
    );
    // An explicit zero is rejected before the sources are compared.
    let zero: RopeParameterSet = serde_json::from_str(
        r#"{"full_attention": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 0}}"#,
    )
    .unwrap();
    let error = zero
        .to_rope_config("full_attention", &top_level)
        .unwrap_err();
    assert!(error.to_string().contains("nonzero"), "{error}");
}

/// Expected YaRN amplitude outcome for one convention.
#[derive(Debug, Clone, Copy)]
enum Amplitude {
    /// The translation succeeds with this amplitude.
    Value(f64),
    /// Strict translation fails at this field; `Ignore` yields the value.
    Unused(&'static str, f64),
}

/// One amplitude case: `(attention_factor, mscale, mscale_all_dim, expectations)`.
///
/// The expectations are, in order, the Hugging Face convention with an
/// unscaled softmax, the Hugging Face convention with the `mscale_all_dim`
/// softmax correction, and the DeepSeek convention (whose amplitude reads
/// `mscale_all_dim` itself, so the softmax setting cannot change it).
type AmplitudeCase = (Option<f64>, Option<f64>, Option<f64>, [Amplitude; 3]);

/// Each convention resolves `attention_factor`, `mscale`, and `mscale_all_dim` as documented.
#[test]
fn yarn_amplitude_follows_each_mscale_convention() {
    use Amplitude::{Unused, Value};
    let factor: f64 = 16.0;
    let m = |multiplier: f64| yarn_mscale(factor, multiplier);
    let default = m(1.0);
    let cases: [AmplitudeCase; 11] = [
        (None, None, None, [Value(default); 3]),
        (Some(1.5), None, None, [Value(1.5); 3]),
        (None, Some(1.0), Some(0.8), [Value(m(1.0) / m(0.8)); 3]),
        (
            None,
            None,
            Some(0.8),
            [
                Unused("mscale_all_dim", default),
                Value(default),
                Value(m(1.0) / m(0.8)),
            ],
        ),
        (
            None,
            None,
            Some(0.0),
            [
                Unused("mscale_all_dim", default),
                Value(default),
                Value(default),
            ],
        ),
        (
            None,
            Some(0.7),
            None,
            [
                Unused("mscale", default),
                Unused("mscale", default),
                Value(m(0.7)),
            ],
        ),
        (
            None,
            Some(0.0),
            Some(0.8),
            [
                Unused("mscale", default),
                Unused("mscale", default),
                Value(1.0 / m(0.8)),
            ],
        ),
        (
            None,
            Some(1.0),
            Some(0.0),
            [
                Unused("mscale", default),
                Unused("mscale", default),
                Value(m(1.0)),
            ],
        ),
        (
            Some(1.5),
            Some(1.0),
            Some(1.0),
            [
                Unused("mscale", 1.5),
                Unused("mscale", 1.5),
                Unused("attention_factor", 1.0),
            ],
        ),
        (
            Some(1.5),
            None,
            Some(1.0),
            [
                Unused("mscale_all_dim", 1.5),
                Value(1.5),
                Unused("attention_factor", 1.0),
            ],
        ),
        (
            Some(1.5),
            Some(0.5),
            None,
            [
                Unused("mscale", 1.5),
                Unused("mscale", 1.5),
                Unused("attention_factor", m(0.5)),
            ],
        ),
    ];
    for (attention_factor, mscale, mscale_all_dim, [unscaled, squared, deepseek]) in cases {
        let rope = RopeParameters {
            rope_type: RopeType::Yarn,
            factor: Some(factor),
            original_max_position_embeddings: Some(32),
            attention_factor,
            mscale,
            mscale_all_dim,
            ..RopeParameters::default()
        };
        for (convention, softmax, expected) in [
            (
                YarnMscaleConvention::HuggingFace,
                SoftmaxMscale::Unscaled,
                unscaled,
            ),
            (
                YarnMscaleConvention::HuggingFace,
                SoftmaxMscale::MscaleAllDimSquared,
                squared,
            ),
            (
                YarnMscaleConvention::DeepSeekReference,
                SoftmaxMscale::Unscaled,
                deepseek,
            ),
            (
                YarnMscaleConvention::DeepSeekReference,
                SoftmaxMscale::MscaleAllDimSquared,
                deepseek,
            ),
        ] {
            let strict = context()
                .with_yarn_mscale_convention(convention)
                .with_softmax_mscale(softmax);
            let label = format!(
                "{convention:?} {softmax:?} {attention_factor:?} {mscale:?} {mscale_all_dim:?}"
            );
            let value = match expected {
                Value(value) => {
                    let strict_config = rope.to_rope_config(&strict).expect(&label);
                    assert_eq!(yarn_of(strict_config).attention_factor(), value, "{label}");
                    value
                }
                Unused(unused, value) => {
                    let error = rope.to_rope_config(&strict).expect_err(&label);
                    assert_eq!(field(&error), format!("test.rope.{unused}"), "{label}");
                    value
                }
            };
            let lenient = strict.with_unused_fields(UnusedRopeFields::Ignore);
            let actual = yarn_of(rope.to_rope_config(&lenient).expect(&label)).attention_factor();
            assert_eq!(actual, value, "{label}");
        }
    }
}

/// Invalid amplitudes are reported at the field they were derived from.
#[test]
fn invalid_yarn_amplitudes_name_their_source_field() {
    let explicit = params(r#"{"rope_type": "yarn", "factor": 4.0, "attention_factor": -1.0}"#);
    rejects(&explicit, &context(), "test.rope.attention_factor");
    let derived = params(r#"{"rope_type": "yarn", "factor": 4.0, "mscale_all_dim": -100.0}"#);
    let error = rejects(
        &derived,
        &context().with_yarn_mscale_convention(YarnMscaleConvention::DeepSeekReference),
        "test.rope.mscale",
    );
    assert!(
        error.to_string().contains("mscale / mscale_all_dim ratio"),
        "{error}"
    );
}

/// The logarithmic correction is one at or below unit scale and otherwise linear in ln.
#[test]
fn yarn_mscale_matches_the_reference_definition() {
    assert_eq!(yarn_mscale(1.0, 5.0), 1.0);
    assert_eq!(yarn_mscale(0.5, 5.0), 1.0);
    assert_eq!(yarn_mscale(40.0, 0.0), 1.0);
    assert_eq!(yarn_mscale(40.0, 0.707), 1.0 + 0.1 * 0.707 * 40.0_f64.ln());
}

/// The softmax multiplier follows the declared convention for every scaled algorithm.
///
/// Expected values follow DeepSeek's reference (`softmax_scale` whenever
/// `rope_scaling` is set) and Transformers 5.2 `deepseek_v3` (`scaling`
/// whenever `rope_type != "default"`): a linear config with
/// `mscale_all_dim = 1` and factor 4 gives `(1 + 0.1 ln 4)^2 = 1.29648`.
#[test]
fn softmax_scale_multiplier_follows_the_declared_convention() {
    let squared = context().with_softmax_mscale(SoftmaxMscale::MscaleAllDimSquared);
    let multiplier =
        |json: &str, context: &RopeContext| params(json).softmax_scale_multiplier(context);
    let linear = r#"{"rope_type": "linear", "factor": 4.0, "mscale_all_dim": 1.0}"#;
    // An unscaled softmax never applies the correction.
    assert_eq!(multiplier(linear, &context()).unwrap(), 1.0);
    let probe = multiplier(linear, &squared).unwrap();
    assert_eq!(probe, yarn_mscale(4.0, 1.0).powi(2));
    assert!((probe - 1.296_476_992_780_706_3).abs() < 1e-15, "{probe}");
    for json in [
        r#"{"rope_type": "dynamic", "factor": 4.0, "mscale_all_dim": 1.0}"#,
        r#"{"rope_type": "yarn", "factor": 4.0, "mscale_all_dim": 1.0}"#,
        r#"{"rope_type": "llama3", "factor": 4.0, "mscale_all_dim": 1.0}"#,
    ] {
        assert_eq!(multiplier(json, &squared).unwrap(), probe, "{json}");
    }
    assert_eq!(
        multiplier(
            r#"{"rope_type": "yarn", "factor": 40.0, "mscale_all_dim": 0.707}"#,
            &squared
        )
        .unwrap(),
        yarn_mscale(40.0, 0.707).powi(2)
    );
    // Default RoPE, an absent or zero `mscale_all_dim`, and a unit factor give one.
    for json in [
        "{}",
        r#"{"mscale_all_dim": 1.0}"#,
        r#"{"rope_type": "yarn", "factor": 4.0}"#,
        r#"{"rope_type": "yarn", "factor": 4.0, "mscale_all_dim": 0.0}"#,
        r#"{"rope_type": "linear", "factor": 1.0, "mscale_all_dim": 1.0}"#,
    ] {
        assert_eq!(multiplier(json, &squared).unwrap(), 1.0, "{json}");
    }
    // The correction needs a usable factor and a positive m(f, mscale_all_dim).
    for (json, expected) in [
        (
            r#"{"rope_type": "longrope", "mscale_all_dim": 1.0}"#,
            "test.rope.factor",
        ),
        (
            r#"{"rope_type": "linear", "factor": 0.5, "mscale_all_dim": 1.0}"#,
            "test.rope.factor",
        ),
        (
            r#"{"rope_type": "yarn", "factor": 4.0, "mscale_all_dim": -100.0}"#,
            "test.rope.mscale_all_dim",
        ),
    ] {
        let error = multiplier(json, &squared).unwrap_err();
        assert_eq!(field(&error), expected, "{json}: {error}");
    }
    let nan = RopeParameters {
        rope_type: RopeType::Yarn,
        factor: Some(f64::NAN),
        mscale_all_dim: Some(1.0),
        ..RopeParameters::default()
    };
    assert_eq!(
        field(&nan.softmax_scale_multiplier(&squared).unwrap_err()),
        "test.rope.factor"
    );
    // A non-finite multiplier gives a non-finite correction.
    let non_finite = RopeParameters {
        rope_type: RopeType::Linear,
        factor: Some(4.0),
        mscale_all_dim: Some(f64::INFINITY),
        ..RopeParameters::default()
    };
    let error = non_finite.softmax_scale_multiplier(&squared).unwrap_err();
    assert_eq!(field(&error), "test.rope.mscale_all_dim");
    assert!(error.to_string().contains("finite and positive"), "{error}");
}

/// Translation validates the declared softmax correction and accepts what it reads.
#[test]
fn declared_softmax_correction_is_validated_during_translation() {
    let squared = context().with_softmax_mscale(SoftmaxMscale::MscaleAllDimSquared);
    // `mscale_all_dim` is read by every scaled algorithm and changes no table.
    for rope_type in ROPE_TYPES {
        let mut rope = minimal(rope_type);
        let label = format!("{rope_type:?}");
        if rope_type == RopeType::Default {
            rope.mscale_all_dim = Some(1.0);
            let error = rejects(&rope, &squared, "test.rope.mscale_all_dim");
            assert!(error.to_string().contains("is not read by"), "{error}");
            continue;
        }
        // The correction needs an explicit factor even where RoPE defaults it.
        rope.factor.get_or_insert(2.0);
        let reference = rope.to_rope_config(&squared).unwrap();
        rope.mscale_all_dim = Some(1.0);
        // The tables are unchanged; for YaRN, the Hugging Face amplitude
        // ignores a lone `mscale_all_dim`.
        assert_eq!(
            rope.to_rope_config(&squared).expect(&label),
            reference,
            "{label}"
        );
        assert_eq!(
            rope.softmax_scale_multiplier(&squared).unwrap(),
            yarn_mscale(2.0, 1.0).powi(2),
            "{label}"
        );
    }
    // An invalid correction fails translation itself, also in a per-layer scope.
    let invalid = params(r#"{"rope_type": "linear", "factor": 4.0, "mscale_all_dim": -100.0}"#);
    rejects(&invalid, &squared, "test.rope.mscale_all_dim");
    let mut layers = std::collections::BTreeMap::new();
    layers.insert("full_attention".to_owned(), invalid.clone());
    let layered = RopeParameterSet::from(RopeParametersByLayerType::new(layers).unwrap());
    rejects_layer(
        &layered,
        "full_attention",
        &squared,
        "test.rope.full_attention.mscale_all_dim",
    );
    let missing_factor = params(
        r#"{"rope_type": "longrope", "short_factor": [1, 1, 1, 1, 1, 1, 1, 1], "long_factor": [2, 2, 2, 2, 2, 2, 2, 2], "mscale_all_dim": 1.0}"#,
    );
    let error = rejects(&missing_factor, &squared, "test.rope.factor");
    assert!(error.to_string().contains("softmax correction"), "{error}");
    // Without the declaration the unread value is reported instead.
    rejects(&invalid, &context(), "test.rope.mscale_all_dim");
    assert!(
        invalid
            .to_rope_config(&context())
            .unwrap_err()
            .to_string()
            .contains("is not read by linear RoPE")
    );
}

/// The squared softmax multiplier, not only its root, must be a normal positive `f32`.
///
/// Regression for fuzz finding ROPE-SOFTMAX-1: `mscale_all_dim = 1e300` gives a
/// finite correction `m = 1 + 0.1 * 1e300 * ln 2`, so translation succeeded,
/// but the multiplier `m^2` is infinite. With `factor = e` (so `ln f = 1`),
/// `mscale_all_dim = 1e20` gives `m^2 ~= 1e38`, inside the `f32` range, while
/// `mscale_all_dim = 1e21` gives `m^2 ~= 1e40`, finite in `f64` but not in `f32`.
#[test]
fn squared_softmax_multipliers_must_fit_f32() {
    let squared = context().with_softmax_mscale(SoftmaxMscale::MscaleAllDimSquared);
    let overflowing = params(r#"{"rope_type": "linear", "factor": 2.0, "mscale_all_dim": 1e300}"#);
    let error = rejects(&overflowing, &squared, "test.rope.mscale_all_dim");
    assert!(error.to_string().contains("softmax multiplier"), "{error}");
    assert!(overflowing.softmax_scale_multiplier(&squared).is_err());
    let e = std::f64::consts::E;
    let inside = RopeParameters {
        rope_type: RopeType::Linear,
        factor: Some(e),
        mscale_all_dim: Some(1e20),
        ..RopeParameters::default()
    };
    inside
        .to_rope_config(&squared)
        .expect("m^2 ~= 1e38 fits f32");
    let multiplier = inside.softmax_scale_multiplier(&squared).unwrap();
    assert_eq!(multiplier, yarn_mscale(e, 1e20).powi(2));
    assert!((multiplier as f32).is_finite());
    let outside = RopeParameters {
        mscale_all_dim: Some(1e21),
        ..inside.clone()
    };
    assert!(yarn_mscale(e, 1e21).powi(2).is_finite());
    rejects(&outside, &squared, "test.rope.mscale_all_dim");
    // Per-layer entries are checked in their own scope.
    let mut layers = std::collections::BTreeMap::new();
    layers.insert("full_attention".to_owned(), overflowing);
    let layered = RopeParameterSet::from(RopeParametersByLayerType::new(layers).unwrap());
    rejects_layer(
        &layered,
        "full_attention",
        &squared,
        "test.rope.full_attention.mscale_all_dim",
    );
    // Without the declaration nothing squares the correction, and the
    // unread value is reported as such.
    let error = rejects(
        &params(r#"{"rope_type": "linear", "factor": 2.0, "mscale_all_dim": 1e300}"#),
        &context(),
        "test.rope.mscale_all_dim",
    );
    assert!(error.to_string().contains("is not read by"), "{error}");
}

/// Bases below one and subnormal LongRoPE factors fail translation contextually.
///
/// Regressions for fuzz findings ROPE-F32-1/2: both used to translate and then
/// overflow the `f32` inverse frequencies.
#[test]
fn bases_and_factors_that_overflow_f32_frequencies_are_rejected() {
    for theta in ["1e-300", "0.5", "0.9999999999999999"] {
        let rope = params(&format!(
            r#"{{"rope_type": "default", "rope_theta": {theta}}}"#
        ));
        let error = rejects(&rope, &context(), "test.rope.rope_theta");
        assert!(error.to_string().contains("at least 1"), "{error}");
    }
    params(r#"{"rope_type": "default", "rope_theta": 1.0}"#)
        .to_rope_config(&context())
        .expect("a base of one keeps every frequency at one");
    // A context-supplied base is checked the same way.
    rejects(
        &RopeParameters::default(),
        &context().with_rope_theta(Some(0.25)),
        "test.rope.rope_theta",
    );
    let longrope = |short: &str| {
        params(&format!(
            r#"{{"rope_type": "longrope", "short_factor": [{short}, 1, 1, 1, 1, 1, 1, 1], "long_factor": [1, 1, 1, 1, 1, 1, 1, 1], "original_max_position_embeddings": 64}}"#
        ))
    };
    for factor in ["1e-45", "1e-39", "1e39"] {
        let error = rejects(&longrope(factor), &context(), "test.rope.short_factor");
        assert!(error.to_string().contains("normal positive f32"), "{error}");
    }
    let config = longrope("1.1754943508222875e-38")
        .to_rope_config(&context())
        .expect("f32::MIN_POSITIVE is a valid factor");
    for evaluated in [1, 64, 65] {
        let frequencies = crate::position::RotaryFrequencies::new(config, evaluated).unwrap();
        assert!(frequencies.inverse().iter().all(|omega| omega.is_finite()));
    }
}

/// LongRoPE needs both tables, defaults its factor, and honours an explicit amplitude.
#[test]
fn longrope_resolves_tables_factor_and_amplitude() {
    let short = "[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]";
    let long = "[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]";
    let long_rope = |extra: &str| {
        params(&format!(
            r#"{{"rope_type": "longrope", "short_factor": {short}, "long_factor": {long}{extra}}}"#
        ))
    };
    let of = |config: RopeConfig| match config.scaling() {
        RopeScaling::LongRope(long) => long,
        other => panic!("expected LongRoPE, got {other:?}"),
    };
    let implied = of(long_rope(r#", "original_max_position_embeddings": 32"#)
        .to_rope_config(&context())
        .unwrap());
    assert_eq!(implied.factor(), 4.0);
    assert_eq!(implied.original_max_position_embeddings(), 32);
    assert_eq!(
        implied.attention_factor(),
        (1.0 + 4.0_f64.ln() / 32.0_f64.ln()).sqrt()
    );
    let explicit = of(long_rope(r#", "factor": 2.0, "attention_factor": 1.25"#)
        .to_rope_config(&context().with_original_max_position_embeddings(Some(64)))
        .unwrap());
    assert_eq!(explicit.factor(), 2.0);
    assert_eq!(explicit.attention_factor(), 1.25);
    assert_eq!(explicit.original_max_position_embeddings(), 64);
    // Without an original context the factor is max / max = 1 and the amplitude 1.
    assert_eq!(
        of(long_rope("").to_rope_config(&context()).unwrap()).attention_factor(),
        1.0
    );

    let invalid_long = format!(
        r#"{{"rope_type": "longrope", "short_factor": {short}, "long_factor": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -2.0]}}"#
    );
    for (rope, expected) in [
        (
            params(&format!(
                r#"{{"rope_type": "longrope", "long_factor": {long}}}"#
            )),
            "test.rope.short_factor",
        ),
        (
            params(&format!(
                r#"{{"rope_type": "longrope", "short_factor": {short}}}"#
            )),
            "test.rope.long_factor",
        ),
        (
            params(
                r#"{"rope_type": "longrope", "short_factor": [1.0], "long_factor": [1.0, 2.0]}"#,
            ),
            "test.rope.long_factor",
        ),
        (
            params(r#"{"rope_type": "longrope", "short_factor": [1.0], "long_factor": [1.0]}"#),
            "test.rope.short_factor",
        ),
        (
            params(r#"{"rope_type": "longrope", "short_factor": [], "long_factor": []}"#),
            "test.rope.short_factor",
        ),
        (params(&invalid_long), "test.rope.long_factor"),
        (
            long_rope(r#", "attention_factor": 0.0"#),
            "test.rope.attention_factor",
        ),
        (long_rope(r#", "factor": 0.5"#), "test.rope.factor"),
    ] {
        rejects(&rope, &context(), expected);
    }
    rejects(
        &long_rope(r#", "original_max_position_embeddings": 0"#),
        &context(),
        "test.rope.original_max_position_embeddings",
    );
    let error = rejects(
        &long_rope(r#", "original_max_position_embeddings": 256"#),
        &context(),
        "test.rope.original_max_position_embeddings",
    );
    assert!(error.to_string().contains("below one"), "{error}");
}

/// Llama 3 needs its factor and both strictly ordered band multipliers.
#[test]
fn llama3_requires_ordered_band_factors() {
    let llama = |extra: &str| {
        params(&format!(
            r#"{{"rope_type": "llama3", "factor": 8.0, "original_max_position_embeddings": 32{extra}}}"#
        ))
    };
    rejects(
        &llama(r#", "high_freq_factor": 4.0"#),
        &context(),
        "test.rope.low_freq_factor",
    );
    rejects(
        &llama(r#", "low_freq_factor": 1.0"#),
        &context(),
        "test.rope.high_freq_factor",
    );
    rejects(
        &llama(r#", "low_freq_factor": 4.0, "high_freq_factor": 4.0"#),
        &context(),
        "test.rope.low_freq_factor",
    );
    rejects(
        &params(r#"{"rope_type": "llama3", "low_freq_factor": 1.0, "high_freq_factor": 4.0}"#),
        &context(),
        "test.rope.factor",
    );
    let config = llama(r#", "low_freq_factor": 1.0, "high_freq_factor": 4.0"#)
        .to_rope_config(&context())
        .unwrap();
    assert_eq!(
        config.scaling(),
        RopeScaling::Llama3(Llama3RopeScaling::new(8.0, 1.0, 4.0, 32).unwrap())
    );
}

/// Proportional RoPE spans the head and reads the partial factor as its pair fraction.
#[test]
fn proportional_reads_partial_factor_as_a_pair_fraction() {
    let whole = context().with_partial_rotary(PartialRotary::FullHeadOnly);
    let config = params(r#"{"rope_type": "proportional", "partial_rotary_factor": 0.6}"#)
        .to_rope_config(&whole)
        .unwrap();
    assert_eq!(config.rotary_dimension(), HEAD);
    assert_eq!(
        config.scaling(),
        RopeScaling::Proportional(ProportionalRopeScaling::new(0.6, 1.0).unwrap())
    );
    let scaled = params(r#"{"rope_type": "proportional", "factor": 2.0}"#)
        .to_rope_config(&context().with_partial_rotary_factor(Some(0.25)))
        .unwrap();
    assert_eq!(
        scaled.scaling(),
        RopeScaling::Proportional(ProportionalRopeScaling::new(0.25, 2.0).unwrap())
    );
    rejects(
        &params(r#"{"rope_type": "proportional", "partial_rotary_factor": 1.5}"#),
        &context(),
        "test.rope.partial_rotary_factor",
    );
    rejects(
        &params(r#"{"rope_type": "proportional", "factor": 0.5}"#),
        &context(),
        "test.rope.factor",
    );
}

// ---------------------------------------------------------------------------
// Strictness.
// ---------------------------------------------------------------------------

/// Returns parameters that translate successfully for `rope_type` with only required fields.
fn minimal(rope_type: RopeType) -> RopeParameters {
    let mut rope = RopeParameters {
        rope_type,
        ..RopeParameters::default()
    };
    match rope_type {
        RopeType::Default | RopeType::Proportional => {}
        RopeType::Linear | RopeType::Dynamic | RopeType::Yarn => rope.factor = Some(2.0),
        RopeType::LongRope => {
            rope.short_factor = Some(vec![1.0; HEAD / 2]);
            rope.long_factor = Some(vec![2.0; HEAD / 2]);
        }
        RopeType::Llama3 => {
            rope.factor = Some(2.0);
            rope.low_freq_factor = Some(1.0);
            rope.high_freq_factor = Some(4.0);
        }
    }
    rope
}

/// Sets one algorithm-specific field to a value every reader accepts.
fn set_field(rope: &mut RopeParameters, field: AlgorithmField) {
    match field {
        AlgorithmField::Factor => rope.factor = Some(2.0),
        AlgorithmField::OriginalMaxPositionEmbeddings => {
            rope.original_max_position_embeddings = Some(MAX);
        }
        AlgorithmField::AttentionFactor => rope.attention_factor = Some(1.25),
        AlgorithmField::BetaFast => rope.beta_fast = Some(24.0),
        AlgorithmField::BetaSlow => rope.beta_slow = Some(2.0),
        AlgorithmField::Mscale => rope.mscale = Some(1.0),
        AlgorithmField::MscaleAllDim => rope.mscale_all_dim = Some(1.0),
        AlgorithmField::Truncate => rope.truncate = Some(false),
        AlgorithmField::ShortFactor => rope.short_factor = Some(vec![1.0; HEAD / 2]),
        AlgorithmField::LongFactor => rope.long_factor = Some(vec![2.0; HEAD / 2]),
        AlgorithmField::LowFreqFactor => rope.low_freq_factor = Some(1.0),
        AlgorithmField::HighFreqFactor => rope.high_freq_factor = Some(4.0),
    }
}

/// All algorithms in serialized-schema order.
const ROPE_TYPES: [RopeType; 7] = [
    RopeType::Default,
    RopeType::Linear,
    RopeType::Dynamic,
    RopeType::Yarn,
    RopeType::LongRope,
    RopeType::Llama3,
    RopeType::Proportional,
];

/// Every algorithm rejects exactly the fields it does not read, unless told to ignore them.
#[test]
fn every_unread_field_is_rejected_or_ignored_by_policy() {
    let lenient = context().with_unused_fields(UnusedRopeFields::Ignore);
    let mut rejected = 0;
    for rope_type in ROPE_TYPES {
        let base = minimal(rope_type);
        let reference = base.to_rope_config(&context()).expect("minimal parameters");
        for algorithm_field in AlgorithmField::ALL {
            if algorithm_field.is_present(&base) {
                continue;
            }
            let mut rope = base.clone();
            set_field(&mut rope, algorithm_field);
            assert!(algorithm_field.is_present(&rope));
            let label = format!("{rope_type:?} {}", algorithm_field.name());
            if algorithm_field.is_read_by(rope_type, SoftmaxMscale::Unscaled) {
                if matches!(
                    algorithm_field,
                    AlgorithmField::Mscale | AlgorithmField::MscaleAllDim
                ) {
                    // Alone, either key is unused by the Hugging Face amplitude.
                    let error = rope.to_rope_config(&context()).expect_err(&label);
                    assert_eq!(
                        field(&error),
                        format!("test.rope.{}", algorithm_field.name()),
                        "{label}"
                    );
                    assert!(error.to_string().contains("only together"), "{error}");
                    assert_eq!(
                        rope.to_rope_config(&lenient).expect(&label),
                        reference,
                        "{label}"
                    );
                } else {
                    rope.to_rope_config(&context()).expect(&label);
                }
            } else {
                let error = rope.to_rope_config(&context()).expect_err(&label);
                assert_eq!(
                    field(&error),
                    format!("test.rope.{}", algorithm_field.name()),
                    "{label}"
                );
                assert!(error.to_string().contains("is not read by"), "{error}");
                assert_eq!(
                    rope.to_rope_config(&lenient).expect(&label),
                    reference,
                    "{label}: an ignored field must not change the result"
                );
                rejected += 1;
            }
        }
    }
    // 12 fields x 7 algorithms minus the 21 (algorithm, field) pairs that are read.
    assert_eq!(rejected, 63);
}

/// An odd whole-head width is reported at `head_dim`, an odd strict prefix at the partial factor.
#[test]
fn odd_rotary_widths_name_the_field_to_repair() {
    let odd = RopeContext::new("test", 15, MAX);
    rejects(&RopeParameters::default(), &odd, "test.head_dim");
    rejects(
        &RopeParameters::default(),
        &odd.with_partial_rotary(PartialRotary::FullHeadOnly),
        "test.head_dim",
    );
    rejects(
        &params(r#"{"rope_type": "proportional", "partial_rotary_factor": 0.5}"#),
        &odd,
        "test.head_dim",
    );
    rejects(
        &params(r#"{"partial_rotary_factor": 0.5}"#),
        &RopeContext::new("test", 30, MAX),
        "test.rope.partial_rotary_factor",
    );
    // `validate_for` names the same fields.
    let validated = |rope: RopeParameters, head_dim| {
        rope.validate_for("test", head_dim, MAX, None)
            .unwrap_err()
            .field
    };
    assert_eq!(validated(RopeParameters::default(), 15), "head_dim");
    assert_eq!(validated(RopeParameters::default(), 1), "head_dim");
    assert_eq!(
        validated(params(r#"{"partial_rotary_factor": 0.5}"#), 30),
        "partial_rotary_factor"
    );
    assert_eq!(
        validated(params(r#"{"partial_rotary_factor": 0.9}"#), 3),
        "partial_rotary_factor"
    );
    rejects(
        &RopeParameters::default(),
        &RopeContext::new("test", 1, MAX),
        "test.head_dim",
    );
}

/// Multimodal sections are rejected unless the caller declares it consumes them.
#[test]
fn multimodal_sections_require_explicit_handling() {
    let qwen = params(r#"{"mrope_section": [2, 1, 1], "mrope_interleaved": true}"#);
    rejects(&qwen, &context(), "test.rope.mrope_section");
    rejects(
        &params(r#"{"mrope_interleaved": false}"#),
        &context(),
        "test.rope.mrope_interleaved",
    );
    let accepted = context().with_mrope(MropeHandling::TextPositions);
    assert_eq!(
        qwen.to_rope_config(&accepted).unwrap(),
        RopeParameters::default()
            .to_rope_config(&context())
            .unwrap()
    );
    qwen.to_rope_config(&context().with_unused_fields(UnusedRopeFields::Ignore))
        .unwrap();
}

/// Every construction error maps to the field a config author can repair.
#[test]
fn construction_errors_map_to_repairable_fields() {
    let binding = context();
    let scope = Scope {
        context: &binding,
        prefix: "rope".to_owned(),
        per_layer_type: false,
    };
    let partial = "test.rope.partial_rotary_factor";
    let cases = [
        (RopeConfigError::ZeroHeadDimension, "test.head_dim"),
        (RopeConfigError::ZeroRotaryDimension, partial),
        (
            RopeConfigError::RotaryDimensionExceedsHead {
                rotary_dimension: 4,
                head_dimension: 2,
            },
            partial,
        ),
        (
            RopeConfigError::OddRotaryDimension {
                rotary_dimension: 3,
            },
            partial,
        ),
        (
            RopeConfigError::OddRotaryDimension {
                rotary_dimension: HEAD,
            },
            "test.head_dim",
        ),
        (
            RopeConfigError::DynamicNtkDimensionTooSmall {
                rotary_dimension: HEAD,
            },
            "test.head_dim",
        ),
        (
            RopeConfigError::InvalidPartialRotaryFactor { factor: 2.0 },
            partial,
        ),
        (
            RopeConfigError::NonIntegralPartialRotaryDimension {
                head_dimension: 8,
                factor: 0.3,
                computed: 2.4,
            },
            partial,
        ),
        (
            RopeConfigError::DynamicNtkDimensionTooSmall {
                rotary_dimension: 2,
            },
            partial,
        ),
        (
            RopeConfigError::ProportionalRequiresFullRotation {
                rotary_dimension: 4,
                head_dimension: 8,
            },
            partial,
        ),
        (
            RopeConfigError::InvalidProportionalFraction { fraction: 2.0 },
            partial,
        ),
        (
            RopeConfigError::InvalidTheta { theta: -1.0 },
            "test.rope.rope_theta",
        ),
        (
            RopeConfigError::YarnThetaTooSmall { theta: 1.0 },
            "test.rope.rope_theta",
        ),
        (
            RopeConfigError::InvalidScalingFactor {
                kind: "linear",
                factor: 0.5,
            },
            "test.rope.factor",
        ),
        (
            RopeConfigError::ZeroOriginalContext { kind: "YaRN" },
            "test.rope.original_max_position_embeddings",
        ),
        (
            RopeConfigError::InvalidYarnAttentionFactor {
                attention_factor: 0.0,
            },
            "test.rope.attention_factor",
        ),
        (
            RopeConfigError::InvalidLongRopeAttentionFactor {
                attention_factor: 0.0,
            },
            "test.rope.attention_factor",
        ),
        (
            RopeConfigError::InvalidYarnBeta {
                name: "beta_slow",
                value: 0.0,
            },
            "test.rope.beta_slow",
        ),
        (
            RopeConfigError::InvalidYarnBetaOrder {
                beta_fast: 1.0,
                beta_slow: 2.0,
            },
            "test.rope.beta_fast",
        ),
        (
            RopeConfigError::InvalidLlama3FrequencyFactors {
                low_freq_factor: 4.0,
                high_freq_factor: 1.0,
            },
            "test.rope.low_freq_factor",
        ),
        (
            RopeConfigError::InvalidLongRopeFactor {
                table: "long_factor",
                index: 0,
                value: 0.0,
            },
            "test.rope.long_factor",
        ),
        (
            RopeConfigError::LongRopeFactorLengthMismatch { short: 1, long: 2 },
            "test.rope.long_factor",
        ),
        (
            RopeConfigError::LongRopeFactorCountOutOfRange {
                count: 0,
                maximum: 128,
            },
            "test.rope.short_factor",
        ),
        (
            RopeConfigError::LongRopeFactorCount {
                expected: 4,
                actual: 2,
            },
            "test.rope.short_factor",
        ),
    ];
    for (error, expected) in cases {
        let mapped = scope.rope_error(error);
        assert_eq!(field(&mapped), expected, "{error:?}");
        assert!(mapped.to_string().contains(&error.to_string()), "{mapped}");
    }
}

// ---------------------------------------------------------------------------
// Parity with the private family translators.
//
// The recommended family contexts below follow each family's reference
// (Transformers 5.2, or DeepSeek's `modeling_deepseek.py`). They equal the
// frozen private translators on every pinned checkpoint and on the families'
// own test configs, except for these intended behavior changes, each
// asserted below:
//
// * A required `factor` that is absent is an error (it defaulted to one).
// * Dynamic NTK scales against `max_position_embeddings`, so a nested
//   `original_max_position_embeddings` that differs from it is an unused
//   field; `DynamicNtkContext::OriginalMaxPositionEmbeddings` restores the
//   legacy rule exactly.
// * Qwen and Muse YaRN read `mscale`/`mscale_all_dim` with the Hugging Face
//   rule (the ratio when both are nonzero; otherwise the keys are unused)
//   and honor `truncate: false`; the private translators ignored all three.
// * DeepSeek rejects `attention_factor` beside a `mscale` key (it was
//   dropped) and applies the softmax correction to every scaled algorithm,
//   as the reference does (it was YaRN-only).
// * Unused fields and disagreeing sources are errors (`Ignore` restores the
//   Transformers resolution), and LongRoPE, Llama 3, and proportional RoPE
//   are translated instead of rejected.
// ---------------------------------------------------------------------------

mod frozen {
    //! Verbatim copies of the private family translators at commit `d20896a`.
    //!
    //! The shared translator replaces `qwen3_5::model::qwen_rope_config`,
    //! `qwen3_moe::model::qwen_rope_config`,
    //! `deepseek_v2::model::{deepseek_rope_config, deepseek_attention_scale}`,
    //! and `muse_glimmer::model::muse_rope_config`. They are private to their
    //! families, so these frozen oracles keep the pre-adoption behavior
    //! testable after the families switch to the shared code.

    use crate::models::config::RopeType;
    use crate::models::deepseek_v2::DeepseekV2Config;
    use crate::models::muse_glimmer::MuseGlimmerTextConfig;
    use crate::models::qwen3_5::Qwen3_5TextConfig;
    use crate::models::qwen3_moe::Qwen3MoeConfig;
    use crate::position::{
        DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling, YarnConfig,
    };
    use crate::{Error, Result};

    /// Frozen `qwen3_5::model::qwen_rope_config`.
    pub(super) fn qwen3_5(config: &Qwen3_5TextConfig) -> Result<RopeConfig> {
        let parameters = config.rope_parameters.clone().unwrap_or_default();
        let theta = parameters
            .rope_theta
            .or(config.rope_theta)
            .unwrap_or(10_000.0);
        let partial = parameters
            .partial_rotary_factor
            .or(config.partial_rotary_factor)
            .unwrap_or(1.0);
        let scaling = match parameters.rope_type {
            RopeType::Default => RopeScaling::None,
            RopeType::Linear => LinearRopeScaling::new(parameters.factor.unwrap_or(1.0))
                .map(RopeScaling::from)
                .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?,
            RopeType::Dynamic => DynamicNtkScaling::new(
                parameters.factor.unwrap_or(1.0),
                parameters
                    .original_max_position_embeddings
                    .unwrap_or(config.max_position_embeddings),
            )
            .map(RopeScaling::from)
            .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?,
            RopeType::Yarn => {
                let mut yarn = YarnConfig::new(
                    parameters.factor.unwrap_or(1.0),
                    parameters
                        .original_max_position_embeddings
                        .unwrap_or(config.max_position_embeddings),
                )
                .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?;
                if let Some(factor) = parameters.attention_factor {
                    yarn = yarn
                        .with_attention_factor(factor)
                        .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?;
                }
                yarn = yarn
                    .with_beta_overrides(parameters.beta_fast, parameters.beta_slow)
                    .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?;
                RopeScaling::Yarn(yarn)
            }
            unsupported => {
                return Err(Error::configuration(
                    "qwen3_5.rope_type",
                    format!("{unsupported:?} is not implemented for the Qwen text runtime"),
                ));
            }
        };
        RopeConfig::from_partial_factor(config.head_dim, partial, theta, scaling)
            .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))
    }

    /// Frozen `qwen3_moe::model::qwen_rope_config`.
    pub(super) fn qwen3_moe(config: &Qwen3MoeConfig) -> Result<RopeConfig> {
        let parameters = config.rope_parameters.clone().unwrap_or_default();
        let theta = parameters
            .rope_theta
            .or(config.rope_theta)
            .unwrap_or(10_000.0);
        let scaling = match parameters.rope_type {
            RopeType::Default => RopeScaling::None,
            RopeType::Linear => LinearRopeScaling::new(parameters.factor.unwrap_or(1.0))
                .map(RopeScaling::from)
                .map_err(|error| Error::configuration("qwen3_moe.rope", error.to_string()))?,
            RopeType::Dynamic => DynamicNtkScaling::new(
                parameters.factor.unwrap_or(1.0),
                parameters
                    .original_max_position_embeddings
                    .unwrap_or(config.max_position_embeddings),
            )
            .map(RopeScaling::from)
            .map_err(|error| Error::configuration("qwen3_moe.rope", error.to_string()))?,
            RopeType::Yarn => {
                let mut yarn = YarnConfig::new(
                    parameters.factor.unwrap_or(1.0),
                    parameters
                        .original_max_position_embeddings
                        .unwrap_or(config.max_position_embeddings),
                )
                .map_err(|error| Error::configuration("qwen3_moe.rope", error.to_string()))?;
                if let Some(factor) = parameters.attention_factor {
                    yarn = yarn.with_attention_factor(factor).map_err(|error| {
                        Error::configuration("qwen3_moe.rope", error.to_string())
                    })?;
                }
                yarn = yarn
                    .with_beta_overrides(parameters.beta_fast, parameters.beta_slow)
                    .map_err(|error| Error::configuration("qwen3_moe.rope", error.to_string()))?;
                RopeScaling::Yarn(yarn)
            }
            unsupported => {
                return Err(Error::configuration(
                    "qwen3_moe.rope_type",
                    format!("{unsupported:?} is not implemented for Qwen3 MoE"),
                ));
            }
        };
        RopeConfig::new(config.head_dim, config.head_dim, theta, scaling)
            .map_err(|error| Error::configuration("qwen3_moe.rope", error.to_string()))
    }

    /// Frozen `deepseek_v2::model::deepseek_rope_config`.
    pub(super) fn deepseek_v2(config: &DeepseekV2Config) -> Result<RopeConfig> {
        let parameters = config.rope_scaling.clone().unwrap_or_default();
        let theta = parameters.rope_theta.unwrap_or(config.rope_theta);
        let scaling = match parameters.rope_type {
            RopeType::Default => RopeScaling::None,
            RopeType::Linear => LinearRopeScaling::new(parameters.factor.unwrap_or(1.0))
                .map(RopeScaling::from)
                .map_err(|error| Error::configuration("deepseek_v2.rope", error.to_string()))?,
            RopeType::Dynamic => DynamicNtkScaling::new(
                parameters.factor.unwrap_or(1.0),
                parameters
                    .original_max_position_embeddings
                    .unwrap_or(config.max_position_embeddings),
            )
            .map(RopeScaling::from)
            .map_err(|error| Error::configuration("deepseek_v2.rope", error.to_string()))?,
            RopeType::Yarn => {
                let factor = parameters.factor.unwrap_or(1.0);
                let mut yarn = YarnConfig::new(
                    factor,
                    parameters
                        .original_max_position_embeddings
                        .unwrap_or(config.max_position_embeddings),
                )
                .map_err(|error| Error::configuration("deepseek_v2.rope", error.to_string()))?;
                let amplitude =
                    if parameters.mscale.is_some() || parameters.mscale_all_dim.is_some() {
                        yarn_mscale(factor, parameters.mscale.unwrap_or(1.0))
                            / yarn_mscale(factor, parameters.mscale_all_dim.unwrap_or(0.0))
                    } else {
                        parameters
                            .attention_factor
                            .unwrap_or_else(|| yarn.attention_factor())
                    };
                yarn = yarn
                    .with_attention_factor(amplitude)
                    .map_err(|error| Error::configuration("deepseek_v2.rope", error.to_string()))?;
                yarn = yarn
                    .with_beta_overrides(parameters.beta_fast, parameters.beta_slow)
                    .map_err(|error| Error::configuration("deepseek_v2.rope", error.to_string()))?;
                RopeScaling::Yarn(yarn)
            }
            unsupported => {
                return Err(Error::configuration(
                    "deepseek_v2.rope_type",
                    format!("{unsupported:?} is not implemented for DeepSeek-V2"),
                ));
            }
        };
        RopeConfig::new(
            config.qk_rope_head_dim,
            config.qk_rope_head_dim,
            theta,
            scaling,
        )
        .map_err(|error| Error::configuration("deepseek_v2.rope", error.to_string()))
    }

    /// Frozen `deepseek_v2::model::yarn_mscale`.
    fn yarn_mscale(factor: f64, multiplier: f64) -> f64 {
        if factor <= 1.0 {
            1.0
        } else {
            1.0 + 0.1 * multiplier * factor.ln()
        }
    }

    /// Frozen `deepseek_v2::model::deepseek_attention_scale`.
    pub(super) fn deepseek_attention_scale(config: &DeepseekV2Config) -> Result<f64> {
        let mut scale = (config.qk_head_dim() as f64).sqrt().recip();
        if let Some(parameters) = &config.rope_scaling
            && parameters.rope_type == RopeType::Yarn
            && let Some(multiplier) = parameters.mscale_all_dim
            && multiplier != 0.0
        {
            let factor = parameters.factor.ok_or_else(|| {
                Error::configuration("deepseek_v2.rope.factor", "is required for YaRN")
            })?;
            scale *= yarn_mscale(factor, multiplier).powi(2);
        }
        Ok(scale)
    }

    /// Frozen `muse_glimmer::model::muse_rope_config`.
    pub(super) fn muse_glimmer(config: &MuseGlimmerTextConfig, theta: f64) -> Result<RopeConfig> {
        let parameters = config.rope_parameters.clone().unwrap_or_default();
        let scaling = match parameters.rope_type {
            RopeType::Default => RopeScaling::None,
            RopeType::Linear => LinearRopeScaling::new(parameters.factor.unwrap_or(1.0))
                .map(RopeScaling::from)
                .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))?,
            RopeType::Dynamic => DynamicNtkScaling::new(
                parameters.factor.unwrap_or(1.0),
                parameters
                    .original_max_position_embeddings
                    .unwrap_or(config.max_position_embeddings),
            )
            .map(RopeScaling::from)
            .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))?,
            RopeType::Yarn => {
                let mut yarn = YarnConfig::new(
                    parameters.factor.unwrap_or(1.0),
                    parameters
                        .original_max_position_embeddings
                        .unwrap_or(config.max_position_embeddings),
                )
                .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))?;
                if let Some(factor) = parameters.attention_factor {
                    yarn = yarn.with_attention_factor(factor).map_err(|error| {
                        Error::configuration("muse_glimmer.rope", error.to_string())
                    })?;
                }
                yarn = yarn
                    .with_beta_overrides(parameters.beta_fast, parameters.beta_slow)
                    .map_err(|error| {
                        Error::configuration("muse_glimmer.rope", error.to_string())
                    })?;
                RopeScaling::Yarn(yarn)
            }
            unsupported => {
                return Err(Error::configuration(
                    "muse_glimmer.rope_type",
                    format!("{unsupported:?} is not implemented for Muse-Glimmer"),
                ));
            }
        };
        RopeConfig::new(config.head_dim, config.head_dim, theta, scaling)
            .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))
    }
}

/// Recommended Qwen3.5 context: the Transformers rules with text-only M-RoPE.
fn qwen3_5_context(config: &Qwen3_5TextConfig) -> RopeContext {
    RopeContext::new("qwen3_5", config.head_dim, config.max_position_embeddings)
        .with_rope_theta(config.rope_theta)
        .with_partial_rotary_factor(config.partial_rotary_factor)
        .with_mrope(MropeHandling::TextPositions)
}

/// Shared replacement for `qwen3_5::model::qwen_rope_config` under `context`.
fn shared_qwen3_5_with(config: &Qwen3_5TextConfig, context: &RopeContext) -> Result<RopeConfig> {
    config
        .rope_parameters
        .clone()
        .unwrap_or_default()
        .to_rope_config(context)
}

/// Shared replacement for `qwen3_5::model::qwen_rope_config`.
fn shared_qwen3_5(config: &Qwen3_5TextConfig) -> Result<RopeConfig> {
    shared_qwen3_5_with(config, &qwen3_5_context(config))
}

/// Recommended Qwen3-MoE context: the Transformers rules for whole heads.
fn qwen3_moe_context(config: &Qwen3MoeConfig) -> RopeContext {
    RopeContext::new("qwen3_moe", config.head_dim, config.max_position_embeddings)
        .with_rope_theta(config.rope_theta)
        .with_partial_rotary(PartialRotary::FullHeadOnly)
}

/// Shared replacement for `qwen3_moe::model::qwen_rope_config` under `context`.
fn shared_qwen3_moe_with(config: &Qwen3MoeConfig, context: &RopeContext) -> Result<RopeConfig> {
    config
        .rope_parameters
        .clone()
        .unwrap_or_default()
        .to_rope_config(context)
}

/// Shared replacement for `qwen3_moe::model::qwen_rope_config`.
fn shared_qwen3_moe(config: &Qwen3MoeConfig) -> Result<RopeConfig> {
    shared_qwen3_moe_with(config, &qwen3_moe_context(config))
}

/// Recommended DeepSeek-V2 context: the reference amplitude and softmax correction.
fn deepseek_context(config: &DeepseekV2Config) -> RopeContext {
    RopeContext::new(
        "deepseek_v2",
        config.qk_rope_head_dim,
        config.max_position_embeddings,
    )
    .with_rope_theta(Some(config.rope_theta))
    .with_partial_rotary(PartialRotary::FullHeadOnly)
    .with_yarn_mscale_convention(YarnMscaleConvention::DeepSeekReference)
    .with_softmax_mscale(SoftmaxMscale::MscaleAllDimSquared)
}

/// Shared replacement for `deepseek_v2::model::deepseek_rope_config` under `context`.
fn shared_deepseek_v2_with(config: &DeepseekV2Config, context: &RopeContext) -> Result<RopeConfig> {
    config
        .rope_scaling
        .clone()
        .unwrap_or_default()
        .to_rope_config(context)
}

/// Shared replacement for `deepseek_v2::model::deepseek_rope_config`.
fn shared_deepseek_v2(config: &DeepseekV2Config) -> Result<RopeConfig> {
    shared_deepseek_v2_with(config, &deepseek_context(config))
}

/// Shared replacement for `deepseek_v2::model::deepseek_attention_scale`.
fn shared_deepseek_attention_scale(config: &DeepseekV2Config) -> Result<f64> {
    let multiplier = config
        .rope_scaling
        .clone()
        .unwrap_or_default()
        .softmax_scale_multiplier(&deepseek_context(config))?;
    Ok((config.qk_head_dim() as f64).sqrt().recip() * multiplier)
}

/// Recommended Muse Glimmer context for one rotary layer of base `theta`.
fn muse_context(config: &MuseGlimmerTextConfig, theta: f64) -> RopeContext {
    RopeContext::new(
        "muse_glimmer",
        config.head_dim,
        config.max_position_embeddings,
    )
    .with_layer_rope_theta(theta)
    .with_partial_rotary(PartialRotary::FullHeadOnly)
}

/// Shared replacement for `muse_glimmer::model::muse_rope_config` under `context`.
fn shared_muse_glimmer_with(
    config: &MuseGlimmerTextConfig,
    context: &RopeContext,
) -> Result<RopeConfig> {
    config
        .rope_parameters
        .clone()
        .unwrap_or_default()
        .to_rope_config(context)
}

/// Shared replacement for `muse_glimmer::model::muse_rope_config`.
fn shared_muse_glimmer(config: &MuseGlimmerTextConfig, theta: f64) -> Result<RopeConfig> {
    shared_muse_glimmer_with(config, &muse_context(config, theta))
}

/// Adds the legacy dynamic-NTK rule that reproduces the private translators exactly.
fn legacy_dynamic(context: RopeContext) -> RopeContext {
    context.with_dynamic_ntk_context(DynamicNtkContext::OriginalMaxPositionEmbeddings)
}

/// Pinned checkpoint fixtures translate identically through both paths.
#[test]
fn shared_translator_reproduces_private_translators_on_pinned_fixtures() {
    for json in [
        include_str!("../../../tests/fixtures/qwen3_5_0_8b.json"),
        include_str!("../../../tests/fixtures/qwen3_8_27b.json"),
    ] {
        let config = Qwen3_5TextConfig::from_json_str(json).unwrap();
        assert_eq!(
            shared_qwen3_5(&config).unwrap(),
            frozen::qwen3_5(&config).unwrap()
        );
    }
    let config =
        Qwen3MoeConfig::from_json_str(include_str!("../../../tests/fixtures/qwen3_30b_a3b.json"))
            .unwrap();
    assert_eq!(
        shared_qwen3_moe(&config).unwrap(),
        frozen::qwen3_moe(&config).unwrap()
    );
    for json in [
        include_str!("../../../tests/fixtures/deepseek_v2.json"),
        include_str!("../../../tests/fixtures/deepseek_v2_lite.json"),
    ] {
        let config = DeepseekV2Config::from_json_str(json).unwrap();
        assert_eq!(
            shared_deepseek_v2(&config).unwrap(),
            frozen::deepseek_v2(&config).unwrap()
        );
        assert_eq!(
            shared_deepseek_attention_scale(&config).unwrap(),
            frozen::deepseek_attention_scale(&config).unwrap()
        );
    }
    let mut muse_layers = 0;
    for json in [
        include_str!("../../../tests/fixtures/muse_glimmer.json"),
        include_str!("../../../tests/fixtures/muse_glimmer_30b.json"),
    ] {
        let config = MuseGlimmerTextConfig::from_json_str(json).unwrap();
        for theta in config.resolved_layer_rope_theta() {
            if theta > 0.0 {
                assert_eq!(
                    shared_muse_glimmer(&config, theta).unwrap(),
                    frozen::muse_glimmer(&config, theta).unwrap()
                );
                muse_layers += 1;
            }
        }
    }
    assert!(muse_layers > 0, "Muse fixtures contain rotary layers");
}

/// The synthetic configurations used by the families' own tests translate identically.
#[test]
fn shared_translator_reproduces_private_translators_on_family_test_configs() {
    // qwen3_5 model/recipe/intervention/decode tests.
    let qwen_tiny = |max_position_embeddings| Qwen3_5TextConfig {
        head_dim: 4,
        max_position_embeddings,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        ..Qwen3_5TextConfig::default()
    };
    for config in [Qwen3_5TextConfig::default(), qwen_tiny(64), qwen_tiny(32)] {
        assert_eq!(
            shared_qwen3_5(&config).unwrap(),
            frozen::qwen3_5(&config).unwrap()
        );
    }
    // The decode test's dynamic NTK keeps a nested trained context of two
    // positions below max_position_embeddings = 64. Transformers scales
    // against 64, so the recommended context reports the unread value; the
    // legacy rule reproduces the private translator's limit of two.
    let mut dynamic = qwen_tiny(64);
    dynamic.rope_parameters = Some(RopeParameters {
        rope_type: RopeType::Dynamic,
        factor: Some(2.0),
        original_max_position_embeddings: Some(2),
        ..RopeParameters::default()
    });
    let error = shared_qwen3_5(&dynamic).unwrap_err();
    assert_eq!(
        field(&error),
        "qwen3_5.rope.original_max_position_embeddings"
    );
    let legacy = shared_qwen3_5_with(&dynamic, &legacy_dynamic(qwen3_5_context(&dynamic))).unwrap();
    assert_eq!(legacy, frozen::qwen3_5(&dynamic).unwrap());
    assert_eq!(legacy.scaling().cached_context_limit(), Some(2));
    let transformers = shared_qwen3_5_with(
        &dynamic,
        &qwen3_5_context(&dynamic).with_unused_fields(UnusedRopeFields::Ignore),
    )
    .unwrap();
    assert_eq!(transformers.scaling().cached_context_limit(), Some(64));

    // qwen3_moe model/checkpoint/load tests and the mixed-projection fixtures.
    let moe = |head_dim, max_position_embeddings, rope_theta| Qwen3MoeConfig {
        head_dim,
        max_position_embeddings,
        rope_theta,
        ..Qwen3MoeConfig::default()
    };
    for config in [
        Qwen3MoeConfig::default(),
        moe(4, 32, Some(1_000_000.0)),
        moe(2, 16, Some(10_000.0)),
        moe(4, 32, Some(10_000.0)),
    ] {
        assert_eq!(
            shared_qwen3_moe(&config).unwrap(),
            frozen::qwen3_moe(&config).unwrap()
        );
    }
    for rope_type in [RopeType::Linear, RopeType::Yarn] {
        let mut mixed = moe(4, 32, Some(10_000.0));
        mixed.rope_parameters = Some(RopeParameters {
            rope_type,
            rope_theta: Some(100.0),
            factor: Some(2.0),
            original_max_position_embeddings: Some(8),
            attention_factor: (rope_type == RopeType::Yarn).then_some(1.25),
            beta_fast: (rope_type == RopeType::Yarn).then_some(4.0),
            beta_slow: (rope_type == RopeType::Yarn).then_some(1.0),
            ..RopeParameters::default()
        });
        // The linear fixture's unread original context and the nested theta
        // overriding the top-level one are rejected; `Ignore` resolves them
        // as Transformers does, which is also what the private translator did.
        let error = shared_qwen3_moe(&mixed).unwrap_err();
        let expected = if rope_type == RopeType::Linear {
            "qwen3_moe.rope.original_max_position_embeddings"
        } else {
            "qwen3_moe.rope.rope_theta"
        };
        assert_eq!(field(&error), expected);
        let lenient = qwen3_moe_context(&mixed).with_unused_fields(UnusedRopeFields::Ignore);
        assert_eq!(
            shared_qwen3_moe_with(&mixed, &lenient).unwrap(),
            frozen::qwen3_moe(&mixed).unwrap()
        );
    }

    // deepseek_v2 tiny config and the one-sided beta override test.
    let mut deepseek = DeepseekV2Config {
        qk_rope_head_dim: 2,
        max_position_embeddings: 32,
        rope_theta: 10_000.0,
        rope_scaling: None,
        ..DeepseekV2Config::default()
    };
    let mut deepseek_configs = vec![DeepseekV2Config::default(), deepseek.clone()];
    for (beta_fast, beta_slow) in [(Some(24.0), None), (None, Some(2.0))] {
        deepseek.rope_scaling = Some(RopeParameters {
            rope_type: RopeType::Yarn,
            factor: Some(2.0),
            original_max_position_embeddings: Some(16),
            beta_fast,
            beta_slow,
            ..RopeParameters::default()
        });
        deepseek_configs.push(deepseek.clone());
    }
    for config in &deepseek_configs {
        assert_eq!(
            shared_deepseek_v2(config).unwrap(),
            frozen::deepseek_v2(config).unwrap()
        );
        assert_eq!(
            shared_deepseek_attention_scale(config).unwrap(),
            frozen::deepseek_attention_scale(config).unwrap()
        );
    }

    // muse_glimmer tiny configs and the 32K stress configuration.
    let muse = |head_dim, max_position_embeddings| MuseGlimmerTextConfig {
        head_dim,
        max_position_embeddings,
        ..MuseGlimmerTextConfig::default()
    };
    for config in [
        MuseGlimmerTextConfig::default(),
        muse(4, 32),
        muse(2, 16),
        muse(4, 32_768),
    ] {
        for theta in [500_000.0, 10_000.0] {
            assert_eq!(
                shared_muse_glimmer(&config, theta).unwrap(),
                frozen::muse_glimmer(&config, theta).unwrap()
            );
        }
    }
}

/// Algorithms the private translators rejected are now translated, not silently dropped.
#[test]
fn shared_translator_extends_private_translators_to_every_algorithm() {
    let mut config = Qwen3_5TextConfig {
        head_dim: HEAD,
        max_position_embeddings: MAX,
        partial_rotary_factor: None,
        ..Qwen3_5TextConfig::default()
    };
    for rope_type in [RopeType::LongRope, RopeType::Llama3, RopeType::Proportional] {
        config.rope_parameters = Some(minimal(rope_type));
        assert!(frozen::qwen3_5(&config).is_err(), "{rope_type:?}");
        assert!(shared_qwen3_5(&config).is_ok(), "{rope_type:?}");
    }
}

/// The documented YaRN divergences, at the reviewer's probe values.
///
/// With factor 4, `mscale = 1` and `mscale_all_dim = 0.5`, Transformers 5.2's
/// `_compute_yarn_parameters` gives `m(4, 1) / m(4, 0.5) = 1.0648`, while the
/// private translators kept `m(4, 1) = 1.1386`; `truncate: false` is now
/// honored and a missing `factor` is reported.
#[test]
fn shared_translator_documents_its_yarn_divergences() {
    let config = |rope: RopeParameters| Qwen3MoeConfig {
        head_dim: HEAD,
        max_position_embeddings: MAX,
        rope_theta: None,
        rope_parameters: Some(rope),
        ..Qwen3MoeConfig::default()
    };
    let ratio = config(RopeParameters {
        rope_type: RopeType::Yarn,
        factor: Some(4.0),
        mscale: Some(1.0),
        mscale_all_dim: Some(0.5),
        ..RopeParameters::default()
    });
    let shared = yarn_of(shared_qwen3_moe(&ratio).unwrap()).attention_factor();
    let frozen = yarn_of(frozen::qwen3_moe(&ratio).unwrap()).attention_factor();
    assert_eq!(shared, yarn_mscale(4.0, 1.0) / yarn_mscale(4.0, 0.5));
    assert!((shared - 1.064_822).abs() < 1e-6, "{shared}");
    assert!((frozen - 1.138_629).abs() < 1e-6, "{frozen}");

    let untruncated = config(RopeParameters {
        rope_type: RopeType::Yarn,
        factor: Some(4.0),
        truncate: Some(false),
        ..RopeParameters::default()
    });
    assert!(!yarn_of(shared_qwen3_moe(&untruncated).unwrap()).truncate());
    assert!(yarn_of(frozen::qwen3_moe(&untruncated).unwrap()).truncate());

    let missing = config(RopeParameters {
        rope_type: RopeType::Linear,
        ..RopeParameters::default()
    });
    assert_eq!(
        field(&shared_qwen3_moe(&missing).unwrap_err()),
        "qwen3_moe.rope.factor"
    );
    assert!(frozen::qwen3_moe(&missing).is_ok());

    // DeepSeek's reference applies the softmax correction to linear RoPE too.
    let linear = DeepseekV2Config {
        rope_scaling: Some(RopeParameters {
            rope_type: RopeType::Linear,
            factor: Some(4.0),
            mscale_all_dim: Some(1.0),
            ..RopeParameters::default()
        }),
        ..DeepseekV2Config::default()
    };
    let base = (linear.qk_head_dim() as f64).sqrt().recip();
    assert_eq!(
        shared_deepseek_attention_scale(&linear).unwrap(),
        base * yarn_mscale(4.0, 1.0).powi(2)
    );
    assert_eq!(frozen::deepseek_attention_scale(&linear).unwrap(), base);
    assert_eq!(
        shared_deepseek_v2(&linear).unwrap(),
        frozen::deepseek_v2(&linear).unwrap()
    );
}

/// Strategy for a family scaling setup: `(type, factor, nested original, theta)`.
///
/// `factor` is sometimes absent, and the nested original context often equals
/// the families' `max_position_embeddings` (256), so both the success and the
/// documented-divergence paths are exercised.
fn scaled_parameters() -> impl Strategy<Value = (RopeType, Option<f64>, usize, f64)> {
    (
        prop_oneof![
            Just(RopeType::Default),
            Just(RopeType::Linear),
            Just(RopeType::Dynamic),
            Just(RopeType::Yarn),
        ],
        prop_oneof![1 => Just(None), 6 => (1.0f64..64.0).prop_map(Some)],
        prop_oneof![Just(FAMILY_MAX), 1usize..=FAMILY_MAX],
        1.5f64..1.0e7,
    )
}

/// `max_position_embeddings` of the property-test family configs.
const FAMILY_MAX: usize = 256;

/// YaRN-only extras of a generated family config.
#[derive(Debug, Clone, Copy, Default)]
struct YarnExtras {
    /// `attention_factor`.
    attention_factor: Option<f64>,
    /// `beta_fast`.
    beta_fast: Option<f64>,
    /// `beta_slow`.
    beta_slow: Option<f64>,
    /// `mscale`.
    mscale: Option<f64>,
    /// `mscale_all_dim`.
    mscale_all_dim: Option<f64>,
    /// `truncate`.
    truncate: Option<bool>,
}

/// Strategy for [`YarnExtras`] with valid betas and occasionally zero `mscale` keys.
fn yarn_extras() -> impl Strategy<Value = YarnExtras> {
    let mscale = || proptest::option::of(prop_oneof![1 => Just(0.0), 4 => 0.01f64..2.0]);
    (
        proptest::option::of(0.5f64..2.0),
        proptest::option::of(8.0f64..64.0),
        proptest::option::of(0.5f64..4.0),
        mscale(),
        mscale(),
        proptest::option::of(any::<bool>()),
    )
        .prop_map(
            |(attention_factor, beta_fast, beta_slow, mscale, mscale_all_dim, truncate)| {
                YarnExtras {
                    attention_factor,
                    beta_fast,
                    beta_slow,
                    mscale,
                    mscale_all_dim,
                    truncate,
                }
            },
        )
}

/// Builds parameters carrying only the fields `rope_type` reads; YaRN keeps `extras`.
fn family_parameters(
    rope_type: RopeType,
    factor: Option<f64>,
    original: usize,
    theta: f64,
    extras: YarnExtras,
) -> RopeParameters {
    let yarn = rope_type == RopeType::Yarn;
    let keep = |value: Option<f64>| value.filter(|_| yarn);
    RopeParameters {
        rope_type,
        rope_theta: Some(theta),
        factor: factor.filter(|_| rope_type != RopeType::Default),
        original_max_position_embeddings: matches!(rope_type, RopeType::Dynamic | RopeType::Yarn)
            .then_some(original),
        attention_factor: keep(extras.attention_factor),
        beta_fast: keep(extras.beta_fast),
        beta_slow: keep(extras.beta_slow),
        mscale: keep(extras.mscale),
        mscale_all_dim: keep(extras.mscale_all_dim),
        truncate: extras.truncate.filter(|_| yarn),
        ..RopeParameters::default()
    }
}

/// What a recommended strict family context yields relative to its frozen translator.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(
    clippy::large_enum_variant,
    reason = "a test-only prediction built once per property case and compared by value"
)]
enum FamilyOutcome {
    /// Exactly the frozen translator's configuration.
    Frozen,
    /// A configuration error at `<family>.rope.<field>`.
    Error(&'static str),
    /// The Transformers result, which differs from the frozen YaRN translation
    /// in its amplitude, its truncation, or both.
    Transformers(RopeConfig),
}

/// Predicts the Qwen/Muse recommended-context outcome from the documented rules.
///
/// `head_dim` is the whole rotated head and `theta` the resolved base.
fn qwen_family_outcome(rope: &RopeParameters, head_dim: usize, theta: f64) -> FamilyOutcome {
    if rope.rope_type != RopeType::Default && rope.factor.is_none() {
        return FamilyOutcome::Error("factor");
    }
    if rope.rope_type == RopeType::Dynamic
        && rope.original_max_position_embeddings != Some(FAMILY_MAX)
    {
        return FamilyOutcome::Error("original_max_position_embeddings");
    }
    if rope.rope_type != RopeType::Yarn {
        return FamilyOutcome::Frozen;
    }
    let factor = rope.factor.expect("checked above");
    // The Hugging Face amplitude rule with an unscaled softmax, in the
    // translator's check order: an explicit amplitude wins and makes both
    // `mscale` keys unused; otherwise only a pair of nonzero keys is read.
    let nonzero = |value: Option<f64>| value.is_some_and(|value| value != 0.0);
    let amplitude = if let Some(attention_factor) = rope.attention_factor {
        if rope.mscale.is_some() {
            return FamilyOutcome::Error("mscale");
        }
        if rope.mscale_all_dim.is_some() {
            return FamilyOutcome::Error("mscale_all_dim");
        }
        Some(attention_factor)
    } else if nonzero(rope.mscale) && nonzero(rope.mscale_all_dim) {
        let ratio = |mscale: Option<f64>| yarn_mscale(factor, mscale.expect("nonzero"));
        Some(ratio(rope.mscale) / ratio(rope.mscale_all_dim))
    } else if rope.mscale.is_some() {
        return FamilyOutcome::Error("mscale");
    } else if rope.mscale_all_dim.is_some() {
        return FamilyOutcome::Error("mscale_all_dim");
    } else {
        None
    };
    let ratio = rope.attention_factor.is_none() && amplitude.is_some();
    let untruncated = rope.truncate == Some(false);
    if !ratio && !untruncated {
        return FamilyOutcome::Frozen;
    }
    let original = rope.original_max_position_embeddings.unwrap_or(FAMILY_MAX);
    let mut yarn = YarnConfig::new(factor, original).unwrap();
    if let Some(amplitude) = amplitude {
        yarn = yarn.with_attention_factor(amplitude).unwrap();
    }
    let yarn = yarn
        .with_beta_overrides(rope.beta_fast, rope.beta_slow)
        .unwrap()
        .with_truncate(!untruncated);
    FamilyOutcome::Transformers(RopeConfig::new(head_dim, head_dim, theta, yarn.into()).unwrap())
}

/// Checks one family translation against its predicted outcome and the frozen oracle.
///
/// `legacy` translates with [`legacy_dynamic`] added to the family context.
fn check_family_outcome(
    family: &str,
    expected: FamilyOutcome,
    shared: Result<RopeConfig>,
    legacy: Result<RopeConfig>,
    frozen: RopeConfig,
) -> std::result::Result<(), TestCaseError> {
    match expected {
        FamilyOutcome::Frozen => prop_assert_eq!(shared.unwrap(), frozen, "{}", family),
        FamilyOutcome::Error(expected_field) => {
            let error = shared.unwrap_err();
            prop_assert_eq!(field(&error), format!("{family}.rope.{expected_field}"));
            if expected_field == "original_max_position_embeddings" {
                // The legacy dynamic-NTK rule reproduces the frozen result exactly.
                prop_assert_eq!(legacy.unwrap(), frozen, "{}", family);
            }
        }
        FamilyOutcome::Transformers(transformers) => {
            prop_assert_eq!(shared.unwrap(), transformers, "{}", family);
        }
    }
    Ok(())
}

proptest! {
    /// Qwen, Qwen-MoE, and Muse recommended contexts follow the documented rules.
    ///
    /// Outside the documented divergences they equal the frozen translators;
    /// inside, they equal the Transformers result or report the field.
    #[test]
    fn family_contexts_reproduce_frozen_translators(
        (rope_type, factor, original, theta) in scaled_parameters(),
        head_pairs in 2usize..=64,
        extras in yarn_extras(),
    ) {
        let head_dim = 2 * head_pairs;
        let parameters = family_parameters(rope_type, factor, original, theta, extras);
        let expected = qwen_family_outcome(&parameters, head_dim, theta);

        let qwen = Qwen3_5TextConfig {
            head_dim,
            max_position_embeddings: FAMILY_MAX,
            rope_parameters: Some(parameters.clone()),
            partial_rotary_factor: None,
            ..Qwen3_5TextConfig::default()
        };
        check_family_outcome(
            "qwen3_5",
            expected,
            shared_qwen3_5(&qwen),
            shared_qwen3_5_with(&qwen, &legacy_dynamic(qwen3_5_context(&qwen))),
            frozen::qwen3_5(&qwen).unwrap(),
        )?;
        let moe = Qwen3MoeConfig {
            head_dim,
            max_position_embeddings: FAMILY_MAX,
            rope_theta: Some(theta),
            rope_parameters: Some(parameters.clone()),
            ..Qwen3MoeConfig::default()
        };
        check_family_outcome(
            "qwen3_moe",
            expected,
            shared_qwen3_moe(&moe),
            shared_qwen3_moe_with(&moe, &legacy_dynamic(qwen3_moe_context(&moe))),
            frozen::qwen3_moe(&moe).unwrap(),
        )?;
        let muse = MuseGlimmerTextConfig {
            head_dim,
            max_position_embeddings: FAMILY_MAX,
            rope_parameters: Some(parameters),
            ..MuseGlimmerTextConfig::default()
        };
        check_family_outcome(
            "muse_glimmer",
            expected,
            shared_muse_glimmer(&muse, theta),
            shared_muse_glimmer_with(&muse, &legacy_dynamic(muse_context(&muse, theta))),
            frozen::muse_glimmer(&muse, theta).unwrap(),
        )?;
    }

    /// The recommended DeepSeek context follows the reference and the documented changes.
    #[test]
    fn deepseek_context_reproduces_frozen_translator(
        (rope_type, factor, original, theta) in scaled_parameters(),
        head_pairs in 2usize..=64,
        extras in yarn_extras(),
    ) {
        let mut parameters = family_parameters(rope_type, factor, original, theta, extras);
        // The reference's softmax correction reads `mscale_all_dim` for linear
        // and dynamic RoPE too.
        if matches!(rope_type, RopeType::Linear | RopeType::Dynamic) {
            parameters.mscale_all_dim = extras.mscale_all_dim;
        }
        let config = DeepseekV2Config {
            qk_rope_head_dim: 2 * head_pairs,
            max_position_embeddings: FAMILY_MAX,
            rope_theta: theta,
            rope_scaling: Some(parameters.clone()),
            ..DeepseekV2Config::default()
        };
        let frozen = frozen::deepseek_v2(&config).unwrap();
        let shared = shared_deepseek_v2(&config);
        let has_mscale = parameters.mscale.is_some() || parameters.mscale_all_dim.is_some();
        if rope_type != RopeType::Default && parameters.factor.is_none() {
            let error = shared.unwrap_err();
            prop_assert_eq!(field(&error), "deepseek_v2.rope.factor");
            return Ok(());
        }
        if rope_type == RopeType::Dynamic && original != FAMILY_MAX {
            let error = shared.unwrap_err();
            prop_assert_eq!(field(&error), "deepseek_v2.rope.original_max_position_embeddings");
            let legacy = shared_deepseek_v2_with(&config, &legacy_dynamic(deepseek_context(&config)));
            prop_assert_eq!(legacy.unwrap(), frozen);
            return Ok(());
        }
        if rope_type == RopeType::Yarn && parameters.attention_factor.is_some() && has_mscale {
            // The private translator silently dropped attention_factor here.
            let error = shared.unwrap_err();
            prop_assert_eq!(field(&error), "deepseek_v2.rope.attention_factor");
            let lenient = deepseek_context(&config).with_unused_fields(UnusedRopeFields::Ignore);
            let lenient = parameters.to_rope_config(&lenient).unwrap();
            prop_assert_eq!(
                lenient,
                RopeConfig::new(
                    frozen.head_dimension(),
                    frozen.rotary_dimension(),
                    frozen.theta(),
                    yarn_of(frozen)
                        .with_truncate(parameters.truncate.unwrap_or(true))
                        .into(),
                )
                .unwrap()
            );
            return Ok(());
        }
        let shared = shared.unwrap();
        if rope_type == RopeType::Yarn {
            let truncate = parameters.truncate.unwrap_or(true);
            prop_assert_eq!(yarn_of(shared).truncate(), truncate);
            prop_assert_eq!(yarn_of(shared).with_truncate(true), yarn_of(frozen));
        } else {
            prop_assert_eq!(shared, frozen);
        }
        let base = (config.qk_head_dim() as f64).sqrt().recip();
        let expected = match (parameters.mscale_all_dim, parameters.factor) {
            (Some(all), Some(factor)) if all != 0.0 && rope_type != RopeType::Default => {
                base * yarn_mscale(factor, all).powi(2)
            }
            _ => base,
        };
        prop_assert_eq!(shared_deepseek_attention_scale(&config).unwrap(), expected);
        if rope_type == RopeType::Yarn {
            // YaRN was already corrected by the private translator.
            prop_assert_eq!(expected, frozen::deepseek_attention_scale(&config).unwrap());
        }
    }
}

proptest! {
    /// Both mscale conventions agree whenever both keys are nonzero and no amplitude is given.
    #[test]
    fn mscale_conventions_agree_on_released_deepseek_shapes(
        factor in 1.0f64..64.0,
        mscale in 0.01f64..2.0,
        mscale_all_dim in 0.01f64..2.0,
    ) {
        let rope = RopeParameters {
            rope_type: RopeType::Yarn,
            factor: Some(factor),
            original_max_position_embeddings: Some(32),
            mscale: Some(mscale),
            mscale_all_dim: Some(mscale_all_dim),
            ..RopeParameters::default()
        };
        let deepseek =
            context().with_yarn_mscale_convention(YarnMscaleConvention::DeepSeekReference);
        prop_assert_eq!(
            rope.to_rope_config(&context()).unwrap(),
            rope.to_rope_config(&deepseek).unwrap()
        );
    }

    /// Arbitrary parameters never panic, and the lenient policy only widens acceptance.
    ///
    /// Under `Reject` every unused field and disagreement is an error, so a
    /// strict success must take the same path, and give the same result, as
    /// the lenient policy.
    #[test]
    fn arbitrary_parameters_translate_totally_and_monotonically(
        type_index in 0usize..7,
        scalars in proptest::collection::vec(proptest::option::of(-2.0f64..64.0), 9),
        original in proptest::option::of(0usize..300),
        table in proptest::option::of(proptest::collection::vec(0.5f64..4.0, 0..10)),
        truncate in proptest::option::of(any::<bool>()),
        head_dim in 0usize..40,
        max_position_embeddings in 0usize..300,
        top_level_theta in proptest::option::of(0.5f64..1.0e6),
        top_level_partial in proptest::option::of(0.0f64..1.2),
        deepseek in any::<bool>(),
    ) {
        let rope_type = ROPE_TYPES[type_index];
        let rope = RopeParameters {
            rope_type,
            rope_theta: scalars[0].map(|theta| theta * 1000.0),
            partial_rotary_factor: scalars[1].map(|value| value / 32.0),
            factor: scalars[2],
            original_max_position_embeddings: original,
            attention_factor: scalars[3],
            beta_fast: scalars[4],
            beta_slow: scalars[5],
            mscale: scalars[6],
            mscale_all_dim: scalars[7],
            short_factor: table.clone(),
            long_factor: table,
            low_freq_factor: scalars[8],
            high_freq_factor: scalars[8].map(|low| low + 1.0),
            truncate,
            ..RopeParameters::default()
        };
        let convention = if deepseek {
            YarnMscaleConvention::DeepSeekReference
        } else {
            YarnMscaleConvention::HuggingFace
        };
        let strict = RopeContext::new("prop", head_dim, max_position_embeddings)
            .with_rope_theta(top_level_theta)
            .with_partial_rotary_factor(top_level_partial)
            .with_yarn_mscale_convention(convention);
        let lenient = rope.to_rope_config(&strict.with_unused_fields(UnusedRopeFields::Ignore));
        if let Ok(config) = rope.to_rope_config(&strict) {
            prop_assert_eq!(Some(config), lenient.as_ref().ok().copied());
        }
        if let Ok(config) = lenient {
            prop_assert!(config.rotary_dimension() <= config.head_dimension());
            prop_assert!(config.rotary_dimension().is_multiple_of(2));
            prop_assert!(config.theta().is_finite() && config.theta() > 0.0);
            let amplitude = config.scaling().amplitude();
            prop_assert!(amplitude.is_finite() && amplitude > 0.0);
            let translated = match config.scaling() {
                RopeScaling::None => RopeType::Default,
                RopeScaling::Linear(_) => RopeType::Linear,
                RopeScaling::DynamicNtk(_) => RopeType::Dynamic,
                RopeScaling::Yarn(_) => RopeType::Yarn,
                RopeScaling::LongRope(_) => RopeType::LongRope,
                RopeScaling::Llama3(_) => RopeType::Llama3,
                RopeScaling::Proportional(_) => RopeType::Proportional,
            };
            prop_assert_eq!(translated, rope_type);
        }
    }
}
