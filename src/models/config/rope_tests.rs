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

    for json in ["null", "3", r#"{"factor": "two"}"#] {
        assert!(
            serde_json::from_str::<RopeParameterSet>(json).is_err(),
            "{json} must not parse"
        );
    }
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
        RopeParameterSet::from(params(r#"{"rope_type": "linear", "factor": 2.0}"#)),
        RopeParameterSet::from(RopeParametersByLayerType::new(layers)),
    ] {
        let value = serde_json::to_value(&set).unwrap();
        assert_eq!(
            serde_json::from_value::<RopeParameterSet>(value).unwrap(),
            set
        );
    }
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
        &plain,
        &RopeContext::new("test", 2, MAX),
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

/// Expected YaRN amplitude outcome for one convention.
#[derive(Debug, Clone, Copy)]
enum Amplitude {
    /// The translation succeeds with this amplitude.
    Value(f64),
    /// Strict translation fails at this field; `Ignore` yields the value.
    Unused(&'static str, f64),
}

/// One amplitude case: `(attention_factor, mscale, mscale_all_dim, Hugging Face, DeepSeek)`.
type AmplitudeCase = (Option<f64>, Option<f64>, Option<f64>, Amplitude, Amplitude);

/// Each convention resolves `attention_factor`, `mscale`, and `mscale_all_dim` as documented.
#[test]
fn yarn_amplitude_follows_each_mscale_convention() {
    let factor: f64 = 16.0;
    let m = |multiplier: f64| yarn_mscale(factor, multiplier);
    let default = m(1.0);
    let cases: [AmplitudeCase; 10] = [
        (
            None,
            None,
            None,
            Amplitude::Value(default),
            Amplitude::Value(default),
        ),
        (
            Some(1.5),
            None,
            None,
            Amplitude::Value(1.5),
            Amplitude::Value(1.5),
        ),
        (
            None,
            Some(1.0),
            Some(0.8),
            Amplitude::Value(m(1.0) / m(0.8)),
            Amplitude::Value(m(1.0) / m(0.8)),
        ),
        (
            None,
            None,
            Some(0.8),
            Amplitude::Value(default),
            Amplitude::Value(m(1.0) / m(0.8)),
        ),
        (
            None,
            Some(0.7),
            None,
            Amplitude::Unused("mscale", default),
            Amplitude::Value(m(0.7)),
        ),
        (
            None,
            Some(0.0),
            Some(0.8),
            Amplitude::Unused("mscale", default),
            Amplitude::Value(1.0 / m(0.8)),
        ),
        (
            None,
            Some(1.0),
            Some(0.0),
            Amplitude::Unused("mscale", default),
            Amplitude::Value(m(1.0)),
        ),
        (
            Some(1.5),
            Some(1.0),
            Some(1.0),
            Amplitude::Unused("mscale", 1.5),
            Amplitude::Unused("attention_factor", 1.0),
        ),
        (
            Some(1.5),
            None,
            Some(1.0),
            Amplitude::Value(1.5),
            Amplitude::Unused("attention_factor", 1.0),
        ),
        (
            Some(1.5),
            Some(0.5),
            None,
            Amplitude::Unused("mscale", 1.5),
            Amplitude::Unused("attention_factor", m(0.5)),
        ),
    ];
    for (attention_factor, mscale, mscale_all_dim, hugging_face, deepseek) in cases {
        let rope = RopeParameters {
            rope_type: RopeType::Yarn,
            factor: Some(factor),
            original_max_position_embeddings: Some(32),
            attention_factor,
            mscale,
            mscale_all_dim,
            ..RopeParameters::default()
        };
        for (convention, expected) in [
            (YarnMscaleConvention::HuggingFace, hugging_face),
            (YarnMscaleConvention::DeepSeekReference, deepseek),
        ] {
            let strict = context().with_yarn_mscale_convention(convention);
            let label =
                format!("{convention:?} {attention_factor:?} {mscale:?} {mscale_all_dim:?}");
            let value = match expected {
                Amplitude::Value(value) => {
                    let strict_config = rope.to_rope_config(&strict).expect(&label);
                    assert_eq!(yarn_of(strict_config).attention_factor(), value, "{label}");
                    value
                }
                Amplitude::Unused(unused, value) => {
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

/// DeepSeek's softmax multiplier applies only to YaRN with a nonzero `mscale_all_dim`.
#[test]
fn softmax_scale_multiplier_applies_only_to_yarn_with_mscale_all_dim() {
    let multiplier = |json: &str| params(json).yarn_softmax_scale_multiplier(&context());
    assert_eq!(multiplier("{}").unwrap(), 1.0);
    assert_eq!(
        multiplier(r#"{"rope_type": "linear", "factor": 4.0, "mscale_all_dim": 1.0}"#).unwrap(),
        1.0
    );
    assert_eq!(
        multiplier(r#"{"rope_type": "yarn", "factor": 4.0}"#).unwrap(),
        1.0
    );
    assert_eq!(
        multiplier(r#"{"rope_type": "yarn", "factor": 4.0, "mscale_all_dim": 0.0}"#).unwrap(),
        1.0
    );
    assert_eq!(
        multiplier(r#"{"rope_type": "yarn", "factor": 40.0, "mscale_all_dim": 0.707}"#).unwrap(),
        yarn_mscale(40.0, 0.707).powi(2)
    );
    let error = multiplier(r#"{"rope_type": "yarn", "mscale_all_dim": 1.0}"#).unwrap_err();
    assert_eq!(field(&error), "test.rope.factor");
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
            if algorithm_field.is_read_by(rope_type) {
                // `mscale` alone is unused under the Hugging Face convention.
                if algorithm_field != AlgorithmField::Mscale {
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

/// Qwen3.5 context reproducing `qwen3_5::model::qwen_rope_config`.
fn qwen3_5_context(config: &Qwen3_5TextConfig) -> RopeContext {
    RopeContext::new("qwen3_5", config.head_dim, config.max_position_embeddings)
        .with_rope_theta(config.rope_theta)
        .with_partial_rotary_factor(config.partial_rotary_factor)
        .with_mrope(MropeHandling::TextPositions)
        .with_dynamic_ntk_context(DynamicNtkContext::OriginalMaxPositionEmbeddings)
}

/// Shared replacement for `qwen3_5::model::qwen_rope_config`.
fn shared_qwen3_5(config: &Qwen3_5TextConfig) -> Result<RopeConfig> {
    config
        .rope_parameters
        .clone()
        .unwrap_or_default()
        .to_rope_config(&qwen3_5_context(config))
}

/// Qwen3-MoE context reproducing `qwen3_moe::model::qwen_rope_config`.
fn qwen3_moe_context(config: &Qwen3MoeConfig) -> RopeContext {
    RopeContext::new("qwen3_moe", config.head_dim, config.max_position_embeddings)
        .with_rope_theta(config.rope_theta)
        .with_partial_rotary(PartialRotary::FullHeadOnly)
        .with_dynamic_ntk_context(DynamicNtkContext::OriginalMaxPositionEmbeddings)
        .with_unused_fields(UnusedRopeFields::Ignore)
}

/// Shared replacement for `qwen3_moe::model::qwen_rope_config`.
fn shared_qwen3_moe(config: &Qwen3MoeConfig) -> Result<RopeConfig> {
    config
        .rope_parameters
        .clone()
        .unwrap_or_default()
        .to_rope_config(&qwen3_moe_context(config))
}

/// DeepSeek-V2 context reproducing `deepseek_v2::model::deepseek_rope_config`.
fn deepseek_context(config: &DeepseekV2Config) -> RopeContext {
    RopeContext::new(
        "deepseek_v2",
        config.qk_rope_head_dim,
        config.max_position_embeddings,
    )
    .with_rope_theta(Some(config.rope_theta))
    .with_partial_rotary(PartialRotary::FullHeadOnly)
    .with_yarn_mscale_convention(YarnMscaleConvention::DeepSeekReference)
    .with_dynamic_ntk_context(DynamicNtkContext::OriginalMaxPositionEmbeddings)
}

/// Shared replacement for `deepseek_v2::model::deepseek_rope_config`.
fn shared_deepseek_v2(config: &DeepseekV2Config) -> Result<RopeConfig> {
    config
        .rope_scaling
        .clone()
        .unwrap_or_default()
        .to_rope_config(&deepseek_context(config))
}

/// Shared replacement for `deepseek_v2::model::deepseek_attention_scale`.
fn shared_deepseek_attention_scale(config: &DeepseekV2Config) -> Result<f64> {
    let multiplier = config
        .rope_scaling
        .clone()
        .unwrap_or_default()
        .yarn_softmax_scale_multiplier(&deepseek_context(config))?;
    Ok((config.qk_head_dim() as f64).sqrt().recip() * multiplier)
}

/// Shared replacement for `muse_glimmer::model::muse_rope_config`.
fn shared_muse_glimmer(config: &MuseGlimmerTextConfig, theta: f64) -> Result<RopeConfig> {
    config
        .rope_parameters
        .clone()
        .unwrap_or_default()
        .to_rope_config(
            &RopeContext::new(
                "muse_glimmer",
                config.head_dim,
                config.max_position_embeddings,
            )
            .with_layer_rope_theta(theta)
            .with_partial_rotary(PartialRotary::FullHeadOnly)
            .with_dynamic_ntk_context(DynamicNtkContext::OriginalMaxPositionEmbeddings),
        )
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
    // The pinned Qwen3-MoE checkpoint also passes the strict policy.
    let strict = qwen3_moe_context(&config).with_unused_fields(UnusedRopeFields::Reject);
    assert_eq!(
        config
            .rope_parameters
            .clone()
            .unwrap_or_default()
            .to_rope_config(&strict)
            .unwrap(),
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
    // qwen3_5 model/recipe/intervention/decode tests, plus the decode test's dynamic NTK.
    let qwen_tiny = |max_position_embeddings| Qwen3_5TextConfig {
        head_dim: 4,
        max_position_embeddings,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        ..Qwen3_5TextConfig::default()
    };
    let mut qwen_configs = vec![Qwen3_5TextConfig::default(), qwen_tiny(64), qwen_tiny(32)];
    let mut dynamic = qwen_tiny(64);
    dynamic.rope_parameters = Some(RopeParameters {
        rope_type: RopeType::Dynamic,
        factor: Some(2.0),
        original_max_position_embeddings: Some(2),
        ..RopeParameters::default()
    });
    qwen_configs.push(dynamic.clone());
    for config in &qwen_configs {
        assert_eq!(
            shared_qwen3_5(config).unwrap(),
            frozen::qwen3_5(config).unwrap()
        );
    }
    // The decode test relies on the legacy trained context of two positions.
    assert_eq!(
        shared_qwen3_5(&dynamic)
            .unwrap()
            .scaling()
            .cached_context_limit(),
        Some(2)
    );

    // qwen3_moe model/checkpoint/load tests and the mixed-projection fixtures.
    let moe = |head_dim, max_position_embeddings, rope_theta| Qwen3MoeConfig {
        head_dim,
        max_position_embeddings,
        rope_theta,
        ..Qwen3MoeConfig::default()
    };
    let mut moe_configs = vec![
        Qwen3MoeConfig::default(),
        moe(4, 32, Some(1_000_000.0)),
        moe(2, 16, Some(10_000.0)),
        moe(4, 32, Some(10_000.0)),
    ];
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
        // Strictly, the linear fixture's unread original context and the
        // nested theta overriding the top-level one are rejected; the exact
        // reproduction therefore needs `UnusedRopeFields::Ignore`.
        let strict = qwen3_moe_context(&mixed).with_unused_fields(UnusedRopeFields::Reject);
        let error = mixed
            .rope_parameters
            .clone()
            .unwrap()
            .to_rope_config(&strict)
            .unwrap_err();
        let expected = if rope_type == RopeType::Linear {
            "qwen3_moe.rope.original_max_position_embeddings"
        } else {
            "qwen3_moe.rope.rope_theta"
        };
        assert_eq!(field(&error), expected);
        moe_configs.push(mixed);
    }
    for config in &moe_configs {
        assert_eq!(
            shared_qwen3_moe(config).unwrap(),
            frozen::qwen3_moe(config).unwrap()
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

/// Strategy for a family-compatible scaling setup: `(type, factor, original, theta)`.
fn scaled_parameters() -> impl Strategy<Value = (RopeType, f64, usize, f64)> {
    (
        prop_oneof![
            Just(RopeType::Default),
            Just(RopeType::Linear),
            Just(RopeType::Dynamic),
            Just(RopeType::Yarn),
        ],
        1.0f64..64.0,
        1usize..=256,
        1.5f64..1.0e7,
    )
}

/// Builds parameters carrying only the fields `rope_type` reads.
///
/// `yarn_extras` is `(attention_factor, beta_fast, beta_slow)`, kept for YaRN only.
fn family_parameters(
    rope_type: RopeType,
    factor: f64,
    original: usize,
    theta: f64,
    yarn_extras: (Option<f64>, Option<f64>, Option<f64>),
) -> RopeParameters {
    let (attention_factor, beta_fast, beta_slow) = yarn_extras;
    let yarn = rope_type == RopeType::Yarn;
    RopeParameters {
        rope_type,
        rope_theta: Some(theta),
        factor: (rope_type != RopeType::Default).then_some(factor),
        original_max_position_embeddings: matches!(rope_type, RopeType::Dynamic | RopeType::Yarn)
            .then_some(original),
        attention_factor: attention_factor.filter(|_| yarn),
        beta_fast: beta_fast.filter(|_| yarn),
        beta_slow: beta_slow.filter(|_| yarn),
        ..RopeParameters::default()
    }
}

proptest! {
    /// Qwen, Qwen-MoE, and Muse adoption calls equal the frozen translators.
    #[test]
    fn family_contexts_reproduce_frozen_translators(
        (rope_type, factor, original, theta) in scaled_parameters(),
        head_pairs in 2usize..=64,
        attention_factor in proptest::option::of(0.5f64..2.0),
        beta_fast in proptest::option::of(8.0f64..64.0),
        beta_slow in proptest::option::of(0.5f64..4.0),
    ) {
        let head_dim = 2 * head_pairs;
        let parameters = family_parameters(
            rope_type,
            factor,
            original,
            theta,
            (attention_factor, beta_fast, beta_slow),
        );
        let qwen = Qwen3_5TextConfig {
            head_dim,
            max_position_embeddings: 256,
            rope_parameters: Some(parameters.clone()),
            partial_rotary_factor: None,
            ..Qwen3_5TextConfig::default()
        };
        prop_assert_eq!(shared_qwen3_5(&qwen).unwrap(), frozen::qwen3_5(&qwen).unwrap());
        let moe = Qwen3MoeConfig {
            head_dim,
            max_position_embeddings: 256,
            rope_parameters: Some(parameters.clone()),
            ..Qwen3MoeConfig::default()
        };
        prop_assert_eq!(shared_qwen3_moe(&moe).unwrap(), frozen::qwen3_moe(&moe).unwrap());
        let muse = MuseGlimmerTextConfig {
            head_dim,
            max_position_embeddings: 256,
            rope_parameters: Some(parameters),
            ..MuseGlimmerTextConfig::default()
        };
        prop_assert_eq!(
            shared_muse_glimmer(&muse, theta).unwrap(),
            frozen::muse_glimmer(&muse, theta).unwrap()
        );
    }

    /// The DeepSeek adoption call equals the frozen translator and attention scale.
    #[test]
    fn deepseek_context_reproduces_frozen_translator(
        (rope_type, factor, original, theta) in scaled_parameters(),
        head_pairs in 2usize..=64,
        mscale in proptest::option::of(0.0f64..2.0),
        mscale_all_dim in proptest::option::of(0.0f64..2.0),
        attention_factor in proptest::option::of(0.5f64..2.0),
    ) {
        let mut parameters =
            family_parameters(rope_type, factor, original, theta, (None, None, None));
        if rope_type == RopeType::Yarn {
            parameters.mscale = mscale;
            parameters.mscale_all_dim = mscale_all_dim;
            parameters.attention_factor = attention_factor;
        }
        let config = DeepseekV2Config {
            qk_rope_head_dim: 2 * head_pairs,
            max_position_embeddings: 256,
            rope_theta: theta,
            rope_scaling: Some(parameters.clone()),
            ..DeepseekV2Config::default()
        };
        let frozen = frozen::deepseek_v2(&config).unwrap();
        let ambiguous = parameters.attention_factor.is_some()
            && (parameters.mscale.is_some() || parameters.mscale_all_dim.is_some());
        match shared_deepseek_v2(&config) {
            Ok(shared) => {
                prop_assert!(!ambiguous);
                prop_assert_eq!(shared, frozen);
            }
            Err(error) => {
                // The private translator silently dropped attention_factor here.
                prop_assert!(ambiguous, "{}", error);
                prop_assert_eq!(field(&error), "deepseek_v2.rope.attention_factor");
                let lenient = deepseek_context(&config).with_unused_fields(UnusedRopeFields::Ignore);
                prop_assert_eq!(parameters.to_rope_config(&lenient).unwrap(), frozen);
            }
        }
        prop_assert_eq!(
            shared_deepseek_attention_scale(&config).unwrap(),
            frozen::deepseek_attention_scale(&config).unwrap()
        );
    }

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
