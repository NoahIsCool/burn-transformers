//! Deterministic regression seeds for the RoPE-translator fuzz harness.

use super::*;

/// A context with Transformers defaults for 64-wide heads and 4096 positions.
fn default_context() -> RopeContext {
    RopeContext::new(ARCHITECTURE, 64, 4096)
}

/// Every algorithm with its required fields translates, and the checks pass.
#[test]
fn every_algorithm_translates_and_evaluates() {
    let documents = [
        json!({}),
        json!({"rope_type": "default", "rope_theta": 1_000_000.0}),
        json!({"rope_type": "linear", "factor": 4.0}),
        json!({"type": "dynamic", "factor": 2.0}),
        json!({"rope_type": "yarn", "factor": 32.0, "original_max_position_embeddings": 4096,
               "beta_fast": 32.0, "beta_slow": 1.0, "truncate": false}),
        json!({"rope_type": "longrope", "short_factor": vec![1.0; 32], "long_factor": vec![2.0; 32],
               "original_max_position_embeddings": 1024}),
        json!({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
               "high_freq_factor": 4.0, "original_max_position_embeddings": 1024}),
        json!({"rope_type": "proportional", "partial_rotary_factor": 0.25}),
        json!({"full_attention": {"rope_type": "default"},
               "sliding_attention": {"rope_type": "linear", "factor": 8.0}}),
    ];
    for document in documents {
        let text = document.to_string();
        let set: RopeParameterSet = serde_json::from_str(&text).expect("parses");
        let layer = match &set {
            RopeParameterSet::Uniform(_) => "full_attention",
            RopeParameterSet::ByLayerType(_) => "sliding_attention",
        };
        set.to_rope_config(layer, &default_context())
            .unwrap_or_else(|error| panic!("{text}: {error}"));
        check_document(
            &text,
            &default_context(),
            &["full_attention", "sliding_attention"],
        );
    }
}

/// Unused, conflicting, and malformed fields fail with contextual paths.
#[test]
fn rejected_documents_name_the_offending_field() {
    let cases = [
        (json!({"rope_type": "linear"}), "fuzz.rope.factor"),
        (
            json!({"rope_type": "default", "factor": 2.0}),
            "fuzz.rope.factor",
        ),
        (
            json!({"rope_type": "yarn", "factor": 0.5}),
            "fuzz.rope.factor",
        ),
        (
            json!({"rope_type": "longrope", "short_factor": [1.0], "long_factor": [1.0, 2.0]}),
            "fuzz.rope.long_factor",
        ),
        (
            json!({"rope_type": "default", "mrope_section": [1, 2]}),
            "fuzz.rope.mrope_section",
        ),
    ];
    for (document, field) in cases {
        let set: RopeParameterSet = serde_json::from_value(document.clone()).expect("parses");
        match set.to_rope_config("full_attention", &default_context()) {
            Err(Error::Configuration { field: actual, .. }) => {
                assert_eq!(actual, field, "{document}")
            }
            other => panic!("{document}: {other:?}"),
        }
        check_document(&document.to_string(), &default_context(), &[]);
    }
    // Shapes rejected by the parser itself.
    for text in [
        r#"{"rope_type": "su"}"#,
        r#"{"rope_type": "yarn", "type": "linear"}"#,
        r#"{"full_attention": {}, "rope_theta": 10000.0}"#,
    ] {
        assert!(
            serde_json::from_str::<RopeParameterSet>(text).is_err(),
            "{text}"
        );
        check_document(text, &default_context(), &[]);
    }
}

/// Ignore-policy translation agrees with Reject wherever Reject succeeds.
#[test]
fn ignore_policy_extends_reject_policy() {
    let set: RopeParameterSet =
        serde_json::from_value(json!({"rope_type": "yarn", "factor": 4.0, "mscale": 1.0,
                                       "mscale_all_dim": 1.0}))
        .unwrap();
    let context = default_context().with_softmax_mscale(SoftmaxMscale::MscaleAllDimSquared);
    check_translation(&set, "full_attention", &context);
    let strict = set.to_rope_config("full_attention", &context).unwrap();
    let lenient = set
        .to_rope_config(
            "full_attention",
            &context.with_unused_fields(UnusedRopeFields::Ignore),
        )
        .unwrap();
    assert_eq!(strict, lenient);
}

/// Layer-type validation needs exactly the used layer types.
#[test]
fn layer_types_must_match_the_mapping_keys() {
    let set: RopeParameterSet = serde_json::from_value(json!({
        "full_attention": {}, "sliding_attention": {"rope_type": "linear", "factor": 2.0}
    }))
    .unwrap();
    check_layer_types(&set, &["full_attention", "sliding_attention"]);
    check_layer_types(&set, &["full_attention"]);
    check_layer_types(
        &set,
        &["full_attention", "sliding_attention", "chunked_attention"],
    );
    assert!(
        set.validate_layer_types(ARCHITECTURE, ["sliding_attention", "full_attention"])
            .is_ok()
    );
}

/// Committed seeds and boundary inputs keep their meaning.
#[test]
fn committed_seeds_pass() {
    for seed in [
        &include_bytes!("../seeds/rope_parameters/yarn-deepseek")[..],
        &include_bytes!("../seeds/rope_parameters/per-layer-type")[..],
        &include_bytes!("../seeds/rope_parameters/raw-json")[..],
    ] {
        exercise(seed);
    }
    exercise(&[]);
    exercise(&[255; 128]);
}

/// Pseudo-random inputs cover many algorithms, contexts, and key mixes.
#[test]
fn pseudo_random_inputs_translate_or_fail_contextually() {
    for input in crate::lcg_inputs(0x726f_7065, 1024, 96) {
        exercise(&input);
    }
}

/// ROPE-F32-1: a positive base below one overflows `f32` inverse frequencies.
///
/// `RopeConfig::new` accepts any finite positive theta, but pair `i` has
/// frequency `theta^(-2i/d)`; `theta = 1e-300` gives `1e150` for `i = 1` of a
/// four-wide head, which rounds to `inf` in `f32`. Position zero then has the
/// angle `0 * inf = NaN`. Fix in `src/position/rope.rs`: require every
/// inverse frequency to be finite in `f32` (equivalently bound `theta` below),
/// or reject `theta < 1`, which no checkpoint uses.
#[test]
#[ignore = "ROPE-F32-1: open follow-up in src/position/rope.rs (f32 overflow for theta < 1)"]
fn accepted_bases_have_finite_frequencies() {
    let set: RopeParameterSet =
        serde_json::from_value(json!({"rope_type": "default", "rope_theta": 1e-300})).unwrap();
    let context = RopeContext::new(ARCHITECTURE, 4, 16);
    if let Ok(config) = set.to_rope_config("full_attention", &context) {
        let frequencies = RotaryFrequencies::new(config, 16).unwrap();
        assert!(frequencies.inverse().iter().all(|omega| omega.is_finite()));
    }
}

/// ROPE-F32-2: a tiny positive LongRoPE factor overflows `f32` inverse frequencies.
///
/// Factors only need to be finite and positive after rounding to `f32`, so the
/// smallest subnormal `1e-45` passes and divides `omega_0 = 1` into `7e44`,
/// which is `inf` in `f32`. Fix in `src/position/rope_scaling.rs`: require
/// `omega / factor` to stay finite, e.g. factors of at least `f32::MIN_POSITIVE`.
#[test]
#[ignore = "ROPE-F32-2: open follow-up in src/position/rope_scaling.rs (tiny LongRoPE factors)"]
fn accepted_long_rope_factors_have_finite_frequencies() {
    let set: RopeParameterSet = serde_json::from_value(json!({
        "rope_type": "longrope", "short_factor": [1e-45, 1.0], "long_factor": [1.0, 1.0],
        "original_max_position_embeddings": 8
    }))
    .unwrap();
    let context = RopeContext::new(ARCHITECTURE, 4, 16);
    if let Ok(config) = set.to_rope_config("full_attention", &context) {
        let frequencies = RotaryFrequencies::new(config, 4).unwrap();
        assert!(frequencies.inverse().iter().all(|omega| omega.is_finite()));
    }
}

/// ROPE-PARSE-1: a JSON array parses as a positional `RopeParameters` struct.
///
/// Serde's derived struct deserializer also accepts sequences, and
/// `RopeParameterSet` passes every non-object value to it, so `[]` becomes
/// default RoPE and `["yarn"]` sets `rope_type` positionally. Transformers
/// only accepts a mapping. Fix in `src/models/config/rope.rs`
/// (`RopeParameterSet::from_json_value`): reject values that are not objects.
#[test]
#[ignore = "ROPE-PARSE-1: open follow-up in src/models/config/rope.rs (arrays parse positionally)"]
fn arrays_are_not_rope_parameters() {
    for text in ["[]", r#"["yarn"]"#] {
        assert!(
            serde_json::from_str::<RopeParameterSet>(text).is_err(),
            "{text}"
        );
    }
}

/// Per-layer-type entries get the same softmax-multiplier and error checks as flat ones.
///
/// Translation validates the softmax correction of every entry, so the
/// harness checks the multiplier of per-layer entries too (it once checked
/// flat mappings only), and both unused-field policies must fail contextually.
#[test]
fn per_layer_softmax_multipliers_and_lenient_errors_are_checked() {
    let context = default_context().with_softmax_mscale(SoftmaxMscale::MscaleAllDimSquared);
    let document = json!({
        "full_attention": {"rope_type": "yarn", "factor": 4.0, "mscale_all_dim": 1.0},
        "sliding_attention": {"rope_type": "linear", "factor": 2.0, "mscale_all_dim": 0.5},
    });
    let set: RopeParameterSet = serde_json::from_value(document.clone()).expect("parses");
    for layer in ["full_attention", "sliding_attention"] {
        set.to_rope_config(layer, &context).expect("translates");
        let parameters = set.parameters_for(layer).expect("entry");
        let multiplier = parameters.softmax_scale_multiplier(&context).unwrap();
        assert!(
            multiplier > 1.0 && multiplier.is_finite(),
            "{layer}: {multiplier}"
        );
        check_softmax_multiplier(parameters, &context);
    }
    check_document(
        &document.to_string(),
        &context,
        &["full_attention", "sliding_attention"],
    );
    // An entry that fails under both policies (a missing factor) is contextual
    // either way; the lenient error is now checked as well.
    let failing = json!({"full_attention": {"rope_type": "linear"}});
    let set: RopeParameterSet = serde_json::from_value(failing.clone()).expect("parses");
    let lenient = context.with_unused_fields(UnusedRopeFields::Ignore);
    let error = set.to_rope_config("full_attention", &lenient).unwrap_err();
    check_error(&error, "lenient");
    check_document(&failing.to_string(), &context, &["full_attention"]);
}

/// A non-configuration or uncontextual error is reported as a harness failure.
#[test]
#[should_panic(expected = "uncontextual lenient field other.rope.factor")]
fn uncontextual_errors_are_rejected() {
    check_error(
        &Error::configuration("other.rope.factor", "is required"),
        "lenient",
    );
}

/// Campaign regression `crash-3a306a4f` (minimized): a huge `mscale_all_dim`
/// translates, and the harness tolerates the open overflow below.
#[test]
fn campaign_huge_mscale_all_dim_is_tolerated() {
    exercise(&[
        0x08, 0x35, 0x35, 0x35, 0x35, 0x35, 0x35, 0x35, 0x35, 0x35, 0x35, 0x35, 0x35, 0x35, 0x36,
        0x35, 0x5d, 0xff, 0x0d,
    ]);
}

/// ROPE-SOFTMAX-1: a translated softmax correction overflows when squared.
///
/// Translation checks that `m = 1 + 0.1 * mscale_all_dim * ln(factor)` is
/// finite and positive, but the multiplier is `m^2`, which is `inf` once
/// `m > 1.3e154` (here `mscale_all_dim = 1e300`). A family applying it would
/// scale its softmax by infinity. Fix in `src/models/config/rope.rs`
/// (`softmax_scale_multiplier`): validate `m.powi(2)` instead of `m`.
#[test]
#[ignore = "ROPE-SOFTMAX-1: open follow-up in src/models/config/rope.rs (squared correction overflows)"]
fn translated_softmax_corrections_are_finite() {
    let set: RopeParameterSet = serde_json::from_value(json!({
        "rope_type": "linear", "factor": 2.0, "mscale_all_dim": 1e300
    }))
    .unwrap();
    let context = default_context().with_softmax_mscale(SoftmaxMscale::MscaleAllDimSquared);
    if set.to_rope_config("full_attention", &context).is_ok() {
        let RopeParameterSet::Uniform(parameters) = set else {
            unreachable!("flat mapping")
        };
        let multiplier = parameters.softmax_scale_multiplier(&context).unwrap();
        assert!(multiplier.is_finite(), "multiplier {multiplier}");
    }
}
