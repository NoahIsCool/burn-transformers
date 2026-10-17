//! Hugging Face `rope_parameters` through the shared RoPE translator.
//!
//! [`RopeParameterSet`] parses the Transformers 5.x `rope_parameters` value (a
//! flat mapping, or one mapping per attention layer type), and
//! [`RopeParameterSet::to_rope_config`] resolves it against a [`RopeContext`]
//! that carries the family's top-level fields and conventions. For arbitrary
//! JSON, generated either structurally (every algorithm and key, boundary
//! values, legacy `type` keys, per-layer nesting) or from raw bytes, and an
//! arbitrary context, the harness checks:
//!
//! 1. **Totality.** Parsing and translation never panic. A translation error,
//!    under either unused-field policy, is a contextual [`Error::Configuration`]
//!    whose field path starts with the context's architecture name.
//! 2. **Validity.** A translated configuration spans the context's head width,
//!    rotates a positive even prefix of it, and evaluates (at contexts `1`,
//!    `max_position_embeddings`, one past it, and twice it) to `rotary / 2`
//!    finite, non-negative inverse frequencies with an amplitude that is a
//!    normal positive `f32`. (Findings ROPE-F32-1/2, fixed: a base below one or
//!    a subnormal LongRoPE factor used to translate and overflow `f32`.)
//! 3. **Policy monotonicity.** Whatever [`UnusedRopeFields::Reject`] accepts,
//!    [`UnusedRopeFields::Ignore`] accepts with the identical configuration.
//!    For every translated entry, flat or per layer type, the softmax
//!    multiplier is a normal positive `f32` (finding ROPE-SOFTMAX-1, fixed:
//!    translation once validated `m` but not `m^2`).
//! 4. **Round trip.** A parsed set serializes and parses back to the same JSON,
//!    integers and floats exactly (see `crate::floats_round_trip`).
//! 5. **Layer types.** `validate_layer_types` accepts exactly when every used
//!    layer type has an entry and every entry is used.

use std::collections::BTreeSet;

use serde_json::{Map, Number, Value, json};
use transformers::{
    Error,
    models::config::{
        DynamicNtkContext, MropeHandling, PartialRotary, RopeContext, RopeParameterSet,
        RopeParameters, SoftmaxMscale, UnusedRopeFields, YarnMscaleConvention,
    },
    position::{LongRopeScaling, RopeScaling, RotaryFrequencies},
};

use crate::Bytes;

/// Architecture name of every context; error paths must start with it.
const ARCHITECTURE: &str = "fuzz";
/// Layer-type names used for nesting and lookups; the empty name is legal JSON.
const LAYER_TYPES: [&str; 5] = [
    "full_attention",
    "sliding_attention",
    "chunked_attention",
    "",
    "linear_attention",
];
/// Algorithm spellings, including unknown and legacy ones that must be rejected.
const ROPE_TYPES: [&str; 10] = [
    "default",
    "linear",
    "dynamic",
    "yarn",
    "longrope",
    "llama3",
    "proportional",
    "su",
    "mrope",
    "YARN",
];
/// Every serialized `RopeParameters` key plus an unknown key.
const KEYS: [&str; 17] = [
    "rope_theta",
    "partial_rotary_factor",
    "factor",
    "original_max_position_embeddings",
    "attention_factor",
    "beta_fast",
    "beta_slow",
    "mscale",
    "mscale_all_dim",
    "short_factor",
    "long_factor",
    "low_freq_factor",
    "high_freq_factor",
    "mrope_section",
    "mrope_interleaved",
    "truncate",
    "unknown_key",
];
/// Integer-like JSON values for context-length keys.
const INTEGERS: [i128; 8] = [0, 1, 2, 4, 4096, 131_072, u64::MAX as i128, -1];

/// Decodes a JSON number from the boundary-float table, or `null` for a non-finite draw.
fn number(bytes: &mut Bytes<'_>) -> Value {
    Number::from_f64(bytes.interesting_float()).map_or(Value::Null, Value::Number)
}

/// Decodes one JSON value suited to `key`, sometimes deliberately mistyped.
fn value(bytes: &mut Bytes<'_>, key: &str) -> Value {
    match bytes.below(8) {
        0 => Value::Null,
        1 => json!("text"),
        2 => json!(bytes.flag()),
        _ => match key {
            "original_max_position_embeddings" => {
                let integer = bytes.pick(&INTEGERS);
                serde_json::from_str(&integer.to_string()).expect("integer literal")
            }
            "short_factor" | "long_factor" => {
                Value::Array((0..bytes.below(6)).map(|_| number(bytes)).collect())
            }
            "mrope_section" => {
                Value::Array((0..bytes.below(4)).map(|_| json!(bytes.below(5))).collect())
            }
            "mrope_interleaved" | "truncate" => json!(bytes.flag()),
            _ => number(bytes),
        },
    }
}

/// Decodes one flat RoPE mapping with up to eight keys.
fn mapping(bytes: &mut Bytes<'_>) -> Value {
    let mut object = Map::new();
    match bytes.below(4) {
        0 => {}
        1 => {
            object.insert("type".into(), json!(bytes.pick(&ROPE_TYPES)));
        }
        2 => {
            // Current and legacy keys together: equal or conflicting.
            object.insert("rope_type".into(), json!(bytes.pick(&ROPE_TYPES)));
            object.insert("type".into(), json!(bytes.pick(&ROPE_TYPES)));
        }
        _ => {
            object.insert("rope_type".into(), json!(bytes.pick(&ROPE_TYPES)));
        }
    }
    for _ in 0..bytes.below(9) {
        let key = bytes.pick(&KEYS);
        let value = value(bytes, key);
        object.insert(key.into(), value);
    }
    Value::Object(object)
}

/// Decodes a `rope_parameters` value: flat, per layer type, mixed, or not an object.
fn parameters(bytes: &mut Bytes<'_>) -> Value {
    match bytes.below(8) {
        0 => Value::Array(Vec::new()),
        1 => json!({}),
        2 | 3 => {
            let mut layers = Map::new();
            for _ in 0..bytes.below(3) + 1 {
                layers.insert(bytes.pick(&LAYER_TYPES).into(), mapping(bytes));
            }
            if bytes.below(4) == 0 {
                // A flat key beside layer objects must be rejected as mixed.
                layers.insert("rope_theta".into(), json!(10_000.0));
            }
            Value::Object(layers)
        }
        _ => mapping(bytes),
    }
}

/// Decodes a family context; see [`exercise`] for its bytes.
fn context(bytes: &mut Bytes<'_>) -> RopeContext {
    let head_dim = bytes.pick(&[0, 1, 2, 4, 6, 8, 10, 64, 128, 256, usize::MAX]);
    let max_positions = bytes.pick(&[0, 1, 2, 16, 4096, 131_072, usize::MAX]);
    let flags = u16::from_le_bytes([bytes.byte(), bytes.byte()]);
    let mut context = RopeContext::new(ARCHITECTURE, head_dim, max_positions);
    if flags & 1 != 0 {
        context = context.with_rope_theta(Some(bytes.interesting_float()));
    }
    if flags & 2 != 0 {
        context = context.with_partial_rotary_factor(Some(bytes.interesting_float()));
    }
    if flags & 4 != 0 {
        let original = bytes.pick(&[0, 1, 16, 4096, usize::MAX]);
        context = context.with_original_max_position_embeddings(Some(original));
    }
    if flags & 8 != 0 {
        context = context.with_partial_rotary(PartialRotary::FullHeadOnly);
    }
    if flags & 16 != 0 {
        context = context.with_yarn_mscale_convention(YarnMscaleConvention::DeepSeekReference);
    }
    if flags & 32 != 0 {
        context = context.with_softmax_mscale(SoftmaxMscale::MscaleAllDimSquared);
    }
    if flags & 64 != 0 {
        context =
            context.with_dynamic_ntk_context(DynamicNtkContext::OriginalMaxPositionEmbeddings);
    }
    if flags & 128 != 0 {
        context = context.with_mrope(MropeHandling::TextPositions);
    }
    if flags & 256 != 0 {
        context = context.with_default_rope_theta(bytes.interesting_float());
    }
    if flags & 512 != 0 {
        context = context.with_layer_rope_theta(bytes.interesting_float());
    }
    context
}

/// The largest magnitude of any integer position, `2^63`, as the device's `f32` angle factor.
///
/// Every accepted frequency `omega` must keep the angle `2^63 * omega`
/// finite in `f32`, the precision in which the device forms angles.
pub(crate) const LARGEST_POSITION: f32 = 9_223_372_036_854_775_808.0;

/// Whether `value` rounds to a normal positive `f32`, the documented range of
/// RoPE amplitudes and of the softmax multiplier.
fn normal_positive_f32(value: f64) -> bool {
    let rounded = value as f32;
    rounded.is_finite() && rounded >= f32::MIN_POSITIVE
}

/// Checks one successful translation.
fn check_config(config: transformers::position::RopeConfig, context: &RopeContext) {
    assert_eq!(config.head_dimension(), context.head_dim());
    let rotary = config.rotary_dimension();
    assert!(rotary > 0 && rotary.is_multiple_of(2) && rotary <= context.head_dim());
    // Large heads would make the frequency tables expensive; real heads are at most 512.
    if context.head_dim() > 4096 {
        return;
    }
    let maximum = context.max_position_embeddings();
    for evaluated in [
        1,
        maximum,
        maximum.saturating_add(1),
        maximum.saturating_mul(2),
    ] {
        let frequencies = RotaryFrequencies::new(config, evaluated).expect("non-zero context");
        assert_eq!(frequencies.inverse().len(), rotary / 2);
        assert_eq!(frequencies.rotary_dimension(), rotary);
        let amplitude = frequencies.amplitude();
        assert!(normal_positive_f32(amplitude), "amplitude {amplitude}");
        assert!(config.theta() >= 1.0, "accepted base {}", config.theta());
        if let RopeScaling::LongRope(long) = config.scaling() {
            assert!(
                long.short_factors()
                    .iter()
                    .chain(long.long_factors())
                    .all(|factor| factor.is_finite() && *factor >= LongRopeScaling::MIN_FACTOR),
                "{config:?}"
            );
        }
        for &omega in frequencies.inverse() {
            assert!(
                omega.is_finite() && omega >= 0.0,
                "inverse frequency {omega} for {config:?}"
            );
            assert!(
                (LARGEST_POSITION * omega).is_finite(),
                "angle 2^63 * {omega} overflows for {config:?}"
            );
        }
    }
}

/// Asserts that a translation error is contextual.
///
/// Every error must be [`Error::Configuration`] whose field path starts with
/// the context's architecture name, under either unused-field policy.
fn check_error(error: &Error, policy: &str) {
    match error {
        Error::Configuration { field, .. } => assert!(
            field.starts_with(ARCHITECTURE),
            "uncontextual {policy} field {field}"
        ),
        other => panic!("non-configuration {policy} translation error {other:?}"),
    }
}

/// Checks the softmax multiplier of an entry whose translation succeeded.
///
/// Translation validates the multiplier for flat and per-layer-type entries
/// alike, so [`RopeParameters::softmax_scale_multiplier`] must then succeed
/// with a normal positive `f32` value (finding ROPE-SOFTMAX-1, fixed: the
/// square of a validated correction used to overflow).
fn check_softmax_multiplier(parameters: &RopeParameters, context: &RopeContext) {
    let multiplier = parameters
        .softmax_scale_multiplier(context)
        .expect("translation validated the softmax correction");
    assert!(
        normal_positive_f32(multiplier),
        "softmax multiplier {multiplier}"
    );
}

/// Translates one layer type and checks every property of a result.
fn check_translation(set: &RopeParameterSet, layer_type: &str, context: &RopeContext) {
    let strict = set.to_rope_config(layer_type, context);
    let lenient_context = context.with_unused_fields(UnusedRopeFields::Ignore);
    let lenient = set.to_rope_config(layer_type, &lenient_context);
    match strict {
        Ok(config) => {
            check_config(config, context);
            assert_eq!(
                lenient.as_ref().ok(),
                Some(&config),
                "Ignore must accept what Reject accepts"
            );
            let parameters = set
                .parameters_for(layer_type)
                .expect("a translated layer type has parameters");
            check_softmax_multiplier(parameters, context);
        }
        Err(error) => check_error(&error, "strict"),
    }
    match lenient {
        Ok(config) => {
            check_config(config, &lenient_context);
            let parameters = set
                .parameters_for(layer_type)
                .expect("a translated layer type has parameters");
            check_softmax_multiplier(parameters, &lenient_context);
        }
        Err(error) => check_error(&error, "lenient"),
    }
}

/// Checks layer-type validation against the set of keys.
fn check_layer_types(set: &RopeParameterSet, used: &[&str]) {
    let result = set.validate_layer_types(ARCHITECTURE, used.iter().copied());
    let expected = match set {
        RopeParameterSet::Uniform(_) => true,
        RopeParameterSet::ByLayerType(by_type) => {
            let keys: BTreeSet<&str> = by_type.iter().map(|(name, _)| name).collect();
            let used: BTreeSet<&str> = used.iter().copied().collect();
            keys == used
        }
    };
    assert_eq!(result.is_ok(), expected);
}

/// Parses and checks one `rope_parameters` document against `context`.
fn check_document(text: &str, context: &RopeContext, used: &[&str]) {
    let Ok(set) = serde_json::from_str::<RopeParameterSet>(text) else {
        return;
    };
    let encoded = serde_json::to_value(&set).expect("parsed sets serialize");
    let decoded: RopeParameterSet =
        serde_json::from_value(encoded.clone()).expect("serialized sets parse");
    let reencoded = serde_json::to_value(&decoded).expect("parsed sets serialize");
    assert!(
        crate::json_equal(&encoded, &reencoded),
        "{encoded} became {reencoded}"
    );
    let mut layer_types: BTreeSet<&str> = LAYER_TYPES.into_iter().collect();
    if let RopeParameterSet::ByLayerType(by_type) = &set {
        layer_types.extend(by_type.iter().map(|(name, _)| name));
    }
    for layer_type in layer_types {
        check_translation(&set, layer_type, context);
    }
    check_layer_types(&set, used);
}

/// Decodes and checks one translation problem.
///
/// Byte layout: head width and `max_position_embeddings` (table indices), two
/// context flag bytes (bit 0 top-level theta, 1 top-level partial factor, 2
/// top-level original context, 3 whole-head family, 4 DeepSeek YaRN amplitude,
/// 5 softmax correction, 6 legacy dynamic context, 7 text-only MRoPE, 8
/// default-theta override, 9 per-layer theta) followed by their values, a mask
/// of used layer types, and a mode byte: below 32 reads the rest of the input as
/// raw JSON text, otherwise a structured document follows.
pub fn exercise(input: &[u8]) {
    let mut bytes = Bytes::new(input);
    let context = context(&mut bytes);
    let used_mask = bytes.byte();
    let used: Vec<&str> = LAYER_TYPES
        .iter()
        .enumerate()
        .filter(|(index, _)| used_mask & (1 << index) != 0)
        .map(|(_, name)| *name)
        .collect();
    if bytes.byte() < 32 {
        let text = String::from_utf8_lossy(bytes.take(crate::MAX_INPUT_BYTES));
        check_document(&text, &context, &used);
    } else {
        let document = parameters(&mut bytes).to_string();
        check_document(&document, &context, &used);
    }
}

#[cfg(test)]
#[path = "rope_parameters_tests.rs"]
mod tests;
