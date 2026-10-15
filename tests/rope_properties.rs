//! Metamorphic properties of the Hugging Face RoPE translator and its frequency tables.
//!
//! The fuzz target `rope_parameters` checks that arbitrary documents translate
//! or fail contextually. These properties relate *pairs* of documents or
//! algorithms, which byte-level fuzzing cannot express:
//!
//! - **Shape equivalence.** A flat mapping and a per-layer-type mapping that
//!   gives a layer type the same parameters translate identically for that
//!   layer (outside the documented per-layer quirks), and the legacy `type`
//!   key means the same as `rope_type`.
//! - **Linear scaling** divides every default inverse frequency by its factor.
//! - **Dynamic NTK** equals default RoPE up to the trained context and only
//!   lowers frequencies beyond it (a larger base).
//! - **YaRN** keeps every frequency between its interpolated value
//!   `omega / factor` and its extrapolated value `omega`.
//!
//! Cases default to 128 and honor `PROPTEST_CASES`; the RNG seed is fixed.

use proptest::prelude::*;
use serde_json::{Value, json};
use transformers::{
    models::config::{RopeContext, RopeParameterSet},
    position::RotaryFrequencies,
};

/// Number of cases: `PROPTEST_CASES` when set, else `default`.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Relative `f32` tolerance for frequencies evaluated through different formulas.
const TOLERANCE: f32 = 4e-7;

/// Parses a `rope_parameters` value.
fn parse(value: &Value) -> Option<RopeParameterSet> {
    serde_json::from_value(value.clone()).ok()
}

/// Inverse frequencies of `value` at `evaluated` context, or `None` when it does not translate.
fn frequencies(value: &Value, context: &RopeContext, evaluated: usize) -> Option<Vec<f32>> {
    let config = parse(value)?
        .to_rope_config("full_attention", context)
        .ok()?;
    Some(
        RotaryFrequencies::new(config, evaluated)
            .expect("non-zero context")
            .inverse()
            .to_vec(),
    )
}

/// Strategy for a flat mapping of any algorithm with plausible field values.
fn flat_mappings() -> impl Strategy<Value = Value> {
    let factor = prop::sample::select(vec![1.0, 2.0, 4.0, 8.0, 0.5]);
    let original = prop::sample::select(vec![16_u64, 64, 1024]);
    prop_oneof![
        Just(json!({"rope_type": "default"})),
        factor.clone().prop_map(|factor| json!({"rope_type": "linear", "factor": factor})),
        (factor.clone(), original.clone()).prop_map(|(factor, original)| {
            json!({"rope_type": "dynamic", "factor": factor, "original_max_position_embeddings": original})
        }),
        (factor.clone(), original.clone(), 1.0f64..4.0, 8.0f64..64.0).prop_map(
            |(factor, original, slow, fast)| json!({
                "rope_type": "yarn", "factor": factor, "original_max_position_embeddings": original,
                "beta_slow": slow, "beta_fast": fast
            })
        ),
        (factor.clone(), original).prop_map(|(factor, original)| json!({
            "rope_type": "llama3", "factor": factor, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": original
        })),
        (prop::sample::select(vec![0.25, 0.5, 1.0]), factor).prop_map(|(fraction, factor)| json!({
            "rope_type": "proportional", "partial_rotary_factor": fraction, "factor": factor
        })),
    ]
    .prop_flat_map(|mapping| {
        (Just(mapping), proptest::option::of(prop::sample::select(vec![10_000.0, 500_000.0])))
    })
    .prop_map(|(mut mapping, theta)| {
        if let Some(theta) = theta {
            mapping["rope_theta"] = json!(theta);
        }
        mapping
    })
}

/// Strategy for a family context: head width, trained context, and top-level theta.
fn contexts() -> impl Strategy<Value = RopeContext> {
    (
        prop::sample::select(vec![4_usize, 8, 64, 128]),
        prop::sample::select(vec![16_usize, 128, 4096]),
        proptest::option::of(prop::sample::select(vec![10_000.0, 1_000_000.0])),
    )
        .prop_map(|(head_dim, max_positions, theta)| {
            RopeContext::new("property", head_dim, max_positions).with_rope_theta(theta)
        })
}

/// Whether `actual` is within [`TOLERANCE`] of `expected`, relatively.
fn close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= TOLERANCE * expected.abs()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: cases(128),
        rng_seed: proptest::test_runner::RngSeed::Fixed(20_261_015),
        ..ProptestConfig::default()
    })]

    /// A one-entry per-layer mapping translates its layer exactly like the flat mapping.
    #[test]
    fn per_layer_mapping_matches_the_flat_mapping(mapping in flat_mappings(), context in contexts()) {
        let flat = parse(&mapping).expect("flat mappings parse");
        let layered = parse(&json!({"full_attention": mapping.clone()})).expect("layered mappings parse");
        prop_assert!(matches!(layered, RopeParameterSet::ByLayerType(_)));
        let flat = flat.to_rope_config("full_attention", &context);
        let layered = layered.to_rope_config("full_attention", &context);
        prop_assert_eq!(flat.as_ref().ok(), layered.as_ref().ok(), "{}", mapping);
        prop_assert_eq!(flat.is_ok(), layered.is_ok());
    }

    /// The legacy `type` key is equivalent to `rope_type`, alone or together with it.
    #[test]
    fn legacy_type_key_is_equivalent(mapping in flat_mappings()) {
        let current = parse(&mapping).expect("flat mappings parse");
        let algorithm = mapping["rope_type"].clone();
        let mut legacy = mapping.clone();
        legacy.as_object_mut().unwrap().remove("rope_type");
        legacy["type"] = algorithm.clone();
        prop_assert_eq!(parse(&legacy), Some(current.clone()));
        let mut both = mapping;
        both["type"] = algorithm;
        prop_assert_eq!(parse(&both), Some(current));
    }

    /// Linear scaling divides every default inverse frequency by its factor.
    #[test]
    fn linear_scaling_divides_frequencies(
        context in contexts(),
        factor in prop::sample::select(vec![1.0, 2.0, 3.0, 8.0]),
        evaluated in 1usize..10_000,
    ) {
        let base = frequencies(&json!({}), &context, evaluated).expect("default RoPE translates");
        let scaled = frequencies(&json!({"rope_type": "linear", "factor": factor}), &context, evaluated)
            .expect("linear RoPE translates");
        for (&omega, &divided) in base.iter().zip(&scaled) {
            prop_assert!(close(divided, (f64::from(omega) / factor) as f32) || close(divided, omega / factor as f32),
                "{omega} / {factor} became {divided}");
        }
    }

    /// Dynamic NTK equals default RoPE up to the trained context and lowers frequencies beyond.
    #[test]
    fn dynamic_ntk_only_lowers_frequencies_past_the_trained_context(
        context in contexts(),
        factor in prop::sample::select(vec![1.0, 2.0, 4.0]),
        evaluated in 1usize..20_000,
    ) {
        prop_assume!(context.head_dim() > 2);
        let mapping = json!({"rope_type": "dynamic", "factor": factor});
        let base = frequencies(&json!({}), &context, evaluated).expect("default RoPE translates");
        let dynamic = frequencies(&mapping, &context, evaluated).expect("dynamic RoPE translates");
        for (&omega, &scaled) in base.iter().zip(&dynamic) {
            if evaluated <= context.max_position_embeddings() {
                prop_assert_eq!(scaled, omega);
            } else {
                prop_assert!(scaled <= omega * (1.0 + TOLERANCE), "{omega} rose to {scaled}");
            }
        }
    }

    /// YaRN frequencies interpolate between `omega / factor` and `omega`.
    #[test]
    fn yarn_frequencies_stay_between_interpolation_and_extrapolation(
        context in contexts(),
        factor in prop::sample::select(vec![1.0, 2.0, 16.0, 40.0]),
        original in prop::sample::select(vec![16_u64, 256, 4096]),
        truncate in any::<bool>(),
    ) {
        let mapping = json!({
            "rope_type": "yarn", "factor": factor,
            "original_max_position_embeddings": original, "truncate": truncate
        });
        let base = frequencies(&json!({}), &context, 1).expect("default RoPE translates");
        let yarn = frequencies(&mapping, &context, 1).expect("YaRN translates");
        for (&omega, &blended) in base.iter().zip(&yarn) {
            let low = omega / factor as f32;
            prop_assert!(blended >= low * (1.0 - TOLERANCE) && blended <= omega * (1.0 + TOLERANCE),
                "{blended} outside [{low}, {omega}]");
        }
    }
}
