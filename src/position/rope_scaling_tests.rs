//! Validation, accessors, and cross-checks for Llama 3, LongRoPE, and proportional RoPE.

use super::*;
use crate::position::{RopeConfig, RopeScaling, YarnConfig};

/// Llama 3 accepts the published 3.1 parameters and exposes them unchanged.
#[test]
fn llama3_accepts_reference_parameters() {
    let llama = Llama3RopeScaling::new(8.0, 1.0, 4.0, 8192).expect("Llama 3.1 parameters");
    assert_eq!(llama.factor(), 8.0);
    assert_eq!(llama.low_freq_factor(), 1.0);
    assert_eq!(llama.high_freq_factor(), 4.0);
    assert_eq!(llama.original_max_position_embeddings(), 8192);
}

/// Llama 3 band multipliers must be finite, positive, and strictly increasing.
#[test]
fn llama3_rejects_degenerate_bands_and_factors() {
    for (low, high) in [
        (4.0, 4.0),
        (4.0, 1.0),
        (0.0, 4.0),
        (-1.0, 4.0),
        (1.0, f64::INFINITY),
        (f64::NAN, 4.0),
    ] {
        // NaN never compares equal, so match the variant rather than the payload.
        assert!(
            matches!(
                Llama3RopeScaling::new(8.0, low, high, 8192),
                Err(RopeConfigError::InvalidLlama3FrequencyFactors { .. })
            ),
            "low={low}, high={high}",
        );
    }
    assert!(matches!(
        Llama3RopeScaling::new(0.5, 1.0, 4.0, 8192),
        Err(RopeConfigError::InvalidScalingFactor {
            kind: "Llama 3",
            ..
        })
    ));
    assert_eq!(
        Llama3RopeScaling::new(8.0, 1.0, 4.0, 0),
        Err(RopeConfigError::ZeroOriginalContext { kind: "Llama 3" })
    );
}

/// Wavelengths exactly on a band boundary belong to the interpolated middle band.
#[test]
fn llama3_band_boundaries_follow_strict_reference_comparisons() {
    let llama = Llama3RopeScaling::new(8.0, 1.0, 4.0, 8192).expect("valid");
    let tau = core::f64::consts::TAU;
    // High-frequency band: unchanged.
    assert_eq!(llama.scale_frequency(1.0), 1.0);
    // Low-frequency band: divided by the factor.
    let long = tau / 10_000.0;
    assert_eq!(llama.scale_frequency(long), long / 8.0);
    // Exactly at the low boundary (wavelength 8192): smooth = (1 - 1) / 3 = 0.
    let at_low = tau / 8192.0;
    assert!((llama.scale_frequency(at_low) - at_low / 8.0).abs() < 1e-18);
    // Exactly at the high boundary (wavelength 2048): smooth = (4 - 1) / 3 = 1.
    let at_high = tau / 2048.0;
    assert!((llama.scale_frequency(at_high) - at_high).abs() < 1e-15);
}

/// LongRoPE stores both tables, the default amplitude, and selects tables by context.
#[test]
fn long_rope_stores_tables_and_selects_by_context() {
    let long = LongRopeScaling::new(&[1.0, 2.0], &[3.0, 4.0], 4096, 32.0).expect("valid longrope");
    assert_eq!(long.short_factors(), &[1.0, 2.0]);
    assert_eq!(long.long_factors(), &[3.0, 4.0]);
    assert_eq!(long.frequency_count(), 2);
    assert_eq!(long.original_max_position_embeddings(), 4096);
    assert_eq!(long.factor(), 32.0);
    let expected = (1.0 + 32.0_f64.ln() / 4096.0_f64.ln()).sqrt();
    assert!((long.attention_factor() - expected).abs() < 1e-15);
    assert_eq!(long.factors_for_context(1), &[1.0, 2.0]);
    assert_eq!(long.factors_for_context(4096), &[1.0, 2.0]);
    assert_eq!(long.factors_for_context(4097), &[3.0, 4.0]);
    let debug = format!("{long:?}");
    assert!(debug.contains("short_factors: [1.0, 2.0]") && !debug.contains("1.0, 1.0, 1.0"));
}

/// Unit extension and a single-token original context keep a unit amplitude.
#[test]
fn long_rope_default_amplitude_degenerates_to_one() {
    let unit = LongRopeScaling::new(&[1.0], &[1.0], 4096, 1.0).expect("unit factor");
    assert_eq!(unit.attention_factor(), 1.0);
    let single = LongRopeScaling::new(&[1.0], &[1.0], 1, 8.0).expect("single-token context");
    assert_eq!(single.attention_factor(), 1.0);
    let explicit = unit.with_attention_factor(1.5).expect("explicit amplitude");
    assert_eq!(explicit.attention_factor(), 1.5);
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            unit.with_attention_factor(invalid),
            Err(RopeConfigError::InvalidLongRopeAttentionFactor { .. })
        ));
    }
}

/// LongRoPE rejects empty, mismatched, oversized, and non-finite or non-positive tables.
#[test]
fn long_rope_rejects_invalid_tables() {
    assert_eq!(
        LongRopeScaling::new(&[1.0, 2.0], &[1.0], 4096, 2.0),
        Err(RopeConfigError::LongRopeFactorLengthMismatch { short: 2, long: 1 })
    );
    assert_eq!(
        LongRopeScaling::new(&[], &[], 4096, 2.0),
        Err(RopeConfigError::LongRopeFactorCountOutOfRange {
            count: 0,
            maximum: MAX_LONG_ROPE_FREQUENCIES,
        })
    );
    let too_many = vec![1.0; MAX_LONG_ROPE_FREQUENCIES + 1];
    assert_eq!(
        LongRopeScaling::new(&too_many, &too_many, 4096, 2.0),
        Err(RopeConfigError::LongRopeFactorCountOutOfRange {
            count: MAX_LONG_ROPE_FREQUENCIES + 1,
            maximum: MAX_LONG_ROPE_FREQUENCIES,
        })
    );
    let full = vec![1.0; MAX_LONG_ROPE_FREQUENCIES];
    assert!(LongRopeScaling::new(&full, &full, 4096, 2.0).is_ok());
    // 1e40 overflows f32, 1e-50 flushes to zero after rounding, and 1e-45 is
    // subnormal (ROPE-F32-2: dividing by it overflows the frequency table).
    let subnormal = f64::from(f32::MIN_POSITIVE) / 2.0;
    for value in [
        0.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        1e40,
        1e-50,
        1e-45,
        subnormal,
    ] {
        let error = LongRopeScaling::new(&[1.0, 1.0], &[1.0, value], 4096, 2.0)
            .expect_err("invalid long factor");
        assert!(matches!(
            error,
            RopeConfigError::InvalidLongRopeFactor {
                table: "long_factor",
                index: 1,
                ..
            }
        ));
        let error = LongRopeScaling::new(&[value, 1.0], &[1.0, 1.0], 4096, 2.0)
            .expect_err("invalid short factor");
        assert!(matches!(
            error,
            RopeConfigError::InvalidLongRopeFactor {
                table: "short_factor",
                index: 0,
                ..
            }
        ));
    }
    assert_eq!(
        LongRopeScaling::new(&[1.0], &[1.0], 0, 2.0),
        Err(RopeConfigError::ZeroOriginalContext { kind: "LongRoPE" })
    );
    assert!(matches!(
        LongRopeScaling::new(&[1.0], &[1.0], 4096, 0.5),
        Err(RopeConfigError::InvalidScalingFactor {
            kind: "LongRoPE",
            ..
        })
    ));
}

/// LongRoPE accepts exactly the factors that round to normal positive `f32` values.
///
/// Regression for fuzz finding ROPE-F32-2: the smallest subnormal `1e-45` was
/// accepted and divided a base frequency of one into `7e44`, which is `inf` in
/// `f32`. The boundaries `f32::MIN_POSITIVE` and `f32::MAX` (and an `f64` just
/// above it that rounds down to it) are accepted and keep every frequency finite.
#[test]
fn long_rope_factors_must_be_normal_f32_values() {
    let largest = f64::from(f32::MAX);
    let rounds_to_largest = largest * (1.0 + f64::EPSILON);
    assert_eq!(rounds_to_largest as f32, f32::MAX);
    for factor in [f64::from(f32::MIN_POSITIVE), largest, rounds_to_largest] {
        let long = LongRopeScaling::new(&[factor, 1.0], &[1.0, factor], 8, 2.0)
            .expect("normal f32 factors are accepted");
        // A base of one is the extreme: every unscaled frequency is one.
        for theta in [1.0, 10_000.0] {
            for context in [1, 8, 9] {
                let frequencies = crate::position::inverse_frequencies(
                    RopeConfig::new(4, 4, theta, long.into()).expect("valid"),
                    context,
                )
                .expect("frequencies");
                assert!(
                    frequencies
                        .iter()
                        .all(|omega| omega.is_finite() && *omega >= 0.0),
                    "factor {factor}, theta {theta}, context {context}: {frequencies:?}"
                );
            }
        }
    }
    for factor in [1e-45, f64::from(f32::MIN_POSITIVE) * 0.999] {
        assert!(matches!(
            LongRopeScaling::new(&[factor, 1.0], &[1.0, 1.0], 8, 2.0),
            Err(RopeConfigError::InvalidLongRopeFactor {
                table: "short_factor",
                index: 0,
                ..
            })
        ));
    }
}

/// LongRoPE amplitudes must round to normal positive `f32` values.
///
/// The amplitude multiplies `f32` cosine/sine tables, so `1e39` would turn
/// `sin(0) = 0` at position zero into `0 * inf = NaN`.
#[test]
fn long_rope_amplitudes_must_be_normal_f32_values() {
    let long = LongRopeScaling::new(&[1.0], &[1.0], 8, 2.0).expect("valid");
    for amplitude in [f64::from(f32::MIN_POSITIVE), 1.0, f64::from(f32::MAX)] {
        assert_eq!(
            long.with_attention_factor(amplitude)
                .expect("normal amplitude")
                .attention_factor(),
            amplitude
        );
    }
    for amplitude in [0.0, -1.0, 1e-46, 1e39, f64::INFINITY, f64::NAN] {
        assert!(matches!(
            long.with_attention_factor(amplitude),
            Err(RopeConfigError::InvalidLongRopeAttentionFactor { .. })
        ));
    }
}

/// The rotary configuration requires exactly one LongRoPE factor per complex pair.
#[test]
fn rope_config_cross_checks_long_rope_table_width() {
    let long = LongRopeScaling::new(&[1.0; 4], &[2.0; 4], 64, 2.0).expect("valid");
    assert!(RopeConfig::new(8, 8, 10_000.0, long.into()).is_ok());
    assert_eq!(
        RopeConfig::new(12, 12, 10_000.0, long.into()),
        Err(RopeConfigError::LongRopeFactorCount {
            expected: 6,
            actual: 4,
        })
    );
}

/// Proportional RoPE counts rotated pairs with the reference floor division.
#[test]
fn proportional_counts_rotated_pairs_like_the_reference() {
    let quarter = ProportionalRopeScaling::new(0.25, 1.0).expect("valid");
    assert_eq!(quarter.rotated_fraction(), 0.25);
    assert_eq!(quarter.factor(), 1.0);
    assert_eq!(quarter.rotated_pairs(512), 64);
    assert_eq!(quarter.rotated_pairs(16), 2);
    let sixty = ProportionalRopeScaling::new(0.6, 1.0).expect("valid");
    // int(0.6 * 16 // 2) = int(9.6 // 2) = 4.
    assert_eq!(sixty.rotated_pairs(16), 4);
    assert_eq!(
        ProportionalRopeScaling::new(0.0, 1.0)
            .expect("NoPE")
            .rotated_pairs(16),
        0
    );
    assert_eq!(
        ProportionalRopeScaling::new(1.0, 1.0)
            .expect("full")
            .rotated_pairs(16),
        8
    );
}

/// Proportional RoPE validates its fraction, factor, and full-head rotation.
#[test]
fn proportional_rejects_invalid_parameters_and_partial_rotation() {
    for fraction in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            ProportionalRopeScaling::new(fraction, 1.0),
            Err(RopeConfigError::InvalidProportionalFraction { .. })
        ));
    }
    assert!(matches!(
        ProportionalRopeScaling::new(0.5, 0.5),
        Err(RopeConfigError::InvalidScalingFactor {
            kind: "proportional",
            ..
        })
    ));
    let proportional = ProportionalRopeScaling::new(0.5, 1.0).expect("valid");
    assert_eq!(
        RopeConfig::new(16, 8, 10_000.0, proportional.into()),
        Err(RopeConfigError::ProportionalRequiresFullRotation {
            rotary_dimension: 8,
            head_dimension: 16,
        })
    );
    assert!(RopeConfig::new(16, 16, 10_000.0, proportional.into()).is_ok());
}

/// Context dependence, cached-decode limits, and amplitudes follow each policy's definition.
#[test]
fn scaling_policy_metadata_matches_definitions() {
    let long = LongRopeScaling::new(&[1.0], &[2.0], 128, 4.0).expect("valid");
    let dynamic = crate::position::DynamicNtkScaling::new(2.0, 64).expect("valid");
    let yarn = YarnConfig::new(4.0, 32)
        .expect("valid")
        .with_attention_factor(1.3)
        .expect("valid amplitude");
    let cases: [(RopeScaling, bool, Option<usize>, f64); 7] = [
        (RopeScaling::None, false, None, 1.0),
        (
            crate::position::LinearRopeScaling::new(2.0)
                .expect("valid")
                .into(),
            false,
            None,
            1.0,
        ),
        (dynamic.into(), true, Some(64), 1.0),
        (yarn.into(), false, None, 1.3),
        (
            Llama3RopeScaling::new(8.0, 1.0, 4.0, 8192)
                .expect("valid")
                .into(),
            false,
            None,
            1.0,
        ),
        (long.into(), true, Some(128), long.attention_factor()),
        (
            ProportionalRopeScaling::new(0.5, 1.0)
                .expect("valid")
                .into(),
            false,
            None,
            1.0,
        ),
    ];
    for (scaling, dependent, limit, amplitude) in cases {
        assert_eq!(scaling.is_context_dependent(), dependent, "{scaling:?}");
        assert_eq!(scaling.cached_context_limit(), limit, "{scaling:?}");
        assert_eq!(scaling.amplitude(), amplitude, "{scaling:?}");
    }
}

/// New validation errors name the offending parameters.
#[test]
fn new_errors_have_actionable_messages() {
    let cases = [
        (
            RopeConfigError::InvalidLlama3FrequencyFactors {
                low_freq_factor: 4.0,
                high_freq_factor: 1.0,
            },
            "Llama 3 low_freq_factor (4) and high_freq_factor (1)",
        ),
        (
            RopeConfigError::LongRopeFactorLengthMismatch { short: 2, long: 3 },
            "short_factor has 2 entries but long_factor has 3",
        ),
        (
            RopeConfigError::LongRopeFactorCountOutOfRange {
                count: 0,
                maximum: 128,
            },
            "1..=128 factors per table, got 0",
        ),
        (
            RopeConfigError::InvalidLongRopeFactor {
                table: "long_factor",
                index: 3,
                value: -1.0,
            },
            "long_factor[3] must be a normal positive f32 value (1.1754944e-38..=3.4028235e38), got -1",
        ),
        (
            RopeConfigError::LongRopeFactorCount {
                expected: 6,
                actual: 4,
            },
            "one factor per rotary pair (6), got 4",
        ),
        (
            RopeConfigError::InvalidLongRopeAttentionFactor {
                attention_factor: 0.0,
            },
            "LongRoPE attention factor must be a normal positive f32 value (1.1754944e-38..=3.4028235e38), got 0",
        ),
        (
            RopeConfigError::InvalidProportionalFraction { fraction: 2.0 },
            "rotated fraction must be finite and in [0, 1], got 2",
        ),
        (
            RopeConfigError::ProportionalRequiresFullRotation {
                rotary_dimension: 8,
                head_dimension: 16,
            },
            "rotates the full head (16), got rotary dimension 8",
        ),
    ];
    for (error, fragment) in cases {
        let message = error.to_string();
        assert!(message.contains(fragment), "{message}");
    }
}
