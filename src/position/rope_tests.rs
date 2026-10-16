//! RoPE configuration invariants, scaling policies, and singular numerical inputs.

use super::*;

/// Full and partial even rotary widths retain their validated head geometry.
#[test]
fn validates_full_and_partial_rotary_dimensions() {
    let full = RopeConfig::new(128, 128, 10_000.0, RopeScaling::None).expect("valid full RoPE");
    assert_eq!(full.head_dimension(), 128);
    assert_eq!(full.rotary_dimension(), 128);
    assert_eq!(full.theta(), 10_000.0);
    assert_eq!(full.scaling(), RopeScaling::None);
    assert_eq!(full.partial_rotary_factor(), 1.0);

    let partial = RopeConfig::from_partial_factor(256, 0.25, 10_000.0, RopeScaling::None)
        .expect("valid partial RoPE");
    assert_eq!(partial.rotary_dimension(), 64);
    assert_eq!(partial.partial_rotary_factor(), 0.25);
}

/// Zero, odd, and over-wide rotary dimensions report their specific configuration errors.
#[test]
fn rejects_every_dimension_failure() {
    assert_eq!(
        RopeConfig::new(0, 2, 10_000.0, RopeScaling::None),
        Err(RopeConfigError::ZeroHeadDimension)
    );
    assert_eq!(
        RopeConfig::new(8, 0, 10_000.0, RopeScaling::None),
        Err(RopeConfigError::ZeroRotaryDimension)
    );
    assert_eq!(
        RopeConfig::new(8, 10, 10_000.0, RopeScaling::None),
        Err(RopeConfigError::RotaryDimensionExceedsHead {
            rotary_dimension: 10,
            head_dimension: 8,
        })
    );
    assert_eq!(
        RopeConfig::new(8, 3, 10_000.0, RopeScaling::None),
        Err(RopeConfigError::OddRotaryDimension {
            rotary_dimension: 3,
        })
    );
}

/// Nonfinite frequency bases and bases below one cannot construct a rotary configuration.
#[test]
fn rejects_invalid_theta_values() {
    let below_one = 1.0 - f64::EPSILON / 2.0;
    for theta in [
        0.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1e-300,
        0.5,
        below_one,
    ] {
        assert!(matches!(
            RopeConfig::new(8, 8, theta, RopeScaling::None),
            Err(RopeConfigError::InvalidTheta { .. })
        ));
    }
}

/// Bases of at least one keep every `f32` inverse frequency finite and in `[0, 1]`.
///
/// Regression for fuzz finding ROPE-F32-1: `theta = 1e-300` used to validate,
/// and pair one of a four-wide head had frequency `1e150`, which is `inf` in
/// `f32` (position zero then had the angle `0 * inf = NaN`). The extreme
/// accepted bases, one (every frequency is one) and `f64::MAX` (every
/// frequency but the first underflows to zero), stay finite.
#[test]
fn accepted_bases_keep_frequencies_finite() {
    assert!(matches!(
        RopeConfig::new(4, 4, 1e-300, RopeScaling::None),
        Err(RopeConfigError::InvalidTheta { theta }) if theta == 1e-300
    ));
    for theta in [1.0, 2.0, 10_000.0, 1e300, f64::MAX] {
        let config = RopeConfig::new(64, 64, theta, RopeScaling::None).expect("theta >= 1");
        let frequencies = crate::position::inverse_frequencies(config, 1).expect("frequencies");
        assert_eq!(frequencies[0], 1.0, "theta {theta}");
        assert!(
            frequencies
                .iter()
                .all(|omega| omega.is_finite() && (0.0..=1.0).contains(omega)),
            "theta {theta}: {frequencies:?}"
        );
    }
}

/// Fractional widths must resolve unambiguously to a positive even integer prefix.
#[test]
fn fractional_dimension_rejects_invalid_or_ambiguous_products() {
    assert_eq!(
        rotary_dimension_from_partial_factor(0, 1.0),
        Err(RopeConfigError::ZeroHeadDimension)
    );
    for factor in [0.0, -0.5, 1.1, f64::INFINITY, f64::NAN] {
        assert!(matches!(
            rotary_dimension_from_partial_factor(16, factor),
            Err(RopeConfigError::InvalidPartialRotaryFactor { .. })
        ));
    }
    assert_eq!(
        rotary_dimension_from_partial_factor(7, 0.5),
        Err(RopeConfigError::NonIntegralPartialRotaryDimension {
            head_dimension: 7,
            factor: 0.5,
            computed: 3.5,
        })
    );
    assert_eq!(
        rotary_dimension_from_partial_factor(12, 0.25),
        Err(RopeConfigError::OddRotaryDimension {
            rotary_dimension: 3,
        })
    );
}

/// Linear and dynamic-NTK policies preserve their validated extension parameters.
#[test]
fn validates_linear_and_dynamic_ntk_scaling() {
    let linear = LinearRopeScaling::new(4.0).expect("valid factor");
    assert_eq!(linear.factor(), 4.0);
    let linear_rope =
        RopeConfig::new(64, 64, 10_000.0, linear.into()).expect("valid linear config");
    assert_eq!(linear_rope.scaling(), RopeScaling::Linear(linear));

    let dynamic = DynamicNtkScaling::new(8.0, 4096).expect("valid dynamic scaling");
    assert_eq!(dynamic.factor(), 8.0);
    assert_eq!(dynamic.original_max_position_embeddings(), 4096);
    assert!(RopeConfig::new(64, 64, 10_000.0, dynamic.into()).is_ok());
    assert_eq!(
        RopeConfig::new(2, 2, 10_000.0, dynamic.into()),
        Err(RopeConfigError::DynamicNtkDimensionTooSmall {
            rotary_dimension: 2,
        })
    );
}

/// Invalid extension multipliers, zero original context, and singular NTK widths fail.
#[test]
fn scaling_rejects_bad_factors_and_contexts() {
    for factor in [0.5, 0.0, -2.0, f64::INFINITY, f64::NAN] {
        assert!(matches!(
            LinearRopeScaling::new(factor),
            Err(RopeConfigError::InvalidScalingFactor { kind: "linear", .. })
        ));
        assert!(matches!(
            DynamicNtkScaling::new(factor, 128),
            Err(RopeConfigError::InvalidScalingFactor {
                kind: "dynamic NTK",
                ..
            })
        ));
        assert!(matches!(
            YarnConfig::new(factor, 128),
            Err(RopeConfigError::InvalidScalingFactor { kind: "YaRN", .. })
        ));
    }
    assert_eq!(
        DynamicNtkScaling::new(2.0, 0),
        Err(RopeConfigError::ZeroOriginalContext {
            kind: "dynamic NTK",
        })
    );
    assert_eq!(
        YarnConfig::new(2.0, 0),
        Err(RopeConfigError::ZeroOriginalContext { kind: "YaRN" })
    );
}

/// YaRN defaults and independent overrides preserve ordered transition boundaries.
#[test]
fn yarn_defaults_and_overrides_are_validated() {
    let yarn = YarnConfig::new(4.0, 8192).expect("valid YaRN config");
    assert_eq!(yarn.factor(), 4.0);
    assert_eq!(yarn.original_max_position_embeddings(), 8192);
    assert!((yarn.attention_factor() - (1.0 + 0.1 * 4.0f64.ln())).abs() < 1e-15);
    assert_eq!(yarn.beta_fast(), 32.0);
    assert_eq!(yarn.beta_slow(), 1.0);

    let customized = yarn
        .with_attention_factor(1.25)
        .expect("positive attention factor")
        .with_betas(64.0, 2.0)
        .expect("ordered betas");
    assert_eq!(customized.attention_factor(), 1.25);
    assert_eq!(customized.beta_fast(), 64.0);
    assert_eq!(customized.beta_slow(), 2.0);

    let fast_only = yarn
        .with_beta_overrides(Some(24.0), None)
        .expect("one-sided fast override");
    assert_eq!(fast_only.beta_fast(), 24.0);
    assert_eq!(fast_only.beta_slow(), 1.0);
    let slow_only = yarn
        .with_beta_overrides(None, Some(2.0))
        .expect("one-sided slow override");
    assert_eq!(slow_only.beta_fast(), 32.0);
    assert_eq!(slow_only.beta_slow(), 2.0);
}

/// YaRN rejects nonfinite, nonpositive, and inverted amplitude or beta parameters.
#[test]
fn yarn_rejects_all_invalid_attention_and_beta_values() {
    let yarn = YarnConfig::new(2.0, 4096).expect("valid YaRN config");
    // 1e39 overflows the f32 tables it scales (`0 * inf = NaN` at position
    // zero) and 1e-46 rounds to zero.
    for attention_factor in [0.0, -1.0, f64::INFINITY, f64::NAN, 1e39, 1e-46] {
        assert!(matches!(
            yarn.with_attention_factor(attention_factor),
            Err(RopeConfigError::InvalidYarnAttentionFactor { .. })
        ));
    }
    for attention_factor in [f64::from(f32::MIN_POSITIVE), f64::from(f32::MAX)] {
        assert_eq!(
            yarn.with_attention_factor(attention_factor)
                .expect("normal f32 amplitude")
                .attention_factor(),
            attention_factor
        );
    }
    for beta in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(matches!(
            yarn.with_betas(beta, 1.0),
            Err(RopeConfigError::InvalidYarnBeta {
                name: "beta_fast",
                ..
            })
        ));
        assert!(matches!(
            yarn.with_betas(32.0, beta),
            Err(RopeConfigError::InvalidYarnBeta {
                name: "beta_slow",
                ..
            })
        ));
    }
    assert_eq!(
        yarn.with_betas(1.0, 1.0),
        Err(RopeConfigError::InvalidYarnBetaOrder {
            beta_fast: 1.0,
            beta_slow: 1.0,
        })
    );
    assert_eq!(
        yarn.with_betas(1.0, 2.0),
        Err(RopeConfigError::InvalidYarnBetaOrder {
            beta_fast: 1.0,
            beta_slow: 2.0,
        })
    );
}

/// YaRN's logarithmic transition rejects a unit frequency base before execution.
#[test]
fn yarn_cross_validates_frequency_base() {
    let yarn = YarnConfig::new(2.0, 4096).expect("valid YaRN config");
    assert_eq!(
        RopeConfig::new(64, 64, 1.0, yarn.into()),
        Err(RopeConfigError::YarnThetaTooSmall { theta: 1.0 })
    );
    assert!(RopeConfig::new(64, 64, 10_000.0, yarn.into()).is_ok());
}

/// The `f32` range check rounds like the tables and rejects NaN, zero, and subnormals.
#[test]
fn normal_positive_f32_range_is_exact_at_its_boundaries() {
    let largest = f64::from(f32::MAX);
    let smallest = f64::from(f32::MIN_POSITIVE);
    for accepted in [smallest, 1.0, largest, largest * (1.0 + f64::EPSILON)] {
        assert!(is_normal_positive_f32(accepted), "{accepted}");
    }
    for rejected in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        -1.0,
        // Just below the boundary: a subnormal after rounding (values within
        // half a subnormal step of it would round up and be accepted).
        smallest * 0.999,
        1e-45,
        largest * 1.000_001,
    ] {
        assert!(!is_normal_positive_f32(rejected), "{rejected}");
    }
}

/// An extension factor of one is valid for every scaling policy.
#[test]
fn identity_scaling_factors_are_allowed() {
    assert!(LinearRopeScaling::new(1.0).is_ok());
    assert!(DynamicNtkScaling::new(1.0, 1).is_ok());
    let yarn = YarnConfig::new(1.0, 1).expect("identity extension is valid");
    assert_eq!(yarn.attention_factor(), 1.0);
}

/// Every error variant explains the invalid scalar or dimensional constraint.
#[test]
fn validation_errors_have_actionable_messages() {
    let errors = [
        RopeConfigError::ZeroHeadDimension.to_string(),
        RopeConfigError::ZeroRotaryDimension.to_string(),
        RopeConfigError::RotaryDimensionExceedsHead {
            rotary_dimension: 10,
            head_dimension: 8,
        }
        .to_string(),
        RopeConfigError::OddRotaryDimension {
            rotary_dimension: 3,
        }
        .to_string(),
        RopeConfigError::InvalidTheta { theta: 0.0 }.to_string(),
        RopeConfigError::InvalidPartialRotaryFactor { factor: 2.0 }.to_string(),
        RopeConfigError::NonIntegralPartialRotaryDimension {
            head_dimension: 7,
            factor: 0.5,
            computed: 3.5,
        }
        .to_string(),
        RopeConfigError::InvalidScalingFactor {
            kind: "linear",
            factor: 0.5,
        }
        .to_string(),
        RopeConfigError::ZeroOriginalContext { kind: "YaRN" }.to_string(),
        RopeConfigError::DynamicNtkDimensionTooSmall {
            rotary_dimension: 2,
        }
        .to_string(),
        RopeConfigError::YarnThetaTooSmall { theta: 1.0 }.to_string(),
        RopeConfigError::InvalidYarnAttentionFactor {
            attention_factor: 0.0,
        }
        .to_string(),
        RopeConfigError::InvalidYarnBeta {
            name: "beta_fast",
            value: 0.0,
        }
        .to_string(),
        RopeConfigError::InvalidYarnBetaOrder {
            beta_fast: 1.0,
            beta_slow: 2.0,
        }
        .to_string(),
    ];
    for message in errors {
        assert!(!message.is_empty());
    }
}

/// Dimension checks report the first failed condition, even when later ones fail too.
#[test]
fn dimension_checks_report_the_first_failure() {
    assert_eq!(
        validate_dimensions(0, 0),
        Err(RopeConfigError::ZeroHeadDimension)
    );
    assert_eq!(
        validate_dimensions(4, 0),
        Err(RopeConfigError::ZeroRotaryDimension)
    );
    // Odd and too wide: the width check comes first.
    assert_eq!(
        validate_dimensions(4, 5),
        Err(RopeConfigError::RotaryDimensionExceedsHead {
            rotary_dimension: 5,
            head_dimension: 4,
        })
    );
    assert_eq!(
        validate_dimensions(5, 5),
        Err(RopeConfigError::OddRotaryDimension {
            rotary_dimension: 5
        })
    );
    assert_eq!(validate_dimensions(5, 4), Ok(()));
    assert_eq!(validate_dimensions(usize::MAX, usize::MAX - 1), Ok(()));
}

/// Each scaling policy adds exactly its own width rule.
#[test]
fn scaling_width_rules_match_each_policy() {
    assert_eq!(
        check_scaling_width(2, 2, ScalingWidthRule::Unconstrained),
        Ok(())
    );
    assert_eq!(
        check_scaling_width(8, 2, ScalingWidthRule::DynamicNtk),
        Err(RopeConfigError::DynamicNtkDimensionTooSmall {
            rotary_dimension: 2
        })
    );
    assert_eq!(
        check_scaling_width(8, 4, ScalingWidthRule::DynamicNtk),
        Ok(())
    );
    assert_eq!(
        check_scaling_width(8, 6, ScalingWidthRule::LongRope { factor_count: 4 }),
        Err(RopeConfigError::LongRopeFactorCount {
            expected: 3,
            actual: 4,
        })
    );
    assert_eq!(
        check_scaling_width(8, 6, ScalingWidthRule::LongRope { factor_count: 3 }),
        Ok(())
    );
    assert_eq!(
        check_scaling_width(8, 6, ScalingWidthRule::FullHead),
        Err(RopeConfigError::ProportionalRequiresFullRotation {
            rotary_dimension: 6,
            head_dimension: 8,
        })
    );
    assert_eq!(
        check_scaling_width(8, 8, ScalingWidthRule::FullHead),
        Ok(())
    );
    assert_eq!(
        validate_original_context("test", 0),
        Err(RopeConfigError::ZeroOriginalContext { kind: "test" })
    );
    assert_eq!(validate_original_context("test", 1), Ok(()));
}
