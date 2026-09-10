use super::*;

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

#[test]
fn rejects_invalid_theta_values() {
    for theta in [0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert!(matches!(
            RopeConfig::new(8, 8, theta, RopeScaling::None),
            Err(RopeConfigError::InvalidTheta { .. })
        ));
    }
}

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

#[test]
fn yarn_rejects_all_invalid_attention_and_beta_values() {
    let yarn = YarnConfig::new(2.0, 4096).expect("valid YaRN config");
    for attention_factor in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(matches!(
            yarn.with_attention_factor(attention_factor),
            Err(RopeConfigError::InvalidYarnAttentionFactor { .. })
        ));
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

#[test]
fn yarn_cross_validates_frequency_base() {
    let yarn = YarnConfig::new(2.0, 4096).expect("valid YaRN config");
    assert_eq!(
        RopeConfig::new(64, 64, 1.0, yarn.into()),
        Err(RopeConfigError::YarnThetaTooSmall { theta: 1.0 })
    );
    assert!(RopeConfig::new(64, 64, 10_000.0, yarn.into()).is_ok());
}

#[test]
fn identity_scaling_factors_are_allowed() {
    assert!(LinearRopeScaling::new(1.0).is_ok());
    assert!(DynamicNtkScaling::new(1.0, 1).is_ok());
    let yarn = YarnConfig::new(1.0, 1).expect("identity extension is valid");
    assert_eq!(yarn.attention_factor(), 1.0);
}

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
