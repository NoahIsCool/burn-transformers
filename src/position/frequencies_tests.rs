//! Golden inverse-frequency parity with Hugging Face RoPE initializers and identity reductions.
//!
//! Constants below were generated with the local Hugging Face Transformers 5.2.0
//! `ROPE_INIT_FUNCTIONS` (torch 2.10.0) except proportional RoPE, which 5.2
//! lacks and was generated with Transformers 5.8.0. Each case builds a
//! `PretrainedConfig` with `hidden_size = head_dim`, `num_attention_heads = 1`,
//! `head_dim`, `max_position_embeddings`, and the shown `rope_parameters`, then
//! prints `ROPE_INIT_FUNCTIONS[rope_type](config, "cpu", seq_len=...)`:
//!
//! ```text
//! python3 -I - <<'PY'
//! from transformers import PretrainedConfig
//! from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
//! def show(rope_type, head_dim, max_pos, params, seq_len=None):
//!     c = PretrainedConfig(); c.hidden_size = head_dim; c.num_attention_heads = 1
//!     c.head_dim = head_dim; c.max_position_embeddings = max_pos; c.rope_parameters = dict(params)
//!     inv, att = ROPE_INIT_FUNCTIONS[rope_type](c, "cpu", seq_len=seq_len)
//!     print(att, [float(v) for v in inv.tolist()])
//! show("yarn", 32, 262144, dict(rope_type="yarn", rope_theta=1e4, factor=4.0,
//!      original_max_position_embeddings=65536))
//! # ... one call per constant below, with the parameters named in its test.
//! PY
//! ```
//!
//! The reference evaluates in F32 while this crate evaluates in F64 and rounds
//! once, so values are compared with a relative tolerance of `2e-6`.

use super::{RotaryFrequencies, inverse_frequencies, yarn_correction_range};
use crate::position::{
    DynamicNtkScaling, LinearRopeScaling, Llama3RopeScaling, LongRopeScaling,
    ProportionalRopeScaling, RopeConfig, RopeScaling, YarnConfig,
};

/// The largest magnitude of any integer position, `2^63`, as the device's `f32` angle factor.
const LARGEST_POSITION: f32 = 9_223_372_036_854_775_808.0;

/// Relative tolerance between F64-then-F32 frequencies and the F32 reference.
const RELATIVE_TOLERANCE: f64 = 2e-6;

/// Asserts element-wise relative agreement, treating exact zeros as exact.
fn assert_frequencies(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "frequency count");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(actual);
        if expected == 0.0 {
            assert_eq!(actual, 0.0, "pair {index} must not rotate");
            continue;
        }
        let relative = ((actual - expected) / expected).abs();
        assert!(
            relative <= RELATIVE_TOLERANCE,
            "pair {index}: actual {actual} expected {expected} (relative error {relative})"
        );
    }
}

/// Builds a full-width configuration for a scaling policy.
fn config(dimension: usize, theta: f64, scaling: impl Into<RopeScaling>) -> RopeConfig {
    RopeConfig::new(dimension, dimension, theta, scaling.into()).expect("valid golden config")
}

/// YaRN with an upper correction bound above `d/2 - 1` clamps at `d - 1`, as the reference does.
///
/// Parameters: head_dim 32, theta 1e4, factor 4, original 65536 (HF bounds 10..17;
/// the previous `d/2 - 1` clamp used 10..15).
#[test]
fn yarn_wide_correction_range_matches_reference() {
    let yarn = YarnConfig::new(4.0, 65_536).expect("valid yarn");
    assert_eq!(yarn_correction_range(yarn, 10_000.0, 32), (10.0, 17.0));
    let expected = [
        1.0,
        0.5623413324356079,
        0.3162277638912201,
        0.17782793939113617,
        0.10000000149011612,
        0.05623412877321243,
        0.03162277862429619,
        0.017782794311642647,
        0.009999999776482582,
        0.005623413249850273,
        0.003162277862429619,
        0.001587749575264752,
        0.0007857143646106124,
        0.0003815887321252376,
        0.0001807015942176804,
        8.256296860054135e-05,
    ];
    let frequencies = RotaryFrequencies::new(config(32, 10_000.0, yarn), 1).expect("frequencies");
    assert_frequencies(frequencies.inverse(), &expected);
    assert!((frequencies.amplitude() - 1.138629436111989).abs() < 1e-12);
}

/// gpt-oss-style `truncate = false` keeps fractional ramp bounds.
///
/// Parameters: head_dim 32, theta 150000, factor 32, original 4096, betas 32/1.
#[test]
fn yarn_without_truncation_matches_reference() {
    let yarn = YarnConfig::new(32.0, 4096)
        .expect("valid yarn")
        .with_truncate(false);
    assert!(!yarn.truncate());
    let (low, high) = yarn_correction_range(yarn, 150_000.0, 32);
    assert!(low.fract() != 0.0 && high.fract() != 0.0);
    let expected = [
        1.0,
        0.4747820496559143,
        0.22541800141334534,
        0.1070244237780571,
        0.05081327259540558,
        0.019334999844431877,
        0.006794959306716919,
        0.002093792660161853,
        0.0004564839182421565,
        3.830881178146228e-05,
        1.818833698052913e-05,
        8.635495760245249e-06,
        4.099978468730114e-06,
        1.9465962850517826e-06,
        9.242089618055616e-07,
        4.38797854940276e-07,
    ];
    let frequencies = RotaryFrequencies::new(config(32, 150_000.0, yarn), 1).expect("frequencies");
    assert_frequencies(frequencies.inverse(), &expected);
    assert!((frequencies.amplitude() - 1.3465735902799727).abs() < 1e-12);
}

/// The pinned DeepSeek-V2-Lite YaRN geometry is unchanged by the corrected clamp.
///
/// Parameters: head_dim 16, theta 1e4, factor 40, original 4096, mscale = mscale_all_dim
/// (amplitude 1).
#[test]
fn yarn_deepseek_geometry_matches_reference() {
    let yarn = YarnConfig::new(40.0, 4096)
        .expect("valid yarn")
        .with_attention_factor(1.0)
        .expect("unit amplitude");
    let expected = [
        1.0,
        0.3162277638912201,
        0.10000000149011612,
        0.023914724588394165,
        0.005124999675899744,
        0.0008498621755279601,
        2.499999936844688e-05,
        7.905694474175107e-06,
    ];
    assert_frequencies(
        &inverse_frequencies(config(16, 10_000.0, yarn), 1).expect("frequencies"),
        &expected,
    );
}

/// Llama 3.1 band interpolation covers the high, medium, and low wavelength bands.
///
/// Parameters: head_dim 32, theta 500000, factor 8, low 1, high 4, original 8192.
#[test]
fn llama3_matches_reference() {
    let llama = Llama3RopeScaling::new(8.0, 1.0, 4.0, 8192).expect("valid llama3");
    let expected = [
        1.0,
        0.44036662578582764,
        0.193922758102417,
        0.08539710193872452,
        0.03760603070259094,
        0.016560440883040428,
        0.00729266507551074,
        0.0032114461064338684,
        0.0005248460220173001,
        7.784655463183299e-05,
        3.428102354519069e-05,
        1.5096217794052791e-05,
        6.647869668086059e-06,
        2.927499735960737e-06,
        1.289173155782919e-06,
        5.677088097399974e-07,
    ];
    let frequencies = RotaryFrequencies::new(config(32, 500_000.0, llama), 1).expect("frequencies");
    assert_frequencies(frequencies.inverse(), &expected);
    assert_eq!(frequencies.amplitude(), 1.0);
}

/// LongRoPE factors used by the short/long golden cases.
const SHORT: [f64; 8] = [1.0, 1.02, 1.05, 1.1, 1.3, 1.7, 2.4, 3.1];
/// Long-context factors for the same eight pairs.
const LONG: [f64; 8] = [1.0, 1.25, 1.9, 3.5, 7.0, 12.5, 24.0, 41.0];

/// LongRoPE selects the short table through the original context and the long table after it.
///
/// Parameters: head_dim 16, theta 1e4, original 4096, max positions 131072 (factor 32);
/// `seq_len` 4096 (short) and 4097 (long).
#[test]
fn long_rope_switches_tables_at_the_original_context() {
    let long = LongRopeScaling::new(&SHORT, &LONG, 4096, 32.0).expect("valid longrope");
    let rope = config(16, 10_000.0, long);
    let short_expected = [
        1.0,
        0.31002721190452576,
        0.095238097012043,
        0.028747979551553726,
        0.007692307699471712,
        0.0018601632909849286,
        0.00041666667675599456,
        0.0001020089621306397,
    ];
    let long_expected = [
        1.0,
        0.2529822289943695,
        0.05263157933950424,
        0.009035078808665276,
        0.0014285714132711291,
        0.0002529822231736034,
        4.166666622040793e-05,
        7.712872502452228e-06,
    ];
    let short = RotaryFrequencies::new(rope, 4096).expect("short frequencies");
    let long = RotaryFrequencies::new(rope, 4097).expect("long frequencies");
    assert_frequencies(short.inverse(), &short_expected);
    assert_frequencies(long.inverse(), &long_expected);
    for frequencies in [short, long] {
        assert!((frequencies.amplitude() - 1.1902380714238083).abs() < 1e-12);
    }
}

/// Partial LongRoPE (Phi-4-mini style) with an explicit amplitude matches the reference.
///
/// Parameters: head_dim 16, partial 0.75 (rotary 12), theta 1e4, original 4096,
/// factor 8, attention_factor 1.15, `seq_len` 8000.
#[test]
fn partial_long_rope_with_explicit_amplitude_matches_reference() {
    let long = LongRopeScaling::new(
        &[1.0, 1.1, 1.3, 1.6, 2.0, 2.5],
        &[1.0, 1.5, 3.0, 6.0, 12.0, 24.0],
        4096,
        8.0,
    )
    .expect("valid longrope")
    .with_attention_factor(1.15)
    .expect("valid amplitude");
    let rope =
        RopeConfig::from_partial_factor(16, 0.75, 10_000.0, long.into()).expect("partial longrope");
    let expected = [
        1.0,
        0.14362896978855133,
        0.015471960417926311,
        0.0016666667070239782,
        0.0001795361895347014,
        1.9339957361808047e-05,
    ];
    let frequencies = RotaryFrequencies::new(rope, 8000).expect("frequencies");
    assert_frequencies(frequencies.inverse(), &expected);
    assert_eq!(frequencies.amplitude(), 1.15);
    assert_eq!(frequencies.rotary_dimension(), 12);
}

/// Proportional RoPE rotates a leading fraction of full-head pairs and zeroes the rest.
///
/// Parameters (Transformers 5.8.0): head_dim 16, theta 1e6, fraction 0.25; and head_dim
/// 16, theta 1e4, fraction 0.6, factor 2.
#[test]
fn proportional_matches_reference() {
    let quarter = ProportionalRopeScaling::new(0.25, 1.0).expect("valid proportional");
    assert_frequencies(
        &inverse_frequencies(config(16, 1_000_000.0, quarter), 1).expect("frequencies"),
        &[1.0, 0.17782793939113617, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    let scaled = ProportionalRopeScaling::new(0.6, 2.0).expect("valid proportional");
    assert_frequencies(
        &inverse_frequencies(config(16, 10_000.0, scaled), 1).expect("frequencies"),
        &[
            0.5,
            0.15811388194561005,
            0.05000000074505806,
            0.015811389312148094,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    );
}

/// Linear and dynamic NTK frequencies match the reference closed forms.
///
/// Parameters: head_dim 8, theta 1e4; linear factor 4; dynamic factor 2 with
/// `max_position_embeddings` 4096 evaluated at `seq_len` 10000.
#[test]
fn linear_and_dynamic_match_reference() {
    let linear = LinearRopeScaling::new(4.0).expect("valid linear");
    assert_frequencies(
        &inverse_frequencies(config(8, 10_000.0, linear), 1).expect("frequencies"),
        &[
            0.25,
            0.02500000037252903,
            0.0024999999441206455,
            0.0002500000118743628,
        ],
    );
    let dynamic = DynamicNtkScaling::new(2.0, 4096).expect("valid dynamic");
    assert_frequencies(
        &inverse_frequencies(config(8, 10_000.0, dynamic), 10_000).expect("frequencies"),
        &[
            1.0,
            0.06362354755401611,
            0.004047955386340618,
            0.0002575452672317624,
        ],
    );
}

/// A degenerate YaRN band widens by 0.001 instead of dividing by zero.
#[test]
fn yarn_degenerate_band_is_widened_like_the_reference() {
    // d=4, theta=100, original 10^6: floor(low) = 3 and ceil(high) = 6, which
    // clamps to d - 1 = 3, so the band collapses and the reference adds 0.001.
    let yarn = YarnConfig::new(2.0, 1_000_000).expect("valid yarn");
    assert_eq!(yarn_correction_range(yarn, 100.0, 4), (3.0, 3.001));
    let frequencies = inverse_frequencies(config(4, 100.0, yarn), 1).expect("frequencies");
    assert!(
        frequencies
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    );
}

/// Zero evaluated context is rejected for every policy, including context-free ones.
#[test]
fn zero_context_is_rejected() {
    for scaling in [
        RopeScaling::None,
        LinearRopeScaling::new(2.0).expect("linear").into(),
        Llama3RopeScaling::new(8.0, 1.0, 4.0, 64)
            .expect("llama3")
            .into(),
    ] {
        let error = RotaryFrequencies::new(config(8, 10_000.0, scaling), 0)
            .expect_err("zero context is invalid");
        assert!(matches!(error, crate::Error::Configuration { field, .. }
            if field == "rope.evaluated_context"));
    }
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(64))]

    /// Every policy at its identity setting reproduces unscaled RoPE frequencies exactly.
    #[test]
    fn identity_settings_reduce_to_default_rope(
        pairs in 2usize..=32,
        theta in 2.0f64..1.0e7,
        original in 1usize..100_000,
        context in 1usize..200_000,
    ) {
        let dimension = 2 * pairs;
        let base = inverse_frequencies(config(dimension, theta, RopeScaling::None), context)
            .expect("base");
        let ones = vec![1.0; pairs];
        let identities: Vec<RopeScaling> = vec![
            LinearRopeScaling::new(1.0).expect("linear").into(),
            YarnConfig::new(1.0, original).expect("yarn").into(),
            Llama3RopeScaling::new(1.0, 1.0, 4.0, original).expect("llama3").into(),
            LongRopeScaling::new(&ones, &ones, original, 1.0).expect("longrope").into(),
            ProportionalRopeScaling::new(1.0, 1.0).expect("proportional").into(),
            // Dynamic NTK is the identity only within its original context.
            DynamicNtkScaling::new(1.0, original.max(context)).expect("dynamic").into(),
        ];
        for scaling in identities {
            let frequencies = RotaryFrequencies::new(config(dimension, theta, scaling), context)
                .expect("identity frequencies");
            proptest::prop_assert_eq!(frequencies.amplitude(), 1.0);
            for (actual, expected) in frequencies.inverse().iter().zip(&base) {
                let relative = f64::from((actual - expected).abs()) / f64::from(*expected);
                proptest::prop_assert!(relative <= 1e-6, "{scaling:?}: {actual} vs {expected}");
            }
        }
    }

    /// Every configuration the validators accept has finite `f32` frequencies, angles, and amplitude.
    ///
    /// Bases span `10^-320..10^308` and LongRoPE factors and amplitudes
    /// `10^-50..10^40`, so the search crosses every `f32` and `f64` boundary
    /// (findings ROPE-F32-1/2). Rejection is exact: a base is refused iff it is
    /// below one, a factor iff it does not round to an `f32` in
    /// `[2^-64, f32::MAX]`, and an amplitude iff it does not round to a normal
    /// positive `f32`. Angles are checked as the device forms them, an `f32`
    /// product with the largest integer position `2^63` (review finding
    /// ROPE-ANGLE-OVERFLOW: frequency checks alone missed `p * omega = inf`).
    #[test]
    fn accepted_configurations_have_finite_tables(
        pairs in 1usize..=8,
        theta_exponent in -320.0f64..308.0,
        factor_exponents in proptest::collection::vec(-50.0f64..40.0, 2),
        amplitude_exponent in -50.0f64..40.0,
        context in 1usize..32,
    ) {
        let dimension = 2 * pairs;
        let theta = 10f64.powf(theta_exponent);
        let normal = |value: f64| {
            let rounded = value as f32;
            rounded.is_finite() && rounded >= f32::MIN_POSITIVE
        };
        let factor_in_range = |value: f64| {
            let rounded = value as f32;
            rounded.is_finite() && rounded >= LongRopeScaling::MIN_FACTOR
        };
        let short: Vec<f64> = (0..pairs)
            .map(|index| 10f64.powf(factor_exponents[index % 2]))
            .collect();
        let long = vec![1.0; pairs];
        let amplitude = 10f64.powf(amplitude_exponent);
        let long_rope = LongRopeScaling::new(&short, &long, 16, 2.0)
            .and_then(|scaling| scaling.with_attention_factor(amplitude));
        proptest::prop_assert_eq!(
            long_rope.is_ok(),
            short.iter().all(|&factor| factor_in_range(factor)) && normal(amplitude)
        );
        let yarn = YarnConfig::new(4.0, 16).and_then(|yarn| yarn.with_attention_factor(amplitude));
        proptest::prop_assert_eq!(yarn.is_ok(), normal(amplitude));
        let mut policies = vec![
            RopeScaling::None,
            LinearRopeScaling::new(4.0).expect("linear").into(),
            DynamicNtkScaling::new(4.0, 8).expect("dynamic").into(),
            Llama3RopeScaling::new(4.0, 1.0, 4.0, 16).expect("llama3").into(),
            ProportionalRopeScaling::new(0.5, 2.0).expect("proportional").into(),
        ];
        policies.extend(long_rope.ok().map(RopeScaling::from));
        policies.extend(yarn.ok().map(RopeScaling::from));
        for scaling in policies {
            let accepted = RopeConfig::new(dimension, dimension, theta, scaling);
            // YaRN needs `ln(theta) > 0` and dynamic NTK a rotary width above two.
            let structural = match scaling {
                RopeScaling::Yarn(_) => theta > 1.0,
                RopeScaling::DynamicNtk(_) => dimension > 2,
                _ => true,
            };
            proptest::prop_assert_eq!(accepted.is_ok(), theta >= 1.0 && structural);
            let Ok(config) = accepted else { continue };
            for evaluated in [1, context, 16, 17, 64] {
                let frequencies = RotaryFrequencies::new(config, evaluated).expect("context");
                for omega in frequencies.inverse() {
                    proptest::prop_assert!(omega.is_finite() && *omega >= 0.0, "{config:?}: {omega}");
                    let angle = LARGEST_POSITION * omega;
                    proptest::prop_assert!(angle.is_finite(), "{config:?}: 2^63 * {omega}");
                }
                proptest::prop_assert!(normal(frequencies.amplitude()), "{config:?}");
            }
        }
    }

    /// Extension policies never raise a frequency and keep every value finite.
    #[test]
    fn extension_policies_only_slow_rotation(
        pairs in 2usize..=32,
        theta in 2.0f64..1.0e7,
        factor in 1.0f64..64.0,
        original in 16usize..100_000,
    ) {
        let dimension = 2 * pairs;
        let base = inverse_frequencies(config(dimension, theta, RopeScaling::None), 1)
            .expect("base");
        let policies: Vec<RopeScaling> = vec![
            LinearRopeScaling::new(factor).expect("linear").into(),
            YarnConfig::new(factor, original).expect("yarn").into(),
            YarnConfig::new(factor, original).expect("yarn").with_truncate(false).into(),
            Llama3RopeScaling::new(factor, 1.0, 4.0, original).expect("llama3").into(),
            ProportionalRopeScaling::new(0.5, factor).expect("proportional").into(),
        ];
        for scaling in policies {
            let frequencies = inverse_frequencies(config(dimension, theta, scaling), 1)
                .expect("scaled frequencies");
            for (actual, expected) in frequencies.iter().zip(&base) {
                proptest::prop_assert!(actual.is_finite() && *actual >= 0.0);
                proptest::prop_assert!(*actual <= expected * (1.0 + 1e-6), "{scaling:?}");
                proptest::prop_assert!(*actual >= expected / factor as f32 * (1.0 - 1e-6) || *actual == 0.0);
            }
        }
    }
}
