//! Rotary-table evaluation-precision tests: F64 references and historical F32 bit identity.

use core::f64::consts::TAU;

use burn::{
    backend::Flex,
    tensor::{DType, Int, Tensor, TensorData, backend::Backend},
};

use super::{
    RotaryAngles, apply_rotary, inverse_frequencies, inverse_frequencies_f64, mrope_angles,
    mrope_angles_for_dtype, rotary_angles, rotary_angles_for_dtype,
};
use crate::position::{DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling, YarnConfig};
use crate::precision::test_support::{
    assert_bitwise_eq, assert_f32_path_is_distinguishable, assert_matches_f64_reference,
    max_relative_error,
};

/// Deterministic CPU tensors with F32 default values and I32 indices.
type TestBackend = Flex<f32, i32>;

/// Context used by the fixtures; dynamic NTK is above its original context here.
const CONTEXT: usize = 4096;

/// Two rows of absolute positions up to `CONTEXT - 1`.
const POSITIONS: [[i32; 6]; 2] = [[0, 1, 7, 100, 2047, 4095], [3, 5, 11, 999, 1000, 4093]];

/// Every scaling family, full and partial rotary widths, and both YaRN amplitudes.
fn rope_configs() -> Vec<(RopeConfig, usize)> {
    let yarn = YarnConfig::new(4.0, 128).expect("valid YaRN");
    vec![
        (
            RopeConfig::new(8, 8, 10_000.0, RopeScaling::None).expect("full"),
            CONTEXT,
        ),
        (
            RopeConfig::new(8, 6, 10_000.0, RopeScaling::None).expect("partial"),
            CONTEXT,
        ),
        (
            RopeConfig::new(16, 16, 1_000_000.0, RopeScaling::None).expect("large base"),
            CONTEXT,
        ),
        (
            RopeConfig::new(
                8,
                8,
                10_000.0,
                LinearRopeScaling::new(2.0).expect("linear").into(),
            )
            .expect("linear rope"),
            CONTEXT,
        ),
        (
            RopeConfig::new(
                8,
                8,
                100.0,
                DynamicNtkScaling::new(2.0, 1024).expect("dynamic").into(),
            )
            .expect("dynamic rope"),
            512,
        ),
        (
            RopeConfig::new(
                8,
                8,
                100.0,
                DynamicNtkScaling::new(2.0, 1024).expect("dynamic").into(),
            )
            .expect("dynamic rope"),
            CONTEXT,
        ),
        (
            RopeConfig::new(16, 16, 10_000.0, yarn.into()).expect("default YaRN"),
            CONTEXT,
        ),
        (
            RopeConfig::new(
                16,
                12,
                10_000.0,
                yarn.with_attention_factor(1.25)
                    .expect("amplitude")
                    .with_betas(16.0, 2.0)
                    .expect("betas")
                    .into(),
            )
            .expect("partial YaRN"),
            CONTEXT,
        ),
    ]
}

/// Frozen copy of the YaRN correction dimension used by the historical frequencies.
fn historical_yarn_correction_dimension(
    rotations: f64,
    dimension: usize,
    theta: f64,
    original_context: usize,
) -> f64 {
    dimension as f64 * (original_context as f64 / (rotations * TAU)).ln() / (2.0 * theta.ln())
}

/// Frozen copy of `inverse_frequencies` before F64 tables were introduced.
fn historical_inverse_frequencies(config: RopeConfig, evaluated_context: usize) -> Vec<f32> {
    assert_ne!(evaluated_context, 0, "fixtures use a non-zero context");
    let dimension = config.rotary_dimension();
    let half = dimension / 2;
    let (base, divisor) = match config.scaling() {
        RopeScaling::None => (config.theta(), 1.0),
        RopeScaling::Linear(linear) => (config.theta(), linear.factor()),
        RopeScaling::DynamicNtk(dynamic) => {
            let original = dynamic.original_max_position_embeddings();
            if evaluated_context <= original {
                (config.theta(), 1.0)
            } else {
                let context_ratio = evaluated_context as f64 / original as f64;
                let stretch = dynamic.factor() * context_ratio - (dynamic.factor() - 1.0);
                let exponent = dimension as f64 / (dimension as f64 - 2.0);
                (config.theta() * stretch.powf(exponent), 1.0)
            }
        }
        RopeScaling::Yarn(yarn) => {
            let base = config.theta();
            let low = historical_yarn_correction_dimension(
                yarn.beta_fast(),
                dimension,
                base,
                yarn.original_max_position_embeddings(),
            )
            .floor()
            .clamp(0.0, (half.saturating_sub(1)) as f64);
            let high = historical_yarn_correction_dimension(
                yarn.beta_slow(),
                dimension,
                base,
                yarn.original_max_position_embeddings(),
            )
            .ceil()
            .clamp(0.0, (half.saturating_sub(1)) as f64);
            let width = (high - low).max(1e-3);
            return (0..half)
                .map(|index| {
                    let exponent = 2.0 * index as f64 / dimension as f64;
                    let extrapolated = base.powf(-exponent);
                    let interpolated = extrapolated / yarn.factor();
                    let ramp = ((index as f64 - low) / width).clamp(0.0, 1.0);
                    let extrapolation_weight = 1.0 - ramp;
                    (interpolated * (1.0 - extrapolation_weight)
                        + extrapolated * extrapolation_weight) as f32
                })
                .collect();
        }
    };
    (0..half)
        .map(|index| {
            let exponent = 2.0 * index as f64 / dimension as f64;
            (base.powf(-exponent) / divisor) as f32
        })
        .collect()
}

/// Frozen copy of the historical half-width angle to cosine/sine table conversion.
fn historical_finish_angles<B: Backend>(
    config: RopeConfig,
    angle: Tensor<B, 3>,
) -> RotaryAngles<B> {
    let factor = match config.scaling() {
        RopeScaling::Yarn(yarn) => yarn.attention_factor(),
        _ => 1.0,
    };
    let duplicated = Tensor::cat(vec![angle.clone(), angle], 2);
    RotaryAngles {
        cosine: duplicated.clone().cos().mul_scalar(factor).unsqueeze_dim(1),
        sine: duplicated.sin().mul_scalar(factor).unsqueeze_dim(1),
        rotary_dimension: config.rotary_dimension(),
    }
}

/// Frozen copy of the historical F32 `rotary_angles` tensor graph.
fn historical_rotary_angles<B: Backend>(
    config: RopeConfig,
    position_ids: Tensor<B, 2, Int>,
    evaluated_context: usize,
) -> RotaryAngles<B> {
    let device = position_ids.device();
    let frequencies = historical_inverse_frequencies(config, evaluated_context);
    let half = frequencies.len();
    let frequencies = Tensor::<B, 1>::from_data(TensorData::new(frequencies, [half]), &device)
        .reshape([1, 1, half]);
    let angle = position_ids.float().cast(DType::F32).unsqueeze_dim::<3>(2) * frequencies;
    historical_finish_angles(config, angle)
}

/// Frozen copy of the historical F32 `mrope_angles` tensor graph.
fn historical_mrope_angles<B: Backend>(
    config: RopeConfig,
    position_ids: Tensor<B, 3, Int>,
    sections: [usize; 3],
    evaluated_context: usize,
) -> RotaryAngles<B> {
    let half = config.rotary_dimension() / 2;
    let device = position_ids.device();
    let frequencies = historical_inverse_frequencies(config, evaluated_context);
    let frequency = Tensor::<B, 1>::from_data(TensorData::new(frequencies, [half]), &device)
        .reshape([1, 1, half]);
    let axis_angle = |axis| {
        position_ids
            .clone()
            .narrow(0, axis, 1)
            .squeeze_dim::<2>(0)
            .float()
            .cast(DType::F32)
            .unsqueeze_dim::<3>(2)
            * frequency.clone()
    };
    let temporal = axis_angle(0);
    let height = axis_angle(1);
    let width = axis_angle(2);
    let mut temporal_mask = vec![1.0_f32; half];
    let mut height_mask = vec![0.0_f32; half];
    let mut width_mask = vec![0.0_f32; half];
    for index in (1..sections[1].saturating_mul(3).min(half)).step_by(3) {
        temporal_mask[index] = 0.0;
        height_mask[index] = 1.0;
    }
    for index in (2..sections[2].saturating_mul(3).min(half)).step_by(3) {
        temporal_mask[index] = 0.0;
        width_mask[index] = 1.0;
    }
    let dense_mask = |values: Vec<f32>| {
        Tensor::<B, 1>::from_data(TensorData::new(values, [half]), &device).reshape([1, 1, half])
    };
    let angle = temporal * dense_mask(temporal_mask)
        + height * dense_mask(height_mask)
        + width * dense_mask(width_mask);
    historical_finish_angles(config, angle)
}

/// Independent closed-form F64 oracle for every RoPE frequency-scaling family.
///
/// Uses `omega_i = theta^(-2i/d)` with linear division, the dynamic-NTK base
/// `theta * (f * L / L0 - (f - 1))^(d / (d - 2))`, or YaRN's clamped ramp
/// `omega_i * (1 - ramp_i) + (omega_i / f) * ramp_i` between correction
/// dimensions `d ln(L0 / (2 pi beta)) / (2 ln theta)`.
fn oracle_frequencies(config: RopeConfig, evaluated_context: usize) -> Vec<f64> {
    let dimension = config.rotary_dimension() as f64;
    let half = config.rotary_dimension() / 2;
    let theta = config.theta();
    let power = |base: f64, index: usize| base.powf(-2.0 * index as f64 / dimension);
    match config.scaling() {
        RopeScaling::None => (0..half).map(|index| power(theta, index)).collect(),
        RopeScaling::Linear(linear) => (0..half)
            .map(|index| power(theta, index) / linear.factor())
            .collect(),
        RopeScaling::DynamicNtk(dynamic) => {
            let original = dynamic.original_max_position_embeddings() as f64;
            let context = evaluated_context as f64;
            let base = if context <= original {
                theta
            } else {
                let stretch = dynamic.factor() * context / original - (dynamic.factor() - 1.0);
                theta * stretch.powf(dimension / (dimension - 2.0))
            };
            (0..half).map(|index| power(base, index)).collect()
        }
        RopeScaling::Yarn(yarn) => {
            let original = yarn.original_max_position_embeddings() as f64;
            let boundary =
                |beta: f64| dimension * (original / (TAU * beta)).ln() / (2.0 * theta.ln());
            let last = (half - 1) as f64;
            let low = boundary(yarn.beta_fast()).floor().clamp(0.0, last);
            let high = boundary(yarn.beta_slow()).ceil().clamp(0.0, last);
            let width = (high - low).max(1e-3);
            (0..half)
                .map(|index| {
                    let extrapolated = power(theta, index);
                    let ramp = ((index as f64 - low) / width).clamp(0.0, 1.0);
                    extrapolated * (1.0 - ramp) + extrapolated / yarn.factor() * ramp
                })
                .collect()
        }
    }
}

/// Cosine/sine amplitude applied by YaRN, or one for every other scaling family.
fn amplitude(config: RopeConfig) -> f64 {
    match config.scaling() {
        RopeScaling::Yarn(yarn) => yarn.attention_factor(),
        _ => 1.0,
    }
}

/// Host F64 tables `[batch, 1, token, rotary]` from per-channel positions and frequencies.
///
/// `phase(token, frequency)` returns `p * omega` for one complex pair; split-half
/// layout repeats each pair's phase in channels `i` and `i + half`.
fn host_tables(
    tokens: usize,
    half: usize,
    factor: f64,
    phase: impl Fn(usize, usize) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut cosine = Vec::new();
    let mut sine = Vec::new();
    for token in 0..tokens {
        for channel in 0..2 * half {
            let angle = phase(token, channel % half);
            cosine.push(angle.cos() * factor);
            sine.push(angle.sin() * factor);
        }
    }
    (cosine, sine)
}

/// Creates `[2, 6]` position IDs from [`POSITIONS`].
fn position_tensor() -> Tensor<TestBackend, 2, Int> {
    Tensor::from_ints(POSITIONS, &Default::default())
}

/// F64 frequencies match an independent closed form; F32 frequencies are their roundings.
///
/// The F32 vector also reproduces the frozen historical host formula bit for bit.
#[test]
fn frequencies_match_oracles_and_the_historical_f32_rounding() {
    for (config, context) in rope_configs() {
        let exact = inverse_frequencies_f64(config, context).expect("valid context");
        let oracle = oracle_frequencies(config, context);
        let error = max_relative_error(&exact, &oracle, f64::MIN_POSITIVE);
        assert!(error <= 4.0 * f64::EPSILON, "{config:?}: {error:e}");
        let single = inverse_frequencies(config, context).expect("valid context");
        let rounded = exact.iter().map(|value| *value as f32).collect::<Vec<_>>();
        let historical = historical_inverse_frequencies(config, context);
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&single), bits(&rounded), "{config:?}");
        assert_eq!(bits(&single), bits(&historical), "{config:?}");
    }
}

/// F64 activations receive F64 tables matching host `cos(p omega)` and `sin(p omega)`.
///
/// The legacy F32 tables miss the same reference by orders of magnitude more
/// than F64 rounding, so any F32 intermediate would fail the F64 tolerance.
#[test]
fn f64_rotary_tables_match_an_independent_f64_reference() {
    for (config, context) in rope_configs() {
        let half = config.rotary_dimension() / 2;
        let frequencies = inverse_frequencies_f64(config, context).expect("valid context");
        let angles = rotary_angles_for_dtype(config, position_tensor(), context, DType::F64)
            .expect("valid F64 tables");
        let legacy = rotary_angles(config, position_tensor(), context).expect("valid F32 tables");
        assert_eq!(angles.rotary_dimension, config.rotary_dimension());
        assert_eq!(angles.cosine.dims(), [2, 1, 6, config.rotary_dimension()]);
        let (cosine, sine) = host_tables(2 * 6, half, amplitude(config), |token, frequency| {
            f64::from(POSITIONS[token / 6][token % 6]) * frequencies[frequency]
        });
        for (table, legacy, expected, name) in [
            (&angles.cosine, &legacy.cosine, &cosine, "cosine"),
            (&angles.sine, &legacy.sine, &sine, "sine"),
        ] {
            let context = format!("{config:?} {name}");
            let actual = table.to_data();
            assert_eq!(actual.dtype, DType::F64, "{context}");
            assert_matches_f64_reference(&actual, expected, 1.0, &context);
            assert_f32_path_is_distinguishable(&legacy.to_data(), expected, 1.0, &context);
        }
    }
}

/// Reduced and single-precision activations keep the historical F32 tables bit for bit.
#[test]
fn reduced_and_single_precision_tables_match_the_historical_graph_bitwise() {
    for (config, context) in rope_configs() {
        let historical = historical_rotary_angles(config, position_tensor(), context);
        let legacy = rotary_angles(config, position_tensor(), context).expect("valid tables");
        let mut tables = vec![("legacy".to_owned(), legacy)];
        for dtype in [DType::F16, DType::BF16, DType::Flex32, DType::F32] {
            tables.push((
                format!("{dtype:?}"),
                rotary_angles_for_dtype(config, position_tensor(), context, dtype)
                    .expect("valid tables"),
            ));
        }
        for (name, angles) in tables {
            let context = format!("{config:?} {name}");
            assert_eq!(angles.rotary_dimension, historical.rotary_dimension);
            assert_bitwise_eq(
                &angles.cosine.into_data(),
                &historical.cosine.to_data(),
                &context,
            );
            assert_bitwise_eq(
                &angles.sine.into_data(),
                &historical.sine.to_data(),
                &context,
            );
        }
    }
}

/// Multimodal F64 tables use distinct axis phases per interleaved section in F64.
#[test]
fn f64_mrope_tables_match_an_independent_f64_reference() {
    let config = RopeConfig::new(12, 12, 10_000.0, RopeScaling::None).expect("valid rope");
    let sections = [2, 2, 2];
    let axes = [[[1, 50, 4000]], [[7, 3000, 2]], [[999, 11, 4095]]];
    let positions = || Tensor::<TestBackend, 3, Int>::from_ints(axes, &Default::default());
    let frequencies = inverse_frequencies_f64(config, CONTEXT).expect("valid context");
    let angles = mrope_angles_for_dtype(config, positions(), sections, CONTEXT, DType::F64)
        .expect("valid F64 tables");
    let legacy = mrope_angles(config, positions(), sections, CONTEXT).expect("valid F32 tables");
    // Height owns frequencies 1 and 4, width owns 2 and 5, and time owns 0 and 3.
    let axis = |frequency: usize| match frequency % 3 {
        1 if frequency < 3 * sections[1] => 1,
        2 if frequency < 3 * sections[2] => 2,
        _ => 0,
    };
    let (cosine, sine) = host_tables(3, 6, 1.0, |token, frequency| {
        f64::from(axes[axis(frequency)][0][token]) * frequencies[frequency]
    });
    for (table, legacy, expected, name) in [
        (&angles.cosine, &legacy.cosine, &cosine, "cosine"),
        (&angles.sine, &legacy.sine, &sine, "sine"),
    ] {
        let actual = table.to_data();
        assert_eq!(actual.dtype, DType::F64, "{name}");
        assert_matches_f64_reference(&actual, expected, 1.0, name);
        assert_f32_path_is_distinguishable(&legacy.to_data(), expected, 1.0, name);
    }
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let historical = historical_mrope_angles(config, positions(), sections, CONTEXT);
        let actual = mrope_angles_for_dtype(config, positions(), sections, CONTEXT, dtype)
            .expect("valid tables");
        let context = format!("{dtype:?} multimodal");
        assert_bitwise_eq(
            &actual.cosine.into_data(),
            &historical.cosine.into_data(),
            &context,
        );
        assert_bitwise_eq(
            &actual.sine.into_data(),
            &historical.sine.into_data(),
            &context,
        );
    }
}

/// F64 query/key rotation matches the split-half complex rotation and keeps the suffix exact.
#[test]
fn f64_rotation_matches_an_independent_complex_rotation() {
    let device = Default::default();
    let config = RopeConfig::new(8, 6, 10_000.0, RopeScaling::None).expect("partial rope");
    let positions = [[11, 2047, 4093]];
    let frequencies = inverse_frequencies_f64(config, CONTEXT).expect("valid context");
    let values = (0..2 * 3 * 8)
        .map(|index| ((index as f64 + 1.0) * 0.377).cos() * 1.5)
        .collect::<Vec<_>>();
    let input = || {
        Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values.clone(), [1, 2, 3, 8]),
            (&device, DType::F64),
        )
    };
    let ids = || Tensor::<TestBackend, 2, Int>::from_ints(positions, &device);
    let frequencies = &frequencies;
    let expected = values
        .chunks_exact(8)
        .enumerate()
        .flat_map(|(row, channels)| {
            let token = row % 3;
            (0..8).map(move |channel| {
                if channel >= 6 {
                    return channels[channel];
                }
                let pair = channel % 3;
                let angle = f64::from(positions[0][token]) * frequencies[pair];
                if channel < 3 {
                    channels[channel] * angle.cos() - channels[channel + 3] * angle.sin()
                } else {
                    channels[channel] * angle.cos() + channels[channel - 3] * angle.sin()
                }
            })
        })
        .collect::<Vec<_>>();

    let tables = rotary_angles_for_dtype(config, ids(), CONTEXT, DType::F64).expect("tables");
    let (query, key) = apply_rotary(input(), input(), &tables).expect("valid rotation");
    for (rotated, name) in [(query, "query"), (key, "key")] {
        let actual = rotated.into_data();
        assert_eq!(actual.dtype, DType::F64, "{name}");
        assert_matches_f64_reference(&actual, &expected, 1.0, name);
        let suffix = crate::precision::test_support::f64_values(&actual);
        for (index, value) in suffix
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 8 >= 6)
        {
            assert_eq!(value.to_bits(), values[index].to_bits(), "{name} suffix");
        }
    }
    let legacy = rotary_angles(config, ids(), CONTEXT).expect("F32 tables");
    let (narrowed, _) = apply_rotary(input(), input(), &legacy).expect("valid rotation");
    assert_f32_path_is_distinguishable(&narrowed.into_data(), &expected, 1.0, "F32 tables");
}

/// F64 tables convert integer positions above `2^24` exactly instead of through F32.
#[test]
fn f64_tables_distinguish_positions_beyond_f32_integer_precision() {
    let config = RopeConfig::new(4, 4, 10_000.0, RopeScaling::None).expect("valid rope");
    let boundary = 1_i32 << 24;
    let positions = || {
        Tensor::<TestBackend, 2, Int>::from_ints([[boundary, boundary + 1]], &Default::default())
    };
    let context = boundary as usize + 2;
    let frequencies = inverse_frequencies_f64(config, context).expect("valid context");
    let angles =
        rotary_angles_for_dtype(config, positions(), context, DType::F64).expect("F64 tables");
    let (cosine, _) = host_tables(2, 2, 1.0, |token, frequency| {
        f64::from(boundary + token as i32) * frequencies[frequency]
    });
    let actual = angles.cosine.into_data();
    assert_matches_f64_reference(&actual, &cosine, 1.0, "beyond 2^24");
    let values = crate::precision::test_support::f64_values(&actual);
    assert!(
        (values[0] - values[4]).abs() > 0.1,
        "adjacent F64 positions differ"
    );

    let legacy = rotary_angles(config, positions(), context)
        .expect("F32 tables")
        .cosine
        .into_data();
    let legacy = legacy.as_slice::<f32>().expect("F32 storage");
    assert_eq!(
        legacy[0].to_bits(),
        legacy[4].to_bits(),
        "F32 positions round 2^24 + 1 down to 2^24"
    );
}

/// The F64 builders keep every input validation of the historical builders.
#[test]
fn f64_builders_reject_empty_positions_zero_context_and_bad_sections() {
    let device = Default::default();
    let config = RopeConfig::new(8, 6, 10_000.0, RopeScaling::None).expect("valid rope");
    assert!(inverse_frequencies_f64(config, 0).is_err());
    let empty = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(Vec::<i32>::new(), [1, 0]),
        &device,
    );
    assert!(matches!(
        rotary_angles_for_dtype(config, empty, 8, DType::F64),
        Err(crate::Error::Shape { tensor, .. }) if tensor == "rope.position_ids"
    ));
    let positions = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
    assert!(matches!(
        rotary_angles_for_dtype(config, positions, 0, DType::F64),
        Err(crate::Error::Configuration { field, .. }) if field == "rope.evaluated_context"
    ));
    let axes = Tensor::<TestBackend, 3, Int>::from_ints([[[1, 2]], [[3, 4]], [[5, 6]]], &device);
    assert!(matches!(
        mrope_angles_for_dtype(config, axes.clone(), [1, 1, 0], 8, DType::F64),
        Err(crate::Error::Configuration { field, .. }) if field == "mrope.sections"
    ));
    assert!(matches!(
        mrope_angles_for_dtype(config, axes.narrow(0, 0, 2), [1, 1, 1], 8, DType::F64),
        Err(crate::Error::Shape { tensor, .. }) if tensor == "mrope.position_ids"
    ));
}
