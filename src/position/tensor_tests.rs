//! Rotary tensor parity, partial-channel preservation, and scaling-boundary tests.

use burn::{
    backend::Flex,
    tensor::{DType, Int, Tensor, TensorData, Tolerance},
};

use super::{
    RotaryAngles, apply_rotary, apply_rotary_to, interleaved_to_split_half, mrope_angles,
    rotary_angles, rotary_angles_from_frequencies, split_half_to_interleaved,
};
use crate::position::{
    DynamicNtkScaling, LinearRopeScaling, LongRopeScaling, ProportionalRopeScaling, RopeConfig,
    RopeScaling, RotaryFrequencies, YarnConfig, inverse_frequencies,
};

/// Deterministic CPU tensors with F32 values and I32 indices.
type TestBackend = Flex<f32, i32>;

/// Position zero is the identity and partial RoPE leaves suffix channels intact.
#[test]
fn zero_position_and_partial_suffix_are_identity() {
    let device = Default::default();
    let config = RopeConfig::new(6, 4, 10_000.0, RopeScaling::None).expect("valid rope");
    let positions = Tensor::<TestBackend, 2, Int>::from_ints([[0]], &device);
    let angles = rotary_angles(config, positions, 1).expect("valid positions");
    let query = Tensor::from_floats([[[[1.0, 2.0, 3.0, 4.0, 8.0, 9.0]]]], &device);
    let (rotated, _) = apply_rotary(query.clone(), query.clone(), &angles).expect("valid shapes");
    rotated
        .into_data()
        .assert_approx_eq::<f32>(&query.into_data(), Tolerance::absolute(1e-6));
}

/// Split-half rotation matches the scalar complex-number definition.
#[test]
fn split_half_rotation_matches_reference_values() {
    let device = Default::default();
    let config = RopeConfig::new(4, 4, 10_000.0, RopeScaling::None).expect("valid rope");
    let angles = rotary_angles(
        config,
        Tensor::<TestBackend, 2, Int>::from_ints([[1]], &device),
        2,
    )
    .expect("valid positions");
    let input = Tensor::from_floats([[[[1.0, 2.0, 3.0, 4.0]]]], &device);
    let (actual, _) = apply_rotary(input.clone(), input, &angles).expect("valid shapes");
    let expected = [[[[
        1.0_f32 * 1.0_f32.cos() - 3.0 * 1.0_f32.sin(),
        2.0 * 0.01_f32.cos() - 4.0 * 0.01_f32.sin(),
        3.0 * 1.0_f32.cos() + 1.0 * 1.0_f32.sin(),
        4.0 * 0.01_f32.cos() + 2.0 * 0.01_f32.sin(),
    ]]]];
    actual
        .into_data()
        .assert_approx_eq::<f32>(&TensorData::from(expected), Tolerance::absolute(1e-5));
}

/// Equal multimodal axes collapse exactly to ordinary text RoPE.
#[test]
fn equal_mrope_axes_match_standard_angles() {
    let device = Default::default();
    let config = RopeConfig::new(8, 8, 10_000.0, RopeScaling::None).expect("valid rope");
    let text = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 2]], &device);
    let standard = rotary_angles(config, text.clone(), 3).expect("valid text positions");
    let multimodal = text.unsqueeze_dim::<3>(0).repeat_dim(0, 3);
    let multimodal =
        mrope_angles(config, multimodal, [2, 1, 1], 3).expect("valid multimodal positions");
    assert_eq!(standard.cosine.into_data(), multimodal.cosine.into_data());
    assert_eq!(standard.sine.into_data(), multimodal.sine.into_data());
}

/// Linear and dynamic NTK scaling use the declared context semantics.
#[test]
fn inverse_frequency_scaling_matches_closed_forms() {
    let linear = RopeConfig::new(
        4,
        4,
        10_000.0,
        LinearRopeScaling::new(2.0).expect("valid linear").into(),
    )
    .expect("valid rope");
    let frequency = inverse_frequencies(linear, 8).expect("valid context");
    assert!((frequency[0] - 0.5).abs() < 1e-6);
    assert!((frequency[1] - 0.005).abs() < 1e-6);

    let dynamic_scaling = DynamicNtkScaling::new(2.0, 4).expect("valid dynamic");
    let dynamic = RopeConfig::new(4, 4, 100.0, dynamic_scaling.into()).expect("valid rope");
    assert_eq!(
        inverse_frequencies(dynamic, 4).expect("base context"),
        vec![1.0, 0.1]
    );
    let extended = inverse_frequencies(dynamic, 8).expect("extended context");
    assert_eq!(extended[0], 1.0);
    assert!(extended[1] < 0.1);
}

/// YaRN interpolates low frequencies and applies its table amplitude.
#[test]
fn yarn_interpolates_and_scales_tables() {
    let device = Default::default();
    let yarn = YarnConfig::new(4.0, 128)
        .expect("valid yarn")
        .with_attention_factor(1.25)
        .expect("valid amplitude");
    let config = RopeConfig::new(8, 8, 10_000.0, yarn.into()).expect("valid rope");
    let base = inverse_frequencies(
        RopeConfig::new(8, 8, 10_000.0, RopeScaling::None).expect("base rope"),
        256,
    )
    .expect("base frequencies");
    let scaled = inverse_frequencies(config, 256).expect("yarn frequencies");
    for (scaled, base) in scaled.iter().zip(base) {
        assert!(*scaled <= base + f32::EPSILON);
        assert!(*scaled >= base / 4.0 - f32::EPSILON);
    }
    let angles = rotary_angles(
        config,
        Tensor::<TestBackend, 2, Int>::from_ints([[0]], &device),
        256,
    )
    .expect("valid angles");
    angles.cosine.into_data().assert_approx_eq::<f32>(
        &TensorData::from([[[[1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25]]]]),
        Tolerance::absolute(1e-6),
    );
}

/// Shape and section mismatches return contextual errors.
#[test]
fn rejects_invalid_positions_sections_and_application_shapes() {
    let device = Default::default();
    let config = RopeConfig::new(4, 4, 10_000.0, RopeScaling::None).expect("valid rope");
    assert!(inverse_frequencies(config, 0).is_err());
    assert!(
        mrope_angles(
            config,
            Tensor::<TestBackend, 3, Int>::zeros([2, 1, 1], &device),
            [1, 1, 0],
            1,
        )
        .is_err()
    );
    let angles = rotary_angles(
        config,
        Tensor::<TestBackend, 2, Int>::zeros([1, 2], &device),
        2,
    )
    .expect("valid angles");
    assert!(
        apply_rotary(
            Tensor::zeros([1, 2, 1, 4], &device),
            Tensor::zeros([1, 2, 2, 4], &device),
            &angles,
        )
        .is_err()
    );
}

/// Empty axes and inconsistent frequency partitions fail before broadcasting.
#[test]
fn empty_rotary_axes_and_inconsistent_mrope_sections_are_contextual() {
    let device = Default::default();
    let config = RopeConfig::new(8, 6, 10_000.0, RopeScaling::None).expect("valid rope");
    for shape in [[0, 1], [1, 0], [0, 0]] {
        let positions = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(Vec::<i32>::new(), shape),
            &device,
        );
        let error = rotary_angles(config, positions, 8).expect_err("nonempty axes required");
        assert!(
            matches!(error, crate::Error::Shape { tensor, expected, actual }
            if tensor == "rope.position_ids"
                && expected == "non-empty [batch, sequence]"
                && actual == format!("[{}, {}]", shape[0], shape[1]))
        );
    }
    for shape in [[3, 0, 1], [3, 1, 0], [3, 0, 0]] {
        let positions = Tensor::<TestBackend, 3, Int>::from_data(
            TensorData::new(Vec::<i32>::new(), shape),
            &device,
        );
        let error = mrope_angles(config, positions, [1, 1, 1], 8)
            .expect_err("all three axes must describe a nonempty batch");
        assert!(
            matches!(error, crate::Error::Shape { tensor, expected, actual }
            if tensor == "mrope.position_ids"
                && expected == "[3, batch>0, sequence>0]"
                && actual == format!("[{}, {}, {}]", shape[0], shape[1], shape[2]))
        );
    }
    let positions =
        Tensor::<TestBackend, 3, Int>::from_ints([[[1, 2]], [[3, 4]], [[5, 6]]], &device);
    for sections in [[0, 0, 0], [1, 1, 0], [2, 1, 1]] {
        let error = mrope_angles(config, positions.clone(), sections, 8)
            .expect_err("partition must account for every complex frequency");
        assert!(
            matches!(error, crate::Error::Configuration { field, reason }
            if field == "mrope.sections"
                && reason == format!("must sum to 3, found {}", sections.iter().sum::<usize>()))
        );
    }
    let error = mrope_angles(config, positions.clone(), [usize::MAX, 1, 0], 8)
        .expect_err("partition sum must not wrap");
    assert!(
        matches!(error, crate::Error::Configuration { field, reason }
        if field == "mrope.sections" && reason == "sum overflows usize")
    );
    let valid = mrope_angles(config, positions, [1, 1, 1], 8)
        .expect("correcting the section partition succeeds");
    assert_eq!(valid.cosine.dims(), [1, 1, 2, 6]);
    assert_eq!(valid.sine.dims(), [1, 1, 2, 6]);
}

/// Distinct temporal/height/width phases occupy interleaved complex frequencies.
#[test]
fn distinct_mrope_axes_match_a_scalar_complex_phase_oracle() {
    let device = Default::default();
    let config = RopeConfig::new(8, 6, 10_000.0, RopeScaling::None).expect("partial rope");
    let axes = [[[1, 2]], [[3, 4]], [[5, 6]]];
    let angles = mrope_angles(
        config,
        Tensor::<TestBackend, 3, Int>::from_ints(axes, &device),
        [1, 1, 1],
        8,
    )
    .expect("one complex frequency per axis");
    let cosine = angles
        .cosine
        .into_data()
        .to_vec::<f32>()
        .expect("F32 cosine");
    let sine = angles.sine.into_data().to_vec::<f32>().expect("F32 sine");
    for token in 0..2 {
        for (frequency, axis) in axes.iter().enumerate() {
            // theta^(-2j/d) multiplies the selected axis; split-half RoPE repeats
            // each complex phase for its real and imaginary tensor channels.
            let phase =
                f64::from(axis[0][token]) * 10_000.0_f64.powf(-2.0 * frequency as f64 / 6.0);
            for channel in [frequency, frequency + 3] {
                let index = token * 6 + channel;
                assert!(cosine[index].is_finite() && sine[index].is_finite());
                assert!((f64::from(cosine[index]) - phase.cos()).abs() < 1e-6);
                assert!((f64::from(sine[index]) - phase.sin()).abs() < 1e-6);
            }
        }
    }
    assert!((cosine[0] - cosine[6]).abs() > 0.5);
}

/// Oversized height and width sections are clamped by the interleaving, not rejected.
///
/// With `half = 6` frequencies, height can hold `(6 + 1) / 3 = 2` and width
/// `6 / 3 = 2`, so sections `[0, 3, 3]` give height frequencies 1 and 4,
/// width 2 and 5, and temporal the remaining 0 and 3, as Transformers'
/// `[1 : 3 * h : 3]` and `[2 : 3 * w : 3]` slices do.
#[test]
fn oversized_mrope_sections_are_clamped_like_the_reference_slices() {
    let device = Default::default();
    let config = RopeConfig::new(12, 12, 10_000.0, RopeScaling::None).expect("full rope");
    let positions = [1, 3, 5];
    let angles = mrope_angles(
        config,
        Tensor::<TestBackend, 3, Int>::from_ints([[[1]], [[3]], [[5]]], &device),
        [0, 3, 3],
        8,
    )
    .expect("sections sum to half the rotary width");
    let cosine = angles
        .cosine
        .into_data()
        .to_vec::<f32>()
        .expect("F32 cosine");
    let sine = angles.sine.into_data().to_vec::<f32>().expect("F32 sine");
    for (frequency, axis) in [0, 1, 2, 0, 1, 2].into_iter().enumerate() {
        let phase = f64::from(positions[axis]) * 10_000.0_f64.powf(-2.0 * frequency as f64 / 12.0);
        for channel in [frequency, frequency + 6] {
            assert!((f64::from(cosine[channel]) - phase.cos()).abs() < 1e-6);
            assert!((f64::from(sine[channel]) - phase.sin()).abs() < 1e-6);
        }
    }
}

/// Deterministic `[batch, heads, sequence, width]` activations with distinct entries.
fn activations(shape: [usize; 4]) -> Tensor<TestBackend, 4> {
    let count = shape.iter().product::<usize>();
    let device = Default::default();
    (Tensor::<TestBackend, 1, Int>::arange(0..count as i64, &device)
        .float()
        .reshape(shape)
        / 7.0)
        .sin()
}

/// The single-tensor primitive equals pair rotation and accepts independent lengths.
#[test]
fn single_tensor_rotation_matches_pair_rotation_and_supports_cross_lengths() {
    let device = Default::default();
    let config = RopeConfig::new(6, 4, 10_000.0, RopeScaling::None).expect("partial rope");
    let positions = Tensor::<TestBackend, 2, Int>::from_ints([[3, 4, 5]], &device);
    let angles = rotary_angles(config, positions, 6).expect("angles");
    let query = activations([1, 2, 3, 6]);
    let key = activations([1, 2, 3, 6]) * 0.5;
    let (pair_query, pair_key) = apply_rotary(query.clone(), key.clone(), &angles).expect("pair");
    assert_eq!(
        apply_rotary_to(query, &angles).expect("query").into_data(),
        pair_query.into_data()
    );
    assert_eq!(
        apply_rotary_to(key, &angles).expect("key").into_data(),
        pair_key.into_data()
    );

    // A single cached key position rotates independently of a three-token query.
    let single = rotary_angles(
        config,
        Tensor::<TestBackend, 2, Int>::from_ints([[5]], &device),
        6,
    )
    .expect("single angle");
    let rotated = apply_rotary_to(activations([1, 2, 1, 6]), &single).expect("one key");
    assert_eq!(rotated.dims(), [1, 2, 1, 6]);
    let error = apply_rotary_to(activations([1, 2, 2, 6]), &single).expect_err("length mismatch");
    assert!(matches!(error, crate::Error::Shape { tensor, .. } if tensor == "rope.input_angles"));
    let error = apply_rotary_to(activations([1, 2, 1, 2]), &single).expect_err("too narrow");
    assert!(matches!(error, crate::Error::Shape { tensor, .. } if tensor == "rope.input_angles"));
}

/// Tables are created in F32, convert once with `to_dtype`, and rotate F16 operands in F16.
#[test]
fn tables_are_f32_and_convert_to_the_operand_dtype() {
    let device = Default::default();
    let config = RopeConfig::new(4, 4, 10_000.0, RopeScaling::None).expect("rope");
    let angles = rotary_angles(
        config,
        Tensor::<TestBackend, 2, Int>::from_ints([[1, 2049]], &device),
        2050,
    )
    .expect("angles");
    assert_eq!(angles.cosine.dtype(), DType::F32);
    assert_eq!(angles.sine.dtype(), DType::F32);
    let half = angles.clone().to_dtype(DType::F16);
    assert_eq!(half.cosine.dtype(), DType::F16);
    assert_eq!(half.sine.dtype(), DType::F16);
    let unchanged = angles.clone().to_dtype(DType::F32);
    assert_eq!(
        unchanged.cosine.into_data(),
        angles.cosine.clone().into_data()
    );

    let query = activations([1, 1, 2, 4]).cast(DType::F16);
    let lazily = apply_rotary_to(query.clone(), &angles).expect("lazy cast");
    // `eagerly` uses tables cast once up front; the results must be identical.
    let eagerly = apply_rotary_to(query.clone(), &half).expect("pre-cast tables");
    assert_eq!(lazily.dtype(), DType::F16);
    assert_eq!(lazily.into_data(), eagerly.into_data());

    // Mixed-dtype Q/K each rotate in their own precision.
    let key = activations([1, 1, 2, 4]);
    let (rotated_query, rotated_key) =
        apply_rotary(query, key, &angles).expect("Q and K may use different dtypes");
    assert_eq!(rotated_query.dtype(), DType::F16);
    assert_eq!(rotated_key.dtype(), DType::F32);
}

/// Precomputed frequencies reproduce per-call tables exactly.
#[test]
fn precomputed_frequencies_reproduce_rotary_angles() {
    let device = Default::default();
    let yarn = YarnConfig::new(4.0, 64)
        .expect("yarn")
        .with_attention_factor(1.2)
        .expect("amplitude");
    let config = RopeConfig::new(8, 8, 10_000.0, yarn.into()).expect("rope");
    let positions = Tensor::<TestBackend, 2, Int>::from_ints([[0, 7, 300], [1, 2, 3]], &device);
    let direct = rotary_angles(config, positions.clone(), 301).expect("direct");
    let frequencies = RotaryFrequencies::new(config, 301).expect("frequencies");
    let shared = rotary_angles_from_frequencies(&frequencies, positions).expect("shared");
    assert_eq!(direct.cosine.into_data(), shared.cosine.into_data());
    assert_eq!(direct.sine.into_data(), shared.sine.into_data());
    assert_eq!(direct.rotary_dimension, shared.rotary_dimension);
}

/// LongRoPE tables carry their amplitude and switch factor tables with the evaluated context.
#[test]
fn long_rope_tables_scale_amplitude_and_switch_with_context() {
    let device = Default::default();
    let long = LongRopeScaling::new(&[1.0, 2.0], &[1.0, 8.0], 16, 4.0)
        .expect("longrope")
        .with_attention_factor(1.5)
        .expect("amplitude");
    let config = RopeConfig::new(4, 4, 10_000.0, long.into()).expect("rope");
    let positions = Tensor::<TestBackend, 2, Int>::from_ints([[0, 10]], &device);
    let short = rotary_angles(config, positions.clone(), 16).expect("short");
    let long = rotary_angles(config, positions, 17).expect("long");
    let short_cosine = short.cosine.into_data().to_vec::<f32>().expect("f32");
    let long_cosine = long.cosine.into_data().to_vec::<f32>().expect("f32");
    // Position zero has cosine 1 scaled by the amplitude in every channel.
    assert!(short_cosine[..4].iter().all(|&value| value == 1.5));
    // Pair 1 at position 10: frequency 0.01 / 2 (short) versus 0.01 / 8 (long).
    let expected_short = 1.5 * (10.0_f64 * 0.01 / 2.0).cos();
    let expected_long = 1.5 * (10.0_f64 * 0.01 / 8.0).cos();
    assert!((f64::from(short_cosine[5]) - expected_short).abs() < 1e-6);
    assert!((f64::from(long_cosine[5]) - expected_long).abs() < 1e-6);
}

/// Every accepted LongRoPE factor keeps the angle tables finite at every integer position.
///
/// Review finding ROPE-ANGLE-OVERFLOW: a factor of `f32::MIN_POSITIVE` kept
/// the inverse frequency `1 / r` finite (`8.5e37`), but the angle `p / r`
/// overflowed from position 4 on and `cos(inf)` is NaN. Factors below
/// [`LongRopeScaling::MIN_FACTOR`] (`2^-64`) are now refused, and those at or
/// above it stay finite up to the largest `i32` position (and, by the bound
/// `2^63 * 2^64 = 2^127 < f32::MAX`, every `i64` position too).
#[test]
fn accepted_long_rope_factors_keep_angles_finite_at_every_position() {
    let device = Default::default();
    let floor = f64::from(LongRopeScaling::MIN_FACTOR);
    assert_eq!(floor, 2f64.powi(-64));
    for factor in [
        1e-45,
        f64::from(f32::MIN_POSITIVE),
        1e-37,
        1e-20,
        floor * 0.999,
        floor,
        1e-19,
        1.0,
    ] {
        let Ok(long) = LongRopeScaling::new(&[factor, 1.0], &[1.0, 1.0], 4096, 2.0) else {
            assert!(factor < floor, "factor {factor} must be accepted");
            continue;
        };
        assert!(factor >= floor, "factor {factor} must be refused");
        let config = RopeConfig::new(4, 4, 1.0, long.into()).expect("rope");
        let positions = Tensor::<TestBackend, 2, Int>::from_ints(
            [[0, 1, 3, 4, 5, 35, 4095, 1 << 24, i32::MAX - 1, i32::MAX]],
            &device,
        );
        let angles = rotary_angles(config, positions, 4096).expect("angles");
        for table in [angles.cosine, angles.sine] {
            let values = table.into_data().to_vec::<f32>().expect("f32 table");
            assert!(
                values.iter().all(|value| value.is_finite()),
                "factor {factor}: {values:?}"
            );
        }
    }
}

/// Proportional RoPE leaves its zero-frequency pairs, which interleave the head, unrotated.
#[test]
fn proportional_tail_pairs_are_not_rotated() {
    let device = Default::default();
    let proportional = ProportionalRopeScaling::new(0.5, 1.0).expect("proportional");
    let config = RopeConfig::new(8, 8, 10_000.0, proportional.into()).expect("rope");
    let angles = rotary_angles(
        config,
        Tensor::<TestBackend, 2, Int>::from_ints([[9]], &device),
        10,
    )
    .expect("angles");
    let input = activations([1, 1, 1, 8]);
    let rotated = apply_rotary_to(input.clone(), &angles).expect("rotated");
    let input = input.into_data().to_vec::<f32>().expect("input");
    let rotated = rotated.into_data().to_vec::<f32>().expect("rotated");
    // Pairs (2, 6) and (3, 7) have frequency zero.
    for channel in [2, 3, 6, 7] {
        assert_eq!(rotated[channel], input[channel], "channel {channel}");
    }
    for channel in [0, 4] {
        assert_ne!(rotated[channel], input[channel], "channel {channel}");
    }
}

/// Interleaved pairs round-trip and rotate exactly as adjacent complex numbers.
#[test]
fn interleaved_layout_round_trips_and_matches_complex_rotation() {
    let device = Default::default();
    let config = RopeConfig::new(6, 6, 100.0, RopeScaling::None).expect("rope");
    let angles = rotary_angles(
        config,
        Tensor::<TestBackend, 2, Int>::from_ints([[3]], &device),
        4,
    )
    .expect("angles");
    let interleaved = activations([1, 1, 1, 6]);
    let split = interleaved_to_split_half(interleaved.clone()).expect("split");
    assert_eq!(
        split_half_to_interleaved(split.clone())
            .expect("interleave")
            .into_data(),
        interleaved.clone().into_data()
    );
    let rotated = split_half_to_interleaved(apply_rotary_to(split, &angles).expect("rotate"))
        .expect("interleave")
        .into_data()
        .to_vec::<f32>()
        .expect("f32");
    let input = interleaved.into_data().to_vec::<f32>().expect("f32");
    for pair in 0..3 {
        let phase = 3.0_f64 * 100.0_f64.powf(-2.0 * pair as f64 / 6.0);
        let (real, imaginary) = (f64::from(input[2 * pair]), f64::from(input[2 * pair + 1]));
        let expected_real = real * phase.cos() - imaginary * phase.sin();
        let expected_imaginary = real * phase.sin() + imaginary * phase.cos();
        assert!((f64::from(rotated[2 * pair]) - expected_real).abs() < 1e-5);
        assert!((f64::from(rotated[2 * pair + 1]) - expected_imaginary).abs() < 1e-5);
    }
    for (function, name) in [
        (
            interleaved_to_split_half::<TestBackend> as fn(_) -> _,
            "rope.interleaved_input",
        ),
        (
            split_half_to_interleaved::<TestBackend>,
            "rope.split_half_input",
        ),
    ] {
        let error = function(activations([1, 1, 1, 3])).expect_err("odd width");
        assert!(matches!(error, crate::Error::Shape { tensor, .. } if tensor == name));
    }
}

/// Hand-assembled tables with an odd rotary width are rejected before slicing.
#[test]
fn odd_rotary_tables_are_rejected() {
    let device = Default::default();
    let angles = RotaryAngles::<TestBackend> {
        cosine: Tensor::ones([1, 1, 1, 3], &device),
        sine: Tensor::zeros([1, 1, 1, 3], &device),
        rotary_dimension: 3,
    };
    let error = apply_rotary_to(activations([1, 1, 1, 4]), &angles).expect_err("odd width");
    assert!(matches!(error, crate::Error::Shape { tensor, .. } if tensor == "rope.input_angles"));
}

/// Marks the re-executed test binary that runs the half-precision child.
const HALF_DEFAULT_CHILD: &str = "TRANSFORMERS_ROPE_HALF_DEFAULT_CHILD";

/// RoPE tables remain exact F32 when the device's default float dtype is F16.
///
/// Burn's default dtype is per device and process-global, so the check runs in
/// a fresh child process of this test binary that sets F16 before creating any
/// tensor. Previously, positions converted through the default dtype rounded
/// 2049 to 2048 in F16, and F16 frequencies were multiplied with F32 angles.
#[test]
fn rope_tables_stay_f32_under_half_precision_default_dtype() {
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "position::tensor::tests::half_precision_default_child",
            "--ignored",
            "--test-threads=1",
        ])
        .env(HALF_DEFAULT_CHILD, "1")
        .output()
        .expect("spawn child test process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "child failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Child half of [`rope_tables_stay_f32_under_half_precision_default_dtype`].
#[test]
#[ignore = "runs in a fresh process spawned by rope_tables_stay_f32_under_half_precision_default_dtype"]
fn half_precision_default_child() {
    if std::env::var_os(HALF_DEFAULT_CHILD).is_none() {
        return;
    }
    let device = Default::default();
    burn::tensor::set_default_dtypes::<TestBackend>(
        &device,
        DType::F16,
        burn::tensor::IntDType::I32,
    )
    .expect("fresh process has no tensors yet");
    assert_eq!(
        Tensor::<TestBackend, 1>::zeros([1], &device).dtype(),
        DType::F16
    );
    let config = RopeConfig::new(2, 2, 10_000.0, RopeScaling::None).expect("rope");
    let positions = Tensor::<TestBackend, 2, Int>::from_ints([[2049]], &device);
    let angles = rotary_angles(config, positions, 2050).expect("angles");
    assert_eq!(angles.cosine.dtype(), DType::F32);
    let cosine = angles
        .cosine
        .into_data()
        .to_vec::<f32>()
        .expect("f32 table");
    assert!((f64::from(cosine[0]) - 2049.0_f64.cos()).abs() < 1e-4);
    let mrope = mrope_angles(
        config,
        Tensor::<TestBackend, 3, Int>::from_ints([[[2049]], [[2049]], [[2049]]], &device),
        [1, 0, 0],
        2050,
    )
    .expect("mrope angles");
    let cosine = mrope.cosine.into_data().to_vec::<f32>().expect("f32 table");
    assert!((f64::from(cosine[0]) - 2049.0_f64.cos()).abs() < 1e-4);
}

/// Q/K batch, sequence, or width disagreement, or Q against the tables, is one shape error.
///
/// Batch, sequence, and width are compared between Q and K before the tables
/// are checked against Q, so each mismatch is rejected even when the other
/// operand alone would fit the tables. Head counts may differ, as in grouped
/// query attention, so a Q with more heads than K still rotates.
#[test]
fn query_key_rotation_rejects_every_operand_mismatch() {
    let device = Default::default();
    let config = RopeConfig::new(4, 4, 10_000.0, RopeScaling::None).expect("rope");
    let angles = rotary_angles(
        config,
        Tensor::<TestBackend, 2, Int>::from_ints([[0, 1]], &device),
        2,
    )
    .expect("angles");
    let fitting = Tensor::<TestBackend, 4>::zeros([1, 2, 2, 4], &device);
    for (query, key) in [
        // Q/K batch.
        ([1, 2, 2, 4], [2, 2, 2, 4]),
        // Q/K sequence.
        ([1, 2, 2, 4], [1, 2, 3, 4]),
        // Q/K width.
        ([1, 2, 2, 4], [1, 2, 2, 6]),
        // Q batch against the tables.
        ([2, 2, 2, 4], [2, 2, 2, 4]),
        // Q width below the rotary width.
        ([1, 2, 2, 2], [1, 2, 2, 2]),
    ] {
        let error = apply_rotary(
            Tensor::<TestBackend, 4>::zeros(query, &device),
            Tensor::<TestBackend, 4>::zeros(key, &device),
            &angles,
        )
        .unwrap_err();
        assert!(
            matches!(&error, crate::Error::Shape { tensor, actual, .. }
                if tensor == "rope.query_key_angles"
                    && actual.starts_with(&format!("query={query:?}, key={key:?}"))),
            "{query:?} {key:?}: {error:?}"
        );
    }
    let (query, key) = apply_rotary(
        Tensor::<TestBackend, 4>::zeros([1, 4, 2, 4], &device),
        fitting.clone(),
        &angles,
    )
    .expect("grouped query heads rotate");
    assert_eq!((query.dims(), key.dims()), ([1, 4, 2, 4], [1, 2, 2, 4]));
    assert!(apply_rotary(fitting.clone(), fitting, &angles).is_ok());
}

/// Converting tables casts both when only one already has the requested dtype.
#[test]
fn table_conversion_casts_mixed_dtype_tables() {
    let device = Default::default();
    let mixed = RotaryAngles::<TestBackend> {
        cosine: Tensor::ones([1, 1, 1, 2], &device),
        sine: Tensor::<TestBackend, 4>::zeros([1, 1, 1, 2], &device).cast(DType::F16),
        rotary_dimension: 2,
    };
    let converted = mixed.to_dtype(DType::F32);
    assert_eq!(converted.cosine.dtype(), DType::F32);
    assert_eq!(converted.sine.dtype(), DType::F32);
    assert_eq!(converted.rotary_dimension, 2);
    assert_eq!(
        converted.sine.into_data().to_vec::<f32>().unwrap(),
        [0.0, 0.0]
    );
}

/// Exact bit patterns of a floating tensor, read in its own dtype.
fn bit_pattern(tensor: Tensor<TestBackend, 4>) -> Vec<u64> {
    let data = tensor.into_data();
    match data.dtype {
        DType::F64 => data
            .to_vec::<f64>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect(),
        DType::F32 => data
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        DType::F16 => data
            .to_vec::<burn::tensor::f16>()
            .unwrap()
            .iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        DType::BF16 => data
            .to_vec::<burn::tensor::bf16>()
            .unwrap()
            .iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        other => panic!("unexpected dtype {other:?}"),
    }
}

/// Rotates `input` both ways and requires identical bits, shape, and dtype.
///
/// `fused` states whether the Flex kernel must handle the layout itself
/// (`false` for the flipped view it hands back to the broadcast form).
fn assert_fused_matches_broadcast(
    input: Tensor<TestBackend, 4>,
    angles: &RotaryAngles<TestBackend>,
    fused: bool,
    label: &str,
) {
    let (cosine, sine) = super::tables_in(angles, input.dtype());
    let rotary = angles.rotary_dimension;
    assert_eq!(
        super::flex::rotate_prefix(&input, &cosine, &sine, rotary).is_some(),
        fused,
        "{label}"
    );
    let expected = super::rotate_prefix_broadcast(input.clone(), &cosine, &sine, rotary);
    let actual = super::rotate_prefix(input, &cosine, &sine, rotary);
    assert_eq!(actual.dims(), expected.dims(), "{label}");
    assert_eq!(actual.dtype(), expected.dtype(), "{label}");
    assert_eq!(bit_pattern(actual), bit_pattern(expected), "{label}");
}

/// The fused Flex rotary kernel equals the broadcast formulation bit for bit.
///
/// Covers F32, F64, F16, and BF16; full and partial rotary widths (an
/// unrotated suffix); batch two with distinct position rows; and every
/// operand layout the kernel reads: contiguous `[b, h, s, w]`, the
/// heads-swapped view of `[b, s, h, w]` that projections produce, an offset
/// slice of a larger buffer, a channel-strided view (the per-element path),
/// and a flipped view (which falls back). Tables narrowed from longer tables
/// exercise strided table reads.
#[test]
fn fused_flex_rotation_matches_the_broadcast_formulation_bitwise() {
    let device = Default::default();
    let (batch, heads, sequence) = (2, 3, 5);
    let positions =
        Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 2, 3, 4], [7, 9, 11, 13, 40]], &device);
    for (width, rotary) in [(8, 8), (10, 6), (4, 2)] {
        let config = RopeConfig::new(width, rotary, 10_000.0, RopeScaling::None).expect("rope");
        let angles = rotary_angles(config, positions.clone(), 64).expect("angles");
        for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
            let label =
                |layout: &str| format!("{layout}, {dtype:?}, width {width}, rotary {rotary}");
            let contiguous = activations([batch, heads, sequence, width])
                .mul_scalar(3.0)
                .cast(dtype);
            assert_fused_matches_broadcast(contiguous, &angles, true, &label("contiguous"));
            let swapped = activations([batch, sequence, heads, width])
                .cast(dtype)
                .swap_dims(1, 2);
            assert_fused_matches_broadcast(swapped, &angles, true, &label("heads-swapped"));
            let offset = activations([batch, heads + 1, sequence, width])
                .cast(dtype)
                .narrow(1, 1, heads);
            assert_fused_matches_broadcast(offset, &angles, true, &label("offset slice"));
            let channel_strided = activations([batch, heads, width, sequence])
                .cast(dtype)
                .swap_dims(2, 3);
            assert_fused_matches_broadcast(
                channel_strided,
                &angles,
                true,
                &label("channel-strided"),
            );
            let flipped = activations([batch, heads, sequence, width])
                .cast(dtype)
                .flip([2]);
            assert_fused_matches_broadcast(flipped, &angles, false, &label("flipped"));
        }
        let long = rotary_angles(
            config,
            Tensor::<TestBackend, 2, Int>::from_ints(
                [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]],
                &device,
            ),
            64,
        )
        .expect("long angles");
        let narrowed = RotaryAngles {
            cosine: long.cosine.narrow(2, 2, sequence),
            sine: long.sine.narrow(2, 2, sequence),
            rotary_dimension: rotary,
        };
        assert_fused_matches_broadcast(
            activations([batch, heads, sequence, width]),
            &narrowed,
            true,
            &format!("narrowed tables, width {width}"),
        );
    }
}

/// Grouped query and key heads rotate through the fused kernel exactly as before.
#[test]
fn fused_flex_rotation_preserves_grouped_query_key_rotation() {
    let device = Default::default();
    let config = RopeConfig::new(8, 8, 10_000.0, RopeScaling::None).expect("rope");
    let angles = rotary_angles(
        config,
        Tensor::<TestBackend, 2, Int>::from_ints([[3, 4, 5], [1, 2, 3]], &device),
        16,
    )
    .expect("angles");
    let query = activations([2, 3, 4, 8]).swap_dims(1, 2);
    let key = activations([2, 3, 2, 8]).mul_scalar(0.5).swap_dims(1, 2);
    let (rotated_query, rotated_key) =
        apply_rotary(query.clone(), key.clone(), &angles).expect("grouped heads");
    let (cosine, sine) = super::tables_in(&angles, DType::F32);
    assert_eq!(
        bit_pattern(rotated_query),
        bit_pattern(super::rotate_prefix_broadcast(query, &cosine, &sine, 8))
    );
    assert_eq!(
        bit_pattern(rotated_key),
        bit_pattern(super::rotate_prefix_broadcast(key, &cosine, &sine, 8))
    );
}

/// Release-mode timing of the fused Flex rotary kernel against the broadcast formulation.
///
/// Operands are heads-swapped views of projected `[1, s, H, d]` activations:
/// a SmolLM-geometry query (`H = 9`, `d = 64`), a Qwen3-4B query (`H = 32`,
/// `d = 128`) at 512 prefill tokens, and the SmolLM decode token. Medians of
/// 15 interleaved samples. Run with `cargo test --release --lib
/// position::tensor::tests::rotary_prefill_timing -- --ignored --nocapture`.
#[test]
#[ignore = "release-mode timing benchmark"]
fn rotary_prefill_timing() {
    use std::time::Instant;
    let device = Default::default();
    for (heads, width, sequence) in [(9, 64, 512), (32, 128, 512), (9, 64, 1)] {
        let config = RopeConfig::new(width, width, 10_000.0, RopeScaling::None).expect("rope");
        let positions = Tensor::<TestBackend, 1, Int>::arange(0..sequence as i64, &device)
            .reshape([1, sequence]);
        let angles = rotary_angles(config, positions, sequence).expect("angles");
        let operand = activations([1, sequence, heads, width]).swap_dims(1, 2);
        let (mut broadcast, mut fused) = (Vec::new(), Vec::new());
        for _ in 0..15 {
            let started = Instant::now();
            let _ = super::rotate_prefix_broadcast(
                operand.clone(),
                &angles.cosine,
                &angles.sine,
                width,
            )
            .into_data();
            broadcast.push(started.elapsed().as_secs_f64());
            let started = Instant::now();
            let _ = super::rotate_prefix(operand.clone(), &angles.cosine, &angles.sine, width)
                .into_data();
            fused.push(started.elapsed().as_secs_f64());
        }
        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2] * 1e6
        };
        let (before, after) = (median(&mut broadcast), median(&mut fused));
        eprintln!(
            "rope [1, {heads}, {sequence}, {width}] heads-swapped F32: broadcast {before:.1} us, fused {after:.1} us ({:.1}x)",
            before / after
        );
    }
}
