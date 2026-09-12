//! Rotary tensor parity, partial-channel preservation, and scaling-boundary tests.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData, Tolerance},
};

use super::{apply_rotary, inverse_frequencies, mrope_angles, rotary_angles};
use crate::position::{DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling, YarnConfig};

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
