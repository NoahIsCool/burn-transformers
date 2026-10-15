//! Exactness tests for packed parameter installation on the Flex backend.

use burn::{
    backend::Flex,
    nn::{LinearConfig, LinearLayout},
    tensor::{DType, TensorData},
};

use super::*;
use crate::checkpoint::quantization::{QuantizedWeight, SymmetricQuantization};

/// Quantizes a `[3, 64]` matrix whose groups straddle the transposed layout's rows.
fn straddling_weight(scheme: SymmetricQuantization) -> QuantizedWeight {
    let values = (0..192)
        .map(|index| ((index as f32) * 0.37).sin() * (1.0 + (index / 64) as f32 * 40.0))
        .collect::<Vec<_>>();
    quantize_tensor_data(
        &TensorData::new(values, [3, 64]),
        WeightQuantizationConfig {
            scheme,
            group_size: QuantizationGroupSize::new(32).unwrap(),
        },
    )
    .unwrap()
}

/// Returns the artifact's exact `W_hat` transposed into the column-layout runtime shape.
fn artifact_transposed(weight: &QuantizedWeight) -> Vec<f32> {
    let values = weight.dequantize_f32().unwrap().to_vec::<f32>().unwrap();
    let (rows, columns) = (3, 64);
    (0..columns)
        .flat_map(|column| (0..rows).map(move |row| (row, column)))
        .map(|(row, column)| values[row * columns + column])
        .collect()
}

/// Flex re-derives block scales after a quantized transpose.
#[test]
fn flex_quantized_transpose_is_reported_inexact() {
    assert!(!packed_layout_is_exact::<Flex>(&Default::default()));
    // The cached answer is stable.
    assert!(!packed_layout_is_exact::<Flex>(&Default::default()));
}

/// Dequantizing first installs exactly the artifact's values in a column-layout weight.
#[test]
fn dequantize_policy_installs_the_exact_artifact_values() {
    for scheme in [SymmetricQuantization::Q8, SymmetricQuantization::Q4] {
        let device = Default::default();
        let mut linear = LinearConfig::new(64, 3)
            .with_bias(false)
            .with_layout(LinearLayout::Col)
            .init::<Flex>(&device);
        let weight = straddling_weight(scheme);
        let expected = artifact_transposed(&weight);
        let outcome = install_packed_parameter(
            &mut linear.weight,
            weight.into_burn_qfloat_data().unwrap(),
            PackedLayoutPolicy::DequantizeWhenInexact,
        )
        .unwrap();
        assert_eq!(outcome, PackedInstall::Dequantized);
        assert_eq!(linear.weight.val().dims(), [64, 3]);
        let installed = linear.weight.val().into_data().to_vec::<f32>().unwrap();
        assert_eq!(installed, expected, "{scheme:?}");
    }
}

/// The legacy path requantizes on Flex; the exact policy refuses without mutation.
#[test]
fn backend_default_requantizes_and_require_exact_refuses() {
    let device = Default::default();
    let weight = straddling_weight(SymmetricQuantization::Q4);
    let expected = artifact_transposed(&weight);

    let mut legacy = LinearConfig::new(64, 3)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init::<Flex>(&device);
    let outcome = install_packed_parameter(
        &mut legacy.weight,
        weight.clone().into_burn_qfloat_data().unwrap(),
        PackedLayoutPolicy::BackendDefault,
    )
    .unwrap();
    assert_eq!(outcome, PackedInstall::Requantized);
    assert!(matches!(legacy.weight.val().dtype(), DType::QFloat(_)));
    let requantized = legacy
        .weight
        .val()
        .dequantize()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_ne!(
        requantized, expected,
        "Flex requantization changed the artifact"
    );

    let mut strict = LinearConfig::new(64, 3)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init::<Flex>(&device);
    assert_eq!(
        install_packed_parameter(
            &mut strict.weight,
            weight.into_burn_qfloat_data().unwrap(),
            PackedLayoutPolicy::RequireExact,
        ),
        Err(ParameterInstallError::InexactPackedLayout)
    );
    assert!(!strict.weight.is_initialized());
}

/// Without a layout-changing mapper, packed storage is exact and stays packed.
#[test]
fn row_layout_keeps_exact_packed_storage() {
    let device = Default::default();
    let mut linear = LinearConfig::new(3, 64)
        .with_bias(false)
        .with_layout(LinearLayout::Row)
        .init::<Flex>(&device);
    // Row layout stores `[input, output]`; reuse the `[3, 64]` artifact directly.
    let weight = straddling_weight(SymmetricQuantization::Q8);
    let expected = weight.dequantize_f32().unwrap().to_vec::<f32>().unwrap();
    let outcome = install_packed_parameter(
        &mut linear.weight,
        weight.into_burn_qfloat_data().unwrap(),
        PackedLayoutPolicy::RequireExact,
    )
    .unwrap();
    assert_eq!(outcome, PackedInstall::Packed);
    assert!(matches!(linear.weight.val().dtype(), DType::QFloat(_)));
    let installed = linear
        .weight
        .val()
        .dequantize()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(installed, expected);
}

/// Float data bypasses every packed rule.
#[test]
fn float_data_takes_the_ordinary_install_path() {
    let device = Default::default();
    let mut linear = LinearConfig::new(2, 3)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init::<Flex>(&device);
    let outcome = install_packed_parameter(
        &mut linear.weight,
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]),
        PackedLayoutPolicy::RequireExact,
    )
    .unwrap();
    assert_eq!(outcome, PackedInstall::Float);
    assert_eq!(linear.weight.val().dims(), [2, 3]);
}

/// CubeCL keeps transposed packed weights as exact strided views (integrator-run).
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires a Metal-capable macOS host"]
fn metal_quantized_transpose_is_exact_and_stays_packed() {
    use burn::backend::Metal;

    let device = Default::default();
    assert!(packed_layout_is_exact::<Metal>(&device));
    let mut linear = LinearConfig::new(64, 3)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init::<Metal>(&device);
    let weight = straddling_weight(SymmetricQuantization::Q4);
    let outcome = install_packed_parameter(
        &mut linear.weight,
        weight.into_burn_qfloat_data().unwrap(),
        PackedLayoutPolicy::RequireExact,
    )
    .unwrap();
    assert_eq!(outcome, PackedInstall::Packed);
    assert!(matches!(linear.weight.val().dtype(), DType::QFloat(_)));
}
