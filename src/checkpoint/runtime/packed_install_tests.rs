//! Exactness tests for packed parameter installation on the Flex backend.

use burn::{
    backend::Flex,
    nn::{LinearConfig, LinearLayout},
    tensor::{DType, TensorData},
};
use proptest::prelude::*;

use super::*;
use crate::checkpoint::quantization::{QuantizedWeight, SymmetricQuantization};

/// Builds a lazy `[3, 64]` parameter (the artifact's own shape) with `mapper` on load.
fn mapped_parameter(
    mapper: impl Fn(Tensor<Flex, 2>) -> Tensor<Flex, 2> + Send + Sync + 'static,
) -> Param<Tensor<Flex, 2>> {
    let device = Default::default();
    Param::uninitialized(
        ParamId::new(),
        move |device, _| Tensor::<Flex, 2>::zeros([3, 64], device),
        device,
        false,
        [3, 64].into(),
    )
    .load_mapper(mapper)
}

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

/// The mapper probe separates identity, axis permutations, and everything else.
#[test]
fn mapper_probe_classifies_identity_permutations_and_arithmetic() {
    let device = Default::default();
    let plain = Param::from_tensor(Tensor::<Flex, 2>::zeros([3, 64], &device));
    assert_eq!(load_mapper_effect(&plain), MapperEffect::Identity);
    let col = LinearConfig::new(64, 3)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init::<Flex>(&device);
    assert_eq!(
        load_mapper_effect(&col.weight),
        MapperEffect::AxisPermutation
    );
    let cases: [(&str, Param<Tensor<Flex, 2>>, MapperEffect); 5] = [
        (
            "noop closure",
            mapped_parameter(|value| value),
            MapperEffect::Identity,
        ),
        (
            "detached transpose",
            mapped_parameter(|value| value.transpose().detach()),
            MapperEffect::AxisPermutation,
        ),
        (
            "scaling",
            mapped_parameter(|value| value / 2.0),
            MapperEffect::Other,
        ),
        (
            "abs",
            mapped_parameter(|value| value.abs()),
            MapperEffect::Other,
        ),
        (
            "shape-swapping reshape",
            mapped_parameter(|value| {
                let [rows, columns] = value.dims();
                value.reshape([columns, rows])
            }),
            MapperEffect::Other,
        ),
    ];
    for (label, parameter, expected) in cases {
        assert_eq!(load_mapper_effect(&parameter), expected, "{label}");
        assert!(
            !parameter.is_initialized(),
            "{label}: probe initialized the parameter"
        );
    }
}

/// A rank-3 permutation is recovered from the probe, not only a 2-D transpose.
#[test]
fn mapper_probe_recognizes_higher_rank_permutations() {
    let device = Default::default();
    let parameter = Param::from_tensor(Tensor::<Flex, 3>::zeros([4, 4, 4], &device))
        .load_mapper(|value| value.permute([2, 0, 1]));
    assert_eq!(
        load_mapper_effect(&parameter),
        MapperEffect::AxisPermutation
    );
}

/// Axis recovery follows Burn's `permute` convention and rejects non-permutations.
#[test]
fn axis_permutation_recovers_axes_and_rejects_rearrangements_it_cannot_explain() {
    let input = probe_dims::<3>();
    assert_eq!(input, [2, 3, 5]);
    assert_eq!(axis_permutation(&input, &[2, 3, 5]), Some([0, 1, 2]));
    assert_eq!(axis_permutation(&input, &[5, 2, 3]), Some([2, 0, 1]));
    assert_eq!(axis_permutation(&input, &[2, 3]), None, "rank change");
    assert_eq!(
        axis_permutation(&input, &[2, 3, 7]),
        None,
        "unknown dimension"
    );
    assert_eq!(
        axis_permutation(&input, &[2, 2, 3]),
        None,
        "repeated dimension"
    );
    assert_eq!(axis_permutation(&input, &[30]), None, "flattening reshape");
    assert_eq!(axis_permutation::<0>(&[], &[]), Some([]));
}

/// Probe dimensions stay pairwise distinct past the tabulated primes.
#[test]
fn probe_dimensions_are_pairwise_distinct_for_every_rank() {
    let dims = probe_dims::<12>();
    for (first, &left) in dims.iter().enumerate() {
        for &right in &dims[first + 1..] {
            assert_ne!(left, right, "{dims:?}");
        }
    }
}

proptest! {
    /// Permuting the probe dimensions by any permutation is recovered exactly.
    #[test]
    fn axis_permutation_inverts_every_rank_four_permutation(
        axes in Just([0_usize, 1, 2, 3]).prop_shuffle()
    ) {
        let input = probe_dims::<4>();
        let output = axes.map(|axis| input[axis]);
        prop_assert_eq!(axis_permutation(&input, &output), Some(axes));
    }
}

/// A value-changing mapper is never reported as exact packed storage.
#[test]
fn scaling_mapper_is_inexact_under_every_policy() {
    let weight = straddling_weight(SymmetricQuantization::Q8);
    let halved = weight
        .dequantize_f32()
        .unwrap()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(|value| value / 2.0)
        .collect::<Vec<_>>();

    let mut strict = mapped_parameter(|value| value / 2.0);
    assert_eq!(
        install_packed_parameter(
            &mut strict,
            weight.clone().into_burn_qfloat_data().unwrap(),
            PackedLayoutPolicy::RequireExact,
        ),
        Err(ParameterInstallError::InexactPackedLayout)
    );
    assert!(!strict.is_initialized());

    let mut exact = mapped_parameter(|value| value / 2.0);
    let outcome = install_packed_parameter(
        &mut exact,
        weight.clone().into_burn_qfloat_data().unwrap(),
        PackedLayoutPolicy::DequantizeWhenInexact,
    )
    .unwrap();
    assert_eq!(outcome, PackedInstall::Dequantized);
    // Halving is exact in binary floating point, so the result is bit-exact.
    assert_eq!(exact.val().into_data().to_vec::<f32>().unwrap(), halved);

    let mut legacy = mapped_parameter(|value| value / 2.0);
    let outcome = install_packed_parameter(
        &mut legacy,
        weight.into_burn_qfloat_data().unwrap(),
        PackedLayoutPolicy::BackendDefault,
    )
    .unwrap();
    assert_eq!(outcome, PackedInstall::Transformed);
    assert!(legacy.is_initialized());
}

/// Ordinary preconditions are reported before any exactness decision.
#[test]
fn packed_install_reports_shape_errors_before_exactness() {
    let device = Default::default();
    let mut linear = LinearConfig::new(64, 4)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init::<Flex>(&device);
    let weight = straddling_weight(SymmetricQuantization::Q4);
    for policy in [
        PackedLayoutPolicy::BackendDefault,
        PackedLayoutPolicy::DequantizeWhenInexact,
        PackedLayoutPolicy::RequireExact,
    ] {
        assert_eq!(
            install_packed_parameter(
                &mut linear.weight,
                weight.clone().into_burn_qfloat_data().unwrap(),
                policy,
            ),
            Err(ParameterInstallError::Shape {
                expected: vec![4, 64],
                actual: vec![3, 64],
            }),
            "{policy:?}"
        );
        assert!(!linear.weight.is_initialized());
    }
}

/// Autodiff backends get the contextual error, never a Burn panic.
#[cfg(feature = "autodiff")]
#[test]
fn autodiff_packed_install_returns_quantized_autodiff_for_every_policy() {
    /// Flex with autodiff, which cannot construct QFloat tensors in Burn 0.21.
    type AdBackend = burn::backend::Autodiff<Flex>;
    let device = Default::default();
    assert!(!packed_layout_is_exact::<AdBackend>(&device));
    let weight = straddling_weight(SymmetricQuantization::Q8);
    for layout in [LinearLayout::Row, LinearLayout::Col] {
        let (input, output) = match layout {
            LinearLayout::Row => (3, 64),
            LinearLayout::Col => (64, 3),
        };
        for policy in [
            PackedLayoutPolicy::BackendDefault,
            PackedLayoutPolicy::DequantizeWhenInexact,
            PackedLayoutPolicy::RequireExact,
        ] {
            let mut linear = LinearConfig::new(input, output)
                .with_bias(false)
                .with_layout(layout)
                .init::<AdBackend>(&device);
            let before = linear.weight.id;
            assert_eq!(
                install_packed_parameter(
                    &mut linear.weight,
                    weight.clone().into_burn_qfloat_data().unwrap(),
                    policy,
                ),
                Err(ParameterInstallError::QuantizedAutodiff),
                "{layout:?} {policy:?}"
            );
            assert_eq!(linear.weight.id, before);
            assert!(!linear.weight.is_initialized());
        }
    }
}
