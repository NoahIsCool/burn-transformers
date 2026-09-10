use burn::{
    backend::Flex,
    module::Param,
    nn::{Linear, LinearConfig},
    tensor::{DType, Tensor, TensorData},
};

use super::linear;
use crate::checkpoint::quantization::{
    QuantizationGroupSize, SymmetricQuantization, WeightQuantizationConfig, quantize_tensor_data,
};

/// Lazy row/column projections retain exact forward values and optimizer leaves.
#[cfg(feature = "autodiff")]
#[test]
fn autodiff_linear_initialization_preserves_layout_values_and_gradients() {
    use burn::{backend::Autodiff, module::Initializer, nn::LinearLayout};

    type AdBackend = Autodiff<Flex>;
    let device = Default::default();
    for layout in [LinearLayout::Row, LinearLayout::Col] {
        let config = LinearConfig::new(2, 3)
            .with_layout(layout)
            .with_initializer(Initializer::Constant { value: 0.25 });
        let projection = super::init_linear::<AdBackend>(&config, &device);
        assert!(!projection.weight.is_initialized());
        let weight_data = projection.weight.to_data();
        assert_eq!(weight_data, config.init::<Flex>(&device).weight.to_data());

        let input =
            Tensor::<AdBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device).require_grad();
        let gradients = linear(input.clone(), &projection).unwrap().sum().backward();
        assert_eq!(
            projection.weight.grad(&gradients).unwrap().to_data(),
            TensorData::from([[4.0_f32, 4.0, 4.0], [6.0, 6.0, 6.0]])
        );
        assert_eq!(
            projection
                .bias
                .as_ref()
                .unwrap()
                .grad(&gradients)
                .unwrap()
                .to_data(),
            TensorData::from([2.0_f32, 2.0, 2.0])
        );
        assert!(
            input.grad(&gradients).is_some(),
            "the activation graph stays connected"
        );
    }
}

/// Freezing before or after lazy materialization never re-enables gradients.
#[cfg(feature = "autodiff")]
#[test]
fn autodiff_linear_initializer_respects_frozen_parameters() {
    use burn::{backend::Autodiff, nn::LinearLayout};

    type AdBackend = Autodiff<Flex>;
    let device = Default::default();
    for materialize_before_freeze in [false, true] {
        let mut projection = super::init_linear::<AdBackend>(
            &LinearConfig::new(2, 3).with_layout(LinearLayout::Col),
            &device,
        );
        if materialize_before_freeze {
            let _ = projection.weight.val();
        }
        projection.weight = projection.weight.set_require_grad(false);
        projection.bias = projection.bias.map(|bias| bias.set_require_grad(false));
        assert_eq!(
            projection.weight.is_initialized(),
            materialize_before_freeze
        );
        let input = Tensor::<AdBackend, 2>::from_floats([[1.0, 2.0]], &device).require_grad();
        let gradients = linear(input.clone(), &projection).unwrap().sum().backward();
        assert!(!projection.weight.val().is_require_grad());
        assert!(projection.weight.grad(&gradients).is_none());
        assert!(projection.bias.as_ref().unwrap().grad(&gradients).is_none());
        assert!(input.grad(&gradients).is_some());
    }
}

/// Quantization-preserving dispatch is numerically identical for float weights.
#[test]
fn full_precision_projection_matches_burn_linear() {
    let device = Default::default();
    let projection = LinearConfig::new(4, 3).init::<Flex>(&device);
    let input = Tensor::<Flex, 3>::from_floats(
        [
            [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]],
            [[3.0, 1.0, -2.0, 0.5], [0.0, 0.0, 0.0, 1.0]],
        ],
        &device,
    );
    let expected = projection.forward(input.clone()).into_data();
    let actual = linear(input, &projection)
        .expect("validated projection")
        .into_data();
    actual.assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
}

/// A native QFloat weight remains packed across the public projection boundary.
#[test]
fn qfloat_weight_executes_without_changing_parameter_dtype() {
    let device = Default::default();
    let checkpoint_values = (0..256)
        .map(|index| (index as f32 - 127.5) / 32.0)
        .collect::<Vec<_>>();
    let quantized = quantize_tensor_data(
        &TensorData::new(checkpoint_values, [4, 64]),
        WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q4,
            group_size: QuantizationGroupSize::new(32).expect("supported group"),
        },
    )
    .expect("reference quantization");
    let packed = Tensor::<Flex, 2>::from_data(
        quantized
            .into_burn_qfloat_data()
            .expect("Burn-compatible matrix"),
        &device,
    )
    .transpose();
    assert!(matches!(packed.dtype(), DType::QFloat(_)));
    // Flex conservatively requantizes block tensors after a transpose;
    // CubeCL represents the same operation as a packed-stride view. Use
    // the runtime parameter itself as the backend-independent oracle for
    // the projection boundary. The adapter's byte-exact dequantization is
    // tested independently before any layout mapper is applied.
    let reference = packed.clone().dequantize();
    let projection = Linear {
        weight: Param::from_tensor(packed),
        bias: None,
    };
    let input = Tensor::<Flex, 2>::ones([3, 64], &device);
    let expected = input.clone().matmul(reference).into_data();
    let actual = linear(input, &projection)
        .expect("packed projection")
        .into_data();
    assert!(matches!(projection.weight.val().dtype(), DType::QFloat(_)));
    actual.assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
}

/// Invalid projection shapes return context instead of reaching a matmul panic.
#[test]
fn invalid_projection_shape_is_rejected() {
    let device = Default::default();
    let projection = LinearConfig::new(4, 3).init::<Flex>(&device);
    let input = Tensor::<Flex, 2>::zeros([2, 5], &device);
    assert!(linear(input, &projection).is_err());
}

/// The feature-unification guard distinguishes Q4 Metal from safe paths.
#[cfg(all(feature = "metal", feature = "rocm"))]
#[test]
fn packed_metal_autotune_guard_is_narrow() {
    let q4 = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 64], [1, 64]),
        WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q4,
            group_size: QuantizationGroupSize::new(32).expect("supported group"),
        },
    )
    .expect("Q4 reference quantization")
    .into_burn_qfloat_data()
    .expect("Burn-compatible Q4")
    .dtype;
    let q8 = quantize_tensor_data(
        &TensorData::new(vec![1.0_f32; 64], [1, 64]),
        WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q8,
            group_size: QuantizationGroupSize::new(32).expect("supported group"),
        },
    )
    .expect("Q8 reference quantization")
    .into_burn_qfloat_data()
    .expect("Burn-compatible Q8")
    .dtype;

    assert!(super::packed_metal_autotune_is_unavailable(
        q4,
        "fusion<cubecl<wgpu<msl>>>"
    ));
    assert!(super::packed_metal_autotune_is_unavailable(
        q8,
        "fusion<cubecl<wgpu<msl>>>"
    ));
    assert!(!super::packed_metal_autotune_is_unavailable(
        q4,
        "fusion<cubecl<cuda>>"
    ));
}

/// Metal accepts a Q4 RHS through CubeCL's quantized matmul binding.
///
/// Burn 0.21's CubeCL `q_matmul` passes packed values and their scale handle
/// as an `InputBinding::quantized`; this test protects the public call path
/// and verifies that no API-level conversion changes the parameter dtype.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires a Metal-capable macOS host"]
fn metal_qfloat_projection_executes_quantized_matmul_dispatch() {
    use burn::backend::Metal;

    let device = Default::default();
    let checkpoint_values = (0..(64 * 64))
        .map(|index| ((index % 127) as f32 - 63.0) / 16.0)
        .collect::<Vec<_>>();
    let quantized = quantize_tensor_data(
        &TensorData::new(checkpoint_values, [64, 64]),
        WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q4,
            group_size: QuantizationGroupSize::new(64).expect("supported group"),
        },
    )
    .expect("reference quantization");
    let dequantized = quantized
        .dequantize_f32()
        .expect("reference dequantization")
        .to_vec::<f32>()
        .expect("f32 reference data");
    let expected = dequantized
        .chunks_exact(64)
        .map(|row| row.iter().sum::<f32>())
        .collect::<Vec<_>>();
    let packed = Tensor::<Metal, 2>::from_data(
        quantized
            .into_burn_qfloat_data()
            .expect("Burn-compatible matrix"),
        &device,
    )
    .transpose();
    let projection = Linear {
        weight: Param::from_tensor(packed),
        bias: None,
    };
    assert!(matches!(projection.weight.val().dtype(), DType::QFloat(_)));

    let actual = linear(Tensor::<Metal, 2>::ones([1, 64], &device), &projection)
        .expect("Metal packed projection")
        .into_data()
        .to_vec::<f32>()
        .expect("Metal output should be readable as f32");

    assert!(matches!(projection.weight.val().dtype(), DType::QFloat(_)));
    TensorData::new(actual, [1, 64]).assert_approx_eq::<f32>(
        &TensorData::new(expected, [1, 64]),
        burn::tensor::Tolerance::relative(2e-2),
    );
}

/// Q4 Metal dispatch remains finite for every distinct Qwen3.5 projection shape.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires a Metal-capable macOS host"]
fn metal_qfloat_projection_covers_qwen35_shapes() {
    use burn::backend::Metal;

    let device = Default::default();
    for (output_width, input_width) in [
        (16, 1024),
        (512, 1024),
        (1024, 2048),
        (1024, 3584),
        (2048, 1024),
        (3584, 1024),
        (4096, 1024),
        (6144, 1024),
    ] {
        let elements = output_width * input_width;
        let checkpoint_values = (0..elements)
            .map(|index| ((index % 127) as f32 - 63.0) / 64.0)
            .collect::<Vec<_>>();
        let quantized = quantize_tensor_data(
            &TensorData::new(checkpoint_values, [output_width, input_width]),
            WeightQuantizationConfig {
                scheme: SymmetricQuantization::Q4,
                group_size: QuantizationGroupSize::new(64).expect("supported group"),
            },
        )
        .expect("reference quantization");
        let packed = Tensor::<Metal, 2>::from_data(
            quantized
                .into_burn_qfloat_data()
                .expect("Burn-compatible matrix"),
            &device,
        )
        .transpose();
        let projection = Linear {
            weight: Param::from_tensor(packed),
            bias: None,
        };
        let actual = linear(
            Tensor::<Metal, 2>::ones([1, input_width], &device).cast(DType::F16),
            &projection,
        )
        .expect("Metal packed projection")
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .expect("Metal output should be readable as f32");
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "Q4 projection [{input_width}, {output_width}] produced non-finite output"
        );
    }
}
