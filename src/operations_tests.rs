//! Linear-operation tests for dense and packed execution, record layouts, and gradient ownership.

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

/// Open user modules receive contextual errors rather than a backend matmul panic.
#[test]
fn projection_rejects_mixed_float_dtypes_and_accepts_explicit_conversion() {
    let device = Default::default();
    let input = Tensor::<Flex, 2>::ones([1, 2], &device).cast(DType::F64);
    let mut projection = LinearConfig::new(2, 3).init::<Flex>(&device);
    assert!(
        linear(input.clone(), &projection)
            .unwrap_err()
            .to_string()
            .contains("linear.weight dtype")
    );
    projection.weight = Param::from_tensor(projection.weight.val().cast(DType::F64));
    assert!(
        linear(input.clone(), &projection)
            .unwrap_err()
            .to_string()
            .contains("linear.bias dtype")
    );
    projection.bias = projection
        .bias
        .map(|bias| Param::from_tensor(bias.val().cast(DType::F64)));
    let output = linear(input, &projection).unwrap();
    assert_eq!(output.dims(), [1, 3]);
    assert_eq!(output.dtype(), DType::F64);
}

/// Lazy row/column projections retain exact forward values and optimizer leaves.
#[cfg(feature = "autodiff")]
#[test]
fn autodiff_linear_initialization_preserves_layout_values_and_gradients() {
    use burn::{backend::Autodiff, module::Initializer, nn::LinearLayout};

    /// Differentiable CPU backend for lazy layout initialization and gradient comparisons.
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

    /// Differentiable CPU backend used to test freezing before and after materialization.
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

/// Loading a checkpoint establishes fresh parameter leaves after layout mapping.
#[cfg(feature = "autodiff")]
#[test]
fn autodiff_linear_record_reload_preserves_values_gradients_and_freezing() {
    use burn::{backend::Autodiff, module::Module, nn::LinearLayout};

    /// Differentiable CPU backend for model-record layout and trainability checks.
    type AdBackend = Autodiff<Flex>;
    let device = Default::default();
    let config = LinearConfig::new(2, 3)
        .with_bias(false)
        .with_layout(LinearLayout::Col);
    let original = super::init_linear::<AdBackend>(&config, &device);
    let expected = linear(
        Tensor::<AdBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device),
        &original,
    )
    .expect("original projection")
    .to_data();
    for frozen in [false, true] {
        let restored = super::init_linear::<AdBackend>(&config, &device);
        let restored = if frozen { restored.no_grad() } else { restored };
        let restored = restored.load_record(original.clone().into_record());
        let input =
            Tensor::<AdBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device).require_grad();
        let output = linear(input.clone(), &restored).expect("restored projection");
        assert_eq!(output.to_data(), expected);
        let gradients = output.sum().backward();
        if frozen {
            assert!(!restored.weight.val().is_require_grad());
            assert!(restored.weight.grad(&gradients).is_none());
        } else {
            assert_eq!(
                restored
                    .weight
                    .grad(&gradients)
                    .expect("restored optimizer leaf")
                    .to_data(),
                TensorData::from([[4.0_f32, 4.0, 4.0], [6.0, 6.0, 6.0]])
            );
        }
        assert!(
            input.grad(&gradients).is_some(),
            "forward inputs stay connected"
        );
        assert!(
            original.weight.grad(&gradients).is_none(),
            "loading severs the source parameter graph"
        );
    }
}

/// The autodiff-only load fix leaves ordinary row/column records unchanged.
#[test]
fn inference_linear_records_preserve_layout_and_values() {
    use burn::{module::Module, nn::LinearLayout};

    let device = Default::default();
    for layout in [LinearLayout::Row, LinearLayout::Col] {
        let config = LinearConfig::new(2, 3).with_layout(layout);
        let original = super::init_linear::<Flex>(&config, &device);
        let input = Tensor::<Flex, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let expected = linear(input.clone(), &original)
            .expect("original")
            .to_data();
        let record = original.into_record();
        assert_eq!(
            record.weight.lazy_shape().to_vec(),
            match layout {
                LinearLayout::Row => vec![2, 3],
                LinearLayout::Col => vec![3, 2],
            }
        );
        let restored = super::init_linear::<Flex>(&config, &device).load_record(record);
        assert_eq!(
            linear(input, &restored).expect("restored").to_data(),
            expected
        );
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

/// Empty activation axes report their shape before matmul dispatch.
#[test]
fn projection_contract_empty_inputs_are_contextual_errors() {
    let device = Default::default();
    let projection = Linear {
        weight: Param::from_tensor(Tensor::<Flex, 2>::from_floats([[2.0]], &device)),
        bias: None,
    };
    for shape in [[0, 1], [2, 0]] {
        let input =
            Tensor::<Flex, 2>::from_data(TensorData::new(Vec::<f32>::new(), shape), &device);
        assert!(
            matches!(linear(input, &projection), Err(crate::Error::Shape {
            tensor, expected, actual
        }) if tensor == "linear.input" && expected == "non-zero dimensions ending in the projection input width" && actual == format!("{shape:?}"))
        );
    }
    let output = linear(Tensor::<Flex, 1>::from_floats([3.0], &device), &projection).unwrap();
    assert_eq!(output.dims(), [1]);
    assert_eq!(output.to_data().to_vec::<f32>().unwrap(), [6.0]);
}

/// Packed activations are rejected without materializing or altering parameters.
#[test]
fn projection_contract_packed_activations_require_explicit_dequantization() {
    let device = Default::default();
    for scheme in [SymmetricQuantization::Q4, SymmetricQuantization::Q8] {
        let input = Tensor::<Flex, 2>::from_data(
            quantize_tensor_data(
                &TensorData::new(vec![1.0_f32; 32], [1, 32]),
                WeightQuantizationConfig {
                    scheme,
                    group_size: QuantizationGroupSize::new(32).unwrap(),
                },
            )
            .unwrap()
            .into_burn_qfloat_data()
            .unwrap(),
            &device,
        );
        let projection = Linear {
            weight: Param::from_tensor(Tensor::<Flex, 2>::ones([32, 1], &device)),
            bias: None,
        };
        let dtype = input.dtype();
        assert!(matches!(dtype, DType::QFloat(_)));
        assert!(
            matches!(linear(input.clone(), &projection), Err(crate::Error::BackendCapability(reason))
            if reason.contains("not quantized activations"))
        );
        assert_eq!(input.dtype(), dtype);
        assert_eq!(projection.weight.val().dtype(), DType::F32);
        assert_eq!(
            projection.weight.to_data().to_vec::<f32>().unwrap(),
            vec![1.0; 32]
        );
        let output = linear(input.dequantize(), &projection).unwrap();
        assert_eq!(output.dims(), [1, 1]);
        let values = output.to_data().to_vec::<f32>().unwrap();
        assert!(values[0].is_finite());
        assert!((values[0] - 32.0).abs() < 1e-4);
    }
}

/// A malformed bias returns its own context and a corrected bias works exactly.
#[test]
fn projection_contract_bias_width_rejection_preserves_weight_values() {
    let device = Default::default();
    let mut projection = Linear {
        weight: Param::from_tensor(Tensor::<Flex, 2>::from_floats(
            [[1.0, 2.0], [3.0, 4.0]],
            &device,
        )),
        bias: Some(Param::from_tensor(Tensor::<Flex, 1>::from_floats(
            [9.0],
            &device,
        ))),
    };
    let input = Tensor::<Flex, 3>::from_floats([[[2.0, 3.0], [4.0, 5.0]]], &device);
    assert!(
        matches!(linear(input.clone(), &projection), Err(crate::Error::Shape {
        tensor, expected, actual
    }) if tensor == "linear.bias" && expected == "[2]" && actual == "[1]")
    );
    assert_eq!(
        projection.weight.to_data(),
        TensorData::from([[1.0_f32, 2.0], [3.0, 4.0]])
    );
    assert_eq!(
        projection.bias.as_ref().unwrap().to_data(),
        TensorData::from([9.0_f32])
    );
    projection.bias = Some(Param::from_tensor(Tensor::<Flex, 1>::from_floats(
        [1.0, -1.0],
        &device,
    )));
    let output = linear(input, &projection).unwrap();
    assert_eq!(output.dims(), [1, 2, 2]);
    assert_eq!(
        output.to_data(),
        TensorData::from([[[12.0_f32, 15.0], [20.0, 27.0]]])
    );
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
