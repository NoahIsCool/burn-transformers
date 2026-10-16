//! Linear-operation tests for dense and packed execution, record layouts, and gradient ownership,
//! plus the tied-readout product [`super::matmul_transposed`] and the storage-order helpers
//! behind contiguous random initialization ([`super::contiguous_where_preferred`]).

use burn::{
    backend::Flex,
    module::Param,
    nn::{Linear, LinearConfig},
    tensor::{DType, Tensor, TensorData},
};

use super::{linear, matmul_transposed};
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

/// Deterministic `[rows, columns]` values whose products stay exactly representable.
fn small_integers(rows: usize, columns: usize, salt: usize) -> TensorData {
    TensorData::new(
        (0..rows * columns)
            .map(|index| ((index * 7 + salt) % 11) as f32 - 5.0)
            .collect::<Vec<_>>(),
        [rows, columns],
    )
}

/// The tied readout equals Burn's `module::linear` against the transposed matrix at every rank.
#[test]
fn matmul_transposed_matches_burn_linear_for_every_rank() {
    let device = Default::default();
    let matrix = Tensor::<Flex, 2>::from_data(small_integers(5, 4, 3), &device);
    let reference = |input: Tensor<Flex, 3>| {
        burn::tensor::module::linear(input, matrix.clone().transpose(), None).into_data()
    };

    let batched = Tensor::<Flex, 3>::from_data(
        TensorData::new(small_integers(6, 4, 1).to_vec::<f32>().unwrap(), [2, 3, 4]),
        &device,
    );
    let output = matmul_transposed(batched.clone(), matrix.clone()).unwrap();
    assert_eq!(output.dims(), [2, 3, 5]);
    assert_eq!(output.into_data(), reference(batched.clone()));

    let rows = matmul_transposed(batched.clone().reshape([6, 4]), matrix.clone()).unwrap();
    assert_eq!(rows.reshape([2, 3, 5]).into_data(), reference(batched));

    let vector = Tensor::<Flex, 1>::from_floats([1.0, -2.0, 3.0, 0.5], &device);
    let expected = matrix
        .clone()
        .matmul(vector.clone().reshape([4, 1]))
        .reshape([5])
        .into_data();
    assert_eq!(
        matmul_transposed(vector, matrix).unwrap().into_data(),
        expected
    );
}

/// Shape, dtype and packing mismatches are contextual errors before any matmul.
#[test]
fn matmul_transposed_rejects_invalid_operands() {
    let device = Default::default();
    let matrix = Tensor::<Flex, 2>::from_data(small_integers(5, 4, 0), &device);
    let input = Tensor::<Flex, 2>::from_data(small_integers(2, 4, 0), &device);

    let wrong_width = Tensor::<Flex, 2>::zeros([2, 3], &device);
    assert!(matches!(
        matmul_transposed(wrong_width, matrix.clone()),
        Err(crate::Error::Shape { tensor, .. }) if tensor == "matmul_transposed.matrix"
    ));
    let empty_rows =
        Tensor::<Flex, 2>::from_data(TensorData::new(Vec::<f32>::new(), [0, 4]), &device);
    assert!(matches!(
        matmul_transposed(input.clone(), empty_rows),
        Err(crate::Error::Shape { tensor, .. }) if tensor == "matmul_transposed.matrix"
    ));
    let empty_input =
        Tensor::<Flex, 2>::from_data(TensorData::new(Vec::<f32>::new(), [0, 4]), &device);
    assert!(matches!(
        matmul_transposed(empty_input, matrix.clone()),
        Err(crate::Error::Shape { tensor, .. }) if tensor == "matmul_transposed.input"
    ));
    assert!(matches!(
        matmul_transposed(input.clone().cast(DType::F64), matrix.clone()),
        Err(crate::Error::BackendCapability(message)) if message.contains("must match")
    ));
    let packed = Tensor::<Flex, 2>::from_data(
        quantize_tensor_data(
            &TensorData::new(vec![0.5_f32; 64], [2, 32]),
            WeightQuantizationConfig {
                scheme: SymmetricQuantization::Q8,
                group_size: QuantizationGroupSize::new(32).unwrap(),
            },
        )
        .unwrap()
        .into_burn_qfloat_data()
        .unwrap(),
        &device,
    );
    assert!(matches!(
        matmul_transposed(Tensor::<Flex, 2>::ones([1, 32], &device), packed),
        Err(crate::Error::BackendCapability(message)) if message.contains("dequantize")
    ));
}

/// Gradients of the tied readout reach both the activations and the matrix.
#[cfg(feature = "autodiff")]
#[test]
fn matmul_transposed_differentiates_both_operands() {
    use burn::backend::Autodiff;

    /// Differentiable CPU backend for the tied-readout gradient check.
    type AdBackend = Autodiff<Flex>;
    let device = Default::default();
    let matrix = Tensor::<AdBackend, 2>::from_data(small_integers(3, 2, 0), &device).require_grad();
    let input =
        Tensor::<AdBackend, 3>::from_floats([[[1.0, 2.0], [3.0, 4.0]]], &device).require_grad();
    let gradients = matmul_transposed(input.clone(), matrix.clone())
        .unwrap()
        .sum()
        .backward();
    // d(sum)/dM[o, i] = sum over rows of x[r, i]; d(sum)/dx[r, i] = sum over o of M[o, i].
    assert_eq!(
        matrix.grad(&gradients).unwrap().to_data(),
        TensorData::from([[4.0_f32, 6.0], [4.0, 6.0], [4.0, 6.0]])
    );
    let column_sums = small_integers(3, 2, 0)
        .to_vec::<f32>()
        .unwrap()
        .chunks(2)
        .fold([0.0_f32; 2], |sums, row| {
            [sums[0] + row[0], sums[1] + row[1]]
        });
    assert_eq!(
        input.grad(&gradients).unwrap().to_data(),
        TensorData::from([[column_sums, column_sums]])
    );
}

// ---------------------------------------------------------------------------
// Storage order of initialized and cached tensors
// ---------------------------------------------------------------------------

/// Returns the Flex primitive of `tensor` to inspect its physical layout.
fn flex_primitive<const D: usize>(tensor: Tensor<Flex, D>) -> burn::backend::flex::FlexTensor {
    tensor.into_primitive().tensor()
}

/// Irregular F32 values `sin(0.37 i)` laid out row-major in `shape`.
fn irregular<const D: usize>(shape: [usize; D]) -> Tensor<Flex, D> {
    let elements = shape.iter().product::<usize>();
    Tensor::<Flex, 1, burn::tensor::Int>::arange(0..elements as i64, &Default::default())
        .float()
        .mul_scalar(0.37)
        .sin()
        .reshape(shape)
}

/// Only the inference Flex backend asks for contiguous copies; autodiff never does.
#[test]
fn only_inference_flex_prefers_contiguous_storage() {
    let device = Default::default();
    assert!(super::prefers_contiguous_storage::<Flex>(&device));
    #[cfg(feature = "autodiff")]
    assert!(!super::prefers_contiguous_storage::<
        burn::backend::Autodiff<Flex>,
    >(&device));
}

/// Materialization is an exact permutation: shape, dtype and every bit survive.
#[test]
fn contiguous_materialization_preserves_every_value_and_dtype() {
    for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
        let view = irregular([2, 3, 4, 5]).cast(dtype).swap_dims(1, 2);
        assert!(!flex_primitive(view.clone()).is_contiguous(), "{dtype:?}");
        let stored = super::contiguous_where_preferred(view.clone());
        assert!(flex_primitive(stored.clone()).is_contiguous(), "{dtype:?}");
        assert_eq!(stored.dims(), [2, 4, 3, 5]);
        assert_eq!(stored.dtype(), dtype);
        assert_eq!(stored.into_data(), view.into_data(), "{dtype:?}");
    }
}

/// Already contiguous tensors and transposed matrices take the same exact path.
#[test]
fn contiguous_materialization_handles_contiguous_and_matrix_views() {
    let contiguous = irregular([3, 4]);
    let stored = super::contiguous_where_preferred(contiguous.clone());
    assert!(flex_primitive(stored.clone()).is_contiguous());
    assert_eq!(stored.into_data(), contiguous.clone().into_data());

    let transposed = contiguous.transpose();
    let stored = super::contiguous_where_preferred(transposed.clone());
    assert!(flex_primitive(stored.clone()).is_contiguous());
    assert_eq!(stored.dims(), [4, 3]);
    assert_eq!(stored.into_data(), transposed.into_data());
}

/// Packed tensors are returned untouched: their block scales are layout-bound.
#[test]
fn contiguous_materialization_leaves_packed_tensors_unchanged() {
    let device = Default::default();
    let data = quantize_tensor_data(
        &small_integers(2, 32, 1),
        WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q8,
            group_size: QuantizationGroupSize::new(32).unwrap(),
        },
    )
    .unwrap();
    let packed = Tensor::<Flex, 2>::from_data(data.into_burn_qfloat_data().unwrap(), &device);
    let dtype = packed.dtype();
    assert!(matches!(dtype, DType::QFloat(_)));
    let returned = super::contiguous_where_preferred(packed.clone());
    assert_eq!(returned.dtype(), dtype);
    assert_eq!(returned.into_data(), packed.into_data());
}

/// Autodiff tensors keep their view and therefore their gradient path.
#[cfg(feature = "autodiff")]
#[test]
fn contiguous_materialization_keeps_autodiff_graphs_connected() {
    /// Differentiable CPU backend whose tensors must never be copied off the graph.
    type AdBackend = burn::backend::Autodiff<Flex>;
    let device = Default::default();
    let input = Tensor::<AdBackend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)
        .require_grad();
    let weights =
        Tensor::<AdBackend, 2>::from_floats([[1.0, 10.0], [100.0, 1000.0], [0.5, 2.0]], &device);
    let gradients = (super::contiguous_where_preferred(input.clone().transpose()) * weights)
        .sum()
        .backward();
    assert_eq!(
        input.grad(&gradients).expect("connected").to_data(),
        TensorData::from([[1.0_f32, 100.0, 0.5], [10.0, 1000.0, 2.0]])
    );
}

/// A lazy initializer chained with the materialization keeps its exact values.
///
/// This is the composition `init_linear` installs on inference Flex: Burn's
/// column-layout initializer yields the transposed view of an `[out, in]`
/// draw, and the chained mapper stores it contiguously at the first read.
#[test]
fn chained_materialization_preserves_lazy_initializer_values() {
    let device = Default::default();
    let drawn = irregular([3, 5]);
    let source = drawn.clone();
    let lazy = Param::uninitialized(
        burn::module::ParamId::new(),
        move |_, _| source.clone().transpose(),
        device,
        false,
        [5, 3].into(),
    )
    .init_mapper(super::contiguous_where_preferred::<Flex, 2>);
    assert!(!lazy.is_initialized());
    let value = lazy.val();
    assert!(flex_primitive(value.clone()).is_contiguous());
    assert_eq!(value.into_data(), drawn.transpose().into_data());
}

/// Random column-layout weights are contiguous on inference Flex and keep Burn's distribution.
///
/// Burn draws a column-layout weight as `[out, in]` with Kaiming-uniform bound
/// `1/sqrt(out)` (Burn 0.21 passes `d_output` as the fan-in of that layout) and
/// exposes its transposed view. The installed weight must be contiguous,
/// remain lazy until read, stay inside that bound, and round-trip a serialized
/// record through the unchanged save/load mappers with exactly the same values.
/// Row layouts are already contiguous.
///
/// The contiguous layout is a property of random initialization only. A Burn
/// recorder serializes the `[out, in]` checkpoint orientation in row-major
/// order, and Burn's column-layout load mapper exposes it as a strided
/// `[in, out]` view; `load_record` never runs the initializer, so it never runs
/// the contiguous copy either. The final layout assertion pins that documented
/// limitation, so a Burn change that alters it is noticed. Products against
/// the two layouts may differ by F32 reassociation inside Flex's GEMM only.
#[test]
fn inference_flex_random_column_projections_are_contiguous() {
    use burn::{
        module::Module,
        nn::LinearLayout,
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        tensor::Tolerance,
    };

    let device = Default::default();
    let (input, output) = (24, 40);
    let config = LinearConfig::new(input, output)
        .with_bias(false)
        .with_layout(LinearLayout::Col);
    let projection = super::init_linear::<Flex>(&config, &device);
    assert!(!projection.weight.is_initialized());
    let weight = projection.weight.val();
    assert_eq!(weight.dims(), [input, output]);
    assert!(flex_primitive(weight.clone()).is_contiguous());
    let bound = 1.0 / (output as f32).sqrt();
    let values = weight.to_data().to_vec::<f32>().unwrap();
    assert!(values.iter().all(|value| value.abs() <= bound));
    assert!(values.iter().any(|&value| value != values[0]));

    let probe = irregular([2, input]);
    let expected = linear(probe.clone(), &projection).unwrap().into_data();
    let record = projection.into_record();
    assert_eq!(record.weight.val().dims(), [output, input]);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = recorder.record(record, ()).unwrap();
    let restored = super::init_linear::<Flex>(&config, &device)
        .load_record(recorder.load(bytes, &device).unwrap());
    let restored_weight = restored.weight.val();
    assert_eq!(restored_weight.to_data(), weight.into_data());
    linear(probe, &restored)
        .unwrap()
        .into_data()
        .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-6));
    assert!(
        !flex_primitive(restored_weight).is_contiguous(),
        "Burn-record loads keep the load mapper's strided view"
    );

    let row = super::init_linear::<Flex>(&config.clone().with_layout(LinearLayout::Row), &device);
    assert!(flex_primitive(row.weight.val()).is_contiguous());
}

/// Rank-zero activations have no projection axis and are rejected before any reshape.
///
/// Burn's creation APIs refuse rank zero, but a reshape to `[]` produces one;
/// both products must report the missing axis instead of indexing `dims[D - 1]`.
#[test]
fn rank_zero_activations_are_contextual_errors() {
    let device = Default::default();
    let scalar = Tensor::<Flex, 1>::from_floats([3.0], &device).reshape::<0, [usize; 0]>([]);
    let projection = Linear {
        weight: Param::from_tensor(Tensor::<Flex, 2>::from_floats([[2.0]], &device)),
        bias: None,
    };
    assert!(matches!(
        linear(scalar.clone(), &projection),
        Err(crate::Error::Shape { tensor, expected, actual })
            if tensor == "linear.input"
                && expected == "a tensor with at least one dimension"
                && actual == "rank zero"
    ));
    assert!(matches!(
        matmul_transposed(scalar, Tensor::<Flex, 2>::from_floats([[2.0]], &device)),
        Err(crate::Error::Shape { tensor, expected, actual })
            if tensor == "matmul_transposed.input"
                && expected == "a tensor with at least one dimension"
                && actual == "rank zero"
    ));
}

/// Autodiff refuses a packed weight before dispatch, even when the weight is frozen.
///
/// Burn 0.21 cannot differentiate a quantized matmul, including the input
/// gradient through a frozen packed matrix, so the boundary must name the
/// limitation and the explicit-dequantization workaround instead of panicking
/// in backward.
#[cfg(feature = "autodiff")]
#[test]
fn autodiff_projection_rejects_packed_weights() {
    /// Differentiable CPU backend that must not see packed parameters.
    type AdBackend = burn::backend::Autodiff<Flex>;
    let device = Default::default();
    let packed = quantize_tensor_data(
        &small_integers(32, 32, 3),
        WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q8,
            group_size: QuantizationGroupSize::new(32).expect("supported group"),
        },
    )
    .expect("reference quantization")
    .into_burn_qfloat_data()
    .expect("Burn-compatible matrix");
    // Burn 0.21's autodiff backend cannot upload packed data itself; wrap the
    // inner backend's packed primitive, which autodiff stores unchanged.
    let burn::tensor::TensorPrimitive::QFloat(inner) =
        Tensor::<Flex, 2>::from_data(packed, &device).into_primitive()
    else {
        panic!("packed data must upload as a quantized primitive");
    };
    let weight =
        Tensor::<AdBackend, 2>::from_primitive(burn::tensor::TensorPrimitive::QFloat(inner));
    assert!(matches!(weight.dtype(), DType::QFloat(_)));
    let projection = Linear {
        weight: Param::from_tensor(weight).set_require_grad(false),
        bias: None,
    };
    let input = Tensor::<AdBackend, 2>::ones([1, 32], &device).require_grad();
    match linear(input, &projection) {
        Err(crate::Error::BackendCapability(message)) => {
            assert!(message.contains("autodiff does not support packed Q4/Q8 linear weights"));
            assert!(message.contains("explicitly dequantize"));
        }
        other => panic!("expected a backend capability error, got {other:?}"),
    }
}

/// A projection with no output channels is a shape error, not an empty product.
#[test]
fn zero_width_projections_are_rejected() {
    let device = Default::default();
    let projection = Linear {
        weight: Param::from_tensor(Tensor::<Flex, 2>::from_data(
            TensorData::new(Vec::<f32>::new(), [2, 0]),
            &device,
        )),
        bias: None,
    };
    assert!(matches!(
        linear(Tensor::<Flex, 2>::ones([1, 2], &device), &projection),
        Err(crate::Error::Shape { tensor, expected, actual })
            if tensor == "linear.weight" && expected == "[2, nonzero_output]" && actual == "[2, 0]"
    ));
}
