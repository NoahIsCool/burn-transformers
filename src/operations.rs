//! Quantization-preserving transformer operations.
//!
//! Burn's high-level linear module currently extracts a full-precision
//! primitive from a quantized tensor before dispatch. In contrast, tensor
//! matrix multiplication retains [`burn::tensor::TensorPrimitive::QFloat`] and
//! lets CubeCL consume packed values and scales inside its matmul kernel. This
//! module supplies the narrow projection boundary needed by transformer layers
//! while continuing to store parameters in Burn's ordinary [`burn::nn::Linear`]
//! modules.
//!
//! For an input with shape `[..., I]`, weight `[I, O]`, and optional bias `[O]`,
//! [`linear`] evaluates `reshape(x, [-1, I]) @ W + b` and restores
//! `[..., O]`. Flattening only the activation's leading axes leaves packed
//! weight block boundaries and packing dimensions unchanged.

use burn::{
    nn::Linear,
    tensor::{DType, Tensor, backend::Backend},
};

use crate::{Error, Result};

/// Applies a Burn linear module without dequantizing a packed weight up front.
///
/// Full-precision weights take the ordinary backend matmul path. A `QFloat`
/// weight reaches the backend's quantized matmul dispatch, which CubeCL can
/// execute from packed Q8/Q4 storage. Activations deliberately remain floating
/// point in v1; quantized activations are rejected because reshaping sub-byte
/// tensors is not generally implemented by Burn 0.21.
pub fn linear<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    projection: &Linear<B>,
) -> Result<Tensor<B, D>> {
    if D == 0 {
        return Err(Error::shape(
            "linear.input",
            "a tensor with at least one dimension",
            "rank zero",
        ));
    }
    let mut output_shape = input.dims();
    if output_shape.contains(&0) {
        return Err(Error::shape(
            "linear.input",
            "non-zero dimensions ending in the projection input width",
            format!("{output_shape:?}"),
        ));
    }
    if matches!(input.dtype(), DType::QFloat(_)) {
        return Err(Error::BackendCapability(
            "v1 packed linear supports floating activations with a quantized weight, not quantized activations"
                .into(),
        ));
    }

    let weight = projection.weight.val();
    #[cfg(all(feature = "metal", feature = "rocm"))]
    if packed_metal_autotune_is_unavailable(weight.dtype(), &B::name(&weight.device())) {
        return Err(Error::BackendCapability(
            "packed Q4/Q8 Metal matmul is unavailable when the `rocm` crate feature is also enabled: Burn 0.21 feature unification enables a CubeCL autotuner with quantized-unsafe candidates; build the Metal runtime with `--no-default-features --features std,metal` (plus optional tokenizer features)"
                .into(),
        ));
    }
    let [weight_input, weight_output] = weight.dims();
    let input_width = output_shape[D - 1];
    if weight_input != input_width || weight_output == 0 {
        return Err(Error::shape(
            "linear.weight",
            format!("[{input_width}, nonzero_output]"),
            format!("[{weight_input}, {weight_output}]"),
        ));
    }
    if let Some(bias) = &projection.bias {
        let [bias_width] = bias.shape().dims();
        if bias_width != weight_output {
            return Err(Error::shape(
                "linear.bias",
                format!("[{weight_output}]"),
                format!("[{bias_width}]"),
            ));
        }
    }
    let rows = output_shape[..D - 1]
        .iter()
        .try_fold(1_usize, |product, dimension| {
            product.checked_mul(*dimension)
        })
        .ok_or_else(|| Error::shape("linear.input", "representable leading axes", "overflow"))?;
    output_shape[D - 1] = weight_output;

    let mut output = input.reshape([rows, input_width]).matmul(weight);
    if let Some(bias) = &projection.bias {
        output = output + bias.val().unsqueeze();
    }
    Ok(output.reshape(output_shape))
}

/// Detects Burn 0.21's known packed Metal/autotune feature-unification hazard.
///
/// Enabling this crate's ROCm feature activates `burn-cubecl/autotune` for the
/// shared CubeCL dependency. Cargo then applies that strategy to Metal as well,
/// where Burn 0.21 can benchmark quantized-incompatible candidate kernels. The
/// backend name check limits the guard to Metal; non-Metal execution keeps its
/// independently validated path. Both Q4 and Q8 use packed U32 storage and can
/// enter candidates whose logical dequantized vector width differs from the
/// floating activation width.
#[cfg(all(feature = "metal", feature = "rocm"))]
fn packed_metal_autotune_is_unavailable(dtype: DType, backend_name: &str) -> bool {
    matches!(dtype, DType::QFloat(_)) && backend_name.contains("wgpu<msl>")
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        module::Param,
        nn::{Linear, LinearConfig},
        tensor::{DType, Tensor, TensorData},
    };

    use super::linear;
    use crate::checkpoint::quantization::{
        QuantizationGroupSize, SymmetricQuantization, WeightQuantizationConfig,
        quantize_tensor_data,
    };

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
}
