//! Opt-in execution proof for Burn-native packed Q4/Q8 Metal projections.
//!
//! Run this gate with an exact single-backend build so CubeCL autotuning is not
//! feature-unified into Metal:
//!
//! ```text
//! cargo test --manifest-path crates/transformers/Cargo.toml \
//!   --no-default-features --features std,kernel-tracing \
//!   --test packed_metal_kernels -- --ignored --exact --nocapture
//! ```

#![cfg(all(
    feature = "kernel-tracing",
    target_os = "macos",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]

#[path = "support/kernel_trace.rs"]
mod kernel_trace;

use burn::{
    backend::Metal,
    module::Param,
    nn::Linear,
    tensor::{DType, Tensor, TensorData, backend::Backend},
};
use kernel_trace::{
    calibrated_dequantization_kernel_names, capture_kernel_launches, compact_kernel_id,
    packed_matmul_launches, whole_weight_dequantization_launches,
};
use transformers::{
    checkpoint::quantization::{
        QuantizationGroupSize, SymmetricQuantization, WeightQuantizationConfig,
        quantize_tensor_data,
    },
    operations::linear,
};

/// Input and output width used to force one nontrivial packed projection.
const MATRIX_WIDTH: usize = 64;

/// Builds a transposed Burn QFloat weight in the runtime `[input, output]` layout.
fn packed_weight(
    scheme: SymmetricQuantization,
    device: &<Metal as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<Metal, 2> {
    let values = (0..MATRIX_WIDTH * MATRIX_WIDTH)
        .map(|index| ((index % 127) as f32 - 63.0) / 32.0)
        .collect::<Vec<_>>();
    let quantized = quantize_tensor_data(
        &TensorData::new(values, [MATRIX_WIDTH, MATRIX_WIDTH]),
        WeightQuantizationConfig {
            scheme,
            group_size: QuantizationGroupSize::new(64).expect("group 64 must be supported"),
        },
    )
    .expect("the deterministic matrix must quantize");
    Tensor::<Metal, 2>::from_data(
        quantized
            .into_burn_qfloat_data()
            .expect("the matrix must have a Burn-compatible packed representation"),
        device,
    )
    .transpose()
}

/// Proves one packed scheme reaches a quantized matmul kernel without dequantization.
///
/// A one-row F16 input must submit exactly one packed matmul. Three-row inputs
/// must submit one packed matmul with F16 activations and `f32_three_row_launches`
/// with F32 activations: one for Q8, and one per row for Q4, whose multi-row
/// Burn 0.21 kernels are wrong with F32 activations and which
/// `operations::linear` therefore multiplies row by row. None may launch a
/// whole-weight dequantization.
fn assert_packed_projection_kernel(
    scheme: SymmetricQuantization,
    scheme_marker: &str,
    f32_three_row_launches: usize,
) {
    let device = Default::default();
    let weight = packed_weight(scheme, &device);
    assert!(matches!(weight.dtype(), DType::QFloat(_)));

    let calibration_weight = weight.clone();
    let (_, calibration_trace) = capture_kernel_launches(|| {
        calibration_weight
            .dequantize()
            .into_data()
            .to_vec::<f32>()
            .expect("explicit dequantization must be readable")
    });
    let dequantization_names = calibrated_dequantization_kernel_names(&calibration_trace);
    assert!(
        !dequantization_names.is_empty(),
        "explicit QFloat::dequantize must calibrate at least one dequantization kernel; trace: {calibration_trace:#?}"
    );

    let projection = Linear {
        weight: Param::from_tensor(weight),
        bias: None,
    };
    let input = Tensor::<Metal, 2>::ones([1, MATRIX_WIDTH], &device).cast(DType::F16);
    let (_, projection_trace) = capture_kernel_launches(|| {
        let output = linear(input, &projection).expect("packed Metal projection must dispatch");
        <Metal<f32, i32> as Backend>::sync(&device)
            .expect("packed Metal projection must synchronize");
        output
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .expect("packed Metal output must be readable")
    });

    let packed = packed_matmul_launches(&projection_trace);
    assert_eq!(
        packed.len(),
        1,
        "one projection must submit exactly one packed matmul; trace: {projection_trace:#?}"
    );
    assert!(packed[0].id.contains("QuantizedView"));
    // The checkpoint adapter initially packs dimension zero in `[output,
    // input]`. The zero-copy transpose into Burn's `[input, output]` linear
    // layout correctly moves that packed physical axis to dimension one.
    assert!(compact_kernel_id(packed[0]).contains("PackedU32(1"));
    assert!(
        packed[0].id.contains(scheme_marker),
        "packed kernel ID must encode {scheme_marker}: {}",
        packed[0].id
    );
    let dequantizations =
        whole_weight_dequantization_launches(&projection_trace, &dequantization_names);
    assert!(
        dequantizations.is_empty(),
        "packed matmul must not launch a calibrated whole-weight dequantization kernel: {dequantizations:#?}"
    );
    assert!(matches!(projection.weight.val().dtype(), DType::QFloat(_)));

    for (dtype, expected) in [(DType::F16, 1), (DType::F32, f32_three_row_launches)] {
        let rows = Tensor::<Metal, 2>::ones([3, MATRIX_WIDTH], &device).cast(dtype);
        let (_, rows_trace) = capture_kernel_launches(|| {
            let output = linear(rows, &projection).expect("packed Metal projection must dispatch");
            <Metal<f32, i32> as Backend>::sync(&device)
                .expect("packed Metal projection must synchronize");
            output
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()
                .expect("packed Metal output must be readable")
        });
        assert_eq!(
            packed_matmul_launches(&rows_trace).len(),
            expected,
            "a three-row {dtype:?} {scheme_marker} projection must submit {expected} packed matmuls; trace: {rows_trace:#?}"
        );
        assert!(
            whole_weight_dequantization_launches(&rows_trace, &dequantization_names).is_empty(),
            "a three-row packed projection must not dequantize the weight: {rows_trace:#?}"
        );
    }
    eprintln!(
        "{scheme_marker} packed Metal kernel: launches=1, quantized_view=true, packed_axis=1, whole_weight_dequantizations=0; three F32 rows: launches={f32_three_row_launches}"
    );
}

/// Captures Q4 and Q8 CubeCL IDs and rejects whole-matrix dequantization fallback.
#[test]
#[ignore = "requires a Metal-capable macOS host"]
fn packed_q4_q8_matmul_launches_quantized_metal_kernels() {
    assert_packed_projection_kernel(SymmetricQuantization::Q4, "Q4S", 3);
    assert_packed_projection_kernel(SymmetricQuantization::Q8, "Q8S", 1);
}
