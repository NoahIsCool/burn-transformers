//! Test support: a family sink's parameter installer keeps packed weights exact.
//!
//! Every family sink funnels its parameter writes through one private
//! `install(target, parameter, data)` helper. A quantized Burnpack sidecar
//! streams packed (QFloat) payloads into that helper, and the column-layout
//! linear weights it targets transpose the artifact on load. On a backend whose
//! quantized transpose requantizes (Burn Flex 0.21), a plain float install
//! would silently store a re-blocked approximation (the QNT-2 defect). This
//! check feeds a helper the payload most sensitive to that defect: Q4 blocks of
//! a `[3, 64]` artifact whose rows differ in scale by 40x, so a requantized
//! block mixes rows.

use burn::{
    backend::Flex,
    module::Param,
    nn::{LinearConfig, LinearLayout},
    tensor::{DType, Tensor, TensorData},
};

use super::packed_layout_is_exact;
use crate::checkpoint::quantization::{
    QuantizationGroupSize, SymmetricQuantization, WeightQuantizationConfig, quantize_tensor_data,
};

/// Asserts that `install` stores a packed column-layout weight as exactly the artifact.
///
/// `install` is called twice on fresh lazy `[64, 3]` column-layout linear
/// weights (artifact `[3, 64]`): once with a packed Q4 payload and once with
/// the same values as plain F32. The packed install must hold the artifact's
/// dequantized values bit for bit, staying packed exactly when the backend's
/// quantized layout is exact; the float install must be unchanged.
///
/// # Panics
///
/// When `install` fails or stores any other value.
pub(crate) fn assert_install_keeps_packed_weights_exact<E: core::fmt::Debug>(
    mut install: impl FnMut(&mut Param<Tensor<Flex, 2>>, TensorData) -> Result<(), E>,
) {
    let device = Default::default();
    let values = (0..192)
        .map(|index| ((index as f32) * 0.37).sin() * (1.0 + (index / 64) as f32 * 40.0))
        .collect::<Vec<_>>();
    let source = TensorData::new(values, [3, 64]);
    let packed = quantize_tensor_data(
        &source,
        WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q4,
            group_size: QuantizationGroupSize::new(32).unwrap(),
        },
    )
    .unwrap();
    let artifact = packed.dequantize_f32().unwrap().to_vec::<f32>().unwrap();
    let column_weight = || {
        LinearConfig::new(64, 3)
            .with_bias(false)
            .with_layout(LinearLayout::Col)
            .init::<Flex>(&device)
            .weight
    };
    // Runtime `[64, 3]` entry `(column, row)` is artifact entry `(row, column)`.
    let transposed = |values: &[f32]| -> Vec<u32> {
        (0..64)
            .flat_map(|column| (0..3).map(move |row| values[row * 64 + column].to_bits()))
            .collect()
    };

    let mut weight = column_weight();
    install(&mut weight, packed.into_burn_qfloat_data().unwrap()).unwrap();
    let installed = weight.val();
    let stays_packed = matches!(installed.dtype(), DType::QFloat(_));
    assert_eq!(
        stays_packed,
        packed_layout_is_exact::<Flex>(&device),
        "packed storage must be kept exactly when the backend's layout is exact"
    );
    let installed = if stays_packed {
        installed.dequantize()
    } else {
        installed
    };
    let bits = installed
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(
        bits,
        transposed(&artifact),
        "installed weight differs from the artifact"
    );

    let mut weight = column_weight();
    install(&mut weight, source.clone()).unwrap();
    let installed = weight.val();
    assert_eq!(installed.dtype(), DType::F32);
    let bits = installed
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(bits, transposed(&source.to_vec::<f32>().unwrap()));
}
