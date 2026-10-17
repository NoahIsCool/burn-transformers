//! Burn Flex 0.21: layout ops on block-quantized tensors requantize.
//!
//! `q_swap_dims`, `q_permute`, `q_flip`, `q_expand`, `q_slice`, `q_reshape` and
//! `q_select` on a `QuantLevel::Block` tensor dequantize it, apply the op to
//! the floats, and quantize the result again with fresh blocks along the new
//! last axis (`block_safe_layout_op` in `burn-flex/src/ops/qtensor.rs`). A
//! transpose is metadata-only on every other backend, but here it changes the
//! values: each new block mixes elements of different original blocks, so a
//! small-magnitude row shares a scale with a large one and loses its precision.
//! The canonical case is a `LinearLayout::Col` weight loaded from a packed
//! `[out, in]` checkpoint and used as its transpose.

use burn::backend::Flex;
use burn::tensor::quantization::{
    QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
use burn::tensor::{Tensor, TensorData};

type B = Flex;

/// Compares transposing before and after dequantization per original row.
fn main() {
    let device = Default::default();
    // Two 64-wide rows, the second 100x smaller, quantized in 32-element blocks along each row.
    let (rows, cols) = (2, 64);
    let values: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let (r, c) = (i / cols, i % cols);
            let v = ((c as f32 * 0.37).sin() + 0.1) * if r == 0 { 1.0 } else { 0.01 };
            v
        })
        .collect();
    for (value, store) in [
        (QuantValue::Q8S, QuantStore::Native),
        (QuantValue::Q8S, QuantStore::PackedU32(0)),
        (QuantValue::Q4S, QuantStore::PackedU32(0)),
    ] {
        let scheme = QuantScheme::default()
            .with_value(value)
            .with_param(QuantParam::F32)
            .with_store(store)
            .with_level(QuantLevel::block([1, 32]))
            .with_mode(QuantMode::Symmetric);
        let w = Tensor::<B, 2>::from_data(TensorData::new(values.clone(), [rows, cols]), &device)
            .quantize_dynamic(&scheme);
        // Reference: dequantize in the stored layout, then transpose the floats.
        let expected: Vec<f32> = w
            .clone()
            .dequantize()
            .transpose()
            .into_data()
            .to_vec()
            .unwrap();
        // The same transpose applied to the quantized tensor.
        let actual: Vec<f32> = w.transpose().dequantize().into_data().to_vec().unwrap();
        // Relative error per original row (columns of the transposed tensor).
        for row in 0..rows {
            let (mut num, mut den) = (0.0f32, 0.0f32);
            for c in 0..cols {
                let i = c * rows + row;
                num = num.max((actual[i] - expected[i]).abs());
                den = den.max(expected[i].abs());
            }
            println!(
                "{value:?} {store:?}: transposed then dequantized, original row {row} (scale {}): max relative change {:.1e}",
                if row == 0 { "1" } else { "0.01" },
                num / den
            );
        }
    }
}
