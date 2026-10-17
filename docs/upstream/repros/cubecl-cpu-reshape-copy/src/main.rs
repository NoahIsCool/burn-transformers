//! Burn 0.21's CubeCL CPU backend copies a reshaped strided view wrongly.
//!
//! When a reshape cannot be expressed by new strides, `burn-cubecl` copies the
//! tensor into a buffer of the new shape with `cubecl_std::tensor::copy_into`.
//! On a CPU client whose input's innermost stride is not 1 (a transposed view,
//! or a view expanded along its last axis), cubecl-std 0.10 takes the
//! "perpendicular" copy kernel, which reads the input at the *output's*
//! coordinates. That is only correct when input and output have the same
//! shape; a reshape gives them different shapes, so the kernel reads the
//! wrong elements, and beyond the input buffer when the output has more
//! elements than the input's storage. GPU clients take the general copy and
//! are correct.
//!
//! Each case below compares the CubeCL CPU result with the same operation
//! evaluated on the host.

use burn::backend::Cpu;
use burn::tensor::{Int, Tensor, TensorData};

/// The backend under test: CubeCL's CPU runtime with F32 floats and I32 indices.
type B = Cpu;

/// Prints one case and returns whether it was wrong.
fn report(name: &str, actual: Vec<f32>, expected: Vec<f32>) -> bool {
    let wrong = actual != expected;
    println!("{name}:");
    println!("  cubecl-cpu {actual:?}");
    println!(
        "  expected   {expected:?}{}",
        if wrong { "   <-- wrong" } else { "" }
    );
    wrong
}

/// Reads any float tensor to the host.
fn host<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().convert::<f32>().to_vec().unwrap()
}

/// Reads any integer tensor to the host as F32.
fn host_int<const D: usize>(tensor: Tensor<B, D, Int>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .unwrap()
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

/// Runs every case and prints how many were wrong.
fn main() {
    let device = Default::default();
    let mut wrong = 0;
    let mut cases = 0;

    // 1. A transposed matrix flattened: [2, 3] -> swap -> [3, 2] (strides [1, 3]) -> [6].
    let matrix = Tensor::<B, 2>::from_data(
        TensorData::new(vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3]),
        &device,
    );
    let transposed = matrix.clone().swap_dims(0, 1);
    cases += 1;
    wrong += report(
        "transpose [2, 3] then reshape to [6]",
        host(transposed.clone().reshape([6])),
        vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
    ) as usize;
    // Control: the same view copied without changing its shape is correct.
    cases += 1;
    wrong += report(
        "transpose [2, 3], read without a reshape (control)",
        host(transposed),
        vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
    ) as usize;

    // 2. Interleaved pairs to split halves, as RoPE layouts do:
    //    [8] -> [4, 2] -> swap -> [2, 4] (strides [1, 2]) -> [8].
    let pairs = Tensor::<B, 1>::from_data(
        TensorData::new((0..8).map(|value| value as f32).collect::<Vec<_>>(), [8]),
        &device,
    );
    cases += 1;
    wrong += report(
        "de-interleave [8] via [4, 2] -> swap -> reshape [8]",
        host(pairs.reshape([4, 2]).swap_dims(0, 1).reshape([8])),
        vec![0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0],
    ) as usize;

    // 3. A trailing unit axis expanded and flattened:
    //    [2, 2, 1] -> expand [2, 2, 3] (strides [2, 1, 0]) -> reshape [2, 6].
    let columns = Tensor::<B, 3>::from_data(
        TensorData::new(vec![0.0f32, 1.0, 1.0, 0.0], [2, 2, 1]),
        &device,
    );
    let expected = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    cases += 1;
    wrong += report(
        "float [2, 2, 1] expand [2, 2, 3] then reshape [2, 6]",
        host(columns.clone().expand([2, 2, 3]).reshape([2, 6])),
        expected.clone(),
    ) as usize;
    // The integer copy reads past the four-element input buffer.
    cases += 1;
    wrong += report(
        "int [2, 2, 1] expand [2, 2, 3] then reshape [2, 6]",
        host_int(columns.clone().int().expand([2, 2, 3]).reshape([2, 6])),
        expected.clone(),
    ) as usize;
    cases += 1;
    wrong += report(
        "bool [2, 2, 1] expand [2, 2, 3] then reshape [2, 6]",
        host_int(
            columns
                .clone()
                .greater_elem(0.5)
                .expand([2, 2, 3])
                .reshape([2, 6])
                .int(),
        ),
        expected.clone(),
    ) as usize;
    // Control: the expanded view read without a reshape is correct.
    cases += 1;
    wrong += report(
        "float [2, 2, 1] expand [2, 2, 3], read without a reshape (control)",
        host(columns.expand([2, 2, 3])),
        expected,
    ) as usize;

    println!("{wrong} of {cases} cases are wrong");
}
