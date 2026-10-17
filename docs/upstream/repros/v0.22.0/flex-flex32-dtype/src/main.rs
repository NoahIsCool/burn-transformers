//! Burn 0.22: a `DType::Flex32` float tensor on Flex panics in every kernel.
//!
//! Two ordinary requests create a `Flex32`-tagged Flex tensor: `zeros` (or
//! `empty`) with `FloatDType::Flex32`, and `from_data` of `TensorData` built from
//! `flex32` elements (as read back from, or saved by, a CubeCL `flex32` model).
//! Flex kernels dispatch only on `F32`/`F64`/`F16`/`BF16`, so sorting, top-k,
//! arithmetic, reductions, softmax, matmul and even `cast(F32)` then panic, and
//! `ones`/`full` reach `unreachable!()`. `float_random` already produces `F32`
//! for a `Flex32` request. (Converting `f32` data with
//! `convert_dtype(DType::Flex32)` yields `F32` data in 0.22.0; `main` keeps the
//! `Flex32` marker there too.)

use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::tensor::{DType, Device, Tensor, TensorData, activation::softmax, flex32};

/// One operation under test: maps a tensor to its values as `f32`.
type Op = Box<dyn Fn(Tensor<2>) -> Vec<f32>>;

/// The operations tried on each `Flex32` tensor.
fn operations() -> Vec<(&'static str, Op)> {
    let values = |t: Tensor<2>| t.cast(DType::F32).into_data().try_to_vec::<f32>().unwrap();
    vec![
        ("sort", Box::new(move |t| values(t.sort(1)))),
        ("topk(2)", Box::new(move |t| values(t.topk(2, 1)))),
        (
            "argsort",
            Box::new(|t| {
                t.argsort(1)
                    .float()
                    .into_data()
                    .convert::<f32>()
                    .try_to_vec::<f32>()
                    .unwrap()
            }),
        ),
        ("add", Box::new(move |t| values(t.clone() + t))),
        (
            "sum",
            Box::new(|t| {
                t.sum()
                    .cast(DType::F32)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap()
            }),
        ),
        ("softmax", Box::new(move |t| values(softmax(t, 1)))),
        (
            "matmul",
            Box::new(move |t| values(t.clone().matmul(t.transpose()))),
        ),
        (
            "ones_like + 1",
            Box::new(move |t| values(t.ones_like() + 1.0)),
        ),
        ("cast(F32)", Box::new(move |t| values(t))),
    ]
}

/// Runs every operation on `tensor`, prints each outcome, and returns the panic count.
fn run_operations(label: &str, tensor: &Tensor<2>) -> usize {
    println!("{label}: tensor dtype {:?}", tensor.dtype());
    let mut panics = 0;
    for (name, op) in operations() {
        match catch_unwind(AssertUnwindSafe(|| op(tensor.clone()))) {
            Ok(v) => println!("  {name:<14} ok {v:?}"),
            Err(_) => {
                panics += 1;
                println!("  {name:<14} panicked");
            }
        }
    }
    panics
}

/// Creates `Flex32` tensors on Flex both ways and reports which operations panic.
fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let device = Device::flex();
    let data = TensorData::new(
        [1.0_f32, 4.0, 3.0, 2.0].map(flex32::from_f32).to_vec(),
        [1, 4],
    );
    println!("TensorData from flex32 elements: dtype {:?}", data.dtype());
    let from_data = Tensor::<2>::from_data(data.clone(), &device);
    let zeros = Tensor::<2>::zeros([1, 4], (&device, DType::Flex32));
    let panics = run_operations("from_data(flex32 elements) on Flex", &from_data)
        + run_operations("zeros(.., Flex32) on Flex", &zeros);
    println!(
        "  TensorData(flex32).try_to_vec::<f32>(): {:?}",
        data.try_to_vec::<f32>()
    );
    println!("{panics} of 18 operations panicked");
}
