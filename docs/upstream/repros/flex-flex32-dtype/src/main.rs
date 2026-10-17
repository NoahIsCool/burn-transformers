//! Burn 0.21: a `DType::Flex32` float tensor on Flex panics in every kernel.
//!
//! `TensorData` tagged `Flex32` (read back from a CubeCL `flex32` model, or
//! saved by one) loads into Flex with the tag intact, and so do
//! `zeros`/`empty` with `FloatDType::Flex32`. Flex kernels dispatch only on
//! `F32`/`F64`/`F16`/`BF16`, so sorting, top-k, arithmetic, reductions,
//! softmax, matmul and even `cast(F32)` then panic with "unsupported dtype
//! Flex32" (`float_ones`/`float_full` reach `unreachable!()`), although
//! `float_random` already produces `F32` for a `Flex32` request.
//! Separately, `TensorData::to_vec::<f32>()` rejects `Flex32` data although its
//! bytes are f32, which also breaks the backend-agnostic fallback sort
//! (`burn_backend::ops::sort`) for any backend that keeps the tag.

use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::backend::Flex;
use burn::tensor::{DType, Tensor, TensorData, activation::softmax};

/// The backend under test.
type B = Flex;

/// Runs common operations on `Flex32` tensors and reports which panic.
fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let device = Default::default();
    let mut data = TensorData::from([[1.0_f32, 4.0, 3.0, 2.0]]);
    data.dtype = DType::Flex32;
    let logits = Tensor::<B, 2>::from_data(data, (&device, DType::Flex32));
    println!(
        "Flex32 data loaded as Flex32 on Flex: tensor dtype {:?}",
        logits.dtype()
    );
    let zeros = Tensor::<B, 2>::zeros([1, 4], (&device, DType::Flex32));
    println!(
        "zeros requested as Flex32 on Flex: tensor dtype {:?}",
        zeros.dtype()
    );
    /// One operation under test: maps a tensor to its values as `f32`.
    type Op = Box<dyn Fn(Tensor<B, 2>) -> Vec<f32>>;
    let values = |t: Tensor<B, 2>| t.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
    let ops: Vec<(&str, Op)> = vec![
        ("sort", Box::new(move |t| values(t.sort(1)))),
        ("topk(2)", Box::new(move |t| values(t.topk(2, 1)))),
        (
            "argsort",
            Box::new(|t| {
                t.argsort(1)
                    .float()
                    .into_data()
                    .convert::<f32>()
                    .to_vec()
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
                    .to_vec::<f32>()
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
    ];
    let mut panics = 0;
    for (name, op) in ops {
        match catch_unwind(AssertUnwindSafe(|| op(logits.clone()))) {
            Ok(v) => println!("  {name:<14} ok {v:?}"),
            Err(_) => {
                panics += 1;
                println!("  {name:<14} panicked");
            }
        }
    }
    let mut data = TensorData::from([1.0_f32, 2.0]);
    data.dtype = DType::Flex32;
    println!(
        "  TensorData(Flex32).to_vec::<f32>(): {:?}",
        data.to_vec::<f32>()
    );
    println!("{panics} of 9 operations panicked");
}
