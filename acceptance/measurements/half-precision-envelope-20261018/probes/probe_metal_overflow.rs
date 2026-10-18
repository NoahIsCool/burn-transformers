//! Scratch probe: does Burn Metal F16 arithmetic overflow to infinity above 65,504 (not committed)?
#![allow(missing_docs)]
#![cfg(all(feature = "metal", target_os = "macos"))]
use burn::{backend::Metal, tensor::{DType, Tensor, TensorData}};

fn host(t: Tensor<Metal, 1>) -> Vec<f32> { t.cast(DType::F32).into_data().to_vec::<f32>().unwrap() }

#[test]
fn metal_f16_overflow_semantics() {
    let device = Default::default();
    let big = Tensor::<Metal, 1>::from_data(TensorData::new(vec![60000.0f32, 65504.0, 65519.0, 65520.0, 70000.0, 1e6], [6]), &device);
    eprintln!("F32 -> F16 cast: {:?}", host(big.clone().cast(DType::F16)));
    let a = Tensor::<Metal, 1>::from_data(TensorData::new(vec![51941.5f32, 60000.0, 65504.0, -60000.0], [4]), &device).cast(DType::F16);
    let b = Tensor::<Metal, 1>::from_data(TensorData::new(vec![21568.1f32, 10000.0, 32.0, -10000.0], [4]), &device).cast(DType::F16);
    eprintln!("F16 add: {:?}", host(a.clone() + b.clone()));
    eprintln!("F16 mul x2: {:?}", host(a.clone() * 2.0));
    eprintln!("F16 square: {:?}", host(a.clone() * a.clone()));
    let m = Tensor::<Metal, 2>::from_data(TensorData::new(vec![300.0f32; 4], [1, 4]), &device).cast(DType::F16);
    let n = Tensor::<Metal, 2>::from_data(TensorData::new(vec![300.0f32; 4], [4, 1]), &device).cast(DType::F16);
    eprintln!("F16 matmul 4 x 300 x 300 = 360000: {:?}", m.matmul(n).cast(DType::F32).into_data().to_vec::<f32>().unwrap());
    let s = Tensor::<Metal, 1>::from_data(TensorData::new(vec![40000.0f32; 4], [4]), &device).cast(DType::F16);
    eprintln!("F16 sum of 4 x 40000: {:?}", host(s.clone().sum()));
    eprintln!("F16 mean of 4 x 40000: {:?}", host(s.mean()));
}
