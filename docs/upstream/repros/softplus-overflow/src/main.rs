//! Burn 0.21 `activation::softplus` evaluates `log(exp(beta * x) + 1) / beta`.
//!
//! In F32 `exp` overflows above 88.72, so softplus returns `inf` there and its
//! gradient is `NaN`; below about -17 `exp(x) + 1` rounds to 1 and the value
//! collapses to 0 (relative error 100%). PyTorch evaluates
//! `x if beta * x > 20 else log1p(exp(beta * x)) / beta`.

use burn::backend::{Autodiff, Flex};
use burn::tensor::{Tensor, activation::softplus};

type B = Autodiff<Flex>;

/// Prints values and gradients against PyTorch's thresholded softplus.
fn main() {
    let device = Default::default();
    let inputs = [
        -30.0f32, -20.0, -1.0, 0.0, 1.0, 20.0, 50.0, 88.0, 89.0, 100.0,
    ];
    let x = Tensor::<B, 1>::from_floats(inputs, &device).require_grad();
    let y = softplus(x.clone(), 1.0);
    let grads = y.clone().sum().backward();
    let values: Vec<f32> = y.into_data().to_vec().unwrap();
    let gradient: Vec<f32> = x.grad(&grads).unwrap().into_data().to_vec().unwrap();

    println!(
        "{:>8} {:>14} {:>14} {:>12} {:>12}",
        "x", "burn value", "reference", "burn grad", "reference"
    );
    let mut bad = 0;
    for ((&x, &value), &grad) in inputs.iter().zip(&values).zip(&gradient) {
        let x64 = x as f64;
        let reference = if x64 > 20.0 { x64 } else { x64.exp().ln_1p() };
        let sigmoid = 1.0 / (1.0 + (-x64).exp());
        let value_ok = (value as f64 - reference).abs() <= 1e-6 * reference.abs().max(1e-30) * 10.0;
        let grad_ok = (grad as f64 - sigmoid).abs() <= 1e-6;
        if !(value_ok && grad_ok) {
            bad += 1;
        }
        println!(
            "{x:>8} {value:>14.6e} {reference:>14.6e} {grad:>12.6} {sigmoid:>12.6}{}",
            if value_ok && grad_ok {
                ""
            } else {
                "   <-- wrong"
            }
        );
    }
    println!(
        "{bad} of {} inputs have a wrong value or gradient",
        inputs.len()
    );
}
