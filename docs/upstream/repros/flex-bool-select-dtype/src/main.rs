//! Burn Flex 0.21: Boolean `select` and `gather` return tensors tagged `U8`.
//!
//! `bool_select`, `bool_gather` and `bool_select_or` reuse the numeric `u8`
//! kernels, which label their output `DType::U8` instead of the input's
//! `DType::Bool(..)`. The bytes are right, but Boolean consumers that check the
//! dtype then fail: `Tensor::cat` with another Boolean tensor panics, and
//! `into_data().to_vec::<bool>()` returns a `TypeMismatch` error.

use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::backend::Flex;
use burn::tensor::{Bool, Int, Tensor, TensorData};

/// The backend under test.
type B = Flex;

/// Selects and gathers Boolean rows and reads them back as Booleans.
fn main() {
    // Keep the panic message on one line for the report below.
    std::panic::set_hook(Box::new(|info| {
        let message = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or("?");
        println!("    panic: {}", message.lines().next().unwrap_or(""));
    }));
    let device = Default::default();
    let mask = Tensor::<B, 2, Bool>::from_bool(
        TensorData::from([[true, false], [false, true], [true, true]]),
        &device,
    );
    let rows = Tensor::<B, 1, Int>::from_ints([2, 0], &device);
    let columns = Tensor::<B, 2, Int>::from_ints([[1, 0], [0, 0], [1, 1]], &device);
    let other_row = Tensor::<B, 2, Bool>::from_bool(TensorData::from([[true, false]]), &device);

    println!("input dtype:  {:?}", mask.dtype());
    let mut failures = 0;
    for (name, result) in [
        ("select", mask.clone().select(0, rows.clone())),
        ("gather", mask.clone().gather(1, columns.clone())),
    ] {
        println!("{name} dtype: {:?}", result.dtype());
        let read = result.clone().into_data().to_vec::<bool>();
        println!("  to_vec::<bool>(): {read:?}");
        failures += read.is_err() as usize;
        let cat = catch_unwind(AssertUnwindSafe(|| {
            Tensor::cat(vec![result.clone(), other_row.clone()], 0).into_data()
        }));
        println!(
            "  cat with a Bool tensor: {}",
            match &cat {
                Ok(data) => format!("ok {:?}", data.to_vec::<bool>()),
                Err(_) => "panicked".to_string(),
            }
        );
        failures += cat.is_err() as usize;
    }
    println!("{failures} of 4 Boolean reads failed");
}
