//! Burn 0.21's CubeCL maximum of `-inf` values is `f32::MIN`, and the minimum of `+inf` is `f32::MAX`.
//!
//! cubek-reduce 0.2 seeds its `Max` reduction with `min_value()`, the lowest
//! *finite* value of the element type, and `Min` with `max_value()`. A row
//! whose every entry is `-inf` therefore reduces to `-3.4028235e38` instead
//! of `-inf`, and a row of `+inf` to `3.4028235e38`. Code that compares a row
//! with its own maximum (`x.equal(x.max_dim(d))`, a common way to find the
//! argmax set or to implement a deterministic top-k) then finds no match.
//!
//! The reductions run on the CubeCL CPU backend and, with `--features metal`,
//! on Metal; Flex is the control.

use burn::backend::{Cpu, Flex};
use burn::tensor::{Tensor, backend::Backend};

/// Rows of `-inf`, of `+inf`, and a finite control, `[3, 4]`.
const ROWS: [[f32; 4]; 3] = [
    [f32::NEG_INFINITY; 4],
    [f32::INFINITY; 4],
    [-1.0, 2.0, -3.0, 0.5],
];

/// Exact expected `[max_dim, min_dim]` of each row of [`ROWS`].
const EXPECTED: [[f32; 2]; 3] = [
    [f32::NEG_INFINITY, f32::NEG_INFINITY],
    [f32::INFINITY, f32::INFINITY],
    [2.0, -3.0],
];

/// Reduces every row of [`ROWS`] on `B`, prints the results, and returns the number of wrong values.
fn run<B: Backend>(name: &str) -> usize {
    let device = Default::default();
    let rows = Tensor::<B, 2>::from_floats(ROWS, &device);
    let maxima: Vec<f32> = rows
        .clone()
        .max_dim(1)
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();
    let minima: Vec<f32> = rows
        .clone()
        .min_dim(1)
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();
    let argmax: Vec<i64> = rows
        .clone()
        .argmax(1)
        .into_data()
        .convert::<i64>()
        .to_vec()
        .unwrap();
    // The idiom the wrong maximum breaks: which entries equal their row maximum?
    let matches: Vec<bool> = rows
        .clone()
        .equal(rows.max_dim(1))
        .any_dim(1)
        .int()
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .unwrap()
        .into_iter()
        .map(|matched| matched != 0)
        .collect();
    let mut wrong = 0;
    println!("{name}:");
    for (row, expected) in EXPECTED.iter().enumerate() {
        let actual = [maxima[row], minima[row]];
        let bad = actual != *expected || !matches[row];
        wrong += usize::from(actual[0] != expected[0]) + usize::from(actual[1] != expected[1]);
        println!(
            "  row {row}: max {:>15e} min {:>15e}  (expected {:e}, {:e}); argmax {}; some entry equals the max: {}{}",
            actual[0],
            actual[1],
            expected[0],
            expected[1],
            argmax[row],
            matches[row],
            if bad { "   <-- wrong" } else { "" }
        );
    }
    wrong
}

/// Runs the reductions on every enabled backend and prints how many values were wrong.
fn main() {
    let mut wrong = run::<Flex>("flex (control)");
    wrong += run::<Cpu>("cubecl-cpu");
    #[cfg(feature = "metal")]
    {
        wrong += run::<burn::backend::Metal>("metal");
    }
    println!("{wrong} wrong reduction values");
}
