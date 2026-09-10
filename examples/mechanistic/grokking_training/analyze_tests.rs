//! Grokking sinusoid-fit regression.

use super::*;
#[test]
fn affine_sinusoid_fit_retains_a_nonzero_activation_mean() {
    let mut fit = Fit::default();
    for a in 0..P {
        for b in 0..P {
            let x = (std::f64::consts::TAU * 7.0 * (a + b) as f64 / P as f64).cos();
            fit.add(x, 3.0 + 2.0 * x);
        }
    }
    assert!(
        fit.fve() < 0.0,
        "legacy metric penalizes the unmodeled mean"
    );
    assert!((fit.affine_fve() - 1.0).abs() < 1e-10);
}
