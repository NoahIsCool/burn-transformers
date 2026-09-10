//! Small, explicit statistics for the paper's complete modular-addition grid.
//!
//! Flattened F32 logits have axes `[a,b,c]`, each of length 59, and class
//! `(a+b) mod 59` is correct. Statistical sums use F64 population moments.
//! These diagnostics describe full-grid behavior; they are not training losses
//! or a held-out evaluation. Formula fits compare both standardized tensors,
//! preserving the paper’s `2*correlation-1` convention.
use super::model::P;

/// Return argmax accuracy over all 59² rows of a complete `[a,b,class]` grid.
pub fn accuracy(logits: &[f32]) -> f64 {
    let correct = logits
        .chunks_exact(P)
        .enumerate()
        .filter(|(pair, row)| {
            let prediction = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            prediction == (pair / P + pair % P) % P
        })
        .count();
    correct as f64 / (P * P) as f64
}

/// Compute the F64 population standard deviation of a nonempty sample.
/// The denominator is N rather than N-1, matching the paper’s grid statistics.
fn std(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

/// Definition 4.2: mean within-difference standard deviation / global deviation.
/// All standard deviations are population statistics (`ddof=0`).
pub fn distance_irrelevance(logits: &[f32]) -> f64 {
    let correct: Vec<f64> = (0..P * P)
        .map(|pair| logits[pair * P + (pair / P + pair % P) % P] as f64)
        .collect();
    let within: f64 = (0..P)
        .map(|difference| {
            let row: Vec<_> = (0..P)
                .map(|a| correct[a * P + (a + difference) % P])
                .collect();
            std(&row)
        })
        .sum::<f64>()
        / P as f64;
    within / std(&correct)
}

/// Table 2: standardize BOTH complete tensors, then 1 - variance(error).
/// Algebraically this is `2*correlation-1`; it is not squared correlation.
/// Clock predicts `cos(omega*(a+b-c))`; Pizza multiplies that by
/// `abs(cos(omega*(a-b)/2))`, where `omega = 2*pi*frequency/59`.
pub fn formula_fve(logits: &[f32], frequency: usize, pizza: bool) -> f64 {
    let (mut x, mut y, mut xx, mut yy, mut xy) = (0., 0., 0., 0., 0.);
    for (index, &actual) in logits.iter().enumerate() {
        let (a, b, c) = (
            (index / P / P) as f64,
            (index / P % P) as f64,
            (index % P) as f64,
        );
        let omega = std::f64::consts::TAU * frequency as f64 / P as f64;
        let mut predicted = (omega * (a + b - c)).cos();
        if pizza {
            predicted *= (omega * (a - b) / 2.).cos().abs();
        }
        let actual = actual as f64;
        x += predicted;
        y += actual;
        xx += predicted * predicted;
        yy += actual * actual;
        xy += predicted * actual;
    }
    let n = logits.len() as f64;
    let correlation = (xy - x * y / n) / ((xx - x * x / n) * (yy - y * y / n)).sqrt();
    2. * correlation - 1.
}

/// Return the largest absolute F32 difference, requiring equal flattened lengths.
pub fn max_error(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max)
}

/// Check numerical parity with an independently evaluated PyTorch graph.
/// Require finite complete grids and per-logit error `<= 2e-3 + 2e-5*abs(ref)`,
/// identical accuracy, and distance-irrelevance agreement within `2e-5`.
pub fn assert_reference(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), P * P * P);
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert!(
            (a - b).abs() <= 2e-3 + 2e-5 * b.abs(),
            "reference logit {index}: actual {a}, reference {b}"
        );
    }
    assert_eq!(
        accuracy(actual),
        accuracy(expected),
        "reference predictions"
    );
    assert!(
        (distance_irrelevance(actual) - distance_irrelevance(expected)).abs() < 2e-5,
        "reference distance irrelevance"
    );
    max_error(actual, expected)
}
