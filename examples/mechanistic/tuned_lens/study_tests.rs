//! Checks probability normalization and loss accuracy for a large vocabulary with a significant low-logit tail.
use super::*;

/// Checks tail mass, forward KL, and shifted-target cross entropy against independent scalar F64 formulas.
#[test]
fn long_vocabulary_tail_remains_normalized_and_has_accurate_kl() {
    // One large logit followed by 50,303 small but collectively significant
    // alternatives. Compare with an independent scalar F64 calculation.
    let vocabulary = 50_304;
    let mut values = vec![-12.0_f32; vocabulary * 2];
    values[0] = 0.0;
    values[vocabulary] = 0.0;
    let final_log_probs = log_probabilities(Tensor::from_data(
        TensorData::new(values, [1, 2, vocabulary]),
        &Default::default(),
    ))
    .unwrap();
    let uniform_log_probs =
        log_probabilities(Tensor::zeros([1, 2, vocabulary], &Default::default())).unwrap();
    let log_normalizer = (1.0 + (vocabulary - 1) as f64 * (-12.0_f64).exp()).ln();
    let large = (-log_normalizer).exp();
    let small = (-12.0 - log_normalizer).exp();
    let expected_kl = large * (-log_normalizer + (vocabulary as f64).ln())
        + (vocabulary - 1) as f64 * small * (-12.0 - log_normalizer + (vocabulary as f64).ln());
    let mass: f64 = final_log_probs
        .clone()
        .narrow(1, 0, 1)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(|value| f64::from(value).exp())
        .sum();
    assert!((mass - 1.0).abs() < 3e-6, "mass={mass}");
    let (kl_sum, ce_sum) = losses(uniform_log_probs, final_log_probs, &[0, 1]).unwrap();
    assert!((kl_sum / 2.0 - expected_kl).abs() < 2e-5);
    assert!((ce_sum - (vocabulary as f64).ln()).abs() < 2e-6);
}
