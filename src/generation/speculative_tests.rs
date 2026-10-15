//! Speculative acceptance tests: losslessness, determinism, family parity, and prompt lookup.

use burn::{
    backend::Flex,
    tensor::{Tensor, TensorData},
};
use proptest::prelude::*;

use super::{
    SpeculativeVerifier, Verdict, argmax, normalized_probabilities, prompt_lookup_draft,
    sample_weights,
};
use crate::generation::{SamplingStrategy, SplitMix64};

/// Deterministic F32 CPU backend.
type TestBackend = Flex<f32, i32>;

/// The rejection rule exactly as Qwen3.5 and Nemotron-H implemented it before sharing.
///
/// Kept verbatim (state-incrementing SplitMix64, accept, residual, inverse
/// CDF) so the shared verifier can be shown to reproduce every decision bit
/// for bit, which keeps the families' seeded outputs unchanged on adoption.
struct FamilyReference {
    /// Wrapping SplitMix64 state.
    state: u64,
}

impl FamilyReference {
    /// Next 53-bit uniform in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        ((value >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Accept test `u < min(1, p / q)`.
    fn accept(&mut self, proposal: u32, p: &[f64], q: &[f64]) -> bool {
        let index = proposal as usize;
        self.next_unit() < (p[index] / q[index]).min(1.0)
    }

    /// Inverse-CDF draw from the positive residual.
    fn sample_residual(&mut self, p: &[f64], q: &[f64]) -> u32 {
        let residual = p
            .iter()
            .zip(q)
            .map(|(&target, &draft)| (target - draft).max(0.0))
            .collect::<Vec<_>>();
        let total = residual.iter().sum::<f64>();
        let threshold = self.next_unit() * total;
        let mut cumulative = 0.0;
        for (index, &weight) in residual.iter().enumerate() {
            cumulative += weight;
            if threshold < cumulative {
                return index as u32;
            }
        }
        (residual.len() - 1) as u32
    }
}

/// Normalizes non-negative weights into a distribution.
fn normalize(weights: &[f64]) -> Vec<f64> {
    let total = weights.iter().sum::<f64>();
    weights.iter().map(|weight| weight / total).collect()
}

/// Greedy verdicts accept exactly the target's own choice.
#[test]
fn greedy_verdicts_compare_with_the_target_choice() {
    assert_eq!(Verdict::greedy(3, 3), Verdict::Accepted);
    assert_eq!(Verdict::greedy(3, 1), Verdict::Rejected { correction: 1 });
    let mut verifier = SpeculativeVerifier::new(SamplingStrategy::Greedy, 9);
    let target = [0.1, 0.6, 0.3];
    let draft = [0.2, 0.2, 0.6];
    assert_eq!(
        verifier.verify(1, &target, &draft).unwrap(),
        Verdict::Accepted
    );
    assert_eq!(
        verifier.verify(2, &target, &draft).unwrap(),
        Verdict::Rejected { correction: 1 }
    );
    assert_eq!(
        verifier.verify_deterministic(0, &target).unwrap(),
        Verdict::Rejected { correction: 1 }
    );
    // Greedy verification consumes no randomness.
    assert_eq!(verifier.stream().position(), 0);
    assert_eq!(verifier.strategy(), SamplingStrategy::Greedy);
}

/// Speculative output is distributed exactly like target sampling.
///
/// For one verification with proposal `y ~ q`, the emitted token is `y` when
/// accepted and a residual draw otherwise; its distribution must be `p`.
#[test]
fn verified_tokens_follow_the_target_distribution() {
    let target = normalize(&[0.5, 0.2, 0.2, 0.1]);
    let draft = normalize(&[0.1, 0.6, 0.1, 0.2]);
    let mut verifier = SpeculativeVerifier::new(SamplingStrategy::Sample, 3);
    let mut proposals = SplitMix64::new(4);
    let draws = 40_000;
    let mut counts = [0_usize; 4];
    for _ in 0..draws {
        let proposal = sample_weights(&draft, proposals.next_unit_f64()).unwrap();
        let emitted = match verifier.verify(proposal, &target, &draft).unwrap() {
            Verdict::Accepted => proposal,
            Verdict::Rejected { correction } => correction,
        };
        counts[emitted as usize] += 1;
    }
    for (count, expected) in counts.iter().zip(&target) {
        let observed = *count as f64 / draws as f64;
        // Four standard errors of a binomial proportion at p <= 0.5.
        assert!(
            (observed - expected).abs() < 0.01,
            "{observed} vs {expected}"
        );
    }
}

/// A deterministic drafter (one-hot draft) is also lossless.
#[test]
fn deterministic_drafts_keep_the_target_distribution() {
    let target = normalize(&[0.3, 0.45, 0.25]);
    for proposal in 0..3_u32 {
        let mut verifier =
            SpeculativeVerifier::new(SamplingStrategy::Sample, 11 + u64::from(proposal));
        let draws = 30_000;
        let mut counts = [0_usize; 3];
        for _ in 0..draws {
            let emitted = match verifier.verify_deterministic(proposal, &target).unwrap() {
                Verdict::Accepted => proposal,
                Verdict::Rejected { correction } => {
                    assert_ne!(correction, proposal);
                    correction
                }
            };
            counts[emitted as usize] += 1;
        }
        for (count, expected) in counts.iter().zip(&target) {
            let observed = *count as f64 / draws as f64;
            assert!(
                (observed - expected).abs() < 0.012,
                "{observed} vs {expected}"
            );
        }
    }
    let mut verifier = SpeculativeVerifier::new(SamplingStrategy::Sample, 0);
    assert!(verifier.verify_deterministic(3, &target).is_err());
}

/// Copying the verifier snapshots its stream, so a rollback replays the same draws.
#[test]
fn verifier_copies_replay_identical_decisions() {
    let target = normalize(&[0.25, 0.25, 0.5]);
    let draft = normalize(&[0.6, 0.2, 0.2]);
    let mut verifier = SpeculativeVerifier::new(SamplingStrategy::Sample, 77);
    let snapshot = verifier;
    let first = (0..16)
        .map(|_| verifier.verify(0, &target, &draft).unwrap())
        .collect::<Vec<_>>();
    let mut restored = snapshot;
    let replay = (0..16)
        .map(|_| restored.verify(0, &target, &draft).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(first, replay);
    assert_eq!(verifier, restored);
}

/// Malformed distributions and draws fail with context instead of panicking.
#[test]
fn invalid_distributions_are_rejected() {
    let mut verifier = SpeculativeVerifier::new(SamplingStrategy::Sample, 0);
    assert!(verifier.verify(3, &[0.5, 0.5], &[0.5, 0.5]).is_err());
    assert!(verifier.verify(0, &[1.0], &[0.5, 0.5]).is_err());
    assert!(verifier.accept(0, &[1.0, 0.0], &[0.0, 1.0]).is_err());
    assert!(verifier.sample_residual(&[1.0], &[0.5, 0.5]).is_err());
    // Identical distributions leave no residual mass.
    assert!(verifier.sample_residual(&[0.5, 0.5], &[0.5, 0.5]).is_err());
    assert!(sample_weights(&[1.0, -0.5], 0.5).is_err());
    assert!(sample_weights(&[0.0, 0.0], 0.5).is_err());
    assert!(sample_weights(&[1.0], 1.0).is_err());
    assert!(sample_weights(&[1.0], f64::NAN).is_err());
    assert!(argmax(&[]).is_err());
}

/// Argmax resolves ties to the first index, like backend argmax.
#[test]
fn argmax_prefers_the_first_maximum() {
    assert_eq!(argmax(&[0.2, 0.4, 0.4]).unwrap(), 1);
    assert_eq!(argmax(&[0.5]).unwrap(), 0);
}

/// Host probabilities match an F64 softmax and reject non-normalizable rows.
#[test]
fn normalized_probabilities_validate_logits() {
    let device = Default::default();
    let probabilities = normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(
        [[0.0, 1.0, f32::NEG_INFINITY]],
        &device,
    ))
    .unwrap();
    let expected = normalize(&[1.0, 1.0_f64.exp(), 0.0]);
    for (actual, expected) in probabilities.iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-7);
    }
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    for invalid in [
        [[f32::NAN, 0.0]],
        [[f32::INFINITY, 0.0]],
        [[f32::NEG_INFINITY, f32::NEG_INFINITY]],
    ] {
        assert!(
            normalized_probabilities(Tensor::<TestBackend, 2>::from_floats(invalid, &device))
                .is_err()
        );
    }
    assert!(normalized_probabilities(Tensor::<TestBackend, 2>::zeros([2, 3], &device)).is_err());
    assert!(
        normalized_probabilities(Tensor::<TestBackend, 2>::from_data(
            TensorData::new(Vec::<f32>::new(), [1, 0]),
            &device
        ))
        .is_err()
    );
}

/// Prompt lookup prefers longer n-grams and returns nothing without a match.
#[test]
fn prompt_lookup_prefers_the_longest_matching_ngram() {
    let history = [9, 1, 2, 7, 8, 2, 5, 1, 2];
    // The 2-gram "1 2" occurs at 1 and continues with 7 8 2.
    assert_eq!(prompt_lookup_draft(&history, 1, 3, 3), [7, 8, 2]);
    // The 1-gram "2" alone also matches first at index 2.
    assert_eq!(prompt_lookup_draft(&history, 1, 1, 2), [7, 8]);
    assert!(prompt_lookup_draft(&history, 3, 3, 3).is_empty());
    assert!(prompt_lookup_draft(&[], 1, 2, 3).is_empty());
    assert!(prompt_lookup_draft(&history, 0, 2, 0).is_empty());
}

proptest! {
    /// The shared verifier reproduces the families' previous decisions bit for bit.
    #[test]
    fn shared_verifier_matches_family_reference(
        seed in any::<u64>(),
        target in proptest::collection::vec(0.0_f64..1.0, 2..8),
        draft in proptest::collection::vec(0.05_f64..1.0, 8),
        proposals in proptest::collection::vec(0_usize..8, 1..12),
    ) {
        let vocabulary = target.len();
        let mut target = target;
        target[0] += 0.01;
        let target = normalize(&target);
        let draft = normalize(&draft[..vocabulary]);
        let mut shared = SpeculativeVerifier::new(SamplingStrategy::Sample, seed);
        let mut reference = FamilyReference { state: seed };
        for proposal in proposals {
            let proposal = (proposal % vocabulary) as u32;
            let expected = if reference.accept(proposal, &target, &draft) {
                Verdict::Accepted
            } else {
                // A rejection implies p(y) < q(y), so the residual has mass.
                Verdict::Rejected { correction: reference.sample_residual(&target, &draft) }
            };
            prop_assert_eq!(shared.verify(proposal, &target, &draft).unwrap(), expected);
        }
    }

    /// Residual corrections only land where the target exceeds the draft.
    #[test]
    fn residual_draws_have_positive_residual_mass(
        target in proptest::collection::vec(0.0_f64..1.0, 2..10),
        draft in proptest::collection::vec(0.0_f64..1.0, 10),
        seed in any::<u64>(),
    ) {
        let vocabulary = target.len();
        let mut target = target;
        target[0] += 1.0;
        let target = normalize(&target);
        let draft = normalize(&draft[..vocabulary].iter().map(|value| value + 0.01).collect::<Vec<_>>());
        let mut verifier = SpeculativeVerifier::new(SamplingStrategy::Sample, seed);
        if let Ok(correction) = verifier.sample_residual(&target, &draft) {
            let index = correction as usize;
            prop_assert!(target[index] > draft[index]);
        }
    }

    /// Inverse-CDF sampling never returns a zero-weight index, at any uniform.
    #[test]
    fn weighted_sampling_skips_zero_weights(
        weights in proptest::collection::vec(proptest::sample::select(vec![0.0, 1e-300, 0.25, 1.0]), 1..12),
        bits in any::<u64>(),
    ) {
        let total = weights.iter().sum::<f64>();
        let uniform = crate::generation::random::unit_f64(bits);
        match sample_weights(&weights, uniform) {
            Ok(index) => prop_assert!(weights[index as usize] > 0.0),
            Err(_) => prop_assert!(total == 0.0),
        }
        if total > 0.0 {
            let last = sample_weights(&weights, crate::generation::random::unit_f64(u64::MAX)).unwrap();
            prop_assert!(weights[last as usize] > 0.0);
        }
    }
}
