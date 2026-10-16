//! Family-independent pieces of lossless speculative decoding.
//!
//! Speculative decoding (Leviathan, Kalman, and Matias, 2023; Chen et al.,
//! 2023) lets a cheap drafter propose tokens `y_1..y_K` from distributions
//! `q_i`, then checks them with one target pass that yields `p_i`. Proposal
//! `y_i` is accepted with probability
//!
//! ```text
//! a_i = min(1, p_i(y_i) / q_i(y_i))
//! ```
//!
//! and on the first rejection the target emits a correction drawn from the
//! normalized residual `r(x) ∝ max(0, p_i(x) - q_i(x))`. If all `K` proposals
//! are accepted, the target emits one bonus token from `p_(K+1)`. Every emitted
//! token is then distributed exactly as target sampling would be, because
//! `q(x) a(x) + (1 - sum_y q(y) a(y)) r(x) = p(x)`. Greedy decoding uses the
//! degenerate case: accept exactly when the target's own choice equals `y_i`,
//! otherwise emit that choice.
//!
//! This module holds what every drafter shares:
//!
//! - [`SpeculativeVerifier`] applies the acceptance rule with a dedicated
//!   [`SplitMix64`] stream, so verification never perturbs token sampling
//!   and a copied verifier restores the stream after a rolled-back cycle;
//! - [`speculative_commit`] computes how many emitted tokens and cache entries
//!   a cycle publishes (verified with Verus; see `bookkeeping`);
//! - [`normalized_probabilities`], [`argmax`], and [`sample_weights`] are the
//!   host-side distribution helpers used by batch-one verifiers;
//! - [`prompt_lookup_draft`] is a model-free drafter that proposes the tokens
//!   that followed an earlier occurrence of the current n-gram.
//!
//! Model-specific work (MTP heads, cache transactions) stays in the families.
//! Host probability vectors cost `O(vocabulary)` per verified token; that is
//! small next to a target forward pass for the batch-one decoders that use
//! them.

use burn::tensor::{Tensor, activation::softmax, backend::Backend};

use crate::{Error, Result};

use super::bookkeeping::prompt_lookup_draft_span;
use super::{SamplingStrategy, SplitMix64, metadata, probability_accumulation_dtype};

pub use super::bookkeeping::{SpeculativeCommit, SpeculativeOutcome, speculative_commit};

/// Outcome of verifying one draft proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The proposal is kept; verification continues with the next one.
    Accepted,
    /// The proposal is discarded and `correction` is emitted in its place.
    Rejected {
        /// Target-chosen replacement token.
        correction: u32,
    },
}

impl Verdict {
    /// Greedy verification: accept exactly when the target chose the same token.
    #[must_use]
    pub const fn greedy(proposal: u32, target_choice: u32) -> Self {
        if proposal == target_choice {
            Self::Accepted
        } else {
            Self::Rejected {
                correction: target_choice,
            }
        }
    }
}

/// Applies the speculative acceptance rule with its own deterministic stream.
///
/// The verifier is `Copy`: saving a copy before a cycle and restoring it after
/// a rollback replays exactly the same uniforms. Each stochastic verification
/// consumes one uniform for the accept test and, after a rejection, one more
/// for the residual draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeVerifier {
    /// Whether proposals are checked greedily or by rejection sampling.
    strategy: SamplingStrategy,
    /// Uniforms for acceptance tests and residual sampling.
    stream: SplitMix64,
}

impl SpeculativeVerifier {
    /// Creates a verifier whose uniforms come from the stream selected by `seed`.
    ///
    /// Callers should derive `seed` from the generation seed with a domain
    /// constant so verification draws never coincide with token sampling.
    #[must_use]
    pub const fn new(strategy: SamplingStrategy, seed: u64) -> Self {
        Self {
            strategy,
            stream: SplitMix64::new(seed),
        }
    }

    /// Returns the configured verification strategy.
    #[must_use]
    pub const fn strategy(&self) -> SamplingStrategy {
        self.strategy
    }

    /// Returns the current random stream, for inspection or snapshots.
    #[must_use]
    pub const fn stream(&self) -> SplitMix64 {
        self.stream
    }

    /// Verifies `proposal` against target probabilities `target` and draft probabilities `draft`.
    ///
    /// Greedy verification compares with the first maximum of `target`.
    /// Stochastic verification applies the rejection rule from the module
    /// documentation. Both vectors must be normalized distributions over the
    /// same vocabulary; the draft must give the proposal positive mass.
    pub fn verify(&mut self, proposal: u32, target: &[f64], draft: &[f64]) -> Result<Verdict> {
        check_distributions(proposal, target, draft)?;
        match self.strategy {
            SamplingStrategy::Greedy => Ok(Verdict::greedy(proposal, argmax(target)?)),
            SamplingStrategy::Sample => {
                if self.accept(proposal, target, draft)? {
                    Ok(Verdict::Accepted)
                } else {
                    let correction = self.sample_residual(target, draft)?;
                    Ok(Verdict::Rejected { correction })
                }
            }
        }
    }

    /// Verifies a proposal from a deterministic drafter, i.e. `q = one_hot(proposal)`.
    ///
    /// Prompt lookup and other model-free drafters choose a token with
    /// certainty. The rule then reduces to accepting with probability
    /// `p(y)` and, on rejection, sampling `p` with `y` removed. No draft
    /// vector is allocated.
    pub fn verify_deterministic(&mut self, proposal: u32, target: &[f64]) -> Result<Verdict> {
        let index = proposal as usize;
        if index >= target.len() {
            return Err(Error::Generation(format!(
                "draft token {proposal} is outside the target vocabulary of {}",
                target.len()
            )));
        }
        match self.strategy {
            SamplingStrategy::Greedy => Ok(Verdict::greedy(proposal, argmax(target)?)),
            SamplingStrategy::Sample => {
                // min(1, p(y) / 1) = p(y) for a normalized target.
                if self.stream.next_unit_f64() < target[index].min(1.0) {
                    return Ok(Verdict::Accepted);
                }
                let mut residual = target.to_vec();
                residual[index] = 0.0;
                let correction = sample_weights(&residual, self.stream.next_unit_f64())?;
                Ok(Verdict::Rejected { correction })
            }
        }
    }

    /// Draws the accept test `u < min(1, p(y) / q(y))` for a stochastic proposal.
    pub fn accept(&mut self, proposal: u32, target: &[f64], draft: &[f64]) -> Result<bool> {
        check_distributions(proposal, target, draft)?;
        let index = proposal as usize;
        let draft_mass = draft[index];
        if !(draft_mass.is_finite() && draft_mass > 0.0) {
            return Err(Error::Generation(
                "draft selected a token with zero or invalid probability".into(),
            ));
        }
        Ok(self.stream.next_unit_f64() < (target[index] / draft_mass).min(1.0))
    }

    /// Samples the normalized positive residual `max(0, p - q)` after a rejection.
    pub fn sample_residual(&mut self, target: &[f64], draft: &[f64]) -> Result<u32> {
        if target.len() != draft.len() {
            return Err(Error::Generation(
                "target and draft distributions have different vocabularies".into(),
            ));
        }
        let residual = target
            .iter()
            .zip(draft)
            .map(|(&target, &draft)| (target - draft).max(0.0))
            .collect::<Vec<_>>();
        sample_weights(&residual, self.stream.next_unit_f64())
    }
}

/// Checks that a proposal indexes two equally sized distributions.
fn check_distributions(proposal: u32, target: &[f64], draft: &[f64]) -> Result<()> {
    if proposal as usize >= target.len() || target.len() != draft.len() {
        return Err(Error::Generation(format!(
            "proposal {proposal} does not index target and draft distributions of sizes {} and {}",
            target.len(),
            draft.len()
        )));
    }
    Ok(())
}

/// Converts batch-one processed logits `[1, vocabulary]` into a host probability vector.
///
/// Uses the same backend softmax that [`super::LogitsProcessor`] samples from,
/// in at least F32, and renormalizes in F64 so the vector sums to one up to
/// rounding. Rows with NaN, `+inf`, or no finite logit are rejected.
///
/// For F32 logits the result is bit-identical to the conversion Qwen3.5 and
/// Nemotron-H used before this module existed (a softmax in the logit dtype
/// read through F32), so their seeded F32 runs are unchanged on adoption.
/// Two cases change on purpose. F16/BF16 logits are promoted to F32 before
/// the softmax instead of normalizing in half precision, matching Hugging
/// Face, which upcasts logits to F32. F64 probabilities are no longer
/// rounded through F32.
pub fn normalized_probabilities<B: Backend>(logits: Tensor<B, 2>) -> Result<Vec<f64>> {
    let [batch, vocabulary] = logits.dims();
    if batch != 1 || vocabulary == 0 {
        return Err(Error::shape(
            "speculative.logits",
            "[1, nonzero_vocabulary]",
            format!("[{batch}, {vocabulary}]"),
        ));
    }
    let dtype = probability_accumulation_dtype(logits.dtype());
    let logits = logits.cast(dtype);
    let values = metadata::tensor_f64_values(logits.clone(), "processed logits")?;
    if values.iter().any(|value| value.is_nan()) {
        return Err(Error::Generation("processed logits contain NaN".into()));
    }
    if values
        .iter()
        .any(|value| value.is_infinite() && value.is_sign_positive())
    {
        return Err(Error::Generation(
            "processed logits contain positive infinity".into(),
        ));
    }
    if !values.iter().any(|value| value.is_finite()) {
        return Err(Error::Generation(
            "processed logits assign zero mass to every token".into(),
        ));
    }
    let mut probabilities = metadata::tensor_f64_values(softmax(logits, 1), "probabilities")?;
    if probabilities
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(Error::Generation(
            "processed logits produced invalid softmax probabilities".into(),
        ));
    }
    let total = probabilities.iter().sum::<f64>();
    if !(total.is_finite() && total > 0.0) {
        return Err(Error::Generation(
            "processed logits cannot be normalized".into(),
        ));
    }
    probabilities.iter_mut().for_each(|value| *value /= total);
    Ok(probabilities)
}

/// Returns the first index of the greatest probability, matching tensor argmax ties.
pub fn argmax(probabilities: &[f64]) -> Result<u32> {
    let (index, _) = probabilities
        .iter()
        .copied()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .ok_or_else(|| Error::Generation("cannot select from an empty distribution".into()))?;
    u32::try_from(index).map_err(|_| Error::Generation("selected token does not fit u32".into()))
}

/// Samples an index from non-negative weights by inverse CDF.
///
/// With running sums `c_k`, returns the first `k` with `u * c_(V-1) < c_k`.
/// For `u` in `[0, 1)` the threshold rounds strictly below the total, and the
/// running sum only grows at positive weights, so the result always has
/// positive weight; zero-weight tokens are never returned.
///
/// # Errors
///
/// Returns [`Error::Generation`] when `uniform` is outside `[0, 1)`, a weight
/// is negative or non-finite, the weights sum to zero, or the sum of the
/// finite weights overflows `f64`.
pub fn sample_weights(weights: &[f64], uniform: f64) -> Result<u32> {
    if !(0.0..1.0).contains(&uniform) {
        return Err(Error::Generation(format!(
            "inverse-CDF uniform {uniform} is outside [0, 1)"
        )));
    }
    if weights
        .iter()
        .any(|&weight| !(weight.is_finite() && weight >= 0.0))
    {
        return Err(Error::Generation(
            "sampling weights must be finite and non-negative".into(),
        ));
    }
    // Every weight is finite and non-negative here, so the total is either
    // `+inf` (the finite weights overflowed when summed) or finite and >= 0.
    let total = weights.iter().sum::<f64>();
    if total == f64::INFINITY {
        return Err(Error::Generation(
            "sampling weights sum overflows f64".into(),
        ));
    }
    if total <= 0.0 {
        return Err(Error::Generation(
            "sampling weights have no positive probability mass".into(),
        ));
    }
    let threshold = uniform * total;
    let mut cumulative = 0.0;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if threshold < cumulative {
            return u32::try_from(index)
                .map_err(|_| Error::Generation("sampled token does not fit u32".into()));
        }
    }
    // Unreachable for `uniform < 1`, kept total for robustness: return the
    // last token with positive mass, never a zero-weight one.
    let last = weights
        .iter()
        .rposition(|&weight| weight > 0.0)
        .ok_or_else(|| Error::Generation("sampling weights lost their mass".into()))?;
    u32::try_from(last).map_err(|_| Error::Generation("sampled token does not fit u32".into()))
}

/// Proposes draft tokens by prompt lookup, without any draft model.
///
/// Tries n-gram sizes from `max_ngram` down to `min_ngram`: for the first size
/// whose trailing n-gram of `history` occurred earlier, returns up to
/// `max_draft_tokens` tokens that followed the earliest such occurrence
/// (Hugging Face `prompt_lookup_num_tokens`, vLLM's `ngram` method). Returns
/// an empty slice when nothing matches. Accepted drafts must be verified
/// with [`SpeculativeVerifier::verify_deterministic`].
#[must_use]
pub fn prompt_lookup_draft(
    history: &[u32],
    min_ngram: usize,
    max_ngram: usize,
    max_draft_tokens: usize,
) -> &[u32] {
    // Verified: the largest matching size in `max(min_ngram, 1)..=max_ngram` wins.
    match prompt_lookup_draft_span(history, min_ngram, max_ngram, max_draft_tokens) {
        Some((_, start, end)) => &history[start..end],
        None => &[],
    }
}

#[cfg(test)]
#[path = "speculative_tests.rs"]
mod tests;
