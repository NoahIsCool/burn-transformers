//! Algebraic properties of logits processing that a byte-level fuzzer cannot state directly.
//!
//! The fuzz target `logits_processing` compares each processed value with a
//! scalar oracle. These properties instead relate *several* processing calls:
//!
//! - **Row independence and permutation equivariance.** Each batch row is
//!   processed on its own, so a batch equals its rows processed one at a time,
//!   and permuting rows permutes the result.
//! - **Penalty monotonicity.** With non-negative additive penalties, raising
//!   presence or frequency never raises any processed logit.
//! - **Bias additivity.** Without penalties, a logit bias `b` on an
//!   unconstrained token adds exactly `b` (in `f32`) to that token only.
//! - **Warpers keep the mode.** Temperature, top-k, top-p, and min-p never
//!   remove every most likely token, up to near-ties: top-p ranks rounded
//!   `f32` probabilities, so a logit one ulp above another can be dropped in
//!   its favor (open finding TOPP-TIE-1 in `fuzz/README.md`; Hugging Face
//!   ranks by logit). Repetition penalty `1.5` and quarter-grid offsets can
//!   produce such pairs, e.g. `1.0 / 1.5 - 0.5` and `0.25 / 1.5`.
//!
//! Proptest shrinks a failure to its smallest batch, vocabulary, and history.
//! Cases default to 64 and honor `PROPTEST_CASES`; the RNG seed is fixed.

#![cfg(feature = "flex")]

use std::collections::BTreeMap;

use burn::{
    backend::Flex,
    tensor::{Tensor, TensorData},
};
use proptest::prelude::*;
use transformers::{GenerationConfig, LogitsProcessor, PenaltyScope, SamplingStrategy};

/// Deterministic CPU backend with F32 values and I32 indices.
type Cpu = Flex<f32, i32>;

/// Relative tolerance for comparisons of separately launched `f32` reductions.
const TOLERANCE: f32 = 1e-6;

/// Relative gap below a row maximum within which a token still counts as a mode.
///
/// Covers a few `f32` ulps, the near-ties that top-p may split (TOPP-TIE-1).
/// Set it to zero once `filter_top_p` ranks by logit.
const NEAR_TIE: f32 = 1e-6;

/// Number of cases: `PROPTEST_CASES` when set, else `default`.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// One processing problem: rows with prompt boundaries and quarter-grid logits.
#[derive(Debug, Clone)]
struct Batch {
    /// Vocabulary size.
    vocabulary: usize,
    /// Prompt followed by generated tokens, per row.
    rows: Vec<Vec<u32>>,
    /// Prompt length per row.
    prompts: Vec<usize>,
    /// Row-major `[batch, vocabulary]` logits.
    logits: Vec<f32>,
}

impl Batch {
    /// Number of rows.
    fn batch(&self) -> usize {
        self.rows.len()
    }

    /// Row `index` as a one-row batch.
    fn row(&self, index: usize) -> Self {
        Self {
            vocabulary: self.vocabulary,
            rows: vec![self.rows[index].clone()],
            prompts: vec![self.prompts[index]],
            logits: self.logits[index * self.vocabulary..(index + 1) * self.vocabulary].to_vec(),
        }
    }

    /// The batch with its rows in reverse order.
    fn reversed(&self) -> Self {
        let mut reversed = Self {
            vocabulary: self.vocabulary,
            rows: Vec::new(),
            prompts: Vec::new(),
            logits: Vec::new(),
        };
        for index in (0..self.batch()).rev() {
            let row = self.row(index);
            reversed.rows.extend(row.rows);
            reversed.prompts.extend(row.prompts);
            reversed.logits.extend(row.logits);
        }
        reversed
    }
}

/// Strategy for a batch of one to three rows over a vocabulary of two to eight tokens.
fn batches() -> impl Strategy<Value = Batch> {
    (2usize..=8, 1usize..=3).prop_flat_map(|(vocabulary, batch)| {
        let token = 0..u32::try_from(vocabulary).unwrap();
        let row = proptest::collection::vec(token, 0..=6).prop_flat_map(|row| {
            let length = row.len();
            (Just(row), 0..=length)
        });
        (
            proptest::collection::vec(row, batch),
            proptest::collection::vec(-16i8..=16, batch * vocabulary),
        )
            .prop_map(move |(rows, logits)| {
                let (rows, prompts) = rows.into_iter().unzip();
                Batch {
                    vocabulary,
                    rows,
                    prompts,
                    logits: logits
                        .into_iter()
                        .map(|value| f32::from(value) / 4.0)
                        .collect(),
                }
            })
    })
}

/// Strategy for a moderate configuration over `vocabulary` tokens.
fn configs(vocabulary: usize) -> impl Strategy<Value = GenerationConfig> {
    let token = 0..u32::try_from(vocabulary).unwrap();
    (
        (any::<bool>(), prop::sample::select(vec![0.5, 1.0, 2.0])),
        (
            proptest::option::of(1..=vocabulary),
            proptest::option::of(prop::sample::select(vec![0.3, 0.9, 1.0])),
            proptest::option::of(prop::sample::select(vec![0.0, 0.1, 0.5])),
        ),
        (
            prop::sample::select(vec![1.0, 0.5, 1.5]),
            prop::sample::select(vec![0.0, 0.5, 2.0]),
            prop::sample::select(vec![0.0, 0.25, 1.0]),
            any::<bool>(),
        ),
        (0usize..=3, 0usize..=3),
        proptest::collection::btree_map(token.clone(), -2i8..=2, 0..=2),
        proptest::collection::vec(token, 0..=2),
    )
        .prop_map(
            |(
                (sample, temperature),
                (top_k, top_p, min_p),
                (repetition_penalty, presence_penalty, frequency_penalty, prompt_scope),
                (no_repeat_ngram_size, min_new_tokens),
                bias,
                eos_token_ids,
            )| GenerationConfig {
                strategy: if sample {
                    SamplingStrategy::Sample
                } else {
                    SamplingStrategy::Greedy
                },
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                presence_penalty,
                frequency_penalty,
                additive_penalty_scope: if prompt_scope {
                    PenaltyScope::PromptAndGenerated
                } else {
                    PenaltyScope::Generated
                },
                no_repeat_ngram_size,
                min_new_tokens,
                max_new_tokens: 8,
                logit_bias: bias
                    .into_iter()
                    .map(|(token, bias)| (token, f64::from(bias) / 2.0))
                    .collect::<BTreeMap<_, _>>(),
                eos_token_ids,
                ..GenerationConfig::default()
            },
        )
}

/// A batch together with a configuration for its vocabulary.
fn problems() -> impl Strategy<Value = (Batch, GenerationConfig)> {
    batches().prop_flat_map(|batch| {
        let vocabulary = batch.vocabulary;
        (Just(batch), configs(vocabulary))
    })
}

/// Processes `batch` under `config` through the stateless host-row entry point.
fn process(batch: &Batch, config: &GenerationConfig) -> Vec<f32> {
    let device = Default::default();
    let processor = LogitsProcessor::<Cpu>::new(config.clone(), batch.vocabulary, &device)
        .expect("strategies produce valid configurations");
    let logits = Tensor::<Cpu, 2>::from_data(
        TensorData::new(batch.logits.clone(), [batch.batch(), batch.vocabulary]),
        &device,
    );
    let rows: Vec<&[u32]> = batch.rows.iter().map(Vec::as_slice).collect();
    processor
        .process_with_prompt_lengths(logits, &rows, &batch.prompts)
        .expect("valid rows")
        .into_data()
        .to_vec::<f32>()
        .expect("F32 logits")
}

/// Whether two processed values agree: both `-inf`, or within [`TOLERANCE`].
fn agrees(left: f32, right: f32) -> bool {
    left == right || (left - right).abs() <= TOLERANCE * left.abs().max(1.0)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: cases(64),
        rng_seed: proptest::test_runner::RngSeed::Fixed(20_261_015),
        ..ProptestConfig::default()
    })]

    /// A batch equals its rows processed one at a time, and reversing rows reverses it.
    #[test]
    fn rows_are_processed_independently_and_equivariantly((batch, config) in problems()) {
        let together = process(&batch, &config);
        let vocabulary = batch.vocabulary;
        for index in 0..batch.batch() {
            let alone = process(&batch.row(index), &config);
            let span = &together[index * vocabulary..(index + 1) * vocabulary];
            for (&left, &right) in span.iter().zip(&alone) {
                prop_assert!(agrees(left, right), "row {index}: {left} vs {right}");
            }
        }
        let reversed = process(&batch.reversed(), &config);
        let rows = batch.batch();
        for index in 0..rows {
            let original = &together[index * vocabulary..(index + 1) * vocabulary];
            let mirrored = &reversed[(rows - 1 - index) * vocabulary..(rows - index) * vocabulary];
            for (&left, &right) in original.iter().zip(mirrored) {
                prop_assert!(agrees(left, right));
            }
        }
    }

    /// Raising non-negative presence or frequency penalties never raises a greedy logit.
    #[test]
    fn additive_penalties_are_monotone(
        (batch, config) in problems(),
        extra_presence in 0.0f64..2.0,
        extra_frequency in 0.0f64..2.0,
    ) {
        let base = GenerationConfig { strategy: SamplingStrategy::Greedy, ..config };
        let stronger = GenerationConfig {
            presence_penalty: base.presence_penalty + extra_presence,
            frequency_penalty: base.frequency_penalty + extra_frequency,
            ..base.clone()
        };
        let weak = process(&batch, &base);
        let strong = process(&batch, &stronger);
        for (&before, &after) in weak.iter().zip(&strong) {
            if before == f32::NEG_INFINITY {
                prop_assert_eq!(after, f32::NEG_INFINITY);
            } else {
                prop_assert!(after <= before + TOLERANCE * before.abs().max(1.0), "{before} rose to {after}");
            }
        }
    }

    /// Without penalties, a bias on an unconstrained token adds exactly that bias.
    #[test]
    fn logit_bias_is_additive_on_unconstrained_tokens(
        batch in batches(),
        token_seed in any::<prop::sample::Index>(),
        bias in -8i8..=8,
    ) {
        let token = u32::try_from(token_seed.index(batch.vocabulary)).unwrap();
        let bias = f64::from(bias) / 4.0;
        let plain = GenerationConfig::default();
        let biased = GenerationConfig {
            logit_bias: BTreeMap::from([(token, bias)]),
            ..plain.clone()
        };
        let without = process(&batch, &plain);
        let with = process(&batch, &biased);
        for (index, (&before, &after)) in without.iter().zip(&with).enumerate() {
            if index % batch.vocabulary == token as usize {
                prop_assert_eq!(after, before + bias as f32);
            } else {
                prop_assert_eq!(after.to_bits(), before.to_bits());
            }
        }
    }

    /// Sampling warpers keep at least one most likely token (up to [`NEAR_TIE`]) of every row.
    #[test]
    fn warpers_keep_a_most_likely_token((batch, config) in problems()) {
        let sampled_config = GenerationConfig { strategy: SamplingStrategy::Sample, ..config };
        let greedy_config = GenerationConfig {
            strategy: SamplingStrategy::Greedy,
            ..sampled_config.clone()
        };
        let greedy = process(&batch, &greedy_config);
        let sampled = process(&batch, &sampled_config);
        for row in 0..batch.batch() {
            let span = row * batch.vocabulary..(row + 1) * batch.vocabulary;
            let maximum = greedy[span.clone()].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if maximum == f32::NEG_INFINITY {
                continue;
            }
            let mode = maximum - NEAR_TIE * maximum.abs().max(1.0);
            let kept = span
                .clone()
                .any(|index| greedy[index] >= mode && sampled[index].is_finite());
            prop_assert!(kept, "row {row} lost every mode: {:?}", &sampled[span]);
        }
    }
}
