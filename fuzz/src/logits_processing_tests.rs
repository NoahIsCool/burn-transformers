//! Deterministic regression seeds for the logits-processing fuzz harness.

use super::*;

/// Builds a moderate case directly, bypassing the byte decoder.
fn case(config: GenerationConfig, vocabulary: usize, rows: &[&[u32]], prompts: &[usize]) -> Case {
    let logits = (0..rows.len() * vocabulary)
        .map(|index| f32::from(u8::try_from(index % 9).unwrap()) / 2.0 - 2.0)
        .collect();
    Case {
        vocabulary,
        config,
        rows: rows.iter().map(|row| row.to_vec()).collect(),
        prompts: prompts.to_vec(),
        logits,
        extreme: false,
    }
}

/// Processes one row of `case` and returns its values.
fn processed(case: &Case) -> Vec<f32> {
    let processor =
        LogitsProcessor::<Backend>::new(case.config.clone(), case.vocabulary, &Default::default())
            .unwrap();
    host(
        processor
            .process_with_prompt_lengths(case.logits_tensor(), &case.row_slices(), &case.prompts)
            .unwrap(),
    )
}

/// Every control at once over a ragged batch, through every path, greedy and sampled.
#[test]
fn ragged_rows_with_every_control_match_the_oracle() {
    let config = GenerationConfig {
        min_new_tokens: 3,
        eos_token_ids: vec![4],
        no_repeat_ngram_size: 2,
        repetition_penalty: 1.25,
        presence_penalty: 0.5,
        frequency_penalty: 0.25,
        logit_bias: BTreeMap::from([(1, 0.75), (4, 2.0)]),
        ..GenerationConfig::default()
    };
    let rows: [&[u32]; 3] = [&[1, 2, 1, 2, 1], &[], &[3, 3, 0]];
    for scope in [PenaltyScope::Generated, PenaltyScope::PromptAndGenerated] {
        for strategy in [SamplingStrategy::Greedy, SamplingStrategy::Sample] {
            let config = GenerationConfig {
                additive_penalty_scope: scope,
                strategy,
                top_k: Some(4),
                top_p: Some(0.9),
                min_p: Some(0.05),
                temperature: 0.5,
                ..config.clone()
            };
            check_case(&case(config, 5, &rows, &[2, 0, 3]));
        }
    }
}

/// The n-gram ban and EOS suppression produce `-inf` exactly where the oracle says.
#[test]
fn hard_constraints_ban_exactly_the_predicted_tokens() {
    let config = GenerationConfig {
        min_new_tokens: 2,
        eos_token_ids: vec![0],
        no_repeat_ngram_size: 3,
        ..GenerationConfig::default()
    };
    // Row `1 2 3 1 2`: the 2-token suffix `1 2` was followed by 3 before.
    let banned = case(config.clone(), 4, &[&[1, 2, 3, 1, 2]], &[4]);
    let values = processed(&banned);
    assert_eq!(values[3], f32::NEG_INFINITY);
    // One generated token is below the minimum of two, so EOS is masked too.
    assert_eq!(values[0], f32::NEG_INFINITY);
    assert!(values[1].is_finite() && values[2].is_finite());
    check_case(&banned);
    // With two generated tokens EOS is allowed again.
    let allowed = case(config, 4, &[&[1, 2, 3, 1, 2]], &[3]);
    assert!(processed(&allowed)[0].is_finite());
}

/// A unigram constraint bans every token already seen, including prompt tokens.
#[test]
fn unigram_constraint_bans_every_seen_token() {
    let config = GenerationConfig {
        no_repeat_ngram_size: 1,
        ..GenerationConfig::default()
    };
    let unigram = case(config, 3, &[&[2, 0]], &[1]);
    let values = processed(&unigram);
    assert_eq!(values[0], f32::NEG_INFINITY);
    assert_eq!(values[2], f32::NEG_INFINITY);
    assert!(values[1].is_finite());
    check_case(&unigram);
}

/// Out-of-vocabulary and overlong-prompt rows are refused on every strict path.
#[test]
fn invalid_rows_are_rejected_by_each_path() {
    let counting = GenerationConfig {
        repetition_penalty: 2.0,
        ..GenerationConfig::default()
    };
    // Out of vocabulary in the prompt: strict paths reject; the tensor path
    // rejects because a penalty reads values.
    check_case(&case(counting.clone(), 3, &[&[3, 0]], &[1]));
    // The same prompt with no history-dependent control: the tensor path
    // never reads values and accepts it, as documented.
    let neutral = case(GenerationConfig::default(), 3, &[&[3, 0]], &[1]);
    assert!(!neutral.rows_are_valid());
    assert!(incremental_history(&neutral).is_ok());
    check_case(&neutral);
    // Out of vocabulary among generated tokens: `record` always rejects.
    let generated = case(GenerationConfig::default(), 3, &[&[0, u32::MAX]], &[1]);
    assert!(incremental_history(&generated).is_err());
    check_case(&generated);
    // A prompt longer than its row.
    check_case(&case(counting, 3, &[&[0], &[1, 2]], &[2, 0]));
}

/// All-banned rows stay `-inf` through the warpers instead of turning into NaN.
#[test]
fn fully_banned_rows_stay_negative_infinity() {
    let config = GenerationConfig {
        strategy: SamplingStrategy::Sample,
        no_repeat_ngram_size: 1,
        top_p: Some(0.5),
        min_p: Some(0.5),
        ..GenerationConfig::default()
    };
    let banned = case(config, 2, &[&[0, 1], &[0]], &[0, 0]);
    let values = processed(&banned);
    assert!(values[..2].iter().all(|&value| value == f32::NEG_INFINITY));
    check_case(&banned);
}

/// Seeds in `fuzz/seeds/logits_processing` keep their documented meaning.
#[test]
fn committed_seeds_pass() {
    for seed in [
        &include_bytes!("../seeds/logits_processing/ragged-sampling")[..],
        &include_bytes!("../seeds/logits_processing/out-of-vocabulary")[..],
        &include_bytes!("../seeds/logits_processing/extreme-parameters")[..],
    ] {
        exercise(seed);
    }
    exercise(&[]);
    exercise(&[255; 256]);
}

/// Pseudo-random inputs cover many configuration, history, and logits mixes.
#[test]
fn pseudo_random_inputs_agree_with_the_oracles() {
    for input in crate::lcg_inputs(0x6c6f_6769, 384, 96) {
        exercise(&input);
    }
}

/// Builds a one-row processor over logits `[0, 1, -1, 2]` with history `[1]`.
fn single_row(config: GenerationConfig) -> Vec<f32> {
    let row: &[u32] = &[1];
    let mut overflow = case(config, 4, &[row], &[0]);
    overflow.logits = vec![0.0, 1.0, -1.0, 2.0];
    processed(&overflow)
}

/// The message of a generation error, or a panic naming the unexpected variant.
fn generation_message(error: transformers::Error) -> String {
    match error {
        transformers::Error::Generation(message) => message,
        other => panic!("expected a generation error, got {other:?}"),
    }
}

/// GEN-F32-1 (fixed): a finite additive penalty beyond `f32::MAX` turned unseen tokens into NaN.
///
/// `frequency_penalty = 1e39` validated, but the processor multiplied the
/// `f32` counts by the penalty rounded to `f32` (`inf`): `0 * inf = NaN` for
/// every unseen token. Validation now rejects additive penalties whose `f32`
/// rounding is infinite, naming the field and the range; the largest finite
/// penalties are accepted and process without NaN.
#[test]
fn finite_additive_penalties_never_produce_nan() {
    for (config, field) in [
        (
            GenerationConfig {
                frequency_penalty: 1e39,
                ..GenerationConfig::default()
            },
            "frequency_penalty",
        ),
        (
            GenerationConfig {
                presence_penalty: -1e39,
                ..GenerationConfig::default()
            },
            "presence_penalty",
        ),
    ] {
        let message = generation_message(config.validate(4).unwrap_err());
        assert!(message.starts_with(field), "{message}");
        assert!(message.contains("3.4028235e38"), "{message}");
    }
    let largest = f64::from(f32::MAX);
    for config in [
        GenerationConfig {
            frequency_penalty: largest,
            ..GenerationConfig::default()
        },
        GenerationConfig {
            presence_penalty: -largest,
            ..GenerationConfig::default()
        },
    ] {
        assert!(config.validate(4).is_ok());
        assert!(single_row(config).iter().all(|value| !value.is_nan()));
    }
}

/// GEN-F32-2 (fixed): a positive repetition penalty below the `f32` range divided zero by zero.
///
/// `repetition_penalty = 1e-46` validated but rounded to `0.0_f32`; a seen
/// token with logit `0` became `0 / 0 = NaN`. Divisors must now round to
/// normal positive `f32` values; the smallest one is accepted, and the
/// quotients it overflows saturate instead of producing NaN or `+inf`.
#[test]
fn positive_repetition_penalty_never_produces_nan() {
    let underflow = GenerationConfig {
        repetition_penalty: 1e-46,
        ..GenerationConfig::default()
    };
    let message = generation_message(underflow.validate(4).unwrap_err());
    assert!(message.starts_with("repetition_penalty"), "{message}");
    let row: &[u32] = &[0, 1];
    let mut smallest = case(
        GenerationConfig {
            repetition_penalty: f64::from(f32::MIN_POSITIVE),
            ..GenerationConfig::default()
        },
        4,
        &[row],
        &[0],
    );
    smallest.logits = vec![0.0, 1.0, -1.0, 2.0];
    assert!(
        processed(&smallest)
            .iter()
            .all(|value| value.is_finite() || *value == f32::NEG_INFINITY)
    );
}

/// GEN-F32-3 (fixed): a huge finite logit bias on a constrained token added `inf + -inf`.
///
/// Bias and the `-inf` hard constraints shared one scatter-add, so a bias of
/// `1e39` (infinite in `f32`) on a token banned by the n-gram constraint
/// yielded NaN instead of `-inf`. Biases must now round to finite `f32`
/// values, positive overflow saturates before the constraints run, and the
/// constraints come last, so a banned token is exactly `-inf` under the
/// largest valid bias.
#[test]
fn constrained_tokens_stay_negative_infinity_under_any_valid_bias() {
    let rejected = GenerationConfig {
        logit_bias: BTreeMap::from([(1, 1e39)]),
        no_repeat_ngram_size: 1,
        ..GenerationConfig::default()
    };
    let message = generation_message(rejected.validate(4).unwrap_err());
    assert!(message.starts_with("logit_bias for token 1"), "{message}");
    let config = GenerationConfig {
        logit_bias: BTreeMap::from([(1, f64::from(f32::MAX))]),
        no_repeat_ngram_size: 1,
        ..GenerationConfig::default()
    };
    assert!(config.validate(4).is_ok());
    assert_eq!(single_row(config)[1], f32::NEG_INFINITY);
}

/// GEN-F32-4 (fixed): a positive temperature below the `f32` range divided logits by zero.
///
/// `temperature = 1e-46` validated but rounded to `0.0_f32`, so sampling
/// processed `0 / 0 = NaN` and `+-x / 0 = +-inf`. The temperature must now
/// round to a normal positive `f32`; the smallest such temperature is
/// accepted and divides these logits exactly (`2 / 2^-126 = 2^127`).
#[test]
fn positive_temperature_never_produces_nan() {
    let rejected = GenerationConfig {
        strategy: SamplingStrategy::Sample,
        temperature: 1e-46,
        ..GenerationConfig::default()
    };
    let message = generation_message(rejected.validate(4).unwrap_err());
    assert!(message.starts_with("temperature"), "{message}");
    let config = GenerationConfig {
        temperature: f64::from(f32::MIN_POSITIVE),
        ..rejected
    };
    assert!(config.validate(4).is_ok());
    let values = single_row(config);
    assert!(values.iter().all(|value| value.is_finite()), "{values:?}");
    assert_eq!(values[3], 2.0 / f32::MIN_POSITIVE);
}

/// Campaign regression `crash-cfb301be` (minimized): top-p may split logits one
/// `f32` ulp apart because their rounded probabilities tie (open finding
/// TOPP-TIE-1; the harness tolerates it until fixed).
#[test]
fn campaign_top_p_near_ties_may_split() {
    exercise(&[
        0x00, 0x04, 0x02, 0x01, 0x01, 0x01, 0x03, 0x01, 0x03, 0x01, 0x01, 0x02, 0x01, 0x16, 0x00,
        0x00, 0x00, 0x00, 0x03, 0x03, 0x03, 0xb6, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16,
        0xd8, 0xd8,
    ]);
}

/// TOPP-TIE-1: top-p must never drop a larger logit while keeping a smaller one.
///
/// Hugging Face 5.2 `TopPLogitsWarper` sorts the *logits* (`torch.sort(scores)`)
/// and only then takes the softmax and cumulative sum of the sorted values,
/// so a strictly larger logit always ranks ahead of a smaller one. The
/// library's `filter_top_p` (`src/generation.rs`) sorts rounded `f32`
/// *probabilities*: logits one ulp apart, such as `0.1` and `0.10000001`,
/// get the same probability, the sort keeps their original order, and the
/// larger logit is dropped (campaign crash `crash-cfb301be`). Fix: sort the
/// logits descending and compute softmax and cumulative mass over the sorted
/// logits, as Hugging Face does; then remove the near-tie slack in
/// `check_warpers` and the property suite.
#[test]
#[ignore = "TOPP-TIE-1: open follow-up in src/generation.rs (top-p ranks rounded probabilities)"]
fn top_p_ranks_by_logit_like_hugging_face() {
    for (smaller, larger) in [(0.1_f32, 0.100_000_01_f32), (-0.166_666_67, -0.166_666_66)] {
        assert!(smaller < larger);
        for top_p in [0.3, 0.5] {
            let config = GenerationConfig {
                strategy: SamplingStrategy::Sample,
                top_p: Some(top_p),
                ..GenerationConfig::default()
            };
            let row: &[u32] = &[];
            let mut tie = case(config, 2, &[row], &[0]);
            tie.logits = vec![smaller, larger];
            let values = processed(&tie);
            assert!(
                values[1].is_finite(),
                "top_p {top_p} dropped {larger} but kept {smaller}: {values:?}"
            );
        }
    }
}
