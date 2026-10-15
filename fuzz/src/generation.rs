//! Generation configuration and stopping state checked against independent scalar oracles.
//!
//! Two properties are checked for every input:
//!
//! 1. **Validation.** [`GenerationConfig::validate`] accepts a configuration
//!    exactly when [`expected_valid`], an independent restatement of every
//!    documented rule, does. That covers the sampling warpers (`temperature`,
//!    `top_k`, `top_p`, `min_p`), the three penalties, `min_new_tokens <=
//!    max_new_tokens`, special-token and stop-sequence IDs, and finite in-vocabulary
//!    `logit_bias` entries. An accepted configuration also survives a JSON round
//!    trip with the same verdict; floats may move by a few ulps there (see
//!    `crate::floats_round_trip`).
//! 2. **Stopping.** A [`GenerationTracker`] fed up to 128 tokens into a small
//!    ragged batch reports exactly the finish reasons of [`expected_finish`]:
//!    EOS first, then the longest stop suffix (earliest on ties) once the row has
//!    `min_new_tokens` tokens, then the length limit. Rejected pushes change nothing,
//!    and pushes after a finish are idempotent.

use std::collections::BTreeMap;

use transformers::{FinishReason, GenerationConfig, GenerationTracker, PenaltyScope};

use crate::Bytes;

/// Computes finish precedence independently of the production stop decision.
///
/// EOS wins; otherwise the longest matching stop sequence (earliest configured
/// on ties) counts only once `tokens.len() >= min_new_tokens`; otherwise the
/// length limit.
fn expected_finish(config: &GenerationConfig, tokens: &[u32]) -> Option<FinishReason> {
    let &last = tokens.last()?;
    if config.eos_token_ids.contains(&last) {
        return Some(FinishReason::EndOfSequence { token_id: last });
    }
    let mut selected: Option<(usize, usize)> = None;
    if tokens.len() >= config.min_new_tokens {
        for (index, stop) in config.stop_sequences.iter().enumerate() {
            if tokens.ends_with(stop) && selected.is_none_or(|(_, length)| stop.len() > length) {
                selected = Some((index, stop.len()));
            }
        }
    }
    selected
        .map(|(index, length)| FinishReason::StopSequence { index, length })
        .or_else(|| (tokens.len() >= config.max_new_tokens).then_some(FinishReason::Length))
}

/// Restates every documented [`GenerationConfig::validate`] rule for `vocabulary` tokens.
///
/// Every comparison with NaN is false, so a NaN parameter fails each check.
fn expected_valid(config: &GenerationConfig, vocabulary: usize) -> bool {
    let in_vocabulary = |token: u32| u64::from(token) < vocabulary as u64;
    let finite_positive = |value: f64| value.is_finite() && value > 0.0;
    vocabulary > 0
        && config.max_new_tokens > 0
        && config.min_new_tokens <= config.max_new_tokens
        && finite_positive(config.temperature)
        && config.top_k.is_none_or(|k| k >= 1 && k <= vocabulary)
        && config.top_p.is_none_or(|p| finite_positive(p) && p <= 1.0)
        && config.min_p.is_none_or(|p| (0.0..=1.0).contains(&p))
        && finite_positive(config.repetition_penalty)
        && config.presence_penalty.is_finite()
        && config.frequency_penalty.is_finite()
        && config
            .eos_token_ids
            .iter()
            .all(|&token| in_vocabulary(token))
        && config.pad_token_id.is_none_or(in_vocabulary)
        && config
            .stop_sequences
            .iter()
            .all(|stop| !stop.is_empty() && stop.iter().all(|&token| in_vocabulary(token)))
        && config
            .logit_bias
            .iter()
            .all(|(&token, &bias)| in_vocabulary(token) && bias.is_finite())
}

/// Decodes a token ID that is usually in the vocabulary and sometimes just outside it.
fn token(bytes: &mut Bytes<'_>, vocabulary: usize) -> u32 {
    match bytes.byte() {
        250 => u32::MAX,
        251 => u32::try_from(vocabulary).unwrap_or(u32::MAX),
        value => u32::from(value) % u32::try_from(vocabulary + 1).unwrap_or(u32::MAX),
    }
}

/// Decodes an optional value: `None` when the selector byte is even.
fn optional<'a, T>(bytes: &mut Bytes<'a>, decode: impl FnOnce(&mut Bytes<'a>) -> T) -> Option<T> {
    if bytes.flag() {
        Some(decode(bytes))
    } else {
        None
    }
}

/// Decodes a configuration that sets every field, valid or not.
///
/// Byte layout, in order: temperature, top-k, top-p, min-p, the three penalties
/// (floats via [`Bytes::interesting_float`]), penalty scope, `max_new_tokens`,
/// `min_new_tokens`, n-gram size, up to three logit-bias entries, up to two EOS
/// IDs, an optional pad ID, and up to three stop sequences of up to three tokens
/// (length zero is a deliberately invalid empty stop).
fn decode_config(bytes: &mut Bytes<'_>, vocabulary: usize) -> GenerationConfig {
    let temperature = bytes.interesting_float();
    let top_k = optional(bytes, |bytes| match bytes.byte() {
        255 => usize::MAX,
        value => usize::from(value) % (vocabulary + 2),
    });
    let top_p = optional(bytes, Bytes::interesting_float);
    let min_p = optional(bytes, Bytes::interesting_float);
    let repetition_penalty = bytes.interesting_float();
    let presence_penalty = bytes.interesting_float();
    let frequency_penalty = bytes.interesting_float();
    let additive_penalty_scope = if bytes.flag() {
        PenaltyScope::PromptAndGenerated
    } else {
        PenaltyScope::Generated
    };
    let max_new_tokens = bytes.size();
    let min_new_tokens = bytes.size();
    let no_repeat_ngram_size = bytes.size();
    let logit_bias = (0..bytes.below(4))
        .map(|_| (token(bytes, vocabulary), bytes.interesting_float()))
        .collect::<BTreeMap<_, _>>();
    let eos_token_ids = (0..bytes.below(3))
        .map(|_| token(bytes, vocabulary))
        .collect();
    let pad_token_id = optional(bytes, |bytes| token(bytes, vocabulary));
    let stop_sequences = (0..bytes.below(4))
        .map(|_| {
            (0..bytes.below(4))
                .map(|_| token(bytes, vocabulary))
                .collect()
        })
        .collect();
    GenerationConfig {
        max_new_tokens,
        min_new_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        additive_penalty_scope,
        no_repeat_ngram_size,
        logit_bias,
        eos_token_ids,
        pad_token_id,
        stop_sequences,
        ..GenerationConfig::default()
    }
}

/// Checks that an accepted configuration survives JSON serialization.
///
/// Integer, ID, and enum fields must round-trip exactly; floats approximately.
/// The round-tripped configuration must keep the validation verdict.
fn check_round_trip(config: &GenerationConfig, vocabulary: usize) {
    let encoded = serde_json::to_string(config).expect("finite configuration serializes");
    let decoded: GenerationConfig =
        serde_json::from_str(&encoded).expect("serialized configuration parses");
    let floats = |config: &GenerationConfig| {
        let mut values = vec![
            config.temperature,
            config.repetition_penalty,
            config.presence_penalty,
            config.frequency_penalty,
        ];
        values.extend(config.top_p);
        values.extend(config.min_p);
        values.extend(config.logit_bias.values());
        values
    };
    let original = floats(config);
    let restored = floats(&decoded);
    assert_eq!(original.len(), restored.len());
    for (left, right) in original.into_iter().zip(restored) {
        assert!(
            crate::floats_round_trip(left, right),
            "{left} became {right}"
        );
    }
    let without_floats = |config: &GenerationConfig| GenerationConfig {
        temperature: 1.0,
        top_p: config.top_p.map(|_| 1.0),
        min_p: config.min_p.map(|_| 0.0),
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        logit_bias: config
            .logit_bias
            .keys()
            .map(|&token| (token, 0.0))
            .collect(),
        ..config.clone()
    };
    assert_eq!(without_floats(config), without_floats(&decoded));
    assert!(decoded.validate(vocabulary).is_ok());
}

/// Exercises raw JSON, a fully decoded configuration, and bounded stop-state transitions.
///
/// The whole input is first tried as configuration JSON. Then the first byte
/// selects the vocabulary (1..=32); the configuration follows
/// ([`decode_config`]); then the tracker script of [`exercise_tracker`].
pub fn exercise(input: &[u8]) {
    let input = &input[..input.len().min(crate::MAX_INPUT_BYTES)];
    if let Ok(config) = serde_json::from_slice::<GenerationConfig>(input) {
        for vocabulary in [0, 1, 32] {
            let valid = config.validate(vocabulary).is_ok();
            assert_eq!(valid, expected_valid(&config, vocabulary), "{config:?}");
            if valid {
                check_round_trip(&config, vocabulary);
            }
        }
    }
    let mut bytes = Bytes::new(input);
    let vocabulary = usize::from(bytes.byte() % 32) + 1;
    let candidate = decode_config(&mut bytes, vocabulary);
    let valid = candidate.validate(vocabulary).is_ok();
    assert_eq!(
        valid,
        expected_valid(&candidate, vocabulary),
        "{candidate:?}"
    );
    assert_eq!(
        GenerationTracker::new(candidate.clone(), vocabulary, 1).is_ok(),
        valid
    );
    if valid {
        check_round_trip(&candidate, vocabulary);
    }
    exercise_tracker(&mut bytes, vocabulary);
}

/// Drives a valid tracker through a bounded push script against [`expected_finish`].
///
/// Byte layout: `max_new_tokens` (1..=32), `min_new_tokens` (at most the
/// maximum), one EOS ID, up to three stops of one to four tokens, the batch size
/// (1..=4), then up to 128 `(row, token)` pushes. Rows and tokens may be one
/// past their bound to exercise rejection.
fn exercise_tracker(bytes: &mut Bytes<'_>, vocabulary: usize) {
    let max_new_tokens = bytes.below(32) + 1;
    let vocabulary_id = |byte: u8| u32::from(byte) % vocabulary as u32;
    let config = GenerationConfig {
        max_new_tokens,
        min_new_tokens: bytes.below(max_new_tokens + 1),
        eos_token_ids: vec![vocabulary_id(bytes.byte())],
        stop_sequences: (0..bytes.below(4))
            .map(|_| {
                (0..bytes.below(4) + 1)
                    .map(|_| vocabulary_id(bytes.byte()))
                    .collect()
            })
            .collect(),
        ..GenerationConfig::default()
    };
    let batch = bytes.below(4) + 1;
    assert!(GenerationTracker::new(config.clone(), vocabulary, 0).is_err());
    let mut tracker = GenerationTracker::new(config.clone(), vocabulary, batch).unwrap();
    assert_eq!(tracker.batch_size(), batch);
    assert_eq!(tracker.vocabulary_size(), vocabulary);
    let mut generated = vec![Vec::new(); batch];
    let mut finished = vec![None; batch];
    for _ in 0..128 {
        let sequence = bytes.below(batch + 1);
        let token = u32::from(bytes.byte()) % (vocabulary as u32 + 1);
        let observed = tracker.push(sequence, token).map(|value| value.cloned());
        if sequence >= batch || (finished[sequence].is_none() && token as usize >= vocabulary) {
            assert!(observed.is_err());
        } else {
            if finished[sequence].is_none() {
                generated[sequence].push(token);
                finished[sequence] = expected_finish(&config, &generated[sequence]);
            }
            assert_eq!(observed.unwrap(), finished[sequence]);
        }
        for row in 0..batch {
            assert_eq!(tracker.generated(row).unwrap(), generated[row]);
            assert_eq!(tracker.finish_reason(row).unwrap(), finished[row].as_ref());
            assert!(generated[row].len() <= config.max_new_tokens);
        }
        assert!(tracker.generated(batch).is_err());
        assert!(tracker.finish_reason(batch).is_err());
        assert_eq!(tracker.all_finished(), finished.iter().all(Option::is_some));
    }
}

#[cfg(test)]
#[path = "generation_tests.rs"]
mod tests;
