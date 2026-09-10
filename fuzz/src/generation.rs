//! Generation configuration and stopping state checked against an independent scalar oracle.

use transformers::{FinishReason, GenerationConfig, GenerationTracker};

/// Computes finish precedence independently: EOS, the longest matching stop, then length.
fn expected_finish(config: &GenerationConfig, tokens: &[u32]) -> Option<FinishReason> {
    let &last = tokens.last()?;
    if config.eos_token_ids.contains(&last) {
        return Some(FinishReason::EndOfSequence { token_id: last });
    }
    let mut selected: Option<(usize, usize)> = None;
    for (index, stop) in config.stop_sequences.iter().enumerate() {
        if tokens.ends_with(stop) && selected.is_none_or(|(_, length)| stop.len() > length) {
            selected = Some((index, stop.len()));
        }
    }
    selected
        .map(|(index, length)| FinishReason::StopSequence { index, length })
        .or_else(|| (tokens.len() >= config.max_new_tokens).then_some(FinishReason::Length))
}

/// Exercises raw JSON and all floating-point bit patterns, then bounded stop-state transitions.
pub fn exercise(input: &[u8]) {
    let input = &input[..input.len().min(crate::MAX_INPUT_BYTES)];
    if let Ok(config) = serde_json::from_slice::<GenerationConfig>(input) {
        let _ = config.validate(32);
    }
    let mut bytes = crate::Bytes::new(input);
    let vocabulary = usize::from(bytes.byte() % 32) + 1;
    let mut candidate = GenerationConfig {
        temperature: bytes.float(),
        top_k: Some(bytes.size()),
        top_p: Some(bytes.float()),
        repetition_penalty: bytes.float(),
        presence_penalty: bytes.float(),
        max_new_tokens: bytes.size(),
        ..GenerationConfig::default()
    };
    let should_validate = candidate.temperature.is_finite()
        && candidate.temperature > 0.0
        && candidate
            .top_k
            .is_some_and(|k| (1..=vocabulary).contains(&k))
        && candidate
            .top_p
            .is_some_and(|p| p.is_finite() && p > 0.0 && p <= 1.0)
        && candidate.repetition_penalty.is_finite()
        && candidate.repetition_penalty > 0.0
        && candidate.presence_penalty.is_finite()
        && candidate.max_new_tokens > 0;
    assert_eq!(candidate.validate(vocabulary).is_ok(), should_validate);
    // A second valid config keeps every input exploring the stopping state machine, even when
    // its arbitrary floating-point payload was rejected by the parameter validator.
    candidate = GenerationConfig {
        max_new_tokens: usize::from(bytes.byte() % 32) + 1,
        eos_token_ids: vec![u32::from(bytes.byte()) % vocabulary as u32],
        stop_sequences: (0..usize::from(bytes.byte() % 4))
            .map(|_| {
                (0..usize::from(bytes.byte() % 4) + 1)
                    .map(|_| u32::from(bytes.byte()) % vocabulary as u32)
                    .collect()
            })
            .collect(),
        ..GenerationConfig::default()
    };
    let batch = usize::from(bytes.byte() % 4) + 1;
    let mut tracker = GenerationTracker::new(candidate.clone(), vocabulary, batch).unwrap();
    let mut generated = vec![Vec::new(); batch];
    let mut finished = vec![None; batch];
    for _ in 0..128 {
        let sequence = usize::from(bytes.byte()) % (batch + 1);
        let token = u32::from(bytes.byte()) % (vocabulary as u32 + 1);
        let observed = tracker.push(sequence, token).map(|value| value.cloned());
        if sequence >= batch || (finished[sequence].is_none() && token as usize >= vocabulary) {
            assert!(observed.is_err());
        } else {
            if finished[sequence].is_none() {
                generated[sequence].push(token);
                finished[sequence] = expected_finish(&candidate, &generated[sequence]);
            }
            assert_eq!(observed.unwrap(), finished[sequence]);
        }
        for row in 0..batch {
            assert_eq!(tracker.generated(row).unwrap(), generated[row]);
            assert_eq!(tracker.finish_reason(row).unwrap(), finished[row].as_ref());
            assert!(generated[row].len() <= candidate.max_new_tokens);
        }
        assert_eq!(tracker.all_finished(), finished.iter().all(Option::is_some));
    }
}

#[cfg(test)]
#[path = "generation_tests.rs"]
mod tests;
