//! Pure protocol and output checks for the separate public causal-generation benchmark.
//!
//! A continuation of `N` tokens requires one prefill selection and `N - 1` cached
//! decode calls. Its end-to-end latency includes public coordinator bookkeeping,
//! cache allocation/destruction, processed log probabilities, and stopping logic;
//! it is not interchangeable with model-only forced-decode throughput. This fixed
//! oracle reaches the length limit, so an early EOS or stop is a correctness failure,
//! not an opportunity to report a shorter elapsed time for different work.

use super::benchmark_metrics::{MEASURED_REPETITIONS, WARMUP_REPETITIONS, token_hash};
use serde::Serialize;
use serde_json::{Value, json};
use transformers::{CausalGenerationOutput, FinishReason, GenerationConfig, SamplingStrategy};

/// Distinct series identity; historical model-API timings must not be relabeled with it.
pub const MODE: &str = "greedy_public_generate_causal_coordinator";
/// Complete natural continuation authenticated by the existing official oracle.
pub const MAX_NEW_TOKENS: usize = 32;

/// Selects reproducible greedy controls while retaining the checkpoint's EOS/PAD defaults.
pub fn generation_config(
    defaults: &GenerationConfig,
    vocabulary_size: usize,
) -> Result<GenerationConfig, String> {
    let config = GenerationConfig {
        strategy: SamplingStrategy::Greedy,
        max_new_tokens: MAX_NEW_TOKENS,
        seed: 0,
        ..defaults.clone()
    };
    config
        .validate(vocabulary_size)
        .map_err(|error| error.to_string())?;
    if config.eos_token_ids.is_empty()
        || config.repetition_penalty != 1.0
        || config.presence_penalty != 0.0
        || !config.stop_sequences.is_empty()
    {
        return Err(
            "the pinned greedy oracle requires EOS and neutral penalties/no extra stops".into(),
        );
    }
    Ok(config)
}

/// Counts prompt and actually consumed feedback tokens with checked, bounded arithmetic.
pub fn cache_capacity(prompt_length: usize) -> Result<usize, String> {
    let capacity = prompt_length
        .checked_add(MAX_NEW_TOKENS - 1)
        .ok_or("public generation capacity overflow")?;
    if prompt_length == 0 || capacity > super::MAX_CONTEXT {
        return Err(
            "public generation must fit a nonempty prompt and continuation within 32K".into(),
        );
    }
    Ok(capacity)
}

/// Describes exactly the work enclosed by each public API timer, without inventing sub-timings.
pub fn protocol(config: &GenerationConfig, prompt_length: usize) -> Result<Value, String> {
    if config.strategy != SamplingStrategy::Greedy
        || config.seed != 0
        || config.max_new_tokens != MAX_NEW_TOKENS
    {
        return Err("public benchmark generation controls differ from the fixed protocol".into());
    }
    Ok(json!({
        "mode": MODE,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "batch_size": 1,
        "valid_source_tokens": prompt_length,
        "padding": "none",
        "cache_capacity": cache_capacity(prompt_length)?,
        "cache_window": null,
        "maximum_total_context": super::MAX_CONTEXT,
        "generation": config,
        "prefill_chunk_size": null,
        "expected_generated_tokens": MAX_NEW_TOKENS,
        "expected_cached_decode_calls": MAX_NEW_TOKENS - 1,
        "forced_decode_steps": 0,
        "seed": 0,
        "tokenization_included": false,
        "prompt_tensor_construction_included": false,
        "cache_allocation_and_destruction_included": true,
        "host_selection_log_probabilities_and_stopping_included": true,
        "streaming_callback": false,
        "reference_validation_included": false,
        "synchronization": "Backend::sync immediately before generate_causal entry and after successful return; end-to-end wall time only",
        "comparison_scope": "separate public-coordinator series; not the historical model-API loop or forced-decode samples"
    }))
}

/// Complete canonical output identity and terminal metadata from one validated repetition.
#[derive(Debug, Clone, Serialize)]
pub struct CanonicalOutput {
    /// Number of valid prompt tokens reported by the coordinator.
    pub prompt_length: usize,
    /// Actual generated length, including an EOS/stop suffix if one were emitted.
    pub generated_length: usize,
    /// Every generated token, in order, with no decoded-text normalization.
    pub token_ids: Vec<u32>,
    /// SHA-256 of the canonical concatenated little-endian u32 token IDs.
    pub tokens_u32le_sha256: String,
    /// Processed log probabilities returned by the public coordinator, not recomputed externally.
    pub log_probabilities: Vec<f64>,
    /// Actual coordinator termination reason; this fixed oracle requires `Length`.
    pub finish_reason: FinishReason,
}

/// Rejects shortened, altered, malformed or nonfinite outputs before accepting a timing sample.
pub fn validate_output(
    output: &CausalGenerationOutput,
    prompt_length: usize,
    expected_ids: &[u32],
) -> Result<CanonicalOutput, String> {
    cache_capacity(prompt_length)?;
    if expected_ids.len() != MAX_NEW_TOKENS {
        return Err(
            "the public benchmark requires the complete authenticated 32-token oracle".into(),
        );
    }
    let [sequence] = output.sequences.as_slice() else {
        return Err("the public benchmark requires exactly one generated sequence".into());
    };
    if sequence.prompt_length != prompt_length
        || sequence.token_ids != expected_ids
        || sequence.finish_reason != FinishReason::Length
    {
        return Err(
            "public generation differs from the authenticated length-terminated oracle".into(),
        );
    }
    if sequence.log_probabilities.len() != MAX_NEW_TOKENS
        || sequence
            .log_probabilities
            .iter()
            .any(|value| !value.is_finite() || *value > 0.0)
    {
        return Err(
            "public generation requires one finite nonpositive log probability per token".into(),
        );
    }
    Ok(CanonicalOutput {
        prompt_length,
        generated_length: sequence.token_ids.len(),
        token_ids: sequence.token_ids.clone(),
        tokens_u32le_sha256: token_hash(&sequence.token_ids),
        log_probabilities: sequence.log_probabilities.clone(),
        finish_reason: sequence.finish_reason.clone(),
    })
}

#[cfg(test)]
mod tests {
    //! Tests for fixed greedy controls and complete public-generation benchmark metadata.

    use super::*;
    use transformers::GeneratedSequence;

    /// Supplies a small synthetic output to exercise metadata validation without model allocation.
    fn output() -> CausalGenerationOutput {
        CausalGenerationOutput {
            sequences: vec![GeneratedSequence {
                prompt_length: 23,
                token_ids: (0..MAX_NEW_TOKENS as u32).collect(),
                log_probabilities: vec![-0.5; MAX_NEW_TOKENS],
                finish_reason: FinishReason::Length,
            }],
        }
    }

    /// Greedy normalization is deterministic but does not erase checkpoint EOS/PAD metadata.
    #[test]
    fn controls_preserve_stopping_defaults_and_pin_seed() {
        let defaults = GenerationConfig {
            strategy: SamplingStrategy::Sample,
            seed: 99,
            max_new_tokens: 128,
            eos_token_ids: vec![61, 62],
            pad_token_id: Some(63),
            ..GenerationConfig::default()
        };
        let actual = generation_config(&defaults, 64).unwrap();
        assert_eq!(actual.strategy, SamplingStrategy::Greedy);
        assert_eq!(actual.seed, 0);
        assert_eq!(actual.max_new_tokens, 32);
        assert_eq!(actual.eos_token_ids, [61, 62]);
        assert_eq!(actual.pad_token_id, Some(63));
        assert_eq!(defaults.seed, 99);
        assert!(generation_config(&defaults, 63).is_err());
        assert!(generation_config(&GenerationConfig::default(), 64).is_err());
        for invalid in [
            GenerationConfig {
                repetition_penalty: 1.1,
                ..actual.clone()
            },
            GenerationConfig {
                presence_penalty: 0.1,
                ..actual.clone()
            },
            GenerationConfig {
                stop_sequences: vec![vec![2]],
                ..actual.clone()
            },
            GenerationConfig {
                temperature: f64::NAN,
                ..actual
            },
        ] {
            assert!(generation_config(&invalid, 64).is_err());
        }
    }

    /// The last predicted token is never fed back after the length limit is reached.
    #[test]
    fn cache_bounds_count_31_feedback_tokens_not_128_forced_steps() {
        assert_eq!(cache_capacity(23).unwrap(), 54);
        assert_eq!(
            cache_capacity(super::super::MAX_CONTEXT - 31).unwrap(),
            32_768
        );
        for invalid in [0, super::super::MAX_CONTEXT - 30, usize::MAX] {
            assert!(cache_capacity(invalid).is_err());
        }
    }

    /// Public end-to-end series cannot silently claim historical model-only timing boundaries.
    #[test]
    fn protocol_separates_coordinator_work_and_rejects_drift() {
        let config = generation_config(
            &GenerationConfig {
                eos_token_ids: vec![63],
                ..GenerationConfig::default()
            },
            64,
        )
        .unwrap();
        let actual = protocol(&config, 23).unwrap();
        assert_eq!(actual["mode"], MODE);
        assert_ne!(actual["mode"], "greedy_model_api_loop");
        assert_eq!(actual["warmup_repetitions"], 3);
        assert_eq!(actual["measured_repetitions"], 10);
        assert_eq!(actual["cache_capacity"], 54);
        assert_eq!(actual["expected_cached_decode_calls"], 31);
        assert_eq!(actual["forced_decode_steps"], 0);
        assert_eq!(actual["tokenization_included"], false);
        assert_eq!(actual["cache_allocation_and_destruction_included"], true);
        for invalid in [
            GenerationConfig {
                seed: 1,
                ..config.clone()
            },
            GenerationConfig {
                max_new_tokens: 31,
                ..config.clone()
            },
            GenerationConfig {
                strategy: SamplingStrategy::Sample,
                ..config.clone()
            },
        ] {
            assert!(protocol(&invalid, 23).is_err());
        }
        assert!(protocol(&config, 0).is_err());
    }

    /// Every accepted sample retains the full canonical tokens, actual length and finish reason.
    #[test]
    fn complete_output_evidence_keeps_actual_coordinator_metadata() {
        let output = output();
        let ids = &output.sequences[0].token_ids;
        let actual = validate_output(&output, 23, ids).unwrap();
        assert_eq!(actual.prompt_length, 23);
        assert_eq!(actual.generated_length, 32);
        assert_eq!(actual.token_ids, *ids);
        assert_eq!(actual.tokens_u32le_sha256, token_hash(ids));
        assert_eq!(actual.log_probabilities, vec![-0.5; 32]);
        assert_eq!(actual.finish_reason, FinishReason::Length);
        assert_eq!(
            serde_json::to_value(actual).unwrap()["finish_reason"],
            "length"
        );
    }

    /// Token equality alone cannot hide incorrect row counts, prompt lengths or early termination.
    #[test]
    fn altered_tokens_lengths_and_terminal_metadata_fail_closed() {
        let valid = output();
        let ids = &valid.sequences[0].token_ids;
        assert!(validate_output(&valid, 0, ids).is_err());
        assert!(validate_output(&valid, 23, &ids[..31]).is_err());
        assert!(validate_output(&CausalGenerationOutput { sequences: vec![] }, 23, ids).is_err());
        let mut invalid = valid.clone();
        invalid.sequences.push(invalid.sequences[0].clone());
        assert!(validate_output(&invalid, 23, ids).is_err());
        invalid = valid.clone();
        invalid.sequences[0].prompt_length = 22;
        assert!(validate_output(&invalid, 23, ids).is_err());
        invalid = valid.clone();
        invalid.sequences[0].token_ids[31] += 1;
        assert!(validate_output(&invalid, 23, ids).is_err());
        invalid = valid.clone();
        invalid.sequences[0].token_ids.pop();
        assert!(validate_output(&invalid, 23, ids).is_err());
        for reason in [
            FinishReason::EndOfSequence { token_id: 31 },
            FinishReason::StopSequence {
                index: 0,
                length: 1,
            },
        ] {
            invalid = valid.clone();
            invalid.sequences[0].finish_reason = reason;
            assert!(validate_output(&invalid, 23, ids).is_err());
        }
    }

    /// Log-probability readback must be complete and finite, even when greedy tokens agree.
    #[test]
    fn incomplete_or_nonfinite_probability_readbacks_are_rejected() {
        let valid = output();
        let ids = &valid.sequences[0].token_ids;
        for length in [0, 31, 33] {
            let mut invalid = valid.clone();
            invalid.sequences[0].log_probabilities.resize(length, -0.5);
            assert!(validate_output(&invalid, 23, ids).is_err());
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.1] {
            let mut invalid = valid.clone();
            invalid.sequences[0].log_probabilities[31] = value;
            assert!(validate_output(&invalid, 23, ids).is_err());
        }
    }
}
