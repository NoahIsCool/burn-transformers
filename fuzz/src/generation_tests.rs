//! Deterministic regression seeds for the generation fuzz harness.

use super::*;

/// EOS overrides a simultaneous stop and ties among equal-length stops preserve input order.
#[test]
fn oracle_precedence_and_tracker_edges() {
    let config = GenerationConfig {
        max_new_tokens: 1,
        eos_token_ids: vec![2],
        stop_sequences: vec![vec![1], vec![0, 1], vec![0, 1]],
        ..GenerationConfig::default()
    };
    assert_eq!(
        expected_finish(&config, &[2]),
        Some(FinishReason::EndOfSequence { token_id: 2 })
    );
    assert_eq!(
        expected_finish(&config, &[0, 1]),
        Some(FinishReason::StopSequence {
            index: 1,
            length: 2
        })
    );
    exercise(&[]);
    exercise(&[255; 512]);
    exercise(&(0..=255).collect::<Vec<_>>());
}

/// A stop sequence completed before `min_new_tokens` is ignored; EOS still finishes.
#[test]
fn stops_wait_for_the_minimum_length_but_eos_does_not() {
    let config = GenerationConfig {
        max_new_tokens: 6,
        min_new_tokens: 3,
        eos_token_ids: vec![7],
        stop_sequences: vec![vec![1, 2], vec![2]],
        ..GenerationConfig::default()
    };
    // Two tokens ending in both stops: below the minimum, so nothing finishes.
    assert_eq!(expected_finish(&config, &[1, 2]), None);
    // The third token reaches the minimum; the longer stop wins.
    assert_eq!(
        expected_finish(&config, &[0, 1, 2]),
        Some(FinishReason::StopSequence {
            index: 0,
            length: 2
        })
    );
    // EOS is not delayed by the tracker (the logits processor masks it instead).
    assert_eq!(
        expected_finish(&config, &[7]),
        Some(FinishReason::EndOfSequence { token_id: 7 })
    );
    let mut tracker = GenerationTracker::new(config.clone(), 8, 1).unwrap();
    assert_eq!(tracker.push(0, 1).unwrap(), None);
    assert_eq!(tracker.push(0, 2).unwrap(), None);
    assert_eq!(
        tracker.push(0, 2).unwrap(),
        Some(&FinishReason::StopSequence {
            index: 1,
            length: 1
        })
    );
}

/// Every newer validation rule is restated by the oracle and enforced by the library.
#[test]
fn newer_controls_are_validated_like_the_oracle() {
    let valid = GenerationConfig {
        min_new_tokens: 4,
        max_new_tokens: 4,
        min_p: Some(0.0),
        frequency_penalty: -2.0,
        logit_bias: BTreeMap::from([(3, -100.0)]),
        no_repeat_ngram_size: usize::MAX,
        ..GenerationConfig::default()
    };
    assert!(expected_valid(&valid, 4));
    assert!(valid.validate(4).is_ok());
    check_round_trip(&valid, 4);
    let invalid = [
        GenerationConfig {
            min_new_tokens: 5,
            max_new_tokens: 4,
            ..valid.clone()
        },
        GenerationConfig {
            min_p: Some(f64::NAN),
            ..valid.clone()
        },
        GenerationConfig {
            min_p: Some(1.0 + f64::EPSILON),
            ..valid.clone()
        },
        GenerationConfig {
            frequency_penalty: f64::INFINITY,
            ..valid.clone()
        },
        GenerationConfig {
            logit_bias: BTreeMap::from([(4, 0.0)]),
            ..valid.clone()
        },
        GenerationConfig {
            logit_bias: BTreeMap::from([(0, f64::NEG_INFINITY)]),
            ..valid.clone()
        },
        GenerationConfig {
            stop_sequences: vec![Vec::new()],
            ..valid.clone()
        },
        GenerationConfig {
            pad_token_id: Some(u32::MAX),
            ..valid.clone()
        },
    ];
    for config in invalid {
        assert!(!expected_valid(&config, 4), "{config:?}");
        assert!(config.validate(4).is_err(), "{config:?}");
    }
}

/// Raw JSON with the newer keys takes the JSON branch of the harness.
#[test]
fn json_configurations_with_newer_keys_round_trip() {
    exercise(
        br#"{"max_new_tokens":3,"min_new_tokens":2,"min_p":0.1,"frequency_penalty":0.5,
            "logit_bias":{"0":1.5},"no_repeat_ngram_size":2,"stop_sequences":[[0,0]]}"#,
    );
    exercise(br#"{"min_p":-0.0,"logit_bias":{"31":-1e300},"temperature":1e-300}"#);
    exercise(include_bytes!("../seeds/generation/config.json"));
}

/// Pseudo-random inputs cover many configuration and push mixes.
#[test]
fn pseudo_random_inputs_agree_with_the_oracles() {
    for input in crate::lcg_inputs(0x6765_6e65, 512, 160) {
        exercise(&input);
    }
}

/// Campaign regression `crash-8872c8a0` (minimized): a JSON float far below one
/// read back a few ulps away before `serde_json`'s `float_roundtrip` parser was
/// enabled; the round trip is now exact and checked exactly.
#[test]
fn campaign_float_round_trip_is_exact() {
    exercise(&[
        0x72, 0x65, 0x0a, 0x0a, 0x0a, 0xff, 0x22, 0x54, 0x6f, 0x3a, 0x31, 0x36, 0x7d, 0x0a, 0x54,
        0x3a, 0x0a, 0x0a,
    ]);
    // Without `float_roundtrip` this read back as `3.799796971636185e-262`.
    let float = 3.799_796_971_636_184_6e-262_f64;
    let reparsed: f64 = serde_json::from_str(&serde_json::to_string(&float).unwrap()).unwrap();
    assert_eq!(float.to_bits(), reparsed.to_bits());
    assert!(!crate::floats_round_trip(float, 3.799_796_971_636_185e-262));
    assert!(!crate::floats_round_trip(1.0, 1.0 + 1e-12));
}
