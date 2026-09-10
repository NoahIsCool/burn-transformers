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
