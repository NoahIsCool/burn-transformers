//! Deterministic regression seeds for the configuration fuzz harness.

use super::*;

/// Layer-loop bounds are independent of intentionally unbounded arithmetic probes.
#[test]
fn bounds_work_without_hiding_dimension_overflow() {
    assert!(!bounded_work(
        &serde_json::json!({"num_hidden_layers": 129})
    ));
    assert!(bounded_work(&serde_json::json!({"head_dim": usize::MAX})));
    exercise(br#"{"model_type":"t5","num_layers":2,"num_decoder_layers":2}"#);
    exercise(br#"{"text_config":{"model_type":"qwen3_5_text"}}"#);
    exercise(br#"{"time_step_limit":[0.0,Infinity]}"#);
    exercise(b"\xff");
}
