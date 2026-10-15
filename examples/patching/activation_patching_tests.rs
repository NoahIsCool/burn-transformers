//! Exactness checks for the generic activation-patching sweep on a tiny random model.
//!
//! A real checkpoint is not needed: the identities checked here hold for any
//! causal decoder, so a two-block random Llama-family model built through
//! `AutoCausalLm` exercises the same trait calls as a loaded checkpoint.

use super::*;
use burn::tensor::{Int, backend::Backend};

/// Two-block random Llama-family model selected from its configuration, as for a checkpoint.
fn tiny_model() -> AutoCausalLm<Cpu> {
    let device = Default::default();
    Cpu::seed(&device, 7);
    AutoCausalLm::<Cpu>::from_config_json(
        r#"{"model_type": "llama", "vocab_size": 16, "hidden_size": 8,
            "intermediate_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "max_position_embeddings": 16}"#,
        &device,
    )
    .expect("valid tiny configuration")
}

/// One unpadded five-token prompt with axes `[batch = 1, token]`.
fn prompt(ids: [i32; 5]) -> CausalLmInput<Cpu> {
    CausalLmInput {
        token_ids: Tensor::<Cpu, 2, Int>::from_ints([ids], &Default::default()),
        token_mask: None,
        position_ids: None,
    }
}

/// After the last block the readout sees only the final token, so patches there recover 0 or 1 exactly.
#[test]
fn final_block_patches_recover_exactly_zero_or_one() {
    let model = tiny_model();
    let sweep = patching_sweep(
        &model,
        prompt([1, 2, 3, 4, 5]),
        prompt([1, 2, 9, 4, 5]),
        2..5,
        6,
        7,
    )
    .unwrap();
    assert_eq!(sweep.recovered.len(), 2, "one row per block");
    assert!(sweep.recovered.iter().all(|row| row.len() == 3));
    assert_eq!(sweep.recovered[1], [0.0, 0.0, 1.0]);
    assert_ne!(sweep.clean, sweep.corrupt);
}

/// Tokens before the first difference carry identical residuals in both runs and recover nothing.
#[test]
fn tokens_before_the_change_recover_exactly_nothing() {
    let model = tiny_model();
    let sweep = patching_sweep(
        &model,
        prompt([1, 2, 3, 4, 5]),
        prompt([1, 2, 9, 4, 5]),
        0..5,
        6,
        7,
    )
    .unwrap();
    for row in &sweep.recovered {
        assert_eq!(row[..2], [0.0, 0.0]);
    }
}

/// Identical prompts leave no logit difference to recover, which is reported instead of dividing by zero.
#[test]
fn identical_prompts_are_rejected() {
    let model = tiny_model();
    let error = patching_sweep(
        &model,
        prompt([1, 2, 3, 4, 5]),
        prompt([1, 2, 3, 4, 5]),
        2..5,
        6,
        7,
    )
    .err()
    .expect("no gap to recover");
    assert!(error.to_string().contains("nothing to recover"));
}

/// The metric subtracts the distractor's final-token logit from the answer's.
#[test]
fn logit_difference_reads_the_two_named_logits() {
    let logits = Tensor::<Cpu, 3>::from_floats([[[0.5, 2.0, -1.0, 3.0]]], &Default::default());
    assert_eq!(logit_difference(logits.clone(), 3, 1), 1.0);
    assert_eq!(logit_difference(logits, 2, 0), -1.5);
}
