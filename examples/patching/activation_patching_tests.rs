//! Exactness checks for the generic activation-patching sweep on a tiny random model.
//!
//! A real checkpoint is not needed: the identities checked here hold for any
//! causal decoder, so a two-block random Llama-family model built through
//! `AutoCausalLm` exercises the same trait calls as a loaded checkpoint. A
//! tiny GPT-2 shows that a family `AutoCausalLm` does not dispatch runs
//! through the same sweep as a concrete model type.
//!
//! The identities need only a logit gap the sweep accepts (at least `1e-3`),
//! but which random weights give one is a matter of the draw: re-seeding
//! Burn's generator, 0.76% of draws scored both prompts alike on the Llama
//! model and 0.5% on GPT-2, and the Llama gap's median was only `0.063`.
//! Burn Flex keeps one process-wide generator and draws parameters lazily, on
//! first use, so tests on parallel threads used to interleave their draws; a
//! lock then made each test's draw repeatable, but the weights still depended
//! on Burn's generator and initializers. The weights now come from a fixed
//! seed alone ([`deterministic_parameters`]), chosen for a wide gap, so no test
//! needs to coordinate with another.

use super::*;
use burn::tensor::{Device, Int};
use transformers::models::gpt2::{Gpt2Config, Gpt2ForCausalLm};

#[path = "../../tests/support/deterministic_parameters.rs"]
mod deterministic_parameters;

use deterministic_parameters::deterministic_parameters;

/// Seed of both tiny models' weights.
///
/// It gives logit gaps of `0.96` (Llama family) and `1.06` (GPT-2), ten
/// times the `0.1` that [`seeded_model_is_reproducible_and_separates_the_prompts`]
/// requires. Over seeds 0 to 1,999 the median gaps were `0.21` and `0.081`,
/// and 4 and 13 seeds fell below the sweep's `1e-3` threshold.
const SEED: u64 = 1896;

/// Two-block random Llama-family model selected from its configuration, as for a checkpoint.
///
/// Its weights come from [`SEED`] alone, whatever Burn's generator holds.
fn tiny_model() -> AutoCausalLm<Cpu> {
    tiny_llama(SEED)
}

/// The two-block Llama-family model of [`tiny_model`] with weights from `seed`.
fn tiny_llama(seed: u64) -> AutoCausalLm<Cpu> {
    let device: Device<Cpu> = Default::default();
    let model = AutoCausalLm::<Cpu>::from_config_json(
        r#"{"model_type": "llama", "vocab_size": 16, "hidden_size": 8,
            "intermediate_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "max_position_embeddings": 16}"#,
        &device,
    )
    .expect("valid tiny configuration");
    deterministic_parameters(model, seed)
}

/// A two-block GPT-2 of the same widths as [`tiny_llama`], with weights from `seed`.
///
/// Every random parameter is replaced, so the configured initializer range
/// does not matter; LayerNorm scales and zero biases keep their constants.
fn tiny_gpt2(seed: u64) -> Gpt2ForCausalLm<Cpu> {
    let model = Gpt2ForCausalLm::<Cpu>::init(
        Gpt2Config {
            vocab_size: 16,
            n_embd: 8,
            n_head: 2,
            n_layer: 2,
            n_positions: 16,
            n_inner: Some(16),
            ..Default::default()
        },
        &Default::default(),
    )
    .expect("valid tiny GPT-2 configuration");
    deterministic_parameters(model, seed)
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

/// Runs the two-block sweep over the changed token and every later one on a freshly built model.
fn seeded_sweep() -> Sweep {
    let model = tiny_model();
    patching_sweep(
        &model,
        prompt([1, 2, 3, 4, 5]),
        prompt([1, 2, 9, 4, 5]),
        2..5,
        6,
        7,
    )
    .expect("the seeded model separates the prompts")
}

/// The seeded weights are reproducible and separate the prompts far beyond the 1e-3 rejection threshold.
///
/// The exactness tests above depend on both properties. Pinning them here
/// means a future change to the deterministic weights or to the model's
/// numerics fails with this test's message instead of "nothing to recover".
#[test]
fn seeded_model_is_reproducible_and_separates_the_prompts() {
    let first = seeded_sweep();
    let second = seeded_sweep();
    assert_eq!((first.clean, first.corrupt), (second.clean, second.corrupt));
    assert_eq!(first.recovered, second.recovered);
    // SEED gives a gap of about 0.96 on this configuration.
    let gap = (first.clean - first.corrupt).abs();
    assert!(
        gap > 0.1,
        "logit-difference gap {gap} is too close to the 1e-3 threshold"
    );
}

/// GPT-2 is not an `AutoCausalLm` family, but its concrete model satisfies the sweep's two trait bounds.
///
/// GPT-2 applies a final LayerNorm after the last block. It acts on each token
/// separately, so patching the final token after the last block still
/// reproduces the clean logits exactly and the other tokens recover nothing.
#[test]
fn concrete_gpt2_runs_through_the_same_sweep() {
    let model = tiny_gpt2(SEED);
    let sweep = patching_sweep(
        &model,
        prompt([1, 2, 3, 4, 5]),
        prompt([1, 2, 9, 4, 5]),
        2..5,
        6,
        7,
    )
    .unwrap();
    // SEED gives a gap of about 1.06 on this configuration.
    assert!((sweep.clean - sweep.corrupt).abs() > 0.1);
    assert_eq!(sweep.recovered[1], [0.0, 0.0, 1.0]);
}
