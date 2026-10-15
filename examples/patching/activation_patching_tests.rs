//! Exactness checks for the generic activation-patching sweep on a tiny random model.
//!
//! A real checkpoint is not needed: the identities checked here hold for any
//! causal decoder, so a two-block random Llama-family model built through
//! `AutoCausalLm` exercises the same trait calls as a loaded checkpoint.
//!
//! The random weights must be the same on every run. Flex keeps one
//! process-wide random generator, `Backend::seed` resets it, and parameters
//! are drawn lazily on their first use inside a forward pass. Tests run on
//! parallel threads, so without coordination one test's draws interleave with
//! another's seed and the model changes from run to run; about 1% of runs then
//! built a model whose two prompts scored alike. Each test therefore holds
//! [`RNG_LOCK`] from seeding to its last forward pass.

use std::sync::{Mutex, MutexGuard};

use super::*;
use burn::tensor::{Device, Int, backend::Backend};

/// Serializes the tests that seed and draw from Flex's process-wide random generator.
static RNG_LOCK: Mutex<()> = Mutex::new(());

/// Builds a model from freshly seeded random weights while holding [`RNG_LOCK`].
///
/// The returned guard keeps every other test away from the shared random
/// generator. Bind it to a named variable such as `_rng`, not `_`, so it lives
/// until the end of the test: the weights are only drawn during the first
/// forward pass. A test that panicked while holding the lock leaves nothing
/// half-initialized behind, so a poisoned lock is simply taken over.
fn seeded<M>(build: impl FnOnce(&Device<Cpu>) -> M) -> (MutexGuard<'static, ()>, M) {
    let guard = RNG_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let device = Default::default();
    Cpu::seed(&device, 7);
    (guard, build(&device))
}

/// Two-block random Llama-family model selected from its configuration, as for a checkpoint.
fn tiny_model() -> (MutexGuard<'static, ()>, AutoCausalLm<Cpu>) {
    seeded(|device| {
        AutoCausalLm::<Cpu>::from_config_json(
            r#"{"model_type": "llama", "vocab_size": 16, "hidden_size": 8,
                "intermediate_size": 16, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                "max_position_embeddings": 16}"#,
            device,
        )
        .expect("valid tiny configuration")
    })
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
    let (_rng, model) = tiny_model();
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
    let (_rng, model) = tiny_model();
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
    let (_rng, model) = tiny_model();
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

/// Runs the two-block sweep over the changed token and every later one on a freshly seeded model.
fn seeded_sweep() -> Sweep {
    let (_rng, model) = tiny_model();
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
/// means a future change to initialization or to the seed handling fails with
/// this test's message instead of an intermittent "nothing to recover".
#[test]
fn seeded_model_is_reproducible_and_separates_the_prompts() {
    let first = seeded_sweep();
    let second = seeded_sweep();
    assert_eq!((first.clean, first.corrupt), (second.clean, second.corrupt));
    assert_eq!(first.recovered, second.recovered);
    // Seed 7 gives a gap of about 0.23 on this configuration.
    let gap = (first.clean - first.corrupt).abs();
    assert!(
        gap > 0.1,
        "logit-difference gap {gap} is too close to the 1e-3 threshold"
    );
}
