//! Llama-family parity with Transformers 5.2 on tiny random-weight checkpoints.
//!
//! Each fixture under `tests/fixtures/tiny_goldens/<case>` holds a seeded F32
//! checkpoint, its `config.json`, and a `golden.json` produced by
//! `scripts/generate_tiny_goldens.py` with Transformers 5.2.0 and PyTorch
//! 2.10.0 (eager attention, CPU, deterministic). The golden records two rows
//! of 12 tokens (row 1 left-padded by 4) with explicit `cumsum(mask) - 1`
//! positions, their full-vocabulary logits, row 0's per-layer hidden states,
//! and a cached greedy continuation.
//!
//! Every case is loaded through the strict automatic loader and checked for
//!
//! 1. full-vocabulary logits at every valid position of both rows;
//! 2. every hidden state of row 0 (Transformers reports the embedding, each
//!    block output but the last, and the final-normalized states);
//! 3. stateless-versus-cached agreement: a batched prefill of the first six
//!    columns followed by six batched single-token decode steps, in which
//!    the rows have different cached lengths (6 versus 2 valid tokens), and
//!    which also matches the golden logits;
//! 4. the cached greedy continuation, compared at every step whose top-2
//!    logit margin exceeds twice the tolerance (the reference's tokens are
//!    fed back, so a closer race, a tie at the backend's precision, is
//!    skipped), at least two steps per case.
//!
//! The checks are generic over the backend (shared helpers live in
//! `tests/support/tiny_goldens.rs`). They run on Flex as ordinary tests and,
//! ignored, on Metal (`--features metal`, macOS) and CUDA (`--features
//! cuda`):
//!
//! ```text
//! cargo test --locked --features metal --test tiny_goldens_llama -- --ignored --nocapture --test-threads 1
//! cargo test --locked --features cuda --test tiny_goldens_llama -- --ignored --nocapture --test-threads 1
//! ```
//!
//! # Tolerance
//!
//! Both sides run F32 with different matmul and reduction orders; through
//! four blocks of width 32 the largest observed logit deviation over all
//! seven cases is `7.2e-7` (logits have magnitude about one; the stateless
//! test prints each case's gap with `--nocapture`). The F32 tolerance is
//! `1e-5` absolute: over ten times that noise, so other CPUs' SIMD and thread
//! splits pass, and far below any architectural error (a wrong norm
//! placement, bias, rotary layout, or window moves logits by `1e-2` or more).
//! Metal multiplies in F32 and is held to the accelerator F32 tolerance of
//! `tests/support/tiny_goldens.rs`, `1e-5` plus `1e-5` of the compared
//! magnitude; on an Apple M3 Max its largest deviations over the seven cases
//! are `9.1e-7` for logits (biased, tied Llama; `7.5e-7` cached) and `2.9e-6`
//! for hidden states (OLMo 2, whose states reach `7.8`). A CUDA GPU that
//! rounds F32 matrix inputs to TF32 (compute capability 8.0 or newer) is held
//! to the TF32 bound of `tests/support/backend_numerics.rs`, `1e-2` plus
//! `1e-2` of the compared set's largest magnitude; an older GPU multiplies in
//! F32 and meets the accelerator F32 tolerance. The bound is chosen by
//! probing the device.

#![cfg(feature = "flex")]

#[path = "support/backend_numerics.rs"]
mod backend_numerics;
#[path = "support/tiny_goldens.rs"]
mod tiny_goldens;

use backend_numerics::{Bound, host};
use burn::{backend::Flex, tensor::backend::Backend};
use tiny_goldens::{
    F32_TOLERANCE, Golden, PADDING, check_cached, check_greedy, check_stateless, fixture, ints,
    load, mask,
};
use transformers::{
    CausalLmInput, ForwardOptions, cache::CacheConfig, models::AutoCausalConfigRef,
};

/// Deterministic F32 CPU backend.
type B = Flex<f32, i32>;

/// Every Llama-family fixture case.
const CASES: [&str; 7] = [
    "llama3",
    "llama-bias-tied",
    "qwen2",
    "qwen3",
    "mistral-sliding",
    "smollm3",
    "olmo2",
];

/// Stateless logits and hidden states match the reference at every valid position.
fn stateless_logits_and_hidden_states<A: Backend>(device: &A::Device, bound: Bound) {
    for case in CASES {
        let golden = Golden::case(case);
        let model = load::<A>(&fixture(case), device);
        check_stateless(&model, &golden, bound, device, case);
        assert_eq!(model.config().vocabulary_size(), golden.vocabulary);
        let AutoCausalConfigRef::Llama(config) = model.config() else {
            panic!("{case}: not registered as the Llama family");
        };
        assert_eq!(config.model_type, golden.model_type);
    }
}

/// Batched ragged prefill and decode reproduce the stateless and reference logits.
fn batched_cached_decode<A: Backend>(device: &A::Device, bound: Bound) {
    for case in CASES {
        let golden = Golden::case(case);
        let model = load::<A>(&fixture(case), device);
        check_cached(&model, &golden, 16, bound, device, case);
    }
}

/// The cached greedy continuation matches the reference at every decisive step, at least two per case.
fn cached_greedy_continuation<A: Backend>(device: &A::Device, bound: Bound) {
    for case in CASES {
        let golden = Golden::case(case);
        let model = load::<A>(&fixture(case), device);
        check_greedy(&model, &golden, bound, device, case);
    }
}

/// The left-padded row's padding never enters the cache or shifts positions.
fn padded_prefill_equals_unpadded_prefill<A: Backend>(device: &A::Device, bound: Bound) {
    let golden = Golden::case("llama3");
    let model = load::<A>(&fixture("llama3"), device);
    let row: Vec<i64> = golden.input_ids[1][PADDING..].to_vec();
    let new_state = || {
        model
            .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, device)
            .unwrap()
    };
    let (mut padded, mut plain) = (new_state(), new_state());
    let options = ForwardOptions::default();
    let with_padding = model
        .prefill(
            CausalLmInput {
                token_ids: ints(&[golden.input_ids[1].clone()], device),
                token_mask: Some(mask(&[golden.attention_mask[1].clone()], device)),
                position_ids: None,
            },
            &mut padded,
            &options,
        )
        .unwrap();
    let without = model
        .prefill(
            CausalLmInput {
                token_ids: ints(&[row], device),
                token_mask: None,
                position_ids: None,
            },
            &mut plain,
            &options,
        )
        .unwrap();
    bound.assert_close(
        &host(with_padding.logits),
        &host(without.logits),
        "padded vs unpadded",
    );
}

/// Stateless logits and hidden states match the reference at every valid position.
#[test]
fn stateless_logits_and_hidden_states_match_transformers() {
    stateless_logits_and_hidden_states::<B>(&Default::default(), F32_TOLERANCE);
}

/// Batched ragged prefill and decode reproduce the stateless and reference logits.
#[test]
fn batched_cached_decode_matches_stateless_and_reference_logits() {
    batched_cached_decode::<B>(&Default::default(), F32_TOLERANCE);
}

/// The cached greedy continuation matches the reference at every decisive step.
#[test]
fn cached_greedy_continuation_matches_transformers() {
    cached_greedy_continuation::<B>(&Default::default(), F32_TOLERANCE);
}

/// The left-padded row's padding never enters the cache or shifts positions.
#[test]
fn padded_prefill_equals_unpadded_prefill_of_the_same_tokens() {
    padded_prefill_equals_unpadded_prefill::<B>(&Default::default(), F32_TOLERANCE);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    //! Metal/F32 counterparts of the Flex checks, held to the probed bound.

    use burn::backend::Metal;

    use super::tiny_goldens::golden_bound;

    /// Metal/F32: stateless logits and hidden states of every case.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_stateless_logits_and_hidden_states_match_transformers() {
        let device = Default::default();
        super::stateless_logits_and_hidden_states::<Metal>(&device, golden_bound::<Metal>(&device));
    }

    /// Metal/F32: batched ragged prefill and decode.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_batched_cached_decode_matches_reference_logits() {
        let device = Default::default();
        super::batched_cached_decode::<Metal>(&device, golden_bound::<Metal>(&device));
    }

    /// Metal/F32: the cached greedy continuation.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_cached_greedy_continuation_matches_transformers() {
        let device = Default::default();
        super::cached_greedy_continuation::<Metal>(&device, golden_bound::<Metal>(&device));
    }

    /// Metal/F32: padding never enters the cache.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_padded_prefill_equals_unpadded_prefill_of_the_same_tokens() {
        let device = Default::default();
        super::padded_prefill_equals_unpadded_prefill::<Metal>(
            &device,
            golden_bound::<Metal>(&device),
        );
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    //! CUDA counterparts of the Flex checks, held to the probed bound (F32 or TF32).

    use burn::backend::Cuda;

    use super::tiny_goldens::golden_bound;

    /// CUDA: stateless logits and hidden states of every case.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_stateless_logits_and_hidden_states_match_transformers() {
        let device = Default::default();
        super::stateless_logits_and_hidden_states::<Cuda>(&device, golden_bound::<Cuda>(&device));
    }

    /// CUDA: batched ragged prefill and decode.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_batched_cached_decode_matches_reference_logits() {
        let device = Default::default();
        super::batched_cached_decode::<Cuda>(&device, golden_bound::<Cuda>(&device));
    }

    /// CUDA: the cached greedy continuation.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_cached_greedy_continuation_matches_transformers() {
        let device = Default::default();
        super::cached_greedy_continuation::<Cuda>(&device, golden_bound::<Cuda>(&device));
    }

    /// CUDA: padding never enters the cache.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_padded_prefill_equals_unpadded_prefill_of_the_same_tokens() {
        let device = Default::default();
        super::padded_prefill_equals_unpadded_prefill::<Cuda>(
            &device,
            golden_bound::<Cuda>(&device),
        );
    }
}
