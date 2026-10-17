//! gpt-oss parity with Transformers 5.2 on a tiny random-weight checkpoint.
//!
//! `tests/fixtures/tiny_goldens/gpt-oss` holds a seeded F32 checkpoint, its
//! `config.json`, and a `golden.json` produced by
//! `scripts/generate_tiny_goldens.py` with Transformers 5.2.0 and PyTorch
//! 2.10.0 (eager attention, CPU, deterministic). The configuration exercises
//! every gpt-oss mechanism at width 32: four layers alternating sliding
//! (window 4) and full attention over 12 tokens, learned sinks drawn from
//! `N(0, 0.5²)`, biased Q/K/V/O, a biased router choosing 2 of 4 experts by
//! top-k-then-softmax, clamped SwiGLU experts with biases over an interleaved
//! `gate_up_proj`, and YaRN RoPE (factor 4, `truncate: false`). The golden
//! records two rows of 12 tokens (row 1 left-padded by 4) with explicit
//! `cumsum(mask) - 1` positions, their full-vocabulary logits, row 0's
//! per-layer hidden states, and a cached greedy continuation.
//!
//! The checkpoint is loaded through the strict automatic loader and checked
//! for
//!
//! 1. full-vocabulary logits at every valid position of both rows, under
//!    the default fused attention and under the eager F32-softmax graph (the
//!    two sink implementations);
//! 2. every hidden state of row 0 (Transformers reports the embedding, each
//!    block output but the last, and the final-normalized states);
//! 3. stateless-versus-cached agreement: a batched prefill of the first six
//!    columns followed by six batched single-token decode steps (rows of 6
//!    and 2 cached tokens, so the sliding layers' ring caches wrap and the
//!    rows' windows differ), which also match the golden logits, at two
//!    cache capacities;
//! 4. the cached greedy continuation, compared at every step whose top-2
//!    logit margin exceeds the tolerance (the reference's tokens are fed
//!    back, so a tie is skipped);
//! 5. router diagnostics, and the load-balancing objective against
//!    Transformers' pinned `aux_loss` for the same padded batch.
//!
//! The checks are generic over the backend (shared helpers live in
//! `tests/support/tiny_goldens.rs`). They run on Flex as ordinary tests and,
//! ignored, on Metal (`--features metal`, macOS) and CUDA (`--features
//! cuda`):
//!
//! ```text
//! cargo test --locked --features metal --test tiny_goldens_gpt_oss -- --ignored --nocapture --test-threads 1
//! cargo test --locked --features cuda --test tiny_goldens_gpt_oss -- --ignored --nocapture --test-threads 1
//! ```
//!
//! # Tolerance
//!
//! Both sides run F32 with different matmul and reduction orders. Through
//! four blocks of width 32 with routing, the largest observed logit
//! deviation is printed with `--nocapture` (about `1e-6`); the F32 tolerance
//! is `1e-5` absolute, far below any architectural error (a wrong sink,
//! clamp, bias, router order, rotary layout, or window moves logits by `1e-2`
//! or more). A flipped expert selection would also move logits by far more
//! than the tolerance, so passing proves every routing decision matches.
//!
//! Metal multiplies in F32 and is held to the accelerator F32 tolerance of
//! `tests/support/tiny_goldens.rs` (`1e-5` plus `1e-5` of the compared
//! magnitude); on an Apple M3 Max its logits deviate by at most `1.07e-6`
//! (`7.8e-7` eager, `7.3e-7` cached at both capacities), its hidden states by
//! `2.1e-6`, and its load-balancing loss by `2.4e-7`. A CUDA
//! GPU that rounds F32 matrix inputs to TF32 (compute capability 8.0 or
//! newer) is held to the TF32 bound of `tests/support/backend_numerics.rs`,
//! `1e-2` plus `1e-2` of the compared set's largest magnitude; an older GPU
//! multiplies in F32 and meets the accelerator F32 tolerance. The bound is
//! chosen by probing the device. A flipped expert choice moves logits by
//! more than either bound, so routing parity holds on every backend.

#![cfg(feature = "flex")]

#[path = "support/backend_numerics.rs"]
mod backend_numerics;
#[path = "support/tiny_goldens.rs"]
mod tiny_goldens;

use backend_numerics::{Bound, Gap, host};
use burn::{backend::Flex, tensor::backend::Backend};
use tiny_goldens::{
    F32_TOLERANCE, Golden, PREFILL, cached_logits, check_all_logits, check_greedy, check_stateless,
    fixture, load,
};
use transformers::{
    AutoCausalLm, CausalLmInput, ForwardOptions, LogitsSelection,
    attention::AttentionPrecisionMode,
    models::{AutoCausalConfigRef, gpt_oss::GptOssConfig},
};

/// Deterministic F32 CPU backend.
type B = Flex<f32, i32>;

/// Transformers' unscaled `aux_loss` for the golden batch (padding-masked, all four layers).
///
/// Pinned with Transformers 5.2.0 and PyTorch 2.10.0 from the fixture
/// itself (`golden.json` predates this check and does not record it):
///
/// ```text
/// model = AutoModelForCausalLM.from_pretrained(fixture, dtype=torch.float32,
///                                              attn_implementation="eager").eval()
/// golden = json.load(open(fixture / "golden.json"))
/// output = model(input_ids=tensor(golden["input_ids"]),
///                attention_mask=tensor(golden["attention_mask"]),
///                position_ids=tensor(golden["position_ids"]),
///                output_router_logits=True)
/// output.aux_loss.item()  # 2.020416736602783
/// ```
///
/// `router_aux_loss_coef` scales it only inside the training loss, so the
/// crate's [`ForwardOptions::auxiliary_losses`] output is compared unscaled.
const REFERENCE_LOAD_BALANCING_LOSS: f32 = 2.020_416_7;

/// The fixture case.
const CASE: &str = "gpt-oss";

/// Reads the fixture's golden, which must be a gpt-oss one.
fn golden() -> Golden {
    let golden = Golden::case(CASE);
    assert_eq!(golden.model_type, "gpt_oss");
    golden
}

/// Loads the fixture through the strict automatic loader on `device`.
fn load_model<A: Backend>(device: &A::Device) -> AutoCausalLm<A> {
    load::<A>(&fixture(CASE), device)
}

/// Stateless logits and hidden states match the reference at every valid position.
fn stateless_logits_and_hidden_states<A: Backend>(device: &A::Device, bound: Bound) {
    let golden = golden();
    let model = load_model::<A>(device);
    let AutoCausalConfigRef::GptOss(config) = model.config() else {
        panic!("not registered as the gpt-oss family");
    };
    assert_eq!(config.model_type, "gpt_oss");
    assert_eq!(model.config().vocabulary_size(), golden.vocabulary);
    check_stateless(&model, &golden, bound, device, "gpt-oss fused");
}

/// The eager F32-softmax graph (explicit sink column) matches the reference too.
///
/// With sinks every attention-pattern row sums to `1 - p_sink`, in `[0, 1)`.
fn eager_sink_attention<A: Backend>(device: &A::Device, bound: Bound) {
    let golden = golden();
    let AutoCausalLm::GptOss(model) = load_model::<A>(device) else {
        panic!("gpt_oss is not registered as the gpt-oss family");
    };
    let model = model.with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    let output = model
        .forward(
            golden.input(device),
            &ForwardOptions {
                logits: LogitsSelection::All,
                attentions: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    let largest = check_all_logits(&golden, &host(output.logits), bound, "eager");
    eprintln!("gpt-oss: largest eager logit gap {:.2e}", largest.deviation);
    assert_eq!(output.attentions.len(), golden.hidden_shape[0] - 1);
    for pattern in output.attentions {
        let [batch, heads, queries, keys] = pattern.dims();
        let sums = host(pattern.sum_dim(3));
        assert_eq!(sums.len(), batch * heads * queries);
        assert!(keys > 0);
        assert!(
            sums.iter().all(|&sum| (0.0..1.0).contains(&sum)),
            "{sums:?}"
        );
    }
}

/// Batched ragged prefill and decode reproduce the reference logits, with ring and full caches.
///
/// Both capacities exceed the window, so the window-4 layers use a wrapping
/// ring; capacity 12 is exactly filled by the longer row.
fn batched_cached_decode<A: Backend>(device: &A::Device, bound: Bound) {
    let golden = golden();
    let model = load_model::<A>(device);
    let v = golden.vocabulary;
    for capacity in [16, 12] {
        let mut largest = Gap::default();
        for (step, logits) in cached_logits(&model, &golden, capacity, device)
            .iter()
            .enumerate()
        {
            let column = PREFILL - 1 + step;
            for row in 0..2 {
                largest = largest.max(bound.assert_close(
                    &logits[row * v..(row + 1) * v],
                    golden.logits_at(row, column),
                    &format!("capacity {capacity}: cached logits[{row}, {column}]"),
                ));
            }
        }
        eprintln!(
            "gpt-oss: largest cached logit gap {:.2e} (capacity {capacity})",
            largest.deviation
        );
    }
}

/// The cached greedy continuation matches the reference at every decisive step.
fn cached_greedy_continuation<A: Backend>(device: &A::Device, bound: Bound) {
    check_greedy(&load_model::<A>(device), &golden(), bound, device, CASE);
}

/// Router diagnostics are returned on request and the load-balancing objective equals the reference's.
///
/// The objective concatenates every layer's router logits over the valid
/// tokens only, so a wrong padding mask, layer order, routing rule, or
/// softmax moves it far beyond the tolerance. Without the mask the padding
/// tokens join the averages: that loss must differ from the reference by
/// over `1e-3`, and by more than twice our masked gap, so the comparison
/// resolves the mask under any bound that admits the masked loss.
fn router_logits_and_load_balancing_loss<A: Backend>(device: &A::Device, bound: Bound) {
    let golden = golden();
    let model = load_model::<A>(device);
    let output = model
        .forward(
            golden.input(device),
            &ForwardOptions {
                router_logits: true,
                auxiliary_losses: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    let layers = golden.hidden_shape[0] - 1;
    assert_eq!(output.routers.len(), layers);
    for router in &output.routers {
        let logits = router.logits.as_ref().expect("router logits");
        assert_eq!(logits.dims(), [2, 12, 4]);
    }
    let loss = host(output.load_balancing_loss.expect("load-balancing loss"));
    assert_eq!(loss.len(), 1);
    let gap = bound.assert_close(
        &loss,
        &[REFERENCE_LOAD_BALANCING_LOSS],
        "load-balancing loss",
    );
    let unmasked = host(
        model
            .forward(
                CausalLmInput {
                    token_mask: None,
                    ..golden.input(device)
                },
                &ForwardOptions {
                    auxiliary_losses: true,
                    ..ForwardOptions::default()
                },
            )
            .unwrap()
            .load_balancing_loss
            .expect("load-balancing loss"),
    );
    let moved = (unmasked[0] - REFERENCE_LOAD_BALANCING_LOSS).abs();
    eprintln!(
        "gpt-oss: load-balancing loss gap {:.2e}; unmasked loss moves by {moved:.2e}",
        gap.deviation
    );
    assert!(moved > 1e-3, "the padding mask must matter: {unmasked:?}");
    assert!(
        moved > 2.0 * gap.deviation,
        "the masked loss must be nearer the reference than the unmasked one: {unmasked:?}"
    );
}

/// Stateless logits and hidden states match the reference at every valid position.
#[test]
fn stateless_logits_and_hidden_states_match_transformers() {
    stateless_logits_and_hidden_states::<B>(&Default::default(), F32_TOLERANCE);
}

/// The eager F32-softmax graph (explicit sink column) matches the reference too.
#[test]
fn eager_sink_attention_matches_transformers() {
    eager_sink_attention::<B>(&Default::default(), F32_TOLERANCE);
}

/// Batched ragged prefill and decode reproduce the reference logits, with ring and full caches.
#[test]
fn batched_cached_decode_matches_reference_logits() {
    batched_cached_decode::<B>(&Default::default(), F32_TOLERANCE);
}

/// The cached greedy continuation matches the reference at every decisive step.
#[test]
fn cached_greedy_continuation_matches_transformers() {
    cached_greedy_continuation::<B>(&Default::default(), F32_TOLERANCE);
}

/// Router diagnostics are returned on request and the load-balancing objective equals the reference's.
#[test]
fn router_logits_and_load_balancing_loss_are_reported() {
    router_logits_and_load_balancing_loss::<B>(&Default::default(), F32_TOLERANCE);
}

/// The fixture's configuration round-trips through the schema.
#[test]
fn fixture_config_round_trips() {
    let text = std::fs::read_to_string(fixture(CASE).join("config.json")).unwrap();
    let config: GptOssConfig = serde_json::from_str(&text).unwrap();
    let again: GptOssConfig =
        serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
    assert_eq!(config, again);
    let architecture = config.architecture().unwrap();
    assert_eq!(architecture.sliding_window(), Some(4));
    assert_eq!(architecture.num_layers(), 4);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    //! Metal/F32 counterparts of the Flex checks, held to the probed bound.

    use burn::backend::Metal;

    use super::tiny_goldens::golden_bound;

    /// Metal/F32: stateless logits and hidden states.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_stateless_logits_and_hidden_states_match_transformers() {
        let device = Default::default();
        super::stateless_logits_and_hidden_states::<Metal>(&device, golden_bound::<Metal>(&device));
    }

    /// Metal/F32: the eager sink graph.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_eager_sink_attention_matches_transformers() {
        let device = Default::default();
        super::eager_sink_attention::<Metal>(&device, golden_bound::<Metal>(&device));
    }

    /// Metal/F32: batched ragged prefill and decode at two capacities.
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

    /// Metal/F32: router diagnostics and the load-balancing objective.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_router_logits_and_load_balancing_loss_are_reported() {
        let device = Default::default();
        super::router_logits_and_load_balancing_loss::<Metal>(
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

    /// CUDA: stateless logits and hidden states.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_stateless_logits_and_hidden_states_match_transformers() {
        let device = Default::default();
        super::stateless_logits_and_hidden_states::<Cuda>(&device, golden_bound::<Cuda>(&device));
    }

    /// CUDA: the eager sink graph.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_eager_sink_attention_matches_transformers() {
        let device = Default::default();
        super::eager_sink_attention::<Cuda>(&device, golden_bound::<Cuda>(&device));
    }

    /// CUDA: batched ragged prefill and decode at two capacities.
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

    /// CUDA: router diagnostics and the load-balancing objective.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_router_logits_and_load_balancing_loss_are_reported() {
        let device = Default::default();
        super::router_logits_and_load_balancing_loss::<Cuda>(
            &device,
            golden_bound::<Cuda>(&device),
        );
    }
}
