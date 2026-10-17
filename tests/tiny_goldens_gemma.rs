//! Gemma parity with Transformers 5.2 on tiny random-weight checkpoints.
//!
//! Each fixture under `tests/fixtures/tiny_goldens/<case>` holds a seeded F32
//! checkpoint, its `config.json`, and a `golden.json` produced by
//! `scripts/generate_tiny_goldens.py` with Transformers 5.2.0 and PyTorch
//! 2.10.0 (eager attention, CPU, deterministic):
//!
//! * `gemma2`: sandwich norms, `(1 + w)` RMSNorm, the `sqrt(d)` embedding
//!   scale, attention (50) and final (30) soft-caps, alternating
//!   sliding/global layers with window 4, `query_pre_attn_scalar = 16`;
//! * `gemma3-text`: per-head Q/K RMSNorm, three sliding layers (local RoPE
//!   base 1e4) and one global layer (base 1e6 with linear scaling 8).
//!
//! The golden records two rows of 12 tokens (row 1 left-padded by 4) with
//! explicit `cumsum(mask) - 1` positions, their full-vocabulary logits,
//! row 0's per-layer hidden states, and a cached greedy continuation. Every
//! case is loaded through the strict automatic loader and checked for
//!
//! 1. full-vocabulary logits at every valid position of both rows, with
//!    explicit and with implicit positions;
//! 2. every hidden state of row 0 (Transformers reports the scaled
//!    embedding, each block output but the last, and the final-normalized
//!    states);
//! 3. stateless-versus-cached agreement: a batched prefill of the first six
//!    columns followed by six batched single-token decode steps, in which
//!    the rows have different cached lengths (6 versus 2 valid tokens) and
//!    the sliding layers' ring (window 4 below capacity 16) wraps;
//! 4. chunked ragged steps: multi-token prefills on a non-empty state with
//!    the rows' new tokens at different offsets, then decode steps;
//! 5. the cached greedy continuation, compared at every step whose top-2
//!    logit margin exceeds the tolerance (the reference's tokens are fed
//!    back, so a tie is skipped), at least two steps per case;
//! 6. a hooked pass with no actions, which must equal the plain pass.
//!
//! Two further checks reuse the fixtures with one changed field:
//! `gemma3-text` with the mixed RoPE mapping Transformers 5.2 itself writes
//! for single-kind `layer_types` (its logits are bit-identical in
//! Transformers), and `gemma2` with a saturating attention soft-cap, against
//! reference logits embedded below ([`SATURATED_CAP_LOGITS`]).
//!
//! The checks are generic over the backend (shared helpers live in
//! `tests/support/tiny_goldens.rs`). They run on Flex as ordinary tests and,
//! ignored, on Metal (`--features metal`, macOS) and CUDA (`--features
//! cuda`):
//!
//! ```text
//! cargo test --locked --features metal --test tiny_goldens_gemma -- --ignored --nocapture --test-threads 1
//! cargo test --locked --features cuda --test tiny_goldens_gemma -- --ignored --nocapture --test-threads 1
//! ```
//!
//! # Tolerance
//!
//! Both sides run F32 with different matmul and reduction orders. The
//! largest observed logit deviation is printed by the stateless test with
//! `--nocapture`: `1.5e-6` for `gemma2` and `7.5e-7` for `gemma3-text` on
//! Flex. The F32 tolerance is `1e-5` absolute, as for the Llama family: over
//! six times that noise. Most mechanisms move the fixture logits far more
//! than that (the final soft-cap alone by `1.7e-3`), but the fixture's
//! attention soft-cap is nearly linear: its raw scores stay below `0.8`, so
//! `50 · tanh(s / 50)` differs from `s` by under `1e-4` and removing the cap
//! moves the reference logits by only `1.4e-5`. The attention soft-cap is
//! therefore validated separately in its saturated regime (`c = 0.25`,
//! `s / c` up to `3.2`), where removing it moves the reference logits by
//! `1.6e-1`.
//!
//! Metal multiplies in F32 and is held to the accelerator F32 tolerance of
//! `tests/support/tiny_goldens.rs`, `1e-5` plus `1e-5` of the compared
//! magnitude. On an Apple M3 Max its logits deviate by at most `3.6e-6`
//! (`gemma2`; `8.9e-7` for `gemma3-text`) in the stateless, cached, and
//! chunked checks, the saturated-cap logits by `1.8e-6`, and the hidden
//! states by `1.14e-5` at magnitude `7.0`: a relative `1.6e-6`, the reason
//! accelerators carry a relative term. A CUDA GPU that rounds F32 matrix
//! inputs to TF32 (compute capability 8.0 or newer) is held to the TF32 bound
//! of `tests/support/backend_numerics.rs`, `1e-2` plus `1e-2` of the compared
//! set's largest magnitude; an older GPU multiplies in F32 and meets the
//! accelerator F32 tolerance. The bound is chosen by probing the device. Under the TF32 bound
//! the near-linear cap-50 attention soft-cap is not resolved at all, and the
//! saturated-cap check relies on the run being nearer the cap-0.25 reference
//! than the cap-50 logits rather than on the bound alone.

#![cfg(feature = "flex")]

#[path = "support/backend_numerics.rs"]
mod backend_numerics;
#[path = "support/tiny_goldens.rs"]
mod tiny_goldens;

use backend_numerics::{Bound, Gap, host};
use burn::{backend::Flex, tensor::backend::Backend};
use serde_json::Value;
use tiny_goldens::{
    F32_TOLERANCE, Golden, all_logits, check_all_logits, check_cached, check_greedy,
    check_stateless, fixture, load, load_options,
};
use transformers::{
    AutoCausalConfig, AutoCausalLm, CausalLmInput, ForwardOptions,
    attention::AttentionPrecisionMode,
    cache::CacheConfig,
    checkpoint::LoadOptions,
    intervention::Hooks,
    models::{
        AutoCausalConfigRef,
        config::ModelConfig,
        gemma::{GemmaConfig, GemmaForCausalLm},
    },
};

/// Deterministic F32 CPU backend.
type B = Flex<f32, i32>;

/// The saturating attention soft-cap of [`saturated_attention_soft_cap`].
const SATURATED_CAP: f64 = 0.25;

/// Transformers 5.2.0 / PyTorch 2.10.0 logits of the `gemma2` fixture with `attn_logit_softcapping = 0.25`.
///
/// Row `r` holds the full-vocabulary logits at column 11 (the last token) of
/// golden row `r`, evaluated on the golden's `[2, 12]` input with its mask
/// and explicit positions. Generated by loading the fixture directory with
/// only that config field changed through
/// `AutoModelForCausalLM.from_pretrained(dir, dtype=torch.float32,
/// attn_implementation="eager")` (deterministic algorithms, one thread) and
/// printing `logits[r, 11]` with the shortest round-tripping F32 decimals.
#[rustfmt::skip]
const SATURATED_CAP_LOGITS: [[f32; 96]; 2] = [
    [
        0.080410786, -0.062165733, 1.1950735, -0.45232457, -0.024619656, -0.33439034, 0.49978557, -0.1891573,
        -0.18436584, 0.033353224, -0.6443288, 0.09398291, -0.16325302, -0.22073758, 0.4506666, -0.6690248,
        -0.39392787, -0.8471039, 0.9414215, 0.5986156, 0.23222503, 0.47232264, -0.6180521, -0.74616796,
        -0.027371837, 1.0451167, -0.006099058, -0.47767314, -1.1228348, 0.40746498, -0.037376944, -0.02927381,
        -0.31098926, 0.20272945, -0.4141247, 0.36145356, 0.37196916, -0.31649804, 0.2866434, 0.72156817,
        0.7614632, 0.6847492, -0.70724887, 0.18555884, 0.5461645, -0.0238588, -0.14263988, 0.5177211,
        -0.25995746, 0.061452705, -0.5277534, -0.32069612, -0.21989027, -0.22897817, 0.35718012, 0.2966191,
        0.7087778, 0.9837074, -1.232841, 0.16052036, -0.50823075, 0.31771716, -0.6735309, 0.92631406,
        -0.25065318, 0.5887374, -0.7883113, 0.34913623, -0.21094403, 0.0820058, 0.066152714, -0.64551693,
        0.27562442, -0.061180837, 0.32283133, 0.31036246, 0.77586323, -0.14035024, -0.19968393, 0.4668352,
        -0.5957396, 0.13775449, 0.21772295, 0.12502812, 0.11769687, 0.41633213, -0.17533368, -0.42765096,
        -0.6141539, 0.33403814, 0.40393946, 0.55848217, 0.44641858, -0.33351457, -0.1629649, -0.5812342,
    ],
    [
        0.7140069, -0.40306434, -0.16278343, 0.13483709, 0.14456432, -0.30637354, 0.56220025, -0.070171066,
        0.023996916, -0.68152833, 0.00026353978, 0.51946473, -0.66360384, 0.219646, -0.07103044, -0.6015403,
        -0.9624278, -0.7143337, 0.24519967, 0.07126393, -0.5209237, 0.1312301, -0.02722449, -0.5511985,
        0.2836748, 0.3200615, -0.13101953, 0.37003174, -0.6907697, -1.2379903, -0.122609854, 0.5019733,
        -0.6918362, 0.6004666, -0.16098423, 0.054331064, 0.5981049, 0.39023587, 0.5565224, -0.8889507,
        -0.67160743, 0.5670865, -0.42710406, 0.05835094, 0.4645384, -0.6256203, 0.5240668, 0.019135827,
        0.23917037, -0.6529421, 0.03728036, -0.8105051, 0.076176904, -0.39297122, 0.020733887, -0.19644517,
        0.2012139, 0.36606923, 0.23345484, -0.48791987, 0.21946216, -0.1603554, -0.7847105, -0.058060218,
        0.18972678, 0.11836044, 0.5492129, 0.2550795, -0.049434528, 0.15526673, 0.20383173, 0.2239104,
        -0.32822448, 0.12996183, -0.07564942, 0.16423233, 0.8100103, -0.68111694, -0.046970017, 0.73013294,
        -0.32614687, -0.41361576, 0.6690056, 0.43560174, -0.34155017, 0.31354046, -0.7725762, 0.07368493,
        0.090886384, -0.20643434, 0.6338207, -0.65476257, -0.6873444, 0.8476233, -0.076154575, 0.052333176,
    ],
];

/// Every Gemma fixture case and its text `model_type`.
const CASES: [(&str, &str); 2] = [("gemma2", "gemma2"), ("gemma3-text", "gemma3_text")];

/// A fresh directory for one modified copy of a fixture, removed when dropped.
///
/// Each call gets its own directory under Cargo's per-target temporary
/// directory, so the Flex and accelerator variants of a test never share
/// files.
fn scratch(name: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(name)
        .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
        .expect("create a fixture copy directory")
}

/// Stateless logits and hidden states match the reference, with explicit and implicit positions.
fn stateless_logits_and_hidden_states<A: Backend>(device: &A::Device, bound: Bound) {
    for (case, _) in CASES {
        let golden = Golden::case(case);
        let model = load::<A>(&fixture(case), device);
        check_stateless(&model, &golden, bound, device, case);
        // Implicit positions are the same valid-token ordinals.
        let implicit = model
            .forward(
                CausalLmInput {
                    position_ids: None,
                    ..golden.input(device)
                },
                &all_logits(),
            )
            .unwrap();
        check_all_logits(
            &golden,
            &host(implicit.logits),
            bound,
            &format!("{case} implicit"),
        );
    }
}

/// Batched ragged prefill and decode reproduce the stateless and reference logits.
fn batched_cached_decode<A: Backend>(device: &A::Device, bound: Bound) {
    for (case, _) in CASES {
        let golden = Golden::case(case);
        let model = load::<A>(&fixture(case), device);
        check_cached(&model, &golden, 16, bound, device, case);
    }
}

/// Multi-token ragged steps on a non-empty state reproduce the reference.
///
/// Row 0 prefills five tokens, then five more, then decodes; row 1
/// (left-padded by four) prefills only padding plus its first token, then
/// its next five tokens, so the two rows' new tokens sit at different
/// offsets of each step's block and the sliding layers' history plus fresh
/// segment exceed the window. Padding positions must return zero logits.
fn chunked_ragged_steps<A: Backend>(device: &A::Device, bound: Bound) {
    for (case, _) in CASES {
        let golden = Golden::case(case);
        let model = load::<A>(&fixture(case), device);
        let columns = golden.columns();
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 16).unwrap(), 2, device)
            .unwrap();
        let v = golden.vocabulary;
        let mut largest = Gap::default();
        for range in [0..5, 5..10] {
            let output = model
                .prefill(
                    golden.columns_input(range.clone(), device),
                    &mut state,
                    &all_logits(),
                )
                .unwrap();
            let logits = host(output.logits);
            let width = range.len();
            for row in 0..2 {
                for (offset, column) in range.clone().enumerate() {
                    let at = &logits[(row * width + offset) * v..(row * width + offset + 1) * v];
                    if golden.valid(row, column) {
                        largest = largest.max(bound.assert_close(
                            at,
                            golden.logits_at(row, column),
                            &format!("{case} chunk logits[{row}, {column}]"),
                        ));
                    } else {
                        assert!(at.iter().all(|&value| value == 0.0), "{case}: padding");
                    }
                }
            }
        }
        for column in 10..columns {
            let output = model
                .decode(
                    golden.step(column, device),
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap();
            let logits = host(output.logits);
            for row in 0..2 {
                largest = largest.max(bound.assert_close(
                    &logits[row * v..(row + 1) * v],
                    golden.logits_at(row, column),
                    &format!("{case} decode logits[{row}, {column}]"),
                ));
            }
        }
        eprintln!(
            "{case}: largest chunked-step logit gap {:.2e}",
            largest.deviation
        );
    }
}

/// The cached greedy continuation matches the reference for at least two decisive tokens.
fn cached_greedy_continuation<A: Backend>(device: &A::Device, bound: Bound) {
    for (case, _) in CASES {
        let golden = Golden::case(case);
        let model = load::<A>(&fixture(case), device);
        let compared = check_greedy(&model, &golden, bound, device, case);
        assert!(
            compared >= 2,
            "{case}: only {compared} decisive greedy tokens"
        );
    }
}

/// A hooked pass without actions exposes the catalog and equals the plain pass.
fn hooked_pass_without_actions<A: Backend>(device: &A::Device, bound: Bound) {
    for (case, _) in CASES {
        let golden = Golden::case(case);
        let model = load::<A>(&fixture(case), device);
        let input = CausalLmInput {
            position_ids: None,
            ..golden.input(device)
        };
        // Embedding, final norm, and twelve points per block.
        assert_eq!(
            model.activation_points().unwrap().len(),
            2 + 4 * 12,
            "{case}"
        );
        let hooked = model
            .forward_with_hooks(input, &all_logits(), &mut Hooks::new())
            .unwrap();
        check_all_logits(
            &golden,
            &host(hooked.logits),
            bound,
            &format!("{case} hooked"),
        );
    }
}

/// The RoPE mapping Transformers 5.2 writes for single-kind `layer_types` reproduces the reference.
///
/// `save_pretrained` adds `"rope_type": "default"` and `"rope_theta": null`
/// beside the per-type objects whenever `layer_types` uses one attention
/// kind. Transformers reads only the per-type objects; its logits for this
/// rewritten `gemma3-text` copy equal the golden bit for bit.
fn mixed_rope_mapping<A: Backend>(device: &A::Device, bound: Bound) {
    let source = fixture("gemma3-text");
    let directory = scratch("gemma3-text-mixed-rope");
    let mut config: Value =
        serde_json::from_str(&std::fs::read_to_string(source.join("config.json")).unwrap())
            .unwrap();
    config["rope_parameters"]["rope_type"] = "default".into();
    config["rope_parameters"]["rope_theta"] = Value::Null;
    std::fs::write(directory.path().join("config.json"), config.to_string()).unwrap();
    std::fs::copy(
        source.join("model.safetensors"),
        directory.path().join("model.safetensors"),
    )
    .unwrap();
    let golden = Golden::read(&source);
    let model = load::<A>(directory.path(), device);
    let logits = model
        .forward(golden.input(device), &all_logits())
        .unwrap()
        .logits;
    check_all_logits(&golden, &host(logits), bound, "mixed RoPE mapping");
}

/// The attention soft-cap matches Transformers where tanh saturates.
///
/// The `gemma2` weights are reloaded with `attn_logit_softcapping` changed
/// from 50 to [`SATURATED_CAP`]; raw scores reach `0.8`, so `s / c` reaches
/// `3.2`. Both the fused and the eager attention graphs must reproduce
/// [`SATURATED_CAP_LOGITS`], which differ from the cap-50 golden by at least
/// `1.7e-2` at the compared positions (`1.6e-1` at worst over all valid
/// positions), over a thousand times the F32 tolerance. The run must also be
/// nearer the saturated reference than the cap-50 logits: its gap must stay
/// below half that difference, which keeps the check decisive under the
/// looser TF32 bound.
fn saturated_attention_soft_cap<A: Backend>(device: &A::Device, bound: Bound) {
    let source = fixture("gemma2");
    let golden = Golden::read(&source);
    let mut config =
        GemmaConfig::from_json_str(&std::fs::read_to_string(source.join("config.json")).unwrap())
            .unwrap();
    config.attn_logit_softcapping = Some(SATURATED_CAP);
    let mut model = GemmaForCausalLm::<A>::init(config, device).unwrap();
    model
        .load_checkpoint(&source, &LoadOptions::default())
        .expect("strict load with the changed cap");
    let eager = model
        .clone()
        .with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    let (columns, vocabulary) = (golden.columns(), golden.vocabulary);
    for (graph, model) in [("fused", model), ("eager", eager)] {
        let logits = host(
            model
                .forward(golden.input(device), &all_logits())
                .unwrap()
                .logits,
        );
        for (row, expected) in SATURATED_CAP_LOGITS.iter().enumerate() {
            let start = (row * columns + columns - 1) * vocabulary;
            let actual = &logits[start..start + vocabulary];
            let gap = bound.assert_close(actual, expected, &format!("{graph} row {row}"));
            let moved = Gap::measure(expected, golden.logits_at(row, columns - 1)).deviation;
            eprintln!(
                "{graph} row {row}: max |Δ| = {:e}; cap 50 vs 0.25 = {moved:e}",
                gap.deviation
            );
            assert!(
                moved > 1e-2,
                "{graph} row {row}: the saturated cap is decisive"
            );
            assert!(
                2.0 * gap.deviation < moved,
                "{graph} row {row}: the run is nearer the cap-50 logits than the saturated reference"
            );
        }
    }
}

/// Writes the `gemma3-text` fixture as a multimodal `gemma3` checkpoint into a fresh directory.
///
/// The config becomes `{"model_type": "gemma3", "text_config": …}` with the
/// text config's `final_logit_softcapping` set to `cap`; every tensor moves
/// from `model.` to the current multimodal prefix `model.language_model.`,
/// and a one-tensor stand-in vision tower is added.
fn multimodal_copy(name: &str, cap: Option<f64>) -> tempfile::TempDir {
    use safetensors::{SafeTensors, tensor::TensorView};
    let source = fixture("gemma3-text");
    let directory = scratch(name);
    let mut text: Value =
        serde_json::from_str(&std::fs::read_to_string(source.join("config.json")).unwrap())
            .unwrap();
    text["final_logit_softcapping"] = serde_json::json!(cap);
    let wrapper = serde_json::json!({
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": text,
        "vision_config": {"model_type": "siglip_vision_model"},
    });
    std::fs::write(directory.path().join("config.json"), wrapper.to_string()).unwrap();
    let bytes = std::fs::read(source.join("model.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let vision = [0_u8; 8];
    let mut views: Vec<(String, TensorView<'_>)> = tensors
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let renamed = format!(
                "model.language_model.{}",
                name.strip_prefix("model.").unwrap()
            );
            (renamed, view)
        })
        .collect();
    views.push((
        VISION_STAND_IN.into(),
        TensorView::new(safetensors::Dtype::F32, vec![2], &vision).unwrap(),
    ));
    std::fs::write(
        directory.path().join("model.safetensors"),
        safetensors::serialize(views, None).unwrap(),
    )
    .unwrap();
    directory
}

/// The stand-in vision-tower tensor of [`multimodal_copy`].
const VISION_STAND_IN: &str = "model.vision_tower.vision_model.post_layernorm.weight";

/// The registry loads a multimodal wrapper's text tower and refuses a final soft-cap it would skip.
///
/// With a `null` cap the multimodal copy reproduces the text-only golden and
/// reports exactly its vision tensor as intentionally ignored. With a cap,
/// the load fails before any payload is read: `Gemma3ForConditionalGeneration`
/// never applies the final soft-cap, so executing it would silently diverge
/// from the checkpoint's own reference.
fn multimodal_wrappers<A: Backend>(device: &A::Device, bound: Bound) {
    let golden = Golden::case("gemma3-text");
    let directory = multimodal_copy("gemma3-multimodal", None);
    let loaded = AutoCausalLm::<A>::load_local(directory.path(), &load_options(), device)
        .expect("strict multimodal load");
    assert_eq!(loaded.load_report().allowlisted_ignored, [VISION_STAND_IN]);
    let model = loaded.into_model();
    assert_eq!(model.family(), AutoCausalConfig::Gemma);
    let logits = model
        .forward(golden.input(device), &all_logits())
        .unwrap()
        .logits;
    check_all_logits(&golden, &host(logits), bound, "multimodal wrapper");

    let capped = multimodal_copy("gemma3-multimodal-capped", Some(30.0));
    let Err(error) = AutoCausalLm::<A>::load_local(capped.path(), &load_options(), device) else {
        panic!("a multimodal final soft-cap must be refused");
    };
    assert!(
        error.to_string().contains("final_logit_softcapping"),
        "{error}"
    );
}

/// The fixtures load into the Gemma registry entry with their own `model_type`.
#[test]
fn fixtures_load_as_the_gemma_family() {
    for (case, model_type) in CASES {
        let model = load::<B>(&fixture(case), &Default::default());
        assert_eq!(model.family(), AutoCausalConfig::Gemma, "{case}");
        let AutoCausalConfigRef::Gemma(config) = model.config() else {
            panic!("{case}: not registered as the Gemma family");
        };
        assert_eq!(config.model_type, model_type);
        assert_eq!(model.config().vocabulary_size(), 96);
    }
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

/// Multi-token ragged steps on a non-empty state reproduce the reference.
#[test]
fn chunked_ragged_steps_match_the_reference() {
    chunked_ragged_steps::<B>(&Default::default(), F32_TOLERANCE);
}

/// The cached greedy continuation matches the reference at every decisive step.
#[test]
fn cached_greedy_continuation_matches_transformers() {
    cached_greedy_continuation::<B>(&Default::default(), F32_TOLERANCE);
}

/// A hooked pass without actions exposes the catalog and equals the plain pass.
#[test]
fn hooked_pass_without_actions_equals_the_plain_pass() {
    hooked_pass_without_actions::<B>(&Default::default(), F32_TOLERANCE);
}

/// The RoPE mapping Transformers 5.2 writes for single-kind `layer_types` reproduces the reference.
#[test]
fn transformers_5_2_mixed_rope_mapping_reproduces_the_reference() {
    mixed_rope_mapping::<B>(&Default::default(), F32_TOLERANCE);
}

/// The attention soft-cap matches Transformers where tanh saturates.
#[test]
fn saturated_attention_soft_cap_matches_transformers() {
    saturated_attention_soft_cap::<B>(&Default::default(), F32_TOLERANCE);
}

/// The registry loads a multimodal wrapper's text tower and refuses a final soft-cap it would skip.
#[test]
fn multimodal_wrappers_load_through_the_registry_without_a_final_soft_cap() {
    multimodal_wrappers::<B>(&Default::default(), F32_TOLERANCE);
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

    /// Metal/F32: batched ragged prefill and decode.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_batched_cached_decode_matches_reference_logits() {
        let device = Default::default();
        super::batched_cached_decode::<Metal>(&device, golden_bound::<Metal>(&device));
    }

    /// Metal/F32: multi-token ragged steps on a non-empty state.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_chunked_ragged_steps_match_the_reference() {
        let device = Default::default();
        super::chunked_ragged_steps::<Metal>(&device, golden_bound::<Metal>(&device));
    }

    /// Metal/F32: the cached greedy continuation.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_cached_greedy_continuation_matches_transformers() {
        let device = Default::default();
        super::cached_greedy_continuation::<Metal>(&device, golden_bound::<Metal>(&device));
    }

    /// Metal/F32: a hooked pass without actions.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_hooked_pass_without_actions_equals_the_plain_pass() {
        let device = Default::default();
        super::hooked_pass_without_actions::<Metal>(&device, golden_bound::<Metal>(&device));
    }

    /// Metal/F32: the Transformers 5.2 mixed RoPE mapping.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_mixed_rope_mapping_reproduces_the_reference() {
        let device = Default::default();
        super::mixed_rope_mapping::<Metal>(&device, golden_bound::<Metal>(&device));
    }

    /// Metal/F32: the saturated attention soft-cap.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_saturated_attention_soft_cap_matches_transformers() {
        let device = Default::default();
        super::saturated_attention_soft_cap::<Metal>(&device, golden_bound::<Metal>(&device));
    }

    /// Metal/F32: the multimodal wrapper's text tower.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_multimodal_wrappers_load_without_a_final_soft_cap() {
        let device = Default::default();
        super::multimodal_wrappers::<Metal>(&device, golden_bound::<Metal>(&device));
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

    /// CUDA: batched ragged prefill and decode.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_batched_cached_decode_matches_reference_logits() {
        let device = Default::default();
        super::batched_cached_decode::<Cuda>(&device, golden_bound::<Cuda>(&device));
    }

    /// CUDA: multi-token ragged steps on a non-empty state.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_chunked_ragged_steps_match_the_reference() {
        let device = Default::default();
        super::chunked_ragged_steps::<Cuda>(&device, golden_bound::<Cuda>(&device));
    }

    /// CUDA: the cached greedy continuation.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_cached_greedy_continuation_matches_transformers() {
        let device = Default::default();
        super::cached_greedy_continuation::<Cuda>(&device, golden_bound::<Cuda>(&device));
    }

    /// CUDA: a hooked pass without actions.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_hooked_pass_without_actions_equals_the_plain_pass() {
        let device = Default::default();
        super::hooked_pass_without_actions::<Cuda>(&device, golden_bound::<Cuda>(&device));
    }

    /// CUDA: the Transformers 5.2 mixed RoPE mapping.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_mixed_rope_mapping_reproduces_the_reference() {
        let device = Default::default();
        super::mixed_rope_mapping::<Cuda>(&device, golden_bound::<Cuda>(&device));
    }

    /// CUDA: the saturated attention soft-cap.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_saturated_attention_soft_cap_matches_transformers() {
        let device = Default::default();
        super::saturated_attention_soft_cap::<Cuda>(&device, golden_bound::<Cuda>(&device));
    }

    /// CUDA: the multimodal wrapper's text tower.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_multimodal_wrappers_load_without_a_final_soft_cap() {
        let device = Default::default();
        super::multimodal_wrappers::<Cuda>(&device, golden_bound::<Cuda>(&device));
    }
}
