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
//! 4. the cached greedy continuation, compared token by token while our
//!    top-2 logit margin exceeds the tolerance;
//! 5. router diagnostics and the reference load-balancing objective.
//!
//! # Tolerance
//!
//! Both sides run F32 with different matmul and reduction orders. Through
//! four blocks of width 32 with routing, the largest observed logit
//! deviation is printed with `--nocapture` (about `1e-6`); [`TOLERANCE`] is
//! `1e-5`, far below any architectural error (a wrong sink, clamp, bias,
//! router order, rotary layout, or window moves logits by `1e-2` or more). A
//! flipped expert selection would also move logits by far more than the
//! tolerance, so passing proves every routing decision matches.

#![cfg(feature = "flex")]

use std::path::{Path, PathBuf};

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use serde_json::Value;
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, DecodeInput, ForwardOptions, LogitsSelection,
    TokenMask,
    attention::AttentionPrecisionMode,
    cache::CacheConfig,
    models::{
        AutoCausalConfigRef,
        gpt_oss::{GptOssConfig, GptOssForCausalLm},
    },
};

/// Deterministic F32 CPU backend.
type B = Flex<f32, i32>;

/// Absolute tolerance on logits and hidden states; see the module documentation.
const TOLERANCE: f32 = 1e-5;

/// Columns prefilled before cached single-token decoding.
const PREFILL: usize = 6;

/// The fixture case.
const CASE: &str = "gpt-oss";

/// One parsed `golden.json`.
struct Golden {
    /// `[2, 12]` token IDs.
    input_ids: Vec<Vec<i64>>,
    /// `[2, 12]` validity (1 = token).
    attention_mask: Vec<Vec<i64>>,
    /// `[2, 12]` explicit positions.
    position_ids: Vec<Vec<i64>>,
    /// Row-major `[2, 12, V]` logits.
    logits: Vec<f32>,
    /// Vocabulary size `V`.
    vocabulary: usize,
    /// Row-major `[L + 1, 12, d]` hidden states of row 0.
    hidden_states: Vec<f32>,
    /// `[L + 1, 12, d]`.
    hidden_shape: [usize; 3],
    /// Greedy prompt tokens.
    greedy_prompt: Vec<i64>,
    /// Reference greedy continuation.
    greedy_tokens: Vec<i64>,
}

impl Golden {
    /// Reads `<root>/golden.json`.
    fn read(root: &Path) -> Self {
        let text = std::fs::read_to_string(root.join("golden.json")).expect("read golden.json");
        let value: Value = serde_json::from_str(&text).expect("parse golden.json");
        assert_eq!(value["model_type"], "gpt_oss");
        let matrix = |key: &str| -> Vec<Vec<i64>> {
            serde_json::from_value(value[key].clone()).expect("integer matrix")
        };
        let floats = |key: &str| -> Vec<f32> {
            serde_json::from_value::<Vec<f64>>(value[key].clone())
                .expect("float list")
                .into_iter()
                .map(|value| value as f32)
                .collect()
        };
        let shape: Vec<usize> = serde_json::from_value(value["logits_shape"].clone()).unwrap();
        let hidden_shape: Vec<usize> =
            serde_json::from_value(value["hidden_states_row0_shape"].clone()).unwrap();
        Self {
            input_ids: matrix("input_ids"),
            attention_mask: matrix("attention_mask"),
            position_ids: matrix("position_ids"),
            logits: floats("logits"),
            vocabulary: shape[2],
            hidden_states: floats("hidden_states_row0"),
            hidden_shape: hidden_shape.try_into().unwrap(),
            greedy_prompt: serde_json::from_value(value["greedy_prompt"].clone()).unwrap(),
            greedy_tokens: serde_json::from_value(value["greedy_tokens"].clone()).unwrap(),
        }
    }

    /// Returns the reference logits of `(row, column)`.
    fn logits_at(&self, row: usize, column: usize) -> &[f32] {
        let columns = self.input_ids[0].len();
        let start = (row * columns + column) * self.vocabulary;
        &self.logits[start..start + self.vocabulary]
    }

    /// Returns whether `(row, column)` is a real token.
    fn valid(&self, row: usize, column: usize) -> bool {
        self.attention_mask[row][column] != 0
    }
}

/// Returns the fixture directory.
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/tiny_goldens/{CASE}"))
}

/// Loads the fixture through the strict automatic loader.
fn load() -> AutoCausalLm<B> {
    let options = AutoLoadOptions {
        load_tokenizer: false,
        ..AutoLoadOptions::default()
    };
    AutoCausalLm::<B>::load_local(fixture(), &options, &Default::default())
        .unwrap_or_else(|error| panic!("strict load failed: {error}"))
        .into_model()
}

/// Loads the fixture as the concrete family model.
fn load_concrete() -> GptOssForCausalLm<B> {
    match load() {
        AutoCausalLm::GptOss(model) => model,
        _ => panic!("gpt_oss is not registered as the gpt-oss family"),
    }
}

/// Uploads a row-major integer matrix.
fn ints(rows: &[Vec<i64>]) -> Tensor<B, 2, Int> {
    let shape = [rows.len(), rows[0].len()];
    Tensor::from_data(TensorData::new(rows.concat(), shape), &Default::default())
}

/// Uploads a validity matrix as a token mask.
fn mask(rows: &[Vec<i64>]) -> TokenMask<B> {
    TokenMask::new(ints(rows).equal_elem(1))
}

/// Reads an F32 tensor to the host.
fn host<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

/// Asserts `|actual - expected| <= TOLERANCE` elementwise and returns the largest gap.
fn assert_close(actual: &[f32], expected: &[f32], context: &str) -> f32 {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    let gap = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(gap <= TOLERANCE, "{context}: max |Δ| = {gap:e}");
    gap
}

/// Index and margin of the largest logit.
fn top_two(logits: &[f32]) -> (usize, f32) {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    (order[0], logits[order[0]] - logits[order[1]])
}

/// The full golden input with explicit positions.
fn golden_input(golden: &Golden) -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: ints(&golden.input_ids),
        token_mask: Some(mask(&golden.attention_mask)),
        position_ids: Some(ints(&golden.position_ids)),
    }
}

/// Compares stateless logits at every valid position and returns the largest gap.
fn assert_logits_match(golden: &Golden, logits: Vec<f32>, context: &str) -> f32 {
    let columns = golden.input_ids[0].len();
    let mut largest = 0.0_f32;
    for row in 0..2 {
        for column in (0..columns).filter(|&column| golden.valid(row, column)) {
            let start = (row * columns + column) * golden.vocabulary;
            largest = largest.max(assert_close(
                &logits[start..start + golden.vocabulary],
                golden.logits_at(row, column),
                &format!("{context} logits[{row}, {column}]"),
            ));
        }
    }
    largest
}

/// Stateless logits and hidden states match the reference at every valid position.
#[test]
fn stateless_logits_and_hidden_states_match_transformers() {
    let golden = Golden::read(&fixture());
    let model = load();
    let AutoCausalConfigRef::GptOss(config) = model.config() else {
        panic!("not registered as the gpt-oss family");
    };
    assert_eq!(config.model_type, "gpt_oss");
    assert_eq!(model.config().vocabulary_size(), golden.vocabulary);
    let output = model
        .forward(
            golden_input(&golden),
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    let largest = assert_logits_match(&golden, host(output.logits), "fused");
    eprintln!("gpt-oss: largest stateless logit gap {largest:e}");
    let [entries, columns, width] = golden.hidden_shape;
    let layers = entries - 1;
    assert_eq!(output.hidden_states.len(), layers + 2);
    let reference_index = |ours: usize| if ours == layers + 1 { layers } else { ours };
    for (ours, state) in output.hidden_states.into_iter().enumerate() {
        if ours == layers {
            continue; // Transformers replaces the last block output by the final norm.
        }
        let row0 = host(state.narrow(0, 0, 1));
        let reference = reference_index(ours);
        let expected =
            &golden.hidden_states[reference * columns * width..(reference + 1) * columns * width];
        assert_close(&row0, expected, &format!("hidden_states[{ours}]"));
    }
}

/// The eager F32-softmax graph (explicit sink column) matches the reference too.
#[test]
fn eager_sink_attention_matches_transformers() {
    let golden = Golden::read(&fixture());
    let model = load_concrete().with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    let output = model
        .forward(
            golden_input(&golden),
            &ForwardOptions {
                logits: LogitsSelection::All,
                attentions: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    let largest = assert_logits_match(&golden, host(output.logits), "eager");
    eprintln!("gpt-oss: largest eager logit gap {largest:e}");
    // With sinks every pattern row sums to 1 - p_sink < 1, and never below zero.
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

/// Runs a batched prefill of the first six columns and six one-token decode steps.
fn cached_logits(model: &AutoCausalLm<B>, golden: &Golden, capacity: usize) -> Vec<Vec<f32>> {
    let columns = golden.input_ids[0].len();
    let device = Default::default();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, capacity).unwrap(), 2, &device)
        .unwrap();
    let prefix = |matrix: &[Vec<i64>], range: std::ops::Range<usize>| {
        matrix
            .iter()
            .map(|row| row[range.clone()].to_vec())
            .collect::<Vec<_>>()
    };
    let prefill = model
        .prefill(
            CausalLmInput {
                token_ids: ints(&prefix(&golden.input_ids, 0..PREFILL)),
                token_mask: Some(mask(&prefix(&golden.attention_mask, 0..PREFILL))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let mut steps = vec![host(prefill.logits)];
    for column in PREFILL..columns {
        let output = model
            .decode(
                DecodeInput {
                    token_ids: ints(&prefix(&golden.input_ids, column..column + 1)),
                    position_ids: None,
                    token_mask: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        steps.push(host(output.logits));
    }
    assert_eq!(
        transformers::DecodeState::lengths(&state),
        [columns, columns - 4],
        "cached lengths count valid tokens only"
    );
    steps
}

/// Batched ragged prefill and decode reproduce the reference logits, with ring and full caches.
#[test]
fn batched_cached_decode_matches_reference_logits() {
    let golden = Golden::read(&fixture());
    let model = load();
    // Both capacities exceed the window, so the window-4 layers use a
    // wrapping ring; capacity 12 is exactly filled by the longer row.
    for capacity in [16, 12] {
        let steps = cached_logits(&model, &golden, capacity);
        let v = golden.vocabulary;
        for (step, logits) in steps.iter().enumerate() {
            let column = PREFILL - 1 + step;
            for row in 0..2 {
                assert_close(
                    &logits[row * v..(row + 1) * v],
                    golden.logits_at(row, column),
                    &format!("capacity {capacity}: cached logits[{row}, {column}]"),
                );
            }
        }
    }
}

/// The cached greedy continuation matches the reference while the choice is decisive.
#[test]
fn cached_greedy_continuation_matches_transformers() {
    let golden = Golden::read(&fixture());
    let model = load();
    let device = Default::default();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, &device)
        .unwrap();
    let mut logits = host(
        model
            .prefill(
                CausalLmInput {
                    token_ids: ints(std::slice::from_ref(&golden.greedy_prompt)),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits,
    );
    for (step, &expected) in golden.greedy_tokens.iter().enumerate() {
        let (token, margin) = top_two(&logits);
        if margin <= TOLERANCE {
            break; // A tie at F32 precision; later tokens may legitimately differ.
        }
        assert_eq!(token as i64, expected, "greedy token {step}");
        logits = host(
            model
                .decode(
                    DecodeInput {
                        token_ids: ints(&[vec![expected]]),
                        position_ids: None,
                        token_mask: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap()
                .logits,
        );
    }
}

/// Router diagnostics and the load-balancing objective are returned on request.
#[test]
fn router_logits_and_load_balancing_loss_are_reported() {
    let golden = Golden::read(&fixture());
    let model = load();
    let output = model
        .forward(
            golden_input(&golden),
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
    // E · Σ_{j,e} f_{j,e} p_e lies in (0, E · K]; uniform routing gives K = 2.
    assert!(loss[0] > 0.0 && loss[0] <= 8.0, "{loss:?}");
}

/// The fixture's configuration round-trips through the schema.
#[test]
fn fixture_config_round_trips() {
    let text = std::fs::read_to_string(fixture().join("config.json")).unwrap();
    let config: GptOssConfig = serde_json::from_str(&text).unwrap();
    let again: GptOssConfig =
        serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
    assert_eq!(config, again);
    let architecture = config.architecture().unwrap();
    assert_eq!(architecture.sliding_window(), Some(4));
    assert_eq!(architecture.num_layers(), 4);
}
