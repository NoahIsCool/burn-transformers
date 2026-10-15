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
//! 4. the cached greedy continuation, compared token by token only while our
//!    top-2 logit margin exceeds the tolerance (a closer race is a tie at F32
//!    precision, not a parity failure).
//!
//! # Tolerance
//!
//! Both sides run F32 with different matmul and reduction orders; through
//! four blocks of width 32 the largest observed logit deviation over all
//! seven cases is `7.2e-7` (logits have magnitude about one; the stateless
//! test prints each case's gap with `--nocapture`). [`TOLERANCE`] is `1e-5`:
//! over ten times that noise, so other CPUs' SIMD and thread splits pass,
//! and far below any architectural error (a wrong norm placement, bias,
//! rotary layout, or window moves logits by `1e-2` or more).

#![cfg(feature = "flex")]

use std::path::{Path, PathBuf};

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use serde_json::Value;
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, DecodeInput, ForwardOptions, LogitsSelection,
    TokenMask, cache::CacheConfig,
};

/// Deterministic F32 CPU backend.
type B = Flex<f32, i32>;

/// Absolute tolerance on logits and hidden states; see the module documentation.
const TOLERANCE: f32 = 1e-5;

/// Columns prefilled before cached single-token decoding.
const PREFILL: usize = 6;

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

/// One parsed `golden.json`.
struct Golden {
    /// Hugging Face `model_type`.
    model_type: String,
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
            model_type: value["model_type"].as_str().unwrap().into(),
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

/// Returns the fixture directory of `case`.
fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/tiny_goldens/{case}"))
}

/// Loads `case` through the strict automatic loader.
fn load(case: &str) -> AutoCausalLm<B> {
    let options = AutoLoadOptions {
        load_tokenizer: false,
        ..AutoLoadOptions::default()
    };
    AutoCausalLm::<B>::load_local(fixture(case), &options, &Default::default())
        .unwrap_or_else(|error| panic!("{case}: strict load failed: {error}"))
        .into_model()
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

/// Stateless logits and hidden states match the reference at every valid position.
#[test]
fn stateless_logits_and_hidden_states_match_transformers() {
    for case in CASES {
        let golden = Golden::read(&fixture(case));
        let model = load(case);
        let output = model
            .forward(
                CausalLmInput {
                    token_ids: ints(&golden.input_ids),
                    token_mask: Some(mask(&golden.attention_mask)),
                    position_ids: Some(ints(&golden.position_ids)),
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    hidden_states: true,
                    ..ForwardOptions::default()
                },
            )
            .unwrap();
        let columns = golden.input_ids[0].len();
        let logits = host(output.logits);
        let mut largest = 0.0_f32;
        for row in 0..2 {
            for column in (0..columns).filter(|&column| golden.valid(row, column)) {
                let start = (row * columns + column) * golden.vocabulary;
                largest = largest.max(assert_close(
                    &logits[start..start + golden.vocabulary],
                    golden.logits_at(row, column),
                    &format!("{case} logits[{row}, {column}]"),
                ));
            }
        }
        eprintln!("{case}: largest stateless logit gap {largest:e}");
        let [entries, _, width] = golden.hidden_shape;
        let layers = entries - 1;
        assert_eq!(output.hidden_states.len(), layers + 2, "{case}");
        let reference_index = |ours: usize| if ours == layers + 1 { layers } else { ours };
        for (ours, state) in output.hidden_states.into_iter().enumerate() {
            if ours == layers {
                continue; // Transformers replaces the last block output by the final norm.
            }
            let row0 = host(state.narrow(0, 0, 1));
            let reference = reference_index(ours);
            let expected = &golden.hidden_states
                [reference * columns * width..(reference + 1) * columns * width];
            assert_close(&row0, expected, &format!("{case} hidden_states[{ours}]"));
        }
        assert_eq!(model.config().vocabulary_size(), golden.vocabulary);
        let transformers::models::AutoCausalConfigRef::Llama(config) = model.config() else {
            panic!("{case}: not registered as the Llama family");
        };
        assert_eq!(config.model_type, golden.model_type);
    }
}

/// Batched ragged prefill and decode reproduce the stateless and reference logits.
#[test]
fn batched_cached_decode_matches_stateless_and_reference_logits() {
    for case in CASES {
        let golden = Golden::read(&fixture(case));
        let model = load(case);
        let columns = golden.input_ids[0].len();
        let device = Default::default();
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 16).unwrap(), 2, &device)
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
        let v = golden.vocabulary;
        for (step, logits) in steps.iter().enumerate() {
            let column = PREFILL - 1 + step;
            for row in 0..2 {
                assert_close(
                    &logits[row * v..(row + 1) * v],
                    golden.logits_at(row, column),
                    &format!("{case} cached logits[{row}, {column}]"),
                );
            }
        }
        assert_eq!(
            transformers::DecodeState::lengths(&state),
            [columns, columns - 4],
            "{case}: cached lengths count valid tokens only"
        );
    }
}

/// The cached greedy continuation matches the reference while the choice is decisive.
#[test]
fn cached_greedy_continuation_matches_transformers() {
    for case in CASES {
        let golden = Golden::read(&fixture(case));
        let model = load(case);
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
            assert_eq!(token as i64, expected, "{case}: greedy token {step}");
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
}

/// The left-padded row's padding never enters the cache or shifts positions.
#[test]
fn padded_prefill_equals_unpadded_prefill_of_the_same_tokens() {
    let golden = Golden::read(&fixture("llama3"));
    let model = load("llama3");
    let device = Default::default();
    let row: Vec<i64> = golden.input_ids[1][4..].to_vec();
    let mut padded = model
        .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, &device)
        .unwrap();
    let mut plain = model
        .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, &device)
        .unwrap();
    let options = ForwardOptions::default();
    let with_padding = model
        .prefill(
            CausalLmInput {
                token_ids: ints(&[golden.input_ids[1].clone()]),
                token_mask: Some(mask(&[golden.attention_mask[1].clone()])),
                position_ids: None,
            },
            &mut padded,
            &options,
        )
        .unwrap();
    let without = model
        .prefill(
            CausalLmInput {
                token_ids: ints(&[row]),
                token_mask: None,
                position_ids: None,
            },
            &mut plain,
            &options,
        )
        .unwrap();
    assert_close(
        &host(with_padding.logits),
        &host(without.logits),
        "padded vs unpadded",
    );
}
