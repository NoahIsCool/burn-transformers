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
//! 5. the cached greedy continuation, compared token by token only while
//!    our top-2 logit margin exceeds the tolerance;
//! 6. a hooked pass with no actions, which must equal the plain pass.
//!
//! # Tolerance
//!
//! Both sides run F32 with different matmul and reduction orders. The
//! largest observed logit deviation is printed by the stateless test with
//! `--nocapture`: `1.5e-6` for `gemma2` and `7.5e-7` for `gemma3-text` on
//! Flex. [`TOLERANCE`] is `1e-5`, as for the Llama family: over six times
//! that noise, and far below any architectural error (a missing `1 + w`, an
//! unscaled embedding, a wrong RoPE base, window, or soft-cap moves logits
//! by `1e-2` or more).

#![cfg(feature = "flex")]

use std::path::{Path, PathBuf};

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use serde_json::Value;
use transformers::{
    AutoCausalConfig, AutoCausalLm, AutoLoadOptions, CausalLmInput, DecodeInput, ForwardOptions,
    LogitsSelection, TokenMask, cache::CacheConfig, intervention::Hooks,
    models::AutoCausalConfigRef,
};

/// Deterministic F32 CPU backend.
type B = Flex<f32, i32>;

/// Absolute tolerance on logits and hidden states; see the module documentation.
const TOLERANCE: f32 = 1e-5;

/// Columns prefilled before cached single-token decoding.
const PREFILL: usize = 6;

/// Every Gemma fixture case and its text `model_type`.
const CASES: [(&str, &str); 2] = [("gemma2", "gemma2"), ("gemma3-text", "gemma3_text")];

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

    /// Returns the number of columns per row.
    fn columns(&self) -> usize {
        self.input_ids[0].len()
    }

    /// Returns the reference logits of `(row, column)`.
    fn logits_at(&self, row: usize, column: usize) -> &[f32] {
        let start = (row * self.columns() + column) * self.vocabulary;
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

/// Returns columns `range` of every row.
fn slice(matrix: &[Vec<i64>], range: std::ops::Range<usize>) -> Vec<Vec<i64>> {
    matrix
        .iter()
        .map(|row| row[range.clone()].to_vec())
        .collect()
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

/// Checks `[2, 12, V]` logits at every valid position against the golden.
fn check_all_logits(golden: &Golden, logits: &[f32], context: &str) -> f32 {
    let (columns, vocabulary) = (golden.columns(), golden.vocabulary);
    let mut largest = 0.0_f32;
    for row in 0..2 {
        for column in (0..columns).filter(|&column| golden.valid(row, column)) {
            let start = (row * columns + column) * vocabulary;
            largest = largest.max(assert_close(
                &logits[start..start + vocabulary],
                golden.logits_at(row, column),
                &format!("{context} logits[{row}, {column}]"),
            ));
        }
    }
    largest
}

/// The fixtures load into the Gemma registry entry with their own `model_type`.
#[test]
fn fixtures_load_as_the_gemma_family() {
    for (case, model_type) in CASES {
        let model = load(case);
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
    for (case, _) in CASES {
        let golden = Golden::read(&fixture(case));
        let model = load(case);
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            hidden_states: true,
            ..ForwardOptions::default()
        };
        let output = model
            .forward(
                CausalLmInput {
                    token_ids: ints(&golden.input_ids),
                    token_mask: Some(mask(&golden.attention_mask)),
                    position_ids: Some(ints(&golden.position_ids)),
                },
                &options,
            )
            .unwrap();
        let largest = check_all_logits(&golden, &host(output.logits), case);
        eprintln!("{case}: largest stateless logit gap {largest:e}");
        let columns = golden.columns();
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
        // Implicit positions are the same valid-token ordinals.
        let implicit = model
            .forward(
                CausalLmInput {
                    token_ids: ints(&golden.input_ids),
                    token_mask: Some(mask(&golden.attention_mask)),
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..ForwardOptions::default()
                },
            )
            .unwrap();
        check_all_logits(&golden, &host(implicit.logits), &format!("{case} implicit"));
    }
}

/// Batched ragged prefill and decode reproduce the stateless and reference logits.
#[test]
fn batched_cached_decode_matches_stateless_and_reference_logits() {
    for (case, _) in CASES {
        let golden = Golden::read(&fixture(case));
        let model = load(case);
        let columns = golden.columns();
        let device = Default::default();
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 16).unwrap(), 2, &device)
            .unwrap();
        let prefill = model
            .prefill(
                CausalLmInput {
                    token_ids: ints(&slice(&golden.input_ids, 0..PREFILL)),
                    token_mask: Some(mask(&slice(&golden.attention_mask, 0..PREFILL))),
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
                        token_ids: ints(&slice(&golden.input_ids, column..column + 1)),
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

/// Multi-token ragged steps on a non-empty state reproduce the reference.
///
/// Row 0 prefills three tokens, then five, then decodes; row 1 (left-padded
/// by four) prefills only padding plus its first token, then the next five
/// tokens with its padding columns interleaved differently, so the two
/// rows' new tokens sit at different offsets of each step's block and the
/// sliding layers' history plus fresh segment exceed the window.
#[test]
fn chunked_ragged_steps_match_the_reference() {
    for (case, _) in CASES {
        let golden = Golden::read(&fixture(case));
        let model = load(case);
        let columns = golden.columns();
        let device = Default::default();
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 16).unwrap(), 2, &device)
            .unwrap();
        let all = ForwardOptions {
            logits: LogitsSelection::All,
            ..ForwardOptions::default()
        };
        let v = golden.vocabulary;
        for range in [0..5, 5..10] {
            let output = model
                .prefill(
                    CausalLmInput {
                        token_ids: ints(&slice(&golden.input_ids, range.clone())),
                        token_mask: Some(mask(&slice(&golden.attention_mask, range.clone()))),
                        position_ids: None,
                    },
                    &mut state,
                    &all,
                )
                .unwrap();
            let logits = host(output.logits);
            let width = range.len();
            for row in 0..2 {
                for (offset, column) in range.clone().enumerate() {
                    let at = &logits[(row * width + offset) * v..(row * width + offset + 1) * v];
                    if golden.valid(row, column) {
                        assert_close(
                            at,
                            golden.logits_at(row, column),
                            &format!("{case} chunk logits[{row}, {column}]"),
                        );
                    } else {
                        assert!(at.iter().all(|&value| value == 0.0), "{case}: padding");
                    }
                }
            }
        }
        for column in 10..columns {
            let output = model
                .decode(
                    DecodeInput {
                        token_ids: ints(&slice(&golden.input_ids, column..column + 1)),
                        position_ids: None,
                        token_mask: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap();
            let logits = host(output.logits);
            for row in 0..2 {
                assert_close(
                    &logits[row * v..(row + 1) * v],
                    golden.logits_at(row, column),
                    &format!("{case} decode logits[{row}, {column}]"),
                );
            }
        }
    }
}

/// The cached greedy continuation matches the reference while the choice is decisive.
#[test]
fn cached_greedy_continuation_matches_transformers() {
    for (case, _) in CASES {
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
        let mut compared = 0;
        for (step, &expected) in golden.greedy_tokens.iter().enumerate() {
            let (token, margin) = top_two(&logits);
            if margin <= TOLERANCE {
                break; // A tie at F32 precision; later tokens may legitimately differ.
            }
            assert_eq!(token as i64, expected, "{case}: greedy token {step}");
            compared += 1;
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
        assert!(
            compared >= 2,
            "{case}: only {compared} decisive greedy tokens"
        );
    }
}

/// A hooked pass without actions exposes the catalog and equals the plain pass.
#[test]
fn hooked_pass_without_actions_equals_the_plain_pass() {
    for (case, _) in CASES {
        let golden = Golden::read(&fixture(case));
        let model = load(case);
        let input = || CausalLmInput {
            token_ids: ints(&golden.input_ids),
            token_mask: Some(mask(&golden.attention_mask)),
            position_ids: None,
        };
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            ..ForwardOptions::default()
        };
        // Embedding, final norm, and twelve points per block.
        assert_eq!(
            model.activation_points().unwrap().len(),
            2 + 4 * 12,
            "{case}"
        );
        let hooked = model
            .forward_with_hooks(input(), &options, &mut Hooks::new())
            .unwrap();
        check_all_logits(&golden, &host(hooked.logits), &format!("{case} hooked"));
    }
}
