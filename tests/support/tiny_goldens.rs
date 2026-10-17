//! Fixture access and backend-generic checks shared by the tiny Transformers golden suites.
//!
//! `tests/tiny_goldens_llama.rs`, `tests/tiny_goldens_gemma.rs`, and
//! `tests/tiny_goldens_gpt_oss.rs` compare seeded tiny checkpoints with
//! goldens that `scripts/generate_tiny_goldens.py` wrote from Transformers
//! 5.2.0 and PyTorch 2.10.0 (eager attention, CPU, deterministic). Every
//! golden records two rows of 12 tokens (row 1 left-padded by 4) with
//! explicit `cumsum(mask) - 1` positions, their full-vocabulary logits, row
//! 0's per-layer hidden states, and a cached greedy continuation.
//!
//! Every check here takes the backend as a type parameter and a [`Bound`],
//! so the suites run one set of assertions on Flex (ordinary tests) and in
//! ignored Metal and CUDA modules. Flex uses [`F32_TOLERANCE`];
//! [`golden_bound`] picks an accelerator's bound from its measured matrix
//! precision: [`ACCELERATOR_F32_TOLERANCE`] for an F32 device, the TF32
//! bound of `backend_numerics` for a CUDA GPU that rounds F32 matrix inputs
//! to TF32.
//!
//! # Tolerance
//!
//! Flex and Transformers both run F32 with different matmul and reduction
//! orders; through four blocks of width 32 the largest observed logit
//! deviation is about `1e-6` (logits have magnitude below `1.8`), and
//! [`F32_TOLERANCE`], `1e-5` absolute, is over six times every case's noise
//! while far below any architectural error (a wrong norm placement, bias,
//! rotary layout, window, sink, or routing decision moves logits by `1e-2`
//! or more). Flex's hidden states, which reach magnitude 8, also stay within
//! it (`3.3e-6`).
//!
//! An F32 accelerator reduces in yet another order, and its noise grows with
//! the compared magnitude: Metal's logits stay within `3.6e-6`, but one
//! `gemma2` hidden state of magnitude `7.0` deviates by `1.14e-5`, a relative
//! `1.6e-6` that an absolute `1e-5` cannot hold. Metal, and CUDA GPUs that
//! multiply F32 matrices in F32, are therefore held to
//! [`ACCELERATOR_F32_TOLERANCE`]: `1e-5` plus `1e-5` of the compared set's
//! largest magnitude, six times that relative deviation and still three
//! orders of magnitude below an architectural error. Each suite records its
//! measured Metal deviations. A TF32 CUDA run is held to the TF32 bound
//! (`1e-2` plus `1e-2` of the largest magnitude), which qualifies execution
//! on that GPU but not F32 numerics.
//!
//! The greedy check feeds the reference's tokens, so every step sees the
//! reference's prefix; a step whose top-2 margin lies within the bound's
//! allowance is a tie at the backend's precision and is skipped rather than
//! ending the comparison.

#![allow(
    dead_code,
    reason = "each tiny-golden suite includes this module and uses a subset"
)]

use std::path::{Path, PathBuf};

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use serde_json::Value;
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, DecodeInput, DecodeState, ForwardOptions,
    LogitsSelection, TokenMask, cache::CacheConfig,
};

use crate::backend_numerics::{Bound, Gap, MatmulPrecision, host};

/// Tolerance of Flex against the goldens: `1e-5` absolute.
///
/// See the module documentation for its justification.
pub(crate) const F32_TOLERANCE: Bound = Bound {
    absolute: 1e-5,
    relative: 0.0,
};

/// Tolerance of an F32 accelerator against the goldens: `1e-5` plus `1e-5` of the compared magnitude.
///
/// See the module documentation for its justification.
pub(crate) const ACCELERATOR_F32_TOLERANCE: Bound = Bound {
    absolute: 1e-5,
    relative: 1e-5,
};

/// Columns prefilled before cached single-token decoding.
pub(crate) const PREFILL: usize = 6;

/// Leading padding columns of the golden's second row.
pub(crate) const PADDING: usize = 4;

/// Returns an accelerator's golden bound after probing how it multiplies F32 matrices.
///
/// An F32 device gets [`ACCELERATOR_F32_TOLERANCE`], a TF32 one the TF32
/// bound; the choice is printed so a run's log records it.
pub(crate) fn golden_bound<B: Backend>(device: &B::Device) -> Bound {
    let precision = MatmulPrecision::probe::<B>(device);
    let bound = precision.bound(ACCELERATOR_F32_TOLERANCE);
    eprintln!("F32 matrix products use {precision:?} inputs; golden bound {bound:?}");
    bound
}

/// One parsed `golden.json`.
pub(crate) struct Golden {
    /// Hugging Face `model_type`.
    pub(crate) model_type: String,
    /// `[2, 12]` token IDs.
    pub(crate) input_ids: Vec<Vec<i64>>,
    /// `[2, 12]` validity (1 = token).
    pub(crate) attention_mask: Vec<Vec<i64>>,
    /// `[2, 12]` explicit positions.
    pub(crate) position_ids: Vec<Vec<i64>>,
    /// Row-major `[2, 12, V]` logits.
    pub(crate) logits: Vec<f32>,
    /// Vocabulary size `V`.
    pub(crate) vocabulary: usize,
    /// Row-major `[L + 1, 12, d]` hidden states of row 0.
    pub(crate) hidden_states: Vec<f32>,
    /// `[L + 1, 12, d]`.
    pub(crate) hidden_shape: [usize; 3],
    /// Greedy prompt tokens.
    pub(crate) greedy_prompt: Vec<i64>,
    /// Reference greedy continuation.
    pub(crate) greedy_tokens: Vec<i64>,
}

impl Golden {
    /// Reads `<root>/golden.json`.
    pub(crate) fn read(root: &Path) -> Self {
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

    /// Reads the golden of fixture `case`.
    pub(crate) fn case(case: &str) -> Self {
        Self::read(&fixture(case))
    }

    /// Returns the number of columns per row.
    pub(crate) fn columns(&self) -> usize {
        self.input_ids[0].len()
    }

    /// Returns the reference logits of `(row, column)`.
    pub(crate) fn logits_at(&self, row: usize, column: usize) -> &[f32] {
        let start = (row * self.columns() + column) * self.vocabulary;
        &self.logits[start..start + self.vocabulary]
    }

    /// Returns whether `(row, column)` is a real token.
    pub(crate) fn valid(&self, row: usize, column: usize) -> bool {
        self.attention_mask[row][column] != 0
    }

    /// The full `[2, 12]` input with its mask and explicit positions.
    pub(crate) fn input<B: Backend>(&self, device: &B::Device) -> CausalLmInput<B> {
        CausalLmInput {
            token_ids: ints(&self.input_ids, device),
            token_mask: Some(mask(&self.attention_mask, device)),
            position_ids: Some(ints(&self.position_ids, device)),
        }
    }

    /// Columns `range` of both rows with their mask and implicit positions.
    pub(crate) fn columns_input<B: Backend>(
        &self,
        range: std::ops::Range<usize>,
        device: &B::Device,
    ) -> CausalLmInput<B> {
        CausalLmInput {
            token_ids: ints(&slice(&self.input_ids, range.clone()), device),
            token_mask: Some(mask(&slice(&self.attention_mask, range), device)),
            position_ids: None,
        }
    }

    /// Column `column` of both rows as one unmasked decode step.
    pub(crate) fn step<B: Backend>(&self, column: usize, device: &B::Device) -> DecodeInput<B> {
        DecodeInput {
            token_ids: ints(&slice(&self.input_ids, column..column + 1), device),
            position_ids: None,
            token_mask: None,
        }
    }
}

/// Returns the fixture directory of `case`.
pub(crate) fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/tiny_goldens/{case}"))
}

/// Strict automatic-loader options without a tokenizer.
pub(crate) fn load_options() -> AutoLoadOptions {
    AutoLoadOptions {
        load_tokenizer: false,
        ..AutoLoadOptions::default()
    }
}

/// Loads a checkpoint directory through the strict automatic loader on `device`.
pub(crate) fn load<B: Backend>(directory: &Path, device: &B::Device) -> AutoCausalLm<B> {
    AutoCausalLm::<B>::load_local(directory, &load_options(), device)
        .unwrap_or_else(|error| panic!("{}: strict load failed: {error}", directory.display()))
        .into_model()
}

/// Uploads a row-major integer matrix.
pub(crate) fn ints<B: Backend>(rows: &[Vec<i64>], device: &B::Device) -> Tensor<B, 2, Int> {
    let shape = [rows.len(), rows[0].len()];
    Tensor::from_data(TensorData::new(rows.concat(), shape), device)
}

/// Uploads a validity matrix as a token mask.
pub(crate) fn mask<B: Backend>(rows: &[Vec<i64>], device: &B::Device) -> TokenMask<B> {
    TokenMask::new(ints::<B>(rows, device).equal_elem(1))
}

/// Returns columns `range` of every row.
pub(crate) fn slice(matrix: &[Vec<i64>], range: std::ops::Range<usize>) -> Vec<Vec<i64>> {
    matrix
        .iter()
        .map(|row| row[range.clone()].to_vec())
        .collect()
}

/// Index and margin of the largest logit.
pub(crate) fn top_two(logits: &[f32]) -> (usize, f32) {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    (order[0], logits[order[0]] - logits[order[1]])
}

/// Options requesting the logits of every position.
pub(crate) fn all_logits() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    }
}

/// Checks `[2, 12, V]` logits at every valid position against the golden; returns the largest gap.
#[track_caller]
pub(crate) fn check_all_logits(
    golden: &Golden,
    logits: &[f32],
    bound: Bound,
    context: &str,
) -> Gap {
    let (columns, vocabulary) = (golden.columns(), golden.vocabulary);
    let mut largest = Gap::default();
    for row in 0..2 {
        for column in (0..columns).filter(|&column| golden.valid(row, column)) {
            let start = (row * columns + column) * vocabulary;
            largest = largest.max(bound.assert_close(
                &logits[start..start + vocabulary],
                golden.logits_at(row, column),
                &format!("{context} logits[{row}, {column}]"),
            ));
        }
    }
    largest
}

/// Checks a stateless forward with hidden states: logits at every valid position and row 0's states.
///
/// Transformers reports the embedding, each block output but the last, and
/// the final-normalized states; this crate reports every block output and
/// then the final-normalized states, so its last block output is skipped.
/// Prints and returns the largest logit gap.
pub(crate) fn check_stateless<B: Backend>(
    model: &AutoCausalLm<B>,
    golden: &Golden,
    bound: Bound,
    device: &B::Device,
    context: &str,
) -> Gap {
    let output = model
        .forward(
            golden.input(device),
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    let largest = check_all_logits(golden, &host(output.logits), bound, context);
    let columns = golden.columns();
    let [entries, _, width] = golden.hidden_shape;
    let layers = entries - 1;
    assert_eq!(output.hidden_states.len(), layers + 2, "{context}");
    let reference_index = |ours: usize| if ours == layers + 1 { layers } else { ours };
    let mut hidden = Gap::default();
    for (ours, state) in output.hidden_states.into_iter().enumerate() {
        if ours == layers {
            continue; // Transformers replaces the last block output by the final norm.
        }
        let row0 = host(state.narrow(0, 0, 1));
        let reference = reference_index(ours);
        let expected =
            &golden.hidden_states[reference * columns * width..(reference + 1) * columns * width];
        hidden = hidden.max(bound.assert_close(
            &row0,
            expected,
            &format!("{context} hidden_states[{ours}]"),
        ));
    }
    eprintln!(
        "{context}: largest stateless logit gap {:.2e}, hidden-state gap {:.2e} (scale {:.2})",
        largest.deviation, hidden.deviation, hidden.scale
    );
    largest
}

/// Runs a batched prefill of the first [`PREFILL`] columns and one-token decode steps to the end.
///
/// Returns the `[2, V]` logits of the prefill and of every step; the rows
/// then hold 12 and 8 cached tokens, which the decode state must report.
pub(crate) fn cached_logits<B: Backend>(
    model: &AutoCausalLm<B>,
    golden: &Golden,
    capacity: usize,
    device: &B::Device,
) -> Vec<Vec<f32>> {
    let columns = golden.columns();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, capacity).unwrap(), 2, device)
        .unwrap();
    let prefill = model
        .prefill(
            golden.columns_input(0..PREFILL, device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let mut steps = vec![host(prefill.logits)];
    for column in PREFILL..columns {
        let output = model
            .decode(
                golden.step(column, device),
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        steps.push(host(output.logits));
    }
    assert_eq!(
        DecodeState::lengths(&state),
        [columns, columns - PADDING],
        "cached lengths count valid tokens only"
    );
    steps
}

/// Checks the cached logits of [`cached_logits`] against the golden at every step.
///
/// Prints and returns the largest gap.
pub(crate) fn check_cached<B: Backend>(
    model: &AutoCausalLm<B>,
    golden: &Golden,
    capacity: usize,
    bound: Bound,
    device: &B::Device,
    context: &str,
) -> Gap {
    let vocabulary = golden.vocabulary;
    let mut largest = Gap::default();
    for (step, logits) in cached_logits(model, golden, capacity, device)
        .iter()
        .enumerate()
    {
        let column = PREFILL - 1 + step;
        for row in 0..2 {
            largest = largest.max(bound.assert_close(
                &logits[row * vocabulary..(row + 1) * vocabulary],
                golden.logits_at(row, column),
                &format!("{context} cached logits[{row}, {column}]"),
            ));
        }
    }
    eprintln!(
        "{context}: largest cached logit gap {:.2e} (capacity {capacity})",
        largest.deviation
    );
    largest
}

/// Compares the cached greedy continuation at every decisive step.
///
/// The reference's tokens are fed back, so each step's logits continue the
/// reference's own prefix. A step is decisive when our top-2 margin exceeds
/// the bound's allowance at the step's logit scale; a closer race is a tie
/// at the backend's precision and is skipped. Prints the count and the
/// smallest compared margin, and returns the count.
pub(crate) fn check_greedy<B: Backend>(
    model: &AutoCausalLm<B>,
    golden: &Golden,
    bound: Bound,
    device: &B::Device,
    context: &str,
) -> usize {
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 16).unwrap(), 1, device)
        .unwrap();
    let mut logits = host(
        model
            .prefill(
                CausalLmInput {
                    token_ids: ints(std::slice::from_ref(&golden.greedy_prompt), device),
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
    let mut smallest = f32::INFINITY;
    for (step, &expected) in golden.greedy_tokens.iter().enumerate() {
        let (token, margin) = top_two(&logits);
        let scale = logits
            .iter()
            .fold(0.0_f32, |scale, &value| scale.max(value.abs()));
        if margin > bound.allowed(scale) {
            assert_eq!(token as i64, expected, "{context}: greedy token {step}");
            compared += 1;
            smallest = smallest.min(margin);
        }
        logits = host(
            model
                .decode(
                    DecodeInput {
                        token_ids: ints(&[vec![expected]], device),
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
    eprintln!(
        "{context}: {compared} of {} greedy tokens decisive and matching (smallest margin {smallest:.2e})",
        golden.greedy_tokens.len()
    );
    compared
}
