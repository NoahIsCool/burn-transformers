//! Batched cached execution: parity with the stateless graph and with row-by-row
//! execution, ring wraparound, lifecycle operations, atomic failures, and timing.
//!
//! Cached and stateless passes evaluate the same F32 mathematics in different
//! shapes (one block of new tokens against a cache versus the whole
//! sequence), so they differ only by reduction order: observed gaps are below
//! `2e-6`, and [`TOLERANCE`] is `2e-5`.

use burn::tensor::{Int, Tensor, TensorData};

use super::super::test_support::{
    B, VOCABULARY, WINDOW, assert_close, host, input, ints, mask, max_gap, model_from, tiny_config,
    tiny_model,
};
use super::*;
use crate::{
    Error, TokenMask,
    architecture::bool_tensor_from_values,
    models::config::RopeParameters,
    models::llama::{LlamaFamilyConfig, LlamaVariant},
};

/// Absolute tolerance between cached and stateless F32 logits.
const TOLERANCE: f32 = 2e-5;

/// Ten tokens per row; row 1 is left-padded by three.
const TOKENS: [[i64; 10]; 2] = [
    [3, 17, 8, 25, 11, 4, 29, 6, 19, 2],
    [0, 0, 0, 12, 5, 30, 9, 21, 14, 7],
];

/// Validity of [`TOKENS`].
const VALID: [[bool; 10]; 2] = [
    [true; 10],
    [
        false, false, false, true, true, true, true, true, true, true,
    ],
];

/// Allocates a state with `capacity` tokens per row.
fn state(model: &LlamaForCausalLm<B>, rows: usize, capacity: usize) -> LlamaDecodeState<B> {
    model
        .new_decode_state(
            CacheConfig::new(rows, capacity).unwrap(),
            rows,
            &Default::default(),
        )
        .unwrap()
}

/// Requests every column's logits.
fn all_logits() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    }
}

/// Returns columns `start..start + N` of every row of a `[2, 10]` matrix.
fn columns<T: Copy, const N: usize>(matrix: [[T; 10]; 2], start: usize) -> [[T; N]; 2] {
    matrix.map(|row| core::array::from_fn(|column| row[start + column]))
}

/// Decodes one token per row without a mask.
fn decode_step(
    model: &LlamaForCausalLm<B>,
    cache: &mut LlamaDecodeState<B>,
    tokens: Tensor<B, 2, Int>,
) -> Tensor<B, 3> {
    model
        .decode(
            DecodeInput {
                token_ids: tokens,
                position_ids: None,
                token_mask: None,
            },
            cache,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits
}

/// Prefill of five columns, then five batched one-token decode steps, equals stateless logits.
#[test]
fn batched_prefill_and_decode_match_the_stateless_forward_for_every_variant() {
    for variant in LlamaVariant::ALL {
        let model = tiny_model(variant, 7);
        let stateless = model
            .forward(input(ints(TOKENS), Some(mask(VALID)), None), &all_logits())
            .unwrap()
            .logits;
        // Capacity 12 exceeds the window, so sliding layers use a wrapping ring.
        let mut cache = state(&model, 2, 12);
        let prefill = model
            .prefill(
                input(
                    ints(columns::<i64, 5>(TOKENS, 0)),
                    Some(mask(columns::<bool, 5>(VALID, 0))),
                    None,
                ),
                &mut cache,
                &ForwardOptions::default(),
            )
            .unwrap();
        assert_close(
            prefill.logits,
            stateless.clone().narrow(1, 4, 1),
            TOLERANCE,
            &format!("{variant:?} prefill"),
        );
        for column in 5..10 {
            let step = decode_step(&model, &mut cache, ints(columns::<i64, 1>(TOKENS, column)));
            assert_close(
                step,
                stateless.clone().narrow(1, column, 1),
                TOLERANCE,
                &format!("{variant:?} decode column {column}"),
            );
        }
        assert_eq!(cache.lengths(), [10, 7], "{variant:?}");
    }
}

/// Runs `tokens` through a fresh batch-one state: the row-by-row oracle.
fn single_row_logits(model: &LlamaForCausalLm<B>, tokens: &[i64]) -> Tensor<B, 3> {
    let mut cache = state(model, 1, 16);
    let tokens = Tensor::<B, 1, Int>::from_ints(tokens, &Default::default()).unsqueeze_dim(0);
    model
        .prefill(input(tokens, None, None), &mut cache, &all_logits())
        .unwrap()
        .logits
}

/// Left, right, and interior padding in one batch equal three independent rows.
#[test]
fn ragged_padding_layouts_in_one_batch_equal_independent_rows() {
    let tokens = [
        [0, 0, 4, 9, 13, 21],
        [6, 2, 27, 0, 0, 0],
        [5, 0, 18, 0, 22, 8],
    ];
    let valid = [
        [false, false, true, true, true, true],
        [true, true, true, false, false, false],
        [true, false, true, false, true, true],
    ];
    for variant in LlamaVariant::ALL {
        let model = tiny_model(variant, 11);
        let mut cache = state(&model, 3, 16);
        let batched = model
            .prefill(
                input(ints(tokens), Some(mask(valid)), None),
                &mut cache,
                &all_logits(),
            )
            .unwrap()
            .logits;
        assert_eq!(cache.lengths(), [4, 3, 4]);
        for (row, (tokens, valid)) in tokens.iter().zip(valid).enumerate() {
            let kept: Vec<i64> = tokens
                .iter()
                .zip(valid)
                .filter_map(|(&token, keep)| keep.then_some(token))
                .collect();
            let alone = single_row_logits(&model, &kept);
            let mut index = 0;
            for (column, &keep) in valid.iter().enumerate() {
                let batched = batched.clone().narrow(0, row, 1).narrow(1, column, 1);
                if keep {
                    let context = format!("{variant:?} row {row} column {column}");
                    let alone = alone.clone().narrow(1, index, 1);
                    assert_close(batched, alone, TOLERANCE, &context);
                    index += 1;
                } else {
                    assert!(host(batched).iter().all(|&value| value == 0.0));
                }
            }
        }
    }
}

/// Ragged decode steps over rows of different lengths equal row-by-row decoding.
#[test]
fn ragged_decode_steps_equal_row_by_row_decoding() {
    let model = tiny_model(LlamaVariant::SmolLm3, 17);
    let prompts: [&[i64]; 3] = [&[4, 9, 13, 21, 2], &[6, 2], &[5, 18, 22]];
    let mut cache = state(&model, 3, 12);
    model
        .prefill(
            input(
                ints([[4, 9, 13, 21, 2], [0, 0, 0, 6, 2], [0, 0, 5, 18, 22]]),
                Some(mask([
                    [true; 5],
                    [false, false, false, true, true],
                    [false, false, true, true, true],
                ])),
                None,
            ),
            &mut cache,
            &ForwardOptions::default(),
        )
        .unwrap();
    let continuation = [[7, 8, 9], [10, 11, 12], [13, 14, 15]];
    let mut batched = Vec::new();
    for step in 0..3 {
        let tokens = ints(continuation.map(|row| [row[step]]));
        batched.push(decode_step(&model, &mut cache, tokens));
    }
    for (row, prompt) in prompts.iter().enumerate() {
        let mut single = state(&model, 1, 12);
        let tokens = Tensor::<B, 1, Int>::from_ints(*prompt, &Default::default()).unsqueeze_dim(0);
        model
            .prefill(
                input(tokens, None, None),
                &mut single,
                &ForwardOptions::default(),
            )
            .unwrap();
        for (step, logits) in batched.iter().enumerate() {
            let alone = decode_step(&model, &mut single, ints([[continuation[row][step]]]));
            let context = format!("row {row} step {step}");
            assert_close(logits.clone().narrow(0, row, 1), alone, TOLERANCE, &context);
        }
    }
}

/// Uploads a `[rows, columns]` integer matrix given as equally long row slices.
fn int_matrix(rows: &[&[i64]]) -> Tensor<B, 2, Int> {
    let columns = rows[0].len();
    assert!(rows.iter().all(|row| row.len() == columns), "ragged matrix");
    Tensor::from_data(
        TensorData::new(rows.concat(), [rows.len(), columns]),
        &Default::default(),
    )
}

/// Uploads a `[rows, columns]` validity matrix given as equally long row slices.
fn mask_matrix(rows: &[&[bool]]) -> TokenMask<B> {
    let indicators: Vec<Vec<i64>> = rows
        .iter()
        .map(|row| row.iter().map(|&valid| i64::from(valid)).collect())
        .collect();
    let indicators: Vec<&[i64]> = indicators.iter().map(Vec::as_slice).collect();
    TokenMask::new(int_matrix(&indicators).equal_elem(1))
}

/// One ragged prefill of a chunked run over three rows.
struct Chunk {
    /// `[rows][columns]` token IDs; padding slots hold 0.
    tokens: &'static [&'static [i64]],
    /// `[rows][columns]` validity of `tokens`.
    valid: &'static [&'static [bool]],
    /// Committed row lengths after the chunk.
    lengths: [usize; 3],
}

/// The three ragged prefills of the chunked run (see the test below).
const CHUNKS: [Chunk; 3] = [
    Chunk {
        tokens: &[&[3, 17, 8, 25, 11], &[0, 0, 12, 5, 30], &[9, 0, 21, 14, 0]],
        valid: &[
            &[true; 5],
            &[false, false, true, true, true],
            &[true, false, true, true, false],
        ],
        lengths: [5, 3, 3],
    },
    Chunk {
        tokens: &[&[0, 0, 4, 29], &[6, 19, 2, 10], &[7, 23, 16, 0]],
        valid: &[
            &[false, false, true, true],
            &[true; 4],
            &[true, true, true, false],
        ],
        lengths: [7, 7, 6],
    },
    Chunk {
        tokens: &[&[12, 0, 3], &[5, 26, 0], &[14, 2, 9]],
        valid: &[&[true, false, true], &[true, true, false], &[true; 3]],
        lengths: [9, 9, 9],
    },
];

/// The unpadded three-token block decoded after [`CHUNKS`].
const VERIFIED: Chunk = Chunk {
    tokens: &[&[13, 22, 5], &[18, 3, 27], &[24, 15, 20]],
    valid: &[&[true; 3], &[true; 3], &[true; 3]],
    lengths: [12, 12, 12],
};

/// Four chunks through one state equal the stateless pass of each row's kept tokens.
///
/// Every chunk after the first is a multi-token step on a non-empty state,
/// as in chunked prefill or speculative verification:
///
/// 1. a ragged prefill of five columns on an empty state, with left and
///    interior padding, compacted by a gather (lengths become `[5, 3, 3]`);
/// 2. a prefill of four columns with only left and right padding, so the
///    block is a slice in which row 0 starts at slot `c_0 = 2`, over rows of
///    different lengths; row 1 appends four tokens, more than the window of
///    3, so its ring layers wrap and attend over their history plus a fresh
///    segment longer than the window (lengths become `[7, 7, 6]`);
/// 3. a prefill with interior padding over rows of different non-zero
///    lengths, a gathered block with `L_b > 0` (lengths become `[9, 9, 9]`);
/// 4. a three-token `decode` without padding on equal lengths, a dense
///    block over a full read with no invalid key, which takes the
///    causal-only shortcut with more keys (12) than queries (3).
///
/// Valid columns must match the stateless logits of the row's kept tokens
/// at the same ordinal; padded columns must be exact zeros.
#[test]
fn chunked_steps_on_non_empty_states_match_the_stateless_forward_for_every_variant() {
    for variant in LlamaVariant::ALL {
        let model = tiny_model(variant, 19);
        let mut cache = state(&model, 3, 16);
        let mut steps = Vec::new();
        for chunk in &CHUNKS {
            let logits = model
                .prefill(
                    input(
                        int_matrix(chunk.tokens),
                        Some(mask_matrix(chunk.valid)),
                        None,
                    ),
                    &mut cache,
                    &all_logits(),
                )
                .unwrap()
                .logits;
            assert_eq!(cache.lengths(), chunk.lengths, "{variant:?}");
            steps.push((logits, chunk));
        }
        let logits = model
            .decode(
                DecodeInput {
                    token_ids: int_matrix(VERIFIED.tokens),
                    position_ids: None,
                    token_mask: None,
                },
                &mut cache,
                &all_logits(),
            )
            .unwrap()
            .logits;
        assert_eq!(cache.lengths(), VERIFIED.lengths, "{variant:?}");
        steps.push((logits, &VERIFIED));

        for row in 0..3 {
            let kept: Vec<i64> = steps
                .iter()
                .flat_map(|(_, chunk)| {
                    chunk.tokens[row]
                        .iter()
                        .zip(chunk.valid[row])
                        .filter_map(|(&token, &keep)| keep.then_some(token))
                })
                .collect();
            let stateless = model
                .forward(input(int_matrix(&[&kept]), None, None), &all_logits())
                .unwrap()
                .logits;
            let mut ordinal = 0;
            for (step, (logits, chunk)) in steps.iter().enumerate() {
                for (column, &keep) in chunk.valid[row].iter().enumerate() {
                    let cached = logits.clone().narrow(0, row, 1).narrow(1, column, 1);
                    if keep {
                        let context = format!("{variant:?} step {step} row {row} column {column}");
                        let reference = stateless.clone().narrow(1, ordinal, 1);
                        assert_close(cached, reference, TOLERANCE, &context);
                        ordinal += 1;
                    } else {
                        assert!(host(cached).iter().all(|&value| value == 0.0));
                    }
                }
            }
            assert_eq!(ordinal, kept.len());
        }
    }
}

/// A finished row is a no-op with zero logits while its peers advance.
#[test]
fn inactive_decode_rows_return_zeros_and_keep_their_length() {
    let model = tiny_model(LlamaVariant::Qwen2, 3);
    let mut cache = state(&model, 2, 16);
    model
        .prefill(
            input(ints([[1, 2, 3], [4, 5, 6]]), None, None),
            &mut cache,
            &ForwardOptions::default(),
        )
        .unwrap();
    let step = model
        .decode(
            DecodeInput {
                token_ids: ints([[7], [8]]),
                position_ids: None,
                token_mask: Some(mask([[true], [false]])),
            },
            &mut cache,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(cache.lengths(), [4, 3]);
    let logits = host(step.logits);
    assert!(logits[VOCABULARY..].iter().all(|&value| value == 0.0));
    assert!(logits[..VOCABULARY].iter().any(|&value| value != 0.0));

    let idle = model
        .decode(
            DecodeInput {
                token_ids: ints([[7], [8]]),
                position_ids: None,
                token_mask: Some(mask([[false], [false]])),
            },
            &mut cache,
            &ForwardOptions {
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    assert_eq!(
        cache.lengths(),
        [4, 3],
        "an all-inactive call changes nothing"
    );
    assert!(host(idle.logits).iter().all(|&value| value == 0.0));
    assert_eq!(
        idle.hidden_states.len(),
        model.architecture().num_layers() + 2
    );
}

/// Explicit positions rotate; padded sentinels are ignored; implicit positions are ordinals.
#[test]
fn explicit_positions_equal_implicit_ordinals_and_ignore_padded_sentinels() {
    let model = tiny_model(LlamaVariant::Llama, 5);
    let tokens = ints([[0, 0, 4, 9, 13], [6, 2, 27, 11, 3]]);
    let valid = [[false, false, true, true, true], [true; 5]];
    let run = |positions: Option<Tensor<B, 2, Int>>| {
        let mut cache = state(&model, 2, 8);
        model
            .prefill(
                input(tokens.clone(), Some(mask(valid)), positions),
                &mut cache,
                &all_logits(),
            )
            .unwrap()
            .logits
    };
    let implicit = run(None);
    let explicit = run(Some(ints([[999, -5, 0, 1, 2], [0, 1, 2, 3, 4]])));
    assert_close(explicit, implicit.clone(), 0.0, "explicit ordinals");
    // RoPE depends on position differences, so stretch the gaps.
    let stretched = run(Some(ints([[0, 0, 0, 2, 7], [0, 1, 2, 3, 4]])));
    let gap = max_gap(
        &host(stretched.clone().narrow(0, 0, 1).narrow(1, 3, 2)),
        &host(implicit.clone().narrow(0, 0, 1).narrow(1, 3, 2)),
    );
    assert!(gap > 1e-4, "explicit positions must change rotation");
    assert_close(
        stretched.narrow(0, 1, 1),
        implicit.narrow(0, 1, 1),
        0.0,
        "the other row is unaffected",
    );
}

/// Padding sentinels outside the vocabulary never reach the embedding lookup.
#[test]
fn padding_sentinels_outside_the_vocabulary_are_inert() {
    let model = tiny_model(LlamaVariant::Llama, 6);
    let valid = [[false, false, true, true], [true, true, true, false]];
    let run = |tokens: [[i64; 4]; 2]| {
        let mut cache = state(&model, 2, 8);
        let prefill = model
            .prefill(
                input(ints(tokens), Some(mask(valid)), None),
                &mut cache,
                &all_logits(),
            )
            .unwrap()
            .logits;
        let step = model
            .decode(
                DecodeInput {
                    token_ids: ints([[9], [-1]]),
                    position_ids: None,
                    token_mask: Some(mask([[true], [false]])),
                },
                &mut cache,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        (prefill, step)
    };
    let (clean_prefill, clean_step) = run([[0, 0, 4, 5], [6, 7, 8, 0]]);
    let (sentinel_prefill, sentinel_step) = run([[-1, 999, 4, 5], [6, 7, 8, i64::from(i32::MAX)]]);
    assert_close(sentinel_prefill, clean_prefill, 0.0, "prefill");
    assert_close(sentinel_step, clean_step, 0.0, "decode");
}

/// Reordering rows permutes later outputs; truncation replays like the original step.
#[test]
fn reorder_truncate_and_reset_follow_the_rows() {
    let model = tiny_model(LlamaVariant::Qwen3, 9);
    let prompt = ints([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]);
    let mut reference = state(&model, 2, 12);
    model
        .prefill(
            input(prompt.clone(), None, None),
            &mut reference,
            &all_logits(),
        )
        .unwrap();
    let expected = decode_step(&model, &mut reference, ints([[13], [14]]));

    let mut swapped = state(&model, 2, 12);
    model
        .prefill(input(prompt, None, None), &mut swapped, &all_logits())
        .unwrap();
    swapped.reorder(&[1, 0]).unwrap();
    let permuted = decode_step(&model, &mut swapped, ints([[14], [13]]));
    for (row, source) in [(0, 1), (1, 0)] {
        assert_close(
            permuted.clone().narrow(0, row, 1),
            expected.clone().narrow(0, source, 1),
            TOLERANCE,
            &format!("reordered row {row}"),
        );
    }

    // Rolling back the decoded token and replaying it reproduces the step.
    reference.truncate(&[6, 6]).unwrap();
    let replayed = decode_step(&model, &mut reference, ints([[13], [14]]));
    assert_close(replayed, expected, TOLERANCE, "truncate then replay");
    reference.reset().unwrap();
    assert_eq!(reference.lengths(), [0, 0]);
}

/// Ring rows reorder like full rows, but cannot rewind past overwritten slots.
#[test]
fn ring_rows_reorder_but_reject_unsafe_rewinds() {
    let model = tiny_model(LlamaVariant::Mistral, 9);
    let mut cache = state(&model, 2, 12);
    model
        .prefill(
            input(
                ints([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
                None,
                None,
            ),
            &mut cache,
            &all_logits(),
        )
        .unwrap();
    cache.reorder(&[1, 1]).unwrap();
    let duplicated = decode_step(&model, &mut cache, ints([[13], [13]]));
    assert_close(
        duplicated.clone().narrow(0, 0, 1),
        duplicated.narrow(0, 1, 1),
        0.0,
        "duplicated rows",
    );
    // The window of 3 has already overwritten the slot of position 4.
    let error = cache.truncate(&[4, 4]).unwrap_err();
    assert!(matches!(error, Error::TensorCache(_)), "{error}");
    assert_eq!(cache.lengths(), [7, 7]);
}

/// Clean tails hold after allocation, steps, reorders, resets, and truncation to zero.
#[test]
fn clean_tails_follow_the_lifecycle() {
    let model = tiny_model(LlamaVariant::Qwen3, 14);
    let mut cache = state(&model, 2, 8);
    assert!(cache.clean_tail);
    let prompt = ints([[1, 2, 3, 4], [5, 6, 7, 8]]);
    let options = ForwardOptions::default();
    model
        .prefill(input(prompt.clone(), None, None), &mut cache, &options)
        .unwrap();
    cache.reorder(&[1, 0]).unwrap();
    cache.truncate(&[0, 4]).unwrap();
    assert!(
        cache.clean_tail,
        "steps, reorders, and truncation to zero keep tails clean"
    );
    model
        .prefill(input(ints([[9], [10]]), None, None), &mut cache, &options)
        .unwrap();
    cache.truncate(&[1, 2]).unwrap();
    assert!(
        !cache.clean_tail,
        "a shortened row keeps bytes beyond its end"
    );
    let _ = decode_step(&model, &mut cache, ints([[11], [12]]));
    assert!(!cache.clean_tail, "later steps do not clean old bytes");
    cache.reset().unwrap();
    assert!(cache.clean_tail);
}

/// Returns a two-row state whose row 0 holds NaN keys and values beyond its end.
///
/// Both rows prefill six tokens; truncation to `[2, 5]` keeps row 0's slots
/// `2..6` as stale bytes, which are then overwritten with NaN in every
/// full-storage layer, as a rejected and overflowed draft could leave them.
fn poisoned_state(model: &LlamaForCausalLm<B>) -> LlamaDecodeState<B> {
    let mut cache = state(model, 2, 8);
    model
        .prefill(
            input(
                ints([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
                None,
                None,
            ),
            &mut cache,
            &ForwardOptions::default(),
        )
        .unwrap();
    cache.truncate(&[2, 5]).unwrap();
    let nan = Tensor::<B, 4>::full([1, 2, 4, 4], f32::NAN, &Default::default());
    for layer in 0..model.architecture().num_layers() {
        cache
            .tensors
            .main_mut()
            .native_kv_mut(0)
            .unwrap()
            .write_sequence(layer, 0, 2, nan.clone(), nan.clone())
            .unwrap();
    }
    cache
}

/// Non-finite bytes left beyond a truncated row never reach attention outputs.
///
/// An excluded key receives probability exactly zero, but `0 · NaN = NaN`,
/// so only zeroing its value keeps the row finite. The second half forces
/// the clean-tail flag over the same poisoned state, which skips that
/// zeroing, and checks that the poison then does reach the logits: the
/// first half is therefore sensitive to the masking, on Flex's fused kernel
/// as on any other.
#[test]
fn stale_non_finite_bytes_beyond_a_truncated_row_are_masked() {
    let model = tiny_model(LlamaVariant::Qwen3, 15);
    let mut cache = poisoned_state(&model);
    assert!(!cache.clean_tail, "a shortened row clears the flag");
    let poisoned = decode_step(&model, &mut cache, ints([[13], [14]]));

    let mut fresh = state(&model, 2, 8);
    model
        .prefill(
            input(
                ints([[0, 0, 0, 1, 2], [7, 8, 9, 10, 11]]),
                Some(mask([[false, false, false, true, true], [true; 5]])),
                None,
            ),
            &mut fresh,
            &ForwardOptions::default(),
        )
        .unwrap();
    let expected = decode_step(&model, &mut fresh, ints([[13], [14]]));
    assert!(host(poisoned.clone()).iter().all(|value| value.is_finite()));
    assert_close(poisoned, expected, TOLERANCE, "masked stale bytes");

    let mut unmasked = poisoned_state(&model);
    unmasked.clean_tail = true;
    let leaked = decode_step(&model, &mut unmasked, ints([[13], [14]]));
    assert!(
        host(leaked.narrow(0, 0, 1))
            .iter()
            .any(|value| !value.is_finite()),
        "without value zeroing the poison must reach row 0's logits"
    );
}

/// Requested hidden states return to the caller's columns with zeros at padding.
#[test]
fn cached_hidden_states_and_selections_match_the_stateless_pass() {
    let model = tiny_model(LlamaVariant::Olmo2, 13);
    let options = ForwardOptions {
        logits: LogitsSelection::Positions(vec![9, 4]),
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let stateless = model
        .forward(input(ints(TOKENS), Some(mask(VALID)), None), &options)
        .unwrap();
    let mut cache = state(&model, 2, 16);
    let cached = model
        .prefill(
            input(ints(TOKENS), Some(mask(VALID)), None),
            &mut cache,
            &options,
        )
        .unwrap();
    assert_close(
        cached.logits,
        stateless.logits,
        TOLERANCE,
        "selected logits",
    );
    assert_eq!(cached.hidden_states.len(), stateless.hidden_states.len());
    let pairs = cached
        .hidden_states
        .into_iter()
        .zip(stateless.hidden_states);
    for (index, (cached, stateless)) in pairs.enumerate() {
        let padding = host(cached.clone().narrow(0, 1, 1).narrow(1, 0, 3));
        assert!(padding.iter().all(|&value| value == 0.0), "state {index}");
        assert_close(
            cached.narrow(1, 3, 7),
            stateless.narrow(1, 3, 7),
            TOLERANCE,
            &format!("hidden state {index}"),
        );
    }
}

/// Sliding layers use a ring exactly when the window is below the capacity.
#[test]
fn sliding_layers_use_a_ring_only_below_the_capacity() {
    let expected = [
        (LlamaVariant::Llama, 0),
        (LlamaVariant::Mistral, 3),
        (LlamaVariant::Qwen2, 2),
        (LlamaVariant::Qwen3, 0),
        (LlamaVariant::SmolLm3, 1),
        (LlamaVariant::Olmo2, 0),
    ];
    for (variant, ring) in expected {
        let model = tiny_model(variant, 1);
        assert_eq!(state(&model, 1, 8).ring_layers(), ring, "{variant:?}");
        assert_eq!(state(&model, 1, WINDOW).ring_layers(), 0, "{variant:?}");
    }
}

/// A sliding model decodes identically with ring and with full storage.
#[test]
fn ring_storage_equals_full_storage_within_the_capacity() {
    let model = tiny_model(LlamaVariant::Mistral, 21);
    let mut ring = state(&model, 1, 8);
    let mut full = state(&model, 1, WINDOW);
    let options = ForwardOptions::default();
    let first_ring = model.prefill(input(ints([[5]]), None, None), &mut ring, &options);
    let first_full = model.prefill(input(ints([[5]]), None, None), &mut full, &options);
    assert_close(
        first_ring.unwrap().logits,
        first_full.unwrap().logits,
        TOLERANCE,
        "prefill",
    );
    for token in [9, 14] {
        let ring = decode_step(&model, &mut ring, ints([[token]]));
        let full = decode_step(&model, &mut full, ints([[token]]));
        assert_close(ring, full, TOLERANCE, "ring vs full");
    }
}

/// Every rejected call leaves the committed state untouched.
#[test]
fn rejected_calls_leave_the_state_unchanged() {
    let model = tiny_model(LlamaVariant::Qwen3, 4);
    let mut cache = state(&model, 2, 4);
    let options = ForwardOptions::default();
    model
        .prefill(
            input(ints([[1, 2], [3, 4]]), None, None),
            &mut cache,
            &options,
        )
        .unwrap();
    let selection = ForwardOptions {
        logits: LogitsSelection::Positions(vec![1]),
        ..ForwardOptions::default()
    };
    let attentions = ForwardOptions {
        attentions: true,
        ..ForwardOptions::default()
    };
    let calls: [(CausalLmInput<B>, &ForwardOptions, &str); 6] = [
        (
            input(ints([[1], [2]]), Some(mask([[true], [false]])), None),
            &options,
            "empty prompt row",
        ),
        (
            input(ints([[1, 2, 3], [4, 5, 6]]), None, None),
            &options,
            "capacity",
        ),
        (
            input(ints([[1], [99]]), None, None),
            &options,
            "token range",
        ),
        (
            input(ints([[1], [2]]), None, Some(ints([[2], [64]]))),
            &options,
            "position range",
        ),
        (
            input(ints([[1], [2]]), None, None),
            &selection,
            "logits selection",
        ),
        (
            input(ints([[1], [2]]), None, None),
            &attentions,
            "attention capture",
        ),
    ];
    for (call, options, context) in calls {
        assert!(
            model.prefill(call, &mut cache, options).is_err(),
            "{context}"
        );
        assert_eq!(cache.lengths(), [2, 2], "{context} changed the state");
    }
    let next = decode_step(&model, &mut cache, ints([[5], [6]]));
    assert_eq!(next.dims(), [2, 1, VOCABULARY], "the state stays usable");
}

/// States are bound to their model's layout and to its trained context.
#[test]
fn foreign_states_and_oversized_capacities_are_rejected() {
    let model = tiny_model(LlamaVariant::Qwen3, 4);
    let other = tiny_model(LlamaVariant::Mistral, 4);
    let options = ForwardOptions::default();
    let mut foreign = state(&other, 1, 8);
    let error = model
        .prefill(input(ints([[1]]), None, None), &mut foreign, &options)
        .unwrap_err();
    assert!(matches!(error, Error::Cache(_)), "{error}");

    let too_long = model.new_decode_state(CacheConfig::new(1, 65).unwrap(), 1, &Default::default());
    assert!(matches!(too_long, Err(Error::Configuration { .. })));

    // The state allocates no speculative staging, so no transaction can begin.
    let mut cache = state(&model, 1, 8);
    assert!(cache.tensors.begin_transaction().is_err());
}

/// LongRoPE switches factor tables past its original context, which cached keys cannot follow.
#[test]
fn context_dependent_rope_stops_at_the_original_context() {
    let parameters: RopeParameters = serde_json::from_value(serde_json::json!({
        "rope_type": "longrope",
        "rope_theta": 10000.0,
        "short_factor": [1.0, 1.0],
        "long_factor": [2.0, 4.0],
        "original_max_position_embeddings": 4,
    }))
    .unwrap();
    let config = LlamaFamilyConfig {
        rope_parameters: Some(parameters),
        ..tiny_config(LlamaVariant::Llama)
    };
    let model = model_from(config, 2);
    let mut cache = state(&model, 1, 8);
    let options = ForwardOptions::default();
    model
        .prefill(
            input(ints([[1, 2, 3, 4]]), None, None),
            &mut cache,
            &options,
        )
        .unwrap();
    let error = model
        .decode(
            DecodeInput {
                token_ids: ints([[5]]),
                position_ids: None,
                token_mask: None,
            },
            &mut cache,
            &options,
        )
        .unwrap_err();
    assert!(matches!(error, Error::BackendCapability(_)), "{error}");
    assert_eq!(cache.lengths(), [4]);
}

/// Measures batched prefill and decode against row-by-row execution on Flex.
///
/// Eight rows run as one batch and as eight batch-one states, both with
/// ragged left-padded prompts (48 down to 34 tokens) and with uniform
/// 48-token prompts. Repetitions are interleaved and each phase reports its
/// best of seven runs, which suppresses interference from other processes
/// on a shared host. Run in release mode: `cargo test --release --lib
/// models::llama::model::decode::tests::batched_decode_speedup -- --ignored
/// --nocapture`.
#[test]
#[ignore = "timing benchmark; run explicitly in release mode"]
fn batched_decode_speedup_over_row_by_row() {
    use std::time::{Duration, Instant};
    /// Batch rows.
    const ROWS: usize = 8;
    /// Padded prompt width.
    const PROMPT: usize = 48;
    /// Decode steps after the prompt.
    const STEPS: usize = 32;
    /// Interleaved repetitions per configuration.
    const REPETITIONS: usize = 7;
    let config = LlamaFamilyConfig {
        vocab_size: 2048,
        hidden_size: 256,
        intermediate_size: 704,
        num_hidden_layers: 4,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        head_dim: 32,
        max_position_embeddings: 256,
        ..tiny_config(LlamaVariant::Qwen3)
    };
    let model = model_from(config, 1);
    let device = Default::default();
    let prompt: Vec<i64> = (0..ROWS * PROMPT)
        .map(|index| (index * 37 % 2000) as i64)
        .collect();
    let options = ForwardOptions::default();
    let step_tokens = |rows: usize, step: usize| {
        Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![(step % 100) as i64; rows], [rows, 1]),
            &device,
        )
    };
    // Row r keeps its last PROMPT - 2r tokens when ragged, every token otherwise.
    let validity = |ragged: bool| -> Vec<bool> {
        (0..ROWS * PROMPT)
            .map(|index| !ragged || index % PROMPT >= 2 * (index / PROMPT))
            .collect()
    };
    // Returns (prefill, decode) wall time of one batched run.
    let batched_run = |valid: &[bool]| {
        let mut batched = state(&model, ROWS, PROMPT + STEPS);
        let tokens = Tensor::<B, 2, Int>::from_data(
            TensorData::new(prompt.clone(), [ROWS, PROMPT]),
            &device,
        );
        let mask = TokenMask::new(bool_tensor_from_values::<B, 2>(
            valid.to_vec(),
            [ROWS, PROMPT],
            &device,
        ));
        let started = Instant::now();
        let logits = model
            .prefill(input(tokens, Some(mask), None), &mut batched, &options)
            .unwrap()
            .logits;
        let _ = host(logits);
        let prefill = started.elapsed();
        let started = Instant::now();
        for step in 0..STEPS {
            let _ = host(decode_step(&model, &mut batched, step_tokens(ROWS, step)));
        }
        (prefill, started.elapsed())
    };
    // Returns (prefill, decode) wall time of eight batch-one runs.
    let row_run = |valid: &[bool]| {
        let (mut prefill, mut decode) = (Duration::ZERO, Duration::ZERO);
        for row in 0..ROWS {
            let kept: Vec<i64> = (0..PROMPT)
                .filter(|&column| valid[row * PROMPT + column])
                .map(|column| prompt[row * PROMPT + column])
                .collect();
            let mut single = state(&model, 1, PROMPT + STEPS);
            let tokens = Tensor::<B, 1, Int>::from_ints(kept.as_slice(), &device).unsqueeze_dim(0);
            let started = Instant::now();
            let logits = model
                .prefill(input(tokens, None, None), &mut single, &options)
                .unwrap()
                .logits;
            let _ = host(logits);
            prefill += started.elapsed();
            let started = Instant::now();
            for step in 0..STEPS {
                let _ = host(decode_step(&model, &mut single, step_tokens(1, step)));
            }
            decode += started.elapsed();
        }
        (prefill, decode)
    };
    let best = |runs: &[(Duration, Duration)]| {
        let prefill = runs.iter().map(|run| run.0).min().unwrap();
        let decode = runs.iter().map(|run| run.1).min().unwrap();
        (prefill, decode)
    };
    let ratio = |rows: Duration, batch: Duration| rows.as_secs_f64() / batch.as_secs_f64();
    for ragged in [true, false] {
        let valid = validity(ragged);
        let (mut batched_runs, mut row_runs) = (Vec::new(), Vec::new());
        let _ = (batched_run(&valid), row_run(&valid));
        for _ in 0..REPETITIONS {
            batched_runs.push(batched_run(&valid));
            row_runs.push(row_run(&valid));
        }
        let (batched_prefill, batched_decode) = best(&batched_runs);
        let (row_prefill, row_decode) = best(&row_runs);
        eprintln!(
            "{} prompts, {ROWS} rows: prefill batched {batched_prefill:?} vs row-by-row {row_prefill:?} ({:.2}x); {STEPS} decode steps batched {batched_decode:?} ({:?}/step) vs row-by-row {row_decode:?} ({:.2}x)",
            if ragged { "ragged" } else { "uniform" },
            ratio(row_prefill, batched_prefill),
            batched_decode / STEPS as u32,
            ratio(row_decode, batched_decode),
        );
    }
}
