//! Batched cached Gemma execution: parity with the stateless graph and with
//! row-by-row execution, ring wraparound under both RoPE sets, lifecycle
//! operations, atomic failures, foreign states, and timing.
//!
//! The cache step itself is the dense-decoder engine's and is tested in
//! depth by the Llama family (clean tails, poisoned stale bytes, every
//! padding layout); these tests prove that Gemma's sandwich blocks,
//! soft-caps, scaled embedding, and per-layer-type rotary tables run through
//! it exactly as through the stateless pass. Cached and stateless passes
//! evaluate the same F32 mathematics in different shapes, so they differ
//! only by reduction order: observed gaps are below `2e-6`, and
//! [`TOLERANCE`] is `2e-5`.

use burn::tensor::{Int, Tensor, TensorData};

use super::super::test_support::{
    B, VOCABULARY, WINDOW, assert_close, host, input, ints, mask, max_gap, model_from, tiny_config,
    tiny_model,
};
use super::*;
use crate::{
    Error, LogitsSelection, TokenMask,
    architecture::bool_tensor_from_values,
    models::gemma::{GemmaConfig, GemmaVariant},
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
fn state(model: &GemmaForCausalLm<B>, rows: usize, capacity: usize) -> GemmaDecodeState<B> {
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
    model: &GemmaForCausalLm<B>,
    cache: &mut GemmaDecodeState<B>,
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

/// Runs `tokens` through a fresh batch-one state: the row-by-row oracle.
fn single_row_logits(model: &GemmaForCausalLm<B>, tokens: &[i64]) -> Tensor<B, 3> {
    let mut cache = state(model, 1, 16);
    let tokens = Tensor::<B, 1, Int>::from_ints(tokens, &Default::default()).unsqueeze_dim(0);
    model
        .prefill(input(tokens, None, None), &mut cache, &all_logits())
        .unwrap()
        .logits
}

/// Prefill of five columns, then five batched one-token decode steps, equals stateless logits.
#[test]
fn batched_prefill_and_decode_match_the_stateless_forward_for_every_variant() {
    for variant in GemmaVariant::ALL {
        let model = tiny_model(variant, 7);
        let stateless = model
            .forward(input(ints(TOKENS), Some(mask(VALID)), None), &all_logits())
            .unwrap()
            .logits;
        // Capacity 12 exceeds the window, so sliding layers use a wrapping ring.
        let mut cache = state(&model, 2, 12);
        assert_eq!(
            cache.ring_layers(),
            if variant == GemmaVariant::Gemma2 {
                2
            } else {
                3
            }
        );
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
    for variant in GemmaVariant::ALL {
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
    for variant in GemmaVariant::ALL {
        let model = tiny_model(variant, 17);
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
        let continuation = [[7, 8, 9, 10], [10, 11, 12, 13], [13, 14, 15, 16]];
        let mut batched = Vec::new();
        for step in 0..4 {
            let tokens = ints(continuation.map(|row| [row[step]]));
            batched.push(decode_step(&model, &mut cache, tokens));
        }
        for (row, prompt) in prompts.iter().enumerate() {
            let mut single = state(&model, 1, 12);
            let tokens =
                Tensor::<B, 1, Int>::from_ints(*prompt, &Default::default()).unsqueeze_dim(0);
            model
                .prefill(
                    input(tokens, None, None),
                    &mut single,
                    &ForwardOptions::default(),
                )
                .unwrap();
            for (step, logits) in batched.iter().enumerate() {
                let alone = decode_step(&model, &mut single, ints([[continuation[row][step]]]));
                let context = format!("{variant:?} row {row} step {step}");
                assert_close(logits.clone().narrow(0, row, 1), alone, TOLERANCE, &context);
            }
        }
    }
}

/// One cached call of a chunked run: its logits and the tokens it appended.
struct CachedStep {
    /// `[rows, columns, V]` logits of the call.
    logits: Tensor<B, 3>,
    /// `[rows][columns]` token IDs of the call.
    tokens: Vec<Vec<i64>>,
    /// `[rows][columns]` validity of `tokens`.
    valid: Vec<Vec<bool>>,
}

/// Multi-token chunks on non-empty states (gathered and sliced blocks) equal the stateless pass.
///
/// Three ragged prefills grow the rows to `[5, 3, 3]`, `[7, 7, 6]`, and
/// `[9, 9, 9]` tokens: the first gathers interior padding, the second is a
/// slice in which the rows start at different slots and row 1 appends more
/// tokens (four) than the window (three), so its ring layers attend over
/// their history plus a fresh segment longer than the window, and the third
/// gathers again over non-empty rows. A final unpadded three-token decode
/// takes the dense causal shortcut on the global layers.
#[test]
fn chunked_steps_on_non_empty_states_match_the_stateless_forward() {
    /// One ragged prefill: `[rows][columns]` tokens and their validity.
    type Chunk = ([[i64; 5]; 3], [[bool; 5]; 3]);
    let chunks: [Chunk; 3] = [
        (
            [[3, 17, 8, 25, 11], [0, 0, 12, 5, 30], [9, 0, 21, 14, 0]],
            [
                [true; 5],
                [false, false, true, true, true],
                [true, false, true, true, false],
            ],
        ),
        (
            [[0, 0, 4, 29, 0], [6, 19, 2, 10, 0], [7, 23, 16, 0, 0]],
            [
                [false, false, true, true, false],
                [true, true, true, true, false],
                [true, true, true, false, false],
            ],
        ),
        (
            [[12, 0, 3, 0, 0], [5, 26, 0, 0, 0], [14, 2, 9, 0, 0]],
            [
                [true, false, true, false, false],
                [true, true, false, false, false],
                [true, true, true, false, false],
            ],
        ),
    ];
    let lengths = [[5, 3, 3], [7, 7, 6], [9, 9, 9]];
    let verified = [[13, 22, 5], [18, 3, 27], [24, 15, 20]];
    for variant in GemmaVariant::ALL {
        let model = tiny_model(variant, 19);
        let mut cache = state(&model, 3, 16);
        let mut steps: Vec<CachedStep> = Vec::new();
        for ((tokens, valid), lengths) in chunks.iter().zip(lengths) {
            let logits = model
                .prefill(
                    input(ints(*tokens), Some(mask(*valid)), None),
                    &mut cache,
                    &all_logits(),
                )
                .unwrap()
                .logits;
            assert_eq!(cache.lengths(), lengths, "{variant:?}");
            steps.push(CachedStep {
                logits,
                tokens: tokens.iter().map(|row| row.to_vec()).collect(),
                valid: valid.iter().map(|row| row.to_vec()).collect(),
            });
        }
        let logits = model
            .decode(
                DecodeInput {
                    token_ids: ints(verified),
                    position_ids: None,
                    token_mask: None,
                },
                &mut cache,
                &all_logits(),
            )
            .unwrap()
            .logits;
        assert_eq!(cache.lengths(), [12, 12, 12], "{variant:?}");
        steps.push(CachedStep {
            logits,
            tokens: verified.iter().map(|row| row.to_vec()).collect(),
            valid: vec![vec![true; 3]; 3],
        });
        for row in 0..3 {
            let kept: Vec<i64> = steps
                .iter()
                .flat_map(|step| {
                    step.tokens[row]
                        .iter()
                        .zip(&step.valid[row])
                        .filter_map(|(&token, &keep)| keep.then_some(token))
                        .collect::<Vec<_>>()
                })
                .collect();
            let tokens = Tensor::<B, 1, Int>::from_ints(kept.as_slice(), &Default::default())
                .unsqueeze_dim(0);
            let stateless = model
                .forward(input(tokens, None, None), &all_logits())
                .unwrap()
                .logits;
            let mut ordinal = 0;
            for (index, step) in steps.iter().enumerate() {
                for (column, &keep) in step.valid[row].iter().enumerate() {
                    let cached = step.logits.clone().narrow(0, row, 1).narrow(1, column, 1);
                    if keep {
                        let context = format!("{variant:?} step {index} row {row} column {column}");
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
    let model = tiny_model(GemmaVariant::Gemma2, 3);
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
    assert_eq!(idle.hidden_states.len(), 4 + 2);
}

/// Explicit positions rotate both RoPE sets; padded sentinels are ignored.
#[test]
fn explicit_positions_equal_implicit_ordinals_and_ignore_padded_sentinels() {
    let model = tiny_model(GemmaVariant::Gemma3Text, 5);
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

/// Reordering rows permutes later outputs; truncation replays like the original step.
///
/// Rings can rewind only while they still hold every key the shorter rows
/// need: before a sliding layer's ring wraps (two prompt tokens and one
/// decoded token fit a window of three), a rollback is exact; after it
/// wraps, a rollback is refused and the state is unchanged.
#[test]
fn reorder_truncate_and_reset_follow_the_rows() {
    let model = tiny_model(GemmaVariant::Gemma3Text, 9);
    let prompt = ints([[1, 2], [7, 8]]);
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
    reference.truncate(&[2, 2]).unwrap();
    let replayed = decode_step(&model, &mut reference, ints([[13], [14]]));
    assert_close(replayed, expected, TOLERANCE, "truncate then replay");
    let _ = decode_step(&model, &mut reference, ints([[15], [16]]));
    let error = reference.truncate(&[3, 3]).unwrap_err();
    assert!(matches!(error, Error::TensorCache(_)), "{error}");
    assert_eq!(reference.lengths(), [4, 4]);
    reference.reset().unwrap();
    assert_eq!(reference.lengths(), [0, 0]);
    assert_eq!(reference.batch_size(), 2);
    assert_eq!(reference.capacity(), 12);
    assert_eq!(reference.tensor_state().capacity(), 12);
    assert_eq!(reference.device(), &Default::default());
}

/// Sliding layers decode identically with ring and with full storage.
#[test]
fn ring_storage_equals_full_storage_within_the_capacity() {
    for variant in GemmaVariant::ALL {
        let model = tiny_model(variant, 21);
        let mut ring = state(&model, 1, 8);
        let mut full = state(&model, 1, WINDOW);
        assert!(ring.ring_layers() > 0 && full.ring_layers() == 0);
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
            assert_close(ring, full, TOLERANCE, &format!("{variant:?} ring vs full"));
        }
    }
}

/// Every rejected call leaves the committed state untouched.
#[test]
fn rejected_calls_leave_the_state_unchanged() {
    let model = tiny_model(GemmaVariant::Gemma2, 4);
    let mut cache = state(&model, 2, 4);
    let options = ForwardOptions::default();
    model
        .prefill(
            input(ints([[1, 2], [3, 4]]), None, None),
            &mut cache,
            &options,
        )
        .unwrap();
    let attentions = ForwardOptions {
        attentions: true,
        ..ForwardOptions::default()
    };
    let calls: [(CausalLmInput<B>, &ForwardOptions, &str); 4] = [
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
            input(ints([[1], [2]]), None, None),
            &attentions,
            "attention capture",
        ),
    ];
    for (call, options, context) in calls {
        let error = model.prefill(call, &mut cache, options).unwrap_err();
        assert_eq!(cache.lengths(), [2, 2], "{context} changed the state");
        if context == "token range" {
            assert!(error.to_string().contains("gemma.input_ids"), "{error}");
        }
        if context == "attention capture" {
            assert!(error.to_string().contains("Gemma"), "{error}");
        }
    }
    let next = decode_step(&model, &mut cache, ints([[5], [6]]));
    assert_eq!(next.dims(), [2, 1, VOCABULARY], "the state stays usable");
}

/// States are bound to their model's layout and to its trained context.
#[test]
fn foreign_states_and_oversized_capacities_are_rejected() {
    let model = tiny_model(GemmaVariant::Gemma3Text, 4);
    let other = tiny_model(GemmaVariant::Gemma2, 4);
    let options = ForwardOptions::default();
    let mut foreign = state(&other, 1, 8);
    let error = model
        .prefill(input(ints([[1]]), None, None), &mut foreign, &options)
        .unwrap_err();
    assert!(matches!(error, Error::Cache(_)), "{error}");
    assert!(error.to_string().contains("Gemma"), "{error}");
    let too_long = model.new_decode_state(CacheConfig::new(1, 65).unwrap(), 1, &Default::default());
    let error = too_long.unwrap_err();
    assert!(
        error.to_string().contains("gemma.decode_state.capacity"),
        "{error}"
    );
}

/// Measures batched prefill and decode against row-by-row execution on Flex.
///
/// A Gemma 3-style model (width 256, eight layers with a 1-in-4 global
/// schedule, window 16 so sliding layers use rings) runs eight rows as one
/// batch and as eight batch-one states, with ragged left-padded prompts (48
/// down to 34 tokens) and with uniform 48-token prompts. Repetitions are
/// interleaved and each phase reports its best of seven runs. Run in
/// release mode: `cargo test --release --lib
/// models::gemma::model::decode::tests::batched_decode_speedup -- --ignored
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
    let config = GemmaConfig {
        vocab_size: 2048,
        hidden_size: 256,
        intermediate_size: 704,
        num_hidden_layers: 8,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        head_dim: 32,
        max_position_embeddings: 256,
        sliding_window: 16,
        sliding_window_pattern: Some(4),
        ..tiny_config(GemmaVariant::Gemma3Text)
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
