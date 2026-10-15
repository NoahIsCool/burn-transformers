//! Batched cached execution: parity with the stateless graph and with
//! row-by-row execution, padding layouts, ring wraparound, lifecycle
//! operations, clean-tail tracking, atomic failures, and timing.
//!
//! Cached and stateless passes evaluate the same F32 mathematics in
//! different shapes (one block of new tokens against a cache versus the
//! whole sequence), so they differ only by reduction order: observed gaps
//! are below `2e-6`, and [`TOLERANCE`] is `2e-5`. A flipped expert choice
//! would move logits far more, so passing also proves identical routing.

use burn::tensor::{Int, Tensor, TensorData};

use super::super::test_support::{
    B, VOCABULARY, WINDOW, assert_close, host, input, ints, mask, max_gap, model_from, tiny_config,
    tiny_model,
};
use super::*;
use crate::TokenMask;

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
fn state(model: &GptOssForCausalLm<B>, rows: usize, capacity: usize) -> GptOssDecodeState<B> {
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
    model: &GptOssForCausalLm<B>,
    cache: &mut GptOssDecodeState<B>,
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

/// Uploads one row of tokens.
fn row(tokens: &[i64]) -> Tensor<B, 2, Int> {
    Tensor::from_data(
        TensorData::new(tokens.to_vec(), [1, tokens.len()]),
        &Default::default(),
    )
}

/// Prefill of five columns then five one-token batched steps equals the stateless logits.
#[test]
fn batched_prefill_and_decode_match_the_stateless_forward() {
    let model = tiny_model(7);
    let stateless = model
        .forward(input(ints(TOKENS), Some(mask(VALID)), None), &all_logits())
        .unwrap()
        .logits;
    // Capacity 12 exceeds the window, so the sliding layers use a wrapping ring.
    let mut cache = state(&model, 2, 12);
    const { assert!(WINDOW < 12) };
    assert_eq!(cache.ring_layers(), 2);
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
        "prefill",
    );
    for column in 5..10 {
        let step = decode_step(&model, &mut cache, ints(columns::<i64, 1>(TOKENS, column)));
        assert_close(
            step,
            stateless.clone().narrow(1, column, 1),
            TOLERANCE,
            &format!("decode column {column}"),
        );
    }
    assert_eq!(cache.lengths(), [10, 7]);
}

/// Runs each row alone through single-row states: the per-row reference path.
fn row_by_row(
    model: &GptOssForCausalLm<B>,
    prompts: &[Vec<i64>],
    steps: &[Vec<i64>],
) -> Vec<Vec<Vec<f32>>> {
    prompts
        .iter()
        .zip(steps)
        .map(|(prompt, continuation)| {
            let mut cache = state(model, 1, 16);
            let mut logits = vec![host(
                model
                    .prefill(
                        input(row(prompt), None, None),
                        &mut cache,
                        &ForwardOptions::default(),
                    )
                    .unwrap()
                    .logits,
            )];
            for &token in continuation {
                logits.push(host(decode_step(model, &mut cache, row(&[token]))));
            }
            logits
        })
        .collect()
}

/// Ragged batched execution equals running every row alone.
#[test]
fn ragged_batches_match_row_by_row_execution() {
    let model = tiny_model(9);
    // Rows of 6, 2, and 4 prompt tokens (left, right, and interior padding).
    let tokens = [
        [4, 9, 13, 2, 30, 7],
        [0, 0, 0, 0, 5, 11],
        [3, 0, 8, 0, 14, 22],
    ];
    let valid = [
        [true; 6],
        [false, false, false, false, true, true],
        [true, false, true, false, true, true],
    ];
    let prompts = vec![vec![4, 9, 13, 2, 30, 7], vec![5, 11], vec![3, 8, 14, 22]];
    let continuation = [[6, 1, 19], [27, 3, 3], [10, 15, 2]];
    let steps: Vec<Vec<i64>> = continuation.iter().map(|row| row.to_vec()).collect();
    let reference = row_by_row(&model, &prompts, &steps);
    let mut cache = state(&model, 3, 16);
    let prefill = model
        .prefill(
            input(ints(tokens), Some(mask(valid)), None),
            &mut cache,
            &ForwardOptions::default(),
        )
        .unwrap();
    let mut batched = vec![host(prefill.logits)];
    for step in 0..3 {
        let tokens = ints(continuation.map(|row| [row[step]]));
        batched.push(host(decode_step(&model, &mut cache, tokens)));
    }
    for (step, logits) in batched.iter().enumerate() {
        for (index, rows) in reference.iter().enumerate() {
            let gap = max_gap(
                &logits[index * VOCABULARY..(index + 1) * VOCABULARY],
                &rows[step],
            );
            assert!(gap <= TOLERANCE, "step {step}, row {index}: {gap:e}");
        }
    }
    assert_eq!(cache.lengths(), [9, 5, 7]);
}

/// `All` logits and hidden states return to the caller's columns with zeros at padding.
#[test]
fn cached_outputs_restore_caller_columns() {
    let model = tiny_model(11);
    let tokens = [[3, 0, 8, 25, 0, 4], [0, 12, 5, 0, 0, 9]];
    let valid = [
        [true, false, true, true, false, true],
        [false, true, true, false, false, true],
    ];
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        router_logits: true,
        auxiliary_losses: true,
        ..ForwardOptions::default()
    };
    let stateless = model
        .forward(input(ints(tokens), Some(mask(valid)), None), &options)
        .unwrap();
    let mut cache = state(&model, 2, 8);
    let cached = model
        .prefill(
            input(ints(tokens), Some(mask(valid)), None),
            &mut cache,
            &options,
        )
        .unwrap();
    let loss = |output: &CausalLmOutput<B>| host(output.load_balancing_loss.clone().unwrap())[0];
    let (cached_loss, stateless_loss) = (loss(&cached), loss(&stateless));
    let keep = mask(valid)
        .as_tensor()
        .clone()
        .float()
        .unsqueeze_dim::<3>(2);
    assert_close(
        cached.logits.clone(),
        stateless.logits * keep.clone(),
        TOLERANCE,
        "all logits",
    );
    let padded = host(cached.logits.narrow(1, 1, 1).narrow(0, 0, 1));
    assert!(
        padded.iter().all(|&value| value == 0.0),
        "padding is exact zero"
    );
    assert_eq!(cached.hidden_states.len(), 3 + 2);
    for (index, (ours, theirs)) in cached
        .hidden_states
        .into_iter()
        .zip(stateless.hidden_states)
        .enumerate()
    {
        assert_close(
            ours,
            theirs * keep.clone(),
            TOLERANCE,
            &format!("hidden {index}"),
        );
    }
    assert_eq!(cached.routers.len(), 3);
    for (ours, theirs) in cached.routers.iter().zip(&stateless.routers) {
        assert_close(
            ours.logits.clone().unwrap(),
            theirs.logits.clone().unwrap() * keep.clone(),
            TOLERANCE,
            "router logits",
        );
    }
    assert!((cached_loss - stateless_loss).abs() <= 1e-5);
}

/// Fully masked decode rows are exact-zero no-ops; an all-inactive call changes nothing.
#[test]
fn inactive_rows_decode_to_zeros_without_state_change() {
    let model = tiny_model(13);
    let mut cache = state(&model, 2, 8);
    model
        .prefill(
            input(ints([[1, 2, 3], [4, 5, 6]]), None, None),
            &mut cache,
            &ForwardOptions::default(),
        )
        .unwrap();
    let output = model
        .decode(
            DecodeInput {
                token_ids: ints([[7], [-1]]),
                token_mask: Some(mask([[true], [false]])),
                position_ids: None,
            },
            &mut cache,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(cache.lengths(), [4, 3]);
    let logits = host(output.logits);
    assert!(logits[VOCABULARY..].iter().all(|&value| value == 0.0));
    let idle = model
        .decode(
            DecodeInput {
                token_ids: ints([[0], [0]]),
                token_mask: Some(mask([[false], [false]])),
                position_ids: None,
            },
            &mut cache,
            &ForwardOptions {
                hidden_states: true,
                router_logits: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    assert_eq!(cache.lengths(), [4, 3]);
    assert!(host(idle.logits).iter().all(|&value| value == 0.0));
    assert_eq!(idle.hidden_states.len(), 3 + 2);
    assert_eq!(idle.routers.len(), 3);
    // Prefill refuses empty rows.
    assert!(
        model
            .prefill(
                input(ints([[1], [2]]), Some(mask([[true], [false]])), None),
                &mut cache,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert_eq!(cache.lengths(), [4, 3]);
}

/// A prompt longer than the window and a long continuation wrap the ring correctly.
#[test]
fn ring_wraparound_matches_the_stateless_forward() {
    let model = tiny_model(15);
    let sequence: Vec<i64> = (0..15).map(|index| (index * 7 % 31) + 1).collect();
    let stateless = model
        .forward(input(row(&sequence), None, None), &all_logits())
        .unwrap()
        .logits;
    for capacity in [15, 32] {
        let mut cache = state(&model, 1, capacity);
        let prefill = model
            .prefill(
                input(row(&sequence[..9]), None, None),
                &mut cache,
                &all_logits(),
            )
            .unwrap();
        assert_close(
            prefill.logits,
            stateless.clone().narrow(1, 0, 9),
            TOLERANCE,
            "long prefill",
        );
        for (column, &token) in sequence.iter().enumerate().skip(9) {
            let step = decode_step(&model, &mut cache, row(&[token]));
            assert_close(
                step,
                stateless.clone().narrow(1, column, 1),
                TOLERANCE,
                &format!("capacity {capacity}, column {column}"),
            );
        }
    }
}

/// A window at least the capacity keeps every layer in full storage, with the same results.
#[test]
fn windows_wider_than_the_capacity_use_full_storage() {
    let model = model_from(
        GptOssConfig {
            sliding_window: Some(40),
            ..tiny_config()
        },
        17,
    );
    let mut cache = state(&model, 2, 12);
    assert_eq!(cache.ring_layers(), 0);
    let stateless = model
        .forward(input(ints(TOKENS), Some(mask(VALID)), None), &all_logits())
        .unwrap()
        .logits;
    let prefill = model
        .prefill(
            input(ints(TOKENS), Some(mask(VALID)), None),
            &mut cache,
            &all_logits(),
        )
        .unwrap();
    let keep = mask(VALID)
        .as_tensor()
        .clone()
        .float()
        .unsqueeze_dim::<3>(2);
    assert_close(prefill.logits, stateless * keep, TOLERANCE, "full storage");
    // A full-attention-only schedule needs no ring either.
    let global = model_from(
        GptOssConfig {
            layer_types: vec!["full_attention".into(); 3],
            ..tiny_config()
        },
        17,
    );
    assert_eq!(state(&global, 1, 12).ring_layers(), 0);
}

/// Explicit positions rotate instead of the implicit cache positions.
#[test]
fn explicit_positions_rotate_cached_tokens() {
    let model = tiny_model(19);
    let mut implicit = state(&model, 1, 8);
    let mut explicit = state(&model, 1, 8);
    let options = ForwardOptions::default();
    let a = model
        .prefill(input(row(&[5, 6, 7]), None, None), &mut implicit, &options)
        .unwrap()
        .logits;
    let b = model
        .prefill(
            input(row(&[5, 6, 7]), None, Some(row(&[0, 1, 2]))),
            &mut explicit,
            &options,
        )
        .unwrap()
        .logits;
    assert_close(a, b, 0.0, "prefill positions");
    let shifted = model
        .decode(
            DecodeInput {
                token_ids: row(&[8]),
                token_mask: None,
                position_ids: Some(row(&[20])),
            },
            &mut explicit,
            &options,
        )
        .unwrap()
        .logits;
    let plain = decode_step(&model, &mut implicit, row(&[8]));
    assert!(
        max_gap(&host(shifted), &host(plain)) > 1e-4,
        "positions matter"
    );
}

/// Reorder, truncate, and reset keep the cache consistent with the stateless graph.
#[test]
fn lifecycle_operations_preserve_parity() {
    let model = tiny_model(21);
    let stateless = model
        .forward(input(ints(TOKENS), Some(mask(VALID)), None), &all_logits())
        .unwrap()
        .logits;
    let mut cache = state(&model, 2, 12);
    model
        .prefill(
            input(
                ints(columns::<i64, 8>(TOKENS, 0)),
                Some(mask(columns::<bool, 8>(VALID, 0))),
                None,
            ),
            &mut cache,
            &ForwardOptions::default(),
        )
        .unwrap();
    // Swapping rows swaps their continuations.
    cache.reorder(&[1, 0]).unwrap();
    let swapped = decode_step(&model, &mut cache, ints([[TOKENS[1][8]], [TOKENS[0][8]]]));
    assert_close(
        swapped.clone().narrow(0, 0, 1),
        stateless.clone().narrow(0, 1, 1).narrow(1, 8, 1),
        TOLERANCE,
        "reordered row 0",
    );
    assert_close(
        swapped.narrow(0, 1, 1),
        stateless.clone().narrow(0, 0, 1).narrow(1, 8, 1),
        TOLERANCE,
        "reordered row 1",
    );
    // A wrapped ring no longer holds the window before its oldest key, so it
    // refuses to roll back; nothing changes.
    cache.reorder(&[1, 0]).unwrap();
    assert_eq!(cache.lengths(), [9, 6]);
    assert!(cache.truncate(&[8, 5]).is_err());
    assert_eq!(cache.lengths(), [9, 6]);
    cache.reset().unwrap();
    assert_eq!(cache.lengths(), [0, 0]);
    let again = model
        .prefill(
            input(ints(TOKENS), Some(mask(VALID)), None),
            &mut cache,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    assert_close(again, stateless.narrow(1, 9, 1), TOLERANCE, "after reset");
    assert_eq!(cache.batch_size(), 2);
    assert_eq!(cache.capacity(), 12);
    assert_eq!(cache.tensor_state().lengths(), [10, 7]);
    let _ = cache.device();
}

/// Full storage rolls back and replays exactly (speculative-decoding rollback).
#[test]
fn full_storage_truncation_replays_exactly() {
    let model = model_from(
        GptOssConfig {
            sliding_window: Some(40),
            ..tiny_config()
        },
        27,
    );
    let stateless = model
        .forward(input(ints(TOKENS), Some(mask(VALID)), None), &all_logits())
        .unwrap()
        .logits;
    let mut cache = state(&model, 2, 12);
    assert_eq!(cache.ring_layers(), 0);
    model
        .prefill(
            input(
                ints(columns::<i64, 9>(TOKENS, 0)),
                Some(mask(columns::<bool, 9>(VALID, 0))),
                None,
            ),
            &mut cache,
            &ForwardOptions::default(),
        )
        .unwrap();
    cache.truncate(&[7, 4]).unwrap();
    assert_eq!(cache.lengths(), [7, 4]);
    for column in 7..10 {
        let step = decode_step(&model, &mut cache, ints(columns::<i64, 1>(TOKENS, column)));
        assert_close(
            step,
            stateless.clone().narrow(1, column, 1),
            TOLERANCE,
            &format!("replayed column {column}"),
        );
    }
}

/// Returns the tiny model with every layer in full storage (window 40 > capacity).
fn full_storage_model(seed: u64) -> GptOssForCausalLm<B> {
    model_from(
        GptOssConfig {
            sliding_window: Some(40),
            ..tiny_config()
        },
        seed,
    )
}

/// Clean tails hold after allocation, steps, reorders, resets, and truncation to zero.
#[test]
fn clean_tails_follow_the_lifecycle() {
    let model = full_storage_model(29);
    let mut cache = state(&model, 2, 8);
    assert!(cache.clean_tail);
    let options = ForwardOptions::default();
    model
        .prefill(
            input(ints([[1, 2, 3, 4], [5, 6, 7, 8]]), None, None),
            &mut cache,
            &options,
        )
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
    // Truncating to the current length keeps every byte reachable.
    cache.truncate(&[1, 5]).unwrap();
    assert!(cache.clean_tail, "a no-op truncation keeps tails clean");
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
fn poisoned_state(model: &GptOssForCausalLm<B>) -> GptOssDecodeState<B> {
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
    let architecture = model.architecture();
    let nan = Tensor::<B, 4>::full(
        [1, architecture.key_value_heads, 4, architecture.head_dim],
        f32::NAN,
        &Default::default(),
    );
    let full = cache.layout.full.expect("every layer uses full storage");
    for slot in 0..cache.layout.slots.len() {
        cache
            .tensors
            .main_mut()
            .native_kv_mut(full.index)
            .unwrap()
            .write_sequence(slot, 0, 2, nan.clone(), nan.clone())
            .unwrap();
    }
    cache
}

/// Non-finite bytes left beyond a truncated row never reach attention outputs.
///
/// An excluded key receives probability exactly zero, but `0 · NaN = NaN`,
/// so only zeroing its value keeps the row finite. The second half forces
/// the clean-tail flag over the same poisoned state, which skips that
/// zeroing, and checks that the poison then does escape attention: the
/// router refuses the non-finite scores it produces (the routed MLP's
/// finiteness guard), so the step fails instead of returning logits. The
/// first half is therefore sensitive to the masking.
#[test]
fn stale_non_finite_bytes_beyond_a_truncated_row_are_masked() {
    let model = full_storage_model(31);
    let mut cache = poisoned_state(&model);
    assert_eq!(cache.ring_layers(), 0);
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
    let error = model
        .decode(
            DecodeInput {
                token_ids: ints([[13], [14]]),
                position_ids: None,
                token_mask: None,
            },
            &mut unmasked,
            &ForwardOptions::default(),
        )
        .expect_err("without value zeroing the poison must escape attention");
    assert!(error.to_string().contains("non-finite router"), "{error}");
    assert_eq!(
        unmasked.lengths(),
        [2, 5],
        "the failed step publishes nothing"
    );
    assert!(
        !unmasked.clean_tail,
        "a step failing after its write-ahead leaves tails unclean"
    );
}

/// A failing step changes nothing; the same state then continues exactly.
#[test]
fn failed_steps_leave_the_state_unchanged() {
    let model = tiny_model(23);
    let stateless = model
        .forward(input(row(&[1, 2, 3, 4]), None, None), &all_logits())
        .unwrap()
        .logits;
    let mut cache = state(&model, 1, 8);
    model
        .prefill(
            input(row(&[1, 2, 3]), None, None),
            &mut cache,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert!(
        model
            .decode(
                DecodeInput {
                    token_ids: row(&[VOCABULARY as i64]),
                    token_mask: None,
                    position_ids: None,
                },
                &mut cache,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert_eq!(cache.lengths(), [3]);
    let step = decode_step(&model, &mut cache, row(&[4]));
    assert_close(step, stateless.narrow(1, 3, 1), TOLERANCE, "after failure");
    // Overflowing the capacity fails before any write.
    let mut full = state(&model, 1, 2);
    assert!(
        model
            .prefill(
                input(row(&[1, 2, 3]), None, None),
                &mut full,
                &ForwardOptions::default()
            )
            .is_err()
    );
    assert_eq!(full.lengths(), [0]);
}

/// States are bound to their model; invalid capacities and requests are refused.
#[test]
fn incompatible_states_and_requests_are_refused() {
    let model = tiny_model(25);
    let other = model_from(
        GptOssConfig {
            sliding_window: Some(5),
            ..tiny_config()
        },
        25,
    );
    let mut foreign = state(&other, 1, 8);
    assert!(
        model
            .prefill(
                input(row(&[1, 2]), None, None),
                &mut foreign,
                &ForwardOptions::default()
            )
            .is_err()
    );
    assert!(
        model
            .new_decode_state(CacheConfig::new(1, 65).unwrap(), 1, &Default::default())
            .is_err()
    );
    let mut cache = state(&model, 1, 8);
    let attentions = ForwardOptions {
        attentions: true,
        ..ForwardOptions::default()
    };
    assert!(matches!(
        model.prefill(input(row(&[1, 2]), None, None), &mut cache, &attentions),
        Err(Error::BackendCapability(_))
    ));
    assert!(
        model
            .prefill(
                input(row(&[1, 2]), None, None),
                &mut cache,
                &ForwardOptions {
                    logits: LogitsSelection::Positions(vec![5]),
                    ..ForwardOptions::default()
                },
            )
            .is_err()
    );
    assert_eq!(cache.lengths(), [0]);
    // The trait object path delegates to the same implementation.
    let stateless = CausalLanguageModel::forward(
        &model,
        input(row(&[1, 2]), None, None),
        &ForwardOptions::default(),
    )
    .unwrap();
    assert_eq!(stateless.logits.dims(), [1, 1, VOCABULARY]);
    let _: Option<TokenMask<B>> = None;
}

/// Measures batched against row-by-row decoding on Flex (`--release --ignored --nocapture`).
///
/// Two 4-layer models with 8 experts (2 selected), of width 256 and 1024,
/// decode 8 rows for 32 steps after a 16-token prefill, once as one batched
/// pass per step and once row by row. Both paths are checked to agree
/// before the timings are printed.
#[test]
#[ignore = "timing measurement; run with --release --ignored --nocapture"]
fn batched_decoding_outpaces_row_by_row_decoding() {
    for width in [256, 1024] {
        let config = GptOssConfig {
            vocab_size: 512,
            hidden_size: width,
            intermediate_size: width,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: width / 8,
            num_local_experts: 8,
            num_experts_per_tok: 2,
            sliding_window: Some(16),
            layer_types: GptOssConfig::default_layer_types(4),
            max_position_embeddings: 128,
            ..tiny_config()
        };
        // Burn's fan-in-scaled initialization keeps wide activations realistic.
        let model = GptOssForCausalLm::<B>::init(config, &Default::default()).unwrap();
        time_decoding(&model, width);
    }
}

/// Times batched and row-by-row decoding of one model; see the caller.
fn time_decoding(model: &GptOssForCausalLm<B>, width: usize) {
    let (rows, prompt, steps) = (8, 16, 32);
    let token = |row: usize, column: usize| ((row * 131 + column * 17) % 509) as i64 + 1;
    let prompts: Vec<Vec<i64>> = (0..rows)
        .map(|r| (0..prompt).map(|c| token(r, c)).collect())
        .collect();
    let continuation: Vec<Vec<i64>> = (0..rows)
        .map(|r| (prompt..prompt + steps).map(|c| token(r, c)).collect())
        .collect();
    let device = Default::default();

    let started = std::time::Instant::now();
    let mut cache = model
        .new_decode_state(CacheConfig::new(rows, 64).unwrap(), rows, &device)
        .unwrap();
    let tokens =
        Tensor::<B, 2, Int>::from_data(TensorData::new(prompts.concat(), [rows, prompt]), &device);
    let options = ForwardOptions::default();
    let mut batched = vec![host(
        model
            .prefill(input(tokens, None, None), &mut cache, &options)
            .unwrap()
            .logits,
    )];
    for step in 0..steps {
        let column: Vec<i64> = continuation.iter().map(|row| row[step]).collect();
        let tokens = Tensor::<B, 2, Int>::from_data(TensorData::new(column, [rows, 1]), &device);
        batched.push(host(decode_step(model, &mut cache, tokens)));
    }
    let batched_time = started.elapsed();

    let started = std::time::Instant::now();
    let reference = prompts
        .iter()
        .zip(&continuation)
        .map(|(prompt, continuation)| {
            let mut cache = model
                .new_decode_state(CacheConfig::new(1, 64).unwrap(), 1, &device)
                .unwrap();
            let mut logits = vec![host(
                model
                    .prefill(input(row(prompt), None, None), &mut cache, &options)
                    .unwrap()
                    .logits,
            )];
            for &token in continuation {
                logits.push(host(decode_step(model, &mut cache, row(&[token]))));
            }
            logits
        })
        .collect::<Vec<_>>();
    let row_time = started.elapsed();

    // Relative to the logit scale; a flipped expert choice moves logits by O(scale).
    let vocabulary = 512;
    for (step, logits) in batched.iter().enumerate() {
        for (index, rows) in reference.iter().enumerate() {
            let scale = rows[step].iter().fold(1.0_f32, |m, v| m.max(v.abs()));
            let gap = max_gap(
                &logits[index * vocabulary..(index + 1) * vocabulary],
                &rows[step],
            );
            assert!(
                gap <= 1e-5 * scale,
                "width {width}, step {step}, row {index}: {gap:e} at scale {scale:e}"
            );
        }
    }
    eprintln!(
        "gpt-oss Flex decode, width {width}, {rows} rows x {steps} steps: batched {batched_time:?}, row by row {row_time:?} ({:.2}x)",
        row_time.as_secs_f64() / batched_time.as_secs_f64()
    );
}
