//! Token-history tests: count tensors, n-gram index, validation, and incremental updates.

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, TensorData},
};

use super::{CountView, NgramIndex, TokenHistory};
use crate::{GenerationConfig, TokenMask, generation::PenaltyScope};

/// Deterministic F32 CPU backend.
type TestBackend = Flex<f32, i32>;

/// Reads one `[batch, vocabulary]` count view to host integers.
fn counts(history: &TokenHistory<TestBackend>, view: CountView) -> Vec<i32> {
    history
        .counts(view)
        .expect("tracked count view")
        .clone()
        .into_data()
        .to_vec::<i32>()
        .expect("integer counts")
}

/// Configuration whose penalties read both count views.
fn penalized() -> GenerationConfig {
    GenerationConfig {
        repetition_penalty: 1.2,
        presence_penalty: 0.5,
        ..Default::default()
    }
}

/// Prompt counts include only valid tokens; generated counts start at zero.
#[test]
fn prompt_counts_skip_masked_sentinels() {
    let device = Default::default();
    let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 99, 1], [2, 3, 99]], &device);
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, false, true], [true, true, false]].into(),
        &device,
    ));
    let history = TokenHistory::from_prompt(&penalized(), 4, prompt, Some(mask)).unwrap();
    assert_eq!(counts(&history, CountView::All), [0, 2, 0, 0, 0, 0, 1, 1]);
    assert_eq!(counts(&history, CountView::Generated), [0; 8]);
    assert_eq!(history.generated_lengths(), [0, 0]);
    assert_eq!(history.batch_size(), 2);
    assert_eq!(history.vocabulary_size(), 4);
}

/// Recording updates both count tensors and lengths only for active rows.
#[test]
fn record_updates_active_rows_only() {
    let device = Default::default();
    let mut history = TokenHistory::<TestBackend>::empty(&penalized(), 3, 2, &device).unwrap();
    history.record(&[2, 1], &[true, false]).unwrap();
    history.record(&[2, 0], &[true, true]).unwrap();
    history.record(&[0, 0], &[false, false]).unwrap();
    assert_eq!(counts(&history, CountView::All), [0, 0, 2, 1, 0, 0]);
    assert_eq!(counts(&history, CountView::Generated), [0, 0, 2, 1, 0, 0]);
    assert_eq!(history.generated_lengths(), [2, 1]);
    assert!(history.record(&[3, 0], &[true, false]).is_err());
    assert!(history.record(&[0], &[true]).is_err());
    // An inactive row may carry any value; it is never indexed.
    history.record(&[u32::MAX, 1], &[false, true]).unwrap();
}

/// Without history-dependent controls the prompt is never read or stored.
#[test]
fn unused_controls_keep_no_state() {
    let device = Default::default();
    let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[-5, 99]], &device);
    let history = TokenHistory::from_prompt(&GenerationConfig::default(), 4, prompt, None).unwrap();
    for view in [CountView::All, CountView::Generated] {
        let error = history.counts(view).unwrap_err();
        assert!(error.to_string().contains("tracks no"), "{error}");
    }
    assert!(history.banned_tokens(0).is_empty());
}

/// Invalid prompt IDs, shapes, and batches fail with context.
#[test]
fn invalid_prompts_are_rejected() {
    let device = Default::default();
    for value in [-1, 4] {
        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[0, value]], &device);
        let error = TokenHistory::from_prompt(&penalized(), 4, prompt, None).unwrap_err();
        assert!(error.to_string().contains("generation.history"), "{error}");
    }
    let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1]], &device);
    let wrong_mask = TokenMask::new(Tensor::from_bool([[true]].into(), &device));
    assert!(TokenHistory::from_prompt(&penalized(), 4, prompt, Some(wrong_mask)).is_err());
    assert!(TokenHistory::<TestBackend>::empty(&penalized(), 4, 0, &device).is_err());
    let history = Tensor::<TestBackend, 2, Int>::from_ints([[0]], &device);
    assert!(TokenHistory::from_generated_tensor(&penalized(), 4, 2, history, None).is_err());
}

/// The stateless path counts history as generated and its lengths when needed.
#[test]
fn generated_tensor_history_counts_as_output() {
    let device = Default::default();
    let config = GenerationConfig {
        frequency_penalty: 0.5,
        min_new_tokens: 2,
        max_new_tokens: 4,
        ..Default::default()
    };
    let history = Tensor::<TestBackend, 2, Int>::from_ints([[1, 1, 7]], &device);
    let mask = TokenMask::new(Tensor::from_bool([[true, true, false]].into(), &device));
    let history = TokenHistory::from_generated_tensor(&config, 3, 1, history, Some(mask)).unwrap();
    assert_eq!(counts(&history, CountView::Generated), [0, 2, 0]);
    // Frequency with the default scope reads only the generated view.
    assert!(history.counts(CountView::All).is_err());
    assert_eq!(history.generated_lengths(), [2]);

    let both = GenerationConfig {
        repetition_penalty: 1.1,
        ..config
    };
    let history = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 1]], &device);
    let history = TokenHistory::from_generated_tensor(&both, 3, 1, history, None).unwrap();
    assert_eq!(counts(&history, CountView::All), [0, 2, 1]);
    assert_eq!(counts(&history, CountView::Generated), [0, 2, 1]);
}

/// The n-gram index follows Hugging Face's lookup rule for every size.
#[test]
fn ngram_index_bans_seen_continuations() {
    let mut index = NgramIndex::default();
    for token in [1, 2, 3, 1, 2] {
        index.push(token, 3);
    }
    assert_eq!(index.banned(3), [3]);
    let mut unigram = NgramIndex::default();
    for token in [4, 2, 4] {
        unigram.push(token, 1);
    }
    assert_eq!(unigram.banned(1), [2, 4]);
    let mut short = NgramIndex::default();
    short.push(1, 4);
    assert!(short.banned(4).is_empty());
}

/// Histories built directly from tensors on a device match the incremental path.
#[test]
fn incremental_and_bulk_counts_agree() {
    let device = Default::default();
    let tokens = [2_u32, 0, 2, 1];
    let mut incremental = TokenHistory::<TestBackend>::empty(&penalized(), 3, 1, &device).unwrap();
    for token in tokens {
        incremental.record(&[token], &[true]).unwrap();
    }
    let bulk = TokenHistory::from_generated_tensor(
        &penalized(),
        3,
        1,
        Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(tokens.map(i64::from).to_vec(), [1, 4]),
            &device,
        ),
        None,
    )
    .unwrap();
    for view in [CountView::All, CountView::Generated] {
        assert_eq!(counts(&incremental, view), counts(&bulk, view));
    }
}

/// Only the count views a configuration's penalties read are allocated.
#[test]
fn histories_track_only_the_count_views_their_penalties_read() {
    let device = Default::default();
    let cases = [
        (GenerationConfig::default(), [false, false]),
        (
            GenerationConfig {
                repetition_penalty: 1.3,
                ..Default::default()
            },
            [true, false],
        ),
        (
            GenerationConfig {
                presence_penalty: 0.2,
                ..Default::default()
            },
            [false, true],
        ),
        (
            GenerationConfig {
                frequency_penalty: 0.2,
                additive_penalty_scope: PenaltyScope::PromptAndGenerated,
                ..Default::default()
            },
            [true, false],
        ),
        (
            GenerationConfig {
                repetition_penalty: 0.8,
                frequency_penalty: -0.4,
                ..Default::default()
            },
            [true, true],
        ),
    ];
    for (config, [all, generated]) in cases {
        let prompt = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);
        let mut history = TokenHistory::from_prompt(&config, 3, prompt, None).unwrap();
        history.record(&[2], &[true]).unwrap();
        assert_eq!(history.counts(CountView::All).is_ok(), all, "{config:?}");
        assert_eq!(
            history.counts(CountView::Generated).is_ok(),
            generated,
            "{config:?}"
        );
        if all {
            assert_eq!(counts(&history, CountView::All), [0, 1, 2]);
        }
        if generated {
            assert_eq!(counts(&history, CountView::Generated), [0, 0, 1]);
        }
        let empty = TokenHistory::<TestBackend>::empty(&config, 3, 1, &device).unwrap();
        assert_eq!(empty.counts(CountView::All).is_ok(), all);
        assert_eq!(empty.counts(CountView::Generated).is_ok(), generated);
    }
}

/// Recording into one clone never changes another (copy-on-write count buffers).
#[test]
fn cloned_histories_update_independently() {
    let device = Default::default();
    let mut original = TokenHistory::<TestBackend>::empty(&penalized(), 3, 1, &device).unwrap();
    original.record(&[1], &[true]).unwrap();
    let mut branch = original.clone();
    branch.record(&[2], &[true]).unwrap();
    original.record(&[0], &[true]).unwrap();
    for view in [CountView::All, CountView::Generated] {
        assert_eq!(counts(&original, view), [1, 1, 0]);
        assert_eq!(counts(&branch, view), [0, 1, 1]);
    }
    assert_eq!(original.generated_lengths(), [2]);
    assert_eq!(branch.generated_lengths(), [2]);
}

/// Host token rows with a prompt boundary equal a prompt history plus recorded tokens.
#[test]
fn token_rows_match_prompt_history_with_recorded_tokens() {
    let device = Default::default();
    let config = GenerationConfig {
        repetition_penalty: 1.2,
        frequency_penalty: 0.3,
        no_repeat_ngram_size: 2,
        min_new_tokens: 3,
        max_new_tokens: 4,
        ..Default::default()
    };
    // Ragged rows: [prompt | generated] = [1 2 1 | 2 1] and [3 | ] and [ | 3 3].
    let rows: [&[u32]; 3] = [&[1, 2, 1, 2, 1], &[3], &[3, 3]];
    let prompt_lengths = [3, 1, 0];
    let from_rows =
        TokenHistory::<TestBackend>::from_token_rows(&config, 4, &rows, &prompt_lengths, &device)
            .unwrap();

    let prompt =
        Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 1], [3, 0, 0], [0, 0, 0]], &device);
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [
            [true, true, true],
            [true, false, false],
            [false, false, false],
        ]
        .into(),
        &device,
    ));
    let mut recorded = TokenHistory::from_prompt(&config, 4, prompt, Some(mask)).unwrap();
    recorded.record(&[2, 0, 3], &[true, false, true]).unwrap();
    recorded.record(&[1, 0, 3], &[true, false, true]).unwrap();

    for view in [CountView::All, CountView::Generated] {
        assert_eq!(
            counts(&from_rows, view),
            counts(&recorded, view),
            "{view:?}"
        );
    }
    assert_eq!(
        counts(&from_rows, CountView::Generated),
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    );
    assert_eq!(from_rows.generated_lengths(), [2, 0, 2]);
    assert_eq!(from_rows.generated_lengths(), recorded.generated_lengths());
    for row in 0..3 {
        assert_eq!(
            from_rows.banned_tokens(row),
            recorded.banned_tokens(row),
            "row {row}"
        );
    }
    assert_eq!(from_rows.banned_tokens(0), [2]);
    assert_eq!(from_rows.banned_tokens(2), [3]);
}

/// Rows without any token and configurations without state are handled cheaply.
#[test]
fn token_rows_without_tokens_or_controls_keep_minimal_state() {
    let device = Default::default();
    let empty: [&[u32]; 2] = [&[], &[]];
    let history =
        TokenHistory::<TestBackend>::from_token_rows(&penalized(), 3, &empty, &[0, 0], &device)
            .unwrap();
    assert_eq!(counts(&history, CountView::All), [0; 6]);
    assert_eq!(history.generated_lengths(), [0, 0]);
    let plain = TokenHistory::<TestBackend>::from_token_rows(
        &GenerationConfig::default(),
        3,
        &[&[2, 1]],
        &[1],
        &device,
    )
    .unwrap();
    assert!(plain.counts(CountView::All).is_err());
    assert_eq!(plain.generated_lengths(), [1]);
}

/// Malformed host rows fail with context before any device work.
#[test]
fn token_rows_reject_malformed_input() {
    let device = Default::default();
    let config = penalized();
    let build = |rows: &[&[u32]], lengths: &[usize]| {
        TokenHistory::<TestBackend>::from_token_rows(&config, 4, rows, lengths, &device)
            .unwrap_err()
            .to_string()
    };
    assert!(build(&[], &[]).contains("generation.history_rows"));
    assert!(build(&[&[1]], &[0, 0]).contains("generation.history_rows"));
    assert!(build(&[&[1]], &[2]).contains("fewer than its prompt length 2"));
    let error = build(&[&[1], &[0, 4]], &[0, 1]);
    assert!(error.contains("token 4 at row 1, column 1"), "{error}");
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(48))]

    /// Host rows reproduce prompt history plus per-step records for random ragged batches.
    #[test]
    fn token_rows_equal_incremental_history(
        rows in proptest::collection::vec(
            (proptest::collection::vec(0_u32..5, 0..5), proptest::collection::vec(0_u32..5, 0..4)),
            1..4,
        ),
        repetition in proptest::bool::ANY,
        prompt_scope in proptest::bool::ANY,
        ngram in 0_usize..4,
    ) {
        let device = Default::default();
        let vocabulary = 5;
        let config = GenerationConfig {
            repetition_penalty: if repetition { 1.5 } else { 1.0 },
            presence_penalty: 0.25,
            additive_penalty_scope: if prompt_scope {
                PenaltyScope::PromptAndGenerated
            } else {
                PenaltyScope::Generated
            },
            no_repeat_ngram_size: ngram,
            ..Default::default()
        };
        let batch = rows.len();
        let prompt_width = rows.iter().map(|(prompt, _)| prompt.len()).max().unwrap();
        let mut prompt_ids = Vec::new();
        let mut prompt_valid = Vec::new();
        for (prompt, _) in &rows {
            for column in 0..prompt_width {
                prompt_ids.push(prompt.get(column).map_or(0, |&token| token as i32));
                prompt_valid.push(column < prompt.len());
            }
        }
        let mut incremental = TokenHistory::<TestBackend>::from_prompt(
            &config,
            vocabulary,
            Tensor::from_data(TensorData::new(prompt_ids, [batch, prompt_width]), &device),
            Some(TokenMask::new(Tensor::from_data(
                TensorData::new(prompt_valid, [batch, prompt_width]),
                &device,
            ))),
        )
        .unwrap();
        let steps = rows.iter().map(|(_, generated)| generated.len()).max().unwrap();
        for step in 0..steps {
            let tokens = rows.iter().map(|(_, generated)| generated.get(step).copied().unwrap_or(0)).collect::<Vec<_>>();
            let active = rows.iter().map(|(_, generated)| step < generated.len()).collect::<Vec<_>>();
            incremental.record(&tokens, &active).unwrap();
        }
        let joined = rows
            .iter()
            .map(|(prompt, generated)| prompt.iter().chain(generated).copied().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let slices = joined.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let prompt_lengths = rows.iter().map(|(prompt, _)| prompt.len()).collect::<Vec<_>>();
        let from_rows = TokenHistory::<TestBackend>::from_token_rows(
            &config,
            vocabulary,
            &slices,
            &prompt_lengths,
            &device,
        )
        .unwrap();
        for view in [CountView::All, CountView::Generated] {
            proptest::prop_assert_eq!(
                from_rows.counts(view).is_ok(),
                incremental.counts(view).is_ok()
            );
            if from_rows.counts(view).is_ok() {
                proptest::prop_assert_eq!(counts(&from_rows, view), counts(&incremental, view));
            }
        }
        proptest::prop_assert_eq!(from_rows.generated_lengths(), incremental.generated_lengths());
        for row in 0..batch {
            proptest::prop_assert_eq!(from_rows.banned_tokens(row), incremental.banned_tokens(row));
        }
    }
}

/// Times per-step count updates against the previous clone-and-scatter of both views.
///
/// The previous update scattered into `counts.clone()` for both views, which
/// keeps a second reference alive and forces a full-buffer copy on backends
/// with in-place scatter. On Flex every scatter copies regardless, so this
/// measures only the saved view; CubeCL additionally avoids the copy.
///
/// Run with `cargo test --release --lib history_record_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing benchmark; run in release mode with --ignored --nocapture"]
fn history_record_timing() {
    use burn::tensor::IndexingUpdateOp;
    use std::time::Instant;

    let device = Default::default();
    let (batch, vocabulary, steps) = (8, 256_000, 64);
    let config = GenerationConfig {
        repetition_penalty: 1.1,
        ..Default::default()
    };
    let tokens = |step: usize| {
        (0..batch)
            .map(|row| ((step * 7_919 + row * 104_729) % vocabulary) as u32)
            .collect::<Vec<_>>()
    };
    let active = vec![true; batch];

    let started = Instant::now();
    let mut previous_all = Tensor::<TestBackend, 2, Int>::zeros([batch, vocabulary], &device);
    let mut previous_generated = previous_all.clone();
    for step in 0..steps {
        let ids = tokens(step).into_iter().map(i64::from).collect::<Vec<_>>();
        let ids =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(ids, [batch, 1]), &device);
        let ones = Tensor::<TestBackend, 2, Int>::ones([batch, 1], &device);
        previous_all =
            previous_all
                .clone()
                .scatter(1, ids.clone(), ones.clone(), IndexingUpdateOp::Add);
        previous_generated =
            previous_generated
                .clone()
                .scatter(1, ids, ones, IndexingUpdateOp::Add);
    }
    let previous_counts = previous_all.into_data().to_vec::<i32>().unwrap();
    let previous = started.elapsed();

    let started = Instant::now();
    let mut history =
        TokenHistory::<TestBackend>::empty(&config, vocabulary, batch, &device).unwrap();
    for step in 0..steps {
        history.record(&tokens(step), &active).unwrap();
    }
    let current_counts = counts(&history, CountView::All);
    let current = started.elapsed();

    assert_eq!(previous_counts, current_counts);
    println!(
        "count updates, batch {batch}, vocabulary {vocabulary}, {steps} steps: previous {:.3} ms/step, current {:.3} ms/step ({:.1}x)",
        previous.as_secs_f64() * 1e3 / steps as f64,
        current.as_secs_f64() * 1e3 / steps as f64,
        previous.as_secs_f64() / current.as_secs_f64()
    );
}
