//! Token-history tests: count tensors, n-gram index, validation, and incremental updates.

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, TensorData},
};

use super::{NgramIndex, TokenHistory};
use crate::{GenerationConfig, TokenMask};

/// Deterministic F32 CPU backend.
type TestBackend = Flex<f32, i32>;

/// Reads a `[batch, vocabulary]` count tensor to host integers.
fn counts(tensor: &Tensor<TestBackend, 2, Int>) -> Vec<i32> {
    tensor
        .clone()
        .into_data()
        .to_vec::<i32>()
        .expect("integer counts")
}

/// Configuration that enables count tensors.
fn penalized() -> GenerationConfig {
    GenerationConfig {
        repetition_penalty: 1.2,
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
    let state = history.counts().expect("penalties need counts");
    assert_eq!(counts(&state.all), [0, 2, 0, 0, 0, 0, 1, 1]);
    assert_eq!(counts(&state.generated), [0; 8]);
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
    let state = history.counts().unwrap();
    assert_eq!(counts(&state.all), [0, 0, 2, 1, 0, 0]);
    assert_eq!(counts(&state.generated), [0, 0, 2, 1, 0, 0]);
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
    assert!(history.counts().is_none());
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
    let state = history.counts().unwrap();
    assert_eq!(counts(&state.generated), [0, 2, 0]);
    assert_eq!(counts(&state.all), [0, 2, 0]);
    assert_eq!(history.generated_lengths(), [2]);
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
    assert_eq!(
        counts(&incremental.counts().unwrap().generated),
        counts(&bulk.counts().unwrap().generated)
    );
}
