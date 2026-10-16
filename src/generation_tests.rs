//! Generation-policy tests for deterministic sampling, penalties, stopping, and invalid requests.

use std::collections::BTreeMap;

use burn::backend::Flex;
use burn::tensor::{DType, Int, Tensor, TensorData, Tolerance};

use super::{
    FinishReason, GenerationConfig, GenerationTracker, LogitsProcessor, PenaltyScope,
    SamplingStrategy, splitmix64_uniform,
};
use crate::TokenMask;

/// Deterministic F32 CPU backend for token-selection and stopping fixtures.
type TestBackend = Flex<f32, i32>;

/// Invalid sampling controls fail before any tensor operation can panic.
#[test]
fn configuration_validation_rejects_invalid_ranges() {
    let invalid = [
        GenerationConfig {
            max_new_tokens: 0,
            ..Default::default()
        },
        GenerationConfig {
            temperature: 0.0,
            ..Default::default()
        },
        GenerationConfig {
            top_k: Some(5),
            ..Default::default()
        },
        GenerationConfig {
            top_p: Some(0.0),
            ..Default::default()
        },
        GenerationConfig {
            repetition_penalty: -1.0,
            ..Default::default()
        },
        GenerationConfig {
            stop_sequences: vec![Vec::new()],
            ..Default::default()
        },
        GenerationConfig {
            top_p: Some(f64::NAN),
            ..Default::default()
        },
        GenerationConfig {
            repetition_penalty: f64::INFINITY,
            ..Default::default()
        },
        GenerationConfig {
            presence_penalty: f64::NAN,
            ..Default::default()
        },
        GenerationConfig {
            eos_token_ids: vec![4],
            ..Default::default()
        },
        GenerationConfig {
            pad_token_id: Some(4),
            ..Default::default()
        },
        GenerationConfig {
            stop_sequences: vec![vec![4]],
            ..Default::default()
        },
    ];
    for config in invalid {
        assert!(config.validate(4).is_err());
    }
    assert!(GenerationConfig::default().validate(0).is_err());
}

/// Greedy selection preserves row boundaries and returns each maximum.
#[test]
fn greedy_selects_each_rows_maximum() {
    let device = Default::default();
    let processor = LogitsProcessor::<TestBackend>::new(GenerationConfig::default(), 4, &device)
        .expect("valid processor");
    let logits = Tensor::from_floats([[0.0, 4.0, 1.0, -1.0], [3.0, 2.0, 7.0, 1.0]], &device);
    processor
        .process_and_select(logits, None)
        .expect("valid scores")
        .into_data()
        .assert_eq(&TensorData::from([1_i32, 2]), false);
}

/// Selected-token log probabilities describe the processed distribution.
#[test]
fn selection_reports_processed_log_probability() {
    let device = Default::default();
    let processor = LogitsProcessor::<TestBackend>::new(GenerationConfig::default(), 2, &device)
        .expect("valid processor");
    let selection = processor
        .process_and_select_with_log_probs(Tensor::from_floats([[0.0, 1.0]], &device), None)
        .expect("valid scores");
    selection
        .token_ids
        .into_data()
        .assert_eq(&TensorData::from([1_i32]), false);
    selection
        .log_probabilities
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::from([-0.313_261_7_f32]),
            Tolerance::absolute(1e-6),
        );
}

/// Every public processor boundary rejects empty or mismatched score tensors.
#[test]
fn processor_rejects_invalid_shapes_before_backend_sampling() {
    let device = Default::default();
    let processor = LogitsProcessor::<TestBackend>::new(GenerationConfig::default(), 4, &device)
        .expect("valid processor");
    assert!(
        processor
            .process(Tensor::zeros([0, 4], &device), None)
            .is_err()
    );
    assert!(processor.select(Tensor::zeros([1, 3], &device)).is_err());
    assert!(
        processor
            .process(
                Tensor::zeros([2, 4], &device),
                Some(Tensor::<TestBackend, 2, Int>::zeros([1, 1], &device)),
            )
            .is_err()
    );
}

/// Penalty history validates token ranges before constructing one-hot tensors.
#[test]
fn penalty_history_rejects_invalid_ids_and_accepts_empty_history() {
    let device = Default::default();
    let config = GenerationConfig {
        presence_penalty: 0.5,
        ..Default::default()
    };
    let processor =
        LogitsProcessor::<TestBackend>::new(config, 4, &device).expect("valid processor");
    let logits = Tensor::from_floats([[3.0, 2.0, 1.0, 0.0]], &device);
    for history in [
        Tensor::<TestBackend, 2, Int>::from_ints([[-1]], &device),
        Tensor::<TestBackend, 2, Int>::from_ints([[4]], &device),
    ] {
        assert!(processor.process(logits.clone(), Some(history)).is_err());
    }
    processor
        .process(
            logits.clone(),
            Some(Tensor::<TestBackend, 2, Int>::zeros([1, 0], &device)),
        )
        .expect("an empty history applies no penalties")
        .into_data()
        .assert_eq(&logits.into_data(), false);
}

/// Padded sentinels neither reach one-hot indexing nor accrue token penalties.
#[test]
fn masked_history_ignores_sentinels_and_validates_mask_shape() {
    let device = Default::default();
    let config = GenerationConfig {
        presence_penalty: 2.0,
        ..Default::default()
    };
    let processor =
        LogitsProcessor::<TestBackend>::new(config, 4, &device).expect("valid processor");
    let logits = Tensor::from_floats([[0.0, 4.0, 3.0, 2.0]], &device);
    let history = Tensor::<TestBackend, 2, Int>::from_ints([[99, 1]], &device);
    let mask = TokenMask::new(Tensor::from_bool([[false, true]].into(), &device));
    let processed = processor
        .process_with_mask(logits, Some(history), Some(mask))
        .expect("masked sentinel is never indexed")
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(processed, [0.0, 2.0, 3.0, 2.0]);

    let history = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1]], &device);
    let wrong = TokenMask::new(Tensor::from_bool([[true]].into(), &device));
    assert!(
        processor
            .process_with_mask(Tensor::zeros([1, 4], &device), Some(history), Some(wrong))
            .is_err()
    );
    assert!(
        processor
            .process_with_mask(
                Tensor::zeros([1, 4], &device),
                None,
                Some(TokenMask::new(Tensor::from_bool([[true]].into(), &device)))
            )
            .is_err()
    );
}

/// Independent processors reproduce the same multi-step stochastic stream.
#[test]
fn sampling_seed_is_deterministic() {
    let device = Default::default();
    let config = GenerationConfig {
        strategy: SamplingStrategy::Sample,
        seed: 91,
        ..Default::default()
    };
    let logits = Tensor::<TestBackend, 2>::zeros([64, 4], &device);
    let first =
        LogitsProcessor::<TestBackend>::new(config.clone(), 4, &device).expect("valid processor");
    let second = LogitsProcessor::<TestBackend>::new(config, 4, &device).expect("valid processor");
    for _ in 0..4 {
        let first_ids = first
            .process_and_select(logits.clone(), None)
            .expect("valid scores")
            .into_data();
        let second_ids = second
            .process_and_select(logits.clone(), None)
            .expect("valid scores")
            .into_data();
        assert_eq!(first_ids, second_ids);
    }
}

/// Runtime float casts retain the same uniform stream and inverse-CDF bins.
#[test]
fn sampling_runtime_dtypes_preserve_seeded_uniforms() {
    let device = Default::default();
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let processor = LogitsProcessor::<TestBackend>::new(
            GenerationConfig {
                strategy: SamplingStrategy::Sample,
                seed: 91,
                ..Default::default()
            },
            4,
            &device,
        )
        .expect("valid processor");
        for step in 0..3 {
            let actual = processor
                .select(Tensor::<TestBackend, 2>::zeros([64, 4], &device).cast(dtype))
                .expect("runtime dtype selection")
                .into_data()
                .to_vec::<i32>()
                .expect("integer selections");
            let expected: Vec<i32> = (0..64)
                .map(|row| {
                    let uniform = splitmix64_uniform(91, step * 64 + row);
                    // Uniform logits have exact quarter-interval bins.
                    (1..=4).filter(|&bin| (bin as f32) * 0.25 < uniform).count() as i32
                })
                .collect();
            assert_eq!(actual, expected, "dtype {dtype:?}, draw {step}");
        }
    }
}

/// Masked penalties, temperature, top-k, top-p, and sampling compose in each dtype.
#[test]
fn runtime_dtypes_support_penalties_and_probability_filters() {
    let device = Default::default();
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let penalty_processor = LogitsProcessor::<TestBackend>::new(
            GenerationConfig {
                repetition_penalty: 2.0,
                presence_penalty: 0.5,
                ..Default::default()
            },
            4,
            &device,
        )
        .expect("valid penalties");
        let history = Tensor::from_ints([[0, 1, 99], [1, 1, 99]], &device);
        let mask = TokenMask::new(Tensor::from_bool(
            [[true, true, false], [true, true, false]].into(),
            &device,
        ));
        let logits = Tensor::from_floats([[4.0, -3.0, 2.0, 1.0], [-1.0, 4.0, 2.0, 0.0]], &device)
            .cast(dtype);
        let penalized = penalty_processor
            .process_with_mask(logits, Some(history), Some(mask))
            .expect("masked mixed-dtype penalties");
        assert_eq!(penalized.dtype(), dtype);
        assert_eq!(
            super::metadata::host_float_data_f64(penalized.clone().into_data(), "penalties")
                .expect("floating output"),
            [1.5, -6.5, 2.0, 1.0, -1.0, 1.5, 2.0, 0.0],
            "runtime dtype {dtype:?}"
        );
        let sampler = LogitsProcessor::<TestBackend>::new(
            GenerationConfig {
                strategy: SamplingStrategy::Sample,
                temperature: 0.5,
                top_k: Some(2),
                top_p: Some(0.7),
                ..Default::default()
            },
            4,
            &device,
        )
        .expect("valid filters");
        let processed = sampler
            .process(penalized, None)
            .expect("runtime dtype filters");
        assert_eq!(processed.dtype(), dtype);
        let values: Vec<f64> = super::metadata::host_float_data_f64(
            processed.clone().into_data(),
            "probability filters",
        )
        .expect("floating filtered output");
        // After temperature, the largest two scores differ by one, so
        // their larger probability exceeds 0.7 and only token two remains.
        assert_eq!(values.iter().filter(|value| value.is_finite()).count(), 2);
        assert_eq!(values[2], 4.0);
        assert_eq!(values[6], 4.0);
        sampler
            .select(processed)
            .expect("runtime dtype sample")
            .into_data()
            .assert_eq(&TensorData::from([2_i32, 2]), false);
    }
    assert_eq!(
        super::probability_accumulation_dtype(DType::F64),
        DType::F64
    );
}

/// Builds one `[1, 4]` logits row tagged as Flex32 storage.
fn flex32_logits(
    device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<TestBackend, 2> {
    let mut data = TensorData::from([[1.0_f32, 2.0, 3.0, 4.0]]);
    data.dtype = DType::Flex32;
    Tensor::from_data(data, (device, DType::Flex32))
}

/// Unsupported Flex32 sorts fail before Burn's panicking fallback; greedy never sorts.
#[test]
fn flex32_sort_filters_fail_closed_before_backend_operations() {
    let device = Default::default();
    let filters = [
        (Some(1), None),
        (Some(4), None),
        (None, Some(0.7)),
        (Some(2), Some(1.0)),
    ];
    for (top_k, top_p) in filters {
        let config = GenerationConfig {
            strategy: SamplingStrategy::Sample,
            top_k,
            top_p,
            ..Default::default()
        };
        let processor =
            LogitsProcessor::<TestBackend>::new(config.clone(), 4, &device).expect("valid config");
        // Top-k equal to the vocabulary keeps every token and needs no sort.
        if config.sorts_vocabulary(4) {
            let error = processor
                .process(flex32_logits(&device), None)
                .expect_err("Flex32 sort must fail before unsupported CPU tensor math");
            assert!(matches!(error, crate::Error::BackendCapability(_)));
            assert!(error.to_string().contains("Flex32 top-k/top-p"));
        } else {
            assert_eq!((top_k, top_p), (Some(4), None));
        }
        // Greedy decoding ignores truncation, so the same filters are inert.
        let greedy = LogitsProcessor::<TestBackend>::new(
            GenerationConfig {
                strategy: SamplingStrategy::Greedy,
                ..config
            },
            4,
            &device,
        )
        .expect("valid greedy config");
        assert_eq!(
            greedy
                .process(flex32_logits(&device), None)
                .expect("greedy requests no sort")
                .dtype(),
            DType::Flex32
        );
    }
    assert_eq!(
        super::probability_accumulation_dtype(DType::Flex32),
        DType::F32
    );
}

/// Stream cloning snapshots state, uniforms stay interior, and exhaustion is safe.
#[test]
fn sampling_stream_boundaries_are_explicit() {
    for (seed, index) in [(0, 0), (u64::MAX, 0), (0, u64::MAX), (u64::MAX, u64::MAX)] {
        let uniform = splitmix64_uniform(seed, index);
        assert!(uniform > 0.0 && uniform < 1.0);
    }

    let device = Default::default();
    let config = GenerationConfig {
        strategy: SamplingStrategy::Sample,
        seed: 17,
        ..Default::default()
    };
    let processor =
        LogitsProcessor::<TestBackend>::new(config, 3, &device).expect("valid processor");
    processor
        .process_and_select(Tensor::zeros([1, 3], &device), None)
        .expect("first draw");
    let clone = processor.clone();
    assert_eq!(
        processor
            .process_and_select(Tensor::zeros([8, 3], &device), None)
            .expect("continued draw")
            .into_data(),
        clone
            .process_and_select(Tensor::zeros([8, 3], &device), None)
            .expect("cloned draw")
            .into_data()
    );

    processor
        .sample_counter
        .store(u64::MAX - 1, std::sync::atomic::Ordering::Relaxed);
    assert!(processor.select(Tensor::zeros([2, 3], &device)).is_err());
}

/// Applies `top_k` sampling filtering to host rows and returns the processed scores.
fn top_k_filtered(rows: &[Vec<f32>], top_k: usize) -> Vec<f32> {
    let device = Default::default();
    let vocabulary = rows[0].len();
    let config = GenerationConfig {
        strategy: SamplingStrategy::Sample,
        top_k: Some(top_k),
        ..Default::default()
    };
    let processor =
        LogitsProcessor::<TestBackend>::new(config, vocabulary, &device).expect("valid processor");
    let flat = rows.iter().flatten().copied().collect::<Vec<_>>();
    let logits = Tensor::from_data(TensorData::new(flat, [rows.len(), vocabulary]), &device);
    processor
        .process(logits, None)
        .expect("valid scores")
        .into_data()
        .to_vec::<f32>()
        .expect("float output")
}

/// Hugging Face's top-k rule: keep `z_i >= z_(k)`, the k-th greatest score of the row.
fn reference_top_k(row: &[f32], top_k: usize) -> Vec<f32> {
    let mut sorted = row.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    let boundary = sorted[top_k - 1];
    row.iter()
        .map(|&score| {
            if score < boundary {
                f32::NEG_INFINITY
            } else {
                score
            }
        })
        .collect()
}

/// Top-k keeps every token tied with the k-th score, as Hugging Face's `TopKLogitsWarper` does.
///
/// With scores `[3, 2, 2, 1]` and `k = 2` the boundary score is 2, so both
/// tied tokens survive; keeping exactly two would drop one of them by sort
/// order, which neither Hugging Face nor this crate does.
#[test]
fn top_k_keeps_ties_with_the_boundary_score() {
    let output = top_k_filtered(&[vec![3.0, 2.0, 2.0, 1.0]], 2);
    assert_eq!(output, [3.0, 2.0, 2.0, f32::NEG_INFINITY]);
    // Without ties exactly k tokens survive, in any position.
    let output = top_k_filtered(&[vec![0.5, 4.0, -1.0, 2.0]], 2);
    assert_eq!(output, [f32::NEG_INFINITY, 4.0, f32::NEG_INFINITY, 2.0]);
    // An all-equal row keeps everything; k = 1 keeps every maximum.
    assert_eq!(top_k_filtered(&[vec![1.0; 4]], 1), [1.0; 4]);
    let output = top_k_filtered(&[vec![5.0, 1.0, 5.0, 0.0]], 1);
    assert_eq!(output, [5.0, f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY]);
    // Rows are independent, and earlier -inf constraints stay excluded.
    let output = top_k_filtered(
        &[
            vec![f32::NEG_INFINITY, 1.0, 1.0, 0.0],
            vec![2.0, 2.0, 2.0, 3.0],
        ],
        2,
    );
    assert_eq!(
        output,
        [
            f32::NEG_INFINITY,
            1.0,
            1.0,
            f32::NEG_INFINITY,
            2.0,
            2.0,
            2.0,
            3.0
        ]
    );
}

proptest::proptest! {
    /// Top-k filtering matches Hugging Face's boundary rule on rows with many ties.
    #[test]
    fn top_k_matches_the_hugging_face_boundary_rule(
        rows in proptest::collection::vec(proptest::collection::vec(-3_i8..=3, 6), 1..4),
        top_k in 1_usize..=6,
    ) {
        let rows = rows
            .iter()
            .map(|row| row.iter().map(|&value| f32::from(value) * 0.5).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let expected = rows
            .iter()
            .flat_map(|row| reference_top_k(row, top_k))
            .collect::<Vec<_>>();
        proptest::prop_assert_eq!(top_k_filtered(&rows, top_k), expected);
    }
}

/// Nucleus filtering includes the item that crosses the target probability.
#[test]
fn top_p_keeps_crossing_token() {
    let device = Default::default();
    let config = GenerationConfig {
        strategy: SamplingStrategy::Sample,
        top_p: Some(0.8),
        ..Default::default()
    };
    let processor =
        LogitsProcessor::<TestBackend>::new(config, 3, &device).expect("valid processor");
    let output = processor
        .process(Tensor::from_floats([[2.0, 1.0, 0.0]], &device), None)
        .expect("valid scores")
        .into_data()
        .to_vec::<f32>()
        .expect("float output");
    assert!(output[0].is_finite());
    assert!(output[1].is_finite());
    assert!(output[2].is_infinite());
}

/// Repetition is sign-aware and presence applies only to observed IDs.
#[test]
fn token_penalties_match_standard_logit_rules() {
    let device = Default::default();
    let config = GenerationConfig {
        repetition_penalty: 2.0,
        presence_penalty: 0.5,
        ..Default::default()
    };
    let processor =
        LogitsProcessor::<TestBackend>::new(config, 4, &device).expect("valid processor");
    let logits = Tensor::from_floats([[4.0, -3.0, 2.0, 1.0]], &device);
    let history = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 1]], &device);
    processor
        .process(logits, Some(history))
        .expect("valid scores")
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::from([[1.5, -6.5, 2.0, 1.0]]),
            Tolerance::absolute(1e-5),
        );
}

/// Tracker gives EOS precedence and never mutates completed sequences.
#[test]
fn tracker_handles_eos_and_is_idempotent() {
    let config = GenerationConfig {
        eos_token_ids: vec![2],
        stop_sequences: vec![vec![1, 2]],
        ..Default::default()
    };
    let mut tracker = GenerationTracker::new(config, 8, 1).expect("valid tracker");
    assert_eq!(tracker.push(0, 1).expect("in range"), None);
    assert_eq!(
        tracker.push(0, 2).expect("in range"),
        Some(&FinishReason::EndOfSequence { token_id: 2 })
    );
    tracker
        .push(0, u32::MAX)
        .expect("completed push is ignored before token validation");
    assert_eq!(tracker.generated(0).expect("in range"), &[1, 2]);
    assert!(tracker.all_finished());
}

/// Longest suffix wins and length termination remains a fallback.
#[test]
fn tracker_resolves_stops_then_length() {
    let config = GenerationConfig {
        max_new_tokens: 3,
        stop_sequences: vec![vec![2], vec![1, 2]],
        ..Default::default()
    };
    let mut tracker = GenerationTracker::new(config, 8, 2).expect("valid tracker");
    tracker.push(0, 1).expect("in range");
    assert_eq!(
        tracker.push(0, 2).expect("in range"),
        Some(&FinishReason::StopSequence {
            index: 1,
            length: 2,
        })
    );
    tracker.push(1, 4).expect("in range");
    tracker.push(1, 5).expect("in range");
    assert_eq!(
        tracker.push(1, 6).expect("in range"),
        Some(&FinishReason::Length)
    );
    assert!(tracker.finish_reason(2).is_err());
    assert!(tracker.generated(2).is_err());
    assert!(tracker.push(2, 0).is_err());
}

/// Tracker bounds and allocation failures are recoverable and mutation-free.
#[test]
fn tracker_rejects_invalid_tokens_and_impossible_batch_capacity() {
    let mut tracker =
        GenerationTracker::new(GenerationConfig::default(), 4, 2).expect("valid tracker");
    assert_eq!(tracker.batch_size(), 2);
    assert_eq!(tracker.vocabulary_size(), 4);
    assert!(tracker.push(0, 4).is_err());
    assert!(tracker.generated(0).expect("in range").is_empty());
    assert_eq!(tracker.finish_reason(0).expect("in range"), None);
    assert!(
        GenerationTracker::new(GenerationConfig::default(), 4, usize::MAX).is_err(),
        "capacity overflow must not reach an infallible allocation"
    );
}

/// Empty generation batches fail explicitly without changing valid processor controls.
#[test]
fn empty_tracker_batches_are_rejected_with_the_callers_exact_controls() {
    let config = GenerationConfig {
        max_new_tokens: 3,
        min_new_tokens: 1,
        strategy: SamplingStrategy::Sample,
        temperature: 0.75,
        top_k: Some(2),
        top_p: Some(0.9),
        min_p: Some(0.05),
        repetition_penalty: 1.1,
        presence_penalty: 0.2,
        frequency_penalty: 0.1,
        additive_penalty_scope: PenaltyScope::PromptAndGenerated,
        no_repeat_ngram_size: 3,
        logit_bias: BTreeMap::from([(1, 0.5)]),
        seed: 91,
        eos_token_ids: vec![3],
        pad_token_id: Some(0),
        stop_sequences: vec![vec![1, 2]],
    };
    let processor = LogitsProcessor::<TestBackend>::new(config.clone(), 4, &Default::default())
        .expect("valid reusable generation controls");
    assert_eq!(processor.config(), &config);
    assert_eq!(processor.clone().config(), &config);
    let error = GenerationTracker::new(processor.config().clone(), 4, 0)
        .expect_err("there is no sequence to track in an empty batch");
    assert!(matches!(error, crate::Error::Generation(ref reason)
        if reason == "generation batch_size must be greater than zero"));
    let mut tracker = GenerationTracker::new(processor.config().clone(), 4, 1).unwrap();
    assert_eq!(tracker.push(0, 1).unwrap(), None);
    assert_eq!(
        tracker.push(0, 2).unwrap(),
        Some(&FinishReason::StopSequence {
            index: 0,
            length: 2
        })
    );
    assert_eq!(tracker.generated(0).unwrap(), &[1, 2]);
    assert_eq!(processor.config(), &config);
}

// ---------------------------------------------------------------------------
// Sampler correctness (inverse CDF, greedy semantics, precision)
// ---------------------------------------------------------------------------

/// Deterministic pseudo-normal logits from a linear congruential generator.
///
/// Summing four uniforms on `[-1, 1)` approximates a normal shape; `scale`
/// sets the spread. Fixtures stay reproducible without a random dependency.
fn pseudo_normal_logits(count: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / (1_u64 << 24) as f32) * 2.0 - 1.0
    };
    (0..count)
        .map(|_| scale * (uniform() + uniform() + uniform() + uniform()))
        .collect()
}

/// Reads every element of a float tensor as `f64`, whatever its runtime dtype.
fn host_values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f64> {
    super::metadata::host_float_data_f64(tensor.into_data(), "test values").expect("float data")
}

/// A large vocabulary's F32 running sum ends below one, yet a masked last ID is never drawn.
///
/// With 152k realistic logits the backend's F32 softmax and sequential
/// cumulative sum end measurably below one (about `7e-6` for this fixture;
/// other seeds overshoot instead). The old unscaled rule counted every running
/// sum as below a uniform above that total, reached `V`, and clamped to the
/// last token, which is masked here and has zero probability.
#[test]
fn inverse_cdf_never_selects_a_zero_probability_tail_at_vocabulary_scale() {
    let device = Default::default();
    let vocabulary = 151_936;
    let mut values = pseudo_normal_logits(vocabulary, 1.0, 0);
    values[vocabulary - 1] = f32::NEG_INFINITY;
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(values.clone(), [1, vocabulary]),
        &device,
    );
    let cumulative = burn::tensor::activation::softmax(logits.clone(), 1).cumsum(1);
    let total = host_values(cumulative.clone().narrow(1, vocabulary - 1, 1))[0];
    let largest_uniform = 1.0 - 2.0_f32.powi(-24);
    assert!(
        total < f64::from(largest_uniform),
        "fixture must exhibit a cumulative deficit, total = {total}"
    );
    // The previous rule: every running sum is below u, so the count is V.
    let unscaled = cumulative
        .lower_elem(largest_uniform)
        .int()
        .sum()
        .into_data()
        .to_vec::<i32>()
        .unwrap()[0];
    assert_eq!(unscaled as usize, vocabulary);
    for uniform in [largest_uniform, 0.999_999, 0.5, 2.0_f32.powi(-24)] {
        let selected = super::sample_with_uniforms(logits.clone(), vec![uniform])
            .into_data()
            .to_vec::<i32>()
            .expect("integer selection")[0] as usize;
        assert!(
            selected < vocabulary - 1,
            "u = {uniform} selected {selected}"
        );
        assert!(values[selected].is_finite());
    }
}

/// Uniform 4-way bins are unchanged by total scaling because their running sum is exact.
#[test]
fn inverse_cdf_keeps_exact_bins_for_exact_totals() {
    let device = Default::default();
    let logits = Tensor::<TestBackend, 2>::zeros([4, 4], &device);
    let uniforms = vec![0.1, 0.3, 0.6, 0.9];
    let selected = super::sample_with_uniforms(logits, uniforms)
        .into_data()
        .to_vec::<i32>()
        .expect("integer selection");
    assert_eq!(selected, [0, 1, 2, 3]);
}

/// Greedy decoding ignores temperature and every truncation warper, like Hugging Face.
#[test]
fn greedy_ignores_temperature_and_truncation_warpers() {
    let device = Default::default();
    let processor = LogitsProcessor::<TestBackend>::new(
        GenerationConfig {
            strategy: SamplingStrategy::Greedy,
            temperature: 0.25,
            top_k: Some(1),
            top_p: Some(0.3),
            min_p: Some(0.9),
            ..Default::default()
        },
        4,
        &device,
    )
    .expect("valid processor");
    let logits = Tensor::<TestBackend, 2>::from_floats([[1.0, 3.0, 2.0, 0.5]], &device);
    let processed = processor
        .process(logits.clone(), None)
        .expect("greedy processing");
    assert_eq!(host_values(processed.clone()), host_values(logits.clone()));
    let selection = processor
        .select_with_log_probs(processed)
        .expect("greedy selection");
    assert_eq!(
        selection.token_ids.into_data().to_vec::<i32>().unwrap(),
        [1]
    );
    // The reported score is the untruncated model distribution, not log(1) = 0.
    let exact = 3.0
        - [1.0_f64, 3.0, 2.0, 0.5]
            .map(f64::exp)
            .iter()
            .sum::<f64>()
            .ln();
    let reported = host_values(selection.log_probabilities)[0];
    assert!((reported - exact).abs() < 1e-6, "{reported} vs {exact}");
}

/// Half-precision logits report F32 log probabilities with no F16 log-sum-exp error.
#[test]
fn half_precision_log_probabilities_are_computed_in_f32() {
    let device = Default::default();
    let processor = LogitsProcessor::<TestBackend>::new(GenerationConfig::default(), 5, &device)
        .expect("valid processor");
    let values = [7.3_f32, -2.1, 6.9, 0.25, 7.25];
    for dtype in [DType::F16, DType::BF16] {
        let logits =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(values.to_vec(), [1, 5]), &device)
                .cast(dtype);
        // Reference: exact F64 log-softmax of the already-rounded half values.
        let rounded = host_values(logits.clone());
        let maximum = rounded.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let normalizer = maximum
            + rounded
                .iter()
                .map(|value| (value - maximum).exp())
                .sum::<f64>()
                .ln();
        let selection = processor
            .select_with_log_probs(logits)
            .expect("half-precision selection");
        assert_eq!(selection.log_probabilities.dtype(), DType::F32);
        let token = selection.token_ids.into_data().to_vec::<i32>().unwrap()[0] as usize;
        let reported = host_values(selection.log_probabilities)[0];
        assert!(
            (reported - (rounded[token] - normalizer)).abs() < 1e-6,
            "{dtype:?}: {reported} vs {}",
            rounded[token] - normalizer
        );
    }
}

// ---------------------------------------------------------------------------
// Token history, penalties, and the new controls
// ---------------------------------------------------------------------------

/// Builds a one-row prompt history with the given generated tokens recorded.
fn history_after(
    config: &GenerationConfig,
    vocabulary: usize,
    prompt: &[i32],
    generated: &[u32],
) -> super::TokenHistory<TestBackend> {
    let device = Default::default();
    let prompt = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(prompt.to_vec(), [1, prompt.len()]),
        &device,
    );
    let mut history = super::TokenHistory::from_prompt(config, vocabulary, prompt, None)
        .expect("valid prompt history");
    for &token in generated {
        history.record(&[token], &[true]).expect("valid record");
    }
    history
}

/// Repetition counts the prompt; presence and frequency count generated tokens by default.
#[test]
fn additive_penalties_count_generated_tokens_unless_scoped_to_the_prompt() {
    let device = Default::default();
    let logits = [4.0_f32, -3.0, 2.0, 1.0, 1.0];
    for (scope, expected) in [
        // Prompt tokens 0 and 1 are only repetition-penalized; token 2 was
        // generated twice: 2 / 2 - 0.25 * 2 - 0.5.
        (PenaltyScope::Generated, [2.0, -6.0, 0.0, 1.0, 1.0]),
        // Prompt tokens now also pay presence and frequency once.
        (
            PenaltyScope::PromptAndGenerated,
            [1.25, -6.75, 0.0, 1.0, 1.0],
        ),
    ] {
        let config = GenerationConfig {
            repetition_penalty: 2.0,
            presence_penalty: 0.5,
            frequency_penalty: 0.25,
            additive_penalty_scope: scope,
            ..Default::default()
        };
        let processor =
            LogitsProcessor::<TestBackend>::new(config.clone(), 5, &device).expect("processor");
        let history = history_after(&config, 5, &[0, 1], &[2, 2]);
        let processed = processor
            .process_with_history(
                Tensor::from_data(TensorData::new(logits.to_vec(), [1, 5]), &device),
                &history,
            )
            .expect("penalized logits");
        let actual = host_values(processed);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "{scope:?}: {actual} vs {expected}"
            );
        }
    }
}

/// Logit bias is added before the repetition penalty, matching vLLM and Hugging Face order.
#[test]
fn logit_bias_precedes_penalties_and_validates_tokens() {
    let device = Default::default();
    let config = GenerationConfig {
        repetition_penalty: 2.0,
        logit_bias: BTreeMap::from([(0, 2.0), (3, -1.0)]),
        ..Default::default()
    };
    let processor = LogitsProcessor::<TestBackend>::new(config.clone(), 4, &device).unwrap();
    for dtype in [DType::F16, DType::F32, DType::F64] {
        let processed = processor
            .process(
                Tensor::<TestBackend, 2>::from_floats([[1.0, 1.0, 0.5, 0.5]], &device).cast(dtype),
                Some(Tensor::from_ints([[0, 3]], &device)),
            )
            .expect("biased logits");
        assert_eq!(processed.dtype(), dtype);
        // Token 0: (1 + 2) / 2; token 3: (0.5 - 1) * 2.
        assert_eq!(host_values(processed), [1.5, 1.0, 0.5, -1.0]);
    }
    for bias in [BTreeMap::from([(4, 1.0)]), BTreeMap::from([(1, f64::NAN)])] {
        let invalid = GenerationConfig {
            logit_bias: bias,
            ..Default::default()
        };
        assert!(invalid.validate(4).is_err());
    }
}

/// EOS is impossible below `min_new_tokens` and allowed once the minimum is reached.
#[test]
fn min_new_tokens_suppress_eos_only_until_the_minimum() {
    let device = Default::default();
    let config = GenerationConfig {
        max_new_tokens: 4,
        min_new_tokens: 2,
        eos_token_ids: vec![3],
        ..Default::default()
    };
    let processor = LogitsProcessor::<TestBackend>::new(config.clone(), 4, &device).unwrap();
    let logits = || Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0, 2.0, 9.0]], &device);
    for (generated, eos_allowed) in [(&[][..], false), (&[1][..], false), (&[1, 2][..], true)] {
        let history = history_after(&config, 4, &[0], generated);
        let processed = host_values(processor.process_with_history(logits(), &history).unwrap());
        assert_eq!(processed[3].is_finite(), eos_allowed, "after {generated:?}");
        assert_eq!(processed[..3], [0.0, 1.0, 2.0]);
    }
    // The stateless path counts its whole history as generated.
    let stateless = host_values(processor.process(logits(), None).unwrap());
    assert_eq!(stateless[3], f64::NEG_INFINITY);
    let stateless = host_values(
        processor
            .process(logits(), Some(Tensor::from_ints([[1, 2]], &device)))
            .unwrap(),
    );
    assert_eq!(stateless[3], 9.0);
    let invalid = GenerationConfig {
        max_new_tokens: 1,
        min_new_tokens: 2,
        ..Default::default()
    };
    assert!(invalid.validate(4).is_err());
}

/// Stop sequences are ignored before `min_new_tokens`, and EOS still finishes a row.
#[test]
fn tracker_defers_stop_sequences_until_min_new_tokens() {
    let config = GenerationConfig {
        max_new_tokens: 5,
        min_new_tokens: 3,
        eos_token_ids: vec![7],
        stop_sequences: vec![vec![1]],
        ..Default::default()
    };
    let mut tracker = GenerationTracker::new(config, 8, 2).expect("tracker");
    assert_eq!(tracker.push(0, 1).unwrap(), None);
    assert_eq!(tracker.push(0, 1).unwrap(), None);
    assert_eq!(
        tracker.push(0, 1).unwrap(),
        Some(&FinishReason::StopSequence {
            index: 0,
            length: 1
        })
    );
    assert_eq!(
        tracker.push(1, 7).unwrap(),
        Some(&FinishReason::EndOfSequence { token_id: 7 })
    );
}

/// Banned n-gram continuations follow Hugging Face `NoRepeatNGramLogitsProcessor`.
#[test]
fn no_repeat_ngram_bans_only_completions_of_seen_ngrams() {
    let device = Default::default();
    for (size, expected_banned) in [(1, vec![1, 2, 3]), (2, vec![3]), (3, vec![3]), (6, vec![])] {
        let config = GenerationConfig {
            no_repeat_ngram_size: size,
            ..Default::default()
        };
        let processor = LogitsProcessor::<TestBackend>::new(config.clone(), 5, &device).unwrap();
        // Prompt [1, 2, 3, 1] then generated [2]: the history ends in "1 2".
        let history = history_after(&config, 5, &[1, 2, 3, 1], &[2]);
        let processed = host_values(
            processor
                .process_with_history(Tensor::zeros([1, 5], &device), &history)
                .unwrap(),
        );
        let banned = (0..5)
            .filter(|&token| processed[token] == f64::NEG_INFINITY)
            .collect::<Vec<_>>();
        assert_eq!(banned, expected_banned, "n = {size}");
    }
}

/// Min-p keeps tokens with at least `min_p` times the top probability, after temperature.
#[test]
fn min_p_keeps_tokens_relative_to_the_most_likely_one() {
    let device = Default::default();
    // With T = 0.5, p_i / p_max = exp(2 (z_i - z_max)) = 1, e^-1, e^-2, e^-4.
    let logits = [3.0_f32, 2.5, 2.0, 1.0];
    for (min_p, kept) in [(0.0, 4), (0.1, 3), (0.2, 2), (0.5, 1), (1.0, 1)] {
        let processor = LogitsProcessor::<TestBackend>::new(
            GenerationConfig {
                strategy: SamplingStrategy::Sample,
                temperature: 0.5,
                min_p: Some(min_p),
                ..Default::default()
            },
            4,
            &device,
        )
        .unwrap();
        let processed = host_values(
            processor
                .process(
                    Tensor::from_data(TensorData::new(logits.to_vec(), [1, 4]), &device),
                    None,
                )
                .unwrap(),
        );
        assert_eq!(
            processed.iter().filter(|value| value.is_finite()).count(),
            kept,
            "min_p = {min_p}"
        );
        assert!(processed[0].is_finite());
    }
    for min_p in [-0.1, 1.1, f64::NAN] {
        let invalid = GenerationConfig {
            min_p: Some(min_p),
            ..Default::default()
        };
        assert!(invalid.validate(4).is_err());
    }
}

/// A history must come from a configuration with the processor's history-dependent controls.
#[test]
fn processors_reject_incompatible_token_histories() {
    let device = Default::default();
    let plain = GenerationConfig::default();
    let penalized = GenerationConfig {
        frequency_penalty: 0.5,
        ..Default::default()
    };
    let ngram = GenerationConfig {
        no_repeat_ngram_size: 2,
        ..Default::default()
    };
    let logits = || Tensor::<TestBackend, 2>::zeros([1, 4], &device);
    let plain_history = super::TokenHistory::empty(&plain, 4, 1, &device).unwrap();
    for config in [penalized.clone(), ngram] {
        let processor = LogitsProcessor::<TestBackend>::new(config, 4, &device).unwrap();
        assert!(
            processor
                .process_with_history(logits(), &plain_history)
                .is_err()
        );
    }
    // A generated-only history cannot serve prompt-scoped additive penalties.
    let generated_scope = super::TokenHistory::empty(&penalized, 4, 1, &device).unwrap();
    let prompt_scoped = GenerationConfig {
        additive_penalty_scope: PenaltyScope::PromptAndGenerated,
        ..penalized.clone()
    };
    let processor = LogitsProcessor::<TestBackend>::new(prompt_scoped, 4, &device).unwrap();
    let error = processor
        .process_with_history(logits(), &generated_scope)
        .unwrap_err();
    assert!(
        error.to_string().contains("prompt-and-generated"),
        "{error}"
    );
    let processor = LogitsProcessor::<TestBackend>::new(plain.clone(), 4, &device).unwrap();
    let wide = super::TokenHistory::empty(&plain, 4, 2, &device).unwrap();
    assert!(processor.process_with_history(logits(), &wide).is_err());
    let other_vocabulary = super::TokenHistory::empty(&plain, 5, 1, &device).unwrap();
    assert!(
        processor
            .process_with_history(logits(), &other_vocabulary)
            .is_err()
    );
}

/// The prompt boundary decides which tokens count as generated; the legacy path counts all.
///
/// With a four-token prompt and `min_new_tokens = 3`, the prompt-aware path
/// still suppresses EOS before the first generated token, whereas the
/// stateless `process` path treats the prompt as output and does not. The
/// same split applies to presence and frequency under the default scope.
#[test]
fn prompt_lengths_separate_prompt_from_generated_tokens() {
    let device = Default::default();
    let config = GenerationConfig {
        max_new_tokens: 4,
        min_new_tokens: 3,
        eos_token_ids: vec![0],
        presence_penalty: 1.0,
        ..Default::default()
    };
    let processor = LogitsProcessor::<TestBackend>::new(config, 4, &device).unwrap();
    let logits = || Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    let history: &[u32] = &[1, 2, 3, 1];

    let aware = host_values(
        processor
            .process_with_prompt_lengths(logits(), &[history], &[4])
            .unwrap(),
    );
    assert_eq!(aware, [f64::NEG_INFINITY, 2.0, 3.0, 4.0]);
    // Generated tokens 3 and 1: EOS stays suppressed, and presence counts
    // only those two, not prompt token 2.
    let aware = host_values(
        processor
            .process_with_prompt_lengths(logits(), &[history], &[2])
            .unwrap(),
    );
    assert_eq!(aware, [f64::NEG_INFINITY, 1.0, 3.0, 3.0]);

    let legacy_history = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 1]], &device);
    let legacy = host_values(processor.process(logits(), Some(legacy_history)).unwrap());
    assert_eq!(legacy, [1.0, 1.0, 2.0, 3.0]);
}

/// The prompt-aware path promotes half-precision logits and validates its inputs.
#[test]
fn prompt_length_processing_promotes_half_logits_and_checks_inputs() {
    let device = Default::default();
    let processor = LogitsProcessor::<TestBackend>::new(
        GenerationConfig {
            repetition_penalty: 1.05,
            ..Default::default()
        },
        3,
        &device,
    )
    .unwrap();
    // 20.0 / 1.05 = 19.047..., which F16 would round to 19.046875.
    let half = Tensor::<TestBackend, 2>::from_floats([[20.0, 1.0, -2.0]], &device).cast(DType::F16);
    let processed = processor
        .process_with_prompt_lengths(half.clone(), &[&[0]], &[1])
        .unwrap();
    assert_eq!(processed.dtype(), DType::F32);
    let values = host_values(processed);
    assert!(
        (values[0] - f64::from(20.0_f32 / 1.05_f32)).abs() < 1e-6,
        "{values:?}"
    );
    assert_eq!(&values[1..], [1.0, -2.0]);
    let legacy = processor
        .process(
            half,
            Some(Tensor::<TestBackend, 2, Int>::from_ints([[0]], &device)),
        )
        .unwrap();
    assert_eq!(legacy.dtype(), DType::F16);

    let logits = || Tensor::<TestBackend, 2>::zeros([1, 3], &device);
    let wide = Tensor::<TestBackend, 2>::zeros([1, 4], &device);
    assert!(
        processor
            .process_with_prompt_lengths(wide, &[&[0]], &[1])
            .unwrap_err()
            .to_string()
            .contains("generation.logits")
    );
    assert!(
        processor
            .process_with_prompt_lengths(logits(), &[&[0], &[1]], &[1, 1])
            .unwrap_err()
            .to_string()
            .contains("generation.token_history")
    );
    assert!(
        processor
            .process_with_prompt_lengths(logits(), &[&[3]], &[0])
            .is_err()
    );
    assert!(
        processor
            .process_with_prompt_lengths(logits(), &[&[1]], &[2])
            .is_err()
    );
}

/// Generation floats round-trip through JSON bit for bit, however long their exponent.
///
/// `serde_json`'s `float_roundtrip` parser makes the shortest representation
/// parse back exactly; without it the first two values below read back an ulp
/// away (`3.7997969716361846e-262` became `3.799796971636185e-262`).
#[test]
fn configuration_floats_round_trip_exactly() {
    for value in [
        3.799_796_971_636_184_6e-262,
        2.291_712_365_432_881e-9,
        5e-324,
        f64::MAX,
        0.1,
        1e-46,
    ] {
        let config = GenerationConfig {
            temperature: value,
            repetition_penalty: value,
            presence_penalty: -value,
            frequency_penalty: value,
            top_p: Some(value),
            min_p: Some(value),
            logit_bias: BTreeMap::from([(3, value)]),
            ..Default::default()
        };
        let text = serde_json::to_string(&config).unwrap();
        let restored: GenerationConfig = serde_json::from_str(&text).unwrap();
        let bits = |config: &GenerationConfig| {
            [
                config.temperature,
                config.repetition_penalty,
                config.presence_penalty,
                config.frequency_penalty,
                config.top_p.unwrap(),
                config.min_p.unwrap(),
                config.logit_bias[&3],
            ]
            .map(f64::to_bits)
        };
        assert_eq!(bits(&config), bits(&restored), "{text}");
    }
}

/// Older serialized configurations load with neutral defaults for the newer controls.
#[test]
fn configuration_serde_defaults_and_round_trips_new_controls() {
    let legacy: GenerationConfig =
        serde_json::from_str(r#"{"max_new_tokens": 9, "presence_penalty": 0.5}"#).unwrap();
    assert_eq!(
        legacy,
        GenerationConfig {
            max_new_tokens: 9,
            presence_penalty: 0.5,
            ..Default::default()
        }
    );
    let full = GenerationConfig {
        min_new_tokens: 2,
        min_p: Some(0.05),
        frequency_penalty: -0.5,
        additive_penalty_scope: PenaltyScope::PromptAndGenerated,
        no_repeat_ngram_size: 3,
        logit_bias: BTreeMap::from([(1, -100.0), (7, 2.5)]),
        ..Default::default()
    };
    let json = serde_json::to_string(&full).unwrap();
    assert!(json.contains(r#""additive_penalty_scope":"prompt_and_generated""#));
    assert_eq!(
        serde_json::from_str::<GenerationConfig>(&json).unwrap(),
        full
    );
    assert!(
        GenerationConfig {
            frequency_penalty: f64::INFINITY,
            ..Default::default()
        }
        .validate(4)
        .is_err()
    );
}

// ---------------------------------------------------------------------------
// Property tests against straightforward host references
// ---------------------------------------------------------------------------

/// Naive F64 reference for bias, constraints-free penalties of one row.
///
/// `all` counts every observed token; `additive` counts the tokens in the
/// configured scope. Mirrors the documented equations directly.
fn reference_penalties(
    logits: &[f32],
    all: &[usize],
    additive: &[usize],
    config: &GenerationConfig,
) -> Vec<f64> {
    logits
        .iter()
        .enumerate()
        .map(|(token, &logit)| {
            let mut value = f64::from(logit)
                + config
                    .logit_bias
                    .get(&(token as u32))
                    .copied()
                    .unwrap_or(0.0);
            if all[token] > 0 && config.repetition_penalty != 1.0 {
                value = if value < 0.0 {
                    value * config.repetition_penalty
                } else {
                    value / config.repetition_penalty
                };
            }
            value -= config.frequency_penalty * additive[token] as f64;
            if additive[token] > 0 {
                value -= config.presence_penalty;
            }
            value
        })
        .collect()
}

/// Banned next tokens of one sequence, computed by scanning every n-gram.
fn reference_banned(tokens: &[u32], size: usize) -> Vec<u32> {
    let mut banned = Vec::new();
    if size == 0 || tokens.len() + 1 < size {
        return banned;
    }
    let prefix = &tokens[tokens.len() + 1 - size..];
    for start in 0..tokens.len().saturating_sub(size - 1) {
        if &tokens[start..start + size - 1] == prefix {
            banned.push(tokens[start + size - 1]);
        }
    }
    banned.sort_unstable();
    banned.dedup();
    banned
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(48))]

    /// Incremental device counts reproduce the naive penalty equations for every row.
    #[test]
    fn incremental_penalties_match_naive_reference(
        vocabulary in 2_usize..9,
        prompt in proptest::collection::vec((0_u32..9, proptest::bool::ANY), 0..6),
        generated in proptest::collection::vec(0_u32..9, 0..5),
        repetition in proptest::sample::select(vec![1.0, 1.3, 0.7]),
        presence in proptest::sample::select(vec![0.0, 0.4, -0.2]),
        frequency in proptest::sample::select(vec![0.0, 0.3]),
        prompt_scope in proptest::bool::ANY,
        bias in proptest::sample::select(vec![0.0, 1.5]),
    ) {
        let device = Default::default();
        let config = GenerationConfig {
            repetition_penalty: repetition,
            presence_penalty: presence,
            frequency_penalty: frequency,
            additive_penalty_scope: if prompt_scope {
                PenaltyScope::PromptAndGenerated
            } else {
                PenaltyScope::Generated
            },
            logit_bias: if bias == 0.0 { BTreeMap::new() } else { BTreeMap::from([(1, bias)]) },
            ..Default::default()
        };
        let processor = LogitsProcessor::<TestBackend>::new(config.clone(), vocabulary, &device).unwrap();
        // Masked prompt entries hold an out-of-vocabulary sentinel.
        let prompt_ids = prompt
            .iter()
            .map(|&(token, valid)| if valid { (token as usize % vocabulary) as i32 } else { 99 })
            .collect::<Vec<_>>();
        let prompt_valid = prompt.iter().map(|&(_, valid)| valid).collect::<Vec<_>>();
        let prompt_tensor = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(prompt_ids.clone(), [1, prompt_ids.len()]),
            &device,
        );
        let mask = crate::TokenMask::new(Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
            TensorData::new(prompt_valid.clone(), [1, prompt_valid.len()]),
            &device,
        ));
        let mut history = super::TokenHistory::from_prompt(&config, vocabulary, prompt_tensor, Some(mask)).unwrap();
        let generated = generated
            .iter()
            .map(|&token| token % vocabulary as u32)
            .collect::<Vec<_>>();
        for &token in &generated {
            history.record(&[token], &[true]).unwrap();
        }
        let mut prompt_counts = vec![0; vocabulary];
        for (&token, &valid) in prompt_ids.iter().zip(&prompt_valid) {
            if valid {
                prompt_counts[token as usize] += 1;
            }
        }
        let mut generated_counts = vec![0; vocabulary];
        for &token in &generated {
            generated_counts[token as usize] += 1;
        }
        let all = prompt_counts.iter().zip(&generated_counts).map(|(a, b)| a + b).collect::<Vec<_>>();
        let additive = if prompt_scope { all.clone() } else { generated_counts.clone() };
        let logits = pseudo_normal_logits(vocabulary, 2.0, vocabulary as u64 + generated.len() as u64);
        let expected = reference_penalties(&logits, &all, &additive, &config);
        let actual = host_values(
            processor
                .process_with_history(
                    Tensor::from_data(TensorData::new(logits.clone(), [1, vocabulary]), &device),
                    &history,
                )
                .unwrap(),
        );
        for (actual, expected) in actual.iter().zip(&expected) {
            proptest::prop_assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
        }

        // Host rows with the prompt boundary reproduce the incremental path bit for bit.
        let row = prompt_ids
            .iter()
            .zip(&prompt_valid)
            .filter(|&(_, &valid)| valid)
            .map(|(&token, _)| token as u32)
            .chain(generated.iter().copied())
            .collect::<Vec<_>>();
        let prompt_length = prompt_valid.iter().filter(|&&valid| valid).count();
        let from_rows = host_values(
            processor
                .process_with_prompt_lengths(
                    Tensor::from_data(TensorData::new(logits.clone(), [1, vocabulary]), &device),
                    &[&row],
                    &[prompt_length],
                )
                .unwrap(),
        );
        proptest::prop_assert_eq!(&from_rows, &actual);

        // The stateless history path counts every supplied token as generated.
        let mut combined_ids = prompt_ids.clone();
        let mut combined_valid = prompt_valid.clone();
        combined_ids.extend(generated.iter().map(|&token| token as i32));
        combined_valid.extend(generated.iter().map(|_| true));
        let stateless = host_values(
            processor
                .process_with_mask(
                    Tensor::from_data(TensorData::new(logits.clone(), [1, vocabulary]), &device),
                    Some(Tensor::from_data(TensorData::new(combined_ids.clone(), [1, combined_ids.len()]), &device)),
                    Some(crate::TokenMask::new(Tensor::from_data(
                        TensorData::new(combined_valid.clone(), [1, combined_valid.len()]),
                        &device,
                    ))),
                )
                .unwrap(),
        );
        let expected = reference_penalties(&logits, &all, &all, &config);
        for (actual, expected) in stateless.iter().zip(&expected) {
            proptest::prop_assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
        }
    }

    /// Inverse-CDF sampling returns a positive-probability token and matches a host reference.
    #[test]
    fn sampled_tokens_have_positive_probability_and_match_reference(
        logits in proptest::collection::vec(-8.0_f32..8.0, 2..40),
        masked in proptest::collection::vec(proptest::bool::ANY, 40),
        bucket in 0_u64..(1 << 23),
    ) {
        let device = Default::default();
        let mut logits = logits;
        let vocabulary = logits.len();
        for (logit, &is_masked) in logits.iter_mut().zip(&masked).skip(1) {
            if is_masked {
                *logit = f32::NEG_INFINITY;
            }
        }
        let uniform = super::random::interior_unit_f32(bucket << 41);
        let selected = super::sample_with_uniforms(
            Tensor::<TestBackend, 2>::from_data(TensorData::new(logits.clone(), [1, vocabulary]), &device),
            vec![uniform],
        )
        .into_data()
        .to_vec::<i32>()
        .unwrap()[0] as usize;
        proptest::prop_assert!(selected < vocabulary);
        proptest::prop_assert!(logits[selected].is_finite());
        // Host reference in F64; skip draws within rounding of a bin edge.
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = logits.iter().map(|&z| f64::from(z - maximum).exp()).collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        let threshold = f64::from(uniform) * total;
        let mut cumulative = 0.0;
        let mut near_edge = false;
        let mut expected = vocabulary - 1;
        for (index, weight) in weights.iter().enumerate() {
            cumulative += weight;
            near_edge |= (cumulative - threshold).abs() < 1e-5 * total;
            if threshold <= cumulative {
                expected = index;
                break;
            }
        }
        if !near_edge {
            proptest::prop_assert_eq!(selected, expected);
        }
    }

    /// The n-gram index bans exactly the continuations a full scan finds.
    #[test]
    fn ngram_bans_match_full_scan(
        prompt in proptest::collection::vec(0_u32..4, 0..8),
        generated in proptest::collection::vec(0_u32..4, 0..8),
        size in 1_usize..5,
    ) {
        let config = GenerationConfig { no_repeat_ngram_size: size, ..Default::default() };
        let prompt_ids = prompt.iter().map(|&token| token as i32).collect::<Vec<_>>();
        let history = history_after(&config, 4, &prompt_ids, &generated);
        let mut tokens = prompt.clone();
        tokens.extend(&generated);
        proptest::prop_assert_eq!(history.banned_tokens(0).to_vec(), reference_banned(&tokens, size));
    }

    /// Min-p in log space keeps exactly the tokens with `p_i >= min_p * p_max`.
    #[test]
    fn min_p_matches_probability_space_reference(
        logits in proptest::collection::vec(-6.0_f32..6.0, 2..24),
        min_p in 0.001_f64..1.0,
        temperature in 0.3_f64..2.0,
    ) {
        let device = Default::default();
        let vocabulary = logits.len();
        let processor = LogitsProcessor::<TestBackend>::new(
            GenerationConfig {
                strategy: SamplingStrategy::Sample,
                temperature,
                min_p: Some(min_p),
                ..Default::default()
            },
            vocabulary,
            &device,
        )
        .unwrap();
        let processed = host_values(
            processor
                .process(Tensor::from_data(TensorData::new(logits.clone(), [1, vocabulary]), &device), None)
                .unwrap(),
        );
        let scaled = logits.iter().map(|&z| f64::from(z) / temperature).collect::<Vec<_>>();
        let maximum = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for (index, &z) in scaled.iter().enumerate() {
            let ratio = (z - maximum).exp();
            if (ratio - min_p).abs() > 1e-5 * min_p.max(1e-3) {
                proptest::prop_assert_eq!(processed[index].is_finite(), ratio >= min_p, "token {}", index);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Timing benchmark (run explicitly in release mode)
// ---------------------------------------------------------------------------

/// The previous penalty implementation, kept as a timing and parity reference.
///
/// Every call validated the history with two blocking scalar reads, built a
/// `[batch, history, vocabulary]` one-hot, and summed it over the history.
fn one_hot_reference_penalties(
    logits: Tensor<TestBackend, 2>,
    history: Tensor<TestBackend, 2, Int>,
    vocabulary: usize,
    repetition: f64,
    presence: f64,
) -> Tensor<TestBackend, 2> {
    let minimum = history.clone().min().into_scalar();
    let maximum = history.clone().max().into_scalar();
    assert!(minimum >= 0 && (maximum as usize) < vocabulary);
    let observed = history
        .one_hot::<3>(vocabulary)
        .sum_dim(1)
        .squeeze_dim::<2>(1)
        .greater_elem(0);
    let penalized = logits.clone().div_scalar(repetition).mask_where(
        logits.clone().lower_elem(0.0),
        logits.clone().mul_scalar(repetition),
    );
    let logits = logits.mask_where(observed.clone(), penalized);
    logits - observed.float().mul_scalar(presence)
}

/// Times one-hot penalties against incremental history over a growing decode history.
///
/// Run with `cargo test --release --lib penalty_history_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing benchmark; run in release mode with --ignored --nocapture"]
fn penalty_history_timing() {
    use std::time::Instant;

    let device = Default::default();
    let vocabulary = 32_000;
    let prompt_length = 1_024;
    let steps = 16;
    let config = GenerationConfig {
        repetition_penalty: 1.1,
        presence_penalty: 0.5,
        additive_penalty_scope: PenaltyScope::PromptAndGenerated,
        ..Default::default()
    };
    let processor =
        LogitsProcessor::<TestBackend>::new(config.clone(), vocabulary, &device).unwrap();
    let prompt = (0..prompt_length)
        .map(|index| ((index * 7_919) % vocabulary) as i64)
        .collect::<Vec<_>>();
    let generated = (0..steps)
        .map(|index| ((index * 104_729) % vocabulary) as u32)
        .collect::<Vec<_>>();
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(pseudo_normal_logits(vocabulary, 2.0, 3), [1, vocabulary]),
        &device,
    );

    let started = Instant::now();
    let mut history_ids = prompt.clone();
    let mut reference = None;
    for &token in &generated {
        let history = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(history_ids.clone(), [1, history_ids.len()]),
            &device,
        );
        let processed = one_hot_reference_penalties(logits.clone(), history, vocabulary, 1.1, 0.5);
        reference = Some(host_values(processed));
        history_ids.push(i64::from(token));
    }
    let one_hot = started.elapsed();

    let started = Instant::now();
    let mut history = super::TokenHistory::from_prompt(
        &config,
        vocabulary,
        Tensor::from_data(TensorData::new(prompt, [1, prompt_length]), &device),
        None,
    )
    .unwrap();
    let mut incremental = None;
    for &token in &generated {
        let processed = processor
            .process_with_history(logits.clone(), &history)
            .unwrap();
        incremental = Some(host_values(processed));
        history.record(&[token], &[true]).unwrap();
    }
    let counts = started.elapsed();

    assert_eq!(reference, incremental);
    println!(
        "penalties over {steps} steps, prompt {prompt_length}, vocabulary {vocabulary}: one-hot {:.2} ms/step, incremental {:.2} ms/step ({:.1}x)",
        one_hot.as_secs_f64() * 1e3 / steps as f64,
        counts.as_secs_f64() * 1e3 / steps as f64,
        one_hot.as_secs_f64() / counts.as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// Finite arithmetic: validation in the processing precision and saturation
// ---------------------------------------------------------------------------

/// The message of a generation error, or a panic for any other variant.
fn generation_message(error: crate::Error) -> String {
    match error {
        crate::Error::Generation(message) => message,
        other => panic!("expected a generation error, got {other:?}"),
    }
}

/// Finite `f64` controls that round to zero, a subnormal, or infinity in F32 are rejected.
///
/// Regressions for fuzz findings GEN-F32-1..4: each of these validated, and
/// processing then produced NaN (`0 * inf`, `0 / 0`, `inf + -inf`). Every
/// error names the field, F32, and its representable range; values that are
/// not even finite and positive keep the established messages.
#[test]
fn float_controls_are_validated_in_f32() {
    let largest = f64::from(f32::MAX);
    let smallest = f64::from(f32::MIN_POSITIVE);
    let cases = [
        (
            GenerationConfig {
                frequency_penalty: 1e39,
                ..Default::default()
            },
            "frequency_penalty 1e39 overflows F32; it must round to a finite F32 value in [-3.4028235e38, 3.4028235e38]",
        ),
        (
            GenerationConfig {
                presence_penalty: -1e39,
                ..Default::default()
            },
            "presence_penalty -1e39 overflows F32",
        ),
        (
            GenerationConfig {
                repetition_penalty: 1e-46,
                ..Default::default()
            },
            "repetition_penalty 1e-46 rounds to 0.0 in F32; it must round to a normal positive F32 value in [1.1754944e-38, 3.4028235e38]",
        ),
        (
            GenerationConfig {
                temperature: 1e-46,
                ..Default::default()
            },
            "temperature 1e-46 rounds to 0.0 in F32",
        ),
        (
            GenerationConfig {
                temperature: smallest / 2.0,
                ..Default::default()
            },
            "temperature",
        ),
        (
            GenerationConfig {
                repetition_penalty: 1e39,
                ..Default::default()
            },
            "repetition_penalty 1e39 rounds to inf in F32",
        ),
        (
            GenerationConfig {
                logit_bias: BTreeMap::from([(1, 1e39)]),
                ..Default::default()
            },
            "logit_bias for token 1 1e39 overflows F32",
        ),
        (
            GenerationConfig {
                temperature: 0.0,
                ..Default::default()
            },
            "temperature must be finite and greater than zero",
        ),
        (
            GenerationConfig {
                frequency_penalty: f64::NAN,
                ..Default::default()
            },
            "frequency_penalty must be finite, found NaN",
        ),
        (
            GenerationConfig {
                logit_bias: BTreeMap::from([(2, f64::INFINITY)]),
                ..Default::default()
            },
            "logit_bias for token 2 must be finite, found inf",
        ),
    ];
    for (config, expected) in cases {
        let message = generation_message(config.validate(4).unwrap_err());
        assert!(message.starts_with(expected), "{message}");
    }
    // Both ends of the range are accepted, including an `f64` just above
    // `f32::MAX` that rounds down to it.
    let rounds_to_largest = largest * (1.0 + f64::EPSILON);
    for value in [smallest, largest, rounds_to_largest] {
        GenerationConfig {
            temperature: value,
            repetition_penalty: value,
            presence_penalty: -value,
            frequency_penalty: value,
            logit_bias: BTreeMap::from([(0, value), (1, -value)]),
            ..Default::default()
        }
        .validate(4)
        .expect("boundary controls are representable in F32");
    }
}

/// Checks that processed logits are finite or `-inf`, and that selection stays defined.
fn assert_finite_or_excluded(processor: &LogitsProcessor<TestBackend>, processed: Vec<f64>) {
    assert!(
        processed
            .iter()
            .all(|value| value.is_finite() || *value == f64::NEG_INFINITY),
        "{processed:?}"
    );
    let vocabulary = processor.vocabulary_size();
    let rows = processed.len() / vocabulary;
    let tensor = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(
            processed
                .iter()
                .map(|&value| value as f32)
                .collect::<Vec<_>>(),
            [rows, vocabulary],
        ),
        &Default::default(),
    );
    let selection = processor.select_with_log_probs(tensor).unwrap();
    let tokens = selection.token_ids.into_data().to_vec::<i32>().unwrap();
    let log_probabilities = host_values(selection.log_probabilities);
    for (row, (&token, &log_probability)) in tokens.iter().zip(&log_probabilities).enumerate() {
        let values = &processed[row * vocabulary..(row + 1) * vocabulary];
        if values.iter().any(|value| value.is_finite()) {
            assert!(values[token as usize].is_finite(), "row {row}: {values:?}");
            assert!(log_probability.is_finite() && log_probability <= 0.0);
        }
    }
}

/// Boundary controls on extreme logits never yield NaN or `+inf`, in either strategy.
///
/// Before saturation, `repetition_penalty = f32::MIN_POSITIVE` divided a logit
/// of four into `+inf`, `frequency_penalty = -f32::MAX` times a count of two
/// overflowed to `-inf` and met a `-inf` logit (`-inf - -inf = NaN`), and a
/// `+inf` sum met a constraint's `-inf`.
#[test]
fn boundary_controls_keep_processed_logits_finite_or_excluded() {
    let device = Default::default();
    let largest = f64::from(f32::MAX);
    let smallest = f64::from(f32::MIN_POSITIVE);
    // Token 4 is excluded by the model; tokens 0 and 1 are huge.
    let logits = [3.0e38_f32, -3.0e38, 4.0, -4.0, f32::NEG_INFINITY, 0.0];
    let rows: [&[u32]; 2] = [&[0, 1, 2, 3, 4, 4, 2, 2], &[5, 5, 5]];
    for strategy in [SamplingStrategy::Greedy, SamplingStrategy::Sample] {
        for (repetition, frequency, presence, bias, temperature) in [
            (smallest, -largest, largest, largest, smallest),
            (largest, largest, -largest, -largest, largest),
            (smallest, largest, 0.0, largest, 0.5),
            (1.0, -largest, 0.0, 0.0, smallest),
        ] {
            let config = GenerationConfig {
                strategy,
                temperature,
                repetition_penalty: repetition,
                frequency_penalty: frequency,
                presence_penalty: presence,
                additive_penalty_scope: PenaltyScope::PromptAndGenerated,
                logit_bias: BTreeMap::from([(0, bias), (2, bias), (5, -bias)]),
                no_repeat_ngram_size: 1,
                min_new_tokens: 2,
                eos_token_ids: vec![3],
                top_k: Some(5),
                top_p: Some(0.9),
                min_p: Some(0.1),
                ..Default::default()
            };
            let processor =
                LogitsProcessor::<TestBackend>::new(config.clone(), 6, &device).expect("valid");
            let batch = Tensor::<TestBackend, 2>::from_data(
                TensorData::new([logits, logits].concat(), [2, 6]),
                &device,
            );
            let processed = host_values(
                processor
                    .process_with_prompt_lengths(batch, &rows, &[7, 3])
                    .expect("processed"),
            );
            // The unigram ban excludes every seen token; EOS (token 3) is
            // suppressed below two generated tokens in both rows.
            for (row, seen) in [(0, [0, 1, 2, 3, 4].as_slice()), (1, [5].as_slice())] {
                for &token in seen.iter().chain([&3]) {
                    assert_eq!(processed[row * 6 + token], f64::NEG_INFINITY, "{config:?}");
                }
            }
            assert_finite_or_excluded(&processor, processed);
        }
    }
}

/// Saturation and the frequency clamp are exact no-ops when nothing overflows.
///
/// Ordinary configurations keep their bits: the expected values are the
/// documented formulas evaluated in F32.
#[test]
fn saturation_preserves_values_that_do_not_overflow() {
    let device = Default::default();
    let config = GenerationConfig {
        strategy: SamplingStrategy::Sample,
        temperature: 0.5,
        repetition_penalty: 1.5,
        frequency_penalty: 0.25,
        presence_penalty: 0.5,
        logit_bias: BTreeMap::from([(1, 0.75)]),
        ..Default::default()
    };
    let processor = LogitsProcessor::<TestBackend>::new(config, 4, &device).unwrap();
    let logits = Tensor::<TestBackend, 2>::from_floats([[2.0, -1.0, 0.5, 3.0]], &device);
    let processed = processor
        .process_with_prompt_lengths(logits, &[&[0, 1, 1]], &[0])
        .unwrap();
    let (r, f, p, t) = (1.5_f32, 0.25_f32, 0.5_f32, 0.5_f32);
    let expected = [
        (2.0 / r - f - p) / t,
        ((-1.0 + 0.75) * r - 2.0 * f - p) / t,
        0.5 / t,
        3.0 / t,
    ];
    assert_eq!(
        host_values(processed),
        expected.map(f64::from),
        "saturation must not change finite results"
    );
}

/// The frequency product saturates instead of meeting an infinity of the same sign.
#[test]
fn frequency_products_saturate_before_subtraction() {
    let device = Default::default();
    let config = GenerationConfig {
        frequency_penalty: -f64::from(f32::MAX),
        additive_penalty_scope: PenaltyScope::PromptAndGenerated,
        ..Default::default()
    };
    let processor = LogitsProcessor::<TestBackend>::new(config, 3, &device).unwrap();
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![f32::NEG_INFINITY, 1.0, 2.0], [1, 3]),
        &device,
    );
    // Token 0 (logit -inf) and token 1 were seen twice: c * f = -2 * MAX.
    let processed = host_values(
        processor
            .process_with_prompt_lengths(logits, &[&[0, 0, 1, 1]], &[4])
            .unwrap(),
    );
    assert_eq!(
        processed,
        [f64::NEG_INFINITY, f64::from(f32::MAX), 2.0],
        "-inf stays excluded and the overflow saturates"
    );
}

/// F16 occurrence counts above 65504 saturate before they are cast to the logits dtype.
///
/// Review finding GEN-F16-COUNT-INF: a count of 65600 became `inf` in F16,
/// and a frequency penalty that rounds to zero there (`1e-8`) turned it into
/// `inf * 0 = NaN`, which the product clamp then mapped to the *largest*
/// logit, so the most repeated token became the most likely. Counts now
/// saturate at the dtype's largest value first, so the product is finite or
/// saturates, and a zero penalty leaves logits unchanged.
#[test]
fn half_precision_counts_saturate_before_the_cast() {
    let device = Default::default();
    let history = Tensor::<TestBackend, 2, Int>::zeros([1, 65_600], &device);
    for (penalty, expected) in [
        (1e-8, [1.0, 2.0, 3.0]),
        // 65504 * 0.5 = 32752 is exact in F16; 1 - 32752 rounds to -32752.
        (0.5, [-32_752.0, 2.0, 3.0]),
        // 65504 * 2 overflows F16 and saturates at 65504.
        (2.0, [-65_504.0, 2.0, 3.0]),
        (-2.0, [65_504.0, 2.0, 3.0]),
    ] {
        let config = GenerationConfig {
            frequency_penalty: penalty,
            ..Default::default()
        };
        let processor = LogitsProcessor::<TestBackend>::new(config, 3, &device).unwrap();
        let logits =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0]], &device).cast(DType::F16);
        let processed = processor.process(logits, Some(history.clone())).unwrap();
        assert_eq!(processed.dtype(), DType::F16);
        assert_eq!(
            host_values(processed.cast(DType::F32)),
            expected,
            "penalty {penalty}"
        );
    }
}

/// Half-precision logits processed in their own dtype check controls against that dtype.
///
/// F16 cannot represent a frequency penalty of `1e5` (its largest finite
/// value is 65504) or a temperature of `1e-5` (subnormal in F16); BF16 rounds
/// `3.4e38` up to infinity. The error names the field and dtype and points at
/// [`super::upcast_half_precision`], after which the same controls work.
#[test]
fn half_precision_logits_check_controls_in_their_dtype() {
    let device = Default::default();
    for (config, dtype, field) in [
        (
            GenerationConfig {
                frequency_penalty: 1e5,
                ..Default::default()
            },
            DType::F16,
            "frequency_penalty 100000.0 overflows F16",
        ),
        (
            GenerationConfig {
                strategy: SamplingStrategy::Sample,
                temperature: 1e-5,
                ..Default::default()
            },
            DType::F16,
            "temperature 1e-5 rounds to",
        ),
        (
            GenerationConfig {
                logit_bias: BTreeMap::from([(0, 3.4e38)]),
                ..Default::default()
            },
            DType::BF16,
            "logit_bias for token 0 3.4e38 overflows BF16",
        ),
    ] {
        let processor = LogitsProcessor::<TestBackend>::new(config, 3, &device).unwrap();
        let logits = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let error = processor
            .process(
                logits.clone().cast(dtype),
                Some(Tensor::from_ints([[0]], &device)),
            )
            .unwrap_err();
        let message = generation_message(error);
        assert!(message.starts_with(field), "{message}");
        assert!(message.contains("upcast_half_precision"), "{message}");
        let promoted = processor
            .process(
                super::upcast_half_precision(logits.cast(dtype)),
                Some(Tensor::from_ints([[0]], &device)),
            )
            .expect("F32 represents the control");
        assert_eq!(promoted.dtype(), DType::F32);
        // The coordinators' entry point promotes automatically.
        let rows: [&[u32]; 1] = [&[0]];
        let half = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0]], &device).cast(dtype);
        assert!(
            processor
                .process_with_prompt_lengths(half, &rows, &[0])
                .is_ok()
        );
    }
}

/// Every scalar-precision range matches the format's own constants.
#[test]
fn scalar_precisions_round_like_their_formats() {
    use super::ScalarPrecision;
    use burn::tensor::{bf16, f16};
    assert_eq!(ScalarPrecision::of(DType::F16), ScalarPrecision::F16);
    assert_eq!(ScalarPrecision::of(DType::BF16), ScalarPrecision::Bf16);
    assert_eq!(ScalarPrecision::of(DType::F32), ScalarPrecision::F32);
    assert_eq!(ScalarPrecision::of(DType::Flex32), ScalarPrecision::F32);
    assert_eq!(ScalarPrecision::of(DType::F64), ScalarPrecision::F64);
    assert_eq!(ScalarPrecision::F16.largest(), 65504.0);
    assert_eq!(
        ScalarPrecision::F16.smallest_normal(),
        f16::MIN_POSITIVE.to_f64()
    );
    assert_eq!(ScalarPrecision::Bf16.largest(), bf16::MAX.to_f64());
    assert_eq!(ScalarPrecision::F64.round(1e-300), 1e-300);
    assert_eq!(ScalarPrecision::F32.round(1e-46), 0.0);
    assert_eq!(ScalarPrecision::F16.round(1e5), f64::INFINITY);
    assert!(ScalarPrecision::F16.narrower_than_f32());
    assert!(ScalarPrecision::Bf16.narrower_than_f32());
    assert!(!ScalarPrecision::F32.narrower_than_f32());
    assert!(!ScalarPrecision::F64.narrower_than_f32());
    for precision in [
        ScalarPrecision::F16,
        ScalarPrecision::Bf16,
        ScalarPrecision::F32,
        ScalarPrecision::F64,
    ] {
        precision
            .check_divisor("x", precision.smallest_normal())
            .unwrap();
        precision.check_divisor("x", precision.largest()).unwrap();
        precision.check_finite("x", -precision.largest()).unwrap();
        assert!(
            precision
                .check_divisor("x", precision.smallest_normal() / 2.0)
                .is_err()
        );
        assert!(precision.check_divisor("x", -1.0).is_err());
        assert!(precision.check_finite("x", f64::NAN).is_err());
    }
    // F64 has no wider rounding: only non-finite values fail.
    ScalarPrecision::F64.check_finite("x", 1e300).unwrap();
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(96))]

    /// Any valid configuration on finite-or-`-inf` logits yields finite-or-`-inf` output.
    ///
    /// Controls are drawn log-uniformly across the whole F32 range (so most
    /// draws overflow some intermediate), logits across `+-f32::MAX`, and
    /// histories repeat tokens; the processed logits must never be NaN or
    /// `+inf`, and selection must pick a finite logit with a finite log
    /// probability whenever the row keeps one.
    #[test]
    fn valid_configurations_never_produce_nan(
        sample in proptest::bool::ANY,
        exponents in proptest::collection::vec(-37.0f64..38.0, 5),
        signs in proptest::collection::vec(proptest::bool::ANY, 3),
        logit_exponents in proptest::collection::vec(-3.0f64..38.5, 5),
        logit_signs in proptest::collection::vec(proptest::bool::ANY, 5),
        excluded in 0usize..6,
        history in proptest::collection::vec(0u32..5, 0..8),
        top_p in proptest::option::of(0.05f64..1.0),
        min_p in proptest::option::of(0.0f64..1.0),
    ) {
        let device = Default::default();
        let signed = |exponent: f64, negative: bool| {
            let magnitude = 10f64.powf(exponent).min(f64::from(f32::MAX));
            if negative { -magnitude } else { magnitude }
        };
        let config = GenerationConfig {
            strategy: if sample { SamplingStrategy::Sample } else { SamplingStrategy::Greedy },
            temperature: 10f64.powf(exponents[0]),
            repetition_penalty: 10f64.powf(exponents[1]),
            frequency_penalty: signed(exponents[2], signs[0]),
            presence_penalty: signed(exponents[3], signs[1]),
            logit_bias: BTreeMap::from([(1, signed(exponents[4], signs[2]))]),
            additive_penalty_scope: PenaltyScope::PromptAndGenerated,
            top_p,
            min_p,
            ..Default::default()
        };
        proptest::prop_assume!(config.validate(5).is_ok());
        let processor = LogitsProcessor::<TestBackend>::new(config, 5, &device).unwrap();
        let mut logits: Vec<f32> = logit_exponents
            .iter()
            .zip(&logit_signs)
            .map(|(&exponent, &negative)| signed(exponent, negative) as f32)
            .collect();
        if excluded < 5 {
            logits[excluded] = f32::NEG_INFINITY;
        }
        let processed = host_values(
            processor
                .process_with_prompt_lengths(
                    Tensor::from_data(TensorData::new(logits, [1, 5]), &device),
                    &[&history],
                    &[0],
                )
                .unwrap(),
        );
        assert_finite_or_excluded(&processor, processed);
    }
}
