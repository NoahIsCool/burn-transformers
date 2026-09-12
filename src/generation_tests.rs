//! Generation-policy tests for deterministic sampling, penalties, stopping, and invalid requests.

use burn::backend::Flex;
use burn::tensor::{DType, Int, Tensor, TensorData, Tolerance};

use super::{
    FinishReason, GenerationConfig, GenerationTracker, LogitsProcessor, SamplingStrategy,
    splitmix64_uniform,
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
            super::causal::host_float_data_f64(penalized.clone().into_data(), "penalties")
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
        let values = super::causal::host_float_data_f64(
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

/// Unsupported Flex32 filters fail before reaching Burn's panicking sort fallback.
#[test]
fn flex32_sort_filters_fail_closed_before_backend_operations() {
    let device = Default::default();
    for strategy in [SamplingStrategy::Greedy, SamplingStrategy::Sample] {
        for (top_k, top_p) in [
            (Some(1), None),
            (Some(4), None),
            (None, Some(0.7)),
            (Some(2), Some(1.0)),
        ] {
            let processor = LogitsProcessor::<TestBackend>::new(
                GenerationConfig {
                    strategy,
                    top_k,
                    top_p,
                    ..Default::default()
                },
                4,
                &device,
            )
            .expect("valid sampling configuration");
            let mut data = TensorData::from([[1.0_f32, 2.0, 3.0, 4.0]]);
            data.dtype = DType::Flex32;
            let logits = Tensor::from_data(data, (&device, DType::Flex32));
            let error = processor
                .process(logits, None)
                .expect_err("Flex32 sort must fail before unsupported CPU tensor math");
            assert!(matches!(error, crate::Error::BackendCapability(_)));
            assert!(error.to_string().contains("Flex32 top-k/top-p"));
        }
    }
    for top_p in [None, Some(1.0)] {
        let processor = LogitsProcessor::<TestBackend>::new(
            GenerationConfig {
                top_p,
                ..Default::default()
            },
            4,
            &device,
        )
        .expect("unfiltered greedy configuration");
        let mut data = TensorData::from([[1.0_f32, 2.0, 3.0, 4.0]]);
        data.dtype = DType::Flex32;
        let logits = Tensor::from_data(data, (&device, DType::Flex32));
        assert_eq!(
            processor
                .process(logits, None)
                .expect("no sort requested")
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

/// Top-k retains exactly k finite entries, including tied boundary values.
#[test]
fn top_k_is_exact_even_at_ties() {
    let device = Default::default();
    let config = GenerationConfig {
        strategy: SamplingStrategy::Sample,
        top_k: Some(2),
        ..Default::default()
    };
    let processor =
        LogitsProcessor::<TestBackend>::new(config, 4, &device).expect("valid processor");
    let output = processor
        .process(Tensor::from_floats([[3.0, 2.0, 2.0, 1.0]], &device), None)
        .expect("valid scores")
        .into_data()
        .to_vec::<f32>()
        .expect("float output");
    assert_eq!(output.iter().filter(|value| value.is_finite()).count(), 2);
    assert_eq!(output[0], 3.0);
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
        strategy: SamplingStrategy::Sample,
        temperature: 0.75,
        top_k: Some(2),
        top_p: Some(0.9),
        repetition_penalty: 1.1,
        presence_penalty: 0.2,
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
