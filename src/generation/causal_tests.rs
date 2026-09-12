//! Causal-generation coordinator tests with real caches and injected public model failures.

use burn::{
    backend::Flex,
    tensor::{
        Bool, Int, Tensor,
        backend::{Backend, BackendTypes},
    },
};

use super::{CausalGenerationRequest, GenerationToken, generate_causal, generate_causal_streaming};
use crate::{
    CausalLmInput, DecodeState, Error, FinishReason, GenerationConfig, SamplingStrategy, TokenMask,
    cache::CacheConfig,
    models::qwen3_5::{Qwen3_5ForCausalLm, Qwen3_5LayerType, Qwen3_5TextConfig},
};

/// Deterministic F32 CPU backend used by causal generation fixtures.
type TestBackend = Flex<f32, i32>;

/// Replaces one real model output with an exact nonfinite score fixture.
///
/// A constant NaN or infinite row makes normalized log probabilities
/// undefined even if argmax returns an in-range token. Retaining the real
/// Qwen cache makes this a coordinator boundary test, not a tracker-only test.
#[derive(burn::module::Module, Debug)]
struct NonfiniteModel<B: Backend> {
    /// Real Qwen graph and cache implementation exercised by the coordinator.
    inner: Qwen3_5ForCausalLm<B>,
    /// Select decode rather than prefill as the corrupted output boundary.
    #[module(skip)]
    fail_on_decode: bool,
    /// NaN or infinite score inserted into the selected model output.
    #[module(skip)]
    corrupt_value: f32,
    /// Number of prefill calls observed before generation rejects the result.
    #[module(skip)]
    prefill_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Number of decode calls observed before generation rejects the result.
    #[module(skip)]
    decode_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl<B: Backend> crate::CausalLanguageModel<B> for NonfiniteModel<B> {
    /// Configuration of the wrapped Qwen graph.
    type Config = Qwen3_5TextConfig;
    /// Unmodified external decode state of the wrapped Qwen model.
    type State = <Qwen3_5ForCausalLm<B> as crate::CausalLanguageModel<B>>::State;

    /// Leaves stateless evaluation untouched; only cached outputs inject faults.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &crate::ForwardOptions,
    ) -> crate::Result<crate::CausalLmOutput<B>> {
        self.inner.forward(input, options)
    }

    /// Allocates the real architecture's external cache without changing policy.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> crate::Result<Self::State> {
        self.inner.new_decode_state(config, batch_size, device)
    }

    /// Counts prompt calls and optionally corrupts the first selectable distribution.
    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &crate::ForwardOptions,
    ) -> crate::Result<crate::CausalLmOutput<B>> {
        self.prefill_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut output = self.inner.prefill(input, state, options)?;
        if !self.fail_on_decode {
            output.logits = Tensor::full(
                output.logits.dims(),
                self.corrupt_value,
                &output.logits.device(),
            );
        }
        Ok(output)
    }

    /// Counts feedback calls and injects the fault only after one valid token.
    fn decode(
        &self,
        input: crate::DecodeInput<B>,
        state: &mut Self::State,
        options: &crate::ForwardOptions,
    ) -> crate::Result<crate::CausalLmOutput<B>> {
        self.decode_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut output = self.inner.decode(input, state, options)?;
        output.logits = Tensor::full(
            output.logits.dims(),
            self.corrupt_value,
            &output.logits.device(),
        );
        Ok(output)
    }
}

/// Active nonfinite probabilities abort before emission or any later model call.
#[test]
fn active_nonfinite_probabilities_cancel_before_the_invalid_token() {
    use std::sync::{Arc, atomic::AtomicUsize};

    for corrupt_value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for fail_on_decode in [false, true] {
            let (inner, device) = tiny_model();
            let model = NonfiniteModel {
                inner,
                fail_on_decode,
                corrupt_value,
                prefill_calls: Arc::new(AtomicUsize::new(0)),
                decode_calls: Arc::new(AtomicUsize::new(0)),
            };
            let mut events = Vec::new();
            let error = generate_causal_streaming(
                &model,
                request(
                    &device,
                    GenerationConfig {
                        max_new_tokens: 3,
                        ..Default::default()
                    },
                ),
                |event| {
                    events.push(event);
                    Ok(())
                },
            )
            .expect_err("an undefined active probability cannot be emitted");
            assert!(matches!(error, Error::Generation(_)));
            assert!(
                error
                    .to_string()
                    .contains("sequence 0 produced non-finite selected log probability"),
                "{error}"
            );
            let valid_tokens = usize::from(fail_on_decode);
            assert_eq!(events.len(), valid_tokens);
            for event in events {
                assert_eq!((event.sequence, event.step), (0, 0));
                assert!(event.log_probability.is_finite());
                assert_eq!(event.finish_reason, None);
            }
            assert_eq!(
                model
                    .prefill_calls
                    .load(std::sync::atomic::Ordering::Relaxed),
                1
            );
            assert_eq!(
                model
                    .decode_calls
                    .load(std::sync::atomic::Ordering::Relaxed),
                valid_tokens,
                "the coordinator must not decode after rejecting a probability"
            );
        }
    }
}

/// Runtime half-precision selected values cannot be decoded as the backend's default F32.
#[test]
fn selected_log_probabilities_respect_runtime_float_dtype() {
    use burn::tensor::DType;

    let device = Default::default();
    for strategy in [SamplingStrategy::Greedy, SamplingStrategy::Sample] {
        let processor = crate::LogitsProcessor::<TestBackend>::new(
            GenerationConfig {
                strategy,
                ..GenerationConfig::default()
            },
            2,
            &device,
        )
        .expect("token processor");
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            let logits = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0], [0.0, 0.0]], &device)
                .cast(dtype);
            let selected = processor
                .select_with_log_probs(logits)
                .expect("selected probabilities");
            assert_eq!(selected.log_probabilities.dtype(), dtype);
            assert_eq!(selected.log_probabilities.dims(), [2]);
            let values = super::tensor_f64_values(
                selected.log_probabilities,
                "selected probability fixture",
            )
            .expect("runtime dtype-aware host conversion");
            assert_eq!(values.len(), 2);
            assert!(
                values
                    .iter()
                    .all(|value| (*value + core::f64::consts::LN_2).abs() < 0.01)
            );
        }
    }
}

/// F64 values keep sub-F32 precision, and malformed or non-floating storage is recoverable.
#[test]
fn host_probability_conversion_preserves_f64_and_rejects_invalid_storage() {
    use burn::tensor::{DType, TensorData};

    let exact = -1.0_f64 - 2.0_f64.powi(-40);
    let data = TensorData::new(vec![exact], [1]);
    assert_eq!(
        super::host_float_data_f64(data, "F64 fixture").expect("native F64"),
        [exact]
    );
    let integer = TensorData::new(vec![1_i32], [1]);
    let error = super::host_float_data_f64(integer, "integer fixture")
        .expect_err("probabilities must use float storage");
    assert!(matches!(error, Error::BackendCapability(_)));
    assert!(error.to_string().contains("integer fixture"));
    let mut malformed = TensorData::new(vec![0_u8; 3], [3]);
    malformed.dtype = DType::F32;
    let error = super::host_float_data_f64(malformed, "truncated fixture")
        .expect_err("bad element size must not panic");
    assert!(error.to_string().contains("truncated fixture"));
}

/// Flex32 uses exact F32 host bytes even though its dtype tag is distinct.
#[test]
fn flex32_host_probability_storage_is_lossless_and_checked() {
    use burn::tensor::{DType, TensorData};

    let values = [-1.0_f32 - 2.0_f32.powi(-20), -0.0, f32::MIN_POSITIVE];
    let mut data = TensorData::new(values.to_vec(), [values.len()]);
    // This is CubeCL's host-data contract; Flex CPU cannot execute Flex32
    // casts, so ordinary CPU CI covers storage independently of GPU kernels.
    data.dtype = DType::Flex32;
    let actual = super::host_float_data_f64(data, "Flex32 fixture")
        .expect("Flex32 is supported floating storage");
    assert_eq!(actual, values.map(f64::from));
    assert!(actual[1].is_sign_negative());
    let mut malformed = TensorData::new(vec![0_u8; 3], [3]);
    malformed.dtype = DType::Flex32;
    let error = super::host_float_data_f64(malformed, "truncated Flex32 fixture")
        .expect_err("invalid Flex32 element size is recoverable");
    assert!(error.to_string().contains("truncated Flex32 fixture"));
}

/// The native Flex32 Metal backend supports selection, penalties, and host readback.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "requires serialized Metal execution"]
fn native_flex32_generation_readback() {
    use burn::{
        backend::{Metal, wgpu::flex32},
        tensor::DType,
    };

    /// Metal backend using the native flexible-precision float representation.
    type NativeFlex32 = Metal<flex32>;
    let device = Default::default();
    for strategy in [SamplingStrategy::Greedy, SamplingStrategy::Sample] {
        let processor = crate::LogitsProcessor::<NativeFlex32>::new(
            GenerationConfig {
                strategy,
                repetition_penalty: 2.0,
                presence_penalty: 0.5,
                temperature: 0.5,
                ..GenerationConfig::default()
            },
            4,
            &device,
        )
        .expect("native Flex32 processor");
        // Keep sorting out of this readback regression: Burn 0.21's generic
        // Flex32 sort fallback currently reinterprets its tag as F32.
        let logits = Tensor::<NativeFlex32, 2>::from_floats([[4.0, -3.0, 32.0, 1.0]], &device)
            // CubeCL aliases a direct F32->Flex32 cast without changing
            // its tag; F16->Flex32 allocates genuinely Flex32-typed storage.
            .cast(DType::F16)
            .cast(DType::Flex32);
        assert_eq!(logits.dtype(), DType::Flex32);
        assert_eq!(logits.clone().cast(DType::F32).dtype(), DType::Flex32);
        let history = Tensor::<NativeFlex32, 2, Int>::from_ints([[0, 1]], &device);
        let selected = processor
            .process_and_select_with_log_probs(logits, Some(history))
            .expect("native Flex32 probability processing");
        assert_eq!(selected.log_probabilities.dtype(), DType::Flex32);
        assert_eq!(
            super::tensor_i64_values(selected.token_ids, "Flex32 selection").expect("selected ID"),
            [2]
        );
        let values = super::tensor_f64_values(selected.log_probabilities, "Flex32 log probability")
            .expect("native Flex32 host readback");
        assert_eq!(values.len(), 1);
        assert!(values[0].is_finite() && values[0] <= 0.0);
    }
}

/// Builds a deterministic tiny model with a real external KV cache.
fn tiny_model() -> (
    Qwen3_5ForCausalLm<TestBackend>,
    <TestBackend as BackendTypes>::Device,
) {
    let device = Default::default();
    TestBackend::seed(&device, 991);
    let config = Qwen3_5TextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 64,
        layer_types: vec![Qwen3_5LayerType::FullAttention],
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_conv_kernel_dim: 3,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        mtp_num_hidden_layers: 0,
        ..Default::default()
    };
    let model =
        Qwen3_5ForCausalLm::init(config, &device).expect("tiny generation model must initialize");
    (model, device)
}

/// Constructs a single-row prompt and a cache large enough for generation.
fn request(
    device: &<TestBackend as BackendTypes>::Device,
    generation: GenerationConfig,
) -> CausalGenerationRequest<TestBackend> {
    CausalGenerationRequest {
        prompt: CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], device),
            token_mask: None,
            position_ids: None,
        },
        cache: CacheConfig::new(1, 8).expect("test cache config"),
        generation,
        prefill_chunk_size: None,
        vocabulary_size: 32,
    }
}

/// Greedy generation emits ordered events and aligned finite probabilities.
#[test]
fn greedy_generation_streams_complete_length_limited_output() {
    let (model, device) = tiny_model();
    let mut events = Vec::<GenerationToken>::new();
    let output = generate_causal_streaming(
        &model,
        request(
            &device,
            GenerationConfig {
                max_new_tokens: 3,
                ..Default::default()
            },
        ),
        |event| {
            events.push(event);
            Ok(())
        },
    )
    .expect("cached generation should succeed");
    let sequence = &output.sequences[0];
    assert_eq!(sequence.prompt_length, 2);
    assert_eq!(sequence.token_ids.len(), 3);
    assert_eq!(sequence.log_probabilities.len(), 3);
    assert!(
        sequence
            .log_probabilities
            .iter()
            .all(|value| value.is_finite())
    );
    assert_eq!(sequence.finish_reason, FinishReason::Length);
    assert_eq!(events.len(), 3);
    assert_eq!(
        events.iter().map(|event| event.step).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        events.last().and_then(|event| event.finish_reason.clone()),
        Some(FinishReason::Length)
    );
}

/// EOS and multi-token suffixes terminate at the first matching step.
#[test]
fn eos_and_stop_sequences_terminate_generation() {
    let (model, device) = tiny_model();
    let baseline = generate_causal(
        &model,
        request(
            &device,
            GenerationConfig {
                max_new_tokens: 3,
                ..Default::default()
            },
        ),
    )
    .expect("baseline generation");
    let tokens = baseline.sequences[0].token_ids.clone();

    let eos = generate_causal(
        &model,
        request(
            &device,
            GenerationConfig {
                max_new_tokens: 3,
                eos_token_ids: vec![tokens[0]],
                ..Default::default()
            },
        ),
    )
    .expect("EOS generation");
    assert_eq!(eos.sequences[0].token_ids, vec![tokens[0]]);
    assert_eq!(
        eos.sequences[0].finish_reason,
        FinishReason::EndOfSequence {
            token_id: tokens[0]
        }
    );

    let stopped = generate_causal(
        &model,
        request(
            &device,
            GenerationConfig {
                max_new_tokens: 3,
                stop_sequences: vec![tokens[..2].to_vec()],
                ..Default::default()
            },
        ),
    )
    .expect("stop-sequence generation");
    assert_eq!(stopped.sequences[0].token_ids, tokens[..2]);
    assert_eq!(
        stopped.sequences[0].finish_reason,
        FinishReason::StopSequence {
            index: 0,
            length: 2
        }
    );
}

/// Independent seeded processors reproduce a complete stochastic sequence.
#[test]
fn stochastic_generation_is_reproducible() {
    let (model, device) = tiny_model();
    let generation = GenerationConfig {
        max_new_tokens: 4,
        strategy: SamplingStrategy::Sample,
        temperature: 0.8,
        top_k: Some(8),
        top_p: Some(0.9),
        seed: 17,
        ..Default::default()
    };
    let first = generate_causal(&model, request(&device, generation.clone()))
        .expect("first seeded generation");
    let second =
        generate_causal(&model, request(&device, generation)).expect("second seeded generation");
    assert_eq!(first, second);
}

/// Ragged prompt rows preserve their semantic lengths throughout decoding.
#[test]
fn ragged_batch_uses_true_valid_masks() {
    let (model, device) = tiny_model();
    let output = generate_causal(
        &model,
        CausalGenerationRequest {
            prompt: CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                    [[1, 31, 2], [3, 31, 31]],
                    &device,
                ),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, false, true], [true, false, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            cache: CacheConfig::new(2, 4).expect("test cache"),
            generation: GenerationConfig {
                max_new_tokens: 2,
                repetition_penalty: 1.1,
                presence_penalty: 0.2,
                ..Default::default()
            },
            prefill_chunk_size: None,
            vocabulary_size: 32,
        },
    )
    .expect("ragged generation");
    assert_eq!(output.sequences.len(), 2);
    assert_eq!(output.sequences[0].prompt_length, 2);
    assert_eq!(output.sequences[1].prompt_length, 1);
    assert!(
        output
            .sequences
            .iter()
            .all(|sequence| sequence.token_ids.len() == 2)
    );
}

/// Chunked prefill retains each ragged row's last active logits across padding-only chunks.
#[test]
fn chunked_prefill_matches_one_shot_for_ragged_batches() {
    let (model, device) = tiny_model();
    let request = CausalGenerationRequest {
        prompt: CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                [[1, 2, 3, 31, 31, 31], [31, 31, 4, 5, 31, 6]],
                &device,
            ),
            token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                [
                    [true, true, true, false, false, false],
                    [false, false, true, true, false, true],
                ]
                .into(),
                &device,
            ))),
            position_ids: None,
        },
        cache: CacheConfig::new(2, 8).expect("test cache"),
        generation: GenerationConfig {
            max_new_tokens: 3,
            ..Default::default()
        },
        prefill_chunk_size: None,
        vocabulary_size: 32,
    };
    let one_shot = generate_causal(&model, request.clone()).expect("one-shot prefill");
    let mut chunked_request = request;
    chunked_request.prefill_chunk_size = Some(2);
    let chunked = generate_causal(&model, chunked_request).expect("chunked prefill");

    assert_eq!(chunked.sequences.len(), one_shot.sequences.len());
    for (chunked, one_shot) in chunked.sequences.iter().zip(&one_shot.sequences) {
        assert_eq!(chunked.prompt_length, one_shot.prompt_length);
        assert_eq!(chunked.token_ids, one_shot.token_ids);
        assert_eq!(chunked.finish_reason, one_shot.finish_reason);
        assert_eq!(
            chunked.log_probabilities.len(),
            one_shot.log_probabilities.len()
        );
        for (&actual, &expected) in chunked
            .log_probabilities
            .iter()
            .zip(&one_shot.log_probabilities)
        {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "chunked log probability {actual} differs from one-shot {expected}"
            );
        }
    }
}

/// A zero prefill bound is rejected before any model operation.
#[test]
fn zero_prefill_chunk_size_is_invalid() {
    let (model, device) = tiny_model();
    let mut invalid = request(
        &device,
        GenerationConfig {
            max_new_tokens: 1,
            ..Default::default()
        },
    );
    invalid.prefill_chunk_size = Some(0);

    let error = generate_causal(&model, invalid).expect_err("zero cannot advance chunks");
    assert!(error.to_string().contains("prefill_chunk_size"));
}

/// Explicit prompt positions advance once per committed continuation token.
#[test]
fn explicit_positions_continue_from_last_valid_token() {
    let (model, device) = tiny_model();
    let mut generation_request = request(
        &device,
        GenerationConfig {
            max_new_tokens: 2,
            ..Default::default()
        },
    );
    generation_request.prompt.position_ids = Some(Tensor::from_ints([[7, 9]], &device));
    let output = generate_causal(&model, generation_request)
        .expect("generation with explicit positions should succeed");
    assert_eq!(output.sequences[0].token_ids.len(), 2);
}

/// Coordinator errors are raised before entering model inference when possible.
#[test]
fn request_validation_rejects_bad_masks_and_cache_geometry() {
    let (model, device) = tiny_model();
    let generation = GenerationConfig {
        max_new_tokens: 3,
        ..Default::default()
    };
    let mut wrong_batch = request(&device, generation.clone());
    wrong_batch.cache = CacheConfig::new(2, 8).expect("valid but mismatched cache");
    assert!(generate_causal(&model, wrong_batch).is_err());

    let mut too_small = request(&device, generation.clone());
    too_small.cache = CacheConfig::new(1, 3).expect("valid but undersized cache");
    assert!(generate_causal(&model, too_small).is_err());

    let mut wrong_mask = request(&device, generation.clone());
    wrong_mask.prompt.token_mask =
        Some(TokenMask::new(Tensor::from_bool([[true]].into(), &device)));
    assert!(generate_causal(&model, wrong_mask).is_err());

    let mut empty_row = request(&device, generation);
    empty_row.prompt.token_mask = Some(TokenMask::new(Tensor::from_bool(
        [[false, false]].into(),
        &device,
    )));
    assert!(generate_causal(&model, empty_row).is_err());
}

/// A callback can cancel generation without panicking or emitting later tokens.
#[test]
fn streaming_callback_errors_cancel_generation() {
    let (model, device) = tiny_model();
    let mut calls = 0;
    let result = generate_causal_streaming(
        &model,
        request(
            &device,
            GenerationConfig {
                max_new_tokens: 3,
                ..Default::default()
            },
        ),
        |_| {
            calls += 1;
            Err(Error::Generation("consumer disconnected".into()))
        },
    );
    assert!(
        matches!(result, Err(Error::Generation(message)) if message == "consumer disconnected")
    );
    assert_eq!(calls, 1);
}

/// Malformed positions and impossible logical capacity fail before any token event.
#[test]
fn request_position_and_capacity_boundaries_are_contextual() {
    let (model, device) = tiny_model();
    let generation = GenerationConfig {
        max_new_tokens: 3,
        ..Default::default()
    };
    for (positions, expected) in [
        (vec![7], "generation.prompt.position_ids"),
        (vec![7, -1], "negative last valid position"),
        (vec![7, i32::MAX], "portable signed index range"),
    ] {
        let length = positions.len();
        let mut invalid = request(&device, generation.clone());
        invalid.prompt.position_ids = Some(Tensor::from_data(
            burn::tensor::TensorData::new(positions, [1, length]),
            &device,
        ));
        let error = generate_causal_streaming(&model, invalid, |_| {
            panic!("invalid request must not emit a token")
        })
        .expect_err("invalid explicit positions");
        assert!(error.to_string().contains(expected), "{error}");
    }
    let mut overflow = request(&device, generation);
    overflow.generation.max_new_tokens = usize::MAX;
    let error = generate_causal(&model, overflow).expect_err("capacity sum overflows");
    assert!(
        error
            .to_string()
            .contains("required cache capacity overflows")
    );
}

/// Last-logit selection enforces every axis, including a wrong vocabulary declaration.
#[test]
fn model_output_shape_errors_are_contextual_in_every_prefill_mode() {
    let (model, device) = tiny_model();
    for chunk_size in [None, Some(1), Some(2), Some(usize::MAX)] {
        let mut invalid = request(
            &device,
            GenerationConfig {
                max_new_tokens: 1,
                ..Default::default()
            },
        );
        invalid.vocabulary_size = 33;
        invalid.prefill_chunk_size = chunk_size;
        let error = generate_causal(&model, invalid).expect_err("model has 32 logits");
        assert!(error.to_string().contains("generation.model_logits"));
        assert!(error.to_string().contains("33"));
    }
    for shape in [[2, 1, 32], [1, 2, 32], [1, 1, 31]] {
        let error =
            super::extract_last_logits(Tensor::<TestBackend, 3>::zeros(shape, &device), 1, 32)
                .expect_err("one wrong output axis");
        assert!(error.to_string().contains("generation.model_logits"));
    }
}

/// Chunk-size fast paths preserve exact greedy results and full prompt position IDs.
#[test]
fn dense_chunking_and_unbounded_fast_paths_preserve_positions() {
    let (model, device) = tiny_model();
    let mut input = request(
        &device,
        GenerationConfig {
            max_new_tokens: 3,
            ..Default::default()
        },
    );
    input.prompt.position_ids = Some(Tensor::from_ints([[4, 7]], &device));
    let baseline = generate_causal(&model, input.clone()).expect("one-shot positioned prompt");
    for chunk in [1, 2, usize::MAX] {
        let mut chunked = input.clone();
        chunked.prefill_chunk_size = Some(chunk);
        let output = generate_causal(&model, chunked).expect("bounded positioned prompt");
        assert_eq!(
            output.sequences[0].token_ids,
            baseline.sequences[0].token_ids
        );
        for (actual, expected) in output.sequences[0]
            .log_probabilities
            .iter()
            .zip(&baseline.sequences[0].log_probabilities)
        {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }
}

/// Helper preflight rejects rectangular metadata errors before invoking a model.
#[test]
fn chunked_prefill_rejects_invalid_compact_metadata() {
    use crate::{CausalLanguageModel, DecodeState, ForwardOptions};

    let (model, device) = tiny_model();
    let request = request(
        &device,
        GenerationConfig {
            max_new_tokens: 1,
            ..Default::default()
        },
    );
    let mut state = model
        .new_decode_state(request.cache, 1, &device)
        .expect("state");
    for (mask, size, expected) in [
        (vec![true, true], 0, "prefill_chunk_size"),
        (vec![true], 1, "prompt mask contains 1 entries, expected 2"),
    ] {
        let error = super::prefill_prompt(
            &model,
            request.prompt.clone(),
            &mut state,
            &ForwardOptions::default(),
            &mask,
            Some(size),
            32,
        )
        .expect_err("bad prefill metadata");
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(state.lengths(), [0]);
    }
}

/// Position planning validates overflow, signed bounds, ragged inputs, and padding.
#[test]
fn compact_position_metadata_has_checked_boundaries() {
    let error = super::prompt_lengths(&[], usize::MAX, 2).expect_err("product overflow");
    assert!(error.to_string().contains("element count overflows"));
    let error = super::prompt_lengths(&[true], 1, 2).expect_err("ragged mask");
    assert!(error.to_string().contains("expected 2"));
    assert_eq!(
        super::prompt_lengths(&[true, false, true, false], 2, 2).expect("valid rows"),
        [1, 1]
    );
    for (positions, valid, batch, sequence, tokens, message) in [
        (vec![], vec![], usize::MAX, 2, 1, "element count overflows"),
        (vec![0], vec![true, true], 1, 2, 1, "not rectangular"),
        (
            vec![0],
            vec![true],
            1,
            1,
            usize::MAX,
            "signed position domain",
        ),
        (vec![0], vec![false], 1, 1, 1, "no valid position"),
        (
            vec![-1],
            vec![true],
            1,
            1,
            1,
            "negative last valid position",
        ),
        (
            vec![i64::MAX],
            vec![true],
            1,
            1,
            2,
            "explicit positions overflow",
        ),
        (
            vec![i64::from(i32::MAX)],
            vec![true],
            1,
            1,
            2,
            "portable signed index range",
        ),
    ] {
        let error = super::next_explicit_positions(&positions, &valid, batch, sequence, tokens)
            .expect_err("invalid compact position metadata");
        assert!(error.to_string().contains(message), "{error}");
    }
    assert_eq!(
        super::next_explicit_positions(
            &[i64::MAX, 7, -1, i64::from(i32::MAX)],
            &[false, true, false, true],
            2,
            2,
            1,
        )
        .expect("masked sentinels are ignored and last sampled token is not decoded"),
        [8, i64::from(i32::MAX) + 1]
    );
    let mut positions = [5, i64::MAX];
    super::advance_positions(&mut positions, &[true, false]).expect("inactive row is untouched");
    assert_eq!(positions, [6, i64::MAX]);
    let error = super::advance_positions(&mut positions, &[true]).expect_err("ragged activity");
    assert!(error.to_string().contains("position batch"));
    assert_eq!(positions, [6, i64::MAX]);
    let error = super::advance_positions(&mut [i64::MAX], &[true]).expect_err("signed overflow");
    assert!(
        error
            .to_string()
            .contains("explicit position overflow for row 0")
    );
}

/// Inactive nonfinite logits are zeroed without modifying still-active rows.
#[test]
fn neutralization_is_row_local_and_checks_batch_metadata() {
    let device = Default::default();
    let logits =
        Tensor::<TestBackend, 2>::from_floats([[f32::NAN, f32::INFINITY], [1.0, -2.0]], &device);
    let values = super::neutralize_inactive_logits(logits.clone(), &[false, true])
        .expect("inactive nonfinite row")
        .into_data()
        .to_vec::<f32>()
        .expect("F32 values");
    assert_eq!(values, [0.0, 0.0, 1.0, -2.0]);
    let error = super::neutralize_inactive_logits(logits, &[false])
        .expect_err("activity must cover the batch");
    assert!(error.to_string().contains("active-row count 1"));
}

/// Public output assembly rejects incomplete tracking or misaligned log probabilities.
#[test]
fn output_assembly_never_fabricates_terminal_metadata() {
    let config = GenerationConfig {
        max_new_tokens: 1,
        ..Default::default()
    };
    let tracker = super::GenerationTracker::new(config, 32, 1).expect("tracker");
    for (lengths, probabilities, expected) in [
        (vec![], vec![vec![]], "inconsistent batch dimensions"),
        (vec![2], vec![], "inconsistent batch dimensions"),
        (
            vec![2],
            vec![vec![-1.0]],
            "0 tokens but 1 log probabilities",
        ),
        (vec![2], vec![vec![]], "sequence 0 did not finish"),
    ] {
        let error = super::finish_output(tracker.clone(), lengths, probabilities)
            .expect_err("inconsistent output must fail");
        assert!(error.to_string().contains(expected), "{error}");
    }
    let mut completed = tracker;
    completed.push(0, 7).expect("terminal token");
    let output = super::finish_output(completed, vec![2], vec![vec![-0.5]])
        .expect("aligned completed metadata");
    assert_eq!(output.sequences[0].token_ids, [7]);
    assert_eq!(output.sequences[0].log_probabilities, [-0.5]);
    assert_eq!(output.sequences[0].finish_reason, FinishReason::Length);
}

/// Feedback observed by a real batched cache after the first row has finished.
#[derive(Clone, Debug, PartialEq, Eq)]
struct FinishedRowFeedback {
    /// Padded token IDs, in row order.
    ids: Vec<i64>,
    /// True-valid activity bits supplied to decode.
    valid: Vec<bool>,
    /// Explicit positions; an inactive sequence must not advance.
    positions: Vec<i64>,
    /// Real cache lengths after the successful feedback call.
    lengths: Vec<usize>,
}

/// Deterministic output fixture retaining Qwen's actual cache and forward graph.
///
/// Row zero emits EOS immediately. Row one emits two ordinary tokens and then
/// EOS. Subsequent row-zero logits are deliberately NaN: only the inactive row
/// may be neutralized, and it must never receive a second token or cache write.
#[derive(burn::module::Module, Debug)]
struct FinishedRowModel<B: Backend> {
    /// Real Qwen model retaining production cache behavior.
    inner: Qwen3_5ForCausalLm<B>,
    /// Ordered model boundaries reached by the generation coordinator.
    #[module(skip)]
    calls: std::sync::Arc<std::sync::Mutex<Vec<&'static str>>>,
    /// Tokens, masks, and cache lengths fed back after a row has finished.
    #[module(skip)]
    feedback: std::sync::Arc<std::sync::Mutex<Vec<FinishedRowFeedback>>>,
    /// Optional model boundary that returns an injected recoverable failure.
    #[module(skip)]
    fail_at: Option<&'static str>,
}

impl FinishedRowModel<TestBackend> {
    /// Records entry before injecting a contextual, recoverable model error.
    fn enter(&self, stage: &'static str) -> crate::Result<()> {
        self.calls.lock().unwrap().push(stage);
        if self.fail_at == Some(stage) {
            return Err(Error::Cache(format!("synthetic {stage} failure")));
        }
        Ok(())
    }

    /// Creates exact, well-separated active logits and toxic inactive logits.
    fn scores(
        step: usize,
        device: &<TestBackend as BackendTypes>::Device,
    ) -> Tensor<TestBackend, 3> {
        let mut values = vec![-20.0_f32; 64];
        if step == 0 {
            values[2] = 20.0;
        } else {
            values[..32].fill(f32::NAN);
        }
        values[32 + [4, 5, 2][step]] = 20.0;
        Tensor::from_data(burn::tensor::TensorData::new(values, [2, 1, 32]), device)
    }
}

impl crate::CausalLanguageModel<TestBackend> for FinishedRowModel<TestBackend> {
    /// Configuration inherited from the wrapped Qwen fixture.
    type Config = Qwen3_5TextConfig;
    /// Real cache state used to verify finished-row no-op updates.
    type State =
        <Qwen3_5ForCausalLm<TestBackend> as crate::CausalLanguageModel<TestBackend>>::State;

    /// Keeps stateless evaluation identical to the underlying Qwen model.
    fn forward(
        &self,
        input: CausalLmInput<TestBackend>,
        options: &crate::ForwardOptions,
    ) -> crate::Result<crate::CausalLmOutput<TestBackend>> {
        self.inner.forward(input, options)
    }

    /// Allocates the real external state, unless this boundary is fault-injected.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &<TestBackend as BackendTypes>::Device,
    ) -> crate::Result<Self::State> {
        self.enter("cache")?;
        self.inner.new_decode_state(config, batch_size, device)
    }

    /// Runs real prompt inference before replacing only the selected logits.
    fn prefill(
        &self,
        input: CausalLmInput<TestBackend>,
        state: &mut Self::State,
        options: &crate::ForwardOptions,
    ) -> crate::Result<crate::CausalLmOutput<TestBackend>> {
        self.enter("prefill")?;
        let mut output = self.inner.prefill(input, state, options)?;
        output.logits = Self::scores(0, &output.logits.device());
        Ok(output)
    }

    /// Observes active-only feedback, then tests nonfinite finished-row handling.
    fn decode(
        &self,
        input: crate::DecodeInput<TestBackend>,
        state: &mut Self::State,
        options: &crate::ForwardOptions,
    ) -> crate::Result<crate::CausalLmOutput<TestBackend>> {
        use crate::DecodeState;

        self.enter("decode")?;
        let ids = super::tensor_i64_values(input.token_ids.clone(), "feedback IDs").unwrap();
        let valid = input
            .token_mask
            .as_ref()
            .unwrap()
            .as_tensor()
            .clone()
            .into_data()
            .to_vec::<bool>()
            .unwrap();
        let positions =
            super::tensor_i64_values(input.position_ids.clone().unwrap(), "feedback positions")
                .unwrap();
        let mut output = self.inner.decode(input, state, options)?;
        let mut feedback = self.feedback.lock().unwrap();
        feedback.push(FinishedRowFeedback {
            ids,
            valid,
            positions,
            lengths: state.lengths().to_vec(),
        });
        output.logits = Self::scores(feedback.len(), &output.logits.device());
        Ok(output)
    }
}

/// Constructs a two-row real-cache request with distinct explicit position offsets.
fn finished_row_fixture(
    strategy: SamplingStrategy,
    fail_at: Option<&'static str>,
) -> (
    FinishedRowModel<TestBackend>,
    CausalGenerationRequest<TestBackend>,
) {
    let (inner, device) = tiny_model();
    let model = FinishedRowModel {
        inner,
        calls: Default::default(),
        feedback: Default::default(),
        fail_at,
    };
    let request = CausalGenerationRequest {
        prompt: CausalLmInput {
            token_ids: Tensor::from_ints([[1, 3], [7, 8]], &device),
            token_mask: None,
            position_ids: Some(Tensor::from_ints([[5, 6], [10, 11]], &device)),
        },
        cache: CacheConfig::new(2, 8).unwrap(),
        generation: GenerationConfig {
            max_new_tokens: 5,
            eos_token_ids: vec![2],
            pad_token_id: Some(0),
            strategy,
            top_k: Some(1),
            presence_penalty: 0.5,
            repetition_penalty: 1.2,
            ..Default::default()
        },
        prefill_chunk_size: None,
        vocabulary_size: 32,
    };
    (model, request)
}

/// Early-finished rows remain inactive while their batch peers continue generating.
#[test]
fn mixed_eos_batches_do_not_emit_or_cache_finished_row_tokens() {
    for strategy in [SamplingStrategy::Greedy, SamplingStrategy::Sample] {
        let (model, request) = finished_row_fixture(strategy, None);
        let mut events = Vec::new();
        let output = generate_causal_streaming(&model, request, |event| {
            events.push(event);
            Ok(())
        })
        .expect("inactive NaN logits cannot poison the remaining row");
        assert_eq!(output.sequences[0].token_ids, [2]);
        assert_eq!(output.sequences[1].token_ids, [4, 5, 2]);
        for row in &output.sequences {
            assert_eq!(
                row.finish_reason,
                FinishReason::EndOfSequence { token_id: 2 }
            );
            assert_eq!(row.prompt_length, 2);
            assert_eq!(row.log_probabilities.len(), row.token_ids.len());
            assert!(row.log_probabilities.iter().all(|value| value.is_finite()));
        }
        assert_eq!(
            events
                .iter()
                .map(|e| (e.sequence, e.step, e.token_id))
                .collect::<Vec<_>>(),
            [(0, 0, 2), (1, 0, 4), (1, 1, 5), (1, 2, 2)]
        );
        assert_eq!(
            *model.calls.lock().unwrap(),
            ["cache", "prefill", "decode", "decode"]
        );
        assert_eq!(
            *model.feedback.lock().unwrap(),
            [
                FinishedRowFeedback {
                    ids: vec![0, 4],
                    valid: vec![false, true],
                    positions: vec![7, 12],
                    lengths: vec![2, 3]
                },
                FinishedRowFeedback {
                    ids: vec![0, 5],
                    valid: vec![false, true],
                    positions: vec![7, 13],
                    lengths: vec![2, 4]
                },
            ]
        );
    }
}

/// Errors at each model boundary propagate unchanged and stop future emissions.
#[test]
fn model_boundary_failures_cancel_generation_at_the_exact_call() {
    for stage in ["cache", "prefill", "decode"] {
        let (model, request) = finished_row_fixture(SamplingStrategy::Greedy, Some(stage));
        let mut events = Vec::new();
        let error = generate_causal_streaming(&model, request, |event| {
            events.push(event);
            Ok(())
        })
        .expect_err("model failure must stop generation");
        assert!(
            matches!(&error, Error::Cache(message) if message == &format!("synthetic {stage} failure"))
        );
        let expected_calls: &[&str] = match stage {
            "cache" => &["cache"],
            "prefill" => &["cache", "prefill"],
            _ => &["cache", "prefill", "decode"],
        };
        assert_eq!(*model.calls.lock().unwrap(), expected_calls);
        assert_eq!(events.len(), if stage == "decode" { 2 } else { 0 });
        assert!(model.feedback.lock().unwrap().is_empty());
    }
}

/// Cancelling the first callback stops the same batch before emitting a peer row.
#[test]
fn callback_failure_mid_batch_suppresses_later_rows_and_feedback() {
    let (model, request) = finished_row_fixture(SamplingStrategy::Greedy, None);
    let mut events = Vec::new();
    let error = generate_causal_streaming(&model, request, |event| {
        events.push(event);
        Err(Error::Generation("subscriber disconnected".into()))
    })
    .expect_err("callback cancellation must not process another row");
    assert!(matches!(error, Error::Generation(message) if message == "subscriber disconnected"));
    assert_eq!(events.len(), 1);
    assert_eq!((events[0].sequence, events[0].token_id), (0, 2));
    assert_eq!(*model.calls.lock().unwrap(), ["cache", "prefill"]);
    assert!(model.feedback.lock().unwrap().is_empty());
}

/// A public model implementation may return an error or malformed output tensor.
#[derive(Clone, Copy, Debug)]
enum CoordinatorFault {
    /// A model error after a real cache update must still discard the session.
    ModelError,
    /// A wrong output axis must fail before selecting another token.
    LogitShape(
        /// Invalid `[batch, sequence, vocabulary]` dimensions returned to the caller.
        [usize; 3],
    ),
}

/// Records real cached calls and the committed prefix just before a model fault.
#[derive(Debug, PartialEq, Eq)]
struct CoordinatorCall {
    /// The public entry point selected for this prompt chunk or generated token.
    stage: &'static str,
    /// Number of physical input columns processed by this call.
    width: usize,
    /// Real per-row logical lengths after applying the input's validity mask.
    lengths: Vec<usize>,
}

/// Real Qwen state wrapped only to observe the coordinator-owned session lifetime.
#[derive(Debug)]
struct CoordinatorState<B: Backend> {
    /// The architecture's real external KV cache.
    inner: <Qwen3_5ForCausalLm<B> as crate::CausalLanguageModel<B>>::State,
    /// Session destruction is observable even when generation returns an error.
    drops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl<B: Backend> DecodeState<B> for CoordinatorState<B> {
    /// Delegates batch geometry without fabricating tracker state.
    fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }
    /// Delegates the real cache's logical capacity.
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    /// Exposes lengths committed by the real architecture.
    fn lengths(&self) -> &[usize] {
        self.inner.lengths()
    }
    /// Delegates reset without altering the lifetime probe.
    fn reset(&mut self) -> crate::Result<()> {
        self.inner.reset()
    }
    /// Delegates real tensor and metadata reordering.
    fn reorder(&mut self, indices: &[usize]) -> crate::Result<()> {
        self.inner.reorder(indices)
    }
    /// Delegates the architecture's supported rewind behavior.
    fn truncate(&mut self, lengths: &[usize]) -> crate::Result<()> {
        self.inner.truncate(lengths)
    }
    /// Returns the cache's actual allocation device.
    fn device(&self) -> &B::Device {
        self.inner.device()
    }
}

impl<B: Backend> Drop for CoordinatorState<B> {
    /// Records destruction of the session after either success or failure.
    fn drop(&mut self) {
        self.drops.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Fault-injecting public model adapter; numerical execution and caches stay real.
#[derive(burn::module::Module, Debug)]
struct CoordinatorModel<B: Backend> {
    /// Production Qwen numerical graph retained under fault injection.
    inner: Qwen3_5ForCausalLm<B>,
    /// Committed cache work observed at each public model return boundary.
    #[module(skip)]
    calls: std::sync::Arc<std::sync::Mutex<Vec<CoordinatorCall>>>,
    /// Number of dropped cache sessions, proving cleanup after errors.
    #[module(skip)]
    drops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// One-based call ordinal and fault injected after that model call.
    #[module(skip)]
    fault: Option<(usize, CoordinatorFault)>,
}

impl CoordinatorModel<TestBackend> {
    /// Records committed real-cache work before optionally replacing the return value.
    fn returned(
        &self,
        stage: &'static str,
        width: usize,
        state: &CoordinatorState<TestBackend>,
        mut output: crate::CausalLmOutput<TestBackend>,
    ) -> crate::Result<crate::CausalLmOutput<TestBackend>> {
        let mut calls = self.calls.lock().unwrap();
        calls.push(CoordinatorCall {
            stage,
            width,
            lengths: state.lengths().to_vec(),
        });
        if let Some((ordinal, fault)) = self.fault
            && calls.len() == ordinal
        {
            match fault {
                CoordinatorFault::ModelError => {
                    return Err(Error::Checkpoint(format!(
                        "model failed after {stage} call {ordinal}"
                    )));
                }
                CoordinatorFault::LogitShape(shape) => {
                    output.logits = Tensor::zeros(shape, &output.logits.device())
                }
            }
        }
        Ok(output)
    }
}

impl crate::CausalLanguageModel<TestBackend> for CoordinatorModel<TestBackend> {
    /// Configuration inherited from the real model fixture.
    type Config = Qwen3_5TextConfig;
    /// Real cache state augmented with a destruction counter.
    type State = CoordinatorState<TestBackend>;

    /// Leaves stateless inference unchanged.
    fn forward(
        &self,
        input: CausalLmInput<TestBackend>,
        options: &crate::ForwardOptions,
    ) -> crate::Result<crate::CausalLmOutput<TestBackend>> {
        self.inner.forward(input, options)
    }
    /// Allocates a real cache and attaches only a destruction probe.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &<TestBackend as BackendTypes>::Device,
    ) -> crate::Result<Self::State> {
        Ok(CoordinatorState {
            inner: self.inner.new_decode_state(config, batch_size, device)?,
            drops: self.drops.clone(),
        })
    }
    /// Executes dense prompt chunks before fault-injecting the public return boundary.
    fn prefill(
        &self,
        input: CausalLmInput<TestBackend>,
        state: &mut Self::State,
        options: &crate::ForwardOptions,
    ) -> crate::Result<crate::CausalLmOutput<TestBackend>> {
        let width = input.token_ids.dims()[1];
        let output = self.inner.prefill(input, &mut state.inner, options)?;
        self.returned("prefill", width, state, output)
    }
    /// Executes ragged chunks or feedback before fault-injecting the return boundary.
    fn decode(
        &self,
        input: crate::DecodeInput<TestBackend>,
        state: &mut Self::State,
        options: &crate::ForwardOptions,
    ) -> crate::Result<crate::CausalLmOutput<TestBackend>> {
        let width = input.token_ids.dims()[1];
        let output = self.inner.decode(input, &mut state.inner, options)?;
        self.returned("decode", width, state, output)
    }
}

/// Builds a dense or ragged two-row request with four physical prompt columns.
fn coordinator_contract_fixture(
    ragged: bool,
    fault: CoordinatorFault,
) -> (
    CoordinatorModel<TestBackend>,
    CausalGenerationRequest<TestBackend>,
) {
    let (inner, device) = tiny_model();
    let model = CoordinatorModel {
        inner,
        calls: Default::default(),
        drops: Default::default(),
        fault: Some((2, fault)),
    };
    let request = CausalGenerationRequest {
        prompt: CausalLmInput {
            token_ids: Tensor::from_ints([[1, 2, 3, 4], [5, 6, 7, 8]], &device),
            token_mask: ragged.then(|| {
                TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true, false, false], [false, false, true, true]].into(),
                    &device,
                ))
            }),
            position_ids: None,
        },
        cache: CacheConfig::new(2, 8).unwrap(),
        generation: GenerationConfig {
            max_new_tokens: 3,
            ..Default::default()
        },
        prefill_chunk_size: Some(1),
        vocabulary_size: 32,
    };
    (model, request)
}

/// Failed later prompt chunks propagate exactly and discard partially filled sessions.
#[test]
fn coordinator_contract_later_chunk_model_errors_drop_state_before_any_emission() {
    for ragged in [false, true] {
        let (mut model, request) =
            coordinator_contract_fixture(ragged, CoordinatorFault::ModelError);
        let mut events = Vec::new();
        let error = generate_causal_streaming(&model, request.clone(), |event| {
            events.push(event);
            Ok(())
        })
        .unwrap_err();
        let stage = if ragged { "decode" } else { "prefill" };
        assert!(
            matches!(error, Error::Checkpoint(message) if message == format!("model failed after {stage} call 2"))
        );
        assert!(events.is_empty());
        let calls = model.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[0],
            CoordinatorCall {
                stage,
                width: 1,
                lengths: if ragged { vec![1, 0] } else { vec![1, 1] }
            }
        );
        assert_eq!(
            calls[1],
            CoordinatorCall {
                stage,
                width: 1,
                lengths: if ragged { vec![2, 0] } else { vec![2, 2] }
            }
        );
        drop(calls);
        assert_eq!(model.drops.load(std::sync::atomic::Ordering::SeqCst), 1);
        // Reusing the same model starts a fresh cache, not the abandoned prefix.
        model.fault = None;
        model.calls.lock().unwrap().clear();
        let output = generate_causal(&model, request).unwrap();
        assert!(output.sequences.iter().all(|row| row.token_ids.len() == 3));
        assert_eq!(
            model.calls.lock().unwrap()[0].lengths,
            if ragged { vec![1, 0] } else { vec![1, 1] }
        );
        assert_eq!(model.drops.load(std::sync::atomic::Ordering::SeqCst), 2);
    }
}

/// A malformed later chunk must not publish tokens even after an earlier valid chunk.
#[test]
fn coordinator_contract_later_chunk_logits_validate_every_output_axis() {
    for ragged in [false, true] {
        for shape in [[1, 1, 32], [2, 2, 32], [2, 1, 31]] {
            let (model, request) =
                coordinator_contract_fixture(ragged, CoordinatorFault::LogitShape(shape));
            let mut events = Vec::new();
            let error = generate_causal_streaming(&model, request, |event| {
                events.push(event);
                Ok(())
            })
            .unwrap_err();
            let expected = Error::shape(
                "generation.model_logits",
                "[2, 1, 32] for prefill chunk 1..2",
                format!("{shape:?}"),
            );
            assert_eq!(format!("{error:?}"), format!("{expected:?}"));
            assert!(events.is_empty());
            assert_eq!(model.calls.lock().unwrap().len(), 2);
            assert_eq!(model.drops.load(std::sync::atomic::Ordering::SeqCst), 1);
        }
    }
}

/// Invalid generated-step logits cannot emit a second token or retain cache state.
#[test]
fn coordinator_contract_decode_logits_fail_after_exactly_one_valid_batch() {
    for shape in [[1, 1, 32], [2, 2, 32], [2, 1, 31]] {
        let (model, mut request) =
            coordinator_contract_fixture(false, CoordinatorFault::LogitShape(shape));
        request.prefill_chunk_size = None;
        let mut events = Vec::new();
        let error = generate_causal_streaming(&model, request, |event| {
            events.push(event);
            Ok(())
        })
        .unwrap_err();
        let expected = Error::shape(
            "generation.model_logits",
            "[2, 1, 32]",
            format!("{shape:?}"),
        );
        assert_eq!(format!("{error:?}"), format!("{expected:?}"));
        assert_eq!(
            events
                .iter()
                .map(|event| (event.sequence, event.step))
                .collect::<Vec<_>>(),
            [(0, 0), (1, 0)]
        );
        assert!(
            events
                .iter()
                .all(|event| event.log_probability.is_finite() && event.finish_reason.is_none())
        );
        assert_eq!(
            model
                .calls
                .lock()
                .unwrap()
                .iter()
                .map(|call| call.stage)
                .collect::<Vec<_>>(),
            ["prefill", "decode"]
        );
        assert_eq!(model.drops.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}

/// Empty physical prompt axes fail before a model session or callback can become observable.
#[test]
fn release_v2_empty_prompt_axes_reject_before_session_allocation() {
    for shape in [[0, 2], [2, 0]] {
        for chunk_size in [None, Some(1)] {
            let (mut model, mut request) =
                coordinator_contract_fixture(false, CoordinatorFault::ModelError);
            model.fault = None;
            let device = request.prompt.token_ids.device();
            request.prompt.token_ids = Tensor::zeros(shape, &device);
            request.prefill_chunk_size = chunk_size;
            let mut callbacks = 0;
            let error = generate_causal_streaming(&model, request, |_| {
                callbacks += 1;
                Ok(())
            })
            .expect_err("an empty prompt cannot start a generation session");
            assert!(matches!(error, Error::Shape { ref tensor, ref actual, .. }
                if tensor == "generation.prompt.token_ids" && actual == &format!("{shape:?}")));
            assert_eq!(callbacks, 0);
            assert!(model.calls.lock().unwrap().is_empty());
            assert_eq!(model.drops.load(std::sync::atomic::Ordering::SeqCst), 0);
        }
    }
}

/// Streaming preserves longest-stop precedence, generated-only matching, and terminal probabilities.
#[test]
fn release_v2_overlapping_stops_stream_at_the_exact_cache_boundary() {
    let (model, mut request) = finished_row_fixture(SamplingStrategy::Sample, None);
    request.cache = CacheConfig::new(2, 3).unwrap();
    request.generation.max_new_tokens = 2;
    request.generation.eos_token_ids.clear();
    // Row one's prompt ends in 8: [8, 4] must not terminate its first generated
    // token. At token 5 the longest generated suffix wins over [5] and Length.
    request.generation.stop_sequences = vec![vec![8, 4], vec![5], vec![4, 5], vec![2]];
    let mut events = Vec::new();
    let output = generate_causal_streaming(&model, request, |event| {
        events.push(event);
        Ok(())
    })
    .expect("the last emitted token needs no extra feedback cache slot");
    let first_reason = FinishReason::StopSequence {
        index: 3,
        length: 1,
    };
    let second_reason = FinishReason::StopSequence {
        index: 2,
        length: 2,
    };
    assert_eq!(output.sequences[0].token_ids, [2]);
    assert_eq!(output.sequences[1].token_ids, [4, 5]);
    assert_eq!(output.sequences[0].finish_reason, first_reason);
    assert_eq!(output.sequences[1].finish_reason, second_reason);
    for sequence in &output.sequences {
        assert_eq!(sequence.prompt_length, 2);
        // The fixture's top-k=1 distribution has exactly one possible token,
        // so its processed probability is one, even with repetition penalties.
        assert_eq!(
            sequence.log_probabilities,
            vec![0.0; sequence.token_ids.len()]
        );
    }
    assert_eq!(
        events,
        [
            GenerationToken {
                sequence: 0,
                step: 0,
                token_id: 2,
                log_probability: 0.0,
                finish_reason: Some(first_reason)
            },
            GenerationToken {
                sequence: 1,
                step: 0,
                token_id: 4,
                log_probability: 0.0,
                finish_reason: None
            },
            GenerationToken {
                sequence: 1,
                step: 1,
                token_id: 5,
                log_probability: 0.0,
                finish_reason: Some(second_reason)
            },
        ]
    );
    assert_eq!(*model.calls.lock().unwrap(), ["cache", "prefill", "decode"]);
    assert_eq!(
        *model.feedback.lock().unwrap(),
        [FinishedRowFeedback {
            ids: vec![0, 4],
            valid: vec![false, true],
            positions: vec![7, 12],
            lengths: vec![2, 3],
        }]
    );
}
