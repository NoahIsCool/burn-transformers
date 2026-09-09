//! Real-checkpoint v0.1 acceptance through the public Qwen loading and generation APIs.
//!
//! The ordinary test run exercises host-side evidence helpers only. The ignored
//! Metal gates require the pinned checkpoint and independently authenticated
//! official oracle. Padding is semantic (`true == valid`): adding physical PAD
//! columns must not change token positions, recurrent history, KV lengths, or
//! next-token distributions. Every numerical comparison covers the complete
//! vocabulary and retains the release cosine threshold of 0.999.

#![cfg(all(target_os = "macos", feature = "metal", feature = "chat-templates"))]

use std::{
    fs::OpenOptions,
    io::Read,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

use burn::{
    backend::Metal,
    tensor::{DType, Int, Tensor, TensorData, backend::BackendTypes},
};
use sha2::{Digest, Sha256};
use transformers::{
    AutoCausalLm, AutoLoadOptions, BatchEncoding, CausalGenerationRequest, ChatMessage,
    ChatTemplateOptions, DecodeInput, DecodeState, EncodeOptions, FinishReason, ForwardOptions,
    GeneratedSequence, GenerationConfig, GenerationToken, LoadedModel, Padding, PaddingSide,
    SamplingStrategy,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
    generate_causal, generate_causal_streaming,
};

#[path = "support/qwen35_release_reference.rs"]
mod qwen35_release_reference;

/// Device type selected by Burn's Metal backend.
type Device = <Metal as BackendTypes>::Device;
/// Exact upstream identity covered by this acceptance test.
const REPOSITORY: &str = "Qwen/Qwen3.5-0.8B";
/// Immutable upstream revision authenticated before and after each test.
const REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
/// Prompt selected solely by the official-reference stability policy.
const USER_PROMPT: &str = "Explain why the seasons change in four complete sentences.";
/// Number of categories in the model's padded language head.
const VOCABULARY: usize = 248_320;

/// Streams exact regular-file bytes without following a final symlink.
///
/// The shared oracle reader calls this function for target checkpoint identity.
/// Fixed-size reads keep the 1.7 GB shard out of an additional host allocation.
fn sha256_regular_file(path: &Path, expected_bytes: u64) -> String {
    const O_NOFOLLOW: i32 = 0x100;
    const O_NONBLOCK: i32 = 0x4;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW | O_NONBLOCK)
        .open(path)
        .unwrap_or_else(|error| panic!("open authenticated file {}: {error}", path.display()));
    let metadata = file.metadata().expect("inspect authenticated descriptor");
    assert!(
        metadata.is_file(),
        "authenticated input must be a regular file"
    );
    assert_eq!(
        metadata.len(),
        expected_bytes,
        "authenticated file length changed"
    );
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut digest = Sha256::new();
    let mut observed = 0_u64;
    loop {
        let count = file
            .read(&mut buffer)
            .expect("read authenticated descriptor");
        if count == 0 {
            break;
        }
        observed = observed
            .checked_add(count as u64)
            .expect("file byte count fits u64");
        assert!(
            observed <= expected_bytes,
            "authenticated input grew while hashing"
        );
        digest.update(&buffer[..count]);
    }
    assert_eq!(
        observed, expected_bytes,
        "authenticated input shrank while hashing"
    );
    format!("{:x}", digest.finalize())
}

/// Authenticates external evidence before Auto loading allocates model parameters.
fn load_profile(
    device: &Device,
) -> (
    LoadedModel<AutoCausalLm<Metal>>,
    qwen35_release_reference::ReferenceBundle,
) {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR").expect("set TRANSFORMERS_QWEN35_DIR"),
    );
    let reference_parent = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_REFERENCE_DIR")
            .expect("set TRANSFORMERS_QWEN35_REFERENCE_DIR"),
    );
    let reference = qwen35_release_reference::load(&reference_parent);
    reference.validate_checkpoint(&root);
    let loaded = AutoCausalLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            repository: Some(REPOSITORY.into()),
            revision: Some(REVISION.into()),
            ..AutoLoadOptions::default()
        },
        device,
    )
    .expect("strict public Auto loader must consume the authenticated checkpoint");
    assert!(matches!(loaded.model(), AutoCausalLm::Qwen3_5(_)));
    assert_eq!(loaded.model().config().vocabulary_size(), VOCABULARY);
    assert_eq!(loaded.source().repository(), Some(REPOSITORY));
    assert_eq!(loaded.source().revision(), Some(REVISION));
    assert_eq!(
        loaded.source().root(),
        root.canonicalize().expect("checkpoint root")
    );
    assert!(loaded.load_report().is_strictly_valid());
    assert_eq!(loaded.load_report().allowlisted_ignored.len(), 168);
    assert!(loaded.load_report().quantized.is_empty());
    let stats = loaded
        .execution_stats()
        .expect("Auto loading retains execution statistics");
    assert_eq!(stats.tensors_applied, 320);
    assert_eq!(stats.aliases_applied, 1);
    loaded
        .tokenizer()
        .expect("public Auto loading attaches the local tokenizer")
        .validate_model_vocabulary(VOCABULARY)
        .expect("tokenizer vocabulary compatibility");
    (loaded, reference)
}

/// Renders the checkpoint's exact non-thinking chat delimiter convention.
fn chat_options() -> ChatTemplateOptions {
    ChatTemplateOptions {
        add_generation_prompt: true,
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    }
}

/// Selects greedy generation while preserving normalized EOS and PAD defaults.
fn greedy_config(loaded: &LoadedModel<AutoCausalLm<Metal>>, length: usize) -> GenerationConfig {
    let mut generation = loaded.generation_defaults().clone();
    assert!(!generation.eos_token_ids.is_empty());
    // This checkpoint leaves model PAD unspecified. Tokenization supplies its
    // own PAD, while generation may retain None and mask its inert placeholder.
    assert!(generation.pad_token_id.is_none_or(|token| token == 248_044));
    assert_eq!(
        loaded
            .tokenizer()
            .expect("tokenizer")
            .metadata()
            .special_tokens
            .pad
            .as_ref()
            .map(|token| token.id),
        Some(248_044)
    );
    assert_eq!(generation.repetition_penalty, 1.0);
    assert_eq!(generation.presence_penalty, 0.0);
    generation.strategy = SamplingStrategy::Greedy;
    generation.max_new_tokens = length;
    generation
}

/// Checks token identity and terminal metadata returned by either generation API.
fn assert_sequence(sequence: &GeneratedSequence, prompt_length: usize, expected: &[u32]) {
    assert_eq!(sequence.prompt_length, prompt_length);
    assert_eq!(sequence.token_ids, expected);
    assert_eq!(sequence.log_probabilities.len(), expected.len());
    assert!(
        sequence
            .log_probabilities
            .iter()
            .all(|value| value.is_finite() && *value <= 0.0)
    );
    assert_eq!(sequence.finish_reason, FinishReason::Length);
}

/// Computes full-vector cosine in F64, rejecting malformed or non-finite rows.
fn cosine(left: &[f32], right: &[f32]) -> f64 {
    assert!(!left.is_empty() && left.len() == right.len());
    assert!(left.iter().chain(right).all(|value| value.is_finite()));
    let (mut dot, mut left_square, mut right_square) = (0_f64, 0_f64, 0_f64);
    for (&left, &right) in left.iter().zip(right) {
        let (left, right) = (f64::from(left), f64::from(right));
        dot += left * right;
        left_square += left * left;
        right_square += right * right;
    }
    assert!(left_square > 0.0 && right_square > 0.0);
    dot / (left_square.sqrt() * right_square.sqrt())
}

/// Uses numeric comparison so signed-zero ties select the first vocabulary ID.
fn top1(row: &[f32]) -> u32 {
    assert!(!row.is_empty() && row.iter().all(|value| value.is_finite()));
    let mut best = 0;
    for index in 1..row.len() {
        if row[index] > row[best] {
            best = index;
        }
    }
    u32::try_from(best).expect("vocabulary indices fit u32")
}

/// Transfers selected `[batch, 1, vocabulary]` logits for numerical comparisons.
fn logits_rows(logits: Tensor<Metal, 3>, batch: usize) -> Vec<Vec<f32>> {
    assert_eq!(logits.dims(), [batch, 1, VOCABULARY]);
    logits
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("Metal F32 logits")
        .chunks_exact(VOCABULARY)
        .map(<[f32]>::to_vec)
        .collect()
}

/// Checks full-vocabulary similarity and the exact greedy decision together.
fn assert_logits(label: &str, actual: &[f32], expected: &[f32]) {
    let similarity = cosine(actual, expected);
    assert!(
        similarity >= 0.999,
        "{label}: cosine {similarity} below 0.999"
    );
    assert_eq!(
        top1(actual),
        top1(expected),
        "{label}: greedy decision changed"
    );
}

/// Proves tokenizer padding contains actual masked columns and preserves valid IDs.
fn assert_genuine_padding(padded: &BatchEncoding, rows: &[BatchEncoding]) {
    assert_eq!(padded.batch_size(), 2);
    assert_ne!(rows[0].token_ids()[0].len(), rows[1].token_ids()[0].len());
    assert!(padded.valid_tokens().iter().flatten().any(|valid| !valid));
    for (index, single) in rows.iter().enumerate() {
        let valid_ids = padded.token_ids()[index]
            .iter()
            .zip(&padded.valid_tokens()[index])
            .filter_map(|(&token, &valid)| valid.then_some(token))
            .collect::<Vec<_>>();
        assert_eq!(valid_ids, single.token_ids()[0]);
    }
}

/// Exercises strict Auto loading, chat rendering, tokenization, and streamed generation.
#[test]
#[ignore = "requires authenticated Qwen3.5-0.8B artifacts and serialized Metal execution"]
fn qwen35_public_chat_generation_matches_official_32_tokens() {
    let device = Default::default();
    let (loaded, reference) = load_profile(&device);
    let tokenizer = loaded.tokenizer().expect("local tokenizer");
    let messages = [ChatMessage::text("user", USER_PROMPT)];
    assert_eq!(
        tokenizer
            .render_chat(&messages, &chat_options())
            .expect("chat rendering"),
        reference.prompt()
    );
    let encoding = tokenizer
        .encode_chat(&messages, &chat_options(), &EncodeOptions::default())
        .expect("chat tokenization");
    assert_eq!(
        encoding.token_ids()[0]
            .iter()
            .copied()
            .map(i64::from)
            .collect::<Vec<_>>(),
        reference.input_ids()
    );
    let expected = reference
        .greedy_token_ids()
        .iter()
        .map(|&token| u32::try_from(token).expect("token ID"))
        .collect::<Vec<_>>();
    let prompt_length = encoding.token_ids()[0].len();
    let mut events = Vec::<GenerationToken>::new();
    let output = generate_causal_streaming(
        loaded.model(),
        CausalGenerationRequest {
            prompt: encoding.to_causal_input(&device).expect("token tensors"),
            cache: CacheConfig::new(1, prompt_length + expected.len() - 1)
                .expect("exact minimum cache capacity"),
            generation: greedy_config(&loaded, expected.len()),
            prefill_chunk_size: Some(256),
            vocabulary_size: VOCABULARY,
        },
        |event| {
            events.push(event);
            Ok(())
        },
    )
    .expect("public streaming generation must reproduce the official continuation");
    assert_eq!(output.sequences.len(), 1);
    assert_sequence(&output.sequences[0], prompt_length, &expected);
    assert_eq!(events.len(), expected.len());
    for (step, event) in events.iter().enumerate() {
        assert_eq!(
            (event.sequence, event.step, event.token_id),
            (0, step, expected[step])
        );
        assert_eq!(
            event.log_probability,
            output.sequences[0].log_probabilities[step]
        );
        assert_eq!(
            event.finish_reason,
            (step + 1 == expected.len()).then_some(FinishReason::Length)
        );
    }
    assert!(
        !tokenizer
            .decode(&expected, true)
            .expect("decode public continuation")
            .is_empty()
    );
    reference.validate_checkpoint(loaded.source().root());
}

/// Compares padded batch execution with independently evaluated unpadded rows.
#[test]
#[ignore = "requires authenticated Qwen3.5-0.8B artifacts and serialized Metal execution"]
fn qwen35_padded_batch_matches_independent_prefill_decode_and_generation() {
    const STEPS: usize = 4;
    let device = Default::default();
    let (loaded, reference) = load_profile(&device);
    let tokenizer = loaded.tokenizer().expect("local tokenizer");
    let rendered = [USER_PROMPT, "Hello"].map(|text| {
        tokenizer
            .render_chat(&[ChatMessage::text("user", text)], &chat_options())
            .expect("render each chat")
    });
    let encode_options = EncodeOptions {
        add_special_tokens: false,
        ..EncodeOptions::default()
    };
    let singles = rendered
        .iter()
        .map(|prompt| {
            tokenizer
                .encode(prompt, &encode_options)
                .expect("independent unpadded tokenization")
        })
        .collect::<Vec<_>>();
    let lengths = singles
        .iter()
        .map(|row| row.token_ids()[0].len())
        .collect::<Vec<_>>();
    let capacity = *lengths.iter().max().expect("two prompts") + STEPS - 1;
    let mut independent = Vec::new();
    for single in &singles {
        independent.push(
            generate_causal(
                loaded.model(),
                CausalGenerationRequest {
                    prompt: single.to_causal_input(&device).expect("independent prompt"),
                    cache: CacheConfig::new(1, capacity).expect("independent cache"),
                    generation: greedy_config(&loaded, STEPS),
                    prefill_chunk_size: None,
                    vocabulary_size: VOCABULARY,
                },
            )
            .expect("independent public generation")
            .sequences
            .remove(0),
        );
    }
    for side in [PaddingSide::Left, PaddingSide::Right] {
        let padded = tokenizer
            .encode_batch(
                &rendered,
                &EncodeOptions {
                    padding: Padding::Longest,
                    padding_side: Some(side),
                    ..encode_options.clone()
                },
            )
            .expect("genuinely padded two-row tokenization");
        assert_genuine_padding(&padded, &singles);
        let input = padded
            .to_causal_input(&device)
            .expect("padded token tensors");
        let batch_full = logits_rows(
            loaded
                .model()
                .forward(input.clone(), &ForwardOptions::default())
                .expect("padded full forward")
                .logits,
            2,
        );
        let mut batch_state = loaded
            .model()
            .new_decode_state(
                CacheConfig::new(2, capacity).expect("batch cache"),
                2,
                &device,
            )
            .expect("batch state");
        let mut batch_rows = logits_rows(
            loaded
                .model()
                .prefill(input.clone(), &mut batch_state, &ForwardOptions::default())
                .expect("padded prefill")
                .logits,
            2,
        );
        let mut single_states = Vec::new();
        let mut single_rows = Vec::new();
        for (row, single) in singles.iter().enumerate() {
            let input = single.to_causal_input(&device).expect("independent input");
            let full = logits_rows(
                loaded
                    .model()
                    .forward(input.clone(), &ForwardOptions::default())
                    .expect("independent full forward")
                    .logits,
                1,
            )
            .remove(0);
            assert_logits(
                "padded versus independent full forward",
                &batch_full[row],
                &full,
            );
            if row == 0 {
                assert_logits(
                    "stable prompt official stateless row",
                    &full,
                    reference.prompt_logits(),
                );
                assert_logits(
                    "stable prompt official teacher row zero",
                    &full,
                    reference.teacher_row(0),
                );
            }
            let mut state = loaded
                .model()
                .new_decode_state(
                    CacheConfig::new(1, capacity).expect("single cache"),
                    1,
                    &device,
                )
                .expect("single state");
            single_rows.push(
                logits_rows(
                    loaded
                        .model()
                        .prefill(input, &mut state, &ForwardOptions::default())
                        .expect("independent prefill")
                        .logits,
                    1,
                )
                .remove(0),
            );
            single_states.push(state);
        }
        assert_eq!(batch_state.lengths(), lengths);
        for step in 0..STEPS {
            for row in 0..2 {
                assert_logits(
                    "padded versus independent cached logits",
                    &batch_rows[row],
                    &single_rows[row],
                );
                assert_eq!(top1(&batch_rows[row]), independent[row].token_ids[step]);
                assert_eq!(single_states[row].lengths(), &[lengths[row] + step]);
            }
            assert_logits(
                "stable prompt official cached decision",
                &batch_rows[0],
                reference.cached_row(step),
            );
            assert_eq!(
                batch_state.lengths(),
                &[lengths[0] + step, lengths[1] + step]
            );
            if step + 1 < STEPS {
                let tokens = batch_rows
                    .iter()
                    .map(|row| i64::from(top1(row)))
                    .collect::<Vec<_>>();
                batch_rows = logits_rows(
                    loaded
                        .model()
                        .decode(
                            DecodeInput {
                                token_ids: Tensor::<Metal, 2, Int>::from_data(
                                    TensorData::new(tokens.clone(), [2, 1]),
                                    &device,
                                ),
                                token_mask: None,
                                position_ids: None,
                            },
                            &mut batch_state,
                            &ForwardOptions::default(),
                        )
                        .expect("padded batch decode")
                        .logits,
                    2,
                );
                for row in 0..2 {
                    single_rows[row] = logits_rows(
                        loaded
                            .model()
                            .decode(
                                DecodeInput {
                                    token_ids: Tensor::<Metal, 2, Int>::from_data(
                                        TensorData::new(vec![tokens[row]], [1, 1]),
                                        &device,
                                    ),
                                    token_mask: None,
                                    position_ids: None,
                                },
                                &mut single_states[row],
                                &ForwardOptions::default(),
                            )
                            .expect("independent decode")
                            .logits,
                        1,
                    )
                    .remove(0);
                }
            }
        }
        let generated = generate_causal(
            loaded.model(),
            CausalGenerationRequest {
                prompt: input,
                cache: CacheConfig::new(2, capacity).expect("generation batch cache"),
                generation: greedy_config(&loaded, STEPS),
                prefill_chunk_size: Some(7),
                vocabulary_size: VOCABULARY,
            },
        )
        .expect("padded public generation with physical prefill chunks");
        assert_eq!(generated.sequences.len(), 2);
        for row in 0..2 {
            assert_sequence(
                &generated.sequences[row],
                lengths[row],
                &independent[row].token_ids,
            );
        }
    }
    reference.validate_checkpoint(loaded.source().root());
}

/// Pure diagnostic math must reject corrupted vectors and preserve first-index ties.
#[test]
fn numerical_helpers_check_finiteness_geometry_and_signed_zero_ties() {
    assert!((cosine(&[1.0, 2.0], &[2.0, 4.0]) - 1.0).abs() < 1e-12);
    assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    assert_eq!(top1(&[-0.0, 0.0]), 0);
    assert_eq!(top1(&[1.0, 3.0, 3.0]), 1);
    assert!(std::panic::catch_unwind(|| cosine(&[f32::NAN], &[1.0])).is_err());
    assert!(std::panic::catch_unwind(|| cosine(&[0.0], &[1.0])).is_err());
    assert!(std::panic::catch_unwind(|| cosine(&[1.0], &[1.0, 2.0])).is_err());
}

/// Descriptor-based authentication rejects symlinks and exact-length mismatches.
#[test]
fn file_authentication_checks_exact_size_and_final_symlinks() {
    let temporary = tempfile::tempdir().expect("temporary evidence");
    let file = temporary.path().join("bytes");
    std::fs::write(&file, b"abc").expect("fixture bytes");
    assert_eq!(
        sha256_regular_file(&file, 3),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert!(std::panic::catch_unwind(|| sha256_regular_file(&file, 2)).is_err());
    let alias = temporary.path().join("alias");
    std::os::unix::fs::symlink(&file, &alias).expect("fixture symlink");
    assert!(std::panic::catch_unwind(|| sha256_regular_file(&alias, 3)).is_err());
}
