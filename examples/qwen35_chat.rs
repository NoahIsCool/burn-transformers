//! Runs local text-only chat generation with the pinned Qwen3.5-0.8B profile.
//!
//! The model repository is multimodal, but this crate intentionally loads only
//! its language tower. The example casts the BF16 checkpoint to F16 while each
//! tensor is streamed because that is the validated M3 Metal execution policy.
//!
//! Tokenized chat uses `[batch, token]` IDs with causal visibility. The prepared
//! BF16 weights are streamed into the explicit F16 Metal execution policy; this
//! program does not imply quantized-training support. Decode state is external to
//! parameters: its capacity covers the complete prompt plus the requested token
//! budget, and prefill is chunked into at most 256 tokens. Generation uses the
//! checkpoint's defaults after overriding `max_new_tokens`. No optimizer records
//! or training dropout participate in this inference-only path.

#[cfg(all(target_os = "macos", feature = "metal", feature = "chat-templates"))]
use std::{env, error::Error as StdError, io, path::PathBuf};

#[cfg(all(target_os = "macos", feature = "metal", feature = "chat-templates"))]
use burn::backend::Metal;
#[cfg(all(target_os = "macos", feature = "metal", feature = "chat-templates"))]
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalGenerationRequest, ChatMessage, ChatTemplateOptions,
    EncodeOptions, Error,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
    generate_causal,
};

/// Immutable upstream identity used by the v0.1 Qwen profile.
#[cfg(all(target_os = "macos", feature = "metal", feature = "chat-templates"))]
const QWEN_REPOSITORY: &str = "Qwen/Qwen3.5-0.8B";

/// Exact artifact revision covered by the committed acceptance evidence.
#[cfg(all(target_os = "macos", feature = "metal", feature = "chat-templates"))]
const QWEN_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";

/// Loads one pinned local checkpoint, renders a no-thinking chat, and generates text.
#[cfg(all(target_os = "macos", feature = "metal", feature = "chat-templates"))]
fn main() -> Result<(), Box<dyn StdError>> {
    let (root, prompt, max_new_tokens) = arguments()?;
    let device = Default::default();
    let options = AutoLoadOptions {
        checkpoint: LoadOptions {
            dtype_policy: DTypePolicy::Cast(TensorDType::F16),
            ..LoadOptions::default()
        },
        repository: Some(QWEN_REPOSITORY.into()),
        revision: Some(QWEN_REVISION.into()),
        ..AutoLoadOptions::default()
    };
    let loaded = AutoCausalLm::<Metal>::load_local(&root, &options, &device)?;
    let tokenizer = loaded
        .tokenizer()
        .ok_or_else(|| Error::Tokenizer("the local Qwen profile requires tokenizer.json".into()))?;
    let encoded = tokenizer.encode_chat(
        &[ChatMessage::text("user", prompt)],
        &ChatTemplateOptions {
            add_generation_prompt: true,
            enable_thinking: Some(false),
            ..ChatTemplateOptions::default()
        },
        &EncodeOptions::default(),
    )?;
    let prompt = encoded.to_causal_input::<Metal>(&device)?;
    let [batch, prompt_width] = prompt.token_ids.dims();

    let mut generation = loaded.generation_defaults().clone();
    generation.max_new_tokens = max_new_tokens;
    let cache_capacity = prompt_width
        .checked_add(max_new_tokens.saturating_sub(1))
        .ok_or_else(|| io::Error::other("prompt and generation length overflow usize"))?;
    let vocabulary_size = loaded.model().config().vocabulary_size();
    let output = generate_causal(
        loaded.model(),
        CausalGenerationRequest {
            prompt,
            cache: CacheConfig::new(batch, cache_capacity)?,
            generation,
            prefill_chunk_size: Some(256),
            vocabulary_size,
        },
    )?;
    let sequence = output
        .sequences
        .first()
        .ok_or_else(|| io::Error::other("generation returned no sequence"))?;
    println!("{}", tokenizer.decode(&sequence.token_ids, true)?);
    Ok(())
}

/// Parses `MODEL_DIR [PROMPT] [MAX_NEW_TOKENS]` without adding a CLI dependency.
#[cfg(all(target_os = "macos", feature = "metal", feature = "chat-templates"))]
fn arguments() -> Result<(PathBuf, String, usize), Box<dyn StdError>> {
    let mut arguments = env::args_os();
    let executable = arguments
        .next()
        .unwrap_or_else(|| "qwen35_chat".into())
        .to_string_lossy()
        .into_owned();
    let root = arguments.next().map(PathBuf::from).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("usage: {executable} MODEL_DIR [PROMPT] [MAX_NEW_TOKENS]"),
        )
    })?;
    let prompt = arguments
        .next()
        .map(|value| {
            value.into_string().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "PROMPT must be valid UTF-8")
            })
        })
        .transpose()?
        .unwrap_or_else(|| "Explain why a KV cache speeds up decoding in one sentence.".into());
    let max_new_tokens = arguments
        .next()
        .map(|value| {
            value
                .to_str()
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "MAX_NEW_TOKENS must be valid UTF-8",
                    )
                })?
                .parse::<usize>()
                .map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("invalid MAX_NEW_TOKENS: {error}"),
                    )
                })
        })
        .transpose()?
        .unwrap_or(64);
    if max_new_tokens == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "MAX_NEW_TOKENS must be greater than zero",
        )
        .into());
    }
    if arguments.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("usage: {executable} MODEL_DIR [PROMPT] [MAX_NEW_TOKENS]"),
        )
        .into());
    }
    Ok((root, prompt, max_new_tokens))
}

/// Explains the build-time feature contract on unsupported example builds.
#[cfg(not(all(target_os = "macos", feature = "metal", feature = "chat-templates")))]
fn main() {
    eprintln!(
        "qwen35_chat requires macOS and --no-default-features --features std,metal,chat-templates"
    );
}
