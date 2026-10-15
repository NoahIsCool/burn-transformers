//! Streams a text-only chat reply from the pinned Qwen3.5-0.8B checkpoint on Metal.
//!
//! ```text
//! cargo run --release --locked --no-default-features --features std,metal,chat-templates \
//!   --example qwen35_chat -- MODEL_DIR ["PROMPT"] [MAX_NEW_TOKENS]
//! ```
//!
//! macOS only. The model repository is multimodal, but this crate intentionally
//! loads only its language tower. The BF16 checkpoint is cast to F16 while each
//! tensor is streamed because that is the validated M3 Metal execution policy.
//!
//! Tokenized chat uses `[batch, token]` IDs with causal visibility. Decode state
//! is external to parameters: its capacity covers the complete prompt plus the
//! requested token budget, and prefill is chunked into at most 256 tokens.
//! Generation uses the checkpoint's defaults after overriding `max_new_tokens`;
//! settings in `generation_config.json` that the crate does not implement are
//! listed on standard error. Text is streamed as it is generated: an
//! incremental decoder emits each piece once its UTF-8 characters are complete,
//! so multi-byte characters split across tokens never print as replacement
//! glyphs. No optimizer records or training dropout participate in this
//! inference-only path.

use std::{
    env,
    error::Error as StdError,
    io::{self, Write},
    path::PathBuf,
};

use burn::backend::Metal;
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalGenerationRequest, ChatMessage, ChatTemplateOptions,
    EncodeOptions, Error,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
    generate_causal_streaming,
};

/// Immutable upstream identity used by the v0.1 Qwen profile.
const QWEN_REPOSITORY: &str = "Qwen/Qwen3.5-0.8B";

/// Exact artifact revision covered by the committed acceptance evidence.
const QWEN_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";

/// Loads one pinned local checkpoint, renders a no-thinking chat, and streams the reply.
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
    for (setting, value) in loaded.unsupported_generation_settings() {
        eprintln!("ignoring unsupported generation setting {setting} = {value}");
    }
    let tokenizer = loaded
        .tokenizer()
        .ok_or("the local Qwen profile requires tokenizer.json")?;
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
    // The last sampled token is never fed back, so it needs no cache slot.
    let cache_capacity = prompt_width
        .checked_add(max_new_tokens - 1)
        .ok_or("prompt and generation length overflow usize")?;
    let vocabulary_size = loaded.model().config().vocabulary_size();
    let mut text = tokenizer.decode_stream(true);
    let mut stdout = io::stdout().lock();
    generate_causal_streaming(
        loaded.model(),
        CausalGenerationRequest {
            prompt,
            cache: CacheConfig::new(batch, cache_capacity)?,
            generation,
            prefill_chunk_size: Some(256),
            vocabulary_size,
        },
        |token| {
            if let Some(piece) = text.step(token.token_id)? {
                write!(stdout, "{piece}")
                    .and_then(|()| stdout.flush())
                    .map_err(|error| Error::Generation(format!("failed to print: {error}")))?;
            }
            Ok(())
        },
    )?;
    writeln!(stdout)?;
    Ok(())
}

/// Parses `MODEL_DIR [PROMPT] [MAX_NEW_TOKENS]` without adding a CLI dependency.
fn arguments() -> Result<(PathBuf, String, usize), Box<dyn StdError>> {
    let usage = "usage: qwen35_chat MODEL_DIR [PROMPT] [MAX_NEW_TOKENS]";
    let mut arguments = env::args().skip(1);
    let root = arguments.next().map(PathBuf::from).ok_or(usage)?;
    let prompt = arguments
        .next()
        .unwrap_or_else(|| "Explain why a KV cache speeds up decoding in one sentence.".into());
    let max_new_tokens = match arguments.next() {
        Some(value) => value.parse::<usize>()?,
        None => 64,
    };
    if max_new_tokens == 0 || arguments.next().is_some() {
        return Err(usage.into());
    }
    Ok((root, prompt, max_new_tokens))
}
