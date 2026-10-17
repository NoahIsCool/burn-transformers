//! Loads a local Llama-family checkpoint, tokenizes a prompt, and generates text.
//!
//! Any `llama`, `mistral`, `qwen2`, `qwen3`, `smollm3`, or `olmo2` checkpoint
//! directory works: the automatic registry reads `config.json`, strictly loads
//! every SafeTensors weight (cast to F32 while streaming), and loads
//! `tokenizer.json`. Nothing is downloaded. Generation uses the checkpoint's
//! `generation_config.json` defaults with the requested token budget, one
//! batched prefill, and cached single-token decode steps on the CPU.
//!
//! ```text
//! cargo run --release --locked --features tokenizers --example llama_generate -- MODEL_DIR ["PROMPT"] [MAX_NEW_TOKENS]
//! ```

use std::{env, error::Error as StdError, path::PathBuf};

use burn::backend::Flex;
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalGenerationRequest, EncodeOptions, Error,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
    generate_causal,
};

/// CPU backend with F32 activations.
type Cpu = Flex<f32, i32>;

/// Loads the model, generates a continuation, and prints it.
fn main() -> Result<(), Box<dyn StdError>> {
    let (root, text, max_new_tokens) = arguments()?;
    let device = Default::default();
    let options = AutoLoadOptions {
        checkpoint: LoadOptions {
            dtype_policy: DTypePolicy::Cast(TensorDType::F32),
            ..LoadOptions::default()
        },
        ..AutoLoadOptions::default()
    };
    let loaded = AutoCausalLm::<Cpu>::load_local(&root, &options, &device)?;
    let tokenizer = loaded
        .tokenizer()
        .ok_or_else(|| Error::Tokenizer("the model directory has no tokenizer.json".into()))?;
    let prompt = tokenizer
        .encode(&text, &EncodeOptions::default())?
        .to_causal_input::<Cpu>(&device)?;
    let [batch, prompt_tokens] = prompt.token_ids.dims();

    let mut generation = loaded.generation_defaults().clone();
    generation.max_new_tokens = max_new_tokens;
    let output = generate_causal(
        loaded.model(),
        CausalGenerationRequest {
            prompt,
            cache: CacheConfig::new(batch, prompt_tokens + max_new_tokens)?,
            generation,
            prefill_chunk_size: None,
            vocabulary_size: loaded.model().config().vocabulary_size(),
        },
    )?;
    for sequence in &output.sequences {
        println!("{text}{}", tokenizer.decode(&sequence.token_ids, true)?);
    }
    Ok(())
}

/// Parses `MODEL_DIR [PROMPT] [MAX_NEW_TOKENS]`.
fn arguments() -> Result<(PathBuf, String, usize), Box<dyn StdError>> {
    let usage = "usage: llama_generate MODEL_DIR [PROMPT] [MAX_NEW_TOKENS]";
    let mut arguments = env::args().skip(1);
    let root = arguments.next().map(PathBuf::from).ok_or(usage)?;
    let prompt = arguments
        .next()
        .unwrap_or_else(|| "The three primary colors are".into());
    let max_new_tokens = match arguments.next() {
        Some(value) => value.parse::<usize>()?,
        None => 32,
    };
    if max_new_tokens == 0 || arguments.next().is_some() {
        return Err(usage.into());
    }
    Ok((root, prompt, max_new_tokens))
}
