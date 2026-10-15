//! Loads the pinned official FLAN-T5-small artifact and runs cached beam search.
//!
//! The runtime is deliberately local-only: download the immutable checkpoint
//! outside the program and pass its directory as the first argument.
//!
//! ```text
//! cargo run --release --locked --features tokenizers --example flan_t5_small -- \
//!   /path/to/flan-t5-small "translate English to German: The house is wonderful."
//! ```
//!
//! Flex executes F32 tensors with I32 IDs. Encoder attention is bidirectional;
//! decoder self-attention is causal, while cross-attention reads valid source
//! tokens. Source masks travel with tokenized input and cached decode state stays
//! external to model parameters. Four beams accumulate log-probability scores
//! under the generator's configured defaults; at most 64 new tokens are produced.
//! The decoded text goes to standard output and the best hypothesis's finish
//! reason, length-normalized score, and length to standard error. This
//! inference-only path uses no autodiff optimizer or training dropout.

use std::{error::Error as StdError, path::PathBuf};

use burn::backend::Flex;
use transformers::{
    AutoLoadOptions, AutoSeq2SeqLm, EncodeOptions,
    models::t5::{T5BeamSearchOptions, generate_t5_beam_search},
};

/// CPU backend with F32 activations and I32 token IDs.
type Cpu = Flex<f32, i32>;

/// Loads, tokenizes, generates, and decodes through the public release surface.
fn main() -> Result<(), Box<dyn StdError>> {
    let usage = "usage: flan_t5_small MODEL_DIR [PROMPT]";
    let mut arguments = std::env::args().skip(1);
    let root = arguments.next().map(PathBuf::from).ok_or(usage)?;
    let prompt = arguments
        .next()
        .unwrap_or_else(|| "translate English to German: The house is wonderful.".into());
    if arguments.next().is_some() {
        return Err(usage.into());
    }

    let device = Default::default();
    let loaded = AutoSeq2SeqLm::<Cpu>::load_local(
        &root,
        &AutoLoadOptions {
            repository: Some("google/flan-t5-small".into()),
            revision: Some("0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab".into()),
            ..AutoLoadOptions::default()
        },
        &device,
    )?;
    let tokenizer = loaded
        .tokenizer()
        .ok_or("the model directory has no tokenizer.json")?;
    let source = tokenizer
        .encode(&prompt, &EncodeOptions::default())?
        .to_causal_input::<Cpu>(&device)?;
    // The automatic registry is open to future encoder-decoder families; the
    // T5 helper reads its decoder-start, padding, and EOS tokens from the config.
    let AutoSeq2SeqLm::T5(model) = loaded.model() else {
        return Err("expected a T5-family checkpoint".into());
    };
    let generated = generate_t5_beam_search(
        model,
        source,
        T5BeamSearchOptions {
            max_new_tokens: 64,
            num_beams: 4,
            num_return_sequences: 1,
            ..T5BeamSearchOptions::default()
        },
    )?;
    let best = generated
        .sequences
        .first()
        .and_then(|hypotheses| hypotheses.first())
        .ok_or("beam search returned no hypothesis")?;
    println!("{}", tokenizer.decode(&best.token_ids, true)?);
    eprintln!(
        "finish={:?} score={:.6} tokens={}",
        best.finish_reason,
        best.score,
        best.token_ids.len()
    );
    Ok(())
}
