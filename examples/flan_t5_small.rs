//! Loads the pinned official FLAN-T5-small artifact and runs cached beam search.
//!
//! The runtime is deliberately local-only. Download the immutable checkpoint outside the program,
//! enable the `flex` and `tokenizers` features, and pass its directory as the first argument:
//!
//! ```text
//! cargo run --release \
//!   --no-default-features --features std,flex,tokenizers \
//!   --example flan_t5_small -- /path/to/flan-t5-small \
//!   "translate English to German: The house is wonderful."
//! ```
//!
//! Flex executes F32 tensors with I32 IDs. Encoder attention is bidirectional;
//! decoder self-attention is causal, while cross-attention reads valid source
//! tokens. Source masks travel with tokenized input and cached decode state stays
//! external to model parameters. Four beams accumulate log-probability scores
//! under the generator's configured defaults; at most 64 new tokens are produced.
//! This inference-only path uses no autodiff optimizer or training dropout.

#[cfg(all(feature = "flex", feature = "tokenizers"))]
use std::{error::Error as StdError, ffi::OsString, io, path::PathBuf};

#[cfg(all(feature = "flex", feature = "tokenizers"))]
use burn::backend::Flex;
#[cfg(all(feature = "flex", feature = "tokenizers"))]
use transformers::{
    AutoLoadOptions, AutoSeq2SeqLm, EncodeOptions,
    models::t5::{T5BeamSearchOptions, generate_t5_beam_search},
};

/// Explains the requirements when the Flex backend or local tokenization is disabled.
#[cfg(not(all(feature = "flex", feature = "tokenizers")))]
fn main() {
    eprintln!("enable the `flex` and `tokenizers` features to run the FLAN-T5-small example");
}

/// Loads, tokenizes, generates, and decodes through the public release surface.
#[cfg(all(feature = "flex", feature = "tokenizers"))]
fn main() -> Result<(), Box<dyn StdError>> {
    let mut arguments = std::env::args_os().skip(1);
    let root = arguments.next().map(PathBuf::from).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: flan_t5_small MODEL_DIRECTORY [PROMPT]",
        )
    })?;
    let prompt = arguments
        .next()
        .unwrap_or_else(|| OsString::from("translate English to German: The house is wonderful."));
    let prompt = prompt
        .into_string()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "prompt must be UTF-8"))?;
    if arguments.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected at most one prompt argument",
        )
        .into());
    }

    let device = Default::default();
    let loaded = AutoSeq2SeqLm::<Flex<f32, i32>>::load_local(
        &root,
        &AutoLoadOptions {
            repository: Some("google/flan-t5-small".into()),
            revision: Some("0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab".into()),
            ..AutoLoadOptions::default()
        },
        &device,
    )?;
    let tokenizer = loaded.tokenizer().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "the local directory does not contain tokenizer.json",
        )
    })?;
    let source = tokenizer
        .encode(&prompt, &EncodeOptions::default())?
        .to_causal_input::<Flex<f32, i32>>(&device)?;
    let model = match loaded.model() {
        AutoSeq2SeqLm::T5(model) => model,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "automatic seq2seq loading returned a non-T5 model",
            )
            .into());
        }
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
        .and_then(|sequences| sequences.first())
        .ok_or_else(|| io::Error::other("beam search returned no hypothesis"))?;
    println!("{}", tokenizer.decode(&best.token_ids, true)?);
    eprintln!(
        "finish={:?} score={:.6} tokens={}",
        best.finish_reason,
        best.score,
        best.token_ids.len()
    );
    Ok(())
}
