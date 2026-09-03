#![deny(missing_docs)]
#![forbid(unsafe_code)]

//! Backend-generic implementations of modern transformer architectures.
//!
//! The crate separates reusable transformer mathematics from checkpoint-family
//! wiring. Models are generic over a [`burn::tensor::backend::Backend`], while
//! autoregressive caches remain external mutable state. This split preserves a
//! stateless, differentiable forward graph for future training without forcing
//! inference sessions into the serialized parameter tree.
//!
//! # Tensor conventions
//!
//! Unless a type documents a more specific layout, token tensors use
//! `[batch, sequence]`, hidden tensors use `[batch, sequence, hidden]`, and
//! attention tensors use `[batch, heads, query, key]`. Boolean token masks use
//! `true` for a real token and `false` for padding.

pub mod architecture;
pub mod attention;
pub mod cache;
pub mod checkpoint;
pub mod delta_net;
pub mod error;
pub mod feed_forward;
pub mod generation;
pub mod loaded;
pub mod models;
pub mod moe;
pub mod normalization;
pub mod operations;
pub mod position;
#[cfg(feature = "tokenizers")]
pub mod tokenization;

pub use architecture::{
    ArchitectureConfig, CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput,
    DecodeState, EncoderOutput, ForwardOptions, LogitsSelection, Seq2SeqInput,
    Seq2SeqLanguageModel, Seq2SeqOutput, TokenMask,
};
pub use error::{Error, Result};
pub use generation::{
    CausalGenerationOutput, CausalGenerationRequest, FinishReason, GeneratedSequence,
    GenerationConfig, GenerationToken, GenerationTracker, LogitsProcessor, SamplingStrategy,
    TokenSelection, generate_causal, generate_causal_streaming,
};
pub use loaded::{LoadedModel, ModelSourceMetadata, ModelTokenizer};
pub use models::{
    AutoCausalConfig, AutoCausalLm, AutoLoadOptions, AutoSeq2SeqConfig, AutoSeq2SeqLm,
};
#[cfg(feature = "tokenizers")]
pub use tokenization::{
    BatchEncoding, ChatMessage, ChatTemplateOptions, EncodeOptions, LocalTokenizer, Padding,
    PaddingSide, SpecialToken, SpecialTokens, TokenizerMetadata, TruncationSide,
};
