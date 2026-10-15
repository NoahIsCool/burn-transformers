//! Model-specific transformer topology.
//!
//! A model module describes only the choices that affect tensor shapes or forward
//! semantics. Runtime policy (devices, kernels, quantization, and batching) belongs
//! outside these schemas. Shared serialized concepts such as RoPE live in
//! [`config`], while each architecture validates its own stronger invariants.

mod auto;
pub mod config;
pub mod custom;
pub mod deepseek_v2;
pub mod gemma;
pub mod gpt2;
pub mod gpt_neox;
pub mod llama;
pub mod muse_glimmer;
pub mod nemotron_h;
pub mod qwen3_5;
pub mod qwen3_moe;
mod shared;
pub mod t5;

pub use auto::{
    AutoCausalConfig, AutoCausalConfigRef, AutoCausalDecodeState, AutoCausalLm, AutoLoadOptions,
    AutoSeq2SeqConfig, AutoSeq2SeqConfigRef, AutoSeq2SeqDecodeState, AutoSeq2SeqLm,
};

/// Qwen3.8 checkpoints intentionally retain Qwen3.5's `qwen3_5_text` model type.
///
/// Keeping this as a type alias prevents an artificial architecture fork: the
/// released 27B checkpoint differs in learned parameters and metadata, not in the
/// Rust topology schema needed to wire its text transformer.
pub type Qwen38TextConfig = qwen3_5::Qwen3_5TextConfig;
