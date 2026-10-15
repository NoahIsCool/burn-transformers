//! The Gemma family: Gemma 2 and the Gemma 3 text decoder.
//!
//! Gemma 2 and Gemma 3 are grouped-query decoders with *sandwich* norms
//! (each sublayer's input and output are RMS-normalized), zero-centered
//! `(1 + w)` RMSNorm computed in F32, a `sqrt(d)`-scaled embedding, GeGLU
//! feed-forwards, tied embeddings, and an interleaving of sliding-window
//! (local) and global attention layers:
//!
//! ```text
//! h_0     = Embed(tokens) · round_dtype(sqrt(d))                [batch, token, d]
//! x'      = x + N_post_attn(Attn(N_in(x)))
//! h_{l+1} = x' + N_post_ffw(GeGLU(N_pre_ffw(x')))
//! logits  = softcap_final(N_final(h_L) · Eᵀ)
//! ```
//!
//! | `model_type` | distinctive mechanism | reference |
//! | --- | --- | --- |
//! | `gemma2` | attention-logit and final-logit soft-capping, alternating local/global layers | `models/gemma2/modeling_gemma2.py` |
//! | `gemma3_text` | per-head Q/K RMSNorm, separate local/global RoPE bases with scaling on global layers, 5:1 local/global schedule | `models/gemma3/modeling_gemma3.py` |
//!
//! A Gemma 3 *multimodal* checkpoint (`model_type: "gemma3"`,
//! `Gemma3ForConditionalGeneration`) is accepted by reading its
//! `text_config` and its language-model tensors. Its SigLIP vision tower
//! and multimodal projector are **not executed**: their tensors are
//! explicitly exempted from strict loading and listed in the load report's
//! `allowlisted_ignored`, so image inputs are unsupported while text
//! generation is exactly the text tower's.
//!
//! # Pinned reference
//!
//! Every translation rule, default, and numerical convention was read from
//! Hugging Face Transformers **5.2.0** (the PyPI release, with PyTorch
//! 2.10.0): `models/gemma2/{configuration,modeling}_gemma2.py`,
//! `models/gemma3/{configuration,modeling}_gemma3.py`,
//! `modeling_rope_utils.py`, `masking_utils.py`, and
//! `utils/output_capturing.py`. The committed tiny goldens
//! (`tests/fixtures/tiny_goldens/{gemma2,gemma3-text}`) were generated with
//! that exact pair by `scripts/generate_tiny_goldens.py`.
//!
//! # Module map
//!
//! * [`config`](GemmaConfig): the serialized `config.json` schema with
//!   per-`model_type` defaults;
//! * [`architecture`](GemmaArchitecture): the documented translation table
//!   from a config to executable flags, and the single place of validation;
//! * `plan`: Verus-verified integer planning (the default local/global
//!   schedule and the per-layer rotary sets);
//! * `checkpoint` / `load`: the strict tensor contract over the three
//!   published namespaces and the streamed parameter installation;
//! * [`model`]: the sandwich-normalized block and the graph; the stateless,
//!   hooked, staged, and genuinely batched cached passes are the Llama
//!   family's shared dense-decoder engine.
//!
//! # Support status
//!
//! Implemented and validated against the pinned reference on tiny
//! random-weight checkpoints of both types (`tests/tiny_goldens_gemma.rs`:
//! full-vocabulary logits at every valid position of an unpadded and a
//! left-padded row, per-layer hidden states, cached/uncached and
//! batched-ragged agreement, greedy continuations; largest logit gap
//! `1.5e-6`). Real-checkpoint acceptance is a separate gate.

mod architecture;
mod checkpoint;
mod config;
mod load;
pub mod model;
mod plan;

pub use architecture::{GemmaArchitecture, GemmaLayerSchedule, GemmaLayerType, MAX_LAYERS};
pub use checkpoint::{GemmaCheckpointContract, GemmaCheckpointOptions, GemmaCheckpointSource};
pub use config::{GemmaConfig, GemmaVariant};
pub use load::{GemmaCheckpointLoadError, GemmaLoadError};
pub use model::{GemmaDecodeState, GemmaDecoderLayer, GemmaForCausalLm, GemmaModel};
