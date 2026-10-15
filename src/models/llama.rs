//! The Llama family: one dense decoder for the most-used open architectures.
//!
//! Llama 2/3.x, Mistral, Qwen2/2.5, Qwen3 dense, SmolLM3, and OLMo 2 are the
//! same pre-normalized (or, for OLMo 2, post-normalized) grouped-query decoder
//! with a SwiGLU MLP, RMSNorm, and rotary positions. They differ only in a
//! handful of flags, so this module implements them once:
//!
//! ```text
//! h_0     = Embed(tokens)                                   [batch, token, d]
//! h_{l+1} = Block_l(h_l),   l = 0 .. L-1
//! logits  = RMSNorm(h_L) · Wᵀ          W = lm_head, or Embed when tied
//! ```
//!
//! | `model_type` | distinctive mechanism | reference |
//! | --- | --- | --- |
//! | `llama` | `llama3` RoPE scaling, optional attention/MLP bias, tied head | `models/llama/modeling_llama.py` |
//! | `mistral` | sliding window on every layer | `models/mistral/modeling_mistral.py` |
//! | `qwen2` | bias on the query/key/value projections only | `models/qwen2/modeling_qwen2.py` |
//! | `qwen3` | per-head query/key RMSNorm, explicit `head_dim` | `models/qwen3/modeling_qwen3.py` |
//! | `smollm3` | periodic NoPE layers (`no_rope_layers`) | `models/smollm3/modeling_smollm3.py` |
//! | `olmo2` | post-sublayer RMSNorm, full-width query/key RMSNorm | `models/olmo2/modeling_olmo2.py` |
//!
//! # Pinned reference
//!
//! Every translation rule, default, and numerical convention was read from
//! Hugging Face Transformers **5.2.0** (the PyPI release, with PyTorch 2.10.0):
//! the `configuration_*.py` and `modeling_*.py` files named above plus
//! `modeling_rope_utils.py`, `masking_utils.py`, and
//! `utils/output_capturing.py`. The committed tiny goldens
//! (`tests/fixtures/tiny_goldens/{llama3,llama-bias-tied,qwen2,qwen3,mistral-sliding,smollm3,olmo2}`)
//! were generated with that exact pair by `scripts/generate_tiny_goldens.py`.
//!
//! # Module map
//!
//! * [`config`](LlamaFamilyConfig): the serialized `config.json` schema with
//!   per-`model_type` defaults;
//! * [`architecture`](LlamaArchitecture): the documented translation table
//!   from a config to executable flags, and the single place of validation;
//! * `plan`: Verus-verified integer planning (NoPE and sliding schedules,
//!   cache-group numbering, batched-step slot positions and gathers);
//! * `checkpoint` / `load`: the strict tensor contract (every name listed)
//!   and the streamed parameter installation;
//! * [`model`]: the graph, stateless and hooked forward passes, staged
//!   execution, and genuinely batched cached prefill/decode. These execution
//!   paths are written once against a crate-private dense-decoder interface
//!   (`model::graph`), so other dense decoder families can share the engine.
//!
//! # Support status
//!
//! Implemented and validated against the pinned reference on tiny
//! random-weight checkpoints of all six types (full-vocabulary logits,
//! per-layer hidden states, cached/uncached agreement, batched ragged decode,
//! greedy continuations), and on Flex/F32 against locally generated
//! references for the real checkpoints recorded in the test
//! `tests/llama_family_artifacts.rs`. Real-checkpoint Metal/F16 acceptance is
//! a separate gate.

mod architecture;
mod checkpoint;
mod config;
mod load;
pub mod model;
mod plan;

pub use architecture::{
    LlamaArchitecture, LlamaLayerSchedule, NormRounding, RotaryPrecision, SublayerNorm,
};
pub use checkpoint::{LlamaCheckpointContract, LlamaCheckpointOptions};
pub use config::{LlamaFamilyConfig, LlamaVariant};
pub use load::{LlamaCheckpointLoadError, LlamaLoadError};
pub use model::{LlamaDecodeState, LlamaDecoderLayer, LlamaForCausalLm, LlamaModel};
