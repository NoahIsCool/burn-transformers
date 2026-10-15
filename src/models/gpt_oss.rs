//! OpenAI gpt-oss: a sparse mixture-of-experts decoder with attention sinks.
//!
//! gpt-oss-20b and gpt-oss-120b (`model_type: "gpt_oss"`) are pre-normalized
//! grouped-query decoders whose every block routes tokens to a few of many
//! experts:
//!
//! ```text
//! h_0     = Embed(tokens)                                        [batch, token, d]
//! h_{l+1} = h' + MoE(N₂(h')),   h' = h_l + Attn_l(N₁(h_l))
//! logits  = RMSNorm(h_L) · Wᵀ
//! ```
//!
//! Its distinctive mechanisms all come from shared components:
//!
//! | mechanism | shared component |
//! | --- | --- |
//! | learned per-head attention sinks, biased Q/K/V/O | [`GroupedQueryAttention`](crate::attention::GroupedQueryAttention) |
//! | alternating 128-token sliding / full attention (`layer_types`) | [`AttentionConnectivity`](crate::attention::AttentionConnectivity), ring KV groups |
//! | YaRN RoPE with `truncate: false` | the shared HF RoPE translator |
//! | biased router, top-k on logits, softmax over the selected logits | [`RouterConfig`](crate::moe::RouterConfig) (`ScoringOrder::TopKThenScore`) |
//! | clamped SwiGLU experts `(clamp(u, ±7) + 1) · g σ(1.702 g)` with biases | [`GatedFeedForward::forward_clamped`](crate::feed_forward::GatedFeedForward::forward_clamped), [`ClampedSwiGlu`](crate::feed_forward::ClampedSwiGlu) |
//! | MXFP4 expert weights | exact load-time decoding ([`MappingPlan::dequantize_mxfp4`](crate::checkpoint::MappingPlan::dequantize_mxfp4)) |
//!
//! # Pinned reference
//!
//! Every translation rule, default, and numerical convention was read from
//! Hugging Face Transformers **5.2.0** (PyTorch 2.10.0):
//! `models/gpt_oss/{configuration,modeling}_gpt_oss.py`,
//! `integrations/mxfp4.py`, `modeling_rope_utils.py`, and `masking_utils.py`.
//! The committed tiny golden `tests/fixtures/tiny_goldens/gpt-oss` was
//! generated with that pair by `scripts/generate_tiny_goldens.py`; the MXFP4
//! goldens in `src/checkpoint/mxfp4_goldens.json` cover the released
//! `openai/gpt-oss-20b` at revision
//! `6cee5e81ee83917806bbde320786a8fb61efebee`.
//!
//! # Module map
//!
//! * [`config`](GptOssConfig): the serialized `config.json` schema with the
//!   reference defaults;
//! * [`architecture`](GptOssArchitecture): the translation table from a
//!   config to executable choices, and the single place of validation;
//! * `plan`: Verus-verified integer planning (the default layer schedule,
//!   cache-group numbering, interleaved expert columns);
//! * `checkpoint` / `load`: the strict tensor contract for dense and MXFP4
//!   checkpoints and the streamed installation that splits stacked experts;
//! * [`model`]: the graph, stateless and hooked passes, staged execution,
//!   and genuinely batched cached prefill/decode with ring caches for the
//!   sliding layers.
//!
//! # Support status
//!
//! Implemented and validated on Flex/F32 against the pinned reference on a
//! tiny random-weight checkpoint (full-vocabulary logits, per-layer hidden
//! states, stateless/cached/batched agreement, greedy continuation), and the
//! MXFP4 decoder bit for bit on real gpt-oss-20b expert tensors. The
//! real-checkpoint logit gate `tests/gpt_oss_artifacts.rs` is opt-in;
//! Metal/F16 acceptance of gpt-oss-20b is a separate gate.

mod architecture;
mod checkpoint;
mod config;
mod load;
pub mod model;
mod plan;

pub use architecture::{GptOssArchitecture, GptOssWeightFormat};
pub use checkpoint::{GptOssCheckpointContract, GptOssCheckpointOptions};
pub use config::GptOssConfig;
pub use load::{GptOssCheckpointLoadError, GptOssLoadError};
pub use model::{
    GptOssDecodeState, GptOssDecoderLayer, GptOssForCausalLm, GptOssModel, GptOssSparseMlp,
    gpt_oss_load_balancing_loss,
};
