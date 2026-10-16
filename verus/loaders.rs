//! Verus harness for the architecture-owned checkpoint loaders.
//!
//! Verifies, in place, the specifications attached to the pure integer cores
//! that the family checkpoint contracts and tensor sinks call:
//!
//! - `src/models/gpt_neox/load/packed.rs`: the packed query/key/value split.
//!   `packed_rows` accepts exactly the geometries the sink can split and
//!   returns `R = 3 q d` and `q d` rows; `packed_payload` returns the row and
//!   payload byte sizes exactly or reports overflow; `packed_component` never
//!   indexes outside the payload and returns each component's bytes in head
//!   order, byte `o` being payload byte `(3 floor(o / b) + c) b + o mod b`,
//!   and `lemma_packed_source_bijection` proves the three components
//!   partition the payload; `packed_staging_peak` is the exact staging peak
//!   `P + max_i floor(B_i / 3)` or reports `u64` overflow.
//! - `src/models/{qwen3_5,qwen3_moe,deepseek_v2,muse_glimmer,nemotron_h,t5}/
//!   checkpoint/dimensions.rs`: every family's checkpoint tensor widths.
//!   `checkpoint_dimensions` copies each plain extent unchanged and computes
//!   each fused width exactly — Qwen3.5's gated query `2 q d` and DeltaNet
//!   convolution channels `2 Kh Kd + Vh Vd`, DeepSeek-V2's latent-attention
//!   widths `H (n + r)`, `c + r`, `H (n + v)`, `H v` and shared expert `m s`,
//!   Nemotron-H's Mamba split `C_conv = I + 2 G N` and fused input
//!   projection `2 I + 2 G N + H`, the MTP fusion width `2 h`, and every
//!   family's query and key/value widths — or names the first term that
//!   overflows, in each contract's original checking order. Qwen3.5's
//!   `query_projection` selects the gated width exactly when the output gate
//!   is fused, and T5's `feed_forward_sublayer` (shared by the contract and
//!   the tensor sink) numbers the feed-forward network after the block's
//!   attention sublayers.
//!
//! See `verus/README.md` for the method, toolchain, and command. String
//! parsing of tensor names, `format!`-built inventories, Burn tensor
//! construction, dtype sizes, and parameter installation stay in the
//! unverified wrappers and are covered by each family's checkpoint and load
//! tests.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/gpt_neox/load/packed.rs"]
pub mod gpt_neox_packed;

#[path = "../src/models/qwen3_5/checkpoint/dimensions.rs"]
pub mod qwen3_5_dimensions;

#[path = "../src/models/qwen3_moe/checkpoint/dimensions.rs"]
pub mod qwen3_moe_dimensions;

#[path = "../src/models/muse_glimmer/checkpoint/dimensions.rs"]
pub mod muse_glimmer_dimensions;

#[path = "../src/models/deepseek_v2/checkpoint/dimensions.rs"]
pub mod deepseek_v2_dimensions;

#[path = "../src/models/nemotron_h/checkpoint/dimensions.rs"]
pub mod nemotron_h_dimensions;

#[path = "../src/models/t5/checkpoint/dimensions.rs"]
pub mod t5_dimensions;
