//! Verus harness for the T5 and GPT model families' pure planning logic.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/t5/model/packing.rs`, the host plans behind batched T5
//! stateful decoding:
//!
//! - `compaction_sources`: the flat gather that packs every row's valid tokens
//!   right-aligned into a `[batch, Q]` block never overflows, has exactly
//!   `batch * Q` entries, and addresses only elements of the `[batch, S]`
//!   input.
//! - `scatter_sources`: the flat gather that restores the `[batch, S]` layout
//!   never overflows, has exactly `batch * S` entries, and (for a nonempty
//!   block) addresses only elements of the `[batch, Q]` block.
//! - `groups_preserved`: exactly characterizes reorders that keep every row
//!   inside its group of `group_size` consecutive rows, which is what lets
//!   beam search leave T5's encoder binding and cross-attention keys in place.
//!
//! See `verus/README.md` for the method, toolchain, and command.
//!
//! The planning file is dependency-free, so it is included as a single module.
//! The Burn tensor programs driven by these plans (token packing, state
//! unpacking, write-ahead cache appends, masks, and the relative bias) remain
//! covered by the Rust tests in `src/models/t5/model/*_tests.rs`, including
//! parity against the per-row reference algorithm and the stateless forward.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/t5/model/packing.rs"]
pub mod packing;
