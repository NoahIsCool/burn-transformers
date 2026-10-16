//! Verus harness for the integer planning of the recurrent mixers.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/delta_net/plan.rs`: Gated DeltaNet prefill planning. The chunk
//!   width and count cover the tokens with less than one chunk of padding
//!   and no overflow; balanced chunk blocks are consecutive, start at zero,
//!   end at the chunk count, hold between one and the limit of chunks, and
//!   follow the exact balanced size formula; the doubling-inverse schedule
//!   starts at the least power of two not below the chunk width and halves
//!   exactly down to two; the fused convolution width is exact or rejected
//!   exactly on overflow; integer configuration checks accept exactly the
//!   admissible geometry; and the recurrence-versus-chunked choice equals its
//!   closed-form cost comparison and is monotone.
//! - `src/mamba2/plan.rs`: Mamba-2 bookkeeping. Projection and convolution
//!   widths are exact or name the first overflowing step in validation
//!   order; geometry and row-count checks accept exactly the admissible
//!   inputs; right-padding flags, row-end history columns, per-token window
//!   columns, and batch-row selectors follow their closed forms or are
//!   rejected exactly when an index exceeds `i64`; the mode grouping holds
//!   every row once, in ascending order, by effective mode (inactive rows
//!   take the first active row's mode), with a restore permutation that is a
//!   two-sided inverse of the grouped order; and the SSD scan blocks hold
//!   `clamp(budget / plane, 1, chunks)` chunks and tile the tokens in
//!   chunk-aligned blocks with only the last padded, by less than one chunk.
//!
//! Not proved here: the Burn tensor algebra (chunk-major rearrangement, the
//! triangular inverse itself, causal convolutions, token compaction, the SSD
//! segment sums, and the recurrences) and floating-point numerics; the mixer
//! unit tests cover those. Loop invariants are attributes on `while`
//! statements, which needs `proc_macro_hygiene` under the Verus compiler only.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/delta_net/plan.rs"]
pub mod delta_net_plan;

#[path = "../src/mamba2/plan.rs"]
pub mod mamba2_plan;

/// Mirrors the crate path `crate::delta_net::plan` used by the Mamba-2 plan.
pub mod delta_net {
    pub use crate::delta_net_plan as plan;
}
