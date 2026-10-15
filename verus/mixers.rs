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
//!
//! Not proved here: the Burn tensor algebra (chunk-major rearrangement, the
//! triangular inverse itself, causal convolutions, and the recurrences) and
//! floating-point numerics; the mixer unit tests cover those. Loop invariants
//! are attributes on `while` statements, which needs `proc_macro_hygiene`
//! under the Verus compiler only.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
#![feature(proc_macro_hygiene)]

#[path = "../src/delta_net/plan.rs"]
pub mod delta_net_plan;
