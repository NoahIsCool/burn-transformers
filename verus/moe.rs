//! Verus harness for mixture-of-experts dispatch bookkeeping.
//!
//! Verifies, in place, the specifications attached to `src/moe/assignment.rs`:
//! the stable counting sort that groups routed `(token, slot)` assignments by
//! expert. The proof covers overflow freedom and in-bounds indexing, precise
//! reporting of an out-of-range expert ID, and, for a successful plan,
//! offsets that partition all assignments by per-expert counts with every
//! range holding only that expert's assignments in strictly increasing order.
//!
//! Not proved here: the device gathers, expert evaluation, and scatter-add
//! that consume the plan, or the router kernels that produce expert IDs; the
//! MoE unit tests cover those. Loop invariants are attributes on `while`
//! statements, which needs `proc_macro_hygiene` under the Verus compiler only.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
#![feature(proc_macro_hygiene)]

#[path = "../src/moe/assignment.rs"]
pub mod assignment;
