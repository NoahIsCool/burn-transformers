//! Verus harness for mixture-of-experts dispatch bookkeeping and geometry.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/moe/assignment.rs`: the stable counting sort that groups routed
//!   `(token, slot)` assignments by expert. The proof covers overflow freedom
//!   and in-bounds indexing, precise reporting of the first out-of-range
//!   expert ID, and, for a successful plan, offsets that partition all
//!   assignments by per-expert counts with every range holding only that
//!   expert's assignments in strictly increasing order, plus the explicit
//!   inverse: assignment `a` sits at its expert's offset plus the number of
//!   earlier assignments of that expert, so the order is a bijection. The
//!   memory-bounding block planner makes every dispatch block a non-empty run
//!   of whole experts whose assignment count fits the row budget (or a single
//!   expert that alone exceeds it) and is maximal; the block boundaries
//!   partition all experts; per-expert counts and block-local row slices are
//!   exact and never pass their block's end. The host copy of routed IDs is
//!   rejected for a wrong length or reports its first non-finite token, and
//!   otherwise flattens to exactly the ID columns in token-major order; the
//!   assignment count is exact and addressable by `i64` indices.
//! - `src/moe/geometry.rs`: expert-bank, router-cardinality, and
//!   group-limited routing checks accept exactly the admissible integer
//!   configurations and name the first violated condition; equal groups tile
//!   the experts and the eligible count `T * (E / G)` cannot overflow; the
//!   load-balancing expert-ID range check is exact.
//!
//! Not proved here: the device gathers, expert evaluation, and scatter-add
//! that consume the plan, or the router kernels (scores, group pruning, and
//! the deterministic top-k tie rule) that produce expert IDs; the MoE unit
//! tests cover those. Loop invariants are attributes on `while` statements,
//! which needs `proc_macro_hygiene` under the Verus compiler only.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/moe/assignment.rs"]
pub mod assignment;

#[path = "../src/moe/geometry.rs"]
pub mod geometry;
