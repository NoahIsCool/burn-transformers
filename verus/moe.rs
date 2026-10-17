//! Verus harness for mixture-of-experts dispatch bookkeeping and geometry.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/moe/assignment.rs`: the stable counting sort that groups routed
//!   `(token, slot)` assignments by expert. `plan_expert_assignments` (with
//!   `scatter_by_expert` and `set_entry`) is overflow-free, indexes in bounds,
//!   reports exactly the first out-of-range expert ID, and, for a successful
//!   plan, returns offsets that partition all assignments by per-expert
//!   counts with every range holding only that expert's assignments in
//!   strictly increasing order, plus the explicit inverse: assignment `a`
//!   sits at its expert's offset plus the number of earlier assignments of
//!   that expert, so the order is a bijection. The memory-bounding block
//!   planner (`next_dispatch_block`, `dispatch_blocks`) makes every dispatch
//!   block a non-empty run of whole experts whose assignment count fits the
//!   row budget (or a single expert that alone exceeds it) and is maximal;
//!   the block boundaries partition all experts. Per-expert counts
//!   (`range_lengths`) and block-local row slices (`block_expert_rows`) are
//!   exact and never pass their block's end, as are each block's slice of the
//!   order (`block_row_range`); the order and slot count convert to `i64`
//!   indices exactly (`signed_order`, `signed_index`). The host copy of routed
//!   IDs (`routed_ids`) is rejected for a wrong length or reports its first
//!   non-finite token, and otherwise flattens to exactly the ID columns in
//!   token-major order; the assignment count (`routed_assignment_count`) is
//!   exactly `tokens * slots` when addressable by `i64` indices.
//! - `src/moe/geometry.rs`: `check_expert_bank`, `check_router_cardinality`,
//!   and `check_group_geometry` accept exactly the admissible integer
//!   configurations and name the first violated condition; equal groups tile
//!   the experts (`lemma_groups_tile`) and the eligible count `T * (E / G)`
//!   cannot overflow; `group_widths` reshapes the experts into exactly `G`
//!   groups of `E / G` and its top-two group score ranks `min(2, E / G)`
//!   experts, at most one group's; `expert_ids_within` is the exact
//!   load-balancing expert-ID range check.
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
