//! Verus harness for the cache's pure planning and lifecycle code.
//!
//! Verifies, in place, the specifications attached to:
//!
//! * `src/cache/ring.rs`: standard and mirrored ring write and read planning,
//!   staging plans, sliding-window starts, and staged-commit planning,
//!   including the slot invariant `slot(p) = p mod capacity` for every token
//!   written, mirrored, or read. Mirrored writes and staged commits cover
//!   exactly the retained suffix, and every planner fails exactly when its
//!   overflow or bound condition holds.
//! * `src/cache/lengths.rs`: the length ledger behind every decode state and
//!   speculative transaction. Each transition (advance, begin, commit,
//!   rollback, truncate, reorder, reset) preserves `committed <= working <=
//!   capacity`, computes its documented formula without overflow, and is
//!   rejected exactly when its rule is violated, naming the first offending
//!   row or the blocking transaction state. Proposal lengths are exact.
//! * `src/cache/retention.rs`: ring and recurrent rewind predicates, proved
//!   sound and complete against the ring eviction model (row scans name the
//!   lowest offending row), the working window
//!   that splits a speculative read into committed and staged parts, and the
//!   absolute position held by each ring slot in a batched slot-order read.
//! * `src/cache/rows.rs`: grouping of batch rows into runs that share one
//!   tensor assignment; runs are ordered, disjoint, carry their rows'
//!   parameters, and cover exactly the rows with new tokens.
//! * `src/cache/tensor/plan.rs`: integer planning shared by the tensor
//!   caches. Allocation scalar counts are exact running products checked
//!   against overflow, the portable `i32` element limit, and byte size;
//!   append ends are exact and within capacity; the staged-commit scan
//!   accepts exactly fully staged, in-proposal prefixes and names the first
//!   failure; truncation clears exactly the rows reset from history; beam
//!   sources and host index vectors are checked and converted exactly;
//!   staging lengths stay within the staging capacity.
//! * `src/cache/tensor/recurrent/plan.rs`: recurrent snapshot indexing. The
//!   working and committed snapshot of `n` tokens is `n - 1`, a uniform batch
//!   shares one snapshot exactly when every row staged the same positive
//!   count, and each flat gather index `b * P + max(s_b, 1) - 1` decodes to
//!   row `b` and its own snapshot, failing exactly beyond `i32::MAX`.
//! * `src/cache/tensor/kv/plan.rs`: native KV planning. The committed time
//!   extent is exactly the capacity, `W`, or `2W` and every inconsistent
//!   storage/window configuration is rejected with its first rule; committed
//!   reads keep `min(L, W)` positions; working reads check every end and
//!   split each window exactly as the verified working window does; mirrored
//!   runs read the slot of each matching position below `2W`.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/cache/lengths.rs"]
pub mod lengths;
#[path = "../src/cache/retention.rs"]
pub mod retention;
#[path = "../src/cache/ring.rs"]
pub mod ring;
#[path = "../src/cache/rows.rs"]
pub mod rows;
#[path = "../src/cache/tensor/plan.rs"]
pub mod tensor_plan;
#[path = "../src/cache/tensor/recurrent/plan.rs"]
pub mod recurrent_plan;
#[path = "../src/cache/tensor/kv/plan.rs"]
pub mod kv_plan;

/// Mirrors the crate paths `crate::cache::*` used by the tensor-cache planners.
pub mod cache {
    pub use crate::{lengths, retention, ring, rows};

    /// Mirrors `crate::cache::tensor`.
    pub mod tensor {
        pub use crate::tensor_plan as plan;
    }
}
