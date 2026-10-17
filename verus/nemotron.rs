//! Verus harness for Nemotron-H's batched-execution planning.
//!
//! Verifies, in place, the specifications attached to these files.
//!
//! `src/models/nemotron_h/model/plan.rs`:
//!
//! - `plan_compaction`: every entry of the compact-batch maps that gather,
//!   scatter, pack, unpack, and select the last token of ragged rows equals
//!   its documented equation. The compact width is the largest row count,
//!   the scatter inverts the gather on valid columns, and a plan is refused
//!   exactly for malformed columns or unrepresentable extents.
//! - `CompactPlan` accessors: `batch` and `tokens` return `B` and `T`, and
//!   `is_dense` (`T = B * N`) and `is_identity` (`N = S`, all valid) are
//!   exactly the premises under which the plan's maps are proved to be
//!   identities, so the tensor wrappers skip only identity gathers.
//! - `slot_position` and `query_positions`: the exact absolute query
//!   position of every compact slot. Padded slots repeat a row's last
//!   readable position, so no query of a row with history loses every key,
//!   and the refusal rule holds in both directions.
//! - `mamba_step`, `transactional_mamba`, `mtp_length_after_truncate`,
//!   `mtp_lengths_after_truncate`, and `verification_tokens`: the reference
//!   Mamba-2 path, the recurrent-staging policy (naming the first row that
//!   lacks state), the shifted MTP cursor after a truncation, and the
//!   one-pass verification length.
//! - `plan_cache_slots`: each kind's blocks are numbered exactly `0..count`
//!   in schedule order, so attention and Mamba-2 blocks address distinct
//!   allocated layers of their own caches.
//! - `longest_end`, `active_window`, `all_equal`, `exceeds_limit`, and
//!   `convolution_channels`: the sliding-window activity test, uniform
//!   history, the context limit, and the Mamba-2 convolution width.
//!
//! `src/models/nemotron_h/model/cycle.rs`:
//!
//! - `bounded_end`: every append of a speculative cycle (target suffix, MTP
//!   pairs, MTP positions, staging reservation) is accepted exactly when
//!   `start + count <= limit`, and each rejection names its cause.
//! - `shifted_alignment`: the alignment `L_target = L_mtp + 1` is checked
//!   exactly.
//! - `trailing_positions`: an MTP step's last `q` queries sit exactly at
//!   `end - q, …, end - 1`, refused exactly when one exceeds `i64::MAX`.
//!
//! `src/models/nemotron_h/routing.rs`:
//!
//! - `routing_groups`: the group-limited routing check accepts exactly the
//!   feasible configurations, reports the eligible expert count, never
//!   divides by zero, and (for evenly divided experts) never lets a token
//!   request more experts than exist.
//!
//! The planning file reuses the shared strictly-increasing check, so the
//! shared planning module `src/models/shared/plan.rs` is included at its
//! crate path and re-verified here; its own functions are documented in
//! `verus/models.rs`. See `verus/README.md` for the method, toolchain, and
//! command.
//!
//! Burn tensor programs driven by these plans (index selects, cache steps,
//! and attention masks) remain covered by the Rust tests in
//! `src/models/nemotron_h/model/*_tests.rs`.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/shared/plan.rs"]
pub mod shared_plan;

/// Mirrors the crate path `crate::models::shared::plan` used by the Nemotron-H plans.
pub mod models {
    /// Model-independent family glue.
    pub mod shared {
        pub use crate::shared_plan as plan;
    }
}

#[path = "../src/models/nemotron_h/model/plan.rs"]
pub mod plan;

#[path = "../src/models/nemotron_h/model/cycle.rs"]
pub mod cycle;

#[path = "../src/models/nemotron_h/routing.rs"]
pub mod routing;
