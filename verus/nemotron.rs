//! Verus harness for Nemotron-H's batched-execution planning.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/nemotron_h/model/plan.rs`: the compact-batch maps that
//! gather, scatter, pack, unpack, and select the last token of ragged rows
//! (every entry equals its documented equation, the compact width is the
//! largest row count, the scatter is the inverse of the gather on valid
//! columns, and a plan is refused exactly for malformed columns or
//! unrepresentable extents), the exact absolute query position of every
//! compact slot (padded slots repeat a row's last readable position, so no
//! query of a row with history loses every key), the reference Mamba-2 path
//! choice, the transactional recurrent-staging policy, the shifted MTP cursor
//! after a truncation (per row and for a whole batch), the one-pass
//! verification length, the hybrid schedule's cache slots (each kind's
//! blocks are numbered exactly `0..count` in schedule order, so attention
//! and Mamba-2 blocks address distinct allocated layers of their own caches),
//! the sliding-window activity test, uniform-history detection, the
//! context-limit check, and the Mamba-2 convolution width. The planning
//! file reuses the shared strictly-increasing check, so the shared planning
//! module is included at its crate path. See `verus/README.md` for the
//! method, toolchain, and command.
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
