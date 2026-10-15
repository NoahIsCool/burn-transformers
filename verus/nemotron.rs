//! Verus harness for Nemotron-H's batched-execution planning.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/nemotron_h/model/plan.rs`: the compact-batch maps that
//! gather, scatter, pack, unpack, and select the last token of ragged rows
//! (exact lengths and in-range addresses, including the appended zero rows),
//! absolute query positions, the reference Mamba-2 path choice, the
//! transactional recurrent-staging policy, the shifted MTP cursor after a
//! truncation, and the one-pass verification length. The planning file
//! reuses the shared strictly-increasing check, so the shared planning module
//! is included at its crate path. See `verus/README.md` for the method,
//! toolchain, and command.
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
