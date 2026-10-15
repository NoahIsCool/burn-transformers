//! Verus harness for the experiment-integration package's pure translation logic.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/models/auto/defaults.rs`: Hugging Face's sampling `top_k` rule for
//!   `generation_config.json`. An explicit zero disables top-k and nothing
//!   else does; otherwise the requested (or default 50) cutoff is clamped to
//!   the vocabulary, and for any nonempty vocabulary the result lies in
//!   `1..=vocabulary`, so a resolved checkpoint default always passes
//!   `GenerationConfig::validate`.
//! - `src/models/custom/mamba2_geometry.rs`: the Mamba-2 head grouping of a
//!   modular specification. A group count exists exactly when the group size
//!   is positive and divides the heads, and it then multiplies back to the
//!   head count, so translation never silently regroups heads.
//!
//! See `verus/README.md` for the method, toolchain, and command. Parsing,
//! tensor construction, and the remaining generation-default semantics are
//! covered by the Rust test suite.

#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/models/auto/defaults.rs"]
pub mod defaults;

#[path = "../src/models/custom/mamba2_geometry.rs"]
pub mod mamba2_geometry;
