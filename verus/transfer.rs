//! Verus harness for the transfer plan's pure precedence and tie bookkeeping.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/transfer/plan.rs`: target resolution follows the documented
//!   precedence exactly (kept targets are never selected, then the first
//!   exact rule, then a unique prefix or matching rule; several fallbacks are
//!   reported as overlapping). A pre-existing target tie is accepted exactly
//!   when it is untouched, explicitly untied, or selected with one shared
//!   source identity, transformation, and trainability. The fan-out check
//!   returns exactly the smallest source path reaching two target tie
//!   classes, so acceptance means the source-path-to-class map is a function.
//!   Within a fan-out group, every entry takes the identity of the first
//!   member of its target tie class, and two entries share an identity
//!   exactly when they share a class.
//!
//! See `verus/README.md` for the method, toolchain, and command. Path string
//! matching, parameter discovery, tensor staging and installation, value
//! equality of tied sources, and error formatting remain covered by the Rust
//! test suite.

// Loop invariants are attributes on `while` expressions, which needs this
// feature under Verus only; ordinary builds never compile this harness.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/transfer/plan.rs"]
pub mod plan;
