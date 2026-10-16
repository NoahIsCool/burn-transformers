//! Verus harness for the transfer plan's pure precedence and tie bookkeeping.
//!
//! Verifies, in place, the specifications attached to `src/transfer/plan.rs`:
//!
//! - `select_rule` resolves a target by the documented precedence exactly:
//!   kept targets are never selected, then the first exact rule wins, then a
//!   unique prefix or matching rule; several fallbacks without an exact rule
//!   are reported as overlapping, so rule order never silently chooses
//!   between overlapping prefixes.
//! - `target_tie` accepts a pre-existing target tie exactly when it is
//!   untouched, explicitly untied, or selected with one shared source
//!   identity, transformation, and trainability (compared by
//!   `same_selection`).
//! - `first_fanned_out_path` returns exactly the smallest source path reaching
//!   two target tie classes, so acceptance (`None`) means the
//!   source-path-to-class map is a function.
//! - `class_leaders` gives every entry of a fan-out group the identity of the
//!   first member of its target tie class, so two entries share an identity
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
