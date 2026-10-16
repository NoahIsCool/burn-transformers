//! Verus harness for the Gemma layer schedule, rotary sets, and namespace detection.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/gemma/plan.rs`:
//!
//! - `sliding_by_pattern`: the overflow-free default local/global rule equals
//!   the reference's `(l + 1) mod p != 0` for every layer and every positive
//!   period;
//! - `default_period`: the period is 2 for Gemma 2 and the configured
//!   `sliding_window_pattern` (6 when absent) for Gemma 3, positive unless a
//!   configured zero (rejected first);
//! - `sliding_schedule`: the whole default schedule applies that rule to
//!   every layer, so the global layers are the 1-based multiples of the
//!   period and there are exactly `⌊L / p⌋` of them;
//! - `rotary_sets` and `rotary_set_count`: every layer rotates (no NoPE
//!   layer) with its type's set, below the number of sets (one when shared,
//!   two otherwise); with shared sets every layer reads set 0, and otherwise
//!   two layers read the same set exactly when they have the same type, so
//!   no layer rotates with the other type's parameters;
//! - `unique_flag`: a checkpoint's namespace is selected exactly when one
//!   namespace's embedding is present, `Absent` and `Ambiguous` otherwise;
//! - `checkpoint_extent` and `projection_extent`: the contract's `u64`
//!   extents are exact, or name the `usize` or `u64` overflow.
//!
//! See `verus/README.md` for the method, toolchain, and command.
//!
//! The planning file is dependency-free, so it is included as a single
//! module. The tensor code driven by these plans (the per-layer-type rotary
//! tables and sliding masks), the string matching that sets the namespace
//! flags, and the mapping of flags to `GemmaLayerType` remain covered by
//! `src/models/gemma/{architecture,checkpoint,model}_tests.rs` and the tiny
//! goldens in `tests/tiny_goldens_gemma.rs`.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/gemma/plan.rs"]
pub mod plan;
