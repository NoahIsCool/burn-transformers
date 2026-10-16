//! Verus harness for the integer cores of model-configuration validation.
//!
//! Verifies, in place, the specifications attached to
//! `src/models/config/plan.rs`:
//!
//! * `mrope_section_fault`: multimodal RoPE sections are accepted exactly when
//!   they are non-empty, positive, and `2 * sum(sections)` equals the rotary
//!   width (an overflowing sum or doubling is a mismatch); empty or zero
//!   sections are reported before a width mismatch;
//! * `divisibility_fault`: grouped heads are accepted exactly when both counts
//!   are non-zero and the remainder is zero, with the exact quotient
//!   `numerator = groups * denominator` and `groups >= 1`; faults follow the
//!   order zero numerator, zero denominator, remainder;
//! * `token_id_fault`: a token-ID field is accepted exactly when it is
//!   non-empty and every ID is below the vocabulary size, emptiness being
//!   reported before an out-of-range ID;
//! * `original_context_fits`: exactly `0 < original <= max_position_embeddings`;
//! * `resolve_original_context`: the trained RoPE context follows the
//!   precedence table (`spec_resolved_original`), is positive, and every
//!   rejection (nested zero, per-layer or flat conflict under the reject
//!   policy, resolved zero) holds exactly in check order with its values;
//! * `infinity_token_starts` (with `infinity_at` and the one-byte automaton
//!   step `lexer_advance`, which equals `lexer_step`): the legacy `Infinity`
//!   lexer lists exactly the token starts outside JSON strings, as defined by
//!   the string/escape automaton `lexer_state` (sound: each listed offset
//!   starts a token outside a string; complete: every such start is listed),
//!   in increasing order and at least eight bytes apart, so the caller's
//!   slices are ordered and non-overlapping.
//!
//! Not proved here: serde decoding, floating-point checks (theta, factors,
//! partial rotary products), the string-keyed per-layer-type lookups, and the
//! `ConfigError`/`Error` construction in the callers, which the unit tests in
//! `src/models/config_tests.rs` and `src/models/config/*_tests.rs` cover.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/config/plan.rs"]
pub mod plan;
