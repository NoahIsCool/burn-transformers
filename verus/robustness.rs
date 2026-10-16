//! Verus harness for the chat-template sandbox's size arithmetic.
//!
//! Verifies, in place, the specifications attached to
//! `src/tokenization/chat_template/limits.rs`:
//!
//! - `output_limit` and `total_budget` are at least their floors, exact below
//!   saturation, and `usize::MAX` above it, without overflowing;
//! - `charged_total` succeeds exactly when `produced + bytes <= total` and
//!   then returns that sum, so the render budget's counter never exceeds its
//!   budget and never wraps;
//! - `repeated_size` returns the exact product `length * count` or `None` on
//!   overflow, so `'ab' * 2^63` cannot pass as a small string;
//! - `replacement_bound` and `indentation_bound` never overflow and are at
//!   least the input length (the indentation bound is exact below
//!   saturation).
//!
//! That these bounds dominate the text MiniJinja's `replace` and `indent`
//! produce depends on string semantics and is covered by the Rust tests (an
//! example table and a property test against `str::replace`). The operator
//! rewrite, the bounded filters, and MiniJinja itself are outside the proof.
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]

#[path = "../src/tokenization/chat_template/limits.rs"]
pub mod limits;
