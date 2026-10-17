//! Verus harness for the chat-template sandbox's size and budget arithmetic.
//!
//! Verifies, in place, the specifications attached to
//! `src/tokenization/chat_template/limits.rs`. Every result below is exact and
//! overflow-free; "saturated" means clamped to `usize::MAX` (or `u64::MAX`),
//! a bound that then fails every limit check.
//!
//! - `output_limit` is `1 MiB + 4 * context_bytes` and `total_budget` is
//!   `16 * output_bytes`, each saturated and at least its floor (the 1 MiB
//!   floor, or the output limit itself);
//! - `render_limits` derives every per-render limit from those formulas: the
//!   per-value limit equals the output limit, the sequence-item limit is the
//!   output limit raised to at least `2^20`, and the cumulative budget is at
//!   least the per-value limit, so a fresh render can always produce one
//!   maximal value;
//! - `charged_total` succeeds exactly when `produced + bytes <= total` and then
//!   returns that sum, so the render budget's counter never exceeds its
//!   budget and never wraps;
//! - `text_allowance` is `min(value_bytes, total_bytes - produced)` (zero once
//!   the counter is past its budget), and while the counter is within its
//!   budget, `bytes <= allowance` holds exactly when a text reservation of
//!   `bytes` succeeds (`text_reservable`), so an allowance-sized buffer never
//!   builds text the following reservation refuses, and vice versa;
//! - `repeated_size` returns the exact product `length * count` or `None` on
//!   overflow, so `'ab' * 2^63` cannot pass as a small string, and
//!   `copy_charge` is the saturated product `items * values * item_bytes`, so
//!   a charge never undercounts what it copies;
//! - `replacement_bound` is exactly the saturated
//!   `length + matches * max(to - from, 0)`, where `matches` is the possible
//!   match count (`characters + 1` for an empty pattern, `length / from`
//!   otherwise) lowered to the requested count, and `indentation_bound` is
//!   exactly the saturated `lines * (width + 1) + length` (and `usize::MAX`
//!   for a width of `usize::MAX`); both are at least the input length;
//! - `appended_within` accepts a write exactly when the saturated grown
//!   length stays within the limit (for a limit below `usize::MAX`, exactly
//!   `current + added <= limit`), so a bounded buffer that starts within its
//!   limit never passes it;
//! - `cleanup_fuel` is exactly `fuel_per_step * (assignments + groups + 1)`,
//!   with the step count saturated at `usize::MAX` and the product at
//!   `u64::MAX`.
//!
//! That the replacement and indentation bounds dominate the text MiniJinja's
//! `replace` and `indent` produce depends on string semantics and is covered
//! by the Rust tests (an example table and a property test against
//! `str::replace`). The operator rewrite, the bounded filters, and MiniJinja
//! itself are outside the proof. See `verus/README.md` for the method,
//! toolchain, and command.

#![allow(unused)]

#[path = "../src/tokenization/chat_template/limits.rs"]
pub mod limits;
