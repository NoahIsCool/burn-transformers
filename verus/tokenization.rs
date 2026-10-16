//! Verus harness for the tokenizer's pure layout and calendar arithmetic.
//!
//! Verifies, in place, the specifications attached to
//! `src/tokenization/layout.rs`:
//!
//! - `content_token_budget` reserves exactly the post-processor's template
//!   width and rejects limits that cannot hold it;
//! - `padding_split` places all padding on the configured side and widens a
//!   row exactly to its target;
//! - `pad_token_row` keeps every real token at its shifted index and fills
//!   only the padding positions with the pad ID;
//! - `validity_mask` marks exactly the real-token block as valid.
//!
//! and to `src/tokenization/calendar.rs`, against a transcription of CPython's
//! `datetime` helpers (`_is_leap`, `_days_before_year`, `_days_in_month`,
//! `_days_before_month`, `_ymd2ord`):
//!
//! - `civil_time` returns a value exactly for the instants of years 1 through
//!   9999, and `civil_time_in_supported_range` computes, without overflow,
//!   Python's UTC reading of the instant: a valid date (`1 <= day <=
//!   days_in_month`) whose proleptic Gregorian ordinal is `floor(t / 86400) +
//!   719163`, the time of day `hour * 3600 + minute * 60 + second == t mod
//!   86400`, Python's `weekday()` `(ordinal + 6) mod 7`, and the day of year
//!   `_days_before_month + day`;
//! - `is_leap_year` equals Python's `_is_leap` on non-negative years.
//!
//! Hinnant's year-of-era estimate is shown to name the March-based year of
//! every day of an era from its monotonicity plus a `by (compute_only)`
//! evaluation of each of the 400 years' first and last days.
//!
//! Tokenizer encoding, special-token insertion, and tensor conversion are
//! outside the proof and remain covered by the Rust tests. See
//! `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
// Loop invariants are attributes on `while` expressions, which stable Rust
// gates behind this feature; Verus's pinned toolchain accepts it.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/tokenization/calendar.rs"]
pub mod calendar;
#[path = "../src/tokenization/layout.rs"]
pub mod layout;
