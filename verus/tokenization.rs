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
//! and to `src/tokenization/calendar.rs`:
//!
//! - `civil_time` accepts exactly the instants of years 1 through 9999 and,
//!   with `civil_time_in_supported_range`, computes every calendar field
//!   without overflow and inside its range (month 1-12, day 1-31, day of year
//!   1-366, weekday 0-6, hour 0-23, minute and second 0-59);
//! - `is_leap_year` is total on non-negative years.
//!
//! Calendar *correctness* (that the fields name the right date) is not
//! proved; Python goldens and a day-counting property test cover it.
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
