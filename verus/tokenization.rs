//! Verus harness for the tokenizer's pure integer and byte-scanning cores.
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
//! - `validity_mask` marks exactly the real-token block as valid;
//! - `encoding_limit_violation` reports exactly the first violated rule of
//!   `max_length > 0`, fixed width `> 0`, and `max_length <= width`;
//! - `longest_row` bounds every row and is attained by one;
//! - `pad_batch` returns one row and one mask per input, each exactly the
//!   single-row padding of its input to the target (so the batch is
//!   rectangular), or the index of the first row longer than the target;
//!   `unpadded_masks` marks every token of every unpadded row valid;
//! - `first_ragged_row` finds the first token/mask pair whose width differs
//!   from the batch width, and `flatten_token_rows` / `flatten_mask_rows` lay
//!   rectangular rows out row-major: entry `r * width + c` is row `r`,
//!   column `c`.
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
//! - `is_leap_year` equals Python's `_is_leap` on non-negative years, and
//!   `seconds_before_epoch` floors a pre-epoch host clock reading exactly.
//!
//! Hinnant's year-of-era estimate is shown to name the March-based year of
//! every day of an era from its monotonicity plus a `by (compute_only)`
//! evaluation of each of the 400 years' first and last days.
//!
//! and to `src/tokenization/vocabulary.rs`:
//!
//! - `declared_special_id` accepts exactly the declared JSON IDs that fit
//!   `u32`; `reconcile_special_id` returns the vocabulary's own ID, and only
//!   when no ID or the same ID was declared; `id_fits_vocabulary` holds
//!   exactly for IDs below the model's vocabulary size.
//!
//! and to `src/tokenization/pycompat/decimal.rs`, the integer core of
//! Python's `round`:
//!
//! - `checked_power_of_ten` is exactly `10^places`, or `None` exactly when it
//!   exceeds `i128::MAX` (`i128::checked_pow`);
//! - `round_integer_half_even` returns the multiple of `10^places` nearest to
//!   the value, ties to the even quotient (a multiple at most half a unit
//!   away, exactly half only with an even quotient), or `None` exactly when
//!   that multiple lies outside `i128`;
//! - `increment_decimal_digits` adds exactly one to the number its ASCII
//!   digits spell, growing by a digit only when every digit was `9`;
//! - `round_digits_to_tens` (Python's `round(float, -k)` on a float's exact
//!   decimal expansion) returns the digits of the quotient by `10^k` plus
//!   round-half-even's carry, a fraction breaking ties upward; its
//!   comparison of the dropped digits with `50...0` orders twice their value
//!   against the unit exactly.
//!
//! and to `src/tokenization/chat_template/scan.rs`, the tag scanner shared by
//! the chat-template source rewrites:
//!
//! - `next_tag_opening` and `find_pattern` return exactly the first tag
//!   opening or pattern occurrence at or after their start;
//! - `skip_string` moves strictly forward to just past the closing quote or
//!   to the end; `find_tag_end` returns only offsets where the closing
//!   delimiter occurs; `basic_tag_end` accepts exactly
//!   `[-+]? ws* name ws* [-+]? %}` and `raw_section_end` ends at such an
//!   `endraw` tag;
//! - `next_tag` returns spans that start at or after the scan position, open
//!   with their kind's delimiter, close with the matching one, and end inside
//!   the source, so the rewrite loop makes progress and slices in bounds;
//!   `marker_widths` never exceeds the tag body, and `char_boundary_at_most`
//!   returns the largest character boundary at or below its limit (against
//!   vstd's UTF-8 model of `str`); `is_identifier`, which picks the namespace
//!   attributes the sandbox clears to break reference cycles, accepts exactly
//!   the ASCII identifiers.
//!
//! and to `src/tokenization/chat_template/guard/lexer.rs`, the operator
//! guard's tag-body lexer:
//!
//! - `lex_spans` partitions a body into ordered, disjoint, non-empty tokens
//!   separated only by ASCII whitespace, each of its kind's lexical class (a
//!   quoted string, a number starting with a digit, a maximal identifier, or a
//!   one- or two-byte symbol from MiniJinja's tables), so no operator byte can
//!   hide inside a name; its errors name an unterminated quote or a byte no
//!   token starts with;
//! - `string_end` ends just past the matching quote, and `number_end`
//!   consumes at least the leading digit and only number characters, accepts
//!   a sign only after an exponent letter, a point at most once and before
//!   any exponent (never with a radix prefix), at most one decimal exponent,
//!   and reports a float exactly for a decimal number containing `.` or an
//!   exponent; number tokens carry these facts.
//!
//! The sandbox's size and budget arithmetic in
//! `src/tokenization/chat_template/limits.rs` is checked by
//! `verus/robustness.rs`.
//!
//! Outside the proofs, and covered by the Rust tests: tokenizer encoding,
//! truncation, and post-processing (the `tokenizers` crate), streamed
//! decoding (`tokenizers::DecodeStream`), the Burn tensor constructors, which
//! tags `next_tag` skips as comments and raw sections (its contract covers
//! delimiters and progress, not that classification), the guard's recursive
//! parser and its nesting counters, the keyword-argument rewrite's lexer, the
//! `format` filter's printf width scan, `strftime` and float formatting, and
//! the MiniJinja engine itself. See `verus/README.md` for the method,
//! toolchain, and command.

#![allow(unused)]
// Loop invariants are attributes on `while` expressions, which stable Rust
// gates behind this feature; Verus's pinned toolchain accepts it.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/tokenization/calendar.rs"]
pub mod calendar;
#[path = "../src/tokenization/layout.rs"]
pub mod layout;
#[path = "../src/tokenization/pycompat/decimal.rs"]
pub mod decimal;
#[path = "../src/tokenization/vocabulary.rs"]
pub mod vocabulary;
#[path = "../src/tokenization/chat_template/scan.rs"]
pub mod scan;
#[path = "../src/tokenization/chat_template/guard/lexer.rs"]
pub mod lexer;
