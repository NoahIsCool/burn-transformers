//! Dependency-free byte scanning of chat-template source.
//!
//! The source rewrites (`{% generation %}` blocks, conditional keyword
//! arguments, and the operator guard) only change text MiniJinja would parse
//! as an expression or statement tag. Finding those tags is pure index
//! arithmetic over the template's bytes, which lives here so Verus can check
//! it:
//!
//! - [`next_tag`] walks the source like MiniJinja's lexer: template text,
//!   `{# comments #}`, and `{% raw %}...{% endraw %}` sections are skipped;
//!   an expression (`{{ ... }}`) or statement (`{% ... %}`) tag ends at the
//!   first closing delimiter outside string literals and open brackets
//!   ([`find_tag_end`]);
//! - [`marker_widths`] measures a tag's whitespace-control markers (`-`, `+`).
//!
//! Every delimiter, quote, and marker is ASCII, so every offset these
//! functions return lies on a UTF-8 character boundary of the source.
//!
//! # Verified behavior
//!
//! `verus/tokenization.rs` checks every function here in place, against a
//! step-by-step reference model of MiniJinja 2.19's lexer written as
//! recursive spec functions (`string_close`, `skip_tag_space`,
//! `basic_tag_scan`, `raw_end_scan`, `tag_end_scan`, `next_tag_scan`). For
//! all inputs the proofs establish that no index leaves its buffer and no
//! arithmetic overflows, and:
//!
//! - [`next_tag`] returns exactly the model's next tag (`next_tag_scan`):
//!   no tag MiniJinja would parse is skipped, and no comment, raw section, or
//!   template text is returned as a tag. A returned span starts at or after
//!   its position, opens with the delimiter of its kind, closes with the
//!   matching delimiter, and ends after the position, so a scan over the
//!   source makes progress and every slice it takes is in bounds;
//! - [`find_tag_end`] returns exactly the first closing delimiter at bracket
//!   depth zero outside string literals (`tag_end_scan`), and
//!   [`skip_string`] exactly the offset after the first unescaped closing
//!   quote, or the end of the buffer (`string_skip_end`);
//! - [`basic_tag_end`] returns exactly `basic_tag_scan`, which finds every
//!   tag `[-+]? ws* name ws* [-+]? %}` (`is_basic_tag`) and only those, for
//!   names such as `raw` and `endraw` that start with a non-space byte; and
//!   [`raw_section_end`] ends at the first such `endraw` tag;
//! - [`next_tag_opening`] returns exactly the first `{{`, `{%`, or `{#` at or
//!   after its start, and [`find_pattern`] the first occurrence of a pattern;
//! - [`marker_widths`] never measures more than the tag body holds;
//! - [`char_boundary_at_most`] returns the largest character boundary at or
//!   below its limit;
//! - [`is_identifier`] accepts exactly the ASCII identifiers.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::string::StringSliceAdditionalSpecFns;

/// The kinds of Jinja tag a source rewrite may change.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TagKind {
    /// An expression tag, `{{ ... }}`.
    Expression,
    /// A statement tag, `{% ... %}`.
    Statement,
}

/// The three delimiters that open a tag or comment.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TagOpening {
    /// `{{`.
    Expression,
    /// `{%`.
    Statement,
    /// `{#`.
    Comment,
}

/// One expression or statement tag: its kind and byte range, delimiters included.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TagSpan {
    /// Whether the tag is an expression or a statement.
    pub(crate) kind: TagKind,
    /// Offset of the opening `{`.
    pub(crate) start: usize,
    /// Offset just past the closing `}`.
    pub(crate) end: usize,
}

/// `raw`, the basic tag that opens a raw section (spelled per byte so Verus sees its contents).
#[cfg_attr(verus_keep_ghost, verus_verify)]
const RAW_NAME: [u8; 3] = [b'r', b'a', b'w'];

/// `endraw`, the basic tag that closes a raw section.
#[cfg_attr(verus_keep_ghost, verus_verify)]
const ENDRAW_NAME: [u8; 6] = [b'e', b'n', b'd', b'r', b'a', b'w'];

#[cfg(verus_keep_ghost)]
verus! {

/// Longest buffer the scanners accept: Rust allocations never exceed `isize::MAX` bytes.
pub(crate) open spec fn scannable(bytes: Seq<u8>) -> bool {
    bytes.len() <= isize::MAX
}

/// `%}`, the closing delimiter of a statement tag.
pub(crate) open spec fn statement_close_bytes() -> Seq<u8> {
    seq![37u8, 125u8]
}

/// `}}`, the closing delimiter of an expression tag.
pub(crate) open spec fn expression_close_bytes() -> Seq<u8> {
    seq![125u8, 125u8]
}

/// `{%`, the opening delimiter of a statement tag.
pub(crate) open spec fn statement_open_bytes() -> Seq<u8> {
    seq![123u8, 37u8]
}

/// The tag opening at `index`, if `bytes[index..]` starts with `{{`, `{%`, or `{#`.
pub(crate) open spec fn opening_at(bytes: Seq<u8>, index: int) -> Option<TagOpening> {
    if 0 <= index && index + 1 < bytes.len() && bytes[index] == 123 {
        if bytes[index + 1] == 123 {
            Some(TagOpening::Expression)
        } else if bytes[index + 1] == 37 {
            Some(TagOpening::Statement)
        } else if bytes[index + 1] == 35 {
            Some(TagOpening::Comment)
        } else {
            None
        }
    } else {
        None
    }
}

/// `pattern` occurs in `bytes` at `index`.
pub(crate) open spec fn occurs_at(bytes: Seq<u8>, index: int, pattern: Seq<u8>) -> bool {
    &&& 0 <= index
    &&& index + pattern.len() <= bytes.len()
    &&& bytes.subrange(index, index + pattern.len()) == pattern
}

/// MiniJinja's whitespace around a basic tag name: ASCII space, tab, LF, FF, CR.
pub(crate) open spec fn is_tag_space(byte: u8) -> bool {
    byte == 32 || byte == 9 || byte == 10 || byte == 12 || byte == 13
}

/// A whitespace-control marker, `-` or `+`.
pub(crate) open spec fn is_marker(byte: u8) -> bool {
    byte == 45 || byte == 43
}

/// An ASCII letter.
pub(crate) open spec fn is_letter(byte: u8) -> bool {
    (65 <= byte <= 90) || (97 <= byte <= 122)
}

/// An ASCII letter, digit, or `_`.
pub(crate) open spec fn is_identifier_byte(byte: u8) -> bool {
    is_letter(byte) || (48 <= byte <= 57) || byte == 95
}

/// Bytes `from..to` are all tag whitespace.
pub(crate) open spec fn all_space(bytes: Seq<u8>, from: int, to: int) -> bool {
    forall|i: int| from <= i < to ==> #[trigger] is_tag_space(bytes[i])
}

/// `bytes[from..end]` is a basic tag `[-+]? ws* name ws* [-+]? %}` (the text after `{%`).
///
/// A leading marker is taken when present (as MiniJinja's `skip_basic_tag`
/// does), then whitespace, the name, whitespace, an optional marker, and `%}`.
pub(crate) open spec fn is_basic_tag(bytes: Seq<u8>, from: int, name: Seq<u8>, end: int) -> bool {
    exists|name_start: int, close: int|
        #![trigger occurs_at(bytes, name_start, name), occurs_at(bytes, close, statement_close_bytes())]
        {
            let name_end = name_start + name.len();
            let lead = if from < bytes.len() && is_marker(bytes[from]) {
                from + 1
            } else {
                from
            };
            let trail = if close > name_end && is_marker(bytes[close - 1]) {
                close - 1
            } else {
                close
            };
            &&& lead <= name_start
            &&& name_end <= close
            &&& close + 2 == end
            &&& end <= bytes.len()
            &&& occurs_at(bytes, name_start, name)
            &&& occurs_at(bytes, close, statement_close_bytes())
            &&& all_space(bytes, lead, name_start)
            &&& all_space(bytes, name_end, trail)
        }
}

// ---------------------------------------------------------------------------
// Reference model of MiniJinja's tag scanning
//
// The recursive spec functions below transcribe, step by step, how MiniJinja
// 2.19's lexer moves through template source: where a string literal closes,
// where whitespace ends, where a basic `raw`/`endraw` tag ends, where a tag's
// closing delimiter sits at bracket depth zero, and which tag comes next once
// comments and raw sections are skipped. The scanners are proved to return
// exactly these values; the declarative facts (`is_basic_tag`, `occurs_at`,
// `opening_at`) are proved alongside them.
// ---------------------------------------------------------------------------

/// `#}`, the closing delimiter of a comment.
pub(crate) open spec fn comment_close_bytes() -> Seq<u8> {
    seq![35u8, 125u8]
}

/// `raw`, the basic tag that opens a raw section.
pub(crate) open spec fn raw_name_bytes() -> Seq<u8> {
    seq![114u8, 97u8, 119u8]
}

/// `endraw`, the basic tag that closes a raw section.
pub(crate) open spec fn endraw_name_bytes() -> Seq<u8> {
    seq![101u8, 110u8, 100u8, 114u8, 97u8, 119u8]
}

/// Where a string literal closed by `quote` ends, scanning its body from `index`.
///
/// A backslash skips the byte after it, the first unescaped `quote` closes
/// the literal (the result is the offset just past it), and running off the
/// end leaves the literal unterminated (`None`).
pub(crate) open spec fn string_close(bytes: Seq<u8>, quote: u8, index: int) -> Option<int>
    decreases bytes.len() + 1 - index,
{
    if index < 0 || index >= bytes.len() {
        None
    } else if bytes[index] == 92 {
        string_close(bytes, quote, index + 2)
    } else if bytes[index] == quote {
        Some(index + 1)
    } else {
        string_close(bytes, quote, index + 1)
    }
}

/// Where scanning resumes after the string literal opened by `quote` at `start`.
///
/// Just past the closing quote, or the end of the buffer when unterminated.
pub(crate) open spec fn string_skip_end(bytes: Seq<u8>, start: int, quote: u8) -> int {
    match string_close(bytes, quote, start + 1) {
        Some(end) => end,
        None => bytes.len() as int,
    }
}

/// The first offset at or after `index` that is not tag whitespace (or the end of the buffer).
pub(crate) open spec fn skip_tag_space(bytes: Seq<u8>, index: int) -> int
    decreases bytes.len() - index,
{
    if 0 <= index < bytes.len() && is_tag_space(bytes[index]) {
        skip_tag_space(bytes, index + 1)
    } else {
        index
    }
}

/// MiniJinja's `skip_basic_tag` on the text after `{%`: the end of `[-+]? ws* name ws* [-+]? %}`.
///
/// An optional marker, greedy whitespace, the name, greedy whitespace, an
/// optional marker, and `%}`; `None` when any part is missing.
pub(crate) open spec fn basic_tag_scan(bytes: Seq<u8>, from: int, name: Seq<u8>) -> Option<int> {
    let lead = if from < bytes.len() && is_marker(bytes[from]) {
        from + 1
    } else {
        from
    };
    let name_start = skip_tag_space(bytes, lead);
    if !occurs_at(bytes, name_start, name) {
        None
    } else {
        let spaces_end = skip_tag_space(bytes, name_start + name.len());
        let close = if spaces_end < bytes.len() && is_marker(bytes[spaces_end]) {
            spaces_end + 1
        } else {
            spaces_end
        };
        if occurs_at(bytes, close, statement_close_bytes()) {
            Some(close + 2)
        } else {
            None
        }
    }
}

/// The first offset at or after `from` where `pattern` occurs (`str::find`).
pub(crate) open spec fn first_occurrence(bytes: Seq<u8>, from: int, pattern: Seq<u8>) -> Option<int>
    decreases bytes.len() + 1 - from,
{
    if from < 0 || from > bytes.len() {
        None
    } else if occurs_at(bytes, from, pattern) {
        Some(from)
    } else {
        first_occurrence(bytes, from + 1, pattern)
    }
}

/// The first offset at or after `from` where `{{`, `{%`, or `{#` opens.
pub(crate) open spec fn first_opening(bytes: Seq<u8>, from: int) -> Option<int>
    decreases bytes.len() - from,
{
    if from < 0 || from >= bytes.len() {
        None
    } else if opening_at(bytes, from).is_some() {
        Some(from)
    } else {
        first_opening(bytes, from + 1)
    }
}

/// End of a raw section's `{% endraw %}`: the first `{%` at or after `search` that starts one.
pub(crate) open spec fn raw_end_scan(bytes: Seq<u8>, search: int) -> Option<int>
    decreases bytes.len() + 1 - search via raw_end_scan_decreases
{
    match first_occurrence(bytes, search, statement_open_bytes()) {
        None => None,
        Some(opening) => match basic_tag_scan(bytes, opening + 2, endraw_name_bytes()) {
            Some(end) => Some(end),
            None => raw_end_scan(bytes, opening + 2),
        },
    }
}

/// Where a tag's `closing` delimiter sits, scanning from `index` at bracket depth `depth`.
///
/// String literals are skipped whole; `([{` deepen and `)]}` shallow the
/// nesting; the delimiter counts only at depth zero. `None` when the buffer
/// ends first.
pub(crate) open spec fn tag_end_scan(bytes: Seq<u8>, closing: Seq<u8>, index: int, depth: int) -> Option<
    int,
>
    decreases bytes.len() + 1 - index via tag_end_scan_decreases
{
    if index < 0 || index >= bytes.len() {
        None
    } else if bytes[index] == 39 || bytes[index] == 34 {
        tag_end_scan(bytes, closing, string_skip_end(bytes, index, bytes[index]), depth)
    } else if depth == 0 && occurs_at(bytes, index, closing) {
        Some(index)
    } else if bytes[index] == 40 || bytes[index] == 91 || bytes[index] == 123 {
        tag_end_scan(bytes, closing, index + 1, depth + 1)
    } else if bytes[index] == 41 || bytes[index] == 93 || bytes[index] == 125 {
        tag_end_scan(bytes, closing, index + 1, depth - 1)
    } else {
        tag_end_scan(bytes, closing, index + 1, depth)
    }
}

/// The next expression or statement tag at or after `search`: its kind, start, and end.
///
/// Template text is skipped up to the next opening. A comment resumes the
/// scan after its `#}`; a `{% raw %}` tag resumes it after the matching
/// `{% endraw %}`; any other statement or expression tag ends at its
/// closing delimiter found by [`tag_end_scan`]. An unterminated comment, raw
/// section, or tag ends the scan (`None`).
pub(crate) open spec fn next_tag_scan(bytes: Seq<u8>, search: int) -> Option<(TagKind, int, int)>
    decreases bytes.len() + 1 - search via next_tag_scan_decreases
{
    match first_opening(bytes, search) {
        None => None,
        Some(start) => {
            let body = start + 2;
            if opening_at(bytes, start) == Some(TagOpening::Comment) {
                match first_occurrence(bytes, body, comment_close_bytes()) {
                    None => None,
                    Some(close) => next_tag_scan(bytes, close + 2),
                }
            } else if opening_at(bytes, start) == Some(TagOpening::Statement) {
                match basic_tag_scan(bytes, body, raw_name_bytes()) {
                    Some(after) => match raw_end_scan(bytes, after) {
                        None => None,
                        Some(end) => next_tag_scan(bytes, end),
                    },
                    None => match tag_end_scan(bytes, statement_close_bytes(), body, 0) {
                        None => None,
                        Some(close) => Some((TagKind::Statement, start, close + 2)),
                    },
                }
            } else {
                match tag_end_scan(bytes, expression_close_bytes(), body, 0) {
                    None => None,
                    Some(close) => Some((TagKind::Expression, start, close + 2)),
                }
            }
        },
    }
}

/// A closed string ends after its scan position, inside the buffer, just past the closing quote.
pub(crate) proof fn lemma_string_close_bounds(bytes: Seq<u8>, quote: u8, index: int)
    requires
        0 <= index,
    ensures
        string_close(bytes, quote, index) matches Some(end) ==> index < end <= bytes.len()
            && bytes[end - 1] == quote,
    decreases bytes.len() + 1 - index,
{
    if index < bytes.len() {
        if bytes[index] == 92 {
            lemma_string_close_bounds(bytes, quote, index + 2);
        } else if bytes[index] != quote {
            lemma_string_close_bounds(bytes, quote, index + 1);
        }
    }
}

/// The first occurrence found is at or after the start and the pattern really occurs there.
pub(crate) proof fn lemma_first_occurrence_bounds(bytes: Seq<u8>, from: int, pattern: Seq<u8>)
    ensures
        first_occurrence(bytes, from, pattern) matches Some(index) ==> from <= index && occurs_at(
            bytes,
            index,
            pattern,
        ),
    decreases bytes.len() + 1 - from,
{
    if 0 <= from <= bytes.len() && !occurs_at(bytes, from, pattern) {
        lemma_first_occurrence_bounds(bytes, from + 1, pattern);
    }
}

/// The first opening found is at or after the start and a tag really opens there.
pub(crate) proof fn lemma_first_opening_bounds(bytes: Seq<u8>, from: int)
    ensures
        first_opening(bytes, from) matches Some(index) ==> from <= index && opening_at(
            bytes,
            index,
        ).is_some(),
    decreases bytes.len() - from,
{
    if 0 <= from < bytes.len() && opening_at(bytes, from).is_none() {
        lemma_first_opening_bounds(bytes, from + 1);
    }
}

/// Whitespace skipping stays in the buffer, crosses only whitespace, and stops at a non-space byte.
pub(crate) proof fn lemma_skip_tag_space(bytes: Seq<u8>, index: int)
    requires
        0 <= index <= bytes.len(),
    ensures
        index <= skip_tag_space(bytes, index) <= bytes.len(),
        all_space(bytes, index, skip_tag_space(bytes, index)),
        skip_tag_space(bytes, index) < bytes.len() ==> !is_tag_space(
            bytes[skip_tag_space(bytes, index)],
        ),
    decreases bytes.len() - index,
{
    if index < bytes.len() && is_tag_space(bytes[index]) {
        lemma_skip_tag_space(bytes, index + 1);
    }
}

/// Whitespace skipping from `index` stops at `stop` when only whitespace lies between them and `stop` is not whitespace.
pub(crate) proof fn lemma_skip_tag_space_stops(bytes: Seq<u8>, index: int, stop: int)
    requires
        0 <= index <= stop <= bytes.len(),
        all_space(bytes, index, stop),
        stop < bytes.len() ==> !is_tag_space(bytes[stop]),
    ensures
        skip_tag_space(bytes, index) == stop,
    decreases stop - index,
{
    if index < stop {
        assert(is_tag_space(bytes[index]));
        lemma_skip_tag_space_stops(bytes, index + 1, stop);
    }
}

/// A basic tag found from `from` ends after it and inside the buffer.
pub(crate) proof fn lemma_basic_tag_scan_bounds(bytes: Seq<u8>, from: int, name: Seq<u8>)
    requires
        0 <= from <= bytes.len(),
    ensures
        basic_tag_scan(bytes, from, name) matches Some(end) ==> from < end <= bytes.len(),
{
    let lead = if from < bytes.len() && is_marker(bytes[from]) {
        from + 1
    } else {
        from
    };
    lemma_skip_tag_space(bytes, lead);
    let name_start = skip_tag_space(bytes, lead);
    if occurs_at(bytes, name_start, name) {
        lemma_skip_tag_space(bytes, name_start + name.len());
    }
}

/// Every basic tag the declarative grammar admits is the one the scan finds, when the name starts with a non-space byte.
///
/// Whitespace skipping is greedy and the name's first byte is not
/// whitespace, so the name can only start where the skip stops; the same
/// holds for the closing marker and `%}`. Hence [`is_basic_tag`] has at most
/// one end and [`basic_tag_scan`] returns it.
pub(crate) proof fn lemma_basic_tag_scan_complete(bytes: Seq<u8>, from: int, name: Seq<u8>, end: int)
    requires
        0 <= from <= bytes.len(),
        name.len() > 0,
        !is_tag_space(name[0]),
        is_basic_tag(bytes, from, name, end),
    ensures
        basic_tag_scan(bytes, from, name) == Some(end),
{
    let (name_start, close) = choose|name_start: int, close: int|
        #![trigger occurs_at(bytes, name_start, name), occurs_at(bytes, close, statement_close_bytes())]
        {
            let name_end = name_start + name.len();
            let lead = if from < bytes.len() && is_marker(bytes[from]) {
                from + 1
            } else {
                from
            };
            let trail = if close > name_end && is_marker(bytes[close - 1]) {
                close - 1
            } else {
                close
            };
            &&& lead <= name_start
            &&& name_end <= close
            &&& close + 2 == end
            &&& end <= bytes.len()
            &&& occurs_at(bytes, name_start, name)
            &&& occurs_at(bytes, close, statement_close_bytes())
            &&& all_space(bytes, lead, name_start)
            &&& all_space(bytes, name_end, trail)
        };
    let lead = if from < bytes.len() && is_marker(bytes[from]) {
        from + 1
    } else {
        from
    };
    let name_end = name_start + name.len();
    assert(bytes.subrange(name_start, name_end)[0] == name[0]);
    lemma_skip_tag_space_stops(bytes, lead, name_start);
    assert(bytes.subrange(close, close + 2)[0] == statement_close_bytes()[0]);
    if close > name_end && is_marker(bytes[close - 1]) {
        lemma_skip_tag_space_stops(bytes, name_end, close - 1);
    } else {
        lemma_skip_tag_space_stops(bytes, name_end, close);
    }
}

/// A raw section's end found from `search` lies after it and inside the buffer.
pub(crate) proof fn lemma_raw_end_scan_bounds(bytes: Seq<u8>, search: int)
    requires
        0 <= search,
    ensures
        raw_end_scan(bytes, search) matches Some(end) ==> search < end <= bytes.len(),
    decreases bytes.len() + 1 - search,
{
    lemma_first_occurrence_bounds(bytes, search, statement_open_bytes());
    if let Some(opening) = first_occurrence(bytes, search, statement_open_bytes()) {
        lemma_basic_tag_scan_bounds(bytes, opening + 2, endraw_name_bytes());
        if basic_tag_scan(bytes, opening + 2, endraw_name_bytes()).is_none() {
            lemma_raw_end_scan_bounds(bytes, opening + 2);
        }
    }
}

/// Termination of [`raw_end_scan`]: each retry resumes after the `{%` it rejected.
#[via_fn]
pub(crate) proof fn raw_end_scan_decreases(bytes: Seq<u8>, search: int) {
    lemma_first_occurrence_bounds(bytes, search, statement_open_bytes());
}

/// Termination of [`tag_end_scan`]: a skipped string ends after its opening quote.
#[via_fn]
pub(crate) proof fn tag_end_scan_decreases(bytes: Seq<u8>, closing: Seq<u8>, index: int, depth: int) {
    if 0 <= index < bytes.len() {
        lemma_string_close_bounds(bytes, bytes[index], index + 1);
    }
}

/// Termination of [`next_tag_scan`]: a skipped comment or raw section ends after the scan position.
#[via_fn]
pub(crate) proof fn next_tag_scan_decreases(bytes: Seq<u8>, search: int) {
    lemma_first_opening_bounds(bytes, search);
    if let Some(start) = first_opening(bytes, search) {
        lemma_first_occurrence_bounds(bytes, start + 2, comment_close_bytes());
        lemma_basic_tag_scan_bounds(bytes, start + 2, raw_name_bytes());
        if let Some(after) = basic_tag_scan(bytes, start + 2, raw_name_bytes()) {
            lemma_raw_end_scan_bounds(bytes, after);
        }
    }
}

} // verus!

/// Whether `byte` is ASCII whitespace as `char::is_ascii_whitespace` defines it.
///
/// Verified: exactly space, tab, line feed, form feed, and carriage return.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(space =>
    ensures
        space == is_tag_space(byte),
))]
pub(crate) fn is_ascii_space(byte: u8) -> bool {
    byte == b' ' || byte == b'\t' || byte == b'\n' || byte == 0x0c || byte == b'\r'
}

/// Whether `pattern` occurs in `bytes` at `index`.
///
/// Verified: exactly when the pattern fits there and every byte matches.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
        found == occurs_at(bytes@, index as int, pattern@),
))]
pub(crate) fn starts_with_at(bytes: &[u8], index: usize, pattern: &[u8]) -> bool {
    if index > bytes.len() || pattern.len() > bytes.len() - index {
        return false;
    }
    let mut offset = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            offset <= pattern.len(),
            index + pattern.len() <= bytes.len(),
            forall|i: int| 0 <= i < offset ==> bytes@[index + i] == pattern@[i],
        decreases pattern.len() - offset,
    ))]
    while offset < pattern.len() {
        if bytes[index + offset] != pattern[offset] {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(bytes@.subrange(index as int, index + pattern.len())[offset as int]
                    != pattern@[offset as int]);
            }
            return false;
        }
        offset += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(bytes@.subrange(index as int, index + pattern.len()) =~= pattern@);
    }
    true
}

/// Offset of the first occurrence of `pattern` in `bytes` at or after `from`.
///
/// This is `str::find` on `&source[from..]`, shifted by `from`. Verified: the
/// result is the first offset at or after `from` where the pattern occurs
/// (`first_occurrence`), and `None` means it occurs nowhere there.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        scannable(bytes@),
        from <= bytes.len(),
    ensures
        match found {
            Some(index) => first_occurrence(bytes@, from as int, pattern@) == Some(index as int),
            None => first_occurrence(bytes@, from as int, pattern@).is_none(),
        },
        match found {
            Some(index) => {
                &&& from <= index
                &&& occurs_at(bytes@, index as int, pattern@)
                &&& forall|i: int| from <= i < index ==> !#[trigger] occurs_at(bytes@, i, pattern@)
            },
            None => forall|i: int| from <= i ==> !#[trigger] occurs_at(bytes@, i, pattern@),
        },
))]
pub(crate) fn find_pattern(bytes: &[u8], from: usize, pattern: &[u8]) -> Option<usize> {
    let mut index = from;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            from <= index <= bytes.len() + 1,
            scannable(bytes@),
            forall|i: int| from <= i < index ==> !#[trigger] occurs_at(bytes@, i, pattern@),
            first_occurrence(bytes@, index as int, pattern@) == first_occurrence(
                bytes@,
                from as int,
                pattern@,
            ),
        decreases bytes.len() + 1 - index,
    ))]
    while index <= bytes.len() {
        if starts_with_at(bytes, index, pattern) {
            return Some(index);
        }
        index += 1;
    }
    None
}

/// Offset and delimiter of the first `{{`, `{%`, or `{#` at or after `from`.
///
/// Verified: exactly the first tag opening at or after `from`
/// (`first_opening`); `None` when there is none.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
        match found {
            Some((index, _)) => first_opening(bytes@, from as int) == Some(index as int),
            None => first_opening(bytes@, from as int).is_none(),
        },
        match found {
            Some((index, opening)) => {
                &&& from <= index
                &&& opening_at(bytes@, index as int) == Some(opening)
                &&& forall|i: int| from <= i < index ==> #[trigger] opening_at(bytes@, i).is_none()
            },
            None => forall|i: int| from <= i ==> #[trigger] opening_at(bytes@, i).is_none(),
        },
))]
pub(crate) fn next_tag_opening(bytes: &[u8], from: usize) -> Option<(usize, TagOpening)> {
    let mut index = from;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            from <= index,
            forall|i: int| from <= i < index ==> #[trigger] opening_at(bytes@, i).is_none(),
            first_opening(bytes@, index as int) == first_opening(bytes@, from as int),
        decreases bytes.len() - index,
    ))]
    while index < bytes.len() {
        if bytes[index] == b'{' && index + 1 < bytes.len() {
            if bytes[index + 1] == b'{' {
                return Some((index, TagOpening::Expression));
            }
            if bytes[index + 1] == b'%' {
                return Some((index, TagOpening::Statement));
            }
            if bytes[index + 1] == b'#' {
                return Some((index, TagOpening::Comment));
            }
        }
        index += 1;
    }
    None
}

/// Returns the index just past the string literal opened by `quote` at `start`.
///
/// A backslash escapes the next byte. An unterminated literal runs to the end
/// of the buffer. Verified: the result is exactly `string_skip_end`, just
/// past the first `quote` byte after `start` that no backslash escapes, or
/// the buffer length when there is none; so it is after `start`, at most the
/// buffer length, and unless it is the buffer length just past a `quote`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(end =>
    requires
        start < bytes.len(),
        scannable(bytes@),
    ensures
        end == string_skip_end(bytes@, start as int, quote),
        start < end <= bytes.len(),
        end == bytes.len() || (start + 2 <= end && bytes@[end - 1] == quote),
))]
pub(crate) fn skip_string(bytes: &[u8], start: usize, quote: u8) -> usize {
    let mut index = start + 1;
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_string_close_bounds(bytes@, quote, start + 1);
    }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            start < index <= bytes.len() + 1,
            scannable(bytes@),
            string_close(bytes@, quote, index as int) == string_close(bytes@, quote, start + 1),
        decreases bytes.len() + 1 - index,
    ))]
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            index += 2;
        } else if bytes[index] == quote {
            return index + 1;
        } else {
            index += 1;
        }
    }
    bytes.len()
}

/// Finds a tag's closing delimiter, at or after `from`, outside string literals and open brackets.
///
/// `from` is just after the opening delimiter. Like MiniJinja's lexer, the
/// delimiter only counts at bracket depth zero, so a `}}` that closes a
/// nested dict literal does not end an expression tag; a closing bracket
/// without an opening one makes the depth negative, which no delimiter can
/// then end. Verified: the result is exactly `tag_end_scan` from `from` at
/// depth zero, so it is the first offset where the delimiter occurs outside
/// every string literal at bracket depth zero, and `None` means there is
/// none; a returned offset is at or after `from` and the closing delimiter
/// occurs there.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        scannable(bytes@),
    ensures
        match found {
            Some(index) => tag_end_scan(bytes@, closing@, from as int, 0) == Some(index as int),
            None => tag_end_scan(bytes@, closing@, from as int, 0).is_none(),
        },
        found matches Some(index) ==> from <= index && occurs_at(bytes@, index as int, closing@),
))]
pub(crate) fn find_tag_end(bytes: &[u8], from: usize, closing: &[u8]) -> Option<usize> {
    let mut depth: isize = 0;
    let mut index = from;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            from <= index,
            scannable(bytes@),
            -(index as int) <= depth <= index as int,
            tag_end_scan(bytes@, closing@, index as int, depth as int) == tag_end_scan(
                bytes@,
                closing@,
                from as int,
                0,
            ),
        decreases bytes.len() - index,
    ))]
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'\'' || byte == b'"' {
            index = skip_string(bytes, index, byte);
            continue;
        }
        // Byte comparison: `index` may sit inside a multi-byte character.
        if depth == 0 && starts_with_at(bytes, index, closing) {
            return Some(index);
        }
        if byte == b'(' || byte == b'[' || byte == b'{' {
            depth += 1;
        } else if byte == b')' || byte == b']' || byte == b'}' {
            depth -= 1;
        }
        index += 1;
    }
    None
}

/// Matches `[-+]? ws* name ws* [-+]? %}` at `from` (just after `{%`) and returns its end.
///
/// This is MiniJinja's `skip_basic_tag`, which recognizes `raw` and `endraw`
/// before any expression parsing. Verified: the result is exactly
/// `basic_tag_scan`; a returned end is the end of such a tag
/// (`is_basic_tag`), and for a name that starts with a non-space byte
/// (`raw`, `endraw`) every such tag is found: `None` means there is none, and
/// a tag has only the one end returned.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        scannable(bytes@),
        from <= bytes.len(),
    ensures
        match found {
            Some(end) => basic_tag_scan(bytes@, from as int, name@) == Some(end as int),
            None => basic_tag_scan(bytes@, from as int, name@).is_none(),
        },
        found matches Some(end) ==> from < end <= bytes.len() && is_basic_tag(
            bytes@,
            from as int,
            name@,
            end as int,
        ),
        name.len() > 0 && !is_tag_space(name@[0]) ==> forall|end: int| #[trigger] is_basic_tag(
            bytes@,
            from as int,
            name@,
            end,
        ) ==> (found matches Some(found_end) && found_end == end),
))]
pub(crate) fn basic_tag_end(bytes: &[u8], from: usize, name: &[u8]) -> Option<usize> {
    #[cfg(verus_keep_ghost)]
    proof! {
        if name.len() > 0 && !is_tag_space(name@[0]) {
            assert forall|end: int| #[trigger] is_basic_tag(bytes@, from as int, name@, end) implies
                basic_tag_scan(bytes@, from as int, name@) == Some(end) by {
                lemma_basic_tag_scan_complete(bytes@, from as int, name@, end);
            }
        }
    }
    let mut index = from;
    if index < bytes.len() && (bytes[index] == b'-' || bytes[index] == b'+') {
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof_decl! {
        let ghost lead = index;
    }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            lead <= index <= bytes.len(),
            all_space(bytes@, lead as int, index as int),
            skip_tag_space(bytes@, index as int) == skip_tag_space(bytes@, lead as int),
        decreases bytes.len() - index,
    ))]
    while index < bytes.len() && is_ascii_space(bytes[index]) {
        index += 1;
    }
    if !starts_with_at(bytes, index, name) {
        return None;
    }
    #[cfg(verus_keep_ghost)]
    proof_decl! {
        let ghost name_start = index;
    }
    index += name.len();
    #[cfg(verus_keep_ghost)]
    proof_decl! {
        let ghost name_end = index;
    }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            name_end <= index <= bytes.len(),
            all_space(bytes@, name_end as int, index as int),
            skip_tag_space(bytes@, index as int) == skip_tag_space(bytes@, name_end as int),
        decreases bytes.len() - index,
    ))]
    while index < bytes.len() && is_ascii_space(bytes[index]) {
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof_decl! {
        let ghost spaces_end = index;
    }
    if index < bytes.len() && (bytes[index] == b'-' || bytes[index] == b'+') {
        index += 1;
    }
    let statement_close = [b'%', b'}'];
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(statement_close@ =~= statement_close_bytes());
    }
    if !starts_with_at(bytes, index, &statement_close) {
        return None;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        let close = index as int;
        assert(occurs_at(bytes@, close, statement_close_bytes()));
        assert(occurs_at(bytes@, name_start as int, name@));
        if close == spaces_end && close > name_end {
            assert(is_tag_space(bytes@[close - 1]));
        }
    }
    Some(index + 2)
}

/// End of a raw section's body plus its `{% endraw %}` tag, scanning from `from`.
///
/// As in MiniJinja, the first `{%` that starts an `endraw` tag ends the
/// section; every other byte is data. Verified: the result is exactly
/// `raw_end_scan`, the end of the `endraw` tag after the first `{%` at or
/// after `from` that starts one; it is after `from`, at most the buffer
/// length, and ends a basic `endraw` tag (`is_basic_tag`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        scannable(bytes@),
        from <= bytes.len(),
    ensures
        match found {
            Some(end) => raw_end_scan(bytes@, from as int) == Some(end as int),
            None => raw_end_scan(bytes@, from as int).is_none(),
        },
        found matches Some(end) ==> from < end <= bytes.len() && exists|opening: int|
            from <= opening && #[trigger] occurs_at(bytes@, opening, statement_open_bytes())
                && is_basic_tag(bytes@, opening + 2, endraw_name_bytes(), end as int),
))]
pub(crate) fn raw_section_end(bytes: &[u8], from: usize) -> Option<usize> {
    let mut search = from;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            from <= search <= bytes.len(),
            scannable(bytes@),
            raw_end_scan(bytes@, search as int) == raw_end_scan(bytes@, from as int),
        decreases bytes.len() - search,
    ))]
    loop {
        let statement_open = [b'{', b'%'];
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(statement_open@ =~= statement_open_bytes());
        }
        let opening = find_pattern(bytes, search, &statement_open)?;
        let after_opening = opening + 2;
        let endraw: &[u8] = &ENDRAW_NAME;
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(endraw@ =~= endraw_name_bytes());
        }
        if let Some(end) = basic_tag_end(bytes, after_opening, endraw) {
            return Some(end);
        }
        search = after_opening;
    }
}

/// Finds the next expression or statement tag at or after `position`.
///
/// Template text is skipped, and so are `{# comments #}` and `{% raw %}`
/// sections (through their `{% endraw %}`). An unterminated comment, raw
/// section, or tag ends the scan (`None`): MiniJinja's parser then rejects
/// the remainder itself. Verified: the result is exactly `next_tag_scan`,
/// the reference model of MiniJinja's lexer, so no tag MiniJinja would parse
/// is skipped and none it treats as data is returned; a returned span starts
/// at or after `position`, opens with `{{` (expressions) or `{%`
/// (statements), closes with the matching `}}` or `%}`, holds both
/// delimiters (`start + 4 <= end`), and ends inside the buffer, so a scan
/// that resumes at `end` makes progress.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        scannable(bytes@),
        position <= bytes.len(),
    ensures
        match found {
            Some(span) => next_tag_scan(bytes@, position as int) == Some(
                (span.kind, span.start as int, span.end as int),
            ),
            None => next_tag_scan(bytes@, position as int).is_none(),
        },
        found matches Some(span) ==> {
            &&& position <= span.start
            &&& span.start + 4 <= span.end <= bytes.len()
            &&& match span.kind {
                TagKind::Expression => {
                    &&& opening_at(bytes@, span.start as int) == Some(TagOpening::Expression)
                    &&& occurs_at(bytes@, span.end - 2, expression_close_bytes())
                },
                TagKind::Statement => {
                    &&& opening_at(bytes@, span.start as int) == Some(TagOpening::Statement)
                    &&& occurs_at(bytes@, span.end - 2, statement_close_bytes())
                },
            }
        },
))]
pub(crate) fn next_tag(bytes: &[u8], position: usize) -> Option<TagSpan> {
    let mut search = position;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            position <= search <= bytes.len(),
            scannable(bytes@),
            next_tag_scan(bytes@, search as int) == next_tag_scan(bytes@, position as int),
        decreases bytes.len() - search,
    ))]
    loop {
        let (start, opening) = next_tag_opening(bytes, search)?;
        let body_start = start + 2;
        match opening {
            TagOpening::Comment => {
                let comment_close = [b'#', b'}'];
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(comment_close@ =~= comment_close_bytes());
                }
                let close = find_pattern(bytes, body_start, &comment_close)?;
                #[cfg(verus_keep_ghost)]
                proof! {
                    lemma_first_occurrence_bounds(bytes@, body_start as int, comment_close_bytes());
                }
                search = close + 2;
            }
            TagOpening::Statement => {
                let raw: &[u8] = &RAW_NAME;
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(raw@ =~= raw_name_bytes());
                }
                if let Some(after) = basic_tag_end(bytes, body_start, raw) {
                    // Like MiniJinja, copy everything up to the matching `{% endraw %}`.
                    search = raw_section_end(bytes, after)?;
                } else {
                    let statement_close = [b'%', b'}'];
                    let close = find_tag_end(bytes, body_start, &statement_close)?;
                    #[cfg(verus_keep_ghost)]
                    proof! {
                        assert(statement_close@ =~= statement_close_bytes());
                        assert(bytes@.subrange(close as int, close + 2) =~= statement_close_bytes());
                    }
                    return Some(TagSpan {
                        kind: TagKind::Statement,
                        start,
                        end: close + 2,
                    });
                }
            }
            TagOpening::Expression => {
                let expression_close = [b'}', b'}'];
                let close = find_tag_end(bytes, body_start, &expression_close)?;
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(expression_close@ =~= expression_close_bytes());
                    assert(bytes@.subrange(close as int, close + 2) =~= expression_close_bytes());
                }
                return Some(TagSpan {
                    kind: TagKind::Expression,
                    start,
                    end: close + 2,
                });
            }
        }
    }
}

/// Widths of the leading and trailing whitespace-control markers of a tag's inner text.
///
/// `inner` is the tag without its `{%`/`%}` (or `{{`/`}}`) delimiters. A
/// leading `-` or `+` is the opening marker; a trailing one, when it is not
/// the opening marker itself, is the closing marker. Verified: each width is
/// one exactly when its marker is present, and together they never exceed
/// the inner text, so `inner[open..len - close]` is a valid range.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(widths =>
    ensures
        widths.0 == if inner.len() > 0 && is_marker(inner@[0]) { 1usize } else { 0 },
        widths.1 == if inner.len() > widths.0 && is_marker(inner@[inner.len() - 1]) {
            1usize
        } else {
            0
        },
        widths.0 + widths.1 <= inner.len(),
))]
pub(crate) fn marker_widths(inner: &[u8]) -> (usize, usize) {
    let open = if !inner.is_empty() && (inner[0] == b'-' || inner[0] == b'+') {
        1
    } else {
        0
    };
    let close = if inner.len() > open
        && (inner[inner.len() - 1] == b'-' || inner[inner.len() - 1] == b'+')
    {
        1
    } else {
        0
    };
    (open, close)
}

/// Whether `name` is an ASCII identifier MiniJinja's lexer reads as one name.
///
/// That is a letter or `_` followed by letters, digits, and `_`. Verified:
/// exactly when the bytes are non-empty, the first is a letter or `_`, and
/// every byte is a letter, digit, or `_`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(identifier =>
    ensures
        identifier == (name.len() > 0 && (is_letter(name@[0]) || name@[0] == 95) && forall|i: int|
            0 <= i < name.len() ==> is_identifier_byte(#[trigger] name@[i])),
))]
// Verus has no specification for `u8::is_ascii_alphanumeric` or `RangeInclusive::contains`.
#[allow(clippy::manual_is_ascii_check, clippy::manual_range_contains)]
pub(crate) fn is_identifier(name: &[u8]) -> bool {
    if name.is_empty() {
        return false;
    }
    let first = name[0];
    if !(first == b'_' || (b'a' <= first && first <= b'z') || (b'A' <= first && first <= b'Z')) {
        return false;
    }
    let mut index = 1;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            1 <= index <= name.len(),
            is_letter(name@[0]) || name@[0] == 95,
            forall|i: int| 0 <= i < index ==> is_identifier_byte(#[trigger] name@[i]),
        decreases name.len() - index,
    ))]
    while index < name.len() {
        let byte = name[index];
        if !(byte == b'_'
            || (b'a' <= byte && byte <= b'z')
            || (b'A' <= byte && byte <= b'Z')
            || (b'0' <= byte && byte <= b'9'))
        {
            return false;
        }
        index += 1;
    }
    true
}

/// The largest character boundary of `text` at or below `limit`.
///
/// Used to shorten a tag for an error message without splitting a
/// character. Verified: the result is at most `limit`, is a character
/// boundary, and no larger offset up to `limit` is one; the backward search
/// stops at zero at the latest, which is always a boundary.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(end =>
    requires
        limit <= text.spec_bytes().len(),
    ensures
        end <= limit,
        vstd::utf8::is_char_boundary(text.spec_bytes(), end as int),
        forall|later: int| end < later <= limit
            ==> !vstd::utf8::is_char_boundary(text.spec_bytes(), later),
))]
pub(crate) fn char_boundary_at_most(text: &str, limit: usize) -> usize {
    let mut end = limit;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            end <= limit,
            limit <= text.spec_bytes().len(),
            forall|later: int| end < later <= limit
                ==> !vstd::utf8::is_char_boundary(text.spec_bytes(), later),
        decreases end,
    ))]
    while !text.is_char_boundary(end) {
        #[cfg(verus_keep_ghost)]
        proof! {
            vstd::utf8::encode_utf8_valid_utf8(text@);
            vstd::utf8::is_char_boundary_start_end_of_seq(text.spec_bytes());
        }
        end -= 1;
    }
    end
}

#[cfg(test)]
#[path = "scan_tests.rs"]
mod tests;
