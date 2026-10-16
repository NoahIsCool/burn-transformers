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
//! `verus/tokenization.rs` checks every function here in place. For all
//! inputs the proofs establish that no index leaves its buffer and no
//! arithmetic overflows, and:
//!
//! - [`next_tag_opening`] returns exactly the first `{{`, `{%`, or `{#` at or
//!   after its start, and [`find_pattern`] the first occurrence of a pattern;
//! - [`skip_string`] moves strictly forward, to just past the closing quote or
//!   to the end of the buffer;
//! - [`find_tag_end`] returns only offsets where the closing delimiter really
//!   occurs;
//! - [`basic_tag_end`] accepts exactly `[-+]? ws* name ws* [-+]? %}`, and
//!   [`raw_section_end`] ends at such an `endraw` tag;
//! - [`next_tag`] returns a span that starts at or after its position, opens
//!   with the delimiter of its kind, closes with the matching delimiter, and
//!   ends after the position, so a scan over the source makes progress and
//!   every slice it takes is in bounds;
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
/// result is the first offset at or after `from` where the pattern occurs,
/// and `None` means it occurs nowhere there.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        scannable(bytes@),
        from <= bytes.len(),
    ensures
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
            from <= index,
            scannable(bytes@),
            forall|i: int| from <= i < index ==> !#[trigger] occurs_at(bytes@, i, pattern@),
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
/// Verified: exactly the first tag opening at or after `from`; `None` when
/// there is none.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
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
/// of the buffer. Verified: the result is after `start` and at most the
/// buffer length, and unless it is the buffer length it is just past a
/// `quote` byte.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(end =>
    requires
        start < bytes.len(),
        scannable(bytes@),
    ensures
        start < end <= bytes.len(),
        end == bytes.len() || (start + 2 <= end && bytes@[end - 1] == quote),
))]
pub(crate) fn skip_string(bytes: &[u8], start: usize, quote: u8) -> usize {
    let mut index = start + 1;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            start < index <= bytes.len() + 1,
            scannable(bytes@),
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
/// then end. Verified: a returned offset is at or after `from` and the
/// closing delimiter occurs there.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        scannable(bytes@),
    ensures
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
/// before any expression parsing. Verified: a returned end is exactly the end
/// of such a tag ([`is_basic_tag`]).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        scannable(bytes@),
        from <= bytes.len(),
    ensures
        found matches Some(end) ==> from < end <= bytes.len() && is_basic_tag(
            bytes@,
            from as int,
            name@,
            end as int,
        ),
))]
pub(crate) fn basic_tag_end(bytes: &[u8], from: usize, name: &[u8]) -> Option<usize> {
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
    if !starts_with_at(bytes, index, &statement_close) {
        return None;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        let close = index as int;
        assert(statement_close@ =~= statement_close_bytes());
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
/// section; every other byte is data. Verified: a returned end is after
/// `from`, at most the buffer length, and ends a basic `endraw` tag that
/// starts after a `{%` at or after `from`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        scannable(bytes@),
        from <= bytes.len(),
    ensures
        found matches Some(end) ==> from < end <= bytes.len() && exists|opening: int|
            from <= opening && #[trigger] occurs_at(bytes@, opening, statement_open_bytes())
                && is_basic_tag(bytes@, opening + 2, b"endraw"@, end as int),
))]
pub(crate) fn raw_section_end(bytes: &[u8], from: usize) -> Option<usize> {
    let mut search = from;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            from <= search <= bytes.len(),
            scannable(bytes@),
        decreases bytes.len() - search,
    ))]
    loop {
        let statement_open = [b'{', b'%'];
        let opening = find_pattern(bytes, search, &statement_open)?;
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(statement_open@ =~= statement_open_bytes());
        }
        let after_opening = opening + 2;
        if let Some(end) = basic_tag_end(bytes, after_opening, b"endraw") {
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
/// the remainder itself. Verified: a returned span starts at or after
/// `position`, opens with `{{` (expressions) or `{%` (statements), closes with
/// the matching `}}` or `%}`, holds both delimiters (`start + 4 <= end`), and
/// ends inside the buffer, so a scan that resumes at `end` makes progress.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        scannable(bytes@),
        position <= bytes.len(),
    ensures
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
        decreases bytes.len() - search,
    ))]
    loop {
        let (start, opening) = next_tag_opening(bytes, search)?;
        let body_start = start + 2;
        match opening {
            TagOpening::Comment => {
                let comment_close = [b'#', b'}'];
                let close = find_pattern(bytes, body_start, &comment_close)?;
                search = close + 2;
            }
            TagOpening::Statement => {
                if let Some(after) = basic_tag_end(bytes, body_start, b"raw") {
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
