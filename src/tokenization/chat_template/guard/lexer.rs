//! Dependency-free byte lexer of the operator guard, mirroring MiniJinja 2.19.
//!
//! The guard parses each tag body to find every `+`, `~`, `*`, `**`, and `:`
//! it must route through a bounded filter. Its lexer is pure index arithmetic
//! over the body's bytes and lives here so Verus can check it:
//!
//! - ASCII whitespace separates tokens;
//! - two-byte symbols (`//`, `**`, `==`, `!=`, `>=`, `<=`) win over one-byte
//!   ones (`+-*/%.,:~|=><()[]{}`);
//! - strings honour backslash escapes ([`string_end`]);
//! - numbers follow MiniJinja's state machine ([`number_end`]): digits, `_`,
//!   one `.` fraction, an exponent with an optional sign, radix prefixes;
//! - identifiers are ASCII letters, digits, and `_` (MiniJinja is built
//!   without its `unicode` feature).
//!
//! # Verified behavior
//!
//! `verus/tokenization.rs` checks these functions in place. For every input
//! the proofs establish that no index leaves the buffer and that:
//!
//! - [`lex_spans`] partitions the body: its spans are non-empty, ordered, and
//!   disjoint, every byte outside them is ASCII whitespace, and each span is
//!   of its kind's lexical class (a quoted string, a number starting with a
//!   digit, a maximal identifier, or a one- or two-byte symbol from the
//!   tables above), so no operator byte can hide inside a name; an error
//!   names an unterminated quote or a byte no token can start with;
//! - [`string_end`] ends just past the matching quote;
//! - [`number_end`] consumes at least the leading digit and only number
//!   characters, accepts a sign only after an exponent letter, a point only
//!   once and before any exponent (and never in a prefixed number), at most
//!   one decimal exponent, and reports a float exactly for a decimal number
//!   with a `.` or exponent.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// The kinds of token the expression grammar distinguishes.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// An identifier or keyword such as `name`, `if`, or `not`.
    Name,
    /// A string literal, quotes included.
    String,
    /// An integer literal (decimal or with a `0b`/`0o`/`0x` prefix).
    Integer,
    /// A floating-point literal (with a fraction or exponent).
    Float,
    /// An operator or punctuation, such as `+`, `**`, `(`, or `==`.
    Symbol,
}

/// One token's kind and byte range in the tag body.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Span {
    /// What the token is.
    pub(crate) kind: Kind,
    /// Offset of the token's first byte.
    pub(crate) start: usize,
    /// Offset just past the token's last byte.
    pub(crate) end: usize,
}

/// Why a tag body cannot be lexed.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LexError {
    /// A string literal opened at `start` never closes.
    UnterminatedString {
        /// Offset of the opening quote.
        start: usize,
    },
    /// No token can start with the byte at `start`.
    UnexpectedCharacter {
        /// Offset of the byte.
        start: usize,
    },
}

/// States of MiniJinja's number scanner.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberState {
    /// After a `0b`, `0o`, or `0x` prefix.
    Radix,
    /// Integer digits.
    Integer,
    /// After the decimal point.
    Fraction,
    /// Just after `e` or `E`.
    Exponent,
    /// After the exponent's sign or first digit.
    ExponentSign,
}

#[cfg(verus_keep_ghost)]
verus! {

/// Longest buffer the lexer accepts: Rust allocations never exceed `isize::MAX` bytes.
pub(crate) open spec fn lexable(bytes: Seq<u8>) -> bool {
    bytes.len() <= isize::MAX
}

/// ASCII whitespace as `u8::is_ascii_whitespace` defines it: space, tab, LF, FF, CR.
pub(crate) open spec fn spec_space(byte: u8) -> bool {
    byte == 32 || byte == 9 || byte == 10 || byte == 12 || byte == 13
}

/// An ASCII decimal digit.
pub(crate) open spec fn spec_digit(byte: u8) -> bool {
    48 <= byte <= 57
}

/// An ASCII letter.
pub(crate) open spec fn spec_letter(byte: u8) -> bool {
    (65 <= byte <= 90) || (97 <= byte <= 122)
}

/// A byte that may continue an identifier: an ASCII letter, digit, or `_`.
pub(crate) open spec fn spec_name_byte(byte: u8) -> bool {
    spec_letter(byte) || spec_digit(byte) || byte == 95
}

/// One of MiniJinja's one-byte symbols, `+-*/%.,:~|=><()[]{}`.
pub(crate) open spec fn spec_single_symbol(byte: u8) -> bool {
    byte == 43 || byte == 45 || byte == 42 || byte == 47 || byte == 37 || byte == 46 || byte == 44
        || byte == 58 || byte == 126 || byte == 124 || byte == 61 || byte == 62 || byte == 60
        || byte == 40 || byte == 41 || byte == 91 || byte == 93 || byte == 123 || byte == 125
}

/// One of MiniJinja's two-byte symbols at `index`: `//`, `**`, `==`, `!=`, `>=`, `<=`.
pub(crate) open spec fn spec_double_symbol(bytes: Seq<u8>, index: int) -> bool {
    &&& 0 <= index
    &&& index + 1 < bytes.len()
    &&& {
        let (first, second) = (bytes[index], bytes[index + 1]);
        ||| first == 47 && second == 47
        ||| first == 42 && second == 42
        ||| second == 61 && (first == 61 || first == 33 || first == 62 || first == 60)
    }
}

/// A byte a number may contain: a digit, `_`, `.`, `+`, `-`, a letter `a-f`/`A-F`, or a radix letter.
pub(crate) open spec fn spec_number_byte(byte: u8) -> bool {
    ||| spec_digit(byte)
    ||| byte == 95 || byte == 46 || byte == 43 || byte == 45
    ||| (97 <= byte <= 102) || (65 <= byte <= 70)
    ||| byte == 98 || byte == 66 || byte == 111 || byte == 79 || byte == 120 || byte == 88
}

/// The radix a number starting at `start` declares: 2, 8, or 16 with a prefix, else 10.
pub(crate) open spec fn number_radix(bytes: Seq<u8>, start: int) -> int {
    if start + 2 <= bytes.len() && bytes[start] == 48 {
        let letter = bytes[start + 1];
        if letter == 98 || letter == 66 {
            2
        } else if letter == 111 || letter == 79 {
            8
        } else if letter == 120 || letter == 88 {
            16
        } else {
            10
        }
    } else {
        10
    }
}

/// Some byte in `from..to` is a decimal point or an exponent letter.
pub(crate) open spec fn has_float_mark(bytes: Seq<u8>, from: int, to: int) -> bool {
    exists|i: int| from <= i < to && #[trigger] is_float_mark(bytes[i])
}

/// `.`, `e`, or `E`.
pub(crate) open spec fn is_float_mark(byte: u8) -> bool {
    byte == 46 || byte == 101 || byte == 69
}

/// Some byte in `from..to` is an exponent letter `e` or `E`.
pub(crate) open spec fn has_exponent_mark(bytes: Seq<u8>, from: int, to: int) -> bool {
    exists|i: int| from <= i < to && (#[trigger] bytes[i] == 101 || bytes[i] == 69)
}

/// Some byte in `from..to` is a decimal point.
pub(crate) open spec fn has_point(bytes: Seq<u8>, from: int, to: int) -> bool {
    exists|i: int| from <= i < to && #[trigger] bytes[i] == 46
}

/// The number text in `from..to` has the shape of a float: a decimal point
/// only before any exponent and at most once, and at most one exponent letter
/// (when decimal); a non-decimal number has no point at all.
pub(crate) open spec fn has_literal_shape(bytes: Seq<u8>, from: int, to: int, decimal: bool) -> bool {
    &&& forall|j: int| from <= j < to && #[trigger] bytes[j] == 46
        ==> decimal && !has_point(bytes, from, j) && !has_exponent_mark(bytes, from, j)
    &&& decimal ==> forall|j: int| from <= j < to && (#[trigger] bytes[j] == 101 || bytes[j] == 69)
        ==> !has_exponent_mark(bytes, from, j)
}

/// Every byte in `from..to` is a number byte, and a sign appears only after an exponent letter.
pub(crate) open spec fn is_number_text(bytes: Seq<u8>, from: int, to: int) -> bool {
    forall|i: int| from <= i < to ==> {
        &&& spec_number_byte(#[trigger] bytes[i])
        &&& (bytes[i] == 43 || bytes[i] == 45) ==> has_exponent_mark(bytes, from, i)
    }
}

/// `span` is one token of its kind's lexical class.
pub(crate) open spec fn is_token(bytes: Seq<u8>, span: Span) -> bool {
    let (start, end) = (span.start as int, span.end as int);
    &&& start < end <= bytes.len()
    &&& match span.kind {
        Kind::Name => {
            &&& (spec_letter(bytes[start]) || bytes[start] == 95)
            &&& forall|i: int| start <= i < end ==> spec_name_byte(#[trigger] bytes[i])
            &&& (end < bytes.len() ==> !spec_name_byte(bytes[end]))
        },
        Kind::String => {
            &&& (bytes[start] == 39 || bytes[start] == 34)
            &&& start + 2 <= end
            &&& bytes[end - 1] == bytes[start]
        },
        Kind::Integer | Kind::Float => {
            let decimal = number_radix(bytes, start) == 10;
            &&& spec_digit(bytes[start])
            &&& is_number_text(bytes, start, end)
            &&& has_literal_shape(bytes, start, end, decimal)
            &&& (span.kind == Kind::Float) == (decimal && has_float_mark(bytes, start, end))
        },
        Kind::Symbol => {
            ||| end == start + 2 && spec_double_symbol(bytes, start)
            ||| end == start + 1 && spec_single_symbol(bytes[start]) && !spec_double_symbol(bytes, start)
        },
    }
}

/// Bytes `from..to` are all ASCII whitespace.
pub(crate) open spec fn all_whitespace(bytes: Seq<u8>, from: int, to: int) -> bool {
    forall|i: int| from <= i < to ==> spec_space(#[trigger] bytes[i])
}

/// The end of the span before span `index`, or zero for the first.
pub(crate) open spec fn previous_end(spans: Seq<Span>, index: int) -> int {
    if index == 0 {
        0
    } else {
        spans[index - 1].end as int
    }
}

/// `spans` tokenize `bytes[..covered]`: ordered tokens separated only by whitespace.
pub(crate) open spec fn tokenizes(bytes: Seq<u8>, spans: Seq<Span>, covered: int) -> bool {
    &&& forall|i: int| 0 <= i < spans.len() ==> #[trigger] is_token(bytes, spans[i])
    &&& forall|i: int| 0 <= i < spans.len() ==> previous_end(spans, i) <= #[trigger] spans[i].start
    &&& forall|i: int| 0 <= i < spans.len()
        ==> all_whitespace(bytes, previous_end(spans, i), #[trigger] spans[i].start as int)
    &&& previous_end(spans, spans.len() as int) <= covered
    &&& all_whitespace(bytes, previous_end(spans, spans.len() as int), covered)
}

} // verus!

/// Whether `byte` is ASCII whitespace (`u8::is_ascii_whitespace`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(space => ensures space == spec_space(byte)))]
fn is_space(byte: u8) -> bool {
    byte == b' ' || byte == b'\t' || byte == b'\n' || byte == 0x0c || byte == b'\r'
}

/// Whether `byte` is an ASCII digit (`u8::is_ascii_digit`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(digit => ensures digit == spec_digit(byte)))]
// Verus has no specification for `u8::is_ascii_digit` or `RangeInclusive::contains`.
#[allow(clippy::manual_is_ascii_check, clippy::manual_range_contains)]
fn is_digit(byte: u8) -> bool {
    b'0' <= byte && byte <= b'9'
}

/// Whether `byte` may continue an identifier (`_` or `u8::is_ascii_alphanumeric`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(name => ensures name == spec_name_byte(byte)))]
// Verus has no specification for `u8::is_ascii_alphanumeric` or `RangeInclusive::contains`.
#[allow(clippy::manual_is_ascii_check, clippy::manual_range_contains)]
fn is_name_byte(byte: u8) -> bool {
    byte == b'_'
        || (b'a' <= byte && byte <= b'z')
        || (b'A' <= byte && byte <= b'Z')
        || (b'0' <= byte && byte <= b'9')
}

/// Whether `byte` is one of MiniJinja's one-byte symbols.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(symbol => ensures symbol == spec_single_symbol(byte)))]
fn is_single_symbol(byte: u8) -> bool {
    matches!(
        byte,
        b'+' | b'-'
            | b'*'
            | b'/'
            | b'%'
            | b'.'
            | b','
            | b':'
            | b'~'
            | b'|'
            | b'='
            | b'>'
            | b'<'
            | b'('
            | b')'
            | b'['
            | b']'
            | b'{'
            | b'}'
    )
}

/// Whether a two-byte symbol (`//`, `**`, `==`, `!=`, `>=`, `<=`) starts at `index`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(symbol =>
    requires
        lexable(bytes@),
    ensures
        symbol == spec_double_symbol(bytes@, index as int),
))]
fn is_double_symbol(bytes: &[u8], index: usize) -> bool {
    if index >= bytes.len() || index + 1 >= bytes.len() {
        return false;
    }
    let (first, second) = (bytes[index], bytes[index + 1]);
    (first == b'/' && second == b'/')
        || (first == b'*' && second == b'*')
        || (second == b'=' && matches!(first, b'=' | b'!' | b'>' | b'<'))
}

/// Returns the index just past the string literal opened at `start`, or `None` if unterminated.
///
/// A backslash escapes the next byte. Verified: a returned end is at least
/// two past `start`, inside the buffer, and just past a byte equal to the
/// opening quote.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(end =>
    requires
        start < bytes.len(),
        lexable(bytes@),
    ensures
        end matches Some(end) ==> start + 2 <= end <= bytes.len() && bytes@[end - 1] == bytes@[start as int],
))]
pub(crate) fn string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let quote = bytes[start];
    let mut index = start + 1;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            start < index <= bytes.len() + 1,
            lexable(bytes@),
            quote == bytes@[start as int],
        decreases bytes.len() + 1 - index,
    ))]
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            index += 2;
        } else if bytes[index] == quote {
            return Some(index + 1);
        } else {
            index += 1;
        }
    }
    None
}

/// Returns the end of the number starting at `start` and whether it is a float.
///
/// Mirrors MiniJinja's `eat_number`: a radix prefix allows only digits (and
/// hexadecimal letters for `0x`); a decimal number allows one fraction, then
/// an exponent whose first character may be a sign; `_` is allowed anywhere.
/// Verified: the number covers at least its leading digit and contains only
/// number characters, with a sign only after an exponent letter, at most one
/// point and only before any exponent (none in a prefixed number), and at
/// most one decimal exponent; it is a float exactly when it is decimal and
/// contains a `.` or exponent.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(number =>
    requires
        start < bytes.len(),
        spec_digit(bytes@[start as int]),
        lexable(bytes@),
    ensures
        start < number.0 <= bytes.len(),
        is_number_text(bytes@, start as int, number.0 as int),
        number_radix(bytes@, start as int) == 10 ==> number.1 == has_float_mark(
            bytes@,
            start as int,
            number.0 as int,
        ),
        number_radix(bytes@, start as int) != 10 ==> !number.1,
        has_literal_shape(
            bytes@,
            start as int,
            number.0 as int,
            number_radix(bytes@, start as int) == 10,
        ),
))]
pub(crate) fn number_end(bytes: &[u8], start: usize) -> (usize, bool) {
    let radix = if start + 2 <= bytes.len() && bytes[start] == b'0' {
        match bytes[start + 1] {
            b'b' | b'B' => 2,
            b'o' | b'O' => 8,
            b'x' | b'X' => 16,
            _ => 10,
        }
    } else {
        10
    };
    let (mut index, mut state) = if radix == 10 {
        (start, NumberState::Integer)
    } else {
        (start + 2, NumberState::Radix)
    };
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            start <= index <= bytes.len(),
            radix == number_radix(bytes@, start as int),
            radix == 10 ==> state == NumberState::Integer,
            radix != 10 ==> state == NumberState::Radix && start + 2 <= index,
            radix == 10 ==> index == start || start < index,
            is_number_text(bytes@, start as int, index as int),
            !has_float_mark(bytes@, start as int, index as int) || radix != 10,
            has_literal_shape(bytes@, start as int, index as int, radix == 10),
            radix == 10 ==> !has_point(bytes@, start as int, index as int),
            radix == 10 ==> !has_exponent_mark(bytes@, start as int, index as int),
        decreases bytes.len() - index,
    ))]
    while index < bytes.len() && is_digit(bytes[index]) {
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(start < index);
    }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            start < index <= bytes.len(),
            radix == number_radix(bytes@, start as int),
            radix != 10 ==> state == NumberState::Radix,
            radix == 10 ==> state != NumberState::Radix,
            is_number_text(bytes@, start as int, index as int),
            radix == 10 ==> ((state != NumberState::Integer) == has_float_mark(
                bytes@,
                start as int,
                index as int,
            )),
            (state == NumberState::Exponent || state == NumberState::ExponentSign)
                ==> has_exponent_mark(bytes@, start as int, index as int),
            has_literal_shape(bytes@, start as int, index as int, radix == 10),
            radix == 10 && state == NumberState::Integer ==> !has_point(
                bytes@,
                start as int,
                index as int,
            ) && !has_exponent_mark(bytes@, start as int, index as int),
            radix == 10 && state == NumberState::Fraction ==> !has_exponent_mark(
                bytes@,
                start as int,
                index as int,
            ),
        decreases bytes.len() - index,
    ))]
    while index < bytes.len() {
        let byte = bytes[index];
        state = if byte == b'.' && matches!(state, NumberState::Integer) {
            NumberState::Fraction
        } else if (byte == b'E' || byte == b'e')
            && matches!(state, NumberState::Integer | NumberState::Fraction)
        {
            NumberState::Exponent
        } else if (byte == b'+' || byte == b'-' || is_digit(byte))
            && matches!(state, NumberState::Exponent)
        {
            // The exponent's sign or first digit.
            NumberState::ExponentSign
        } else if is_digit(byte)
            || (radix == 16
                && matches!(state, NumberState::Radix)
                && matches!(byte, b'a'..=b'f' | b'A'..=b'F'))
            || byte == b'_'
        {
            state
        } else {
            break;
        };
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_extend_number_text(bytes@, start as int, index as int);
        }
        index += 1;
    }
    (
        index,
        !matches!(state, NumberState::Integer | NumberState::Radix),
    )
}

#[cfg(verus_keep_ghost)]
verus! {

/// Extending number text by one byte keeps earlier facts and adds the new byte's.
pub(crate) proof fn lemma_extend_number_text(bytes: Seq<u8>, from: int, to: int)
    requires
        from <= to < bytes.len(),
    ensures
        has_float_mark(bytes, from, to + 1) == (has_float_mark(bytes, from, to) || is_float_mark(bytes[to])),
        has_exponent_mark(bytes, from, to + 1) == (has_exponent_mark(bytes, from, to) || bytes[to] == 101
            || bytes[to] == 69),
        has_exponent_mark(bytes, from, to) ==> has_exponent_mark(bytes, from, to + 1),
        has_point(bytes, from, to + 1) == (has_point(bytes, from, to) || bytes[to] == 46),
{
    if has_point(bytes, from, to + 1) && bytes[to] != 46 {
        let i = choose|i: int| from <= i < to + 1 && #[trigger] bytes[i] == 46;
        assert(i < to);
    }
    if has_float_mark(bytes, from, to + 1) && !is_float_mark(bytes[to]) {
        let i = choose|i: int| from <= i < to + 1 && #[trigger] is_float_mark(bytes[i]);
        assert(i < to);
    }
    if has_exponent_mark(bytes, from, to + 1) && !(bytes[to] == 101 || bytes[to] == 69) {
        let i = choose|i: int| from <= i < to + 1 && (#[trigger] bytes[i] == 101 || bytes[i] == 69);
        assert(i < to);
    }
}

} // verus!

/// Splits a tag body into token spans exactly as MiniJinja 2.19's block lexer does.
///
/// Verified: on success the spans tokenize the whole body (see the module
/// documentation): ordered, disjoint, non-empty tokens of their kinds'
/// lexical classes, separated only by ASCII whitespace. An error names an
/// unterminated string's opening quote or a byte no token starts with.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(lexed =>
    requires
        lexable(bytes@),
    ensures
        match lexed {
            Ok(spans) => tokenizes(bytes@, spans@, bytes.len() as int),
            Err(LexError::UnterminatedString { start }) => {
                &&& start < bytes.len()
                &&& (bytes@[start as int] == 39 || bytes@[start as int] == 34)
            },
            Err(LexError::UnexpectedCharacter { start }) => {
                &&& start < bytes.len()
                &&& !spec_space(bytes@[start as int])
                &&& !spec_digit(bytes@[start as int])
                &&& !spec_name_byte(bytes@[start as int])
                &&& !spec_single_symbol(bytes@[start as int])
                &&& bytes@[start as int] != 39 && bytes@[start as int] != 34
            },
        },
))]
pub(crate) fn lex_spans(bytes: &[u8]) -> Result<Vec<Span>, LexError> {
    let mut spans: Vec<Span> = Vec::new();
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= bytes.len(),
            lexable(bytes@),
            tokenizes(bytes@, spans@, index as int),
        decreases bytes.len() - index,
    ))]
    while index < bytes.len() {
        let byte = bytes[index];
        if is_space(byte) {
            index += 1;
            continue;
        }
        let start = index;
        let kind = if is_double_symbol(bytes, index) {
            index += 2;
            Kind::Symbol
        } else if byte == b'\'' || byte == b'"' {
            index = match string_end(bytes, index) {
                Some(end) => end,
                None => return Err(LexError::UnterminatedString { start }),
            };
            Kind::String
        } else if is_digit(byte) {
            let (end, float) = number_end(bytes, index);
            index = end;
            if float { Kind::Float } else { Kind::Integer }
        } else if is_single_symbol(byte) {
            index += 1;
            Kind::Symbol
        } else if is_name_byte(byte) {
            #[cfg_attr(verus_keep_ghost, verus_spec(
                invariant
                    start <= index <= bytes.len(),
                    start == index || start < index,
                    spec_name_byte(bytes@[start as int]),
                    forall|i: int| start <= i < index ==> spec_name_byte(#[trigger] bytes@[i]),
                decreases bytes.len() - index,
            ))]
            while index < bytes.len() && is_name_byte(bytes[index]) {
                index += 1;
            }
            Kind::Name
        } else {
            return Err(LexError::UnexpectedCharacter { start });
        };
        let span = Span {
            kind,
            start,
            end: index,
        };
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(is_token(bytes@, span));
            lemma_push_token(bytes@, spans@, span);
        }
        spans.push(span);
    }
    Ok(spans)
}

#[cfg(verus_keep_ghost)]
verus! {

/// Appending a token after whitespace extends a tokenization to the token's end.
pub(crate) proof fn lemma_push_token(bytes: Seq<u8>, spans: Seq<Span>, span: Span)
    requires
        tokenizes(bytes, spans, span.start as int),
        is_token(bytes, span),
    ensures
        tokenizes(bytes, spans.push(span), span.end as int),
{
    let extended = spans.push(span);
    assert(previous_end(extended, spans.len() as int) == previous_end(spans, spans.len() as int));
    assert forall|i: int| 0 <= i < extended.len() implies #[trigger] is_token(bytes, extended[i]) by {
        if i < spans.len() {
            assert(extended[i] == spans[i]);
        }
    }
    assert forall|i: int| 0 <= i < extended.len() implies previous_end(extended, i)
        <= #[trigger] extended[i].start by {
        if i < spans.len() {
            assert(extended[i] == spans[i]);
            assert(previous_end(extended, i) == previous_end(spans, i));
        }
    }
    assert forall|i: int| 0 <= i < extended.len() implies all_whitespace(
        bytes,
        previous_end(extended, i),
        #[trigger] extended[i].start as int,
    ) by {
        if i < spans.len() {
            assert(extended[i] == spans[i]);
            assert(previous_end(extended, i) == previous_end(spans, i));
        }
    }
    assert(previous_end(extended, extended.len() as int) == span.end);
}

} // verus!

#[cfg(test)]
#[path = "lexer_tests.rs"]
mod tests;
