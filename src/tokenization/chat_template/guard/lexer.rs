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
//! - [`lex_spans`] decides the body's tokenization. On success its spans are
//!   non-empty, ordered, and disjoint, every byte outside them is ASCII
//!   whitespace, and each span is of its kind's lexical class: a quoted
//!   string closed exactly where the string scan closes it ([`string_close`]),
//!   a number ending exactly where MiniJinja's number state machine stops
//!   ([`number_stops_at`]), a maximal identifier, or a one- or two-byte symbol
//!   from the tables above, two-byte symbols first. These classes leave no
//!   choice, so the result is the body's only tokenization
//!   ([`lemma_tokenization_unique`]) and the guard sees exactly the operator
//!   tokens of that tokenization: no operator byte can hide inside a name, a
//!   string, or a number, and none can be invented by splitting one. An error
//!   means the body has no tokenization at all ([`lemma_no_tokenization`]);
//!   it names an unterminated quote or a byte no token can start with;
//! - [`string_end`] returns exactly [`string_close`]: just past the first
//!   quote equal to the opening one that no backslash escapes, or `None`;
//! - [`number_end`] stops exactly where the transcribed state machine
//!   ([`number_step`]) stops and reports a float exactly for a fraction or
//!   exponent final state. Consequently the number consumes at least the
//!   leading digit and only number characters, accepts a sign only after an
//!   exponent letter, a point only once and before any exponent (and never
//!   in a prefixed number), at most one decimal exponent, and is a float
//!   exactly when it is decimal with a `.` or exponent.

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
pub(crate) enum NumberState {
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
            &&& string_close(bytes, bytes[start], start + 1) == Some(end)
        },
        Kind::Integer | Kind::Float => {
            let decimal = number_radix(bytes, start) == 10;
            &&& spec_digit(bytes[start])
            &&& number_stops_at(bytes, start, end)
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

/// Where a string literal closed by `quote` ends, scanning its body from `index`.
///
/// A transcription of MiniJinja's string scan: a backslash skips the byte
/// after it, the first unescaped `quote` closes the literal (the result is
/// the offset just past it), and running off the end leaves the literal
/// unterminated (`None`).
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

/// An ASCII hexadecimal letter, `a-f` or `A-F`.
pub(crate) open spec fn spec_hex_letter(byte: u8) -> bool {
    (97 <= byte <= 102) || (65 <= byte <= 70)
}

/// MiniJinja's number-scanner transition on `byte`, or `None` where the number ends.
///
/// A point starts the fraction of an integer; `e`/`E` starts the exponent of
/// an integer or fraction; a sign or digit right after the exponent letter
/// moves past it; digits and `_` (and, after a `0x` prefix, hexadecimal
/// letters) keep the state; every other byte ends the number.
pub(crate) open spec fn number_step(state: NumberState, radix: int, byte: u8) -> Option<NumberState> {
    if byte == 46 && state == NumberState::Integer {
        Some(NumberState::Fraction)
    } else if (byte == 69 || byte == 101) && (state == NumberState::Integer || state
        == NumberState::Fraction) {
        Some(NumberState::Exponent)
    } else if (byte == 43 || byte == 45 || spec_digit(byte)) && state == NumberState::Exponent {
        Some(NumberState::ExponentSign)
    } else if spec_digit(byte) || (radix == 16 && state == NumberState::Radix && spec_hex_letter(
        byte,
    )) || byte == 95 {
        Some(state)
    } else {
        None
    }
}

/// Where the scanner starts consuming bytes: after the radix prefix, if any.
pub(crate) open spec fn number_base(bytes: Seq<u8>, start: int) -> int {
    if number_radix(bytes, start) == 10 {
        start
    } else {
        start + 2
    }
}

/// The scanner's state after consuming `bytes[number_base..end]`, or `None` if it stops earlier.
///
/// It starts in `Integer` (decimal) or `Radix` (after a prefix) and applies
/// [`number_step`] to each byte in turn.
pub(crate) open spec fn number_run(bytes: Seq<u8>, start: int, end: int) -> Option<NumberState>
    decreases end - number_base(bytes, start),
{
    let radix = number_radix(bytes, start);
    let base = number_base(bytes, start);
    if end < base {
        None
    } else if end == base {
        Some(
            if radix == 10 {
                NumberState::Integer
            } else {
                NumberState::Radix
            },
        )
    } else {
        match number_run(bytes, start, end - 1) {
            Some(state) => number_step(state, radix, bytes[end - 1]),
            None => None,
        }
    }
}

/// MiniJinja's scanner, run from `start`, consumes exactly `bytes[start..end]`.
///
/// It accepts every byte before `end`, and `end` is the end of the body or a
/// byte its state cannot take ([`lemma_number_end_unique`]: at most one such
/// `end` exists).
pub(crate) open spec fn number_stops_at(bytes: Seq<u8>, start: int, end: int) -> bool {
    &&& end <= bytes.len()
    &&& match number_run(bytes, start, end) {
        Some(state) => end == bytes.len() || number_step(
            state,
            number_radix(bytes, start),
            bytes[end],
        ).is_none(),
        None => false,
    }
}

/// The scanner's final state for the number `bytes[start..end]` is a fraction or exponent state.
pub(crate) open spec fn number_is_float(bytes: Seq<u8>, start: int, end: int) -> bool {
    match number_run(bytes, start, end) {
        Some(state) => state != NumberState::Integer && state != NumberState::Radix,
        None => false,
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
/// A backslash escapes the next byte. Verified: the result is exactly
/// [`string_close`] of the body, so the literal closes at the first quote
/// equal to the opening one that no backslash escapes, and `None` means no
/// such quote exists; a returned end is at least two past `start` and inside
/// the buffer.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(end =>
    requires
        start < bytes.len(),
        lexable(bytes@),
    ensures
        match end {
            Some(end) => string_close(bytes@, bytes@[start as int], start + 1) == Some(end as int),
            None => string_close(bytes@, bytes@[start as int], start + 1).is_none(),
        },
        end matches Some(end) ==> start + 2 <= end <= bytes.len() && bytes@[end - 1] == bytes@[start as int],
))]
pub(crate) fn string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let quote = bytes[start];
    let mut index = start + 1;
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_string_close_bounds(bytes@, quote, start + 1);
    }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            start < index <= bytes.len() + 1,
            lexable(bytes@),
            quote == bytes@[start as int],
            string_close(bytes@, quote, index as int) == string_close(bytes@, quote, start + 1),
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
/// Verified: the end is exactly where MiniJinja's state machine
/// ([`number_step`]) stops, so every byte before it is accepted and the byte
/// at it is not ([`number_stops_at`], which at most one end satisfies), and
/// the float flag is exactly whether the final state is a fraction or
/// exponent. Consequently the number covers at least its leading digit and
/// contains only number characters, with a sign only after an exponent
/// letter, at most one point and only before any exponent (none in a
/// prefixed number), and at most one decimal exponent; it is a float exactly
/// when it is decimal and contains a `.` or exponent.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(number =>
    requires
        start < bytes.len(),
        spec_digit(bytes@[start as int]),
        lexable(bytes@),
    ensures
        number_stops_at(bytes@, start as int, number.0 as int),
        number.1 == number_is_float(bytes@, start as int, number.0 as int),
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
            number_run(bytes@, start as int, index as int) == Some(state),
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
            number_base(bytes@, start as int) <= index,
            number_run(bytes@, start as int, index as int) == Some(state),
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
        ensures
            index == bytes.len() || number_step(state, radix as int, bytes@[index as int]).is_none(),
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

/// A scanner run that consumes `bytes[..end]` also consumes every shorter prefix past the base.
pub(crate) proof fn lemma_number_run_prefix(bytes: Seq<u8>, start: int, end: int, middle: int)
    requires
        number_run(bytes, start, end).is_some(),
        number_base(bytes, start) <= middle <= end,
    ensures
        number_run(bytes, start, middle).is_some(),
    decreases end - middle,
{
    if middle < end {
        lemma_number_run_prefix(bytes, start, end - 1, middle);
    }
}

/// The scanner stops at one place only: [`number_stops_at`] determines the number's end.
pub(crate) proof fn lemma_number_end_unique(bytes: Seq<u8>, start: int, first: int, second: int)
    requires
        number_stops_at(bytes, start, first),
        number_stops_at(bytes, start, second),
    ensures
        first == second,
{
    if first < second {
        lemma_number_run_prefix(bytes, start, second, first + 1);
    } else if second < first {
        lemma_number_run_prefix(bytes, start, first, second + 1);
    }
}

} // verus!

/// Splits a tag body into token spans exactly as MiniJinja 2.19's block lexer does.
///
/// Verified: on success the spans tokenize the whole body (see the module
/// documentation): ordered, disjoint, non-empty tokens of their kinds'
/// lexical classes, separated only by ASCII whitespace, and they are the
/// body's only tokenization ([`lemma_tokenization_unique`]). An error means
/// the body has no tokenization at all; it names an unterminated string's
/// opening quote or a byte no token starts with.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(lexed =>
    requires
        lexable(bytes@),
    ensures
        match lexed {
            Ok(spans) => {
                &&& tokenizes(bytes@, spans@, bytes.len() as int)
                &&& forall|other: Seq<Span>| #[trigger] tokenizes(bytes@, other, bytes.len() as int)
                    ==> other == spans@
            },
            Err(_) => forall|other: Seq<Span>| !#[trigger] tokenizes(
                bytes@,
                other,
                bytes.len() as int,
            ),
        },
        match lexed {
            Ok(_) => true,
            Err(LexError::UnterminatedString { start }) => {
                &&& start < bytes.len()
                &&& (bytes@[start as int] == 39 || bytes@[start as int] == 34)
                &&& string_close(bytes@, bytes@[start as int], start + 1).is_none()
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
                None => {
                    #[cfg(verus_keep_ghost)]
                    proof! {
                        lemma_no_tokenization(bytes@, spans@, start as int);
                    }
                    return Err(LexError::UnterminatedString { start });
                }
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
                    spec_name_byte(bytes@[start as int]),
                    forall|i: int| start <= i < index ==> spec_name_byte(#[trigger] bytes@[i]),
                decreases bytes.len() - index,
            ))]
            while index < bytes.len() && is_name_byte(bytes[index]) {
                index += 1;
            }
            Kind::Name
        } else {
            #[cfg(verus_keep_ghost)]
            proof! {
                lemma_no_tokenization(bytes@, spans@, start as int);
            }
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
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|other: Seq<Span>| #[trigger] tokenizes(bytes@, other, bytes.len() as int)
            implies other == spans@ by {
            lemma_tokenization_unique(bytes@, spans@, other);
        }
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

/// A token is determined by where it starts.
///
/// The first byte fixes the kind (a two-byte symbol, a quote, a digit, a
/// one-byte symbol, or a name byte), and each kind fixes the end: a symbol's
/// width, the string scan's close, the number scanner's stop
/// ([`lemma_number_end_unique`]), or a name's maximal run.
pub(crate) proof fn lemma_token_unique(bytes: Seq<u8>, first: Span, second: Span)
    requires
        is_token(bytes, first),
        is_token(bytes, second),
        first.start == second.start,
    ensures
        first == second,
{
    let start = first.start as int;
    if first.kind == Kind::Name && second.kind == Kind::Name {
        if first.end < second.end {
            assert(spec_name_byte(bytes[first.end as int]));
        } else if second.end < first.end {
            assert(spec_name_byte(bytes[second.end as int]));
        }
    }
    if (first.kind == Kind::Integer || first.kind == Kind::Float) && (second.kind == Kind::Integer
        || second.kind == Kind::Float) {
        lemma_number_end_unique(bytes, start, first.end as int, second.end as int);
    }
}

/// Two token starts reached from `from` across whitespace only are the same offset.
pub(crate) proof fn lemma_token_start_unique(bytes: Seq<u8>, from: int, first: Span, second: Span)
    requires
        is_token(bytes, first),
        is_token(bytes, second),
        from <= first.start,
        from <= second.start,
        all_whitespace(bytes, from, first.start as int),
        all_whitespace(bytes, from, second.start as int),
    ensures
        first.start == second.start,
{
    if first.start < second.start {
        assert(spec_space(bytes[first.start as int]));
    } else if second.start < first.start {
        assert(spec_space(bytes[second.start as int]));
    }
}

/// Two tokenizations of prefixes of the same body agree on their first `count` tokens.
pub(crate) proof fn lemma_tokenization_prefix(
    bytes: Seq<u8>,
    first: Seq<Span>,
    first_covered: int,
    second: Seq<Span>,
    second_covered: int,
    count: int,
)
    requires
        tokenizes(bytes, first, first_covered),
        tokenizes(bytes, second, second_covered),
        0 <= count <= first.len(),
        count <= second.len(),
    ensures
        forall|i: int| 0 <= i < count ==> #[trigger] first[i] == second[i],
    decreases count,
{
    if count > 0 {
        let last = count - 1;
        lemma_tokenization_prefix(bytes, first, first_covered, second, second_covered, last);
        if last > 0 {
            assert(first[last - 1] == second[last - 1]);
        }
        assert(previous_end(first, last) == previous_end(second, last));
        assert(is_token(bytes, first[last]));
        assert(is_token(bytes, second[last]));
        lemma_token_start_unique(bytes, previous_end(first, last), first[last], second[last]);
        lemma_token_unique(bytes, first[last], second[last]);
    }
}

/// A body has exactly one tokenization: [`lex_spans`]'s result is the only one.
pub(crate) proof fn lemma_tokenization_unique(bytes: Seq<u8>, first: Seq<Span>, second: Seq<Span>)
    requires
        tokenizes(bytes, first, bytes.len() as int),
        tokenizes(bytes, second, bytes.len() as int),
    ensures
        first == second,
{
    let shared = if first.len() <= second.len() {
        first.len() as int
    } else {
        second.len() as int
    };
    let covered = bytes.len() as int;
    lemma_tokenization_prefix(bytes, first, covered, second, covered, shared);
    if shared > 0 {
        assert(first[shared - 1] == second[shared - 1]);
    }
    assert(previous_end(first, shared) == previous_end(second, shared));
    if first.len() < second.len() {
        assert(is_token(bytes, second[shared]));
        assert(previous_end(second, shared) <= second[shared].start);
        assert(spec_space(bytes[second[shared].start as int]));
    } else if second.len() < first.len() {
        assert(is_token(bytes, first[shared]));
        assert(previous_end(first, shared) <= first[shared].start);
        assert(spec_space(bytes[first[shared].start as int]));
    }
    assert(first =~= second);
}

/// A body has no tokenization when a prefix tokenizes up to a non-space byte no token can start with.
///
/// Any tokenization of the whole body agrees with the prefix's tokens
/// ([`lemma_tokenization_prefix`]) and must then place a token at `start`.
pub(crate) proof fn lemma_no_tokenization(bytes: Seq<u8>, spans: Seq<Span>, start: int)
    requires
        tokenizes(bytes, spans, start),
        0 <= start < bytes.len(),
        !spec_space(bytes[start]),
        forall|span: Span| span.start == start ==> !#[trigger] is_token(bytes, span),
    ensures
        forall|other: Seq<Span>| !#[trigger] tokenizes(bytes, other, bytes.len() as int),
{
    assert forall|other: Seq<Span>| !#[trigger] tokenizes(bytes, other, bytes.len() as int) by {
        if tokenizes(bytes, other, bytes.len() as int) {
            let count = spans.len() as int;
            if other.len() < count {
                let shared = other.len() as int;
                lemma_tokenization_prefix(bytes, spans, start, other, bytes.len() as int, shared);
                if shared > 0 {
                    assert(spans[shared - 1] == other[shared - 1]);
                }
                assert(previous_end(spans, shared) == previous_end(other, shared));
                assert(is_token(bytes, spans[shared]));
                assert(previous_end(spans, shared) <= spans[shared].start);
                assert(spec_space(bytes[spans[shared].start as int]));
            } else {
                lemma_tokenization_prefix(bytes, spans, start, other, bytes.len() as int, count);
                if count > 0 {
                    assert(spans[count - 1] == other[count - 1]);
                }
                assert(previous_end(spans, count) == previous_end(other, count));
                if other.len() == count {
                    assert(spec_space(bytes[start]));
                } else {
                    let next = other[count];
                    assert(is_token(bytes, next));
                    assert(previous_end(other, count) <= next.start);
                    if next.start < start {
                        assert(spec_space(bytes[next.start as int]));
                    } else if start < next.start {
                        assert(spec_space(bytes[start]));
                    }
                }
            }
        }
    }
}

} // verus!

#[cfg(test)]
#[path = "lexer_tests.rs"]
mod tests;
