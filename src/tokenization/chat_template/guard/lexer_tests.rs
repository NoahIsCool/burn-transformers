//! Tests for the guard's byte lexer: token classes, numbers, strings, and errors.

use proptest::prelude::*;

use super::*;

/// Lexes `body` into `(kind, text)` pairs.
fn tokens(body: &str) -> Vec<(Kind, &str)> {
    lex_spans(body.as_bytes())
        .unwrap()
        .into_iter()
        .map(|span| (span.kind, &body[span.start..span.end]))
        .collect()
}

/// Bodies split into MiniJinja's tokens, two-byte symbols first.
#[test]
fn bodies_split_into_minijinja_tokens() {
    assert_eq!(
        tokens(" a_1 ** 'x\\'y' // 2.5e-3|f(k=0x1F) "),
        [
            (Kind::Name, "a_1"),
            (Kind::Symbol, "**"),
            (Kind::String, "'x\\'y'"),
            (Kind::Symbol, "//"),
            (Kind::Float, "2.5e-3"),
            (Kind::Symbol, "|"),
            (Kind::Name, "f"),
            (Kind::Symbol, "("),
            (Kind::Name, "k"),
            (Kind::Symbol, "="),
            (Kind::Integer, "0x1F"),
            (Kind::Symbol, ")"),
        ]
    );
    assert_eq!(
        tokens("a<=b!=c>=d==e<f"),
        [
            (Kind::Name, "a"),
            (Kind::Symbol, "<="),
            (Kind::Name, "b"),
            (Kind::Symbol, "!="),
            (Kind::Name, "c"),
            (Kind::Symbol, ">="),
            (Kind::Name, "d"),
            (Kind::Symbol, "=="),
            (Kind::Name, "e"),
            (Kind::Symbol, "<"),
            (Kind::Name, "f"),
        ]
    );
    assert!(tokens(" \t\n").is_empty());
    assert_eq!(
        lex_spans(b"a 'x"),
        Err(LexError::UnterminatedString { start: 2 })
    );
    assert_eq!(
        lex_spans(b"a ! b"),
        Err(LexError::UnexpectedCharacter { start: 2 })
    );
    assert_eq!(
        lex_spans("é".as_bytes()),
        Err(LexError::UnexpectedCharacter { start: 0 })
    );
}

/// Numbers follow MiniJinja's scanner, including radix prefixes and exponents.
#[test]
fn numbers_follow_minijinja_states() {
    for (text, end, float) in [
        ("12", 2, false),
        ("1_000", 5, false),
        ("1.5", 3, true),
        ("1.5.2", 3, true),
        ("1e5", 3, true),
        ("1e+5", 4, true),
        ("1e_+5", 5, true),
        ("1e5+2", 3, true),
        ("1+2", 1, false),
        ("0x1fE", 5, false),
        ("0b102", 5, false),
        ("0o7.5", 3, false),
        ("0b", 2, false),
        ("0e1", 3, true),
        ("1E-", 3, true),
        ("1.e", 3, true),
    ] {
        assert_eq!(number_end(text.as_bytes(), 0), (end, float), "{text}");
    }
    assert_eq!(string_end(b"'a\\'b'c", 0), Some(6));
    assert_eq!(string_end(b"\"a'", 0), None);
    assert_eq!(string_end(b"'\\", 0), None);
}

proptest! {
    /// Spans are ordered, in bounds, and separated only by whitespace.
    #[test]
    fn spans_partition_the_body(body in "[a-z0-9_ '\"+*/=<>!.e\\-~:|(){}\\[\\],]{0,32}") {
        let Ok(spans) = lex_spans(body.as_bytes()) else {
            return Ok(());
        };
        let mut previous = 0;
        for span in &spans {
            prop_assert!(previous <= span.start && span.start < span.end && span.end <= body.len());
            prop_assert!(body[previous..span.start].bytes().all(|byte| byte.is_ascii_whitespace()));
            previous = span.end;
        }
        prop_assert!(body[previous..].bytes().all(|byte| byte.is_ascii_whitespace()));
    }
}

/// Byte classes equal their standard-library counterparts for every byte.
#[test]
fn byte_classes_match_the_standard_library() {
    for byte in 0..=u8::MAX {
        assert_eq!(is_space(byte), byte.is_ascii_whitespace(), "{byte:#x}");
        assert_eq!(
            is_name_byte(byte),
            byte.is_ascii_alphanumeric() || byte == b'_',
            "{byte:#x}"
        );
    }
}

/// A two-byte symbol needs both bytes inside the body.
#[test]
fn double_symbols_need_two_bytes_in_bounds() {
    assert!(is_double_symbol(b"//", 0));
    assert!(!is_double_symbol(b"//", 1));
    assert!(!is_double_symbol(b"//", 2));
    assert!(!is_double_symbol(b"/", 0));
}
