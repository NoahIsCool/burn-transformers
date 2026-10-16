//! Tests for the byte-level template scanner: tag boundaries, raw sections, and markers.

use proptest::prelude::*;

use super::*;

/// Collects every tag span of `source`, scanning the way the source rewrites do.
fn spans(source: &str) -> Vec<(TagKind, &str)> {
    let mut found = Vec::new();
    let mut position = 0;
    while let Some(span) = next_tag(source.as_bytes(), position) {
        found.push((span.kind, &source[span.start..span.end]));
        position = span.end;
    }
    found
}

/// Tags end where MiniJinja's lexer ends them; text, comments, and raw sections are skipped.
#[test]
fn tags_end_at_the_first_top_level_delimiter() {
    assert_eq!(
        spans("a{{ x }}b{% if y %}c{# {{ no }} #}d{% raw %}{{ no }}{% endraw %}{{ '}}' }}"),
        [
            (TagKind::Expression, "{{ x }}"),
            (TagKind::Statement, "{% if y %}"),
            (TagKind::Expression, "{{ '}}' }}"),
        ]
    );
    assert_eq!(
        spans("{{ {'a': {'b': 1}} }}"),
        [(TagKind::Expression, "{{ {'a': {'b': 1}} }}")]
    );
    assert_eq!(
        spans("{%- raw +%}{% generation {% x %}{%+ endraw -%}{% generation %}"),
        [(TagKind::Statement, "{% generation %}")]
    );
    // Unterminated comments, raw sections, and tags end the scan.
    for source in ["{# x", "{% raw %}{{ x }}", "{{ (x }}", "{{ x", "{ % x %}"] {
        assert!(spans(source).is_empty(), "{source}");
    }
    assert_eq!(next_tag(b"{{ x }}", 7), None);
}

/// The helpers keep `str` semantics on their byte offsets.
#[test]
fn scanner_helpers_match_reference_offsets() {
    assert_eq!(
        next_tag_opening(b"a{b{%", 0),
        Some((3, TagOpening::Statement))
    );
    assert_eq!(next_tag_opening(b"{#", 1), None);
    assert_eq!(next_tag_opening(b"{", 0), None);
    assert_eq!(find_pattern(b"ab#}#}", 0, b"#}"), Some(2));
    assert_eq!(find_pattern(b"ab#}#}", 3, b"#}"), Some(4));
    assert_eq!(find_pattern(b"ab", 2, b""), Some(2));
    assert_eq!(find_pattern(b"ab", 0, b"abc"), None);
    assert_eq!(skip_string(b"'a\\'b' c", 0, b'\''), 6);
    assert_eq!(skip_string(b"'abc", 0, b'\''), 4);
    assert_eq!(skip_string(b"'\\", 0, b'\''), 2);
    assert_eq!(
        basic_tag_end(b"{%- endraw\t+%}tail", 2, b"endraw"),
        Some(14)
    );
    assert_eq!(basic_tag_end(b" endrawx %}", 0, b"endraw"), None);
    assert_eq!(basic_tag_end(b" endraw", 0, b"endraw"), None);
    assert_eq!(raw_section_end(b"a{%x%}b{% endraw %}c", 0), Some(19));
    assert_eq!(raw_section_end(b"a{% endraw", 0), None);
    assert_eq!(find_tag_end(b" ')}}' ) }}", 0, b"}}"), None);
    assert_eq!(find_tag_end(b" ( ')}}' ) }}", 0, b"}}"), Some(11));
    assert_eq!(marker_widths(b"- x +"), (1, 1));
    assert_eq!(marker_widths(b"-"), (1, 0));
    assert_eq!(marker_widths(b"+x"), (1, 0));
    assert_eq!(marker_widths(b" x -"), (0, 1));
    assert_eq!(marker_widths(b""), (0, 0));
    assert!(is_ascii_space(0x0c));
    assert!(!is_ascii_space(0x0b));
}

/// Shortening never splits a character.
#[test]
fn boundaries_never_split_characters() {
    assert_eq!(char_boundary_at_most("abc", 2), 2);
    assert_eq!(char_boundary_at_most("aé", 2), 1);
    assert_eq!(char_boundary_at_most("é", 1), 0);
    assert_eq!(char_boundary_at_most("a😀b", 4), 1);
    assert_eq!(char_boundary_at_most("a😀b", 5), 5);
}

proptest! {
    /// Every span is in bounds, delimited by its kind, and after the previous one.
    #[test]
    fn spans_are_delimited_and_ordered(source in "[ab{}%#'\"()\\[\\]\\-+ é]{0,40}") {
        let bytes = source.as_bytes();
        let mut position = 0;
        while let Some(span) = next_tag(bytes, position) {
            prop_assert!(position <= span.start && span.start + 4 <= span.end && span.end <= bytes.len());
            let (opening, closing) = match span.kind {
                TagKind::Expression => ("{{", "}}"),
                TagKind::Statement => ("{%", "%}"),
            };
            prop_assert!(source[span.start..].starts_with(opening));
            prop_assert!(source[..span.end].ends_with(closing));
            position = span.end;
        }
    }

    /// The opening finder agrees with a `str::find` reference.
    #[test]
    fn openings_match_str_find(source in "[a{%#}]{0,24}", from in 0_usize..24) {
        let from = from.min(source.len());
        let reference = source[from..]
            .match_indices('{')
            .map(|(offset, _)| from + offset)
            .find_map(|index| match source.as_bytes().get(index + 1) {
                Some(b'{') => Some((index, TagOpening::Expression)),
                Some(b'%') => Some((index, TagOpening::Statement)),
                Some(b'#') => Some((index, TagOpening::Comment)),
                _ => None,
            });
        prop_assert_eq!(next_tag_opening(source.as_bytes(), from), reference);
    }
}
