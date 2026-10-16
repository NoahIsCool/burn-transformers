//! Operator-guard tests: exact rewrites, MiniJinja precedence, failure policy, and nesting limits.

use proptest::prelude::*;

use super::*;

/// Rewrites `source`, panicking on a guard error.
fn guarded(source: &str) -> String {
    guard_operators(source)
        .unwrap_or_else(|error| panic!("{source}: {error}"))
        .into_owned()
}

/// Every guarded operator becomes a bounded filter with MiniJinja's precedence.
#[test]
fn operators_become_bounded_filters_with_minijinja_precedence() {
    let cases = [
        ("{{ a + b }}", "{{ a|__bounded_add(b) }}"),
        ("{{ a ~ b }}", "{{ a|__bounded_concat(b) }}"),
        ("{{ a * b }}", "{{ a|__bounded_mul(b) }}"),
        // Same-precedence chains become left-to-right filter chains.
        (
            "{{ a + b + c }}",
            "{{ a|__bounded_add(b)|__bounded_add(c) }}",
        ),
        // An unguarded operator of the same level is parenthesized first.
        ("{{ a - b + c }}", "{{ (a - b)|__bounded_add(c) }}"),
        ("{{ a + b - c }}", "{{ a|__bounded_add(b) - c }}"),
        ("{{ x ** y * z }}", "{{ (x ** y)|__bounded_mul(z) }}"),
        ("{{ a / b * c }}", "{{ (a / b)|__bounded_mul(c) }}"),
        // Tighter levels nest inside the right operand.
        (
            "{{ a ~ b * c }}",
            "{{ a|__bounded_concat(b|__bounded_mul(c)) }}",
        ),
        (
            "{{ a * b ~ c + d }}",
            "{{ a|__bounded_mul(b)|__bounded_concat(c)|__bounded_add(d) }}",
        ),
        (
            "{{ (a + b) * c }}",
            "{{ (a|__bounded_add(b))|__bounded_mul(c) }}",
        ),
        // Unary minus, postfix, filters, and tests already bind tighter.
        ("{{ -x * 2 }}", "{{ -x|__bounded_mul(2) }}"),
        (
            "{{ a.b[0](c)|upper ~ d }}",
            "{{ a.b[0](c)|upper|__bounded_concat(d) }}",
        ),
        (
            "{{ x is divisibleby 3 * 2 }}",
            "{{ x is divisibleby 3|__bounded_mul(2) }}",
        ),
        // Adjacent string literals are one operand.
        ("{{ 'a' 'b' * 2 }}", "{{ 'a' 'b'|__bounded_mul(2) }}"),
        // Comparisons, logic, and conditionals keep their text around rewritten operands.
        (
            "{{ a + 1 > b and not c ~ d in e if f else g * 2 }}",
            "{{ a|__bounded_add(1) > b and not c|__bounded_concat(d) in e if f else g|__bounded_mul(2) }}",
        ),
        // Containers, subscripts, slices, and call arguments are rewritten inside.
        (
            "{{ [a + b, {'k': c * 2}, (d ~ e, f)] }}",
            "{{ [a|__bounded_add(b), {'k': c|__bounded_mul(2)}, (d|__bounded_concat(e), f)] }}",
        ),
        (
            "{{ x[a + 1:b * 2:c] }}",
            "{{ x[a|__bounded_add(1):b|__bounded_mul(2):c] }}",
        ),
        (
            "{{ f(a ~ b, *rest, k=c + d, **more) }}",
            "{{ f(a|__bounded_concat(b), *rest, k=c|__bounded_add(d), **more) }}",
        ),
        // Whitespace markers survive.
        ("{{- a + b -}}", "{{- a|__bounded_add(b) -}}"),
        ("{{+ a * b +}}", "{{+ a|__bounded_mul(b) +}}"),
    ];
    for (source, expected) in cases {
        assert_eq!(guarded(source), expected, "{source}");
    }
}

/// Statements are parsed far enough to rewrite every expression they contain.
#[test]
fn statements_rewrite_their_expressions() {
    let cases = [
        ("{% set x = a + b %}", "{% set x = a|__bounded_add(b) %}"),
        (
            "{% set ns.text = ns.text ~ ns.text %}",
            "{% set ns.text = ns.text|__bounded_concat(ns.text) %}",
        ),
        (
            "{% set a, b = c * 2, d + e %}",
            "{% set a, b = c|__bounded_mul(2), d|__bounded_add(e) %}",
        ),
        (
            "{% set x | replace('a', 'b' * 3) %}",
            "{% set x | replace('a', 'b'|__bounded_mul(3)) %}",
        ),
        (
            "{%- for m in messages + extra if m.role ~ 'x' recursive -%}",
            "{%- for m in messages|__bounded_add(extra) if m.role|__bounded_concat('x') recursive -%}",
        ),
        (
            "{% for (k, v) in items * 2 %}",
            "{% for (k, v) in items|__bounded_mul(2) %}",
        ),
        ("{% if a * 2 > 3 %}", "{% if a|__bounded_mul(2) > 3 %}"),
        ("{% elif a ~ b %}", "{% elif a|__bounded_concat(b) %}"),
        (
            "{% with a = b + 1, (c, d) = e * 2 %}",
            "{% with a = b|__bounded_add(1), (c, d) = e|__bounded_mul(2) %}",
        ),
        (
            "{% macro m(x, y=1 + 2) %}",
            "{% macro m(x, y=1|__bounded_add(2)) %}",
        ),
        (
            "{% call(u) m(u + 1) %}",
            "{% call(u) m(u|__bounded_add(1)) %}",
        ),
        (
            "{% filter replace('a', 'b' ~ 'c') | upper %}",
            "{% filter replace('a', 'b'|__bounded_concat('c')) | upper %}",
        ),
        ("{% do f(a * b) %}", "{% do f(a|__bounded_mul(b)) %}"),
        (
            "{% include 'a' ~ b ignore missing with context %}",
            "{% include 'a'|__bounded_concat(b) ignore missing with context %}",
        ),
        (
            "{% import 'a' ~ b as m without context %}",
            "{% import 'a'|__bounded_concat(b) as m without context %}",
        ),
        (
            "{% from 'a' ~ b import x as y, z with context %}",
            "{% from 'a'|__bounded_concat(b) import x as y, z with context %}",
        ),
        (
            "{% extends 'a' * 2 %}",
            "{% extends 'a'|__bounded_mul(2) %}",
        ),
        (
            "{% autoescape a + b %}",
            "{% autoescape a|__bounded_add(b) %}",
        ),
    ];
    for (source, expected) in cases {
        assert_eq!(guarded(source), expected, "{source}");
    }
}

/// Text, comments, raw sections, string literals, splats, and exponents are never rewritten.
#[test]
fn non_operators_are_left_alone() {
    for source in [
        "a + b ~ c * d",
        "{# a + b #}",
        "{% raw %}{{ a + b }}{% endraw %}",
        "{{ 'a + b ~ c * d' }}",
        "{{ \"it's \\\" + \" }}",
        "{{ f(*args, **kwargs) }}",
        "{{ 1e+5 }}{{ 2E-3 }}",
        "{{ x - y // z % w ** v }}",
        "{% endfor %}{% else %}{% break %}{% endmacro %}",
        "{% block content %}{% endblock content %}",
    ] {
        assert!(
            matches!(guard_operators(source), Ok(Cow::Borrowed(_))),
            "{source}"
        );
    }
    // A statement keyword MiniJinja does not know is kept for it to reject.
    assert!(matches!(
        guard_operators("{% frobnicate a + b %}"),
        Ok(Cow::Borrowed(_))
    ));
}

/// Unreadable tags are refused only when they contain a guarded operator.
#[test]
fn unreadable_tags_with_guarded_operators_are_refused() {
    for source in [
        "{{ a + }}",
        "{{ a * * b }}",
        "{% for x in a + b c %}",
        "{{ a + b c }}",
        "{{ f(a, k=1, 2 ~ 3) b }}",
        "{{ x.1.5 + 1 }}",
        "{{ a ! b + c }}",
        // `not` in operand position is a variable to MiniJinja but would be the
        // operator once the operand is re-read inside filter arguments.
        "{{ 0 * not (0) }}",
        "{{ 1 + not.x * 3 }}",
    ] {
        let error = guard_operators(source).expect_err(source);
        assert!(error.to_string().contains("cannot be bounded"), "{error}");
        assert_eq!(error.offset, 0, "{source}");
    }
    // The same mistakes without guarded operators are MiniJinja's to report.
    for source in ["{{ a b }}", "{{ }}", "{% for x in %}", "{{ a ! b }}"] {
        assert!(
            matches!(guard_operators(source), Ok(Cow::Borrowed(_))),
            "{source}"
        );
    }
    // Errors name the tag's offset and shorten long tags.
    let long = format!("ok {{{{ a + {} }}}}", "b ".repeat(100));
    let error = guard_operators(&long).unwrap_err();
    assert_eq!(error.offset, 3);
    assert!(
        error.tag.ends_with("...") && error.tag.len() <= 83,
        "{}",
        error.tag
    );
}

/// Nesting and chains beyond the limits are refused before MiniJinja can overflow its stack.
#[test]
fn excessive_nesting_and_chains_are_refused() {
    let nested = |depth: usize| format!("{{{{ {}1{} }}}}", "(".repeat(depth), ")".repeat(depth));
    // Each parenthesis is a primary and an expression level, as in MiniJinja.
    assert!(guard_operators(&nested(90)).is_ok());
    for source in [
        nested(MAX_NESTING),
        format!("{{{{ {}x }}}}", "- ".repeat(MAX_CHAIN + 1)),
        format!("{{{{ {}x }}}}", "not ".repeat(MAX_CHAIN + 1)),
        format!("{{{{ x{} }}}}", " - 1".repeat(MAX_CHAIN + 1)),
        format!("{{{{ x{} }}}}", "|upper".repeat(MAX_CHAIN + 1)),
        format!("{{{{ x{} }}}}", ".a".repeat(MAX_CHAIN + 1)),
        format!("{{{{ x{} }}}}", " if y else x".repeat(MAX_NESTING + 1)),
    ] {
        assert!(guard_operators(&source).is_err(), "{}", &source[..40]);
    }
    // Chains within the limit are accepted.
    let chain = format!("{{{{ x{} }}}}", " + 1".repeat(MAX_CHAIN));
    assert!(guard_operators(&chain).is_ok());
}

/// Rewrites keep every line break, so error locations after a rewritten tag do not move.
///
/// The operator and its surrounding whitespace disappear in a rewrite; their
/// line breaks move inside the filter call, and the whitespace around a tag
/// body is kept as written.
#[test]
fn rewrites_keep_line_breaks() {
    for source in [
        "{{\n  a\n  +\n  b\n}}",
        "{%-\n set x = a ~\n b\n-%}",
        "{{ a *\n\n b + c\n }}",
        "{{a+b}}",
        "{%+ if a * b +%}",
    ] {
        let rewritten = guarded(source);
        assert_ne!(rewritten, source);
        assert_eq!(
            rewritten.matches('\n').count(),
            source.matches('\n').count(),
            "{rewritten}"
        );
    }
    assert_eq!(guarded("{{a+b}}"), "{{a|__bounded_add(b)}}");
    assert_eq!(guarded("{{\n a\n + b }}"), "{{\n a|__bounded_add(\nb) }}");
}

/// Comma-separated lists of any length are flat and are not capped like chains.
///
/// MiniJinja parses, compiles, and drops list, map, and tuple items, call
/// arguments, and macro parameters iteratively, so only operator, unary,
/// postfix, filter, and conditional chains (which it handles recursively)
/// are limited.
#[test]
fn long_flat_lists_are_rewritten() {
    let count = MAX_CHAIN + 100;
    let join = |item: &dyn Fn(usize) -> String| (0..count).map(item).collect::<Vec<_>>().join(", ");
    for source in [
        format!("{{{{ [{}] }}}}", join(&|index| format!("{index} + 1"))),
        format!("{{{{ ({}) }}}}", join(&|index| format!("{index} + 1"))),
        format!(
            "{{{{ {{{}}} }}}}",
            join(&|index| format!("'k{index}': {index} * 2"))
        ),
        format!(
            "{{{{ f({}) }}}}",
            join(&|index| format!("k{index}=a ~ {index}"))
        ),
        format!(
            "{{% macro m({}) %}}{{% endmacro %}}",
            join(&|index| format!("p{index}={index} * 2"))
        ),
    ] {
        let rewritten = guarded(&source);
        assert_eq!(
            rewritten.matches("__bounded_").count(),
            count,
            "{}",
            &source[..40]
        );
    }
}

/// The lexer reads numbers, strings, and names exactly as MiniJinja does.
#[test]
fn lexer_matches_minijinja_tokens() {
    let tokens =
        lex("0x1F_ff 1_000 2.5e-3 1.e5 0b101 'a\\'b' \"q\" name_1 ** // == != <= >= + .").unwrap();
    let summary: Vec<(Kind, &str)> = tokens
        .iter()
        .map(|token| (token.kind, token.text))
        .collect();
    assert_eq!(
        summary,
        [
            (Kind::Integer, "0x1F_ff"),
            (Kind::Integer, "1_000"),
            (Kind::Float, "2.5e-3"),
            (Kind::Float, "1.e5"),
            (Kind::Integer, "0b101"),
            (Kind::String, "'a\\'b'"),
            (Kind::String, "\"q\""),
            (Kind::Name, "name_1"),
            (Kind::Symbol, "**"),
            (Kind::Symbol, "//"),
            (Kind::Symbol, "=="),
            (Kind::Symbol, "!="),
            (Kind::Symbol, "<="),
            (Kind::Symbol, ">="),
            (Kind::Symbol, "+"),
            (Kind::Symbol, "."),
        ]
    );
    assert!(lex("'open").is_err());
    assert!(lex("a ! b").is_err());
    assert!(lex("é").is_err());
    assert_eq!(number_end(b"12abc", 0), (2, false));
    assert_eq!(number_end(b"0xfg", 0), (3, false));
    assert_eq!(string_end(b"'a\\", 0), None);
    assert_eq!(shorten("short"), "short");
}

prop_compose! {
    /// A random token soup drawn from operators, operands, and punctuation.
    fn token_soup()(pieces in prop::collection::vec(prop::sample::select(vec![
        "a", "1", "'s'", "+", "-", "*", "~", "**", "//", "(", ")", "[", "]", "{", "}",
        ",", ":", "|", "upper", "is", "not", "in", "and", "if", "else", "=", ".", "x",
        "f(", "-1", "%", "== ", "*rest", "**kw",
    ]), 0..24)) -> String {
        pieces.join(" ")
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// The guard is total: any tag either rewrites (keeping every token outside
    /// guarded operators) or is kept or refused, and never panics. A rewritten
    /// tag contains no bare guarded operator, and rewriting is idempotent.
    #[test]
    fn guard_is_total_and_leaves_no_guarded_operator(body in token_soup()) {
        let source = format!("{{{{ {body} }}}}");
        // Kept and refused tags need no further check: the guard did not panic.
        if let Ok(Cow::Owned(rewritten)) = guard_operators(&source) {
            let inner = &rewritten[2..rewritten.len() - 2];
            let tag = format!("{{{{{inner}}}}}");
            let tokens = lex(statement_parts(&tag).1).unwrap();
            // Remaining `*` tokens can only be splats, directly after `(` or `,`.
            for (index, token) in tokens.iter().enumerate() {
                if token.is_guarded() {
                    prop_assert_eq!(token.text, "*");
                    prop_assert!(index > 0 && matches!(tokens[index - 1].text, "(" | ","));
                }
            }
            prop_assert_eq!(guard_operators(&rewritten).unwrap().into_owned(), rewritten.clone());
        }
    }
}
