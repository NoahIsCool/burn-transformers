//! Operator-guard tests: exact rewrites, MiniJinja precedence, failure policy, and nesting limits.

use proptest::prelude::*;

use super::lexer::{number_end, string_end};
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
        // Containers, subscripts, and call arguments are rewritten inside.
        (
            "{{ [a + b, {'k': c * 2}, (d ~ e, f)] }}",
            "{{ [a|__bounded_add(b), {'k': c|__bounded_mul(2)}, (d|__bounded_concat(e), f)] }}",
        ),
        ("{{ x[a + 1] }}", "{{ x[a|__bounded_add(1)] }}"),
        // Slices become the bounded slice filter, with `none` for omitted bounds.
        (
            "{{ x[a + 1:b * 2:c] }}",
            "{{ x|__bounded_slice(a|__bounded_add(1), b|__bounded_mul(2), c) }}",
        ),
        ("{{ x[1:] }}", "{{ x|__bounded_slice(1, none, none) }}"),
        ("{{ x[:2] }}", "{{ x|__bounded_slice(none, 2, none) }}"),
        ("{{ x[::-1] }}", "{{ x|__bounded_slice(none, none, -1) }}"),
        ("{{ x[:] }}", "{{ x|__bounded_slice(none, none, none) }}"),
        // A filter takes no postfix syntax, so a following one parenthesizes it.
        (
            "{{ x[1:].a[0](b)[::2] }}",
            "{{ (x|__bounded_slice(1, none, none)).a[0](b)|__bounded_slice(none, none, 2) }}",
        ),
        (
            "{{ x[1:][2:] }}",
            "{{ (x|__bounded_slice(1, none, none))|__bounded_slice(2, none, none) }}",
        ),
        // Unary minus applies before postfix syntax, as in MiniJinja.
        ("{{ -x[1:] }}", "{{ -x|__bounded_slice(1, none, none) }}"),
        // A sliced test argument is parenthesized, so the filter binds to it.
        (
            "{{ a is sameas b[1:] }}",
            "{{ a is sameas (b|__bounded_slice(1, none, none)) }}",
        ),
        (
            "{{ x[1:] is string and x[1:] + y }}",
            "{{ x|__bounded_slice(1, none, none) is string and x|__bounded_slice(1, none, none)|__bounded_add(y) }}",
        ),
        // Splatted call arguments are charged.
        (
            "{{ f(a ~ b, *rest, k=c + d, **more) }}",
            "{{ f(a|__bounded_concat(b), *rest|__bounded_splat, k=c|__bounded_add(d), **more|__bounded_splat) }}",
        ),
        (
            "{{ f(*(a if b else c), **m[1:]) }}",
            "{{ f(*(a if b else c)|__bounded_splat, **m|__bounded_slice(1, none, none)|__bounded_splat) }}",
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

/// Text, comments, raw sections, string literals, subscripts, and exponents are never rewritten.
#[test]
fn non_operators_are_left_alone() {
    for source in [
        "a + b ~ c * d",
        "{# a + b #}",
        "{% raw %}{{ a + b }}{% endraw %}",
        "{{ 'a + b ~ c * d' }}",
        "{{ \"it's \\\" + \" }}",
        "{{ x[0] }}{{ m['a'] }}{{ x.0 }}",
        "{{ f(*args|__bounded_splat, **kwargs|__bounded_splat) }}",
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
        // Slices (`:`) and `**` splats are guarded too.
        "{{ a[1: b] c }}",
        "{{ f(**k) b }}",
        "{{ a ! b[1:] }}",
    ] {
        let error = guard_operators(source).expect_err(source);
        assert!(error.to_string().contains("cannot be bounded"), "{error}");
        assert_eq!(error.offset, 0, "{source}");
    }
    // The same mistakes without guarded operators are MiniJinja's to report.
    for source in [
        "{{ a b }}",
        "{{ }}",
        "{% for x in %}",
        "{{ a ! b }}",
        "{{ a[1] b }}",
    ] {
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

/// Syntax errors in tags with guarded operators are refused with the parser's exact expectation.
///
/// MiniJinja rejects the same tags at compile time; the guard refuses them
/// first because it cannot rewrite what it cannot parse. `not` after an
/// operand must begin `not in`, and a filter needs a name.
#[test]
fn guarded_syntax_errors_name_the_expected_token() {
    for (source, reason) in [
        ("{{ a + 1 not b }}", "expected `in`, found `b`"),
        ("{{ a + 1 not }}", "expected `in`, found the end of the tag"),
        ("{{ a + 1 | 5 }}", "expected a name, found `5`"),
    ] {
        let error = guard_operators(source).unwrap_err();
        assert_eq!(error.reason, reason, "{source}");
        assert_eq!((error.offset, error.tag.as_str()), (0, source));
    }
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

/// The guard's filter names are the sandbox's, so every rewrite reaches a bounded filter.
#[test]
fn rewrites_name_the_sandbox_filters() {
    assert_eq!(SPLAT_SUFFIX, format!("|{}", sandbox::SPLAT_FILTER));
    assert!(guarded("{{ s[1:] }}").contains(sandbox::SLICE_FILTER));
    assert!(guarded("{{ a + b }}").contains(sandbox::ADD_FILTER));
    assert!(guarded("{{ a ~ b }}").contains(sandbox::CONCAT_FILTER));
    assert!(guarded("{{ a * b }}").contains(sandbox::MULTIPLY_FILTER));
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
        "f(", "-1", "%", "== ", "*rest", "**kw", "[1:]", "[::-1]", "sameas", "divisibleby",
    ]), 0..24)) -> String {
        pieces.join(" ")
    }
}

/// Strategy for syntactically varied MiniJinja expressions (types are not considered).
///
/// It draws every construct of the expression grammar: binary operators of
/// every precedence level, unary minus and `not` chains, conditionals with
/// and without `else`, postfix attributes, integer attributes, subscripts,
/// slices with any combination of bounds, calls with positional, keyword,
/// `*`, and `**` arguments, filters with and without arguments,
/// parenthesis-free and called tests (negated or not), string runs, and
/// list, tuple, and map literals.
fn syntax_expression() -> impl Strategy<Value = String> {
    let leaf = prop::sample::select(vec![
        "a",
        "x1",
        "1",
        "2.5",
        "0x1f",
        "'s'",
        "'a' \"b\"",
        "true",
        "none",
        "loop",
        "f()",
    ])
    .prop_map(str::to_owned);
    leaf.prop_recursive(5, 48, 3, |inner| {
        let binary = prop::sample::select(vec![
            "+", "-", "*", "/", "//", "%", "**", "~", "==", "!=", "<", ">=", "in", "not in", "and",
            "or",
        ]);
        let bound = || prop::option::of(inner.clone());
        prop_oneof![
            (inner.clone(), binary, inner.clone())
                .prop_map(|(left, operator, right)| format!("{left} {operator} {right}")),
            inner.clone().prop_map(|operand| format!("-{operand}")),
            inner.clone().prop_map(|operand| format!("not {operand}")),
            inner.clone().prop_map(|operand| format!("({operand})")),
            (
                inner.clone(),
                inner.clone(),
                prop::option::of(inner.clone())
            )
                .prop_map(|(yes, test, no)| match no {
                    Some(no) => format!("{yes} if {test} else {no}"),
                    None => format!("{yes} if {test}"),
                }),
            inner
                .clone()
                .prop_map(|operand| format!("({operand}).name.0")),
            (inner.clone(), inner.clone())
                .prop_map(|(operand, index)| format!("a[{operand}][{index}]")),
            (bound(), bound(), bound(), any::<bool>()).prop_map(|(start, stop, step, two)| {
                let part = |bound: Option<String>| bound.unwrap_or_default();
                if two {
                    format!("a[{}:{}]", part(start), part(stop))
                } else {
                    format!("a.b[{}:{}:{}]", part(start), part(stop), part(step))
                }
            }),
            (inner.clone(), inner.clone(), inner.clone()).prop_map(|(first, keyword, splat)| {
                format!("f({first}, k={keyword}, *{splat}, **kw)")
            }),
            (inner.clone(), inner.clone()).prop_map(|(operand, argument)| {
                format!("({operand}) | upper | replace({argument}, 'x') | default(d=1)")
            }),
            (inner.clone(), inner.clone(), any::<bool>()).prop_map(
                |(operand, argument, negate)| {
                    let not = if negate { "not " } else { "" };
                    format!("({operand}) is {not}divisibleby {argument}")
                }
            ),
            (inner.clone(), inner.clone())
                .prop_map(|(operand, argument)| { format!("({operand}) is sameas({argument})") }),
            (inner.clone(), inner.clone())
                .prop_map(|(first, second)| format!("[{first}, {second}]")),
            (inner.clone(), inner.clone())
                .prop_map(|(first, second)| format!("({first}, {second},)")),
            (inner.clone(), inner.clone()).prop_map(|(key, value)| format!("{{{key}: {value}}}")),
        ]
    })
}

/// Parses an expression tag body with the guard's grammar, returning the refusal if any.
fn parse_expression(body: &str) -> Result<(), Refusal> {
    let tokens = lex(body).map_err(Refusal::Syntax)?;
    let mut parser = Parser {
        source: body,
        tokens: &tokens,
        position: 0,
        depth: 0,
        rewrites: 0,
    };
    parser.expression_tag().map(drop)
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

    /// The guard reads every expression MiniJinja compiles, and its rewrite compiles too.
    ///
    /// Review finding GUARD-DIFFERENTIAL-COVERAGE asked for a check that no
    /// tag carrying a guarded operator, slice, or splat is kept verbatim
    /// while MiniJinja accepts it. Because a tag is kept verbatim only when
    /// it has nothing to rewrite or the guard cannot parse it, this checks
    /// the stronger property that the guard's grammar covers MiniJinja's:
    /// whenever MiniJinja compiles `{{ body }}`, the guard parses the body,
    /// and the rewritten tag compiles as well. The one documented exception
    /// is a variable named `not` used as an operand, which the guard refuses.
    #[test]
    fn guard_reads_every_expression_minijinja_compiles(
        body in prop_oneof![syntax_expression(), token_soup()]
    ) {
        let source = format!("{{{{ {body} }}}}");
        let environment = minijinja::Environment::new();
        if environment.template_from_str(&source).is_err() {
            // Invalid syntax: the guard may keep or refuse it; MiniJinja rejects it.
            return Ok(());
        }
        match parse_expression(&body) {
            Ok(()) => {}
            Err(Refusal::Syntax(reason)) if reason.contains("variable named `not`") => {
                return Ok(());
            }
            Err(refusal) => prop_assert!(false, "{source}: {refusal:?}"),
        }
        let rewritten = guard_operators(&source).unwrap();
        prop_assert!(environment.template_from_str(&rewritten).is_ok(), "{}", rewritten);
    }
}

/// Less common statement shapes, trailing commas, and dotted filter names still rewrite.
#[test]
fn uncommon_statement_shapes_rewrite_their_expressions() {
    let cases = [
        ("{% call m(u + 1) %}", "{% call m(u|__bounded_add(1)) %}"),
        (
            "{% include 'a' ~ b %}",
            "{% include 'a'|__bounded_concat(b) %}",
        ),
        (
            "{% from 'a' ~ b import x %}",
            "{% from 'a'|__bounded_concat(b) import x %}",
        ),
        (
            "{% set a, b = c * 2, %}",
            "{% set a, b = c|__bounded_mul(2), %}",
        ),
        (
            "{% set a, b, c = d * 2, e, f %}",
            "{% set a, b, c = d|__bounded_mul(2), e, f %}",
        ),
        (
            "{% for a, in x + y %}",
            "{% for a, in x|__bounded_add(y) %}",
        ),
        (
            "{% for (a,) in x + y %}",
            "{% for (a,) in x|__bounded_add(y) %}",
        ),
        (
            "{% macro m(x, y=1 + 2,) %}",
            "{% macro m(x, y=1|__bounded_add(2),) %}",
        ),
        (
            "{% filter a.b('x' ~ y) %}",
            "{% filter a.b('x'|__bounded_concat(y)) %}",
        ),
        ("{{ f(a + 1,) }}", "{{ f(a|__bounded_add(1),) }}"),
        ("{{ [a + 1,] }}", "{{ [a|__bounded_add(1),] }}"),
        ("{{ () + a }}", "{{ ()|__bounded_add(a) }}"),
    ];
    for (source, expected) in cases {
        assert_eq!(guarded(source), expected, "{source}");
    }
}

/// Statements missing a required keyword are refused when they hold a guarded operator.
#[test]
fn malformed_statements_with_guarded_operators_are_refused() {
    for source in [
        "{% (a + b) %}",
        "{% for x of a + b %}",
        "{% include 'a' ~ b ignore x %}",
        "{% include 'a' ~ b with x %}",
        "{% include 'a' ~ b with 1 %}",
        "{% import 'a' ~ b %}",
        "{% from 'a' ~ b %}",
        "{% filter a.('x' ~ y) %}",
    ] {
        let error = guard_operators(source).expect_err(source);
        assert!(
            error.to_string().contains("cannot be bounded"),
            "{source}: {error}"
        );
    }
}
