//! Environment tests against goldens rendered by Hugging Face's `render_jinja_template`.
//!
//! Expected strings were produced with `transformers` 5.2.0 and Jinja2 3.1.5;
//! `strftime` goldens use `datetime.fromtimestamp(t, timezone.utc)`.

use proptest::prelude::*;
use serde_json::json;

use super::*;

/// 2024-07-26T13:05:09Z, a Friday and the 208th day of a leap year.
const FRIDAY_AFTERNOON: i64 = 1_721_999_109;

/// Renders a template that must succeed, with a fixed clock.
fn rendered(source: &str, context: serde_json::Value) -> String {
    render_template(source, &context, Some(FRIDAY_AFTERNOON))
        .unwrap_or_else(|error| panic!("{source}: {error:#}"))
}

/// `trim_blocks` and `lstrip_blocks` reproduce Zephyr's multi-line template exactly.
#[test]
fn block_whitespace_matches_hugging_face() {
    let zephyr = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}";
    let messages = json!([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
        {"role": "user", "content": "again"}
    ]);
    assert_eq!(
        rendered(
            zephyr,
            json!({"messages": messages, "add_generation_prompt": true, "eos_token": "</s>"})
        ),
        "<|system|>\nBe brief.</s>\n<|user|>\nhello</s>\n<|assistant|>\nworld</s>\n<|user|>\nagain</s>\n<|assistant|>\n"
    );
    // Indented block tags are stripped by `lstrip_blocks`; one trailing newline is dropped.
    assert_eq!(
        rendered("    {% if true %}\nx\n    {% endif %}\n", json!({})),
        "x\n"
    );
}

/// Undefined variables are falsy, print nothing, and iterate as empty, as in Jinja2.
#[test]
fn undefined_variables_are_lenient() {
    assert_eq!(
        rendered(
            "{% if documents %}docs{% endif %}{% if xml_tools %}x{% endif %}[{{ undefined_var }}]{{ bos_token }}{% for item in missing %}{{ item }}{% endfor %}{% if missing is defined %}d{% else %}u{% endif %}end",
            json!({"documents": null}),
        ),
        "[]uend"
    );
    // Attribute access on an undefined value still fails, as with Jinja2's default `Undefined`.
    assert!(render_template("{{ missing.attribute }}", &json!({}), None).is_err());
}

/// `{% generation %}` blocks render their body unchanged, honoring whitespace control.
#[test]
fn generation_blocks_are_transparent() {
    let template = "{% for m in messages %}{% if m.role == 'assistant' %}{% generation %}<a>{{ m.content }}</a>{% endgeneration %}{% else %}{%- generation -%}  {{ m.content }}  {%- endgeneration -%}{% endif %}{% endfor %}";
    let messages = json!([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
        {"role": "user", "content": "again"}
    ]);
    assert_eq!(
        rendered(template, json!({ "messages": messages })),
        "Be brief.hello<a>world</a>again"
    );
}

/// The rewrite touches only real statements, never comments, raw blocks, or other tags.
#[test]
fn generation_rewrite_preserves_comments_raw_blocks_and_other_tags() {
    assert!(matches!(
        rewrite_generation_blocks("{{ x }}"),
        Cow::Borrowed("{{ x }}")
    ));
    assert_eq!(
        rewrite_generation_blocks(
            "{# {% generation %} #}{% raw %}{% generation %}{% endraw %}{%+ generation -%}x{%- endgeneration +%}{% if generation %}{% endif %}"
        ),
        "{# {% generation %} #}{% raw %}{% generation %}{% endraw %}{%+ if true -%}x{%- endif +%}{% if generation %}{% endif %}"
    );
    // An unterminated tag is copied verbatim and left for the parser to reject.
    assert_eq!(
        rewrite_generation_blocks("a {% generation"),
        "a {% generation"
    );
    assert_eq!(
        rendered(
            "{% raw %}{% generation %}{% endraw %}{# generation #}",
            json!({})
        ),
        "{% generation %}"
    );
}

/// Tag-like text inside string literals, nested brackets, and raw sections stays data.
///
/// Goldens from transformers 5.2.0. The first case is the review probe: a
/// rewrite that ignored `{{ }}` tags changed `'x{% generation %}y'` into
/// `'x{% if true %}y'`.
#[test]
fn generation_rewrite_leaves_literals_and_raw_text_alone() {
    assert_eq!(
        rendered(
            "{{ 'x{% generation %}y' }}|{% set s = '%}' %}{% generation %}a{% endgeneration %}{{ s }}|{{ {'a': {'b': 1}} }}|{{ {'k': '{% generation %}'}['k'] }}",
            json!({}),
        ),
        "x{% generation %}y|a%}|{'a': {'b': 1}}|{% generation %}"
    );
    // A raw section ends at the first `endraw` tag even inside an unterminated literal.
    assert_eq!(
        rendered(
            "{% raw %}{{ '{% endraw %}{% generation %}x{% endgeneration %}",
            json!({})
        ),
        "{{ 'x"
    );
}

/// The shared scanner ends tags exactly where MiniJinja's lexer does.
#[test]
fn tag_scanner_mirrors_minijinja_boundaries() {
    // `}}` closing a nested dict does not end the expression, so the keyword
    // argument after it is still inside the tag.
    assert_eq!(
        parenthesize_conditional_keyword_arguments("{{ f(d={'a': {'b': 1}}, k=v if c else w) }}"),
        "{{ f(d={'a': {'b': 1}}, k=(v if c else w)) }}"
    );
    // Whitespace-controlled raw markers, and `{%` inside a raw section, are data.
    assert_eq!(
        rewrite_generation_blocks(
            "{%- raw +%}{% generation {% x %}{%+ endraw -%}{% generation %}y{% endgeneration %}"
        ),
        "{%- raw +%}{% generation {% x %}{%+ endraw -%}{% if true %}y{% endif %}"
    );
    // Unterminated raw sections, comments, and bracketed tags leave the source borrowed.
    for source in [
        "{% raw %}{% generation %}",
        "{# {% generation %}",
        "{% generation (%}{% endgeneration %}",
        "{ % generation %}",
    ] {
        assert!(
            matches!(rewrite_generation_blocks(source), Cow::Borrowed(_)),
            "{source}"
        );
    }
    assert_eq!(basic_tag_length("- endraw\t+%}tail", "endraw"), Some(12));
    assert_eq!(basic_tag_length(" endrawx %}", "endraw"), None);
    assert_eq!(basic_tag_length(" endraw", "endraw"), None);
    assert_eq!(raw_section_length("a{%x%}b{% endraw %}c"), Some(19));
    assert_eq!(raw_section_length("a{% endraw"), None);
    assert_eq!(find_tag_end(" ')}}' ) }}", "}}"), None);
    assert_eq!(find_tag_end(" ( ')}}' ) }}", "}}"), Some(11));
}

/// `raise_exception` aborts rendering with the template's message.
#[test]
fn raise_exception_propagates_the_message() {
    let error =
        render_template("{{ raise_exception('one tool call') }}", &json!({}), None).unwrap_err();
    assert!(format!("{error:#}").contains("one tool call"), "{error:#}");
}

/// Fuel and recursion limits stop runaway templates.
#[test]
fn runaway_templates_are_bounded() {
    let error = render_template(
        "{% for i in range(100000) %}{% for j in range(100000) %}{% endfor %}{% endfor %}",
        &json!({}),
        None,
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::OutOfFuel);
    let error = render_template(
        "{% macro f(n) %}{{ f(n + 1) }}{% endmacro %}{{ f(0) }}",
        &json!({}),
        None,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("recursion"), "{error:#}");
}

/// `strftime` matches Python's C-locale output, including the `-` padding flag.
#[test]
fn strftime_matches_python() {
    let civil = calendar::civil_time(FRIDAY_AFTERNOON).unwrap();
    for (format, expected) in [
        ("%d %b %Y", "26 Jul 2024"),
        ("%Y-%m-%d %H:%M:%S", "2024-07-26 13:05:09"),
        (
            "%A %a %B %b %j %I %p %y %%",
            "Friday Fri July Jul 208 01 PM 24 %",
        ),
        ("%e|%-d|%-m|%-H|%-I|%-M|%-S|%-j", "26|26|7|13|1|5|9|208"),
        (
            "%F %T %R %D %x %X %C %u %w %h %Z %z",
            "2024-07-26 13:05:09 13:05 07/26/24 07/26/24 13:05:09 20 5 5 Jul UTC +0000",
        ),
        ("%c", "Fri Jul 26 13:05:09 2024"),
        ("a%nb%tc", "a\nb\tc"),
        ("plain text", "plain text"),
    ] {
        assert_eq!(strftime(format, &civil).unwrap(), expected, "{format}");
    }
    let midnight = calendar::civil_time(0).unwrap();
    assert_eq!(
        strftime("%I %p %e %a %j %w %u", &midnight).unwrap(),
        "12 AM  1 Thu 001 4 4"
    );
    for (format, expected) in [
        ("%Q", "unsupported strftime directive %Q"),
        ("50%", "incomplete directive"),
        ("%-", "incomplete directive"),
    ] {
        let error = strftime(format, &civil).unwrap_err();
        assert!(
            format!("{error:#}").contains(expected),
            "{format}: {error:#}"
        );
    }
}

/// `strftime_now` uses the fixed instant when given and the host clock otherwise.
#[test]
fn strftime_now_uses_the_fixed_or_host_clock() {
    assert_eq!(
        rendered("{{ strftime_now('%d %b %Y') }}", json!({})),
        "26 Jul 2024"
    );
    let year = render_template("{{ strftime_now('%Y') }}", &json!({}), None).unwrap();
    let year: i64 = year.parse().unwrap();
    assert!((2024..=9999).contains(&year), "host year {year}");
    let error = render_template(
        "{{ strftime_now('%Y') }}",
        &json!({}),
        Some(calendar::MAX_UNIX_SECONDS + 1),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("outside years 1 through 9999"));
}

/// Llama 3.x selects `strftime_now` for its date line only when no date was passed.
#[test]
fn llama_style_date_fallback_prefers_caller_dates() {
    let template = "{%- if not date_string is defined %}{%- if strftime_now is defined %}{%- set date_string = strftime_now('%d %b %Y') %}{%- else %}{%- set date_string = '26 Jul 2024' %}{%- endif %}{%- endif %}Today Date: {{ date_string }}";
    assert_eq!(rendered(template, json!({})), "Today Date: 26 Jul 2024");
    assert_eq!(
        rendered(template, json!({"date_string": "01 Jan 2030"})),
        "Today Date: 01 Jan 2030"
    );
}

/// Conditional keyword-argument values render as in Jinja2 once parenthesized.
#[test]
fn conditional_keyword_arguments_match_jinja2() {
    // Golden from transformers 5.2.0 with t='v' and u=''.
    assert_eq!(
        rendered(
            "{% set ns = namespace(name=t if t else 'none', other=(1 if t else 2), plain='x') %}{{ ns.name }}|{{ ns.other }}|{{ ns.plain }}|{{ dict(a='y' if t == 'v' else 'n', b=[1 if t else 2, 3]) | tojson }}|{{ 'a=b' if t else 'c' }}|{{ dict(k=u if u else 'empty') | tojson }}",
            json!({"t": "v", "u": ""}),
        ),
        "v|1|x|{\"a\": \"y\", \"b\": [1, 3]}|a=b|{\"k\": \"empty\"}"
    );
}

/// The keyword rewrite wraps only conditional values and leaves everything else intact.
#[test]
fn keyword_rewrite_is_minimal_and_skips_text_comments_and_strings() {
    for (source, expected) in [
        (
            "{% set n = namespace(name=a if a else '', b=1) %}",
            "{% set n = namespace(name=(a if a else ''), b=1) %}",
        ),
        (
            "{{ f(x=g(1, y=2) if c else 3, z=w) }}",
            "{{ f(x=(g(1, y=2) if c else 3), z=w) }}",
        ),
        (
            "{{ f(k=v if c else d) }} text f(k=v if c else d) {# f(k=v if c else d) #}",
            "{{ f(k=(v if c else d)) }} text f(k=v if c else d) {# f(k=v if c else d) #}",
        ),
        (
            "{% raw %}{{ f(k=v if c else d) }}{% endraw %}",
            "{% raw %}{{ f(k=v if c else d) }}{% endraw %}",
        ),
        (
            "{{ f('k=v if c else d', k=x == y) }}{{ a if b else c }}",
            "{{ f('k=v if c else d', k=x == y) }}{{ a if b else c }}",
        ),
        ("{{ é(k=v if c else 'é') }}", "{{ é(k=(v if c else 'é')) }}"),
        ("{{ f(k=v if c else d", "{{ f(k=v if c else d"),
        (
            "plain text without conditionals",
            "plain text without conditionals",
        ),
    ] {
        assert_eq!(
            parenthesize_conditional_keyword_arguments(source),
            expected,
            "{source}"
        );
    }
    assert!(matches!(
        parenthesize_conditional_keyword_arguments("{{ a if b else c }}"),
        Cow::Borrowed(_)
    ));
}

/// `is iterable` and `is sequence` follow Python, so `none` is neither.
#[test]
fn type_tests_follow_python() {
    // Golden from transformers 5.2.0.
    assert_eq!(
        rendered(
            "{{ none is iterable }}|{{ 'a' is iterable }}|{{ {} is iterable }}|{{ 1 is iterable }}|{{ missing is iterable }}|{{ none is sequence }}|{{ 'a' is sequence }}|{{ {} is sequence }}|{{ [1] is sequence }}|{{ 1 is sequence }}|{{ missing is sequence }}|{{ true is iterable }}",
            json!({}),
        ),
        "False|True|True|False|True|False|True|True|True|False|True|False"
    );
    // Nemotron-H's guard no longer evaluates `none | length`.
    assert_eq!(
        rendered(
            "{%- if tools is iterable and tools | length > 0 %}tools{% else %}none{% endif %}",
            json!({"tools": null}),
        ),
        "none"
    );
}

proptest! {
    /// On arbitrary Jinja-like text the source rewrites never panic, and the
    /// keyword rewrite only ever inserts parentheses.
    #[test]
    fn source_rewrites_are_total_and_conservative(
        pieces in proptest::collection::vec(
            prop_oneof![
                Just("{{"), Just("}}"), Just("{%"), Just("%}"), Just("{#"), Just("#}"),
                Just("("), Just(")"), Just("="), Just("=="), Just(","), Just("'"), Just("\""),
                Just(" if "), Just(" else "), Just("generation"), Just("endgeneration"),
                Just("raw"), Just("endraw"), Just("x"), Just("é"), Just("\\"), Just(" "),
            ],
            0..48,
        )
    ) {
        let source = pieces.concat();
        // Each rewrite is a fixed point after one pass: it never creates a
        // tag that a second pass would change.
        let generation = rewrite_generation_blocks(&source);
        prop_assert_eq!(rewrite_generation_blocks(&generation), generation.clone());
        if !source.contains("generation") {
            prop_assert!(matches!(generation, Cow::Borrowed(_)));
        }
        let wrapped = parenthesize_conditional_keyword_arguments(&source);
        prop_assert_eq!(parenthesize_conditional_keyword_arguments(&wrapped), wrapped.clone());
        let without_parentheses = |text: &str| text.replace(['(', ')'], "");
        prop_assert_eq!(without_parentheses(&wrapped), without_parentheses(&source));
        prop_assert_eq!(
            wrapped.matches('(').count() - source.matches('(').count(),
            wrapped.matches(')').count() - source.matches(')').count()
        );
    }
}
