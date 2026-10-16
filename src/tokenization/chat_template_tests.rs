//! Environment tests against goldens rendered by Hugging Face's `render_jinja_template`.
//!
//! Expected strings were produced with `transformers` 5.2.0 and Jinja2 3.1.5;
//! `strftime` goldens use `datetime.fromtimestamp(t, timezone.utc)`.

use proptest::prelude::*;
use serde_json::json;

use super::*;
use sandbox::{RenderBudget, RenderLimits};

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

// ---------------------------------------------------------------------------
// Sandbox regressions and differential checks against plain MiniJinja.
// ---------------------------------------------------------------------------

/// CHAT-1: string repetition by a count past `isize::MAX` fails contextually.
///
/// MiniJinja folded `'ab' * 4611686018427387904` while *compiling* and its
/// `str::repeat` panicked with "capacity overflow". The guard turns `*` into a
/// filter call, which is never folded, and the bounded filter checks the
/// product first.
#[test]
fn huge_string_repetition_fails_contextually() {
    for source in [
        "{{ 'ab' * 4611686018427387904 }}",
        "{{ 4611686018427387904 * 'ab' }}",
        "{% set x = 'ab' * (2 ** 62) %}",
        "{{ ('ab' * 1000000) * 1000000 }}",
    ] {
        let error = render_template(source, &json!({}), None).unwrap_err();
        assert!(
            format!("{error:#}").contains("resource limit"),
            "{source}: {error:#}"
        );
    }
}

/// CHAT-2: one expression can no longer render (or build) a prompt of hundreds of megabytes.
#[test]
fn oversized_values_and_prompts_fail_contextually() {
    for source in [
        "{{ 'ab' * 100000000 }}",
        "{% set ns = namespace(s='ab') %}{% for i in range(30) %}{% set ns.s = ns.s ~ ns.s %}{% endfor %}",
        "{% set ns = namespace(s='ab') %}{% for i in range(30) %}{% set ns.s = ns.s + ns.s %}{% endfor %}",
        "{% for i in range(60) %}{{ 'x' * 1000000 }}{% endfor %}",
    ] {
        let started = std::time::Instant::now();
        let error = render_template(source, &json!({}), None).unwrap_err();
        assert!(
            format!("{error:#}").contains("resource limit"),
            "{source}: {error:#}"
        );
        assert!(started.elapsed().as_secs_f64() < 5.0, "{source}");
    }
}

/// Retaining input-sized copies in a loop fails within the budget, quickly.
///
/// Review finding CHAT-UNCHARGED-RETENTION: with a 1 MB string, keeping
/// `s | list` alive 20 times reached 467 MiB and `s | reverse` 300 times
/// 299 MiB, against a documented 16 MiB budget, and both renders succeeded.
/// Every copy is now charged before it is made.
#[test]
fn retained_copies_fail_within_the_budget() {
    for copy in [
        "s | list",
        "s | reverse",
        "s[1:]",
        "s.strip()",
        "s.split('b')",
        "[s] | map('lower') | list",
        "range(*[0, 100000]) | list",
    ] {
        let source = format!(
            "{{% set s = 'a' * 1000000 %}}{{% set ns = namespace(l=none) %}}{{% for i in range(300) %}}{{% set ns.l = [ns.l, {copy}] %}}{{% endfor %}}ok"
        );
        let started = std::time::Instant::now();
        let error = render_template(&source, &json!({}), None).unwrap_err();
        assert!(
            format!("{error:#}").contains("resource limit"),
            "{copy}: {error:#}"
        );
        assert!(started.elapsed().as_secs_f64() < 10.0, "{copy}");
    }
}

/// CHAT-3: doubling a list materializes it within the item limit instead of chaining lazily.
///
/// MiniJinja's `seq + seq` is a lazy chain: 24 doublings cost about 100 units
/// of fuel, but `| length` then walks 2^24 items through 24 nested iterators.
/// The bounded `+` builds a list and stops at 2^20 items or at the render's
/// cumulative budget, whichever comes first.
#[test]
fn doubled_sequences_finish_quickly() {
    let started = std::time::Instant::now();
    let error = render_template(
        "{% set ns = namespace(l=[1]) %}{% for i in range(24) %}{% set ns.l = ns.l + ns.l %}{% endfor %}{{ ns.l | length }}",
        &json!({}),
        None,
    )
    .unwrap_err();
    // The doubling stops at the sequence-item limit or the cumulative budget,
    // whichever it reaches first (the budget, for an empty context).
    assert!(format!("{error:#}").contains("resource limit"), "{error:#}");
    assert!(started.elapsed().as_secs_f64() < 2.0);
    // Within the limit the doubled list is exact.
    assert_eq!(
        rendered(
            "{% set ns = namespace(l=[1]) %}{% for i in range(10) %}{% set ns.l = ns.l + ns.l %}{% endfor %}{{ ns.l | length }}",
            json!({}),
        ),
        "1024"
    );
}

/// A guard refusal surfaces as a rendering error naming the tag.
#[test]
fn guard_refusals_are_rendering_errors() {
    let error = render_template("ok {{ a + }}", &json!({}), None).unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("cannot be bounded"), "{message}");
    assert!(message.contains("at byte 3"), "{message}");
}

/// Renders `source` with MiniJinja's own operators: every rewrite except the operator guard.
fn render_unguarded(source: &str, context: &serde_json::Value) -> Result<String, Error> {
    let source =
        parenthesize_conditional_keyword_arguments(&rewrite_generation_blocks(source)).into_owned();
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Lenient);
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    let budget = Arc::new(RenderBudget::new(RenderLimits::for_context_bytes(0)));
    pycompat::install(&mut environment, &budget);
    environment.render_str(&source, Value::from_serialize(context))
}

/// Strategy for scalar expressions over numbers, strings, and context names.
fn scalar_expression() -> impl Strategy<Value = String> {
    let leaf = prop_oneof![
        (0_i64..6).prop_map(|value| value.to_string()),
        Just("2.5".to_owned()),
        Just("-1".to_owned()),
        Just("'ab'".to_owned()),
        Just("''".to_owned()),
        Just("true".to_owned()),
        Just("none".to_owned()),
        Just("x".to_owned()),
        Just("s".to_owned()),
        Just("missing".to_owned()),
    ];
    leaf.prop_recursive(4, 32, 2, |inner| {
        let binary = prop::sample::select(vec![
            "+", "-", "*", "~", "/", "//", "%", "**", "==", "<", "and", "or", "in",
        ]);
        prop_oneof![
            (inner.clone(), binary, inner.clone())
                .prop_map(|(left, operator, right)| format!("{left} {operator} {right}")),
            inner.clone().prop_map(|operand| format!("({operand})")),
            inner.clone().prop_map(|operand| format!("-{operand}")),
            inner.clone().prop_map(|operand| format!("not {operand}")),
            inner
                .clone()
                .prop_map(|operand| format!("{operand} | string")),
            inner.clone().prop_map(|operand| format!("{operand} | abs")),
            (inner.clone(), inner.clone(), inner)
                .prop_map(|(yes, test, no)| format!("{yes} if {test} else {no}")),
        ]
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(768))]

    /// Guarded expressions render exactly as MiniJinja's own operators do.
    ///
    /// For scalar operands (where the bounded operators delegate to, or
    /// replicate, the engine) results and failures must agree: the rewrite
    /// preserves precedence, associativity, and evaluation order. The only
    /// exception is the guard's documented refusal of a variable named `not`
    /// used as an operand.
    #[test]
    fn guarded_expressions_match_minijinja(expression in scalar_expression()) {
        let source = format!("{{{{ {expression} }}}}|{{% set y = {expression} %}}{{{{ y }}}}");
        let context = json!({"x": 3, "s": "q"});
        let guarded = render_template(&source, &context, Some(0));
        let plain = render_unguarded(&source, &context);
        // The one documented refusal: MiniJinja reads `not` after a binary
        // operator as a variable, which the guard cannot rewrite faithfully.
        prop_assume!(!matches!(&guarded, Err(error) if error.to_string().contains("variable named `not`")));
        match (&guarded, &plain) {
            (Ok(guarded), Ok(plain)) => prop_assert_eq!(guarded, plain, "{}", source),
            (Err(_), Err(_)) => {}
            _ => prop_assert!(false, "{source}: guarded {guarded:?}, plain {plain:?}"),
        }
    }
}

/// Errors of operators the bounded filters delegate report the template's own line.
///
/// Review finding ENGINE-ERROR-LOCATION: the engine's operator ran in a helper
/// expression, so its errors said `<expression>:1` instead of the template's
/// line. They now carry no location of their own and the render attaches the
/// operator's line, as for the plain operator; rewritten tags also keep their
/// line breaks, so lines after a multi-line rewritten tag do not move.
#[test]
fn delegated_operator_errors_report_the_template_line() {
    let context = json!({"messages": [{"content": ["a", "b"]}]});
    for (source, line) in [
        ("first line\n{{ none + 1 }}", 2),
        (
            "{%- set ns = namespace(\n  a=1 +\n  2) -%}\n{{\n  'x' ~ ns.a\n}}\n{{ 'a' + messages[0]['content'] }}",
            7,
        ),
        ("{{ 1 * 2 }}\n\n{{ {'a': 1} * 2 }}", 3),
        ("\n\n\n{{ 170141183460469231731687303715884105727 + 1 }}", 4),
    ] {
        let guarded = render_template(source, &context, None).unwrap_err();
        let plain = render_unguarded(source, &context).unwrap_err();
        assert_eq!(guarded.line(), Some(line), "{source}: {guarded:#}");
        assert_eq!(guarded.line(), plain.line(), "{source}: {plain:#}");
        assert_eq!(guarded.kind(), plain.kind(), "{source}");
        assert_eq!(guarded.detail(), plain.detail(), "{source}");
        assert!(
            !format!("{guarded:#}").contains("<expression>"),
            "{guarded:#}"
        );
    }
}

/// Long literals and argument lists render like MiniJinja instead of being refused.
///
/// Review finding GUARD-FLAT-LITERAL-REFUSAL: the guard capped flat
/// comma-separated lists at 1 024 items with a message about operator chains.
#[test]
fn long_flat_lists_render_like_minijinja() {
    let join = |item: &dyn Fn(usize) -> String| (0..1_100).map(item).collect::<Vec<_>>().join(", ");
    let plain_items = join(&|index| index.to_string());
    let guarded_items = join(&|index| format!("{index} + 1"));
    for source in [
        format!("{{% set xs = [{plain_items}] %}}{{{{ xs | length }}}}"),
        format!("{{{{ [{guarded_items}] | sum }}}}"),
        format!("{{{{ ({guarded_items}) | last }}}}"),
        format!(
            "{{{{ {{{}}} | length }}}}",
            join(&|index| format!("'k{index}': {index} * 2"))
        ),
        format!(
            "{{{{ dict({}) | length }}}}",
            join(&|index| format!("k{index}='a' ~ {index}"))
        ),
        format!(
            "{{% macro m({}) %}}{{{{ p1099 }}}}{{% endmacro %}}{{{{ m() }}}}",
            join(&|index| format!("p{index}={index} * 2"))
        ),
    ] {
        let guarded = render_template(&source, &json!({}), None)
            .unwrap_or_else(|error| panic!("{}: {error:#}", &source[..60]));
        let plain = render_unguarded(&source, &json!({})).unwrap();
        assert_eq!(guarded, plain, "{}", &source[..60]);
    }
}

/// The value types the typed template generator draws expressions of.
#[derive(Debug, Clone, Copy)]
enum Type {
    /// Integers and floats.
    Number,
    /// Strings.
    Text,
    /// Lists, tuples, and lazy sequences.
    Sequence,
    /// Maps.
    Map,
    /// Conditions.
    Boolean,
}

/// Parenthesizes `text` unless it is already a single primary.
///
/// Postfix syntax, filters, and tests bind tighter than every binary
/// operator, so a compound operand needs parentheses to keep the type the
/// generator meant (binary operands are left bare on purpose, to exercise
/// the guard's precedence). Quoted literals, bracketed literals, and
/// parenthesized groups are primaries; so is anything without a space.
fn atom(text: String) -> String {
    let enclosed = |open: char, close: char| {
        text.starts_with(open) && text.ends_with(close) && {
            let mut depth = 0_i32;
            let mut quote = None;
            let mut closes_last = true;
            for (index, character) in text.char_indices() {
                match (quote, character) {
                    (Some(q), c) if c == q => quote = None,
                    (Some(_), _) => {}
                    (None, '\'' | '"') => quote = Some(character),
                    (None, '(' | '[' | '{') => depth += 1,
                    (None, ')' | ']' | '}') => {
                        depth -= 1;
                        if depth == 0 && index + 1 < text.len() {
                            closes_last = false;
                        }
                    }
                    _ => {}
                }
            }
            closes_last
        }
    };
    let quoted = (text.starts_with('\'')
        && text.ends_with('\'')
        && text[1..text.len() - 1].find('\'').is_none())
        || (text.starts_with('"')
            && text.ends_with('"')
            && text[1..text.len() - 1].find('"').is_none());
    if !text.contains(' ')
        || quoted
        || enclosed('(', ')')
        || enclosed('[', ']')
        || enclosed('{', '}')
    {
        text
    } else {
        format!("({text})")
    }
}

/// Strategy for one of `options`, each built from sub-expressions one level shallower.
fn pick(options: Vec<BoxedStrategy<String>>) -> BoxedStrategy<String> {
    prop::strategy::Union::new(options).boxed()
}

/// Strategy for an expression of type `kind` with at most `depth` levels of nesting.
///
/// The grammar covers what the operator guard rewrites or must see through:
/// the guarded operators and their precedence neighbours, unary minus and
/// `not`, postfix attributes, subscripts, slices, and calls (methods with
/// arguments), filters with positional and keyword arguments,
/// parenthesis-free tests, list, tuple, and map literals, conditionals, and
/// `*`/`**` call arguments. Operands are typed so most expressions evaluate,
/// and `+`/`*` never combine sequences (whose list semantics are the
/// guard's documented difference from MiniJinja).
fn typed(kind: Type, depth: u32) -> BoxedStrategy<String> {
    let leaves: Vec<&'static str> = match kind {
        Type::Number => vec!["0", "1", "2", "5", "2.5", "-1", "x", "(l | length)"],
        Type::Text => vec!["'ab'", "''", "s", "'x y'", "\"q\""],
        Type::Sequence => vec!["l", "[1, 2]", "[]", "['a', s]", "(1, 'b')"],
        Type::Map => vec!["m", "{'a': 1}", "dict(a=2)"],
        Type::Boolean => vec!["true", "false", "x > 1", "s in 'qs'"],
    };
    let leaf = prop::sample::select(leaves).prop_map(str::to_owned).boxed();
    if depth == 0 {
        return leaf;
    }
    let number = || typed(Type::Number, depth - 1);
    let text = || typed(Type::Text, depth - 1);
    let sequence = || typed(Type::Sequence, depth - 1);
    let map = || typed(Type::Map, depth - 1);
    let boolean = || typed(Type::Boolean, depth - 1);
    let same = || typed(kind, depth - 1);
    let small = || prop::sample::select(vec!["0", "1", "2", "-1", "-2", "none", "x"]);
    let any = || pick(vec![number(), text(), sequence(), map(), boolean()]);
    let mut options = vec![
        leaf,
        same().prop_map(|inner| format!("({inner})")).boxed(),
        (same(), boolean(), same())
            .prop_map(|(yes, test, no)| format!("{yes} if {test} else {no}"))
            .boxed(),
    ];
    match kind {
        Type::Number => options.extend([
            (
                number(),
                prop::sample::select(vec!["+", "-", "*", "/", "//", "%", "**"]),
                number(),
            )
                .prop_map(|(left, operator, right)| format!("{left} {operator} {right}"))
                .boxed(),
            number().prop_map(|operand| format!("-{operand}")).boxed(),
            text()
                .prop_map(|operand| format!("{} | length", atom(operand)))
                .boxed(),
            sequence()
                .prop_map(|operand| format!("{} | length", atom(operand)))
                .boxed(),
            number()
                .prop_map(|operand| format!("{} | abs", atom(operand)))
                .boxed(),
            number()
                .prop_map(|operand| format!("{} | round(1)", atom(operand)))
                .boxed(),
            map()
                .prop_map(|operand| format!("{}.a | default(0)", atom(operand)))
                .boxed(),
            (sequence(), small())
                .prop_map(|(operand, index)| format!("{}[{index}] | default(7)", atom(operand)))
                .boxed(),
            sequence()
                .prop_map(|operand| format!("dict(**{{'k': {operand}}}).k | length"))
                .boxed(),
        ]),
        Type::Text => options.extend([
            (text(), prop::sample::select(vec!["+", "~"]), text())
                .prop_map(|(left, operator, right)| format!("{left} {operator} {right}"))
                .boxed(),
            (text(), any())
                .prop_map(|(left, right)| format!("{left} ~ {right}"))
                .boxed(),
            (text(), small())
                .prop_map(|(left, count)| format!("{left} * {count}"))
                .boxed(),
            (small(), text())
                .prop_map(|(count, right)| format!("{count} * {right}"))
                .boxed(),
            (text(), small(), small(), small())
                .prop_map(|(operand, start, stop, step)| {
                    format!("{}[{start}:{stop}:{step}]", atom(operand))
                })
                .boxed(),
            (text(), small())
                .prop_map(|(operand, start)| format!("{}[{start}:]", atom(operand)))
                .boxed(),
            text()
                .prop_map(|operand| format!("{}[::-1]", atom(operand)))
                .boxed(),
            text()
                .prop_map(|operand| format!("{}[1:].upper()", atom(operand)))
                .boxed(),
            text()
                .prop_map(|operand| format!("{} | upper", atom(operand)))
                .boxed(),
            (text(), text(), text())
                .prop_map(|(operand, from, to)| {
                    format!("{} | replace({from}, {to})", atom(operand))
                })
                .boxed(),
            text()
                .prop_map(|operand| format!("{}.strip('a')", atom(operand)))
                .boxed(),
            (sequence(), text())
                .prop_map(|(operand, separator)| format!("{} | join({separator})", atom(operand)))
                .boxed(),
            (sequence(), text())
                .prop_map(|(operand, separator)| format!("{} | join(d={separator})", atom(operand)))
                .boxed(),
            any()
                .prop_map(|operand| format!("{} | string", atom(operand)))
                .boxed(),
            sequence()
                .prop_map(|operand| format!("'%s|%s' | format(*{operand})"))
                .boxed(),
            (text(), small())
                .prop_map(|(operand, index)| format!("{}[{index}]", atom(operand)))
                .boxed(),
        ]),
        Type::Sequence => options.extend([
            (sequence(), small(), small())
                .prop_map(|(operand, start, stop)| format!("{}[{start}:{stop}]", atom(operand)))
                .boxed(),
            sequence()
                .prop_map(|operand| format!("{}[::-1]", atom(operand)))
                .boxed(),
            sequence()
                .prop_map(|operand| format!("{} | list", atom(operand)))
                .boxed(),
            text()
                .prop_map(|operand| format!("{} | list", atom(operand)))
                .boxed(),
            sequence()
                .prop_map(|operand| format!("{} | reverse | list", atom(operand)))
                .boxed(),
            text()
                .prop_map(|operand| format!("{}.split()", atom(operand)))
                .boxed(),
            text()
                .prop_map(|operand| format!("{}.split('a')[1:]", atom(operand)))
                .boxed(),
            (any(), any())
                .prop_map(|(first, second)| format!("[{first}, {second}]"))
                .boxed(),
            (any(), any())
                .prop_map(|(first, second)| format!("({first}, {second},)"))
                .boxed(),
            sequence()
                .prop_map(|operand| format!("{} | batch(2) | list", atom(operand)))
                .boxed(),
            sequence()
                .prop_map(|operand| format!("{} | map('string') | list", atom(operand)))
                .boxed(),
            sequence()
                .prop_map(|operand| format!("{} | select | list", atom(operand)))
                .boxed(),
            map()
                .prop_map(|operand| format!("{}.keys()", atom(operand)))
                .boxed(),
            map()
                .prop_map(|operand| format!("{} | dictsort", atom(operand)))
                .boxed(),
            sequence()
                .prop_map(|operand| format!("range(*{}[:2]) | list", atom(operand)))
                .boxed(),
        ]),
        Type::Map => options.extend([
            (text(), any())
                .prop_map(|(key, value)| format!("{{{key}: {value}, 'z': 0}}"))
                .boxed(),
            (map(), any())
                .prop_map(|(base, value)| format!("dict({base}, k={value})"))
                .boxed(),
            map()
                .prop_map(|operand| format!("dict(**{operand})"))
                .boxed(),
        ]),
        Type::Boolean => options.extend([
            (
                number(),
                prop::sample::select(vec!["==", "!=", "<", ">=", "in"]),
                number(),
            )
                .prop_map(|(left, operator, right)| format!("{left} {operator} {right}"))
                .boxed(),
            (text(), text())
                .prop_map(|(left, right)| format!("{left} not in {right}"))
                .boxed(),
            boolean()
                .prop_map(|operand| format!("not {operand}"))
                .boxed(),
            (
                boolean(),
                prop::sample::select(vec!["and", "or"]),
                boolean(),
            )
                .prop_map(|(left, operator, right)| format!("{left} {operator} {right}"))
                .boxed(),
            (number(), small())
                .prop_map(|(operand, divisor)| {
                    format!("{} is divisibleby {divisor}", atom(operand))
                })
                .boxed(),
            (text(), text())
                .prop_map(|(left, right)| format!("{} is sameas {}[1:]", atom(left), atom(right)))
                .boxed(),
            any()
                .prop_map(|operand| format!("{} is not none", atom(operand)))
                .boxed(),
            any()
                .prop_map(|operand| format!("{} is string", atom(operand)))
                .boxed(),
            (text(), text())
                .prop_map(|(left, right)| format!("{}.startswith({right})", atom(left)))
                .boxed(),
        ]),
    }
    pick(options)
}

/// Strategy for a template of one to four lines, each a statement or expression tag.
///
/// Every line holds one construct, so a failing line has the same number with
/// and without the guard.
fn typed_template() -> impl Strategy<Value = String> {
    let any = || {
        pick(vec![
            typed(Type::Number, 2),
            typed(Type::Text, 2),
            typed(Type::Sequence, 2),
            typed(Type::Map, 2),
            typed(Type::Boolean, 2),
        ])
    };
    let line = pick(vec![
        any().prop_map(|value| format!("{{{{ {value} }}}}")).boxed(),
        any().prop_map(|value| format!("{{% set y = {value} %}}{{{{ y }}}}")).boxed(),
        (typed(Type::Sequence, 2), typed(Type::Boolean, 1))
            .prop_map(|(items, test)| {
                format!("{{% for i in ({items}) if {test} %}}{{{{ i }}}},{{% endfor %}}")
            })
            .boxed(),
        (typed(Type::Boolean, 2), any())
            .prop_map(|(test, value)| {
                format!("{{% if {test} %}}{{{{ {value} }}}}{{% elif not {test} %}}b{{% endif %}}")
            })
            .boxed(),
        (any(), any())
            .prop_map(|(first, second)| {
                format!("{{% set a, b = {first}, {second} %}}{{{{ a }}}}|{{{{ b }}}}")
            })
            .boxed(),
        (any(), any())
            .prop_map(|(default, argument)| {
                format!(
                    "{{% macro mm(p={default}) %}}[{{{{ p }}}}]{{% endmacro %}}{{{{ mm() }}}}{{{{ mm({argument}) }}}}"
                )
            })
            .boxed(),
        any()
            .prop_map(|value| format!("{{% filter upper %}}{{{{ {value} }}}}{{% endfilter %}}"))
            .boxed(),
        any()
            .prop_map(|value| format!("{{% set t %}}{{{{ {value} }}}}{{% endset %}}{{{{ t }}}}"))
            .boxed(),
        any()
            .prop_map(|value| format!("{{% with w = {value} %}}{{{{ w }}}}{{% endwith %}}"))
            .boxed(),
        (any(), any())
            .prop_map(|(first, second)| {
                format!(
                    "{{% set ns = namespace(v={first}) %}}{{% set ns.v = {second} %}}{{{{ ns.v }}}}"
                )
            })
            .boxed(),
    ]);
    prop::collection::vec(line, 1..=4).prop_map(|lines| lines.join("\n"))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Typed templates render exactly as MiniJinja does, failures included.
    ///
    /// Review finding GUARD-DIFFERENTIAL-COVERAGE: the scalar property above
    /// never generated postfix syntax, filters with arguments, tests without
    /// parentheses, calls with keyword or `*`/`**` arguments, literals, or
    /// statement tags. Here successes must print the same text, and failures
    /// must agree on the error kind, detail, and line.
    #[test]
    fn typed_templates_match_minijinja(source in typed_template()) {
        let context = json!({"x": 3, "s": "q", "l": [1, 2, 3], "m": {"a": 1, "b": [1, 2]}});
        let guarded = render_template(&source, &context, Some(0));
        // Plain MiniJinja panics on `n is divisibleby 0`; the sandbox reports an error.
        let Ok(plain) = std::panic::catch_unwind(|| render_unguarded(&source, &context)) else {
            prop_assert!(guarded.is_err(), "{}: plain MiniJinja panicked", source);
            return Ok(());
        };
        match (&guarded, &plain) {
            (Ok(guarded), Ok(plain)) => prop_assert_eq!(guarded, plain, "{}", source),
            // The guard refuses only tags MiniJinja cannot parse either.
            (Err(guarded), Err(_)) if guarded.to_string().contains("cannot be bounded") => {}
            (Err(guarded), Err(plain)) => {
                prop_assert_eq!(guarded.kind(), plain.kind(), "{}: {:#} vs {:#}", source, guarded, plain);
                prop_assert_eq!(guarded.detail(), plain.detail(), "{}", source);
                prop_assert_eq!(guarded.line(), plain.line(), "{}", source);
            }
            _ => prop_assert!(false, "{source}: guarded {guarded:?}, plain {plain:?}"),
        }
    }
}

/// Representative template idioms render identically with and without the guard.
#[test]
fn official_template_idioms_render_like_minijinja() {
    let context = json!({
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
        ],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "bos_token": "<s>",
        "add_generation_prompt": true,
    });
    for source in [
        "{{ bos_token }}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
        "{%- set ns = namespace(system='', index=0) -%}{%- for m in messages -%}{%- if m.role == 'system' -%}{%- set ns.system = ns.system ~ m.content ~ '\\n\\n' -%}{%- endif -%}{%- set ns.index = ns.index + 1 -%}{%- endfor -%}[{{ ns.system | trim }}|{{ ns.index }}]",
        "{%- for m in messages[1:] -%}{{ (loop.index0 * 2 + 1) ~ ':' ~ m.content | upper }}{%- if not loop.last %}, {% endif -%}{%- endfor -%}",
        "{%- for tool in tools %}{{ '- ' + tool.function.name + ': ' + (tool.function.parameters | tojson) + '\\n' }}{% endfor -%}",
        "{%- for m in messages if m.tool_calls is defined -%}{%- for call in m.tool_calls -%}{{ '<tool_call>' ~ {'name': call.function.name, 'arguments': call.function.arguments} | tojson ~ '</tool_call>' }}{%- endfor -%}{%- endfor -%}",
        "{%- set content = messages[-1].content -%}{{ '-' * 3 }}{{ content[:2] ~ ('…' if content | length > 2 else '') }}{{ '=' * (messages | length) }}",
        "{%- macro render(m, sep='\\n') -%}{{ m.role | capitalize ~ ': ' ~ m.content ~ sep }}{%- endmacro -%}{%- for m in messages %}{{ render(m, sep=' | ' if loop.last else '\\n') }}{% endfor %}",
    ] {
        let guarded = render_template(source, &context, Some(0))
            .unwrap_or_else(|error| panic!("{source}: {error:#}"));
        let plain = render_unguarded(source, &context)
            .unwrap_or_else(|error| panic!("{source}: {error:#}"));
        assert_eq!(guarded, plain, "{source}");
    }
}
