//! Sandbox tests: limit arithmetic, bounded operators and builtins, output bounds, and panics.

use std::{io::Write as _, sync::atomic::AtomicUsize};

use serde_json::json;

use super::super::{render_template, render_template_with};
use super::*;

/// Small limits that make every bound cheap to reach.
const SMALL: RenderLimits = RenderLimits {
    output_bytes: 256,
    value_bytes: 64,
    sequence_items: 8,
    total_bytes: 1024,
};

/// Renders `source` under `limits` with an empty context and a pinned clock.
fn render_with(source: &str, limits: RenderLimits) -> Result<String, Error> {
    render_template_with(source, &json!({}), Some(0), limits, |_| {})
}

/// Renders `source` under generous default limits.
fn render(source: &str) -> Result<String, Error> {
    render_template(source, &json!({}), Some(0))
}

/// Asserts that rendering fails with a resource-limit error.
fn assert_limited(source: &str, limits: RenderLimits) {
    let error = render_with(source, limits).expect_err(source);
    assert!(
        error.to_string().contains("resource limit"),
        "{source}: {error}"
    );
}

/// Limits scale with the serialized context and saturate instead of overflowing.
#[test]
fn limits_follow_the_context_size() {
    let empty = RenderLimits::for_context_bytes(0);
    assert_eq!(empty.output_bytes, 1024 * 1024);
    assert_eq!(empty.value_bytes, empty.output_bytes);
    assert_eq!(empty.sequence_items, 1 << 20);
    assert_eq!(empty.total_bytes, 16 * empty.output_bytes);
    let large = RenderLimits::for_context_bytes(1 << 20);
    assert_eq!(large.output_bytes, empty.output_bytes + 4 * (1 << 20));
    let huge = RenderLimits::for_context_bytes(usize::MAX);
    assert_eq!(huge.output_bytes, usize::MAX);
    assert_eq!(huge.total_bytes, usize::MAX);
    let context = json!({"messages": [{"role": "user", "content": "hello"}]});
    let measured = RenderLimits::for_context(&context);
    let bytes = serde_json::to_vec(&context).unwrap().len();
    assert_eq!(measured, RenderLimits::for_context_bytes(bytes));
    let mut counter = ByteCounter(0);
    counter.flush().unwrap();
    assert_eq!(counter.write(b"abc").unwrap(), 3);
}

/// The budget refuses oversized values and charges production cumulatively.
#[test]
fn budget_checks_values_items_and_totals() {
    let budget = RenderBudget::new(SMALL);
    assert_eq!(budget.limits(), SMALL);
    assert_eq!(budget.text_allowance(), 64);
    budget.reserve_text(64, "a").unwrap();
    assert_eq!(budget.produced(), 64);
    let error = budget.reserve_text(65, "big").unwrap_err().to_string();
    assert!(error.contains("big would produce 65 bytes"), "{error}");
    budget.reserve_items(8, "list").unwrap();
    assert_eq!(budget.produced(), 64 + 8 * size_of::<Value>());
    assert!(budget.reserve_items(9, "list").is_err());
    // Exhaust the total: the allowance shrinks to what remains.
    while budget.reserve_text(64, "fill").is_ok() {}
    assert!(budget.text_allowance() < 64);
    let error = budget.reserve_text(64, "more").unwrap_err().to_string();
    assert!(error.contains("1024-byte budget"), "{error}");
    // A charge that would overflow `usize` is refused, not wrapped.
    assert!(
        RenderBudget::new(RenderLimits::for_context_bytes(usize::MAX))
            .reserve_text(usize::MAX, "x")
            .is_ok()
    );
}

/// Limited text refuses the write that would pass its limit.
#[test]
fn limited_text_stops_at_its_limit() {
    let mut text = LimitedText::new(5);
    assert!(fmt::Write::write_str(&mut text, "abc").is_ok());
    assert!(!text.exceeded());
    assert!(fmt::Write::write_str(&mut text, "def").is_err());
    assert!(text.exceeded());
    assert_eq!(text.into_string(), "abc");
    let budget = RenderBudget::new(SMALL);
    assert_eq!(
        display_within(&budget, &Value::from(vec![1, 2]), "print").unwrap(),
        "[1, 2]"
    );
    let long = Value::from(vec![Value::from("x".repeat(40)); 2]);
    assert!(display_within(&budget, &long, "print").is_err());
}

/// The bounded operators keep MiniJinja's results for every supported operand pair.
#[test]
fn bounded_operators_keep_minijinja_semantics() {
    let cases = [
        (
            "{{ 1 + 2 }}|{{ 1.5 + 1 }}|{{ 'a' + 'b' }}|{{ true + 1 }}",
            "3|2.5|ab|2",
        ),
        (
            "{{ 2 * 3 }}|{{ 2.5 * 2 }}|{{ 'ab' * 3 }}|{{ 2 * 'xy' }}|{{ 'a' * 0 }}",
            "6|5.0|ababab|xyxy|",
        ),
        (
            "{{ 1 ~ 2 }}|{{ none ~ true }}|{{ missing ~ 'x' }}|{{ [1] ~ 'y' }}",
            "12|nonetrue|x|[1]y",
        ),
        // Sequences become lists, as in Python: indexable and printed with Python quotes.
        (
            "{{ [1] + [2, 3] }}|{{ ([1] + [2])[1] }}|{{ (['a'] * 2) }}|{{ ([1, 2] * 2) | length }}",
            "[1, 2, 3]|2|['a', 'a']|4",
        ),
        ("{{ range(2) + [5] }}|{{ [0] * 0 }}", "[0, 1, 5]|[]"),
        // Integer arithmetic stays exact and checked.
        (
            "{{ 9007199254740993 + 1 }}|{{ 2 ** 64 * 2 }}",
            "9007199254740994|36893488147419103232",
        ),
    ];
    for (source, expected) in cases {
        assert_eq!(render(source).unwrap(), expected, "{source}");
    }
    for (source, message) in [
        (
            "{{ 'a' * -1 }}",
            "strings can only be multiplied with integers",
        ),
        (
            "{{ 'a' * 1.5 }}",
            "strings can only be multiplied with integers",
        ),
        (
            "{{ [1] * 'x' }}",
            "strings can only be multiplied with integers",
        ),
        (
            "{{ [1] * -2 }}",
            "sequences and iterables can only be multiplied with integers",
        ),
        // A map on the left is the candidate, so this is the engine's type error.
        ("{{ {'a': 1} * [1] }}", "unsupported types"),
        ("{{ {} + 1 }}", "unsupported types"),
        ("{{ 'a' + 1 }}", "unsupported types"),
        ("{{ missing + 1 }}", "unsupported types"),
        (
            "{{ 170141183460469231731687303715884105727 + 1 }}",
            "unable to calculate",
        ),
    ] {
        let error = render(source).expect_err(source).to_string();
        assert!(error.contains(message), "{source}: {error}");
    }
}

/// Each operator refuses a result past the per-value limits before allocating it.
#[test]
fn bounded_operators_refuse_oversized_results() {
    for source in [
        "{{ 'ab' * 33 }}",
        "{{ 'ab' * 4611686018427387904 }}",
        "{{ 'x' * 18446744073709551615 }}",
        "{% set s = 'x' * 40 %}{{ s + s }}",
        "{% set s = 'x' * 40 %}{{ s ~ s }}",
        "{{ [1, 2, 3] * 3 }}",
        "{{ [1] * 18446744073709551615 }}",
        "{{ range(5) + range(5) }}",
        "{% set ns = namespace(l=[1]) %}{% for i in range(30) %}{% set ns.l = ns.l + ns.l %}{% endfor %}",
    ] {
        assert_limited(source, SMALL);
    }
    // At the limits themselves, the operators succeed.
    assert_eq!(render_with("{{ 'ab' * 32 }}", SMALL).unwrap().len(), 64);
    assert_eq!(
        render_with("{{ ([1, 2] * 4) | length }}", SMALL).unwrap(),
        "8"
    );
}

/// The cumulative budget stops many small productions that stay alive together.
#[test]
fn cumulative_production_is_bounded() {
    assert_limited(
        "{% set ns = namespace(l=[]) %}{% for i in range(100) %}{% set ns.l = ns.l + ['x' * 60] %}{% endfor %}",
        SMALL,
    );
    // Prints are charged too, including prints into captured blocks.
    assert_limited(
        "{% for i in range(100) %}{% set s %}{{ 'x' * 60 }}{% endset %}{% endfor %}",
        SMALL,
    );
}

/// The rendered prompt is bounded however it is printed.
#[test]
fn rendered_output_is_bounded() {
    let error = render_with("{% for i in range(10) %}{{ 'x' * 30 }}{% endfor %}", SMALL)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("rendered prompt would exceed 256 bytes"),
        "{error}"
    );
    // Literal template text counts as well.
    let text = "y".repeat(300);
    assert_limited(&text, SMALL);
    assert_eq!(render_with(&"y".repeat(256), SMALL).unwrap().len(), 256);
    let mut output = BoundedOutput::new(3);
    assert_eq!(io::Write::write(&mut output, b"ab").unwrap(), 2);
    assert!(io::Write::write(&mut output, b"cd").is_err());
    assert!(output.exceeded());
    io::Write::flush(&mut output).unwrap();
    assert_eq!(output.into_string().unwrap(), "ab");
    let mut invalid = BoundedOutput::new(3);
    io::Write::write_all(&mut invalid, &[0xff]).unwrap();
    assert!(invalid.into_string().is_err());
}

/// Builtins whose size an argument controls are checked before they allocate.
#[test]
fn sized_builtins_are_bounded() {
    for source in [
        "{{ 'a\nb' | indent(100) }}",
        "{{ 'a\nb' | indent(width=100) }}",
        "{{ ('x' * 20) | replace('', 'yyyy') }}",
        "{{ ('x' * 20) | replace('x', 'yyyyy') }}",
        "{{ '%100s' | format('a') }}",
        "{{ '%.100f' | format(1.0) }}",
        "{{ '%99999999999999999999999s' | format('a') }}",
        "{{ [1] | batch(100, 0) }}",
        "{{ [1] | slice(100) }}",
        "{{ range(5) | chain(range(5)) | list }}",
    ] {
        assert_limited(source, SMALL);
    }
    let cases = [
        ("{{ 'a\nb' | indent(2) }}", "a\n  b"),
        ("{{ 'a\nb' | indent(width=1, first=true) }}", " a\n b"),
        ("{{ 'aXa' | replace('X', '--') }}", "a--a"),
        ("{{ '%s-%03d|%%' | format('x', 7) }}", "x-007|%"),
        ("{{ [1, 2, 3] | batch(2, 0) | list }}", "[[1, 2], [3, 0]]"),
        ("{{ [1, 2, 3] | slice(2) | list }}", "[[1, 2], [3]]"),
        ("{{ [1] | chain([2], [3]) }}", "[1, 2, 3]"),
        ("{{ {'a': 1} | chain({'b': 2}) | length }}", "2"),
    ];
    for (source, expected) in cases {
        assert_eq!(render_with(source, SMALL).unwrap(), expected, "{source}");
    }
    assert_eq!(replaced_length("abc", "", "xy", None), 3 + 4 * 2);
    assert_eq!(replaced_length("aaaa", "a", "bb", Some(2)), 6);
    assert_eq!(replaced_length("aaaa", "aa", "", None), 4);
}

/// Builtins that print arbitrary values with `Display` are bounded while printing.
#[test]
fn printing_builtins_are_bounded() {
    // A container that holds one long string twice prints twice its length.
    let doubled = "{% set s = 'x' * 40 %}{% set d = [s, s] %}";
    for filter in [
        "upper",
        "lower",
        "title",
        "capitalize",
        "escape",
        "e",
        "pprint",
        "string",
        "trim",
        "safe",
        "tojson",
    ] {
        assert_limited(&format!("{doubled}{{{{ d | {filter} }}}}"), SMALL);
    }
    for test in ["startingwith('x')", "endingwith('x')"] {
        assert_limited(&format!("{doubled}{{{{ d is {test} }}}}"), SMALL);
    }
    assert_limited(&format!("{doubled}{{{{ d }}}}"), SMALL);
    assert_limited(&format!("{doubled}{{{{ d | join('') }}}}"), SMALL);
    assert_limited(&format!("{doubled}{{{{ d ~ '' }}}}"), SMALL);
    let cases = [
        (
            "{{ 'aB' | upper }}|{{ 3 | lower }}|{{ 'a b' | title }}|{{ 'aB' | capitalize }}",
            "AB|3|A B|Ab",
        ),
        (
            "{{ '<a>' | e }}|{{ 1 | escape }}|{{ '<b>' | safe | e }}",
            "&lt;a&gt;|1|<b>",
        ),
        (
            "{{ 'abc' is startingwith('a') }}|{{ 12 is endingwith(2) }}",
            "True|True",
        ),
        ("{{ [1] | pprint }}", "[\n    1,\n]"),
    ];
    for (source, expected) in cases {
        assert_eq!(render_with(source, SMALL).unwrap(), expected, "{source}");
    }
    let error = render_with("{{ 'a' | upper(**{'x': 1}) }}", SMALL).unwrap_err();
    assert!(!error.to_string().contains("resource limit"), "{error}");
}

/// Python-compatible methods check their result size before allocating.
#[test]
fn python_methods_are_bounded() {
    for source in [
        "{{ ('x' * 20).replace('x', 'yyyyy') }}",
        "{{ 'ab'.replace('', 'z' * 30) }}",
        "{{ ('y' * 40).join(['a', 'b', 'c']) }}",
        "{{ ('ß' * 15).upper() }}",
    ] {
        assert_limited(source, SMALL);
    }
    assert_eq!(
        render_with(
            "{{ 'a-b'.replace('-', '+') }}|{{ ','.join(['a', 'b']) }}|{{ 'ß'.upper() }}",
            SMALL
        )
        .unwrap(),
        "a+b|a,b|SS"
    );
}

/// Renders `source` on a plain MiniJinja environment with only the Python-compatibility layer.
///
/// This is the reference for the bounded replacements: the same builtins
/// without any charge, and no operator guard.
fn render_plain(source: &str) -> Result<String, Error> {
    let mut environment = Environment::new();
    environment.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
    let budget = Arc::new(RenderBudget::new(RenderLimits::for_context_bytes(0)));
    super::super::super::pycompat::install(&mut environment, &budget);
    environment.render_str(source, ())
}

/// Builtins and methods that copy their input are charged, so retained copies stay within the budget.
///
/// Review finding CHAT-UNCHARGED-RETENTION: `s | list` (one 24-byte `Value`
/// per character) or `s | reverse` kept alive in a namespace in a loop
/// reached 467 MiB against a 16 MiB budget, and the render succeeded. Every
/// copying builtin, function, and Python method now charges its result before
/// it is built, while its output stays exactly MiniJinja's.
#[test]
fn copying_builtins_are_charged() {
    let prelude = "{% set s = 'abcdefgh' %}{% set m = {'a': 1, 'b': 2, 'c': 3} %}";
    let copies = [
        ("s | list", "['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']"),
        ("s | reverse", "hgfedcba"),
        ("s | sort | join", "abcdefgh"),
        ("s | unique | list | length", "8"),
        ("s | map('upper') | join", "ABCDEFGH"),
        ("s | select('in', 'aeiou') | join", "ae"),
        ("s | reject('in', 'aeiou') | join", "bcdfgh"),
        ("[m] | selectattr('a') | length", "1"),
        ("[m] | rejectattr('a') | length", "0"),
        ("m | dictsort", "[['a', 1], ['b', 2], ['c', 3]]"),
        ("[m, m] | groupby('a') | length", "1"),
        ("'a\nb' | lines", "['a', 'b']"),
        ("dict(m, d=4) | length", "4"),
        ("m.items() | length", "3"),
        ("m.keys()", "['a', 'b', 'c']"),
        ("m.values()", "[1, 2, 3]"),
        ("(' ' ~ s ~ ' ').strip()", "abcdefgh"),
        ("s.split('d')", "['abc', 'efgh']"),
        ("'a\nb'.splitlines()", "['a', 'b']"),
        ("s.removeprefix('ab')", "cdefgh"),
        ("range(3) | reverse | list", "[2, 1, 0]"),
        ("(s | list)[::-1] | reverse | join", "abcdefgh"),
    ];
    for (copy, expected) in copies {
        let source = format!("{prelude}{{{{ {copy} }}}}");
        assert_eq!(render(&source).unwrap(), expected, "{copy}");
        assert_eq!(
            render(&source).unwrap(),
            render_plain(&source).unwrap(),
            "{copy}"
        );
        // 200 retained copies of at least 6 bytes or one item exceed 1 024 bytes.
        let retained = format!(
            "{prelude}{{% set ns = namespace(l=none) %}}{{% for i in range(200) %}}{{% set ns.l = [ns.l, {copy}] %}}{{% endfor %}}ok"
        );
        assert_limited(&retained, SMALL);
    }
    // A collected lazy iterable is charged for the collection, too.
    assert_eq!(
        render("{{ 'a b' | split | list }}|{{ 'a b' | split | reverse | list }}").unwrap(),
        "['a', 'b']|['b', 'a']"
    );
    // `debug()` would print the whole engine state without a limit; it is not a global.
    assert_eq!(render("{{ debug is defined }}").unwrap(), "False");
    assert!(render("{{ debug() }}").is_err());
}

/// Slices and splatted call arguments, which the engine copies, are charged; lazy views are free.
///
/// The guard rewrites `value[a:b:c]` into the bounded slice filter and
/// `f(*x, **y)` into charged arguments; the results stay exactly MiniJinja's.
#[test]
fn slices_and_splats_are_charged() {
    let prelude = "{% set s = 'abcdefgh' %}{% set l = [1, 2, 3, 4, 5, 6, 7, 8] %}{% set m = {'a': 1, 'b': 2} %}";
    let cases = [
        ("s[1:]", "bcdefgh"),
        ("s[::-1]", "hgfedcba"),
        ("s[1:6:2]", "bdf"),
        ("s[-3:]", "fgh"),
        ("s[none:2]", "ab"),
        ("l[2:4] | list", "[3, 4]"),
        ("l[::-3] | list", "[8, 5, 2]"),
        ("range(*[1, 3]) | list", "[1, 2]"),
        ("dict(**m) | length", "2"),
        ("[s[1:]] | first", "bcdefgh"),
        // Undefined values slice to an empty list, as in MiniJinja.
        ("missing[1:] | list", "[]"),
        ("(s ~ s)[7:9]", "ha"),
    ];
    for (copy, expected) in cases {
        let source = format!("{prelude}{{{{ {copy} }}}}");
        assert_eq!(render(&source).unwrap(), expected, "{copy}");
        assert_eq!(render_plain(&source).unwrap(), expected, "{copy}");
    }
    for copy in [
        "s[1:]",
        "s[::-1]",
        "'%s%s%s%s%s%s%s%s' | format(*l)",
        "dict(**m) | length",
    ] {
        let retained = format!(
            "{prelude}{{% set ns = namespace(l=none) %}}{{% for i in range(200) %}}{{% set ns.l = [ns.l, {copy}] %}}{{% endfor %}}ok"
        );
        assert_limited(&retained, SMALL);
    }
    // A sequence slice is a lazy view of the same items and costs nothing.
    let views = format!(
        "{prelude}{{% set ns = namespace(l=none) %}}{{% for i in range(200) %}}{{% set ns.l = [ns.l, l[1:]] %}}{{% endfor %}}ok"
    );
    assert_eq!(render_with(&views, SMALL).unwrap(), "ok");
    // Engine errors keep MiniJinja's message.
    let error = render(&format!("{prelude}{{{{ s[::0] }}}}")).unwrap_err();
    assert_eq!(error.detail(), Some("cannot slice by step size of 0"));
    let error = render(&format!("{prelude}{{{{ 3[1:] }}}}")).unwrap_err();
    assert_eq!(
        error.detail(),
        Some("value of type number cannot be sliced")
    );
}

/// `divisibleby` reports an integer zero divisor instead of panicking inside MiniJinja.
///
/// Found by the typed differential property: MiniJinja's test computes
/// `a % b` on `i128` unchecked, so `3 is divisibleby 0` panicked (and
/// `i128::MIN is divisibleby -1` overflowed). Jinja2 raises
/// `ZeroDivisionError`; every other answer is MiniJinja's.
#[test]
fn divisibleby_never_panics() {
    let error = render("{{ 3 is divisibleby 0 }}").unwrap_err();
    assert_eq!(
        error.detail(),
        Some("divisibleby: integer division or modulo by zero")
    );
    let cases = [
        ("{{ 6 is divisibleby 3 }}", "True"),
        ("{{ 7 is divisibleby(3) }}", "False"),
        ("{{ 6 is divisibleby -1 }}", "True"),
        (
            "{{ -170141183460469231731687303715884105728 is divisibleby -1 }}",
            "True",
        ),
        ("{{ 3.0 is divisibleby 0 }}", "False"),
        ("{{ 4.5 is divisibleby 1.5 }}", "True"),
        ("{{ 'a' is divisibleby 2 }}", "False"),
    ];
    for (source, expected) in cases {
        assert_eq!(render(source).unwrap(), expected, "{source}");
    }
}

/// `strftime_now` checks its largest possible result before formatting and charges its result.
#[test]
fn strftime_is_bounded() {
    assert_limited("{{ strftime_now('%c' * 30) }}", SMALL);
    assert_eq!(
        render("{{ strftime_now('%Y-%m-%d') }}").unwrap(),
        "1970-01-01"
    );
    // SMALL allows 64-byte values: a 4-byte format may expand to 48 bytes,
    // a 6-byte one to 72, which is refused before formatting.
    assert_eq!(
        render_with("{{ strftime_now('%c%c') }}", SMALL).unwrap(),
        "Thu Jan  1 00:00:00 1970Thu Jan  1 00:00:00 1970"
    );
    assert_limited("{{ strftime_now('%c%c%c') }}", SMALL);
    // No directive expands by more than `STRFTIME_EXPANSION`.
    let civil = super::super::calendar::civil_time(253_402_300_799).unwrap();
    for directive in (b'A'..=b'Z').chain(b'a'..=b'z').map(char::from) {
        for format in [format!("%{directive}"), format!("%-{directive}")] {
            if let Ok(text) = super::super::strftime(&format, &civil) {
                assert!(
                    text.len() <= 2 * super::super::STRFTIME_EXPANSION,
                    "{format}: {text}"
                );
            }
        }
    }
}

/// A panic inside the engine or a callback becomes a contextual error.
#[test]
fn panics_never_escape_rendering() {
    let error = render_template_with(
        "{{ explode() }}",
        &json!({}),
        Some(0),
        RenderLimits::for_context_bytes(0),
        |environment| {
            environment.add_function("explode", || -> Result<Value, Error> {
                panic!("kaboom");
            });
        },
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("panicked (kaboom)"), "{error}");
    let error = render_template_with(
        "{{ explode() }}",
        &json!({}),
        Some(0),
        RenderLimits::for_context_bytes(0),
        |environment| {
            environment.add_function("explode", || -> Result<Value, Error> {
                std::panic::panic_any(7_u8);
            });
        },
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("non-text panic payload"), "{error}");
}

/// Format widths are read like printf: flags, width, then precision.
#[test]
fn format_widths_follow_printf() {
    let budget = RenderBudget::new(SMALL);
    for format in [
        "%s",
        "%5d",
        "%-10s",
        "%.3f",
        "%08.2f",
        "100%",
        "%%",
        "%",
        "%%99999999",
        "%(a)5s",
        "%(unterminated",
    ] {
        check_format_widths(&budget, format).unwrap_or_else(|error| panic!("{format}: {error}"));
    }
    for format in [
        "%65s",
        "%.65f",
        "%1.65f",
        "%#0+ -99s",
        "%% %65s",
        "%(a)65s",
        "%(a)-.65f",
    ] {
        assert!(check_format_widths(&budget, format).is_err(), "{format}");
    }
    // A mapping key works end to end, as in Python's `'%(a)s' % {'a': 1}`.
    assert_eq!(
        render_with("{{ '%(a)3s|%%' | format({'a': 'x'}) }}", SMALL).unwrap(),
        "  x|%"
    );
}

/// A template value that counts its own destruction, to observe whether a render leaked it.
///
/// Each test owns its counter, so tests running in parallel cannot disturb
/// one another's counts.
#[derive(Debug)]
struct Probe(
    /// Drop counter shared with the test, incremented once per destroyed probe.
    Arc<AtomicUsize>,
);

impl minijinja::value::Object for Probe {}

impl Drop for Probe {
    /// Counts the drop.
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// Renders `source` with a `probe()` global and returns the result and the probes dropped since.
///
/// Every probe the template creates must be dropped by the time the render
/// returns; `explode()` panics inside the render.
fn render_with_probes(source: &str) -> (Result<String, Error>, usize) {
    let dropped = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&dropped);
    let result = render_template_with(
        source,
        &json!({}),
        Some(0),
        RenderLimits::for_context_bytes(0),
        move |environment| {
            environment.add_function("probe", move || {
                Value::from_object(Probe(Arc::clone(&counter)))
            });
            environment.add_function("explode", || -> Result<Value, Error> {
                panic!("unwinding through the render");
            });
        },
    );
    let dropped = dropped.load(Ordering::SeqCst);
    (result, dropped)
}

/// Reference cycles through namespaces are broken when the render ends.
///
/// Without the registry, `{% set ns.me = ns %}` (or two namespaces holding
/// each other) kept the namespace, and everything it reaches, alive forever:
/// each render of such a template leaked its values.
#[test]
fn namespace_cycles_do_not_outlive_the_render() {
    let cases = [
        (
            "{% set ns = namespace(p=probe()) %}{% set ns.me = ns %}ok",
            true,
        ),
        (
            "{% set a = namespace(p=probe()) %}{% set b = namespace(a=a) %}{% set a.b = b %}ok",
            true,
        ),
        // An error, and a guard refusal before rendering, also release them.
        (
            "{% set ns = namespace(p=probe()) %}{% set ns.me = ns %}{{ 1 // 0 }}",
            false,
        ),
        (
            "{% set ns = namespace(p=probe()) %}{% set ns.l = [ns] %}{{ explode() }}",
            false,
        ),
    ];
    for (source, succeeds) in cases {
        let (result, dropped) = render_with_probes(source);
        assert_eq!(result.is_ok(), succeeds, "{source}: {result:?}");
        assert_eq!(dropped, 1, "{source}: the probe leaked");
    }
}

/// Cleanup work is proportional to the attributes it clears, not to namespaces times names.
///
/// Review finding CHAT-CLEANUP-AMPLIFICATION: the cleanup assigned the union
/// of every attribute name to every namespace, inserting new entries as it
/// went, without fuel. One namespace with 1 000 names plus 20 000 empty
/// namespaces took 2.97 s and 2.3 GB inside the destructor. The plan now
/// groups namespaces by the names they hold and clears only values that can
/// hold a reference.
#[test]
fn namespace_cleanup_costs_only_what_the_render_built() {
    let registry = Arc::new(NamespaceRegistry::default());
    let budget = Arc::new(RenderBudget::new(RenderLimits::for_context_bytes(0)));
    let mut environment = Environment::new();
    install_namespace_registry(&mut environment, &registry, &budget);
    let mut source = String::from("{% set ns = namespace() %}");
    for index in 0..1_000 {
        source.push_str(&format!("{{% set ns.a{index} = {index} %}}"));
    }
    source.push_str("{% set ns.me = ns %}{% set ns.all = [ns, 'x'] %}");
    source.push_str("{% for i in range(20000) %}{% set x = namespace(i=i) %}{% endfor %}");
    environment.render_str(&source, ()).unwrap();
    let plan = registry.cleanup_plan();
    assert_eq!(plan.assignments(), 2);
    assert_eq!(plan.names().collect::<Vec<_>>(), ["all", "me"]);
    assert_eq!(
        plan.source(),
        "{% for n in groups[0] %}{% set n.all = none %}{% endfor %}\
         {% for n in groups[1] %}{% set n.me = none %}{% endfor %}"
    );
    // Two assignments, two loops, and the template itself.
    assert_eq!(plan.fuel(), 5 * CleanupPlan::FUEL_PER_STEP);
    let started = std::time::Instant::now();
    registry.break_cycles();
    assert!(started.elapsed().as_secs_f64() < 1.0);
    assert_eq!(registry.cleanup_plan().assignments(), 0);
}

/// Every reference-holding attribute is cleared, whatever its name and however many share it.
#[test]
fn namespace_cleanup_clears_every_reference_holder() {
    let names = [
        "none", "true", "false", "if", "else", "not", "and", "or", "in", "is", "loop", "for",
        "set", "self", "_x9",
    ];
    for name in names {
        let source = format!(
            "{{% set ns = namespace(p=probe()) %}}{{% set ns.{name} = ns %}}\
             {{% for i in range(3) %}}{{% set other = namespace(o=ns) %}}{{% set other.{name} = [other] %}}{{% endfor %}}ok"
        );
        let (result, dropped) = render_with_probes(&source);
        assert_eq!(result.unwrap(), "ok", "{name}");
        assert_eq!(dropped, 1, "{name}: the probe leaked");
    }
}

/// `namespace()` charges the namespace and every attribute it copies, which the registry keeps alive.
#[test]
fn namespaces_are_charged_to_the_budget() {
    // `dict()` copies seven entries (336 bytes); each namespace copies them
    // again as eight items (192 bytes) plus seven bytes of keys.
    assert_limited(
        "{% set d = dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7) %}{% for i in range(6) %}{% set x = namespace(d) %}{% endfor %}",
        SMALL,
    );
    assert_eq!(
        render_with(
            "{% set d = dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7) %}{% for i in range(2) %}{% set x = namespace(d) %}{% endfor %}{{ namespace(d).g }}",
            SMALL,
        )
        .unwrap(),
        "7"
    );
    // More attributes than one sequence may hold, and key text past the value limit.
    assert_limited(
        "{% set x = namespace(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9) %}",
        SMALL,
    );
    assert_limited("{% set x = namespace({'k' * 65: 1}) %}", SMALL);
    // A non-map default keeps MiniJinja's own error.
    let error = render_with("{% set x = namespace(3) %}", SMALL).unwrap_err();
    assert!(error.to_string().contains("expected object"), "{error}");
}

/// The registry clears only identifier-named attributes, and an empty registry is a no-op.
#[test]
fn namespace_registry_clears_identifier_attributes() {
    // An empty registry has nothing to clear.
    NamespaceRegistry::default().break_cycles();
    let mut environment = Environment::new();
    install_namespace_registry(
        &mut environment,
        &Arc::new(NamespaceRegistry::default()),
        &Arc::new(RenderBudget::new(RenderLimits::for_context_bytes(0))),
    );
    assert_eq!(
        environment
            .render_str("{% set ns = namespace({'a b': 1}, ) %}{{ ns['a b'] }}", ())
            .unwrap(),
        "1"
    );
    assert!(is_identifier("_a1") && is_identifier("loop"));
    assert!(
        !is_identifier("") && !is_identifier("1a") && !is_identifier("a b") && !is_identifier("é")
    );
}

/// Asserts that rendering `source` under generous limits fails with an error whose detail contains `detail`.
fn assert_render_error(source: &str, detail: &str) {
    let error = render(source).expect_err(source);
    assert!(format!("{error:#}").contains(detail), "{source}: {error:#}");
}

/// Repetition is commutative over a sequence and refuses counts whose size overflows.
#[test]
fn repetition_accepts_either_operand_order_and_refuses_overflowing_counts() {
    assert_eq!(render("{{ 3 * [1, 2] }}").unwrap(), "[1, 2, 1, 2, 1, 2]");
    assert_eq!(render("{{ 2 * 'ab' }}").unwrap(), "abab");
    // 3 * 9223372036854775807 exceeds usize::MAX (2 * 9223372036854775807 + 1),
    // so the byte and item counts overflow.
    assert_render_error(
        "{{ 'abc' * 9223372036854775807 }}",
        "string repetition `*` of 3 bytes 9223372036854775807 times overflows",
    );
    assert_render_error(
        "{{ [1, 2, 3] * 9223372036854775807 }}",
        "sequence repetition `*` of 3 items 9223372036854775807 times overflows",
    );
}

/// Operand mixes MiniJinja rejects remain its errors rather than resource-limit errors.
#[test]
fn rejected_operand_mixes_are_minijinja_errors() {
    for source in [
        "{{ [1] + 2 }}",
        "{{ [1]|dictsort }}",
        "{{ 'abc' is startingwith(x=1) }}",
        "{{ 'abc'|replace('a', x='b') }}",
    ] {
        let error = render(source).expect_err(source);
        assert!(
            !format!("{error:#}").contains("resource limit"),
            "{source}: {error:#}"
        );
    }
    assert_render_error(
        "{{ 'abc' is startingwith(x=1) }}",
        "cannot convert kwargs to string",
    );
}

/// `chain` merges maps lazily but materializes a map mixed with a sequence.
#[test]
fn chain_materializes_a_map_mixed_with_a_sequence() {
    assert_eq!(render("{{ {'a': 1}|chain([2]) }}").unwrap(), "['a', 2]");
    assert_eq!(
        render("{{ ({'a': 1}|chain({'b': 2}))|list }}").unwrap(),
        "['a', 'b']"
    );
}

/// Namespace attributes that templates cannot assign by name are left to the drop.
///
/// `{% set ns.name = ... %}` accepts identifiers only, so a key such as
/// `"not an id"` (from `namespace(mapping)`) can never close a cycle and the
/// cleanup plan skips it, while the identifier key holding a list is cleared.
#[test]
fn namespace_cleanup_skips_attributes_without_identifier_names() {
    let registry = Arc::new(NamespaceRegistry::default());
    let budget = Arc::new(RenderBudget::new(RenderLimits::for_context_bytes(0)));
    let mut environment = Environment::new();
    install_namespace_registry(&mut environment, &registry, &budget);
    environment
        .render_str(
            "{% set ns = namespace({'not an id': [1], 'ok': [2]}) %}{{ ns['not an id'] }}",
            (),
        )
        .unwrap();
    let plan = registry.cleanup_plan();
    assert_eq!(plan.names().collect::<Vec<_>>(), ["ok"]);
    assert_eq!(plan.assignments(), 1);
}

/// Printing failures name the operation and tell a limit apart from a value's own failure.
#[test]
fn printing_errors_distinguish_limits_from_failing_values() {
    let budget = RenderBudget::new(SMALL);
    let failed = printing_error(&budget, "pprint", false);
    assert_eq!(failed.kind(), ErrorKind::InvalidOperation);
    assert_eq!(failed.detail(), Some("pprint failed to print a value"));
    let limited = printing_error(&budget, "pprint", true);
    assert!(
        limited
            .detail()
            .is_some_and(|detail| detail.contains("pprint would print more than the 64 bytes")),
        "{limited:#}"
    );
    let bare = without_location(&Error::from(ErrorKind::UndefinedError));
    assert_eq!(bare.kind(), ErrorKind::UndefinedError);
    assert_eq!(bare.detail(), None);
}
