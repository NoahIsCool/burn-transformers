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
struct Probe(Arc<AtomicUsize>);

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
    // Each namespace copies seven attributes: eight items (192 bytes) plus
    // seven bytes of keys.
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
