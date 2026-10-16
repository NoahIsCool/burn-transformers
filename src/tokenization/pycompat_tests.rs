//! Python-compatibility tests against goldens rendered by Hugging Face's Jinja2 environment.
//!
//! Expected strings were produced with `transformers` 5.2.0
//! (`render_jinja_template`) and CPython 3.11. Inputs use key-sorted maps so
//! the goldens do not depend on Python's insertion order (see the module docs).

use minijinja::{Environment, UndefinedBehavior};
use proptest::prelude::*;
use serde_json::json;

use super::*;
use crate::tokenization::chat_template::sandbox::RenderLimits;

/// A budget with the limits of an empty template context.
fn budget() -> Arc<RenderBudget> {
    Arc::new(RenderBudget::new(RenderLimits::for_context_bytes(0)))
}

/// Renders `source` with only this module's callbacks installed.
fn render(source: &str, context: serde_json::Value) -> Result<String, Error> {
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Lenient);
    install(&mut environment, &budget());
    environment.render_str(source, context)
}

/// Renders a template that must succeed.
fn rendered(source: &str, context: serde_json::Value) -> String {
    render(source, context).unwrap_or_else(|error| panic!("{source}: {error:#}"))
}

/// `{{ value }}` prints like Python's `str()`, including nested reprs.
#[test]
fn printing_matches_python_str() {
    assert_eq!(
        rendered(
            "{{ true }}|{{ false }}|{{ none }}|{{ 1.0 }}|{{ 1e20 }}|{{ 3 }}|{{ [1, 'a', none, true, 2.5, {'k': \"it's\"}] }}|{{ {'a': [1], 'b': 'q\"uote'} }}|{{ value }}|{{ ['tab\\there', 'nl\\nx', 'back\\\\slash', 'both\\'\"'] }}|{{ missing }}",
            json!({"value": {"y": "é", "z": [1.5, null]}}),
        ),
        "True|False|None|1.0|1e+20|3|[1, 'a', None, True, 2.5, {'k': \"it's\"}]|{'a': [1], 'b': 'q\"uote'}|{'y': 'é', 'z': [1.5, None]}|['tab\\there', 'nl\\nx', 'back\\\\slash', 'both\\'\"']|"
    );
}

/// Float formatting matches CPython's `repr` across notation boundaries.
#[test]
fn float_repr_matches_cpython() {
    for (value, expected) in [
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (1.0, "1.0"),
        (0.1, "0.1"),
        (1e16, "1e+16"),
        (1e15, "1000000000000000.0"),
        (123_456_789.123, "123456789.123"),
        (1e-4, "0.0001"),
        (1e-5, "1e-05"),
        (1.5e-7, "1.5e-07"),
        (2.5e300, "2.5e+300"),
        (-3.75, "-3.75"),
        (1.0 / 3.0, "0.3333333333333333"),
        (5e-324, "5e-324"),
        (f64::MAX, "1.7976931348623157e+308"),
        (100.0, "100.0"),
        (1e22, "1e+22"),
    ] {
        assert_eq!(python_float_repr(value), expected, "{value:e}");
    }
}

/// `tojson` reproduces `json.dumps` defaults, indentation, separators, and escaping.
#[test]
fn tojson_matches_json_dumps() {
    let tools = json!([{
        "function": {
            "description": "the user's <city> & \"weather\"",
            "name": "get_weather",
            "parameters": {
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "type": "object"
            }
        },
        "type": "function"
    }]);
    let numbers = json!({
        "a": 1.0, "b": 1e20, "c": 1e-05, "d": 2.5, "e": 0.0001, "f": 1e16,
        "g": 1_234_567_890_123_456.0, "h": -0.0, "i": 7, "j": -3, "k": 0.1
    });
    let template = "{{ tools | tojson }}\n{{ tools[0] | tojson(indent=2) }}\n{{ tools | tojson(separators=(',', ':')) }}\n{{ tools[0].function | tojson(sort_keys=True) }}\n{{ 'é<&>\\'' | tojson }}|{{ 'é' | tojson(ensure_ascii=True) }}|{{ '😀\\n\\t\\u0001' | tojson(ensure_ascii=True) }}\n{{ nums | tojson }}\n{{ [[], {}, [1, [2]], none, true, false] | tojson(indent=0) }}";
    let expected = "[{\"function\": {\"description\": \"the user's <city> & \\\"weather\\\"\", \"name\": \"get_weather\", \"parameters\": {\"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"], \"type\": \"object\"}}, \"type\": \"function\"}]\n{\n  \"function\": {\n    \"description\": \"the user's <city> & \\\"weather\\\"\",\n    \"name\": \"get_weather\",\n    \"parameters\": {\n      \"properties\": {\n        \"city\": {\n          \"type\": \"string\"\n        }\n      },\n      \"required\": [\n        \"city\"\n      ],\n      \"type\": \"object\"\n    }\n  },\n  \"type\": \"function\"\n}\n[{\"function\":{\"description\":\"the user's <city> & \\\"weather\\\"\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"}},\"type\":\"function\"}]\n{\"description\": \"the user's <city> & \\\"weather\\\"\", \"name\": \"get_weather\", \"parameters\": {\"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"], \"type\": \"object\"}}\n\"é<&>'\"|\"\\u00e9\"|\"\\ud83d\\ude00\\n\\t\\u0001\"\n{\"a\": 1.0, \"b\": 1e+20, \"c\": 1e-05, \"d\": 2.5, \"e\": 0.0001, \"f\": 1e+16, \"g\": 1234567890123456.0, \"h\": -0.0, \"i\": 7, \"j\": -3, \"k\": 0.1}\n[\n[],\n{},\n[\n1,\n[\n2\n]\n],\nnull,\ntrue,\nfalse\n]";
    assert_eq!(
        rendered(template, json!({"tools": tools, "nums": numbers})),
        expected
    );
    assert_eq!(
        rendered(
            "{{ v | tojson(indent='\\t') }}|{{ v | tojson(indent=-1) }}|{{ 'x' | tojson(true) }}|{{ v | tojson(separators=[';', '=']) }}|{{ {1: 'a', none: 'c', 2.5: 'd'} | tojson(sort_keys=true) }}",
            json!({"v": {"a": [1, 2], "b": {}}}),
        ),
        "{\n\t\"a\": [\n\t\t1,\n\t\t2\n\t],\n\t\"b\": {}\n}|{\n\"a\": [\n1,\n2\n],\n\"b\": {}\n}|\"x\"|{\"a\"=[1;2];\"b\"={}}|{\"1\": \"a\", \"2.5\": \"d\", \"null\": \"c\"}"
    );
}

/// Invalid `tojson` arguments and unserializable values fail with context.
#[test]
fn tojson_rejects_invalid_arguments() {
    for (template, expected) in [
        ("{{ 1 | tojson(indent=[1]) }}", "indent must be an integer"),
        ("{{ 1 | tojson(indent=5000) }}", "must not exceed 1024"),
        ("{{ 1 | tojson(separators=',') }}", "pair of strings"),
        ("{{ 1 | tojson(separators=[1, 2]) }}", "pair of strings"),
        ("{{ 1 | tojson(bogus=1) }}", "bogus"),
        ("{{ missing | tojson }}", "not JSON serializable"),
        ("{{ {(1, 2): 1} | tojson }}", "keys must be"),
    ] {
        let error = render(template, json!({})).unwrap_err();
        assert!(
            format!("{error:#}").contains(expected),
            "{template}: {error:#}"
        );
    }
}

/// Python `str` methods return the same results as CPython.
#[test]
fn string_methods_match_cpython() {
    assert_eq!(
        rendered(
            "{{ ' a  b\\tc '.split() | join('|') }}|{{ 'a,b,,c'.split(',', 1) | join('|') }}|{{ 'a-b-c'.replace('-', '+') }}|{{ 'a-b-c'.replace('-', '+', 1) }}|{{ 'hello wORLD'.title() }}|{{ 'hello wORLD'.capitalize() }}|{{ 'abcabc'.count('b') }}|{{ 'abc'.find('c') }}|{{ 'abc'.find('z') }}|{{ '12'.isdigit() }}|{{ 'ab'.isalpha() }}|{{ ' '.isspace() }}|{{ 'Ab'.islower() }}|{{ 'a\\nb'.splitlines() | join('|') }}|{{ 'a b c'.rsplit(' ', 1) | join('|') }}|{{ 'x'.join(['1','2']) }}|{{ 'abc'.startswith(('x', 'a')) }}|{{ 'abc'.endswith(('x', 'z')) }}|{{ 'aXb'.lower() }}",
            json!({}),
        ),
        "a|b|c|a|b,,c|a+b+c|a+b-c|Hello World|Hello world|2|2|-1|True|True|True|False|a|b|a b|c|1x2|True|False|axb"
    );
    assert_eq!(
        rendered(
            "{{ '  a  b  c  '.split(none, 1) }}|{{ '  a  b  c  '.rsplit(none, 1) }}|{{ 'a,b,c'.rsplit(',', 1) }}|{{ 'a b'.split(' ', 0) }}|{{ ''.split() }}|{{ ''.split(',') }}|{{ 'a\\r\\nb\\rc\\n'.splitlines() }}|{{ 'x'.split() }}",
            json!({}),
        ),
        "['a', 'b  c  ']|['  a  b', 'c']|['a,b', 'c']|['a b']|[]|['']|['a', 'b', 'c']|['x']"
    );
    assert_eq!(
        rendered(
            "{{ \"they're bill's friends\".title() }}|{{ 'ÉCOLE élève'.capitalize() }}|{{ 'AbC'.swapcase() }}|{{ ''.isdigit() }}|{{ 'ABC'.isupper() }}|{{ '1A'.isupper() }}|{{ 'héllo'.find('l') }}|{{ 'héllo'.rfind('l') }}|{{ 'abc'.count('') }}|{{ 'abc'.replace('', '-') }}|{{ 'abc'.replace('', '-', 2) }}|{{ 'prefix-x'.removeprefix('prefix-') }}|{{ 'x.txt'.removesuffix('.txt') }}|{{ 'xxhixx'.strip('x') }}",
            json!({}),
        ),
        "They'Re Bill'S Friends|École élève|aBc|False|True|True|2|3|4|-a-b-c-|-a-bc|x|x|hi"
    );
    assert_eq!(
        rendered(
            "{{ 'xyhelloyx'.lstrip('xy') }}|{{ 'xyhelloyx'.rstrip('xy') }}|{{ '  x  '.strip() }}|{{ 'hello'.strip('') }}|{{ 'HeLLo'.upper() }}|{{ 'abc'.index('b') }}|{{ 'abcb'.rindex('b') }}|{{ 'x'.strip(none) }}",
            json!({}),
        ),
        "helloyx|xyhello|x|hello|HELLO|1|3|x"
    );
}

/// Python `dict` methods expose keys, values, pairs, and defaults.
#[test]
fn dict_methods_match_cpython() {
    assert_eq!(
        rendered(
            "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}|{{ d.get('a') }}|{{ d.get('zz') }}|{{ d.get('zz', 5) }}|{{ d.keys() | list }}|{{ d.values() | list }}",
            json!({"d": {"a": 1, "b": [2]}}),
        ),
        "a=1;b=[2];|1|None|5|['a', 'b']|[1, [2]]"
    );
}

/// `string`, `join`, `trim`, and `safe` convert values with Python's `str()`.
///
/// Official tool templates print arguments with `args_value | string` and
/// schema fields with `json_dict[key] | string`; MiniJinja's built-ins would
/// print `true`/`none` where Python prints `True`/`None`.
#[test]
fn string_filters_print_python_values() {
    assert_eq!(
        rendered(
            "{{ true|string }}|{{ none|string }}|{{ 1.5|string }}|{{ [true, none, 'a']|string }}|{{ {'k': [1, none]}|string }}|{{ missing|string }}|{{ 'x'|string }}|{{ 3|string }}|{{ 1e16|string }}",
            json!({}),
        ),
        "True|None|1.5|[True, None, 'a']|{'k': [1, None]}||x|3|1e+16"
    );
    assert_eq!(
        rendered(
            "{{ [1, true, none, 1.5, 'a', ['b']]|join(',') }}|{{ items|join('-', attribute='a') }}|{{ items|join(attribute='b.0') }}|{{ missing|join(',') }}|{{ 'ab'|join(1) }}|{{ [1,2]|join }}|{{ items|join(d='+', attribute='a') }}",
            json!({"items": [{"a": 1, "b": [true]}, {"a": true, "b": [null]}]}),
        ),
        "1,True,None,1.5,a,['b']|1-True|TrueNone||a1b|12|1+True"
    );
    assert_eq!(
        rendered(
            "[{{ '  x \u{1c}'|trim }}]|[{{ 'xxhixx'|trim('x') }}]|[{{ true|trim }}]|[{{ none|trim }}]|[{{ missing|trim }}]|{{ true|safe }}|{{ none|safe }}",
            json!({}),
        ),
        "[x]|[hi]|[True]|[None]|[]|True|None"
    );
    // A safe string stays safe through `string`, as Jinja2's `soft_str` keeps `Markup`.
    let mut environment = Environment::new();
    install(&mut environment, &budget());
    let value = environment
        .render_str("{{ ('<b>' | safe | string) is safe }}", ())
        .unwrap();
    assert_eq!(value, "True");
}

/// `round` follows Python: ties to even on the exact binary value, `ceil`/`floor` methods.
#[test]
fn round_matches_python() {
    assert_eq!(
        rendered(
            "{{ 2.5|round }}|{{ 3.5|round }}|{{ -2.5|round }}|{{ 0.125|round(2) }}|{{ 2.675|round(2) }}|{{ 3|round }}|{{ 3|round(1, 'ceil') }}|{{ 2.25|round(1, 'floor') }}|{{ 1234.5|round(-2) }}|{{ 1250.0|round(-2) }}|{{ 1350|round(-2) }}|{{ 25|round(-1) }}|{{ -25|round(-1) }}|{{ 35|round(-1, 'floor') }}|{{ true|round }}|{{ 1e16|round }}|{{ 2.5|round(method='ceil') }}|{{ -0.4|round }}|{{ 0.5|round(precision=0) }}|{{ 7.0|round(-1, 'ceil') }}|{{ 155|round(-1, 'ceil') }}",
            json!({}),
        ),
        "2.0|4.0|-2.0|0.12|2.67|3|3.0|2.2|1200.0|1200.0|1400|20|-20|30.0|1|1e+16|3.0|-0.0|0.0|10.0|160.0"
    );
    // CPython limits: huge `ndigits` keeps the value, huge negative `ndigits` gives a signed zero.
    assert_eq!(
        round_float_half_even(0.1, 400).unwrap().to_bits(),
        0.1_f64.to_bits()
    );
    assert_eq!(
        round_float_half_even(-12.0, -400).unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert!(round_float_half_even(f64::NAN, 0).unwrap().is_nan());
    assert_eq!(
        round_float_half_even(f64::INFINITY, -2).unwrap(),
        f64::INFINITY
    );
    // Golden: CPython `round(sys.float_info.max, -303)`; at -304 and beyond it overflows.
    assert_eq!(round_float_half_even(f64::MAX, -303).unwrap(), 1.797_69e308);
    assert!(round_float_half_even(f64::MAX, -304).is_err());
    assert_eq!(round_integer_half_even(i128::MAX, 39), Some(0));
    assert_eq!(round_integer_half_even(i128::MAX, 1), None);
}

/// Review finding (Verus): rounding `i128::MIN` to tens or hundreds overflowed.
///
/// The lower multiple was computed before the direction was chosen, so a
/// template could reach a debug panic or, in release builds, a wrapped
/// positive result; the nearest multiple is now reached in one checked step.
#[test]
fn rounding_the_smallest_integer_is_exact_or_an_overflow_error() {
    let minimum = "(-170141183460469231731687303715884105727 - 1)";
    assert_eq!(
        rendered(&format!("{{{{ {minimum} | round(-2) }}}}"), json!({})),
        "-170141183460469231731687303715884105700"
    );
    let error = render(&format!("{{{{ {minimum} | round(-1) }}}}"), json!({})).unwrap_err();
    assert!(error.to_string().contains("out of range"), "{error}");
}

/// Invalid filter arguments fail inside the template with Python-style context.
#[test]
fn filters_reject_invalid_arguments() {
    for (template, expected) in [
        ("{{ 'x'|round }}", "requires a number"),
        ("{{ 1.5|round(1, 'up') }}", "method must be"),
        ("{{ 1.5|round(1.5) }}", "precision must be an integer"),
        ("{{ 1.5|round(1, 'ceil', 3) }}", "too many arguments"),
        ("{{ 1.5|round(digits=1) }}", "digits"),
        (
            "{{ 1.7976931348623157e308|round(-308) }}",
            "too large to represent",
        ),
        ("{{ 1e300|round(400, 'ceil') }}", "out of range"),
        ("{{ 1.0|round(-400, 'floor') }}", "out of range"),
        ("{{ 1e300|round(10, 'ceil') }}", "non-finite"),
        ("{{ none|join(',') }}", "not iterable"),
        ("{{ 5|join(',') }}", "cannot join"),
        ("{{ [1]|join(',', 'a', 'b') }}", "too many arguments"),
        ("{{ [1]|join(sep=',') }}", "sep"),
        ("{{ [{'a': {}}]|join(attribute='a.b.c') }}", "undefined"),
        ("{{ 'x'|trim(1) }}", "chars must be str"),
    ] {
        let error = render(template, json!({})).unwrap_err();
        assert!(
            format!("{error:#}").to_lowercase().contains(expected),
            "{template}: {error:#}"
        );
    }
}

/// `title()` resets words only after cased letters, and whitespace is Python's set.
#[test]
fn casing_and_whitespace_follow_python() {
    assert_eq!(
        rendered(
            "{{ '中a'.title() }}|{{ 'hello wORLD'.title() }}|{{ '1st 2nd'.title() }}|{{ 'a\u{1c}b'.split() }}|{{ '\u{1f}x\u{1f}'.strip() }}|{{ '\u{1c}'.isspace() }}|{{ 'a\u{1c}b c'.rsplit(none, 1) }}",
            json!({}),
        ),
        "中A|Hello World|1St 2Nd|['a', 'b']|x|True|['a\\x1cb', 'c']"
    );
}

/// The whitespace class equals CPython 3.11's `str.isspace()` over every scalar value.
#[test]
fn python_whitespace_matches_cpython_exhaustively() {
    /// `[c for c in map(chr, range(0x110000)) if c.isspace()]` in CPython 3.11.
    const CPYTHON: [u32; 29] = [
        0x9, 0xa, 0xb, 0xc, 0xd, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001,
        0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029,
        0x202f, 0x205f, 0x3000,
    ];
    let ours = (0..=0x10_ffff_u32)
        .filter_map(char::from_u32)
        .filter(|character| is_python_whitespace(*character))
        .map(u32::from)
        .collect::<Vec<_>>();
    assert_eq!(ours, CPYTHON);
}

/// Wrong arity, wrong argument types, and unknown methods fail inside the template.
#[test]
fn methods_reject_invalid_calls() {
    for (template, expected) in [
        ("{{ 'x'.startswith() }}", "takes exactly 1"),
        ("{{ 'x'.endswith('a', 'b') }}", "takes exactly 1"),
        ("{{ 'x'.startswith(1) }}", "must be str or a tuple"),
        ("{{ 'x'.startswith([1]) }}", "must be str or a tuple"),
        ("{{ 'x'.strip(1) }}", "must be str or None"),
        ("{{ 'x'.strip('a', 'b') }}", "takes 0 to 1"),
        ("{{ 'x'.lower(1) }}", "takes exactly 0"),
        ("{{ 'x'.split(',', 'a') }}", "must be int"),
        ("{{ 'x'.split('') }}", "empty separator"),
        ("{{ 'x'.replace(1, 'a') }}", "must be str"),
        ("{{ 'x'.index('z') }}", "substring not found"),
        ("{{ ','.join([1]) }}", "every item to be a str"),
        ("{{ ','.join(1) }}", "must be iterable"),
        ("{{ 'x'.splitlines(true) }}", "takes exactly 0"),
        ("{{ {}.get() }}", "takes 1 to 2"),
        ("{{ {}.items(1) }}", "takes exactly 0"),
        ("{{ 'x'.unsupported() }}", "unknown method"),
        ("{{ {}.unsupported() }}", "unknown method"),
        ("{{ [1].count(1) }}", "unknown method"),
    ] {
        let error = render(template, json!({})).unwrap_err();
        assert!(
            format!("{error:#}").contains(expected),
            "{template}: {error:#}"
        );
    }
}

/// Printing and serialization stop at a bounded depth instead of exhausting the stack.
#[test]
fn deep_nesting_is_rejected() {
    let mut deep = json!(1);
    for _ in 0..=MAX_NESTING + 1 {
        deep = json!([deep]);
    }
    for template in ["{{ deep }}", "{{ deep | tojson }}"] {
        let error = render(template, json!({ "deep": deep })).unwrap_err();
        assert!(
            format!("{error:#}").contains("nesting exceeds"),
            "{error:#}"
        );
    }
}

proptest! {
    /// `tojson` output parses back to the same JSON value (default, indented, ASCII-only).
    #[test]
    fn tojson_round_trips_through_a_json_parser(value in json_value()) {
        for template in [
            "{{ v | tojson }}",
            "{{ v | tojson(indent=2) }}",
            "{{ v | tojson(ensure_ascii=true) }}",
        ] {
            let text = rendered(template, json!({ "v": value }));
            let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
            prop_assert_eq!(&parsed, &value);
        }
    }

    /// Integer rounding picks the nearer multiple of `10^places`, ties to the even multiple.
    ///
    /// The reference compares absolute distances to both neighbors instead of
    /// the production remainder comparison.
    #[test]
    fn integer_rounding_matches_nearest_multiple(value in any::<i64>(), places in 1_u32..=18) {
        let value = i128::from(value);
        let unit = 10_i128.pow(places);
        let below = value.div_euclid(unit) * unit;
        let above = below + unit;
        let expected = match (value - below).cmp(&(above - value)) {
            core::cmp::Ordering::Less => below,
            core::cmp::Ordering::Greater => above,
            core::cmp::Ordering::Equal if (below / unit) % 2 == 0 => below,
            core::cmp::Ordering::Equal => above,
        };
        prop_assert_eq!(round_integer_half_even(value, places), Some(expected));
    }

    /// Float rounding to tens agrees with exact rational arithmetic on half-integers.
    ///
    /// `m / 2` is exact in `f64` for `|m| < 2^53`, so the reference rounds the
    /// rational `m / 2` to a multiple of `10^places` with integers alone.
    #[test]
    fn float_rounding_to_tens_matches_rational_reference(
        doubled in -(1_i64 << 52)..(1_i64 << 52),
        places in 1_usize..=12,
    ) {
        #[allow(clippy::cast_precision_loss)] // |doubled| < 2^52 is exact.
        let value = doubled as f64 / 2.0;
        let unit = 2 * 10_i128.pow(u32::try_from(places).unwrap()); // 10^places in halves
        let numerator = i128::from(doubled);
        let below = numerator.div_euclid(unit) * unit;
        let above = below + unit;
        let nearest = match (numerator - below).cmp(&(above - numerator)) {
            core::cmp::Ordering::Less => below,
            core::cmp::Ordering::Greater => above,
            core::cmp::Ordering::Equal if (below / unit) % 2 == 0 => below,
            core::cmp::Ordering::Equal => above,
        };
        #[allow(clippy::cast_precision_loss)] // Multiples below 2^53 are exact.
        let expected = (nearest / 2) as f64;
        let rounded = round_float_half_even(value, -i64::try_from(places).unwrap()).unwrap();
        prop_assert_eq!(rounded, expected);
        prop_assert_eq!(rounded.is_sign_negative(), value.is_sign_negative());
    }

    /// Rounding eighths to hundredths matches ties-to-even on the exact rational.
    #[test]
    fn float_rounding_to_hundredths_matches_rational_reference(eighths in -1_000_000_i64..1_000_000) {
        #[allow(clippy::cast_precision_loss)] // Small integers are exact.
        let value = eighths as f64 / 8.0;
        // value * 100 = 25 * eighths / 2: round that half-integer to the even neighbor.
        let doubled = 25 * eighths;
        let floor = doubled.div_euclid(2);
        let hundredths = if doubled.rem_euclid(2) == 0 || floor % 2 == 0 { floor } else { floor + 1 };
        #[allow(clippy::cast_precision_loss)] // Small integers are exact.
        let expected = hundredths as f64 / 100.0;
        let rounded = round_float_half_even(value, 2).unwrap();
        prop_assert_eq!(rounded.abs().to_bits(), expected.abs().to_bits());
    }

    /// Float repr always parses back to the identical `f64` and stays visibly a float.
    #[test]
    fn float_repr_round_trips(bits in any::<u64>()) {
        let value = f64::from_bits(bits);
        prop_assume!(value.is_finite());
        let text = python_float_repr(value);
        prop_assert_eq!(text.parse::<f64>().unwrap().to_bits(), value.to_bits());
        prop_assert!(text.contains('.') || text.contains('e'));
    }
}

/// Generates JSON values with strings, integers, and nesting (floats are covered separately).
fn json_value() -> impl Strategy<Value = serde_json::Value> {
    let leaf = prop_oneof![
        Just(serde_json::Value::Null),
        any::<bool>().prop_map(serde_json::Value::from),
        any::<i64>().prop_map(serde_json::Value::from),
        "\\PC{0,12}".prop_map(serde_json::Value::from),
        "[\\x00-\\x1f\"\\\\/<>&']{0,6}".prop_map(serde_json::Value::from),
    ];
    leaf.prop_recursive(4, 32, 6, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 0..5).prop_map(serde_json::Value::from),
            proptest::collection::btree_map("[a-z]{1,4}", inner, 0..5)
                .prop_map(|map| serde_json::Value::Object(map.into_iter().collect())),
        ]
    })
}

/// Reprs escape carriage returns and invisible format characters as CPython does.
///
/// Expected text: CPython 3.11 `str([...])`, `json.dumps`, and `str` methods.
#[test]
fn reprs_escape_invisible_characters_and_non_finite_numbers() {
    assert_eq!(
        rendered(
            "{{ [s] }}|{{ [1e308 * 10, -1e308 * 10, (1e308 * 10) - (1e308 * 10)] }}|{{ [1e308 * 10, (1e308 * 10) - (1e308 * 10)] | tojson }}",
            json!({"s": "a\rb|\u{200b}|\u{2028}|\u{2060}|\u{feff}|\u{200f}|\u{202e}|\u{206f}"}),
        ),
        "['a\\rb|\\u200b|\\u2028|\\u2060|\\ufeff|\\u200f|\\u202e|\\u206f']|[inf, -inf, nan]|[Infinity, NaN]"
    );
}

/// `tojson` follows `json.dumps` for boolean indents, explicit `None`s, and boolean keys.
#[test]
fn tojson_accepts_boolean_indents_explicit_nones_and_boolean_keys() {
    assert_eq!(
        rendered(
            "{{ v | tojson(false, true) }}|{{ v | tojson(indent=false) }}|{{ v | tojson(false, none) }}|{{ v | tojson(indent=none, separators=none) }}|{{ {false: 2, true: 1} | tojson }}|{{ s | tojson }}",
            json!({"v": {"a": [1]}, "s": "a\u{8}b\u{c}c"}),
        ),
        "{\n \"a\": [\n  1\n ]\n}|{\n\"a\": [\n1\n]\n}|{\"a\": [1]}|{\"a\": [1]}|{\"false\": 2, \"true\": 1}|\"a\\bb\\fc\""
    );
    // An unserializable member fails the whole value, as `json.dumps` raises.
    let error = render("{{ {'a': missing} | tojson }}", json!({})).unwrap_err();
    assert!(
        format!("{error:#}").contains("is not JSON serializable"),
        "{error:#}"
    );
}

/// `join` and `trim` accept positional attributes and explicit `None` like Jinja2.
///
/// A digit-only attribute part indexes a list; any other part, such as `-1`,
/// is a string key and yields an undefined (empty) item, as in Jinja2's
/// `make_attrgetter`. A non-string attribute indexes the item directly.
#[test]
fn join_and_trim_accept_positional_and_none_arguments() {
    assert_eq!(
        rendered(
            "{{ rows | join(', ', 'name') }}|{{ [1, 2] | join(',', none) }}|{{ pairs | join(',', 1) }}|{{ pairs | join(',', '-1') }}|{{ '  x  ' | trim(none) }}",
            json!({"rows": [{"name": "a"}, {"name": "b"}], "pairs": [[1, 2], [3, 4]]}),
        ),
        "a, b|1,2|2,4|,|x"
    );
}

/// Whitespace-only `rsplit()` is empty, and digits alone are neither lower nor upper case.
#[test]
fn uncased_and_whitespace_only_strings_follow_cpython() {
    assert_eq!(
        rendered(
            "{{ '   '.rsplit() }}|{{ '123'.islower() }}|{{ '123'.isupper() }}",
            json!({}),
        ),
        "[]|False|False"
    );
}

/// Functions keep MiniJinja's spelling; map-like objects print their attributes as a `dict`.
///
/// Neither matches Jinja2, which prints Python object reprs such as
/// `<function range>`; templates do not print these values in practice.
#[test]
fn foreign_values_print_with_minijinja_spelling_within_the_limit() {
    assert_eq!(
        rendered("{{ range }}|{{ [range] }}", json!({})),
        "function|[function]"
    );
    assert_eq!(
        rendered("{% macro m() %}{% endmacro %}{{ [m] }}", json!({})),
        "[{'name': 'm', 'arguments': [], 'caller': False}]"
    );
    let mut out = String::new();
    let error = display_into(&Value::from("abcdef"), &mut out, 3).unwrap_err();
    assert!(
        format!("{error:#}").contains("would exceed the 3 bytes"),
        "{error:#}"
    );
    assert!(out.is_empty());
    display_into(&Value::from("abc"), &mut out, 3).unwrap();
    assert_eq!(out, "abc");
}
