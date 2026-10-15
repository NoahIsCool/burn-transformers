//! Python-compatibility tests against goldens rendered by Hugging Face's Jinja2 environment.
//!
//! Expected strings were produced with `transformers` 5.2.0
//! (`render_jinja_template`) and CPython 3.11. Inputs use key-sorted maps so
//! the goldens do not depend on Python's insertion order (see the module docs).

use minijinja::{Environment, UndefinedBehavior};
use proptest::prelude::*;
use serde_json::json;

use super::*;

/// Renders `source` with only this module's callbacks installed.
fn render(source: &str, context: serde_json::Value) -> Result<String, Error> {
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Lenient);
    environment.set_formatter(python_formatter);
    environment.set_unknown_method_callback(python_method);
    environment.add_filter("tojson", tojson);
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
