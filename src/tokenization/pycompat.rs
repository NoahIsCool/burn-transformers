//! Python semantics that Hugging Face chat templates rely on, for MiniJinja.
//!
//! Hugging Face renders chat templates with Jinja2 *in Python*, so official
//! templates freely call Python `str` and `dict` methods, print Python values,
//! and serialize with `json.dumps`. MiniJinja implements Jinja syntax but not
//! Python's object model. This module supplies the missing pieces:
//!
//! - [`python_method`]: the `str` and `dict` methods templates call, such as
//!   `content.split()`, `name.startswith(("a", "b"))`, and `props.items()`.
//! - [`python_formatter`]: `{{ value }}` output as Python's `str()` would print
//!   it, so booleans render as `True`/`False`, `none` as `None`, floats in
//!   `repr` form, and containers as `['a', 1]` / `{'k': 'v'}`.
//! - [`tojson`]: the filter Hugging Face installs in place of Jinja's own,
//!   `json.dumps(x, ensure_ascii=False, indent=None, separators=None,
//!   sort_keys=False)`. Unlike MiniJinja's built-in filter it neither
//!   HTML-escapes `<>&'` nor compacts separators, so tool schemas tokenize
//!   exactly as in Python.
//!
//! Map key order follows the order MiniJinja exposes. With the crate's current
//! `serde_json`/`minijinja` features, maps are key-sorted before they reach a
//! template; Python preserves insertion order. Enabling both crates'
//! `preserve_order` features makes the order match without changing this code.
//!
//! Recursive printing and serialization stop at [`MAX_NESTING`] levels:
//! templates are untrusted programs and could otherwise build a structure deep
//! enough to exhaust the native stack.

use core::fmt::Write as _;

use minijinja::{
    Error, ErrorKind, Output, State, Value,
    value::{Kwargs, ValueKind},
};

/// Deepest container nesting printed or serialized before failing.
pub(crate) const MAX_NESTING: usize = 256;

// Shared error constructors.

/// Builds a template error with Python-style context, such as `str.split`.
fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidOperation, message.into())
}

/// Rejects a container nested deeper than [`MAX_NESTING`].
fn check_depth(depth: usize) -> Result<(), Error> {
    if depth > MAX_NESTING {
        Err(invalid(format!(
            "value nesting exceeds {MAX_NESTING} levels"
        )))
    } else {
        Ok(())
    }
}

// Python `str()` / `repr()` printing.

/// Writes `{{ value }}` exactly as Python's `str(value)` would.
///
/// Strings print raw, undefined values print nothing (Jinja2's lenient
/// `Undefined`), and every other primitive or container uses Python `repr`.
pub(crate) fn python_formatter(
    out: &mut Output<'_>,
    _state: &State<'_, '_>,
    value: &Value,
) -> Result<(), Error> {
    match value.kind() {
        ValueKind::Undefined => Ok(()),
        ValueKind::String => out
            .write_str(value.as_str().unwrap_or_default())
            .map_err(Error::from),
        ValueKind::None | ValueKind::Bool | ValueKind::Number | ValueKind::Seq | ValueKind::Map => {
            let mut text = String::new();
            python_repr(value, &mut text, 0)?;
            out.write_str(&text).map_err(Error::from)
        }
        _ => write!(out, "{value}").map_err(Error::from),
    }
}

/// Appends Python's `repr(value)`; nested strings are quoted, as in a list's repr.
pub(crate) fn python_repr(value: &Value, out: &mut String, depth: usize) -> Result<(), Error> {
    check_depth(depth)?;
    match value.kind() {
        ValueKind::Undefined => out.push_str("Undefined"),
        ValueKind::None => out.push_str("None"),
        ValueKind::Bool => out.push_str(if value.is_true() { "True" } else { "False" }),
        ValueKind::Number => out.push_str(&python_number(value, "inf", "nan")?),
        ValueKind::String => python_string_repr(value.as_str().unwrap_or_default(), out),
        ValueKind::Map => {
            out.push('{');
            for (index, key) in value.try_iter()?.enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_repr(&key, out, depth + 1)?;
                out.push_str(": ");
                python_repr(&value.get_item(&key)?, out, depth + 1)?;
            }
            out.push('}');
        }
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            for (index, item) in value.try_iter()?.enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_repr(&item, out, depth + 1)?;
            }
            out.push(']');
        }
        _ => {
            let _ = write!(out, "{value}");
        }
    }
    Ok(())
}

/// Appends Python's `repr` of a string, choosing quotes the way CPython does.
///
/// CPython uses single quotes unless the text contains a single quote and no
/// double quote. Control characters use `\xNN` escapes. Non-ASCII text is kept
/// verbatim except for a few non-printable separators and format characters,
/// which approximates `str.isprintable()` without a Unicode category table.
fn python_string_repr(text: &str, out: &mut String) {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ if character == quote => {
                out.push('\\');
                out.push(quote);
            }
            '\0'..='\x1f' | '\x7f'..='\u{a0}' | '\u{ad}' => {
                let _ = write!(out, "\\x{:02x}", u32::from(character));
            }
            '\u{200b}'..='\u{200f}'
            | '\u{2028}'..='\u{202e}'
            | '\u{2060}'..='\u{206f}'
            | '\u{feff}' => {
                let _ = write!(out, "\\u{:04x}", u32::from(character));
            }
            _ => out.push(character),
        }
    }
    out.push(quote);
}

/// Formats a number as Python does; non-finite floats use the given spellings.
///
/// `repr` spells them `inf`/`nan`, while `json.dumps` uses `Infinity`/`NaN`.
fn python_number(value: &Value, infinity: &str, nan: &str) -> Result<String, Error> {
    if value.is_integer() {
        return Ok(value.to_string());
    }
    let float = f64::try_from(value.clone())?;
    Ok(if float.is_nan() {
        nan.to_owned()
    } else if float.is_infinite() {
        format!("{}{infinity}", if float < 0.0 { "-" } else { "" })
    } else {
        python_float_repr(float)
    })
}

/// Formats a finite float exactly like CPython's `float.__repr__`.
///
/// Both languages print the shortest digit string that round-trips. Python
/// then chooses positional notation for decimal exponents in `[-4, 16)` and
/// scientific notation (`1e+16`, `1e-05`) otherwise, always keeping a
/// fractional part in positional form (`1.0`).
pub(crate) fn python_float_repr(value: f64) -> String {
    // Rust's `{:e}` yields the shortest round-trip digits, e.g. `-1.25e-5`.
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("`{:e}` always contains an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` exponent is an integer");
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if (-4..16).contains(&exponent) {
        if exponent >= 0 {
            // `exponent + 1` digits precede the decimal point.
            let point = usize::try_from(exponent).expect("non-negative exponent") + 1;
            if digits.len() <= point {
                let zeros = "0".repeat(point - digits.len());
                format!("{sign}{digits}{zeros}.0")
            } else {
                format!("{sign}{}.{}", &digits[..point], &digits[point..])
            }
        } else {
            let zeros = "0".repeat(usize::try_from(-exponent - 1).expect("small exponent"));
            format!("{sign}0.{zeros}{digits}")
        }
    } else {
        let mantissa = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{sign}{mantissa}e{exponent_sign}{:02}",
            exponent.unsigned_abs()
        )
    }
}

// `json.dumps`-compatible `tojson` filter.

/// Formatting controls of Python's `json.dumps`.
struct JsonOptions {
    /// Escape every non-ASCII character as `\uXXXX` (UTF-16 surrogates above the BMP).
    ensure_ascii: bool,
    /// Per-level indentation; `None` keeps the whole value on one line.
    indent: Option<String>,
    /// Separator written between array items and between object members.
    item_separator: String,
    /// Separator written between an object key and its value.
    key_separator: String,
    /// Emit object members ordered by key rather than in map order.
    sort_keys: bool,
}

/// The `tojson` filter Hugging Face installs: `json.dumps` with its defaults.
///
/// Positional arguments follow Hugging Face's Python signature
/// `tojson(x, ensure_ascii=False, indent=None, separators=None,
/// sort_keys=False)`; all four may also be passed by keyword. As in Python,
/// `indent` may be an integer (spaces per level) or a string, and supplying
/// it changes the default item separator from `", "` to `","`.
pub(crate) fn tojson(
    value: &Value,
    ensure_ascii: Option<Value>,
    indent: Option<Value>,
    kwargs: Kwargs,
) -> Result<Value, Error> {
    let ensure_ascii = match ensure_ascii {
        Some(value) => Some(value),
        None => kwargs.get::<Option<Value>>("ensure_ascii")?,
    }
    .is_some_and(|value| value.is_true());
    let indent = match indent {
        Some(value) => Some(value),
        None => kwargs.get::<Option<Value>>("indent")?,
    };
    let indent = match indent {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) => Some(json_indent(&value)?),
    };
    let separators = kwargs.get::<Option<Value>>("separators")?;
    let sort_keys = kwargs
        .get::<Option<Value>>("sort_keys")?
        .is_some_and(|value| value.is_true());
    kwargs.assert_all_used()?;
    let (item_separator, key_separator) = match separators {
        Some(separators) if !separators.is_none() => json_separators(&separators)?,
        _ if indent.is_some() => (",".to_owned(), ": ".to_owned()),
        _ => (", ".to_owned(), ": ".to_owned()),
    };
    let options = JsonOptions {
        ensure_ascii,
        indent,
        item_separator,
        key_separator,
        sort_keys,
    };
    let mut out = String::new();
    write_json(value, &options, 0, &mut out)?;
    // Templates concatenate this text into prompts; it must not be HTML-escaped.
    Ok(Value::from_safe_string(out))
}

/// Interprets Python's `indent`: an integer count of spaces or a literal string.
fn json_indent(value: &Value) -> Result<String, Error> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_owned());
    }
    if value.kind() == ValueKind::Bool {
        // `" " * True` is one space in Python.
        return Ok(if value.is_true() { " " } else { "" }.to_owned());
    }
    let count = value
        .as_i64()
        .ok_or_else(|| invalid("tojson indent must be an integer, string, or none"))?;
    // Python repeats a string a non-positive number of times as "".
    let count = usize::try_from(count.max(0))
        .ok()
        .filter(|count| *count <= 1024)
        .ok_or_else(|| invalid("tojson indent must not exceed 1024 spaces"))?;
    Ok(" ".repeat(count))
}

/// Interprets Python's `separators=(item_separator, key_separator)`.
fn json_separators(value: &Value) -> Result<(String, String), Error> {
    let parts = value
        .try_iter()
        .map_err(|_| invalid("tojson separators must be a pair of strings"))?
        .collect::<Vec<_>>();
    match parts.as_slice() {
        [item, key] => match (item.as_str(), key.as_str()) {
            (Some(item), Some(key)) => Ok((item.to_owned(), key.to_owned())),
            _ => Err(invalid("tojson separators must be a pair of strings")),
        },
        _ => Err(invalid("tojson separators must be a pair of strings")),
    }
}

/// Serializes one value with `json.dumps` layout rules.
fn write_json(
    value: &Value,
    options: &JsonOptions,
    depth: usize,
    out: &mut String,
) -> Result<(), Error> {
    check_depth(depth)?;
    match value.kind() {
        ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => out.push_str(&python_number(value, "Infinity", "NaN")?),
        ValueKind::String => {
            write_json_string(
                value.as_str().unwrap_or_default(),
                options.ensure_ascii,
                out,
            );
        }
        ValueKind::Seq | ValueKind::Iterable => {
            let items = value.try_iter()?.collect::<Vec<_>>();
            write_json_container(out, options, depth, '[', ']', items.len(), |index, out| {
                write_json(&items[index], options, depth + 1, out)
            })?;
        }
        ValueKind::Map => {
            let mut members = value
                .try_iter()?
                .map(|key| Ok((json_key(&key)?, value.get_item(&key)?)))
                .collect::<Result<Vec<_>, Error>>()?;
            if options.sort_keys {
                members.sort_by(|left, right| left.0.cmp(&right.0));
            }
            write_json_container(
                out,
                options,
                depth,
                '{',
                '}',
                members.len(),
                |index, out| {
                    let (key, member) = &members[index];
                    write_json_string(key, options.ensure_ascii, out);
                    out.push_str(&options.key_separator);
                    write_json(member, options, depth + 1, out)
                },
            )?;
        }
        kind => {
            return Err(invalid(format!(
                "Object of type {kind} is not JSON serializable"
            )));
        }
    }
    Ok(())
}

/// Writes `open`, the separated items, and `close`, with optional indentation.
fn write_json_container(
    out: &mut String,
    options: &JsonOptions,
    depth: usize,
    open: char,
    close: char,
    count: usize,
    mut write_item: impl FnMut(usize, &mut String) -> Result<(), Error>,
) -> Result<(), Error> {
    out.push(open);
    if count == 0 {
        out.push(close);
        return Ok(());
    }
    for index in 0..count {
        if index > 0 {
            out.push_str(&options.item_separator);
        }
        if let Some(indent) = &options.indent {
            out.push('\n');
            out.push_str(&indent.repeat(depth + 1));
        }
        write_item(index, out)?;
    }
    if let Some(indent) = &options.indent {
        out.push('\n');
        out.push_str(&indent.repeat(depth));
    }
    out.push(close);
    Ok(())
}

/// Converts a map key the way `json.dumps` does: strings, numbers, booleans, `None`.
fn json_key(key: &Value) -> Result<String, Error> {
    match key.kind() {
        ValueKind::String => Ok(key.as_str().unwrap_or_default().to_owned()),
        ValueKind::Number => python_number(key, "Infinity", "NaN"),
        ValueKind::Bool => Ok(if key.is_true() { "true" } else { "false" }.to_owned()),
        ValueKind::None => Ok("null".to_owned()),
        kind => Err(invalid(format!(
            "keys must be str, int, float, bool or None, not {kind}"
        ))),
    }
}

/// Writes a JSON string literal with Python's escaping rules.
///
/// `json.dumps` escapes quotes, backslashes, and C0 controls (with the short
/// forms `\b \f \n \r \t`). It does not escape `/` or `<>&'`. With
/// `ensure_ascii`, every non-ASCII character becomes lowercase `\uXXXX`.
fn write_json_string(text: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\0'..='\x1f' => {
                let _ = write!(out, "\\u{:04x}", u32::from(character));
            }
            _ if ensure_ascii && !(' '..='~').contains(&character) => {
                let mut units = [0_u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            _ => out.push(character),
        }
    }
    out.push('"');
}

// Jinja2 type tests whose answers depend on Python's object model.

/// Jinja2's `iterable` test: whether Python's `iter(value)` succeeds.
///
/// Strings, sequences, and mappings iterate in Python, and so does Jinja2's
/// lenient `Undefined`; `None`, numbers, and booleans do not. MiniJinja's
/// built-in test also accepts `none`, which sends templates such as
/// `{% if tools is iterable and tools | length > 0 %}` into `none | length`.
pub(crate) fn is_iterable(value: &Value) -> bool {
    matches!(
        value.kind(),
        ValueKind::Undefined
            | ValueKind::String
            | ValueKind::Bytes
            | ValueKind::Seq
            | ValueKind::Map
            | ValueKind::Iterable
    )
}

/// Jinja2's `sequence` test: whether the value has a length and supports indexing.
///
/// In Python this includes strings and dictionaries as well as lists, and
/// Jinja2's `Undefined`; MiniJinja's built-in test accepts only lists.
pub(crate) fn is_sequence(value: &Value) -> bool {
    matches!(
        value.kind(),
        ValueKind::Undefined
            | ValueKind::String
            | ValueKind::Bytes
            | ValueKind::Seq
            | ValueKind::Map
    )
}

// Python `str` and `dict` methods.

/// Implements the Python `str`/`dict` methods used by Hugging Face templates.
///
/// Installed as MiniJinja's unknown-method callback. Methods on other value
/// kinds, and unknown method names, report MiniJinja's `UnknownMethod` error.
pub(crate) fn python_method(
    _state: &State<'_, '_>,
    value: &Value,
    method: &str,
    arguments: &[Value],
) -> Result<Value, Error> {
    match value.kind() {
        ValueKind::String => string_method(value.as_str().unwrap_or_default(), method, arguments),
        ValueKind::Map => map_method(value, method, arguments),
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

/// Typed view of a method call's positional arguments, with Python-style arity errors.
struct Arguments<'a> {
    /// Qualified method name used in errors, such as `str.split`.
    method: String,
    /// Positional arguments supplied by the template.
    values: &'a [Value],
}

impl<'a> Arguments<'a> {
    /// Wraps the arguments of `receiver.method(...)`.
    fn new(receiver: &str, method: &str, values: &'a [Value]) -> Self {
        Self {
            method: format!("{receiver}.{method}"),
            values,
        }
    }

    /// Requires between `minimum` and `maximum` positional arguments.
    fn arity(&self, minimum: usize, maximum: usize) -> Result<(), Error> {
        if (minimum..=maximum).contains(&self.values.len()) {
            Ok(())
        } else if minimum == maximum {
            Err(invalid(format!(
                "{}() takes exactly {minimum} argument(s) ({} given)",
                self.method,
                self.values.len()
            )))
        } else {
            Err(invalid(format!(
                "{}() takes {minimum} to {maximum} arguments ({} given)",
                self.method,
                self.values.len()
            )))
        }
    }

    /// Returns argument `index` as a string, or `None` when omitted or `none`.
    fn optional_str(&self, index: usize) -> Result<Option<&'a str>, Error> {
        match self.values.get(index) {
            None => Ok(None),
            Some(value) if value.is_none() => Ok(None),
            Some(value) => value.as_str().map(Some).ok_or_else(|| {
                invalid(format!(
                    "{}() argument {} must be str or None",
                    self.method,
                    index + 1
                ))
            }),
        }
    }

    /// Returns required string argument `index`.
    fn str(&self, index: usize) -> Result<&'a str, Error> {
        self.values
            .get(index)
            .and_then(Value::as_str)
            .ok_or_else(|| {
                invalid(format!(
                    "{}() argument {} must be str",
                    self.method,
                    index + 1
                ))
            })
    }

    /// Returns an optional integer argument; `-1` and omission mean "no limit".
    fn limit(&self, index: usize) -> Result<Option<usize>, Error> {
        match self.values.get(index) {
            None => Ok(None),
            Some(value) => {
                let count = value.as_i64().ok_or_else(|| {
                    invalid(format!(
                        "{}() argument {} must be int",
                        self.method,
                        index + 1
                    ))
                })?;
                Ok(usize::try_from(count).ok())
            }
        }
    }
}

/// Dispatches one Python `str` method.
fn string_method(text: &str, method: &str, values: &[Value]) -> Result<Value, Error> {
    let arguments = Arguments::new("str", method, values);
    let strings =
        |parts: Vec<&str>| Value::from(parts.into_iter().map(Value::from).collect::<Vec<_>>());
    match method {
        "strip" | "lstrip" | "rstrip" => {
            arguments.arity(0, 1)?;
            let characters = arguments.optional_str(0)?;
            Ok(Value::from(strip(text, method, characters)))
        }
        "split" | "rsplit" => {
            arguments.arity(0, 2)?;
            let separator = arguments.optional_str(0)?;
            let limit = arguments.limit(1)?;
            let parts = split(text, separator, limit, method == "rsplit")?;
            Ok(strings(parts))
        }
        "splitlines" => {
            arguments.arity(0, 0)?;
            Ok(strings(splitlines(text)))
        }
        "startswith" | "endswith" => {
            arguments.arity(1, 1)?;
            let candidates = affixes(&values[0], &arguments.method)?;
            let found = candidates.iter().any(|affix| {
                if method == "startswith" {
                    text.starts_with(affix.as_str())
                } else {
                    text.ends_with(affix.as_str())
                }
            });
            Ok(Value::from(found))
        }
        "replace" => {
            arguments.arity(2, 3)?;
            let (old, new) = (arguments.str(0)?, arguments.str(1)?);
            Ok(Value::from(match arguments.limit(2)? {
                Some(count) => text.replacen(old, new, count),
                None => text.replace(old, new),
            }))
        }
        "removeprefix" | "removesuffix" => {
            arguments.arity(1, 1)?;
            let affix = arguments.str(0)?;
            let stripped = if method == "removeprefix" {
                text.strip_prefix(affix)
            } else {
                text.strip_suffix(affix)
            };
            Ok(Value::from(stripped.unwrap_or(text)))
        }
        "lower" | "upper" | "title" | "capitalize" | "swapcase" => {
            arguments.arity(0, 0)?;
            Ok(Value::from(change_case(text, method)))
        }
        "count" => {
            arguments.arity(1, 1)?;
            let needle = arguments.str(0)?;
            let count = if needle.is_empty() {
                text.chars().count() + 1
            } else {
                text.matches(needle).count()
            };
            Ok(Value::from(count))
        }
        "find" | "rfind" | "index" | "rindex" => {
            arguments.arity(1, 1)?;
            let needle = arguments.str(0)?;
            let byte_index = if method.starts_with('r') {
                text.rfind(needle)
            } else {
                text.find(needle)
            };
            // Python indexes code points, not UTF-8 bytes.
            match byte_index {
                Some(index) => Ok(Value::from(text[..index].chars().count())),
                None if method.ends_with("find") => Ok(Value::from(-1)),
                None => Err(invalid(format!(
                    "{}(): substring not found",
                    arguments.method
                ))),
            }
        }
        "join" => {
            arguments.arity(1, 1)?;
            let parts = values[0]
                .try_iter()
                .map_err(|_| invalid("str.join() argument must be iterable"))?
                .map(|item| {
                    item.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| invalid("str.join() expects every item to be a str"))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Value::from(parts.join(text)))
        }
        "isdigit" | "isdecimal" | "isnumeric" | "isalpha" | "isalnum" | "isspace" | "islower"
        | "isupper" => {
            arguments.arity(0, 0)?;
            Ok(Value::from(classify(text, method)))
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

/// Implements `strip`, `lstrip`, and `rstrip` with optional character sets.
fn strip<'a>(text: &'a str, method: &str, characters: Option<&str>) -> &'a str {
    let in_set = |character: char| match characters {
        Some(set) => set.contains(character),
        None => character.is_whitespace(),
    };
    match method {
        "lstrip" => text.trim_start_matches(in_set),
        "rstrip" => text.trim_end_matches(in_set),
        _ => text.trim_matches(in_set),
    }
}

/// Implements `split`/`rsplit`, including Python's whitespace-run mode.
///
/// With no separator, runs of whitespace separate fields and empty fields are
/// discarded. `limit` caps the number of splits; the remainder stays intact.
fn split<'a>(
    text: &'a str,
    separator: Option<&str>,
    limit: Option<usize>,
    from_right: bool,
) -> Result<Vec<&'a str>, Error> {
    match separator {
        Some("") => Err(invalid("str.split(): empty separator")),
        Some(separator) => Ok(match (limit, from_right) {
            (None, _) => text.split(separator).collect(),
            (Some(limit), false) => text.splitn(limit + 1, separator).collect(),
            (Some(limit), true) => {
                let mut parts = text.rsplitn(limit + 1, separator).collect::<Vec<_>>();
                parts.reverse();
                parts
            }
        }),
        None if from_right => {
            let mut parts = Vec::new();
            let mut rest = text.trim_end();
            while !rest.is_empty() {
                if limit.is_some_and(|limit| parts.len() == limit) {
                    parts.push(rest);
                    break;
                }
                let start = rest
                    .char_indices()
                    .rev()
                    .find(|(_, character)| character.is_whitespace())
                    .map_or(0, |(index, character)| index + character.len_utf8());
                parts.push(&rest[start..]);
                rest = rest[..start].trim_end();
            }
            parts.reverse();
            Ok(parts)
        }
        None => {
            let mut parts = Vec::new();
            let mut rest = text.trim_start();
            while !rest.is_empty() {
                if limit.is_some_and(|limit| parts.len() == limit) {
                    parts.push(rest);
                    break;
                }
                let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
                parts.push(&rest[..end]);
                rest = rest[end..].trim_start();
            }
            Ok(parts)
        }
    }
}

/// Implements `splitlines()` over Python's universal line boundaries.
fn splitlines(text: &str) -> Vec<&str> {
    let is_boundary = |character: char| {
        matches!(
            character,
            '\n' | '\r'
                | '\u{b}'
                | '\u{c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        )
    };
    let mut lines = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let Some(end) = rest.find(is_boundary) else {
            lines.push(rest);
            break;
        };
        lines.push(&rest[..end]);
        // `\r\n` is one boundary.
        let width = if rest[end..].starts_with("\r\n") {
            2
        } else {
            rest[end..].chars().next().map_or(1, char::len_utf8)
        };
        rest = &rest[end + width..];
    }
    lines
}

/// Accepts a string or a tuple/list of strings for `startswith`/`endswith`.
fn affixes(value: &Value, method: &str) -> Result<Vec<String>, Error> {
    if let Some(text) = value.as_str() {
        return Ok(vec![text.to_owned()]);
    }
    let error = || invalid(format!("{method}() argument must be str or a tuple of str"));
    if !matches!(value.kind(), ValueKind::Seq) {
        return Err(error());
    }
    value
        .try_iter()?
        .map(|item| item.as_str().map(str::to_owned).ok_or_else(error))
        .collect()
}

/// Implements Python's case-changing methods.
fn change_case(text: &str, method: &str) -> String {
    match method {
        "lower" => text.to_lowercase(),
        "upper" => text.to_uppercase(),
        "swapcase" => text
            .chars()
            .flat_map(|character| {
                if character.is_lowercase() {
                    character.to_uppercase().collect::<Vec<_>>()
                } else {
                    character.to_lowercase().collect::<Vec<_>>()
                }
            })
            .collect(),
        "capitalize" => {
            let mut characters = text.chars();
            characters.next().map_or_else(String::new, |first| {
                first
                    .to_uppercase()
                    .chain(characters.as_str().to_lowercase().chars())
                    .collect()
            })
        }
        // `title`: a letter after a cased letter is lowercased, otherwise uppercased.
        _ => {
            let mut previous_cased = false;
            let mut titled = String::with_capacity(text.len());
            for character in text.chars() {
                if previous_cased {
                    titled.extend(character.to_lowercase());
                } else {
                    titled.extend(character.to_uppercase());
                }
                previous_cased = character.is_alphabetic();
            }
            titled
        }
    }
}

/// Implements Python's `str.is*` predicates; every predicate is false for "".
fn classify(text: &str, method: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let has_cased = text
        .chars()
        .any(|character| character.is_lowercase() || character.is_uppercase());
    match method {
        "isdigit" | "isdecimal" => text.chars().all(|character| character.is_ascii_digit()),
        "isnumeric" => text.chars().all(char::is_numeric),
        "isalpha" => text.chars().all(char::is_alphabetic),
        "isalnum" => text.chars().all(char::is_alphanumeric),
        "isspace" => text.chars().all(char::is_whitespace),
        "islower" => has_cased && !text.chars().any(char::is_uppercase),
        _ => has_cased && !text.chars().any(char::is_lowercase),
    }
}

/// Dispatches one Python `dict` method: `items`, `keys`, `values`, or `get`.
fn map_method(map: &Value, method: &str, values: &[Value]) -> Result<Value, Error> {
    let arguments = Arguments::new("dict", method, values);
    let keys = || map.try_iter().map(Iterator::collect::<Vec<_>>);
    match method {
        "items" => {
            arguments.arity(0, 0)?;
            let pairs = keys()?
                .into_iter()
                .map(|key| Ok(Value::from(vec![key.clone(), map.get_item(&key)?])))
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Value::from(pairs))
        }
        "keys" => {
            arguments.arity(0, 0)?;
            Ok(Value::from(keys()?))
        }
        "values" => {
            arguments.arity(0, 0)?;
            let values = keys()?
                .iter()
                .map(|key| map.get_item(key))
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Value::from(values))
        }
        "get" => {
            arguments.arity(1, 2)?;
            let found = map.get_item(&values[0])?;
            Ok(if found.is_undefined() {
                values.get(1).cloned().unwrap_or(Value::from(()))
            } else {
                found
            })
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

#[cfg(test)]
#[path = "pycompat_tests.rs"]
mod tests;
