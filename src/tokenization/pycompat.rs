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
//!   it ([`python_str`]), so booleans render as `True`/`False`, `none` as
//!   `None`, floats in `repr` form, and containers as `['a', 1]` / `{'k': 'v'}`.
//! - [`string`], [`join`], [`safe`], and [`trim`]: Jinja2's filters that turn
//!   values into text, with Python's `str()` instead of MiniJinja's spelling,
//!   so `true | string` is `True` as official tool templates expect.
//! - [`round`]: Jinja2's `round(value, precision, method)` with Python's
//!   ties-to-even rounding and the `ceil`/`floor` methods.
//! - [`tojson`]: the filter Hugging Face installs in place of Jinja's own,
//!   `json.dumps(x, ensure_ascii=False, indent=None, separators=None,
//!   sort_keys=False)`. Unlike MiniJinja's built-in filter it neither
//!   HTML-escapes `<>&'` nor compacts separators, so tool schemas tokenize
//!   exactly as in Python.
//! - [`is_iterable`] and [`is_sequence`]: Jinja2's type tests with Python's
//!   answers for `None`, strings, and dictionaries.
//!
//! [`install`] registers all of them.
//!
//! # Remaining differences from Python
//!
//! - **Map key order.** Maps reach templates key-sorted: with the crate's
//!   current `serde_json` and `minijinja` features, JSON objects are sorted
//!   maps, while Python preserves insertion order. Iteration, printing, and
//!   `tojson` therefore list keys alphabetically. Enabling both crates'
//!   `preserve_order` features makes the order match without changing this
//!   code.
//! - **`~` concatenation** converts operands with MiniJinja's spelling
//!   (`'x' ~ true` is `xtrue`, Python gives `xTrue`); MiniJinja exposes no
//!   hook for the operator. Official templates convert non-strings with
//!   `| string` before concatenating, which is exact.
//! - **`%` string formatting** (`'%s' % x`) is not supported and fails the render.
//! - **Unicode classes are approximated**: `isdigit`/`isdecimal` accept ASCII
//!   digits only, `title`/`capitalize` uppercase where Python title-cases
//!   (they differ for digraphs such as `ǆ`), and `repr` escapes a fixed set of
//!   non-printable characters instead of consulting Unicode categories.
//! - Other text filters (`upper`, `replace`, `indent`, ...) convert non-string
//!   input with MiniJinja's spelling.
//!
//! Recursive printing and serialization stop at [`MAX_NESTING`] levels:
//! templates are untrusted programs and could otherwise build a structure deep
//! enough to exhaust the native stack. Every text and list this module
//! produces is also bounded by the render's [`RenderBudget`] while it is
//! built: printing stops at the first item past the allowance, and the
//! methods that copy (`str.replace`, `str.join`, `strip`, `split`,
//! `splitlines`, `removeprefix`, case changes, `dict.items`, `keys`, and
//! `values`) charge their result size before allocating it.

use core::fmt::Write as _;
use std::sync::Arc;

use minijinja::{
    Environment, Error, ErrorKind, Output, State, Value,
    value::{Kwargs, ValueKind},
};

use super::chat_template::sandbox::{self, LimitedText, RenderBudget};

// Documents itself with `//!`; an outer `///` here would change its link scope.
mod decimal; // Integer rounding and decimal increments for `round` (Verus-verified).

use decimal::{round_digits_to_tens, round_integer_half_even};

/// Deepest container nesting printed or serialized before failing.
pub(crate) const MAX_NESTING: usize = 256;

/// Installs every Python-compatibility callback, filter, and test in `environment`.
///
/// This is the single list of MiniJinja behaviors this module replaces: value
/// printing, `str`/`dict` methods, the `tojson`, `string`, `safe`, `trim`,
/// `join`, and `round` filters, and the `iterable` and `sequence` tests.
/// Every callback that builds text charges `budget`.
pub(crate) fn install(environment: &mut Environment<'_>, budget: &Arc<RenderBudget>) {
    let shared = Arc::clone(budget);
    environment
        .set_formatter(move |out, state, value| python_formatter(out, state, value, &shared));
    let shared = Arc::clone(budget);
    environment.set_unknown_method_callback(move |state, value, method, arguments| {
        python_method(state, value, method, arguments, &shared)
    });
    let shared = Arc::clone(budget);
    environment.add_filter(
        "tojson",
        move |value: &Value, ensure_ascii: Option<Value>, indent: Option<Value>, kwargs: Kwargs| {
            tojson(value, ensure_ascii, indent, kwargs, &shared)
        },
    );
    let shared = Arc::clone(budget);
    environment.add_filter("string", move |value: &Value| string(value, &shared));
    let shared = Arc::clone(budget);
    environment.add_filter("safe", move |value: &Value| safe(value, &shared));
    let shared = Arc::clone(budget);
    environment.add_filter("trim", move |value: &Value, characters: Option<Value>| {
        trim(value, characters, &shared)
    });
    let shared = Arc::clone(budget);
    environment.add_filter(
        "join",
        move |value: &Value, separator: Option<Value>, attribute: Option<Value>, kwargs: Kwargs| {
            join(value, separator, attribute, kwargs, &shared)
        },
    );
    environment.add_filter("round", round);
    environment.add_test("iterable", is_iterable);
    environment.add_test("sequence", is_sequence);
}

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

/// Writes `{{ value }}` exactly as Python's `str(value)` would; see [`python_str`].
///
/// Every print, including prints into `{% set %}` blocks, macro results, and
/// `caller()`, is charged to `budget`. Strings are written without a copy.
pub(crate) fn python_formatter(
    out: &mut Output<'_>,
    _state: &State<'_, '_>,
    value: &Value,
    budget: &RenderBudget,
) -> Result<(), Error> {
    if value.kind() == ValueKind::String {
        let text = value.as_str().unwrap_or_default();
        budget.reserve_text(text.len(), "printing a value")?;
        return out.write_str(text).map_err(Error::from);
    }
    out.write_str(&python_str(value, budget, "printing a value")?)
        .map_err(Error::from)
}

/// Converts a value to text exactly as Python's `str(value)` would, within `budget`.
///
/// Strings are returned raw, undefined values are empty (Jinja2's lenient
/// `Undefined.__str__`), and every other primitive or container uses Python
/// `repr`: `True`, `None`, `1.0`, `['a', 1]`, `{'k': None}`. Values with no
/// Python counterpart, such as macros, keep MiniJinja's spelling. Printing
/// stops with a resource-limit error as soon as the text would exceed the
/// render's allowance; the result is charged to `budget` as `operation`.
pub(crate) fn python_str(
    value: &Value,
    budget: &RenderBudget,
    operation: &str,
) -> Result<String, Error> {
    let limit = budget.text_allowance();
    let text = match value.kind() {
        ValueKind::Undefined => String::new(),
        ValueKind::String => value.as_str().unwrap_or_default().to_owned(),
        ValueKind::None | ValueKind::Bool | ValueKind::Number | ValueKind::Seq | ValueKind::Map => {
            let mut text = String::new();
            python_repr(value, &mut text, 0, limit)?;
            text
        }
        _ => {
            let mut text = String::new();
            display_into(value, &mut text, limit)?;
            text
        }
    };
    budget.reserve_text(text.len(), operation)?;
    Ok(text)
}

/// Fails once `out` holds more than `limit` bytes.
///
/// Printing checks after every leaf and container item, so a value whose
/// text is enormous (a container that holds one value many times) costs at
/// most the limit plus one leaf before it is refused.
fn check_printed(out: &str, limit: usize) -> Result<(), Error> {
    if out.len() > limit {
        Err(invalid(format!(
            "chat template resource limit: printing a value would exceed the {limit} bytes this render may still produce"
        )))
    } else {
        Ok(())
    }
}

/// Appends MiniJinja's `Display` text of `value` to `out` without passing `limit` bytes.
fn display_into(value: &Value, out: &mut String, limit: usize) -> Result<(), Error> {
    let mut text = LimitedText::new(limit.saturating_sub(out.len()));
    if write!(text, "{value}").is_err() {
        return check_printed(out, limit).and_then(|()| {
            Err(invalid(format!(
                "chat template resource limit: printing a value would exceed the {limit} bytes this render may still produce"
            )))
        });
    }
    out.push_str(&text.into_string());
    Ok(())
}

/// Appends Python's `repr(value)`; nested strings are quoted, as in a list's repr.
///
/// Fails with a resource-limit error once `out` would pass `limit` bytes.
pub(crate) fn python_repr(
    value: &Value,
    out: &mut String,
    depth: usize,
    limit: usize,
) -> Result<(), Error> {
    check_depth(depth)?;
    check_printed(out, limit)?;
    match value.kind() {
        ValueKind::Undefined => out.push_str("Undefined"),
        ValueKind::None => out.push_str("None"),
        ValueKind::Bool => out.push_str(if value.is_true() { "True" } else { "False" }),
        ValueKind::Number => out.push_str(&python_number(value, "inf", "nan")?),
        ValueKind::String => {
            let text = value.as_str().unwrap_or_default();
            // A repr is at least as long as its text plus two quotes.
            check_printed(out, limit.saturating_sub(text.len().saturating_add(2)))?;
            python_string_repr(text, out);
        }
        ValueKind::Map => {
            out.push('{');
            for (index, key) in value.try_iter()?.enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_repr(&key, out, depth + 1, limit)?;
                out.push_str(": ");
                python_repr(&value.get_item(&key)?, out, depth + 1, limit)?;
            }
            out.push('}');
        }
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            for (index, item) in value.try_iter()?.enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_repr(&item, out, depth + 1, limit)?;
            }
            out.push(']');
        }
        _ => display_into(value, out, limit)?,
    }
    check_printed(out, limit)
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
    /// Most bytes the serialization may reach (the render's text allowance).
    limit: usize,
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
    budget: &RenderBudget,
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
        limit: budget.text_allowance(),
    };
    let mut out = String::new();
    write_json(value, &options, 0, &mut out)?;
    budget.reserve_text(out.len(), "tojson")?;
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
    check_printed(out, options.limit)?;
    match value.kind() {
        ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => out.push_str(&python_number(value, "Infinity", "NaN")?),
        ValueKind::String => {
            let text = value.as_str().unwrap_or_default();
            // JSON text is at least as long as the string plus two quotes.
            check_printed(
                out,
                options.limit.saturating_sub(text.len().saturating_add(2)),
            )?;
            write_json_string(text, options.ensure_ascii, out);
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
    check_printed(out, options.limit)
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

// Jinja2 filters that convert values with Python's `str()`.
//
// MiniJinja's built-in versions of these filters convert non-strings with
// Rust's `Display`, which spells booleans and null `true`/`none`. Official
// tool-calling templates print tool arguments and schema fields through
// `| string`, so those spellings would change prompt text (`True` vs `true`).

/// Jinja2's `string` filter: `soft_str(value)`.
///
/// Strings, including safe strings, pass through unchanged; every other value
/// becomes [`python_str`] text, so `true | string` is `True`.
pub(crate) fn string(value: &Value, budget: &RenderBudget) -> Result<Value, Error> {
    if value.kind() == ValueKind::String {
        return Ok(value.clone());
    }
    python_str(value, budget, "string").map(Value::from)
}

/// Jinja2's `safe` filter: `Markup(value)`, Python `str()` text marked safe.
pub(crate) fn safe(value: &Value, budget: &RenderBudget) -> Result<Value, Error> {
    python_str(value, budget, "safe").map(Value::from_safe_string)
}

/// Jinja2's `trim(value, chars=None)`: `str(value).strip(chars)`.
///
/// Without `chars`, Python's whitespace set is stripped (see
/// [`is_python_whitespace`]).
pub(crate) fn trim(
    value: &Value,
    characters: Option<Value>,
    budget: &RenderBudget,
) -> Result<Value, Error> {
    let characters = match &characters {
        Some(set) if !set.is_none() => Some(
            set.as_str()
                .ok_or_else(|| invalid("trim() chars must be str or None"))?,
        ),
        _ => None,
    };
    let text = python_str(value, budget, "trim")?;
    Ok(Value::from(strip(&text, "strip", characters)))
}

/// Jinja2's `join(value, d='', attribute=None)` with Python `str()` items.
///
/// Each item (or its `attribute`, a dotted path such as `"function.name"`
/// whose digit parts index lists) is converted with [`python_str`], so
/// `[1, true, none] | join(',')` is `1,True,None`. An undefined value joins
/// to `""`; `none` is rejected, as Python cannot iterate it.
pub(crate) fn join(
    value: &Value,
    separator: Option<Value>,
    attribute: Option<Value>,
    kwargs: Kwargs,
    budget: &RenderBudget,
) -> Result<Value, Error> {
    let separator = match separator {
        Some(separator) => Some(separator),
        None => kwargs.get::<Option<Value>>("d")?,
    };
    let attribute = match attribute {
        Some(attribute) => Some(attribute),
        None => kwargs.get::<Option<Value>>("attribute")?,
    };
    kwargs.assert_all_used()?;
    let separator = match separator {
        Some(separator) => python_str(&separator, budget, "join")?,
        None => String::new(),
    };
    if value.is_undefined() {
        return Ok(Value::from(""));
    }
    if value.is_none() {
        return Err(invalid("join(): 'NoneType' object is not iterable"));
    }
    let items = value.try_iter().map_err(|_| {
        invalid(format!(
            "join(): cannot join value of type {}",
            value.kind()
        ))
    })?;
    let limit = budget.text_allowance();
    let mut joined = String::new();
    for (index, item) in items.enumerate() {
        if index > 0 {
            check_printed(&joined, limit.saturating_sub(separator.len()))?;
            joined.push_str(&separator);
        }
        let item = match &attribute {
            Some(path) if !path.is_none() => attribute_path(&item, path)?,
            _ => item,
        };
        joined.push_str(&python_str(&item, budget, "join")?);
        check_printed(&joined, limit)?;
    }
    budget.reserve_text(joined.len(), "join")?;
    Ok(Value::from(joined))
}

/// Looks up Jinja2's `attribute=` argument: an integer index or a dotted path.
///
/// Path parts made of ASCII digits index by integer, as Jinja2's
/// `make_attrgetter` does; missing keys yield `Undefined`.
fn attribute_path(item: &Value, path: &Value) -> Result<Value, Error> {
    let Some(path) = path.as_str() else {
        return item.get_item(path);
    };
    let mut current = item.clone();
    for part in path.split('.') {
        let key = match part.parse::<i64>() {
            Ok(index) if part.bytes().all(|byte| byte.is_ascii_digit()) => Value::from(index),
            _ => Value::from(part),
        };
        current = current.get_item(&key)?;
    }
    Ok(current)
}

/// Jinja2's `round(value, precision=0, method='common')`.
///
/// `common` is Python's built-in `round`: integers stay integers (rounded to
/// tens for negative `precision`), floats round half to even on their exact
/// binary value, so `2.5 | round` is `2.0` and `2.675 | round(2)` is `2.67`.
/// `ceil` and `floor` follow Jinja2's formula
/// `math.ceil(value * 10**precision) / 10**precision` and always return a
/// float. Booleans count as the integers 0 and 1, as in Python.
pub(crate) fn round(
    value: &Value,
    precision: Option<Value>,
    method: Option<Value>,
    kwargs: Kwargs,
) -> Result<Value, Error> {
    let precision = match precision {
        Some(precision) => Some(precision),
        None => kwargs.get::<Option<Value>>("precision")?,
    };
    let method = match method {
        Some(method) => Some(method),
        None => kwargs.get::<Option<Value>>("method")?,
    };
    kwargs.assert_all_used()?;
    let precision = match precision {
        None => 0,
        Some(precision) => integer_argument(&precision)
            .and_then(|precision| i64::try_from(precision).ok())
            .ok_or_else(|| invalid("round() precision must be an integer"))?,
    };
    let method = match &method {
        None => "common",
        Some(method) => method.as_str().unwrap_or_default(),
    };
    let number = RoundOperand::from_value(value)?;
    match method {
        "common" => number.round_common(precision),
        "ceil" => number.round_directed(precision, f64::ceil),
        "floor" => number.round_directed(precision, f64::floor),
        _ => Err(invalid(
            "round() method must be 'common', 'ceil' or 'floor'",
        )),
    }
}

/// Returns a template integer (or boolean, as 0/1) as `i128`.
fn integer_argument(value: &Value) -> Option<i128> {
    match value.kind() {
        ValueKind::Bool => Some(i128::from(value.is_true())),
        ValueKind::Number if value.is_integer() => i128::try_from(value.clone()).ok(),
        _ => None,
    }
}

/// A number accepted by [`round`], keeping Python's int/float distinction.
enum RoundOperand {
    /// An integer or boolean.
    Integer(i128),
    /// A floating-point number.
    Float(f64),
}

impl RoundOperand {
    /// Classifies a template value, rejecting non-numbers.
    fn from_value(value: &Value) -> Result<Self, Error> {
        if let Some(integer) = integer_argument(value) {
            return Ok(Self::Integer(integer));
        }
        match value.kind() {
            ValueKind::Number => Ok(Self::Float(f64::try_from(value.clone())?)),
            kind => Err(invalid(format!("round() requires a number, not {kind}"))),
        }
    }

    /// Python's `round(value, precision)`.
    fn round_common(self, precision: i64) -> Result<Value, Error> {
        match self {
            Self::Integer(integer) if precision >= 0 => Ok(Value::from(integer)),
            Self::Integer(integer) => {
                let places = u32::try_from(precision.unsigned_abs()).unwrap_or(u32::MAX);
                round_integer_half_even(integer, places)
                    .map(Value::from)
                    .ok_or_else(|| invalid("round(): rounded integer is out of range"))
            }
            Self::Float(float) => round_float_half_even(float, precision).map(Value::from),
        }
    }

    /// Jinja2's `ceil`/`floor` methods: `direction(value * 10**p) / 10**p`.
    ///
    /// The scale is `float(10**p)`, correctly rounded, which is exact for
    /// `|p| <= 22`; Python divides by the exact integer `10**p`, so results
    /// for larger positive `p` may differ in the last bit.
    fn round_directed(self, precision: i64, direction: fn(f64) -> f64) -> Result<Value, Error> {
        let value = match self {
            // `ceil(n * 10**p) / 10**p` is exactly `n` for p >= 0; Python's
            // true division then rounds it to the nearest float, as `as` does.
            Self::Integer(integer) if precision >= 0 => return Ok(Value::from(integer as f64)),
            Self::Integer(integer) => integer as f64,
            Self::Float(float) => float,
        };
        // Correctly rounded `10^precision`; it under- or overflows beyond |precision| ~ 308.
        let scale = format!("1e{precision}").parse::<f64>().unwrap_or(0.0);
        if scale == 0.0 || !scale.is_finite() {
            return Err(invalid(format!(
                "round() precision {precision} is out of range for method ceil/floor"
            )));
        }
        let scaled = direction(value * scale);
        if !scaled.is_finite() {
            return Err(invalid(
                "round(): cannot convert a non-finite value to an integer",
            ));
        }
        Ok(Value::from(scaled / scale))
    }
}

/// CPython's `round(float, ndigits)`: correctly rounded, ties to even.
///
/// CPython rounds the exact binary value through `dtoa`. For `ndigits >= 0`,
/// Rust's fixed-precision formatting performs the same exact, ties-to-even
/// rounding, and parsing the digits back is correctly rounded. Negative
/// `ndigits` rounds to tens via [`round_float_to_tens`]. As in CPython,
/// non-finite values and `ndigits > 323` return the value unchanged, and a
/// result too large for `f64` is an error.
fn round_float_half_even(value: f64, ndigits: i64) -> Result<f64, Error> {
    /// CPython's `NDIGITS_MAX`: beyond this many places every double is unchanged.
    const MAX_FRACTION_DIGITS: i64 = 323;
    if !value.is_finite() || ndigits > MAX_FRACTION_DIGITS {
        return Ok(value);
    }
    let rounded = match usize::try_from(ndigits) {
        Ok(places) => format!("{value:.places$}")
            .parse::<f64>()
            .map_err(|error| invalid(format!("round(): {error}")))?,
        Err(_) => {
            let places = usize::try_from(ndigits.unsigned_abs()).unwrap_or(usize::MAX);
            round_float_to_tens(value, places)
        }
    };
    if rounded.is_finite() {
        Ok(rounded)
    } else {
        Err(invalid("round(): rounded value too large to represent"))
    }
}

/// Rounds a finite float to a multiple of `10^places` (`places >= 1`), ties to even.
///
/// The integer part's exact decimal expansion is split into kept digits and
/// `places` dropped digits. The dropped part, plus any fractional part,
/// decides the direction against half a unit (`5` followed by zeros); an
/// exact tie rounds the kept digits to even. The sign is reapplied at the end,
/// so negative values round symmetrically and may produce `-0.0`, as in Python.
fn round_float_to_tens(value: f64, places: usize) -> f64 {
    // Every finite double is below 10^309, so it rounds to zero at 310+ places.
    if places > 309 {
        return 0.0_f64.copysign(value);
    }
    let magnitude = value.abs();
    let whole = magnitude.trunc();
    // `{:.0}` prints an integral double's exact decimal expansion; the
    // verified digit rounding keeps the quotient by `10^places`, rounded half
    // to even with the fraction breaking ties upward.
    let digits = format!("{whole:.0}");
    let kept = round_digits_to_tens(digits.as_bytes(), places, magnitude != whole);
    let kept = String::from_utf8(kept).expect("ASCII digits stay valid UTF-8");
    // Parsing an exact decimal is correctly rounded; overflow yields infinity.
    let rounded: f64 = format!("{kept}e{places}").parse().unwrap_or(f64::INFINITY);
    rounded.copysign(value)
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
    budget: &RenderBudget,
) -> Result<Value, Error> {
    match value.kind() {
        ValueKind::String => string_method(
            value.as_str().unwrap_or_default(),
            method,
            arguments,
            budget,
        ),
        ValueKind::Map => map_method(value, method, arguments, budget),
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

/// Dispatches one Python `str` method; every method that builds a string or list charges `budget` first.
fn string_method(
    text: &str,
    method: &str,
    values: &[Value],
    budget: &RenderBudget,
) -> Result<Value, Error> {
    let arguments = Arguments::new("str", method, values);
    // Copies are charged before they are made: one value per part plus the parts' text.
    let strings = |parts: Vec<&str>| -> Result<Value, Error> {
        budget.reserve_items(parts.len(), &arguments.method)?;
        let bytes = parts.iter().map(|part| part.len()).sum::<usize>();
        budget.reserve_text(bytes, &arguments.method)?;
        Ok(Value::from(
            parts.into_iter().map(Value::from).collect::<Vec<_>>(),
        ))
    };
    let substring = |part: &str| -> Result<Value, Error> {
        budget.reserve_text(part.len(), &arguments.method)?;
        Ok(Value::from(part))
    };
    match method {
        "strip" | "lstrip" | "rstrip" => {
            arguments.arity(0, 1)?;
            let characters = arguments.optional_str(0)?;
            substring(strip(text, method, characters))
        }
        "split" | "rsplit" => {
            arguments.arity(0, 2)?;
            let separator = arguments.optional_str(0)?;
            let limit = arguments.limit(1)?;
            strings(split(text, separator, limit, method == "rsplit")?)
        }
        "splitlines" => {
            arguments.arity(0, 0)?;
            strings(splitlines(text))
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
            let limit = arguments.limit(2)?;
            budget.reserve_text(
                sandbox::replaced_length(text, old, new, limit),
                "str.replace",
            )?;
            Ok(Value::from(match limit {
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
            substring(stripped.unwrap_or(text))
        }
        "lower" | "upper" | "title" | "capitalize" | "swapcase" => {
            arguments.arity(0, 0)?;
            // Unicode case mapping lengthens text by at most 3x (`ß` -> `SS`).
            budget.reserve_text(text.len().saturating_mul(3), &arguments.method)?;
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
            let separators = text.len().saturating_mul(parts.len().saturating_sub(1));
            let bytes = parts
                .iter()
                .map(String::len)
                .fold(separators, usize::saturating_add);
            budget.reserve_text(bytes, "str.join")?;
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
        None => is_python_whitespace(character),
    };
    match method {
        "lstrip" => text.trim_start_matches(in_set),
        "rstrip" => text.trim_end_matches(in_set),
        _ => text.trim_matches(in_set),
    }
}

/// Python's `str.isspace()` character class, used by `strip()` and `split()`.
///
/// CPython counts characters with bidirectional class `WS`, `B`, or `S`, or
/// category `Zs`. That is Unicode `White_Space` (Rust's
/// [`char::is_whitespace`]) plus the ASCII separators `\x1c` through `\x1f`.
pub(crate) fn is_python_whitespace(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
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
            let mut rest = text.trim_end_matches(is_python_whitespace);
            while !rest.is_empty() {
                if limit.is_some_and(|limit| parts.len() == limit) {
                    parts.push(rest);
                    break;
                }
                let start = rest
                    .char_indices()
                    .rev()
                    .find(|(_, character)| is_python_whitespace(*character))
                    .map_or(0, |(index, character)| index + character.len_utf8());
                parts.push(&rest[start..]);
                rest = rest[..start].trim_end_matches(is_python_whitespace);
            }
            parts.reverse();
            Ok(parts)
        }
        None => {
            let mut parts = Vec::new();
            let mut rest = text.trim_start_matches(is_python_whitespace);
            while !rest.is_empty() {
                if limit.is_some_and(|limit| parts.len() == limit) {
                    parts.push(rest);
                    break;
                }
                let end = rest.find(is_python_whitespace).unwrap_or(rest.len());
                parts.push(&rest[..end]);
                rest = rest[end..].trim_start_matches(is_python_whitespace);
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
        // `title`: a character after a cased letter is lowercased, otherwise
        // uppercased (Python title-cases it; they differ only for digraphs like `ǆ`).
        _ => {
            let mut previous_cased = false;
            let mut titled = String::with_capacity(text.len());
            for character in text.chars() {
                if previous_cased {
                    titled.extend(character.to_lowercase());
                } else {
                    titled.extend(character.to_uppercase());
                }
                // Python's `is_cased`: only letters with case (not CJK) reset the word.
                previous_cased = character.is_lowercase() || character.is_uppercase();
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
        "isspace" => text.chars().all(is_python_whitespace),
        "islower" => has_cased && !text.chars().any(char::is_uppercase),
        _ => has_cased && !text.chars().any(char::is_lowercase),
    }
}

/// Dispatches one Python `dict` method: `items`, `keys`, `values`, or `get`.
///
/// The three list-building methods charge `budget` before they copy.
fn map_method(
    map: &Value,
    method: &str,
    values: &[Value],
    budget: &RenderBudget,
) -> Result<Value, Error> {
    let arguments = Arguments::new("dict", method, values);
    let entries = map.len().unwrap_or(0);
    let keys = || map.try_iter().map(Iterator::collect::<Vec<_>>);
    match method {
        "items" => {
            arguments.arity(0, 0)?;
            // A two-item list per entry, inside the outer list.
            budget.reserve_copies(entries, 3, &arguments.method)?;
            let pairs = keys()?
                .into_iter()
                .map(|key| Ok(Value::from(vec![key.clone(), map.get_item(&key)?])))
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Value::from(pairs))
        }
        "keys" => {
            arguments.arity(0, 0)?;
            budget.reserve_items(entries, &arguments.method)?;
            Ok(Value::from(keys()?))
        }
        "values" => {
            arguments.arity(0, 0)?;
            budget.reserve_items(entries, &arguments.method)?;
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
