//! Hugging Face-compatible, resource-bounded chat-template rendering.
//!
//! Hugging Face's `apply_chat_template` compiles a checkpoint's template with
//! `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
//! extensions=[AssistantTracker, loopcontrols])`, Jinja2's lenient
//! `Undefined`, a Python `json.dumps` `tojson` filter, and the globals
//! `raise_exception` and `strftime_now`. [`render_template`] reproduces that
//! environment on MiniJinja:
//!
//! | Hugging Face / Jinja2                     | Here                                          |
//! |-------------------------------------------|-----------------------------------------------|
//! | `trim_blocks`, `lstrip_blocks`            | the same MiniJinja whitespace switches        |
//! | lenient `Undefined` (`{% if x %}` is falsy) | [`UndefinedBehavior::Lenient`]                |
//! | `loopcontrols` (`break`, `continue`)      | MiniJinja's `loop_controls` feature           |
//! | `{% generation %}` (assistant masks)      | rewritten to a transparent `{% if true %}`    |
//! | `f(key=a if c else b)` keyword arguments  | rewritten to `f(key=(a if c else b))`         |
//! | `is iterable` / `is sequence` tests       | Python's answers for `None`, `str`, and `dict` |
//! | `tojson` = `json.dumps(...)`              | [`super::pycompat::tojson`]                   |
//! | Python `str`/`dict` methods and printing  | [`super::pycompat`] callbacks                 |
//! | `strftime_now(format)` (host local time)  | UTC, or a caller-fixed instant                |
//!
//! Templates are untrusted programs. The environment has no loader, so
//! templates cannot include files; execution is metered by fuel, recursion is
//! capped, and printed or serialized values have a bounded nesting depth.

use std::{
    borrow::Cow,
    time::{SystemTime, UNIX_EPOCH},
};

use minijinja::{Environment, Error, ErrorKind, UndefinedBehavior, Value};

use super::{calendar, pycompat};

/// Instruction budget limiting each chat-template evaluation.
const CHAT_TEMPLATE_FUEL: u64 = 250_000;
/// Maximum template call depth, bounding macro and include-like recursion.
const CHAT_TEMPLATE_RECURSION_LIMIT: usize = 64;

/// Renders one checkpoint template with Hugging Face's environment semantics.
///
/// `context` holds the template variables (`messages`, `tools`, special
/// tokens, ...). `fixed_time` pins the instant `strftime_now` formats, in Unix
/// seconds; `None` reads the host clock.
pub(crate) fn render_template(
    source: &str,
    context: &serde_json::Value,
    fixed_time: Option<i64>,
) -> Result<String, Error> {
    let source =
        parenthesize_conditional_keyword_arguments(&rewrite_generation_blocks(source)).into_owned();
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Lenient);
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment.set_recursion_limit(CHAT_TEMPLATE_RECURSION_LIMIT);
    environment.set_fuel(Some(CHAT_TEMPLATE_FUEL));
    environment.set_formatter(pycompat::python_formatter);
    environment.set_unknown_method_callback(pycompat::python_method);
    environment.add_filter("tojson", pycompat::tojson);
    environment.add_test("iterable", pycompat::is_iterable);
    environment.add_test("sequence", pycompat::is_sequence);
    environment.add_function("raise_exception", raise_exception);
    environment.add_function("strftime_now", move |format: &str| {
        strftime_now(format, fixed_time)
    });
    environment.render_str(&source, Value::from_serialize(context))
}

/// Implements Hugging Face's `raise_exception(message)` template global.
fn raise_exception(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

// Assistant-mask blocks.

/// Makes Hugging Face's `{% generation %}` blocks transparent.
///
/// Hugging Face's `AssistantTracker` extension uses
/// `{% generation %}...{% endgeneration %}` only to record which characters an
/// assistant produced; the rendered text is the block body unchanged. MiniJinja
/// has no such tag, so each one is rewritten to `{% if true %}`/`{% endif %}`
/// with its whitespace-control markers (`-`, `+`) preserved, which renders
/// byte-for-byte the same text. Comments and `{% raw %}` sections are copied
/// verbatim. Templates without the word `generation` are borrowed unchanged.
pub(crate) fn rewrite_generation_blocks(source: &str) -> Cow<'_, str> {
    if !source.contains("generation") {
        return Cow::Borrowed(source);
    }
    let mut rewritten = String::with_capacity(source.len());
    let mut rest = source;
    let mut in_raw = false;
    while let Some(start) = rest.find("{%").into_iter().chain(rest.find("{#")).min() {
        let closing = if rest[start..].starts_with("{#") {
            "#}"
        } else {
            "%}"
        };
        let Some(length) = rest[start + 2..].find(closing) else {
            // Unterminated: copy the remainder verbatim for the parser to reject.
            break;
        };
        let end = start + 2 + length + 2;
        rewritten.push_str(&rest[..start]);
        let tag = &rest[start..end];
        rest = &rest[end..];
        if closing == "#}" {
            rewritten.push_str(tag);
            continue;
        }
        let (open_marker, keyword, close_marker) = statement_parts(tag);
        let replacement = match keyword {
            "raw" if !in_raw => {
                in_raw = true;
                None
            }
            "endraw" if in_raw => {
                in_raw = false;
                None
            }
            "generation" if !in_raw => Some("if true"),
            "endgeneration" if !in_raw => Some("endif"),
            _ => None,
        };
        match replacement {
            Some(statement) => {
                rewritten.push_str("{%");
                rewritten.push_str(open_marker);
                rewritten.push(' ');
                rewritten.push_str(statement);
                rewritten.push(' ');
                rewritten.push_str(close_marker);
                rewritten.push_str("%}");
            }
            None => rewritten.push_str(tag),
        }
    }
    rewritten.push_str(rest);
    Cow::Owned(rewritten)
}

/// Splits `{%- keyword -%}` into its opening marker, trimmed body, and closing marker.
fn statement_parts(tag: &str) -> (&str, &str, &str) {
    let inner = &tag[2..tag.len() - 2];
    let marker = |text: &str| -> usize { usize::from(text.starts_with(['-', '+'])) };
    let open = marker(inner);
    let close = usize::from(inner.len() > open && inner.ends_with(['-', '+']));
    let body = &inner[open..inner.len() - close];
    (&inner[..open], body.trim(), &inner[inner.len() - close..])
}

// Conditional keyword arguments.

/// Parenthesizes conditional expressions used as keyword-argument values.
///
/// Jinja2 accepts `namespace(name=tcid if tcid else '')`, but MiniJinja parses
/// keyword-argument values without the conditional operator, so official
/// templates (Muse Glimmer's tool-result branch, for one) fail to compile.
/// Wrapping such a value in parentheses, `name=(tcid if tcid else '')`, is
/// semantically neutral in Jinja and accepted by both engines.
///
/// Only expression and statement tags are scanned; template text, comments,
/// and `{% raw %}` sections are copied unchanged. Inside a tag, string
/// literals are skipped, and a keyword argument is an identifier followed by a
/// single `=` directly inside parentheses. Its value ends at the next `,` or
/// `)` at the same nesting depth and is wrapped only if it contains a
/// top-level `if`.
pub(crate) fn parenthesize_conditional_keyword_arguments(source: &str) -> Cow<'_, str> {
    if !source.contains(" if ") {
        return Cow::Borrowed(source);
    }
    let mut rewritten = String::with_capacity(source.len() + 16);
    let mut rest = source;
    let mut in_raw = false;
    let mut changed = false;
    while let Some(start) = ["{{", "{%", "{#"]
        .iter()
        .filter_map(|opening| rest.find(opening))
        .min()
    {
        let opening = &rest[start..start + 2];
        let closing = match opening {
            "{{" => "}}",
            "{%" => "%}",
            _ => "#}",
        };
        let Some(length) = find_tag_end(&rest[start + 2..], closing) else {
            break;
        };
        let end = start + 2 + length + 2;
        rewritten.push_str(&rest[..start]);
        let tag = &rest[start..end];
        rest = &rest[end..];
        if opening == "{%" {
            match statement_parts(tag).1 {
                "raw" => in_raw = true,
                "endraw" => in_raw = false,
                _ => {}
            }
        }
        if in_raw || opening == "{#" {
            rewritten.push_str(tag);
            continue;
        }
        let body = &tag[2..tag.len() - 2];
        let wrapped = wrap_conditional_keyword_values(body);
        changed |= matches!(wrapped, Cow::Owned(_));
        rewritten.push_str(opening);
        rewritten.push_str(&wrapped);
        rewritten.push_str(closing);
    }
    rewritten.push_str(rest);
    if changed {
        Cow::Owned(rewritten)
    } else {
        Cow::Borrowed(source)
    }
}

/// Finds the closing delimiter of a tag body, skipping quoted string literals.
fn find_tag_end(body: &str, closing: &str) -> Option<usize> {
    let bytes = body.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            quote @ (b'\'' | b'"') if closing != "#}" => index = skip_string(bytes, index, quote),
            // Byte comparison: `index` may sit inside a multi-byte character.
            _ if bytes[index..].starts_with(closing.as_bytes()) => return Some(index),
            _ => index += 1,
        }
    }
    None
}

/// Returns the index just past the string literal opened by `quote` at `start`.
fn skip_string(bytes: &[u8], start: usize, quote: u8) -> usize {
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            byte if byte == quote => return index + 1,
            _ => index += 1,
        }
    }
    bytes.len()
}

/// One lexical item of a tag body that matters for keyword-argument detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lexeme<'a> {
    /// Identifier or keyword, such as `name` or `if`.
    Word(&'a str),
    /// One punctuation character such as `(` or `,`.
    Punct(u8),
    /// Two-character operator such as `==` or `<=`, distinct from `=`.
    Operator,
    /// String literal or number; opaque here.
    Literal,
}

/// Splits a tag body into lexemes with their byte ranges.
fn lex(body: &str) -> Vec<(Lexeme<'_>, usize, usize)> {
    let bytes = body.as_bytes();
    let mut lexemes = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let start = index;
        let byte = bytes[index];
        let lexeme = if byte.is_ascii_whitespace() {
            index += 1;
            continue;
        } else if byte == b'\'' || byte == b'"' {
            index = skip_string(bytes, index, byte);
            Lexeme::Literal
        } else if byte.is_ascii_alphabetic() || byte == b'_' {
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            Lexeme::Word(&body[start..index])
        } else if byte.is_ascii_digit() {
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'.')
            {
                index += 1;
            }
            Lexeme::Literal
        } else if matches!(byte, b'=' | b'!' | b'<' | b'>') && bytes.get(index + 1) == Some(&b'=') {
            index += 2;
            Lexeme::Operator
        } else {
            // Advance by a whole character so non-ASCII text stays on a boundary.
            index += body[index..].chars().next().map_or(1, char::len_utf8);
            Lexeme::Punct(byte)
        };
        lexemes.push((lexeme, start, index));
    }
    lexemes
}

/// Wraps every conditional keyword-argument value in one tag body.
fn wrap_conditional_keyword_values(body: &str) -> Cow<'_, str> {
    let lexemes = lex(body);
    // Byte offsets receiving an opening or closing parenthesis.
    let mut insertions = Vec::new();
    let mut brackets = Vec::new();
    for (position, (lexeme, _, _)) in lexemes.iter().enumerate() {
        match lexeme {
            Lexeme::Punct(open @ (b'(' | b'[' | b'{')) => brackets.push(*open),
            Lexeme::Punct(b')' | b']' | b'}') => {
                brackets.pop();
            }
            Lexeme::Punct(b'=')
                if brackets.last() == Some(&b'(')
                    && position > 0
                    && matches!(lexemes[position - 1].0, Lexeme::Word(_)) =>
            {
                if let Some((value_start, value_end)) =
                    conditional_value_span(&lexemes, position + 1)
                {
                    insertions.push((value_start, '('));
                    insertions.push((value_end, ')'));
                }
            }
            _ => {}
        }
    }
    if insertions.is_empty() {
        return Cow::Borrowed(body);
    }
    insertions.sort_by_key(|(offset, _)| *offset);
    let mut wrapped = String::with_capacity(body.len() + insertions.len());
    let mut copied = 0;
    for (offset, parenthesis) in insertions {
        wrapped.push_str(&body[copied..offset]);
        wrapped.push(parenthesis);
        copied = offset;
    }
    wrapped.push_str(&body[copied..]);
    Cow::Owned(wrapped)
}

/// Returns the byte span of a keyword value starting at lexeme `first` if it is conditional.
fn conditional_value_span(
    lexemes: &[(Lexeme<'_>, usize, usize)],
    first: usize,
) -> Option<(usize, usize)> {
    let mut depth = 0_usize;
    let mut conditional = false;
    let mut last = None;
    for (lexeme, _, end) in &lexemes[first..] {
        match lexeme {
            Lexeme::Punct(b'(' | b'[' | b'{') => depth += 1,
            Lexeme::Punct(b')' | b']' | b'}') if depth == 0 => break,
            Lexeme::Punct(b')' | b']' | b'}') => depth -= 1,
            Lexeme::Punct(b',') if depth == 0 => break,
            Lexeme::Word("if") if depth == 0 => conditional = true,
            _ => {}
        }
        last = Some(*end);
    }
    let start = lexemes.get(first)?.1;
    match (conditional, last) {
        (true, Some(end)) => Some((start, end)),
        _ => None,
    }
}

// `strftime_now`.

/// Formats the current (or caller-fixed) UTC instant like Python's `strftime`.
fn strftime_now(format: &str, fixed_time: Option<i64>) -> Result<String, Error> {
    let seconds = match fixed_time {
        Some(seconds) => seconds,
        None => host_unix_seconds()?,
    };
    let civil = calendar::civil_time(seconds).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("strftime_now instant {seconds} is outside years 1 through 9999"),
        )
    })?;
    strftime(format, &civil)
}

/// Reads the host clock as whole Unix seconds, rounding toward negative infinity.
fn host_unix_seconds() -> Result<i64, Error> {
    let clock_error = || Error::new(ErrorKind::InvalidOperation, "host clock is out of range");
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(elapsed) => i64::try_from(elapsed.as_secs()).map_err(|_| clock_error()),
        Err(before) => {
            let before = before.duration();
            let whole = i64::try_from(before.as_secs()).map_err(|_| clock_error())?;
            Ok(-whole - i64::from(before.subsec_nanos() > 0))
        }
    }
}

/// Abbreviated and full English weekday names, Monday first (C locale).
const WEEKDAYS: [(&str, &str); 7] = [
    ("Mon", "Monday"),
    ("Tue", "Tuesday"),
    ("Wed", "Wednesday"),
    ("Thu", "Thursday"),
    ("Fri", "Friday"),
    ("Sat", "Saturday"),
    ("Sun", "Sunday"),
];
/// Abbreviated and full English month names, January first (C locale).
const MONTHS: [(&str, &str); 12] = [
    ("Jan", "January"),
    ("Feb", "February"),
    ("Mar", "March"),
    ("Apr", "April"),
    ("May", "May"),
    ("Jun", "June"),
    ("Jul", "July"),
    ("Aug", "August"),
    ("Sep", "September"),
    ("Oct", "October"),
    ("Nov", "November"),
    ("Dec", "December"),
];

/// Formats civil time with the C-locale `strftime` directives templates use.
///
/// Supported: `%a %A %b %h %B %c %C %d %D %e %F %H %I %j %m %M %n %p %R %S %t
/// %T %u %w %x %X %y %Y %z %Z %%`, plus the glibc/BSD `-` flag that drops
/// padding from numeric fields (`%-d`). Week-number and locale-specific
/// directives are rejected rather than guessed.
pub(crate) fn strftime(format: &str, civil: &calendar::CivilTime) -> Result<String, Error> {
    let mut out = String::with_capacity(format.len() + 16);
    let mut characters = format.chars();
    while let Some(character) = characters.next() {
        if character != '%' {
            out.push(character);
            continue;
        }
        let mut directive = characters.next();
        let unpadded = directive == Some('-');
        if unpadded {
            directive = characters.next();
        }
        let directive = directive.ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidOperation,
                "strftime format ends with an incomplete directive",
            )
        })?;
        push_directive(&mut out, directive, unpadded, civil)?;
    }
    Ok(out)
}

/// Appends one `strftime` directive.
fn push_directive(
    out: &mut String,
    directive: char,
    unpadded: bool,
    civil: &calendar::CivilTime,
) -> Result<(), Error> {
    // Calendar fields are proven in range, so these indices cannot fail.
    let weekday = WEEKDAYS[usize::try_from(civil.weekday).unwrap_or(0)];
    let month = MONTHS[usize::try_from(civil.month - 1).unwrap_or(0)];
    let twelve_hour = match civil.hour % 12 {
        0 => 12,
        hour => hour,
    };
    let number = |value: i64, width: usize| {
        if unpadded {
            value.to_string()
        } else {
            format!("{value:0width$}")
        }
    };
    let text = match directive {
        'a' => weekday.0.to_owned(),
        'A' => weekday.1.to_owned(),
        'b' | 'h' => month.0.to_owned(),
        'B' => month.1.to_owned(),
        'c' => format!(
            "{} {} {:>2} {:02}:{:02}:{:02} {}",
            weekday.0, month.0, civil.day, civil.hour, civil.minute, civil.second, civil.year
        ),
        'C' => number(civil.year / 100, 2),
        'd' => number(civil.day, 2),
        'D' | 'x' => format!(
            "{:02}/{:02}/{:02}",
            civil.month,
            civil.day,
            civil.year % 100
        ),
        'e' if unpadded => civil.day.to_string(),
        'e' => format!("{:>2}", civil.day),
        'F' => format!("{:04}-{:02}-{:02}", civil.year, civil.month, civil.day),
        'H' => number(civil.hour, 2),
        'I' => number(twelve_hour, 2),
        'j' => number(civil.day_of_year, 3),
        'm' => number(civil.month, 2),
        'M' => number(civil.minute, 2),
        'n' => "\n".to_owned(),
        'p' => if civil.hour < 12 { "AM" } else { "PM" }.to_owned(),
        'R' => format!("{:02}:{:02}", civil.hour, civil.minute),
        'S' => number(civil.second, 2),
        't' => "\t".to_owned(),
        'T' | 'X' => format!("{:02}:{:02}:{:02}", civil.hour, civil.minute, civil.second),
        'u' => (civil.weekday + 1).to_string(),
        'w' => ((civil.weekday + 1) % 7).to_string(),
        'y' => number(civil.year % 100, 2),
        'Y' => number(civil.year, 4),
        'z' => "+0000".to_owned(),
        'Z' => "UTC".to_owned(),
        '%' => "%".to_owned(),
        other => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("unsupported strftime directive %{other}"),
            ));
        }
    };
    out.push_str(&text);
    Ok(())
}

#[cfg(test)]
#[path = "chat_template_tests.rs"]
mod tests;
