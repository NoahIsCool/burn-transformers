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
//! | `tojson` = `json.dumps(...)`              | `pycompat::tojson`                            |
//! | `string`, `join`, `safe`, `trim` (`str()`) | `pycompat` filters printing `True`/`None`    |
//! | `round` (ties to even, `ceil`/`floor`)    | `pycompat::round`                             |
//! | Python `str`/`dict` methods and printing  | `pycompat` callbacks                          |
//! | `strftime_now(format)` (host local time)  | UTC, or a caller-fixed instant                |
//! | dict insertion order                      | **not matched**: keys arrive sorted           |
//! | `'x' ~ value` with a bool or `None`       | **not matched**: prints `true`/`none`         |
//! | `'%s' % value` string formatting          | **not supported**: rendering fails            |
//!
//! The unmatched rows, and smaller Unicode-class approximations, are
//! described in the `pycompat` module documentation. Official templates
//! convert values with `| string` before `~`, which is matched exactly.
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
    pycompat::install(&mut environment);
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

// Lexical tag scanning shared by the source rewrites.

/// The kinds of Jinja tag a source rewrite may change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TagKind {
    /// An expression tag, `{{ ... }}`.
    Expression,
    /// A statement tag, `{% ... %}`.
    Statement,
}

/// Applies `rewrite` to every expression and statement tag MiniJinja would parse.
///
/// The scan follows MiniJinja's lexer so a rewrite can never see text the
/// engine treats as data:
///
/// - template text, `{# comments #}`, and `{% raw %}...{% endraw %}` sections
///   are copied byte for byte;
/// - a tag ends at the first closing delimiter that lies outside string
///   literals and outside open `(`, `[`, or `{` brackets, so neither
///   `{{ '%}' }}` nor `{{ {'a': {'b': 1}} }}` is cut short.
///
/// `rewrite` receives a tag's full text, delimiters included, and returns its
/// replacement, or `None` to keep it. An unterminated tag or raw section ends
/// the scan, and the remainder is copied for MiniJinja's parser to reject.
/// An unchanged source is borrowed, not copied.
fn rewrite_tags(
    source: &str,
    mut rewrite: impl FnMut(TagKind, &str) -> Option<String>,
) -> Cow<'_, str> {
    let mut rewritten = String::new();
    // Bytes of `source` already copied into `rewritten`; zero until the first change.
    let mut copied = 0;
    let mut position = 0;
    while let Some((start, opening)) = next_tag_opening(source, position) {
        let body_start = start + 2;
        if opening == "{#" {
            let Some(length) = source[body_start..].find("#}") else {
                break;
            };
            position = body_start + length + 2;
            continue;
        }
        if opening == "{%"
            && let Some(length) = basic_tag_length(&source[body_start..], "raw")
        {
            // Like MiniJinja, copy everything up to the matching `{% endraw %}`.
            let Some(raw_length) = raw_section_length(&source[body_start + length..]) else {
                break;
            };
            position = body_start + length + raw_length;
            continue;
        }
        let (kind, closing) = if opening == "{{" {
            (TagKind::Expression, "}}")
        } else {
            (TagKind::Statement, "%}")
        };
        let Some(length) = find_tag_end(&source[body_start..], closing) else {
            break;
        };
        let end = body_start + length + closing.len();
        if let Some(replacement) = rewrite(kind, &source[start..end]) {
            rewritten.push_str(&source[copied..start]);
            rewritten.push_str(&replacement);
            copied = end;
        }
        position = end;
    }
    if copied == 0 {
        Cow::Borrowed(source)
    } else {
        rewritten.push_str(&source[copied..]);
        Cow::Owned(rewritten)
    }
}

/// Returns the byte offset and delimiter of the first `{{`, `{%`, or `{#` at or after `from`.
fn next_tag_opening(source: &str, from: usize) -> Option<(usize, &'static str)> {
    let bytes = source.as_bytes();
    let mut index = from;
    while let Some(offset) = source[index..].find('{') {
        let brace = index + offset;
        match bytes.get(brace + 1) {
            Some(b'{') => return Some((brace, "{{")),
            Some(b'%') => return Some((brace, "{%")),
            Some(b'#') => return Some((brace, "{#")),
            _ => index = brace + 1,
        }
    }
    None
}

/// Matches `[-+]? ws* name ws* [-+]? %}` at the start of `text` (the text after `{%`).
///
/// Returns the matched length. This is MiniJinja's `skip_basic_tag`, which
/// recognizes `raw` and `endraw` before any expression parsing.
fn basic_tag_length(text: &str, name: &str) -> Option<usize> {
    /// Skips the ASCII whitespace MiniJinja allows around the tag name.
    fn skip_whitespace(text: &str) -> &str {
        text.trim_start_matches(|character: char| character.is_ascii_whitespace())
    }
    let rest = text.strip_prefix(['-', '+']).unwrap_or(text);
    let rest = skip_whitespace(rest).strip_prefix(name)?;
    let rest = skip_whitespace(rest);
    let rest = rest.strip_prefix(['-', '+']).unwrap_or(rest);
    let rest = rest.strip_prefix("%}")?;
    Some(text.len() - rest.len())
}

/// Returns the length of a raw section's body plus its `{% endraw %}` tag.
///
/// As in MiniJinja, the first `{%` that starts an `endraw` tag ends the
/// section; every other byte is data.
fn raw_section_length(text: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(offset) = text[from..].find("{%") {
        let after_opening = from + offset + 2;
        if let Some(length) = basic_tag_length(&text[after_opening..], "endraw") {
            return Some(after_opening + length);
        }
        from = after_opening;
    }
    None
}

/// Finds a tag's closing delimiter outside string literals and open brackets.
///
/// `body` starts just after the opening delimiter. Like MiniJinja's lexer,
/// the delimiter only counts at bracket depth zero, so a `}}` that closes a
/// nested dict literal does not end an expression tag.
fn find_tag_end(body: &str, closing: &str) -> Option<usize> {
    let bytes = body.as_bytes();
    let mut depth = 0_isize;
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            quote @ (b'\'' | b'"') => {
                index = skip_string(bytes, index, quote);
                continue;
            }
            // Byte comparison: `index` may sit inside a multi-byte character.
            _ if depth == 0 && bytes[index..].starts_with(closing.as_bytes()) => {
                return Some(index);
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            _ => {}
        }
        index += 1;
    }
    None
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

// Assistant-mask blocks.

/// Makes Hugging Face's `{% generation %}` blocks transparent.
///
/// Hugging Face's `AssistantTracker` extension uses
/// `{% generation %}...{% endgeneration %}` only to record which characters an
/// assistant produced; the rendered text is the block body unchanged. MiniJinja
/// has no such tag, so each one is rewritten to `{% if true %}`/`{% endif %}`
/// with its whitespace-control markers (`-`, `+`) preserved, which renders
/// byte-for-byte the same text. Only real statement tags are rewritten: the
/// same words inside text, comments, raw sections, or string literals (such as
/// `{{ '{% generation %}' }}`) are data and stay unchanged. Templates without
/// the word `generation` are borrowed unchanged.
pub(crate) fn rewrite_generation_blocks(source: &str) -> Cow<'_, str> {
    if !source.contains("generation") {
        return Cow::Borrowed(source);
    }
    rewrite_tags(source, |kind, tag| {
        if kind != TagKind::Statement {
            return None;
        }
        let (open_marker, keyword, close_marker) = statement_parts(tag);
        let statement = match keyword {
            "generation" => "if true",
            "endgeneration" => "endif",
            _ => return None,
        };
        Some(format!("{{%{open_marker} {statement} {close_marker}%}}"))
    })
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
/// Only expression and statement tags are scanned (see [`rewrite_tags`]).
/// Inside a tag, string literals are skipped, and a keyword argument is an
/// identifier followed by a single `=` directly inside parentheses. Its value
/// ends at the next `,` or `)` at the same nesting depth and is wrapped only
/// if it contains a top-level `if`.
pub(crate) fn parenthesize_conditional_keyword_arguments(source: &str) -> Cow<'_, str> {
    if !source.contains("if") {
        return Cow::Borrowed(source);
    }
    rewrite_tags(source, |_, tag| {
        let (opening, body, closing) = (&tag[..2], &tag[2..tag.len() - 2], &tag[tag.len() - 2..]);
        match wrap_conditional_keyword_values(body) {
            Cow::Borrowed(_) => None,
            Cow::Owned(wrapped) => Some(format!("{opening}{wrapped}{closing}")),
        }
    })
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
