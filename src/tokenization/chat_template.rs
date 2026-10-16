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
//! Sequence `+` and `*` produce lists, as in Python (MiniJinja alone builds
//! lazy iterables, which print with JSON-style quotes and cannot be indexed).
//!
//! # Sandbox
//!
//! Templates are untrusted programs: a checkpoint can ship any template. Each
//! render is bounded as follows.
//!
//! - **Capabilities.** The environment has no loader, so `include`,
//!   `import`, and `extends` fail; the only globals are `raise_exception`,
//!   `strftime_now`, and MiniJinja's `range`, `dict`, and `namespace`
//!   (MiniJinja's `debug()`, which prints the whole engine state without a
//!   limit and which Hugging Face's environment lacks, is removed).
//! - **Instructions.** At most 250 000 units of fuel (one per VM
//!   instruction); macro and call recursion at most 64 levels; `range` at most
//!   100 000 items (MiniJinja's own limit).
//! - **Parsing.** The `guard` module refuses expressions nested deeper than
//!   200 levels or chaining more than 1 024 operators (unary, binary,
//!   comparison, conditional, postfix, or filter), which would otherwise
//!   overflow MiniJinja's recursive parser, code generator, or syntax-tree
//!   destructor (a native stack overflow aborts the process). Flat lists
//!   (literal items, call arguments, macro parameters) are not limited.
//! - **Output.** The rendered prompt streams into a buffer of at most
//!   `1 MiB + 4c` bytes, where `c` is the size of the template context
//!   (`messages`, `tools`, ...) serialized as JSON; the write that crosses it
//!   fails the render.
//! - **Values.** Every operation that builds a new string or container is
//!   checked against the same per-value limit (and a sequence against as
//!   many items as a value may hold bytes, at least 2^20), and charged to a
//!   cumulative budget of 16 times the output limit, counting sequence items
//!   as 24 bytes. Since every copy is charged when it is made, the budget also
//!   bounds what a template can keep alive (in namespaces, lists, or
//!   captures). The charged operations are:
//!   - the binary operators `+`, `~`, and `*`, string and byte slices
//!     (`s[a:b:c]`), and `*sequence` / `**mapping` call arguments, all
//!     rewritten into bounded filters by the `guard` module (so constant
//!     operands are not folded at compile time either);
//!   - every `{{ }}` print, including prints into `{% set %}` blocks,
//!     macros, and `caller()`;
//!   - the Python-compatible filters and methods (`string`, `join`, `trim`,
//!     `safe`, `tojson`, and the `str` methods `replace`, `join`, `strip`,
//!     `split`, `splitlines`, `removeprefix`, `removesuffix`, and case
//!     changes, and the `dict` methods `items`, `keys`, and `values`);
//!   - the builtins whose size an argument controls or that print arbitrary
//!     values (`indent`, `replace`, `format`, `batch`, `slice`, `chain`,
//!     `upper`, `lower`, `title`, `capitalize`, `escape`, `pprint`, and the
//!     `startingwith`/`endingwith` tests);
//!   - the builtins that copy their input (`list`, `sort`, `unique`, `map`,
//!     `select`, `reject`, `selectattr`, `rejectattr`, `dictsort`, `groupby`,
//!     `reverse`, `lines`), the `dict()` and `namespace()` functions, and
//!     `strftime_now`, whose format is checked against its largest possible
//!     expansion before it is formatted.
//!
//!   Checks happen before the value is built, except for string slices,
//!   which the engine builds (at most one input-sized copy) before the
//!   result is charged. Sequence slices, `split`, `items`, and `zip` are
//!   lazy views that copy nothing.
//! - **Panics.** A panic inside MiniJinja is caught and reported as a
//!   rendering error; it never unwinds into the caller. MiniJinja's own
//!   `divisibleby` test panics on an integer divisor of zero, so it is
//!   replaced by a checked version that reports the error Jinja2 raises.
//!   Some negative-step slices of empty or short values still panic inside
//!   MiniJinja (`''[::-1]`); those become rendering errors, not crashes.
//! - **Cycles.** Namespaces are MiniJinja's only mutable values, so a
//!   template can make one reference itself (`{% set ns.me = ns %}`), and the
//!   cycle would outlive the render and leak everything it reaches. Every
//!   namespace a render creates is recorded (and charged to the budget, since
//!   the record keeps it alive), and its attributes that can hold a
//!   reference are cleared when the render ends (even by a panic), as
//!   MiniJinja already does for macro closures. The clearing costs one
//!   assignment per such attribute, which the render itself already paid for.
//!
//! What remains unbounded, because MiniJinja exposes no hook for it:
//!
//! - **Shared structure in comparisons.** Containers can hold one value many
//!   times (`{% set ns.x = [ns.x, ns.x] %}` doubles the printed size per
//!   iteration while allocating a constant). Printing such a value is bounded
//!   as above, but `==`, `<`, `in`, `sort`, `unique`, `min`, `max`, and
//!   `dictsort` compare element by element and can take time exponential in
//!   the number of instructions.
//! - **Literal text replayed into captures.** Template text inside a loop is
//!   appended to a `{% set %}` block, macro result, or `caller()` without a
//!   print, so a capture can grow by the template's size per instruction (at
//!   most fuel times template size); only the rendered prompt itself is
//!   bounded.
//! - **Constant-size work per instruction.** Each instruction may still
//!   allocate a small constant amount that is not charged (a two-item list
//!   literal, a loop frame, a macro call's argument list); the fuel limit
//!   bounds the total. Peak memory is therefore about the cumulative budget
//!   times a small copy factor (MiniJinja's transient copies and Rust's
//!   string growth) plus fuel times that constant.
//! - **Allocation failure.** An allocation MiniJinja cannot satisfy aborts
//!   the process (Rust's allocation-failure behavior), rather than panicking;
//!   the bounds above keep every allocation they cover below the limits.

use std::{
    borrow::Cow,
    convert::Infallible,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use minijinja::{Environment, Error, ErrorKind, UndefinedBehavior, Value};

use super::{calendar, pycompat};

mod guard;
mod limits;
pub(crate) mod sandbox;
mod scan;

use sandbox::{BoundedOutput, CycleBreaker, NamespaceRegistry, RenderBudget, RenderLimits};
use scan::{TagKind, TagSpan, marker_widths, next_tag, skip_string};

/// Instruction budget limiting each chat-template evaluation.
const CHAT_TEMPLATE_FUEL: u64 = 250_000;
/// Maximum template call depth, bounding macro and include-like recursion.
const CHAT_TEMPLATE_RECURSION_LIMIT: usize = 64;

/// Renders one checkpoint template with Hugging Face's environment semantics.
///
/// `context` holds the template variables (`messages`, `tools`, special
/// tokens, ...). `fixed_time` pins the instant `strftime_now` formats, in Unix
/// seconds; `None` reads the host clock. Resource limits follow from the
/// context's size; see the module documentation.
///
/// # Errors
///
/// Returns MiniJinja's error for a template that fails to compile or render,
/// exceeds a resource limit, or panics inside the engine.
pub(crate) fn render_template(
    source: &str,
    context: &serde_json::Value,
    fixed_time: Option<i64>,
) -> Result<String, Error> {
    let limits = RenderLimits::for_context(context);
    render_template_with(source, context, fixed_time, limits, |_| {})
}

/// [`render_template`] with explicit limits and an environment extension.
///
/// Tests use the limits to exercise each bound cheaply and the extension to
/// register extra callbacks; production passes [`RenderLimits::for_context`]
/// and no extension. Every source rewrite, the compilation, and the render
/// run inside `catch_unwind`: a panic in MiniJinja (or in a callback) becomes
/// an error instead of unwinding into the caller.
pub(crate) fn render_template_with(
    source: &str,
    context: &serde_json::Value,
    fixed_time: Option<i64>,
    limits: RenderLimits,
    extend: impl FnOnce(&mut Environment<'_>),
) -> Result<String, Error> {
    catch_unwind(AssertUnwindSafe(|| {
        render_bounded(source, context, fixed_time, limits, extend)
    }))
    .unwrap_or_else(|payload| {
        let detail = payload
            .downcast_ref::<&str>()
            .map(|message| (*message).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a non-text panic payload".to_owned());
        Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("the template engine panicked ({detail}); the template was rejected"),
        ))
    })
}

/// Rewrites, compiles, and renders one template under `limits`.
fn render_bounded(
    source: &str,
    context: &serde_json::Value,
    fixed_time: Option<i64>,
    limits: RenderLimits,
    extend: impl FnOnce(&mut Environment<'_>),
) -> Result<String, Error> {
    let rewritten =
        parenthesize_conditional_keyword_arguments(&rewrite_generation_blocks(source)).into_owned();
    let source = guard::guard_operators(&rewritten)
        .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))?
        .into_owned();
    let budget = Arc::new(RenderBudget::new(limits));
    let registry = Arc::new(NamespaceRegistry::default());
    // Declared before the environment and template so it drops after them,
    // once nothing but cycles can still hold the namespaces.
    let _cycles = CycleBreaker(Arc::clone(&registry));
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Lenient);
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment.set_recursion_limit(CHAT_TEMPLATE_RECURSION_LIMIT);
    environment.set_fuel(Some(CHAT_TEMPLATE_FUEL));
    pycompat::install(&mut environment, &budget);
    sandbox::install(&mut environment, &budget);
    sandbox::install_namespace_registry(&mut environment, &registry, &budget);
    environment.add_function("raise_exception", raise_exception);
    let shared = Arc::clone(&budget);
    environment.add_function("strftime_now", move |format: &str| {
        strftime_now(format, fixed_time, &shared)
    });
    extend(&mut environment);
    let template = environment.template_from_str(&source)?;
    let mut output = BoundedOutput::new(limits.output_bytes);
    let rendered = template
        .render_captured_to(Value::from_serialize(context), &mut output)
        .map(drop);
    if output.exceeded() {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            format!(
                "chat template resource limit: the rendered prompt would exceed {} bytes",
                limits.output_bytes
            ),
        ));
    }
    rendered?;
    output.into_string()
}

/// Implements Hugging Face's `raise_exception(message)` template global.
fn raise_exception(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

// Lexical tag scanning shared by the source rewrites.

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
    match try_rewrite_tags(source, |kind, _, tag| {
        Ok::<_, Infallible>(rewrite(kind, tag))
    }) {
        Ok(rewritten) => rewritten,
        Err(never) => match never {},
    }
}

/// [`rewrite_tags`] with a fallible rewrite that also receives each tag's byte offset.
///
/// The first error stops the scan and is returned. Tag boundaries come from
/// the verified [`next_tag`]: every span lies inside the source, starts at or
/// after the previous one's end, and is delimited by ASCII bytes, so every
/// slice below is in bounds and on character boundaries.
fn try_rewrite_tags<E>(
    source: &str,
    mut rewrite: impl FnMut(TagKind, usize, &str) -> Result<Option<String>, E>,
) -> Result<Cow<'_, str>, E> {
    let mut rewritten = String::new();
    // Bytes of `source` already copied into `rewritten`; zero until the first change.
    let mut copied = 0;
    let mut position = 0;
    while let Some(TagSpan { kind, start, end }) = next_tag(source.as_bytes(), position) {
        if let Some(replacement) = rewrite(kind, start, &source[start..end])? {
            rewritten.push_str(&source[copied..start]);
            rewritten.push_str(&replacement);
            copied = end;
        }
        position = end;
    }
    Ok(if copied == 0 {
        Cow::Borrowed(source)
    } else {
        rewritten.push_str(&source[copied..]);
        Cow::Owned(rewritten)
    })
}

/// Splits `{%- keyword -%}` into its opening marker, trimmed body, and closing marker.
///
/// `tag` holds both two-byte delimiters; the marker widths come from the
/// verified [`marker_widths`], which never exceed the inner text.
fn statement_parts(tag: &str) -> (&str, &str, &str) {
    let inner = &tag[2..tag.len() - 2];
    let (open, close) = marker_widths(inner.as_bytes());
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

/// One lexical item of a tag body that matters for keyword-argument detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lexeme<'a> {
    /// Identifier or keyword, such as `name` or `if`.
    Word(
        /// The identifier's text, borrowed from the tag body.
        &'a str,
    ),
    /// One punctuation character such as `(` or `,`.
    Punct(
        /// The punctuation character's ASCII byte.
        u8,
    ),
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

/// Most bytes [`strftime`] writes per byte of its format.
///
/// The longest directive, `%c`, writes 24 bytes (`Fri Jul 26 13:05:09 2024`,
/// years have at most four digits) for its two format bytes; every other
/// directive writes less, and literal text is copied byte for byte.
const STRFTIME_EXPANSION: usize = 12;

/// Formats the current (or caller-fixed) UTC instant like Python's `strftime`.
///
/// The result is at most [`STRFTIME_EXPANSION`] times the format's length.
/// That bound must fit the render's remaining text allowance before
/// formatting starts, and the actual result is charged to `budget`.
fn strftime_now(
    format: &str,
    fixed_time: Option<i64>,
    budget: &RenderBudget,
) -> Result<String, Error> {
    let bound = format.len().saturating_mul(STRFTIME_EXPANSION);
    if bound > budget.text_allowance() {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            format!(
                "chat template resource limit: strftime_now with a {}-byte format could produce {bound} bytes, above the {} bytes this render may still produce",
                format.len(),
                budget.text_allowance()
            ),
        ));
    }
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
    let text = strftime(format, &civil)?;
    budget.reserve_text(text.len(), "strftime_now")?;
    Ok(text)
}

/// Reads the host clock as whole Unix seconds, rounding toward negative infinity.
fn host_unix_seconds() -> Result<i64, Error> {
    let clock_error = || Error::new(ErrorKind::InvalidOperation, "host clock is out of range");
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(elapsed) => i64::try_from(elapsed.as_secs()).map_err(|_| clock_error()),
        Err(before) => {
            let before = before.duration();
            let whole = i64::try_from(before.as_secs()).map_err(|_| clock_error())?;
            // Verified floor of `-(whole + fraction)`.
            Ok(calendar::seconds_before_epoch(
                whole,
                before.subsec_nanos() > 0,
            ))
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
