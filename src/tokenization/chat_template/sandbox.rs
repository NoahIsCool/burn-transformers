//! Resource limits for one chat-template render, and the bounded operations that enforce them.
//!
//! MiniJinja meters a template with fuel, one unit per VM instruction, but a
//! single instruction can do unbounded work: `'ab' * n` allocates `2n` bytes,
//! `s ~ s` doubles a string, `l + l` builds a lazy chain whose iteration cost
//! doubles with every concatenation, and `x | indent(n)` pads every line. The
//! engine has no hook for any of these, so this module supplies replacements
//! that check sizes *before* allocating:
//!
//! - [`RenderLimits`] fixes, per render, the largest rendered prompt, the
//!   largest single string, the longest single sequence, and a cumulative
//!   budget for everything the bounded operations produce. Limits scale with
//!   the size of the caller's template context, so a long conversation is
//!   never refused for being long.
//! - [`RenderBudget`] is the per-render counter those operations charge. It
//!   counts *production*, not live memory (MiniJinja does not report frees),
//!   so it bounds peak memory from above.
//! - The `__bounded_*` filters implement `+`, `~`, and `*` with MiniJinja's
//!   exact semantics plus those checks. The source rewrite in the `guard`
//!   module routes every binary use of the operators to them. Sequences are
//!   materialized as lists instead of MiniJinja's lazy chains, which is also
//!   what Python's Jinja2 (and so Hugging Face) produces.
//! - [`install`] also replaces the builtin filters whose output size is set by
//!   an argument (`indent`, `replace`, `format`, `batch`, `slice`, `chain`) or
//!   that print arbitrary values with MiniJinja's `Display` (`upper`, `lower`,
//!   `title`, `capitalize`, `escape`, `pprint`, and the `startingwith` and
//!   `endingwith` tests) with bounded versions.
//!
//! Numbers keep MiniJinja's own arithmetic: the bounded operators delegate
//! every operand pair they do not size-check to the engine's operator.

use core::fmt;
use std::{
    borrow::Cow,
    collections::BTreeSet,
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use minijinja::{
    Environment, Error, ErrorKind, State, Value, filters, functions, tests as builtin_tests,
    value::{Kwargs, Rest, ValueKind},
};

use super::limits;

/// Filter that implements `left + right` (see [`bounded_add`]).
pub(crate) const ADD_FILTER: &str = "__bounded_add";
/// Filter that implements `left ~ right` (see [`bounded_concat`]).
pub(crate) const CONCAT_FILTER: &str = "__bounded_concat";
/// Filter that implements `left * right` (see [`bounded_multiply`]).
pub(crate) const MULTIPLY_FILTER: &str = "__bounded_mul";

/// Longest sequence one bounded operation may build.
///
/// A chat has at most a few thousand messages; a million items costs 24 MiB
/// of `Value`s.
const SEQUENCE_ITEMS: usize = 1 << 20;
/// Bytes charged per sequence item: one MiniJinja `Value`.
const ITEM_BYTES: usize = size_of::<Value>();

/// Size limits of one render, derived from the size of its template context.
///
/// With `c` the serialized context bytes, the rendered prompt may hold
/// `16 MiB + 4c` bytes ([`Self::output_bytes`]), and so may any single string
/// ([`Self::value_bytes`]); a sequence may hold `2^20` items; and every
/// bounded operation together may produce `16` times the prompt limit
/// ([`Self::total_bytes`]), counting sequence items as 24 bytes each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RenderLimits {
    /// Largest rendered prompt, in bytes.
    pub(crate) output_bytes: usize,
    /// Largest string one operation may produce, in bytes.
    pub(crate) value_bytes: usize,
    /// Longest sequence one operation may produce, in items.
    pub(crate) sequence_items: usize,
    /// Cumulative bytes all bounded operations may produce in one render.
    pub(crate) total_bytes: usize,
}

impl RenderLimits {
    /// Limits for a template context that serializes to `context_bytes` of JSON.
    ///
    /// Every product saturates, so even an absurd context size yields valid
    /// (merely huge) limits rather than an overflow.
    pub(crate) fn for_context_bytes(context_bytes: usize) -> Self {
        let output_bytes = limits::output_limit(context_bytes);
        Self {
            output_bytes,
            value_bytes: output_bytes,
            sequence_items: SEQUENCE_ITEMS,
            total_bytes: limits::total_budget(output_bytes),
        }
    }

    /// Limits for `context`, measured as its compact JSON serialization.
    pub(crate) fn for_context(context: &serde_json::Value) -> Self {
        let mut counter = ByteCounter(0);
        // Writing into a counter cannot fail; a failure would only shrink the limit.
        let _ = serde_json::to_writer(&mut counter, context);
        Self::for_context_bytes(counter.0)
    }
}

/// An [`io::Write`] sink that only counts bytes, for sizing a serialization without storing it.
struct ByteCounter(usize);

impl io::Write for ByteCounter {
    /// Counts `buffer` and discards it.
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0 = self.0.saturating_add(buffer.len());
        Ok(buffer.len())
    }

    /// Nothing is buffered.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The per-render production counter that bounded operations charge.
///
/// MiniJinja may call filters from any thread it runs on, so the counter is
/// atomic; one render runs on one thread, so `Relaxed` ordering suffices.
#[derive(Debug)]
pub(crate) struct RenderBudget {
    /// Fixed limits of this render.
    limits: RenderLimits,
    /// Bytes produced so far by every bounded operation, at most `limits.total_bytes`.
    produced: AtomicUsize,
}

impl RenderBudget {
    /// A fresh budget with nothing produced.
    pub(crate) fn new(limits: RenderLimits) -> Self {
        Self {
            limits,
            produced: AtomicUsize::new(0),
        }
    }

    /// The limits of this render.
    pub(crate) fn limits(&self) -> RenderLimits {
        self.limits
    }

    /// Bytes produced so far.
    pub(crate) fn produced(&self) -> usize {
        self.produced.load(Ordering::Relaxed)
    }

    /// Largest text an operation may still produce: the per-value limit or the remaining budget.
    pub(crate) fn text_allowance(&self) -> usize {
        self.limits
            .value_bytes
            .min(self.limits.total_bytes.saturating_sub(self.produced()))
    }

    /// Charges a string of `bytes` produced by `operation`, or explains which limit it exceeds.
    pub(crate) fn reserve_text(&self, bytes: usize, operation: &str) -> Result<(), Error> {
        if bytes > self.limits.value_bytes {
            return Err(limit_error(format!(
                "{operation} would produce {bytes} bytes, above the {}-byte limit for one value",
                self.limits.value_bytes
            )));
        }
        self.charge(bytes, operation)
    }

    /// Charges a sequence of `items` produced by `operation`, or explains which limit it exceeds.
    pub(crate) fn reserve_items(&self, items: usize, operation: &str) -> Result<(), Error> {
        if items > self.limits.sequence_items {
            return Err(limit_error(format!(
                "{operation} would produce {items} items, above the {}-item limit for one sequence",
                self.limits.sequence_items
            )));
        }
        self.charge(items.saturating_mul(ITEM_BYTES), operation)
    }

    /// Adds `bytes` to the cumulative counter unless that would exceed the total budget.
    fn charge(&self, bytes: usize, operation: &str) -> Result<(), Error> {
        let total = self.limits.total_bytes;
        self.produced
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |produced| {
                limits::charged_total(produced, bytes, total)
            })
            .map(|_| ())
            .map_err(|produced| {
                limit_error(format!(
                    "{operation} would raise the bytes this render has produced from {produced} past its {total}-byte budget"
                ))
            })
    }
}

/// A resource-limit error with the sandbox's prefix.
fn limit_error(detail: String) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!("chat template resource limit: {detail}"),
    )
}

/// A [`fmt::Write`] buffer that refuses to grow past a fixed number of bytes.
///
/// Formatting stops with [`fmt::Error`] at the first write that would exceed
/// the limit, so printing a value whose text is enormous costs at most the
/// limit in time and memory.
#[derive(Debug, Default)]
pub(crate) struct LimitedText {
    /// Text written so far, never longer than `limit`.
    text: String,
    /// Largest permitted length in bytes.
    limit: usize,
    /// Whether a write was refused.
    exceeded: bool,
}

impl LimitedText {
    /// An empty buffer that holds at most `limit` bytes.
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            text: String::new(),
            limit,
            exceeded: false,
        }
    }

    /// Whether a write was refused.
    pub(crate) fn exceeded(&self) -> bool {
        self.exceeded
    }

    /// The text written so far.
    pub(crate) fn into_string(self) -> String {
        self.text
    }
}

impl fmt::Write for LimitedText {
    /// Appends `text` if it fits, else records the overflow and fails.
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.text.len().saturating_add(text.len()) > self.limit {
            self.exceeded = true;
            return Err(fmt::Error);
        }
        self.text.push_str(text);
        Ok(())
    }
}

/// Formats `value` with MiniJinja's `Display` (`{}`) within the budget and charges the result.
///
/// This is the text MiniJinja's `~`, `Cow<str>` arguments, and `escape` use
/// for non-strings; containers that hold one value many times can print far
/// more text than they occupy, so the size is checked while printing.
pub(crate) fn display_within(
    budget: &RenderBudget,
    value: &Value,
    operation: &str,
) -> Result<String, Error> {
    let mut text = LimitedText::new(budget.text_allowance());
    if fmt::Write::write_fmt(&mut text, format_args!("{value}")).is_err() {
        return Err(printing_error(budget, operation, text.exceeded()));
    }
    let text = text.into_string();
    budget.reserve_text(text.len(), operation)?;
    Ok(text)
}

/// The error for a print that failed, distinguishing the limit from a value's own failure.
fn printing_error(budget: &RenderBudget, operation: &str, exceeded: bool) -> Error {
    if exceeded {
        limit_error(format!(
            "{operation} would print more than the {} bytes this render may still produce",
            budget.text_allowance()
        ))
    } else {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("{operation} failed to print a value"),
        )
    }
}

/// Converts an argument to text the way MiniJinja's `Cow<str>` arguments do, within the budget.
///
/// Strings pass through without a copy; keyword arguments are refused, as
/// MiniJinja refuses them; every other value is printed with `Display`.
fn text_argument<'a>(
    budget: &RenderBudget,
    value: &'a Value,
    operation: &str,
) -> Result<Cow<'a, str>, Error> {
    if let Some(text) = string_operand(value) {
        return Ok(Cow::Borrowed(text));
    }
    if value.is_kwargs() {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "cannot convert kwargs to string",
        ));
    }
    display_within(budget, value, operation).map(Cow::Owned)
}

// Bounded operators.

/// The engine operators the bounded filters delegate to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    /// `+`.
    Add,
    /// `*`.
    Multiply,
}

impl Operator {
    /// The expression that applies the engine's own operator to `left` and `right`.
    fn expression(self) -> &'static str {
        match self {
            Self::Add => "left + right",
            Self::Multiply => "left * right",
        }
    }
}

/// Applies MiniJinja's own operator, for operands whose result size is not a concern.
///
/// Numbers (checked `i128` and `f64` arithmetic) and every error case go
/// through the engine, so results and messages stay exactly MiniJinja's.
/// The two-variable expression is not a constant, so it is never folded.
fn engine_operation(
    state: &State<'_, '_>,
    operator: Operator,
    left: Value,
    right: Value,
) -> Result<Value, Error> {
    state
        .env()
        .compile_expression(operator.expression())?
        .eval(minijinja::context! { left => left, right => right })
}

/// Whether MiniJinja's `+` treats `value` as a sequence (a list or a lazy iterable).
fn is_sequence(value: &Value) -> bool {
    matches!(value.kind(), ValueKind::Seq | ValueKind::Iterable)
}

/// Collects every item of `values` into one list of at most the sequence limit.
///
/// Iteration stops one item past the limit, so a lazy iterable of unknown
/// (or enormous) length costs at most the limit to reject.
fn collect_items<'a>(
    budget: &RenderBudget,
    values: impl IntoIterator<Item = &'a Value>,
    operation: &str,
) -> Result<Vec<Value>, Error> {
    let limit = budget.limits().sequence_items;
    let mut items = Vec::new();
    for value in values {
        for item in value.try_iter()? {
            if items.len() == limit {
                return Err(limit_error(format!(
                    "{operation} would produce more than the {limit}-item limit for one sequence"
                )));
            }
            items.push(item);
        }
    }
    budget.reserve_items(items.len(), operation)?;
    Ok(items)
}

/// `left + right` with MiniJinja's semantics and bounded results.
///
/// - Two sequences concatenate into a list of at most the sequence limit
///   (MiniJinja builds a lazy chain; Python's Jinja2 builds a list).
/// - Two strings concatenate exactly as MiniJinja's `[left, right].concat()`,
///   after the length is checked.
/// - Everything else (numbers, and the type errors of mixed operands) is the
///   engine's own `+`.
pub(crate) fn bounded_add(
    state: &State<'_, '_>,
    budget: &RenderBudget,
    left: Value,
    right: Value,
) -> Result<Value, Error> {
    if is_sequence(&left) && is_sequence(&right) {
        return collect_items(budget, [&left, &right], "sequence concatenation `+`")
            .map(Value::from);
    }
    if let (Some(first), Some(second)) = (string_operand(&left), string_operand(&right)) {
        budget.reserve_text(
            first.len().saturating_add(second.len()),
            "string concatenation `+`",
        )?;
        return Ok(Value::from([first, second].concat()));
    }
    engine_operation(state, Operator::Add, left, right)
}

/// The text of a string operand, as MiniJinja's `+` coerces it (not bytes).
fn string_operand(value: &Value) -> Option<&str> {
    (value.kind() == ValueKind::String)
        .then(|| value.as_str())
        .flatten()
}

/// `left ~ right`: both operands printed with `Display` into one string, within the budget.
///
/// This is MiniJinja's `format!("{left}{right}")`, including its spelling of
/// booleans and `none` and the empty text of undefined values.
pub(crate) fn bounded_concat(
    budget: &RenderBudget,
    left: &Value,
    right: &Value,
) -> Result<Value, Error> {
    const OPERATION: &str = "string concatenation `~`";
    let mut text = LimitedText::new(budget.text_allowance());
    if fmt::Write::write_fmt(&mut text, format_args!("{left}{right}")).is_err() {
        return Err(printing_error(budget, OPERATION, text.exceeded()));
    }
    let text = text.into_string();
    budget.reserve_text(text.len(), OPERATION)?;
    Ok(Value::from(text))
}

/// `left * right` with MiniJinja's semantics and bounded results.
///
/// - A string times a non-negative integer (either order, the left operand
///   checked first) repeats the string, after `len * count` is checked
///   without overflow; any other count fails with MiniJinja's message.
/// - A sequence times a count repeats its items into a list of at most the
///   sequence limit (MiniJinja builds a lazy repetition), with MiniJinja's
///   messages for a non-integer count and an unsized iterable.
/// - Everything else is the engine's own `*`.
pub(crate) fn bounded_multiply(
    state: &State<'_, '_>,
    budget: &RenderBudget,
    left: Value,
    right: Value,
) -> Result<Value, Error> {
    if let Some((text, count)) = left
        .as_str()
        .map(|text| (text, &right))
        .or_else(|| right.as_str().map(|text| (text, &left)))
    {
        let count = count.as_usize().ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidOperation,
                "strings can only be multiplied with integers",
            )
        })?;
        let bytes = limits::repeated_size(text.len(), count).ok_or_else(|| {
            limit_error(format!(
                "string repetition `*` of {} bytes {count} times overflows",
                text.len()
            ))
        })?;
        budget.reserve_text(bytes, "string repetition `*`")?;
        return Ok(Value::from(text.repeat(count)));
    }
    // Like MiniJinja: the first operand that is an object is the candidate,
    // and it is repeated only when it is a sequence; a map on the left with a
    // list on the right is therefore the engine's type error.
    let candidate = if left.as_object().is_some() {
        Some((&left, &right))
    } else if right.as_object().is_some() {
        Some((&right, &left))
    } else {
        None
    };
    if let Some((sequence, count)) = candidate.filter(|(sequence, _)| is_sequence(sequence)) {
        return repeat_sequence(budget, sequence, count);
    }
    engine_operation(state, Operator::Multiply, left, right)
}

/// Repeats the items of `sequence` `count` times into one bounded list.
fn repeat_sequence(budget: &RenderBudget, sequence: &Value, count: &Value) -> Result<Value, Error> {
    const OPERATION: &str = "sequence repetition `*`";
    let count = count.as_usize().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidOperation,
            "sequences and iterables can only be multiplied with integers",
        )
    })?;
    let length = sequence.len().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidOperation,
            "cannot repeat unsized iterables",
        )
    })?;
    let items = limits::repeated_size(length, count).ok_or_else(|| {
        limit_error(format!(
            "{OPERATION} of {length} items {count} times overflows"
        ))
    })?;
    budget.reserve_items(items, OPERATION)?;
    let once = sequence.try_iter()?.collect::<Vec<_>>();
    let mut repeated = Vec::with_capacity(items);
    for _ in 0..count {
        repeated.extend(once.iter().take(length).cloned());
    }
    Ok(Value::from(repeated))
}

// Bounded replacements for builtin filters and tests.

/// Installs the bounded operators and every bounded builtin replacement in `environment`.
///
/// Filters and tests registered here shadow MiniJinja's builtins of the same
/// name; templates cannot redefine filters, so the replacements always apply.
pub(crate) fn install(environment: &mut Environment<'_>, budget: &Arc<RenderBudget>) {
    let shared = Arc::clone(budget);
    environment.add_filter(
        ADD_FILTER,
        move |state: &State<'_, '_>, left: Value, right: Value| {
            bounded_add(state, &shared, left, right)
        },
    );
    let shared = Arc::clone(budget);
    environment.add_filter(CONCAT_FILTER, move |left: Value, right: Value| {
        bounded_concat(&shared, &left, &right)
    });
    let shared = Arc::clone(budget);
    environment.add_filter(
        MULTIPLY_FILTER,
        move |state: &State<'_, '_>, left: Value, right: Value| {
            bounded_multiply(state, &shared, left, right)
        },
    );
    install_text_filters(environment, budget);
    install_sized_filters(environment, budget);
}

/// Replaces the builtins that print arbitrary values: case changes, `escape`, `pprint`, and two tests.
fn install_text_filters(environment: &mut Environment<'_>, budget: &Arc<RenderBudget>) {
    /// A builtin `fn(Cow<str>) -> String` filter.
    type CaseFilter = fn(Cow<'_, str>) -> String;
    for (name, filter) in [
        ("upper", filters::upper as CaseFilter),
        ("lower", filters::lower),
        ("title", filters::title),
        ("capitalize", filters::capitalize),
    ] {
        let shared = Arc::clone(budget);
        environment.add_filter(name, move |value: Value| -> Result<String, Error> {
            let text = text_argument(&shared, &value, name)?;
            // Unicode case mapping can lengthen text (`ß` -> `SS`) by at most 3x.
            shared.reserve_text(text.len().saturating_mul(3), name)?;
            Ok(filter(text))
        });
    }
    for name in ["escape", "e"] {
        let shared = Arc::clone(budget);
        environment.add_filter(
            name,
            move |state: &State<'_, '_>, value: Value| -> Result<Value, Error> {
                // HTML escaping writes at most six bytes (`&quot;`) per input byte.
                if value.as_str().is_none() {
                    let text = display_within(&shared, &value, name)?;
                    shared.reserve_text(text.len().saturating_mul(6), name)?;
                    return filters::escape(state, &Value::from(text));
                }
                let length = value.as_str().map_or(0, str::len);
                shared.reserve_text(length.saturating_mul(6), name)?;
                filters::escape(state, &value)
            },
        );
    }
    let shared = Arc::clone(budget);
    environment.add_filter("pprint", move |value: Value| -> Result<String, Error> {
        let mut text = LimitedText::new(shared.text_allowance());
        if fmt::Write::write_fmt(&mut text, format_args!("{value:#?}")).is_err() {
            return Err(printing_error(&shared, "pprint", text.exceeded()));
        }
        let text = text.into_string();
        shared.reserve_text(text.len(), "pprint")?;
        Ok(text)
    });
    /// A builtin `fn(Cow<str>, Cow<str>) -> bool` test.
    type AffixTest = fn(Cow<'_, str>, Cow<'_, str>) -> bool;
    for (name, test) in [
        ("startingwith", builtin_tests::is_startingwith as AffixTest),
        ("endingwith", builtin_tests::is_endingwith),
    ] {
        let shared = Arc::clone(budget);
        environment.add_test(
            name,
            move |value: Value, other: Value| -> Result<bool, Error> {
                let value = text_argument(&shared, &value, name)?;
                let other = text_argument(&shared, &other, name)?;
                Ok(test(value, other))
            },
        );
    }
}

/// Replaces the builtins whose output size an argument controls.
fn install_sized_filters(environment: &mut Environment<'_>, budget: &Arc<RenderBudget>) {
    let shared = Arc::clone(budget);
    environment.add_filter(
        "indent",
        move |value: Value,
              width: Option<usize>,
              first: Option<bool>,
              blank: Option<bool>,
              kwargs: Kwargs|
              -> Result<String, Error> {
            let text = text_argument(&shared, &value, "indent")?.into_owned();
            let resolved = match width {
                Some(width) => width,
                None => kwargs.peek::<Option<usize>>("width")?.unwrap_or(4),
            };
            // Every line gains at most `width` spaces and keeps its newline.
            let lines = text.bytes().filter(|&byte| byte == b'\n').count() + 1;
            shared.reserve_text(
                limits::indentation_bound(text.len(), lines, resolved),
                "indent",
            )?;
            filters::indent(text, width, first, blank, kwargs)
        },
    );
    let shared = Arc::clone(budget);
    environment.add_filter(
        "replace",
        move |state: &State<'_, '_>,
              value: Value,
              from: Value,
              to: Value|
              -> Result<String, Error> {
            let text = text_argument(&shared, &value, "replace")?;
            let from = text_argument(&shared, &from, "replace")?;
            let to = text_argument(&shared, &to, "replace")?;
            shared.reserve_text(replaced_length(&text, &from, &to, None), "replace")?;
            Ok(filters::replace(state, text, from, to))
        },
    );
    let shared = Arc::clone(budget);
    environment.add_filter(
        "format",
        move |format: &str, arguments: Rest<Value>| -> Result<String, Error> {
            check_format_widths(&shared, format)?;
            let mut printed = format.len();
            for argument in arguments.iter() {
                let text = display_within(&shared, argument, "format")?;
                printed = printed.saturating_add(text.len());
            }
            shared.reserve_text(printed, "format")?;
            filters::format(format, arguments)
        },
    );
    for name in ["batch", "slice"] {
        let shared = Arc::clone(budget);
        environment.add_filter(
            name,
            move |state: &State<'_, '_>,
                  value: Value,
                  count: usize,
                  fill: Option<Value>|
                  -> Result<Value, Error> {
                // `batch` pads its last group to `count` items and `slice`
                // builds `count` groups, so `count` itself must be bounded.
                let items = value.len().unwrap_or(0);
                shared.reserve_items(items.saturating_add(count), name)?;
                if name == "batch" {
                    filters::batch(state, value, count, fill)
                } else {
                    filters::slice(state, value, count, fill)
                }
            },
        );
    }
    let shared = Arc::clone(budget);
    environment.add_filter(
        "chain",
        move |state: &State<'_, '_>, value: Value, others: Rest<Value>| -> Result<Value, Error> {
            // Maps merge into one lazy view whose size is the sum of its parts,
            // like a list; sequences are materialized within the limit.
            let all = std::iter::once(&value).chain(others.iter());
            if value.kind() == ValueKind::Map
                && others.iter().all(|other| other.kind() == ValueKind::Map)
            {
                let entries = all.map(|map| map.len().unwrap_or(0)).sum::<usize>();
                shared.reserve_items(entries, "chain")?;
                return filters::chain(state, value, others);
            }
            collect_items(&shared, all, "chain").map(Value::from)
        },
    );
}

/// Upper bound of `text.replace(from, to)` (or `replacen` with `count`) in bytes.
///
/// See [`limits::replacement_bound`]; this measures the strings.
pub(crate) fn replaced_length(text: &str, from: &str, to: &str, count: Option<usize>) -> usize {
    limits::replacement_bound(
        text.len(),
        text.chars().count(),
        from.len(),
        to.len(),
        count,
    )
}

/// Rejects printf field widths or precisions in `format` above the text allowance.
///
/// MiniJinja's `format` pads a conversion such as `%999999999s` to its width
/// before any size check. Fields are read the way its printf parser reads
/// them: `%%` is a literal percent sign; otherwise an optional `(key)`, flags
/// (`-+ #0`), width digits, and `.` with precision digits.
fn check_format_widths(budget: &RenderBudget, format: &str) -> Result<(), Error> {
    let allowance = budget.text_allowance();
    let bytes = format.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            index += 1;
            continue;
        }
        index += 1;
        if bytes.get(index) == Some(&b'%') {
            index += 1;
            continue;
        }
        if bytes.get(index) == Some(&b'(') {
            match bytes[index..].iter().position(|&byte| byte == b')') {
                Some(close) => index += close + 1,
                // MiniJinja rejects an unterminated key itself.
                None => return Ok(()),
            }
        }
        while index < bytes.len() && b"-+ #0".contains(&bytes[index]) {
            index += 1;
        }
        for _ in 0..2 {
            let start = index;
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                index += 1;
            }
            let digits = &format[start..index];
            if !digits.is_empty()
                && digits
                    .parse::<usize>()
                    .map_or(true, |width| width > allowance)
            {
                return Err(limit_error(format!(
                    "format field width {digits} exceeds the {allowance} bytes this render may still produce"
                )));
            }
            if index < bytes.len() && bytes[index] == b'.' {
                index += 1;
            } else {
                break;
            }
        }
    }
    Ok(())
}

// Reference cycles through namespaces.

/// Every namespace a render creates, so their attributes can be cleared when it ends.
///
/// Namespaces are MiniJinja's only mutable values, so they are the only way a
/// template can build a reference cycle (`{% set ns.me = ns %}`, or two
/// namespaces that hold each other). Values are reference counted, so a cycle
/// outlives the render and leaks everything it reaches, at most one render
/// budget per render, but without bound across renders. MiniJinja clears
/// macro closures itself; this registry does the same for namespaces.
#[derive(Debug, Default)]
pub(crate) struct NamespaceRegistry {
    /// Namespaces created by the render's `namespace()` calls, in order.
    namespaces: Mutex<Vec<Value>>,
}

impl NamespaceRegistry {
    /// Records a namespace created by the render.
    fn record(&self, namespace: Value) {
        if let Ok(mut namespaces) = self.namespaces.lock() {
            namespaces.push(namespace);
        }
    }

    /// Sets every attribute of every recorded namespace to `none`, breaking any cycle.
    ///
    /// A cycle through a namespace needs an attribute assigned after the
    /// namespace existed, and `{% set ns.name = value %}` only assigns
    /// identifier names, so clearing the identifier-named attributes breaks
    /// every cycle. Templates cannot assign attributes any other way, so the
    /// clearing runs as a generated template,
    /// `{% for n in namespaces %}{% set n.name = none %}...{% endfor %}`, over
    /// the union of attribute names. Failures are ignored: this runs from a
    /// destructor, possibly while a panic unwinds.
    pub(crate) fn break_cycles(&self) {
        let namespaces = match self.namespaces.lock() {
            Ok(mut namespaces) => std::mem::take(&mut *namespaces),
            Err(_) => return,
        };
        if namespaces.is_empty() {
            return;
        }
        let mut names = BTreeSet::new();
        for namespace in &namespaces {
            if let Ok(keys) = namespace.try_iter() {
                names.extend(
                    keys.filter_map(|key| key.as_str().map(str::to_owned))
                        .filter(|name| is_identifier(name)),
                );
            }
        }
        if names.is_empty() {
            return;
        }
        let mut source = String::from("{% for n in namespaces %}");
        for name in &names {
            source.push_str("{% set n.");
            source.push_str(name);
            source.push_str(" = none %}");
        }
        source.push_str("{% endfor %}");
        let environment = Environment::new();
        let _ = environment.render_str(
            &source,
            minijinja::context! { namespaces => Value::from(namespaces) },
        );
    }
}

/// Whether `name` is an ASCII identifier MiniJinja's lexer reads as one name.
fn is_identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|first| first == b'_' || first.is_ascii_alphabetic())
        && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
}

/// Clears a render's namespaces when dropped, including while a panic unwinds.
#[derive(Debug)]
pub(crate) struct CycleBreaker(pub(crate) Arc<NamespaceRegistry>);

impl Drop for CycleBreaker {
    /// Breaks every namespace cycle the render built.
    fn drop(&mut self) {
        self.0.break_cycles();
    }
}

/// Replaces `namespace()` with MiniJinja's own function plus registration in `registry`.
pub(crate) fn install_namespace_registry(
    environment: &mut Environment<'_>,
    registry: &Arc<NamespaceRegistry>,
) {
    let shared = Arc::clone(registry);
    environment.add_function(
        "namespace",
        move |defaults: Option<Value>| -> Result<Value, Error> {
            let namespace = functions::namespace(defaults)?;
            shared.record(namespace.clone());
            Ok(namespace)
        },
    );
}

// Bounded rendered output.

/// An [`io::Write`] target for the rendered prompt that refuses to grow past the output limit.
///
/// MiniJinja streams output through it, so a template that prints too much
/// fails at the write that crosses the limit instead of after building the
/// whole prompt.
#[derive(Debug)]
pub(crate) struct BoundedOutput {
    /// Rendered UTF-8 bytes so far, never more than `limit`.
    bytes: Vec<u8>,
    /// Largest permitted length in bytes.
    limit: usize,
    /// Whether a write was refused.
    exceeded: bool,
}

impl BoundedOutput {
    /// An empty output that holds at most `limit` bytes.
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            exceeded: false,
        }
    }

    /// Whether a write was refused.
    pub(crate) fn exceeded(&self) -> bool {
        self.exceeded
    }

    /// The rendered prompt.
    ///
    /// MiniJinja writes only `str` data, so the bytes are UTF-8; invalid bytes
    /// would be reported rather than silently replaced.
    pub(crate) fn into_string(self) -> Result<String, Error> {
        String::from_utf8(self.bytes).map_err(|error| {
            Error::new(
                ErrorKind::InvalidOperation,
                format!("rendered chat template is not UTF-8: {error}"),
            )
        })
    }
}

impl io::Write for BoundedOutput {
    /// Appends `buffer` if it fits, else records the overflow and fails.
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.bytes.len().saturating_add(buffer.len()) > self.limit {
            self.exceeded = true;
            return Err(io::Error::other("rendered chat template output limit"));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    /// Nothing is buffered.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "sandbox_tests.rs"]
mod tests;
