//! Source rewrite that routes size-amplifying syntax of untrusted templates through bounded filters.
//!
//! MiniJinja meters templates with fuel, one unit per instruction, but a
//! single `+`, `~`, or `*` can allocate without bound (`'ab' * n`, `s ~ s`,
//! `l + l`), a slice copies its string (`s[1:]`), a `*sequence` or
//! `**mapping` call argument copies its operand, and the engine has no hook
//! for any of them. It even folds constant operands while *compiling*:
//! `{{ 'ab' * 4611686018427387904 }}` panics with "capacity overflow" before
//! a single unit of fuel is spent. [`guard_operators`] therefore rewrites the
//! source before compilation so that each of them calls a bounded filter
//! from the `sandbox` module with the original semantics plus size checks:
//!
//! | Source                | Rewritten                                       |
//! |-----------------------|-------------------------------------------------|
//! | `a + b`               | `a|__bounded_add(b)`                            |
//! | `a ~ b * c`           | `a|__bounded_concat(b|__bounded_mul(c))`        |
//! | `a + b + c`           | `a|__bounded_add(b)|__bounded_add(c)`           |
//! | `a - b + c`           | `(a - b)|__bounded_add(c)`                      |
//! | `x ** y * z`          | `(x ** y)|__bounded_mul(z)`                     |
//! | `s[1:]`, `l[::-1]`    | `s|__bounded_slice(1, none, none)`, `l|__bounded_slice(none, none, -1)` |
//! | `s[1:].upper()`       | `(s|__bounded_slice(1, none, none)).upper()`    |
//! | `f(*xs, **kw)`        | `f(*xs|__bounded_splat, **kw|__bounded_splat)`  |
//!
//! A filter binds tighter than every binary operator and as tightly as
//! postfix syntax, so the left operand is parenthesized unless it already is
//! a unary-level expression (an operand, possibly negated, with postfix,
//! filter, or test syntax, or an earlier rewrite). Chains of the same
//! precedence become filter chains, which evaluate left to right exactly as
//! the left-associative operators did and add no parser nesting. A filter
//! call is never constant-folded, operands are evaluated in the original
//! order, and templates cannot assign to filter names, so `{% set %}` cannot
//! shadow the bounded operators. A rewrite keeps the whitespace around the
//! tag body, and the line breaks around a removed operator move inside the
//! filter call, so every tag keeps its line count and errors report the same
//! line numbers as the original template.
//!
//! A slice applies to everything before it in its postfix chain, unary minus
//! included (`-x[1:]` is `(-x)[1:]` in MiniJinja), which is exactly what a
//! filter on that text applies to. A filter cannot take further postfix
//! syntax, so the text is parenthesized before a following `.`, `[`, or `(`,
//! and a sliced parenthesis-free test argument is parenthesized so the
//! filter binds to the argument rather than to the test (`x is sameas (y|
//! __bounded_slice(...))` passes one argument, as `x is sameas y[1:]` did).
//! An omitted bound becomes `none`, the constant MiniJinja compiles for it.
//!
//! # Grammar
//!
//! Each expression and statement tag (found by the shared tag scanner, which
//! skips text, comments, raw sections, and string literals) is lexed and
//! parsed with MiniJinja 2.19's expression grammar. From loosest to tightest:
//! the conditional `a if b else c`; `or`; `and`; `not`; comparisons (`==`,
//! `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`); `+` and `-`; `~`; `*`, `/`,
//! `//`, and `%`; `**`; unary `-`; then a primary (literal, name, string run,
//! parenthesized expression or tuple, list, or map) followed by postfix
//! attribute, subscript, slice, and call syntax and finally filters
//! (`| name(args)`) and tests (`is [not] name arg`). Call arguments may be
//! positional, `name=value` (the value has no conditional), `*splat`, or
//! `**splat`. Statement tags are parsed far enough to find their expressions:
//! `for`, `if`, `elif`, `set` (including block `set` with a filter chain),
//! `with`, `macro`, `call`, `filter`, `do`, `autoescape`, `include`,
//! `import`, `from`, `extends`, and `block`; tags without expressions (`else`,
//! `endfor`, `break`, ...) are kept verbatim.
//!
//! # Failure policy and nesting limits
//!
//! A tag the parser cannot read is kept verbatim when it contains no `+`,
//! `~`, `*`, `**`, or `:` token: MiniJinja then rejects the syntax itself or
//! runs no guarded operation. The same applies to a variable named `not` used as an
//! operand (`x * not(y)`): MiniJinja reads it as a name there, but a
//! rewritten operand is re-read inside filter arguments, where `not` would
//! become the operator, so such a tag is refused rather than changed. A tag that does contain one is refused with a
//! [`GuardError`], so syntax this parser does not understand can never carry
//! an unbounded operator past the guard. An unknown statement keyword is kept
//! verbatim either way, because MiniJinja rejects it while parsing, before
//! anything is folded or evaluated.
//!
//! MiniJinja guards its parser recursion (150 levels) but not chains of unary
//! operators (`- - - x`, `not not x`), which it parses recursively, nor the
//! depth of the syntax tree its code generator and destructor then recurse
//! over (`a - b - c - ...` is a left-deep tree). A template with a few hundred
//! thousand such operators therefore overflows the native stack, which aborts
//! the process. This rewrite refuses expressions nested deeper than
//! [`MAX_NESTING`] levels and chains (unary, binary, comparison, conditional,
//! postfix, or filter) longer than [`MAX_CHAIN`], far beyond any published
//! template. Comma-separated lists (list, tuple, and map items, call
//! arguments, macro parameters) are flat in MiniJinja's parser, code
//! generator, and destructor, so their length is not limited.

use std::borrow::Cow;

use super::{TagKind, sandbox, scan::char_boundary_at_most, statement_parts, try_rewrite_tags};

// Documents itself with `//!`; an outer `///` here would change its link scope.
mod lexer; // Byte-level tag-body lexer mirroring MiniJinja (Verus-verified).

use lexer::{Kind, LexError, Span, lex_spans};

/// Deepest parenthesized or bracketed nesting accepted; MiniJinja's own parser stops at 150.
pub(crate) const MAX_NESTING: usize = 200;
/// Longest chain of unary, binary, postfix, or filter syntax accepted in one expression.
pub(crate) const MAX_CHAIN: usize = 1024;

/// A template the guard refuses, with the offending tag and the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GuardError {
    /// Byte offset of the tag in the source the guard received.
    pub(crate) offset: usize,
    /// The tag text, shortened to at most 80 bytes for messages.
    pub(crate) tag: String,
    /// Why the tag was refused.
    pub(crate) reason: String,
}

impl std::fmt::Display for GuardError {
    /// Names the tag, its offset, and the reason.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "template tag `{}` at byte {} cannot be bounded: {}",
            self.tag, self.offset, self.reason
        )
    }
}

/// Rewrites every binary `+`, `~`, and `*` in `source` into a bounded filter call.
///
/// Returns the source unchanged (borrowed) when it has none.
///
/// # Errors
///
/// Returns a [`GuardError`] for a tag that uses a guarded operator in syntax
/// the parser cannot read, or that nests or chains beyond the limits above.
pub(crate) fn guard_operators(source: &str) -> Result<Cow<'_, str>, GuardError> {
    try_rewrite_tags(source, |kind, offset, tag| {
        guard_tag(kind, tag).map_err(|reason| GuardError {
            offset,
            tag: shorten(tag),
            reason,
        })
    })
}

/// Counts the line feeds in `text`, the line breaks MiniJinja counts for error locations.
fn newline_count(text: &str) -> usize {
    text.bytes().filter(|&byte| byte == b'\n').count()
}

/// Shortens a tag to at most 80 bytes on a character boundary for an error message.
fn shorten(tag: &str) -> String {
    const LIMIT: usize = 80;
    if tag.len() <= LIMIT {
        return tag.to_owned();
    }
    // Verified: the largest character boundary at or below the limit.
    let end = char_boundary_at_most(tag, LIMIT);
    format!("{}...", &tag[..end])
}

/// Rewrites one tag (delimiters included), or returns `None` to keep it verbatim.
fn guard_tag(kind: TagKind, tag: &str) -> Result<Option<String>, String> {
    let (open_marker, body, close_marker) = statement_parts(tag);
    let tokens = match lex(body) {
        Ok(tokens) => tokens,
        // MiniJinja's lexer fails on the same input, unless it only skipped a
        // guarded byte inside text this lexer misread; refuse those.
        Err(reason) if body.contains(['+', '~', '*', ':']) => return Err(reason),
        Err(_) => return Ok(None),
    };
    let mut parser = Parser {
        source: body,
        tokens: &tokens,
        position: 0,
        depth: 0,
        rewrites: 0,
    };
    let parsed = match kind {
        TagKind::Expression => parser.expression_tag(),
        TagKind::Statement => parser.statement_tag(),
    };
    match parsed {
        Ok(Some(text)) if parser.rewrites > 0 => {
            // Keep the whitespace around the body, newlines included, so
            // every later line keeps its number in error messages.
            let inner = &tag[2 + open_marker.len()..tag.len() - 2 - close_marker.len()];
            let leading = &inner[..inner.len() - inner.trim_start().len()];
            let trailing = &inner[inner.trim_end().len()..];
            Ok(Some(format!(
                "{}{open_marker}{leading}{text}{trailing}{close_marker}{}",
                &tag[..2],
                &tag[tag.len() - 2..]
            )))
        }
        Ok(_) => Ok(None),
        Err(Refusal::Limit(reason)) => Err(reason),
        Err(Refusal::Syntax(reason)) if tokens.iter().any(Token::needs_guard) => Err(reason),
        Err(Refusal::Syntax(_)) => Ok(None),
    }
}

// Lexing.

/// One token of a tag body with its byte range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Token<'a> {
    /// What the token is.
    kind: Kind,
    /// The token's source text.
    text: &'a str,
    /// Byte offset of the first byte in the tag body.
    start: usize,
    /// Byte offset just past the last byte in the tag body.
    end: usize,
}

impl Token<'_> {
    /// Whether this is a `+`, `~`, or `*` symbol (an operator or a `*` splat).
    fn is_guarded(&self) -> bool {
        self.kind == Kind::Symbol && matches!(self.text, "+" | "~" | "*")
    }

    /// Whether a tag holding this token may need a rewrite: a guarded operator,
    /// a `**` splat (or power), or the `:` of a slice (or map entry).
    ///
    /// A tag the parser cannot read is refused if it holds one of these, so
    /// syntax the parser misreads can never carry an unbounded operation past
    /// the guard.
    fn needs_guard(&self) -> bool {
        self.is_guarded() || (self.kind == Kind::Symbol && matches!(self.text, "**" | ":"))
    }
}

/// Splits a tag body into tokens exactly as MiniJinja 2.19's block lexer does.
///
/// ASCII whitespace separates tokens; strings honour backslash escapes;
/// numbers follow MiniJinja's state machine (digits, `_`, one `.` fraction,
/// an exponent with an optional sign, radix prefixes); identifiers are ASCII
/// (MiniJinja is built without its `unicode` feature). The spans come from
/// the verified [`lex_spans`], which partitions the body into tokens of their
/// kinds' lexical classes; every span boundary is an ASCII byte or the end,
/// so each token's text is a valid `str` slice.
fn lex(body: &str) -> Result<Vec<Token<'_>>, String> {
    let spans = lex_spans(body.as_bytes()).map_err(|error| match error {
        LexError::UnterminatedString { start } => {
            format!("unterminated string literal at byte {start}")
        }
        LexError::UnexpectedCharacter { start } => {
            format!("unexpected character at byte {start}")
        }
    })?;
    Ok(spans
        .into_iter()
        .map(|Span { kind, start, end }| Token {
            kind,
            text: &body[start..end],
            start,
            end,
        })
        .collect())
}

// Parsing.

/// An expression's source span and (possibly rewritten) text.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Fragment {
    /// Byte offset of the expression's first token.
    start: usize,
    /// Byte offset just past its last token.
    end: usize,
    /// The expression's text with every guarded operator rewritten.
    text: String,
    /// Whether a trailing `| filter` would bind to the whole expression.
    unary: bool,
    /// Whether the expression is a bare variable name (a keyword-argument key).
    name: bool,
    /// Whether the text ends in a filter added by a slice rewrite, which must
    /// be parenthesized before postfix syntax or as a test argument.
    open_filter: bool,
}

/// Rebuilds a construct's text from the source between its child expressions.
struct Splice<'a> {
    /// The tag body.
    source: &'a str,
    /// Source offset up to which text has been copied.
    cursor: usize,
    /// Text built so far.
    text: String,
}

impl<'a> Splice<'a> {
    /// Starts a construct at `start`.
    fn new(source: &'a str, start: usize) -> Self {
        Self {
            source,
            cursor: start,
            text: String::new(),
        }
    }

    /// Copies source up to `child` and then appends the child's rewritten text.
    fn child(&mut self, child: &Fragment) {
        self.text.push_str(&self.source[self.cursor..child.start]);
        self.text.push_str(&child.text);
        self.cursor = child.end;
    }

    /// Copies source up to `end` and moves the text built so far out, leaving the splice at `end`.
    fn take(&mut self, end: usize) -> String {
        self.text.push_str(&self.source[self.cursor..end]);
        self.cursor = end;
        std::mem::take(&mut self.text)
    }

    /// Replaces the text built so far with `text` and resumes copying source at `cursor`.
    fn restart(&mut self, text: String, cursor: usize) {
        self.text = text;
        self.cursor = cursor;
    }

    /// Copies the remaining source up to `end` and returns the construct's text.
    fn finish(mut self, end: usize) -> String {
        self.text.push_str(&self.source[self.cursor..end]);
        self.text
    }
}

/// Binary operators of one precedence level and, for guarded ones, their filter.
type Level = &'static [(&'static str, Option<&'static str>)];

/// `or`.
const OR: Level = &[("or", None)];
/// `and`.
const AND: Level = &[("and", None)];
/// `+` (guarded) and `-`.
const SUM: Level = &[("+", Some(sandbox::ADD_FILTER)), ("-", None)];
/// `~` (guarded).
const CONCAT: Level = &[("~", Some(sandbox::CONCAT_FILTER))];
/// `*` (guarded), `/`, `//`, and `%`.
const PRODUCT: Level = &[
    ("*", Some(sandbox::MULTIPLY_FILTER)),
    ("/", None),
    ("//", None),
    ("%", None),
];
/// `**`.
const POWER: Level = &[("**", None)];
/// The filter call appended to a splatted call argument (see `sandbox::SPLAT_FILTER`).
///
/// An argument that already ends in it is left alone, so rewriting is idempotent.
const SPLAT_SUFFIX: &str = "|__bounded_splat";
/// Comparison symbols; `in` and `not in` are handled by name.
const COMPARISONS: [&str; 6] = ["==", "!=", "<", "<=", ">", ">="];
/// Names that are constants rather than variables.
const CONSTANTS: [&str; 6] = ["true", "True", "false", "False", "none", "None"];

/// A recursive-descent parser over one tag body that mirrors MiniJinja's grammar.
struct Parser<'a> {
    /// The tag body the tokens index into.
    source: &'a str,
    /// The body's tokens.
    tokens: &'a [Token<'a>],
    /// Index of the next unread token.
    position: usize,
    /// Current expression and primary nesting depth.
    depth: usize,
    /// Number of operators rewritten so far.
    rewrites: usize,
}

/// Why a tag could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Refusal {
    /// Syntax this parser does not read; refused only if the tag has a guarded operator.
    Syntax(String),
    /// Nesting or a chain past the limits; always refused, to protect MiniJinja's stack.
    Limit(String),
}

impl From<String> for Refusal {
    /// Treats a plain message as a syntax refusal.
    fn from(reason: String) -> Self {
        Self::Syntax(reason)
    }
}

/// A parse step: the parsed construct or the reason it was refused.
type Parse<T> = Result<T, Refusal>;

impl<'a> Parser<'a> {
    // Token access.

    /// The next unread token.
    fn peek(&self) -> Option<Token<'a>> {
        self.tokens.get(self.position).copied()
    }

    /// The token `offset` places after the next one.
    fn peek_at(&self, offset: usize) -> Option<Token<'a>> {
        self.tokens.get(self.position + offset).copied()
    }

    /// Whether the next token is the symbol `symbol`.
    fn at_symbol(&self, symbol: &str) -> bool {
        self.peek()
            .is_some_and(|token| token.kind == Kind::Symbol && token.text == symbol)
    }

    /// Whether the next token is the name `name`.
    fn at_name(&self, name: &str) -> bool {
        self.peek()
            .is_some_and(|token| token.kind == Kind::Name && token.text == name)
    }

    /// Whether every token has been read.
    fn at_end(&self) -> bool {
        self.position == self.tokens.len()
    }

    /// Consumes the next token if it is the symbol `symbol`.
    fn eat_symbol(&mut self, symbol: &str) -> bool {
        let found = self.at_symbol(symbol);
        if found {
            self.position += 1;
        }
        found
    }

    /// Consumes the next token if it is the name `name`.
    fn eat_name(&mut self, name: &str) -> bool {
        let found = self.at_name(name);
        if found {
            self.position += 1;
        }
        found
    }

    /// Consumes the symbol `symbol` or reports what was found instead.
    fn expect_symbol(&mut self, symbol: &str) -> Parse<Token<'a>> {
        match self.peek() {
            Some(token) if token.kind == Kind::Symbol && token.text == symbol => {
                self.position += 1;
                Ok(token)
            }
            other => Err(self.unexpected(other, &format!("`{symbol}`")).into()),
        }
    }

    /// Consumes a name token or reports what was found instead.
    fn expect_name(&mut self) -> Parse<Token<'a>> {
        match self.peek() {
            Some(token) if token.kind == Kind::Name => {
                self.position += 1;
                Ok(token)
            }
            other => Err(self.unexpected(other, "a name").into()),
        }
    }

    /// Describes an unexpected token (or the end of the tag).
    fn unexpected(&self, found: Option<Token<'a>>, expected: &str) -> String {
        match found {
            Some(token) => format!("expected {expected}, found `{}`", token.text),
            None => format!("expected {expected}, found the end of the tag"),
        }
    }

    /// Requires that the whole body was read.
    fn expect_end(&self) -> Parse<()> {
        match self.peek() {
            None => Ok(()),
            found => Err(self.unexpected(found, "the end of the tag").into()),
        }
    }

    /// Byte offset of the next token, or of the end of the body.
    fn offset(&self) -> usize {
        self.peek().map_or(self.source.len(), |token| token.start)
    }

    /// Byte offset just past the last consumed token.
    fn last_end(&self) -> usize {
        self.position
            .checked_sub(1)
            .map_or(0, |index| self.tokens[index].end)
    }

    /// An unchanged fragment covering `start..end`.
    fn verbatim(&self, start: usize, end: usize, unary: bool) -> Fragment {
        Fragment {
            start,
            end,
            text: self.source[start..end].to_owned(),
            unary,
            name: false,
            open_filter: false,
        }
    }

    /// Enters one nesting level (an expression or a primary), refusing excessive depth.
    fn enter(&mut self) -> Parse<()> {
        self.depth += 1;
        if self.depth > MAX_NESTING {
            return Err(Refusal::Limit(format!(
                "expressions nest more than {MAX_NESTING} levels"
            )));
        }
        Ok(())
    }

    /// Refuses a chain that has grown past [`MAX_CHAIN`] links.
    fn check_chain(length: usize) -> Parse<()> {
        if length > MAX_CHAIN {
            return Err(Refusal::Limit(format!(
                "an expression chains more than {MAX_CHAIN} operators"
            )));
        }
        Ok(())
    }

    // Tags.

    /// Parses an expression tag: one expression and nothing else.
    fn expression_tag(&mut self) -> Parse<Option<String>> {
        let expression = self.expression()?;
        self.expect_end()?;
        Ok(Some(expression.text))
    }

    /// Parses a statement tag far enough to rewrite its expressions.
    ///
    /// Returns `None` for statements without expressions and for unknown
    /// keywords, which MiniJinja rejects while parsing.
    fn statement_tag(&mut self) -> Parse<Option<String>> {
        let Some(keyword) = self.peek().filter(|token| token.kind == Kind::Name) else {
            return Err(self.unexpected(self.peek(), "a statement keyword").into());
        };
        self.position += 1;
        let mut splice = Splice::new(self.source, 0);
        match keyword.text {
            "for" => {
                self.assignment_target(false)?;
                if !self.eat_name("in") {
                    return Err(self.unexpected(self.peek(), "`in`").into());
                }
                splice.child(&self.expression_without_conditional()?);
                if self.eat_name("if") {
                    splice.child(&self.expression()?);
                }
                self.eat_name("recursive");
            }
            "if" | "elif" => splice.child(&self.expression_without_conditional()?),
            "set" => self.set_statement(&mut splice)?,
            "with" => {
                while !self.at_end() {
                    if self.position > 1 {
                        self.expect_symbol(",")?;
                    }
                    if self.eat_symbol("(") {
                        self.assignment_target(false)?;
                        self.expect_symbol(")")?;
                    } else {
                        self.expect_name()?;
                    }
                    self.expect_symbol("=")?;
                    splice.child(&self.expression()?);
                }
            }
            "macro" => {
                self.expect_name()?;
                self.expect_symbol("(")?;
                self.macro_parameters(&mut splice)?;
            }
            "call" => {
                if self.eat_symbol("(") {
                    self.macro_parameters(&mut splice)?;
                }
                splice.child(&self.expression()?);
            }
            "filter" => self.filter_chain(&mut splice)?,
            "do" | "autoescape" | "extends" => splice.child(&self.expression()?),
            "include" => {
                splice.child(&self.expression()?);
                self.context_marker();
                if self.eat_name("ignore") {
                    if !self.eat_name("missing") {
                        return Err(self.unexpected(self.peek(), "`missing`").into());
                    }
                    self.context_marker();
                }
            }
            "import" => {
                splice.child(&self.expression()?);
                if !self.eat_name("as") {
                    return Err(self.unexpected(self.peek(), "`as`").into());
                }
                splice.child(&self.expression()?);
                self.context_marker();
            }
            "from" => {
                splice.child(&self.expression()?);
                if !self.eat_name("import") {
                    return Err(self.unexpected(self.peek(), "`import`").into());
                }
                while !self.at_end() && !self.context_marker() {
                    self.eat_symbol(",");
                    self.expect_name()?;
                    if self.eat_name("as") {
                        self.expect_name()?;
                    }
                }
            }
            _ => return Ok(None),
        }
        self.expect_end()?;
        Ok(Some(splice.finish(self.source.len())))
    }

    /// Parses `set target = value[, value...]` or a block `set target [| filters]`.
    fn set_statement(&mut self, splice: &mut Splice<'_>) -> Parse<()> {
        self.assignment_target(true)?;
        if self.at_end() {
            return Ok(());
        }
        if self.eat_symbol("|") {
            self.position -= 1;
            return self.filter_chain(splice);
        }
        self.expect_symbol("=")?;
        splice.child(&self.expression()?);
        if self.eat_symbol(",") {
            while !self.at_end() {
                splice.child(&self.expression()?);
                if !self.eat_symbol(",") {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Skips an assignment target: names (dotted for `set`), commas, and parenthesized groups.
    fn assignment_target(&mut self, dotted: bool) -> Parse<()> {
        let mut first = true;
        loop {
            if !first && !self.eat_symbol(",") {
                return Ok(());
            }
            first = false;
            if self.at_symbol(")") || self.at_end() || self.at_name("in") {
                return Ok(());
            }
            if self.eat_symbol("(") {
                self.assignment_target(dotted)?;
                self.expect_symbol(")")?;
            } else {
                self.expect_name()?;
                while dotted && self.eat_symbol(".") {
                    self.expect_name()?;
                }
            }
            if !self.at_symbol(",") {
                return Ok(());
            }
        }
    }

    /// Parses macro or call-block parameters after `(`: names with optional defaults.
    ///
    /// Parameter lists, like every comma-separated list below, are flat: MiniJinja
    /// parses, compiles, and drops them iteratively, so their length is not capped.
    fn macro_parameters(&mut self, splice: &mut Splice<'_>) -> Parse<()> {
        let mut first = true;
        loop {
            if self.eat_symbol(")") {
                return Ok(());
            }
            if !first {
                self.expect_symbol(",")?;
                if self.eat_symbol(")") {
                    return Ok(());
                }
            }
            self.expect_name()?;
            if self.eat_symbol("=") {
                splice.child(&self.expression()?);
            }
            first = false;
        }
    }

    /// Parses a `{% filter %}` or block-`set` chain: `name(args) | name(args) ...`.
    fn filter_chain(&mut self, splice: &mut Splice<'_>) -> Parse<()> {
        let mut count = 0;
        while !self.at_end() {
            if count > 0 || self.at_symbol("|") {
                self.expect_symbol("|")?;
            }
            self.dotted_name()?;
            if self.at_symbol("(") {
                for argument in self.call_arguments()? {
                    splice.child(&argument);
                }
            }
            count += 1;
            Self::check_chain(count)?;
        }
        Ok(())
    }

    /// Consumes `with context` or `without context` if present.
    fn context_marker(&mut self) -> bool {
        let marker = self.at_name("with") || self.at_name("without");
        let context = self
            .peek_at(1)
            .is_some_and(|token| token.kind == Kind::Name && token.text == "context");
        if marker && context {
            self.position += 2;
        }
        marker && context
    }

    // Expressions, loosest first.

    /// Parses a full expression, including the conditional form.
    fn expression(&mut self) -> Parse<Fragment> {
        self.enter()?;
        let expression = self.conditional();
        self.depth -= 1;
        expression
    }

    /// Parses an expression without a top-level conditional (`parse_expr_noif`).
    fn expression_without_conditional(&mut self) -> Parse<Fragment> {
        self.binary(Self::conjunction, OR)
    }

    /// Parses an `and` chain over `not` expressions.
    fn conjunction(&mut self) -> Parse<Fragment> {
        self.binary(Self::not, AND)
    }

    /// Parses `a if b else c`, chained left to right like MiniJinja.
    fn conditional(&mut self) -> Parse<Fragment> {
        let first = self.expression_without_conditional()?;
        if !self.at_name("if") {
            return Ok(first);
        }
        let start = first.start;
        let mut splice = Splice::new(self.source, start);
        splice.child(&first);
        let mut links = 0;
        while self.eat_name("if") {
            splice.child(&self.expression_without_conditional()?);
            if self.eat_name("else") {
                // MiniJinja recurses here without a depth guard; count the level.
                self.enter()?;
                let alternative = self.conditional();
                self.depth -= 1;
                splice.child(&alternative?);
            }
            links += 1;
            Self::check_chain(links)?;
        }
        let end = self.last_end();
        Ok(Fragment {
            start,
            end,
            text: splice.finish(end),
            unary: false,
            name: false,
            open_filter: false,
        })
    }

    /// Parses a left-associative chain of one precedence level over `operand`.
    ///
    /// Guarded operators become filter calls on the accumulated left side
    /// (parenthesized unless unary-level); other operators keep their source.
    fn binary(
        &mut self,
        operand: fn(&mut Self) -> Parse<Fragment>,
        level: Level,
    ) -> Parse<Fragment> {
        let first = operand(self)?;
        let start = first.start;
        let mut text = first.text;
        let mut unary = first.unary;
        let mut name = first.name;
        let mut end = first.end;
        let mut links = 0;
        while let Some(token) = self.peek() {
            let Some(&(_, filter)) = level.iter().find(|(symbol, _)| {
                token.text == *symbol && matches!(token.kind, Kind::Symbol | Kind::Name)
            }) else {
                break;
            };
            self.position += 1;
            let right = operand(self)?;
            links += 1;
            Self::check_chain(links)?;
            match filter {
                Some(filter) => {
                    let left = if unary { text } else { format!("({text})") };
                    // The operator and the whitespace around it disappear;
                    // their newlines move inside the filter call so later
                    // lines keep their numbers.
                    let newlines = "\n".repeat(newline_count(&self.source[end..right.start]));
                    text = format!("{left}|{filter}({newlines}{})", right.text);
                    unary = true;
                    self.rewrites += 1;
                }
                None => {
                    text.push_str(&self.source[end..right.start]);
                    text.push_str(&right.text);
                    unary = false;
                }
            }
            name = false;
            end = right.end;
        }
        Ok(Fragment {
            start,
            end,
            text,
            unary,
            name,
            open_filter: false,
        })
    }

    /// Parses `not ...`; MiniJinja reads consecutive `not`s recursively, so they are capped.
    fn not(&mut self) -> Parse<Fragment> {
        let start = self.offset();
        let mut count = 0;
        while self.eat_name("not") {
            count += 1;
            Self::check_chain(count)?;
        }
        let operand = self.comparison()?;
        if count == 0 {
            return Ok(operand);
        }
        let mut splice = Splice::new(self.source, start);
        splice.child(&operand);
        Ok(Fragment {
            start,
            end: operand.end,
            text: splice.finish(operand.end),
            unary: false,
            name: false,
            open_filter: false,
        })
    }

    /// Parses comparisons, including `in` and `not in`, over sums.
    fn comparison(&mut self) -> Parse<Fragment> {
        let first = self.binary(Self::concatenation, SUM)?;
        let start = first.start;
        let mut splice = Splice::new(self.source, start);
        splice.child(&first);
        let mut end = first.end;
        let mut links = 0;
        loop {
            if self.eat_name("not") {
                if !self.eat_name("in") {
                    return Err(self.unexpected(self.peek(), "`in`").into());
                }
            } else if !(self.eat_name("in")
                || COMPARISONS.iter().any(|symbol| self.eat_symbol(symbol)))
            {
                break;
            }
            let right = self.binary(Self::concatenation, SUM)?;
            splice.child(&right);
            end = right.end;
            links += 1;
            Self::check_chain(links)?;
        }
        if links == 0 {
            return Ok(first);
        }
        Ok(Fragment {
            start,
            end,
            text: splice.finish(end),
            unary: false,
            name: false,
            open_filter: false,
        })
    }

    /// Parses a `~` chain over products.
    fn concatenation(&mut self) -> Parse<Fragment> {
        self.binary(Self::product, CONCAT)
    }

    /// Parses a `*`, `/`, `//`, `%` chain over powers.
    fn product(&mut self) -> Parse<Fragment> {
        self.binary(Self::power, PRODUCT)
    }

    /// Parses a `**` chain over unary expressions.
    fn power(&mut self) -> Parse<Fragment> {
        self.binary(Self::unary, POWER)
    }

    /// Parses unary minus, a primary, postfix syntax, then filters and tests.
    fn unary(&mut self) -> Parse<Fragment> {
        let operand = self.unary_postfix()?;
        let start = operand.start;
        let mut splice = Splice::new(self.source, start);
        splice.child(&operand);
        let mut links = 0;
        let mut name = operand.name;
        loop {
            if self.eat_symbol("|") {
                self.dotted_name()?;
                if self.at_symbol("(") {
                    for argument in self.call_arguments()? {
                        splice.child(&argument);
                    }
                }
            } else if self.eat_name("is") {
                self.eat_name("not");
                self.dotted_name()?;
                if self.at_symbol("(") {
                    for argument in self.call_arguments()? {
                        splice.child(&argument);
                    }
                } else if self.starts_test_argument() {
                    let argument = self.unary_postfix()?;
                    if argument.open_filter {
                        // `x is t y|f` would filter the test's result;
                        // `x is t (y|f)` passes the one argument instead.
                        let text = format!("({})", argument.text);
                        splice.child(&Fragment { text, ..argument });
                    } else {
                        splice.child(&argument);
                    }
                }
            } else {
                break;
            }
            name = false;
            links += 1;
            Self::check_chain(links)?;
        }
        let end = self.last_end();
        Ok(Fragment {
            start,
            end,
            text: splice.finish(end),
            unary: true,
            name,
            open_filter: false,
        })
    }

    /// Whether the next token starts MiniJinja's parenthesis-free test argument.
    fn starts_test_argument(&self) -> bool {
        let Some(token) = self.peek() else {
            return false;
        };
        match token.kind {
            Kind::Name => !matches!(token.text, "and" | "or" | "else" | "is"),
            Kind::String | Kind::Integer | Kind::Float => true,
            Kind::Symbol => matches!(token.text, "+" | "-" | "[" | "{"),
        }
    }

    /// Parses leading minus signs, a primary, and postfix syntax (no filters).
    ///
    /// Slices are rewritten into the bounded slice filter (see
    /// [`Self::subscript`]); a filter cannot take further postfix syntax, so
    /// the text so far is parenthesized before the next `.`, `[`, or `(`.
    fn unary_postfix(&mut self) -> Parse<Fragment> {
        let start = self.offset();
        let mut minus = 0;
        while self.eat_symbol("-") {
            minus += 1;
            Self::check_chain(minus)?;
        }
        let primary = self.primary()?;
        let mut splice = Splice::new(self.source, start);
        splice.child(&primary);
        let mut name = primary.name && minus == 0;
        let mut open_filter = false;
        let mut links = 0;
        while let Some(token) = self
            .peek()
            .filter(|token| token.kind == Kind::Symbol && matches!(token.text, "." | "[" | "("))
        {
            if open_filter {
                let text = splice.take(token.start);
                splice.restart(format!("({text})"), token.start);
                open_filter = false;
            }
            match token.text {
                "." => {
                    self.position += 1;
                    match self.peek() {
                        Some(token) if matches!(token.kind, Kind::Name | Kind::Integer) => {
                            self.position += 1;
                        }
                        other => {
                            return Err(self.unexpected(other, "an attribute or index").into());
                        }
                    }
                }
                "[" => {
                    self.position += 1;
                    open_filter = self.subscript(&mut splice, token)?;
                }
                _ => {
                    for argument in self.call_arguments()? {
                        splice.child(&argument);
                    }
                }
            }
            name = false;
            links += 1;
            Self::check_chain(links)?;
        }
        let end = self.last_end();
        Ok(Fragment {
            start,
            end,
            text: splice.finish(end),
            unary: true,
            name,
            open_filter,
        })
    }

    /// Parses a subscript or slice after its `[` token `open`.
    ///
    /// A subscript `value[index]` keeps its source. A slice
    /// `value[start:stop:step]` becomes
    /// `value|__bounded_slice(start, stop, step)` on the text built so far,
    /// with `none` for an omitted bound (MiniJinja compiles an omitted bound
    /// as the constant `none`, so this is exact). Slicing a string copies it,
    /// and the engine has no hook for that. Line breaks inside the brackets
    /// move inside the call. Returns whether the text now ends in the filter.
    fn subscript(&mut self, splice: &mut Splice<'_>, open: Token<'a>) -> Parse<bool> {
        let mut bounds: [Option<Fragment>; 3] = [None, None, None];
        if !self.at_symbol(":") {
            bounds[0] = Some(self.expression()?);
        }
        let slice = self.eat_symbol(":");
        if slice {
            if !self.at_symbol("]") && !self.at_symbol(":") {
                bounds[1] = Some(self.expression()?);
            }
            if self.eat_symbol(":") && !self.at_symbol("]") {
                bounds[2] = Some(self.expression()?);
            }
        }
        let close = self.expect_symbol("]")?;
        if !slice {
            if let Some(index) = &bounds[0] {
                splice.child(index);
            }
            return Ok(false);
        }
        let operand = splice.take(open.start);
        let mut inner_newlines = 0;
        let arguments = bounds
            .iter()
            .map(|bound| match bound {
                Some(bound) => {
                    inner_newlines += newline_count(&self.source[bound.start..bound.end]);
                    bound.text.as_str()
                }
                None => "none",
            })
            .collect::<Vec<_>>()
            .join(", ");
        let newlines =
            "\n".repeat(newline_count(&self.source[open.start..close.end]) - inner_newlines);
        splice.restart(
            format!("{operand}|{}({arguments}{newlines})", sandbox::SLICE_FILTER),
            close.end,
        );
        self.rewrites += 1;
        Ok(true)
    }

    /// Parses a filter or test name: `name` or `name.name...`.
    fn dotted_name(&mut self) -> Parse<()> {
        self.expect_name()?;
        while self.at_symbol(".")
            && self
                .peek_at(1)
                .is_some_and(|token| token.kind == Kind::Name)
        {
            self.position += 2;
        }
        Ok(())
    }

    /// Parses `(args)`, returning every argument expression (keys and splat markers excluded).
    fn call_arguments(&mut self) -> Parse<Vec<Fragment>> {
        self.expect_symbol("(")?;
        let mut arguments = Vec::new();
        loop {
            if self.eat_symbol(")") {
                return Ok(arguments);
            }
            if !arguments.is_empty() {
                self.expect_symbol(",")?;
                if self.eat_symbol(")") {
                    return Ok(arguments);
                }
            }
            // `**mapping` and `*sequence` splats: these stars are not products.
            let splat = self.eat_symbol("**") || self.eat_symbol("*");
            let argument = self.expression()?;
            if !splat && argument.name && self.eat_symbol("=") {
                arguments.push(self.expression_without_conditional()?);
            } else if splat && !argument.text.ends_with(SPLAT_SUFFIX) {
                // MiniJinja copies a splatted operand into the call's
                // arguments; the bounded splat filter charges that copy.
                let operand = if argument.unary {
                    argument.text
                } else {
                    format!("({})", argument.text)
                };
                self.rewrites += 1;
                arguments.push(Fragment {
                    text: format!("{operand}{SPLAT_SUFFIX}"),
                    unary: true,
                    name: false,
                    open_filter: false,
                    ..argument
                });
            } else {
                arguments.push(argument);
            }
        }
    }

    /// Parses a primary: a literal, name, string run, group or tuple, list, or map.
    fn primary(&mut self) -> Parse<Fragment> {
        self.enter()?;
        let primary = self.primary_inner();
        self.depth -= 1;
        primary
    }

    /// [`Self::primary`] inside one nesting level.
    fn primary_inner(&mut self) -> Parse<Fragment> {
        let Some(token) = self.peek() else {
            return Err(self.unexpected(None, "an expression").into());
        };
        self.position += 1;
        let start = token.start;
        match (token.kind, token.text) {
            // MiniJinja reads `not` in operand position (`x * not(y)`) as a
            // variable, but a rewritten operand is re-read inside filter
            // arguments or parentheses, where `not` is the operator; such a
            // tag cannot be rewritten faithfully.
            (Kind::Name, "not") => Err(Refusal::Syntax(
                "a variable named `not` cannot be an operand of a bounded operator".to_owned(),
            )),
            (Kind::Name, text) => {
                let mut fragment = self.verbatim(start, token.end, true);
                fragment.name = !CONSTANTS.contains(&text);
                Ok(fragment)
            }
            (Kind::String, _) => {
                // Adjacent string literals form one constant.
                while self.peek().is_some_and(|token| token.kind == Kind::String) {
                    self.position += 1;
                }
                Ok(self.verbatim(start, self.last_end(), true))
            }
            (Kind::Integer | Kind::Float, _) => Ok(self.verbatim(start, token.end, true)),
            (Kind::Symbol, "(") => self.group(start),
            (Kind::Symbol, "[") => self.items(start, "]", false),
            (Kind::Symbol, "{") => self.items(start, "}", true),
            _ => Err(format!("expected an expression, found `{}`", token.text).into()),
        }
    }

    /// Parses `()`, `(expr)`, or a tuple `(a, b, ...)` after its `(`.
    fn group(&mut self, start: usize) -> Parse<Fragment> {
        if self.eat_symbol(")") {
            return Ok(self.verbatim(start, self.last_end(), true));
        }
        let first = self.expression()?;
        let mut splice = Splice::new(self.source, start);
        splice.child(&first);
        let mut name = first.name;
        if self.at_symbol(",") {
            name = false;
            loop {
                if self.eat_symbol(")") {
                    break;
                }
                self.expect_symbol(",")?;
                if self.eat_symbol(")") {
                    break;
                }
                splice.child(&self.expression()?);
            }
        } else {
            self.expect_symbol(")")?;
        }
        let end = self.last_end();
        Ok(Fragment {
            start,
            end,
            text: splice.finish(end),
            unary: true,
            name,
            open_filter: false,
        })
    }

    /// Parses list items up to `]`, or map entries `key: value` up to `}`.
    fn items(&mut self, start: usize, close: &str, map: bool) -> Parse<Fragment> {
        let mut splice = Splice::new(self.source, start);
        let mut first = true;
        loop {
            if self.eat_symbol(close) {
                break;
            }
            if !first {
                self.expect_symbol(",")?;
                if self.eat_symbol(close) {
                    break;
                }
            }
            splice.child(&self.expression()?);
            if map {
                self.expect_symbol(":")?;
                splice.child(&self.expression()?);
            }
            first = false;
        }
        let end = self.last_end();
        Ok(Fragment {
            start,
            end,
            text: splice.finish(end),
            unary: true,
            name: false,
            open_filter: false,
        })
    }
}

#[cfg(test)]
#[path = "guard_tests.rs"]
mod tests;
