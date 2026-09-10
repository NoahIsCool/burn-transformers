"""Parse a deliberately strict subset of Rust 1.92 compiler dep-info.

This is compiler-input evidence, not a Make interpreter, source classifier or
coverage exclusion proof. It performs no filesystem access, path normalization,
environment expansion or shell evaluation. A caller must bind the bytes to the
actual compiler invocation, working directory, source inventory and toolchain.
Dep-info does not record arbitrary untracked filesystem I/O by procedural macros.

The producer is pinned to rustc 1.92.0, commit
ded5c06cf21d2b93bffd5d884aa6e96934ee4234, functions ``escape_dep_filename``,
``escape_dep_env`` and ``write_out_deps`` in:
https://github.com/rust-lang/rust/blob/ded5c06cf21d2b93bffd5d884aa6e96934ee4234/compiler/rustc_interface/src/passes.rs

Crucially, this producer escapes ONLY ASCII spaces in prerequisite filenames.
For real files ``with space.rs``, ``hash#file.rs``, ``money$file.rs`` and
``slash\\file.rs``, it emits ``with\\ space.rs``, ``hash#file.rs``,
``money$file.rs`` and ``slash\\file.rs`` respectively. Generic Make parsing would
misread some of these. We accept escaped spaces, but reject hash/dollar/literal
backslash filenames, other Make metacharacters, tabs and colon-bearing paths.
Conventional Make escapes such as ``\\#``, ``\\\\`` and ``$$`` are NOT silently
accepted as if rustc had emitted them. Windows paths and non-UTF-8 paths are not
supported. Unsafe output-target spellings are also rejected.

Rules can have multiple targets; rustc repeats the same prerequisites for its
separate outputs. The output-target names are deliberately not returned: rustc
prints them WITHOUT escaping spaces, so one spaced target and several targets
cannot be distinguished. This ambiguity does not change prerequisite paths.
Every dependency must also have rustc's single-input, empty-prerequisite rule.
Those rules are cross-checked, preventing a truncated or ambiguously split list
from silently becoming a partial input inventory. Duplicate paths are retained
once, in first-seen order. Multiple output rules must agree exactly.

As a narrowly specified extension, a continuation is accepted only after an
ASCII-space separator and before a nonempty, space-indented next line. Rustc
1.92 does not itself wrap rules. Mid-token continuations and ambiguous raw
backslashes fail closed; no general Make whitespace folding is attempted.

Only ``# env-dep:KEY[=VALUE]`` comments are accepted. Their independent rustc
encoding decodes backslash-n, backslash-r and doubled backslashes; spaces,
hashes, dollars and additional equals signs remain literal. Absent and empty
values remain distinct. Environment values are never included in error text.
"""

from collections.abc import Iterator
from typing import NamedTuple


RUSTC_RELEASE = "1.92.0"
RUSTC_COMMIT = "ded5c06cf21d2b93bffd5d884aa6e96934ee4234"
_ENV_PREFIX = "# env-dep:"
_UNSUPPORTED_PATH_CHARACTERS = frozenset("#$:;|=*?[]%~&()")


class DependencyInfoError(ValueError):
    """Malformed or unsupported dep-info cannot establish an input inventory."""


class EnvironmentDependency(NamedTuple):
    """A compiler-observed variable; ``None`` means absent, not an empty value."""

    name: str
    value: str | None


class DependencyInfo(NamedTuple):
    """Unresolved compiler input paths and separately decoded environment data."""

    dependencies: tuple[str, ...]
    environment: tuple[EnvironmentDependency, ...]
    output_rule_count: int


def _logical_lines(text: str, context: str) -> Iterator[tuple[int, str]]:
    """Join only explicit between-token continuations, preserving source lines.

    Comments are recognized first: an escaped environment backslash at physical
    end-of-line must never swallow the following comment or dependency rule.
    CRLF is supported, but isolated carriage returns are not treated as breaks.
    """

    physical = text.replace("\r\n", "\n").split("\n")
    index = 0
    while index < len(physical):
        number = index + 1
        line = physical[index]
        index += 1
        if line.startswith("#"):
            yield number, line
            continue
        while line.endswith("\\"):
            location = f"{context}: line {number}"
            if not line.endswith(" \\") or line.endswith("\\ \\"):
                raise DependencyInfoError(f"{location}: unsupported mid-token continuation")
            if index >= len(physical):
                raise DependencyInfoError(f"{location}: unterminated continuation")
            following = physical[index]
            index += 1
            if not following.startswith(" ") or not following.strip(" "):
                raise DependencyInfoError(f"{location}: continuation needs an indented token")
            if following.lstrip(" ").startswith("#"):
                raise DependencyInfoError(f"{location}: comment cannot continue a rule")
            line = line[:-1] + following.lstrip(" ")
        yield number, line


def _path_words(text: str, location: str) -> tuple[str, ...]:
    """Decode space-separated paths with only rustc's literal-space escape.

    Unknown escapes are errors, not dropped backslashes. Restricting syntax
    also rejects Make variables, recipes, assignments, patterns and wildcards.
    Unicode is preserved byte-for-byte at the string level, without NFC folding.
    """

    words = []
    current = []
    index = 0
    while index < len(text):
        char = text[index]
        index += 1
        if char == "\\":
            if index == len(text) or text[index] != " ":
                raise DependencyInfoError(f"{location}: unsupported or ambiguous path escape")
            current.append(" ")
            index += 1
        elif char == " ":
            if current:
                words.append("".join(current))
                current = []
        elif char in _UNSUPPORTED_PATH_CHARACTERS or ord(char) < 32 or ord(char) == 127:
            raise DependencyInfoError(f"{location}: unsupported path or Make syntax")
        else:
            current.append(char)
    if current:
        words.append("".join(current))
    return tuple(words)


def _decode_environment(text: str, location: str) -> str:
    """Decode the producer's three environment escapes without printing values."""

    result = []
    index = 0
    escapes = {"n": "\n", "r": "\r", "\\": "\\"}
    while index < len(text):
        char = text[index]
        index += 1
        if char == "\\":
            if index == len(text) or text[index] not in escapes:
                raise DependencyInfoError(f"{location}: unsupported environment escape")
            char = escapes[text[index]]
            index += 1
        result.append(char)
    return "".join(result)


def parse_dep_info(text: str, *, context: str = "<dep-info>") -> DependencyInfo:
    """Validate one Rust 1.92 dep-info document and return its input inventory.

    Empty documents, input-free documents, missing/extra phony input rules,
    conflicting output inventories and conflicting environment observations are
    errors. Relative paths remain relative to the compiler's working directory;
    no existence, symlink, source-language or provenance claims are made here.
    Unknown comments (including optional nightly checksum comments) are errors.
    Callers must decode bytes as strict UTF-8, not with replacement or loss.
    """

    if not isinstance(text, str):
        raise DependencyInfoError(f"{context}: dep-info must be decoded UTF-8 text")
    if any(char in text for char in ("\x00", "\ufffd", "\ufeff")):
        raise DependencyInfoError(f"{context}: NUL, replacement characters and BOM are unsupported")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        raise DependencyInfoError(f"{context}: undecodable Unicode is unsupported")
    if "\r" in text.replace("\r\n", ""):
        raise DependencyInfoError(f"{context}: isolated carriage return is unsupported")

    prerequisites = None
    phony_inputs = set()
    environment = {}
    output_rule_count = 0
    phase = "outputs"
    for number, line in _logical_lines(text, context):
        location = f"{context}: line {number}"
        if not line.strip(" "):
            continue
        if line.startswith(_ENV_PREFIX):
            if prerequisites is None:
                raise DependencyInfoError(f"{location}: environment comment precedes input rules")
            phase = "environment"
            encoded_name, separator, encoded_value = line[len(_ENV_PREFIX):].partition("=")
            name = _decode_environment(encoded_name, location)
            if not name:
                raise DependencyInfoError(f"{location}: empty environment variable name")
            value = _decode_environment(encoded_value, location) if separator else None
            if name in environment and environment[name] != value:
                raise DependencyInfoError(f"{location}: conflicting environment observations")
            environment[name] = value
            continue
        if line.startswith("#"):
            raise DependencyInfoError(f"{location}: unsupported comment")
        if phase == "environment":
            raise DependencyInfoError(f"{location}: rule follows environment comments")
        if line.count(":") != 1:
            raise DependencyInfoError(f"{location}: expected exactly one rule separator")
        targets_text, inputs_text = line.split(":", 1)
        targets = _path_words(targets_text, location)
        inputs = _path_words(inputs_text, location)
        if not targets:
            raise DependencyInfoError(f"{location}: rule has no target")
        if inputs:
            if phase != "outputs":
                raise DependencyInfoError(f"{location}: output rule follows phony input rules")
            if prerequisites is not None and prerequisites != inputs:
                raise DependencyInfoError(f"{location}: output prerequisite inventories disagree")
            prerequisites = inputs
            output_rule_count += 1
        else:
            if len(targets) != 1:
                raise DependencyInfoError(f"{location}: phony input rule must name one path")
            phase = "phony"
            phony_inputs.add(targets[0])

    if prerequisites is None:
        raise DependencyInfoError(f"{context}: no output rule with compiler inputs")
    if set(prerequisites) != phony_inputs:
        raise DependencyInfoError(f"{context}: prerequisite and phony input inventories disagree")
    return DependencyInfo(
        tuple(dict.fromkeys(prerequisites)),
        tuple(EnvironmentDependency(name, value) for name, value in environment.items()),
        output_rule_count,
    )
