#!/usr/bin/env python3
"""Capture native rustc artifact messages for an explicitly configured coverage run.

This development-only Cargo workspace wrapper forwards compiler output unchanged
and records artifacts named by rustc's JSON protocol. It never infers output
filenames from crate names or Makefile targets. Records bind observed bytes and
the actual argv/cwd/compiler version; they are not source exclusions, a compiler
dependency-closure proof, or an attestation against malicious compiler plugins.

The caller must create TFM_COVERAGE_CAPTURE_ROOT and provide the existing
TFM_COVERAGE_ARTIFACT_ROOT. Both are local directories. Native artifact paths
must remain under the latter, and final-component symlinks are rejected.
Do not point these variables at an artifact/model volume. The intended locations
are distinct subdirectories of the crate's normal Cargo target directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from coverage_provenance import fingerprint_file


MAX_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_VERSION_BYTES = 64 * 1024
SUPPORTED_EMITS = frozenset(("dep-info", "metadata", "link"))
PURE_PRINT_QUERIES = frozenset(("cfg", "sysroot", "target-libdir", "target-list", "target-cpus",
                              "target-features", "relocation-models", "code-models", "tls-models",
                              "file-names", "calling-conventions", "target-spec-json", "crate-name",
                              "split-debuginfo"))


class CompilerCaptureError(ValueError):
    """A native observation cannot be safely recorded as compiler evidence."""


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    """Reject ambiguous duplicate JSON keys without echoing diagnostic contents."""

    result = {}
    for key, value in pairs:
        if key in result:
            raise CompilerCaptureError("duplicate compiler JSON key")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    """Rust's native JSON protocol never emits NaN or infinity literals."""

    raise CompilerCaptureError("non-finite compiler JSON constant")


def artifact_message(line: bytes) -> dict | None:
    """Decode only native artifact messages; ordinary diagnostics remain opaque.

    Non-JSON stderr is forwarded by the wrapper but is not an artifact record.
    A malformed JSON-looking line fails capture rather than silently dropping
    potentially truncated metadata. These bounds limit retained protocol data,
    not the compiler's own memory consumption.
    """

    if not isinstance(line, bytes) or len(line) > MAX_MESSAGE_BYTES:
        raise CompilerCaptureError("compiler message exceeds byte bound or has wrong type")
    if not line.lstrip().startswith(b"{"):
        return None
    try:
        value = json.loads(line.decode("utf-8"), object_pairs_hook=unique_object,
                           parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise CompilerCaptureError("malformed compiler JSON message") from error
    if not isinstance(value, dict) or value.get("$message_type") != "artifact":
        return None
    path, emit = value.get("artifact"), value.get("emit")
    if not isinstance(path, str) or not path or "\0" in path:
        raise CompilerCaptureError("invalid native artifact path")
    if not isinstance(emit, str) or emit not in SUPPORTED_EMITS:
        raise CompilerCaptureError("unsupported native artifact emission kind")
    return {"artifact": path, "emit": emit}


def checked_directory(path: Path) -> Path:
    """Require an existing non-symlink directory before resolving its location."""

    if path.is_symlink() or not path.is_dir():
        raise CompilerCaptureError("capture/output root must be an existing non-symlink directory")
    return path.resolve(strict=True)


def artifact_identities(messages: list[dict], cwd: Path, output_root: Path) -> list[dict]:
    """Hash compiler-named files only inside the configured build-output root.

    Relative native filenames are resolved against rustc's actual working
    directory. Dependency bytes are copied separately by the caller, after
    identity validation. This function neither parses nor classifies sources.
    """

    root = checked_directory(output_root)
    records = []
    seen = set()
    for message in messages:
        declared = Path(message["artifact"])
        path = declared if declared.is_absolute() else cwd / declared
        if path.is_symlink():
            raise CompilerCaptureError("native artifact is a symlink")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise CompilerCaptureError("native artifact escapes the configured output root")
        if resolved in seen:
            raise CompilerCaptureError("duplicate native artifact identity")
        seen.add(resolved)
        records.append({"path": str(resolved), "declared_path": message["artifact"],
                        "emit": message["emit"],
                        "identity": fingerprint_file(path, maximum_bytes=MAX_ARTIFACT_BYTES)})
    if sum(record["emit"] == "dep-info" for record in records) != 1:
        raise CompilerCaptureError("successful compiler invocation must name exactly one dep-info artifact")
    if not any(record["emit"] in ("metadata", "link") for record in records):
        raise CompilerCaptureError("successful compiler invocation has no compiled artifact")
    return records


def emission_requested(arguments: list[str]) -> bool:
    """Separate Cargo's rustc probes from explicit compilation invocations."""

    return any(value == "--emit" or value.startswith("--emit=") for value in arguments)


def compiler_probe(arguments: list[str]) -> bool:
    """Positively recognize non-emitting version and Cargo metadata queries.

    Default-emission compiles and response files are not probes. In particular,
    ``--print=link-args`` can still link a program, so arbitrary print options
    cannot bypass artifact capture. The compiler itself validates other options
    accompanying these explicitly non-emitting metadata queries.
    """

    if any(value.startswith("@") for value in arguments) or emission_requested(arguments):
        return False
    if arguments in (["-V"], ["-vV"], ["-Vv"], ["--version"], ["--version", "--verbose"]):
        return True
    queries = []
    for index, value in enumerate(arguments):
        if value == "--print":
            if index + 1 >= len(arguments):
                return False
            queries.extend(arguments[index + 1].split(","))
        elif value.startswith("--print="):
            queries.extend(value[len("--print="):].split(","))
    return bool(queries) and all(query in PURE_PRINT_QUERIES for query in queries)


def write_record(path: Path, record: dict) -> None:
    """Create one invocation record exclusively; never overwrite earlier evidence."""

    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


def compiler_identity(executable: str, cwd: Path, environment: dict[str, str]) -> str:
    """Observe rustc's small version response without unbounded memory buffering.

    Native output is spooled to unlinked temporary files, then read with a strict
    cap. The timeout bounds execution, not a hostile executable's disk writes;
    this wrapper assumes an explicitly selected, trusted local compiler.
    Producer support is validated separately from this observation.
    """

    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        result = subprocess.run([executable, "-Vv"], cwd=cwd, env=environment,
                                stdout=output, stderr=errors, check=False, timeout=30)
        output.seek(0)
        errors.seek(0)
        version = output.read(MAX_VERSION_BYTES + 1)
        if result.returncode != 0 or errors.read(1) or len(version) > MAX_VERSION_BYTES:
            raise CompilerCaptureError("compiler identity probe failed, exceeded its bound or emitted diagnostics")
    try:
        text = version.decode("utf-8")
    except UnicodeError as error:
        raise CompilerCaptureError("compiler identity is not UTF-8") from error
    if not text.startswith("rustc "):
        raise CompilerCaptureError("compiler identity probe did not identify rustc")
    return text


def capture(arguments: list[str], environment: dict[str, str], cwd: Path) -> int:
    """Run one compiler child, forwarding stderr and preserving native exit status.

    Capture validation failure turns an otherwise successful invocation into a
    failed wrapper result. Native failures retain their original exit status and
    cannot contribute artifact identities. A per-process filename and exclusive
    creation prevent concurrent invocations from overwriting each other's data.
    """

    if not arguments:
        raise CompilerCaptureError("missing compiler executable")
    if any(value.startswith("@") for value in arguments[1:]):
        raise CompilerCaptureError("compiler response files are unsupported for artifact capture")
    if compiler_probe(arguments[1:]):
        return subprocess.run(arguments, cwd=cwd, env=environment, check=False).returncode
    if not emission_requested(arguments[1:]):
        raise CompilerCaptureError("non-probe compiler invocation must request explicit artifact emission")
    capture_root = checked_directory(Path(environment["TFM_COVERAGE_CAPTURE_ROOT"]))
    output_root = checked_directory(Path(environment["TFM_COVERAGE_ARTIFACT_ROOT"]))
    compiler = compiler_identity(arguments[0], cwd, environment)
    record = {"schema_version": 1, "command": arguments, "cwd": str(cwd.resolve()),
              "compiler_version": compiler, "python_version": sys.version,
              "process_exit_code": None, "capture_valid": False,
              "compiler_dependency_closure_verified": False,
              "production_only_exclusions_qualified": False}
    messages = []
    capture_error = None
    with subprocess.Popen(arguments, cwd=cwd, env=environment, stderr=subprocess.PIPE) as child:
        assert child.stderr is not None
        while line := child.stderr.readline(MAX_MESSAGE_BYTES + 1):
            sys.stderr.buffer.write(line)
            sys.stderr.buffer.flush()
            if capture_error is None:
                try:
                    message = artifact_message(line)
                    if message is not None:
                        if len(messages) >= 16:
                            raise CompilerCaptureError("too many native artifacts")
                        messages.append(message)
                except ValueError as error:
                    capture_error = str(error)
        code = child.wait()
    record["process_exit_code"] = code
    if code == 0 and capture_error is None:
        try:
            records = artifact_identities(messages, cwd, output_root)
            for artifact in records:
                if artifact["emit"] == "dep-info":
                    source = Path(artifact["path"])
                    # Native dependency files are small; use the same bounded
                    # identity before/after copying rather than trusting size.
                    identity = fingerprint_file(source, maximum_bytes=MAX_MESSAGE_BYTES)
                    if identity != artifact["identity"]:
                        raise CompilerCaptureError("dep-info changed before capture")
                    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                    with os.fdopen(os.open(source, flags), "rb") as stream:
                        payload = stream.read(MAX_MESSAGE_BYTES + 1)
                    if len(payload) > MAX_MESSAGE_BYTES or identity != fingerprint_file(source, maximum_bytes=MAX_MESSAGE_BYTES):
                        raise CompilerCaptureError("dep-info changed during capture")
                    destination = capture_root / f"rustc-{os.getpid()}.d"
                    with destination.open("xb") as stream:
                        stream.write(payload)
                    if identity != fingerprint_file(destination, maximum_bytes=MAX_MESSAGE_BYTES):
                        raise CompilerCaptureError("retained dep-info differs from compiler output")
                    artifact["retained_dep_info"] = destination.name
            for artifact in records:
                if artifact["identity"] != fingerprint_file(Path(artifact["path"]), maximum_bytes=MAX_ARTIFACT_BYTES):
                    raise CompilerCaptureError("compiled artifact changed during capture")
            record["artifacts"] = records
            record["capture_valid"] = True
        except (OSError, ValueError) as error:
            capture_error = str(error)
    if capture_error is not None:
        record["capture_error"] = capture_error
    write_record(capture_root / f"rustc-{os.getpid()}.json", record)
    if code == 0 and not record["capture_valid"]:
        print("error: compiler artifact evidence could not be captured", file=sys.stderr)
        return 1
    return code


def main() -> int:
    """Enter only when explicitly selected as Cargo's workspace compiler wrapper."""

    try:
        return capture(sys.argv[1:], dict(os.environ), Path.cwd())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"error: compiler artifact capture: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
