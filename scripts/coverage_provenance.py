"""Bind coverage observations to bounded snapshots of local crate inputs.

Snapshots authenticate bytes, not the completeness of the compiler dependency
graph. The declared input roots include Rust code and test fixtures; external
artifacts, build products and runtime dependencies are deliberately not read.
Compiler-generated dependency/source evidence is still required before deciding
that a test-only syntax candidate is safe to remove from a coverage denominator.

Pre/post comparison detects ordinary concurrent source changes. It is not a
signature or protection against an adversary changing and restoring bytes
between observations. Filesystem containment assumes non-adversarial directory
concurrency; final-component symlinks and non-regular files are rejected.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat


SOURCE_ROOTS = ("src", "tests", "examples", "benches", "acceptance", "scripts")
# Python bytecode and standalone Rust build output can change during a gate.
# These exclusions apply only below scripts, not to arbitrary source paths.
SCRIPT_OUTPUT_DIRECTORIES = ("__pycache__", "target", ".git")
ROOT_INPUTS = ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain", "rust-toolchain.toml", ".cargo/config", ".cargo/config.toml")
MANDATORY_INPUTS = ("Cargo.toml", "Cargo.lock", "src/lib.rs")
HASH_BUFFER_BYTES = 64 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILES = 16_384
MAX_DIRECTORY_DEPTH = 128
INVENTORY_DOMAIN = b"transformers.coverage.input-snapshot.v1\0"


class CoverageProvenanceError(ValueError):
    """A source identity or observation cannot be safely bound to its snapshot."""


def _normalized_relative(path: str) -> bool:
    """Accept one portable root-relative spelling, never traversal or aliases."""

    return (
        isinstance(path, str)
        and bool(path)
        and "\\" not in path
        and "\0" not in path
        and ":" not in path
        and all(part not in {"", ".", ".."} for part in path.split("/"))
    )


def _in_scope(path: str) -> bool:
    """Limit snapshots to declared input files and source/fixture directories."""

    return _normalized_relative(path) and (
        path in ROOT_INPUTS or any(path.startswith(root + "/") for root in SOURCE_ROOTS)
    )


def _stamp(metadata: os.stat_result) -> tuple[int, ...]:
    """Observe identity, length and mutation timestamps on the opened file."""

    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)


def fingerprint_file(path: Path, *, maximum_bytes: int = MAX_FILE_BYTES) -> dict:
    """Stream SHA-256 and physical-line count without trusting a path size hint.

    A line is terminated only by LF, matching Rust/LLVM source coordinates; a
    final non-LF byte contributes one extra line. CRLF bytes are not normalized.
    The size bound is enforced both before and during reads, and metadata must
    remain stable through the final descriptor check. No file bytes are emitted.
    """

    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise CoverageProvenanceError("file byte bound must be a positive integer")
    path = Path(path)
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode):
        raise CoverageProvenanceError(f"source is not a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or _stamp(before) != _stamp(observed):
            raise CoverageProvenanceError(f"source identity changed before read: {path}")
        if before.st_size > maximum_bytes:
            raise CoverageProvenanceError(f"source exceeds {maximum_bytes} byte bound: {path}")
        hasher = hashlib.sha256()
        size = newlines = 0
        last = b""
        while chunk := stream.read(min(HASH_BUFFER_BYTES, maximum_bytes - size + 1)):
            size += len(chunk)
            if size > maximum_bytes:
                raise CoverageProvenanceError(f"source grew beyond byte bound during read: {path}")
            hasher.update(chunk)
            newlines += chunk.count(b"\n")
            last = chunk[-1:]
        after = os.fstat(stream.fileno())
        if _stamp(before) != _stamp(after) or size != before.st_size:
            raise CoverageProvenanceError(f"source changed during read: {path}")
    return {"sha256": hasher.hexdigest(), "bytes": size, "lines": newlines + int(bool(last) and last != b"\n")}


def _inventory_digest(files: dict) -> str:
    """Hash an unambiguous sorted identity map independently of checkout path."""

    encoded = json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(INVENTORY_DOMAIN + encoded).hexdigest()


def capture_sources(
    crate: Path, *, maximum_file_bytes: int = MAX_FILE_BYTES,
    maximum_total_bytes: int = MAX_TOTAL_BYTES, maximum_files: int = MAX_FILES,
) -> dict:
    """Snapshot every regular input under the fixed source/fixture policy.

    Git ignore rules are irrelevant: an ignored Rust module can still compile.
    The external ``models`` symlink and ``target`` are outside the declared
    roots and never traversed. Symlinks anywhere within an input root fail
    closed, including dangling links, instead of hiding an unobserved input.
    """

    for label, bound in (("file bytes", maximum_file_bytes), ("total bytes", maximum_total_bytes), ("files", maximum_files)):
        if type(bound) is not int or bound <= 0:
            raise CoverageProvenanceError(f"{label} bound must be a positive integer")
    root = Path(crate).resolve(strict=True)
    if not root.is_dir():
        raise CoverageProvenanceError("crate root must be a directory")
    paths = []
    directories = entries = 0

    def visit(path: Path, depth: int = 0) -> None:
        """Walk only checked directories, preserving all file names and types."""

        nonlocal directories, entries
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            directories += 1
            if depth > MAX_DIRECTORY_DEPTH or directories > maximum_files:
                raise CoverageProvenanceError("source inventory exceeds directory traversal bound")
            children = []
            # Path.iterdir uses eager os.listdir on Python 3.11. scandir keeps
            # enumeration bounded before sorting even a single wide directory.
            with os.scandir(path) as directory:
                for entry in directory:
                    entries += 1
                    if entries > maximum_files * 2:
                        raise CoverageProvenanceError("source inventory exceeds directory entry bound")
                    child = path / entry.name
                    relative = child.relative_to(root).parts
                    if relative[0] == "scripts" and child.name in SCRIPT_OUTPUT_DIRECTORIES:
                        if child.is_symlink() or not child.is_dir():
                            raise CoverageProvenanceError(f"script output exclusion is not a regular directory: {child}")
                        continue
                    children.append(child)
            for child in sorted(children):
                visit(child, depth + 1)
        elif stat.S_ISREG(mode):
            if len(paths) >= maximum_files:
                raise CoverageProvenanceError("source inventory exceeds file bound")
            relative = path.relative_to(root).as_posix()
            if not _in_scope(relative):
                raise CoverageProvenanceError(f"noncanonical source input: {relative!r}")
            paths.append((relative, path))
        else:
            raise CoverageProvenanceError(f"source input is not a regular file or directory: {path}")

    for name in SOURCE_ROOTS:
        path = root / name
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_dir():
                raise CoverageProvenanceError(f"source root is not a non-symlink directory: {path}")
            visit(path)
    if (root / ".cargo").is_symlink():
        raise CoverageProvenanceError("settings directory is a symlink")
    for name in ROOT_INPUTS:
        path = root / name
        if path.exists() or path.is_symlink():
            # Optional settings cannot inherit a symlinked .cargo directory.
            if path.parent != root and path.parent.is_symlink():
                raise CoverageProvenanceError(f"settings directory is a symlink: {path.parent}")
            if not path.is_file() or path.is_symlink():
                raise CoverageProvenanceError(f"root input is not a regular non-symlink file: {path}")
            visit(path)
    files = {}
    total = 0
    for relative, path in sorted(paths):
        identity = fingerprint_file(path, maximum_bytes=maximum_file_bytes)
        total += identity["bytes"]
        if total > maximum_total_bytes:
            raise CoverageProvenanceError("source inventory exceeds total byte bound")
        files[relative] = identity
    missing = set(MANDATORY_INPUTS) - files.keys()
    if missing:
        raise CoverageProvenanceError(f"missing mandatory source inputs: {sorted(missing)}")
    return {
        "schema_version": 1,
        "crate_root": str(root),
        "scope_roots": list(SOURCE_ROOTS),
        "root_inputs": list(ROOT_INPUTS),
        "script_output_directory_exclusions": list(SCRIPT_OUTPUT_DIRECTORIES),
        "compiler_dependency_closure_verified": False,
        "files": files,
        "total_bytes": total,
        "inventory_sha256": _inventory_digest(files),
    }


def require_same_sources(before: dict, after: dict) -> None:
    """Reject changed, added, removed or relocated source before using counters."""

    if before["crate_root"] != after["crate_root"]:
        raise CoverageProvenanceError("coverage source root changed during observation")
    left, right = before["files"], after["files"]
    added, removed = sorted(right.keys() - left.keys()), sorted(left.keys() - right.keys())
    changed = sorted(name for name in left.keys() & right.keys() if left[name] != right[name])
    if added or removed or changed:
        raise CoverageProvenanceError(f"coverage inputs changed: added={added}, removed={removed}, changed={changed}")
    if before["inventory_sha256"] != after["inventory_sha256"]:
        raise CoverageProvenanceError("coverage source inventory digest changed")


def validate_classification_sources(snapshot: dict, report: dict) -> None:
    """Bind parsed source buffers without adopting any exclusion or cfg proof."""

    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report.get("schema_version") != 1 or report.get("purpose") != "supplementary_source_classification_not_a_coverage_gate":
        raise CoverageProvenanceError("unsupported source-classifier schema")
    if report.get("qualified_for_coverage_exclusions") is not False or report.get("macro_expansion_resolved") is not False:
        raise CoverageProvenanceError("classifier unexpectedly claims qualified exclusions or macro expansion")
    if report.get("root") != snapshot["crate_root"]:
        raise CoverageProvenanceError("classifier root differs from source snapshot")
    files = report.get("files")
    if not isinstance(files, list) or not files:
        raise CoverageProvenanceError("classifier must contain a nonempty file inventory")
    seen = set()
    for entry in files:
        if not isinstance(entry, dict) or not _in_scope(entry.get("path")):
            raise CoverageProvenanceError("invalid classifier source path")
        name = entry["path"]
        if name in seen:
            raise CoverageProvenanceError(f"duplicate classifier source: {name}")
        seen.add(name)
        if type(entry.get("byte_length")) is not int or type(entry.get("line_count")) is not int:
            raise CoverageProvenanceError(f"invalid classifier source dimensions: {name}")
        identity = snapshot["files"].get(name)
        if identity is None or identity != {"sha256": entry.get("sha256"), "bytes": entry.get("byte_length"), "lines": entry.get("line_count")}:
            raise CoverageProvenanceError(f"classifier parsed different source bytes: {name}")
