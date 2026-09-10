#!/usr/bin/env python3
"""Read-only, byte-exact audit of the transformers subtree history extraction.

Usage: python3 verify_history.py ORIGINAL_REPO FILTERED_GIT_DIR COMMIT_MAP

The adjacent ``ref-map`` is required unless ``--ref-map`` is supplied. Every
nonzero commit mapping must preserve the original ``crates/transformers`` tree,
raw author/committer header lines, and the entire commit message. With no pruned
commits, every parent ID must be rewritten exactly through the commit map, in
original order. Other headers (for example signatures) are not compared.
Every ref-map row must match both repositories, and the filtered repository must
have exactly the surviving mapped refs. Explicitly inventoried operational tree
refs may be omitted only with ``--omitted-original-refs``. No files or Git objects
are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import BinaryIO


PREFIX = (b"crates", b"transformers")
OID_RE = re.compile(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class AuditError(Exception):
    """Describe invalid evidence or a failed local Git read."""


def git_environment() -> dict[str, str]:
    """Disable optional writes/replacement objects and inherited repository overrides."""
    env = os.environ.copy()
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        env.pop(name, None)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def git_read(arguments: list[str]) -> bytes:
    """Run a non-mutating Git query and preserve stdout bytes without decoding."""
    result = subprocess.run(
        ["git", "--no-replace-objects", *arguments],
        env=git_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise AuditError(
            f"Git read failed ({result.returncode}): {arguments!r}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def is_zero(oid: bytes) -> bool:
    """Recognize filter-repo's all-zero removed-object sentinel."""
    return bool(oid) and not oid.strip(b"0")


def require_oid(oid: bytes, context: str, *, allow_zero: bool = False) -> None:
    """Require a complete lowercase SHA-1/SHA-256 object ID, optionally zero."""
    if OID_RE.fullmatch(oid) is None or (is_zero(oid) and not allow_zero):
        raise AuditError(f"Invalid object ID in {context}: {oid!r}")


def read_map(path: Path, width: int) -> tuple[list[tuple[bytes, ...]], str]:
    """Read a strict filter-repo map and return rows plus the original-file digest."""
    data = path.read_bytes()
    lines = data.splitlines()
    expected_header = [b"old", b"new"] + ([b"ref"] if width == 3 else [])
    if not lines or lines[0].split() != expected_header:
        raise AuditError(f"Unexpected header in {path}")
    rows: list[tuple[bytes, ...]] = []
    seen: set[bytes] = set()
    for number, line in enumerate(lines[1:], 2):
        fields = tuple(line.split())
        if len(fields) != width:
            raise AuditError(f"Expected {width} fields at {path}:{number}")
        require_oid(fields[0], f"{path}:{number} old")
        require_oid(fields[1], f"{path}:{number} new", allow_zero=True)
        identity = fields[0] if width == 2 else fields[2]
        if identity in seen:
            raise AuditError(f"Duplicate map identity at {path}:{number}: {identity!r}")
        if width == 3 and not identity.startswith(b"refs/"):
            raise AuditError(f"Non-ref name at {path}:{number}: {identity!r}")
        seen.add(identity)
        rows.append(fields)
    if not rows:
        raise AuditError(f"Empty evidence map: {path}")
    return rows, hashlib.sha256(data).hexdigest()


def read_exact(stream: BinaryIO, length: int) -> bytes:
    """Consume an exact framed object payload without assuming one read fills it."""
    pieces = []
    remaining = length
    while remaining:
        piece = stream.read(remaining)
        if not piece:
            raise AuditError("Unexpected EOF in cat-file object payload")
        pieces.append(piece)
        remaining -= len(piece)
    return b"".join(pieces)


def omitted_refs(path: Path | None) -> tuple[dict[bytes, bytes], dict[str, object] | None]:
    """Read exact original-only Codex operational tree identities, never a wildcard."""
    if path is None:
        return {}, None
    path = path.resolve(strict=True)
    data = path.read_bytes()
    result: dict[bytes, bytes] = {}
    for number, line in enumerate(data.splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise AuditError(f"Expected object ID and ref at {path}:{number}")
        oid, name = fields
        require_oid(oid, f"{path}:{number}")
        if not name.startswith(b"refs/codex/turn-diffs/") or name in result:
            raise AuditError(f"Invalid/duplicate operational ref at {path}:{number}: {name!r}")
        result[name] = oid
    if not result:
        raise AuditError(f"Empty operational-ref inventory: {path}")
    return result, {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "refs": [{"ref": os.fsdecode(name), "original_oid": oid.decode("ascii")} for name, oid in sorted(result.items())],
    }


class ObjectStore:
    """Read Git objects through one local cat-file batch process per repository."""

    def __init__(self, git_dir: Path):
        """Validate the storage hash format and open a read-only object stream."""
        self.git_dir = git_dir
        self.arguments = [f"--git-dir={git_dir}"]
        storage = git_read([*self.arguments, "rev-parse", "--show-object-format"]).strip()
        if storage not in (b"sha1", b"sha256"):
            raise AuditError(f"Unsupported Git object format: {storage!r}")
        self.oid_bytes = 20 if storage == b"sha1" else 32
        self.trees: dict[bytes, dict[bytes, tuple[bytes, bytes]]] = {}
        self.process = subprocess.Popen(
            ["git", "--no-replace-objects", *self.arguments, "cat-file", "--batch"],
            env=git_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def close(self) -> None:
        """Close the read stream and require a successful batch-process exit."""
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        code = self.process.wait()
        if code:
            raise AuditError(f"cat-file failed for {self.git_dir}: exit {code}")

    def object(self, oid: bytes, kind: bytes) -> bytes:
        """Return one exact object body, rejecting missing objects or wrong types."""
        require_oid(oid, f"object in {self.git_dir}")
        if len(oid) != self.oid_bytes * 2:
            raise AuditError(f"Object ID/storage-format mismatch: {oid!r}")
        source = self.process.stdout
        sink = self.process.stdin
        if source is None or sink is None:
            raise AuditError("cat-file stream was not opened")
        sink.write(oid + b"\n")
        sink.flush()
        header = source.readline().split()
        if len(header) != 3 or header[0] != oid:
            raise AuditError(f"Unexpected cat-file result for {oid!r}: {header!r}")
        try:
            length = int(header[2])
        except ValueError as error:
            raise AuditError(f"Invalid cat-file size: {header!r}") from error
        if length < 0:
            raise AuditError(f"Negative cat-file size: {header!r}")
        payload = read_exact(source, length)
        if source.read(1) != b"\n":
            raise AuditError(f"Invalid cat-file payload delimiter for {oid!r}")
        if header[1] != kind:
            raise AuditError(f"Unexpected object kind for {oid!r}: {header[1]!r}, expected {kind!r}")
        return payload

    def tree(self, oid: bytes) -> dict[bytes, tuple[bytes, bytes]]:
        """Parse binary Git tree entries, preserving raw names and exact object IDs."""
        if oid in self.trees:
            return self.trees[oid]
        body = self.object(oid, b"tree")
        entries: dict[bytes, tuple[bytes, bytes]] = {}
        cursor = 0
        while cursor < len(body):
            space = body.find(b" ", cursor)
            nul = body.find(b"\0", space + 1)
            end = nul + 1 + self.oid_bytes
            if space < cursor or nul <= space + 1 or end > len(body):
                raise AuditError(f"Malformed tree object: {oid!r}")
            mode = body[cursor:space]
            name = body[space + 1 : nul]
            entry_oid = body[nul + 1 : end].hex().encode("ascii")
            if name in entries:
                raise AuditError(f"Duplicate tree entry: {oid!r} {name!r}")
            entries[name] = (mode, entry_oid)
            cursor = end
        self.trees[oid] = entries
        return entries

    def extracted_tree(self, root: bytes) -> bytes:
        """Resolve exactly the original crates/transformers tree, without Git path guessing."""
        current = root
        for component in PREFIX:
            entry = self.tree(current).get(component)
            if entry is None or entry[0] not in (b"40000", b"040000"):
                raise AuditError(
                    f"Original commit lacks directory crates/transformers (at {component!r})"
                )
            current = entry[1]
        self.object(current, b"tree")
        return current


def commit_fields(body: bytes) -> tuple[bytes, bytes, bytes, bytes]:
    """Extract the tree ID, exact identity header lines, and unmodified message bytes."""
    header, separator, message = body.partition(b"\n\n")
    if not separator:
        raise AuditError("Commit lacks its header/message separator")
    result = []
    for key in (b"tree", b"author", b"committer"):
        matches = [line for line in header.split(b"\n") if line.startswith(key + b" ")]
        if len(matches) != 1:
            raise AuditError(f"Commit has {len(matches)} {key!r} headers")
        result.append(matches[0])
    tree = result[0][len(b"tree ") :]
    require_oid(tree, "commit tree")
    return tree, result[1], result[2], message


def refs(git_dir: Path) -> dict[bytes, bytes]:
    """Read every stored ref's unpeeled object identity, including annotated tags."""
    raw = git_read(
        [f"--git-dir={git_dir}", "for-each-ref", "--format=%(objectname) %(refname)"]
    )
    result: dict[bytes, bytes] = {}
    for line in raw.splitlines():
        parts = line.split(b" ", 1)
        if len(parts) != 2:
            raise AuditError(f"Malformed for-each-ref line: {line!r}")
        oid, name = parts
        require_oid(oid, f"ref {name!r}")
        if name in result:
            raise AuditError(f"Duplicate ref: {name!r}")
        result[name] = oid
    return result


def commit_parents(body: bytes) -> list[bytes]:
    """Read the ordered original parent list, including every merge parent."""
    header, separator, _message = body.partition(b"\n\n")
    if not separator:
        raise AuditError("Commit lacks its header/message separator")
    parents = [line[len(b"parent ") :] for line in header.split(b"\n") if line.startswith(b"parent ")]
    for parent in parents:
        require_oid(parent, "commit parent")
    return parents


def verify(arguments: argparse.Namespace) -> dict[str, object]:
    """Compare every mapped commit and ref, collecting all ordinary content mismatches."""
    original_repo = arguments.original_repo.resolve(strict=True)
    original_git = Path(
        os.fsdecode(git_read(["-C", str(original_repo), "rev-parse", "--absolute-git-dir"]).strip())
    ).resolve(strict=True)
    filtered_git = arguments.filtered_git_dir.resolve(strict=True)
    if original_git == filtered_git:
        raise AuditError("Original and filtered repositories must be distinct")
    commit_map = arguments.commit_map.resolve(strict=True)
    ref_map = (arguments.ref_map or commit_map.with_name("ref-map")).resolve(strict=True)
    commits, commit_digest = read_map(commit_map, 2)
    mapped_refs, ref_digest = read_map(ref_map, 3)
    omitted, omission_evidence = omitted_refs(arguments.omitted_original_refs)
    mapping = dict(commits)
    verify_parents = all(not is_zero(new) for _old, new in commits)
    failures: list[dict[str, object]] = []
    kept = 0
    pruned = 0
    original_store = ObjectStore(original_git)
    try:
        filtered_store = ObjectStore(filtered_git)
        try:
            for old, new in commits:
                try:
                    old_body = original_store.object(old, b"commit")
                    old_tree, old_author, old_committer, old_message = commit_fields(old_body)
                    if is_zero(new):
                        pruned += 1
                        continue
                    kept += 1
                    new_body = filtered_store.object(new, b"commit")
                    new_tree, new_author, new_committer, new_message = commit_fields(new_body)
                    if verify_parents:
                        parents = commit_parents(old_body)
                        missing = [parent for parent in parents if parent not in mapping]
                        if missing:
                            raise AuditError(f"Original parents missing from commit map: {missing!r}")
                        expected_parents = [mapping[parent] for parent in parents]
                        actual_parents = commit_parents(new_body)
                        if expected_parents != actual_parents:
                            failures.append({"kind": "commit_parent_mismatch", "old": old.decode("ascii"), "new": new.decode("ascii"), "expected": [parent.decode("ascii") for parent in expected_parents], "actual": [parent.decode("ascii") for parent in actual_parents]})
                    filtered_store.object(new_tree, b"tree")
                    extracted = original_store.extracted_tree(old_tree)
                    for label, expected, actual in (
                        ("tree", extracted, new_tree),
                        ("author", old_author, new_author),
                        ("committer", old_committer, new_committer),
                        ("message", old_message, new_message),
                    ):
                        if expected != actual:
                            failures.append(
                                {
                                    "kind": f"commit_{label}_mismatch",
                                    "old": old.decode("ascii"),
                                    "new": new.decode("ascii"),
                                    "expected_sha256": hashlib.sha256(expected).hexdigest(),
                                    "actual_sha256": hashlib.sha256(actual).hexdigest(),
                                    **(
                                        {"expected_tree": expected.decode("ascii"), "actual_tree": actual.decode("ascii")}
                                        if label == "tree"
                                        else {}
                                    ),
                                }
                            )
                except AuditError as error:
                    failures.append(
                        {"kind": "commit_read_error", "old": old.decode("ascii"), "new": new.decode("ascii"), "error": str(error)}
                    )
        finally:
            filtered_store.close()
    finally:
        original_store.close()

    original_refs = refs(original_git)
    filtered_refs = refs(filtered_git)
    mapped_names = {row[2] for row in mapped_refs}
    expected_refs = {name: new for _old, new, name in mapped_refs if not is_zero(new)}
    for old, new, name in mapped_refs:
        display = os.fsdecode(name)
        if original_refs.get(name) != old:
            failures.append({"kind": "original_ref_mismatch", "ref": display, "expected": old.decode("ascii"), "actual": os.fsdecode(original_refs.get(name, b"<absent>"))})
        actual = filtered_refs.get(name)
        if (is_zero(new) and actual is not None) or (not is_zero(new) and actual != new):
            failures.append({"kind": "filtered_ref_mismatch", "ref": display, "expected": None if is_zero(new) else new.decode("ascii"), "actual": None if actual is None else actual.decode("ascii")})
    for name in sorted(filtered_refs.keys() - expected_refs.keys()):
        failures.append({"kind": "unexpected_filtered_ref", "ref": os.fsdecode(name), "oid": filtered_refs[name].decode("ascii")})
    # All-ref extraction must not silently omit source refs from the evidence map.
    for name in sorted(original_refs.keys() - mapped_names - omitted.keys()):
        failures.append({"kind": "unmapped_original_ref", "ref": os.fsdecode(name), "oid": original_refs[name].decode("ascii")})
    for name, oid in sorted(omitted.items()):
        if name in mapped_names or original_refs.get(name) != oid or name in filtered_refs:
            failures.append({"kind": "operational_omission_identity_mismatch", "ref": os.fsdecode(name), "expected_original_oid": oid.decode("ascii")})
        if git_read([f"--git-dir={original_git}", "cat-file", "-t", oid.decode("ascii")]).strip() != b"tree":
            failures.append({"kind": "operational_omission_not_tree", "ref": os.fsdecode(name), "oid": oid.decode("ascii")})
    return {
        "schema_version": 1,
        "passed": not failures,
        "original_repository": str(original_repo),
        "original_git_dir": str(original_git),
        "filtered_git_dir": str(filtered_git),
        "subtree": "crates/transformers",
        "commit_map": {"path": str(commit_map), "sha256": commit_digest},
        "ref_map": {"path": str(ref_map), "sha256": ref_digest},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_head": git_read([f"--git-dir={original_git}", "rev-parse", "HEAD"]).strip().decode("ascii"),
        "filtered_head": git_read([f"--git-dir={filtered_git}", "rev-parse", "HEAD"]).strip().decode("ascii"),
        "omitted_original_operational_refs": omission_evidence,
        "omitted_original_operational_ref_count": len(omitted),
        "mapped_commits": len(commits),
        "preserved_commits": kept,
        "pruned_commits": pruned,
        "distinct_preserved_commits": len({new for _old, new in commits if not is_zero(new)}),
        "mapped_refs": len(mapped_refs),
        "surviving_refs": len(expected_refs),
        "pruned_refs": len(mapped_refs) - len(expected_refs),
        "checked": ["exact_extracted_tree_oid", "raw_author_header", "raw_committer_header", "raw_message_bytes", "original_and_resulting_ref_identities", "all_ref_map_completeness"] + (["exact_ordered_parent_id_mapping"] if verify_parents else []),
        "not_checked": (["rewritten_parent_ids_with_pruned_commits"] if not verify_parents else []) + ["other_commit_headers_or_signatures", "annotated_tag_object_metadata", "unreachable_unmapped_objects", "working_tree_files"],
        "failures": failures,
    }


def main() -> int:
    """Print machine-readable evidence and fail closed for malformed or mismatching data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_repo", type=Path)
    parser.add_argument("filtered_git_dir", type=Path)
    parser.add_argument("commit_map", type=Path)
    parser.add_argument("--ref-map", type=Path)
    parser.add_argument("--omitted-original-refs", "--omitted-refs", type=Path)
    arguments = parser.parse_args()
    try:
        report = verify(arguments)
    except (AuditError, OSError, ValueError) as error:
        report = {"schema_version": 1, "passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
