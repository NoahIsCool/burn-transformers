#!/usr/bin/env python3
"""Enforce authored Rust documentation with the repository's pinned Syn helper.

This wrapper never compiles code. Build scripts/coverage-layout first, then pass
its binary with --classifier or place it in the repository's normal target dir.
Only syntax-proven test bodies are exempt; file naming never proves test status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


SOURCE_ROOTS = ("src", "examples", "fuzz", "scripts")


def digest(path: Path) -> str:
    """Hash exact file bytes without depending on platform-specific utilities."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def discover_sources(root: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Inventory authored Rust roots, preserving explicit test/fixture inputs."""
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.rs"],
        cwd=root, check=True, stdout=subprocess.PIPE,
    ).stdout
    included, excluded = [], []
    for name in sorted(set(output.decode().split("\0")) - {""}):
        parts = Path(name).parts
        if not parts or parts[0] not in SOURCE_ROOTS:
            excluded.append({"path": name, "reason": "outside authored production roots (historical evidence, integration tests, or retained artifacts)"})
        elif "fixtures" in parts or "tests" in parts:
            excluded.append({"path": name, "reason": "test-harness/fixture discovery root; a production module reference still causes auditing"})
        else:
            included.append(name)
    return included, excluded


def default_entries(root: Path, sources: list[str]) -> list[str]:
    """Select explicit crate roots; later orphan inspection prevents silent omissions."""
    entries = {
        name for name in sources
        if (name in {"src/lib.rs", "fuzz/src/lib.rs"}
            or (name.startswith("examples/") and len(Path(name).parts) == 2)
            or name.startswith("fuzz/fuzz_targets/")
            or (name.startswith("scripts/") and Path(name).name in {"lib.rs", "main.rs"}
                and Path(name).parent.name == "src"))
    }
    return sorted(name for name in entries if (root / name).is_file())


def component(path: str) -> str:
    """Group findings by stable repository components for contributor reports."""
    if path.startswith("src/checkpoint/") or path == "src/checkpoint.rs":
        return "checkpoint"
    if path.startswith("src/models/"):
        return "models"
    if path.startswith("src/"):
        return "shared"
    if path.startswith("scripts/"):
        return "tooling"
    if path.startswith("fuzz/"):
        return "fuzz"
    if path.startswith("examples/"):
        return "examples"
    return "other"


def invoke(classifier: Path, root: Path, entries: list[str]) -> dict:
    """Run the source-only parser with explicit containment and entry arguments."""
    try:
        result = subprocess.run(
            [str(classifier), "--documentation", str(root), *entries],
            cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or b"").decode(errors="replace").strip()
        raise ValueError(f"documentation helper exited {error.returncode}: {detail}") from error
    report = json.loads(result.stdout)
    if report.get("schema_version") != 1 or Path(report.get("root", "")).resolve() != root:
        raise ValueError("documentation helper returned an unsupported schema or another source root")
    return report


def audit(root: Path, classifier: Path, entries: list[str] | None = None) -> dict:
    """Bind syntax findings to a stable authored-source inventory and helper binary."""
    sources, excluded = discover_sources(root)
    before = {name: digest(root / name) for name in sources}
    selected = entries if entries is not None else default_entries(root, sources)
    if not selected:
        raise ValueError("no Rust crate entries found")
    report = invoke(classifier, root, selected)
    visited = {file["path"] for file in report["files"]}
    orphans = sorted(set(sources) - visited)
    if orphans:
        # An unreferenced *_tests.rs name is not evidence of test ownership.
        # Audit it conservatively as source, rather than silently dropping it.
        report = invoke(classifier, root, selected + orphans)
    after_sources, _ = discover_sources(root)
    if sources != after_sources or any(digest(root / name) != value for name, value in before.items()):
        raise ValueError("authored Rust sources changed during documentation auditing")
    for file in report["files"]:
        path = (root / file["path"]).resolve()
        if not path.is_relative_to(root) or digest(path) != file["sha256"]:
            raise ValueError(f"{file['path']}: source no longer matches the parsed snapshot")
    groups: dict[str, dict[str, int]] = {}
    missing, placements, opaque = [], [], []
    for file in report["files"]:
        group = component(file["path"])
        counts = groups.setdefault(group, {"production_functions": 0, "production_modules": 0,
            "missing_function_style": 0, "missing_module_style": 0, "excluded_test_sites": 0,
            "inline_test_bodies": 0, "unresolved_macro_functions": 0})
        for site in file["sites"]:
            if site["test_only"]:
                counts["excluded_test_sites"] += 1
                continue
            module = site["kind"] == "module"
            counts["production_modules" if module else "production_functions"] += 1
            if not site["style_present"]:
                counts["missing_module_style" if module else "missing_function_style"] += 1
                missing.append({"component": group, "path": file["path"], "line": site["span"]["start"]["line"],
                    "name": site["name"], "kind": site["kind"], "required_style": site["required_style"],
                    "rustdoc_present": site["rustdoc_present"], "computed_documentation": site["computed_documentation"],
                    "origin": site["origin"]})
        for span in file.get("inline_test_bodies", []):
            counts["inline_test_bodies"] += 1
            placements.append({"component": group, "path": file["path"], "line": span["start"]["line"]})
        for boundary in file["macros"]:
            if not boundary["test_only"]:
                for span in boundary["unresolved_functions"]:
                    counts["unresolved_macro_functions"] += 1
                    opaque.append({"component": group, "path": file["path"], "line": span["start"]["line"], "macro": boundary["name"]})
    report["inventory"] = {"source_sha256": before, "excluded_discovery_sources": excluded,
        "conservative_orphan_entries": orphans, "classifier_sha256": digest(classifier),
        "classifier": str(classifier), "groups": groups, "missing": missing,
        "inline_test_bodies": placements, "unresolved_macro_functions": opaque}
    return report


def main(argv: list[str] | None = None) -> int:
    """Write an auditable inventory and fail on missing style or unresolved syntax."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory-only", action="store_true", help="report style findings without failing for them; unresolved syntax still fails")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    classifier = args.classifier
    if classifier is None:
        classifier = root / "target/debug/transformers-coverage-layout"
    try:
        classifier = classifier.resolve(strict=True)
        report = audit(root, classifier)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        for group, counts in sorted(report["inventory"]["groups"].items()):
            print(f"{group}: {counts['missing_function_style']}/{counts['production_functions']} functions and "
                f"{counts['missing_module_style']}/{counts['production_modules']} modules missing required line docs; "
                f"{counts['inline_test_bodies']} inline test bodies; {counts['unresolved_macro_functions']} opaque authored functions")
        if not report["complete"]:
            print("documentation audit incomplete: inspect source diagnostics and opaque authored macro functions", file=sys.stderr)
            return 2
        return 0 if report["passed"] or args.inventory_only else 1
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"documentation audit: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
