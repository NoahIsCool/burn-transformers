"""Run explicit, local source-release gates without downloading model artifacts.

Commands execute without a shell and stop at the first failure. Logs and the
command manifest are written to a new directory below Cargo's normal target
directory; no source files, Git refs, checkpoints, or support flags are changed.
Hardware acceptance and fuzz campaigns remain separate, serialized workflows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


CRATE = Path(__file__).resolve().parents[1]
FEATURES = "std,flex,tokenizers,chat-templates,autodiff"
GATES = ("source", "msrv", "coverage", "dependencies", "accelerator-compile", "package")
# These are maintenance notices, not vulnerability waivers. New IDs fail.
MAINTENANCE_NOTICES = ("RUSTSEC-2025-0141", "RUSTSEC-2024-0436")


def require_clean_coverage_log(log: str) -> None:
    """Reject compiler/exporter warnings even when the native process exits zero.

    LLVM may export partial coverage after reporting mismatched function data.
    Such counters must not enter a gate. No warning is allowlisted here: a new
    diagnostic requires review instead of silently accepting weaker evidence.
    ANSI terminal styling is ignored so it cannot hide a diagnostic prefix.
    """

    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log)
    diagnostic = re.compile(r"^\s*(?:.*?:\s+)?(?:warning|error)(?:\[[^]\n]+\])?:(?:\s|$)", re.IGNORECASE)
    for line in plain.splitlines():
        if diagnostic.match(line):
            raise ValueError(f"unqualified coverage diagnostic: {line.strip()}")


def coverage_summary(report: dict, crate: Path) -> dict:
    """Aggregate LLVM's line counters, retaining all compiled production files.

    Inline unit-test bodies and macro expansions are included by LLVM. This
    deliberately makes no production-only or branch-coverage claim. Integration
    harness files are not part of either denominator; experimental model source
    remains in the crate-wide denominator. Both groups must meet their gate.
    """

    files = []
    seen = set()
    for data in report["data"]:
        for entry in data["files"]:
            path = Path(entry["filename"]).resolve()
            if not path.is_relative_to(crate.resolve() / "src"):
                continue
            relative = path.relative_to(crate.resolve()).as_posix()
            if relative in seen:
                raise ValueError(f"duplicate coverage file: {relative}")
            seen.add(relative)
            lines = entry["summary"]["lines"]
            count, covered = lines["count"], lines["covered"]
            if type(count) is not int or type(covered) is not int or not 0 <= covered <= count:
                raise ValueError(f"invalid coverage counters: {relative}")
            files.append((relative, count, covered))
    result = {}
    for label, threshold in (("crate", 90), ("shared", 95)):
        selected = [row for row in files if label == "crate" or not row[0].startswith("src/models/")]
        count = sum(row[1] for row in selected)
        covered = sum(row[2] for row in selected)
        if count == 0:
            raise ValueError(f"empty coverage denominator: {label}")
        result[label] = {
            "lines": count, "covered": covered, "percent": 100 * covered / count,
            "minimum_percent": threshold, "passed": 100 * covered >= threshold * count,
        }
    result["scope"] = "compiled src/**/*.rs, including inline unit tests and generated expansions"
    result["branch_coverage_measured"] = False
    result["passed"] = all(result[label]["passed"] for label in ("crate", "shared"))
    return result


def command_plan(gate: str, output: Path, toolchain: str, msrv: str, offline: bool) -> list[list[str]]:
    """Build auditable argument arrays; platform checks never execute GPU code."""

    cargo = ["cargo", f"+{toolchain}"]
    manifest = ["--manifest-path", str(CRATE / "Cargo.toml")]
    locked = ["--locked"] + (["--offline"] if offline else [])
    feature_args = ["--no-default-features", "--features", FEATURES]
    layout_manifest = ["--manifest-path", str(CRATE / "scripts/coverage-layout/Cargo.toml")]
    # Both standalone packages share the runtime crate's normal target location.
    # This avoids creating a nested build directory inside packaged tool sources.
    layout_target = ["--target-dir", str(CRATE / "target")]
    if gate == "source":
        return [
            cargo + ["fmt"] + manifest + ["--check"],
            cargo + ["clippy"] + manifest + locked + ["--all-targets"] + feature_args + ["--", "-D", "warnings"],
            cargo + ["test"] + manifest + locked + feature_args,
            cargo + ["check"] + manifest + locked + ["--no-default-features"],
            cargo + ["check"] + manifest + locked + ["--no-default-features", "--features", "std"],
            cargo + ["doc"] + manifest + locked + feature_args + ["--no-deps"],
            [sys.executable, "-m", "unittest", "discover", "-s", str(CRATE / "scripts/tests"), "-v"],
            cargo + ["fmt"] + layout_manifest + ["--check"],
            cargo + ["clippy"] + layout_manifest + locked + layout_target + ["--all-targets", "--", "-D", "warnings"],
            cargo + ["test"] + layout_manifest + locked + layout_target,
        ]
    if gate == "msrv":
        return [
            ["cargo", f"+{msrv}", "test"] + manifest + locked + feature_args,
            ["cargo", f"+{msrv}", "test"] + layout_manifest + locked + layout_target,
        ]
    if gate == "coverage":
        return [cargo + ["llvm-cov"] + manifest + locked + feature_args + [
            "--lib", "--tests", "--json", "--output-path", str(output / "coverage.json"),
        ]]
    if gate == "dependencies":
        audit = ["cargo", "audit", "--file", str(CRATE / "Cargo.lock"), "--deny", "warnings"]
        if offline:
            audit += ["--no-fetch", "--no-yanked"]
        for advisory in MAINTENANCE_NOTICES:
            audit += ["--ignore", advisory]
        return [
            audit,
            ["cargo", "deny"] + manifest + locked
            + ["--config", str(CRATE / "deny.toml"), "check", "licenses", "sources"],
        ]
    if gate == "accelerator-compile":
        return [
            cargo + ["check"] + manifest + locked
            + ["--no-default-features", "--features", f"std,{backend},tokenizers,chat-templates"]
            for backend in ("metal", "cuda", "rocm")
        ]
    if gate == "package":
        # The parent repository may contain unrelated changes. Package listing
        # is diagnostic; the verified final archive still requires clean Git.
        return [
            cargo + ["package"] + manifest + locked + ["--list"],
            cargo + ["package"] + manifest + locked,
        ]
    raise ValueError(f"unknown gate: {gate}")


def require_msrv(toolchain: str) -> None:
    """Reject an alias that no longer resolves to the declared Rust 1.92 MSRV."""

    version = subprocess.check_output(["rustc", f"+{toolchain}", "--version"], text=True).split()
    if len(version) < 2 or version[1] != "1.92.0":
        raise ValueError(f"MSRV alias {toolchain!r} resolves to {' '.join(version)}, not rustc 1.92.0")


def validate_package_listing(listing: str) -> int:
    """Fail before packaging if Cargo follows an artifact path or includes weights.

    Cargo's slash-free include patterns match at any depth, even below a local
    symlink. Check its actual listing instead of assuming `.gitignore` prevails
    over an explicit package include. Compiler metadata warnings are not paths.
    """

    allowed_files = {"Cargo.toml", "Cargo.toml.orig", "Cargo.lock", "deny.toml",
                     "LICENSE-MIT", "LICENSE-APACHE", ".cargo_vcs_info.json"}
    allowed_roots = {"src", "examples", "tests", "scripts", "acceptance"}
    files = set()
    for raw in listing.splitlines():
        if not raw.strip() or raw.startswith("warning:") or raw.lstrip().startswith(("|", "= note:")):
            continue
        path = Path(raw)
        if path.is_absolute() or any(part in {"..", "__pycache__", "target", ".git"} for part in path.parts):
            raise ValueError(f"unsafe package entry: {raw}")
        if path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth", ".bpk", ".gguf", ".pyc", ".pyo"}:
            raise ValueError(f"runtime artifact in source package: {raw}")
        root_doc = len(path.parts) == 1 and path.suffix == ".md"
        if raw not in allowed_files and not root_doc and path.parts[0] not in allowed_roots:
            raise ValueError(f"unallowlisted package entry: {raw}")
        if raw in files:
            raise ValueError(f"duplicate package entry: {raw}")
        files.add(raw)
    if not {"Cargo.toml", "Cargo.lock", "README.md", "src/lib.rs"}.issubset(files):
        raise ValueError("incomplete Cargo package listing")
    return len(files)


def main() -> int:
    """Execute only requested gate groups and record failures without promotion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="append", choices=GATES, required=True)
    parser.add_argument("--toolchain", default="stable")
    parser.add_argument("--msrv-toolchain", default="1.92.0")
    parser.add_argument("--offline", action="store_true", help="requires pre-fetched dependencies/advisories")
    parser.add_argument("--dry-run", action="store_true", help="print commands without creating files")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = CRATE / "target/release-gates" / stamp
    plan = [(gate, command) for gate in args.gate for command in command_plan(
        gate, output, args.toolchain, args.msrv_toolchain, args.offline,
    )]
    if args.dry_run:
        for gate, command in plan:
            print(f"[{gate}] {shlex.join(command)}")
        return 0
    if "msrv" in args.gate:
        require_msrv(args.msrv_toolchain)
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.setdefault("CARGO_INCREMENTAL", "0")
    environment["CARGO_TERM_COLOR"] = "never"
    environment["RUSTDOCFLAGS"] = environment.get("RUSTDOCFLAGS", "") + " -D warnings"
    results = {"schema_version": 1, "created_utc": stamp, "gates": [], "passed": False}
    for index, (gate, command) in enumerate(plan):
        log = output / f"{index:02d}-{gate}.log"
        print(f"[{gate}] {shlex.join(command)}\nlog: {log}", flush=True)
        started = datetime.now(timezone.utc)
        with log.open("x") as stream:
            completed = subprocess.run(command, cwd=CRATE, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
        row = {"gate": gate, "command": command, "exit_code": completed.returncode,
               "process_exit_code": completed.returncode,
               "seconds": (datetime.now(timezone.utc) - started).total_seconds(), "log": log.name}
        results["gates"].append(row)
        try:
            if completed.returncode == 0 and gate == "coverage":
                require_clean_coverage_log(log.read_text())
                row["coverage"] = coverage_summary(json.loads((output / "coverage.json").read_text()), CRATE)
                if not row["coverage"]["passed"]:
                    row["exit_code"] = 1
            if completed.returncode == 0 and gate == "package" and "--list" in command:
                row["package_files"] = validate_package_listing(log.read_text())
        except (OSError, ValueError, KeyError, TypeError) as error:
            # Preserve the distinction between a successful native process and
            # invalid evidence, including a malformed/missing export or unsafe
            # package inventory. Never leave an interrupted validation green.
            row["validation_error"] = str(error)
            row["exit_code"] = 1
        # A successfully executed prefix is not completion: an interruption or
        # missing tool at the next step must never leave a passing manifest.
        results["passed"] = index + 1 == len(plan) and all(item["exit_code"] == 0 for item in results["gates"])
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        if row["exit_code"] != 0:
            print(f"FAILED: {gate}; see {log} and {output / 'results.json'}", flush=True)
            return 1
    print(f"Requested gates passed; no release promotion performed. Evidence: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"release gate error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
