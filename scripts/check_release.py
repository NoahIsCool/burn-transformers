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

from coverage_provenance import capture_sources, fingerprint_file, require_same_sources


CRATE = Path(__file__).resolve().parents[1]
FEATURES = "std,flex,tokenizers,chat-templates,autodiff"
GATES = ("source", "msrv", "coverage", "dependencies", "accelerator-compile", "package")
# These are maintenance notices, not vulnerability waivers. New IDs fail.
MAINTENANCE_NOTICES = ("RUSTSEC-2025-0141", "RUSTSEC-2024-0436")
RUNNER_INPUTS = (Path(__file__).resolve(), Path(__file__).resolve().with_name("coverage_provenance.py"))


def coverage_tool_versions(command: list[str], crate: Path, environment: dict) -> list[dict]:
    """Record the selected compiler/Cargo/exporter versions, not binary attestation."""

    if len(command) < 3 or command[0] != "cargo" or not command[1].startswith("+") or command[2] != "llvm-cov":
        raise ValueError("coverage command has no explicit Cargo toolchain")
    toolchain = command[1]
    commands = (["rustc", toolchain, "--version", "--verbose"],
                ["cargo", toolchain, "--version"], ["cargo", toolchain, "llvm-cov", "--version"])
    versions = []
    for invocation in commands:
        output = subprocess.check_output(invocation, cwd=crate, env=environment,
                                         stderr=subprocess.STDOUT, text=True, timeout=30)
        require_clean_coverage_log(output)
        if not output.strip():
            raise ValueError(f"empty coverage tool identity: {invocation[0]}")
        versions.append({"command": invocation, "output": output})
    return versions


def capture_coverage_context(crate: Path, command: list[str], environment: dict) -> dict:
    """Bind declared source and runner bytes, without claiming compiler closure.

    Only explicit compiler-control environment variables are retained. This is
    an ordinary local reproducibility record, not an attestation that wrappers,
    build scripts, procedural macros or inherited Cargo settings are hermetic.
    """

    return {
        "schema_version": 1,
        "sources": capture_sources(crate),
        "runner_inputs": {path.name: fingerprint_file(path) for path in RUNNER_INPUTS},
        "tool_versions": coverage_tool_versions(command, crate, environment),
        "command": command,
        "working_directory": str(crate.resolve()),
        "compiler_environment": {name: environment[name] for name in (
            "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_INCREMENTAL",
            "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR", "RUSTC", "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
        ) if name in environment},
        "compiler_dependency_closure_verified": False,
        "production_only_exclusions_qualified": False,
    }


def require_same_coverage_context(before: dict, after: dict) -> None:
    """Reject input edits before consulting any native coverage counters."""

    require_same_sources(before["sources"], after["sources"])
    for key in ("runner_inputs", "tool_versions", "command", "working_directory", "compiler_environment"):
        if before[key] != after[key]:
            raise ValueError(f"coverage context changed: {key}")


def write_evidence(path: Path, value: dict) -> None:
    """Retain one human-readable JSON observation inside a fresh run directory."""

    path.write_text(json.dumps(value, indent=2) + "\n")


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


def coverage_source_relative(filename: str, crate: Path) -> str | None:
    """Resolve LLVM filenames against the command cwd, retaining only src inputs."""

    if not isinstance(filename, str) or not filename or "\0" in filename:
        raise ValueError("invalid coverage filename")
    root = crate.resolve()
    path = Path(filename)
    path = (path if path.is_absolute() else root / path).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root / "src") else None


def require_bound_coverage_sources(report: dict, snapshot: dict, crate: Path) -> None:
    """Reject demonstrably stale src entries without claiming complete closure."""

    for data in report["data"]:
        for entry in data["files"]:
            relative = coverage_source_relative(entry["filename"], crate)
            if relative is not None and (not relative.endswith(".rs") or relative not in snapshot["files"]):
                raise ValueError(f"coverage source is not a captured Rust input: {relative}")


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
            relative = coverage_source_relative(entry["filename"], crate)
            if relative is None:
                continue
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
        if path.parts[:2] == ("scripts", "coverage-layout"):
            raise ValueError(f"repository-only standalone audit tool in runtime package: {raw}")
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
    if len(args.gate) != len(set(args.gate)):
        raise ValueError("duplicate gate groups would overwrite their evidence")
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
        row = {"gate": gate, "command": command, "exit_code": 1,
               "process_exit_code": None, "log": log.name}
        results["gates"].append(row)
        try:
            if gate == "coverage":
                before = capture_coverage_context(CRATE, command, environment)
                before_path = output / "coverage-inputs-before.json"
                write_evidence(before_path, before)
                row["input_evidence"] = {"before": before_path.name,
                                         "sources_unchanged": False,
                                         "production_only_exclusions_qualified": False}
            with log.open("x") as stream:
                completed = subprocess.run(command, cwd=CRATE, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
            row["process_exit_code"] = row["exit_code"] = completed.returncode
            if gate == "coverage":
                # Capture terminal observations even when the native process
                # fails. Failure never adopts counters or a passing manifest.
                after = capture_coverage_context(CRATE, command, environment)
                after_path = output / "coverage-inputs-after.json"
                write_evidence(after_path, after)
                row["input_evidence"]["after"] = after_path.name
                row["log_identity"] = fingerprint_file(log, maximum_bytes=512 * 1024 * 1024)
                report_path = output / "coverage.json"
                if report_path.exists() or report_path.is_symlink():
                    row["report_identity"] = fingerprint_file(report_path, maximum_bytes=2 * 1024 * 1024 * 1024)
                require_same_coverage_context(before, after)
                row["input_evidence"]["sources_unchanged"] = True
            if completed.returncode == 0 and gate == "coverage":
                require_clean_coverage_log(log.read_text())
                report = json.loads((output / "coverage.json").read_text())
                require_bound_coverage_sources(report, before["sources"], CRATE)
                row["coverage"] = coverage_summary(report, CRATE)
                # Detect ordinary edits during parsing as well as during the
                # native command. The hashes do not authenticate old binaries.
                if row["report_identity"] != fingerprint_file(report_path, maximum_bytes=2 * 1024 * 1024 * 1024):
                    raise ValueError("coverage export changed during validation")
                if row["log_identity"] != fingerprint_file(log, maximum_bytes=512 * 1024 * 1024):
                    raise ValueError("coverage log changed during validation")
                if not row["coverage"]["passed"]:
                    row["exit_code"] = 1
            if completed.returncode == 0 and gate == "package" and "--list" in command:
                row["package_files"] = validate_package_listing(log.read_text())
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            # Preserve the distinction between a successful native process and
            # invalid evidence, including a malformed/missing export or unsafe
            # package inventory. Never leave an interrupted validation green.
            row["validation_error"] = str(error)
            row["exit_code"] = 1
            row.pop("coverage", None)
        row["seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
        # A successfully executed prefix is not completion: an interruption or
        # missing tool at the next step must never leave a passing manifest.
        results["passed"] = index + 1 == len(plan) and all(item["exit_code"] == 0 for item in results["gates"])
        write_evidence(output / "results.json", results)
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
