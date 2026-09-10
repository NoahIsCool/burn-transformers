#!/usr/bin/env python3
"""One-off v0.2 validation: print the plan by default; execute only with --run.

Run from an authorized local session after the source and all four paper studies
are ready. No subprocess uses a shell, installs dependencies, or prepares weights.
Existing output directories are never overwritten; choose another --output for
a retry so failed records remain available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


REPO = Path("/Users/nover/repos/transformers")
DEFAULT_OUTPUT = REPO / "acceptance/measurements/v0_2-interface-validation-20260910"
FEATURES = "std,flex,tokenizers,chat-templates,autodiff"
COMPILE = ["--release", "--locked", "--offline"]
SOURCE_FEATURES = ["--no-default-features", "--features", FEATURES]
EXAMPLES = [
    ("activation_workbench", []),
    ("attention_head_attribution", []),
    ("custom_composition", []),
    ("t5_classifier", []),
    ("mla_encoder_decoder", []),
    ("qwen_prefix", ["--random"]),
    ("grokking_fourier", []),
    ("clock_and_pizza", []),
    ("tuned_lens", []),
    ("greater_than", []),
]
REPORTS = {
    "clock_and_pizza": "target/mechanistic/clock_and_pizza/rust_results.json",
    "tuned_lens": "target/mechanistic/tuned_lens/rust_result.json",
    "greater_than": "target/mechanistic/greater_than/result.json",
}
ARTIFACT_RECORDS = {
    "grokking": "examples/mechanistic/grokking_fourier.result.json",
    "clock_and_pizza": "examples/mechanistic/clock_and_pizza/result.json",
    "tuned_lens": "examples/mechanistic/tuned_lens/result.json",
    "greater_than": "examples/mechanistic/greater_than/result.json",
    "qwen_prefix": "acceptance/measurements/qwen-prefix-v02-20260910.json",
    "qwen_prefix_replay": "acceptance/measurements/qwen-prefix-v02-20260910/replay_verification.json",
    "flan_t5_small": "acceptance/measurements/v0_2-flan-t5-small-flex-20260910/manifest.json",
}
SCOPE = {
    "purpose": "One fixed local v0.2 source/interface validation and prepared CPU example replay.",
    "rust_test_features": FEATURES,
    "clippy_features": "all features, compile/lint only",
    "examples": "Six download-free examples and four already-prepared mechanistic studies.",
    "fresh_build_meaning": "Direct binaries are hashed after a successful cargo build --examples; Cargo may reuse its validated build cache.",
    "artifact_records": "Separate historical study/checkpoint evidence, linked and hashed for context; not relabeled as fresh checks.",
    "inventory": "Source, Cargo metadata, fixtures and structured protocols; historical measurement records are included separately because Python tests consume them as fixtures.",
    "exclusions": "Markdown, generated example results/figures, this validation output, other v0.2 validation outputs, target/, root models/, .git/, Python bytecode. This is not a hermetic dependency/toolchain attestation.",
    "qualification": "No coverage measurement, accelerator runtime validation, package qualification, training validation, or release/support promotion is performed or claimed.",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> dict:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    result = {"sha256": sha.hexdigest(), "bytes": path.stat().st_size}
    if path.is_symlink():
        result["symlink_target"] = os.readlink(path)
    return result


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def excluded(path: Path, output: Path) -> bool:
    relative = path.relative_to(REPO)
    if path == output or output in path.parents:
        return True
    if any(part in {"target", ".git", "__pycache__", "node_modules"} for part in relative.parts):
        return True
    if relative.parts[0] == "models" or path.suffix.lower() in {".md", ".markdown", ".pyc", ".pyo"}:
        return True
    if any(part.startswith("v0_2-interface-validation-") for part in relative.parts):
        return True
    if relative.parts[:2] == ("examples", "mechanistic"):
        if path.suffix.lower() in {".png", ".svg", ".pdf", ".html"}:
            return True
        if path.name in {"result.json", "results.json"} or path.name.endswith((".result.json", "_result.json", "_results.json")):
            return True
    return False


def inventory(output: Path) -> dict:
    files = set()
    for name in ("src", "tests", "examples", "scripts", "benches", ".cargo", "acceptance"):
        root = REPO / name
        if root.exists():
            # Do not traverse ignored build directories, including scripts/target.
            for directory, children, names in os.walk(root, followlinks=False):
                children[:] = [child for child in children if not excluded(Path(directory) / child, output)]
                files.update(Path(directory) / name for name in names)
    for pattern in ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain*", "requirements*.txt", "pyproject.toml"):
        files.update(REPO.glob(pattern))
    groups = {"source_and_protocol_inputs": {}, "historical_measurement_test_fixtures": {}}
    for path in sorted(files):
        if not path.is_file() or excluded(path, output):
            continue
        relative = path.relative_to(REPO).as_posix()
        group = "historical_measurement_test_fixtures" if relative.startswith("acceptance/measurements/") else "source_and_protocol_inputs"
        groups[group][relative] = digest(path)
    # These two documents are existence-checked by Python tests. Their text is
    # deliberately outside the validation snapshot, like other Markdown.
    groups["documentation_presence"] = {
        relative: {"exists": (REPO / relative).is_file()}
        for relative in ("scripts/COVERAGE_EVIDENCE.md", "scripts/WHOLE_FILE_COVERAGE.md")
    }
    groups["runner"] = {str(Path(__file__).resolve()): digest(Path(__file__))}
    encoded = json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": groups}


def changes(before: dict, after: dict) -> dict:
    result = {}
    for group in before["files"]:
        old, new = before["files"][group], after["files"][group]
        difference = {
            "added": sorted(new.keys() - old.keys()),
            "removed": sorted(old.keys() - new.keys()),
            "modified": sorted(key for key in old.keys() & new.keys() if old[key] != new[key]),
        }
        if any(difference.values()):
            result[group] = difference
    return result


def plan() -> list[dict]:
    steps = [
        ("git_revision", ["git", "rev-parse", "HEAD"], None),
        ("git_status", ["git", "status", "--porcelain=v1", "--untracked-files=all"], None),
        ("rustc_version", ["rustc", "--version", "--verbose"], None),
        ("cargo_version", ["cargo", "--version", "--verbose"], None),
        # rustup run does not install a missing toolchain without --install.
        ("msrv_version", ["rustup", "run", "1.92.0", "rustc", "--version", "--verbose"], "msrv"),
        ("python_version", [sys.executable, "--version"], None),
        ("paper_python_versions", [sys.executable, "-c", "import json,sys,torch,safetensors,numpy; print(json.dumps({'executable':sys.executable,'python':sys.version,'torch':str(torch.__version__),'safetensors':safetensors.__version__,'numpy':numpy.__version__},sort_keys=True))"], None),
        ("rust_all_targets", ["cargo", "test", *COMPILE, *SOURCE_FEATURES, "--all-targets"], "rust"),
        ("rust_doctests", ["cargo", "test", *COMPILE, *SOURCE_FEATURES, "--doc"], "rust"),
        ("clippy_all_features", ["cargo", "clippy", *COMPILE, "--all-targets", "--all-features", "--", "-D", "warnings"], None),
        ("rustfmt", ["cargo", "fmt", "--check"], None),
        ("minimal_features", ["cargo", "check", *COMPILE, "--no-default-features"], None),
        ("std_features", ["cargo", "check", *COMPILE, "--no-default-features", "--features", "std"], None),
        ("rustdoc", ["cargo", "doc", *COMPILE, *SOURCE_FEATURES, "--no-deps"], None),
        ("msrv_all_targets", ["rustup", "run", "1.92.0", "cargo", "check", *COMPILE, *SOURCE_FEATURES, "--all-targets"], None),
        ("python_unittest", [sys.executable, "-m", "unittest", "discover", "-s", "scripts/tests", "-v"], "python"),
        ("tuned_lens_python_unittest", [sys.executable, "-m", "unittest", "discover", "-s", "examples/mechanistic/tuned_lens", "-p", "*_tests.py", "-v"], "python"),
        ("greater_than_python_unittest", [sys.executable, "-m", "unittest", "discover", "-s", "examples/mechanistic/greater_than", "-p", "test_*.py", "-v"], "python"),
        ("build_examples", ["cargo", "build", *COMPILE, *SOURCE_FEATURES, "--examples"], None),
    ]
    steps.extend((f"example_{name}", [str(REPO / "target/release/examples" / name), *arguments], "example") for name, arguments in EXAMPLES)
    return [{"name": name, "argv": argv, "parse": parser} for name, argv, parser in steps]


def test_counts(kind: str | None, stdout: str, stderr: str, required: bool = True) -> dict | None:
    if kind == "rust":
        summaries = []
        pattern = r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out;"
        for match in re.finditer(pattern, stdout, re.MULTILINE):
            summaries.append(dict(status=match[1], **dict(zip(("passed", "failed", "ignored", "measured", "filtered_out"), map(int, match.groups()[1:])))))
        if not summaries:
            if required:
                raise RuntimeError("Successful Rust test command produced no libtest summary")
            return {"format": "Rust libtest", "totals": None, "suite_summaries": []}
        totals = {key: sum(summary[key] for summary in summaries) for key in ("passed", "failed", "ignored", "measured", "filtered_out")}
        if required and (totals["failed"] or any(summary["status"] != "ok" for summary in summaries)):
            raise RuntimeError("Rust test output reports failures")
        return {"format": "Rust libtest", "totals": totals, "suite_summaries": summaries}
    if kind == "python":
        matches = list(re.finditer(r"^Ran (\d+) tests? in ([0-9.]+)s$", stderr, re.MULTILINE))
        if not matches:
            if required:
                raise RuntimeError("Successful unittest command produced no test count")
            return {"format": "Python unittest", "tests_run": None}
        last = matches[-1]
        status = re.search(r"^(OK|FAILED)(?: \(([^\n]*)\))?$", stderr[last.end():], re.MULTILINE)
        if status is None or required and status[1] != "OK":
            raise RuntimeError("Python unittest final summary is missing or failed")
        details = {key.strip(): value for key, value in re.findall(r"([a-z ]+)=(\d+)", status[2] or "")}
        counts = {name.replace(" ", "_"): int(details.get(name, 0)) for name in ("skipped", "expected failures", "unexpected successes", "failures", "errors")}
        if required and not int(last[1]):
            raise RuntimeError("Python unittest discovery ran zero tests")
        return {"format": "Python unittest", "status": status[1], "tests_run": int(last[1]), "normal_passed": int(last[1]) - sum(counts.values()), **counts, "reported_runtime_seconds": float(last[2])}
    return None


def require_inputs() -> None:
    if os.environ.get("CARGO_BUILD_TARGET"):
        raise RuntimeError("Unset CARGO_BUILD_TARGET: this run executes native host binaries")
    required = [f"examples/{name}.rs" for name, _ in EXAMPLES]
    required += list(ARTIFACT_RECORDS.values())
    required += ["target/mechanistic/grokking.safetensors"]
    required += [f"target/mechanistic/clock_and_pizza/{name}" for name in ("pizza.safetensors", "clock.safetensors", "reference.json")]
    for study in ("tuned_lens", "greater_than"):
        required += [f"target/mechanistic/{study}/{name}" for name in ("preparation.json", "fixture.json", "reference.json", "reference.safetensors", "model/config.json", "model/model.safetensors", "model/tokenizer.json")]
    required += ["target/mechanistic/tuned_lens/lens.safetensors"]
    missing = [name for name in required if not (REPO / name).is_file()]
    if missing:
        raise RuntimeError("Prepared studies and accepted records must exist before validation: " + ", ".join(missing))


def execute(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "replayed_results").mkdir()
    shutil.copyfile(__file__, output / "validation_runner.py")
    environment = os.environ.copy()
    overrides = {"CARGO_TARGET_DIR": str(REPO / "target"), "CARGO_NET_OFFLINE": "true", "CARGO_TERM_COLOR": "never", "PYTHONDONTWRITEBYTECODE": "1"}
    environment.update(overrides)
    manifest = {
        "schema_version": 1, "status": "running", "started_at_utc": now(),
        "cwd": str(REPO), "host": platform.platform(), "scope": SCOPE,
        "runner": {"path": "validation_runner.py", **digest(output / "validation_runner.py")},
        "environment_overrides": overrides,
        "compiler_environment": {name: environment.get(name) for name in ("RUSTUP_TOOLCHAIN", "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_BUILD_TARGET")},
        "plan": plan(), "commands": [], "separate_artifact_records": {},
    }
    before = None
    child = None
    started = time.monotonic()
    returncode = 1
    try:
        before = inventory(output)
        write_json(output / "source-inventory-before.json", before)
        manifest["input_snapshot_sha256"] = before["sha256"]
        write_json(output / "manifest.json", manifest)
        require_inputs()
        for label, relative in ARTIFACT_RECORDS.items():
            path = REPO / relative
            manifest["separate_artifact_records"][label] = {"repository_path": relative, "link_from_record": os.path.relpath(path, output), **digest(path)}
        binaries = {}
        for index, step in enumerate(manifest["plan"], start=1):
            row = {**step, "status": "running", "started_at_utc": now()}
            manifest["commands"].append(row)
            write_json(output / "manifest.json", manifest)
            print(f"[{index}/{len(manifest['plan'])}] {step['name']}", flush=True)
            stdout_path = output / "logs" / f"{index:02d}-{step['name']}.stdout.log"
            stderr_path = output / "logs" / f"{index:02d}-{step['name']}.stderr.log"
            example = step["name"].removeprefix("example_") if step["parse"] == "example" else None
            previous_report_mtime = None
            if example:
                if digest(Path(step["argv"][0])) != binaries[example]:
                    raise RuntimeError(f"Example binary changed after build: {example}")
                row["executable"] = binaries[example]
                if example in REPORTS and (REPO / REPORTS[example]).exists():
                    previous_report_mtime = (REPO / REPORTS[example]).stat().st_mtime_ns
            command_started = time.monotonic()
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                try:
                    launch_argv = step["argv"]
                    if example and sys.platform == "darwin":
                        launch_argv = ["/usr/bin/time", "-l", *launch_argv]
                    row["launch_argv"] = launch_argv
                    child = subprocess.Popen(launch_argv, cwd=REPO, env=environment, stdout=stdout, stderr=stderr, shell=False, start_new_session=True)
                    row["returncode"] = child.wait()
                except BaseException:
                    if child is not None and child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                    raise
                finally:
                    row["runtime_seconds"] = time.monotonic() - command_started
                    row["finished_at_utc"] = now()
                    child = None
            row["stdout"] = {"path": stdout_path.relative_to(output).as_posix(), **digest(stdout_path)}
            row["stderr"] = {"path": stderr_path.relative_to(output).as_posix(), **digest(stderr_path)}
            row["status"] = "passed" if row["returncode"] == 0 else "failed"
            current = inventory(output)
            row["input_snapshot_sha256_after"] = current["sha256"]
            difference = changes(before, current)
            if difference:
                row["input_changes"] = difference
                raise RuntimeError(f"Source/protocol/test-fixture inputs changed during {step['name']}")
            stdout_text, stderr_text = stdout_path.read_text(errors="replace"), stderr_path.read_text(errors="replace")
            row["test_counts"] = test_counts(step["parse"], stdout_text, stderr_text, required=row["returncode"] == 0)
            if example and sys.platform == "darwin" and row["returncode"] == 0:
                peak = re.search(r"^\s*(\d+)\s+maximum resident set size\s*$", stderr_text, re.MULTILINE)
                if peak is None or "Operation not permitted" in stderr_text:
                    raise RuntimeError(f"Example resource measurement failed: {example}")
                row["peak_process_rss_bytes"] = int(peak[1])
                row["memory_method"] = "macOS /usr/bin/time -l child maximum resident set size; bytes, not system footprint"
            if row["returncode"]:
                raise RuntimeError(f"{step['name']} exited with status {row['returncode']}; see its preserved logs")
            if step["parse"] == "msrv" and not re.search(r"^release: 1\.92\.0$", stdout_text, re.MULTILINE):
                raise RuntimeError("The 1.92.0 toolchain did not report exactly Rust 1.92.0")
            if step["name"] == "build_examples":
                binaries = {name: digest(REPO / "target/release/examples" / name) for name, _ in EXAMPLES}
                manifest["example_binaries_after_build"] = binaries
            if example and digest(Path(step["argv"][0])) != binaries[example]:
                raise RuntimeError(f"Example binary changed during execution: {example}")
            if example in REPORTS:
                report = REPO / REPORTS[example]
                if report.stat().st_mtime_ns == previous_report_mtime:
                    raise RuntimeError(f"Example did not refresh its output report: {example}")
                json.loads(report.read_text())
                copied = output / "replayed_results" / f"{example}.json"
                shutil.copyfile(report, copied)
                row["fresh_result"] = {"path": copied.relative_to(output).as_posix(), "original_path": REPORTS[example], **digest(copied)}
            write_json(output / "manifest.json", manifest)
        manifest["status"] = "passed"
        returncode = 0
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        if manifest["commands"]:
            last = manifest["commands"][-1]
            last["status"] = "failed"
            for stream in ("stdout", "stderr"):
                path = output / "logs" / f"{len(manifest['commands']):02d}-{last['name']}.{stream}.log"
                if path.exists():
                    last[stream] = {"path": path.relative_to(output).as_posix(), **digest(path)}
        (output / "failure.log").write_text(traceback.format_exc())
        returncode = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        try:
            after = inventory(output)
            write_json(output / "source-inventory-after.json", after)
            if before is not None:
                manifest["input_changes"] = changes(before, after)
                manifest["inputs_unchanged"] = not manifest["input_changes"]
                if manifest["input_changes"]:
                    manifest["status"] = "failed"
                    returncode = 1
        except Exception as error:
            manifest["final_inventory_error"] = str(error)
            manifest["status"] = "failed"
            returncode = 1
        manifest["finished_at_utc"] = now()
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["completed_commands"] = sum(row["status"] == "passed" for row in manifest["commands"])
        manifest["test_counts_by_command"] = {row["name"]: row["test_counts"] for row in manifest["commands"] if row.get("test_counts") is not None}
        manifest["exit_code"] = returncode
        write_json(output / "manifest.json", manifest)
        print(f"{manifest['status']}: {output / 'manifest.json'}", flush=True)
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the fixed checks; omit to print the plan without running commands")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="new evidence directory; an existing directory is never overwritten")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"cwd": str(REPO), "output": str(args.output.resolve()), "scope": SCOPE, "commands": plan()}, indent=2))
        return 0
    return execute(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
