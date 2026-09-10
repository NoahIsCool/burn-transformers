"""Create-only, CPU-only execution records for the bounded v0.1 fuzz refresh.

The sanitizer targets run directly after cargo-fuzz builds them, retaining their
native exit codes and bounding each complete campaign independently. Historical
corpora and artifacts are never opened for writing. This development script is
not part of the library or its packaged runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import time


TARGETS = (
    ("configuration", 483013554),
    ("index_artifacts", 20260909),
    ("masks_positions", 20260909),
    ("cache_transactions", 20260909),
    ("generation", 20260909),
)
OVERRIDES = (
    "ASAN_OPTIONS", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER", "CARGO_TARGET_DIR", "RUSTC", "LLVM_PROFILE_FILE",
)
SYMBOLIZER = Path("/opt/homebrew/opt/llvm/bin/llvm-symbolizer")


def utc() -> str:
    """Return a timezone-explicit timestamp without altering wall-clock configuration."""
    return datetime.now(timezone.utc).isoformat()


def file_identity(path: Path) -> dict:
    """Hash a regular file incrementally and reject a size change during readback."""
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed while hashing: {path}")
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


def inventory(root: Path) -> dict:
    """Record a deterministic relative-path inventory, including an empty directory."""
    if not root.exists():
        return {"present": False, "files": {}, "total_bytes": 0}
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"unexpected symlink: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = file_identity(path)
    return {"present": True, "files": files, "total_bytes": sum(x["bytes"] for x in files.values())}


def source_inventory(crate: Path) -> dict:
    """Hash the tracked compilation inputs relevant to the runtime and fuzz harnesses."""
    scopes = [
        "Cargo.toml", "Cargo.lock", "src", "fuzz/Cargo.toml", "fuzz/Cargo.lock",
        "fuzz/src", "fuzz/fuzz_targets", "fuzz/seeds",
    ]
    output = subprocess.check_output(["git", "ls-files", "-z", "--", *scopes], cwd=crate)
    paths = sorted(value.decode() for value in output.split(b"\0") if value)
    if not paths:
        raise ValueError("empty tracked compilation-input inventory")
    return {path: file_identity(crate / path) for path in paths}


def write_json(path: Path, value: dict) -> None:
    """Persist a new record without overwriting an earlier successful or failed run."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def execute(label: str, command: list[str], cwd: Path, env: dict, out: Path, timeout: int) -> dict:
    """Capture a native process result, retaining logs even on errors or group timeout."""
    log = out / f"{label}.log"
    started = utc()
    clock = time.monotonic()
    timed_out = False
    print(f"START {label}: {json.dumps(command)}", flush=True)
    with log.open("xb") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    record = {
        "label": label, "command": command, "cwd": str(cwd), "started_utc": started,
        "ended_utc": utc(), "elapsed_seconds": time.monotonic() - clock,
        "native_returncode": returncode, "wall_timeout_seconds": timeout,
        "timed_out": timed_out, "log_path": str(log), "log": file_identity(log),
    }
    write_json(out / f"{label}.command.json", record)
    print(f"END {label}: exit={returncode}, timeout={timed_out}", flush=True)
    return record


def campaign_result(record: dict, text: str, artifacts: dict) -> dict:
    """Require completed input accounting rather than equating an early zero exit with success."""
    counts = re.findall(r"^stat::number_of_executed_units:\s*(\d+)\s*$", text, re.MULTILINE)
    statistics = dict(re.findall(r"^(stat::[a-z_]+):\s*(.*?)\s*$", text, re.MULTILINE))
    warnings = [line for line in text.splitlines()
                if re.search(r"\b(?:WARNING|ERROR|SUMMARY):", line)]
    failures = []
    if record["native_returncode"] != 0 or record["timed_out"]:
        failures.append("native process failed or exceeded its wall deadline")
    if counts != ["10000"]:
        failures.append("expected exactly one final 10000-unit counter")
    if warnings:
        failures.append("diagnostics require review")
    if artifacts["files"]:
        failures.append("failure artifact directory is not empty")
    return {"passed": not failures, "failures": failures, "final_statistics": statistics,
            "executed_units": int(counts[0]) if len(counts) == 1 else None,
            "diagnostics": warnings, "artifacts": artifacts}


def main() -> int:
    """Run deterministic checks, one instrumented build and five serialized bounded campaigns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_dir() or any(out.iterdir()):
        raise ValueError("output must be a new, empty result directory")
    crate = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    inherited = {key: env.get(key) for key in OVERRIDES}
    if any(value for value in inherited.values()):
        raise ValueError(f"compiler/sanitizer overrides require explicit review: {inherited}")
    env.update({"CARGO_NET_OFFLINE": "true", "ASAN_SYMBOLIZER_PATH": str(SYMBOLIZER)})
    free = shutil.disk_usage(crate).free
    if free < 20 * 1024 ** 3:
        raise ValueError(f"20 GiB preflight failed: {free} bytes free; request external fallback")
    before = source_inventory(crate)
    report = {
        "schema_version": 1, "status": "running", "started_utc": utc(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=crate).decode().strip(),
        "source_before": before, "dependency_closure_verified": False,
        "disk_preflight_free_bytes": free, "target_directory": str(crate / "fuzz/target"),
        "runner": file_identity(Path(__file__)), "inherited_overrides": inherited,
        "explicit_environment": {key: env[key] for key in ["CARGO_NET_OFFLINE", "ASAN_SYMBOLIZER_PATH"]},
        "build_std": False, "sanitizer": "address",
        "native_campaign_asan_options": "detect_odr_violation=0",
        "historical_corpora_imported": False, "commands": [], "campaigns": {},
    }
    write_json(out / "preflight.json", report)
    try:
        for label, command in [
            ("rustc", ["rustc", "+nightly", "--version", "--verbose"]),
            ("cargo-fuzz", [str(Path.home() / ".cargo/bin/cargo-fuzz"), "--version"]),
            ("symbolizer", [str(SYMBOLIZER), "--version"]),
            ("deterministic", ["cargo", "+nightly", "test", "--manifest-path", "fuzz/Cargo.toml", "--locked", "--lib"]),
            ("clippy", ["cargo", "+nightly", "clippy", "--manifest-path", "fuzz/Cargo.toml", "--locked", "--lib", "--tests", "--", "-D", "warnings"]),
            ("asan-build", ["cargo", "+nightly", "fuzz", "build", "--sanitizer", "address", "--target", "aarch64-apple-darwin", "--verbose"]),
        ]:
            record = execute(label, command, crate, env, out, 1800)
            report["commands"].append(record)
            if record["native_returncode"] != 0 or record["timed_out"]:
                raise RuntimeError(f"{label} failed; see retained native result")
        report["tools"] = {
            "cargo_fuzz": file_identity(Path.home() / ".cargo/bin/cargo-fuzz"),
            "symbolizer": file_identity(SYMBOLIZER),
            "rustc": file_identity(Path.home() / ".rustup/toolchains/nightly-aarch64-apple-darwin/bin/rustc"),
        }
        if source_inventory(crate) != before:
            raise RuntimeError("compilation inputs changed during build")
        native_env = env | {"ASAN_OPTIONS": "detect_odr_violation=0"}
        for target, seed in TARGETS:
            corpus = out / "corpus" / target
            artifacts = out / "artifacts" / target
            corpus.mkdir(parents=True, exist_ok=False)
            artifacts.mkdir(parents=True, exist_ok=False)
            executable = crate / "fuzz/target/aarch64-apple-darwin/release" / target
            binary_before = file_identity(executable)
            command = [str(executable), str(corpus), f"fuzz/seeds/{target}",
                       f"-artifact_prefix={artifacts}/", "-runs=10000", "-max_len=4096",
                       "-timeout=10", "-rss_limit_mb=2048", f"-seed={seed}", "-print_final_stats=1"]
            record = execute(target, command, crate, native_env, out, 900)
            artifact_inventory = inventory(artifacts)
            result = campaign_result(record, Path(record["log_path"]).read_text(errors="replace"), artifact_inventory)
            result.update({"command": record, "seed": seed, "executable_path": str(executable),
                           "executable_before": binary_before, "executable_after": file_identity(executable),
                           "initial_corpus": {"present": True, "files": {}, "total_bytes": 0},
                           "final_corpus": inventory(corpus)})
            if result["executable_after"] != binary_before:
                result["passed"] = False
                result["failures"].append("instrumented executable changed during execution")
            report["campaigns"][target] = result
            write_json(out / f"{target}.result.json", result)
            if not result["passed"]:
                raise RuntimeError(f"{target} failed; artifacts and diagnostics retained")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["source_after"] = source_inventory(crate)
        report["source_unchanged"] = report["source_after"] == before
        if not report["source_unchanged"]:
            report["status"] = "failed"
        report["ended_utc"] = utc()
        report["disk_final_free_bytes"] = shutil.disk_usage(crate).free
        write_json(out / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
