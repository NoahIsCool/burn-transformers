#!/usr/bin/env python3
"""Repeat the preserved Qwen prefix comparison using authenticated local weights."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time


def run(command, cwd):
    started = time.monotonic()
    process = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    record = {
        "command": command,
        "cwd": str(cwd),
        "exit_code": process.returncode,
        "wall_seconds": time.monotonic() - started,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    if process.returncode:
        raise RuntimeError(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Local pinned Qwen3.5-0.8B directory")
    parser.add_argument("--output-dir", type=Path, help="New output directory; defaults to a temporary directory")
    arguments = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    repository = bundle.parents[2]
    model = arguments.model_dir.resolve()
    source = json.loads((bundle / "source_metadata.json").read_text())
    for expected in source["files"]:
        path = model / expected["file"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(16 * 1024 * 1024):
                digest.update(chunk)
        if path.stat().st_size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Checkpoint identity mismatch: {path}")

    if arguments.output_dir:
        output = arguments.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="qwen-prefix-replay-"))
    print(f"Evidence directory: {output}", flush=True)
    build = run([
        "cargo", "build", "--release", "--locked", "--offline",
        "--example", "qwen_prefix", "--message-format=json",
    ], repository)
    libraries = {}
    example = None
    for line in build["stdout"].splitlines():
        item = json.loads(line)
        if item.get("reason") != "compiler-artifact":
            continue
        name = item["target"]["name"]
        if name in {"transformers", "burn", "serde_json"}:
            matches = [path for path in item["filenames"] if path.endswith(".rlib")]
            if matches:
                libraries[name] = matches[0]
        if name == "qwen_prefix":
            example = item.get("executable")
    if set(libraries) != {"transformers", "burn", "serde_json"} or not example:
        raise RuntimeError("Cargo did not report the required libraries and example")
    (output / "cargo_build.json").write_text(json.dumps(build, indent=2) + "\n")

    example_run = run([example, str(model), "2"], repository)
    (output / "example_execution.json").write_text(json.dumps(example_run, indent=2) + "\n")
    executable = output / "compare_prefix"
    command = [
        "rustc", "--edition=2024", "-C", "opt-level=3", str(bundle / "compare_prefix.rs"),
        "-L", f"dependency={Path(libraries['transformers']).parent}", "-o", str(executable),
    ]
    for name, library in libraries.items():
        command.extend(["--extern", f"{name}={library}"])
    compilation = run(command, repository)
    (output / "harness_build.json").write_text(json.dumps(compilation, indent=2) + "\n")
    execution = run([str(executable), str(model)], repository)
    (output / "comparison_execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    report = json.loads(execution["stdout"])
    (output / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Exact equality: {report['hidden_exact_equal']}; maximum error: {report['hidden_max_absolute_error']}")
    print(f"Read {report['source_payload_bytes']} prefix payload bytes; captured {report['capture_shape']} {report['capture_dtype']}")


if __name__ == "__main__":
    main()
