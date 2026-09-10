#!/usr/bin/env python3
"""Execute the fixed CPU protocol and retain its invocation and source provenance."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCES = [
    "examples/greater_than_train.rs",
    "examples/mechanistic/greater_than_training/support.rs",
    "examples/mechanistic/greater_than_training/protocol.json",
    "examples/mechanistic/greater_than/fixture.json",
    "src/models/gpt2.rs", "src/models/gpt2/model.rs", "src/models/gpt2/checkpoint.rs",
    "src/intervention.rs", "src/training.rs", "Cargo.toml", "Cargo.lock",
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

before = {name: sha(ROOT / name) for name in SOURCES}
for name in SOURCES:
    destination = HERE / "executed-source" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, destination)
command = ["cargo", "run", "--release", "--locked", "--offline", "--features", "autodiff", "--example", "greater_than_train"]
started_utc = datetime.now(timezone.utc).isoformat()
started = time.monotonic()
with (HERE / "stdout.log").open("x") as stdout, (HERE / "stderr.log").open("x") as stderr:
    completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
execution = {
    "command": command, "cwd": str(ROOT), "started_utc": started_utc,
    "exit_code": completed.returncode, "wall_seconds_including_cargo": time.monotonic() - started,
    "source_sha256": before,
    "source_unchanged_during_execution": before == {name: sha(ROOT / name) for name in SOURCES},
    "binary_sha256": sha(ROOT / "target/release/examples/greater_than_train"),
    "execution_wrapper_sha256": sha(Path(__file__)),
    "stdout_sha256": sha(HERE / "stdout.log"), "stderr_sha256": sha(HERE / "stderr.log"),
}
if completed.returncode == 0:
    result = ROOT / "target/mechanistic/greater_than_training/full/result.json"
    shutil.copyfile(result, HERE / "result.json")
    execution["result_sha256"] = sha(result)
(HERE / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
print(json.dumps(execution, indent=2))
raise SystemExit(completed.returncode)
