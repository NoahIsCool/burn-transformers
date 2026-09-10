#!/usr/bin/env python3
"""Record a clean native replay, its peak memory, binary hash, and source inventory.

Build the example first. This wrapper uses Python's process resource accounting,
which also works where macOS `time -l` cannot read the restricted clock sysctl.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory():
    files = list((REPOSITORY / "src").rglob("*.rs"))
    files += [REPOSITORY / name for name in ("Cargo.toml", "Cargo.lock", "examples/tuned_lens_train.rs")]
    files += list(HERE.glob("*.rs")) + [HERE / "manifest.json"]
    return {str(path.relative_to(REPOSITORY)): sha(path) for path in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=REPOSITORY / "target/release/examples/tuned_lens_train")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "target/mechanistic/tuned_lens_training")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = inventory()
    command = [str(args.binary), str(REPOSITORY / "target/mechanistic/tuned_lens"), str(args.output)]
    started = time.monotonic()
    with (args.output / "stdout.log").open("w") as stdout, (args.output / "stderr.log").open("w") as stderr:
        completed = subprocess.run(command, cwd=REPOSITORY, stdout=stdout, stderr=stderr, check=False)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    after = inventory()
    report = {"command": command, "exit_code": completed.returncode,
              "wall_seconds": time.monotonic() - started,
              "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
              "maximum_resident_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
              "binary_sha256": sha(args.binary), "source_before": before,
              "source_after": after, "sources_unchanged_during_execution": before == after,
              "measurement_source_sha256": sha(Path(__file__))}
    (args.output / "measurement.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if not key.startswith("source_")}, indent=2))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
