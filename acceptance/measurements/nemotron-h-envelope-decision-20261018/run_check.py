"""Campaign driver: runs one source check or bundle-only test from the clean worktree and logs it.

Usage: python3 -I run_check.py EXPECTED_COMMIT LOG -- COMMAND...

LOG is a path relative to this campaign directory. The log opens with
`== source <commit> dirty=<n>` (`dirty` counts `git status --porcelain` lines
outside this campaign directory), `== cmd <command>`, every
`CARGO_TARGET_DIR`/`RUSTDOCFLAGS`/`TRANSFORMERS_*` variable, the start time,
uptime, and the free space of the build volume; it closes with the end time,
the wall time, `== source after run <commit> dirty=<n>`, and, last,
`== exit=<status>`. A checkout that is not clean at EXPECTED_COMMIT is refused
before the command runs (exit 3).
"""

import datetime as dt
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[2]
CAMPAIGN = HERE.relative_to(CHECKOUT).as_posix() + "/"
BUILD_VOLUME = "/Volumes/bag of holding"
PREFIXES = ("CARGO_TARGET_DIR", "RUSTDOCFLAGS", "TRANSFORMERS_")


def source_identity():
    """(HEAD, number of `git status --porcelain` lines outside this campaign directory)."""
    head = subprocess.run(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(CHECKOUT), "status", "--porcelain"], capture_output=True, text=True,
                            check=True).stdout.splitlines()
    return head, sum(not line[3:].startswith(CAMPAIGN) for line in status)


def now():
    """The current UTC time, to the second."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def main():
    """Runs the command once; returns its exit status."""
    expected, relative, separator, *command = sys.argv[1:]
    if separator != "--" or not command:
        raise SystemExit("usage: run_check.py EXPECTED_COMMIT LOG -- COMMAND...")
    log_path = HERE / relative
    log_path.parent.mkdir(parents=True, exist_ok=True)
    head, dirty = source_identity()
    with open(log_path, "x") as log:
        def write(line):
            """Writes one line and flushes it before the command's output."""
            log.write(line + "\n")
            log.flush()

        write(f"== source {head} dirty={dirty}")
        write(f"== cmd {shlex.join(command)}")
        for key in sorted(os.environ):
            if key.startswith(PREFIXES):
                write(f"== env {key}={os.environ[key]}")
        uptime = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
        free = shutil.disk_usage(BUILD_VOLUME).free / 2**30
        write(f"== start {now()} uptime {uptime}")
        write(f"== start free {BUILD_VOLUME} {free:.1f} GiB")
        if head != expected or dirty:
            write(f"REFUSED: the checkout is at {head} with {dirty} changes, not clean at {expected}")
            write("== exit=3")
            return 3
        started = time.monotonic()
        status = subprocess.run(command, cwd=CHECKOUT, stdout=log, stderr=subprocess.STDOUT).returncode
        write(f"== end {now()} wall={time.monotonic() - started:.1f}s")
        after, after_dirty = source_identity()
        write(f"== source after run {after} dirty={after_dirty}")
        write(f"== exit={status}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
