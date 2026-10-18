"""Scratch (half2 S9): runs the gate tests that run_gates.py could not run as planned (not committed).

Usage: python3 -I run_extra.py BINARIES_JSON GROUP...

Groups:
  muse-official  the Muse-Glimmer-30B official gate from a binary built in a clean Git
                 clone at the gate head (gates/muse-git): the gate records Git
                 provenance (`git rev-parse --show-toplevel` in its compiled
                 CARGO_MANIFEST_DIR), which the git-archive export cannot give, so its
                 run from the export stopped before loading anything (harness error);
  gemma-refusal  the non-ignored Gemma F16 refusal test, run without `--ignored`
                 (run_gates.py passes `--ignored`, which filters it out).

Each run uses run_gates.py's log format, watchdog, lock, and runs.jsonl line;
the first log line names the commit and how its tree was obtained.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_gates as rg  # noqa: E402

G = rg.G
CLONE = os.path.join(G, "muse-git")


def plans(group):
    """(name, label, target, kind, test, env, cwd, source note, ignored) of a group's runs."""
    muse = {"TRANSFORMERS_MUSE_GLIMMER_DIR": f"{rg.M}/meta-models/Muse-Glimmer-30B-a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
            "TRANSFORMERS_MUSE_GLIMMER_REFERENCE_DIR": rg.ACC}
    head = rg.EXPORT_SHA
    return {
        "muse-official": [("muse-glimmer-30b.f16.metal.official-git", "M", "muse_glimmer_artifacts", "metal/F16",
                           "matches_pinned_muse_glimmer_official_f16_reference_on_metal", muse, CLONE,
                           f"clean Git clone {CLONE} checked out at {head}, tree identical to the export {rg.SRC}", True)],
        "gemma-refusal": [("gemma.f16.refusal-unit.f16_runs_are_refused_with_the_measured_overflow", "A", "gemma_artifacts",
                           "unit", "f16_runs_are_refused_with_the_measured_overflow", {}, rg.SRC,
                           f"git archive export {rg.SRC}", False)],
    }[group]


def run_one(name, label, target, kind, test, extra, cwd, note, ignored, binaries, summary):
    """Runs one test in its own process with run_gates.py's watchdog; returns the exit status."""
    binary = binaries[label][target]
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRANSFORMERS_")}
    env.update(extra)
    cmd = ["/usr/bin/time", "-l", binary] + (["--ignored"] if ignored else []) + ["--exact", "--nocapture", "--test-threads", "1", test]
    path = os.path.join(rg.LOGS, name + ".log")
    if os.path.exists(path):
        raise SystemExit(f"refusing to replace {path}")
    with open(path, "w", buffering=1) as log:
        log.write(f"== source {rg.EXPORT_SHA} ({note})\n")
        log.write(f"== binary {binary} sha256 {rg.sha256(binary)} build {label}\n")
        log.write(f"== {kind} {test}\n")
        for key in sorted(extra):
            log.write(f"== env {key}={extra[key]}\n")
        log.write(f"== cwd {cwd}\n== cmd {' '.join(cmd)}\n")
        load = os.getloadavg()
        log.write(f"== start {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                  f"load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f} free {rg.free_pct()}%\n")
        log.flush()
        swap0, swapin0 = rg.swap_used_mib(), rg.swapins()
        last_swapin, pressured = swapin0, 0
        min_free, max_growth, reason = 100, 0.0, None
        start = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            time.sleep(1)
            free = rg.free_pct()
            growth = rg.swap_used_mib() - swap0
            current = rg.swapins()
            if free < 30:
                pressured += current - last_swapin
            last_swapin = current
            min_free, max_growth = min(min_free, free), max(max_growth, growth)
            if free < 10:
                reason = f"free memory {free}%"
            elif growth > 8192:
                reason = f"swap grew by {growth:.0f} MiB"
            elif pressured * rg.PAGE > 1 << 30:
                reason = f"pressured swap-ins {pressured * rg.PAGE >> 20} MiB"
            if reason:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
        wall = time.monotonic() - start
        line = (f"== exit={proc.returncode} wall={wall:.1f}s min_free_pct={min_free} "
                f"max_swap_growth_mib={max_growth:.0f} swapin_mib={(rg.swapins() - swapin0) * rg.PAGE >> 20} "
                f"pressured_swapin_mib={pressured * rg.PAGE >> 20}" + (f" watchdog=\"{reason}\"" if reason else ""))
        log.write(line + "\n")
    text = open(path, errors="replace").read()
    peak = re.findall(r"^\s*(\d+)\s+peak memory footprint", text, re.M)
    summary.write(json.dumps({"name": name, "kind": kind, "test": test, "exit": proc.returncode,
                              "wall_seconds": round(wall, 1), "peak_memory_footprint_bytes": int(peak[-1]) if peak else None,
                              "watchdog": reason, "log": path, "log_sha256": rg.sha256(path),
                              "result": "pass" if "test result: ok. 1 passed" in text else "fail"}) + "\n")
    summary.flush()
    return proc.returncode


def main():
    signal.signal(signal.SIGTERM, rg.terminate)
    signal.signal(signal.SIGHUP, rg.terminate)
    binaries = json.load(open(sys.argv[1]))
    summary = open(os.path.join(rg.LOGS, "runs.jsonl"), "a", buffering=1)
    driver = open(os.path.join(rg.LOGS, "driver.log"), "a", buffering=1)
    now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    for group in sys.argv[2:]:
        if not rg.mounted():
            driver.write(f"{now()} STOP: a volume is not mounted\n")
            return 5
        held = False
        try:
            if group != "gemma-refusal":
                driver.write(f"{now()} waiting for the lock ({group}, run_extra.py)\n")
                while True:
                    try:
                        os.mkdir(rg.LOCK)
                        held = True
                        break
                    except FileExistsError:
                        time.sleep(60)
                driver.write(f"{now()} lock acquired ({group}, run_extra.py)\n")
            for run in plans(group):
                status = run_one(*run, binaries, summary)
                driver.write(f"{now()} {run[0]} exit={status}\n")
        finally:
            if held:
                os.rmdir(rg.LOCK)
                driver.write(f"{now()} lock released ({group}, run_extra.py)\n")
    driver.write(f"{now()} done {' '.join(sys.argv[2:])} (run_extra.py)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
