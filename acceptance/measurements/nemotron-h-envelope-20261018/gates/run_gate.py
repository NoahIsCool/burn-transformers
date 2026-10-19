"""Campaign driver: runs prebuilt Nemotron-H test binaries one test per process, from the clean worktree at the code head.

Usage: python3 -I run_gate.py EXPECTED_COMMIT BINARY NAME TEST [ignored|plain] [KEY=VALUE...]

The caller holds the heavy-model lock for checkpoint runs (run_locked_gate.sh).
Each run's log opens with `== source <commit> dirty=<n>`, the binary and its
SHA-256, the test, every TRANSFORMERS_* variable, the command, the start
time, uptime, the processes using the most CPU, free memory, swap, and free
disk; runs under /usr/bin/time -l with the graph driver's watchdog; and
closes with `== exit=`, the watchdog's figures, `== source after run`, and
the host sample at the end. One JSON line per run goes to logs/runs.jsonl.
"""

import datetime as dt
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
CHECKOUT = Path("/Users/nover/repos/transformers-worktrees/tfm-nemotron-envelope")
SPEC = importlib.util.spec_from_file_location("run_graph", HERE.parent / "generator" / "run_graph.py")
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)


def source_identity():
    """(HEAD, number of `git status --porcelain` lines outside this campaign's log directories)."""
    head = subprocess.run(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(CHECKOUT), "status", "--porcelain"], capture_output=True, text=True,
                            check=True).stdout.splitlines()
    campaign = "acceptance/measurements/nemotron-h-envelope-20261018/"
    return head, sum(not line[3:].startswith(campaign) for line in status)


def main():
    """Runs one test; returns its exit status."""
    expected, binary, name, test, mode, *pairs = sys.argv[1:]
    env = dict(os.environ)
    variables = dict(pair.split("=", 1) for pair in pairs)
    env.update(variables)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{name}.log"
    command = [binary] + (["--ignored"] if mode == "ignored" else []) + [
        "--exact", "--nocapture", "--test-threads", "1", test]
    head, dirty = source_identity()
    figures = {"min_free_pct": 100, "max_swap_growth_mib": 0, "swapin_mib": 0, "pressured_swapin_mib": 0,
               "min_root_free_gib": None, "killed": None}
    started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with open(log_path, "w") as log:
        write = lambda line: (log.write(line + "\n"), log.flush())
        write(f"== source {head} dirty={dirty}")
        write(f"== binary {binary} sha256 {GRAPH.sha256(binary)}")
        write(f"== test {test} ({mode})")
        for key in sorted(variables):
            write(f"== env {key}={variables[key]}")
        write(f"== cwd {CHECKOUT}")
        write(f"== cmd /usr/bin/time -l {' '.join(command)}")
        for line in GRAPH.host_lines("start"):
            write(line)
        if head != expected or dirty:
            write(f"REFUSED: the checkout is at {head} with {dirty} changes, not clean at {expected}")
            return 3
        swap0, swapin0 = GRAPH.swap_used_bytes() or 0, GRAPH.swapins_bytes() or 0
        started = time.monotonic()
        child = subprocess.Popen(["/usr/bin/time", "-l"] + command, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=CHECKOUT, env=env, start_new_session=True)

        def watchdog():
            """The graph driver's policy: memory pressure, swap growth, pressured swap-ins, system volume."""
            pressured, last = 0, swapin0
            while child.poll() is None:
                free, swap, swapin = GRAPH.free_memory_percent(), GRAPH.swap_used_bytes(), GRAPH.swapins_bytes()
                root = shutil.disk_usage("/").free
                figures["min_root_free_gib"] = round(min(root / 2**30, figures["min_root_free_gib"] or 1e9), 1)
                if free is not None:
                    figures["min_free_pct"] = min(figures["min_free_pct"], free)
                if swap is not None:
                    figures["max_swap_growth_mib"] = max(figures["max_swap_growth_mib"], (swap - swap0) >> 20)
                if swapin is not None:
                    if free is not None and free < GRAPH.PRESSURED_FREE_PERCENT:
                        pressured += swapin - last
                    last = swapin
                    figures["swapin_mib"], figures["pressured_swapin_mib"] = (swapin - swapin0) >> 20, pressured >> 20
                reason = None
                if free is not None and free < GRAPH.MIN_FREE_MEMORY_PERCENT:
                    reason = f"free memory {free}% below {GRAPH.MIN_FREE_MEMORY_PERCENT}%"
                elif swap is not None and swap - swap0 > GRAPH.MAX_SWAP_GROWTH_BYTES:
                    reason = f"swap grew by {(swap - swap0) >> 20} MiB"
                elif pressured > GRAPH.MAX_PRESSURED_SWAPIN_BYTES:
                    reason = f"{pressured >> 20} MiB swapped in below {GRAPH.PRESSURED_FREE_PERCENT}% free"
                elif root < GRAPH.MIN_ROOT_FREE_BYTES:
                    reason = f"system volume down to {root / 2**30:.1f} GiB free"
                if reason:
                    figures["killed"] = reason
                    GRAPH.kill_group(child)
                    return
                time.sleep(5)

        thread = threading.Thread(target=watchdog, daemon=True)
        thread.start()
        status = child.wait()
        thread.join(timeout=60)
        wall = time.monotonic() - started
        killed = f" WATCHDOG_KILLED: {figures['killed']}" if figures["killed"] else ""
        write(f"== exit={status} wall={wall:.1f}s min_free_pct={figures['min_free_pct']} "
              f"max_swap_growth_mib={figures['max_swap_growth_mib']} swapin_mib={figures['swapin_mib']} "
              f"pressured_swapin_mib={figures['pressured_swapin_mib']} min_root_free_gib={figures['min_root_free_gib']}{killed}")
        after, after_dirty = source_identity()
        write(f"== source after run {after} dirty={after_dirty}")
        for line in GRAPH.host_lines("end"):
            write(line)
    text = log_path.read_text(errors="replace")
    record = {"name": name, "test": test, "mode": mode, "binary": binary, "binary_sha256": GRAPH.sha256(binary),
              "source_commit": head, "started_at": started_at, "status": status, "wall_seconds": round(wall, 1),
              "result": "pass" if "test result: ok. 1 passed" in text else "fail", **figures}
    record.update(GRAPH.time_figures(text))
    with open(LOGS / "runs.jsonl", "a") as runs:
        runs.write(json.dumps(record) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
