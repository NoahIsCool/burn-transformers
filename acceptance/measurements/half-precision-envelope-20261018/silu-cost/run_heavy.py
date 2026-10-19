"""Runs a list of heavy commands one process at a time under the shared heavy-model lock (scratch, not committed).

Usage: python3 -I run_heavy.py PLAN_JSON

PLAN_JSON is a list of {"log": path, "cmd": [...], "env": {...}, "cwd": dir,
"header": [lines]}. The script takes the lock (mkdir, retried every 60 s),
then runs each command in its own process with a clean TRANSFORMERS_*
environment plus "env", under /usr/bin/time -l and a once-per-second watchdog
(kill below 10% free memory, above 8 GiB swap growth, or above 1 GiB of
swap-ins counted while free memory is below 30%). Each log starts with the
header lines, the environment, the command, the UTC start time and the load
average, holds the complete output, and ends with an "== exit=" line. The lock
is always released, including on failure or interruption.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

LOCK = "/Users/nover/repos/transformers-worktrees/.heavy-model.lock"
PAGE = 16384


def free_pct():
    """Returns memory_pressure's system-wide free percentage."""
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    return int(re.search(r"free percentage: (\d+)%", out).group(1))


def swap_used_mib():
    """Returns used swap in MiB."""
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1))


def swapins():
    """Returns the cumulative swap-in page count."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int(re.search(r"Swapins:\s+(\d+)", out).group(1))


def now():
    """Returns the UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_one(item):
    """Runs one plan item with the watchdog; returns the closing line."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRANSFORMERS_")}
    env.update(item.get("env", {}))
    cmd = ["/usr/bin/time", "-l"] + item["cmd"]
    os.makedirs(os.path.dirname(item["log"]), exist_ok=True)
    with open(item["log"], "w", buffering=1) as log:
        for line in item.get("header", []):
            log.write(f"== {line}\n")
        for key in sorted(item.get("env", {})):
            log.write(f"== env {key}={item['env'][key]}\n")
        log.write(f"== cwd {item.get('cwd', os.getcwd())}\n== cmd {' '.join(cmd)}\n")
        load = os.getloadavg()
        log.write(f"== start {now()} load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}\n")
        log.flush()
        swap0, swapin0 = swap_used_mib(), swapins()
        last_swapin, pressured = swapin0, 0
        min_free, max_growth, reason = 100, 0.0, None
        start = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=item.get("cwd"), env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        while proc.poll() is None:
            time.sleep(1)
            free = free_pct()
            growth = swap_used_mib() - swap0
            current = swapins()
            if free < 30:
                pressured += current - last_swapin
            last_swapin = current
            min_free, max_growth = min(min_free, free), max(max_growth, growth)
            if free < 10:
                reason = f"free memory {free}%"
            elif growth > 8192:
                reason = f"swap grew by {growth:.0f} MiB"
            elif pressured * PAGE > 1 << 30:
                reason = f"pressured swap-ins {pressured * PAGE >> 20} MiB"
            if reason:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
        wall = time.monotonic() - start
        load = os.getloadavg()
        line = (f"== exit={proc.returncode} wall={wall:.1f}s end={now()} load_end {load[0]:.2f} "
                f"min_free_pct={min_free} max_swap_growth_mib={max_growth:.0f} "
                f"swapin_mib={(swapins() - swapin0) * PAGE >> 20} pressured_swapin_mib={pressured * PAGE >> 20}"
                + (f" watchdog=\"{reason}\"" if reason else ""))
        log.write(line + "\n")
    return line


def main():
    plan = json.load(open(sys.argv[1]))
    summary = sys.argv[1] + ".summary"
    with open(summary, "a", buffering=1) as out:
        while True:
            try:
                os.mkdir(LOCK)
                break
            except FileExistsError:
                out.write(f"{now()} waiting for {LOCK}\n")
                time.sleep(60)
        out.write(f"{now()} lock acquired\n")
        try:
            for item in plan:
                line = run_one(item)
                out.write(f"{os.path.basename(item['log'])}: {line}\n")
        finally:
            os.rmdir(LOCK)
            out.write(f"{now()} lock released\n")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
