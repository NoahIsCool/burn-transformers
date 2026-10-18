"""Campaign driver: one official Nemotron-H F16 graph under /usr/bin/time -l with a memory and disk watchdog.

Usage: python3 -I run_graph.py LABEL EXPORT

LABEL is one of GRAPHS below; EXPORT is a `git archive` export of the commit
whose scripts/generate_nemotron_h_f16_graph_reference.py runs. The caller
(run_locked.sh) holds the heavy-model lock. The log opens with the source,
tool digest, command, start time, uptime, the processes using the most CPU,
free memory, swap, and free disk, and closes with the exit status, the
watchdog's figures, and the same host sample at the end. Every finished run
appends one JSON line to logs/runs.jsonl.
"""

import datetime as dt
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
PYTHON = "/Volumes/bag of holding/models/transformers-acceptance/envs/muse-cfa309-py311/bin/python"
MODEL = "/Volumes/bag of holding/models/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
CHECKOUT = "/Volumes/bag of holding/models/hf-transformers-reference"
ACCEPTANCE = Path("/Volumes/bag of holding/models/transformers-acceptance")
ORACLE = ACCEPTANCE / "nemotron-3.5-lightning-30b-a3b-a9904d24-reference.bundle"
PRECISION = ACCEPTANCE / "nemotron-3.5-lightning-30b-a3b-a9904d24-precision.bundle"
GRAPHS_DIR = ACCEPTANCE / "nemotron-3.5-lightning-30b-a3b-a9904d24-f16-graphs"
CONTROLS_DIR = ACCEPTANCE / "nemotron-h-envelope-20261018"
OFFLOAD = Path("/Volumes/T9/transformers-offload/nemotron-h-envelope-20261018")
#: label -> (device, attention, output, resident budget in GiB). A resident MPS load
#: (attempt 1) grew swap by 12.5 GiB in 80 s, so MPS uses the CPU's 16 GiB budget.
GRAPHS = {
    "cpu-sdpa": ("cpu", "sdpa", GRAPHS_DIR / "f16-cpu-sdpa.bundle", 16),
    "mps-sdpa": ("mps", "sdpa", GRAPHS_DIR / "f16-mps-sdpa.bundle", 16),
    "mps-eager-control": ("mps", "eager", CONTROLS_DIR / "f16-mps-eager-control.bundle", 16),
}
#: Watchdog policy of the earlier rounds, plus a system-volume reserve.
MIN_FREE_MEMORY_PERCENT = 10
PRESSURED_FREE_PERCENT = 30
MAX_PRESSURED_SWAPIN_BYTES = 1 << 30
MAX_SWAP_GROWTH_BYTES = 8 << 30
MIN_ROOT_FREE_BYTES = 10 << 30
#: A graph starts only when this much memory is free and the system volume
#: (which holds macOS swap) has this much room, waiting up to START_WAIT_SECONDS.
START_FREE_PERCENT = 70
START_ROOT_FREE_BYTES = 15 << 30
START_WAIT_SECONDS = 1800


def sha256(path):
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def run_text(command):
    """Stdout of a short command, or its error."""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout
    except Exception as error:  # the sample is evidence, never fatal
        return f"<{error}>"


def free_memory_percent():
    """System-wide free memory percentage from memory_pressure, or None."""
    for line in run_text(["memory_pressure", "-Q"]).splitlines():
        if "free percentage" in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("%"))
    return None


def swap_used_bytes():
    """vm.swapusage's used figure in bytes, or None."""
    match = re.search(r"used = ([0-9.]+)M", run_text(["sysctl", "vm.swapusage"]))
    return int(float(match.group(1)) * 2**20) if match else None


def swapins_bytes():
    """vm_stat's cumulative swap-ins in bytes, or None."""
    text = run_text(["vm_stat"])
    page = re.search(r"page size of (\d+) bytes", text)
    count = re.search(r"Swapins:\s+(\d+)", text)
    return int(count.group(1)) * int(page.group(1)) if page and count else None


def host_lines(label):
    """Uptime, top CPU processes, memory, swap, and disk free as log lines."""
    lines = [f"== {label} {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}",
             f"== {label} uptime {run_text(['uptime']).strip()}"]
    top = run_text(["ps", "-Ao", "pid,pcpu,rss,etime,comm", "-r"]).splitlines()[:9]
    lines += [f"== {label} top-cpu {line}" for line in top]
    lines.append(f"== {label} free-memory {free_memory_percent()}% swap-used {swap_used_bytes()} bytes")
    for volume in ("/", "/Volumes/bag of holding", "/Volumes/T9"):
        lines.append(f"== {label} disk {volume} free {shutil.disk_usage(volume).free / 2**30:.1f} GiB")
    return lines


def time_figures(text):
    """Peak footprint, maximum resident set, and real seconds from /usr/bin/time -l."""
    pick = lambda pattern, cast: [cast(v) for v in re.findall(pattern, text, re.M)][-1:] or [None]
    return {"peak_memory_footprint_bytes": pick(r"^\s*(\d+)\s+peak memory footprint", int)[0],
            "maximum_resident_set_bytes": pick(r"^\s*(\d+)\s+maximum resident set size", int)[0],
            "time_real_seconds": pick(r"^\s*([0-9.]+) real", float)[0]}


def kill_group(child):
    """Kills the child's whole process group (attempt 1 killed only /usr/bin/time and orphaned the generator)."""
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    """Runs one graph; returns its exit status."""
    label, export = sys.argv[1], Path(sys.argv[2]).resolve()
    device, attention, output, budget = GRAPHS[label]
    tool = export / "scripts/generate_nemotron_h_f16_graph_reference.py"
    commit = (export / "SOURCE_COMMIT").read_text().strip()
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{label}.log"
    command = [PYTHON, "-I", str(tool), "--model-dir", MODEL, "--transformers-dir", CHECKOUT,
               "--oracle-dir", str(ORACLE), "--precision-dir", str(PRECISION), "--device", device,
               "--attn-implementation", attention, "--output", str(output)]
    if budget is not None:
        OFFLOAD.mkdir(parents=True, exist_ok=True)
        command += ["--max-resident-gib", str(budget), "--offload-dir", str(OFFLOAD)]
    status, wall, started_at = None, 0.0, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    figures = {"min_free_pct": 100, "max_swap_growth_mib": 0, "swapin_mib": 0, "pressured_swapin_mib": 0,
               "min_root_free_gib": None, "killed": None}
    with open(log_path, "a") as log:
        write = lambda line: (log.write(line + "\n"), log.flush())
        write(f"== source {commit} (git archive export {export})")
        write(f"== tool {tool} sha256 {sha256(tool)}")
        write(f"== cmd /usr/bin/time -l {json.dumps(command)}")
        for line in host_lines("start"):
            write(line)
        if output.exists():
            write(f"SKIP: {output} exists")
            return 0
        waited = 0
        ready = lambda: ((free_memory_percent() or 0) >= START_FREE_PERCENT
                         and shutil.disk_usage("/").free >= START_ROOT_FREE_BYTES)
        while not ready() and waited < START_WAIT_SECONDS:
            time.sleep(60)
            waited += 60
        free, root = free_memory_percent() or 0, shutil.disk_usage("/").free
        write(f"== start gate: free-memory {free}% system-volume {root / 2**30:.1f} GiB after waiting {waited}s "
              f"(needs {START_FREE_PERCENT}% and {START_ROOT_FREE_BYTES >> 30} GiB)")
        if not ready():
            write("REFUSED: the host has not recovered enough free memory or system-volume space")
            return 4
        swap0, swapin0 = swap_used_bytes() or 0, swapins_bytes() or 0
        started = time.monotonic()
        # A session of its own lets the watchdog kill /usr/bin/time and the generator together.
        child = subprocess.Popen(["/usr/bin/time", "-l"] + command, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)

        def watchdog():
            """Kills the child on memory pressure, swap growth, pressured swap-ins, or a full system volume."""
            pressured = 0
            last_swapin = swapin0
            while child.poll() is None:
                free, swap, swapin = free_memory_percent(), swap_used_bytes(), swapins_bytes()
                root = shutil.disk_usage("/").free
                figures["min_root_free_gib"] = round(min(root / 2**30, figures["min_root_free_gib"] or 1e9), 1)
                if free is not None:
                    figures["min_free_pct"] = min(figures["min_free_pct"], free)
                if swap is not None:
                    figures["max_swap_growth_mib"] = max(figures["max_swap_growth_mib"], (swap - swap0) >> 20)
                if swapin is not None:
                    if free is not None and free < PRESSURED_FREE_PERCENT:
                        pressured += swapin - last_swapin
                    last_swapin = swapin
                    figures["swapin_mib"] = (swapin - swapin0) >> 20
                    figures["pressured_swapin_mib"] = pressured >> 20
                reason = None
                if free is not None and free < MIN_FREE_MEMORY_PERCENT:
                    reason = f"free memory {free}% below {MIN_FREE_MEMORY_PERCENT}%"
                elif swap is not None and swap - swap0 > MAX_SWAP_GROWTH_BYTES:
                    reason = f"swap grew by {(swap - swap0) >> 20} MiB"
                elif pressured > MAX_PRESSURED_SWAPIN_BYTES:
                    reason = f"{pressured >> 20} MiB swapped in below {PRESSURED_FREE_PERCENT}% free"
                elif root < MIN_ROOT_FREE_BYTES:
                    reason = f"system volume down to {root / 2**30:.1f} GiB free"
                if reason:
                    figures["killed"] = reason
                    kill_group(child)
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
        for line in host_lines("end"):
            write(line)
        manifest = output / "reference.json"
        reference = sha256(manifest) if status == 0 and manifest.exists() else None
        if reference:
            write(f"== reference.json sha256 {reference}")
    record = {"label": label, "device": device, "attention": attention, "bundle": str(output), "status": status,
              "source_commit": commit, "tool_sha256": sha256(tool), "started_at": started_at,
              "wall_seconds": round(wall, 1), "reference_json_sha256": reference, **figures}
    record.update(time_figures(log_path.read_text(errors="replace")))
    with open(LOGS / "runs.jsonl", "a") as runs:
        runs.write(json.dumps(record) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
