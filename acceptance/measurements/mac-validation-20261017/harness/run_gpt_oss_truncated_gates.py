"""Runs the truncated gpt-oss-20b F32 gates one process at a time (scratch harness, not committed).

Usage: python3 -I run_gpt_oss_truncated_gates.py BINARY OUT_DIR [ONLY_PREFIX...]

Runs exactly what final_gates.sh runs for "gpt-oss-20b truncated references":
for the 1-layer and then the 4-layer F32 bundle, the four Flex tests and then
the four Metal tests, each with only TRANSFORMERS_GPT_OSS_DIR and
TRANSFORMERS_GPT_OSS_REFERENCE_DIR set (no dtype, tolerance or minimum
override, so the F32 bundle selects F32 and the F32 tolerances apply).

The caller holds the shared heavy-model lock. Before each test this script
waits for any running source gate (full_gate.sh), checks that the worktree is
clean, and writes a per-test log that starts with `== source <sha> dirty=0`,
names the binary (with its SHA-256), the backend/dtype, every TRANSFORMERS_*
variable, and the exact command, then holds the test's complete stdout and
stderr and the full `/usr/bin/time -l` summary, and closes with an `== exit=`
line carrying wall time and the watchdog's memory figures.

With GATE_EXPORT_DIR and GATE_EXPORT_SHA set, the tests run from that
`git archive` export of the commit instead of the worktree; the export is
checked against the commit's tree (scratch verify_export.py, read-only) and
its mismatch count is the logged `dirty` figure.

Watchdog (sampled once per second, the policy of the earlier replay records):
kill on free memory below 10%, swap growth above 8 GiB, or swap-ins above
1 GiB counted only while free memory is below 30%.
"""

import hashlib
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = "/Users/nover/repos/transformers-worktrees/tfm-polish-half"
EXPORT_DIR = os.environ.get("GATE_EXPORT_DIR")
EXPORT_SHA = os.environ.get("GATE_EXPORT_SHA")
WT = EXPORT_DIR or REPO
VERIFY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_export.py")
A = "/Volumes/bag of holding/models/transformers-acceptance"
CHECKPOINT = "/Volumes/T9/models/openai/gpt-oss-20b"
PAGE = 16384

BUNDLES = [("1layer", "gpt-oss-20b-6cee5e81-1layer-f32-reference.bundle"),
           ("4layer", "gpt-oss-20b-6cee5e81-4layer-f32-reference.bundle")]
TESTS = ["prompt_logits_match_the_reference", "cached_greedy_matches_the_reference",
         "batched_prefill_matches_the_reference", "cached_long_prompt_matches_the_reference"]


def plan():
    """Yields (log name, backend/dtype label, test name, TRANSFORMERS_* env) in final_gates.sh order."""
    for short, bundle in BUNDLES:
        env = {"TRANSFORMERS_GPT_OSS_DIR": CHECKPOINT,
               "TRANSFORMERS_GPT_OSS_REFERENCE_DIR": f"{A}/{bundle}"}
        for backend, prefix in (("flex", "flex_"), ("metal", "metal::metal_")):
            for test in TESTS:
                yield (f"gpt-oss-20b-{short}.f32.{backend}.{test}", f"{backend.capitalize()}/F32",
                       prefix + test, dict(env))


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


def source():
    """Returns (commit sha, mismatch count) of the export, or (HEAD sha, porcelain lines) of the worktree."""
    if EXPORT_DIR:
        out = subprocess.run([sys.executable, "-I", VERIFY, REPO, EXPORT_SHA, EXPORT_DIR],
                             capture_output=True, text=True, check=True).stdout
        return EXPORT_SHA, int(re.search(r"dirty=(\d+)", out).group(1))
    sha = subprocess.run(["git", "-C", WT, "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", WT, "status", "--porcelain"], capture_output=True,
                           text=True, check=True).stdout
    return sha, len(dirty.splitlines())


def wait_for_source_gates():
    """Waits while a source gate (full_gate.sh) runs."""
    while subprocess.run(["pgrep", "-f", "[f]ull_gate.sh"], capture_output=True).returncode == 0:
        time.sleep(60)


def main():
    """Runs every planned test (or those whose log name starts with a given prefix)."""
    binary, out_dir = sys.argv[1], sys.argv[2]
    only = sys.argv[3:]
    summary = open(os.path.join(out_dir, "summary.log"), "a", buffering=1)
    for log_name, label, test, extra in plan():
        if only and not any(log_name.startswith(p) for p in only):
            continue
        wait_for_source_gates()
        digest = hashlib.sha256(open(binary, "rb").read()).hexdigest()
        sha, dirty = source()
        if dirty:
            summary.write(f"== refusing {log_name}: dirty tree ({dirty} lines)\n")
            sys.exit(3)
        env = {k: v for k, v in os.environ.items() if not k.startswith("TRANSFORMERS_")}
        env.update(extra)
        cmd = ["/usr/bin/time", "-l", binary, "--ignored", "--exact", "--nocapture",
               "--test-threads", "1", test]
        path = os.path.join(out_dir, log_name + ".log")
        with open(path, "w", buffering=1) as log:
            log.write(f"== source {sha} dirty={dirty}\n")
            if EXPORT_DIR:
                log.write(f"== export {EXPORT_DIR} (git archive of {sha}, verified against its tree)\n")
            log.write(f"== binary {binary} sha256 {digest}\n")
            log.write(f"== {label} {test}\n")
            for key in sorted(extra):
                log.write(f"== env {key}={extra[key]}\n")
            log.write(f"== cwd {WT}\n== cmd {' '.join(cmd)}\n")
            load = os.getloadavg()
            log.write(f"== start {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                      f"load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f} "
                      f"free_pct {free_pct()} swap_used_mib {swap_used_mib():.0f}\n")
            log.flush()
            swap0, swapin0 = swap_used_mib(), swapins()
            last_swapin, pressured = swapin0, 0
            min_free, max_growth, reason = 100, 0.0, None
            start = time.monotonic()
            proc = subprocess.Popen(cmd, cwd=WT, env=env, stdout=log, stderr=subprocess.STDOUT,
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
            total_swapin = (swapins() - swapin0) * PAGE >> 20
            line = (f"== exit={proc.returncode} wall={wall:.1f}s min_free_pct={min_free} "
                    f"max_swap_growth_mib={max_growth:.0f} swapin_mib={total_swapin} "
                    f"pressured_swapin_mib={pressured * PAGE >> 20}"
                    + (f" watchdog=\"{reason}\"" if reason else ""))
            log.write(line + "\n")
        summary.write(f"{log_name} {label} {test}: {line}\n")


if __name__ == "__main__":
    main()
