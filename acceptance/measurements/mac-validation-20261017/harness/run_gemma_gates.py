"""Runs the gemma-3-4b-it real-checkpoint gates one process at a time (scratch harness, not committed).

Usage: python3 -I run_gemma_gates.py BINARY OUT_DIR [ONLY_PREFIX...]

With FLAN_T5_BINARY set, the two FLAN-T5-small F32 gates of final_gates.sh run
first from that binary (same variables as final_gates.sh).

The caller holds the shared heavy-model lock. Before each test this script
waits for any running source gate (full_gate.sh), checks that the worktree is
clean, and writes a per-test log that starts with `== source <sha> dirty=0`,
names the binary (with its SHA-256), the backend/dtype, every TRANSFORMERS_*
variable, and the exact command, then holds the test's complete stdout and
stderr and the full `/usr/bin/time -l` summary, and closes with an `== exit=`
line carrying wall time and the watchdog's memory figures.

Order: (b) the ten F32 gates exactly as final_gates.sh runs them (same three
variables, no dtype, no minimum override), Flex then Metal; then (a) the
bundle-only calibration test and the ten half-precision gates, Flex/BF16 then
Metal/F16, each with the dtype and both BF16 twins. Only the long cached
half-precision test sets TRANSFORMERS_GEMMA_MIN_GREEDY=0, alone in its
process, as the module documentation prescribes.

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

WT = "/Users/nover/repos/transformers-worktrees/tfm-polish-half"
R = "/Volumes/T9/transformers-references"
PAGE = 16384

BASE = {
    "TRANSFORMERS_GEMMA_DIR": "/Volumes/T9/models/google/gemma-3-4b-it",
    "TRANSFORMERS_GEMMA_REFERENCE_DIR": f"{R}/gemma-3-4b-it-093f9f38",
    "TRANSFORMERS_GEMMA_LONG_REFERENCE_DIR": f"{R}/gemma-3-4b-it-093f9f38-long",
}
TWINS = {
    "TRANSFORMERS_GEMMA_BF16_REFERENCE_DIR": f"{R}/gemma-3-4b-it-093f9f38-bf16",
    "TRANSFORMERS_GEMMA_LONG_BF16_REFERENCE_DIR": f"{R}/gemma-3-4b-it-093f9f38-long-bf16",
}

TESTS = ["prompt_logits_match_the_reference", "cached_greedy_matches_the_reference",
         "batched_prefill_matches_the_reference", "long_context_logits_cross_the_sliding_window",
         "long_context_cached_decode_wraps_the_rings"]


M = "/Volumes/bag of holding/models"
FLAN = {
    "TRANSFORMERS_FLAN_T5_SMALL_DIR": f"{M}/google/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
    "TRANSFORMERS_FLAN_T5_SMALL_REFERENCE_DIR": f"{M}/transformers-acceptance/flan-t5-small-0fc9ddf7-reference.bundle",
}


def plan():
    """Yields (log name, backend/dtype label, test name, TRANSFORMERS_* env, binary or None) in run order."""
    flan = os.environ.get("FLAN_T5_BINARY")
    if flan:
        for backend in ("flex", "metal"):
            yield (f"flan-t5-small.f32.{backend}", f"{backend.capitalize()}/F32",
                   f"flan_t5_small_{backend}_matches_official_reference", dict(FLAN), flan)
    # (b) F32, exactly as final_gates.sh.
    for backend, prefix in (("flex", "flex_"), ("metal", "metal::metal_")):
        for test in TESTS:
            yield f"gemma-3-4b-it.f32.{backend}.{test}", f"{backend.capitalize()}/F32", prefix + test, dict(BASE), None
    # (a) the calibration figures from the four bundles alone, then half precision.
    yield ("gemma-3-4b-it.calibration.official_bf16_references_calibrate_the_half_precision_bounds",
           "bundles only", "official_bf16_references_calibrate_the_half_precision_bounds",
           dict(BASE, **TWINS), None)
    for backend, prefix, dtype in (("flex", "flex_", "bf16"), ("metal", "metal::metal_", "f16")):
        for test in TESTS:
            env = dict(BASE, **TWINS, TRANSFORMERS_GEMMA_DTYPE=dtype)
            if test == "long_context_cached_decode_wraps_the_rings":
                env["TRANSFORMERS_GEMMA_MIN_GREEDY"] = "0"
            yield (f"gemma-3-4b-it.{dtype}.{backend}.{test}", f"{backend.capitalize()}/{dtype.upper()}",
                   prefix + test, env, None)


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
    """Returns (HEAD sha, number of porcelain lines) of the worktree."""
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
    default_binary, out_dir = sys.argv[1], sys.argv[2]
    only = sys.argv[3:]
    summary = open(os.path.join(out_dir, "summary.log"), "a", buffering=1)
    for log_name, label, test, extra, binary in plan():
        if only and not any(log_name.startswith(p) for p in only):
            continue
        binary = binary or default_binary
        digest = hashlib.sha256(open(binary, "rb").read()).hexdigest()
        wait_for_source_gates()
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
            log.write(f"== binary {binary} sha256 {digest}\n")
            log.write(f"== {label} {test}\n")
            for key in sorted(extra):
                log.write(f"== env {key}={extra[key]}\n")
            log.write(f"== cwd {WT}\n== cmd {' '.join(cmd)}\n")
            load = os.getloadavg()
            log.write(f"== start {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                      f"load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}\n")
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
