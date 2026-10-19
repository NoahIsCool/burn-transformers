"""Review round (half-precision review, October 18): reruns the gates the round's fixes could affect, one process at a time.

Usage: python3 -I run_review.py BINARIES_JSON GROUP...

A copy of the follow-ups round's rerun/harness/run_rerun.py with these
changes: EXPECTED is the round's code head, the groups are the round's
(every gemma-3-4b-it gate, the expected failures of Gemma F16 on Metal, now
uncalibrated, and Metal BF16, the bundle-only Gemma tests and the new unit
test, and this round's operation-level tests, Flex and Metal, from the
library test binary), and a group may name ordinary (non-ignored) tests.
Every group except the bundle-only one takes
/Users/nover/repos/transformers-worktrees/.heavy-model.lock (mkdir, retried
every 60 s) and releases it when the group ends, also on failure or
interruption. Each test runs in its own process under the earlier rounds'
memory watchdog, from the worktree checkout, which must stay at EXPECTED and
clean. One JSON line per run is appended to logs/runs.jsonl.
"""

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(H)
LOGS = os.path.join(ROOT, "logs")
CHECKOUT = "/Users/nover/repos/transformers-worktrees/tfm-half-followups2"
EXPECTED = "1e38eecf989a1ae568b67b2e9fcdd9b2ebdbadb8"
LOCK = "/Users/nover/repos/transformers-worktrees/.heavy-model.lock"
PAGE = 16384
R = "/Volumes/T9/transformers-references"
S = "7a50e44"
M = "/Volumes/bag of holding/models"

LLAMA = [
    # (name, checkpoint, truth, decode truth, F32 tolerance, BF16 min greedy, F16 min greedy)
    ("SmolLM-135M", f"{M}/HuggingFaceTB/SmolLM-135M-1d461723eec654e65efdc40cf49301c89c0c92f4",
     "SmolLM-135M-1d461723", "SmolLM-135M-1d461723-f32-cpu-eager", "1e-3", "0", None),
    ("Qwen3-4B", "/Volumes/T9/models/Qwen/Qwen3-4B", "Qwen3-4B-531c80e2", "Qwen3-4B-531c80e2-f32-cpu-eager",
     "1e-3", None, None),
    ("SmolLM3-3B", "/Volumes/T9/models/HuggingFaceTB/SmolLM3-3B", "SmolLM3-3B-1fa6fa28",
     "SmolLM3-3B-1fa6fa28-f32-cpu-eager", "1e-3", "0", None),
    ("Qwen2-7B-f64", "/Volumes/T9/models/Qwen/Qwen2-7B", "Qwen2-7B-453ed157-f64", "Qwen2-7B-453ed157-f64-cpu-eager",
     "2.5e-3", "0", "0"),
    ("Llama-3.1-8B", "/Volumes/T9/models/meta-llama/Llama-3.1-8B", "Llama-3.1-8B-d04e592b",
     "Llama-3.1-8B-d04e592b-f32-cpu-eager", "1e-3", None, None),
]
LLAMA_TESTS = ["prompt_logits_match_the_reference", "cached_greedy_matches_the_reference",
               "batched_prefill_matches_the_reference"]
GEMMA_TESTS = ["prompt_logits_match_the_reference", "cached_greedy_matches_the_reference",
               "batched_prefill_matches_the_reference", "long_context_logits_cross_the_sliding_window",
               "long_context_cached_decode_wraps_the_rings"]
GEMMA_TRUTH = f"{R}/gemma-3-4b-it-093f9f38"
GEMMA_BASE = {"TRANSFORMERS_GEMMA_DIR": "/Volumes/T9/models/google/gemma-3-4b-it",
              "TRANSFORMERS_GEMMA_REFERENCE_DIR": GEMMA_TRUTH,
              "TRANSFORMERS_GEMMA_DECODE_TRUTH_DIR": f"{GEMMA_TRUTH}-f32-cpu-eager",
              "TRANSFORMERS_GEMMA_LONG_REFERENCE_DIR": f"{GEMMA_TRUTH}-long",
              "TRANSFORMERS_GEMMA_LONG_DECODE_TRUTH_DIR": f"{GEMMA_TRUTH}-long-f32-cpu-eager"}


def official(truth, dtype):
    """The four official graph bundles of a truth, as a PATH-style list."""
    return ":".join(f"{R}/{truth}-{dtype}-{d}-{a}-{S}" for d in ("cpu", "mps") for a in ("eager", "sdpa"))


def llama_base(name):
    """The checkpoint, truth, and decode-truth variables of one Llama-family checkpoint."""
    _, ckpt, truth, decode, *_ = next(c for c in LLAMA if c[0] == name)
    return {"TRANSFORMERS_LLAMA_FAMILY_DIR": ckpt, "TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR": f"{R}/{truth}",
            "TRANSFORMERS_LLAMA_FAMILY_DECODE_TRUTH_DIR": f"{R}/{decode}"}


def llama_plan(name):
    """Runs of one Llama-family checkpoint: F32 Flex and Metal, Flex/BF16, Metal/F16 (as in run_gates.py)."""
    _, ckpt, truth, decode, tol, bf16_min, f16_min = next(c for c in LLAMA if c[0] == name)
    base = llama_base(name)
    runs = []
    for backend, prefix in (("flex", "flex_"), ("metal", "metal::metal_")):
        for test in LLAMA_TESTS:
            runs.append((f"{name}.f32.{backend}.{test}", "llama_family_artifacts", f"{backend}/F32", prefix + test,
                         dict(base, TRANSFORMERS_LLAMA_FAMILY_TOLERANCE=tol), True))
    for backend, prefix, dtype, minimum in (("flex", "flex_", "bf16", bf16_min), ("metal", "metal::metal_", "f16", f16_min)):
        for test in LLAMA_TESTS:
            env = dict(base, TRANSFORMERS_LLAMA_FAMILY_DTYPE=dtype,
                       TRANSFORMERS_LLAMA_FAMILY_OFFICIAL_DIRS=official(truth, dtype))
            if minimum is not None and test == "cached_greedy_matches_the_reference":
                env["TRANSFORMERS_LLAMA_FAMILY_MIN_GREEDY"] = minimum
            runs.append((f"{name}.{dtype}.{backend}.{test}", "llama_family_artifacts",
                         f"{backend}/{dtype.upper()}", prefix + test, env, True))
    return runs


def gemma_plan():
    """Runs of gemma-3-4b-it: F32 Flex and Metal, Flex/BF16 (as in run_gates.py)."""
    runs = []
    for backend, prefix in (("flex", "flex_"), ("metal", "metal::metal_")):
        for test in GEMMA_TESTS:
            runs.append((f"gemma.f32.{backend}.{test}", "gemma_artifacts", f"{backend}/F32", prefix + test,
                         dict(GEMMA_BASE), True))
    for test in GEMMA_TESTS:
        env = dict(GEMMA_BASE, TRANSFORMERS_GEMMA_DTYPE="bf16",
                   TRANSFORMERS_GEMMA_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38", "bf16"),
                   TRANSFORMERS_GEMMA_LONG_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38-long", "bf16"))
        if test == "long_context_cached_decode_wraps_the_rings":
            env["TRANSFORMERS_GEMMA_MIN_GREEDY"] = "0"
        runs.append((f"gemma.bf16.flex.{test}", "gemma_artifacts", "flex/BF16", "flex_" + test, env, True))
    return runs


def expected_failures_plan():
    """Profiles that cannot pass: Gemma F16 on Metal (every gate; uncalibrated) and Metal BF16 (one gate per family).

    Each is expected to fail before loading, with the uncalibrated-envelope
    message or the contextual load error; none is a gate pass.
    """
    runs = []
    for test in GEMMA_TESTS:
        env = dict(GEMMA_BASE, TRANSFORMERS_GEMMA_DTYPE="f16",
                   TRANSFORMERS_GEMMA_F16_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38", "f16"),
                   TRANSFORMERS_GEMMA_LONG_F16_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38-long", "f16"))
        runs.append((f"uncalibrated.gemma.f16.metal.{test}", "gemma_artifacts", "metal/F16 (uncalibrated)",
                     "metal::metal_" + test, env, True))
    env = dict(GEMMA_BASE, TRANSFORMERS_GEMMA_DTYPE="bf16",
               TRANSFORMERS_GEMMA_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38", "bf16"),
               TRANSFORMERS_GEMMA_LONG_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38-long", "bf16"))
    runs.append(("refusal.gemma.bf16.metal.prompt_logits_match_the_reference", "gemma_artifacts", "metal/BF16 (refused)",
                 "metal::metal_prompt_logits_match_the_reference", env, True))
    return runs


def calibration_plan():
    """Gemma's bundle-only envelope tests (no checkpoint, no Burn model) and the official-variable unit test."""
    runs = []
    gemma = {k: v for k, v in GEMMA_BASE.items() if k != "TRANSFORMERS_GEMMA_DIR"}
    for test in ("official_graphs_calibrate_the_half_precision_envelope", "official_graphs_admit_each_other_leave_one_out"):
        env = dict(gemma, TRANSFORMERS_GEMMA_DTYPE="bf16",
                   TRANSFORMERS_GEMMA_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38", "bf16"),
                   TRANSFORMERS_GEMMA_LONG_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38-long", "bf16"))
        runs.append((f"calibration.gemma.bf16.{test}", "gemma_artifacts", "bundles/BF16", test, env, True))
    env = dict(gemma, TRANSFORMERS_GEMMA_F16_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38", "f16"),
               TRANSFORMERS_GEMMA_LONG_F16_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38-long", "f16"))
    runs.append(("calibration.gemma.f16.official_f16_graphs_overflow_so_gemma_has_no_f16_envelope", "gemma_artifacts",
                 "bundles/F16", "official_f16_graphs_overflow_so_gemma_has_no_f16_envelope", env, True))
    runs.append(("unit.gemma.official_graphs_follow_the_run_dtype", "gemma_artifacts", "unit",
                 "official_graphs_follow_the_run_dtype", {}, False))
    return runs


#: This round's operation-level tests in the library test binary: (name, test path, ignored).
OPERATIONS = [
    ("eager-emulation", "attention::dispatch::tests::half_precision_eager_attention_rounds_like_transformers_eager", False),
    ("eager-scale", "attention::dispatch::tests::half_precision_eager_scores_multiply_by_the_unrounded_scale", False),
    ("eager-folding", "attention::dispatch::tests::folded_groups_match_repeated_key_value_in_reduced_precision", False),
    ("mamba2-silu", "mamba2::tests::convolution_silu_is_rounded_once_in_half_precision", False),
    ("delta-net-silu", "delta_net::tests::convolution_silu_is_rounded_once_in_half_precision", False),
    ("flex-f16-matmul", "operations::tests::flex_f16_matmul_keeps_f16_partial_sums", False),
    ("metal-f16-matmul", "operations::tests::metal_f16_matmul_is_an_f32_sum_rounded_once", True),
    ("metal-eager", "attention::dispatch::tests::metal_half_precision_eager_attention_matches_transformers_eager", True),
]


def operations_plan():
    """This round's operation-level tests, each in its own process (the Metal ones ignored by default)."""
    return [(f"operations.{name}", "lib", "metal" if ignored else "flex", test, {}, ignored)
            for name, test, ignored in OPERATIONS]


def plan(group):
    """Runs of a group."""
    if group == "calibration":
        return calibration_plan()
    if group == "gemma":
        return gemma_plan()
    if group == "expected-failures":
        return expected_failures_plan()
    if group == "operations":
        return operations_plan()
    raise SystemExit(f"unknown group {group}")


def git(*args):
    """Output of a git command in the checkout."""
    return subprocess.run(["git", "-C", CHECKOUT, *args], capture_output=True, text=True, check=True).stdout


def source_identity():
    """(HEAD, number of `git status --porcelain` lines) of the checkout."""
    return git("rev-parse", "HEAD").strip(), len(git("status", "--porcelain").splitlines())


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


def sha256(path):
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def mounted():
    """Whether both external volumes are mounted."""
    out = subprocess.run(["mount"], capture_output=True, text=True).stdout
    return all(f" on {v} " in out for v in ("/Volumes/bag of holding", "/Volumes/T9"))


def run_one(name, target, kind, test, extra, ignored, binaries, summary):
    """Runs one test in its own process with the watchdog; returns the exit status."""
    head, dirty = source_identity()
    if head != EXPECTED or dirty != 0:
        raise SystemExit(f"STOP: checkout at {head} dirty={dirty}, expected {EXPECTED} dirty=0")
    binary = binaries[target]
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRANSFORMERS_")}
    env.update(extra)
    cmd = (["/usr/bin/time", "-l", binary] + (["--ignored"] if ignored else [])
           + ["--exact", "--nocapture", "--test-threads", "1", test])
    path = os.path.join(LOGS, name + ".log")
    if os.path.exists(path):
        raise SystemExit(f"refusing to replace {path}")
    with open(path, "w", buffering=1) as log:
        log.write(f"== source {head} dirty={dirty}\n")
        log.write(f"== checkout {CHECKOUT} (branch {git('branch', '--show-current').strip()})\n")
        log.write(f"== binary {binary} sha256 {sha256(binary)}\n")
        log.write(f"== {kind} {test}\n")
        for key in sorted(extra):
            log.write(f"== env {key}={extra[key]}\n")
        log.write(f"== cwd {CHECKOUT}\n== cmd {' '.join(cmd)}\n")
        load = os.getloadavg()
        log.write(f"== start {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                  f"load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f} free {free_pct()}%\n")
        log.flush()
        swap0, swapin0 = swap_used_mib(), swapins()
        last_swapin, pressured = swapin0, 0
        min_free, max_growth, reason = 100, 0.0, None
        start = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=CHECKOUT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
        end_head, end_dirty = source_identity()
        line = (f"== exit={proc.returncode} wall={wall:.1f}s min_free_pct={min_free} "
                f"max_swap_growth_mib={max_growth:.0f} swapin_mib={(swapins() - swapin0) * PAGE >> 20} "
                f"pressured_swapin_mib={pressured * PAGE >> 20}" + (f" watchdog=\"{reason}\"" if reason else ""))
        log.write(line + "\n")
        log.write(f"== source after run {end_head} dirty={end_dirty}\n")
    text = open(path, errors="replace").read()
    peak = re.findall(r"^\s*(\d+)\s+peak memory footprint", text, re.M)
    summary.write(json.dumps({"name": name, "kind": kind, "test": test, "exit": proc.returncode,
                              "wall_seconds": round(wall, 1), "peak_memory_footprint_bytes": int(peak[-1]) if peak else None,
                              "watchdog": reason, "log": path, "log_sha256": sha256(path),
                              "result": "pass" if "test result: ok. 1 passed" in text else "fail"}) + "\n")
    summary.flush()
    if end_head != EXPECTED or end_dirty != 0:
        raise SystemExit(f"STOP: checkout changed during {name}: {end_head} dirty={end_dirty}")
    return proc.returncode


def terminate(signum, _frame):
    """Turns SIGTERM and SIGHUP into SystemExit, so a held lock is released by the `finally` block."""
    raise SystemExit(128 + signum)


def stamp():
    """The current UTC time in ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    """Runs each named group, taking the heavy-model lock for every real-checkpoint group."""
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    binaries = json.load(open(sys.argv[1]))
    groups = sys.argv[2:]
    os.makedirs(LOGS, exist_ok=True)
    summary = open(os.path.join(LOGS, "runs.jsonl"), "a", buffering=1)
    driver = open(os.path.join(LOGS, "driver.log"), "a", buffering=1)
    for group in groups:
        if not mounted():
            driver.write(f"{stamp()} STOP: a volume is not mounted\n")
            return 5
        needs_lock = group != "calibration"
        held = False
        try:
            if needs_lock:
                driver.write(f"{stamp()} waiting for the lock ({group})\n")
                while True:
                    try:
                        os.mkdir(LOCK)
                        held = True
                        break
                    except FileExistsError:
                        time.sleep(60)
                driver.write(f"{stamp()} lock acquired ({group})\n")
            for run in plan(group):
                status = run_one(*run, binaries, summary)
                driver.write(f"{stamp()} {run[0]} exit={status}\n")
        finally:
            if held:
                os.rmdir(LOCK)
                driver.write(f"{stamp()} lock released ({group})\n")
    driver.write(f"{stamp()} done {' '.join(groups)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
