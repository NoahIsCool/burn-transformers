"""Scratch (rerun at 8181605): reruns the half-precision real-checkpoint gates one process at a time (not committed).

Usage: python3 -I run_rerun.py BINARIES_JSON GROUP...

A copy of half2/gates/run_gates.py (the official-envelope round's runner) with
three changes: the tests run from the worktree checkout itself, every log's
first line is `== source <HEAD> dirty=<n>` from that checkout (a run refuses to
start unless HEAD is the expected commit and the checkout is clean), and the
groups add the refusal checks (Gemma F16 on Metal, Metal BF16). Every
real-checkpoint group takes /Users/nover/repos/transformers-worktrees/.heavy-model.lock
(mkdir, retried every 60 s) and releases it when the group ends, also on failure
or interruption. Each test runs in its own process under the earlier rounds'
memory watchdog. One JSON line per run is appended to logs/runs.jsonl.
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
EXPECTED = "818160545802c1374f7607d26ce20cc86d2d758d"
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


def refusal_plan():
    """Profiles the implementation did not keep: Gemma F16 on Metal (every gate) and Metal BF16 (one gate per family).

    Each is expected to fail with its contextual refusal; none is a gate pass.
    """
    runs = []
    for test in GEMMA_TESTS:
        env = dict(GEMMA_BASE, TRANSFORMERS_GEMMA_DTYPE="f16",
                   TRANSFORMERS_GEMMA_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38", "f16"),
                   TRANSFORMERS_GEMMA_LONG_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38-long", "f16"))
        runs.append((f"refusal.gemma.f16.metal.{test}", "gemma_artifacts", "metal/F16 (refused)", "metal::metal_" + test,
                     env, True))
    env = dict(GEMMA_BASE, TRANSFORMERS_GEMMA_DTYPE="bf16",
               TRANSFORMERS_GEMMA_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38", "bf16"),
               TRANSFORMERS_GEMMA_LONG_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38-long", "bf16"))
    runs.append(("refusal.gemma.bf16.metal.prompt_logits_match_the_reference", "gemma_artifacts", "metal/BF16 (refused)",
                 "metal::metal_prompt_logits_match_the_reference", env, True))
    env = dict(llama_base("SmolLM-135M"), TRANSFORMERS_LLAMA_FAMILY_DTYPE="bf16",
               TRANSFORMERS_LLAMA_FAMILY_OFFICIAL_DIRS=official("SmolLM-135M-1d461723", "bf16"))
    runs.append(("refusal.SmolLM-135M.bf16.metal.prompt_logits_match_the_reference", "llama_family_artifacts",
                 "metal/BF16 (refused)", "metal::metal_prompt_logits_match_the_reference", env, True))
    return runs


def calibration_plan():
    """Bundle-only envelope tests (no checkpoint, no Burn model) and the non-ignored Gemma F16 refusal unit test."""
    runs = []
    for name, _, truth, decode, *_ in LLAMA:
        for dtype in ("bf16", "f16"):
            for test in ("official_graphs_calibrate_the_half_precision_envelope",
                         "official_graphs_admit_each_other_leave_one_out"):
                env = {"TRANSFORMERS_LLAMA_FAMILY_DTYPE": dtype, "TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR": f"{R}/{truth}",
                       "TRANSFORMERS_LLAMA_FAMILY_DECODE_TRUTH_DIR": f"{R}/{decode}",
                       "TRANSFORMERS_LLAMA_FAMILY_OFFICIAL_DIRS": official(truth, dtype)}
                runs.append((f"calibration.{name}.{dtype}.{test}", "llama_family_artifacts", f"bundles/{dtype.upper()}",
                             test, env, True))
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
    runs.append(("gemma.f16.refusal-unit.f16_runs_are_refused_with_the_measured_overflow", "gemma_artifacts", "unit",
                 "f16_runs_are_refused_with_the_measured_overflow", {}, False))
    return runs


def plan(group):
    """Runs of a group."""
    if group == "calibration":
        return calibration_plan()
    if group == "gemma":
        return gemma_plan()
    if group == "refusals":
        return refusal_plan()
    return llama_plan(group)


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
