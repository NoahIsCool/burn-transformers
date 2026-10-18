"""Scratch (half2 S9): runs the real-checkpoint gates one process at a time under the heavy-model lock (not committed).

Usage: python3 -I run_gates.py BINARIES_JSON GROUP...

BINARIES_JSON maps a build label (A, B, C, D) to {test target: binary path}.
Each GROUP takes /Users/nover/repos/transformers-worktrees/.heavy-model.lock
(mkdir, retried every 60 s), runs its tests one process each, and releases the
lock when it ends, also on failure or interruption. A run's log starts with the
export's source identity, the binary and its SHA-256, the dtype/backend label,
every TRANSFORMERS_* variable, and the exact command; it holds the complete
output and the full `/usr/bin/time -l` summary, and ends with an `== exit=`
line carrying wall time and the watchdog's memory figures. The watchdog kills
a run on free memory below 10%, swap growth above 8 GiB, or swap-ins above
1 GiB while free memory is below 30% (the earlier replay records' policy).
Both external volumes must be mounted before each group. One JSON line per
run is appended to logs/runs.jsonl.
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

G = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(G, "src")
EXPORT_SHA = open(os.path.join(G, "src.EXPORT")).read().strip()
LOGS = os.path.join(G, "logs")
LOCK = "/Users/nover/repos/transformers-worktrees/.heavy-model.lock"
PAGE = 16384
R = "/Volumes/T9/transformers-references"
S = "7a50e44"
M = "/Volumes/bag of holding/models"
ACC = f"{M}/transformers-acceptance"

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


def official(truth, dtype):
    """The four official graph bundles of a truth, as a PATH-style list."""
    return ":".join(f"{R}/{truth}-{dtype}-{d}-{a}-{S}" for d in ("cpu", "mps") for a in ("eager", "sdpa"))


def llama_plan(name):
    """Runs of one Llama-family checkpoint: F32 Flex and Metal, Flex/BF16, Metal/F16."""
    _, ckpt, truth, decode, tol, bf16_min, f16_min = next(c for c in LLAMA if c[0] == name)
    base = {"TRANSFORMERS_LLAMA_FAMILY_DIR": ckpt, "TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR": f"{R}/{truth}",
            "TRANSFORMERS_LLAMA_FAMILY_DECODE_TRUTH_DIR": f"{R}/{decode}"}
    runs = []
    for backend, prefix in (("flex", "flex_"), ("metal", "metal::metal_")):
        for test in LLAMA_TESTS:
            runs.append((f"{name}.f32.{backend}.{test}", "A", "llama_family_artifacts", f"{backend}/F32", prefix + test,
                         dict(base, TRANSFORMERS_LLAMA_FAMILY_TOLERANCE=tol)))
    for backend, prefix, dtype, minimum in (("flex", "flex_", "bf16", bf16_min), ("metal", "metal::metal_", "f16", f16_min)):
        for test in LLAMA_TESTS:
            env = dict(base, TRANSFORMERS_LLAMA_FAMILY_DTYPE=dtype,
                       TRANSFORMERS_LLAMA_FAMILY_OFFICIAL_DIRS=official(truth, dtype))
            if minimum is not None and test == "cached_greedy_matches_the_reference":
                env["TRANSFORMERS_LLAMA_FAMILY_MIN_GREEDY"] = minimum
            runs.append((f"{name}.{dtype}.{backend}.{test}", "A", "llama_family_artifacts",
                         f"{backend}/{dtype.upper()}", prefix + test, env))
    return runs


def gemma_plan():
    """Runs of gemma-3-4b-it: F32 Flex and Metal, Flex/BF16 (F16 is refused)."""
    base = {"TRANSFORMERS_GEMMA_DIR": "/Volumes/T9/models/google/gemma-3-4b-it",
            "TRANSFORMERS_GEMMA_REFERENCE_DIR": GEMMA_TRUTH,
            "TRANSFORMERS_GEMMA_DECODE_TRUTH_DIR": f"{GEMMA_TRUTH}-f32-cpu-eager",
            "TRANSFORMERS_GEMMA_LONG_REFERENCE_DIR": f"{GEMMA_TRUTH}-long",
            "TRANSFORMERS_GEMMA_LONG_DECODE_TRUTH_DIR": f"{GEMMA_TRUTH}-long-f32-cpu-eager"}
    runs = []
    for backend, prefix in (("flex", "flex_"), ("metal", "metal::metal_")):
        for test in GEMMA_TESTS:
            runs.append((f"gemma.f32.{backend}.{test}", "A", "gemma_artifacts", f"{backend}/F32", prefix + test, dict(base)))
    for test in GEMMA_TESTS:
        env = dict(base, TRANSFORMERS_GEMMA_DTYPE="bf16",
                   TRANSFORMERS_GEMMA_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38", "bf16"),
                   TRANSFORMERS_GEMMA_LONG_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38-long", "bf16"))
        if test == "long_context_cached_decode_wraps_the_rings":
            env["TRANSFORMERS_GEMMA_MIN_GREEDY"] = "0"
        runs.append((f"gemma.bf16.flex.{test}", "A", "gemma_artifacts", "flex/BF16", "flex_" + test, env))
    return runs


def calibration_plan():
    """Bundle-only envelope tests (no checkpoint, no Burn model)."""
    runs = []
    for name, _, truth, decode, *_ in LLAMA:
        for dtype in ("bf16", "f16"):
            for test in ("official_graphs_calibrate_the_half_precision_envelope",
                         "official_graphs_admit_each_other_leave_one_out"):
                env = {"TRANSFORMERS_LLAMA_FAMILY_DTYPE": dtype, "TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR": f"{R}/{truth}",
                       "TRANSFORMERS_LLAMA_FAMILY_DECODE_TRUTH_DIR": f"{R}/{decode}",
                       "TRANSFORMERS_LLAMA_FAMILY_OFFICIAL_DIRS": official(truth, dtype)}
                runs.append((f"calibration.{name}.{dtype}.{test}", "A", "llama_family_artifacts", f"bundles/{dtype.upper()}", test, env))
    gemma = {"TRANSFORMERS_GEMMA_REFERENCE_DIR": GEMMA_TRUTH,
             "TRANSFORMERS_GEMMA_DECODE_TRUTH_DIR": f"{GEMMA_TRUTH}-f32-cpu-eager",
             "TRANSFORMERS_GEMMA_LONG_REFERENCE_DIR": f"{GEMMA_TRUTH}-long",
             "TRANSFORMERS_GEMMA_LONG_DECODE_TRUTH_DIR": f"{GEMMA_TRUTH}-long-f32-cpu-eager"}
    for test in ("official_graphs_calibrate_the_half_precision_envelope", "official_graphs_admit_each_other_leave_one_out"):
        env = dict(gemma, TRANSFORMERS_GEMMA_DTYPE="bf16",
                   TRANSFORMERS_GEMMA_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38", "bf16"),
                   TRANSFORMERS_GEMMA_LONG_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38-long", "bf16"))
        runs.append((f"calibration.gemma.bf16.{test}", "A", "gemma_artifacts", "bundles/BF16", test, env))
    env = dict(gemma, TRANSFORMERS_GEMMA_F16_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38", "f16"),
               TRANSFORMERS_GEMMA_LONG_F16_OFFICIAL_DIRS=official("gemma-3-4b-it-093f9f38-long", "f16"))
    runs.append(("calibration.gemma.f16.official_f16_graphs_overflow_so_gemma_has_no_f16_envelope", "A", "gemma_artifacts",
                 "bundles/F16", "official_f16_graphs_overflow_so_gemma_has_no_f16_envelope", env))
    runs.append(("gemma.f16.refusal.f16_runs_are_refused_with_the_measured_overflow", "A", "gemma_artifacts", "unit",
                 "f16_runs_are_refused_with_the_measured_overflow", {}))
    return runs


def large_plan(group):
    """Runs of the large checkpoints whose numerics the SiLU and eager changes touch."""
    q35 = {"TRANSFORMERS_QWEN35_DIR": f"{M}/qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17",
           "TRANSFORMERS_QWEN35_REFERENCE_DIR": ACC}
    q38 = {"TRANSFORMERS_QWEN38_DIR": f"{M}/Qwen/Qwen3.8-27B-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
           "TRANSFORMERS_QWEN38_REFERENCE_DIR": f"{ACC}/qwen3.8-27b-official-reference-1d4bf0f2"}
    muse = {"TRANSFORMERS_MUSE_GLIMMER_DIR": f"{M}/meta-models/Muse-Glimmer-30B-a4e59da52a7bc87ae7251dd5545c0dd437c44b68"}
    qmoe = {"TRANSFORMERS_QWEN3_MOE_DIR": f"{M}/Qwen/Qwen3-30B-A3B-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"}
    nemo = {"TRANSFORMERS_NEMOTRON_H_DIR": f"{M}/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-a9904d24bcc1d289a1950fa9d2b978c47cf903b9"}
    plans = {
        "qwen35": [("qwen3.5-0.8b.f16.metal.parity", "B", "local_artifacts", "metal/F16",
                    "loads_pinned_qwen35_weights_and_runs_metal_inference", q35)],
        "qwen35-bench": [("qwen3.5-0.8b.f16.metal.baseline-benchmark", "B", "qwen35_benchmarks", "metal/F16",
                          "qwen::qwen35_metal_baseline_benchmark",
                          dict(q35, TRANSFORMERS_BENCH_METADATA_JSON=f"{G}/bench/environment-baseline.json",
                               TRANSFORMERS_BENCH_OUTPUT_JSON=f"{G}/bench/qwen35-baseline-benchmark.json")),
                         ("qwen3.5-0.8b.f16.metal.public-causal-greedy-benchmark", "B", "qwen35_benchmarks", "metal/F16",
                          "qwen::qwen35_metal_public_causal_greedy_benchmark",
                          dict(q35, TRANSFORMERS_BENCH_METADATA_JSON=f"{G}/bench/environment-public.json",
                               TRANSFORMERS_BENCH_OUTPUT_JSON=f"{G}/bench/qwen35-public-causal-greedy-benchmark.json"))],
        "qwen38": [("qwen3.8-27b.f16.metal.official", "B", "local_artifacts", "metal/F16",
                    "matches_pinned_qwen38_official_reference_on_metal", q38),
                   ("qwen3.8-27b.f16.metal.mtp-smoke", "B", "local_artifacts", "metal/F16",
                    "loads_pinned_qwen38_text_and_mtp_and_runs_metal", q38)],
        "muse": [("muse-glimmer-30b.f16.metal.official", "D", "muse_glimmer_artifacts", "metal/F16",
                  "matches_pinned_muse_glimmer_official_f16_reference_on_metal",
                  dict(muse, TRANSFORMERS_MUSE_GLIMMER_REFERENCE_DIR=ACC)),
                 ("muse-glimmer-30b.f16.metal.streaming", "B", "muse_glimmer_artifacts", "metal/F16",
                  "loads_pinned_muse_glimmer_f16_and_audits_streaming", muse),
                 ("muse-glimmer-30b.q8.metal.smoke", "C", "muse_glimmer_artifacts", "metal/Q8",
                  "smokes_and_audits_pinned_muse_glimmer_q8_burnpack_on_metal",
                  dict(muse, TRANSFORMERS_MUSE_GLIMMER_Q8_BURNPACK=f"{ACC}/muse-glimmer-30b-q8-a4e59da5/muse-glimmer-30b-a4e59da5-q8-g64-v2.bpk")),
                 ("muse-glimmer-30b.q4.metal.smoke", "C", "muse_glimmer_artifacts", "metal/Q4",
                  "smokes_and_audits_pinned_muse_glimmer_q4_burnpack_on_metal",
                  dict(muse, TRANSFORMERS_MUSE_GLIMMER_Q4_BURNPACK=f"{ACC}/muse-glimmer-30b-q4-a4e59da5/muse-glimmer-30b-a4e59da5-q4-g64-v2.bpk"))],
        "deepseek": [("deepseek-v2-lite.metadata", "B", "deepseek_v2_artifacts", "host",
                      "validates_pinned_deepseek_v2_lite_config_headers_plan_memory_and_tokenizer",
                      {"TRANSFORMERS_DEEPSEEK_V2_DIR": f"{M}/deepseek-ai/DeepSeek-V2-Lite-604d5664dddd88a0433dbae533b7fe9472482de0"}),
                     ("deepseek-v2-lite.f16.metal.smoke", "B", "deepseek_v2_artifacts", "metal/F16",
                      "loads_pinned_deepseek_v2_lite_f16_and_runs_metal_smoke",
                      {"TRANSFORMERS_DEEPSEEK_V2_DIR": f"{M}/deepseek-ai/DeepSeek-V2-Lite-604d5664dddd88a0433dbae533b7fe9472482de0"})],
        "qwen3moe": [("qwen3-30b-a3b.f16.metal.smoke", "B", "qwen3_moe_artifacts", "metal/F16",
                      "loads_pinned_qwen3_moe_f16_and_runs_metal_smoke", qmoe),
                     ("qwen3-30b-a3b.q8.metal.smoke", "B", "qwen3_moe_artifacts", "metal/Q8",
                      "loads_pinned_qwen3_moe_q8_and_runs_metal_smoke",
                      dict(qmoe, TRANSFORMERS_QWEN3_MOE_Q8_BURNPACK=f"{ACC}/qwen3-30b-a3b-q8-ad44e777/qwen3-30b-a3b-ad44e777-q8-g64-v2.bpk")),
                     ("qwen3-30b-a3b.q4.metal.smoke", "B", "qwen3_moe_artifacts", "metal/Q4",
                      "loads_pinned_qwen3_moe_q4_and_runs_metal_smoke",
                      dict(qmoe, TRANSFORMERS_QWEN3_MOE_Q4_BURNPACK=f"{ACC}/qwen3-30b-a3b-q4-ad44e777/qwen3-30b-a3b-ad44e777-q4-g64-v2.bpk")),
                     ("qwen3-30b-a3b.q8.metal.audit", "C", "qwen3_moe_artifacts", "metal/Q8",
                      "audits_pinned_qwen3_moe_q8_decode_packed_metal_kernels",
                      dict(qmoe, TRANSFORMERS_QWEN3_MOE_Q8_BURNPACK=f"{ACC}/qwen3-30b-a3b-q8-ad44e777/qwen3-30b-a3b-ad44e777-q8-g64-v2.bpk")),
                     ("qwen3-30b-a3b.q4.metal.audit", "C", "qwen3_moe_artifacts", "metal/Q4",
                      "audits_pinned_qwen3_moe_q4_decode_packed_metal_kernels",
                      dict(qmoe, TRANSFORMERS_QWEN3_MOE_Q4_BURNPACK=f"{ACC}/qwen3-30b-a3b-q4-ad44e777/qwen3-30b-a3b-ad44e777-q4-g64-v2.bpk"))],
        "nemotron": [("nemotron-h.f16.metal.smoke", "B", "nemotron_h_artifacts", "metal/F16",
                      "loads_pinned_nemotron_f16_and_smokes_base_inference", nemo),
                     ("nemotron-h.f16.metal.official-precision", "B", "nemotron_h_artifacts", "metal/F16",
                      "compares_pinned_nemotron_f16_with_official_base_reference",
                      dict(nemo, TRANSFORMERS_NEMOTRON_H_REFERENCE_DIR=f"{ACC}/nemotron-3.5-lightning-30b-a3b-a9904d24-reference.bundle",
                           TRANSFORMERS_NEMOTRON_H_PRECISION_DIR=f"{ACC}/nemotron-3.5-lightning-30b-a3b-a9904d24-precision.bundle"))],
        "qwen38-quantized": [("qwen3.8-27b.q8.metal.convert-and-smoke", "B", "local_artifacts", "host/Q8 + metal",
                              "quantizes_pinned_qwen38_weights_to_q8_burnpack",
                              dict(q38, TRANSFORMERS_QWEN38_Q8_OUTPUT_DIR=f"{ACC}/half-precision-envelope-20261018/qwen3.8-27b-q8-1d4bf0f2")),
                             ("qwen3.8-27b.q4.metal.convert-and-smoke", "B", "local_artifacts", "host/Q4 + metal",
                              "quantizes_pinned_qwen38_weights_to_q4_burnpack",
                              dict(q38, TRANSFORMERS_QWEN38_Q4_OUTPUT_DIR=f"{ACC}/half-precision-envelope-20261018/qwen3.8-27b-q4-1d4bf0f2"))],
    }
    return plans[group]


def plan(group):
    """Runs of a group."""
    if group == "calibration":
        return calibration_plan()
    if group == "gemma":
        return gemma_plan()
    names = [c[0] for c in LLAMA]
    if group in names:
        return llama_plan(group)
    return large_plan(group)


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


def run_one(name, label, target, kind, test, extra, binaries, summary):
    """Runs one test in its own process with the watchdog; returns the exit status."""
    binary = binaries[label][target]
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRANSFORMERS_")}
    env.update(extra)
    cmd = ["/usr/bin/time", "-l", binary, "--ignored", "--exact", "--nocapture", "--test-threads", "1", test]
    path = os.path.join(LOGS, name + ".log")
    with open(path, "w", buffering=1) as log:
        log.write(f"== source {EXPORT_SHA} (git archive export {SRC})\n")
        log.write(f"== binary {binary} sha256 {sha256(binary)} build {label}\n")
        log.write(f"== {kind} {test}\n")
        for key in sorted(extra):
            log.write(f"== env {key}={extra[key]}\n")
        log.write(f"== cwd {SRC}\n== cmd {' '.join(cmd)}\n")
        load = os.getloadavg()
        log.write(f"== start {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                  f"load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f} free {free_pct()}%\n")
        log.flush()
        swap0, swapin0 = swap_used_mib(), swapins()
        last_swapin, pressured = swapin0, 0
        min_free, max_growth, reason = 100, 0.0, None
        start = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
        line = (f"== exit={proc.returncode} wall={wall:.1f}s min_free_pct={min_free} "
                f"max_swap_growth_mib={max_growth:.0f} swapin_mib={(swapins() - swapin0) * PAGE >> 20} "
                f"pressured_swapin_mib={pressured * PAGE >> 20}" + (f" watchdog=\"{reason}\"" if reason else ""))
        log.write(line + "\n")
    text = open(path, errors="replace").read()
    peak = re.findall(r"^\s*(\d+)\s+peak memory footprint", text, re.M)
    summary.write(json.dumps({"name": name, "kind": kind, "test": test, "exit": proc.returncode,
                              "wall_seconds": round(wall, 1), "peak_memory_footprint_bytes": int(peak[-1]) if peak else None,
                              "watchdog": reason, "log": path, "log_sha256": sha256(path),
                              "result": "pass" if "test result: ok. 1 passed" in text else "fail"}) + "\n")
    summary.flush()
    return proc.returncode


def terminate(signum, _frame):
    """Turns SIGTERM and SIGHUP into SystemExit, so a held lock is released by the `finally` block."""
    raise SystemExit(128 + signum)


def main():
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    binaries = json.load(open(sys.argv[1]))
    groups = sys.argv[2:]
    os.makedirs(LOGS, exist_ok=True)
    summary = open(os.path.join(LOGS, "runs.jsonl"), "a", buffering=1)
    driver = open(os.path.join(LOGS, "driver.log"), "a", buffering=1)
    for group in groups:
        if not mounted():
            driver.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} STOP: a volume is not mounted\n")
            return 5
        needs_lock = group != "calibration"
        held = False
        try:
            if needs_lock:
                driver.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} waiting for the lock ({group})\n")
                while True:
                    try:
                        os.mkdir(LOCK)
                        held = True
                        break
                    except FileExistsError:
                        time.sleep(60)
                driver.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} lock acquired ({group})\n")
            for run in plan(group):
                status = run_one(*run, binaries, summary)
                driver.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {run[0]} exit={status}\n")
        finally:
            if held:
                os.rmdir(LOCK)
                driver.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} lock released ({group})\n")
    driver.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} done {' '.join(groups)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
