"""Scratch driver (not committed): every heavy run of the Gemma 3 half-precision follow-up, under the lock.

Usage: python3 -I heavy_driver.py OUT_DIR

Takes /Users/nover/repos/transformers-worktrees/.heavy-model.lock with the mkdir protocol (polling
every 60 s), runs each step in its own process under /usr/bin/time -l with a watchdog, writes one
log per step, and releases the lock when done, including on failure or interruption.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path("/Users/nover/repos/transformers-worktrees/.heavy-model.lock")
G = Path("/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/followups/gemma")
CT = Path("/Volumes/bag of holding/cargo-targets")
UNPATCHED = CT / "followups-gemma/release/deps"
PATCHED = CT / "followups-gemma-patched/release/deps"
BURN022 = CT / "followups-gemma-burn022/release/burn022-metal-bf16-probe"
GATE = UNPATCHED / "gemma_artifacts-7e9fb558e3797b0a"
PROBE = UNPATCHED / "metal_bf16_probe-5c2b593db1bdc288"
GATE_P = PATCHED / "gemma_artifacts-34288ae37266667d"
PROBE_P = PATCHED / "metal_bf16_probe-e1d5bee79a62a658"
REF = Path("/Volumes/T9/transformers-references")
MODELS = Path("/Volumes/T9/models/google")
SCRIPTS = G / "scripts2"

GEMMA_ENV = {
    "TRANSFORMERS_GEMMA_DIR": str(MODELS / "gemma-3-4b-it"),
    "TRANSFORMERS_GEMMA_REFERENCE_DIR": str(REF / "gemma-3-4b-it-093f9f38"),
    "TRANSFORMERS_GEMMA_BF16_REFERENCE_DIR": str(REF / "gemma-3-4b-it-093f9f38-bf16"),
    "TRANSFORMERS_GEMMA_LONG_REFERENCE_DIR": str(REF / "gemma-3-4b-it-093f9f38-long"),
    "TRANSFORMERS_GEMMA_LONG_BF16_REFERENCE_DIR": str(REF / "gemma-3-4b-it-093f9f38-long-bf16"),
}
METAL_TESTS = [
    "metal::metal_prompt_logits_match_the_reference",
    "metal::metal_cached_greedy_matches_the_reference",
    "metal::metal_batched_prefill_matches_the_reference",
    "metal::metal_long_context_logits_cross_the_sliding_window",
    "metal::metal_long_context_cached_decode_wraps_the_rings",
]
PY_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}


def steps(out):
    """Every step as (name, argv, extra environment, timeout seconds)."""
    plan = []
    test_args = ["--ignored", "--exact", "--nocapture", "--test-threads", "1"]
    plan.append(("probe-unpatched", [str(PROBE), *test_args, "metal_bf16_capabilities"], {}, 600))
    plan.append(("probe-unpatched-kernel-log", [str(PROBE), *test_args, "metal_bf16_capabilities"],
                 {"CUBECL_DEBUG_LOG": str(out / "cubecl-kernels-unpatched.log")}, 600))
    plan.append(("probe-patched", [str(PROBE_P), *test_args, "metal_bf16_capabilities"], {}, 600))
    plan.append(("probe-burn022", [str(BURN022)], {}, 600))
    for test in METAL_TESTS:
        extra = {"TRANSFORMERS_GEMMA_DTYPE": "bf16", **GEMMA_ENV}
        if "long_context_cached" in test:
            extra["TRANSFORMERS_GEMMA_MIN_GREEDY"] = "0"
        plan.append((f"gate-bf16-unpatched.{test.split('::')[-1]}", [str(GATE), *test_args, test], extra, 900))
    plan.append(("gate-f16-unpatched.metal_prompt_logits_match_the_reference",
                 [str(GATE), *test_args, METAL_TESTS[0]], {"TRANSFORMERS_GEMMA_DTYPE": "f16", **GEMMA_ENV}, 900))
    for test in METAL_TESTS:
        extra = {"TRANSFORMERS_GEMMA_DTYPE": "bf16", **GEMMA_ENV}
        if "long_context_cached" in test:
            extra["TRANSFORMERS_GEMMA_MIN_GREEDY"] = "0"
        plan.append((f"gate-bf16-patched.{test.split('::')[-1]}", [str(GATE_P), *test_args, test], extra, 900))
    py = [sys.executable, "-I"]
    for size in ["4b", "12b", "27b"]:
        plan.append((f"weights-{size}", [*py, str(SCRIPTS / "gemma3_weight_ranges.py"), str(MODELS / f"gemma-3-{size}-it"),
                                          str(out / f"weights-{size}.json")], PY_ENV, 1800))
    short = REF / "gemma-3-4b-it-093f9f38"
    points = str(SCRIPTS / "gemma3_overflow_points.py")
    for size, dtype, bundle, flags in [
        ("4b", "float32", short, ["--no-bos"]),
        ("4b", "float16", short, ["--no-bos"]),
        ("4b", "bfloat16", short, []),
        ("4b", "float32", REF / "gemma-3-4b-it-093f9f38-long", []),
    ]:
        tag = "long-" if "long" in bundle.name else ""
        plan.append((f"points-{size}-{tag}{dtype}", [*py, points, str(MODELS / f"gemma-3-{size}-it"), str(bundle), dtype,
                                                      str(out / f"points-{size}-{tag}{dtype}.json"), *flags], PY_ENV, 1800))
    plan.append(("clamp-4b", [*py, str(SCRIPTS / "gemma3_clamped_variants.py"), str(MODELS / "gemma-3-4b-it"), str(short),
                              str(REF / "gemma-3-4b-it-093f9f38-bf16"), str(out / "clamp-4b.json"),
                              "float16", "float16-clamp", "float32-clamp", "bfloat16"], PY_ENV, 1800))
    for size in ["12b", "27b"]:
        for dtype, flags in [("bfloat16", ["--no-bos"]), ("float16", [])]:
            plan.append((f"points-{size}-{dtype}", [*py, points, str(MODELS / f"gemma-3-{size}-it"), str(short), dtype,
                                                    str(out / f"points-{size}-{dtype}.json"), *flags], PY_ENV, 2400))
    return plan


def run(name, argv, extra, timeout, out):
    """Runs one step in its own process group with a watchdog; returns the exit code."""
    log = out / f"{name}.log"
    env = dict(os.environ, **extra)
    with log.open("w") as handle:
        handle.write(f"== step {name}\n== start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                     f"load {os.getloadavg()}\n== env {extra}\n== cmd {argv}\n")
        handle.flush()
        if not Path(argv[0]).exists():
            handle.write(f"== SKIPPED: {argv[0]} does not exist\n")
            return None
        started = time.monotonic()
        process = subprocess.Popen(["/usr/bin/time", "-l", *argv], stdout=handle, stderr=subprocess.STDOUT,
                                   env=env, cwd=str(G), start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            handle.write(f"\n== WATCHDOG: killed after {timeout} s\n")
        handle.write(f"\n== exit={code} wall={time.monotonic() - started:.1f}s end "
                     f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    return code


def terminate(signum, _frame):
    """Turns SIGTERM/SIGHUP into SystemExit so the lock is released in the finally block."""
    raise SystemExit(f"signal {signum}")


def main():
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "driver.log"
    with summary.open("a") as handle:
        handle.write(f"== waiting for the lock at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    while True:
        try:
            LOCK.mkdir()
            break
        except FileExistsError:
            time.sleep(60)
    try:
        with summary.open("a") as handle:
            handle.write(f"== lock taken at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        for binary in [GATE, PROBE, GATE_P, PROBE_P, BURN022]:
            if binary.exists():
                digest = subprocess.run(["shasum", "-a", "256", str(binary)], capture_output=True, text=True).stdout.split()[0]
                with summary.open("a") as handle:
                    handle.write(f"== sha256 {digest} {binary}\n")
        for name, argv, extra, timeout in steps(out):
            code = run(name, argv, extra, timeout, out)
            with summary.open("a") as handle:
                handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {name}: exit={code}\n")
    finally:
        LOCK.rmdir()
        with summary.open("a") as handle:
            handle.write(f"== lock released at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")


if __name__ == "__main__":
    main()
