"""Scratch driver (half2 S2): official F16/BF16 graph bundles from the committed C2 generator (not committed).

Usage: python3 -I make_bundles.py KEY [VARIANT...]

KEY names a truth (F32, F64 for Qwen2-7B). For each variant the generator
exported from commit 7a50e44 (tool/scripts/generate_causal_lm_reference.py,
frozen at TOOL_SHA256) runs in its own process under /usr/bin/time -l with a
memory watchdog, with the truth's prompt texts in order, its greedy length,
--decode-tokens-from the truth, and --prompt-decode for the short truths.
Output: /Volumes/T9/transformers-references/<truth>-<variant>-7a50e44, never
replacing a bundle; the Qwen2-7B F64 decode truth is
Qwen2-7B-453ed157-f64-cpu-eager. The caller holds the heavy-model lock.
Each finished run appends one JSON line to logs/runs.jsonl.
"""

import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "tool" / "scripts" / "generate_causal_lm_reference.py"
TOOL_SHA256 = "a4ed0db48e9b4e6c816bf23c25f56cf39498d06165af49a0655c841d1b56a602"
SOURCE = "7a50e44"
REFERENCES = Path("/Volumes/T9/transformers-references")
LOGS = HERE / "logs"
MIN_FREE = 40 * 2**30
MIN_FREE_MEMORY_PERCENT = 8
CHECKPOINTS = {
    "SmolLM-135M": ("SmolLM-135M-1d461723",
                    "/Volumes/bag of holding/models/HuggingFaceTB/SmolLM-135M-1d461723eec654e65efdc40cf49301c89c0c92f4", True),
    "Qwen3-4B": ("Qwen3-4B-531c80e2", "/Volumes/T9/models/Qwen/Qwen3-4B", True),
    "SmolLM3-3B": ("SmolLM3-3B-1fa6fa28", "/Volumes/T9/models/HuggingFaceTB/SmolLM3-3B", True),
    "Qwen2-7B": ("Qwen2-7B-453ed157-f64", "/Volumes/T9/models/Qwen/Qwen2-7B", True),
    "Llama-3.1-8B": ("Llama-3.1-8B-d04e592b", "/Volumes/T9/models/meta-llama/Llama-3.1-8B", True),
    "gemma-3-4b-it": ("gemma-3-4b-it-093f9f38", "/Volumes/T9/models/google/gemma-3-4b-it", True),
    "gemma-long": ("gemma-3-4b-it-093f9f38-long", "/Volumes/T9/models/google/gemma-3-4b-it", False),
}
VARIANTS = {
    "bf16-cpu-eager": ("bfloat16", "cpu", "eager"), "bf16-cpu-sdpa": ("bfloat16", "cpu", "sdpa"),
    "bf16-mps-eager": ("bfloat16", "mps", "eager"), "bf16-mps-sdpa": ("bfloat16", "mps", "sdpa"),
    "f16-cpu-eager": ("float16", "cpu", "eager"), "f16-cpu-sdpa": ("float16", "cpu", "sdpa"),
    "f16-mps-eager": ("float16", "mps", "eager"), "f16-mps-sdpa": ("float16", "mps", "sdpa"),
    "f64-decode-truth": ("float64", "cpu", "eager"),
}


def sha256(path):
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def free_memory_percent():
    """System-wide free memory percentage reported by memory_pressure, or None."""
    try:
        text = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    for line in text.splitlines():
        if "free percentage" in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("%"))
    return None


def time_figures(log_text):
    """Parses /usr/bin/time -l's maximum resident set, peak footprint (bytes), and real time of the last run."""
    rss = re.findall(r"^\s*(\d+)\s+maximum resident set size", log_text, re.M)
    peak = re.findall(r"^\s*(\d+)\s+peak memory footprint", log_text, re.M)
    real = re.findall(r"^\s*([0-9.]+) real", log_text, re.M)
    return {"maximum_resident_set_bytes": int(rss[-1]) if rss else None,
            "peak_memory_footprint_bytes": int(peak[-1]) if peak else None,
            "time_real_seconds": float(real[-1]) if real else None}


def output_name(truth_name, variant):
    """Bundle directory name of a variant."""
    if variant == "f64-decode-truth":
        return f"{truth_name}-cpu-eager"
    return f"{truth_name}-{variant}-{SOURCE}"


def run(key, variant):
    """Generates one bundle; returns the exit status."""
    truth_name, model_dir, short = CHECKPOINTS[key]
    dtype, device, attention = VARIANTS[variant]
    truth = REFERENCES / truth_name
    output = REFERENCES / output_name(truth_name, variant)
    log_path = LOGS / f"{output.name}.log"
    manifest = json.loads((truth / "reference.json").read_text())
    command = ["/usr/bin/time", "-l", sys.executable, "-I", str(TOOL), "--model-dir", model_dir,
               "--output", str(output), "--repository", manifest["provenance"]["repository"],
               "--revision", manifest["provenance"]["revision"], "--dtype", dtype, "--device", device,
               "--attn-implementation", attention, "--greedy-tokens", str(len(manifest["greedy"]["steps"])),
               "--decode-tokens-from", str(truth)]
    if short and variant != "f64-decode-truth":
        command.append("--prompt-decode")
    for prompt in manifest["prompts"]:
        command += ["--prompt", prompt["text"]]
    free = shutil.disk_usage(REFERENCES).free
    started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    wall, lowest, status, reference_sha = 0.0, [100], None, None
    with open(log_path, "a") as log:
        log.write(f"== source {SOURCE} (git archive export) tool {TOOL} sha256 {sha256(TOOL)}\n")
        log.write(f"== start {started_at} free {free / 2**30:.1f} GiB free-memory {free_memory_percent()}%\n")
        log.write(f"== cmd {json.dumps(command)}\n")
        log.flush()
        if sha256(TOOL) != TOOL_SHA256:
            log.write("REFUSED: the generator changed since it was frozen\n")
            return 3
        if free < MIN_FREE:
            log.write(f"REFUSED: only {free / 2**30:.1f} GiB free on {REFERENCES}\n")
            return 4
        if output.exists():
            log.write(f"SKIP: {output} exists\n")
            return 0
        started = time.monotonic()
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

        def watchdog():
            """Kills the child when free memory falls below the floor."""
            while child.poll() is None:
                percent = free_memory_percent()
                if percent is not None:
                    lowest[0] = min(lowest[0], percent)
                    if percent < MIN_FREE_MEMORY_PERCENT:
                        child.kill()
                        log.write(f"WATCHDOG: free memory {percent}% < {MIN_FREE_MEMORY_PERCENT}%, killed\n")
                        return
                time.sleep(5)

        thread = threading.Thread(target=watchdog, daemon=True)
        thread.start()
        status = child.wait()
        thread.join(timeout=30)
        wall = time.monotonic() - started
        log.write(f"== exit={status} wall={wall:.0f}s lowest-free-memory={lowest[0]}%\n")
        if status == 0:
            reference_sha = sha256(output / "reference.json")
            log.write(f"== reference.json sha256 {reference_sha}\n")
    record = {"key": key, "variant": variant, "bundle": str(output), "status": status, "source": SOURCE,
              "tool_sha256": TOOL_SHA256, "started_at": started_at, "wall_seconds": round(wall, 2),
              "lowest_free_memory_percent": lowest[0], "reference_json_sha256": reference_sha,
              "log": str(log_path), "log_sha256": sha256(log_path)}
    record.update(time_figures(log_path.read_text(errors="replace")))
    with open(LOGS / "runs.jsonl", "a") as runs:
        runs.write(json.dumps(record) + "\n")
    return status


def main():
    """Runs the requested variants of one checkpoint in order."""
    key = sys.argv[1]
    variants = sys.argv[2:] or [v for v in VARIANTS if v != "f64-decode-truth"]
    LOGS.mkdir(parents=True, exist_ok=True)
    worst = 0
    for variant in variants:
        status = run(key, variant)
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} {key} {variant} exit={status}", flush=True)
        worst = worst or status
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
