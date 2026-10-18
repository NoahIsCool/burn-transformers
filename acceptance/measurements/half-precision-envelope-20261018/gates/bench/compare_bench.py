"""Scratch (half2 S9): compares this round's Qwen3.5-0.8B benchmarks with the approved September 10 baseline (not committed).

Usage: python3 -I compare_bench.py WORKTREE BENCH_DIR

Reads the committed September 10 summaries (qwen-controlled-current-summary.json
for the baseline benchmark, qwen-public-summary.json for the public causal
greedy benchmark) and this round's benchmark JSON files, and prints, for every
median timing both reports carry, the relative change (positive = slower),
flagged ALERT above 5% and REGRESSION-CANDIDATE above 10% (BENCHMARKS.md: a
greater-than-10% regression fails only after a controlled rerun confirms it on
the same benchmark series). Token identities and cosines are compared exactly.
"""

import json
import sys
from pathlib import Path


def benchmark_of(summary):
    """The benchmark report inside a committed summary (or the report itself)."""
    return summary.get("benchmark", summary) if isinstance(summary.get("benchmark"), dict) else summary


def medians(report):
    """Every (label, median seconds) pair of a benchmark report."""
    out = {}
    out["cold_first_token"] = report["cold_first_token"]["seconds"]
    out["strict_load"] = report["strict_load_seconds"]
    out["natural_greedy_total"] = report["natural_greedy"]["total_time"]["median_seconds"]
    for series in report["warm_series"]:
        for metric in ("prefill", "first_token_latency", "decode_model_only", "decode_including_greedy_readback"):
            if isinstance(series.get(metric), dict):
                out[f"source {series['source_length']} {metric}"] = series[metric]["median_seconds"]
    return out


def identities(report):
    """Token hashes and reference cosines that must not move without a numerics change."""
    cold = report["cold_first_token"]
    natural = report["natural_greedy"]
    ids = {
        "cold tokens": cold["tokens_u32le_sha256"],
        "natural tokens": natural.get("tokens_u32le_sha256", natural.get("expected_tokens_u32le_sha256")),
        "prompt cosine": cold["prompt_reference_cosine"],
        "minimum cached cosine": cold["minimum_cached_reference_cosine"],
        "minimum teacher cosine": cold["minimum_teacher_reference_cosine"],
    }
    for series in report["warm_series"]:
        ids[f"source {series['source_length']} predictions"] = series["predicted_tokens_u32le_sha256"]
    if "per_repetition_outputs" in natural:
        ids["public log probabilities"] = natural["per_repetition_outputs"][0]["output"]["log_probabilities"]
    return ids


def main():
    worktree, bench = Path(sys.argv[1]), Path(sys.argv[2])
    base = worktree / "acceptance/measurements/small-model-performance-20260910"
    pairs = [("baseline", base / "qwen-controlled-current-summary.json", bench / "qwen35-baseline-benchmark.json"),
             ("public causal greedy", base / "qwen-public-summary.json", bench / "qwen35-public-causal-greedy-benchmark.json")]
    worst = float("-inf")
    for name, old_path, new_path in pairs:
        old, new = benchmark_of(json.loads(old_path.read_text())), json.loads(new_path.read_text())
        print(f"== {name}: {old_path.name} (September 10) against {new_path.name}")
        print(f"   benchmark ids: {old['benchmark']} / {new['benchmark']}")
        a, b = medians(old), medians(new)
        for label in a:
            if label not in b:
                print(f"   {label}: missing in the new report")
                continue
            change = b[label] / a[label] - 1
            flag = "REGRESSION-CANDIDATE" if change > 0.10 else ("ALERT" if change > 0.05 else "ok")
            if not label.startswith(("cold", "strict")):
                worst = max(worst, change)
            print(f"   {label}: {a[label]:.6f} s -> {b[label]:.6f} s ({change * 100:+.1f}%) {flag}")
        ia, ib = identities(old), identities(new)
        for label in ia:
            same = "identical" if ia[label] == ib.get(label) else "DIFFERENT"
            if isinstance(ia[label], list):
                gap = max(abs(x - y) for x, y in zip(ia[label], ib[label]))
                print(f"   {label}: {len(ia[label])} values, max |change| {gap:.3g} {same}")
            else:
                print(f"   {label}: {ia[label]} -> {ib.get(label)} {same}")
    print(f"== most positive warm or natural-loop median change {worst * 100:+.1f}% (cold first token and strict load are single samples)")


if __name__ == "__main__":
    main()
