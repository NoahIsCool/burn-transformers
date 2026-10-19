"""Scratch (half2 S9): summarizes the controlled Qwen3.5-0.8B benchmark rerun, base 7bd057e against head f9f78b5 (not committed).

Usage: python3 -I compare_control.py CONTROL_DIR

Reads CONTROL_DIR/bench/r<round>.<base|head>.<baseline|public>.json, written by
three alternated rounds (base then head, head then base, base then head) of
the same two benchmarks, and prints per metric every round's value for both
binaries, the median over rounds, and the head/base ratio of the medians. Cold
first token and strict load are single samples per run; warm metrics are the
benchmark's own medians over ten measured repetitions. Token and prediction
hashes are compared exactly between the two binaries.
"""

import json
import statistics
import sys
from pathlib import Path


def metrics(report):
    """(label, seconds) pairs of one benchmark report."""
    out = {"cold first token": report["cold_first_token"]["seconds"], "strict load": report["strict_load_seconds"],
           "natural greedy total (median)": report["natural_greedy"]["total_time"]["median_seconds"]}
    for series in report["warm_series"]:
        for metric in ("prefill", "first_token_latency", "decode_model_only", "decode_including_greedy_readback"):
            out[f"source {series['source_length']} {metric} (median)"] = series[metric]["median_seconds"]
    return out


def hashes(report):
    """Exact token identities of one report."""
    natural = report["natural_greedy"]
    out = {"cold": report["cold_first_token"]["tokens_u32le_sha256"],
           "natural": natural.get("tokens_u32le_sha256", natural.get("expected_tokens_u32le_sha256"))}
    for series in report["warm_series"]:
        out[f"source {series['source_length']}"] = series["predicted_tokens_u32le_sha256"]
    return out


def main():
    root = Path(sys.argv[1]) / "bench"
    for kind in ("baseline", "public"):
        print(f"== {kind} benchmark")
        values = {"base": {}, "head": {}}
        identities = {"base": set(), "head": set()}
        for side in ("base", "head"):
            for number in (1, 2, 3):
                report = json.loads((root / f"r{number}.{side}.{kind}.json").read_text())
                for label, seconds in metrics(report).items():
                    values[side].setdefault(label, []).append(seconds)
                identities[side].add(json.dumps(hashes(report), sort_keys=True))
        for label in values["base"]:
            base, head = values["base"][label], values["head"][label]
            ratio = statistics.median(head) / statistics.median(base)
            print(f"   {label}: base {[round(v, 4) for v in base]} median {statistics.median(base):.4f} s | "
                  f"head {[round(v, 4) for v in head]} median {statistics.median(head):.4f} s | head/base {ratio:.3f}")
        same = identities["base"] == identities["head"] and len(identities["base"]) == 1
        print(f"   token and prediction hashes: {'identical across all six runs' if same else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
