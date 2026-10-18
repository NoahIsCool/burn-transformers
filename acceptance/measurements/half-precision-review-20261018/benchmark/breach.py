"""Review round: the Qwen3.5-0.8B cold first token against the approved September 10 baseline.

Usage: python3 -I breach.py WORKTREE

Reads only committed files: the approved baseline summaries
(``small-model-performance-20260910/qwen-controlled-current-summary.json`` and
``qwen-public-summary.json``, authenticated by that directory's manifest), and the official-envelope round's gate-run
benchmarks (``half-precision-envelope-20261018/gates/bench/``) and controlled
rerun (``gates/control/bench/r<round>.<base|head>.<kind>.json``), each
authenticated first by the digest its record lists. For both benchmarks it
prints every cold-first-token sample (a single sample per run) with its
change from the approved baseline, flagged as BENCHMARKS.md's policy flags a
change: ALERT above 5%, and above 10% a regression that fails once a
controlled rerun confirms it on the same benchmark series. It then states,
per binary, how many controlled samples exceed 1.1 times the baseline and
the median's change, and the head/base ratio of the medians.
"""

import hashlib
import json
import statistics
import sys
from pathlib import Path


def verified(path, digests, key):
    """The JSON at ``path``, after checking its SHA-256 against ``digests[key]``."""
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != digests[key]:
        raise SystemExit(f"{key}: sha256 {actual} does not match the record's {digests[key]}")
    return json.loads(data)


def flag(change):
    """BENCHMARKS.md's flag for a relative change (positive = slower)."""
    return "ABOVE-10%" if change > 0.10 else ("ALERT" if change > 0.05 else "ok")


def main():
    """Prints the comparison for both benchmarks."""
    measurements = Path(sys.argv[1]) / "acceptance/measurements"
    manifest = json.loads((measurements / "small-model-performance-20260910/manifest.json").read_text())
    envelope = json.loads((measurements / "half-precision-envelope-20261018.json").read_text())
    files = envelope["files"]
    baselines = {name: entry["sha256"] for name, entry in manifest["records"].items()}
    print("== approved baseline: September 10 controlled summaries (single cold samples); thresholds 5% alert, 10% confirmed failure")
    pairs = [("baseline", "qwen-controlled-current-summary.json", "qwen35-baseline-benchmark.json"),
             ("public", "qwen-public-summary.json", "qwen35-public-causal-greedy-benchmark.json")]
    for kind, summary_name, gate_name in pairs:
        summary_path = measurements / "small-model-performance-20260910" / summary_name
        summary = verified(summary_path, baselines, summary_name)
        base_seconds = summary["benchmark"]["cold_first_token"]["seconds"]
        gate = verified(measurements / "half-precision-envelope-20261018/gates/bench" / gate_name, files,
                        f"gates/bench/{gate_name}")
        gate_seconds = gate["cold_first_token"]["seconds"]
        print(f"== {kind} benchmark: approved cold first token {base_seconds:.6f} s; 1.05x {1.05 * base_seconds:.6f} s, "
              f"1.10x {1.10 * base_seconds:.6f} s")
        change = gate_seconds / base_seconds - 1
        print(f"   gate run (f9f78b5): {gate_seconds:.6f} s ({change:+.1%}) {flag(change)}")
        samples = {}
        for side, source in (("base", "7bd057e"), ("head", "f9f78b5")):
            values = []
            for number in (1, 2, 3):
                name = f"gates/control/bench/r{number}.{side}.{kind}.json"
                report = verified(measurements / "half-precision-envelope-20261018" / name, files, name)
                values.append(report["cold_first_token"]["seconds"])
            samples[side] = values
            above = sum(value > 1.10 * base_seconds for value in values)
            median = statistics.median(values)
            described = ", ".join(f"{value:.4f} ({value / base_seconds - 1:+.1%})" for value in values)
            print(f"   controlled rerun {side} ({source}): {described}; {above} of 3 above 1.10x; "
                  f"median {median:.4f} s ({median / base_seconds - 1:+.1%}) {flag(median / base_seconds - 1)}")
        every = samples["base"] + samples["head"]
        ratio = statistics.median(samples["head"]) / statistics.median(samples["base"])
        print(f"   controlled rerun, both binaries: {sum(v > 1.10 * base_seconds for v in every)} of 6 samples above 1.10x "
              f"(range {min(every):.4f} to {max(every):.4f} s); head/base median ratio {ratio:.3f}")


if __name__ == "__main__":
    main()
