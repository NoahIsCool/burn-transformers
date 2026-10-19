"""Scratch (rerun at 8181605): the statistic nearest its bound in each rerun log (not committed).

Usage: python3 -I margins.py > logs/margins.log

Reads every `<stat> |Δ| = A ... (bound B)` (or `tolerance B`, `provisional
tolerance B`) pair of each run's test output and prints the largest ratio A/B
per run, kind of bound, and dtype, then the largest ratios overall. Ratios are
read from the logs only; nothing is recomputed.
"""

import json
import os
import re

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(H)
PAIR = re.compile(r"(max|mean) \|Δ\| = ([-+\d.eE]+|NaN|inf)([^()]*)\(((?:provisional )?(?:bound|tolerance)) ([-+\d.eE]+)\)")


def main():
    """Prints per-run and overall nearest-to-bound statistics."""
    runs = [json.loads(line) for line in open(os.path.join(ROOT, "logs", "runs.jsonl"))]
    every = []
    for run in runs:
        best = {}
        inside = False
        for line in open(run["log"], errors="replace"):
            if line.startswith("running "):
                inside = True
            if not inside:
                continue
            if line.startswith("test result"):
                break
            for stat, value, _where, kind, bound in PAIR.findall(line):
                try:
                    ratio = float(value) / float(bound)
                except (ValueError, ZeroDivisionError):
                    ratio = float("inf")
                key = (stat, kind)
                text = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
                if key not in best or ratio > best[key][0]:
                    best[key] = (ratio, text)
                every.append((ratio, run["name"], stat, kind, text))
        print(f"== {run['name']} ({run['result']}, exit {run['exit']})")
        for (stat, kind), (ratio, text) in sorted(best.items()):
            print(f"   {stat:4s} vs {kind:22s} {ratio:8.4f}  {text[:260]}")
    print("== nearest 15 statistics to their bound, over all runs")
    for ratio, name, stat, kind, text in sorted(every, reverse=True)[:15]:
        print(f"   {ratio:8.4f} {name} [{stat} vs {kind}] {text[:240]}")


if __name__ == "__main__":
    main()
