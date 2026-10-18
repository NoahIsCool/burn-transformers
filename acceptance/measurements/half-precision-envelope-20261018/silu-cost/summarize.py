"""Scratch (half2 S9): summarizes this round's SiLU end-to-end cost series (not committed).

Usage: python3 -I summarize.py LOG_DIR

The SiLU round's summarize_perf.py, reading this round's log names
(`<series>.r<round>.<base|patched>.half2.log`): groups the silu-bench JSON
lines by (backend, dtype, source length, metric) and binary, and prints each
round's value, the median over rounds, and the patched/base ratio of the
medians, followed by every run's exit line.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

rows = defaultdict(lambda: defaultdict(list))
exits = []
for path in sorted(Path(sys.argv[1]).glob("*.half2.log")):
    label = path.name.split(".")[-3]
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            for metric in ("prefill_median_s", "decode_median_s", "prefill_tok_per_s", "decode_tok_per_s"):
                rows[(d["backend"], d["dtype"], f"source{d['source']}", metric)][label].append(d[metric])
        elif line.startswith("== exit="):
            exits.append(f"{path.name}: {line}")
for key, by in rows.items():
    base, patched = by.get("base", []), by.get("patched", [])
    if not base or not patched:
        continue
    mb, mp = statistics.median(base), statistics.median(patched)
    print(f"{' '.join(key):45s} base {[round(v, 4) for v in base]} median {mb:.4f} | "
          f"patched {[round(v, 4) for v in patched]} median {mp:.4f} | ratio {mp / mb:.3f}")
for line in exits:
    print(line)
