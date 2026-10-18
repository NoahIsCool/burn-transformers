"""Scratch: summarizes silu-bench JSON lines across alternated rounds (not committed).

Usage: python3 -I summarize_perf.py LOG_DIR PREFIX
Groups lines by (kind, backend, dtype, shape) and source label (from the log name),
and prints each round's median, the median over rounds, and the patched/base ratio.
"""
import json, re, statistics, sys
from collections import defaultdict
from pathlib import Path

rows = defaultdict(lambda: defaultdict(list))
for path in sorted(Path(sys.argv[1]).glob(sys.argv[2] + "*.log")):
    label = path.name.split(".")[-2]
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            if d["kind"] == "act":
                key = (d["backend"], d["dtype"], f"[1,{d['tokens']},{d['intermediate']}]", "act_us")
                rows[key][label].append(d["median_us"])
            else:
                for metric in ("prefill_median_s", "decode_median_s", "prefill_tok_per_s", "decode_tok_per_s"):
                    rows[(d["backend"], d["dtype"], f"source{d['source']}", metric)][label].append(d[metric])
for key, by in rows.items():
    base, patched = by.get("base", []), by.get("patched", [])
    if not base or not patched:
        continue
    mb, mp = statistics.median(base), statistics.median(patched)
    print(f"{' '.join(key):60s} base {[round(v, 3) for v in base]} median {mb:.3f} | patched {[round(v, 3) for v in patched]} median {mp:.3f} | ratio {mp / mb:.3f}")
