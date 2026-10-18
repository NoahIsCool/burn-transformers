"""Scratch: per-block residual agreement at one position between Burn and official dumps (not committed).

Usage: python3 -I blocks.py HIDDEN POSITION LABEL=DIR ... -- LABEL=DIR ...

Every DIR holds prompt<i>/layer<l>.bin (raw F32 [tokens, hidden]). For each pair of
a source before "--" and a source after it, prints the share of identical entries
at POSITION after every block (prompts pooled) and the max |delta| there, then the
first block at which agreement drops below 99% and below 90%.
"""

import sys
from pathlib import Path

import numpy as np


def parse(items):
    """Splits LABEL=DIR arguments."""
    return [(item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in items]


def main():
    hidden, position = int(sys.argv[1]), int(sys.argv[2])
    args = sys.argv[3:]
    split = args.index("--")
    left, right = parse(args[:split]), parse(args[split + 1:])
    prompts = sorted(p.name for p in left[0][1].iterdir() if p.name.startswith("prompt"))
    layers = sorted(int(p.stem[5:]) for p in (left[0][1] / prompts[0]).glob("layer*.bin"))
    for a_label, a in left:
        for b_label, b in right:
            shares, worst = [], []
            for layer in layers:
                same = total = 0
                gap = 0.0
                for prompt in prompts:
                    x = np.fromfile(a / prompt / f"layer{layer}.bin", dtype="<f4").reshape(-1, hidden)[position]
                    y = np.fromfile(b / prompt / f"layer{layer}.bin", dtype="<f4").reshape(-1, hidden)[position]
                    same += int((x == y).sum())
                    total += x.size
                    gap = max(gap, float(np.abs(x.astype(np.float64) - y).max()))
                shares.append(same / total)
                worst.append(gap)
            below99 = next((l for l, s in zip(layers, shares) if s < 0.99), None)
            below90 = next((l for l, s in zip(layers, shares) if s < 0.90), None)
            line = " ".join(f"L{l}:{100 * s:.1f}%" for l, s in zip(layers, shares))
            print(f"{a_label} vs {b_label} (position {position}): first block <99% {below99}, <90% {below90}")
            print(f"    identical {line}")
            print("    max|d| " + " ".join(f"L{l}:{g:.3g}" for l, g in zip(layers, worst)))


if __name__ == "__main__":
    main()
