"""Scratch helper: first non-finite residual block of an official-variant dump, per prompt."""
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
for prompt in sorted(root.glob("prompt*")):
    names = ["embedding"] + sorted((p.stem for p in prompt.glob("layer*.bin")), key=lambda s: int(s[5:])) + ["final_norm"]
    first = None
    for name in names:
        values = np.fromfile(prompt / f"{name}.bin", dtype="<f4")
        if not np.isfinite(values).all():
            first = name
            break
        peak = np.abs(values).max()
    print(f"{root.name}/{prompt.name}: first non-finite point {first}; largest finite |x| before it {peak:.1f}")
