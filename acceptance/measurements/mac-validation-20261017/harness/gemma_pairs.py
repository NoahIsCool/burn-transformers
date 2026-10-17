"""Scratch diagnostic: pairwise identity of two dumps at one position, point by point (not committed).

Usage: python3 -I gemma_pairs.py DATA_DIR PROMPT POSITION A B [A B ...]
where A and B are source directories relative to DATA_DIR (e.g. burn/flex-bf16-fused official/cpu-bfloat16-eager-burngelu).
Prints, per captured point (embedding, layer i, final_norm, logits), max |A - B| and the fraction of identical values.
"""
import sys
from pathlib import Path
import numpy as np

data, prompt, position = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
pairs = list(zip(sys.argv[4::2], sys.argv[5::2]))
for a, b in pairs:
    da, db = data / a / f"prompt{prompt}", data / b / f"prompt{prompt}"
    layers = sorted(int(p.stem[5:]) for p in db.glob("layer*.bin"))
    names = ["embedding"] + [f"layer{i}" for i in layers] + ["final_norm", "logits"]
    cells = []
    first_diff = None
    for name in names:
        x = np.fromfile(da / f"{name}.bin", dtype="<f4")
        y = np.fromfile(db / f"{name}.bin", dtype="<f4")
        width = x.size // (y.size // x.size * 0 + 1)
        tokens = len(np.fromfile(db / "embedding.bin", dtype="<f4")) // 2560
        x = x.reshape(tokens, -1)[position]; y = y.reshape(tokens, -1)[position]
        same = float(np.mean(x == y)); gap = float(np.abs(x.astype(np.float64) - y).max())
        if first_diff is None and same < 1.0:
            first_diff = name
        cells.append(f"{name}:{gap:.3g}/{same:.3f}")
    print(f"prompt {prompt} position {position}: {a} vs {b}: first differing point {first_diff}")
    print("   " + " ".join(cells))
