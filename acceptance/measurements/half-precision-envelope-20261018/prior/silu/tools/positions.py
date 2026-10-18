"""Scratch: splits each Burn dump's logit distance to a reference into position 0 and positions >= 1 (not committed).

Usage: python3 -I positions.py DATA_DIR NAME TRUTH_BUNDLE OFFICIAL_BF16_BUNDLE
For flex-bf16 the reference is the official BF16 bundle; for metal-f16 it is the
official CPU F16 eager logits under DATA_DIR/NAME/torch-full (when present).
Prints the mean |delta| over all prompts' position-0 rows and over all later rows.
"""
import json, sys
from pathlib import Path
import numpy as np


def bundle(root):
    """Returns the bundle's per-prompt logits as float64 [T, V]."""
    m = json.loads((Path(root) / "reference.json").read_text()); v = m["vocabulary_size"]
    return [np.fromfile(Path(root) / p["logits_file"], dtype="<f4").astype(np.float64).reshape(len(p["token_ids"]), v)
            for p in m["prompts"]]


data, name = Path(sys.argv[1]), sys.argv[2]
truth, official = bundle(sys.argv[3]), bundle(sys.argv[4])
for profile in ("flex-bf16", "metal-f16"):
    if profile == "flex-bf16":
        refs, label = official, "official-bf16"
    else:
        root = data / name / "torch-full" / "cpu-float16-eager"
        if not root.exists():
            continue
        refs = [np.fromfile(root / f"prompt{i}" / "logits.bin", "<f4").astype(np.float64).reshape(t.shape) for i, t in enumerate(truth)]
        if not all(np.isfinite(r).all() for r in refs):
            continue
        label = "official-cpu-f16"
    for source in ("base", "patched"):
        first, later = [], []
        for i, ref in enumerate(refs):
            b = np.fromfile(data / name / "burn" / f"{profile}-{source}" / f"prompt{i}" / "logits.bin", "<f4").astype(np.float64).reshape(ref.shape)
            g = np.abs(b - ref)
            first.append(g[0]); later.append(g[1:].ravel())
        first, later = np.concatenate(first), np.concatenate(later)
        print(f"{name:14s} {profile} {source:8s} to {label}: pos0 mean {first.mean():.5f} max {first.max():.4f} | pos>=1 mean {later.mean():.5f} max {later.max():.4f}")
