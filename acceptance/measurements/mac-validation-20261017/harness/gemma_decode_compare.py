"""Scratch diagnostic: Burn cached decode versus the same run's stateless pass, per step (not committed).

Usage: python3 -I gemma_decode_compare.py DATA_DIR PREFILL CONFIG...
Reads DATA_DIR/burn-decode/<config>/{logits_plain,decode_logits}.bin and the official F32 stateless pass
DATA_DIR/official-decode/cpu-float32-eager/stateless.bin (rows from PREFILL). Prints per-step max |delta|
cached vs stateless, the count above 0.5, and each pass's per-step max |delta| from the official F32 pass.
"""
import sys
from pathlib import Path
import numpy as np

data, prefill = Path(sys.argv[1]), int(sys.argv[2])
V = 262208
truth = np.fromfile(data / "official-decode/cpu-float32-eager/stateless.bin", dtype="<f4").astype(np.float64).reshape(-1, V)
for config in sys.argv[3:]:
    d = data / "burn-decode" / config
    plain = np.fromfile(d / "logits_plain.bin", dtype="<f4").astype(np.float64).reshape(-1, V)[prefill:]
    cached = np.fromfile(d / "decode_logits.bin", dtype="<f4").astype(np.float64).reshape(-1, V)
    gaps = np.abs(cached - plain).max(axis=1)
    print(f"{config}: {len(gaps)} decode steps from position {prefill}: cached vs stateless max |d| = {gaps.max():.6e} "
          f"at step {int(gaps.argmax())}; steps above 0.5: {int((gaps > 0.5).sum())}; median {np.median(gaps):.4e}")
    print(f"{config}: per-step cached-vs-stateless " + " ".join(f"{g:.4f}" for g in gaps))
    print(f"{config}: per-step max |d| to official F32, cached " + " ".join(f"{g:.4f}" for g in np.abs(cached - truth).max(axis=1)))
    print(f"{config}: per-step max |d| to official F32, stateless " + " ".join(f"{g:.4f}" for g in np.abs(plain - truth).max(axis=1)))
