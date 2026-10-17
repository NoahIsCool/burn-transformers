"""Scratch diagnostic: Burn cached-decode versus stateless gaps per step (not committed).

Usage: python3 -I compare_decode.py BUNDLE DATA_DIR NAME

For every DATA_DIR/NAME-burn-decode/<config>/ written by burn_decode_dumps.sh, prints
each decode step's max |delta| between decode_logits and the stateless logits at the
same position, and both passes' per-step max |delta| from the Burn Flex F32 stateless
pass (config flex-f32-fused), which stands in for the F32 truth on the greedy tokens.
"""

import json
import sys
from pathlib import Path

import numpy as np


def main():
    bundle, data, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    manifest = json.loads((bundle / "reference.json").read_text())
    vocabulary = manifest["vocabulary_size"]
    first = len(manifest["prompts"][0]["token_ids"])
    length = first + len(manifest["greedy"]["steps"])
    root = data / f"{name}-burn-decode"
    reference = None
    f32 = root / "flex-f32-fused"
    if f32.exists():
        reference = np.fromfile(f32 / "logits_plain.bin", dtype="<f4").astype(np.float64).reshape(length, vocabulary)[first:]
    for config in sorted(p for p in root.iterdir() if p.is_dir()):
        stateless = np.fromfile(config / "logits_plain.bin", dtype="<f4").astype(np.float64).reshape(length, vocabulary)[first:]
        decoded = np.fromfile(config / "decode_logits.bin", dtype="<f4").astype(np.float64).reshape(-1, vocabulary)
        gaps = np.abs(decoded - stateless).max(axis=1)
        worst = int(gaps.argmax())
        print(f"{name} burn:{config.name}: {len(gaps)} decode steps from position {first}: decode vs stateless max |d| = "
              f"{gaps.max():.6e} at step {worst}; steps above 0.5: {int((gaps > 0.5).sum())}; median {np.median(gaps):.4e}")
        print(f"  per-step decode-vs-stateless " + " ".join(f"{g:.4f}" for g in gaps))
        if reference is not None:
            print(f"  per-step max |d| to Flex F32, decode " + " ".join(f"{g:.4f}" for g in np.abs(decoded - reference).max(axis=1)))
            print(f"  per-step max |d| to Flex F32, stateless " + " ".join(f"{g:.4f}" for g in np.abs(stateless - reference).max(axis=1)))


if __name__ == "__main__":
    main()
