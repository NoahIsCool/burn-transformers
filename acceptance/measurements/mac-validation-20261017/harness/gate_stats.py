"""Scratch diagnostic: judges prompt-logit dumps by the half-precision rule (not committed).

Usage: python3 -I gate_stats.py TRUTH_BUNDLE OFFICIAL_BF16_BUNDLE DIR...

Each DIR holds prompt<i>/logits.bin (raw F32 [tokens, V]) from burn-dump or
official_variants.py. Prints, per prompt, max |delta| (position) and mean |delta|
from the truth bundle next to the bounds 1.2 x the official BF16 bundle's own max and
mean (computed in float64 here, as the support module does), and PASS/FAIL.
"""

import json
import sys
from pathlib import Path

import numpy as np

SLACK = 1.2


def read(root):
    """Returns (manifest, [logits as float64 [T, V]])."""
    manifest = json.loads((root / "reference.json").read_text())
    vocabulary = manifest["vocabulary_size"]
    return manifest, [
        np.fromfile(root / p["logits_file"], dtype="<f4").astype(np.float64).reshape(len(p["token_ids"]), vocabulary)
        for p in manifest["prompts"]
    ]


def main():
    truth_dir, official_dir = Path(sys.argv[1]), Path(sys.argv[2])
    _, truth = read(truth_dir)
    _, official = read(official_dir)
    for directory in map(Path, sys.argv[3:]):
        verdicts = []
        for index, (t, o) in enumerate(zip(truth, official)):
            bound_max = SLACK * np.abs(o - t).max()
            bound_mean = SLACK * np.abs(o - t).mean()
            path = directory / f"prompt{index}" / "logits.bin"
            if not path.exists():
                continue
            actual = np.fromfile(path, dtype="<f4").astype(np.float64).reshape(t.shape)
            delta = np.abs(actual - t)
            ok_max, ok_mean = delta.max() <= bound_max, delta.mean() <= bound_mean
            verdicts.append(ok_max and ok_mean)
            print(f"{directory.name} prompt {index}: max {delta.max():.4f} at pos {int(delta.argmax()) // t.shape[1]} "
                  f"(bound {bound_max:.4f}{'' if ok_max else ' EXCEEDED'}), mean {delta.mean():.5f} "
                  f"(bound {bound_mean:.5f}{'' if ok_mean else ' EXCEEDED'})")
        print(f"{directory.name}: {'PASS' if all(verdicts) else 'FAIL'} ({sum(verdicts)} of {len(verdicts)} prompts within both bounds)")


if __name__ == "__main__":
    main()
