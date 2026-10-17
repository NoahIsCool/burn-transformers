"""Scratch diagnostic: compares Burn dumps with official-variant dumps and the truth bundle (not committed).

Usage: python3 -I compare_dumps.py TRUTH_BUNDLE DATA_DIR NAME [POSITION]

Reads DATA_DIR/NAME-burn/<config>/prompt<i>/ and DATA_DIR/NAME-official*/<variant>/prompt<i>/
(raw F32 written by burn-dump and official_variants.py). Prints, per prompt and
source, the logits' max |delta| (position) and mean |delta| from the truth bundle,
and the distance of each Burn configuration from each official variant; then, at
POSITION (default 0) of every prompt, the residual stream's max |delta| from the
official F32 pass after every block for every source.
"""

import json
import sys
from pathlib import Path

import numpy as np


def load(directory, name, shape):
    """Reads one raw F32 file as float64, or None."""
    path = directory / f"{name}.bin"
    if not path.exists():
        return None
    return np.fromfile(path, dtype="<f4").astype(np.float64).reshape(shape)


def main():
    truth_dir, data, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    position = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    manifest = json.loads((truth_dir / "reference.json").read_text())
    vocabulary = manifest["vocabulary_size"]
    sources = {}
    for root in sorted(data.glob(f"{name}-burn")) + sorted(data.glob(f"{name}-official*")):
        for config in sorted(p for p in root.iterdir() if p.is_dir()):
            label = ("burn:" if root.name.endswith("-burn") else "torch:") + config.name
            sources[label] = config
    for index, prompt in enumerate(manifest["prompts"]):
        tokens = len(prompt["token_ids"])
        truth = np.fromfile(truth_dir / prompt["logits_file"], dtype="<f4").astype(np.float64).reshape(tokens, vocabulary)
        print(f"== prompt {index} ({tokens} tokens): logits against the truth bundle")
        logits = {}
        for label, config in sources.items():
            value = load(config / f"prompt{index}", "logits", (tokens, vocabulary))
            if value is None:
                continue
            logits[label] = value
            delta = np.abs(value - truth)
            where = int(delta.argmax()) // vocabulary
            per_position = delta.max(axis=1)
            print(f"  {label:45s} max {delta.max():.6f} (pos {where}) mean {delta.mean():.6f} "
                  f"max excluding pos 0 {per_position[1:].max() if tokens > 1 else 0:.6f}")
        burns = [label for label in logits if label.startswith("burn:")]
        for burn in burns:
            distances = sorted((np.abs(logits[burn] - value).max(), label) for label, value in logits.items()
                               if label != burn)
            print(f"  distance of {burn}: " + "; ".join(f"{label} {d:.4f}" for d, label in distances[:6]))
        print(f"== prompt {index}: residual at position {position}, max |delta| from torch:cpu-float32-eager per block")
        reference = sources.get("torch:cpu-float32-eager")
        layers = sorted(int(p.stem[5:]) for p in (reference / f"prompt{index}").glob("layer*.bin"))
        rows = {}
        for label, config in sources.items():
            directory = config / f"prompt{index}"
            row = []
            for layer in layers:
                actual = load(directory, f"layer{layer}", (tokens, -1))
                expected = load(reference / f"prompt{index}", f"layer{layer}", (tokens, -1))
                if actual is None:
                    break
                row.append(np.abs(actual[position] - expected[position]).max())
            final = load(directory, "final_norm", (tokens, -1))
            if final is not None and row:
                expected = load(reference / f"prompt{index}", "final_norm", (tokens, -1))
                row.append(np.abs(final[position] - expected[position]).max())
                rows[label] = row
        picks = list(range(0, len(layers), max(1, len(layers) // 10))) + [len(layers) - 1, len(layers)]
        print("  " + " " * 45 + " ".join(f"{('L' + str(i)) if i < len(layers) else 'norm':>8s}" for i in picks))
        for label, row in rows.items():
            print(f"  {label:45s}" + " ".join(f"{row[i]:8.3f}" for i in picks))


if __name__ == "__main__":
    main()
