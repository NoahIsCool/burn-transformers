"""Scratch diagnostic: compares Gemma Burn dumps with official-variant dumps and the truth bundle (not committed).

Usage: python3 -I gemma_compare.py TRUTH_BUNDLE OFFICIAL_BF16_BUNDLE DATA_DIR [POSITION...]

Reads DATA_DIR/burn/<config>/prompt<i>/ (burn-dump-gemma) and DATA_DIR/official*/<variant>/prompt<i>/
(gemma_official_variants.py), raw F32. Prints, per prompt and source, the logits' max |delta|
(position) and mean |delta| from the truth with the rule's bounds (1.2 x the official BF16
bundle's statistics) and whether they hold; each Burn configuration's distance from every other
source; at each POSITION (default 0 and 1) the residual stream's max |delta| from
torch:cpu-float32-eager after the embedding, every block, and the final norm, plus the fraction of
residual entries identical to torch:cpu-bfloat16-eager; and, for every source with non-finite
values, the first non-finite point and the largest finite |value| per point.
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


def points(directory):
    """Returns the captured point names in stream order."""
    layers = sorted(int(p.stem[5:]) for p in directory.glob("layer*.bin"))
    return ["embedding"] + [f"layer{i}" for i in layers] + ["final_norm"]


def main():
    truth_dir, official_dir, data = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    positions = [int(p) for p in sys.argv[4:]] or [0, 1]
    manifest = json.loads((truth_dir / "reference.json").read_text())
    official_manifest = json.loads((official_dir / "reference.json").read_text())
    vocabulary = manifest["vocabulary_size"]
    sources = {}
    for root in [data / "burn", data / "official", data / "official-f16"]:
        if not root.is_dir():
            continue
        for config in sorted(p for p in root.iterdir() if p.is_dir()):
            label = ("burn:" if root.name == "burn" else "torch:") + config.name
            sources[label] = config
    for index, prompt in enumerate(manifest["prompts"]):
        tokens = len(prompt["token_ids"])
        truth = np.fromfile(truth_dir / prompt["logits_file"], dtype="<f4").astype(np.float64).reshape(tokens, vocabulary)
        official = np.fromfile(official_dir / official_manifest["prompts"][index]["logits_file"],
                               dtype="<f4").astype(np.float64).reshape(tokens, vocabulary)
        official_delta = np.abs(official - truth)
        bound_max, bound_mean = 1.2 * official_delta.max(), 1.2 * official_delta.mean()
        print(f"== prompt {index} ({tokens} tokens): logits against the truth; bounds max {bound_max:.6f} "
              f"mean {bound_mean:.6f}; official BF16 per-position max "
              + " ".join(f"{v:.3f}" for v in official_delta.max(axis=1)))
        logits = {}
        for label, config in sources.items():
            value = load(config / f"prompt{index}", "logits", (tokens, vocabulary))
            if value is None:
                continue
            logits[label] = value
            if not np.isfinite(value).all():
                print(f"  {label:42s} NON-FINITE logits ({int((~np.isfinite(value)).sum())} entries)")
                continue
            delta = np.abs(value - truth)
            where = int(delta.argmax()) // vocabulary
            verdict = "ok" if delta.max() <= bound_max and delta.mean() <= bound_mean else "OUTSIDE"
            print(f"  {label:42s} max {delta.max():.6f} (pos {where}) mean {delta.mean():.6f} {verdict}; "
                  f"per-position max " + " ".join(f"{v:.3f}" for v in delta.max(axis=1)))
        for burn in [label for label in logits if label.startswith("burn:")]:
            if not np.isfinite(logits[burn]).all():
                continue
            distances = sorted((np.abs(logits[burn] - value).max(), label) for label, value in logits.items()
                               if label != burn and np.isfinite(value).all())
            print(f"  distance of {burn}: " + "; ".join(f"{label} {d:.4f}" for d, label in distances[:8]))
        reference = sources.get("torch:cpu-float32-eager")
        bf16 = sources.get("torch:cpu-bfloat16-eager")
        if reference is None:
            continue
        names = points(reference / f"prompt{index}")
        for position in positions:
            if position >= tokens:
                continue
            print(f"== prompt {index}: residual at position {position}, max |delta| from torch:cpu-float32-eager "
                  f"(and fraction identical to torch:cpu-bfloat16-eager)")
            picks = [0, 1, 2, 3, 4, 6, 9, 13, 17, 21, 25, 29, 33, 34, 35]
            print("  " + " " * 42 + " ".join(f"{names[i]:>13s}" for i in picks if i < len(names)))
            for label, config in sources.items():
                directory = config / f"prompt{index}"
                cells = []
                for i in picks:
                    if i >= len(names):
                        continue
                    actual = load(directory, names[i], (tokens, -1))
                    expected = load(reference / f"prompt{index}", names[i], (tokens, -1))
                    if actual is None:
                        cells.append(f"{'-':>13s}")
                        continue
                    gap = np.abs(actual[position] - expected[position]).max()
                    same = ""
                    if bf16 is not None:
                        twin = load(bf16 / f"prompt{index}", names[i], (tokens, -1))
                        same = f"/{np.mean(actual[position] == twin[position]):.2f}"
                    cells.append(f"{gap:8.3f}{same:>5s}")
                print(f"  {label:42s}" + " ".join(cells))
        for label, config in sources.items():
            directory = config / f"prompt{index}"
            first = None
            magnitudes = []
            for name in names:
                value = load(directory, name, (tokens, -1))
                if value is None:
                    break
                finite = np.isfinite(value)
                magnitudes.append(np.abs(value[finite]).max() if finite.any() else float("nan"))
                if first is None and not finite.all():
                    bad = np.argwhere(~finite)
                    first = (name, int(bad[0][0]), int((~finite).sum()))
            if first is not None or label.startswith(("burn:metal-f16", "burn:flex-f16", "torch:cpu-float16",
                                                      "torch:mps-float16")):
                print(f"  {label}: first non-finite {first}; largest finite |value| per point: "
                      + " ".join(f"{names[i]}={m:.4g}" for i, m in enumerate(magnitudes)))


if __name__ == "__main__":
    main()
