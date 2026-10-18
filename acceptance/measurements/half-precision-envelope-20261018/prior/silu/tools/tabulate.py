"""Scratch: condenses every checkpoint's analysis.json into the SiLU experiment's comparison tables (not committed).

Usage: python3 -I tabulate.py DATA_DIR NAME...

Prints, per checkpoint and Burn profile, for base and patched: each prompt's
max and mean |delta| to the truth bundle as a fraction of the 1.2x bound, the
distance (max, mean) to the official BF16 bundle and, for Metal/F16, to the
official CPU F16 eager logits; then the token-weighted mean distance over all
prompts; and block 0's SiLU and position-0 residual agreement with PyTorch.
"""

import json
import sys
from pathlib import Path


def main():
    data = Path(sys.argv[1])
    for name in sys.argv[2:]:
        result = json.loads((data / name / "analysis.json").read_text())
        print(f"### {name}")
        for profile, official_key in (("flex-bf16", "official-bf16"), ("metal-f16", "torch-cpu-float16-eager")):
            print(f"  {profile} (distance reference: {official_key})")
            pooled = {}
            for row in result["prompts"]:
                cells = []
                for label in ("base", "patched"):
                    burn = row["burn"].get(f"{profile}-{label}")
                    if burn is None:
                        continue
                    distance = burn["distances"].get(official_key)
                    bf16 = burn["distances"].get("official-bf16")
                    text = (f"{label}: truth max {burn['max']:.4f} ({burn['max_ratio']:.3f}) mean {burn['mean']:.5f} "
                            f"({burn['mean_ratio']:.3f})")
                    if distance:
                        text += f" | to {official_key} max {distance[0]:.4f} mean {distance[1]:.5f}"
                        acc = pooled.setdefault(label, [0.0, 0])
                        acc[0] += distance[1] * row["tokens"]
                        acc[1] += row["tokens"]
                    if profile == "metal-f16" and bf16:
                        text += f" | to official-bf16 mean {bf16[1]:.5f}"
                    cells.append(text)
                print(f"    p{row['prompt']} ({row['tokens']} tok, bound max {row['bound_max']:.4f} mean {row['bound_mean']:.5f})")
                for cell in cells:
                    print(f"        {cell}")
            if pooled:
                text = "; ".join(f"{label} {acc[0] / acc[1]:.5f}" for label, acc in pooled.items())
                print(f"    token-weighted mean distance to {official_key}: {text}")
        block0 = result.get("block0", {})
        for profile, dtype in (("flex-bf16", "bfloat16"), ("metal-f16", "float16")):
            for label in ("base", "patched"):
                entry = block0.get(f"{profile}-{label}", {})
                silu = entry.get("silu vs F.silu-cpu")
                composed = entry.get("silu vs x*sigmoid(x)-cpu")
                pos0 = entry.get(f"layer0-pos0 vs cpu-{dtype}-eager")
                pos0_mps = entry.get(f"layer0-pos0 vs mps-{dtype}-eager")
                pre0 = entry.get(f"pre-pos0 vs cpu-{dtype}-eager")
                if not silu:
                    continue
                pct = lambda e: f"{100 * e['differing'] / e['count']:.3f}% of {e['count']}"
                print(f"  block0 {profile}-{label}: SiLU(Burn pre-act) != F.silu-cpu {pct(silu)}; != x*sigmoid(x) {pct(composed)}; "
                      f"gate pre-act pos0 != torch {pct(pre0)}; residual pos0 != official cpu {pct(pos0)}, mps {pct(pos0_mps)}")


if __name__ == "__main__":
    main()
