"""Scratch analysis of the half-precision SiLU experiment (not committed).

Usage: python3 -I analyze.py DATA_DIR NAME TRUTH_BUNDLE OFFICIAL_BF16_BUNDLE

DATA_DIR/NAME/burn/<config>/prompt<i>/ hold silu-dump output (configs such as
flex-bf16-base, flex-bf16-patched, metal-f16-base, metal-f16-patched);
DATA_DIR/NAME/torch-block0/<variant>/prompt<i>/ hold torch_block0.py output;
DATA_DIR/NAME/torch-full/<variant>/prompt<i>/logits.bin hold optional full
official logits (official_variants.py). Prints:

1. per prompt and Burn configuration, the gate statistics against the truth
   bundle (max |delta| and its position, mean |delta|, F64 as the support
   module computes them) next to the bounds 1.2 x the official BF16 bundle's
   own statistics, and the distance (max and mean |delta|) to the official
   BF16 logits and to any full official F16 logits;
2. block 0: Burn's SiLU output against torch's F.silu and against
   x * torch.sigmoid(x) applied to Burn's own gate pre-activation (same input,
   CPU and MPS), Burn's pre-activation against torch's, and Burn's residual
   after block 0 against torch's with each SiLU (identical entries, max |delta|,
   at position 0 and over every position).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

SLACK = 1.2


def read_bundle(root):
    """Returns (manifest, [logits as float64 [T, V]])."""
    manifest = json.loads((root / "reference.json").read_text())
    vocabulary = manifest["vocabulary_size"]
    return manifest, [
        np.fromfile(root / p["logits_file"], dtype="<f4").astype(np.float64).reshape(len(p["token_ids"]), vocabulary)
        for p in manifest["prompts"]
    ]


def load(path, shape=None):
    """Reads raw F32 as float32 (reshaped), or None when absent."""
    if not path.exists():
        return None
    value = np.fromfile(path, dtype="<f4")
    return value.reshape(shape) if shape is not None else value


def torch_dtype(config):
    """Maps a config name to its torch half dtype."""
    return torch.bfloat16 if "bf16" in config or "bfloat16" in config else torch.float16


def main():
    data, name, truth_dir, official_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    root = data / name
    manifest, truth = read_bundle(truth_dir)
    _, official = read_bundle(official_dir)
    vocabulary = manifest["vocabulary_size"]
    burn_configs = sorted(p.name for p in (root / "burn").iterdir() if p.is_dir())
    full = sorted(p.name for p in (root / "torch-full").iterdir() if p.is_dir()) if (root / "torch-full").exists() else []
    results = {"name": name, "prompts": []}
    print(f"### {name}: truth {truth_dir.name} ({manifest['provenance'].get('dtype')}), official BF16 {official_dir.name}")
    for index, (t, o) in enumerate(zip(truth, official)):
        tokens = t.shape[0]
        bound_max, bound_mean = SLACK * np.abs(o - t).max(), SLACK * np.abs(o - t).mean()
        print(f"== prompt {index} ({tokens} tokens): bounds max {bound_max:.6f} mean {bound_mean:.6f} "
              f"(official BF16 max {np.abs(o - t).max():.6f} mean {np.abs(o - t).mean():.6f})")
        row = {"prompt": index, "tokens": tokens, "bound_max": bound_max, "bound_mean": bound_mean, "burn": {}}
        refs = {"official-bf16": o}
        for suffix in ("cpu-sdpa", "mps-eager", "mps-sdpa"):
            extra = official_dir.parent / f"{official_dir.name}-{suffix}"
            if (extra / "reference.json").exists():
                refs[f"official-bf16-{suffix}"] = read_bundle(extra)[1][index]
        for variant in full:
            value = load(root / "torch-full" / variant / f"prompt{index}" / "logits.bin", (tokens, vocabulary))
            if value is not None:
                refs[f"torch-{variant}"] = value.astype(np.float64)
        for config in burn_configs:
            directory = root / "burn" / config / f"prompt{index}"
            logits = load(directory / "logits.bin", (tokens, vocabulary))
            if logits is None:
                continue
            plain = load(directory / "logits_plain.bin", (tokens, vocabulary))
            logits = logits.astype(np.float64)
            delta = np.abs(logits - t)
            maximum, mean = float(delta.max()), float(delta.mean())
            position = int(delta.argmax()) // vocabulary
            verdict = ("ok" if maximum <= bound_max else "MAX EXCEEDED") + ", " + (
                "ok" if mean <= bound_mean else "MEAN EXCEEDED")
            distances = {}
            for label, reference in refs.items():
                if np.isfinite(reference).all():
                    gap = np.abs(logits - reference)
                    distances[label] = (float(gap.max()), float(gap.mean()))
            hooked_equals_plain = plain is not None and np.array_equal(plain.astype(np.float64), logits)
            print(f"  {config:22s} max {maximum:.6f} @pos {position} ({maximum / bound_max:.3f} of bound) "
                  f"mean {mean:.6f} ({mean / bound_mean:.3f} of bound) [{verdict}] hooked==plain {hooked_equals_plain}")
            print("  " + " " * 22 + " distance " + "; ".join(
                f"{label} max {d[0]:.6f} mean {d[1]:.6f}" for label, d in distances.items()))
            row["burn"][config] = {"max": maximum, "position": position, "mean": mean,
                                   "max_ratio": maximum / bound_max, "mean_ratio": mean / bound_mean,
                                   "distances": distances, "hooked_equals_plain": hooked_equals_plain}
        results["prompts"].append(row)
    # Block 0.
    print(f"== block 0 ({name})")
    block0 = {}
    torch_root = root / "torch-block0"
    for config in burn_configs:
        if "f32" in config:
            continue
        dtype = torch_dtype(config)
        torch_variants = [v for v in (sorted(p.name for p in torch_root.iterdir() if p.is_dir()) if torch_root.exists() else [])
                          if (("bfloat16" in v) == (dtype == torch.bfloat16))]
        stats = {"silu_vs_torch": {}, "pre_vs_torch": {}, "layer0_vs_torch": {}}
        totals = {}
        for index, t in enumerate(truth):
            tokens = t.shape[0]
            directory = root / "burn" / config / f"prompt{index}"
            pre = load(directory / "mlp0_pre.bin")
            act = load(directory / "mlp0_act.bin")
            layer0 = load(directory / "layer0.bin")
            if pre is None:
                continue
            x = torch.from_numpy(pre.copy()).to(dtype)
            assert torch.equal(x.to(torch.float32), torch.from_numpy(pre.copy())), "pre-activation is not exact in the dtype"
            burn_act = torch.from_numpy(act.copy())
            for device in ("cpu", "mps"):
                xd = x.to(device)
                once = torch.nn.functional.silu(xd).cpu().to(torch.float32)
                composed = (xd * torch.sigmoid(xd)).cpu().to(torch.float32)
                for label, value in ((f"F.silu-{device}", once), (f"x*sigmoid(x)-{device}", composed)):
                    key = label
                    agg = totals.setdefault(("silu", key), [0, 0, 0.0])
                    differ = int((value != burn_act).sum())
                    agg[0] += differ
                    agg[1] += value.numel()
                    agg[2] = max(agg[2], float((value - burn_act).abs().max()))
            hidden = layer0.size // tokens
            for variant in torch_variants:
                tdir = torch_root / variant / f"prompt{index}"
                tpre = load(tdir / "mlp0_pre.bin")
                tlayer = load(tdir / "layer0.bin")
                if tpre is not None:
                    width = pre.size // tokens
                    for scope, (bb, rr) in (("pos0", (pre[:width], tpre[:width])), ("all", (pre, tpre))):
                        agg = totals.setdefault((f"pre-{scope}", variant), [0, 0, 0.0])
                        agg[0] += int((bb != rr).sum())
                        agg[1] += bb.size
                        agg[2] = max(agg[2], float(np.abs(rr.astype(np.float64) - bb).max()))
                if tlayer is not None:
                    b = layer0.reshape(tokens, hidden).astype(np.float64)
                    r = tlayer.reshape(tokens, hidden).astype(np.float64)
                    for scope, (bb, rr) in (("pos0", (b[:1], r[:1])), ("all", (b, r))):
                        agg = totals.setdefault((f"layer0-{scope}", variant), [0, 0, 0.0])
                        agg[0] += int((bb != rr).sum())
                        agg[1] += bb.size
                        agg[2] = max(agg[2], float(np.abs(bb - rr).max()))
        print(f"  {config}:")
        for (kind, label), (differ, count, worst) in sorted(totals.items()):
            print(f"    {kind:12s} vs {label:34s} differing {differ:8d} of {count:9d} ({100 * differ / count:7.3f}%) "
                  f"max |delta| {worst:.6g}")
            block0.setdefault(config, {})[f"{kind} vs {label}"] = {"differing": differ, "count": count, "max": worst}
    results["block0"] = block0
    (root / "analysis.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
