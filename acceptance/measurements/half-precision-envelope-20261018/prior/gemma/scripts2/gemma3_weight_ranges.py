"""Scratch diagnostic (not committed): Gemma 3 language-model weight ranges against F16.

Usage: python3 -I gemma3_weight_ranges.py MODEL_DIR OUT_JSON

Reads every language-model tensor of a local SafeTensors checkpoint one at a time (no model is
built, nothing is downloaded) and reports, per tensor, the largest |w|, how many entries exceed
65504 (the F16 maximum; they become inf when cast), and how many nonzero entries fall below
2**-24 (the smallest F16 subnormal; they flush to zero). For the RMSNorm weights it also reports
the largest effective gain |1 + w|, because Gemma's zero-centered norm multiplies by (1 + w).
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

F16_MAX = 65504.0
F16_MIN_SUBNORMAL = 2.0 ** -24


def main():
    model_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    by_file = {}
    for name, file in index["weight_map"].items():
        if "vision_tower" in name or "multi_modal_projector" in name:
            continue
        by_file.setdefault(file, []).append(name)
    rows = []
    for file, names in sorted(by_file.items()):
        with safe_open(model_dir / file, framework="pt") as handle:
            for name in sorted(names):
                w = handle.get_tensor(name).to(torch.float32)
                a = w.abs()
                row = {
                    "name": name,
                    "dtype_stored": str(handle.get_slice(name).get_dtype()),
                    "shape": list(w.shape),
                    "max_abs": float(a.max()),
                    "over_f16_max": int((a > F16_MAX).sum()),
                    "nonzero_below_f16_subnormal": int(((a > 0) & (a < F16_MIN_SUBNORMAL)).sum()),
                }
                if name.endswith("norm.weight"):
                    gain = (1.0 + w).abs()
                    row["max_gain"] = float(gain.max())
                    row["argmax_gain"] = int(gain.argmax())
                rows.append(row)
    worst = max(rows, key=lambda r: r["max_abs"])
    norms = [r for r in rows if "max_gain" in r]
    top_gains = sorted(norms, key=lambda r: -r["max_gain"])[:8]
    summary = {
        "model_dir": str(model_dir),
        "tensors": len(rows),
        "largest_abs_weight": {k: worst[k] for k in ("name", "max_abs")},
        "tensors_with_entries_over_f16_max": [r["name"] for r in rows if r["over_f16_max"]],
        "nonzero_entries_below_f16_subnormal": sum(r["nonzero_below_f16_subnormal"] for r in rows),
        "top_norm_gains": [{k: r[k] for k in ("name", "max_gain", "argmax_gain")} for r in top_gains],
        "rows": rows,
    }
    out_json.write_text(json.dumps(summary))
    print(f"{model_dir}: {len(rows)} language-model tensors; largest |w| {worst['max_abs']:.6g} ({worst['name']}); "
          f"tensors with |w| > 65504: {len(summary['tensors_with_entries_over_f16_max'])}; "
          f"nonzero |w| < 2^-24: {summary['nonzero_entries_below_f16_subnormal']}")
    for r in top_gains:
        print(f"   gain |1+w| {r['max_gain']:.6g} at channel {r['argmax_gain']}: {r['name']}")


if __name__ == "__main__":
    main()
