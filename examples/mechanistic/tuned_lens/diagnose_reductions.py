#!/usr/bin/env python3
"""Optional reproduction of the wide-vocabulary metric accuracy diagnosis.

Runs the same four prepared records through the unchanged F32 reference model.
Compares the original PyTorch F32 analysis, simulated scalar/pairwise F32
reductions, and two independent F64 calculations. No thresholds, checkpoints,
primary reference artifacts, or experiment results are modified.

    python3 examples/mechanistic/tuned_lens/diagnose_reductions.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from prepare import fingerprint, strict_json, verified_output_cache
from reference import HERE, OUTPUT, PythiaReference, atomic_write, file_hash


def sum_f32(values, method):
    """Compare reduction order while keeping each resulting scalar F32."""
    if method == "sequential":
        return np.cumsum(values, axis=-1, dtype=np.float32)[..., -1:]
    if method == "wide":
        return values.sum(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
    while values.shape[-1] > 1:
        width = values.shape[-1]
        if width % 2:
            zero = np.zeros((*values.shape[:-1], 1), dtype=np.float32)
            values = np.concatenate([values, zero], axis=-1)
            width += 1
        values = values.reshape(*values.shape[:-1], width // 2, 2).sum(axis=-1, dtype=np.float32)
    return values


def log_probs_f32(logits, method):
    shifted = logits.numpy() - logits.numpy().max(axis=-1, keepdims=True)
    normalizer = sum_f32(np.exp(shifted, dtype=np.float32), method)
    return shifted - np.log(normalizer, dtype=np.float32)


def numpy_f64_kl(final, raw):
    def log_probs(logits):
        values = logits.numpy().astype(np.float64)
        values -= values.max(axis=-1, keepdims=True)
        return values - np.log(np.exp(values).sum(axis=-1, keepdims=True, dtype=np.float64))
    final_probs, raw_probs = log_probs(final), log_probs(raw)
    return float((np.exp(final_probs) * (final_probs - raw_probs)).sum(axis=-1).mean())


def diagnose(final, raw):
    row = {}
    for dtype, name in ((torch.float32, "original_torch_f32"), (torch.float64, "torch_f64")):
        p = F.log_softmax(final.to(dtype), dim=-1)
        q = F.log_softmax(raw.to(dtype), dim=-1)
        row[name + "_kl"] = (p.exp() * (p - q)).sum(dim=-1).double().mean().item()
        row[name + "_probability_mass_mean"] = p.double().exp().sum(dim=-1).mean().item()
    for method in ("sequential", "pairwise", "wide"):
        p, q = log_probs_f32(final, method), log_probs_f32(raw, method)
        terms = np.exp(p, dtype=np.float32) * (p - q)
        row[method + "_f32_kl"] = float(sum_f32(terms, method).mean(dtype=np.float64))
        row[method + "_probability_mass_mean"] = float(np.exp(p, dtype=np.float32).sum(axis=-1, dtype=np.float64).mean())
    row["numpy_f64_kl"] = numpy_f64_kl(final, raw)
    return row


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "reduction_diagnostics.json")
    parser.add_argument("--export-logits", action="store_true", help="also write the 206 MB diagnostic.safetensors")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not verified_output_cache(OUTPUT, fingerprint(HERE / "manifest.json")):
        raise ValueError("Preparation is absent or stale; run prepare.py first")
    config = strict_json((OUTPUT / "model/config.json").read_bytes())
    fixture = strict_json((OUTPUT / "fixture.json").read_bytes())
    model = PythiaReference(load_file(OUTPUT / "model/model.safetensors"), config)
    records, exported = [], {}
    for record in fixture["records"]:
        ids = torch.tensor([record["input_ids"]], dtype=torch.long)
        final, states = model.forward(ids)
        raw = model.readout(states[0])
        index = record["record_index"]
        records.append({"record_index": index, "input_tokens": ids.numel(), **diagnose(final, raw)})
        if args.export_logits:
            exported[f"final.{index}"] = final.contiguous()
            exported[f"raw0.{index}"] = raw.contiguous()
    count = sum(row["input_tokens"] for row in records)
    methods = ("original_torch_f32", "sequential_f32", "pairwise_f32", "wide_f32", "torch_f64", "numpy_f64")
    result = {
        "schema_version": 1,
        "method": "Same actual F32 final and embedding-lens logits; only analysis arithmetic differs. NumPy simulates reduction order, not backend-specific exponential approximations.",
        "torch": str(torch.__version__), "numpy": np.__version__,
        "preparation_sha256": file_hash(OUTPUT / "preparation.json"),
        "diagnostic_script_sha256": file_hash(Path(__file__)),
        "records": records,
        "aggregate_raw0_kl": {
            method: sum(row[method + "_kl"] * row["input_tokens"] for row in records) / count
            for method in methods
        },
    }
    if args.export_logits:
        save_file(exported, OUTPUT / "diagnostic.safetensors")
    atomic_write(args.output, (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps(result["aggregate_raw0_kl"], indent=2))
    print(f"Saved reproducible numerical diagnosis: {args.output}")


if __name__ == "__main__":
    main()
