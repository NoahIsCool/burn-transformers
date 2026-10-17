"""Scratch diagnostic: the official implementation's own cached-decode versus stateless spread (not committed).

Usage: python3 -I official_decode.py MODEL_DIR TRUTH_BUNDLE OUT_DIR VARIANT...

For prompt 0 of the bundle followed by its greedy tokens (the sequence the crate's
cached greedy test teacher-forces), each variant (device-dtype-attention, as in
official_variants.py) computes one stateless pass over the whole sequence and a
cached pass (prefill of prompt 0, then one token per step with the KV cache).
Prints, per decode step, max |delta| between the cached and stateless logits at
the same position, and each pass's max |delta| from the F32 stateless pass (the
cpu-float32-eager variant, which must be listed first). Saves the logits as raw
F32 under OUT_DIR/<variant>/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers


def main():
    model_dir, truth_dir, out = map(Path, sys.argv[1:4])
    variants = sys.argv[4:]
    manifest = json.loads((truth_dir / "reference.json").read_text())
    prompt = manifest["prompts"][0]["token_ids"]
    sequence = prompt + [step["token"] for step in manifest["greedy"]["steps"]]
    first = len(prompt)
    torch.use_deterministic_algorithms(True)
    transformers.utils.logging.disable_progress_bar()
    reference = None
    for name in variants:
        device, dtype, attention = name.split("-")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=getattr(torch, dtype), attn_implementation=attention
        )
        model.eval().to(device)
        with torch.no_grad():
            stateless = model(input_ids=torch.tensor([sequence], device=device), use_cache=False).logits[0]
            stateless = stateless.to(torch.float32).cpu().numpy()[first:]
            output = model(input_ids=torch.tensor([prompt], device=device), use_cache=True)
            past = output.past_key_values
            cached = []
            for token in sequence[first:]:
                output = model(input_ids=torch.tensor([[token]], device=device), past_key_values=past, use_cache=True)
                past = output.past_key_values
                cached.append(output.logits[0, -1].to(torch.float32).cpu().numpy())
            cached = np.stack(cached)
        if reference is None:
            reference = stateless.astype(np.float64)
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        stateless.astype("<f4").tofile(target / "stateless.bin")
        cached.astype("<f4").tofile(target / "cached.bin")
        gaps = np.abs(cached.astype(np.float64) - stateless).max(axis=1)
        to_truth_cached = np.abs(cached - reference).max(axis=1)
        to_truth_stateless = np.abs(stateless - reference).max(axis=1)
        worst = int(gaps.argmax())
        print(f"{name}: {len(gaps)} decode steps from position {first}: cached vs stateless max |d| = "
              f"{gaps.max():.6e} at step {worst}; steps above 0.5: {int((gaps > 0.5).sum())}; "
              f"median {np.median(gaps):.4e}", flush=True)
        print(f"{name}: per-step cached-vs-stateless " + " ".join(f"{g:.4f}" for g in gaps), flush=True)
        print(f"{name}: per-step max |d| to F32, cached " + " ".join(f"{g:.4f}" for g in to_truth_cached), flush=True)
        print(f"{name}: per-step max |d| to F32, stateless " + " ".join(f"{g:.4f}" for g in to_truth_stateless),
              flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
