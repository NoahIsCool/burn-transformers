"""Scratch diagnostic: the official implementation's own half-precision spread (not committed).

Usage: python3 -I official_variants.py MODEL_DIR TRUTH_BUNDLE OFFICIAL_BF16_BUNDLE OUT_DIR [VARIANT...]

Evaluates the local checkpoint with Transformers in several dtype/device/attention
variants (local files only), compares every prompt's logits with the truth bundle
(F32 or F64), prints each prompt's max |delta| (position, vocabulary entry) and mean
|delta|, and saves each prompt's per-block residual stream (after each decoder layer,
before the final norm; plus the embedding and the final norm) and logits as raw F32
under OUT_DIR/<variant>/prompt<i>/ for a block-by-block comparison with Burn dumps.
The cpu-bfloat16-eager variant is the bundle generator's configuration; its logits
are also compared with the official BF16 bundle byte for byte.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers

VARIANTS = {
    "cpu-float32-eager": ("cpu", "float32", "eager"),
    "cpu-bfloat16-eager": ("cpu", "bfloat16", "eager"),
    "cpu-bfloat16-sdpa": ("cpu", "bfloat16", "sdpa"),
    "mps-bfloat16-eager": ("mps", "bfloat16", "eager"),
    "mps-bfloat16-sdpa": ("mps", "bfloat16", "sdpa"),
    "cpu-float16-eager": ("cpu", "float16", "eager"),
    "mps-float16-eager": ("mps", "float16", "eager"),
}
# "-burnsilu" variants replace each MLP's fused SiLU (one rounding of x * sigmoid(x)
# computed in float) by x * sigmoid(x) in the activation dtype (sigmoid rounded, then
# the product rounded), which is the operation order of Burn's activation::silu.
for _name, _value in list(VARIANTS.items()):
    VARIANTS[_name + "-burnsilu"] = _value + ("burnsilu",)


class RoundedSigmoidSilu(torch.nn.Module):
    """SiLU as x * sigmoid(x) with the sigmoid rounded to the activation dtype first."""

    def forward(self, x):
        return x * torch.sigmoid(x)


def read_bundle(root):
    """Returns (manifest, [token ids], [logits [T, V] as float64])."""
    manifest = json.loads((root / "reference.json").read_text())
    vocabulary = manifest["vocabulary_size"]
    prompts, logits = [], []
    for prompt in manifest["prompts"]:
        tokens = prompt["token_ids"]
        values = np.fromfile(root / prompt["logits_file"], dtype="<f4").reshape(len(tokens), vocabulary)
        prompts.append(tokens)
        logits.append(values.astype(np.float64))
    return manifest, prompts, logits


def stats(actual, truth):
    """Returns max |delta|, its (position, entry), and mean |delta| in float64."""
    delta = np.abs(actual.astype(np.float64) - truth)
    index = int(delta.argmax())
    return float(delta.max()), divmod(index, truth.shape[1]), float(delta.mean())


def main():
    model_dir, truth_dir, official_dir, out = map(Path, sys.argv[1:5])
    chosen = sys.argv[5:] or list(VARIANTS)
    _, prompts, truth = read_bundle(truth_dir)
    official_manifest, _, official = read_bundle(official_dir)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    transformers.utils.logging.disable_progress_bar()
    summary = {}
    for name in chosen:
        device, dtype, attention = VARIANTS[name][:3]
        burnsilu = len(VARIANTS[name]) > 3
        started = time.monotonic()
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=getattr(torch, dtype), attn_implementation=attention
        )
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    parameter.data = parameter.data.to(getattr(torch, dtype))
        if burnsilu:
            for layer in model.model.layers:
                layer.mlp.act_fn = RoundedSigmoidSilu()
        model.eval().to(device)
        captured = {}

        def hook(label):
            def record(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[label] = tensor[0].to(torch.float32).cpu().numpy()
            return record

        handles = [model.model.embed_tokens.register_forward_hook(hook("embedding"))]
        handles += [layer.register_forward_hook(hook(f"layer{i}")) for i, layer in enumerate(model.model.layers)]
        handles.append(model.model.norm.register_forward_hook(hook("final_norm")))
        rows = []
        for index, tokens in enumerate(prompts):
            captured.clear()
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([tokens], device=device), use_cache=False).logits
            logits = logits[0].to(torch.float32).cpu().numpy()
            maximum, (position, entry), mean = stats(logits, truth[index])
            line = (f"{name} prompt {index}: {len(tokens)} tokens, max |d| = {maximum:.6e} at position {position} "
                    f"token {entry}, mean |d| = {mean:.6e}")
            if name == "cpu-bfloat16-eager":
                same = np.array_equal(logits.astype(np.float32), official[index].astype(np.float32))
                line += f"; identical to the official BF16 bundle: {same}"
            print(line, flush=True)
            rows.append({"prompt": index, "max": maximum, "position": position, "entry": entry, "mean": mean})
            target = out / name / f"prompt{index}"
            target.mkdir(parents=True, exist_ok=True)
            logits.astype("<f4").tofile(target / "logits.bin")
            for label, value in captured.items():
                value.astype("<f4").tofile(target / f"{label}.bin")
        for handle in handles:
            handle.remove()
        summary[name] = {"rows": rows, "seconds": time.monotonic() - started}
        del model
        if device == "mps":
            torch.mps.empty_cache()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"torch {torch.__version__} transformers {transformers.__version__}; official bundle "
          f"{official_dir} {official_manifest['provenance'].get('dtype')}")


if __name__ == "__main__":
    main()
