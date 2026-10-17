"""Scratch diagnostic: the official Gemma 3 implementation's own half-precision spread (not committed).

Usage: python3 -I gemma_official_variants.py MODEL_DIR TRUTH_BUNDLE OFFICIAL_BF16_BUNDLE OUT_DIR [VARIANT...]

Loads the local checkpoint exactly as scripts/generate_causal_lm_reference.py does
(AutoModelForCausalLM, local files only, every floating parameter cast to the dtype,
the chosen attention implementation), evaluates every prompt of the truth bundle,
and prints each prompt's max |delta| (position, vocabulary entry) and mean |delta|
against the truth (F32), with the rule's bounds (1.2 x the official BF16 statistics).
cpu-bfloat16-eager is the generator's configuration and is compared byte for byte
with the official BF16 bundle. Saves each prompt's per-layer residual stream (after
each decoder layer; plus the scaled embedding and the final norm) and logits as raw
F32 under OUT_DIR/<variant>/prompt<i>/ for a block-by-block comparison with Burn.

Variants ending in "-burngelu" replace every MLP's fused GELU-tanh (computed in F32
inside one kernel and rounded once) by the operation sequence of Burn 0.21's
activation::gelu_approximate as Flex evaluates it in the activation dtype: every
elementwise operation rounds its result to the dtype, and every scalar operand is
first rounded to the dtype (0.044715 -> 0.044677734 and sqrt(2/pi) -> 0.796875 in
BF16), see burn-flex 0.21 ops/binary.rs scalar_op. "-f32gelu" variants are the
control: the same op sequence evaluated in F32 and rounded once.
"""

import json
import math
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
for _name, _value in list(VARIANTS.items()):
    VARIANTS[_name + "-burngelu"] = _value + ("burngelu",)
    VARIANTS[_name + "-f32gelu"] = _value + ("f32gelu",)

SQRT_2_OVER_PI = (2.0 / math.sqrt(math.pi)) * (1.0 / math.sqrt(2.0))


class BurnGeluTanh(torch.nn.Module):
    """Burn's gelu_approximate with each op rounded to the activation dtype and dtype-rounded scalars."""

    def forward(self, x):
        dtype = x.dtype
        if dtype == torch.float32:
            return torch.nn.functional.gelu(x, approximate="tanh")

        def r(value):
            return value.to(dtype).to(torch.float32)

        c1 = float(torch.tensor(0.044715, dtype=torch.float32).to(dtype).float())
        c2 = float(torch.tensor(SQRT_2_OVER_PI, dtype=torch.float32).to(dtype).float())
        xf = x.to(torch.float32)
        cube = r(xf.pow(3.0))
        inner = r(xf + r(cube * c1))
        inner = r(inner * c2)
        shifted = r(r(torch.tanh(inner)) + 1.0)
        return r(r(xf * shifted) * 0.5).to(dtype)


class F32GeluTanh(torch.nn.Module):
    """GELU-tanh evaluated in F32 and rounded once to the activation dtype."""

    def forward(self, x):
        return torch.nn.functional.gelu(x.to(torch.float32), approximate="tanh").to(x.dtype)


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


def text_model(model):
    """Returns the decoder that holds embed_tokens, layers, and norm."""
    inner = model.model
    return getattr(inner, "language_model", inner)


def main():
    model_dir, truth_dir, official_dir, out = map(Path, sys.argv[1:5])
    chosen = sys.argv[5:] or list(VARIANTS)
    _, prompts, truth = read_bundle(truth_dir)
    official_manifest, _, official = read_bundle(official_dir)
    bounds = []
    for index in range(len(prompts)):
        maximum, _, mean = stats(official[index], truth[index])
        bounds.append((1.2 * maximum, 1.2 * mean))
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    transformers.utils.logging.disable_progress_bar()
    summary = {}
    for name in chosen:
        device, dtype, attention = VARIANTS[name][:3]
        gelu = VARIANTS[name][3] if len(VARIANTS[name]) > 3 else None
        started = time.monotonic()
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=getattr(torch, dtype), attn_implementation=attention
        )
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    parameter.data = parameter.data.to(getattr(torch, dtype))
        decoder = text_model(model)
        if gelu is not None:
            for layer in decoder.layers:
                layer.mlp.act_fn = BurnGeluTanh() if gelu == "burngelu" else F32GeluTanh()
        model.eval().to(device)
        captured = {}

        def hook(label):
            def record(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[label] = tensor[0].to(torch.float32).cpu().numpy()
            return record

        handles = [decoder.embed_tokens.register_forward_hook(hook("embedding"))]
        handles += [layer.register_forward_hook(hook(f"layer{i}")) for i, layer in enumerate(decoder.layers)]
        handles.append(decoder.norm.register_forward_hook(hook("final_norm")))
        rows = []
        for index, tokens in enumerate(prompts):
            captured.clear()
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([tokens], device=device), use_cache=False).logits
            logits = logits[0].to(torch.float32).cpu().numpy()
            finite = bool(np.isfinite(logits).all())
            maximum, (position, entry), mean = stats(logits, truth[index])
            bound_max, bound_mean = bounds[index]
            verdict = "within" if finite and maximum <= bound_max and mean <= bound_mean else "OUTSIDE"
            line = (f"{name} prompt {index}: {len(tokens)} tokens, max |d| = {maximum:.6e} at position {position} "
                    f"token {entry} (bound {bound_max:.6e}), mean |d| = {mean:.6e} (bound {bound_mean:.6e}); "
                    f"finite={finite}; {verdict} the rule")
            if name == "cpu-bfloat16-eager":
                same = np.array_equal(logits.astype(np.float32), official[index].astype(np.float32))
                line += f"; identical to the official BF16 bundle: {same}"
            print(line, flush=True)
            rows.append({"prompt": index, "max": maximum, "position": position, "entry": entry, "mean": mean,
                         "finite": finite})
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
