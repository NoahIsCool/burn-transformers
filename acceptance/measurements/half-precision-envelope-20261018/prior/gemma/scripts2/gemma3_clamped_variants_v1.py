"""Scratch diagnostic (not committed): would a clamped F16 Gemma 3 be a usable experiment?

Usage: python3 -I gemma3_clamped_variants.py MODEL_DIR TRUTH_BUNDLE OFFICIAL_BF16_BUNDLE OUT_JSON VARIANT...

Loads the local checkpoint as scripts/generate_causal_lm_reference.py does (AutoModelForCausalLM,
local files only, every floating parameter cast to the dtype, eager attention, CPU), evaluates
every prompt of the F32 truth bundle, and compares the logits with the truth: max |delta|, mean
|delta|, the rule's bounds (1.2 x the official BF16 statistics), and teacher-forced argmax
agreement at every position. Nothing is downloaded.

Variants:
  float16            the unmodified model in F16 (overflows; the baseline);
  float16-clamp      F16 with Transformers' own F16 clamp (T5/BART convention: when a residual
                     tensor holds an inf, clamp it to +-(finfo(float16).max - 1000), otherwise to
                     +-finfo(float16).max), applied to the residual after attention (the input of
                     pre_feedforward_layernorm) and after each decoder layer;
  float32-clamp      F32 with the same clamp applied wherever |x| would overflow F16 (> 65504),
                     isolating the clamp's effect from F16 rounding;
  bfloat16           the unmodified model in BF16 (the calibration's configuration, for scale).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers

F16_MAX = float(torch.finfo(torch.float16).max)


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
    """Returns max |delta|, its (position, entry), mean |delta|, and argmax agreement per position."""
    delta = np.abs(actual.astype(np.float64) - truth)
    index = int(delta.argmax())
    agree = (actual.argmax(axis=1) == truth.argmax(axis=1)).tolist()
    return float(delta.max()), divmod(index, truth.shape[1]), float(delta.mean()), agree


def text_model(model):
    """Returns the decoder that holds embed_tokens, layers, and norm."""
    inner = model.model
    return getattr(inner, "language_model", inner)


def clamp(x, counter):
    """Transformers' F16 clamp convention; in F32 it fires where an F16 value would overflow."""
    if x.dtype == torch.float16:
        overflowed = bool(torch.isinf(x).any())
        value = F16_MAX - 1000 if overflowed else F16_MAX
    else:
        overflowed = bool((x.abs() > F16_MAX).any())
        value = F16_MAX - 1000 if overflowed else float("inf")
    if overflowed:
        counter["clamped_tensors"] += 1
        counter["clamped_entries"] += int((x.abs() > value).sum())
    return torch.clamp(x, min=-value, max=value)


def main():
    model_dir, truth_dir, official_dir, out_json = map(Path, sys.argv[1:5])
    variants = sys.argv[5:]
    _, prompts, truth = read_bundle(truth_dir)
    _, _, official = read_bundle(official_dir)
    bounds = []
    for index in range(len(prompts)):
        maximum, _, mean, _ = stats(official[index], truth[index])
        bounds.append((1.2 * maximum, 1.2 * mean, maximum, mean))
    torch.manual_seed(0)
    transformers.utils.logging.disable_progress_bar()
    summary = {"torch": torch.__version__, "transformers": transformers.__version__, "variants": {}}
    for name in variants:
        dtype = {"float16": torch.float16, "float16-clamp": torch.float16, "float32-clamp": torch.float32,
                 "bfloat16": torch.bfloat16}[name]
        started = time.monotonic()
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=dtype, attn_implementation="eager"
        )
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    parameter.data = parameter.data.to(dtype)
        model.eval()
        decoder = text_model(model)
        counter = {"clamped_tensors": 0, "clamped_entries": 0}
        handles = []
        if name.endswith("-clamp"):
            for layer in decoder.layers:
                handles.append(layer.pre_feedforward_layernorm.register_forward_pre_hook(
                    lambda _m, args: (clamp(args[0], counter),) + tuple(args[1:])))

                def clamp_output(_m, _a, output):
                    if isinstance(output, tuple):
                        return (clamp(output[0], counter),) + tuple(output[1:])
                    return clamp(output, counter)
                handles.append(layer.register_forward_hook(clamp_output))
        rows = []
        for index, tokens in enumerate(prompts):
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([tokens]), use_cache=False).logits
            logits = logits[0].to(torch.float32).numpy()
            finite = bool(np.isfinite(logits).all())
            bound_max, bound_mean, official_max, official_mean = bounds[index]
            if finite:
                maximum, (position, entry), mean, agree = stats(logits, truth[index])
            else:
                maximum, position, entry, mean, agree = float("nan"), None, None, float("nan"), []
            verdict = "within" if finite and maximum <= bound_max and mean <= bound_mean else "OUTSIDE"
            print(f"{name} prompt {index}: {len(tokens)} tokens, finite={finite}; max |d| = {maximum:.6g} at "
                  f"position {position} token {entry} (official BF16 {official_max:.6g}, bound {bound_max:.6g}); "
                  f"mean |d| = {mean:.6g} (official BF16 {official_mean:.6g}, bound {bound_mean:.6g}); {verdict} the rule; "
                  f"argmax agrees with F32 at {sum(agree)}/{len(tokens)} positions", flush=True)
            rows.append({"prompt": index, "tokens": len(tokens), "finite": finite, "max": maximum,
                         "position": position, "entry": entry, "mean": mean, "argmax_agree": agree,
                         "bound_max": bound_max, "bound_mean": bound_mean})
        for handle in handles:
            handle.remove()
        seconds = time.monotonic() - started
        print(f"{name}: clamp fired on {counter['clamped_tensors']} tensors, {counter['clamped_entries']} entries "
              f"(all prompts); {seconds:.1f} s", flush=True)
        summary["variants"][name] = {"rows": rows, "clamp": counter, "seconds": seconds}
        del model
    out_json.write_text(json.dumps(summary, indent=1))
    print(f"torch {torch.__version__} transformers {transformers.__version__}; truth {truth_dir}; official {official_dir}")


if __name__ == "__main__":
    main()
