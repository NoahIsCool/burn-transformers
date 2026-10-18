"""Scratch diagnostic (not committed): where Gemma 3's activations leave the F16 range, sub-block by sub-block.

Usage: python3 -I gemma3_overflow_points.py MODEL_DIR BUNDLE_DIR DTYPE OUT_JSON [MAX_PROMPTS] [--no-bos]

With --no-bos, every prompt is evaluated a second time without its leading BOS token, to test
whether the overflow depends on the BOS token at position 0.

Loads the local checkpoint as scripts/generate_causal_lm_reference.py does (AutoModelForCausalLM,
local files only, every floating parameter cast to DTYPE, eager attention, CPU) and evaluates the
token IDs of every prompt in BUNDLE_DIR/reference.json. Hooks record, for every decoder layer and
in execution order, the largest finite |value| (with its position and hidden channel) and the
number of non-finite values at each point:

  in, input_norm, q, k, v, attn_ctx (o_proj input), o_proj, post_attn_norm, resid_mid (in + post_attn_norm,
  added in DTYPE as the model does), pre_ff_norm, gate, up, act_x_up (down_proj input), down,
  post_ff_norm, out (layer output = resid_mid + post_ff_norm).

For an F32 or BF16 run it reports the first point whose magnitude reaches 65520 (the smallest
value that rounds to +inf in F16 under round-to-nearest-even); for an F16 run, the first point
holding a non-finite value. Nothing is downloaded.
"""
import json
import sys
import time
from pathlib import Path

import torch
import transformers

F16_OVERFLOW = 65520.0
POINTS = ["in", "input_norm", "q", "k", "v", "attn_ctx", "o_proj", "post_attn_norm", "resid_mid",
          "pre_ff_norm", "gate", "up", "act_x_up", "down", "post_ff_norm", "out"]


def summarize(tensor):
    """Returns max finite |x|, its (position, channel), position-0 and later-position maxima, non-finite count."""
    t = tensor[0].detach().to(torch.float32)
    t = t.reshape(t.shape[0], -1) if t.dim() > 2 else t
    finite = torch.isfinite(t)
    nonfinite = int((~finite).sum())
    a = torch.where(finite, t.abs(), torch.zeros_like(t))
    index = int(a.argmax())
    position, channel = divmod(index, a.shape[1])
    return {
        "max": float(a.max()),
        "position": position,
        "channel": channel,
        "pos0": float(a[0].max()),
        "later": float(a[1:].max()) if a.shape[0] > 1 else 0.0,
        "nonfinite": nonfinite,
        "first_nonfinite_position": int((~finite).any(dim=1).nonzero()[0]) if nonfinite else None,
    }


def text_model(model):
    """Returns the decoder that holds embed_tokens, layers, and norm."""
    inner = model.model
    return getattr(inner, "language_model", inner)


def main():
    args = [a for a in sys.argv[1:] if a != "--no-bos"]
    no_bos = "--no-bos" in sys.argv[1:]
    model_dir, bundle_dir, dtype_name, out_json = args[0:4]
    max_prompts = int(args[4]) if len(args) > 4 else None
    dtype = getattr(torch, dtype_name)
    manifest = json.loads((Path(bundle_dir) / "reference.json").read_text())
    prompts = [p["token_ids"] for p in manifest["prompts"]][:max_prompts]
    labels = [f"prompt {i}" for i in range(len(prompts))]
    if no_bos:
        bos = prompts[0][0]
        assert all(p[0] == bos for p in prompts), "every prompt must start with the same BOS id"
        prompts = prompts + [p[1:] for p in prompts]
        labels = labels + [f"prompt {i} without BOS {bos}" for i in range(len(labels))]
    torch.manual_seed(0)
    transformers.utils.logging.disable_progress_bar()
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
    load_seconds = time.monotonic() - started
    config = decoder.config
    events = []
    stash = {}

    def record(layer, point, tensor):
        events.append((layer, point, summarize(tensor)))

    handles = []
    for i, layer in enumerate(decoder.layers):
        def pre_layer(_m, args, kwargs, i=i):
            x = args[0] if args else kwargs["hidden_states"]
            stash["in"] = x
            record(i, "in", x)
        handles.append(layer.register_forward_pre_hook(pre_layer, with_kwargs=True))
        handles.append(layer.input_layernorm.register_forward_hook(lambda _m, _a, o, i=i: record(i, "input_norm", o)))
        attn = layer.self_attn
        handles.append(attn.q_proj.register_forward_hook(lambda _m, _a, o, i=i: record(i, "q", o)))
        handles.append(attn.k_proj.register_forward_hook(lambda _m, _a, o, i=i: record(i, "k", o)))
        handles.append(attn.v_proj.register_forward_hook(lambda _m, _a, o, i=i: record(i, "v", o)))
        handles.append(attn.o_proj.register_forward_pre_hook(lambda _m, a, i=i: record(i, "attn_ctx", a[0])))
        handles.append(attn.o_proj.register_forward_hook(lambda _m, _a, o, i=i: record(i, "o_proj", o)))

        def post_attn(_m, _a, o, i=i):
            record(i, "post_attn_norm", o)
            mid = stash["in"] + o
            stash["mid"] = mid
            record(i, "resid_mid", mid)
        handles.append(layer.post_attention_layernorm.register_forward_hook(post_attn))
        handles.append(layer.pre_feedforward_layernorm.register_forward_hook(lambda _m, _a, o, i=i: record(i, "pre_ff_norm", o)))
        mlp = layer.mlp
        handles.append(mlp.gate_proj.register_forward_hook(lambda _m, _a, o, i=i: record(i, "gate", o)))
        handles.append(mlp.up_proj.register_forward_hook(lambda _m, _a, o, i=i: record(i, "up", o)))
        handles.append(mlp.down_proj.register_forward_pre_hook(lambda _m, a, i=i: record(i, "act_x_up", a[0])))
        handles.append(mlp.down_proj.register_forward_hook(lambda _m, _a, o, i=i: record(i, "down", o)))
        handles.append(layer.post_feedforward_layernorm.register_forward_hook(lambda _m, _a, o, i=i: record(i, "post_ff_norm", o)))
        handles.append(layer.register_forward_hook(lambda _m, _a, o, i=i: record(i, "out", o[0] if isinstance(o, tuple) else o)))

    results = []
    for index, tokens in enumerate(prompts):
        events.clear()
        t0 = time.monotonic()
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([tokens]), use_cache=False).logits
        seconds = time.monotonic() - t0
        logits_finite = bool(torch.isfinite(logits).all())
        first_over = None
        first_nonfinite = None
        for layer_index, point, s in events:
            if first_nonfinite is None and s["nonfinite"]:
                first_nonfinite = {"layer": layer_index, "point": point, **s}
            if first_over is None and s["max"] >= F16_OVERFLOW:
                first_over = {"layer": layer_index, "point": point, **s}
        per_point = {}
        for point in POINTS:
            rows = [(l, s) for l, p, s in events if p == point]
            best = max(rows, key=lambda r: r[1]["max"])
            per_point[point] = {"max": best[1]["max"], "layer": best[0], "position": best[1]["position"],
                                "channel": best[1]["channel"],
                                "layers_at_or_above_65520": [l for l, s in rows if s["max"] >= F16_OVERFLOW],
                                "max_later_positions": max(s["later"] for _, s in rows)}
        layers = {}
        for l, p, s in events:
            layers.setdefault(l, {})[p] = s
        results.append({"prompt": index, "label": labels[index], "first_token": tokens[0], "tokens": len(tokens), "seconds": seconds,
                        "logits_finite": logits_finite, "first_point_at_or_above_65520": first_over,
                        "first_nonfinite_point": first_nonfinite, "per_point_max": per_point, "layers": layers})
        print(f"{labels[index]} ({len(tokens)} tokens, first id {tokens[0]}, {seconds:.1f} s): logits finite={logits_finite}; "
              f"first point >= 65520: {first_over and (first_over['layer'], first_over['point'], round(first_over['max'], 1), first_over['position'], first_over['channel'])}; "
              f"first non-finite: {first_nonfinite and (first_nonfinite['layer'], first_nonfinite['point'], first_nonfinite['first_nonfinite_position'])}",
              flush=True)
        for point in POINTS:
            pp = per_point[point]
            print(f"   {point:>14}: max {pp['max']:.4g} (layer {pp['layer']}, pos {pp['position']}, ch {pp['channel']}); "
                  f"later positions max {pp['max_later_positions']:.4g}; layers >= 65520: {pp['layers_at_or_above_65520'][:6]}"
                  f"{'...' if len(pp['layers_at_or_above_65520']) > 6 else ''} ({len(pp['layers_at_or_above_65520'])})", flush=True)
        outs = " ".join(f"{l}:{layers[l]['out']['max']:.4g}@{layers[l]['out']['position']}" for l in sorted(layers))
        print(f"   layer outputs max|x|@position: {outs}", flush=True)
        if first_nonfinite is not None:
            l = first_nonfinite["layer"]
            path = " ".join(f"{p}:{layers[l][p]['max']:.4g}/{layers[l][p]['nonfinite']}" for p in POINTS if p in layers[l])
            print(f"   layer {l} points max|x|/non-finite: {path}", flush=True)
    for handle in handles:
        handle.remove()
    summary = {
        "model_dir": model_dir, "bundle": bundle_dir, "dtype": dtype_name,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "hidden_size": config.hidden_size, "layers": config.num_hidden_layers,
        "load_seconds": load_seconds, "prompts": results,
    }
    Path(out_json).write_text(json.dumps(summary))
    print(f"torch {torch.__version__} transformers {transformers.__version__}; {model_dir} {dtype_name}; load {load_seconds:.1f} s")


if __name__ == "__main__":
    main()
