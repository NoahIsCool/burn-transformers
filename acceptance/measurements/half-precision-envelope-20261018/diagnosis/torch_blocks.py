"""Scratch (half2 S9 diagnosis): per-block activations of Qwen2-7B in the official graph (not committed).

Usage: python3 -I torch_blocks.py GENERATOR MODEL_DIR TRUTH_REFERENCE_JSON DTYPE DEVICE ATTENTION OUT_NPZ

Loads the checkpoint exactly as the committed generator does (its
`load_causal_lm`, imported from GENERATOR), evaluates every prompt of the truth
bundle statelessly with the truth's token IDs, and records, per decoder block
and prompt, as F32 arrays `[tokens, channels]`: the residual entering the block
(`pre`), the attention output before its residual addition (`attn`), the MLP
input to `down_proj` (`act`, `act_fn(gate) * up`), the MLP output (`mlp`), and
the residual after the block (`post`). For eager attention it also records,
per block and prompt, the largest |q k^T| entry the eager product holds in the
model dtype before the scale (`scores_raw_max`, as F32; inf if it overflowed),
its count of non-finite entries, and the (head, query, key) of the largest
finite entry. Values are converted to F32 exactly from F16/BF16 (and rounded
from F64). Nothing is written but OUT_NPZ.
"""

import importlib.util
import json
import sys

import numpy as np


def main():
    generator, model_dir, truth, dtype, device, attention, out = sys.argv[1:8]
    spec = importlib.util.spec_from_file_location("generator", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch
    from transformers.models.qwen2 import modeling_qwen2

    manifest = json.load(open(truth))
    prompts = [prompt["token_ids"] for prompt in manifest["prompts"]]
    model = module.load_causal_lm(model_dir, dtype, None, attention, device)
    layers = model.model.layers
    record = {}
    current = {"prompt": None}

    def put(name, layer, tensor):
        """Stores one [tokens, channels] activation of the current prompt."""
        key = f"p{current['prompt']}.l{layer}.{name}"
        record[key] = tensor.detach().to("cpu", torch.float32).reshape(-1, tensor.shape[-1]).numpy()

    original = modeling_qwen2.eager_attention_forward

    def eager(module_, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        """Records the eager product's range in the model dtype, then runs the official function."""
        key_states = modeling_qwen2.repeat_kv(key, module_.num_key_value_groups)
        raw = torch.matmul(query, key_states.transpose(2, 3))
        finite = torch.isfinite(raw)
        layer = module_.layer_idx
        prefix = f"p{current['prompt']}.l{layer}"
        record[f"{prefix}.scores_nonfinite"] = np.array(int((~finite).sum()))
        magnitude = raw.abs().to(torch.float32)
        record[f"{prefix}.scores_raw_max"] = np.array(float(magnitude.max()))
        masked = torch.where(finite, magnitude, torch.zeros_like(magnitude))
        index = int(masked.flatten().argmax())
        _, heads, queries, keys = raw.shape
        record[f"{prefix}.scores_finite_argmax"] = np.array(
            [index // (queries * keys) % heads, index // keys % queries, index % keys, float(masked.flatten()[index])])
        return original(module_, query, key, value, attention_mask, scaling, dropout, **kwargs)

    modeling_qwen2.eager_attention_forward = eager
    handles = []
    for index, layer in enumerate(layers):
        def pre(module_, args, kwargs, index=index):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            put("pre", index, hidden)
        def post(module_, args, kwargs, output, index=index):
            put("post", index, output[0] if isinstance(output, tuple) else output)
        def attn(module_, args, output, index=index):
            put("attn", index, output[0] if isinstance(output, tuple) else output)
        def mlp(module_, args, output, index=index):
            put("mlp", index, output)
        def down(module_, args, index=index):
            put("act", index, args[0])
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(layer.register_forward_hook(post, with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(attn))
        handles.append(layer.mlp.register_forward_hook(mlp))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(down))
    with torch.no_grad():
        for number, tokens in enumerate(prompts):
            current["prompt"] = number
            ids = torch.tensor([tokens], device=device)
            logits = model(input_ids=ids, use_cache=False).logits
            record[f"p{number}.logits_finite_rows"] = torch.isfinite(logits[0]).all(dim=-1).cpu().numpy()
    np.savez_compressed(out, **record)
    print(f"{dtype} {device} {attention}: {len(prompts)} prompts, {len(layers)} blocks, {len(record)} arrays -> {out}")


if __name__ == "__main__":
    main()
