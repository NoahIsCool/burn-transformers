"""Scratch sensitivity check: how far known half-precision fidelity defects move the official BF16 prompt logits (not committed).

Usage: python3 -I prompt_defects.py MODEL_DIR TRUTH_BUNDLE OUT_JSON

Loads the official Transformers model in BF16 on the CPU with eager attention
through round 2's frozen copy of the extended generator (the graph of the
committed BF16 twin), evaluates the truth bundle's prompts (token IDs from its
manifest) once cleanly ("none", which must reproduce the twin's logits) and
once per injected defect, and reports each prompt's max and mean |Δ| from the
truth logits (F64 differences, F64 sums). Each defect is a precision mistake
a half-precision port can make while staying "correct" in F32:

  silu-twice-rounded : act_fn(x) = x * sigmoid(x) in BF16 (the sigmoid rounded
                       before the product, as Burn's half-precision SiLU does)
  rmsnorm-bf16       : every *RMSNorm computes its mean square and rsqrt in BF16
                       instead of F32
  softmax-bf16       : eager attention normalizes with BF16 intermediates (exponentials,
                       their sum and the quotient each rounded) instead of F32 and one rounding
  rope-angles-bf16   : rotary angles position * inv_freq formed in BF16 before
                       cos/sin (instead of F32)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tool" / "scripts"))
import generate_causal_lm_reference as generator  # noqa: E402

DEFECTS = ["none", "silu-twice-rounded", "rmsnorm-bf16", "softmax-bf16", "rope-angles-bf16"]


def measure(actual, truth):
    """Max |Δ| (first position) and mean |Δ| in F64."""
    delta = np.abs(actual.astype(np.float64) - truth)
    flat = int(delta.argmax())
    return {"max": float(delta.max()), "position": flat // truth.shape[1], "mean": float(delta.sum() / delta.size)}


def bf16_rms_forward(norm):
    """An RMSNorm forward that keeps the input dtype throughout (no F32 upcast)."""
    original = norm.forward

    def forward(hidden_states):
        """Mean square and rsqrt in the input dtype; same weight convention as the original."""
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + norm.variance_epsilon)
        return norm.weight * normed

    return original, forward


class TwiceRoundedSilu(torch.nn.Module):
    """x * sigmoid(x) in the input dtype: the sigmoid is rounded before the product."""

    def forward(self, x):
        """Two roundings in half precision, one in F32."""
        return x * torch.sigmoid(x)


def apply(model, defect):
    """Injects `defect` into `model`; returns a function that undoes it."""
    undo = []
    module = sys.modules[type(model.model).__module__]
    if defect == "silu-twice-rounded":
        for mlp in (m for m in model.modules() if hasattr(m, "act_fn") and hasattr(m, "gate_proj")):
            original = mlp.act_fn
            mlp.act_fn = TwiceRoundedSilu()
            undo.append(lambda mlp=mlp, original=original: setattr(mlp, "act_fn", original))
    elif defect == "rmsnorm-bf16":
        for norm in (m for m in model.modules() if type(m).__name__.endswith("RMSNorm")):
            original, forward = bf16_rms_forward(norm)
            norm.forward = forward
            undo.append(lambda norm=norm: delattr(norm, "forward"))
    elif defect == "softmax-bf16":
        original = module.eager_attention_forward

        def eager(attn_module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            """The module's eager attention with the softmax in the query dtype."""
            key_states = module.repeat_kv(key, attn_module.num_key_value_groups)
            value_states = module.repeat_kv(value, attn_module.num_key_value_groups)
            weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                weights = weights + attention_mask
            # PyTorch's own BF16 softmax already computes in F32 and rounds once, so the defect is
            # spelled out: exponentials, their sum, and the quotient each rounded to the input dtype.
            exponentials = torch.exp(weights - weights.amax(dim=-1, keepdim=True))
            weights = exponentials / exponentials.sum(dim=-1, keepdim=True)
            output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
            return output, weights

        module.eager_attention_forward = eager
        undo.append(lambda: setattr(module, "eager_attention_forward", original))
    elif defect == "rope-angles-bf16":
        rotary = model.model.rotary_emb

        def forward(x, position_ids):
            """cos/sin from angles formed in BF16."""
            inv = rotary.inv_freq.to(torch.bfloat16)
            freqs = (position_ids.to(torch.bfloat16)[:, :, None] * inv[None, None, :]).float()
            emb = torch.cat((freqs, freqs), dim=-1)
            return (emb.cos() * rotary.attention_scaling).to(x.dtype), (emb.sin() * rotary.attention_scaling).to(x.dtype)

        rotary.forward = forward
        undo.append(lambda: delattr(rotary, "forward"))
    return lambda: [step() for step in reversed(undo)]


def main():
    model_dir, truth_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    manifest = json.loads((truth_dir / "reference.json").read_text())
    vocabulary = manifest["vocabulary_size"]
    truth = [np.fromfile(truth_dir / p["logits_file"], dtype="<f4").astype(np.float64).reshape(len(p["token_ids"]), vocabulary)
             for p in manifest["prompts"]]
    model = generator.load_causal_lm(model_dir, "bfloat16")
    result = {"truth": str(truth_dir), "truth_manifest_sha256": generator.sha256_file(truth_dir / "reference.json"),
              "tool_sha256": generator.sha256_file(Path(generator.__file__)),
              "script_sha256": generator.sha256_file(Path(__file__).resolve()), "model_class": type(model).__name__,
              "defects": {}}
    for defect in DEFECTS:
        undo = apply(model, defect)
        try:
            rows = []
            for index, prompt in enumerate(manifest["prompts"]):
                logits = generator.prompt_logits(model, prompt["token_ids"]).numpy()
                stats = measure(logits, truth[index])
                if defect == "none":
                    stats["sha256_f32le"] = __import__("hashlib").sha256(generator.tensor_f32le(torch.from_numpy(logits))).hexdigest()
                rows.append(stats)
                print(f"{defect:20s} prompt {index}: max {stats['max']:.6g} at {stats['position']} mean {stats['mean']:.6g}", flush=True)
        finally:
            undo()
        result["defects"][defect] = rows
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
