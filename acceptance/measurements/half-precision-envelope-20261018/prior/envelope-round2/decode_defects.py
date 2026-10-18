"""Scratch sensitivity check: how large injected decode-path defects make the self-consistency gap (not committed).

Round 2 copy: prompts, greedy tokens and the longest prompt's truth logits now
come from the truth bundle (no F32 model is loaded), and the tool is round 2's
frozen copy of the extended generator.

Usage: python3 -I decode_defects.py MODEL_DIR TRUTH_BUNDLE OUT_JSON [GRAPH...]   (GRAPH = device-attention, default cpu-sdpa mps-eager)

The official Transformers model (BF16) decodes prompt 0 followed by the truth
bundle's greedy continuation, and each decode step is compared with one clean
stateless BF16 pass of the same graph at the same position (the crate gate's
decode-vs-stateless gap). Defects are injected into the cached decode path
only:
  none            : no defect (the graph's own gap)
  rope+1          : every decode step at position + 1 (off-by-one rotary offset)
  skip-newest-key : the newest token's own key/value is masked out of its decode step
  skip-prev-key   : the previous token's key/value is masked out (a lost cache append)
  kv-e5m2 / kv-e4m3: the whole KV cache rounded to FP8 after every step (a precision defect)
  rope-bf16       : decode-step rotary angles position * inv_freq rounded to BF16 (a precision defect)
It also decodes the longest prompt after a prefill of its first quarter, as the
gate's second decode check does, and reports that check's max and mean
deviation from the local F32 eager evaluation (the truth's code path).
"""

import json
import sys
from pathlib import Path

import torch
import transformers

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tool" / "scripts"))
import generate_causal_lm_reference as generator  # noqa: E402

DEFECTS = ["none", "rope+1", "skip-newest-key", "skip-prev-key", "kv-e5m2", "kv-e4m3", "rope-bf16"]


def round_cache(past, dtype):
    """Rounds every cached key and value to `dtype` and back, in place."""
    for layer in past.layers:
        device = layer.keys.device
        layer.keys = layer.keys.cpu().to(dtype).to(layer.keys.dtype).to(device)
        layer.values = layer.values.cpu().to(dtype).to(layer.values.dtype).to(device)


class Bf16Angles:
    """Replaces the rotary module's forward for single-position calls by BF16-rounded angles."""

    def __init__(self, rotary):
        self.rotary = rotary
        self.original = rotary.forward

    def __call__(self, x, position_ids):
        if position_ids.shape[-1] != 1:
            return self.original(x, position_ids)
        inv = self.rotary.inv_freq.to(torch.bfloat16).to(x.device)
        freqs = (position_ids.to(torch.bfloat16)[:, :, None] * inv[None, None, :]).float()
        emb = torch.cat((freqs, freqs), dim=-1)
        return ((emb.cos() * self.rotary.attention_scaling).to(x.dtype),
                (emb.sin() * self.rotary.attention_scaling).to(x.dtype))


def step(model, past, token, defect, device):
    """One cached decode step with the defect applied; returns the logits row (F64, CPU)."""
    length = past.get_seq_length()
    kwargs = {}
    if defect == "rope+1":
        kwargs["position_ids"] = torch.tensor([[length + 1]], device=device)
    if defect in ("skip-newest-key", "skip-prev-key"):
        mask = torch.ones(1, length + 1, dtype=torch.long, device=device)
        mask[0, -1 if defect == "skip-newest-key" else -2] = 0
        kwargs["attention_mask"] = mask
        kwargs["position_ids"] = torch.tensor([[length]], device=device)
    output = model(input_ids=torch.tensor([[token]], device=device), past_key_values=past, use_cache=True, **kwargs)
    if defect == "kv-e5m2":
        round_cache(output.past_key_values, torch.float8_e5m2)
    if defect == "kv-e4m3":
        round_cache(output.past_key_values, torch.float8_e4m3fn)
    return output.logits[0, -1].float().cpu().double(), output.past_key_values


def decode(model, prompt, tokens, defect, device):
    """Prefill (clean, but cache-rounded for KV defects) then defective decode steps."""
    with torch.no_grad():
        output = model(input_ids=torch.tensor([prompt], device=device), use_cache=True)
        past = output.past_key_values
        if defect == "kv-e5m2":
            round_cache(past, torch.float8_e5m2)
        if defect == "kv-e4m3":
            round_cache(past, torch.float8_e4m3fn)
        rows = []
        for token in tokens:
            row, past = step(model, past, token, defect, device)
            rows.append(row)
    return torch.stack(rows), output


def main():
    model_dir, truth_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    graphs = sys.argv[4:] or ["cpu-sdpa", "mps-eager"]
    manifest = json.loads((truth_dir / "reference.json").read_text())
    prompts = [p["token_ids"] for p in manifest["prompts"]]
    greedy = [s["token"] for s in manifest["greedy"]["steps"]]
    longest = max(range(len(prompts)), key=lambda i: len(prompts[i]))
    vocabulary = manifest["vocabulary_size"]
    import numpy as np
    truth_longest = torch.from_numpy(np.fromfile(truth_dir / manifest["prompts"][longest]["logits_file"], dtype="<f4")
                                     .astype(np.float64).reshape(len(prompts[longest]), vocabulary))
    result = {"truth": str(truth_dir), "truth_manifest_sha256": generator.sha256_file(truth_dir / "reference.json"),
              "tool_sha256": generator.sha256_file(Path(generator.__file__)), "prompt0": prompts[0], "greedy": greedy,
              "longest": longest, "graphs": {}}
    for graph in graphs:
        device, attention = graph.split("-")
        model = generator.load_causal_lm(model_dir, "bfloat16", attention=attention, device=device)
        stateless = generator.prompt_logits(model, prompts[0] + greedy)[len(prompts[0]):].double()
        rows = {}
        for defect in DEFECTS:
            if defect.startswith("kv-e") and device == "mps":
                pass  # rounding goes through the CPU, which works for MPS tensors too
            patch = None
            if defect == "rope-bf16":
                rotary = model.model.rotary_emb
                patch = Bf16Angles(rotary)
                rotary.forward = patch
            try:
                cached, _ = decode(model, prompts[0], greedy, defect, device)
                gaps = (cached - stateless).abs().max(dim=1).values
                # The gate's decoded-prompt check: first quarter prefilled, the rest decoded with the defect.
                tokens = prompts[longest]
                prefill = max(len(tokens) // 4, 1)
                with torch.no_grad():
                    first = model(input_ids=torch.tensor([tokens[:prefill]], device=device), use_cache=True)
                    past = first.past_key_values
                    if defect == "kv-e5m2":
                        round_cache(past, torch.float8_e5m2)
                    if defect == "kv-e4m3":
                        round_cache(past, torch.float8_e4m3fn)
                    assembled = [first.logits[0].float().cpu().double()]
                    for token in tokens[prefill:]:
                        row, past = step(model, past, token, defect, device)
                        assembled.append(row[None])
                assembled = torch.cat(assembled)
                delta = (assembled - truth_longest).abs()
            finally:
                if patch is not None:
                    del model.model.rotary_emb.forward
            rows[defect] = {"gap_max": float(gaps.max()), "gap_steps_above_0.5": int((gaps > 0.5).sum()),
                            "gap_median": float(gaps.median()), "per_step": [round(float(g), 5) for g in gaps],
                            "decoded_prompt_max": float(delta.max()), "decoded_prompt_mean": float(delta.mean())}
            print(f"{graph} {defect:16s} decode-vs-stateless max {gaps.max():.4f} (steps > 0.5: {int((gaps > 0.5).sum())}, "
                  f"median {gaps.median():.4f}); decoded prompt {longest} vs F32 max {delta.max():.4f} mean {delta.mean():.5f}",
                  flush=True)
        result["graphs"][graph] = rows
        del model
        if device == "mps":
            torch.mps.empty_cache()
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
