"""Scratch: reproduce round 2's BF16 leave-one-out ratios with the committed derivation script (Qwen2-7B on its F32 decode-row proxy)."""
import importlib.util, json, sys
from pathlib import Path
WT = Path(sys.argv[1]); R = Path("/Volumes/T9/transformers-references")
spec = importlib.util.spec_from_file_location("d", WT / "scripts/derive_half_precision_envelope.py"); d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
truths = ["SmolLM-135M-1d461723", "Qwen3-4B-531c80e2", "SmolLM3-3B-1fa6fa28", "Qwen2-7B-453ed157-f64", "Llama-3.1-8B-d04e592b", "gemma-3-4b-it-093f9f38", "gemma-3-4b-it-093f9f38-long"]
original = d.validate_decode_truth
def proxy(truth, decode_truth):
    if truth.dtype == "float64":  # round 2 had only the F32 rows for Qwen2-7B
        return d.validate_decode_record(truth, decode_truth)
    return original(truth, decode_truth)
d.validate_decode_truth = proxy
cases = [{"name": t, "truth": str(R / t), "decode_truth": str(R / f"{t}-f32-cpu-eager"), "graphs": [str(R / f"{t}-bf16-{dev}-{att}") for dev, att in d.GRAPHS]} for t in truths]
result = d.derive(cases, "bfloat16")
Path("round2-bf16.json").write_text(json.dumps(result, indent=1))
print(d.render(result))
for stat in ("max", "mean"):
    print(stat, [(round(e["ratio"], 4), e["case"], e["kind"], e["prompt"], e["graph"]) for e in result["slacks"][stat]["largest"][:4]])
