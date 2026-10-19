"""Scratch check (round 2, copied from round 1 with two decode-record checks added): new graph bundles against the existing bundles and the previous round's diagnostics (not committed).

Usage: python3 -I verify_bundles.py KEY

For one checkpoint KEY (see make_bundles.CHECKPOINTS):
* every new bundle's logits files match their recorded SHA-256s and shapes;
* prompts (text and token IDs), vocabulary, model class, config and weight
  digests, and versions equal the truth's, and the decode record teacher-forces
  the truth's greedy tokens from the truth's exact manifest;
* bf16-cpu-eager reproduces the committed-script BF16 bundle (<truth>-bf16)
  byte for byte (prompt logits and greedy steps), and f32-cpu-eager the F32
  truth (when the truth is F32);
* where the previous round's official_variants.py / official_decode.py
  diagnostics saved logits for the same graph, they are compared byte for byte.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_bundles  # noqa: E402

REFERENCES = make_bundles.REFERENCES
DIAG = Path("/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/mac-validation")
#: key -> (official_variants data dir, official_decode data dir) of the previous round
PREVIOUS = {
    "SmolLM-135M": ("llama-family/diagnostics/data/SmolLM-135M-official", "llama-family/diagnostics/data/SmolLM-135M-official-decode"),
    "Qwen3-4B": ("llama-family/diagnostics/data/Qwen3-4B-official", "llama-family/diagnostics/data/Qwen3-4B-official-decode"),
    "SmolLM3-3B": ("llama-family/diagnostics/data/SmolLM3-3B-official", "llama-family/diagnostics/data/SmolLM3-3B-official-decode"),
    "Qwen2-7B": ("llama-family/diagnostics/data/Qwen2-7B-f64-official", "llama-family/diagnostics/data/Qwen2-7B-f64-official-decode"),
    "Llama-3.1-8B": ("llama-family/diagnostics/data/Llama-3.1-8B-official", "llama-family/diagnostics/data/Llama-3.1-8B-official-decode"),
    "gemma-3-4b-it": ("gemma/diagnostics/data/official", "gemma/diagnostics/data/official-decode"),
}
GRAPH = {"bf16-cpu-eager": "cpu-bfloat16-eager", "bf16-cpu-sdpa": "cpu-bfloat16-sdpa",
         "bf16-mps-eager": "mps-bfloat16-eager", "bf16-mps-sdpa": "mps-bfloat16-sdpa",
         "f32-cpu-eager": "cpu-float32-eager"}


def sha(path):
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    key = sys.argv[1]
    truth_name, _ = make_bundles.CHECKPOINTS[key]
    truth_dir = REFERENCES / truth_name
    truth = json.loads((truth_dir / "reference.json").read_text())
    truth_sha = sha(truth_dir / "reference.json")
    failures = 0

    def check(condition, message):
        nonlocal failures
        print(("ok   " if condition else "FAIL ") + message)
        failures += not condition

    old_bf16 = REFERENCES / f"{truth_name}-bf16"
    for variant in make_bundles.VARIANTS:
        root = REFERENCES / f"{truth_name}-{variant}"
        if not root.exists():
            print(f"missing {root}")
            failures += 1
            continue
        manifest = json.loads((root / "reference.json").read_text())
        p, tp = manifest["provenance"], truth["provenance"]
        dtype, device, attention = make_bundles.VARIANTS[variant]
        check((p["dtype"], p["device"], p["attention_implementation"]) == (dtype, device, attention),
              f"{variant}: provenance graph {p['dtype']}/{p['device']}/{p['attention_implementation']}")
        check(p["script_sha256"] == make_bundles.TOOL_SHA256, f"{variant}: script sha256 {p['script_sha256'][:12]}")
        for field in ("config_sha256", "weights_sha256", "transformers", "torch", "repository", "revision"):
            check(p[field] == tp[field], f"{variant}: provenance.{field} equals the truth's")
        check((manifest["model_type"], manifest["model_class"], manifest["vocabulary_size"]) ==
              (truth["model_type"], truth["model_class"], truth["vocabulary_size"]), f"{variant}: model and vocabulary")
        check([(x["text"], x["token_ids"]) for x in manifest["prompts"]] ==
              [(x["text"], x["token_ids"]) for x in truth["prompts"]], f"{variant}: prompts and token IDs")
        check(len(manifest["greedy"]["steps"]) == len(truth["greedy"]["steps"]), f"{variant}: greedy length")
        vocabulary = manifest["vocabulary_size"]
        for prompt in manifest["prompts"]:
            path = root / prompt["logits_file"]
            values = np.fromfile(path, dtype="<f4")
            check(sha(path) == prompt["logits_sha256"] and values.size == prompt["logits_shape"][0] * vocabulary
                  and prompt["logits_shape"] == [len(prompt["token_ids"]), vocabulary] and np.isfinite(values).all(),
                  f"{variant}: {prompt['logits_file']} digest, shape, finite")
        record = manifest["decode_consistency"]
        check(record["tokens_source"]["manifest_sha256"] == truth_sha, f"{variant}: decode tokens from the truth manifest")
        check([s["fed_token"] for s in record["steps"]] == [s["token"] for s in truth["greedy"]["steps"]],
              f"{variant}: decode feeds the truth's greedy tokens")
        steps = len(record["steps"])
        check(record["first_position"] == len(truth["prompts"][0]["token_ids"]) and record["prompt_index"] == 0
              and record["logits_shape"] == [steps, vocabulary], f"{variant}: decode record starts after prompt 0, shape")
        check(record["max_abs_gap"] == max(s["max_abs_gap"] for s in record["steps"]), f"{variant}: decode max equals the step maximum")
        arrays = {}
        for side in ("cached", "stateless"):
            path = root / record[f"{side}_logits_file"]
            arrays[side] = np.fromfile(path, dtype="<f4").reshape(steps, vocabulary)
            check(sha(path) == record[f"{side}_logits_sha256"] and np.isfinite(arrays[side]).all(),
                  f"{variant}: decode {side} digest, finite")
        gaps = np.abs(arrays["cached"].astype(np.float64) - arrays["stateless"]).max(axis=1)
        check(np.array_equal(gaps, np.array([s["max_abs_gap"] for s in record["steps"]])),
              f"{variant}: recorded gaps recomputed (max {gaps.max():.6g})")
        # Reproduction of existing bundles.
        if variant == "bf16-cpu-eager" and old_bf16.exists():
            old = json.loads((old_bf16 / "reference.json").read_text())
            check([x["logits_sha256"] for x in manifest["prompts"]] == [x["logits_sha256"] for x in old["prompts"]],
                  f"{variant}: prompt logits byte-identical to {old_bf16.name}")
            check(manifest["greedy"] == old["greedy"], f"{variant}: greedy identical to {old_bf16.name}")
        if variant == "f32-cpu-eager" and tp["dtype"] == "float32":
            check([x["logits_sha256"] for x in manifest["prompts"]] == [x["logits_sha256"] for x in truth["prompts"]],
                  f"{variant}: prompt logits byte-identical to the truth {truth_name}")
            check(manifest["greedy"] == truth["greedy"], f"{variant}: greedy identical to the truth")
        if variant == "f32-cpu-eager" and tp["dtype"] != "float32":
            check([s["token"] for s in manifest["greedy"]["steps"]] == [s["token"] for s in truth["greedy"]["steps"]],
                  f"{variant}: F32 greedy tokens equal the {tp['dtype']} truth's")
        # Previous round's diagnostics, when they saved this graph.
        if key in PREVIOUS:
            variants_dir, decode_dir = (DIAG / part for part in PREVIOUS[key])
            graph = GRAPH[variant]
            for index, prompt in enumerate(manifest["prompts"]):
                old = variants_dir / graph / f"prompt{index}" / "logits.bin"
                if old.exists():
                    same = np.array_equal(np.fromfile(old, dtype="<f4"), np.fromfile(root / prompt["logits_file"], dtype="<f4"))
                    check(same, f"{variant}: prompt {index} identical to the previous diagnostic {old.parent.parent.name}")
            for side in ("cached", "stateless"):
                old = decode_dir / graph / f"{side}.bin"
                if old.exists():
                    same = np.array_equal(np.fromfile(old, dtype="<f4").reshape(steps, vocabulary), arrays[side])
                    check(same, f"{variant}: decode {side} identical to the previous diagnostic")
    print(f"{key}: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
