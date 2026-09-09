#!/usr/bin/env python3
"""Choose the first numerically stable Qwen acceptance prompt using only PyTorch.

The candidate order is fixed before execution. No Burn output participates in
selection. Every candidate must satisfy the existing full-vector cosine, exact
cached/teacher top-1, 32 non-EOS decisions, and greater-than-two-ULP margin gates.
The create-only report records every attempted candidate and all source hashes;
it is selection evidence, not an accepted full-vector oracle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import generate_qwen35_08b_reference as reference


CANDIDATES = (
    "Explain in four sentences why caching attention keys and values speeds up autoregressive text generation.",
    "Describe the water cycle in four complete sentences.",
    "Explain how binary search works in four complete sentences.",
    "Explain the difference between RAM and persistent storage in four sentences.",
    "Explain in four sentences why a cup of hot tea cools down in a room.",
    "Describe in four sentences how plants use sunlight to grow.",
    "Explain in four sentences what happens when a web browser requests a page.",
    "Describe how a refrigerator keeps food cold in four complete sentences.",
    "Explain why the seasons change in four complete sentences.",
    "Describe the purpose of unit tests in four complete sentences.",
    "Explain how a bicycle's gears help a rider climb a hill in four sentences.",
    "Describe how a compiler turns source code into an executable in four sentences.",
)
BUNDLE_NAME = "qwen3.5-0.8b-2fc06364-prompt-selection.bundle"
REPORT_NAME = "selection.json"


def evidence_passes(record: dict[str, Any]) -> bool:
    """Apply the original numerical gates, independently of historical outputs."""

    return (
        len(record["token_ids"]) == reference.GREEDY_TOKENS
        and record["token_ids"] == record["teacher_token_ids"]
        and not set(record["token_ids"]) & set(reference.EXPECTED_EOS_TOKEN_IDS)
        and len(record["decision_stability"]) == reference.GREEDY_TOKENS
        and all(item["stable_under_budget"] for item in record["decision_stability"])
        and record["prompt_cached_cosine"] >= reference.SELF_PARITY_COSINE
        and len(record["cached_teacher_cosines"]) == reference.GREEDY_TOKENS
        and all(value >= reference.SELF_PARITY_COSINE for value in record["cached_teacher_cosines"])
    )


def evaluate_candidate(torch: Any, model: Any, tokenizer: Any, user_prompt: str) -> dict[str, Any]:
    """Measure one fixed chat prompt with explicit masks and eager F16 inference."""

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not 1 <= len(tokens) <= 256:
        raise RuntimeError("candidate must contain between 1 and 256 prompt tokens")
    ids = torch.tensor([tokens], dtype=torch.long, device="mps")
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        stateless = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
        prompt_row = stateless.logits[0, -1].float().cpu().contiguous()
        cached = model(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
        rows, generated, decisions = [], [], []
        for step in range(reference.GREEDY_TOKENS):
            row = cached.logits[0, -1].float().cpu().contiguous()
            if cached.logits.dtype != torch.float16 or row.numel() != reference.VOCAB_SIZE:
                raise RuntimeError("candidate logits must be full-vocabulary F16")
            if not torch.isfinite(row).all().item():
                raise RuntimeError("candidate logits must be finite")
            values, indices = row.topk(2)
            token = int(indices[0])
            decisions.append(reference.COMMON.greedy_decision_stability(
                step=step, top1_token_id=token, top2_token_id=int(indices[1]),
                top1_logit=float(values[0]), top2_logit=float(values[1]),
                eos_token_ids=reference.EXPECTED_EOS_TOKEN_IDS,
            ))
            rows.append(row)
            generated.append(token)
            if step + 1 < reference.GREEDY_TOKENS:
                cached = model(
                    input_ids=torch.tensor([[token]], dtype=torch.long, device="mps"),
                    attention_mask=torch.ones((1, len(tokens) + step + 1), dtype=torch.long, device="mps"),
                    past_key_values=cached.past_key_values, use_cache=True, logits_to_keep=1,
                )
        teacher_ids = tokens + generated[:-1]
        teacher = model(
            input_ids=torch.tensor([teacher_ids], dtype=torch.long, device="mps"),
            attention_mask=torch.ones((1, len(teacher_ids)), dtype=torch.long, device="mps"),
            use_cache=False, logits_to_keep=0,
        )
        teacher_rows = teacher.logits[0, len(tokens) - 1:].float().cpu().contiguous()
        if list(teacher_rows.shape) != [reference.GREEDY_TOKENS, reference.VOCAB_SIZE]:
            raise RuntimeError("teacher rows must cover exactly the 32 generation positions")
        if not torch.isfinite(teacher_rows).all().item():
            raise RuntimeError("teacher logits must be finite")
        cosines = [reference.COMMON.tensor_cosine(torch, row, teacher_rows[i]) for i, row in enumerate(rows)]
        result = {
            "user_prompt": user_prompt, "prompt": prompt, "input_ids": tokens,
            "token_ids": generated, "teacher_token_ids": teacher_rows.argmax(-1).tolist(),
            "decision_stability": decisions,
            "prompt_cached_cosine": reference.COMMON.tensor_cosine(torch, prompt_row, rows[0]),
            "cached_teacher_cosines": cosines,
            "minimum_margin_to_epsilon_ratio": min(item["margin_to_epsilon_ratio"] for item in decisions),
            "minimum_cached_teacher_cosine": min(cosines),
            "decoded_text": tokenizer.decode(generated, skip_special_tokens=False),
        }
        result["passes"] = evidence_passes(result)
        return result


def main() -> int:
    """Authenticate the fixed environment, select once, and retain its provenance."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    model_dir, output_dir, _ = reference.ensure_external_paths(arguments.model_dir, arguments.output_dir)
    destination = output_dir / BUNDLE_NAME
    if os.path.lexists(destination):
        raise FileExistsError(f"selection report already exists: {destination}")
    script = Path(__file__).resolve()
    generator = reference.git_provenance(script)
    producer = reference.git_provenance(Path(reference.__file__).resolve())
    packages, distributions, distribution_digest = reference.COMMON.verify_packages()
    artifacts, artifact_digest, identities = reference.verify_checkpoint(model_dir)

    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5

    sources = reference.COMMON.verify_reference_sources(Path(transformers.__file__).resolve().parent)
    if any((modeling_qwen3_5.is_fast_path_available, modeling_qwen3_5.causal_conv1d_fn,
            modeling_qwen3_5.chunk_gated_delta_rule, modeling_qwen3_5.fused_recurrent_gated_delta_rule)):
        raise RuntimeError("selection requires pinned eager fallback kernels")
    if not torch.backends.mps.is_available():
        raise RuntimeError("selection requires Apple MPS")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.mps.set_per_process_memory_fraction(1.0)
    config = Qwen3_5Config.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False).text_config
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        model_dir, config=config, dtype=torch.float16, device_map={"": "mps"},
        attn_implementation="eager", local_files_only=True, trust_remote_code=False,
        use_safetensors=True, weights_only=True, low_cpu_mem_usage=True, output_loading_info=True,
    )
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(name):
            raise RuntimeError(f"official load reported {name}: {info[name]}")
    if sum(parameter.numel() for parameter in model.parameters()) != reference.EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("official parameter inventory changed")
    if any(parameter.dtype != torch.float16 for parameter in model.parameters()):
        raise RuntimeError("official parameter dtype changed")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    attempts, selected = [], None
    for index, candidate in enumerate(CANDIDATES):
        try:
            result = evaluate_candidate(torch, model, tokenizer, candidate)
        except RuntimeError as error:
            result = {"user_prompt": candidate, "passes": False, "error": str(error)}
        result["candidate_index"] = index
        attempts.append(result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        if result["passes"]:
            selected = index
            break
    if reference.COMMON.verify_packages() != (packages, distributions, distribution_digest):
        raise RuntimeError("package inventory changed during selection")
    if reference.verify_checkpoint(model_dir) != (artifacts, artifact_digest, identities):
        raise RuntimeError("checkpoint changed during selection")
    if reference.COMMON.verify_reference_sources(Path(transformers.__file__).resolve().parent) != sources:
        raise RuntimeError("reference source changed during selection")
    if reference.git_provenance(script) != generator or reference.git_provenance(Path(reference.__file__).resolve()) != producer:
        raise RuntimeError("selection source changed during execution")
    report = {
        "schema_version": 1, "kind": "official-reference-only-prompt-selection",
        "repository": reference.REPOSITORY, "revision": reference.REVISION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [sys.executable, str(script), *sys.argv[1:]],
        "policy": "first fixed candidate with all 32 margins >2 binary16 ULP, exact cached/teacher top1 and all cosines >=0.999; no Burn observations",
        "candidate_order": list(CANDIDATES), "attempts": attempts, "selected_index": selected,
        "generator": generator, "producer": producer, "packages": packages,
        "distributions": distributions, "distribution_manifest_sha256": distribution_digest,
        "reference_sources": sources, "artifacts": artifacts, "artifact_manifest_sha256": artifact_digest,
        "execution": {"dtype": "float16", "device": "mps", "attention": "eager",
                      "deterministic_algorithms": True, "seed": 0, "all_one_masks": True,
                      "hub_kernels": False, "float32_matmul_precision": "highest"},
    }
    payload = reference.COMMON.strict_json_bytes(report)
    staging = Path(tempfile.mkdtemp(prefix=f".{BUNDLE_NAME}.staging-", dir=output_dir))
    reference.COMMON.write_create_only(staging / REPORT_NAME, payload)
    reference.COMMON.fsync_directory(staging)
    reference.COMMON.rename_directory_create_only(staging, destination)
    reference.COMMON.fsync_directory(output_dir)
    print(f"selection report: {destination / REPORT_NAME} bytes={len(payload)} sha256={hashlib.sha256(payload).hexdigest()}")
    if selected is None:
        raise RuntimeError("no fixed candidate passed the unchanged acceptance gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
