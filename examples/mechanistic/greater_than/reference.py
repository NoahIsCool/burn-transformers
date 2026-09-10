#!/usr/bin/env python3
"""Run the frozen Greater-Than experiment with independent PyTorch hooks.

    python3 examples/mechanistic/greater_than/reference.py

Requires torch 2.10.0, transformers 5.2.0, safetensors, NumPy, and tokenizers.
First run prepare.py. This is inference only: no fitting or prompt selection.
Model computation stays F32; probability analysis uses the full vocabulary in
F64. Every frozen condition is evaluated on every prompt pair.
"""

import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import safetensors
from safetensors.torch import save_file
import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel

from prepare import DEFAULT_ROOT, HERE, describe_file, sha256_file, verify_preparation, write_json


def probability_metrics(logits, numeric_ids, suffixes):
    """Do not renormalize the 100 year tokens: other vocabulary mass matters."""
    if logits.dtype != torch.float32 or logits.ndim != 2:
        raise ValueError("Expected [batch, vocabulary] unchanged F32 logits")
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite model logits")
    probabilities = torch.softmax(logits.double(), dim=-1)[:, numeric_ids]
    rows = []
    for values, suffix in zip(probabilities.tolist(), suffixes):
        rows.append({"pd": math.fsum(values[suffix + 1:]) - math.fsum(values[:suffix + 1]),
                     "sharpness": values[suffix + 1] - values[suffix - 1],
                     "numeric_mass": math.fsum(values), "probabilities": values})
    return rows


def analytic_metrics(logits, numeric_ids, suffix):
    """Independent NumPy exp + Python fsum check over every vocabulary token."""
    raw = np.asarray(logits, dtype=np.float64)
    weights = np.exp(raw - raw.max())
    denominator = math.fsum(weights.tolist())
    values = [float(weights[token]) / denominator for token in numeric_ids]
    return {"pd": math.fsum(values[suffix + 1:]) - math.fsum(values[:suffix + 1]),
            "sharpness": values[suffix + 1] - values[suffix - 1],
            "numeric_mass": math.fsum(values), "probabilities": values}


@contextmanager
def capture_readout(model, position):
    """Observe the actual post-projection MLP tensors and pre-final-LN residual."""
    captures = {}
    handles = []

    def mlp_hook(layer):
        def observe(_module, _inputs, output):
            captures[layer] = output[:, position:position + 1].detach().clone()
        return observe

    def residual_hook(_module, inputs):
        captures["residual"] = inputs[0][:, position:position + 1].detach().clone()

    try:
        for layer, block in enumerate(model.transformer.h):
            handles.append(block.mlp.register_forward_hook(mlp_hook(layer)))
        handles.append(model.transformer.ln_f.register_forward_pre_hook(residual_hook))
        yield captures
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def interventions(model, edits, position):
    """Install ordered callbacks, then remove them even if a forward fails.

    Each edit is (layer, replacement). None means zero; a tensor replaces only
    the selected token. Two edits on one layer run in their registration order.
    """
    handles = []

    def callback(replacement):
        def edit(_module, _inputs, output):
            changed = output.clone()
            target = changed[:, position:position + 1]
            if replacement is None:
                target.zero_()
            else:
                if replacement.shape != target.shape or replacement.dtype != output.dtype:
                    raise ValueError("Replacement shape/dtype does not match selected MLP output")
                target.copy_(replacement)
            return changed
        return edit

    try:
        for layer, replacement in edits:
            handles.append(model.transformer.h[layer].mlp.register_forward_hook(callback(replacement)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def forward(model, ids):
    return model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, 0].detach()


def run_condition(model, condition, clean_ids, corrupt_ids, clean, corrupt, manifest):
    """Express node substitutions separately from direct readout substitutions."""
    position = manifest["readout_position"]
    if condition == "node_patch_all_late":
        edits = [(layer, clean[layer]) for layer in manifest["late_layers"]]
        destination = corrupt_ids
    elif condition == "empty_clean":
        edits, destination = [], clean_ids
    elif condition.startswith("direct_"):
        layer = int(condition.rsplit("_", 1)[1])
        residual = clean["residual"] - clean[layer] + corrupt[layer]
        return model.lm_head(model.transformer.ln_f(residual))[:, 0].detach()
    else:
        operation, layer_text = condition.rsplit("_", 1)
        layer = int(layer_text)
        destination = clean_ids
        if operation == "node_patch":
            edits, destination = [(layer, clean[layer])], corrupt_ids
        elif operation == "node_reverse":
            edits = [(layer, corrupt[layer])]
        elif operation == "zero":
            edits = [(layer, None)]
        elif operation == "identity":
            edits = [(layer, clean[layer])]
        elif operation == "restore":
            edits = [(layer, None), (layer, clean[layer])]
        else:
            raise ValueError(f"Unknown condition: {condition}")
    with interventions(model, edits, position):
        return forward(model, destination)


def load_model(root, manifest):
    """Explicitly select the author's exact GELU graph with unchanged weights."""
    config = GPT2Config.from_pretrained(root / "model", local_files_only=True)
    if (config.n_layer, config.n_head, config.n_embd, config.vocab_size,
        config.layer_norm_epsilon, config.activation_function) != (12, 12, 768, 50257, 1e-5, "gelu_new"):
        raise ValueError("Unexpected pinned GPT-2 configuration")
    config.activation_function = manifest["activation"]
    model = GPT2LMHeadModel.from_pretrained(
        root / "model", config=config, local_files_only=True, attn_implementation="eager",
    ).float().eval()
    if model.config._attn_implementation != "eager" or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Reference did not select eager F32 arithmetic")
    if model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr():
        raise ValueError("GPT-2 readout is not tied to token embeddings")
    return model


@torch.inference_mode()
def evaluate(model, manifest, fixture):
    entries = fixture["entries"]
    conditions = manifest["conditions"]
    oracle_indices = manifest["oracle_indices"]
    oracle_lookup = {index: position for position, index in enumerate(oracle_indices)}
    oracle = {f"logits.{condition}": torch.empty(len(oracle_indices), manifest["vocabulary_size"])
              for condition in conditions}
    for kind in ("clean", "corrupt"):
        for layer in manifest["direct_layers"]:
            oracle[f"mlp.{kind}.{layer}"] = torch.empty(len(entries), 1, 768)
        oracle[f"residual.{kind}"] = torch.empty(len(entries), 1, 768)
    controls = {condition: {"maximum_logits_error": 0.0, "passed": True}
                for condition in conditions if condition in ("empty_clean", "observe_clean")
                or condition.startswith(("identity_", "restore_"))}
    analytic_error = {"pd": 0.0, "sharpness": 0.0, "numeric_mass": 0.0, "probabilities": 0.0}
    rows = []
    started = time.monotonic()
    for start in range(0, len(entries), manifest["batch_size"]):
        end = min(start + manifest["batch_size"], len(entries))
        batch = entries[start:end]
        clean_ids = torch.tensor([entry["clean_ids"] for entry in batch], dtype=torch.int64)
        corrupt_ids = torch.tensor([entry["corrupt_ids"] for entry in batch], dtype=torch.int64)
        suffixes = [entry["year"] % 100 for entry in batch]
        clean_logits = forward(model, clean_ids)
        with capture_readout(model, manifest["readout_position"]) as clean:
            observed_logits = forward(model, clean_ids)
        with capture_readout(model, manifest["readout_position"]) as corrupt:
            corrupt_logits = forward(model, corrupt_ids)
        for kind, captures in (("clean", clean), ("corrupt", corrupt)):
            for layer in manifest["direct_layers"]:
                oracle[f"mlp.{kind}.{layer}"][start:end].copy_(captures[layer])
            oracle[f"residual.{kind}"][start:end].copy_(captures["residual"])
        for condition in conditions:
            if condition == "clean":
                logits = clean_logits
            elif condition == "corrupt":
                logits = corrupt_logits
            elif condition == "observe_clean":
                logits = observed_logits
            else:
                logits = run_condition(model, condition, clean_ids, corrupt_ids, clean, corrupt, manifest)
            measured = probability_metrics(logits, manifest["numeric_token_ids"], suffixes)
            if condition in controls:
                error = float((logits - clean_logits).abs().max())
                controls[condition]["maximum_logits_error"] = max(controls[condition]["maximum_logits_error"], error)
                controls[condition]["passed"] &= error <= manifest["numerical_tolerances"]["controls_logits_absolute"]
            for local, measurement in enumerate(measured):
                index = start + local
                rows.append({"condition": condition, "index": index, **measurement})
                if index in oracle_lookup:
                    oracle[f"logits.{condition}"][oracle_lookup[index]].copy_(logits[local])
                    independent = analytic_metrics(logits[local].numpy(), manifest["numeric_token_ids"], suffixes[local])
                    for key in ("pd", "sharpness", "numeric_mass"):
                        analytic_error[key] = max(analytic_error[key], abs(measurement[key] - independent[key]))
                    analytic_error["probabilities"] = max(analytic_error["probabilities"],
                        max(abs(a - b) for a, b in zip(measurement["probabilities"], independent["probabilities"])))
        print(f"Reference: {end}/{len(entries)} pairs, all {len(conditions)} conditions; {time.monotonic() - started:.1f} s", flush=True)
    ordering = {name: index for index, name in enumerate(conditions)}
    rows.sort(key=lambda row: (ordering[row["condition"]], row["index"]))
    if len(rows) != len(entries) * len(conditions):
        raise ValueError("Reference row inventory is incomplete")
    validation = {"controls": controls, "independent_analysis": {
        "method": "NumPy F64 exp of max-shifted unchanged F32 logits; Python math.fsum denominator over all 50257 tokens",
        "conditions": len(conditions), "prompts_per_condition": len(oracle_indices),
        "maximum_absolute_errors": analytic_error, "tolerance": 1e-12,
        "passed": all(error <= 1e-12 for error in analytic_error.values()),
    }}
    return rows, oracle, validation, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", "--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if torch.__version__.split("+")[0] != "2.10.0" or transformers.__version__ != "5.2.0":
        raise ValueError("This numerical reference pins torch 2.10.0 and transformers 5.2.0")
    root = args.directory.resolve()
    manifest, fixture, _preparation = verify_preparation(root)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    transformers.utils.logging.disable_progress_bar()
    model = load_model(root, manifest)
    rows, oracle, validation, seconds = evaluate(model, manifest, fixture)
    oracle_path = root / "reference.safetensors"
    temporary = root / ".reference.safetensors.tmp"
    try:
        save_file(oracle, temporary, metadata={"format": "pt", "protocol": manifest["protocol"],
                  "manifest_sha256": sha256_file(HERE / "manifest.json"),
                  "fixture_sha256": sha256_file(root / "fixture.json"),
                  "reference_sha256": sha256_file(Path(__file__))})
        temporary.replace(oracle_path)
    finally:
        temporary.unlink(missing_ok=True)
    result = {
        "schema_version": 1, "protocol": manifest["protocol"],
        "provenance": {"manifest_sha256": sha256_file(HERE / "manifest.json"),
                       "fixture_sha256": sha256_file(root / "fixture.json"),
                       "preparation_sha256": sha256_file(root / "preparation.json"),
                       "prepare_sha256": sha256_file(HERE / "prepare.py"),
                       "reference_sha256": sha256_file(Path(__file__)),
                       "versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                                    "transformers": transformers.__version__, "numpy": np.__version__,
                                    "safetensors": safetensors.__version__},
                       "host": {"system": platform.system(), "machine": platform.machine()},
                       "dtype": "F32", "analysis_dtype": "F64", "activation": manifest["activation"],
                       "activation_override": "Source config gelu_new -> author graph gelu; weights unchanged",
                       "attention": "eager", "evaluation_mode": True, "training": False,
                       "threads": 1, "interop_threads": 1, "batch_size": manifest["batch_size"]},
        "oracle_indices": manifest["oracle_indices"], "rows": rows,
        **validation, "elapsed_seconds": seconds,
        "artifacts": {"reference.safetensors": describe_file(oracle_path, root)},
    }
    write_json(root / "reference.json", result)
    write_json(root / "reference_artifacts.json", {
        "protocol": manifest["protocol"],
        "artifacts": {name: describe_file(root / name, root) for name in ("reference.json", "reference.safetensors")},
    })
    controls_pass = all(control["passed"] for control in validation["controls"].values())
    print(f"Wrote {len(rows)} reference rows and {len(oracle)} oracle tensors to {root}")
    if not controls_pass or not validation["independent_analysis"]["passed"]:
        raise ValueError("Reference controls failed; retained measured artifacts without changing tolerances")
    print("All identity/restore/observation controls and independent full-vocabulary analysis checks passed")


if __name__ == "__main__":
    main()
