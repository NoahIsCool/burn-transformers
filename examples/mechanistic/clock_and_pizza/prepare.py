#!/usr/bin/env python3
"""Prepare two pinned, small author checkpoints and independent CPU references.

Run from the repository root:
    python3 examples/mechanistic/clock_and_pizza/prepare.py

Dependencies: numpy, torch, safetensors. Only the two checkpoints and their two
JSON configurations are downloaded (1.71 MB total). All outputs stay in ignored
target/mechanistic/clock_and_pizza/. Rust itself requires no Python or network.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import safetensors
from safetensors.torch import load, save
import torch

from reference import calculate


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[2] / "target/mechanistic/clock_and_pizza"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(data, expected, description):
    if digest(data) != expected:
        raise ValueError(f"SHA-256 mismatch: {description}")


def atomic_write(path, data):
    """Expose a complete file only; clean up partial writes after failures."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_file(specification, commit, offline):
    path = OUTPUT / specification["filename"]
    if path.exists():
        data = path.read_bytes()
    else:
        if offline:
            raise ValueError(f"Missing cached source {path}; rerun without --offline")
        url = (
            "https://raw.githubusercontent.com/fjzzq2002/pizza/"
            f"{commit}/code/save/{specification['filename']}"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(specification["bytes"] + 1)
        if len(data) != specification["bytes"]:
            raise ValueError(f"Unexpected download length: {url}")
        verify(data, specification["sha256"], url)
        atomic_write(path, data)
    if len(data) != specification["bytes"]:
        raise ValueError(f"Unexpected cached source length: {path}")
    verify(data, specification["sha256"], str(path))
    return path


def validate_tensors(tensors, shapes, description):
    if set(tensors) != set(shapes):
        raise ValueError(f"Unexpected tensor names in {description}")
    for name, shape in shapes.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Not a tensor: {description}: {name}")
        if tensor.dtype != torch.float32 or list(tensor.shape) != shape:
            raise ValueError(f"Unexpected dtype/shape: {description}: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite tensor: {description}: {name}")


def validate_config(config, model):
    required = {
        "C": 59, "n_heads": 4, "d_model": 128, "n_layers": 1,
        "act_fn": "ReLU", "attn_coeff": model["attention_coefficient"],
        "runid": model["run_id"], "diff_vocab": 0, "eqn_sign": 0,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Unexpected author configuration {key}: {config.get(key)!r}")
    # The author JSON includes a lambda as text. It is data, never evaluated.


def check_paper_measurements(name, measured, checks):
    expected_q = checks["baseline_q"][name]
    actual_q = measured["baseline"]["distance_irrelevance_q"]
    if abs(actual_q - expected_q) > checks["q_absolute_tolerance"]:
        raise ValueError(f"{name} baseline q disagrees with the predeclared paper target")
    if name == "pizza":
        for circle, expected in checks["pizza_table2"].items():
            for metric in ("clock_fve", "pizza_fve"):
                error = abs(100 * measured[circle][metric] - expected[metric + "_percent"])
                if error > checks["fve_tolerance_percentage_points"]:
                    raise ValueError(f"{circle} {metric} disagrees with paper Table 2")


def cache_fingerprint(manifest_data):
    return {
        "manifest_sha256": digest(manifest_data),
        "prepare_sha256": digest(Path(__file__).read_bytes()),
        "reference_sha256": digest((HERE / "reference.py").read_bytes()),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "safetensors": safetensors.__version__,
    }


def validate_output_cache(metadata, fingerprint, shapes, model_names):
    if metadata.get("fingerprint") != fingerprint:
        return False
    if set(metadata.get("outputs", {})) != set(model_names):
        raise ValueError("Incomplete reference.json; rerun with --refresh")
    for name in model_names:
        path = OUTPUT / f"{name}.safetensors"
        specification = metadata["outputs"][name]
        if not path.exists():
            raise ValueError(f"Missing derived output {path}; rerun with --refresh")
        data = path.read_bytes()
        verify(data, specification["sha256"], f"{path}; rerun with --refresh to rebuild")
        if len(data) != specification["bytes"]:
            raise ValueError(f"Unexpected output length: {path}")
        validate_tensors(load(data), shapes, str(path))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="use verified cached sources only")
    parser.add_argument("--refresh", action="store_true", help="rebuild derived files from verified sources")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest_data = (HERE / "manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Verify all inputs even on a cache hit. A corrupt source never falls back to
    # an unrestricted pickle loader, a different checkpoint, or a different URL.
    sources = {}
    for name, model in manifest["models"].items():
        sources[name] = {
            kind: source_file(model[kind], manifest["paper"]["commit"], args.offline)
            for kind in ("checkpoint", "config")
        }
    shapes = manifest["raw_tensor_shapes"] | manifest["derived_tensor_shapes"]
    fingerprint = cache_fingerprint(manifest_data)
    metadata_path = OUTPUT / "reference.json"
    if metadata_path.exists() and not args.refresh:
        if validate_output_cache(
            json.loads(metadata_path.read_text()), fingerprint, shapes, manifest["models"]
        ):
            print(f"Verified cached checkpoints and references: {OUTPUT}")
            return

    metadata = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "paper": manifest["paper"],
        "protocol": manifest["protocol"],
        "predeclared_checks": manifest["predeclared_checks"],
        "tensor_dtype": "F32",
        "tensor_shapes": shapes,
        "outputs": {},
    }
    outputs = {}
    for name, model in manifest["models"].items():
        config = json.loads(sources[name]["config"].read_text())
        validate_config(config, model)
        # Restricted tensor-only unpickling is mandatory. Never add a fallback
        # to weights_only=False or import the downloaded author's Python code.
        weights = torch.load(sources[name]["checkpoint"], map_location="cpu", weights_only=True)
        validate_tensors(weights, manifest["raw_tensor_shapes"], name)
        references, measurements = calculate(weights, model, manifest["protocol"])
        tensors = dict(weights) | references
        validate_tensors(tensors, shapes, name)
        check_paper_measurements(name, measurements, manifest["predeclared_checks"])
        data = save({key: value.contiguous() for key, value in tensors.items()})
        outputs[name] = data
        metadata["outputs"][name] = {
            "filename": f"{name}.safetensors",
            "bytes": len(data),
            "sha256": digest(data),
            "sources": {kind: model[kind] for kind in ("checkpoint", "config")},
            "author_config": config,
            "original_training_seed": None,
            "original_split_membership": None,
            "measurements": measurements,
        }
        baseline = measurements["baseline"]
        print(f"{name}: accuracy={baseline['accuracy']:.6f}, q={baseline['distance_irrelevance_q']:.6f}")

    # Compute and validate everything before publishing. Metadata is written
    # last, so an interrupted publication cannot validate a mixed output set.
    for name, data in outputs.items():
        atomic_write(OUTPUT / f"{name}.safetensors", data)
    atomic_write(metadata_path, (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode())
    print(f"Wrote verified weights, PCA tensors, logits, and measurements to {OUTPUT}")


if __name__ == "__main__":
    main()
