#!/usr/bin/env python3
"""Measure the official Nemotron-H graph's own half-precision noise against F32.

The authenticated base-model oracle (``generate_nemotron_h_base_reference.py``)
is an MPS/F16 evaluation. Its rows carry F16 rounding of their own, so a Burn
F16 run compared with them sees the sum of two implementations' rounding. This
offline maintainer tool evaluates the same official class, from the same
clean Transformers checkout pinned to ``cfa309cfa34b5497a32fd50af49aa97aac94c054``,
on identical inputs in wider and other half precisions on the CPU:

* ``bfloat16``: the checkpoint's native dtype;
* ``float32``: BF16-resident weights, each decoder block's parameters widened
  to F32 only while that block runs (BF16 widens to F32 exactly, so every block
  computes what an F32 model computes while only one block is held in F32;
  the embedding, final norm, and head are widened once);
* ``float16``: the oracle's own dtype, loaded the same way the oracle loads it,
  but on another device.

Every evaluation feeds the oracle's prompt and its 32 recorded greedy tokens,
never its own argmax, so all rows describe identical inputs. Each records the
stateless prompt row, the 32 teacher-forced rows of one stateless chunk
(``use_cache=False``), and the 32 cached rows of prefill plus 31 one-token
decodes, called exactly as the oracle calls them. The router-correction buffers
stay F32 in every evaluation, as in the official class.

The manifest compares every row with the F32 row of the same path (cosine
accumulated in F64), including the MPS/F16 oracle's rows, which therefore
measures the reference's own half-precision noise. Inputs are authenticated:
the oracle JSON and the three logit files by pinned SHA-256, the checkpoint's
config by SHA-256, and the two official Nemotron-H source files by SHA-256 in a
clean checkout at the pinned revision. Outputs are staged in a hidden sibling
directory and published by one rename that never replaces an existing bundle.

Usage::

    PYTHONPATH=/path/to/transformers-cfa309/src python3 -I \\
        scripts/generate_nemotron_h_precision_reference.py \\
        --model-dir /path/to/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \\
        --transformers-dir /path/to/transformers-cfa309 \\
        --oracle-dir /path/to/nemotron-3.5-lightning-30b-a3b-a9904d24-reference.bundle \\
        --output /path/to/nemotron-3.5-lightning-30b-a3b-a9904d24-precision.bundle

Memory is the BF16 or F16 model (about 63 GB) plus one F32 block (about
5 GB); the two half-precision loads run one after the other.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence

# Keep every library offline before ML packages are imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_HUB_KERNELS"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

#: Bundle format version, bumped on any incompatible change.
SCHEMA_VERSION = 1
#: Official Transformers revision that produced the oracle.
TRANSFORMERS_REVISION = "cfa309cfa34b5497a32fd50af49aa97aac94c054"
#: Official Nemotron-H sources executed, by checkout-relative path.
SOURCE_SHA256 = {
    "src/transformers/models/nemotron_h/configuration_nemotron_h.py": "00a4c60bd302d796b25c6ae25c608a5ee7e27f9a0adea92911b0c45470232052",
    "src/transformers/models/nemotron_h/modeling_nemotron_h.py": "d42067429e3268d1163579f5d257fe0baa9a2b02edba5fcb175fb8d42c9b6af4",
}
#: Pinned checkpoint revision and its exact config identity.
CHECKPOINT_REVISION = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
CONFIG_SHA256 = "a3827a0f5e311547b40943dc081e3ff2f8a277466e8c1a3df2291e8db8a7617c"
#: Oracle file stem and the pinned digests of the files read from it.
ORACLE_STEM = "nemotron-3.5-lightning-30b-a3b-a9904d24-"
ORACLE_SHA256 = {
    "reference.json": "bd3abb417f2fee82d1bef8bb654ca846fa4816351b1d9d7d0d057105f92de196",
    "prompt-logits.f32le": "f3c014bb043e9ffa3a114a4296ca0e0c2165c4abd63b942df6d910a4383a1dff",
    "teacher-prefix-logits.f32le": "c1878c5e1396ddf01f3dfa85729ea58aa879ece28507399bdaf7addf4a65601b",
    "cached-prefix-logits.f32le": "b3a1ae948e55e6db409e0f9973c41c8767c874c6d64b1d351cfff7bbafa72480",
}
#: Vocabulary width of every row.
VOCABULARY = 131_072
#: Decisions per path.
DECISIONS = 32
#: Evaluations as (label, load dtype, evaluation dtype), grouped by load dtype.
EVALUATIONS = (
    ("bfloat16", "bfloat16", "bfloat16"),
    ("float32", "bfloat16", "float32"),
    ("float16", "float16", "float16"),
)
#: Row kinds recorded for every evaluation, with their row counts.
KINDS = (("prompt", 1), ("teacher", DECISIONS), ("cached", DECISIONS))
#: Name of the bundle's manifest.
MANIFEST = "reference.json"


def sha256_file(path: Path) -> str:
    """Hashes a file's exact bytes in 16 MiB blocks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_rows(payload: bytes, rows: int, width: int = VOCABULARY) -> list[list[float]]:
    """Decodes `rows` little-endian binary32 rows of `width` values each."""

    if len(payload) != rows * width * 4:
        raise ValueError(f"expected {rows * width * 4} bytes, found {len(payload)}")
    values = struct.unpack(f"<{rows * width}f", payload)
    return [list(values[row * width : (row + 1) * width]) for row in range(rows)]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity accumulated in F64; both vectors must be nonzero and equally long."""

    if len(left) != len(right):
        raise ValueError("cosine needs equally long vectors")
    dot = math.fsum(a * b for a, b in zip(left, right))
    norms = math.sqrt(math.fsum(a * a for a in left)) * math.sqrt(math.fsum(b * b for b in right))
    if norms == 0.0:
        raise ValueError("cosine needs nonzero vectors")
    return dot / norms


def argmax(row: Sequence[float]) -> int:
    """Returns the lowest index of the row's maximum."""

    return max(range(len(row)), key=lambda index: (row[index], -index))


def summary(cosines: Sequence[float], top1_agreements: int) -> dict[str, Any]:
    """Packages per-row cosines with their minimum (first on ties) and the top-1 agreement count."""

    if not cosines:
        raise ValueError("a summary needs at least one row")
    minimum = min(range(len(cosines)), key=lambda index: (cosines[index], index))
    return {
        "cosines": list(cosines),
        "minimum": cosines[minimum],
        "minimum_decision": minimum,
        "top1_agreements": top1_agreements,
        "rows": len(cosines),
    }


def summarize(rows: Sequence[Sequence[float]], truth: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Compares small row lists with same-path truth rows (see `summarize_tensors` for real rows)."""

    if len(rows) != len(truth):
        raise ValueError("summaries need equally many rows")
    return summary(
        [cosine(row, reference) for row, reference in zip(rows, truth)],
        sum(argmax(row) == argmax(reference) for row, reference in zip(rows, truth)),
    )


def summarize_tensors(rows: Any, truth: Any) -> dict[str, Any]:
    """Compares `[rows, vocabulary]` tensors row by row, with cosines accumulated in F64."""

    if rows.shape != truth.shape:
        raise ValueError(f"summaries need equal shapes, got {list(rows.shape)} and {list(truth.shape)}")
    left, right = rows.double(), truth.double()
    cosines = (left * right).sum(dim=1) / (left.norm(dim=1) * right.norm(dim=1))
    return summary(cosines.tolist(), int((left.argmax(dim=1) == right.argmax(dim=1)).sum()))


def read_oracle(oracle_dir: Path, pinned: dict[str, str] = ORACLE_SHA256) -> dict[str, Any]:
    """Authenticates and reads the oracle's token IDs and its prompt, teacher, and cached rows."""

    digests = {}
    for suffix, expected in pinned.items():
        path = oracle_dir / f"{ORACLE_STEM}{suffix}"
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"oracle file {path.name} has SHA-256 {actual}, expected {expected}")
        digests[path.name] = actual
    record = json.loads((oracle_dir / f"{ORACLE_STEM}reference.json").read_text())
    prompt_ids = [int(token) for token in record["input_ids"]]
    generated = [int(token) for token in record["generation"]["token_ids"]]
    if len(generated) != DECISIONS:
        raise ValueError(f"the oracle must record {DECISIONS} greedy tokens")

    def payload(suffix: str, count: int) -> bytes:
        """Reads one authenticated oracle logit file after checking its row geometry."""

        data = (oracle_dir / f"{ORACLE_STEM}{suffix}").read_bytes()
        if len(data) != count * VOCABULARY * 4:
            raise ValueError(f"oracle file {suffix} must hold {count} full-vocabulary rows")
        return data

    return {
        "prompt_ids": prompt_ids,
        "generated_ids": generated,
        "digests": digests,
        "payloads": {
            "prompt": payload("prompt-logits.f32le", 1),
            "teacher": payload("teacher-prefix-logits.f32le", DECISIONS),
            "cached": payload("cached-prefix-logits.f32le", DECISIONS),
        },
    }


def verify_checkout(checkout: Path) -> dict[str, Any]:
    """Requires the clean pinned Transformers checkout with the oracle's Nemotron-H sources."""

    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != TRANSFORMERS_REVISION:
        raise ValueError(f"Transformers checkout is at {revision}, expected {TRANSFORMERS_REVISION}")
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise ValueError("the Transformers checkout must be clean")
    for relative, expected in SOURCE_SHA256.items():
        actual = sha256_file(checkout / relative)
        if actual != expected:
            raise ValueError(f"{relative} has SHA-256 {actual}, expected {expected}")
    return {"revision": revision, "clean": True, "sources_sha256": dict(SOURCE_SHA256)}


def import_official(checkout: Path) -> tuple[Any, Any]:
    """Imports PyTorch and the checkout's Transformers, refusing any other installation."""

    sys.path.insert(0, str(checkout / "src"))
    import torch
    import transformers

    origin = Path(transformers.__file__).resolve()
    if not origin.is_relative_to((checkout / "src").resolve()):
        raise ValueError(f"imported Transformers from {origin}, not the pinned checkout")
    return torch, transformers


def load_official(torch: Any, transformers: Any, model_dir: Path, dtype: str) -> Any:
    """Loads the official class on the CPU in `dtype` with eager attention; checks the F32 router buffers."""

    model = transformers.NemotronHForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=getattr(torch, dtype),
        attn_implementation="eager",
    ).eval()
    corrections = [
        buffer for name, buffer in model.named_buffers() if name.endswith("e_score_correction_bias")
    ]
    if len(corrections) != 23 or {buffer.dtype for buffer in corrections} != {torch.float32}:
        raise ValueError("the official class must keep its 23 router-correction buffers in F32")
    return model


@contextlib.contextmanager
def blocks_in_dtype(torch: Any, model: Any, dtype: Any) -> Iterator[None]:
    """Runs every decoder block in `dtype` while it executes; restores the model on exit.

    Parameters outside ``model.model.layers`` are widened for the duration;
    each block's parameters are widened by a forward pre-hook and restored by a
    forward hook, so at most one block is held in `dtype`. Buffers are never
    touched, so the F32 router corrections stay F32.
    """

    blocks = model.model.layers
    inside = {id(parameter) for parameter in blocks.parameters()}
    outside = []
    with torch.no_grad():
        for parameter in model.parameters():
            if id(parameter) not in inside and parameter.is_floating_point() and parameter.dtype != dtype:
                outside.append((parameter, parameter.data))
                parameter.data = parameter.data.to(dtype)

    def widen(block: Any, _arguments: Any) -> None:
        """Replaces the block's parameters by `dtype` copies before it runs."""

        stash = []
        with torch.no_grad():
            for parameter in block.parameters():
                if parameter.is_floating_point() and parameter.dtype != dtype:
                    stash.append((parameter, parameter.data))
                    parameter.data = parameter.data.to(dtype)
        block._precision_stash = stash

    def restore(block: Any, _arguments: Any, _output: Any) -> None:
        """Puts the loaded parameters back once the block has run."""

        for parameter, data in block._precision_stash:
            parameter.data = data
        block._precision_stash = []

    handles = [handle for block in blocks for handle in (block.register_forward_pre_hook(widen), block.register_forward_hook(restore))]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for parameter, data in outside:
            parameter.data = data


def evaluate(torch: Any, model: Any, prompt_ids: list[int], generated_ids: list[int]) -> dict[str, Any]:
    """Runs the oracle's stateless prompt, teacher-forced, and cached calls on identical inputs."""

    def ones(length: int) -> Any:
        """An all-valid attention mask for one row of `length` tokens."""

        return torch.ones((1, length), dtype=torch.long)

    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    teacher_ids = prompt_ids + generated_ids[:-1]
    first = len(prompt_ids) - 1
    with torch.no_grad():
        stateless = model(
            input_ids=prompt, attention_mask=ones(len(prompt_ids)), use_cache=False, logits_to_keep=1, return_dict=True
        ).logits[0, -1:].float()
        teacher = model(
            input_ids=torch.tensor([teacher_ids], dtype=torch.long),
            attention_mask=ones(len(teacher_ids)),
            use_cache=False,
            logits_to_keep=0,
            return_dict=True,
        ).logits[0, first : first + DECISIONS].float()
        output = model(
            input_ids=prompt, attention_mask=ones(len(prompt_ids)), use_cache=True, logits_to_keep=1, return_dict=True
        )
        cached = [output.logits[0, -1].float()]
        for decision in range(1, DECISIONS):
            logical_length = len(prompt_ids) + decision - 1
            output = model(
                input_ids=torch.tensor([[generated_ids[decision - 1]]], dtype=torch.long),
                attention_mask=ones(logical_length + 1),
                past_key_values=output.past_key_values,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            cached.append(output.logits[0, -1].float())
    return {"prompt": stateless, "teacher": teacher, "cached": torch.stack(cached)}


def tensor_f32le(tensor: Any) -> bytes:
    """Serializes an F32 CPU tensor as row-major little-endian binary32."""

    return tensor.contiguous().numpy().astype("<f4", copy=False).tobytes()


def publish(staging: Path, destination: Path) -> None:
    """Renames a finished staging directory into place without replacing anything."""

    if destination.exists():
        raise FileExistsError(f"refusing to replace existing bundle {destination}")
    staging.chmod(0o755)
    os.rename(staging, destination)


def build_bundle(arguments: argparse.Namespace, staging: Path) -> dict[str, Any]:
    """Evaluates every precision, writes its rows into `staging`, and returns the manifest."""

    model_dir = arguments.model_dir.resolve(strict=True)
    config_sha256 = sha256_file(model_dir / "config.json")
    if config_sha256 != CONFIG_SHA256:
        raise ValueError(f"config.json has SHA-256 {config_sha256}, expected {CONFIG_SHA256}")
    checkout = verify_checkout(arguments.transformers_dir.resolve(strict=True))
    oracle = read_oracle(arguments.oracle_dir.resolve(strict=True))
    torch, transformers = import_official(arguments.transformers_dir.resolve(strict=True))
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    rows: dict[str, dict[str, Any]] = {
        "float16-mps-oracle": {
            kind: torch.frombuffer(bytearray(oracle["payloads"][kind]), dtype=torch.float32).reshape(count, VOCABULARY)
            for kind, count in KINDS
        }
    }
    files = {}
    loaded_dtype, model = None, None
    for label, load_dtype, evaluation_dtype in EVALUATIONS:
        if load_dtype != loaded_dtype:
            model = None
            gc.collect()
            model = load_official(torch, transformers, model_dir, load_dtype)
            loaded_dtype = load_dtype
        context = (
            blocks_in_dtype(torch, model, getattr(torch, evaluation_dtype))
            if evaluation_dtype != load_dtype
            else contextlib.nullcontext()
        )
        with context:
            result = evaluate(torch, model, oracle["prompt_ids"], oracle["generated_ids"])
        rows[label] = {}
        for kind, count in KINDS:
            tensor = result[kind]
            if list(tensor.shape) != [count, VOCABULARY] or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{label} {kind} rows are malformed")
            name = f"{label}-{kind}-logits.f32le"
            (staging / name).write_bytes(tensor_f32le(tensor))
            files[name] = sha256_file(staging / name)
            rows[label][kind] = tensor.contiguous()
    model = None
    gc.collect()

    comparisons = {
        label: {kind: summarize_tensors(rows[label][kind], rows["float32"][kind]) for kind, _ in KINDS}
        for label in ("float16-mps-oracle", "float16", "bfloat16")
    }
    comparisons["float16-mps-oracle-vs-float16"] = {
        kind: summarize_tensors(rows["float16-mps-oracle"][kind], rows["float16"][kind]) for kind, _ in KINDS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "repository": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "revision": CHECKPOINT_REVISION,
        "prompt_ids": oracle["prompt_ids"],
        "generated_ids": oracle["generated_ids"],
        "evaluations": [
            {"label": label, "load_dtype": load, "evaluation_dtype": evaluation, "device": "cpu"}
            for label, load, evaluation in EVALUATIONS
        ],
        "files_sha256": files,
        "comparisons_against_float32": comparisons,
        "provenance": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "script": "scripts/generate_nemotron_h_precision_reference.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "transformers": transformers.__version__,
            "transformers_checkout": checkout,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "model_dir": str(model_dir),
            "config_sha256": config_sha256,
            "oracle_dir": str(arguments.oracle_dir.resolve()),
            "oracle_sha256": oracle["digests"],
            "attention_implementation": "eager",
        },
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True, help="pinned local checkpoint")
    parser.add_argument("--transformers-dir", type=Path, required=True, help="clean checkout at the pinned revision")
    parser.add_argument("--oracle-dir", type=Path, required=True, help="authenticated MPS/F16 oracle bundle")
    parser.add_argument("--output", type=Path, required=True, help="bundle directory to create")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Builds the bundle in a hidden staging directory and publishes it, or removes the staging."""

    arguments = parse_arguments(argv)
    destination = arguments.output.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing bundle {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        manifest = build_bundle(arguments, staging)
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        publish(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    summary = {
        label: {kind: [round(item["minimum"], 9), item["minimum_decision"]] for kind, item in kinds.items()}
        for label, kinds in manifest["comparisons_against_float32"].items()
    }
    print(json.dumps({"bundle": str(destination), "minimum_cosines": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
