#!/usr/bin/env python3
"""Generate a local F32 reference bundle for any Hugging Face causal language model.

This offline maintainer tool records what the official Transformers
implementation computes for a local checkpoint, so the Rust crate can be
compared against it without Python at test time. It

* loads ``AutoModelForCausalLM`` from a local directory only (no Hub access),
  in F32 (or, with ``--dtype bfloat16``, BF16) on the CPU with eager
  attention and deterministic kernels;
* tokenizes a few fixed prompts with the checkpoint's own tokenizer and
  records their token IDs, so consumers need no tokenizer;
* writes every prompt position's full-vocabulary logits as raw little-endian
  F32 files (``logits-<index>.f32le``, shape ``[tokens, vocabulary]``);
* records a cached greedy continuation of the first prompt (default 32
  tokens) with each step's top-2 logit margin, so consumers can skip steps
  that are ties at F32 precision;
* records provenance: tool versions, interpreter and platform, the caller's
  repository and revision, and SHA-256 digests of the configuration, every
  weight file, this script, and every output file.

The bundle is assembled in a hidden sibling directory and published by one
rename that refuses to replace an existing bundle; a failed or interrupted
run removes its staging directory, so partial bundles never accumulate.

Usage::

    python3 scripts/generate_causal_lm_reference.py \\
        --model-dir /path/to/checkpoint --output /path/to/references/name \\
        --repository HuggingFaceTB/SmolLM-135M --revision <commit>

Checkpoints too large to evaluate in F32 can be evaluated in BF16 with
``--dtype bfloat16``; logits are still written as F32. ``--num-hidden-layers N``
keeps only the first ``N`` blocks (with the real embedding, final norm, and
head), a real-weight smoke reference that fits in memory at F32. gpt-oss is the
motivating case: without CUDA kernels Transformers decodes its MXFP4
experts to BF16, and the F32 model (about 78 GiB for gpt-oss-20b) does not
fit a 96 GiB host, while the BF16 one needs about 55 GiB while loading.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

# Keep every library offline and deterministic before ML packages are imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_HUB_KERNELS"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

#: Bundle format version, bumped on any incompatible change.
SCHEMA_VERSION = 1
#: Prompts tokenized and evaluated by default; the first seeds the greedy run.
DEFAULT_PROMPTS = (
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Rotary position embeddings encode relative distance because",
)
#: Default length of the cached greedy continuation.
DEFAULT_GREEDY_TOKENS = 32
#: Name of the bundle's manifest.
MANIFEST = "reference.json"
#: Evaluation dtypes accepted by ``--dtype``; logits are always written as F32.
DTYPES = ("float32", "bfloat16")


def sha256_file(path: Path) -> str:
    """Hashes a file's exact bytes in 16 MiB blocks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_f32le(values: Sequence[float]) -> bytes:
    """Encodes floats as consecutive little-endian IEEE-754 binary32 values."""

    return struct.pack(f"<{len(values)}f", *values)


def decode_f32le(payload: bytes) -> list[float]:
    """Decodes consecutive little-endian binary32 values."""

    if len(payload) % 4:
        raise ValueError("an f32le payload must be a multiple of four bytes")
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))


def top_two_margin(logits: Sequence[float]) -> tuple[int, float]:
    """Returns the argmax (lowest index on ties) and the gap to the runner-up."""

    if len(logits) < 2:
        raise ValueError("a margin needs at least two logits")
    best = max(range(len(logits)), key=lambda index: (logits[index], -index))
    runner_up = max(value for index, value in enumerate(logits) if index != best)
    return best, float(logits[best] - runner_up)


def weight_files(model_dir: Path) -> list[Path]:
    """Lists the checkpoint's SafeTensors files in name order."""

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    return files


def publish(staging: Path, destination: Path) -> None:
    """Renames a finished staging directory into place without replacing anything."""

    if destination.exists():
        raise FileExistsError(f"refusing to replace existing bundle {destination}")
    staging.chmod(0o755)
    os.rename(staging, destination)


def provenance(model_dir: Path, arguments: argparse.Namespace) -> dict[str, Any]:
    """Records tool versions, platform, caller identity, and input digests."""

    import torch
    import transformers

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/generate_causal_lm_reference.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repository": arguments.repository,
        "revision": arguments.revision,
        "model_dir": str(model_dir),
        "config_sha256": sha256_file(model_dir / "config.json"),
        "weights_sha256": {path.name: sha256_file(path) for path in weight_files(model_dir)},
        "dtype": arguments.dtype,
        "num_hidden_layers": arguments.num_hidden_layers,
        "device": "cpu",
        "attention_implementation": "eager",
    }


def truncate_config(config: Any, layers: int) -> None:
    """Keeps only the first `layers` decoder blocks of a configuration, in place.

    `num_hidden_layers` and, when present, the per-layer `layer_types`
    schedule are cut; every other field is unchanged, so the truncated model
    loads the checkpoint's first blocks, embedding, final norm, and head.
    """

    if layers < 1 or layers > config.num_hidden_layers:
        raise ValueError(f"cannot keep {layers} of {config.num_hidden_layers} layers")
    config.num_hidden_layers = layers
    if getattr(config, "layer_types", None):
        config.layer_types = list(config.layer_types[:layers])


def cast_floating_parameters(model: Any, dtype: Any) -> int:
    """Casts every floating parameter of `model` to the torch `dtype` in place; returns how many changed.

    Buffers are deliberately left alone. Transformers keeps some of them in
    F32 whatever the model dtype, most importantly the rotary ``inv_freq``
    tables: rounding those to BF16 moves every rotary angle (by up to 0.28 rad
    at position 183 for gpt-oss-20b's YaRN frequencies), so a BF16 reference
    would no longer be ordinary BF16 inference. ``torch.nn.Module.to`` casts
    buffers too and is therefore not used. Integer parameters and those
    already in `dtype` are untouched; tied parameters are visited once.
    """

    import torch

    changed = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype)
                changed += 1
    return changed


def load_causal_lm(model_dir: Path, dtype: str = "float32", layers: int | None = None) -> Any:
    """Loads the eager model, in `dtype` (one of `DTYPES`), from local files only.

    With `layers`, only the first `layers` decoder blocks are built and
    loaded (see `truncate_config`); the checkpoint's other blocks are ignored.

    Quantized checkpoints can leave dequantized parameters in another dtype:
    without CUDA kernels Transformers decodes MXFP4 experts to BF16 whatever
    `dtype` says, and its grouped expert kernels then cast F32 activations
    down to BF16, so an "F32" reference would not be F32. Every floating
    parameter is therefore cast to `dtype` after loading (exact for the BF16
    decodings; nothing changes for an ordinary checkpoint or for
    ``bfloat16``), while buffers keep the dtype Transformers chose (see
    `cast_floating_parameters`).
    """

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, not {dtype!r}")
    import torch
    import transformers

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    transformers.utils.logging.disable_progress_bar()
    config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if layers is not None:
        truncate_config(config, layers)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        dtype=getattr(torch, dtype),
        attn_implementation="eager",
    )
    cast_floating_parameters(model, getattr(torch, dtype))
    model.eval()
    return model


def load_model(model_dir: Path, dtype: str = "float32", layers: int | None = None) -> tuple[Any, Any]:
    """Loads the checkpoint's own tokenizer and its eager model (see `load_causal_lm`)."""

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, not {dtype!r}")
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return tokenizer, load_causal_lm(model_dir, dtype, layers)


def prompt_logits(model: Any, token_ids: list[int]) -> Any:
    """Returns every position's full-vocabulary F32 logits `[tokens, vocabulary]` for one prompt."""

    import torch

    with torch.no_grad():
        output = model(input_ids=torch.tensor([token_ids]), use_cache=False)
    return output.logits[0].to(torch.float32).contiguous()


def tensor_f32le(tensor: Any) -> bytes:
    """Serializes an F32 CPU tensor as row-major little-endian binary32."""

    return tensor.contiguous().numpy().astype("<f4", copy=False).tobytes()


def greedy_continuation(model: Any, token_ids: list[int], steps: int) -> list[dict[str, Any]]:
    """Decodes `steps` greedy tokens with the model's own KV cache."""

    import torch

    records = []
    with torch.no_grad():
        output = model(input_ids=torch.tensor([token_ids]), use_cache=True)
        for _ in range(steps):
            token, margin = top_two_margin(output.logits[0, -1].to(torch.float32).tolist())
            records.append({"token": token, "margin": margin})
            output = model(
                input_ids=torch.tensor([[token]]),
                past_key_values=output.past_key_values,
                use_cache=True,
            )
    return records


def build_bundle(arguments: argparse.Namespace, staging: Path) -> dict[str, Any]:
    """Evaluates the prompts and writes the logits files into `staging`."""

    model_dir = arguments.model_dir.resolve(strict=True)
    tokenizer, model = load_model(model_dir, arguments.dtype, arguments.num_hidden_layers)
    prompts = []
    for index, text in enumerate(arguments.prompts):
        token_ids = tokenizer(text, add_special_tokens=arguments.add_special_tokens)["input_ids"]
        logits = prompt_logits(model, token_ids)
        name = f"logits-{index}.f32le"
        (staging / name).write_bytes(tensor_f32le(logits))
        last_top1, last_margin = top_two_margin(logits[-1].tolist())
        prompts.append(
            {
                "text": text,
                "token_ids": token_ids,
                "logits_file": name,
                "logits_shape": list(logits.shape),
                "logits_sha256": sha256_file(staging / name),
                "last_top1": last_top1,
                "last_margin": last_margin,
            }
        )
    greedy = greedy_continuation(model, prompts[0]["token_ids"], arguments.greedy_tokens)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": model.config.model_type,
        "model_class": type(model).__name__,
        "vocabulary_size": prompts[0]["logits_shape"][1],
        "provenance": provenance(model_dir, arguments),
        "prompts": prompts,
        "greedy": {"prompt_index": 0, "steps": greedy},
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True, help="local checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="bundle directory to create")
    parser.add_argument("--repository", help="upstream repository recorded as provenance")
    parser.add_argument("--revision", help="upstream commit recorded as provenance")
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help="prompt text (repeatable); defaults to three fixed prompts",
    )
    parser.add_argument("--greedy-tokens", type=int, default=DEFAULT_GREEDY_TOKENS)
    parser.add_argument(
        "--dtype",
        choices=DTYPES,
        default="float32",
        help="evaluation dtype; logits are written as F32 either way",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        help="keep only the first N decoder blocks (a real-weight smoke reference)",
    )
    parser.add_argument(
        "--no-special-tokens",
        dest="add_special_tokens",
        action="store_false",
        help="tokenize without the tokenizer's special tokens",
    )
    arguments = parser.parse_args(argv)
    if arguments.prompts is None:
        arguments.prompts = list(DEFAULT_PROMPTS)
    if arguments.greedy_tokens < 1:
        parser.error("--greedy-tokens must be at least one")
    if arguments.num_hidden_layers is not None and arguments.num_hidden_layers < 1:
        parser.error("--num-hidden-layers must be at least one")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    """Builds the bundle in a hidden staging directory and publishes it.

    Any failure before publication (an exception, an out-of-memory error
    surfacing as one, or ``KeyboardInterrupt``) removes the staging directory
    and re-raises, leaving the references volume as it was.
    """

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
    print(json.dumps({"bundle": str(destination), "prompts": len(manifest["prompts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
