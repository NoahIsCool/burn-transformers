#!/usr/bin/env python3
"""Generate a local F32 reference bundle for any Hugging Face causal language model.

This offline maintainer tool records what the official Transformers
implementation computes for a local checkpoint, so the Rust crate can be
compared against it without Python at test time. It

* loads ``AutoModelForCausalLM`` from a local directory only (no Hub access),
  in F32 (or, with ``--dtype float64`` or ``bfloat16``, F64 or BF16) on the CPU with eager
  attention (unless ``--device`` or ``--attn-implementation`` selects another
  graph) and deterministic kernels;
* tokenizes a few fixed prompts with the checkpoint's own tokenizer and
  records their token IDs, so consumers need no tokenizer;
* writes every prompt position's full-vocabulary logits as raw little-endian
  F32 files (``logits-<index>.f32le``, shape ``[tokens, vocabulary]``);
* records a cached greedy continuation of the first prompt (default 32
  tokens) with each step's top-2 logit margin, so consumers can skip steps
  that are ties at F32 precision;
* with ``--decode-tokens-from``, records cached-decode against stateless
  logits on a truth bundle's greedy tokens (see below);
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

``--bf16-resident`` evaluates such a model in F32 anyway. It loads the
checkpoint in BF16 and casts each decoder block's parameters to ``--dtype``
only while that block runs, restoring them afterwards; the embedding, final
norm, and head are cast once. Only one block is ever held in F32, and because
BF16 widens to F32 exactly, the result equals an F32 model's when two
conditions hold, which the tool checks before evaluating anything:

* every floating tensor in the checkpoint is stored as BF16 (gpt-oss's MXFP4
  blocks are integers and Transformers decodes them to BF16), so no weight is
  rounded on the way in;
* every floating buffer of the BF16 model equals, in dtype and value, the
  buffer a plain ``--dtype`` load would hold. Buffers are not cast, and a
  model built under a BF16 default dtype can hold BF16-rounded constants:
  Gemma3's ``embed_scale``, ``sqrt(hidden_size)``, becomes ``5.65625``
  instead of ``5.656854`` on the tiny fixture, which moves its logits by up
  to ``9.3e-5``, so Gemma3 is refused. The comparison instantiates the same
  configuration in ``--dtype`` with its parameters on the meta device, so it
  allocates only the buffers. Persistent floating buffers, whose values come
  from the checkpoint, are refused.

gpt-oss passes both checks: its only floating buffers are the F32 rotary
tables, which Transformers computes in F32 under either default dtype.

``--dtype float64`` removes the reference's own rounding error, which matters
for checkpoints with massive activations. Qwen2-7B's first-position hidden
states reach magnitudes near 7,000; on its third prompt, F32 Transformers
itself deviates from an F64 evaluation by 1.6e-3 in a first-position logit,
while this crate's F32 Flex result deviates by only 2.0e-4. Comparing F32
implementations against an F64 bundle measures each implementation's error
instead of the sum of two. Memory is twice the F32 requirement (about 61 GB
of weights for a 7B checkpoint), Transformers still computes its rotary
tables in F32, and kernels without an F64 implementation (such as gpt-oss's
grouped expert products) refuse F64 inputs.

``--device`` and ``--attn-implementation`` select another graph that
Transformers itself runs: ``mps`` (Apple's Metal Performance Shaders, which
has no F64) and ``sdpa`` (PyTorch's fused ``scaled_dot_product_attention``).
The model is loaded on the CPU as before and then moved, so only the
kernels differ. The defaults, ``cpu`` and ``eager``, compute exactly what
the tool computed before these options existed. One BF16 bundle per graph
measures how far the official implementation's own half-precision graphs
spread around a single F32 truth; a BF16 gate calibrated on one graph alone
is narrower than that spread.

``--decode-tokens-from TRUTH`` also records decode self-consistency, the
quantity the crate's half-precision cached greedy gates bound. Prompt 0 is
prefilled into the model's own KV cache, the greedy tokens of the bundle
``TRUTH`` (whose prompt 0 must tokenize identically) are fed one at a time,
and each step's logits are compared with one stateless pass over prompt 0
and all of those tokens at the same position. Teacher-forcing the truth's
tokens, rather than this evaluation's own greedy choices, keeps the sequence
identical to the one a gate decodes even where the two continuations
diverge. Both sets of logits are written as ``decode-cached.f32le`` and
``decode-stateless.f32le`` (``[steps, vocabulary]``) with each step's
largest absolute difference, taken in F64.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
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
DTYPES = ("float32", "float64", "bfloat16")
#: Devices accepted by ``--device``; the model is loaded on the CPU and moved.
DEVICES = ("cpu", "mps")
#: Transformers attention implementations accepted by ``--attn-implementation``.
ATTENTION_IMPLEMENTATIONS = ("eager", "sdpa")
#: Names of the decode self-consistency logits files (``--decode-tokens-from``).
DECODE_CACHED_FILE = "decode-cached.f32le"
DECODE_STATELESS_FILE = "decode-stateless.f32le"


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
        "bf16_resident": arguments.bf16_resident,
        "num_hidden_layers": arguments.num_hidden_layers,
        "device": arguments.device,
        "attention_implementation": arguments.attn_implementation,
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


#: SafeTensors dtype names of floating-point tensors.
FLOATING_SAFETENSORS_DTYPES = frozenset({"F16", "BF16", "F32", "F64", "F8_E4M3", "F8_E5M2"})


def checkpoint_floating_dtypes(model_dir: Path) -> set[str]:
    """Returns the SafeTensors dtypes of every floating tensor stored in the checkpoint."""

    found = set()
    for path in weight_files(model_dir):
        with path.open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(length))
        for name, record in header.items():
            if name != "__metadata__" and record["dtype"] in FLOATING_SAFETENSORS_DTYPES:
                found.add(record["dtype"])
    return found


def require_bfloat16_checkpoint(model_dir: Path) -> None:
    """Refuses `--bf16-resident` unless loading in BF16 rounds no stored weight."""

    dtypes = checkpoint_floating_dtypes(model_dir)
    if dtypes != {"BF16"}:
        raise ValueError(
            f"--bf16-resident needs a checkpoint whose floating tensors are all BF16, found {sorted(dtypes)}"
        )


def decoder_blocks(model: Any) -> Any:
    """Returns the model's unique `ModuleList` holding `config.num_hidden_layers` blocks."""

    import torch

    count = model.config.num_hidden_layers
    candidates = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.ModuleList) and len(module) == count
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one {count}-block ModuleList, found {len(candidates)}")
    return candidates[0]


def stream_blocks_in_dtype(model: Any, dtype: Any) -> int:
    """Casts each decoder block's floating parameters to `dtype` only while it runs.

    Parameters outside the blocks (embedding, final norm, head) are cast once
    with `cast_floating_parameters`. Each block keeps its loaded parameters;
    a forward pre-hook swaps in `dtype` copies and a forward hook restores the
    originals, so at most one block is held in `dtype` at a time. Buffers are
    never touched (see `cast_floating_parameters`), so the result equals a
    plain `dtype` model only if they already match one; `load_bf16_resident`
    proves that before streaming. Returns the number of blocks hooked.
    """

    import torch

    blocks = decoder_blocks(model)
    inside = {id(parameter) for parameter in blocks.parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            if id(parameter) not in inside and parameter.is_floating_point() and parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype)

    def widen(block: Any, _arguments: Any) -> None:
        """Replaces the block's parameters by `dtype` copies before it runs."""

        stash = []
        with torch.no_grad():
            for parameter in block.parameters():
                if parameter.is_floating_point() and parameter.dtype != dtype:
                    stash.append((parameter, parameter.data))
                    parameter.data = parameter.data.to(dtype)
        block._reference_stash = stash

    def restore(block: Any, _arguments: Any, _output: Any) -> None:
        """Puts the loaded parameters back once the block has run."""

        for parameter, data in block._reference_stash:
            parameter.data = data
        block._reference_stash = []

    for block in blocks:
        block.register_forward_pre_hook(widen)
        block.register_forward_hook(restore)
    return len(blocks)


def instantiation_buffers(
    model_dir: Path, dtype: str, layers: int | None = None
) -> tuple[dict[str, Any], frozenset[str]]:
    """Returns the buffers a plain `dtype` load of the checkpoint would hold, and the persistent names.

    This repeats what ``from_pretrained`` does to buffers without allocating
    any parameter: the configuration (truncated like `load_causal_lm` when
    `layers` is given) is instantiated on the meta device under the `dtype`
    default, every non-persistent buffer is materialized on the CPU, and the
    model's own ``initialize_weights`` fills them, as ``from_pretrained`` does
    for buffers that no checkpoint provides. Parameters stay on the meta
    device, where initialization is a no-op. Persistent buffers are left on the
    meta device: their values would come from the checkpoint, so they are only
    named, in the returned set. Buffer names follow ``named_buffers``, which
    lists a shared buffer once, as it does for the loaded model.
    """

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, not {dtype!r}")
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if layers is not None:
        truncate_config(config, layers)
    with torch.device("meta"):
        skeleton = transformers.AutoModelForCausalLM.from_config(
            config, dtype=getattr(torch, dtype), attn_implementation="eager"
        )
    persistent = frozenset(skeleton.state_dict(keep_vars=True))
    names = [name for name, _buffer in skeleton.named_buffers()]
    for name in names:
        if name not in persistent:
            owner, _, local = name.rpartition(".")
            buffer = skeleton.get_buffer(name)
            skeleton.get_submodule(owner).register_buffer(
                local, torch.empty_like(buffer, device="cpu"), persistent=False
            )
    skeleton.initialize_weights()
    return {name: skeleton.get_buffer(name) for name in names}, persistent


def buffer_mismatches(model: Any, expected: dict[str, Any], persistent: frozenset[str]) -> list[str]:
    """Describes every floating buffer of `model` that a plain load would not hold identically.

    `expected` and `persistent` come from `instantiation_buffers`. A floating
    buffer passes only when both sides hold it with the same dtype, shape, and
    exact values (``torch.equal``, so a NaN never passes). A floating buffer
    on one side only, or a persistent one, is reported too: either way the
    loaded model is not known to compute what the plain one computes. Integer
    buffers are not numerical constants of the forward pass and are ignored.
    The result is sorted by buffer name and empty when nothing differs.
    """

    import torch

    actual = {name: buffer for name, buffer in model.named_buffers() if buffer.is_floating_point()}
    wanted = {name: buffer for name, buffer in expected.items() if buffer.is_floating_point()}
    problems = []
    for name in sorted(actual.keys() | wanted.keys()):
        if name not in wanted:
            problems.append(f"{name} (not a floating buffer of a plain load)")
        elif name not in actual:
            problems.append(f"{name} (not a floating buffer of the BF16 load)")
        elif name in persistent:
            problems.append(f"{name} (persistent; its value comes from the checkpoint)")
        elif actual[name].dtype != wanted[name].dtype:
            problems.append(f"{name} ({actual[name].dtype} instead of {wanted[name].dtype})")
        elif actual[name].shape != wanted[name].shape or not torch.equal(
            actual[name].cpu(), wanted[name].cpu()
        ):
            problems.append(f"{name} (values differ)")
    return problems


def require_bf16_resident_inputs(model_dir: Path, dtype: str) -> None:
    """Refuses `--bf16-resident` before any load unless `dtype` is wider than BF16 and the checkpoint all BF16."""

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, not {dtype!r}")
    if dtype == "bfloat16":
        raise ValueError("--bf16-resident needs a wider --dtype than bfloat16")
    require_bfloat16_checkpoint(model_dir)


def load_bf16_resident(model_dir: Path, dtype: str = "float32", layers: int | None = None) -> Any:
    """Loads a BF16 checkpoint in BF16 and runs its blocks in `dtype` (see `stream_blocks_in_dtype`).

    The run is refused with ``ValueError`` unless it reproduces a plain
    `dtype` load exactly: before loading, `require_bf16_resident_inputs`
    checks the requested dtype and the stored weights; after loading, every
    floating buffer must match `instantiation_buffers` (see
    `buffer_mismatches`), because streaming never casts buffers.
    """

    import torch

    require_bf16_resident_inputs(model_dir, dtype)
    model = load_causal_lm(model_dir, "bfloat16", layers)
    expected, persistent = instantiation_buffers(model_dir, dtype, layers)
    problems = buffer_mismatches(model, expected, persistent)
    if problems:
        raise ValueError(
            f"--bf16-resident cannot reproduce a {dtype} model: these floating buffers of the BF16 "
            f"load differ from a {dtype} instantiation: {'; '.join(problems)}"
        )
    stream_blocks_in_dtype(model, getattr(torch, dtype))
    return model


def require_graph(dtype: str, attention: str, device: str) -> None:
    """Refuses an unknown attention implementation or device, and F64 on a device without it.

    Raises ``ValueError`` before anything is loaded: MPS has no F64 kernels,
    so an F64 evaluation there would fail mid-run or silently run elsewhere.
    """

    if attention not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError(f"attention must be one of {ATTENTION_IMPLEMENTATIONS}, not {attention!r}")
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, not {device!r}")
    if device == "mps" and dtype == "float64":
        raise ValueError("MPS has no float64 kernels; evaluate float64 on the cpu device")


def load_causal_lm(
    model_dir: Path,
    dtype: str = "float32",
    layers: int | None = None,
    attention: str = "eager",
    device: str = "cpu",
) -> Any:
    """Loads the model, in `dtype` (one of `DTYPES`), from local files only.

    With `layers`, only the first `layers` decoder blocks are built and
    loaded (see `truncate_config`); the checkpoint's other blocks are ignored.
    `attention` selects the Transformers attention implementation (`eager` by
    default, or `sdpa`) and `device` where the loaded model runs (see
    `require_graph`). The model is always loaded and cast on the CPU and only
    then moved, so every device starts from the same parameters; moving by
    device alone never changes a dtype, so buffers keep theirs.

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
    require_graph(dtype, attention, device)
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
        attn_implementation=attention,
    )
    cast_floating_parameters(model, getattr(torch, dtype))
    if device != "cpu":
        model.to(device)
    model.eval()
    return model


def load_model(
    model_dir: Path,
    dtype: str = "float32",
    layers: int | None = None,
    bf16_resident: bool = False,
    attention: str = "eager",
    device: str = "cpu",
) -> tuple[Any, Any]:
    """Loads the checkpoint's own tokenizer and its model (see `load_causal_lm`).

    With `bf16_resident`, the model is loaded in BF16 and its blocks run in
    `dtype` (see `load_bf16_resident`, which refuses any run that would not
    reproduce a plain `dtype` load); that mode is defined for the eager CPU
    graph only and refuses any other `attention` or `device`.
    """

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, not {dtype!r}")
    require_graph(dtype, attention, device)
    if bf16_resident:
        if (attention, device) != ("eager", "cpu"):
            raise ValueError("--bf16-resident evaluates the eager CPU graph only")
        # Fail before any load: the mode is exact only for wider dtypes and BF16 checkpoints.
        require_bf16_resident_inputs(model_dir, dtype)
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if not bf16_resident:
        return tokenizer, load_causal_lm(model_dir, dtype, layers, attention, device)
    return tokenizer, load_bf16_resident(model_dir, dtype, layers)


def model_input(model: Any, token_ids: list[int]) -> Any:
    """Returns one unpadded row of token IDs `[1, tokens]` on the model's device."""

    import torch

    return torch.tensor([token_ids], device=model.device)


def prompt_logits(model: Any, token_ids: list[int]) -> Any:
    """Returns every position's full-vocabulary F32 logits `[tokens, vocabulary]` for one prompt, on the CPU."""

    import torch

    with torch.no_grad():
        output = model(input_ids=model_input(model, token_ids), use_cache=False)
    return output.logits[0].to(torch.float32).cpu().contiguous()


def tensor_f32le(tensor: Any) -> bytes:
    """Serializes an F32 CPU tensor as row-major little-endian binary32."""

    return tensor.contiguous().numpy().astype("<f4", copy=False).tobytes()


def greedy_continuation(model: Any, token_ids: list[int], steps: int) -> list[dict[str, Any]]:
    """Decodes `steps` greedy tokens with the model's own KV cache."""

    import torch

    records = []
    with torch.no_grad():
        output = model(input_ids=model_input(model, token_ids), use_cache=True)
        for _ in range(steps):
            token, margin = top_two_margin(output.logits[0, -1].to(torch.float32).tolist())
            records.append({"token": token, "margin": margin})
            output = model(
                input_ids=model_input(model, [token]),
                past_key_values=output.past_key_values,
                use_cache=True,
            )
    return records


def teacher_tokens(bundle: Path, prompt_ids: list[int], vocabulary: int) -> tuple[list[int], dict[str, Any]]:
    """Reads the greedy tokens a decode self-consistency record feeds, from the truth bundle `bundle`.

    The truth's greedy continuation must extend its prompt 0, which must hold
    exactly `prompt_ids`, and every token must index the `vocabulary`;
    otherwise ``ValueError`` is raised, because the record would describe
    another sequence than the one a gate decodes. Returns the tokens and the
    source's identity (SHA-256 of its exact manifest bytes and its dtype).
    """

    manifest_path = bundle / MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if manifest["prompts"][0]["token_ids"] != prompt_ids:
        raise ValueError(f"{manifest_path}: prompt 0 has other token IDs than this evaluation's prompt 0")
    greedy = manifest["greedy"]
    if greedy["prompt_index"] != 0:
        raise ValueError(f"{manifest_path}: the greedy continuation does not extend prompt 0")
    tokens = [step["token"] for step in greedy["steps"]]
    if not tokens:
        raise ValueError(f"{manifest_path}: the greedy continuation is empty")
    if any(not isinstance(token, int) or isinstance(token, bool) or not 0 <= token < vocabulary for token in tokens):
        raise ValueError(f"{manifest_path}: a greedy token is not an integer below {vocabulary}")
    source = {
        "manifest_sha256": sha256_file(manifest_path),
        "dtype": manifest["provenance"]["dtype"],
    }
    return tokens, source


def decode_logits(model: Any, prompt_ids: list[int], tokens: list[int]) -> tuple[Any, Any]:
    """Returns the cached-decode and stateless F32 logits `[len(tokens), vocabulary]` of `tokens` after a prompt.

    Cached: prompt 0 is prefilled into the model's own KV cache, then each of
    `tokens` is fed alone; step `s` feeds `tokens[s]` at position
    `len(prompt_ids) + s` and returns that position's logits. Stateless: one
    pass without a cache over the prompt followed by every token, read at the
    same positions. Both are returned on the CPU.
    """

    import torch

    with torch.no_grad():
        output = model(input_ids=model_input(model, prompt_ids), use_cache=True)
        cached = []
        for token in tokens:
            output = model(
                input_ids=model_input(model, [token]),
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            cached.append(output.logits[0, -1].to(torch.float32).cpu())
        stateless = model(input_ids=model_input(model, prompt_ids + tokens), use_cache=False).logits
    stateless = stateless[0, len(prompt_ids):].to(torch.float32).cpu().contiguous()
    return torch.stack(cached).contiguous(), stateless


def step_gaps(cached: Sequence[Sequence[float]], stateless: Sequence[Sequence[float]]) -> list[float]:
    """Returns, per decode step, the largest `|cached - stateless|` over the vocabulary, taken in F64.

    Both arguments are equally shaped `[steps, vocabulary]` rows of F32 values
    (anything whose rows iterate as floats). A non-finite value on either side
    raises ``ValueError``: the gap would not bound anything, so no record is
    written.
    """

    if len(cached) != len(stateless):
        raise ValueError("cached and stateless logits hold different numbers of steps")
    gaps = []
    for step, (left, right) in enumerate(zip(cached, stateless)):
        left, right = [float(value) for value in left], [float(value) for value in right]
        if len(left) != len(right) or not left:
            raise ValueError(f"decode step {step}: cached and stateless rows differ in length or are empty")
        if not all(math.isfinite(value) for value in left + right):
            raise ValueError(f"decode step {step}: the logits hold a non-finite value")
        gaps.append(max(abs(a - b) for a, b in zip(left, right)))
    return gaps


def decode_record(
    model: Any, prompt_ids: list[int], tokens: list[int], source: dict[str, Any], staging: Path
) -> dict[str, Any]:
    """Writes both decode logits files into `staging` and returns the manifest's ``decode_consistency`` record."""

    cached, stateless = decode_logits(model, prompt_ids, tokens)
    gaps = step_gaps(cached.tolist(), stateless.tolist())
    (staging / DECODE_CACHED_FILE).write_bytes(tensor_f32le(cached))
    (staging / DECODE_STATELESS_FILE).write_bytes(tensor_f32le(stateless))
    first = len(prompt_ids)
    return {
        "prompt_index": 0,
        "first_position": first,
        "tokens_source": source,
        "logits_shape": list(cached.shape),
        "cached_logits_file": DECODE_CACHED_FILE,
        "cached_logits_sha256": sha256_file(staging / DECODE_CACHED_FILE),
        "stateless_logits_file": DECODE_STATELESS_FILE,
        "stateless_logits_sha256": sha256_file(staging / DECODE_STATELESS_FILE),
        "steps": [
            {"position": first + step, "fed_token": token, "max_abs_gap": gap}
            for step, (token, gap) in enumerate(zip(tokens, gaps))
        ],
        "max_abs_gap": max(gaps),
    }


def build_bundle(arguments: argparse.Namespace, staging: Path) -> dict[str, Any]:
    """Evaluates the prompts and writes the logits files into `staging`."""

    model_dir = arguments.model_dir.resolve(strict=True)
    tokenizer, model = load_model(
        model_dir,
        arguments.dtype,
        arguments.num_hidden_layers,
        arguments.bf16_resident,
        arguments.attn_implementation,
        arguments.device,
    )
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
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model.config.model_type,
        "model_class": type(model).__name__,
        "vocabulary_size": prompts[0]["logits_shape"][1],
        "provenance": provenance(model_dir, arguments),
        "prompts": prompts,
        "greedy": {"prompt_index": 0, "steps": greedy},
    }
    if arguments.decode_tokens_from is not None:
        prompt_ids = prompts[0]["token_ids"]
        tokens, source = teacher_tokens(arguments.decode_tokens_from, prompt_ids, manifest["vocabulary_size"])
        manifest["decode_consistency"] = decode_record(model, prompt_ids, tokens, source, staging)
    return manifest


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
        "--bf16-resident",
        action="store_true",
        help="load in BF16 and run each block in --dtype only while it executes (BF16 checkpoints only)",
    )
    parser.add_argument(
        "--no-special-tokens",
        dest="add_special_tokens",
        action="store_false",
        help="tokenize without the tokenizer's special tokens",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="cpu",
        help="device the loaded model runs on (loading and casting stay on the CPU)",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=ATTENTION_IMPLEMENTATIONS,
        default="eager",
        help="Transformers attention implementation",
    )
    parser.add_argument(
        "--decode-tokens-from",
        type=Path,
        help="truth bundle whose greedy tokens are teacher-forced to record decode self-consistency",
    )
    arguments = parser.parse_args(argv)
    if arguments.device == "mps" and arguments.dtype == "float64":
        parser.error("--device mps has no float64 kernels")
    if arguments.bf16_resident and (arguments.device, arguments.attn_implementation) != ("cpu", "eager"):
        parser.error("--bf16-resident evaluates the eager CPU graph only")
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
