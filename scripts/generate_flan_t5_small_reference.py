#!/usr/bin/env python3
"""Generate the immutable FLAN-T5-small v0.1 reference bundle.

This maintainer-only program executes ``google/flan-t5-small`` revision
``0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`` through the official
Transformers T5 implementation.  It never contacts a model hub: every input is
an authenticated regular file in a caller-supplied local directory.  The
result covers full-vocabulary prompt, cached, teacher-forced, padded-batch, and
512-token boundary logits plus deterministic greedy and beam continuations.

The output directory must be outside both this crate and the checkpoint.  A
complete bundle is written to a hidden staging directory, synced, and exposed
with a create-only atomic rename.  Existing bundles are never replaced.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
import platform
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# Disable implicit network and device fallbacks before any ML package import.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


REPOSITORY = "google/flan-t5-small"
REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
REFERENCE_CLASS = "transformers.T5ForConditionalGeneration"
REFERENCE_DEVICE = "cpu"
REFERENCE_DTYPE = "float32"
VOCAB_SIZE = 32_128
TOKENIZER_VOCAB_SIZE = 32_100
HIDDEN_SIZE = 512
MAX_SOURCE_TOKENS = 512
MAX_GREEDY_TOKENS = 32
BEAM_COUNT = 4
BEAM_MAX_NEW_TOKENS = 16
DECODER_START_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 0
EXPECTED_PHYSICAL_TENSORS = 190
EXPECTED_CHECKPOINT_TENSOR_BYTES = 307_844_608
EXPECTED_PARAMETER_COUNT = 76_961_152
MAX_HEADER_BYTES = 16 * 1024 * 1024
MIN_OUTPUT_FREE_BYTES = 64 * 1024 * 1024
SELF_PARITY_COSINE = 0.999_999
COMPATIBILITY_ATOL = 1.0e-4
COMPATIBILITY_RTOL = 1.0e-4

PROBE_INPUT_IDS = [71, 8, 57]
TRANSLATION_PROMPT = "translate English to German: The house is wonderful."
TRANSLATION_INPUT_IDS = [13_959, 1_566, 12, 2_968, 10, 37, 629, 19, 1_627, 5, 1]
TRANSLATION_BEAM_IDS = [644, 4_598, 229, 9_685, 5, 1]
PADDED_SECOND_PROMPT = "summarize: Transformers use attention."
PADDED_SECOND_INPUT_IDS = [21_603, 10, 31_220, 7, 169, 1_388, 5, 1]
LONG_FORM_PROMPT = (
    "translate English to German: In the early morning, the scientist walked slowly "
    "through the quiet garden, carefully recording how sunlight touched every leaf, "
    "how birds moved between the branches, and how the cool air changed as the city "
    "began to wake around her."
)
LONG_FORM_INPUT_IDS = [
    13_959, 1_566, 12, 2_968, 10, 86, 8, 778, 1_379, 6, 8, 17_901, 3,
    6_412, 5_665, 190, 8, 4_354, 2_004, 6, 4_321, 5_592, 149, 16_285,
    15_880, 334, 8_384, 6, 149, 6_331, 2_301, 344, 8, 9_678, 6, 11,
    149, 8, 1_633, 799, 2_130, 38, 8, 690, 1_553, 12, 7_178, 300, 160,
    5, 1,
]

RAW_FILES = {
    "encoder_final": "flan-t5-small-0fc9ddf7-encoder-final.f32le",
    "decoder_first_final": "flan-t5-small-0fc9ddf7-decoder-first-final.f32le",
    "first_logits": "flan-t5-small-0fc9ddf7-first-logits.f32le",
    "cached_logits": "flan-t5-small-0fc9ddf7-cached-logits-32x32128.f32le",
    "teacher_logits": "flan-t5-small-0fc9ddf7-teacher-logits-32x32128.f32le",
    "padded_first_logits": "flan-t5-small-0fc9ddf7-padded-first-logits.f32le",
    "boundary_first_logits": "flan-t5-small-0fc9ddf7-boundary-first-logits.f32le",
}
JSON_FILENAME = "flan-t5-small-0fc9ddf7-reference.json"
BUNDLE_DIRNAME = "flan-t5-small-0fc9ddf7-reference.bundle"

EXPECTED_FILES = {
    "config.json": (
        1_401,
        "439aa0fecf5a5546a1def68b1fc45e538e2c94528ce805378daf091e2bf6e4de",
    ),
    "generation_config.json": (
        147,
        "f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a",
    ),
    "model.safetensors": (
        307_867_048,
        "495fa51e204676f1a857a9fc13c4c89f3f5ba9f480b898cebca02add25e6d749",
    ),
    "special_tokens_map.json": (
        2_201,
        "5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f",
    ),
    "spiece.model": (
        791_656,
        "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
    ),
    "tokenizer.json": (
        2_424_064,
        "fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a",
    ),
    "tokenizer_config.json": (
        2_539,
        "fcde0f79bffda3688119c94330866a8fbf8de20ae65a8c492c9bd47c704655a0",
    ),
}
ALLOWED_ROOT_DIRECTORIES = {".cache"}

EXPECTED_PACKAGES = {
    "numpy": "2.4.3",
    "safetensors": "0.7.0",
    "tokenizers": "0.22.2",
    "torch": "2.10.0",
    "transformers": "5.2.0",
}


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest for exact in-memory bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash one regular file incrementally without artifact-sized allocation."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize deterministic JSON while rejecting non-finite float spellings."""

    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def json_without_duplicate_keys(path: Path) -> Any:
    """Parse UTF-8 JSON while rejecting ambiguous duplicate object keys."""

    def object_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=object_hook)


def json_bytes_without_duplicate_keys(payload: bytes, context: str) -> Any:
    """Parse UTF-8 JSON bytes while rejecting duplicate object keys."""

    def object_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {context}")
            result[key] = value
        return result

    return json.loads(payload.decode("utf-8"), object_pairs_hook=object_hook)


def regular_file(root: Path, name: str) -> Path:
    """Resolve one pinned basename without accepting symlinks or traversal."""

    relative = Path(name)
    if relative.is_absolute() or relative.name != name or ".." in relative.parts:
        raise ValueError(f"artifact path is not a local basename: {name!r}")
    unresolved = root / name
    metadata = unresolved.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact must be a nonsymlink regular file: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact escapes checkpoint root: {unresolved}")
    return resolved


def verify_artifacts(root: Path) -> list[dict[str, Any]]:
    """Authenticate the closed checkpoint inventory and return stable records."""

    observed = {entry.name for entry in root.iterdir()}
    allowed = set(EXPECTED_FILES) | ALLOWED_ROOT_DIRECTORIES
    if observed != allowed:
        raise ValueError(
            "checkpoint root inventory differs: "
            f"missing={sorted(allowed - observed)}, unexpected={sorted(observed - allowed)}"
        )
    records = []
    for directory in ALLOWED_ROOT_DIRECTORIES:
        path = root / directory
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"allowed root entry must be a nonsymlink directory: {path}")
    for name, (expected_bytes, expected_sha256) in sorted(EXPECTED_FILES.items()):
        path = regular_file(root, name)
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if (actual_bytes, actual_sha256) != (expected_bytes, expected_sha256):
            raise ValueError(
                f"artifact identity differs for {name}: expected "
                f"({expected_bytes}, {expected_sha256}), found ({actual_bytes}, {actual_sha256})"
            )
        records.append(
            {"relative_path": name, "bytes": actual_bytes, "sha256": actual_sha256}
        )
    return records


def canonical_records_digest(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash sorted path, length, and digest triples with unambiguous separators."""

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["relative_path"])):
        digest.update(str(record["relative_path"]).encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def verify_config(root: Path) -> None:
    """Require the exact executable T5-small topology and generation tokens."""

    config = json_without_duplicate_keys(root / "config.json")
    expected = {
        "model_type": "t5",
        "d_model": HIDDEN_SIZE,
        "d_kv": 64,
        "d_ff": 1_024,
        "num_layers": 8,
        "num_decoder_layers": 8,
        "num_heads": 6,
        "vocab_size": VOCAB_SIZE,
        "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": 128,
        "feed_forward_proj": "gated-gelu",
        "tie_word_embeddings": False,
        "decoder_start_token_id": DECODER_START_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "n_positions": MAX_SOURCE_TOKENS,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(
                f"config field {field!r} differs: expected {value!r}, found {config.get(field)!r}"
            )
    generation = json_without_duplicate_keys(root / "generation_config.json")
    for field, value in {
        "decoder_start_token_id": DECODER_START_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
    }.items():
        if generation.get(field) != value:
            raise ValueError(f"generation config field {field!r} differs")


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int, int]:
    """Read and structurally validate a bounded SafeTensors header envelope."""

    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("SafeTensors file has no complete header-length prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes == 0 or header_bytes > MAX_HEADER_BYTES:
            raise ValueError(f"SafeTensors header length is invalid: {header_bytes}")
        if 8 + header_bytes > file_bytes:
            raise ValueError("SafeTensors header exceeds file size")
        payload = handle.read(header_bytes)
    if len(payload) != header_bytes:
        raise ValueError("SafeTensors header read was truncated")
    header = json_bytes_without_duplicate_keys(payload, str(path))
    if not isinstance(header, dict):
        raise ValueError("SafeTensors header must be an object")
    return header, 8 + header_bytes, file_bytes


def verify_safetensors(root: Path) -> dict[str, Any]:
    """Validate F32 tensor count, shapes, and gap-free payload ranges."""

    header, data_start, file_bytes = read_safetensors_header(root / "model.safetensors")
    metadata = header.pop("__metadata__", None)
    if metadata != {"format": "pt"}:
        raise ValueError(f"unexpected SafeTensors metadata: {metadata!r}")
    if len(header) != EXPECTED_PHYSICAL_TENSORS:
        raise ValueError(f"expected {EXPECTED_PHYSICAL_TENSORS} tensors, found {len(header)}")
    ranges: list[tuple[int, int, str]] = []
    for name, tensor in header.items():
        if not isinstance(name, str) or not isinstance(tensor, dict):
            raise ValueError("SafeTensors entries must map string names to objects")
        if tensor.get("dtype") != "F32":
            raise ValueError(f"tensor {name!r} is not F32")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(value) is not int or value <= 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int or value < 0 for value in offsets)
        ):
            raise ValueError(f"tensor {name!r} has invalid shape or offsets")
        elements = math.prod(shape)
        start, end = offsets
        if end - start != elements * 4:
            raise ValueError(f"tensor {name!r} byte range disagrees with its F32 shape")
        ranges.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise ValueError(f"tensor {name!r} begins at {start}, expected {cursor}")
        cursor = end
    if cursor != EXPECTED_CHECKPOINT_TENSOR_BYTES or data_start + cursor != file_bytes:
        raise ValueError("SafeTensors payload length differs from the pinned checkpoint")
    inventory = [
        {
            "name": name,
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "data_offsets": tensor["data_offsets"],
        }
        for name, tensor in sorted(header.items())
    ]
    return {
        "header_bytes": data_start - 8,
        "tensors": len(header),
        "tensor_bytes": cursor,
        "inventory_sha256": sha256_bytes(strict_json_bytes({"tensors": inventory})),
    }


def verify_packages() -> list[dict[str, Any]]:
    """Require exact distribution versions and record their installed metadata."""

    records = []
    for name, expected_version in sorted(EXPECTED_PACKAGES.items()):
        distribution = importlib.metadata.distribution(name)
        if distribution.version != expected_version:
            raise RuntimeError(
                f"package {name} version differs: expected {expected_version}, "
                f"found {distribution.version}"
            )
        files = list(distribution.files or ())
        metadata_entry = next((entry for entry in files if entry.name == "METADATA"), None)
        record_entry = next((entry for entry in files if entry.name == "RECORD"), None)
        if metadata_entry is None or record_entry is None:
            raise RuntimeError(f"package {name} has no complete dist-info identity")
        identity: dict[str, Any] = {"name": name, "version": distribution.version}
        for label, entry in (("metadata", metadata_entry), ("record", record_entry)):
            path = Path(distribution.locate_file(entry)).resolve(strict=True)
            identity[label] = {
                "relative_path": str(entry),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        records.append(identity)
    return records


def git_generator_identity() -> dict[str, Any]:
    """Require a tracked, committed generator and capture its exact Git state."""

    script = Path(__file__).resolve(strict=True)
    crate = script.parents[1]
    repository = crate.parents[1]
    relative = script.relative_to(repository)
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=repository,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", str(relative)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(f"reference generator is dirty: {status.strip()}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "relative_path": str(relative),
        "bytes": script.stat().st_size,
        "sha256": sha256_file(script),
        "git_head": head,
        "git_dirty": False,
    }


def boundary_input_ids() -> list[int]:
    """Construct a deterministic 512-token nonpadding source ending in EOS."""

    prefix = [PROBE_INPUT_IDS[index % len(PROBE_INPUT_IDS)] for index in range(511)]
    return prefix + [EOS_TOKEN_ID]


def u32le_sha256(values: Sequence[int]) -> str:
    """Hash a token sequence in the Rust gate's canonical little-endian format."""

    return sha256_bytes(b"".join(struct.pack("<I", value) for value in values))


def top_k(values: Sequence[float], count: int) -> list[dict[str, Any]]:
    """Return stable descending logits with token-ID tie breaking."""

    if count <= 0 or count > len(values) or not all(math.isfinite(value) for value in values):
        raise ValueError("top-k requires a finite nonempty vector and valid count")
    winners = heapq.nsmallest(count, enumerate(values), key=lambda pair: (-pair[1], pair[0]))
    return [{"token_id": index, "logit": float(value)} for index, value in winners]


def tensor_payload(torch: Any, numpy: Any, tensor: Any, shape: Sequence[int]) -> tuple[bytes, list[float]]:
    """Convert one finite tensor to contiguous little-endian F32 bytes and values."""

    observed = list(tensor.shape)
    if observed != list(shape):
        raise RuntimeError(f"tensor shape changed: expected {list(shape)}, found {observed}")
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    array = numpy.asarray(array, dtype=numpy.dtype("<f4"), order="C")
    if not numpy.isfinite(array).all():
        raise RuntimeError("tensor contains non-finite values")
    return array.tobytes(order="C"), [float(value) for value in array.reshape(-1)]


def tensor_descriptor(payload: bytes, shape: Sequence[int], values: Sequence[float]) -> dict[str, Any]:
    """Describe a raw tensor with complete integrity and useful numerical probes."""

    expected_bytes = math.prod(shape) * 4
    if len(payload) != expected_bytes or len(values) != math.prod(shape):
        raise RuntimeError("raw F32 payload length disagrees with its shape")
    flattened = values
    summary: dict[str, Any] = {
        "finite": all(math.isfinite(value) for value in flattened),
        "l2_norm": math.sqrt(math.fsum(value * value for value in flattened)),
        "sum": math.fsum(flattened),
        "minimum": min(flattened),
        "maximum": max(flattened),
    }
    if shape[-1] == VOCAB_SIZE:
        summary["first_row_top10"] = top_k(flattened[:VOCAB_SIZE], 10)
    return {
        "encoding": "little_endian_f32",
        "dtype": "F32",
        "shape": list(shape),
        "elements": len(flattened),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "summary": summary,
    }


def cosine(torch: Any, left: Any, right: Any) -> float:
    """Compute an F64 cosine over two equal finite tensors."""

    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError("cosine tensors have different shapes")
    left = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    value = torch.nn.functional.cosine_similarity(left, right, dim=0, eps=1.0e-30)
    result = float(value.item())
    if not math.isfinite(result):
        raise RuntimeError("cosine is non-finite")
    return result


def parity_summary(torch: Any, left: Any, right: Any) -> dict[str, Any]:
    """Prove elementwise reference self-parity in addition to high cosine."""

    difference = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
    tolerance = COMPATIBILITY_ATOL + COMPATIBILITY_RTOL * right.detach().float().cpu().abs()
    failures = int((difference > tolerance).sum().item())
    similarity = cosine(torch, left, right)
    if failures or similarity < SELF_PARITY_COSINE:
        raise RuntimeError(
            f"reference paths disagree: failures={failures}, cosine={similarity}"
        )
    return {
        "cosine": similarity,
        "maximum_absolute_error": float(difference.max().item()),
        "failure_count_at_atol_rtol_1e_4": failures,
    }


def source_identity(module: Any) -> dict[str, Any]:
    """Record the exact official Python source file behind a loaded module."""

    path = Path(module.__file__).resolve(strict=True)
    return {"module": module.__name__, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def execute_reference(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Run official F32 inference and return raw payloads plus manifest fields."""

    import numpy
    import torch
    import transformers
    import transformers.generation.utils
    import transformers.models.t5.modeling_t5
    import transformers.tokenization_utils_tokenizers

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    tokenizer = transformers.T5TokenizerFast.from_pretrained(root, local_files_only=True)
    if len(tokenizer) != TOKENIZER_VOCAB_SIZE:
        raise RuntimeError(f"tokenizer vocabulary changed: {len(tokenizer)}")
    for prompt, expected in (
        (TRANSLATION_PROMPT, TRANSLATION_INPUT_IDS),
        (PADDED_SECOND_PROMPT, PADDED_SECOND_INPUT_IDS),
        (LONG_FORM_PROMPT, LONG_FORM_INPUT_IDS),
    ):
        observed = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if observed != expected:
            raise RuntimeError(f"tokenizer IDs changed for {prompt!r}: {observed}")

    started = time.monotonic()
    model, loading_info = transformers.T5ForConditionalGeneration.from_pretrained(
        root,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    load_seconds = time.monotonic() - started
    model.eval()
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"official checkpoint load was not strict: {loading_info}")
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("official runtime parameter count changed")
    # Transformers 5 normalizes the legacy flag, but separately records whether
    # decoder outputs are scaled.  The released FLAN checkpoint has an untied,
    # unscaled LM head even though the normalized compatibility flag is true.
    if not model.config.tie_word_embeddings or model.config.scale_decoder_outputs:
        raise RuntimeError("Transformers T5 legacy head normalization changed")
    if model.shared.weight.data_ptr() == model.lm_head.weight.data_ptr():
        raise RuntimeError("FLAN-T5-small LM head unexpectedly aliases shared embeddings")

    device = torch.device(REFERENCE_DEVICE)
    model.to(device)
    source_ids = torch.tensor([LONG_FORM_INPUT_IDS], dtype=torch.long, device=device)
    source_mask = torch.ones_like(source_ids)
    start_ids = torch.tensor([[DECODER_START_TOKEN_ID]], dtype=torch.long, device=device)
    payloads: dict[str, bytes] = {}
    descriptors: dict[str, Any] = {}

    with torch.inference_mode():
        started = time.monotonic()
        encoder = model.encoder(
            input_ids=source_ids,
            attention_mask=source_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        first = model(
            encoder_outputs=encoder,
            attention_mask=source_mask,
            decoder_input_ids=start_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        prompt_seconds = time.monotonic() - started

        generated: list[int] = []
        margins: list[float] = []
        cached_rows = []
        past = None
        next_input = start_ids
        started = time.monotonic()
        for _step in range(MAX_GREEDY_TOKENS):
            step = model(
                encoder_outputs=encoder,
                attention_mask=source_mask,
                decoder_input_ids=next_input,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = step.logits[:, -1, :]
            values, indices = torch.topk(logits, 2, dim=-1, largest=True, sorted=True)
            token = int(indices[0, 0].item())
            generated.append(token)
            margins.append(float((values[0, 0] - values[0, 1]).item()))
            cached_rows.append(logits)
            past = step.past_key_values
            next_input = indices[:, :1]
        cached_seconds = time.monotonic() - started
        if EOS_TOKEN_ID in generated:
            raise RuntimeError("long-form prompt reached EOS before 32 decisions")
        if min(margins) <= 0.25:
            raise RuntimeError(f"greedy fixture is not stable enough: minimum margin {min(margins)}")
        cached_logits = torch.cat(cached_rows, dim=0)

        teacher_ids = torch.tensor(
            [[DECODER_START_TOKEN_ID] + generated[:-1]], dtype=torch.long, device=device
        )
        started = time.monotonic()
        teacher = model(
            encoder_outputs=encoder,
            attention_mask=source_mask,
            decoder_input_ids=teacher_ids,
            use_cache=False,
            return_dict=True,
        )
        teacher_seconds = time.monotonic() - started
        teacher_logits = teacher.logits[0]
        cached_teacher_parity = parity_summary(torch, cached_logits, teacher_logits)
        if cached_logits.argmax(-1).tolist() != generated or teacher_logits.argmax(-1).tolist() != generated:
            raise RuntimeError("cached or teacher-forced top-1 differs from greedy continuation")

        padded_ids = torch.tensor(
            [TRANSLATION_INPUT_IDS, PADDED_SECOND_INPUT_IDS + [PAD_TOKEN_ID] * 3],
            dtype=torch.long,
            device=device,
        )
        padded_mask = torch.tensor(
            [[1] * 11, [1] * 8 + [0] * 3], dtype=torch.long, device=device
        )
        padded = model(
            input_ids=padded_ids,
            attention_mask=padded_mask,
            decoder_input_ids=torch.zeros((2, 1), dtype=torch.long, device=device),
            use_cache=False,
            return_dict=True,
        ).logits[:, -1, :]
        individual_rows = []
        for row_ids in (TRANSLATION_INPUT_IDS, PADDED_SECOND_INPUT_IDS):
            row = torch.tensor([row_ids], dtype=torch.long, device=device)
            individual_rows.append(
                model(
                    input_ids=row,
                    attention_mask=torch.ones_like(row),
                    decoder_input_ids=start_ids,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, -1, :]
            )
        padded_parity = parity_summary(torch, padded, torch.cat(individual_rows, dim=0))

        boundary_ids = boundary_input_ids()
        boundary = torch.tensor([boundary_ids], dtype=torch.long, device=device)
        started = time.monotonic()
        boundary_logits = model(
            input_ids=boundary,
            attention_mask=torch.ones_like(boundary),
            decoder_input_ids=start_ids,
            use_cache=False,
            return_dict=True,
        ).logits[:, -1, :]
        boundary_seconds = time.monotonic() - started

        beam = model.generate(
            input_ids=torch.tensor([TRANSLATION_INPUT_IDS], dtype=torch.long, device=device),
            attention_mask=torch.ones((1, len(TRANSLATION_INPUT_IDS)), dtype=torch.long, device=device),
            max_new_tokens=BEAM_MAX_NEW_TOKENS,
            num_beams=BEAM_COUNT,
            num_return_sequences=1,
            length_penalty=1.0,
            early_stopping=False,
            do_sample=False,
        )[0].tolist()
        if beam != [DECODER_START_TOKEN_ID] + TRANSLATION_BEAM_IDS:
            raise RuntimeError(f"four-beam continuation changed: {beam}")

    tensors = {
        "encoder_final": (encoder.last_hidden_state, [1, len(LONG_FORM_INPUT_IDS), HIDDEN_SIZE]),
        "decoder_first_final": (first.decoder_hidden_states[-1], [1, 1, HIDDEN_SIZE]),
        "first_logits": (first.logits, [1, 1, VOCAB_SIZE]),
        "cached_logits": (cached_logits, [MAX_GREEDY_TOKENS, VOCAB_SIZE]),
        "teacher_logits": (teacher_logits, [MAX_GREEDY_TOKENS, VOCAB_SIZE]),
        "padded_first_logits": (padded, [2, VOCAB_SIZE]),
        "boundary_first_logits": (boundary_logits, [1, VOCAB_SIZE]),
    }
    for name, (tensor, shape) in tensors.items():
        payload, values = tensor_payload(torch, numpy, tensor, shape)
        payloads[RAW_FILES[name]] = payload
        descriptors[name] = {"relative_path": RAW_FILES[name], **tensor_descriptor(payload, shape, values)}

    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    return payloads, {
        "reference_sources": [
            source_identity(transformers.models.t5.modeling_t5),
            source_identity(transformers.generation.utils),
            source_identity(transformers.tokenization_utils_tokenizers),
        ],
        "runtime_config": {
            "serialized_tie_word_embeddings": False,
            "normalized_tie_word_embeddings": bool(model.config.tie_word_embeddings),
            "scale_decoder_outputs": bool(model.config.scale_decoder_outputs),
            "lm_head_is_independent": model.shared.weight.data_ptr() != model.lm_head.weight.data_ptr(),
        },
        "loading_info": {key: sorted(value) if isinstance(value, set) else value for key, value in loading_info.items()},
        "tensors": descriptors,
        "generation": {
            "prompt": LONG_FORM_PROMPT,
            "source_token_ids": LONG_FORM_INPUT_IDS,
            "decoder_start_token_id": DECODER_START_TOKEN_ID,
            "max_new_tokens": MAX_GREEDY_TOKENS,
            "token_ids": generated,
            "decoded_text": decoded,
            "margins": margins,
            "minimum_margin": min(margins),
            "first_eos_position": None,
            "cached_teacher_parity": cached_teacher_parity,
        },
        "beam_generation": {
            "prompt": TRANSLATION_PROMPT,
            "source_token_ids": TRANSLATION_INPUT_IDS,
            "num_beams": BEAM_COUNT,
            "max_new_tokens": BEAM_MAX_NEW_TOKENS,
            "token_ids": TRANSLATION_BEAM_IDS,
            "decoded_text": tokenizer.decode(TRANSLATION_BEAM_IDS, skip_special_tokens=True),
            "finish_reason": f"eos:{EOS_TOKEN_ID}",
        },
        "padded_batch": {
            "source_token_ids": [TRANSLATION_INPUT_IDS, PADDED_SECOND_INPUT_IDS + [PAD_TOKEN_ID] * 3],
            "valid_mask": [[True] * 11, [True] * 8 + [False] * 3],
            "individual_parity": padded_parity,
        },
        "context_boundary": {
            "source_length": MAX_SOURCE_TOKENS,
            "pattern": PROBE_INPUT_IDS,
            "terminal_token_id": EOS_TOKEN_ID,
            "token_ids_u32le_sha256": u32le_sha256(boundary_input_ids()),
        },
        "timings_seconds": {
            "load": load_seconds,
            "prompt": prompt_seconds,
            "cached_generation_32": cached_seconds,
            "teacher_forced_32": teacher_seconds,
            "context_boundary_512": boundary_seconds,
        },
    }


def ensure_paths(model_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve external roots and require that the final bundle is absent."""

    model = model_dir.expanduser().resolve(strict=True)
    output = output_dir.expanduser().resolve(strict=True)
    crate = Path(__file__).resolve().parents[1]
    if not model.is_dir() or not output.is_dir():
        raise ValueError("model and output paths must be existing directories")
    if output == crate or output.is_relative_to(crate):
        raise ValueError("output directory must be external to the crate")
    if output == model or output.is_relative_to(model):
        raise ValueError("output directory must be outside the checkpoint")
    if shutil.disk_usage(output).free < MIN_OUTPUT_FREE_BYTES:
        raise OSError(f"output directory has less than {MIN_OUTPUT_FREE_BYTES} free bytes")
    bundle = output / BUNDLE_DIRNAME
    if os.path.lexists(bundle):
        raise FileExistsError(f"create-only reference bundle already exists: {bundle}")
    return model, output, bundle


def fsync_directory(path: Path) -> None:
    """Persist directory entries before or after publication."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_directory_create_only(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing path."""

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(source), os.fsencode(destination), 0x0000_0004)
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 0x0000_0001)
    else:
        raise RuntimeError(f"atomic no-replace rename is unsupported on {sys.platform}")
    if result != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


def publish(output: Path, payloads: Mapping[str, bytes]) -> Path:
    """Durably expose a complete fixed-name bundle through one atomic rename."""

    required = set(RAW_FILES.values()) | {JSON_FILENAME}
    if set(payloads) != required or any(Path(name).name != name for name in payloads):
        raise ValueError("reference payload inventory is incomplete or contains paths")
    destination = output / BUNDLE_DIRNAME
    if os.path.lexists(destination):
        raise FileExistsError(f"create-only reference bundle already exists: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{BUNDLE_DIRNAME}.staging-", dir=output))
    renamed = False
    try:
        for name in sorted(payloads):
            path = staging / name
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payloads[name])
                handle.flush()
                os.fsync(handle.fileno())
        fsync_directory(staging)
        rename_directory_create_only(staging, destination)
        renamed = True
        fsync_directory(output)
    except BaseException as error:
        location = destination if renamed else staging
        state = "published bundle" if renamed else "recoverable reference staging directory"
        error.add_note(f"{state}: {location}")
        raise
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit local model and external output directories."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Authenticate inputs, execute the official graph, and publish one bundle."""

    args = parse_args(argv)
    model, output, destination = ensure_paths(args.model_dir, args.output_dir)
    artifacts = verify_artifacts(model)
    verify_config(model)
    checkpoint = verify_safetensors(model)
    packages = verify_packages()
    generator = git_generator_identity()
    preflight = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "destination": str(destination),
        "artifact_manifest_sha256": canonical_records_digest(artifacts),
        "checkpoint": checkpoint,
        "packages": {record["name"]: record["version"] for record in packages},
        "generator": generator,
    }
    if args.preflight_only:
        print(json.dumps(preflight, allow_nan=False, indent=2, sort_keys=True))
        return 0

    raw_payloads, reference = execute_reference(model)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "reference_class": REFERENCE_CLASS,
        "execution": {
            "device": REFERENCE_DEVICE,
            "dtype": REFERENCE_DTYPE,
            "deterministic_algorithms": True,
            "seed": 0,
            "local_files_only": True,
            "use_safetensors": True,
        },
        "environment": {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
                "platform": platform.platform(),
            },
            "packages": packages,
            "offline_environment": {
                key: os.environ[key]
                for key in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                    "TOKENIZERS_PARALLELISM",
                    "HF_DEACTIVATE_ASYNC_LOAD",
                    "PYTORCH_ENABLE_MPS_FALLBACK",
                )
            },
        },
        "provenance": {
            "generator": generator,
            "artifacts": artifacts,
            "artifact_manifest_sha256": canonical_records_digest(artifacts),
            "checkpoint": checkpoint,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        **reference,
    }
    manifest_bytes = strict_json_bytes(manifest)
    payloads = {**raw_payloads, JSON_FILENAME: manifest_bytes}
    published = publish(output, payloads)
    # A post-publication read proves the visible directory contains exactly the
    # bytes that were authenticated before rename.
    if {entry.name for entry in published.iterdir()} != set(payloads):
        raise RuntimeError("published reference inventory changed after rename")
    for name, expected in payloads.items():
        if sha256_file(published / name) != sha256_bytes(expected):
            raise RuntimeError(f"published payload hash changed for {name}")
    print(
        json.dumps(
            {
                "bundle": str(published),
                "manifest_sha256": sha256_bytes(manifest_bytes),
                "payload_bytes": sum(len(payload) for payload in raw_payloads.values()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
