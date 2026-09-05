#!/usr/bin/env python3
"""Generate the pinned Muse-Glimmer-30B text-only MPS F16 oracle bundle.

This is an offline maintainer tool, not Rust runtime code.  It loads the exact
official multimodal class from a clean local Transformers checkout pinned to
``cfa309cfa34b5497a32fd50af49aa97aac94c054`` and exercises only its language
path.  It records one full-vocabulary prompt row, 54 last-token hidden trace
rows, 32 cached greedy decisions, and the corresponding stateless
teacher-forced top-1 decisions.

The three output files are fsynced inside a hidden sibling staging directory
and atomically published with no-replace semantics.  Interrupted runs expose no
partial final bundle and retain staging evidence for diagnosis.
"""

from __future__ import annotations

import argparse
import array
import ctypes
import datetime as dt
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


# These guards are installed before any optional ML import.  Every model and
# tokenizer load also passes local_files_only=True and trust_remote_code=False.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["USE_HUB_KERNELS"] = "0"


REPOSITORY = "meta-models/Muse-Glimmer-30B"
REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
REFERENCE_REPOSITORY = "huggingface/transformers"
REFERENCE_REVISION = "cfa309cfa34b5497a32fd50af49aa97aac94c054"
REFERENCE_CLASS = "transformers.MuseGlimmerForConditionalGeneration"
REFERENCE_SOURCE = "src/transformers/models/muse_glimmer/modeling_muse_glimmer.py"
REFERENCE_SOURCE_BYTES = 53_259
REFERENCE_SOURCE_SHA256 = (
    "459498a758446f580fb4a9203c64ab4be68488b40a04acd379dd62658d5bad71"
)

PROMPT = "Hello world, cached decoding."
INPUT_IDS = [19_873, 3_817, 24, 41_346, 63_623, 26]
VOCAB_SIZE = 202_048
HIDDEN_SIZE = 6_656
HIDDEN_TRACE_ROWS = 54
GREEDY_TOKENS = 32
EXPECTED_EOS_TOKEN_IDS = [200_001, 200_008]
EXPECTED_PARAMETER_COUNT = 29_776_626_688
EXPECTED_CHECKPOINT_TENSOR_BYTES = 59_553_253_376
EXPECTED_PHYSICAL_TENSORS = 1_436
SELF_PARITY_COSINE = 0.999
TOP1_STABILITY_SAFETY_FACTOR = 2.0
MAX_EXACT_JSON_INTEGER = (1 << 53) - 1

JSON_FILENAME = "muse-glimmer-30b-a4e59da-reference.json"
LOGITS_FILENAME = "muse-glimmer-30b-a4e59da-prompt-logits.f32le"
HIDDEN_FILENAME = "muse-glimmer-30b-a4e59da-hidden-last.f32le"
BUNDLE_DIRNAME = "muse-glimmer-30b-a4e59da-reference.bundle"

EXPECTED_DEPENDENCY_VERSIONS = {
    "accelerate": "1.12.0",
    "numpy": "2.4.3",
    "safetensors": "0.7.0",
    "tokenizers": "0.22.2",
    "torch": "2.10.0",
}
EXPECTED_TRANSFORMERS_VERSION = "5.15.0.dev0"

EXPECTED_ARTIFACTS = {
    "chat_template.jinja": (
        9_992,
        "cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678",
    ),
    "config.json": (
        5_109,
        "5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14",
    ),
    "generation_config.json": (
        202,
        "1fa51889b1f8d3659802dedaa27e005b81e5c58483f13ecf13f2d97306bc6e35",
    ),
    "model.safetensors.index.json": (
        132_674,
        "7d817b4dccb1b123fc6c1939356c65cee3a0ad462a5b821ac88280990a27d1ba",
    ),
    "processor_config.json": (
        1_084,
        "97e2a486dd9866b81f40cf4b8bc0c9ced9a7cd8a5bc65aa4cc2f4de0712dae77",
    ),
    "tokenizer.json": (
        28_129_897,
        "c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d",
    ),
    "tokenizer_config.json": (
        79_936,
        "781e6c74f571642c71202167b67d9255b28cc439bdda1582ff31346182f5a9c5",
    ),
    "model-00001-of-00002.safetensors": (
        49_950_112_952,
        "8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694",
    ),
    "model-00002-of-00002.safetensors": (
        9_603_322_320,
        "b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381",
    ),
}
EXPECTED_ARTIFACT_MANIFEST_SHA256 = (
    "5b8265f3a0fe6b07668caa8e5b949667b8f26f664550dfcc86d35f93fb7971ab"
)

ALLOCATOR_WARMUP_TARGET = "transformers.modeling_utils.caching_allocator_warmup"
ALLOCATOR_WARMUP_POLICY = "mps-exclusive-noop-during-from-pretrained"
ALLOCATOR_WARMUP_SOURCE_MODULE = "transformers.modeling_utils"
EXPECTED_ALLOCATOR_WARMUP_CALLS = 1
MINIMUM_MPS_LOAD_HEADROOM_BYTES = 2_000_000_000


def sha256_bytes(payload: bytes) -> str:
    """Return the canonical lowercase SHA-256 digest for bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Stream a regular file into SHA-256 without a shard-sized allocation."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def json_without_duplicate_keys(path: Path) -> Any:
    """Parse one JSON file while rejecting ambiguous duplicate object keys."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicates)


def checked_json_nonnegative_integer(value: Any, field: str) -> int:
    """Accept an exact interoperable JSON integer and reject bools or rounding."""

    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError(f"{field} must be a JSON integer, found {value!r}")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"{field} must be exactly integral, found {value!r}")
    if value < 0 or value > MAX_EXACT_JSON_INTEGER:
        raise ValueError(
            f"{field} must lie in 0..={MAX_EXACT_JSON_INTEGER}, found {value!r}"
        )
    result = int(value)
    if isinstance(value, float) and float(result) != value:
        raise ValueError(f"{field} cannot be losslessly canonicalized")
    return result


def strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize stable human-readable JSON while rejecting NaN and infinity."""

    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def f32le_bytes(values: Iterable[float]) -> bytes:
    """Encode values as canonical little-endian IEEE-754 binary32 bytes."""

    floats = array.array("f", values)
    if floats.itemsize != 4:
        raise RuntimeError(f"platform float item size is {floats.itemsize}, expected 4")
    if sys.byteorder != "little":
        floats.byteswap()
    return floats.tobytes()


def top_k_records(values: Sequence[float], count: int) -> list[dict[str, float | int]]:
    """Return descending top-k records with token-ID tie breaking."""

    if count <= 0 or count > len(values):
        raise ValueError(f"top-k count {count} is outside 1..={len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("cannot rank non-finite logits")
    winners = heapq.nsmallest(count, enumerate(values), key=lambda item: (-item[1], item[0]))
    return [{"token_id": index, "logit": float(value)} for index, value in winners]


def vector_summary(values: Sequence[float]) -> dict[str, Any]:
    """Return deterministic integrity statistics for one finite vector."""

    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("vector must be nonempty and finite")
    total = math.fsum(float(value) for value in values)
    squared = math.fsum(float(value) * float(value) for value in values)
    return {
        "finite": True,
        "sum": total,
        "l2_norm": math.sqrt(squared),
        "top10": top_k_records(values, 10),
    }


def binary16_ulp(value: float) -> float:
    """Return wider adjacent binary16 spacing at an exact finite F16 value."""

    if not math.isfinite(value):
        raise ValueError("binary16 ULP is undefined for non-finite values")
    try:
        roundtrip = struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError as error:
        raise ValueError(f"value {value!r} lies outside finite binary16") from error
    if roundtrip != value:
        raise ValueError(f"value {value!r} is not exactly binary16")
    magnitude = abs(value)
    if magnitude < 2.0**-14:
        return 2.0**-24
    return math.ldexp(1.0, math.floor(math.log2(magnitude)) - 10)


def decision_record(
    step: int, top_ids: Sequence[int], top_values: Sequence[float], eos_ids: Sequence[int]
) -> dict[str, Any]:
    """Prove one greedy choice survives a one-ULP perturbation per score."""

    if len(top_ids) != 2 or len(top_values) != 2 or top_ids[0] == top_ids[1]:
        raise ValueError("greedy stability requires two distinct ranked tokens")
    if top_ids[0] in eos_ids:
        raise RuntimeError(f"greedy decision {step} reached EOS token {top_ids[0]}")
    margin = float(top_values[0]) - float(top_values[1])
    epsilon = max(binary16_ulp(float(top_values[0])), binary16_ulp(float(top_values[1])))
    ratio = margin / epsilon
    if not math.isfinite(ratio) or ratio <= TOP1_STABILITY_SAFETY_FACTOR:
        raise RuntimeError(
            f"greedy decision {step} is not stable: margin={margin}, "
            f"epsilon={epsilon}, ratio={ratio}"
        )
    return {
        "step": step,
        "top1_token_id": int(top_ids[0]),
        "top2_token_id": int(top_ids[1]),
        "top1_logit": float(top_values[0]),
        "top2_logit": float(top_values[1]),
        "margin": margin,
        "elementwise_f16_ulp": epsilon,
        "margin_to_epsilon_ratio": ratio,
        "stable_under_two_sided_one_ulp_budget": True,
    }


def canonical_artifact_digest(artifacts: Sequence[Mapping[str, Any]]) -> str:
    """Bind sorted artifact names, byte lengths, and digests together."""

    lines = [
        f"{item['relative_path']}\0{item['bytes']}\0{item['sha256']}\n"
        for item in sorted(artifacts, key=lambda item: str(item["relative_path"]))
    ]
    return sha256_bytes("".join(lines).encode("utf-8"))


def path_lexists(path: Path) -> bool:
    """Return whether a path or dangling symlink occupies a directory entry."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def write_create_only(path: Path, payload: bytes) -> None:
    """Create and fsync one file, never replacing an existing path."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def fsync_directory(path: Path) -> None:
    """Fsync directory metadata after creating files or renaming a bundle."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_directory_create_only(source: Path, destination: Path) -> None:
    """Atomically rename a sibling directory without replacing a destination."""

    if source.parent != destination.parent:
        raise ValueError("atomic publication requires sibling directories")
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise RuntimeError("Darwin libc does not expose renamex_np")
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x0000_0004)
    elif sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise RuntimeError("Linux libc does not expose renameat2")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 0x0000_0001)
    else:
        raise RuntimeError(f"atomic no-replace rename unsupported on {sys.platform!r}")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def publish_reference_bundle(output_dir: Path, payloads: Mapping[str, bytes]) -> Path:
    """Durably publish a complete fixed-name bundle through one atomic rename."""

    expected_names = {JSON_FILENAME, LOGITS_FILENAME, HIDDEN_FILENAME}
    if set(payloads) != expected_names:
        raise ValueError(f"reference payload set changed: {sorted(payloads)}")
    bundle = output_dir / BUNDLE_DIRNAME
    if path_lexists(bundle):
        raise FileExistsError(f"create-only reference bundle exists: {bundle}")
    staging = Path(tempfile.mkdtemp(prefix=f".{BUNDLE_DIRNAME}.staging-", dir=output_dir))
    try:
        for name in sorted(payloads):
            write_create_only(staging / name, payloads[name])
        os.chmod(staging, 0o755)
        fsync_directory(staging)
        rename_directory_create_only(staging, bundle)
        fsync_directory(output_dir)
    except BaseException as error:
        error.add_note(f"recoverable reference staging directory: {staging}")
        raise
    return bundle


def run_git(checkout: Path, *arguments: str) -> str:
    """Run a read-only Git query and return normalized UTF-8 output."""

    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def verify_transformers_checkout(checkout: Path) -> dict[str, Any]:
    """Require the exact clean official source commit and model-file identity."""

    checkout = checkout.expanduser().resolve(strict=True)
    if not checkout.is_dir():
        raise ValueError(f"Transformers checkout is not a directory: {checkout}")
    head = run_git(checkout, "rev-parse", "HEAD")
    if head != REFERENCE_REVISION:
        raise RuntimeError(f"Transformers HEAD changed: expected {REFERENCE_REVISION}, found {head}")
    status = run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("Transformers reference checkout must be completely clean")
    source = checkout / REFERENCE_SOURCE
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"reference source must be a regular non-symlink file: {source}")
    observed = {"relative_path": REFERENCE_SOURCE, "bytes": source.stat().st_size, "sha256": sha256_file(source)}
    expected = {
        "relative_path": REFERENCE_SOURCE,
        "bytes": REFERENCE_SOURCE_BYTES,
        "sha256": REFERENCE_SOURCE_SHA256,
    }
    if observed != expected:
        raise RuntimeError(f"official Muse source changed: expected {expected}, found {observed}")
    return {
        "repository": REFERENCE_REPOSITORY,
        "revision": head,
        "clean": True,
        "checkout": str(checkout),
        "source": observed,
    }


def verify_dependency_versions() -> dict[str, str]:
    """Require the prepared oracle environment's exact dependency versions."""

    observed = {
        name: importlib.metadata.version(name) for name in EXPECTED_DEPENDENCY_VERSIONS
    }
    if observed != EXPECTED_DEPENDENCY_VERSIONS:
        raise RuntimeError(
            f"reference dependencies changed: expected {EXPECTED_DEPENDENCY_VERSIONS}, found {observed}"
        )
    return observed


def verify_checkpoint(model_dir: Path) -> tuple[list[dict[str, Any]], list[int]]:
    """Authenticate every model artifact and validate exact topology metadata."""

    config = json_without_duplicate_keys(model_dir / "config.json")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("config must contain object-valued text_config")
    expected_schedule = [
        item for _ in range(13) for item in ("sliding_attention",) * 3 + ("full_attention",)
    ]
    expected_theta = [item for _ in range(13) for item in (500_000.0, 500_000.0, 500_000.0, 0)]
    expected_fields = {
        "model_type": "muse_glimmer_text",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": 52,
        "layer_types": expected_schedule,
        "layer_rope_theta": expected_theta,
    }
    for name, expected in expected_fields.items():
        if text.get(name) != expected:
            raise ValueError(f"text config {name!r} changed")
    generation = json_without_duplicate_keys(model_dir / "generation_config.json")
    eos_ids = generation.get("eos_token_id")
    if eos_ids != EXPECTED_EOS_TOKEN_IDS or any(type(item) is not int for item in eos_ids):
        raise ValueError(f"generation EOS IDs changed: {eos_ids!r}")
    index = json_without_duplicate_keys(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != EXPECTED_PHYSICAL_TENSORS:
        raise ValueError("checkpoint index tensor inventory changed")
    total_size = checked_json_nonnegative_integer(
        index.get("metadata", {}).get("total_size"), "metadata.total_size"
    )
    if total_size != EXPECTED_CHECKPOINT_TENSOR_BYTES:
        raise ValueError(f"checkpoint tensor bytes changed: {total_size}")
    expected_shards = {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    if set(weight_map.values()) != expected_shards:
        raise ValueError("checkpoint index shard names changed")

    artifacts = []
    for relative_path, (expected_bytes, expected_hash) in sorted(EXPECTED_ARTIFACTS.items()):
        path = model_dir / relative_path
        metadata = path.lstat()
        if path.is_symlink() or not path.is_file() or metadata.st_size != expected_bytes:
            raise RuntimeError(f"checkpoint artifact identity changed: {relative_path}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise RuntimeError(f"checkpoint artifact hash changed: {relative_path}")
        artifacts.append(
            {"relative_path": relative_path, "bytes": expected_bytes, "sha256": observed_hash}
        )
    digest = canonical_artifact_digest(artifacts)
    if digest != EXPECTED_ARTIFACT_MANIFEST_SHA256:
        raise RuntimeError(f"artifact manifest digest changed: {digest}")
    return artifacts, list(eos_ids)


def ensure_external_paths(
    model_dir: Path, transformers_dir: Path, output_dir: Path
) -> tuple[Path, Path, Path, Path]:
    """Resolve isolated inputs, an external output parent, and absent bundle."""

    model = model_dir.expanduser().resolve(strict=True)
    reference = transformers_dir.expanduser().resolve(strict=True)
    output = output_dir.expanduser().resolve(strict=True)
    crate = Path(__file__).resolve(strict=True).parents[1]
    for label, path in (("model", model), ("Transformers", reference), ("output", output)):
        if not path.is_dir():
            raise ValueError(f"{label} path is not a directory: {path}")
    if model == reference or model.is_relative_to(reference) or reference.is_relative_to(model):
        raise ValueError("model and Transformers checkouts must be disjoint")
    if output == crate or output.is_relative_to(crate):
        raise ValueError("reference output must remain outside the crate")
    if output in (model, reference) or output.is_relative_to(model) or output.is_relative_to(reference):
        raise ValueError("reference output must remain outside both inputs")
    bundle = output / BUNDLE_DIRNAME
    if path_lexists(bundle):
        raise FileExistsError(f"create-only reference bundle exists: {bundle}")
    return model, reference, output, bundle


def load_with_verified_mps_warmup_noop(
    modeling_utils: Any,
    torch: Any,
    expected_model_class: type,
    loader: Callable[[], Any],
) -> tuple[Any, dict[str, Any]]:
    """Disable only the oversized, performance-only MPS allocator warmup."""

    if getattr(modeling_utils, "__name__", None) != ALLOCATOR_WARMUP_SOURCE_MODULE:
        raise RuntimeError("allocator warmup owner changed")
    original = getattr(modeling_utils, "caching_allocator_warmup", None)
    if not callable(original):
        raise RuntimeError("allocator warmup callable is unavailable")
    calls = 0
    entries = 0

    def verified_noop(model: Any, expanded_device_map: Any, hf_quantizer: Any) -> None:
        nonlocal calls, entries
        calls += 1
        if calls > EXPECTED_ALLOCATOR_WARMUP_CALLS:
            raise RuntimeError("allocator warmup was called more than once")
        if type(model) is not expected_model_class or hf_quantizer is not None:
            raise RuntimeError("allocator warmup arguments changed")
        if not isinstance(expanded_device_map, dict) or not expanded_device_map:
            raise RuntimeError("allocator warmup device map must be nonempty")
        device_types = {torch.device(value).type for value in expanded_device_map.values()}
        if device_types != {"mps"}:
            raise RuntimeError(f"allocator warmup map escaped MPS: {sorted(device_types)}")
        entries = len(expanded_device_map)

    modeling_utils.caching_allocator_warmup = verified_noop
    remained_scoped = False
    try:
        loaded = loader()
        remained_scoped = modeling_utils.caching_allocator_warmup is verified_noop
    finally:
        modeling_utils.caching_allocator_warmup = original
    if not remained_scoped or modeling_utils.caching_allocator_warmup is not original:
        raise RuntimeError("allocator warmup override escaped its load scope")
    if calls != EXPECTED_ALLOCATOR_WARMUP_CALLS:
        raise RuntimeError(f"allocator warmup call count changed: {calls}")
    return loaded, {
        "target": ALLOCATOR_WARMUP_TARGET,
        "policy": ALLOCATOR_WARMUP_POLICY,
        "call_count": calls,
        "expanded_device_map_entries": entries,
        "expanded_device_types": ["mps"],
        "hf_quantizer_present": False,
        "restored": True,
    }


def loading_info_summary(loading_info: Mapping[str, Any]) -> dict[str, Any]:
    """Record strict empty loading diagnostics without trusting iteration order."""

    result = {}
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = sorted(str(value) for value in loading_info.get(name, []))
        payload = "".join(f"{value}\n" for value in values).encode("utf-8")
        result[name] = {"count": len(values), "sha256": sha256_bytes(payload)}
        if values:
            raise RuntimeError(f"strict model load reported {name}: {values[:8]}")
    return result


def git_generator_provenance(script: Path) -> dict[str, Any]:
    """Bind output to the exact clean committed generator source."""

    crate = script.resolve(strict=True).parents[1]
    repository = Path(run_git(crate, "rev-parse", "--show-toplevel")).resolve(strict=True)
    status = run_git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("reference generator repository must be completely clean")
    relative = script.resolve(strict=True).relative_to(crate).as_posix()
    payload = script.read_bytes()
    return {
        "relative_path": relative,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "git_head": run_git(repository, "rev-parse", "HEAD"),
        "git_dirty": False,
    }


def tensor_cosine(torch: Any, left: Any, right: Any) -> float:
    """Compute finite full-vector cosine with float64 CPU accumulation."""

    left = left.to(dtype=torch.float64)
    right = right.to(dtype=torch.float64)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    value = float((torch.dot(left, right) / denominator).item())
    if not math.isfinite(value):
        raise RuntimeError("cosine similarity became non-finite")
    return value


def capture_prompt_hidden_trace(
    torch: Any, language_model: Any, run: Callable[[], Any]
) -> tuple[Any, list[str], Any]:
    """Capture embed-norm, 52 post-layer, and final-norm last-token rows."""

    # Muse applies a parameter-free RMS normalization immediately after token
    # lookup. Burn's first diagnostic is that normalized embedding, so hook
    # ``embed_norm`` rather than the raw ``embed_tokens`` output.
    labels = ["embed_norm", *[f"layers.{index}" for index in range(52)], "norm"]
    modules = [language_model.embed_norm, *language_model.layers, language_model.norm]
    captured = []
    handles = []
    for label, module in zip(labels, modules, strict=True):
        def hook(_module: Any, _inputs: Any, output: Any, *, trace_label: str = label) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if list(tensor.shape) != [1, len(INPUT_IDS), HIDDEN_SIZE]:
                raise RuntimeError(f"hidden trace {trace_label} shape changed: {list(tensor.shape)}")
            captured.append((trace_label, tensor[0, -1].float().cpu().contiguous()))

        handles.append(module.register_forward_hook(hook))
    try:
        output = run()
    finally:
        for handle in handles:
            handle.remove()
    observed_labels = [label for label, _ in captured]
    if observed_labels != labels:
        raise RuntimeError(f"hidden trace order changed: {observed_labels}")
    trace = torch.stack([row for _, row in captured])
    if list(trace.shape) != [HIDDEN_TRACE_ROWS, HIDDEN_SIZE]:
        raise RuntimeError(f"hidden trace shape changed: {list(trace.shape)}")
    return output, labels, trace


def binary_descriptor(relative_path: str, payload: bytes, shape: Sequence[int]) -> dict[str, Any]:
    """Describe one finite F32LE tensor payload without embedding it in JSON."""

    elements = math.prod(shape)
    if len(payload) != elements * 4:
        raise ValueError(f"binary payload {relative_path} has inconsistent shape")
    return {
        "relative_path": relative_path,
        "encoding": "ieee754-f32-little-endian",
        "shape": list(shape),
        "elements": elements,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def checked_duration(label: str, value: float) -> float:
    """Reject invalid timing evidence before serialization."""

    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"{label} duration is invalid: {value!r}")
    return value


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately MPS-only, local-only command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--transformers-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pinned official graph and atomically publish its oracle bundle."""

    arguments = parse_arguments(argv)
    model_dir, transformers_dir, output_dir, bundle = ensure_external_paths(
        arguments.model_dir, arguments.transformers_dir, arguments.output_dir
    )
    reference_source = verify_transformers_checkout(transformers_dir)
    dependencies = verify_dependency_versions()
    artifacts, eos_ids = verify_checkpoint(model_dir)

    sys.path.insert(0, str(transformers_dir / "src"))
    import torch
    import transformers
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        DynamicCache,
        MuseGlimmerForConditionalGeneration,
    )
    from transformers import modeling_utils

    if Path(transformers.__file__).resolve(strict=True).parent != (transformers_dir / "src/transformers").resolve(strict=True):
        raise RuntimeError("imported Transformers package escaped the pinned checkout")
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Transformers version changed: expected {EXPECTED_TRANSFORMERS_VERSION}, "
            f"found {transformers.__version__}"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("Muse-Glimmer reference generation requires MPS")
    recommended_mps_memory = int(torch.mps.recommended_max_memory())
    required_mps_memory = (
        EXPECTED_CHECKPOINT_TENSOR_BYTES + MINIMUM_MPS_LOAD_HEADROOM_BYTES
    )
    if recommended_mps_memory < required_mps_memory:
        raise RuntimeError(
            "insufficient recommended MPS memory for the pinned F16 model: "
            f"recommended={recommended_mps_memory}, required={required_mps_memory}"
        )
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.mps.set_per_process_memory_fraction(1.0)
    torch.set_grad_enabled(False)

    config = AutoConfig.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, revision=REVISION
    )
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    load_started = time.perf_counter()
    loaded, warmup = load_with_verified_mps_warmup_noop(
        modeling_utils,
        torch,
        MuseGlimmerForConditionalGeneration,
        lambda: MuseGlimmerForConditionalGeneration.from_pretrained(
            model_dir,
            config=config,
            dtype=torch.float16,
            device_map={"": "mps"},
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
            revision=REVISION,
            use_safetensors=True,
            weights_only=True,
            output_loading_info=True,
        ),
    )
    model, loading_info = loaded
    torch.mps.synchronize()
    load_seconds = checked_duration("load", time.perf_counter() - load_started)
    if type(model) is not MuseGlimmerForConditionalGeneration:
        raise RuntimeError(f"loaded unexpected class {type(model)!r}")
    model.eval()
    if (
        model.config.text_config._attn_implementation != "eager"
        or model.config.vision_config._attn_implementation != "eager"
        or bool(getattr(model, "use_kernels", False))
    ):
        raise RuntimeError("loaded model escaped the official eager, kernel-free path")
    loading_summary = loading_info_summary(loading_info)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"loaded parameter count changed: {parameter_count}")
    dtype_elements: dict[str, int] = {}
    for parameter in model.parameters():
        dtype = str(parameter.dtype).removeprefix("torch.")
        dtype_elements[dtype] = dtype_elements.get(dtype, 0) + parameter.numel()
    if dtype_elements != {"float16": EXPECTED_PARAMETER_COUNT}:
        raise RuntimeError(f"loaded parameter dtypes changed: {dtype_elements}")
    torch.mps.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        revision=REVISION,
        use_fast=True,
    )
    observed_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    if observed_ids != INPUT_IDS:
        raise RuntimeError(f"pinned prompt tokenization changed: {observed_ids}")
    if tokenizer.eos_token_id != 200_001 or tokenizer.pad_token_id != 200_018:
        raise RuntimeError(
            "pinned tokenizer EOS/PAD identity changed: "
            f"{tokenizer.eos_token_id}/{tokenizer.pad_token_id}"
        )
    input_ids = torch.tensor([INPUT_IDS], dtype=torch.long, device="mps")
    prompt_mask = torch.ones_like(input_ids, dtype=torch.long, device="mps")

    with torch.inference_mode():
        stateless_started = time.perf_counter()
        stateless, trace_labels, hidden_trace = capture_prompt_hidden_trace(
            torch,
            model.model.language_model,
            lambda: model(
                input_ids=input_ids,
                attention_mask=prompt_mask,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            ),
        )
        torch.mps.synchronize()
        stateless_seconds = checked_duration(
            "stateless prompt", time.perf_counter() - stateless_started
        )
        if stateless.logits.dtype != torch.float16:
            raise RuntimeError(f"stateless logits dtype changed: {stateless.logits.dtype}")
        stateless_row = stateless.logits[0, -1].float().cpu().contiguous()
        if list(stateless_row.shape) != [VOCAB_SIZE] or not bool(torch.isfinite(stateless_row).all()):
            raise RuntimeError("stateless full-vocabulary row is invalid")

        generation_started = time.perf_counter()
        cached = model(
            input_ids=input_ids,
            attention_mask=prompt_mask,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        cache = cached.past_key_values
        generated: list[int] = []
        cached_rows = []
        decisions = []
        for step in range(GREEDY_TOKENS):
            if (
                cached.logits.dtype != torch.float16
                or not isinstance(cache, DynamicCache)
                or cache.get_seq_length() != len(INPUT_IDS) + step
            ):
                raise RuntimeError(f"cached output invalid at decision {step}")
            row = cached.logits[0, -1].float().cpu().contiguous()
            if list(row.shape) != [VOCAB_SIZE] or not bool(torch.isfinite(row).all()):
                raise RuntimeError(f"cached vocabulary row invalid at decision {step}")
            cached_rows.append(row)
            ranked = top_k_records(row.tolist(), 2)
            record = decision_record(
                step,
                [int(item["token_id"]) for item in ranked],
                [float(item["logit"]) for item in ranked],
                eos_ids,
            )
            token_id = int(record["top1_token_id"])
            decisions.append(record)
            generated.append(token_id)
            if step + 1 < GREEDY_TOKENS:
                cached = model(
                    input_ids=torch.tensor([[token_id]], dtype=torch.long, device="mps"),
                    attention_mask=torch.ones(
                        (1, len(INPUT_IDS) + step + 1), dtype=torch.long, device="mps"
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                cache = cached.past_key_values
        torch.mps.synchronize()
        generation_seconds = checked_duration(
            "cached generation", time.perf_counter() - generation_started
        )

        teacher_ids = INPUT_IDS + generated[:-1]
        teacher_started = time.perf_counter()
        teacher = model(
            input_ids=torch.tensor([teacher_ids], dtype=torch.long, device="mps"),
            attention_mask=torch.ones((1, len(teacher_ids)), dtype=torch.long, device="mps"),
            use_cache=False,
            logits_to_keep=0,
            return_dict=True,
        )
        torch.mps.synchronize()
        teacher_seconds = checked_duration(
            "teacher-forced stateless", time.perf_counter() - teacher_started
        )
        teacher_logits = teacher.logits[0].float().cpu().contiguous()
        positions = list(range(len(INPUT_IDS) - 1, len(INPUT_IDS) - 1 + GREEDY_TOKENS))
        stateless_top1 = [int(teacher_logits[position].argmax().item()) for position in positions]
        if stateless_top1 != generated:
            raise RuntimeError(
                f"official cached/stateless top-1 mismatch: {generated} != {stateless_top1}"
            )
        prefix_cosines = [
            tensor_cosine(torch, cached_row, teacher_logits[position])
            for cached_row, position in zip(cached_rows, positions, strict=True)
        ]
        if min(prefix_cosines) < SELF_PARITY_COSINE:
            raise RuntimeError(f"cached/stateless cosine below {SELF_PARITY_COSINE}")
        prompt_cached_cosine = tensor_cosine(torch, stateless_row, cached_rows[0])
        if prompt_cached_cosine < SELF_PARITY_COSINE:
            raise RuntimeError("prompt cached/stateless cosine is below the gate")

    logits_values = [float(value) for value in stateless_row.tolist()]
    hidden_values = [float(value) for value in hidden_trace.flatten().tolist()]
    logits_payload = f32le_bytes(logits_values)
    hidden_payload = f32le_bytes(hidden_values)
    logits_descriptor = binary_descriptor(LOGITS_FILENAME, logits_payload, [VOCAB_SIZE])
    logits_descriptor.update(vector_summary(logits_values))
    logits_descriptor["top1_token_id"] = int(stateless_row.argmax().item())
    hidden_descriptor = binary_descriptor(
        HIDDEN_FILENAME, hidden_payload, [HIDDEN_TRACE_ROWS, HIDDEN_SIZE]
    )
    hidden_descriptor["labels"] = trace_labels

    script = Path(__file__).resolve(strict=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    record = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "reference": {
            "repository": REFERENCE_REPOSITORY,
            "revision": REFERENCE_REVISION,
            "class": REFERENCE_CLASS,
            "source": reference_source["source"],
            "checkout_clean": reference_source["clean"],
        },
        "prompt": PROMPT,
        "input_ids": INPUT_IDS,
        "execution": {
            "dtype": "float16",
            "device": "mps",
            "attention_implementation": "eager",
            "attention_precision": {
                "score_equation": "(rms(q) * qk_scale_factor) @ rms(k)^T / sqrt(head_dim)",
                "softmax_accumulator_dtype": "float32",
                "probability_output_dtype": "float16",
            },
            "hub_kernels_enabled": False,
            "deterministic_algorithms": True,
            "seed": 0,
            "use_cache": True,
            "explicit_all_one_attention_masks": True,
            "float32_matmul_precision": "highest",
            "mps_per_process_memory_fraction": 1.0,
            "allocator_warmup_override": warmup,
        },
        "environment": {
            "generated_at_utc": now,
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": {**dependencies, "transformers": transformers.__version__},
            "device": {
                "backend_available": True,
                "recommended_max_memory_bytes": recommended_mps_memory,
                "required_f16_load_bytes": required_mps_memory,
                "minimum_load_headroom_bytes": MINIMUM_MPS_LOAD_HEADROOM_BYTES,
            },
            "offline_environment": {
                name: os.environ[name]
                for name in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                    "TOKENIZERS_PARALLELISM",
                    "HF_DEACTIVATE_ASYNC_LOAD",
                    "PYTORCH_ENABLE_MPS_FALLBACK",
                    "USE_HUB_KERNELS",
                )
            },
        },
        "provenance": {
            "generator": git_generator_provenance(script),
            "artifacts": artifacts,
            "artifact_manifest_sha256": canonical_artifact_digest(artifacts),
            "physical_tensor_count": EXPECTED_PHYSICAL_TENSORS,
            "checkpoint_tensor_bytes": EXPECTED_CHECKPOINT_TENSOR_BYTES,
            "loaded_parameter_count": parameter_count,
            "parameter_dtype_elements": dtype_elements,
            "loading_info": loading_summary,
        },
        "outputs": {
            "prompt_logits": logits_descriptor,
            "hidden_last_trace": hidden_descriptor,
        },
        "generation": {
            "max_new_tokens": GREEDY_TOKENS,
            "eos_token_ids": eos_ids,
            "token_ids": generated,
            "cached_prefix_top1": generated,
            "stateless_prefix_top1": stateless_top1,
            "decisions": decisions,
            "minimum_margin": min(float(item["margin"]) for item in decisions),
            "minimum_margin_to_epsilon_ratio": min(
                float(item["margin_to_epsilon_ratio"]) for item in decisions
            ),
            "decoded_text": tokenizer.decode(
                generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
            "prompt_cached_cosine": prompt_cached_cosine,
            "minimum_cached_stateless_prefix_cosine": min(prefix_cosines),
        },
        "timings_seconds": {
            "load": load_seconds,
            "stateless_prompt": stateless_seconds,
            "cached_generation": generation_seconds,
            "teacher_forced_stateless": teacher_seconds,
        },
    }
    json_payload = strict_json_bytes(record)
    published = publish_reference_bundle(
        output_dir,
        {
            JSON_FILENAME: json_payload,
            LOGITS_FILENAME: logits_payload,
            HIDDEN_FILENAME: hidden_payload,
        },
    )
    if published != bundle:
        raise RuntimeError(f"published unexpected bundle path: {published}")
    print(f"published={published}")
    print(f"json_sha256={sha256_bytes(json_payload)}")
    print(f"logits_sha256={sha256_bytes(logits_payload)}")
    print(f"hidden_sha256={sha256_bytes(hidden_payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
