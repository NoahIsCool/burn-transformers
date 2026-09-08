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

The pinned source, checkpoint, and 11 primary wheel payloads are content
authenticated. CPython/stdlib/OS payloads and recorded transitive distributions
remain explicit environment trust inputs; the generated record never calls
those bytes authenticated.
"""

from __future__ import annotations

import argparse
import array
import base64
import ctypes
import datetime as dt
import hashlib
import heapq
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import stat
import struct
import subprocess
import sys
import sysconfig
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


def _early_source_snapshot(path: Path) -> tuple[bytes, tuple[int, int, int, int, int]]:
    """Read one bounded Python source through a stable non-following descriptor."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("source authentication requires O_NOFOLLOW/O_NONBLOCK")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 2 * 1024 * 1024:
            raise RuntimeError(f"generator source is not a bounded regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            payload = handle.read(2 * 1024 * 1024 + 1)
            after = os.fstat(handle.fileno())
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if len(payload) != before.st_size or identity(before) != identity(after):
            raise RuntimeError(f"generator source changed during snapshot: {path}")
        return payload, identity(before)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


SELF_SOURCE_PATH = Path(__file__).resolve(strict=True)
_SELF_SOURCE_BYTES, _SELF_SOURCE_IDENTITY = _early_source_snapshot(SELF_SOURCE_PATH)
_recompiled_self = compile(
    _SELF_SOURCE_BYTES,
    sys._getframe().f_code.co_filename,
    "exec",
    dont_inherit=True,
    optimize=sys.flags.optimize,
)
if _recompiled_self != sys._getframe().f_code:
    raise RuntimeError(
        "executing Muse generator code differs from its authenticated source snapshot"
    )
del _recompiled_self


REPOSITORY = "meta-models/Muse-Glimmer-30B"
REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
REFERENCE_REPOSITORY = "huggingface/transformers"
REFERENCE_REVISION = "cfa309cfa34b5497a32fd50af49aa97aac94c054"
REFERENCE_CLASS = "transformers.MuseGlimmerForConditionalGeneration"
REFERENCE_SOURCE_VERSION = "5.17.0.dev0"
CHECKPOINT_CONFIG_TRANSFORMERS_VERSION = "5.15.0.dev0"
REFERENCE_SOURCE = "src/transformers/models/muse_glimmer/modeling_muse_glimmer.py"
REFERENCE_SOURCE_BYTES = 53_259
REFERENCE_SOURCE_SHA256 = (
    "459498a758446f580fb4a9203c64ab4be68488b40a04acd379dd62658d5bad71"
)

# This is the literal, already-rendered chat prompt. Pinning the expansion makes
# the oracle independent of the template's dynamic current-date default.
PROMPT = (
    "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\n"
    "Knowledge cutoff: 2026-01-04.\n"
    "Current date: 2026-09-08.\n\n"
    "Reasoning strength: high.\n\n"
    '# Valid recipients: "self", "user".<|eot|>'
    "<|start|>user<|message|>Explain why the sky appears blue in one concise paragraph."
    "<|eot|><|start|>assistant"
)
INPUT_IDS = [
    200_000,
    200_022,
    15_651,
    200_023,
    4_662,
    583,
    262,
    19_933,
    12_133,
    43_910,
    335,
    111_198,
    67_059,
    38,
    220,
    837,
    34,
    25,
    1_761,
    25,
    2_468,
    335,
    8_048,
    4_282,
    38,
    220,
    837,
    34,
    25,
    2_961,
    25,
    2_834,
    1_574,
    34_956,
    300,
    9_762,
    38,
    2_244,
    1_574,
    15,
    14_757,
    73_965,
    38,
    392,
    2_540,
    706,
    392,
    1_556,
    4_205,
    200_008,
    200_022,
    1_556,
    200_023,
    51_015,
    5_959,
    290,
    21_752,
    14_774,
    10_950,
    310,
    1_085,
    81_080,
    24_429,
    26,
    200_008,
    200_022,
    140_680,
]
VOCAB_SIZE = 202_048
HIDDEN_SIZE = 6_656
HIDDEN_TRACE_ROWS = 54
GREEDY_TOKENS = 32
EXPECTED_EOS_TOKEN_IDS = [200_001, 200_008]
EXPECTED_PARAMETER_COUNT = 29_776_626_688
EXPECTED_CHECKPOINT_TENSOR_BYTES = 59_553_253_376
EXPECTED_PHYSICAL_TENSORS = 1_436
EXPECTED_ALLOCATOR_WARMUP_ENTRIES = 1_436
SELF_PARITY_COSINE = 0.999
TOP1_STABILITY_SAFETY_FACTOR = 2.0
MAX_EXACT_JSON_INTEGER = (1 << 53) - 1
EXPECTED_PYTHON_VERSION = (3, 11, 9, "final", 0)

INTEGRITY_THREAT_MODEL = (
    "Pre/post descriptor identities and SHA-256 detect persistent mutation and "
    "ordinary races. An actively malicious same-UID process that swaps a pathname "
    "only while Transformers or SafeTensors reopens it, then restores the original "
    "before postflight, is outside scope because it can also tamper with this process."
)
INTERPRETER_TRUST_BOUNDARY = (
    "CPython implementation and 3.11.9 final version are enforced and the executable "
    "paths are recorded, but interpreter, standard-library, system-library, Git "
    "executable, and OS payload bytes are trusted environment inputs rather than "
    "content-authenticated."
)
TRANSITIVE_TRUST_BOUNDARY = (
    "The 11 listed primary wheels have exact versions and RECORD payload manifests. "
    "Every other imported third-party module must belong to a recorded installed "
    "distribution, but those transitive distribution payloads are trusted from the "
    "selected offline virtual environment rather than content-authenticated."
)

JSON_FILENAME = "muse-glimmer-30b-a4e59da-reference.json"
LOGITS_FILENAME = "muse-glimmer-30b-a4e59da-prompt-logits.f32le"
HIDDEN_FILENAME = "muse-glimmer-30b-a4e59da-hidden-last.f32le"
BUNDLE_DIRNAME = "muse-glimmer-30b-a4e59da-reference.bundle"

EXPECTED_DEPENDENCY_VERSIONS = {
    "accelerate": "1.12.0",
    "filelock": "3.17.0",
    "huggingface-hub": "1.5.0",
    "numpy": "2.4.3",
    "packaging": "25.0",
    "PyYAML": "6.0.2",
    "regex": "2026.1.15",
    "safetensors": "0.8.0",
    "tokenizers": "0.23.1",
    "torch": "2.10.0",
    "tqdm": "4.67.1",
}

PACKAGE_IMPORT_NAMES = {
    "accelerate": "accelerate",
    "filelock": "filelock",
    "huggingface-hub": "huggingface_hub",
    "numpy": "numpy",
    "packaging": "packaging",
    "PyYAML": "yaml",
    "regex": "regex",
    "safetensors": "safetensors",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "tqdm": "tqdm",
}

# Each tuple pins all non-bytecode, in-environment payloads listed by the
# wheel's RECORD: verified file count, verified bytes, ignored bytecode count,
# intentionally excluded external launcher paths, and the canonical SHA-256
# manifest. Bytecode cannot execute under the private-cache policy below, but
# its count is still pinned so RECORD coverage cannot drift silently. Launcher
# scripts are not imported by this direct-Python workflow and contain
# environment-specific shebangs.
EXPECTED_DISTRIBUTION_PAYLOADS = {
    "PyYAML": (24, 581_466, 18, (), "a2413d950416eb3e271ce2e79d0a0d9af772b2f509e4f6368ac21a21059951c8"),
    "accelerate": (
        90,
        1_482_046,
        84,
        (
            "../../../bin/accelerate",
            "../../../bin/accelerate-config",
            "../../../bin/accelerate-estimate-memory",
            "../../../bin/accelerate-launch",
            "../../../bin/accelerate-merge-weights",
        ),
        "a4777b667bf3cf092af2ac15403be4b1217f4eada9dea1c93729d0ba1fce374c",
    ),
    "filelock": (14, 42_043, 9, (), "fa010d76505458640b82b4245cc7e88a0279adaa5d35939a3f62632f86f41d9f"),
    "huggingface-hub": (
        174,
        2_379_101,
        164,
        ("../../../bin/hf", "../../../bin/tiny-agents"),
        "e31e33b4619feb0de20b31a105b3dd4a48a0b5c7082a879c39730cfab8950a38",
    ),
    "numpy": (
        1_038,
        20_253_880,
        487,
        ("../../../bin/f2py", "../../../bin/numpy-config"),
        "ea91cc1904675f4a171c29d31fda752911857c256eef4b463f79b5f2647aa17c",
    ),
    "packaging": (23, 236_488, 16, (), "28a5d31e3899005c3c4765b86bbdf2617439c46a113eba215187948e1914cebb"),
    "regex": (10, 1_248_895, 4, (), "39c375db510bf8720c449985de413ec2b506bc22b5cecf51d0a0209817537a4f"),
    "safetensors": (16, 1_211_818, 7, (), "cfa1be33517b651209e190e761fcdac179935a7df4fdbc8215bb20573cacb0b0"),
    "tokenizers": (33, 8_879_675, 16, (), "384ef072334a048d972eecdb2d86f4b7e359fc3ac7b1054b3d00adb926f2c0c7"),
    "torch": (
        12_338,
        338_441_672,
        2_199,
        ("../../../bin/torchfrtrace", "../../../bin/torchrun"),
        "ac5e7ce059b386aeecec89677baacc9875bad01557c4947ed98e05d14634b3c6",
    ),
    "tqdm": (40, 222_872, 32, ("../../../bin/tqdm",), "e821f85778c550267aa17a5c38b7db7e388892fbaf18cf996718c7d960be9704"),
}
EXPECTED_DISTRIBUTION_MANIFEST_SHA256 = (
    "6eca7252cebb0750fa344a7d05419a4511f3ecf985e6f893dadf3e0e94c2b2a6"
)
EXPECTED_ABSENT_DISTRIBUTIONS = ("flash-attn", "kernels", "torchao")
EXPECTED_PRIMARY_VIRTUAL_MODULES = {
    ("torch.classes", "_classes.py", "torch._classes", "_Classes"),
    ("torch.ops", "_ops.py", "torch._ops", "_Ops"),
}

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

# These four upstream files are semantically inert to local Transformers
# loading. They may be present when `hf download` fetches the complete pinned
# snapshot. The Hub's hidden local-directory cache is likewise allowed but is
# never traversed by this verifier or passed as the model root.
EXPECTED_INERT_ROOT_FILES = {
    ".gitattributes": 1_570,
    "LICENSE": 11_358,
    "README.md": 16_892,
    "USAGE_POLICY.md": 5_230,
}
ALLOWED_INERT_ROOT_DIRECTORIES = frozenset({".cache"})

ALLOCATOR_WARMUP_TARGET = "transformers.modeling_utils.caching_allocator_warmup"
ALLOCATOR_WARMUP_POLICY = "mps-official-noop-verified-during-from-pretrained"
ALLOCATOR_WARMUP_SOURCE_MODULE = "transformers.modeling_utils"
ALLOCATOR_WARMUP_SCOPE = "MuseGlimmerForConditionalGeneration.from_pretrained"
ALLOCATOR_WARMUP_REASON = (
    "Pinned Transformers 5.17 explicitly skips allocator warmup on MPS. The scoped "
    "wrapper validates the complete expanded parameter map, then delegates to that "
    "official no-op branch without changing model tensors or numerical execution."
)
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


def file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return descriptor identity fields used by pre/post integrity checks."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def authenticated_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
    retain_payload: bool = False,
    maximum_payload_bytes: int = 64 * 1024 * 1024,
) -> tuple[dict[str, Any], tuple[int, int, int, int, int], bytes | None]:
    """Hash a stable regular-file descriptor without following its final component."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("authenticated reads require O_NOFOLLOW/O_NONBLOCK")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"pinned input is not a regular file: {path}")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise RuntimeError(
                f"pinned input {path} has {before.st_size} bytes, expected {expected_bytes}"
            )
        if retain_payload and before.st_size > maximum_payload_bytes:
            raise RuntimeError(f"refusing to retain oversized pinned input: {path}")
        digest = hashlib.sha256()
        payload = bytearray() if retain_payload else None
        observed_bytes = 0
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while chunk := handle.read(8 * 1024 * 1024):
                observed_bytes += len(chunk)
                if observed_bytes > before.st_size:
                    raise RuntimeError(f"pinned input grew while being read: {path}")
                digest.update(chunk)
                if payload is not None:
                    payload.extend(chunk)
            after = os.fstat(handle.fileno())
        if observed_bytes != before.st_size or file_identity(after) != file_identity(before):
            raise RuntimeError(f"pinned input changed while being read: {path}")
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: expected {expected_sha256}, "
                f"found {observed_sha256}"
            )
        return (
            {"bytes": before.st_size, "sha256": observed_sha256},
            file_identity(before),
            bytes(payload) if payload is not None else None,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def assert_early_source_unchanged() -> None:
    """Reauthenticate the exact generator bytes immediately before publication."""

    _, identity, payload = authenticated_file(
        SELF_SOURCE_PATH,
        expected_sha256=hashlib.sha256(_SELF_SOURCE_BYTES).hexdigest(),
        expected_bytes=len(_SELF_SOURCE_BYTES),
        retain_payload=True,
        maximum_payload_bytes=2 * 1024 * 1024,
    )
    if identity != _SELF_SOURCE_IDENTITY or payload != _SELF_SOURCE_BYTES:
        raise RuntimeError("executed Muse generator source changed during the run")


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, Any]], label: str | Path
) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} in {label}")
        result[key] = value
    return result


def json_without_duplicate_keys(path: Path) -> Any:
    """Parse one JSON file while rejecting ambiguous duplicate object keys."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=lambda pairs: _reject_duplicate_pairs(pairs, path),
        )


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
    authenticated, identity, _ = authenticated_file(
        source,
        expected_sha256=REFERENCE_SOURCE_SHA256,
        expected_bytes=REFERENCE_SOURCE_BYTES,
    )
    observed = {"relative_path": REFERENCE_SOURCE, **authenticated}
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
        "source_identity": identity,
    }


def verify_python_runtime() -> dict[str, Any]:
    """Require the selected CPython version and record its unauthenticated paths."""

    observed = (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
        sys.version_info.releaselevel,
        sys.version_info.serial,
    )
    if platform.python_implementation() != "CPython" or observed != EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            f"Muse oracle requires CPython {EXPECTED_PYTHON_VERSION}, found "
            f"{platform.python_implementation()} {observed}"
        )
    executable = Path(sys.executable)
    if not executable.is_absolute() or not executable.exists():
        raise RuntimeError(f"Python executable path is not absolute and present: {executable}")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Muse oracle must run inside an explicit Python virtual environment")
    return {
        "version": sys.version,
        "version_info": {
            "major": observed[0],
            "minor": observed[1],
            "micro": observed[2],
            "releaselevel": observed[3],
            "serial": observed[4],
        },
        "executable": str(executable),
        "resolved_executable": str(executable.resolve(strict=True)),
        "prefix": str(Path(sys.prefix).resolve(strict=True)),
        "base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def canonical_distribution_digest(distributions: Mapping[str, Mapping[str, Any]]) -> str:
    """Bind all exact package versions and authenticated RECORD payload manifests."""

    lines = []
    for name in sorted(distributions):
        item = distributions[name]
        lines.append(
            f"{name}\0{item['version']}\0{item['verified_files']}\0"
            f"{item['verified_bytes']}\0{item['ignored_bytecode_files']}\0"
            f"{item['payload_manifest_sha256']}\n"
        )
        lines.extend(
            f"external\0{path}\n" for path in item["excluded_external_scripts"]
        )
    return sha256_bytes("".join(lines).encode("utf-8"))


def verify_distribution_payloads() -> tuple[
    dict[str, str],
    dict[str, dict[str, Any]],
    str,
    dict[str, dict[str, tuple[int, int, int, int, int]]],
]:
    """Authenticate every importable payload declared by each exact wheel RECORD."""

    versions: dict[str, str] = {}
    summaries: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, tuple[int, int, int, int, int]]] = {}
    for name in sorted(EXPECTED_DEPENDENCY_VERSIONS):
        distribution = importlib.metadata.distribution(name)
        version = distribution.version
        if version != EXPECTED_DEPENDENCY_VERSIONS[name]:
            raise RuntimeError(
                f"reference package {name!r} changed: expected "
                f"{EXPECTED_DEPENDENCY_VERSIONS[name]}, found {version}"
            )
        entries = distribution.files
        if not entries:
            raise RuntimeError(f"distribution {name!r} exposes no RECORD entries")
        canonical = bytearray()
        package_identities: dict[str, tuple[int, int, int, int, int]] = {}
        verified_files = 0
        verified_bytes = 0
        excluded_external_scripts = []
        ignored_bytecode_files = 0
        record_entries = 0
        for entry in sorted(entries, key=str):
            relative_path = str(entry)
            if relative_path.endswith(".pyc") and "/__pycache__/" in f"/{relative_path}":
                ignored_bytecode_files += 1
                continue
            if relative_path.startswith("../"):
                excluded_external_scripts.append(relative_path)
                continue
            if entry.name == "RECORD" and entry.parent.name.endswith(".dist-info"):
                if entry.hash is not None or entry.size is not None:
                    raise RuntimeError(f"distribution {name!r} RECORD self-entry changed")
                record_entries += 1
                continue
            if entry.hash is None or entry.hash.mode != "sha256" or entry.size is None:
                raise RuntimeError(
                    f"distribution {name!r} has an unauthenticated RECORD entry: "
                    f"{relative_path!r}"
                )
            padded = entry.hash.value + "=" * (-len(entry.hash.value) % 4)
            try:
                expected_hash = base64.urlsafe_b64decode(padded.encode("ascii")).hex()
            except (ValueError, UnicodeEncodeError) as error:
                raise RuntimeError(
                    f"distribution {name!r} has malformed RECORD hash for {relative_path!r}"
                ) from error
            if len(expected_hash) != 64:
                raise RuntimeError(f"distribution {name!r} RECORD hash is not SHA-256")
            expected_bytes = int(entry.size)
            path = Path(distribution.locate_file(entry))
            authenticated, identity, _ = authenticated_file(
                path,
                expected_sha256=expected_hash,
                expected_bytes=expected_bytes,
            )
            resolved = str(path.resolve(strict=True))
            if resolved in package_identities:
                raise RuntimeError(f"distribution {name!r} has a duplicate payload path")
            package_identities[resolved] = identity
            canonical.extend(relative_path.encode("utf-8"))
            canonical.append(0)
            canonical.extend(str(authenticated["bytes"]).encode("ascii"))
            canonical.append(0)
            canonical.extend(authenticated["sha256"].encode("ascii"))
            canonical.append(ord("\n"))
            verified_files += 1
            verified_bytes += expected_bytes
        if record_entries != 1:
            raise RuntimeError(f"distribution {name!r} RECORD coverage changed")
        payload_digest = sha256_bytes(bytes(canonical))
        observed = (
            verified_files,
            verified_bytes,
            ignored_bytecode_files,
            tuple(excluded_external_scripts),
            payload_digest,
        )
        expected = EXPECTED_DISTRIBUTION_PAYLOADS[name]
        if observed != expected:
            raise RuntimeError(
                f"distribution {name!r} payload manifest changed: "
                f"expected {expected}, found {observed}"
            )
        versions[name] = version
        summaries[name] = {
            "version": version,
            "verified_files": verified_files,
            "verified_bytes": verified_bytes,
            "ignored_bytecode_files": ignored_bytecode_files,
            "excluded_external_scripts": excluded_external_scripts,
            "payload_manifest_sha256": payload_digest,
        }
        identities[name] = package_identities
    manifest = canonical_distribution_digest(summaries)
    if manifest != EXPECTED_DISTRIBUTION_MANIFEST_SHA256:
        raise RuntimeError(f"distribution manifest digest changed: {manifest}")
    return versions, summaries, manifest, identities


def verify_absent_optional_distributions() -> list[str]:
    """Require optional external kernel packages to be absent from the oracle."""

    absent = []
    for name in EXPECTED_ABSENT_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            absent.append(name)
            continue
        raise RuntimeError(f"optional package {name!r} must be absent, found {version}")
    return absent


def prepare_authenticated_imports(
    output_dir: Path,
    distribution_identities: Mapping[str, Mapping[str, tuple[int, int, int, int, int]]],
    transformers_root: Path,
) -> tuple[Path, dict[str, str], tuple[bool, str | None]]:
    """Resolve imports to authenticated payloads and disable adjacent bytecode."""

    protected_imports = (*PACKAGE_IMPORT_NAMES.values(), "transformers")
    imported = sorted(
        module_name
        for module_name in sys.modules
        if any(
            module_name == import_name or module_name.startswith(f"{import_name}.")
            for import_name in protected_imports
        )
    )
    if imported:
        raise RuntimeError(f"heavy packages imported before provenance preflight: {imported}")
    resolutions = {}
    for distribution_name, import_name in sorted(PACKAGE_IMPORT_NAMES.items()):
        specification = importlib.util.find_spec(import_name)
        if specification is None or specification.origin is None:
            raise RuntimeError(f"cannot resolve authenticated package {import_name!r}")
        origin = str(Path(specification.origin).resolve(strict=True))
        if origin not in distribution_identities[distribution_name]:
            raise RuntimeError(
                f"package {import_name!r} resolves outside authenticated RECORD payloads"
            )
        resolutions[distribution_name] = origin
    transformers_spec = importlib.util.find_spec("transformers")
    expected_transformers = (transformers_root / "__init__.py").resolve(strict=True)
    if (
        transformers_spec is None
        or transformers_spec.origin is None
        or Path(transformers_spec.origin).resolve(strict=True) != expected_transformers
    ):
        raise RuntimeError("Transformers import does not resolve to the pinned checkout")
    resolutions["transformers-source"] = str(expected_transformers)
    cache_root = Path(
        tempfile.mkdtemp(prefix=".muse-authenticated-pycache-", dir=output_dir)
    ).resolve(strict=True)
    previous = (sys.dont_write_bytecode, sys.pycache_prefix)
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(cache_root)
    return cache_root, resolutions, previous


def bind_imported_modules(
    modules: Mapping[str, Any],
    distribution_identities: Mapping[str, Mapping[str, tuple[int, int, int, int, int]]],
) -> dict[str, str]:
    """Prove every imported primary dependency came from its authenticated wheel."""

    if set(modules) != set(EXPECTED_DEPENDENCY_VERSIONS):
        raise RuntimeError("imported dependency binding set changed")
    bindings = {}
    for name, module in sorted(modules.items()):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"imported dependency {name!r} has no concrete file")
        path = str(Path(module_file).resolve(strict=True))
        if path not in distribution_identities[name]:
            raise RuntimeError(f"imported dependency {name!r} escaped its authenticated wheel")
        bindings[name] = path
    return bindings


def audit_imported_module_origins(
    transformers_root: Path,
    distribution_identities: Mapping[
        str, Mapping[str, tuple[int, int, int, int, int]]
    ],
) -> dict[str, str]:
    """Attribute all concrete module files to an explicit runtime trust domain.

    Primary wheel files and the Transformers source tree are authenticated by
    earlier checks. CPython/stdlib files and other installed distributions are
    accepted only under the explicit environment trust boundary recorded in
    the bundle. An arbitrary file injected through the working directory or
    `PYTHONPATH` has no such attribution and is rejected.
    """

    transformers_root = transformers_root.resolve(strict=True)
    stdlib_root = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    primary_paths: dict[str, str] = {}
    for distribution_name, identities in distribution_identities.items():
        for path in identities:
            previous = primary_paths.setdefault(path, distribution_name)
            if previous != distribution_name:
                raise RuntimeError(f"primary wheel payload ownership overlaps at {path}")

    packages_to_distributions = importlib.metadata.packages_distributions()
    distribution_path_cache: dict[
        tuple[str, str, str], set[str]
    ] = {}
    transitive: dict[str, str] = {}

    def matching_transitive_distribution(top_level: str, origin: str) -> tuple[str, str] | None:
        candidates = packages_to_distributions.get(top_level, [])
        matches = set()
        for candidate in dict.fromkeys(candidates):
            for distribution in importlib.metadata.distributions(name=candidate):
                canonical_name = distribution.metadata.get("Name") or candidate
                key = (
                    canonical_name,
                    distribution.version,
                    str(Path(distribution.locate_file(".")).resolve(strict=True)),
                )
                paths = distribution_path_cache.get(key)
                if paths is None:
                    entries = distribution.files
                    if not entries:
                        raise RuntimeError(
                            f"transitive distribution {canonical_name!r} has no RECORD entries"
                        )
                    paths = set()
                    for entry in entries:
                        path = Path(distribution.locate_file(entry))
                        try:
                            paths.add(str(path.resolve(strict=True)))
                        except FileNotFoundError:
                            continue
                    distribution_path_cache[key] = paths
                if origin in paths:
                    matches.add((canonical_name, distribution.version))
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError(
                f"module {top_level!r} has ambiguous distribution owners: {sorted(matches)}"
            )
        return next(iter(matches))

    rejected = []
    for module_name, module in sorted(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        if module_file.startswith("<") and module_file.endswith(">"):
            specification = getattr(module, "__spec__", None)
            if getattr(specification, "origin", None) in {"built-in", "frozen"}:
                continue
            rejected.append((module_name, module_file, "unattributed virtual origin"))
            continue
        virtual_identity = (
            module_name,
            module_file,
            type(module).__module__,
            type(module).__name__,
        )
        if virtual_identity in EXPECTED_PRIMARY_VIRTUAL_MODULES:
            continue
        try:
            origin_path = Path(module_file).resolve(strict=True)
        except (FileNotFoundError, OSError):
            rejected.append((module_name, module_file, "missing origin"))
            continue
        origin = str(origin_path)
        if origin_path == SELF_SOURCE_PATH:
            continue
        if origin_path.is_relative_to(transformers_root):
            if module_name != "transformers" and not module_name.startswith("transformers."):
                rejected.append((module_name, origin, "foreign name in Transformers tree"))
            continue
        if origin in primary_paths:
            continue
        top_level = module_name.partition(".")[0]
        owner = matching_transitive_distribution(top_level, origin)
        if owner is not None:
            name, version = owner
            if name in EXPECTED_DEPENDENCY_VERSIONS:
                rejected.append((module_name, origin, "primary wheel escaped authenticated payload"))
                continue
            previous = transitive.setdefault(name, version)
            if previous != version:
                rejected.append((module_name, origin, "conflicting transitive versions"))
            continue
        if origin_path.is_relative_to(stdlib_root) and not any(
            part in {"site-packages", "dist-packages"} for part in origin_path.parts
        ):
            continue
        rejected.append((module_name, origin, "unattributed module origin"))
    if rejected:
        raise RuntimeError(f"imported modules escaped recorded trust domains: {rejected[:8]}")
    return dict(sorted(transitive.items()))


def finalize_authenticated_imports(
    cache_root: Path, previous: tuple[bool, str | None]
) -> dict[str, bool]:
    """Require the private bytecode directory to stay empty and restore state."""

    try:
        entries = list(cache_root.iterdir())
        if entries:
            raise RuntimeError("authenticated bytecode cache unexpectedly changed")
        cache_root.rmdir()
    finally:
        sys.dont_write_bytecode, sys.pycache_prefix = previous
    return {
        "adjacent_bytecode_ignored": True,
        "writes_disabled": True,
        "private_cache_remained_empty": True,
        "restored": True,
    }


def verify_checkpoint(model_dir: Path) -> tuple[
    list[dict[str, Any]],
    list[int],
    dict[str, tuple[int, int, int, int, int]],
]:
    """Authenticate every model artifact and validate exact topology metadata."""

    verify_checkpoint_root_inventory(model_dir)
    artifacts = []
    identities = {}
    controls = {}
    for relative_path, (expected_bytes, expected_hash) in sorted(EXPECTED_ARTIFACTS.items()):
        retain = relative_path in {
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
        }
        authenticated, identity, payload = authenticated_file(
            model_dir / relative_path,
            expected_sha256=expected_hash,
            expected_bytes=expected_bytes,
            retain_payload=retain,
        )
        artifacts.append({"relative_path": relative_path, **authenticated})
        identities[relative_path] = identity
        if retain:
            assert payload is not None
            controls[relative_path] = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=lambda pairs: _reject_duplicate_pairs(
                    pairs, relative_path
                ),
            )

    config = controls["config.json"]
    if config.get("transformers_version") != CHECKPOINT_CONFIG_TRANSFORMERS_VERSION:
        raise ValueError("checkpoint config Transformers metadata changed")
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
    generation = controls["generation_config.json"]
    eos_ids = generation.get("eos_token_id")
    if eos_ids != EXPECTED_EOS_TOKEN_IDS or any(type(item) is not int for item in eos_ids):
        raise ValueError(f"generation EOS IDs changed: {eos_ids!r}")
    index = controls["model.safetensors.index.json"]
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

    digest = canonical_artifact_digest(artifacts)
    if digest != EXPECTED_ARTIFACT_MANIFEST_SHA256:
        raise RuntimeError(f"artifact manifest digest changed: {digest}")
    return artifacts, list(eos_ids), identities


def verify_checkpoint_root_inventory(model_dir: Path) -> None:
    """Reject any root entry that could alter local model/tokenizer selection.

    `from_pretrained` probes several optional filenames before settling on an
    indexed SafeTensors checkpoint. Authenticating only the intended files
    would therefore leave room for an injected single-file checkpoint,
    adapter, tokenizer overlay, or custom configuration to take precedence.
    """

    expected = set(EXPECTED_ARTIFACTS)
    observed = set()
    for entry in model_dir.iterdir():
        name = entry.name
        if name in observed:
            raise RuntimeError(f"duplicate checkpoint root entry: {name!r}")
        observed.add(name)
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"checkpoint root entry must not be a symlink: {entry}")
        if name in expected:
            expected_bytes = EXPECTED_ARTIFACTS[name][0]
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
                raise RuntimeError(f"consumed checkpoint artifact changed: {entry}")
        elif name in EXPECTED_INERT_ROOT_FILES:
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size != EXPECTED_INERT_ROOT_FILES[name]
            ):
                raise RuntimeError(f"inert checkpoint document identity changed: {entry}")
        elif name in ALLOWED_INERT_ROOT_DIRECTORIES:
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(f"hidden checkpoint cache changed type: {entry}")
        else:
            raise RuntimeError(
                f"unexpected checkpoint root entry could influence local loading: {entry}"
            )
    missing = expected - observed
    if missing:
        raise FileNotFoundError(f"required checkpoint artifacts are missing: {sorted(missing)}")


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


def load_with_verified_mps_warmup(
    modeling_utils: Any,
    torch: Any,
    expected_model_class: type,
    loader: Callable[[], Any],
) -> tuple[Any, dict[str, Any]]:
    """Validate the load map and delegate to the official MPS no-op warmup."""

    if getattr(modeling_utils, "__name__", None) != ALLOCATOR_WARMUP_SOURCE_MODULE:
        raise RuntimeError("allocator warmup owner changed")
    original = getattr(modeling_utils, "caching_allocator_warmup", None)
    if (
        not callable(original)
        or getattr(original, "__module__", None) != ALLOCATOR_WARMUP_SOURCE_MODULE
        or getattr(original, "__name__", None) != "caching_allocator_warmup"
    ):
        raise RuntimeError("allocator warmup callable is unavailable")
    calls = 0
    entries = 0

    def verified_warmup(model: Any, expanded_device_map: Any, hf_quantizer: Any) -> None:
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
        if entries != EXPECTED_ALLOCATOR_WARMUP_ENTRIES:
            raise RuntimeError(
                f"allocator warmup map has {entries} entries, expected "
                f"{EXPECTED_ALLOCATOR_WARMUP_ENTRIES}"
            )
        original(model, expanded_device_map, hf_quantizer)

    modeling_utils.caching_allocator_warmup = verified_warmup
    remained_scoped = False
    try:
        loaded = loader()
        remained_scoped = modeling_utils.caching_allocator_warmup is verified_warmup
    finally:
        modeling_utils.caching_allocator_warmup = original
    if not remained_scoped or modeling_utils.caching_allocator_warmup is not original:
        raise RuntimeError("allocator warmup override escaped its load scope")
    if calls != EXPECTED_ALLOCATOR_WARMUP_CALLS:
        raise RuntimeError(f"allocator warmup call count changed: {calls}")
    return loaded, {
        "target": ALLOCATOR_WARMUP_TARGET,
        "policy": ALLOCATOR_WARMUP_POLICY,
        "scope": ALLOCATOR_WARMUP_SCOPE,
        "reason": ALLOCATOR_WARMUP_REASON,
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
    """Bind output to the authenticated, committed generator without unrelated dirt."""

    crate = script.resolve(strict=True).parents[1]
    repository = Path(run_git(crate, "rev-parse", "--show-toplevel")).resolve(strict=True)
    relative_repository_path = script.resolve(strict=True).relative_to(repository)
    status = run_git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        str(relative_repository_path),
    )
    if status:
        raise RuntimeError("reference generator source must be committed and clean")
    relative = script.resolve(strict=True).relative_to(crate).as_posix()
    assert_early_source_unchanged()
    return {
        "relative_path": relative,
        "bytes": len(_SELF_SOURCE_BYTES),
        "sha256": sha256_bytes(_SELF_SOURCE_BYTES),
        "git_head": run_git(repository, "rev-parse", "HEAD"),
        "git_tree": run_git(repository, "rev-parse", "HEAD^{tree}"),
        "git_dirty": False,
    }


def tensor_cosine(torch: Any, left: Any, right: Any) -> float:
    """Compute finite full-vector cosine with float64 CPU accumulation."""

    left = left.to(dtype=torch.float64)
    right = right.to(dtype=torch.float64)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= 0.0:
        raise RuntimeError("cosine similarity has a non-positive or non-finite norm")
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
    # ``embed_tokens.embed_norm`` rather than the raw embedding output. The
    # official text model deliberately nests this norm inside its nn.Embedding
    # subclass; there is no ``language_model.embed_norm`` attribute.
    labels = ["embed_norm", *[f"layers.{index}" for index in range(52)], "norm"]
    modules = [
        language_model.embed_tokens.embed_norm,
        *language_model.layers,
        language_model.norm,
    ]
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
    python_runtime = verify_python_runtime()
    reference_source = verify_transformers_checkout(transformers_dir)
    (
        dependencies,
        distribution_provenance,
        distribution_manifest,
        distribution_identities,
    ) = verify_distribution_payloads()
    absent_optional_distributions = verify_absent_optional_distributions()
    artifacts, eos_ids, checkpoint_identities = verify_checkpoint(model_dir)

    sys.path.insert(0, str(transformers_dir / "src"))
    transformers_root = (transformers_dir / "src/transformers").resolve(strict=True)
    bytecode_cache, import_resolutions, previous_bytecode_state = (
        prepare_authenticated_imports(
            output_dir, distribution_identities, transformers_root
        )
    )
    import accelerate
    import filelock
    import huggingface_hub
    import numpy
    import packaging
    import regex
    import safetensors
    import tokenizers
    import torch
    import tqdm
    import transformers
    import yaml
    from transformers import (
        AutoTokenizer,
        DynamicCache,
        MuseGlimmerConfig,
        MuseGlimmerForConditionalGeneration,
        TokenizersBackend,
    )
    from transformers import modeling_utils
    from transformers.models.muse_glimmer import (
        configuration_muse_glimmer,
        modeling_muse_glimmer,
    )

    import_bindings = bind_imported_modules(
        {
            "accelerate": accelerate,
            "filelock": filelock,
            "huggingface-hub": huggingface_hub,
            "numpy": numpy,
            "packaging": packaging,
            "PyYAML": yaml,
            "regex": regex,
            "safetensors": safetensors,
            "tokenizers": tokenizers,
            "torch": torch,
            "tqdm": tqdm,
        },
        distribution_identities,
    )
    pre_execution_transitive_distributions = audit_imported_module_origins(
        transformers_root, distribution_identities
    )
    del accelerate, filelock, huggingface_hub, numpy, packaging, regex
    del safetensors, tokenizers, tqdm, yaml

    if Path(transformers.__file__).resolve(strict=True).parent != transformers_root:
        raise RuntimeError("imported Transformers package escaped the pinned checkout")
    expected_imported_sources = {
        "configuration": (
            transformers_root / "models/muse_glimmer/configuration_muse_glimmer.py"
        ).resolve(strict=True),
        "modeling": (
            transformers_root / "models/muse_glimmer/modeling_muse_glimmer.py"
        ).resolve(strict=True),
    }
    observed_imported_sources = {
        "configuration": Path(configuration_muse_glimmer.__file__).resolve(strict=True),
        "modeling": Path(modeling_muse_glimmer.__file__).resolve(strict=True),
    }
    if observed_imported_sources != expected_imported_sources:
        raise RuntimeError("imported Muse source modules escaped the pinned checkout")
    escaped_modules = []
    for name, module in sys.modules.items():
        if not name.startswith("transformers"):
            continue
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str):
            path = Path(module_file).resolve(strict=True)
            if not path.is_relative_to(transformers_root):
                escaped_modules.append((name, str(path)))
    if escaped_modules:
        raise RuntimeError(f"Transformers modules escaped the pinned checkout: {escaped_modules[:8]}")
    if transformers.__version__ != REFERENCE_SOURCE_VERSION:
        raise RuntimeError(
            f"Transformers source version changed: expected {REFERENCE_SOURCE_VERSION}, "
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

    config = MuseGlimmerConfig.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, revision=REVISION
    )
    if type(config) is not MuseGlimmerConfig:
        raise RuntimeError(f"loaded unexpected configuration class {type(config)!r}")
    if config.transformers_version != CHECKPOINT_CONFIG_TRANSFORMERS_VERSION:
        raise RuntimeError("parsed checkpoint Transformers metadata changed")
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    load_started = time.perf_counter()
    loaded, warmup = load_with_verified_mps_warmup(
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
            use_kernels=False,
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
        or model.use_kernels is not False
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
    if type(tokenizer) is not TokenizersBackend:
        raise RuntimeError(f"loaded unexpected tokenizer class {type(tokenizer)!r}")
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

    post_reference_source = verify_transformers_checkout(transformers_dir)
    (
        post_dependencies,
        post_distribution_provenance,
        post_distribution_manifest,
        post_distribution_identities,
    ) = verify_distribution_payloads()
    post_artifacts, post_eos_ids, post_checkpoint_identities = verify_checkpoint(
        model_dir
    )
    transitive_distributions = audit_imported_module_origins(
        transformers_root, distribution_identities
    )
    if (
        post_reference_source["source_identity"]
        != reference_source["source_identity"]
        or post_dependencies != dependencies
        or post_distribution_provenance != distribution_provenance
        or post_distribution_manifest != distribution_manifest
        or post_distribution_identities != distribution_identities
        or post_artifacts != artifacts
        or post_eos_ids != eos_ids
        or post_checkpoint_identities != checkpoint_identities
        or any(
            transitive_distributions.get(name) != version
            for name, version in pre_execution_transitive_distributions.items()
        )
    ):
        raise RuntimeError("authenticated source, package, or checkpoint state changed")
    assert_early_source_unchanged()
    bytecode_policy = finalize_authenticated_imports(
        bytecode_cache, previous_bytecode_state
    )

    script = Path(__file__).resolve(strict=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    record = {
        "schema_version": 2,
        "repository": REPOSITORY,
        "revision": REVISION,
        "reference": {
            "repository": REFERENCE_REPOSITORY,
            "revision": REFERENCE_REVISION,
            "class": REFERENCE_CLASS,
            "source_version": REFERENCE_SOURCE_VERSION,
            "checkpoint_config_transformers_version": CHECKPOINT_CONFIG_TRANSFORMERS_VERSION,
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
            "integrity_threat_model": INTEGRITY_THREAT_MODEL,
            "authenticated_import_resolutions": import_resolutions,
            "imported_module_bindings": import_bindings,
            "authenticated_import_bytecode_policy": bytecode_policy,
            "allocator_warmup_override": warmup,
        },
        "environment": {
            "generated_at_utc": now,
            "python": python_runtime,
            "packages": dependencies,
            "distribution_provenance": distribution_provenance,
            "distribution_manifest_sha256": distribution_manifest,
            "absent_optional_distributions": absent_optional_distributions,
            "transitive_distributions": transitive_distributions,
            "trust_boundary": {
                "interpreter_and_stdlib": INTERPRETER_TRUST_BOUNDARY,
                "interpreter_and_stdlib_content_authenticated": False,
                "record_authenticated_primary_distributions": sorted(
                    EXPECTED_DEPENDENCY_VERSIONS
                ),
                "transitive_distributions": TRANSITIVE_TRUST_BOUNDARY,
                "transitive_distribution_content_authenticated": False,
                "all_module_origins_attributed": True,
            },
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
            "integrity_postflight_passed": True,
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
