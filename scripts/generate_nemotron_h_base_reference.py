#!/usr/bin/env python3
"""Generate the pinned Nemotron-H 3.5 Lightning base-model MPS F16 oracle.

This is an offline maintainer tool, not Rust runtime code.  It loads the exact
official causal language-model class from a clean local Transformers checkout pinned to
``cfa309cfa34b5497a32fd50af49aa97aac94c054`` and exercises only its language
path.  It records one full-vocabulary prompt row, all 32 cached and
teacher-forced full-vocabulary rows, and 54-row hidden traces for stateless
prompt, cached-prefill, and first recurrent-decode execution.

The seven output files are fsynced inside a hidden sibling staging directory
and atomically published with no-replace semantics.  Interrupted runs expose no
partial final bundle and retain staging evidence for diagnosis.

The pinned sources, checkpoint, and 11 primary wheel payloads are content
authenticated. CPython/stdlib/OS payloads and recorded transitive distributions
remain explicit environment trust inputs; the generated record never calls
those bytes authenticated.

The official Transformers class intentionally does not instantiate the
checkpoint's ``mtp.*`` namespace.  This tool authenticates and inventories all
270 physical MTP tensors before loading, proves the pinned class's sole
unexpected-key filter is exactly ``mtp.*``, then records only base-model
outputs.  It cannot be used as evidence for the separate MTP graph.
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
import shutil
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
        "executing Nemotron-H generator code differs from its authenticated source snapshot"
    )
del _recompiled_self


REPOSITORY = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
REVISION = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
REFERENCE_REPOSITORY = "huggingface/transformers"
REFERENCE_REVISION = "cfa309cfa34b5497a32fd50af49aa97aac94c054"
REFERENCE_CLASS = "transformers.NemotronHForCausalLM"
REFERENCE_SOURCE_VERSION = "5.17.0.dev0"
CHECKPOINT_CONFIG_TRANSFORMERS_VERSION = "4.57.6"
REFERENCE_SOURCE = "src/transformers/models/nemotron_h/modeling_nemotron_h.py"
REFERENCE_SOURCES = {
    REFERENCE_SOURCE: (
        50_797,
        "d42067429e3268d1163579f5d257fe0baa9a2b02edba5fcb175fb8d42c9b6af4",
    ),
    "src/transformers/models/nemotron_h/configuration_nemotron_h.py": (
        13_031,
        "00a4c60bd302d796b25c6ae25c608a5ee7e27f9a0adea92911b0c45470232052",
    ),
}

# This short literal smoke prompt is already committed in the crate's
# Nemotron acceptance scaffold. Pinning its tokenization avoids chat-template
# or date-dependent behavior in the numerical oracle.
PROMPT = "Hello world, cached decoding."
INPUT_IDS = [22_177, 4_304, 1_044, 57_424, 59_417, 1_046]
VOCAB_SIZE = 131_072
HIDDEN_SIZE = 2_688
HIDDEN_TRACE_ROWS = 54
GREEDY_TOKENS = 32
# The first fail-closed diagnostic produced this binary16-stable continuation.
# The authenticated tokenizer maps 1256 to two space markers and 1010 to a
# newline marker, so the pinned model repeats a whitespace-only two-token cycle.
EXPECTED_GENERATED_TOKEN_IDS = [
    1_256, 1_010, 1_256, 1_010, 1_256, 1_010, 1_256, 1_010,
    1_256, 1_010, 1_256, 1_010, 1_256, 1_010, 1_256, 1_010,
    1_256, 1_010, 1_256, 1_010, 1_256, 1_010, 1_256, 1_010,
    1_256, 1_010, 1_256, 1_010, 1_256, 1_010, 1_256, 1_010,
]
EXPECTED_RAW_LAYER_TYPES = [
    "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
    "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
    "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
    "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
    "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
    "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba",
    "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba",
    "moe", "mamba", "moe",
]
EXPECTED_LAYER_TYPES = [
    {"mamba": "linear_attention", "attention": "full_attention"}.get(item, item)
    for item in EXPECTED_RAW_LAYER_TYPES
]
EXPECTED_EOS_TOKEN_IDS = [2, 11]
EXPECTED_CYCLE_TOKEN_PIECES = ["ĠĠ", "Ċ"]
EXPECTED_DECODED_CONTINUATION = "  \n" * 16
EXPECTED_PARAMETER_COUNT = 31_577_937_344
EXPECTED_BUFFER_ELEMENTS = 2_944
EXPECTED_MODEL_STATE_TENSORS = 401
EXPECTED_BASE_PHYSICAL_TENSORS = 6_243
EXPECTED_MTP_PHYSICAL_TENSORS = 270
EXPECTED_PHYSICAL_TENSORS = 6_513
EXPECTED_BF16_TENSORS = 6_489
EXPECTED_F32_TENSORS = 24
EXPECTED_BASE_LOADED_BYTES = 63_155_886_464
EXPECTED_MTP_TENSOR_BYTES = 2_670_652_160
EXPECTED_PHYSICAL_TENSOR_BYTES = 65_826_538_624
EXPECTED_REPORTED_TENSOR_BYTES = 65_842_365_568
EXPECTED_REPORTED_SIZE_DELTA = 15_826_944
EXPECTED_ALLOCATOR_WARMUP_ENTRIES = EXPECTED_MODEL_STATE_TENSORS
SELF_PARITY_COSINE = 0.999
CROSS_RUNTIME_PARITY_COSINE = 0.999
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

JSON_FILENAME = "nemotron-3.5-lightning-30b-a3b-a9904d24-reference.json"
LOGITS_FILENAME = "nemotron-3.5-lightning-30b-a3b-a9904d24-prompt-logits.f32le"
HIDDEN_FILENAME = "nemotron-3.5-lightning-30b-a3b-a9904d24-hidden-last.f32le"
CACHED_LOGITS_FILENAME = "nemotron-3.5-lightning-30b-a3b-a9904d24-cached-prefix-logits.f32le"
TEACHER_LOGITS_FILENAME = "nemotron-3.5-lightning-30b-a3b-a9904d24-teacher-prefix-logits.f32le"
CACHED_PREFILL_HIDDEN_FILENAME = (
    "nemotron-3.5-lightning-30b-a3b-a9904d24-cached-prefill-hidden-last.f32le"
)
CACHED_DECODE_HIDDEN_FILENAME = (
    "nemotron-3.5-lightning-30b-a3b-a9904d24-cached-decode-1-hidden-last.f32le"
)
BUNDLE_DIRNAME = "nemotron-3.5-lightning-30b-a3b-a9904d24-reference.bundle"

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
        9_867,
        "58933db77d3099b4f78c55a38347a72e1ea05b97d6bd8f38775303dc0194e0a9",
    ),
    "config.json": (
        2_352,
        "a3827a0f5e311547b40943dc081e3ff2f8a277466e8c1a3df2291e8db8a7617c",
    ),
    "generation_config.json": (
        210,
        "0ed94823ae737b507aef6af01e837041b7aa2fb0306793a310cd201d89a31ff4",
    ),
    "model.safetensors.index.json": (
        612_207,
        "67f21da80ce245a3e24967f54eef3fa10e67a63eba16ef19a2d0569f06103f50",
    ),
    "special_tokens_map.json": (
        563,
        "e9435fefd6d838fd9fcbbc44b97a8e3ff322be7f6dfb7e4fd2468586574bb52b",
    ),
    "tokenizer.json": (
        17_077_484,
        "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
    ),
    "tokenizer_config.json": (
        177_209,
        "10f93eabcb9b1602fbb991d6308e787ce1df28ee9cd7a1c6d1e8c3f338b957bc",
    ),
    "model-00001-of-00014.safetensors": (4_991_210_024, "7a3e74d24969eac6657cb9b6422091fd062b7d30fd4bc90ecee6c3d7f692626a"),
    "model-00002-of-00014.safetensors": (4_992_601_016, "a80c3846e959f6dda34aa031cbb668b766643691ac7e5c9f43f62e6fb1221059"),
    "model-00003-of-00014.safetensors": (4_992_601_368, "996bf3a46668e14075cb82259390267c2b1f71446caeb3a5f1f9e0bd7a079ae7"),
    "model-00004-of-00014.safetensors": (4_995_692_800, "cce1d79e7ff2e2b746b9d97166156aa1dcec796116675cc7fd1aa70a8d73ccb2"),
    "model-00005-of-00014.safetensors": (4_980_553_832, "9338f59d3fd5f074288c71a9559452d5b5c581da00e9ccb9fe7aa2ce9b04d2ad"),
    "model-00006-of-00014.safetensors": (4_989_423_536, "e947fa4097c68263ba3d097fceb32a300a22cebd9d909fb5198af7d18a4022fe"),
    "model-00007-of-00014.safetensors": (4_992_601_488, "c7362630bf4256275ca58354ed8f31506cec67ffab30573e753703fd5ae48795"),
    "model-00008-of-00014.safetensors": (4_992_601_520, "1b33a5a6b5505b25b57c90ddda19e9cd1598176468917df0c56116fa6f6072cb"),
    "model-00009-of-00014.safetensors": (4_995_692_800, "fcc9b48caff427beaf1f87b78cf4ab42bc2a3c43b3c487bccb013ba22d7468b2"),
    "model-00010-of-00014.safetensors": (4_992_601_520, "d1924fd54d1b77a15aff8df591a317e8a82a58e46d6f5742fa21e90e0be3cfa6"),
    "model-00011-of-00014.safetensors": (4_995_692_800, "e9d0b83bcf04bd4d5758b94e6495a3f1595d121d82faffaa27ce6a884e2c8757"),
    "model-00012-of-00014.safetensors": (4_995_692_816, "fce184f112cb0b97024cadc60f993f6f660e365d04e21c240fb3c75747bd3a01"),
    "model-00013-of-00014.safetensors": (3_249_723_504, "f427d037d2a3e4d44323325aa38571e3551e8ad702f469c3dacc4e59ea2e50d1"),
    "model-00014-of-00014.safetensors": (2_670_685_240, "64577b275ca4e7e5266eae0903674f7f46ec2a8cbf4f4f1a3207f80d503cd1d0"),
}
EXPECTED_ARTIFACT_MANIFEST_SHA256 = (
    "cb72063ee5b215548d9adbc67052de5f54871f1e28b80a0e0db9782afd4ad759"
)

# These upstream files are semantically inert to local Transformers
# loading. They may be present when `hf download` fetches the complete pinned
# snapshot. The Hub's hidden local-directory cache is likewise allowed but is
# never traversed by this verifier or passed as the model root.
EXPECTED_INERT_ROOT_FILES = {
    ".gitattributes": 1_690,
    "LICENSE": 2_693,
    "README.md": 85_345,
    "accuracy_plot.png": 140_266,
    "agentic_coding_benchmarks.png": 219_966,
    "bias.md": 3_263,
    "explainability.md": 2_699,
    "privacy.md": 828,
    "safety.md": 2_145,
}
ALLOWED_INERT_ROOT_DIRECTORIES = frozenset({".cache"})

ALLOCATOR_WARMUP_TARGET = "transformers.modeling_utils.caching_allocator_warmup"
ALLOCATOR_WARMUP_POLICY = "mps-official-noop-verified-during-from-pretrained"
ALLOCATOR_WARMUP_SOURCE_MODULE = "transformers.modeling_utils"
ALLOCATOR_WARMUP_SCOPE = "NemotronHForCausalLM.from_pretrained"
ALLOCATOR_WARMUP_REASON = (
    "Pinned Transformers 5.17 explicitly skips allocator warmup on MPS. The scoped "
    "wrapper validates the complete expanded parameter map, then delegates to that "
    "official no-op branch without changing model tensors or numerical execution."
)
EXPECTED_ALLOCATOR_WARMUP_CALLS = 1
MINIMUM_MPS_LOAD_HEADROOM_BYTES = 2_000_000_000
# The first concurrent diagnostic attempt consumed roughly 35 GiB of temporary
# system-volume space while MPS compiled eager expert graphs. Requiring 40 GiB
# before authenticating or loading leaves a measured margin above that observed
# footprint; the independent 10 GiB runtime reserve stops subsequent phases if
# another process begins consuming the same APFS container.
MINIMUM_SYSTEM_PREFLIGHT_FREE_BYTES = 40 * 1024**3
MINIMUM_SYSTEM_RUNTIME_FREE_BYTES = 10 * 1024**3
MINIMUM_OUTPUT_FREE_BYTES = 1 * 1024**3


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
        raise RuntimeError("executed Nemotron-H generator source changed during the run")


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


def vector_comparison_diagnostic(
    left: Sequence[float],
    right: Sequence[float],
    *,
    comparison: str,
    left_name: str,
    right_name: str,
    gate_cosine: float,
) -> dict[str, Any]:
    """Describe a failed vector-parity gate without hiding numerical drift.

    Dot products and norms use :func:`math.fsum` so the diagnostic itself does
    not inherit device reduction order. Relative error uses the right-hand
    vector as the reference and clamps only its denominator to ``1e-12``;
    reporting that convention keeps large near-zero relative errors honest.
    """

    if not comparison or not left_name or not right_name:
        raise ValueError("comparison and vector names must be nonempty")
    if len(left) != len(right) or not left:
        raise ValueError("comparison vectors must be equally sized and nonempty")
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not math.isfinite(gate_cosine) or not all(
        math.isfinite(value) for value in [*left_values, *right_values]
    ):
        raise ValueError("comparison vectors and gate cosine must be finite")
    absolute_errors = [
        abs(left_value - right_value)
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ]
    relative_errors = [
        error / max(abs(right_value), 1.0e-12)
        for error, right_value in zip(absolute_errors, right_values, strict=True)
    ]
    max_absolute_index = max(
        range(len(absolute_errors)), key=absolute_errors.__getitem__
    )
    max_relative_index = max(
        range(len(relative_errors)), key=relative_errors.__getitem__
    )
    left_norm = math.sqrt(math.fsum(value * value for value in left_values))
    right_norm = math.sqrt(math.fsum(value * value for value in right_values))
    dot = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("comparison-vector norms must be nonzero")

    def ranked(values: Sequence[float]) -> dict[str, Any]:
        """Return the leading two coordinates and their exact score margin."""

        leaders = top_k_records(values, 2)
        return {
            "top1_token_id": leaders[0]["token_id"],
            "top1_logit": leaders[0]["logit"],
            "top2_token_id": leaders[1]["token_id"],
            "top2_logit": leaders[1]["logit"],
            "margin": float(leaders[0]["logit"]) - float(leaders[1]["logit"]),
        }

    return {
        "comparison": comparison,
        "left_name": left_name,
        "right_name": right_name,
        "elements": len(left_values),
        "gate_cosine": gate_cosine,
        "recomputed_float64_cosine": dot / (left_norm * right_norm),
        "dot": dot,
        "left_l2_norm": left_norm,
        "right_l2_norm": right_norm,
        "max_absolute_error": absolute_errors[max_absolute_index],
        "max_absolute_error_index": max_absolute_index,
        "left_at_max_absolute_error": left_values[max_absolute_index],
        "right_at_max_absolute_error": right_values[max_absolute_index],
        "max_relative_error": relative_errors[max_relative_index],
        "max_relative_error_index": max_relative_index,
        "left_at_max_relative_error": left_values[max_relative_index],
        "right_at_max_relative_error": right_values[max_relative_index],
        "relative_error_denominator": "max(abs(right_reference), 1e-12)",
        "left_top2": ranked(left_values),
        "right_top2": ranked(right_values),
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

    expected_names = {
        JSON_FILENAME,
        LOGITS_FILENAME,
        HIDDEN_FILENAME,
        CACHED_LOGITS_FILENAME,
        TEACHER_LOGITS_FILENAME,
        CACHED_PREFILL_HIDDEN_FILENAME,
        CACHED_DECODE_HIDDEN_FILENAME,
    }
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
    """Require the exact clean official source commit and model-file identities."""

    checkout = checkout.expanduser().resolve(strict=True)
    if not checkout.is_dir():
        raise ValueError(f"Transformers checkout is not a directory: {checkout}")
    head = run_git(checkout, "rev-parse", "HEAD")
    if head != REFERENCE_REVISION:
        raise RuntimeError(f"Transformers HEAD changed: expected {REFERENCE_REVISION}, found {head}")
    status = run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("Transformers reference checkout must be completely clean")
    sources = []
    identities = {}
    for relative_path, (expected_bytes, expected_sha256) in sorted(
        REFERENCE_SOURCES.items()
    ):
        authenticated, identity, _ = authenticated_file(
            checkout / relative_path,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
        observed = {"relative_path": relative_path, **authenticated}
        expected = {
            "relative_path": relative_path,
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }
        if observed != expected:
            raise RuntimeError(
                f"official Nemotron-H source changed: expected {expected}, "
                f"found {observed}"
            )
        sources.append(observed)
        identities[relative_path] = identity
    return {
        "repository": REFERENCE_REPOSITORY,
        "revision": head,
        "clean": True,
        "checkout": str(checkout),
        "sources": sources,
        "source_identities": identities,
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
            f"Nemotron-H oracle requires CPython {EXPECTED_PYTHON_VERSION}, found "
            f"{platform.python_implementation()} {observed}"
        )
    executable = Path(sys.executable)
    if not executable.is_absolute() or not executable.exists():
        raise RuntimeError(f"Python executable path is not absolute and present: {executable}")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Nemotron-H oracle must run inside an explicit Python virtual environment")
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
        tempfile.mkdtemp(prefix=".nemotron-authenticated-pycache-", dir=output_dir)
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


def read_safetensors_header(
    path: Path, expected_file_bytes: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Parse one authenticated SafeTensors header without mapping tensor payloads.

    SafeTensors offsets are relative to the first byte after the padded JSON
    header. Requiring a gap-free, size-consistent interval partition prevents a
    valid-looking header from hiding trailing, overlapping, or truncated data.
    """

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("SafeTensors header validation requires O_NOFOLLOW/O_NONBLOCK")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_file_bytes:
            raise RuntimeError(f"SafeTensors file identity changed: {path}")
        prefix = os.read(descriptor, 8)
        if len(prefix) != 8:
            raise ValueError(f"truncated SafeTensors length prefix: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes == 0 or header_bytes > 4 * 1024 * 1024 or header_bytes % 8:
            raise ValueError(f"invalid SafeTensors header length {header_bytes}: {path}")
        payload = bytearray()
        while len(payload) < header_bytes:
            chunk = os.read(descriptor, header_bytes - len(payload))
            if not chunk:
                raise ValueError(f"truncated SafeTensors header: {path}")
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if file_identity(before) != file_identity(after):
            raise RuntimeError(f"SafeTensors file changed while reading header: {path}")
    finally:
        os.close(descriptor)

    header = json.loads(
        bytes(payload).decode("utf-8"),
        object_pairs_hook=lambda pairs: _reject_duplicate_pairs(pairs, str(path)),
    )
    if not isinstance(header, dict):
        raise ValueError(f"SafeTensors header must be an object: {path}")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError(f"SafeTensors metadata changed type: {path}")

    dtype_bytes = {"BF16": 2, "F32": 4}
    tensors: dict[str, dict[str, Any]] = {}
    ordered_ranges = []
    for name, item in header.items():
        if not isinstance(name, str) or not name or not isinstance(item, dict):
            raise ValueError(f"invalid SafeTensors tensor entry: {path}")
        if set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"SafeTensors tensor schema changed for {name!r}")
        dtype = item["dtype"]
        shape = item["shape"]
        offsets = item["data_offsets"]
        if dtype not in dtype_bytes:
            raise ValueError(f"unsupported checkpoint dtype {dtype!r} for {name!r}")
        if not isinstance(shape, list) or any(
            type(dimension) is not int or dimension <= 0 for dimension in shape
        ):
            raise ValueError(f"invalid SafeTensors shape for {name!r}: {shape!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"invalid SafeTensors offsets for {name!r}: {offsets!r}")
        expected_span = math.prod(shape) * dtype_bytes[dtype]
        if offsets[1] - offsets[0] != expected_span:
            raise ValueError(f"SafeTensors byte span disagrees with {name!r} shape")
        normalized = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
        tensors[name] = normalized
        ordered_ranges.append((offsets[0], offsets[1], name))

    cursor = 0
    for start, end, name in sorted(ordered_ranges):
        if start != cursor:
            raise ValueError(
                f"SafeTensors payload ranges are not gap-free at {name!r}: "
                f"expected {cursor}, found {start}"
            )
        cursor = end
    if 8 + header_bytes + cursor != expected_file_bytes:
        raise ValueError(f"SafeTensors payload size does not consume complete file: {path}")
    return tensors, {
        "header_bytes_excluding_prefix": header_bytes,
        "tensor_payload_bytes": cursor,
        "physical_tensors": len(tensors),
        "metadata": dict(sorted(metadata.items())),
    }


def canonical_header_inventory_digest(
    tensors: Mapping[str, Mapping[str, Any]]
) -> str:
    """Hash exact name/shard/dtype/shape/range facts in sorted-name order."""

    canonical = bytearray()
    for name in sorted(tensors):
        item = tensors[name]
        canonical.extend(name.encode("utf-8"))
        canonical.append(0)
        canonical.extend(str(item["shard"]).encode("utf-8"))
        canonical.append(0)
        canonical.extend(str(item["dtype"]).encode("ascii"))
        canonical.append(0)
        for dimension in item["shape"]:
            canonical.extend(str(dimension).encode("ascii"))
            canonical.append(ord(","))
        canonical.append(0)
        canonical.extend(str(item["data_offsets"][0]).encode("ascii"))
        canonical.append(ord(":"))
        canonical.extend(str(item["data_offsets"][1]).encode("ascii"))
        canonical.append(ord("\n"))
    return sha256_bytes(bytes(canonical))


def verify_checkpoint(model_dir: Path) -> tuple[
    list[dict[str, Any]],
    list[int],
    dict[str, tuple[int, int, int, int, int]],
    dict[str, Any],
]:
    """Authenticate every artifact and separate base tensors from physical MTP."""

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
    expected_fields = {
        "model_type": "nemotron_h",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": 52,
        "layers_block_type": EXPECTED_RAW_LAYER_TYPES,
        "mtp_layers_block_type": ["attention", "moe"],
        "num_nextn_predict_layers": 1,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "n_groups": 8,
        "conv_kernel": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "n_routed_experts": 128,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "mamba_ssm_cache_dtype": "float32",
    }
    for name, expected in expected_fields.items():
        if config.get(name) != expected:
            raise ValueError(f"Nemotron-H config {name!r} changed")
    generation = controls["generation_config.json"]
    eos_ids = generation.get("eos_token_id")
    if eos_ids != EXPECTED_EOS_TOKEN_IDS or any(type(item) is not int for item in eos_ids):
        raise ValueError(f"generation EOS IDs changed: {eos_ids!r}")
    index = controls["model.safetensors.index.json"]
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != EXPECTED_PHYSICAL_TENSORS:
        raise ValueError("checkpoint index tensor inventory changed")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint index metadata must be an object")
    total_size = checked_json_nonnegative_integer(metadata.get("total_size"), "metadata.total_size")
    total_parameters = checked_json_nonnegative_integer(
        metadata.get("total_parameters"), "metadata.total_parameters"
    )
    if total_size != EXPECTED_REPORTED_TENSOR_BYTES:
        raise ValueError(f"checkpoint reported tensor bytes changed: {total_size}")
    if total_parameters != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"checkpoint reported parameter count changed: {total_parameters}")
    expected_shards = {
        f"model-{index:05}-of-00014.safetensors" for index in range(1, 15)
    }
    if set(weight_map.values()) != expected_shards:
        raise ValueError("checkpoint index shard names changed")

    physical: dict[str, dict[str, Any]] = {}
    shard_records = []
    for shard in sorted(expected_shards):
        tensors, shard_record = read_safetensors_header(
            model_dir / shard, EXPECTED_ARTIFACTS[shard][0]
        )
        for name, item in tensors.items():
            if name in physical:
                raise ValueError(f"duplicate physical checkpoint tensor {name!r}")
            physical[name] = {**item, "shard": shard}
        shard_records.append({"relative_path": shard, **shard_record})
    if set(weight_map) != set(physical):
        missing = sorted(set(weight_map) - set(physical))
        unexpected = sorted(set(physical) - set(weight_map))
        raise ValueError(
            f"index/header tensor names differ: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    for name, shard in weight_map.items():
        if physical[name]["shard"] != shard:
            raise ValueError(f"index assigns {name!r} to the wrong shard")

    mtp_names = sorted(name for name in physical if name.startswith("mtp."))
    base_names = sorted(name for name in physical if not name.startswith("mtp."))
    if len(mtp_names) != EXPECTED_MTP_PHYSICAL_TENSORS:
        raise ValueError(f"physical MTP tensor count changed: {len(mtp_names)}")
    if len(base_names) != EXPECTED_BASE_PHYSICAL_TENSORS:
        raise ValueError(f"physical base tensor count changed: {len(base_names)}")
    if any(physical[name]["shard"] != "model-00014-of-00014.safetensors" for name in mtp_names):
        raise ValueError("physical MTP tensors escaped their pinned isolated shard")
    if any(physical[name]["shard"] == "model-00014-of-00014.safetensors" for name in base_names):
        raise ValueError("the pinned MTP-only shard unexpectedly contains base tensors")

    dtype_counts = {
        dtype: sum(1 for item in physical.values() if item["dtype"] == dtype)
        for dtype in ("BF16", "F32")
    }
    if dtype_counts != {"BF16": EXPECTED_BF16_TENSORS, "F32": EXPECTED_F32_TENSORS}:
        raise ValueError(f"physical checkpoint dtype counts changed: {dtype_counts}")
    physical_bytes = sum(
        item["data_offsets"][1] - item["data_offsets"][0]
        for item in physical.values()
    )
    mtp_bytes = sum(
        physical[name]["data_offsets"][1] - physical[name]["data_offsets"][0]
        for name in mtp_names
    )
    if physical_bytes != EXPECTED_PHYSICAL_TENSOR_BYTES or mtp_bytes != EXPECTED_MTP_TENSOR_BYTES:
        raise ValueError(
            f"physical checkpoint bytes changed: total={physical_bytes}, mtp={mtp_bytes}"
        )

    moe_layers = [index for index, kind in enumerate(EXPECTED_LAYER_TYPES) if kind == "moe"]
    router_names = [f"backbone.layers.{index}.mixer.gate.weight" for index in moe_layers]
    for name in router_names:
        item = physical.get(name)
        if item is None or item["dtype"] != "BF16" or item["shape"] != [128, HIDDEN_SIZE]:
            raise ValueError(f"provenance-bound reported-size router changed: {name}")
    router_bytes = sum(
        physical[name]["data_offsets"][1] - physical[name]["data_offsets"][0]
        for name in router_names
    )
    reported_delta = total_size - physical_bytes
    if (
        len(router_names) != 23
        or router_bytes != EXPECTED_REPORTED_SIZE_DELTA
        or reported_delta != EXPECTED_REPORTED_SIZE_DELTA
    ):
        raise ValueError(
            f"upstream total_size defect no longer matches exact base routers: "
            f"delta={reported_delta}, routers={router_bytes}"
        )

    inventory_digest = canonical_header_inventory_digest(physical)
    if inventory_digest != "e255cec4fe6d43dfddaed52b3a5ef719c111d6c9821ce7ff53219080b50762c4":
        raise RuntimeError(f"physical header inventory digest changed: {inventory_digest}")

    digest = canonical_artifact_digest(artifacts)
    if digest != EXPECTED_ARTIFACT_MANIFEST_SHA256:
        raise RuntimeError(f"artifact manifest digest changed: {digest}")
    inventory = {
        "physical_tensors": len(physical),
        "base_physical_tensors": len(base_names),
        "mtp_physical_tensors": len(mtp_names),
        "mtp_name_sha256": sha256_bytes(
            "".join(f"{name}\n" for name in mtp_names).encode("utf-8")
        ),
        "dtype_tensor_counts": dtype_counts,
        "physical_tensor_bytes": physical_bytes,
        "base_tensor_bytes": physical_bytes - mtp_bytes,
        "mtp_tensor_bytes": mtp_bytes,
        "reported_tensor_bytes": total_size,
        "reported_size_delta": reported_delta,
        "reported_total_parameters": total_parameters,
        "reported_size_duplicate_router_names": router_names,
        "header_inventory_sha256": inventory_digest,
        "shards": shard_records,
    }
    return artifacts, list(eos_ids), identities, inventory


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


def require_free_space(
    path: Path, minimum_free_bytes: int, purpose: str
) -> dict[str, Any]:
    """Record filesystem capacity and reject an inadequate free-space reserve.

    MPS may compile large eager graphs into temporary files on the macOS system
    volume even when checkpoint and output files live on an external drive.
    Consequently the system and output filesystems are checked independently.
    Repeated runtime samples cannot interrupt a single in-flight kernel build,
    but they prevent the next expensive forward from starting after the reserve
    has been breached.
    """

    if (
        isinstance(minimum_free_bytes, bool)
        or not isinstance(minimum_free_bytes, int)
        or minimum_free_bytes <= 0
    ):
        raise ValueError("minimum free-space reserve must be a positive integer")
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("free-space purpose must be a nonempty string")
    resolved = path.expanduser().resolve(strict=True)
    usage = shutil.disk_usage(resolved)
    record = {
        "path": str(resolved),
        "purpose": purpose,
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "minimum_free_bytes": minimum_free_bytes,
    }
    if record["free_bytes"] < minimum_free_bytes:
        raise RuntimeError(
            f"insufficient free space for {purpose}: path={resolved}, "
            f"free={record['free_bytes']}, required={minimum_free_bytes}"
        )
    return record


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


def capture_last_hidden_trace(
    torch: Any,
    language_model: Any,
    expected_sequence: int,
    run: Callable[[], Any],
) -> tuple[Any, list[str], Any]:
    """Capture embedding, 52 post-residual blocks, and final RMSNorm rows."""

    if expected_sequence <= 0:
        raise ValueError("hidden-trace sequence length must be positive")

    labels = ["embeddings", *[f"layers.{index}" for index in range(52)], "norm_f"]
    modules = [
        language_model.embeddings,
        *language_model.layers,
        language_model.norm_f,
    ]
    captured = []
    handles = []
    for label, module in zip(labels, modules, strict=True):
        def hook(_module: Any, _inputs: Any, output: Any, *, trace_label: str = label) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if list(tensor.shape) != [1, expected_sequence, HIDDEN_SIZE]:
                raise RuntimeError(f"hidden trace {trace_label} shape changed: {list(tensor.shape)}")
            if tensor.dtype != torch.float16 or tensor.device.type != "mps":
                raise RuntimeError(
                    f"hidden trace {trace_label} dtype/device changed: "
                    f"{tensor.dtype}/{tensor.device}"
                )
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


def capture_selected_hidden_traces(
    torch: Any,
    language_model: Any,
    expected_sequence: int,
    positions: Sequence[int],
    run: Callable[[], Any],
) -> tuple[Any, list[str], Any]:
    """Capture all residual-boundary rows at selected sequence positions.

    Hooks retain small cloned F32 rows on MPS until the forward completes, then
    transfer one stacked tensor to CPU. This avoids synchronizing the device 54
    times per forward while still freezing each mutable hook observation.
    Returned axes are ``[selected_position, trace_row, hidden]``.
    """

    selected = list(positions)
    if (
        expected_sequence <= 0
        or not selected
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in selected
        )
        or selected != sorted(set(selected))
        or selected[0] < 0
        or selected[-1] >= expected_sequence
    ):
        raise ValueError("hidden-trace positions must be unique in-range sorted integers")
    labels = ["embeddings", *[f"layers.{index}" for index in range(52)], "norm_f"]
    modules = [
        language_model.embeddings,
        *language_model.layers,
        language_model.norm_f,
    ]
    captured = []
    handles = []
    for label, module in zip(labels, modules, strict=True):
        def hook(_module: Any, _inputs: Any, output: Any, *, trace_label: str = label) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if list(tensor.shape) != [1, expected_sequence, HIDDEN_SIZE]:
                raise RuntimeError(
                    f"selected hidden trace {trace_label} shape changed: "
                    f"{list(tensor.shape)}"
                )
            if tensor.dtype != torch.float16 or tensor.device.type != "mps":
                raise RuntimeError(
                    f"selected hidden trace {trace_label} dtype/device changed: "
                    f"{tensor.dtype}/{tensor.device}"
                )
            captured.append((trace_label, tensor[0, selected].float().clone()))

        handles.append(module.register_forward_hook(hook))
    try:
        output = run()
    finally:
        for handle in handles:
            handle.remove()
    observed_labels = [label for label, _ in captured]
    if observed_labels != labels:
        raise RuntimeError(f"selected hidden trace order changed: {observed_labels}")
    trace = (
        torch.stack([rows for _, rows in captured])
        .permute(1, 0, 2)
        .cpu()
        .contiguous()
    )
    expected_shape = [len(selected), HIDDEN_TRACE_ROWS, HIDDEN_SIZE]
    if list(trace.shape) != expected_shape or trace.dtype != torch.float32:
        raise RuntimeError(
            f"selected hidden trace shape/dtype changed: {list(trace.shape)}/{trace.dtype}"
        )
    return output, labels, trace


def trace_row_architecture(label: str) -> dict[str, Any]:
    """Map one hidden-trace boundary to its preceding architectural operation."""

    if label == "embeddings":
        return {"trace_label": label, "operation": "embedding"}
    if label == "norm_f":
        return {"trace_label": label, "operation": "final_rms_norm"}
    prefix = "layers."
    if not label.startswith(prefix) or not label[len(prefix) :].isdigit():
        raise ValueError(f"unknown hidden-trace label {label!r}")
    layer_index = int(label[len(prefix) :])
    if layer_index >= len(EXPECTED_LAYER_TYPES):
        raise ValueError(f"hidden-trace layer index is out of range: {layer_index}")
    return {
        "trace_label": label,
        "operation": "post_residual_block",
        "layer_index": layer_index,
        "layer_type": EXPECTED_LAYER_TYPES[layer_index],
        "raw_layer_type": EXPECTED_RAW_LAYER_TYPES[layer_index],
    }


def hidden_path_divergence_summary(
    torch: Any,
    cached_traces: Sequence[Any],
    teacher_traces: Any,
    labels: Sequence[str],
) -> dict[str, Any]:
    """Localize cached-recurrent versus teacher-chunked hidden-state drift.

    This is diagnostic evidence, not a cross-runtime acceptance shortcut. Burn
    must match each corresponding official path independently at the declared
    0.999 cosine gate even when the two official algorithms diverge internally.
    """

    if len(cached_traces) != GREEDY_TOKENS or list(teacher_traces.shape) != [
        GREEDY_TOKENS,
        HIDDEN_TRACE_ROWS,
        HIDDEN_SIZE,
    ]:
        raise ValueError("hidden path traces do not cover every generation decision")
    if list(labels) != [
        "embeddings",
        *[f"layers.{index}" for index in range(52)],
        "norm_f",
    ]:
        raise ValueError("hidden path labels changed")
    per_decision = []
    all_cosines = []
    for decision, cached_trace in enumerate(cached_traces):
        if list(cached_trace.shape) != [HIDDEN_TRACE_ROWS, HIDDEN_SIZE]:
            raise ValueError(f"cached hidden trace {decision} shape changed")
        cosines = [
            tensor_cosine(torch, cached_trace[row], teacher_traces[decision, row])
            for row in range(HIDDEN_TRACE_ROWS)
        ]
        all_cosines.append(cosines)
        minimum = min(cosines)
        minimum_row = cosines.index(minimum)
        per_decision.append(
            {
                "decision": decision,
                "minimum_cosine": minimum,
                "minimum_trace_row": minimum_row,
                **trace_row_architecture(labels[minimum_row]),
            }
        )
    worst = min(
        per_decision,
        key=lambda item: (item["minimum_cosine"], item["decision"]),
    )
    worst_decision = int(worst["decision"])
    worst_cosines = all_cosines[worst_decision]
    first_below = next(
        (item for item in per_decision if item["minimum_cosine"] < SELF_PARITY_COSINE),
        None,
    )
    detailed_layers = []
    previous = None
    for row, (label, cosine) in enumerate(zip(labels, worst_cosines, strict=True)):
        entry = {
            "trace_row": row,
            "cosine": cosine,
            "change_from_previous_boundary": (
                None if previous is None else cosine - previous
            ),
            **trace_row_architecture(label),
        }
        detailed_layers.append(entry)
        previous = cosine
    largest_drop = min(
        detailed_layers[1:],
        key=lambda item: (item["change_from_previous_boundary"], item["trace_row"]),
    )
    return {
        "comparison": "cached_recurrent_or_prefill_vs_teacher_forced_chunked",
        "cross_runtime_acceptance_gate": False,
        "corresponding_path_cross_runtime_minimum_cosine": (
            CROSS_RUNTIME_PARITY_COSINE
        ),
        "per_decision": per_decision,
        "first_decision_below_0_999_diagnostic_threshold": first_below,
        "worst_decision": worst,
        "worst_decision_layer_cosines": detailed_layers,
        "worst_decision_largest_boundary_drop": largest_drop,
    }


def capture_implicit_position_ids(
    torch: Any,
    first_attention: Any,
    expected: Sequence[int],
    run: Callable[[], Any],
) -> Any:
    """Prove the official model derived one exact implicit position-ID row."""

    expected = list(expected)
    if not expected:
        raise ValueError("position-ID expectation must be nonempty")
    captured: list[list[int]] = []

    def hook(_module: Any, _inputs: Any, kwargs: Any) -> None:
        if not isinstance(kwargs, dict) or set(kwargs).isdisjoint({"position_ids"}):
            raise RuntimeError("attention pre-hook did not receive position_ids")
        position_ids = kwargs["position_ids"]
        if (
            list(position_ids.shape) != [1, len(expected)]
            or position_ids.dtype != torch.int64
            or position_ids.device.type != "mps"
        ):
            raise RuntimeError(
                "implicit position-ID tensor shape/dtype/device changed: "
                f"{list(position_ids.shape)}/{position_ids.dtype}/{position_ids.device}"
            )
        captured.append([int(value) for value in position_ids[0].cpu().tolist()])

    handle = first_attention.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        output = run()
    finally:
        handle.remove()
    if captured != [expected]:
        raise RuntimeError(f"implicit position IDs changed: {captured} != {[expected]}")
    return output


def validate_official_cache(
    torch: Any,
    cache: Any,
    dynamic_cache_class: type,
    expected_length: int,
) -> dict[str, Any]:
    """Validate all KV, convolution, recurrent, and placeholder cache layers.

    ``DynamicCache.batch_size`` is not an authoritative physical-size query for
    this pinned model.  Its implementation returns ``-1`` when none of the
    concrete cache layers publishes an optional ``batch_size`` attribute, even
    though their initialized key/value tensors have a well-defined batch axis.
    The exact physical tensor shapes checked below establish batch one. Mamba2
    retains an F16 convolution tail ``[1, 6144, 4]`` and an FP32 recurrent SSM
    state ``[1, 64, 64, 128]``; MoE-only blocks remain empty placeholders; and
    the six NoPE GQA blocks retain F16 K/V tensors ``[1, 2, length, 128]``.
    """

    if expected_length <= 0 or expected_length >= 2_048:
        raise ValueError("short oracle cache length must be in 1..2048")
    if type(cache) is not dynamic_cache_class:
        raise RuntimeError(f"official cache class changed: {type(cache)!r}")
    if cache.batch_size != -1:
        raise RuntimeError(
            f"pinned DynamicCache batch-size sentinel changed: {cache.batch_size}"
        )
    if (
        cache.get_seq_length() != expected_length
        or len(cache.layers) != len(EXPECTED_LAYER_TYPES)
    ):
        raise RuntimeError(
            "official cache topology changed: "
            f"length={cache.get_seq_length()}, reported_batch={cache.batch_size}, "
            f"layers={len(cache.layers)}"
        )
    mamba_indices = []
    attention_indices = []
    placeholder_indices = []
    for index, (layer, expected_kind) in enumerate(
        zip(cache.layers, EXPECTED_LAYER_TYPES, strict=True)
    ):
        observed_class = type(layer).__name__
        if expected_kind == "full_attention":
            attention_indices.append(index)
            if observed_class != "DynamicLayer":
                raise RuntimeError(
                    f"official cache layer {index} class changed: {observed_class}"
                )
            if layer.get_seq_length() != expected_length:
                raise RuntimeError(
                    f"official cache layer {index} logical length changed: "
                    f"{layer.get_seq_length()}"
                )
            for name, tensor in (("keys", layer.keys), ("values", layer.values)):
                if (
                    list(tensor.shape) != [1, 2, expected_length, 128]
                    or tensor.dtype != torch.float16
                    or tensor.device.type != "mps"
                ):
                    raise RuntimeError(
                        f"official cache layer {index} {name} geometry changed: "
                        f"{list(tensor.shape)}/{tensor.dtype}/{tensor.device}"
                    )
            continue

        if observed_class != "LinearAttentionLayer":
            raise RuntimeError(
                f"official cache layer {index} class changed: {observed_class}"
            )
        expected_state_keys = {0}
        state_mappings = (
            layer.conv_states,
            layer.recurrent_states,
            layer.is_conv_states_initialized,
            layer.is_recurrent_states_initialized,
            layer.has_previous_state,
            layer.conv_kernel_size,
        )
        if any(set(mapping) != expected_state_keys for mapping in state_mappings):
            raise RuntimeError(f"official cache layer {index} state slots changed")
        if layer.number_of_states != 1 or layer.record_past is not False:
            raise RuntimeError(f"official cache layer {index} state policy changed")

        if expected_kind == "moe":
            placeholder_indices.append(index)
            if (
                layer.conv_states[0] is not None
                or layer.recurrent_states[0] is not None
                or layer.is_conv_states_initialized[0] is not False
                or layer.is_recurrent_states_initialized[0] is not False
                or layer.has_previous_state[0] is not False
                or layer.conv_kernel_size[0] is not None
                or layer.device is not None
                or layer.dtype is not None
            ):
                raise RuntimeError(f"MoE cache placeholder {index} was initialized")
            continue

        if expected_kind != "linear_attention":
            raise RuntimeError(f"unexpected normalized layer kind {expected_kind!r}")
        mamba_indices.append(index)
        if (
            layer.is_conv_states_initialized[0] is not True
            or layer.is_recurrent_states_initialized[0] is not True
            or layer.has_previous_state[0] is not True
            or layer.conv_kernel_size[0] != 4
            or layer.dtype != torch.float16
            or layer.device.type != "mps"
        ):
            raise RuntimeError(f"Mamba cache layer {index} initialization changed")
        conv = layer.conv_states[0]
        recurrent = layer.recurrent_states[0]
        if (
            list(conv.shape) != [1, 6_144, 4]
            or conv.dtype != torch.float16
            or conv.device.type != "mps"
        ):
            raise RuntimeError(f"Mamba cache layer {index} convolution geometry changed")
        if (
            list(recurrent.shape) != [1, 64, 64, 128]
            or recurrent.dtype != torch.float32
            or recurrent.device.type != "mps"
        ):
            raise RuntimeError(f"Mamba cache layer {index} recurrent geometry changed")

    if (
        mamba_indices != [i for i, kind in enumerate(EXPECTED_LAYER_TYPES) if kind == "linear_attention"]
        or attention_indices != [i for i, kind in enumerate(EXPECTED_LAYER_TYPES) if kind == "full_attention"]
        or placeholder_indices != [i for i, kind in enumerate(EXPECTED_LAYER_TYPES) if kind == "moe"]
    ):
        raise RuntimeError("cache layer-kind partition changed")
    return {
        "class": type(cache).__name__,
        "reported_batch_size": cache.batch_size,
        "logical_length": expected_length,
        "mamba_layer_indices": mamba_indices,
        "attention_layer_indices": attention_indices,
        "moe_placeholder_indices": placeholder_indices,
        "attention_key_value_shape": [1, 2, expected_length, 128],
        "attention_dtype": "float16",
        "mamba_conv_shape": [1, 6_144, 4],
        "mamba_conv_dtype": "float16",
        "mamba_recurrent_shape": [1, 64, 64, 128],
        "mamba_recurrent_dtype": "float32",
    }


def tensor_f32le_bytes(torch: Any, tensor: Any) -> bytes:
    """Serialize one finite contiguous CPU F32 tensor without Python-float expansion."""

    if sys.byteorder != "little":
        raise RuntimeError("F32LE oracle generation requires a little-endian host")
    if tensor.dtype != torch.float32 or tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("oracle payload tensor must be contiguous CPU float32")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("oracle payload tensor contains a non-finite value")
    return tensor.numpy().tobytes(order="C")


def binary_descriptor(
    relative_path: str,
    payload: bytes,
    shape: Sequence[int],
    *,
    source_dtype: str,
) -> dict[str, Any]:
    """Describe one finite F32LE tensor payload without embedding it in JSON."""

    elements = math.prod(shape)
    if len(payload) != elements * 4:
        raise ValueError(f"binary payload {relative_path} has inconsistent shape")
    return {
        "relative_path": relative_path,
        "encoding": "ieee754-f32-little-endian",
        "source_dtype": source_dtype,
        "storage_dtype": "float32",
        "shape": list(shape),
        "elements": elements,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "finite": True,
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
    system_space_samples = [
        require_free_space(
            Path("/"),
            MINIMUM_SYSTEM_PREFLIGHT_FREE_BYTES,
            "Nemotron MPS system-volume preflight",
        )
    ]
    output_space_preflight = require_free_space(
        output_dir,
        MINIMUM_OUTPUT_FREE_BYTES,
        "Nemotron immutable bundle preflight",
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
    artifacts, eos_ids, checkpoint_identities, checkpoint_inventory = verify_checkpoint(
        model_dir
    )

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
        NemotronHConfig,
        NemotronHForCausalLM,
        TokenizersBackend,
    )
    from transformers import modeling_utils
    from transformers.models.nemotron_h import (
        configuration_nemotron_h,
        modeling_nemotron_h,
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
            transformers_root / "models/nemotron_h/configuration_nemotron_h.py"
        ).resolve(strict=True),
        "modeling": (
            transformers_root / "models/nemotron_h/modeling_nemotron_h.py"
        ).resolve(strict=True),
    }
    observed_imported_sources = {
        "configuration": Path(configuration_nemotron_h.__file__).resolve(strict=True),
        "modeling": Path(modeling_nemotron_h.__file__).resolve(strict=True),
    }
    if observed_imported_sources != expected_imported_sources:
        raise RuntimeError("imported Nemotron-H source modules escaped the pinned checkout")
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
        raise RuntimeError("Nemotron-H reference generation requires MPS")
    recommended_mps_memory = int(torch.mps.recommended_max_memory())
    required_mps_memory = EXPECTED_BASE_LOADED_BYTES + MINIMUM_MPS_LOAD_HEADROOM_BYTES
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

    config = NemotronHConfig.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, revision=REVISION
    )
    if type(config) is not NemotronHConfig:
        raise RuntimeError(f"loaded unexpected configuration class {type(config)!r}")
    if config.transformers_version != CHECKPOINT_CONFIG_TRANSFORMERS_VERSION:
        raise RuntimeError("parsed checkpoint Transformers metadata changed")
    if (
        config.layer_types != EXPECTED_LAYER_TYPES
        or config.layers_block_type != EXPECTED_LAYER_TYPES
        or config.num_hidden_layers != 52
        or config.num_key_value_heads != 2
        or config.head_dim != 128
        or config.mamba_ssm_cache_dtype != "float32"
        or config.num_nextn_predict_layers != 1
        or config.mtp_layers_block_type != ["full_attention", "moe"]
    ):
        raise RuntimeError("parsed checkpoint cache geometry changed")
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    config.use_mamba_kernels = False
    if NemotronHForCausalLM._keys_to_ignore_on_load_unexpected != [r"mtp.*"]:
        raise RuntimeError("official base model MTP ignore filter changed")
    load_started = time.perf_counter()
    loaded, warmup = load_with_verified_mps_warmup(
        modeling_utils,
        torch,
        NemotronHForCausalLM,
        lambda: NemotronHForCausalLM.from_pretrained(
            model_dir,
            config=config,
            dtype=torch.float16,
            device_map={"": "mps"},
            attn_implementation="eager",
            experts_implementation="eager",
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
    if type(model) is not NemotronHForCausalLM:
        raise RuntimeError(f"loaded unexpected class {type(model)!r}")
    model.eval()
    if (
        model.config._attn_implementation != "eager"
        or model.config._experts_implementation != "eager"
        or model.config.use_mamba_kernels is not False
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
    buffer_count = 0
    buffer_dtype_elements: dict[str, int] = {}
    for name, buffer in model.named_buffers():
        buffer_count += 1
        if not name.endswith(".mixer.gate.e_score_correction_bias"):
            raise RuntimeError(f"unexpected persistent model buffer {name!r}")
        dtype = str(buffer.dtype).removeprefix("torch.")
        buffer_dtype_elements[dtype] = (
            buffer_dtype_elements.get(dtype, 0) + buffer.numel()
        )
    if buffer_count != 23 or buffer_dtype_elements != {
        "float32": EXPECTED_BUFFER_ELEMENTS
    }:
        raise RuntimeError(
            f"loaded correction-buffer inventory changed: "
            f"count={buffer_count}, dtypes={buffer_dtype_elements}"
        )
    state_keys = sorted(model.state_dict())
    if len(state_keys) != EXPECTED_MODEL_STATE_TENSORS or any(
        key.startswith("mtp.") for key in state_keys
    ):
        raise RuntimeError("official base graph state inventory changed")
    if any(parameter.device.type != "mps" for parameter in model.parameters()) or any(
        buffer.device.type != "mps" for buffer in model.buffers()
    ):
        raise RuntimeError("loaded model state escaped MPS")
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
    # The model config retains the pretraining EOS/PAD IDs 2/0 while the
    # checkpoint tokenizer intentionally maps both user-facing EOS and PAD to
    # chat terminator 11. Generation accepts both EOS 2 and 11.
    if tokenizer.eos_token_id != 11 or tokenizer.pad_token_id != 11:
        raise RuntimeError(
            "pinned tokenizer EOS/PAD identity changed: "
            f"{tokenizer.eos_token_id}/{tokenizer.pad_token_id}"
        )
    cycle_token_pieces = tokenizer.convert_ids_to_tokens([1_256, 1_010])
    if cycle_token_pieces != EXPECTED_CYCLE_TOKEN_PIECES:
        raise RuntimeError(
            f"pinned whitespace-cycle token pieces changed: {cycle_token_pieces}"
        )
    input_ids = torch.tensor([INPUT_IDS], dtype=torch.long, device="mps")
    prompt_mask = torch.ones_like(input_ids, dtype=torch.long, device="mps")
    prompt_positions = list(range(len(INPUT_IDS)))

    with torch.inference_mode():
        stateless_started = time.perf_counter()
        stateless, trace_labels, hidden_trace = capture_last_hidden_trace(
            torch,
            model.model,
            len(INPUT_IDS),
            lambda: capture_implicit_position_ids(
                torch,
                model.model.layers[5].mixer,
                prompt_positions,
                lambda: model(
                    input_ids=input_ids,
                    attention_mask=prompt_mask,
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                ),
            ),
        )
        torch.mps.synchronize()
        system_space_samples.append(
            require_free_space(
                Path("/"),
                MINIMUM_SYSTEM_RUNTIME_FREE_BYTES,
                "system reserve after stateless prompt",
            )
        )
        stateless_seconds = checked_duration(
            "stateless prompt", time.perf_counter() - stateless_started
        )
        if (
            list(stateless.logits.shape) != [1, 1, VOCAB_SIZE]
            or stateless.logits.dtype != torch.float32
            or stateless.logits.device.type != "mps"
        ):
            raise RuntimeError(
                "stateless logits shape/dtype/device changed: "
                f"{list(stateless.logits.shape)}/{stateless.logits.dtype}/"
                f"{stateless.logits.device}"
            )
        stateless_row = stateless.logits[0, -1].float().cpu().contiguous()
        if list(stateless_row.shape) != [VOCAB_SIZE] or not bool(torch.isfinite(stateless_row).all()):
            raise RuntimeError("stateless full-vocabulary row is invalid")

        generation_started = time.perf_counter()
        cached, cached_prefill_labels, cached_prefill_hidden_trace = capture_last_hidden_trace(
            torch,
            model.model,
            len(INPUT_IDS),
            lambda: capture_implicit_position_ids(
                torch,
                model.model.layers[5].mixer,
                prompt_positions,
                lambda: model(
                    input_ids=input_ids,
                    attention_mask=prompt_mask,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                ),
            ),
        )
        if cached_prefill_labels != trace_labels:
            raise RuntimeError("cached-prefill hidden-trace labels changed")
        cache = cached.past_key_values
        generated: list[int] = []
        cached_rows = []
        cached_decision_hidden_traces = [cached_prefill_hidden_trace]
        decisions = []
        cached_decode_hidden_trace = None
        cached_decode_labels = None
        decision_logical_lengths = []
        decision_position_ids = []
        first_cache_geometry = None
        final_cache_geometry = None
        for step in range(GREEDY_TOKENS):
            logical_length = len(INPUT_IDS) + step
            cache_geometry = validate_official_cache(
                torch, cache, DynamicCache, logical_length
            )
            if first_cache_geometry is None:
                first_cache_geometry = cache_geometry
            final_cache_geometry = cache_geometry
            if (
                list(cached.logits.shape) != [1, 1, VOCAB_SIZE]
                or cached.logits.dtype != torch.float32
                or cached.logits.device.type != "mps"
            ):
                raise RuntimeError(f"cached output invalid at decision {step}")
            row = cached.logits[0, -1].float().cpu().contiguous()
            if list(row.shape) != [VOCAB_SIZE] or not bool(torch.isfinite(row).all()):
                raise RuntimeError(f"cached vocabulary row invalid at decision {step}")
            system_space_samples.append(
                require_free_space(
                    Path("/"),
                    MINIMUM_SYSTEM_RUNTIME_FREE_BYTES,
                    f"system reserve after cached decision {step}",
                )
            )
            cached_rows.append(row)
            decision_logical_lengths.append(logical_length)
            decision_position_ids.append(logical_length - 1)
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
                decode_position = logical_length

                def run_decode() -> Any:
                    return capture_implicit_position_ids(
                        torch,
                        model.model.layers[5].mixer,
                        [decode_position],
                        lambda: model(
                            input_ids=torch.tensor(
                                [[token_id]], dtype=torch.long, device="mps"
                            ),
                            attention_mask=torch.ones(
                                (1, logical_length + 1),
                                dtype=torch.long,
                                device="mps",
                            ),
                            past_key_values=cache,
                            use_cache=True,
                            logits_to_keep=1,
                            return_dict=True,
                        ),
                    )

                cached, decode_labels, selected_decode_traces = (
                    capture_selected_hidden_traces(
                        torch,
                        model.model,
                        1,
                        [0],
                        run_decode,
                    )
                )
                decode_hidden_trace = selected_decode_traces[0]
                if decode_labels != trace_labels:
                    raise RuntimeError(
                        f"cached decode hidden labels changed at decision {step + 1}"
                    )
                cached_decision_hidden_traces.append(decode_hidden_trace)
                if step == 0:
                    cached_decode_labels = decode_labels
                    cached_decode_hidden_trace = decode_hidden_trace
                cache = cached.past_key_values
        if generated != EXPECTED_GENERATED_TOKEN_IDS:
            raise RuntimeError(
                "pinned official greedy continuation is absent or changed; "
                "review and commit these observed IDs before publication: "
                f"{generated} != {EXPECTED_GENERATED_TOKEN_IDS}"
            )
        decoded_continuation = tokenizer.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded_continuation != EXPECTED_DECODED_CONTINUATION:
            raise RuntimeError(
                "pinned whitespace continuation text changed: "
                f"{decoded_continuation!r}"
            )
        if first_cache_geometry is None or final_cache_geometry is None:
            raise RuntimeError("cache geometry was not captured")
        if cached_decode_hidden_trace is None or cached_decode_labels != trace_labels:
            raise RuntimeError("first recurrent-decode hidden trace was not captured")
        torch.mps.synchronize()
        generation_seconds = checked_duration(
            "cached generation", time.perf_counter() - generation_started
        )

        teacher_ids = INPUT_IDS + generated[:-1]
        positions = list(
            range(len(INPUT_IDS) - 1, len(INPUT_IDS) - 1 + GREEDY_TOKENS)
        )
        teacher_started = time.perf_counter()
        teacher, teacher_trace_labels, teacher_hidden_traces = (
            capture_selected_hidden_traces(
                torch,
                model.model,
                len(teacher_ids),
                positions,
                lambda: capture_implicit_position_ids(
                    torch,
                    model.model.layers[5].mixer,
                    list(range(len(teacher_ids))),
                    lambda: model(
                        input_ids=torch.tensor(
                            [teacher_ids], dtype=torch.long, device="mps"
                        ),
                        attention_mask=torch.ones(
                            (1, len(teacher_ids)), dtype=torch.long, device="mps"
                        ),
                        use_cache=False,
                        logits_to_keep=0,
                        return_dict=True,
                    ),
                ),
            )
        )
        if teacher_trace_labels != trace_labels:
            raise RuntimeError("teacher-forced hidden-trace labels changed")
        if len(cached_decision_hidden_traces) != GREEDY_TOKENS:
            raise RuntimeError(
                "cached hidden traces do not cover every generation decision"
            )
        torch.mps.synchronize()
        system_space_samples.append(
            require_free_space(
                Path("/"),
                MINIMUM_SYSTEM_RUNTIME_FREE_BYTES,
                "system reserve after teacher-forced forward",
            )
        )
        teacher_seconds = checked_duration(
            "teacher-forced stateless", time.perf_counter() - teacher_started
        )
        if (
            list(teacher.logits.shape) != [1, len(teacher_ids), VOCAB_SIZE]
            or teacher.logits.dtype != torch.float32
            or teacher.logits.device.type != "mps"
        ):
            raise RuntimeError(
                "teacher-forced logits shape/dtype/device changed: "
                f"{list(teacher.logits.shape)}/{teacher.logits.dtype}/"
                f"{teacher.logits.device}"
            )
        teacher_logits = teacher.logits[0].float().cpu().contiguous()
        teacher_prefix_logits = teacher_logits[positions].contiguous()
        cached_prefix_logits = torch.stack(cached_rows).contiguous()
        if list(cached_prefix_logits.shape) != [GREEDY_TOKENS, VOCAB_SIZE]:
            raise RuntimeError("cached prefix-logit matrix shape changed")
        if list(teacher_prefix_logits.shape) != [GREEDY_TOKENS, VOCAB_SIZE]:
            raise RuntimeError("teacher prefix-logit matrix shape changed")
        stateless_top1 = [
            int(teacher_prefix_logits[row].argmax().item())
            for row in range(GREEDY_TOKENS)
        ]
        if stateless_top1 != generated:
            raise RuntimeError(
                f"official cached/stateless top-1 mismatch: {generated} != {stateless_top1}"
            )
        prefix_cosines = [
            tensor_cosine(torch, cached_row, teacher_row)
            for cached_row, teacher_row in zip(
                cached_rows, teacher_prefix_logits, strict=True
            )
        ]
        minimum_prefix_cosine = min(prefix_cosines)
        minimum_prefix_cosine_row = prefix_cosines.index(minimum_prefix_cosine)
        logits_path_divergence = vector_comparison_diagnostic(
            cached_rows[minimum_prefix_cosine_row].tolist(),
            teacher_prefix_logits[minimum_prefix_cosine_row].tolist(),
            comparison=(
                "cached_prefill_vs_teacher_forced_stateless"
                if minimum_prefix_cosine_row == 0
                else "cached_recurrent_vs_teacher_forced_stateless"
            ),
            left_name="cached_prefix_logits",
            right_name="teacher_forced_prefix_logits",
            gate_cosine=minimum_prefix_cosine,
        )
        logits_path_divergence["decision"] = minimum_prefix_cosine_row
        logits_path_divergence["cached_binary16_stability"] = decisions[
            minimum_prefix_cosine_row
        ]
        # Decision zero compares the same chunked prefill algorithm with and
        # without cache-state materialization, so it remains a strict 0.999
        # source self-consistency gate. Later decisions compare recurrent Mamba
        # decode against a fresh chunked scan and are retained as two distinct
        # official oracle streams; Burn must match each corresponding stream.
        if prefix_cosines[0] < SELF_PARITY_COSINE:
            comparison = vector_comparison_diagnostic(
                cached_rows[0].tolist(),
                teacher_prefix_logits[0].tolist(),
                comparison="cached_prefill_vs_teacher_forced_stateless",
                left_name="cached_prefill_logits",
                right_name="teacher_forced_prompt_logits",
                gate_cosine=prefix_cosines[0],
            )
            comparison["decision"] = 0
            comparison["cached_binary16_stability"] = decisions[0]
            raise RuntimeError(
                "same-path cached/stateless prefill cosine below "
                f"{SELF_PARITY_COSINE}: diagnostic="
                f"{json.dumps(comparison, allow_nan=False, sort_keys=True)}"
            )
        prompt_cached_cosine = tensor_cosine(torch, stateless_row, cached_rows[0])
        if prompt_cached_cosine < SELF_PARITY_COSINE:
            comparison = vector_comparison_diagnostic(
                stateless_row.tolist(),
                cached_rows[0].tolist(),
                comparison="stateless_prompt_vs_cached_prefill",
                left_name="stateless_prompt_logits",
                right_name="cached_prefill_logits",
                gate_cosine=prompt_cached_cosine,
            )
            raise RuntimeError(
                "prompt cached/stateless cosine is below "
                f"{SELF_PARITY_COSINE}: {prompt_cached_cosine!r}, diagnostic="
                f"{json.dumps(comparison, allow_nan=False, sort_keys=True)}"
            )
        cached_prefill_hidden_cosines = [
            tensor_cosine(torch, stateless_row, cached_row)
            for stateless_row, cached_row in zip(
                hidden_trace, cached_prefill_hidden_trace, strict=True
            )
        ]
        minimum_hidden_cosine = min(cached_prefill_hidden_cosines)
        minimum_hidden_cosine_row = cached_prefill_hidden_cosines.index(
            minimum_hidden_cosine
        )
        if minimum_hidden_cosine < SELF_PARITY_COSINE:
            comparison = vector_comparison_diagnostic(
                hidden_trace[minimum_hidden_cosine_row].tolist(),
                cached_prefill_hidden_trace[minimum_hidden_cosine_row].tolist(),
                comparison="stateless_prompt_hidden_vs_cached_prefill_hidden",
                left_name="stateless_prompt_hidden_trace",
                right_name="cached_prefill_hidden_trace",
                gate_cosine=minimum_hidden_cosine,
            )
            comparison["trace_row"] = minimum_hidden_cosine_row
            comparison["trace_label"] = trace_labels[minimum_hidden_cosine_row]
            raise RuntimeError(
                "cached-prefill/stateless hidden cosine is below "
                f"{SELF_PARITY_COSINE}: minimum={minimum_hidden_cosine!r}, "
                f"trace_row={minimum_hidden_cosine_row}, diagnostic="
                f"{json.dumps(comparison, allow_nan=False, sort_keys=True)}"
            )
        hidden_path_divergence = hidden_path_divergence_summary(
            torch,
            cached_decision_hidden_traces,
            teacher_hidden_traces,
            trace_labels,
        )

    logits_values = [float(value) for value in stateless_row.tolist()]
    logits_payload = tensor_f32le_bytes(torch, stateless_row)
    hidden_payload = tensor_f32le_bytes(torch, hidden_trace)
    cached_logits_payload = tensor_f32le_bytes(torch, cached_prefix_logits)
    teacher_logits_payload = tensor_f32le_bytes(torch, teacher_prefix_logits)
    cached_prefill_hidden_payload = tensor_f32le_bytes(
        torch, cached_prefill_hidden_trace
    )
    cached_decode_hidden_payload = tensor_f32le_bytes(
        torch, cached_decode_hidden_trace
    )
    logits_descriptor = binary_descriptor(
        LOGITS_FILENAME,
        logits_payload,
        [VOCAB_SIZE],
        source_dtype="float32",
    )
    logits_descriptor.update(vector_summary(logits_values))
    logits_descriptor["top1_token_id"] = int(stateless_row.argmax().item())
    hidden_descriptor = binary_descriptor(
        HIDDEN_FILENAME,
        hidden_payload,
        [HIDDEN_TRACE_ROWS, HIDDEN_SIZE],
        source_dtype="float16",
    )
    hidden_descriptor["labels"] = trace_labels
    hidden_descriptor["phase"] = "stateless_prompt"
    hidden_descriptor["token_position"] = len(INPUT_IDS) - 1
    cached_prefill_hidden_descriptor = binary_descriptor(
        CACHED_PREFILL_HIDDEN_FILENAME,
        cached_prefill_hidden_payload,
        [HIDDEN_TRACE_ROWS, HIDDEN_SIZE],
        source_dtype="float16",
    )
    cached_prefill_hidden_descriptor["labels"] = cached_prefill_labels
    cached_prefill_hidden_descriptor["phase"] = "cached_prefill"
    cached_prefill_hidden_descriptor["token_position"] = len(INPUT_IDS) - 1
    cached_decode_hidden_descriptor = binary_descriptor(
        CACHED_DECODE_HIDDEN_FILENAME,
        cached_decode_hidden_payload,
        [HIDDEN_TRACE_ROWS, HIDDEN_SIZE],
        source_dtype="float16",
    )
    cached_decode_hidden_descriptor["labels"] = cached_decode_labels
    cached_decode_hidden_descriptor["phase"] = "cached_decode"
    cached_decode_hidden_descriptor["token_position"] = len(INPUT_IDS)
    cached_logits_descriptor = binary_descriptor(
        CACHED_LOGITS_FILENAME,
        cached_logits_payload,
        [GREEDY_TOKENS, VOCAB_SIZE],
        source_dtype="float32",
    )
    cached_logits_descriptor["row_position_ids"] = decision_position_ids
    cached_logits_descriptor["row_top1_token_ids"] = generated
    teacher_logits_descriptor = binary_descriptor(
        TEACHER_LOGITS_FILENAME,
        teacher_logits_payload,
        [GREEDY_TOKENS, VOCAB_SIZE],
        source_dtype="float32",
    )
    teacher_logits_descriptor["row_position_ids"] = positions
    teacher_logits_descriptor["row_top1_token_ids"] = stateless_top1

    post_reference_source = verify_transformers_checkout(transformers_dir)
    (
        post_dependencies,
        post_distribution_provenance,
        post_distribution_manifest,
        post_distribution_identities,
    ) = verify_distribution_payloads()
    (
        post_artifacts,
        post_eos_ids,
        post_checkpoint_identities,
        post_checkpoint_inventory,
    ) = verify_checkpoint(model_dir)
    transitive_distributions = audit_imported_module_origins(
        transformers_root, distribution_identities
    )
    if (
        post_reference_source["source_identities"]
        != reference_source["source_identities"]
        or post_dependencies != dependencies
        or post_distribution_provenance != distribution_provenance
        or post_distribution_manifest != distribution_manifest
        or post_distribution_identities != distribution_identities
        or post_artifacts != artifacts
        or post_eos_ids != eos_ids
        or post_checkpoint_identities != checkpoint_identities
        or post_checkpoint_inventory != checkpoint_inventory
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
    system_space_samples.append(
        require_free_space(
            Path("/"),
            MINIMUM_SYSTEM_RUNTIME_FREE_BYTES,
            "system reserve before immutable publication",
        )
    )
    output_space_postflight = require_free_space(
        output_dir,
        MINIMUM_OUTPUT_FREE_BYTES,
        "Nemotron immutable bundle postflight",
    )

    script = Path(__file__).resolve(strict=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    record = {
        "schema_version": 3,
        "repository": REPOSITORY,
        "revision": REVISION,
        "reference": {
            "repository": REFERENCE_REPOSITORY,
            "revision": REFERENCE_REVISION,
            "class": REFERENCE_CLASS,
            "source_version": REFERENCE_SOURCE_VERSION,
            "checkpoint_config_transformers_version": CHECKPOINT_CONFIG_TRANSFORMERS_VERSION,
            "sources": reference_source["sources"],
            "checkout_clean": reference_source["clean"],
            "executes_mtp": False,
            "sole_unexpected_key_filter": r"mtp.*",
        },
        "prompt": PROMPT,
        "input_ids": INPUT_IDS,
        "tokenizer": {
            "class": "transformers.TokenizersBackend",
            "add_special_tokens": False,
            "prompt_token_ids": observed_ids,
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "tokenizer_pad_token_id": tokenizer.pad_token_id,
            "model_config_eos_token_id": config.eos_token_id,
            "model_config_pad_token_id": config.pad_token_id,
            "generation_eos_token_ids": eos_ids,
            "whitespace_cycle_token_ids": [1_256, 1_010],
            "whitespace_cycle_token_pieces": cycle_token_pieces,
            "decoded_continuation": decoded_continuation,
        },
        "execution": {
            "dtype": "float16",
            "device": "mps",
            "attention_implementation": "eager",
            "attention_precision": {
                "score_equation": "q @ repeat_kv(k)^T / sqrt(head_dim)",
                "softmax_accumulator_dtype": "float32",
                "probability_output_dtype": "float16",
                "position_encoding": "none_in_pinned_official_attention_source",
            },
            "experts_implementation": "eager_selected_expert_dispatch",
            "mamba_implementation": {
                "prefill": "official_pytorch_chunk_scan_fallback",
                "decode": "official_pytorch_recurrent_state_update_fallback",
                "external_kernels": False,
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
            "cache": {
                "class": "transformers.cache_utils.DynamicCache",
                "batch_size": 1,
                "api_reported_batch_size": cache.batch_size,
                "batch_size_authority": "physical_KV_conv_and_recurrent_tensor_axis_0",
                "layer_count": len(EXPECTED_LAYER_TYPES),
                "layer_types": EXPECTED_LAYER_TYPES,
                "mamba_layers": EXPECTED_LAYER_TYPES.count("linear_attention"),
                "moe_placeholders": EXPECTED_LAYER_TYPES.count("moe"),
                "full_attention_layers": EXPECTED_LAYER_TYPES.count("full_attention"),
                "key_value_heads": 2,
                "head_dimension": 128,
                "attention_tensor_dtype": "float16",
                "mamba_convolution_dtype": "float16",
                "mamba_recurrent_dtype": "float32",
                "tensor_device": "mps",
                "first_decision_geometry": first_cache_geometry,
                "final_decision_geometry": final_cache_geometry,
                "decision_logical_lengths": decision_logical_lengths,
                "decision_physical_lengths": decision_logical_lengths,
                "decision_position_ids": decision_position_ids,
                "decode_input_position_ids": decision_position_ids[1:],
                "teacher_position_ids": list(range(len(teacher_ids))),
                "implicit_position_ids_verified": True,
                "all_layers_validated_at_every_decision": True,
            },
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
            "disk_space": {
                "system_preflight_minimum_free_bytes": (
                    MINIMUM_SYSTEM_PREFLIGHT_FREE_BYTES
                ),
                "system_runtime_minimum_free_bytes": (
                    MINIMUM_SYSTEM_RUNTIME_FREE_BYTES
                ),
                "system_samples": system_space_samples,
                "output_minimum_free_bytes": MINIMUM_OUTPUT_FREE_BYTES,
                "output_preflight": output_space_preflight,
                "output_postflight": output_space_postflight,
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
            "checkpoint_inventory": checkpoint_inventory,
            "physical_tensor_count": EXPECTED_PHYSICAL_TENSORS,
            "physical_checkpoint_tensor_bytes": EXPECTED_PHYSICAL_TENSOR_BYTES,
            "reported_checkpoint_tensor_bytes": EXPECTED_REPORTED_TENSOR_BYTES,
            "base_loaded_tensor_bytes": EXPECTED_BASE_LOADED_BYTES,
            "loaded_parameter_count": parameter_count,
            "parameter_dtype_elements": dtype_elements,
            "persistent_buffer_count": buffer_count,
            "buffer_dtype_elements": buffer_dtype_elements,
            "loaded_state_tensor_count": len(state_keys),
            "loading_info": loading_summary,
            "mtp_filtering": {
                "raw_physical_mtp_tensors": checkpoint_inventory[
                    "mtp_physical_tensors"
                ],
                "raw_mtp_name_sha256": checkpoint_inventory["mtp_name_sha256"],
                "official_filter": r"mtp.*",
                "post_filter_unexpected_keys": loading_summary["unexpected_keys"],
                "remaining_unexplained_anomalies": 0,
                "mtp_tensors_executed": False,
            },
            "integrity_postflight_passed": True,
        },
        "outputs": {
            "prompt_logits": logits_descriptor,
            "hidden_last_trace": hidden_descriptor,
            "cached_prefix_logits": cached_logits_descriptor,
            "teacher_prefix_logits": teacher_logits_descriptor,
            "cached_prefill_hidden_trace": cached_prefill_hidden_descriptor,
            "cached_decode_hidden_trace": cached_decode_hidden_descriptor,
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
            "decoded_text": decoded_continuation,
            "prompt_cached_cosine": prompt_cached_cosine,
            "minimum_cached_stateless_prefix_cosine": minimum_prefix_cosine,
            "minimum_cached_stateless_prefix_cosine_row": (
                minimum_prefix_cosine_row
            ),
            "minimum_cached_prefill_stateless_hidden_cosine": minimum_hidden_cosine,
            "minimum_cached_prefill_stateless_hidden_cosine_row": (
                minimum_hidden_cosine_row
            ),
            "parity_contract": {
                "source_self_consistency": {
                    "cached_teacher_top1_exact_required": True,
                    "cached_teacher_top1_exact_passed": True,
                    "same_path_prefill_minimum_cosine": SELF_PARITY_COSINE,
                    "cached_prefill_teacher_observed_cosine": prefix_cosines[0],
                    "stateless_cached_prefill_observed_cosine": (
                        prompt_cached_cosine
                    ),
                    "stateless_cached_prefill_hidden_minimum_cosine": (
                        minimum_hidden_cosine
                    ),
                    "recurrent_cached_vs_chunked_teacher_cosine_gate": None,
                    "recurrent_cached_vs_chunked_reason": (
                        "distinct official Mamba recurrent and chunk-scan numerical "
                        "paths; divergence is preserved, not normalized"
                    ),
                },
                "cross_runtime": {
                    "minimum_corresponding_path_cosine": (
                        CROSS_RUNTIME_PARITY_COSINE
                    ),
                    "cached_comparison": "Burn cached rows vs official cached rows",
                    "stateless_comparison": (
                        "Burn stateless rows vs official teacher-forced rows"
                    ),
                    "crossing_official_paths_is_not_a_cross_runtime_gate": True,
                },
            },
            "official_path_divergence": {
                "status": "observed_and_preserved",
                "minimum_logits_cosine": minimum_prefix_cosine,
                "minimum_logits_cosine_decision": minimum_prefix_cosine_row,
                "worst_logits_comparison": logits_path_divergence,
                "hidden_localization": hidden_path_divergence,
                "possible_amplifier_not_proven": (
                    "small recurrent/chunked boundary differences can cross "
                    "discontinuous top-6 MoE routing boundaries"
                ),
            },
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
            CACHED_LOGITS_FILENAME: cached_logits_payload,
            TEACHER_LOGITS_FILENAME: teacher_logits_payload,
            CACHED_PREFILL_HIDDEN_FILENAME: cached_prefill_hidden_payload,
            CACHED_DECODE_HIDDEN_FILENAME: cached_decode_hidden_payload,
        },
    )
    if published != bundle:
        raise RuntimeError(f"published unexpected bundle path: {published}")
    print(f"published={published}")
    print(f"json_sha256={sha256_bytes(json_payload)}")
    print(f"logits_sha256={sha256_bytes(logits_payload)}")
    print(f"hidden_sha256={sha256_bytes(hidden_payload)}")
    print(f"cached_logits_sha256={sha256_bytes(cached_logits_payload)}")
    print(f"teacher_logits_sha256={sha256_bytes(teacher_logits_payload)}")
    print(
        "cached_prefill_hidden_sha256="
        f"{sha256_bytes(cached_prefill_hidden_payload)}"
    )
    print(
        "cached_decode_hidden_sha256="
        f"{sha256_bytes(cached_decode_hidden_payload)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
