#!/usr/bin/env python3
"""Generate the pinned Qwen3.8-27B text-only F16 acceptance reference.

This program is intentionally an offline, maintainer-run acceptance tool rather
than part of the Rust runtime.  It instantiates only Transformers'
``Qwen3_5ForCausalLM`` from a pinned local multimodal checkpoint, relying on the
official ``qwen3_5_text`` checkpoint conversion to discard the vision and MTP
tensors.  No Hub client, remote code, sampling, or implicit generation helper is
used.

The output directory must already exist outside this crate and outside the model
checkout.  The two files are fsynced inside a hidden, same-filesystem staging
directory.  That directory is then published as one fixed-name bundle with an
atomic, no-replace rename.  A failed run never exposes a partial final bundle and
leaves its hidden staging directory intact for forensic recovery.
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
from typing import Any, Iterable, Mapping, Sequence


# Offline controls are installed before importing Transformers or Torch.  Local
# calls below repeat `local_files_only=True` and `trust_remote_code=False`; the
# environment is a second, process-wide guard against accidental Hub access.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


REPOSITORY = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
REFERENCE_CLASS = "transformers.Qwen3_5ForCausalLM"
# A non-acceptance selection diagnostic completed all 32 decisions with a
# minimum margin of 0.8828125, a minimum margin-to-epsilon ratio of 113.0, and
# cached/stateless top-1 agreement (minimum cosine 0.9998457587791264); its
# continuation repeated [799, 1330, 2250].  This six-token repetition of three
# distinct words therefore avoids the late near-boundary choice in the
# natural-language fixture while exercising varied recurrent states.  Initial
# `one` and space-prefixed ` one` intentionally have different IDs; the repeated
# ` two three` suffix makes the continuation cycle and whitespace-sensitive
# tokenization structurally auditable.
PROMPT = "one two three one two three"
INPUT_IDS = [588, 1_330, 2_250, 799, 1_330, 2_250]
VOCAB_SIZE = 248_320
GREEDY_TOKENS = 32
EXPECTED_EOS_TOKEN_IDS = [248_046, 248_044]
EXPECTED_PARAMETER_COUNT = 26_895_998_464
MAX_EXACT_JSON_INTEGER = (1 << 53) - 1
BINARY16_ELEMENTWISE_PERTURBATION_ULPS = 1.0
TOP1_STABILITY_SAFETY_FACTOR = 2.0
TOP1_STABILITY_POLICY = "binary16-elementwise-ulp-bound"
TOP1_STABILITY_RATIONALE = (
    "If every score changes by at most epsilon, the top score can fall by epsilon "
    "while its strongest competitor rises by epsilon; margin > 2*epsilon therefore "
    "preserves top-1. Epsilon is one local binary16 ULP of the wider-spaced top-two "
    "logit, budgeting one representable step per score. This fixture screen complements, "
    "and does not replace, exact Burn/reference token parity."
)
SELF_PARITY_COSINE = 0.999
RAW_FILENAME = "qwen3.8-27b-1d4bf0f2-six-token-f16-last-logits.f32le"
JSON_FILENAME = "qwen3.8-27b-1d4bf0f2-reference.json"
BUNDLE_DIRNAME = "qwen3.8-27b-1d4bf0f2-reference.bundle"
ALLOCATOR_WARMUP_TARGET = "transformers.modeling_utils.caching_allocator_warmup"
ALLOCATOR_WARMUP_POLICY = "mps-exclusive-noop-during-from-pretrained"
ALLOCATOR_WARMUP_SCOPE = "Qwen3_5ForCausalLM.from_pretrained"
ALLOCATOR_WARMUP_REASON = (
    "Transformers 5.2 caching allocator warmup attempts an MPS allocation with a "
    "one-dimensional tensor exceeding INT_MAX; skipping this performance-only "
    "allocation does not alter checkpoint tensors or model math."
)
ALLOCATOR_WARMUP_SOURCE_MODULE = "transformers.modeling_utils"
EXPECTED_ALLOCATOR_WARMUP_CALLS = 1


EXPECTED_PACKAGE_VERSIONS = {
    "accelerate": "1.12.0",
    "numpy": "2.4.3",
    "safetensors": "0.7.0",
    "tokenizers": "0.22.2",
    "torch": "2.10.0",
    "transformers": "5.2.0",
}


EXPECTED_REFERENCE_SOURCES = {
    "transformers.cache_utils": {
        "relative_path": "cache_utils.py",
        "bytes": 60_147,
        "sha256": "e4fe20ca1eeb17ea0ce7e62d1dbd0d6eab32e7072044f293612403b8e0c477fd",
    },
    "transformers.configuration_utils": {
        "relative_path": "configuration_utils.py",
        "bytes": 57_894,
        "sha256": "52100272e3bf59551c072a064053e90bc30f1fd0aa9e48fac7ad4b675c63ebcc",
    },
    "transformers.conversion_mapping": {
        "relative_path": "conversion_mapping.py",
        "bytes": 19_302,
        "sha256": "c8efcc6ce2224939299804d495ba4f361ab596e59f33e3db28e9b4629fc481e0",
    },
    "transformers.masking_utils": {
        "relative_path": "masking_utils.py",
        "bytes": 74_373,
        "sha256": "c3c82f7b7b6e03d3f04ba6c6c58a3dd6910623636452ec67ef70e3eb522f9fe7",
    },
    "transformers.modeling_attn_mask_utils": {
        "relative_path": "modeling_attn_mask_utils.py",
        "bytes": 21_930,
        "sha256": "fb3ec46639a19fda1eb99339f4e7770332a0f4f75b9de4abb34a0b1ee379838b",
    },
    "transformers.modeling_utils": {
        "relative_path": "modeling_utils.py",
        "bytes": 247_340,
        "sha256": "2f45c2139ce633e6e3eae9290675dd5815643b37c27f2cc4dd83586fb648f7bb",
    },
    "transformers.models.qwen3_5.configuration_qwen3_5": {
        "relative_path": "models/qwen3_5/configuration_qwen3_5.py",
        "bytes": 14_576,
        "sha256": "2280c6e6bd9d66d7281155243f67cf5fed2756a828af41316566afe611ff16c0",
    },
    "transformers.models.qwen3_5.modeling_qwen3_5": {
        "relative_path": "models/qwen3_5/modeling_qwen3_5.py",
        "bytes": 97_339,
        "sha256": "d1ae3856f53763591ec65054129af46e003a1715efcf7eef2b752a31e85526b8",
    },
}


EXPECTED_DISTRIBUTION_FILES = {
    "accelerate": {
        "METADATA": {
            "relative_path": "accelerate-1.12.0.dist-info/METADATA",
            "bytes": 19_601,
            "sha256": "31d880e2aa1eaffb50dfecb62bc81d53f0262ed19ff374e7ff7a76b41b87c6b8",
        },
        "RECORD": {
            "relative_path": "accelerate-1.12.0.dist-info/RECORD",
            "bytes": 14_253,
            "sha256": "ebc6c296fe7d6bf449f6b776a4839cba2c8e8d8b649346ccfd2d505feea18f62",
        },
    },
    "numpy": {
        "METADATA": {
            "relative_path": "numpy-2.4.3.dist-info/METADATA",
            "bytes": 6_608,
            "sha256": "621bf875d57eee8673f39344ea98ddc370e944e9c8c36b38b430e290f2b682cf",
        },
        "RECORD": {
            "relative_path": "numpy-2.4.3.dist-info/RECORD",
            "bytes": 126_949,
            "sha256": "b4b4d441229ce2edbbd2417b0b993d7b72d0ac801770fe2d32b26d8e1ab29829",
        },
    },
    "safetensors": {
        "METADATA": {
            "relative_path": "safetensors-0.7.0.dist-info/METADATA",
            "bytes": 4_125,
            "sha256": "6809326fbf17444dd520ab19f0a55b7a18476dbe52a430466bbf5389967c2aaa",
        },
        "RECORD": {
            "relative_path": "safetensors-0.7.0.dist-info/RECORD",
            "bytes": 1_573,
            "sha256": "f4f493f241d26041cf94fe7214f88c8f255df145eb064abac8550bc533d38f36",
        },
    },
    "tokenizers": {
        "METADATA": {
            "relative_path": "tokenizers-0.22.2.dist-info/METADATA",
            "bytes": 7_254,
            "sha256": "15a5ddaf489f592b77e0a934c0eeb46b51130a94064ca129232afdb0a3868efc",
        },
        "RECORD": {
            "relative_path": "tokenizers-0.22.2.dist-info/RECORD",
            "bytes": 3_778,
            "sha256": "cd5bb15216c5d083bbe39ccc172f58e4114a8fe5b786f4f8f8239e53053ad9c3",
        },
    },
    "torch": {
        "METADATA": {
            "relative_path": "torch-2.10.0.dist-info/METADATA",
            "bytes": 31_092,
            "sha256": "4b0f1217de69037355b4e582905cdb303a56a488ac6fd172f148d88fed634f54",
        },
        "RECORD": {
            "relative_path": "torch-2.10.0.dist-info/RECORD",
            "bytes": 1_477_169,
            "sha256": "22546c990d209800cd92db2b052a8ea478cee3a6e9253485f282221e0a6225ae",
        },
    },
    "transformers": {
        "METADATA": {
            "relative_path": "transformers-5.2.0.dist-info/METADATA",
            "bytes": 32_319,
            "sha256": "4340d0f550bf77d6595e4efafe066cd88306307577701df8e558ad620551c65f",
        },
        "RECORD": {
            "relative_path": "transformers-5.2.0.dist-info/RECORD",
            "bytes": 394_090,
            "sha256": "98d165cb699677711c15cc42eeeb25f7889fe164ecd90135f34e4b88109f931e",
        },
    },
}
EXPECTED_DISTRIBUTION_MANIFEST_SHA256 = (
    "b814bb3254d51a28c6d74e80c04c3532c9836a0aadc4fab0eed410d6612d59dd"
)


EXPECTED_CONTROL_HASHES = {
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "config.json": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
    "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    "model.safetensors.index.json": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
}


EXPECTED_SHARD_HASHES = {
    "model-00001-of-00018.safetensors": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c",
    "model-00002-of-00018.safetensors": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb",
    "model-00003-of-00018.safetensors": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968",
    "model-00004-of-00018.safetensors": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a",
    "model-00005-of-00018.safetensors": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701",
    "model-00006-of-00018.safetensors": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0",
    "model-00007-of-00018.safetensors": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892",
    "model-00008-of-00018.safetensors": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79",
    "model-00009-of-00018.safetensors": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8",
    "model-00010-of-00018.safetensors": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9",
    "model-00011-of-00018.safetensors": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8",
    "model-00012-of-00018.safetensors": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55",
    "model-00013-of-00018.safetensors": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a",
    "model-00014-of-00018.safetensors": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b",
    "model-00015-of-00018.safetensors": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22",
    "model-00016-of-00018.safetensors": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe",
    "model-00017-of-00018.safetensors": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8",
    "model-00018-of-00018.safetensors": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06",
}


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for an in-memory payload."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Stream one file into SHA-256 without allocating a shard-sized buffer."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def json_without_duplicate_keys(path: Path) -> Any:
    """Parse JSON while rejecting duplicate object keys as ambiguous input."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicates)


def validated_generation_eos_token_ids(config: Mapping[str, Any]) -> list[int]:
    """Return the pinned ordered EOS IDs from the official generation config."""

    eos_token_ids = config.get("eos_token_id")
    if (
        not isinstance(eos_token_ids, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in eos_token_ids)
        or eos_token_ids != EXPECTED_EOS_TOKEN_IDS
    ):
        raise ValueError(
            "generation config EOS IDs changed: "
            f"expected {EXPECTED_EOS_TOKEN_IDS}, found {eos_token_ids!r}"
        )
    return list(eos_token_ids)


def checked_json_nonnegative_integer(value: Any, field: str) -> int:
    """Canonicalize an exact JSON integer within the interoperable F64 range.

    JSON itself has no integer/float distinction, and some checkpoint producers
    serialize byte counts with a trailing ``.0``.  Values through ``2^53 - 1``
    have an unambiguous integer representation in both binary64 JSON consumers
    and Rust's ``u64`` schema.  Booleans are rejected explicitly because Python
    otherwise treats them as integers.
    """

    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError(f"{field} must be a JSON integer, found {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite, found {value!r}")
        if not value.is_integer():
            raise ValueError(f"{field} must be exactly integral, found {value!r}")
    if value < 0 or value > MAX_EXACT_JSON_INTEGER:
        raise ValueError(
            f"{field} must lie in 0..={MAX_EXACT_JSON_INTEGER}, found {value!r}"
        )
    canonical = int(value)
    if isinstance(value, float) and float(canonical) != value:
        raise ValueError(f"{field} cannot be losslessly canonicalized, found {value!r}")
    return canonical


def f32le_bytes(values: Iterable[float]) -> bytes:
    """Encode values as canonical little-endian IEEE-754 binary32 bytes."""

    floats = array.array("f", values)
    if floats.itemsize != 4:
        raise RuntimeError(f"platform float array item size is {floats.itemsize}, expected 4")
    if sys.byteorder != "little":
        floats.byteswap()
    return floats.tobytes()


def top_k_records(values: Sequence[float], count: int) -> list[dict[str, float | int]]:
    """Return stable descending top-k records, breaking equal logits by token ID."""

    if count <= 0 or count > len(values):
        raise ValueError(f"top-k count {count} is outside 1..={len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("cannot rank values containing a non-finite number")
    winners = heapq.nsmallest(count, enumerate(values), key=lambda pair: (-pair[1], pair[0]))
    return [{"token_id": index, "logit": float(value)} for index, value in winners]


def vector_summary(values: Sequence[float], probe_indices: Sequence[int]) -> dict[str, Any]:
    """Compute deterministic integrity statistics for one F32 vocabulary vector."""

    if not values:
        raise ValueError("cannot summarize an empty vector")
    if len(set(probe_indices)) != len(probe_indices):
        raise ValueError("probe indices must be unique")
    if any(index < 0 or index >= len(values) for index in probe_indices):
        raise ValueError("probe index lies outside the vector")
    finite = all(math.isfinite(value) for value in values)
    if not finite:
        raise ValueError("reference logits contain a non-finite value")
    total = math.fsum(float(value) for value in values)
    squared_norm = math.fsum(float(value) * float(value) for value in values)
    norm = math.sqrt(squared_norm)
    if not all(math.isfinite(value) for value in (total, squared_norm, norm)):
        raise ValueError("reference-logit summary produced a non-finite value")
    return {
        "finite": True,
        "l2_norm": norm,
        "sum": total,
        "top10": top_k_records(values, 10),
        "probes": [
            {"token_id": index, "logit": float(values[index])} for index in probe_indices
        ],
    }


def binary16_ulp(value: float) -> float:
    """Return the wider adjacent binary16 spacing at one exact finite F16 value."""

    if not math.isfinite(value):
        raise ValueError("binary16 ULP is undefined for a non-finite value")
    try:
        roundtrip = struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError as error:
        raise ValueError(f"value {value!r} lies outside finite binary16") from error
    if roundtrip != value:
        raise ValueError(f"value {value!r} is not exactly representable as binary16")
    magnitude = abs(value)
    if magnitude < 2.0**-14:
        return 2.0**-24
    exponent = math.floor(math.log2(magnitude))
    return math.ldexp(1.0, exponent - 10)


def greedy_decision_stability(
    *,
    step: int,
    top1_token_id: int,
    top2_token_id: int,
    top1_logit: float,
    top2_logit: float,
    eos_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Measure one top-two decision against a binary16 perturbation proof budget."""

    if top1_token_id == top2_token_id:
        raise RuntimeError(f"greedy top-two token IDs are equal at step {step}")
    if top1_token_id in eos_token_ids:
        raise RuntimeError(
            f"official generation reached EOS token {top1_token_id} at decision {step}; "
            f"all {GREEDY_TOKENS} decisions are required"
        )
    if not math.isfinite(top1_logit) or not math.isfinite(top2_logit):
        raise RuntimeError(f"greedy top-two logits became non-finite at step {step}")
    if top1_logit < top2_logit:
        raise RuntimeError(f"greedy top-two logits are out of order at step {step}")

    margin = top1_logit - top2_logit
    top1_ulp = binary16_ulp(top1_logit)
    top2_ulp = binary16_ulp(top2_logit)
    elementwise_epsilon = (
        max(top1_ulp, top2_ulp) * BINARY16_ELEMENTWISE_PERTURBATION_ULPS
    )
    margin_to_epsilon_ratio = margin / elementwise_epsilon
    values = (
        margin,
        top1_ulp,
        top2_ulp,
        elementwise_epsilon,
        margin_to_epsilon_ratio,
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"greedy stability evidence became non-finite at step {step}")
    return {
        "step": step,
        "top1_token_id": top1_token_id,
        "top2_token_id": top2_token_id,
        "top1_logit": top1_logit,
        "top2_logit": top2_logit,
        "margin": margin,
        "top1_f16_ulp": top1_ulp,
        "top2_f16_ulp": top2_ulp,
        "elementwise_epsilon": elementwise_epsilon,
        "margin_to_epsilon_ratio": margin_to_epsilon_ratio,
        "stable_under_budget": margin_to_epsilon_ratio > TOP1_STABILITY_SAFETY_FACTOR,
    }


def validate_decision_stability(decisions: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Validate all collected decisions and return their minimum margin and ratio."""

    if not decisions:
        raise RuntimeError("cannot validate an empty greedy-decision sequence")
    minimum_margin = min(float(decision["margin"]) for decision in decisions)
    minimum_ratio = min(
        float(decision["margin_to_epsilon_ratio"]) for decision in decisions
    )
    if not math.isfinite(minimum_margin) or not math.isfinite(minimum_ratio):
        raise RuntimeError("greedy stability minima are non-finite")
    unstable = [decision for decision in decisions if not decision["stable_under_budget"]]
    if unstable:
        unstable_steps = [int(decision["step"]) for decision in unstable]
        raise RuntimeError(
            f"binary16 top-1 stability failed at steps {unstable_steps}: "
            f"minimum_margin={minimum_margin}, minimum_margin_to_epsilon_ratio="
            f"{minimum_ratio}, required_ratio>{TOP1_STABILITY_SAFETY_FACTOR}; "
            f"unstable={json.dumps(unstable, allow_nan=False, sort_keys=True)}"
        )
    return minimum_margin, minimum_ratio


def strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize canonical human-readable JSON while rejecting NaN and infinity."""

    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_create_only(path: Path, payload: bytes) -> None:
    """Create and fsync one output, failing rather than replacing any path."""

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
    """Fsync directory metadata so prior file or rename operations are durable."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def path_lexists(path: Path) -> bool:
    """Return whether a path or dangling symlink occupies one directory entry."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def rename_directory_create_only(source: Path, destination: Path) -> None:
    """Atomically rename one directory without replacing any destination entry.

    Darwin's ``renamex_np(RENAME_EXCL)`` and Linux's
    ``renameat2(RENAME_NOREPLACE)`` provide the required atomic create-only
    primitive.  Refusing unsupported platforms is safer than silently falling
    back to ``os.rename``, which may replace an existing empty directory.
    """

    if source.parent != destination.parent:
        raise ValueError("atomic bundle publication requires sibling directories")
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
        raise RuntimeError(
            f"atomic no-replace bundle publication is unsupported on {sys.platform!r}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )


def publish_reference_bundle(
    output_dir: Path, raw_payload: bytes, json_payload: bytes
) -> Path:
    """Durably publish both reference files as one atomic create-only bundle.

    Any exception before the rename deliberately leaves the hidden staging
    directory behind.  Its complete or partial contents can therefore be
    inspected after an interrupted multi-hour reference run; a later invocation
    uses a fresh unique staging name and never mistakes stale evidence for the
    final bundle.
    """

    bundle_path = output_dir / BUNDLE_DIRNAME
    if path_lexists(bundle_path):
        raise FileExistsError(f"create-only reference bundle already exists: {bundle_path}")
    staging_path = Path(
        tempfile.mkdtemp(prefix=f".{BUNDLE_DIRNAME}.staging-", dir=output_dir)
    )
    try:
        write_create_only(staging_path / RAW_FILENAME, raw_payload)
        write_create_only(staging_path / JSON_FILENAME, json_payload)
        os.chmod(staging_path, 0o755)
        fsync_directory(staging_path)
        rename_directory_create_only(staging_path, bundle_path)
        fsync_directory(output_dir)
    except BaseException as error:
        error.add_note(f"recoverable reference staging directory: {staging_path}")
        raise
    return bundle_path


def canonical_artifact_digest(artifacts: Sequence[Mapping[str, Any]]) -> str:
    """Bind all sorted artifact paths, byte lengths, and SHA-256 values together."""

    lines = [
        f"{item['relative_path']}\0{item['bytes']}\0{item['sha256']}\n"
        for item in sorted(artifacts, key=lambda item: str(item["relative_path"]))
    ]
    return sha256_bytes("".join(lines).encode("utf-8"))


def ensure_external_paths(model_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Validate the pinned input, output parent, and absent final bundle path."""

    model_dir = model_dir.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve(strict=True)
    crate_dir = Path(__file__).resolve().parents[1]
    if not model_dir.is_dir():
        raise ValueError(f"model path is not a directory: {model_dir}")
    if not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    if output_dir == crate_dir or output_dir.is_relative_to(crate_dir):
        raise ValueError(f"output directory must be external to the crate: {output_dir}")
    if output_dir == model_dir or output_dir.is_relative_to(model_dir):
        raise ValueError(f"output directory must be outside the checkpoint: {output_dir}")
    bundle_path = output_dir / BUNDLE_DIRNAME
    if path_lexists(bundle_path):
        raise FileExistsError(f"create-only reference bundle already exists: {bundle_path}")
    return model_dir, output_dir, bundle_path


def verified_distribution_file(
    distribution: Any, distribution_name: str, logical_name: str, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify one installed dist-info file and return its strict identity record."""

    files = distribution.files
    if files is None:
        raise RuntimeError(f"installed distribution {distribution_name!r} has no file list")
    candidates = [
        entry
        for entry in files
        if entry.name == logical_name and entry.parent.name.endswith(".dist-info")
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"installed distribution {distribution_name!r} has {len(candidates)} "
            f"candidate {logical_name} files"
        )
    relative_path = str(candidates[0])
    if relative_path != expected["relative_path"]:
        raise RuntimeError(
            f"installed distribution {distribution_name!r} {logical_name} path changed: "
            f"expected {expected['relative_path']!r}, found {relative_path!r}"
        )
    unresolved_path = Path(distribution.locate_file(candidates[0]))
    if unresolved_path.is_symlink():
        raise RuntimeError(
            f"installed distribution {distribution_name!r} {logical_name} is a symlink"
        )
    path = unresolved_path.resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(
            f"installed distribution {distribution_name!r} {logical_name} is not a file"
        )
    observed = {
        "relative_path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if observed != dict(expected):
        raise RuntimeError(
            f"installed distribution {distribution_name!r} {logical_name} changed: "
            f"expected {dict(expected)}, found {observed}"
        )
    return observed


def canonical_distribution_digest(
    distributions: Mapping[str, Mapping[str, Any]],
) -> str:
    """Bind sorted distribution names, versions, METADATA, and RECORD identities."""

    lines: list[str] = []
    for name in sorted(distributions):
        distribution = distributions[name]
        lines.append(f"{name}\0{distribution['version']}\n")
        for logical_name in ("metadata", "record"):
            item = distribution[logical_name]
            lines.append(
                f"{logical_name}\0{item['relative_path']}\0{item['bytes']}\0"
                f"{item['sha256']}\n"
            )
    return sha256_bytes("".join(lines).encode("utf-8"))


def verify_packages() -> tuple[dict[str, str], dict[str, dict[str, Any]], str]:
    """Require exact versions plus installed-wheel METADATA and RECORD identities."""

    versions: dict[str, str] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for name, expected_version in EXPECTED_PACKAGE_VERSIONS.items():
        distribution = importlib.metadata.distribution(name)
        version = distribution.version
        if version != expected_version:
            raise RuntimeError(
                f"reference package {name!r} changed: expected version "
                f"{expected_version}, found {version}"
            )
        expected_files = EXPECTED_DISTRIBUTION_FILES[name]
        versions[name] = version
        provenance[name] = {
            "version": version,
            "metadata": verified_distribution_file(
                distribution, name, "METADATA", expected_files["METADATA"]
            ),
            "record": verified_distribution_file(
                distribution, name, "RECORD", expected_files["RECORD"]
            ),
        }
    manifest_digest = canonical_distribution_digest(provenance)
    if manifest_digest != EXPECTED_DISTRIBUTION_MANIFEST_SHA256:
        raise RuntimeError(
            "installed distribution manifest changed: expected "
            f"{EXPECTED_DISTRIBUTION_MANIFEST_SHA256}, found {manifest_digest}"
        )
    return versions, provenance, manifest_digest


def verify_reference_sources(transformers_root: Path) -> list[dict[str, Any]]:
    """Hash the exact Transformers model, conversion, mask, and cache sources."""

    transformers_root = transformers_root.resolve(strict=True)
    sources: list[dict[str, Any]] = []
    for module, expected in sorted(EXPECTED_REFERENCE_SOURCES.items()):
        relative_path = Path(expected["relative_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"invalid pinned Transformers source path {relative_path}")
        unresolved_path = transformers_root / relative_path
        if unresolved_path.is_symlink():
            raise RuntimeError(f"Transformers source {relative_path} is a symlink")
        path = unresolved_path.resolve(strict=True)
        if not path.is_relative_to(transformers_root) or not path.is_file():
            raise RuntimeError(f"Transformers source escapes its package: {relative_path}")
        observed = {
            "relative_path": relative_path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if observed != expected:
            raise RuntimeError(
                f"official reference source {module} changed: "
                f"expected {expected}, found {observed}"
            )
        sources.append({"module": module, "path": str(path), **observed})
    return sources


def load_with_verified_mps_allocator_warmup_noop(
    modeling_utils: Any,
    torch: Any,
    expected_model_class: type,
    loader: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run one model load with only the unsafe MPS allocator warmup disabled.

    Transformers 5.2 models its optional allocator warmup as one flat F16
    tensor.  The Qwen3.8 element count exceeds Metal's signed 32-bit dimension
    limit even though the real parameter tensors are individually valid.  This
    wrapper replaces only that performance hook, accepts only an all-MPS device
    map and an unquantized Qwen model, and restores the exact original callable
    even when loading raises.
    """

    if getattr(modeling_utils, "__name__", None) != ALLOCATOR_WARMUP_SOURCE_MODULE:
        raise RuntimeError("allocator warmup owner is not transformers.modeling_utils")
    original = getattr(modeling_utils, "caching_allocator_warmup", None)
    if not callable(original):
        raise RuntimeError("Transformers allocator warmup target is not callable")
    if (
        getattr(original, "__module__", None) != ALLOCATOR_WARMUP_SOURCE_MODULE
        or getattr(original, "__name__", None) != "caching_allocator_warmup"
    ):
        raise RuntimeError("Transformers allocator warmup callable identity changed")

    call_count = 0
    expanded_device_map_entries = 0
    expanded_device_types: list[str] = []

    def verified_mps_noop(
        model: Any, expanded_device_map: Any, hf_quantizer: Any
    ) -> None:
        """Validate the intercepted invocation and skip its performance allocation."""

        nonlocal call_count, expanded_device_map_entries, expanded_device_types
        call_count += 1
        if call_count > EXPECTED_ALLOCATOR_WARMUP_CALLS:
            raise RuntimeError("Transformers allocator warmup was invoked more than once")
        if type(model) is not expected_model_class:
            raise RuntimeError(
                "allocator warmup received an unexpected model class: "
                f"{type(model)!r}"
            )
        if hf_quantizer is not None:
            raise RuntimeError("allocator warmup no-op is forbidden for quantized loading")
        if not isinstance(expanded_device_map, dict) or not expanded_device_map:
            raise RuntimeError("allocator warmup requires a nonempty expanded device map")
        if any(not isinstance(name, str) for name in expanded_device_map):
            raise RuntimeError("expanded device-map parameter names must be strings")

        device_types: set[str] = set()
        for assignment in expanded_device_map.values():
            try:
                device = torch.device(assignment)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"invalid expanded device-map assignment {assignment!r}"
                ) from error
            device_types.add(str(device.type))
        if device_types != {"mps"}:
            raise RuntimeError(
                "allocator warmup no-op requires an exclusively MPS expanded device map, "
                f"found {sorted(device_types)}"
            )
        expanded_device_map_entries = len(expanded_device_map)
        expanded_device_types = sorted(device_types)

    setattr(modeling_utils, "caching_allocator_warmup", verified_mps_noop)
    patch_remained_scoped = False
    try:
        loaded = loader()
        patch_remained_scoped = (
            modeling_utils.caching_allocator_warmup is verified_mps_noop
        )
    finally:
        setattr(modeling_utils, "caching_allocator_warmup", original)

    if not patch_remained_scoped:
        raise RuntimeError("allocator warmup override was modified during model loading")
    if modeling_utils.caching_allocator_warmup is not original:
        raise RuntimeError("original allocator warmup callable was not restored")
    if call_count != EXPECTED_ALLOCATOR_WARMUP_CALLS:
        raise RuntimeError(
            "Transformers allocator warmup invocation count changed: "
            f"expected {EXPECTED_ALLOCATOR_WARMUP_CALLS}, found {call_count}"
        )
    provenance = {
        "target": ALLOCATOR_WARMUP_TARGET,
        "policy": ALLOCATOR_WARMUP_POLICY,
        "scope": ALLOCATOR_WARMUP_SCOPE,
        "reason": ALLOCATOR_WARMUP_REASON,
        "upstream_source_module": ALLOCATOR_WARMUP_SOURCE_MODULE,
        "call_count": call_count,
        "expanded_device_map_entries": expanded_device_map_entries,
        "expanded_device_types": expanded_device_types,
        "hf_quantizer_present": False,
        "restored": True,
    }
    return loaded, provenance


def verify_checkpoint(
    model_dir: Path,
) -> tuple[list[dict[str, Any]], int, str, list[int]]:
    """Hash every artifact and return the exact official generation EOS IDs."""

    config = json_without_duplicate_keys(model_dir / "config.json")
    if config.get("model_type") != "qwen3_5":
        raise ValueError("outer config must declare model_type=qwen3_5")
    text_config = config.get("text_config", {})
    expected_text_fields = {
        "model_type": "qwen3_5_text",
        "vocab_size": VOCAB_SIZE,
        "num_hidden_layers": 64,
        "mtp_num_hidden_layers": 1,
    }
    for name, expected in expected_text_fields.items():
        if text_config.get(name) != expected:
            raise ValueError(
                f"text config {name!r} changed: expected {expected!r}, "
                f"found {text_config.get(name)!r}"
            )

    generation_config = json_without_duplicate_keys(model_dir / "generation_config.json")
    eos_token_ids = validated_generation_eos_token_ids(generation_config)

    index = json_without_duplicate_keys(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint index has no object-valued weight_map")
    shard_names = set(weight_map.values())
    if shard_names != set(EXPECTED_SHARD_HASHES):
        raise ValueError(
            "checkpoint shard set differs from the pinned 18-shard revision: "
            f"{sorted(shard_names)}"
        )
    for name in shard_names:
        if Path(name).name != name:
            raise ValueError(f"checkpoint index contains a non-local shard path: {name!r}")
    index_metadata = index.get("metadata")
    if not isinstance(index_metadata, dict):
        raise ValueError("checkpoint index has no object-valued metadata")
    tensor_bytes = checked_json_nonnegative_integer(
        index_metadata.get("total_size"), "checkpoint index metadata.total_size"
    )
    if tensor_bytes != 55_562_855_904:
        raise ValueError(
            "checkpoint tensor-byte declaration changed: "
            f"expected 55562855904, found {tensor_bytes!r}"
        )

    artifacts: list[dict[str, Any]] = []
    expected_hashes = EXPECTED_CONTROL_HASHES | EXPECTED_SHARD_HASHES
    for relative_path, expected_hash in sorted(expected_hashes.items()):
        path = model_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"pinned artifact is missing: {path}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {relative_path}: "
                f"expected {expected_hash}, found {observed_hash}"
            )
        artifacts.append(
            {
                "relative_path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": observed_hash,
            }
        )
    return artifacts, tensor_bytes, canonical_artifact_digest(artifacts), eos_token_ids


def git_provenance(script_path: Path) -> dict[str, Any]:
    """Record the containing Git revision while retaining the script content hash."""

    crate_dir = script_path.parents[1]
    repository_dir = crate_dir.parents[1]

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "relative_path": "scripts/generate_qwen38_reference.py",
        "sha256": sha256_file(script_path),
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", str(script_path))),
    }


def loading_info_summary(loading_info: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Transformers loading diagnostics into stable counts and digests."""

    summary: dict[str, Any] = {}
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = sorted(str(value) for value in loading_info.get(name, []))
        summary[name] = {
            "count": len(values),
            "sha256": sha256_bytes("\n".join(values).encode("utf-8")),
        }
        if values:
            raise RuntimeError(f"text-only reference load reported {name}: {values[:10]}")
    return summary


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately small, non-networked reference-generation CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate an atomic Qwen3.8-27B MPS F16 reference bundle with logits "
            "and 32-token cached/stateless parity metadata from a pinned local checkpoint."
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help=f"local {REPOSITORY} checkout at immutable revision {REVISION}",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "existing external parent directory; the fixed create-only bundle "
            f"{BUNDLE_DIRNAME!r} must not already exist"
        ),
    )
    return parser.parse_args(argv)


def synchronize(torch: Any) -> None:
    """Synchronize MPS before recording a reference wall-clock boundary."""

    torch.mps.synchronize()


def tensor_cosine(torch: Any, left: Any, right: Any) -> float:
    """Compute full-vector cosine similarity with float64 accumulation on the CPU."""

    if not bool(torch.isfinite(left).all().item()) or not bool(
        torch.isfinite(right).all().item()
    ):
        raise ValueError("cosine similarity inputs contain a non-finite value")
    left64 = left.to(dtype=torch.float64)
    right64 = right.to(dtype=torch.float64)
    denominator = torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64)
    denominator_value = float(denominator.item())
    if not math.isfinite(denominator_value):
        raise ValueError("cosine-similarity denominator is non-finite")
    if denominator_value == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    numerator_value = float(torch.dot(left64, right64).item())
    result = numerator_value / denominator_value
    if not math.isfinite(numerator_value) or not math.isfinite(result):
        raise ValueError("cosine similarity produced a non-finite value")
    return result


def device_details(torch: Any) -> dict[str, Any]:
    """Validate and describe the required Apple MPS reference accelerator."""

    if not torch.backends.mps.is_available():
        raise RuntimeError("the Qwen3.8 F16 reference requires an available MPS backend")
    return {
        "kind": "mps",
        "built": bool(torch.backends.mps.is_built()),
        "available": True,
        "machine": platform.machine(),
    }


def checked_duration(label: str, value: float) -> float:
    """Reject non-finite or negative wall-clock evidence before serialization."""

    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"{label} duration is invalid: {value!r}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pinned MPS reference graph and atomically publish one bundle."""

    arguments = parse_arguments(argv)
    model_dir, output_dir, bundle_path = ensure_external_paths(
        arguments.model_dir, arguments.output_dir
    )
    (
        package_versions,
        distribution_provenance,
        distribution_manifest_sha256,
    ) = verify_packages()
    artifacts, checkpoint_tensor_bytes, artifact_digest, eos_token_ids = verify_checkpoint(
        model_dir
    )

    # Heavy imports and model allocation occur only after all cheap preflight and
    # create-only checks pass.  Consequently `--help` and pure helper tests do not
    # initialize an accelerator or import the ML stack.
    import numpy
    import safetensors
    import tokenizers
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForCausalLM, modeling_utils
    from transformers.models.qwen3_5 import configuration_qwen3_5, modeling_qwen3_5

    del numpy, safetensors, tokenizers  # Versions were captured through package metadata.
    transformers_root = Path(transformers.__file__).resolve(strict=True).parent
    reference_sources = verify_reference_sources(transformers_root)
    imported_qwen_sources = {
        module.__name__: Path(module.__file__).resolve(strict=True)
        for module in (configuration_qwen3_5, modeling_qwen3_5)
    }
    expected_qwen_sources = {
        module: (transformers_root / expected["relative_path"]).resolve(strict=True)
        for module, expected in EXPECTED_REFERENCE_SOURCES.items()
        if ".models.qwen3_5." in module
    }
    if imported_qwen_sources != expected_qwen_sources:
        raise RuntimeError(
            "imported Qwen reference modules differ from pinned package sources: "
            f"expected {expected_qwen_sources}, found {imported_qwen_sources}"
        )
    kernel_status = {
        "fast_path_available": bool(modeling_qwen3_5.is_fast_path_available),
        "causal_conv1d_kernel_available": modeling_qwen3_5.causal_conv1d_fn is not None,
        "chunk_gated_delta_kernel_available": modeling_qwen3_5.chunk_gated_delta_rule
        is not None,
        "recurrent_gated_delta_kernel_available": modeling_qwen3_5.fused_recurrent_gated_delta_rule
        is not None,
    }
    if any(kernel_status.values()):
        raise RuntimeError(
            "the pinned official reference requires Transformers' auditable Torch fallback "
            f"kernels, found optional fast-path state {kernel_status}"
        )

    details = device_details(torch)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("Torch did not retain float32_matmul_precision=highest")
    torch.mps.set_per_process_memory_fraction(1.0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_grad_enabled(False)

    outer_config = Qwen3_5Config.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        revision=REVISION,
    )
    text_config = outer_config.text_config
    text_config._attn_implementation = "eager"

    load_started = time.perf_counter()
    (model, loading_info), allocator_warmup_override = (
        load_with_verified_mps_allocator_warmup_noop(
            modeling_utils,
            torch,
            Qwen3_5ForCausalLM,
            lambda: Qwen3_5ForCausalLM.from_pretrained(
                model_dir,
                config=text_config,
                dtype=torch.float16,
                device_map={"": "mps"},
                attn_implementation="eager",
                local_files_only=True,
                trust_remote_code=False,
                revision=REVISION,
                use_safetensors=True,
                weights_only=True,
                low_cpu_mem_usage=True,
                output_loading_info=True,
            ),
        )
    )
    synchronize(torch)
    load_seconds = checked_duration("model load", time.perf_counter() - load_started)
    if type(model) is not Qwen3_5ForCausalLM:
        raise RuntimeError(f"loaded unexpected model class {type(model)!r}")
    load_diagnostics = loading_info_summary(loading_info)
    model.eval()
    loaded_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if loaded_parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "loaded text-model parameter count changed: "
            f"expected {EXPECTED_PARAMETER_COUNT}, found {loaded_parameter_count}"
        )

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
    input_ids = torch.tensor([INPUT_IDS], dtype=torch.long, device="mps")
    prompt_attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="mps")

    with torch.inference_mode():
        stateless_started = time.perf_counter()
        stateless_output = model(
            input_ids=input_ids,
            attention_mask=prompt_attention_mask,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        synchronize(torch)
        stateless_seconds = checked_duration(
            "stateless prompt", time.perf_counter() - stateless_started
        )
        stateless_row = stateless_output.logits[0, -1].float().cpu().contiguous()
        if stateless_output.logits.dtype != torch.float16:
            raise RuntimeError(
                f"stateless logits use {stateless_output.logits.dtype}, expected float16"
            )
        if stateless_row.numel() != VOCAB_SIZE:
            raise RuntimeError(
                f"stateless vocabulary has {stateless_row.numel()} entries, expected {VOCAB_SIZE}"
            )
        if not bool(torch.isfinite(stateless_row).all().item()):
            raise RuntimeError("stateless reference logits contain a non-finite value")

        generation_started = time.perf_counter()
        cached_output = model(
            input_ids=input_ids,
            attention_mask=prompt_attention_mask,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        cache = cached_output.past_key_values
        if cache is None:
            raise RuntimeError("official Qwen cached prompt returned no cache object")
        cached_rows = []
        generated_ids: list[int] = []
        margins: list[float] = []
        decision_stability: list[dict[str, Any]] = []
        for step in range(GREEDY_TOKENS):
            if cached_output.logits.dtype != torch.float16:
                raise RuntimeError(
                    f"cached logits use {cached_output.logits.dtype} at step {step}, "
                    "expected float16"
                )
            row = cached_output.logits[0, -1].float().cpu().contiguous()
            if row.numel() != VOCAB_SIZE:
                raise RuntimeError(
                    f"cached vocabulary has {row.numel()} entries at step {step}, "
                    f"expected {VOCAB_SIZE}"
                )
            if not bool(torch.isfinite(row).all().item()):
                raise RuntimeError(f"cached reference logits became non-finite at step {step}")
            cached_rows.append(row)
            top_values, top_indices = torch.topk(row, k=2, largest=True, sorted=True)
            if not bool(torch.isfinite(top_values).all().item()):
                raise RuntimeError(f"cached top-two logits became non-finite at step {step}")
            token_id = int(top_indices[0].item())
            runner_up_id = int(top_indices[1].item())
            decision = greedy_decision_stability(
                step=step,
                top1_token_id=token_id,
                top2_token_id=runner_up_id,
                top1_logit=float(top_values[0].item()),
                top2_logit=float(top_values[1].item()),
                eos_token_ids=eos_token_ids,
            )
            margin = float(decision["margin"])
            decision_stability.append(decision)
            generated_ids.append(token_id)
            margins.append(margin)
            if step + 1 < GREEDY_TOKENS:
                decode_attention_mask = torch.ones(
                    (1, len(INPUT_IDS) + step + 1), dtype=torch.long, device="mps"
                )
                cached_output = model(
                    input_ids=torch.tensor(
                        [[token_id]], dtype=torch.long, device="mps"
                    ),
                    attention_mask=decode_attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                cache = cached_output.past_key_values
                if cache is None:
                    raise RuntimeError(f"cached decode returned no cache at step {step}")
        synchronize(torch)
        generation_seconds = checked_duration(
            "cached generation", time.perf_counter() - generation_started
        )
        minimum_observed_margin, minimum_margin_to_epsilon_ratio = (
            validate_decision_stability(decision_stability)
        )
        first_eos_position = next(
            (
                index
                for index, token_id in enumerate(generated_ids)
                if token_id in eos_token_ids
            ),
            None,
        )
        completed_32_decisions = (
            len(generated_ids) == GREEDY_TOKENS and first_eos_position is None
        )
        if not completed_32_decisions:
            raise RuntimeError("cached generation did not complete 32 non-EOS decisions")

        teacher_input_ids = INPUT_IDS + generated_ids[:-1]
        teacher_started = time.perf_counter()
        teacher_output = model(
            input_ids=torch.tensor(
                [teacher_input_ids], dtype=torch.long, device="mps"
            ),
            attention_mask=torch.ones(
                (1, len(teacher_input_ids)), dtype=torch.long, device="mps"
            ),
            use_cache=False,
            logits_to_keep=0,
            return_dict=True,
        )
        synchronize(torch)
        teacher_seconds = checked_duration(
            "teacher-forced forward", time.perf_counter() - teacher_started
        )
        teacher_logits = teacher_output.logits[0].float().cpu().contiguous()
        if teacher_output.logits.dtype != torch.float16:
            raise RuntimeError(
                f"teacher-forced logits use {teacher_output.logits.dtype}, expected float16"
            )
        if list(teacher_logits.shape) != [len(teacher_input_ids), VOCAB_SIZE]:
            raise RuntimeError(
                "teacher-forced logits have unexpected shape "
                f"{list(teacher_logits.shape)}, expected "
                f"[{len(teacher_input_ids)}, {VOCAB_SIZE}]"
            )
        if not bool(torch.isfinite(teacher_logits).all().item()):
            raise RuntimeError("teacher-forced reference logits contain a non-finite value")

    all_position_top1 = teacher_logits.argmax(dim=-1).tolist()
    prediction_positions = list(
        range(len(INPUT_IDS) - 1, len(INPUT_IDS) - 1 + GREEDY_TOKENS)
    )
    teacher_continuation = [all_position_top1[position] for position in prediction_positions]
    if teacher_continuation != generated_ids:
        raise RuntimeError(
            "official cached and stateless teacher-forced top-1 predictions differ: "
            f"cached={generated_ids}, stateless={teacher_continuation}"
        )

    per_prefix_cosine = []
    per_prefix_max_abs_error = []
    for cached_row, position in zip(cached_rows, prediction_positions, strict=True):
        teacher_row = teacher_logits[position]
        per_prefix_cosine.append(tensor_cosine(torch, cached_row, teacher_row))
        maximum_error = float((cached_row - teacher_row).abs().max().item())
        if not math.isfinite(maximum_error):
            raise RuntimeError("cached/stateless maximum absolute error is non-finite")
        per_prefix_max_abs_error.append(maximum_error)
    minimum_cosine = min(per_prefix_cosine)
    if not math.isfinite(minimum_cosine):
        raise RuntimeError("cached/stateless minimum cosine is non-finite")
    if minimum_cosine < SELF_PARITY_COSINE:
        raise RuntimeError(
            "official cached/stateless full-logit cosine parity fell below "
            f"{SELF_PARITY_COSINE}: {minimum_cosine}"
        )

    stateless_top1 = int(stateless_row.argmax().item())
    if stateless_top1 != generated_ids[0]:
        raise RuntimeError(
            f"stateless prompt top-1 {stateless_top1} differs from cached {generated_ids[0]}"
        )
    prompt_cached_cosine = tensor_cosine(torch, stateless_row, cached_rows[0])
    if prompt_cached_cosine < SELF_PARITY_COSINE:
        raise RuntimeError(
            f"official prompt cached/stateless cosine {prompt_cached_cosine} is below the gate"
        )

    stateless_values = [float(value) for value in stateless_row.tolist()]
    raw_payload = f32le_bytes(stateless_values)
    if len(raw_payload) != VOCAB_SIZE * 4:
        raise RuntimeError(f"raw vector has unexpected byte length {len(raw_payload)}")
    probes = [0, 1, 2, 100, 1_000, 10_000, 248_000, stateless_top1]
    probes = list(dict.fromkeys(probes))
    summary = vector_summary(stateless_values, probes)
    summary.update(
        {
            "relative_path": RAW_FILENAME,
            "encoding": "ieee754-f32-little-endian",
            "elements": VOCAB_SIZE,
            "bytes": len(raw_payload),
            "sha256": sha256_bytes(raw_payload),
            "top1_token_id": stateless_top1,
        }
    )

    script_path = Path(__file__).resolve(strict=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    parameter_dtypes: dict[str, int] = {}
    for parameter in model.parameters():
        name = str(parameter.dtype).removeprefix("torch.")
        parameter_dtypes[name] = parameter_dtypes.get(name, 0) + parameter.numel()
    if parameter_dtypes != {"float16": EXPECTED_PARAMETER_COUNT}:
        raise RuntimeError(
            "loaded parameter dtypes changed: expected only "
            f"{EXPECTED_PARAMETER_COUNT} float16 elements, found {parameter_dtypes}"
        )
    record = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "reference_class": REFERENCE_CLASS,
        "prompt": PROMPT,
        "input_ids": INPUT_IDS,
        "execution": {
            "dtype": "float16",
            "device": "mps",
            "attention_implementation": "eager",
            "deterministic_algorithms": True,
            "seed": 0,
            "use_cache": True,
            "float32_matmul_precision": "highest",
            "mps_per_process_memory_fraction": 1.0,
            "explicit_all_one_attention_masks": True,
            "allocator_warmup_override": allocator_warmup_override,
            "kernel_status": kernel_status,
        },
        "environment": {
            "generated_at_utc": now,
            "command": [sys.executable, str(script_path), *sys.argv[1:]],
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
            },
            "packages": package_versions,
            "distribution_provenance": distribution_provenance,
            "distribution_manifest_sha256": distribution_manifest_sha256,
            "device": details,
            "offline_environment": {
                name: os.environ[name]
                for name in (
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
            "generator": git_provenance(script_path),
            "reference_sources": reference_sources,
            "artifacts": artifacts,
            "artifact_manifest_sha256": artifact_digest,
            "checkpoint_tensor_bytes": checkpoint_tensor_bytes,
            "loaded_parameter_count": loaded_parameter_count,
            "parameter_dtype_elements": parameter_dtypes,
            "loading_info": load_diagnostics,
        },
        "last_logits": summary,
        "generation": {
            "max_new_tokens": GREEDY_TOKENS,
            "token_ids": generated_ids,
            "margins": margins,
            "stability_policy": {
                "name": TOP1_STABILITY_POLICY,
                "elementwise_perturbation_ulps": BINARY16_ELEMENTWISE_PERTURBATION_ULPS,
                "safety_factor": TOP1_STABILITY_SAFETY_FACTOR,
                "rationale": TOP1_STABILITY_RATIONALE,
            },
            "decision_stability": decision_stability,
            "minimum_observed_margin": minimum_observed_margin,
            "minimum_margin_to_epsilon_ratio": minimum_margin_to_epsilon_ratio,
            "decoded_text": tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "eos_token_ids": eos_token_ids,
            "first_eos_position": first_eos_position,
            "completed_32_decisions": completed_32_decisions,
        },
        "teacher_forced": {
            "input_ids": teacher_input_ids,
            "prediction_positions": prediction_positions,
            "all_position_top1_token_ids": all_position_top1,
            "continuation_top1_token_ids": teacher_continuation,
            "cached_top1_token_ids": generated_ids,
            "top1_equal": True,
            "per_prefix_cosine": per_prefix_cosine,
            "minimum_cosine": minimum_cosine,
            "per_prefix_max_abs_error": per_prefix_max_abs_error,
            "maximum_abs_error": max(per_prefix_max_abs_error),
            "prompt_cached_stateless_cosine": prompt_cached_cosine,
        },
        "timings_seconds": {
            "load": load_seconds,
            "stateless_prompt": stateless_seconds,
            "cached_generation_32": generation_seconds,
            "teacher_forced_all_logits": teacher_seconds,
        },
    }
    json_payload = strict_json_bytes(record)

    published_bundle = publish_reference_bundle(output_dir, raw_payload, json_payload)
    if published_bundle != bundle_path:
        raise RuntimeError(
            f"published unexpected bundle {published_bundle}, expected {bundle_path}"
        )
    raw_path = published_bundle / RAW_FILENAME
    json_path = published_bundle / JSON_FILENAME
    print(f"wrote {raw_path} ({len(raw_payload)} bytes, sha256={summary['sha256']})")
    print(f"wrote {json_path} ({len(json_payload)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
