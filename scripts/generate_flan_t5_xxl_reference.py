#!/usr/bin/env python3
"""Generate the pinned FLAN-T5-XXL F32 Metal acceptance reference.

This is an offline, maintainer-run oracle for ``google/flan-t5-xxl`` revision
``ae7c9136adc7555eeccc78cdd960dfd60fb346ce``.  It verifies every checkpoint
and runtime input before allocating the 11-billion-parameter graph, executes
only the official Transformers T5 implementation, and publishes an immutable
bundle with an atomic create-only rename.

The official checkpoint has a subtle embedding layout which is part of the
oracle contract.  ``encoder.embed_tokens.weight`` is listed by the shard index
but deduplicated physically under ``shared.weight``.  A second, byte-identical
``decoder.embed_tokens.weight`` is physically present, while ``lm_head.weight``
is distinct.  Transformers 5.2 converts the legacy configuration flag into an
always-tied runtime flag, then its loader preserves physical duplicates and
ties only the missing encoder key.  This program validates that exact topology
instead of assuming that the serialized flag and runtime graph are identical.

No Hub client, remote code, stochastic sampling, or implicit device fallback is
allowed.  Heavy ML imports occur only after artifact, output, package, and
source-code preflight.  ``--preflight-only`` performs all immutable artifact
hashing (including the 45 GB of shards) but never imports Torch or allocates a
model.  A long-form prompt must produce 32 stable non-EOS greedy decisions;
their complete cached and stateless logit matrices are serialized alongside a
separate four-beam translation fixture.  Failed publications leave a hidden
staging directory for forensic recovery, and no failure exposes a partial
final bundle.
"""

from __future__ import annotations

import argparse
import base64
import copy
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
import resource
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# Disable every common implicit network or fallback path before Transformers is
# imported.  Each local load below repeats the corresponding API-level guards.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


REPOSITORY = "google/flan-t5-xxl"
REVISION = "ae7c9136adc7555eeccc78cdd960dfd60fb346ce"
REFERENCE_CLASS = "transformers.T5ForConditionalGeneration"
TRANSLATION_PROMPT = "translate English to German: The house is wonderful."
TRANSLATION_INPUT_IDS = [13_959, 1_566, 12, 2_968, 10, 37, 629, 19, 1_627, 5, 1]
LONG_FORM_PROMPT = (
    "translate English to German: In the early morning, the scientist walked slowly "
    "through the quiet garden, carefully recording how sunlight touched every leaf, "
    "how birds moved between the branches, and how the cool air changed as the city "
    "began to wake around her."
)
LONG_FORM_INPUT_IDS = [
    13_959,
    1_566,
    12,
    2_968,
    10,
    86,
    8,
    778,
    1_379,
    6,
    8,
    17_901,
    3,
    6_412,
    5_665,
    190,
    8,
    4_354,
    2_004,
    6,
    4_321,
    5_592,
    149,
    16_285,
    15_880,
    334,
    8_384,
    6,
    149,
    6_331,
    2_301,
    344,
    8,
    9_678,
    6,
    11,
    149,
    8,
    1_633,
    799,
    2_130,
    38,
    8,
    690,
    1_553,
    12,
    7_178,
    300,
    160,
    5,
    1,
]
DECODER_START_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 2
VOCAB_SIZE = 32_128
TOKENIZER_VOCAB_SIZE = 32_100
HIDDEN_SIZE = 4_096
MAX_GREEDY_TOKENS = 32
BEAM_COUNT = 4
BEAM_MAX_NEW_TOKENS = 16
EXPECTED_PHYSICAL_TENSORS = 559
EXPECTED_LOGICAL_INDEX_KEYS = 560
EXPECTED_CHECKPOINT_TENSOR_BYTES = 45_067_714_560
EXPECTED_INDEX_TOTAL_BYTES = 45_594_099_712
EXPECTED_PARAMETER_COUNT = 11_266_928_640
EXPECTED_UNIQUE_PARAMETERS = 559
EXPECTED_LOGICAL_STATE_KEYS = 560
EXPECTED_HEADER_INVENTORY_SHA256 = (
    "794a9c5bb717021bf491ee6942cda6dab2b223a7bb5f7c2861be01248f590a0d"
)
EXPECTED_ARTIFACT_MANIFEST_SHA256 = (
    "bbe4dfea65c912159a98ed8aa0584960e6c89e5fcb607f32eb437fa46e0f2d6e"
)
EXPECTED_DISTRIBUTION_MANIFEST_SHA256 = (
    "b814bb3254d51a28c6d74e80c04c3532c9836a0aadc4fab0eed410d6612d59dd"
)
EXPECTED_REFERENCE_SOURCE_MANIFEST_SHA256 = (
    "06554bbcfc8fc87ce5d4f44cb6ef7da412e6870fe67d75b247d63420d7a58408"
)
MAX_EXACT_JSON_INTEGER = (1 << 53) - 1
MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024
SELF_PARITY_COSINE = 0.999_999
COMPATIBILITY_ATOL = 1.0e-4
COMPATIBILITY_RTOL = 1.0e-4
MIN_OUTPUT_FREE_BYTES = 16 * 1024 * 1024
MIN_MPS_RECOMMENDED_BYTES = EXPECTED_CHECKPOINT_TENSOR_BYTES + 12 * 1024**3

ENCODER_RAW_FILENAME = "flan-t5-xxl-ae7c9136-encoder-final.f32le"
DECODER_RAW_FILENAME = "flan-t5-xxl-ae7c9136-decoder-first-final.f32le"
LOGITS_RAW_FILENAME = "flan-t5-xxl-ae7c9136-first-logits.f32le"
CACHED_LOGITS_RAW_FILENAME = "flan-t5-xxl-ae7c9136-cached-logits-32x32128.f32le"
TEACHER_LOGITS_RAW_FILENAME = "flan-t5-xxl-ae7c9136-teacher-logits-32x32128.f32le"
JSON_FILENAME = "flan-t5-xxl-ae7c9136-reference.json"
BUNDLE_DIRNAME = "flan-t5-xxl-ae7c9136-reference.bundle"
MPS_MEMORY_SAMPLE_SECONDS = 0.02

ALLOCATOR_WARMUP_TARGET = "transformers.modeling_utils.caching_allocator_warmup"
ALLOCATOR_WARMUP_POLICY = "mps-exclusive-noop-during-from-pretrained"
ALLOCATOR_WARMUP_SCOPE = "T5ForConditionalGeneration.from_pretrained"
ALLOCATOR_WARMUP_REASON = (
    "Transformers 5.2 creates one flat F16 warmup tensor with more than INT_MAX "
    "elements for this F32 model. Metal rejects that synthetic one-dimensional "
    "shape; skipping the performance-only allocation leaves checkpoint tensors "
    "and model mathematics unchanged."
)
EXPECTED_ALLOCATOR_WARMUP_CALLS = 1

EXPECTED_PACKAGE_VERSIONS = {
    "accelerate": "1.12.0",
    "numpy": "2.4.3",
    "safetensors": "0.7.0",
    "tokenizers": "0.22.2",
    "torch": "2.10.0",
    "transformers": "5.2.0",
}

EXPECTED_DISTRIBUTION_FILES = {
    "accelerate": {
        "METADATA": (
            "accelerate-1.12.0.dist-info/METADATA",
            19_601,
            "31d880e2aa1eaffb50dfecb62bc81d53f0262ed19ff374e7ff7a76b41b87c6b8",
        ),
        "RECORD": (
            "accelerate-1.12.0.dist-info/RECORD",
            14_253,
            "ebc6c296fe7d6bf449f6b776a4839cba2c8e8d8b649346ccfd2d505feea18f62",
        ),
    },
    "numpy": {
        "METADATA": (
            "numpy-2.4.3.dist-info/METADATA",
            6_608,
            "621bf875d57eee8673f39344ea98ddc370e944e9c8c36b38b430e290f2b682cf",
        ),
        "RECORD": (
            "numpy-2.4.3.dist-info/RECORD",
            126_949,
            "b4b4d441229ce2edbbd2417b0b993d7b72d0ac801770fe2d32b26d8e1ab29829",
        ),
    },
    "safetensors": {
        "METADATA": (
            "safetensors-0.7.0.dist-info/METADATA",
            4_125,
            "6809326fbf17444dd520ab19f0a55b7a18476dbe52a430466bbf5389967c2aaa",
        ),
        "RECORD": (
            "safetensors-0.7.0.dist-info/RECORD",
            1_573,
            "f4f493f241d26041cf94fe7214f88c8f255df145eb064abac8550bc533d38f36",
        ),
    },
    "tokenizers": {
        "METADATA": (
            "tokenizers-0.22.2.dist-info/METADATA",
            7_254,
            "15a5ddaf489f592b77e0a934c0eeb46b51130a94064ca129232afdb0a3868efc",
        ),
        "RECORD": (
            "tokenizers-0.22.2.dist-info/RECORD",
            3_778,
            "cd5bb15216c5d083bbe39ccc172f58e4114a8fe5b786f4f8f8239e53053ad9c3",
        ),
    },
    "torch": {
        "METADATA": (
            "torch-2.10.0.dist-info/METADATA",
            31_092,
            "4b0f1217de69037355b4e582905cdb303a56a488ac6fd172f148d88fed634f54",
        ),
        "RECORD": (
            "torch-2.10.0.dist-info/RECORD",
            1_477_169,
            "22546c990d209800cd92db2b052a8ea478cee3a6e9253485f282221e0a6225ae",
        ),
    },
    "transformers": {
        "METADATA": (
            "transformers-5.2.0.dist-info/METADATA",
            32_319,
            "4340d0f550bf77d6595e4efafe066cd88306307577701df8e558ad620551c65f",
        ),
        "RECORD": (
            "transformers-5.2.0.dist-info/RECORD",
            394_090,
            "98d165cb699677711c15cc42eeeb25f7889fe164ecd90135f34e4b88109f931e",
        ),
    },
}

EXPECTED_DISTRIBUTION_PAYLOADS = {
    "accelerate": (
        95,
        1_483_290,
        84,
        "1a5c1e846d1805592fe0090db994c3afc37f552ca6f4aa8b1d56f4f08bbfbfae",
    ),
    "numpy": (
        1_040,
        20_254_356,
        487,
        "bc49cc4a1cecd0b0e258d25f684b2250330c9672fcb64023b70eaf1f5d463f32",
    ),
    "safetensors": (
        14,
        1_070_455,
        7,
        "d0d712a1786e52b270faa8b558587b1cf41fd8255499c5eb25b967c2c9911afa",
    ),
    "tokenizers": (
        29,
        8_506_298,
        16,
        "b87e4f97df12e3603e619104d1b824792f6704cd405b340208980325844ff257",
    ),
    "torch": (
        12_340,
        338_442_177,
        2_199,
        "e7fc648575316519c0c996f4dea905684caff4aa6edc979e3173a4ee50209570",
    ),
    "transformers": (
        2_128,
        43_591_674,
        2_120,
        "e074d191c22d4b2dc0cb53d1a3afbd2f2cf81a8a2a8e8d8b685a8233c5ce09bc",
    ),
}

EXPECTED_REFERENCE_SOURCES = {
    "transformers.cache_utils": (
        "cache_utils.py",
        60_147,
        "e4fe20ca1eeb17ea0ce7e62d1dbd0d6eab32e7072044f293612403b8e0c477fd",
    ),
    "transformers.configuration_utils": (
        "configuration_utils.py",
        57_894,
        "52100272e3bf59551c072a064053e90bc30f1fd0aa9e48fac7ad4b675c63ebcc",
    ),
    "transformers.generation.configuration_utils": (
        "generation/configuration_utils.py",
        78_958,
        "0146d5211f8be1594e5c3a5dff918173c0ce1cd6a2edfee47befbcfa34f8cc04",
    ),
    "transformers.generation.logits_process": (
        "generation/logits_process.py",
        149_974,
        "f87a1496431d779b7212c29ae5d116051cb76fff6d26727c98a6e6bdba393d87",
    ),
    "transformers.generation.stopping_criteria": (
        "generation/stopping_criteria.py",
        28_818,
        "51b2071268b4046395a183901fe8e366bca419f140fa7a88d31500c5ca32e783",
    ),
    "transformers.generation.utils": (
        "generation/utils.py",
        214_388,
        "fc980c5ce0e6e45ffcaf2a37135e01afe657df91b99d684af145130608419652",
    ),
    "transformers.masking_utils": (
        "masking_utils.py",
        74_373,
        "c3c82f7b7b6e03d3f04ba6c6c58a3dd6910623636452ec67ef70e3eb522f9fe7",
    ),
    "transformers.modeling_attn_mask_utils": (
        "modeling_attn_mask_utils.py",
        21_930,
        "fb3ec46639a19fda1eb99339f4e7770332a0f4f75b9de4abb34a0b1ee379838b",
    ),
    "transformers.modeling_utils": (
        "modeling_utils.py",
        247_340,
        "2f45c2139ce633e6e3eae9290675dd5815643b37c27f2cc4dd83586fb648f7bb",
    ),
    "transformers.models.t5.configuration_t5": (
        "models/t5/configuration_t5.py",
        6_843,
        "a16d4bf808b278d279455f800e20606d19a7b1d1d513b8061fe2b4c24c54862d",
    ),
    "transformers.models.t5.modeling_t5": (
        "models/t5/modeling_t5.py",
        73_290,
        "8b3ee9af023292d10ca63071799bd5174a613a2eba13135c3f80efbd9b983e54",
    ),
    "transformers.models.t5.tokenization_t5": (
        "models/t5/tokenization_t5.py",
        6_388,
        "f2a3906801dedf3ead89e7ca122ac5cac6c819267b0374256771557898b98287",
    ),
    "transformers.tokenization_utils_tokenizers": (
        "tokenization_utils_tokenizers.py",
        59_899,
        "3d908a608063d522052be778a7617a447c491f180f36348954b6d85709fac342",
    ),
}

EXPECTED_CONTROL_FILES = {
    "config.json": (674, "dd923bd687781f336e7845b7e011b7bbeca5fdde5c27f8b06071540340707dbd"),
    "generation_config.json": (
        147,
        "f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a",
    ),
    "model.safetensors.index.json": (
        53_032,
        "bab7fd6107b34412f8575908370f29f44f433617a93d557e57b299d6ee688efe",
    ),
    "tokenizer.json": (
        2_424_064,
        "fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a",
    ),
    "tokenizer_config.json": (
        2_539,
        "ef30e0628a2d067c5d9d02d98cbf9a2040f4a51883e5d16e1809eea548db0eff",
    ),
    "special_tokens_map.json": (
        2_201,
        "5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f",
    ),
    "spiece.model": (791_656, "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"),
}

EXPECTED_SHARDS = {
    "model-00001-of-00005.safetensors": (
        9_452_262_278,
        "2399615e60e983d1e647248b15bfad3c0ccd9473a0e387dbe03284fcaacf21c2",
    ),
    "model-00002-of-00005.safetensors": (
        9_597_007_440,
        "4d2459d60251e60af6314ae4785fb0bea32febb1989434c816de84c0fdae2d88",
    ),
    "model-00003-of-00005.safetensors": (
        9_955_647_022,
        "60a2c0b6f284a55df23ee68ce3065631368d7a6a35e5443b5cd98151f145fd67",
    ),
    "model-00004-of-00005.safetensors": (
        9_999_712_776,
        "5f0cbe2b9a51ab64c93d16993e79bf81cc39f8692f89122cc68fa3b824c2b3cf",
    ),
    "model-00005-of-00005.safetensors": (
        6_063_154_240,
        "b0001f5f48ce5bf0e7e997c57ce5170ce0c0b0d667ee2a72d01d9aa26f40b7b5",
    ),
}

# These files are present in the complete upstream repository but are never
# eligible inputs to this SafeTensors-only reference run.  Their exact names
# and lengths make the root inventory closed without paying another ~120 GB of
# I/O to hash redundant Flax, PyTorch, and TensorFlow payloads.  A selectively
# downloaded checkout may omit any of them; no other root entry is accepted.
EXPECTED_IGNORED_ROOT_FILES = {
    ".gitattributes": 1_434,
    "README.md": 10_306,
    "flax_model-00001-of-00005.msgpack": 9_989_141_334,
    "flax_model-00002-of-00005.msgpack": 9_932_567_678,
    "flax_model-00003-of-00005.msgpack": 9_999_701_724,
    "flax_model-00004-of-00005.msgpack": 9_999_701_629,
    "flax_model-00005-of-00005.msgpack": 4_620_241_312,
    "flax_model.msgpack.index.json": 51_206,
    "pytorch_model-00001-of-00005.bin": 9_452_284_099,
    "pytorch_model-00002-of-00005.bin": 9_597_030_149,
    "pytorch_model-00003-of-00005.bin": 9_955_673_595,
    "pytorch_model-00004-of-00005.bin": 9_999_740_781,
    "pytorch_model-00005-of-00005.bin": 6_063_168_980,
    "pytorch_model.bin.index.json": 50_781,
    "tf_model-00001-of-00005.h5": 9_989_362_864,
    "tf_model-00002-of-00005.h5": 9_932_810_536,
    "tf_model-00003-of-00005.h5": 9_999_968_760,
    "tf_model-00004-of-00005.h5": 9_999_964_640,
    "tf_model-00005-of-00005.h5": 4_620_350_264,
    "tf_model.h5.index.json": 69_580,
}
ALLOWED_IGNORED_ROOT_DIRECTORIES = {".cache"}

EXPECTED_EMBEDDING_CONTENT_HASHES = {
    "shared.weight": "44658aa6f4a8de8808eaaa406ae725410b1108187172d3f28ae3179d45f36377",
    "decoder.embed_tokens.weight": (
        "44658aa6f4a8de8808eaaa406ae725410b1108187172d3f28ae3179d45f36377"
    ),
    "lm_head.weight": "95aee6980c4516df69eaa1f7ef8fad5b985571ee3d1d75c43c1620804c57ccff",
}


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for an in-memory payload."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash one regular file incrementally without shard-sized allocation."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_range(path: Path, offset: int, length: int) -> str:
    """Hash exactly one validated byte range from a regular checkpoint file."""

    if offset < 0 or length < 0:
        raise ValueError("file-range offset and length must be nonnegative")
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"short read while hashing {path} at byte {handle.tell()}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def json_without_duplicate_keys(path: Path) -> Any:
    """Parse one UTF-8 JSON file while rejecting duplicate object keys."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        """Build one object only when every serialized key is unique."""

        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicates)


def json_bytes_without_duplicate_keys(payload: bytes, context: str) -> Any:
    """Parse UTF-8 JSON bytes with duplicate-key rejection and context."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        """Reject ambiguity while constructing a header object."""

        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {context}")
            result[key] = value
        return result

    return json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)


def checked_nonnegative_integer(value: Any, field: str) -> int:
    """Canonicalize an exact JSON integer in the interoperable F64 range."""

    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError(f"{field} must be a JSON integer, found {value!r}")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"{field} must be finite and integral, found {value!r}")
    if value < 0 or value > MAX_EXACT_JSON_INTEGER:
        raise ValueError(f"{field} lies outside 0..={MAX_EXACT_JSON_INTEGER}: {value!r}")
    canonical = int(value)
    if isinstance(value, float) and float(canonical) != value:
        raise ValueError(f"{field} cannot be losslessly canonicalized: {value!r}")
    return canonical


def checked_product(values: Iterable[int], field: str) -> int:
    """Multiply nonnegative dimensions while enforcing the JSON integer bound."""

    product = 1
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} contains invalid dimension {value!r}")
        product *= value
        if product > MAX_EXACT_JSON_INTEGER:
            raise ValueError(f"{field} product exceeds {MAX_EXACT_JSON_INTEGER}")
    return product


def strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize sorted, indented JSON while rejecting non-finite numbers."""

    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def top_k_records(values: Sequence[float], count: int) -> list[dict[str, float | int]]:
    """Return stable descending top-k values with token-ID tie breaking."""

    if count <= 0 or count > len(values):
        raise ValueError(f"top-k count {count} is outside 1..={len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("cannot rank a vector containing non-finite values")
    winners = heapq.nsmallest(count, enumerate(values), key=lambda pair: (-pair[1], pair[0]))
    return [{"token_id": index, "logit": float(value)} for index, value in winners]


def vector_summary(values: Sequence[float], probes: Sequence[int]) -> dict[str, Any]:
    """Compute stable integrity diagnostics for one flattened F32 tensor."""

    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("reference tensor must be nonempty and finite")
    if len(set(probes)) != len(probes) or any(
        index < 0 or index >= len(values) for index in probes
    ):
        raise ValueError("probe indices must be unique and in bounds")
    total = math.fsum(float(value) for value in values)
    norm = math.sqrt(math.fsum(float(value) * float(value) for value in values))
    if not math.isfinite(total) or not math.isfinite(norm):
        raise ValueError("reference tensor summary is non-finite")
    return {
        "finite": True,
        "elements": len(values),
        "sum": total,
        "l2_norm": norm,
        "probes": [{"flat_index": index, "value": float(values[index])} for index in probes],
    }


def logits_summary(values: Sequence[float], probes: Sequence[int]) -> dict[str, Any]:
    """Extend a tensor summary with deterministic vocabulary rankings."""

    summary = vector_summary(values, probes)
    summary["top10"] = top_k_records(values, 10)
    summary["top1_token_id"] = int(summary["top10"][0]["token_id"])
    return summary


def write_create_only(path: Path, payload: bytes) -> None:
    """Create and fsync one output without following or replacing a path."""

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
    """Fsync one directory so prior creates and renames are durable."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def path_lexists(path: Path) -> bool:
    """Return whether any entry, including a dangling symlink, occupies a path."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def rename_directory_create_only(source: Path, destination: Path) -> None:
    """Atomically publish a sibling directory without replacing a destination."""

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
        raise RuntimeError(f"atomic no-replace publication is unsupported on {sys.platform!r}")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def publish_reference_bundle(output_dir: Path, payloads: Mapping[str, bytes]) -> Path:
    """Durably publish a complete, fixed-name reference bundle atomically."""

    required = {
        ENCODER_RAW_FILENAME,
        DECODER_RAW_FILENAME,
        LOGITS_RAW_FILENAME,
        CACHED_LOGITS_RAW_FILENAME,
        TEACHER_LOGITS_RAW_FILENAME,
        JSON_FILENAME,
    }
    if set(payloads) != required:
        raise ValueError(
            "reference payload names differ: "
            f"expected {sorted(required)}, found {sorted(payloads)}"
        )
    if any(Path(name).name != name for name in payloads):
        raise ValueError("reference payload names must be local basenames")
    bundle_path = output_dir / BUNDLE_DIRNAME
    if path_lexists(bundle_path):
        raise FileExistsError(f"create-only reference bundle already exists: {bundle_path}")
    staging = Path(tempfile.mkdtemp(prefix=f".{BUNDLE_DIRNAME}.staging-", dir=output_dir))
    try:
        for name in sorted(payloads):
            write_create_only(staging / name, payloads[name])
        os.chmod(staging, 0o755)
        fsync_directory(staging)
        rename_directory_create_only(staging, bundle_path)
        fsync_directory(output_dir)
    except BaseException as error:
        error.add_note(f"recoverable reference staging directory: {staging}")
        raise
    return bundle_path


def ensure_external_paths(model_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve safe input/output roots and require an absent final bundle."""

    model_dir = model_dir.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve(strict=True)
    crate_dir = Path(__file__).resolve().parents[1]
    if not model_dir.is_dir() or not output_dir.is_dir():
        raise ValueError("model and output paths must be existing directories")
    if output_dir == crate_dir or output_dir.is_relative_to(crate_dir):
        raise ValueError(f"output directory must be external to the crate: {output_dir}")
    if output_dir == model_dir or output_dir.is_relative_to(model_dir):
        raise ValueError(f"output directory must be outside the checkpoint: {output_dir}")
    if shutil.disk_usage(output_dir).free < MIN_OUTPUT_FREE_BYTES:
        raise OSError(f"output directory has less than {MIN_OUTPUT_FREE_BYTES} free bytes")
    bundle = output_dir / BUNDLE_DIRNAME
    if path_lexists(bundle):
        raise FileExistsError(f"create-only reference bundle already exists: {bundle}")
    return model_dir, output_dir, bundle


def regular_file_identity(root: Path, relative_path: str) -> tuple[Path, int]:
    """Resolve one pinned local basename to a nonsymlink regular file."""

    relative = Path(relative_path)
    if relative.is_absolute() or relative.name != relative_path or ".." in relative.parts:
        raise ValueError(f"artifact path is not a local basename: {relative_path!r}")
    unresolved = root / relative
    metadata = unresolved.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact must be a nonsymlink regular file: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact escapes checkpoint root: {unresolved}")
    return resolved, metadata.st_size


def verify_root_inventory(model_dir: Path) -> list[dict[str, Any]]:
    """Close the checkpoint root over consumed and explicitly inert entries.

    Transformers resolves several optional filenames before choosing a weight
    or tokenizer source.  Merely authenticating the intended shards is not
    sufficient: an injected ``model.safetensors``, adapter configuration, or
    added-token overlay could take precedence.  This closed inventory rejects
    every unknown root basename while allowing the pinned repository's
    redundant framework formats and Hugging Face's inert hidden cache.
    """

    consumed = EXPECTED_CONTROL_FILES | EXPECTED_SHARDS
    observed: list[dict[str, Any]] = []
    names: set[str] = set()
    for path in sorted(model_dir.iterdir(), key=lambda item: item.name):
        name = path.name
        if name in names:
            raise ValueError(f"duplicate checkpoint root entry: {name!r}")
        names.add(name)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"checkpoint root entry must not be a symlink: {path}")
        if name in consumed:
            expected_bytes = consumed[name][0]
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
                raise ValueError(
                    f"consumed root artifact changed for {name!r}: "
                    f"kind={stat.S_IFMT(metadata.st_mode)}, bytes={metadata.st_size}"
                )
            observed.append(
                {"relative_path": name, "kind": "consumed", "bytes": metadata.st_size}
            )
        elif name in EXPECTED_IGNORED_ROOT_FILES:
            expected_bytes = EXPECTED_IGNORED_ROOT_FILES[name]
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
                raise ValueError(
                    f"inert root artifact changed for {name!r}: "
                    f"kind={stat.S_IFMT(metadata.st_mode)}, bytes={metadata.st_size}"
                )
            observed.append(
                {
                    "relative_path": name,
                    "kind": "ignored-redundant-upstream-format",
                    "bytes": metadata.st_size,
                }
            )
        elif name in ALLOWED_IGNORED_ROOT_DIRECTORIES:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"inert root directory changed type: {path}")
            observed.append(
                {
                    "relative_path": name,
                    "kind": "ignored-hidden-download-cache",
                    "bytes": None,
                }
            )
        else:
            raise ValueError(
                f"unexpected checkpoint root entry could influence local loading: {path}"
            )
    missing = set(consumed) - names
    if missing:
        raise FileNotFoundError(f"required checkpoint root entries are missing: {sorted(missing)}")
    return observed


def canonical_artifact_digest(artifacts: Sequence[Mapping[str, Any]]) -> str:
    """Bind sorted artifact paths, byte lengths, and content hashes."""

    lines = [
        f"{item['relative_path']}\0{item['bytes']}\0{item['sha256']}\n"
        for item in sorted(artifacts, key=lambda item: str(item["relative_path"]))
    ]
    return sha256_bytes("".join(lines).encode("utf-8"))


def canonical_source_digest(sources: Sequence[Mapping[str, Any]]) -> str:
    """Bind sorted module names and exact source-file identities."""

    lines = [
        f"{item['module']}\0{item['relative_path']}\0{item['bytes']}\0{item['sha256']}\n"
        for item in sorted(sources, key=lambda item: str(item["module"]))
    ]
    return sha256_bytes("".join(lines).encode("utf-8"))


def verified_distribution_file(distribution: Any, name: str, logical: str) -> dict[str, Any]:
    """Verify one installed wheel's pinned METADATA or RECORD identity."""

    candidates = [
        entry
        for entry in distribution.files or ()
        if entry.name == logical and entry.parent.name.endswith(".dist-info")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"distribution {name!r} has {len(candidates)} {logical} candidates")
    expected_path, expected_bytes, expected_hash = EXPECTED_DISTRIBUTION_FILES[name][logical]
    if str(candidates[0]) != expected_path:
        raise RuntimeError(f"distribution {name!r} {logical} path changed")
    unresolved = Path(distribution.locate_file(candidates[0]))
    if unresolved.is_symlink():
        raise RuntimeError(f"distribution {name!r} {logical} is a symlink")
    path = unresolved.resolve(strict=True)
    observed = {
        "relative_path": expected_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    expected = {"relative_path": expected_path, "bytes": expected_bytes, "sha256": expected_hash}
    if observed != expected:
        raise RuntimeError(f"distribution {name!r} {logical} changed: {observed}")
    return observed


def verified_distribution_payload(distribution: Any, name: str) -> dict[str, Any]:
    """Authenticate every executable/data payload covered by a wheel RECORD.

    A hash of ``RECORD`` only authenticates its manifest.  This second pass
    streams every hashed entry and compares both its byte length and URL-safe
    SHA-256 spelling with that authenticated manifest.  Adjacent bytecode is
    deliberately excluded because full runs route imports through a fresh,
    empty bytecode-cache prefix and disable bytecode writes.
    """

    entries = distribution.files
    if not entries:
        raise RuntimeError(f"distribution {name!r} exposes no RECORD entries")
    canonical = bytearray()
    verified_files = 0
    verified_bytes = 0
    ignored_bytecode = 0
    ignored_record = 0
    for entry in sorted(entries, key=str):
        relative_path = str(entry)
        if relative_path.endswith(".pyc") and "/__pycache__/" in f"/{relative_path}":
            ignored_bytecode += 1
            continue
        record_hash = entry.hash
        if record_hash is None:
            if entry.name == "RECORD" and entry.parent.name.endswith(".dist-info"):
                ignored_record += 1
                continue
            raise RuntimeError(
                f"distribution {name!r} has an unauthenticated RECORD entry: "
                f"{relative_path!r}"
            )
        if record_hash.mode != "sha256" or entry.size is None or entry.size < 0:
            raise RuntimeError(
                f"distribution {name!r} has an unsupported RECORD identity: "
                f"{relative_path!r}"
            )
        unresolved = Path(distribution.locate_file(entry))
        metadata = unresolved.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"distribution {name!r} payload is not a nonsymlink regular file: "
                f"{relative_path!r}"
            )
        expected_bytes = int(entry.size)
        if metadata.st_size != expected_bytes:
            raise RuntimeError(
                f"distribution {name!r} payload length changed for {relative_path!r}: "
                f"expected {expected_bytes}, found {metadata.st_size}"
            )
        observed_hash = sha256_file(unresolved)
        encoded_hash = base64.urlsafe_b64encode(bytes.fromhex(observed_hash)).decode("ascii")
        encoded_hash = encoded_hash.rstrip("=")
        if encoded_hash != record_hash.value:
            raise RuntimeError(
                f"distribution {name!r} payload hash changed for {relative_path!r}: "
                f"{observed_hash}"
            )
        canonical.extend(relative_path.encode("utf-8"))
        canonical.append(0)
        canonical.extend(str(expected_bytes).encode("ascii"))
        canonical.append(0)
        canonical.extend(observed_hash.encode("ascii"))
        canonical.append(ord("\n"))
        verified_files += 1
        verified_bytes += expected_bytes
    if verified_files == 0 or ignored_record != 1:
        raise RuntimeError(
            f"distribution {name!r} RECORD coverage changed: "
            f"verified={verified_files}, record_entries={ignored_record}"
        )
    return {
        "verified_files": verified_files,
        "verified_bytes": verified_bytes,
        "ignored_bytecode_files": ignored_bytecode,
        "manifest_sha256": sha256_bytes(bytes(canonical)),
    }


def distribution_import_root(distribution: Any, name: str) -> Path:
    """Resolve the authenticated top-level package directory for one wheel."""

    unresolved = Path(distribution.locate_file(name))
    metadata = unresolved.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"distribution {name!r} import root is not a real directory")
    return unresolved.resolve(strict=True)


def canonical_distribution_digest(distributions: Mapping[str, Mapping[str, Any]]) -> str:
    """Bind package versions and their pinned dist-info identities."""

    lines: list[str] = []
    for name in sorted(distributions):
        item = distributions[name]
        lines.append(f"{name}\0{item['version']}\n")
        for key in ("metadata", "record"):
            file = item[key]
            lines.append(f"{key}\0{file['relative_path']}\0{file['bytes']}\0{file['sha256']}\n")
    return sha256_bytes("".join(lines).encode("utf-8"))


def verify_packages() -> tuple[
    dict[str, str], dict[str, dict[str, Any]], str, dict[str, Path]
]:
    """Authenticate exact wheel versions, RECORD payloads, and import roots."""

    versions: dict[str, str] = {}
    provenance: dict[str, dict[str, Any]] = {}
    import_roots: dict[str, Path] = {}
    for name, expected_version in EXPECTED_PACKAGE_VERSIONS.items():
        distribution = importlib.metadata.distribution(name)
        if distribution.version != expected_version:
            raise RuntimeError(
                f"package {name!r}: expected {expected_version}, "
                f"found {distribution.version}"
            )
        versions[name] = distribution.version
        metadata_identity = verified_distribution_file(distribution, name, "METADATA")
        record_identity = verified_distribution_file(distribution, name, "RECORD")
        payload = verified_distribution_payload(distribution, name)
        expected_payload = EXPECTED_DISTRIBUTION_PAYLOADS[name]
        observed_payload = (
            payload["verified_files"],
            payload["verified_bytes"],
            payload["ignored_bytecode_files"],
            payload["manifest_sha256"],
        )
        if observed_payload != expected_payload:
            raise RuntimeError(
                f"distribution {name!r} authenticated payload manifest changed: "
                f"{observed_payload}"
            )
        provenance[name] = {
            "version": distribution.version,
            "metadata": metadata_identity,
            "record": record_identity,
            "payload": payload,
        }
        import_roots[name] = distribution_import_root(distribution, name)
    digest = canonical_distribution_digest(provenance)
    if digest != EXPECTED_DISTRIBUTION_MANIFEST_SHA256:
        raise RuntimeError(f"distribution manifest changed: {digest}")
    return versions, provenance, digest, import_roots


def verify_reference_sources(transformers_root: Path) -> tuple[list[dict[str, Any]], str]:
    """Verify exact T5, mask, cache, tokenizer, and beam-search source files."""

    transformers_root = transformers_root.resolve(strict=True)
    sources: list[dict[str, Any]] = []
    for module, (relative_name, expected_bytes, expected_hash) in sorted(
        EXPECTED_REFERENCE_SOURCES.items()
    ):
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"invalid pinned source path: {relative}")
        unresolved = transformers_root / relative
        if unresolved.is_symlink():
            raise RuntimeError(f"pinned source is a symlink: {relative}")
        path = unresolved.resolve(strict=True)
        if not path.is_relative_to(transformers_root) or not path.is_file():
            raise RuntimeError(f"pinned source escapes package: {relative}")
        observed = {
            "module": module,
            "path": str(path),
            "relative_path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if observed["bytes"] != expected_bytes or observed["sha256"] != expected_hash:
            raise RuntimeError(f"reference source {module!r} changed: {observed}")
        sources.append(observed)
    digest = canonical_source_digest(sources)
    if digest != EXPECTED_REFERENCE_SOURCE_MANIFEST_SHA256:
        raise RuntimeError(f"reference source manifest changed: {digest}")
    return sources, digest


def prepare_authenticated_imports(
    output_dir: Path, import_roots: Mapping[str, Path]
) -> tuple[Path, dict[str, str], tuple[bool, str | None]]:
    """Bind future imports to authenticated roots and exclude adjacent bytecode.

    Python can execute a timestamp-valid ``__pycache__`` file instead of its
    authenticated source.  Full reference runs therefore require that none of
    the heavy packages is already imported, resolve every top-level package
    before execution, and direct bytecode lookups to a fresh private directory
    while disabling writes.
    """

    already_imported = sorted(
        name
        for name in EXPECTED_PACKAGE_VERSIONS
        if name in sys.modules
        or any(module.startswith(f"{name}.") for module in sys.modules)
    )
    if already_imported:
        raise RuntimeError(
            "heavy reference packages were imported before provenance preflight: "
            f"{already_imported}"
        )
    resolutions: dict[str, str] = {}
    for name, root in sorted(import_roots.items()):
        specification = importlib.util.find_spec(name)
        if specification is None or specification.origin is None:
            raise RuntimeError(f"cannot resolve authenticated package {name!r}")
        origin = Path(specification.origin).resolve(strict=True)
        if not origin.is_relative_to(root):
            raise RuntimeError(
                f"package {name!r} resolves outside its authenticated root: {origin}"
            )
        resolutions[name] = str(origin)
    cache_root = Path(
        tempfile.mkdtemp(prefix=".flan-t5-xxl-authenticated-pycache-", dir=output_dir)
    ).resolve(strict=True)
    previous = (sys.dont_write_bytecode, sys.pycache_prefix)
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(cache_root)
    return cache_root, resolutions, previous


def finalize_authenticated_imports(
    cache_root: Path, previous: tuple[bool, str | None]
) -> dict[str, Any]:
    """Prove no bytecode was consumed/written, remove the prefix, and restore Python."""

    try:
        entries = sorted(path.name for path in cache_root.iterdir())
        if entries:
            raise RuntimeError(f"authenticated bytecode cache unexpectedly changed: {entries}")
        cache_root.rmdir()
    finally:
        sys.dont_write_bytecode, sys.pycache_prefix = previous
    return {
        "adjacent_bytecode_ignored": True,
        "writes_disabled": True,
        "private_cache_remained_empty": True,
        "restored": True,
    }


def bind_imported_modules(
    modules: Mapping[str, Any], import_roots: Mapping[str, Path]
) -> dict[str, str]:
    """Prove each executed top-level module came from its authenticated wheel."""

    if set(modules) != set(import_roots):
        raise RuntimeError(
            f"import binding set changed: modules={sorted(modules)}, "
            f"roots={sorted(import_roots)}"
        )
    bindings: dict[str, str] = {}
    for name, module in sorted(modules.items()):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"imported package {name!r} has no concrete source path")
        path = Path(module_file).resolve(strict=True)
        if not path.is_relative_to(import_roots[name]):
            raise RuntimeError(
                f"imported package {name!r} escaped its authenticated root: {path}"
            )
        bindings[name] = str(path)
    return bindings


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int, int]:
    """Read one bounded SafeTensors header and validate its physical envelope."""

    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"SafeTensors file is shorter than its length prefix: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes == 0 or header_bytes > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"SafeTensors header length is invalid in {path}: {header_bytes}")
        if 8 + header_bytes > file_bytes:
            raise ValueError(f"SafeTensors header exceeds file size in {path}")
        payload = handle.read(header_bytes)
        if len(payload) != header_bytes:
            raise ValueError(f"SafeTensors header was truncated in {path}")
    header = json_bytes_without_duplicate_keys(payload, str(path))
    if not isinstance(header, dict):
        raise ValueError(f"SafeTensors header is not an object: {path}")
    return header, 8 + header_bytes, file_bytes


def canonical_header_inventory_digest(tensors: Mapping[str, Mapping[str, Any]]) -> str:
    """Hash physical name, shard, dtype, shape, and byte range like Rust."""

    payload = bytearray()
    for name in sorted(tensors):
        tensor = tensors[name]
        payload.extend(name.encode("utf-8"))
        payload.append(0)
        payload.extend(str(tensor["shard"]).encode("utf-8"))
        payload.append(0)
        payload.extend(str(tensor["dtype"]).encode("utf-8"))
        payload.append(0)
        for dimension in tensor["shape"]:
            payload.extend(str(dimension).encode("ascii"))
            payload.append(ord(","))
        payload.append(0)
        payload.extend(str(tensor["data_offsets"][0]).encode("ascii"))
        payload.append(ord(":"))
        payload.extend(str(tensor["data_offsets"][1]).encode("ascii"))
        payload.append(ord("\n"))
    return sha256_bytes(bytes(payload))


def verify_checkpoint(model_dir: Path) -> dict[str, Any]:
    """Validate and hash the exact config, index, headers, shards, and aliases."""

    root_inventory = verify_root_inventory(model_dir)
    config = json_without_duplicate_keys(model_dir / "config.json")
    expected_config = {
        "architectures": ["T5ForConditionalGeneration"],
        "d_ff": 10_240,
        "d_kv": 64,
        "d_model": HIDDEN_SIZE,
        "decoder_start_token_id": DECODER_START_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "feed_forward_proj": "gated-gelu",
        "is_encoder_decoder": True,
        "layer_norm_epsilon": 1.0e-6,
        "num_decoder_layers": 24,
        "num_heads": 64,
        "num_layers": 24,
        "pad_token_id": PAD_TOKEN_ID,
        "relative_attention_max_distance": 128,
        "relative_attention_num_buckets": 32,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "vocab_size": VOCAB_SIZE,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"config field {key!r}: expected {expected!r}, "
                f"found {config.get(key)!r}"
            )

    generation = json_without_duplicate_keys(model_dir / "generation_config.json")
    expected_generation = {
        "decoder_start_token_id": DECODER_START_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
    }
    for key, expected in expected_generation.items():
        if generation.get(key) != expected:
            raise ValueError(
                f"generation field {key!r}: expected {expected!r}, "
                f"found {generation.get(key)!r}"
            )

    index = json_without_duplicate_keys(model_dir / "model.safetensors.index.json")
    if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
        raise ValueError("checkpoint index must contain exactly metadata and weight_map")
    metadata = index["metadata"]
    weight_map = index["weight_map"]
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise ValueError("checkpoint index metadata and weight_map must be objects")
    declared_total = checked_nonnegative_integer(
        metadata.get("total_size"), "index.metadata.total_size"
    )
    if (
        declared_total != EXPECTED_INDEX_TOTAL_BYTES
        or len(weight_map) != EXPECTED_LOGICAL_INDEX_KEYS
    ):
        raise ValueError("checkpoint index size or logical key count changed")
    shard_names = set(weight_map.values())
    if shard_names != set(EXPECTED_SHARDS):
        raise ValueError(f"checkpoint shard set changed: {sorted(shard_names)}")
    if any(not isinstance(name, str) or Path(name).name != name for name in weight_map.values()):
        raise ValueError("checkpoint index contains a non-local shard path")

    tensors: dict[str, dict[str, Any]] = {}
    physical_bytes = 0
    for shard_name in sorted(EXPECTED_SHARDS):
        path, observed_file_bytes = regular_file_identity(model_dir, shard_name)
        expected_file_bytes, _ = EXPECTED_SHARDS[shard_name]
        if observed_file_bytes != expected_file_bytes:
            raise ValueError(f"shard size changed for {shard_name}: {observed_file_bytes}")
        header, data_start, file_bytes = read_safetensors_header(path)
        shard_metadata = header.pop("__metadata__", None)
        if shard_metadata != {"format": "pt"}:
            raise ValueError(f"unexpected SafeTensors metadata in {shard_name}: {shard_metadata!r}")
        ranges: list[tuple[int, int, str]] = []
        for name, item in header.items():
            if name in tensors:
                raise ValueError(f"duplicate physical tensor across shards: {name}")
            if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
                raise ValueError(f"invalid SafeTensors record for {name!r}")
            if item["dtype"] != "F32" or not isinstance(item["shape"], list):
                raise ValueError(f"unsupported dtype or shape for {name!r}: {item!r}")
            shape = [checked_nonnegative_integer(value, f"{name}.shape") for value in item["shape"]]
            offsets = item["data_offsets"]
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"invalid data offsets for {name!r}")
            start, end = [
                checked_nonnegative_integer(value, f"{name}.data_offsets")
                for value in offsets
            ]
            byte_length = checked_product(shape, f"{name}.shape") * 4
            if start > end or end - start != byte_length or data_start + end > file_bytes:
                raise ValueError(f"shape/range/file-size mismatch for {name!r}")
            ranges.append((start, end, name))
            tensors[name] = {
                "shard": shard_name,
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, end],
                "file_data_start": data_start,
                "byte_length": byte_length,
            }
            physical_bytes += byte_length
        cursor = 0
        for start, end, name in sorted(ranges):
            if start != cursor:
                raise ValueError(
                    f"non-contiguous or overlapping data before {name!r} "
                    f"in {shard_name}"
                )
            cursor = end
        if data_start + cursor != file_bytes:
            raise ValueError(f"unaccounted trailing data in {shard_name}")

    if (
        len(tensors) != EXPECTED_PHYSICAL_TENSORS
        or physical_bytes != EXPECTED_CHECKPOINT_TENSOR_BYTES
    ):
        raise ValueError(
            f"physical tensor inventory changed: tensors={len(tensors)}, "
            f"bytes={physical_bytes}"
        )
    physical_names = set(tensors)
    index_names = set(weight_map)
    missing_physical = index_names - physical_names
    unexpected_physical = physical_names - index_names
    if missing_physical != {"encoder.embed_tokens.weight"} or unexpected_physical:
        raise ValueError(
            f"index/header key relation changed: missing={missing_physical}, "
            f"extra={unexpected_physical}"
        )
    for name, tensor in tensors.items():
        if weight_map[name] != tensor["shard"]:
            raise ValueError(f"index assigns {name!r} to the wrong physical shard")
    alias_source = tensors["shared.weight"]
    if weight_map["encoder.embed_tokens.weight"] != alias_source["shard"]:
        raise ValueError("encoder embedding alias is no longer assigned to shared.weight's shard")
    if physical_bytes + alias_source["byte_length"] != declared_total:
        raise ValueError("logical index total is not physical bytes plus one embedding alias")
    header_digest = canonical_header_inventory_digest(tensors)
    if header_digest != EXPECTED_HEADER_INVENTORY_SHA256:
        raise ValueError(f"SafeTensors header inventory changed: {header_digest}")

    artifacts: list[dict[str, Any]] = []
    for relative_path, (expected_bytes, expected_hash) in sorted(
        (EXPECTED_CONTROL_FILES | EXPECTED_SHARDS).items()
    ):
        path, observed_bytes = regular_file_identity(model_dir, relative_path)
        if observed_bytes != expected_bytes:
            raise ValueError(f"artifact size changed for {relative_path}: {observed_bytes}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(f"artifact SHA-256 changed for {relative_path}: {observed_hash}")
        artifacts.append(
            {
                "relative_path": relative_path,
                "bytes": observed_bytes,
                "sha256": observed_hash,
            }
        )
    artifact_digest = canonical_artifact_digest(artifacts)
    if artifact_digest != EXPECTED_ARTIFACT_MANIFEST_SHA256:
        raise ValueError(f"artifact manifest changed: {artifact_digest}")

    embedding_hashes: dict[str, str] = {}
    for name, expected_hash in EXPECTED_EMBEDDING_CONTENT_HASHES.items():
        tensor = tensors[name]
        path = model_dir / str(tensor["shard"])
        offset = int(tensor["file_data_start"]) + int(tensor["data_offsets"][0])
        observed_hash = sha256_file_range(path, offset, int(tensor["byte_length"]))
        if observed_hash != expected_hash:
            raise ValueError(f"embedding tensor content changed for {name}: {observed_hash}")
        embedding_hashes[name] = observed_hash
    if embedding_hashes["shared.weight"] != embedding_hashes["decoder.embed_tokens.weight"]:
        raise ValueError("shared and decoder embedding tensors are no longer byte-identical")
    if embedding_hashes["shared.weight"] == embedding_hashes["lm_head.weight"]:
        raise ValueError("the independent LM head unexpectedly equals the shared embedding")

    return {
        "root_inventory": root_inventory,
        "artifacts": artifacts,
        "artifact_manifest_sha256": artifact_digest,
        "header_inventory_sha256": header_digest,
        "physical_tensor_count": len(tensors),
        "logical_index_key_count": len(weight_map),
        "checkpoint_tensor_bytes": physical_bytes,
        "index_declared_total_bytes": declared_total,
        "resolved_index_alias": {
            "alias": "encoder.embed_tokens.weight",
            "storage_key": "shared.weight",
        },
        "embedding_content_sha256": embedding_hashes,
    }


def git_provenance(script_path: Path) -> dict[str, Any]:
    """Record the containing Git revision and whether this generator is dirty."""

    repository_dir = script_path.parents[3]

    def git(*arguments: str) -> str:
        """Run one read-only Git query at the repository root."""

        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "relative_path": "crates/transformers/scripts/generate_flan_t5_xxl_reference.py",
        "sha256": sha256_file(script_path),
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", str(script_path))),
    }


def loading_info_summary(loading_info: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize loading diagnostics and reject every unexplained tensor key."""

    summary: dict[str, Any] = {}
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = sorted(str(value) for value in loading_info.get(name, []))
        summary[name] = {
            "count": len(values),
            "sha256": sha256_bytes("\n".join(values).encode("utf-8")),
        }
        if values:
            raise RuntimeError(f"official T5 load reported {name}: {values[:10]}")
    return summary


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately small, offline reference-generation interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate an atomic FLAN-T5-XXL MPS F32 reference bundle with encoder, "
            "decoder, logits, cached greedy, teacher-forced, and four-beam evidence."
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help=f"local {REPOSITORY} checkout at revision {REVISION}",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=f"existing external parent; create-only {BUNDLE_DIRNAME!r} must be absent",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="hash and validate every artifact without importing Torch or loading the model",
    )
    return parser.parse_args(argv)


def synchronize(torch: Any) -> None:
    """Synchronize Metal before every measured wall-clock boundary."""

    torch.mps.synchronize()


def checked_duration(label: str, value: float) -> float:
    """Reject a negative or non-finite timing before serialization."""

    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"{label} duration is invalid: {value!r}")
    return value


def peak_process_rss_bytes() -> int:
    """Return the operating system's process-resident high-water mark in bytes."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value < 0:
        raise RuntimeError(f"process peak RSS is negative: {value}")
    # Darwin reports bytes; Linux and the BSD-compatible Linux interface report
    # KiB.  These are the only platforms on which atomic publication is enabled.
    return value if sys.platform == "darwin" else value * 1024


class RuntimeMemoryTracker:
    """Sample MPS allocation while retaining the kernel-maintained RSS peak.

    PyTorch exposes instantaneous MPS allocator counters but no Metal peak API.
    A short-lived daemon samples both counters every 20 ms and reports that
    limitation explicitly.  Process RSS comes from the kernel high-water mark
    and therefore does not depend on the sampling interval.
    """

    def __init__(self, torch: Any, interval_seconds: float = MPS_MEMORY_SAMPLE_SECONDS):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
            raise ValueError("memory sampling interval must be positive and finite")
        self.torch = torch
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.sample_count = 0
        self.maximum_current_allocated = 0
        self.maximum_driver_allocated = 0
        self.error: BaseException | None = None

    def sample(self) -> None:
        """Capture one pair of synchronized allocator counters."""

        current = int(self.torch.mps.current_allocated_memory())
        driver = int(self.torch.mps.driver_allocated_memory())
        if current < 0 or driver < 0 or current > driver:
            raise RuntimeError(
                f"invalid MPS allocator counters: current={current}, driver={driver}"
            )
        self.maximum_current_allocated = max(self.maximum_current_allocated, current)
        self.maximum_driver_allocated = max(self.maximum_driver_allocated, driver)
        self.sample_count += 1

    def _run(self) -> None:
        """Poll allocator counters until the owning thread requests shutdown."""

        try:
            while not self.stop_event.wait(self.interval_seconds):
                self.sample()
        except BaseException as error:  # Preserve the sampler failure for the owner.
            self.error = error
            self.stop_event.set()

    def start(self) -> None:
        """Start sampling exactly once after recording an initial observation."""

        if self.thread is not None:
            raise RuntimeError("memory tracker cannot be started twice")
        self.sample()
        self.thread = threading.Thread(
            target=self._run,
            name="flan-t5-xxl-mps-memory-sampler",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop sampling and return honest peak/high-water provenance."""

        if self.thread is None:
            raise RuntimeError("memory tracker was not started")
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, self.interval_seconds * 10.0))
        if self.thread.is_alive():
            raise RuntimeError("memory sampler did not terminate")
        self.sample()
        if self.error is not None:
            raise RuntimeError("MPS memory sampler failed") from self.error
        return {
            "process_peak_rss_bytes": peak_process_rss_bytes(),
            "mps_current_allocated_sampled_peak_bytes": self.maximum_current_allocated,
            "mps_driver_allocated_sampled_peak_bytes": self.maximum_driver_allocated,
            "mps_peak_kind": "sampled-high-water-no-native-metal-peak-api",
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": self.sample_count,
        }


def tensor_cosine(torch: Any, left: Any, right: Any) -> float:
    """Compute vector cosine with CPU float64 accumulation."""

    if left.shape != right.shape or left.numel() == 0:
        raise ValueError("cosine tensors must have equal nonempty shapes")
    left64 = left.reshape(-1).to(dtype=torch.float64, device="cpu")
    right64 = right.reshape(-1).to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(left64).all().item()) or not bool(
        torch.isfinite(right64).all().item()
    ):
        raise ValueError("cosine input contains a non-finite value")
    denominator = torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64)
    denominator_value = float(denominator.item())
    if denominator_value == 0.0 or not math.isfinite(denominator_value):
        raise ValueError("cosine denominator is zero or non-finite")
    result = float(torch.dot(left64, right64).item()) / denominator_value
    if not math.isfinite(result):
        raise ValueError("cosine result is non-finite")
    return result


def tensor_parity_summary(
    torch: Any,
    actual: Any,
    expected: Any,
    *,
    atol: float = COMPATIBILITY_ATOL,
    rtol: float = COMPATIBILITY_RTOL,
) -> dict[str, Any]:
    """Enforce and summarize elementwise cached-versus-stateless F32 parity."""

    if actual.shape != expected.shape or actual.numel() == 0:
        raise ValueError("parity tensors must have equal nonempty shapes")
    actual = actual.detach().to(dtype=torch.float32, device="cpu").contiguous()
    expected = expected.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if not bool(torch.isfinite(actual).all().item()) or not bool(
        torch.isfinite(expected).all().item()
    ):
        raise ValueError("parity tensor contains a non-finite value")
    difference = (actual - expected).abs()
    allowance = atol + rtol * expected.abs()
    failures = difference > allowance
    failure_count = int(failures.sum().item())
    maximum_absolute = float(difference.max().item())
    maximum_normalized = float((difference / allowance).max().item())
    cosine = tensor_cosine(torch, actual, expected)
    if failure_count != 0:
        raise RuntimeError(
            "cached/stateless elementwise parity failed: "
            f"failures={failure_count}, max_abs={maximum_absolute}, "
            f"max_normalized={maximum_normalized}"
        )
    return {
        "atol": atol,
        "rtol": rtol,
        "elements": actual.numel(),
        "failure_count": failure_count,
        "maximum_absolute_error": maximum_absolute,
        "maximum_normalized_error": maximum_normalized,
        "cosine": cosine,
    }


def assert_mps_tensor(torch: Any, tensor: Any, label: str) -> None:
    """Require one representative runtime tensor to remain finite F32 on MPS."""

    if not torch.is_tensor(tensor):
        raise RuntimeError(f"{label} is not a tensor")
    if tensor.device.type != "mps" or tensor.dtype != torch.float32:
        raise RuntimeError(
            f"{label} must be MPS F32, found device={tensor.device}, dtype={tensor.dtype}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{label} contains a non-finite value")


def module_device_inventory(torch: Any, model: Any) -> dict[str, Any]:
    """Prove all model parameters and persistent buffers reside on Metal."""

    parameters = list(model.named_parameters())
    buffers = list(model.named_buffers())
    for kind, entries in (("parameter", parameters), ("buffer", buffers)):
        for name, tensor in entries:
            if tensor.device.type != "mps":
                raise RuntimeError(f"model {kind} {name!r} is not on MPS: {tensor.device}")
    return {
        "parameter_tensors": len(parameters),
        "parameter_elements": sum(tensor.numel() for _, tensor in parameters),
        "buffer_tensors": len(buffers),
        "buffer_elements": sum(tensor.numel() for _, tensor in buffers),
        "device_types": ["mps"],
    }


def cache_device_inventory(
    torch: Any,
    cache: Any,
    *,
    expected_layers: int,
    expected_self_length: int,
    expected_cross_length: int,
) -> dict[str, Any]:
    """Validate exact T5 self/cross-cache topology, shapes, dtype, and device."""

    layers = list(cache)
    if len(layers) != expected_layers:
        raise RuntimeError(
            f"cache layer count changed: expected {expected_layers}, found {len(layers)}"
        )
    tensor_count = 0
    element_count = 0
    for layer_index, layer in enumerate(layers):
        if len(layer) != 6 or layer[2] is not None or layer[5] is not None:
            raise RuntimeError(f"cache layer {layer_index} layout changed")
        for label, tensor, expected_length in (
            ("self_key", layer[0], expected_self_length),
            ("self_value", layer[1], expected_self_length),
            ("cross_key", layer[3], expected_cross_length),
            ("cross_value", layer[4], expected_cross_length),
        ):
            assert_mps_tensor(torch, tensor, f"cache[{layer_index}].{label}")
            expected_shape = [1, 64, expected_length, 64]
            if list(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"cache[{layer_index}].{label} shape changed: "
                    f"expected {expected_shape}, found {list(tensor.shape)}"
                )
            tensor_count += 1
            element_count += tensor.numel()
    return {
        "layers": len(layers),
        "tensor_count": tensor_count,
        "elements": element_count,
        "self_sequence_length": expected_self_length,
        "cross_sequence_length": expected_cross_length,
        "dtype": "float32",
        "device": "mps",
    }


def normalized_generation_config(
    generation_config: Any, overrides: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Materialize and serialize every effective beam-generation setting."""

    effective = copy.deepcopy(generation_config)
    effective.update(
        defaults_only=True,
        **effective._get_default_generation_params(),
    )
    unused = effective.update(defaults_only=False, **dict(overrides))
    if unused:
        raise RuntimeError(f"beam generation overrides were not consumed: {sorted(unused)}")
    effective.validate(strict=True)
    serialized = json.loads(
        effective.to_json_string(use_diff=False, ignore_metadata=False)
    )
    if not isinstance(serialized, dict):
        raise RuntimeError("normalized GenerationConfig did not serialize to an object")
    strict_json_bytes(serialized)
    return effective, serialized


def greedy_decision_evidence(
    step: int, top_ids: Sequence[int], top_values: Sequence[float]
) -> dict[str, Any]:
    """Prove one top-1 choice stable under the crate's F32 parity tolerance."""

    if len(top_ids) != 2 or len(top_values) != 2 or top_ids[0] == top_ids[1]:
        raise ValueError("greedy evidence requires two distinct ranked candidates")
    first, second = (float(top_values[0]), float(top_values[1]))
    if not math.isfinite(first) or not math.isfinite(second) or first < second:
        raise ValueError("greedy top-two scores must be finite and descending")
    first_epsilon = COMPATIBILITY_ATOL + COMPATIBILITY_RTOL * abs(first)
    second_epsilon = COMPATIBILITY_ATOL + COMPATIBILITY_RTOL * abs(second)
    combined_epsilon = first_epsilon + second_epsilon
    margin = first - second
    ratio = margin / combined_epsilon
    return {
        "step": step,
        "top1_token_id": int(top_ids[0]),
        "top2_token_id": int(top_ids[1]),
        "top1_logit": first,
        "top2_logit": second,
        "margin": margin,
        "top1_error_bound": first_epsilon,
        "top2_error_bound": second_epsilon,
        "combined_error_bound": combined_epsilon,
        "margin_to_combined_error_ratio": ratio,
        "stable_under_f32_parity_tolerance": ratio > 1.0,
    }


def validate_decision_evidence(decisions: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Require all greedy decisions to survive the documented error budget."""

    if not decisions:
        raise RuntimeError("cannot validate an empty generation")
    unstable = [
        int(item["step"])
        for item in decisions
        if not item["stable_under_f32_parity_tolerance"]
    ]
    minimum_margin = min(float(item["margin"]) for item in decisions)
    minimum_ratio = min(float(item["margin_to_combined_error_ratio"]) for item in decisions)
    if unstable or not math.isfinite(minimum_margin) or not math.isfinite(minimum_ratio):
        raise RuntimeError(f"greedy decisions fail F32 tolerance stability at steps {unstable}")
    return minimum_margin, minimum_ratio


def device_details(torch: Any) -> dict[str, Any]:
    """Validate and describe the required Apple Metal reference device."""

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("FLAN-T5-XXL reference generation requires an available MPS backend")
    recommended = int(torch.mps.recommended_max_memory())
    if recommended < MIN_MPS_RECOMMENDED_BYTES:
        raise RuntimeError(
            f"MPS recommends only {recommended} bytes; at least "
            f"{MIN_MPS_RECOMMENDED_BYTES} are required"
        )
    return {
        "kind": "mps",
        "built": True,
        "available": True,
        "machine": platform.machine(),
        "recommended_max_memory": recommended,
        "driver_allocated_memory_before_load": int(torch.mps.driver_allocated_memory()),
        "current_allocated_memory_before_load": int(torch.mps.current_allocated_memory()),
    }


def load_with_verified_mps_allocator_warmup_noop(
    modeling_utils: Any, torch: Any, expected_model_class: type, loader: Any
) -> tuple[Any, dict[str, Any]]:
    """Disable only Transformers' impossible synthetic MPS warmup allocation."""

    original = getattr(modeling_utils, "caching_allocator_warmup", None)
    if (
        getattr(modeling_utils, "__name__", None) != "transformers.modeling_utils"
        or not callable(original)
        or getattr(original, "__module__", None) != "transformers.modeling_utils"
        or getattr(original, "__name__", None) != "caching_allocator_warmup"
    ):
        raise RuntimeError("Transformers allocator warmup callable identity changed")
    call_count = 0
    expanded_entries = 0

    def verified_noop(model: Any, expanded_device_map: Any, hf_quantizer: Any) -> None:
        """Validate the intercepted call and omit only its flat dummy tensor."""

        nonlocal call_count, expanded_entries
        call_count += 1
        if call_count > EXPECTED_ALLOCATOR_WARMUP_CALLS or type(model) is not expected_model_class:
            raise RuntimeError("allocator warmup invocation count or model class changed")
        if (
            hf_quantizer is not None
            or not isinstance(expanded_device_map, dict)
            or not expanded_device_map
        ):
            raise RuntimeError(
                "allocator warmup override requires an unquantized nonempty device map"
            )
        devices = {str(torch.device(value).type) for value in expanded_device_map.values()}
        if devices != {"mps"} or any(not isinstance(key, str) for key in expanded_device_map):
            raise RuntimeError(
                f"allocator warmup override requires an all-MPS map, found {devices}"
            )
        expanded_entries = len(expanded_device_map)

    setattr(modeling_utils, "caching_allocator_warmup", verified_noop)
    remained_scoped = False
    try:
        loaded = loader()
        remained_scoped = modeling_utils.caching_allocator_warmup is verified_noop
    finally:
        setattr(modeling_utils, "caching_allocator_warmup", original)
    if not remained_scoped or modeling_utils.caching_allocator_warmup is not original:
        raise RuntimeError("allocator warmup override escaped or was replaced")
    if call_count != EXPECTED_ALLOCATOR_WARMUP_CALLS:
        raise RuntimeError(f"expected one allocator warmup call, found {call_count}")
    return loaded, {
        "target": ALLOCATOR_WARMUP_TARGET,
        "policy": ALLOCATOR_WARMUP_POLICY,
        "scope": ALLOCATOR_WARMUP_SCOPE,
        "reason": ALLOCATOR_WARMUP_REASON,
        "call_count": call_count,
        "expanded_device_map_entries": expanded_entries,
        "expanded_device_types": ["mps"],
        "hf_quantizer_present": False,
        "restored": True,
    }


def tensor_payload(
    torch: Any, numpy: Any, tensor: Any, expected_shape: Sequence[int]
) -> tuple[bytes, list[float]]:
    """Copy one finite F32 tensor to canonical little-endian contiguous bytes."""

    cpu = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if list(cpu.shape) != list(expected_shape):
        raise RuntimeError(
            "reference tensor shape changed: "
            f"expected {list(expected_shape)}, found {list(cpu.shape)}"
        )
    if not bool(torch.isfinite(cpu).all().item()):
        raise RuntimeError("reference tensor contains a non-finite value")
    array = cpu.numpy().astype(numpy.dtype("<f4"), copy=False)
    payload = array.tobytes(order="C")
    if len(payload) != cpu.numel() * 4:
        raise RuntimeError("reference tensor byte count changed during serialization")
    values = [float(value) for value in array.reshape(-1).tolist()]
    return payload, values


def storage_identity(tensor: Any) -> tuple[int, int]:
    """Return a runtime tensor's storage pointer and byte capacity."""

    storage = tensor.untyped_storage()
    return int(storage.data_ptr()), int(storage.nbytes())


def main(argv: Sequence[str] | None = None) -> int:
    """Preflight or execute the pinned F32 reference and publish its bundle."""

    arguments = parse_arguments(argv)
    model_dir, output_dir, expected_bundle = ensure_external_paths(
        arguments.model_dir, arguments.output_dir
    )
    (
        package_versions,
        distribution_provenance,
        distribution_digest,
        import_roots,
    ) = verify_packages()
    checkpoint = verify_checkpoint(model_dir)
    reference_sources, source_digest = verify_reference_sources(import_roots["transformers"])
    if arguments.preflight_only:
        print(
            f"verified {REPOSITORY}@{REVISION}: shards={len(EXPECTED_SHARDS)}, "
            f"physical_tensors={checkpoint['physical_tensor_count']}, "
            f"checkpoint_bytes={checkpoint['checkpoint_tensor_bytes']}, "
            f"artifact_manifest_sha256={checkpoint['artifact_manifest_sha256']}, "
            f"reference_source_manifest_sha256={source_digest}"
        )
        return 0

    # Heavy imports begin only after the complete 45 GB checkpoint identity and
    # every executable wheel payload have passed their nonallocating gates.
    bytecode_cache, import_resolutions, previous_bytecode_state = prepare_authenticated_imports(
        output_dir, import_roots
    )
    import accelerate
    import numpy
    import safetensors
    import tokenizers
    import torch
    import transformers
    from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration, modeling_utils
    from transformers.models.t5 import configuration_t5, modeling_t5, tokenization_t5

    import_bindings = bind_imported_modules(
        {
            "accelerate": accelerate,
            "numpy": numpy,
            "safetensors": safetensors,
            "tokenizers": tokenizers,
            "torch": torch,
            "transformers": transformers,
        },
        import_roots,
    )
    transformers_root = Path(transformers.__file__).resolve(strict=True).parent
    if transformers_root != import_roots["transformers"]:
        raise RuntimeError("the imported Transformers root differs from its authenticated root")
    imported = {
        configuration_t5.__name__: Path(configuration_t5.__file__).resolve(strict=True),
        modeling_t5.__name__: Path(modeling_t5.__file__).resolve(strict=True),
        tokenization_t5.__name__: Path(tokenization_t5.__file__).resolve(strict=True),
    }
    for module, path in imported.items():
        expected_relative = EXPECTED_REFERENCE_SOURCES[module][0]
        if path != (transformers_root / expected_relative).resolve(strict=True):
            raise RuntimeError(f"imported module {module!r} does not match its pinned source")

    details = device_details(torch)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    torch.mps.set_per_process_memory_fraction(1.0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_grad_enabled(False)

    config = T5Config.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, revision=REVISION
    )
    if config.tie_word_embeddings is not True or config.scale_decoder_outputs is not False:
        raise RuntimeError("Transformers' pinned T5 v1.1 compatibility rewrite changed")
    config._attn_implementation = "eager"

    memory_tracker = RuntimeMemoryTracker(torch)
    memory_tracker.start()
    load_started = time.perf_counter()
    (model, loading_info), allocator_override = load_with_verified_mps_allocator_warmup_noop(
        modeling_utils,
        torch,
        T5ForConditionalGeneration,
        lambda: T5ForConditionalGeneration.from_pretrained(
            model_dir,
            config=config,
            dtype=torch.float32,
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
    synchronize(torch)
    memory_tracker.sample()
    load_seconds = checked_duration("model load", time.perf_counter() - load_started)
    if type(model) is not T5ForConditionalGeneration:
        raise RuntimeError(f"loaded unexpected model class {type(model)!r}")
    load_diagnostics = loading_info_summary(loading_info)
    model.eval()

    reported_device_map = getattr(model, "hf_device_map", None)
    normalized_device_map = (
        None
        if reported_device_map is None
        else {
            str(name): str(torch.device(assignment))
            for name, assignment in reported_device_map.items()
        }
    )
    if normalized_device_map is not None and (
        not normalized_device_map
        or {torch.device(value).type for value in normalized_device_map.values()} != {"mps"}
    ):
        raise RuntimeError(f"reported model device map escaped MPS: {normalized_device_map}")
    model_device_inventory = module_device_inventory(torch, model)

    named_parameters = dict(model.named_parameters())
    loaded_parameter_count = sum(parameter.numel() for parameter in named_parameters.values())
    if (
        len(named_parameters) != EXPECTED_UNIQUE_PARAMETERS
        or loaded_parameter_count != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            f"loaded parameter inventory changed: tensors={len(named_parameters)}, "
            f"elements={loaded_parameter_count}"
        )
    if len(model.state_dict()) != EXPECTED_LOGICAL_STATE_KEYS:
        raise RuntimeError(f"logical state key count changed: {len(model.state_dict())}")
    parameter_dtypes: dict[str, int] = {}
    for parameter in named_parameters.values():
        name = str(parameter.dtype).removeprefix("torch.")
        parameter_dtypes[name] = parameter_dtypes.get(name, 0) + parameter.numel()
    if parameter_dtypes != {"float32": EXPECTED_PARAMETER_COUNT}:
        raise RuntimeError(f"loaded parameter dtypes changed: {parameter_dtypes}")

    embedding_storage = {
        name: storage_identity(model.get_parameter(name))
        for name in (
            "shared.weight",
            "encoder.embed_tokens.weight",
            "decoder.embed_tokens.weight",
            "lm_head.weight",
        )
    }
    if embedding_storage["encoder.embed_tokens.weight"] != embedding_storage["shared.weight"]:
        raise RuntimeError("missing encoder embedding was not tied to shared storage")
    independent_storage = {
        embedding_storage[name]
        for name in (
            "shared.weight",
            "decoder.embed_tokens.weight",
            "lm_head.weight",
        )
    }
    if len(independent_storage) != 3:
        raise RuntimeError("physical shared, decoder, and LM-head tensors were unexpectedly tied")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        revision=REVISION,
        use_fast=True,
    )
    observed_translation_ids = tokenizer.encode(
        TRANSLATION_PROMPT, add_special_tokens=True
    )
    if observed_translation_ids != TRANSLATION_INPUT_IDS:
        raise RuntimeError(
            f"pinned translation tokenization changed: {observed_translation_ids}"
        )
    observed_long_form_ids = tokenizer.encode(LONG_FORM_PROMPT, add_special_tokens=True)
    if observed_long_form_ids != LONG_FORM_INPUT_IDS:
        raise RuntimeError(
            f"pinned long-form tokenization changed: {observed_long_form_ids}"
        )
    if (
        tokenizer.eos_token_id != EOS_TOKEN_ID
        or tokenizer.pad_token_id != PAD_TOKEN_ID
        or tokenizer.unk_token_id != UNK_TOKEN_ID
        or tokenizer.vocab_size != TOKENIZER_VOCAB_SIZE
        or getattr(tokenizer, "backend", None) != "tokenizers"
    ):
        raise RuntimeError("pinned tokenizer identity or special-token IDs changed")

    source_ids = torch.tensor([LONG_FORM_INPUT_IDS], dtype=torch.long, device="mps")
    source_mask = torch.ones_like(source_ids, dtype=torch.long, device="mps")
    translation_ids = torch.tensor(
        [TRANSLATION_INPUT_IDS], dtype=torch.long, device="mps"
    )
    translation_mask = torch.ones_like(translation_ids, dtype=torch.long, device="mps")
    decoder_start = torch.tensor([[DECODER_START_TOKEN_ID]], dtype=torch.long, device="mps")
    decoder_start_mask = torch.ones_like(decoder_start, dtype=torch.long, device="mps")
    effective_beam_config, effective_beam_config_record = normalized_generation_config(
        model.generation_config,
        {
            "num_beams": BEAM_COUNT,
            "max_new_tokens": BEAM_MAX_NEW_TOKENS,
            "do_sample": False,
            "early_stopping": True,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": False,
        },
    )
    for field, expected in (
        ("decoder_start_token_id", DECODER_START_TOKEN_ID),
        ("eos_token_id", EOS_TOKEN_ID),
        ("pad_token_id", PAD_TOKEN_ID),
        ("num_beams", BEAM_COUNT),
        ("max_new_tokens", BEAM_MAX_NEW_TOKENS),
    ):
        if getattr(effective_beam_config, field) != expected:
            raise RuntimeError(
                f"effective beam generation field {field!r} changed: "
                f"{getattr(effective_beam_config, field)!r}"
            )

    with torch.inference_mode():
        encoder_started = time.perf_counter()
        encoder_output = model.encoder(
            input_ids=source_ids,
            attention_mask=source_mask,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        synchronize(torch)
        memory_tracker.sample()
        assert_mps_tensor(
            torch, encoder_output.last_hidden_state, "long-form encoder output"
        )
        encoder_seconds = checked_duration("encoder", time.perf_counter() - encoder_started)

        stateless_started = time.perf_counter()
        stateless = model(
            encoder_outputs=encoder_output,
            attention_mask=source_mask,
            decoder_input_ids=decoder_start,
            decoder_attention_mask=decoder_start_mask,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        synchronize(torch)
        memory_tracker.sample()
        stateless_seconds = checked_duration(
            "stateless first decoder", time.perf_counter() - stateless_started
        )
        if stateless.logits.dtype != torch.float32 or stateless.decoder_hidden_states is None:
            raise RuntimeError("stateless T5 output dtype or hidden-state diagnostics changed")
        assert_mps_tensor(torch, stateless.logits, "stateless first logits")
        assert_mps_tensor(
            torch,
            stateless.decoder_hidden_states[-1],
            "stateless first decoder hidden state",
        )
        first_logits = stateless.logits[0, -1].detach().to(device="cpu").contiguous()

        greedy_started = time.perf_counter()
        cached = model(
            encoder_outputs=encoder_output,
            attention_mask=source_mask,
            decoder_input_ids=decoder_start,
            decoder_attention_mask=decoder_start_mask,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        cache = cached.past_key_values
        if cache is None:
            raise RuntimeError("cached T5 prefill returned no cache")
        generated: list[int] = []
        cached_rows: list[Any] = []
        decisions: list[dict[str, Any]] = []
        for step in range(MAX_GREEDY_TOKENS):
            if cached.logits.dtype != torch.float32:
                raise RuntimeError(f"cached logits changed dtype at step {step}")
            assert_mps_tensor(torch, cached.logits, f"cached logits at step {step}")
            row = cached.logits[0, -1].detach().to(device="cpu").contiguous()
            if row.numel() != VOCAB_SIZE or not bool(torch.isfinite(row).all().item()):
                raise RuntimeError(
                    f"cached logits changed shape or became non-finite at step {step}"
                )
            cached_rows.append(row)
            top_values, top_indices = torch.topk(row, k=2, largest=True, sorted=True)
            decision = greedy_decision_evidence(step, top_indices.tolist(), top_values.tolist())
            decisions.append(decision)
            token_id = int(top_indices[0].item())
            generated.append(token_id)
            if token_id == EOS_TOKEN_ID:
                raise RuntimeError(
                    f"long-form greedy sequence reached EOS before the required "
                    f"{MAX_GREEDY_TOKENS} tokens at step {step}"
                )
            if step + 1 == MAX_GREEDY_TOKENS:
                break
            full_length = step + 2
            cached = model(
                encoder_outputs=encoder_output,
                attention_mask=source_mask,
                decoder_input_ids=torch.tensor([[token_id]], dtype=torch.long, device="mps"),
                decoder_attention_mask=torch.ones((1, full_length), dtype=torch.long, device="mps"),
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
            cache = cached.past_key_values
            if cache is None:
                raise RuntimeError(f"cached decode returned no cache after step {step}")
        synchronize(torch)
        memory_tracker.sample()
        greedy_seconds = checked_duration("cached greedy", time.perf_counter() - greedy_started)
        if len(generated) != MAX_GREEDY_TOKENS or EOS_TOKEN_ID in generated:
            raise RuntimeError(
                f"long-form greedy gate must produce exactly {MAX_GREEDY_TOKENS} "
                f"non-EOS tokens, found {generated}"
            )
        minimum_margin, minimum_ratio = validate_decision_evidence(decisions)

        teacher_ids = [DECODER_START_TOKEN_ID, *generated[:-1]]
        teacher_started = time.perf_counter()
        teacher = model(
            encoder_outputs=encoder_output,
            attention_mask=source_mask,
            decoder_input_ids=torch.tensor([teacher_ids], dtype=torch.long, device="mps"),
            decoder_attention_mask=torch.ones(
                (1, len(teacher_ids)), dtype=torch.long, device="mps"
            ),
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        synchronize(torch)
        memory_tracker.sample()
        assert_mps_tensor(torch, teacher.logits, "teacher-forced logits")
        teacher_seconds = checked_duration(
            "teacher-forced decoder", time.perf_counter() - teacher_started
        )
        teacher_rows = teacher.logits[0].detach().to(dtype=torch.float32, device="cpu").contiguous()

        beam_started = time.perf_counter()
        beam = model.generate(
            input_ids=translation_ids,
            attention_mask=translation_mask,
            generation_config=effective_beam_config,
        )
        synchronize(torch)
        memory_tracker.sample()
        if beam.sequences.device.type != "mps":
            raise RuntimeError(f"beam sequences escaped MPS: {beam.sequences.device}")
        beam_seconds = checked_duration("four-beam generation", time.perf_counter() - beam_started)

    if list(teacher_rows.shape) != [MAX_GREEDY_TOKENS, VOCAB_SIZE]:
        raise RuntimeError(f"teacher-forced logits shape changed: {list(teacher_rows.shape)}")
    if len(cached_rows) != MAX_GREEDY_TOKENS:
        raise RuntimeError(f"cached logit row count changed: {len(cached_rows)}")
    cached_matrix = torch.stack(cached_rows, dim=0).contiguous()
    teacher_top1 = teacher_rows.argmax(dim=-1).tolist()
    if teacher_top1 != generated:
        raise RuntimeError(
            f"cached and stateless greedy continuations differ: "
            f"{generated} vs {teacher_top1}"
        )
    matrix_parity = tensor_parity_summary(torch, cached_matrix, teacher_rows)
    per_prefix_cosine = [
        tensor_cosine(torch, cached_row, teacher_rows[index])
        for index, cached_row in enumerate(cached_rows)
    ]
    per_prefix_max_abs = [
        float((cached_row - teacher_rows[index]).abs().max().item())
        for index, cached_row in enumerate(cached_rows)
    ]
    minimum_cosine = min(per_prefix_cosine)
    if minimum_cosine < SELF_PARITY_COSINE or not all(
        math.isfinite(value) for value in per_prefix_max_abs
    ):
        raise RuntimeError(f"cached/stateless self-parity failed: minimum cosine={minimum_cosine}")
    first_parity = tensor_parity_summary(torch, cached_rows[0], first_logits)

    cache_inventory = cache_device_inventory(
        torch,
        cache,
        expected_layers=24,
        expected_self_length=MAX_GREEDY_TOKENS,
        expected_cross_length=len(LONG_FORM_INPUT_IDS),
    )

    beam_ids = [int(value) for value in beam.sequences[0].detach().to(device="cpu").tolist()]
    if not beam_ids or beam_ids[0] != DECODER_START_TOKEN_ID or beam_ids[-1] != EOS_TOKEN_ID:
        raise RuntimeError(f"four-beam output lacks decoder start or terminal EOS: {beam_ids}")

    encoder_payload, encoder_values = tensor_payload(
        torch, numpy, encoder_output.last_hidden_state, [1, len(LONG_FORM_INPUT_IDS), HIDDEN_SIZE]
    )
    decoder_payload, decoder_values = tensor_payload(
        torch, numpy, stateless.decoder_hidden_states[-1], [1, 1, HIDDEN_SIZE]
    )
    logits_payload, logits_values = tensor_payload(torch, numpy, first_logits, [VOCAB_SIZE])
    cached_logits_payload, cached_logits_values = tensor_payload(
        torch, numpy, cached_matrix, [MAX_GREEDY_TOKENS, VOCAB_SIZE]
    )
    teacher_logits_payload, teacher_logits_values = tensor_payload(
        torch, numpy, teacher_rows, [MAX_GREEDY_TOKENS, VOCAB_SIZE]
    )
    runtime_memory = memory_tracker.stop()
    raw_records = {
        "encoder_final_hidden_state": {
            **vector_summary(encoder_values, [0, 1, HIDDEN_SIZE - 1, len(encoder_values) - 1]),
            "relative_path": ENCODER_RAW_FILENAME,
            "shape": [1, len(LONG_FORM_INPUT_IDS), HIDDEN_SIZE],
            "encoding": "ieee754-f32-little-endian",
            "bytes": len(encoder_payload),
            "sha256": sha256_bytes(encoder_payload),
        },
        "decoder_first_final_hidden_state": {
            **vector_summary(decoder_values, [0, 1, 100, HIDDEN_SIZE - 1]),
            "relative_path": DECODER_RAW_FILENAME,
            "shape": [1, 1, HIDDEN_SIZE],
            "encoding": "ieee754-f32-little-endian",
            "bytes": len(decoder_payload),
            "sha256": sha256_bytes(decoder_payload),
        },
        "first_logits": {
            **logits_summary(
                logits_values,
                list(
                    dict.fromkeys(
                        [
                            0,
                            1,
                            2,
                            100,
                            1_000,
                            10_000,
                            VOCAB_SIZE - 1,
                            int(first_logits.argmax().item()),
                        ]
                    )
                ),
            ),
            "relative_path": LOGITS_RAW_FILENAME,
            "shape": [VOCAB_SIZE],
            "encoding": "ieee754-f32-little-endian",
            "bytes": len(logits_payload),
            "sha256": sha256_bytes(logits_payload),
        },
        "cached_greedy_logits": {
            **vector_summary(
                cached_logits_values,
                [
                    0,
                    VOCAB_SIZE - 1,
                    VOCAB_SIZE,
                    16 * VOCAB_SIZE,
                    len(cached_logits_values) - 1,
                ],
            ),
            "relative_path": CACHED_LOGITS_RAW_FILENAME,
            "shape": [MAX_GREEDY_TOKENS, VOCAB_SIZE],
            "encoding": "ieee754-f32-little-endian",
            "bytes": len(cached_logits_payload),
            "sha256": sha256_bytes(cached_logits_payload),
            "row_top1_token_ids": generated,
        },
        "teacher_forced_logits": {
            **vector_summary(
                teacher_logits_values,
                [
                    0,
                    VOCAB_SIZE - 1,
                    VOCAB_SIZE,
                    16 * VOCAB_SIZE,
                    len(teacher_logits_values) - 1,
                ],
            ),
            "relative_path": TEACHER_LOGITS_RAW_FILENAME,
            "shape": [MAX_GREEDY_TOKENS, VOCAB_SIZE],
            "encoding": "ieee754-f32-little-endian",
            "bytes": len(teacher_logits_payload),
            "sha256": sha256_bytes(teacher_logits_payload),
            "row_top1_token_ids": teacher_top1,
        },
    }

    greedy_decoded_text = tokenizer.decode(generated, skip_special_tokens=True)
    beam_decoded_text = tokenizer.decode(beam_ids, skip_special_tokens=True)
    bytecode_policy = finalize_authenticated_imports(
        bytecode_cache, previous_bytecode_state
    )
    script_path = Path(__file__).resolve(strict=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    record = {
        "schema_version": 2,
        "repository": REPOSITORY,
        "revision": REVISION,
        "reference_class": REFERENCE_CLASS,
        "long_form_prompt": LONG_FORM_PROMPT,
        "long_form_input_ids": LONG_FORM_INPUT_IDS,
        "translation_prompt": TRANSLATION_PROMPT,
        "translation_input_ids": TRANSLATION_INPUT_IDS,
        "decoder_start_token_id": DECODER_START_TOKEN_ID,
        "execution": {
            "dtype": "float32",
            "device": "mps",
            "attention_implementation": "eager",
            "deterministic_algorithms": True,
            "seed": 0,
            "explicit_all_one_encoder_and_decoder_masks": True,
            "float32_matmul_precision": "highest",
            "mps_per_process_memory_fraction": 1.0,
            "allocator_warmup_override": allocator_override,
            "serialized_tie_word_embeddings": False,
            "runtime_tie_word_embeddings": True,
            "runtime_scale_decoder_outputs": False,
            "required_non_eos_greedy_tokens": MAX_GREEDY_TOKENS,
            "authenticated_import_bytecode_policy": bytecode_policy,
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
            "distribution_manifest_sha256": distribution_digest,
            "import_resolutions_before_execution": import_resolutions,
            "import_bindings_after_execution": import_bindings,
            "device": details,
            "runtime_memory": runtime_memory,
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
            "reference_source_manifest_sha256": source_digest,
            **checkpoint,
            "loaded_parameter_count": loaded_parameter_count,
            "loaded_unique_parameter_tensors": len(named_parameters),
            "logical_state_keys": len(model.state_dict()),
            "parameter_dtype_elements": parameter_dtypes,
            "model_device_inventory": model_device_inventory,
            "device_map": {
                "requested": {"": "mps"},
                "reported": normalized_device_map,
            },
            "greedy_cache_inventory": cache_inventory,
            "embedding_runtime_storage": {
                "alias_groups": [
                    ["shared.weight", "encoder.embed_tokens.weight"],
                    ["decoder.embed_tokens.weight"],
                    ["lm_head.weight"],
                ],
                "bytes": {
                    name: identity[1] for name, identity in embedding_storage.items()
                },
            },
            "loading_info": load_diagnostics,
        },
        "reference_tensors": raw_records,
        "greedy_generation": {
            "required_new_tokens": MAX_GREEDY_TOKENS,
            "token_ids": generated,
            "decoded_text": greedy_decoded_text,
            "eos_token_id": EOS_TOKEN_ID,
            "contains_eos": False,
            "finish_reason": f"length:{MAX_GREEDY_TOKENS}",
            "decision_stability": decisions,
            "stability_policy": {
                "name": "f32-atol-rtol-top-two-bound",
                "atol": COMPATIBILITY_ATOL,
                "rtol": COMPATIBILITY_RTOL,
                "rationale": (
                    "Each top-two score may move by atol + rtol*abs(score). The "
                    "winner is invariant when its margin exceeds the sum of both bounds."
                ),
            },
            "minimum_margin": minimum_margin,
            "minimum_margin_to_combined_error_ratio": minimum_ratio,
        },
        "teacher_forced": {
            "decoder_input_ids": teacher_ids,
            "cached_top1_token_ids": generated,
            "stateless_top1_token_ids": teacher_top1,
            "top1_equal": True,
            "elementwise_parity": matrix_parity,
            "per_prefix_cosine": per_prefix_cosine,
            "minimum_cosine": minimum_cosine,
            "per_prefix_max_abs_error": per_prefix_max_abs,
            "maximum_abs_error": max(per_prefix_max_abs),
            "first_cached_stateless_parity": first_parity,
        },
        "beam_generation": {
            "num_beams": BEAM_COUNT,
            "max_new_tokens": BEAM_MAX_NEW_TOKENS,
            "do_sample": False,
            "early_stopping": True,
            "token_ids_with_decoder_start": beam_ids,
            "generated_token_ids": beam_ids[1:],
            "decoded_text": beam_decoded_text,
            "finish_reason": f"eos:{EOS_TOKEN_ID}",
            "effective_generation_config": effective_beam_config_record,
        },
        "timings_seconds": {
            "load": load_seconds,
            "encoder": encoder_seconds,
            "stateless_first_decoder": stateless_seconds,
            "cached_greedy": greedy_seconds,
            "teacher_forced_decoder": teacher_seconds,
            "beam_generation": beam_seconds,
        },
    }
    json_payload = strict_json_bytes(record)
    published = publish_reference_bundle(
        output_dir,
        {
            ENCODER_RAW_FILENAME: encoder_payload,
            DECODER_RAW_FILENAME: decoder_payload,
            LOGITS_RAW_FILENAME: logits_payload,
            CACHED_LOGITS_RAW_FILENAME: cached_logits_payload,
            TEACHER_LOGITS_RAW_FILENAME: teacher_logits_payload,
            JSON_FILENAME: json_payload,
        },
    )
    if published != expected_bundle:
        raise RuntimeError(f"published unexpected bundle {published}, expected {expected_bundle}")
    for name, payload in sorted(
        {
            ENCODER_RAW_FILENAME: encoder_payload,
            DECODER_RAW_FILENAME: decoder_payload,
            LOGITS_RAW_FILENAME: logits_payload,
            CACHED_LOGITS_RAW_FILENAME: cached_logits_payload,
            TEACHER_LOGITS_RAW_FILENAME: teacher_logits_payload,
            JSON_FILENAME: json_payload,
        }.items()
    ):
        print(f"wrote {published / name} ({len(payload)} bytes, sha256={sha256_bytes(payload)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
