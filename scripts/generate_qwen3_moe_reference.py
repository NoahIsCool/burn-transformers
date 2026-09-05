#!/usr/bin/env python3
"""Generate the pinned Qwen3-30B-A3B official MPS/F16 reference bundle.

This is an offline maintainer tool, not part of the Rust runtime.  It loads the
official Transformers ``Qwen3MoeForCausalLM`` class from an immutable local
checkpoint, forces the auditable eager attention and eager expert paths, and
records full-vocabulary evidence for one prompt plus every one of 32 greedy
continuation prefixes.  It never calls the Hub or Transformers generation
helpers.

The fixed-name output bundle is published by a same-filesystem atomic,
no-replace directory rename.  Interrupted runs leave a hidden staging directory
for forensic inspection and can never expose a partial final bundle.

Integrity checks cover persistent modifications and ordinary concurrent races.
An actively malicious same-UID process capable of swapping and restoring paths
inside a consumer's reopen window is outside scope; such a process can also
modify this interpreter directly.  That boundary is recorded in every bundle.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import stat
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Mapping, Sequence


# Install every offline control before importing even the pure shared helper.
# ``USE_HUB_KERNELS=0`` is important for Qwen3-MoE because its official source
# carries optional decorators that can otherwise consult kernel repositories.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["USE_HUB_KERNELS"] = "0"


def _early_source_snapshot(path: Path) -> tuple[bytes, tuple[int, int, int, int, int]]:
    """Read one small Python source through a stable non-following descriptor."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("source authentication requires O_NOFOLLOW/O_NONBLOCK")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 2 * 1024 * 1024:
            raise RuntimeError(f"reference source is not a bounded regular file: {path}")
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
            raise RuntimeError(f"reference source changed during snapshot: {path}")
        return payload, identity(before)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


# Reuse the already-audited serialization, hashing, and no-replace primitives.
# Compile its exact authenticated bytes so a path swap cannot redefine the
# evidence format or publication guarantees after the digest check.
COMMON_HELPER_PATH = Path(__file__).resolve().with_name("generate_qwen38_reference.py")
COMMON_HELPER_SHA256 = "e52b3699ae08920ca8f177a7fad0b68d807732d0fae524d6c943347357ed864b"
_common_payload, _COMMON_SOURCE_IDENTITY = _early_source_snapshot(COMMON_HELPER_PATH)
if hashlib.sha256(_common_payload).hexdigest() != COMMON_HELPER_SHA256:
    raise RuntimeError(
        "shared Qwen reference helper changed; audit and update its pinned SHA-256 first"
    )
COMMON = types.ModuleType("_pinned_generate_qwen38_reference")
COMMON.__file__ = str(COMMON_HELPER_PATH)
COMMON.__package__ = ""
# Execute the exact authenticated snapshot.  A SourceFileLoader would reopen
# the path after the digest check and reintroduce a check/use race.
exec(
    compile(
        _common_payload,
        str(COMMON_HELPER_PATH),
        "exec",
        dont_inherit=True,
    ),
    COMMON.__dict__,
)
_COMMON_SOURCE_BYTES = _common_payload
del _common_payload


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
        "executing Qwen3-MoE generator code differs from its authenticated source snapshot"
    )
del _recompiled_self


REPOSITORY = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
REFERENCE_CLASS = "transformers.Qwen3MoeForCausalLM"
REFERENCE_SOURCE_RELEASE = "transformers==5.2.0"
REFERENCE_SOURCE_TAG = "v5.2.0"
REFERENCE_SOURCE_TAG_OBJECT = "0df260f32d71eb44327bed3c67de033dac004c8b"
REFERENCE_SOURCE_COMMIT = "7d9754a05193eb79b1d86aa744b622b8068008cd"
PROMPT = "Hello world, cached decoding."
INPUT_IDS = [9_707, 1_879, 11, 20_579, 47_116, 13]
VOCAB_SIZE = 151_936
GREEDY_TOKENS = 32
EXPECTED_EOS_TOKEN_IDS = [151_645, 151_643]
EXPECTED_PARAMETER_COUNT = 30_532_122_624
EXPECTED_PHYSICAL_TENSORS = 18_867
CHECKPOINT_TENSOR_BYTES = 61_064_245_248
SELF_PARITY_COSINE = 0.999

BINARY16_ELEMENTWISE_PERTURBATION_ULPS = 1.0
TOP1_STABILITY_SAFETY_FACTOR = 2.0
TOP1_STABILITY_POLICY = "binary16-elementwise-ulp-bound"
TOP1_STABILITY_RATIONALE = (
    "If each of the top two scores changes by at most epsilon, the winner can "
    "fall by epsilon while the runner-up rises by epsilon. A margin greater than "
    "2*epsilon therefore preserves top-1. Epsilon is one local binary16 ULP at "
    "the wider-spaced top-two score. This screen complements exact Burn parity."
)
INTEGRITY_THREAT_MODEL = (
    "Pre/post descriptor identities and SHA-256 detect persistent mutation and "
    "ordinary races. An actively malicious same-UID process that swaps a pathname "
    "only while Transformers or SafeTensors reopens it, then restores the original "
    "before postflight, is outside scope because it can also tamper with this process."
)

PROMPT_LOGITS_RAW_FILENAME = (
    "qwen3-30b-a3b-ad44e777-six-token-f16-last-logits.f32le"
)
CACHED_LOGITS_RAW_FILENAME = (
    "qwen3-30b-a3b-ad44e777-cached-logits-32x151936.f32le"
)
TEACHER_LOGITS_RAW_FILENAME = (
    "qwen3-30b-a3b-ad44e777-teacher-prefix-logits-32x151936.f32le"
)
JSON_FILENAME = "qwen3-30b-a3b-ad44e777-reference.json"
BUNDLE_DIRNAME = "qwen3-30b-a3b-ad44e777-reference.bundle"

ALLOCATOR_WARMUP_TARGET = "transformers.modeling_utils.caching_allocator_warmup"
ALLOCATOR_WARMUP_POLICY = "mps-exclusive-noop-during-from-pretrained"
ALLOCATOR_WARMUP_SCOPE = "Qwen3MoeForCausalLM.from_pretrained"
ALLOCATOR_WARMUP_REASON = (
    "Transformers 5.2 flattens the optional allocator warmup into an F16 tensor "
    "whose 30,532,122,624 elements exceed Metal's signed 32-bit dimension limit. "
    "Skipping this performance-only allocation does not change model tensors or math."
)
EXPECTED_ALLOCATOR_WARMUP_CALLS = 1
EXPECTED_EXPANDED_DEVICE_MAP_ENTRIES = 531
MINIMUM_MPS_LOAD_HEADROOM_BYTES = 2_000_000_000


EXPECTED_CONTROL_HASHES = {
    "config.json": "2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors.index.json": "df0d481ec595c55a0ba58426d517390c6214a566ec4ff1c8fc4bbce9f57b3c24",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


EXPECTED_SHARD_HASHES = {
    "model-00001-of-00016.safetensors": "454e77b346a61bfb201d54df60e15158838cf930617ee135113556204f2802b5",
    "model-00002-of-00016.safetensors": "47f015d6e5bb1782a834d75c06113c5e8c77ddca1cb7daf89686a4ec0deda19e",
    "model-00003-of-00016.safetensors": "ac0bf5990f2da995c1e8b77a3149dee71900b4fe1b8a614231dea9647193d96b",
    "model-00004-of-00016.safetensors": "89b01fd34a683c70fdb7f22c2e7090538d65063335ba0db46f3654642b17d1e5",
    "model-00005-of-00016.safetensors": "9849eb3584d928e35bf5956ccfd91b64207467e1fad8304aae68dc4d92d6d6d9",
    "model-00006-of-00016.safetensors": "f7a0f1525557d740158fb692986ad43506e1b5901e7406ae65808c6457bc53ad",
    "model-00007-of-00016.safetensors": "920702a50f27a009e5f676da3b02978dbd794f6ad7854f301d709136154c1a84",
    "model-00008-of-00016.safetensors": "a85bf0cc8a8047c116172d4c08cf4792eb59d4375032708ba3e1a7ffca0f708a",
    "model-00009-of-00016.safetensors": "25cd8aaed86b8e17685efb152912f92448ea3fffb1bdc247f257f04d05907604",
    "model-00010-of-00016.safetensors": "7c3307ee214797c4bc61c0994f81d690e768150df5f14d18edf84e53ba4810dc",
    "model-00011-of-00016.safetensors": "c658cad2842d36fa4c7c7f726f8515dc5670a3d80c94e13257d4e322ffe61988",
    "model-00012-of-00016.safetensors": "8cb898bc5e78492600053d1105c9ff61d7a719f5cc2802df63e41535a652cc26",
    "model-00013-of-00016.safetensors": "599594421f314cb8e8ab4474db0eb490cc6310585067e82d73821254b93319ce",
    "model-00014-of-00016.safetensors": "66d3294e9976b5f01d117a0a0bed768f128a8898eb6937e224d051de2961d363",
    "model-00015-of-00016.safetensors": "2fe000da4fc7399ff6303bb0daec8c415a5d850f5ea53863392758c77fc37664",
    "model-00016-of-00016.safetensors": "0c979e314faf06e94a5e1841abd285d4e26e763c79c1abe7710d543192d9da48",
}


EXPECTED_REFERENCE_SOURCES = {
    "transformers.activations": {"relative_path": "activations.py", "bytes": 13_176, "sha256": "10a5cb482f854c5d3afbe5bb1deab5f14ff21c800d08a38b1028538b9e1f3867"},
    "transformers.cache_utils": {"relative_path": "cache_utils.py", "bytes": 60_147, "sha256": "e4fe20ca1eeb17ea0ce7e62d1dbd0d6eab32e7072044f293612403b8e0c477fd"},
    "transformers.configuration_utils": {"relative_path": "configuration_utils.py", "bytes": 57_894, "sha256": "52100272e3bf59551c072a064053e90bc30f1fd0aa9e48fac7ad4b675c63ebcc"},
    "transformers.conversion_mapping": {"relative_path": "conversion_mapping.py", "bytes": 19_302, "sha256": "c8efcc6ce2224939299804d495ba4f361ab596e59f33e3db28e9b4629fc481e0"},
    "transformers.core_model_loading": {"relative_path": "core_model_loading.py", "bytes": 57_649, "sha256": "034f860989056b7c84ee6b6fbec197ac753d16bec120a588eef053d2d674182c"},
    "transformers.initialization": {"relative_path": "initialization.py", "bytes": 10_979, "sha256": "589934e5f43e62ea2d8923f5c38a0b917d8c8c1377f416c5a1f15b8977f0ecf1"},
    "transformers.integrations.__init__": {"relative_path": "integrations/__init__.py", "bytes": 9_612, "sha256": "f254dc19097673833d373a49acb071ab7d60e6df49c1376a70d4a8038ba47d18"},
    "transformers.integrations.hub_kernels": {"relative_path": "integrations/hub_kernels.py", "bytes": 16_705, "sha256": "d219a69a033322faae53950208748881e5073d2b8ef279b0fc3337a865042e04"},
    "transformers.integrations.moe": {"relative_path": "integrations/moe.py", "bytes": 16_997, "sha256": "1fe2594df6a3e870b87ccdf22c9b40d9b020f99ab3b90fca6ad43eca195ffa13"},
    "transformers.masking_utils": {"relative_path": "masking_utils.py", "bytes": 74_373, "sha256": "c3c82f7b7b6e03d3f04ba6c6c58a3dd6910623636452ec67ef70e3eb522f9fe7"},
    "transformers.modeling_attn_mask_utils": {"relative_path": "modeling_attn_mask_utils.py", "bytes": 21_930, "sha256": "fb3ec46639a19fda1eb99339f4e7770332a0f4f75b9de4abb34a0b1ee379838b"},
    "transformers.modeling_layers": {"relative_path": "modeling_layers.py", "bytes": 11_476, "sha256": "9e4726c07cfc895d95f9a75b0e7d23b8561575080811695e70fa879b0a18dae4"},
    "transformers.modeling_outputs": {"relative_path": "modeling_outputs.py", "bytes": 108_678, "sha256": "bb216b6b9c4b3d69f277367b3711c5f34c8e42d7f98e0a3930a64d107465f7f0"},
    "transformers.modeling_rope_utils": {"relative_path": "modeling_rope_utils.py", "bytes": 52_962, "sha256": "fa61288f66ce00bf060d3a16cc8e58aef2f94fe368c961b684bbb4bd19f90b2d"},
    "transformers.modeling_utils": {"relative_path": "modeling_utils.py", "bytes": 247_340, "sha256": "2f45c2139ce633e6e3eae9290675dd5815643b37c27f2cc4dd83586fb648f7bb"},
    "transformers.processing_utils": {"relative_path": "processing_utils.py", "bytes": 96_329, "sha256": "ea1ed16955c829323ea95566ca8f7f7e01179d99ea4ab290000c980b0b837a77"},
    "transformers.utils.generic": {"relative_path": "utils/generic.py", "bytes": 33_857, "sha256": "4f4b1d947507d946f8c755ded8e160cb15862c122c0f977ed04a1b0be52cd9fe"},
    "transformers.utils.loading_report": {"relative_path": "utils/loading_report.py", "bytes": 10_082, "sha256": "43bc59a23a1a3aa4a9e099f97a3bf73f3aca96c89cbc126e679e4b31fb526449"},
    "transformers.utils.output_capturing": {"relative_path": "utils/output_capturing.py", "bytes": 12_602, "sha256": "af0454a352e7a14a4d52d3d3def16442fb74863bd924f1c6c8fc64e28eae67de"},
    "transformers.models.qwen3_moe.configuration_qwen3_moe": {"relative_path": "models/qwen3_moe/configuration_qwen3_moe.py", "bytes": 10_579, "sha256": "5277762b80c2a5e6158adae7c806989f61d8049af7278217ff2ec5a80833fc6d"},
    "transformers.models.qwen3_moe.modeling_qwen3_moe": {"relative_path": "models/qwen3_moe/modeling_qwen3_moe.py", "bytes": 32_741, "sha256": "566091429f4362f81b5bc52cbb3b3a06603b4ee8b35446ff0087b2b02be02ed0"},
}


EXPECTED_ABSENT_DISTRIBUTIONS = ("flash-attn", "kernels", "torchao")


EXPECTED_DISTRIBUTION_PAYLOADS = {
    "accelerate": (95, 1_483_290, 84, "1a5c1e846d1805592fe0090db994c3afc37f552ca6f4aa8b1d56f4f08bbfbfae"),
    "numpy": (1_040, 20_254_356, 487, "bc49cc4a1cecd0b0e258d25f684b2250330c9672fcb64023b70eaf1f5d463f32"),
    "safetensors": (14, 1_070_455, 7, "d0d712a1786e52b270faa8b558587b1cf41fd8255499c5eb25b967c2c9911afa"),
    "tokenizers": (29, 8_506_298, 16, "b87e4f97df12e3603e619104d1b824792f6704cd405b340208980325844ff257"),
    "torch": (12_340, 338_442_177, 2_199, "e7fc648575316519c0c996f4dea905684caff4aa6edc979e3173a4ee50209570"),
    "transformers": (2_128, 43_591_674, 2_120, "e074d191c22d4b2dc0cb53d1a3afbd2f2cf81a8a2a8e8d8b685a8233c5ce09bc"),
}


def file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return immutable-enough identity fields for one authenticated run.

    Device/inode bind the directory entry to the opened object; size, mtime,
    and ctime expose in-place mutation.  Content SHA-256 remains authoritative.
    """

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
    """Hash one descriptor exactly once without following its final component.

    The descriptor is opened nonblocking so a raced FIFO cannot hang preflight.
    Descriptor metadata is checked both before and after reading, closing the
    usual ``lstat``/``open`` and hash/read races.  Large shards are streamed and
    never retained in memory.
    """

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("authenticated reference reads require O_NOFOLLOW/O_NONBLOCK")
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
            raise RuntimeError(
                f"refusing to retain {before.st_size} bytes from {path}; "
                f"limit is {maximum_payload_bytes}"
            )
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
        record = {"bytes": before.st_size, "sha256": observed_sha256}
        return record, file_identity(before), bytes(payload) if payload is not None else None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_distribution_payloads(
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, str], tuple[int, int, int, int, int]],
]:
    """Authenticate every non-bytecode payload named by each pinned wheel RECORD.

    Pinning RECORD itself only authenticates a list of intended digests.  This
    pass verifies every listed Python module, shared library, executable, and
    data file against that list, while the private bytecode prefix separately
    prevents excluded adjacent ``.pyc`` files from executing.
    """

    summaries: dict[str, dict[str, Any]] = {}
    identities: dict[tuple[str, str], tuple[int, int, int, int, int]] = {}
    for name in sorted(COMMON.EXPECTED_PACKAGE_VERSIONS):
        distribution = importlib.metadata.distribution(name)
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
            padded = record_hash.value + "=" * (-len(record_hash.value) % 4)
            try:
                expected_digest_bytes = base64.urlsafe_b64decode(padded.encode("ascii"))
            except (ValueError, UnicodeEncodeError) as error:
                raise RuntimeError(
                    f"distribution {name!r} has a malformed RECORD hash for "
                    f"{relative_path!r}"
                ) from error
            if len(expected_digest_bytes) != 32:
                raise RuntimeError(
                    f"distribution {name!r} has a non-SHA256 RECORD hash for "
                    f"{relative_path!r}"
                )
            expected_sha256 = expected_digest_bytes.hex()
            expected_bytes = int(entry.size)
            path = Path(distribution.locate_file(entry))
            observed, identity, _ = authenticated_file(
                path,
                expected_sha256=expected_sha256,
                expected_bytes=expected_bytes,
            )
            key = (name, relative_path)
            if key in identities:
                raise RuntimeError(f"duplicate RECORD entry {name!r}:{relative_path!r}")
            identities[key] = identity
            canonical.extend(relative_path.encode("utf-8"))
            canonical.append(0)
            canonical.extend(str(observed["bytes"]).encode("ascii"))
            canonical.append(0)
            canonical.extend(observed["sha256"].encode("ascii"))
            canonical.append(ord("\n"))
            verified_files += 1
            verified_bytes += expected_bytes
        if verified_files == 0 or ignored_record != 1:
            raise RuntimeError(
                f"distribution {name!r} RECORD coverage changed: "
                f"verified={verified_files}, record_entries={ignored_record}"
            )
        summary = {
            "verified_files": verified_files,
            "verified_bytes": verified_bytes,
            "ignored_bytecode_files": ignored_bytecode,
            "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        }
        expected = EXPECTED_DISTRIBUTION_PAYLOADS[name]
        if (
            summary["verified_files"],
            summary["verified_bytes"],
            summary["ignored_bytecode_files"],
            summary["manifest_sha256"],
        ) != expected:
            raise RuntimeError(
                f"distribution {name!r} payload manifest changed: "
                f"expected {expected}, found {summary}"
            )
        summaries[name] = summary
    return summaries, identities


def json_payload_without_duplicate_keys(payload: bytes, label: str) -> Any:
    """Decode one authenticated UTF-8 JSON snapshot and reject duplicate keys."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    return json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)


def assert_early_source_unchanged(
    path: Path,
    payload: bytes,
    identity: tuple[int, int, int, int, int],
) -> None:
    """Reauthenticate one source snapshot immediately before publication."""

    _, observed_identity, observed_payload = authenticated_file(
        path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_bytes=len(payload),
        retain_payload=True,
        maximum_payload_bytes=2 * 1024 * 1024,
    )
    if observed_identity != identity or observed_payload != payload:
        raise RuntimeError(f"executed reference source changed during the run: {path}")


def validated_generation_eos_token_ids(config: Mapping[str, Any]) -> list[int]:
    """Return the exact ordered EOS IDs from the pinned generation config."""

    observed = config.get("eos_token_id")
    if (
        not isinstance(observed, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in observed)
        or observed != EXPECTED_EOS_TOKEN_IDS
    ):
        raise ValueError(
            f"generation EOS IDs changed: expected {EXPECTED_EOS_TOKEN_IDS}, found {observed!r}"
        )
    return list(observed)


def greedy_decision_stability(
    *,
    step: int,
    top1_token_id: int,
    top2_token_id: int,
    top1_logit: float,
    top2_logit: float,
    eos_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Prove one greedy decision stable under a one-ULP F16 score budget."""

    if top1_token_id == top2_token_id:
        raise RuntimeError(f"greedy top-two token IDs are equal at step {step}")
    if top1_token_id in eos_token_ids:
        raise RuntimeError(
            f"official generation reached EOS token {top1_token_id} at step {step}; "
            f"all {GREEDY_TOKENS} decisions are required"
        )
    if not math.isfinite(top1_logit) or not math.isfinite(top2_logit):
        raise RuntimeError(f"greedy top-two logits became non-finite at step {step}")
    if top1_logit < top2_logit:
        raise RuntimeError(f"greedy top-two logits are out of order at step {step}")

    margin = top1_logit - top2_logit
    top1_ulp = COMMON.binary16_ulp(top1_logit)
    top2_ulp = COMMON.binary16_ulp(top2_logit)
    epsilon = max(top1_ulp, top2_ulp) * BINARY16_ELEMENTWISE_PERTURBATION_ULPS
    ratio = margin / epsilon
    if not all(math.isfinite(value) for value in (margin, top1_ulp, top2_ulp, epsilon, ratio)):
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
        "elementwise_epsilon": epsilon,
        "margin_to_epsilon_ratio": ratio,
        "stable_under_budget": ratio > TOP1_STABILITY_SAFETY_FACTOR,
    }


def validate_decision_stability(
    decisions: Sequence[Mapping[str, Any]],
) -> tuple[float, float]:
    """Require all 32 decisions to satisfy the F16 perturbation proof."""

    if len(decisions) != GREEDY_TOKENS:
        raise RuntimeError(
            f"expected {GREEDY_TOKENS} greedy decisions, found {len(decisions)}"
        )
    minimum_margin = min(float(item["margin"]) for item in decisions)
    minimum_ratio = min(float(item["margin_to_epsilon_ratio"]) for item in decisions)
    if not math.isfinite(minimum_margin) or not math.isfinite(minimum_ratio):
        raise RuntimeError("greedy stability minima are non-finite")
    unstable = [int(item["step"]) for item in decisions if not item["stable_under_budget"]]
    if unstable:
        raise RuntimeError(
            f"binary16 top-1 stability failed at steps {unstable}: "
            f"minimum_margin={minimum_margin}, minimum_ratio={minimum_ratio}, "
            f"required_ratio>{TOP1_STABILITY_SAFETY_FACTOR}"
        )
    return minimum_margin, minimum_ratio


def verify_absent_optional_distributions() -> list[str]:
    """Require optional kernel packages to be absent from the reference process."""

    absent: list[str] = []
    for name in EXPECTED_ABSENT_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            absent.append(name)
            continue
        raise RuntimeError(
            f"optional distribution {name!r} must be absent, found version {version}"
        )
    return absent


def verify_reference_sources(
    transformers_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int, int, int, int]]]:
    """Verify exact official Qwen/MoE/cache/mask source bytes and identities."""

    root = transformers_root.resolve(strict=True)
    sources: list[dict[str, Any]] = []
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    for module, expected in sorted(EXPECTED_REFERENCE_SOURCES.items()):
        relative = Path(expected["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"invalid pinned source path {relative}")
        unresolved = root / relative
        if unresolved.is_symlink():
            raise RuntimeError(f"official source is a symlink: {relative}")
        path = unresolved.resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"official source escapes the package: {relative}")
        authenticated, identity, _ = authenticated_file(
            path,
            expected_sha256=expected["sha256"],
            expected_bytes=expected["bytes"],
        )
        observed = {
            "relative_path": relative.as_posix(),
            **authenticated,
        }
        if observed != expected:
            raise RuntimeError(
                f"official reference source {module} changed: expected {expected}, "
                f"found {observed}"
            )
        sources.append({"module": module, "path": str(path), **observed})
        identities[module] = identity
    return sources, identities


def transformers_package_root() -> Path:
    """Resolve the exact installed Transformers package without importing it."""

    distribution = importlib.metadata.distribution("transformers")
    unresolved = Path(distribution.locate_file("transformers"))
    if unresolved.is_symlink():
        raise RuntimeError("installed Transformers package root must not be a symlink")
    root = unresolved.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"Transformers package root is not a directory: {root}")
    return root


def distribution_import_roots() -> dict[str, Path]:
    """Resolve each authenticated wheel's real top-level package directory."""

    roots: dict[str, Path] = {}
    for name in sorted(COMMON.EXPECTED_PACKAGE_VERSIONS):
        distribution = importlib.metadata.distribution(name)
        unresolved = Path(distribution.locate_file(name))
        metadata = unresolved.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"distribution {name!r} import root is not a real directory")
        roots[name] = unresolved.resolve(strict=True)
    return roots


def prepare_authenticated_imports(
    output_dir: Path, import_roots: Mapping[str, Path]
) -> tuple[Path, dict[str, str], tuple[bool, str | None]]:
    """Force heavy imports through source under authenticated wheel roots.

    A fresh empty ``pycache_prefix`` prevents Python from consuming a stale
    adjacent ``.pyc`` whose source path was authenticated but whose bytecode was
    not.  Writes are disabled and the empty directory is checked afterward.
    """

    already_imported = sorted(
        name
        for name in import_roots
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
        tempfile.mkdtemp(prefix=".qwen3-moe-authenticated-pycache-", dir=output_dir)
    ).resolve(strict=True)
    previous = (sys.dont_write_bytecode, sys.pycache_prefix)
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(cache_root)
    return cache_root, resolutions, previous


def bind_imported_modules(
    modules: Mapping[str, Any], import_roots: Mapping[str, Path]
) -> dict[str, str]:
    """Prove each executed top-level module came from its authenticated wheel."""

    if set(modules) != set(import_roots):
        raise RuntimeError("imported package binding set changed")
    bindings: dict[str, str] = {}
    for name, module in sorted(modules.items()):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"imported package {name!r} has no concrete file")
        path = Path(module_file).resolve(strict=True)
        if not path.is_relative_to(import_roots[name]):
            raise RuntimeError(
                f"imported package {name!r} escaped its authenticated root: {path}"
            )
        bindings[name] = str(path)
    return bindings


def finalize_authenticated_imports(
    cache_root: Path, previous: tuple[bool, str | None]
) -> dict[str, Any]:
    """Prove the private bytecode prefix stayed empty and restore interpreter state."""

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


def verify_checkpoint(
    model_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    int,
    str,
    list[int],
    dict[str, tuple[int, int, int, int, int]],
]:
    """Authenticate every local artifact, then validate control-file semantics."""

    artifacts: list[dict[str, Any]] = []
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    control_payloads: dict[str, bytes] = {}
    for relative_path, expected_hash in sorted(
        (EXPECTED_CONTROL_HASHES | EXPECTED_SHARD_HASHES).items()
    ):
        path = model_dir / relative_path
        retain_payload = relative_path in EXPECTED_CONTROL_HASHES
        observed, identity, payload = authenticated_file(
            path,
            expected_sha256=expected_hash,
            retain_payload=retain_payload,
        )
        artifacts.append({"relative_path": relative_path, **observed})
        identities[relative_path] = identity
        if retain_payload:
            assert payload is not None
            control_payloads[relative_path] = payload

    config = json_payload_without_duplicate_keys(
        control_payloads["config.json"], "config.json"
    )
    expected_fields = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2_048,
        "max_position_embeddings": 40_960,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 768,
        "norm_topk_prob": True,
        "num_attention_heads": 32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": VOCAB_SIZE,
    }
    for field, expected in expected_fields.items():
        if config.get(field) != expected:
            raise ValueError(
                f"checkpoint config {field!r} changed: expected {expected!r}, "
                f"found {config.get(field)!r}"
            )

    generation = json_payload_without_duplicate_keys(
        control_payloads["generation_config.json"], "generation_config.json"
    )
    eos_token_ids = validated_generation_eos_token_ids(generation)
    index = json_payload_without_duplicate_keys(
        control_payloads["model.safetensors.index.json"],
        "model.safetensors.index.json",
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != EXPECTED_PHYSICAL_TENSORS:
        raise ValueError(
            "checkpoint index physical tensor count changed: "
            f"expected {EXPECTED_PHYSICAL_TENSORS}, found "
            f"{len(weight_map) if isinstance(weight_map, dict) else type(weight_map)!r}"
        )
    shard_names = set(weight_map.values())
    if shard_names != set(EXPECTED_SHARD_HASHES):
        raise ValueError(f"checkpoint shard set changed: {sorted(shard_names)}")
    if any(not isinstance(name, str) or Path(name).name != name for name in shard_names):
        raise ValueError("checkpoint index contains a non-local shard path")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint index metadata is not an object")
    tensor_bytes = COMMON.checked_json_nonnegative_integer(
        metadata.get("total_size"), "checkpoint index metadata.total_size"
    )
    if tensor_bytes != CHECKPOINT_TENSOR_BYTES:
        raise ValueError(
            f"checkpoint tensor bytes changed: expected {CHECKPOINT_TENSOR_BYTES}, "
            f"found {tensor_bytes}"
        )

    return (
        artifacts,
        tensor_bytes,
        COMMON.canonical_artifact_digest(artifacts),
        eos_token_ids,
        identities,
    )


def ensure_external_paths(model_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve safe external input/output roots and require an absent final bundle."""

    model = model_dir.expanduser().resolve(strict=True)
    output = output_dir.expanduser().resolve(strict=True)
    crate = Path(__file__).resolve().parents[1]
    if not model.is_dir() or not output.is_dir():
        raise ValueError("model and output paths must be existing directories")
    if output == crate or output.is_relative_to(crate):
        raise ValueError(f"output directory must be external to the crate: {output}")
    if output == model or output.is_relative_to(model):
        raise ValueError(f"output directory must be outside the checkpoint: {output}")
    bundle = output / BUNDLE_DIRNAME
    if COMMON.path_lexists(bundle):
        raise FileExistsError(f"create-only reference bundle already exists: {bundle}")
    return model, output, bundle


def publish_reference_bundle(output_dir: Path, payloads: Mapping[str, bytes]) -> Path:
    """Durably publish the complete four-file bundle with no replacement."""

    expected_names = {
        PROMPT_LOGITS_RAW_FILENAME,
        CACHED_LOGITS_RAW_FILENAME,
        TEACHER_LOGITS_RAW_FILENAME,
        JSON_FILENAME,
    }
    if set(payloads) != expected_names:
        raise ValueError(
            f"reference bundle payload set changed: expected {sorted(expected_names)}, "
            f"found {sorted(payloads)}"
        )
    bundle = output_dir / BUNDLE_DIRNAME
    if COMMON.path_lexists(bundle):
        raise FileExistsError(f"create-only reference bundle already exists: {bundle}")
    staging = Path(tempfile.mkdtemp(prefix=f".{BUNDLE_DIRNAME}.staging-", dir=output_dir))
    try:
        for filename in sorted(payloads):
            COMMON.write_create_only(staging / filename, payloads[filename])
        os.chmod(staging, 0o755)
        COMMON.fsync_directory(staging)
        COMMON.rename_directory_create_only(staging, bundle)
        COMMON.fsync_directory(output_dir)
    except BaseException as error:
        error.add_note(f"recoverable reference staging directory: {staging}")
        raise
    return bundle


def load_with_verified_mps_allocator_warmup_noop(
    modeling_utils: Any,
    torch: Any,
    expected_model_class: type,
    loader: Any,
) -> tuple[Any, dict[str, Any]]:
    """Disable only the oversized optional MPS allocator warmup during loading."""

    if getattr(modeling_utils, "__name__", None) != "transformers.modeling_utils":
        raise RuntimeError("allocator warmup owner changed")
    original = getattr(modeling_utils, "caching_allocator_warmup", None)
    if (
        not callable(original)
        or getattr(original, "__module__", None) != "transformers.modeling_utils"
        or getattr(original, "__name__", None) != "caching_allocator_warmup"
    ):
        raise RuntimeError("Transformers allocator warmup callable identity changed")

    calls = 0
    expanded_entries = 0

    def verified_mps_noop(model: Any, expanded_device_map: Any, hf_quantizer: Any) -> None:
        """Check the intercepted call is exactly the safe unquantized MPS case."""

        nonlocal calls, expanded_entries
        calls += 1
        if calls > EXPECTED_ALLOCATOR_WARMUP_CALLS:
            raise RuntimeError("allocator warmup was invoked more than once")
        if type(model) is not expected_model_class or hf_quantizer is not None:
            raise RuntimeError("allocator warmup received an unexpected model or quantizer")
        if not isinstance(expanded_device_map, dict) or not expanded_device_map:
            raise RuntimeError("allocator warmup requires a nonempty device map")
        device_types = {torch.device(value).type for value in expanded_device_map.values()}
        if device_types != {"mps"}:
            raise RuntimeError(
                f"allocator warmup no-op requires only MPS, found {sorted(device_types)}"
            )
        expanded_entries = len(expanded_device_map)

    setattr(modeling_utils, "caching_allocator_warmup", verified_mps_noop)
    remained_scoped = False
    try:
        loaded = loader()
        remained_scoped = modeling_utils.caching_allocator_warmup is verified_mps_noop
    finally:
        setattr(modeling_utils, "caching_allocator_warmup", original)
    if not remained_scoped or modeling_utils.caching_allocator_warmup is not original:
        raise RuntimeError("allocator warmup override escaped its intended scope")
    if calls != EXPECTED_ALLOCATOR_WARMUP_CALLS:
        raise RuntimeError(
            f"expected {EXPECTED_ALLOCATOR_WARMUP_CALLS} allocator call, found {calls}"
        )
    return loaded, {
        "target": ALLOCATOR_WARMUP_TARGET,
        "policy": ALLOCATOR_WARMUP_POLICY,
        "scope": ALLOCATOR_WARMUP_SCOPE,
        "reason": ALLOCATOR_WARMUP_REASON,
        "call_count": calls,
        "expanded_device_map_entries": expanded_entries,
        "expanded_device_types": ["mps"],
        "hf_quantizer_present": False,
        "restored": True,
    }


def loading_info_summary(loading_info: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and require empty official loading diagnostics."""

    result: dict[str, Any] = {}
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = sorted(str(value) for value in loading_info.get(name, []))
        result[name] = {
            "count": len(values),
            "sha256": COMMON.sha256_bytes("\n".join(values).encode("utf-8")),
        }
        if values:
            raise RuntimeError(f"official reference load reported {name}: {values[:10]}")
    return result


def git_provenance(script_path: Path) -> dict[str, Any]:
    """Bind the exact early source snapshots to their containing Git state."""

    repository = script_path.parents[3]

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    snapshots = (
        (script_path, _SELF_SOURCE_BYTES, _SELF_SOURCE_IDENTITY),
        (COMMON_HELPER_PATH, _COMMON_SOURCE_BYTES, _COMMON_SOURCE_IDENTITY),
    )
    records = []
    for path, payload, identity in snapshots:
        assert_early_source_unchanged(path, payload, identity)
        relative = path.relative_to(script_path.parents[1]).as_posix()
        records.append(
            {
                "relative_path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "git_dirty": bool(git("status", "--porcelain", "--", str(path))),
            }
        )
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "files": records,
    }


def tensor_cosine(torch: Any, left: Any, right: Any) -> float:
    """Compute full-vector cosine with F64 CPU accumulation."""

    if not bool(torch.isfinite(left).all().item()) or not bool(
        torch.isfinite(right).all().item()
    ):
        raise ValueError("cosine inputs contain a non-finite value")
    left64 = left.to(dtype=torch.float64)
    right64 = right.to(dtype=torch.float64)
    denominator = torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64)
    denominator_value = float(denominator.item())
    if not math.isfinite(denominator_value) or denominator_value == 0.0:
        raise ValueError("cosine denominator is zero or non-finite")
    result = float(torch.dot(left64, right64).item()) / denominator_value
    if not math.isfinite(result):
        raise ValueError("cosine result is non-finite")
    return result


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately local-only two-path command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate the atomic Qwen3-30B-A3B MPS/F16 official reference "
            "bundle with 32 full-vocabulary teacher-prefix rows."
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
        help=f"existing external parent for create-only {BUNDLE_DIRNAME}",
    )
    return parser.parse_args(argv)


def synchronize(torch: Any) -> None:
    """Wait for all queued MPS work before a timing boundary."""

    torch.mps.synchronize()


def device_details(torch: Any) -> dict[str, Any]:
    """Require and describe the Apple MPS acceptance device."""

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("the Qwen3-MoE F16 reference requires an available MPS backend")
    return {
        "kind": "mps",
        "built": True,
        "available": True,
        "machine": platform.machine(),
    }


def checked_f32_payload(tensor: Any, expected_elements: int, label: str) -> bytes:
    """Serialize one finite contiguous CPU F32 tensor into canonical LE bytes."""

    if tensor.device.type != "cpu" or str(tensor.dtype) != "torch.float32":
        raise RuntimeError(f"{label} must be a CPU float32 tensor")
    if not tensor.is_contiguous() or tensor.numel() != expected_elements:
        raise RuntimeError(
            f"{label} shape/storage changed: elements={tensor.numel()}, "
            f"contiguous={tensor.is_contiguous()}"
        )
    if not bool(tensor.isfinite().all().item()):
        raise RuntimeError(f"{label} contains a non-finite value")
    payload = COMMON.f32le_bytes(float(value) for value in tensor.reshape(-1).tolist())
    if len(payload) != expected_elements * 4:
        raise RuntimeError(f"{label} encoded to an unexpected byte length")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pinned official graph and publish its immutable evidence bundle."""

    arguments = parse_arguments(argv)
    model_dir, output_dir, expected_bundle = ensure_external_paths(
        arguments.model_dir, arguments.output_dir
    )

    # These checks run before Torch/Transformers imports and before any model
    # allocation.  Even a wrong environment or existing output therefore fails
    # cheaply and cannot initiate a multi-gigabyte load.
    package_versions, distributions, distribution_digest = COMMON.verify_packages()
    distribution_payloads, distribution_payload_identities = (
        verify_distribution_payloads()
    )
    for name, payload in distribution_payloads.items():
        distributions[name]["payload"] = payload
    absent_distributions = verify_absent_optional_distributions()
    import_roots = distribution_import_roots()
    transformers_root = transformers_package_root()
    if import_roots["transformers"] != transformers_root:
        raise RuntimeError("Transformers import root differs from authenticated package root")
    reference_sources, reference_source_identities = verify_reference_sources(
        transformers_root
    )
    (
        artifacts,
        tensor_bytes,
        artifact_digest,
        eos_token_ids,
        checkpoint_identities,
    ) = verify_checkpoint(model_dir)

    bytecode_cache, import_resolutions, previous_bytecode_state = (
        prepare_authenticated_imports(output_dir, import_roots)
    )
    import accelerate
    import numpy
    import safetensors
    import tokenizers
    import torch
    import transformers
    from transformers import (
        AutoTokenizer,
        DynamicCache,
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
    )
    from transformers import modeling_utils
    from transformers.models.qwen3_moe import (
        configuration_qwen3_moe,
        modeling_qwen3_moe,
    )

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
    del accelerate, numpy, safetensors, tokenizers
    if Path(transformers.__file__).resolve(strict=True).parent != transformers_root:
        raise RuntimeError("imported Transformers package differs from preflighted package")
    imported_sources = {
        configuration_qwen3_moe.__name__: Path(configuration_qwen3_moe.__file__).resolve(
            strict=True
        ),
        modeling_qwen3_moe.__name__: Path(modeling_qwen3_moe.__file__).resolve(strict=True),
    }
    expected_imports = {
        module: (transformers_root / value["relative_path"]).resolve(strict=True)
        for module, value in EXPECTED_REFERENCE_SOURCES.items()
        if ".models.qwen3_moe." in module
    }
    if imported_sources != expected_imports:
        raise RuntimeError(
            f"imported Qwen3-MoE sources changed: {imported_sources} != {expected_imports}"
        )

    details = device_details(torch)
    recommended_mps_memory = int(torch.mps.recommended_max_memory())
    required_mps_memory = CHECKPOINT_TENSOR_BYTES + MINIMUM_MPS_LOAD_HEADROOM_BYTES
    if recommended_mps_memory < required_mps_memory:
        raise RuntimeError(
            "insufficient recommended MPS memory for the pinned F16 model: "
            f"recommended={recommended_mps_memory}, required={required_mps_memory}"
        )
    details["recommended_max_memory_bytes"] = recommended_mps_memory
    details["required_f16_load_bytes"] = required_mps_memory
    details["minimum_load_headroom_bytes"] = MINIMUM_MPS_LOAD_HEADROOM_BYTES
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    torch.mps.set_per_process_memory_fraction(1.0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_grad_enabled(False)

    config = Qwen3MoeConfig.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        revision=REVISION,
        attn_implementation="eager",
        experts_implementation="eager",
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"

    load_started = time.perf_counter()
    (model, loading_info), allocator_override = load_with_verified_mps_allocator_warmup_noop(
        modeling_utils,
        torch,
        Qwen3MoeForCausalLM,
        lambda: Qwen3MoeForCausalLM.from_pretrained(
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
            output_loading_info=True,
        ),
    )
    synchronize(torch)
    load_seconds = COMMON.checked_duration(
        "model load", time.perf_counter() - load_started
    )
    if type(model) is not Qwen3MoeForCausalLM:
        raise RuntimeError(f"loaded unexpected model class {type(model)!r}")
    if model.config._attn_implementation != "eager":
        raise RuntimeError("loaded model did not retain eager attention")
    if model.config._experts_implementation != "eager":
        raise RuntimeError("loaded model did not retain eager expert dispatch")
    if getattr(model, "use_kernels", None) is not False:
        raise RuntimeError("loaded model unexpectedly enabled external Hub kernels")
    if (
        allocator_override["expanded_device_map_entries"]
        != EXPECTED_EXPANDED_DEVICE_MAP_ENTRIES
    ):
        raise RuntimeError(
            "converted parameter-device map changed: expected "
            f"{EXPECTED_EXPANDED_DEVICE_MAP_ENTRIES}, found "
            f"{allocator_override['expanded_device_map_entries']}"
        )
    load_diagnostics = loading_info_summary(loading_info)
    model.eval()
    torch.mps.empty_cache()
    loaded_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if loaded_parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"loaded parameter count changed: expected {EXPECTED_PARAMETER_COUNT}, "
            f"found {loaded_parameter_count}"
        )
    parameter_dtypes: dict[str, int] = {}
    for parameter in model.parameters():
        dtype = str(parameter.dtype).removeprefix("torch.")
        parameter_dtypes[dtype] = parameter_dtypes.get(dtype, 0) + parameter.numel()
    if parameter_dtypes != {"float16": EXPECTED_PARAMETER_COUNT}:
        raise RuntimeError(f"loaded parameter dtypes changed: {parameter_dtypes}")

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
    if tokenizer.eos_token_id != 151_645 or tokenizer.pad_token_id != 151_643:
        raise RuntimeError(
            "pinned tokenizer EOS/PAD IDs changed: "
            f"eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
        )
    prompt_ids = torch.tensor([INPUT_IDS], dtype=torch.long, device="mps")
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long, device="mps")

    with torch.inference_mode():
        stateless_started = time.perf_counter()
        stateless = model(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        synchronize(torch)
        stateless_seconds = COMMON.checked_duration(
            "stateless prompt", time.perf_counter() - stateless_started
        )
        if stateless.logits.dtype != torch.float16:
            raise RuntimeError(f"stateless logits use {stateless.logits.dtype}, expected F16")
        stateless_row = stateless.logits[0, -1].detach().float().cpu().contiguous()

        generation_started = time.perf_counter()
        cached = model(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        cache = cached.past_key_values
        if not isinstance(cache, DynamicCache):
            raise RuntimeError(
                f"cached prompt returned {type(cache)!r}, expected DynamicCache"
            )
        if cache.get_seq_length() != len(INPUT_IDS):
            raise RuntimeError(
                f"prefill cache length is {cache.get_seq_length()}, "
                f"expected {len(INPUT_IDS)}"
            )
        cached_rows = []
        generated_ids: list[int] = []
        decisions: list[dict[str, Any]] = []
        for step in range(GREEDY_TOKENS):
            if cached.logits.dtype != torch.float16:
                raise RuntimeError(f"cached logits at step {step} are not F16")
            row = cached.logits[0, -1].detach().float().cpu().contiguous()
            if row.numel() != VOCAB_SIZE or not bool(torch.isfinite(row).all().item()):
                raise RuntimeError(f"cached vocabulary row {step} is invalid")
            cached_rows.append(row)
            values, indices = torch.topk(row, k=2, largest=True, sorted=True)
            decision = greedy_decision_stability(
                step=step,
                top1_token_id=int(indices[0].item()),
                top2_token_id=int(indices[1].item()),
                top1_logit=float(values[0].item()),
                top2_logit=float(values[1].item()),
                eos_token_ids=eos_token_ids,
            )
            decisions.append(decision)
            token_id = int(decision["top1_token_id"])
            generated_ids.append(token_id)
            if step + 1 < GREEDY_TOKENS:
                cached = model(
                    input_ids=torch.tensor([[token_id]], dtype=torch.long, device="mps"),
                    attention_mask=torch.ones(
                        (1, len(INPUT_IDS) + step + 1),
                        dtype=torch.long,
                        device="mps",
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                cache = cached.past_key_values
                if not isinstance(cache, DynamicCache):
                    raise RuntimeError(
                        f"cached decode returned {type(cache)!r} at step {step}"
                    )
                expected_cache_length = len(INPUT_IDS) + step + 1
                if cache.get_seq_length() != expected_cache_length:
                    raise RuntimeError(
                        f"cache length at step {step} is {cache.get_seq_length()}, "
                        f"expected {expected_cache_length}"
                    )
        synchronize(torch)
        generation_seconds = COMMON.checked_duration(
            "cached generation", time.perf_counter() - generation_started
        )

        minimum_margin, minimum_ratio = validate_decision_stability(decisions)
        teacher_input_ids = INPUT_IDS + generated_ids[:-1]
        teacher_started = time.perf_counter()
        teacher = model(
            input_ids=torch.tensor([teacher_input_ids], dtype=torch.long, device="mps"),
            attention_mask=torch.ones(
                (1, len(teacher_input_ids)), dtype=torch.long, device="mps"
            ),
            use_cache=False,
            logits_to_keep=0,
            return_dict=True,
        )
        synchronize(torch)
        teacher_seconds = COMMON.checked_duration(
            "teacher-forced forward", time.perf_counter() - teacher_started
        )
        if teacher.logits.dtype != torch.float16:
            raise RuntimeError(f"teacher logits use {teacher.logits.dtype}, expected F16")
        teacher_all = teacher.logits[0].detach().float().cpu().contiguous()

    if list(teacher_all.shape) != [len(teacher_input_ids), VOCAB_SIZE]:
        raise RuntimeError(f"teacher logits shape changed: {list(teacher_all.shape)}")
    if not bool(torch.isfinite(teacher_all).all().item()):
        raise RuntimeError("teacher logits contain a non-finite value")
    prediction_positions = list(
        range(len(INPUT_IDS) - 1, len(INPUT_IDS) - 1 + GREEDY_TOKENS)
    )
    teacher_rows = teacher_all[prediction_positions].contiguous()
    teacher_all_top1 = teacher_all.argmax(dim=-1).tolist()
    teacher_top1 = teacher_rows.argmax(dim=-1).tolist()
    if teacher_top1 != generated_ids:
        raise RuntimeError(
            f"cached/stateless teacher top-1 differs: {generated_ids} != {teacher_top1}"
        )

    per_prefix_cosine = [
        tensor_cosine(torch, cached_row, teacher_rows[index])
        for index, cached_row in enumerate(cached_rows)
    ]
    per_prefix_max_abs_error = [
        float((cached_row - teacher_rows[index]).abs().max().item())
        for index, cached_row in enumerate(cached_rows)
    ]
    minimum_cosine = min(per_prefix_cosine)
    if minimum_cosine < SELF_PARITY_COSINE:
        raise RuntimeError(
            f"official cached/stateless cosine {minimum_cosine} fell below {SELF_PARITY_COSINE}"
        )
    stateless_top1 = int(stateless_row.argmax().item())
    prompt_cached_cosine = tensor_cosine(torch, stateless_row, cached_rows[0])
    if stateless_top1 != generated_ids[0] or prompt_cached_cosine < SELF_PARITY_COSINE:
        raise RuntimeError("official stateless/cached prompt parity failed")

    prompt_payload = checked_f32_payload(stateless_row, VOCAB_SIZE, "prompt logits")
    cached_matrix = torch.stack(cached_rows, dim=0).contiguous()
    cached_payload = checked_f32_payload(
        cached_matrix, GREEDY_TOKENS * VOCAB_SIZE, "cached generation logits"
    )
    teacher_payload = checked_f32_payload(
        teacher_rows, GREEDY_TOKENS * VOCAB_SIZE, "teacher-prefix logits"
    )
    prompt_values = [float(value) for value in stateless_row.tolist()]
    prompt_summary = COMMON.vector_summary(
        prompt_values,
        list(dict.fromkeys([0, 1, 2, 100, 1_000, 10_000, 151_000, stateless_top1])),
    )
    prompt_summary.update(
        {
            "relative_path": PROMPT_LOGITS_RAW_FILENAME,
            "encoding": "ieee754-f32-little-endian",
            "shape": [VOCAB_SIZE],
            "elements": VOCAB_SIZE,
            "bytes": len(prompt_payload),
            "sha256": COMMON.sha256_bytes(prompt_payload),
            "top1_token_id": stateless_top1,
        }
    )
    teacher_summary = {
        "relative_path": TEACHER_LOGITS_RAW_FILENAME,
        "encoding": "ieee754-f32-little-endian",
        "shape": [GREEDY_TOKENS, VOCAB_SIZE],
        "elements": GREEDY_TOKENS * VOCAB_SIZE,
        "bytes": len(teacher_payload),
        "sha256": COMMON.sha256_bytes(teacher_payload),
        "row_top1_token_ids": teacher_top1,
    }
    cached_summary = {
        "relative_path": CACHED_LOGITS_RAW_FILENAME,
        "encoding": "ieee754-f32-little-endian",
        "shape": [GREEDY_TOKENS, VOCAB_SIZE],
        "elements": GREEDY_TOKENS * VOCAB_SIZE,
        "bytes": len(cached_payload),
        "sha256": COMMON.sha256_bytes(cached_payload),
        "row_top1_token_ids": generated_ids,
    }

    decoded_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    del model, cache, cached, stateless, teacher, teacher_all, teacher_rows, cached_rows
    del cached_matrix, stateless_row, tokenizer
    torch.mps.empty_cache()

    # Reauthenticate everything whose bytes or identity could have changed
    # between preflight and execution.  Publication is forbidden unless both
    # complete passes agree, so a replaced source or shard cannot bless output.
    post_packages, post_distributions, post_distribution_digest = COMMON.verify_packages()
    post_distribution_payloads, post_distribution_payload_identities = (
        verify_distribution_payloads()
    )
    for name, payload in post_distribution_payloads.items():
        post_distributions[name]["payload"] = payload
    post_import_roots = distribution_import_roots()
    post_reference_sources, post_reference_source_identities = (
        verify_reference_sources(transformers_root)
    )
    post_checkpoint = verify_checkpoint(model_dir)
    if (
        post_packages != package_versions
        or post_distributions != distributions
        or post_distribution_digest != distribution_digest
        or post_distribution_payload_identities != distribution_payload_identities
        or post_import_roots != import_roots
        or post_reference_sources != reference_sources
        or post_reference_source_identities != reference_source_identities
        or post_checkpoint
        != (
            artifacts,
            tensor_bytes,
            artifact_digest,
            eos_token_ids,
            checkpoint_identities,
        )
    ):
        raise RuntimeError("authenticated package/source/checkpoint state changed during run")
    bytecode_policy = finalize_authenticated_imports(
        bytecode_cache, previous_bytecode_state
    )

    script_path = Path(__file__).resolve(strict=True)
    record = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "reference_class": REFERENCE_CLASS,
        "reference_source_release": REFERENCE_SOURCE_RELEASE,
        "reference_source_tag": REFERENCE_SOURCE_TAG,
        "reference_source_tag_object": REFERENCE_SOURCE_TAG_OBJECT,
        "reference_source_commit": REFERENCE_SOURCE_COMMIT,
        "prompt": PROMPT,
        "input_ids": INPUT_IDS,
        "execution": {
            "dtype": "float16",
            "device": "mps",
            "attention_implementation": "eager",
            "experts_implementation": "eager",
            "deterministic_algorithms": True,
            "seed": 0,
            "use_cache": True,
            "float32_matmul_precision": "highest",
            "mps_per_process_memory_fraction": 1.0,
            "explicit_all_one_attention_masks": True,
            "hub_kernels_enabled": False,
            "integrity_threat_model": INTEGRITY_THREAT_MODEL,
            "authenticated_import_resolutions": import_resolutions,
            "imported_module_bindings": import_bindings,
            "authenticated_import_bytecode_policy": bytecode_policy,
            "allocator_warmup_override": allocator_override,
        },
        "environment": {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "command": [sys.executable, str(script_path), *sys.argv[1:]],
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
            },
            "packages": package_versions,
            "distribution_provenance": distributions,
            "distribution_manifest_sha256": distribution_digest,
            "absent_optional_distributions": absent_distributions,
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
                    "USE_HUB_KERNELS",
                )
            },
        },
        "provenance": {
            "generator": git_provenance(script_path),
            "reference_sources": reference_sources,
            "artifacts": artifacts,
            "artifact_manifest_sha256": artifact_digest,
            "checkpoint_tensor_bytes": tensor_bytes,
            "loaded_parameter_count": loaded_parameter_count,
            "parameter_dtype_elements": parameter_dtypes,
            "loading_info": load_diagnostics,
        },
        "prompt_last_logits": prompt_summary,
        "cached_generation_logits": cached_summary,
        "teacher_prefix_logits": teacher_summary,
        "generation": {
            "max_new_tokens": GREEDY_TOKENS,
            "token_ids": generated_ids,
            "margins": [float(item["margin"]) for item in decisions],
            "stability_policy": {
                "name": TOP1_STABILITY_POLICY,
                "elementwise_perturbation_ulps": BINARY16_ELEMENTWISE_PERTURBATION_ULPS,
                "safety_factor": TOP1_STABILITY_SAFETY_FACTOR,
                "rationale": TOP1_STABILITY_RATIONALE,
            },
            "decision_stability": decisions,
            "minimum_observed_margin": minimum_margin,
            "minimum_margin_to_epsilon_ratio": minimum_ratio,
            "decoded_text": decoded_text,
            "eos_token_ids": eos_token_ids,
            "first_eos_position": None,
            "completed_32_decisions": True,
        },
        "teacher_forced": {
            "input_ids": teacher_input_ids,
            "prediction_positions": prediction_positions,
            "all_position_top1_token_ids": teacher_all_top1,
            "continuation_top1_token_ids": teacher_top1,
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
    json_payload = COMMON.strict_json_bytes(record)
    published = publish_reference_bundle(
        output_dir,
        {
            PROMPT_LOGITS_RAW_FILENAME: prompt_payload,
            CACHED_LOGITS_RAW_FILENAME: cached_payload,
            TEACHER_LOGITS_RAW_FILENAME: teacher_payload,
            JSON_FILENAME: json_payload,
        },
    )
    if published != expected_bundle:
        raise RuntimeError(f"published {published}, expected {expected_bundle}")
    print(
        f"wrote {published} (prompt_sha256={prompt_summary['sha256']}, "
        f"cached_sha256={cached_summary['sha256']}, "
        f"teacher_sha256={teacher_summary['sha256']}, json_bytes={len(json_payload)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
