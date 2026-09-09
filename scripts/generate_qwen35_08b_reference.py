#!/usr/bin/env python3
"""Generate a pinned Qwen3.5-0.8B official MPS/F16 reference bundle.

This is an offline, maintainer-run acceptance tool rather than part of the Rust
runtime. It loads only the immutable local checkpoint with Transformers'
``Qwen3_5ForCausalLM`` class and publishes full-vocabulary F32 evidence for the
prompt, all 32 cached greedy decisions, and the corresponding 32 stateless
teacher-forced positions. It never invokes a Hub API, remote code, sampling, or
Transformers' generation helpers.

The output parent must already exist outside both the repository and checkpoint.
Four files are fsynced in a hidden same-filesystem staging directory and exposed
by one atomic, no-replace directory rename. A failed run leaves its staging
directory for inspection and never exposes a partial final bundle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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


# Apply offline policy before importing even the authenticated shared helper.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["USE_HUB_KERNELS"] = "0"


def _bounded_source_snapshot(path: Path) -> bytes:
    """Read one small source through a non-following, identity-stable descriptor."""

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
        return payload
    finally:
        if descriptor >= 0:
            os.close(descriptor)


# Reuse audited encoding, package/source authentication, and no-replace helpers
# without changing the Qwen3.8 generator whose digest is already independently
# pinned by a different acceptance bundle.
COMMON_HELPER_PATH = Path(__file__).resolve().with_name("generate_qwen38_reference.py")
COMMON_HELPER_SHA256 = "e52b3699ae08920ca8f177a7fad0b68d807732d0fae524d6c943347357ed864b"
_COMMON_SOURCE = _bounded_source_snapshot(COMMON_HELPER_PATH)
if hashlib.sha256(_COMMON_SOURCE).hexdigest() != COMMON_HELPER_SHA256:
    raise RuntimeError(
        "shared Qwen reference helper changed; audit and update its pinned SHA-256"
    )
COMMON = types.ModuleType("_pinned_generate_qwen38_reference")
COMMON.__file__ = str(COMMON_HELPER_PATH)
COMMON.__package__ = ""
exec(
    compile(_COMMON_SOURCE, str(COMMON_HELPER_PATH), "exec", dont_inherit=True),
    COMMON.__dict__,
)


REPOSITORY = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
REFERENCE_CLASS = "transformers.Qwen3_5ForCausalLM"
# Selected before any Burn comparison: fixed candidate 8 was the first to pass
# all 32 >2-ULP margins and exact official cached/teacher decisions. Preserve
# the older Hello fixture as a near-tie diagnostic, not a stable release oracle.
# Selection report: 152690 bytes, SHA-256
# 015dad3595c817252f3e776b148f1af34105de2351d70237a004091260be31cb.
USER_PROMPT = "Explain why the seasons change in four complete sentences."
PROMPT = (
    "<|im_start|>user\n" + USER_PROMPT
    + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
INPUT_IDS = [
    248045, 846, 198, 814, 20139, 3069, 279, 15127, 2222, 303, 2943,
    4434, 22157, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
]
VOCAB_SIZE = 248_320
GREEDY_TOKENS = 32
EXPECTED_EOS_TOKEN_IDS = [248_044]
EXPECTED_PARAMETER_COUNT = 752_393_024
EXPECTED_PHYSICAL_TENSORS = 488
EXPECTED_TENSOR_BYTES = 1_746_882_752
EXPECTED_GREEDY_TOKEN_IDS = [
    760, 8964, 579, 7772, 369, 82951, 506, 449, 8937, 314, 12805, 220,
    17, 18, 13, 20, 11995, 8410, 310, 1141, 62409, 10709, 11, 13969,
    279, 11247, 310, 381, 11784, 310, 279, 7806,
]
SELF_PARITY_COSINE = 0.999

PROMPT_LOGITS_FILENAME = "qwen3.5-0.8b-2fc06364-prompt-logits.f32le"
CACHED_LOGITS_FILENAME = "qwen3.5-0.8b-2fc06364-cached-logits-32x248320.f32le"
TEACHER_LOGITS_FILENAME = "qwen3.5-0.8b-2fc06364-teacher-logits-32x248320.f32le"
JSON_FILENAME = "qwen3.5-0.8b-2fc06364-reference.json"
BUNDLE_DIRNAME = "qwen3.5-0.8b-2fc06364-reference.bundle"

EXPECTED_ARTIFACTS = {
    "chat_template.jinja": (
        7_755,
        "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    ),
    "config.json": (
        2_907,
        "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    ),
    "model.safetensors-00001-of-00001.safetensors": (
        1_746_942_600,
        "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    ),
    "model.safetensors.index.json": (
        50_900,
        "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    ),
    "tokenizer.json": (
        12_807_982,
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    "tokenizer_config.json": (
        16_709,
        "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    ),
}

INTEGRITY_THREAT_MODEL = (
    "Non-following descriptors, exact lengths, SHA-256, and complete pre/post passes "
    "detect persistent changes and ordinary races. A malicious same-UID process able "
    "to restore paths between observations is outside scope because it can also alter "
    "the running interpreter."
)


def descriptor_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return the descriptor fields compared across one complete file read."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_verified_file(
    path: Path, expected_bytes: int, expected_sha256: str, retain: bool
) -> tuple[dict[str, Any], tuple[int, int, int, int, int], bytes | None]:
    """Hash one exact regular file through a stable non-following descriptor."""

    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise RuntimeError(
                f"artifact {path} is not a regular {expected_bytes}-byte file"
            )
        digest = hashlib.sha256()
        retained = bytearray() if retain else None
        observed = 0
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                observed += len(chunk)
                if observed > expected_bytes:
                    raise RuntimeError(f"artifact grew while hashing: {path}")
                digest.update(chunk)
                if retained is not None:
                    retained.extend(chunk)
            after = os.fstat(handle.fileno())
        if observed != expected_bytes or descriptor_identity(before) != descriptor_identity(after):
            raise RuntimeError(f"artifact changed while hashing: {path}")
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, "
                f"found {observed_sha256}"
            )
        record = {
            "relative_path": path.name,
            "bytes": expected_bytes,
            "sha256": observed_sha256,
        }
        return record, descriptor_identity(after), bytes(retained) if retained is not None else None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def parse_json_bytes(payload: bytes, label: str) -> Any:
    """Parse UTF-8 JSON while rejecting duplicate object keys."""

    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}: {error}") from error


def canonical_artifact_digest(artifacts: Sequence[Mapping[str, Any]]) -> str:
    """Hash a sorted, length-prefixed checkpoint inventory."""

    canonical = bytearray(b"qwen35-08b-reference-artifacts-v1\0")
    for artifact in sorted(artifacts, key=lambda item: str(item["relative_path"])):
        for value in (
            str(artifact["relative_path"]).encode("utf-8"),
            str(artifact["bytes"]).encode("ascii"),
            str(artifact["sha256"]).encode("ascii"),
        ):
            canonical.extend(len(value).to_bytes(8, "little"))
            canonical.extend(value)
    return hashlib.sha256(canonical).hexdigest()


def verify_checkpoint(
    model_dir: Path,
) -> tuple[list[dict[str, Any]], str, list[tuple[int, int, int, int, int]]]:
    """Authenticate the complete pinned checkpoint and validate its topology."""

    artifacts: list[dict[str, Any]] = []
    identities: list[tuple[int, int, int, int, int]] = []
    retained: dict[str, bytes] = {}
    for name, (expected_bytes, expected_sha256) in sorted(EXPECTED_ARTIFACTS.items()):
        record, identity, payload = read_verified_file(
            model_dir / name,
            expected_bytes,
            expected_sha256,
            retain=expected_bytes <= 16 * 1024 * 1024,
        )
        artifacts.append(record)
        identities.append(identity)
        if payload is not None:
            retained[name] = payload

    config = parse_json_bytes(retained["config.json"], "config.json")
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_5":
        raise ValueError("outer config must declare model_type=qwen3_5")
    text = config.get("text_config")
    expected_text = {
        "model_type": "qwen3_5_text",
        "vocab_size": VOCAB_SIZE,
        "num_hidden_layers": 24,
        "mtp_num_hidden_layers": 1,
        "max_position_embeddings": 262_144,
        "tie_word_embeddings": True,
    }
    if not isinstance(text, dict):
        raise ValueError("outer config must contain an object-valued text_config")
    for name, expected in expected_text.items():
        if text.get(name) != expected:
            raise ValueError(
                f"text_config.{name} changed: expected {expected!r}, found {text.get(name)!r}"
            )
    eos = text.get("eos_token_id")
    eos_ids = [eos] if isinstance(eos, int) and not isinstance(eos, bool) else eos
    if eos_ids != EXPECTED_EOS_TOKEN_IDS:
        raise ValueError(f"text_config.eos_token_id changed: {eos!r}")
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != 24:
        raise ValueError("text_config.layer_types must contain exactly 24 entries")
    if layer_types.count("linear_attention") != 18 or layer_types.count("full_attention") != 6:
        raise ValueError("Qwen3.5-0.8B hybrid schedule changed")

    index = parse_json_bytes(
        retained["model.safetensors.index.json"], "model.safetensors.index.json"
    )
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("checkpoint index must contain an object-valued weight_map")
    weight_map = index["weight_map"]
    if len(weight_map) != EXPECTED_PHYSICAL_TENSORS:
        raise ValueError(f"checkpoint tensor count changed: {len(weight_map)}")
    shard_names = set(weight_map.values())
    expected_shard = "model.safetensors-00001-of-00001.safetensors"
    if shard_names != {expected_shard}:
        raise ValueError(f"checkpoint shard inventory changed: {sorted(shard_names)!r}")
    if index.get("metadata", {}).get("total_size") != EXPECTED_TENSOR_BYTES:
        raise ValueError("checkpoint index metadata.total_size changed")
    names = list(weight_map)
    counts = (
        sum(name.startswith("model.language_model.") for name in names),
        sum(name.startswith("model.visual.") for name in names),
        sum(name.startswith("mtp.") for name in names),
    )
    if counts != (320, 153, 15):
        raise ValueError(f"checkpoint namespace partition changed: {counts}")
    return artifacts, canonical_artifact_digest(artifacts), identities


def ensure_external_paths(model_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve local roots and require a fresh external bundle destination."""

    model_dir = model_dir.resolve(strict=True)
    output_dir = output_dir.resolve(strict=True)
    repository = Path(__file__).resolve(strict=True).parents[3]
    if not model_dir.is_dir() or not output_dir.is_dir():
        raise ValueError("model-dir and output-dir must be existing directories")
    bundle = output_dir / BUNDLE_DIRNAME
    if os.path.lexists(bundle):
        raise FileExistsError(f"create-only reference bundle already exists: {bundle}")
    for protected, label in ((model_dir, "checkpoint"), (repository, "repository")):
        if output_dir == protected or protected in output_dir.parents:
            raise ValueError(f"output-dir must be outside the {label}")
    if output_dir == model_dir or output_dir in model_dir.parents:
        raise ValueError("output-dir must not contain the checkpoint directory")
    return model_dir, output_dir, bundle


def publish_bundle(output_dir: Path, files: Mapping[str, bytes]) -> Path:
    """Publish a complete fixed-name bundle through a no-replace rename."""

    expected_names = {
        PROMPT_LOGITS_FILENAME,
        CACHED_LOGITS_FILENAME,
        TEACHER_LOGITS_FILENAME,
        JSON_FILENAME,
    }
    if set(files) != expected_names:
        raise ValueError(f"reference bundle file set changed: {sorted(files)!r}")
    destination = output_dir / BUNDLE_DIRNAME
    if os.path.lexists(destination):
        raise FileExistsError(f"create-only reference bundle already exists: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{BUNDLE_DIRNAME}.staging-", dir=output_dir))
    try:
        for name in sorted(files):
            if Path(name).name != name:
                raise ValueError(f"unsafe bundle filename: {name!r}")
            COMMON.write_create_only(staging / name, files[name])
        COMMON.fsync_directory(staging)
        COMMON.rename_directory_create_only(staging, destination)
        COMMON.fsync_directory(output_dir)
    except BaseException as error:
        error.add_note(f"recoverable reference staging directory: {staging}")
        raise
    return destination


def checked_duration(label: str, value: float) -> float:
    """Reject an invalid wall-clock measurement before serialization."""

    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"{label} duration is invalid: {value!r}")
    return value


def reference_diagnostics(
    generated_ids: Sequence[int],
    decisions: Sequence[Mapping[str, Any]],
    teacher_ids: Sequence[int],
    teacher_margins: Sequence[float],
    cached_teacher_cosines: Sequence[float],
    prompt_cached_cosine: float,
) -> dict[str, Any]:
    """Describe every decision before any acceptance check discards a failed run.

    This diagnostic record is printed to stderr only. It is not an accepted
    oracle, does not bypass a numerical gate, and never creates a bundle.
    """

    if any(
        len(values) != GREEDY_TOKENS
        for values in (
            generated_ids, decisions, teacher_ids, teacher_margins, cached_teacher_cosines
        )
    ):
        raise RuntimeError("diagnostic evidence must cover every greedy decision")
    return {
        "kind": "unaccepted-reference-diagnostics",
        "cached_token_ids": list(generated_ids),
        "teacher_token_ids": list(teacher_ids),
        "historical_mismatch_steps": [
            step for step, (actual, expected) in enumerate(
                zip(generated_ids, EXPECTED_GREEDY_TOKEN_IDS, strict=True)
            ) if actual != expected
        ],
        "cached_teacher_mismatch_steps": [
            step for step, (cached, teacher) in enumerate(
                zip(generated_ids, teacher_ids, strict=True)
            ) if cached != teacher
        ],
        "cached_decision_stability": list(decisions),
        "teacher_top_two_margins": list(teacher_margins),
        "unstable_cached_steps": [
            int(decision["step"]) for decision in decisions
            if not decision["stable_under_budget"]
        ],
        "minimum_cached_margin": min(float(decision["margin"]) for decision in decisions),
        "minimum_cached_margin_to_epsilon_ratio": min(
            float(decision["margin_to_epsilon_ratio"]) for decision in decisions
        ),
        "prompt_cached_cosine": prompt_cached_cosine,
        "cached_teacher_cosines": list(cached_teacher_cosines),
        "minimum_cached_teacher_cosine": min(cached_teacher_cosines),
    }


def checked_f32_payload(tensor: Any, expected_elements: int, label: str) -> bytes:
    """Serialize one finite contiguous CPU F32 tensor into canonical LE bytes."""

    if tensor.device.type != "cpu" or str(tensor.dtype) != "torch.float32":
        raise RuntimeError(f"{label} must be a CPU float32 tensor")
    if not tensor.is_contiguous() or tensor.numel() != expected_elements:
        raise RuntimeError(f"{label} has unexpected shape or storage")
    if not bool(tensor.isfinite().all().item()):
        raise RuntimeError(f"{label} contains a non-finite value")
    payload = COMMON.f32le_bytes(float(value) for value in tensor.reshape(-1).tolist())
    if len(payload) != expected_elements * 4:
        raise RuntimeError(f"{label} encoded to an unexpected byte length")
    return payload


def payload_record(filename: str, payload: bytes, shape: Sequence[int]) -> dict[str, Any]:
    """Describe one canonical external tensor payload."""

    elements = math.prod(shape)
    if len(payload) != elements * 4:
        raise ValueError(f"{filename} byte length does not match shape {list(shape)}")
    return {
        "relative_path": filename,
        "encoding": "ieee754-f32-little-endian",
        "shape": list(shape),
        "elements": elements,
        "bytes": len(payload),
        "sha256": COMMON.sha256_bytes(payload),
    }


def git_provenance(script_path: Path) -> dict[str, Any]:
    """Require clean generator sources and record their exact Git identity."""

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

    files = []
    for path, expected_payload in (
        (script_path, _bounded_source_snapshot(script_path)),
        (COMMON_HELPER_PATH, _COMMON_SOURCE),
    ):
        relative = path.relative_to(script_path.parents[1]).as_posix()
        if git("status", "--porcelain", "--", str(path)):
            raise RuntimeError(f"reference generator source must be committed and clean: {relative}")
        files.append(
            {
                "relative_path": relative,
                "bytes": len(expected_payload),
                "sha256": hashlib.sha256(expected_payload).hexdigest(),
            }
        )
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "files": files,
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately local-only two-path command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate the create-only Qwen3.5-0.8B MPS/F16 reference bundle "
            "with prompt, cached, and teacher-forced full-vocabulary logits."
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the official graph and atomically publish authenticated evidence."""

    arguments = parse_arguments(argv)
    model_dir, output_dir, expected_bundle = ensure_external_paths(
        arguments.model_dir, arguments.output_dir
    )
    script_path = Path(__file__).resolve(strict=True)
    generator = git_provenance(script_path)
    packages, distributions, distribution_digest = COMMON.verify_packages()
    artifacts, artifact_digest, artifact_identities = verify_checkpoint(model_dir)

    # Heavy imports and accelerator allocation happen only after all cheap,
    # create-only, package, source, and checkpoint preflight checks pass.
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import configuration_qwen3_5, modeling_qwen3_5

    transformers_root = Path(transformers.__file__).resolve(strict=True).parent
    reference_sources = COMMON.verify_reference_sources(transformers_root)
    imported_sources = {
        module.__name__: Path(module.__file__).resolve(strict=True)
        for module in (configuration_qwen3_5, modeling_qwen3_5)
    }
    expected_sources = {
        name: (transformers_root / record["relative_path"]).resolve(strict=True)
        for name, record in COMMON.EXPECTED_REFERENCE_SOURCES.items()
        if ".models.qwen3_5." in name
    }
    if imported_sources != expected_sources:
        raise RuntimeError("imported Qwen modules differ from pinned package sources")
    kernel_status = {
        "fast_path_available": bool(modeling_qwen3_5.is_fast_path_available),
        "causal_conv1d_kernel_available": modeling_qwen3_5.causal_conv1d_fn is not None,
        "chunk_gated_delta_kernel_available": modeling_qwen3_5.chunk_gated_delta_rule is not None,
        "recurrent_gated_delta_kernel_available": (
            modeling_qwen3_5.fused_recurrent_gated_delta_rule is not None
        ),
    }
    if any(kernel_status.values()):
        raise RuntimeError(f"official reference requires fallback kernels: {kernel_status}")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("the Qwen3.5-0.8B F16 reference requires Apple MPS")

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
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
    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
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
    )
    torch.mps.synchronize()
    load_seconds = checked_duration("model load", time.perf_counter() - load_started)
    if type(model) is not Qwen3_5ForCausalLM:
        raise RuntimeError(f"loaded unexpected model class {type(model)!r}")
    loading_summary: dict[str, dict[str, Any]] = {}
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = sorted(str(value) for value in loading_info.get(name, []))
        loading_summary[name] = {
            "count": len(values),
            "sha256": COMMON.sha256_bytes("\n".join(values).encode("utf-8")),
        }
        if values:
            raise RuntimeError(f"official text load reported {name}: {values[:10]}")
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_dtypes: dict[str, int] = {}
    for parameter in model.parameters():
        dtype = str(parameter.dtype).removeprefix("torch.")
        parameter_dtypes[dtype] = parameter_dtypes.get(dtype, 0) + parameter.numel()
    if parameter_count != EXPECTED_PARAMETER_COUNT or parameter_dtypes != {
        "float16": EXPECTED_PARAMETER_COUNT
    }:
        raise RuntimeError(
            f"loaded parameter inventory changed: count={parameter_count}, dtypes={parameter_dtypes}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        revision=REVISION,
        use_fast=True,
    )
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": USER_PROMPT}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    if rendered_prompt != PROMPT:
        raise RuntimeError("pinned non-thinking chat template rendering changed")
    observed_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    if observed_ids != INPUT_IDS:
        raise RuntimeError(f"pinned prompt tokenization changed: {observed_ids}")
    input_ids = torch.tensor([INPUT_IDS], dtype=torch.long, device="mps")
    prompt_mask = torch.ones_like(input_ids, dtype=torch.long, device="mps")

    with torch.inference_mode():
        prompt_started = time.perf_counter()
        stateless = model(
            input_ids=input_ids,
            attention_mask=prompt_mask,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        torch.mps.synchronize()
        prompt_seconds = checked_duration(
            "stateless prompt", time.perf_counter() - prompt_started
        )
        stateless_row = stateless.logits[0, -1].float().cpu().contiguous()

        cached_started = time.perf_counter()
        cached = model(
            input_ids=input_ids,
            attention_mask=prompt_mask,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        cache = cached.past_key_values
        if cache is None:
            raise RuntimeError("official cached prompt returned no cache")
        cached_rows = []
        generated_ids: list[int] = []
        decisions: list[dict[str, Any]] = []
        for step in range(GREEDY_TOKENS):
            if cached.logits.dtype != torch.float16:
                raise RuntimeError(f"cached logits dtype changed at step {step}")
            row = cached.logits[0, -1].float().cpu().contiguous()
            if row.numel() != VOCAB_SIZE or not bool(torch.isfinite(row).all().item()):
                raise RuntimeError(f"invalid cached logit row at step {step}")
            cached_rows.append(row)
            top_values, top_indices = torch.topk(row, k=2, largest=True, sorted=True)
            token = int(top_indices[0].item())
            decisions.append(
                COMMON.greedy_decision_stability(
                    step=step,
                    top1_token_id=token,
                    top2_token_id=int(top_indices[1].item()),
                    top1_logit=float(top_values[0].item()),
                    top2_logit=float(top_values[1].item()),
                    eos_token_ids=EXPECTED_EOS_TOKEN_IDS,
                )
            )
            generated_ids.append(token)
            if step + 1 < GREEDY_TOKENS:
                cached = model(
                    input_ids=torch.tensor([[token]], dtype=torch.long, device="mps"),
                    attention_mask=torch.ones(
                        (1, len(INPUT_IDS) + step + 1), dtype=torch.long, device="mps"
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                cache = cached.past_key_values
                if cache is None:
                    raise RuntimeError(f"cached decode returned no cache at step {step}")
        torch.mps.synchronize()
        cached_seconds = checked_duration(
            "cached generation", time.perf_counter() - cached_started
        )

        teacher_ids = INPUT_IDS + generated_ids[:-1]
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
            "teacher-forced forward", time.perf_counter() - teacher_started
        )
        teacher_rows = teacher.logits[0, len(INPUT_IDS) - 1:].float().cpu().contiguous()

    if stateless_row.numel() != VOCAB_SIZE or not bool(torch.isfinite(stateless_row).all()):
        raise RuntimeError("invalid stateless prompt row")
    if list(teacher_rows.shape) != [GREEDY_TOKENS, VOCAB_SIZE]:
        raise RuntimeError(f"teacher logit matrix changed shape: {list(teacher_rows.shape)}")
    if not bool(torch.isfinite(teacher_rows).all().item()):
        raise RuntimeError("teacher logit matrix contains non-finite values")
    teacher_top1 = [int(value) for value in teacher_rows.argmax(dim=-1).tolist()]
    cached_teacher_cosines = [
        COMMON.tensor_cosine(torch, cached_row, teacher_rows[index])
        for index, cached_row in enumerate(cached_rows)
    ]
    prompt_cached_cosine = COMMON.tensor_cosine(torch, stateless_row, cached_rows[0])
    teacher_top2 = torch.topk(teacher_rows, k=2, dim=-1, largest=True, sorted=True).values
    teacher_margins = [float(value) for value in (teacher_top2[:, 0] - teacher_top2[:, 1]).tolist()]
    diagnostics = reference_diagnostics(
        generated_ids, decisions, teacher_top1, teacher_margins,
        cached_teacher_cosines, prompt_cached_cosine,
    )
    print(json.dumps(diagnostics, sort_keys=True, allow_nan=False), file=sys.stderr, flush=True)
    if generated_ids != EXPECTED_GREEDY_TOKEN_IDS:
        raise RuntimeError(f"official greedy continuation changed: {generated_ids}")
    if teacher_top1 != generated_ids:
        raise RuntimeError("official cached and teacher-forced top-1 rows differ")
    minimum_margin, minimum_margin_to_epsilon = COMMON.validate_decision_stability(decisions)
    if min(cached_teacher_cosines) < SELF_PARITY_COSINE or prompt_cached_cosine < SELF_PARITY_COSINE:
        raise RuntimeError("official cached/stateless full-vector parity fell below 0.999")

    prompt_payload = checked_f32_payload(stateless_row, VOCAB_SIZE, "prompt logits")
    cached_matrix = torch.stack(cached_rows, dim=0).contiguous()
    cached_payload = checked_f32_payload(
        cached_matrix, GREEDY_TOKENS * VOCAB_SIZE, "cached logits"
    )
    teacher_payload = checked_f32_payload(
        teacher_rows, GREEDY_TOKENS * VOCAB_SIZE, "teacher logits"
    )
    payloads = {
        "prompt": payload_record(PROMPT_LOGITS_FILENAME, prompt_payload, [VOCAB_SIZE]),
        "cached": payload_record(
            CACHED_LOGITS_FILENAME, cached_payload, [GREEDY_TOKENS, VOCAB_SIZE]
        ),
        "teacher": payload_record(
            TEACHER_LOGITS_FILENAME, teacher_payload, [GREEDY_TOKENS, VOCAB_SIZE]
        ),
    }

    # Repeat every external authentication pass before publication.
    post_packages, post_distributions, post_distribution_digest = COMMON.verify_packages()
    post_sources = COMMON.verify_reference_sources(transformers_root)
    post_artifacts, post_artifact_digest, post_identities = verify_checkpoint(model_dir)
    if (
        post_packages != packages
        or post_distributions != distributions
        or post_distribution_digest != distribution_digest
        or post_sources != reference_sources
        or post_artifacts != artifacts
        or post_artifact_digest != artifact_digest
        or post_identities != artifact_identities
    ):
        raise RuntimeError("authenticated package/source/checkpoint state changed during run")
    if git_provenance(script_path) != generator:
        raise RuntimeError("generator Git identity changed during run")

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
            "hub_kernels_enabled": False,
            "kernel_status": kernel_status,
            "integrity_threat_model": INTEGRITY_THREAT_MODEL,
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
            "packages": packages,
            "distribution_provenance": distributions,
            "distribution_manifest_sha256": distribution_digest,
            "device": {"kind": "mps", "built": True, "available": True},
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
            "generator": generator,
            "reference_sources": reference_sources,
            "artifacts": artifacts,
            "artifact_manifest_sha256": artifact_digest,
            "checkpoint_tensor_bytes": EXPECTED_TENSOR_BYTES,
            "loaded_parameter_count": parameter_count,
            "parameter_dtype_elements": parameter_dtypes,
            "loading_info": loading_summary,
        },
        "payloads": payloads,
        "generation": {
            "max_new_tokens": GREEDY_TOKENS,
            "token_ids": generated_ids,
            "eos_token_ids": EXPECTED_EOS_TOKEN_IDS,
            "completed_without_eos": not any(
                token in EXPECTED_EOS_TOKEN_IDS for token in generated_ids
            ),
            "decision_stability": decisions,
            "minimum_observed_margin": minimum_margin,
            "minimum_margin_to_epsilon_ratio": minimum_margin_to_epsilon,
            "decoded_text": tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        },
        "self_parity": {
            "gate": SELF_PARITY_COSINE,
            "prompt_cached_cosine": prompt_cached_cosine,
            "cached_teacher_cosines": cached_teacher_cosines,
            "minimum_cached_teacher_cosine": min(cached_teacher_cosines),
            "cached_top1_token_ids": generated_ids,
            "teacher_top1_token_ids": teacher_top1,
            "top1_equal": True,
        },
        "timings_seconds": {
            "load": load_seconds,
            "stateless_prompt": prompt_seconds,
            "cached_generation_32": cached_seconds,
            "teacher_forced_all_logits": teacher_seconds,
        },
    }
    json_payload = COMMON.strict_json_bytes(record)
    bundle = publish_bundle(
        output_dir,
        {
            PROMPT_LOGITS_FILENAME: prompt_payload,
            CACHED_LOGITS_FILENAME: cached_payload,
            TEACHER_LOGITS_FILENAME: teacher_payload,
            JSON_FILENAME: json_payload,
        },
    )
    if bundle != expected_bundle:
        raise RuntimeError(f"published unexpected bundle path: {bundle}")
    for name, payload in (
        (PROMPT_LOGITS_FILENAME, prompt_payload),
        (CACHED_LOGITS_FILENAME, cached_payload),
        (TEACHER_LOGITS_FILENAME, teacher_payload),
        (JSON_FILENAME, json_payload),
    ):
        print(
            f"wrote {bundle / name} ({len(payload)} bytes, "
            f"sha256={COMMON.sha256_bytes(payload)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
