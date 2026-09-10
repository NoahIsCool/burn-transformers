#!/usr/bin/env python3
"""Verify the released Tuned Lens pair and prepare a small, fixed test fixture.

    python3 examples/mechanistic/tuned_lens/prepare.py
    python3 examples/mechanistic/tuned_lens/prepare.py --offline

Requires torch, safetensors, and tokenizers. Downloads about 781 MB once;
generated files stay in ignored target/mechanistic/tuned_lens/. This program
does not run a model, fit a lens, or execute downloaded Python code.
"""

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.request

import safetensors
from safetensors import safe_open
import tokenizers
import torch


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[2] / "target/mechanistic/tuned_lens"
SOURCE_NAMES = {
    "historical_model", "later_model", "config", "lens", "lens_config",
    "tokenizer", "tokenizer_config", "special_tokens_map", "pile_prefix",
}
OUTPUT_NAMES = {
    "model", "config", "lens", "fixture", "tensor_equivalence", "pile_records",
    "tokenizer", "tokenizer_config", "special_tokens_map",
}


def strict_json(data):
    """Reject duplicate keys and nonstandard NaN/Infinity values."""
    def object_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(data, object_pairs_hook=object_hook, parse_constant=invalid_constant)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path, root):
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_file(path, specification):
    if path.stat().st_size != specification["bytes"]:
        raise ValueError(f"Unexpected byte count: {path}")
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


@contextmanager
def atomic_output(path):
    """Publish only a complete output, removing temporary files on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".prepare-", delete=False) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with atomic_output(path) as handle:
        handle.write(data)


def validate_manifest(manifest):
    required = {
        "schema_version", "paper", "sources", "expected_model_config",
        "expected_lens_config", "protocol", "parity", "export",
    }
    if set(manifest) != required or manifest["schema_version"] != 1:
        raise ValueError("Unexpected manifest schema")
    if set(manifest["sources"]) != SOURCE_NAMES:
        raise ValueError("Unexpected source inventory")
    for name, specification in manifest["sources"].items():
        keys = {"url", "cache_filename", "bytes", "sha256"}
        if name == "pile_prefix":
            keys |= {"http_range", "full_file_bytes", "full_file_sha256"}
        if set(specification) != keys:
            raise ValueError(f"Unexpected source specification: {name}")
        if not specification["url"].startswith("https://huggingface.co/"):
            raise ValueError(f"Unexpected source host: {name}")
        if Path(specification["cache_filename"]).name != specification["cache_filename"]:
            raise ValueError(f"Unsafe cache filename: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", specification["sha256"]):
            raise ValueError(f"Invalid SHA-256: {name}")
        if type(specification["bytes"]) is not int or specification["bytes"] <= 0:
            raise ValueError(f"Invalid source length: {name}")
    protocol = manifest["protocol"]
    if (protocol["records"], protocol["max_input_tokens"], protocol["eos_token_id"]) != (4, 128, 0):
        raise ValueError("This fixed adaptation requires four records and at most 128 input tokens")


def obtain_source(root, name, specification, offline):
    path = root / "sources" / specification["cache_filename"]
    if path.exists():
        verify_file(path, specification)
        return path
    if offline:
        raise ValueError(f"Missing cached source {path}; rerun without --offline")
    print(f"Downloading {name}: {specification['bytes'] / 1_000_000:.2f} MB", flush=True)
    headers = {"Accept-Encoding": "identity"}
    if "http_range" in specification:
        headers["Range"] = specification["http_range"]
    request = urllib.request.Request(specification["url"], headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        if "http_range" in specification:
            expected = f"bytes 0-65535/{specification['full_file_bytes']}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Dataset server did not honor the exact bounded HTTP range")
        elif response.status != 200:
            raise ValueError(f"Unexpected HTTP status for {name}: {response.status}")
        digest = hashlib.sha256()
        count = 0
        with atomic_output(path) as handle:
            while True:
                chunk = response.read(min(8 * 1024 * 1024, specification["bytes"] - count + 1))
                if not chunk:
                    break
                count += len(chunk)
                if count > specification["bytes"]:
                    raise ValueError(f"Source exceeds pinned length: {name}")
                digest.update(chunk)
                handle.write(chunk)
            if count != specification["bytes"] or digest.hexdigest() != specification["sha256"]:
                raise ValueError(f"Downloaded source fails pinned size/hash: {name}")
    return path


def model_schema(config):
    """Return every historical parameter and persisted buffer, without guessing keys."""
    hidden, intermediate, vocab = (config[name] for name in ("hidden_size", "intermediate_size", "vocab_size"))
    result = {
        "gpt_neox.embed_in.weight": ([vocab, hidden], torch.float16, "parameter"),
        "embed_out.weight": ([vocab, hidden], torch.float16, "parameter"),
        "gpt_neox.final_layer_norm.weight": ([hidden], torch.float16, "parameter"),
        "gpt_neox.final_layer_norm.bias": ([hidden], torch.float16, "parameter"),
    }
    rotary_dim = int(hidden // config["num_attention_heads"] * config["rotary_pct"])
    for layer in range(config["num_hidden_layers"]):
        prefix = f"gpt_neox.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            for parameter in ("weight", "bias"):
                result[f"{prefix}.{norm}.{parameter}"] = ([hidden], torch.float16, "parameter")
        for projection, output, input_width in (
            ("attention.query_key_value", 3 * hidden, hidden),
            ("attention.dense", hidden, hidden),
            ("mlp.dense_h_to_4h", intermediate, hidden),
            ("mlp.dense_4h_to_h", hidden, intermediate),
        ):
            result[f"{prefix}.{projection}.weight"] = ([output, input_width], torch.float16, "parameter")
            result[f"{prefix}.{projection}.bias"] = ([output], torch.float16, "parameter")
        context = config["max_position_embeddings"]
        for suffix, shape, dtype in (
            ("bias", [1, 1, context, context], torch.uint8),
            ("masked_bias", [], torch.float16),
            ("rotary_emb.inv_freq", [rotary_dim // 2], torch.float16),
        ):
            result[f"{prefix}.attention.{suffix}"] = (shape, dtype, "derived_buffer")
    return result


def validate_state(state, schema, description):
    if type(state) not in (dict, OrderedDict) or set(state) != set(schema):
        raise ValueError(f"Unexpected tensor inventory in {description}")
    for name, (shape, dtype, role) in schema.items():
        tensor = state[name]
        if type(tensor) is not torch.Tensor:
            raise ValueError(f"Unexpected tensor type: {description}: {name}")
        if list(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"Unexpected dtype/shape: {description}: {name}")
        if tensor.device.type != "cpu" or tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError(f"Expected contiguous CPU storage: {description}: {name}")
        # The historical -1e9 masking sentinel was cast to F16, yielding -inf.
        # It is an excluded, unused legacy buffer, not a learned parameter.
        masked_sentinel = role == "derived_buffer" and name.endswith(".attention.masked_bias")
        if masked_sentinel and bool(torch.isneginf(tensor)):
            continue
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite parameter: {description}: {name}")


def tensor_bytes(tensor):
    """A read-only view for hashing/writing; no tensor-sized bytes copy."""
    return memoryview(tensor.detach().numpy()).cast("B")


def prove_equivalence(historical, later_path, schema):
    rows = {}
    with safe_open(later_path, framework="pt", device="cpu") as later:
        if set(later.keys()) != set(schema):
            raise ValueError("Historical and later SafeTensors key inventories differ")
        metadata = later.metadata()
        for name in sorted(schema):
            first, second = historical[name], later.get_tensor(name)
            if first.dtype != second.dtype or first.shape != second.shape:
                raise ValueError(f"Historical and later tensor metadata differ: {name}")
            # Compare raw storage, so signed zero and all bit patterns matter.
            if not torch.equal(first.reshape(-1).view(torch.uint8), second.reshape(-1).view(torch.uint8)):
                raise ValueError(f"Historical and later tensor bytes differ: {name}")
            first_hash = hashlib.sha256(tensor_bytes(first)).hexdigest()
            second_hash = hashlib.sha256(tensor_bytes(second)).hexdigest()
            if first_hash != second_hash:
                raise ValueError(f"Historical and later tensor hashes differ: {name}")
            rows[name] = {
                "shape": list(first.shape), "dtype": str(first.dtype).removeprefix("torch."),
                "bytes": first.numel() * first.element_size(), "role": schema[name][2],
                "historical_sha256": first_hash, "later_sha256": second_hash,
                "byte_equal": True,
            }
    return rows, metadata


def export_f32(path, tensors, metadata):
    """Stream one F32 tensor at a time into the simple SafeTensors wire format."""
    header = {"__metadata__": metadata}
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        end = offset + tensor.numel() * 4
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [offset, end]}
        offset = end
    encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    raw_hashes = {}
    with atomic_output(path) as handle:
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        for name in sorted(tensors):
            converted = tensors[name].detach().to(dtype=torch.float32).contiguous()
            data = tensor_bytes(converted)
            raw_hashes[name] = hashlib.sha256(data).hexdigest()
            handle.write(data)
            del data, converted
    # Parse our output using an independent format reader before publishing metadata.
    with safe_open(path, framework="pt", device="cpu") as output:
        if set(output.keys()) != set(tensors):
            raise ValueError(f"Incomplete F32 export: {path}")
        for name in output.keys():
            tensor = output.get_tensor(name)
            if tensor.dtype != torch.float32 or tensor.shape != tensors[name].shape:
                raise ValueError(f"Invalid F32 export shape/dtype: {name}")
    return raw_hashes


def prepare_fixture(root, prefix_path, tokenizer_path, protocol, vocab_size):
    prefix = prefix_path.read_bytes()
    complete = prefix[:prefix.rfind(b"\n") + 1]
    lines = complete.splitlines()
    if len(lines) != protocol["complete_source_records"]:
        raise ValueError("Pinned range no longer contains the expected 25 complete records")
    documents = [strict_json(line) for line in lines]
    for record in documents:
        if set(record) != {"text", "meta"} or not isinstance(record["text"], str) or not isinstance(record["meta"], dict):
            raise ValueError("Unexpected Pile JSON record schema")
    with atomic_output(root / "pile_test_first25.jsonl") as handle:
        handle.write(complete)
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    eos = protocol["eos_token_id"]
    if tokenizer.token_to_id("<|endoftext|>") != eos:
        raise ValueError("Historical tokenizer EOS does not match the protocol")
    records = []
    for index, document in enumerate(documents[:protocol["records"]]):
        ids = [eos] + tokenizer.encode(document["text"], add_special_tokens=False).ids
        ids = ids[:protocol["max_input_tokens"]]
        if len(ids) < 2 or not all(0 <= token < vocab_size for token in ids):
            raise ValueError(f"Invalid encoded evaluation record {index}")
        records.append({
            "record_index": index, "text": document["text"], "input_ids": ids,
            "input_tokens": len(ids), "ce_tokens": len(ids) - 1,
        })
    fixture = {
        "schema_version": 1,
        "selection": {
            "source_record_indices": list(range(protocol["records"])),
            "max_input_tokens": protocol["max_input_tokens"], "prepend_eos": True,
            "eos_token_id": eos, "add_special_tokens": False,
            "adaptation": protocol["adaptation"],
        },
        "records": records,
        "totals": {key: sum(record[key] for record in records) for key in ("input_tokens", "ce_tokens")},
    }
    write_json(root / "fixture.json", fixture)
    return fixture


def fingerprint(manifest_path):
    return {
        "manifest_sha256": sha256_file(manifest_path), "prepare_sha256": sha256_file(Path(__file__)),
        "torch": str(torch.__version__), "safetensors": safetensors.__version__,
        "tokenizers": tokenizers.__version__,
    }


def verified_output_cache(root, expected_fingerprint):
    path = root / "preparation.json"
    if not path.exists():
        return False
    metadata = strict_json(path.read_bytes())
    if metadata.get("schema_version") != 1 or metadata.get("fingerprint") != expected_fingerprint:
        return False
    if set(metadata.get("outputs", {})) != OUTPUT_NAMES:
        raise ValueError("Incomplete preparation cache; use --refresh")
    for specification in metadata["outputs"].values():
        path = (root / specification["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Preparation cache output escapes its root")
        verify_file(path, specification)
    if metadata.get("equivalence") != {"all_keys_equal": True, "tensor_count": 184}:
        raise ValueError("Missing full historical/SafeTensors equivalence proof")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--offline", action="store_true", help="require verified cached source files")
    parser.add_argument("--refresh", action="store_true", help="regenerate derived artifacts from verified sources")
    args = parser.parse_args()
    if sys.byteorder != "little":
        raise ValueError("This exact byte-preserving exporter requires a little-endian host")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    manifest_path = HERE / "manifest.json"
    manifest = strict_json(manifest_path.read_bytes())
    validate_manifest(manifest)
    sources = {
        name: obtain_source(root, name, specification, args.offline)
        for name, specification in manifest["sources"].items()
    }
    current_fingerprint = fingerprint(manifest_path)
    if not args.refresh and verified_output_cache(root, current_fingerprint):
        print(f"Verified cached sources, complete equivalence proof, and exports: {root}")
        return
    config = strict_json(sources["config"].read_bytes())
    lens_config = strict_json(sources["lens_config"].read_bytes())
    if config != manifest["expected_model_config"] or lens_config != manifest["expected_lens_config"]:
        raise ValueError("Pinned model/lens configuration does not match its complete schema")
    schema = model_schema(config)
    if len(schema) != 184:
        raise ValueError("Unexpected historical GPT-NeoX tensor count")
    print("Validating all historical tensors and proving later SafeTensors byte equivalence", flush=True)
    # Never substitute weights_only=False or execute a checkpoint's Python code.
    historical = torch.load(sources["historical_model"], map_location="cpu", weights_only=True, mmap=True)
    validate_state(historical, schema, "historical model")
    proof, safetensors_metadata = prove_equivalence(historical, sources["later_model"], schema)
    parameters = {name: tensor for name, tensor in historical.items() if schema[name][2] == "parameter"}
    if len(parameters) != manifest["export"]["model_parameter_count"]:
        raise ValueError("Unexpected learned parameter count")
    model_path = root / "model/model.safetensors"
    exported_hashes = export_f32(model_path, parameters, {
        "format": "pt", "source_sha256": manifest["sources"]["historical_model"]["sha256"],
        "conversion": "Exact F16-to-F32 cast of learned historical parameters; derived buffers explicitly excluded",
    })
    for name, digest in exported_hashes.items():
        proof[name]["export_dtype"] = "F32"
        proof[name]["export_raw_sha256"] = digest
    state_metadata = getattr(historical, "_metadata", None)
    del parameters, historical
    proof_output = {
        "schema_version": 1, "all_keys_equal": True, "tensor_count": len(proof),
        "historical_source": manifest["sources"]["historical_model"],
        "later_source": manifest["sources"]["later_model"],
        "historical_state_metadata": state_metadata, "later_safetensors_metadata": safetensors_metadata,
        "excluded_buffers": [name for name, item in proof.items() if item["role"] == "derived_buffer"],
        "buffer_policy": manifest["export"]["legacy_buffers"], "tensors": proof,
    }
    write_json(root / "tensor_equivalence.json", proof_output)
    lens = torch.load(sources["lens"], map_location="cpu", weights_only=True, mmap=True)
    lens_schema = {
        f"{layer}.{name}": (shape, torch.float32, "parameter")
        for layer in range(12) for name, shape in (("weight", [768, 768]), ("bias", [768]))
    }
    validate_state(lens, lens_schema, "released tuned lens")
    export_f32(root / "lens.safetensors", {f"translators.{name}": value for name, value in lens.items()}, {
        "format": "pt", "source_sha256": manifest["sources"]["lens"]["sha256"],
        "translator": "Residual correction: h + linear(h), then the model final LayerNorm and vocabulary projection",
    })
    del lens
    for name in ("config", "tokenizer", "tokenizer_config", "special_tokens_map"):
        with atomic_output(root / "model" / sources[name].name) as handle:
            with sources[name].open("rb") as source:
                shutil.copyfileobj(source, handle)
    fixture = prepare_fixture(root, sources["pile_prefix"], sources["tokenizer"], manifest["protocol"], config["vocab_size"])
    output_paths = {
        "model": "model/model.safetensors", "config": "model/config.json", "lens": "lens.safetensors",
        "fixture": "fixture.json", "tensor_equivalence": "tensor_equivalence.json",
        "pile_records": "pile_test_first25.jsonl", "tokenizer": "model/tokenizer.json",
        "tokenizer_config": "model/tokenizer_config.json", "special_tokens_map": "model/special_tokens_map.json",
    }
    metadata = {
        "schema_version": 1, "fingerprint": current_fingerprint,
        "sources": {name: dict(specification, path=str(sources[name].relative_to(root))) for name, specification in manifest["sources"].items()},
        "outputs": {name: describe_file(root / path, root) for name, path in output_paths.items()},
        "model_tensor_count": 148, "lens_tensor_count": 24,
        "equivalence": {"all_keys_equal": True, "tensor_count": 184},
        "protocol": manifest["protocol"], "parity": manifest["parity"],
        "fixture_totals": fixture["totals"],
    }
    # Metadata is published last: it cannot validate a half-written output set.
    write_json(root / "preparation.json", metadata)
    print(f"Proved equality of all 184 tensors; exported 148 model parameters and 24 learned lens parameters")
    print(f"Fixture: {fixture['totals']['input_tokens']} KL positions, {fixture['totals']['ce_tokens']} next-token CE terms")
    print(f"Prepared {root}; run reference.py separately for numerical evaluation")


if __name__ == "__main__":
    main()
