#!/usr/bin/env python3
"""Verify local GPT-2 artifacts and the frozen Greater-Than prompt pairs.

    python3 examples/mechanistic/greater_than/prepare.py --checkpoint /path/to/gpt2

Requires tokenizers. This script copies verified local artifacts into ignored
target/mechanistic/greater_than/. It neither downloads nor runs a model.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import tokenizers
from tokenizers import Tokenizer


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[2] / "target/mechanistic/greater_than"
MODEL_FILES = ("model.safetensors", "config.json", "tokenizer.json")


def strict_json(data):
    """Reject ambiguous duplicate keys and non-finite JSON constants."""
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(data, object_pairs_hook=pairs_hook, parse_constant=invalid_constant)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def atomic_output(path):
    """Expose a completed file only after its writer succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".greater-than-", delete=False) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_output(path) as handle:
        handle.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def describe_file(path, root):
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def verify_file(path, digest):
    if sha256_file(path) != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")


def validate_manifest(manifest):
    """Check the protocol assumptions used by both independent runners."""
    if (manifest["protocol"], manifest["activation"], manifest["dtype"],
        manifest["analysis_dtype"]) != ("greater-than-v1", "gelu", "F32", "F64"):
        raise ValueError("Unsupported protocol or arithmetic configuration")
    if (manifest["fixture_entries"], manifest["sequence_length"],
        manifest["readout_position"], manifest["changed_position"],
        manifest["vocabulary_size"], manifest["batch_size"]) != (194, 13, 12, 8, 50257, 8):
        raise ValueError("Unexpected frozen study dimensions")
    expected = ["clean", "corrupt", "empty_clean", "observe_clean"]
    for layer in manifest["node_layers"]:
        expected.extend(f"{name}_{layer}" for name in
                        ("node_patch", "node_reverse", "zero", "identity", "restore"))
    expected.extend(f"direct_{layer}" for layer in manifest["direct_layers"])
    expected.append("node_patch_all_late")
    if (manifest["node_layers"] != [2, 8, 9, 10, 11]
        or manifest["late_layers"] != [8, 9, 10, 11]
        or manifest["direct_layers"] != list(range(12))
        or manifest["conditions"] != expected or len(expected) != 42):
        raise ValueError("Unexpected intervention inventory")
    if set(manifest["artifact_sha256"]) != set(MODEL_FILES):
        raise ValueError("Unexpected checkpoint artifact inventory")


def verify_fixture(fixture, manifest, tokenizer):
    """Check every prompt and ensure corruption changes only the intended token."""
    entries = fixture["entries"]
    if (fixture["protocol"] != manifest["protocol"]
        or fixture["tokenizer_revision"] != manifest["hf_revision"]
        or len(entries) != manifest["fixture_entries"]):
        raise ValueError("Fixture provenance or entry count mismatch")
    numeric_ids = []
    for number in range(100):
        ids = tokenizer.encode(f"{number:02d}", add_special_tokens=False).ids
        if len(ids) != 1:
            raise ValueError(f"Numeric suffix {number:02d} is not one token")
        numeric_ids.append(ids[0])
    if numeric_ids != manifest["numeric_token_ids"]:
        raise ValueError("Numeric token ID inventory differs from the frozen manifest")
    counts = Counter()
    suffixes = {}
    for index, entry in enumerate(entries):
        suffix = entry["year"] % 100
        century = entry["year"] // 100
        if not 11 <= century <= 17 or not 2 <= suffix <= 98:
            raise ValueError(f"Entry {index}: year outside the fixed study pool")
        if (entry["start_suffix_position"], entry["readout_position"]) != (8, 12):
            raise ValueError(f"Entry {index}: incorrect intervention/readout alignment")
        expected_clean = f"<|endoftext|> The {entry['noun']} lasted from the year {entry['year']} to the year {century}"
        expected_corrupt = f"<|endoftext|> The {entry['noun']} lasted from the year {century}01 to the year {century}"
        if (entry["clean"], entry["corrupt"]) != (expected_clean, expected_corrupt):
            raise ValueError(f"Entry {index}: prompt text does not match its metadata")
        for kind in ("clean", "corrupt"):
            ids = tokenizer.encode(entry[kind], add_special_tokens=False).ids
            if ids != entry[f"{kind}_ids"] or len(ids) != 13:
                raise ValueError(f"Entry {index}: {kind} tokenizer mismatch")
        changed = [i for i, pair in enumerate(zip(entry["clean_ids"], entry["corrupt_ids"]))
                   if pair[0] != pair[1]]
        if changed != [8] or entry["clean_ids"][8] != numeric_ids[suffix] or entry["corrupt_ids"][8] != numeric_ids[1]:
            raise ValueError(f"Entry {index}: corruption changed an unintended token")
        counts[entry["split"]] += 1
        suffixes.setdefault(entry["split"], []).append(suffix)
    if dict(counts) != manifest["split_counts"] or any(sorted(values) != list(range(2, 99)) for values in suffixes.values()):
        raise ValueError("Split sizes or suffix coverage differ from the frozen fixture")
    return {"all_prompts_retokenized": True, "paired_prompts": len(entries),
            "single_changed_position": 8, "readout_position": 12,
            "numeric_tokens_verified": 100, "split_counts": dict(counts)}


def verify_preparation(root):
    """Rehash all prepared inputs before any model computation."""
    manifest = strict_json((HERE / "manifest.json").read_bytes())
    validate_manifest(manifest)
    metadata = strict_json((root / "preparation.json").read_bytes())
    expected = {"manifest_sha256": sha256_file(HERE / "manifest.json"),
                "fixture_sha256": sha256_file(HERE / "fixture.json"),
                "prepare_sha256": sha256_file(HERE / "prepare.py")}
    if metadata.get("schema_version") != 1 or metadata.get("fingerprint") != expected:
        raise ValueError("Preparation is stale; rerun prepare.py")
    expected_paths = {f"model/{name}" for name in MODEL_FILES} | {"fixture.json"}
    if set(metadata.get("outputs", {})) != expected_paths:
        raise ValueError("Incomplete prepared artifact inventory")
    for name, specification in metadata["outputs"].items():
        path = root / name
        if specification["path"] != name or path.stat().st_size != specification["bytes"]:
            raise ValueError(f"Prepared artifact size/path mismatch: {name}")
        pinned = manifest["fixture_sha256"] if name == "fixture.json" else manifest["artifact_sha256"][Path(name).name]
        if specification["sha256"] != pinned:
            raise ValueError(f"Prepared artifact digest differs from manifest: {name}")
        verify_file(path, pinned)
    fixture = strict_json((root / "fixture.json").read_bytes())
    validated = verify_fixture(fixture, manifest, Tokenizer.from_file(str(root / "model/tokenizer.json")))
    if metadata["tokenizer_validation"] != validated:
        raise ValueError("Recorded tokenizer validation does not match current inputs")
    return manifest, fixture, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="local directory containing the three pinned GPT-2 artifacts")
    parser.add_argument("--output", "--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.output.resolve()
    manifest = strict_json((HERE / "manifest.json").read_bytes())
    validate_manifest(manifest)
    verify_file(HERE / "fixture.json", manifest["fixture_sha256"])
    for name in MODEL_FILES:
        verify_file(args.checkpoint / name, manifest["artifact_sha256"][name])
    fixture = strict_json((HERE / "fixture.json").read_bytes())
    validation = verify_fixture(fixture, manifest, Tokenizer.from_file(str(args.checkpoint / "tokenizer.json")))
    sources = {f"model/{name}": args.checkpoint / name for name in MODEL_FILES}
    sources["fixture.json"] = HERE / "fixture.json"
    for relative, source in sources.items():
        destination = root / relative
        expected = manifest["fixture_sha256"] if relative == "fixture.json" else manifest["artifact_sha256"][source.name]
        if not destination.exists() or sha256_file(destination) != expected:
            with atomic_output(destination) as handle, source.open("rb") as original:
                shutil.copyfileobj(original, handle)
        verify_file(destination, expected)
    metadata = {
        "schema_version": 1, "protocol": manifest["protocol"],
        "fingerprint": {"manifest_sha256": sha256_file(HERE / "manifest.json"),
                        "fixture_sha256": sha256_file(HERE / "fixture.json"),
                        "prepare_sha256": sha256_file(Path(__file__))},
        "versions": {"tokenizers": tokenizers.__version__},
        "checkpoint": {"repository": manifest["hf_repository"], "revision": manifest["hf_revision"],
                       "activation_override": "gelu; source config remains unchanged (gelu_new)"},
        "outputs": {relative: describe_file(root / relative, root) for relative in sources},
        "tokenizer_validation": validation,
    }
    write_json(root / "preparation.json", metadata)
    verify_preparation(root)
    print(f"Verified all 194 prompt pairs, 100 numeric token IDs, and three checkpoint hashes: {root}")


if __name__ == "__main__":
    main()
