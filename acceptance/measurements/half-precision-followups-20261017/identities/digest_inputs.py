"""Scratch (record of the rerun at 8181605): identities of every input the rerun read (not a gate).

Usage: python3 -I digest_inputs.py RERUN_LOGS_DIR OUTPUT_JSON

Reads the `== env` and `== binary` lines of every per-process log in
RERUN_LOGS_DIR and writes, without modifying anything:

* for every reference bundle a run named (`*_REFERENCE_DIR`,
  `*_DECODE_TRUTH_DIR`, and each entry of `*_OFFICIAL_DIRS`): the SHA-256 and
  size of its `reference.json`, which names every logits file by SHA-256 (the
  gates verify each file they read against it), with the manifest's dtype,
  device, attention implementation, generator digest, and checkpoint
  revision and config digest;
* for every checkpoint directory (`TRANSFORMERS_LLAMA_FAMILY_DIR`,
  `TRANSFORMERS_GEMMA_DIR`): the SHA-256 of its `config.json`, which the gates
  compare with the bundle's provenance before loading;
* for every test binary a log names: its SHA-256 now, beside the one the log
  recorded at run time.

Everything is read from the external volumes; only small files and the two
test binaries are hashed.
"""

import hashlib
import json
from pathlib import Path
import re
import sys
from datetime import datetime, timezone


def sha256(path):
    """SHA-256 and size of a file's bytes."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def manifest_summary(path):
    """The identity fields of a bundle manifest."""
    manifest = json.loads(path.read_text())
    provenance = manifest.get("provenance", {})
    return {
        "dtype": manifest.get("dtype") or provenance.get("dtype"),
        "device": provenance.get("device"),
        "attention_implementation": provenance.get("attention_implementation"),
        "script_sha256": provenance.get("script_sha256"),
        "repository": provenance.get("repository"),
        "revision": provenance.get("revision"),
        "config_sha256": provenance.get("config_sha256"),
        "generated_at": provenance.get("generated_at"),
    }


def main():
    """Writes the identity table."""
    logs = Path(sys.argv[1])
    bundles, checkpoints, binaries = {}, {}, {}
    for log in sorted(logs.glob("*.log")):
        text = log.read_text(errors="replace")
        for key, value in re.findall(r"(?m)^== env ([A-Z0-9_]+)=(.*)$", text):
            if key.endswith(("_REFERENCE_DIR", "_DECODE_TRUTH_DIR")):
                bundles.setdefault(value, set()).add(key)
            elif key.endswith("_OFFICIAL_DIRS"):
                for entry in value.split(":"):
                    bundles.setdefault(entry, set()).add(key)
            elif key in ("TRANSFORMERS_LLAMA_FAMILY_DIR", "TRANSFORMERS_GEMMA_DIR"):
                checkpoints.setdefault(value, set()).add(key)
        for path, digest in re.findall(r"(?m)^== binary (.+) sha256 ([0-9a-f]{64})$", text):
            binaries.setdefault(path, set()).add(digest)
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "bundles": {}, "checkpoints": {}, "binaries": {}}
    for path in sorted(bundles):
        manifest = Path(path) / "reference.json"
        digest, size = sha256(manifest)
        out["bundles"][path] = dict(manifest_summary(manifest), reference_json_sha256=digest,
                                    reference_json_bytes=size, variables=sorted(bundles[path]))
    for path in sorted(checkpoints):
        digest, size = sha256(Path(path) / "config.json")
        out["checkpoints"][path] = {"config_sha256": digest, "config_bytes": size, "variables": sorted(checkpoints[path])}
    for path in sorted(binaries):
        digest, size = sha256(path)
        out["binaries"][path] = {"sha256_now": digest, "bytes": size, "sha256_in_logs": sorted(binaries[path])}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{len(out['bundles'])} bundles, {len(out['checkpoints'])} checkpoints, {len(out['binaries'])} binaries")


if __name__ == "__main__":
    main()
