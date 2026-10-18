"""Scratch (half2 S2): SHA-256 and size of every file of every bundle generated in this round (not committed).

Usage: python3 -I digest_bundles.py OUTPUT_JSON

Reads logs/runs.jsonl beside this script and writes, per bundle directory
name, {file name: {"sha256", "bytes"}} for every file in the bundle.
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    runs = [json.loads(line) for line in (HERE / "logs/runs.jsonl").read_text().splitlines()]
    out = {}
    for run in runs:
        root = Path(run["bundle"])
        out[root.name] = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                          for path in sorted(root.iterdir()) if path.is_file()}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{len(out)} bundles, {sum(len(v) for v in out.values())} files")


if __name__ == "__main__":
    main()
