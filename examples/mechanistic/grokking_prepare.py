#!/usr/bin/env python3
"""Fetch one published 908 KB checkpoint, without loading a PyTorch pickle.

Only Python's standard library is needed. Run from the repository root:
    python3 examples/mechanistic/grokking_prepare.py
The output lives under ignored target/. The Rust example performs no networking.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import urllib.request

EXPECTED_OUTPUT_SHA256 = "28f7212abd53937bc85ee0558536d89b0b882b3cfbec77d6942f605f7d2d4116"


def verify(data, expected, description):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {description}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("target/mechanistic/grokking.safetensors"))
    args = parser.parse_args()
    if args.output.exists():
        verify(args.output.read_bytes(), EXPECTED_OUTPUT_SHA256, str(args.output))
        print(f"Already verified: {args.output}")
        return

    manifest = json.loads(Path(__file__).with_name("grokking_manifest.json").read_text())
    start, end = manifest["range_start"], manifest["range_end"]
    request = urllib.request.Request(manifest["url"], headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        # Refuse a server that ignores Range instead of downloading its 456 MB archive.
        if response.status != 206:
            raise ValueError("Server did not honor the bounded HTTP range request")
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError("Unexpected range length")
    verify(data, manifest["range_sha256"], "published checkpoint range")

    # Safetensors is a JSON header followed by raw contiguous little-endian floats.
    # Offsets/shapes were passively read from the archive metadata; no pickle code
    # is fetched or executed here. Every raw tensor has its own checksum, too.
    header, payload = {}, bytearray()
    for name, tensor in manifest["tensors"].items():
        offset, length = tensor["range_offset"], tensor["length"]
        raw = data[offset:offset + length]
        verify(raw, tensor["sha256"], name)
        header[name] = {"dtype": "F32", "shape": tensor["shape"],
                        "data_offsets": [len(payload), len(payload) + length]}
        payload.extend(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    output = struct.pack("<Q", len(encoded)) + encoded + payload
    verify(output, EXPECTED_OUTPUT_SHA256, "converted checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete file. An interrupted write must not leave a partial
    # checkpoint at the final path that makes every subsequent retry fail.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(output)
        temporary.replace(args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Verified {len(data):,} downloaded bytes; wrote {len(output):,} bytes to {args.output}")


if __name__ == "__main__":
    main()
