"""Compare Hanna et al.'s local original arrays with a local HF GPT-2 checkpoint.

This verifies weight identity after explicit head/Conv1D layout conversions.
It does not claim that the original exact-GELU graph equals HF's tanh-GELU graph.
Downloads and model execution are deliberately separate.
"""

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import torch
from safetensors import safe_open


def corresponding_weight(key, checkpoint):
    """Express one HF tensor in the author's documented array layout."""
    if key in ("t.w.tok_embeds", "t.w.unembed_arr"):
        return "wte.weight", checkpoint.get_tensor("wte.weight")
    if key == "t.w.pos_embeds":
        return "wpe.weight", checkpoint.get_tensor("wpe.weight")
    if key.startswith("final.ln.w."):
        target = "ln_f." + ("bias" if "bias" in key else "weight")
        return target, checkpoint.get_tensor(target)
    match = re.fullmatch(r"([am])(\d+)\.(.*)", key)
    if match is None:
        raise ValueError(f"unrecognized original array {key}")
    kind, index, suffix = match.groups()
    base = f"h.{index}."
    if suffix.startswith("ln.w."):
        target = base + ("ln_1." if kind == "a" else "ln_2.")
        target += "bias" if "bias" in suffix else "weight"
        return target, checkpoint.get_tensor(target)
    if kind == "m":
        mappings = {
            "w.proj_in_arr": ("mlp.c_fc.weight", True),
            "w.in_bias_arr": ("mlp.c_fc.bias", False),
            "w.proj_out_arr": ("mlp.c_proj.weight", True),
            "w.out_bias_arr": ("mlp.c_proj.bias", False),
        }
        target, transpose = mappings[suffix]
        target = base + target
        tensor = checkpoint.get_tensor(target)
        return target, tensor.T if transpose else tensor
    if suffix == "w.o_arr":
        target = base + "attn.c_proj.weight"
        return target, checkpoint.get_tensor(target).T.reshape(768, 12, 64).permute(1, 0, 2)
    if suffix == "w.o_bias_arr":
        target = base + "attn.c_proj.bias"
        return target, checkpoint.get_tensor(target)
    component = "qkv".index(suffix[2])
    bias = "bias" in suffix
    target = base + "attn.c_attn." + ("bias" if bias else "weight")
    packed = checkpoint.get_tensor(target)
    start, end = component * 768, (component + 1) * 768
    tensor = packed[start:end].reshape(12, 64) if bias else packed[:, start:end].T.reshape(12, 64, 768)
    return target, tensor


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    entries = re.findall(
        r"^\s*'([^']+)' \[[^\]]+\] Array ([0-9a-f]+)",
        args.circuit.read_text(), re.MULTILINE,
    )
    if len(entries) != 197:
        raise ValueError(f"expected the pinned GPT-2 circuit's 197 array entries, got {len(entries)}")
    reports = []
    with zipfile.ZipFile(args.archive) as archive, safe_open(args.checkpoint, framework="pt") as checkpoint:
        names = {
            name.removeprefix("tensors_by_hash_cache/").replace("/", ""): name
            for name in archive.namelist() if not name.endswith("/")
        }
        for key, short_hash in entries:
            matches = [full for full in names if full.startswith(short_hash)]
            if len(matches) != 1:
                raise ValueError(f"original array {key} has {len(matches)} matching payloads")
            data = archive.read(names[matches[0]])
            original = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            target, mapped = corresponding_weight(key, checkpoint)
            reports.append({
                "author_key": key, "source_key": target, "shape": list(original.shape),
                "equal": torch.equal(original, mapped),
                "max_abs": (original - mapped).abs().max().item(),
            })
    result = {
        "archive_sha256": sha256(args.archive), "circuit_sha256": sha256(args.circuit),
        "checkpoint_sha256": sha256(args.checkpoint), "arrays": reports,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    equal = sum(row["equal"] for row in reports)
    print(f"{equal}/{len(reports)} arrays bitwise equal; report: {args.output}")
    if equal != len(reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
