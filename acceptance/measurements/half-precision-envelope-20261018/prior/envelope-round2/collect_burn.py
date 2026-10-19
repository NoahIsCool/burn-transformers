"""Scratch (round 2): Burn half-precision statistics from the previous round's logs and dumps (not committed).

Usage: python3 -I collect_burn.py OUT.json

Every value carries its log path, that log's SHA-256, and its `== source <sha>`
line. Prompt statistics come from the gates' own full-precision lines
(`prompt i: ... max |Δ| = ... mean |Δ| = ...`). A gate stops at its first
failing prompt; the prompts it did not reach are recomputed here, exactly
(F64 differences, F64 sum), from the round's burn-dump `logits_plain.bin`
(the ordinary stateless forward, the gate's own path) against the truth
bundle, and marked `"source_kind": "burn-dump"`. Where a gate line and a dump
cover the same prompt, both are kept and compared (`dump_matches_gate`).

Decode self-consistency comes from the cached-greedy gates' lines (the max
over steps when the gate passed; the first step above 0.5 when it failed) and,
where the round dumped Burn's cached decode (`decode_logits.bin`) and
stateless pass (`logits_plain.bin`) on the truth's greedy tokens, from those
dumps step by step (F64 differences).
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path("/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/mac-validation")
LLAMA = ROOT / "llama-family"
GEMMA = ROOT / "gemma"
REFERENCES = Path("/Volumes/T9/transformers-references")
#: key -> (log name stem, truth bundle)
LLAMA_KEYS = {"SmolLM-135M": ("SmolLM-135M", "SmolLM-135M-1d461723"), "Qwen3-4B": ("Qwen3-4B", "Qwen3-4B-531c80e2"),
              "SmolLM3-3B": ("SmolLM3-3B", "SmolLM3-3B-1fa6fa28"), "Qwen2-7B": ("Qwen2-7B-f64", "Qwen2-7B-453ed157-f64"),
              "Llama-3.1-8B": ("Llama-3.1-8B", "Llama-3.1-8B-d04e592b")}
GATE = re.compile(r"^prompt (\d+): (\d+) tokens, max \|Δ\| = ([0-9.e+-]+) at position (\d+) .*mean \|Δ\| = ([0-9.e+-]+)")
LONG = re.compile(r"^long prompt: (\d+) tokens, max \|Δ\| = ([0-9.e+-]+) at position (\d+) .*mean \|Δ\| = ([0-9.e+-]+)")
NONFINITE = re.compile(r"non-finite|NaN|not finite", re.I)
DECODE_MAX = re.compile(r"(?:long )?cached decode: (\d+) decode steps from position (\d+): max \|Δ\| = ([0-9.e+-]+) to the stateless pass, at step (\d+)")
DECODE_FAIL = re.compile(r"cached decode step (\d+) at position (\d+): max \|Δ\| = ([0-9.e+-]+) to the stateless pass exceeds")


def sha(path):
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def source_of(path):
    """Returns the `== source` SHA of a log, or None."""
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("== source "):
            return line.split()[2]
    return None


def provenance(path):
    """Path, digest and source line of a log."""
    return {"log": str(path), "log_sha256": sha(path), "source": source_of(path)}


def gate_prompts(path):
    """Parses a prompt-logits (or long-context logits) gate log into {prompt: stats}."""
    rows = {}
    if not path.exists():
        return rows
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        match = GATE.match(line)
        if match:
            rows[int(match[1])] = {"max": float(match[3]), "position": int(match[4]), "mean": float(match[5]),
                                   "source_kind": "gate", **provenance(path)}
        match = LONG.match(line)
        if match:
            rows[0] = {"max": float(match[2]), "position": int(match[3]), "mean": float(match[4]),
                       "source_kind": "gate", **provenance(path)}
    return rows


def truth_rows(truth_name):
    """The truth's prompt logits as F64 arrays, and its manifest."""
    root = REFERENCES / truth_name
    manifest = json.loads((root / "reference.json").read_text())
    vocabulary = manifest["vocabulary_size"]
    return manifest, [np.fromfile(root / p["logits_file"], dtype="<f4").astype(np.float64).reshape(len(p["token_ids"]), vocabulary)
                      for p in manifest["prompts"]]


def dump_prompt(path, truth):
    """Exact max/mean |Δ| of a dump's logits_plain.bin against the truth rows of one prompt."""
    values = np.fromfile(path, dtype="<f4").astype(np.float64).reshape(truth.shape)
    if not np.isfinite(values).all():
        return {"non_finite": True, "dump": str(path), "dump_sha256": sha(path)}
    delta = np.abs(values - truth)
    flat = int(delta.argmax())
    return {"max": float(delta.max()), "position": flat // truth.shape[1], "mean": float(delta.sum() / delta.size),
            "dump": str(path), "dump_sha256": sha(path)}


def merge(gate_rows, dump_dir, truth, dump_log_stem):
    """Gate rows, filled and cross-checked with burn-dump rows of `dump_dir`/prompt<i>/logits_plain.bin."""
    out = {}
    for index, rows in enumerate(truth):
        row = dict(gate_rows.get(index, {}))
        path = dump_dir / f"prompt{index}" / "logits_plain.bin" if dump_dir else None
        if path is not None and path.exists():
            dumped = dump_prompt(path, rows)
            log = Path(f"{dump_log_stem}.prompt{index}.log")
            if log.exists():
                dumped.update({"dump_log": str(log), "dump_log_sha256": sha(log), "dump_source": source_of(log)})
            if row:
                row["dump_matches_gate"] = (not dumped.get("non_finite") and dumped["max"] == row["max"]
                                            and abs(dumped["mean"] - row["mean"]) <= 1e-15 * max(1.0, row["mean"]))
                row["dump"] = dumped
            else:
                row = {**dumped, "source_kind": "burn-dump"}
        if row:
            out[index] = row
    return out


def gate_decode(path):
    """Parses the decode self-consistency line of a cached greedy (or long cached) gate log."""
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    match = DECODE_FAIL.search(text)
    if match:
        return {"value": float(match[3]), "step": int(match[1]), "position": int(match[2]),
                "kind": "first step above 0.5 (gate stopped)", **provenance(path)}
    match = DECODE_MAX.search(text)
    if match:
        return {"value": float(match[3]), "step": int(match[4]), "steps": int(match[1]), "first_position": int(match[2]),
                "kind": "max over steps", **provenance(path)}
    if NONFINITE.search(text):
        return {"non_finite": True, "kind": "non-finite", **provenance(path)}
    return {"kind": "no decode line (gate stopped before decoding)", **provenance(path)}


def decode_dump(config_dir, manifest, log):
    """Per-step |cached - stateless| maxima of one Burn decode dump, and both passes' rows (F64)."""
    vocabulary = manifest["vocabulary_size"]
    first = len(manifest["prompts"][0]["token_ids"])
    steps = len(manifest["greedy"]["steps"])
    plain = np.fromfile(config_dir / "logits_plain.bin", dtype="<f4").astype(np.float64).reshape(first + steps, vocabulary)[first:]
    decoded = np.fromfile(config_dir / "decode_logits.bin", dtype="<f4").astype(np.float64).reshape(-1, vocabulary)
    gaps = np.abs(decoded - plain).max(axis=1)
    record = {"per_step": [float(g) for g in gaps], "max": float(gaps.max()), "argmax_step": int(gaps.argmax()),
              "dump_dir": str(config_dir), "decode_logits_sha256": sha(config_dir / "decode_logits.bin"),
              "logits_plain_sha256": sha(config_dir / "logits_plain.bin")}
    if log.exists():
        record.update({"dump_log": str(log), "dump_log_sha256": sha(log), "dump_source": source_of(log)})
    return record


def main():
    data = {"prompts": {}, "decode": {}}
    for key, (stem, truth_name) in LLAMA_KEYS.items():
        manifest, truth = truth_rows(truth_name)
        diag = LLAMA / "diagnostics"
        flex = merge(gate_prompts(LLAMA / f"{stem}.bf16.flex.prompt_logits_match_the_reference.log"),
                     diag / "data" / f"{key}-burn" / "flex-bf16-fused" if (diag / "data" / f"{key}-burn").exists() else None,
                     truth, diag / f"{key}.burn-dump.flex-bf16-fused")
        metal = merge(gate_prompts(LLAMA / f"{stem}.f16.metal.prompt_logits_match_the_reference.log"),
                      diag / "data" / f"{key}-burn" / "metal-f16-fused" if (diag / "data" / f"{key}-burn" / "metal-f16-fused").exists() else None,
                      truth, diag / f"{key}.burn-dump.metal-f16-fused")
        entry = {"flex-bf16@53f167b": flex, "metal-f16@53f167b": metal}
        silu = diag / "data" / f"{key}-burn" / "flex-bf16-fused-f32silu"
        if silu.exists():
            entry["flex-bf16-f32silu@53f167b+scratch"] = merge({}, silu, truth, diag / f"{key}.burn-dump.flex-bf16-fused-f32silu")
        data["prompts"][key] = entry
        decode = {"flex-bf16@53f167b": gate_decode(LLAMA / f"{stem}.bf16.flex.cached_greedy_matches_the_reference.log"),
                  "metal-f16@53f167b": gate_decode(LLAMA / f"{stem}.f16.metal.cached_greedy_matches_the_reference.log")}
        dumps = diag / "data" / f"{key}-burn-decode"
        if dumps.exists():
            decode["dumps"] = {config.name: decode_dump(config, manifest, diag / f"{key}.burn-decode.{config.name}.log")
                               for config in sorted(p for p in dumps.iterdir() if p.is_dir())}
        # Round 2 reran the same authenticated burn-dump binary for checkpoints the previous round never dumped.
        rerun = HERE / "data" / f"{key}-burn-decode"
        if rerun.exists():
            decode.setdefault("dumps", {}).update(
                {config.name: decode_dump(config, manifest, HERE / "logs" / f"{key}.burn-decode.{config.name}.log")
                 for config in sorted(p for p in rerun.iterdir() if p.is_dir() and (p / "decode_logits.bin").exists())})
        data["decode"][key] = decode
    # Gemma: 53f167b and c2936fe (the GELU fix); Metal/F16 is non-finite at both.
    manifest, truth = truth_rows("gemma-3-4b-it-093f9f38")
    diag = GEMMA / "diagnostics"
    data["prompts"]["gemma-3-4b-it"] = {
        "flex-bf16@53f167b": merge(gate_prompts(GEMMA / "gemma-3-4b-it.bf16.flex.prompt_logits_match_the_reference.log"),
                                   diag / "data" / "burn" / "flex-bf16-fused", truth, diag / "burn-dump.flex-bf16-fused"),
        "flex-bf16@c2936fe": merge(gate_prompts(GEMMA / "c2936fe" / "gemma-3-4b-it.bf16.flex.prompt_logits_match_the_reference.log"),
                                   None, truth, None),
        "flex-bf16-f32gelu@53f167b+scratch": merge({}, diag / "data" / "burn" / "flex-bf16-fused-f32gelu", truth,
                                                   diag / "burn-dump.flex-bf16-fused-f32gelu"),
        "metal-f16@53f167b": merge({}, diag / "data" / "burn" / "metal-f16-fused", truth, diag / "burn-dump.metal-f16-fused"),
        "metal-f16@c2936fe": {"gate": provenance(GEMMA / "c2936fe" / "gemma-3-4b-it.f16.metal.prompt_logits_match_the_reference.log"),
                              "non_finite": True},
    }
    data["prompts"]["gemma-long"] = {
        "flex-bf16@53f167b": gate_prompts(GEMMA / "gemma-3-4b-it.bf16.flex.long_context_logits_cross_the_sliding_window.log"),
        "flex-bf16@c2936fe": gate_prompts(GEMMA / "c2936fe" / "gemma-3-4b-it.bf16.flex.long_context_logits_cross_the_sliding_window.log"),
        "metal-f16@c2936fe": {"gate": provenance(GEMMA / "c2936fe" / "gemma-3-4b-it.f16.metal.long_context_logits_cross_the_sliding_window.log"),
                              "non_finite": True},
    }
    dumps = diag / "data" / "burn-decode"
    data["decode"]["gemma-3-4b-it"] = {
        "flex-bf16@53f167b": gate_decode(GEMMA / "gemma-3-4b-it.bf16.flex.cached_greedy_matches_the_reference.log"),
        "flex-bf16@c2936fe": gate_decode(GEMMA / "c2936fe" / "gemma-3-4b-it.bf16.flex.cached_greedy_matches_the_reference.log"),
        "metal-f16@c2936fe": gate_decode(GEMMA / "c2936fe" / "gemma-3-4b-it.f16.metal.cached_greedy_matches_the_reference.log"),
        "dumps": {config.name: decode_dump(config, manifest, diag / f"burn-decode.{config.name}.log")
                  for config in sorted(p for p in dumps.iterdir() if p.is_dir())},
    }
    data["decode"]["gemma-long"] = {
        "flex-bf16@53f167b": gate_decode(GEMMA / "gemma-3-4b-it.bf16.flex.long_context_cached_decode_wraps_the_rings.log"),
        "flex-bf16@c2936fe": gate_decode(GEMMA / "c2936fe" / "gemma-3-4b-it.bf16.flex.long_context_cached_decode_wraps_the_rings.log"),
        "metal-f16@c2936fe": gate_decode(GEMMA / "c2936fe" / "gemma-3-4b-it.f16.metal.long_context_cached_decode_wraps_the_rings.log"),
    }
    Path(sys.argv[1]).write_text(json.dumps(data, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
