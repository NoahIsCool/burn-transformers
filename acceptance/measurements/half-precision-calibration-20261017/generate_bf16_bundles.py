#!/usr/bin/env python3
"""Generates the official BF16 twin of each F32/F64 reference bundle (scratch driver, not committed).

For every pair it reads the existing bundle's manifest and reruns
scripts/generate_causal_lm_reference.py with --dtype bfloat16 and exactly the
same checkpoint directory, repository, revision, prompts, greedy length and
layer truncation, under /usr/bin/time -l. Output goes to the log given as the
second argument; the first argument is the worktree to run from.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REFERENCES = Path("/Volumes/T9/transformers-references")
PAIRS = [
    "SmolLM-135M-1d461723",
    "Qwen3-4B-531c80e2",
    "SmolLM3-3B-1fa6fa28",
    "gemma-3-4b-it-093f9f38",
    "Qwen2-7B-453ed157-f64",
    "Llama-3.1-8B-d04e592b",
    "gemma-3-4b-it-093f9f38-long",
]


def main() -> int:
    worktree = Path(sys.argv[1])
    log = Path(sys.argv[2]).open("a")
    only = set(sys.argv[3:])
    status = 0
    for name in PAIRS:
        if only and name not in only:
            continue
        truth = REFERENCES / name
        output = REFERENCES / f"{name}-bf16"
        manifest = json.loads((truth / "reference.json").read_text())
        provenance = manifest["provenance"]
        free = shutil.disk_usage(REFERENCES).free
        command = [
            "/usr/bin/time", "-l", "python3", "-I", "scripts/generate_causal_lm_reference.py",
            "--model-dir", provenance["model_dir"],
            "--output", str(output),
            "--repository", provenance["repository"],
            "--revision", provenance["revision"],
            "--dtype", "bfloat16",
            "--greedy-tokens", str(len(manifest["greedy"]["steps"])),
        ]
        if provenance.get("num_hidden_layers") is not None:
            command += ["--num-hidden-layers", str(provenance["num_hidden_layers"])]
        for prompt in manifest["prompts"]:
            command += ["--prompt", prompt["text"]]
        log.write(f"== {name}-bf16: official BF16 twin of {truth} (reference.json SHA-256 "
                  f"{hashlib.sha256((truth / 'reference.json').read_bytes()).hexdigest()}); "
                  f"{free / 2**30:.0f} GiB free on {REFERENCES}\n")
        shown = [part if "\n" not in part and len(part) < 120 else json.dumps(part[:60] + "…") for part in command]
        log.write("== command " + " ".join(shown) + "\n")
        log.flush()
        if free < 8 * 2**30:
            log.write("== SKIP: less than 8 GiB free\n")
            status = 1
            continue
        started = time.monotonic()
        result = subprocess.run(command, cwd=worktree, stdout=log, stderr=subprocess.STDOUT)
        wall = time.monotonic() - started
        log.write(f"== exit={result.returncode} wall={wall:.1f}s\n")
        if result.returncode == 0:
            written = json.loads((output / "reference.json").read_text())
            same = [p["token_ids"] for p in written["prompts"]] == [p["token_ids"] for p in manifest["prompts"]]
            digest = hashlib.sha256((output / "reference.json").read_bytes()).hexdigest()
            log.write(f"== {output}: reference.json SHA-256 {digest}; token IDs identical to the twin: {same}; "
                      f"dtype {written['provenance']['dtype']}\n")
            status |= 0 if same else 1
        else:
            status = 1
        log.flush()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
