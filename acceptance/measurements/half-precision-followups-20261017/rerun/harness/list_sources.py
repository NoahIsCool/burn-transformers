"""Scratch: lists the source commits of earlier F32 Llama-family and Gemma gate logs (not committed)."""

import collections
import glob
import os

SP = "/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad"
counts = collections.Counter()
for path in glob.glob(f"{SP}/**/*.log", recursive=True):
    base = os.path.basename(path)
    if ".f32." not in base or not any(k in base for k in ("prompt_logits", "cached_greedy", "batched_prefill", "long_context")):
        continue
    if "/followups/rerun/8181605/" in path:
        continue
    try:
        first = open(path, errors="replace").readline().split()
    except OSError:
        continue
    sha = first[2][:12] if len(first) > 2 and first[:2] == ["==", "source"] else "?"
    counts[(sha, os.path.relpath(os.path.dirname(path), SP))] += 1
for (sha, directory), n in sorted(counts.items(), key=lambda x: x[0][1]):
    print(f"{n:4d} {sha} {directory}")
