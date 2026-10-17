"""Scratch diagnostic: approximates the unreached decoded-longest-prompt check from a burn-dump decode run.

Usage: python3 -I judge_longest_decode.py TRUTH_BUNDLE OFFICIAL_BF16_BUNDLE DUMP_DIR PROMPT_INDEX PREFILL

Assembles the stateless rows before PREFILL (the gate uses the quarter prefill's own rows,
so this is an approximation) and the decoded rows after it, and judges them with both
bounds of the prompt (1.2 x the official BF16 max and mean deviation).
"""

import json
import sys
from pathlib import Path

import numpy as np

truth_dir, official_dir, dump = map(Path, sys.argv[1:4])
index, prefill = int(sys.argv[4]), int(sys.argv[5])
manifest = json.loads((truth_dir / "reference.json").read_text())
vocabulary = manifest["vocabulary_size"]
prompt = manifest["prompts"][index]
tokens = len(prompt["token_ids"])
truth = np.fromfile(truth_dir / prompt["logits_file"], dtype="<f4").astype(np.float64).reshape(tokens, vocabulary)
official_prompt = json.loads((official_dir / "reference.json").read_text())["prompts"][index]
official = np.fromfile(official_dir / official_prompt["logits_file"], dtype="<f4").astype(np.float64).reshape(tokens, vocabulary)
stateless = np.fromfile(dump / "logits_plain.bin", dtype="<f4").astype(np.float64).reshape(tokens, vocabulary)
decoded = np.fromfile(dump / "decode_logits.bin", dtype="<f4").astype(np.float64).reshape(tokens - prefill, vocabulary)
delta = np.abs(np.concatenate([stateless[:prefill], decoded]) - truth)
print(f"{dump.parent.name}/{dump.name}: prompt {index} decoded after a {prefill}-token prefill (prefill rows from the "
      f"stateless pass): max {delta.max():.4f} at pos {int(delta.argmax()) // vocabulary} "
      f"(bound {1.2 * np.abs(official - truth).max():.4f}), mean {delta.mean():.5f} "
      f"(bound {1.2 * np.abs(official - truth).mean():.5f})")
