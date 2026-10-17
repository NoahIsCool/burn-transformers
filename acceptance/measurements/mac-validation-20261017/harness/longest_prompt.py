"""Scratch helper: prints the index, comma-joined tokens, and quarter-prefill length of a bundle's longest prompt."""

import json
import sys

manifest = json.load(open(sys.argv[1] + "/reference.json"))
index, prompt = max(enumerate(manifest["prompts"]), key=lambda entry: len(entry[1]["token_ids"]))
tokens = prompt["token_ids"]
print(index, ",".join(map(str, tokens)), max(len(tokens) // 4, 1))
