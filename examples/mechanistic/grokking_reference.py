#!/usr/bin/env python3
"""Independent NumPy oracle for the authors' forward equations; no pickle loading.

Optional verification helper (requires NumPy):
  python3 examples/mechanistic/grokking_reference.py target/mechanistic/grokking.safetensors
Prints first-eight-pair logits as JSON. Redirect to a temporary file to compare
against grokking_reference_logits.json. This is not used by the Rust example.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np

raw = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == "28f7212abd53937bc85ee0558536d89b0b882b3cfbec77d6942f605f7d2d4116"
header_length, = struct.unpack("<Q", raw[:8])
header = json.loads(raw[8:8 + header_length])
weights = {}
for name, item in header.items():
    start, end = item["data_offsets"]
    # F64 accumulation is independent of Burn's F32 matrix multiplication.
    weights[name] = np.frombuffer(raw[8 + header_length + start:8 + header_length + end],
                                 dtype="<f4").reshape(item["shape"]).astype(np.float64)

tokens = np.array([[0, b, 113] for b in range(8)])
x = weights["embed.W_E"].T[tokens] + weights["pos_embed.W_pos"]
q = np.einsum("hfd,bsd->bhsf", weights["blocks.0.attn.W_Q"], x)
k = np.einsum("hfd,bsd->bhsf", weights["blocks.0.attn.W_K"], x)
v = np.einsum("hfd,bsd->bhsf", weights["blocks.0.attn.W_V"], x)
scores = np.einsum("bhqf,bhkf->bhqk", q, k) / np.sqrt(32)
scores = np.where(np.tril(np.ones((3, 3), dtype=bool)), scores, -np.inf)
attention = np.exp(scores - scores.max(axis=-1, keepdims=True))
attention /= attention.sum(axis=-1, keepdims=True)
z = np.einsum("bhqk,bhkf->bhqf", attention, v).transpose(0, 2, 1, 3).reshape(8, 3, 128)
residual = x + z @ weights["blocks.0.attn.W_O"].T
neurons = np.maximum(residual @ weights["blocks.0.mlp.W_in"].T + weights["blocks.0.mlp.b_in"], 0)
output = residual + neurons @ weights["blocks.0.mlp.W_out"].T + weights["blocks.0.mlp.b_out"]
logits = (output @ weights["unembed.W_U"])[:, -1, :113]
print(json.dumps(logits.flatten().tolist()))
