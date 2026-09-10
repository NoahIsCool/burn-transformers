"""Regenerate the small F32 oracle with PyTorch (recorded: torch 2.10.0).

Run from any directory: python3 src/models/gpt_neox/fixtures/reference.py
No pretrained weights or downloads are involved. This implements the equations
in Hugging Face Transformers v4.28.1 modeling_gpt_neox.py using explicit fixed
parameters. Rust tests independently construct the same parameters.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)


def values(shape, offset, scale=0.15):
    return torch.tensor(
        [math.sin(i * 0.17 + offset) * scale for i in range(math.prod(shape))],
        dtype=torch.float32,
    ).reshape(shape)


def linear(x, inputs, outputs, offset):
    return F.linear(x, values([outputs, inputs], offset), values([outputs], offset + 0.4, 0.03))


def norm(x, offset):
    return F.layer_norm(
        x, [8], torch.tensor([0.7 + i * 0.05 for i in range(8)]),
        values([8], offset, 0.03), 1e-5,
    )


positions = torch.tensor([0.0, 2.0, 5.0]).reshape(1, 1, 3, 1)


def rotary(x):
    # Rotate half of each four-channel head using split-half pairing.
    prefix = x[..., :2]
    rotated_half = torch.cat([-prefix[..., 1:], prefix[..., :1]], dim=-1)
    rotated = prefix * positions.cos() + rotated_half * positions.sin()
    return torch.cat([rotated, x[..., 2:]], dim=-1)


hidden = values([13, 8], 0.1)[torch.tensor([[1, 4, 2]])]
block_outputs = []
for layer in range(2):
    offset = 1 + layer * 10
    attention_input = norm(hidden, offset)
    query, key, value = [
        linear(attention_input, 8, 8, offset + component)
        .reshape(1, 3, 2, 4).permute(0, 2, 1, 3)
        for component in range(3)
    ]
    scores = rotary(query) @ rotary(key).transpose(-1, -2) / 2.0
    scores = scores.masked_fill(torch.ones(3, 3, dtype=torch.bool).triu(1), float("-inf"))
    context = (scores.softmax(-1) @ value).permute(0, 2, 1, 3).reshape(1, 3, 8)
    attention = linear(context, 8, 8, offset + 3)
    mlp = linear(F.gelu(linear(norm(hidden, offset + 0.5), 8, 12, offset + 4)), 12, 8, offset + 5)
    hidden = mlp + attention + hidden
    block_outputs.append(hidden.clone())

final = norm(hidden, 30)
logits = F.linear(final, values([13, 8], 31))
Path(__file__).with_name("oracle.json").write_text(json.dumps({
    "logits": logits.flatten().tolist(),
    "block0": block_outputs[0].flatten().tolist(),
    "final": final.flatten().tolist(),
}, indent=2) + "\n")
