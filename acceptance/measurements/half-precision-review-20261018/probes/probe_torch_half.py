"""PyTorch's half-precision rounding points behind the crate's eager attention graph (torch 2.10.0, transformers 5.2.0).

Usage: python3 -I probe_torch_half.py

Prints, on the CPU and on MPS:

1. ``x * s`` with ``s = 1 / sqrt(128)`` a Python float, for every finite BF16
   and F16 ``x``, against ``(x.float() * s).to(dtype)`` evaluated on the CPU
   (the F32 product rounded once), classifying each difference, and how many
   of those values a dtype-rounded scale would move;
2. half-precision matrix products: one plus 4096 terms of ``2^-11`` (exactly
   3), and a ``[16, 256] x [256, 16]`` product of multiples of ``1/64`` in
   ``[-1, 1]`` whose exact sums are exact in F32, against the exact sum
   rounded once;
3. Transformers' own ``eager_attention_forward`` (Llama) on the CPU, in BF16
   and F16, on the grid of ``EagerGrid`` in ``src/attention/dispatch_tests.rs``
   (reconstructed here in F32 arithmetic), as the dtype bits of every output
   in ``[batch, head, query, value]`` order, against the issue's emulation
   ``round(softmax_f32(round(round(QK^T) * s) + mask)) V`` written with torch
   operations. Excluded links carry ``finfo(dtype).min``, as Transformers'
   masks do; the fully masked row is reported separately, since Transformers
   averages it and the crate defines it as zero.

No checkpoint is read; the script only allocates small tensors.
"""

import math
import platform

import numpy as np
import torch
import transformers
from transformers.models.llama.modeling_llama import eager_attention_forward


def finite_values(dtype):
    """Every finite value of a 16-bit float dtype, as a CPU tensor of that dtype."""
    bits = torch.arange(0, 1 << 16, dtype=torch.int32).to(torch.int16)
    values = bits.view(dtype)
    return values[torch.isfinite(values.float())]


def bits16(tensor):
    """The 16-bit patterns of a CPU half-precision tensor, as Python ints."""
    return (tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).tolist()


def subnormal(tensor, dtype):
    """Whether each element of ``tensor`` (of ``dtype``) is a nonzero subnormal."""
    wide = tensor.float().abs()
    return (wide > 0) & (wide < torch.finfo(dtype).smallest_normal)


def scalar_probe(devices):
    """Section 1: the Python-float scale against the F32 product rounded once."""
    scale = 1 / math.sqrt(128)
    print(f"== 1. x * (1/sqrt(128)) with a Python float scale ({scale!r}) against (x.float() * s).to(dtype) on the CPU")
    for dtype in (torch.bfloat16, torch.float16):
        x = finite_values(dtype)
        reference = (x.float() * scale).to(dtype)
        rounded_scale = float(torch.tensor(scale, dtype=dtype))
        moved = (x.float() * rounded_scale).to(dtype)
        moved_count = sum(a != b for a, b in zip(bits16(moved), bits16(reference)))
        print(f"{dtype}: {x.numel()} finite inputs; a {dtype}-rounded scale ({rounded_scale!r}) would move {moved_count}")
        product = x.float() * scale
        product_subnormal = (product != 0) & (product.abs() < torch.finfo(torch.float32).smallest_normal)
        for device in devices:
            actual = (x.to(device) * scale).cpu()
            differ = [i for i, (a, b) in enumerate(zip(bits16(actual), bits16(reference))) if a != b]
            flushed = [i for i in differ if product_subnormal[i].item() and actual[i].float().item() == 0.0]
            to_subnormal = sum(subnormal(reference[i:i + 1], dtype).item() for i in flushed)
            other = [i for i in differ if i not in set(flushed)]
            print(f"   {device} {dtype}: {len(differ)} of {x.numel()} differ from the F32 product rounded once; "
                  f"{len(flushed)} have a subnormal F32 product flushed to zero ({to_subnormal} of them round to a "
                  f"subnormal, {len(flushed) - to_subnormal} to the smallest normal), {len(other)} other")
            for i in other[:5]:
                print(f"      other: x={x[i].float().item()!r} {device}={actual[i].float().item()!r} reference={reference[i].float().item()!r}")


def matmul_probe(devices):
    """Section 2: half-precision matrix products against the exact sum rounded once."""
    print("== 2. half-precision matrix products against the exact sum rounded once")
    left = torch.full((1, 4097), 2.0 ** -11, dtype=torch.float64)
    left[0, 0] = 1.0
    right = torch.ones((4097, 1), dtype=torch.float64)
    rows, depth, columns = 16, 256, 16
    grid = lambda count, seed: torch.tensor(  # noqa: E731
        [(((index * 37 + seed * 11) % 129) - 64) / 64 for index in range(count)], dtype=torch.float64)
    lhs, rhs = grid(rows * depth, 1).reshape(rows, depth), grid(depth * columns, 2).reshape(depth, columns)
    exact = lhs @ rhs
    assert torch.equal(exact.float().double(), exact), "the grid sums are not exact in F32"
    for dtype in (torch.float16, torch.bfloat16):
        for device in devices:
            long = (left.to(dtype).to(device) @ right.to(dtype).to(device)).float().item()
            actual = (lhs.to(dtype).to(device) @ rhs.to(dtype).to(device)).cpu()
            expected = exact.float().to(dtype)
            differ = sum(a != b for a, b in zip(bits16(actual), bits16(expected)))
            print(f"   {device} {dtype}: 1 + 4096 * 2^-11 = {long!r}; [{rows}, {depth}] x [{depth}, {columns}]: "
                  f"{differ} of {rows * columns} differ from the exact sum rounded once")


def eager_grid():
    """The ``EagerGrid`` operands of ``src/attention/dispatch_tests.rs`` in F32 arithmetic, as F32 tensors and a mask."""
    batch, heads, kv_heads, queries, keys, head_dim, value_dim = 2, 4, 2, 3, 5, 8, 4

    def grid(count, seed, steps, half_range):
        out = []
        for index in range(count):
            step = (index * 29 + seed * 13) % (2 * steps + 1)
            out.append(np.float32(step - steps) * np.float32(half_range) / np.float32(steps))
        return torch.tensor(np.array(out, dtype=np.float32))

    query = grid(batch * heads * queries * head_dim, 1, 48, 0.75).reshape(batch, heads, queries, head_dim)
    key = grid(batch * kv_heads * keys * head_dim, 2, 48, 0.75).reshape(batch, kv_heads, keys, head_dim)
    value = grid(batch * kv_heads * keys * value_dim, 3, 8, 1.0).reshape(batch, kv_heads, keys, value_dim)
    bias = grid(heads * queries * keys, 4, 16, 0.45).reshape(1, heads, queries, keys)
    excluded = torch.zeros(batch, heads, queries, keys, dtype=torch.bool)
    excluded[0, 1, 0, :] = True
    excluded[1, 2, 2, 4] = True
    for row in range(queries):
        for column in range(keys):
            if column > row + keys - queries:
                excluded[:, :, row, column] = True
    return query, key, value, bias, excluded, kv_heads


class Module(torch.nn.Module):
    """The two attributes ``eager_attention_forward`` reads from an attention module."""

    def __init__(self, groups):
        """Stores the grouped-query repetition and inference mode."""
        super().__init__()
        self.num_key_value_groups = groups
        self.training = False


def eager_probe():
    """Section 3: Transformers' eager attention on the grid, and the issue's emulation."""
    print("== 3. transformers eager_attention_forward (Llama) on EagerGrid, CPU")
    query, key, value, bias, excluded, kv_heads = eager_grid()
    heads = query.shape[1]
    scaling = 8 ** -0.5
    for dtype in (torch.bfloat16, torch.float16):
        q, k, v, b = (tensor.to(dtype) for tensor in (query, key, value, bias))
        # Flex's F32-to-BF16 cast ignores the three lowest F32 bits; list any operand that cast would round differently.
        suspicious = 0
        for tensor in (query, key, value, bias):
            low = (tensor.contiguous().view(torch.int32) & 0xFFFF)
            suspicious += int(((low >= 0x8001) & (low <= 0x8007)).sum())
        mask = torch.where(excluded, torch.tensor(torch.finfo(dtype).min, dtype=dtype), b.expand_as(excluded))
        output, weights = eager_attention_forward(Module(heads // kv_heads), q, k, v, mask, scaling)
        output = output.transpose(1, 2).contiguous()
        # The emulation, written with torch operations.
        k_rep = k.repeat_interleave(heads // kv_heads, dim=1)
        v_rep = v.repeat_interleave(heads // kv_heads, dim=1)
        scores = (q.float() @ k_rep.float().transpose(2, 3)).to(dtype)
        scores = (scores.float() * scaling).to(dtype)
        t = (scores.float() + mask.float()).to(dtype)
        p = torch.softmax(t.float(), dim=-1).to(dtype)
        emulated = (p.float() @ v_rep.float()).to(dtype)
        assert torch.equal(p, weights), "probabilities differ from the emulation"
        full = excluded.all(dim=-1)
        keep = ~full.unsqueeze(-1).expand_as(output)
        differ = int((output.view(torch.int16) != emulated.view(torch.int16))[keep].sum())
        print(f"{dtype}: eager_attention_forward against the emulation: {differ} of {int(keep.sum())} outputs differ "
              f"(the fully masked row excluded); probabilities equal; {suspicious} operand values in Flex's BF16 cast range 0x8001..=0x8007")
        print(f"{dtype} fully masked row (batch 0, head 1, query 0): {[round(x, 6) for x in output[0, 1, 0].float().tolist()]}")
        print(f"{dtype} output bits [batch, head, query, value]: " + " ".join(f"{bit:04x}" for bit in bits16(output.reshape(-1))))


def main():
    """Runs the three sections on every available device."""
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    print(f"torch {torch.__version__}, transformers {transformers.__version__}, numpy {np.__version__}, "
          f"python {platform.python_version()}, {platform.platform()}, devices {devices}")
    scalar_probe(devices)
    matmul_probe(devices)
    eager_probe()


if __name__ == "__main__":
    main()
