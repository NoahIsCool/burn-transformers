"""Scratch: exhaustively classifies PyTorch's half-precision SiLU rounding on CPU and MPS (not committed).

Usage: python3 -I torch_silu_semantics.py

For every finite BF16 and F16 bit pattern x, compares torch.nn.functional.silu(x)
with (a) the F32 value x * sigmoid(x) rounded once to the dtype, and (b) the
composed graph x * torch.sigmoid(x) with the sigmoid rounded to the dtype first
(Burn's activation::silu order). Prints, per device and dtype, how many entries
F.silu agrees with each, and how often (a) and (b) disagree.
"""

import torch
import torch.nn.functional as F


def all_finite(dtype):
    """Every finite value of a 16-bit float dtype."""
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    values = bits.view(dtype)
    return values[torch.isfinite(values)]


def main():
    print(f"torch {torch.__version__}")
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dtype in (torch.bfloat16, torch.float16):
        x_cpu = all_finite(dtype)
        once_cpu = (x_cpu.float() * torch.sigmoid(x_cpu.float())).to(dtype)
        for device in devices:
            x = x_cpu.to(device)
            fused = F.silu(x).cpu()
            composed = (x * torch.sigmoid(x)).cpu()
            once = once_cpu
            same = lambda a, b: int(((a == b) | (torch.isnan(a) & torch.isnan(b))).sum())
            n = x.numel()
            print(f"{device:4s} {str(dtype):15s} n={n}: F.silu == once-rounded F32 {same(fused, once)}; "
                  f"F.silu == composed {same(fused, composed)}; composed == once {same(composed, once)}; "
                  f"composed differs on {n - same(composed, once)} ({100 * (n - same(composed, once)) / n:.2f}%)")
            # Restrict to the activation range a gate projection occupies.
            mask = (x_cpu.abs() <= 16)
            m = int(mask.sum())
            d = int((composed[mask] != once[mask]).sum())
            print(f"     |x| <= 16: n={m}, composed differs from once-rounded on {d} ({100 * d / m:.2f}%); "
                  f"F.silu differs from once-rounded on {m - same(fused[mask], once[mask])}")


if __name__ == "__main__":
    main()
