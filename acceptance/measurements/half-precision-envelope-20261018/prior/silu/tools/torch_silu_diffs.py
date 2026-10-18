"""Scratch: lists the inputs where PyTorch's F.silu differs from the once-rounded F32 value x*sigmoid(x) and from x/(1+exp(-x)) (not committed)."""
import torch
import torch.nn.functional as F


def all_finite(dtype):
    """Every finite value of a 16-bit float dtype."""
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    values = bits.view(dtype)
    return values[torch.isfinite(values)]


for dtype in (torch.bfloat16, torch.float16):
    x = all_finite(dtype)
    xf = x.float()
    xd = x.double()
    exact = (xd / (1 + torch.exp(-xd)))
    once_mul = (xf * torch.sigmoid(xf)).to(dtype)
    once_div = (xf / (1 + torch.exp(-xf))).to(dtype)
    correct = exact.to(dtype)  # correctly rounded from F64
    for device in ("cpu", "mps"):
        fused = F.silu(x.to(device)).cpu()
        diff = (fused != once_mul)
        ulp_err = lambda v: ((v.double() - exact).abs())
        print(f"{device} {dtype}: F.silu != once(x*sigmoid) on {int(diff.sum())}; != once(x/(1+exp(-x))) on {int((fused != once_div).sum())}; "
              f"!= correctly rounded F64 on {int((fused != correct).sum())}; once(x*sigmoid) != correctly rounded on {int((once_mul != correct).sum())}")
        idx = diff.nonzero().flatten()[:6]
        for i in idx.tolist():
            print(f"    x={x[i].item()!r}: F.silu={fused[i].item()!r} once={once_mul[i].item()!r} exact={exact[i].item()!r}")
        if device == "mps" and dtype == torch.bfloat16:
            xs = x[diff]
            print(f"    differing |x| range: min {xs.abs().min().item()} max {xs.abs().max().item()}; F.silu err vs exact max {ulp_err(fused)[diff].max().item():.3e}, once err max {ulp_err(once_mul)[diff].max().item():.3e}")
