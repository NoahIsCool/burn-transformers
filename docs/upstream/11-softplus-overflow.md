# 11. burn-tensor: `activation::softplus` overflows to `inf` (gradient `NaN`) and underflows to 0

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-tensor` (`tensor::activation`) |
| Version reproduced | 0.21.0 (this crate's pin) |
| Kind | Correctness (non-finite values and gradients for finite inputs) |
| Upstream status, checked 2026-10-16 | **Fixed in v0.22.0** (`softplus` calls `softplus_with_threshold(tensor, beta, DEFAULT_SOFTPLUS_THRESHOLD)`). Do not file; kept as a record. |
| Repro | [`repros/softplus-overflow`](repros/softplus-overflow) |
| Patch | [`patches/11-softplus-overflow.patch`](patches/11-softplus-overflow.patch) (`crates/burn-tensor/src/tensor/activation/base.rs`) |
| Patch verified | Yes: 4 of 10 inputs with a wrong value or gradient → 0 of 10 |
| Workarounds an upstream fix removes | `delta_net::softplus` (`src/delta_net.rs`), PyTorch's thresholded softplus, also used by `src/mamba2.rs` for `dt` |

## Issue draft

**Title:** `activation::softplus` returns `inf` (and a `NaN` gradient) for `beta * x > 88.7` and 0 for `beta * x < -17` in F32

### Describe the bug

`softplus(x, beta)` evaluates `log(exp(beta * x) + 1) / beta`. In F32, `exp` overflows
above 88.72, so the value is `inf` and its gradient `exp/(exp + 1)` is `inf/inf = NaN`;
below about -17, `exp(x) + 1` rounds to 1 and the value collapses to 0. Models such as
Mamba-2 and Gated DeltaNet compute `softplus(dt + dt_bias)` on activations and turn one
`inf` into `NaN` state for the rest of the sequence.

### To reproduce

Standalone project: `repros/softplus-overflow` (`burn = "=0.21.0"`, features `flex`,
`autodiff`): values and gradients for `x ∈ {-30, -20, -1, 0, 1, 20, 50, 88, 89, 100}`
against PyTorch's definition `x if x > 20 else log1p(exp(x))`.

### Actual behavior

```text
       x     burn value      reference    burn grad    reference
     -30     0.000000e0   9.357623e-14     0.000000     0.000000   <-- wrong
     -20     0.000000e0    2.061154e-9     0.000000     0.000000   <-- wrong
      89            inf     8.900000e1          NaN     1.000000   <-- wrong
     100            inf     1.000000e2          NaN     1.000000   <-- wrong
4 of 10 inputs have a wrong value or gradient
```

### Proposed fix

Match PyTorch: linear above `beta * x > 20`, `log1p(exp(beta * x)) / beta` below, clamping
before `exp` so the unselected branch and its gradient stay finite:

```rust
let scaled = tensor.clone().mul_scalar(beta);
let linear = scaled.clone().greater_elem(20.0);
scaled.clamp_max(20.0).exp().log1p().div_scalar(beta).mask_where(linear, tensor)
```

With the patch all ten values and gradients match. (Burn v0.22.0 already ships
`softplus_with_threshold`.)

## Notes for this crate

- After upgrading to v0.22.0, compare `delta_net::softplus` with Burn's `softplus` (same
  threshold, 20) bit for bit before replacing it; the crate's version also evaluates the
  reduced-precision rounding points described in `src/mamba2.rs`.
