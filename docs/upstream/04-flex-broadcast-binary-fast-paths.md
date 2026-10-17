# 04. burn-flex: F32 binary ops fall back to a scalar strided walk for common broadcasts

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-flex` |
| Version reproduced | 0.21.0 (this crate's pin) |
| Kind | Performance (30-70x slower than the dense equivalent) |
| Upstream status, checked 2026-10-16 | Reworked on `main`: the `SharedRow`/`PerRowScalar` paths were removed in favour of a collapsed loop nest (`crate::zip`) that is documented to handle every stride pattern. **Re-measure with the repro against v0.22.0 before filing**; file only if the ratios below persist. |
| Repro | [`repros/flex-broadcast-binary-fast-paths`](repros/flex-broadcast-binary-fast-paths) |
| Patch | [`patches/04-flex-broadcast-binary-fast-paths.patch`](patches/04-flex-broadcast-binary-fast-paths.patch) (`crates/burn-flex/src/ops/binary.rs`) |
| Patch verified | Yes: RoPE product 19.6 ms → 0.38 ms, outer product 1.9 ms → 0.05 ms, results bit-identical |
| Workarounds an upstream fix removes | The fused split-half RoPE kernel `src/position/tensor/flex.rs`; the outer-product-as-matmul `input_update` in `src/mamba2.rs` (`recurrent_scan`); the flattened `SegmentMasks` layout in `src/mamba2.rs` |

## Issue draft

**Title:** burn-flex: F32 add/sub/mul/div take a scalar `StridedIter` path for broadcasts other than "shared row" and "per-row scalar"

### Describe the bug

`binary_op_f32` accelerates exactly two broadcast shapes: one contiguous row shared by
every leading position, and one scalar per row. Everything else, after `expand` turns the
broadcast operand into a stride-0 view, goes to `binary_op_typed`'s general fallback, which
walks both operands element by element with `StridedIter` (and, for a strided left
operand, first copies it with the same kind of walk). Two shapes that dominate
transformer inference miss both fast paths:

- RoPE: queries `[b, h, s, d]` (often the heads-swapped view of `[b, s, h, d]`) times
  cos/sin tables `[b, 1, s, d]` broadcast over heads;
- an outer product `[b, h, 1, n] * [b, h, p, 1]` (a Mamba-2 state update).

### To reproduce

Standalone project: `repros/flex-broadcast-binary-fast-paths` (`burn = "=0.21.0"`,
feature `flex`). Each case times `lhs * rhs` with broadcasting and the same product with
both operands first materialized to the full shape ("dense"), and asserts that both
results are bit-identical.

### Expected behavior

A broadcast product costs about as much as the dense one.

### Actual behavior (Apple M3 Max, median µs)

| case | broadcast | dense | ratio |
|---|---:|---:|---:|
| RoPE `[1,32,512,128]` heads-swapped view `* [1,1,512,128]` | 19,564.5 | 299.5 | 65.3x |
| RoPE `[1,32,512,128]` contiguous `* [1,1,512,128]` | 11,329.3 | 251.7 | 45.0x |
| outer `[1,64,1,128] * [1,64,64,1]` | 1,897.0 | 59.0 | 32.2x |

### Root cause

`crates/burn-flex/src/ops/binary.rs`: `detect_broadcast_pattern` returns `None` unless every
outer axis of the right operand has stride 0 (shared row) or the inner axes are all
stride 0 and the outer axes walk the operand contiguously (per-row scalar). The RoPE table
has a non-zero stride on the batch and sequence axes and stride 0 on heads; the outer
product broadcasts *both* operands. Both reach the `StridedIter` fallback in
`binary_op_typed`.

### Proposed fix

Add a row-wise path for the remaining broadcasts in which each operand's innermost axis is
either contiguous (stride 1) or a repeated scalar (stride 0), with arbitrary outer strides:
iterate the outer index with an odometer, and apply the operation to each row with a
monomorphized, autovectorizable loop (`row op row`, `row op scalar`, `scalar op row`).
Every element is still one F32 operation, so results are bit-identical. The patch also
tries this path before materializing a strided left operand.

Measured with the patch:

| case | broadcast | dense | ratio |
|---|---:|---:|---:|
| RoPE, heads-swapped view | 379.3 | 351.7 | 1.1x |
| RoPE, contiguous | 270.3 | 353.6 | 0.8x |
| outer product | 51.3 | 75.9 | 0.7x |

Rows shorter than eight elements keep the old fallback, and the path is F32-only like the
existing SIMD paths; F16/BF16 could use the same loop with per-element rounding.

### Environment

burn 0.21.0, burn-flex default features, Rust 1.94 nightly, Apple M3 Max, macOS 27.0.1.

## Notes for this crate

- If v0.22.0's `zip` loop nest already reaches dense speed on these shapes, close this
  draft and simply plan to remove the crate's fused RoPE kernel and outer-product
  `input_update` after upgrading (both are bit-identical to the broadcast formulation, so
  their tests can assert equality with it).
