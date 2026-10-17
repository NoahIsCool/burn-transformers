# 14. burn-flex: layout ops on block-quantized tensors requantize and change values

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-flex` |
| Version reproduced | 0.21.0 (this crate's pin) |
| Kind | Correctness (a transpose changes a small row's values by 39% (Q8) or 100% (Q4)) |
| Upstream status, checked 2026-10-16 | **Not fixed** on `main`: `q_swap_dims` still goes through `block_safe_layout_op`, which dequantizes, transposes and calls `quantize_dynamic`. File this one. |
| Repro | [`repros/flex-quantized-layout-requantize`](repros/flex-quantized-layout-requantize) |
| Patch | None: this needs a design change in `FlexQTensor` (see "Proposed fix") |
| Patch verified | Not applicable; the effect is measured |
| Workarounds an upstream fix removes | `packed_layout_is_exact` and the `PackedLayoutPolicy` fallback in `src/checkpoint/runtime/packed_install.rs`, which probe for a requantizing transpose and install exact dequantized values instead (losing the packed memory savings on Flex) |

## Issue draft

**Title:** burn-flex: `swap_dims`/`permute`/`slice`/... on block-quantized tensors dequantize and requantize, so a transpose changes the values

### Describe the bug

For `QuantLevel::Block` tensors, Flex implements `q_swap_dims`, `q_permute`, `q_flip`,
`q_expand`, `q_slice`, `q_reshape` and `q_select` by dequantizing, applying the layout op
to the floats, and quantizing again with fresh blocks along the new last axis. On every
other backend a transpose is metadata-only. Here it changes the values: each new block
mixes elements that belonged to different original blocks, so a row of small magnitude
shares a scale with a large one and loses its precision. The canonical case is a
`LinearLayout::Col` weight loaded from a packed `[out, in]` checkpoint and used as its
transpose.

### To reproduce

Standalone project: `repros/flex-quantized-layout-requantize` (`burn = "=0.21.0"`,
feature `flex`). A `[2, 64]` weight whose second row is 100x smaller than the first is
quantized with 32-element blocks along each row; the repro compares
`w.transpose().dequantize()` with `w.dequantize().transpose()`.

### Actual behavior

```text
Q8S Native:       original row 0 (scale 1): max relative change 0.0e0
Q8S Native:       original row 1 (scale 0.01): max relative change 3.9e-1
Q8S PackedU32(0): original row 1 (scale 0.01): max relative change 3.9e-1
Q4S PackedU32(0): original row 1 (scale 0.01): max relative change 1.0e0
```

### Root cause

`crates/burn-flex/src/ops/qtensor.rs`, `block_safe_layout_op`: for `QuantLevel::Block(_)`
it calls `Flex::dequantize`, applies the op, and `Flex::quantize_dynamic(result, &scheme)`.
`FlexQTensor` stores `scales: Vec<f32>` without a layout, so it cannot describe blocks that
run along an axis other than the last one.

### Proposed fix

Give `FlexQTensor` a layout for its scales (or the block axis and block strides), as
CubeCL's `qparams.scales.metadata` does, so that `swap_dims`/`permute` (and slices aligned
to block boundaries) stay views of the same codes and scales, and dequantization and the
quantized matmul read blocks through that layout. Until then, a documented error or
at least a doc comment on the requantizing behavior would prevent silent precision loss.

### Environment

burn 0.21.0, Flex backend, Rust 1.94 nightly, Apple M3 Max, macOS 27.0.1.
