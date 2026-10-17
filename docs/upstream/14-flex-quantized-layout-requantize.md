# 14. burn-flex: layout ops on block-quantized tensors requantize and change values

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-flex` |
| Version reproduced | 0.21.0 (this crate's pin); not reproducible on 0.22.0 |
| Kind | Correctness on 0.21.0 (a transpose changes a small row's values by 39% (Q8) or 100% (Q4)) |
| Upstream status, checked 2026-10-16 | **Fixed in v0.22.0; do not file.** v0.22.0's `q_swap_dims` and `q_permute` call `scheme.swap_block_dims` / `permute_block_dims` before `block_safe_layout_op`, whose documentation now says a permuted tensor's blocks are permuted with it, as every other backend keeps them; Flex quantize and dequantize index through a multi-axis block layout. Measured with the repro ported to v0.22.0: the transpose changes no value (0.0 for Q8S Native, Q8S packed and Q4S packed; 0.21.0: 0.39, 0.39 and 1.0). |
| Repro | [`repros/flex-quantized-layout-requantize`](repros/flex-quantized-layout-requantize) (0.21.0) |
| Patch | None: on 0.21.0 this needed a design change in `FlexQTensor`, which v0.22.0 made |
| Patch verified | Not applicable; the effect is measured |
| Workarounds an upstream fix removes | `packed_layout_is_exact` and the `PackedLayoutPolicy` fallback in `src/checkpoint/runtime/packed_install.rs`, which probe for a requantizing transpose and install exact dequantized values instead (losing the packed memory savings on Flex). After upgrading, the probe reports exact on Flex and packed weights stay packed; the probe itself can stay as a cheap guard. |

## What remains on v0.22.0 (measured, not filed)

A probe on v0.22.0 (Q4S, `[1, 32]` blocks, `[4, 64]` with rows and halves of very different
magnitude) found every block-aligned layout op exact: transpose, row slices, block-aligned
column slices, flips of either axis, reshapes to `[8, 32]` and `[2, 128]`, and row
`select`. Two residues are inherent or performance-only, so they are not drafted:

- A slice or `select` that cuts across blocks along the blocked axis (columns `16..48`)
  requantizes the new blocks, changing values by up to 2% of the tensor's maximum. A
  selection that leaves a partial block (three columns) panics with "tensor Shape { dims:
  [4, 3] } is not a whole number of BlockSize([32]) blocks", which is at least explicit.
- Every layout op on a block-quantized Flex tensor still dequantizes and requantizes the
  whole tensor: transposing a `[4096, 4096]` Q4S tensor takes about 188 ms (a metadata-only
  view on CubeCL). For this crate that is a one-time cost per column-layout weight at load.

## Issue draft (not to be filed: fixed in v0.22.0)

Kept as the record of the 0.21.0 behaviour this crate works around.

**Title:** burn-flex: `swap_dims`/`permute`/`slice`/... on block-quantized tensors dequantize and requantize, so a transpose changes the values

### Describe the bug

For `QuantLevel::Block` tensors, Flex 0.21.0 implements `q_swap_dims`, `q_permute`,
`q_flip`, `q_expand`, `q_slice`, `q_reshape` and `q_select` by dequantizing, applying the
layout op to the floats, and quantizing again with fresh blocks along the new last axis. On
every other backend a transpose is metadata-only. Here it changes the values: each new block
mixes elements that belonged to different original blocks, so a row of small magnitude
shares a scale with a large one and loses its precision. The canonical case is a
`LinearLayout::Col` weight loaded from a packed `[out, in]` checkpoint and used as its
transpose.

### To Reproduce

Standalone project: `repros/flex-quantized-layout-requantize` (`burn = "=0.21.0"`, feature
`flex`). A `[2, 64]` weight whose second row is 100x smaller than the first is quantized
with 32-element blocks along each row; the repro compares `w.transpose().dequantize()` with
`w.dequantize().transpose()`.

### Actual behavior

```text
Q8S Native:       original row 0 (scale 1): max relative change 0.0e0
Q8S Native:       original row 1 (scale 0.01): max relative change 3.9e-1
Q8S PackedU32(0): original row 1 (scale 0.01): max relative change 3.9e-1
Q4S PackedU32(0): original row 1 (scale 0.01): max relative change 1.0e0
```

### Root cause

`crates/burn-flex/src/ops/qtensor.rs` 0.21.0, `block_safe_layout_op`: for
`QuantLevel::Block(_)` it calls `Flex::dequantize`, applies the op, and
`Flex::quantize_dynamic(result, &scheme)` with the unchanged scheme, so the new blocks run
along the new last axis. `FlexQTensor` stores `scales: Vec<f32>` without a layout, so it
could not describe blocks that run along an axis other than the last one.

### Proposed fix (as drafted for 0.21.0)

Give `FlexQTensor` a layout for its scales (or the block axis and block strides), as
CubeCL's `qparams.scales.metadata` does, so that `swap_dims`/`permute` (and slices aligned
to block boundaries) keep the original blocks. v0.22.0 does this by permuting the block
shape with the data before requantizing.

### Environment

burn 0.21.0, Flex backend, Rust 1.94 nightly, Apple M3 Max, macOS 27.0.1.
