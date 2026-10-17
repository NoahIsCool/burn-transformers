# 10. burn-backend: default `linear` reshapes the weight, copying a strided weight on every call

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-backend` (`ops::modules::linear`), seen on Flex |
| Version reproduced | 0.21.0 (this crate's pin) |
| Kind | Performance (a full pass over the weight per call) |
| Upstream status, checked 2026-10-16 | Partly addressed on `main`: the default `linear` now chooses between flattening `x` (`MergeBatches`) and the old weight-unsqueeze path through a `MatmulTransformPolicy`. Re-run the repro on v0.22.0 to see which path a transposed weight takes; file only if it still copies. |
| Repro | [`repros/linear-weight-reshape-copy`](repros/linear-weight-reshape-copy) |
| Patch | [`patches/10-linear-weight-reshape-copy.patch`](patches/10-linear-weight-reshape-copy.patch) (`crates/burn-backend/src/backend/ops/modules/linear.rs`) |
| Patch verified | Yes: decode readout with a 594 MiB transposed table 94.6 ms → 25.8 ms (one token) and 101.5 ms → 17.3 ms (16 tokens); values and gradients (`x`, weight, bias; contiguous and transposed weights, rank 4) agree to 1.4e-6 |
| Workarounds an upstream fix removes | `operations::matmul_transposed` (`src/operations.rs`) and its tied-readout callers (`src/models/muse_glimmer/model.rs`, `src/models/qwen3_5/model.rs`); the plain flattened product in `operations::linear` and in the FP32 router (`src/moe/router.rs`), which also cites this copy |

## Issue draft

**Title:** Default `ModuleOps::linear` reshapes the weight to `[1, .., d_in, d_out]`; on Flex a transposed weight is copied on every call

### Describe the bug

`burn_backend::ops::modules::linear::linear` prepends unit axes to the weight
(`unsqueeze_leading`, a `float_reshape`) so that a rank-`D` activation can use a batched
matmul. Weights are often strided views: a checkpoint's `[out, in]` matrix used as
`[in, out]` (`LinearLayout::Col`), or a tied embedding table used as `Eᵀ`. On Flex a
reshape of a non-contiguous tensor copies it, so every call copies the whole weight
(594 MiB for a 151,936 x 1,024 F32 readout). `linear_x_backward` does the same to
`weightᵀ`, and `linear_weight_backward` computes one product per batch element and sums.

### To reproduce

Standalone project: `repros/linear-weight-reshape-copy` (`burn = "=0.21.0"`, features
`flex`, `autodiff`). It times `module::linear(x, table.transpose(), None)` for `x` of
`[1, 1, 1024]` and `[1, 16, 1024]`, against a contiguous copy of the weight and against
the same product with `x` flattened to `[rows, 1024]`, and checks values (and, on
`Autodiff<Flex>`, gradients) against an explicit broadcast matmul.

### Expected behavior

`module::linear` costs about the same as the flattened product.

### Actual behavior (Apple M3 Max, median µs)

| tokens | `module::linear`, transposed view | `module::linear`, contiguous | flattened, transposed view |
|---:|---:|---:|---:|
| 1 | 94,559 | 15,202 | 24,302 |
| 16 | 101,469 | 18,998 | 17,305 |

### Root cause

`crates/burn-backend/src/backend/ops/modules/linear.rs`:

```rust
let weight = unsqueeze_leading::<B>(weight, x_ndims);   // float_reshape of a strided view
let output = B::float_matmul(x, weight);
```

and Flex's `Layout::reshape` returns `None` for any non-contiguous layout, so
`FlexTensor::reshape` falls back to `to_contiguous()`.

### Proposed fix

Never reshape the weight; flatten the activation's leading axes instead (a view for the
usual contiguous activation), and restore them afterwards:

```rust
let output = B::float_matmul(flatten_rows::<B>(x), weight);              // [rows, d_in] @ [d_in, d_out]
let output = bias.map_or(output, |b| B::float_add(output, B::float_reshape(b, [1, d_out])));
restore_rows::<B>(output, &x_shape, d_out)
```

The backward functions use the same flattening: `dx = grad_rows @ weightᵀ` (a transpose
view, no reshape) and `dW = x_rowsᵀ @ grad_rows` (one product instead of a per-batch
product plus a sum). Measured with the patch: 25,828 µs (1 token) and 17,284 µs
(16 tokens) for the transposed view; gradients agree with the explicit broadcast matmul to
1.4e-6 (summation order changes in `dW`). A complementary Flex fix would let
`Layout::reshape` express unit-axis insertion on strided layouts as a view.

### Environment

burn 0.21.0, Flex backend, Rust 1.94 nightly, Apple M3 Max, macOS 27.0.1.

## Notes for this crate

- The remaining gap between the transposed view (25.8 ms) and the contiguous weight
  (15-18 ms) for one token is the gemm single-row strided-operand slowdown recorded in
  `src/checkpoint/runtime/storage_order.rs`; the crate's contiguous storage order for
  loaded weights addresses that separately.
