# 10. burn-backend / burn-flex: `linear` reshapes the weight, and Flex copies a strided weight to do it

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn): the default `ModuleOps::linear` in `burn-backend` (`ops::modules::linear`) and `Layout::reshape` in `burn-flex` |
| Version reproduced | 0.21.0 (this crate's pin) and **0.22.0** |
| Kind | Performance (a full pass over the weight per call) |
| Upstream status, checked 2026-10-16 | **Open.** v0.22.0 added a `MatmulTransformPolicy` to the default `linear`, but it keeps the weight-unsqueeze path whenever the input has one batch (`batches == 1`), which is exactly a decode step `[1, tokens, d_in]`; the repro measured on v0.22.0 still copies the weight. `linear.rs` and `layout.rs` on `main` at `df0812e` are identical to v0.22.0. File this one. |
| Repro | [`repros/v0.22.0/linear-weight-reshape-copy`](repros/v0.22.0/linear-weight-reshape-copy) (to file); [`repros/linear-weight-reshape-copy`](repros/linear-weight-reshape-copy) (0.21.0) |
| Patch | To file: [`patches/v0.22.0/10-flex-reshape-unit-axes.patch`](patches/v0.22.0/10-flex-reshape-unit-axes.patch) (`crates/burn-flex/src/layout.rs`; applies to `main` unchanged). For this crate's pin: [`patches/10-linear-weight-reshape-copy.patch`](patches/10-linear-weight-reshape-copy.patch) (v0.21.0, `crates/burn-backend/src/backend/ops/modules/linear.rs`: flatten the activation instead of reshaping the weight; now formatted with Burn's `max_width = 100`) |
| Patch verified | Yes, Apple M3 Max, decode readout through a 594 MiB transposed table. v0.22.0 with the Flex patch: 31.5 ms → 11.7 ms (1 token), 28.8 ms → 13.4 ms (16 tokens), gradients identical to an explicit broadcast matmul; burn-flex's 425 unit tests pass. v0.21.0 with the burn-backend patch: 102.2 ms → 26.5 ms (1 token), 97.3 ms → 13.4 ms (16 tokens), gradients within 1.4e-6. |
| Workarounds an upstream fix removes | `operations::matmul_transposed` (`src/operations.rs`) and its tied-readout callers (`src/models/muse_glimmer/model.rs`, `src/models/qwen3_5/model.rs`); the plain flattened product in `operations::linear` and in the FP32 router (`src/moe/router.rs`), which also cites this copy |

Timings are medians of one run on a shared machine (a 594 MiB copy costs 15-20 ms on it).

## Issue draft

**Title:** burn-flex: reshaping a strided tensor that only gains unit axes copies it, so `module::linear` copies a transposed weight on every call

### Describe the bug

`burn::tensor::module::linear(x, weight, bias)` with `x` of shape `[1, tokens, d_in]` (a
decode step) prepends unit axes to the weight, `[d_in, d_out]` → `[1, d_in, d_out]`, before
a batched matmul: v0.22.0's `MatmulTransformPolicy` keeps that batched form whenever there
is one batch. Weights are often strided views: a checkpoint's `[out, in]` matrix used as
`[in, out]` (`LinearLayout::Col`), or a tied embedding table used as `Eᵀ`. Flex's
`Layout::reshape` returns `None` for every non-contiguous layout, so `FlexTensor::reshape`
copies the whole weight on every call (594 MiB for a 151,936 × 1,024 F32 readout), although
inserting a size-1 axis never needs to move an element.

### To Reproduce

Standalone project `repros/v0.22.0/linear-weight-reshape-copy` (`burn = "=0.22.0"`,
features `flex`, `autodiff`), `cargo run --release`. It times
`module::linear(x, table.transpose(), None)` for `x` of `[1, 1, 1024]` and
`[1, 16, 1024]`, against a contiguous copy of the weight and against the same product with
`x` flattened to `[rows, 1024]`, and first checks values and gradients (`x`, weight, bias;
contiguous and transposed weights, rank 4, on an autodiff device) against an explicit
broadcast matmul.

### Expected behavior

`module::linear` with the transposed view costs about the same as the flattened product.

### Actual behavior (Apple M3 Max, burn 0.22.0, median µs)

| tokens | `module::linear`, transposed view | `module::linear`, contiguous | flattened, transposed view |
|---:|---:|---:|---:|
| 1 | 31,536 | 14,261 | 11,403 |
| 16 | 28,786 | 14,125 | 12,648 |

### Root cause

`crates/burn-backend/src/backend/ops/modules/linear.rs` (`MatmulTransformAction::Keep`,
taken for `batches == 1`):

```rust
let weight = unsqueeze_leading::<B>(weight, ndims);   // float_reshape [d_in, d_out] -> [1, d_in, d_out]
let output = B::float_matmul(x, weight);
```

and `crates/burn-flex/src/layout.rs`:

```rust
pub fn reshape(&self, new_shape: Shape) -> Option<Self> {
    if !self.is_contiguous() {
        return None;              // FlexTensor::reshape then calls to_contiguous()
    }
    ...
```

### Proposed fix

Let `Layout::reshape` express a reshape that only inserts or removes size-1 axes as a view
of any strided layout: keep the non-unit axes, in order, with their strides, and give each
unit axis a positive stride. Every element keeps its address, so this is valid for
transposed, sliced or flipped layouts alike; any other reshape of a non-contiguous layout
still copies. Patch attached (`10-flex-reshape-unit-axes.patch`, a new private
`reshape_unit_axes` helper). Measured with the patch on v0.22.0: 11,666 µs (1 token) and
13,426 µs (16 tokens) for the transposed view, on par with the flattened product; values and
gradients still match the explicit matmul exactly; burn-flex's unit tests pass (Burn's
backend test suite was not run).

An alternative, local to `linear`, is to treat `batches == 1` as `MergeBatches`: with a
single batch the merge is a free view of `x` and changes no tiling. That alone does not help
other callers that add unit axes to strided tensors.

### Environment

burn 0.22.0, Flex backend, Rust 1.99.1, Apple M3 Max, macOS 27.0.1.

## Notes for this crate

- On the crate's pin (0.21.0) the default `linear` always took the unsqueeze path; the
  v0.21.0 patch flattens the activation instead and changes the backward products to
  `dx = grad_rows @ weightᵀ` and `dW = x_rowsᵀ @ grad_rows`, so it also works on backends
  other than Flex. Either fix makes `matmul_transposed` unnecessary for speed.
- The remaining gap between the transposed view and the contiguous weight for one token on
  0.21.0 (26.5 ms versus 17.2 ms) is the gemm single-row strided-operand slowdown recorded
  in `src/checkpoint/runtime/storage_order.rs`; the crate's contiguous storage order for
  loaded weights addresses that separately.
