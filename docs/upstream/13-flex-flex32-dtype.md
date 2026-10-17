# 13. burn-flex: `Flex32` tensors can be created but every kernel panics on them

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-flex` (and, in 0.21.0, `burn-backend`'s `TensorData`) |
| Version reproduced | 0.21.0 (this crate's pin) and **0.22.0** |
| Kind | Correctness (panics on valid requests) |
| Upstream status, checked 2026-10-16 | **Open, narrower in v0.22.0.** In v0.22.0, `Tensor::from_data` converts `Flex32` data to the device's `F32`, so data no longer brings the tag into Flex. But `zeros`/`empty` with `FloatDType::Flex32` still create a `Flex32`-tagged Flex tensor, all 9 operations tried panic on it (measured below), and `float_ones`/`float_full` still end in `unreachable!()`. Same Flex code on `main` at `df0812e`, where [#6069](https://github.com/tracel-ai/burn/pull/6069) ([burn#6068](https://github.com/tracel-ai/burn/issues/6068)) and [#6056](https://github.com/tracel-ai/burn/pull/6056) ([burn#6055](https://github.com/tracel-ai/burn/issues/6055)) since made `TensorData` and CubeCL keep the `Flex32` marker, so more `Flex32` data may reach Flex there. File this one. |
| Related upstream issues | burn#6055 / #6056 and burn#6068 / #6069 (Flex32 data on `main`; neither touches Flex's creation ops) |
| Repro | [`repros/v0.22.0/flex-flex32-dtype`](repros/v0.22.0/flex-flex32-dtype) (to file); [`repros/flex-flex32-dtype`](repros/flex-flex32-dtype) (0.21.0) |
| Patch | To file: [`patches/v0.22.0/13-flex-flex32-dtype.patch`](patches/v0.22.0/13-flex-flex32-dtype.patch) (`crates/burn-flex/src/ops/float.rs`; applies to `main` unchanged). For this crate's pin: [`patches/13-flex-flex32-dtype.patch`](patches/13-flex-flex32-dtype.patch) (v0.21.0, also `crates/burn-backend/src/data/tensor.rs`; its `storage_dtype` helper now sits above the test-module comment block) |
| Patch verified | Yes, Apple M3 Max: v0.22.0 9 of 18 operations panic (all 9 on the `zeros` tensor) → 0 of 18; v0.21.0 9 of 9 → 0 of 9, where `TensorData` tagged `Flex32` also reads as `f32` with the patch; burn-flex's 425 unit tests pass with the v0.22.0 patch |
| Workarounds an upstream fix removes | The Flex32 top-k/top-p guard in `src/generation.rs` (`process_with_history`, module docs) and its test `flex32_sort_filters_fail_closed_before_backend_operations` (`src/generation_tests.rs`), which would then need to expect F32 results |

## Issue draft

**Title:** burn-flex: `zeros`/`empty` with `FloatDType::Flex32` create tensors that every Flex kernel panics on

### Describe the bug

On Flex, `Tensor::zeros(shape, (&device, DType::Flex32))` (and `empty`) returns a tensor
tagged `DType::Flex32`. Flex kernels dispatch only on `F32`/`F64`/`F16`/`BF16`, so almost
everything on that tensor panics: `sort`, `argsort`, `topk`, `+`, `sum`, `softmax`,
`matmul`, `ones_like` (`float_ones` reaches `unreachable!()`), and even `cast(F32)`.
`float_random`, and since v0.22.0 `TensorData` conversion, already map a `Flex32` request to
`F32`, so the intent is clearly "Flex32 is F32 on the CPU". A model written against
CubeCL's `flex32` that is run on Flex for testing hits this on its first `zeros`.

### To Reproduce

Standalone project `repros/v0.22.0/flex-flex32-dtype` (`burn = "=0.22.0"`, feature `flex`),
`cargo run --release`. It creates a Flex tensor from `flex32`-element `TensorData` and one
with `zeros(.., DType::Flex32)`, and runs 9 operations on each:

```rust
let device = Device::flex();
let zeros = Tensor::<2>::zeros([1, 4], (&device, DType::Flex32));
zeros.dtype();            // Flex32
zeros.clone().sort(1);    // panics; so do topk, argsort, +, sum, softmax, matmul, ones_like, cast
```

### Expected behavior

The tensor is stored as `F32` (as `float_random` does) and every operation works.

### Actual behavior (burn 0.22.0)

```text
TensorData from flex32 elements: dtype Flex32
from_data(flex32 elements) on Flex: tensor dtype F32
  sort / topk(2) / argsort / add / sum / softmax / matmul / ones_like + 1 / cast(F32): ok
zeros(.., Flex32) on Flex: tensor dtype Flex32
  sort / topk(2) / argsort / add / sum / softmax / matmul / ones_like + 1 / cast(F32): panicked
  TensorData(flex32).try_to_vec::<f32>(): Err(DTypeMismatch { expected: F32, actual: Flex32 })
9 of 18 operations panicked
```

### Root cause

`crates/burn-flex/src/ops/float.rs`: `float_empty`/`float_zeros` convert
`FloatDType::Flex32` to `DType::Flex32`; `float_ones`/`float_full` match only the four
concrete types and end in `unreachable!()`; `float_cast` has no `Flex32` target.

### Proposed fix

Store `Flex32` as `F32` everywhere, like `float_random` already does: a
`storage_dtype(FloatDType) -> DType` helper used by `float_empty`, `float_zeros`,
`float_ones`, `float_full` and `float_cast`. With the patch the tensor above is `F32` and
every operation succeeds with the expected values.

### Environment

burn 0.22.0, Flex backend, Rust 1.99.1, Apple M3 Max, macOS 27.0.1.

## Notes for this crate

- On the crate's pin (0.21.0) the problem is wider: `Flex32`-tagged `TensorData` keeps its
  tag through `float_from_data`, and `TensorData::to_vec::<f32>()` rejects it, which also
  breaks Burn's backend-agnostic fallback sort for any backend that keeps the tag. The
  v0.21.0 patch also re-tags data in `float_from_data` and lets `matches_target_dtype` accept
  `F32` for `Flex32` data.
- v0.22.0 still rejects `try_to_vec::<f32>()` on `flex32` data, and its fallback sort
  (`burn_backend::ops::sort`) still dispatches `Flex32` to the `f32` kernel. That only
  matters on a backend that keeps the marker (CubeCL on `main` after #6056); it was not
  reproduced, so it is not part of this draft.
- The Metal backend with the default `f32` element rejects `Flex32` data in
  `float_from_data` with "Unsupported dtype" (0.21.0), which is at least explicit.
