# 13. burn-flex: `Flex32` tensors are accepted but every kernel panics on them

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crates `burn-flex` and `burn-backend` (`TensorData`) |
| Version reproduced | 0.21.0 (this crate's pin) |
| Kind | Correctness (panics on valid requests) |
| Upstream status, checked 2026-10-16 | **Not fixed** on `main`: only `float_random` maps `Flex32` to F32; `float_from_data` keeps the tag and `float_ones`/`float_full` still end in `unreachable!()`. File this one. |
| Repro | [`repros/flex-flex32-dtype`](repros/flex-flex32-dtype) |
| Patch | [`patches/13-flex-flex32-dtype.patch`](patches/13-flex-flex32-dtype.patch) (`crates/burn-flex/src/ops/float.rs`, `crates/burn-backend/src/data/tensor.rs`) |
| Patch verified | Yes: 9 of 9 operations panic → 0 of 9; `TensorData` tagged `Flex32` reads as `f32` |
| Workarounds an upstream fix removes | The Flex32 top-k/top-p guard in `src/generation.rs` (`process_with_history`, module docs) and its test `flex32_sort_filters_fail_closed_before_backend_operations` (`src/generation_tests.rs`), which would then need to expect F32 results |

## Issue draft

**Title:** burn-flex: `DType::Flex32` tensors can be created but sort, top-k, arithmetic, reductions and `cast` all panic ("unsupported dtype Flex32")

### Describe the bug

Flex accepts `Tensor::from_data(data, (&device, DType::Flex32))` for `Flex32`-tagged data
(read back from, or saved by, a CubeCL `flex32` model) and
`Tensor::zeros(shape, (&device, DType::Flex32))`, and keeps the `Flex32` tag. Its kernels
dispatch only on `F32`/`F64`/`F16`/`BF16`, so almost everything then panics: `sort`,
`argsort`, `topk`, `+`, `exp`, `sum`, `softmax`, `matmul`, `ones_like` (`float_ones`
reaches `unreachable!()`), and even `cast(F32)`. `float_random` already maps a `Flex32`
request to F32 storage, so the intent is clearly "Flex32 is F32 on the CPU".

Separately, `TensorData::as_slice::<f32>()` / `to_vec::<f32>()` reject `Flex32` data
although its bytes are `f32`, which also breaks the backend-agnostic fallback sort
(`burn_backend::ops::sort`, which dispatches `Flex32` to `f32` and then calls
`as_mut_slice::<f32>().unwrap()`) for any backend that keeps the tag.

### To reproduce

Standalone project: `repros/flex-flex32-dtype` (`burn = "=0.21.0"`, feature `flex`).

### Actual behavior

```text
Flex32 data loaded as Flex32 on Flex: tensor dtype Flex32
zeros requested as Flex32 on Flex: tensor dtype Flex32
  sort / topk(2) / argsort / add / sum / softmax / matmul / ones_like + 1 / cast(F32): panicked
  TensorData(Flex32).to_vec::<f32>(): Err(TypeMismatch("Invalid target element type (expected Flex32, got F32)"))
9 of 9 operations panicked
```

### Root cause

- `crates/burn-flex/src/ops/float.rs`: `float_from_data` passes the `Flex32` tag through;
  `float_empty`/`float_zeros` convert `FloatDType::Flex32` to `DType::Flex32`;
  `float_ones`/`float_full` match only the four concrete types; `float_cast` has no
  `Flex32` target.
- `crates/burn-backend/src/data/tensor.rs`, `TensorData::matches_target_dtype` treats
  `Bool(U8)`/`Bool(U32)` storage as aliases but not `Flex32` as `f32`.

### Proposed fix

- Flex stores `Flex32` as `F32` everywhere, like `float_random` already does: a
  `storage_dtype(FloatDType) -> DType` helper used by `float_empty`, `float_zeros`,
  `float_ones`, `float_full` and `float_cast`, and `float_from_data` re-tags `Flex32` data
  as `F32` (same bytes).
- `matches_target_dtype` accepts `F32` (and `Flex32`) as targets for `Flex32` data.

With the patch the tensors above are `F32`, every operation succeeds with the expected
values, and the `TensorData` read returns `[1.0, 2.0]`.

### Environment

burn 0.21.0, Flex backend, Rust 1.94 nightly, Apple M3 Max, macOS 27.0.1. (The Metal
backend with the default `f32` element rejects `Flex32` data in `float_from_data` with
"Unsupported dtype", which is at least explicit.)
