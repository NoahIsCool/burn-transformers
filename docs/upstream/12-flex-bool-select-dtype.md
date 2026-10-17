# 12. burn-flex: Boolean `select`/`gather` return tensors tagged `U8`

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-flex` |
| Version reproduced | 0.21.0 (this crate's pin) |
| Kind | Correctness (panics and read errors on valid Boolean tensors) |
| Upstream status, checked 2026-10-16 | Probably **not fixed**: on `main`, `bool_select` and `bool_gather` still return `select::<u8>` / `gather_bool` directly without re-tagging (the helpers themselves were not inspected). Re-run the repro on v0.22.0 before filing. |
| Repro | [`repros/flex-bool-select-dtype`](repros/flex-bool-select-dtype) |
| Patch | [`patches/12-flex-bool-select-dtype.patch`](patches/12-flex-bool-select-dtype.patch) (`crates/burn-flex/src/ops/bool.rs`) |
| Patch verified | Yes: 4 of 4 Boolean reads fail → 0 of 4 |
| Workarounds an upstream fix removes | `select_bool_tensor` in `src/architecture/types.rs`, which selects through an integer round trip, and the integer-based Boolean readback helper next to it |

## Issue draft

**Title:** burn-flex: `Tensor<Bool>::select` / `gather` produce `DType::U8` tensors, so `cat` panics and `to_vec::<bool>()` fails

### Describe the bug

On Flex, `bool_select`, `bool_gather` and `bool_select_or` reuse the numeric `u8`
kernels, whose output is labelled `DType::U8` instead of the input's `DType::Bool(..)`.
The bytes are correct, but Boolean consumers that check the dtype fail:
`Tensor::cat` with another Boolean tensor panics ("cat: dtype mismatch: expected U8, got
Bool(Native)"), and `into_data().to_vec::<bool>()` returns `TypeMismatch`.

### To reproduce

Standalone project: `repros/flex-bool-select-dtype` (`burn = "=0.21.0"`, feature `flex`).

```rust
let mask = Tensor::<Flex, 2, Bool>::from_bool(TensorData::from([[true, false], [false, true], [true, true]]), &device);
let selected = mask.clone().select(0, Tensor::<Flex, 1, Int>::from_ints([2, 0], &device));
selected.dtype();                                  // U8, expected Bool(Native)
selected.clone().into_data().to_vec::<bool>();     // Err(TypeMismatch)
Tensor::cat(vec![selected, other_bool_row], 0);    // panics
```

### Actual behavior

```text
input dtype:  Bool(Native)
select dtype: U8
  to_vec::<bool>(): Err(TypeMismatch("Invalid target element type (expected U8, got Bool(Native))"))
  cat with a Bool tensor: panicked
gather dtype: U8
  ... (same)
4 of 4 Boolean reads failed
```

### Root cause

`crates/burn-flex/src/ops/bool.rs`: `bool_select` returns
`gather_scatter::select::<u8>(tensor, dim, indices)` and `bool_gather` returns
`gather_scatter::gather_bool(...)` (= `gather::<u8>`); both create their output with
`E::dtype()`, i.e. `U8`. Other Boolean ops (`bool_not`, comparisons, `scatter_or`) carry
the input's `BoolDType` explicitly.

### Proposed fix

Re-tag the shared kernels' output with the input's Boolean dtype (no copy; the `Arc` is
re-wrapped):

```rust
fn retag_bool(tensor: FlexTensor, dtype: DType) -> FlexTensor {
    FlexTensor::from_arc(tensor.data_arc(), tensor.layout().clone(), dtype)
}
```

applied in `bool_select`, `bool_gather` and `bool_select_or`. With the patch the selected
and gathered tensors are `Bool(Native)`, read back as `[true, true, true, false]` and
`[false, true, false, false, true, true]`, and concatenate with other Boolean tensors.

### Environment

burn 0.21.0, Flex backend, Rust 1.94 nightly, Apple M3 Max, macOS 27.0.1.
