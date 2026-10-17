# 12. burn-flex: Boolean `select`/`gather` return tensors tagged `U8`

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-flex` |
| Version reproduced | 0.21.0 (this crate's pin) and **0.22.0** |
| Kind | Correctness (panics and read errors on valid Boolean tensors) |
| Upstream status, checked 2026-10-16 | **Open.** v0.22.0 reproduces it (4 of 4 Boolean reads fail, measured below). On `main` at `df0812e`, `bool_select` still returns `select::<u8>(..)` and `gather_bool` is still `gather::<u8>(..)`; `bool_select_or` (renamed `bool_select_assign` on `main`) already re-tags its output since v0.22.0. File this one. |
| Repro | [`repros/v0.22.0/flex-bool-select-dtype`](repros/v0.22.0/flex-bool-select-dtype) (to file); [`repros/flex-bool-select-dtype`](repros/flex-bool-select-dtype) (0.21.0) |
| Patch | To file: [`patches/v0.22.0/12-flex-bool-select-dtype.patch`](patches/v0.22.0/12-flex-bool-select-dtype.patch) (v0.22.0) and the same change rebased onto `main`, [`patches/v0.22.0/12-flex-bool-select-dtype-main.patch`](patches/v0.22.0/12-flex-bool-select-dtype-main.patch). For this crate's pin: [`patches/12-flex-bool-select-dtype.patch`](patches/12-flex-bool-select-dtype.patch) (v0.21.0, `crates/burn-flex/src/ops/bool.rs`; its `retag_bool` helper now sits above the test-module comment block) |
| Patch verified | Yes, Apple M3 Max: 4 of 4 Boolean reads fail → 0 of 4, on v0.22.0 and on v0.21.0; burn-flex's 425 unit tests pass with the v0.22.0 patch. The `main` patch was checked with `git apply --check` only (changed lines identical to the v0.22.0 patch). |
| Workarounds an upstream fix removes | `select_bool_tensor` in `src/architecture/types.rs`, which selects through an integer round trip, and the integer-based Boolean readback helper next to it |

## Issue draft

**Title:** burn-flex: `Tensor<_, Bool>::select` / `gather` produce `DType::U8` tensors, so `cat` panics and `try_to_vec::<bool>()` fails

### Describe the bug

On Flex, `bool_select` and `bool_gather` reuse the numeric `u8` kernels, whose output is
labelled `DType::U8` instead of the input's `DType::Bool(..)`. The bytes are correct, but
Boolean consumers that check the dtype fail: `Tensor::cat` with another Boolean tensor
panics ("cat: dtype mismatch: expected U8, got Bool(Native)"), and
`into_data().try_to_vec::<bool>()` returns `DTypeMismatch`. `bool_select_or` had the same
problem and was fixed in v0.22.0 by re-tagging its output; `select` and `gather` were not.

### To Reproduce

Standalone project `repros/v0.22.0/flex-bool-select-dtype` (`burn = "=0.22.0"`, feature
`flex`), `cargo run --release`:

```rust
let device = Device::flex();
let mask = Tensor::<2, Bool>::from_bool(
    TensorData::from([[true, false], [false, true], [true, true]]),
    &device,
);
let selected = mask.clone().select(0, Tensor::<1, Int>::from_ints([2, 0], &device));
selected.dtype();                                    // U8, expected Bool(Native)
selected.clone().into_data().try_to_vec::<bool>();   // Err(DTypeMismatch { .. })
Tensor::cat(vec![selected, other_bool_row], 0);      // panics
```

### Expected behavior

`select` and `gather` of a Boolean tensor return a tensor with the input's Boolean dtype.

### Actual behavior (burn 0.22.0)

```text
input dtype:  Bool(Native)
select dtype: U8
  try_to_vec::<bool>(): Err(DTypeMismatch { expected: Bool(Native), actual: U8 })
    panic: assertion `left == right` failed: cat: dtype mismatch: expected U8, got Bool(Native)
  cat with a Bool tensor: panicked
gather dtype: U8
  ... (same)
4 of 4 Boolean reads failed
```

### Root cause

`crates/burn-flex/src/ops/bool.rs`: `bool_select` returns
`gather_scatter::select::<u8>(tensor, dim, indices)`, and `bool_gather` returns
`gather_scatter::gather_bool(..)`, which is `gather::<u8>(..)`; both create their output
with `E::dtype()`, i.e. `U8`. `select_or` in `gather_scatter.rs` already restores the dtype
("The shared update kernel tags its output as u8; retain the input's bool dtype").

### Proposed fix

Re-tag the output the same way `select_or` does (no copy; the `Arc` is re-wrapped): in
`gather_bool`, and in a new `select_bool` that `bool_select` calls.

```rust
pub fn select_bool(tensor: FlexTensor, dim: usize, indices: FlexTensor) -> FlexTensor {
    let dtype = tensor.dtype();
    let result = select::<u8>(tensor, dim, indices);
    // The shared kernel tags its output as u8; retain the input's bool dtype.
    FlexTensor::from_arc(result.data_arc(), result.layout().clone(), dtype)
}
```

With the patch the selected and gathered tensors are `Bool(Native)`, read back as `[true,
true, true, false]` and `[false, true, false, false, true, true]`, and concatenate with
other Boolean tensors.

### Environment

burn 0.22.0, Flex backend, Rust 1.99.1, Apple M3 Max, macOS 27.0.1.

## Notes for this crate

- On the crate's pin (0.21.0), `bool_select_or` is affected too; the v0.21.0 patch re-tags
  all three with a `retag_bool` helper in `ops/bool.rs`.
