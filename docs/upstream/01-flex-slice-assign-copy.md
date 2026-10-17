# 01. burn-flex: `scatter_nd` copies the whole destination even when it is uniquely owned

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-flex` |
| Version reproduced | 0.21.0 (this crate's pin) and **0.22.0** |
| Kind | Performance (every in-place write is `O(destination)`) |
| Upstream status, checked 2026-10-16 | `slice_assign` and `scatter`: **fixed in v0.22.0** (measured below). `scatter_nd`: **still copies** in v0.22.0 (measured) and on `main` at `df0812e` (same `data_data.to_vec()` code). **File the `scatter_nd` part only.** |
| Repro | [`repros/v0.22.0/flex-slice-assign-copy`](repros/v0.22.0/flex-slice-assign-copy) (to file); [`repros/flex-slice-assign-copy`](repros/flex-slice-assign-copy) (0.21.0) |
| Patch | To file: [`patches/v0.22.0/01-flex-scatter-nd-copy.patch`](patches/v0.22.0/01-flex-scatter-nd-copy.patch) (v0.22.0) and the same change rebased onto `main`, [`patches/v0.22.0/01-flex-scatter-nd-copy-main.patch`](patches/v0.22.0/01-flex-scatter-nd-copy-main.patch). For this crate's pin: [`patches/01-flex-slice-assign-copy.patch`](patches/01-flex-slice-assign-copy.patch) (v0.21.0: `slice_assign`, `scatter_add`, `scatter_nd`) |
| Patch verified | Yes, Apple M3 Max. v0.22.0: one-coordinate `scatter_nd` into a 2 Mi-entry buffer 163.5 µs (Int) / 156.3 µs (F32) → 0.5 / 0.6 µs. v0.21.0: 64 MiB `slice_assign` 1,360 µs → 0.5 µs, `scatter` 176 µs → 0.4 µs, `scatter_nd` 178 µs → 0.5 µs. Copy-on-write of shared destinations unchanged in every run. The `main` patch was checked with `git apply --check` only. |
| Workarounds an upstream fix removes | v0.22.0 already lets us drop `operations::{assign_region, zero_region, write_region}` (`src/operations.rs`, "In-place region writes") and the matching reasoning in `src/cache/tensor.rs`, `src/cache/tensor/kv/batch.rs`, `src/cache/rows.rs`, `src/cache/tensor/recurrent.rs`, plus the per-step `[batch, vocabulary]` copy noted in `src/generation/history.rs`. The `scatter_nd` fix would remove the full-vocabulary copy in `generation.rs::scatter_mask` and the "Flex still copies the whole destination" reason `cache/tensor/kv/batch.rs` gives for rejecting scatter writes. |

All figures are from single runs on a shared machine; repeated runs agree to within about
10-30% for the small timings, and the copy/no-copy contrast holds in every run.

## Issue draft

**Title:** burn-flex: `scatter_nd` copies the whole destination on every call, even when it is uniquely owned

### Describe the bug

`Tensor::scatter_nd` on Flex copies its entire destination buffer on every call, even
when the caller moved a uniquely owned tensor into it. Writing a handful of entries into
a `[batch, vocabulary]` buffer (a sampling mask, a token-count table) therefore costs
`O(batch · vocabulary)` per call instead of `O(updates)`, and allocates a fresh buffer each
time. In v0.22.0 `slice_assign` and `scatter` already update a unique destination in place;
`scatter_nd` was not converted.

### To Reproduce

Standalone project `repros/v0.22.0/flex-slice-assign-copy` (`burn = "=0.22.0"`, feature
`flex`), `cargo run --release`:

```rust
let device = Device::flex();
let coordinate = Tensor::<2, Int>::from_ints([[0, 3]], &device);
let mut mask = Some(Tensor::<2, Int>::zeros([1, vocabulary], &device));
// per step:
let destination = mask.take().unwrap(); // uniquely owned
mask = Some(destination.scatter_nd(
    coordinate.clone(),
    Tensor::<1, Int>::from_ints([7], &device),
    IndexingUpdateOp::Assign,
));
```

It prints the median time per write for `slice_assign` and `scatter` (controls) and for
`scatter_nd` into Int (`Assign`) and F32 (`Add`) buffers, then checks copy-on-write: a write
into a destination that is still shared must leave the other handle unchanged.

### Expected behavior

The time per `scatter_nd` does not depend on the destination size, as for `slice_assign`
and `scatter`; a shared destination is still copied.

### Actual behavior (Apple M3 Max, burn 0.22.0)

| vocabulary | buffer | `scatter_nd` Int, µs | `scatter_nd` F32, µs | `scatter` F32, µs |
|---:|---:|---:|---:|---:|
| 32,768 | 128 KiB | 4.2 | 3.8 | 0.4 |
| 262,144 | 1 MiB | 14.3 | 14.2 | 0.3 |
| 2,097,152 | 8 MiB | 163.5 | 156.3 | 0.3 |

`scatter_nd` grows linearly with the buffer (memcpy bandwidth); `scatter` and `slice_assign`
(0.4-0.7 µs for a 64 MiB plane) do not.

### Root cause

`crates/burn-flex/src/ops/gather_scatter.rs`, `scatter_nd`:

```rust
let data = data.to_contiguous();          // clones the Arc while `data` is alive
...
let data_data: &[E] = data.storage();
let mut result: Vec<E> = data_data.to_vec();   // always a full copy
...
FlexTensor::new(Bytes::from_elems(result), Layout::contiguous(shape), E::dtype())
```

### Proposed fix

Take the destination by value with `into_contiguous()` (which returns a uniquely owned,
whole-buffer contiguous tensor unchanged) and update `storage_mut()` in place, which still
copies a buffer another tensor shares:

```rust
let mut data = data.into_contiguous();
...
let result: &mut [E] = data.storage_mut();
// ... same update loop ...
data
```

This is the pattern `scatter_update` and `slice_write_impl` already use. Patch attached
(`01-flex-scatter-nd-copy.patch` for v0.22.0, `01-flex-scatter-nd-copy-main.patch` for
`main`, identical changed lines). Measured with the patch on v0.22.0: 0.5-0.6 µs per
`scatter_nd` at every size; the repro's copy-on-write check passes before and after, and
burn-flex's own unit tests (425) pass.

### Environment

burn 0.22.0, burn-flex default features (`std`, `simd`, `rayon`), Rust 1.99.1, Apple M3
Max, macOS 27.0.1.

## Notes for this crate

- Measured on the crate's pin (0.21.0, `repros/flex-slice-assign-copy`): `slice_assign` of
  one token into a 4 / 16 / 64 MiB plane took 83 / 355 / 1,360 µs, a one-element `scatter`
  into 32 Ki / 256 Ki / 2 Mi entries 4.8 / 19.1 / 176 µs, and `scatter_nd` 4.9 / 19.9 /
  178 µs. With `patches/01-flex-slice-assign-copy.patch` every write takes 0.3-0.8 µs.
  The 0.21.0 cause of the `slice_assign` copy was `slice_write_impl` calling
  `to_contiguous(&tensor)` while the owned binding was alive, so `Arc::make_mut` always saw
  two owners; v0.22.0 fixed it the same way the patch does.
- To build the 0.21.0 repro against the patched source, apply the patch to a copy of the
  crates.io sources and add
  `--config 'patch.crates-io.burn-flex.path="<copy>/crates/burn-flex"'`.
- `assign_region` stays correct after an upgrade; it becomes removable with v0.22.0.
