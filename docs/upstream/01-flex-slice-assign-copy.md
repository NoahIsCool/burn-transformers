# 01. burn-flex: `slice_assign` and `scatter` copy the whole destination even when it is uniquely owned

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-flex` |
| Version reproduced | 0.21.0 (this crate's pin) |
| Kind | Performance (every in-place write is `O(destination)`) |
| Upstream status, checked 2026-10-16 | `slice_assign`: fixed in **v0.22.0** (`slice_write_impl` reuses a unique, contiguous destination). `scatter_add`: writes in place on `main`. `scatter_nd`: still copies on `main`. File only the `scatter_nd` part, or nothing if `scatter_nd` is fixed by the time this is reviewed. |
| Repro | [`repros/flex-slice-assign-copy`](repros/flex-slice-assign-copy) |
| Patch | [`patches/01-flex-slice-assign-copy.patch`](patches/01-flex-slice-assign-copy.patch) (against burn v0.21.0, `crates/burn-flex`) |
| Patch verified | Yes, Apple M3 Max: one-token write into a 64 MiB plane 1,124 µs → 0.5 µs; copy-on-write of shared destinations unchanged |
| Workarounds an upstream fix removes | `operations::assign_region`, `zero_region` and `write_region` (`src/operations.rs`, "In-place region writes"), and the reasoning in `src/cache/tensor.rs`, `src/cache/tensor/kv/batch.rs`, `src/cache/rows.rs`, `src/cache/tensor/recurrent.rs`; the per-step copy noted in `src/generation/history.rs` |

## Issue draft

**Title:** burn-flex: `slice_assign`, `scatter_add` and `scatter_nd` copy the whole destination even when it is uniquely owned

### Describe the bug

`slice_assign` on Flex copies its entire destination on every call, even when the caller
moved a uniquely owned tensor into it. A decode cache that writes one token into a
preallocated `[batch, heads, capacity, head_dim]` plane therefore pays
`O(batch · heads · capacity · head_dim)` per token instead of `O(region)`, and allocates a
fresh plane each time. `scatter_add` and `scatter_nd` behave the same way.

### To reproduce

Standalone project: `repros/flex-slice-assign-copy` (`burn = "=0.21.0"`, feature `flex`).

```rust
let mut plane = Some(Tensor::<Flex, 4>::zeros([1, 8, capacity, 128], &device));
// per decode step:
let destination = plane.take().unwrap();          // uniquely owned
plane = Some(destination.slice_assign([0..1, 0..8, slot..slot + 1, 0..128], token.clone()));
```

`cargo run --release` prints the median time per write.

### Expected behavior

The time per write does not depend on `capacity`; a shared destination is still copied
(copy-on-write).

### Actual behavior (Apple M3 Max, macOS 27, burn 0.21.0)

| capacity | plane | µs per write |
|---:|---:|---:|
| 1,024 | 4 MiB | 79.9 |
| 4,096 | 16 MiB | 275.5 |
| 16,384 | 64 MiB | 1,123.9 |

That is memcpy bandwidth (about 55-60 GB/s of plane per write). A one-element
`scatter(.., IndexingUpdateOp::Add)` into a `[1, vocabulary]` buffer costs 4.2 / 13.7 /
152.9 µs for 32 Ki / 256 Ki / 2 Mi entries.

### Root cause

`crates/burn-flex/src/ops/slice.rs`, `slice_write_impl`:

```rust
let mut tensor = tensor.to_contiguous();   // returns a clone sharing the Arc
...
let dst = tensor.storage_mut::<E>();       // Arc::make_mut: refcount is 2, so it copies
```

`to_contiguous(&self)` returns `self.clone()` for an already contiguous tensor while the
caller's owned binding `tensor` is still alive, so the `Arc` has two owners and
`Arc::make_mut` in `storage_mut` always copies. `scatter_add` and `scatter_nd`
(`crates/burn-flex/src/ops/gather_scatter.rs`) build the result with
`tensor_data.to_vec()` / `data_data.to_vec()` unconditionally.

### Proposed fix

Release the caller's handle before taking mutable storage, and let the scatter kernels
update `storage_mut()` (which still copies a shared buffer):

```rust
let mut tensor = {
    let contiguous = tensor.to_contiguous();
    drop(tensor);
    contiguous
};
```

The patch applies the same pattern to `scatter_add` and `scatter_nd`. Measured with the
patch: 0.7 / 0.5 / 0.5 µs per `slice_assign` write and 0.4 / 0.3 / 0.2 µs per scatter,
independent of the destination size. The repro's copy-on-write check (writes into a
shared destination leave the other handle unchanged) passes before and after.

(v0.22.0 already reuses a unique destination in `slice_write_impl`, and `main` updates
`scatter_add` in place; `scatter_nd` on `main` still starts with
`let mut result: Vec<E> = data_data.to_vec();`.)

### Environment

burn 0.21.0, burn-flex default features (`std`, `simd`, `rayon`), Rust 1.94 nightly,
Apple M3 Max, macOS 27.0.1.

## Notes for this crate

- Run the repro with `cargo run --release` from its directory. To build against the
  patched source, apply the patch to a copy of the crates.io sources and add
  `--config 'patch.crates-io.burn-flex.path="<copy>/crates/burn-flex"'`.
- The crate's `assign_region` stays correct after an upgrade; it becomes removable once
  the pinned Burn includes the `slice_write_impl` change (v0.22.0). `TokenHistory` would
  stop paying a `[batch, vocabulary]` copy per step once `scatter_add` writes in place.
