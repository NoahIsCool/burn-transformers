# 15. cubecl-std: the CubeCL CPU backend copies a reshaped strided view at the wrong coordinates

| | |
|---|---|
| Upstream | [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl), crate `cubecl-std` (`tensor::contiguous`), reached through `burn-cubecl`'s `reshape` on `burn-cpu` |
| Version reproduced | burn 0.21.0, burn-cpu 0.21.0, cubecl-std 0.10.0 (this crate's pin) |
| Kind | **Correctness** (wrong values, and reads past the input buffer, for an ordinary `reshape`) |
| Upstream status, checked 2026-10-17 | **Fixed in v0.22.0; do not file.** Reported as [cubecl#1670](https://github.com/tracel-ai/cubecl/issues/1670) ("cpu: copy_into sends a reshape to a kernel that assumes the shapes match", closed) and fixed by [cubecl#1671](https://github.com/tracel-ai/cubecl/pull/1671) ("fix(std): keep a reshape out of the perpendicular copy", merged 2026-10-02, `1bb393f`): `copy_into` takes the perpendicular kernel only when `input.shape == output.shape`. The guard ships in cubecl-std 0.11.0, which burn-cubecl 0.22.0 uses. Checked by reading the 0.11.0 sources; the repro was not run on 0.22.0, whose `cpu` feature builds against another prebuilt LLVM (tracel-llvm 23) that this host has not downloaded. |
| Related upstream issues | cubecl#1670 / #1671 |
| Repro | [`repros/cubecl-cpu-reshape-copy`](repros/cubecl-cpu-reshape-copy) |
| Patch | [`patches/15-cubecl-cpu-reshape-copy.patch`](patches/15-cubecl-cpu-reshape-copy.patch) (`crates/cubecl-std/src/tensor/contiguous/launch.rs`, relative to the cubecl repository root; the same guard as cubecl#1671) |
| Patch verified | Yes, on an Apple M3 Max: the repro's 5 of 7 wrong cases → 0 of 7; through this crate, the CubeCL CPU parity cases for Nemotron-H and both modular causal decoders pass `F32_BOUND` (largest deviations `1.8e-7` and `2.8e-7`) instead of deviating by up to `5.8e-1` |
| Workarounds an upstream fix removes | None in the crate (deliberately; see "Notes for this crate"). A Burn upgrade re-enables the ignored tests `cpu::cpu_nemotron_h_matches_flex`, `cpu::cpu_modular_causal_matches_flex`, and the reproducer `cpu::cpu_reshaped_strided_views_match_flex` in `tests/backend_parity.rs` |

## Issue draft (not to be filed: fixed in v0.22.0)

Kept as the record of the 0.21.0 behaviour this crate's CubeCL CPU validation meets.

**Title:** burn-cpu: `reshape` of a transposed or expanded tensor returns wrong values (`copy_into` runs the perpendicular kernel for a shape change)

### Describe the bug

When a reshape cannot be expressed by new strides, `burn-cubecl`'s `reshape` allocates the
new shape and calls `cubecl_std::tensor::copy_into(input, output)`. On a CPU client whose
input's innermost stride is not 1, `copy_into` launches `copy_perpendicular`, which
computes each input offset from the **output's** coordinates and the input's strides. That
is a pure change of layout and is right only when both tensors have the same shape. A
reshape gives them different shapes, so the copy reads the wrong elements, and beyond the
input buffer when the output has more elements than the input's storage (an expanded
view). GPU clients take `copy_gpu_ref`, which indexes the input by its own shape, and are
correct.

Every common way to materialize a permuted or broadcast view by reshaping it is affected
on `burn-cpu`: flattening a transposed matrix, converting interleaved rotary pairs to
split halves (`[.., d] -> [.., d/2, 2] -> swap -> [.., d]`), or flattening an expanded mask.

### To reproduce

Standalone project: `repros/cubecl-cpu-reshape-copy` (`burn = "=0.21.0"`, feature `cpu`).
Each case compares the CubeCL CPU result with the same operation on the host:

```text
cargo run --release
```

### Expected behavior

Every reshape returns the row-major values of the view, as Flex, NdArray, and the GPU
backends do.

### Actual behavior (Apple M3 Max)

```text
transpose [2, 3] then reshape to [6]:
  cubecl-cpu [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  expected   [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]   <-- wrong
transpose [2, 3], read without a reshape (control):
  cubecl-cpu [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
de-interleave [8] via [4, 2] -> swap -> reshape [8]:
  cubecl-cpu [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  expected   [0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]   <-- wrong
float [2, 2, 1] expand [2, 2, 3] then reshape [2, 6]:
  cubecl-cpu [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  expected   [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]   <-- wrong
(int and bool expansions: the same wrong values)
float [2, 2, 1] expand [2, 2, 3], read without a reshape (control): correct
5 of 7 cases are wrong
```

The expanded cases read the four stored values as if they were contiguous and then past
them; in an earlier run the integer copy returned `1065353216` (the bits of `1.0f32`) from
a neighbouring allocation. The controls, which copy the same views into their own shape
(`into_data`), are correct.

### Root cause

cubecl-std 0.10.0, `src/tensor/contiguous/launch.rs`:

```rust
let is_cpu = client.properties().hardware.num_cpu_cores.is_some();
if input.strides[rank - 1] != 1 && is_cpu {
    launch_copy_perpendicular_ref(client, input, output, dtype);
} else {
    copy_gpu_ref(client, input, output, dtype);
};
```

and `src/tensor/contiguous/perpendicular.rs`, `copy_perpendicular`:

```rust
for axis in 0..input.rank() {
    let coordinate = output.coordinate(offset_output * vector_size, axis);
    batch_offset += coordinate * input.stride(axis);
}
```

`burn-cubecl` 0.21.0's `ops::base::reshape` (`ReshapeAction::Recompute`) passes the input
with its old shape and an output of the new shape.

### Proposed fix

Take the perpendicular kernel only for a change of layout (the attached patch; cubecl#1671
merged the same guard):

```rust
let same_shape = input.shape == output.shape;
if input.strides[rank - 1] != 1 && is_cpu && same_shape {
```

With the patch the repro reports 0 of 7 wrong cases.

### Environment

burn 0.21.0 (feature `cpu`; tracel-llvm 20.1.4-7), cubecl-std 0.10.0, Rust 1.94 nightly
(2025-12-22), Apple M3 Max, macOS 27. The GPU path was not changed and not re-run.

## Notes for this crate

- Found while extending `tests/backend_parity.rs` to the CubeCL CPU backend. Three
  executable paths meet the defect: Nemotron-H's group-limited router flattens an expanded
  `[tokens, groups, 1]` group mask (`prune_groups` in `src/moe/router.rs`), so the wrong
  experts become eligible; multi-head latent attention de-interleaves rotary pairs with
  `position::interleaved_to_split_half` (rotary widths of 4 or more; DeepSeek-V2's tiny
  case uses 2, a unit axis that needs no copy); and the modular gated DeltaNet with grouped
  key heads and a padding mask copies a strided view. With the patched cubecl-std every
  CubeCL CPU parity case passes.
- Not worked around in the crate. The defect is a generic `reshape` on one backend that
  the crate does not support (SUPPORT.md), fixed upstream; reshaping a permuted or
  broadcast view is the ordinary way to materialize it, and avoiding it at the three sites
  the tests happen to reach would leave every other such reshape exposed. The affected
  tests are ignored with this report as their reason, CI skips them, and
  `cpu::cpu_reshaped_strided_views_match_flex` reproduces the defect with the crate's own
  `interleaved_to_split_half`.
- The Nemotron-H failure also exposed a separate, all-CubeCL defect, report
  [16](16-cubecl-max-reduction-negative-infinity.md): the wrong mask left rows with every
  expert at `-inf`, and the router's top-k returned an out-of-range expert ID for them.
  That one is fixed in the crate.
- After moving the pin to Burn 0.22, run the three ignored tests
  (`cargo test --features cpu --test backend_parity cpu:: -- --ignored`) and make them
  ordinary.
