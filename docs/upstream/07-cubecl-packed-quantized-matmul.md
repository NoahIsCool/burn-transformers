# 07. burn-cubecl / cubek-matmul: packed block-quantized matmul and `dequantize` return wrong values

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn) (`burn-cubecl`), kernels in [tracel-ai/cubek](https://github.com/tracel-ai/cubek) (`cubek-matmul`) |
| Version reproduced | burn 0.21.0, cubek-matmul 0.2.0, Metal |
| Kind | Correctness (silently wrong products; up to 100% relative error) |
| Upstream status, checked 2026-10-16 | `dequantize` of a transposed packed view: **fixed in v0.22.0** (it now moves the packed axis last and permutes back). The wrong kernels: not checked against cubek `main`; re-run Part 2 of the repro on v0.22.0 before filing. |
| Repro | [`repros/cubecl-packed-quantized-matmul`](repros/cubecl-packed-quantized-matmul) |
| Patch | [`patches/07-cubecl-packed-quantized-matmul.patch`](patches/07-cubecl-packed-quantized-matmul.patch) (`crates/burn-cubecl`: a `dequantize` fix and a mitigation in `launch_matmul`; the kernels themselves are not fixed) |
| Patch verified | Yes, Metal on Apple M3 Max: public `matmul` 8 of 24 products wrong without autotuning and 1 of 24 with it → 0 of 24 in both; `dequantize()` of a transposed packed view relative error 1.0 (Q8) / 1.9e3 (Q4) → 0 |
| Workarounds an upstream fix removes | `packed_metal_autotune_is_unavailable` and the guard in `operations::linear` (`src/operations.rs`), which rejects packed Metal matmul when autotuning is compiled in; the note in `src/checkpoint/runtime/packed_install.rs` that CubeCL `dequantize` ignores the strides of a transposed packed view |

## Issue draft

**Title:** Packed block-quantized matmul on CubeCL returns wrong values (`GemvUnitPerpendicular`, `GemvPlaneParallel` with Q4, unit routines with Q4 and `m > 1`), and `dequantize` mishandles transposed packed views

### Describe the bug

With a right-hand side stored as `QuantStore::PackedU32`, symmetric `Q8S`/`Q4S`,
`QuantLevel::block([1, g])`, several cubek-matmul routines compute wrong products, and
which one runs depends on the shape and on whether autotuning is enabled:

- without autotuning (`MatmulStrategy::Cube` → `Strategy::Auto`, which falls back to
  `SimpleUnit` when the cooperative-matrix routine is unavailable), every packed **Q4**
  product with 2 to 32 left-hand rows is wrong (relative error 1.0, i.e. the output is
  unrelated to the reference);
- with autotuning (Burn's default for the GPU backends), a single-row packed **Q8**
  product whose weight was quantized in place as `[k, n]` is off by about 15%, because the
  autotuner keeps `GemvUnitPerpendicular`, the fastest candidate, which is wrong;
- `dequantize()` of a transposed packed view (`swap_dims` of a packed `[n, k]` matrix, as
  for a column-layout checkpoint weight) returns wrong values (relative error 1.0 for Q8,
  1.9e3 for Q4).

### To reproduce

Standalone project: `repros/cubecl-packed-quantized-matmul` (`burn = "=0.21.0"` with
`std`, `metal`, `flex`; `burn-cubecl = "=0.21.0"` and `cubek = "=0.2.0"` to call each
strategy). The reference for every product is the host F64 product with the same packed
bytes dequantized independently on Flex.

```text
cargo run --release                      # no autotuning
cargo run --release --features autotune  # Burn's default for GPU backends
```

Part 1 uses the public `Tensor::matmul`; Part 2 calls each cubek strategy explicitly with
the operands bound exactly as `burn_cubecl::kernel::matmul::launch_matmul` binds them.

### Expected behavior

Every product matches the reference to F32 rounding (about 1e-6 relative).

### Actual behavior (Metal, Apple M3 Max)

Part 1, `x[m, 1024] · W[1024, 1024]`, max relative error; `W` either the transposed view of
a packed `[n, k]` matrix with 64-element groups along `k` (column-major), or packed
`[k, n]` with 32-element groups along `n` (row-major):

| autotuning | W | value | m = 1 | 2 | 4 | 8 | 32 | 64 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| off | col-major, g64 | Q8S | 1.4e-6 | 1.3e-6 | 1.5e-6 | 1.5e-6 | 1.8e-6 | 1.7e-6 |
| off | col-major, g64 | Q4S | 1.3e-6 | **1.0** | **1.0** | **1.0** | **1.0** | 1.6e-6 |
| off | row-major, g32 | Q8S | 1.4e-6 | 1.4e-6 | 1.4e-6 | 1.6e-6 | 1.5e-6 | 1.8e-6 |
| off | row-major, g32 | Q4S | 1.4e-6 | **1.0** | **1.0** | **1.0** | **1.0** | 1.6e-6 |
| on | col-major, g64 | Q8S, Q4S | all ≤ 2e-6 | | | | | |
| on | row-major, g32 | Q8S | **1.5e-1** | 1.5e-6 | 1.6e-6 | 1.5e-6 | 1.7e-6 | 1.7e-6 |
| on | row-major, g32 | Q4S | all ≤ 2e-6 | | | | | |

Part 2, explicit strategies (32-element groups; strategies that decline a problem are
omitted), max relative error:

| W | value | m | wrong strategies | correct strategies |
|---|---|---:|---|---|
| row-major | Q8S | 1 | `GemvUnitPerpendicular` 2e-1 | `Auto`, `SimpleUnit`, `DoubleUnit` |
| col-major | Q8S | 1 | `GemvUnitPerpendicular` 1e-1 | `Auto`, `SimpleUnit`, `DoubleUnit`, `SimpleVecMat`, `DoubleVecMat`, `GemvPlaneParallel` |
| either | Q8S | 4, 64 | none | all that accept the problem |
| either | Q4S | 1 | `GemvUnitPerpendicular` 1e0, `GemvPlaneParallel` (col-major) 1e0 | `Auto`, `SimpleUnit`, `DoubleUnit` |
| either | Q4S | 4 | `Auto`, `SimpleUnit`, `DoubleUnit` 1e0 | none accepted |
| either | Q4S | 64 | `SimpleUnit`, `DoubleUnit` 1e0 | `Auto` (cooperative-matrix routine) |

### Root cause

- `cubek-matmul` 0.2.0: `GemvUnitPerpendicular` mishandles every packed block-scaled
  right-hand side; `GemvPlaneParallel`, `SimpleUnit` and `DoubleUnit` mishandle packed Q4
  (8 values per `u32`) — the unit routines only when more than one left-hand row is
  processed. Not isolated further than the routine.
- Autotuning (`burn-cubecl/src/kernel/matmul/tune/base.rs`) selects candidates by speed
  only; for quantized inputs it can keep a wrong one.
- `burn-cubecl/src/kernel/quantization/dequantize.rs` (0.21.0) launches the cubek kernel on
  the tensor as is; the kernel unpacks along the innermost axis in storage order, but a
  transposed packed view is packed along the second-to-last axis (`QuantStore::PackedU32(1)`).

### Proposed fix

The kernels need fixing in cubek. The attached `burn-cubecl` patch is a mitigation that
keeps results correct with the current kernels:

1. `dequantize`: a tensor packed along the second-to-last axis is swapped back to its
   storage layout, dequantized, and given the same transposed view (v0.22.0 has an
   equivalent, more general fix).
2. `launch_matmul`: before launching, dequantize a packed block-quantized right-hand side
   for the combinations measured as wrong (every packed operand on
   `GemvUnitPerpendicular`, Q4 on `GemvPlaneParallel`, Q4 with more than one row on
   `Auto`/`SimpleUnit`/`DoubleUnit`). This trades the packed-memory benefit for
   correctness on those launches only.

Measured with the patch: 0 of 24 Part 1 products wrong with and without autotuning, and
`dequantize()` of both transposed views exact. Part 2 calls cubek directly, so it is
unchanged by the patch and keeps documenting the kernel defects.

### Environment

burn 0.21.0, burn-cubecl 0.21.0, cubek-matmul 0.2.0, wgpu Metal (MSL), Apple M3 Max,
macOS 27.0.1, Rust 1.94 nightly. Not run on CUDA/ROCm.

## Notes for this crate

- **Risk for this crate:** its packed Metal path is the non-autotuned one (the crate
  refuses packed Metal matmul when autotuning is compiled in), and its weights are exactly
  the "col-major, 64-element groups" case above. On Burn 0.21, Q4 products with 2 to 32
  activation rows (short prefills, batched decode) are therefore likely wrong on Metal.
  The crate's opt-in Metal packed tests (`src/operations_tests.rs`) feed single-row inputs
  only, and the Q4 acceptance smokes compare Burn paths with each other (cached versus
  stateless), which a deterministic wrong kernel passes. A multi-row Q4 test against a
  dequantized reference is the next step; it was not run as part of this report.
- The autotune guard in the crate protects against the Q8 single-row case but not the
  non-autotuned Q4 case.
