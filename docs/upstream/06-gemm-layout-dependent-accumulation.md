# 06. gemm: results depend on operand strides (horizontal microkernel and row-major gemv reassociate)

| | |
|---|---|
| Upstream | [sarah-quinones/gemm](https://github.com/sarah-quinones/gemm) (formerly `sarah-ek/gemm`), crate `gemm-common` (reached through `burn-flex`'s matmul) |
| Version reproduced | gemm / gemm-common 0.19.0, which both burn-flex 0.21.0 and 0.22.0 depend on (`gemm = "0.19"`) |
| Kind | Numerical reproducibility (bitwise results depend on memory layout) |
| Upstream status, checked 2026-10-16 | **Open; no knob.** `gemm-common/src/gemm.rs` on `main` is byte-identical to 0.19.0 (blob `07626e2`; last push 2025-11-14), so the horizontal path is still gated only by `lhs_cs == 1 && rhs_rs == 1 && m * n <= 256`, and the gemv shortcuts only by shape and strides. File as a feature request with the patch, which applies to `main` unchanged. |
| Repro | [`repros/gemm-layout-dependent-accumulation`](repros/gemm-layout-dependent-accumulation) |
| Patch | [`patches/06-gemm-layout-dependent-accumulation.patch`](patches/06-gemm-layout-dependent-accumulation.patch) (against the gemm repository root, `gemm-common/src/gemm.rs`) |
| Patch verified | Yes, on both architectures: x86-64 (BU SCC login node, Intel Xeon Gold 6242, AVX-512) 5 of 7 products layout-dependent unpatched and with the patch's default, 0 of 7 with the new setting (earlier round; not re-run since, as the patch and repro are unchanged); aarch64 (Apple M3 Max) 3 of 7 unpatched, 0 of 7 with the setting (re-run in this round) |
| Workarounds an upstream fix removes | None needed for correctness; the crate's tests compare results across weight layouts to F32 rounding instead of bitwise (commit `fe9e795`, e.g. `src/models/gpt_oss/model_tests.rs`, `src/training_tests.rs`, `examples/burn_training.rs`), and `src/checkpoint/runtime/storage_order.rs` documents the reassociation |

## Issue draft

**Title:** Opt-in layout-independent accumulation: `A · B` differs bitwise with the strides of `B` (horizontal microkernel, row-major gemv)

### Describe the problem

The kernel that `gemm_basic_generic` picks, and with it the order in which each output
accumulates its `k` products, depends on the operands' strides:

- x86 with FMA: a product with `lhs_cs == 1 && rhs_rs == 1` and `m * n <= 256` (after the
  transposition for a row-major destination: row-major `A` times a column-major `B`, i.e.
  the transpose of a row-major matrix) runs the horizontal microkernel, which sums partial
  dot products in SIMD lanes and reduces them at the end;
- all architectures: `n == 1` / `m == 1` products take `mixed_gemv_rowmajor` (lane-wise
  partial sums) or `mixed_gemv_colmajor` (one sequential pass over `k`) depending on the
  strides;
- everything else runs the packed microkernel, which accumulates `k` in cache blocks.

So `A · B` and `A · B'`, where `B'` holds the same values as `B` in another layout, differ
in the last bits. Frameworks that store weights either way (Burn keeps a checkpoint's
`[out, in]` matrix as a transposed view, or copies it contiguous) see their tests and
checkpoint round trips change bitwise with the storage order, and only on some CPUs.

### To reproduce

Standalone project: `repros/gemm-layout-dependent-accumulation` (`gemm = "=0.19.0"`,
`gemm-common = "=0.19.0"`). It computes `A[m, k] · B[k, n]` into a row-major `C` once
with `B` row-major and once with the same logical `B` stored column-major, and counts the
elements whose bits differ.

### Expected behavior

Either documented layout dependence, or a way to request results that do not depend on
strides.

### Actual behavior

x86-64, Intel Xeon Gold 6242 (FMA, AVX2, AVX-512F), Rust 1.95, `Parallelism::None`:

| m | n | k | elements that differ | max abs diff |
|---:|---:|---:|---:|---:|
| 1 | 64 | 1024 | 61 of 64 | 2.7e-5 |
| 4 | 64 | 1024 | 238 of 256 | 1.7e-5 |
| 16 | 16 | 1024 | 233 of 256 | 1.7e-5 |
| 1 | 256 | 4096 | 253 of 256 | 8.0e-5 |
| 16 | 17 | 1024 | 0 of 272 | 0 |
| 1 | 257 | 1024 | 242 of 257 | 2.9e-5 |
| 64 | 64 | 1024 | 0 of 4096 | 0 |

aarch64 (Apple M3 Max) has no horizontal kernel: the 4x64 and 16x16 rows are identical
there, but the three `m = 1` rows (gemv) still differ.

### Root cause

`gemm-common/src/gemm.rs`, `gemm_basic_generic`: the horizontal branch
(`H_M > 0 && H_N > 0 && !conj_dst && lhs_cs == 1 && rhs_rs == 1 && m * n <= 16 * 16`) and
the four `gemv::mixed_gemv_{rowmajor,colmajor}` shortcuts are chosen from strides alone.
Each accumulates in a different order; there is no setting to bypass them.

### Proposed fix

An opt-in global, next to the existing threshold knobs
(`set_rhs_packing_threshold`, ...):

```rust
pub fn set_layout_independent_accumulation(value: bool);
pub fn get_layout_independent_accumulation() -> bool;
```

When enabled, `gemm_basic_generic` skips the horizontal microkernel and the gemv
shortcuts, so every product takes the packed microkernel, whose order depends only on the
shape (and the cache parameters), not on the strides. The default stays `false`, so
nothing changes unless a caller opts in; the cost of opting in is packing the operands of
matrix-vector products.

Measured with the patch: 0 of 7 products differ on both the Xeon and the M3 Max with the
setting enabled; with the default, the Xeon still shows 5 of 7 (unchanged behavior).

### Environment

gemm 0.19.0; x86-64: Intel Xeon Gold 6242, Linux, Rust 1.95.0 (BU SCC login node
`scc1`, built offline from vendored sources); aarch64: Apple M3 Max, macOS 27.0.1, Rust
1.94 nightly.

## Notes for this crate

- With the setting exposed, Burn Flex could offer it (or enable it in tests) and the
  crate's cross-layout tests could compare bitwise again. Until then, the F32-rounding
  bounds introduced in `fe9e795` are the right policy.
- The SCC run used `/scratch/nover/tfm-upstream-gemm` on `scc1` (node-local scratch, not
  the home quota) with `CARGO_HOME` there as well; it was a single-threaded build and a
  sub-second run, so no batch job was needed.
