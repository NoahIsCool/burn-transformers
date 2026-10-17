# 07. burn-cubecl / cubek-matmul: packed sub-byte matmul fails or returns wrong values, and `dequantize` mishandles transposed packed views

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn) (`burn-cubecl`), kernels in [tracel-ai/cubek](https://github.com/tracel-ai/cubek) (`cubek-matmul`) |
| Version reproduced | burn 0.21.0 / cubek-matmul 0.2.0 (this crate's pin) and **burn 0.22.0 / cubek-matmul 0.3.0**, Metal |
| Kind | Correctness: 0.21.0 returns wrong products silently; 0.22.0 panics at launch for the same products, and autotuning occasionally selects a wrong candidate |
| Upstream status, checked 2026-10-16 | **Open.** `dequantize` of a transposed packed view is fixed in v0.22.0 ([burn#5278](https://github.com/tracel-ai/burn/issues/5278), fixed by [#5281](https://github.com/tracel-ai/burn/pull/5281)). The kernel failures reproduce on v0.22.0 (measured below). [burn#4492](https://github.com/tracel-ai/burn/issues/4492) reported the same `Q4S`, `m > 1` launch failure on WGSL; it was closed on 2026-10-06 without a retest ("the matmul selection was reworked", tracked in [cubecl#834](https://github.com/tracel-ai/cubecl/issues/834), which is still open), and a comment there says Q4S works on Metal, which these measurements contradict. File as a new issue that references #4492. |
| Related upstream issues | burn#4492, cubecl#834, burn#5278 / #5281 |
| Repro | [`repros/v0.22.0/cubecl-packed-quantized-matmul`](repros/v0.22.0/cubecl-packed-quantized-matmul) (to file; public `matmul` only); [`repros/cubecl-packed-quantized-matmul`](repros/cubecl-packed-quantized-matmul) (0.21.0; also calls each cubek strategy directly) |
| Patch | [`patches/07-cubecl-packed-quantized-matmul.patch`](patches/07-cubecl-packed-quantized-matmul.patch) (v0.21.0, `crates/burn-cubecl`: a `dequantize` fix and a mitigation in `launch_matmul`; the kernels themselves are not fixed). No v0.22.0 patch: the 0.22.0 failures need a fix in cubek's routine selection and kernels. |
| Patch verified | Yes, v0.21.0 on Apple M3 Max, with and without autotuning: public `matmul` 18 of 36 F32-activation and 12 of 36 F16-activation products wrong without autotuning (6 and 5 with it) → 0 of 36 and 0 of 36 in both modes; narrow-output products all correct; `dequantize()` of a transposed packed view relative error 1.0 (Q8) and 3.6e3 (Q4) → 0 |
| Workarounds an upstream fix removes | In `operations::linear` (`src/operations.rs`): the per-row evaluation of 4-bit packed products on CubeCL (`packed_rows_need_single_row_products`, commit `8a8f660`), the rejection of 2-bit packed weights (`packed_value_is_unreliable_on_cubecl`), and the packed-Metal autotune guard (`packed_metal_autotune_is_unavailable`); the transposed-`dequantize` note in `src/checkpoint/runtime/packed_install.rs` |

Errors below are the maximum error relative to the largest reference magnitude, from one
run each. Random weights and activations change between runs. Without autotuning the set of
failing products was the same in every run (three on 0.21.0, three on 0.22.0); with
autotuning the selected candidates vary between runs.

## Issue draft

**Title:** CubeCL packed block-quantized matmul (Q4S/Q2S right-hand side): launch fails for `m > 1` without autotuning, and autotuning can pick a wrong candidate

### Describe the bug

With a right-hand side stored as `QuantStore::PackedU32`, symmetric, `[1, g]` blocks,
`Tensor::matmul` on Metal in burn 0.22.0:

- without autotuning, every Q4S product with F32 activations and 2 to 32 left-hand rows
  panics with `Shared memory budget exceeded: kernel requests 73728 bytes, hardware allows
  32768 bytes per cube`; Q2S products with F32 activations and `m > 1` panic the same way
  (294,912 bytes); and every Q2S product with F16 activations, including `m = 1`, panics
  with `Invalid vector size. Got 16 which should not be >8`. For a narrow `[1024, 16]`
  weight, Q4S with F32 activations panics for every `m >= 32`, so no row threshold avoids
  it;
- with autotuning (Burn's default for the GPU backends) the results were correct in three
  of four runs; in one run the autotuner kept a candidate that returned a single-row Q4S ×
  F16 product with 14% relative error, while other candidates failed during the benchmark
  with `quantized view float vector size 4 must be a positive multiple of num_quants 8`.

In burn 0.21.0 the same products did not panic: they silently returned values unrelated to
the right answer (relative error about 1.0, up to 1e31 for Q2S). The 0.22.0 failure is the
symptom of #4492 (`Q4S`, `M > 1`, shared memory), here on Metal/MSL.

### To Reproduce

Standalone project `repros/v0.22.0/cubecl-packed-quantized-matmul` (`burn = "=0.22.0"`,
`default-features = false`, features `std`, `metal`, `flex`):

```text
cargo run --release                      # no autotuning
cargo run --release --features autotune  # Burn's default for GPU backends
```

It multiplies `x[m, 1024]` (F32, or F16 with the same values) by `W[1024, 1024]`, where `W`
is the transposed view of a packed `[n, k]` matrix with 64-element groups along `k`
(column-major, as a `LinearLayout::Col` checkpoint weight) or packed `[k, n]` with
32-element groups along `n` (row-major), for Q8S, Q4S and Q2S and `m` in {1, 2, 4, 8, 32,
64}, plus a narrow `W[1024, 16]`. The reference is the host F64 product with the same
packed bytes dequantized independently on Flex; a product that panics counts as wrong.

### Expected behavior

Every product runs and matches the reference to F32 (about 1e-6) or F16 (about 6e-4)
rounding, whatever the strategy or the autotuner's choice.

### Actual behavior (Metal, Apple M3 Max, burn 0.22.0, no autotuning)

| W | value × activation | m = 1 | 2 | 4 | 8 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| either layout | Q8S × F32, Q8S × F16 | ok | ok | ok | ok | ok | ok |
| either layout | Q4S × F32 | ok | **panic** | **panic** | **panic** | **panic** | ok |
| either layout | Q4S × F16 | ok | ok | ok | ok | ok | ok |
| either layout | Q2S × F32 | ok | **panic** | **panic** | **panic** | **panic** | **panic** |
| either layout | Q2S × F16 | **panic** | **panic** | **panic** | **panic** | **panic** | **panic** |

18 of 36 F32-activation and 12 of 36 F16-activation products fail. Narrow `W[1024, 16]`,
col-major, F32 activations: Q4S runs for `m` = 1, 2, 8 and panics for 32, 33, 64; Q2S
panics for every `m`. With autotuning: 0 of 72 products wrong in three runs, 1 of 72 (Q4S ×
F16, col-major, `m = 1`, error 1.4e-1) in one.

The same table on burn 0.21.0 / cubek-matmul 0.2.0 has wrong values (relative error 1.0,
or 1e14-1e31 for some Q2S products) in exactly the cells marked "panic". With autotuning,
0.21.0 got 6 F32 and 5 F16 products wrong instead: a single-row Q8S × F32 row-major product
off by 12-15%, and Q2S col-major products with `m > 1` (non-finite with F16 activations).

### Root cause

Identified on 0.21.0, where each strategy can be launched directly (Part 2 of
`repros/cubecl-packed-quantized-matmul`, F32 activations, 32-element groups, strategies
that decline a problem omitted):

| value | m | wrong strategies | correct strategies |
|---|---:|---|---|
| Q8S | 1 | `GemvUnitPerpendicular` (1e-1, both layouts) | `Auto`, `SimpleUnit`, `DoubleUnit`; col-major also `SimpleVecMat`, `DoubleVecMat`, `GemvPlaneParallel` |
| Q8S | 4, 64 | none | all that accept the problem |
| Q4S | 1 | `GemvUnitPerpendicular`; col-major `GemvPlaneParallel` (1e0) | `Auto`, `SimpleUnit`, `DoubleUnit` |
| Q4S | 4 | `Auto`, `SimpleUnit`, `DoubleUnit` (1e0) | none accepted |
| Q4S | 64 | `SimpleUnit`, `DoubleUnit` (1e0) | `Auto` (cooperative-matrix routine) |
| Q2S | 1 | `DoubleUnit`, `GemvUnitPerpendicular`; col-major `GemvPlaneParallel` (1e0) | `Auto`, `SimpleUnit` |
| Q2S | 4, 64 | `Auto` (including its cooperative-matrix routine at 64), `SimpleUnit`, `DoubleUnit` (1e0) | none accepted |

`Strategy::Auto` tries the cooperative-matrix routine and falls back to `SimpleUnit` when it
declines (here for `m <= 32`, and for every `m` when `n = 16`). The unit routines handle
sub-byte values (8 or 16 per `u32`) only for a single left-hand row. In cubek-matmul 0.3.0
the same configurations are rejected at launch instead (shared memory, vector size), and
`burn-cubecl`'s matmul unwraps the error. The autotuner ranks candidates by speed only, so
for quantized inputs it can keep a wrong one.

Separately (fixed in 0.22.0 by #5281): `burn-cubecl/src/kernel/quantization/dequantize.rs`
in 0.21.0 launched the kernel on the tensor as is, but a transposed packed view is packed
along the second-to-last axis (`QuantStore::PackedU32(1)`).

### Proposed fix

The routines need fixing in cubek (unit routines with several rows of sub-byte values;
`GemvUnitPerpendicular` and `GemvPlaneParallel` with packed block-scaled operands), and
`Strategy::Auto` should not fall back to a routine that cannot run the problem. As a
mitigation, the attached v0.21.0 `burn-cubecl` patch dequantizes the packed right-hand side
in `launch_matmul` for the combinations measured as unreliable, chosen by the value's bit
width (`QuantValue::size_bits()`): 2-bit values on every routine, every packed value on
`GemvUnitPerpendicular`, sub-byte values on `GemvPlaneParallel`, and sub-byte values with
more than one row on `Auto`/`SimpleUnit`/`DoubleUnit`. This trades the packed-memory
benefit for correctness on those launches only. It also fixes `dequantize` for transposed
views. On 0.21.0 it brings every product in the table to F32/F16 rounding, with and without
autotuning; Part 2 calls cubek directly and is unchanged, documenting the kernel defects. A
0.22.0 equivalent would catch the setup error (or the same combinations) in
`launch_matmul` and retry on the dequantized operand.

### Environment

burn 0.22.0 / burn-cubecl 0.22.0 / cubek-matmul 0.3.0 and burn 0.21.0 / cubek-matmul 0.2.0,
wgpu Metal (MSL), Apple M3 Max, macOS 27.0.1, Rust 1.99.1 (0.22.0) and 1.94 nightly
(0.21.0). Not run on CUDA, ROCm or Vulkan.

## Notes for this crate

- **Fixed in the crate (commit `8a8f660`).** Its packed path is the non-autotuned one, and
  its weights are the "col-major, 64-element groups" case above. Before the fix, Q4 products
  with F32 activations and 2 to 32 rows (short prefills, batched decode) were wrong on Metal;
  the reviewer confirmed it through `operations::linear` (error equal to the output magnitude
  at 2, 4, 8 and 32 rows). `operations::linear` now multiplies 4-bit packed weights on
  CubeCL one row at a time, unless the activations are F16 on Metal (measured correct for
  every shape tried), and rejects 2-bit packed weights there. On 0.22.0 the per-row path
  issues only single-row products, which run correctly there too, so the fix carries over.
- The per-row path costs one packed single-row product per row: an F32 prefill through Q4
  weights on Metal costs about as much as that many decode-step projections (about 0.2 ms
  per row for a `1024 × 1024` weight on the M3 Max). F16 activations are unaffected.
- The recorded Q4 Metal acceptance runs keep retained parameters in F16, so their packed
  projections most likely ran with F16 activations, which these measurements show correct.
  Any packed projection fed F32 activations with more than one row before `8a8f660` was
  wrong, and the cached-versus-stateless comparisons in those smokes could not have
  noticed; re-running them is the way to confirm.
- The autotune guard (`packed_metal_autotune_is_unavailable`) stays: 0.21.0's autotuner picks
  wrong candidates for packed Q8 and Q2, and 0.22.0's sometimes for Q4 × F16.
