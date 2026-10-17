# Upstream reports for the Burn 0.21 limitations this crate works around

These are **drafts for review**; nothing here has been filed or posted. Each report has a
standalone reproduction that depends only on the pinned upstream crates, the exact source
location and root cause, a proposed patch against the published sources, and an issue
title and body in the upstream project's style.

This crate pins `burn = "=0.21.0"`. Burn **0.22.0** was released on 2026-10-06. Every
report was therefore checked against 0.22.0 as well, by running its reproduction ported to
0.22.0 where that was possible, and against Burn `main` at `df0812e` (2026-10-16) by
comparing the relevant source files. Reports that still reproduce on 0.22.0 have a 0.22.0
reproduction and a patch verified on 0.22.0 sources; their issue drafts are written against
0.22.0, which is what a maintainer will run. Reports fixed in 0.22.0 keep their 0.21.0
material as the record of what this crate works around, marked "not to be filed".

## Status and filing order

File, in this order (all still reproduce on 0.22.0):

1. [02](02-autodiff-graph-merge-lost-gradient.md) autodiff graph merge: a valid `backward()`
   panics on 0.22.0 (silently loses gradients on 0.21.0).
2. [07](07-cubecl-packed-quantized-matmul.md) packed sub-byte matmul on CubeCL: launch
   panics on 0.22.0 (wrong values on 0.21.0); new issue referencing the presumptively
   closed burn#4492. File the kernel part with cubek if the maintainers prefer.
3. [12](12-flex-bool-select-dtype.md) Boolean `select`/`gather` tagged `U8`.
4. [13](13-flex-flex32-dtype.md) `Flex32` tensors from `zeros`/`empty` on Flex.
5. [03](03-flex-depthwise-conv1d-fanout.md) depthwise `conv1d` rayon fan-out.
6. [10](10-linear-weight-reshape-copy.md) unit-axis reshape of a strided tensor copies it
   (`module::linear` with a transposed weight).
7. [01](01-flex-slice-assign-copy.md) `scatter_nd` copies its destination (the
   `slice_assign` and `scatter` parts are fixed in 0.22.0).
8. [06](06-gemm-layout-dependent-accumulation.md) gemm layout-independent accumulation, a
   feature request to the separate `gemm` repository.

Do not file (fixed in 0.22.0, measured unless noted): [04](04-flex-broadcast-binary-fast-paths.md),
[05](05-cubecl-flash-attention-causal-alignment.md) (source reading: cubek#406),
[08](08-linear-col-layout-gradient.md), [09](09-param-mapper-valid.md),
[11](11-softplus-overflow.md), [14](14-flex-quantized-layout-requantize.md).

## Index

| # | Report | Upstream | Kind | Verified effect (before → after patch) | Status (0.22.0 / `main`) | In-crate workarounds a fix would let us remove |
|---|---|---|---|---|---|---|
| 01 | [`scatter_nd` copies the whole destination](01-flex-slice-assign-copy.md) | burn-flex | Performance | 0.22.0: 2 Mi-entry `scatter_nd` 163.5 µs → 0.5 µs. 0.21.0: 64 MiB `slice_assign` 1,360 µs → 0.5 µs, `scatter`/`scatter_nd` 176-178 µs → 0.4-0.5 µs | `scatter_nd` **open**; `slice_assign`/`scatter` fixed in 0.22.0 | 0.22.0 already: `operations::{assign_region, zero_region, write_region}`; cache write reasoning in `cache/tensor.rs`, `cache/tensor/kv/batch.rs`, `cache/rows.rs`, `cache/tensor/recurrent.rs`; per-step copy in `generation/history.rs`. `scatter_nd` fix: the copy in `generation.rs::scatter_mask` |
| 02 | [Graph merge leaves an origin in an emptied graph](02-autodiff-graph-merge-lost-gradient.md) | burn-autodiff | **Correctness** | 0.22.0: 1,000 of 1,000 backward passes panic → 0. 0.21.0: 255 / 218 of 1,000 lose the gradient → 0; crate's old router test fails → passes | **Open** | `module::linear` avoidance in `moe/router.rs::router_logits` (keep the regression tests) |
| 03 | [Depthwise `conv1d` rayon fan-out](03-flex-depthwise-conv1d-fanout.md) | burn-flex | Performance | Busy-core p99: 0.22.0 25.8 ms → 0.43 ms; 0.21.0 45.3 ms → 0.35 ms | **Open** | `operations::depthwise_conv1d` (callers in `delta_net.rs`, `mamba2.rs`) |
| 04 | [Broadcast binary ops take a scalar strided walk](04-flex-broadcast-binary-fast-paths.md) | burn-flex | Performance | 0.21.0: RoPE 65x → 1.1x of dense; outer product 32x → 0.7x | **Fixed in 0.22.0** (RoPE 1.0x, outer 0.5x measured) | Fused RoPE kernel `position/tensor/flex.rs`; outer-product `input_update` and flattened `SegmentMasks` in `mamba2.rs` |
| 05 | [Flash attention aligns `is_causal` top-left](05-cubecl-flash-attention-causal-alignment.md) | burn-cubecl / cubek-attention | **Correctness** | 0.21.0: `FlashUnit` error vs Burn's reference 0.50 → declines; square control still flash | **Fixed in 0.22.0** (cubek#406, cubek-attention 0.3.0; source reading) | Square-only `is_causal` and explicit bottom-right mask in `attention/dispatch.rs` |
| 06 | [gemm results depend on operand strides](06-gemm-layout-dependent-accumulation.md) | gemm-common | Reproducibility | x86 (SCC Xeon): 5 of 7 layout-dependent → 0; arm64: 3 of 7 → 0 (opt-in setting) | **Open**; Burn 0.22.0 still uses gemm 0.19 | Cross-layout tests bounded by F32 rounding (`fe9e795`); `checkpoint/runtime/storage_order.rs` notes |
| 07 | [Packed sub-byte matmul fails or is wrong](07-cubecl-packed-quantized-matmul.md) | burn-cubecl / cubek-matmul | **Correctness** | 0.21.0, Metal: 18 + 12 of 72 wrong without autotuning, 6 + 5 with it → 0; transposed `dequantize` error 1.0 / 3.6e3 → 0. 0.22.0: the same 30 of 72 panic at launch; no 0.22.0 patch | **Open** (kernels); `dequantize` fixed in 0.22.0 | Per-row Q4 products and 2-bit rejection in `operations::linear` (`8a8f660`); packed-Metal autotune guard; transposed-`dequantize` note in `checkpoint/runtime/packed_install.rs` |
| 08 | [Mapped (`LinearLayout::Col`) params are not leaves](08-linear-col-layout-gradient.md) | burn-core | **Correctness** | 0.21.0: first SGD step skips the weight → updates it; `load_record` panic → loads | **Fixed in 0.22.0** (both halves measured) | `operations::init_linear` autodiff re-wrap; Row layout in `models/t5/model.rs::t5_linear`; install ordering in `checkpoint/runtime.rs` |
| 09 | [`valid()`/`from_inner()` drop param mappers](09-param-mapper-valid.md) | burn-core | **Correctness** | 0.21.0: reloaded square projection error 0.66 → 0; saved shape `[in,out]` → `[out,in]` | **Fixed in 0.22.0** (measured) | "Save from the training owner" rule in `training.rs` |
| 10 | [Unit-axis reshape of a strided weight copies it](10-linear-weight-reshape-copy.md) | burn-flex (and burn-backend's `linear`) | Performance | 594 MiB transposed readout: 0.22.0 31.5 ms → 11.7 ms (Flex patch); 0.21.0 102.2 ms → 26.5 ms (burn-backend patch) | **Open** | `operations::matmul_transposed` and tied readouts in `models/muse_glimmer/model.rs`, `models/qwen3_5/model.rs` |
| 11 | [`softplus` overflows](11-softplus-overflow.md) | burn-tensor | **Correctness** | 0.21.0: 4 of 10 wrong values/gradients → 0 | **Fixed in 0.22.0** (burn#5373; measured) | `delta_net::softplus` (also used by `mamba2.rs`) |
| 12 | [Boolean `select`/`gather` tagged `U8`](12-flex-bool-select-dtype.md) | burn-flex | **Correctness** | 4 of 4 Boolean reads fail → 0, on 0.22.0 and 0.21.0 | **Open** | `architecture/types.rs::select_bool_tensor` integer round trip |
| 13 | [`Flex32` tensors from `zeros`/`empty` on Flex](13-flex-flex32-dtype.md) | burn-flex | **Correctness** | 0.22.0: 9 of 18 operations panic → 0; 0.21.0: 9 of 9 → 0 | **Open** (narrower than 0.21.0) | Flex32 sort guard in `generation.rs` (and its test) |
| 14 | [Block-quantized layout ops requantize on Flex](14-flex-quantized-layout-requantize.md) | burn-flex | **Correctness** | 0.21.0: transpose changes a small row by 39% (Q8) / 100% (Q4); no patch (design change) | **Fixed in 0.22.0** (exact, measured) | `packed_layout_is_exact` probe and `PackedLayoutPolicy` fallback in `checkpoint/runtime/packed_install.rs` |

Figures are single runs on a shared Apple M3 Max unless a range is given; small timings
vary by about 20% between runs, and every before/after contrast held in every run.

## Crate-side fix made while preparing these reports

Report 07 showed that this crate's packed Metal path returned wrong values: with F32
activations, Burn 0.21's matmul produced results unrelated to the dequantized reference for
packed Q4 weights with 2 to 32 activation rows (and more for narrow outputs), in exactly
this crate's weight layout. The reviewer confirmed it through `operations::linear`. Commit
`8a8f660` fixes it in `src/operations.rs`: on CubeCL backends, 4-bit packed weights are
multiplied one activation row at a time (each row is the measured-correct single-row
kernel on the same packed storage; no dequantization), except for F16 activations on
Metal, which were measured correct; 2-bit packed weights are rejected there with
`Error::BackendCapability`. The Metal tests now compare 1 to 64 rows and every Qwen3.5
projection shape against the host product, and the kernel-trace gate counts the per-row
launches. See report 07's "Notes for this crate" for the cost and what earlier acceptance
runs did and did not exercise.

## Upstream tracker search

On 2026-10-16 the `tracel-ai/burn` tracker was searched (`gh search issues`, open and
closed) for `scatter_nd`, `depthwise conv1d`, `Flex32`, `bool select dtype`,
`select U8 bool`, `graph tape has already been consumed`, `merge_two`, `GraphLocator`,
`reshape non-contiguous copy`, `linear weight reshape`, `quantized matmul Q4`, `Q2S`,
`shared memory budget exceeded` and `rayon conv1d`. No open duplicate was found; the
closed burn#4492 reported report 07's 0.22.0 symptom on WGSL. Related issues, cited in the
reports:

- burn#5573 / #5647 (concurrent `backward()`; different cause from 02), burn#5879 (another
  "tape consumed" panic, from `to_device`; different trigger).
- burn#4492 (Q4S `m > 1` launch failure on WGSL, closed presumptively) and cubecl#834 (open)
  for 07; burn#5278 / #5281 (transposed `dequantize`, fixed in 0.22.0).
- burn#5373 / #5374 (softplus, 11).
- burn#6055 / #6056 and burn#6068 / #6069 (`Flex32` data on `main`, 13).
- cubek#406 (bottom-right causal mask, 05).
- burn#4725 and burn#4659 (reshaping sub-byte and block-quantized tensors; not this crate's
  path, which never reshapes packed weights).

## Layout

- `NN-*.md`: one report per issue; the "Issue draft" section is the text to file, the table
  and "Notes for this crate" are for this repository.
- `repros/<name>/`: standalone Cargo projects against this crate's pin (`burn = "=0.21.0"`;
  `[workspace]` of their own, `publish = false`). `repros/v0.22.0/<name>/`: the same
  reproductions ported to `burn = "=0.22.0"` for the reports to file (0.22.0 removed the
  backend type parameter from `Tensor`, so they are separate projects). `Cargo.lock` and
  `target/` are ignored; to build a 0.21.0 repro with exactly the dependency versions this
  crate validated, copy the crate's `Cargo.lock` into the repro directory first.
- `patches/NN-*.patch`: `git apply -p1` patches against the v0.21.0 sources (Burn patches
  relative to the burn repository root, `crates/<crate>/...`; the gemm patch relative to the
  gemm repository root at 0.19.0, which is also gemm's `main`). Patches 02 and 03 apply
  unchanged to v0.22.0 and `main`.
- `patches/v0.22.0/`: patches against v0.22.0 for the reports to file: 01, 10, 12, 13; the
  `-main` variants of 01 and 12 carry the same changed lines rebased onto `main` at
  `df0812e` (10 and 13 apply to `main` as they are).

`Cargo.toml`'s `include` list does not match anything under `docs/`, so none of this is
packaged (checked with `cargo package --list`); no manifest change is needed.

## Reproducing and verifying a patch

```sh
cd docs/upstream/repros/<name>          # 0.21.0; or repros/v0.22.0/<name>
cargo run --release                     # unpatched (some repros take --features, see each report)
                                        # 0.22.0 repros need Rust 1.95 or newer (e.g. cargo +1.99.1)

# Verify a patch: copy the pristine crates.io sources, apply, and point Cargo at the copy.
mkdir -p /tmp/burn-src/crates
cp -R ~/.cargo/registry/src/index.crates.io-*/burn-flex-0.21.0 /tmp/burn-src/crates/burn-flex
(cd /tmp/burn-src && git apply -p1 <repo>/docs/upstream/patches/01-flex-slice-assign-copy.patch)
cargo run --release --config 'patch.crates-io.burn-flex.path="/tmp/burn-src/crates/burn-flex"'
```

Crates per v0.21.0 patch: 01, 03, 04, 12 `burn-flex`; 02 `burn-autodiff`; 05, 07
`burn-cubecl`; 06 `gemm-common` (run the repro with `--features patched`); 08, 09
`burn-core`; 10 `burn-backend`; 11 `burn-tensor`; 13 `burn-flex` and `burn-backend`. The
patches in `patches/v0.22.0/` are all `burn-flex`; on 0.22.0, reports 02 (`burn-autodiff`)
and 03 (`burn-flex`) use their v0.21.0 patch unchanged. Report 07's repros run twice, with
and without `--features autotune`.

Every v0.21.0 patch applies alone to pristine sources and all together (01-13 in order), and
every patched Burn file passes `rustfmt --edition 2024` with Burn's `max_width = 100`.

## Verification environment

- Apple M3 Max, macOS 27.0.1; Rust 1.94.0-nightly (2025-12-22) for 0.21.0 and Rust 1.99.1
  for 0.22.0 (which requires 1.95); Metal through wgpu (MSL).
- x86-64 for report 06: BU SCC login node `scc1`, Intel Xeon Gold 6242 (FMA, AVX2,
  AVX-512F), Linux, Rust 1.95.0, in the earlier round. The repro and its vendored
  dependencies were built offline in node-local scratch (`/scratch/nover/tfm-upstream-gemm`),
  not in the home quota. Not re-run in this round: SSH to the SCC needs an interactive Duo
  login, and the gemm patch and repro did not change.
- Not run: CUDA, ROCm and Vulkan (reports 05 and 07 were measured on Metal only), and
  Burn's own backend test suite (`burn-backend-tests`, not in the crates.io packages); the
  425 burn-flex unit tests were run with all v0.22.0 burn-flex patches applied.

## Other recorded limitations (not drafted as separate reports)

These are recorded in the source but are feature requests, format limits, or performance
characteristics without a single defect to point at:

- **Autodiff over packed quantized tensors.** Burn 0.21 cannot construct `QFloat` tensors
  on an autodiff backend (it panics) or differentiate through packed matmuls, even with
  frozen weights. The crate rejects this contextually (`src/operations.rs` `linear`,
  `src/checkpoint/runtime.rs`, `src/checkpoint/runtime/packed_install.rs`,
  `src/training.rs`). Upstream ask: return an error instead of panicking, then input
  gradients through frozen packed weights.
- **Tied parameters in optimizers.** Burn 0.21 optimizers update only the first module
  field that owns a `ParamId`; duplicated fields are not aliased (`src/training.rs`
  `validate_trainable_module`, `src/transfer.rs`). Upstream ask: document it or detect
  duplicates in `GradientsParams`.
- **`Module` for `Box<M>`.** Not implemented, so large enum variants hold module graphs
  inline (`src/models/auto.rs`, `src/models/custom/mixer.rs`).
- **Flex performance characteristics** measured in `BENCHMARKS.md`: attention kernels copy
  non-contiguous query/key/value operands on every call, BF16 matmul converts operands to
  contiguous F32 per product, and gemm's single-row product with a strided right-hand side
  is about 2x slower than with contiguous storage (`src/operations.rs`
  "Storage order", `src/checkpoint/runtime/storage_order.rs`).
- **Block-quantized layout ops on Flex 0.22.0** still dequantize and requantize the whole
  tensor (about 188 ms to transpose a `[4096, 4096]` Q4S tensor) although the values are now
  exact; see report 14.
- **CubeCL attention with an explicit scale.** Any `scale: Some(..)`, even the default
  `1/sqrt(head_dim)`, forces the non-fused fallback (`src/models/muse_glimmer/model.rs`).
  A one-line upstream change could treat the default value as `None`.
- **Burnpack format limits** (`UPSTREAM_MAX_TENSORS`, maximum tensor payload in
  `src/checkpoint/quantization/burnpack.rs`) are format constraints, not defects.
- **Checkpoint accounting** in `src/checkpoint/accounting.rs` and the Nemotron-H router
  matrices counted twice upstream concern the model publishers' metadata, not Burn.
