# Upstream reports for the Burn 0.21 limitations this crate works around

These are **drafts for review**; nothing here has been filed or posted. Each report has a
standalone reproduction that depends only on the pinned upstream crates, the exact source
location and root cause, a proposed patch against the published sources, and an issue
title and body in the upstream project's style. Every patch was verified by building the
reproduction against a locally patched copy of the crates.io sources via Cargo `[patch]`,
both together with the other patches and alone (one patch applied to pristine 0.21.0
sources).

This crate pins `burn = "=0.21.0"`. Burn **0.22.0** was released on 2026-10-06; the status
column records what its sources (and `main`) show as of 2026-10-16, so already-fixed items
are not filed twice.

## Index

| # | Report | Upstream | Kind | Verified effect (before → after patch) | Upstream status (2026-10-16) | In-crate workarounds a fix would let us remove |
|---|---|---|---|---|---|---|
| 01 | [`slice_assign`/`scatter` copy the whole destination](01-flex-slice-assign-copy.md) | burn-flex | Performance | 64 MiB plane write 1,124 µs → 0.5 µs | `slice_assign` fixed in 0.22.0; `scatter_nd` still copies on `main` | `operations::{assign_region, zero_region, write_region}`; cache write reasoning in `cache/tensor.rs`, `cache/tensor/kv/batch.rs`, `cache/rows.rs`, `cache/tensor/recurrent.rs`; per-step copy in `generation/history.rs` |
| 02 | [Gradients lost after graph merges](02-autodiff-graph-merge-lost-gradient.md) | burn-autodiff | **Correctness** | 75-269 of 1,000 trials lose the gradient → 0; crate's old router test fails → passes | **Open** in 0.22.0 and `main` | `module::linear` avoidance in `moe/router.rs::router_logits` (keep the regression tests) |
| 03 | [Depthwise `conv1d` rayon fan-out](03-flex-depthwise-conv1d-fanout.md) | burn-flex | Performance | Busy-core p99 50.4 ms → 0.49 ms | **Open** on `main` | `operations::depthwise_conv1d` (callers in `delta_net.rs`, `mamba2.rs`) |
| 04 | [Broadcast binary ops take a scalar strided walk](04-flex-broadcast-binary-fast-paths.md) | burn-flex | Performance | RoPE 65x → 1.1x of dense; outer product 32x → 0.7x | Reworked on `main` (`zip` loop nest); re-measure on 0.22.0 | Fused RoPE kernel `position/tensor/flex.rs`; outer-product `input_update` and flattened `SegmentMasks` in `mamba2.rs` |
| 05 | [Flash attention aligns `is_causal` top-left](05-cubecl-flash-attention-causal-alignment.md) | burn-cubecl / cubek-attention | **Correctness** | `FlashUnit` error vs Burn's reference 0.50 → declines; square control still flash | **Open** on `main` | Square-only `is_causal` and explicit bottom-right mask in `attention/dispatch.rs` |
| 06 | [gemm results depend on operand strides](06-gemm-layout-dependent-accumulation.md) | gemm-common | Reproducibility | x86 (SCC Xeon): 5 of 7 layout-dependent → 0; arm64: 3 of 7 → 0 (opt-in setting) | No setting on `main` | Cross-layout tests bounded by F32 rounding (`fe9e795`); `checkpoint/runtime/storage_order.rs` notes |
| 07 | [Packed quantized matmul / `dequantize` wrong](07-cubecl-packed-quantized-matmul.md) | burn-cubecl / cubek-matmul | **Correctness** | Metal: 8 of 24 (no autotune) and 1 of 24 (autotune) wrong → 0; transposed `dequantize` error 1.0 / 1.9e3 → 0 | `dequantize` fixed in 0.22.0; kernels unverified upstream | Packed-Metal autotune guard in `operations.rs`; transposed-`dequantize` note in `checkpoint/runtime/packed_install.rs` |
| 08 | [Mapped (`LinearLayout::Col`) params are not leaves](08-linear-col-layout-gradient.md) | burn-core | **Correctness** | First SGD step skips the weight → updates it; `load_record` panic → loads | `init_mapper` fixed in 0.22.0; load path unconfirmed | `operations::init_linear` autodiff re-wrap; Row layout in `models/t5/model.rs::t5_linear`; install ordering in `checkpoint/runtime.rs` |
| 09 | [`valid()`/`from_inner()` drop param mappers](09-param-mapper-valid.md) | burn-core | **Correctness** | Reloaded square projection error 0.71 → 0; saved shape `[in,out]` → `[out,in]` | Appears fixed on `main` | "Save from the training owner" rule in `training.rs` |
| 10 | [Default `linear` copies a strided weight per call](10-linear-weight-reshape-copy.md) | burn-backend | Performance | 594 MiB transposed readout 94.6 ms → 25.8 ms | Partly addressed on `main` (`MergeBatches` policy) | `operations::matmul_transposed` and tied readouts in `models/muse_glimmer/model.rs`, `models/qwen3_5/model.rs` |
| 11 | [`softplus` overflows](11-softplus-overflow.md) | burn-tensor | **Correctness** | 4 of 10 wrong values/gradients → 0 | **Fixed in 0.22.0** | `delta_net::softplus` (also used by `mamba2.rs`) |
| 12 | [Boolean `select`/`gather` tagged `U8`](12-flex-bool-select-dtype.md) | burn-flex | **Correctness** | 4 of 4 Boolean reads fail → 0 | Probably open on `main` | `architecture/types.rs::select_bool_tensor` integer round trip |
| 13 | [`Flex32` accepted but unsupported on Flex](13-flex-flex32-dtype.md) | burn-flex, burn-backend | **Correctness** | 9 of 9 ops panic → 0 | **Open** on `main` | Flex32 sort guard in `generation.rs` (and its test) |
| 14 | [Block-quantized layout ops requantize on Flex](14-flex-quantized-layout-requantize.md) | burn-flex | **Correctness** | Transpose changes a small row by 39% (Q8) / 100% (Q4); no patch (design change) | **Open** on `main` | `packed_layout_is_exact` probe and `PackedLayoutPolicy` fallback in `checkpoint/runtime/packed_install.rs` |

Suggested filing order: 02, 07 (after re-running Part 2 on 0.22.0), 05, 14, 13, 12, 03,
06; then 01 (`scatter_nd` only), 04 and 10 if they still reproduce on 0.22.0. Do not file
11; 08 and 09 only if their repros still fail on 0.22.0.

## Risk found while preparing these reports

Report 07 shows that on Metal **without** autotuning (the configuration this crate uses for
packed weights) Burn 0.21's matmul returns unrelated values for packed **Q4** weights with
2 to 32 activation rows, in exactly this crate's weight layout (column-major, 64-element
groups). The crate's opt-in Metal packed tests use single-row inputs and its Q4 smokes
compare Burn paths with each other, so they would not notice. This needs a crate-side
check (multi-row Q4 Metal products against a dequantized reference) independent of the
upstream report.

## Layout

- `NN-*.md`: one report per issue; the "Issue draft" section is the text to file, the
  table and "Notes for this crate" are for this repository.
- `repros/<name>/`: standalone Cargo projects (`[workspace]` of their own, `publish =
  false`). `Cargo.lock` and `target/` are ignored here; to build with exactly the
  dependency versions this crate validated, copy the crate's `Cargo.lock` into the repro
  directory first (Cargo trims it to the repro's graph).
- `patches/NN-*.patch`: `git apply -p1` patches. Burn patches are relative to the burn
  repository root (`crates/<crate>/...`) and apply to the v0.21.0 sources; the gemm patch
  is relative to the gemm repository root (`gemm-common/...`) at 0.19.0.

`Cargo.toml`'s `include` list does not match anything under `docs/`, so none of this is
packaged (checked with `cargo package --list`); no manifest change is needed.

## Reproducing and verifying a patch

```sh
cd docs/upstream/repros/<name>
cargo run --release                        # unpatched behaviour (some repros take --features, see each report)

# Verify a patch: copy the pristine crates.io sources, apply, and point Cargo at the copy.
mkdir -p /tmp/burn-src/crates
cp -R ~/.cargo/registry/src/index.crates.io-*/burn-flex-0.21.0 /tmp/burn-src/crates/burn-flex
(cd /tmp/burn-src && git apply -p1 <repo>/docs/upstream/patches/01-flex-slice-assign-copy.patch)
cargo run --release --config 'patch.crates-io.burn-flex.path="/tmp/burn-src/crates/burn-flex"'
```

Crates per patch: 01, 03, 04, 12 `burn-flex`; 02 `burn-autodiff`; 05, 07 `burn-cubecl`;
06 `gemm-common` (run the repro with `--features patched`); 08, 09 `burn-core`; 10
`burn-backend`; 11 `burn-tensor`; 13 `burn-flex` and `burn-backend`. Report 07's repro runs
twice, with and without `--features autotune`.

## Verification environment

- Apple M3 Max, macOS 27.0.1, Rust 1.94.0-nightly (2025-12-22), Metal through wgpu (MSL).
- x86-64 for report 06: BU SCC login node `scc1`, Intel Xeon Gold 6242 (FMA, AVX2,
  AVX-512F), Linux, Rust 1.95.0. The repro and its vendored dependencies were built
  offline in node-local scratch (`/scratch/nover/tfm-upstream-gemm`), not in the home
  quota; no batch job or SGE project was needed for a single-threaded build and a
  sub-second run.
- Not run: CUDA and ROCm (reports 05 and 07 were measured on Metal only).

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
- **CubeCL attention with an explicit scale.** Any `scale: Some(..)`, even the default
  `1/sqrt(head_dim)`, forces the non-fused fallback (`src/models/muse_glimmer/model.rs`).
  A one-line upstream change could treat the default value as `None`.
- **Burnpack format limits** (`UPSTREAM_MAX_TENSORS`, maximum tensor payload in
  `src/checkpoint/quantization/burnpack.rs`) are format constraints, not defects.
- **Checkpoint accounting** in `src/checkpoint/accounting.rs` and the Nemotron-H router
  matrices counted twice upstream concern the model publishers' metadata, not Burn.
