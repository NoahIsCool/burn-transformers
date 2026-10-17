# 05. burn-cubecl: flash attention aligns `is_causal` top-left, the fallback bottom-right

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-cubecl` (kernels in [tracel-ai/cubek](https://github.com/tracel-ai/cubek), `cubek-attention`) |
| Version reproduced | burn 0.21.0, cubek-attention 0.2.0 (this crate's pin) |
| Kind | Correctness (different results from the same call depending on the kernel selected) |
| Upstream status, checked 2026-10-16 | **Fixed in v0.22.0; do not file.** [cubek#406](https://github.com/tracel-ai/cubek/pull/406) ("Fix attention", merged 2026-07-14, `7fb51b0b`) made `LogicalTileMask::should_mask` bottom-right aligned (`pos.1 + seq_q > pos.0 + seq_kv`, commented as matching Burn's fallback) and updated the CPU reference to `j + seq_q > i + seq_kv`. It ships in cubek-attention 0.3.0, which burn-cubecl 0.22.0 uses. The plain `pos.0 < pos.1` rule remains only for a mask built without bounds, and cubek-attention 0.3.0's only global attention (`global/simple/attention.rs`) always passes `(seq_q, seq_kv)`, so tile-aligned non-square shapes (for example `Q = 512, K = 4096`) are covered too. Checked by reading the 0.3.0 sources; the repro was not ported, because v0.22.0 removed the backend type parameter it uses to call each strategy. |
| Repro | [`repros/cubecl-flash-attention-causal-alignment`](repros/cubecl-flash-attention-causal-alignment) |
| Patch | [`patches/05-cubecl-flash-attention-causal-alignment.patch`](patches/05-cubecl-flash-attention-causal-alignment.patch) (`crates/burn-cubecl`) |
| Patch verified | Yes, Metal on Apple M3 Max: `FlashUnit` on a `Q = 512, K = 4097` causal problem differs from Burn's bottom-right reference by 0.50 unpatched and declines with an explicit error patched; the square control still runs the flash kernel |
| Workarounds an upstream fix removes | `src/attention/dispatch.rs` passes `is_causal` only for square planes and otherwise materializes a bottom-right mask ("Causal alignment"); the opt-in test `metal_causal_suffix_matches_eager_reference` guards it |

## Issue draft (not to be filed: fixed in v0.22.0)

Kept as the record of the 0.21.0 behaviour this crate works around.

**Title:** CubeCL attention: flash kernels use top-left causal alignment while `attention_fallback` uses bottom-right (`is_causal` with `seq_q != seq_kv`)

### Describe the bug

`burn::tensor::module::attention(q, k, v, None, None, { is_causal: true })` defines
causality bottom-right when the query and key lengths differ: query row `i` of `Q` is
token `i + K - Q` and sees keys `0..=i + K - Q` (`attention_fallback`'s
`build_causal_mask` offsets the column index for exactly this; Flex agrees). This is the
case for a cached suffix or chunked prefill. The flash kernels of `cubek-attention`, which
`AttentionStrategy::FlashUnit`, `FlashBlackboxAccelerated` and the autotuner run on every
CubeCL backend, mask `row < col` instead (top-left). The same call therefore returns
different values depending on which kernel the autotuner picked; with top-left alignment
the first rows of a suffix attend to almost nothing.

### To reproduce

Standalone project: `repros/cubecl-flash-attention-causal-alignment` (`burn = "=0.21.0"`,
features `metal`, `flex`; `burn-cubecl = "=0.21.0"` to run each strategy explicitly).
It compares Flex, Metal through the public `attention` (autotuned, Burn's default
features), and Metal with `AttentionStrategy::Fallback` and `FlashUnit`, against F64 host
references with bottom-right and top-left alignment.

### Expected behavior

Every kernel matches the bottom-right reference.

### Actual behavior (Metal, Apple M3 Max; max absolute difference)

`Q = 512` queries over `K = 4097` keys, 4 heads, head dim 64:

| | vs bottom-right (Burn) | vs top-left |
|---|---:|---:|
| Flex | 3.9e-7 | 5.0e-1 |
| Metal, autotuned (default) | 2.3e-6 | 5.0e-1 |
| Metal, `Fallback` | 2.3e-6 | 5.0e-1 |
| Metal, `FlashUnit` | **5.0e-1** | 2.6e-4 |

Control, `Q = K = 512` (both alignments coincide): every kernel matches both references
(`FlashUnit` to 2.5e-4).

On this machine the autotuner happened to keep the fallback for these shapes, so the public
call was right; on a device where a flash candidate wins the benchmark (for example F16 with
tensor cores), the public call returns the top-left result.

### Root cause

`cubek-attention` 0.2.0, `src/components/tile/mask.rs`, `LogicalTileMask::should_mask`:

```rust
let causal_masked = self.causal && pos.0 < pos.1;
```

has no `seq_kv - seq_q` offset (its `cpu_reference.rs` uses the same `j > i` rule), while
`burn-backend`'s `attention_fallback` builds the bottom-right mask. `burn-cubecl`'s
`ModuleOps::attention` routes bias, softcap and custom scales to the fallback, but not
non-square causal problems.

### Proposed fix

The proper fix is an offset in the cubek logical mask
(`pos.0 + (seq_kv - seq_q) < pos.1`), passed as a runtime value. Until then, the attached
`burn-cubecl` patch keeps results consistent:

- `flash_attention` returns `AttentionSetupError::InvalidConfig` for
  `is_causal && seq_q != seq_kv`, so the autotuner discards the flash candidates and an
  explicit flash strategy fails loudly instead of returning top-left results;
- `ModuleOps::attention` sends non-square causal problems straight to the fallback.

Measured with the patch: `FlashUnit` declines the `512 x 4097` problem with the new error;
Flex, the autotuned public call and `Fallback` are unchanged; the square control still
runs `FlashUnit`.

### Environment

burn 0.21.0 (`metal` with default features, i.e. autotuning on), cubek-attention 0.2.0,
Rust 1.94 nightly, Apple M3 Max, macOS 27.0.1. Not run on CUDA/ROCm.

## Notes for this crate

- After upgrading to v0.22.0 the square-only `is_causal` rule and the explicit bottom-right
  mask in `src/attention/dispatch.rs` can be removed; keep the opt-in test
  `metal_causal_suffix_matches_eager_reference`, and extend it with a tile-aligned
  non-square case (`Q = 512, K = 4096`) when doing so, since that path was only checked by
  reading the source.
- Observation from the same 0.21.0 run, not part of this issue: `FlashUnit`'s F32 output
  differs from an F64 reference by about 3e-4 even when it is correct (square case),
  against 7e-7 for the fallback, although it is launched with
  `AccumulatorPrecision::Strict(F32)`.
- Not run on CUDA or ROCm.
