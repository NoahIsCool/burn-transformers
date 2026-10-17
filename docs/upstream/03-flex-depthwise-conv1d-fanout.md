# 03. burn-flex: depthwise `conv1d` fans every channel out to rayon regardless of work size

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-flex` |
| Version reproduced | 0.21.0 (this crate's pin) |
| Kind | Performance (tail latency under CPU contention) |
| Upstream status, checked 2026-10-16 | **Not fixed** on `main`: `conv3d_depthwise_impl` still runs `(0..batch_size * channels).into_par_iter()` with no work threshold. File this one. |
| Repro | [`repros/flex-depthwise-conv1d-fanout`](repros/flex-depthwise-conv1d-fanout) |
| Patch | [`patches/03-flex-depthwise-conv1d-fanout.patch`](patches/03-flex-depthwise-conv1d-fanout.patch) (`crates/burn-flex/src/ops/conv.rs`) |
| Patch verified | Yes: one-column decode convolution, p99 with all cores busy 50.4 ms → 0.49 ms (12.95 ms → 0.58 ms in the isolated patch-only run); results bit-identical to a sequential loop before and after |
| Workarounds an upstream fix removes | `operations::depthwise_conv1d` (`src/operations.rs`, "One-column depthwise convolution") and its callers in `src/delta_net.rs` and `src/mamba2.rs` could call `conv1d` directly again |

## Issue draft

**Title:** burn-flex: depthwise `conv1d` spawns one rayon task per (batch, channel) even for a single output column

### Describe the bug

Recurrent mixers (Mamba-2, Gated DeltaNet) run a `K`-tap depthwise causal convolution on
every decode step. With one new token the convolution has exactly one output column:
`channels · K` multiply-adds, about 25 thousand for 6,144 channels. Flex's depthwise path
nevertheless hands every `(batch, channel)` plane to rayon as a separate task. When other
threads keep the cores busy (a data loader, a second model, a test harness), the calling
thread waits in rayon's join for descheduled workers, and a 10 µs computation takes tens
of milliseconds.

### To reproduce

Standalone project: `repros/flex-depthwise-conv1d-fanout` (`burn = "=0.21.0"`, feature
`flex`). It times `conv1d([1, 6144, 4], [6144, 1, 4], bias, groups = 6144)` against the
same arithmetic as a sequential host loop, first with idle cores and then while one
spinning thread per hardware thread competes for the CPU. It also asserts that Flex's
result equals the sequential loop bit for bit.

### Expected behavior

A convolution this small runs inline and its latency does not depend on how busy the
other cores are.

### Actual behavior (Apple M3 Max, 14 hardware threads)

| | median µs | p99 µs |
|---|---:|---:|
| sequential host loop, idle cores | 11.2 | 13.3 |
| Flex `conv1d`, idle cores | 129.1 | 575.1 |
| sequential host loop, busy cores | 11.1 | 27.3 |
| Flex `conv1d`, busy cores | 182.8 | **50,399.3** |

### Root cause

`crates/burn-flex/src/ops/conv.rs`, `conv3d_depthwise_impl`:

```rust
#[cfg(feature = "rayon")]
{
    (0..batch_size * channels).into_par_iter().for_each(|bc| { ... plane_work(bc, out_plane) });
}
```

Every other parallel loop in Flex is gated by a work threshold
(`ops::PARALLEL_THRESHOLD`, the GEMM thresholds, `MIN_ELEMS_PER_TASK`); this one is not,
so a plane of `out_spatial · kernel = 4` multiply-adds becomes one task.

### Proposed fix

Size tasks by work: each task covers at least `PARALLEL_THRESHOLD` multiply-adds, and a
problem smaller than one task runs inline.

```rust
let plane_work_size = (out_spatial * k_spatial).max(1);
let planes_per_task = crate::ops::PARALLEL_THRESHOLD.div_ceil(plane_work_size);
if batch_size * channels > planes_per_task {
    (0..batch_size * channels).into_par_iter().with_min_len(planes_per_task).for_each(...);
} else {
    for bc in 0..batch_size * channels { ... }   // same body as the non-rayon build
}
```

Measured with the patch: busy-core p99 0.49 ms (idle median 112 µs, p99 127 µs); outputs
unchanged. Large convolutions keep their parallelism. The remaining ~100 µs over the host
loop is per-plane setup in `conv_plane_accumulate` for a one-element plane, which a
dedicated single-column path could remove later.

### Environment

burn 0.21.0, burn-flex default features (`std`, `simd`, `rayon`), Rust 1.94 nightly,
Apple M3 Max, macOS 27.0.1.

## Notes for this crate

- The crate's `depthwise_conv1d` takes the elementwise-product-plus-sum route only for
  the single-column F32 case on inference Flex and is bit-identical to `conv1d`; it can
  stay until the pinned Burn includes a fix, after which the special case is optional.
- Timings under contention are noisy; the effect (tens of milliseconds at p99 versus
  well under one) reproduced in every run.
