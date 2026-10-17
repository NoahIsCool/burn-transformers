# 03. burn-flex: depthwise `conv1d` fans every channel out to rayon regardless of work size

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-flex` |
| Version reproduced | 0.21.0 (this crate's pin) and **0.22.0** |
| Kind | Performance (tail latency under CPU contention) |
| Upstream status, checked 2026-10-16 | **Open.** v0.22.0 reproduces it (measured below); on `main` at `df0812e`, `conv3d_depthwise_impl` still runs `(0..batch_size * channels).into_par_iter()` with no work threshold. File this one. |
| Repro | [`repros/v0.22.0/flex-depthwise-conv1d-fanout`](repros/v0.22.0/flex-depthwise-conv1d-fanout) (to file); [`repros/flex-depthwise-conv1d-fanout`](repros/flex-depthwise-conv1d-fanout) (0.21.0) |
| Patch | [`patches/03-flex-depthwise-conv1d-fanout.patch`](patches/03-flex-depthwise-conv1d-fanout.patch) (`crates/burn-flex/src/ops/conv.rs`); applies unchanged to v0.21.0, v0.22.0 and `main` |
| Patch verified | Yes, Apple M3 Max, one-column decode convolution with all cores busy. v0.22.0: p99 25.8 ms → 0.43 ms. v0.21.0: p99 45.3 ms → 0.35 ms (single runs; earlier runs gave 13-50 ms unpatched). Results bit-identical to a sequential loop before and after; burn-flex's 425 unit tests pass with the patch. |
| Workarounds an upstream fix removes | `operations::depthwise_conv1d` (`src/operations.rs`, "One-column depthwise convolution") and its callers in `src/delta_net.rs` and `src/mamba2.rs` could call `conv1d` directly again |

## Issue draft

**Title:** burn-flex: depthwise `conv1d` spawns one rayon task per (batch, channel) even for a single output column

### Describe the bug

Recurrent mixers (Mamba-2, Gated DeltaNet) run a `K`-tap depthwise causal convolution on
every decode step. With one new token the convolution has exactly one output column:
`channels · K` multiply-adds, about 25 thousand for 6,144 channels. Flex's depthwise path
nevertheless hands every `(batch, channel)` plane to rayon as a separate task. When other
threads keep the cores busy (a data loader, a second model, a test harness), the calling
thread waits in rayon's join for descheduled workers, and a 10 µs computation takes tens of
milliseconds.

### To Reproduce

Standalone project `repros/v0.22.0/flex-depthwise-conv1d-fanout` (`burn = "=0.22.0"`,
feature `flex`), `cargo run --release`. It times
`conv1d([1, 6144, 4], [6144, 1, 4], bias, groups = 6144)` against the same arithmetic as a
sequential host loop, first with idle cores and then while one spinning thread per hardware
thread competes for the CPU. It also asserts that Flex's result equals the sequential loop
bit for bit.

### Expected behavior

A convolution this small runs inline and its latency does not depend on how busy the other
cores are.

### Actual behavior (Apple M3 Max, 14 hardware threads, burn 0.22.0)

| | median µs | p99 µs |
|---|---:|---:|
| sequential host loop, idle cores | 11.2 | 14.4 |
| Flex `conv1d`, idle cores | 104.2 | 213.9 |
| sequential host loop, busy cores | 11.8 | 289.7 |
| Flex `conv1d`, busy cores | 174.0 | **25,849.2** |

(burn 0.21.0 behaves the same: 45,313 µs at p99 with busy cores in the same setup.)

### Root cause

`crates/burn-flex/src/ops/conv.rs`, `conv3d_depthwise_impl`:

```rust
#[cfg(feature = "rayon")]
{
    (0..batch_size * channels).into_par_iter().for_each(|bc| { ... plane_work(bc, out_plane) });
}
```

The elementwise, reduction, gather/scatter, sort and matmul paths size their parallel work
with `ops::PARALLEL_THRESHOLD`, `EXTREMUM_PARALLEL_THRESHOLD`, `MIN_ELEMS_PER_TASK` or the
GEMM thresholds; this loop has no such gate, so a plane of `out_spatial · kernel = 4`
multiply-adds becomes one task. (The pooling loops in `pool.rs` and the other convolution
loops in `conv.rs` fan out per plane or per batch the same way; this report is about the
depthwise one because decode steps hit it every token.)

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

Measured with the patch on v0.22.0: busy-core p99 0.43 ms (median 112 µs; idle median 106
µs, p99 128 µs); outputs unchanged. Large convolutions keep their parallelism. The
remaining ~100 µs over the host loop is per-plane setup in `conv_plane_accumulate` for a
one-element plane, which a dedicated single-column path could remove later.

### Environment

burn 0.22.0, burn-flex default features (`std`, `simd`, `rayon`), Rust 1.99.1, Apple M3
Max, macOS 27.0.1.

## Notes for this crate

- The crate's `depthwise_conv1d` takes the elementwise-product-plus-sum route only for the
  single-column F32 case on inference Flex and is bit-identical to `conv1d`; it can stay
  until the pinned Burn includes a fix, after which the special case is optional.
- Timings under contention are noisy (unpatched busy-core p99 ranged from 13 to 50 ms over
  five runs); tens of milliseconds unpatched versus under half a millisecond patched held
  in every run.
