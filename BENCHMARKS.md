# Benchmark and acceptance methodology

This document defines how performance and numerical results for the
`transformers` crate must be collected. It is a protocol, not a claim that every
listed model has been benchmarked. No public benchmark CLI or stable benchmark
API exists yet. Opt-in internal tests provide repeated small-model baselines
and separate correctness smoke data; the sections below distinguish them.

The architecture targets and immutable upstream revisions are recorded in
[ARCHITECTURES.md](ARCHITECTURES.md). A checkpoint becomes advertised only when
its committed acceptance record satisfies the numerical, generation, memory,
and stress gates below. The bounded v0.1 profiles and explicit September 10
performance-baseline decision are recorded in
[the release record](acceptance/measurements/v0_1-release-20260910.json).
The prior Qwen slowdown remains unexplained; adopting the controlled baseline
does not relabel earlier failures as successful measurements.

## Reproducibility record

Every publishable benchmark or release-gate run must include enough information
to reproduce both the graph and the executable:

- repository, immutable revision, shard count, relevant file SHA-256 hashes,
  checkpoint dtype, and tokenizer/chat-template hashes;
- crate commit, Rust version, Burn version, release/debug profile, enabled Cargo
  features, backend name, parameter dtype policy, and quantization scheme/group
  size when applicable;
- machine model, CPU/GPU, physical or unified memory, operating system and
  driver/runtime versions, power mode, and whether another workload shared the
  accelerator;
- batch size, valid tokens per row, padding side, prompt fixture/hash, cache
  capacity and window, requested output length, sampling policy and seed;
- warm-up count, measured repetition count, synchronization method, timing
  boundaries, memory measurement tool, and whether tokenization or host-side
  sampling is included.

Results that differ in any of these dimensions are separate benchmark series.
In particular, an all-feature compile gate is not comparable to the certified
single-backend packed-Metal build.

## Standard execution protocol

### 1. Validate before timing

Discover every SafeTensors header, build the architecture mapping plan, verify
the committed hashes, and require a strict `LoadReport`. Record all applied,
transformed, aliased, dtype-converted, quantized, allowlisted, missing, and
unexpected tensors. Abort on any unexplained missing or unexpected key.

Run numerical and generation acceptance before treating performance numbers as
publishable. A fast but incorrect or incompletely loaded graph is not a valid
sample.

### 2. Separate cold and warm costs

Measure these phases independently and synchronize the backend at each timing
boundary:

1. **Strict load:** immediately before local artifact discovery through return
   of `LoadedModel`.
2. **Cold first token:** first already-loaded prefill and selection, including
   lazy backend/kernel compilation. Report it separately from strict load.
3. **Warm first-token latency:** synchronized prefill plus first token selection
   after all shapes used by the sample have been warmed.
4. **Warm decode:** single-token cached steps after prefill. Report model-only
   time and end-to-end time with host sampling separately.

Process-start-to-first-token may be useful operationally, but it must be an
additional metric rather than a substitute for the four phases above.

### 3. Warm-up and sampling

Use at least three unmeasured warm-up iterations for every distinct shape and
then ten measured repetitions. Report median, p95, minimum, and maximum wall
time; retain raw samples in the acceptance artifact. Recreate cache state
between repetitions. Do not average a first-compilation run into warm results.

Use batch one as the required baseline. Additional batches are separate series.
The standard prompt lengths are 32, 128, 512, 2,048, 8,192, and 32,768 valid
tokens where the architecture and hardware can represent them. Never count
padding as prefill work. The current real long-context gate stops at 32K;
larger declared contexts receive synthetic cache/index validation only.

For decode throughput, prefill once and run 128 teacher-forced or forced-length
cached steps so EOS does not change the denominator. Separately measure a
deterministic greedy end-to-end generation using the committed prompt and
normal EOS/stop handling. Record which mode produced each number.

## Metric definitions

| Metric | Definition |
| --- | --- |
| Strict load time | Wall time from `load_local` entry through a successful strict `LoadedModel` return. |
| Cold first-token latency | Synchronized time from the first post-load prefill call through token selection, including lazy compilation. |
| Warm time to first token | Synchronized warm prefill and first selection time; excludes artifact loading and tokenization unless explicitly labeled end-to-end. |
| Prefill throughput | Total valid prompt tokens divided by synchronized warm prefill seconds. Report sequence length and batch. |
| Decode throughput | Number of valid one-token cached steps divided by synchronized decode seconds. Report model-only and sampling-inclusive values separately. |
| Cache bytes per token | Allocated cache payload divided by logical cache capacity and sequence count, with fixed per-sequence state also reported separately. State the analytical formula and validate it against allocation where backend tooling permits. |
| Peak host/device memory | High-water memory above an idle-process baseline for each phase. On unified-memory systems report one unified high-water value instead of inventing a host/device split. |
| Output hash | SHA-256 of canonical little-endian token IDs and, for logit fixtures, canonical little-endian F32 values. Record token IDs directly as well. |
| Quantized quality | Mean negative log-likelihood on the committed fixed corpus, compared with the accepted full-precision run using identical tokenization and masking. |

Peak memory must be sampled often enough to observe transient loading and first
dispatch allocations. Report the tool and interval. Process RSS alone is not a
device-memory measurement on discrete accelerators.

## Numerical and quality gates

An acceptance checkpoint must meet all of the following before it is advertised:

- zero unexplained missing or unexpected tensors after strict loading;
- full-model reference-logit cosine similarity of at least `0.999` for the
  specified mixed BF16/F16 comparison;
- the same greedy continuation for at least 32 tokens on a committed
  stable-margin prompt;
- identical cached and stateless top-1 choices at every compared prefix;
- short-prompt, padded-batch, context-boundary, and applicable 32K quantized
  stress tests;
- recorded load time, peak memory, cold and warm first-token latency, prefill
  and decode throughput, cache bytes per token, and output hash.

On the fixed evaluation corpus, Q8 mean negative log-likelihood may regress by
at most 1% and Q4 by at most 5% relative to the accepted full-precision run.
Performance changes above 5% should alert. A greater-than-10% regression fails
only after a controlled rerun confirms it on the same benchmark series.

## Packed-weight protocol

Q4/Q8 results must record the original checkpoint dtype, eligible and excluded
tensor counts, group size, packed payload and artifact bytes, largest
materialized tensor, and whether the backend actually retained `QFloat` through
matmul. Dequantizing a complete matrix before multiplication is reference
behavior, not efficient packed inference.

For Metal, packed acceptance must use an exact single-backend build:

```sh
cargo build --manifest-path Cargo.toml \
  --release --no-default-features --features std,metal,tokenizers
```

The `rocm` feature must not be present. In Burn 0.21 it feature-unifies CubeCL
autotuning into Metal and can expose packed GEMV candidates with incompatible
logical vector widths. The library rejects the known combined-feature path,
and downstream builds that independently enable CubeCL autotuning are not part
of the certified series.

## First repeated small-model timing series

The internal [FLAN harness](tests/small_model_benchmarks.rs) now records three
warmups and ten measured repetitions for each source length, with 128 forced
cached decode steps. The [2026-09-09 Metal/F32 run](acceptance/measurements/flan-t5-small-metal-benchmark-20260909-run.json)
passed its authenticated first/full-cached-vector and 32-token reference checks.
Its [complete raw report](acceptance/measurements/flan-t5-small-metal-benchmark-20260909.json)
retains all ten samples, input/prediction hashes, exact compiler/features,
checkpoint identity, and synchronization boundaries.

| Valid source tokens | Median warm first token | Model-only decode | Including greedy readback |
| ---: | ---: | ---: | ---: |
| 32 | 48.27 ms | 77.68 tokens/s | 67.07 tokens/s |
| 128 | 21.87 ms | 76.65 tokens/s | 66.90 tokens/s |
| 512 | 32.78 ms | 74.81 tokens/s | 66.08 tokens/s |

Different source shapes select different execution costs; these observations
are not smoothed into monotonic scaling. Single-sample strict load and cold
first-token diagnostics were 154.72 ms and 288.61 ms. Normal-EOS greedy
translation returned six tokens at median 1.14094 s across ten warmed runs.
Self-attention cache payload is 24,576 bytes per target token, with cross K/V
and retained encoder output counted separately in each source-length record.

The already-built executable ran without concurrent agent compilers or GPU
inference. macOS reported maximum RSS 620,298,240 bytes and whole-process peak
memory footprint 790,564,288 bytes. These are **not per-phase peaks above idle**,
and the report does not invent separate host/device figures. This first series
is retained for later controlled comparisons; final integrated replay, CPU
series, per-phase memory, and crate-wide release gates remain outstanding.

The [Qwen3.5-0.8B Metal/F16 baseline](acceptance/measurements/qwen3.5-0.8b-metal-benchmark-20260909-run.json)
uses the same three-warmup/ten-sample/128-forced-step protocol, with complete
source authentication before and after the run. Its official prompt/cached/
teacher references and all 32 greedy decisions passed before timing; minimum
compared cosine was `0.999997815`. [Raw samples](acceptance/measurements/qwen3.5-0.8b-metal-benchmark-20260909.json)
include exact hashes and executable/toolchain metadata.

| Valid source tokens | Median warm first token | Model-only decode | Including greedy readback |
| ---: | ---: | ---: | ---: |
| 32 | 65.54 ms | 35.68 tokens/s | 27.91 tokens/s |
| 128 | 138.60 ms | 35.98 tokens/s | 28.10 tokens/s |
| 512 | 525.46 ms | 35.70 tokens/s | 27.94 tokens/s |

Single-sample strict load was 1.348 s and cold first-token latency 415.53 ms.
The separate natural greedy **model-API loop** took median 1.13824 s for 32
tokens, ending at its length bound. That loop is not a benchmark of the public
`generate_causal` coordinator. Actual exposed cache tensors confirm 12,288 KV
bytes per token plus 19,537,920 fixed recurrent bytes. Whole-process maximum
RSS was 2,463,891,456 bytes and peak footprint 2,907,412,480 bytes; phase-specific
peaks above idle remain unmeasured. The compiled 2K/8K/32K chunked-prefill-only
runner has **not run** and cannot contribute latency claims. It forbids decode
beyond a total 32K context.

The [latest correctness refresh](acceptance/measurements/small-model-rc-refresh-20260910.json)
also executes the separate public `generate_causal` benchmark: all three warmups
and ten measured-partition calls produce the same 32 oracle tokens. This does
**not** establish a new timing baseline. The initially supplied metadata said
AC power, but the host probe found battery power. A sandboxed probe reported AC;
this context-dependent disagreement was reproduced and does not establish a
physical power transition. The guarded host follow-up refused to start on battery. The
unchanged raw report is retained with an explicit rejection that overrides its
incorrect power/comparability labels. A stable-power run remains required.
Public-API, FLAN CPU oracle, populated Q4 32K, and unchanged example replays in
that record pass as correctness checks only. Agent CPU compilation was allowed
during those non-timing checks.

### Separate sampled process-memory runs

The development-only [macOS supervisor](scripts/profile_process_memory.py)
observes an already-built benchmark child using `proc_pid_rusage` with a
content-verified Darwin V4 ABI. Opt-in Unix-socket markers bracket strict load,
cold prefill, each complete warm prefill and 128-step decode loop, and natural
generation. No socket or sampler enters the Rust runtime library.

Each phase first waits for an idle window (10 ms and at least five observations
by default). At a requested 1 ms sampling interval, the supervisor records RSS
and physical footprint, absolute sampled maxima, signed differences from idle
medians, query durations, actual sampling/edge gaps, and errors. PID plus kernel
process-start identity is checked throughout. Kernel lifetime high-water is
retained separately. These sampled maxima are **lower bounds on transient
peaks**, not exact peaks and not separate host/device allocations on unified
memory. Raw timestamped JSONL is authenticated by the compact report.

Instrumented runs are explicitly **not timing-comparable** to the uninstrumented
baselines above. The supervisor rejects incomplete phases, stale/invalid
handshakes, probe errors, identity changes, and timed-out children; it never
replaces a failed observation with zero. Its Python/Rust synthetic checks and
an end-to-end 8MiB allocation smoke passed. Real-model memory evidence is
recorded separately only after each supervised run succeeds.

## 2026-10-15 performance-pass series (shared host)

These runs measure the polish branch `claude/tfm-polish-w4-performance` at
commit `e4011b9` against its base `33f620f` and against the September series
above. They are **not** a new controlled baseline: other agents compiled and
tested on the same CPU throughout (load averages between 17 and 55 were sampled
around every run), while the September runs paused agent compilers. Only one
Metal job ran at a time. Medians below are paired with their observed
minimum–maximum so the contention is visible; compare best samples as well as
medians before drawing conclusions.

| Item | Value |
| --- | --- |
| Machine | Mac15,11, Apple M3 Max (10 performance + 4 efficiency cores, 30-core GPU), 103,079,215,104 bytes unified memory |
| OS / power | macOS 27.0.1 (26A434), OS-provided Metal 4; AC power, battery charged, `pmset` powermode 0 |
| Toolchain | `rustc 1.92.0 (ded5c06cf 2025-12-08)` (`cargo +stable`) for every end-to-end series and the `e4011b9` micro-benchmark runs; the earlier micro-benchmark runs used the worktree default `rustc 1.94.0-nightly (4f14395c3 2025-12-22)` |
| Burn | 0.21.0, lockfile unchanged; release profile |
| Features | FLAN/Qwen Metal: `std,metal,tokenizers,chat-templates`; Flex: `std,flex,tokenizers,chat-templates` |
| Protocol | The unchanged harnesses: three warmups and ten measured repetitions per shape, 128 forced cached decode steps, `Backend::sync` at every boundary |
| Raw reports | `/Volumes/bag of holding/cargo-targets/tfm-polish-w4-performance-reports/` (reports, logs, profiles) and `/Volumes/bag of holding/cargo-targets/tfm-polish-w4-performance-bins/` (the timed executables) on the measurement host, outside the repository; every file is identified by its full SHA-256 in the raw evidence manifest at the end of this section |

### Projection weight storage order

A checkpoint stores a projection as `C = [out, in]`; `operations::linear`
multiplies by `W = Cᵀ`. Burn's `LinearLayout::Col` mapper makes `W` a strided
view of `C`, and before this pass every family loader installed exactly that
view. The `linear_layout_*` micro-benchmarks in
`tests/small_model_benchmarks.rs` time one projection per sample with the
strided view, contiguous `[in, out]` storage, and the crate installer's result.
Flex/F32 medians in µs from two runs of the stable-toolchain binary built at
`e4011b9` (five products per sample, fifteen interleaved samples); the ratio
range spans those two runs and four earlier runs of the same harness at
`0472ac1` and `affbb08`:

| Shape (`in -> out`) | Rows | Strided | Installed | Strided / installed |
| --- | ---: | ---: | ---: | --- |
| 1024 -> 4096 | 1 | 617 / 601 | 311 / 293 | 1.98–2.08 |
| 1024 -> 4096 | 8 | 610 / 526 | 763 / 618 | 0.80–0.97 |
| 1024 -> 1024 | 1 | 152 / 149 | 74 / 73 | 2.02–2.07 |
| 1024 -> 1024 | 8 | 370 / 224 | 183 / 137 | 1.46–2.03 |
| 1024 -> 3584 | 1 | 531 / 527 | 257 / 257 | 2.04–2.06 |
| 1024 -> 3584 | 8 | 724 / 443 | 478 / 339 | 1.16–1.51 |
| 3584 -> 1024 | 1 | 515 / 519 | 256 / 259 | 2.00–2.02 |
| 3584 -> 1024 | 8 | 786 / 653 | 632 / 499 | 1.24–1.41 |
| 2048 -> 1024 | 1 | 295 / 303 | 145 / 143 | 2.03–2.11 |
| 2048 -> 1024 | 8 | 524 / 432 | 291 / 281 | 0.96–1.80 |

At 128 and 512 rows every shape stayed within 0.77–1.22 across the six runs,
i.e. within contention noise. The diagnostic `(C · xᵀ)ᵀ` form, which writes a
column-major result, was the fastest 8-row variant for almost every shape and
run (for example 298–409 µs for `1024 -> 4096`), a further batched-decode
opportunity that is not implemented. Single-row products run on one core below Flex's
`192³` parallel threshold, which is why they are stable; the earlier "4–10x"
estimate from the gpt-oss package came from multi-row products under heavy
rayon contention and was not reproduced. `install_float_parameter` therefore
stores transposing-mapper matrices contiguously on inference Flex (the
`checkpoint::runtime::storage_order` module); values, saved records, dtypes and
`Autodiff<Flex>` are unchanged. On Metal (64 products per sample) the strided
view was equal or faster where it matters for decode: in the first run (a
nightly build of `33f620f` with the harness), single-row F32 products took 18–27% less time strided than contiguous, and
single-row F16 products were equal or up to 20% faster strided, while 128- and
512-row F16 products were 8–15% faster contiguous. Metal therefore keeps the
zero-copy view. Metal per-call times varied by up to 5x between runs for some
F32 shapes, 8-row F32 products took about nine times as long as single-row ones
in every layout, and the diagnostic `(C · xᵀ)ᵀ` form was 2–3x faster for
single-row F32 products in one run only; these are recorded as follow-ups
rather than acted on.

### Tied embedding readout

A Flex CPU profile (`/usr/bin/sample`, 1 ms) of SmolLM-135M cached decode at
`33f620f` spent 1,202 of 1,886 main-thread samples (64%) in
`FlexTensor::to_contiguous` under `burn::tensor::module::linear`: Burn's default
`linear` reshapes its weight to the activation rank, and Flex can only reshape
the transposed embedding table by copying all 113 MB of it on every decode
step. `operations::matmul_transposed` multiplies by the transposed view through
one rank-2 product and never reshapes the table; the Llama graph (and with it
Gemma) and gpt-oss use it. Flex's F32 (measured here), F16 and F64 GEMM then
reads the strided view without copying it, but Burn Flex 0.21's BF16 matmul
still makes both operands contiguous and converts them to F32 on every product,
so a BF16 readout on Flex keeps a per-call copy. CubeCL reshapes the same view
by updating strides, so Metal never paid this cost. The readout still multiplies by a strided view:
the micro-benchmark's `576 -> 49152` single-row product took 1.39–1.51x longer
strided than from a contiguous transposed copy, which would cost a second copy
of the table (113 MB for SmolLM-135M, 2.7 GB for Gemma 3 4B in F32) and is not
made. After both changes the largest remaining Flex decode costs in a SmolLM
profile are the tied readout (30% of samples), the feed-forward projections
(41%, single-threaded below Flex's parallel threshold), naive attention copying
cache views (4%), and whole-buffer KV writes (2%).

### SmolLM-135M on Flex/F32

`causal_lm_flex_decode_benchmark` with `TRANSFORMERS_CAUSAL_LM_BENCH_DIR`
pointing at `HuggingFaceTB/SmolLM-135M` revision `1d461723` (strict load, F32).
Medians with minimum–maximum over ten measured repetitions; decode is model-only
over 128 forced steps per row:

| Shape | `33f620f` prefill ms | `e4011b9` prefill ms (runs 1 / 2) | `33f620f` decode tok/s | `e4011b9` decode tok/s (runs 1 / 2) |
| --- | ---: | ---: | ---: | ---: |
| batch 1, 32 tokens | 198.2 (188.4–379.4) | 107.9 (79.5–682.6) / 87.3 (79.6–131.3) | 7.9 (4.9–8.2) | 65.0 (25.4–73.2) / 70.9 (42.6–73.0) |
| batch 1, 128 tokens | 391.0 (337.1–2,344.6) | 311.1 (234.7–1,115.3) / 280.5 (231.0–1,481.8) | 7.8 (5.5–8.1) | 64.7 (29.5–68.9) / 65.1 (51.2–69.2) |
| batch 8, 32 tokens | 589.2 (510.7–2,655.0) | 388.9 (367.7–557.2) / 374.3 (363.1–393.6) | 36.7 (24.1–43.3) | 142.6 (119.9–169.2) / 159.7 (128.0–169.4) |

Cached decode is 8.2–9.0x faster at batch one and 3.9–4.4x faster at batch
eight; warm prefill is 1.3–2.3x faster. Both binaries were built with `cargo
+stable` from `33f620f` and `e4011b9` with the same benchmark sources; the runs
were interleaved head, base, head between 19:37 and 19:57 local time under load
averages of 22–50, and their `/usr/bin/time -l` wall times were 172 s, 851 s and
155 s. Strict load took 0.10 s at `33f620f` and 0.12–0.15 s at `e4011b9`, which
includes the host transposes.

The final-logit checksums differ between the two binaries only in the last
digits, as expected from F32 reassociation. With both changes the Flex/F32
real-checkpoint gates still reproduce their documented maximum logit deviations
and all 32 greedy tokens: SmolLM-135M `9.1e-5`, Qwen3-4B (`531c80e2`) `1.4e-4`
(`tests/llama_family_artifacts.rs`), and Gemma 3 4B (`093f9f38`) `7.2e-5`,
`1.3e-4` and `1.6e-4` over its three prompts plus its long-context ring and
sliding-window checks (`tests/gemma_artifacts.rs`); Gemma's left-padded batched
prefill moved from the documented `1.1e-4` to `1.2e-4`, well inside the `1e-3`
gate. The FLAN-T5-small
official-reference gate reproduces its earlier Flex output hashes bit for bit.

### FLAN-T5-small and Qwen3.5-0.8B against the September series

No timed code path of either checkpoint changed in this pass (T5 uses
row-layout projections and an untied head, and Metal keeps the strided view;
only FLAN loading now probes parameter mappers and may split large transposes
across threads), so the differences below are the cumulative effect of the
earlier polish waves, measured under contention. Strict load, cold first token and natural greedy
generation are single process observations as in the September series.

FLAN-T5-small, Metal/F32, 19:29–19:31 local time under load averages 31–55
(September 10 controlled series in brackets):

| Source tokens | Prefill median ms (min–max) | First token ms | Model-only decode tok/s (min–max) | Including readback tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 46.66 (45.20–47.55) [48.19] | 49.38 [51.65] | 80.74 (75.17–82.82) [75.11] | 67.79 [65.42] |
| 128 | 17.54 (16.47–34.20) [21.22] | 20.19 [23.04] | 80.23 (66.19–81.39) [74.83] | 67.35 [65.37] |
| 512 | 24.95 (24.33–26.87) [31.26] | 27.63 [33.07] | 76.09 (50.79–77.06) [72.77] | 63.90 [63.32] |

The six-token natural greedy translation took a median 0.1164 s against
1.2042 s on September 10, with identical tokens. Whole-process maximum RSS was
657,129,472 bytes and peak footprint 1,107,477,680 bytes.

FLAN-T5-small, Flex/F32, 19:31–19:32 local time under load averages 20–31
(September 10 controlled series in brackets):

| Source tokens | Prefill median ms (min–max) | Model-only decode tok/s (min–max) |
| ---: | ---: | ---: |
| 32 | 24.48 (23.35–26.53) [28.86] | 213.96 (199.06–218.41) [233.70] |
| 128 | 50.04 (48.11–60.47) [91.88] | 145.94 (142.18–150.52) [159.52] |
| 512 | 212.02 (182.76–1,013.57) [388.82] | 56.23 (40.20–62.46) [68.38] |

Prefill is 1.2–1.8x faster (Flex SIMD and rayon are now enabled) and natural
greedy generation took 0.0409 s against 0.3523 s, but cached decode is 8–18%
slower even in its best samples, and the gap grows with source length. A profile
at 512 source tokens attributes 64% of decode samples to Burn Flex's naive
attention converting non-contiguous key/value views with `to_contiguous`
(scalar `StridedIter`): T5 keeps its cross-attention keys and values as
`swap_dims(1, 2)` views, so every step copies them for every layer. This is a
recorded follow-up for the T5/attention owners, not fixed here.

Qwen3.5-0.8B, Metal/F16, first run at 19:22–19:27 local time under load
averages 29–47 (September 10 controlled series, then the September 9 original,
in brackets):

| Source tokens | Prefill median ms (min–max) | First token ms | Model-only decode tok/s median (min–max) | Including readback tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 60.37 (49.29–330.29) [75.65; 57.73] | 68.46 | 22.06 (7.00–27.61) [26.90; 35.68] | 17.68 [22.01; 27.91] |
| 128 | 62.45 (58.20–121.09) [178.11; 130.83] | 70.50 | 29.32 (13.21–32.11) [26.96; 35.98] | 23.52 [22.09; 28.10] |
| 512 | 158.67 (157.55–435.98) [615.80; 517.83] | 167.73 | 25.45 (5.45–31.80) [26.81; 35.70] | 20.80 [22.07; 27.94] |

A second run of the same binary (20:01–20:06 local time, load averages 24–30)
measured prefill medians of 59.63, 98.46 and 166.42 ms and model-only decode
medians of 29.23 (21.69–31.77), 16.63 (6.36–30.90) and 30.30 (27.57–31.52)
tok/s at 32, 128 and 512 tokens; its natural greedy loop took a median 3.304 s.

All reference checks passed before timing in both runs (prompt cosine
`0.9999985`, minimum cached `0.9999977`, minimum teacher `0.9999962`, 32 greedy
decisions). Warm prefill is 2.9–3.9x faster than the controlled September 10
series at 128 and 512 tokens. Decode is host-dispatch bound: the benchmark
thread was busy in every profile sample, 43% of them inside
`GatedDeltaNet::forward_with_state` issuing kernels, so CPU contention from the
concurrent builds directly stretched decode steps (medians swung from 16.6 to
30.3 tok/s between runs, minimum 5.45 tok/s). The best samples of both runs
(27.6–32.1 tok/s) exceed every September 10 sample (at most 27.18 tok/s) by up
to 18% but stay below the original ~36 tok/s. Whether this pass changed the
historical ~25% slowdown therefore remains **undetermined**: it needs a rerun
with agent work paused, as in September. Strict load took 1.117 s and 1.163 s,
cold first token 0.973 s and 0.362 s. Whole-process maximum RSS was about
2,641,700,000 bytes and peak footprint about 4,366,300,000 bytes in both runs,
above the September 2,907,412,480-byte footprint; that increase is unexplained
and recorded as a follow-up.

### Raw evidence manifest (2026-10-15)

The files below are the complete raw evidence behind this section, hashed with
`shasum -a 256` after the runs. They live outside the repository on the
measurement host, so this table is what authenticates a copy: an archived file
whose SHA-256 differs is not the evidence summarized here. Archiving them under
`acceptance/measurements/` is left to the integrator.

Reports, logs and profiles, relative to
`/Volumes/bag of holding/cargo-targets/tfm-polish-w4-performance-reports/`:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `flan-flex-e4011b9/benchmark.json` | 51,677 | `3f718c83b123099c4a452811af9f3622faa29d1008d7c09b066a794879e14a1f` |
| `flan-flex-e4011b9/command.txt` | 830 | `e243ec6fa4933ce4c1bc698a17b6ad71da888732503f9a3326f675db5b38ad22` |
| `flan-flex-e4011b9/environment.json` | 1,533 | `20b863740ea5f25f5f0af8da34733d8ef39a9ced701208603fad07b41d776417` |
| `flan-flex-e4011b9/stderr.log` | 777 | `3961f1daa85ca3977757c37ccf4b3967fae47fbf8fb4da320f91520ce37cdb13` |
| `flan-flex-e4011b9/stdout.log` | 158 | `50c960430c18a312de4bcadd39507ac4eeece18b06b2c359c3dc9349b96f2aa0` |
| `flan-flex-e4011b9/uptime-after.txt` | 68 | `98c46ecf47b4110d703f8780fe8c0ec7e66694275696b47a36741e3c51435bac` |
| `flan-flex-e4011b9/uptime-before.txt` | 68 | `d7347dfa1e763da3253816ed6533b3031be3ad97398cd79d505333f0fee1d5d5` |
| `flan-metal-e4011b9/benchmark.json` | 51,827 | `955b0b555a8696052c014e77efd60414cf6fed1bb7f65eebfd574c06d295c139` |
| `flan-metal-e4011b9/command.txt` | 834 | `504533103b04fcb7d8d8453c1cd27d9d64126ebe680146d13cf8073368704657` |
| `flan-metal-e4011b9/environment.json` | 1,672 | `0418a63a01e349f64f34f64c86a0f4e6f2e4124aadd6601c3ab016147e81fc5d` |
| `flan-metal-e4011b9/stderr.log` | 777 | `cf1ea448babf7a910c669efda90a7cf531927e23177ae884b561f96c295a58f9` |
| `flan-metal-e4011b9/stdout.log` | 158 | `ed5d0762c1fdce3be99abcb3e356cf29ed5bf533ee5c39cb6665049c242027d4` |
| `flan-metal-e4011b9/uptime-after.txt` | 68 | `d4d61cbd4f7749b995aadb38a112953091b474acb1b9a7ba922daaee4ee11efa` |
| `flan-metal-e4011b9/uptime-before.txt` | 68 | `07f089511969c1b64bd9551d65db98cf5141e0fe3510dbfb88fea4809ccc94ed` |
| `layout/flex-e4011b9.jsonl` | 18,401 | `3931ff7da9280ac42bf01bed5713781ca69806bf112a23de17620d98ffbec362` |
| `layout/metal-e4011b9.jsonl` | 19,887 | `2a37dd396bb45ac18bc0eccf829d8f9bfeb1791e6efb6aa70beba925035a4f0b` |
| `profiles/flan-flex-decode-512-sample.txt` | 561,625 | `fc6e9c261df4156ff0f2eb7ec93044e4e785355b18168730bf2b84a76bca1baa` |
| `profiles/qwen35-metal-decode-sample.txt` | 1,701,525 | `00ddb090e19d6f138646da418a4a22994f82df912821e19d05ba0721b277e555` |
| `profiles/smollm-flex-decode-sample-base-33f620f.txt` | 1,027,327 | `2b3708f107001c43b4e405b7fbae7871f798aae51dcac88de2e30d6c3b174458` |
| `profiles/smollm-flex-decode-sample-d72707d.txt` | 1,132,096 | `87b0ec587ee8f3c521264467aca8990ec888eb0d58d46e331520275f950fdad1` |
| `qwen-metal-e4011b9-r2/benchmark.json` | 119,907 | `b90ff1d33fd8405dddef3db39a1ecfc36967f3b4ea9341b9263a15050c7542a7` |
| `qwen-metal-e4011b9-r2/command.txt` | 778 | `f28315e0f8c871ce074304e3bfdc213b4016f963f4130b2954b5aa597bffbb08` |
| `qwen-metal-e4011b9-r2/environment.json` | 1,616 | `41d39cca0c7cbef8681764310704f5761189d0658b90bd1b21e7cb0de0fc0901` |
| `qwen-metal-e4011b9-r2/stderr.log` | 777 | `d3f5ee4c0a269308c59d57cc56b3e75f3e7877826129f094423177eba9f68a9e` |
| `qwen-metal-e4011b9-r2/stdout.log` | 165 | `f78ab6ef9319b86e83cdb1e7312d74f70e23711e608217b9a84507cdd0024881` |
| `qwen-metal-e4011b9-r2/uptime-after.txt` | 68 | `e0740f4eeca56f0e20459a35f0159764fb8ec1911a5dbfa1a6793ebf99a39ed5` |
| `qwen-metal-e4011b9-r2/uptime-before.txt` | 68 | `dab7adb4051e8831fd350d25d2522f6bcf49952f56f7a6f9459b74d434d61f42` |
| `qwen-metal-e4011b9/benchmark.json` | 119,800 | `fcbc267bae41d2015d076c7e35b8c675302233edbf904c381d6e3b07dc429a2a` |
| `qwen-metal-e4011b9/command.txt` | 772 | `4454e6d909973f4a86af289581ade39c1e75c781cb13faf6e723ced10225a849` |
| `qwen-metal-e4011b9/environment.json` | 1,610 | `2e8aa9fa72c88b1e4fd1062ea9056519274d08481748be90ca9b060175d97823` |
| `qwen-metal-e4011b9/stderr.log` | 777 | `3e2934788470b932e8b9c27758e8b1833c603a26075c103d91df463be2d88f23` |
| `qwen-metal-e4011b9/stdout.log` | 165 | `2c01775e31aae97e6f1f63a1b1065940911695f1fd1acc9de7ed3a672ef95f4e` |
| `qwen-metal-e4011b9/uptime-after.txt` | 68 | `c5acd3a8e32f804bb947e41294f96338d2d52914da3ff8898ea2e4e9e01c4116` |
| `qwen-metal-e4011b9/uptime-before.txt` | 68 | `4453d9ac5842791d54bfe990077e6c44fb301dc244a76c103d5669a4810e2d1c` |
| `smollm-flex/base-33f620f-r1.json` | 4,439 | `67fe69dd370b09682dfe5ec906274da93ea047a11d9171c3293ea72478245c3f` |
| `smollm-flex/base-33f620f-r1.time` | 777 | `61583d5b1539d2d16c296bbc3847d673834fa38590bc7f18652ec473e595ace1` |
| `smollm-flex/e4011b9-r1.json` | 4,412 | `73aaf4964f65937b3a17994f826eba52a28e566ab8744519a19421e1d89f53dd` |
| `smollm-flex/e4011b9-r1.time` | 777 | `b078fa3c74732dd8ccf595a0fc976343985c72728a6ec0c0ebb54f79de8d2aa3` |
| `smollm-flex/e4011b9-r2.json` | 4,392 | `17204bba079691365da0811737f3bbb1e94d3a5c9847eb244118a8c532873336` |
| `smollm-flex/e4011b9-r2.time` | 777 | `3d11a251a9670ad6baeef475a6b0e9479f45c0269ca5979a7a36b3dfcc2f85fc` |
| `smollm-flex/log.txt` | 514 | `890ccaed89786285e0ac801fc08d68166224461c2f17ff5ddf329977067dd282` |

Timed executables, in `/Volumes/bag of holding/cargo-targets/tfm-polish-w4-performance-bins/`:

| Executable | Bytes | SHA-256 | Built from | Backs |
| --- | ---: | --- | --- | --- |
| `small_model_benchmarks-e4011b9-stable-flex` | 15,418,928 | `b80c2629d3934a3f52db791bb83b714dcca7ba61b9f4ec5e8bc2f319c1b41c57` | `e4011b9`, `cargo +stable`, `std,flex,tokenizers,chat-templates` | FLAN Flex series (its `benchmark_executable_sha256`); SmolLM `e4011b9` runs 1 and 2, the Flex layout runs and the FLAN Flex profile (by file name and run log) |
| `small_model_benchmarks-base-33f620f-stable-flex` | 15,307,872 | `395bcd6fe0d3626c92c96444043154f14defe882b891cd6265b5a0525e657aa9` | `33f620f` library with the `d72707d` (= `e4011b9`) benchmark sources, `cargo +stable`, `std,flex,tokenizers,chat-templates` | SmolLM `33f620f` run (by file name and run log) |
| `small_model_benchmarks-e4011b9-stable-metal` | 35,319,424 | `c1cff0ca514b749612e7fbb486baaec0b25a46507e47a409113f75854e3c69d5` | `e4011b9`, `cargo +stable`, `std,metal,tokenizers,chat-templates` | FLAN Metal series (its `benchmark_executable_sha256`); Metal layout runs (by file name) |
| `qwen35_benchmarks-e4011b9-stable-metal` | 39,596,656 | `98dd9771ca2e1c0c8e0eeddbb1195e44858d5e4a7684eb67011a269d7ce38140` | `e4011b9`, `cargo +stable`, `std,metal,tokenizers,chat-templates` | both Qwen3.5 Metal runs (their `benchmark_executable_sha256`) and the Qwen profile |
| `small_model_benchmarks-base-33f620f-flex` | 11,828,688 | `a8e5a7cfadcc33e2ed31bfd7d0a571979dd121f930abf710d2bad00c47a7c823` | `33f620f` library with the benchmark sources, worktree nightly toolchain | SmolLM base profile only |
| `small_model_benchmarks-d72707d-flex` | 11,886,800 | `0ab2f78fbd8161aa52dad6bc055e6bac8e82e67f861bfa1b794ff2a534121914` | `d72707d`, worktree nightly | SmolLM profile after both changes only |

How each file is tied to its source:

- The FLAN and Qwen `benchmark.json` reports embed their provenance: the
  benchmark source hash, `crate_commit` (`e4011b9c6ea3837e43fda93d749e6576a7eae27d`
  in all four), the build tree's `git status`, the toolchain and the executable's
  SHA-256, which matches the table above. The second Qwen run reused the first
  run's executable (identical executable hash) after uncommitted edits for
  `c80e9aa` had been made in the build tree, which is why its recorded
  `crate_worktree_status` lists four modified files. Those edits were not in
  the binary: three touch only checkpoint loading and the fourth is this file.
  The first Qwen run and both FLAN runs record a clean tree.
- The SmolLM JSON reports and the layout JSONL files predate the provenance
  fields that `tests/support/benchmark_identity.rs` now adds to those
  harnesses. Their executable is identified only by file name, the run log
  (`smollm-flex/log.txt`, with load averages at each boundary) and the
  `/usr/bin/time -l` records; they do not authenticate themselves.
- The earlier micro-benchmark runs that widen the projection ratio ranges
  above (four Flex runs at `0472ac1` and `affbb08`, and the first Metal run of a
  nightly `33f620f` build) were not archived. Only the ranges quoted in the text
  remain of them.
- Profiles name their sampled executable in their first line.

## 2026-10-15 performance follow-ups (shared host, battery power)

These runs measure the follow-up branch `claude/tfm-polish-w4-perf-followups`
against its base `ac07281` (the integrated performance pass). Commits:
`bba1f70` reads tied vocabulary heads through `operations::matmul_transposed`;
`a9033ac` stores randomly initialized column-layout projections contiguously
on inference Flex; `dc6a91f` caches T5 cross-attention keys and values in
contiguous storage at prefill. Like the series above they are **not** a
controlled baseline: other agents compiled and tested throughout (load
averages 7–43 at run starts), and the laptop ran on battery (76% falling to
54%, `pmset` powermode 0). Runs of different revisions were interleaved, and
each table reports the median of per-run medians with the range of per-run
medians; compare best samples where contention was heavy.

| Item | Value |
| --- | --- |
| Machine | Mac15,11, Apple M3 Max (10 performance + 4 efficiency cores, 30-core GPU), 103,079,215,104 bytes unified memory |
| OS | macOS 27.0.1 (26A434), Darwin 27.0.0, OS-provided Metal 4 |
| Toolchain | `rustc 1.92.0 (ded5c06cf 2025-12-08)` (`cargo +stable`), release profile, Burn 0.21.0, lockfile unchanged |
| Features | FLAN Flex: `std,flex,tokenizers,chat-templates`; library timing harnesses: default (`std,flex`); Qwen Metal memory: `std,metal,tokenizers,chat-templates` |
| Raw evidence | `/Volumes/bag of holding/cargo-targets/tfm-polish-w4-perf-followups-reports/` on the measurement host, authenticated by its `SHA256SUMS` manifest (see the end of this section) |

### FLAN-T5-small Flex/F32 cached decode

`flan::flan_small_flex_benchmark` (three warmups, ten repetitions, 128 forced
decode steps) from the base binary built from a `git archive ac07281` export
and the `dc6a91f` binary (SHA-256 `7cd9c072…3069`). That executable was
first built from the uncommitted working tree whose library sources equal
`dc6a91f`, so the reports of its first two runs name `ac07281` with modified
files; rebuilding at `dc6a91f` reproduced the identical executable. Three
interleaved runs each, 21:25–21:59 local time:

| Source tokens | Prefill ms `ac07281` | Prefill ms `dc6a91f` | Decode tok/s `ac07281` | Decode tok/s `dc6a91f` | September 10 decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 23.97 (23.76–26.22) | 24.23 (23.79–24.25) | 218.4 (209.6–226.4) | 260.3 (257.4–266.6) | 233.70 |
| 128 | 50.83 (50.33–51.11) | 54.00 (52.83–54.64) | 145.1 (144.0–150.2) | 256.7 (253.2–258.1) | 159.52 |
| 512 | 186.08 (185.22–197.26) | 190.33 (187.87–194.68) | 61.3 (58.8–65.4) | 232.9 (230.9–236.6) | 68.38 |

Cached decode is 1.2x, 1.8x and 3.8x faster and now nearly independent of the
source length; it exceeds the September 10 controlled series at every length
(by 11%, 61% and 3.4x), which closes the 8–18% regression recorded above. The
cost is the one-time contiguous copy at prefill, `2 · layers · source · heads
· d_kv` elements (8 decoder layers and 6 · 64 = 384 projected channels for
FLAN-T5-small, so 786,432 F32 elements at 128 source tokens; about 3 ms there,
within noise at 32 and 512), and whole-process peak footprint rose from
472–479 MB to 509–514 MB.
Predicted tokens are identical in every run. The FLAN-T5-small
official-reference gate (`flan_t5_small_flex_matches_official_reference`)
reproduces the earlier Flex output hashes bit for bit, and a unit test shows
that decode logits from the contiguous cache equal those from the former
strided view exactly.

### Tied vocabulary readout and random-initialization layout on Flex/F32

The ignored `models::custom::tests::perf::tied_vocabulary_decode_throughput`
harness times batch-one cached decode (32-token prompt, 32 decode steps,
two warmups) of seeded random models: `modular-tiny` (width 64, vocabulary
1,024, two attention layers), `modular-small` (width 576, vocabulary 49,152 as
in SmolLM-135M, four layers) and `qwen3.5-small` (Qwen3.5-0.8B's width 1,024
and 248,320-token vocabulary, three DeltaNet layers and one attention layer).
All three were built with the same harness source (SHA-256 `9e242c02…b616`)
on top of `ac07281`, `bba1f70` and `dc6a91f`, and run three times each,
interleaved, 21:49–21:57 local time:

| Model | Decode tok/s `ac07281` | `bba1f70` (tied readout) | `dc6a91f` (+ contiguous init) | Prefill ms `ac07281` / `bba1f70` / `dc6a91f` |
| --- | ---: | ---: | ---: | --- |
| modular-tiny | 358 (225–1,261) | 1,172 (691–1,222) | 1,346 (1,285–1,351) | 2.49 / 0.66 / 0.62 |
| modular-small | 7.5 (6.1–8.2) | 49.9 (48.6–51.5) | 53.1 (31.5–53.6) | 121.9 / 16.3 / 15.4 |
| qwen3.5-small | 5.9 (5.3–6.3) | 19.9 (19.5–19.9) | 22.2 (21.8–22.5) | 190.4 / 74.5 / 77.2 |

Before `bba1f70` every tied readout copied the whole table per call (113 MB
for `modular-small`, 1.02 GB for `qwen3.5-small` in F32), prefill included,
which is why prefill also fell 7.5x and 2.6x. Reading the transposed view
directly makes decode 6.6x and 3.4x faster; contiguous random initialization
adds a further 6% and 11% (single-row products against strided weights are
about twice as slow on Flex, but other work dominates a whole step). The tiny
model is overhead-bound and its `ac07281` runs coincided with load averages
up to 43: its best samples (1,281, 1,273 and 1,383 tok/s) differ by at most
8%, within the contention spread. An earlier round with the harness's
unseeded first version agreed (decode 8.2, 49.4 and 54.9 tok/s for `modular-small`; 6.6,
19.8 and 22.3 tok/s for `qwen3.5-small`). Final-logit checksums agree to F32
reassociation (for example 2189.550982 versus 2189.548077 over 49,152
logits); `bba1f70` and
`dc6a91f` are bit-identical for the modular models, while contiguous storage
reassociates the Qwen hybrid's projections (103.546148 versus 103.541986 over
248,320 logits). Every family's new unit test bounds the readout difference
against the former `module::linear` formulation by `1e-6`.

On the real tied checkpoint, Qwen3.5-0.8B (`2fc06364`) loaded on Flex/F32,
a load-and-decode probe built against `ac07281` and `ce602d1` (sources and
executables in the evidence manifest; two interleaved processes each,
22:32–22:33 local time, load averages 21–28) prefilled the official 23-token
reference prompt and decoded 16 greedy tokens three times per process:
prefill took 314–380 ms at `ac07281` and 215–227 ms at `ce602d1`, and decode
ran at 5.0–5.5 and 11.8–12.1 tok/s (2.2x). Both matched the official
Transformers prompt logits at cosine `0.999999169` (maximum absolute error
`0.019312` and `0.019309`) and all 17 reference greedy tokens; their own
last-position logits differ by at most `2.5e-5` (largest logit magnitude 21.1).
Peak footprint was 4.74–4.77 GB in both.

The existing gpt-oss batched-versus-row-by-row timing test (width 1,024, eight
rows, 32 steps) ran in the same binaries: the row-by-row path, which runs
single-row products, took 1.41 s (1.32–3.92), 1.31 s (1.27–1.36) and 1.09 s
(1.05–2.22) at the three revisions, while the batched path stayed at 1.73–2.55
s. At width 256 neither path changed beyond noise.

### Qwen3.5-0.8B Metal/F16 process memory

These measurements predate `45ac457`, which uploads checkpoint weights as
persistent CubeCL allocations instead of carving them from the pool meant for
activations (on the same host, Qwen2-7B F32 loading fell from a 57.0 GB to a
45.5 GB peak footprint and from 25.6 s to 10.7 s). The load phase below,
including the pool retention it attributes after loading, therefore describes
the earlier loader and must be re-measured; the figures are kept as recorded.

The footprint increase recorded above (4.37 GB at `e4011b9` versus the
2,907,412,480-byte, 2.91 GB, peak of the September 9 benchmark run) is **not**
a regression of the polish pass. September's supervised memory runs peaked at
3,014,105,216 bytes (3.01 GB) on September 9
([record](acceptance/measurements/qwen3.5-0.8b-metal-memory-20260909-supervisor.json),
the first column of the table below) and 2,977,486,656 bytes (2.98 GB) on
September 10
([record](acceptance/measurements/small-model-performance-20260910/qwen-memory-phase-summary.json)).
The supervised memory run (`scripts/profile_process_memory.py`
around `qwen::qwen35_metal_baseline_benchmark`, one Metal job at a time) was
repeated on today's host for three source revisions: `103c6ff` (this
repository's September 10 state), `5ab1fa6` (the pre-polish `main`, September
12) and `ac07281`, and once more at this branch's `9e04750` after review
fixes (22:54–22:58 local time, AC power, load averages 19–26, no other test
binary running). Sampled per-phase maxima of physical footprint (resident set
in parentheses), in GB of 10⁹ bytes, each the maximum over all repetitions of
that phase:

| Phase | September 9 supervised run, macOS 15.7.9 | `103c6ff` | `5ab1fa6` | `ac07281` | `9e04750` |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict load | 2.781 (2.518) | 4.310 (2.546) | 4.309 (2.564) | 4.369 (2.642) | 4.367 (2.641) |
| cold prefill | 2.622 | 4.162 (1.844) | 4.162 (1.842) | 4.184 (1.845) | 4.176 (1.842) |
| 32 tokens, decode | 2.818 | 4.218 (1.871) | 4.197 (1.855) | 4.205 (1.862) | 4.197 (1.857) |
| 128 tokens, prefill | 2.679 | 2.591 (0.269) | 3.164 (0.838) | 2.589 (0.259) | 2.588 (0.254) |
| 512 tokens, decode | 3.000 | 2.737 (0.269) | 2.770 (0.292) | 2.806 (0.263) | 2.804 (0.258) |
| natural greedy | 3.014 | 2.736 (0.269) | 2.766 (0.292) | 2.801 (0.263) | 2.789 (0.258) |
| kernel lifetime maximum | 3.014 | 4.310 | 4.309 | 4.369 | 4.367 |

Source code from September 10 reproduces the 4.31 GB peak on macOS 27.0.1
(Metal 4), so the increase follows the operating-system change from macOS
15.7.9 (Metal 3). On the current OS about 1.6 GB of CPU-visible memory
(resident set 1.84 GB after load) stays allocated after strict load and is
released later, during the warm 32-token series (at its 5th to 10th of 13
prefill/decode repetitions in these four runs), without any call from
this crate, consistent with the CubeCL memory pool's periodic deallocation;
steady-state footprint afterwards (2.59–2.81 GB) is below the September
values. The polish pass adds 60 MB (1.4%) to the load peak and changes the
later phases by at most 70 MB. A separate load-only probe built against
`dc6a91f` showed that `Backend::memory_cleanup` after loading, and even
dropping the loaded model, release none of it (footprint 4.12–4.13 GB before
and after), so the retention lies in the backend's allocator rather than in
any tensor this crate keeps alive.

The `9e04750` run also checks the tied vocabulary readout on Metal. This
branch computes Qwen3.5-0.8B's tied head as one rank-2 product against the
transposed embedding view instead of `module::linear`'s rank-3 broadcast.
The benchmark's in-run official-reference checks reproduce the `ac07281` run
digit for digit: prompt-logit cosine `0.9999985227492314`, minimum cached-row
cosine `0.9999976706692852` and minimum teacher-row cosine
`0.9999962022545661` over the 32 reference rows, the same 32 greedy tokens,
and identical predicted tokens in every timed series. On this batch-one path
both formulations issue the same `[source, 1024] · [1024, 248,320]` product
to CubeCL's non-autotuned matmul, so identical F16 logits are expected. With
several rows the former formulation ran a broadcast batch of per-row products
while the new one runs a single `[rows · positions, 1024]` product, so CubeCL
may choose a different kernel there; the padded two-row case in
`tests/qwen35_release_api.rs` was not run here and remains for the
release-gate rerun. The run recorded commit `9e04750` with a clean worktree
and rustc 1.92.0. Peak footprint is unchanged (4.367 GB versus 4.369 GB).

### Evidence manifest (follow-ups)

Every report, log, supervisor sample file, probe source and timed executable
behind this section is listed with its SHA-256 in `SHA256SUMS` in the
raw-evidence directory named above; that manifest itself has SHA-256
`f6b95fbb14bc0cb8e3566b20db0e0a9abdd6ad7e8a128a48fdf51906b1ae99f0`. (It
extends the first published manifest, SHA-256 `127fbc75…8250`, with the
`9e04750` Metal run, its executable, the final FLAN-T5-small gate log and the
run-comparison script; the earlier entries are unchanged.)
Executables of note: FLAN Flex `ac07281` `6c79ea31…9bdb` and `dc6a91f`
`7cd9c072…3069`; library timing binaries
`26fefbbb…4e7f` (`ac07281` + harness), `1c8e918f…6f47` (`bba1f70` + harness)
and `f7eafb90…af4a` (`dc6a91f` + harness); Qwen Metal `103c6ff`
`f2517245…4663`, `5ab1fa6` `ad783adc…2bcf`, `ac07281` `ad1e4c5f…444b` and
`9e04750` `cf1f04a5…9de6`;
Qwen Flex probe `ac07281` `0fdd0d03…d077` and `ce602d1` `8938eac6…db5b`.

## Current smoke evidence

These values establish correctness progress only. They include cold loading or
compilation, were not collected with the full warm sampling protocol above,
and must not be used as comparative performance claims. Their historic records
also omit some required command, toolchain, profile, machine, timing-boundary,
raw-sample, or output-hash metadata. Wall-clock values in this section are
retained only as non-benchmark diagnostics and are not reproducible benchmark
series.

| Artifact | Recorded evidence | Open gates |
| --- | --- | --- |
| [Qwen3.5-0.8B](acceptance/qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17.json), F16 Metal | Full-vector cosine `0.999998221` against Transformers 5.2/MPS F16; identical reference top-10 and 32 greedy tokens; cached/stateless top-1 agreement at every prefix. The first smoke wall time including load and GPU compilation was `142.52 s`; the cached acceptance wall time including cold load was `158.15 s`. | Later repeated warm timing, sampled phase memory and cache accounting are retained in the September 10 series. The bounded F16 profile is supported with the explicitly accepted historical slowdown; long-context quality/performance remains experimental. |
| Same artifact, Q4/G64 Burnpack v2 | The source/plan-bound sidecar contains 186 packed matrices. The [populated 32K stress record](acceptance/measurements/qwen3.5-0.8b-q4-32k-stress-20260909.json) authenticates the complete artifact and all six target files, then fills 32,767 positions in 128 bounded prefill calls and the final position through recurrent decode. All logits and terminal states remained finite; overflow was atomic and reset matched fresh logits. Analytical cache payload is `422,191,104` bytes, comprising `12,288` KV bytes per token and `19,537,920` fixed recurrent bytes. Synchronized one-shot diagnostics were `50.843 s` prefill including full-vocabulary checks and `0.138 s` final decode/readback. Whole-process peak footprint was `3,047,626,496` bytes and maximum RSS `2,556,231,680` bytes. Raw chunk times, input/output hashes, source/executable identity and timing boundaries are retained. | The repetitive stress input is not a quality corpus or official 32K oracle. No warmup/repetition performance baseline or phase-specific high-water memory is claimed. Q4 NLL and Q8 single-backend acceptance remain pending. |
| [Qwen3.8-27B](acceptance/qwen3.8-27b-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.json), F16 Metal | Strictly applied 851 text tensors from 18 shards for the official numerical gate. Burn matched the pinned Transformers 5.2/MPS F16 full-vocabulary prompt logits at cosine `0.999999881`, exact top-1 token 799, all 32 cached greedy tokens, and stateless top-1 at every generated prefix. That cold gate took `218.79 s`; peak RSS was `11,254,513,664` bytes and macOS reported a `63,811,727,296`-byte peak process footprint. A separate text-plus-MTP smoke applied 866 tensors and passed MTP, recurrent decode, and 66 hidden-trace comparisons. | Quantized reference/quality, 32K stress, output hashing, and standardized warm latency/throughput remain pending. |
| Same Qwen3.8 artifact, Q8 Burnpack plus retained F16 Metal tensors | Streaming conversion packed 497 BF16 matrices with group size 64, retained 369 tensors, applied all 866 mapped tensors while keeping at most one source shard open, and allowlisted 333 visual tensors. The validated payload is `27,223,326,720` bytes in a `27,223,642,316`-byte artifact; the largest materialized tensor was `1,350,860,800` bytes. Stateless and cached logits were finite with top-1 token 271 and cosine `1.000000000`; MTP and recurrent decode were also finite. The cold diagnostic test reported `985.53 s`, while its enclosing process took `1241.12 s`; peak RSS was `29,786,243,072` bytes and macOS peak process footprint was `34,317,474,944` bytes. These are end-to-end smoke observations, not standardized benchmark timings. | Reference parity, longer generation, Q8 corpus quality, output hashing, 32K stress, and standardized latency/throughput remain pending. |
| Same Qwen3.8 artifact, Q4 Burnpack plus retained F16 Metal tensors | Streaming conversion packed 497 BF16 matrices with group size 64, retained 369 tensors, applied all 866 mapped tensors while keeping at most one source shard open, and allowlisted 333 visual tensors. The validated payload is `14,412,349,440` bytes in a `14,412,672,972`-byte artifact; the largest materialized tensor was `715,161,600` bytes. Stateless and cached logits were finite with top-1 token 271 and cosine `1.000000000`; MTP and recurrent decode were also finite. Both the cold diagnostic test (`966.10 s`) and its enclosing process (`966.15 s`) include conversion, validation, loading, and smoke execution; peak RSS was `16,752,263,168` bytes and macOS peak process footprint was `20,688,689,408` bytes. These are not standardized benchmark timings. | Reference parity, longer generation, Q4 corpus quality, output hashing, 32K stress, and standardized latency/throughput remain pending. |
| [Qwen3-30B-A3B](acceptance/qwen3-30b-a3b-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39.json), F16 Metal | Strict loading applied all 18,867 tensors from 16 shards. Source-ordered execution opened each shard once, kept at most one open, and reported actual peak host staging of one `622,329,856`-byte tensor. A six-token release smoke passed stateless, cached prefill, all 50 hidden-trace comparisons, and one recurrent decode; cached/stateless logit and minimum hidden-trace cosine both rounded to `1.000000000`. Diagnostic timings were `74.185 s` load, `1.867 s` stateless, `1.509 s` prefill, and `0.726 s` decode. Peak RSS was `2,326,970,368` bytes and macOS reported a `66,569,296,256`-byte peak process footprint. | Official reference full-vector logits, 32-token generation, quantized quality, 32K stress, and standardized warm throughput remain pending. |
| Same Qwen3-MoE artifact, Q8 Burnpack v2 plus retained F16 Metal tensors | Streaming conversion bound all 16 source-shard hashes and the exact F16 load plan, packed 18,625 matrices with group size 64, retained 242 tensors, and published a fully revalidated `32,110,057,725`-byte artifact containing `32,096,174,080` payload bytes. The six-token smoke selected token 1096, decoded token 374, and produced `1.000000000` cached/stateless logit and minimum hidden-trace cosine. A separate traced decode observed exactly 1,345 packed Metal matmuls and zero whole-weight dequantizations. The cold smoke took `868.75 s` process wall time and peaked at `38,081,382,336` bytes of reported process footprint; the traced gate took `889.80 s` and peaked at `38,074,272,064` bytes. | Official reference parity, 32-token generation, Q8 corpus quality, 32K stress, and standardized warm latency/throughput remain pending. |
| Same Qwen3-MoE artifact, Q4 Burnpack v2 plus retained F16 Metal tensors | Streaming conversion used the same exact provenance and partition, producing a fully revalidated `17,006,255,982`-byte artifact with `16,992,092,160` payload bytes. The six-token smoke selected token 1096, decoded token 374, and produced `1.000000000` cached/stateless logit and minimum hidden-trace cosine. Its traced decode likewise observed exactly 1,345 packed Metal matmuls and zero whole-weight dequantizations. The cold smoke took `641.39 s` process wall time and peaked at `23,441,559,232` bytes of reported process footprint; the traced gate took `635.84 s` and peaked at `23,448,391,360` bytes. | Official reference parity, 32-token generation, Q4 corpus quality, 32K stress, and standardized warm latency/throughput remain pending. |
| [DeepSeek-V2-Lite](acceptance/deepseek-v2-lite-604d5664dddd88a0433dbae533b7fe9472482de0.json), F16 Metal | Strict loading applied all 5,291 tensors from four shards, performed 5,099 renames and 5,291 BF16-to-F16 conversions, kept at most one shard open, and reported actual peak host staging of one `419,430,400`-byte tensor. A six-token smoke passed stateless and reference-unabsorbed cached prefill with `1.000000000` logit and minimum-hidden-trace cosine. One compressed-latent absorbed decode matched the internal seven-token stateless path at logit cosine `0.999998964`; its minimum hidden-trace cosine was `0.999978353`. The cache uses 576 scalars per layer/token, or `31,104` bytes per token across 27 F16 layers. Diagnostic timings were `22.376 s` load, `1.317 s` stateless, `0.273 s` prefill, `0.365 s` absorbed decode, and `0.331 s` for the extended stateless oracle. Peak RSS was `2,913,566,720` bytes and macOS reported a `33,982,535,552`-byte peak process footprint. These comparisons are between Burn execution paths, not against the official implementation. | Official reference full-vector logits, 32-token generation, all-prefix cached parity, Q8/Q4 acceptance and quality, padded/context-boundary tests, 32K stress, and standardized warm throughput remain pending. |
| [Muse-Glimmer-30B](acceptance/muse-glimmer-30b-a4e59da52a7bc87ae7251dd5545c0dd437c44b68.json), F16 Metal | The 59.6 GB two-shard checkpoint and all seven schema-3 oracle files are content-authenticated. Strict loading applied 627 text tensors as F16, including 626 renames and 627 BF16-to-F16 conversions, while explicitly allowlisting 809 visual tensors and reporting no unexplained anomalies. Official/Burn cosine was `0.999999616` for prompt and cached-prompt logits; the minima were `0.999943377` for stateless and cached-prefill hidden traces, `0.999987445` for recurrent-decode hidden traces, `0.999999616` across 32 cached rows, and `0.999999333` across 32 teacher rows. Burn cached/teacher minimum cosine was `0.999999726`, cached-prefill/stateless hidden parity was `1.0`, and all 32 greedy tokens matched. The integrated release gate took `331.84 s`; it is a cold end-to-end correctness diagnostic, not a standardized performance sample. The default exact-eager policy bounds each score-like plane to `2^26` elements through synchronized query chunks. A separate 4,097-token Metal trace selected `cubek_attention` for the flash control but not Muse, and proved the two paths' distinct F16 bits while Muse executed 9 chunks of at most 512 queries. A `0.89 s` synthetic envelope gate retained 32 query heads, two KV heads, F16, actual 32K keys, the 2,048-token sliding boundary, terminal RoPE position 32,767, and full/mirrored cache boundaries while reducing head width to four and omitting dense blocks. The exact load-plan gate partitions 417 packed matrices from 210 retained F16 tensors; its shape arithmetic predicts `28,165,343,232` Q8 or `14,911,064,064` Q4 payload bytes. Nonzero two-layer fixtures passed both formats through automatic mixed loading and stateless, cached-prefill, and recurrent-decode execution; a row-unaligned near miss was rejected as reference-only. These fixture results are correctness gates, not real-model performance or quality measurements. | Q8/Q4 official-reference and quality evaluation, a real full-model 32K stress run, real-checkpoint padded/context-boundary cases, memory/output hashes, and standardized warm latency/throughput remain pending. |
| Same Muse artifact, Q8 Burnpack v2 plus retained F16 Metal tensors | The create-only streaming gate packed 417 G64 matrices, retained 210 F16 tensors, applied all 627 text tensors, reported 626 renames and 627 BF16-to-F16 conversions, and allowlisted 809 visual tensors. It opened both source shards once with at most one open, then bound pre- and post-execution hashes of all `59,553,435,272` checkpoint bytes and the exact F16 plan. Full reopen validated every payload. The `28,165,645,337`-byte artifact contains `28,165,343,232` payload bytes; its SHA-256 is `91cfd35bfe0a389cbc2eed2189c4e33312bdce1a0ff50d8452acf06d00f0e8eb`. Conversion took `1,570.845 s`. A separate cold Metal gate reauthenticated the sidecar and source provenance, then reported `595.993 s` load, `5.134 s` stateless, `0.561 s` cached prefill, and `0.247 s` recurrent decode within `743.75 s` total. Cached/stateless logit cosine was `1.000000000`; prompt and decode top-1 IDs were 85,068 and 90. Its trace observed all 417 expected packed matmuls and zero calibrated whole-weight dequantizations. These are correctness diagnostics, not standardized throughput. | Official Q8 reference/quality evaluation, 32-token generation, output hashing, 32K stress, process-memory capture, and standardized warm performance remain pending; not advertised. |
| Same Muse artifact, Q4 Burnpack v2 plus retained F16 Metal tensors | The create-only streaming gate used the same 417-matrix G64 partition, retained 210 F16 tensors, applied all 627 text tensors, and bound both source-shard hashes plus the exact F16 plan before and after execution. It published and fully reopened a `14,911,372,485`-byte artifact containing `14,911,064,064` payload bytes; its largest payload is `756,467,712` bytes and its SHA-256 is `05b69f67799b4dc0228b4298e39d466d753fcc5d6484f61c8920c551017daed3`. Conversion took `1,359.922 s`. A separate cold Metal gate reauthenticated the sidecar and source provenance, then reported `459.031 s` load, `2.636 s` stateless, `0.189 s` cached prefill, and `0.248 s` recurrent decode within `543.38 s` total. Cached/stateless logit cosine was `1.000000000`; prompt and decode top-1 IDs were 85,068 and 90. Its trace observed all 417 expected packed matmuls and zero calibrated whole-weight dequantizations. These are correctness diagnostics, not standardized throughput. | Official Q4 reference/quality evaluation, 32-token generation, output hashing, 32K stress, process-memory capture, and standardized warm performance remain pending; not advertised. |
| [FLAN-T5-small](acceptance/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.json), F32 Flex and Metal | Both backends strictly loaded all 190 F32 tensors and matched an authenticated Transformers 5.2 CPU/F32 oracle for complete encoder, decoder-hidden, first-logit, cached-32, teacher-32, right-padded batch, and 512-token boundary outputs. Minimum cosine was `0.999999999999`, maximum absolute error was `0.000076294`, and every tensor had zero `atol=rtol=1e-4` failures. All 32 stable-margin greedy decisions and the exact four-beam `[644, 4598, 229, 9685, 5, 1]` continuation matched. Canonical per-backend Burn F32 output hashes and the reference-bundle hash are recorded in the acceptance record. One debug-profile Flex diagnostic reported `3.674 s` load, `3.659 s` stateless, `7.440 s` cached-32, `6.911 s` teacher-32, `37.934 s` boundary-512, and `12.674 s` beam execution. The corresponding Metal diagnostic reported `3.106 s`, `0.602 s`, `2.383 s`, `0.649 s`, `0.240 s`, and `4.911 s`. | Later source-bound Metal/Flex timing, sampled Metal phase memory, integrated replays and source gates passed; the bounded F32 profiles are supported for v0.1. These older one-shot values remain correctness diagnostics, not the adopted performance baseline. |
| [FLAN-T5-XXL](acceptance/flan-t5-xxl-ae7c9136adc7555eeccc78cdd960dfd60fb346ce.json), F32 Metal | Strict loading applied 558 runtime tensors from five official shards and verified one physical embedding equivalence. Burn matched official Transformers 5.2/MPS F32 encoder, decoder-hidden, first-logit, 32-row cached-logit, and 32-row teacher-forced fixtures at minimum cosine `0.999999999998`; all 32 greedy tokens and the EOS-terminated four-beam continuation matched exactly. The one-shot cold diagnostic reported `71.107 s` load and `92.06 s` process wall time. Peak RSS was `11,276,173,312` bytes and macOS peak process footprint was `50,210,770,368` bytes. Canonical SHA-256 output hashes and analytical `65,273,856`-byte cache payload are recorded. | Padded/context-boundary execution, Q8/Q4 quality and 32K stress, and standardized warm latency/throughput remain pending. |
| [Nemotron 3.5 Lightning](acceptance/nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json), F16 Metal | The clean cold gate authenticated the complete artifact, strictly applied all 6,513 tensors from fourteen shards with one shard open at a time, and retained 24 FP32 router-correction buffers while converting 6,489 BF16 tensors to F16. Stateless and cached-prefill hidden traces were bit-identical; one recurrent decode matched its teacher-forced top-1 with full-vocabulary cosine `0.999376514`. Diagnostic timings were `102.101 s` load, `8.188 s` stateless, `2.002 s` cached prefill, `1.536 s` recurrent decode, and `319.02 s` including full artifact hashing. These are Burn internal-consistency observations rather than an official-reference comparison. | Official base oracle parity, executable MTP/cache/speculation parity, 32-token generation, Q8/Q4 quality, 32K stress, output/memory reporting, and standardized warm throughput remain pending. |

Only the bounded Qwen3.5-0.8B Metal/F16 base-text and FLAN-T5-small Flex/F32 /
Metal/F32 profiles are now supported. Their
[v0.1 release record](acceptance/measurements/v0_1-release-20260910.json) adopts
the controlled September 10 Qwen timing baseline with explicit user approval,
retaining the unexplained historical slowdown and all old measurements. It
also selects the separate public-coordinator and FLAN timing series for their
own measurement boundaries. Earlier rows describe historical observations,
not new benchmarks or blanket support for other profiles. Muse Glimmer has a committed
official-reference short-context correctness run. Nemotron-H now has a
committed authenticated real-weight Metal base-model smoke, but that run is
Burn internal-consistency evidence rather than official-reference or MTP
parity. FLAN-UL2 still has synthetic and checkpoint-contract coverage without a
real-weight inference acceptance run.

## Reporting checklist

Before committing a new result, verify that:

- the checkpoint revision and every relevant artifact hash are immutable and
  recorded;
- the exact command, Cargo feature set, backend, dtype, hardware, and runtime
  versions are present;
- cold and warm measurements are not mixed;
- backend synchronization surrounds timed work;
- valid-token rather than padded-token counts form throughput denominators;
- raw samples, medians, p95 values, memory high-water marks, and output hashes
  are retained;
- correctness and, for quantized support, the applicable quantized-quality
  gates pass before a support claim; and
- the corresponding acceptance JSON explicitly sets `advertised_supported`
  only after every required gate is complete.
