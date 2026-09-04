# Benchmark and acceptance methodology

This document defines how performance and numerical results for the
`transformers` crate must be collected. It is a protocol, not a claim that every
listed model has been benchmarked. No public benchmark CLI or stable benchmark
harness exists yet; current real-weight evidence comes from opt-in acceptance
tests and is intentionally reported as smoke data rather than a throughput
baseline.

The architecture targets and immutable upstream revisions are recorded in
[ARCHITECTURES.md](ARCHITECTURES.md). A checkpoint becomes advertised only when
its committed acceptance record satisfies the numerical, generation, memory,
and stress gates below.

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
cargo build --manifest-path crates/transformers/Cargo.toml \
  --release --no-default-features --features std,metal,tokenizers
```

The `rocm` feature must not be present. In Burn 0.21 it feature-unifies CubeCL
autotuning into Metal and can expose packed GEMV candidates with incompatible
logical vector widths. The library rejects the known combined-feature path,
and downstream builds that independently enable CubeCL autotuning are not part
of the certified series.

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
| [Qwen3.5-0.8B](acceptance/qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17.json), F16 Metal | Full-vector cosine `0.999998221` against Transformers 5.2/MPS F16; identical reference top-10 and 32 greedy tokens; cached/stateless top-1 agreement at every prefix. The first smoke wall time including load and GPU compilation was `142.52 s`; the cached acceptance wall time including cold load was `158.15 s`. | Standardized warm metrics, peak memory/cache accounting, and wider context runs are pending. |
| Same artifact, Q4 Burnpack | 186 matrices quantized and 134 tensors retained at source precision; full-model Metal logits were finite and cached/stateless cosine was `1.0`. Combined Q8+Q4 conversion/validation took `153.55 s`, which is not a per-format throughput result. | Q4 selected token 13 where F16 selected 11; corpus quality and 32K stress remain pending. Q8 single-backend Metal acceptance remains pending. |
| [Qwen3.8-27B](acceptance/qwen3.8-27b-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.json), F16 Metal | Strictly applied 866 text-plus-MTP tensors from 18 shards and allowlisted exactly 333 visual tensors. A six-token release smoke passed stateless, cached prefill, one MTP prediction, and one recurrent decode; the minimum cosine in each 66-hidden-trace comparison rounded to `1.000000000`. Diagnostic timings were `96.413 s` load, `1.747 s` stateless, `0.279 s` prefill, `0.088 s` MTP, and `0.304 s` decode. Peak RSS was `10,391,191,552` bytes and macOS reported a `63,258,813,376`-byte peak process footprint. | Reference full-vector logits and 32-token generation, Q8/Q4 execution and quality, 32K stress, and standardized warm throughput remain pending. |
| [FLAN-T5-small](acceptance/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.json), F32 Flex | Strictly loaded 190 F32 tensors; the pinned top-10 IDs and ten sampled logits matched, as did cached prefill/decode. Four-beam generation emitted `[644, 4598, 229, 9685, 5, 1]` (`Das Haus ist schön.`), identical to the reference. The `15.33 s` observation includes load and is retained only as a non-benchmark diagnostic; its record lacks the run metadata needed to reproduce it. | Full-vector reference-logit cosine and output hash, 32-token reference generation, accelerator execution, context-boundary stress, standardized warm throughput, and memory metrics are pending. |

Every row remains `advertised_supported: false`. Qwen MoE, DeepSeek V2, Muse
Glimmer, FLAN XXL/UL2, and Nemotron-H implementations currently have
synthetic/reference and checkpoint-contract coverage but no committed
real-weight acceptance run.

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
- correctness and quantized-quality gates pass before any support claim; and
- the corresponding acceptance JSON explicitly sets `advertised_supported`
  only after every required gate is complete.
