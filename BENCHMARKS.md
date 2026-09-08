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
| [Qwen3.8-27B](acceptance/qwen3.8-27b-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.json), F16 Metal | Strictly applied 851 text tensors from 18 shards for the official numerical gate. Burn matched the pinned Transformers 5.2/MPS F16 full-vocabulary prompt logits at cosine `0.999999881`, exact top-1 token 799, all 32 cached greedy tokens, and stateless top-1 at every generated prefix. That cold gate took `218.79 s`; peak RSS was `11,254,513,664` bytes and macOS reported a `63,811,727,296`-byte peak process footprint. A separate text-plus-MTP smoke applied 866 tensors and passed MTP, recurrent decode, and 66 hidden-trace comparisons. | Quantized reference/quality, 32K stress, output hashing, and standardized warm latency/throughput remain pending. |
| Same Qwen3.8 artifact, Q8 Burnpack plus retained F16 Metal tensors | Streaming conversion packed 497 BF16 matrices with group size 64, retained 369 tensors, applied all 866 mapped tensors while keeping at most one source shard open, and allowlisted 333 visual tensors. The validated payload is `27,223,326,720` bytes in a `27,223,642,316`-byte artifact; the largest materialized tensor was `1,350,860,800` bytes. Stateless and cached logits were finite with top-1 token 271 and cosine `1.000000000`; MTP and recurrent decode were also finite. The cold diagnostic test reported `985.53 s`, while its enclosing process took `1241.12 s`; peak RSS was `29,786,243,072` bytes and macOS peak process footprint was `34,317,474,944` bytes. These are end-to-end smoke observations, not standardized benchmark timings. | Reference parity, longer generation, Q8 corpus quality, output hashing, 32K stress, and standardized latency/throughput remain pending. |
| Same Qwen3.8 artifact, Q4 Burnpack plus retained F16 Metal tensors | Streaming conversion packed 497 BF16 matrices with group size 64, retained 369 tensors, applied all 866 mapped tensors while keeping at most one source shard open, and allowlisted 333 visual tensors. The validated payload is `14,412,349,440` bytes in a `14,412,672,972`-byte artifact; the largest materialized tensor was `715,161,600` bytes. Stateless and cached logits were finite with top-1 token 271 and cosine `1.000000000`; MTP and recurrent decode were also finite. Both the cold diagnostic test (`966.10 s`) and its enclosing process (`966.15 s`) include conversion, validation, loading, and smoke execution; peak RSS was `16,752,263,168` bytes and macOS peak process footprint was `20,688,689,408` bytes. These are not standardized benchmark timings. | Reference parity, longer generation, Q4 corpus quality, output hashing, 32K stress, and standardized latency/throughput remain pending. |
| [Qwen3-30B-A3B](acceptance/qwen3-30b-a3b-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39.json), F16 Metal | Strict loading applied all 18,867 tensors from 16 shards. Source-ordered execution opened each shard once, kept at most one open, and reported actual peak host staging of one `622,329,856`-byte tensor. A six-token release smoke passed stateless, cached prefill, all 50 hidden-trace comparisons, and one recurrent decode; cached/stateless logit and minimum hidden-trace cosine both rounded to `1.000000000`. Diagnostic timings were `74.185 s` load, `1.867 s` stateless, `1.509 s` prefill, and `0.726 s` decode. Peak RSS was `2,326,970,368` bytes and macOS reported a `66,569,296,256`-byte peak process footprint. | Official reference full-vector logits, 32-token generation, quantized quality, 32K stress, and standardized warm throughput remain pending. |
| Same Qwen3-MoE artifact, Q8 Burnpack v2 plus retained F16 Metal tensors | Streaming conversion bound all 16 source-shard hashes and the exact F16 load plan, packed 18,625 matrices with group size 64, retained 242 tensors, and published a fully revalidated `32,110,057,725`-byte artifact containing `32,096,174,080` payload bytes. The six-token smoke selected token 1096, decoded token 374, and produced `1.000000000` cached/stateless logit and minimum hidden-trace cosine. A separate traced decode observed exactly 1,345 packed Metal matmuls and zero whole-weight dequantizations. The cold smoke took `868.75 s` process wall time and peaked at `38,081,382,336` bytes of reported process footprint; the traced gate took `889.80 s` and peaked at `38,074,272,064` bytes. | Official reference parity, 32-token generation, Q8 corpus quality, 32K stress, and standardized warm latency/throughput remain pending. |
| Same Qwen3-MoE artifact, Q4 Burnpack v2 plus retained F16 Metal tensors | Streaming conversion used the same exact provenance and partition, producing a fully revalidated `17,006,255,982`-byte artifact with `16,992,092,160` payload bytes. The six-token smoke selected token 1096, decoded token 374, and produced `1.000000000` cached/stateless logit and minimum hidden-trace cosine. Its traced decode likewise observed exactly 1,345 packed Metal matmuls and zero whole-weight dequantizations. The cold smoke took `641.39 s` process wall time and peaked at `23,441,559,232` bytes of reported process footprint; the traced gate took `635.84 s` and peaked at `23,448,391,360` bytes. | Official reference parity, 32-token generation, Q4 corpus quality, 32K stress, and standardized warm latency/throughput remain pending. |
| [DeepSeek-V2-Lite](acceptance/deepseek-v2-lite-604d5664dddd88a0433dbae533b7fe9472482de0.json), F16 Metal | Strict loading applied all 5,291 tensors from four shards, performed 5,099 renames and 5,291 BF16-to-F16 conversions, kept at most one shard open, and reported actual peak host staging of one `419,430,400`-byte tensor. A six-token smoke passed stateless and reference-unabsorbed cached prefill with `1.000000000` logit and minimum-hidden-trace cosine. One compressed-latent absorbed decode matched the internal seven-token stateless path at logit cosine `0.999998964`; its minimum hidden-trace cosine was `0.999978353`. The cache uses 576 scalars per layer/token, or `31,104` bytes per token across 27 F16 layers. Diagnostic timings were `22.376 s` load, `1.317 s` stateless, `0.273 s` prefill, `0.365 s` absorbed decode, and `0.331 s` for the extended stateless oracle. Peak RSS was `2,913,566,720` bytes and macOS reported a `33,982,535,552`-byte peak process footprint. These comparisons are between Burn execution paths, not against the official implementation. | Official reference full-vector logits, 32-token generation, all-prefix cached parity, Q8/Q4 acceptance and quality, padded/context-boundary tests, 32K stress, and standardized warm throughput remain pending. |
| [Muse-Glimmer-30B](acceptance/muse-glimmer-30b-a4e59da52a7bc87ae7251dd5545c0dd437c44b68.json), F16 Metal | The 59.6 GB two-shard checkpoint and all seven schema-3 oracle files are content-authenticated. Strict loading applied 627 text tensors as F16, including 626 renames and 627 BF16-to-F16 conversions, while explicitly allowlisting 809 visual tensors and reporting no unexplained anomalies. Official/Burn cosine was `0.999999616` for prompt and cached-prompt logits; the minima were `0.999943377` for stateless and cached-prefill hidden traces, `0.999987445` for recurrent-decode hidden traces, `0.999999616` across 32 cached rows, and `0.999999333` across 32 teacher rows. Burn cached/teacher minimum cosine was `0.999999726`, cached-prefill/stateless hidden parity was `1.0`, and all 32 greedy tokens matched. The integrated release gate took `331.84 s`; it is a cold end-to-end correctness diagnostic, not a standardized performance sample. The default exact-eager policy bounds each score-like plane to `2^26` elements through synchronized query chunks. A separate 4,097-token Metal trace selected `cubek_attention` for the flash control but not Muse, and proved the two paths' distinct F16 bits while Muse executed 9 chunks of at most 512 queries. A `0.89 s` synthetic envelope gate retained 32 query heads, two KV heads, F16, actual 32K keys, the 2,048-token sliding boundary, terminal RoPE position 32,767, and full/mirrored cache boundaries while reducing head width to four and omitting dense blocks. The exact load-plan gate partitions 417 packed matrices from 210 retained F16 tensors; its shape arithmetic predicts `28,165,343,232` Q8 or `14,911,064,064` Q4 payload bytes. Nonzero two-layer fixtures passed both formats through automatic mixed loading and stateless, cached-prefill, and recurrent-decode execution; a row-unaligned near miss was rejected as reference-only. These fixture results are correctness gates, not real-model performance or quality measurements. | Real Q4 conversion/execution, Q8/Q4 official-reference and quality evaluation, a real full-model 32K stress run, real-checkpoint padded/context-boundary cases, memory/output hashes, and standardized warm latency/throughput remain pending. |
| Same Muse artifact, Q8 Burnpack v2 plus retained F16 Metal tensors | The create-only streaming gate packed 417 G64 matrices, retained 210 F16 tensors, applied all 627 text tensors, reported 626 renames and 627 BF16-to-F16 conversions, and allowlisted 809 visual tensors. It opened both source shards once with at most one open, then bound pre- and post-execution hashes of all `59,553,435,272` checkpoint bytes and the exact F16 plan. Full reopen validated every payload. The `28,165,645,337`-byte artifact contains `28,165,343,232` payload bytes; its SHA-256 is `91cfd35bfe0a389cbc2eed2189c4e33312bdce1a0ff50d8452acf06d00f0e8eb`. Conversion took `1,570.845 s`. A separate cold Metal gate reauthenticated the sidecar and source provenance, then reported `595.993 s` load, `5.134 s` stateless, `0.561 s` cached prefill, and `0.247 s` recurrent decode within `743.75 s` total. Cached/stateless logit cosine was `1.000000000`; prompt and decode top-1 IDs were 85,068 and 90. Its trace observed all 417 expected packed matmuls and zero calibrated whole-weight dequantizations. These are correctness diagnostics, not standardized throughput. | Official Q8 reference/quality evaluation, 32-token generation, output hashing, 32K stress, process-memory capture, and standardized warm performance remain pending; not advertised. |
| [FLAN-T5-small](acceptance/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.json), F32 Flex | Strictly loaded 190 F32 tensors; the pinned top-10 IDs and ten sampled logits matched, as did cached prefill/decode. Four-beam generation emitted `[644, 4598, 229, 9685, 5, 1]` (`Das Haus ist schön.`), identical to the reference. The `15.33 s` observation includes load and is retained only as a non-benchmark diagnostic; its record lacks the run metadata needed to reproduce it. | Full-vector reference-logit cosine and output hash, 32-token reference generation, accelerator execution, context-boundary stress, standardized warm throughput, and memory metrics are pending. |
| [FLAN-T5-XXL](acceptance/flan-t5-xxl-ae7c9136adc7555eeccc78cdd960dfd60fb346ce.json), F32 Metal | Strict loading applied 558 runtime tensors from five official shards and verified one physical embedding equivalence. Burn matched official Transformers 5.2/MPS F32 encoder, decoder-hidden, first-logit, 32-row cached-logit, and 32-row teacher-forced fixtures at minimum cosine `0.999999999998`; all 32 greedy tokens and the EOS-terminated four-beam continuation matched exactly. The one-shot cold diagnostic reported `71.107 s` load and `92.06 s` process wall time. Peak RSS was `11,276,173,312` bytes and macOS peak process footprint was `50,210,770,368` bytes. Canonical SHA-256 output hashes and analytical `65,273,856`-byte cache payload are recorded. | Padded/context-boundary execution, Q8/Q4 quality and 32K stress, and standardized warm latency/throughput remain pending. |
| [Nemotron 3.5 Lightning](acceptance/nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json), F16 Metal | The clean cold gate authenticated the complete artifact, strictly applied all 6,513 tensors from fourteen shards with one shard open at a time, and retained 24 FP32 router-correction buffers while converting 6,489 BF16 tensors to F16. Stateless and cached-prefill hidden traces were bit-identical; one recurrent decode matched its teacher-forced top-1 with full-vocabulary cosine `0.999376514`. Diagnostic timings were `102.101 s` load, `8.188 s` stateless, `2.002 s` cached prefill, `1.536 s` recurrent decode, and `319.02 s` including full artifact hashing. These are Burn internal-consistency observations rather than an official-reference comparison. | Official base oracle parity, executable MTP/cache/speculation parity, 32-token generation, Q8/Q4 quality, 32K stress, output/memory reporting, and standardized warm throughput remain pending. |

Every row remains `advertised_supported: false`. Muse Glimmer has a committed
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
- correctness and quantized-quality gates pass before any support claim; and
- the corresponding acceptance JSON explicitly sets `advertised_supported`
  only after every required gate is complete.
