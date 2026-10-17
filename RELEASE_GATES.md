# Executable v0.1 release gates

The release candidate targets **Qwen3.5-0.8B and FLAN-T5-small**. Larger
implemented families remain experimental. This checklist supplements
[RELEASING.md](RELEASING.md); a green source check does not promote a checkpoint,
publish an archive, or waive any real-weight gate.

## Required development checks

The source gate runs the [comprehensive authored-item documentation check](scripts/coverage-layout/README.md#comprehensive-authored-item-standards)
with `--standards`. It requires literal `///` declarations and `//!` module
documentation, including private types, members, constants, and tests. It rejects
inline unit-test bodies in production modules; nested integration-harness tests
retain their separate placement policy. The source-bound schema 2 report states
the limits of macro inspection; generated declarations are outside the authored
scope. This check supplements compiler-enforced public documentation and review
of mathematical explanations. The raw helper and wrapper retain their legacy
default mode for callers that explicitly need the earlier policy.

Every compile-capable release command records a fresh local storage preflight
and stops below 20 GiB. Explicit target overrides remain visible in the report;
the runner does not delete files or silently relocate build storage.

The [coding-standards source validation](acceptance/measurements/coding-standards-validation-20260910/README.md)
passes all 35 current checks at `af95066`, including Rust 1.92/stable and
Metal/CUDA/ROCm compilation. The [separate coverage record](acceptance/measurements/coding-standards-coverage-20260910/README.md)
passes the unchanged thresholds with qualified whole-test-file exclusions:
39,578 / 43,778 lines (90.406140%) crate-wide and 18,322 / 19,255 (95.154505%)
shared/core. No new exclusions or lowered thresholds were introduced to reach
these results. This bounded line diagnostic does not claim macro closure,
production-only coverage, or branch coverage. Earlier failures and unrun
checkpoint/hardware checks remain explicit in the records.

## Integrated source validation

At `f4c56c4` (the final integrated source of the polish cycle) the full source
gate passed: `cargo fmt`, warning-denied Clippy with all features, 3,604 tests
with default features and 3,920 with `std,flex,tokenizers,chat-templates,autodiff,train`
(all targets, 0 failures), 14 doctests including the guides' Rust snippets,
feature-minimal builds, the Rust 1.92 check, warning-denied public and
private rustdoc, the documentation audit (0 gaps in 22,780 authored items),
the fuzz harness tests, 441 Python tooling tests, and all 25 Verus harnesses
(2,035 verified, 0 errors).

Coverage was measured on that source with `cargo +nightly llvm-cov --branch`
over the library and integration tests (`std,flex,tokenizers,chat-templates,autodiff`),
counting production files only (sibling `*_tests.rs` files, test oracles, and
test support excluded): **97.31% of lines and 95.35% of branches crate-wide**,
96.85% / 96.05% for shared/core (`src/` outside `src/models/`), and
97.81% / 94.33% for the model families. Before the polish cycle's coverage
work (`2816989`) the same measurement gave 93.55% / 90.17% crate-wide and
95.25% / 93.09% shared/core. This is a direct native measurement, not the
qualified whole-test-file diagnostic of
[scripts/WHOLE_FILE_COVERAGE.md](scripts/WHOLE_FILE_COVERAGE.md), which has not
been re-run on this source. The largest remaining gaps are I/O-failure
branches of the Burnpack and checkpoint writers that need a fault-injecting
file abstraction.

## Checks added since v0.1

These commands cover the work recorded in the
[changelog](CHANGELOG.md#unreleased). Unless stated, they run on Flex and need
no downloads. Set `CARGO_TARGET_DIR` when the default target directory fails the
storage preflight.

```sh
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked                                   # unit, integration, tiny goldens, doctests
cargo test --locked --features autodiff --lib         # gradient smoke tests
cargo test --locked --test verus_harness              # proof-harness structure, no Verus needed
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
python3 -m unittest discover -s scripts/tests -v       # Python 3.11+; see below
```

The Python tooling tests need Python 3.11 or newer and the maintainer's pinned
reference environment: nine of them verify the installed PyTorch 2.10.0 and
Transformers 5.2.0 wheels and their provenance, so they error on any other
host. `.github/workflows/ci.yml` lists those nine by test ID and skips only
them on its clean runner; the source and package gates run all of them.

**Tiny reference goldens** run inside `cargo test`:
`tiny_goldens_llama` (seven Llama-family cases), `tiny_goldens_gemma`, and
`tiny_goldens_gpt_oss`, each against Transformers 5.2.0 / PyTorch 2.10.0 at an
F32 tolerance of `1e-5`. Regenerate fixtures only deliberately, with those exact
package versions:

```sh
python3 -I scripts/generate_tiny_goldens.py --output tests/fixtures/tiny_goldens [--case NAME ...]
```

**Property suites** honor `PROPTEST_CASES`; release qualification runs them
with a larger budget:

```sh
PROPTEST_CASES=20000 cargo test --locked --release \
  --test generation_properties --test rope_properties --test connectivity_properties
```

**Verus.** Every harness must verify with the pinned release, and each
harness's verified count must equal the one recorded in the
[harness table](verus/README.md#harnesses): 25 harnesses, 2,035 verified,
0 errors at `889a7ea`. The gate checks the prover version, runs every harness
with `--no-cheating`, compares the counts, and writes `verus.json` with the
per-harness counts and the SHA-256 of every harness and included source:

```sh
VERUS=/path/to/verus python3 scripts/check_release.py --gate verus
```

CI runs the same gate on every push and pull request that changes `src/`,
`verus/`, the gate, or the workflow. A proof change that alters a count must
update the table in the same commit.

**Fuzzing.** Run bounded campaigns for all twelve targets listed in
[fuzz/README.md](fuzz/README.md): `configuration`, `index_artifacts`,
`masks_positions`, `cache_transactions`, `generation`,
`experiment_interfaces`, `logits_processing`, `chat_templates`,
`rope_parameters`, `safetensors_headers`, `attention_connectivity`, and
`modular_specs`. The campaigns recorded in [fuzz/CAMPAIGNS.md](fuzz/CAMPAIGNS.md)
for the seven targets added or extended after v0.1 ran without a sanitizer,
because the nightly AddressSanitizer runtime deadlocks on macOS 27; their
sanitizer requirement stays open until they run on a host where it works.
Open findings are kept as ignored reproducers
(`cargo test --manifest-path fuzz/Cargo.toml --lib -- --ignored`).

### Experimental real-checkpoint gates

These ignored tests compare a local checkpoint with a reference bundle that
[`scripts/generate_causal_lm_reference.py`](scripts/generate_causal_lm_reference.py)
writes from Transformers on CPU with eager attention:

```sh
python3 -I scripts/generate_causal_lm_reference.py --model-dir MODEL_DIR \
  --output BUNDLE_DIR --repository OWNER/NAME --revision COMMIT \
  [--dtype {float32,float64,bfloat16}] [--num-hidden-layers N] [--prompt TEXT ...] [--greedy-tokens N]
```

Each test verifies the checkpoint/bundle pairing through the bundle's
`config.json` SHA-256, loads strictly, and runs on Flex; adding
`--features metal` repeats the checks on Metal. Run them one at a time:

| Gate | Command (after `cargo test --release`) | Environment | Default tolerances |
| --- | --- | --- | --- |
| Llama family | `--test llama_family_artifacts -- --ignored --nocapture --test-threads 1` | `TRANSFORMERS_LLAMA_FAMILY_DIR`, `TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR`; optional `_TOLERANCE`, `_MIN_GREEDY`; `_DTYPE` (`f32` default, `bf16`, `f16`), which for half precision needs `_BF16_REFERENCE_DIR` and refuses `_TOLERANCE` | F32: `1e-3` absolute logits (F32 drift `7.6e-5` to `3.83e-4` on SmolLM-135M, Qwen3-4B, SmolLM3-3B, and Llama-3.1-8B, Flex and Metal); at least 2 decisive greedy tokens. Qwen2-7B's first-position activations reach about 7,000, so an F32 bundle's own rounding (`1.6e-3`) exceeds the tolerance: qualify it against a `--dtype float64` bundle, where Flex stays within `2.9e-4` and Metal reaches `1.9e-3` at one first-position logit, run with `TRANSFORMERS_LLAMA_FAMILY_TOLERANCE=2.5e-3`. Half precision: per prompt `1.2` times the official BF16 maximum and mean deviation from the same F32 or F64 bundle (see below) |
| Gemma | `--test gemma_artifacts -- --ignored --nocapture --test-threads 1` | `TRANSFORMERS_GEMMA_DIR`, `TRANSFORMERS_GEMMA_REFERENCE_DIR`, `TRANSFORMERS_GEMMA_LONG_REFERENCE_DIR`; optional `_TOLERANCE`, `_LONG_TOLERANCE`, `_MIN_GREEDY`; `_DTYPE` (`f32` default, `bf16`, `f16`), which for half precision needs `_BF16_REFERENCE_DIR` and `_LONG_BF16_REFERENCE_DIR` and refuses both tolerances | F32: `1e-3`; `5e-3` across the sliding window, where Transformers' own eager and SDPA graphs differ by `3.2e-3`. Half precision: the Llama-family rule |
| gpt-oss | `--test gpt_oss_artifacts -- --ignored --nocapture --test-threads 1` | `TRANSFORMERS_GPT_OSS_DIR`, `TRANSFORMERS_GPT_OSS_REFERENCE_DIR`; optional `_DTYPE` (defaults to the bundle's dtype), `_TOLERANCE`, `_MEAN_TOLERANCE`, `_MIN_GREEDY` | F32 run and bundle: `1e-3` max, `1e-4` mean; any half-precision side: `2.7` max, Transformers' own full-model BF16 deviation from the full F32 reference (`2.665`) rounded up, and `0.15` mean, kept from the earlier four-block calibration (the official BF16 mean is at most `0.108`, the crate's Flex/BF16 mean `0.112`) |

The Gemma long-context bundle and the gpt-oss truncated bundles are described
in each test's module documentation, which also records the measured results.
Qualify half-precision gpt-oss runs (including Metal F16, with
`TRANSFORMERS_GPT_OSS_DTYPE=f16`) against an F32 bundle: comparing two
half-precision evaluations sums their roundings. The full 24-block F32 bundle
needs `--dtype float32 --bf16-resident`, which keeps the decoded BF16 weights
resident and widens each block to F32 only while it runs (exact; 47 GB peak,
174 s). Against it Metal/F16 deviates by at most `1.15` (mean at most `0.025`),
Flex/BF16 by `2.19` (`0.112`), and Transformers' own BF16 run by `2.665`
(`0.108`) ([reference-precision follow-ups](acceptance/measurements/reference-precision-followups-20261016.json)).
The last figure, on which the `2.7` bound rests, is the record's run
`gpt-oss-20b-transformers-bf16-vs-f32-bundles`. The ignored test
`transformers_bf16_reference_stays_within_the_half_precision_bounds` reproduces
it from the two bundles alone (`TRANSFORMERS_GPT_OSS_BF16_REFERENCE_DIR`,
`TRANSFORMERS_GPT_OSS_F32_REFERENCE_DIR`; no checkpoint). It evaluates no Burn
model, so its pass is not a gate pass.

Half-precision Llama-family and Gemma runs (`TRANSFORMERS_LLAMA_FAMILY_DTYPE`
or `TRANSFORMERS_GEMMA_DTYPE`: `bf16` on Flex, `f16` on Metal) are judged
against the same F32 bundles (F64 for Qwen2-7B), with bounds read at run time
from the official implementation's own BF16 evaluation instead of a constant.
Each F32 bundle has a BF16 twin, generated by the same script with `--dtype
bfloat16`, the same prompts, and the same greedy length, and named by
`TRANSFORMERS_LLAMA_FAMILY_BF16_REFERENCE_DIR`,
`TRANSFORMERS_GEMMA_BF16_REFERENCE_DIR`, or
`TRANSFORMERS_GEMMA_LONG_BF16_REFERENCE_DIR`.
[`tests/support/half_precision_calibration.rs`](tests/support/half_precision_calibration.rs)
pairs them fail closed (configuration and weight digests, model, versions,
vocabulary, truncation, prompts and token IDs, dtype labels) and applies one
rule to every checkpoint: if the official BF16 logits of a prompt deviate from
its F32 logits by at most `M` and on average by `μ`, a Burn run must stay
within `1.2 M` at every position and `1.2 μ` on average, and its cached
prefill, batched rows, and long-context chunk ends within `1.2 M`. The slack
`1.2` is the largest per-prompt ratio of a Burn half-precision statistic to
the official one measured on gpt-oss-20b, `1.1725` (the Flex/BF16 prompt-2
mean against Transformers' own BF16 run, from the
[follow-up record](acceptance/measurements/reference-precision-followups-20261016.json)),
rounded up to two significant digits, as `2.665` was to `2.7`; with `1` that
run would fail four of its eight statistics. It was fixed before any
Llama-family or Gemma half-precision run. Cached prefills and batched rows
must also agree with the stateless pass within gpt-oss's self-consistency
tolerance (`0.5` for F16 and BF16), and the teacher-forced greedy continuation
is compared at every step whose F32 margin exceeds `2.4 M`. The F32 tolerance
overrides are refused in half precision; the F32 gates are unchanged. The
seven official bundles and their calibration are recorded in the
[half-precision calibration record](acceptance/measurements/half-precision-calibration-20261017.json):

| Official BF16 bundle in `/Volumes/T9/transformers-references` | `reference.json` SHA-256 | Max \|Δ\| per prompt | Mean \|Δ\| per prompt | Decisive greedy steps |
| --- | --- | --- | --- | --- |
| `SmolLM-135M-1d461723-bf16` | `5f7bfb10…` | `1.46`, `1.10`, `1.65` | `0.190`, `0.075`, `0.071` | 0 of 32 |
| `Qwen3-4B-531c80e2-bf16` | `4722fa13…` | `0.700`, `0.701`, `0.880` | `0.078`, `0.043`, `0.054` | 26 of 32 |
| `SmolLM3-3B-1fa6fa28-bf16` | `47cf6ca7…` | `6.28`, `6.67`, `5.33` | `0.156`, `0.079`, `0.177` | 0 of 32 |
| `Qwen2-7B-453ed157-f64-bf16` (against the F64 bundle) | `b570f1c8…` | `8.49`, `6.85`, `10.7` | `0.421`, `0.312`, `0.675` | 0 of 32 |
| `Llama-3.1-8B-d04e592b-bf16` | `2d909999…` | `0.137`, `0.212`, `0.182` | `0.017`, `0.022`, `0.023` | 27 of 32 |
| `gemma-3-4b-it-093f9f38-bf16` | `013c2862…` | `0.384`, `0.592`, `0.748` | `0.038`, `0.052`, `0.063` | 26 of 32 |
| `gemma-3-4b-it-093f9f38-long-bf16` | `7b9ed384…` | `8.07` | `0.066` | 0 of 8 |

Where no greedy step is decisive, the cached greedy test needs
`TRANSFORMERS_LLAMA_FAMILY_MIN_GREEDY=0` or `TRANSFORMERS_GEMMA_MIN_GREEDY=0`,
as the gpt-oss half-precision greedy tests do. On the long Gemma prompt the
maximum bound (`9.7`) exceeds what a missing sliding window moves at the
window's first positions, so a half-precision run requires only that the run
without the window fails the rule; the F32 run keeps the tenfold check. The
ignored tests `official_bf16_reference_calibrates_the_half_precision_bounds`
(Llama family) and `official_bf16_references_calibrate_the_half_precision_bounds`
(Gemma) reproduce the figures from the bundles alone; they evaluate no Burn
model, so their passes are not gate passes. No Burn half-precision run of
these two gates exists yet.

These gates establish experimental validation only; none promotes a
checkpoint. The Flex and Metal F32 runs at `f4c56c4` (below) are experimental
evidence on the same terms.

### Artifact-gated tokenizer and chat-template parity

Ignored tokenization tests compare the pinned checkpoints' tokenizers and
official chat templates with goldens from Transformers 5.2.0. They are
required before advertising tokenizer or template parity for these
checkpoints:

```sh
cargo test --locked --features tokenizers,chat-templates --lib tokenization::tests -- --ignored
```

| Environment variable | Pinned checkpoint |
| --- | --- |
| `TRANSFORMERS_FLAN_T5_SMALL_DIR` | `google/flan-t5-small` at `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` |
| `TRANSFORMERS_QWEN35_DIR` | `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` |
| `TRANSFORMERS_NEMOTRON_H_DIR` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` at `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` |
| `TRANSFORMERS_MUSE_GLIMMER_DIR` | `meta-models/Muse-Glimmer-30B` at `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` |

The goldens and the DiffLlama attention references are regenerated and
checked against the test sources, with the pinned Python packages and the
same local directories, by:

```sh
python3 -I scripts/generate_tokenization_goldens.py --check src/tokenization_tests.rs
python3 -I scripts/generate_diffllama_reference.py --check src/attention/differential_tests.rs
```

### Replays required after the polish cycle

The batched-execution, attention, position, DeltaNet, Mamba-2, and loader
changes touch real-checkpoint paths whose recorded evidence predates them.
These replays ran on source `f4c56c4`, each test in its own process, and all 61
passed. The [real-checkpoint replay record](acceptance/measurements/real-checkpoint-replay-20261016.json) keeps the
source, toolchain, lockfile, reference-bundle identities, tolerance override,
and full log; the numerics are identical to the earlier runs at `ade8696` and
`65912c1`:

- Qwen3.5-0.8B Metal/F16 official-reference parity: prompt cosine
  `0.999998523`, cached decisions at least `0.999997671`, teacher-forced rows
  at least `0.999997517` (gate minimum `0.999996`).
- FLAN-T5-small's official-reference gate on Flex/F32 and Metal/F32, with zero
  `1e-4` tolerance failures.
- The Llama-family gate on Flex and Metal in F32, at every prompt position and
  with 32/32 greedy tokens: SmolLM-135M within `1.34e-4`, Qwen3-4B `1.41e-4`,
  SmolLM3-3B `3.83e-4`, Llama-3.1-8B `7.6e-5`, and Qwen2-7B against an F64
  bundle (Flex `2.9e-4`; Metal `1.9e-3` at one first-position logit, where F32
  Transformers itself deviates by `1.6e-3`, under the `2.5e-3` override).
- gemma-3-4b-it on Flex and Metal in F32, including 1,237-token prompts across
  the 1,024-token window (largest deviation beyond the window `1.18e-3`, 8/8
  long greedy tokens).
- gpt-oss-20b truncated to one and four blocks on Flex and Metal against F32
  bundles: within `1.3e-4`, 32/32 greedy tokens, batched and long cached
  decoding.
- DeepSeek-V2-Lite's metadata gate and Metal/F16 smoke: cached/stateless
  cosine `1.0`, absorbed decode `0.999998964`.

The record is experimental evidence, not a support declaration: no checkpoint
above is advertised as supported.

The larger replays ran afterwards on source `8a4404a` (two later commits,
`4f9502a` and `1b48b81`, add only an ignored Nemotron diagnostic), each test in
its own process under a memory watchdog. The
[large-checkpoint replay record](acceptance/measurements/large-checkpoint-replay-20261016.json)
keeps every source, binary, artifact and bundle identity, the tolerance
overrides, three superseded harness attempts with their diagnoses, and the
full log:

- The Qwen3.5-0.8B public example printed the same 32-token text as the
  [September record](acceptance/measurements/public-examples-20260909.json)
  (stdout SHA-256 `44ffd45f…`).
- FLAN-T5-XXL on Metal/F32: minimum cosine `0.999999999998`, 32/32 greedy
  tokens, exact four-beam continuation; all seven output digests are bitwise
  identical to the earlier record.
- Qwen3.8-27B on Metal/F16: official prompt cosine `0.999999873` with exact
  top-1 and 32/32 greedy tokens; the text-plus-MTP smoke; and fresh
  create-only Q8 and Q4 Burnpacks (497 packed tensors each) that installed
  exactly and ran (top-1 `271`, cached cosine `1.0`).
- Qwen3-30B-A3B F16, Q8 and Q4 Metal smokes from the existing v2 sidecars
  (top-1 `1096`, cached and stateless bitwise equal); both traced decodes
  launched all 1,345 packed matmuls and no whole-weight dequantization.
- Muse-Glimmer-30B on Metal/F16: official parity identical to its earlier
  record (minimum hidden cosine `0.999943377`, 32/32 greedy tokens), the strict
  streaming audit, and Q8/Q4 packed smokes with kernel audits.
- Nemotron 3.5 Lightning: the Metal stateless, cached-prefill and one-token
  decode smoke passed (decode/teacher cosine `0.999370862`). The official
  corresponding-path gate ran for the first time and **failed**: the cached
  path matches all 32 official rows (at least `0.999299`), but teacher-forced
  decisions 18 and 30 reach `0.994340` and `0.997652` against the `0.999`
  floor. No defect was found; Burn's own two clamp-equivalent paths differ by
  up to `0.994053` at these sparse-routing rows, so the floor needs a reviewed
  calibration decision ([analysis](NEMOTRON_H_ACCEPTANCE.md#first-official-comparison)).
  The decision-14 diagnostic shows that the `time_step_min` clamp asymmetry,
  not the F16 precision policy, causes the official recurrent/chunked split.
- gpt-oss-20b, all 24 blocks: the BF16 Transformers reference now exists
  (`gpt-oss-20b-6cee5e81-bf16-reference.bundle`, generated in 358 s at a
  29 GB peak footprint). Flex/BF16 passes every gate (prompt logits within
  `1.62`, mean at most `0.064`; batched rows within `0.5`; the 183-token
  cached decode within `1.32`). Metal/F16 passes the cached, batched and
  greedy gates but **fails** prompt logits: prompt 0 deviates by `2.66`
  against the provisional `2.5` half-precision bound (mean `0.109`). Against
  the four-block F32 bundle Metal/F16 stays within `0.122`, while
  Transformers' own BF16 run is `1.73` away there, so the gap is the BF16
  reference's rounding; the bound needs a reviewed full-model calibration or
  an F32 reference. Both greedy tests compare no tokens: the reference's
  first-step margin (`3.06`) is below twice the half-precision tolerance.

The Qwen3.8 and Qwen3-30B-A3B F32 logit digests differ from their pre-polish
records (top-1 tokens and cosines do not): the polish cycle reordered
reductions. The Qwen3.8 smoke's
peak footprint rose from 63.3 GB to 68.3 GB; timings on this shared host are
not performance measurements (both resolved below). None of these replays
advertises a checkpoint.

### Reference-precision and memory follow-ups

The [follow-up record](acceptance/measurements/reference-precision-followups-20261016.json)
settles the three items the replay left for review. It keeps every source
and bundle identity, and every binary digest except that of the `8d55202`
gpt-oss binary behind the two superseded uncalibrated runs (those gates were
rerun with a retained binary at `c15f24e`). It also keeps the alternating
Qwen3.8 rounds with their load averages, the bisection and footprint traces,
two superseded attempts with their diagnoses, and the full log. Everything ran
on this Mac; the offered SCC node was not used (the record's `compute_note`
explains why).

- Nemotron 3.5 Lightning: the official class evaluated on the CPU in F32
  (BF16-resident blocks widened only while they run, which is exact), BF16 and
  F16 shows that the MPS/F16 oracle is itself only `0.994466` from F32 at
  teacher decision 18, below the `0.999` floor it was used to enforce. The gate
  now judges Burn against the official F32 rows of each path, with each path's
  floor set by the official F16 evaluations' own distance from F32 (prompt
  `0.999194`, teacher `0.994466`, cached `0.998635`), and passes at `9a9c730`
  ([analysis](NEMOTRON_H_ACCEPTANCE.md#first-official-comparison)).
- gpt-oss-20b: a full 24-block F32 reference (generated block by block from the
  BF16-resident model at a 46.6 GB peak) shows that the `2.66` was the BF16
  reference's own rounding. Transformers' own BF16 run is `2.665` from F32
  (mean at most `0.108`; run `gpt-oss-20b-transformers-bf16-vs-f32-bundles`,
  measured from the two bundles on a working tree at `55aeb6a`, before its
  test was committed), Flex/BF16 `2.19` (`0.112`), and Metal/F16 at most
  `1.15` (mean at most `0.025`). The half-precision maximum bound is now
  `2.7`, that official deviation rounded up; the mean bound stays at its
  earlier `0.15`. At `c15f24e` all six gates pass against the F32 bundle.
  Metal/F16 against the BF16 bundle (prompt logits within `2.66`, mean at most
  `0.109`) compares two half-precision roundings; it is recorded as an
  informational comparison, not a gate pass. Load order leaves every logit
  bitwise unchanged.
- Qwen3.8-27B timing: alternating rounds at load averages below 4 show no
  stateless-pass regression (base `1.70`, `1.18`, `2.27` and `9.64` s;
  current `0.99` and `2.13` s; fixed `1.17` and `1.07` s). The first Metal
  forward varies widely on this shared host, so the replay's `10.3` s is noise.
- Qwen3.8-27B memory: the regression was real but smaller than recorded. The
  base source itself now measures `66.5`–`66.6` GB here against `68.2` GB for
  current, so the original `63.3` GB is not reproducible from code on this
  host. Bisection finds `ad992d7`, which streamed checkpoints in physical
  source order: the two 2.37 GiB vocabulary matrices were then allocated after
  every per-layer tensor, and the footprint traces are consistent with
  Metal's first-fit memory pages staying fragmented. `c15f24e` installs
  execution groups of at least 1 GiB first and keeps the source order for
  everything else. The smoke then peaks at `60.6` GB in both rounds, 6 GB below
  the base that alternated with it, with logits bitwise identical to current,
  and the official gate passes again (prompt cosine `0.999999873`, 32/32 greedy
  tokens).
- Muse-Glimmer-30B, the other memory-measured checkpoint with such tensors,
  settles 5.4 GB lower after loading, and its strict streaming audit peaks at
  `66.2` GB against `68.9` GB for the pre-fix binary in alternating rounds.
  The audit passes with the new three-open shard order, and the official gate
  reproduces every recorded cosine (minimum hidden `0.999943377`, 32/32 greedy
  tokens). The same pre-fix binary measured `63.5` GB here a day earlier, so
  only same-session peaks are compared. Nemotron-H, Qwen3-30B-A3B,
  FLAN-T5-XXL, and DeepSeek-V2-Lite have no tensor that large at their load
  dtypes, so their schedules are unchanged.
- Other pinned checkpoints do hoist groups, at the dtypes their gates load:
  gpt-oss-20b's embedding and head (1.16 GB each at F16/BF16, 2.32 GB at F32)
  and, at F32, each loaded block's decoded expert `gate_up_proj` (2.12 GB);
  gemma-3-4b-it's tied embedding (2.69 GB at F32); and, at F32, the
  vocabulary matrices of Qwen3-4B (1.56 GB), Llama-3.1-8B (2.10 GB) and
  Qwen2-7B (2.18 GB). Only the full gpt-oss gates were rerun at `c15f24e`.
  The `f4c56c4` F32 gates of the others and of the truncated gpt-oss bundles
  used the previous order and were not rerun (the record's `not_run`); their
  footprints were not re-measured.

Still open after these replays and follow-ups:

- real-weight Nemotron MTP parity, and official F32 hidden traces for its
  base gate;
- the Burn-side half-precision deviation at Nemotron teacher decision 30
  (`0.997612` from F32, where both official F16 evaluations reach `0.99996`),
  which passes only because each path's floor is set by its worst official
  F16 row;
- Q8/Q4 quantized-reference and quality gates (Qwen3.8, Qwen3-30B-A3B, Muse),
  and an official reference for Qwen3-30B-A3B (no bundle exists);
- Flex/BF16 and Metal/F16 runs of the new half-precision Llama-family and
  Gemma gates (their official BF16 bundles and calibration exist; no Burn
  half-precision run has been measured against them), and real Mistral and
  OLMo 2 checkpoints (tiny goldens only);
- reruns of the F32 Llama-family, Gemma, and truncated gpt-oss gates under
  `c15f24e`'s install order.

Record each replay as a new measurement with its exact source identity; do not
overwrite the historical records below.

## v0.1 release qualification

The [release record](acceptance/measurements/v0_1-release-20260910.json) promotes
the bounded Qwen3.5-0.8B Metal/F16 base-text and FLAN-T5-small Flex/F32 /
Metal/F32 profiles. On September 10 the user explicitly approved the controlled
Qwen measurements as the v0.1 baseline, while retaining the unexplained
historical slowdown. This changes the baseline policy, not the measured results
or the future 5% alert / confirmed 10% failure thresholds. No new benchmark is
claimed by that approval. Larger families, quantized quality and real-weight
MTP remain experimental.

The [`b4d6410` source-gate record](acceptance/measurements/v0_1-source-gates-20260910.json)
passes **1,184 Rust tests / 30 opt-in ignores**, all 17 source/Rust 1.92/backend/
package commands, and separate all-feature warning-denied Clippy. Independently
reauthenticated authored-function line coverage is **32,879 / 36,530 (90.0055%)**
crate-wide and **15,560 / 16,377 (95.0113%)** shared/core, passing both unchanged
thresholds with the same 79 whole-test-file exclusions and zero Burn exclusions.
The exact-source manual interpretation is not a general macro-closure or branch
coverage claim. Later sections retain historical failures at their own snapshots.

Both unchanged public examples were replayed from this source after the tokenizer
fix: Qwen/Metal reproduces the previous 32-token text exactly, and FLAN/Flex
reproduces the previous translation and six-token EOS metadata. Full pinned
artifact, binary and source hashes remain identical before/after each run.
These functional checks do not supply new performance measurements.

The original historical Qwen timing gate failed and its cause remains unknown.
The explicit user decision, not old/current agreement alone, permits the new
baseline and the bounded support promotion. The changelog is dated September 10.
Package verification is repeated on the final committed source, with commit and
archive SHA-256 retained beside the local artifact. No publication or tag has
been created. All older sections below retain their snapshot-specific outcomes,
including pre-promotion false flags and then-open gates; they do not override
the current release record.

## Reproducible source checks

The standard-library-only [runner](scripts/check_release.py) executes explicit
groups, fails on the first failed command, and records command arrays, exit
statuses, elapsed times, logs, and coverage summaries in a new directory under
`target/release-gates/`. It never runs a shell, changes Git refs, or downloads
models. `--dry-run` prints its command plan without writing files.

```sh
python3 scripts/check_release.py --gate source --offline
python3 scripts/check_release.py --gate msrv --msrv-toolchain 1.92.0 --offline
python3 scripts/check_release.py --gate coverage --toolchain nightly --offline
python3 scripts/check_release.py --gate dependencies
python3 scripts/check_release.py --gate accelerator-compile --offline
python3 scripts/check_release.py --gate package --offline
python3 scripts/check_release.py --gate guides --offline
VERUS=/path/to/verus python3 scripts/check_release.py --gate verus
```

The `guides` gate compiles every Rust block of `README.md`, `EXPERIMENTS.md`,
`TRAINING.md`, and `V0_2_INTERFACE.md` as a doctest of a scratch crate that
depends on this one by path, as a downstream user would. A fragment that
assumes a model or an input is compiled inside the context named by a hidden
`doctest NAME` comment above it, defined by a `doctest-context NAME` comment in
the same guide; an unknown or unused context fails the gate. CI runs it in the
documentation job.

Run from this crate or use its full script path from the repository root. Install
tools separately; the runner does not install or update them. The reviewed tool
versions are `cargo-llvm-cov 0.6.21`, `cargo-audit 0.22.2`, `cargo-deny 0.20.2`,
and `cargo-fuzz 0.13.2`. LLVM tools must match the selected Rust toolchain.
Source/MSRV audit groups require the full repository checkout: Cargo excludes
nested standalone packages from source archives, including the development-only
Rust classifier. The archive does carry the Verus harnesses and their guide,
because `tests/verus_harness.rs` reads them. After `cargo package` builds the
extracted archive, the package gate runs its default `cargo test` there (in
Cargo's normal target directory, as the verification build does), so a Rust
test that reads a file the archive omits fails the gate. It then runs the
packaged Python tests (`scripts/tests`) from the archive too, so the package
gate needs the same Python 3.11 reference environment as the source gate's
tooling tests; on a clean Python the nine reference-environment tests error.
The archive still does not run repository-only tooling (fuzzing, CI, the
coverage classifier).

crates.io accepts at most 10 MiB per compressed upload, and the recorded
evidence alone is more than 20 MiB compressed, so the published crate carries
only part of it. `Cargo.toml`'s `include` list publishes the checkpoint support
records (`acceptance/*.json`), the one summary record per measurement campaign
(`acceptance/measurements/*.json`) with that directory's README, and the
compact summaries in `acceptance/measurements/small-model-performance-20260910/`,
which the v0.1 release record names by path and SHA-256 as its timing
baseline. Every other campaign directory (raw logs, coverage exports, replay
reports, copied lockfiles, runner scripts) is repository-only, as are the
recorded outputs of the mechanistic example studies: figures, `result.json`
records, and captured `evidence/` and `results/` runs. Their programs,
protocols, fixtures, and tests are published.
[`scripts/package_contents.py`](scripts/package_contents.py) defines this
split. The gate rejects a listing that contains repository-only evidence, and
fails when the verified `.crate` exceeds an 8 MiB budget, which leaves headroom
below the upload limit; it records the archive's size and SHA-256 in its
manifest. Record tests read repository-only files through
`scripts/tests/repository_evidence.py`: run from an extracted archive
(recognized by the `Cargo.toml.orig` that Cargo writes into every archive,
with no `.git` beside it), they skip with a message naming the omitted file;
in a Git checkout a missing file still fails, even when a stray
`Cargo.toml.orig` such as a `git mergetool` backup is present, and a published
record is never skipped. With this split the
archive is about 6.4 MiB compressed and holds about 950 files, against about
28.5 MiB and 2,060 files before.
The MSRV gate checks `rustc --version` exactly: `--msrv-toolchain stable` is
permitted only while that local alias actually resolves to Rust **1.92.0**.
Coverage export warnings (including mismatched profile data) fail validation
even if LLVM exits zero. The manifest retains the native exit status separately
from evidence-validation failure; a partial export is never a passing gate.
Source checks use Flex, tokenizers, templates, and autodiff; accelerator compile
checks select Metal, CUDA, and ROCm separately and execute no kernels. Also run
all-feature Clippy on the equipped build host before release:

```sh
cargo clippy --manifest-path Cargo.toml --locked --all-targets --all-features -- -D warnings
```

The source group also runs tests embedded in examples, retains a separate
doctest pass, and executes the six download-free experiments listed in
`SOURCE_EXPERIMENTS` in the runner. Their assertions check public workflows as
well as compilation. Paper checkpoint preparation and measured reproductions
remain explicit commands in [examples/README.md](examples/README.md); the source
gate never downloads their artifacts. These additions do not refresh or waive
the historical coverage, hardware, or checkpoint qualifications above.

The [v0.2 interface validation](acceptance/measurements/v0_2-interface-validation-20260910-retry/README.md)
passes its 29 recorded local steps, including 1,368 Rust tests, 3 doctests,
371 Python tests, and ten example replays. It binds the final implementation
and protocol inputs by hash; 32 Rust ignores and 14 Python skips remain explicit.
This development record uses release builds because of the local debug-target
symlink. It does not replace the coverage, hardware, or package release gates.

The Rust 1.92 source run at `1aa5ee9` passed 827 unit tests, all ordinary
integration tests, warning-denied Clippy/docs, feature-minimal builds, and 179
Python tests. Its logs are in `target/release-gates/20260909T160336.140259Z/`.
Later changes still require replay; this source gate does not waive coverage.

The subsequent `9cd6667` source run passed 870 unit tests, ordinary integrations,
warning-denied Clippy/docs, minimal builds, and 183 Python tests at
`target/release-gates/20260909T162545.804663Z/`. All-target, all-feature Clippy
also passed on Rust 1.92 at `9f37b33`, including accelerator feature compilation.
These are intermediate source snapshots, not final release promotion.

At `c2cc2b3`, source and Rust 1.92 gates pass 884 unit tests plus ordinary
integrations, 231 Python tests, and warning-denied Clippy/docs. Cargo also verified
the clean 236-file source archive; no weights or build directories are included.
The exact local logs remain at `target/release-gates/20260910T003025.158291Z/`.
The subsequent `28cafe5` snapshot passes the expanded source/MSRV checks (including
24 standalone-classifier tests and 233 Python tests) and verifies a 237-file
runtime archive. Its logs are at `target/release-gates/20260910T004203.127243Z/`.
Cargo's actual inventory excludes the nested standalone classifier; explicit
include entries did not override that behavior and have therefore been removed.

After the initialized-parameter reload fix (`c3f7001`), source/MSRV checks passed
903 unit tests and ordinary integrations, plus 254 Python and 24 standalone
classifier tests. Warning-denied Clippy/docs and minimal builds also passed;
logs are at `target/release-gates/20260910T011423.312808Z/`. The final coverage
runner review subsequently added two stale-filename regressions. A clean
`f5482ea` [Metal replay](acceptance/measurements/small-model-parameter-reload-replay-20260910.json)
passed Qwen F16/Q4 and FLAN against the pinned local artifacts; the unsuccessful
sandbox/GPU-discovery attempt is retained separately. These functional checks
are not new performance, Q4 quality, or production-only coverage measurements.

The clean `43e1cac` snapshot passes all 17 source, exact Rust 1.92, accelerator
compile-only, and package commands at
`target/release-gates/20260910T054040.035347Z/`. Both Rust test runs pass 997 unit
tests plus ordinary integrations and doctests; the standalone classifier passes
29 tests on both toolchains. Python discovery runs 355 tests with eight native
opt-ins skipped. Metal, CUDA, and ROCm compile separately without executing
kernels. Cargo verifies the 377-file archive; its informational warning about
unset documentation/homepage/repository URLs remains. Rustup provisioned the
named `1.92.0` toolchain before these checks; Cargo dependency resolution was
offline and the dependency lockfile is unchanged. This is an intermediate
snapshot: later test edits need replay and no coverage or performance gate is
waived by successful packaging.

The custom-attention fix at `1ded688` restores padding masks in full attention
and valid-token window distances in sliding attention. Its independent review
and combined release-profile suite pass: **1,117 tests passed, 29 existing tests
ignored**, with all-target/all-feature warning-denied Clippy and formatting
checks clean. Five new custom regressions enforce finite F32 parity at absolute
`1e-4`; five additional loader tests cover accounting and override failures.
Qwen/T5 family kernels are unchanged. This is ordinary synthetic validation,
not a new real-checkpoint replay, package verification, or coverage measurement.

The [subsequent coverage refresh at `b7d2d62`](acceptance/measurements/custom-mask-coverage-20260910.json)
passes the same 1,117 Rust tests and independently reproduces the source-bound
whole-file diagnostic: **87.9303% crate-wide / 94.4818% shared/core**. The custom
mask fix adds nine mapped production lines; they remain in the denominator.
The same 78 whole test files are excluded and both targets still fail. All
368 Python tests, including all fourteen native opt-ins, also pass at this
snapshot. Tiny native coverage-separation fixtures demonstrate compiler
behavior only; production-only qualification and release promotion remain false.

At `c7d4c8d`, all 17 source, exact Rust 1.92, accelerator compile-only, and
package commands pass at `target/release-gates/20260910T062918.975092Z/`.
Both Rust suites pass 1,117 tests with 29 existing opt-in tests ignored; each
standalone-classifier suite passes 29 tests. Python discovery runs 369 tests
with fourteen native opt-ins skipped (the prior 368-test snapshot ran all of
them; the additional measurement-consistency test also passes). Cargo verifies
the 380-file, 9.3 MiB source archive, with only the existing informational warning
about unset documentation/homepage/repository URLs. This refresh does not waive
coverage or performance gates and does not execute accelerator kernels.

The [dtype-guard refresh at `41f3f73`](acceptance/measurements/t5-encoder-dtype-refresh-20260910.json)
fixes a reproduced T5 public-prefill dtype panic before cross-attention or cache
mutation. Nine Qwen/T5, six custom-state, and eight Burnpack regressions bring
the combined and coverage suites to **1,140 passed / 29 existing ignored**;
isolated all-target/all-feature warning-denied Clippy also passes. Independent
reauthentication reproduces **32,209 / 36,517 crate lines (88.2028%)** and
**15,472 / 16,364 shared lines (94.5490%)**. All 29 newly mapped runtime lines
are the T5 validation guards and remain counted. The sole additional exclusion
is the new whole custom-state test file; the policy is unchanged. Both thresholds
and production-only qualification remain open.

The same clean snapshot passes FLAN-T5-small's authenticated official-reference
replay on Flex/F32 and Metal/F32, including all seven vector comparisons with
zero `1e-4` tolerance failures, all 32 cached/teacher-forced greedy decisions,
and exact four-beam outputs. The first Flex native test passed but its macOS
timing recorder exited nonzero after a sandbox-denied `kern.clockrate` query;
that attempt is retained separately from the successful unchanged-binary host
retry. Vector hashes are recorded per execution, not claimed equal across
runs. These are functional checks, not new comparable performance or memory
measurements, and the historical Qwen timing failure is not waived.

At `69e4d87`, all 17 source, exact Rust 1.92, accelerator compile-only, and
package commands pass at `target/release-gates/20260910T070622.030574Z/`.
Both Rust suites pass 1,140 tests with 29 existing ignores. Python discovery
passes 372 tests with fourteen native opt-ins skipped; a separate native-enabled
run passes all 372 without skips. Cargo verifies the 383-file, 9.4 MiB source
archive, retaining only its existing informational missing-URL metadata warning.
Metal, CUDA, and ROCm checks compile without executing kernels. This packaging
refresh does not change the preceding coverage, parity, or performance scope.

Cargo's usual target location is retained. Check at least 20 GiB free before
building a new profile; external model artifacts are never a cleanup target.
The current host needed a [documented cache fallback](acceptance/measurements/local-build-cache-relocation-20260910.json)
after dropping below that preflight: inactive incremental data was archived,
and `target/debug/deps` now links to a content-verified cache on the model volume.
The target path remains unchanged; release and coverage artifacts stay local.
This host-local symlink requires the external drive and is never packaged.
The package gate requires clean tracked crate sources and includes verification.
An unrelated change outside this standalone crate is not permission to stage it.
Every include pattern is root-anchored: Cargo otherwise matches slash-free
patterns such as `*.md` inside the external `models` symlink. The runner validates
the actual file listing before archive creation and rejects artifact directories,
weights, generated bytecode, traversal, duplicates, and unallowlisted roots. The
one weight exception is the tiny random-weight goldens the packaged tests read:
regular `tests/fixtures/tiny_goldens/<case>/*.safetensors` files of at most
1 MiB each (the largest is about 0.5 MiB).

## Coverage qualification is separate from a passing aggregate

At [`796933f`](acceptance/measurements/shared-coverage-gate-20260910.json), eight
further public-contract tests pass, bringing the suite to **1,181 passed /
30 ignored**. With unchanged denominators and exclusions, shared/core reaches
**15,560 / 16,377 (95.0113%)**, passing its line-coverage threshold. Crate-wide
coverage is **32,817 / 36,530 (89.8358%)**, still 60 covered lines short of 90%.
Independent reauthentication retains the exact authored-function ownership
interpretation and adds no exclusions. All 17 source, Rust 1.92, accelerator
compile-only and verified-package checks pass at
`target/release-gates/20260910T081227.056975Z/`: 372 Python tests run with 14 native
opt-ins skipped, the classifier passes 29 tests on each toolchain, and Cargo
verifies a 385-file archive. These do not waive the historical performance
failure or promote either checkpoint.

The latest [tokenizer/cache-contract capture at `820d460`](acceptance/measurements/tokenizer-contract-coverage-20260910.json)
passes **1,173 tests / 30 ignored**, including 24 new units since `311fe3a`
and one new opt-in FLAN tokenizer fixture. Independently reauthenticated physical
coverage is **32,687 / 36,530 (89.4799%) crate-wide** and
**15,542 / 16,377 (94.9014%) shared/core**. The same 79 whole test files are
excluded. Thirteen new authored tokenizer-validation lines remain counted;
all 159,675 retained native regions map to authored function items with zero
Burn declaration exclusions. Both numerical thresholds still fail.

The tokenizer fix rejects special-token metadata that previously bypassed the
actual vocabulary. Three regressions reproduced the failure before the fix.
Local Qwen artifact/template checks and FLAN's exact pinned token-ID/padded-mask
fixture both pass after it; these do not constitute new full-model inference or
performance runs. All-feature warning-denied Clippy and formatting pass. Later
test-only additions require a new capture before any improvement is claimed.

The [reviewed `311fe3a` capture](acceptance/measurements/boundary-contract-coverage-20260910.json)
passes **1,149 tests / 29 existing ignores**, including nine new generation,
snapshot, and projection contracts. Its unchanged 79-file test exclusion policy
measures **32,232 / 36,517 crate lines (88.2657%)** and
**15,495 / 16,364 shared lines (94.6896%)**. Twenty-three additional production
lines are covered; neither denominator changed. The release-profile host replay,
isolated all-target/all-feature warning-denied Clippy, and formatting also pass.
The initial sandbox release run failed at a local socket permission check and
is not counted as a pass. No checkpoint, performance, or package replay is implied.

A separate, independently reauthenticated manual audit now resolves ownership
for this **exact source and captured feature configuration**: all 158,491 native
regions in 89 retained files belong to authored function items (including
signatures), with no remaining mapped test body identified. Associated constants,
types, and bodyless trait signatures are not treated as function bodies. The
actual macro/attribute inventory and all 78 test-dependent external module
declarations were checked; all 48 Burn `Module` declaration spans have zero
native coverage footprint. Therefore no generated-code exclusion is added.
This establishes a bounded authored-production interpretation of these physical
counts, not branch coverage or arbitrary procedural-macro I/O attestation.
The snapshot-specific helper is not a general parser: source/configuration
changes require a new audit. Generic classifier flags remain conservative and
unchanged. **The 90%/95% numerical gates still fail; release promotion is false.**

The subsequent test-only `a9f5678` milestone adds four public DeltaNet contracts,
including an independent scalar recurrence oracle. The integrated release-profile
CPU suite passes **1,153 tests / 29 existing ignores**, with all-target/all-feature
warning-denied Clippy and formatting clean. No runtime or dependency changed.
Coverage above remains explicitly measured at `311fe3a`; no numerical improvement
is inferred for these later tests without another capture.

The runner measures all compiled `src/**/*.rs` at **90%** line coverage and
shared/core source (everything outside `src/models/`) at **95%**, with exact
integer threshold comparisons. Experimental model source is retained in the
crate denominator. Integration-test harnesses, examples, dependency source, and
external acceptance payloads are not part of those denominators.

LLVM's unfiltered summaries can also count inline unit-test bodies and generated
expansions. Those legacy aggregate numbers are **not production-only coverage**
and are not branch coverage. Do not represent the historical baseline below as
satisfying the intended reachable-branch policy; a final-candidate coverage run
and targeted uncovered-path tests remain release work. Hardware-only paths receive separate
opt-in runs, not invented coverage percentages.

The first baseline at integration commit `050f18d`, with
`std,flex,tokenizers,chat-templates,autodiff`, ran **764 passed / 3 ignored** unit
tests. LLVM reported **55,080 / 62,799 lines (87.7084%)** across the crate and
**25,640 / 28,325 (90.5207%)** in shared/core files. Both gates fail. This baseline
used `--lib`; the runner also includes ordinary integration tests. It must be
rerun on the final candidate, and baseline results must not be recycled across
source changes.

The integrated snapshot at `df3bea5` ran **814 passed / 3 ignored** unit tests
plus all ordinary integration tests. It measured **57,350 / 64,636 lines
(88.7276%)** crate-wide and **27,766 / 30,025 (92.4763%)** shared/core. Both
thresholds still fail; added tests have not changed the 90%/95% requirements.
This snapshot precedes the next checkpoint-transform test milestone and is
not a final-candidate coverage claim. Its raw report is retained locally under
`target/release-gates/20260909T155105.885161Z/`.

At `c2937b8`, the same legacy counters reached **60,313 / 67,007 (90.0100%)**
crate-wide and **30,729 / 32,396 (94.8543%)** shared/core. The combined gate still
fails. Raw evidence is in `target/release-gates/20260909T162807.171028Z/`.

The fresh `9ff597f` run passes both **legacy** thresholds: **61,177 / 67,846
(90.1704%)** crate-wide and **31,593 / 33,235 (95.0594%)** shared/core. Its
native command and warning-free validation both exited zero. Raw evidence is
retained at `target/release-gates/20260910T012431.410124Z/`; the 61,898,104-byte
export SHA-256 is `50401dea857a9e67fd89e673c58f57e0cc1a4a3b2ddf934485536df2b6f56e31`.
The before/after input records are identical, each with SHA-256
`7b4af73f3c21899ac1acfa67a36957325c9d825519b73a22b508bee8fd4d88cf`.
No exclusions were applied and production-only qualification remains false.
The separate physical-line helper reports 60,332 / 65,876 crate lines and
31,184 / 32,421 shared lines on this same export, **also including tests**;
these are supplementary diagnostics, not replacement release gates.

The same clean snapshot passes source/MSRV/package checks at
`target/release-gates/20260910T012435.106345Z/`: 903 unit tests, ordinary
integrations, 280 Python tests (two opt-in compiler tests skipped in ordinary
discovery), 24 standalone-classifier tests, and a verified 246-file source
archive. The two native parser tests separately passed all four pinned Rust
fixtures with `TFM_DEPINFO_NATIVE_TESTS=1`, and all-target/all-feature Clippy
passed on Rust 1.92 with warnings denied. The thresholds are unchanged; this
aggregate pass does not close the production-only measurement work.

A separate read-only audit of the immutable `211dc24` source/report found that
LLVM's JSON `summary.lines` aggregates function/instantiation groups; it is not
the count of unique physical source lines. Removing 79 compiler-confirmed
literal `#[cfg(test)]` item spans and six verified test-only module files gave
an **exploratory**, non-promoting result of 31,003 / 36,402 (85.1684%) crate-wide
and 14,915 / 16,287 (91.5761%) shared/core. Generated and unknown classifications
were retained. These numbers are not a fully qualified production-only gate:
the source-classification and line-count tooling still needs independent tests,
source/report identity checks, and differential validation against LLVM.
Neither this diagnostic nor a future legacy-counter pass waives that work.

The supplementary physical-line counter now matches every native LCOV line and
count in a [pinned isolated LLVM 21 fixture](scripts/tests/fixtures/coverage_lines/provenance.json):
30 mapped lines / 26 covered, versus the deliberately different function-summary
33 / 27. Compiler, profile merge, and both native exports produced no warnings.
This qualifies that fixture's counter comparison, not source exclusions or a
historical full-crate export against subsequently edited source.

The separate [Rust syntax classifier](scripts/coverage-layout/README.md) has an
independent pinned development lockfile and adversarial source fixtures. It is
repository-only tooling, never packaged with or linked into the runtime. Its
linked README is available in a checkout, not the runtime archive. Both source and MSRV groups
test it explicitly. Its JSON always sets `qualified_for_coverage_exclusions` to
false: macro-generated references and exact source/export binding remain to be
verified before any candidates can affect a release denominator.

New coverage runs retain `coverage-inputs-before.json` and
`coverage-inputs-after.json`: bounded SHA-256 inventories of source, fixtures,
embedded Python scripts, Cargo inputs and the runner itself. Declared generated
directories under `scripts/` are excluded; the external `models` symlink is
never traversed. The runner records the selected compiler/Cargo/exporter version
outputs, command, compiler-control environment, and raw export/log hashes.
Input edits, a changed tool identity, warnings, or edits during report parsing
fail validation even when the native process exits successfully. Native failures
retain terminal input snapshots when those inputs remain readable; a failed
preflight or spawn leaves a failed manifest with no fabricated process status.

These checks detect ordinary local edits, not adversarial edit-and-restore
behavior. They do not prove compiler dependency closure, authenticate preexisting
test binaries, or qualify test-only exclusions. Both qualification flags remain
false, and neither coverage threshold nor denominator policy changes here.

The independent [dep-info parser](scripts/coverage_dependencies.py) is tested
against eight tiny native compilations spanning exact Rust 1.92 and the pinned
Rust 1.94 nightly coverage compiler. Their filename/environment escaping and
rule-writing code are byte-identical; their source-input collection differs and
is documented separately. The parser preserves escaped-space
dependencies and separates environment comments while rejecting unsupported or
ambiguous native path spellings. It does not resolve paths, identify output
executables, or prove exclusions. Enable its exact-compiler differentials with
`TFM_DEPINFO_NATIVE_TESTS=1`; its two producer pins must not be silently generalized
to another toolchain. No dependency-derived exclusions are adopted yet.

The [development compiler wrapper](scripts/coverage_compiler.py) records rustc's
native JSON artifact messages, including the actual dep-info and compiled-output
paths. It forwards stderr, bounds protocol reads, hashes output bytes, retains
dep-info, and preserves native failures separately from capture errors. This
avoids guessing output identities from Rust's ambiguous Makefile target spelling.
Its tiny native tests cover direct stable/nightly rustc and an actual Cargo
workspace library/unit build. Enable them with `TFM_COMPILER_CAPTURE_NATIVE_TESTS=1`.

The runner now offers [opt-in native compiler evidence](scripts/COVERAGE_EVIDENCE.md)
with `--coverage-compiler-evidence`. This requires fresh captures for every selected
target, unchanged compiler-reported inputs, and exact identities for the production
library and test binaries consumed by LLVM. It removes upstream filename filters;
exactly one pinned cargo-llvm-cov option notice is retained, while every compiler
or LLVM warning still fails. Tiny native fixtures exercise the complete runner,
including production code under a directory named `tests`. The first
[full-crate run](acceptance/measurements/native-coverage-binding-20260910.json)
at `57d0560` now passes: 61,572 / 68,219 crate lines (90.2564%) and
31,988 / 33,608 shared/core lines (95.1797%). All 16 workspace targets have fresh
captures, and the export contains exactly their 15 test binaries. The native
82,075,701-byte export and Cargo's distinct serde rewrite agree on these counters.
Pre/post source inventories match; one exact upstream option notice is retained.
The initial sandbox attempt failed on a local Unix-socket permission check and
adopted no coverage; the unchanged host-permission replay passed 1,013 tests with
29 explicit ignores. Neither `capture_valid` nor this aggregate qualifies
production-only exclusions or arbitrary procedural-macro input closure.

The same snapshot passes all 17 source/MSRV/accelerator-compile/package commands
at `target/release-gates/20260910T025559.369412Z/`, including Rust 1.92, all-target
Clippy, documentation, and 317 Python checks (six optional native skips in
ordinary discovery; a separate native-enabled run passes all 317). Cargo verifies
the 289-file archive. Its expected missing-URL metadata warning is retained; no
repository URL is invented. These are snapshot results, not release promotion.

Shared-target worktree validation needs special care. During parallel test-only
edits, ordinary Cargo reused another worktree's test executable and omitted a new
test while reporting `Fresh`. Distinct `RUSTC_WORKSPACE_WRAPPER` paths partition
workspace artifacts while retaining dependency caches; verify the actual test
inventory and dep-info checkout identity. Do not treat a fast cache hit as proof
of the intended source. `cargo clippy` replaces that environment variable with
its own driver, so setting it alone does **not** isolate Clippy. A separate direct
`cargo check` replay uses a unique wrapper that invokes the installed Clippy driver
with its observed `CLIPPY_ARGS` warning-denial protocol. Tiny native differentials
prove both artifact separation and an intentional Clippy-lint failure; the
integrated all-target release replay passes. This protocol is pinned to the
installed Rust 1.92 tools, not a portable Cargo interface. The opt-in coverage
runner creates its own unique wrapper and rejects inherited compiler overrides.

The first restricted structural extraction (`ab47117`) moves only terminal
`cfg(test)` modules in `error.rs`, `checkpoint/dtype.rs`, and `cache/config.rs`
to explicit sibling paths. The reviewed worktree comparison preserves all
production-prefix bytes, 101 production dependencies, and 906 unit-test identities;
unit dependencies change only from 115 to 118 for the three new files. The full
release CPU suite passes 999 tests with 29 existing ignores. Exact before/after
commands and hashes remain locally in `/private/tmp/tfm-extract-comparison.json`
and `/private/tmp/tfm-extract-validation.json`. That comparison records a Cargo
cache-bookkeeping warning and is structural evidence, not a warning-free coverage
run. No inline spans or new files are excluded by the gate yet.

The third extraction (`e6460f2`) separates five more position/cache modules.
Independent worktree checks preserve all five production prefixes, all 101
production inputs, and all 914 test identities; unit inputs change only from
125 to 130. All 911 unit tests pass with three existing ignores, and isolated
all-target Clippy checks 18 fresh targets without diagnostics. The comparison
record is retained at `/private/tmp/tfm-extract3-proof.O5At5m/comparison.json`
(SHA-256 `9269c8fcebdeed7635ed5dfbfb68a752092cc6b221d6badfad97e9daf57cb37f`).
The earlier `57d0560` coverage numbers remain historical, not a measurement of
these newly arranged sources; no new exclusion is inferred from this refactor.

The following shared and Qwen/FLAN batches (`ccc04b8`, `b11795a`) separate 23
additional terminal test modules, again preserving production prefixes and all
914 test identities. Their independent worktree comparisons keep 101 production
inputs unchanged and add only the corresponding unit-test siblings. All unit
tests and isolated all-target Clippy pass in both worktrees. These structural
checks neither execute checkpoints nor reuse an old coverage export.

The [whole-test-file diagnostic](scripts/WHOLE_FILE_COVERAGE.md) now joins a fresh
classifier invocation with authenticated native compiler inputs. A whole-file
candidate can be excluded only if every parsed context is test-only, it belongs
to the unit compilation, and no production compilation reports it as an input.
The exact successful runner row, log name/hash, and strict diagnostic policy are
also required; another clean log in the directory cannot replace the real log.
Native fixtures verify macro/include/alias and conditional-path vetoes. Inline,
generated, unclassified, and ambiguous ownership remains in the denominator.
This separately scoped file policy does not qualify production-only coverage:
remaining inline tests can still inflate its physical-line diagnostic. The
90%/95% requirements are unchanged.

The first [executed v1 diagnostic at `3d5220a`](acceptance/measurements/whole-test-file-coverage-v1-20260910.json)
removed 39 whole test files and measured 90.0779% crate-wide / 95.2700% shared;
remaining inline/property tests could still inflate those values. The following
checkpoint and model extractions (`82da8cb`, `c67d9fe`) preserved 101 production
inputs and all 914 unit-test identities, adding only 34 test-only sibling files.
The narrowly reviewed macro-span policy (`41e0038`) also qualifies macros whose
complete spans inherit independently proven whole-file test ownership; it does
not classify arbitrary generated code as test-only.

The [fresh v2 execution at `c67d9fe`](acceptance/measurements/whole-test-file-coverage-v2-20260910.json)
passes 1,013 tests with 29 existing ignores. Sixteen fresh compiler captures and
15 exact LLVM objects bind 101 production / 187 unit inputs. Independent
revalidation reproduces the diagnostic exactly: 78 files excluded, **31,412 /
36,479 = 86.1098% crate-wide** and **15,324 / 16,364 = 93.6446% shared/core**.
Both intended numerical targets fail. The native test-inclusive aggregate still
passes (90.2428% / 95.1819%); it must not be substituted for this uncovered gap.
Production-only, branch-coverage and release-promotion flags remain false.
The exact native export and command log remain under
`target/release-gates/20260910T043433.769811Z/`. Later source edits need a new run.

The subsequent parsed structural audit is deliberately diagnostic: it checks
test-dependent attributes across items, associated items, expressions and
statements, including module-only production parents absent from LLVM rows.
Opaque macro-generated tests/helpers still prevent a production-only claim even
when the parsed blocker list is empty. Native counterexamples demonstrate why
normalized expanded-AST equality alone loses physical source ownership. The
integrated development checks pass all 353 Python/native tests without skips;
this does not substitute for runtime coverage.

Three new real-cache generation regressions cover early-EOS mixed batches,
inactive nonfinite logits, unchanged finished-row positions/lengths, model
failure propagation, and mid-batch callback cancellation. The CPU unit suite
passes 914 tests with three existing ignores, and isolated all-target Clippy
passes with warnings denied. No runtime inference code changed in this batch.

The following parallel, test-only batches add 24 automatic-model/loader tests,
22 Qwen/T5 cache-contract tests, and 12 artifact/packed-load integrity tests.
They cover every automatic dispatch variant, live reset/reorder/truncate parity,
foreign-cache rejection, byte-preserving failure of malformed pending writes,
encoder binding, missing index-alias storage, duplicate escaped JSON keys,
invalid manifest provenance, corrupted packed payloads, and source-close errors.
Each worktree's applicable tests and isolated warning-denied Clippy pass. Together
with the three generation regressions, this adds 61 tests without changing any
runtime inference code. The combined snapshot requires a fresh coverage run;
the previously recorded diagnostic must not be reused against these new tests.

The first combined run (`b5ca317`) failed a new sparse-MoE fixture: the initial
prompt had left unused experts lazy, and cloning their independent initializer
cells allowed later decode weights to diverge. That failed run contributes no
coverage. The test-only fix explicitly materializes every parameter before
cloning and adds a deterministic dormant-expert regression; tolerances and
runtime dispatch are unchanged. It passes 100 focused parallel replays and
eleven full worktree suites, plus warning-denied Clippy and formatting.

The subsequent [v3 native replay at `bd68375`](acceptance/measurements/whole-test-file-coverage-v3-20260910.json)
passes **1,075 tests, zero failures, 29 existing ignores**. Independent
requalification reproduces the exact diagnostic: **31,960 / 36,479 = 87.6121%**
crate-wide and **15,361 / 16,364 = 93.8707%** shared/core. These are increases of
548 and 37 covered lines with unchanged denominators and the same 78 excluded
test-only files. Both targets still fail. The parsed structural audit reports
101 inspected production files and no blockers; macro-generated ownership is
not resolved, so production-only qualification remains false. Raw evidence is
retained under `target/release-gates/20260910T051536.131067Z/`. The compact record
also identifies the rejected run and fixture-fix validation separately.

The next [checkpoint-contract replay at `09a3d49`](acceptance/measurements/checkpoint-contract-coverage-20260910.json)
adds 24 tests across five existing test-only files: eight bundle-audit checks,
three failure/retry checks, and thirteen mapping/manifest checks. It passes
**1,099 tests, zero failures, 29 existing ignores**, including 997 unit passes.
The independently reproduced whole-file diagnostic reaches **32,046 / 36,479 =
87.8478%** crate-wide and **15,447 / 16,364 = 94.3962%** shared/core: 86 additional
covered shared lines, no denominator or runtime changes. Both targets and
production-only qualification remain open. The normal Cargo target and external
cache were used; the exact capture remains at
`target/release-gates/20260910T053138.635172Z/`. The separately replayed development
suite also passes all 354 Python/native tests without skips.

The later `0602c03` boundary-contract snapshot adds two artifact JSON map-type
tests and six generation tests. Its isolated release-mode combined run passes
**1,107 tests, zero failures, 29 existing ignores** (1,005 unit passes), and
isolated all-target/all-feature Clippy passes with warnings denied on Rust 1.92.
Formatting also passes. These tests validate public error propagation, malformed
model outputs, cache-session destruction and retry, beam error ordering, and
unchanged artifacts after malformed metadata. Production sources are unchanged.
This final small batch has not been remeasured for coverage or repackaged; the
preceding source/package and coverage snapshots remain explicitly separate.

The standalone runtime lockfile is now tracked explicitly (overriding only the
crate-local parent ignore rule). Its reviewed initial SHA-256 is
`819b26befc5ff5040f5ca42b5088826652bd0e713935e56bf8ab30e0b1cc381d`.
This pins the exact tested development graph without constraining downstream
applications to this library's lockfile. The independent fuzz package also keeps
its own lockfile and [campaign evidence](fuzz/CAMPAIGNS.md).

## Dependency policy

`deny.toml` rejects unknown registries and Git sources, and checks licenses of
the resolved all-feature graph, including build and development dependencies.
The small version-specific exceptions cover unchanged `colored 3.1.1` and
`option-ext 0.2.0` under MPL-2.0 and `webpki-roots 1.0.9` under
CDLA-Permissive-2.0. These are not blanket exceptions for future packages or
versions. Downstream binary distributions must retain the applicable dependency
licenses and notices; the crate's own MIT/Apache license does not replace them.
The [cargo-deny license policy documentation](https://embarkstudios.github.io/cargo-deny/checks/licenses/cfg.html)
describes how these exact-version exceptions are applied.

The 2026-09-09 audit of the 636-entry lockfile found **zero known vulnerabilities**
using RustSec database commit `b50980aad8b8f14f77e25a97b32dd94bf008b0af`.
Two maintenance notices remain explicitly tracked:

- `RUSTSEC-2025-0141`: `bincode 2.0.1`, inherited from pinned Burn Core.
- `RUSTSEC-2024-0436`: `paste 1.0.15`, inherited from CubeCL, Flex/GEMM, and
  tokenizers. This is a build-time macro dependency.

The runner waives only these two unmaintained-package notices, not new warnings
or vulnerabilities, with one narrow exception: on 2026-10-17 RustSec published
`RUSTSEC-2026-0367`, `RUSTSEC-2026-0368`, and `RUSTSEC-2026-0369` (zip
extraction through symlinks and SUID bits, fixed in `zip 9.0.0`) against the
`zip 0.6.6` and `zip 7.2.0` entries described below. Those come only from Burn's
optional Candle and LibTorch backends, are never compiled, and have no
compatible fixed release, so the whole-lockfile `cargo audit` scan waives these
three IDs (`UNREACHABLE_ADVISORIES` in `scripts/check_release.py`). `deny.toml`
deliberately does not, so the built-graph check still fails if anything this
crate builds ever depends on a vulnerable `zip`.

The check runs in two parts. `cargo audit` checks every lockfile entry for
vulnerabilities, including Burn's optional backends that no feature here
activates. `cargo deny check advisories` judges advisories and yanked versions
over the graph this crate actually builds with all of its own features
(`deny.toml` lists the same two notices). On 2026-10-16 the connected audit in
CI reported seven advisories in `quinn-proto 0.11.17` and `rustls 0.23.43`,
reached through `cubecl-cpu`'s LLVM bundler download (the `cpu` feature); the
lockfile now pins `quinn-proto 0.11.19` and `rustls 0.23.46`, the newest
versions compatible with Rust 1.92. Every `zip 0.6` and `zip 7.x` release is
yanked, and those entries come only from Burn's optional Candle and LibTorch
backends (`candle-core 0.10.2`, `tch 0.22.0`), which no feature of this crate
enables, so they are never compiled; cargo-deny's built-graph check keeps
rejecting any yanked crate that is. Revisit them when upgrading Burn or tokenizers; do not fork
the inference stack merely to suppress an informational notice. A fresh
connected audit is required before release. The [2026-09-10 connected audit](acceptance/measurements/connected-dependency-audit-20260910/manifest.json)
passes with zero vulnerabilities/new warnings, completed yanked lookup, and
zero license/source errors or warnings. It binds the exact unchanged lockfile,
manifest and policy hashes and retains the two existing maintenance notices;
it is not a waiver of newly reported advisories. `--offline` explicitly skips the
registry's yanked-version lookup and uses the already-fetched advisory database;
it is useful locally but is not a substitute for that final connected audit.

## Gates that the source runner does not substitute for

- Execute bounded sanitizer fuzz campaigns for every fuzz target: configuration,
  shard indexes, masks/positions, cache transactions, generation, experiment
  interfaces, logits processing, chat templates, RoPE parameters, SafeTensors
  headers, attention connectivity, and modular specs. Preserve minimized bugs
  as ordinary regression tests; finite campaigns are not exhaustive proofs.
- Replay authenticated Qwen and FLAN official full-vector references on their
  declared backend/precision, 32 exact stable greedy decisions, cached versus
  uncached top-1, tokenization/templates, padding, and context boundaries.
  The [integrated Rust 1.92/release replay at `6226170`](acceptance/measurements/small-model-functional-replay-6226170-20260909.json)
  passed all four real-checkpoint test cases and both unchanged public examples
  after the parameter-storage, attention-shape, and beam-score fixes. Exact
  commands, source/executable/log hashes, and numerical results are retained;
  subsequent runtime changes still require replay.
  The [post-strict-JSON-fix replay at `27ada1d`](acceptance/measurements/small-model-post-json-validation-replay-20260910.json)
  also passes Qwen F16, its authenticated 186-matrix Q4/v2 cached/stateless smoke,
  and FLAN's seven official vectors with unchanged hashes. It does not imply
  new public-example, performance, 32K stress, or quantized-quality runs.
- Run the real Qwen quantized 32K populated-history stress test. Allocating
  32K capacity or evaluating a single token at position 32767 is insufficient.
  The [2026-09-09 Q4/v2 run](acceptance/measurements/qwen3.5-0.8b-q4-32k-stress-20260909.json)
  now passes actual 32,768-position execution, finite terminal states, overflow
  atomicity, and reset equivalence. Its separate cache-geometry envelope also
  passes. This closes the execution stress gate for this exact artifact, not
  long-context official parity, quantized corpus quality, or warm performance.
- Exercise both public small-model examples from the final candidate.
  The [2026-09-09 public-example run](acceptance/measurements/public-examples-20260909.json)
  passed unchanged FLAN/Flex and Qwen/Metal entry points on Rust 1.92; final
  candidate replay is still required after subsequent implementation changes.
- Record synchronized warmup/repeated load, prefill, and decode timings, peak
  host/device memory, cache bytes per token, and output hashes. Existing
  diagnostic timings are not standardized regression baselines.
  FLAN now has a [repeated Metal/F32 timing series](acceptance/measurements/flan-t5-small-metal-benchmark-20260909-run.json)
  with three warmups and ten samples at 32/128/512 source tokens, 128 forced
  decode steps, canonical hashes, and separate whole-process high-water
  memory. A [separate instrumented memory series](acceptance/measurements/small-model-metal-memory-20260909.json)
  now records all 93 phases for each small model, with per-phase idle baselines,
  actual sampling gaps, and distinct warmup/measured partitions; both completed
  with zero sampling errors. These are sampled OS-accounted process maxima, not
  exact unsampled peaks or separately attributable host/device allocations.
  Instrumented timings cannot be compared to the original timing baseline.
  The matching [Qwen Metal/F16 baseline](acceptance/measurements/qwen3.5-0.8b-metal-benchmark-20260909-run.json)
  now passes the same repeated protocol and records actual hybrid cache bytes.
  Its optional 2K/8K/32K prefill-only timing series has not yet run.
- Reconcile support records and documents only after all applicable gates pass.
  In particular, successful FLAN numerical parity alone does not establish
  completion of crate-wide coverage, dependency, packaging, or performance gates.

The [runtime correctness refresh at `50c80a2`](acceptance/measurements/small-model-rc-refresh-20260910.json)
closes the post-loader-change replay for both Qwen public-API tests, FLAN's seven
Flex/F32 reference vectors, the populated Q4 32K stress path, and both unchanged
examples. The earlier parameter-reload record remains the corresponding Metal
full-vector replay. The new public-coordinator hardware run validates thirteen
exact continuations but is **excluded from performance comparisons** because
the observed power state contradicted its supplied AC metadata. Repeated timing
and sampled phase-memory refreshes are now recorded separately below;
production-only coverage qualification and final packaging remain open.
The [fresh five-target ASan campaign](fuzz/evidence/refresh-80b51e8.json)
now passes all 50,000 configured inputs without sanitizer diagnostics, timeouts,
or failure artifacts, with unchanged source, seed, and lockfile hashes. It is
bounded fuzz evidence, not proof of complete coverage.

The [source-bound timing and memory refresh at `3d5220a`](acceptance/measurements/small-model-performance-20260910/manifest.json)
completed before the test-only refactors. Every run has clean before/after source
and executable identities, successful correctness checks, and host-observed AC
power at both endpoints. There was no continuous power/thermal monitoring or
exclusive desktop GPU access. Compact summaries retain all measured timing
samples; hashes distinguish them from the original local reports.

- Qwen's public `generate_causal` coordinator reproduces all 32 oracle tokens in
  each of 13 calls (three warmups, ten measured), with median **1.751402479 s**
  and p95 **1.76914625 s**. This boundary includes cache allocation/destruction,
  selection, log-probability readback and stopping. It excludes tokenization,
  request-tensor construction and reference validation; no public TTFT or
  model-only timing was measured.
- Qwen's ordinary model-loop timing and an identical-binary rerun confirm an
  approximately **25% decode-throughput regression** from the September 9
  baseline. Rebuilding the exact historical source (`1e4f836`) with the current
  40 matching backend dependency artifacts also produces about 27 tokens/s,
  rather than the historical 36. The immediate [controlled comparison](acceptance/measurements/small-model-performance-20260910/qwen-controlled-historical-comparison.json)
  places all warm median source-to-source differences within **1.247%**. Three
  p95 latency alerts exceed 5%, none exceeds 10%; with ten repetitions p95 is
  the maximum sample. **The historical failure is not waived or explained.**
  No current-source cause or decision to replace the historical gate is assumed.
  A [read-only timing attribution audit](acceptance/measurements/qwen-historical-warm-inference-audit-20260910.json)
  confirms matched native-process boundaries: warm model-only decode accounts
  for 35.709 seconds of the 53.71-second original/rebuilt wall-time difference.
  Whole-process retired instructions also increase, but cannot distinguish CPU
  submission work from GPU waiting or execution. Neither a hardware cause nor a
  software cause is established. A future CPU-stack/GPU-telemetry replay would
  be diagnostic-only; no such profiling run has been performed.
- FLAN Metal repeats the 32/128/512-source-token, 128-decode-step protocol with
  exact prediction hashes, roughly 73–75 model-only decode tokens/s, and no
  repeated metric regression over 10% versus its historical baseline. Some
  TTFT/natural-loop latency alerts exceed 5%. The first **Flex/F32 CPU baseline**
  measures 233.713 / 159.518 / 68.387 model-only tokens/s at those source lengths;
  its six-token EOS continuation takes median 0.352338 s. This is workload-specific
  evidence, not a general backend ranking or a historical CPU regression result.
- Separate Qwen/FLAN memory replays each pass 93 phases with zero sampling or
  validation errors. Observed kernel lifetime physical-footprint high-water marks
  are **2,977,486,656 / 771,181,504 bytes**. The 1 ms requested sampling interval
  achieved roughly 1.25 ms; maximum whole-window gaps were 5.40 / 1.88 ms.
  Signed above-idle phase deltas are retained. Sampled maxima are not exact
  unsampled peaks, no host/device split is measured, and instrumented timings
  cannot enter the uninstrumented performance comparison.

No expensive acceptance runs for larger families belong to this v0.1 phase.
