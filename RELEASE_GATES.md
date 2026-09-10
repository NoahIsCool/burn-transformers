# Executable v0.1 release gates

The release candidate targets **Qwen3.5-0.8B and FLAN-T5-small**. Larger
implemented families remain experimental. This checklist supplements
[RELEASING.md](RELEASING.md); a green source check does not promote a checkpoint,
publish an archive, or waive any real-weight gate.

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
```

Run from this crate or use its full script path from the repository root. Install
tools separately; the runner does not install or update them. The reviewed tool
versions are `cargo-llvm-cov 0.6.21`, `cargo-audit 0.22.2`, `cargo-deny 0.20.2`,
and `cargo-fuzz 0.13.2`. LLVM tools must match the selected Rust toolchain.
Source/MSRV audit groups require the full repository checkout: Cargo excludes
nested standalone packages from source archives, including the development-only
Rust classifier. Ordinary library tests and package checks remain usable from
the archive; archive verification does not claim to run repository-only tooling.
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
weights, generated bytecode, traversal, duplicates, and unallowlisted roots.

## Coverage qualification is separate from a passing aggregate

The runner measures all compiled `src/**/*.rs` at **90%** line coverage and
shared/core source (everything outside `src/models/`) at **95%**, with exact
integer threshold comparisons. Experimental model source is retained in the
crate denominator. Integration-test harnesses, examples, dependency source, and
external acceptance payloads are not part of those denominators.

LLVM currently also counts inline unit-test bodies and generated expansions in
these file summaries. The numbers are **not production-only coverage** and are
not branch coverage. Do not represent this baseline as satisfying the intended
reachable-branch policy; a refined production-only measurement and targeted
uncovered-path tests remain release work. Hardware-only paths receive separate
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
or vulnerabilities. Revisit them when upgrading Burn or tokenizers; do not fork
the inference stack merely to suppress an informational notice. A fresh
connected audit is required before release. The [2026-09-10 connected audit](acceptance/measurements/connected-dependency-audit-20260910/manifest.json)
passes with zero vulnerabilities/new warnings, completed yanked lookup, and
zero license/source errors or warnings. It binds the exact unchanged lockfile,
manifest and policy hashes and retains the two existing maintenance notices;
it is not a waiver of newly reported advisories. `--offline` explicitly skips the
registry's yanked-version lookup and uses the already-fetched advisory database;
it is useful locally but is not a substitute for that final connected audit.

## Gates that the source runner does not substitute for

- Execute bounded sanitizer fuzz campaigns for configuration, shard indexes,
  masks/positions, cache transactions, and generation. Preserve minimized bugs
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
