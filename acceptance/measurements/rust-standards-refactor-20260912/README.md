# Rust standards refactor validation

The local review branch `codex/tfm-rust-standards` refactors checked cache and
transfer operations, validates modular decode history allocations before mutation,
modernizes module roots, and documents authored Rust contracts across the library,
examples, tests, fuzz targets, and tooling. Burn remains pinned at 0.21 and the MSRV
remains Rust 1.92. No branch was pushed or merged.

All final checks below used source
`8444a38ecb9c2eb3aa50ae5af21828c1146b821d`, based on
`103c6ffeb5c5def6f2ce2c273717d65822ec6d07`. The subsequent evidence commit adds only
this directory. [validation.json](validation.json) records the exact commands,
target directories, log hashes, source commits, and preserved agent attribution.

## Behavioral changes and review

- Cache staging retains checked tensor handles and validates both allocations
  before publishing either side, preserving state when a recoverable check fails.
- Transfer alias resolution carries checked entries through grouping and rollback;
  new tests cover multiple groups, tied parameters, and failure boundaries.
- Modular causal and sequence-to-sequence decoding reject overflowing history
  geometry and portable index limits before allocation. Regression tests check
  oversized requests and allocation invariants across reorder, reset, and truncate.
- Model configuration scanning handles UTF-8 boundaries and checked dimensions
  explicitly. Checkpoint paths use contextual recoverable errors.
- Eight existing `mod.rs` roots now use named module files. Reworked unit suites
  use sibling `*_tests.rs` files; integration harnesses retain their appropriate
  structure.

An independent review of the substantive runtime changes found no actionable
regression. Its request to strengthen reorder assertions before resetting decode
state was incorporated and tested.

## Checks

| Check | Result |
| --- | --- |
| Rust tests with `autodiff,tokenizers,chat-templates` | 1,462 tests and 3 doctests passed; 32 opt-in tests ignored |
| Same suite on MSRV Rust 1.92.0 | Same passing counts |
| All targets with `train,tokenizers,chat-templates` | 1,480 tests passed; 32 opt-in tests ignored |
| Rust 1.98.0 stable build with training features | Passed |
| All-target, all-feature Clippy with warnings denied | Passed |
| Minimal, `std`, and `std,autodiff` checks | Passed |
| Private library and all 18 example rustdoc targets, warnings denied | Passed |
| Standalone Rust syntax tool | 52 tests passed on nightly and MSRV; Clippy passed |
| Python tooling suite | 392 reported, including 14 optional skips; 378 executed successfully |
| Both Rust formatting checks and whitespace check | Passed |
| Verus ring arithmetic model | 3 verified, 0 errors with `--no-cheating` |

The general test compiler was Rust 1.94.0-nightly
(`4f14395c3`, 2025-12-22). The installed `stable` alias resolves to 1.92.0, so the
newer stable build explicitly selected Rust 1.98.0 (`88d9e12ae`, 2026-08-18).
The overlapping test suites should not be summed as distinct test cases.

The fresh strict documentation audit passed for **11,516 authored sites across
315 physical files**, with zero missing documentation, misplaced inline unit test
bodies, or opaque authored declarations. [documentation.json](documentation.json)
records per-scope counts and source hashes. The audit covers authored syntax and
recognized test macros; it does not expand arbitrary macros or establish the
semantic correctness of every documentation sentence.

## Measured coverage and proofs

Coverage was measured separately with `cargo-llvm-cov 0.6.21 --branch`, using
LLVM 21.1.8 on the nightly compiler. The instrumented library/integration suite
passed 1,462 tests with 32 opt-in ignores. The expected unstable branch-option
notice is retained in the log.

| Native reported `src/` measure | Covered / total | Coverage |
| --- | ---: | ---: |
| Branches | 5,893 / 7,388 | 79.76% |
| Lines | 87,418 / 93,548 | 93.45% |
| Regions | 133,240 / 140,959 | 94.52% |

These totals retain authored test and generated mappings under the tool's default
filename filters; they are not a production-only or release-qualified coverage
measurement. [coverage.json](coverage.json) contains the native per-file summaries.
Shared/core source excluding `src/models.rs` and `src/models/` measured 85.38%
branches, 96.20% lines, and 96.41% regions. The ring and allocation helpers each
reached 100% branch coverage; remaining gaps include device/dtype guards,
defensive staging/coordinator states, model configuration combinations, and
generated or assertion mappings. Full meaningful branch coverage is not established.

The standalone syntax tool was measured separately at source `e9e7b58`, with
unchanged Rust tool source in the final snapshot: 320/388 branches and 1,566/1,697
lines; CLI argument/output handling reached 20/20 branches. Remaining tool gaps
include defensive source races, span handling, and recursion-depth limits.

[The Verus scope and correspondence notes](../../../verus/cache_ring.md) explain
three proofs of exact-integer append, suffix, and ring-partition bounds. The final
run used Verus `0.2026.09.06.8dea4a2`, Rust 1.98.0, Z3 4.16.0, and `--no-cheating`.
This standalone model has a manually reviewed correspondence to executable code;
it does not prove compiled vector operations, Burn tensors, or floating-point
arithmetic.

## Reproduction and limits

Run commands from the repository root, substituting the target variables defined
in [validation.json](validation.json). Local storage had 11–15 GiB free, below the
20 GiB preflight, so builds used dedicated external targets on
`/Volumes/bag of holding`. Fresh task targets prevented reused artifacts from
hiding source changes across worktrees. The consolidated source runner's dry-run
plan was inspected; its fixed source-filesystem preflight prevented executing it
with this external-target fallback. Applicable individual checks are recorded
above. Large raw exports are preserved externally with hashes in the manifest.

All-feature checks compile Metal/CUDA/ROCm paths but do not establish hardware
execution. This pass did not replay scientific training runs, run experiment main
functions, measure performance, refresh dependency advisories, or qualify a
package release. Historical acceptance evidence and opt-in artifact requirements
remain unchanged.
