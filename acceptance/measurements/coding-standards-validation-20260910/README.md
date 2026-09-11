# Coding standards validation

The final code snapshot is `af95066a4b8c388a8adee8a4c3ddaf29ce577f21`.
All **35 latest checks pass**, with unchanged scoped inputs during each run.
[`manifest.json`](manifest.json) retains 81 executed rows across all attempts; use the latest attempt for each check, rather than treating earlier
snapshots as current. The runner records actual commands, toolchains, storage
preflights, source hashes, logs, and the inputs read by Python checks.

## Executed checks

- **1,452 Rust tests and three doctests pass** on both pinned nightly and Rust
  **1.92.0**. Each test run explicitly ignores 32 opt-in tests. The installed
  stable alias is also 1.92.0; its separate build passes.
- Rustfmt, all-target/all-feature warning-denied Clippy, public rustdoc,
  minimal/std/autodiff feature checks, and **Metal/CUDA/ROCm compilation** pass.
- The authored-documentation audit reports **2,576 functions and 179 modules**
  with no missing line docs, misplaced inline tests, or unresolved authored
  function templates. Generated macro expansion is outside that audit.
- The standalone syntax tool passes **38 tests on each compiler**. Python runs
  **410 checks across five suites: 396 pass and 14 are explicitly skipped**.
- Six no-download demos and five small native Burn training workflows pass,
  including the Learner integration and model/optimizer record checks.
- Fresh connected advisory, license, and source-policy checks pass; see
  [`dependencies/`](dependencies/). Existing maintenance-advisory exceptions
  are unchanged.

The [separate coverage record](../coding-standards-coverage-20260910/README.md)
passes both unchanged thresholds, including the qualified whole-test-file
exclusion diagnostic: **90.406140% crate-wide and 95.154505% shared/core**.
It preserves earlier failed diagnostic attempts and explains the remaining
macro/production-only/branch-coverage limits. The [fuzz record](../coding-standards-fuzz-20260910/README.md)
retains six ordinary oracle tests, fuzz Clippy, and 10,000 successful ASan inputs
at its exact source snapshot; it does not refresh the historical five campaigns.

## Attempts, storage, and scope

Earlier attempts remain visible: Rust fixture type errors and a stopped compiler
after its reported type error are under [`focused-checks/`](focused-checks/).
The first Python run lacked local IPC permission; the enabled rerun passes.
Four Clippy findings in new tests were corrected without weakening assertions.
ROCm attempt 1 stopped **before launching Cargo** when the 20 GiB preflight
failed. Removing eight clean, completed task worktrees recovered space while
retaining their commits and branches; ROCm attempt 2 passes. All current builds
pass the required preflight. The ordinary crate uses its normal local release
target; the independent fuzz package documents its existing external cache.

During native coverage capture, this runner wrote to ignored
`target/coding-standards-validation-records`, then its records were copied here.
The manifests retain the actual execution paths. Subsequent evidence/Markdown
commits do not change the validated Rust sources. Historical records are
preserved. No new real-checkpoint hardware acceptance, scientific paper-training
run, performance benchmark, package promotion, or release-support claim is made.
