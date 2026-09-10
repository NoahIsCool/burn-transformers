# Native compiler evidence (opt-in)

Run the existing coverage gate with fresh compiler and export binding:

```sh
python3 scripts/check_release.py --gate coverage --toolchain nightly --offline --coverage-compiler-evidence
```

This mode requires the exact `rustc 1.94.0-nightly` commit
`4f14395c37db4c1be874e6b0ace6721674223c22`, its LLVM 21.1.8 tools, and
`cargo-llvm-cov 0.6.21`. The mutable `nightly` name is checked, not trusted as a
pin. It uses the crate's normal target directory, preserves dependency caches,
and lets cargo-llvm-cov perform its normal measured-package cleanup. A unique
workspace-wrapper pathname also separates Cargo's workspace artifact cache key.
Every required target must nevertheless have a fresh native capture: a cache hit
or a stale object cannot substitute for that evidence.

The run directory retains the exact compiler argv/cwd/version and native
artifact JSON messages as byte identities, original and copied dep-info hashes,
pre/post source inventories, selected target metadata, wrapper/tool hashes, and
the LLVM export's literal argv/profile/object hashes. The production library is
bound to integration tests through their actual `--extern` argument. All selected
test binaries must match the exact LLVM object list. Opt-in mode passes
`--disable-default-ignore-filename-regex`; the native exporter must have **no**
ignore option. There is no Python regex evaluation or filter emulation.
Unexpected targets,
uncaptured sources, conflicting configuration, native warnings, changed bytes,
unsupported syntax, and incomplete records fail closed.

`llvm-export/native-coverage.json` is the exact native stdout. Cargo's separately
retained `coverage.json` is a serde rewrite with injected metadata; it is not
claimed to have the same bytes. Both existing aggregate thresholds remain in
force. No generated-line or inline-test exclusions are introduced.
Legacy mode retains its existing Cargo directory filters. Opt-in mode removes
those filters, so production paths such as `src/tests/helper.rs` cannot disappear
just because a directory is named `tests`. The runner still selects only `src`
for its counters; additional dependency/harness output is retained within the
same byte limit, and excessive output fails explicitly. This remains an
all-compiled-src aggregate, **not** a production-only measurement.

Pinned cargo-llvm-cov emits the tool-option notice
`warning: --disable-default-ignore-filename-regex option is unstable`.
Only after no-filter binding succeeds, opt-in mode permits exactly one
byte-exact occurrence and records its text/count. Execution is therefore not
described as warning-free. A duplicate, spelling/whitespace/color variant,
missing expected notice, or any compiler/LLVM warning still fails. Legacy mode
continues rejecting every diagnostic. Removing upstream directory filters does
not weaken the numerical/mismatched-profile diagnostic policy.

The successful evidence fields have deliberately narrow meanings:

- `artifact_binding_verified`: the native export objects match fresh captured
  workspace artifacts, with their selected production library link checked.
- `compiler_reported_inputs_verified`: every compiler-reported source input for
  these workspace targets/configurations belongs to the unchanged snapshot.
- `no_export_filter_verified`: the exact native export has no ignore option.
- `compiler_dependency_closure_verified` and
  `production_only_exclusions_qualified` remain **false**. Dep-info does not
  prove arbitrary untracked procedural-macro I/O. A future whole-test-file
  exclusion policy must separately cross-check classifier candidates against
  every captured production context; inline macro spans remain unsupported.

This is local reproducibility evidence, not attestation against a malicious
compiler or a concurrent edit-and-restore adversary. Native tools are trusted;
bounded reads prevent unbounded buffering but cannot bound a malicious tool's
disk writes before its timeout. Inherited compiler/exporter overrides, workspace
build scripts, multiple workspace packages, remapped paths and compiler inputs
outside the captured crate inventory are intentionally unsupported.

Pure regression tests require no compiler. The optional native differential is
a tiny dependency-free offline Cargo fixture (no Burn build, GPU or model data):

```sh
TFM_COVERAGE_BINDING_NATIVE_TESTS=1 python3 -m unittest discover -s scripts/tests -p test_coverage_binding.py -v
```

It verifies same-wrapper cache reuse, fresh-wrapper recompilation, paths with
spaces, both production and unit-test source inventories, transparent Cargo
jobserver inheritance, exact LLVM override behavior, and the actual opt-in
runner. Evidence qualification remains separate from checkpoint support claims.
