# Whole-test-file coverage diagnostic

`coverage_exclusions.py` is a development-only layer over the native compiler
binder, source classifier, and LLVM physical-line counter. It does not change
`check_release.py`, either 90%/95% threshold, or any model support claim.

A file can be removed only when all of these facts agree:

- Every parsed reference context classifies the **entire** file as test-only.
- Complete line ownership is test-only or blank, with no ambiguous diagnostic.
- The file belongs to the captured library unit-test compilation.
- No captured non-test production compilation reports it as an input.
- The classifier and compiler observed the exact snapshotted source bytes.

Repeated equivalent contexts are retained as evidence. The only other permitted
diagnostic is `opaque_macro_or_verbatim` whose complete byte/line/column span
exactly matches a nonempty, parsed, test-only **macro** region in **every** file
context. Verbatim/unresolved syntax, absent spans, unknown cfg/path diagnostics,
and any production context remain vetoes. This exception uses inherited
whole-file ownership, never a macro-name whitelist or expansion assumptions.
The versioned policy records this distinction from the earlier all-macro veto.
Production compiler inputs override
syntax candidates, including macro-generated modules, `include!`, and tracked
`include_str!` inputs. A directory named `tests` never implies exclusion.
Unclassified files, inline tests, and uncertain cfg/path contexts stay in the
denominator. A macro expansion's **other mapped filename** also stays unless that
file independently satisfies the same full proof; qualifying a macro's source
does not propagate its ownership into included/generated files. No individual
generated spans are removed. Experimental model source also stays in scope.

## Invocation and evidence

First build and test the classifier from its pinned standalone
`coverage-layout/Cargo.toml` and lockfile. Treat the selected executable as a
trusted local tool: its hash records the bytes executed, not an attestation of
its build process. Use a distinct workspace wrapper when sharing a Cargo target
between worktrees, or a separate tiny classifier target. No Burn build is needed.

After a fresh `--coverage-compiler-evidence` run, and before changing any captured
source, use the Python API from the crate's `scripts` import path:

```python
from pathlib import Path
from coverage_binding import read_json, validate_capture, write_json
from coverage_exclusions import capture_classifier, qualify_capture, target_entries

crate = Path("/absolute/path/to/crates/transformers")
run = crate / "target/release-gates/EXACT_FRESH_RUN"
setup = read_json(run / "compiler-capture-setup.json")
snapshot = read_json(run / "coverage-inputs-before.json")["sources"]
binding = validate_capture(setup, snapshot, crate)
capture = run / "whole-test-file-classifier"  # must not already exist
capture_classifier(crate, snapshot, Path("/absolute/path/to/trusted-classifier"),
                   capture, target_entries(binding["targets"], crate))
diagnostic = qualify_capture(setup, snapshot, capture / "invocation.json", crate,
                             coverage_log=run / "00-coverage.log",
                             run_results=run / "results.json")
write_json(run / "whole-test-file-diagnostic.json", diagnostic)
```

Choose the actual coverage log from that run's `results.json`; `00-coverage.log`
applies when coverage was its first command. Do not reuse an old capture against
edited source, even if only fixtures or scripts changed. Qualification rechecks
the successful coverage row's exact indexed log name and retained hash, original
command/input/setup/binding records, native status, strict log diagnostic policy,
classifier command and outputs,
tool/source hashes, native report, and source inventory. It requires the pinned
nightly/LLVM and cargo-llvm-cov identities already enforced by the native binder.
Only the one exact unfiltered-option notice is permitted; other warnings fail.

The result retains the unchanged LLVM-summary aggregate, unique physical lines
before exclusions, a separate whole-file diagnostic, and a per-covered-file
ledger identifying exactly which mapped/covered lines were removed and why.
The thresholds use integer comparisons, never rounded displayed percentages.

`whole_test_file_exclusions_qualified=true` refers only to this bounded file
policy. `production_only_exclusions_qualified`, `production_only_measurement`,
`compiler_dependency_closure_verified`, and `release_gate_promotion` remain
**false**. Retained highly covered inline tests can inflate the ratio: this is
not a lower bound on production coverage. Arbitrary untracked procedural-macro
I/O and adversarial edit-and-restore behavior are not proven or claimed.

## Tests

Ordinary discovery exercises malformed ownership, aliasing, missing compiler
contexts, exact thresholds, stale inputs, and failed classifier invocations.
The optional native differential builds only the classifier and a tiny offline
fixture in a temporary target; it uses no Burn, weights, network, or GPU:

```sh
TFM_WHOLE_FILE_COVERAGE_NATIVE_TESTS=1 python3 -m unittest discover -s scripts/tests -p test_coverage_exclusions.py -v
```

It verifies real compiler-discovered macro/include/alias inputs veto candidate
exclusions; whole test files with `macro_rules!`, invocation, `include!`, and the
crate lockfile's exact Proptest 1.11.0 version qualify; and unknown cfg and
conditional paths remain even when their files contain macros. A macro-included
file with no independently parsed module context remains, while a separately
proven test-only module can qualify. Property tests execute unchanged as tests:
they are neither erased nor replaced to improve the denominator.
The offline fixture resolves and captures its own lockfile before the native
run, with Proptest pinned exactly and no Burn dependency. It also verifies that
an added coverage warning invalidates previously acceptable evidence.
It also exercises the actual release runner's retained result records: another
clean log in the same directory cannot substitute for its real coverage log.
