# v0.3 training source validation

This record covers offline source and tooling checks plus small public training
examples. `manifest.json` retains every attempt, exact commands, tool versions,
output hashes, and each step's before/after input inventory. Final status is in
`latest_checks_passed`; earlier unsuccessful attempts remain visible.

Rust checks use release builds and the workspace lockfile (bound by SHA-256, including uncommitted changes). The full CPU profile
is `std,flex,autodiff,train,tokenizers,chat-templates` with default features disabled.
Both nightly and exact Rust 1.92.0 exercise all targets and doctests. Separate
checks cover all-feature clippy, formatting, public documentation, minimal/std/
std+autodiff builds without a backend, and the standalone coverage-layout tool.
The latter runs tool unit tests; it does not measure transformer coverage.

Python checks cover repository tooling and the small Grokking, Clock/Pizza,
Greater-Than, and Tuned Lens protocol/metric tests. They do not run pretrained
models or repeat scientific training. Public replays cover six architecture and
activation demos, three synthetic training tasks, a random Qwen adapter task,
and Burn's native Learner. Their compact result JSON files are retained beside
the execution logs; weights and optimizer records remain in ignored `target/`.

## Final results

All **30 checks pass** against stable inputs. The manifest retains **48 attempts**,
including the diagnosed failures and clean retries.

| Check | Final result |
| --- | --- |
| Nightly all-target Rust tests | 1,400 passed; 32 ignored |
| Exact Rust 1.92.0 all-target tests | 1,400 passed; 32 ignored |
| Doctests | 3 passed per compiler |
| Coverage-layout tool unit tests | 29 passed per compiler |
| Python tooling | 360 passed; 14 skipped |
| Analytic Python suites | 22 passed: Grokking 8, Clock/Pizza 2, Greater-Than 5, Tuned Lens 7 |
| Formatting, all-feature clippy, minimal feature checks, public docs | Passed |
| Six public architecture/intervention demonstrations | Passed |
| Five small native training replays | Passed |

The two compilers repeat the same Rust cases; these are not 2,800 distinct tests.
Ignored Rust cases require their separately documented artifacts/hardware, and
Python skips are retained in the logs. The final all-target Rust and Python
attempts ran outside the workspace sandbox to permit their local IPC tests.

The heterogeneous T5, custom mixer, MLA, and native Learner tasks reached 100%
accuracy on their small held-out synthetic splits and reloaded identical logits.
The random Qwen adapter reached 100% **training-set** accuracy, preserved frozen
parameters, and resumed its next Adam update exactly; it has no held-out claim.
These measurements validate the examples' integration, not real-world task quality.

See `summary.json` for machine-readable totals and links to the retained training
reports. The full-profile toy runs report 0.08–0.73 seconds including their record
round-trips, excluding Cargo overhead; this is an observation, not a speed gate.

## Input binding

Inventories include untracked source files. Rust scopes include source, test
fixtures, Cargo/build configuration, and recursively resolved literal
`include_str!`/`include_bytes!` inputs. Python scopes retain sources and fixtures
plus an audit of actual repository files opened for reading. Evolving study
reports and images are excluded unless a check actually reads or embeds them.
This is source validation, not a hermetic compiler/dependency closure claim.

`runner-versions/` preserves runner revisions by SHA-256. The second revision
creates the parent of the fresh Qwen output directory; the initial missing-parent
failure is preserved. The Qwen example itself did not change.

## Retained unsuccessful attempts

- An initial negative-control test assumed native Burn cross entropy remained
  inaccurate. Burn's `train` feature enables a different reduction path. The
  corrected test uses an explicit sequential float32 sum and preserves the
  helper's existing float64-oracle tolerance; executable loss code did not change.
- A historical-record test compared a 2026-09-09 lockfile hash to the development
  lockfile. It now authenticates retained historical bytes. The historical
  measurement record is unchanged.
- Sandboxed Rust/Python local IPC tests could not bind Unix sockets. Later retries
  use the same tests outside that restriction; no test assertions are removed.

No coverage threshold, accelerator numerical gate, package promotion, long
training run, or published mechanism reproduction is established by this record.
