# Bounded sanitizer campaign evidence

## 2026-09-09: initial v0.1 harnesses

Executed on Apple Silicon/macOS with `cargo-fuzz 0.13.2`,
`rustc 1.94.0-nightly (4f14395c3 2025-12-22)`, and the default AddressSanitizer
configuration. Runtime source base was `eb5d274` plus the seq2seq validation fix
`b72f86a` (worktree cherry-pick `1151605`). The separately committed fuzz lockfile
has SHA-256 `252d0ed980a8452f7c7da1aa54c3da00b256fe9804624270a7c1fb19ba12fea0`.

Each target ran against an initially empty writable corpus plus its committed
seed directory. Every campaign used `-runs=10000 -max_len=4096 -timeout=10
-rss_limit_mb=2048`. All five returned exit status zero; no crash, timeout, or
sanitizer failure artifact was produced.

| Target | Executed inputs | Seed | Result |
| --- | ---: | ---: | --- |
| configuration | 10,000 | 483013554 | Passed; final coverage counter 5151, peak RSS 160 MiB |
| index_artifacts | 10,000 | 20260909 | Passed |
| masks_positions | 10,000 | 20260909 | Passed |
| cache_transactions | 10,000 | 20260909 | Passed; final statistics report 10,000 units, peak RSS 127 MiB |
| generation | 10,000 | 20260909 | Passed; final statistics report 10,000 units, peak RSS 104 MiB |

The five deterministic harness tests and library/test Clippy with warnings
denied also passed. Configuration harness review found a multiplication in
`Seq2SeqSpec::validate` that could panic in debug or wrap to an apparently valid
hidden width in release; the checked-multiply fix and debug/release regression
tests were integrated before these campaigns.

The environment's external symbolizer failed to start, so discovery diagnostics
contain raw Rust symbol names/addresses. Instrumentation and fuzz execution
continued; this is a diagnostic limitation, not disabled sanitization. Install a
matching LLVM symbolizer before investigating a future crash. These short runs
are reproducible smoke evidence, not exhaustive branch coverage or a substitute
for the crate's 90%/95% line gates, GPU numerical tests, or real checkpoint tests.
