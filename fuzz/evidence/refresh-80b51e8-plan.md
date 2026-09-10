# Bounded ASan refresh preparation and execution notes

Status: **completed after explicit host release**; see the
[authenticated compact record](refresh-80b51e8.json). This file retains the
original preparation sketch below. Actual execution used
[`run_refresh.py`](run_refresh.py), with native sanitizer executables after one
cargo-fuzz build, to capture native exit codes directly. Writable corpora and
artifact directories were created below the new result directory
`/private/tmp/transformers-fuzz-refresh.EXC7pm`, not the preliminary relative
corpus paths below. The JSON's exact command arrays are authoritative.

## Source and tool preflight

- Worktree branch: `codex/tfm-fuzz-refresh`, based on
  `80b51e838033cc479ab7d99a97106e5c2309d5a8`.
- Runtime `src` Git tree: `b4d4a96ca38fdb709bce07b6fd384d497e523f98`.
- Fuzz Git tree before adding this plan:
  `816d3d99b0a432f4aee1330de1f0de8983933e2f`.
- Fuzz lockfile SHA-256:
  `252d0ed980a8452f7c7da1aa54c3da00b256fe9804624270a7c1fb19ba12fea0`.
- Installed `cargo-fuzz`: `0.13.2`; binary
  `/Users/nover/.cargo/bin/cargo-fuzz`.
- Installed nightly: `1.94.0-nightly`, commit
  `4f14395c37db4c1be874e6b0ace6721674223c22`, host
  `aarch64-apple-darwin`, compiler LLVM `21.1.8`. `rust-src` is installed.
- Working external symbolizer:
  `/opt/homebrew/opt/llvm/bin/llvm-symbolizer`, Homebrew LLVM `21.1.1`.
  This matches the compiler's LLVM **major**, not its exact patch revision.
  Its version probe succeeded. Actual ASan symbolizer use remains to be checked
  from campaign diagnostics; a successful version probe is not that evidence.
- Read-only disk preflight observed approximately 24 GiB free internally and
  818 GiB on the external model volume. Recheck immediately before compilation.
- The old worktree's corpus and artifact directories remain untouched at
  `/private/tmp/hmm-tfm-fuzz/crates/transformers/fuzz/`. Old ASan release
  executables are absent; its remaining debug directory is not fresh evidence.

The previous campaigns used runtime base `eb5d274` plus `b72f86a` and are not a
current-runtime replay. Relevant production changes since that run include
modular specification arithmetic, cache validation, checkpoint metadata
validation and generation behavior. The committed harnesses, seeds and fuzz
lockfile have not changed. Historical results in `CAMPAIGNS.md` are preserved.

## Commands after explicit host release

Use working directory
`/private/tmp/hmm-tfm-small-bench/crates/transformers`. Use Cargo's normal fuzz
target directory when the new disk preflight still has at least 20 GiB free;
otherwise request an approved external target, recording the exact override.
Do not clean either shared or old target directories.

Set `ASAN_SYMBOLIZER_PATH=/opt/homebrew/opt/llvm/bin/llvm-symbolizer` and
`CARGO_NET_OFFLINE=true` for every command. Record any existing `ASAN_OPTIONS`,
`RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS`, `RUSTC_WRAPPER` and `CARGO_TARGET_DIR`;
do not silently inherit compiler/instrumentation overrides. The intended
refresh has no user-supplied overrides and uses cargo-fuzz's ordinary ASan
defaults (including its `detect_odr_violation=0` setting). `--build-std` is not
enabled: this does not claim ASan instrumentation of the prebuilt Rust standard
library or arbitrary prebuilt native dependencies.

First run the existing five deterministic harness tests and warning-denied
Clippy using the pinned fuzz dependency graph:

```sh
cargo +nightly test --manifest-path fuzz/Cargo.toml --locked --lib
cargo +nightly clippy --manifest-path fuzz/Cargo.toml --locked --lib --tests -- -D warnings
cargo +nightly fuzz build --sanitizer address --target aarch64-apple-darwin
```

Create new, empty writable directories under the ignored
`fuzz/corpus/refresh-80b51e8/`, one for each target below. Refuse to reuse a
nonempty output directory. The second corpus argument is the unchanged
committed seed directory; libFuzzer writes only to the first. The historical
generated corpus is deliberately not modified or imported into this seeded
replay. Each command runs sequentially, not concurrently:

```sh
cargo +nightly fuzz run configuration --sanitizer address --target aarch64-apple-darwin \
  fuzz/corpus/refresh-80b51e8/configuration fuzz/seeds/configuration -- \
  -runs=10000 -max_len=4096 -timeout=10 -rss_limit_mb=2048 \
  -seed=483013554 -print_final_stats=1

cargo +nightly fuzz run index_artifacts --sanitizer address --target aarch64-apple-darwin \
  fuzz/corpus/refresh-80b51e8/index_artifacts fuzz/seeds/index_artifacts -- \
  -runs=10000 -max_len=4096 -timeout=10 -rss_limit_mb=2048 \
  -seed=20260909 -print_final_stats=1

cargo +nightly fuzz run masks_positions --sanitizer address --target aarch64-apple-darwin \
  fuzz/corpus/refresh-80b51e8/masks_positions fuzz/seeds/masks_positions -- \
  -runs=10000 -max_len=4096 -timeout=10 -rss_limit_mb=2048 \
  -seed=20260909 -print_final_stats=1

cargo +nightly fuzz run cache_transactions --sanitizer address --target aarch64-apple-darwin \
  fuzz/corpus/refresh-80b51e8/cache_transactions fuzz/seeds/cache_transactions -- \
  -runs=10000 -max_len=4096 -timeout=10 -rss_limit_mb=2048 \
  -seed=20260909 -print_final_stats=1

cargo +nightly fuzz run generation --sanitizer address --target aarch64-apple-darwin \
  fuzz/corpus/refresh-80b51e8/generation fuzz/seeds/generation -- \
  -runs=10000 -max_len=4096 -timeout=10 -rss_limit_mb=2048 \
  -seed=20260909 -print_final_stats=1
```

The build is separate from campaigns. Each run receives an additional external
900-second wall deadline; timeout is failure, not a shortened successful
campaign. The existing per-input 10-second libFuzzer timeout remains unchanged.
Cargo-fuzz 0.13.2 has no `--locked` build/run flag; authenticate both Cargo
lockfiles before and after and reject any change rather than claiming its
subprocess used `--locked`. Offline operation prevents registry updates but is
not itself proof of unchanged lockfiles.

## Create-only result recording

Use a new `mktemp -d` result directory for logs and machine-readable metadata.
Record an exact argument array, explicit working directory, start/end UTC and
monotonic duration, exit status, wall timeout status and raw log SHA-256 for
every build/test/campaign command. Keep failure logs and artifacts as well as
successful logs; do not overwrite the initial campaign records.

For each campaign require all of:

1. Successful process exit and exactly `stat::number_of_executed_units: 10000`.
2. No crash, timeout, out-of-memory or sanitizer failure artifact. Preserve and
   report any artifact rather than deleting it before evaluating success.
3. No unexplained sanitizer/symbolizer warning. Report symbolizer limitations
   separately from sanitizer enablement if diagnostics expose a problem.
4. Unchanged source, seed and lockfile hashes across build and execution.

The compact committed JSON should retain the five counts, seeds, raw final
statistics, actual peak RSS reported by libFuzzer, failure/artifact inventory,
logs and their hashes, exact source/toolchain/cargo-fuzz/symbolizer identities,
compiled executable hashes, initial/final corpus inventories and sizes, and
all effective instrumentation flags. Keep generated corpora and bulk logs
outside Git. A finite 50,000-input sanitizer campaign is neither exhaustive
coverage nor a GPU/model acceptance claim.
