# Bounded transformer fuzzing

This standalone development package follows the
[Rust Fuzz Book cargo-fuzz workflow](https://rust-fuzz.github.io/book/cargo-fuzz/tutorial.html).
It adds no dependencies or networking to the transformer library. Six libFuzzer
entry points reuse the same harnesses as ordinary deterministic unit tests:

| Target | Exercised code and oracle | Per-input limits |
|---|---|---|
| `configuration` | All six existing family JSON parsers, the legacy Infinity lexer, normalized serialization round trips, modular decoder/seq2seq validation | 4 KiB input; layer-count fields at most 128; no model initialization |
| `index_artifacts` | Raw shard indexes, traversal paths, duplicate keys, raw SafeTensors envelopes and headers; successful inventory bounds and containment | Two fixed-name temporary files; 4 KiB headers, 8 KiB indexes, four shards, 64 tensors, rank eight |
| `masks_positions` | Valid-mask to excluded-mask polarity, broadcast axes, fully masked rows, shape rejection and ragged absolute positions against scalar formulas | Flex CPU; batch ≤4, heads ≤4, query/key lengths ≤16; integer offsets include `usize::MAX` |
| `cache_transactions` | Advance, reset, reorder, truncate, mark, commit, rollback and stale marks against an independent length state machine; ring writes and staged commits against modulo indexing | Batch ≤4, capacity ≤32, at most 128 operations; no physical model cache |
| `generation` | Raw JSON and arbitrary IEEE-754 parameter validation; EOS/stop/length precedence, idempotence and rejection atomicity against a scalar token tracker | Vocabulary ≤32, batch ≤4, output length ≤32, four stops of length ≤4, 128 pushes |

| `experiment_interfaces` | Semantic-axis selection, activation-mask broadcasting, parameter-transfer ties and failure atomicity, stable categorical losses against scalar oracles | 4 KiB input; tensor extents ≤4, at most four parameter owners, five classes; no model initialization |

Accepted invalid input is not automatically a bug: the harness asserts the
documented invariants and allows contextual validation errors. It never catches
panics from production code. Optimized campaigns retain overflow checks, and the
default cargo-fuzz AddressSanitizer instruments the exercised dependency graph.
The independent fuzz-package lockfile is committed to reproduce this graph.
No GPU work, checkpoint downloads or random giant model allocations occur.

From the repository root, install the development tool and run the deterministic
oracles first:

```sh
cargo install cargo-fuzz --version 0.13.2 --locked
cargo test --manifest-path fuzz/Cargo.toml --lib
cargo clippy --manifest-path fuzz/Cargo.toml --lib --tests -- -D warnings
```

Use an installed nightly toolchain with `rust-src`. Check at least 20 GiB of
local free space before building. If that preflight fails, select a sufficiently
large target directory with `--target-dir`; do not delete unrelated build data.
Create a writable corpus directory, and supply the committed seeds as a second,
read-only corpus. For example:

```sh
mkdir -p fuzz/corpus/configuration
cargo +nightly fuzz run configuration \
  fuzz/corpus/configuration fuzz/seeds/configuration -- \
  -runs=10000 -max_len=4096 -timeout=10 -rss_limit_mb=2048
```

Repeat with `index_artifacts`, `masks_positions`, `cache_transactions`, and
`generation`, and `experiment_interfaces`. Keep campaigns bounded by `-runs` or `-max_total_time`; omitted
limits run indefinitely. Seeds remain committed; generated corpora, sanitizer
builds, crash artifacts and coverage profiles are ignored. Reproduce a crash with
`cargo +nightly fuzz run TARGET fuzz/artifacts/TARGET/crash-HASH`, minimize it with
`cargo +nightly fuzz tmin TARGET PATH`, fix the underlying invariant, and retain a
small regression fixture or deterministic unit test before resuming.

These campaigns cover parser, metadata and host/Flex invariants. They do not
replace full-model reference parity, accelerator tests, loader payload mapping
tests or numerical-quality acceptance. Finite smoke campaigns are evidence of
executed inputs, not proof that every reachable branch is covered.

The initial bounded campaign results and limitations are recorded in
[CAMPAIGNS.md](CAMPAIGNS.md). This development package is intentionally outside
the runtime crate's Cargo-package allowlist; use the repository checkout to fuzz.
