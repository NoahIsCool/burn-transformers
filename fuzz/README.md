# Bounded transformer fuzzing

This standalone development package follows the
[Rust Fuzz Book cargo-fuzz workflow](https://rust-fuzz.github.io/book/cargo-fuzz/tutorial.html).
It adds no dependencies or networking to the transformer library. Twelve
libFuzzer entry points reuse the same harnesses as ordinary deterministic unit
tests. Each harness decodes its input with a *total* byte decoder (exhausted
input reads as zero), so every byte string is a valid case, and checks the
library against an independent oracle:

| Target | Exercised code and oracle | Per-input limits |
|---|---|---|
| `configuration` | All six existing family JSON parsers, the legacy Infinity lexer, normalized serialization round trips, modular decoder/seq2seq validation | 4 KiB input; layer-count fields at most 128; no model initialization |
| `index_artifacts` | Raw shard indexes, traversal paths, duplicate keys, raw SafeTensors envelopes and headers; successful inventory bounds and containment | Two fixed-name temporary files; 4 KiB headers, 8 KiB indexes, four shards, 64 tensors, rank eight |
| `safetensors_headers` | *Generated* SafeTensors headers with every dtype (including `F4`, `F6_*`, `F8_E8M0` and MXFP4-style `*_blocks`/`*_scales` pairs), overflow-boundary shapes, perturbed offsets, metadata, padding and length prefixes, plus shard indexes with dropped, phantom, moved, duplicated, unsafe and aliased entries and odd `total_size` spellings; an independent byte-accounting oracle predicts every accept/reject decision and inventoried field, and the official `safetensors` 0.7 parser must accept every accepted file | One shard of at most four tensors (64 KiB data) or three shards; tight and generous limits |
| `masks_positions` | Valid-mask to excluded-mask polarity, broadcast axes, fully masked rows, shape rejection and ragged absolute positions against scalar formulas | Flex CPU; batch ≤4, heads ≤4, query/key lengths ≤16; integer offsets include `usize::MAX` |
| `attention_connectivity` | Global, causal, sliding-window and chunked-local exclusion masks with padded batches and absolute key starts against a scalar position oracle; exact sliding-window key ranges; grouped-query attention under the mask (both precision modes, native KV heads, backend `is_causal`) against an `f64` softmax | Flex CPU; batch ≤3, heads ≤3, query/key lengths ≤8, head width ≤3 |
| `cache_transactions` | The metadata length machine and ring planners against integer oracles, then the tensor decode state (full, ring, mirrored, latent and recurrent groups and an optional MTP bank with its own cursor; batched and per-row step write sets; write-ahead published, abandoned or invalidated; ragged speculative commits with independent MTP prefixes; reorder, truncate, `truncate_mtp`, reset) against per-bank token-history oracles | Rows ≤3, capacity ≤12, 1–2 layers, 1–2 heads/channels, at most 64 operations |
| `generation` | Every `GenerationConfig` rule (warpers, penalties, `min_new_tokens`, special and stop IDs, `logit_bias`) for raw JSON and decoded configurations, JSON round trips, and EOS/stop/length precedence with `min_new_tokens` against a scalar token tracker | Vocabulary ≤32, batch ≤4, output length ≤32, four stops of length ≤4, 128 pushes |
| `logits_processing` | `LogitsProcessor` with `TokenHistory`: ragged and out-of-vocabulary rows, prompt boundaries, bias, EOS suppression, no-repeat n-grams, repetition/presence/frequency penalties, temperature, top-k, top-p and min-p against an `f64` oracle; rebuilt, recorded, stateless and legacy masked histories must agree bit for bit; seeded selection | Vocabulary ≤8, batch ≤3, rows ≤6 tokens |
| `rope_parameters` | Flat and per-layer-type `rope_parameters` (every algorithm and key, legacy `type`, boundary values, raw JSON) through the shared RoPE translator under arbitrary family contexts: contextual errors, valid frequency tables, `Ignore ⊇ Reject`, softmax corrections, round trips, layer-type validation | Head width and context from fixed tables |
| `chat_templates` | Untrusted Jinja templates (Hugging Face idioms, sandbox escapes, raw bytes) and messages through `LocalTokenizer::render_chat` on a word-level fixture: no panics, contextual errors for documented preconditions, determinism with a pinned clock, `encode_chat` consistency, and time/output bounds | 24 fragments, three messages; 2 s and 64 MiB per render |
| `modular_specs` | Typed modular `DecoderSpec`/`Seq2SeqSpec` with every mixer, feed-forward, routing, position, connectivity and normalization option: validation totality, JSON round trips, construction exactly for valid specs without documented capability gaps, finite logits, and prefill+decode parity with the stateless forward | Width ≤16, two layers, vocabulary ≤16, four tokens |
| `experiment_interfaces` | Semantic-axis selection, activation-mask broadcasting, parameter-transfer ties and failure atomicity, stable categorical losses against scalar oracles | 4 KiB input; tensor extents ≤4, at most four parameter owners, five classes; no model initialization |

Accepted invalid input is not automatically a bug: the harness asserts the
documented invariants and allows contextual validation errors. It never catches
panics from production code. Optimized campaigns retain debug assertions and
overflow checks. The independent fuzz-package lockfile is committed to
reproduce the dependency graph. No GPU work, checkpoint downloads or random
giant model allocations occur.

## Open findings

Fuzzing found behaviors that the harnesses deliberately avoid asserting while
their fixes are open, because they live in files outside this package. Each has
a minimized reproducer as an `#[ignore]`d unit test that fails today and should
pass once fixed; run them with `cargo test --manifest-path fuzz/Cargo.toml --lib
-- --ignored`. [CAMPAIGNS.md](CAMPAIGNS.md) records how each was found.

| ID | Test | Behavior |
|---|---|---|
| GEN-F32-1..4 | `logits_processing::tests::{finite_additive_penalties_never_produce_nan, positive_repetition_penalty_never_produces_nan, constrained_tokens_stay_negative_infinity_under_any_valid_bias, positive_temperature_never_produces_nan}` | `GenerationConfig::validate` accepts finite penalties, biases and temperatures whose `f32` rounding is zero or infinite; processing then yields NaN logits |
| CHAT-1 | `chat_templates::tests::huge_string_repetition_fails_contextually` | `'ab' * n` with a huge `n` panics ("capacity overflow") inside `render_chat` |
| CHAT-2 | `chat_templates::tests::rendered_prompts_are_bounded` | One expression renders a prompt of hundreds of megabytes; fuel does not bound memory |
| CHAT-3 | `chat_templates::tests::concatenated_sequences_finish_quickly` | Lazily concatenated sequences make one instruction take seconds to minutes; fuel does not bound time |
| ROPE-F32-1/2 | `rope_parameters::tests::{accepted_bases_have_finite_frequencies, accepted_long_rope_factors_have_finite_frequencies}` | A base `theta < 1` or a tiny LongRoPE factor validates but overflows `f32` inverse frequencies |
| ROPE-SOFTMAX-1 | `rope_parameters::tests::translated_softmax_corrections_are_finite` | Translation validates the `mscale_all_dim` correction `m` but not the multiplier `m^2`, which can be infinite |
| ROPE-PARSE-1 | `rope_parameters::tests::arrays_are_not_rope_parameters` | A JSON array parses as a positional `RopeParameters` struct |

The harnesses assert finiteness only inside the documented envelopes
(`within_f32_envelope`, `within_frequency_envelope`, `abs(mscale_all_dim) <=
1e100`), keep the CHAT-1..3 growth fragments small, and otherwise still require
the absence of panics.

## Running

From the repository root, install the development tool and run the deterministic
oracles first. Each harness also runs a few hundred pseudo-random inputs from a
fixed linear congruential generator, its committed seeds, and boundary inputs:

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

On macOS 27 the nightly AddressSanitizer runtime deadlocks at startup; add
`-s none` (`cargo +nightly fuzz run -s none ...`) to build without a sanitizer.
The harness's own assertions, Rust's bounds and overflow checks, and libFuzzer's
timeout and RSS limits still apply. Flex parallelizes with rayon; set
`RAYON_NUM_THREADS=1` when running several campaigns side by side.

Repeat with every target in the table. Keep campaigns bounded by `-runs` or
`-max_total_time`; omitted limits run indefinitely. To keep a long campaign
running past known open findings, add `-fork=1 -ignore_crashes=1
-ignore_timeouts=1 -ignore_ooms=1`; every failure is still written to the
artifact directory. Seeds remain committed; generated corpora, sanitizer builds,
crash artifacts and coverage profiles are ignored. Reproduce a crash with
`cargo +nightly fuzz run TARGET fuzz/artifacts/TARGET/crash-HASH`, minimize it with
`cargo +nightly fuzz tmin TARGET PATH`, fix the underlying invariant, and retain a
small regression fixture or deterministic unit test before resuming.

These campaigns cover parser, metadata, host/Flex, and tiny-model invariants.
They do not replace full-model reference parity, accelerator tests, loader
payload mapping tests or numerical-quality acceptance. Finite smoke campaigns
are evidence of executed inputs, not proof that every reachable branch is
covered. Algebraic properties that relate several calls (row independence,
penalty monotonicity, padding invariance, RoPE shape equivalence) live in the
proptest suites `tests/generation_properties.rs`, `tests/rope_properties.rs` and
`tests/connectivity_properties.rs`.

Bounded campaign results and limitations are recorded in
[CAMPAIGNS.md](CAMPAIGNS.md). This development package is intentionally outside
the runtime crate's Cargo-package allowlist; use the repository checkout to fuzz.
