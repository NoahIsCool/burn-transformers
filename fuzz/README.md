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
| `configuration` | Every registered family JSON parser (the six Llama-family `model_type`s, Gemma 2 and 3, gpt-oss, GPT-2, GPT-NeoX, Qwen3.5, Qwen3-MoE, DeepSeek-V2, Muse Glimmer, Nemotron-H, T5), the legacy Infinity lexer, normalized serialization round trips, every accepted layer and routed-expert count within the shared `MAX_LAYERS`/`MAX_EXPERTS` bounds, modular decoder/seq2seq validation | 4 KiB input; counts are not filtered, so absurd values must be rejected quickly; no model initialization |
| `index_artifacts` | Raw shard indexes, traversal paths, duplicate keys, raw SafeTensors envelopes and headers; successful inventory bounds and containment | Two fixed-name temporary files; 4 KiB headers, 8 KiB indexes, four shards, 64 tensors, rank eight |
| `safetensors_headers` | *Generated* SafeTensors headers with every dtype (including `F4`, `F6_*`, `F8_E8M0` and MXFP4-style `*_blocks`/`*_scales` pairs), overflow-boundary shapes, perturbed offsets, metadata, padding and length prefixes, plus shard indexes with dropped, phantom, moved, duplicated, unsafe and aliased entries and odd `total_size` spellings; an independent byte-accounting oracle predicts every accept/reject decision and inventoried field, and the official `safetensors` 0.7 parser must accept every accepted file | One shard of at most four tensors (64 KiB data) or three shards; tight and generous limits |
| `masks_positions` | Valid-mask to excluded-mask polarity, broadcast axes, fully masked rows, shape rejection and ragged absolute positions against scalar formulas | Flex CPU; batch ≤4, heads ≤4, query/key lengths ≤16; integer offsets include `usize::MAX` |
| `attention_connectivity` | Global, causal, sliding-window and chunked-local exclusion masks with padded batches and absolute key starts against a scalar position oracle; exact sliding-window key ranges; grouped-query attention under the mask (both precision modes, native KV heads, backend `is_causal`) against an `f64` softmax | Flex CPU; batch ≤3, heads ≤3, query/key lengths ≤8, head width ≤3 |
| `cache_transactions` | Advance, reset, reorder, truncate, mark, commit, rollback and stale marks against an independent length state machine; ring writes and staged commits against modulo indexing; then the tensor decode state (full, ring, mirrored, latent and recurrent groups and an optional MTP bank with its own cursor; batched and per-row step write sets; write-ahead published, abandoned or invalidated; ragged speculative commits with MTP prefixes drawn independently of the main prefix; reorder, truncate, `truncate_mtp`, reset) against per-bank token-history oracles that compare every key and value of per-row and batched reads | Length machine: batch ≤4, capacity ≤32, at most 128 operations (one per four input bytes), no physical cache. Tensor state: rows ≤3, capacity ≤12, 1–2 layers, 1–2 heads/channels, at most 64 operations (one per six input bytes, so long sequences need `-len_control=0`) |
| `generation` | Every `GenerationConfig` rule (warpers, penalties, `min_new_tokens`, special and stop IDs, `logit_bias`, float controls representable in F32) for raw JSON and decoded configurations, bit-exact JSON round trips, and EOS/stop/length precedence with `min_new_tokens` against a scalar token tracker | Vocabulary ≤32, batch ≤4, output length ≤32, four stops of length ≤4, 128 pushes |
| `logits_processing` | `LogitsProcessor` with `TokenHistory`: ragged and out-of-vocabulary rows, prompt boundaries, bias, EOS suppression, no-repeat n-grams, repetition/presence/frequency penalties, temperature, top-k (ties kept), top-p and min-p against an `f64` oracle; rebuilt, recorded, stateless and legacy masked histories must agree bit for bit; every valid configuration, including `f32`-boundary controls, keeps logits finite or `-inf` and selection defined | Vocabulary ≤8, batch ≤3, rows ≤6 tokens |
| `rope_parameters` | Flat and per-layer-type `rope_parameters` (every algorithm and key, legacy `type`, boundary values, raw JSON) through the shared RoPE translator under arbitrary family contexts: contextual errors, finite `f32` frequencies, angles at the largest integer position, and amplitudes for every accepted configuration, `Ignore ⊇ Reject`, finite softmax multipliers, bit-exact round trips, layer-type validation | Head width and context from fixed tables |
| `chat_templates` | Untrusted Jinja templates (Hugging Face idioms, sandbox escapes, raw bytes, huge repetition, padding and doubling counts) and messages through `LocalTokenizer::render_chat` on a word-level fixture: no panics, contextual errors for every documented precondition (all six shadowed message fields, all fifteen reserved variables), determinism with a pinned clock, unnamed template selection against Hugging Face's `get_chat_template` rule, `encode_chat` consistency, and time/output bounds | 24 fragments, three messages; 2 s and 64 MiB per render |
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
| CHAT-4 | `chat_templates::tests::tools_select_the_tool_use_template` | With `tools` and no template name, a checkpoint's named `tool_use` template is ignored and `default` renders; Hugging Face 5.2 selects `tool_use` |
| TOPP-TIE-1 | `logits_processing::tests::top_p_ranks_by_logit_like_hugging_face` | Top-p ranks rounded `f32` probabilities, so it can drop a logit one ulp above a kept one; Hugging Face ranks by logit |

The harnesses tolerate exactly these two: the selection check also accepts
`default` where CHAT-4 should select `tool_use`, and the top-set check lets
top-p split near-ties (TOPP-TIE-1, also in the `tests/generation_properties.rs`
mode property). Otherwise they require the absence of panics and every
documented invariant. Remove each tolerance when its finding is fixed.

One further difference is pinned rather than ignored: `render_chat` rejects an
unknown template name, while Hugging Face renders the name itself as template
source (`get_chat_template` falls through to treating `chat_template` as Jinja).
Rejection is the safer contract; it is a documentation request for the
tokenization module's list of known differences, checked by
`chat_templates::tests::documented_preconditions_are_rejected`.

## Fixed findings

Each fixed finding keeps its reproducer, now an ordinary (not ignored) unit
test, and a focused regression test in the owning library module.

| ID | Reproducer | Fix |
|---|---|---|
| CFG-COUNT-1 | `configuration::tests::{absurd_counts_are_rejected_by_every_family_that_reads_them, committed_seeds_replay_and_cover_every_family}` | Found by review, not by a campaign: the harness skipped every input with a layer count above 128, which hid that the Llama, gpt-oss, Qwen3.5, and Muse validators panicked (`capacity overflow`) at `usize::MAX` and that other families accepted counts their contracts then listed without bound. Every family now checks the shared `MAX_LAYERS`/`MAX_EXPERTS` bounds (and per-expert layouts the ten-million-tensor `MAX_CHECKPOINT_TENSORS` inventory) before allocating, and the harness asserts the bounds on every accepted configuration instead of filtering; one over-bound seed per family is committed |
| GEN-F32-1..4 | `logits_processing::tests::{finite_additive_penalties_never_produce_nan, positive_repetition_penalty_never_produces_nan, constrained_tokens_stay_negative_infinity_under_any_valid_bias, positive_temperature_never_produces_nan}` | `GenerationConfig::validate` checks float controls in F32, the processing precision (divisors round to normal positive values, additive controls to finite ones; F16/BF16 logits are checked in their own dtype), and processing saturates overflow so logits stay finite or `-inf` |
| CHAT-1..3 | `chat_templates::tests::{huge_string_repetition_fails_contextually, rendered_prompts_are_bounded, concatenated_sequences_finish_quickly}` | The renderer rewrites `+`, `~`, `*`, slices, and `*`/`**` call arguments into bounded filters, bounds every print, the size-controlled builtins, and the rendered prompt, charges every builtin, function, and Python method that copies its input (so the budget also bounds what a template keeps alive), materializes sequences within an item limit, and renders under `catch_unwind` (see the `chat_template` module documentation for the guarantees and the remaining gaps) |
| ROPE-SOFTMAX-1 | `rope_parameters::tests::translated_softmax_corrections_are_finite` | The softmax multiplier `m^2`, not only `m`, must be a normal positive `f32` |
| ROPE-F32-1/2 | `rope_parameters::tests::{accepted_bases_have_finite_frequencies, accepted_long_rope_factors_have_finite_frequencies}` | Bases must be at least one, LongRoPE factors must round to at least `2^-64` (`LongRopeScaling::MIN_FACTOR`), and RoPE amplitudes must be normal positive `f32` values, which keeps every `f32` frequency, and every angle at any integer position, finite (a review showed that the earlier floor, `f32::MIN_POSITIVE`, kept frequencies finite but overflowed angles from position 4 on) |
| ROPE-PARSE-1 | `rope_parameters::tests::arrays_are_not_rope_parameters` | `RopeParameters` deserializes only from a JSON object |
| CHAT-5 | `namespace_cycles_do_not_outlive_the_render` (in the library's chat-template sandbox tests) | Found while investigating a campaign's peak RSS: a template could leak memory across renders through namespace reference cycles (`{% set ns.me = ns %}`); every namespace's reference-holding attributes are now cleared when the render ends, grouped by name so the clearing costs only what the render built (`namespace_cleanup_costs_only_what_the_render_built`; the first version cleared every name on every namespace, which a review showed could take seconds and gigabytes) |
| CHAT-6 | `limits_follow_the_context_size` (same module) | Also from that investigation: the 16 MiB prompt floor let a template make `encode_chat` use about 2 GB (the tokenizer keeps over 100 bytes of state per prompt byte); the floor is now 1 MiB plus four times the context |
| CHAT-7 | `divisibleby_never_panics` (same module) | Found by the library's typed differential property, not by a campaign: MiniJinja's `divisibleby` test computes `a % 0` unchecked and panics; a checked replacement reports Jinja2's `ZeroDivisionError` |

The fixes removed the corresponding tolerances: the `f32` and frequency
envelopes, the `|mscale_all_dim| <= 1e100` bound, and the small growth counts of
the chat-template fragments. With `serde_json`'s `float_roundtrip` feature
enabled by the runtime crate (and so in this package), JSON round trips are
compared bit for bit instead of within a relative `1e-14`.

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
