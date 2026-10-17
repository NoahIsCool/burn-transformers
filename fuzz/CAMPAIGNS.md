# Bounded fuzz campaign evidence

Entries are append-only and name their source, toolchain, sanitizer, and limits.

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

## 2026-09-09: current-runtime v0.1 refresh

The [compact authenticated record](evidence/refresh-80b51e8.json) refreshes all
five targets against integration commit
`80b51e838033cc479ab7d99a97106e5c2309d5a8`, runtime source Git tree
`b4d4a96ca38fdb709bce07b6fd384d497e523f98`. All 137 tracked runtime, harness,
seed and lockfile compilation inputs had identical content inventories before
and after execution. Neither Cargo lockfile changed. The initial campaign
above remains historical evidence; it is not relabeled as this replay.

| Target | Executed inputs | Seed | Native exit | LibFuzzer peak RSS |
| --- | ---: | ---: | ---: | ---: |
| configuration | 10,000 | 483013554 | 0 | 157 MiB |
| index_artifacts | 10,000 | 20260909 | 0 | 205 MiB |
| masks_positions | 10,000 | 20260909 | 0 | 292 MiB |
| cache_transactions | 10,000 | 20260909 | 0 | 127 MiB |
| generation | 10,000 | 20260909 | 0 | 100 MiB |

All five final execution counters were exactly 10,000. There were no native
failure exits, timeout/crash/OOM artifacts, or warning/error/summary
diagnostics. Instrumented executable hashes matched before and after each
campaign. Each target used an initially empty writable corpus plus its
unchanged committed seeds, `-runs=10000 -max_len=4096 -timeout=10
-rss_limit_mb=2048 -print_final_stats=1`, and an independent 900-second wall
deadline. Previous generated corpora and artifacts were preserved, not imported
or overwritten. The reported RSS/rate counters are libFuzzer diagnostics, not
controlled comparative benchmark results.

The five deterministic Rust harness tests and warning-denied Clippy passed
before the optimized cargo-fuzz build. Its verbose log confirms ASan,
sanitizer coverage, debug assertions and overflow checks in the runtime,
shared harness and all five entry points. Cargo-fuzz `0.13.2` and nightly
`1.94.0-nightly (4f14395c3 2025-12-22)` were used. The installed Homebrew
symbolizer `21.1.1` was selected explicitly; its version probe passed and
campaigns had no symbolizer warning. It matches compiler LLVM `21.1.8` only at
the major version. No intentional crash or symbolized stack-trace qualification
was performed. Prebuilt standard-library/native dependency closure is not
claimed to be rebuilt or hermetically authenticated.

[The supervisor](evidence/run_refresh.py) builds once, then executes native
targets directly to preserve their exit codes. Its seven Python tests cover
failure accounting, native signals/timeouts, create-only records, corpus
identities and compact/raw-record consistency. Source, executable, tool and
raw-log hashes, exact argument arrays, timestamps, final counters and corpus
inventory digests are retained in the compact record. Bulk logs, generated
corpora and the complete source/corpus inventories remain at
`/private/tmp/transformers-fuzz-refresh.EXC7pm`; their raw report is hashed in
the committed JSON. No GPU work or runtime source changes were involved.

## 2026-09-10: experiment interface invariants

The [new interface campaign](../acceptance/measurements/coding-standards-fuzz-20260910/README.md)
ran `experiment_interfaces` for exactly 10,000 inputs under AddressSanitizer
with seed `20260910`. It passed without crashes, timeouts, or sanitizer
diagnostics. Six ordinary fuzz-oracle tests and all-target warning-denied fuzz
Clippy also passed. The record retains source, binary, seed, command, and storage
identities. This new-target campaign does not refresh the five historical
campaigns above or establish exhaustive branch coverage.

## 2026-10-15: wave-1 cache fix-phase campaigns (as reported)

The cache work package extended `cache_transactions` to the tensor decode state
(1–2 layers and heads, full/ring/mirrored/latent/recurrent groups, write-ahead
endings, per-row and batched read oracles; eight-byte header, operations
`op % 8`) and reported two campaigns at library source `2452075`, run with
`cargo-fuzz 0.13.2` and `-s none` because the nightly ASan runtime deadlocks on
macOS 27: 109,228 inputs in 181 s with default length control (inputs reached
153 bytes; cov 3035, ft 18237, corpus 1054), and 50,782 inputs in 151 s with
`-len_control=0 -max_len=4096` (cov 3035, ft 19718, corpus 1015), both without
failures. They are recorded here from that report and were not re-run; round 2
below refreshes the target, now with an MTP bank, at `00697a4`.

## 2026-10-15: wave-3 targets, findings, and an all-target refresh

Eight targets were added or extended for the polish pass (six new ones:
`logits_processing`, `chat_templates`, `rope_parameters`, `safetensors_headers`,
`attention_connectivity`, `modular_specs`; and the extended `generation` and
`cache_transactions` harnesses). The runtime source was unchanged throughout:
`src/` tree `fa40ba91781e6c7b39554385e598d90c5eaa4d9b`, as at integration commit
`8720be1`. Campaigns used `cargo-fuzz 0.13.2`, `rustc 1.94.0-nightly (4f14395c3
2025-12-22)`, macOS 27.0.1 on Apple Silicon (14 cores), and **no sanitizer**
(`-s none`): the nightly AddressSanitizer runtime deadlocks at startup on this
host. Builds kept debug assertions and overflow checks (the cargo-fuzz default).
Every campaign ran the built executable directly with `RAYON_NUM_THREADS=1`, a
fresh writable corpus on the external build volume plus the committed seeds,
and `-max_len=4096 -timeout=10 -rss_limit_mb=2048 -print_final_stats=1`; the
targets of a round ran side by side, so the rates below are concurrent rates,
not benchmarks. Generated corpora, logs, and artifacts stay on the build volume
(`cargo-targets/tfm-polish-w3-fuzz/campaigns/`) and are not committed.

### Round 1: harness development (uncommitted harnesses, 300 s per target)

Four campaigns stopped on a failure. Each crash was minimized with
`-minimize_crash=1` and kept as a unit test.

| Target | Executed inputs | Outcome | Classification |
| --- | ---: | --- | --- |
| `logits_processing` | 53,095 | `crash-cfb301be` (32 bytes minimized) | **Library deviation TOPP-TIE-1** (first recorded here as an over-strict oracle; reclassified in the review fix phase): top-p ranks rounded `f32` probabilities, so logits one ulp apart (0.8 and 0.79999995) can split, while Hugging Face sorts logits and never drops the larger. The top-set check tolerates near-ties until it is fixed. Regression `campaign_top_p_near_ties_may_split`; ignored reproducer `top_p_ranks_by_logit_like_hugging_face`. |
| `chat_templates` | 4,847 | `crash-86c5a5ad` | Oracle too strict: an empty rendered prompt encodes to no tokens, which `encode_chat` reports. Regression `empty_prompts_fail_to_encode_contextually`. |
| `rope_parameters` | 258,317 | `crash-3a306a4f` (19 bytes) | **Library finding ROPE-SOFTMAX-1**: translation validates the correction `m = 1 + 0.1 * mscale_all_dim * ln(factor)` but the softmax multiplier is `m^2`, which overflows to `inf` (e.g. `mscale_all_dim = 1e300`). Ignored reproducer `translated_softmax_corrections_are_finite`; the harness asserts finiteness only for `abs(mscale_all_dim) <= 1e100`. |
| `generation` | 162,032 | `crash-8872c8a0` (18 bytes) | Oracle too strict: `serde_json` (without `float_roundtrip`) read `3.7997969716361836e-262` back four ulps away; round trips now compare to a relative `1e-14`. Regression `campaign_float_round_trip_is_approximate`. |
| `safetensors_headers` | 441,423 | Passed; cov 2889, ft 7556, corpus 567 | |
| `attention_connectivity` | 1,407,002 | Passed; cov 2004, ft 5894, corpus 439 | |
| `modular_specs` | 773,524 | Passed; cov 9113, ft 32129, corpus 2024 | |
| `chat_templates` (rerun after the fix) | 262,981 | Passed; cov 7539, ft 22145, corpus 1795 | |

Targeted probing while writing the harnesses (not campaign output) found the
remaining open findings: `GenerationConfig::validate` accepts penalties, biases,
and temperatures that are zero or infinite in `f32` and then yield NaN logits
(GEN-F32-1..4); `render_chat` panics on `'ab' * 4611686018427387904` (CHAT-1),
renders a 200 MB prompt from `{{ 'ab' * 100000000 }}` (CHAT-2), and spends
about ten minutes on a 30-step lazily concatenated list that costs about 100
units of fuel (CHAT-3); a base `theta < 1` or a LongRoPE factor of `1e-45`
overflows `f32` frequencies (ROPE-F32-1/2); and a JSON array parses as
positional RoPE parameters (ROPE-PARSE-1). Each has an `#[ignore]`d reproducer
that fails today (see [README.md](README.md#open-findings)). The independent
review of the package added CHAT-4 (unnamed renders ignore a named `tool_use`
template when `tools` is set; found by comparing with Hugging Face 5.2
`get_chat_template`, not by a campaign) and reclassified `crash-cfb301be` as
TOPP-TIE-1.

### Round 2: committed harnesses at `00697a4`

All twelve targets were rebuilt from commit `00697a4` (fuzz package tree
`9d76aa6f0a1a3a8991cee520d070d311a6d6d2d4`, lockfile SHA-256
`f974aac22ee2e71f3115ff6b2e29cc914f1ec5bc17019f47c06743687d6a3637`). Every
campaign exited with status 0, without crash, timeout, or out-of-memory
artifacts.

| Target | Seconds | Executed inputs | exec/s | cov | ft | Corpus | `lim` | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logits_processing` | 300 | 815,906 | 2,710 | 2686 | 7547 | 509 | 4096 | 48 MiB |
| `chat_templates` | 300 | 176,175 | 585 | 7575 | 21426 | 1579 | 85 | 203 MiB |
| `rope_parameters` | 300 | 1,122,463 | 3,729 | 2030 | 6696 | 1715 | 1151 | 47 MiB |
| `safetensors_headers` | 300 | 307,236 | 1,020 | 2889 | 7554 | 580 | 702 | 47 MiB |
| `attention_connectivity` | 300 | 889,579 | 2,955 | 2004 | 5900 | 453 | 4096 | 46 MiB |
| `modular_specs` | 300 | 361,150 | 1,199 | 9085 | 31846 | 1820 | 308 | 64 MiB |
| `generation` | 300 | 983,877 | 3,268 | 1903 | 4871 | 1580 | 887 | 45 MiB |
| `cache_transactions` (with the MTP bank) | 300 | 103,053 | 342 | 3579 | 21279 | 1193 | 139 | 58 MiB |
| `configuration` | 120 | 404,729 | 3,344 | 4071 | 8446 | 1041 | 1307 | 48 MiB |
| `index_artifacts` | 120 | 117,298 | 969 | 1721 | 2805 | 666 | 77 | 44 MiB |
| `masks_positions` | 120 | 322,289 | 2,663 | 523 | 1010 | 110 | 2805 | 43 MiB |
| `experiment_interfaces` | 120 | 392,433 | 3,243 | 2710 | 6880 | 239 | 3012 | 47 MiB |

`lim` is libFuzzer's input-length limit when the campaign ended: with its
default length control, inputs grow gradually, so a short campaign explores
short inputs only (column added in the review fix phase from the retained
logs). At 139 bytes `cache_transactions` ran at most 24 tensor-state operations
per input (one per six bytes), not the harness maximum of 64, so its MTP
sequences were short; at 85 bytes `chat_templates` rarely combined long
raw-byte fragments with full templates and three messages; `index_artifacts`
stopped at 77 bytes. Round 3 reruns the
two changed long-input targets with `-len_control=0`.

`cache_transactions` reached coverage 3579, up from 3035 in the wave-1 fix-phase
campaign, mostly from the MTP bank. These are finite smoke campaigns without a
sanitizer: they show that the executed inputs satisfied every oracle, not that
every reachable branch was covered. The ASan evidence above covers only the
older five targets' dependency graph. The wave-3 fuzz graph adds C and Rust code
that has **never run under any sanitizer**: `tokenizers` with the `onig` regular
expression library (C, exercised on every `chat_templates` input), `minijinja`,
and the official `safetensors` 0.7 parser. The sanitizer requirement of
`RELEASE_GATES.md` therefore remains open for every wave-3 target, in particular
`chat_templates` and `safetensors_headers`, until ASan campaigns run on a host
where the nightly runtime works.

### Round 3: review fix phase at `8089167`

An independent review of the package strengthened five harnesses:
`cache_transactions` (per-row reads of both banks compared by key and value;
MTP commit prefixes drawn from their own selector bits), `chat_templates`
(unnamed template selection against Hugging Face's rule; full shadowed-field and
reserved-variable lists), `rope_parameters` (contextual lenient errors; softmax
multipliers of per-layer entries), and `generation` and `modular_specs` (JSON
round trips compare integers exactly). These five were rebuilt from commit
`8089167` (fuzz package tree `ee8f4f459319a03401394173ad90394924db64e9`, same
lockfile and `src/` tree as round 2; build flags include `-Cdebug-assertions`)
and ran for 300 s each with the round-2 settings, reading the round-2 corpus of
the same target as a second, read-only corpus. `cache_transactions` and
`chat_templates` also used `-len_control=0`, so long inputs were explored from
the start. `cache_transactions` and `chat_templates` started at 17:02:13 EDT and the
other three at 17:03:38 (the first launch of those three failed in the shell
script before any target started), so the rates are again concurrent rates. All five exited with status 0,
without crash, timeout, or out-of-memory artifacts.

| Target | Seconds | Executed inputs | exec/s | cov | ft | Corpus | `lim` | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cache_transactions` | 300 | 86,339 | 286 | 3597 | 23661 | 1310 | 4096 | 56 MiB |
| `chat_templates` | 300 | 240,052 | 797 | 7710 | 28908 | 2026 | 4096 | 439 MiB |
| `rope_parameters` | 300 | 1,503,334 | 4,994 | 2066 | 7141 | 1658 | 4096 | 49 MiB |
| `generation` | 300 | 1,318,410 | 4,380 | 1957 | 5261 | 1630 | 4096 | 46 MiB |
| `modular_specs` | 300 | 671,699 | 2,231 | 9102 | 32859 | 1604 | 1928 | 64 MiB |

With `-len_control=0`, `cache_transactions` kept inputs of up to 996 bytes (97
new corpus entries of at least 384 bytes, enough for all 64 tensor-state
operations), and coverage rose from 3579 to 3597. `chat_templates` kept inputs
of up to 427 bytes; its peak RSS rose to 439 MiB with the longer inputs (the
cause was not profiled), below the 2048 MiB limit. The other
seven targets were not rerun: `logits_processing` changed only in comments and
tests, and the rest did not change. Round 3 is still sanitizer-free smoke
evidence with the limitations stated for round 2.

## 2026-10-15: wave-3 robustness fixes (GEN-F32-1..4, CHAT-1..3, ROPE-*)

The robustness package fixed the eleven open library findings recorded above
(see [README.md](README.md#fixed-findings)) in four library commits plus a
follow-up guard fix and a Verus-checked refactor, un-ignored their reproducers,
and removed the harness tolerances the fixes made unnecessary: the
`within_f32_envelope` and `within_frequency_envelope` checks, the
`|mscale_all_dim| <= 1e100` bound, the small chat-template growth counts (the
fragment table now repeats strings `2^62` times, doubles strings and lists 40
times, and pads, replaces, and formats with widths far past the renderer's
limits), and the relative `1e-14` float round-trip tolerance (the runtime crate
now enables `serde_json`'s `float_roundtrip`, which this package inherits).
The generation oracle restates the new F32 validation rules.

Two harness unit tests already failed on the integration base `614cbf5`,
independently of this package: `logits_processing`'s top-set oracle counted
exactly `k` top-k survivors, but the library (like Hugging Face) keeps every
value tied with the `k`-th, so all-equal rows failed. The oracle now counts
ties, which fixed `campaign_top_p_near_ties_may_split` and
`pseudo_random_inputs_agree_with_the_oracles`. Two `modular_specs` tests
(`capability_gaps_are_reported_at_construction`,
`pseudo_random_specs_validate_construct_and_run`) also fail on the base,
because construction now succeeds for a combination the harness still lists as
a capability gap; that harness is outside this package, so `modular_specs` was
not fuzzed here.

Campaigns used `cargo-fuzz 0.13.2`, `rustc 1.94.0-nightly (4f14395c3
2025-12-22)`, macOS 27 on Apple Silicon, **no sanitizer** (`-s none`, as above),
the cargo-fuzz release profile with debug assertions and overflow checks, and
the unchanged fuzz lockfile (SHA-256
`f974aac22ee2e71f3115ff6b2e29cc914f1ec5bc17019f47c06743687d6a3637`). Each target
ran its built executable directly with `RAYON_NUM_THREADS=1`, a fresh writable
corpus on the external build volume, the committed seeds, and
`-max_len=4096 -timeout=10 -rss_limit_mb=2048 -print_final_stats=1`;
`chat_templates` also used `-len_control=0`. All five targets of a round ran
side by side, so rates are concurrent rates. Every campaign exited with status
0 and wrote no crash, timeout, or out-of-memory artifact.

### Round 1: commit `68085fb` (all four library fixes)

Fuzz package tree `fcd1538a91f715cf001be671ffe1459d6b590a0f`, `src/` tree
`b6f54d6d7872fe0f20c5498054cf4927a0ff74b8`.

| Target | Seconds | Executed inputs | exec/s | cov | ft | Corpus | `lim` | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logits_processing` | 300 | 957,601 | 3,181 | 2817 | 7720 | 506 | 4096 | 49 MiB |
| `generation` | 300 | 1,143,313 | 3,798 | 2225 | 5574 | 1781 | 995 | 46 MiB |
| `rope_parameters` | 300 | 1,306,002 | 4,338 | 2268 | 7311 | 1853 | 1501 | 48 MiB |
| `chat_templates` | 300 | 184,783 | 613 | 8353 | 32522 | 1956 | 4096 | 75 MiB |
| `configuration` | 120 | 501,339 | 4,143 | 4279 | 9171 | 1204 | 1307 | 48 MiB |

### Round 2: commit `9b028bd` (final source of the package)

Same fuzz package tree; `src/` tree `a3fe014876ad936078317cb363284ce939801753`
(adds the refusal of guarded operands named `not`, found by the crate's
differential proptest rather than by a campaign, and the Verus-checked size
arithmetic). Each target also read its round-1 corpus as a second, read-only
corpus.

| Target | Seconds | Executed inputs | exec/s | cov | ft | Corpus | `lim` | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logits_processing` | 300 | 1,108,870 | 3,683 | 2817 | 7672 | 418 | 4096 | 52 MiB |
| `generation` | 300 | 1,328,077 | 4,412 | 2280 | 5980 | 1803 | 4096 | 47 MiB |
| `rope_parameters` | 300 | 1,532,083 | 5,089 | 2312 | 7692 | 1733 | 4096 | 51 MiB |
| `chat_templates` | 300 | 178,682 | 593 | 8523 | 34216 | 1847 | 4096 | 968 MiB |
| `configuration` | 120 | 611,458 | 5,053 | 4415 | 9866 | 1242 | 2409 | 47 MiB |

`chat_templates` reached a process peak of 968 MiB in round 2 (below the
2048 MiB limit). Re-running its whole retained corpus once with
`-malloc_limit_mb=200` peaked at 199 MiB without a single allocation above
200 MiB, so the peak came from a mutated input that was not retained. The
investigation (round 3 below) found two causes and fixed both.

### Round 3: `chat_templates` follow-ups

Each follow-up commit was followed by a 300 s `chat_templates` campaign that
ran alone (not concurrently) with the round-2 settings, reading the corpora of
the earlier runs as read-only corpora. All exited with status 0 and wrote no
artifact.

| After commit | Change | Executed inputs | exec/s | cov | ft | Corpus | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0eff98f` | `format` reads `%%` and `%(key)` like MiniJinja; `*` picks its repeat candidate like MiniJinja | 194,445 | 645 | 8578 | 35052 | 1689 | 1087 MiB |
| `ae85564` | namespace reference cycles are broken when a render ends | 155,972 | 518 | 8786 | 36286 | 1677 | 1228 MiB |
| `90ea52c` | rendered-prompt floor 16 MiB → 4 MiB | 183,545 | 609 | 8848 | 36801 | 1767 | 1191 MiB |
| `a89cb83` | rendered-prompt floor 4 MiB → 1 MiB (final source; `src/` tree `b98f73b6ce0fb62cf698e45ee55ff60f1c641fc5`, fuzz tree `24687d91e870333ce6b9fc41f452bd59280cdcb1`) | 190,037 | 631 | 8853 | 36970 | 1821 | 673 MiB |

The two causes of the high peaks:

- **Namespace reference cycles leaked across renders** (fixed in `ae85564`).
  Namespaces are MiniJinja's only mutable values, so `{% set ns.me = ns %}`
  kept a namespace and everything it reached alive after the render;
  rendering such a template 20 times with a 10 MB value grew RSS by 200 MiB.
  The renderer now records every namespace and clears its attributes when the
  render ends, as MiniJinja already does for macro closures.
- **Encoding a maximal prompt costs far more than rendering it** (bounded in
  `90ea52c` and `a89cb83`). The harness encodes every rendered prompt, and the
  Hugging Face tokenizer keeps per-token strings, offsets, and masks: a 16 MiB
  prompt of `hello ` took 1.9 GB to encode, and a 4 MiB prompt of `a ` took
  911 MiB while rendering it took 12 MiB. Lowering the floor of the prompt
  limit to 1 MiB (the 4x-context term is unchanged) brought the campaign peak
  to 673 MiB, of which the initial corpus replay accounted for 280 MiB.

The worst-case template that keeps every value it builds alive
(`ns.l + ['x' * 1000000]` in a loop) now stops at the 16 MiB production budget
of an empty context. These are finite smoke campaigns without a sanitizer and
with the limitations stated for the earlier rounds; in particular the
documented residual gaps of the renderer (comparisons of containers that share
a value, template text replayed into captures) are not exercised by the
fragment table.

### Round 4: review fix phase at `4657536`

An independent review of the package found that three fixes were incomplete
(the namespace cleanup itself could be amplified to seconds and gigabytes,
copying builtins were not charged to the render budget, and a LongRoPE factor
of `f32::MIN_POSITIVE` kept frequencies finite but overflowed angles) plus
smaller fidelity issues in the operator guard. After the fixes (commits
`a7e743a` through `4657536`), the `chat_templates` fragment table gained the
review's vectors: loops that keep `list`, `reverse`, slice, and `split` copies
alive, string and sequence slices including `[::-1]`, `*`/`**` call arguments,
a 20 000-namespace flood, a self-referencing namespace, a zero `divisibleby`
divisor, and `debug()`. `rope_parameters` now also checks every accepted
frequency times the largest integer position (`2^63`) for overflow.

`src/` tree `f188cf411833cb82bfd56148f6fdbbf4762c5dcb`, fuzz package tree
`bf1a167145e8852cdf63ad605d9130d18b9085d4`, same toolchain, lockfile, and
flags as rounds 1 and 2, all five targets side by side; each target also read
its latest earlier corpus (round 2, or the round-3 `chat_templates` corpus) as
a second, read-only corpus.

| Target | Seconds | Executed inputs | exec/s | cov | ft | Corpus | `lim` | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logits_processing` | 300 | 1,002,119 | 3,329 | 2818 | 7664 | 425 | 4096 | 48 MiB |
| `generation` | 300 | 1,161,375 | 3,858 | 2289 | 6121 | 1721 | 4096 | 46 MiB |
| `rope_parameters` | 300 | 1,338,447 | 4,446 | 2337 | 8006 | 1641 | 4096 | 48 MiB |
| `chat_templates` | 300 | 63,451 | 210 | 9120 | 36368 | 1600 | 4096 | 110 MiB |
| `configuration` | 120 | 431,908 | 3,569 | 4437 | 10127 | 1186 | 2111 | 47 MiB |

All five exited with status 0 and wrote no crash, timeout, or out-of-memory
artifact. `chat_templates` reached
new coverage (cov 9120 against 8853 in round 3) but ran about three times
slower than in round 3: the new fragments are deliberately expensive (one
namespace flood is 20 000 `namespace()` calls, and each retention loop runs
300 iterations until the budget stops it), and the input's fragments are
re-rendered with every mutation. Its peak RSS was 110 MiB, against 673 MiB in
round 3 with the earlier fragment table. Source changes after `4657536` are
documentation, tests, and these records only.

## 2026-10-16: TOPP-TIE-1 fixed (top-p ranks by logit)

The generation-dispatch fix package changed `filter_top_p`
(`src/generation.rs`) to sort the logits and take the softmax and cumulative
mass over the sorted row, as Hugging Face's `TopPLogitsWarper` does, so a logit
one ulp above a kept one is never dropped. Its follow-up removed the
tolerances the finding needed: `check_warpers` requires a strict top set (no
slack for dropped values), `top_p_ranks_by_logit_like_hugging_face` is an
ordinary test, and `tests/generation_properties.rs` requires an exact mode
(`NEAR_TIE` removed; 200,000 cases of `warpers_keep_a_most_likely_token` pass
in a release build).

One bounded campaign checked the strict harness. Source: `src/` tree
`4de396eda955c33ee432191b46345617b218a109`, `fuzz/src` tree
`3d8ef59041eddbb8a5c342c1cb31ef439400e871`, `fuzz/fuzz_targets` tree
`b3a14771236773708930adc9bedd059a6d96b3ef`, the unchanged lockfile (SHA-256
`f974aac22ee2e71f3115ff6b2e29cc914f1ec5bc17019f47c06743687d6a3637`), and the
toolchain and flags of the wave-3 rounds (`cargo-fuzz 0.13.2`,
`rustc 1.94.0-nightly (4f14395c3 2025-12-22)`, no sanitizer,
`RAYON_NUM_THREADS=1`, `-max_len=4096 -timeout=10 -rss_limit_mb=2048
-print_final_stats=1`). It used a fresh writable corpus and read the committed
seeds and the wave-3 round-2 corpus (505 inputs) as read-only corpora. The host
was shared with other builds (load average about 50), so the rate is a
concurrent rate.

| Target | Seconds | Executed inputs | exec/s | cov | ft | Corpus | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logits_processing` | 301 | 945,649 | 3,141 | 2826 | 7699 | 453 | 50 MiB |

It exited with status 0 and wrote no crash, timeout, or out-of-memory
artifact.

## 2026-10-16: CHAT-4 fixed and Transformers 5.2 template discovery

The tokenization fix package made unnamed renders select a checkpoint's
`tool_use` template whenever `tools` is set (CHAT-4), loaded templates from
`chat_template.jinja` and `additional_chat_templates/*.jinja` before
`tokenizer_config.json` as Transformers 5.2 does, and read template sources
with universal newlines. The `chat_templates` harness gained the two template
file storage forms and an exact loaded-templates oracle, and its CHAT-4
tolerance was removed. Every campaign below used
`cargo +nightly fuzz run -s none chat_templates <scratch corpus>
fuzz/seeds/chat_templates -- -max_len=4096 -timeout=10 -rss_limit_mb=2048
-print_final_stats=1` with `RAYON_NUM_THREADS=1` on macOS (without a
sanitizer, since AddressSanitizer deadlocks there):

| Source | Runs | Seconds | New units | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Working tree before the package was split, with the CHAT-4, template-file, and newline fixes | 10,000 | 14 | 769 | 85 MB |
| The same tree, continuing on the grown corpus | 100,000 | 185 | 1,503 | 99 MB |
| Exact sources of `d0c7f9b`, on the 1,903-file corpus | 50,000 | 125 | 397 | 102 MB |
| Exact sources of `cc5107c` (after the dangling-path skip fix), fresh scratch corpus plus seeds | 30,000 | 34 | 1,242 | 74 MB |

All four exited with status 0 and wrote no crash, timeout, or out-of-memory
artifact.

## 2026-10-16: configuration counts bounded (CFG-COUNT-1)

The configuration-robustness fix package bounded every family's layer and
expert counts by the shared `MAX_LAYERS` (2^18) and `MAX_EXPERTS` (2^20)
before any allocation, and the `configuration` harness stopped filtering
absurd counts and asserts the bounds instead. One bounded campaign on source
`26c16bd`:
`cargo +nightly fuzz run -s none --fuzz-dir fuzz configuration <fresh corpus>
fuzz/seeds/configuration -- -runs=50000 -max_len=4096 -timeout=10
-rss_limit_mb=2048`.

| Target | Runs | Seconds | exec/s | cov | ft | Corpus | Fuzzer RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `configuration` | 50,000 | 9 | 5,555 | 5891 | 12790 | 690 entries, 434 KB | 51 MB |

It found no crash, timeout, or out-of-memory input. An earlier run on
`64ddd4d`, at the former 2^20 layer bound, also had no findings.
