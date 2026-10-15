# Migration

This page has two parts: [upgrading code from 0.1.0](#upgrading-from-010) to
the current source, and the [standalone repository
migration](#standalone-repository-migration-september-2026) that extracted the
crate from its original monorepo.

## Upgrading from 0.1.0

The unreleased changes are listed in [CHANGELOG.md](CHANGELOG.md). Most are
additive; the items below can break compilation or change results.

### Compilation

| If your code | Change it to |
| --- | --- |
| Assigns a `bool` to `BeamSearchConfig::early_stopping` | Use `EarlyStopping::Never` to keep the old `false` results, `EarlyStopping::Eager` for `true`, or `.into()` for Transformers' meaning of the boolean (`false` is its heuristic default). |
| Calls `Qwen3_5SpeculativeDecoder` or the Nemotron-H speculative decoder (`select_lead`, `speculate`, `speculate_if_active`) | Pass the new `prompt_length` argument; `0` reproduces the old penalty semantics. |
| Matches `ParameterInit` exhaustively | Add arms for `Ones` and `Constant { value }`. |
| Copies `architecture::RopeScaling` or builds `RopeScaling::Yarn` literally | Clone it; add `truncate: true` (the reference default) to YaRN literals. |
| Calls `ModularLayer::init(…, Some((heads, head_dim)), …)` | Pass `Some(&CrossAttentionSpec::new(heads, head_dim))`. |
| Reads `time_step_min`/`time_step_max` of `NemotronMamba2Config` | Use `time_step_clamp: TimeStepClamp`. |
| Reads fields of `NemotronHDense` | It is now an alias of `feed_forward::DenseFeedForward`; read the projection shapes. |
| Builds public configuration structs with exhaustive literals (`GenerationConfig`, `SparseMoeConfig`, `GroupedQueryAttentionConfig`, `MaskGeometry`, `ChatMessage`, `ChatTemplateOptions`, `SpecialTokens`, `GatedDeltaNetConfig`, `MlaSpec`, `MixtureOfExpertsSpec`, `Seq2SeqSpec`, `Gpt2Config`, `QuantizedCheckpointOptions`, `BurnpackArtifactSummary`) | Add `..Default::default()` (or the constructor) so new fields take their defaults. |
| Uses `LoadedModel::into_parts` or its variants | Use `into_named_parts()` (`LoadedModelParts`); the old accessors are deprecated. |
| Uses `CacheConfig::physical_capacity` | Use `NativeKvCacheSpec::committed_time_slots`. |
| Calls trait methods on a `LoadedModel` | Import `architecture::HookedModel` / `StagedModel`; inherent methods of the same name still take precedence on concrete types. |

### Results and errors

| What changed | What to do |
| --- | --- |
| Presence and frequency penalties count only generated tokens | Set `additive_penalty_scope: PenaltyScope::PromptAndGenerated` to keep counting the prompt. |
| Greedy decoding ignores temperature, top-k, top-p, and min-p; its log probabilities come from the untruncated distribution | Compare greedy log probabilities against Transformers' `normalize_logits=True`. |
| F16/BF16 logits are processed in F32 and selected log probabilities are F32 | Expect `DType::F32` log probabilities from half-precision models. |
| Top-k keeps every token tied with the k-th score | Seeded samples can differ only where boundary logits tie. |
| `generation_config.json` sampling defaults, `min_p`, `no_repeat_ngram_size`, and `min_new_tokens` are applied | Inspect `loaded.generation_defaults()` and `loaded.unsupported_generation_settings()`, or pass `AutoLoadOptions::generation` explicitly. |
| Beam search needs capacity `prompt + max_new_tokens - 1` | Larger caches still work. |
| GPT-NeoX `hidden_states` holds `L + 2` entries | Crate index `L + 1` is Transformers' index `L`. |
| Runtime hook failures are `Error::Intervention` | Match `Error::Intervention { point, reason, cause }`; registration mistakes remain `Error::Configuration`. |
| GPT-2/NeoX token, position, mask, and logits-selection errors are `Error::Shape`; device mismatches there and in T5 decoding are `Error::BackendCapability`; DeepSeek stateful errors are `Error::TensorCache`; modular staged position errors are `Error::Shape` | Update matches; when several errors exist, layout and device errors are reported before value errors, and a T5 source error before a target error. |
| Error texts are unified (`token values in 0..V`, `position values in 0..N`, `{tensor} and {owner} are on different devices`; automatic config errors name the family's field) | Update assertions on message text. |
| `EncodeOptions::max_length` truncates content before special tokens, and serialized truncation/padding are ignored | Request truncation and padding explicitly. |
| Auto discovery with both `model.safetensors` and an index that does not name exactly that file reports ambiguity | Select `ArtifactSelection::Index` or `Single`. |
| Qwen3-MoE sliding windows cover every block | Only configs with `use_sliding_window: true` are affected; released Qwen3-30B-A3B is not. |
| DeepSeek and Muse reject `partial_rotary_factor` other than one | Such configs were silently computed incorrectly before. |
| DeepSeek's `seq_aux = false` balance loss is `1/top_k` of its old value; Nemotron-H's balance loss is Megatron's normalized form | Retune loss coefficients that were tuned against the old values. |
| T5 configs with `relative_attention_max_distance <= buckets / 2` fail validation | Full models need a larger maximum distance (encoder-only recipes still accept `(n/4, n/2]`). |
| Spec validation delegates to the shared components | Now accepted: bidirectional T5 `max_distance > buckets / 4`, YaRN factor 1, any nonzero cross-attention width. Now rejected: odd or small bidirectional T5 bucket counts, MLA `rotary_dim > rotary_head_dim` or an odd rotary sub-head, YaRN `theta <= 1`, projection-width overflow, sinks combined with a soft cap, and sliding windows in encoder stacks. |
| GPT-2 random initialization matches GPT-2; modular T5 biases start from `N(0, 1/hidden_size)` | Seeded random models differ from earlier versions; loaded checkpoints are unaffected. |
| `init_policy()` keeps LayerNorm parameters | Add `ParameterInit::Ones`/`Zeros` overrides to reset them. |
| Fan-out transfer copies are independent parameters | Tie them explicitly in one owner if they must stay shared. |
| `AutoCausalConfig::model_type()` returns the primary type of an alias family (`llama` for `qwen3`) | Read `LoadedModel::source().model_type()` for the checkpoint's own type. |
| A failed cached call leaves lengths and readable slots unchanged but may write bytes beyond the lengths | Compare readable prefixes, not raw cache buffers. |
| Cache steps and read plans are bound to the state and lifecycle version that created them | Create a step after any reset, reorder, truncate, or transaction call, and build one read plan per group per step. |

## Standalone repository migration (September 2026)

On September 10, 2026, `crates/transformers` was extracted from the local
`hmm_compression` repository into `/Users/nover/repos/transformers`. The parent
checkout removes the crate entirely: no submodule, path dependency, or
compatibility symlink remains. The integration branch stays `codex/transformers`.
No remote, registry publication, or release tag is created by this migration.

### Preserved history and artifacts

The extraction filters every original local branch to `crates/transformers/`
and moves that subtree to the new repository root. Commit IDs necessarily
change because trees and parent IDs change. Authors, committers, dates, messages,
and each retained historical crate tree are preserved. Commits unrelated to
the crate may be pruned. The original HMM history is not rewritten; removal
is a normal new commit in that repository. All 431 source commits were retained,
including 16 reflog-only commits protected under
`refs/archive/transformers/<original-commit>`. The 100 local branches and stash
were extracted; 17 app-specific tree snapshots stay only in the original
repository and backup. The complete map verifies parent order as well as trees
and commit metadata. Old auxiliary HMM worktrees remain untouched; create new
worktrees from this standalone repository for future transformer development.

The `migration/` directory records the extraction tool/version, source cut,
verification results, and original-to-extracted commit/ref maps. The local
`.git/migration/hmm-compression-original.bundle` additionally preserves the
unfiltered original branches for reproducing Git-bound historical tools. That
backup is not tracked, packaged, or published. It contains the original
monorepo history, so review it before sharing. The extracted repository does
not depend on the original checkout, this bundle, or shared Git object storage
for ordinary development. `history-verification.json` describes the extracted
cut before the subsequent migration-only documentation commit; its paths are
the locations used during verification. The checked-in verifier can audit a
fresh extraction against an original source mirror at that cut.

The original working directory is moved intact. Its ignored `models` symlink
still points to `/Volumes/bag of holding/models`; downloaded weights are not
copied or removed. Its ignored `target/` build tree and external cache links
are retained. Cargo may rebuild path-sensitive artifacts at the new location.
Unrelated HMM edits remain in the original checkout.

### Current development commands

From the standalone repository root:

```sh
cargo fmt --manifest-path Cargo.toml --check
cargo test --manifest-path Cargo.toml --locked --offline
python3 scripts/check_release.py --gate source --offline
```

The manifest, registry dependencies, nested fuzz package, and coverage helper
package are already standalone. Consumer path dependencies must point at this
checkout, for example `../transformers` from a sibling project. HMM itself has
no dependency on this crate.

### Historical reproduction

Acceptance JSON, producer hashes, embedded reference fixtures, and previously
qualified example sources remain byte-identical. Old absolute paths and commit
IDs in those records describe the original execution, not the current checkout.
The commit map translates historical source identity; it does not retroactively
qualify new executions or rewrite authenticated evidence. Some source comments
also retain historical commands; use README commands for current examples.

These five frozen producers derive the monorepo root from their original
directory depth and **must not be run directly from the flattened checkout**:

| Producer | Original producer commit |
| --- | --- |
| `scripts/generate_qwen35_08b_reference.py` | `06f39e77758a6eaf94b1583de60c531b94de872c` |
| `scripts/generate_qwen3_moe_reference.py` | Use the original commit recorded in the specific authenticated reference bundle. |
| `scripts/generate_qwen38_reference.py` | `b7aab93d8825f0b0458265fa11f5a2d68cb226bc` |
| `scripts/generate_flan_t5_small_reference.py` | `2dcf44c283dbc48377249b79c82f5483b912b2a7` |
| `scripts/generate_flan_t5_xxl_reference.py` | `163c9495b4a2f378edc1cd436cb3717aece81463` |

A wrong ancestor may itself be a Git repository and silently supply unrelated
provenance. These producers intentionally remain unchanged because their exact
bytes authenticate existing official-reference bundles. Ordinary tests and
reference consumers remain usable in the standalone checkout. Muse's Rust
acceptance harness now resolves either layout explicitly and still rejects
dirty or stale source.

To reproduce an old bundle, clone the unfiltered backup into a separate,
previously nonexistent directory and check out the original producer commit:

```sh
git clone /Users/nover/repos/transformers/.git/migration/hmm-compression-original.bundle \
  /private/tmp/transformers-original-layout
git -C /private/tmp/transformers-original-layout fetch \
  /Users/nover/repos/transformers/.git/migration/hmm-compression-original.bundle \
  'refs/archive/transformers/*:refs/archive/transformers/*'
git -C /private/tmp/transformers-original-layout switch --detach \
  06f39e77758a6eaf94b1583de60c531b94de872c
```

The example selects the original Qwen0.8B producer. For another bundle, select
its own authenticated original commit. Run its unchanged producer under
`crates/transformers/scripts/` from that original-layout checkout after
verifying HEAD, clean source, pinned source/helper hashes, Python package/wheel
identities, and checkpoint revision. Use a fresh external output directory;
never overwrite existing reference bundles. A source archive without Git, a
symlink to the flattened scripts, or substituted rewritten commit IDs does not
preserve this provenance contract.
