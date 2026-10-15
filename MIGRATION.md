# Migration

This page has two parts: [upgrading code from 0.1.0](#upgrading-from-010) to
the current source, and the [standalone repository
migration](#standalone-repository-migration-september-2026) that extracted the
crate from its original monorepo.

## Upgrading from 0.1.0

The unreleased changes are listed in [CHANGELOG.md](CHANGELOG.md). Most are
additive; the items below can break compilation or change results. The
compilation table is complete for the public API of 0.1.0 (`b5b1c23`): it was
derived by diffing the rustdoc JSON of both versions and compiling a `use` of
every public 0.1.0 path, and each fix below compiles against the current
source. APIs added after 0.1.0 that changed during this cycle are listed
separately under [unreleased sources](#if-you-built-on-unreleased-sources).

### Compilation

| If your code | Change it to |
| --- | --- |
| Assigns a `bool` to `BeamSearchConfig::early_stopping` | Use `EarlyStopping::Never` to keep the old `false` results, `EarlyStopping::Eager` for `true`, or `.into()` for Transformers' meaning of the boolean (`false` is its heuristic default). |
| Calls `select_lead`, `speculate`, or `speculate_if_active` on `Qwen3_5SpeculativeDecoder` or `NemotronHSpeculativeDecoder` | Pass the new `prompt_length` argument; `0` reproduces the old penalty semantics. |
| Matches `architecture::MixerSpec`, `architecture::FeedForwardSpec`, `architecture::RopeScaling`, `position::RopeScaling`, `QuantizationExclusion`, or `RopeConfigError` exhaustively | Add arms for the new variants or a wildcard arm. (`Error`, `TensorCacheError`, `CheckpointError`, the automatic-family enums, and the other error enums were already `#[non_exhaustive]`.) |
| Copies `architecture::RopeScaling` or builds `RopeScaling::Yarn` literally | Clone it (it is no longer `Copy`); add `truncate: true` (the reference default) to YaRN literals. |
| Reads `time_step_min`/`time_step_max` of `NemotronMamba2Config` | Use `time_step_clamp: TimeStepClamp`. The Nemotron-H Mamba-2 names are now aliases of the shared `mamba2` types. |
| Reads `hidden_size`/`intermediate_size` of `nemotron_h::model::NemotronHDense` | It is an alias of `feed_forward::DenseFeedForward`; read the projection shapes. |
| Names `DeepseekV2MoeRecord`, `NemotronHDenseRecord`, `NemotronMamba2Record`, or `NemotronGroupedGatedRmsNormRecord` (or their `…Item` types) | Use `<DeepseekV2Moe<B> as Module<B>>::Record`, `feed_forward::DenseFeedForwardRecord`, `mamba2::Mamba2Record`, or `mamba2::GatedGroupRmsNormRecord`. |
| Builds `GenerationConfig`, `ChatTemplateOptions`, `SpecialTokens`, `QuantizedCheckpointOptions`, or `RopeParameters` with exhaustive literals | Add `..Default::default()`. |
| Builds `ChatMessage`, `SparseMoeConfig`, `GroupedQueryAttentionConfig`, or `GatedDeltaNetConfig` with exhaustive literals | Start from `ChatMessage::new(role, content)`, `SparseMoeConfig::new`, `GroupedQueryAttentionConfig::new`, or `GatedDeltaNetConfig::new` and assign the fields you change. |
| Builds `MlaSpec`, `MixtureOfExpertsSpec`, or `Seq2SeqSpec` with exhaustive literals (they have no `Default`) | Add the new fields with their serialized defaults: `MlaSpec` `score_scale: ScoreScale::Auto, connectivity: AttentionConnectivity::default()`; `MixtureOfExpertsSpec` `routing: RoutingPolicy::default(), shared_expert_gate: false, expert_bias: false`; `Seq2SeqSpec` `cross_attention_options: AttentionOptions::default()`. |
| Builds `BurnpackArtifactSummary` literally | Set `packed_execution: PackedExecutionSupport` explicitly; loaders normally produce this summary. |
| Builds Burn modules or records literally (`GroupedQueryAttention`, `SparseMoe`, `DeepseekV2Model`, `Qwen3_5TextModel`, or their records) | Initialize from the configuration and load records instead; these gained fields, and `DeepseekV2Model` and `Qwen3_5TextModel` now have private fields. |
| Builds with `#![deny(warnings)]` and calls `LoadedModel::into_parts` (or its variants), `CacheConfig::physical_capacity`, `BurnpackQuantizationManifest::new`, or `BURNPACK_QUANTIZATION_FORMAT_VERSION` | These are deprecated: use `into_named_parts()` (`LoadedModelParts`), `NativeKvCacheSpec::committed_time_slots`, `new_legacy_v1` (or `QuantizedBurnpackSession` for v2), and `BURNPACK_QUANTIZATION_FORMAT_VERSION_V1`/`_CURRENT`. |

### Results and errors

| What changed | What to do |
| --- | --- |
| Presence and frequency penalties count only generated tokens | Set `additive_penalty_scope: PenaltyScope::PromptAndGenerated` to keep counting the prompt. |
| Greedy decoding ignores temperature, top-k, top-p, and min-p; its log probabilities come from the untruncated distribution | Compare greedy log probabilities against Transformers' `normalize_logits=True`. |
| F16/BF16 logits are processed in F32 and selected log probabilities are F32 | Expect `DType::F32` log probabilities from half-precision models. |
| Top-k keeps every token tied with the k-th score | Seeded samples can differ only where boundary logits tie. |
| `generation_config.json` sampling defaults, `min_p`, `no_repeat_ngram_size`, and `min_new_tokens` are applied | Inspect `loaded.generation_defaults()` and `loaded.unsupported_generation_settings()`, or pass `AutoLoadOptions::generation` explicitly. |
| Beam search needs capacity `prompt + max_new_tokens - 1` | Larger caches still work. |
| T5 decoding device mismatches are `Error::BackendCapability`; DeepSeek stateful batch and capacity errors are `Error::TensorCache` | Update matches; when several errors exist, layout and device errors are reported before value errors, and a T5 source error before a target error. |
| Error texts are unified (`token values in 0..V`, `position values in 0..N`, `{tensor} and {owner} are on different devices`; automatic config errors name the family's field) | Update assertions on message text. |
| `EncodeOptions::max_length` truncates content before special tokens, and serialized truncation/padding are ignored | Request truncation and padding explicitly. |
| Auto discovery with both `model.safetensors` and an index that does not name exactly that file reports ambiguity | Select `ArtifactSelection::Index` or `Single`. |
| Qwen3-MoE sliding windows cover every block | Only configs with `use_sliding_window: true` are affected; released Qwen3-30B-A3B is not. |
| DeepSeek and Muse reject `partial_rotary_factor` other than one | Such configs were silently computed incorrectly before. |
| DeepSeek's `seq_aux = false` balance loss is `1/top_k` of its old value; Nemotron-H's balance loss is Megatron's normalized form | Retune loss coefficients that were tuned against the old values. |
| T5 configs with `relative_attention_max_distance <= buckets / 2` fail validation | Full models need a larger maximum distance (encoder-only recipes still accept `(n/4, n/2]`). |
| Spec validation delegates to the shared components | Now accepted: bidirectional T5 `max_distance > buckets / 4`, YaRN factor 1, any nonzero cross-attention width. Now rejected: odd or small bidirectional T5 bucket counts, MLA `rotary_dim > rotary_head_dim` or an odd rotary sub-head, YaRN `theta <= 1`, projection-width overflow, and sliding windows in encoder stacks. |
| Modular T5 biases start from `N(0, 1/hidden_size)` | Seeded random models differ from earlier versions; loaded checkpoints are unaffected. |
| A failed cached call leaves lengths and readable slots unchanged but may write bytes beyond the lengths | Compare readable prefixes, not raw cache buffers. |

### If you built on unreleased sources

These APIs did not exist in 0.1.0 but changed after they first appeared on the
integration branch. Code written against an intermediate v0.2/v0.3 snapshot may
need these changes too.

| What changed | What to do |
| --- | --- |
| `ParameterInit` gained `Ones` and `Constant { value }` | Add arms to exhaustive matches. |
| `ModularLayer::init` takes `Option<&CrossAttentionSpec>` | Pass `Some(&CrossAttentionSpec::new(heads, head_dim))` instead of `Some((heads, head_dim))`. |
| `MaskGeometry` gained a key-start field | Build it with `MaskGeometry::new(…)` and `with_key_start`. |
| GPT-NeoX `hidden_states` holds `L + 2` entries | Crate index `L + 1` is Transformers' index `L`. |
| Runtime hook failures are `Error::Intervention` | Match `Error::Intervention { point, reason, cause }`; registration mistakes remain `Error::Configuration`. |
| GPT-2/NeoX token, position, mask, and logits-selection errors are `Error::Shape` and their device mismatches `Error::BackendCapability`; modular staged position errors are `Error::Shape` | Update matches. |
| GPT-2 random initialization matches GPT-2 | Seeded random GPT-2 models differ; loaded checkpoints are unaffected. |
| `init_policy()` keeps LayerNorm parameters | Add `ParameterInit::Ones`/`Zeros` overrides to reset them. |
| Specs reject attention sinks combined with a soft cap | Use one of the two per attention block. |
| Fan-out transfer copies are independent parameters | Tie them explicitly in one owner if they must stay shared. |
| `AutoCausalConfig::model_type()` returns the primary type of an alias family (`llama` for `qwen3`) | Read `LoadedModel::source().model_type()` for the checkpoint's own type. |
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
