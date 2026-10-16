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
| Assigns a `bool` to `T5BeamSearchOptions::early_stopping` (`generate_t5_beam_search`) | Use `EarlyStopping::Never` to keep 0.1.0's `false` results (search to the length bound), `EarlyStopping::Eager` for `true`, or `EarlyStopping::Heuristic` (the new default; also `false.into()`) for Transformers' meaning of `false`. Serialized `true`, `false`, and `"never"` still deserialize, but a serialized `false` now means the heuristic. |
| Calls `select_lead`, `speculate`, or `speculate_if_active` on `Qwen3_5SpeculativeDecoder` or `NemotronHSpeculativeDecoder` | Pass the new `prompt_length` argument; `0` reproduces the old penalty semantics. |
| Matches `architecture::MixerSpec`, `architecture::FeedForwardSpec`, `architecture::RopeScaling`, `position::RopeScaling`, `QuantizationExclusion`, or `RopeConfigError` exhaustively | Add arms for the new variants or a wildcard arm; `FeedForwardSpec::experts()` treats `MixtureOfExperts` and the new `ClampedMixtureOfExperts` alike. (`Error`, `TensorCacheError`, `CheckpointError`, `QuantizationError`, the family load errors, and the automatic-family enums are `#[non_exhaustive]`.) |
| Matches `CacheConfigError`, `TransactionError`, `DecodeStateError`, `RingPlanError`, `T5BucketError`, `PositionIdError`, or `ConfigLoadError` exhaustively | Add a wildcard arm. None gained a variant, but all seven are now `#[non_exhaustive]`, so later releases can add failure kinds without breaking callers. |
| Copies `architecture::RopeScaling` or builds `RopeScaling::Yarn` literally | Clone it (it is no longer `Copy`); add `truncate: true` (the reference default) to YaRN literals. |
| Reads `time_step_min`/`time_step_max` of `NemotronMamba2Config` | Use `time_step_clamp: TimeStepClamp`. The Nemotron-H Mamba-2 names are now aliases of the shared `mamba2` types. |
| Reads `hidden_size`/`intermediate_size` of `nemotron_h::model::NemotronHDense` | It is an alias of `feed_forward::DenseFeedForward`; read the projection shapes. |
| Names `DeepseekV2MoeRecord`, `NemotronHDenseRecord`, `NemotronMamba2Record`, or `NemotronGroupedGatedRmsNormRecord` (or their `…Item` types) | Use `<DeepseekV2Moe<B> as Module<B>>::Record`, `feed_forward::DenseFeedForwardRecord`, `mamba2::Mamba2Record`, or `mamba2::GatedGroupRmsNormRecord`. |
| Builds `GenerationConfig`, `ChatTemplateOptions`, `SpecialTokens`, `QuantizedCheckpointOptions`, or `RopeParameters` with exhaustive literals | Add `..Default::default()`. |
| Builds `ChatMessage`, `SparseMoeConfig`, `GroupedQueryAttentionConfig`, or `GatedDeltaNetConfig` with exhaustive literals | Start from `ChatMessage::new(role, content)`, `SparseMoeConfig::new`, `GroupedQueryAttentionConfig::new`, or `GatedDeltaNetConfig::new` and assign the fields you change. |
| Builds `MlaSpec`, `MixtureOfExpertsSpec`, `Seq2SeqSpec`, or `DecoderSpec` with exhaustive literals (they have no `Default`) | Add the new fields with their serialized defaults: `MlaSpec` `score_scale: ScoreScale::Auto, connectivity: AttentionConnectivity::default()`; `MixtureOfExpertsSpec` `routing: RoutingPolicy::default(), shared_expert_gate: false, expert_bias: false`; `Seq2SeqSpec` `cross_attention_options: AttentionOptions::default()`; `DecoderSpec` `embedding_scale: None, final_logit_softcap: None`. Serialized JSON is unchanged: absent fields take these defaults, and `None` is not written. |
| Builds `NemotronHConfig` with an exhaustive literal | Add `attention_window: None` or use `..Default::default()`. |
| Reads, sets, or builds `DeltaRuleConfig` with `normalization_epsilon` | Use the `qk_l2_epsilon` field. `DeltaRuleConfig::normalization_epsilon()` is a deprecated accessor, and serialized configs with the old key still load. `GatedDeltaNetConfig::normalization_epsilon` (the output-norm epsilon) is unchanged. |
| Builds `BurnpackArtifactSummary` literally | Set `packed_execution: PackedExecutionSupport` explicitly; loaders normally produce this summary. |
| Builds Burn modules or records literally (`GroupedQueryAttention`, `SparseMoe`, `DeepseekV2Model`, `Qwen3_5TextModel`, or their records) | Initialize from the configuration and load records instead; these gained fields, and `DeepseekV2Model` and `Qwen3_5TextModel` now have private fields. |
| Builds with `#![deny(warnings)]` and calls `LoadedModel::into_parts` (or its variants), `CacheConfig::physical_capacity`, `BurnpackQuantizationManifest::new`, or `BURNPACK_QUANTIZATION_FORMAT_VERSION` | These are deprecated: use `into_named_parts()` (`LoadedModelParts`), `NativeKvCacheSpec::committed_time_slots`, `new_legacy_v1` (or `QuantizedBurnpackSession` for v2), and `BURNPACK_QUANTIZATION_FORMAT_VERSION_V1`/`_CURRENT`. |
| Builds with `#![deny(warnings)]` and calls `MuseGlimmerForCausalLm::load_checkpoint_with_family_options` or `NemotronHForCausalLm::load_checkpoint_with_architecture_options` | Call `load_checkpoint_with` with the same arguments. |

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
| `generate_t5_beam_search` with default options stops with Transformers' heuristic instead of searching to the length bound | Set `early_stopping: EarlyStopping::Never` to reproduce 0.1.0 results. |
| `ModularCausalLm` and `ModularSeq2SeqLm` decode accept a row without history and an entirely inactive step as exact-zero no-ops (0.1.0 returned `Error::BackendCapability` and `Error::Shape`) | Remove handling for those errors; `generate_causal` with `prefill_chunk_size` now works on left-padded batches. |
| A modular decode state rejects a batch that disagrees with its `CacheConfig` as `Error::TensorCache(TensorCacheError::BatchSizeMismatch { .. })`, as every family does (was `Error::Cache`) | Update matches. |
| Top-p ranks candidates by logit, like Transformers' `TopPLogitsWarper` | Seeded samples can differ only where two logits round to the same F32 probability. |
| `AutoCausalLm::load_local` and `AutoSeq2SeqLm::load_local` report discovery and planning failures as `Error::CheckpointPlan(CheckpointError)` and tensor-stream failures as `Error::CheckpointExecution { .. }` (was `Error::Checkpoint(String)` for both); the messages are unchanged | Match the typed variants, for example `Error::CheckpointPlan(CheckpointError::ArtifactNotFound { .. })`; `CheckpointExecution` also carries the partial `LoadReport`. |
| Automatic loading, `T5Config::plan_checkpoint`, and `T5ForConditionalGeneration::load_checkpoint` reject a checkpoint with fewer tensors (plus index aliases) than the configuration needs as `CheckpointError::TooFewTensors` before they build the contract (formerly a strict-validation error after building it). The bound is the depth (for T5, encoder plus decoder blocks), or the exact inventory of a one-to-one contract; T5 returns the error as `T5CheckpointLoadError::Checkpoint`, after any configuration error | No action unless you matched the strict-validation error for such checkpoints; the check is skipped when `LoadOptions::allow_missing` is set. |
| `MuseGlimmerForCausalLm::load_checkpoint` detects the text-only layout instead of assuming multimodal; a checkpoint with both language namespaces fails with `CheckpointError::AmbiguousLanguageNamespace` | Use `load_checkpoint_with(…, MuseGlimmerCheckpointOptions { source })` to force a layout. |
| SafeTensors out-of-section ranges name the first violating tensor in byte-offset order, and byte-length mismatches are reported before layout errors | Update assertions on which tensor an `InvalidDataOffsets` error names. |
| Configurations fail validation (`<family>.<field>: N exceeds the supported maximum of …`, or `the checkpoint would list N tensors …`) for a stack depth above 262,144 (2^18), a routed-expert count above 2^20, or, for Qwen3-MoE, DeepSeek-V2, and Nemotron-H, a checkpoint inventory above 10,000,000 tensors; GPT-2 `n_layer` and GPT-NeoX `num_hidden_layers` were previously accepted up to the `i32` range | Such configurations could not be loaded in practice; `gemma::MAX_LAYERS` changes value from 1,048,576 to 262,144. |
| Nemotron-H `n_routed_experts` above 2^20 reports `nemotron_h.n_routed_experts: N exceeds the supported maximum of 1048576 experts` (was a `topk_group` overflow error), and a tagged `{"__float__": "-Infinity"}` or `"NaN"` anywhere in a config fails with a JSON error naming its path | Update message-text assertions. |
| Nemotron-H ignores the serialized `sliding_window`, as the pinned Transformers graph does; it is no longer validated or applied, and the window error field is `nemotron_h.attention_window` | Set `NemotronHConfig::attention_window` instead to window attention in experiments. |
| For T5 automatic loads, `generation_defaults()` no longer carries `repetition_penalty`, `presence_penalty`, the sample strategy, `temperature`, `top_k`, or `top_p` from `generation_config.json`; all but `presence_penalty` now appear in `unsupported_generation_settings()` | Beam search never applied them; pass explicit settings if you relied on reading them back. |
| `DeltaRuleConfig::validate` names its epsilon `delta_net.rule.qk_l2_epsilon` (was `delta_net.normalization_epsilon`) | Update matches on the error field. |
| An unnamed `render_chat` or `encode_chat` with `tools` set, on a checkpoint that ships both `default` and `tool_use` templates, renders `tool_use` | Set `template_name: Some("default".into())` to keep the old prompt. |
| `LocalTokenizer::from_directory` prefers `chat_template.jinja` and `additional_chat_templates/*.jinja` over the `chat_template` in `tokenizer_config.json`. A template file that counts is rejected if it is empty or whitespace-only, not UTF-8, named `.jinja`, reached through a symlink that leaves the model directory, larger than 16 MiB (or 64 MiB for all template files together), or `additional_chat_templates/default.jinja` beside `chat_template.jinja`; more than 64 `.jinja` entries in `additional_chat_templates` are also rejected. Template paths that are not regular files are skipped, as in Transformers | Remove or fix stale template files; `metadata().chat_templates` shows what was loaded. |
| Template sources with CRLF or CR line endings render with LF, and a newline right after `{% raw %}` is kept, as in Transformers | No action; prompts now match Python. |
| A non-UTF-8 `chat_template.jinja` fails with `Error::Artifact` (was `Error::Io` with kind `InvalidData`). Unknown-name and ambiguous-template errors list the available names: "chat template `x` was not found; the checkpoint provides `a`, `b`" and "checkpoint provides multiple chat templates (`a`, `b`) but no `default`; choose one with `template_name`" (was "... but no default; choose a name"). A malformed `chat_template` in `tokenizer_config.json` is ignored rather than failing when template files are present, and `"chat_template": null` loads | Update error-variant matches and message-text assertions. |

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
| GPT-2 `load_checkpoint`, `Gpt2CheckpointContract::new`/`build_load_plan`, and `Gpt2CheckpointLayout::detect` return `Gpt2CheckpointLoadError`; `Gpt2LoadError` is only the `#[non_exhaustive]` per-tensor sink error | Match `Gpt2CheckpointLoadError::{Configuration, Checkpoint, Namespace, NonFloatParameter, Execution}`; use `execution_failure()` for the partial report. |
| GPT-NeoX `router_logits` and `auxiliary_losses` succeed with empty diagnostics; `attentions: true` is `Error::BackendCapability` (was `Error::Configuration`) | Update matches. |
| `GptNeoXModel` is removed | Use `GptNeoXForCausalLm`. |
| `qwen3_moe::router_logits_point` and `router_weights_point` are deprecated | Use `Layer::router_logits()` and `Layer::router_weights()`, which name the same points for every mixture-of-experts family. |
| Packed Qwen3-MoE and Muse projections are F32 on Flex | Expect `DType::F32` for packed projections unless `packed_layout_is_exact::<B>()` holds. |
| `attend_bounded` with a multi-chunk policy on an autodiff backend evaluates one unbounded call with the same values instead of returning `Error::BackendCapability` | Size training memory for the full score plane; inference backends still chunk. |
| Gemma layout detection is typed: `GemmaCheckpointSource::detect` returns `CheckpointError::LanguageEmbeddingNotFound` or `AmbiguousLanguageEmbedding` (was a `String`), `GemmaForCausalLm::load_checkpoint` reports them as `GemmaCheckpointLoadError::Checkpoint` (the `Namespace` variant is removed), and automatic loading as `Error::CheckpointPlan` (was `Error::Checkpoint(String)`) | Match the typed variants; their `embeddings` field lists the candidate or present embedding names. |
| `Qwen3_5BackboneRecipe::load_pretrained` reports an ambiguous or missing language namespace as `Error::CheckpointPlan(CheckpointError::AmbiguousLanguageNamespace` or `LanguageNamespaceNotFound)` (was a configuration error at `qwen.recipe.source`), and its planning and execution failures are typed like automatic loading's | Update matches; pass `source` explicitly to choose a layout. |

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
