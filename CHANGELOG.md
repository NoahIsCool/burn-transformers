# Changelog

All notable changes to this crate are recorded here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic
versioning, with the additional pre-1.0 compatibility policy documented in
[`SUPPORT.md`](SUPPORT.md).

## [Unreleased]

Upgrade notes for every change below that can break or alter existing code
are collected in [MIGRATION.md](MIGRATION.md#upgrading-from-010).

### Added

#### Model families

- The Llama family (`models::llama`), one dense decoder for six Hugging Face
  model types: `llama` (Llama 2/3.x, SmolLM), `mistral`, `qwen2`, `qwen3`,
  `smollm3`, and `olmo2`, with every RoPE rule, Q/K/V-only bias, per-head or
  full-width Q/K norm, post-sublayer norms, per-layer NoPE, and sliding
  windows with window-sized ring caches. Example: `llama_generate`.
- Gemma 2 and Gemma 3 text (`models::gemma`), including the text tower of
  multimodal `gemma3` checkpoints: sandwich `(1 + w)` RMSNorm, `sqrt(d)`
  embedding scale, attention and final soft caps, local/global schedules with
  ring caches, and per-layer-type RoPE. Example: `gemma_generate`.
- gpt-oss (`models::gpt_oss`): learned attention sinks, alternating sliding and
  full attention, YaRN, a biased top-k-then-softmax router, clamped SwiGLU
  experts, router hooks, and `gpt_oss_load_balancing_loss`.
- All three are registered in `AutoCausalLm`; the Llama family and Gemma
  share one dense-decoder engine. Experimental validation: tiny Transformers 5.2.0 goldens for every
  type and real-weight gates (`tests/llama_family_artifacts.rs`,
  `tests/gemma_artifacts.rs`, `tests/gpt_oss_artifacts.rs`) that passed on
  Flex and Metal in F32 at `f4c56c4` (recorded in
  `acceptance/measurements/real-checkpoint-replay-20261016.json`) for SmolLM-135M, Qwen3-4B, SmolLM3-3B,
  Llama-3.1-8B, Qwen2-7B (against an F64 reference; on Metal under the
  documented `2.5e-3` tolerance), gemma-3-4b-it, and
  truncated gpt-oss-20b; see the
  [model status page](ARCHITECTURE_COMPATIBILITY.md). No new checkpoint is
  advertised as supported.
- DeepSeek-V2 runs V3-style sigmoid `noaux_tc` routing with a correction bias
  and V2 group-limited routing. `GptNeoXForCausalLm` is the one GPT-NeoX
  model name.
- `NemotronHConfig::attention_window` (never serialized) opts Nemotron-H's
  attention into a sliding window for experiments; the pinned graph has none.

#### Experiment interface

- `architecture::HookedModel` and `StagedModel`, both implemented by every
  causal family; T5 and modular seq2seq models implement `HookedModel`, with
  `StagedModel` on `T5Encoder` and the modular encoder/decoder stage views.
  `AutoCausalLm` implements both, `AutoSeq2SeqLm` implements `HookedModel`,
  and `LoadedModel` forwards them, so one generic experiment runs on any model
  or loaded checkpoint. `AutoCausalConfig::supports_experiments()` reports
  support.
- New activation catalogs: DeepSeek MLA and router points, Muse gate and
  sandwich-norm points, Nemotron-H points per block kind, Qwen3-MoE router
  points and a headless backbone, gpt-oss router points, and staged execution
  for all of them. Every mixture-of-experts family addresses its router points
  through the shared `Layer::router_logits`, `router_weights`,
  `routed_output`, and `shared_output` constructors, whose expert axes are
  `Axis::EXPERT` and `Axis::EXPERT_SLOT`.
- `Hooks::capture_dynamic`/`get_dynamic` with `DynamicActivation`,
  `Point::<D>::try_from(PointInfo)`, `PointInfo::rank`, `Layer::new`,
  `Layer::mixer_input`/`mixer_output`, and `Stack::layer`.
- `Error::Intervention { point, reason, cause }` for failures while a hooked
  model runs; registration mistakes stay `Error::Configuration`.
- The derived read-only `AttentionPoints::sink_probability` point and
  `Hooks::apply_sink_probability` for attention with learned sinks.
- `ExecutionContext::contiguous`, `with_encoder`, and `validate_layout`;
  `experiment::RotaryAttention` for RoPE in composed residual blocks.
- `with_attention_precision` on the Llama family, Gemma, gpt-oss, GPT-2, and
  GPT-NeoX (plus `set_attention_precision` on GPT-2 and GPT-NeoX).
- `initialization`: `InitPolicy` with parameter roles, depth-scaled residual
  projections, path overrides, `ParameterInit::{Keep, Normal, Zeros, Ones,
  Constant}`, `validate_for`, `apply`, and lazy `apply_lazy`;
  `ModularCausalLm::init_with` and `scaled_init_policy` (and the seq2seq
  equivalents), also available without a backend type parameter as
  `DecoderSpec::scaled_init_policy` and `Seq2SeqSpec::scaled_init_policy`;
  `Gpt2Config::init_policy` and `GptNeoXConfig::init_policy`, and
  `Gpt2Config::reinit_policy` and `GptNeoXConfig::reinit_policy`, complete
  Hugging Face re-initializations that also reset every LayerNorm.
- Modular specifications: `MixerSpec::AttentionWithOptions`
  (`AttentionOptions`: global, sliding, or chunked-local connectivity, score
  scale, soft cap, sinks, per-projection bias, Q/K-norm scope, SiLU gate),
  `MixerSpec::DifferentialAttention`, executable `MixerSpec::Mamba2`,
  `FeedForwardSpec::{Ungated, ClampedSwiGlu, None}`, MLA score scale and
  connectivity, MoE routing policy, shared-expert gate, and expert biases,
  `Seq2SeqSpec::cross_attention_options`, every RoPE scaling rule, and
  `encoder_stages`/`decoder_stages` views of `ModularSeq2SeqLm`.
  `DecoderSpec::{embedding_scale, final_logit_softcap}` add Gemma's
  dtype-rounded `sqrt(d)` embedding multiplier and the `c·tanh(z/c)` logit cap
  (an encoder stack rejects a cap), and
  `FeedForwardSpec::ClampedMixtureOfExperts { experts, limit, alpha, up_offset }`
  adds gpt-oss's clamped SwiGLU routed and shared experts, with
  `FeedForwardSpec::{experts, expert_glu}`; modular parity tests reproduce
  Gemma 2 and gpt-oss block for block.

#### Shared mechanisms

- Attention: `grouped_query_attention` over native KV heads,
  `AttentionConnectivity` (global, sliding-window, chunked-local) with
  `MaskGeometry::with_key_start`, learned sinks, `ProjectionBias`,
  `QueryKeyNormScope::FullWidth`, `attend_bounded` (score-bounded chunked
  execution) with `AttentionExecutionPolicy::effective_on`, and
  `DifferentialAttention`.
- Feed-forward: `forward_with_hooks` and `mlp_hook_points`, `ClampedSwiGlu`
  and `forward_clamped`; `normalization::scale_free_rms_norm`.
- `moe::router`: `RouterConfig`/`RoutingPolicy` with softmax or sigmoid
  scoring, top-k before or after scoring, greedy or group-limited selection,
  correction bias, renormalization, routed scale, and a shared-expert gate.
- Position: Llama 3, LongRoPE, and proportional RoPE scaling, YaRN `truncate`,
  per-forward rotary tables (`RotaryFrequencies`, `apply_rotary_to`), the
  O(q + k) `T5RelativeBucketTable`, and one Hugging Face RoPE translator
  (`RopeParameters::to_rope_config`, `RopeParameterSet`).
- `mamba2`: a model-independent Mamba-2 mixer with a vectorized chunk scan,
  batched ragged rows, per-token snapshots, and `TimeStepClamp`.
- Cache: batched multi-row KV I/O (`KvBatchAppend`, `NativeKvCache::write_batch`,
  `plan_batch_read`/`read_batch`, `KvBatchSegment::exclusion_mask` and
  `masked_values`), atomic step write sets (`TensorDecodeState::step`,
  `TensorStepWrites::push_*`, `publish_step`), zero-copy write-ahead
  (`write_kv_ahead`, `read_kv_through`), in-place speculative staging,
  single-plane latent groups for MLA, per-group sliding windows, and
  `DecodeState::reorder_within_groups`.
- Generation: `min_new_tokens`, `min_p`, `frequency_penalty`, `logit_bias`,
  `no_repeat_ngram_size`, and `PenaltyScope`; `TokenHistory` with
  `LogitsProcessor::process_with_history` and `process_with_prompt_lengths`;
  `upcast_half_precision`; `generation::speculative` (`SpeculativeVerifier`,
  `speculative_commit`, `prompt_lookup_draft`); `EarlyStopping`; Nemotron-H
  `prefill_speculative`.
- Tokenization: `TokenStreamDecoder` (`LocalTokenizer::decode_stream`);
  chat-template `continue_final_message`, documents, `tool_call_id`, and extra
  message fields; `SpecialTokens` `sep`, `cls`, and `mask`.
- Loading: `LoadedModel::into_named_parts` (`LoadedModelParts`) and
  `unsupported_generation_settings()`; `generation_config.json` `min_p`,
  `no_repeat_ngram_size`, and `min_new_tokens` are applied. Every family
  model loads a local checkpoint directly with `load_checkpoint`, including
  the new `Qwen3_5ForCausalLm::load_checkpoint`, and every family with layout
  or provenance options (Qwen3.5, Llama, Gemma, gpt-oss, DeepSeek-V2,
  Qwen3-MoE, Muse, Nemotron-H) also with `load_checkpoint_with` and its
  `…CheckpointOptions`; the `checkpoint` module section "Loading a concrete
  family" lists them, and the whole-load errors `Gpt2CheckpointLoadError` and
  `Qwen3_5CheckpointLoadError` are exported.
  `CausalLanguageModel::new_decode_state_for` and
  `Seq2SeqLanguageModel::new_decode_state_for` size a decode state from its
  `CacheConfig` alone.
- Errors: `Error::CacheConfig` and `Error::Rope`, so `?` works on
  `CacheConfig::new` and `RopeConfig::new`; `Error::CheckpointPlan`, with a
  `From<CheckpointError>` conversion, for discovery and planning failures, and
  `Error::CheckpointExecution`, which keeps a failed load's target, partial
  `LoadReport`, and `ExecutionStats`; `CheckpointError::LanguageNamespaceNotFound`
  and `AmbiguousLanguageNamespace` (Qwen3.5 and Muse layout detection),
  `LanguageEmbeddingNotFound` and `AmbiguousLanguageEmbedding` (Gemma's
  three-layout detection, which lists the candidate or present embeddings),
  and `TooFewTensors`.
- Configuration bounds: `models::config::MAX_LAYERS` (2^18 = 262,144, the
  largest power of two at which every dense checkpoint contract fits the
  artifact limit), `MAX_EXPERTS` (2^20), and `MAX_CHECKPOINT_TENSORS`
  (10,000,000).
- Numeric-range hazards: `checkpoint::RangeHazard`, a dtype a family declares
  its checkpoints are measured to overflow, with the evidence and whether the
  family certifies a packed path for them (`RangeHazard::with_certified_packed_path`,
  `packed_path_certified`; Gemma 3 declares F16 without one,
  `GemmaVariant::range_hazards` and `GemmaCheckpointContract::range_hazards`);
  `MappingPlan::declare_range_hazards` and `MappingPlan::range_hazards`, which
  carry a family's declarations to the shared planner (serialization omits
  them, so the plan format is unchanged); the generic `check_range_hazards`
  over a strict `LoadPlan`, which `build_load_plan` runs for the declared
  hazards; `CheckpointError::RangeHazard`; and the load option
  `LoadOptions::range_hazards` (`RangeHazardPolicy::Refuse` by default,
  `Accept` as the explicit opt-in).
- Checkpoints: exact MXFP4 decoding (`MappingPlan::dequantize_mxfp4`,
  `TensorOperation::DequantizeMxfp4`), `install_packed_parameter`
  (`PackedInstall`, `packed_layout_is_exact`), `CheckpointError::InvalidTensorHeader`
  and `InvalidTensorName`, `QuantizedCheckpointOptions::unaligned`, and
  `BurnpackArtifactSummary::packed_execution`.
- Tooling: `scripts/generate_causal_lm_reference.py` (local Transformers
  references for any causal checkpoint, with
  `--dtype {float32,float64,bfloat16,float16}` and `--num-hidden-layers`; a
  `float64` reference measures this crate's error alone when an F32
  reference's own rounding is as large as the tolerance, as for Qwen2-7B;
  `--device {cpu,mps}` and `--attn-implementation {eager,sdpa}` select the
  official graph, `--decode-tokens-from` records cached and stateless decode
  rows at a truth's greedy positions, `--prompt-decode` the official
  quarter-prefill decode of the longest prompt, and non-finite results are
  recorded as data), `scripts/derive_half_precision_envelope.py` (the
  official-graph envelope and its leave-one-out slacks, from the bundles
  alone), and `scripts/generate_tiny_goldens.py`;
  `scripts/generate_diffllama_reference.py` and
  `scripts/generate_tokenization_goldens.py` regenerate and check the pinned
  DiffLlama and tokenizer and chat-template goldens.

#### Earlier in this cycle (v0.2 and v0.3)

- A v0.2 intervention interface: typed computation points, token/head/
  channel selections, scoped capture and observation, ordered edits and patches,
  shape/dtype/device validation, and explicit capture gradient policy.
- Stateless activation hooks and discoverable catalogs for Qwen3.5/3.8, T5,
  custom causal/seq2seq models, and shared grouped-query attention. Score and
  pattern access requires explicit eager attention; pattern edits preserve
  causal and padding exclusions.
- Explicit hidden-state embed/run/normalize stages for custom, T5 encoder, and
  Qwen backbones; separate self/cross-attention policies with per-layer overrides,
  custom connectivity and bias; padded T5 decoder hooks with original token indices.
- Editable T5 encoder and Qwen backbone recipes, selective pretrained prefix
  loading, and transactional parameter transfer with shape and tie policies.
- Open residual blocks and registered width adapters for user-defined sequence
  mixers; reusable MLA projection/attention with compressed-state interventions.
- Architecture examples for a heterogeneous T5 classifier, Qwen prefix,
  custom sequence mixer, and MLA encoder-decoder, with native Burn training
  paths for the classifier, composition, MLA, and frozen Qwen adapter.
- A consumer activation guide and download-free workbench demonstrating a
  registered Burn adapter. Grokking's checkpoint-level Fourier reproduction
  now uses the public intervention interface.
- Clock and Pizza's released-model Table 2 and distance measurements, with
  independent reference parity, embedding-circle figures, intervention controls,
  pinned artifacts, and explicit unresolved published accuracy discrepancies.
- A bounded Tuned Lens study with the exact released Pythia/lens pair, registered
  auxiliary translators, layerwise metrics, and a prediction table; independent
  metric references and documented scalar-reduction corrections validate analysis.
- A bounded Greater-Than study with pinned GPT-2 weights and aligned prompts,
  clean/corrupt MLP patches, ablation/restoration controls, separate direct-path
  measurements, and independent full-vocabulary numerical comparisons.
- Experimental stateless GPT-NeoX and GPT-2 modules with strict local imports,
  public activation boundaries, and independent full-weight numerical evidence.
- v0.3 training compatibility: native Burn optimizer loops and an optional
  `train` feature exposing Burn's own Learner, with a runnable supervised example.
- Differentiable stable cross-entropy, log-softmax, and teacher-to-student KL
  losses, plus an optional duplicate-parameter-owner preflight.
- Actual Grokking and Clock/Pizza training, new Tuned Lens translators, and a
  separately labeled Greater-Than distillation extension; fixed protocols,
  learning curves, checkpoint controls, and bounded reproduction claims.
- Training documentation covering connected activation losses, frozen suffixes,
  initialization, dropout, native records, and packed quantization limits.

### Changed

These change observable behavior; see [MIGRATION.md](MIGRATION.md#upgrading-from-010).
Items about APIs added during this cycle (GPT-2, GPT-NeoX, hooks, staged
execution, `ModularLayer`, `InitPolicy`, transfer, and cache steps) affect only
code written against unreleased sources.

- **Package name.** The Cargo package is `burn-transformers` (was
  `transformers`, a crates.io name that belongs to another publisher), after
  the GitHub repository. `[lib] name = "transformers"` keeps every
  `use transformers::…` path, so the library, its doctests, examples, and
  integration tests are unchanged. Dependents change the dependency key
  ([MIGRATION.md](MIGRATION.md#dependency)) to
  `burn-transformers = { git = "https://github.com/NoahIsCool/burn-transformers", … }`
  (or keep it with `transformers = { package = "burn-transformers", … }`, as
  `fuzz/Cargo.toml` now does). An old path dependency fails to resolve, but an
  old git dependency still resolves, to a historical acceptance snapshot named
  `transformers` or to another checkout that an acceptance harness's path
  dependency reaches, so it fails with an unrelated error or builds stale
  source; change it too. `cargo -p` selectors name `burn-transformers`, and
  `cargo package` writes `burn-transformers-0.1.0.crate`. Nothing is
  published.
- **Public API.** Exhaustive matches on `MixerSpec`, `FeedForwardSpec`, both
  `RopeScaling` enums, `QuantizationExclusion`, and `RopeConfigError` need arms
  for their new variants. `GenerationConfig`, `ChatMessage`,
  `ChatTemplateOptions`, `SpecialTokens`, `SparseMoeConfig`,
  `GroupedQueryAttentionConfig`, `GatedDeltaNetConfig`, `MlaSpec`,
  `MixtureOfExpertsSpec`, `Seq2SeqSpec`, `DecoderSpec`, `NemotronHConfig`,
  `RopeParameters`, `QuantizedCheckpointOptions`, and
  `BurnpackArtifactSummary` gained public fields, so exhaustive literals need
  them, and `FeedForwardSpec` gained `ClampedMixtureOfExperts`. The Burn
  records of `DeepseekV2Moe` and of Nemotron-H's dense, Mamba-2, and
  gated-norm modules are now the shared modules' record types.
  `CacheConfigError`, `TransactionError`, `DecodeStateError`,
  `RingPlanError`, `T5BucketError`, `PositionIdError`, and `ConfigLoadError`
  are now `#[non_exhaustive]`, like every other public error enum.
  `DeltaRuleConfig::normalization_epsilon` is renamed `qk_l2_epsilon`: a
  serde alias keeps old configs loading, `normalization_epsilon()` remains as
  a deprecated accessor, and validation names the field
  `delta_net.rule.qk_l2_epsilon`. `T5BeamSearchOptions::early_stopping` is
  `EarlyStopping` (below). The pre-release alias `GptNeoXModel` is removed.
- **Generation.** Presence and frequency penalties count generated tokens only
  by default (`PenaltyScope::PromptAndGenerated` restores prompt counting);
  `LogitsProcessor::process` treats its whole history as generated. Greedy
  decoding ignores temperature, top-k, top-p, and min-p, and reports log
  probabilities of the untruncated distribution. F16/BF16 logits are processed
  in F32 and selected log probabilities are F32. Top-k keeps every token tied
  with the k-th score, as Transformers does. Seeded samples can differ for
  draws within about `1e-5` of a CDF boundary.
- **Checkpoint generation defaults.** `generation_config.json` follows
  Transformers: when sampling, an unset `top_k` becomes `min(50, vocabulary)`,
  an explicit one is clamped to the vocabulary, and `top_p >= 1` disables
  top-p; causal loads also apply `min_p`, `no_repeat_ngram_size`, and
  `min_new_tokens` (clamping an unreachable `min_new_tokens` and reporting
  it). Seq2seq loads keep `repetition_penalty`, `presence_penalty`, and
  sampling neutral in their defaults and report those three processor
  settings, `repetition_penalty`, and `do_sample` with its non-neutral
  `temperature`, `top_k`, and `top_p` as unsupported, because beam search
  does not apply them.
- **Beam search.** `BeamSearchConfig::early_stopping` is the `EarlyStopping`
  enum: `false` now selects Transformers' heuristic default, and the previous
  `false` behavior is `EarlyStopping::Never`. `T5BeamSearchOptions::early_stopping`
  is also `EarlyStopping` (default `Heuristic`); 0.1.0's default `false` is
  `EarlyStopping::Never`. The cache needs capacity
  `prompt + max_new_tokens - 1`; log probabilities are F32.
- **Speculative decoding.** `Qwen3_5SpeculativeDecoder` and the Nemotron-H
  speculative decoder take a `prompt_length` argument; penalties and
  `min_new_tokens` then match `generate_causal`. Seeded stochastic Qwen
  speculation differs from earlier versions; greedy output is unchanged.
- **GPT-NeoX** `hidden_states` has `L + 2` entries (embedding, every block, final
  norm): crate index `L + 1` is Transformers' index `L`. Its activation points
  are listed in emission order.
- **Error kinds and messages.** Runtime hook failures are `Error::Intervention`.
  GPT-2 and GPT-NeoX token, position, mask, and logits-selection errors are
  `Error::Shape` (device mismatches `BackendCapability`); T5 decode-device
  mismatches are `BackendCapability`; DeepSeek stateful batch and capacity
  errors are `Error::TensorCache`; modular staged position errors are
  `Error::Shape`. Value errors now follow every layout error, messages read
  `token values in 0..V`, `position values in 0..N`, and
  `{tensor} and {owner} are on different devices`, and automatic config
  errors name the family's field instead of `auto_causal.config`. Automatic
  loading reports typed `Error::CheckpointPlan` and `Error::CheckpointExecution`
  instead of `Error::Checkpoint(String)`, with unchanged messages, and a
  checkpoint holding fewer tensors than the configuration's depth (or exact
  one-to-one inventory) needs fails with `CheckpointError::TooFewTensors`
  before the contract is built. Direct T5 loading applies the same bound:
  `T5Config::plan_checkpoint` and `T5ForConditionalGeneration::load_checkpoint`
  return `T5CheckpointLoadError::Checkpoint(TooFewTensors)` for a checkpoint
  with fewer tensors than encoder plus decoder blocks, after any configuration
  error and before the contract is built. A modular decode state rejects a batch
  mismatch as `Error::TensorCache(BatchSizeMismatch)`, as every family does.
  GPT-NeoX `attentions: true` is `Error::BackendCapability`.
- **Tokenization.** `EncodeOptions::max_length` truncates content before the
  post-processor's special tokens; truncation and padding serialized in
  `tokenizer.json` are no longer applied implicitly. Chat templates follow
  `apply_chat_template` semantics, including Python `str()` conversions in
  filters and Python rounding; mapping keys reach templates sorted. Template
  discovery follows Transformers 5.2: `chat_template.jinja` (as `default`)
  and `additional_chat_templates/<name>.jinja` replace the `chat_template`
  entry of `tokenizer_config.json`; a template path counts only when it
  resolves to a regular file (and the directory only when it resolves to a
  directory), so dangling or looping symlinks, directories, and other
  non-files are skipped; template files are read with universal newlines; and
  `"chat_template": null` means no template. Without `template_name`,
  `render_chat` and `encode_chat` use a checkpoint's `tool_use` template
  whenever `ChatTemplateOptions::tools` is set and not JSON null, as
  Transformers 5.2's `get_chat_template` does (CHAT-4). Errors for an unknown
  or ambiguous template name list the available names, and a non-UTF-8
  template file is `Error::Artifact`.
- **Models.** Qwen3-MoE sliding windows apply to every block and
  `max_window_layers` is ignored, as in Transformers 5.2. DeepSeek and Muse
  reject `partial_rotary_factor` other than one. DeepSeek's `seq_aux = false`
  balance loss is `1/top_k` of its earlier value. Nemotron-H's balance loss is
  Megatron's normalized form, and `NemotronMamba2Config` uses `time_step_clamp`.
  T5 configs with `relative_attention_max_distance <= buckets / 2` are
  rejected. GPT-2 random initialization matches GPT-2; modular T5 biases start
  from `N(0, 1/hidden_size)`. Stateful DeepSeek and Muse decoding accept any
  sentinel in masked columns. `CacheConfig`'s sliding window only sizes raw
  cache-bank rings; every model family, DeepSeek-V2 now included, ignores it.
  `MuseGlimmerForCausalLm::load_checkpoint` detects the text-only layout
  instead of assuming the multimodal one, and the Qwen3.5 backbone recipe
  uses the same shared namespace classifier as family and automatic loading.
  GPT-NeoX accepts `router_logits` and `auxiliary_losses` and returns empty
  diagnostics. The modular reference decode accepts a row without history and
  an entirely inactive step as exact-zero no-ops, so chunked `generate_causal`
  works for `ModularCausalLm` on left-padded batches.
- **Specs.** `RopeScaling` is no longer `Copy` and `Yarn` has `truncate`;
  `ModularLayer::init` takes `Option<&CrossAttentionSpec>`; validation accepts
  and rejects a few more corner cases (see MIGRATION); an explicit attention
  policy replaces all structural connectivity.
- **Attention.** `GroupedQueryAttention::attend_bounded` on autodiff backends
  evaluates one unbounded call with the same values, as Muse and DeepSeek do,
  instead of returning `Error::BackendCapability`; MLA's bounded path now
  rejects a zero-limit policy on every backend.
- **Transfer.** Fan-out copies of one source parameter are independent; under
  `TiePolicy::Preserve`, fanning out a member of an existing tie is an error.
  `transfer::parameters` lists packed and rank-zero parameters.
- **Checkpoints.** Auto discovery with both `model.safetensors` and an index
  loads through the index only when it names exactly that file, and otherwise
  reports ambiguity. F32 to BF16 and F64 to F16/BF16 casts round to nearest
  even; FP8 conversions fail at plan time. `embed_out` is quantization-eligible.
  Packed Qwen3.5 overrides install exactly (dequantized on Flex, packed on
  Metal). SafeTensors discovery reports out-of-section ranges at the first
  violation in byte-offset order, through the verified tiling, and reports
  header-entry errors such as byte-length mismatches before any layout error.
  `LoadReport::quantized` and `QuantizationRecord` describe the artifact's
  packed form, not in-memory storage. `AutoSeq2SeqLm::load_local` plans the
  checkpoint before it builds the graph, and Nemotron-H and T5 planning
  discover the checkpoint before building their contracts.
- **Caches.** `SequenceLengths::lengths()` reports committed lengths inside a
  transaction. A failed cached call leaves every length and readable slot
  unchanged, but write-ahead may leave bytes beyond the lengths. Steps and read
  plans are bound to the state and lifecycle that produced them.
- **MoE.** A token with a non-finite router score is an error instead of a NaN.
- **Build.** The `flex` feature enables Flex's SIMD kernels and rayon-parallel
  matrix products.
- **Internal structure.** Ragged-row block planning (`BlockLayout` and its
  verified host plan) moved from DeepSeek-V2 into
  `models::shared::{block, block_plan}`; the dense decoders' cache-group
  numbering, step context, and checkpoint extents are verified once in
  `models::shared::decoder_plan`, and Qwen3.5's compact row plan reuses the
  shared row maps; embedding-multiplier rounding and logit soft caps live once
  in `models::shared::scaling`; one verified ceiling (`crate::division`)
  replaces three copies; and modular clamped experts run through the shared
  `SparseMoe` forward. Outputs are bit-identical.
- **Gemma 3 in F16 (behavior change).** Loading a Gemma 3 checkpoint
  (`model_type` `gemma3` or `gemma3_text`) whose parameters, and so
  activations, would be F16, whether cast with `DTypePolicy::Cast(F16)` or
  stored in F16 and kept by `DTypePolicy::Auto`, now fails while planning,
  before any payload is read, with `CheckpointError::RangeHazard`
  (`Error::CheckpointPlan` from `AutoCausalLm::load_local`,
  `GemmaCheckpointLoadError::Checkpoint` from `GemmaForCausalLm::load_checkpoint`,
  and directly from `checkpoint::build_load_plan` given the Gemma contract's
  expected tensors and mappings).
  It used to load and return non-finite logits: the published checkpoints
  overflow F16 in Transformers too (gemma-3-4b-it's block-5 post-MLP residual
  reaches `73,509.6`, beyond `65,504`; 12B and 27B overflow at blocks 6 and 8).
  The message names the measured overflow, the alternatives (F32, and BF16
  where the backend can store it; packed Q4/Q8 weights with F32 activations
  are named as not offered, because no Gemma packed path is certified and the
  automatic loader refuses a Gemma packed sidecar), and the opt-in: `LoadOptions::range_hazards = RangeHazardPolicy::Accept` loads the
  faithful, overflowing graph for experiments. The refusal covers unmeasured
  Gemma 3 sizes too; Gemma 2, F32, and BF16 loads are unchanged. The Gemma
  gate's `TRANSFORMERS_GEMMA_DTYPE=f16` runs now expect the refusal
  (`TRANSFORMERS_GEMMA_RANGE_HAZARDS=accept` evaluates the faithful graph and
  fails), and `LoadOptions` gained the public field `range_hazards`, so
  exhaustive literals need it (serialized options without it load with the
  default).
- **Deprecated:** `LoadedModel::into_parts` (and its variants),
  `CacheConfig::physical_capacity`, `BURNPACK_QUANTIZATION_FORMAT_VERSION`,
  `BurnpackQuantizationManifest::new`,
  `MuseGlimmerForCausalLm::load_checkpoint_with_family_options` and
  `NemotronHForCausalLm::load_checkpoint_with_architecture_options` (use
  `load_checkpoint_with`), `DeltaRuleConfig::normalization_epsilon()` (use the
  `qk_l2_epsilon` field), and the unreleased `qwen3_moe::router_logits_point`
  and `router_weights_point` (use `Layer::router_logits` and
  `router_weights`).
- **Benchmark policy:** by the maintainer's October 18 decision, Qwen3.5-0.8B
  timing is judged against the head `fa8426e` reference measured on the
  current host (Mac15,11, macOS 27.0.1), not the September 10 series from
  macOS 15.7.9. Warm, natural-loop, and strict-load medians keep the 5% alert
  and confirmed 10% failure thresholds; cold first token, which includes
  Metal pipeline compilation, is reported as a median and range and is no
  longer gated. The v0.1 release record is unchanged, and no support claim
  changes ([record](acceptance/measurements/qwen3.5-0.8b-metal-rebaseline-20261018.json),
  [protocol](BENCHMARKS.md#qwen35-08b-benchmark-campaign-protocol)).

### Fixed

- Attention: bottom-right causal alignment for non-square causal calls on
  autotuned CubeCL flash kernels; `i32` position wraparound in connectivity
  masks; sliding windows wider than the key span no longer exclude every key;
  bounded attention validates its full masks.
- Sampling thresholds are scaled by the realized probability total, so the
  selected index always has positive mass.
- RoPE: YaRN's wide correction ranges follow Transformers' `d - 1` clamp;
  configs carrying both `rope_type` and `type` are accepted when they agree;
  Gemma 3 RoPE mappings saved by Transformers 5.2 load.
- DeltaNet: width-one convolutions, padding removed from convolution taps and
  history, PyTorch's thresholded softplus, and exact decay differences. The
  token-by-token recurrence read the wrong rows' decay and update strength
  for every batch above one on CubeCL backends (Burn 0.21's CubeCL reshape of
  a transposed unit axis), so batched gated-DeltaNet decoding and short
  prefills, Qwen3.5's included, were wrong there; Flex was exact. The chunked
  kernel misread single-head factors passed as strided views on the same
  backends. Both were measured on Metal and the CubeCL CPU backend; CUDA and
  ROCm share the reshape code but were not run with the defect.
- Mamba-2: padding cleared by replacement; the reference `dt` graph; the chunk
  scan for a first Nemotron-H prompt token; padded Nemotron-H queries stay
  inside their row's keys.
- DeepSeek: F16/BF16 decoding with an inactive row no longer panics; inactive
  rows keep F32 router logits; the balance loss scale.
- Embedding lookups never see padding sentinels (Llama family, DeepSeek, Muse).
- MoE routing on CubeCL backends (measured on Metal and the CubeCL CPU
  backend; CUDA and ROCm run the same cubek-reduce kernel but were not run): a
  token whose remaining scores were all `-inf` or NaN received an out-of-range
  expert ID, and dispatch rejected it, because Burn 0.21's CubeCL maximum of a
  row of `-inf` is `f32::MIN`; the deterministic top-k now clamps its ranking
  keys to the dtype's lowest finite value, so it stays total on every backend
  without extra launches per selected expert.
- Qwen: a speculative cycle whose commit fails rolls back; oversized
  transactional calls fail before model work; new recipe parameters take the
  checkpoint dtype.
- T5: the decoder bucket bound is validated, bucket tables are bound to the
  bias that built them, and F16 residuals are clamped like the reference.
- Gemma: multimodal checkpoints with a final soft cap the reference never
  applies are refused.
- Feed-forward: tanh-approximate GELU on F16 and BF16 inputs is evaluated in
  F32 and rounded once, as PyTorch's `gelu(approximate="tanh")` (Gemma's
  `gelu_pytorch_tanh`) does. Burn's composed graph rounded every operation and
  both constants to the activation dtype (BF16 `GeluTanh(-2)` was `-0.046875`
  instead of `-0.0454`). F32 and F64 evaluation is unchanged.
- SiLU on F16 and BF16 inputs is evaluated in F32 and rounded once, as
  PyTorch's `silu` (Transformers' `SiLUActivation`) does: in dense and gated
  MLPs, routed and shared experts, the SiLU attention output gate, and the
  Gated DeltaNet and Mamba-2 convolutions. Burn's `x * sigmoid(x)` rounded the
  sigmoid to the activation dtype before the product (BF16 `SiLU(2.8125)` was
  `2.640625` instead of `2.65625`). F32 and F64 evaluation is unchanged, and
  the clamped SwiGLU keeps gpt-oss's operation-by-operation graph.
- Attention: the eager graph (`AttentionPrecisionMode::EagerFp32Softmax`)
  multiplies F16 and BF16 `Q K^T` by the unrounded score scale in F32 and
  rounds once, as Transformers' eager `torch.matmul(query, key.T) * scaling`
  does with a Python float. Burn's `tensor * scalar` rounded the scale to the
  activation dtype first, which for `1 / sqrt(128)` moved 2.4% of BF16 and
  15.5% of F16 scores by one unit in the last place. The eager graph is
  Muse-Glimmer's default in F16 and BF16, whose 128-wide heads make this a
  change of its default half-precision numerics; the default of GPT-2 and
  GPT-NeoX, where it is a no-op wherever the head width makes the scale a
  power of two (GPT-2's 64-wide heads give `1/8`); and the graph of every
  score or pattern hook. F32 and F64 are unchanged, and so is the fused path,
  which every other family uses by default.
- Attention: on Flex, the eager graph's two F16 matrix products (`Q K^T` and
  probabilities times values) are summed in F32 and rounded once, as
  `torch.matmul` sums half-precision operands on the CPU and MPS. Burn Flex
  0.21's F16 GEMM keeps F16 partial sums, so on the operation-level eager test
  10 of 120 probabilities and 45 of 96 outputs differed from Transformers'
  `eager_attention_forward`; the eager graph now matches it bit for bit there
  in BF16 and F16 on every row with a visible key. Flex's BF16 products and
  Metal's F16 products already were F32 sums rounded once (a new ignored
  Metal test pins the latter) and are unchanged, so no gate profile changes.
- Modular models: sinks with a soft cap are rejected at the spec field;
  sliding windows in encoder stacks are rejected; earlier block records still
  load; range contexts are validated once.
- Generation: seq2seq loads report generation constraints beam search ignores.
- Tokenization: content is truncated before special tokens; chat-template tags
  are scanned like MiniJinja's lexer.
- Checkpoints: alias collisions are detected regardless of order; an index
  naming only the single file loads; provenance hashing stops at the first
  failing shard within a fixed buffer budget; element counts multiply left to
  right like SafeTensors; packed installs validate before probing.
- Checkpoint loads install execution groups of at least 1 GiB (vocabulary
  matrices, and at F32 some per-layer tensors such as gpt-oss's decoded expert
  weights) before the source-ordered schedule. Installing them last had raised
  Metal's footprint: the Qwen3.8-27B F16 smoke peaked at 68.2 GB, and now
  peaks at 60.6 GB with bitwise-identical logits. Checkpoints without such a
  tensor keep the same schedule; Muse's conversion and F16 load open three
  shards instead of two
  ([record](acceptance/measurements/reference-precision-followups-20261016.json)).
- Caches: stale steps and foreign read plans are rejected; cache topology is
  locked after prefill.
- Experiments: fan-out transfer ties, unmatched initialization names, and
  validation of composed blocks inside the hook scope (a failed run clears
  earlier captures).
- The reference generator keeps quantized references in the requested dtype
  and rotary tables in F32.
- The reference generator's `--bf16-resident` mode refuses a model whose
  buffers differ from a true F32 build. Gemma 3 builds its `embed_scale` in
  the load dtype, so widening only the blocks would not reproduce F32; the
  pinned gpt-oss-20b, the one family whose F32 bundle uses the mode, passes.
- Keep fully excluded eager-attention rows finite and zero, including rows
  excluded through additive negative-infinity bias.
- Restore autodiff linear parameters as leaves after checkpoint-layout
  transposition, preserving the target's gradient policy and enabling a
  registered experiment to survive serialization and reloading.
- Reject packed quantized autodiff before unsupported Burn operations panic,
  including checkpoint installation; validate the explicit dequantized path.
- Guides: two inline snippets did not compile. `ModularCausalLm::init_with(spec,
  &ModularCausalLm::scaled_init_policy(&spec, 0.02), &device)` needed a
  turbofish and moved `spec` before borrowing it; EXPERIMENTS.md and
  TRAINING.md now show the two-statement form. EXPERIMENTS.md's
  `model.activation_points()?` is `HookedModel::activation_points(&model)?`,
  since concrete types' inherent method returns the list directly. The new
  `scripts/check_release.py --gate guides`, which CI runs, compiles every Rust
  block of the guides as a doctest.
- Configuration robustness: absurd layer and expert counts in `config.json`
  produce contextual `ConfigError`s in every family instead of
  capacity-overflow panics or unbounded allocation. Stack depths above
  `MAX_LAYERS`, routed-expert counts above `MAX_EXPERTS`, and, for Qwen3-MoE,
  DeepSeek-V2, and Nemotron-H, per-expert inventories above
  `MAX_CHECKPOINT_TENSORS` fail validation; `gemma::MAX_LAYERS` re-exports
  `MAX_LAYERS`, so its value drops from 2^20 to 2^18. Qwen3-MoE validation,
  contract building, and model initialization process `mlp_only_layers` in
  `O(L + M)` instead of `O(L · M)` (a 2^16-block override list validated in
  0.01 s instead of 5.9 s in debug).
- Configs written by Transformers 5, which spell infinity as
  `{"__float__": "Infinity"}` (for example Nemotron-H's `time_step_limit`),
  load; a tagged `-Infinity` or `NaN`, or an unknown tag, is rejected with its
  JSON path.
- Nemotron-H ignores the serialized `sliding_window`, as the pinned
  Transformers graph does; `NemotronHConfig::attention_window` opts into a
  window for experiments.
- Packed loading: certified packed Qwen3-MoE and Muse Burnpack loads install
  every packed projection exactly. Projections stay QFloat where the backend
  keeps the artifact's blocks (Metal) and are stored as exact dequantized F32
  where the quantized transpose requantizes (Burn Flex 0.21); previously Flex
  silently requantized them. Every other family's tensor sink does the same
  when driven through the quantized executor directly.
- The packed Q4/Q8 Metal guard also rejects builds with
  `attention-kernel-tracing`, which enables CubeCL autotuning.
- `operations::linear` with a packed Q4 weight on a CubeCL backend no longer
  returns wrong values for multi-row inputs. With F32 activations Burn 0.21's
  multi-row kernels were unrelated to the dequantized reference (relative error
  up to 1.0 for 2 to 32 rows on an M3 Max), so short prefills and batched decode
  through Q4 weights were wrong; such products now run one row at a time on the
  same packed storage, with no dequantization. F16 activations on Metal were
  measured correct and keep one matmul. 2-bit packed weights, which those
  kernels get wrong even for one row, are rejected on CubeCL backends.
- The FP32 router projection (`RouterProjectionPrecision::Fp32`, used by
  DeepSeek-V2 and Nemotron-H) gets a weight and bias gradient on every
  backward pass. Under `Autodiff<Flex>`, a loss that reached the router only
  through its logits used to lose the gradient after the first pass. Logits on
  Flex are bit-identical.
- Automatic decode states forward `reorder_within_groups`, so beam search
  through `AutoSeq2SeqLm` keeps T5's encoder binding and cross-attention K/V in
  place (about 0.15 ms instead of 1.3–1.5 ms per reorder at FLAN-T5-small
  geometry with 4 beams and 512 source tokens on Flex).
- Top-p ranks candidates by logit, like Transformers' `TopPLogitsWarper`, and
  never drops a logit one ulp above a kept one (TOPP-TIE-1).
- Chat templates: the renderer reads `\r\n` and `\r` in template source as
  `\n`, as Jinja2 does, and keeps the newline right after `{% raw %}`, which
  Jinja2 does not trim and MiniJinja did.
- Packaging: the source archive now includes `verus/*.rs` and
  `verus/README.md`, which `tests/verus_harness.rs` reads, so `cargo test`
  passes from the extracted archive (all three Verus structure tests failed
  there before). The package gate now runs the archive's test suite. Its
  listing check had rejected the committed tiny-golden `.safetensors`
  fixtures as weights since they were added, so the gate failed before
  packaging; it now admits them (regular files of at most 1 MiB under
  `tests/fixtures/tiny_goldens/<case>/`) and still rejects any other weight.

### Performance

Unless a bullet says otherwise, measured on Flex CPU in release builds with
small random models, on a host shared with other builds; ratios are against the
earlier row-by-row or per-chunk algorithm, which each test keeps as an oracle.

- Batched cached decoding, eight rows: Qwen3.5 3.1× to 3.6×; Gemma 2.9×
  (ragged) and 3.2× (uniform); DeepSeek 2.5× to 2.7×; Muse 2.2×; Nemotron-H
  2.1×; Llama family 1.9× to 2.3×; T5 1.5× to 2.8× (batch 4) and 1.7× (batch
  8); Qwen3-MoE 1.3× (ragged) to 1.7×; gpt-oss 0.9× to 1.1× at width 256 and
  0.8× at width 1024.
- One-pass speculative verification: Qwen3.5 1.6× per cycle (4 tokens).
- Gated DeltaNet chunked prefill, 1,024 tokens: 40.7 ms versus 87.4 ms (2.15×).
- Mamba-2 chunk scan, 2,048 tokens: 103.6 ms versus 145.5 ms (1.40×).
- T5 relative bias per pass: 2.3× to 4.6×; modular sliding-window masks: 65×.
- Top-k filtering: 1.10× to 1.19×; skipping value masking for clean cache
  tails: about 1.6× per ragged gpt-oss decode step and 30% for the Llama
  family.
- Memory-bounded MoE dispatch: 313 MiB versus 885 MiB peak RSS on the
  dispatch benchmark.
- Cached grouped-query decoding and suffixes fold query groups over native KV
  heads instead of repeating K/V (square causal prefill on the backend-native
  path still repeats K/V to keep the fused `is_causal` kernel); Muse cached
  sliding attention slices window unions (13.6× fewer score elements at 32K);
  input validation and generation bookkeeping read back to the host once per
  step.
- `generate_causal` and seq2seq beam search no longer materialize a negated
  `[batch, vocabulary]` mask per step: about 5–6 ms per step saved at batch 8
  with a 248,320-token vocabulary on Flex, with bit-identical results.
- On Flex, all bit-identical: cache token writes are in place, costing
  O(written slots) instead of O(capacity); the Mamba-2 decode update is a
  `k = 1` outer product (8–18×); Mamba-2 segment transitions use flattened
  tables (about 2.8–4.2×); the one-token DeltaNet and Mamba-2 depthwise
  convolution column avoids rayon (1.5–13×); and RoPE application is one fused
  pass (22–65× on 512-token prefill operands).
- On Metal, checkpoint weights upload as persistent allocations (`45ac457`),
  so CubeCL gives each weight an exact-size page instead of carving it from
  the pool meant for activations. Loading Qwen2-7B with F32 casts (30.5 GB of
  weights) on the Apple M3 Max (96 GB) development host: peak footprint 57.0 GB
  to 45.5 GB and load time 25.6 s to 10.7 s, with the same first-token argmax.
  Values, layouts, and Flex are unchanged. This is a one-run measurement from
  the commit, not yet an acceptance record.

### Verification

- 25 Verus harnesses (2,035 verified, 0 errors, over 88 source files) prove
  the integer planning of the cache ledger and rings, attention chunking and
  projection widths, checkpoint layout, naming, and MXFP4 decoding (including
  its thread partition), generation bookkeeping and speculative commits, MoE
  dispatch, DeltaNet, Mamba-2, and MLA geometry, position indices,
  configuration validation, selections, transfer plans, tokenizer layout and
  chat-template scanning and budgets, family checkpoint loaders, and every
  family's batching and cache-slot plans, including the configuration count
  bounds and exact per-expert checkpoint inventories and one shared ceiling
  division. See [verus/README.md](verus/README.md).
- CI runs the proofs on every push and pull request that changes `src/` or
  `verus/`, through the new `scripts/check_release.py --gate verus`. The gate
  pins the prover release, fails when a harness's count differs from the table
  in `verus/README.md`, and records per-harness counts and input hashes.
- The `verus_harness` test also rejects `exec_allows_no_decreases_clause`,
  `verifier::external`, `verus_verify(external)`, `assume_termination`,
  `inline_air_stmt`, and the external-specification attributes (the first
  three pass `--no-cheating`),
  treats every file containing `verus_keep_ghost` as needing a harness, and
  checks the proof counts quoted by the documentation against the tree.

### Testing

- Tiny Transformers 5.2.0 goldens for the Llama family (seven cases), Gemma
  (two), and gpt-oss, and HF-generated goldens for the shared router, RoPE
  rules, sinks, and differential attention. The tiny-golden checks are
  backend-generic and also run, ignored, on Metal and CUDA. Their greedy check
  compares a step only when the run's own margin exceeds twice the bound's
  largest allowance (a margin above one allowance does not force the
  reference's token), and every case must compare at least two steps.
- Batched-versus-stateless and batched-versus-row-by-row parity suites in every
  family, with ragged and padded rows, ring wraparound, and rollback, reorder,
  and truncation.
- An accelerator-parity suite (`tests/backend_parity.rs`) runs a tiny model of
  every executable family on Metal or CUDA with Flex's exact parameters and
  compares stateless, ragged, and cached prefill-plus-decode logits; each
  device is probed for F32 or TF32 matrix inputs and held to the matching
  bound. Its delta-rule case also runs, not ignored, on Burn's CubeCL CPU
  backend (`--features cpu`), which reproduces the DeltaNet reshape defect,
  and a unit test pins Burn 0.21's reshape decisions behind the workaround.
  Its deviation measure, which the real-checkpoint gates now share, fails
  NaN and infinite values instead of ignoring them.
- The CubeCL CPU backend (`--features cpu`) now runs every family of the
  parity suite and the three tiny-golden suites, held to the F32 bounds after
  a matrix-precision probe; the family and golden cases are ignored for their
  run time, and CI runs them on `ubuntu-latest`. Every case passes except
  Nemotron-H and the modular causal decoders, which meet a Burn 0.21 CubeCL
  CPU copy defect (fixed in Burn 0.22) and stay ignored with an upstream
  report and a reproducer; this backend is validated for testing, not
  supported for inference ([SUPPORT.md](SUPPORT.md#cubecl-cpu-backend-validation)).
- Twelve fuzz targets (new: logits processing, chat templates, RoPE
  parameters, SafeTensors headers, attention connectivity, modular specs, and
  the MTP bank of cache transactions) and property suites for generation,
  RoPE, and connectivity; open findings are kept as ignored reproducers.
- Ignored real-checkpoint gates for the Llama family, Gemma (including a
  long-context bundle), and gpt-oss, on Flex, Metal, and CUDA (FLAN-T5-small's
  release gate gained a CUDA variant too), and artifact-gated tokenizer and
  chat-template parity tests for the pinned FLAN-T5-small, Qwen3.5, Nemotron-H,
  and Muse Glimmer tokenizers. Their cached-greedy checks feed the reference's
  tokens back and skip ties at the tolerance instead of stopping at the first
  one, so the full BF16 gpt-oss-20b bundle meets its default minimum.
- An ignored gpt-oss test reproduces the half-precision bound's calibration
  from the official BF16 and F32 bundles alone (Transformers' own BF16
  deviation, at most `2.665`, mean at most `0.108`); it evaluates no Burn
  model and passes no gate
  ([record](acceptance/measurements/reference-precision-followups-20261016.json)).
- Half-precision variants of the Llama-family and Gemma real-checkpoint gates
  (`TRANSFORMERS_LLAMA_FAMILY_DTYPE`, `TRANSFORMERS_GEMMA_DTYPE`: `bf16` on
  Flex, `f16` on Metal, either in the CUDA variants, where they refuse the F32
  tolerance overrides whatever the TF32 probe finds; no CUDA run is recorded)
  judge a run against the existing F32 (or F64) bundle with per-prompt bounds
  read at run time from an official BF16 twin bundle, which is paired fail
  closed: `1.2` times Transformers' own BF16 maximum and mean deviation, the
  slack being the largest Burn/official ratio measured on gpt-oss-20b,
  rounded up. These bounds calibrate BF16 runs; a Metal/F16 run reuses them as
  an upper limit, since no official F16 bundle can be generated. Their greedy
  continuations use the gates' shared teacher-forced rule, with prompt 0's
  maximum bound as the tolerance. Every cached decode step and both Gemma
  long-context chunk ends are checked against the same run's stateless pass,
  and a prompt decoded token by token against both bounds, whether or not any
  greedy step is decisive. The shared logic and its unit tests are in
  `tests/support/half_precision_calibration.rs`; F32 runs are unchanged. The
  seven official BF16 bundles and their calibration are recorded
  ([record](acceptance/measurements/half-precision-calibration-20261017.json)).
  Their first Burn runs are in the Mac validation record below.
- A Mac validation round records 173 single-process test runs with their
  logs, diagnostics, and harness, each named by source commit and binary
  digest, and checked by `scripts/tests/test_mac_validation_record.py`
  ([record](acceptance/measurements/mac-validation-20261017.json)). Under
  `c15f24e`'s install order the Llama-family, gemma-3-4b-it, truncated
  gpt-oss, and FLAN-T5-small F32 gates repeat every `f4c56c4` line, with
  lower peak footprints for Qwen2-7B, Llama-3.1-8B, and truncated gpt-oss. The
  Muse-Glimmer-30B and Qwen3-30B-A3B Q8/Q4 conversions reproduce their
  authenticated Burnpacks byte for byte, and their packed Metal smokes and
  audits pass. The first Burn half-precision runs of the Llama-family gates
  pass on Metal/F16 and in 10 of 15 cases on Flex/BF16; Gemma's pass 4 of 5
  on Flex/BF16 after the tanh GELU fix and cannot pass on Metal/F16, because
  gemma-3-4b-it overflows F16 in Transformers too. The failures, which the
  record traces to the calibration rule's width, the decode tolerance, and
  the twice-rounded half-precision SiLU, remain open; no bound was changed
  and no checkpoint is promoted.
- The Llama-family and Gemma half-precision gates are now judged against the
  envelope of the four official graphs that Transformers 5.2.0 runs in the
  run's dtype on this host (CPU and MPS, eager and SDPA attention), read at
  run time from `TRANSFORMERS_LLAMA_FAMILY_OFFICIAL_DIRS`,
  `TRANSFORMERS_GEMMA_OFFICIAL_DIRS`, and
  `TRANSFORMERS_GEMMA_LONG_OFFICIAL_DIRS`, instead of `1.2` times the CPU-eager
  BF16 twin, which Transformers' own other BF16 graphs failed on 5 of 7
  bundles. A run passes a prompt when its maximum and mean deviation from the
  F32 (F64) truth stay within `s_max` and `s_mean` times the largest official
  ones; the slacks are the official graphs' pooled leave-one-out maxima,
  rounded up (BF16 `1.5`/`1.3`, F16 `1.7`/`1.7`), derived by
  `scripts/derive_half_precision_envelope.py` and re-derived, never tuned,
  when its inputs change. F16 runs are calibrated on the official F16 graphs
  instead of reusing the BF16 bounds; a truth whose official graph is
  non-finite in the run's dtype (Qwen2-7B and gemma-3-4b-it in F16) fails as
  uncalibrated. Decode steps are judged by truth-anchored rows: the cached and
  stateless decode rows must each meet the official graphs' decode envelope
  against an F32 (F64) CPU-eager decode truth
  (`TRANSFORMERS_*_DECODE_TRUTH_DIR`), which F32 runs also check at their
  tolerance, and the quarter-prefill decoded prompt is judged against the
  official graphs' own cached decode. The `0.5` self-consistency tolerance
  remains only, as provisional, for cached prefills, batched rows, and chunk
  ends. Pairing fails closed on graph identity, generator script, decode
  rows, and the decode truth's digests; the retired single-bundle variables
  are refused. A Gemma `TRANSFORMERS_GEMMA_DTYPE=f16` run reads the official
  F16 graphs (`TRANSFORMERS_GEMMA_F16_OFFICIAL_DIRS` and
  `_LONG_F16_OFFICIAL_DIRS`) and fails as uncalibrated before loading, because
  all of them overflow; Metal is qualified in F32 for Gemma. A gate-level
  refusal of `f16`, which anticipated the maintainer's unconfirmed decision
  (d), was withdrawn.
- Operation-level tests pin the half-precision rounding points of SiLU, tanh
  GELU, the sigmoid output gate (every finite F16 and BF16 input), RMSNorm in
  both conventions, RoPE, and the eager and fused attention scores to the
  references, and pin three Burn deviations as findings: Flex's F32-to-BF16
  cast ignores the three lowest F32 bits, Flex's F16 matmul keeps F16 partial
  sums, and tanh GELU overflows for BF16 inputs of at least `2^127`.
- The once-rounded SiLU is pinned at the Mamba-2 and Gated DeltaNet
  convolution sites themselves, not only in the shared helper: each mixer is
  built so that its F32 recurrent state after one token holds the
  post-convolution activation exactly (an identity convolution and exact
  unit factors), which must equal the F32 SiLU rounded once in BF16 and F16
  (BF16 `SiLU(2.8125) = 2.65625`, not the composed `2.640625`) and Burn's
  graph bit for bit in F32, in both scan modes. Reverting either call site to
  `activation::silu` fails its test.
- An official-envelope round records the 57 official BF16 and F16 bundles
  that the envelope rule reads, the derivation of its slacks, and the Burn
  runs at `f9f78b5`, each in its own process with its log, checked by
  `scripts/tests/test_half_precision_envelope_record.py`
  ([record](acceptance/measurements/half-precision-envelope-20261018.json)).
  Every Llama-family and Gemma gate in F32 and Flex/BF16 passes, and every
  Metal/F16 gate of the four F16-calibrated Llama-family checkpoints; Qwen2-7B
  in F16 is uncalibrated. Qwen3.5-0.8B's parity and memory gates pass
  again, as do the larger checkpoints' official gates, smokes, and audits and
  the Qwen3.8-27B Q8/Q4 conversions, except the Nemotron-H official-precision
  gate, whose cached row at decision 22 falls below its official F16 floor
  (`0.998558` against `0.998635`) with the once-rounded Mamba-2 convolution
  SiLU; a control reverting only that site passes. Qwen3.5-0.8B's benchmarks
  keep their tokens and every warm median, but their single-sample cold
  first token breaches the approved September 10 baseline, in the pre-change
  binary too: not attributable to this branch; the breach of the approved
  baseline awaits the maintainer (review record below). No bound was changed
  and no checkpoint is promoted.
- A half-precision follow-ups record holds the round's decisions verbatim,
  with their rationale and review flags. It also records a rerun of every
  Llama-family and Gemma gate at the integrated head `8181605`: 106 processes
  in the worktree, from rebuilt binaries byte-identical to the `f9f78b5` gate
  binaries, checked by `scripts/tests/test_half_precision_followups_record.py`
  ([record](acceptance/measurements/half-precision-followups-20261017.json)).
  94 pass, and 12 fail as expected: Qwen2-7B's uncalibrated F16 runs, the
  Gemma Metal/F16 refusals, and Metal BF16's load error on Burn 0.21. Every run
  that also ran at `f9f78b5` reproduces its output exactly, and every F32 gate
  repeats the pre-SiLU round's printed statistics. The Nemotron-H
  official-precision failure was not rerun and stands. No checkpoint is
  promoted.
- A half-precision review record holds the review round at `1e38eec`: the
  eager-attention and convolution-SiLU tests, seven failing mutations, the
  torch 2.10.0 probe, and 33 single-process runs (every gemma-3-4b-it gate,
  reproducing `8181605` exactly where it ran there, the Gemma Metal/F16 runs
  failing as uncalibrated, and the operation-level tests on Flex and Metal),
  checked by `scripts/tests/test_half_precision_review_record.py`
  ([record](acceptance/measurements/half-precision-review-20261018.json)).
  It records the Qwen3.5-0.8B cold-first-token breach of the approved
  baseline for the maintainer and corrects the envelope record's benchmark
  verdict and its claim that no real-checkpoint gate selects the eager graph
  (Muse-Glimmer's does). No checkpoint is promoted.
- A Qwen3.5-0.8B benchmark campaign protocol,
  `scripts/qwen35_benchmark_protocol.py`, plans, runs, summarizes, and judges
  campaigns that alternate two binaries in counterbalanced Latin-square
  rounds with a cooldown, load admission, in-run contamination reruns, and a
  steady-state rule; synthetic-campaign tests cover it. Its first campaign
  re-baselines Qwen3.5-0.8B: the head `fa8426e` against the September
  source `c70057e`, both rebuilt with rustc 1.92.0, in 12 rounds (48 clean
  processes). Review replaced the steady-state rule, which had taken the
  earliest window whose half-medians agreed within 2.5%, with a monotone,
  noise-aware rule over whole blocks. Under it the campaign is steady from
  round 5, not round 3. The reference block names its executable and
  compiler, a controlled rerun must run that binary, the compiler is part of
  the series, and cold first token is also reported without the reruns that
  followed contaminated attempts. It is checked by
  `scripts/tests/test_qwen35_rebaseline_record.py` and
  `tests/acceptance_records.rs`
  ([record](acceptance/measurements/qwen3.5-0.8b-metal-rebaseline-20261018.json)).
- The configuration fuzz target covers all eleven family parsers without
  count filters, and the modular-spec generator draws clamped experts,
  embedding scales, and logit caps.

## [0.1.0] - 2026-09-10

### Added

- Bounded support for the pinned Qwen3.5-0.8B Metal/F16 base-text profile and
  FLAN-T5-small Flex/F32 and Metal/F32 profiles. Larger models, quantized
  quality and real-weight MTP remain experimental. This records a local source
  release, not a registry publication or Git tag.
- Backend-generic causal and encoder-decoder inference contracts for Burn.
- Strict, local-only sharded SafeTensors discovery and streaming installation
  with deterministic tensor-level load reports.
- External preallocated KV, recurrent, compressed-latent, sliding-window, and
  cross-attention cache state.
- Tokenization, chat-template rendering, causal generation, and T5 beam search
  behind optional features.
- Modular implementations of normalization, position encodings, attention,
  recurrent mixers, dense feed-forward networks, and sparse experts.
- Experimental model-family implementations and native Q4/Q8 Burnpack tooling;
  exact advertised checkpoint support remains governed by the committed
  acceptance records rather than by architecture presence alone.
- A content-authenticated FLAN-T5-small F32 release profile on Flex and Metal,
  covering strict loading, local tokenization, complete official-reference
  vectors, cached 32-token decoding, beam search, right padding, and the
  checkpoint's 512-token source boundary.
- Authenticated Qwen3.5-0.8B stable-prompt F16 references, public streaming and
  padded-batch checks, populated Q4 32K stress, and repeated small-model Metal
  timing records. These remain bounded evidence, not a full release promotion.
- Separate sampled phase-memory evidence for both small models, retaining idle
  baselines, sampling gaps, and warmup/measured partitions without claiming exact
  transient peaks or a host/device allocation split.
- Opt-in native compiler/export binding for development coverage gates, with
  authenticated source inputs and binary identities, unfiltered exports, and
  strict diagnostics. Passing aggregate coverage is not production-only
  qualification or checkpoint promotion.

### Known limitations

- Qwen's historical decode throughput fell from approximately 36 to 27 tokens/s.
  Rebuilt historical and current sources agree within 1.247% on controlled warm
  medians, but the historical cause remains unknown. The user explicitly approved
  the controlled September 10 measurements as the v0.1 baseline; old evidence
  is retained and future 5% alert / confirmed 10% failure thresholds are unchanged.
- Memory evidence reports OS-accounted process high-water and sampled phase
  maxima, not exact transient peaks or separately attributable host/device usage.

### Fixed

- Validate special-token metadata against the concrete tokenizer vocabulary.
  Reject absent/malformed token text paired with an ID and invalid explicit ID
  types or ranges, while preserving ID-only declarations and null overrides.
  This prevents padding IDs from bypassing model-vocabulary validation.
- Preserve backend-native Boolean dtype tags when importing dynamic token masks
  and gathering beam/cache rows, avoiding Burn 0.21 Flex panics in logical mask
  operations.
- Respect actual runtime float dtypes in selected generation probabilities,
  preserve F64 precision, and fail contextually for unsupported Flex32 sorting.
- Reject empty attention projections, overflowing lazy projection shapes,
  noncanonical provenance paths, and malformed dense parameter byte lengths
  before backend execution or parameter mutation.
- Reject duplicate decoded Burnpack tensor and checkpoint-provenance JSON keys,
  including escape-equivalent names, before a map can silently replace an entry.
- Validate repeated full-precision parameter installation in checkpoint layout
  after lazy parameters initialize. Reject already initialized quantized
  parameters before a save mapper could allocate whole-weight dequantization.
