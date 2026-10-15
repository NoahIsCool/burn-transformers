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
  type and real-weight Flex/F32 gates (`tests/llama_family_artifacts.rs`,
  `tests/gemma_artifacts.rs`, `tests/gpt_oss_artifacts.rs`); see the
  [model status page](ARCHITECTURE_COMPATIBILITY.md). No new checkpoint is
  advertised as supported.
- DeepSeek-V2 runs V3-style sigmoid `noaux_tc` routing with a correction bias
  and V2 group-limited routing. `GptNeoXForCausalLm` is the GPT-NeoX model
  name (`GptNeoXModel` remains an alias).

#### Experiment interface

- `architecture::HookedModel` and `StagedModel`, implemented by every family,
  by `AutoCausalLm` (both) and `AutoSeq2SeqLm` (hooks), and forwarded by
  `LoadedModel`, so one generic experiment runs on any model or loaded
  checkpoint. `AutoCausalConfig::supports_experiments()` reports support.
- New activation catalogs: DeepSeek MLA and router points, Muse gate and
  sandwich-norm points, Nemotron-H points per block kind, Qwen3-MoE router
  points (`router_logits_point`, `router_weights_point`) and a headless
  backbone, gpt-oss router points, and staged execution for all of them.
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
  equivalents); `Gpt2Config::init_policy` and `GptNeoXConfig::init_policy`.
- Modular specifications: `MixerSpec::AttentionWithOptions`
  (`AttentionOptions`: global, sliding, or chunked-local connectivity, score
  scale, soft cap, sinks, per-projection bias, Q/K-norm scope, SiLU gate),
  `MixerSpec::DifferentialAttention`, executable `MixerSpec::Mamba2`,
  `FeedForwardSpec::{Ungated, ClampedSwiGlu, None}`, MLA score scale and
  connectivity, MoE routing policy, shared-expert gate, and expert biases,
  `Seq2SeqSpec::cross_attention_options`, every RoPE scaling rule, and
  `encoder_stages`/`decoder_stages` views of `ModularSeq2SeqLm`.

#### Shared mechanisms

- Attention: `grouped_query_attention` over native KV heads,
  `AttentionConnectivity` (global, sliding-window, chunked-local) with
  `MaskGeometry::with_key_start`, learned sinks, `ProjectionBias`,
  `QueryKeyNormScope::FullWidth`, `attend_bounded` (score-bounded chunked
  execution), and `DifferentialAttention`.
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
  `no_repeat_ngram_size`, and `min_new_tokens` are applied.
- Checkpoints: exact MXFP4 decoding (`MappingPlan::dequantize_mxfp4`,
  `TensorOperation::DequantizeMxfp4`), `install_packed_parameter`
  (`PackedInstall`, `packed_layout_is_exact`), `CheckpointError::InvalidTensorHeader`
  and `InvalidTensorName`, `QuantizedCheckpointOptions::unaligned`, and
  `BurnpackArtifactSummary::packed_execution`.
- Tooling: `scripts/generate_causal_lm_reference.py` (local Transformers
  references for any causal checkpoint, with `--dtype` and
  `--num-hidden-layers`) and `scripts/generate_tiny_goldens.py`.

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

- **Public API.** Exhaustive matches on `MixerSpec`, `FeedForwardSpec`, both
  `RopeScaling` enums, `QuantizationExclusion`, and `RopeConfigError` need arms
  for their new variants. `GenerationConfig`, `ChatMessage`,
  `ChatTemplateOptions`, `SpecialTokens`, `SparseMoeConfig`,
  `GroupedQueryAttentionConfig`, `GatedDeltaNetConfig`, `MlaSpec`,
  `MixtureOfExpertsSpec`, `Seq2SeqSpec`, `RopeParameters`,
  `QuantizedCheckpointOptions`, and `BurnpackArtifactSummary` gained public
  fields, so exhaustive literals need them. The Burn records of
  `DeepseekV2Moe` and of Nemotron-H's dense, Mamba-2, and gated-norm modules
  are now the shared modules' record types.
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
  it), while seq2seq loads report those three as unsupported.
- **Beam search.** `BeamSearchConfig::early_stopping` is the `EarlyStopping`
  enum: `false` now selects Transformers' heuristic default, and the previous
  `false` behavior is `EarlyStopping::Never`. The cache needs capacity
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
  errors name the family's field instead of `auto_causal.config`.
- **Tokenization.** `EncodeOptions::max_length` truncates content before the
  post-processor's special tokens; truncation and padding serialized in
  `tokenizer.json` are no longer applied implicitly. Chat templates follow
  `apply_chat_template` semantics, including Python `str()` conversions in
  filters and Python rounding; mapping keys reach templates sorted.
- **Models.** Qwen3-MoE sliding windows apply to every block and
  `max_window_layers` is ignored, as in Transformers 5.2. DeepSeek and Muse
  reject `partial_rotary_factor` other than one. DeepSeek's `seq_aux = false`
  balance loss is `1/top_k` of its earlier value. Nemotron-H's balance loss is
  Megatron's normalized form, and `NemotronMamba2Config` uses `time_step_clamp`.
  T5 configs with `relative_attention_max_distance <= buckets / 2` are
  rejected. GPT-2 random initialization matches GPT-2; modular T5 biases start
  from `N(0, 1/hidden_size)`. Stateful DeepSeek and Muse decoding accept any
  sentinel in masked columns.
- **Specs.** `RopeScaling` is no longer `Copy` and `Yarn` has `truncate`;
  `ModularLayer::init` takes `Option<&CrossAttentionSpec>`; validation accepts
  and rejects a few more corner cases (see MIGRATION); an explicit attention
  policy replaces all structural connectivity.
- **Transfer.** Fan-out copies of one source parameter are independent; under
  `TiePolicy::Preserve`, fanning out a member of an existing tie is an error.
  `transfer::parameters` lists packed and rank-zero parameters.
- **Checkpoints.** Auto discovery with both `model.safetensors` and an index
  loads through the index only when it names exactly that file, and otherwise
  reports ambiguity. F32 to BF16 and F64 to F16/BF16 casts round to nearest
  even; FP8 conversions fail at plan time. `embed_out` is quantization-eligible.
  Packed Qwen3.5 overrides install exactly (dequantized on Flex, packed on
  Metal).
- **Caches.** `SequenceLengths::lengths()` reports committed lengths inside a
  transaction. A failed cached call leaves every length and readable slot
  unchanged, but write-ahead may leave bytes beyond the lengths. Steps and read
  plans are bound to the state and lifecycle that produced them.
- **MoE.** A token with a non-finite router score is an error instead of a NaN.
- **Build.** The `flex` feature enables Flex's SIMD kernels and rayon-parallel
  matrix products.
- **Deprecated:** `LoadedModel::into_parts` (and its variants),
  `CacheConfig::physical_capacity`, `BURNPACK_QUANTIZATION_FORMAT_VERSION`, and
  `BurnpackQuantizationManifest::new`.

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
  history, PyTorch's thresholded softplus, and exact decay differences.
- Mamba-2: padding cleared by replacement; the reference `dt` graph; the chunk
  scan for a first Nemotron-H prompt token; padded Nemotron-H queries stay
  inside their row's keys.
- DeepSeek: F16/BF16 decoding with an inactive row no longer panics; inactive
  rows keep F32 router logits; the balance loss scale.
- Embedding lookups never see padding sentinels (Llama family, DeepSeek, Muse).
- Qwen: a speculative cycle whose commit fails rolls back; oversized
  transactional calls fail before model work; new recipe parameters take the
  checkpoint dtype.
- T5: the decoder bucket bound is validated, bucket tables are bound to the
  bias that built them, and F16 residuals are clamped like the reference.
- Gemma: multimodal checkpoints with a final soft cap the reference never
  applies are refused.
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
- Caches: stale steps and foreign read plans are rejected; cache topology is
  locked after prefill.
- Experiments: fan-out transfer ties, unmatched initialization names, and
  validation of composed blocks inside the hook scope (a failed run clears
  earlier captures).
- The reference generator keeps quantized references in the requested dtype
  and rotary tables in F32.
- Keep fully excluded eager-attention rows finite and zero, including rows
  excluded through additive negative-infinity bias.
- Restore autodiff linear parameters as leaves after checkpoint-layout
  transposition, preserving the target's gradient policy and enabling a
  registered experiment to survive serialization and reloading.
- Reject packed quantized autodiff before unsupported Burn operations panic,
  including checkpoint installation; validate the explicit dequantized path.

### Performance

Measured on Flex CPU in release builds with small random models, on a host
shared with other builds; ratios are against the earlier row-by-row or
per-chunk algorithm, which each test keeps as an oracle.

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
- Grouped-query attention reads native KV heads without repeating them; Muse
  cached sliding attention slices window unions (13.6× fewer score elements at
  32K); input validation and generation bookkeeping read back to the host once
  per step.

### Verification

- 18 Verus harnesses (408 verified obligations, 0 errors) prove the integer
  planning of the cache ledger and rings, attention chunking and projection
  widths, checkpoint layout and MXFP4 decoding, generation bookkeeping and
  speculative commits, MoE dispatch, position indices, selections, tokenizer
  layout, and every family's batching and cache-slot plans. See
  [verus/README.md](verus/README.md).

### Testing

- Tiny Transformers 5.2.0 goldens for the Llama family (seven cases), Gemma
  (two), and gpt-oss, and HF-generated goldens for the shared router, RoPE
  rules, sinks, and differential attention.
- Batched-versus-stateless and batched-versus-row-by-row parity suites in every
  family, with ragged and padded rows, ring wraparound, and rollback, reorder,
  and truncation.
- Twelve fuzz targets (new: logits processing, chat templates, RoPE
  parameters, SafeTensors headers, attention connectivity, modular specs, and
  the MTP bank of cache transactions) and property suites for generation,
  RoPE, and connectivity; open findings are kept as ignored reproducers.
- Ignored real-checkpoint gates for the Llama family, Gemma (including a
  long-context bundle), and gpt-oss.

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
