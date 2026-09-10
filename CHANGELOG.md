# Changelog

All notable changes to this crate are recorded here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic
versioning, with the additional pre-1.0 compatibility policy documented in
[`SUPPORT.md`](SUPPORT.md).

## [Unreleased]

### Added

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

### Fixed

- Keep fully excluded eager-attention rows finite and zero, including rows
  excluded through additive negative-infinity bias.
- Restore autodiff linear parameters as leaves after checkpoint-layout
  transposition, preserving the target's gradient policy and enabling a
  registered experiment to survive serialization and reloading.
- Reject packed quantized autodiff before unsupported Burn operations panic,
  including checkpoint installation; validate the explicit dequantized path.

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
