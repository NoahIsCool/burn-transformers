# Changelog

All notable changes to this crate are recorded here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic
versioning, with the additional pre-1.0 compatibility policy documented in
[`SUPPORT.md`](SUPPORT.md).

## [Unreleased]

## [0.1.0] - Unreleased

### Added

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

### Fixed

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
