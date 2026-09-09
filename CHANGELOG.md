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

### Fixed

- Preserve backend-native Boolean dtype tags when importing dynamic token masks
  and gathering beam/cache rows, avoiding Burn 0.21 Flex panics in logical mask
  operations.
