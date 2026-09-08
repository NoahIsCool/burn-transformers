# Support policy

This document defines what a support claim means for the `transformers` crate.
It does not broaden the checkpoint matrix in `README.md` or any committed
acceptance record.

## Version compatibility

Version 0.1 requires Rust 1.92 or newer and pins Burn 0.21. Before 1.0, a minor
crate release may contain source-breaking API changes. Patch releases preserve
documented public APIs except where a change is necessary to correct unsound,
unsafe, or materially incorrect behavior.

Default builds enable `std` and Burn Flex. Hardware backends and optional
tokenization or template support are selected with crate features. Enabling a
backend proves that its code compiles; real-checkpoint execution is supported
only where the checkpoint's acceptance record names that backend and precision.

## Checkpoint support

A model-family implementation is not by itself a checkpoint support claim. A
checkpoint is supported only when its exact repository revision has a committed
record under `acceptance/` with `advertised_supported` set to `true`. That record
is authoritative for the validated dtype, backend, tokenizer, context length,
generation behavior, and any documented limitations.

Revisions, repacks, quantization formats, and third-party conversions not named
by such a record are experimental even when their tensor names happen to load.
Quantized support is tracked independently from full-precision support.

## Artifact and provenance boundary

The runtime reads local files only. It does not download models, resolve moving
Hub branches, or authenticate a caller-supplied repository and revision label.
Applications must pin and verify artifacts before calling `load_local`; source
metadata attached through `AutoLoadOptions` is a caller assertion. The strict
loader then validates containment, headers, tensor accounting, mappings,
dtypes, shapes, and the finalized load report for those local files.

Model weights, generated quantized sidecars, and tokenizer artifacts are not
distributed with this crate.

## Problem reports

A useful report includes:

- crate, Rust, Burn, operating-system, and backend versions;
- the exact model repository and immutable revision;
- enabled Cargo features and execution dtype;
- the complete error chain and relevant `LoadReport` categories;
- a minimal prompt or synthetic configuration that reproduces the problem;
- whether cached and stateless execution differ.

Do not attach model weights, access tokens, private prompts, or other secrets.
Large acceptance failures should include hashes and compact output diagnostics
instead of the artifacts themselves.
