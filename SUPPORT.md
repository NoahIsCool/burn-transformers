# Support policy

This document defines what a support claim means for the `transformers` crate.
It does not broaden the [model status page](ARCHITECTURE_COMPATIBILITY.md) or
any committed acceptance record.

## Version compatibility

Version 0.1 requires Rust 1.92 or newer and pins Burn 0.21. Before 1.0, a minor
crate release may contain source-breaking API changes. Patch releases preserve
documented public APIs except where a change is necessary to correct unsound,
unsafe, or materially incorrect behavior.

Default builds enable `std` and Burn Flex. Hardware backends and optional
tokenization or template support are selected with crate features. Enabling a
backend proves that its code compiles; real-checkpoint execution is supported
only where the checkpoint's acceptance record names that backend and precision.

### Host requirement of the `cpu` feature

The `cpu` feature (Burn's CubeCL MLIR backend) has a build-host requirement
that the other features lack. Unless it is already installed, its build
scripts download a prebuilt LLVM 20 (tracel-llvm `20.1.4-7`) and run its
`llvm-config`. The bundler keeps the downloaded archive and its checksum file
in `$HOME/.cache/tracel`, a path it derives from the home directory alone, so
`XDG_CACHE_HOME` does not move it; it extracts the LLVM (about 430 MB on
macOS) into the platform's local data directory: `$XDG_DATA_HOME/tracel`, by
default `~/.local/share/tracel`, on Linux and
`~/Library/Application Support/tracel` on macOS. To keep both off a
quota-limited home directory, set `HOME` (and `XDG_DATA_HOME`, if it is set)
to other storage for the build. The bundled `llvm-config` needs glibc 2.34 or
newer and libstdc++ from GCC 11 or newer (`GLIBCXX_3.4.29`): Ubuntu 22.04 or
newer and RHEL/AlmaLinux 9 or newer qualify; RHEL/AlmaLinux 8
(glibc 2.28, GCC 8), still common on HPC clusters, does not. There the build
stops in `tracel-mlir-rs-macros` or `tracel-tblgen-rs`, before this crate
compiles, with loader errors such as `` version `GLIBC_2.34' not found `` and
`` version `GLIBCXX_3.4.29' not found ``. The bundler sets
`MLIR_SYS_200_PREFIX` and `TABLEGEN_200_PREFIX` itself before reading them,
so pointing those variables at another LLVM has no effect.

No library code depends on `cpu`; it gates tests only: the CubeCL CPU
validation below (`cpu` modules in `tests/backend_parity.rs`, the three
`tests/tiny_goldens_*.rs` suites, and `src/moe/router_tests.rs`), and three
test reports that record whether it is enabled (`cfg!(feature = "cpu")` in
`tests/muse_glimmer_artifacts.rs`, `tests/qwen35_benchmarks.rs`, and
`tests/support/benchmark_identity.rs`). On such a host, omit the feature:
lint with
`--no-default-features --features std,flex,cuda,rocm,tokenizers,chat-templates,autodiff,train`,
which compiles the same crate code, or build inside a container with a newer
userland (Singularity or Apptainer with an Ubuntu 22.04 image, for example).
The `cpu` build itself is then checked only where the requirement holds, such
as CI's `ubuntu-latest` job.

### CubeCL CPU backend validation

The `cpu` feature is a test backend, not a supported inference backend: no
acceptance record names it, and no real checkpoint, reduced precision,
quantized format, or performance figure has been measured on it. It earns its
place in the test suite because it compiles the same `burn-cubecl` kernels as
Metal, CUDA, and ROCm to host code, so their layouts, reshapes, fused
attention, cache writes, and reductions can be checked without a GPU.

What was validated, on an Apple M3 Max (macOS 27, Burn 0.21.0, tracel-llvm
`20.1.4-7`, debug build), with the commands in
[RELEASE_GATES.md](RELEASE_GATES.md#checks-added-since-v01) ("CubeCL CPU parity"):

- every executable family of `tests/backend_parity.rs` except Nemotron-H and
  the two modular causal decoders matches Flex/F32 with identical weights in
  stateless, ragged left-padded, and cached prefill-plus-decode logits within
  the F32 bound `2e-5 + 2e-5 · scale` (largest deviation `6.7e-6` at logit
  scale `37.9`, largest relative `9.2e-7`), and the gated delta rule's
  kernels within `6e-8`;
- the tiny Transformers 5.2.0 goldens of the Llama family (seven cases),
  Gemma 2 and Gemma 3 text, and gpt-oss match within `1e-5 + 1e-5 ·
  magnitude` (largest deviations `2.3e-6` for logits and `5.7e-6` for hidden
  states), and every case's six greedy steps are decisive and match;
- the MoE router's top-k returns in-range, distinct expert IDs for NaN and
  `-inf` scores on this backend, which shares Metal's reduction kernels.

Not validated: Burn 0.21's CubeCL CPU backend copies a reshape of a transposed
or expanded view at the wrong coordinates (fixed in Burn 0.22;
[docs/upstream/15](docs/upstream/15-cubecl-cpu-reshape-copy.md)). Nemotron-H's
group-limited router, the rotary de-interleave of multi-head latent attention,
and a masked gated DeltaNet whose key heads are shared by several value heads
meet it, and those cases deviate from Flex by up to `0.58`; any other graph
that reshapes such a view can return wrong values on this backend. The crate
does not work around it. CI's `cpu-parity` job runs the passing cases on
`ubuntu-latest`; until it has completed a run, the measurements above are this
Mac's only.

## Checkpoint support

A model-family implementation is not by itself a checkpoint support claim. A
checkpoint is supported only when its exact repository revision has a committed
record under `acceptance/` with `advertised_supported` set to `true`. That record
is authoritative for the validated dtype, backend, tokenizer, context length,
generation behavior, and any documented limitations.

Revisions, repacks, quantization formats, and third-party conversions not named
by such a record are experimental even when their tensor names happen to load.
Quantized support is tracked independently from full-precision support. The
same holds for families whose real-checkpoint parity has been measured but not
recorded as accepted, such as the Llama, Gemma, and gpt-oss checkpoints listed
on the [model status page](ARCHITECTURE_COMPATIBILITY.md#real-checkpoint-evidence):
they are experimentally validated, not supported.

Version 0.1 promotes exactly three profiles on the validated Apple M3 Max host:

- Qwen3.5-0.8B, revision `2fc06364715b967f1860aea9cf38778875588b17`,
  Metal/F16 parameters with FP32 recurrent state, base text graph only.
- FLAN-T5-small, revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`,
  Flex/F32 and Metal/F32, including cached greedy and beam generation.

The [release record](acceptance/measurements/v0_1-release-20260910.json) and
each checkpoint's `supported_profiles` describe the tested boundaries and
known limits. Hardware, dtype, quantization, MTP or long-context quality claims
must not be inferred from a family-wide implementation. Qwen's historical
approximately 25% throughput slowdown is accepted as an unresolved limitation;
the user approved the controlled September 10 baseline for future comparisons,
not a claim that the historical regression was explained or repaired.

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
