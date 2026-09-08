# Burn transformer architectures

`transformers` is a backend-generic, inference-first library for modern text
transformers built directly on [Burn](https://burn.dev/). It separates reusable
mathematics from checkpoint-family wiring so attention, recurrent mixers,
mixture-of-experts layers, position encodings, caches, loading, and generation
can be tested independently and recombined in experimental models.

This crate is pre-release. **No checkpoint is currently advertised as fully
supported.** Qwen3.5-0.8B, Qwen3.8-27B, Qwen3-30B-A3B,
DeepSeek-V2-Lite, Muse-Glimmer-30B, FLAN-T5-small, and FLAN-T5-XXL have passed
substantial real-weight bring-up checks, but each still has open release gates.
The remaining acceptance checkpoints are pending real-weight runs.
Configuration parsing or a runnable synthetic graph is not a support claim.

The pinned architecture research and release criteria live in
[ARCHITECTURES.md](ARCHITECTURES.md). The narrower
[configuration compatibility matrix](ARCHITECTURE_COMPATIBILITY.md) records
which upstream schemas can be represented without implying weight or inference
acceptance.

## Current status

| Runtime | Implemented surface | Real-checkpoint status |
| --- | --- | --- |
| Qwen3.5 / Qwen3.8 hybrid | DeltaNet/full-attention schedules, gated GQA, partial RoPE, SwiGLU, external recurrent/KV caches, batch-one MTP speculation, and strict text-tower mapping | [Qwen3.5-0.8B](acceptance/qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17.json) passed reference-logit, 32-token greedy, cached-parity, and packed-Q4 gates. [Qwen3.8-27B](acceptance/qwen3.8-27b-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.json) strictly applied all 866 text/MTP tensors, matched the official F16 full-vocabulary logits at `0.999999881` cosine and all 32 greedy tokens, and passed mixed Q8/F16 and Q4/F16 stateless, cached, MTP, and recurrent-decode smokes. Neither is advertised: quantized corpus quality, standardized throughput and output hashing, and 32K stress gates remain open. |
| Qwen3 MoE | GQA with Q/K normalization, sparse selected-expert dispatch, top-k routing, and cached decode | [Qwen3-30B-A3B](acceptance/qwen3-30b-a3b-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39.json) strictly applied all 18,867 BF16-derived F16 tensors and passed six-token F16, mixed Q8/F16, and mixed Q4/F16 Metal stateless, cached-prefill, hidden-trace, and recurrent-decode smokes. Both packed decode audits observed all 1,345 architecture-declared matmuls without whole-weight dequantization. Official reference logits, 32-token generation, quantized quality, standardized performance, and 32K stress remain open; not advertised. |
| DeepSeek V2 | YaRN, MLA reference prefill, compressed latent-cache absorbed decode, routed and shared experts | [DeepSeek-V2-Lite](acceptance/deepseek-v2-lite-604d5664dddd88a0433dbae533b7fe9472482de0.json) strictly applied all 5,291 BF16-derived F16 tensors and passed a six-token Metal stateless, reference-prefill, compressed-cache, and absorbed-decode smoke. Cached/stateless cosine was `1.0`; absorbed decode matched the internal stateless path at `0.999998964`. This is internal Burn-path parity, not comparison with the official implementation. Official reference logits, 32-token generation, quantization, and 32K stress remain open; not advertised. |
| Muse Glimmer | Sliding/full hybrid attention, RoPE/NoPE schedule, sandwich norms, output gates, final-logit softcap, and exact eager-F16 attention with a bounded query-chunk policy | The exact [Muse-Glimmer-30B artifact](acceptance/muse-glimmer-30b-a4e59da52a7bc87ae7251dd5545c0dd437c44b68.json) authenticated both weight shards and a seven-file schema-3 official oracle, then strictly applied all 627 text tensors as F16 on Metal with no unexplained load diagnostics. The real graph matched official prompt, cached-prefix, teacher-prefix, and hidden-trace fixtures with minimum official/Burn cosine `0.999943377` and reproduced all 32 greedy tokens. Quantized quality, a real 32K stress run, real-checkpoint padded/context-boundary coverage, and standardized performance remain open; not advertised. |
| T5 / FLAN / UL2 | Bidirectional encoder, causal decoder, cross-attention cache, relative-position bias, shared encoder/decoder embeddings with a configurable tied or independent LM head, and cached beam search | [FLAN-T5-small](acceptance/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.json) passed strict F32 loading, cached parity, and a beam fixture on Flex. [FLAN-T5-XXL](acceptance/flan-t5-xxl-ae7c9136adc7555eeccc78cdd960dfd60fb346ce.json) strictly applied all 558 runtime tensors on Metal and matched official Transformers/MPS F32 encoder, decoder-hidden, first-logit, 32-row cached and teacher-forced fixtures at minimum cosine `0.999999999998`, plus exact greedy and four-beam tokens. Neither is advertised; XXL still needs padded/context-boundary, quantized quality/32K, and standardized performance gates. UL2 is pending. |
| Nemotron-H | Interleaved Mamba2, GQA, dense/MoE blocks, ReLU-squared experts, shared experts, and next-token-head mapping | Synthetic golden, cache, and checkpoint-contract tests only; Nemotron 3.5 Lightning real weights are pending. |
| Custom modular models | Serializable causal and seq2seq specifications using full/sliding attention, causal DeltaNet, dense gated FFNs, sparse MoE, configurable normalization, and cross-attention | Intended for controlled experiments, not compatibility with an upstream checkpoint. Custom MLA, Mamba2, bidirectional DeltaNet/sliding attention, and learned gated residuals currently fail with explicit capability errors. |

Qwen MTP speculative generation currently requires batch size one and defaults
to two draft tokens per cycle; callers may configure a different positive draft
count. Ordinary Qwen inference retains rectangular batched execution.

## Design

Family code under `src/models` translates exact upstream configuration, layer
schedules, and tensor names. Shared modules own the reusable operations:

- `attention`, `delta_net`, `feed_forward`, `moe`, `normalization`, and
  `position` implement transformer mathematics without model-name branches.
- `cache` provides preallocated KV, sliding-ring, recurrent, cross-attention,
  and transactional state. Cache tensors are external to model parameters.
- `checkpoint` discovers and validates local SafeTensors, builds strict mapping
  plans, streams tensors into Burn modules, and reports every transformation or
  omission.
- `generation` provides cached causal generation and deterministic cached T5
  beam search. Sampling supports temperature, top-k, top-p, repetition and
  presence penalties, deterministic seeds, EOS/PAD handling, token stop
  sequences, per-token log probabilities, and synchronous streaming callbacks.
- `models::custom` constructs experimental causal or encoder-decoder graphs
  from serializable topology specifications.

The principal contracts are `CausalLanguageModel` and `Seq2SeqLanguageModel`.
Both expose a stateless full-precision `forward` path and explicit
`new_decode_state`, prefill, and decode operations. `DecodeState` exposes
capacity, per-sequence lengths, reset, reorder/duplication, and truncation.
Inputs use `[batch, sequence]` token IDs and optional explicit positions; a
`TokenMask` always means `true == valid token`. `ForwardOptions` projects only
the last valid position by default and makes hidden states, attention data,
router logits, and auxiliary losses opt-in.

`AutoCausalLm` dispatches `deepseek_v2`, `muse_glimmer_text`, `nemotron_h`,
`qwen3_5_text`, and `qwen3_moe`. `AutoSeq2SeqLm` dispatches `t5`. These enum
registries contain executable Burn modules; their presence says nothing about
whether a particular real checkpoint has completed the release gates above.

## Build features

The crate declares Rust 1.92 as its minimum supported Rust version and pins Burn
0.21.

| Feature | Effect |
| --- | --- |
| `default` | Enables `std` and the deterministic `flex` backend used by ordinary tests. |
| `std` | Enables Burn's standard-library support. |
| `flex` | Enables Burn Flex. |
| `cpu` | Enables Burn's CPU backend. |
| `metal` | Enables Burn Metal. |
| `cuda` | Enables Burn CUDA. |
| `rocm` | Enables Burn ROCm. |
| `autodiff` | Enables Burn autodiff for full-precision stateless graphs. Training loops are not included. |
| `tokenizers` | Adds local `tokenizer.json` loading through the `tokenizers` crate. |
| `chat-templates` | Adds bounded local chat-template rendering with MiniJinja and implies `tokenizers`. |

For example, a consumer selecting only Metal and local tokenization can use:

```toml
[dependencies]
burn = { version = "=0.21.0", default-features = false, features = ["metal"] }
transformers = { path = "crates/transformers", default-features = false, features = ["std", "metal", "tokenizers"] }
```

Packed Metal Q4/Q8 has a stricter feature boundary; see
[Quantization](#quantization).

## Local artifacts only

The Rust runtime never contacts Hugging Face, executes remote Python, or
downloads missing files. Download an immutable revision with an external tool,
then pass the resulting directory to `load_local`. For example:

```sh
hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-dir /Volumes/models/qwen3.5-0.8b

hf download google/flan-t5-small \
  --revision 0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab \
  --local-dir /Volumes/models/flan-t5-small
```

A local directory must contain `config.json` and either an official
`model.safetensors` or a contained SafeTensors index and all of its shards.
`tokenizer.json` and tokenizer metadata are optional unless the application
needs tokenization. Discovery rejects escaping shard paths and symlinks,
duplicates, mapping collisions, unsupported dtypes, invalid byte ranges,
missing shards, and shape mismatches before parameter application.

`AutoLoadOptions` controls strict checkpoint policy, tokenizer attachment,
normalized generation defaults, provenance metadata, Qwen MTP loading, and an
optional quantized Burnpack sidecar. A successful call returns `LoadedModel`,
which keeps the executable module, tokenizer, generation defaults, canonical
source metadata, and finalized `LoadReport` together.

### Causal loading and generation

With the `tokenizers` feature enabled, the same API works for any backend that
can execute the selected architecture:

```rust
use std::{error::Error, io, path::Path};

use burn::tensor::backend::Backend;
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalGenerationRequest, EncodeOptions,
    generate_causal,
    cache::CacheConfig,
};

fn generate<B: Backend>(root: &Path, device: &B::Device) -> Result<String, Box<dyn Error>> {
    let options = AutoLoadOptions {
        repository: Some("Qwen/Qwen3.5-0.8B".into()),
        revision: Some("2fc06364715b967f1860aea9cf38778875588b17".into()),
        ..AutoLoadOptions::default()
    };
    let loaded = AutoCausalLm::<B>::load_local(root, &options, device)?;
    let tokenizer = loaded
        .tokenizer()
        .ok_or_else(|| io::Error::other("build with the tokenizers feature"))?;
    let prompt = tokenizer
        .encode("Hello", &EncodeOptions::default())?
        .to_causal_input::<B>(device)?;
    let [batch, prompt_length] = prompt.token_ids.dims();

    let mut generation = loaded.generation_defaults().clone();
    generation.max_new_tokens = 32;
    let capacity = prompt_length
        .checked_add(generation.max_new_tokens)
        .ok_or_else(|| io::Error::other("context capacity overflow"))?;
    let output = generate_causal(
        loaded.model(),
        CausalGenerationRequest {
            prompt,
            cache: CacheConfig::new(batch, capacity)?,
            generation,
            prefill_chunk_size: None,
            vocabulary_size: loaded.model().config().vocabulary_size(),
        },
    )?;
    let sequence = output
        .sequences
        .first()
        .ok_or_else(|| io::Error::other("model returned no sequence"))?;
    Ok(tokenizer.decode(&sequence.token_ids, true)?)
}
```

Applications that need resumable sessions can bypass the coordinator and call
`prefill` and `decode` directly with architecture-matched external state. T5
callers can use `generation::generate_seq2seq_beam_search`, which encodes once
and reorders both decoder and cross-attention state as beams change parents.
For long causal prompts, set `prefill_chunk_size` to a bounded value such as
`Some(256)`: the coordinator advances the same preallocated state segment by
segment and retains the last valid logits independently for every ragged batch
row. Leaving it as `None` preserves a single-call prefill for short prompts and
reference comparisons.

### Building an experimental topology

`DecoderSpec` and `Seq2SeqSpec` derive Serde traits, so an experiment can retain
its exact architecture as JSON and initialize it on any Burn backend:

```rust
use burn::tensor::backend::Backend;
use transformers::{
    Result,
    architecture::DecoderSpec,
    models::custom::ModularCausalLm,
};

fn build_experiment<B: Backend>(
    spec: DecoderSpec,
    device: &B::Device,
) -> Result<ModularCausalLm<B>> {
    ModularCausalLm::init(spec, device)
}
```

The modular runtime validates the complete specification before allocating
parameters. Its current decode state is a correctness-oriented recomputation
path; family runtimes retain the optimized native KV, latent, ring, and
recurrent caches used for performance work.

## Quantization

The native format is deterministic, symmetric, weight-only Q8 or Q4 with
contiguous groups of 32, 64, or 128 values (64 by default). Conversion follows
the validated architecture mapping and streams one transformed tensor at a
time into a memory-mapped Burnpack sidecar. The sidecar is **not** a standalone
checkpoint: excluded parameters are still loaded from the original
SafeTensors. Embeddings, norms, biases, router classifiers, recurrent scalars
and convolutions, MLA latent expansion, and first-generation MTP parameters
remain in source precision.

Automatic packed loading is enabled for `qwen3_5_text` and `qwen3_moe`.
Qwen3.5-0.8B Q4 has passed a finite full-model Metal smoke test and exact
cached/stateless consistency, but its first token differs from F16 and the
quality corpus has not passed. Qwen3.8-27B Q8 and Q4 have each passed strict
streaming conversion and a clean single-backend Metal smoke: 497 matrices were
packed in groups of 64, 369 tensors remained outside each sidecar and loaded as
F16, and both mixed graphs exercised stateless, cached, MTP, and recurrent
paths. The F16 graph now matches the pinned official full-vocabulary logits and
all 32 stable-margin greedy tokens, but quantized reference/longer-generation
parity, the quality corpus, standardized throughput and output hashing, and 32K
stress remain open. Neither format is advertised for production inference yet.

Qwen3-30B-A3B Q8 and Q4 have also passed provenance-bound streaming
conversion and mixed F16/packed Metal stateless, cached-prefill, and recurrent
decode smokes. Each sidecar contains 18,625 packed matrices; 242 embeddings,
norms, and routers remain F16, and runtime MoE dispatch evaluates only selected
experts. The Q4 decode audit observed all 1,345 expected packed matmuls and no
whole-weight dequantization; the Q8 audit proved the same 1,345-of-1,345 packed
launch result. Official-reference and 32-token generation parity, quantized
quality, standardized performance, and 32K stress remain open, so the model is
not yet advertised as fully supported.

Burn 0.21 can execute compatible Burnpack matrices through packed `QFloat`
matmul. The only certified packed-Metal build shape is an exact single-backend
build such as:

```sh
cargo build --manifest-path crates/transformers/Cargo.toml \
  --release --no-default-features --features std,metal,tokenizers
```

Do not include `rocm` in that Metal binary. Burn 0.21 feature-unifies CubeCL
autotuning from ROCm, which can select invalid packed Q4/Q8 Metal GEMV
candidates. The crate rejects the known `metal + rocm` packed path before
dispatch. A downstream dependency can also enable CubeCL autotuning directly;
that untracked combination is not certified.

## Explicit non-goals for this release

- Vision or audio towers, even when their tensors are present in a checkpoint.
- GGUF, AWQ, GPTQ, external FP8 formats, or importing third-party quantized
  artifacts.
- Training loops, optimizers, datasets, or quantized training. Full-precision
  stateless graphs remain the future training surface.
- A CLI, HTTP server, continuous batching, tensor/model parallelism, or
  distributed execution.
- Claims beyond 32K context on the current acceptance hardware.
- Silent approximation of an unavailable architecture feature. Unsupported
  combinations return contextual errors.

## Validation and performance

Ordinary tests use Flex; hardware tests and real-weight tests are opt-in and
ignored unless their pinned local artifacts are supplied. Benchmark results
must separate cold compilation/loading from warm prefill and decode and must
include checkpoint provenance. See [BENCHMARKS.md](BENCHMARKS.md) for the
measurement protocol and the limited smoke evidence recorded so far.
