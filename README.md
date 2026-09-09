# Burn transformer architectures

`transformers` is a backend-generic, inference-first library for modern text
transformers built directly on [Burn](https://burn.dev/). It separates reusable
mathematics from checkpoint-family wiring so attention, recurrent mixers,
mixture-of-experts layers, position encodings, caches, loading, and generation
can be tested independently and recombined in experimental models.

This crate is pre-release. **No checkpoint is currently advertised as fully
supported.** Qwen3.5-0.8B, Qwen3.8-27B, Qwen3-30B-A3B,
DeepSeek-V2-Lite, Muse-Glimmer-30B, FLAN-T5-small, FLAN-T5-XXL, and Nemotron
3.5 Lightning have passed substantial real-weight bring-up checks, but each
still has open release gates. FLAN-UL2 remains pending a real-weight run.
Configuration parsing or a runnable synthetic graph is not a support claim.

The pinned architecture research and release criteria live in
[ARCHITECTURES.md](ARCHITECTURES.md). The narrower
[configuration compatibility matrix](ARCHITECTURE_COMPATIBILITY.md) records
which upstream schemas can be represented without implying weight or inference
acceptance. The bounded first-release usage contract for the smallest Qwen
checkpoint is documented separately in
[Qwen3.5-0.8B v0.1 profile](QWEN3_5_0_8B_RELEASE.md).

## Current status

| Runtime | Implemented surface | Real-checkpoint status |
| --- | --- | --- |
| Qwen3.5 / Qwen3.8 hybrid | DeltaNet/full-attention schedules, gated GQA, partial RoPE, SwiGLU, external recurrent/KV caches, batch-one MTP speculation, and strict text-tower mapping | [Qwen3.5-0.8B](acceptance/qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17.json) passed historical reference-logit, 32-token greedy, cached-parity, and packed-Q4 smokes. Its reviewed 186-matrix Q4/G64 v2 sidecar is source/plan bound and passed stateless/cached Metal execution at cosine `1.0`; Q4 quality remains experimental. Its independently authenticated stable-prompt oracle now passes Metal F16: all 32 cached and stateless greedy decisions match, and minimum full-vocabulary reference cosine is `0.999996242`. Its [populated Q4 32K stress](acceptance/measurements/qwen3.5-0.8b-q4-32k-stress-20260909.json) now passes finite-state, overflow-atomicity, and reset checks; this is execution evidence, not long-context quality. [Qwen3.8-27B](acceptance/qwen3.8-27b-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.json) strictly applied all 866 text/MTP tensors, matched the official F16 full-vocabulary logits at `0.999999881` cosine and all 32 greedy tokens, and passed mixed Q8/F16 and Q4/F16 stateless, cached, MTP, and recurrent-decode smokes. Neither is advertised: public API hardening, release coverage, quantized corpus quality, and standardized performance remain open; Qwen3.8 still needs real 32K stress. |
| Qwen3 MoE | GQA with Q/K normalization, sparse selected-expert dispatch, top-k routing, and cached decode | [Qwen3-30B-A3B](acceptance/qwen3-30b-a3b-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39.json) strictly applied all 18,867 BF16-derived F16 tensors and passed six-token F16, mixed Q8/F16, and mixed Q4/F16 Metal stateless, cached-prefill, hidden-trace, and recurrent-decode smokes. Both packed decode audits observed all 1,345 architecture-declared matmuls without whole-weight dequantization. Official reference logits, 32-token generation, quantized quality, standardized performance, and 32K stress remain open; not advertised. |
| DeepSeek V2 | YaRN, MLA reference prefill, compressed latent-cache absorbed decode, routed and shared experts | [DeepSeek-V2-Lite](acceptance/deepseek-v2-lite-604d5664dddd88a0433dbae533b7fe9472482de0.json) strictly applied all 5,291 BF16-derived F16 tensors and passed a six-token Metal stateless, reference-prefill, compressed-cache, and absorbed-decode smoke. Cached/stateless cosine was `1.0`; absorbed decode matched the internal stateless path at `0.999998964`. This is internal Burn-path parity, not comparison with the official implementation. Official reference logits, 32-token generation, quantization, and 32K stress remain open; not advertised. |
| Muse Glimmer | Sliding/full hybrid attention, RoPE/NoPE schedule, sandwich norms, output gates, final-logit softcap, and exact eager-F16 attention with a bounded query-chunk policy | The exact [Muse-Glimmer-30B artifact](acceptance/muse-glimmer-30b-a4e59da52a7bc87ae7251dd5545c0dd437c44b68.json) authenticated both weight shards and a seven-file schema-3 official oracle, then strictly applied all 627 text tensors as F16 on Metal with no unexplained load diagnostics. The real graph matched official prompt, cached-prefix, teacher-prefix, and hidden-trace fixtures with minimum official/Burn cosine `0.999943377` and reproduced all 32 greedy tokens. Real Q8/G64 and Q4/G64 sidecars have both passed create-only conversion, full reopen, and mixed F16/packed Metal inference; each traced decode launched all 417 packed projections with no whole-weight dequantization. A synthetic 32-head F16 Metal envelope exercised actual 32K keys, terminal RoPE, and full/mirrored cache boundaries with a reduced four-channel head; it is not full-model 32K evidence. Quantized reference/quality, a real 32K stress run, real-checkpoint padded/context-boundary coverage, and standardized performance remain open; not advertised. |
| T5 / FLAN / UL2 | Bidirectional encoder, causal decoder, cross-attention cache, relative-position bias, shared encoder/decoder embeddings with a configurable tied or independent LM head, and cached beam search | [FLAN-T5-small](acceptance/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.json) now has a content-authenticated Transformers 5.2/CPU F32 oracle. Flex and Metal both strictly apply all 190 tensors, match complete encoder, decoder-hidden, first-logit, 32-row cached and teacher-forced, padded-batch, and 512-token boundary vectors with minimum cosine `0.999999999999` and zero `1e-4` tolerance failures, and reproduce all 32 greedy decisions plus the exact four-beam continuation. A [repeated Metal timing series](acceptance/measurements/flan-t5-small-metal-benchmark-20260909-run.json) now covers 32/128/512-token sources with 128 forced decode steps. Its record remains unadvertised while crate-wide coverage, per-phase memory, final replay, and coordinated v0.1 release gates remain open. [FLAN-T5-XXL](acceptance/flan-t5-xxl-ae7c9136adc7555eeccc78cdd960dfd60fb346ce.json) strictly applied all 558 runtime tensors on Metal and matched official Transformers/MPS F32 encoder, decoder-hidden, first-logit, 32-row cached and teacher-forced fixtures at minimum cosine `0.999999999998`, plus exact greedy and four-beam tokens. XXL is not advertised; UL2 is pending. |
| Nemotron-H | Interleaved Mamba2, GQA, dense/MoE blocks, ReLU-squared experts, shared experts, and next-token-head mapping | The exact [Nemotron 3.5 Lightning artifact](acceptance/nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json) authenticated and strictly applied all 6,513 tensors, including 270 MTP tensors, then passed Metal F16 stateless, cached-prefill, and one-token recurrent base-model execution. Stateless and prefill hidden traces were bit-identical; recurrent and teacher-forced logits had cosine `0.999376514` and identical top-1. The exact one-stage MTP body and transactional batch-one speculator pass exhaustive tiny Flex tests. This remains internal-consistency evidence: official base parity, real-weight MTP parity, 32-token generation, quantization, and 32K are open; not advertised. |
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

### FLAN-T5-small loading and generation

The [complete FLAN-T5-small example](examples/flan_t5_small.rs) loads the
pinned official SafeTensors checkpoint through `AutoSeq2SeqLm`, tokenizes one
instruction locally, encodes its source once, and runs cache-aware four-beam
decoding. No Python environment or network access is used at inference time:

```sh
cargo run --release --manifest-path crates/transformers/Cargo.toml \
  --no-default-features --features std,flex,tokenizers \
  --example flan_t5_small -- /Volumes/models/flan-t5-small \
  "translate English to German: The house is wonderful."
```

`generate_t5_beam_search` reads decoder-start, EOS, and PAD IDs from the loaded
T5 configuration. It allocates decoder state for `source_batch * num_beams`,
keeps encoder computation outside the token loop, and reorders decoder self-KV,
encoder rows, and immutable cross-attention KV together whenever beams change
parents. Set `num_beams = 1` for deterministic greedy decoding through the same
cached coordinator.

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
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
};

fn generate<B: Backend>(root: &Path, device: &B::Device) -> Result<String, Box<dyn Error>> {
    let options = AutoLoadOptions {
        checkpoint: LoadOptions {
            // The accepted M3 Metal profile streams BF16 source tensors into F16.
            dtype_policy: DTypePolicy::Cast(TensorDType::F16),
            ..LoadOptions::default()
        },
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
        .checked_add(generation.max_new_tokens.saturating_sub(1))
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

A complete macOS chat program is available as
[`examples/qwen35_chat.rs`](examples/qwen35_chat.rs). It uses
`LocalTokenizer::encode_chat` so the checkpoint template and tokenizer
post-processor cannot insert special delimiters twice.

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

Automatic packed loading is enabled for `qwen3_5_text`, `qwen3_moe`, and
`muse_glimmer_text`.
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

Muse-Glimmer's exact 627-tensor F16 load plan classifies 417 matrices for
Burnpack (all five attention projections, all three SwiGLU projections in each
of 52 layers, and the independent LM head) and retains the embedding plus 209
normalization vectors. Every eligible input width aligns to G64. A nonzero
two-layer fixture has passed Q8 and Q4 streaming conversion, automatic mixed
loading, stateless/prefill/recurrent parity, complete dtype accounting, and a
negative gate proving row-unaligned artifacts remain reference-only. The real
Q8/G64 conversion then revalidated all 417 payloads and published a
`28,165,645,337`-byte v2 artifact with SHA-256
`91cfd35bfe0a389cbc2eed2189c4e33312bdce1a0ff50d8452acf06d00f0e8eb`;
the strict conversion/provenance gate took `1,570.845 s`. A separate traced
Metal gate reauthenticated that sidecar and its source provenance, then ran
stateless, cached-prefill, and recurrent decode. Its cached/stateless cosine was
`1.0`, and the decode launched all 417 packed Q8 matmuls with no whole-weight
dequantization. The real
Q4/G64 conversion has now also published and fully reopened its
`14,911,372,485`-byte v2 artifact with SHA-256
`05b69f67799b4dc0228b4298e39d466d753fcc5d6484f61c8920c551017daed3`;
its separate traced Metal gate likewise ran all 417 packed Q4 matmuls without
whole-weight dequantization and reported cached/stateless cosine `1.0`.
Quantized official-reference and quality evaluation plus full-model 32K
execution remain pending; Muse is not advertised as supported.

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
