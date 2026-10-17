# transformers

Modern transformer architectures for [Burn](https://burn.dev), built for
experiments. Load a pinned checkpoint or build a model from a serializable
spec, then inspect and edit any named activation, change attention rules and
topology, run the stack in stages, and train with native Burn. The same code
runs batched cached inference, and its numerics are checked against pinned
reference implementations, from tiny Transformers-generated goldens to opt-in
real-checkpoint gates.

- **One experiment interface for every model.** The `HookedModel` trait
  captures, ablates, patches, and steers activations on every built-in family,
  on a model loaded from disk, and on your own compositions; `StagedModel`
  splits a causal stack into resumable stages (T5 and modular encoder-decoder
  models expose it on their encoder and decoder stage views).
- **Shared, model-independent mechanisms.** Grouped-query, latent, sliding,
  chunked-local, differential, and sink attention; every Hugging Face RoPE
  rule; Gated DeltaNet and Mamba-2 mixers; dense, gated, clamped, and sparse
  feed-forwards with configurable routing. Families are thin wiring over them.
- **Efficient inference.** Genuinely batched cached prefill and decoding for
  ragged batches, native-KV-head attention, sliding-window ring caches,
  score-bounded long-context attention (Muse Glimmer, DeepSeek-V2, and custom
  code), and packed Q4/Q8 matmuls on Metal.
- **Verified where it matters.** More than 2,500 tests, reference goldens generated
  with Hugging Face Transformers, property tests, twelve fuzz targets, and 25
  Verus harnesses that prove the cache, planning, decoding, and checkpoint
  arithmetic overflow-free and in bounds.

This is a Rust 2024 crate (MSRV 1.92) pinned to Burn 0.21. The package version
is 0.1.0; [support](#model-families-and-status) is deliberately narrower than
what is implemented.

## Quickstart

Run commands from the repository root. Building needs no model download.

### 1. See it work

The workbench builds tiny random models, captures and ablates neurons, inserts
an adapter, enables bidirectional attention, and runs one family-generic
experiment on a modular model, GPT-2, and a Llama-family model:

```sh
cargo run --release --locked --example activation_workbench
```

### 2. Load a checkpoint and generate

The runtime reads local files only. Download an immutable revision with any
external tool, then point an example at the directory:

```sh
hf download HuggingFaceTB/SmolLM-135M \
  --revision 1d461723eec654e65efdc40cf49301c89c0c92f4 --local-dir /models/smollm-135m
cargo run --release --locked --features tokenizers --example llama_generate -- \
  /models/smollm-135m "The capital of France is" 24
```

In your own code, `AutoCausalLm` selects the family from `config.json`, loads
every tensor strictly, and returns the model with its tokenizer and generation
defaults (enable the `tokenizers` feature):

```rust
use std::{error::Error, path::Path};

use burn::backend::Flex;
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalGenerationRequest, EncodeOptions, generate_causal,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
};

type B = Flex<f32, i32>;

fn complete(root: &Path, text: &str) -> Result<String, Box<dyn Error>> {
    let device = Default::default();
    let options = AutoLoadOptions {
        checkpoint: LoadOptions {
            dtype_policy: DTypePolicy::Cast(TensorDType::F32),
            ..LoadOptions::default()
        },
        ..AutoLoadOptions::default()
    };
    let loaded = AutoCausalLm::<B>::load_local(root, &options, &device)?;
    let tokenizer = loaded.tokenizer().ok_or("the directory has no tokenizer.json")?;
    let prompt = tokenizer
        .encode(text, &EncodeOptions::default())?
        .to_causal_input::<B>(&device)?;
    let [batch, prompt_tokens] = prompt.token_ids.dims();

    let mut generation = loaded.generation_defaults().clone();
    generation.max_new_tokens = 24;
    let output = generate_causal(
        loaded.model(),
        CausalGenerationRequest {
            cache: CacheConfig::new(batch, prompt_tokens + generation.max_new_tokens)?,
            prompt,
            generation,
            prefill_chunk_size: None,
            vocabulary_size: loaded.model().config().vocabulary_size(),
        },
    )?;
    Ok(tokenizer.decode(&output.sequences[0].token_ids, true)?)
}
```

The default `DTypePolicy::Auto` keeps a checkpoint's stored float dtype, so a
BF16 checkpoint runs in BF16 even on `Flex<f32>`; `Cast(TensorDType::F32)`, as
above, executes in F32. A load failure is a typed error:
`Error::CheckpointPlan` carries the `CheckpointError` of discovery or strict
planning (for example `ArtifactNotFound`), and `Error::CheckpointExecution`
keeps the partial load report of a failed tensor stream. To load a known family
directly, initialize it from its configuration and call `load_checkpoint` (see
the `checkpoint` module's "Loading a concrete family").

### 3. Capture and edit an activation on any model

Experiments are written once against the `HookedModel` trait. Points have
stable semantic names, such as `decoder.layers.0.residual.post`, and selections
keep their axes, so selecting token 1 of a `[batch, token, channel]` point
yields `[batch, 1, channel]`:

```rust
use burn::tensor::{Tensor, backend::Backend};
use transformers::{
    CausalLmInput, CausalLmOutput, ForwardOptions, Result,
    architecture::HookedModel,
    intervention::{Hooks, Layer},
};

/// Halves token 1's residual stream after block 0; returns the original and the new logits.
fn steer<B: Backend, M>(
    model: &M,
    input: CausalLmInput<B>,
) -> Result<(Tensor<B, 3>, CausalLmOutput<B>)>
where
    M: HookedModel<B, Input = CausalLmInput<B>, Output = CausalLmOutput<B>>,
{
    for point in model.activation_points()? {
        println!("{} {:?}", point.name, point.axes);
    }
    let residual = Layer::decoder(0).residual_post().token(1);
    let mut hooks = Hooks::new();
    let original = hooks.capture(&residual); // registered first: sees the clean value
    hooks.edit(&residual, |x| x * 0.5);
    let steered = model.forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)?;
    Ok((hooks.get(&original)?, steered))
}
```

The function accepts a concrete family, an `AutoCausalLm`, or the
`LoadedModel` returned by `load_local`, unchanged. `StagedModel` adds
`embed`, `run_layers`, and `normalize` for patching residual streams across
runs. The [activation guide](EXPERIMENTS.md) covers selections, patching,
attention policies, custom points, and the per-family point catalogs.

### 4. Build your own model

`DecoderSpec` and `Seq2SeqSpec` describe a topology as data (they derive Serde,
so an experiment can save its exact architecture as JSON). Each block chooses
its own mixer, feed-forward, normalization, and positions:

```rust
use burn::tensor::backend::Backend;
use transformers::{
    Result,
    architecture::{
        Activation, AttentionSpec, DecoderLayerSpec, DecoderSpec, DenseFeedForwardSpec,
        FeedForwardSpec, MixerSpec, NormPlacement, NormalizationSpec, PositionEncodingSpec,
        ResidualTopology, RopeScaling,
    },
    models::custom::ModularCausalLm,
};

/// A two-block grouped-query decoder with RoPE, Q/K norms, and SwiGLU.
fn tiny_decoder<B: Backend>(device: &B::Device) -> Result<ModularCausalLm<B>> {
    let norm = NormalizationSpec::Rms { epsilon: 1e-6, zero_centered: false };
    let block = DecoderLayerSpec {
        mixer: MixerSpec::Attention(AttentionSpec {
            query_heads: 4,
            key_value_heads: 2,
            head_dim: 16,
            query_key_norm: true,
            output_gate: false,
            position: PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: 16,
                scaling: RopeScaling::None,
            },
        }),
        feed_forward: FeedForwardSpec::Dense(DenseFeedForwardSpec {
            intermediate_size: 128,
            activation: Activation::Silu,
            bias: false,
        }),
        mixer_norm: norm,
        feed_forward_norm: norm,
        norm_placement: NormPlacement::Pre,
        residual: ResidualTopology::Add,
    };
    let spec = DecoderSpec {
        vocabulary_size: 256,
        hidden_size: 64,
        layers: vec![block; 2],
        final_norm: norm,
        tie_word_embeddings: true,
        max_position_embeddings: 128,
        embedding_scale: None,
        final_logit_softcap: None,
    };
    // N(0, 0.02^2) weights with GPT-2 depth scaling of the residual writers.
    let policy = spec.scaled_init_policy(0.02);
    ModularCausalLm::init_with(spec, &policy, device)
}
```

The result implements `HookedModel` and `StagedModel`, so `steer` above runs on
it too. Swap a block's mixer for `MixerSpec::AttentionWithOptions` (sliding
or chunked-local connectivity, sinks, soft caps, biases),
`MixerSpec::DifferentialAttention`, `MixerSpec::Mla`, `MixerSpec::DeltaNet`, or
`MixerSpec::Mamba2`; see [Building your own architecture](EXPERIMENTS.md#build-your-own-architecture).
To train it, read the [training guide](TRAINING.md):

```sh
cargo run --release --locked --features autodiff --example t5_classifier -- --train
```

## What is inside

### Experiment interface

- **Typed activation points** with token, head, channel, and batch selections;
  detached or differentiable captures, observers, ordered edits, clean-run
  patches, catalog-driven captures of every point, and custom points in your
  own modules. Failures name the point (`Error::Intervention`).
- **Staged execution**: embed, run any block range, normalize, and resume,
  with positions, padding, and encoder context carried in `HiddenState`.
- **Attention as a choice**: causal, bidirectional, or custom connectivity per
  pass or per layer, plus additive biases and structural pattern transforms.
- **Editable recipes and weight transfer**: change T5 or Qwen block geometry,
  load only a pretrained prefix, and transfer selected tensors with explicit
  shape and tie policies and an auditable report.
- **Explicit initialization**: `InitPolicy` states the random distribution as
  data, including depth-scaled residual projections and lazy application.
- **Native Burn training**: models are ordinary `Module`s; freezing, records,
  optimizers, and the optional Burn `Learner` work unchanged.

### Model families and status

`AutoCausalLm` dispatches the causal families below by `model_type`, and
`AutoSeq2SeqLm` dispatches `t5`. `AutoCausalLm` implements both experiment
traits and `AutoSeq2SeqLm` implements `HookedModel`. A family
being implemented is not a support claim: support requires a committed
acceptance record for an exact checkpoint, backend, and dtype
([SUPPORT.md](SUPPORT.md)). Evidence for every row is on the
[model status page](ARCHITECTURE_COMPATIBILITY.md).

| Family | `model_type` | Validation status |
| --- | --- | --- |
| Qwen3.5 / Qwen3.8 hybrid | `qwen3_5_text` | **Supported (v0.1.0 source):** Qwen3.5-0.8B, Metal/F16, base text; its parity gate passed again at `f4c56c4` ([replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)). Qwen3.8-27B matches official logits on Metal/F16 (experimental). The public example and the Qwen3.8 gates passed again at `8a4404a` ([large-checkpoint replay](acceptance/measurements/large-checkpoint-replay-20261016.json)), and the Qwen3.8 official gate again at `c15f24e` after a loader fix that lowers its Metal peak footprint ([reference-precision follow-ups](acceptance/measurements/reference-precision-followups-20261016.json)). |
| T5 / FLAN / UL2 | `t5` | **Supported (v0.1.0 source):** FLAN-T5-small, Flex/F32 and Metal/F32; both gates passed again at `f4c56c4` ([replay record](acceptance/measurements/real-checkpoint-replay-20261016.json)). FLAN-T5-XXL matches official references on Metal (experimental; passed again at `8a4404a`, [large-checkpoint replay](acceptance/measurements/large-checkpoint-replay-20261016.json)); UL2 pending. |
| Muse Glimmer | `muse_glimmer_text` | Experimental: official-reference parity on Metal/F16 (passed again at `4f9502a`, [large-checkpoint replay](acceptance/measurements/large-checkpoint-replay-20261016.json), and at `c15f24e`, [reference-precision follow-ups](acceptance/measurements/reference-precision-followups-20261016.json)). |
| Llama family | `llama`, `mistral`, `qwen2`, `qwen3`, `smollm3`, `olmo2` | Experimental: tiny Transformers goldens for all six; real-weight Flex and Metal F32 parity for SmolLM-135M, Qwen3-4B, SmolLM3-3B, Llama-3.1-8B, and Qwen2-7B (against an F64 reference; on Metal under the documented `2.5e-3` tolerance). |
| Gemma 2 / Gemma 3 text | `gemma2`, `gemma3_text` (and `gemma3`'s text tower) | Experimental: tiny goldens; gemma-3-4b-it Flex and Metal F32 parity across its sliding window. |
| gpt-oss | `gpt_oss` | Experimental: tiny golden; truncated gpt-oss-20b Flex and Metal F32 parity with exact MXFP4 decoding; full-model Flex/BF16 and Metal/F16 parity against a full F32 reference ([reference-precision follow-ups](acceptance/measurements/reference-precision-followups-20261016.json)). |
| Qwen3-MoE | `qwen3_moe` | Experimental: strict real-weight loading and Metal smokes (passed again at `8a4404a`, [large-checkpoint replay](acceptance/measurements/large-checkpoint-replay-20261016.json)). |
| DeepSeek-V2 | `deepseek_v2` | Experimental: strict real-weight loading and an internal Metal smoke; tiny goldens for V2 and V3-style routing. |
| Nemotron-H | `nemotron_h` | Experimental: strict real-weight loading, an internal Metal smoke, and official base-model parity against the official F32 rows ([reference-precision follow-ups](acceptance/measurements/reference-precision-followups-20261016.json), [analysis](NEMOTRON_H_ACCEPTANCE.md#first-official-comparison)). |
| GPT-2, GPT-NeoX / Pythia | concrete types only | Experimental stateless modules for interpretability studies. |
| Custom modular models | `DecoderSpec`, `Seq2SeqSpec` | For experiments, not checkpoint compatibility. |

The two supported profiles were qualified on the v0.1.0 source. Later work
changed their execution paths; their gates passed again on source `f4c56c4`,
as recorded in the [real-checkpoint replay](acceptance/measurements/real-checkpoint-replay-20261016.json),
and the larger experimental checkpoints were replayed afterwards in the
[large-checkpoint replay](acceptance/measurements/large-checkpoint-replay-20261016.json); see
[v0.1 release qualification](ARCHITECTURE_COMPATIBILITY.md#v01-release-qualification)
and [the required replays](RELEASE_GATES.md#replays-required-after-the-polish-cycle).

### Shared mechanisms

Reusable mathematics lives outside `src/models` and contains no model-name
branches. The [innovation matrix](TRANSFORMER_INNOVATIONS.md) gives each
mechanism's equations, reference, and implementation.

| Area | Mechanisms |
| --- | --- |
| Attention | MHA/GQA/MQA computed on native KV heads; global, sliding-window, and chunked-local connectivity; learned sinks; logit soft caps; Q/K RMSNorm per head or full width; output gates; projection biases; differential attention; multi-head latent attention with absorbed decoding; score-bounded chunked execution. |
| Positions | RoPE with linear, dynamic-NTK, YaRN, Llama 3, LongRoPE, and proportional scaling (one translator for every Hugging Face `rope_type`); partial and multi-axis RoPE; NoPE; T5 relative buckets. Dynamic NTK is translated for every family but rescales only in custom compositions: the built-in families refuse positions beyond `max_position_embeddings`, where it would take effect. |
| Mixers | Gated DeltaNet with block-parallel chunked prefill; Mamba-2 with a vectorized chunk scan; fixed-state recurrent decoding. |
| Feed-forward | SwiGLU/GeGLU, ungated, ReLU², and clamped SwiGLU; sparse experts with one router covering softmax or sigmoid scoring, top-k before or after scoring, group-limited selection, correction bias, renormalization, and shared experts, dispatched in memory-bounded expert blocks. |
| State | Preallocated KV, ring, latent, recurrent, and cross-attention caches outside the parameters; atomic multi-layer step writes; transactional speculative staging ([used by](TRANSFORMER_INNOVATIONS.md#innovation-transactional-cache) DeepSeek-V2, Qwen3.5, and Nemotron-H); reorder, truncate, and reset. |
| Generation | Greedy, temperature, top-k, top-p, min-p, repetition/presence/frequency penalties, logit bias, `min_new_tokens`, no-repeat n-grams, stop sequences, streaming callbacks, cache-aware beam search, and lossless speculative verification. |
| Artifacts | Strict local SafeTensors discovery and streaming, exact dtype casts, exact MXFP4 decoding, native Q4/Q8 Burnpack sidecars, local tokenizers, and sandboxed chat templates with Hugging Face semantics. |

### Performance

- Every family runs cached prefill and decoding as **one model pass per step
  over the whole ragged batch**; rows of different lengths are handled by
  masks and cache segments, not by per-row loops. On Flex CPU with small
  random models and eight rows, batched decoding measured 1.7× to 3.6× faster
  than the earlier row-by-row path for dense and hybrid families and 1.3× to
  2.7× for sparse-expert families; gpt-oss's wide experts are not faster on
  CPU (0.8× to 1.1×). These were measured on a shared host; the
  [changelog](CHANGELOG.md#performance) lists each measurement.
- Cached decoding attends over the stored grouped-query KV heads directly,
  never repeating keys and values per query head.
- Sliding layers keep only their window in ring caches.
- Muse Glimmer and DeepSeek-V2 (MLA) run long attention in query chunks whose
  score planes stay below a configurable `AttentionExecutionPolicy` bound, and
  `GroupedQueryAttention::attend_bounded` offers the same to custom code. The
  other families evaluate each call's full score plane, so for them
  `prefill_chunk_size` is what bounds long causal prompts.
- Input validation and generation bookkeeping read back to the host once per
  step, and speculative drafts are verified in one target pass.
- Flex runs with SIMD kernels and rayon-parallel matrix products.
- On Metal, packed Q4/Q8 Burnpack weights execute through Burn's quantized
  matmul without whole-weight dequantization. Burn 0.21's multi-row Q4
  kernels are wrong with F32 activations, so those products run one row at a
  time (one packed matmul per row); F16 activations keep a single matmul.

Measurement protocols and recorded results are in [BENCHMARKS.md](BENCHMARKS.md).

### Verification

Ordinary tests run on the deterministic Flex backend. Family behavior is
checked against Hugging Face Transformers 5.2 goldens on tiny random-weight
checkpoints; optimized and batched paths are compared with stateless and
row-by-row references; real-checkpoint gates are opt-in. Integer planning in
the cache, attention, mixer, MoE, position, generation, tokenizer, checkpoint,
transfer, and family modules is proved correct in place with
[Verus](verus/README.md). [Fuzz targets](fuzz/README.md)
and property suites cover configuration parsing, RoPE translation, attention
masks, cache transactions, SafeTensors headers, chat templates, and generation.

## Build features

| Feature | Effect |
| --- | --- |
| `default` | Enables `std` and the deterministic `flex` backend used by ordinary tests. |
| `std` | Enables Burn's standard-library support. |
| `flex` | Enables Burn Flex with SIMD kernels and parallel matrix products. |
| `cpu` | Enables Burn's CPU backend. |
| `metal` | Enables Burn Metal. |
| `cuda` | Enables Burn CUDA. |
| `rocm` | Enables Burn ROCm. |
| `autodiff` | Enables Burn autodiff for floating-point stateless graphs and the explicit training examples. |
| `train` | Adds Burn's native Learner/TrainStep framework; implies `std` and `autodiff`. |
| `tokenizers` | Adds local `tokenizer.json` loading through the `tokenizers` crate. |
| `chat-templates` | Adds bounded local chat-template rendering with MiniJinja and implies `tokenizers`. |
| `kernel-tracing` | Test-only. Implies `metal` and Burn tracing so opt-in tests can observe CubeCL kernel launches. |
| `attention-kernel-tracing` | Test-only. Implies `kernel-tracing` and enables CubeCL autotuning (`burn/autotune`) for the fused-attention parity gate; it is not a certified packed-Metal build. |

A consumer in a sibling directory selecting only Metal and local tokenization
can use (adjust the path for your checkout):

```toml
[dependencies]
burn = { version = "=0.21.0", default-features = false, features = ["metal"] }
transformers = { path = "../transformers", default-features = false, features = ["std", "metal", "tokenizers"] }
```

## Local artifacts

The runtime never contacts a model hub, executes remote Python, or downloads
missing files. A model directory must contain `config.json` and either
`model.safetensors` or a SafeTensors index with all of its shards;
`tokenizer.json` and its metadata are needed only for tokenization. Discovery
rejects escaping shard paths and symlinks, duplicates, mapping collisions,
unsupported dtypes, invalid byte ranges, missing shards, and shape mismatches
before any parameter changes. Special-token declarations must resolve to the
tokenizer's vocabulary.

`AutoLoadOptions` controls the strict checkpoint policy, tokenizer attachment,
generation defaults, provenance metadata, Qwen MTP loading, and an optional
quantized sidecar. A successful call returns a `LoadedModel`, which keeps the
model, tokenizer, normalized `generation_config.json` defaults (with any
setting the runtime does not apply listed by
`unsupported_generation_settings()`), source metadata, and the final
`LoadReport` together. `LoadedModel` forwards `HookedModel` and `StagedModel`
to its model.

Other complete programs: [`flan_t5_small`](examples/flan_t5_small.rs) runs
cache-aware four-beam T5 translation, [`qwen35_chat`](examples/qwen35_chat.rs)
renders a checkpoint chat template, and [`gemma_generate`](examples/gemma_generate.rs)
loads Gemma 2 or Gemma 3 text towers.

## Inference API

Every family that `AutoCausalLm` or `AutoSeq2SeqLm` dispatches, and every
modular model, implements one of two contracts from `architecture`, so a
generation loop written against them runs on any of these models (GPT-2 and
GPT-NeoX are stateless study modules without decode state):

- `CausalLanguageModel`: a stateless, differentiable `forward`, plus
  `new_decode_state_for` (a decode state sized by a `CacheConfig`), `prefill`
  (a prompt), and `decode` (one or more new tokens) over external decode
  state.
- `Seq2SeqLanguageModel` (T5): `encode`, a stateless `forward`,
  `new_decode_state_for`, `prefill_decoder`, and `decode`.

Inputs are `[batch, sequence]` token IDs with optional explicit positions and
an optional `TokenMask` in which `true` marks a valid token, so ragged rows can
be left- or right-padded. Cached prefill and decode run the whole batch in one
model pass per step. Decode state lives outside the parameters, and the
`DecodeState` trait exposes its capacity and per-row lengths, `reset`,
`reorder` (gather or duplicate rows, for example beams), and `truncate` (roll
back to earlier lengths). A failed call leaves the lengths and every readable
cache slot unchanged.

```rust
use burn::tensor::{Int, Tensor, backend::Backend};
use transformers::{
    CausalLanguageModel, CausalLmInput, DecodeInput, ForwardOptions, Result, cache::CacheConfig,
};

/// Greedily appends `steps >= 1` tokens to every row, keeping the cache external.
fn greedy<B: Backend, M: CausalLanguageModel<B>>(
    model: &M,
    prompt: CausalLmInput<B>,
    steps: usize,
) -> Result<Tensor<B, 2, Int>> {
    let device = prompt.token_ids.device();
    let [batch, prompt_tokens] = prompt.token_ids.dims();
    // The last sampled token is never fed back, so it needs no cache slot.
    let cache = CacheConfig::new(batch, prompt_tokens + steps.saturating_sub(1))?;
    let mut state = model.new_decode_state_for(cache, &device)?;
    // The defaults project only each row's last valid position: [batch, 1, vocabulary].
    let options = ForwardOptions::default();
    let mut logits = model.prefill(prompt, &mut state, &options)?.logits;
    let mut tokens: Vec<Tensor<B, 2, Int>> = Vec::with_capacity(steps);
    for _ in 0..steps {
        if let Some(previous) = tokens.last() {
            let input = DecodeInput { token_ids: previous.clone(), position_ids: None, token_mask: None };
            logits = model.decode(input, &mut state, &options)?.logits;
        }
        tokens.push(logits.clone().argmax(2).squeeze_dim::<2>(2)); // [batch, 1]
    }
    Ok(Tensor::cat(tokens, 1))
}
```

`ForwardOptions` selects logits with `LogitsSelection` (`Last` by default,
`All`, or explicit `Positions`) and makes hidden states, router logits, and
auxiliary losses opt-in. Its `attentions` flag materializes probabilities only
in stateless passes of DeepSeek-V2, and of the Llama, Gemma, and gpt-oss
families when every block uses `AttentionPrecisionMode::EagerFp32Softmax`;
other families and every cached path return an error. Attention-pattern hook
points are the general route
([EXPERIMENTS.md](EXPERIMENTS.md#attention-scores-patterns-and-sinks)).

`generate_causal` (and `generate_causal_streaming`) wrap this loop with
sampling, penalties, stop sequences, and per-token log probabilities;
`prefill_chunk_size: Some(256)` bounds long prompts while keeping each ragged
row's last valid logits, for every `CausalLanguageModel`, `ModularCausalLm`
included. For T5,
`generate_t5_beam_search` reads the decoder-start, EOS, and PAD IDs from the
configuration, encodes once, and decodes `source_batch * num_beams` rows; each
step reorders the decoder cache within each source's beams while the encoder
output and projected cross-attention keys and values stay in place.
`num_beams = 1` gives deterministic greedy decoding through the same path, and
`generation::generate_seq2seq_beam_search` accepts a longer decoder prompt and
an explicit `BeamSearchConfig`; over `AutoSeq2SeqLm` it also keeps the
cross-attention keys and values in place. Both stop with Transformers'
`EarlyStopping::Heuristic` by default. An `EncoderOutput` passed to T5 must have the
dtype of the model's parameters (states are never cast silently). The first
successful `prefill_decoder` binds it to the state; later calls must pass an
output of the same shape, device, dtype, and mask layout, and the state keeps
using its bound copy.

## Quantization

The native format is deterministic, symmetric, weight-only Q8 or Q4 with
contiguous groups of 32, 64, or 128 values (64 by default). Conversion follows
the validated architecture mapping and streams one tensor at a time into a
memory-mapped Burnpack sidecar. The sidecar is **not** a standalone checkpoint:
excluded parameters (embeddings, norms, biases, routers, recurrent scalars and
convolutions, MLA latent expansion, and first-generation MTP parameters) still
load from the original SafeTensors. Automatic packed loading is enabled for
`qwen3_5_text`, `qwen3_moe`, and `muse_glimmer_text`; the
[status page](ARCHITECTURE_COMPATIBILITY.md#quantized-checkpoints) records
their evidence. No quantized profile is advertised.

Every packed projection is installed as exactly the artifact's values. On
Metal it stays packed QFloat. On Burn Flex 0.21, whose quantized transpose
would re-block the weights, the packed Qwen3.5, Qwen3-MoE, and Muse
projections are held as their exact dequantized F32 values, about 3.6 times
the packed projection bytes. The load report's quantized records describe the
artifact's packed form, not the in-memory storage. No prefill speed figure is
claimed for this path until an idle-host benchmark exists.

The only certified packed-Metal build is an exact single-backend build:

```sh
cargo build --release --no-default-features --features std,metal,tokenizers
```

Do not include `rocm` or `attention-kernel-tracing` in that binary: Burn 0.21
feature-unifies CubeCL autotuning from either, which can select invalid packed
Q4/Q8 Metal kernels. The crate rejects packed Q4/Q8 Metal execution in builds
with either feature before dispatch; a downstream crate that enables CubeCL
autotuning directly is not certified.

gpt-oss's MXFP4 experts are decoded exactly to the load dtype; there is no
packed MXFP4 execution yet.

## Non-goals

- Vision or audio towers, even when their tensors are present in a checkpoint.
- GGUF, AWQ, GPTQ, external FP8 formats, or importing third-party quantized
  artifacts.
- A separate trainer, optimizer, or dataset framework. Experiments use native
  Burn training; packed quantized autodiff remains unsupported in Burn 0.21.
- A CLI, HTTP server, continuous batching, tensor/model parallelism, or
  distributed execution.
- Claims beyond 32K context on the current acceptance hardware.
- Silent approximation of an unavailable architecture feature. Unsupported
  combinations return contextual errors.

## Documentation map

| Read | For |
| --- | --- |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Capturing, editing, and patching activations; staged execution; attention policies; custom modules and specs. |
| [TRAINING.md](TRAINING.md) | Native Burn training, initialization policies, freezing, losses, and records. |
| [examples/README.md](examples/README.md) | Runnable examples and the paper reproductions with their measured results. |
| [ARCHITECTURE_COMPATIBILITY.md](ARCHITECTURE_COMPATIBILITY.md) | Model status: what is implemented, validated, and supported, with every evidence link. |
| [SUPPORT.md](SUPPORT.md) | What a support claim means and the advertised v0.1 profiles. |
| [TRANSFORMER_INNOVATIONS.md](TRANSFORMER_INNOVATIONS.md) | Every mechanism's mathematics, reference, and implementation, per family. |
| [ARCHITECTURES.md](ARCHITECTURES.md) | Pinned research targets, shared-feature ownership, and numerical policy. |
| [CHANGELOG.md](CHANGELOG.md), [MIGRATION.md](MIGRATION.md) | What changed, and how to upgrade from 0.1.0. |
| [BENCHMARKS.md](BENCHMARKS.md) | Performance protocol and measurements. |
| [RELEASE_GATES.md](RELEASE_GATES.md), [RELEASING.md](RELEASING.md) | Release checks, real-checkpoint gates, and their recorded history. |
| [verus/README.md](verus/README.md), [fuzz/README.md](fuzz/README.md) | Formal proofs and fuzzing. |
| [ROADMAP.md](ROADMAP.md), [V0_2_INTERFACE.md](V0_2_INTERFACE.md) | Plans and the experiment-interface contract. |
| [QWEN3_5_0_8B_RELEASE.md](QWEN3_5_0_8B_RELEASE.md), [QWEN3_5_CHECKPOINT_MAPPING.md](QWEN3_5_CHECKPOINT_MAPPING.md), [NEMOTRON_H_ACCEPTANCE.md](NEMOTRON_H_ACCEPTANCE.md) | Model-specific profiles, tensor mappings, and acceptance protocols. |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
