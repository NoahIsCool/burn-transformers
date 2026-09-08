# Transformer architecture compatibility

This document separates configuration compatibility, executable Burn runtimes,
strict checkpoint-loading support, and real-checkpoint acceptance. These are
different claims. A checked configuration means the crate can deserialize the
official text topology (including wrapper and legacy aliases) and reject
contradictory tensor shapes before model allocation. An executable runtime means
that stateless and cached inference paths run in synthetic/reference tests. A
strict loader means that the family owns an explicit tensor contract and can load
local SafeTensors without unexplained keys; it does not prove that the release
checkpoint has actually completed weight application or inference.

No checkpoint is currently advertised as fully supported. Every committed
acceptance record retains `advertised_supported: false`; some record only
artifact authentication and a strict load plan rather than weight application.

## Compatibility matrix

| Family | Reference checkpoint/config | Serialized identity | Wrapper and aliases | Topology represented and validated |
|---|---|---|---|---|
| Qwen3.5 / Qwen3.8 dense hybrid | [Qwen3.8-27B `config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json), [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B/blob/feea018b31f89dc0950e61da42577a7a4ab09169/config.json) | Inner text config remains `qwen3_5_text` for Qwen3.8 | Direct or nested `text_config`; `rope_scaling` alias; `Qwen38TextConfig` Rust alias | Explicit or interval-derived GDN versus full-attention schedule, gated-Q projection, GQA, linear head/state widths, convolution, multimodal RoPE sectioning, and MTP depth/embedding policy |
| Qwen3 MoE | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json) | `qwen3_moe` | Direct or any supported text wrapper; `num_local_experts` and `rope_scaling` aliases | GQA/head dimensions, dense-versus-MoE cadence, dense overrides, expert width/count/top-k, router normalization, RoPE, and optional sliding window |
| DeepSeek-V2 / V2-Lite | [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/config.json) | `deepseek_v2` | Direct or `text_config`, `language_config`, or `llm_config`; old/new RoPE key aliases; nullable Q LoRA rank | MLA Q/K content and rotary widths, independent V width, Q/KV latent ranks, dense prefix and MoE cadence, grouped top-k capacity, shared/routed experts, and YaRN constraints |
| Muse Glimmer text decoder | [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/config.json), [Transformers attention reference](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/muse_glimmer/modeling_muse_glimmer.py) | Inner `muse_glimmer_text` | Nested `text_config` or direct text config; `hidden_act` and `rope_scaling` aliases | GQA, explicit or backward-derived sliding/full schedule, per-layer RoPE/NoPE, sliding window, QK scaling, FP32 softmax for half inputs, post norms, and final-logit scaling/softcap |
| T5 / T5 v1.1 / FLAN-T5 | [FLAN-T5-XL](https://huggingface.co/google/flan-t5-xl/blob/8772db7a7a11f7b08e6be7d7088f7a7fd4813bc5/config.json), [pinned upstream T5 schema](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/t5/configuration_t5.py) | `t5` | Canonical names plus `hidden_size`, `head_dim`, `intermediate_size`, and standard head/depth aliases | Independent residual and attention-inner widths, asymmetric encoder/decoder depth, relative-position buckets/range, dense versus gated FFN layout, output-scaling flag, and decoder start/token IDs |
| UL2 | [Google UL2](https://huggingface.co/google/ul2/blob/71c34c5c6e3e54c5bf2cee360c3edfaa42acc1b3/config.json) | `t5` | `Ul2Config` Rust alias; accepts historic derived `dense_act_fn` and `is_gated_act` | T5 encoder-decoder topology plus gated-SiLU FFN consistency |
| Nemotron-H / Nemotron Nano | [Nemotron 3.5 Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/config.json), [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/config.json) | `nemotron_h` | Compact `hybrid_override_pattern` or modern `layers_block_type`/`layer_types`; historic `mamba_*` aliases; `attention_head_dim`; bare legacy `Infinity` | Exact heterogeneous Mamba/attention/MLP/MoE sequence, NoPE attention with GQA, Mamba state/head/group/convolution/time-step topology, expert routing, redundant norm consistency, and optional MTP block sequence |

## Runtime and acceptance status

The status below is evidence-based and intentionally narrower than the schema
matrix. “Strict loader implemented” means that the production local loader,
family tensor mapping, and synthetic failure tests exist. “Real weights applied”
means the named pinned artifact was actually streamed into a model and executed.

| Target | Executable transformer/runtime surface | Checkpoint evidence | Current acceptance state |
|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` | Qwen3.5/Qwen3.8 hybrid graph with DeltaNet and full-attention layers, recurrent/KV decode state, gated GQA, partial MRoPE, SwiGLU, and optional batch-one, one-layer MTP speculation | Strict local multimodal mapping and real weight application passed. The checked artifact has 488 physical tensors; the base graph consumes 320 tensors, produces one tied alias, and explicitly reports 153 visual plus 15 omitted MTP tensors. | [Real-weight record](acceptance/qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17.json): Metal F16 logit/reference and 32-token greedy parity, cached/stateless parity, and a Q4 packed smoke run passed. Q8 single-backend Metal, quantized quality, and 32K stress remain open; not advertised. |
| `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Same executable hybrid and MTP runtime as the bring-up model, with its exact 64-layer schedule | The pinned 18-shard artifact's exact configuration and all 1,199 physical headers match the committed inventory. Strict loading applied all 866 text-plus-MTP tensors and allowlisted exactly 333 visual tensors; the tokenizer and no-thinking chat template also passed. | [Real-weight record](acceptance/qwen3.8-27b-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.json): Metal F16 matched the official full-vocabulary prompt logits at `0.999999881` cosine, the exact top-1, all 32 cached greedy tokens, and stateless top-1 at every generated prefix. MTP and recurrent decode smokes plus strict mixed Q8/F16 and Q4/F16 smokes also passed. Quantized reference/quality, standardized throughput and output hashing, and 32K stress remain open; not advertised. |
| `Qwen/Qwen3-30B-A3B` at `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | Executable GQA decoder with Q/K normalization, top-8 routing, sparse selected-expert execution, and cached decode | The pinned 16-shard artifact's exact configuration and all 18,867 BF16 tensor headers match the committed inventory. Strict source-ordered loading applied every tensor as F16 with 18,480 explicit renames, opened each shard exactly once, and retained the tokenizer. Provenance-bound Q8 and Q4 conversion packed 18,625 matrices and retained 242 tensors as F16. | [Real-weight record](acceptance/qwen3-30b-a3b-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39.json): six-token F16, mixed Q8/F16, and mixed Q4/F16 Metal stateless, cached-prefill, 50-hidden-trace, and recurrent-decode smokes passed. Both packed decode audits launched all 1,345 expected matmuls without whole-weight dequantization. Official reference logits, 32-token generation, quantized quality, standardized performance, and 32K stress remain open; not advertised. |
| `deepseek-ai/DeepSeek-V2-Lite` at `604d5664dddd88a0433dbae533b7fe9472482de0` | Executable YaRN/MLA decoder with reference unabsorbed prefill, compressed latent-cache absorbed decode, routed top-k experts, and shared experts | The pinned four-shard artifact's exact configuration and all 5,291 BF16 tensor headers match the committed inventory. Strict source-ordered loading applied every tensor as F16 with 5,099 explicit renames, opened each shard exactly once, and retained the tokenizer. | [Real-weight record](acceptance/deepseek-v2-lite-604d5664dddd88a0433dbae533b7fe9472482de0.json): a six-token Metal F16 stateless, reference-unabsorbed cached-prefill, 29-hidden-trace, compressed-cache, and absorbed-decode smoke passed. Cached/stateless cosine was `1.0`; absorbed decode matched the internal stateless path at `0.999998964`. No official-reference numerical comparison has run. Reference logits, 32-token generation, quantization, and 32K stress remain open; not advertised. |
| `meta-models/Muse-Glimmer-30B` at `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` | Executable text decoder with sliding/full attention, RoPE/NoPE scheduling, sandwich norms, attention output gates, final-logit softcap, and score-bounded query chunking for exact eager-F16 inference | The exact two-shard artifact's control files, payload hashes, 1,436 BF16 headers, 627-tensor text mapping, 809-key visual allowlist, F16 load plan, and tokenizer round trip passed. Strict Metal loading applied all 627 language tensors as F16, including 626 explicit renames and BF16-to-F16 conversions, with zero missing, unexpected, mismatched, or error diagnostics. Provenance-bound Q8/G64 and Q4/G64 conversions each packed 417 matrices, retained 210 tensors as F16, and fully reopened their exact v2 sidecars. | [Real-weight record](acceptance/muse-glimmer-30b-a4e59da52a7bc87ae7251dd5545c0dd437c44b68.json): the authenticated seven-file schema-3 oracle covers full-vocabulary prompt logits, every cached and teacher-forced decision row, and stateless, cached-prefill, and first recurrent-decode hidden traces. Burn's minimum official-reference cosine was `0.999943377`, and all 32 official greedy tokens matched. The real Q8 sidecar passed stateless, cached-prefill, and recurrent Metal execution; its trace observed all 417 packed decode matmuls and no whole-weight dequantization. A synthetic Metal gate exercised the 32K attention/cache/position envelope with official head counts but reduced head width and no dense blocks; full-model 32K acceptance remains open. Q4 Metal, quantized reference/quality, real-checkpoint padded/context-boundary coverage, and standardized performance also remain open; not advertised. |
| `google/flan-t5-small` at `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` | Executable T5 encoder-decoder with bidirectional encoding, causal decoding, cross-attention caches, relative-position bias, shared encoder/decoder embeddings, an independent LM head, and cached beam search | The official one-file SafeTensors artifact was strictly loaded and all 190 F32 tensors were applied through the public automatic loader. | [Real-weight record](acceptance/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.json): pinned top-10 and sampled-logit probes, cached prefill/decode parity, and the exact four-beam translation fixture passed on Flex. Full-vector reference-logit cosine, longer generation, accelerator inference, and context-boundary stress remain open; not advertised. |
| `google/flan-t5-xxl` at `ae7c9136adc7555eeccc78cdd960dfd60fb346ce` | Executable 24-layer encoder/24-layer decoder T5 graph with gated GELU, relative-position bias, cross-attention caches, greedy decode, and beam search | All five official F32 SafeTensors shards were content-verified. Strict loading applied all 558 runtime parameters, verified the physical embedding equivalence, performed 433 checkpoint-layout transposes, and retained the tokenizer. | [Real-weight record](acceptance/flan-t5-xxl-ae7c9136adc7555eeccc78cdd960dfd60fb346ce.json): Metal matched five official Transformers/MPS F32 tensor fixtures at minimum cosine `0.999999999998`, all 32 cached greedy tokens, all teacher-forced top-1 rows, and the exact four-beam continuation. Padded/context-boundary, quantized quality/32K, and standardized performance gates remain open; not advertised. |
| `google/flan-ul2` at `452d74ce28ac4a7f211d6ba3ef0717027f7a8074` | The executable T5 runtime represents UL2's gated-SiLU topology | Strict T5-family mapping exists. The pinned target still requires the planned verified offline PyTorch-to-SafeTensors conversion at this revision. | No real-weight acceptance record; not advertised. |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` at `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` | Executable heterogeneous Mamba2/GQA/dense-MLP/MoE graph with ReLU-squared experts, shared expert, recurrent/KV state, and a one-stage next-token head with transactional batch-one speculation | The exact fourteen-shard artifact was authenticated and all 6,513 tensors were strictly streamed into Metal F16, including 270 MTP tensors. The exact MTP body, independent KV bank, partial target/MTP commits, rollback, greedy verification, and lossless stochastic rejection pass tiny Flex tests. The Nano revision `bf77c3174f68ad409e1c2aa60daeb46e32d1c606` is configuration-compatible only. | [Real-weight record](acceptance/nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json): stateless, cached-prefill, and recurrent base-model smokes passed; recurrent versus teacher-forced logits had cosine `0.999376514` and identical top-1. This is internal consistency, not an official oracle comparison. Official base parity, real-weight MTP parity, 32-token generation, quantization, and 32K remain open; not advertised. |

## Loader behavior

`ModelConfig::from_json_str` first parses strict JSON after one deliberately narrow
compatibility repair: an unquoted positive `Infinity` token is converted to `null`.
NVIDIA's original Nemotron-H configs used that Python JSON extension for an
unbounded time-step maximum, while newer artifacts use `null`. Strings containing
the word `Infinity` are unchanged.

For a multimodal repository, only `text_config` is consumed. The compatibility
aliases `language_config` and `llm_config` support older/custom wrappers. Vision
metadata is ignored because this crate milestone covers transformer text backbones
only. Unknown non-topology metadata is also ignored so harmless Hub metadata does
not break loading; unknown enum values and contradictory shape fields fail closed.

`AutoCausalLm::load_local` now dispatches executable `deepseek_v2`,
`muse_glimmer_text`, `nemotron_h`, `qwen3_5_text`, and `qwen3_moe` graphs.
`AutoSeq2SeqLm::load_local` dispatches `t5`. Both APIs are local-only: they
discover SafeTensors, validate the architecture-owned mapping before applying
weights, attach an optional local tokenizer, and return the final strict
`LoadReport`. Unknown checkpoint tensors fail closed unless an architecture
explicitly allowlists them, as Qwen and Muse do for their visual towers.

## Known gaps and non-claims

- A schema match alone still does not prove checkpoint compatibility. The strict
  family contract compares every observed name, shape, and dtype before weight
  application, and only a real-artifact run establishes evidence for one pinned
  revision.
- Qwen3.8 uses the Qwen3.5 architecture discriminator in the official artifact.
  No invented `qwen3_8_text` serialized value is accepted.
- Qwen3.8's complete text-plus-MTP weights now load and execute on Metal F16,
  while its mixed Q8/F16 and Q4/F16 graphs have each passed a strict smoke run.
  The F16 graph also matches the pinned official full-vocabulary logits and all
  32 stable-margin greedy tokens. Quantized reference/quality,
  standardized-throughput and output-hash, and 32K gates are still pending.
- Muse Glimmer vision topology is intentionally not represented. Only its nested
  text decoder is extracted.
- Nemotron-H's historical configs are not RFC-compliant JSON because they contain
  `Infinity`; only the published positive-infinity case is repaired. `NaN` and
  negative infinity remain errors.
- Qwen3 MoE's pinned 30B-A3B artifact has passed complete F16 weight application,
  provenance-bound Q8/Q4 conversion, and short Metal stateless/cached/decode
  smokes in both packed formats. Q4 and Q8 both passed their 1,345-launch
  packed-kernel audits without whole-weight dequantization. Official reference,
  generation, quantized-quality, standardized-performance, and 32K gates remain
  open.
- DeepSeek V2 Lite has passed complete F16 weight application and a short Metal
  stateless/reference-prefill/compressed-cache/absorbed-decode smoke. Its
  cached and decode comparisons are internal Burn-path parity, not official
  numerical-reference acceptance; generation, quantization, and 32K gates also
  remain open.
- Muse Glimmer has authenticated its exact real artifact and seven-file official
  oracle, strictly applied all 627 text tensors as Metal F16, and passed its
  short-context numerical and 32-token parity gate. Its real Q8 conversion,
  full sidecar reopen, mixed packed-Metal execution, and 417-launch decode audit
  passed. Its Q4 sidecar conversion and full reopen also passed, while Q4
  packed-Metal execution, quantized reference/quality, real-32K,
  real-checkpoint padded/context-boundary, and standardized-performance gates
  remain open. FLAN-UL2 has an executable
  synthetic runtime and strict loader but no real-artifact inference acceptance
  record. Nemotron-H 3.5 has passed strict real-weight loading and a short
  Burn-internal Metal base-model smoke, and now has a separately authenticated
  official base oracle; Burn-versus-official and real-weight MTP parity remain
  open.
- MTP execution, tokenization, cached generation, T5 beam search, and native Q4/Q8
  are runtime facilities rather than configuration claims. Automatic packed
  checkpoint loading is limited to `muse_glimmer_text`, `qwen3_5_text`, and
  `qwen3_moe`; v2 sidecars bind exact checkpoint, load-plan, and payload hashes,
  while legacy v1 artifacts retain full source-payload validation. Qwen3.8 and
  Qwen3-MoE Q8/Q4 Metal smokes passed, while Muse has traced Q8 Metal evidence;
  their quantized-reference and quality gates remain open.
- Qwen MTP speculative generation currently accepts batch size one only and
  defaults to two draft tokens per cycle. A different positive draft count is
  configurable; ordinary Qwen inference supports batches.
- Multimodal execution, tensor/expert parallelism, external GGUF/AWQ/GPTQ/FP8
  formats, server/distributed execution, and training loops remain out of scope.
