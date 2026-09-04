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

No checkpoint is currently advertised as fully supported. The three real-weight
acceptance records all retain `advertised_supported: false`, and the remaining
families do not yet have real-weight acceptance records.

## Compatibility matrix

| Family | Reference checkpoint/config | Serialized identity | Wrapper and aliases | Topology represented and validated |
|---|---|---|---|---|
| Qwen3.5 / Qwen3.8 dense hybrid | [Qwen3.8-27B `config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json), [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B/blob/feea018b31f89dc0950e61da42577a7a4ab09169/config.json) | Inner text config remains `qwen3_5_text` for Qwen3.8 | Direct or nested `text_config`; `rope_scaling` alias; `Qwen38TextConfig` Rust alias | Explicit or interval-derived GDN versus full-attention schedule, gated-Q projection, GQA, linear head/state widths, convolution, multimodal RoPE sectioning, and MTP depth/embedding policy |
| Qwen3 MoE | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json) | `qwen3_moe` | Direct or any supported text wrapper; `num_local_experts` and `rope_scaling` aliases | GQA/head dimensions, dense-versus-MoE cadence, dense overrides, expert width/count/top-k, router normalization, RoPE, and optional sliding window |
| DeepSeek-V2 / V2-Lite | [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/config.json) | `deepseek_v2` | Direct or `text_config`, `language_config`, or `llm_config`; old/new RoPE key aliases; nullable Q LoRA rank | MLA Q/K content and rotary widths, independent V width, Q/KV latent ranks, dense prefix and MoE cadence, grouped top-k capacity, shared/routed experts, and YaRN constraints |
| Muse Glimmer text decoder | [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/config.json) | Inner `muse_glimmer_text` | Nested `text_config` or direct text config; `hidden_act` and `rope_scaling` aliases | GQA, explicit or backward-derived sliding/full schedule, per-layer RoPE/NoPE, sliding window, QK scaling, post norms, and final-logit scaling/softcap |
| T5 / T5 v1.1 / FLAN-T5 | [FLAN-T5-XL](https://huggingface.co/google/flan-t5-xl/blob/8772db7a7a11f7b08e6be7d7088f7a7fd4813bc5/config.json), [pinned upstream T5 schema](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/t5/configuration_t5.py) | `t5` | Canonical names plus `hidden_size`, `head_dim`, `intermediate_size`, and standard head/depth aliases | Independent residual and attention-inner widths, asymmetric encoder/decoder depth, relative-position buckets/range, dense versus gated FFN layout, output-scaling flag, and decoder start/token IDs |
| UL2 | [Google UL2](https://huggingface.co/google/ul2/blob/71c34c5c6e3e54c5bf2cee360c3edfaa42acc1b3/config.json) | `t5` | `Ul2Config` Rust alias; accepts historic derived `dense_act_fn` and `is_gated_act` | T5 encoder-decoder topology plus gated-SiLU FFN consistency |
| Nemotron-H / Nemotron Nano | [Nemotron 3.5 Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/config.json), [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/config.json) | `nemotron_h` | Compact `hybrid_override_pattern` or modern `layers_block_type`/`layer_types`; historic `mamba_*` aliases; `attention_head_dim`; bare legacy `Infinity` | Exact heterogeneous Mamba/attention/MLP/MoE sequence, attention GQA/RoPE, Mamba state/head/group/convolution/time-step topology, expert routing, redundant norm consistency, and optional MTP block sequence |

## Runtime and acceptance status

The status below is evidence-based and intentionally narrower than the schema
matrix. “Strict loader implemented” means that the production local loader,
family tensor mapping, and synthetic failure tests exist. “Real weights applied”
means the named pinned artifact was actually streamed into a model and executed.

| Target | Executable transformer/runtime surface | Checkpoint evidence | Current acceptance state |
|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` | Qwen3.5/Qwen3.8 hybrid graph with DeltaNet and full-attention layers, recurrent/KV decode state, gated GQA, partial MRoPE, SwiGLU, and optional batch-one, one-layer MTP speculation | Strict local multimodal mapping and real weight application passed. The checked artifact has 488 physical tensors; the base graph consumes 320 tensors, produces one tied alias, and explicitly reports 153 visual plus 15 omitted MTP tensors. | [Real-weight record](acceptance/qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17.json): Metal F16 logit/reference and 32-token greedy parity, cached/stateless parity, and a Q4 packed smoke run passed. Q8 single-backend Metal, quantized quality, and 32K stress remain open; not advertised. |
| `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Same executable hybrid and MTP runtime as the bring-up model, with its exact 64-layer schedule | The pinned 18-shard artifact's exact configuration and all 1,199 physical headers match the committed inventory. Strict loading applied all 866 text-plus-MTP tensors and allowlisted exactly 333 visual tensors; the tokenizer and no-thinking chat template also passed. | [Real-weight record](acceptance/qwen3.8-27b-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.json): Metal F16 stateless, cached prefill, MTP, and recurrent decode passed with `1.0` cosine at all 66 compared hidden traces. Reference logits, 32-token generation, Q8/Q4 quality, and 32K stress remain open; not advertised. |
| `Qwen/Qwen3-30B-A3B` at `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | Executable GQA decoder with Q/K normalization, top-8 routing, sparse selected-expert execution, and cached decode | Strict family mapping and automatic local loader are implemented and exercised with synthetic fixtures. | No pinned real-weight load or inference acceptance has been recorded; not advertised. |
| `deepseek-ai/DeepSeek-V2-Lite` at `604d5664dddd88a0433dbae533b7fe9472482de0` | Executable YaRN/MLA decoder with reference unabsorbed prefill, compressed latent-cache absorbed decode, routed top-k experts, and shared experts | Strict family mapping and automatic local loader are implemented and exercised with synthetic golden, cache, and checkpoint-contract tests. | No pinned real-weight load or inference acceptance has been recorded; not advertised. |
| `meta-models/Muse-Glimmer-30B` at `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` | Executable text decoder with sliding/full attention, RoPE/NoPE scheduling, sandwich norms, attention output gates, and final-logit softcap | Strict text-only/multimodal mapping and automatic local loader are implemented and exercised with synthetic fixtures. | No pinned real-weight load or inference acceptance has been recorded; not advertised. |
| `google/flan-t5-small` at `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` | Executable T5 encoder-decoder with bidirectional encoding, causal decoding, cross-attention caches, relative-position bias, shared encoder/decoder embeddings, an independent LM head, and cached beam search | The official one-file SafeTensors artifact was strictly loaded and all 190 F32 tensors were applied through the public automatic loader. | [Real-weight record](acceptance/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.json): pinned top-10 and sampled-logit probes, cached prefill/decode parity, and the exact four-beam translation fixture passed on Flex. Full-vector reference-logit cosine, longer generation, accelerator inference, and context-boundary stress remain open; not advertised. |
| `google/flan-t5-xxl` at `ae7c9136adc7555eeccc78cdd960dfd60fb346ce` and `google/flan-ul2` at `452d74ce28ac4a7f211d6ba3ef0717027f7a8074` | The executable T5 runtime represents XXL's gated-GELU and UL2's gated-SiLU topology | Strict T5-family mapping and local loading exist, but neither pinned target has completed a real artifact run. UL2 additionally requires the planned verified offline PyTorch-to-SafeTensors conversion at this revision. | No real-weight acceptance record; not advertised. |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` at `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` | Executable heterogeneous Mamba2/GQA/dense-MLP/MoE graph with ReLU-squared experts, shared expert, recurrent/KV state, and next-token-head mapping | Strict family mapping, automatic local loader, BF16 conversion policy, and synthetic configuration/math/cache/checkpoint tests are implemented. The Nano revision `bf77c3174f68ad409e1c2aa60daeb46e32d1c606` is configuration-compatible only. | Neither the 3.5 Lightning target nor Nano has a real-weight inference acceptance record; not advertised. |

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
- Qwen3.8's complete text-plus-MTP weights now load and execute on Metal F16, but
  its reference-logit, 32-token generation, quantized-quality, and 32K gates are
  still pending.
- Muse Glimmer vision topology is intentionally not represented. Only its nested
  text decoder is extracted.
- Nemotron-H's historical configs are not RFC-compliant JSON because they contain
  `Infinity`; only the published positive-infinity case is repaired. `NaN` and
  negative infinity remain errors.
- Qwen3 MoE, DeepSeek V2 Lite, Muse Glimmer, FLAN-T5 XXL/UL2, and Nemotron-H 3.5
  have executable synthetic runtimes and strict loaders but no real-weight
  acceptance record.
- MTP execution, tokenization, cached generation, T5 beam search, and native Q4/Q8
  are runtime facilities rather than configuration claims. Automatic packed
  checkpoint loading is currently limited to `qwen3_5_text`; Q8 Metal and quality
  gates remain open.
- Qwen MTP speculative generation currently accepts batch size one only and
  defaults to two draft tokens per cycle. A different positive draft count is
  configurable; ordinary Qwen inference supports batches.
- Multimodal execution, tensor/expert parallelism, external GGUF/AWQ/GPTQ/FP8
  formats, server/distributed execution, and training loops remain out of scope.
