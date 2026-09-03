# Transformer architecture compatibility matrix

This document is the source-of-truth research record for the first supported
release of the `transformers` crate. A model is not considered supported merely
because its configuration parses: support requires strict weight loading,
reference-logit parity, cache parity, generation parity, and a recorded hardware
acceptance run.

The checkpoint revisions below are immutable. Reference fixtures are generated
from Hugging Face Transformers revision
`cfa309cfa34b5497a32fd50af49aa97aac94c054`; Qwen multi-token-prediction
semantics are cross-checked against vLLM revision
`bb363db9a5ec2edc7b39e99b00af363a89d1fb81`. Burn is pinned to the annotated
`v0.21.0` tag at `546cacb55fe00168854d19bdf0a5d79bd8060e03`.

## Compatibility targets

| Family | Acceptance checkpoint | Immutable revision | Transformer features | Checkpoint handling |
| --- | --- | --- | --- | --- |
| Qwen3.5 text / Qwen3.8 | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 64-layer 3:1 DeltaNet/full-attention schedule; zero-centered RMSNorm; partial MRoPE; gated GQA; SwiGLU; one MTP layer | Load `model.language_model.*`, `lm_head.*`, and eventually `mtp.*`; report `model.visual.*` as an explicitly ignored modality |
| Qwen3.5 bring-up | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | Same configuration family used to validate parsing, tokenizer integration, and checkpoint mapping before the 27B run | Strict sharded SafeTensors |
| Qwen3 MoE | `Qwen/Qwen3-30B-A3B` | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | GQA and Q/K normalization; 128 routed experts; top-8 routing; no shared expert | Strict sharded SafeTensors; selected-expert dispatch must remain sparse |
| DeepSeek V2 | `deepseek-ai/DeepSeek-V2-Lite` | `604d5664dddd88a0433dbae533b7fe9472482de0` | Multi-head latent attention; compressed latent decode cache; YaRN; top-6 routed and two shared experts | Strict sharded SafeTensors; remote Python is never executed |
| Muse Glimmer | `meta-models/Muse-Glimmer-30B` | `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` | Three sliding-window RoPE blocks per full NoPE block; sandwich norms; query scaling; attention gates; final logit softcap | Load the text tower only and report visual tensors explicitly |
| T5 / FLAN bring-up | `google/flan-t5-small` | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` | Bidirectional encoder; causal decoder; cross attention; bucketed relative bias; tied embeddings; gated feed-forward | Single-file official SafeTensors |
| T5 / FLAN full | `google/flan-t5-xxl` | `ae7c9136adc7555eeccc78cdd960dfd60fb346ce` | 24 encoder and 24 decoder layers; gated GELU | Official five-shard SafeTensors only |
| T5 / UL2 | `google/flan-ul2` | `452d74ce28ac4a7f211d6ba3ef0717027f7a8074` | T5 encoder-decoder inference graph with gated SiLU | The pinned repository has PyTorch shards only; convert offline one shard at a time with `weights_only=True`, then validate every tensor against the source index |
| Nemotron-H 3.5 | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` | Interleaved Mamba2, GQA, and MoE; 128 routed experts; top-6 routing plus a shared expert; ReLU-squared experts; next-token head | Strict sharded SafeTensors and BF16-to-F16 streaming conversion on Metal |
| Nemotron-H compatibility | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `bf77c3174f68ad409e1c2aa60daeb46e32d1c606` | Earlier Nemotron-H 3 configuration in the same family | Configuration and synthetic compatibility tests; 3.5 Lightning remains the real-weight target |

## Shared feature ownership

Architecture modules translate checkpoint configuration, wire exact layer
schedules, and declare explicit checkpoint mappings. They do not own reusable
mathematics. Reusable implementations are separated into these domains:

- **Attention:** causal full attention, GQA/MQA, bounded sliding attention,
  output gating, and MLA prefill/decode paths.
- **Mixers:** DeltaNet and Mamba2 recurrent decode plus chunked prefill.
- **Feed-forward and MoE:** dense gated projections, activation variants,
  stable router softmax, deterministic top-k selection, sparse dispatch, and
  optional shared experts.
- **Position:** ordinary and partial RoPE, MRoPE-compatible text positions,
  YaRN scaling, NoPE layers, and T5 bucketed relative bias.
- **State:** native-KV-head caches, sliding rings, MLA latent state,
  convolution/recurrent state, cross-attention projections, beam reordering,
  and transactional speculative updates.
- **Artifacts:** local-only configuration/tokenizer discovery, strict sharded
  SafeTensors loading, streaming dtype conversion, Burn-native quantization,
  and provenance-bearing Burnpack persistence.

## Qwen3.8 tensor contract

The top-level Qwen3.8 repository is multimodal, while its language configuration
uses model type `qwen3_5_text`. Text inference therefore selects the language
tower rather than constructing a fake vision module. Base inference consumes
language and LM-head tensors. The MTP milestone additionally consumes exactly
the 15 `mtp.*` tensors in the checkpoint index; it reuses the base token
embedding and untied LM head instead of defining dedicated copies.

The repeating language schedule contains three gated DeltaNet blocks followed
by one gated full-attention block. DeltaNet cache state consists of causal
convolution history and a recurrent key/value matrix. Full attention stores only
native KV heads. MTP owns an independent one-layer KV cache and uses
transactional target state so a rejected speculative suffix can be discarded
without corrupting either form of state.

## Numerical and acceptance policy

- Full-precision stateless forwards remain differentiable and keep caches
  outside serialized model parameters.
- Metal loads BF16 source weights as F16; FP32 is retained for sensitive router,
  normalization, and recurrent-state calculations where required by the
  reference implementation.
- Q8 and Q4 are weight-only, blockwise Burn formats. Embeddings, norms, biases,
  recurrent scalars/state, and first-generation MTP weights remain F16.
- Configuration parsing or successful tensor allocation alone never grants
  support status. The acceptance record must contain zero unexplained key
  mismatches, reference-logit metrics, cached/uncached parity, generated token
  IDs, peak memory, and throughput.
- Multimodal towers, external quantized formats, distributed execution, server
  APIs, and training loops are deliberately outside this release.

## Primary references

- Qwen3.8 configuration and weight index:
  <https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0>
- Qwen3 MoE: <https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39>
- DeepSeek V2 Lite: <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/604d5664dddd88a0433dbae533b7fe9472482de0>
- Muse Glimmer: <https://huggingface.co/meta-models/Muse-Glimmer-30B/tree/a4e59da52a7bc87ae7251dd5545c0dd437c44b68>
- FLAN-T5 XXL: <https://huggingface.co/google/flan-t5-xxl/tree/ae7c9136adc7555eeccc78cdd960dfd60fb346ce>
- FLAN-UL2: <https://huggingface.co/google/flan-ul2/tree/452d74ce28ac4a7f211d6ba3ef0717027f7a8074>
- Nemotron-H 3.5: <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/tree/a9904d24bcc1d289a1950fa9d2b978c47cf903b9>

