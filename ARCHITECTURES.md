# Architecture research targets and policy

This document is the source-of-truth research record for the architectures the
`transformers` crate implements: the pinned checkpoints and references each
family was built against, which shared component owns each mechanism, and the
numerical policy for accepting a checkpoint. It started as the plan for the
first supported release. A model is not considered supported merely
because its configuration parses: support requires strict weight loading,
reference-logit parity, cache parity, generation parity, and a recorded hardware
acceptance run.

The research targets below are broader than the completed v0.1 release scope.
Only the exact Qwen3.5-0.8B Metal/F16 base-text and FLAN-T5-small Flex/F32 /
Metal/F32 profiles are promoted; see [the release record](acceptance/measurements/v0_1-release-20260910.json).
All other families and quantized-quality or real-weight MTP claims remain
experimental. Historical checkpoint bring-up descriptions below are not current
release-status declarations.

The checkpoint revisions below are immutable. Muse and the current shared
architecture fixtures pin Hugging Face Transformers revision
`cfa309cfa34b5497a32fd50af49aa97aac94c054`; Qwen3-MoE's executable oracle
instead pins the released Transformers 5.2 commit
`7d9754a05193eb79b1d86aa744b622b8068008cd`. Qwen multi-token-prediction
semantics are cross-checked against vLLM revision
`bb363db9a5ec2edc7b39e99b00af363a89d1fb81`. The Llama, Gemma, and gpt-oss
families added after v0.1 follow the released Transformers **5.2.0** (PyPI, with
PyTorch 2.10.0), the same release as commit `7d9754a…`; their tiny goldens and
real-checkpoint references were generated with that pair. Burn is pinned to the
annotated `v0.21.0` tag at `546cacb55fe00168854d19bdf0a5d79bd8060e03`.

## Compatibility targets

| Family | Acceptance checkpoint | Immutable revision | Transformer features | Checkpoint handling |
| --- | --- | --- | --- | --- |
| Qwen3.5 text / Qwen3.8 | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 64-layer 3:1 DeltaNet/full-attention schedule; zero-centered RMSNorm; partial MRoPE; gated GQA; SwiGLU; one MTP layer | Load `model.language_model.*`, `lm_head.*`, and optionally all `mtp.*`; report `model.visual.*` as an explicitly ignored modality |
| Qwen3.5 bring-up | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | Same configuration family used to validate parsing, tokenizer integration, and checkpoint mapping before the 27B run | Strict sharded SafeTensors |
| Qwen3 MoE | `Qwen/Qwen3-30B-A3B` | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | GQA and Q/K normalization; 128 routed experts; top-8 routing; no shared expert | Strict sharded SafeTensors; selected-expert dispatch must remain sparse |
| DeepSeek V2 | `deepseek-ai/DeepSeek-V2-Lite` | `604d5664dddd88a0433dbae533b7fe9472482de0` | Multi-head latent attention; compressed latent decode cache; YaRN; top-6 routed and two shared experts | Strict sharded SafeTensors; remote Python is never executed |
| Muse Glimmer | `meta-models/Muse-Glimmer-30B` | `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` | Three sliding-window RoPE blocks per full NoPE block; sandwich norms; query scaling; attention gates; final logit softcap | Load the text tower only and report visual tensors explicitly |
| T5 / FLAN bring-up | `google/flan-t5-small` | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` | Bidirectional encoder; causal decoder; cross attention; bucketed relative bias; shared encoder/decoder embeddings; untied LM head; gated feed-forward | Single-file official SafeTensors |
| T5 / FLAN full | `google/flan-t5-xxl` | `ae7c9136adc7555eeccc78cdd960dfd60fb346ce` | 24 encoder and 24 decoder layers; gated GELU | Official five-shard SafeTensors only |
| T5 / UL2 | `google/flan-ul2` | `452d74ce28ac4a7f211d6ba3ef0717027f7a8074` | T5 encoder-decoder inference graph with gated SiLU | The pinned repository has PyTorch shards only; convert offline one shard at a time with `weights_only=True`, then validate every tensor against the source index |
| Nemotron-H 3.5 | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` | Interleaved Mamba2, GQA, and MoE; 128 routed experts; top-6 routing plus a shared expert; ReLU-squared experts; next-token head | Strict sharded SafeTensors and BF16-to-F16 streaming conversion on Metal |
| Nemotron-H compatibility | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `bf77c3174f68ad409e1c2aa60daeb46e32d1c606` | Earlier Nemotron-H 3 configuration in the same family | Configuration and synthetic compatibility tests; 3.5 Lightning remains the real-weight target |
| Llama family (`llama`) | `HuggingFaceTB/SmolLM-135M` | `1d461723eec654e65efdc40cf49301c89c0c92f4` | 30-layer GQA decoder (9 query / 3 KV heads), RoPE, SwiGLU, tied head | Strict SafeTensors; BF16 cast exactly to F32 |
| Llama family (`qwen3`) | `Qwen/Qwen3-4B` | `531c80e289d6cff3a7cd8c0db8110231d23a6f7a` | Per-head Q/K RMSNorm, explicit `head_dim` 128, RoPE base 1e6 | Strict SafeTensors |
| Llama family (`smollm3`) | `HuggingFaceTB/SmolLM3-3B` | `1fa6fa281b5c635c89171a2b53769609cc1199c6` | NoPE in every fourth layer (9 of 36) | Strict SafeTensors |
| Llama family (`qwen2`) | `Qwen/Qwen2-7B` | `453ed1575b739b5b03ce3758b23befdb0967f40e` | Bias on Q/K/V only, untied head | Strict SafeTensors |
| Gemma 3 text | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` | Sandwich `(1 + w)` RMSNorm, `sqrt(d)` embedding, 5:1 local (window 1024) / global attention, per-layer-type RoPE, Q/K norm, GeGLU | Text tower through `text_config`; the vision tower is reported as intentionally ignored |
| gpt-oss | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | 24 layers alternating 128-token sliding and full attention with learned sinks, YaRN, 32 experts with top-4 biased routing, clamped SwiGLU | Strict SafeTensors; MXFP4 experts decoded exactly at load |

## Shared feature ownership

Architecture modules translate checkpoint configuration, wire exact layer
schedules, and declare explicit checkpoint mappings. They do not own reusable
mathematics. Reusable implementations are separated into these domains:

- **Attention:** MHA/GQA/MQA over native KV heads (cached decoding and
  suffixes fold query groups instead of repeating K/V; square causal prefill on
  the backend-native path keeps the fused `is_causal` kernel and repeats K/V
  there), bottom-right causal alignment on every backend,
  global/sliding/chunked-local connectivity, learned sinks, score soft caps,
  projection biases, per-head or full-width Q/K norms, output gating,
  differential attention, score-bounded chunked execution, and MLA
  prefill/decode paths.
- **Mixers:** Gated DeltaNet and the model-independent Mamba-2 mixer
  (`src/mamba2.rs`), with recurrent decode, chunked prefill, batched ragged
  rows, and per-token snapshots for speculation.
- **Feed-forward and MoE:** dense gated, ungated, ReLU², and clamped SwiGLU
  projections; one router covering softmax or sigmoid scoring, top-k before or
  after scoring, group-limited selection, correction bias, renormalization,
  routed scale, and shared-expert gates; memory-bounded sparse dispatch.
- **Position:** ordinary and partial RoPE, MRoPE-compatible text positions,
  linear, dynamic-NTK, YaRN, Llama 3, LongRoPE, and proportional scaling behind
  one Hugging Face configuration translator, NoPE layers, and T5 bucketed
  relative bias.
- **State:** native-KV-head caches, per-group sliding rings, single-plane MLA
  latent groups, convolution/recurrent state, cross-attention projections,
  group-preserving beam reordering, batched multi-row I/O, atomic step write
  sets, and transactional speculative updates.
- **Family glue:** `models::shared` validates each pass's token, mask, and
  position inputs with one fused host readback, resolves positions, and plans
  the compaction and scatter of ragged rows (Verus-verified), so families do
  not reimplement input checking or batching.
- **Artifacts:** local-only configuration/tokenizer discovery, strict sharded
  SafeTensors loading, exact streaming dtype conversion, exact MXFP4 decoding,
  Burn-native quantization, and provenance-bearing Burnpack persistence.

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

The first runtime slice implements batch-one greedy verification and stochastic
rejection sampling. `K` defaults to two. Prompt setup shifts `(x_(i+1), h_i)`
into the MTP cache, accepted target and MTP prefixes commit atomically, and an
all-accepted cycle performs one auxiliary catch-up step to preserve the invariant
`target_length = mtp_length + 1`. The stochastic correction distribution is
`normalize((p - q)_+)`, which preserves the target marginal. The pinned 27B
checkpoint now passes strict text-plus-MTP loading and a Metal F16 smoke covering
stateless, cached-prefill, MTP, and recurrent-decode execution. Support remains
unadvertised until reference logits, longer token parity, quantized quality,
long-context memory, and standardized throughput gates pass.

The exact namespace rewrites, tensor-shape equations, interleaved gated-query
layout contract, MTP inventory, and metadata-only golden acceptance counts are
recorded in [QWEN3_5_CHECKPOINT_MAPPING.md](QWEN3_5_CHECKPOINT_MAPPING.md).

## Numerical and acceptance policy

- Full-precision stateless forwards remain differentiable and keep caches
  outside serialized model parameters.
- Metal loads BF16 source weights as F16; FP32 is retained for sensitive router,
  normalization, and recurrent-state calculations where required by the
  reference implementation.
- Q8 and Q4 are weight-only, blockwise Burn formats. Embeddings, norms, biases,
  recurrent scalars/state, and first-generation MTP weights remain F16.
- Burnpack matrices execute through Burn's native `QFloat` matmul without
  whole-matrix dequantization. The certified Metal configuration is an exact
  single-backend build: `--no-default-features --features std,metal`, plus
  optional tokenizer features. Burn 0.21's ROCm dependency enables CubeCL
  autotuning globally; when it is feature-unified into a Metal binary, invalid
  packed Q4/Q8 GEMV candidates can be selected. The library rejects that known
  combination before dispatch, and all-feature builds remain compile/test gates
  rather than packed-Metal acceptance builds. A downstream crate can enable the
  CubeCL autotune feature independently, so such builds are not certified until
  Burn exposes a strategy selector or fixes the quantized tuner candidates.
- Configuration parsing or successful tensor allocation alone never grants
  support status. The acceptance record must contain zero unexplained key
  mismatches, reference-logit metrics, cached/uncached parity, generated token
  IDs, peak memory, and throughput.
- T5 in F16 implements only the reference's residual clamp guard; run T5 in
  BF16 or F32.
- Multimodal towers, external quantized formats, distributed execution, server
  APIs, and a separate training framework are deliberately outside this
  release; training uses native Burn.

## Experimental interpretability families

GPT-2 and GPT-NeoX/Pythia are stateless modules for the published-weight
interpretability studies, with strict local imports, hooks, and staged
execution; they are not registered in `AutoCausalLm` because they have no
external decode state. GPT-NeoX's model type is `GptNeoXForCausalLm`
(`GptNeoXModel` remains an alias). Both configurations provide `init_policy()`
reproducing Hugging Face random initialization (GPT-2 with its depth-scaled
residual projections); the policies keep LayerNorm parameters as constructed.

## Primary references

- Qwen3.8 configuration and weight index:
  <https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0>
- Qwen3 MoE: <https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39>
- DeepSeek V2 Lite: <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/604d5664dddd88a0433dbae533b7fe9472482de0>
- Muse Glimmer: <https://huggingface.co/meta-models/Muse-Glimmer-30B/tree/a4e59da52a7bc87ae7251dd5545c0dd437c44b68>
- FLAN-T5 XXL: <https://huggingface.co/google/flan-t5-xxl/tree/ae7c9136adc7555eeccc78cdd960dfd60fb346ce>
- FLAN-UL2: <https://huggingface.co/google/flan-ul2/tree/452d74ce28ac4a7f211d6ba3ef0717027f7a8074>
- Nemotron-H 3.5: <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/tree/a9904d24bcc1d289a1950fa9d2b978c47cf903b9>
- SmolLM-135M: <https://huggingface.co/HuggingFaceTB/SmolLM-135M/tree/1d461723eec654e65efdc40cf49301c89c0c92f4>
- Qwen3-4B: <https://huggingface.co/Qwen/Qwen3-4B/tree/531c80e289d6cff3a7cd8c0db8110231d23a6f7a>
- SmolLM3-3B: <https://huggingface.co/HuggingFaceTB/SmolLM3-3B/tree/1fa6fa281b5c635c89171a2b53769609cc1199c6>
- Qwen2-7B: <https://huggingface.co/Qwen/Qwen2-7B/tree/453ed1575b739b5b03ce3758b23befdb0967f40e>
- Gemma 3 4B IT: <https://huggingface.co/google/gemma-3-4b-it/tree/093f9f388b31de276ce2de164bdc2081324b9767>
- gpt-oss-20b: <https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee>
- Transformers 5.2.0 reference commit: <https://github.com/huggingface/transformers/tree/7d9754a05193eb79b1d86aa744b622b8068008cd>
