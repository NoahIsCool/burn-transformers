# Transformer architecture and innovation matrix

This document maps the transformer-side mechanisms used by the exact, pinned
checkpoint revisions targeted by this crate. It was assembled from the pinned
configuration, the model author's or framework's reference implementation, and
the corresponding Burn implementation. The matrix describes architecture, not
release support: a check mark does **not** mean that all real-weight acceptance
gates have passed. See [the model status page](ARCHITECTURE_COMPATIBILITY.md)
for that separate question.

For v0.1, only the pinned Q3.5-0.8 Metal/F16 base-text and FLAN-S Flex/F32 /
Metal/F32 profiles are promoted. This does not promote every checked mechanism
in their columns: quantized quality, real-weight MTP and multimodal paths remain
experimental. The [release record](acceptance/measurements/v0_1-release-20260910.json)
retains Qwen's user-accepted historical performance limitation. No architectural
feature was changed to adopt the new timing baseline.

The first matrix covers the checkpoints targeted for v0.1; a
[second matrix](#families-added-after-v01) covers the Llama, Gemma, and gpt-oss
families added since, at the pinned checkpoints their real-weight gates use.
[Shared mechanisms beyond the pinned graphs](#shared-mechanisms-beyond-the-pinned-graphs)
lists implemented components that no pinned checkpoint column exercises.

Only the executable text graph is counted. Vision towers, tokenizer behavior,
quantization formats, serving systems, and broad training recipes such as FLAN
instruction tuning and UL2's Mixture-of-Denoisers are deliberately outside the
matrix. Those recipes change how weights are learned, but not the transformer
operations executed by this transformer crate. Architecture-coupled router
objectives are included as optional forward diagnostics; the runtime computes
them only when requested and never adds them to inference logits.

## Experiment composition and training

The pinned-family matrix below describes reference graphs. The experiment API
also supports intentionally different graphs; their correctness is evaluated
with explicit mathematical and task-level tests, independently of checkpoint
support. See [EXPERIMENTS.md](EXPERIMENTS.md) and [TRAINING.md](TRAINING.md) for
complete consumer workflows.

| Implemented capability | Composition and validation boundary |
|---|---|
| Typed activation selection, observation, and edits | Select layers, tokens, heads, and channels; preserve gradients for intermediate losses; borrow registered Burn modules in ordered callbacks. Captures document axes and precision, and padding remains enforced after edits. |
| Staged execution and attention policies | Embed, run selected blocks, and normalize explicitly. Causal, bidirectional, custom connectivity, and additive biases are selectable where supported; cached interventions require a separate state contract. |
| Editable T5/Qwen recipes and parameter transfer | Change block geometry, extract prefixes, and transfer compatible tensors with explicit transformations and tie reports. Tiny synthetic tests and bounded real-prefix comparisons do not promote arbitrary modified graphs to checkpoint support. |
| Reusable mixers and residual blocks | Compose shared MLA, GQA, feed-forward modules, optional norms, and width adapters. The MLA encoder-decoder example trains a custom graph without DeepSeek family wiring. |
| Native Burn training compatibility | Full-precision graphs use Burn autodiff, optimizers, records, and optional Learner integration. Single parameter ownership, freezing, differentiable captures, dropout in inserted modules, and optimizer resume have explicit tests. Packed quantized inference remains separate from training support. |
| Family-generic experiment traits | `HookedModel` is implemented by every built-in family and `StagedModel` by every causal family (T5 stages through `T5Encoder`, modular seq2seq models through their encoder and decoder stage views); `AutoCausalLm` implements both, `AutoSeq2SeqLm` hooks, and `LoadedModel` forwards both. The workbench runs one activation-cache, ablation, and staged-patching function on three families, and staged stacks recompose the full stack exactly. |
| Explicit initialization | `InitPolicy` assigns initializers by parameter role with depth-scaled residual projections, path overrides, constants, and lazy application. Hugging Face `initializer_range` normal initialization and GPT-2 residual scaling are implemented and tested on synthetic models; this makes no checkpoint claim. |
| Positional composition | `RotaryAttention` adds RoPE with any shared scaling rule (including YaRN and dynamic NTK, selected by the evaluated context) to composed residual blocks; `GroupedQueryAttention` used directly as a block mixer is position-free. |

The [v0.3 training record](acceptance/measurements/v0_3-training-validation-20260910/README.md)
retains exact source identities and executed experiments. Grokking actually
trains modular addition and demonstrates delayed held-out generalization and
frequency interventions. Newly trained Tuned Lens translators improve held-out
KL, and a Greater-Than activation adapter improves its bounded transfer task.
Clock and Pizza training reaches task accuracy but does **not** reproduce the
specified representative circular mechanism. These results are scoped to their
recorded protocols; documentation edits do not turn archived measurements into
current-source validation. Experimental GPT-2 and GPT-NeoX/Pythia implementations
support the pinned interpretability workflows without expanding the advertised
release-support matrix.

## Models and source basis

The short names below are the column headings in the main matrix. Checkpoint
configuration links are immutable; a framework reference uses its canonical
source when the checkpoint does not bundle model code. “Local implementation”
names the family wiring; shared kernels are linked from every innovation
description below.

| Short name | Exact checkpoint and immutable configuration | Primary reference implementation | Local implementation |
|---|---|---|---|
| **Q3.5-0.8** | [Qwen/Qwen3.5-0.8B @ `2fc063…`](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json) | [Qwen3.5 reference](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/qwen3_5/modeling_qwen3_5.py), [MTP reference](https://github.com/vllm-project/vllm/blob/bb363db9a5ec2edc7b39e99b00af363a89d1fb81/vllm/model_executor/models/qwen3_5_mtp.py) | [`qwen3_5`](src/models/qwen3_5/model.rs) |
| **Q3.8-27** | [Qwen/Qwen3.8-27B @ `1d4bf0…`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json) | [Qwen3.5 reference](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/qwen3_5/modeling_qwen3_5.py) | [`qwen3_5`](src/models/qwen3_5/model.rs) |
| **Q3-MoE** | [Qwen/Qwen3-30B-A3B @ `ad44e7…`](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json) | [Qwen3-MoE reference, Transformers 5.2.0 @ `7d9754…`](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) | [`qwen3_moe`](src/models/qwen3_moe/model.rs) |
| **DSV2-L** | [deepseek-ai/DeepSeek-V2-Lite @ `604d56…`](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/config.json) | [DeepSeek's pinned reference](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/modeling_deepseek.py), [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434) | [`deepseek_v2`](src/models/deepseek_v2/model.rs) |
| **Muse** | [meta-models/Muse-Glimmer-30B @ `a4e59d…`](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/config.json) | [Muse-Glimmer reference @ `cfa309…`](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/muse_glimmer/modular_muse_glimmer.py) | [`muse_glimmer`](src/models/muse_glimmer/model.rs) |
| **FLAN-S** | [google/flan-t5-small @ `0fc9dd…`](https://huggingface.co/google/flan-t5-small/blob/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab/config.json) | [Pinned T5 reference](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/t5/modeling_t5.py), [T5 paper](https://arxiv.org/abs/1910.10683) | [`t5`](src/models/t5/model.rs) |
| **FLAN-XXL** | [google/flan-t5-xxl @ `ae7c91…`](https://huggingface.co/google/flan-t5-xxl/blob/ae7c9136adc7555eeccc78cdd960dfd60fb346ce/config.json) | [Pinned T5 reference](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/t5/modeling_t5.py), [FLAN paper](https://arxiv.org/abs/2210.11416) | [`t5`](src/models/t5/model.rs) |
| **FLAN-UL2** | [google/flan-ul2 @ `452d74…`](https://huggingface.co/google/flan-ul2/blob/452d74ce28ac4a7f211d6ba3ef0717027f7a8074/config.json) | [Pinned T5 reference](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/t5/modeling_t5.py), [UL2 paper](https://arxiv.org/abs/2205.05131) | [`t5`](src/models/t5/model.rs) |
| **Nem-3.5** | [NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 @ `a9904d…`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/config.json) | [NVIDIA architecture/recipe](https://docs.nvidia.com/nemotron/nightly/nemotron/lightning35/pretrain.html), [Nemotron-H base reference @ `cfa309…`](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/nemotron_h/modeling_nemotron_h.py), [vLLM MTP graph @ `6e448d…`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/nemotron_h_mtp.py) | [`nemotron_h`](src/models/nemotron_h/model.rs) |
| **Nem-Nano** | [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 @ `bf77c3…`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/config.json) | [Pinned NVIDIA reference](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/modeling_nemotron_h.py), [technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) | [`nemotron_h`](src/models/nemotron_h/model.rs) |
| **SmolLM-135** | [HuggingFaceTB/SmolLM-135M @ `1d4617…`](https://huggingface.co/HuggingFaceTB/SmolLM-135M/blob/1d461723eec654e65efdc40cf49301c89c0c92f4/config.json) | [Llama reference, Transformers 5.2.0](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/llama/modeling_llama.py) | [`llama`](src/models/llama/model.rs) |
| **Q3-4** | [Qwen/Qwen3-4B @ `531c80…`](https://huggingface.co/Qwen/Qwen3-4B/blob/531c80e289d6cff3a7cd8c0db8110231d23a6f7a/config.json) | [Qwen3 reference, Transformers 5.2.0](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/qwen3/modeling_qwen3.py) | [`llama`](src/models/llama/model.rs) |
| **SmolLM3-3** | [HuggingFaceTB/SmolLM3-3B @ `1fa6fa…`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/1fa6fa281b5c635c89171a2b53769609cc1199c6/config.json) | [SmolLM3 reference, Transformers 5.2.0](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/smollm3/modeling_smollm3.py) | [`llama`](src/models/llama/model.rs) |
| **Q2-7** | [Qwen/Qwen2-7B @ `453ed1…`](https://huggingface.co/Qwen/Qwen2-7B/blob/453ed1575b739b5b03ce3758b23befdb0967f40e/config.json) | [Qwen2 reference, Transformers 5.2.0](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/qwen2/modeling_qwen2.py) | [`llama`](src/models/llama/model.rs) |
| **Gemma3-4** | [google/gemma-3-4b-it @ `093f9f…`](https://huggingface.co/google/gemma-3-4b-it/blob/093f9f388b31de276ce2de164bdc2081324b9767/config.json), text tower | [Gemma 3 reference, Transformers 5.2.0](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/gemma3/modeling_gemma3.py) | [`gemma`](src/models/gemma/model.rs) |
| **gpt-oss-20** | [openai/gpt-oss-20b @ `6cee5e…`](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json) | [gpt-oss reference](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/gpt_oss/modeling_gpt_oss.py) and [MXFP4 integration](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/integrations/mxfp4.py), Transformers 5.2.0 | [`gpt_oss`](src/models/gpt_oss/model.rs) |

The last six checkpoints were added after v0.1. Their columns appear in the
[second matrix](#families-added-after-v01); Mistral, OLMo 2, and Gemma 2 are
covered by tiny Transformers 5.2.0 goldens only and have no pinned-checkpoint
column. Llama 3 RoPE scaling has a tiny golden and, in addition, a real
Llama-3.1-8B gate run on Flex and Metal in F32 (within `7.6e-5`); that
checkpoint gets a column once its full revision is recorded (the repository
names only the prefix `d04e592b`).

Qwen3.8's inner text model is officially serialized as `qwen3_5_text`, not
`qwen3_8_text`. Nemotron Nano is included because the crate validates its
configuration shape; that is not a claim of real-weight acceptance.

## Innovation matrix

**Legend:** `✓` means the exact pinned graph uses the mechanism throughout the
relevant component, `enc/dec` identifies a T5 stack, and `—` means absent. In
block rows, `n/L` means affected/base blocks; other slash values name their unit
in the row or cell (for example top-k/experts, rotary/head channels, or
buckets/maximum distance). Counts after `+ MTP` include an auxiliary prediction
body in addition to the base backbone. Every innovation name links to its
mathematical and implementation description.

| Transformer innovation | Q3.5-0.8 | Q3.8-27 | Q3-MoE | DSV2-L | Muse | FLAN-S | FLAN-XXL | FLAN-UL2 | Nem-3.5 | Nem-Nano |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [Decoder-only causal stack](#innovation-decoder-only) | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — | ✓ | ✓ |
| [Encoder–decoder sequence-to-sequence stack](#innovation-encoder-decoder) | — | — | — | — | — | ✓ | ✓ | ✓ | — | — |
| [Bidirectional encoder self-attention](#innovation-bidirectional-encoder) | — | — | — | — | — | ✓ | ✓ | ✓ | — | — |
| [Decoder cross-attention](#innovation-cross-attention) | — | — | — | — | — | ✓ | ✓ | ✓ | — | — |
| [Pre-normalized residual path](#innovation-pre-norm) | ✓ | ✓ | ✓ | ✓ | ✓ + post-norms | ✓ | ✓ | ✓ | ✓ | ✓ |
| [Heterogeneous block scheduling](#innovation-heterogeneous-schedule) | 18 Delta + 6 attn | 48 Delta + 16 attn | — | 1 dense + 26 MoE | 39 SWA + 13 full | — | — | — | 23 Mamba + 23 MoE + 6 attn | 23 Mamba + 23 MoE + 6 attn |
| [Full-context softmax self-attention](#innovation-full-attention) | 6/24 + MTP | 16/64 + MTP | 48/48 | 27/27 MLA | 13/52 | enc/dec | enc/dec | enc/dec | 6/52 + MTP | 6/52 |
| [Multi-head attention (MHA)](#innovation-mha) | — | — | — | — | — | enc/dec | enc/dec | enc/dec | — | — |
| [Grouped-query attention (GQA)](#innovation-gqa) | 6/24 + MTP | 16/64 + MTP | 48/48 | — | 52/52 | — | — | — | 6/52 + MTP | 6/52 |
| [Multi-head latent attention (MLA)](#innovation-mla) | — | — | — | 27/27 | — | — | — | — | — | — |
| [Absorbed latent-cache decoding](#innovation-absorbed-mla) | — | — | — | ✓ | — | — | — | — | — | — |
| [Sliding-window attention](#innovation-sliding-attention) | — | — | — | — | 39/52, window 2048 | — | — | — | — | — |
| [NoPE attention](#innovation-nope) | — | — | — | — | 13/52 | — | — | — | 6/52 + MTP | 6/52 |
| [Per-head Q/K normalization](#innovation-qk-norm) | 6/24 + MTP | 16/64 + MTP | 48/48 | — | 52/52, scale-free | — | — | — | — | — |
| [Multiplicative attention-output gating](#innovation-attention-gate) | 6/24 + MTP | 16/64 + MTP | — | — | 52/52 | — | — | — | — | — |
| [Extra normalized-query scaling](#innovation-query-scaling) | — | — | — | — | ×3.87 | — | — | — | — | — |
| [Unscaled dot-product attention](#innovation-unscaled-attention) | — | — | — | — | — | ✓ | ✓ | ✓ | — | — |
| [Rotary position embedding (RoPE)](#innovation-rope) | partial, attn only | partial, attn only | ✓ | split subspace | SWA only | — | — | — | — | — |
| [Partial RoPE](#innovation-partial-rope) | 64/256 channels | 64/256 channels | — | — | — | — | — | — | — | — |
| [Decoupled NoPE/RoPE head channels](#innovation-decoupled-rope) | — | — | — | 128 + 64 | — | — | — | — | — | — |
| [Interleaved multi-axis RoPE (MRoPE)](#innovation-mrope) | config 11/11/10; text = 1-D | config 11/11/10; text = 1-D | — | — | — | — | — | — | — | — |
| [YaRN context scaling](#innovation-yarn) | — | — | — | factor 40 | — | — | — | — | — | — |
| [Bucketed relative-position bias](#innovation-t5-relative-bias) | — | — | — | — | — | self-attn 32/128 | self-attn 32/128 | self-attn 32/128 | — | — |
| [Gated DeltaNet](#innovation-deltanet) | 18/24 | 48/64 | — | — | — | — | — | — | — | — |
| [Mamba2 selective state-space mixer](#innovation-mamba2) | — | — | — | — | — | — | — | — | 23/52 | 23/52 |
| [Depthwise causal convolution](#innovation-depthwise-convolution) | Delta layers, width 4 | Delta layers, width 4 | — | — | — | — | — | — | Mamba layers, width 4 | Mamba layers, width 4 |
| [Fixed-state recurrent decoding](#innovation-recurrent-decode) | Delta only | Delta only | — | — | — | — | — | — | Mamba2 only, 23/52 | Mamba2 only, 23/52 |
| [Chunk-parallel recurrent prefill](#innovation-chunked-prefill) | Delta | Delta | — | — | — | — | — | — | Mamba2 | Mamba2 |
| [SwiGLU / gated-SiLU feed-forward](#innovation-swiglu) | ✓ | ✓ | experts | dense + experts | ✓ | — | — | ✓ | — | — |
| [Gated-GELU feed-forward](#innovation-gated-gelu) | — | — | — | — | — | ✓ | ✓ | — | — | — |
| [ReLU² feed-forward](#innovation-relu-squared) | — | — | — | — | — | — | — | — | experts | experts |
| [Sparse top-k mixture of experts](#innovation-sparse-moe) | — | — | 8/128 | 6/64 | — | — | — | — | 6/128 | 6/128 |
| [Selected-expert-only dispatch](#innovation-selected-dispatch) | — | — | ✓ | ✓ | — | — | — | — | ✓ | ✓ |
| [Always-active shared experts](#innovation-shared-experts) | — | — | — | one 2×-wide module | — | — | — | — | one 2×-wide module | one 2×-wide module |
| [Post-top-k route renormalization](#innovation-route-renormalization) | — | — | ✓ | — | — | — | — | — | ✓ | ✓ |
| [Correction-biased sigmoid routing](#innovation-sigmoid-routing) | — | — | — | — | — | — | — | — | ✓ | ✓ |
| [Auxiliary load-balancing objective](#innovation-router-aux-loss) | — | — | coefficient 0.001 | coefficient 0.001 | — | — | — | — | seq-aux 0.0001 (recipe) | seq-aux 0.0001 (recipe) |
| [RMS normalization](#innovation-rmsnorm) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| [Zero-centered RMSNorm parameters](#innovation-zero-centered-rmsnorm) | ✓ | ✓ | — | — | block norms | — | — | — | — | — |
| [Round-before-scale RMSNorm](#innovation-round-before-scale) | — | — | ✓ | ✓ | — | — | — | — | ✓ | ✓ |
| [Sandwich normalization](#innovation-sandwich-norm) | — | — | — | — | ✓ | — | — | — | — | — |
| [Scale-free embedding normalization](#innovation-scale-free-embedding) | — | — | — | — | ✓ | — | — | — | — | — |
| [Shared encoder/decoder token embedding](#innovation-shared-stack-embedding) | — | — | — | — | — | ✓ | ✓ | ✓ | — | — |
| [Tied input embedding and LM head](#innovation-tied-lm-head) | ✓ | — | — | — | — | — | — | ✓ (schema default) | — | — |
| [Final-logit tanh softcap](#innovation-logit-softcap) | — | — | — | — | 20 | — | — | — | — | — |
| [Multi-token/next-token prediction head](#innovation-mtp) | 1 block | 1 block | — | — | — | — | — | — | 1 stage: attn + MoE | — |
| [Preallocated projected-KV cache](#innovation-kv-cache) | attn layers | attn layers | ✓ | —, latent cache | ✓ | decoder self-attn | decoder self-attn | decoder self-attn | attn layers | attn layers |
| [Sliding ring KV cache](#innovation-ring-cache) | — | — | — | — | ✓ | — | — | — | — | — |
| [Immutable cross-attention KV cache](#innovation-cross-kv-cache) | — | — | — | — | — | ✓ | ✓ | ✓ | — | — |
| [External recurrent-state cache](#innovation-recurrent-cache) | Delta | Delta | — | — | — | — | — | — | Mamba2 | Mamba2 |
| [Transactional speculative cache](#innovation-transactional-cache) | target + MTP | target + MTP | — | target only | — | — | — | — | target + MTP | — |
| [Lossless speculative decoding](#innovation-speculative-decoding) | MTP, batch 1 | MTP, batch 1 | — | — | — | — | — | — | MTP, batch 1 | — |
| [Cache-aware beam search](#innovation-beam-search) | — | — | — | — | — | ✓ | ✓ | ✓ | — | — |


<a id="families-added-after-v01"></a>
### Families added after v0.1

Same legend. Every column is a real-weight gate subject, run on Flex and Metal in F32; see the
[status page](ARCHITECTURE_COMPATIBILITY.md#real-checkpoint-evidence) for its
measured parity.

| Transformer innovation | SmolLM-135 | Q3-4 | SmolLM3-3 | Q2-7 | Gemma3-4 | gpt-oss-20 |
|---|---:|---:|---:|---:|---:|---:|
| [Decoder-only causal stack](#innovation-decoder-only) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| [Pre-normalized residual path](#innovation-pre-norm) | ✓ | ✓ | ✓ | ✓ | ✓ + post-norms | ✓ |
| [Grouped-query attention (GQA)](#innovation-gqa) | 9 q / 3 kv | 32 / 8 | 16 / 4 | 28 / 4 | 8 / 4 | 64 / 8 |
| [Sliding-window attention](#innovation-sliding-attention) | — | — | — | — | 29/34, window 1024 | 12/24, window 128 |
| [NoPE attention](#innovation-nope) | — | — | 9/36 | — | — | — |
| [Per-head Q/K normalization](#innovation-qk-norm) | — | 36/36 | — | — | 34/34 | — |
| [Attention projection biases](#innovation-projection-bias) | — | — | — | Q/K/V | — | Q/K/V/O |
| [Learned attention sinks](#innovation-attention-sinks) | — | — | — | — | — | 24/24 |
| [Rotary position embedding (RoPE)](#innovation-rope) | θ 1e4 | θ 1e6 | θ 5e6, 27/36 | θ 1e6 | local θ 1e4; global θ 1e6, linear ×8 | YaRN |
| [YaRN context scaling](#innovation-yarn) | — | — | — | — | — | factor 32, untruncated |
| [SwiGLU / gated-SiLU feed-forward](#innovation-swiglu) | ✓ | ✓ | ✓ | ✓ | — | — |
| [Gated-GELU feed-forward](#innovation-gated-gelu) | — | — | — | — | ✓ (tanh GELU) | — |
| [Clamped SwiGLU](#innovation-clamped-swiglu) | — | — | — | — | — | experts |
| [Sparse top-k mixture of experts](#innovation-sparse-moe) | — | — | — | — | — | 4/32 |
| [Shared router policies](#innovation-router-policies) | — | — | — | — | — | biased, top-k then softmax |
| [MXFP4 expert weights](#innovation-mxfp4) | — | — | — | — | — | ✓ |
| [RMS normalization](#innovation-rmsnorm) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| [Zero-centered RMSNorm parameters](#innovation-zero-centered-rmsnorm) | — | — | — | — | ✓ | — |
| [Sandwich normalization](#innovation-sandwich-norm) | — | — | — | — | ✓ | — |
| [Scaled token embedding](#innovation-embedding-scale) | — | — | — | — | `sqrt(d)` | — |
| [Tied input embedding and LM head](#innovation-tied-lm-head) | ✓ | ✓ | ✓ | — | ✓ | — |
| [Preallocated projected-KV cache](#innovation-kv-cache) | ✓ | ✓ | ✓ | ✓ | 5 global layers | 12 full layers |
| [Sliding ring KV cache](#innovation-ring-cache) | — | — | — | — | 29 rings | 12 rings |
| [Batched cached decoding](#innovation-batched-decoding) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

<a id="shared-mechanisms-beyond-the-pinned-graphs"></a>
### Shared mechanisms beyond the pinned graphs

These components are implemented in shared code and validated against
reference implementations on synthetic or tiny checkpoints, but no pinned
checkpoint column above uses them. They make no checkpoint claim.

| Mechanism | Validation | Where it is available |
| --- | --- | --- |
| [Llama 3, LongRoPE, dynamic-NTK, and proportional RoPE scaling](#innovation-rope-scaling) | Inverse-frequency goldens from Transformers' `modeling_rope_utils.py` (5.2.0; 5.8.0 for proportional); a Llama 3.1-style tiny golden; Llama 3 scaling also in a real Llama-3.1-8B gate run (Flex and Metal F32, within `7.6e-5`) | Every RoPE family through the shared translator; custom specs; `RotaryAttention` |
| [Attention logit soft cap](#innovation-attention-softcap) | Gemma 2 tiny golden, including a saturated cap | Gemma 2; custom specs |
| [Post-sublayer norms and full-width Q/K norm](#innovation-post-norm) | OLMo 2 tiny golden | Llama family (`olmo2`); custom specs |
| [Chunked-local connectivity](#innovation-sliding-attention) | Scalar mask oracle, property tests, fuzzing | Custom specs and shared attention |
| [Differential attention](#innovation-differential-attention) | Transformers DiffLlama | Custom specs and shared attention |
| [Sigmoid, group-limited, and correction-biased routing](#innovation-router-policies) | Transformers Mixtral, Qwen3-MoE, DeepSeek-V2/V3, and gpt-oss routing goldens | DeepSeek-V2 (`noaux_tc`, `group_limited_greedy`); custom specs |
| [Mamba-2 in modular specs](#innovation-mamba2) | Bit-exact against the shared kernel; stateless versus cached and padded rows | Custom causal stacks and seq2seq decoders |

## Mathematical and technical descriptions

### Topology and residual organization

<a id="innovation-decoder-only"></a>
### Decoder-only causal stack

A decoder-only language model factorizes a token sequence as
$p(x_{1:n})=\prod_t p(x_t\mid x_{<t})$. At layer input $X$, causal self-attention
adds a mask $M_{ij}=-\infty$ for $j>i$, so the hidden state at position $i$
cannot depend on a future token. The vocabulary projection of $h_i$ therefore
defines the distribution for the following token. DeltaNet and Mamba2 layers
enforce the same directionality through a left-to-right recurrence rather than
an explicit triangular attention matrix.

**Implementation:** family wiring in
[`qwen3_5/model.rs`](src/models/qwen3_5/model.rs),
[`qwen3_moe/model.rs`](src/models/qwen3_moe/model.rs),
[`deepseek_v2/model.rs`](src/models/deepseek_v2/model.rs),
[`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs), and
[`nemotron_h/model.rs`](src/models/nemotron_h/model.rs). **Reference:** the
[original Transformer](https://arxiv.org/abs/1706.03762).

<a id="innovation-encoder-decoder"></a>
### Encoder–decoder sequence-to-sequence stack

The encoder first builds contextual source states $H=\operatorname{Enc}(x)$.
The autoregressive decoder then represents
$p(y\mid x)=\prod_t p(y_t\mid y_{<t},H)$, alternating causal target
self-attention with cross-attention to $H$. This separates source understanding
from target generation and lets the encoder run exactly once per request.

**Implementation:** [`t5/model.rs`](src/models/t5/model.rs) and the public
sequence-to-sequence contracts in [`architecture.rs`](src/architecture.rs).
**Reference:** [T5](https://arxiv.org/abs/1910.10683).

<a id="innovation-bidirectional-encoder"></a>
### Bidirectional encoder self-attention

Encoder attention omits the causal triangle. Apart from padding, every source
position may attend to every other source position:
$A=\operatorname{softmax}(QK^\mathsf{T}+B+M_{\text{pad}})V$. Consequently each
encoder output can encode both left and right context; it is not itself a valid
autoregressive next-token state.

**Implementation:** [`T5Stack::forward_encoder`](src/models/t5/model.rs) and the
bidirectional bucket mode in [`position/t5.rs`](src/position/t5.rs).
**Reference:** [T5's encoder–decoder architecture](https://arxiv.org/abs/1910.10683).

<a id="innovation-cross-attention"></a>
### Decoder cross-attention

Cross-attention uses decoder states for queries but encoder states for keys and
values: $Q=H_dW_Q$, $K=H_eW_K$, $V=H_eW_V$, and
$Y=\operatorname{softmax}(QK^\mathsf{T}+M_e)V$. It is the path through which
every generated target token retrieves source information. Since $H_e$ does not
change during decoding, its projected $K,V$ tensors can be computed once.

**Implementation:** [`T5Block`](src/models/t5/model.rs) and its stateful path in
[`t5/model/decode.rs`](src/models/t5/model/decode.rs). **Reference:**
[T5](https://arxiv.org/abs/1910.10683).

<a id="innovation-pre-norm"></a>
### Pre-normalized residual path

The canonical pre-norm update is $x' = x + F(N(x))$, rather than
$N(x+F(x))$. The residual branch therefore retains an exact identity path for
signals and gradients while each mixer sees normalized input. All listed models
use this organization. Muse additionally normalizes the output of each
sublayer before residual addition, as described under
[sandwich normalization](#innovation-sandwich-norm).

**Implementation:** concrete blocks in
[`qwen3_5/model.rs`](src/models/qwen3_5/model.rs),
[`qwen3_moe/model.rs`](src/models/qwen3_moe/model.rs),
[`deepseek_v2/model.rs`](src/models/deepseek_v2/model.rs),
[`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs),
[`t5/model.rs`](src/models/t5/model.rs), and
[`nemotron_h/model.rs`](src/models/nemotron_h/model.rs), with normalization from
[`normalization.rs`](src/normalization.rs). **Reference:** the exact ordering in
the pinned implementations linked in [Models and source basis](#models-and-source-basis).

<a id="innovation-heterogeneous-schedule"></a>
### Heterogeneous block scheduling

Instead of repeating one transformer block, a schedule chooses a mixer or FFN
kind $F_{s_\ell}$ for residual block $\ell$:
$x_{\ell+1}=x_\ell+F_{s_\ell}(N(x_\ell))$. Qwen uses three DeltaNet layers per
full-attention layer. Muse uses three sliding-RoPE blocks per full NoPE block.
DeepSeek changes only its FFN branch after the first layer. Nemotron's 52-entry
schedule contains 23 Mamba2, 23 MoE, and 6 NoPE-GQA residual blocks; a Nemotron
“block” is one mixer/FFN residual operation, not an attention-plus-FFN pair.

**Implementation:** schedule parsing and exact wiring in
[`qwen3_5.rs`](src/models/qwen3_5.rs),
[`muse_glimmer.rs`](src/models/muse_glimmer.rs),
[`deepseek_v2.rs`](src/models/deepseek_v2.rs), and
[`nemotron_h.rs`](src/models/nemotron_h.rs).

### Attention mechanisms

<a id="innovation-full-attention"></a>
### Full-context softmax self-attention

For each head, projected queries, keys, and values form
$Y=\operatorname{softmax}(QK^\mathsf{T}/\sqrt{d_h}+M+B)V$. The mask $M$
enforces causality or padding and $B$ carries an optional positional bias. A
full-context layer can directly compare every permitted token pair, but prefill
cost is $O(n^2d_h)$ and its ordinary decode cache grows as $O(nd_h)$ per head.
Most families use the displayed inverse-root scale. T5 instead uses scale one.
DeepSeek computes the same softmax over an MLA factorization and uses its YaRN-
adjusted scale
$\bigl(1/\sqrt{192}\bigr)\operatorname{mscale}(40,0.707)^2$.

**Implementation:** shared attention and its opt-in eager-FP32-softmax precision
mode in [`attention.rs`](src/attention.rs), MLA in
[`mla.rs`](src/mla.rs), and Muse's specialized
selection in [`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs).
**Reference:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

<a id="innovation-mha"></a>
### Multi-head attention (MHA)

MHA gives every one of $H$ query heads its own key and value head. With per-head
width $d_k$, $Q,K,V$ have inner width $Hd_k$, and head results are concatenated
before $W_O$. T5 allows $Hd_k\ne d_{model}$: FLAN-T5-small uses
$6\times64=384$ attention channels around a 512-wide residual stream.

**Implementation:** `query_heads == key_value_heads` in
[`GroupedQueryAttention`](src/attention.rs), configured by
[`T5Attention`](src/models/t5/model.rs). **Reference:**
[the Transformer MHA definition](https://arxiv.org/abs/1706.03762).

<a id="innovation-gqa"></a>
### Grouped-query attention (GQA)

GQA keeps $H_q$ independently projected query heads but only $H_{kv}$ key/value
heads, with $H_q$ divisible by $H_{kv}$. Query head $h$ consumes KV head
$\lfloor h/(H_q/H_{kv})\rfloor$. It approaches MHA expressivity while shrinking
the decode cache and K/V projection bandwidth by $H_q/H_{kv}$. The cache stores
native KV heads, and attention folds query groups instead of repeating K/V: the
$g=H_q/H_{kv}$ query heads of a group become extra query rows of one call over
their shared KV head, $[B,H_q,Q,d]\to[B,H_{kv},gQ,d]$, which is mathematically
identical because softmax is row-wise. Only square causal prefill on the
backend-native path keeps the backend's fused `is_causal` kernel and repeats
K/V there; cached decoding and suffixes always fold.
Causality is bottom-right aligned on every backend: query row $i$ of $Q$ sits
at key position $i+K-Q$, which also fixes CubeCL flash kernels that align
top-left.

**Implementation:** [`attention/dispatch.rs`](src/attention/dispatch.rs) and the
native-head cache in [`cache/tensor/kv.rs`](src/cache/tensor/kv.rs).
**Reference:** [Grouped-Query Attention](https://arxiv.org/abs/2305.13245).

<a id="innovation-mla"></a>
### Multi-head latent attention (MLA)

MLA compresses each token to a shared latent $c_t=W_{DKV}x_t\in\mathbb{R}^r$.
Per-head up-projections recover content keys and values,
$k^C_{t,h}=W^K_hc_t$ and $v_{t,h}=W^V_hc_t$. A small, shared rotary key
$k^R_t$ is retained separately, while each query is split into content and
rotary channels. DeepSeek-V2-Lite uses $r=512$, 16 heads, 128 NoPE content
channels, 64 rotary channels, and 128 value channels. Its query path is direct
because the pinned `q_lora_rank` is null.

**Implementation:** [`mla.rs`](src/mla.rs).
**Reference:** [DeepSeek-V2](https://arxiv.org/abs/2405.04434) and the
[pinned reference code](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/modeling_deepseek.py).

<a id="innovation-absorbed-mla"></a>
### Absorbed latent-cache decoding

MLA's up-projections can be algebraically absorbed during decode:
$q_h^C(W_h^Kc_t)=(q_h^CW_h^K)c_t$ and
$\sum_t p_t(W_h^Vc_t)=W_h^V(\sum_t p_tc_t)$. Historical expanded keys and
values therefore need not be materialized. The local cache stores only the
512-wide latent and 64-wide rotary key—576 scalars per layer and token—for the
pinned Lite model. This is an optimized realization: DeepSeek's pinned Python
reference expands K/V into its generic dynamic cache.

**Implementation:** absorbed math in
[`mla.rs`](src/mla.rs) and external state in
[`deepseek_v2/model/decode.rs`](src/models/deepseek_v2/model/decode.rs).
**Reference:** [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

<a id="innovation-sliding-attention"></a>
### Sliding-window attention

For query position $i$ and window $W$, local causal attention admits only
$\max(0,i-W+1)\le j\le i$. A sparse window kernel can reduce prefill work to
$O(nW)$ once $n\gg W$, and decode needs only the newest $W$ projected K/V
entries. The one-shot portable Burn path still applies a dense $n\times n$
mask during fresh-prompt prefill, so its compute remains $O(n^2)$. Causal
generation can instead split the prompt into cache-backed chunks of at most
$C$ physical columns. That does not change global attention's total quadratic
work, but bounds each temporary score/mask to $O(HCn)$ rather than $O(Hn^2)$;
sliding layers retain only their ring-backed history. Ragged chunks preserve
each row's latest valid logits even when a later chunk is padding-only. Muse
uses window 2,048 in 39 of 52 text blocks, interleaved with 13 global blocks
that restore long-range connectivity. When a valid-token mask contains padding
or an interior hole, Muse and custom modular attention measure $i$ and $j$ in
cumulative valid-token ordinals. Invalid source columns consume no window
distance, matching the cache path's semantic row compaction for left padding,
right padding, and interior holes. Explicit positional-encoding IDs may contain
offsets or gaps; they do not change this valid-token window distance. Custom full attention also
preserves key padding when no sliding window is requested.

The Llama family, Gemma, gpt-oss, and custom models draw these masks from
one shared rule, `AttentionConnectivity`: global, a trailing sliding window of $W$ positions,
or chunked-local attention in which position $p$ sees only its aligned chunk
$[C\lfloor p/C\rfloor, C\lfloor p/C\rfloor + C)$. Masks are padding-aware,
use absolute positions (a cache that dropped its first keys passes their count
as the key start), and act exactly as unrestricted when a window exceeds the
key span. Gemma 2/3, gpt-oss, and the Llama family's Mistral and Qwen
sliding schedules use window-sized ring caches per layer group; Muse's cached
sliding layers slice each query chunk to the union of its rows' windows.

**Implementation:** the shared rule in
[`attention/connectivity.rs`](src/attention/connectivity.rs); masking and layer selection in
[`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs), cached execution in
[`muse_glimmer/model/decode.rs`](src/models/muse_glimmer/model/decode.rs), and
ring planning in [`cache/ring.rs`](src/cache/ring.rs). The custom reference path
uses [`custom/common.rs`](src/models/custom/common.rs), with compact token-history
recomputation rather than a projected-KV ring. Bounded orchestration is
implemented by `CausalGenerationRequest::prefill_chunk_size` in
[`generation/causal.rs`](src/generation/causal.rs).
**Reference:** the [pinned Muse configuration](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/config.json).

<a id="innovation-nope"></a>
### NoPE attention

NoPE applies neither an absolute/rotary transformation nor a learned relative
position bias to Q/K. Attention scores are content-only apart from their causal
mask. Causality still exposes order, so NoPE does not make a decoder
permutation-invariant. Muse uses global NoPE layers; both pinned Nemotron
checkpoints use NoPE in every GQA block even though their schemas retain
`rope_theta` and `partial_rotary_factor` compatibility fields.

**Implementation:** Muse's per-layer choice in
[`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs) and Nemotron's direct
position-free attention in [`nemotron_h/model.rs`](src/models/nemotron_h/model.rs).
**Reference:** [NVIDIA's Nemotron technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
and the [pinned Muse reference](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/muse_glimmer/modular_muse_glimmer.py).

<a id="innovation-qk-norm"></a>
### Per-head Q/K normalization

Before position transforms and dot products, each head is rescaled as
$\hat q=q/\sqrt{\operatorname{mean}(q^2)+\epsilon}\odot\gamma_q$ and likewise
for $k$. This controls score magnitudes independently of residual-stream scale.
Qwen learns per-channel scales; Qwen3.5/3.8 stores those scales zero-centered.
Qwen3 dense and Gemma 3 use the same per-head form. Muse instead uses
parameter-free (“scale-free”) head normalization. OLMo 2 normalizes the whole
projected width before splitting heads (`QueryKeyNormScope::FullWidth`).

**Implementation:** learned normalization in [`attention.rs`](src/attention.rs)
and Muse's variant in [`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs).
**Reference:** the pinned Qwen and Muse implementations in
[Models and source basis](#models-and-source-basis).

<a id="innovation-attention-gate"></a>
### Multiplicative attention-output gating

An additional projection produces $g$ and modulates the concatenated head
result before the output projection:
$y=W_O(a\odot\sigma(g))$. Qwen stores `[query | gate]` per head inside a fused
Q projection; Muse has a distinct gate projection. Qwen3.8 serializes
`output_gate_type="swish"`, but the pinned reference does not consult that
field to choose the full-attention gate: full attention explicitly uses
$\sigma(g)$. DeltaNet's output gated RMSNorm uses SiLU/swish through
`hidden_act="silu"`, so the serialized field is consistent metadata rather than
a full-attention activation switch.

**Implementation:** fused Qwen gates in [`attention.rs`](src/attention.rs) and
the separate Muse path in [`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs).
**Reference:** the pinned [Qwen3.8 config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
and [Qwen3.5 code](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

<a id="innovation-query-scaling"></a>
### Extra normalized-query scaling

Muse normalizes Q and K without learned scales, then applies
$q\leftarrow3.87q$. Standard $1/\sqrt{128}$ scaling is still applied exactly
once by scaled dot-product attention, so the effective score multiplier is
$3.87/\sqrt{128}$. The pinned eager graph first materializes scores in the query
dtype, promotes only softmax to FP32, casts the probabilities back, and then
multiplies by V. Burn 0.21 flash attention instead keeps QK accumulation in
strict FP32; an adversarial Metal F16 gate confirms that omitting the score
rounding changes output bits. Muse therefore opts into the shared explicit
eager mode for F16, while F32 retains backend-native arithmetic.

The production Muse path does not advertise flash attention. Burn 0.21's
ordinary `metal` feature does not enable its attention autotuner, and the
opt-in CubeCL flash candidate uses the numerically different strict-FP32 QK
graph. The opt-in Metal trace gate observes `cubek_attention` for that control,
observes no flash launch for Muse's explicit path, and distinguishes eager F16
bits `[1a3b, 3268, 157f, 3a5d]` from flash bits
`[1a38, 3266, 157e, 3a5e]`. The 4,097-token discriminator executes Muse as
nine chunks of at most 512 queries. Instead, a validated
[`AttentionExecutionPolicy`](src/attention.rs)
keeps short oracle prompts in one unchanged call and splits longer prompts only
along the query axis. Every query chunk still sees its entire allowed key set,
so this is exact row-wise softmax rather than an attention approximation.
Synchronized cleanup between chunks changes peak score-like storage from
$O(QK)$ to $O(CK)$ for configured query bound $C$; the default caps each plane
at $2^{26}$ elements and at 512 queries. At Muse's 32 heads and 32K global
context that produces 512 chunks of 64 queries, each with at most $2^{26}$
score elements, instead of one 34-billion-element plane. Unmasked or explicitly
all-valid sliding layers also slice each chunk's union of visible keys, at most
$W+C-1$ keys for window $W$; genuinely padded inputs retain all physical key
columns because semantic valid-token
ordinals can span holes, but remain score-bounded by reducing $C$. Parameters,
Q/K/V, output activations, and autodiff tapes are outside this bound, and
multi-chunk autodiff is rejected rather than mislabeled as bounded. Query
chunking can select a different backend GEMM implementation at long lengths, so
bit-for-bit claims remain confined to the unchanged single-chunk oracle path.

Its scale remains unset, selecting the mathematically identical default factor
and preserving fused eligibility only where a future kernel can also reproduce
the required rounding graph. This scaling changes attention entropy without
letting raw projection norms determine temperature.

The unbounded exact-eager graph would still be impractical at 32K: one
full-attention score tensor has shape `[1, 32, 32768, 32768]`, or
34,359,738,368 elements (68,719,476,736 bytes in F16), before the causal mask
and 137,438,953,472-byte FP32 softmax temporary. The query-chunk policy removes
that all-at-once allocation without changing the row-wise attention equation.

The ignored `stress_muse_32k_attention_envelope_on_metal` gate separates
allocation/path evidence from full-checkpoint acceptance. It exhaustively
expands the same schedule consumed by production: 512 global planes of
`[1, 32, 64, 32768]` (exactly $2^{26}$ elements each) and 64 local planes whose
largest shape is `[1, 32, 512, 2559]` (41,926,656 elements). It then actually
executes two 32K-key Metal suffixes. The global suffix crosses the chunk
boundary and reproduces the pinned eager-F16/FP32-softmax discriminator; the
local suffix crosses the 2048-window boundary while slicing absolute key
positions 30,208 through 32,767. The same gate derives all-valid positions
through 32,767, constructs the terminal RoPE table, and writes/reads both a
full KV cache and a mirrored 2048-token ring at the 32K boundary. The fixture
keeps the official 32 query heads, two KV heads, GQA grouping, F16 dtype, and
mask geometry, but deliberately reduces head width to four and omits the 52
dense decoder blocks.

That distinction matters: the synthetic gate proves the per-plane allocation
ceiling, mask/window arithmetic, cache indexing, and real Metal execution. It
does **not** prove that the 55.7 GB language graph plus its linear activations
fits at 32K, execute all 34,359,738,368 global score evaluations for each of 13
NoPE layers, measure full-model throughput, or establish 30B logits parity at
that length. Muse therefore still needs a real-weight 32K run before 32K
checkpoint support can be advertised; this gate cannot upgrade the acceptance
status by itself. Run it on macOS with `cargo test --features metal
stress_muse_32k_attention_envelope_on_metal -- --ignored --exact --nocapture`.

**Implementation:** [`MuseGlimmerAttention`](src/models/muse_glimmer/model.rs).
**Reference:** the [pinned Muse-Glimmer reference](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/muse_glimmer/modular_muse_glimmer.py).

<a id="innovation-attention-sinks"></a>
### Learned attention sinks

Each gpt-oss head learns one sink logit $s_h$ that joins the softmax as an
extra column and is then dropped:
$p_j=e^{z_j}/(e^{s_h}+\sum_k e^{z_k})$. A row can therefore place mass
$p_{sink}=1-\sum_jp_j$ on nothing, instead of being forced onto real keys
(the StreamingLLM attention-sink effect), and a row whose keys are all
excluded yields zero context. The eager path appends the column explicitly;
the fused path keeps one backend call by appending one zero-value sink key per
KV head with selector channels, which is exact up to rounding of $s_h$ by the
score scale. Sinks work in full and ring caches.

**Implementation:** [`attention/dispatch.rs`](src/attention/dispatch.rs);
the derived read-only point `AttentionPoints::sink_probability` exposes
$p_{sink}$ to custom attention. **Reference:** validated against Transformers'
gpt-oss eager attention.

<a id="innovation-projection-bias"></a>
### Attention projection biases

A bias may be attached to any of the Q, K, V, and O projections
independently: Qwen2 biases Q/K/V only, gpt-oss all four.

**Implementation:** `ProjectionBias` in [`attention.rs`](src/attention.rs).

<a id="innovation-differential-attention"></a>
### Differential attention

The DIFF Transformer pairs query heads $i$ and $i+H/2$ and subtracts their
softmax maps, $A_i-\lambda A_{i+H/2}$, to cancel attention noise common to
both, with a learned $\lambda=e^{\lambda_{q1}\cdot\lambda_{k1}}-e^{\lambda_{q2}\cdot\lambda_{k2}}+\lambda_{init}$.
Each pair's output is RMS-normalized and scaled by $1-\lambda_{init}$. The
crate runs it over native KV heads with RoPE or no positions.

**Implementation:** [`attention/differential.rs`](src/attention/differential.rs),
available in custom specs. **Reference:**
[Differential Transformer](https://arxiv.org/abs/2410.05258), validated against
Transformers' DiffLlama; no checkpoint support.

<a id="innovation-unscaled-attention"></a>
### Unscaled dot-product attention

T5 computes $\operatorname{softmax}(QK^\mathsf{T}+B+M)V$ with scale exactly
one, not $1/\sqrt{d_k}$. Its checkpoint-specific projection initialization was
designed around that convention, so inserting the common inverse-root factor
would change logits rather than merely improve stability.

**Implementation:** [`T5Attention`](src/models/t5/model.rs) configures the
shared primitive in [`attention.rs`](src/attention.rs) with `scale = 1.0`.
**Reference:** the [pinned T5 implementation](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/t5/modeling_t5.py).

### Position representations

<a id="innovation-rope"></a>
### Rotary position embedding (RoPE)

RoPE treats pairs of channels as complex coordinates and rotates pair $i$ at
position $p$ by angle $p\omega_i$, where
$\omega_i=\theta^{-2i/d_{rot}}$. Thus
$(R_pq)^\mathsf{T}(R_jk)=q^\mathsf{T}R_{j-p}k$, making the score depend on
relative displacement without an additive embedding. RoPE is applied to Q/K,
never V.

Every RoPE family obtains its configuration from one translator of Hugging
Face `rope_scaling`/`rope_parameters` (including per-layer-type sets such as
Gemma 3's local and global tables), and builds its cosine/sine tables once per
forward pass rather than once per layer.

**Implementation:** validated frequencies in
[`position/rope.rs`](src/position/rope.rs), tensor rotation in
[`position/tensor.rs`](src/position/tensor.rs), and the configuration
translator in [`models/config/rope.rs`](src/models/config/rope.rs).
**Reference:** [RoFormer](https://arxiv.org/abs/2104.09864).

<a id="innovation-rope-scaling"></a>
### RoPE context-extension rules

Each rule rescales the base frequencies $\omega_i=\theta^{-2i/d}$ per complex
pair, following Transformers' `modeling_rope_utils.py`:

- **Linear** divides positions by a factor (Gemma 3's global layers use 8).
- **Dynamic NTK** grows the base $\theta$ once the evaluated context exceeds the
  original one, so the table depends on the longest position in the pass.
- **Llama 3** keeps short wavelengths, divides long wavelengths by `factor`,
  and blends linearly between `original / high_freq_factor` and
  `original / low_freq_factor`.
- **LongRoPE** divides each pair by a searched factor, using the short table
  up to the original context and the long table beyond, and scales the tables
  by an attention factor.
- **Proportional** rotates only a leading fraction of pairs over the whole
  head; the remaining pairs have frequency zero.

**Implementation:** [`position/rope_scaling.rs`](src/position/rope_scaling.rs)
and [`position/frequencies.rs`](src/position/frequencies.rs). Each rule is
checked against inverse frequencies generated by the reference; no checkpoint
column above uses Llama 3, LongRoPE, or proportional scaling. Llama 3 scaling
has also run end to end in the Llama-3.1-8B gate (see the
[status page](ARCHITECTURE_COMPATIBILITY.md#real-checkpoint-evidence)).

<a id="innovation-partial-rope"></a>
### Partial RoPE

Partial RoPE rotates a leading subspace and leaves the remainder content-only:
$q=[R_pq_{0:d_r},q_{d_r:d_h}]$. Qwen3.5/3.8 sets
$d_r=0.25\times256=64$ per full-attention head. This preserves 192 channels
whose coordinates are not directly phase-rotated while providing relative
position information in the other 64.

**Implementation:** rotary-width validation in
[`position/rope.rs`](src/position/rope.rs), prefix application in
[`position/tensor.rs`](src/position/tensor.rs), and Qwen config translation in
[`qwen3_5/model.rs`](src/models/qwen3_5/model.rs).
**Reference:** the exact Qwen configs linked above.

<a id="innovation-decoupled-rope"></a>
### Decoupled NoPE/RoPE head channels

DeepSeek MLA explicitly decomposes each query and key as
$q_h=[q_h^C,q_h^R]$ and $k_h=[k_h^C,k^R]$. Its score is
$(q_h^C)^\mathsf{T}k_h^C+(R_pq_h^R)^\mathsf{T}(R_jk^R)$: a 128-channel
content term plus a 64-channel positional term in Lite. Unlike generic partial
RoPE, this split is part of MLA's factorization and permits the rotary key to be
shared and cached separately from the compressed content latent.

**Implementation:** [`mla.rs`](src/mla.rs).
**Reference:** [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

<a id="innovation-mrope"></a>
### Interleaved multi-axis RoPE (MRoPE)

MRoPE assigns rotary frequency pairs to time, height, and width position axes.
Qwen's 64 rotated channels form 32 complex pairs, interleaved in sections
`[11, 11, 10]`. For text, all three axis IDs are equal, so the construction
collapses exactly to ordinary one-dimensional RoPE. The shared module implements
explicit multi-axis angle construction, but the local Qwen text paths currently
call the ordinary 1-D constructor; that is numerically equivalent for text and
does not imply a vision tower.

**Implementation:** [`mrope_angles`](src/position/tensor.rs) implements the
general helper; [`qwen3_5/model.rs`](src/models/qwen3_5/model.rs) applies its
text-equivalent 1-D specialization.
**Reference:** the [Qwen3.8 configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json).

<a id="innovation-yarn"></a>
### YaRN context scaling

YaRN blends interpolated and extrapolated rotary frequencies. If $\omega_i$ is
the base frequency and $f$ the extension factor, the local implementation forms
$\omega_i'= (1-\gamma_i)\omega_i/f+\gamma_i\omega_i$, where the ramp
$\gamma_i$ is chosen from the number of rotations that dimension completed in
the original context. DeepSeek-V2-Lite uses factor 40 from an original 4,096
positions. Because its `mscale` and `mscale_all_dim` are both 0.707, the rotary-
table amplitude ratio is exactly one; the non-unit correction instead appears
in the attention-logit scale as
$\operatorname{mscale}(40,0.707)^2/\sqrt{192}$. The official Lite release
advertises 32K use; the larger config ceiling is not by itself an acceptance
claim.

gpt-oss uses factor 32 from 4,096 original positions with `truncate: false`,
which keeps the fractional correction range instead of flooring and ceiling
it. As in the reference, the correction range is clamped to $[0, d-1]$, where
$d$ is the full rotary width rather than the pair count.

**Implementation:** [`YarnConfig`](src/position/rope.rs), the correction range
in [`position/frequencies.rs`](src/position/frequencies.rs), and tensor
angles in [`position/tensor.rs`](src/position/tensor.rs). **Reference:**
[YaRN](https://arxiv.org/abs/2309.00071), the pinned DeepSeek config, and the
pinned gpt-oss config.

<a id="innovation-t5-relative-bias"></a>
### Bucketed relative-position bias

T5 maps displacement $r=j-i$ into a bucket. Small distances get exact buckets;
larger distances are logarithmically coarsened and saturate at
`max_distance`. A learned table $E\in\mathbb{R}^{B\times H}$ supplies
$b_{hij}=E[\operatorname{bucket}(j-i),h]$, which is added to raw attention
scores. The encoder divides buckets between past and future. Causal decoder
self-attention clamps future-key displacement to bucket zero, after which the
causal mask excludes those scores. Decoder cross-attention has no relative-
position table. All pinned variants use 32 buckets and maximum distance 128;
only block zero of each stack owns the table, and later blocks reuse its
computed bias. Each pass builds one bucket table over the $O(q+k)$ distinct
displacements and gathers the bias from it, instead of bucketing all $q\times k$
pairs on the host per layer.

**Implementation:** bucketing in [`position/t5.rs`](src/position/t5.rs), learned
lookup in [`position/t5_bias.rs`](src/position/t5_bias.rs), and reuse in
[`t5/model.rs`](src/models/t5/model.rs). **Reference:**
[T5](https://arxiv.org/abs/1910.10683).

### Recurrent and linear-time sequence mixers

<a id="innovation-deltanet"></a>
### Gated DeltaNet

For normalized key/query vectors, each token updates a matrix memory:

$$
S'_t=e^{g_t}S_{t-1},\qquad
e_t=v_t-k_t^\mathsf{T}S'_t,\qquad
S_t=S'_t+k_t(\beta_te_t)^\mathsf{T},\qquad
y_t=q_t^\mathsf{T}S_t/\sqrt{d_k}.
$$

The delta writes the prediction error rather than repeatedly adding a value for
the same key, while $e^{g_t}$ provides learned forgetting. Qwen precedes this
recurrence with a causal depthwise convolution and follows it with gated RMS
normalization and an output projection. Qwen3.8 genuinely uses asymmetric head
counts: 16 key/query heads and 48 value/decay heads.

Chunked prefill lays chunks out densely, inverts each chunk's unit
lower-triangular system by recursive block doubling rather than forward
substitution, and forms the decay differences $G_i-G_j$ by exact subtraction
(not a matrix product, so TF32 cannot perturb $e^{G_i-G_j}$). One-token or
very small prefills fall back to the recurrence, and a snapshot recurrence
records the state after every token so speculative verification can commit any
accepted prefix. Padding is removed from convolution taps and history, and
softplus uses PyTorch's threshold.

**Implementation:** recurrent and triangular chunk forms in
[`delta_net.rs`](src/delta_net.rs), state in
[`cache/tensor/recurrent.rs`](src/cache/tensor/recurrent.rs).
**Reference:** [Gated Delta Networks](https://arxiv.org/abs/2412.06464) and the
pinned Qwen3.5 implementation.
Both public rule paths reject empty query axes, misaligned value axes, and
incorrect decay shapes before tensor math while preserving supplied state.
Their tests include an independent host-scalar evaluation of the update above;
a width-one convolution is also checked as a valid zero-history state.

<a id="innovation-mamba2"></a>
### Mamba2 selective state-space mixer

For head $h$, value channel $p$, and state coordinate $n$, Nemotron updates

$$
S_t[h,p,n]=e^{\Delta_t[h]A[h]}S_{t-1}[h,p,n]
 +\Delta_t[h]x_t[h,p]B_t[h,n],
$$

then reads
$y_t[h,p]=\sum_nS_t[h,p,n]C_t[h,n]+D[h]x_t[h,p]$.
$A=-e^{A_{log}}$ is stable; token-selected $\Delta,B,C$ make the SSM selective.
The Lightning checkpoint uses 64 heads of width 64; the Nano configuration
uses 128. Both use state size 128, eight shared B/C groups, and convolution
width 4. Precision is path-dependent rather than uniformly FP32. Chunked
prefill applies its configured $\Delta$ limits in activation dtype, then
promotes $x,B,C,\Delta$, scan outputs, and state arithmetic to FP32. Recurrent
decode does not apply those limits: decay and state accumulation are FP32, but
$(\Delta B)x$, the state-to-$C$ contraction, the $D$ skip, and output retain the
activation dtype. An explicitly configured F16/BF16 recurrent cache rounds
after every recurrent token; chunked prefill rounds only its terminal state
when publishing to such a reduced-precision cache. Lightning instead declares
`mamba_ssm_cache_dtype = "float32"`. Its authenticated official F16 MPS run
physically retained every `[1, 64, 64, 128]` recurrent tensor in FP32 while its
`[1, 6144, 4]` convolution tails remained F16.

Bias addition and softplus follow the reference $\Delta$ graph, with
PyTorch's thresholded softplus. Its semantic token mask sets $\Delta=0$ (and
replaces, rather than multiplies, padded inputs), so invalid tokens are exact
identity transitions. This intentionally differs from the pinned fallback,
which masks $x,B,C$ but leaves $\Delta$ active. Stateful model entry points
compact valid tokens first; a direct masked mixer call still advances the
convolution through physical columns.

The $\Delta$ clamp is path-specific in the references. At `cfa309…`,
Nemotron-H and Zamba2 clamp the chunk scan to $[\Delta_{min},\infty)$, while
Mamba2, Bamba, GraniteMoeHybrid, and Falcon-H1 clamp it to `time_step_limit`;
the single-step update is unclamped for all of them, and NVIDIA's Nano remote
code clamps both. `TimeStepClamp` states the choice per family.

The mixer is shared and model-independent: Nemotron-H and modular
specifications use the same kernel. The chunk scan processes many chunks per
block with $C\cdot B^\mathsf{T}$ formed once per group (1.40× faster than the
former per-head scan on a 2,048-token Flex workload), ragged rows advance in
one pass, and a snapshot mode returns the state after every token for
speculative verification.

**Implementation:** [`mamba2.rs`](src/mamba2.rs), wired by
[`nemotron_h/model.rs`](src/models/nemotron_h/model.rs) and the modular
[`custom/mixer.rs`](src/models/custom/mixer.rs).
**Reference:** [Transformers are SSMs / Mamba-2](https://arxiv.org/abs/2405.21060),
[pinned Nemotron-H fallback @ `cfa309…`](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/nemotron_h/modeling_nemotron_h.py),
and [its pinned Zamba2 gated normalization @ `cfa309…`](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/zamba2/modeling_zamba2.py).

<a id="innovation-depthwise-convolution"></a>
### Depthwise causal convolution

Before the recurrence, each channel receives a short local filter
$z_{t,c}=\sum_{\ell=0}^{K-1}w_{c,\ell}x_{t-\ell,c}$. “Depthwise” means channels
do not mix inside this convolution; “causal” means no $x_{t+\ell}$ appears.
It gives DeltaNet and Mamba2 an inexpensive local inductive bias and requires
only $K-1$ historical values per channel during decode. Both pinned families use
$K=4$.

**Implementation:** convolution setup and state handling in
[`delta_net.rs`](src/delta_net.rs) and [`mamba2.rs`](src/mamba2.rs).

<a id="innovation-recurrent-decode"></a>
### Fixed-state recurrent decoding

DeltaNet and Mamba2 reduce an arbitrarily long processed prefix to a fixed-size
state plus a short convolution history. A new token advances that state once,
so decode work and state bytes do not grow with prefix length for those layers.
This contrasts with attention, which must retain or revisit an increasing KV
history (or a bounded sliding window).

**Implementation:** reference recurrences in [`delta_net.rs`](src/delta_net.rs)
and [`mamba2.rs`](src/mamba2.rs), with
external storage in [`cache/tensor/recurrent.rs`](src/cache/tensor/recurrent.rs).

<a id="innovation-chunked-prefill"></a>
### Chunk-parallel recurrent prefill

Token-by-token recurrence is poor at exploiting accelerator parallelism over a
long prompt. DeltaNet algebraically resolves the within-chunk updates as a
causal triangular system followed by batched matrix products. Mamba2 constructs
a masked semiseparable transfer matrix within each chunk and carries only the
terminal state between chunks. Both paths are mathematically equivalent to the
transparent recurrence up to floating-point reassociation.

That qualification is operationally important for sparse hybrids. In the
authenticated Lightning F16 oracle, cached recurrent and teacher-forced chunked
execution retained identical top-1 tokens for 32 decisions but reached logit
cosine `0.9215047947571176` at decision 22. Hidden localization at decision 14
first moved from cosine 1.0 to `0.8920338531579629` after Mamba layer 0, then to
`0.4418906779817166` after the following MoE layer. The trace proves that the
path difference begins in Mamba and is amplified across the next sparse block;
it does not prove different expert IDs because the oracle did not capture router
selections. Corresponding-path Burn comparisons therefore keep the 0.999 gate,
while the two official paths remain separate reference streams.

**Implementation:** [`chunked_gated_delta_rule`](src/delta_net.rs) and the
chunked mode in [`mamba2.rs`](src/mamba2.rs).
The ignored real-weight gate binds each Burn stream to its matching official
stream in
[`tests/support/nemotron_h_parity.rs`](tests/support/nemotron_h_parity.rs).
**Acceptance evidence:** [Nemotron-H 3.5 Lightning](NEMOTRON_H_ACCEPTANCE.md).

### Feed-forward and expert routing

<a id="innovation-swiglu"></a>
### SwiGLU / gated-SiLU feed-forward

SwiGLU uses two input projections and a multiplicative content path:
$\operatorname{FFN}(x)=W_d(\operatorname{SiLU}(W_gx)\odot W_ux)$, where
$\operatorname{SiLU}(z)=z\sigma(z)$. It adds a learned gate without recurrent
state. Across the listed Qwen variants, Qwen3.5/3.8 uses dense SwiGLU while
Qwen3-MoE uses it inside routed experts. Muse uses dense SwiGLU; DeepSeek uses
it in dense, routed, and shared branches; FLAN-UL2 selects T5's `gated-silu`
variant.

**Implementation:** [`GatedFeedForward`](src/feed_forward.rs), instantiated by
the relevant family model files. **Reference:**
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).

<a id="innovation-clamped-swiglu"></a>
### Clamped SwiGLU

gpt-oss bounds both GLU branches before combining them: with $g=\min(W_gx,\ell)$
and $u=\operatorname{clamp}(W_ux,-\ell,\ell)$, the unit computes
$(u+1)\,g\,\sigma(\alpha g)$ with $\ell=7$ and $\alpha=1.702$, and the stacked
checkpoint tensor interleaves gate and up columns (gate on even, up on odd).
The clamp keeps rare large activations from dominating the expert output.

**Implementation:** `ClampedSwiGlu` and `GatedFeedForward::forward_clamped` in
[`feed_forward.rs`](src/feed_forward.rs); the de-interleaving loader in
[`gpt_oss/load.rs`](src/models/gpt_oss/load.rs). **Reference:** the pinned
gpt-oss implementation linked above.

<a id="innovation-gated-gelu"></a>
### Gated-GELU feed-forward

This T5 1.1/FLAN variant computes
$W_d(\operatorname{GELU}(W_gx)\odot W_ux)$. GELU smoothly weights a value by
its Gaussian activation probability; the pinned FLAN Small and XXL checkpoints
use the historical `gated-gelu`/`gelu_new` convention, while UL2 changes the gate
activation to SiLU.

**Implementation:** activation selection and gated projections in
[`feed_forward.rs`](src/feed_forward.rs), configuration translation in
[`t5.rs`](src/models/t5.rs). **Reference:** the pinned FLAN configs above.

<a id="innovation-relu-squared"></a>
### ReLU² feed-forward

Nemotron's experts are ordinary two-linear MLPs, not GLUs:
$\operatorname{FFN}(x)=W_d(\max(0,W_ux)^2)$. Squaring increases the separation
between small and large positive activations while retaining exact zeros for
negative inputs. The same activation is used in routed and shared experts.

**Implementation:** [`NemotronHDense`](src/models/nemotron_h/model.rs), with the
activation enum in [`architecture/spec.rs`](src/architecture/spec.rs).
**Reference:** the pinned NVIDIA implementation linked above.

<a id="innovation-sparse-moe"></a>
### Sparse top-k mixture of experts

A router scores $E$ independent FFNs and selects a set $I_t$ of only $K$ for
token $x_t$. The routed result is
$y_t=\sum_{e\in I_t}w_{t,e}\operatorname{Expert}_e(x_t)$. Parameter capacity can
therefore grow with $E$ while per-token expert FLOPs grow with $K$. Exact pairs
are Qwen 8/128, DeepSeek 6/64, and Nemotron 6/128. Qwen and DeepSeek experts are
SwiGLU; Nemotron experts are ReLU².

**Implementation:** shared MoE and router in [`moe.rs`](src/moe.rs) and
[`moe/router.rs`](src/moe/router.rs), used by Qwen3-MoE, DeepSeek, Nemotron-H,
gpt-oss, and custom specs. **Reference:** [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

<a id="innovation-router-policies"></a>
### Shared router policies

Published routers differ along independent axes, each expressed as data in one
shared router rather than as family code: scoring by softmax or sigmoid;
selecting the top $K$ of the scores, or the top $K$ logits and then scoring only
those (gpt-oss: $w=\operatorname{softmax}(z_S)$, no renormalization); greedy or
group-limited selection (DeepSeek); a correction bias used for selection only
(DeepSeek-V3 `noaux_tc`, Nemotron-H); renormalization; a routed scale; and a
learned shared-expert gate. Routing runs in F32, ties resolve to the smallest
expert ID on every backend, and a token with non-finite router scores is an
error rather than a silent NaN. Dispatch evaluates experts in memory-bounded
blocks of whole experts.

**Implementation:** [`moe/router.rs`](src/moe/router.rs) and
[`moe/assignment.rs`](src/moe/assignment.rs) (Verus-verified block planning).
**Reference:** goldens generated from Transformers' Mixtral, Qwen3-MoE,
DeepSeek-V2/V3, and gpt-oss routing.

<a id="innovation-mxfp4"></a>
### MXFP4 expert weights

gpt-oss stores routed-expert projections in the OCP microscaling FP4 format:
each block of 32 values shares one E8M0 scale $2^{s-127}$, and each value is a
4-bit E2M1 code (two per byte, low nibble first). The loader decodes every
value exactly, $\operatorname{fp4}(c)\cdot2^{s-127}$, to the requested dtype
while streaming one tensor at a time, bit for bit equal to Transformers'
`convert_moe_packed_tensors`. There is no packed MXFP4 execution, so decoded
experts occupy full load-dtype memory.

**Implementation:** [`checkpoint/mxfp4.rs`](src/checkpoint/mxfp4.rs), whose
integer core (shape acceptance, offsets, nibble split, and the exact binary64
encoding of each value) is Verus-verified. **Reference:** the pinned
Transformers MXFP4 integration linked above.

<a id="innovation-selected-dispatch"></a>
### Selected-expert-only dispatch

Sparse routing saves compute only if unselected expert matrices are never
evaluated. The portable Burn path groups the compact `(token, top-k slot)`
indices by expert, gathers just those token rows, evaluates that expert, weights
its outputs, and index-adds them back. Stable expert-ID tie breaking makes the
discrete route deterministic across backends while gradients remain on selected
inputs and route weights. The weighting boundary is architecture-selectable:
Qwen3-MoE multiplies each reduced-precision expert output by its normalized FP32
route weight in FP32, rounds that individual contribution to the residual dtype,
and then index-adds it. DeepSeek instead keeps every contribution and the whole
routed sum in FP32 before one final cast. These orders can differ by an F16 ULP,
so they are represented by an explicit `RoutedWeightingPrecision` policy rather
than implicit backend dtype promotion.

**Implementation:** grouping and differentiable dispatch in
[`moe.rs`](src/moe.rs), reused or specialized by all three MoE families.

<a id="innovation-shared-experts"></a>
### Always-active shared experts

A shared branch is evaluated for every token and added to the sparse routed
mixture. It supplies common capacity that need not be redundantly learned by
specialists. DeepSeek serializes its two expert-equivalent widths as one SwiGLU
module of intermediate size $2\times1408$. Nemotron likewise has one shared
ReLU² module of width 3712, exactly twice a routed expert's width 1856; this is
one module even when prose counts two expert-equivalent widths.

**Implementation:** [`DeepseekV2Moe`](src/models/deepseek_v2/model.rs) and
[`NemotronHMoe`](src/models/nemotron_h/model.rs); the generic option lives in
[`SparseMoe`](src/moe.rs).

<a id="innovation-route-renormalization"></a>
### Post-top-k route renormalization

After selecting $I_t$, renormalization replaces each selected score by
$w_{t,e}=s_{t,e}/\sum_{j\in I_t}s_{t,j}$. The selected mixture then has unit
total route mass before any family-wide scale. Qwen3-MoE and Nemotron enable
this. DeepSeek-V2-Lite intentionally leaves its top-six weights as an
unnormalized subset of the global 64-way softmax.

**Implementation:** [`SparseMoe`](src/moe.rs), DeepSeek's exact ordering in
[`deepseek_v2/model.rs`](src/models/deepseek_v2/model.rs), and Nemotron's router
in [`nemotron_h/model.rs`](src/models/nemotron_h/model.rs).

<a id="innovation-sigmoid-routing"></a>
### Correction-biased sigmoid routing

Nemotron first forms independent scores $s=\sigma(W_rx)$. A checkpointed FP32
correction buffer—updated during training from expert-load statistics, not by
backpropagation—is added only when choosing top-k expert IDs. Mixture weights
use the corresponding uncorrected scores, renormalize them, and apply scale 2.5.
This expert-bias mechanism is the aux-loss-free component, although the released
Nano training and Lightning recipe also use a small sequence-level auxiliary
loss. Lightning sets both group counts to one. Its group-pruning stage is
therefore the identity, not a one-class one-hot operation: every expert remains
eligible before the final top-six selection.

DeepSeek-V3-style routing uses the same mechanism (`scoring_func: sigmoid`,
`topk_method: noaux_tc`) and is executable through the DeepSeek family and the
shared router, validated on tiny Transformers goldens only.

**Implementation:** the shared router in [`moe/router.rs`](src/moe/router.rs),
configured by [`NemotronHMoe`](src/models/nemotron_h/model.rs).
**Reference:** [NVIDIA's Lightning recipe](https://docs.nvidia.com/nemotron/nightly/nemotron/lightning35/pretrain.html).

<a id="innovation-router-aux-loss"></a>
### Auxiliary load-balancing objective

Qwen3-MoE's pinned Transformers 5.2 path has a subtle double-softmax convention.
The router computes $p=\operatorname{softmax}(r)$ in FP32, and the framework's
output recorder captures $p$ even though the public tuple calls it
`router_logits`. The causal-LM loss function concatenates every sparse-layer and
token row, applies $q=\operatorname{softmax}(p)$ again, recomputes top-k
indicators $a_{l,t,j,e}$, and evaluates
$L_{bal}=E\sum_{j,e}\operatorname{mean}_{l,t}(a_{l,t,j,e})
\operatorname{mean}_{l,t}(q_{l,t,e})$. The local API preserves genuinely raw
per-layer projections for diagnostics, performs both softmax operations when
the family loss is requested, and returns the single unscaled stack-wide scalar
as `CausalLmOutput::load_balancing_loss`; per-layer diagnostic loss fields stay
empty rather than duplicating a global value.

DeepSeek's pinned `seq_aux=true` instead computes the assignment frequency and
mean probability independently for each sequence, sums their product over
experts, and averages over the batch. Both forms penalize agreement between high
average probability and over-assignment to the same experts; padding is excluded
from every count and mean. The pinned Qwen and DeepSeek configs both declare
coefficient 0.001. The local DeepSeek diagnostic applies `aux_loss_alpha`; the
local Qwen scalar remains raw, so a training caller must apply
`router_aux_loss_coef` before adding it. NVIDIA's Nano report and
Lightning-derived recipe use sequence-level balancing with coefficient 0.0001
alongside correction-biased routing. The local Nemotron path likewise returns
the raw diagnostic; the 0.0001 shown in the matrix is recipe provenance, not a
multiplier silently applied at inference. The Nemotron-H value is Megatron's
normalized sigmoid switch loss, which equals one under perfect balance. gpt-oss
reproduces Transformers' unscaled `aux_loss`. None of these losses changes
inference logits internally.

Cached diagnostics describe only the newly valid tokens, not the complete
retained prefix. Tiny Nemotron regressions independently evaluate its sigmoid
load-balancing formula in host F64 and distinguish new-token loss from a
full-history loss. Qwen-MoE and DeepSeek regressions likewise compare gapped
input masks with compact stateless inputs and require inactive diagnostic rows
to remain zero. These are synthetic contract tests, not additional real-weight
or training acceptance.

**Implementation:** Qwen's pinned stack function in
[`qwen3_moe/model.rs`](src/models/qwen3_moe/model.rs), shared
[`load_balancing_loss`](src/moe.rs), and DeepSeek's sequence variant and
Nemotron's sigmoid-score variant in their respective family
[`model.rs`](src/models/deepseek_v2/model.rs) and
[`model.rs`](src/models/nemotron_h/model.rs). Model-wide and per-layer outputs
are distinguished in [`architecture/types.rs`](src/architecture/types.rs).
**Reference:** the
[Nano technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
and [NVIDIA recipe](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotronh/h100/nemotron_3_nano.py).

### Normalization, embeddings, and output heads

<a id="innovation-rmsnorm"></a>
### RMS normalization

RMSNorm rescales without subtracting a mean:
$y=x/\sqrt{\operatorname{mean}(x^2)+\epsilon}\odot\gamma$. It preserves the
directional offset of the vector and avoids LayerNorm's centering reduction.
The local implementation accumulates the mean square in FP32 for reduced-
precision inputs and casts according to each checkpoint's ordering. Division
by a square root, multiplication by `pow(mean_square, -0.5)`, and
multiplication by a reciprocal square root are algebraically equivalent but
can differ by an F16 ULP. The shared configuration retains these operation
graphs explicitly: Muse ordinary and scale-free norms use the power form,
while its centered block norms and Nemotron-H use the reciprocal-square-root
form. Nemotron additionally rounds the unit-RMS activation before learned
scaling.

**Implementation:** [`normalization.rs`](src/normalization.rs). **Reference:**
[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).

<a id="innovation-zero-centered-rmsnorm"></a>
### Zero-centered RMSNorm parameters

The normalized activation is conventional, but the checkpoint stores a learned
offset $w$ and uses scale $\gamma=1+w$. Initializing $w=0$ therefore gives the
identity scale and keeps learned values centered near zero. Qwen3.5/3.8 uses
this throughout its text graph. Muse uses it for the four norms inside each
block, with FP32 `rsqrt` normalization and scale multiplication before the
final cast, while its final norm stores an ordinary direct scale and uses
`pow(mean_square, -0.5)`.

**Implementation:** `RmsNormConfig::zero_centered` in
[`normalization.rs`](src/normalization.rs), selected in
[`qwen3_5/model.rs`](src/models/qwen3_5/model.rs) and
[`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs).

<a id="innovation-round-before-scale"></a>
### Round-before-scale RMSNorm

Reduced-precision ordering is observable. Qwen3-MoE and DeepSeek compute
$u=x_{f32}/\sqrt{\operatorname{mean}(x_{f32}^2)+\epsilon}$, cast $u$ back to the
residual dtype, and only then multiply by the learned scale. Nemotron-H uses
the same early rounding point but spells the inverse root as FP32 `rsqrt`.
Other families may multiply in FP32 and round once at the end. Matching both
the inverse-root graph and this extra rounding point is necessary for
checkpoint-level logit parity.

**Implementation:** `round_before_scale` in
[`normalization.rs`](src/normalization.rs), configured by
[`qwen3_moe/model.rs`](src/models/qwen3_moe/model.rs) and
[`deepseek_v2/model.rs`](src/models/deepseek_v2/model.rs), and
[`nemotron_h/model.rs`](src/models/nemotron_h/model.rs).

<a id="innovation-sandwich-norm"></a>
### Sandwich normalization

Muse wraps each sublayer with both a pre- and a post-normalizer:
$x'=x+N_{post,a}(A(N_{pre,a}(x)))$ and
$y=x'+N_{post,f}(F(N_{pre,f}(x')))$. The post norms use epsilon $10^{-8}$,
while pre norms use $10^{-5}$. The extra normalization bounds what each branch
injects into the residual stream instead of controlling only its input.

**Implementation:** [`MuseGlimmerDecoderLayer`](src/models/muse_glimmer/model.rs).
**Reference:** the pinned Muse config and reference implementation above.

<a id="innovation-scale-free-embedding"></a>
### Scale-free embedding normalization

Before Muse enters its first block, every token embedding is divided by its
root mean square but receives no learned $\gamma$:
$e'=e/\sqrt{\operatorname{mean}(e^2)+\epsilon}$. This fixes the residual
stream's initial scale without adding a vector of parameters or coupling it to
the embedding table's norm. The pinned implementation spells the inverse root
as `pow(mean_square, -0.5)` to preserve Torch/JAX compiler parity; the same
operation graph is used by Muse's scale-free per-head Q/K norms.

**Implementation:** the shared `scale_free_rms_norm` in
[`normalization.rs`](src/normalization.rs), used by
[`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs).

<a id="innovation-embedding-scale"></a>
### Scaled token embedding

Gemma multiplies the embedding by $\sqrt{d}$ before the first block,
$h_0=E[x]\cdot\operatorname{round}_{dtype}(\sqrt{d})$, with the scalar rounded to
the activation dtype first as the reference does. Tied embeddings with
$N(0,\sigma^2)$ rows otherwise start the residual stream far below unit RMS.

**Implementation:** [`gemma/model.rs`](src/models/gemma/model.rs).
**Reference:** the pinned Gemma 3 implementation.

<a id="innovation-post-norm"></a>
### Post-sublayer normalization and full-width Q/K norm

OLMo 2 normalizes each sublayer's *output* instead of its input:
$x'=x+N(A(x))$, $y=x'+N(F(x'))$, with RMSNorm computed in F32. It also
RMS-normalizes the full query and key projections before splitting them into
heads. Gemma's sandwich normalization combines both placements.

**Implementation:** the Llama-family block in
[`llama/model.rs`](src/models/llama/model.rs) and `QueryKeyNormScope` in
[`attention.rs`](src/attention.rs). **Reference:**
[OLMo 2, Transformers 5.2.0](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/olmo2/modeling_olmo2.py).
Validated on a tiny golden only.

<a id="innovation-shared-stack-embedding"></a>
### Shared encoder/decoder token embedding

T5 uses one vocabulary table $E$ for both source tokens and target-prefix
tokens. Encoder and decoder lookup therefore share lexical coordinates even
when the output LM head is independent. This is distinct from input/output
tying: FLAN Small and XXL share encoder/decoder input embeddings but explicitly
set `tie_word_embeddings=false`.

**Implementation:** the single `shared` embedding in
[`T5ForConditionalGeneration`](src/models/t5/model.rs), with checkpoint aliases
in [`t5/checkpoint.rs`](src/models/t5/checkpoint.rs).

<a id="innovation-tied-lm-head"></a>
### Tied input embedding and LM head

With weight tying, vocabulary logits use the same matrix as token lookup rather
than a separately learned $W_{LM}$. Qwen uses $\ell=hE^\mathsf{T}$; T5's tied
convention uses $\ell=(h/\sqrt{d_{model}})E^\mathsf{T}$. Tying reduces parameters
and constrains input/output token geometry. Qwen3.5-0.8B enables the tie; Qwen3.8
does not. The pinned FLAN-UL2 config omits the field and therefore inherits the
pinned T5 schema default `true`; FLAN Small/XXL explicitly set it false.

**Implementation:** family projection paths in
[`qwen3_5/model.rs`](src/models/qwen3_5/model.rs) and
[`t5/model.rs`](src/models/t5/model.rs), plus alias planning in their
`checkpoint.rs` files. **Reference:** the pinned
[T5 configuration default](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/t5/configuration_t5.py).

<a id="innovation-logit-softcap"></a>
### Final-logit tanh softcap

Muse maps raw vocabulary scores $z$ to
$\ell=T\tanh(mz/T)$ with $T=20$ and
$m=0.19611613513818404$. Near zero this is approximately $mz$; at large
magnitude it smoothly saturates at $\pm20$. The map bounds extreme confidence
while remaining differentiable.

Gemma 2 applies the same map to the final logits with $T=30$ and $m=1$, and
to the attention scores before masking, $s'=c\tanh(s/c)$ with $c=50$
(<a id="innovation-attention-softcap"></a>attention logit soft cap). Gemma 3
configs set both caps to null: its attention never applies a score cap and its
multimodal wrapper never applies the final cap, so the crate refuses non-null
values there rather than computing a different graph.

**Implementation:** [`MuseGlimmerForCausalLm::forward`](src/models/muse_glimmer/model.rs),
the Gemma graph in [`gemma/model.rs`](src/models/gemma/model.rs), and the
attention soft cap in [`attention.rs`](src/attention.rs).
**Reference:** the pinned Muse configuration and Gemma 2 implementation.

<a id="innovation-mtp"></a>
### Multi-token/next-token prediction head

Qwen's released auxiliary graph fuses the already selected next-token embedding
with the target model's preceding hidden state:
$u_t=W_c[N(E[x_{t+1}]);N(h_t)]$, then runs one full-attention Qwen block and a
final norm. The base embedding and LM head are reused to predict $x_{t+2}$; no
private vocabulary matrices exist. Nemotron 3.5 similarly fuses shifted
embedding and base hidden state, but its one prediction **stage** contains an
attention residual block followed by an MoE residual block. Nemotron Nano has no
MTP fields. The ordinary pinned Hugging Face Qwen forward ignores `mtp.*`; the
crate's MTP-aware path is explicit and opt-in.

**Implementation:** [`qwen3_5/model/mtp.rs`](src/models/qwen3_5/model/mtp.rs),
[`NemotronHNextTokenHead`](src/models/nemotron_h/model.rs), and Nemotron's
independent incremental MTP bank in
[`nemotron_h/model/mtp.rs`](src/models/nemotron_h/model/mtp.rs), with exact tensor
contracts in each family's `checkpoint.rs`.

### Stateful inference and generation

<a id="innovation-kv-cache"></a>
### Preallocated projected-KV cache

Autoregressive attention reuses historical projections. At step $t$, only
$q_t,k_t,v_t$ are new; cached $K_{<t},V_{<t}$ are concatenated logically before
computing the row of attention scores. Preallocated storage makes append cost
independent of allocator behavior and keeps cache state external to model
parameters. GQA stores only native KV heads. DeepSeek uses its smaller latent
cache instead of projected K/V; recurrent layers use their own state form.

**Implementation:** [`cache/tensor/kv.rs`](src/cache/tensor/kv.rs), coordinated
by [`cache/tensor/state.rs`](src/cache/tensor/state.rs) and each family's
`model/decode.rs`.

<a id="innovation-batched-decoding"></a>
### Batched cached decoding

Every family advances all rows of a ragged batch in one model pass per step:
each row's new tokens are written into its own cache slots (written ahead
into full storage, so attention reads one zero-copy view of the history),
rows with different lengths are reconciled by exclusion masks and cache
segments, finished rows are no-op padding, and all layers' writes are
published atomically through one step write set, so a failed step changes no
readable state. Earlier versions looped over rows one at a time. Equivalence is
tested against the stateless forward and against the former row-by-row path on
tiny models (left and right padding, fully masked rows, ring wraparound,
rollback, reorder, and truncation). On Flex CPU with eight rows the batched
path decoded 1.7× to 3.6× faster for dense and hybrid families and 1.3× to
2.7× for sparse-expert families; gpt-oss's wide experts are not faster on CPU.

**Implementation:** the step API in
[`cache/tensor/state/step.rs`](src/cache/tensor/state/step.rs), batched KV
I/O in [`cache/tensor/kv/batch.rs`](src/cache/tensor/kv/batch.rs), and each
family's `model/decode.rs`.

<a id="innovation-ring-cache"></a>
### Sliding ring KV cache

A windowed layer maps absolute position $p$ to logical slot $p\bmod W$ and
retains only the newest $W$ entries. The local mirrored layout allocates $2W$,
writes each logical value at both $p\bmod W$ and $p\bmod W+W$, then exposes one
contiguous chronological $W$-wide slice across the wrap point. If a speculative
proposal could overwrite a still-committed slot, proposal tensors must be
staged until acceptance; changing only the logical length cannot restore
destroyed K/V values.

Each sliding cache group has its own window, so Gemma's 1,024-token local
layers and gpt-oss's 128-token layers keep rings while global layers keep full
storage. A ring can roll back only within its window; speculative proposals in
a ring are staged.

**Implementation:** safe plans in [`cache/ring.rs`](src/cache/ring.rs), tensor
storage in [`cache/tensor/kv.rs`](src/cache/tensor/kv.rs), and the Muse,
Llama-family, Gemma, and gpt-oss decode states.

<a id="innovation-cross-kv-cache"></a>
### Immutable cross-attention KV cache

Encoder states are fixed during target generation, so each decoder layer can
compute $K_e=H_eW_K$ and $V_e=H_eW_V$ once. Every subsequent step projects only
its target query. Unlike self-KV, these tensors never append. Beam search
keeps every reorder inside a source's group of beam rows, whose encoder output
and cross-attention K/V are identical copies, so T5 leaves both in place
instead of gathering them on every step.

**Implementation:** [`T5CrossAttentionCache`](src/models/t5/model/decode.rs).
T5 validates caller-supplied encoder dtype before these matrix products, using
the actual model parameters rather than the backend's default floating type.
F16/F64 matrix parameters with F32 normalization scales are valid on the tested
Flex path. An incompatible encoder dtype returns a contextual error without
mutating cache state; the runtime does not silently convert it. After prefill,
the retained encoder and cross-KV remain authoritative, including after beam
reordering. Later encoder arguments are checked for structural compatibility
(including dtype), not full tensor-value identity.
The [dtype-guard replay](acceptance/measurements/t5-encoder-dtype-refresh-20260910.json)
strictly loads the pinned FLAN-T5-small checkpoint on Flex/F32 and Metal/F32;
both pass all seven official vector comparisons at `atol=rtol=1e-4` plus the
exact cached/teacher-forced 32-token and four-beam decisions. This is numerical
correctness evidence, not bit-identical-logit or performance evidence.

<a id="innovation-recurrent-cache"></a>
### External recurrent-state cache

The cache for a recurrent mixer contains its sufficient state rather than one
entry per historical token. Qwen stores each DeltaNet matrix memory plus the
short convolution tail. Nemotron stores each Mamba2 SSM tensor plus its
convolution tail. Keeping this state outside Burn parameters makes reset,
reorder, truncate, and failure-atomic publication explicit and keeps stateless
full-precision graphs autodiff-safe.

For the exact Lightning base oracle at physical batch one, the 23 Mamba slots
hold F16 convolution tensors `[1, 6144, 4]` and FP32 SSM tensors
`[1, 64, 64, 128]`. Six attention slots at layers 5, 12, 19, 26, 33, and 42
hold F16 K/V tensors `[1, 2, L, 128]`; 23 MoE schedule positions have no cache
payload. The aggregate `DynamicCache.batch_size` value is `-1` because it is an
API sentinel, not because physical batch geometry is unknown.

**Implementation:** [`cache/tensor/recurrent.rs`](src/cache/tensor/recurrent.rs)
and [`cache/tensor/state.rs`](src/cache/tensor/state.rs), consumed by Qwen and
Nemotron decode modules.
The snapshot API validates matrix and convolution shapes and dtypes before
publishing either tensor. Regression tests now verify that per-row and batched
rejections preserve both an existing pending proposal and committed state,
including their independent F32/F16 storage, and that retry, ragged prefix
commit, and a subsequent rollback remain usable.

<a id="innovation-transactional-cache"></a>
### Transactional speculative cache

Speculation tentatively advances both the target and draft states. A transaction
records a mark, stages appends, and then either `commit_prefix(a)` to publish
only the accepted prefix or `rollback()` to restore the committed logical state.
DeepSeek-V2's decode state implements this as the `cache::TransactionalCache`
trait, with one uniform accepted prefix. Qwen3.5 and Nemotron-H expose inherent
`mark`, `commit_prefixes(mark, target, mtp)`, and `rollback` instead, because
their MTP bank commits a prefix of its own; generic code bounded on the trait
reaches only DeepSeek-V2. The other families' decode states have no
transactions and roll back with `truncate`.
Rollback leaves committed cache tensors unchanged and discards staging metadata;
rejected staging bytes may remain allocated but are no longer addressable. Qwen
maintains an independent MTP KV cache alongside target GQA/DeltaNet state;
Nemotron does the same alongside target GQA/Mamba2 state. Both require the
target cursor to remain one token ahead of the shifted MTP cursor at stable
cycle boundaries. Per-token recurrent snapshots make partial Mamba2/DeltaNet
commits exact even though their terminal recurrences cannot be inverted.
Architecture-level truncation derives the surviving shifted prefix and
prevalidates both banks before either cursor or physical tensor changes; this
prevents reset/rewind operations from leaving stale auxiliary KV reachable.

**Implementation:** lifecycle rules in
[`cache/transaction.rs`](src/cache/transaction.rs), tensor snapshots in
[`cache/tensor/state.rs`](src/cache/tensor/state.rs), Qwen integration in
[`qwen3_5/model/decode.rs`](src/models/qwen3_5/model/decode.rs), and Nemotron
integration in [`nemotron_h/model/decode.rs`](src/models/nemotron_h/model/decode.rs).

<a id="innovation-speculative-decoding"></a>
### Lossless speculative decoding

An MTP distribution $q_i$ proposes tokens that the target distribution $p_i$
then verifies. Greedy mode accepts exactly when proposal and target argmax
match. Stochastic mode accepts proposed token $y_i$ with
$a_i=\min(1,p_i(y_i)/q_i(y_i))$; on rejection it samples from normalized
$(p_i-q_i)_+$. This rejection rule preserves the target model's marginal
distribution exactly. If all drafts pass, the target emits one bonus token.
The current Qwen and Nemotron coordinators are intentionally batch-one and
default to two drafts. Both now verify all drafts in one target pass and
rebuild the MTP history from target hidden states, as vLLM does. The shared
rule (`generation::speculative`) also provides deterministic verification and
prompt-lookup drafting for models without an MTP head. Nemotron follows vLLM's shifted pair convention:
auxiliary cache position `i` consumes the next-token embedding `E[x_(i+1)]`
with the preceding target or auxiliary hidden state. Explicit positions must be
contiguous because vLLM delegates those KV slots to its scheduler, whereas the
local cache owns them directly.
Both APIs execute one speculative cycle; their caller still owns the global
token budget, stop-sequence matching, and padding policy, matching the split
between vLLM's model-side speculator and scheduler. Greedy verification keeps
the full vocabulary on the backend and compares only selected argmax IDs;
stochastic rejection still materializes normalized distributions until a
backend-native verification kernel is justified.

**Implementation:** [`generation/speculative.rs`](src/generation/speculative.rs)
(the acceptance rule, with Verus-verified commit arithmetic),
[`qwen3_5/model/speculative.rs`](src/models/qwen3_5/model/speculative.rs),
and [`nemotron_h/model/speculative.rs`](src/models/nemotron_h/model/speculative.rs).
**Reference:** [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
and the [pinned vLLM autoregressive speculator](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py).

<a id="innovation-beam-search"></a>
### Cache-aware beam search

Beam search retains the $B$ highest-scoring target prefixes. Each expansion adds
token log probability to cumulative score, then prunes globally within a source
row; finalized hypotheses can be ranked with a length penalty. Stateful search
must reorder decoder self-KV by the parent-beam indices, otherwise scores and
hidden histories silently diverge. Because a reorder never leaves a source's
group of rows (a Verus-verified property), the encoder output is expanded once
and never re-gathered. Candidates come from a bounded heap rather than a full
sort, log probabilities are computed in at least F32, the cache needs capacity
`prompt + max_new_tokens - 1`, and `EarlyStopping` mirrors Transformers'
`true`, `false` (heuristic, the default), and `"never"` settings.

**Implementation:** the generic coordinator in
[`generation/seq2seq.rs`](src/generation/seq2seq.rs) and T5 defaults in
[`t5/generation.rs`](src/models/t5/generation.rs).
A valid restricted vocabulary can leave fewer live hypotheses than beam slots.
Unused rows are PAD-masked no-ops: their cache lengths do not advance. If fewer
distinct terminal hypotheses are reachable than `num_return_sequences`, the
coordinator reports an error rather than duplicating a result. Deterministic
tests cover both outcomes and the exact reorder/token/mask history.

## Reference/runtime distinctions discovered during review

- **Architecture is not acceptance.** The matrix records graph features even
  where the crate currently has only schema, synthetic, or checkpoint-contract
  evidence. Consult the status page linked at the top before claiming
  support for a checkpoint.
- **Nemotron is NoPE.** The exact Nano and 3.5 attention implementations do not
  apply RoPE. Serialized RoPE-shaped fields are compatibility metadata, not an
  executed position transform.
- **Nemotron 3.5 has a pinned index-accounting defect.** At revision `a9904d…`,
  the fourteen SafeTensors headers contain 65,826,538,624 physical tensor bytes,
  while `metadata.total_size` declares 65,842,365,568. The 15,826,944-byte
  difference is exactly 23 BF16 base-router matrices of shape `[128, 2688]`;
  the equally shaped MTP router is not part of the difference. Generic manifest
  discovery therefore remains strict. Only the Nemotron checkpoint contract can
  reconcile this total, and only after matching the exact repository, revision,
  raw config/index SHA-256 identities, all 23 base-router names and descriptors,
  and the separately required MTP router. See
  [`checkpoint/accounting.rs`](src/checkpoint/accounting.rs) and
  [`nemotron_h/checkpoint.rs`](src/models/nemotron_h/checkpoint.rs).
- **Nemotron's real-weight smoke is not oracle parity.** The authenticated
  Lightning artifact now strictly applies all 6,513 tensors to the Metal F16
  graph in fourteen-shard physical order. Its six-token base smoke proves
  finite stateless, cached-prefill, and recurrent-decode execution, exact
  cache topology, bit-identical stateless/prefill hidden traces, and
  recurrent/teacher top-1 agreement. It also exposed the single-group router
  edge above. That recorded real-weight smoke installed but did not execute the
  270 MTP tensors. Subsequent tiny Flex tests execute the pinned one-stage graph,
  shifted independent KV cache, and transactional speculative coordinator, but
  they do not establish official real-weight parity. A separate ignored gate now
  compiles and is ready to compare all 32 Burn stateless and cached decisions
  with their corresponding official rows, but that 30B Metal gate has not been
  executed. No official-reference, 32-token, quantized, or 32K pass follows from
  implementation or compile-only validation. See
  [`NEMOTRON_H_ACCEPTANCE.md`](NEMOTRON_H_ACCEPTANCE.md).
- **Nemotron's two official Mamba paths are distinct reference streams.** The
  pinned base oracle records the recurrent cached row and teacher-forced
  chunk-scan row for all 32 decisions. Their top-1 IDs are identical, but their
  minimum cosine is `0.9215047947571176` at decision 22. Layer traces show a
  Mamba-originated difference followed by larger MoE-boundary divergence. This
  source self-consistency observation does not relax the 0.999 Burn-versus-
  official gate on each corresponding path and does not establish different
  expert selection. Typed oracle rows make the ignored gate reject a crossed
  path before computing cosine; ordered summaries also prevent a missing or
  duplicated decision from masquerading as complete coverage. Stateless prompt,
  teacher-forced, cached-prefill, cached-decode, and hidden-trace checks remain
  separately labeled.
- **Nemotron's cache batch property is a sentinel, not physical geometry.** The
  authenticated `DynamicCache.batch_size` reports `-1` even though every
  concrete Mamba, convolution, and attention cache tensor has leading extent
  one. Cache validation must inspect physical layer state while preserving the
  API-reported sentinel as a separate fact.
- **A normalized T5 flag is not proof of physical weight tying.** FLAN-UL2's
  pinned config omits `tie_word_embeddings`; the pinned upstream T5 schema and
  the local Serde default both resolve that omission to `true`. FLAN Small and
  XXL explicitly serialize `false`. Transformers 5.2 normalizes the loaded
  FLAN-Small runtime flag back to `true`, yet the loaded `lm_head` remains
  physically independent and the decoder output is not divided by
  $\sqrt{d_{model}}$. The Burn contract therefore follows the serialized
  checkpoint, tensor inventory, storage identity, and executed scaling path
  together rather than trusting that post-load Boolean in isolation.
- **Burn 0.21 Flex Boolean gathers require dtype normalization.** Importing a
  dynamic host `Vec<bool>` or selecting rows from a Boolean tensor can leave a
  numeric `U8` dtype tag even though the Rust tensor kind is Boolean; a later
  `bool_not` then rejects it. Host masks and beam/cache row gathers therefore
  pass through integer indicators and a nonzero comparison, which asks the
  active backend to construct its own logical representation. The regression
  is covered in [`architecture/types.rs`](src/architecture/types.rs), and the
  cache-aware FLAN gate exercises both tokenization and repeated beam reorders.
- **Qwen's two gates are different.** Full attention uses a sigmoid output
  gate. The Qwen3.8 `"swish"` field describes DeltaNet's output gated norm. The
  pinned reference hard-codes SiLU for that norm, so the local configuration
  accepts only the equivalent `silu`/`swish` spellings instead of pretending a
  serialized sigmoid variant is implemented.
- **Qwen runtime-affecting compatibility fields fail closed.** The released
  Qwen3.5/3.8 recurrent association state is FP32, the inherited
  `mlp_only_layers` option is removed by the pinned reference, and no released
  dedicated-MTP-embedding tensor contract exists. Non-FP32 state requests,
  non-empty MLP-only schedules, and dedicated MTP embeddings are rejected
  during config validation. Stateless explicit positions now share the cached
  path's declared-context bound and derive scaled-RoPE context from their actual
  maximum rather than physical sequence width.
- **Qwen3.5-0.8B official parity now has a reproducible evidence path.** The
  earlier Metal result's full-vocabulary vector was not retained, so it remains
  historical evidence rather than a releasable oracle. A local-only,
  create-only producer now authenticates the exact checkpoint, Transformers
  5.2 package files and executed Qwen sources, then writes full prompt, 32-row
  cached, and 32-row teacher-forced F32 payloads. Its Rust consumer fails closed
  unless all four external files match independent digest anchors and requires
  every full-vector cosine plus all 32 top-1 decisions. The
  small committed JSON fixture cannot silently substitute for that bundle.
  The Burn replay also authenticates all six target checkpoint artifacts before
  load and after inference; matching headers alone do not prove checkpoint
  identity. Failed producer runs print separate historical-token, cached versus
  teacher-token, full-vector cosine, and top-two-margin diagnostics before
  rejecting the run, without publishing an accepted bundle or relaxing gates.
  The completed official-only prompt selection and producer run now retain a
  stable 23-token non-thinking chat oracle: all 32 margins exceed two F16 ULPs
  (minimum six). The reviewed four-file bundle passes Burn Metal with minimum
  official/Burn full-vector cosine `0.999996242` and all 32 cached/stateless
  decisions identical. The old `Hello` near-tie remains a sampled-logit
  diagnostic. This closes the retained-reference gap, not the remaining API,
  coverage, or standardized-performance gates.
- **Mixed-precision generation uses runtime storage types.** Backend defaults
  do not determine the dtype of loaded model outputs. Real F16 Qwen API tests
  exposed selected-log-probability readback and sampling-threshold type errors.
  Readback now dispatches on storage dtype, penalties match logits, and half
  probability/CDF sums use F32 while preserving native F64 and the seeded RNG
  stream. The public Auto-load/chat/streaming path now matches all 32 official
  tokens. Real left/right-padded unequal-length batches match independent
  unpadded full/cached execution and chunked public generation; padding neither
  consumes recurrent history nor shifts logical positions. These tests do not
  change the architecture matrix or relax oracle tolerances.
- **Qwen3.5-0.8B Q4 execution and its 32K envelope are separate claims.** The
  reviewed group-64 Q4 sidecar is format v2, binds the exact source checkpoint
  and F16 load plan, contains 186 packed matrices, and has complete-file SHA-256
  `cca925f24ab3b61a76fbbd5aefb8e0f2052da1b4636de9e996b890859cae5fd5`.
  Its one-token stateless/cached Metal smoke reached cosine `1.0`. A distinct
  ignored gate authenticates that sidecar, allocates the six full-attention KV
  layers at capacity 32,768, checks eighteen DeltaNet matrices remain FP32 while
  convolution history remains F16, and places one stateless token at position
  32,767. That envelope now passes on Metal, but cannot by itself establish
  populated-history execution or promote Q4 beyond experimental status.
  The separate `qwen35_q4_v2_populates_and_decodes_the_32k_history_on_metal`
  acceptance test fills 32,767 positions through bounded 256-token prefill
  calls, then writes the last position through recurrent decode. It checks
  finite chunk logits, populated KV positions in all six global layers, every
  terminal DeltaNet matrix/convolution value, overflow failure atomicity, and
  reset-versus-stateless equivalence. Input and output hashes, per-chunk times,
  and analytical fixed/per-token cache bytes are emitted. The
  [completed Metal stress record](acceptance/measurements/qwen3.5-0.8b-q4-32k-stress-20260909.json)
  reports all 32,768 positions committed, finite state throughout, atomic
  overflow rejection, and exact reset-versus-stateless logits. KV storage uses
  12,288 bytes per token plus 19,537,920 fixed recurrent bytes: 422,191,104
  bytes at 32K. Neither this stress input nor the envelope is an
  official-reference or quantized-NLL quality corpus. One-shot synchronized
  timings and whole-process memory high-water marks are not warm benchmarks.
- **Qwen3-MoE has two equivalent expert storage layouts.** Its official
  checkpoint exposes independent gate/up/down tensors for every expert. The
  pinned Transformers 5.2 loader keeps the `qwen3_moe` architecture but aliases
  its checkpoint conversion to the `qwen2_moe` rules, stacking those split
  tensors into packed three-dimensional gate-up and down weights: 18,867
  physical checkpoint tensors become 531 named parameter tensors. The Burn
  full-precision graph retains independently addressable expert modules.
  Separately, `SparseMoe::dispatch_selected_experts` groups assignments and
  skips experts with no assigned tokens. This storage difference does not
  change the routed SwiGLU equation. See the pinned
  [`conversion_mapping.py`](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/conversion_mapping.py)
  and
  [`modeling_qwen3_moe.py`](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py),
  plus [`qwen3_moe/load.rs`](src/models/qwen3_moe/load.rs) and
  [`moe.rs`](src/moe.rs).
- **Muse's serialized Transformers version is provenance, not executable
  identity.** The checkpoint config records `5.15.0.dev0`, while the immutable
  `cfa309…` source revision used for numerical acceptance identifies itself as
  `5.17.0.dev0`. Oracle generation therefore authenticates the Git revision and
  source bytes and records the config value separately; it never substitutes
  one version string for the other.
- **Muse's short-context numerical gate is not its 32K gate.** The authenticated
  seven-file oracle retains every cached and teacher-forced vocabulary row,
  cached-prefill and first recurrent-decode hidden traces, and exact cache
  lengths/positions. The real F16 graph passed those comparisons and all 32
  greedy decisions. The bounded query-chunk policy removes the former
  all-at-once score allocation, but a complete 32K real-checkpoint stress run
  remains required before advertising long-context support.
- **Muse's pinned `DynamicCache.batch_size` is a sentinel, not physical
  geometry.** Its concrete dynamic KV layers do not publish the optional
  `batch_size` attribute, so the aggregate property returns `-1` even after
  initialization. Oracle validation establishes batch one from every key and
  value tensor's leading axis and records both that physical fact and the API
  sentinel; treating `-1` as a real batch size would incorrectly reject the
  official cached graph.
- **Muse's native-quantization boundary is projection-only.** The pinned
  627-tensor F16 plan contains 417 eligible matrices: five attention and three
  SwiGLU projections in each of 52 layers, plus the independent LM head. The
  embedding and 209 normalization vectors remain F16; the pinned configuration
  has no attention biases. Every eligible input width is divisible by the G64
  group size. Nonzero two-layer fixtures prove that automatic Q8 and Q4 loading
  keeps all eligible parameters on packed-aware linear paths through stateless,
  cached-prefill, and recurrent-decode execution. A deliberately row-unaligned
  Burnpack is rejected as reference-only. The real Q8/G64 streaming conversion
  bound both source shards and the exact F16 plan before and after execution,
  revalidated all 417 payloads, and produced a `28,165,645,337`-byte v2 sidecar
  with SHA-256
  `91cfd35bfe0a389cbc2eed2189c4e33312bdce1a0ff50d8452acf06d00f0e8eb`.
  A separate traced Metal gate reauthenticated the sidecar and checkpoint
  provenance, ran stateless, cached-prefill, and recurrent decode, and observed
  exactly 417 packed Q8 matmuls with no whole-weight dequantization. Its
  cached/stateless logit cosine was `1.0`. The real Q4/G64 streaming conversion
  has independently bound the same sources and F16 plan, reopened all 417
  payloads, and published a `14,911,372,485`-byte v2 sidecar with SHA-256
  `05b69f67799b4dc0228b4298e39d466d753fcc5d6484f61c8920c551017daed3`.
  A separate Q4 trace also observed all 417 packed matmuls without whole-weight
  dequantization, with cached/stateless logit cosine `1.0`. Separate ignored Q8
  and Q4 quality gates now teacher-force the exact committed 32-token sequence
  through packed recurrent decode, align every row and absolute position with
  the authenticated official cached F16 fixture, and apply stable F64
  full-vocabulary NLL budgets of 1% and 5%, respectively. They also report
  official/quantized top-one agreement and minimum full-vector cosine as
  diagnostics. Neither quality gate has run; quantized quality and long-context
  execution remain pending, so this is not a support claim.
- **Full-precision training safety includes parameter initialization.** Burn
  0.21's column-layout initializer can transpose an already tracked tensor into
  a non-leaf. Shared lazy initialization now establishes the leaf after this
  layout transform; checkpoint installation similarly postpones gradient policy
  until after mapping. No forward computation is detached. Tests check row/column
  values, frozen parameters, checkpoint loading, and finite nonzero gradients
  through all projections, norms, recurrent scalars, and convolutions in tied
  and untied tiny 3:1 Qwen hybrid graphs. This established the initialization
  prerequisite; the subsequent [v0.3 training workflows](TRAINING.md) use native
  Burn optimizers and Learner integration. A separate trainer framework remains
  outside this crate's scope.
- **Release coverage is measured separately from feature presence.** The first
  Flex/tokenizer/template/autodiff unit-suite baseline passes 764 tests but
  reaches only 87.7084% LLVM line coverage crate-wide and 90.5207% shared/core,
  below the required 90%/95%. Inline test bodies and macro expansions are still
  counted; these are not production-only or branch-coverage claims. The
  [executable release gates](RELEASE_GATES.md) preserve experimental source in
  the denominator and track the remaining acceptance requirements. Fuzz review
  also found a modular seq2seq cross-attention width multiplication overflow;
  checked arithmetic now rejects the wrap-to-valid-width case in debug and
  optimized builds.
  Modular MLA's content-plus-rotary head width now also uses checked addition;
  malformed serialized dimensions return a contextual configuration error
  before dispatch. Targeted normalization, position, recurrent, MoE, tokenizer,
  and template boundary tests supplement the baseline without changing gates.
  The later integrated `df3bea5` snapshot runs 814 unit tests plus ordinary
  integrations and improves measured line coverage to 88.7276% crate-wide and
  92.4763% shared/core. Both gates still fail and remain release requirements.
  Five independent bounded sanitizer fuzz targets then covered configuration,
  artifacts, masks/positions, cache transactions, and generation; each completed
  10,000 inputs without a crash. The fuzz package now has twelve targets, adding
  experiment interfaces, logits processing with token history, chat templates,
  the RoPE translator, generated SafeTensors and MXFP4 headers with an
  official-parser differential, attention connectivity with masked attention,
  and modular specs with cache parity ([campaign record](fuzz/CAMPAIGNS.md)).
  Finite campaigns are not exhaustive coverage.
- **Performance evidence is separate from architecture correctness.** FLAN's
  first [repeated Metal/F32 series](acceptance/measurements/flan-t5-small-metal-benchmark-20260909-run.json)
  validates official vectors before three warmups and ten measured runs per
  32/128/512-token source. Each decode sample commits 128 forced tokens so EOS
  cannot shorten the work denominator. Model-only and greedy-readback-inclusive
  timing, fixed cross-attention versus per-token self-attention cache bytes,
  and output hashes are separate metrics. Whole-process high-water memory is
  reported without claiming per-phase or discrete host/device measurements.
  The matching [Qwen F16 baseline](acceptance/measurements/qwen3.5-0.8b-metal-benchmark-20260909-run.json)
  also passes repeated 32/128/512-token source measurements and all 32 official
  decisions. Its observed hybrid-cache storage matches the separate KV-per-token
  and fixed DeltaNet formulas. Compiled longer-prefill timings remain unexecuted;
  they are not inferred from the already-passing Q4 32K stress.
- **Generation honors runtime float storage.** Selected F16/BF16/F32/F64 and
  Flex32 log probabilities are read by their actual dtype, preserving F64
  precision without transferring a new full-vocabulary buffer. Half-precision
  probability accumulation requests F32. Burn 0.21's Flex32 sorting fallback
  remains unsupported: active top-k/top-p filters fail with a capability error,
  while native Metal Flex32 greedy/unfiltered sampling has an executed test.
  Both small-model public examples also passed [local Rust 1.92 execution](acceptance/measurements/public-examples-20260909.json).
  The later [integrated replay](acceptance/measurements/small-model-functional-replay-6226170-20260909.json)
  reconfirms both families' official vectors, exact decisions, public APIs and
  examples after the latest storage, attention, and beam-score validation fixes.
- **Strict artifact identity includes JSON key uniqueness.** Burnpack tensor
  inventories and checkpoint-shard provenance now reuse the same duplicate-key
  Serde visitor as SafeTensors metadata/index maps. Identical, conflicting, and
  escape-equivalent duplicate names fail instead of silently selecting the last
  value. Payload hashes do not replace this structural validation.
  The [post-fix real-artifact replay](acceptance/measurements/small-model-post-json-validation-replay-20260910.json)
  passes Qwen F16/Q4 and FLAN F32; all seven FLAN full-vector hashes remain
  unchanged. This is loader/correctness evidence, not quantized corpus quality.
- **Checkpoint layout survives parameter initialization.** Burn's lazy shape
  becomes the runtime shape after initialization. Repeated full-precision
  installation now validates in saved/checkpoint layout, preserving column
  transposes, autodiff leaf policy, and the original value on failure. Reloading
  an initialized QFloat parameter fails before its save mapper can allocate a
  whole-weight dequantization temporary; initial lazy packed loading and
  quantized inference are unaffected. Burnpack tests also verify retryable
  staging after conversion, provenance, and late publication failures.
  A [post-fix Metal replay](acceptance/measurements/small-model-parameter-reload-replay-20260910.json)
  passes Qwen F16/Q4 and FLAN, retaining exact output hashes and the separate
  failed sandbox attempt. Initialized-reload behavior is exercised by synthetic
  regression tests, not inferred from these first-load checkpoint runs.
- **Coverage evidence is bound to observed inputs.** Pre/post source and fixture
  hashes now detect ordinary concurrent edits, including changes to Python
  reference scripts embedded by Rust tests. The runner retains raw report/log
  hashes and selected tool versions, but does not treat these as compiler
  dependency closure or permission to exclude test-only syntax candidates.
  A fresh `9ff597f` run passes the unchanged legacy 90%/95% thresholds
  (90.1704% crate-wide, 95.0594% shared/core). Those counts still include inline
  tests and expansions; production-only qualification remains open.
  Initial test-module separation preserves production dependencies and test
  identities, and native compiler-artifact capture now has isolated Cargo/rustc
  fixtures. These development tools do not yet authorize coverage exclusions.
  The [opt-in runner](scripts/COVERAGE_EVIDENCE.md) now binds fresh production
  and test artifacts to compiler-reported inputs and the exact native LLVM
  export, without upstream filename filters. Native fixture success is not a
  full-crate production-only qualification. Parallel work also exposed Cargo
  reusing another worktree's test binary in a shared target directory; distinct
  workspace wrappers and explicit test/dep-info identities protect ordinary
  test runs. Clippy overrides that wrapper setting and needs separate validation.
  The first [full-crate native-bound run](acceptance/measurements/native-coverage-binding-20260910.json)
  at `57d0560` passes unchanged 90%/95% aggregate thresholds (90.2564% crate-wide,
  95.1797% shared/core), with all 16 compiler captures and the exact 15 LLVM test
  objects authenticated. No test or generated-code exclusions are applied; this
  does not establish production-only coverage. Isolated direct-driver Clippy and
  source/MSRV/accelerator-compile/package checks also pass at that snapshot.
  A separate [whole-test-file policy](scripts/WHOLE_FILE_COVERAGE.md) now requires
  classifier ownership, exact unit/production compiler input sets, source hashes,
  and the actual successful coverage-command log to agree before removing a file.
  Macro/include/alias references from production veto removal. Mechanically
  verified shared and Qwen/FLAN test-module extraction makes this distinction
  possible without changing model execution. Remaining inline tests are retained;
  this partial diagnostic is not yet a production-only gate.
  The later [b4d6410 capture](acceptance/measurements/v0_1-source-gates-20260910.json)
  independently binds every retained native region to authored production
  function items and passes 90.005% crate-wide / 95.011% shared/core coverage.
  It excludes 79 verified whole test files and no Burn-generated spans. This
  exact-source qualification supersedes the earlier numerical gaps, not the
  conservative generic-tool flags; it is not a general macro-closure or branch
  coverage claim.
- **Public generation overhead has its own benchmark protocol.** Qwen's
  existing model-API loop remains unchanged; a separate ignored benchmark now
  times actual `generate_causal` calls through automatic model dispatch, including
  cache allocation, selection, log-probability readback, and stopping. It validates
  all 32 oracle tokens in each of three warmups and ten measured repetitions.
  The [hardware correctness run](acceptance/measurements/small-model-rc-refresh-20260910.json)
  now passes all thirteen complete calls with exact oracle tokens and finite
  log probabilities. Its timings are explicitly rejected: the operator metadata
  incorrectly declared AC power while the host probe reported battery power.
  A sandboxed probe reported AC; differing probe contexts do not establish a
  physical power transition.
  A [replacement host-AC-endpoint run](acceptance/measurements/small-model-performance-20260910/manifest.json)
  now validates all thirteen calls and measures a 1.751402479 s median for 32
  tokens. Continuous power monitoring was not performed. This is a separate
  public-coordinator timing boundary, not a replacement for the model-loop baseline.
- **Small-model release checks distinguish correctness from performance.**
  The same refresh passes both Qwen public-API tests, all seven FLAN Flex/F32
  oracle comparisons, the genuinely populated Q4 32K history (including atomic
  overflow and reset equivalence), and both unchanged public examples. Runtime
  source/executable hashes and raw native output are retained. The connected
  [dependency audit](acceptance/measurements/connected-dependency-audit-20260910/manifest.json)
  also passes advisory, yanked-version, license and source checks without new
  waivers. None of these results promotes experimental families or substitutes
  for production-only coverage and refreshed performance/memory gates.
  The [refreshed sanitizer campaigns](fuzz/evidence/refresh-80b51e8.json) also pass
  10,000 inputs for each of five targets, with no sanitizer diagnostics or
  failure artifacts. This finite campaign is not exhaustive branch coverage.
- **Memory measurements distinguish storage from observed process footprint.**
  The test-only Darwin supervisor brackets synchronized phases, samples a real
  idle baseline, and retains raw timestamps, gaps, and errors. The [executed Qwen/FLAN series](acceptance/measurements/small-model-metal-memory-20260909.json)
  passed 93 phases per model without sampling errors, with warmups kept separate
  from measured repetitions. Sampled maxima are not exact transient peaks;
  kernel lifetime high-water and analytical
  cache bytes remain distinct. Instrumentation is excluded from comparable
  timing series and adds no runtime networking or dependencies.
  The [refreshed source-bound series](acceptance/measurements/small-model-performance-20260910/manifest.json)
  repeats all 93 phases per model without errors and retains signed above-idle
  deltas plus raw-trace hashes; kernel lifetime physical-footprint high-water
  observations are 2,977,486,656 bytes for Qwen and 771,181,504 for FLAN.
- **Production tests and performance comparisons need explicit denominators.**
  After the remaining test modules were separated without changing production
  code, the [native-bound v2 diagnostic](acceptance/measurements/whole-test-file-coverage-v2-20260910.json)
  removes 78 verified test-only files: coverage is 86.1098% crate-wide and
  93.6446% shared/core, below the unchanged 90%/95% targets. All 1,013 tests pass,
  but the higher test-inclusive aggregate does not close the gap; production-only
  qualification remains false. Targeted dispatch/cache/integrity tests are next.
  Qwen's repeated timing remains about 25% slower than its historical baseline;
  the exact historical source rebuilt on the same host now shows the same drop.
  Paired rebuilt-old/current medians differ by at most 1.247%, with three p95
  alerts. This narrows attribution but neither explains nor waives the historical
  regression. FLAN now also has a first Flex CPU timing baseline. These
  measurements do not promote either checkpoint or the experimental families.
- **Batched stopping is part of cache correctness.** New coordinator regressions
  keep a real tiny Qwen cache while one row emits EOS before its peer. Greedy
  and seeded top-1 sampling both preserve the finished row's length and explicit
  position, supply false-valid padding on feedback, and ignore deliberately
  nonfinite logits only for that inactive row. Model-boundary errors and a
  callback failure midway through a batch stop subsequent emissions and decode
  calls. These are deterministic API/state tests, not a new hardware benchmark.
  A further 58 parallel regressions exercise automatic dispatch/loading, Qwen/T5
  cache transactions and binding, and artifact/packed-payload integrity. Their
  worktree tests and warning-denied Clippy pass without runtime changes. The
  combined suite is measured separately before making any new coverage claim.
- **Sparse-model parity fixtures must materialize dormant experts before cloning.**
  Burn 0.21 deliberately gives a cloned uninitialized parameter an independent
  initialization cell. A sparse prompt initializes only selected experts, so
  later tokens can give two apparently identical fixture clones different
  weights. The combined run at `b5ca317` exposed this on the first MoE decode;
  its failed run is not coverage evidence. The corrected test fixture visits
  every parameter before cloning. A counter-valued lazy initializer proves the
  skipped-expert failure deterministically, without relying on random seeds.
  Runtime expert dispatch and parity tolerances are unchanged.
  The [fresh integrated v3 diagnostic](acceptance/measurements/whole-test-file-coverage-v3-20260910.json)
  passes all 1,075 executed tests. With the same 78 test-only files excluded and
  unchanged denominators, coverage rises to 87.6121% crate-wide and 93.8707%
  shared/core; both still fail their targets. The structural audit finds no
  parsed inline-test blockers in the 101 production files. Opaque/generated
  mappings remain included, and production-only qualification stays false.
- **Loader failure contracts extend beyond tensor mathematics.** A further
  [24 checkpoint-contract tests](acceptance/measurements/checkpoint-contract-coverage-20260910.json)
  cover interrupted concatenation without partial target publication, exact
  physical-equivalence memory limits, non-destructive sidecar publication and
  retry, mixed-precision split/alias plans, transactional provenance binding,
  and contradictory retained load counters. The clean combined run passes
  1,099 tests and raises shared/core coverage to 94.3962% (87.8478% crate-wide)
  with unchanged denominators. Runtime code, support claims, coverage targets,
  and qualification requirements are unchanged.
- **Generation owns its failed session, not the caller's next request.** New
  public-contract tests execute real tiny Qwen cache updates before injecting a
  model error or malformed returned logits. They verify no premature streaming
  tokens, session destruction, and a fresh cache when the same model is reused.
  Beam tests retain exact encoder/cache/decoder error categories and reject
  malformed returned encoder diagnostics before another step. Together with
  two JSON map-type tests, the combined suite now passes 1,107 tests; all-feature
  Clippy is warning-free. These are test-only safeguards, not a new coverage
  measurement or hardware acceptance claim.
- **Gemma's multimodal wrapper differs from its text model.** In Transformers
  5.2, `Gemma3ForCausalLM` applies `final_logit_softcapping` but
  `Gemma3ForConditionalGeneration` and PaliGemma never do, and Gemma 3
  attention never applies `attn_logit_softcapping`. The crate rejects those
  settings where the reference ignores them instead of computing a different
  graph. Transformers 5.2's `save_pretrained` also writes inert
  `rope_type`/`rope_theta` keys beside Gemma 3's per-layer-type RoPE mapping;
  they are accepted and dropped.
- **Qwen3-MoE windows follow Transformers 5.2.** At the pinned `7d9754…`
  reference, an enabled sliding window applies to every block and
  `max_window_layers` is ignored (Transformers 4.51 kept blocks below it on
  full attention). The crate follows 5.2 and has a windowed golden test; the
  released Qwen3-30B-A3B disables its window, so it is unaffected.
- **Half-precision gpt-oss comparisons are loose by construction.**
  Transformers' BF16 eager path rounds its attention and router softmaxes to
  BF16, while this crate normalizes in F32. On four real blocks the crate's BF16
  run is no closer to the BF16 reference than to the F32 one, and the BF16
  reference itself is 1.73 logits from F32; half-precision runs are best
  qualified against a truncated F32 reference.
- **MLA compression is an optimized local decode path.** DeepSeek's published
  Python reference expands K/V for a generic cache, whereas algebraic absorption
  lets this crate retain only the latent and rotary key.
- **MTP presence is not ordinary-forward execution.** Base Hugging Face Qwen
  code allowlists the 15 `mtp.*` tensors; the crate and the cited vLLM path
  explicitly execute them. The pinned official Nemotron base loader likewise
  filters exactly 270 physical `mtp.*` keys as its sole unexpected-key class
  and executes none of them. Nemotron 3.5 has one MTP stage containing two
  residual blocks, not two prediction stages; its parity evidence must come
  from the separately pinned vLLM graph.
