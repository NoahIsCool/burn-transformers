# Transformer architecture and innovation matrix

This document maps the transformer-side mechanisms used by the exact, pinned
checkpoint revisions targeted by this crate. It was assembled from the pinned
configuration, the model author's or framework's reference implementation, and
the corresponding Burn implementation. The matrix describes architecture, not
release support: a check mark does **not** mean that all real-weight acceptance
gates have passed. See [the runtime status table](README.md) and
[the compatibility evidence matrix](ARCHITECTURE_COMPATIBILITY.md) for that
separate question.

Only the executable text graph is counted. Vision towers, tokenizer behavior,
quantization formats, serving systems, and broad training recipes such as FLAN
instruction tuning and UL2's Mixture-of-Denoisers are deliberately outside the
matrix. Those recipes change how weights are learned, but not the transformer
operations executed by this inference crate. Architecture-coupled router
objectives are included as optional forward diagnostics; the runtime computes
them only when requested and never adds them to inference logits.

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
| **Nem-3.5** | [NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 @ `a9904d…`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/config.json) | [NVIDIA architecture/recipe](https://docs.nvidia.com/nemotron/nightly/nemotron/lightning35/pretrain.html), [Nemotron-H reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py) | [`nemotron_h`](src/models/nemotron_h/model.rs) |
| **Nem-Nano** | [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 @ `bf77c3…`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/config.json) | [Pinned NVIDIA reference](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/modeling_nemotron_h.py), [technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) | [`nemotron_h`](src/models/nemotron_h/model.rs) |

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
| [Round-before-scale RMSNorm](#innovation-round-before-scale) | — | — | ✓ | ✓ | — | — | — | — | — | — |
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
| [Transactional speculative cache](#innovation-transactional-cache) | target + MTP | target + MTP | — | — | — | — | — | — | — | — |
| [Lossless speculative decoding](#innovation-speculative-decoding) | MTP, batch 1 | MTP, batch 1 | — | — | — | — | — | — | — | — |
| [Cache-aware beam search](#innovation-beam-search) | — | — | — | — | — | ✓ | ✓ | ✓ | — | — |

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
sequence-to-sequence contracts in [`architecture/mod.rs`](src/architecture/mod.rs).
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

**Implementation:** shared attention in [`attention.rs`](src/attention.rs), MLA
in [`deepseek_v2/mla.rs`](src/models/deepseek_v2/mla.rs), and Muse's specialized
path in [`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs).
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
native KV heads; repetition happens only for the attention operation.

**Implementation:** [`attention.rs`](src/attention.rs) and the native-head cache
in [`cache/tensor/kv.rs`](src/cache/tensor/kv.rs). **Reference:**
[Grouped-Query Attention](https://arxiv.org/abs/2305.13245).

<a id="innovation-mla"></a>
### Multi-head latent attention (MLA)

MLA compresses each token to a shared latent $c_t=W_{DKV}x_t\in\mathbb{R}^r$.
Per-head up-projections recover content keys and values,
$k^C_{t,h}=W^K_hc_t$ and $v_{t,h}=W^V_hc_t$. A small, shared rotary key
$k^R_t$ is retained separately, while each query is split into content and
rotary channels. DeepSeek-V2-Lite uses $r=512$, 16 heads, 128 NoPE content
channels, 64 rotary channels, and 128 value channels. Its query path is direct
because the pinned `q_lora_rank` is null.

**Implementation:** [`deepseek_v2/mla.rs`](src/models/deepseek_v2/mla.rs).
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
[`deepseek_v2/mla.rs`](src/models/deepseek_v2/mla.rs) and external state in
[`deepseek_v2/model/decode.rs`](src/models/deepseek_v2/model/decode.rs).
**Reference:** [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

<a id="innovation-sliding-attention"></a>
### Sliding-window attention

For query position $i$ and window $W$, local causal attention admits only
$\max(0,i-W+1)\le j\le i$. A sparse window kernel can reduce prefill work to
$O(nW)$ once $n\gg W$, and decode needs only the newest $W$ projected K/V
entries. The current portable Burn path, however, applies a dense $n\times n$
mask during fresh-prompt prefill, so its prefill compute remains $O(n^2)$; its
ring-backed decode history is window-bounded. Muse uses window 2,048 in 39 of
52 text blocks, interleaved with 13 global blocks that restore long-range
connectivity.

**Implementation:** masking and layer selection in
[`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs), cached execution in
[`muse_glimmer/model/decode.rs`](src/models/muse_glimmer/model/decode.rs), and
ring planning in [`cache/ring.rs`](src/cache/ring.rs).
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
Muse instead uses parameter-free (“scale-free”) head normalization.

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
$3.87/\sqrt{128}$. Softmax is accumulated in FP32 and cast back to the query
dtype. The Burn path pre-scales Q by 3.87 but leaves the attention primitive's
optional scale unset: that selects its mathematically identical default factor
while preserving fused-kernel eligibility. This changes attention entropy
without letting raw projection norms determine temperature.

**Implementation:** [`MuseGlimmerAttention`](src/models/muse_glimmer/model.rs).
**Reference:** the [pinned Muse-Glimmer reference](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/muse_glimmer/modular_muse_glimmer.py).

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

**Implementation:** validated frequencies in
[`position/rope.rs`](src/position/rope.rs) and tensor rotation in
[`position/tensor.rs`](src/position/tensor.rs). **Reference:**
[RoFormer](https://arxiv.org/abs/2104.09864).

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

**Implementation:** [`deepseek_v2/mla.rs`](src/models/deepseek_v2/mla.rs).
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

**Implementation:** [`YarnConfig`](src/position/rope.rs) and tensor frequencies
in [`position/tensor.rs`](src/position/tensor.rs). **Reference:**
[YaRN](https://arxiv.org/abs/2309.00071) and the pinned DeepSeek config.

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
computed bias.

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

**Implementation:** recurrent and triangular chunk forms in
[`delta_net.rs`](src/delta_net.rs), state in
[`cache/tensor/recurrent.rs`](src/cache/tensor/recurrent.rs).
**Reference:** [Gated Delta Networks](https://arxiv.org/abs/2412.06464) and the
pinned Qwen3.5 implementation.

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
The pinned models use 64 heads of width 64, state size 128, eight shared B/C
groups, convolution width 4, and FP32 scan state.

**Implementation:** [`nemotron_h/model/mamba2.rs`](src/models/nemotron_h/model/mamba2.rs).
**Reference:** [Transformers are SSMs / Mamba-2](https://arxiv.org/abs/2405.21060).

<a id="innovation-depthwise-convolution"></a>
### Depthwise causal convolution

Before the recurrence, each channel receives a short local filter
$z_{t,c}=\sum_{\ell=0}^{K-1}w_{c,\ell}x_{t-\ell,c}$. “Depthwise” means channels
do not mix inside this convolution; “causal” means no $x_{t+\ell}$ appears.
It gives DeltaNet and Mamba2 an inexpensive local inductive bias and requires
only $K-1$ historical values per channel during decode. Both pinned families use
$K=4$.

**Implementation:** convolution setup and state handling in
[`delta_net.rs`](src/delta_net.rs) and
[`nemotron_h/model/mamba2.rs`](src/models/nemotron_h/model/mamba2.rs).

<a id="innovation-recurrent-decode"></a>
### Fixed-state recurrent decoding

DeltaNet and Mamba2 reduce an arbitrarily long processed prefix to a fixed-size
state plus a short convolution history. A new token advances that state once,
so decode work and state bytes do not grow with prefix length for those layers.
This contrasts with attention, which must retain or revisit an increasing KV
history (or a bounded sliding window).

**Implementation:** reference recurrences in [`delta_net.rs`](src/delta_net.rs)
and [`nemotron_h/model/mamba2.rs`](src/models/nemotron_h/model/mamba2.rs), with
external storage in [`cache/tensor/recurrent.rs`](src/cache/tensor/recurrent.rs).

<a id="innovation-chunked-prefill"></a>
### Chunk-parallel recurrent prefill

Token-by-token recurrence is poor at exploiting accelerator parallelism over a
long prompt. DeltaNet algebraically resolves the within-chunk updates as a
causal triangular system followed by batched matrix products. Mamba2 constructs
a masked semiseparable transfer matrix within each chunk and carries only the
terminal state between chunks. Both paths are mathematically equivalent to the
transparent recurrence up to floating-point reassociation.

**Implementation:** [`chunked_gated_delta_rule`](src/delta_net.rs) and the
chunked mode in [`nemotron_h/model/mamba2.rs`](src/models/nemotron_h/model/mamba2.rs).

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

**Implementation:** shared softmax MoE in [`moe.rs`](src/moe.rs), DeepSeek
specialization in [`deepseek_v2/model.rs`](src/models/deepseek_v2/model.rs), and
Nemotron's sigmoid router in [`nemotron_h/model.rs`](src/models/nemotron_h/model.rs).
**Reference:** [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

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
loss.

**Implementation:** [`NemotronHMoe`](src/models/nemotron_h/model.rs).
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
multiplier silently applied at inference. None of these losses changes inference
logits internally.

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
precision inputs and casts according to each checkpoint's ordering.

**Implementation:** [`normalization.rs`](src/normalization.rs). **Reference:**
[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).

<a id="innovation-zero-centered-rmsnorm"></a>
### Zero-centered RMSNorm parameters

The normalized activation is conventional, but the checkpoint stores a learned
offset $w$ and uses scale $\gamma=1+w$. Initializing $w=0$ therefore gives the
identity scale and keeps learned values centered near zero. Qwen3.5/3.8 uses
this throughout its text graph. Muse uses it for the four norms inside each
block, while its final norm stores an ordinary direct scale.

**Implementation:** `RmsNormConfig::zero_centered` in
[`normalization.rs`](src/normalization.rs), selected in
[`qwen3_5/model.rs`](src/models/qwen3_5/model.rs) and
[`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs).

<a id="innovation-round-before-scale"></a>
### Round-before-scale RMSNorm

Reduced-precision ordering is observable. Qwen3-MoE and DeepSeek compute
$u=x_{f32}/\sqrt{\operatorname{mean}(x_{f32}^2)+\epsilon}$, cast $u$ back to the
residual dtype, and only then multiply by the learned scale. Other families may
multiply in FP32 and round once at the end. Matching this extra rounding point
is necessary for checkpoint-level logit parity.

**Implementation:** `round_before_scale` in
[`normalization.rs`](src/normalization.rs), configured by
[`qwen3_moe/model.rs`](src/models/qwen3_moe/model.rs) and
[`deepseek_v2/model.rs`](src/models/deepseek_v2/model.rs).

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
the embedding table's norm.

**Implementation:** `scale_free_rms_norm` in
[`muse_glimmer/model.rs`](src/models/muse_glimmer/model.rs).

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

**Implementation:** [`MuseGlimmerForCausalLm::forward`](src/models/muse_glimmer/model.rs).
**Reference:** the pinned Muse configuration.

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

**Implementation:** [`qwen3_5/model/mtp.rs`](src/models/qwen3_5/model/mtp.rs)
and [`NemotronHNextTokenHead`](src/models/nemotron_h/model.rs), with exact tensor
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

<a id="innovation-ring-cache"></a>
### Sliding ring KV cache

A windowed layer maps absolute position $p$ to logical slot $p\bmod W$ and
retains only the newest $W$ entries. The local mirrored layout allocates $2W$,
writes each logical value at both $p\bmod W$ and $p\bmod W+W$, then exposes one
contiguous chronological $W$-wide slice across the wrap point. If a speculative
proposal could overwrite a still-committed slot, proposal tensors must be
staged until acceptance; changing only the logical length cannot restore
destroyed K/V values.

**Implementation:** safe plans in [`cache/ring.rs`](src/cache/ring.rs), tensor
storage in [`cache/tensor/kv.rs`](src/cache/tensor/kv.rs), and Muse integration
in [`muse_glimmer/model/decode.rs`](src/models/muse_glimmer/model/decode.rs).

<a id="innovation-cross-kv-cache"></a>
### Immutable cross-attention KV cache

Encoder states are fixed during target generation, so each decoder layer can
compute $K_e=H_eW_K$ and $V_e=H_eW_V$ once. Every subsequent step projects only
its target query. Unlike self-KV, these tensors never append; beam search only
reorders or shares their batch rows together with the corresponding encoder
output.

**Implementation:** [`T5CrossAttentionCache`](src/models/t5/model/decode.rs).

<a id="innovation-recurrent-cache"></a>
### External recurrent-state cache

The cache for a recurrent mixer contains its sufficient state rather than one
entry per historical token. Qwen stores each DeltaNet matrix memory plus the
short convolution tail. Nemotron stores each Mamba2 SSM tensor plus its
convolution tail. Keeping this state outside Burn parameters makes reset,
reorder, truncate, and failure-atomic publication explicit and keeps stateless
full-precision graphs autodiff-safe.

**Implementation:** [`cache/tensor/recurrent.rs`](src/cache/tensor/recurrent.rs)
and [`cache/tensor/state.rs`](src/cache/tensor/state.rs), consumed by Qwen and
Nemotron decode modules.

<a id="innovation-transactional-cache"></a>
### Transactional speculative cache

Speculation tentatively advances both the target and draft states. A transaction
records a mark, stages appends, and then either `commit_prefix(a)` to publish
only the accepted prefix or `rollback()` to restore the committed logical state.
Rollback leaves committed cache tensors unchanged and discards staging metadata;
rejected staging bytes may remain allocated but are no longer addressable. Qwen
maintains an independent MTP KV cache alongside target GQA/DeltaNet state; cache
alignment requires the target to remain one token ahead of MTP.

**Implementation:** lifecycle rules in
[`cache/transaction.rs`](src/cache/transaction.rs), tensor snapshots in
[`cache/tensor/state.rs`](src/cache/tensor/state.rs), and Qwen integration in
[`qwen3_5/model/decode.rs`](src/models/qwen3_5/model/decode.rs).

<a id="innovation-speculative-decoding"></a>
### Lossless speculative decoding

An MTP distribution $q_i$ proposes tokens that the target distribution $p_i$
then verifies. Greedy mode accepts exactly when proposal and target argmax
match. Stochastic mode accepts proposed token $y_i$ with
$a_i=\min(1,p_i(y_i)/q_i(y_i))$; on rejection it samples from normalized
$(p_i-q_i)_+$. This rejection rule preserves the target model's marginal
distribution exactly. If all drafts pass, the target emits one bonus token.
The current Qwen coordinator is intentionally batch-one and defaults to two
drafts. Nemotron's next-token head exists, but no equivalent coordinator is yet
implemented for it.

**Implementation:** [`qwen3_5/model/speculative.rs`](src/models/qwen3_5/model/speculative.rs).
**Reference:** [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192).

<a id="innovation-beam-search"></a>
### Cache-aware beam search

Beam search retains the $B$ highest-scoring target prefixes. Each expansion adds
token log probability to cumulative score, then prunes globally within a source
row; finalized hypotheses can be ranked with a length penalty. Stateful search
must reorder decoder self-KV, encoder rows, and immutable cross-KV by the same
parent-beam indices, otherwise scores and hidden histories silently diverge.

**Implementation:** the generic coordinator in
[`generation/seq2seq.rs`](src/generation/seq2seq.rs) and T5 defaults in
[`t5/generation.rs`](src/models/t5/generation.rs).

## Reference/runtime distinctions discovered during review

- **Architecture is not acceptance.** The matrix records graph features even
  where the crate currently has only schema, synthetic, or checkpoint-contract
  evidence. Consult the two status documents linked at the top before claiming
  support for a checkpoint.
- **Nemotron is NoPE.** The exact Nano and 3.5 attention implementations do not
  apply RoPE. Serialized RoPE-shaped fields are compatibility metadata, not an
  executed position transform.
- **FLAN-UL2's LM head is tied.** Its pinned config omits
  `tie_word_embeddings`; the pinned upstream T5 schema and the local Serde
  default both resolve that omission to `true`. FLAN Small and XXL explicitly
  set `false`.
- **Qwen's two gates are different.** Full attention uses a sigmoid output
  gate. The Qwen3.8 `"swish"` field describes DeltaNet's output gated norm.
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
- **MLA compression is an optimized local decode path.** DeepSeek's published
  Python reference expands K/V for a generic cache, whereas algebraic absorption
  lets this crate retain only the latent and rotary key.
- **MTP presence is not ordinary-forward execution.** Base Hugging Face Qwen
  code allowlists the 15 `mtp.*` tensors; the crate and the cited vLLM path
  explicitly execute them. Nemotron 3.5 has one MTP stage containing two
  residual blocks, not two prediction stages.
