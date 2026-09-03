# Qwen3.5 and Qwen3.8 checkpoint mapping

This document records the exact metadata-only tensor contract implemented by
`models::qwen3_5`. It was checked against the official SafeTensors index and
headers at these immutable revisions; no tensor payload was downloaded:

- `Qwen/Qwen3.5-0.8B` at
  `2fc06364715b967f1860aea9cf38778875588b17`
- `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

The corresponding golden inventories contain all 488 and 1,199 tensor names,
shapes, and dtypes reported by those official headers. They are metadata test
fixtures, not derived model weights.

## Namespaces and strictness

The official releases use `Qwen3_5ForConditionalGeneration`, whose checkpoint
language tower is rooted at `model.language_model`. The internal causal-LM graph
uses the stable `model` root, so mapping removes exactly `.language_model`:

```text
model.language_model.layers.3.self_attn.k_proj.weight
                              -> model.layers.3.self_attn.k_proj.weight
```

Text-only `Qwen3_5ForCausalLM` artifacts already use the `model` root and are
mapped without a prefix rewrite. Only conditional-generation sources allowlist
`model.visual.*`. Every concrete visual key is preserved in `LoadReport` as an
allowlisted ignored tensor; unknown non-visual tensors still fail strict load
planning.

For tied embeddings, the checkpoint's sole embedding tensor produces
`model.embed_tokens.weight`, and `lm_head.weight` is a storage alias of that
target. Untied checkpoints consume `lm_head.weight` directly. The logical
expected inventory therefore always includes both parameters.

## Decoder formulas

Let `H` be `hidden_size`, `I` be `intermediate_size`, `Nq` and `Nkv` be the full
attention query and KV head counts, and `D` be `head_dim`. Dense SwiGLU layers
have `gate_proj, up_proj: [I, H]` and `down_proj: [H, I]`. Every decoder layer
also has two RMS normalization vectors of shape `[H]`.

Full-attention tensors are:

| Tensor suffix | Shape |
| --- | --- |
| `self_attn.q_proj.weight` | `[2 * Nq * D, H]` when gated, otherwise `[Nq * D, H]` |
| `self_attn.k_proj.weight`, `v_proj.weight` | `[Nkv * D, H]` |
| `self_attn.o_proj.weight` | `[H, Nq * D]` |
| `self_attn.q_norm.weight`, `k_norm.weight` | `[D]` |

If `attention_bias` is enabled, Q/K/V biases match their projection output
widths and the output bias has shape `[H]`.

Qwen's gated query is not serialized as a flat `[all Q, all gates]` pair. The
reference implementation first views projection output as
`[..., Nq, 2 * D]` and splits the last dimension. Consequently, source rows are
conceptually `[Nq, 2, D, ...]`, with Q then gate adjacent inside each head. The
mapping preserves that order exactly because the Qwen family attention graph
performs the same per-head reshape and split. Query biases and MTP query
projections follow the identical layout. A generic flat split into two channel
halves would be incorrect.

For DeltaNet, define `K = linear_num_key_heads * linear_key_head_dim` and
`V = linear_num_value_heads * linear_value_head_dim`. Its exact tensor set is:

| Tensor suffix | Shape |
| --- | --- |
| `linear_attn.A_log`, `dt_bias` | `[linear_num_value_heads]` |
| `linear_attn.in_proj_a.weight`, `in_proj_b.weight` | `[linear_num_value_heads, H]` |
| `linear_attn.in_proj_qkv.weight` | `[2 * K + V, H]` |
| `linear_attn.conv1d.weight` | `[2 * K + V, 1, linear_conv_kernel_dim]` |
| `linear_attn.in_proj_z.weight` | `[V, H]` |
| `linear_attn.norm.weight` | `[linear_value_head_dim]` |
| `linear_attn.out_proj.weight` | `[H, V]` |

The asymmetry is intentional: decay and timestep parameters index value heads,
not key heads. The 27B checkpoint therefore stores 48 rows for `A_log`,
`dt_bias`, `in_proj_a`, and `in_proj_b`, despite having 16 linear key heads.

## Multi-token prediction

When enabled, released checkpoints contribute four shared MTP tensors:

```text
mtp.fc.weight                         [H, 2 * H]
mtp.pre_fc_norm_embedding.weight      [H]
mtp.pre_fc_norm_hidden.weight         [H]
mtp.norm.weight                       [H]
```

Each `mtp.layers.i` is a full-attention decoder layer with the same attention,
normalization, and SwiGLU dimensions as the base network. Thus the released
single-layer MTP contributes 15 tensors in total. It has no dedicated embedding
or output head. The safe default is `include_mtp = false`, because the base
causal-LM graph does not instantiate an MTP parameter sink; this records `mtp.*`
as an explicit optional omission. Setting it to true consumes all declared
tensors and is valid only for an MTP-capable graph. A config that requests
dedicated MTP embeddings is rejected until a released checkpoint establishes
exact key names and sharing semantics.

## Golden acceptance counts

The full-inventory, MTP-enabled contract has these counts:

| Checkpoint | Text/MTP source tensors | Logical targets | Tied aliases | Reported visual tensors |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 335 | 336 | 1 | 153 |
| Qwen3.8-27B | 866 | 866 | 0 | 333 |

With the executable base-graph default, Qwen3.5-0.8B consumes 320 tensors,
produces 321 logical targets through one tied alias, and reports 168 ignored
tensors (153 visual plus 15 MTP). Qwen3.8-27B consumes and produces 851 tensors
and reports 348 ignored tensors (333 visual plus 15 MTP).

Tests synthesize manifests from the complete official header inventories and
require strict planning with exactly these counts. They also exercise a tiny
two-layer text-only fixture, safe-default MTP omission, malformed gated-query
widths, projection biases, checked dimension overflow, and allowlist
deduplication.

## Primary sources

- [Qwen3.5-0.8B official repository](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)
- [Qwen3.8-27B official repository](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
- [Hugging Face Qwen3.5 reference implementation](https://github.com/huggingface/transformers/blob/cfa309cfa34b5497a32fd50af49aa97aac94c054/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
