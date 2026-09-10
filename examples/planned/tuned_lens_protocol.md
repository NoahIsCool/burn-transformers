# Tuned Lens artifact and protocol prerequisite

Research verified 2026-09-10. This prerequisite note records the original pins
and protocol. The [completed study](../mechanistic/tuned_lens/README.md) now
provides execution, measured results, and a documented correction to analysis
precision; model artifacts, fixture, and acceptance limits remain unchanged.

## Pin the original released pair

Use [paper v4 (26 November 2023)](https://arxiv.org/abs/2303.08112v4).
The current v6 adds updated Muon training and describes the earlier SGD lenses
as undertrained, so its newer results are not targets for this released lens.

The lens configuration explicitly names base model revision
`7e57cc978f5da949f028f36b5baf8f5d6c3281b1`, not an unpinned `main`.

| Artifact | Pinned location | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Model | [pytorch_model.bin](https://huggingface.co/EleutherAI/pythia-160m-deduped-v0/resolve/7e57cc978f5da949f028f36b5baf8f5d6c3281b1/pytorch_model.bin) | 375,035,089 | `2949767274300e90904f58f0c6cf9ff6c174a35e29d07b2784c3051c459ea410` |
| Model config | [config.json](https://huggingface.co/EleutherAI/pythia-160m-deduped-v0/resolve/7e57cc978f5da949f028f36b5baf8f5d6c3281b1/config.json) | 569 | `76eb275107220e450d31258f792a2efcbee109d8b62ae0088260057dec06362f` |
| Released lens | [params.pt](https://huggingface.co/spaces/AlignmentResearch/tuned-lens/resolve/1ac7285852a22309f571c2555efc37375d0c4cda/lens/EleutherAI/pythia-160m-deduped-v0/params.pt) | 28,353,795 | `9644fe7196ae4ab63ce2ce1c7a22f63ae548021a27bac57c4c1cdd200d982cf8` |
| Lens config | [config.json](https://huggingface.co/spaces/AlignmentResearch/tuned-lens/resolve/1ac7285852a22309f571c2555efc37375d0c4cda/lens/EleutherAI/pythia-160m-deduped-v0/config.json) | 220 | `6d1e47917f86bb031e2eb6ea6099a32f914b052fa343061ba62eacb32ddfd364` |

Fetch `tokenizer.json` (2,113,710 bytes), `tokenizer_config.json` (394 bytes), and
`special_tokens_map.json` (99 bytes) from the same historical model revision.
The tokenizer JSON Git blob is `f74dfbfab8f97770a87769c739fb080c21c8bacc`;
preparation must hash its actual bytes. BOS/EOS/UNK are `<|endoftext|>` / ID 0;
there is no added prefix space. Model vocabulary includes 50,304 rows.

The later model revision `26b9eeef106f66faf9b45930f92430a8a77c8799` retains the
identical PyTorch file hash and also provides `model.safetensors` (374,998,696
bytes; SHA-256 `8c134a441807d1614c36959b81fce3596f7e0dd6e5c43c903ad38f4cf9198cf0`).
Prefer the historical pair with strict `torch.load(weights_only=True)` during
conversion, or explicitly verify the later SafeTensors tensors are equivalent.

## Architecture and import contracts

Pythia-160M-v0 has 12 blocks, hidden width 768, 12 heads of width 64, MLP width
3,072, affine LayerNorm epsilon `1e-5`, exact GELU, context 2,048, and untied
input/output embeddings. Attention and MLP run in parallel from separately
normalized copies of the same block input: `x + attention(LN1(x)) + mlp(LN2(x))`.
Only the first 16 channels of each Q/K head receive split-half RoPE, base 10,000.
It uses biased Q/K/V/O and MLP projections; vocabulary projection is bias-free.

Checkpoint names are `gpt_neox.embed_in.weight`, `embed_out.weight`,
`gpt_neox.final_layer_norm.{weight,bias}`, and per block:

- `input_layernorm.{weight,bias}` and `post_attention_layernorm.{weight,bias}`;
- `attention.query_key_value.weight [2304,768]` and bias `[2304]`;
- `attention.dense.weight [768,768]` and bias `[768]`;
- `mlp.dense_h_to_4h.weight [3072,768]`, bias `[3072]`;
- `mlp.dense_4h_to_h.weight [768,3072]`, bias `[768]`.

Packed QKV is **head-major**: reshape output channels `[12,3,64]`, select Q/K/V,
then flatten `[12,64]`. A simple three-way split of dimension 2304 is incorrect.
Weights use PyTorch `[output,input]` orientation. All model parameter payloads
are F16; F32 execution must explicitly cast them. The checkpoint also includes
non-parameter causal mask buffers `[1,1,2048,2048]` (U8), scalar masked biases,
and rotary inverse frequencies `[8]` (F16). Treat ignored buffers explicitly;
do not silently discard unknown parameters. Pin reference-library behavior for
RoPE cache construction rather than assuming saved F16 frequencies are used.
[Reference GPT-NeoX implementation](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/gpt_neox/modeling_gpt_neox.py)

The released lens contains 12 F32 affine corrections with keys `0.weight` through
`11.weight`, each `[768,768]`, and corresponding biases `[768]`. The author code
computes **`h + Linear_i(h)`**, then applies the backbone's final LayerNorm and
vocabulary projection. These saved weights are corrections to identity, not the
complete affine matrix from the paper's mathematical notation.
[Pinned translator code](https://github.com/AlignmentResearch/tuned-lens/blob/65f3135ad1e849e2e5b0f7c5adbd857c787bfb64/tuned_lens/nn/lenses.py#L306-L315)

Translator 0 consumes embeddings; translators 1–11 consume residual outputs of
blocks 0–10. They correspond to Hugging Face `hidden_states[:-1]`. Block 11's
final normalized state is excluded. A raw logit lens applies the same final
LayerNorm and readout without the affine correction. Capture embeddings and
`Layer::decoder(i).residual_post()` to match these boundaries.

## Reference evaluation and bounded target

Paper v4 describes separate validation slices, concatenated into 2,048-token
chunks, with 16.4M evaluation tokens. Figure 5 compares layerwise predictive
loss; its y-axis is bits per byte. No exact Pythia-160M-v0 numeric table or source
subset IDs are published there. Author tutorials instead train on Pile
`val.jsonl` and evaluate on `test.jsonl`; preserve that distinction.
[Tutorial](https://github.com/AlignmentResearch/tuned-lens/blob/65f3135ad1e849e2e5b0f7c5adbd857c787bfb64/docs/source/tutorials/training_and_evaluating_lenses.rst)

Author evaluation uses `KL(final || lens)` at matching token positions, including
the final position. Next-token CE uses predictions `0..T-2` against input tokens
`1..T-1`. Natural logarithms produce nats per token; conversion to bits per byte
requires the author's corpus-specific token/byte factor. With one process and
batch size one, the full budget floors to 8,007 chunks: 16,398,336 input positions
for KL, 16,390,329 CE terms. Preprocessing inserts EOS between documents and once
at the beginning, then chunks; it does not insert EOS at every chunk boundary.
[Evaluation](https://github.com/AlignmentResearch/tuned-lens/blob/65f3135ad1e849e2e5b0f7c5adbd857c787bfb64/tuned_lens/scripts/eval_loop.py#L259-L305),
[preprocessing](https://github.com/AlignmentResearch/tuned-lens/blob/65f3135ad1e849e2e5b0f7c5adbd857c787bfb64/tuned_lens/data.py#L48-L113)

A suitable small fixture source is the author's
[Pile validation/test release](https://huggingface.co/datasets/EleutherAI/pile_val_test),
revision `05b327037e6301f256d8df32193756edc4c8e3bd`.
`test.jsonl` is 1,372,688,804 bytes, SHA-256
`f87cb69612a54c672cb5afb0f3221ec15c05fae1f34798ae8973561a4b6074da`.
A bounded HTTP range `bytes=0-65535` is available (206 response); its 65,536 bytes
have SHA-256 `9ca7b4cc0e0689a185365ad657ae8b451083d7089d27b115c9e90c03962afe92`
and contain 25 complete JSON records. No full corpus download is required.

Proposed default: the first four complete test records, each truncated to 128
input tokens, with per-record EOS handling documented and token IDs hashed.
This is a **short-context bounded adaptation**, not the paper's full evaluation.
Report all 12 translators' KL and CE plus raw-logit and final-model baselines;
show a readable token-by-layer table. Verify imported outputs against the pinned
author implementation on identical tokens. Establish numerical parity tolerances
before measuring Rust results. The scientific target is the Figure 5 comparison
method and predicted direction; do not invent published numeric pass thresholds
or require improvements at every layer on a tiny new fixture. Report exceptions.

The pinned author code commit above predates v4 and preserves the relevant
semantics; the authors do not identify it as their exact experimental run commit.
No lens fitting, model training, or full example execution was performed here.
