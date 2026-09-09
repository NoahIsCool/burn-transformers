# Qwen3.5-0.8B v0.1 profile

This pre-advertisement profile defines the intended bounded Qwen support claim
for the first crate release. It applies to `Qwen/Qwen3.5-0.8B` at immutable revision
`2fc06364715b967f1860aea9cf38778875588b17` and to text inference only. The
repository contains a vision tower and a one-layer multi-token-prediction (MTP)
head; neither is part of the default v0.1 execution graph.

The machine-readable record intentionally remains `advertised_supported=false`.
The retained, independently reviewed official-reference bundle now passes its
Metal replay. Promotion still requires the long-context gate described below
and the crate-wide [release gates](RELEASE_GATES.md); numerical parity alone is
not a complete release qualification.

The detailed tensor map is recorded in
[QWEN3_5_CHECKPOINT_MAPPING.md](QWEN3_5_CHECKPOINT_MAPPING.md), and machine-readable
real-checkpoint evidence is in
[the acceptance record](acceptance/qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17.json).

## Artifact preparation

The Rust runtime performs no network access. Download the exact revision with
an external Hugging Face CLI:

```sh
hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-dir /Volumes/models/qwen3.5-0.8b
```

The local root must contain `config.json`, `model.safetensors.index.json`, its
single referenced SafeTensors shard, and—when text tokenization or chat
rendering is requested—`tokenizer.json`, `tokenizer_config.json`, and
`chat_template.jinja`. The loader canonicalizes the root and rejects escaping
shard or tokenizer symlinks.

The official checkpoint stores BF16 parameters. On the accepted M3 Metal path,
select `DTypePolicy::Cast(TensorDType::F16)`. Conversion occurs one mapped
tensor at a time; the loader does not stage a second full copy of the model.
`DTypePolicy::Auto` is appropriate only when the selected backend can execute
the checkpoint's source dtype faithfully.

## Runnable chat example

On macOS, the packaged example performs strict loading, renders the checkpoint
chat template without thinking tokens, tokenizes it without duplicating special
delimiters, prefills an external cache, and decodes a continuation:

```sh
cargo run --release \
  --manifest-path crates/transformers/Cargo.toml \
  --no-default-features \
  --features std,metal,chat-templates \
  --example qwen35_chat -- \
  /Volumes/models/qwen3.5-0.8b \
  "Explain why a KV cache speeds up decoding in one sentence." \
  64
```

The example uses prefill chunks of at most 256 physical token columns. The
cache capacity is the valid prompt width plus `max_new_tokens - 1`: logits from
the prompt select the first new token, so the final emitted token never needs
to be decoded back into cache state.

## Strict load contract

For the default base-only profile, a successful load has these exact logical
outcomes:

- 320 text storage tensors are read and applied;
- one LM-head alias reuses the tied token embedding;
- 153 visual tensors and 15 MTP tensors are explicitly allowlisted and reported;
- no unexplained missing or unexpected tensors remain;
- tokenizer IDs, when attached, are checked against the 248,320-row embedding;
- normalized generation EOS IDs are checked against the same vocabulary.

`LoadedModel` retains the final `LoadReport`, source-read counters, tokenizer,
generation defaults, canonical local root, and optional caller-supplied upstream
identity. Repository and revision strings are provenance labels; callers that
need content authentication should verify the hashes in the acceptance record
before loading.

## Inference and cache semantics

Qwen3.5-0.8B contains eighteen Gated DeltaNet layers and six full-attention
layers in a three-to-one schedule. Decode state therefore combines FP32
recurrent association matrices and convolution history with native GQA KV
caches. State lives outside model parameters and supports reset, reorder with
duplication, truncate, and failure-atomic prefill/decode.

Both stateless and cached paths reject empty batches, out-of-vocabulary IDs,
malformed masks, fully masked prefill rows, negative positions, positions at or
beyond the declared context, and requests that exceed cache capacity. A false
decode mask entry is an exact no-op for a finished batch row. `true` always
means a valid token.

The high-level generation coordinator supports greedy and seeded stochastic
selection, temperature, top-k, top-p, repetition and presence penalties,
multiple EOS IDs, token stop sequences, log probabilities, streaming callbacks,
ragged rectangular batches, and optional bounded prefill chunks. Use
`LocalTokenizer::encode_chat` for checkpoint chat templates; it deliberately
disables tokenizer post-processing after rendering so role delimiters appear
exactly once.

## Validated evidence and explicit exclusions

The historical real F16 Metal run strictly loaded the exact checkpoint, matched
the official full-vocabulary prompt logits at cosine `0.999998221`, reproduced
the official top ten IDs, generated the same 32 greedy tokens, and matched
cached and stateless top-1 decisions at every tested prefix. Its 993,280-byte
raw official logit vector was not retained, however. The committed offline
[`generate_qwen35_08b_reference.py`](scripts/generate_qwen35_08b_reference.py)
producer creates one immutable external bundle containing the prompt row, all
32 cached decision rows, all 32 teacher-forced rows, and strict provenance. The
Rust consumer requires independent SHA-256 anchors for all four files and then
makes every full-vector comparison mandatory; there is no optional-oracle
fallback. The stable-prompt bundle has now been generated, independently pinned,
and replayed successfully as described below. The committed small `Hello` JSON
fixture remains useful sampled-logit regression evidence but is not a replacement
for the full-vector gate.

Generate the bundle into an existing external parent only after installing the
package versions authenticated by the producer:

```sh
python3 crates/transformers/scripts/generate_qwen35_08b_reference.py \
  --model-dir /Volumes/models/qwen3.5-0.8b \
  --output-dir /Volumes/models/transformers-acceptance
```

After a second maintainer reviews and pins the four printed digests, the bundle
can be authenticated without loading weights and then replayed against Burn:

```sh
TRANSFORMERS_QWEN35_REFERENCE_DIR=/Volumes/models/transformers-acceptance \
cargo test --release --manifest-path crates/transformers/Cargo.toml \
  --no-default-features --features std,metal,chat-templates \
  --test local_artifacts validates_pinned_qwen35_official_reference_bundle \
  -- --ignored --exact

TRANSFORMERS_QWEN35_DIR=/Volumes/models/qwen3.5-0.8b \
TRANSFORMERS_QWEN35_REFERENCE_DIR=/Volumes/models/transformers-acceptance \
cargo test --release --manifest-path crates/transformers/Cargo.toml \
  --no-default-features --features std,metal,chat-templates \
  --test local_artifacts loads_pinned_qwen35_weights_and_runs_metal_inference \
  -- --ignored --exact
```

A separately reviewed Q4/G64 format-v2 sidecar is exactly `280,044,715` bytes
with SHA-256
`cca925f24ab3b61a76fbbd5aefb8e0f2052da1b4636de9e996b890859cae5fd5`.
It binds the source checkpoint and load plan, contains 186 packed matrices and
keeps 134 tensors in F16, and passed stateless/cached Metal smoke execution with
top-1 token 13 and cosine `1.000000000`. The ignored 32K envelope gate pins this
exact artifact, revalidates every payload and provenance record, allocates six
full-GQA KV layers plus eighteen FP32-DeltaNet recurrent layers at capacity
32,768, and executes one stateless token at explicit position 32,767. That gate
has not yet been run and, even when green, is deliberately not a 32K prefill or
long-context quality result.

Tiny deterministic tests cover hybrid prefill/decode equivalence, ragged
padding, cache lifecycle and failure atomicity, MTP transaction restoration,
and invalid public inputs.

The following remain experimental and are not part of the v0.1 Qwen support
claim:

- the vision tower and multimodal position inputs;
- MTP/speculative decoding with real weights, despite the implemented
  batch-one transaction API;
- Q4 and Q8 quality guarantees (the authenticated packed Q4 result is execution
  smoke only);
- training loops and dropout behavior;
- long-context performance or numerical guarantees pending the coordinated 32K
  cache/position envelope and a separate populated-history stress run;
- dynamic-NTK cached decoding after the original context, which would require
  re-rotating stored keys.

Configuration values that would select an unimplemented graph fail closed.
Specifically, the recurrent state must be FP32, the inherited `mlp_only_layers`
list must be empty, dedicated MTP embeddings are rejected, and DeltaNet output
gating accepts only the equivalent `silu` and `swish` spellings used by the
pinned reference and released metadata.

## Stable reference prompt selection

The original `Hello` continuation contains a one-binary16-ULP decision at token
14 and is retained as a historical sampled-logit diagnostic, not a stable exact
generation oracle. The reference-only selector in commit `3a94d3d` fixes twelve
ordinary explanatory chat prompts before evaluation. Candidate 8, “Explain why
the seasons change in four complete sentences.”, is the first to satisfy all
32 greater-than-two-ULP margins, exact official cached/teacher decisions, and
full-vector self-cosine of at least 0.999. No Burn outputs participate in selection.

Its 23-token non-thinking chat prompt has minimum margin/ULP ratio **6.0** and
minimum official cached/teacher cosine **0.999997463582**. The external
`qwen3.5-0.8b-2fc06364-prompt-selection.bundle/selection.json` retains the fixed
candidate list and all nine attempted results: **152,690 bytes**, SHA-256
`015dad3595c817252f3e776b148f1af34105de2351d70237a004091260be31cb`.
Producer and consumer pin this separate release prompt and correctly slice
teacher-forced logits starting at `prompt_length - 1`. The four-file oracle was
generated at `06f39e7`, independently rehashed, and replayed on Burn Metal F16.
It matches all 32 greedy decisions on cached and teacher-forced paths; minimum
full-vocabulary reference cosine is **0.999996242** (gate **0.999**). The acceptance
record pins every payload and its producer provenance. Both target checkpoint
and reference sources are authenticated; this result still does not establish
32K stress, standardized performance, or crate-wide release completion.
