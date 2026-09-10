# GPT-2 numerical validation

This module is a stateless GPT-2 implementation for experiments. It provides
public embeddings, blocks, affine normalization, tied readout, and activation
hooks. It has no cache API. The validated execution profile is Flex/F32 on CPU;
training behavior and checkpoint quantization are outside this validation.

`Gpt2Activation::GeluNew` implements standard GPT-2's tanh approximation.
`Gpt2Activation::Gelu` implements exact GELU. These choices are explicit because
the Greater-Than authors' released circuit specifies exact GELU while the
standard Hugging Face checkpoint configuration specifies `gelu_new`.

For direct component effects, capture `Layer::decoder(last).residual_post()`
and use `model.readout(residual)` after changing that residual. Readout applies
final LayerNorm followed by the tied vocabulary projection. By contrast,
`model.project(hidden)` expects states that are already normalized.

## Compact regression fixture

Ordinary tests reconstruct a tiny synthetic checkpoint in a temporary directory.
The deterministic parameter generator is independent of transformer execution;
its shapes and PyTorch reference outputs are in `fixtures/tiny_reference.json`.
No weight file is tracked. Regenerate the reference with:

```sh
python3 src/models/gpt2/prepare_reference.py
cargo test --release --offline --locked --features autodiff models::gpt2 --lib
```

Tests cover both GELUs, checkpoint-native Conv1D orientation and fused QKV
splitting, actual MLP-output boundaries, empty/capture/edit/restoration hooks,
public block composition, causal/bidirectional visibility, padding-aware logits,
strict loading failures, physical/index-only tied-weight equivalence, recoverable
normalization contract errors, unchanged capture-only gradients, and registered
adapter gradients.
Gradient checks do not train a model.

## Full checkpoint reference

The pinned model and tokenizer are
[`openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e`](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e).
Its 548,105,171-byte safetensors file has verified SHA-256
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
The local preparation helper never downloads files. With that checkpoint ready:

```sh
python3 src/models/gpt2/prepare_reference.py \
  --checkpoint /local/gpt2 --activation gelu_new --output /tmp/gpt2-reference.json
GPT2_CHECKPOINT=/local/gpt2 GPT2_REFERENCE=/tmp/gpt2-reference.json \
  cargo test --release --offline --locked --features autodiff \
  models::gpt2::model::tests::pretrained_gpt2_matches_pytorch_last_logits_and_late_mlp_outputs \
  --lib -- --ignored --nocapture
```

Repeat with `--activation gelu` for the authors' declared activation graph.
The reference fixture records the selected configuration, so the Rust test
executes the same choice. Preparation used PyTorch 2.10.0, Transformers 5.2.0,
one CPU thread, eager attention, evaluation mode, and no cache. Both reference
prompts are recorded as exact token IDs in the helper. They are numerical
development cases, not the Greater-Than held-out evaluation.

## Provenance and tolerance investigation

The initial all-logits absolute bound of `3e-4` failed: standard GPT-2's maximum
error was `4.4250488e-4`. Inspection found raw logits between approximately
`-269` and `-186`, with much smaller probability differences. A subsequent
probability-only absolute bound of `5e-6` also failed. These bounds were changed
**after observing these development outputs**; the revised bounds are not
predeclared scientific acceptance criteria.

The revised numerical checks use `1e-4 + 1e-5 * abs(reference)` for logits and
MLP outputs, and `1e-5 + 1e-4 * abs(reference_probability)` for probabilities.
They check the entire vocabulary and every captured late-MLP element, not just
top predictions. Raw measurements, artifact checksums, and the tolerance history
are recorded in [`fixtures/pretrained_validation.json`](fixtures/pretrained_validation.json).

| Reference graph | Maximum logit error | Maximum probability error | Maximum late-MLP error |
| --- | ---: | ---: | ---: |
| Standard tanh GELU | 0.000442505 | 0.00000643308 | 0.000335693 |
| Exact GELU | 0.000350952 | 0.0000108315 | 0.000457764 |

The original author's pinned tensor archive has SHA-256
`d66c9efcf7c3edcd1a4344d3e21374309d21bc8a2b617d36bc71bab6d42590eb`.
`compare_author_weights.py` verified all **197 original array entries bitwise
equal** to the HF checkpoint after explicit head and Conv1D layout conversions.
The two declared GELU graphs nevertheless differ: their PyTorch outputs on these
two prompts differ by up to 0.0499725 in logits and 0.000114982 in probabilities.
Changing the activation silently would therefore be incorrect.

Full rust-circuit execution has not been compared. This establishes weight
identity, a documented graph choice, and agreement with an independent PyTorch
implementation. It does not reproduce the Greater-Than circuit or its published
aggregate results; that experiment follows the prior two planned examples.
