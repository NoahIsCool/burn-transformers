# Greater-Than: pinned protocol and source investigation

This investigation supplied the protocol for the now-implemented
[Greater-Than example](../mechanistic/greater_than/README.md). Clock and Pizza
and Tuned Lens passed before the study began. The frozen 194-prompt experiment
has now run in both Rust and independent PyTorch, with all numerical and identity
checks passing. This note preserves the source discrepancies and bounded claim;
the example guide and measured records describe the actual outcomes.

## What to reproduce

Use Hanna, Liu, and Variengien's
[paper version 5, 2 November 2023](https://arxiv.org/html/2305.00586v5).
The bounded target is its year-span behavior and the direct contribution of
late MLPs to that behavior (sections 2 and 3.2–3.3). Layers are **zero-based**:
MLPs 8–11 are the last four of twelve; MLP 10 is the penultimate block.
The paper reports 81.7% mean probability difference and 6.0% cutoff sharpness
on 10,000 prompts, and a 56% performance drop from patching MLP 10's direct
contribution. These are published context, **not thresholds for our smaller
fixture**. Full circuit sufficiency and necessity require additional path
interventions; node replacement alone does not reproduce those results.

Pin [the author repository](https://github.com/hannamw/gpt2-greater-than/tree/0e6dc8c2da87f7d2befeb2a4dd432fae404ea35a)
at `0e6dc8c2da87f7d2befeb2a4dd432fae404ea35a`. Its released scripts do not fix
random seeds or distribute the sampled 490/10,000-example prompt sets. Therefore
our fixed fixture supports a **bounded adaptation of the published protocol**,
not a claim to reproduce the original sampled aggregate exactly.

## Model and tokenizer provenance

The author's actual model is `cache/gelu_12_tied.circ` plus arrays from the
[original tensor archive](https://rrserve.s3.us-west-2.amazonaws.com/remix/remix_tensors.zip?versionId=3llqs2FIJ7iAYSmEON.TR62nOCOODM4U).
The archive was reachable on 10 September 2026: 779,065,832 bytes, S3 version
`3llqs2FIJ7iAYSmEON.TR62nOCOODM4U`, last modified 16 February 2023.
The multipart ETag `58d607fee94e04718be6acaf3034d48a-149` is **not SHA-256**.
The subsequently downloaded archive has SHA-256
`d66c9efcf7c3edcd1a4344d3e21374309d21bc8a2b617d36bc71bab6d42590eb`.
Its `.circ` contains content-addressed array references, including tied
embedding/unembedding `1f668798fb95d0e16b2a0143`.

A convenient separately pinned import candidate is
[openai-community/gpt2 at 607a30d783dfa663caf39e06633721c8d4cfcd7e](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e).
Its architecture is 12 blocks, 12 attention heads, residual width 768, head width
64, MLP width 3072, learned positions, and a tied 50,257-token vocabulary.
The public `models::gpt2` module supplies its stateless graph and local importer.

**Do not assume these two graphs are numerically identical.** The author's
`.circ` specifies `gelu`, whereas the pinned Hugging Face configuration specifies
`gelu_new`. The reviewed
[rust-circuit evaluator](https://github.com/redwoodresearch/rust_circuit_public/blob/c422bd7bd232e33204a94eddaac25c5efd75dcf4/rr_util/rust_circuit_type_utils.py)
distinguishes PyTorch's exact GELU from its tanh approximation. Resolve this
difference explicitly: subsequent comparison found **all 197 author array
entries bitwise identical** to these HF weights after the documented layout
conversions. The source graphs still select different GELUs. On the two
reference prompts, their PyTorch last logits differ by as much as 0.04997 and
their probabilities by 0.000115. Use `Gpt2Activation::Gelu` to reproduce the
author's declared activation; unmodified HF `GeluNew` remains a graph adaptation.
The weight comparison is reproducible with
`src/models/gpt2/compare_author_weights.py`. Full rust-circuit execution has not
been compared; shared mathematical conventions and PyTorch reference execution
are the current validation boundary.

The reviewed
[original tokenizer loader](https://github.com/redwoodresearch/rust_circuit_public/blob/c422bd7bd232e33204a94eddaac25c5efd75dcf4/python/rust_circuit/module_library.py)
calls unpinned `GPT2TokenizerFast.from_pretrained("gpt2")`, adds `[END]`, and sets
it as the pad token. The experimental prompts do not use `[END]` or padding.
Pin the tokenizer files below for our fixture, without adding an automatic BOS
or EOS token; the prompt itself explicitly contains `<|endoftext|>`.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Author `cache/gelu_12_tied.circ` | See pinned source | `0d83b82ac8f40a5770757b3364ee99ad5cc3d32474a38366646ed746f9c83d23` |
| Author `cache/potential_nouns.txt` | See pinned source | `2bdc734b1f4cc84700a7d13ec9da6c9648df94c21eb474ca61d475169178cc13` |
| Author `cache/logit_indices.pt` | See pinned source | `2adf55658bb37a5f2a26ec471acb10ebaba87e1105463aae29e28f25fd100150` |
| HF `model.safetensors` | 548,105,171 | `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` |
| HF `config.json` | 665 | `0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4` |
| HF `tokenizer.json` | 1,355,256 | `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` |
| HF `vocab.json` | 1,042,301 | `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783` |
| HF `merges.txt` | 456,318 | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |
| HF `tokenizer_config.json` | 26 | `5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f` |

All listed hashes, including the full safetensors file, have now been checked
against downloaded bytes. The original model archive and checkpoint remain in
temporary storage, outside the tracked source tree.

## Prompt restrictions and a fixed held-out fixture

Follow the author's
[dataset generator](https://github.com/hannamw/gpt2-greater-than/blob/0e6dc8c2da87f7d2befeb2a4dd432fae404ea35a/dataset.py)
and [year filter](https://github.com/hannamw/gpt2-greater-than/blob/0e6dc8c2da87f7d2befeb2a4dd432fae404ea35a/utils.py).
A year is eligible when its space-prefixed string tokenizes as two tokens with
a two-character second token. Exclude the first and last eligible year of each
century. Use the released 120-noun file in its original order.

There are two important discrepancies to preserve in the documentation:

- The main code regenerates 768 eligible years in `1000..1900`; one paper
  sentence instead specifies centuries 11–17. The committed
  `cache/valid_years.pt` contains 4,680 values and is **not** this experimental
  pool. Regenerate the pool with the documented filter.
- For century 10, natural corruption changes two token identities:
  `1021` becomes `[Ġ10, 21]`, while `1001` becomes `[Ġ100, 1]`.
  A claim that only the suffix token changes is false for that century.

For the readable default, restrict centuries to 11–17: 617 eligible years,
suffixes 02–98. Label this as a deliberate fixture adaptation. Pre-register
97 development and 97 held-out examples, one per suffix in each split. Choose
year and noun independently of model outputs:

1. For each split (`development`, then `held_out`) and suffix `YY=02..98`, hash
   the ASCII string `greater-than-v1/{split}/{YY:02d}` using SHA-256.
2. Interpret digest bytes 0–7 as an unsigned big-endian integer and reduce
   modulo the ascending eligible-year pool for that suffix.
3. Interpret bytes 8–15 the same way and reduce modulo 120 for the noun index.
4. Build clean and corrupted strings exactly as below. Tokenize without
   automatically adding special tokens. Save both token-ID lists in the fixture.

```text
clean:   <|endoftext|> The war lasted from the year 1732 to the year 17
corrupt: <|endoftext|> The war lasted from the year 1701 to the year 17
```

All 194 proposed pairs were tokenizer-checked: 13 tokens, no padding, only
position 8 changes, readout position 12. Their clean strings are disjoint across
splits. The example above has IDs:

```text
clean:   [50256,383,1175,15436,422,262,614,1596,2624,284,262,614,1596]
corrupt: [50256,383,1175,15436,422,262,614,1596, 486,284,262,614,1596]
```

The initial research fixture was saved as `/tmp/greater-than-fixture.json`
(79,395 bytes, SHA-256
`82620130e2483f70ef14bedce36555f3e963fbf38f0be36c163b217a37e30503`).
The identical bytes are now checked in as
[`fixture.json`](../mechanistic/greater_than/fixture.json), beside the frozen
manifest. Tokenization was checked with `tokenizers==0.22.2`;
the 100 output indices also exactly match the released `logit_indices.pt`.

## Metrics and intervention semantics

At the last input position, compute softmax over **all 50,257 logits**. Let
`p[y]` be the resulting probability of the single token spelling `00` through
`99`, without a preceding space. Do not renormalize these 100 probabilities.
For the original clean suffix `YY`, report:

```text
probability_difference = sum(p[YY+1..100]) - sum(p[0..YY+1])
cutoff_sharpness        = p[YY+1] - p[YY-1]
```

Keep that clean `YY` as the target when evaluating corruption or interventions;
evaluating corrupt output against `01` would measure a different task. Compute
metrics per example, then mean and sample standard deviation. Average
probabilities, not logits, when plotting grouped distributions. The author's
[large-dataset evaluation](https://github.com/hannamw/gpt2-greater-than/blob/0e6dc8c2da87f7d2befeb2a4dd432fae404ea35a/big_ds_experiments.py)
uses this order of operations.

The most manageable faithful path measurement is the **direct MLP-to-readout
effect** in the author's
[circuit-discovery procedure](https://github.com/hannamw/gpt2-greater-than/blob/0e6dc8c2da87f7d2befeb2a4dd432fae404ea35a/circuit_discovery.py).
Capture MLP **output after its output projection, before residual addition** at
END for clean and corrupt runs. Capture the clean final residual **before** its
final normalization. For layer `L`, evaluate the readout on:

```text
clean_final_residual - clean_mlp_output[L] + corrupt_mlp_output[L]
```

Apply final LayerNorm and tied vocabulary projection normally. Earlier and later
components retain their clean-run values. This arithmetic implements the single
direct path; injecting corrupt output inside block L instead would additionally
change downstream computations. Verify the arithmetic construction against an
independent reference before claiming faithful path patching. Test all 12 MLPs
on the development fixture, then report the preselected 8–11 and all control
layers on held-out prompts; do not select favorable layers after seeing held-out
results. No further backbone runs are required per layer for this direct path.

Also demonstrate the requested **clean-to-corrupt node replacement** separately:
replace `Layer::decoder(L).mlp_output().token(12)` during a corrupt forward pass
with its paired clean tensor. Label its effect as node patching. It is a
different experiment from the direct-path measurement.

## Controls, claims, and gates

- Establish clean/corrupt output parity against the pinned reference; preserve
  exact token IDs, evaluation mode, F32 dtype, LayerNorm epsilon, activation
  convention, attention scaling/masking, and weight mapping.
- Empty hooks and capture-only execution must preserve logits. Replacing a
  tensor with its capture from the same run is the identity control. Follow an
  ablation with its clean replacement to test restoration.
- Report reverse patching (corrupt into clean), zero ablation, and an off-target
  control layer. The published procedure primarily uses matched `01` input,
  not Gaussian noise or zero ablation; mark the latter controls as additions.
- Report raw clean, corrupt, and patched probability differences and their gap
  before any normalized recovery. For recovery use
  `(patched-corrupt)/(clean-corrupt)` only if `abs(clean-corrupt)>1e-6`; report
  the undefined count. This denominator threshold is our explicit reporting
  convention, not a published scientific threshold.
- Freeze numerical parity tolerances on the development set **before** examining
  held-out behavior. Do not retrofit an expected effect size to observed data.
  Report whether late MLPs actually show the predicted effect, including failures.
- Record expected and measured values separately, per-prompt outcomes, exact
  source/backend revisions, and measured runtime/peak memory. Neither download
  size nor model parameter count establishes a CPU runtime budget.

The completed study reports 8,148 condition/prompt measurements, all controls,
the preselected direct and ordinary-forward node interventions, and measured
runtime. Clean prompts are disjoint across splits, but ten corrupted control
strings overlap; repeated corruptions mean prompt SD is descriptive variation.
The paper's direct-effect magnitude is contextual absolute probability-difference
change, not a normalized recovery target. The implementation reports these
quantities separately and does not claim the full published circuit.
