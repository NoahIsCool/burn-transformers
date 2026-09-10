# Planned Mechanistic Interpretability Examples

This is the implementation brief agreed in September 2026. It preserves the
original requirements; current implementations and measured claims live in
[the examples index](../README.md). Clock and Pizza, Tuned Lens, and Greater-Than
are implemented with measured, bounded claims. Later candidates remain plans.
The subsequent [training guide](../../TRAINING.md) and current index supersede
this brief's deferrals of Grokking training and new Tuned Lens translators.

The goal is human ease of use: a reader should understand the research question,
find the intermediate tensor being examined or changed, and modify the experiment
without learning a separate framework. These examples should test the library's
ability to inspect, intervene in, and compose transformers. Inference parity is
the prerequisite; a clear, measurable experiment is the outcome.

## Implementation progression

Complete the first three in this order, building on the existing grokking example:

| Order | Example | Main capability exercised |
| --- | --- | --- |
| 1 | Clock and Pizza | Fundamental attention changes and interventions on learned representations |
| 2 | Tuned Lens | Applying small auxiliary models to intermediate activations |
| 3 | Greater-Than | Comparing clean, corrupted, and patched executions |

Othello-GPT is the recommended next visual teaching example. Activation-patching
methodology should inform every relevant example from the beginning; a standalone
comparison can follow. Activation Addition is a later candidate. RAVEL belongs
in the v0.3 training progression. These later candidates are not prerequisites
for completing the first three.

## Shared implementation requirements

- **Keep the experiment readable.** Use an ordinary Rust example with a short
  `main()` or `run()`: load, establish a baseline, select a tensor, inspect or
  modify it, and report the effect. Put artifact conversion and model plumbing
  in small helpers. Explain tensor axes and intervention locations in plain text.
- **Exercise public interfaces.** Use the documented activation API and normal
  model execution. A missing capability should lead to a reusable library
  improvement, not an example-specific private hook. Model-family wiring should
  remain separate from shared transformer mathematics.
- **Make experiments easy to change.** Put layer, token, head, subspace, or
  intervention-strength choices near the experiment. Show one useful variation
  and explain its expected meaning. Prefer explicit code over a configuration
  framework or a large command-line interface.
- **Bound the default workload.** Prefer released checkpoints, fixed small
  datasets, and CPU execution where practical. Separate preparation from local
  execution. Document downloads, memory, runtime, and optional larger runs;
  measure these before advertising a budget. Do not make a training run or
  broad hyperparameter sweep the default.
- **Preserve provenance.** Pin the paper version, checkpoint revision, tokenizer,
  dataset or prompt fixture, and conversion procedure. Record artifact checksums,
  seeds, backend, dtype, sample counts, and code revision with results. Keep
  weights and generated artifacts in ignored storage.
- **Separate reproduction from adaptation.** Name the exact published result
  being tested. Establish the reference protocol and numerical tolerance before
  judging the output. Different checkpoints, smaller datasets, or simplified
  interventions must be labeled as adaptations or partial reproductions.
- **Use meaningful controls.** Check baseline parity, identity interventions,
  and restoration where applicable. Add matched random or reversed interventions
  when they address the scientific question. Report failures and off-target
  effects, not only favorable prompts or plots.

Checkpoint analysis belongs in v0.2. Training backbones, fitting probes or lenses,
and optimizing learned intervention modules belong in v0.3. An existing
checkpoint does not imply its architecture, importer, or required activation
interface is already supported. Identify those dependencies before implementation.

## 1. Clock and Pizza

**Read:** Zhong et al., [The Clock and the Pizza: Two Stories in Mechanistic
Explanation of Neural Networks](https://arxiv.org/abs/2306.17844), NeurIPS 2023;
[author code and representative checkpoints](https://github.com/fjzzq2002/pizza).

**Question:** How can transformers solve the same modular-addition task using
different internal algorithms, and what changes when we isolate parts of their
learned representations?

**Bounded implementation:**

1. Import the released representative models and match their reference outputs.
   Evaluate addition modulo 59 over all 3,481 input pairs.
2. Implement their actual attention behavior. The paper's architecture uses
   unmasked attention and mixes the post-softmax matrix with an all-ones matrix:
   `A' = alpha * A + (1 - alpha) * J`. Here `J` is not a normalized uniform
   probability matrix. Preserve this distinction explicitly in the recipe.
3. Inspect the learned token embeddings and identify their circular components
   using the paper's PCA procedure. Isolate the relevant embedding components,
   rerun the model, and compare logits and accuracy with the baseline.
4. Reproduce a bounded published measurement, such as the circle-isolation
   results in section 3.3, alongside the paper's distance-irrelevance diagnostic.
   Match checkpoint and protocol before assigning numerical pass criteria.

**Deliver:** a small embedding-circle plot, an accuracy/intervention table, and
an explanation connecting the geometry to the model's computation. Include
identity and restoration controls. Distinguish embedding interventions from
changes to positional embeddings or later residual activations.

**Why first:** this extends grokking while exercising a fundamental change to
attention. Checkpoint analysis is small; the paper's training sweeps are outside
the default scope. Retraining architectures with different attention rules is a
v0.3 extension. Changing attention in an already trained model does not by itself
reproduce how a different algorithm emerges during training.

## 2. Tuned Lens

**Read:** Belrose et al., [Eliciting Latent Predictions from Transformers with
the Tuned Lens](https://arxiv.org/abs/2303.08112);
[author implementation](https://github.com/AlignmentResearch/tuned-lens) and
[pretrained-lens loading guide](https://github.com/AlignmentResearch/tuned-lens/blob/main/docs/source/tutorials/loading_pretrained_lenses.rst).

**Question:** What predictions can a small learned translator recover from each
layer's intermediate state?

**Bounded implementation:**

1. Select a small released model with a matching pretrained lens. The author's
   loading guide uses `EleutherAI/pythia-160m-deduped-v0`; verify the exact paired
   artifacts before committing to this fixture.
2. Add or verify the faithful Pythia/GPT-NeoX recipe and model/lens import mappings.
   Capture residual states at the boundaries expected by those lens parameters.
3. Apply each layer's pretrained affine translator as a normal auxiliary module,
   then use the correct final normalization and vocabulary readout. Compare
   against the raw logit lens and the model's final prediction.
4. Evaluate a pinned held-out text fixture. Report layerwise KL divergence from
   the final distribution and next-token cross-entropy, with clear definitions.

**Deliver:** a readable token-by-layer prediction table or heatmap, aggregate
metrics, and a short example replacing a translator with another module. Verify
that observing states leaves the backbone output unchanged and that imported
lens outputs match their reference implementation.

**Claim boundary:** a compact evaluation demonstrates the released lens on the
chosen fixture; it does not reproduce every model and dataset in the paper.
Predictive decoding is not proof that a layer causally uses that prediction.
Fitting new translators and comparing training objectives are v0.3 extensions.

## 3. Greater-Than

**Read:** Hanna et al., [How does GPT-2 compute greater-than?: Interpreting
mathematical abilities in a pre-trained language model](https://arxiv.org/abs/2305.00586),
NeurIPS 2023; [author code](https://github.com/hannamw/gpt2-greater-than).

**Question:** Which intermediate computations make GPT-2 favor years greater
than a supplied starting year?

**Bounded implementation:**

1. Verify GPT-2 small checkpoint and tokenizer parity. Build a deterministic
   fixture following the paper's tokenization and prompt restrictions, for
   example: `The war lasted from the year 1732 to the year 17`.
2. Plot the probability of valid two-digit completions. Implement the paper's
   probability-difference metric: probability mass above the starting suffix
   minus mass at or below it. Use original vocabulary probabilities rather than
   silently renormalizing the two-digit subset. Also report cutoff sharpness.
3. Run matched clean and corrupted prompts. Patch selected MLP activations from
   the clean run into the corrupted run; begin with the reported late MLPs
   (layers 8–11, with the indexing convention documented).
4. Compare baseline, corruption, patching, ablation, and restoration. Use held-out
   prompts for conclusions and report per-prompt variation as well as averages.

**Deliver:** a year-probability curve and a compact layer/intervention table. Make
the source run, destination run, token alignment, and selected tensor apparent
in the Rust code. Report the clean-to-corrupted gap before normalized recovery;
handle negligible denominators explicitly.

**Claim boundary:** the initial example is a bounded activation-patching study.
The paper's circuit analysis uses path patching; ordinary node replacement does
not reproduce that full circuit. Add faithful path interventions separately if
claiming circuit reproduction. Published aggregate numbers from the full dataset
are not pass thresholds for a reduced prompt fixture.

## Following examples

### Othello-GPT: inspect a board, then change its representation

**Read:** Li et al., [Emergent World Representations: Exploring a Sequence Model
Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382), ICLR 2023;
[author code and artifact links](https://github.com/likenneth/othello_world).

Use the released transformer and matching nonlinear board-state probe to display
the true board beside the decoded board for a legal game prefix. Follow the
paper's latent intervention procedure to change one square's representation,
resume computation, and measure how the predicted moves respond.

Report probe accuracy, effects on moves legal under the changed board, and
preservation of untargeted squares. Include no-op, restoration, and matched
control interventions. A successful probe alone is not the causal result.

First verify artifact availability and import parity; linked downloads have not
been validated here. The intervention may require gradients with respect to
activations, so audit that dependency separately from parameter training. Keep
backbone/probe training out of the default run. This is the strongest candidate
for a visual teaching example after the first three.

### Activation patching: compare methods and metrics

**Read:** Zhang and Nanda, [Towards Best Practices of Activation Patching in
Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042).

Apply its methodological lessons throughout the examples. A later standalone
experiment should compare semantic token-replacement corruption with Gaussian
embedding noise, and probability-based measurements with logit differences.
Show whether those choices change which components appear important.

Reuse a small existing fixture where possible. Label that reuse a methodological
demonstration unless it matches a published subexperiment. Document alignment,
corruption strength, baseline gaps, and possible out-of-distribution behavior;
avoid interpreting one ranking as a universally correct circuit.

### Activation Addition: steer with a contrastive activation vector

**Read:** Turner et al., [Activation Addition: Steering Language Models Without
Optimization](https://arxiv.org/abs/2308.10248v1). Pin this original paper version
when choosing its protocol.

Capture activations from contrasting prompts, subtract them, and inject a scaled
direction at the specified layer and token positions. Sweep a few coefficients
on held-out prompts and report the targeted behavioral effect together with
off-target language-model likelihood. Include zero, reversed, and norm-matched
random directions.

The original work used GPT-2 XL and did not establish the same clear behavior in
GPT-2 small or medium. Budget accordingly. Using an already supported smaller
model is a useful adaptation, but needs its own evidence and must not be called
an exact reproduction. Do not evaluate only handpicked generated text.

### RAVEL: change one attribute while preserving others — v0.3

**Read:** Huang et al., [RAVEL: Evaluating Interpretability Methods on
Disentangling Language Model Representations](https://aclanthology.org/2024.acl-long.470/),
ACL 2024; [benchmark repository](https://github.com/explanare/ravel).

Use its evaluation pattern for learned intervention modules: change one entity
attribute while preserving other attributes. Score both intervention success and
isolation of the change. Keep fitting, calibration, and evaluation data separate.

The main paper uses Llama-2-7B; a reduced smaller-model version is an adaptation
requiring independent validation. This is a training-stage stretch goal, not a
low-compute v0.2 requirement.

## Definition of done and handoff

For each implemented example, provide:

1. A readable Rust entry point and only the supporting helpers it needs.
2. Exact preparation and local execution commands, with artifact sizes and
   measured default runtime/memory. An analysis run should not download files.
3. A protocol note linking the paper and author code, identifying the reproduced
   measurement, controls, tolerances, deviations, and remaining limitations.
4. Machine-readable measured results plus a small human-readable table or plot.
   Record expected and observed values separately; never populate results with
   published values in place of measurements.
5. Focused checks for import parity and the scientific invariants used by the
   example, plus relevant formatting and library checks when code changes.
6. An entry in the examples index and any necessary public-interface guide
   updates. Claim only the capabilities and results actually validated.

Suggested instruction for the main implementation chat:

> Implement the examples in `examples/planned/README.md`, starting with Clock and
> Pizza, then Tuned Lens, then Greater-Than. Complete and validate each bounded
> example before proceeding to the next. Follow the shared requirements and
> definition of done, prioritize readable consumer code, and label adaptations
> honestly. Keep training work for v0.3 and treat the later candidates as follow-on
> work after this initial progression.
