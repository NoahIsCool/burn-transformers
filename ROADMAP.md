# Transformer Experimentation Roadmap

## Purpose and priorities

Make established transformer mechanisms easy to inspect, change, and combine in
Rust/Burn experiments, with either random initialization or selected pretrained
weights. Editing intermediate activations and inserting learned processing into
the forward computation is the first priority. Researchers must also be able to
change attention rules, layer arrangements, dimensions, and task heads.

Faithful checkpoint inference is the initial correctness threshold and remains
a regression check. Hosting, generation, and quantized execution are useful
consequences of reusable implementations; adding supported model profiles does
not take priority over the experiment interface.

Milestones below define planned outcomes, not release dates. The progress
section distinguishes implemented interfaces from remaining work.
Current checkpoint support remains defined by [SUPPORT.md](SUPPORT.md) and the
existing [release gates](RELEASE_GATES.md).

## Starting point: v0.1

The crate already separates shared mathematics from family wiring, exposes
parameterized Burn modules, supports heterogeneous custom specifications, and
tests gradients through full-precision stateless graphs. Supported Qwen3.5-0.8B
and FLAN-T5-small profiles provide concrete inference evidence.

At that starting point, the main user contracts centered language-model logits
and decode state. `ForwardOptions` could retain layer outputs without selective
intervention. Important block/backbone methods were private; custom specifications
fixed residual width and rejected some implemented mechanisms, including MLA.
The generic encoder-decoder specification was not a faithful editable T5 recipe.
Training workflows were intentionally deferred.

## v0.2: establish the experiment interface

The full target is specified in [V0_2_INTERFACE.md](V0_2_INTERFACE.md). The
activation interface alone is not v0.2 completion. The agreed paper progression
also includes Clock and Pizza, Tuned Lens, and Greater-Than, with the bounded
protocols in [examples/planned/README.md](examples/planned/README.md).

### Implementation progress

The experiment interfaces are implemented and documented in
[EXPERIMENTS.md](EXPERIMENTS.md):

- Typed layer/operation addresses with token, head, channel, and batch selection;
  detached or differentiable captures, lightweight observers, ordered edits,
  clean-run replacement, validation, and custom typed computation points.
- Stateless hooks on custom causal/seq2seq, Qwen3.5/3.8, T5, and shared attention;
  discoverable catalogs identify the points each execution path supports.
  Every family now implements the generic `HookedModel` trait, and every
  causal family also implements `StagedModel` (see
  [after v0.3](#progress-after-v03)).
- Attention Q/K/V and head contexts retain native dispatch. An explicit eager
  precision choice enables score/pattern edits while preserving exclusions.
- Ordinary Burn modules can be borrowed by edits and registered in an owning
  `Module`; tiny autodiff and record tests validate the parameter boundary.
- Custom full-attention models offer causal/bidirectional execution and a
  hidden-only path. Qwen exposes individual blocks and its text backbone.
- A readable [activation workbench](examples/activation_workbench.rs) exercises
  the API without downloads; the grokking reproduction now uses public hooks.
- Explicit embed/run/normalize stages for custom causal/seq2seq, T5 encoder, and
  Qwen; positions, padding, and normalization status travel with the hidden tensor.
- Independent self/cross-attention policies, per-layer causal/bidirectional/custom
  connectivity, additive bias, and structural attention-pattern transformations.
- Editable T5/Qwen recipes, streaming Qwen prefix loading, and transactional
  parameter transfer with shape, transformation, and parameter-tie reports.
- Open Rust sequence mixers and feed-forward modules, optional normalization,
  and registered residual-width adapters; shared MLA works in custom specifications
  and the [MLA encoder-decoder example](examples/mla_encoder_decoder.rs).
- T5 padded decoder interventions retain rectangular token indices and enforce
  validity before observation and after edits.
- Real-checkpoint [Qwen prefix extraction](acceptance/measurements/qwen-prefix-v02-20260910.json)
  loads only the selected tensors and exactly matches the same normalized prefix
  from the full model. [FLAN-T5 encoder extraction and staged hooks](acceptance/measurements/v0_2-flan-t5-small-flex-20260910/manifest.json)
  exactly preserve the complete encoder output; the existing Flex reference
  gate also passes. These are bounded correctness checks, not new support claims.
- [Clock and Pizza](examples/mechanistic/clock_and_pizza/README.md) reproduces all
  six Table 2 measurements and both distance diagnostics using released weights.
  Its controls and one unresolved published accuracy discrepancy are recorded.
- [Tuned Lens](examples/mechanistic/tuned_lens/README.md) uses the exact released
  Pythia/lens pair on a pinned four-record fixture. All 12 translators improve
  KL and cross-entropy; independent replay passes the original error limits.
  Diagnosed numerical corrections and earlier failed comparisons are retained.
- [Greater-Than](examples/mechanistic/greater_than/README.md) compares 42 conditions
  on 194 pinned prompt pairs against an independent GPT-2 reference. Late-layer
  patches restore much of the greater-than preference; identity and restoration
  controls preserve logits exactly. Direct-path effects and deviations from the
  original sampling protocol are reported separately.

The historical [initial implementation replay](examples/mechanistic/v0_2_examples.result.json)
records 1,250 passing Rust tests, one doctest, all-feature Clippy, and three
successful examples at its recorded source snapshot. It predates the broader
composition/recipe changes above and is not current-source qualification.

The **v0.2 implementation and bounded experiment acceptance are complete**.
The [final validation record](acceptance/measurements/v0_2-interface-validation-20260910-retry/README.md)
passes all 29 steps with an unchanged input inventory: 1,368 Rust tests,
3 doctests, 371 Python tests, all-feature Clippy, formatting, documentation,
feature/MSRV checks, and all ten examples. It records 32 opt-in Rust ignores
and 14 Python skips separately. The rejected earlier inventory attempt is
preserved; its only change was a newly generated plot-validation report.

This is an unreleased implementation milestone. New coverage, hardware, and
package qualification remain required before extending release claims; the
package version remains 0.1.0. Cached interventions need a separate state
contract. The v0.3 training implementation is described below.

### 1. Select, observe, and modify intermediate tensors

Start with useful, explicitly defined computation points: embeddings; residual
streams before and after sublayers; attention projections, scores, probabilities,
and head outputs; feed-forward pre/post-activation tensors and outputs; and final
normalization. Identify stack, layer, operation,
and tensor role without requiring users to know checkpoint parameter names.
Document each point's rank, axes, dtype, masking, and numerical meaning.
Distinguish scores before and after masking. Let custom modules expose additional
typed intervention points instead of limiting experiments to a closed registry.

Provide convenient layer/point selection and token/head/channel selection where
the selected tensor actually has those axes. Support three distinct operations:

- Observe selected tensors, retaining or reducing only what is requested.
- Transform or replace selected activations during execution, including
  ablation, clean-to-corrupted patching, and activation steering.
- Insert ordinary parameterized Burn modules into the computation. Their
  parameters must participate in module traversal, saving, freezing, and later
  optimization; a closure holding unregistered parameters is insufficient.

Define deterministic callback ordering, replacement shape/device/dtype checks,
error behavior, and whether retained activations preserve gradients or detach.
Make interventions scoped to an explicit forward call. Do not implicitly copy
tensors to the host, capture every activation, or materialize attention matrices.
When an execution kernel cannot expose a requested internal point, use a
documented reference path or return a capability error.

**Exit criteria:** public-API examples can select a layer/token, inspect its
activation, patch it from another run, ablate a head, and insert a small module.
No observer and identity interventions preserve outputs and gradients within
declared tolerances. Invalid replacements return contextual errors. Tiny
autodiff tests verify inserted-module gradient flow before full training support.

### 2. Expose execution independently of language-model heads

Make embeddings, blocks, stacks, normalization, and task heads independently
usable. Support executing a block range and resuming from hidden tensors with
explicit positions, masks, encoder context, and normalization boundaries.
Researchers should be able to use a pretrained prefix followed by their own
blocks and classifier without editing library internals.

Keep stateless intervention the first supported execution path. For cached or
recurrent execution, specify which interventions affect existing state and when
recomputation is required. Reject unsupported combinations before mutation;
changing an earlier activation cannot silently reuse its old dependent cache.

**Exit criteria:** splitting and resuming an unchanged stack matches a complete
forward pass; a prefix plus replacement head runs through public APIs; supported
cached identity paths retain parity, and unsupported edits fail explicitly.

### 3. Make attention behavior a composable choice

Causality belongs to the experiment's attention policy, not its model-family
name. Existing T5 encoders are already bidirectional, and shared attention
already accepts noncausal execution. Expose causal, bidirectional, and explicit
custom connectivity at the stack/layer interface, separately from padding and
additive attention bias. Document Boolean conventions at each boundary.

Allow researchers to replace the attention operation or other sublayers with
Rust/Burn modules when changing the fundamental computation, such as attention
normalization, score construction, or sequence mixing. A fixed enum of existing
mechanisms must not be the only extension path. Full-sequence bidirectional
execution should support diffusion-style experimentation without claiming that
an autoregressive KV cache is valid for it.

**Exit criteria:** the same small architecture runs causally and bidirectionally;
changing a future token affects an earlier output only when permitted. Custom
mask tests cover padding and fully excluded rows. A custom sublayer works from
an example outside `src/`.

### 4. Provide faithful editable recipes and selective initialization

Offer established architectures as editable recipes that preserve their exact
scaling, normalization, positional conventions, parameter sharing, and ordering
until deliberately changed. Begin with T5 and the validated Qwen profile.
Separate architecture construction from random or pretrained initialization.

Permit per-layer head counts, feed-forward widths, block counts, and replacement
modules. Distinguish attention projection, feed-forward, residual, and output
widths. Residual-width changes need explicit adapters and validated connections;
checkpoint shape changes need an explicit transformation or reinitialization
policy. Report parameters reused, transformed, omitted, or newly initialized.

Move reusable MLA execution behind a model-independent interface and demonstrate
its use in an encoder-decoder. Preserve compatible parameter sharing and make
intentional untying explicit when extracting or replacing pretrained components.

**Exit criteria:** unmodified recipes retain existing parity; a T5-derived random
encoder accepts heterogeneous layers and a classifier head; a Qwen prefix loads
with an auditable transfer report; a tiny MLA encoder-decoder executes without
family-specific branches in shared mathematics.

## Experiments are an acceptance surface

**Grokking is the lead experiment for interface design.** Its workflow—capture
post-ReLU activations, project onto or remove a learned subspace, then continue
the model—must be concise and readable through the public v0.2 API. The smaller
attention-head example introduces selection, ablation, and restoration.

The first selected targets are:

| Source | Bounded experiment and acceptance criterion | Claim boundary |
| --- | --- | --- |
| Elhage et al., [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (2021) | Reconstruct a tiny attention-only model's logits from its additive head/OV contributions; measure numerical error, zero-ablation effect, and clean-patch restoration. | Mathematical reproduction with fixed weights; not evidence that a trained model learns a particular circuit. Causal/bidirectional future-token sensitivity is an additional API control, not a paper result. |
| Nanda et al., [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217) (2023) | Reproduce Table 1's ten activation/Fourier variance measurements on the published modular-addition checkpoint. Separately compare performance after preserving, removing, and restoring the corresponding latent subspace. | Table 1 is the quantitative paper target. Full-graph callback ablations have their own measurements and do not claim the paper's exact nullspace loss. Training dynamics belong to v0.3. |

The [reference examples](examples/README.md) record measured results separately
from these release criteria. Grokking and the attention-head identity now use
public `Hooks` at custom typed points in small, explicit mathematical models.
The workbench demonstrates those hooks on a built-in modular model.
Every result must distinguish expected and observed values, identify
the exact checkpoint and protocol where applicable, and state incomplete steps.

Keep small, human-readable examples in [examples/](examples/) with the paper's
primary-source citation, exact claim tested, deterministic data/seed, command,
metric, expected behavior, negative controls, and measured resource use. Prefer
CPU-sized experiments without checkpoint downloads for routine checks.

Distinguish a paper-result reproduction using its specified trained model and
protocol from a reduced reproduction and a constructed demonstration of its
mechanism. Hand-set weights cannot establish that training discovers a circuit;
internal parity cannot establish agreement with a paper. Record observed results
and omitted conditions instead of treating successful execution as reproduction.

These examples should drive interface decisions: their experiment logic must be
visible in ordinary Rust, use public APIs, and avoid a large helper framework.
Expand existing source checks with the examples' numerical assertions as each
experiment is completed. Keep checkpoint parity and experiment acceptance as
separate evidence.

## v0.3: train and adapt the experiments

Make every experiment work naturally with Burn's existing training facilities.
The crate owns transformer computation and activation access; Burn or the
consumer owns the training loop, optimizer, batching, schedule, and persistence.
No parallel trainer framework is needed. The consumer contract is in
[TRAINING.md](TRAINING.md).

### Implemented surface

- Ordinary Burn modules and optimizer loops train edited T5 encoders, custom
  sequence mixers and width adapters, MLA encoder-decoders, and inserted modules
  in frozen Qwen backbones. Small tasks measure learning and checkpoint reloads.
- An optional `train` feature and [Burn Learner example](examples/burn_training.rs)
  exercise native `TrainStep`, `InferenceStep`, batching, dropout, and checkpoints.
- Connected captures support intermediate losses. Frozen floating-point suffixes
  propagate input gradients; native model/optimizer records reproduce the next
  deterministic update. A preflight detects duplicate trainable parameter owners.
- Pairwise differentiable losses address large-vocabulary F32 reduction error.
  Packed autodiff is rejected explicitly; dequantized training and detached
  inference features have separate documented contracts.
- [Grokking training](examples/mechanistic/grokking_training/README.md) runs the
  full 40,000-update modular-addition protocol before analyzing learned Fourier
  components. The fixed Metal/F32 run reached 100% held-out accuracy after a
  prolonged train/test gap; preserving/removing/restoring the learned subspace
  measured 100%/9.60%/100% full-domain accuracy.
- [Clock and Pizza](examples/mechanistic/clock_and_pizza_training/README.md)
  trains both attention settings; [Tuned Lens](examples/mechanistic/tuned_lens_training/README.md)
  learns fresh translators with a frozen pretrained readout; [Greater-Than](examples/mechanistic/greater_than_training/README.md)
  adds a conditioned learned-replacement extension. Each retains its original
  checkpoint analysis and reports the new study's actual scientific scope.

### Measured training outcomes

| Experiment | Bounded observed result |
| --- | --- |
| Grokking | Full 40,000-update run shows delayed held-out generalization; final train/test accuracy is 100%, with causal learned-subspace ablation controls. |
| T5 classifier, custom mixer, MLA encoder-decoder | All three learn their fixed synthetic held-out tasks, with nonzero gradients, changed parameters, and exact model-record reloads. |
| Frozen Qwen adapter | Random and real pretrained prefixes permit task and auxiliary gradients through frozen blocks, preserve all backbone bytes, and resume Adam exactly; training-set fixture only. |
| Native Burn Learner | Learns the held-out symbolic task with explicit dropout, native metrics/checkpoints, deterministic evaluation, and exact checkpoint reload. |
| Tuned Lens | Three fresh translators improve held-out KL and next-token loss; independent first-gradient/update comparisons and native momentum resume pass. |
| Greater-Than extension | Held-out distillation KL falls from 0.5622 to 0.0500 nats; frozen weights and resumed update remain exact. This is a new conditioned replacement study. |
| Clock/Pizza | Both fixed 20,000-update conditions reach 100% held-out accuracy and show different gradient symmetry, but neither meets the paper's strict circularity threshold. Representative mechanism retraining is **not reproduced** by these seeds. |

The **v0.3 training compatibility implementation is validated**. The
[final source record](acceptance/measurements/v0_3-training-validation-20260910/README.md)
passes all 30 checks: 1,400 Rust tests and three doctests on both the development
compiler and Rust 1.92, 382 Python tests, all-feature Clippy, formatting,
documentation, minimal-feature checks, and eleven inexpensive example runs.
It separately records 32 opt-in Rust ignores, 14 Python skips, and 29 tests of
the coverage-layout tool on each compiler. Earlier failed test assumptions and
sandbox-only failures remain in the record. Long scientific training runs have
their own executed-source, binary, protocol, and artifact evidence linked above.

### Acceptance requirements

- Document native training/evaluation mode, explicit dropout, initialization,
  reproducible seeds, loss composition, freezing, and parameter ownership.
- Validate probability and loss reductions on large vocabularies. The Tuned Lens
  study exposed scalar F32 tail-mass loss; training metrics must have explicit
  numerical checks across backends as well as correct mathematical definitions.
- Extend the grokking example to training modular addition, recording held-out
  loss and Fourier progress measures across checkpoints. Keep checkpoint-level
  intervention replication separate from replication of training dynamics;
  specify seeds, data split, optimizer, regularization, and compute budget.
- Train a randomly initialized T5-derived sentiment classifier; support changed
  per-layer dimensions, pooling, and classification heads independently of an LM
  vocabulary projection.
- Train a small module inserted into a frozen pretrained backbone, then save and
  reload the complete experiment with equivalent outputs.
- Train a tiny MLA encoder-decoder and exercise full-sequence noncausal inputs
  with an explicit task objective and corruption policy.

**Exit criteria:** small deterministic tasks show optimizer updates and learning
against declared baselines; frozen parameters remain unchanged; inserted modules
receive gradients; resumed training restores the required state. Documentation
identifies backend, precision, and quantization restrictions. Passing a backward
pass alone is not training acceptance. Paper protocols are fixed before results;
short smoke runs do not establish grokking or reproduce population-level claims.
Measured training evidence and final source checks are recorded separately from
the historical v0.2 gates. Coverage, package, and hardware qualification remain
release work; this implementation does not promote checkpoint support.

## Progress after v0.3

A polish cycle broadened and hardened the implementation without changing the
milestones above or the advertised checkpoint support:

- **One experiment interface for every model.** `HookedModel` is implemented
  by every family and `StagedModel` by every causal family (including
  DeepSeek-V2, Muse Glimmer, Nemotron-H, Qwen3-MoE, GPT-2, GPT-NeoX, and the
  modular models); T5 and modular seq2seq models stage through their encoder
  and decoder views. Both are dispatched by `AutoCausalLm` (`AutoSeq2SeqLm`
  dispatches hooks) and forwarded by `LoadedModel`. The workbench runs one generic experiment on three families.
- **New families.** The dense-decoder Llama family (`llama`, `mistral`,
  `qwen2`, `qwen3`, `smollm3`, `olmo2`), Gemma 2/3 text, and gpt-oss, each with
  tiny Transformers goldens and Flex/F32 real-checkpoint gates. Their
  Metal and F16 acceptance remains open.
- **Shared mechanisms.** Connectivity masks, sinks, soft caps, differential
  attention, every RoPE scaling rule, one router for softmax/sigmoid and
  grouped routing, clamped SwiGLU, a shared Mamba-2 mixer, MXFP4 decoding, and
  explicit initialization policies, all usable from modular specs.
- **Batched cached execution.** Every family decodes ragged batches in one
  pass per step through atomic cache step writes.
- **Verification.** 18 Verus harnesses and twelve fuzz targets.

Open work carried forward: cached-state interventions, T5 staged views on the
full encoder-decoder model, registering GPT-2 and GPT-NeoX in `AutoCausalLm`
(they need external decode state), Metal and half-precision acceptance of the
new families, and the release gates in [RELEASE_GATES.md](RELEASE_GATES.md).

## Scope boundaries

Do not require a universal graph DSL, exhaustive access to every temporary
tensor, a new trainer framework, a serving platform, or larger checkpoint
qualification to complete v0.2. Prefer a small stable set of typed computation
points and ordinary Rust composition, expanding it when a concrete experiment
needs more access. Quantization remains a supported research direction where
implemented, with explicit execution and gradient capabilities; trainable
quantized kernels are not assumed.
