# v0.2 Interface and Acceptance Contract

This preserves the v0.2 scope. The subsequent [v0.3 training contract](TRAINING.md)
and [training examples](examples/README.md) now extend these interfaces through
native Burn optimization, freezing, and checkpoints. The generic
`architecture::HookedModel` and `StagedModel` traits later made the hook and
staged interfaces below uniform across every model family; see
[EXPERIMENTS.md](EXPERIMENTS.md#one-experiment-any-model).

## Consumer workflow

The library should let a researcher write the experiment directly: construct or
load an architecture, select an intermediate tensor, transform it, and continue
the computation. Parameters are ordinary Burn modules. Execution contexts,
captures, and caches are separate from the parameter tree.

The intended staged interface is consistent across custom, T5, and Qwen backbones:

```rust
let mut state = model.embed(input)?;
state.context = state.context.with_attention(AttentionPolicy::bidirectional());
let state = model.run_layers(state, 0..2)?;
// Inspect or process state.hidden here, using an owned Burn module.
let state = model.run_layers(state, 2..layer_count)?;
let state = model.normalize(state)?;
let predictions = task_head.forward(state.hidden);
```

This is an interface contract; concrete model ownership signatures are documented
with each implementation. `_with_hooks` variants accept the same `Hooks` used by
ordinary forward calls. State records the next layer and whether final
normalization has occurred; skipped or repeated ranges and double normalization
are errors. Positions and padding accompany the tensor when execution resumes.
Explicitly normalizing a prefix is permitted for a new task head.

## Activation intervention

Use `Layer::encoder(i)` / `decoder(i)` and named computation points. Selections
preserve rank and declare token, query, key, head, batch, and channel semantics.
The consumer should not need checkpoint parameter names to inspect a tensor.

- Capture only requested values; detach by default, retain gradients explicitly.
- Observe through a callback without retaining a full activation cache.
- Edit or replace the selected tensor in registration order, including patches
  from another run and learned processing owned by a registered `Module`.
- Validate availability before callbacks, and validate replacement shape, dtype,
  and device before downstream use. Failed calls invalidate captures.
- Allow custom typed points and custom Rust modules, with one explicit outer
  hook scope. Do not require a graph language or closed list of operations.

## Architecture and attention changes

`ExecutionContext` holds positions, token validity, optional source context, and
attention overrides. Self- and cross-attention policies are distinct. Per-layer
overrides take precedence over a pass default; absent overrides retain the
architecture's exact conventions. Custom connectivity uses **true = allowed**;
primitive exclusion masks use **true = excluded**. Padding remains independent.

Changing attention normalization or mixing must be part of the module's ordinary
computation, visible to subsequent consumer hooks. The Clock and Pizza mixture
`alpha * softmax(scores) + (1 - alpha) * ones` is a structural example. Eager
score/pattern access is explicit; it must not silently change native arithmetic.

Open `HiddenTransform` modules, residual blocks, and registered width adapters
support arbitrary processing between blocks. Editable T5 and Qwen recipes retain
their family conventions until changed and permit per-layer heads, feed-forward
widths, and layer counts. MLA belongs to a reusable module and must work in an
encoder-decoder without a DeepSeek-specific branch in shared mathematics.

## Initialization and state

Random construction and pretrained transfer are separate operations. Transfer
must report reused, transformed, omitted, and newly initialized parameters.
Shape changes require an explicit transformation or reinitialization policy.
Preserve compatible parameter ties; deliberate untying must be visible.
Rejected transfers cannot partially mutate the user's target module.

Stateless composition is the v0.2 path. Ordinary cached inference retains its
existing contract. Unsupported interventions or changed architectures must not
silently reuse incompatible cache state. Training backbones, fitting lenses,
and optimizing adapters remain v0.3 work.

## Completion evidence

Completion requires current-source evidence for every item below, not only a
working hook demonstration. The interface audit and
[final validation](acceptance/measurements/v0_2-interface-validation-20260910-retry/README.md)
complete this implementation contract. `ROADMAP.md` and the example protocols
record the measured scopes and remaining release/training work.

| Requirement | Evidence required |
| --- | --- |
| Activation selection and inserted modules | Public examples plus output/gradient parity, ordered patching, malformed-input errors, parameter traversal and record round trips. |
| Split/resume and task heads | Unchanged full/split parity for custom, T5, Qwen; pretrained prefix plus replacement head; invalid boundary tests. |
| Fundamental attention changes | Causal/bidirectional/custom connectivity tests, padding and fully excluded rows, structural pattern transforms, an external custom sublayer. |
| Editable recipes and transfer | Faithful baseline parity, heterogeneous T5 classifier, selective Qwen prefix loading/report, shape-change and tie tests. |
| Reusable MLA | Tiny encoder-decoder execution, latent interventions, cross-attention with different sequence lengths, and unchanged DeepSeek regression tests. |
| Existing paper examples | Grokking Table 1 and head-additivity numerical assertions reproduced through public APIs. |
| Clock and Pizza | Pinned representative checkpoints, author-reference parity, published geometric/diagnostic measurements, circle plot, isolation/restoration controls, documented discrepancies. |
| Tuned Lens | Exact released model/lens pairing, reference translator parity at correct residual boundaries, held-out layerwise KL/CE and readable prediction visualization. |
| Greater-Than | Faithful model/tokenizer, pinned aligned clean/corrupt fixture, original-vocabulary probability difference and sharpness, patch/ablation/restoration measurements and plots. |
| Human usability | Short readable Rust experiment bodies, documented setup and tensor meaning, obvious experiment choices, no private hook framework. |
| Validation | Tests for all new behavior, warning-free lint/docs, feature checks, current-source replay records, and applicable release gates before release claims. |

Complete Clock and Pizza before the Tuned Lens study, then Greater-Than; their
independent model/import prerequisites may be developed in parallel. The later
Othello, activation-addition, and RAVEL candidates in the planned examples guide
are explicitly follow-on work, not substitutes for this progression.
