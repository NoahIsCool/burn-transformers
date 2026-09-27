# Train transformer experiments with Burn

An experiment is an ordinary Burn `Module`. Its backbone, inserted processing,
and task head use Burn's autodiff, optimizers, records, and data loaders. The
crate supplies transformer computation and activation access; the training
procedure stays in your application. These workflows use pinned Burn 0.21.

## Start with a working example

All four commands below train small models locally without downloads:

```sh
cargo run --release --locked --features autodiff --example t5_classifier -- --train
cargo run --release --locked --features autodiff --example custom_composition -- --train
cargo run --release --locked --features autodiff --example mla_encoder_decoder -- --train
cargo run --release --locked --features train --example burn_training
```

The first three show explicit optimizer loops, held-out task scores, and record
round trips. The last uses Burn's `TrainStep`, `InferenceStep`, `Learner`, and
`SupervisedTraining`, including its file checkpointer and an inserted module
with dropout. `autodiff` suffices for manual loops; optional `train` also enables
Burn's own training framework. See [the examples](examples/training/README.md)
for the small tasks and their limits; synthetic sentiment is an integration
test, not evidence of natural-language classification quality.
These commands use default `std,flex`. With defaults disabled, select
`std,flex,autodiff` for manual loops or `train,flex` for the Learner example.

## Own parameters; borrow them in hooks

Derive `Module` on a struct containing the backbone, adapter, and head. Create
hooks inside its forward method and borrow the adapter from `self`:

```rust
hooks.edit(Layer::decoder(0).residual_post(), |hidden| {
    hidden.clone() + self.adapter.forward(hidden) * 0.1
});
```

The callback executes the adapter. The containing module registers its
parameters for optimization and checkpointing. Do not create a new adapter
inside the callback or leave learned tensors in an unregistered closure.
Build a fresh forward graph for every update. After computing your scalar loss,
the update is standard Burn:

```rust
let gradients = GradientsParams::from_grads(loss.backward(), &model);
model = optimizer.step(learning_rate, model, gradients);
```

The runnable examples include imports and complete model definitions. Use
`training::validate_trainable_module(&model)?` before optimization if your
composition may duplicate parameter owners. Register a shared parameter once
and reuse it in the forward graph: Burn 0.21 optimizers do not preserve ties
between separate fields carrying the same `ParamId`. The preflight reports
such trainable duplicates; inspecting duplicate owners can initialize them.

## Train on intermediate activations

`capture()` returns a detached observation. `capture_with_grad()` keeps the
selected tensor connected, so an auxiliary loss can update preceding parameters:

```rust
let selected = Layer::decoder(0).residual_post().token(0);
let capture = hooks.capture_with_grad(selected);
// Execute the model with these hooks, then retrieve the captured tensor.
let hidden = hooks.get(&capture)?;
let loss = task_loss + hidden.square().mean() * 0.01;
```

See [the activation guide](EXPERIMENTS.md) for the exact retrieval interface and
selection rules. A connected capture promises a differentiable loss path; it
does not promise Burn will retain that nonleaf tensor's own gradient from an
unrelated downstream loss. Edits preserve their returned tensor's graph.
Explicit `detach()` or conversion to host data breaks that path.

## Freeze parameters without cutting the graph

Apply Burn's `Module::no_grad()` to a floating-point backbone while keeping it
on `Autodiff<B>`. Its parameters remain fixed, but later frozen blocks still
propagate gradients to an earlier inserted adapter. Detaching the backbone's
output instead would prevent that adapter from learning through it.

[train_qwen_adapter.rs](examples/train_qwen_adapter.rs) demonstrates this with a
random or locally loaded Qwen prefix. It checks task and intermediate-loss
gradients, unchanged backbone tensors, and an identical next optimizer update
after resume. The fixture reports training-set accuracy only.

Use the same ownership principle for parameter groups: keep their parameters
discoverable and apply the desired policies through Burn's optimizer APIs or
your explicit loop. No additional parameter-group or trainer framework is
introduced by this crate.

## Evaluation, initialization, and records

Seed the backend before initializing and materializing parameters. Burn can
initialize parameters lazily; cloning an uninitialized parameter can produce
independent draws. Inspect the actual owner's values when recording baselines.
Different backends or libraries need not produce the same weights from a seed.

Use `model.valid()` for evaluation on the inner backend. Explicit Burn `Dropout`
modules then become inactive. Freezing parameters does not disable dropout.
Existing family forwards implement their documented deterministic graphs;
loading a checkpoint configuration does not automatically reproduce upstream
training dropout. Add regularization explicitly to the experiment when needed.

**Save the original training owner before converting to evaluation:**

```rust
let model_record = model.clone().into_record();
let optimizer_record = optimizer.to_record();
// Recreate the architecture and freezing policy, load both native records,
// then use resumed_model.valid() to evaluate.
```

Burn 0.21's `valid()` and `from_inner()` discard parameter save/load mappers.
Saving their result can transpose column-layout linear weights incorrectly on
reload, including silently for square matrices. Save the original owner, load
into a fresh owner, and reserve `valid()` for forward evaluation. Burn's Learner
checkpointer saves the training owner. Records do not by themselves save your
data cursor, schedule, RNG, or hook construction; retain those alongside the
architecture and experiment configuration. The Qwen adapter, Grokking, and
Tuned Lens examples verify model and optimizer state together; the three small
architecture examples demonstrate model-record reloads only.

## Losses and execution capabilities

`training::cross_entropy`, `log_softmax`, and `kl_divergence` use differentiable
pairwise reductions to avoid loss of F32 tail probability on Flex at large
vocabularies. KL is `KL(teacher || student)` and is averaged over examples.
Teacher tensors are not implicitly detached. Supply finite F32/F64 logits;
explicitly cast lower precision logits to F32 before loss computation. The
helpers validate shapes and targets but do not choose padding or task policy.
Cross-entropy validates label bounds by reading their minimum and maximum,
which synchronizes the device; logits remain on their original device.
Pairwise reduction does not remove every F32 limit: very small cross entropy
can round to zero near probability one. Paper guides identify F64-reference
and F32-training differences explicitly.

For unpadded host-generated batches, modular causal models also offer
`prepare_host_tokens(Vec<i32>, [batch, sequence])` and
`forward_prepared(&prepared, &options)`. Preparation checks every token and the
complete shape on the host, uploads the tokens, and constructs contiguous
positions starting at zero. The opaque input binds the checked payload to its
vocabulary/context bounds and device. Repeated forwarding validates metadata
without the ordinary path's five token/position scalar readbacks. It uses the
same arithmetic and supports `forward_prepared_with_hooks`; output logits still
default to the last position, so sequence training must request `All`. Input
preparation adds no BOS or label shift. Padded data and explicit position IDs
continue to use the ordinary checked tensor-input API. This avoids redundant
validation synchronizations; backend kernel speedups must be measured separately.

| Computation | Training contract |
| --- | --- |
| Floating-point stateless graph | Native Burn autodiff; custom operations must support gradients. |
| Frozen floating-point suffix | Input gradients continue through its operations. |
| Packed Q4/Q8 graph on autodiff | Unsupported in Burn 0.21; rejected with a contextual error. |
| Explicitly dequantized frozen weights | Floating-point training path, with its memory and compute costs. |
| Packed inference prefix on the inner backend | May supply detached features to a trainable head. |
| Cached or recurrent decode interventions | Require a separate state contract; use stateless full-sequence training here. |

## Train, then test the mechanism

[Grokking](examples/mechanistic/grokking_training/README.md) trains modular
addition from random weights, records delayed held-out generalization, and
intervenes on the learned Fourier circuit. [Clock and Pizza](examples/mechanistic/clock_and_pizza_training/README.md)
trains two attention rules and measures their learned geometry.
[Tuned Lens](examples/mechanistic/tuned_lens_training/README.md) learns new
translators on a frozen Pythia backbone. [Greater-Than](examples/mechanistic/greater_than_training/README.md)
adds a learned replacement as an explicitly new distillation experiment.
Each guide separates its measured training result from the retained analysis
of published weights, and declares compute, data splits, and reproduction limits.

The [v0.3 validation record](acceptance/measurements/v0_3-training-validation-20260910/README.md)
records 30 passing source/tooling/example checks, including 1,400 Rust tests on
both the development compiler and Rust 1.92. Scientific runs retain separate
protocols and evidence; this does not extend release or checkpoint support.
