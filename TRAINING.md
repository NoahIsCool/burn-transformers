# Train transformer experiments with Burn

An experiment is an ordinary Burn `Module`. Its backbone, inserted processing,
and task head use Burn's autodiff, optimizers, records, and data loaders. The
crate supplies transformer computation and activation access; the training
procedure stays in your application. These workflows use pinned Burn 0.21.

<!-- Maintainers: `python3 scripts/check_release.py --gate guides` compiles
every Rust block in this guide. A fragment names the hidden context it compiles
in with a `doctest NAME` comment directly above its fence; the matching
`doctest-context NAME` comment, beside the fragment that first uses it, holds
that context with `__FRAGMENT__` where the fragment goes. -->

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

<!-- doctest-context adapter-method
use burn::{nn::Linear, tensor::backend::Backend};
use transformers::intervention::{Hooks, Layer};

struct Experiment<B: Backend> {
    adapter: Linear<B>,
}

impl<B: Backend> Experiment<B> {
    fn fragment(&self) {
        let mut hooks = Hooks::<B>::new();
        __FRAGMENT__
    }
}
-->
<!-- doctest adapter-method -->
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

<!-- doctest-context optimizer-step
use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    tensor::{Tensor, backend::AutodiffBackend},
};

fn fragment<B: AutodiffBackend, M: AutodiffModule<B>, O: Optimizer<M, B>>(
    mut model: M,
    optimizer: &mut O,
    loss: Tensor<B, 1>,
    learning_rate: f64,
) -> M {
    __FRAGMENT__
    model
}
-->
<!-- doctest optimizer-step -->
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

<!-- doctest-context connected-capture
use burn::tensor::{Tensor, backend::Backend};
use transformers::{Result, intervention::{Hooks, Layer}};

fn fragment<B: Backend>(task_loss: Tensor<B, 1>) -> Result<()> {
    let mut hooks = Hooks::<B>::new();
    __FRAGMENT__
    Ok(())
}
-->
<!-- doctest connected-capture -->
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

To start from pretrained weights of a known family, initialize the model from
its configuration and call its `load_checkpoint` method. For the advertised
Qwen3.5 family, create the model with
`Qwen3_5ForCausalLm::init(config, &device)` and then call
`model.load_checkpoint(root, &LoadOptions::default())` on it. The `checkpoint`
module documentation's "Loading a concrete family" lists every family's method
and options, and `AutoCausalLm::load_local` chooses the family from
`config.json` instead.

Use the same ownership principle for parameter groups: keep their parameters
discoverable and apply the desired policies through Burn's optimizer APIs or
your explicit loop. No additional parameter-group or trainer framework is
introduced by this crate.

## Initialize deliberately

Comparisons between architectures trained from random weights are meaningful
only when every candidate starts from a stated distribution. Burn's defaults
are Kaiming-uniform `Linear` weights and `N(0, 1)` embeddings; with tied
embeddings the initial logits then have standard deviation about `sqrt(d)`,
so the softmax starts nearly one-hot. Measured on a one-layer modular model
with `d = 256` and `V = 512`: Burn's defaults give logit standard deviation
15.9 and an initial loss of about 250 nats, while `N(0, 0.02^2)` gives 0.32
and 6.33 nats, close to the uniform `ln V = 6.24`.

`InitPolicy` states the distribution as data and applies it to any Burn
module by parameter role (embedding, projection, residual projection, bias,
other), with path overrides taking precedence:

<!-- doctest-context init-policy
use burn::{module::Module, tensor::backend::Backend};

fn fragment<B: Backend, M: Module<B>>(model: M, layers: usize) -> transformers::Result<()> {
    __FRAGMENT__
    Ok(())
}
-->
<!-- doctest init-policy -->
```rust
use transformers::initialization::{InitPolicy, ParameterInit};

// N(0, 0.02^2) embeddings and projections, zero biases, GPT-2's 1/sqrt(2L)
// scaling for the projections that write the residual, and the final norm's
// scale reset to one; other parameters keep their constructed values.
let policy = InitPolicy::normal(0.02)
    .with_depth_scaling(layers, ["o_proj", "down_proj"])
    .with_override("final_norm", ParameterInit::Ones);
policy.validate_for(&model)?; // dry run: every name must match the model
let (model, report) = policy.apply(model)?;
```

- `ParameterInit` is `Keep`, `Normal { std }`, `Zeros`, `Ones`, or
  `Constant { value }`. An override path matches whole dot-separated
  components, as an exact parameter path or a module prefix; the last
  matching override wins.
- Names are checked before anything changes: an override that matches no
  parameter, or a residual projection that names no `Linear` field, is an
  error rather than a silently skipped rule.
- Replaced values keep their parameter identity, dtype, device, trainability,
  and save/load mapper, so records and optimizers see the same parameters;
  tied parameters receive one shared tensor; packed quantized parameters are
  kept. `report` lists every parameter with its role and initializer.
- `apply` materializes every replaced parameter. `apply_lazy` keeps lazy
  parameters lazy and replaces their deferred initializer, so a parameter
  overwritten by a loaded record is never drawn. Burn still computes its own
  default once at the first read before drawing the policy's value, draws
  happen in first-read order, and clones taken before the first read draw
  independently; use `apply` when that matters.
- Seed the backend first (`B::seed(&device, seed)`) for reproducible draws.
  Different backends need not produce the same weights from a seed.

Model helpers build the matching policy. For a modular causal model
(`ModularSeq2SeqLm` has the same pair):

<!-- doctest-context modular-spec
use burn::tensor::backend::Backend;
use transformers::{Result, architecture::DecoderSpec, models::custom::ModularCausalLm};

fn fragment<B: Backend>(spec: DecoderSpec, device: B::Device) -> Result<()> {
    __FRAGMENT__
    Ok(())
}
-->
<!-- doctest modular-spec -->
```rust
let policy = spec.scaled_init_policy(0.02);
let model = ModularCausalLm::<B>::init_with(spec, &policy, &device)?;
```

`scaled_init_policy` names the residual projections the spec actually
contains, and `Gpt2Config::init_policy()` / `GptNeoXConfig::init_policy()`
reproduce those families' Hugging Face initialization. The GPT policies keep LayerNorm
parameters as constructed; add `ParameterInit::Ones` and `Zeros` overrides to
reset them as well. Modular T5 relative-position biases start from
`N(0, 1/hidden_size)`, the reference initialization.

## Evaluation and records

Use `model.valid()` for evaluation on the inner backend. Explicit Burn `Dropout`
modules then become inactive. Freezing parameters does not disable dropout.
Existing family forwards implement their documented deterministic graphs;
loading a checkpoint configuration does not automatically reproduce upstream
training dropout. Add regularization explicitly to the experiment when needed.

**Save the original training owner before converting to evaluation:**

<!-- doctest-context records
use burn::{module::AutodiffModule, optim::Optimizer, tensor::backend::AutodiffBackend};

fn fragment<B: AutodiffBackend, M: AutodiffModule<B>, O: Optimizer<M, B>>(model: M, optimizer: O) {
    __FRAGMENT__
}
-->
<!-- doctest records -->
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
Pairwise reduction (folding contiguous halves) does not remove every F32
limit: very small cross entropy can round to zero near probability one. Paper
guides identify F64-reference and F32-training differences explicitly.

Sparse-expert families return their router balance objective when
`ForwardOptions` requests it, as a diagnostic that never changes the logits.
Qwen3-MoE and gpt-oss (`gpt_oss_load_balancing_loss`) reproduce Transformers'
stack-wide `aux_loss` without its coefficient, and Nemotron-H returns
Megatron's normalized sigmoid switch loss (one under perfect balance), also
unweighted: multiply these by the checkpoint's coefficient before adding them
to a training loss. DeepSeek's value follows its pinned reference and already
includes `aux_loss_alpha`; its `seq_aux = false` form is now `1/top_k` of what
earlier versions returned, which had reused the Qwen formula. Padding is
excluded from every count and mean. The FP32 router projection of DeepSeek-V2
and Nemotron-H receives a gradient on every backward pass, including a pass
over an auxiliary loss alone.

Copying one pretrained parameter to several targets with `TransferPlan`
creates independent parameters that train separately; `TiePolicy::Preserve`
keeps only ties that already exist in the source. Tie parameters explicitly in
one owner if they must stay shared.

| Computation | Training contract |
| --- | --- |
| Floating-point stateless graph | Native Burn autodiff; custom operations must support gradients. |
| Frozen floating-point suffix | Input gradients continue through its operations. |
| Packed Q4/Q8 graph on autodiff | Unsupported in Burn 0.21; rejected with a contextual error. |
| Explicitly dequantized frozen weights | Floating-point training path, with its memory and compute costs. |
| Packed inference prefix on the inner backend | May supply detached features to a trainable head. |
| Cached or recurrent decode interventions | Require a separate state contract; use stateless full-sequence training here. |
| Long sequences under autodiff | Score-bounded attention chunking bounds inference only: Muse, DeepSeek, and `GroupedQueryAttention::attend_bounded` switch to one unbounded call on autodiff backends (`AttentionExecutionPolicy::effective_on`), so memory grows with the full score plane. |

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
