# Transformer experiments

Start with [activation_workbench.rs](examples/activation_workbench.rs). It builds
tiny transformers, captures one token's neurons, ablates and restores them,
inserts a small module, enables bidirectional attention, and then runs one
family-generic experiment on a modular decoder, GPT-2, and a Llama-family
model. No downloads are needed:

```sh
cargo run --release --locked --example activation_workbench
```

The fragments below assume an initialized model, an input, and a backend `B`;
the workbench supplies the complete setup. For a trained circuit, read the
[grokking experiment](examples/mechanistic/grokking.md).

<!-- Maintainers: `python3 scripts/check_release.py --gate guides` compiles
every Rust block in this guide. A fragment names the hidden context it compiles
in with a `doctest NAME` comment directly above its fence; the matching
`doctest-context NAME` comment, beside the fragment that first uses it, holds
that context with `__FRAGMENT__` where the fragment goes. -->

For architecture changes, start with these complete examples:

| Experiment | What to change |
| --- | --- |
| [T5 classifier](examples/t5_classifier.rs) | Per-layer heads and MLP widths, a module between stages, and a masked pooling head. |
| [Qwen prefix](examples/qwen_prefix.rs) | Select pretrained blocks, inspect their neurons, and replace the vocabulary head. |
| [Custom composition](examples/custom_composition.rs) | Replace attention with your own sequence mixer, omit normalization, and connect different residual widths. |
| [MLA encoder-decoder](examples/mla_encoder_decoder.rs) | Reuse compressed attention for self/cross-attention and edit its latent source representation. |

The default commands demonstrate architecture and intervention with untrained
task heads. Add `--features autodiff` and `-- --train` to the T5, composition,
and MLA examples to train them. The [training guide](TRAINING.md) also covers
frozen Qwen adapters and native Burn Learner integration. The
[paper examples](examples/README.md) include actual training and retained
published-checkpoint measurements.

## One experiment, any model

Two traits make an experiment independent of the model family:

- `HookedModel` lists a model's activation points and runs a complete
  stateless forward pass inside one hook scope.
- `StagedModel` runs a stack in explicit stages: `embed`, `run_layers` over a
  contiguous block range, and `normalize`, exchanging a `HiddenState`.

Every causal family implements both. T5 implements `HookedModel` for the full
model and both traits for `T5Encoder`; a modular seq2seq model implements
`HookedModel` and stages through `encoder_stages()` and `decoder_stages(…)`.
`AutoCausalLm` implements both for every registered family, `AutoSeq2SeqLm`
implements `HookedModel`, and a `LoadedModel` forwards both to its model, so
the same function runs on a concrete family, on an automatically selected
model, or directly on a checkpoint loaded with `load_local`:

```rust
use burn::tensor::backend::Backend;
use transformers::{
    CausalLmInput, CausalLmOutput, ForwardOptions, Result,
    architecture::{HookedModel, StagedModel},
    intervention::{Hooks, Layer},
};

fn residual_experiment<B: Backend, M>(
    model: &M,
    clean: CausalLmInput<B>,
    corrupted: CausalLmInput<B>,
) -> Result<()>
where
    M: HookedModel<B, Input = CausalLmInput<B>, Output = CausalLmOutput<B>> + StagedModel<B>,
{
    // Activation cache: capture every point the model lists, whatever its rank.
    let mut cache = Hooks::new();
    let handles = model
        .activation_points()?
        .into_iter()
        .map(|point| cache.capture_dynamic(point))
        .collect::<Vec<_>>();
    let output = model.forward_with_hooks(clean.clone(), &ForwardOptions::default(), &mut cache)?;
    for handle in &handles {
        let activation = cache.get_dynamic(handle)?; // still on the device
        println!("{}: {:?}", handle.point().name, activation.shape);
    }

    // Ablation: zero the residual stream leaving block 0.
    let mut ablate = Hooks::new();
    ablate.edit(Layer::decoder(0).residual_post(), |x| x.zeros_like());
    let ablated = model.forward_with_hooks(clean.clone(), &ForwardOptions::default(), &mut ablate)?;

    // Patching across runs: resume a corrupted run from the clean block-0 residual.
    let clean_state = model.run_layers(model.embed(clean)?, 0..1)?;
    let mut patched = model.run_layers(model.embed(corrupted)?, 0..1)?;
    patched.hidden = clean_state.hidden;
    let patched = model.normalize(model.run_layers(patched, 1..model.num_layers())?)?;
    Ok(())
}
```

Concrete types keep their own inherent methods of the same names, which take
precedence in method-call syntax (Qwen's inherent `embed`, for example, borrows
its input); generic code bounded by the traits sees the uniform signatures.

## Capture what you need

Choose a computation point, then optionally narrow it. Indices start at zero;
selecting one token or head keeps that dimension with length one.

<!-- doctest-context hooked-model
use burn::tensor::backend::Backend;
use transformers::{CausalLmInput, CausalLmOutput, ForwardOptions, Result, architecture::HookedModel};

fn fragment<B: Backend, M>(model: &M, input: CausalLmInput<B>, options: ForwardOptions) -> Result<()>
where
    M: HookedModel<B, Input = CausalLmInput<B>, Output = CausalLmOutput<B>>,
{
    __FRAGMENT__
    Ok(())
}
-->
<!-- doctest hooked-model -->
```rust
use transformers::intervention::{Hooks, Layer};

let neurons = Layer::decoder(0).mlp_activation().token(1);
let mut hooks = Hooks::<B>::new();
let saved = hooks.capture(&neurons);
let output = model.forward_with_hooks(input.clone(), &options, &mut hooks)?;
let activation = hooks.get(&saved)?; // [batch, 1, intermediate_width]
```

Captures stay on the tensor's device and detach from the gradient graph by
default. Use `capture_with_grad` when a loss must depend on a capture. Use
`observe` to compute a small statistic without storing the activation in hooks:

<!-- doctest-context hooks
use burn::tensor::backend::Backend;
use transformers::intervention::{Hooks, Layer};

fn fragment<B: Backend>() {
    let neurons = Layer::decoder(0).mlp_activation().token(1);
    let mut hooks = Hooks::<B>::new();
    __FRAGMENT__
}
-->
<!-- doctest hooks -->
```rust
hooks.observe(&neurons, |x| println!("mean: {:?}", x.mean().into_scalar()));
```

Here `into_scalar()` explicitly transfers a result to the host. An observer
sees a detached view and never rewrites the activation, so it adds no edit or
constraint work: observing an attention pattern costs nothing quadratic.
Register all actions before the forward call. Reusing hooks reruns the actions
and replaces their captures; a failed call discards partial and previous
captures.

## Change an activation or patch another run

<!-- doctest-context neuron-patch
use burn::tensor::backend::Backend;
use transformers::{
    CausalLmInput, CausalLmOutput, ForwardOptions, Result,
    architecture::HookedModel,
    intervention::{Hooks, Layer},
};

fn fragment<B: Backend, M>(
    model: &M,
    input: CausalLmInput<B>,
    corrupted_input: CausalLmInput<B>,
    options: ForwardOptions,
) -> Result<()>
where
    M: HookedModel<B, Input = CausalLmInput<B>, Output = CausalLmOutput<B>>,
{
    let neurons = Layer::decoder(0).mlp_activation().token(1);
    let mut hooks = Hooks::<B>::new();
    let saved = hooks.capture(&neurons);
    model.forward_with_hooks(input.clone(), &options, &mut hooks)?;
    let activation = hooks.get(&saved)?;
    __FRAGMENT__
    Ok(())
}
-->
<!-- doctest neuron-patch -->
```rust
let mut ablate = Hooks::<B>::new();
ablate.edit(&neurons, |x| x.zeros_like());
let ablated = model.forward_with_hooks(input.clone(), &options, &mut ablate)?;

let mut patch = Hooks::<B>::new();
patch.replace(&neurons, activation);
let patched = model.forward_with_hooks(corrupted_input, &options, &mut patch)?;
```

Actions at the same point execute in registration order. A capture before an
edit records the original; a capture afterward records the changed tensor.
Callbacks receive only the selected slice. Other entries at that point remain
intact, while later computation can propagate the change.

Replacements must retain the selected shape, dtype, and device. Use `try_edit`
for a callback that returns the crate's `Result`. Errors come in two kinds:

- **Registration mistakes**, found before any computation (an unknown point,
  an axis the point does not have, an empty or duplicate selection), are
  `Error::Configuration`.
- **Failures while the model runs** (a selection outside the tensor, reported
  as `outside extent N`; a callback error; a shape-changing edit; a violated
  constraint) are `Error::Intervention { point, reason, cause }`. It names the
  point and keeps a callback's original error in `cause` for matching.

To change a block's width, compose modules with explicit adapters instead of
returning a different shape from a tensor edit.

## Insert your own processing

An edit can borrow an ordinary Burn module:

<!-- doctest-context experiment-adapter
use burn::{nn::Linear, tensor::backend::Backend};
use transformers::intervention::{Hooks, Layer};

/// Owns the adapter, as the workbench's `ExperimentModel` does.
struct Experiment<B: Backend> {
    adapter: Linear<B>,
}

fn fragment<B: Backend>(experiment: &Experiment<B>) {
    let mut hooks = Hooks::<B>::new();
    __FRAGMENT__
}
-->
<!-- doctest experiment-adapter -->
```rust
hooks.edit(Layer::decoder(0).residual_post(), |hidden| {
    hidden.clone() + experiment.adapter.forward(hidden) * 0.1
});
```

Own the backbone and adapter together in a struct with `#[derive(Module)]`, as
the workbench does. This registers the adapter for parameter traversal,
checkpointing, and optimization; the temporary closure only executes it.
Hooks do not detach edit results; gradient connectivity follows your callback's
operations and replacement tensors. Use `capture_with_grad` for a differentiable
patch. [Training examples](TRAINING.md) validate actual optimizer updates,
learning, freezing, and model/optimizer record round trips. Keep a shared weight
in one registered owner when optimizing; duplicate parameter fields do not
remain tied under Burn 0.21 optimizers.

## Find the right point

`HookedModel::activation_points(&model)?` returns the catalog for the
execution path you are about to run (a concrete family type's inherent
`activation_points()` returns the same list without a `Result`): each
`PointInfo` has a stable name and its axes in tensor order. Generic tools capture any listed point with `Hooks::capture_dynamic`, or
recover a typed point with `Point::<D>::try_from(info)?`. Requesting a point
the catalog does not list fails before computation.

| Address | Tensor meaning |
| --- | --- |
| `Stack::Encoder.embedding()` / `Stack::Decoder.final_norm()` | Embedded or final normalized `[batch, token, channel]` states. |
| `Layer::encoder(i)` / `Layer::decoder(i)` | Stack and zero-based block index. |
| `.residual_pre()` / `.residual_post()` | Residual before / after the complete block. |
| `.attention_input()` / `.attention_output()` | Mixer input after any pre-normalization / output before residual addition. `.mixer_input()` / `.mixer_output()` are the same points under a name that also fits recurrent mixers. |
| `.mlp_input()` / `.mlp_output()` | Feed-forward input / output before residual addition. |
| `.mlp_pre_activation()` / `.mlp_activation()` | Dense projection before / after its nonlinearity. In gated MLPs these expose the gate branch **before multiplication**. |
| `.self_attention().query()` / `.key()` / `.value()` | `[batch, head, token, channel]`; Q/K after the model's position and normalization transforms. K/V keep native heads under grouped-query attention and use expanded heads for MLA. |
| `.self_attention().head_context()` | Per-query-head context before the output gate and output projection. |
| `.self_attention().scores()` / `.pattern()` | `[batch, head, query, key]`; scaled and soft-capped scores before bias/masking / effective softmax weights. |

Decoder cross-attention uses `.cross_attention()` and the corresponding
`cross_attention_input/output()` points. Select with `.token(i)`, `.head(i)`,
`.batch(i)`, `.channels(start..end)`, or `.slice(Axis::Token, start..end)`.
Attention matrices use `.query_token(i)` and `.key_token(i)` instead of `.token(i)`.
For example, on a full-attention layer, ablate one head before its output projection:

<!-- doctest hooks -->
```rust
hooks.edit(Layer::decoder(0).self_attention().head_context().head(0), |x| {
    x.zeros_like()
});
```

Catalogs omit what an execution path cannot expose faithfully. Every
mixture-of-experts family addresses its router points through the same
`Layer` constructors, and their axes are `Axis::EXPERT` and
`Axis::EXPERT_SLOT`, so `Layer::decoder(3).router_logits().slice(Axis::EXPERT,
5..6)` selects expert 5's logits on any of them. Family specifics:

| Family | Notes on its catalog |
| --- | --- |
| Llama family, Gemma | Embedding; per block `residual.pre`, `attention.input`, Q/K/V, head context, `attention.output`, `mlp.input`, `mlp.pre_activation`, `mlp.activation`, `mlp.output`, `residual.post`; final norm. Gemma's embedding point is the `sqrt(d)`-scaled residual, and its `attention.output`/`mlp.output` follow the post-sublayer norms. |
| gpt-oss | Like the Llama family, but the MLP exposes `mlp.router_logits` (`Layer::router_logits()`, `[batch, token, expert]`, before top-k: editing it changes which experts run) and `mlp.router_weights` (`Layer::router_weights()`, `[batch, token, expert_slot]` F32 mixture weights) instead of neuron points. |
| Qwen3-MoE | The same two router points on sparse blocks (`Layer::router_logits()` and `router_weights()`); dense blocks expose neuron points. `Qwen3MoeModel` is a headless backbone with its own staged execution. |
| Qwen3.5 / Qwen3.8 | DeltaNet blocks expose mixer input/output; full-attention blocks expose the attention points; the text backbone and individual blocks are public. |
| DeepSeek-V2 | MLA's `kv_latent` (and `query_latent` for low-rank-query checkpoints), the expanded Q/K/V, scores, pattern, and head context; sparse blocks add `mlp.router_logits`, `mlp.routed_output`, and `mlp.shared_output` (`Layer::router_logits()`, `routed_output()`, and `shared_output()`). Requesting an MLA point makes that layer evaluate one full score plane. |
| Muse Glimmer | Q/K/V, head context, `attention.gate_logits`, `attention.gate`, and the sandwich `post_attention_norm`/`post_feedforward_norm` points; no scores or pattern, because its exact path dispatches backend kernels. |
| Nemotron-H | Points depend on the block kind: Mamba-2 blocks expose mixer input/output, attention blocks the attention points, dense blocks neuron points, and MoE blocks only their combined output. |
| T5, GPT-2, GPT-NeoX | The full point set above, with encoder and cross-attention points for T5. GPT-NeoX has no router, so it answers `router_logits` and `auxiliary_losses` requests with empty diagnostics. |
| Custom modular models | Attention, MLA, and dense feed-forwards expose their full points; differential attention, DeltaNet, and Mamba-2 expose only mixer input/output; sparse MoE exposes its outer boundaries. |

With padding, the points of the Llama family, Gemma, gpt-oss, and the T5
decoder carry validity masks: padded entries read as zero before observers run
and stay zero after edits, so an edit cannot revive a padded token or leak it
into a valid query. T5 padded decoder hooks use the original rectangular token
indices, even with left padding or holes; logical positions follow valid-token
order. In other families padded keys are excluded from attention, but padded
activations are not guaranteed to be zero; select valid tokens explicitly.

### Attention scores, patterns, and sinks

Scores and patterns exist only when a layer materializes them. Grouped-query
attention requires an explicit `AttentionPrecisionMode::EagerFp32Softmax`
choice, which materializes the full quadratic attention matrix, so select
short sequences. Custom models, the Llama family, Gemma, gpt-oss, GPT-2, and
GPT-NeoX expose `with_attention_precision` (GPT-2 and GPT-NeoX also
`set_attention_precision`); standalone attention uses
`GroupedQueryAttentionConfig::with_precision_mode`, and Qwen attention modules
have a public `precision_mode` field. For a loaded `AutoCausalLm`, take the
concrete variant from `loaded.into_model()` first. Check the resulting catalog.

<!-- doctest-context modular-model
use burn::tensor::backend::Backend;
use transformers::models::custom::ModularCausalLm;

fn fragment<B: Backend>(model: ModularCausalLm<B>) {
    __FRAGMENT__
}
-->
<!-- doctest modular-model -->
```rust
use transformers::attention::AttentionPrecisionMode;
let model = model.with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
```

Q/K/V and head-context hooks preserve the configured attention dispatch.
Switching precision changes the numerical execution graph; compare an
intervention with a clean run using the **same** precision setting. Gemma's
scores are soft-capped before masking, as in the reference.

With learned attention sinks (gpt-oss), each softmax row includes one sink
logit that is dropped afterwards, so an emitted pattern row sums to
`1 - p_sink`. The derived, read-only point `AttentionPoints::sink_probability`
(`[batch, head, query]`) exposes `p_sink = 1 - sum_k pattern`; an attention
implementation emits it with `Hooks::apply_sink_probability` and lists it in
its catalog. Edit the pattern, not the sink probability, to change the
computation.

## Run in stages

Execution stages make a prefix and a replacement task head explicit:

<!-- doctest-context staged-encoder
use burn::{nn::Linear, tensor::backend::Backend};
use transformers::{CausalLmInput, Result, architecture::StagedModel};

fn fragment<B: Backend, E: StagedModel<B>>(
    encoder: &E,
    input: CausalLmInput<B>,
    adapter: &Linear<B>,
    classifier: &Linear<B>,
) -> Result<()> {
    __FRAGMENT__
    Ok(())
}
-->
<!-- doctest staged-encoder -->
```rust
let mut state = encoder.embed(input)?;
state = encoder.run_layers(state, 0..1)?;
state.hidden = state.hidden.clone() + adapter.forward(state.hidden);
state = encoder.run_layers(state, 1..3)?;
state = encoder.normalize(state)?;
let predictions = classifier.forward(state.hidden);
```

`HiddenState` carries positions, padding, the next layer index, the evaluated
rotary context, and whether final normalization has happened. Running a range
never adds embeddings or final normalization. Ranges must continue from
`next_layer`; normalized states cannot resume. You may normalize a prefix
deliberately for a new task head. Each `_with_hooks` call is its own hook scope
whose catalog covers only that stage; use one `Hooks` collection per stage.

`embed` validates the input once, and each `run_layers` call validates the
context once rather than per block. If you edit `state.context` fields
directly, call `state.context.validate(shape, &device)` before resuming.
`ExecutionContext::contiguous(batch, tokens, &device)` builds an unpadded
context, and `with_encoder` attaches an encoder context. Staged Llama-family
and Gemma states reject an evaluated context beyond
`max_position_embeddings`, which would otherwise select different rotary
tables.

Custom seq2seq models provide `encoder_stages()` and `decoder_stages(encoder)`
views that implement `StagedModel`, plus the original `embed_encoder`,
`run_encoder_layers`, `normalize_encoder`, and decoder methods. Pass a
normalized encoder state's `into_encoder_context()?` to the decoder to
preserve source positions and padding. T5 provides staged execution through
its owned `T5Encoder` (`T5ForConditionalGeneration::into_encoder`). State
carries execution metadata, not parameters or decode caches.

## Change attention and use another task head

Change attention through the context before running the relevant stages:

<!-- doctest-context staged-state
use burn::tensor::{Bool, Tensor, backend::Backend};
use transformers::{experiment::HiddenState, intervention::Layer};

fn fragment<B: Backend>(mut state: HiddenState<B>, allowed: Tensor<B, 4, Bool>) {
    __FRAGMENT__
}
-->
<!-- doctest staged-state -->
```rust
use transformers::experiment::AttentionPolicy;

state.context = state.context
    .with_attention(AttentionPolicy::bidirectional())
    .with_layer_attention(Layer::encoder(1), AttentionPolicy::custom(allowed));
```

`allowed` has axes `[batch, head, query, key]`, with singleton broadcasting.
Here **true means allowed**; padding stays independent. Add `.with_bias(bias)`
for an additive score bias, or `.with_allowed(allowed)` to restrict a causal
policy. A layer override takes precedence over the pass default. Cross-attention
has separate `with_cross_attention` and `with_layer_cross_attention` settings.
Absent overrides preserve the original architecture's conventions. An explicit
policy replaces a layer's structural rule entirely, sliding window and
chunked-local connectivity included, while padding still applies.

Which families honor policies: attention layers of the Llama family, Gemma,
gpt-oss, Qwen, DeepSeek, Muse, Nemotron-H, T5, and custom models do (MLA and
Muse layers with an override evaluate one full score plane). Recurrent mixers
(DeltaNet, Mamba-2) reject non-default policies before callbacks run; dense
and MoE blocks mix no tokens and ignore them. Staged GPT-2 and GPT-NeoX reject
attention policies.

Custom full-attention models support `forward_bidirectional` and
`forward_bidirectional_with_hooks`. Padding remains excluded; future tokens
become visible. For classification or another task, `forward_hidden` and
`forward_hidden_bidirectional` return normalized states without computing
vocabulary logits; pool them and apply your own head. Both have `_with_hooks`
variants. T5 and custom seq2seq models provide `encode_with_hooks` for encoder
states.

For an unpadded batch, a custom sentiment head can use mean pooling:

<!-- doctest-context modular-classifier
use burn::{nn::Linear, tensor::backend::Backend};
use transformers::{CausalLmInput, Result, models::custom::ModularCausalLm};

fn fragment<B: Backend>(
    model: &ModularCausalLm<B>,
    input: CausalLmInput<B>,
    classifier: &Linear<B>,
) -> Result<()> {
    __FRAGMENT__
    Ok(())
}
-->
<!-- doctest modular-classifier -->
```rust
let hidden = model.forward_hidden_bidirectional(input)?;
let pooled = hidden.mean_dim(1).squeeze_dim::<2>(1);
let class_logits = classifier.forward(pooled);
```

For padded batches, exclude padded positions from the pooling sum and divisor.

Pattern edits may produce unnormalized or negative weights. Causal/padding
exclusions are reapplied after each edit, before subsequent captures. To expose
future tokens, change the attention policy. At the shared `attend_with_hooks`
boundary, `causal` and the explicit mask control visibility; Boolean attention
masks mean **true = excluded**, while model token masks mean **true = valid**.

For fundamental attention changes, `attend_with_pattern` accepts a structural
transformation before consumer pattern hooks. The
[Clock and Pizza model](examples/mechanistic/clock_and_pizza/model.rs) uses
`alpha * softmax(scores) + (1 - alpha) * ones`; its Pizza rows deliberately sum
to two. Its clean reference therefore tests the changed computation itself.

For long eager contexts, `GroupedQueryAttention::attend_bounded` runs exact
attention in query chunks whose score planes stay below a configured element
bound. Keys start at absolute position zero, and it accepts no additive bias
or hooks. On autodiff backends, whose tape keeps every chunk, the policy
evaluates as one unbounded call with the same values
(`AttentionExecutionPolicy::effective_on`), as in Muse and DeepSeek. Structural masks for your own kernels come from
`AttentionConnectivity::exclusion_mask` with a `MaskGeometry`; give the
geometry `with_key_start(n)` when a cache has dropped the first `n` keys, so
chunk and window boundaries stay absolute.

## Edit a recipe and choose which weights to reuse

`T5EncoderRecipe::from_config(config)?` preserves T5's normalization,
relative-position bias, feed-forward convention, and ordering. Edit
`recipe.layers[i].attention_heads`, `.head_dim`, or `.feed_forward_width`, and
resize the layer vector before `init`. An unchanged recipe preserves the shared
relative-bias graph. A new head count needs a compatible bias table; the recipe
reuses the earliest compatible table or initializes one explicitly.

Construct a new architecture before transferring existing parameters:

<!-- doctest-context transfer
use burn::{module::Module, tensor::backend::Backend};
use transformers::Result;

fn fragment<B: Backend, S: Module<B>, T: Module<B>>(pretrained: S, new_encoder: T) -> Result<()> {
    __FRAGMENT__
    Ok(())
}
-->
<!-- doctest transfer -->
```rust
use transformers::transfer::{ShapeChangePolicy, TransferPlan};

let (encoder, report) = TransferPlan::new()
    .copy_prefix("shared", "shared")
    .copy_prefix("encoder", "encoder")
    .on_shape_change(ShapeChangePolicy::Reinitialize)
    .apply(&pretrained, &new_encoder)?;
```

The default shape policy rejects mismatches. Reinitialization keeps the target's
existing initialization; explicit transformation callbacks can instead change
selected weights. Reports list reused, transformed, omitted, and newly initialized
parameters, plus preserved ties and deliberate untying. Validation completes
before the cloned result changes; source and target stay intact. Copies preserve
target trainability and start a fresh gradient graph. Dtype/device changes are
explicit transformations.

Copying one source parameter to several targets (fan-out, for example to
duplicate a layer) creates independent parameters that train separately;
`TiePolicy::Preserve` keeps only ties that already exist in the source, and
fanning out a member of such a tie is an error. `transfer::parameters(&model)`
discovers module paths, including packed quantized and rank-zero parameters
(`ParameterInfo::is_transferable()` tells which can be copied); a plan fails
only if it selects one that cannot.

For Qwen, `Qwen3_5BackboneRecipe::from_local_config(path)?.prefix(count)?` reads
the architecture without loading weights. `init_random` and `load_pretrained`
are distinct operations. The latter streams selected checkpoint tensors and
reports omitted blocks and the discarded vocabulary head; it does not first
construct the entire pretrained model. See the prefix example for load options.

## Compose with compressed latent attention

The shared `transformers::mla::MultiLatentAttention` module works in self-attention
and cross-attention, with independent query/source lengths, optional rotary
positions, additive bias, and causal or bidirectional visibility. It is also
available through custom `MixerSpec::Mla` recipes. Run the complete
[MLA encoder-decoder example](examples/mla_encoder_decoder.rs):

```sh
cargo run --release --locked --example mla_encoder_decoder
```

`kv_latent()` selects normalized compressed source states **before** expansion
into keys and values. For an initialized MLA module, edit those states directly:

<!-- doctest-context mla-cross-attention
use burn::{nn::Linear, tensor::{Tensor, backend::Backend}};
use transformers::{Result, intervention::{Hooks, Layer}, mla::MultiLatentAttention};

fn fragment<B: Backend>(
    mla: &MultiLatentAttention<B>,
    adapter: &Linear<B>,
    query: Tensor<B, 3>,
    encoder_states: Tensor<B, 3>,
) -> Result<()> {
    __FRAGMENT__
    Ok(())
}
-->
<!-- doctest mla-cross-attention -->
```rust
let points = Layer::decoder(0).cross_attention();
let mut hooks = Hooks::<B>::new();
hooks.edit(points.kv_latent(), |states| {
    states.clone() + adapter.forward(states) * 0.1
});
let output = hooks.run(mla.activation_points(points), |hooks| {
    let projection = mla.project_with_hooks(query, encoder_states, points, hooks)?;
    mla.attend_with_hooks(projection, None, None, false, points, hooks)
})?;
```

Here token selections on `kv_latent()` index **source** tokens. The optional
`query_latent()` point belongs to projection and exists only when a query
bottleneck is configured. `projection_points()` and `attention_points()` expose
the individual stages' catalogs; `activation_points()` combines both. MLA's
explicit reference graph already exposes scores and patterns with F32 softmax;
no additional precision switch is required. It materializes expanded K/V and
the full attention matrix. The example owns its adapter as a registered module.

## Build your own architecture

`DecoderSpec` (causal stacks) and `Seq2SeqSpec` (encoder-decoder) describe a
topology as serializable data; `ModularCausalLm` and `ModularSeq2SeqLm` build
it on any backend. Validation translates every block into the same shared
runtime configuration that construction uses, so a spec that validates builds,
and errors name the exact field (for example
`layers[3].mixer.attention.options.attention_sinks`).
The [README](README.md#4-build-your-own-model) shows a complete small spec.

Each block chooses its own mixer, feed-forward, normalization (`Rms`, optionally
zero-centered, or `Layer`), placement (`Pre`, `Post`, or `Sandwich`), and
positions:

| Choice | Spec |
| --- | --- |
| Grouped-query attention | `MixerSpec::Attention` (compact) or `MixerSpec::AttentionWithOptions` with `AttentionOptions`: global, sliding-window, or chunked-local `connectivity`; explicit `score_scale`; `logit_softcap`; learned `attention_sinks` (not combinable with a soft cap); per-projection `projection_bias`; per-head or full-width `query_key_norm_scope`; sigmoid or SiLU output gate. |
| Multi-head latent attention | `MixerSpec::Mla`, with its own score scale and connectivity. |
| Differential attention | `MixerSpec::DifferentialAttention`: paired heads `A_i - lambda A_(i+H/2)` with RoPE or no positions. |
| Recurrent mixers | `MixerSpec::DeltaNet` (Gated DeltaNet); `MixerSpec::Mamba2`, executed by the shared Mamba-2 kernel with Hugging Face `Mamba2Config` defaults for settings the spec does not carry and a group-wise gated norm (the `mamba_ssm` convention). Both are causal only. |
| Positions | `PositionEncodingSpec::Rope` with `RopeScaling::{None, Linear, DynamicNtk, Yarn, Llama3, LongRope, Proportional}`, `T5RelativeBias`, or `None`. |
| Feed-forward | `Dense` (gated), `Ungated`, `ClampedSwiGlu`, `MixtureOfExperts` (with a `RoutingPolicy`: softmax or sigmoid scoring, top-k before or after scoring, group-limited selection, correction bias; plus shared experts, a shared-expert gate, and expert biases), `ClampedMixtureOfExperts` (gpt-oss's clamped SwiGLU routed and shared experts; `experts.activation` must be `silu`), or `None` for attention-only blocks. |
| Embedding and readout | `DecoderSpec::embedding_scale` multiplies token embeddings by a dtype-rounded scalar (Gemma's `sqrt(d)`), and `DecoderSpec::final_logit_softcap` caps logits at `c · tanh(z / c)` (Gemma 2); an encoder stack rejects a cap. |
| Cross-attention | `Seq2SeqSpec::cross_attention_options` sets its score scale, soft cap, sinks, and biases. |

Bidirectional encoder stacks reject sliding windows and recurrent mixers.
`ResidualTopology::Gated` stays reserved and is rejected at construction: no
shipped family or pinned reference defines a gated residual, so its semantics
would be invented, and executing it would add parameters to the stable block
record layout. The `architecture::spec` documentation's "Not expressible"
table lists the family mechanisms a spec cannot describe (Muse Glimmer's
scale-free embedding norm and bounded logit map, GPT-2 absolute positions,
GPT-NeoX parallel residuals, OLMo 2 output-only norms, and MTP heads), and the
Gemma 2 and gpt-oss modular parity tests show specs reproducing those
families block for block. Both models implement `HookedModel` and
`StagedModel` (seq2seq through its stage views).
Initialize from scratch with a depth-scaled policy (see
[initialization](TRAINING.md#initialize-deliberately)):

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

Modular decoding is a correctness-oriented reference path: attention layers
cache their token history and recompute, and recurrent mixers replay their
history. Family runtimes keep the optimized native KV, latent, ring, and
recurrent caches used for performance work. The modular decode state accepts
inactive and never-started rows as exact-zero no-ops (such a row is evaluated
over one uncommitted history slot), so `generate_causal` with
`prefill_chunk_size` works on left-padded batches.

## Extend the computation

Implement `HiddenTransform<B>` on an ordinary `#[derive(Module)]` type to define
your own sequence mixer or feed-forward computation. `ResidualBlock` composes
the two with explicit normalization, including `without_normalization`.
`WidthAdapter` owns input/output linear projections around an inner module:
`WidthAdapter::new(block, 8, 12, 6, &device)?` maps eight channels into a
twelve-channel block and returns six. These projections participate in Burn
parameter traversal, records, and gradients. The custom composition example
shows the entire forward path and a mixer with no attention operation.

`GroupedQueryAttention` used directly as a `HiddenTransform` is position-free
(NoPE): it reads padding and the attention policy from the context, never its
positions. Wrap it in `experiment::RotaryAttention` to rotate queries and keys
by the context's positions, with any RoPE scaling; dynamic NTK and LongRoPE
select their tables from the context's evaluated length. `ResidualBlock::run`
and `run_with_hooks` validate the context once, inside the hook scope.

In your own module, declare additional typed points and list them in its catalog:

<!-- doctest-context custom-point
use burn::{nn::Linear, tensor::{Tensor, backend::Backend}};
use transformers::{Result, intervention::Hooks};

fn fragment<B: Backend>(hidden: Tensor<B, 2>, readout: &Linear<B>) -> Result<()> {
    let mut hooks = Hooks::<B>::new();
    __FRAGMENT__
    Ok(())
}
-->
<!-- doctest custom-point -->
```rust
use transformers::intervention::{Axis, Point};

let point = Point::custom("probe.bottleneck", [Axis::Batch, Axis::Channel]);
let output = hooks.run(vec![point.info()], |hooks| {
    let hidden = hooks.apply(point, hidden)?;
    Ok(readout.forward(hidden))
})?;
```

Use one outer `run` scope and emit points with `apply` inside submodules; nested
scopes are rejected. Custom attention or other processing can be ordinary Rust
and Burn tensor operations, as in the grokking model. Hooks do not require a
graph language or checkpoint naming convention.

### Cached decoding for your own modules

Family runtimes share one cache protocol, available to your own models through
`cache::TensorDecodeState`. One cached forward step:

1. creates a `step(token_counts)` write set after any lifecycle call
   (reset, reorder, truncate, transactions); a step is rejected as stale if
   anything else moves the cursors before it is published;
2. per attention layer, builds one read plan per cache group (or, for full
   storage, writes the new keys ahead with `write_kv_ahead` and reads through
   them with `read_kv_through`), and masks invalid slots with the segment's
   `exclusion_mask` and `masked_values`;
3. records each layer's new state with the `push_*` methods;
4. publishes everything at once with `publish_step`, which checks that every
   layer and active row was written exactly once and either advances all
   lengths or changes nothing readable.

Ring groups (sliding windows), single-plane latent groups (MLA), recurrent
groups, and speculative transactions use the same steps. The `cache` module
documentation states the invariants in full.

### Custom generation loops

`generate_causal` covers ordinary sampling. For your own decode loop,
`TokenHistory` tracks per-row token counts incrementally and
`LogitsProcessor::process_with_history` applies penalties, bias, and filters
exactly as the coordinator does; `process_with_prompt_lengths` does the same
from host token rows with an explicit prompt boundary. `generation::speculative`
provides the lossless acceptance rule (`SpeculativeVerifier`), the accepted
prefix arithmetic, and `prompt_lookup_draft` for draft-model-free speculation.

## Current boundaries

Interventions remain stateless; hooks do not modify cached generation or
recurrent decode state. Changed bidirectional computation requires a fresh
full-sequence execution. Recurrent architectures keep their explicit
capability restrictions. Inside a Nemotron-H transaction, multi-token steps
follow token-by-token decode semantics, which can differ from a multi-token
prefill outside one where the reference clamps prefill time steps.

The v0.2 interfaces passed their [recorded validation](acceptance/measurements/v0_2-interface-validation-20260910-retry/README.md).
The v0.3 [training contract](TRAINING.md) identifies gradient, mode, record,
and precision boundaries. The package version and advertised checkpoint support
remain v0.1 until release qualification.
