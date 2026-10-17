# Transformer experiments

Start with [activation_workbench.rs](examples/activation_workbench.rs). It builds
a tiny transformer, captures one token's neurons, ablates and restores them,
inserts a small module, and enables bidirectional attention. No downloads are
needed:

```sh
cargo run --release --locked --example activation_workbench
```

The examples below are fragments using an initialized model, input, and backend
`B`. The workbench supplies the complete setup. For a trained circuit, read the
[grokking experiment](examples/mechanistic/grokking.md).

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

## Capture what you need

Choose a computation point, then optionally narrow it. Indices start at zero;
selecting one token or head keeps that dimension with length one.

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

```rust
hooks.observe(&neurons, |x| println!("mean: {:?}", x.mean().into_scalar()));
```

Here `into_scalar()` explicitly transfers a result to the host. Register all
actions before the forward call. Reusing hooks reruns the actions and replaces
their captures; a failed call discards partial and previous captures.

## Change an activation or patch another run

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
for a callback that returns the crate's `Result`. Missing points, invalid axes,
out-of-range selections, and incompatible replacements return contextual errors.
To change a block's width, compose modules with explicit adapters instead of
returning a different shape from a tensor edit.

## Insert your own processing

An edit can borrow an ordinary Burn module:

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

Concrete supported models expose `activation_points()`, returning names and
axis layouts. Auto model enums return `Result<Vec<PointInfo>>`; after loading,
use `loaded.model().activation_points()?` and
`loaded.model().forward_with_hooks(...)`. Unsupported families report an error.

| Address | Tensor meaning |
| --- | --- |
| `Stack::Encoder.embedding()` / `Stack::Decoder.final_norm()` | Embedded or final normalized `[batch, token, channel]` states. |
| `Layer::encoder(i)` / `Layer::decoder(i)` | Stack and zero-based block index. |
| `.residual_pre()` / `.residual_post()` | Residual before / after the complete block. |
| `.attention_input()` / `.attention_output()` | Mixer input after any pre-normalization / output before residual addition. |
| `.mlp_input()` / `.mlp_output()` | Feed-forward input / output before residual addition. |
| `.mlp_pre_activation()` / `.mlp_activation()` | Dense projection before / after its nonlinearity. In gated MLPs these expose the gate branch **before multiplication**. |
| `.self_attention().query()` / `.key()` / `.value()` | `[batch, head, token, channel]`; Q/K include the caller's position and normalization transforms. K/V use native heads for grouped-query attention and expanded heads for MLA. |
| `.self_attention().head_context()` | Per-query-head context before the output gate and output projection. |
| `.self_attention().scores()` / `.pattern()` | `[batch, head, query, key]`; scaled/softcapped scores before bias/masking / effective softmax weights. |

Decoder cross-attention uses `.cross_attention()` and the corresponding
`cross_attention_input/output()` points. Select with `.token(i)`, `.head(i)`,
`.batch(i)`, `.channels(start..end)`, or `.slice(Axis::Token, start..end)`.
Attention matrices use `.query_token(i)` and `.key_token(i)` instead of `.token(i)`.
For example, on a full-attention layer, ablate one head before its output projection:

```rust
hooks.edit(Layer::decoder(0).self_attention().head_context().head(0), |x| {
    x.zeros_like()
});
```

Catalogs omit unavailable internals: DeltaNet exposes mixer boundaries, and
sparse MoE exposes outer feed-forward boundaries rather than dense-neuron points.

## Change attention and use another task head

Execution stages make a prefix and a replacement task head explicit:

```rust
let mut state = encoder.embed(input)?;
state = encoder.run_layers(state, 0..1)?;
state.hidden = state.hidden.clone() + adapter.forward(state.hidden);
state = encoder.run_layers(state, 1..3)?;
state = encoder.normalize(state)?;
let predictions = classifier.forward(state.hidden);
```

`HiddenState` carries positions, padding, the next layer index, and whether final
normalization has happened. Running a range never adds embeddings or final
normalization. Ranges must continue from `next_layer`; normalized states cannot
resume. You may normalize a prefix deliberately for a new task head. Each
`_with_hooks` call has a catalog scoped to that stage; register only points that
the requested stage executes.

Custom and T5 encoder stages take an owned input; Qwen's `embed` borrows it.
Custom seq2seq provides `embed_encoder`, `run_encoder_layers`,
`normalize_encoder`, and corresponding decoder methods. Pass a normalized encoder
state's `into_encoder_context()?` to `embed_decoder` to preserve source positions
and padding. State carries execution metadata, not parameters or decode caches.

Change attention through that context before running the relevant stages:

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
Absent overrides preserve the original architecture's conventions. Recurrent
DeltaNet rejects attention policies it cannot implement before callbacks run.

Custom full-attention models support `forward_bidirectional` and
`forward_bidirectional_with_hooks`. Padding remains excluded; future tokens
become visible. For classification or another task, `forward_hidden` and
`forward_hidden_bidirectional` return normalized states without computing
vocabulary logits; pool them and apply your own head. Both have `_with_hooks`
variants. T5 and custom seq2seq models provide `encode_with_hooks` for encoder
states. Qwen exposes its text backbone and individual block `forward` methods;
staged full-attention Qwen layers accept explicit attention policies.

For an unpadded batch, a custom sentiment head can use mean pooling:

```rust
let hidden = model.forward_hidden_bidirectional(input)?;
let pooled = hidden.mean_dim(1).squeeze_dim::<2>(1);
let class_logits = classifier.forward(pooled);
```

For padded batches, exclude padded positions from the pooling sum and divisor.

Inspecting grouped-query attention scores or patterns requires an explicit
`AttentionPrecisionMode::EagerFp32Softmax` choice. It materializes the full
quadratic attention matrix, so select short sequences. Custom models expose
`with_attention_precision` for this choice; standalone attention uses
`GroupedQueryAttentionConfig::with_precision_mode`. Qwen attention modules have
a public `precision_mode` field. Check the resulting activation catalog.

```rust
use transformers::attention::AttentionPrecisionMode;
let model = model.with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
```

Q/K/V and head-context hooks preserve the configured attention dispatch.
Switching precision changes the numerical execution graph; compare an
intervention with a clean run using the **same** precision setting.

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

## Edit a recipe and choose which weights to reuse

`T5EncoderRecipe::from_config(config)?` preserves T5's normalization,
relative-position bias, feed-forward convention, and ordering. Edit
`recipe.layers[i].attention_heads`, `.head_dim`, or `.feed_forward_width`, and
resize the layer vector before `init`. An unchanged recipe preserves the shared
relative-bias graph. A new head count needs a compatible bias table; the recipe
reuses the earliest compatible table or initializes one explicitly.

Construct a new architecture before transferring existing parameters:

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
explicit transformations. `transfer::parameters(&model)` discovers module paths.

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

## Extend the computation

Implement `HiddenTransform<B>` on an ordinary `#[derive(Module)]` type to define
your own sequence mixer or feed-forward computation. `ResidualBlock` composes
the two with explicit normalization, including `without_normalization`.
`WidthAdapter` owns input/output linear projections around an inner module:
`WidthAdapter::new(block, 8, 12, 6, &device)?` maps eight channels into a
twelve-channel block and returns six. These projections participate in Burn
parameter traversal, records, and gradients. The custom composition example
shows the entire forward path and a mixer with no attention operation.

In your own module, declare additional typed points and list them in its catalog:

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

## Evaluate in double precision

To bound F32 rounding when measuring small numerical differences, cast every
floating parameter to F64 on Flex, for example with a Burn `ModuleMapper` whose
`map_float` returns `param.map(|tensor| tensor.cast(DType::F64))`. Embeddings,
projections, SwiGLU, backend attention, and vocabulary heads then run in F64.
Rounding-sensitive shared operations follow `precision::at_least_f32`: RMS
normalization, explicit eager softmax, and rotary cosine/sine tables are
evaluated in F64 for F64 activations, while F16, BF16, and F32 keep their FP32
graphs bit for bit. Model families build rotary tables for their activation
dtype; when rotating your own tensors, call `rotary_angles_for_dtype` with their
dtype instead of the F32-only `rotary_angles`.

A few components still evaluate selected arithmetic in F32 by design and are
not covered by this rule: sparse-MoE routing (router probabilities everywhere,
plus FP32 expert weighting or accumulation in Qwen3-MoE, DeepSeek-V2, and
Nemotron-H), Gated DeltaNet recurrences, and Nemotron-H Mamba2 state. Choose
dense attention and SwiGLU layers when every operation must be F64.

## Current boundaries

Interventions remain stateless; hooks do not modify cached generation or recurrent
decode state. Changed bidirectional computation requires a fresh full-sequence
execution. Recurrent architectures keep their explicit capability restrictions.

T5 padded decoder hooks use the original rectangular token indices, even with
left padding or holes. Invalid token slots stay zero at declared activation
points before observers and after edits; logical positions follow valid-token
order. Self/cross-attention still excludes padded keys. This makes token selection
stable without exposing compacted per-row indices.

The v0.2 interfaces passed their [recorded validation](acceptance/measurements/v0_2-interface-validation-20260910-retry/README.md).
The v0.3 [training contract](TRAINING.md) identifies gradient, mode, record,
and precision boundaries. The package version and advertised checkpoint support
remain v0.1 until release qualification.
