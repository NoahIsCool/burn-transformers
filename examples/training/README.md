# Train the architecture examples

These small CPU tasks turn the v0.2 composition demonstrations into actual learning
experiments. Each starts from random weights, uses a plain Burn optimizer loop, and
needs no downloads. Running the parent example without `--train` still runs its
original architecture and intervention demonstration.

```sh
cargo run --release --locked --offline --features autodiff --example t5_classifier -- --train
cargo run --release --locked --offline --features autodiff --example custom_composition -- --train
cargo run --release --locked --offline --features autodiff --example mla_encoder_decoder -- --train
```

Default features supply `std` and `flex`; with `--no-default-features`, enable
`std,flex,autodiff`. Dependencies must already be cached when using `--offline`.

| Example | What learns | Held-out task |
| --- | --- | --- |
| [T5 classifier](t5_classifier.rs) | Three encoder blocks with 2/4/1 attention heads and 32/48/24 MLP widths, a stage-inserted adapter, and a two-class head | Positive/negative adjective classification in two sentence templates; noun/adjective combinations are held out |
| [Custom composition](custom_composition.rs) | GQA, a sequence-mean mixer, learned width boundaries 8 → 12 → 6, and a head | Whether a three-token sequence has a warm-color majority; token sequences are held out |
| [MLA encoder-decoder](mla_encoder_decoder.rs) | Encoder MLA, decoder MLA, cross-attention MLA, a compressed-state adapter, and a count head | Two known decoder query tokens ask for warm/cool counts in a held-out source pair |

The tasks test training and composition, not real-world sentiment quality or a
published scientific result. All words or symbols occur during training. Custom
composition is intentionally permutation invariant: disjoint sequences can share
a multiset. The MLA split holds out both orders of each selected source pair.

| Task | Seed | Updates | Optimizer | Train / held-out rows |
| --- | ---: | ---: | --- | ---: |
| T5 | 73 | 100 | Adam, learning rate `0.01` | 24 / 8 |
| Custom mixer | 89 | 120 | Adam, learning rate `0.01` | 52 / 12 |
| MLA | 101 | 600 | AdamW, learning rate `0.003`, weight decay `0.1` | 51 / 13 |

MLA reports accuracy over both query answers per source pair. The weight decay
helps this small network generalize beyond the observed pairs. These are fixed
example settings, not a claim of robustness across initialization seeds.

## Read the training loop

Each file keeps its optimizer loop explicit:

1. Build an `Autodiff<Flex<f32, i32>>` model and a deterministic train/test split.
2. Run the registered Burn module and compute `transformers::training::cross_entropy`.
3. Call `loss.backward()`, `GradientsParams::from_grads`, and `adam.step` (AdamW with weight decay for MLA).
4. Evaluate with `model.valid()` on the non-autodiff backend.
5. Save the trained module with Burn's `BinFileRecorder`, reconstruct the same
   architecture, and load its record.

T5 inserts its adapter between public encoder stages and pools only valid tokens
(`true` means valid). The custom mixer adds a small feature penalty through
`capture_with_grad`; ordinary captures are detached. MLA borrows its registered
adapter in a temporary `hooks.edit` callback. The adapter remains owned by the
model, so Adam and records include it.

The MLA training decoder uses **bidirectional attention** because its inputs are
two known query tokens. Neither answer is fed into the decoder. Its default
architecture demo keeps causal decoder attention. No positional encoding is
needed for these order-independent source counts.

## Outputs and checks

Each run writes `model.bin` and `result.json` under
`target/training/<example>/`, replacing the previous run there. The report records
seed, optimizer settings, update count, split sizes, training and held-out scores, elapsed
time, parameter changes, and reload error. These are model records; this example
does not save optimizer state for resuming training.

All runs require finite nonzero first-step gradients and measured changes in
representative parameters from each intended component, held-out loss below half
its initial value, accuracy at least 87.5%, and reloaded logits within `1e-6`.
The thresholds are smoke checks for these deliberately simple tasks.

Save the trained module, **before converting it to `valid()`**: Burn 0.21 drops
parameter layout mappers in that conversion. Saving the evaluation copy and
loading it into a fresh checkpoint-layout module can apply a checkpoint transpose
to weights already in runtime layout.
The record also does not define architecture or the callback; recreate both in
Rust, as each reload does here.

```sh
cargo test --release --locked --offline --features autodiff \
  --example t5_classifier --example custom_composition --example mla_encoder_decoder
```
