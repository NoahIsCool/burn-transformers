# Train an adapter inside a frozen Qwen backbone

Run the small CPU example with randomly initialized transformer weights:

```sh
cargo run --release --locked --offline --features autodiff --example train_qwen_adapter
```

To reuse the first two blocks of a local checkpoint:

```sh
cargo run --release --locked --offline --features autodiff --example train_qwen_adapter -- \
  --model-dir /path/to/Qwen3.5-0.8B --steps 40
```

`--output-dir NEW_DIRECTORY` selects where records are written. Otherwise the
example creates a temporary directory and prints its path. Local weights are
read-only and loaded as F32; the pretrained run needs several GiB of memory.

The [training loop](../train_qwen_adapter.rs) owns three ordinary Burn modules:
a frozen backbone, a small residual adapter after block 0, and a two-class head
after block 1. `backbone.no_grad()` freezes parameters while preserving gradients
through its operations. `model.valid()` is used for evaluation after training.

Eight synthetic sequences teach the head to distinguish final token IDs 1/2
from 3/4. Cross entropy trains the classifier. A small squared-activation penalty
uses `capture_with_grad` on eight MLP channels in block 1, allowing an auxiliary
loss to update the earlier adapter. Both gradient routes are checked separately.
These training-set results illustrate optimization; they do not measure sentiment
accuracy, generalization, or pretrained-model quality.

The run verifies that every frozen parameter byte stays identical and both
trainable modules change. Halfway through, it saves and reloads ordinary Burn
records, then checks that the next Adam update exactly matches uninterrupted
training. The output includes:

- `trainable-midpoint.bin` and `adam-midpoint.bin`: the resumed checkpoint.
- `trainable-final.bin` and `adam-final.bin`: the completed run's parameters and optimizer.
- `backbone_recipe.json`: architecture needed alongside the original frozen weights.
- `report.json`: losses, gradient norms, frozen-parameter hashes, and resume checks.

Reconstruct the same backbone and freezing policy when loading these records.
The small trainable record contains the adapter and replacement head; reuse the
original frozen source separately. The default seed-42 run reduced task loss
from 0.5842 to 0.02075 in 60 steps, fitting all eight synthetic examples.
