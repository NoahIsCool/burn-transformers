# Qwen adapter training smoke

The manifest preserves executed source snapshots, binary and artifact hashes,
actual commands, and separate random/pretrained reports. Both runs use native
Burn Adam to train an inserted adapter and replacement classifier through a later
frozen Qwen block. Every frozen parameter byte stays identical, task and auxiliary
gradient routes remain connected, and a restored optimizer update matches exactly.

These are eight-example synthetic training-task checks. They provide no held-out
quality, sentiment, new checkpoint-support, or benchmark qualification. The real
checkpoint identity is pinned in the preceding Qwen prefix validation record.

From the repository root, with cached dependencies:

```sh
cargo run --release --locked --offline --features autodiff --example train_qwen_adapter
cargo run --release --locked --offline --features autodiff --example train_qwen_adapter -- \
  --model-dir models/qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17 --steps 40
```

Fresh small trainable-module and optimizer records are written to the printed
temporary directory. Their hashes are retained in the manifest; parameter
binaries and original backbone weights are not included here. Source files were
unchanged during both accepted executions. The full guide is
`examples/training/qwen_adapter.md`.
