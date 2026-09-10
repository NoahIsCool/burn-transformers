# Qwen prefix composition validation

This record exercises the v0.2 prefix API against local Qwen3.5-0.8B weights at
revision `2fc06364715b967f1860aea9cf38778875588b17`. It establishes selective
loading and parity between two execution paths in this crate. It does not
establish new checkpoint support, task accuracy, independent mathematical
correctness, or performance qualification.

The preserved `compare_prefix.rs` is the exact temporary harness executed for
the original measurement. It loads a two-block recipe with F32 conversion,
captures the second block's activated MLP channels at token position 2, and
normalizes its output. It separately loads the complete text model, executes
its first two public blocks individually, applies its final norm, and asserts
exact equality of all 3,072 hidden values. Both paths use fixed IDs `[1, 2, 3]`.

From the repository root, repeat both the public example and the comparison:

```sh
python3 acceptance/measurements/qwen-prefix-v02-20260910/replay.py /path/to/Qwen3.5-0.8B
```

The runner authenticates the three checkpoint files against the recorded
SHA-256 hashes, builds with `--release --locked --offline`, resolves the actual
Cargo library paths, and compiles the preserved harness with `rustc -C opt-level=3`.
It writes fresh execution evidence to a new temporary directory. Dependencies
must already be cached; the checkpoint is read-only. Allow several GiB of memory
for the prefix and complete F32 text model.

`source_metadata.json` records checkpoint and relevant source-file hashes.
`harness_build.json` and `comparison_execution.json` retain the actual original
commands; their historical temporary and library paths are not prerequisites
for replay. `comparison_report.json` retains the full parameter disposition
report. The sibling summary JSON lists artifact hashes.

The direct example exited successfully. An earlier `/usr/bin/time -l` wrapper
returned an error because sandboxed macOS `sysctl` metadata was unavailable;
that attempt is retained in `example_execution.json`. Reported elapsed times
come from the successful direct run and harness. They are single local smoke
measurements without a controlled benchmark protocol.
