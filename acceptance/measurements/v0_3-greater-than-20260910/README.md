# Greater-Than learned replacement: accepted execution

This record preserves the first full endpoint of the fixed conditioned
learned-replacement protocol. It is a new distillation extension using clean
features inside a corrupted GPT-2 forward, not the original paper's training or
full-model retraining. The clean features explicitly supply year information.

The readable [example and results](../../../examples/mechanistic/greater_than_training/README.md)
explain the experiment. The frozen protocol preceded both training and endpoint
evaluation; its SHA-256 is
`6e3ec845d2e35d2f8c955c4f6a6db699c9fa4c7aa475826a77b9a1a1a6bddd48`.

## Exact execution

From `/Users/nover/repos/transformers`, the preserved wrapper was executed as:

```sh
python3 acceptance/measurements/v0_3-greater-than-20260910/run.py
```

Its actual child command, including compilation, was:

```sh
cargo run --release --locked --offline --features autodiff \
  --example greater_than_train
```

`execution.json` records the timestamp, wall duration, command, source hashes,
and executable hash. `stdout.log` and `stderr.log` preserve the actual output;
`executed-source/` preserves the used example, helper, protocol, fixture, and
selected library/Cargo files. This selected inventory is not an exhaustive
transitive dependency snapshot. The wrapper verified that these sources did not
change during execution. The original binary SHA-256 was
`ef13f7aedb475761ac1d0dfc73da794954980e506e2b504f1ee0351b48a2120b`.
`run.py` is a historical invocation and rejects its existing log files. To repeat
without overwriting evidence, run the Cargo command with
`-- --output-dir target/mechanistic/greater_than_training/repeat-NEW` instead.

Preparation uses the existing authenticated local
`target/mechanistic/greater_than` directory. Checkpoint hashes and pinned revision
are included in the protocol. No network request occurred. Checkpoint weights,
compiled binaries, and generated parameter/optimizer records are not copied into
this evidence directory.

## Observed result and checks

After 256 Adam updates, held-out oracle-to-student KL decreased from 0.56219 to
0.05005 nats over 97 clean prompts. Student probability difference moved from
−0.44608 to 0.27959, against the original clean-output patch's 0.27188. The student
retains a less sharp cutoff than that teacher. It receives clean conditioning;
this does not establish recovery from corrupted inputs alone.

Every frozen backbone parameter byte was unchanged. The student had a finite,
nonzero gradient through the frozen later block and readout, changed parameters,
and exact next-step equality after reloading its own and Adam's native records.
Training took 98.88 seconds, and the complete executable took 136.52 seconds;
Cargo-inclusive wall time was 139.88 seconds. These timings describe this run.

`result.json` retains all 970 condition/prompt rows and the 256 training losses.
`manifest.json` inventories the retained files. Student and Adam record hashes
and sizes are in `result.json`; their local files remain under
`target/mechanistic/greater_than_training/full`. The copied consumer result has
SHA-256 `b55293d1c70888b7803be435d84260ecffcf1a30a3cc9ff7d178c4a51bdb1e43`.

`smoke-result.json` also retains the earlier eight-development-prompt, two-step
smoke, which was used only for cost and
engineering checks. It performed no held-out evaluation and did not change the
predeclared 256-step protocol.
