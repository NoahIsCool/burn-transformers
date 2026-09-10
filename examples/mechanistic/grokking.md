# Grokking: train and interpret a Fourier circuit

Start with [the from-scratch Burn training experiment](grokking_training/README.md)
to reproduce delayed generalization and interpret the circuit that run learns.
This page documents the complementary archived-checkpoint experiment, which
reproduces the paper's exact Table 1 measurements.

[Nanda et al., *Progress measures for grokking via mechanistic interpretability*
(ICLR 2023)](https://arxiv.org/abs/2301.05217) explain how a small transformer learns
addition modulo 113. Its neurons represent sines and cosines of the input sum.
This example reproduces **Table 1's ten measurements** from the authors' trained
checkpoint, then demonstrates a related intervention through public typed hooks.

## Run

From the repository root, with the default Flex features:

```sh
python3 examples/mechanistic/grokking_prepare.py
cargo run --release --locked --example grokking_fourier
```

The preparation command needs only Python's standard library. It downloads
910,593 bytes and writes a 908,300-byte checkpoint under ignored `target/mechanistic/`.
The Rust program runs entirely locally, evaluates all **12,769** input pairs,
and checks its results with assertions. No GPU, tokenizer, dataset, or training
run is required. Two Apple Silicon CPU executions took 0.30–0.53 seconds and
19–25 MB peak RSS, excluding compilation/download; other hosts will differ.

## Read the experiment

Start with [grokking_fourier.rs](../grokking_fourier.rs). The steps are:

1. Compose the MLP output and unembedding matrices to obtain the neuron-to-logit map.
2. Extract its cosine/sine directions at the paper's frequencies `{14,35,41,42,52}`.
3. Capture the 512 post-ReLU MLP activations. Measure how well each direction's
   activation follows `cos(2πk(a+b)/113)` or `sin(2πk(a+b)/113)`.
4. Orthonormalize the ten directions. A callback keeps their projection, removes
   it, or restores it, before executing the model's ordinary output layers.

[GrokkingTransformer](grokking_model.rs) contains the import plumbing and the
one-block architecture: four attention heads, width 128, a 512-neuron ReLU MLP,
learned positions, no LayerNorm. It uses this crate's public grouped-query
attention. The final `=` query can see all three input tokens; computing only
that query is equivalent to the full causal forward pass for this single layer.

The model declares `grokking.last_token.mlp.activation` as a rank-two
`[batch, channel]` point. The experiment uses the library's `Hooks::capture`,
`edit`, and `get`; `continue_with_hooks` provides the enclosing scope and emits
that point before the ordinary readout. Shape, dtype, device, action ordering,
and capture lifetime follow the [shared activation contract](../../EXPERIMENTS.md).
This illustrates adding a custom point without a separate callback framework.

## Results and their limits

Table 1 reports 93.2–98.2% explained variance. Our ten results are within **0.09
percentage points** of those printed values; assertions allow 0.2 points.
[grokking_fourier.result.json](grokking_fourier.result.json) records all values.
The [v0.2 replay](v0_2_examples.result.json) records the same experiment after
migration to the public hook API; the original evidence remains intact.

| Additional full-model intervention | Accuracy | Cross entropy |
| --- | ---: | ---: |
| Unchanged | 100% | 2.4122e-7 |
| Keep ten-dimensional subspace | 100% | 2.4048e-7 |
| Remove that subspace | 13.7286% | 3.9795 |
| Remove, then restore | 100% | 2.4122e-7 |

These ablations retain the full residual path and biases. They demonstrate the
subspace's importance, but **do not reproduce §4.4's quoted 5.27 nullspace loss
or 50% loss reduction**. The precise ablation implementation behind those numbers
is absent from the committed analysis notebook; the source of the difference is
not established here. Removing this subspace does not reduce accuracy to chance
(1/113). We report the protocol and observed result without claiming equivalence.

A capture-only pass leaves logits unchanged. Restoring activations recovers
logits within 7.63e-6. Eight complete logit rows agree with an independent NumPy
implementation within 1.53e-5. Its optional regeneration command is:

```sh
python3 examples/mechanistic/grokking_reference.py target/mechanistic/grokking.safetensors > /tmp/grokking-reference.json
```

The archived checkpoint was already trained. This analysis reproduces a
final-state mechanistic result. The [training companion](grokking_training/README.md)
now starts from random parameters and measures delayed generalization separately;
neither experiment establishes seed robustness or every paper figure.

## Checkpoint provenance

The [authors' analysis notebook, commit `23d2c64`](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/23d2c64dd1f8a5ca65efaf27e15c2b2cd47dedf1/Grokking_Analysis.ipynb)
selects `full_run_data['state_dicts'][400]`. Its metadata maps this to epoch 40,000
and seed 0. We use that state from
[author Jess Smith's archive at immutable revision `3799fe7`](https://huggingface.co/smithjessk/grokking-saved-runs/tree/3799fe7b4c6abd4d81df7e55972fbefc6a9f0c3f/original_architecture/deprecated/seed_0).
Despite the directory name `deprecated`, this archive is byte-identical by the
Hub's published SHA-256 to the [named `full_run_data.pth` copy](https://huggingface.co/callummcdougall/grokking_full_run_data/tree/80e5dbb5744dd3e958e45e56f7562a4d50c7400c)
and reproduces the notebook's mainline measurements. Other small deprecated
checkpoints are different runs and are not used.

[grokking_manifest.json](grokking_manifest.json) pins the URL, archive digest,
checkpoint index, tensor shapes, byte offsets, and per-tensor SHA-256 checksums.
The downloader verifies the bounded byte range and every tensor; it does not
fetch/hash the entire 456 MB archive. Archive and metadata hashes are provenance
records; preparation validates the range, tensors, and converted file. No pickle
is loaded or executed. The
converted safetensors SHA-256 is
`28f7212abd53937bc85ee0558536d89b0b882b3cfbec77d6942f605f7d2d4116`,
which the Rust example verifies before loading. Checkpoint weights are not
redistributed in this repository.
