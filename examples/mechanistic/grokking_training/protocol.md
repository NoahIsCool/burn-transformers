# Grokking training protocol

This experiment starts from random parameters and trains modular addition with
Burn. Its target is **delayed generalization**, followed by Fourier analysis of
the model it actually trained. The [frozen manifest](manifest.json) defines the
run before its outcomes; it is distinct from the archived-checkpoint
[Table 1 analysis](../grokking.md).

## Author configuration

The [paper, Section 3](https://arxiv.org/html/2301.05217v3#S3) specifies addition
modulo 113, a 30% training split, one transformer block, width 128, four 32-wide
heads, and a 512-wide ReLU MLP. Positions are learned; there is no LayerNorm,
dropout, or embedding/readout tying. Both residual connections remain present.
The paper describes 40,000 epochs; the released script defaults to 50,000.

The [pinned author training source](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/23d2c64dd1f8a5ca65efaf27e15c2b2cd47dedf1/transformers.py)
provides the remaining choices:

- Train on all 114 logits, including the equals token. Drop that class only for
  the later 113-class Fourier analysis.
- Initialize weights with zero-mean normal samples, standard deviation
  `1 / sqrt(128)`. The untied unembedding uses `1 / sqrt(114)` instead. In
  particular, the MLP output matrix uses the residual width, not the MLP width.
- Initialize both MLP biases to zero and train them. Attention has no biases.
- Use full-batch AdamW: learning rate `0.001`, weight decay `1`, betas
  `(0.9, 0.98)`, default epsilon `1e-8`; decay includes all learned parameters.
- Warm up for ten updates. The zero-based update `u` uses
  `0.001 * min(u / 10, 1)`, matching the author's optimizer-then-scheduler order.
  The first update accumulates moments with zero learning rate.

## Deterministic split and initialization

[train_split.json](train_split.json) preserves the 3,830 ordered training indices
from Python's `Random(0).shuffle` of lexicographic pairs. Decode an index as
`a = index / 113`, `b = index % 113`. The 8,939 remaining pairs form the entire
test set; their evaluation order does not affect the metric. The manifest pins
the fixture's SHA-256 and generator version.

Burn's initialization seed is independently fixed at zero. The author's
`Config.seed` seeds the split, but the pinned trainer does not call
`torch.manual_seed`; matching its initialized weights is therefore not claimed.

## Measurement and numerical boundaries

Run 40,000 optimizer updates without early stopping. Evaluate both complete
splits at update zero, every 100 updates through 1,000, then every 500 updates.
Record accuracy, cross entropy, and normalized embedding/readout Fourier power.
Select the learned five strongest nonzero Fourier frequency pairs after training;
do not substitute the archived model's frequencies.

For each frequency `k = 1..56`, pair power adds the squared orthonormal cosine
and sine coefficients, summed across the other matrix axis. Normalize by total
nonzero-frequency power, excluding the equals-token class and constant component.
Rank these powers in `W_mlp_out @ W_unembed[:, 0..113]` to select the five
frequencies; break equal-power ties toward smaller `k`.

The preregistered delayed-generalization criterion requires a recorded point
with training accuracy above 99% and test accuracy below 50%, followed by a
later point with test accuracy above 99%. Report transition intervals at the
measurement resolution, actual final results, and unsuccessful runs. This is
one-run evidence, not reproduction of the paper's five-seed aggregate.

Forward parameters and arithmetic use F32; reported cross entropy uses F64
analysis of F32 logits. The author's notebook uses F64 `log_softmax`, while its
[separate helper](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/23d2c64dd1f8a5ca65efaf27e15c2b2cd47dedf1/helpers.py#L96)
actually casts to F32 despite its F64 comment. Our differentiable loss and backend
choice are recorded explicitly. Ten-update backend timing probes may choose
execution hardware, but the measured run begins afresh with this configuration.
