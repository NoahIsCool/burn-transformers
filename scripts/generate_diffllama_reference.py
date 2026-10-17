#!/usr/bin/env python3
"""Regenerate the pinned DiffLlama attention goldens of the differential-attention tests.

``src/attention/differential_tests.rs`` compares the crate's
``DifferentialAttention`` with Hugging Face's ``DiffLlamaAttention``
(``layer_idx = 1``) for grouped (two key/value heads), multi-query (one), and
multi-head (four) layouts. This offline maintainer tool recomputes those
outputs with the installed reference and prints the Rust constant
``DIFF_LLAMA_REFERENCES`` in the file's exact literal style, so the goldens
can be audited and regenerated from the tree.

The fixture is fully deterministic and needs no checkpoint:

* configuration: hidden width 8, four query heads of width 2,
  ``rms_norm_eps = 1e-5``, no projection biases, eager attention;
* weights: projection ``p`` (q, k, v, o for seeds 1..=4) holds
  ``values(out * in, seed, 0.05)`` in PyTorch's ``[out, in]`` order, where
  ``values`` is :func:`fixture_values`, evaluated in F64 and rounded to F32
  exactly as the Rust test's helper does;
* lambdas ``q1 = [0.3, -0.2]``, ``k1 = [0.5, 0.4]``, ``q2 = [-0.1, 0.6]``,
  ``k2 = [0.2, -0.3]``;
* hidden states ``values(24, 9, 0.1)`` as ``[1, 3, 8]``, identity RoPE
  (``cos = 1``, ``sin = 0``), and an additive causal mask.

Usage::

    python3 -I scripts/generate_diffllama_reference.py [--check src/attention/differential_tests.rs]

``--check`` compares the regenerated outputs with the constant committed in
the named Rust file as F32 values (see :func:`f32_mismatches`) and exits
nonzero on any difference. The run is refused unless the installed
``transformers`` and ``torch`` are the pinned releases.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

# The reference is built from a local configuration object; never contact a hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

#: Releases the committed goldens were generated with.
PINNED_VERSIONS = {"transformers": "5.2.0", "torch": "2.10.0"}
#: Residual width of the fixture.
HIDDEN_SIZE = 8
#: Query heads of the fixture (paired into two differential heads).
QUERY_HEADS = 4
#: Width of one query/key/value head.
HEAD_DIM = 2
#: Tokens of the single fixture row.
TOKENS = 3
#: DiffLlama's layer index; it sets `lambda_init = 0.8 - 0.6 exp(-0.3)`.
LAYER_INDEX = 1
#: RMS-norm stabilizer of the pair norm (DiffLlama's constructor default).
RMS_NORM_EPS = 1e-5
#: Key/value head counts in the order of the Rust constant: grouped, multi-query, multi-head.
KEY_VALUE_HEADS = (2, 1, 4)
#: Learned lambda vectors, one entry per head channel.
LAMBDAS = {
    "lambda_q1": (0.3, -0.2),
    "lambda_k1": (0.5, 0.4),
    "lambda_q2": (-0.1, 0.6),
    "lambda_k2": (0.2, -0.3),
}
#: Projection seeds in the Rust test's order.
PROJECTION_SEEDS = (("q_proj", 1), ("k_proj", 2), ("v_proj", 3), ("o_proj", 4))
#: Scale of projection weights and of the hidden states.
WEIGHT_SCALE = 0.05
HIDDEN_SCALE = 0.1
#: Seed of the hidden states.
HIDDEN_SEED = 9
#: Name of the Rust constant this tool regenerates.
CONSTANT = "DIFF_LLAMA_REFERENCES"


def to_f32(value: float) -> float:
    """Rounds a Python float (F64) to the nearest F32, as Rust's `as f32` does."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


def fixture_values(count: int, seed: int, scale: float) -> list[float]:
    """Returns the fixture sequence, evaluated in F64 and rounded to F32.

    Entry ``i`` is ``(((37 i + 11 seed) mod 23) · 0.1 - 1.1) · scale / 0.1``,
    the same expression as the Rust test helper ``values``.
    """

    out = []
    for index in range(count):
        base = ((index * 37 + seed * 11) % 23) * 0.1 - 1.1
        out.append(to_f32(base * scale / 0.1))
    return out


def rust_float(value: float) -> str:
    """Formats a golden value as the committed Rust literal: nine significant digits.

    Fraction digits are grouped in threes with underscores (Clippy's
    ``unreadable_literal``), trailing zeros dropped, and integral values keep
    one fractional digit so the literal stays an `f64`.
    """

    text = f"{value:.9g}"
    if "e" in text or "E" in text:
        text = f"{value:.9f}".rstrip("0")
    if "." not in text:
        text += ".0"
    sign = "-" if text.startswith("-") else ""
    whole, fraction = text.lstrip("-").split(".")
    groups = [fraction[start : start + 3] for start in range(0, len(fraction), 3)]
    return f"{sign}{whole}.{'_'.join(groups)}"


def rust_constant(references: Sequence[tuple[int, Sequence[float]]]) -> str:
    """Renders `(key_value_heads, outputs)` pairs as the Rust `DIFF_LLAMA_REFERENCES` constant."""

    lines = [f"const {CONSTANT}: [(usize, [f64; {HIDDEN_SIZE * TOKENS}]); {len(references)}] = ["]
    for heads, outputs in references:
        lines.append("    (")
        lines.append(f"        {heads},")
        lines.append("        [")
        lines.extend(f"            {rust_float(value)}," for value in outputs)
        lines.append("        ],")
        lines.append("    ),")
    lines.append("];")
    return "\n".join(lines)


def committed_constant(rust_source: str) -> str:
    """Extracts the committed `DIFF_LLAMA_REFERENCES` constant from a Rust test file."""

    start = rust_source.index(f"const {CONSTANT}:")
    end = rust_source.index("\n];", start) + len("\n];")
    return rust_source[start:end]


def parse_constant(constant: str) -> list[tuple[int, list[float]]]:
    """Parses a `DIFF_LLAMA_REFERENCES` constant into `(key_value_heads, outputs)` pairs."""

    body = constant.split("= [", 1)[1]
    references = []
    for entry in body.split("(")[1:]:
        heads_text, values_text = entry.split(",", 1)
        values_text = values_text.split("[", 1)[1].split("]", 1)[0]
        values = [float(token.replace("_", "")) for token in values_text.split(",") if token.strip()]
        references.append((int(heads_text.strip()), values))
    return references


def f32_mismatches(
    committed: Sequence[tuple[int, Sequence[float]]],
    regenerated: Sequence[tuple[int, Sequence[float]]],
) -> list[str]:
    """Lists every entry whose committed and regenerated values are different F32 numbers.

    The goldens are F32 outputs printed with nine significant digits, which
    identify an F32 value uniquely but are not unique strings: two literals
    within half an F32 ulp denote the same output. Comparing the rounded F32
    values therefore checks the goldens exactly, independent of formatting.
    """

    problems = []
    if [heads for heads, _ in committed] != [heads for heads, _ in regenerated]:
        return ["key/value head layouts differ"]
    for (heads, left), (_, right) in zip(committed, regenerated):
        if len(left) != len(right):
            problems.append(f"kv={heads}: {len(left)} committed values, {len(right)} regenerated")
            continue
        for index, (old, new) in enumerate(zip(left, right)):
            if to_f32(old) != to_f32(new):
                problems.append(f"kv={heads}[{index}]: committed {old!r}, regenerated {new!r}")
    return problems


def check_versions(installed: dict[str, str]) -> None:
    """Refuses to run against reference releases other than the pinned ones."""

    for package, pinned in PINNED_VERSIONS.items():
        version = installed.get(package)
        if version is None or version.split("+")[0] != pinned:
            raise SystemExit(f"{package} {version} is installed; the goldens are pinned to {pinned}")


def evaluate(key_value_heads: int, torch: Any, modeling: Any, config_class: Any) -> list[float]:
    """Runs `DiffLlamaAttention` on the fixture and returns its `[1, 3, 8]` output, flattened."""

    config = config_class(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=QUERY_HEADS,
        num_key_value_heads=key_value_heads,
        head_dim=HEAD_DIM,
        rms_norm_eps=RMS_NORM_EPS,
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    layer = modeling.DiffLlamaAttention(config, layer_idx=LAYER_INDEX).eval()
    with torch.no_grad():
        for name, seed in PROJECTION_SEEDS:
            linear = getattr(layer, name)
            rows, columns = linear.weight.shape
            weight = fixture_values(rows * columns, seed, WEIGHT_SCALE)
            linear.weight.copy_(torch.tensor(weight, dtype=torch.float32).reshape(rows, columns))
        for name, entries in LAMBDAS.items():
            getattr(layer, name).copy_(torch.tensor(entries, dtype=torch.float32))
        hidden = torch.tensor(
            fixture_values(HIDDEN_SIZE * TOKENS, HIDDEN_SEED, HIDDEN_SCALE), dtype=torch.float32
        ).reshape(1, TOKENS, HIDDEN_SIZE)
        cos = torch.ones(1, TOKENS, HEAD_DIM, dtype=torch.float32)
        sin = torch.zeros(1, TOKENS, HEAD_DIM, dtype=torch.float32)
        future = torch.triu(torch.ones(TOKENS, TOKENS, dtype=torch.bool), diagonal=1)
        mask = torch.zeros(1, 1, TOKENS, TOKENS, dtype=torch.float32)
        mask = mask.masked_fill(future, torch.finfo(torch.float32).min)
        output = layer(hidden, (cos, sin), attention_mask=mask)[0]
    return [float(value) for value in output.reshape(-1).tolist()]


def main(argv: Sequence[str] | None = None) -> int:
    """Prints the regenerated constant, or checks it against a committed Rust file."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--check",
        type=Path,
        help="Rust test file whose committed constant must equal the regenerated one",
    )
    arguments = parser.parse_args(argv)

    import torch
    import transformers
    from transformers import DiffLlamaConfig
    from transformers.models.diffllama import modeling_diffllama

    check_versions({"transformers": transformers.__version__, "torch": torch.__version__})
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    references = [
        (heads, evaluate(heads, torch, modeling_diffllama, DiffLlamaConfig))
        for heads in KEY_VALUE_HEADS
    ]
    constant = rust_constant(references)
    if arguments.check is None:
        print(constant)
        return 0
    committed = parse_constant(committed_constant(arguments.check.read_text(encoding="utf-8")))
    problems = f32_mismatches(committed, references)
    if problems:
        print(f"{arguments.check}: {CONSTANT} differs from the regenerated reference:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(constant, file=sys.stderr)
        return 1
    count = sum(len(values) for _, values in references)
    print(
        f"{arguments.check}: all {count} {CONSTANT} values equal transformers "
        f"{transformers.__version__} / torch {torch.__version__} at F32"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
