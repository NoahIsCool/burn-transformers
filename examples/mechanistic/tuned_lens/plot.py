#!/usr/bin/env python3
"""Plot a successful Rust Tuned Lens run; never run inference or download.

Requires Matplotlib. From the repository root:
    python3 examples/mechanistic/tuned_lens/plot.py
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_RESULT = HERE.parents[2] / "target/mechanistic/tuned_lens/rust_result.json"


def load_result(path):
    """Refuse failed checks or incompatible data before creating any figure."""
    result = json.loads(path.read_text())
    parity, controls = result["parity"], result["controls"]
    if parity.get("passed") is not True or controls.get("passed") is not True:
        raise ValueError("Plotting requires a Rust run with passing parity and controls")
    limits = [
        (parity["maximum_logit_error"], 0.005),
        (parity["maximum_metric_error"], 0.0002),
        (controls["observation_logit_error"], 1e-6),
        (controls["zero_translator_logit_error"], 1e-6),
    ]
    if any(not math.isfinite(value) or value < 0 or value > limit for value, limit in limits):
        raise ValueError("Recorded numerical checks exceed the predeclared limits")
    metrics = result["metrics"]
    if (metrics["input_tokens"], metrics["ce_tokens"]) != (512, 508):
        raise ValueError("This figure describes the fixed four-record, 128-token adaptation")
    if [layer["index"] for layer in metrics["layers"]] != list(range(12)):
        raise ValueError("Expected twelve translator boundaries in order")
    values = [metrics["final_ce"]] + [
        layer[name] for layer in metrics["layers"]
        for name in ("raw_kl", "tuned_kl", "raw_ce", "tuned_ce")
    ]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Expected finite, nonnegative losses in nats per token")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=HERE / "prediction_quality.png")
    args = parser.parse_args()
    result = load_result(args.result)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": "#334155", "text.color": "#172A3A",
        "xtick.color": "#526375", "ytick.color": "#526375",
        "svg.fonttype": "none", "svg.hashsalt": "tuned-lens-bounded-study",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.7))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.23, top=0.72, wspace=0.22)
    fig.suptitle("Reading intermediate predictions with Tuned Lens", fontsize=18, weight="bold", y=0.965)
    fig.text(0.5, 0.906, "Pythia-160M-v0  ·  4 records × 128 tokens  ·  released lens, no fitting",
             ha="center", fontsize=11)
    fig.text(0.5, 0.861, "Bounded short-context adaptation of the paper’s comparison",
             ha="center", fontsize=10, color="#526375")

    layers = result["metrics"]["layers"]
    indices = [layer["index"] for layer in layers]
    for ax, metric, title, label in zip(
        axes, ("kl", "ce"), ("Match to the final model", "Next-token prediction"),
        ("KL(final ∥ lens) · nats/token", "Cross-entropy · nats/token"),
    ):
        raw, = ax.plot(indices, [layer[f"raw_{metric}"] for layer in layers],
                       color="#64758D", marker="o", markersize=4.8, linewidth=2.1,
                       label="Raw logit lens")
        tuned, = ax.plot(indices, [layer[f"tuned_{metric}"] for layer in layers],
                         color="#087F8C", marker="D", markersize=4.4, linewidth=2.4,
                         label="Tuned Lens")
        ax.set(title=title, xlabel="Translator index (0 = embeddings)", ylabel=label)
        ax.set_xticks(indices)
        ax.set_xlim(-0.3, 11.3)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#E0E6ED", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E1")
        ax.tick_params(axis="both", length=3)

    baseline = result["metrics"]["final_ce"]
    final = axes[1].axhline(baseline, color="#A45A25", linestyle=(0, (4, 3)),
                           linewidth=1.5, label=f"Final model CE ({baseline:.3f})")
    fig.legend(handles=[raw, tuned, final], loc="upper center", bbox_to_anchor=(0.5, 0.827),
               ncol=3, frameon=False, handlelength=2.5, columnspacing=2.5)
    fig.text(0.075, 0.125, "Lower is better.  KL uses all 512 input positions; CE uses 508 next-token targets.", fontsize=10)
    fig.text(0.075, 0.081, "Translator 0 reads token embeddings; translators 1–11 read outputs of blocks 0–10.",
             fontsize=9.5, color="#526375")
    fig.text(0.075, 0.040, "Measured in Rust / Burn Flex, F32. Independent reference parity and intervention controls passed.",
             fontsize=9, color="#526375")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(args.result.read_bytes()).hexdigest()
    description = f"Four-record short-context adaptation. Source rust_result.json SHA-256: {source_hash}"
    fig.savefig(args.output, dpi=180, facecolor="white",
                metadata={"Title": "Tuned Lens prediction quality", "Description": description})
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white",
                metadata={"Title": "Tuned Lens prediction quality", "Description": description,
                          "Creator": "plot.py", "Date": None})
    plt.close(fig)
    print(f"Plotted verified Rust results: {args.output}")


if __name__ == "__main__":
    main()
