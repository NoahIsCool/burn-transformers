#!/usr/bin/env python3
"""Plot training-only learning curves and the untouched held-out comparison."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=HERE.parents[2] / "target/mechanistic/tuned_lens_training")
    args = parser.parse_args()
    report = json.loads((args.root / "result.json").read_text())
    layers = report["layers"]
    colors = ["#176B9C", "#BE6519", "#438749"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
    for layer, color in zip(layers, colors):
        losses = np.asarray([point["training_kl"] for point in layer["curve"]])
        axes[0].plot(np.arange(1, len(losses) + 1), losses, color=color, alpha=0.18, linewidth=0.8)
        averaged = np.convolve(losses, np.ones(12) / 12, mode="valid")
        axes[0].plot(np.arange(12, len(losses) + 1), averaged, color=color,
                     label=f"Stage {layer['stage']}")
    axes[0].set(title="New translators learn from frozen activations",
                xlabel="Optimizer update\nLines: 12-update mean; faint: individual batches", ylabel="Training KL (nats/token)")
    axes[0].legend(frameon=False)
    positions = np.arange(len(layers))
    for axis, metric, title in [(axes[1], "kl", "Held-out teacher → lens KL"),
                                 (axes[2], "ce", "Held-out next-token cross entropy")]:
        axis.bar(positions - 0.18, [layer["initial"][metric] for layer in layers],
                 width=0.36, color="#B8BFC5", label="Initial / raw lens")
        axis.bar(positions + 0.18, [layer["final"][metric] for layer in layers],
                 width=0.36, color=colors, label="Trained lens")
        axis.set(xticks=positions, xticklabels=[f"Stage {layer['stage']}" for layer in layers],
                 title=title, ylabel="Nats/token")
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Tuned Lens training with Burn · 384 training tokens, 128 held-out tokens",
                 fontweight="bold", fontsize=13)
    for extension in ("png", "svg"):
        fig.savefig(args.root / f"training.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
