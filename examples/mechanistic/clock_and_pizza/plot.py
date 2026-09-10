#!/usr/bin/env python3
"""Plot the prepared embeddings with Matplotlib; never run inference or download."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("target/mechanistic/clock_and_pizza"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("embedding_circles.png"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), layout="constrained")
    fig.suptitle("Two learned algorithms, circular token embeddings", fontsize=16, weight="bold")
    token_ids = np.arange(59)
    for row, model in enumerate(("pizza", "clock")):
        scores = load_file(str(args.artifacts / f"{model}.safetensors"))["pca.scores"]
        for circle in range(3):
            ax = axes[row, circle]
            x, y = scores[:, 2 * circle:2 * circle + 2].T
            scatter = ax.scatter(x, y, c=token_ids, cmap="twilight_shifted", vmin=0, vmax=58, s=31)
            for token in (0, 1, 2, 3, 4):
                ax.annotate(str(token), (x[token], y[token]), xytext=(4, 4),
                            textcoords="offset points", fontsize=8, weight="bold")
            ax.set_title(f"{model.capitalize()} · PC {2 * circle + 1}–{2 * circle + 2}", fontsize=11)
            ax.set_xlabel(f"Principal component {2 * circle + 1}")
            ax.set_ylabel(f"Principal component {2 * circle + 2}")
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(scatter, ax=axes, label="Token value (0–58)", shrink=0.75, pad=0.025)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    print(args.output)


if __name__ == "__main__":
    main()
