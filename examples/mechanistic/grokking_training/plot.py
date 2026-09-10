#!/usr/bin/env python3
"""Plot a Burn grokking run; read local measurements without training or downloads.

    python3 examples/mechanistic/grokking_training/plot.py

The five plotted frequencies are selected from the final observed readout power.
This is a retrospective description of the learned circuit, not a prediction.
"""

import argparse
import json
import math
from pathlib import Path


def load_progress(path):
    """Require ordered measurements and probability-like normalized powers."""
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("progress must contain a nonempty list of observations")
    previous = -1
    for row in rows:
        step = row["step"]
        if not isinstance(step, int) or isinstance(step, bool) or step <= previous:
            raise ValueError("observation steps must be nonnegative and strictly increasing")
        previous = step
        for name in ("train_loss", "test_loss", "train_accuracy", "test_accuracy"):
            value = row[name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name.endswith("accuracy") and value > 1:
                raise ValueError("accuracies must be fractions in [0, 1]")
        for name in ("embedding_fourier_power", "readout_fourier_power"):
            power = row[name]
            if len(power) != 56 or any(not math.isfinite(x) or x < 0 for x in power):
                raise ValueError(f"{name} must contain 56 finite nonnegative frequency powers")
            total = math.fsum(power)
            # All-zero weights have no nonzero Fourier energy to normalize.
            if total != 0 and not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"{name} must sum to one, or zero for an empty spectrum")
    return rows


def strongest_frequencies(power, count=5):
    """Return one-based frequency indices, ranking energy then smaller index."""
    return sorted(range(1, len(power) + 1), key=lambda k: (-power[k - 1], k))[:count]


def delayed_generalization(rows):
    """Apply the frozen strict thresholds in temporal order to saved observations."""
    memorization = next(
        (row["step"] for row in rows if row["train_accuracy"] > 0.99 and row["test_accuracy"] < 0.50),
        None,
    )
    generalization = next(
        (row["step"] for row in rows
         if memorization is not None and row["step"] > memorization and row["test_accuracy"] > 0.99),
        None,
    )
    return memorization, generalization


def plot(rows, output):
    # Import Matplotlib only when rendering; protocol tests need no extra packages.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    fig.subplots_adjust(left=0.065, right=0.965, bottom=0.25, top=0.79, wspace=0.32)
    fig.suptitle("Grokking from random initialization", fontsize=17, weight="bold", y=0.98)
    steps = [row["step"] for row in rows]
    colors = {"train": "#185c9d", "test": "#c35d17"}
    for split in ("train", "test"):
        axes[0].semilogy(steps, [max(row[f"{split}_loss"], 1e-12) for row in rows],
                         label=split.capitalize(), color=colors[split], linewidth=1.8)
        axes[1].plot(steps, [100 * row[f"{split}_accuracy"] for row in rows],
                     label=split.capitalize(), color=colors[split], linewidth=1.8)
    axes[0].set(title="Loss on every train and test pair", ylabel="Cross entropy (nats, log scale)")
    axes[0].legend(frameon=False)
    axes[1].set(title="Delayed generalization", ylabel="Accuracy (%)", ylim=(-2, 102))
    axes[1].axhline(99, color="0.5", alpha=0.5, linewidth=0.8, linestyle=":")
    axes[1].legend(frameon=False)
    frequencies = strongest_frequencies(rows[-1]["readout_fourier_power"])
    frequency_colors = ["#365eaa", "#d77823", "#279777", "#955fa6", "#bc5262"]
    for frequency, color in zip(frequencies, frequency_colors):
        for field, style in (("embedding_fourier_power", "--"), ("readout_fourier_power", "-")):
            axes[2].plot(steps, [100 * row[field][frequency - 1] for row in rows],
                         color=color, linestyle=style, linewidth=1.5,
                         label=f"k = {frequency}" if style == "-" else None)
    axes[2].set(title="Five strongest final readout frequencies", ylabel="Nonzero Fourier power (%)")
    highest_power = max(row[field][frequency - 1] for row in rows for frequency in frequencies
                        for field in ("embedding_fourier_power", "readout_fourier_power"))
    axes[2].set_ylim(0, max(1, 130 * highest_power))
    frequency_legend = axes[2].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    axes[2].add_artist(frequency_legend)
    axes[2].legend(handles=[Line2D([0], [0], color="0.3", label="Readout"),
                            Line2D([0], [0], color="0.3", linestyle="--", label="Embedding")],
                   frameon=False, fontsize=8, loc="upper right")
    for ax in axes:
        ax.set_xlabel("Completed Burn optimizer updates")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_xlim(left=0, right=max(steps[-1], 1))
    memorization, generalization = delayed_generalization(rows)
    if generalization is None:
        status = "The preregistered delayed-generalization criterion has not been observed."
    else:
        status = (f"Recorded memorization: update {memorization:,}; "
                  f"recorded later generalization: update {generalization:,}.")
    completion = f"Measurements through update {steps[-1]:,}"
    fig.text(0.5, 0.875, f"{completion} · 3,830 training pairs / 8,939 held-out pairs", ha="center", fontsize=10)
    fig.text(0.065, 0.125, status, fontsize=10)
    fig.text(0.065, 0.077, "Frequencies are selected from the last measurement. Spectra exclude the constant component and equals-token class.",
             fontsize=9, color="0.3")
    if any(row[name] == 0 for row in rows for name in ("train_loss", "test_loss")):
        fig.text(0.065, 0.031, "Zero measured losses are displayed at 10⁻¹² on the logarithmic axis.", fontsize=8, color="0.3")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        destination = output.with_suffix(suffix)
        fig.savefig(destination, dpi=180, facecolor="white")
        print(destination)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("progress", type=Path, nargs="?", default=Path("target/mechanistic/grokking-training/progress.json"))
    parser.add_argument("--output", type=Path, help="Output stem or PNG/SVG path; defaults beside the measurements")
    args = parser.parse_args()
    plot(load_progress(args.progress), args.output or args.progress.with_name("training_progress.png"))


if __name__ == "__main__":
    main()
