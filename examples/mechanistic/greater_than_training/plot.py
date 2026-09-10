#!/usr/bin/env python3
"""Plot the frozen learned-replacement endpoint without selecting runs or prompts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CONDITIONS = ["clean", "corrupt", "oracle", "initial_student", "trained_student"]
LABELS = ["Clean", "Corrupt", "Oracle patch", "Initial student", "Trained student"]
COLORS = ["#276c9e", "#a74f4f", "#8157a0", "#77858c", "#24836e"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=HERE / "result.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    report = json.loads(args.result.read_text())
    expected_protocol = digest(HERE / "protocol.json")
    if (report["protocol_sha256"] != expected_protocol or report["smoke_only"]
            or not report["engineering_checks_passed"]
            or not report["source_unchanged_during_execution"]):
        raise ValueError("Require a successful full result from the frozen protocol")
    training = report["training"]
    if training["steps"] != 256 or len(training["loss_history"]) != 256:
        raise ValueError("Require the fixed 256-update endpoint")
    losses = np.asarray(training["loss_history"])
    if not np.isfinite(losses).all():
        raise ValueError("Nonfinite training loss")
    held_out = [row for row in report["rows"] if row["split"] == "held_out"]
    if any(sum(row["condition"] == condition for row in held_out) != 97
           for condition in CONDITIONS):
        raise ValueError("Require every one of the 97 held-out prompts")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axis = axes[0, 0]
    axis.plot(np.arange(1, 257), losses, color=COLORS[-1], alpha=0.28, linewidth=0.8)
    axis.plot(np.arange(8, 257), np.convolve(losses, np.ones(8) / 8, "valid"),
              color=COLORS[-1], linewidth=1.8, label="8-update mean")
    axis.set(title="A  Fixed training run", xlabel="Optimizer update",
             ylabel="KL(oracle || student), nats")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.15)

    for axis, metric, title, ylabel in [
        (axes[0, 1], "teacher_kl", "B  Held-out distillation", "KL(oracle || condition), nats"),
        (axes[1, 0], "probability_difference", "C  Held-out Greater-Than behavior", "Probability difference"),
    ]:
        summaries = [report["summary"][f"held_out/{condition}"][metric]
                     for condition in CONDITIONS]
        positions = np.arange(len(CONDITIONS))
        axis.bar(positions, [item["mean"] for item in summaries],
                 yerr=[item["sample_sd"] for item in summaries],
                 color=COLORS, width=0.68, capsize=3,
                 error_kw={"elinewidth": 1, "ecolor": "#41494e"})
        axis.set(title=title, ylabel=ylabel, xticks=positions,
                 xticklabels=["Clean", "Corrupt", "Oracle\npatch", "Initial\nstudent", "Trained\nstudent"])
        axis.axhline(0, color="#525c63", linewidth=0.7)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)

    axis = axes[1, 1]
    offsets = np.arange(-8, 9)
    for condition, label, color in zip(CONDITIONS, LABELS, COLORS):
        rows = [row for row in held_out
                if row["condition"] == condition and 10 <= row["suffix"] <= 89]
        if len(rows) != 80:
            raise ValueError("Require the same 80 cutoff-aligned prompts at every offset")
        curve = np.asarray([[row["numeric_probabilities"][row["suffix"] + int(offset)]
                             for offset in offsets] for row in rows]).mean(axis=0)
        axis.plot(offsets, curve, color=color, linewidth=1.8, label=label,
                  linestyle="--" if condition == "initial_student" else "-")
    axis.axvline(0, color="#525c63", linewidth=0.8, linestyle=":")
    axis.set(title="D  Ending-year probability near the cutoff",
             xlabel="Ending suffix − clean starting suffix", ylabel="Full-vocabulary probability")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.15)
    fig.suptitle("Learned MLP10 replacement in frozen GPT-2", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.018,
             "Clean features explicitly supply year information · one fixed seed · bars: 97 held-out clean prompts, mean ± sample SD\n"
             "Curves: the same 80 prompts at each offset · probabilities are never renormalized over numeric tokens",
             ha="center", fontsize=9, color="#41494e")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.9, hspace=0.48, wspace=0.32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for extension in ("png", "svg"):
        output = args.output_dir / f"training.{extension}"
        fig.savefig(output, dpi=180, metadata={"Creator": "Greater-Than learned-replacement plotting"}
                    if extension == "svg" else None)
        artifacts[output.name] = digest(output)
    plt.close(fig)
    validation = {
        "result_sha256": digest(args.result), "protocol_sha256": expected_protocol,
        "plot_source_sha256": digest(Path(__file__)), "artifacts_sha256": artifacts,
        "matplotlib": matplotlib.__version__, "numpy": np.__version__,
        "training_updates": 256, "held_out_rows_per_condition": 97,
        "cutoff_curve_rows_per_condition_and_offset": 80,
        "error_bars": "descriptive sample standard deviation, not confidence intervals",
        "probabilities": "F64 analysis of F32 full-vocabulary logits; numeric subset not renormalized",
    }
    (args.output_dir / "plot_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
