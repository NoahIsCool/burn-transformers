#!/usr/bin/env python3
"""Plot validated Greater-Than measurements; never run the model or fit a curve.

Run ``python3 examples/mechanistic/greater_than/plot.py`` after the experiment.
The only dependencies are NumPy and Matplotlib. PNG and SVG files are written
beside this script by default. All plotted numbers come from held-out rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
CURVES = {
    "clean": ("Clean prompt", "#245c99", "-"),
    "corrupt": ("Corrupt prompt (01)", "#7b828a", "--"),
    "node_patch_10": ("Clean MLP 10 → corrupt run", "#c66825", "-"),
    "node_patch_all_late": ("Clean MLPs 8–11 → corrupt run", "#6f579b", "-"),
}
COMPARISONS = [
    ("clean", "Clean prompt"),
    ("corrupt", "Corrupt prompt (01)"),
    ("node_patch_2", "Node patch · MLP 2"),
    ("node_patch_8", "Node patch · MLP 8"),
    ("node_patch_9", "Node patch · MLP 9"),
    ("node_patch_10", "Node patch · MLP 10"),
    ("node_patch_11", "Node patch · MLP 11"),
    ("node_patch_all_late", "Node patch · MLPs 8–11"),
    ("direct_10", "Direct readout · MLP 10"),
]


def mean_sd(values):
    """Summarize variation between prompts, not uncertainty in the mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("A summary requires at least two finite observations.")
    return float(values.mean()), float(values.std(ddof=1))


def load_measurements(result_path, manifest_path, fixture_path):
    """Fail closed on unvalidated, incomplete, or mismatched measurements."""
    result = json.loads(result_path.read_text())
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    fixture_bytes = fixture_path.read_bytes()
    fixture = json.loads(fixture_bytes)["entries"]
    if result.get("validation", {}).get("passed") is not True:
        raise ValueError("Refusing to plot: result.validation.passed must be true.")
    if (manifest["protocol"], manifest["activation"], manifest["vocabulary_size"]) != ("greater-than-v1", "gelu", 50257):
        raise ValueError("This figure describes only the frozen exact-GELU GPT-2 protocol.")
    if hashlib.sha256(fixture_bytes).hexdigest() != manifest["fixture_sha256"]:
        raise ValueError("The fixture does not match the frozen manifest hash.")
    for artifact, contents in (("manifest", manifest_bytes), ("fixture", fixture_bytes)):
        if result.get("provenance", {}).get(f"{artifact}_sha256") != hashlib.sha256(contents).hexdigest():
            raise ValueError(f"The result is not bound to this frozen {artifact}.")

    # Key by prompt identity so paired interventions cannot silently be reordered.
    indexed = {}
    for row in result["rows"]:
        index = row["index"]
        if not isinstance(index, int) or not 0 <= index < len(fixture):
            raise ValueError(f"Invalid fixture index: {index!r}.")
        entry = fixture[index]
        if (row["split"], row["suffix"]) != (entry["split"], entry["year"] % 100):
            raise ValueError(f"Prompt {index} has mismatched fixture metadata.")
        key = (index, row["condition"])
        if key in indexed:
            raise ValueError(f"Duplicate measurement: {key}.")
        for metric in ("pd", "sharpness", "numeric_mass"):
            if not math.isfinite(row[metric]):
                raise ValueError(f"Non-finite {metric} at {key}.")
        mass = row["numeric_mass"]
        if not 0 <= mass <= 1 + 1e-10 or max(abs(row["pd"]), abs(row["sharpness"])) > mass + 1e-10:
            raise ValueError(f"Metrics exceed their probability bounds at {key}.")
        probabilities = row.get("probabilities")
        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=np.float64)
            if probabilities.shape != (100,) or not np.isfinite(probabilities).all():
                raise ValueError(f"Expected 100 finite numeric probabilities at {key}.")
            if (probabilities < 0).any() or (probabilities > 1).any():
                raise ValueError(f"Probability outside [0, 1] at {key}.")
            suffix = row["suffix"]
            expected = {
                "numeric_mass": probabilities.sum(),
                "pd": probabilities[suffix + 1 :].sum() - probabilities[: suffix + 1].sum(),
                "sharpness": probabilities[suffix + 1] - probabilities[suffix - 1],
            }
            if expected["numeric_mass"] > 1 + 1e-10:
                raise ValueError(f"Numeric mass exceeds full-vocabulary probability at {key}.")
            for metric, value in expected.items():
                if not math.isclose(row[metric], value, abs_tol=1e-10):
                    raise ValueError(f"Saved {metric} disagrees with probabilities at {key}.")
        indexed[key] = row

    held_out = [i for i, entry in enumerate(fixture) if entry["split"] == "held_out"]
    if len(held_out) != manifest["split_counts"]["held_out"]:
        raise ValueError("Held-out fixture count disagrees with the manifest.")
    for index in held_out:
        for condition in manifest["conditions"]:
            if (index, condition) not in indexed:
                raise ValueError(f"Missing held-out condition: {(index, condition)}.")
        for condition in CURVES:
            if indexed[index, condition].get("probabilities") is None:
                raise ValueError(f"Missing plotted probabilities: {(index, condition)}.")
    return indexed, held_out, manifest


def plot_probabilities(axis, indexed, held_out):
    # This cohort is fixed from suffixes alone, before examining model outputs.
    # Its 80 prompts support EVERY offset; the sample does not change along x.
    cohort = [index for index in held_out if 10 <= indexed[index, "clean"]["suffix"] <= 89]
    if len(cohort) != 80:
        raise ValueError("Expected one held-out prompt for each suffix 10 through 89.")
    offsets = np.arange(-10, 11)
    for condition, (label, color, style) in CURVES.items():
        curves = np.asarray([
            np.asarray(indexed[index, condition]["probabilities"])[
                indexed[index, condition]["suffix"] + offsets
            ]
            for index in cohort
        ])
        mean = curves.mean(axis=0)
        deviation = curves.std(axis=0, ddof=1)
        # No subset renormalization, smoothing, clipping, or logit averaging.
        axis.fill_between(offsets, mean - deviation, mean + deviation, color=color, alpha=0.09)
        axis.plot(offsets, mean, color=color, linestyle=style, linewidth=2, label=label)
    axis.axvline(0.5, color="#414853", linestyle=":", linewidth=1)
    axis.set_xlim(-10, 10)
    axis.set_xticks([-10, -5, 0, 5, 10])
    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    axis.set_xlabel("Completion suffix − original start suffix (YY)")
    axis.set_ylabel("Probability in the full vocabulary")
    axis.set_title("A  Numeric completions around the cutoff", loc="left", fontweight="bold")
    axis.text(0.02, 0.97, "80 held-out prompts · suffixes 10–89", transform=axis.transAxes,
              va="top", fontsize=9, color="#555c66")


def plot_interventions(axis, recovery_axis, indexed, held_out, manifest):
    axis.set_title("B  Effect of replacing an intermediate tensor", loc="left", fontweight="bold")
    positions = np.arange(len(COMPARISONS))
    clean_mean, _ = mean_sd([indexed[i, "clean"]["pd"] for i in held_out])
    corrupt_mean, _ = mean_sd([indexed[i, "corrupt"]["pd"] for i in held_out])
    axis.axvline(clean_mean, color=CURVES["clean"][1], alpha=0.3, linewidth=1)
    axis.axvline(corrupt_mean, color=CURVES["corrupt"][1], alpha=0.4, linestyle="--", linewidth=1)
    recovery_axis.set_title("Paired effect\nmean ± SD", fontsize=10, loc="left", pad=8)
    for position, (condition, _) in zip(positions, COMPARISONS):
        rows = [indexed[i, condition] for i in held_out]
        mean, deviation = mean_sd([row["pd"] for row in rows])
        color = CURVES.get(condition, (None, "#435c75", None))[1]
        if condition == "direct_10":
            color = "#9f4973"
        axis.errorbar(mean, position, xerr=deviation, fmt="o", color=color,
                      markersize=5.5, capsize=3, elinewidth=1.25)

        # A direct-readout replacement starts from CLEAN, so show its signed
        # change from clean. Node recovery starts from CORRUPT and is distinct.
        if condition.startswith("direct_"):
            changes = [row["pd"] - indexed[index, "clean"]["pd"] for index, row in zip(held_out, rows)]
            for row, expected in zip(rows, changes):
                if row.get("recovery") is not None:
                    raise ValueError("Direct-readout effects must not be labelled node recovery.")
                if not math.isclose(row["change_from_clean"], expected, abs_tol=1e-10):
                    raise ValueError("Saved direct-readout change disagrees with the clean baseline.")
            change_mean, change_sd = mean_sd(changes)
            recovery_axis.text(0, position, f"Δ clean: {change_mean:+.2f} ± {change_sd:.2f}",
                               va="center", fontsize=9, color=color)
            continue

        # Recovery is paired per prompt; a ratio of aggregate means is different.
        recoveries = []
        for index, row in zip(held_out, rows):
            clean, corrupt = indexed[index, "clean"]["pd"], indexed[index, "corrupt"]["pd"]
            gap = clean - corrupt
            expected = (row["pd"] - corrupt) / gap if abs(gap) > manifest["numerical_tolerances"]["negligible_gap"] else None
            saved = row.get("recovery")
            if expected is None:
                if saved is not None:
                    raise ValueError("Recovery must be null for a negligible clean–corrupt gap.")
            elif saved is not None:
                if not math.isclose(saved, expected, abs_tol=1e-10):
                    raise ValueError("Saved recovery disagrees with its paired probability differences.")
                recoveries.append(saved)
            elif condition in ("clean", "corrupt"):
                # Baselines may omit their tautological recoveries (1 and 0).
                recoveries.append(expected)
            else:
                raise ValueError(f"Missing defined recovery for {condition}, prompt {index}.")
        undefined = len(rows) - len(recoveries)
        if len(recoveries) >= 2:
            recovery_mean, recovery_sd = mean_sd(recoveries)
            label = f"{recovery_mean:.2f} ± {recovery_sd:.2f}"
        else:
            label = "undefined"
        if undefined:
            label += f" [{undefined} null]"
        recovery_axis.text(0, position, label, va="center", fontsize=9, color=color)

    axis.set_yticks(positions, [label for _, label in COMPARISONS], fontsize=9)
    axis.set_ylim(len(COMPARISONS) - 0.5, -0.6)
    axis.set_xlabel("Probability difference (mean ± sample SD)")
    axis.axhline(7.5, color="#c6ccd2", linewidth=0.8)
    recovery_axis.set_ylim(axis.get_ylim())
    recovery_axis.axis("off")
    axis.text(0, -0.25, f"{len(held_out)} held-out clean prompts · mean clean–corrupt gap: {clean_mean - corrupt_mean:.3f}",
              transform=axis.transAxes, fontsize=9, color="#555c66")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=HERE / "result.json")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--fixture", type=Path, default=HERE / "fixture.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    indexed, held_out, manifest = load_measurements(args.result, args.manifest, args.fixture)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#333a43", "text.color": "#252c35",
        "axes.edgecolor": "#bcc3ca", "xtick.color": "#505964", "ytick.color": "#505964",
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    figure = plt.figure(figsize=(15, 6.2))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.15, 0.88, 0.35], wspace=0.6)
    probability_axis = figure.add_subplot(grid[0])
    effect_axis = figure.add_subplot(grid[1])
    recovery_axis = figure.add_subplot(grid[2])
    plot_probabilities(probability_axis, indexed, held_out)
    plot_interventions(effect_axis, recovery_axis, indexed, held_out, manifest)
    handles, labels = probability_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.065, 0.115),
                  ncol=4, frameon=False, fontsize=9, columnspacing=2)
    for axis in (probability_axis, effect_axis):
        axis.grid(axis="x" if axis is effect_axis else "y", color="#e5e8ec", linewidth=0.7)
        axis.set_axisbelow(True)
    figure.suptitle("Greater-Than behavior and activation interventions", x=0.07, ha="left",
                   fontsize=17, fontweight="bold", y=0.97)
    figure.text(0.07, 0.905, "GPT-2 small · exact GELU · bounded node-patching adaptation and separate direct-readout measurement",
                fontsize=10, color="#555c66")
    figure.text(0.07, 0.035,
                "Bands and bars show descriptive variation across clean prompts (±1 sample SD); corrupted controls can repeat. Layers are zero-based.\n"
                "Node patches replace clean MLP outputs during corrupt inference. Direct readout replaces only MLP 10's contribution to the clean final residual.\n"
                "Paired effects: node recovery = (PD − corrupt PD) / (clean PD − corrupt PD); direct Δ clean = PD − clean PD. Numeric probabilities are never renormalized.",
                fontsize=8.5, linespacing=1.5, color="#555c66")
    figure.subplots_adjust(left=0.07, right=0.985, top=0.8, bottom=0.3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        destination = args.output_dir / f"year_probabilities.{extension}"
        figure.savefig(destination, dpi=180, metadata={"Creator": "Greater-Than plot.py"})
        print(destination)
    plt.close(figure)


if __name__ == "__main__":
    main()
