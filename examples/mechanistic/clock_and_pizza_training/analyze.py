#!/usr/bin/env python3
"""Inspect a freshly Burn-trained model: verified arrays, PCA, q, and readable plots."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

P, HIDDEN, NEURONS = 59, 128, 512


def load(directory):
    result = json.loads((directory / "result.json").read_text())
    arrays = {}
    shapes = {"logits": (P, P, P), "embedding": (P, HIDDEN),
              "attention": (P, P, 4, 2, 2), "neurons": (P, P, NEURONS)}
    for name, shape in shapes.items():
        raw = (directory / f"{name}.f32").read_bytes()
        if hashlib.sha256(raw).hexdigest() != result["artifacts"][name]["sha256"]:
            raise ValueError(f"{name}: artifact SHA-256 mismatch")
        arrays[name] = np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float64)
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name}: nonfinite activations")
    return result, arrays


def geometry(arrays):
    centered = arrays["embedding"] - arrays["embedding"].mean(axis=0)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u * singular
    spectrum = np.abs(np.fft.rfft(scores, axis=0)) ** 2
    frequencies = spectrum[1:].argmax(axis=0) + 1
    circularities = 2 * spectrum[frequencies, np.arange(P)] / (P * (scores * scores).sum(axis=0))
    correct = arrays["logits"][np.arange(P)[:, None], np.arange(P)[None, :],
                               (np.arange(P)[:, None] + np.arange(P)[None, :]) % P]
    by_difference = np.empty((P, P))
    for difference in range(P):
        a = np.arange(P)
        b = (a + difference) % P
        by_difference[difference, (a + b) % P] = correct[a, b]
    q = by_difference.std(axis=1).mean() / correct.std()
    return {"distance_irrelevance": float(q),
            "circularity_first_four_pcs": float(circularities[:4].mean()),
            "dominant_frequency_first_six_pcs": frequencies[:6].tolist(),
            "circularity_first_six_pcs": circularities[:6].tolist(),
            "variance_fraction_first_six_pcs": ((singular[:6] ** 2) / (singular ** 2).sum()).tolist(),
            "paper_circular_threshold": 0.995,
            "interpretation": "q and circularity are descriptions, not forced alpha-derived mechanism labels"}, scores, by_difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("target/mechanistic/clock_and_pizza_training/analysis"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    reports = []
    for directory in args.directories:
        result, arrays = load(directory)
        report, scores, by_difference = geometry(arrays)
        if abs(report["distance_irrelevance"] - result["measurements"][-1]["distance_irrelevance"]) > 1e-10:
            raise AssertionError("independent NumPy and Rust q disagree")
        symmetry_path = directory / "gradient_symmetry.json"
        if symmetry_path.exists():
            report["gradient_symmetricity"] = json.loads(symmetry_path.read_text())["mean_cosine_similarity"]
        report["options"] = result["options"]
        report["final_measurement"] = result["measurements"][-1]
        reports.append(report)
        runs.append((result, arrays, scores, by_difference))
    (args.output / "analysis.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(runs), 5, figsize=(19, 4 * len(runs)), squeeze=False)
    for row, (result, arrays, scores, differences) in enumerate(runs):
        alpha = result["options"]["alpha"]
        history = result["measurements"]
        axes[row, 0].plot([m["step"] for m in history], [m["train_accuracy"] for m in history], label="train")
        axes[row, 0].plot([m["step"] for m in history], [m["test_accuracy"] for m in history], label="held out")
        axes[row, 0].set(xlabel="Burn AdamW update", ylabel="Accuracy", ylim=(0, 1.03), title=f"Attention alpha={alpha:g}")
        axes[row, 0].set_xscale("symlog", linthresh=100)
        axes[row, 0].set_xlim(0, history[-1]["step"])
        axes[row, 0].legend()
        axes[row, 1].scatter(scores[:, 0], scores[:, 1], c=np.arange(P), cmap="hsv", s=24)
        axes[row, 1].set(xlabel="Embedding PC1", ylabel="Embedding PC2", title="Learned token geometry", aspect="equal")
        im = axes[row, 2].imshow(differences, aspect="auto", origin="lower")
        axes[row, 2].set(xlabel="Sum a+b (mod59)", ylabel="Difference b-a (mod59)", title=f"Correct logit; q={reports[row]['distance_irrelevance']:.3f}")
        fig.colorbar(im, ax=axes[row, 2], shrink=.7)
        # A fixed first head, last query: selection is predeclared, not chosen for a striking map.
        im = axes[row, 3].imshow(arrays["attention"][:, :, 0, 1, 0], origin="lower", aspect="auto", vmin=0, vmax=1)
        axes[row, 3].set(xlabel="Second token b", ylabel="First token a", title="Head0: final query → first key")
        fig.colorbar(im, ax=axes[row, 3], shrink=.7)
        # Order the 32 most variable neurons using training activations, then show all input differences.
        neurons = arrays["neurons"]
        pairs = json.loads((args.directories[row] / "protocol.json").read_text())["train_pairs"]
        top = np.argsort(neurons.reshape(P * P, NEURONS)[pairs].var(axis=0))[-32:]
        activity = np.stack([neurons[np.arange(P), (np.arange(P) + d) % P].mean(axis=0) for d in range(P)])
        im = axes[row, 4].imshow(activity[:, top], origin="lower", aspect="auto")
        axes[row, 4].set(xlabel="32 neurons ranked by training variance", ylabel="Difference b-a (mod59)", title="Captured last-token ReLU activity")
        fig.colorbar(im, ax=axes[row, 4], shrink=.7)
    fig.suptitle("Random initialization → Burn training → activation inspection", fontsize=17)
    fig.tight_layout()
    for extension in ["png", "svg"]:
        fig.savefig(args.output / f"training_and_activations.{extension}", dpi=150)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for index, (result, _, _, _) in enumerate(runs):
        history = result["measurements"]
        steps = [m["step"] for m in history]
        color = f"C{index}"
        label = f"alpha={result['options']['alpha']:g}"
        axes[0].plot(steps, [m["train_accuracy"] for m in history], "--", color=color, label=label + " train")
        axes[0].plot(steps, [m["test_accuracy"] for m in history], color=color, label=label + " held out")
        axes[1].plot(steps, [m["test_cross_entropy"] for m in history], color=color, label=label)
        axes[2].plot(steps, [m["distance_irrelevance"] for m in history], color=color, label=label)
    for axis, label in zip(axes, ["Accuracy", "Held-out cross entropy (nats)", "Distance irrelevance q"]):
        axis.set(xlabel="Burn AdamW update", ylabel=label)
        axis.set_xscale("symlog", linthresh=100)
        axis.set_xlim(left=0)
        axis.legend(fontsize=8)
    axes[0].set_ylim(0, 1.03)
    axes[1].set_yscale("log")
    fig.suptitle("All recorded measurements: learning and mechanism diagnostics")
    fig.tight_layout()
    for extension in ["png", "svg"]:
        fig.savefig(args.output / f"learning_curves.{extension}", dpi=150)


if __name__ == "__main__":
    main()
