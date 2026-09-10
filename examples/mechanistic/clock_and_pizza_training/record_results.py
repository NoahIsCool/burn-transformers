#!/usr/bin/env python3
"""Preserve small evidence files; leave weights and large activations in ignored storage."""
import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("target/mechanistic/clock_and_pizza_training"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    if (args.output / "result.json").exists():
        raise FileExistsError("Preserve existing result.json; choose another --output for new evidence")
    details = args.output / "results"
    details.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name in ["alpha0-paper", "alpha1-paper"]:
        source = args.source / name
        target = details / name
        target.mkdir()
        for filename in ["protocol.json", "result.json", "learning_curve.json", "gradient_symmetry.json"]:
            shutil.copyfile(source / filename, target / filename)
        for suffix in [".log", "-timing.log"]:
            shutil.copyfile(Path("/tmp") / f"clock-training-{name}{suffix}", target / ("stdout.log" if suffix == ".log" else "timing.log"))
        result = json.loads((source / "result.json").read_text())
        runs[name] = {
            "command": f"target/release/examples/clock_and_pizza_train --backend metal --paper --alpha {result['options']['alpha']:g} --output {source}",
            "exit_code": 0,
            "final_measurement": result["measurements"][-1],
            "capture_max_logit_error": result["capture_max_logit_error"],
            "record_round_trip_max_logit_error": result["record_round_trip_max_logit_error"],
            "ignored_artifact_hashes": {p.name: sha256(p) for p in source.iterdir() if p.is_file()},
            "evidence_hashes": {p.name: sha256(p) for p in target.iterdir() if p.is_file()},
        }
    failed = details / "failed-gradient-smoke"
    failed.mkdir()
    for filename in ["protocol.json", "learning_curve.json"]:
        shutil.copyfile(args.source / "smoke-metal-final" / filename, failed / filename)
    shutil.copyfile(Path("/tmp/clock-training-smoke-metal-final.log"), failed / "stdout-stderr.log")
    analysis = json.loads((args.source / "analysis" / "analysis.json").read_text())
    for filename in ["analysis.json", "training_and_activations.png", "training_and_activations.svg", "learning_curves.png", "learning_curves.svg"]:
        shutil.copyfile(args.source / "analysis" / filename, args.output / filename)
    for name in ["clock-training-final-tests.log", "clock-training-final-clippy.log", "clock-training-final-legacy-replay.log", "clock-training-smoke-flex-final.log"]:
        shutil.copyfile(Path("/tmp") / name, details / name)
    report = {
        "schema_version": 1,
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "paper": "https://arxiv.org/html/2306.17844v2",
        "protocol_sha256": sha256(Path(__file__).with_name("manifest.json")),
        "claim": "Two native Burn training runs from random weights under the published architecture, initialization distributions, optimizer and update budget. New seed/split and F32 loss are explicit adaptations. No seed selection or outcome-filtered retries.",
        "runs": runs,
        "mechanisms": analysis,
        "engineering_trials": [
            {"directory": "benchmark-flex-alpha0", "steps": 20, "purpose": "initial backend benchmark; original batched MLP"},
            {"directory": "benchmark-metal-alpha0", "steps": 20, "purpose": "initial GPU benchmark; original batched MLP; sandbox hid GPU adapter, approved host retry succeeded"},
            {"directory": "benchmark-metal-flatten-alpha0", "steps": 100, "purpose": "equivalent flattened MLP benchmark"},
            {"directory": "benchmark-flex-flatten-alpha0", "steps": 100, "purpose": "portable backend benchmark after equivalent flattening"},
            {"directory": "smoke-metal-final", "steps": 5, "outcome": "failed after updates: reshaped hook capture was not the exact leaf needed for Burn gradient lookup; fixed by retaining the original leaf in the callback before scientific runs"},
            {"directory": "smoke-flex-final", "steps": 5, "outcome": "passed final portable smoke and derivative/record checks; not learning evidence"},
        ],
        "limitations": [
            "Neither fixed run satisfies the paper circularity threshold of 0.995: alpha0=.439215, alpha1=.923548. Both reach 100% task accuracy, but representative circular Clock/Pizza mechanisms are not reproduced. Both runs remain visible.",
            "The paper uses F64 training cross entropy; these Metal runs use differentiable pairwise F32 cross entropy. Recurrent confident-loss and q fluctuations remain in all recorded curves.",
            "The author seeds and exact split memberships are unpublished; a single seed per condition does not estimate the paper's phase diagram.",
            "Historical checkpoint Table 2/q reproductions remain a separate reference experiment.",
        ],
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
