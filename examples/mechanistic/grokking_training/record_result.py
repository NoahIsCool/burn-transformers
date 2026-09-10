#!/usr/bin/env python3
"""Preserve one completed, fixed-protocol Burn run without copying its weights."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import re
import shlex
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transition(rows: list[dict], field: str, threshold: float) -> dict | None:
    for index, row in enumerate(rows):
        if row[field] > threshold:
            return {"lower_exclusive_step": rows[index - 1]["step"] if index else None,
                    "upper_inclusive_step": row["step"], "accuracy": row[field]}
    return None


def preserve(run: Path, stdout: Path, timing: Path, destination: Path) -> dict:
    evidence = destination / "evidence"
    if (destination / "result.json").exists():
        raise ValueError("result.json already exists; preserve the original result")
    evidence.mkdir(parents=True, exist_ok=True)
    run_info = json.loads((run / "run.json").read_text())
    rows = json.loads((run / "progress.json").read_text())
    outcome = json.loads((run / "outcome.json").read_text())
    inventory = json.loads((run / "source-inventory.json").read_text())
    checkpoint = json.loads((run / "latest.json").read_text())
    if checkpoint["step"] != 40000 or rows[-1]["step"] != 40000:
        raise ValueError("record the complete 40,000-update protocol")
    if not outcome["delayed_generalization_observed"]:
        raise ValueError("retain failed run separately; do not record it as a successful reproduction")
    for name in ["run.json", "progress.json", "outcome.json", "learned-circuit.json", "source-inventory.json"]:
        shutil.copyfile(run / name, evidence / name)
    shutil.copyfile(stdout, evidence / "stdout.log")
    shutil.copyfile(timing, evidence / "timing.log")

    # The executed source remains readable in this small archive, including the
    # pre-hardening trainer and original zero-intercept diagnostic. No weights.
    archive = evidence / "executed-source.tar.gz"
    with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for path in sorted((run / "source-snapshot").rglob("*")):
                if not path.is_file():
                    continue
                data = path.read_bytes()
                info = tarfile.TarInfo(str(path.relative_to(run / "source-snapshot")))
                info.size = len(data)
                info.mode = 0o644
                info.mtime = 0
                tar.addfile(info, io.BytesIO(data))

    def metrics(row: dict) -> dict:
        return {name: row[name] for name in ["step", "train_accuracy", "test_accuracy", "train_loss", "test_loss"]}

    initial, final = metrics(rows[0]), metrics(rows[-1])
    timing_text = timing.read_text()
    stdout_text = stdout.read_text()
    real_seconds = float(re.search(r"([0-9.]+) real", timing_text).group(1))
    resident_bytes = int(re.search(r"(\d+)  maximum resident set size", timing_text).group(1))
    footprint_bytes = int(re.search(r"(\d+)  peak memory footprint", timing_text).group(1))
    loop_seconds = float(re.search(r"40000 updates in ([0-9.]+)s", stdout_text).group(1))
    settings = run_info["settings"]
    command = shlex.join(["/usr/bin/time", "-l", "target/release/examples/grokking_train",
                         "--backend", settings["backend"], "--steps", str(settings["steps"]),
                         "--output", str(settings["output"])])
    if settings["seed"] != 0:
        command += " --seed " + str(settings["seed"])
    features = "autodiff,metal" if settings["backend"] == "metal" else "autodiff"
    memorization = next(row for row in rows if row["train_accuracy"] > 0.99 and row["test_accuracy"] < 0.5)
    final_directory = Path(checkpoint["directory"])
    artifact_hashes = {"initial_model": sha256(run / "initial-model.bin"),
                       "final_model": sha256(final_directory / "model.bin"),
                       "final_optimizer": sha256(final_directory / "optimizer.bin")}
    if artifact_hashes["initial_model"] == artifact_hashes["final_model"]:
        raise ValueError("the recorded model did not change")
    result = {
        "status": "completed_from_scratch",
        "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "claim": "One fixed-seed Burn run reproduces delayed generalization and a causal learned Fourier subspace; not original weights, seed-aggregate curves or Table 1 entries.",
        "protocol_sha256": sha256(HERE / "manifest.json"),
        "run_protocol_sha256": run_info["protocol_sha256"],
        "training_command": command,
        "settings": settings,
        "build_command": "cargo build --release --locked --offline --features " + features + " --example grokking_train",
        "training_binary_sha256": inventory["binary_sha256"],
        "backend": "Burn 0.21 Autodiff<" + settings["backend"].title() + ">, F32",
        "updates": 40000,
        "initial": initial,
        "memorization_observation": metrics(memorization),
        "training_accuracy_above_99_percent": transition(rows, "train_accuracy", .99),
        "heldout_accuracy_above_99_percent": transition(rows, "test_accuracy", .99),
        "final": final,
        "grokking_criterion_passed": True,
        "performance": {"wall_seconds_including_initialization_records_and_analysis": real_seconds,
                        "training_loop_seconds": loop_seconds,
                        "maximum_resident_set_bytes": resident_bytes,
                        "peak_memory_footprint_bytes": footprint_bytes,
                        "scope": "one local Apple Silicon Metal run; no portable throughput claim"},
        "model_and_optimizer_sha256": artifact_hashes,
        "learned_circuit": json.loads((run / "learned-circuit.json").read_text()),
        "source_provenance": {
            "snapshot_timing": inventory["capture_timing"],
            "captured_utc": inventory["captured_utc"],
            "modified_after_binary_build": [row["path"] for row in inventory["files"] if row["modified_after_binary_build"]],
            "drift_explanation": "Only two test-source files changed before snapshot; neither is compiled into this example. The training graph remained frozen through the complete run.",
            "archive": "evidence/executed-source.tar.gz",
            "archive_sha256": sha256(archive),
        },
        "post_run_changes": [
            "Resume now rejects earlier step horizons, benchmark state, and changed protocol/training-source fingerprints.",
            "Read-only --analyze loads the learned checkpoint and writes a separate replay report.",
            "The legacy zero-intercept FVE diagnostic is retained; an explicit affine fit was added after detecting nonzero activation means. This does not change training, frequency selection, or intervention results.",
        ],
        "retained_evidence": {str(path.relative_to(destination)): sha256(path) for path in sorted(evidence.iterdir()) if path.is_file()},
    }
    if result["protocol_sha256"] != result["run_protocol_sha256"]:
        raise ValueError("frozen manifest changed after training")
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    result = preserve(args.run, args.stdout, args.timing, args.output)
    print(json.dumps({"final": result["final"], "archive": result["source_provenance"]}, indent=2))


if __name__ == "__main__":
    main()
