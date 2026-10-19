"""Assemble the October 18 Qwen3.5-0.8B re-baseline record from its campaign directory.

Usage: python3 -I build_record.py CRATE_ROOT PROSE_JSON

Reads acceptance/measurements/qwen3.5-0.8b-metal-rebaseline-20261018/ (the
campaign, build, and precondition evidence) with the crate's
scripts/qwen35_benchmark_protocol.py, recomputes every statistic from the raw
reports, and writes acceptance/measurements/qwen3.5-0.8b-metal-rebaseline-20261018.json.
PROSE_JSON supplies the reviewed prose fields (measured, inferred, and the
like); every number in the record comes from the evidence.

The October 19 review amendment rebuilt the record with the corrected
steady-state rule, the binary identity in the reference block, the compiler in
the series identity, and the cold-first-token view without the samples that
followed discarded attempts. It also evaluates the original rule with the
campaign's own script copy (harness/qwen35_benchmark_protocol.py), start by
start, and records the head's medians over the original window beside the
corrected ones. No measurement file changed.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

NAME = "qwen3.5-0.8b-metal-rebaseline-20261018"
SEPTEMBER = "small-model-performance-20260910"
SEPTEMBER_SUMMARIES = {"baseline": "qwen-controlled-current-summary.json", "public": "qwen-public-summary.json"}
#: Files the October 19 review amendment added; every other non-harness file is evidence from the first version.
AMENDMENT_FILES = ("analysis/analysis-20261019.log",)
HISTORICAL = ("v0_1-release-20260910.json", f"{SEPTEMBER}/manifest.json",
              f"{SEPTEMBER}/qwen-controlled-current-summary.json", f"{SEPTEMBER}/qwen-public-summary.json",
              "half-precision-review-20261018.json", "half-precision-envelope-20261018.json",
              "half-precision-followups-20261017.json")


def sha256(path):
    """Hexadecimal SHA-256 of one file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field(text, key):
    """The value of an ``== key value`` line of a build log."""
    for line in text.splitlines():
        if line.startswith(f"== {key} "):
            return line[len(key) + 4:]
    raise KeyError(key)


def load(path, name):
    """Import a protocol script under ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_rule(protocol, original, samples, script):
    """The campaign script's original steady-state rule evaluated start by start, and the head's medians under it.

    The original rule took the earliest window of at least
    ``original.STEADY_WINDOW_ROUNDS`` final rounds whose half-medians agree
    within ``original.STEADY_DRIFT``; it never checked later windows.
    """
    rounds = max(sample["round"] for sample in samples)
    cells = {}
    for sample in samples:
        cells.setdefault((sample["label"], sample["benchmark"]), []).append(sample)
    values = {}
    for (label, kind), runs in sorted(cells.items()):
        runs = sorted(runs, key=lambda run: run["round"])
        for metric in protocol.gated_metric_names(kind):
            values[(label, kind, metric)] = [run["metrics"]["gated"][metric]["median"] for run in runs]
    starts = {}
    for first in range(1, rounds - original.STEADY_WINDOW_ROUNDS + 2):
        drifting = {}
        for key, series in values.items():
            window = series[first - 1:]
            half = len(window) // 2
            drift = protocol.median(window[-half:]) / protocol.median(window[:half]) - 1.0
            if abs(drift) > original.STEADY_DRIFT:
                drifting["/".join(key)] = drift
        starts[str(first)] = drifting
    chosen = original.summarize(samples)["steady_state"]
    earliest = next((int(first) for first, drifting in starts.items() if not drifting), None)
    if chosen["first_round"] != earliest:
        raise SystemExit(f"original rule mismatch: script {chosen['first_round']}, recomputed {earliest}")
    medians = {kind: {metric: protocol.median(values[("head", kind, metric)][chosen["first_round"] - 1:])
                      for metric in protocol.gated_metric_names(kind)} for kind in protocol.BENCHMARKS}
    counts = {}
    for sample in samples:
        if sample["round"] >= chosen["first_round"]:
            cell = counts.setdefault(f"{sample['label']}/{sample['benchmark']}", {})
            cell[str(sample["position"])] = cell.get(str(sample["position"]), 0) + 1
    counts = {cell: dict(sorted(positions.items())) for cell, positions in sorted(counts.items())}
    return {"script": "harness/qwen35_benchmark_protocol.py", "script_sha256": sha256(script),
            "window_rounds_minimum": original.STEADY_WINDOW_ROUNDS, "drift_tolerance": original.STEADY_DRIFT,
            "first_round": chosen["first_round"], "rounds": chosen["rounds"],
            "maximum_absolute_drift": chosen["maximum_absolute_drift"],
            "drifting_by_start": starts, "position_counts": counts, "head_medians": medians}


def main():
    crate, prose = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    protocol = load(crate / "scripts/qwen35_benchmark_protocol.py", "protocol")
    measurements = crate / "acceptance/measurements"
    root = measurements / NAME
    campaign = root / "campaign"
    original = load(root / "harness/qwen35_benchmark_protocol.py", "original")
    plan = json.loads((campaign / "plan.json").read_text())
    samples = protocol.load_campaign(campaign)
    summary = protocol.summarize(samples)
    summary["series"] = protocol.campaign_series(samples)
    reference = protocol.reference_block(summary, "head")
    control = protocol.reference_block(summary, "september")
    before = original_rule(protocol, original, samples, root / "harness/qwen35_benchmark_protocol.py")
    reference_change = {
        kind: {metric: {"original_window": before["head_medians"][kind][metric],
                        "corrected_window": reference["benchmarks"][kind]["gated"][metric]["median"],
                        "corrected_over_original": reference["benchmarks"][kind]["gated"][metric]["median"]
                        / before["head_medians"][kind][metric]}
               for metric in protocol.gated_metric_names(kind)}
        for kind in protocol.BENCHMARKS}
    rows = []
    for planned, loaded in zip(plan["samples"], samples):
        host = json.loads((campaign / planned["host"]).read_text())
        report = json.loads((campaign / planned["report"]).read_text())
        provenance = report["provenance"]
        rows.append({
            "index": planned["index"], "round": planned["round"], "position": planned["position"],
            "label": planned["label"], "benchmark": planned["benchmark"], "test": planned["test"],
            "report": f"campaign/{planned['report']}", "report_sha256": sha256(campaign / planned["report"]),
            "log": f"campaign/{planned['log']}", "host_record": f"campaign/{planned['host']}",
            "exit": host["exit"], "wall_seconds": host["wall_seconds"],
            "crate_commit": provenance["crate_commit"]["value"],
            "crate_worktree_status": provenance["crate_worktree_status"]["value"],
            "started_utc": host["host_before"]["utc"], "ended_utc": host["host_after"]["utc"],
            "admission": host["admission"],
            "loadavg_before": host["host_before"]["loadavg"], "loadavg_after": host["host_after"]["loadavg"],
            "power_before": host["host_before"]["power"].splitlines()[0],
            "metal_cache_growth_bytes": host["metal_cache_growth"]["growth_bytes"],
            "cold_first_token_seconds": loaded["metrics"]["diagnostic"]["cold_first_token"],
            "gated": loaded["metrics"]["gated"],
            "tokens": loaded["tokens"],
            "discarded_attempts": loaded["discarded_attempts"],
        })
    binaries = []
    for binary, directory in zip(plan["binaries"], ("head-fa8426e", "september-c70057e")):
        preflight = (root / "build" / directory / "preflight.txt").read_text()
        result = (root / "build" / directory / "result.txt").read_text()
        binaries.append({
            "label": binary["label"], "source_commit": field(preflight, "commit"),
            "src_tree": field(preflight, "src_tree"), "worktree": field(preflight, "tree"),
            "dirty_before_build": int(field(preflight, "dirty")), "dirty_after_build": int(field(result, "dirty_after")),
            "cargo_lock_sha256": field(preflight, "cargo_lock_sha256"),
            "cargo_toml_sha256": field(preflight, "cargo_toml_sha256"),
            "toolchain": "rustc 1.92.0 (ded5c06cf 2025-12-08), cargo 1.92.0 (344c4567c 2025-10-21)",
            "features": ["std", "metal", "tokenizers", "chat-templates"], "profile": "release",
            "command": (root / "build" / directory / "command.txt").read_text().strip(),
            "target_directory": field(preflight, "target"),
            "path": binary["binary"], "sha256": binary["sha256"], "bytes": int(field(result, "bytes")),
            "build_log": f"build/{directory}/result.txt",
        })
    september = {}
    for kind, name in SEPTEMBER_SUMMARIES.items():
        data = json.loads((measurements / SEPTEMBER / name).read_text())["benchmark"]
        metrics = protocol.run_metrics(data, kind)
        september[kind] = {"summary": f"{SEPTEMBER}/{name}", "sha256": sha256(measurements / SEPTEMBER / name),
                           "gated": {key: value["median"] for key, value in metrics["gated"].items()},
                           "cold_first_token": metrics["diagnostic"]["cold_first_token"],
                           "tokens": protocol.token_identity(data, kind)}
    comparisons = {}
    for kind in protocol.BENCHMARKS:
        rows_kind = {}
        for metric in protocol.gated_metric_names(kind) + ["cold_first_token"]:
            if metric == "cold_first_token":
                head = reference["benchmarks"][kind]["diagnostic"]["cold_first_token"]["median"]
                base = control["benchmarks"][kind]["diagnostic"]["cold_first_token"]["median"]
                approved = september[kind]["cold_first_token"]
            else:
                head = reference["benchmarks"][kind]["gated"][metric]["median"]
                base = control["benchmarks"][kind]["gated"][metric]["median"]
                approved = september[kind]["gated"][metric]
            rows_kind[metric] = {"head": head, "september_rebuild": base, "september_10_approved": approved,
                                 "head_over_september_rebuild": head / base,
                                 "head_over_september_10": head / approved,
                                 "september_rebuild_over_september_10": base / approved}
        comparisons[kind] = rows_kind
    hosts = sorted(campaign.glob("host/*.host.json"))
    host_records = [json.loads(path.read_text()) for path in hosts]
    contaminated = []
    for path, host in zip(hosts, host_records):
        if host.get("clean") is False:
            stem = path.name.removesuffix(".host.json")
            contaminated.append({
                "host_record": f"campaign/host/{path.name}", "log": f"campaign/logs/{stem}.log",
                "report": f"campaign/bench/{stem}.json", "round": host["sample"]["round"],
                "position": host["sample"]["position"], "label": host["sample"]["label"],
                "benchmark": host["sample"]["benchmark"], "admission": host["admission"],
                "in_run_load_1m_maximum": host["in_run_load_1m_maximum"], "wall_seconds": host["wall_seconds"]})
    clean = [host for host in host_records if host.get("clean")]
    reference_checks = {}
    for planned in plan["samples"]:
        cold = json.loads((campaign / planned["report"]).read_text())["cold_first_token"]
        checks = reference_checks.setdefault(planned["label"], {"prompt_reference_cosine": set(),
                                                               "minimum_cached_reference_cosine": [],
                                                               "minimum_teacher_reference_cosine": [],
                                                               "reference_cached_decisions": set()})
        checks["prompt_reference_cosine"].add(cold["prompt_reference_cosine"])
        checks["minimum_cached_reference_cosine"].append(cold["minimum_cached_reference_cosine"])
        checks["minimum_teacher_reference_cosine"].append(cold["minimum_teacher_reference_cosine"])
        checks["reference_cached_decisions"].add(cold["reference_cached_decisions"])
    reference_checks = {label: {"prompt_reference_cosine": sorted(checks["prompt_reference_cosine"]),
                                "minimum_cached_reference_cosine": min(checks["minimum_cached_reference_cosine"]),
                                "minimum_teacher_reference_cosine": min(checks["minimum_teacher_reference_cosine"]),
                                "reference_cached_decisions": sorted(checks["reference_cached_decisions"])}
                        for label, checks in reference_checks.items()}
    execution = {
        "first_sample_started_utc": min(host["host_before"]["utc"] for host in host_records),
        "last_sample_ended_utc": max(host["host_after"]["utc"] for host in host_records),
        "processes": len(host_records), "clean_samples": len(clean), "contaminated_attempts": contaminated,
        "maximum_admitting_load_1m": max(host["admission"]["admitting_load_1m"] for host in clean),
        "maximum_admitting_load_5m": max(host["admission"]["admitting_load_5m"] for host in clean),
        "maximum_in_run_load_1m_of_clean_samples": max(host["in_run_load_1m_maximum"] for host in clean),
        "minimum_cooldown_seconds": min(host["admission"]["cooldown_seconds"] for host in host_records),
        "maximum_load_wait_seconds": max(host["admission"]["load_wait_seconds"] for host in host_records),
        "power": sorted({host["host_before"]["power"].splitlines()[0] for host in host_records}),
        "in_run_reference_checks": reference_checks,
        "rounds_planned_first": 8,
        "discarded_attempts_per_sample": {
            f"r{sample['round']:02d}.p{sample['position']}.{sample['label']}.{sample['benchmark']}":
                len(sample["discarded_attempts"]) for sample in samples if sample["discarded_attempts"]},
        "steady_state_after_8_rounds": protocol.summarize([sample for sample in samples if sample["round"] <= 8])[
            "steady_state"],
    }
    files = {path.relative_to(root).as_posix(): sha256(path) for path in sorted(root.rglob("*"))
             if path.is_file() and "__pycache__" not in path.relative_to(root).parts}
    evidence = {name: digest for name, digest in files.items()
                if not name.startswith("harness/") and name not in AMENDMENT_FILES}
    record = {
        "schema_version": 1,
        "scope": "qwen3.5-0.8b_metal_f16_benchmark_rebaseline_not_a_support_declaration",
        "date_utc": "2026-10-18",
        "review_amendment": {
            **prose["review_amendment"], "added_files": list(AMENDMENT_FILES),
            "unchanged_evidence_files": len(evidence),
            "unchanged_evidence_sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
        },
        "advertised_supported": False,
        "support_claims_changed": False,
        "repository": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "profile": "qwen3.5-0.8b-metal-f16",
        **prose["head"],
        "performance_policy": {
            **prose["performance_policy"],
            "supersedes": {
                **prose["supersedes"],
                "record": "v0_1-release-20260910.json",
                "record_sha256": sha256(measurements / "v0_1-release-20260910.json"),
                "qwen_model_loop_baseline": f"{SEPTEMBER}/{SEPTEMBER_SUMMARIES['baseline']}",
                "qwen_model_loop_baseline_sha256": september["baseline"]["sha256"],
                "qwen_public_coordinator_baseline": f"{SEPTEMBER}/{SEPTEMBER_SUMMARIES['public']}",
                "qwen_public_coordinator_baseline_sha256": september["public"]["sha256"],
            },
            "future_alert_percent": round(protocol.ALERT * 100),
            "future_failure_percent_after_controlled_rerun": round(protocol.FAIL * 100),
            "gated_metrics": {kind: protocol.gated_metric_names(kind) for kind in protocol.BENCHMARKS},
            "diagnostic_metrics": list(protocol.DIAGNOSTIC_METRICS),
            "cold_first_token_gated": False,
            "cold_first_token_reported_as": ["median", "range"],
        },
        "protocol": {
            "document": "BENCHMARKS.md#qwen35-08b-benchmark-campaign-protocol",
            "script": "scripts/qwen35_benchmark_protocol.py",
            "campaign_script_copy": "harness/qwen35_benchmark_protocol.py",
            "campaign_script_sha256": plan["script_sha256"],
            "benchmarks": protocol.BENCHMARKS,
            "rounds": plan["rounds"], "layouts": [list(layout) for layout in protocol.LAYOUTS],
            "constants": plan["constants"],
            "minimum_rounds": protocol.MINIMUM_ROUNDS, "maximum_rounds": protocol.MAXIMUM_ROUNDS,
            "block_rounds": protocol.BLOCK_ROUNDS,
            "lock": "/Users/nover/repos/transformers-worktrees/.heavy-model.lock, held by harness/driver.sh for the whole campaign",
        },
        "host": {**plan["host_identity"], "accelerator": plan["accelerator"], **prose["host"]},
        "binaries": binaries,
        "artifacts": plan["samples"][0]["artifacts"],
        "campaign_directory": NAME,
        "files": files,
        "execution": execution,
        "samples": rows,
        "steady_state_rule": {
            **prose["steady_state_rule"],
            "script": "scripts/qwen35_benchmark_protocol.py",
            "window_rounds_minimum": protocol.STEADY_WINDOW_ROUNDS, "block_rounds": protocol.BLOCK_ROUNDS,
            "drift_tolerance": protocol.STEADY_DRIFT, "z_limit": protocol.STEADY_Z,
            "original_rule": {**prose["original_rule"], **before},
            "head_reference_change": reference_change,
        },
        "steady_state": summary["steady_state"],
        "cold_first_token": {
            **{kind: {label: summary["cells"][f"{label}/{kind}"]["diagnostic"]["cold_first_token"]
                      for label in ("head", "september")} for kind in protocol.BENCHMARKS},
            "discarded_attempt_exposure": prose["discarded_attempt_exposure"],
        },
        "reference": reference,
        "september_rebuild": control,
        "september_10_approved": september,
        "comparisons": comparisons,
        **prose["tail"],
        "historical_records": {name: sha256(measurements / name) for name in HISTORICAL},
    }
    (measurements / f"{NAME}.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
