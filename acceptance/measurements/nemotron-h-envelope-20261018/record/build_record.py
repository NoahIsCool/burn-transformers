"""Nemotron-H envelope round: writes acceptance/measurements/nemotron-h-envelope-20261018.json from its campaign directory.

Usage: python3 -I build_record.py WORKTREE

Reads the campaign directory ``acceptance/measurements/nemotron-h-envelope-20261018/``
(the registered method, the graph-generation logs of both attempts, the
copied manifests and digests of every bundle, the derivation and its
assessment, the gate build and run logs, and the source validation) and
writes the record with every file's SHA-256. Prose is the round's account;
every status, digest, figure, and quote is read from the campaign files (a
quote is a whole line, or a part of one), so the record's Python test can
recompute them.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

CAMPAIGN = "nemotron-h-envelope-20261018"
RECORD = f"{CAMPAIGN}.json"
BASE = "fa8426e5dc8cbcb269c29e211b119ea0acd61054"
METHOD_COMMIT = "8ad5fb96577d8f54c04109b2f1d2c53cfa1c1682"
DERIVATION_COMMIT = "0a6a1245b79868cecf3df17168af07f1bd37e458"
CODE_HEAD = "2354b22c1f7dd005da22ad7a9b95c0ae408f32f5"
SOURCE = "2cf83d7030867dae96b7b23d2a8bbdc3efe06fc2"
DOCS = "bae4f05aeab3b7542894d6e98362650d516956ec"
GENERATOR_COMMIT = "c25a2cb48eacfe5ff0a6cf543795175cd745a4ad"
ATTEMPT_1_COMMIT = "4155bf0a915fb491833e1552f2e48bec7e808b24"
HISTORICAL = ["half-precision-calibration-20261017.json", "mac-validation-20261017.json",
              "half-precision-envelope-20261018.json", "half-precision-followups-20261017.json",
              "half-precision-review-20261018.json", "reference-precision-followups-20261016.json"]
CHECKPOINT_RECORD = "nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json"
GATE_LOG = "gates/logs/nemotron-h.f16.metal.official-precision.log"
LOO_LOG = "gates/logs/bundles.official_f16_graphs_admit_each_other_leave_one_out.log"


def sha256(data):
    """Hexadecimal SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def line(text, pattern):
    """The first line of ``text`` containing ``pattern`` (which must occur)."""
    for candidate in text.splitlines():
        if pattern in candidate:
            return candidate
    raise SystemExit(f"{pattern!r} not found")


def main():
    """Builds the record."""
    worktree = Path(sys.argv[1])
    measurements = worktree / "acceptance/measurements"
    campaign = measurements / CAMPAIGN

    def text(relative):
        """Text of a campaign file."""
        return (campaign / relative).read_text()

    def evidence(relative, *patterns):
        """A {log, quotes} entry quoting the first line holding each pattern."""
        body = text(relative)
        return {"log": relative, "quotes": [line(body, pattern) for pattern in patterns]}

    def fragment(relative, pattern):
        """A {log, quotes} entry quoting the first match of a regular expression."""
        return {"log": relative, "quotes": [re.search(pattern, text(relative), re.M).group(0)]}

    envelope = json.loads(text("derivation/envelope.json"))
    digests = json.loads(text("bundles/bundle-digests.json"))
    paths = envelope["paths"]

    graphs = []
    sources = {"cpu-eager": ("precision", "float16 rows of the precision bundle (October 16)"),
               "mps-eager": ("oracle", "the authenticated MPS/F16 oracle (September 8, macOS 15.7.9)"),
               "cpu-sdpa": ("f16-cpu-sdpa", "generator/logs/cpu-sdpa.log"),
               "mps-sdpa": ("f16-mps-sdpa", "generator/logs/mps-sdpa.log")}
    runs_by_label = {json.loads(raw)["label"]: json.loads(raw) for raw in text("generator/logs/runs.jsonl").splitlines()}
    for name in envelope["graphs"]:
        key, source = sources[name]
        entry = {"graph": name, "source": source, "bundle": digests[key]["path"],
                 "worst_rows": {kind: {"normalized_deviation": path["graphs"][name]["max"],
                                       "cosine": path["graphs"][name]["minimum_cosine"],
                                       "decision": path["graphs"][name]["decision"],
                                       "leave_one_out": path["leave_one_out"][name]}
                                for kind, path in paths.items()}}
        if key.startswith("f16-"):
            label = key.removeprefix("f16-")
            run = runs_by_label[label]
            manifest = json.loads(text(f"bundles/{key}/reference.json"))
            entry.update({
                "reference_json_sha256": digests[key]["files"]["reference.json"]["sha256"],
                "files_sha256": {file: value["sha256"] for file, value in digests[key]["files"].items()},
                "generator_commit": GENERATOR_COMMIT,
                "generator_sha256": manifest["provenance"]["script_sha256"],
                "resolved": manifest["graph"]["resolved"],
                "experts_implementation_requested": manifest["graph"]["experts_implementation_requested"],
                "max_resident_gib": manifest["provenance"]["max_resident_gib"],
                "offloaded_blocks": manifest["provenance"]["offloaded_blocks"],
                "offload_execution_devices": manifest["provenance"]["offload_execution_devices"],
                "self_check": manifest["provenance"]["self_check"],
                "log": f"generator/logs/{label}.log",
                "wall_seconds": run["wall_seconds"],
                "peak_memory_footprint_bytes": run["peak_memory_footprint_bytes"],
                "max_swap_growth_mib": run["max_swap_growth_mib"],
                "phase_seconds": {t["phase"]: t["seconds"] for t in manifest["provenance"]["timings"] if "seconds" in t},
                "host_load_average_at_start": manifest["provenance"]["timings"][0]["host"]["load_average"],
            })
        else:
            entry["reference_json_sha256"] = digests[key]["files"][
                "reference.json" if key == "precision" else
                "nemotron-3.5-lightning-30b-a3b-a9904d24-reference.json"]["sha256"]
        graphs.append(entry)

    control_manifest = json.loads(text("bundles/f16-mps-eager-control/reference.json"))
    control_run = runs_by_label["mps-eager-control"]
    (control,) = envelope["controls_reported_not_adopted"]
    controls = [{
        "graph": "mps-eager", "repeats": "the MPS/F16 oracle", "adopted": False,
        "bundle": digests["f16-mps-eager-control"]["path"],
        "reference_json_sha256": digests["f16-mps-eager-control"]["files"]["reference.json"]["sha256"],
        "files_sha256": {file: value["sha256"] for file, value in digests["f16-mps-eager-control"]["files"].items()},
        "log": "generator/logs/mps-eager-control.log",
        "wall_seconds": control_run["wall_seconds"],
        "peak_memory_footprint_bytes": control_run["peak_memory_footprint_bytes"],
        "platform": control_manifest["provenance"]["platform"],
        "paths": {kind: {"cosine": value["control"]["minimum_cosine"], "decision": value["control"]["decision"],
                         "normalized_deviation": value["control"]["max"],
                         "minimum_cosine_to_the_oracle": value["minimum_cosine_to_the_repeated_graph"],
                         "bitwise_equal_to_the_oracle": value["bitwise_equal_to_the_repeated_graph"]}
                  for kind, value in control["paths"].items()},
        "replacing_the_oracle": control["replacing"],
        "added_as_a_fifth_graph": control["adding"],
    }]

    attempt = [json.loads(raw) for raw in text("generator/logs/attempt-1/runs.jsonl").splitlines()]
    gate_runs = [json.loads(raw) for raw in text("gates/logs/runs.jsonl").splitlines()]
    build = text("gates/logs/build.log")
    binaries = [{"path": path, "sha256": digest} for path, digest in
                re.findall(r"^== binary (.+) sha256 ([0-9a-f]{64})$", build, re.M)]
    gate_text = text(GATE_LOG)

    def burn(kind):
        """Burn's per-path figures from the gate log."""
        found = re.search(rf"Nemotron Burn F16 against the F32 truth {kind}: (\d+) rows, minimum cosine ([0-9.]+) at "
                          rf"decision (\d+) \(delta ([0-9.e-]+), ([0-9.]+) of the envelope, ([0-9.]+) of the bound "
                          rf"([0-9.e-]+); floor ([0-9.]+)\)", gate_text)
        rows, cosine, decision, delta, of_envelope, of_bound, bound, floor = found.groups()
        cosines = [float(v) for v in re.search(rf"{kind} cosines: \[(.*)\]", gate_text).group(1).split(", ")]
        return {"rows": int(rows), "minimum_cosine": float(cosine), "decision": int(decision),
                "normalized_deviation": float(delta), "of_envelope": float(of_envelope),
                "of_bound": float(of_bound), "bound": float(bound), "cosine_floor": float(floor),
                "cosines": cosines}

    burn_rows = {kind.lower(): burn(kind) for kind in ("Prompt", "Teacher", "Cached")}
    runs = []
    for run in gate_runs:
        log = f"gates/logs/{run['name']}.log"
        runs.append({"name": run["name"], "test": run["test"], "kind": "bundle-only" if run["name"].startswith("bundles.")
                     else "real-checkpoint", "exit": run["status"], "result": run["result"], "expected": "pass",
                     "heavy_model_lock": not run["name"].startswith("bundles."),
                     "wall_seconds": run["wall_seconds"], "peak_memory_footprint_bytes": run["peak_memory_footprint_bytes"],
                     "max_swap_growth_mib": run["max_swap_growth_mib"], "log": log})

    validation_logs = sorted(p.name for p in (campaign / "validation").iterdir())
    checks = []
    for name in validation_logs:
        lines = text(f"validation/{name}").splitlines()
        checks.append({"name": name.removesuffix(".log"), "log": f"validation/{name}",
                       "command": lines[1].removeprefix("== cmd "),
                       "commit": re.match(r"== source ([0-9a-f]{40})", lines[0]).group(1),
                       "exit": int(lines[-1].removeprefix("== exit="))})

    record = {
        "schema_version": 1,
        "scope": "nemotron_h_official_precision_envelope_round_not_a_support_declaration",
        "date_utc": "2026-10-18",
        "advertised_supported": False,
        "passed": all(run["result"] == "pass" for run in runs),
        "passed_note": "Every run of the round passed: the bundle-only leave-one-out re-derivation and the Nemotron-H "
                       "Metal/F16 official-precision gate under the new rule. That is a gate pass, not a support claim.",
        "base_commit": BASE,
        "source_commit": SOURCE,
        "code_head": CODE_HEAD,
        "documentation_commit": DOCS,
        "head_note": f"The gate and bundle-only binary was built from {SOURCE[:7]} in the clean worktree; its Rust "
                     f"sources equal the code head {CODE_HEAD[:7]} (the commit after it changes only "
                     f"NEMOTRON_H_ACCEPTANCE.md). Its library code (src/, Cargo.*) equals the base {BASE[:7]}: the round "
                     "changes only tests/, scripts/, documentation, and this campaign. Later commits change only "
                     "documentation, this record, its campaign directory, and its Python test.",
        "campaign_directory": CAMPAIGN,
        "host": "Apple M3 Max, 96 GiB unified memory, macOS 27.0.1 (26A434), shared with desktop applications and "
                "other sessions; external volumes /Volumes/bag of holding and /Volumes/T9 mounted throughout",
        "harness": "Official graphs: generator/run_locked.sh and generator/run_graph.py, one graph per process under "
                   "/Users/nover/repos/transformers-worktrees/.heavy-model.lock (mkdir, retried every 60 s, rmdir in "
                   "a trap), from a git archive export of the generator commit, under /usr/bin/time -l with the earlier "
                   "rounds' memory watchdog plus a 10 GiB system-volume reserve; each log records uptime, the processes "
                   "using the most CPU, free memory, swap, and free disk at start and end, and every timed phase in a "
                   "manifest carries a load-average and top-process sample. Burn: gates/run_locked_gate.sh and "
                   "gates/run_gate.py, the prebuilt release binary, one test per process from the clean worktree at the "
                   "source commit (`dirty` counts changes outside this campaign directory), the real-checkpoint gate "
                   "under the same lock and watchdog, the bundle-only test without the lock.",
        "method": {"file": "derivation/method.txt", "commit": METHOD_COMMIT,
                   "registered_utc": re.search(r"Registered (\S+Z)", text("derivation/method.txt")).group(1),
                   "summary": "Four official F16 graphs {cpu, mps} x {eager, sdpa}; per row the normalized deviation "
                              "sqrt(2 (1 - cosine)) from the F32 truth row; per path the envelope E = largest worst-row "
                              "deviation, each graph's leave-one-out ratio over the largest other graph, and the slack "
                              "s = the largest ratio rounded up to two significant digits; a Burn row passes when its "
                              "deviation is at most s E (cosine at least 1 - (s E)^2 / 2) with exact top-1. Registered "
                              "before any new official graph or Burn row of the round."},
        "graphs": graphs,
        "controls": controls,
        "attempts": [{
            "attempt": 1, "generator_commit": ATTEMPT_1_COMMIT, "logs": "generator/logs/attempt-1/",
            "runs": [{"label": run["label"], "status": run["status"], "wall_seconds": run["wall_seconds"],
                      "killed": run["killed"], "max_swap_growth_mib": run["max_swap_growth_mib"],
                      "min_root_free_gib": run["min_root_free_gib"]} for run in attempt],
            "account": "The MPS SDPA graph loaded the whole F16 model onto MPS; swap grew by 12,478 MiB in 80 s and the "
                       "system volume fell from 22.6 to 11.8 GiB, and the watchdog stopped it. The CPU SDPA graph then "
                       "started into that swap and was stopped 10 s later when the system volume reached 9.9 GiB. Both "
                       "kills reached only /usr/bin/time: the two generators kept running until they were stopped by "
                       "hand about a minute later (system volume at 5.2 GiB, recovering to 20.3 GiB as the swap files "
                       "shrank); the queued control was stopped at its start gate. The driver now kills the process "
                       "group, waits for 70% free memory and 15 GiB on the system volume before each graph, and reports "
                       "the real exit status (attempt 1's driver.log shows exit=0 for killed runs, a $? read after a "
                       "command substitution). No bundle was published by attempt 1."}],
        "derivation": {
            "result": "derivation/envelope.json", "tables": "derivation/envelope.txt",
            "assessment": "derivation/assessment.txt", "commit": DERIVATION_COMMIT,
            "derived_utc": re.search(r"== derived_utc (\S+)", text("derivation/envelope.txt")).group(1),
            "assessed_utc": re.search(r"Written (\S+Z)", text("derivation/assessment.txt")).group(1),
            "graph_count": envelope["graph_count"],
            "slacks": envelope["slacks"],
            "largest_ratio": {kind: path["largest_ratio"] for kind, path in paths.items()},
            "largest_ratio_graph": {kind: path["largest_ratio_graph"] for kind, path in paths.items()},
            "envelope": {kind: path["envelope"] for kind, path in paths.items()},
            "envelope_graph": {kind: path["envelope_graph"] for kind, path in paths.items()},
            "bound": {kind: path["bound"] for kind, path in paths.items()},
            "cosine_floor": {kind: path["cosine_floor"] for kind, path in paths.items()},
            "previous_floor": {"prompt": 0.999194145, "teacher": 0.994466421, "cached": 0.998635047},
            "reported_not_adopted": envelope["reported_not_adopted"],
        },
        "burn": {"commit": SOURCE, "library_code_equals": BASE, "paths": burn_rows,
                 "top1": "32/32 on the teacher and cached paths and 1/1 on the prompt row: the gate reported no top-1 or bound violation",
                 "summary_line": line(gate_text, "Nemotron 3.5 official corresponding-path parity:")},
        "binaries": binaries,
        "build_log": "gates/logs/build.log",
        "runs": runs,
        "tallies": {"runs": len(runs), "pass": sum(r["result"] == "pass" for r in runs),
                    "fail": sum(r["result"] == "fail" for r in runs),
                    "as_expected": sum(r["result"] == r["expected"] for r in runs)},
        "findings": [
            {"id": "envelope-slacks-are-rounding", "status": "settled: the compiled slacks are the derivation's",
             "summary": "From the four official graphs alone, the largest leave-one-out ratios are 1.0002 (prompt, CPU "
                        "eager), 1.0042 (teacher, CPU SDPA), and 1.0046 (cached, MPS SDPA), so every slack is 1.1, the "
                        "rounding up to two significant digits rather than a measured spread. The worst rows coincide "
                        "across graphs (teacher decision 18 in three graphs, cached decision 22 in both MPS graphs), so "
                        "the graphs agree at their spikes; the per-path slack rests on four ratios, twelve pooled, where "
                        "the envelope round's rest on 80 to 128. The cosine floors become 0.999025 (prompt), 0.993248 "
                        "(teacher), and 0.998333 (cached), from 0.999194, 0.994466, and 0.998635.",
             "evidence": [evidence("derivation/envelope.txt",
                                   "envelope E 4.014610e-02 (cpu-eager decision 0); slack 1.1 from 1.0002",
                                   "envelope E 1.056400e-01 (cpu-sdpa decision 18); slack 1.1 from 1.0042",
                                   "envelope E 5.248829e-02 (mps-sdpa decision 22); slack 1.1 from 1.0046"),
                          evidence(LOO_LOG, "Cached: envelope 5.248829e-2 (mps-sdpa), slack 1.1")]},
            {"id": "gate-passes-on-the-rounded-slack", "status": "passed; the rule awaits review",
             "summary": "The Nemotron-H Metal/F16 official-precision gate passes under the envelope rule. Cached decision "
                        "22 reproduces the envelope round's failing cosine 0.998557984 to every printed digit; it is "
                        "1.0231 of the four-graph envelope and 0.930 of the bound. No official graph lies further than "
                        "1.0046 outside the envelope of the other three, so the pass depends on the slack's rounding "
                        "(a 1.0046 ratio rounded to 1.1). Teacher rows reach at least 0.996039 (decision 30, 0.84 of the "
                        "envelope), the prompt row 0.999200 (0.996 of the envelope), with exact top-1 on every row.",
             "evidence": [evidence(GATE_LOG,
                                   "Cached: 32 rows, minimum cosine 0.998557984489 at decision 22 (delta 5.370318e-2, 1.0231 of the envelope",
                                   "Teacher: 32 rows, minimum cosine 0.996039426551 at decision 30",
                                   "test result: ok. 1 passed")]},
            {"id": "teacher-decision-30", "status": "open: deserves a targeted look",
             "summary": "Burn's teacher row 30 is 0.996039 from F32 (0.997612 at 9a9c730, before the once-rounded SiLU), "
                        "where every official F16 graph is near 0.99996. The per-path bound admits it (the teacher "
                        "envelope is set by decision 18), so it is a Burn-side deviation the gate does not explain.",
             "evidence": [evidence(GATE_LOG, "f32_truth_teacher_minimum=0.996039426551@30")]},
            {"id": "mps-control-not-bitwise", "status": "recorded; cause not separated",
             "summary": "Repeating the oracle's MPS eager graph with this round's tool (16 GiB budget, macOS 27.0.1; the "
                        "oracle ran resident on macOS 15.7.9) is not bitwise equal to the oracle: cosine to it at least "
                        "0.999999588 (prompt), 0.999686218 (teacher, decision 28), 0.999869702 (cached, decision 8). Its "
                        "cached decision 22 (0.998595604) lies 1.0097 of the envelope out. Measured: the difference; "
                        "inferred, not measured: the OS change rather than offload, since the tiny official model shows "
                        "offload bitwise neutral on MPS. Replacing the oracle with it, or adding it, would set the cached "
                        "floor to 0.998300681; it is reported, not adopted.",
             "evidence": [evidence("derivation/envelope.txt",
                                   "- teacher: max delta 1.050496e-01 at decision 18 (cosine 0.994482288); minimum cosine to mps-eager 0.999686218, bitwise equal False",
                                   "- cached: max delta 5.299803e-02 at decision 22 (cosine 0.998595604)")]},
            {"id": "teacher-spike-is-attention-path", "status": "measured",
             "summary": "The teacher decision-18 spike (cosine near 0.9944) appears in CPU SDPA, MPS eager, and MPS SDPA "
                        "but not in CPU eager (0.998632 at decision 27), so it follows the attention path, not the "
                        "device. CPU SDPA's mean teacher deviation is 1.59 times the largest of the other graphs' "
                        "(reported, not gated).",
             "evidence": [evidence("derivation/envelope.txt",
                                   "- cpu-sdpa: max delta 1.056400e-01 at decision 18 (cosine 0.994420097), mean delta 2.886415e-02")]},
            {"id": "resident-mps-load-swaps", "status": "fixed in the driver; recorded",
             "summary": "A resident F16 load of the 30B model onto MPS grew swap by 12.5 GiB in 80 s on this host and left "
                        "the system volume too full for the next graph; the watchdog's kill orphaned both generators. "
                        "MPS graphs now use the CPU's 16 GiB block budget with offloaded blocks executing on MPS, and "
                        "the driver kills the process group.",
             "evidence": [evidence("generator/logs/attempt-1/mps-sdpa.log", "WATCHDOG_KILLED: swap grew by 12478 MiB"),
                          evidence("generator/logs/attempt-1/cpu-sdpa.log", "WATCHDOG_KILLED: system volume down to 9.9 GiB free")]},
            {"id": "closer-to-the-f16-oracle", "status": "measured; attribution inferred",
             "summary": "Against the F16 oracle Burn moved closer than at 9a9c730 (cached minimum 0.999701 from 0.999299, "
                        "hidden traces at least 0.999995 from 0.999802), while its cached decision 22 moved further from "
                        "F32. Attributed to the once-rounded SiLU matching the official F.silu by the envelope round's "
                        "isolate control (that site alone restores the 9a9c730 statistics); not re-measured here.",
             "evidence": [evidence(GATE_LOG, "f16_oracle_cached_minimum=0.999700840921")]},
        ],
        "decisions": [{"id": "nemotron-envelope-rule", "user_review": True, "implemented": True,
                       "what": "The two-graph floor is replaced by the registered four-graph envelope rule with per-path "
                               "slacks 1.1/1.1/1.1, the envelope computed in the test from authenticated rows, a "
                               "fail-closed reader of the two SDPA bundles, a bundle-only re-derivation test, and every "
                               "row measured before judgment. Implemented because the derivation is sound by the "
                               "registered method; adopting it as the gate's rule awaits the maintainer.",
                       "commits": [METHOD_COMMIT, "4155bf0a915fb491833e1552f2e48bec7e808b24",
                                   "8ef285eb5c3713abac69604767dd419b4dd1e19c", GENERATOR_COMMIT,
                                   "b4e2bdbeb6e011d77ba29b0e10903593bf59925c", DERIVATION_COMMIT, CODE_HEAD, SOURCE, DOCS]}],
        "awaiting_maintainer": [
            "Adopting the four-graph envelope rule for Nemotron-H: its slacks are the rounding's 1.1 over coincident worst "
            "rows, and the gate passes only because 1.0046 rounds to 1.1 (Burn's cached row 22 is 1.0231 of the envelope).",
            "Whether a repeat of an official graph (the MPS control moves the cached envelope by about 1%), more official "
            "graphs (for example other experts implementations or a regenerated oracle on the current macOS), or pooling "
            "with other checkpoints should widen the evidence behind the slack.",
            "Burn's teacher decision 30 (0.996039 from F32 where every official F16 graph is near 0.99996).",
        ],
        "not_run": [
            "The crate-wide test and warning-denied Clippy matrix (--all-targets --all-features), rustdoc, the documentation "
            "audit, and check_release.py: no library code changed; only the Nemotron target and the Python tools were checked.",
            "Every other real-checkpoint gate, benchmark, and smoke, including the Nemotron-H Metal smoke: the library code "
            "equals the base commit.",
            "CUDA, ROCm, and coverage.",
            "Official F32 hidden traces and real-weight MTP parity for Nemotron-H (still open).",
        ],
        "source_validation": {"checks": checks, "passed": all(check["exit"] == 0 for check in checks)},
        "historical_records": {name: sha256((measurements / name).read_bytes()) for name in HISTORICAL}
                              | {f"../{CHECKPOINT_RECORD}": sha256((measurements.parent / CHECKPOINT_RECORD).read_bytes())},
    }
    files = {}
    for path in sorted(campaign.rglob("*")):
        if path.is_file():
            files[path.relative_to(campaign).as_posix()] = sha256(path.read_bytes())
    record["files"] = files
    (measurements / RECORD).write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {RECORD}: {len(files)} files, passed={record['passed']}")


if __name__ == "__main__":
    main()
