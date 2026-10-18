"""Nemotron-H envelope decision: writes acceptance/measurements/nemotron-h-envelope-decision-20261018.json.

Usage: python3 -I build_record.py WORKTREE

Reads this campaign directory (the bundle-only run and the source-validation
logs at the documentation commit) and the envelope round's committed evidence
(its record, gate log, derivation, control manifest, and generator logs), and
writes the decision record with every campaign file's SHA-256. Prose is the
round's account; every status, digest, figure, and quote is read from those
files (a quote is a whole line, or a part of one), so the record's Python test
can recompute them. The envelope record is never modified.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

CAMPAIGN = "nemotron-h-envelope-decision-20261018"
RECORD = f"{CAMPAIGN}.json"
ENVELOPE_CAMPAIGN = "nemotron-h-envelope-20261018"
ENVELOPE_RECORD = f"{ENVELOPE_CAMPAIGN}.json"
ENVELOPE_COMMIT = "995ae022fd4358f33432b5f7a0605ac8352d5136"
GATE_COMMIT = "a5d4d86eace4c447a28ad5bf35c89ac54bfcc644"
CODE_HEAD = "530a82a1f5714a5a1f7eb8425b1a53d8e5dcd166"
DOCS = "603ec5c41d4fcb8b4895010157f347dda4f57e26"
#: The decision exactly as the task relayed it to this session.
DECISION_QUOTE = ("the Nemotron-H official-precision gate stays an open finding. Do NOT adopt the four-graph "
                  "envelope rule. Its slack is 1.1 only because leave-one-out ratios of at most 1.0046 round up, "
                  "and Burn's cached decision 22 sits at 1.023 times the envelope.")
HISTORICAL = ["half-precision-calibration-20261017.json", "mac-validation-20261017.json",
              "half-precision-envelope-20261018.json", "half-precision-followups-20261017.json",
              "half-precision-review-20261018.json", "reference-precision-followups-20261016.json",
              ENVELOPE_RECORD, "../nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json"]
BUNDLE_LOG = "logs/bundles.reports_official_f16_floors_and_four_graph_envelope.log"
GATE_LOG = f"{ENVELOPE_CAMPAIGN}/gates/logs/nemotron-h.f16.metal.official-precision.log"
#: Source checks, in run order: (name, log, purpose).
CHECKS = [
    ("fmt", "validation/fmt.log", "formatting"),
    ("clippy-default-all-features", "validation/clippy-default-all-features.log",
     "warning-denied Clippy, default toolchain, all targets and features"),
    ("clippy-1.99.1-all-features", "validation/clippy-1.99.1-all-features.log",
     "warning-denied Clippy, Rust 1.99.1, all targets and features"),
    ("clippy-1.99.1-backends", "validation/clippy-1.99.1-backends.log",
     "warning-denied Clippy, Rust 1.99.1, every backend feature without defaults"),
    ("test-default", "validation/test-default.log", "Rust tests and doctests, default features"),
    ("test-std-flex-train", "validation/test-std-flex-train.log",
     "Rust tests and doctests, std,flex,tokenizers,chat-templates,autodiff,train"),
    ("msrv-1.92.0-check", "validation/msrv-1.92.0-check.log", "MSRV check, all targets"),
    ("rustdoc-private", "validation/rustdoc-private.log", "warning-denied private rustdoc"),
    ("python-tests", "validation/python-tests.log", "Python tooling tests"),
    ("documentation-audit", "validation/documentation-audit.log", "documentation audit with --standards"),
]


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
        """Text of a file below acceptance/measurements."""
        return (measurements / relative).read_text()

    def evidence(relative, *patterns):
        """A {log, quotes} entry, the log relative to acceptance/measurements, quoting the first line holding each pattern."""
        body = text(relative)
        return {"log": relative, "quotes": [line(body, pattern) for pattern in patterns]}

    def fragment(relative, pattern):
        """A {log, quotes} entry quoting the first match of a regular expression."""
        return {"log": relative, "quotes": [re.search(pattern, text(relative), re.M).group(0)]}

    envelope_record = json.loads(text(ENVELOPE_RECORD))
    derivation = json.loads(text(f"{ENVELOPE_CAMPAIGN}/derivation/envelope.json"))
    control = json.loads(text(f"{ENVELOPE_CAMPAIGN}/bundles/f16-mps-eager-control/reference.json"))
    burn = envelope_record["burn"]["paths"]
    paths = derivation["paths"]
    bundle_log = text(f"{CAMPAIGN}/{BUNDLE_LOG}")
    floors_line = line(bundle_log, "Nemotron official F16 floors (the gate's rule)")
    floors = dict(zip(("prompt", "teacher", "cached"),
                      map(float, re.search(r"\[([0-9., ]+)\]$", floors_line).group(1).split(", "))))
    gate_log = text(GATE_LOG)
    summary_line = line(gate_log, "Nemotron 3.5 official corresponding-path parity:")

    def floor_row(kind):
        """Graph and decision whose cosine sets ``kind``'s floor (the worse of mps-eager and cpu-eager)."""
        graph = min(("mps-eager", "cpu-eager"), key=lambda name: paths[kind]["graphs"][name]["minimum_cosine"])
        return {"graph": graph, "decision": paths[kind]["graphs"][graph]["decision"],
                "cosine": paths[kind]["graphs"][graph]["minimum_cosine"]}

    verdict_paths = {}
    failing = []
    for kind in ("prompt", "teacher", "cached"):
        cosines = burn[kind]["cosines"]
        below = [decision for decision, cosine in enumerate(cosines) if cosine < floors[kind]]
        failing += [{"path": kind, "decision": decision, "cosine": burn[kind]["minimum_cosine"]
                     if decision == burn[kind]["decision"] else cosines[decision], "floor": floors[kind]}
                    for decision in below]
        verdict_paths[kind] = {
            "rows": len(cosines),
            "minimum_cosine": burn[kind]["minimum_cosine"],
            "decision": burn[kind]["decision"],
            "floor": floors[kind],
            "floor_row": floor_row(kind),
            "margin": round(burn[kind]["minimum_cosine"] - floors[kind], 12),
            "rows_below_floor": below,
        }

    diagnostic_paths = {}
    for kind in ("prompt", "teacher", "cached"):
        path = paths[kind]
        diagnostic_paths[kind] = {
            "envelope": path["envelope"],
            "envelope_graph": path["envelope_graph"],
            "largest_leave_one_out": path["largest_ratio"],
            "largest_leave_one_out_graph": path["largest_ratio_graph"],
            "registered_method_slack_not_adopted": path["slack"],
            "burn_worst_of_envelope": burn[kind]["of_envelope"],
            "burn_beyond_every_official_graph": burn[kind]["of_envelope"] > path["largest_ratio"],
        }

    row30 = {name: graph["cosines"][30] for name, graph in paths["teacher"]["graphs"].items()}
    row30["mps-eager-control"] = control["comparisons"]["float32"]["teacher"]["cosines"][30]
    row18 = {name: graph["cosines"][18] for name, graph in paths["teacher"]["graphs"].items()}
    hidden = {key: float(value) for key, value in re.findall(
        r"(stateless_hidden|cached_prefill_hidden|cached_decode_hidden)_minimum_cosine=([0-9.]+)@", summary_line)}

    checks = []
    for name, relative, purpose in CHECKS:
        body = text(f"{CAMPAIGN}/{relative}")
        lines = body.splitlines()
        check = {"name": name, "purpose": purpose, "log": relative,
                 "command": lines[1].removeprefix("== cmd "), "commit": lines[0].split()[2],
                 "exit": int(lines[-1].removeprefix("== exit="))}
        passed = [int(count) for count in re.findall(r"^test result: ok\. (\d+) passed", body, re.M)]
        failed = [int(count) for count in re.findall(r"; (\d+) failed;", body)]
        if passed:
            check["rust_tests_passed"] = sum(passed)
            check["rust_tests_failed"] = sum(failed)
        python = re.search(r"^Ran (\d+) tests? in", body, re.M)
        if python:
            check["python_tests_ran"] = int(python.group(1))
            check["python_result"] = re.findall(r"^(?:OK|FAILED)\b.*$", body, re.M)[-1]
        if name == "documentation-audit":
            check["gaps"] = sum(int(gaps) for gaps in re.findall(r": (\d+)/\d+ authored sites missing", body))
        checks.append(check)

    record = {
        "schema_version": 1,
        "scope": "nemotron_h_official_precision_envelope_decision_not_a_support_declaration",
        "date_utc": "2026-10-18",
        "advertised_supported": False,
        "passed": False,
        "passed_note": ("The record states a maintainer decision and the gate verdict it implies: the Nemotron-H "
                        "Metal/F16 official-precision gate fails under its restored two-graph floors at cached "
                        "decision 22, an open finding. The bundle-only run and every source check passed; they are "
                        "not a gate pass."),
        "extends": {"record": ENVELOPE_RECORD, "commit": ENVELOPE_COMMIT,
                    "sha256": sha256((measurements / ENVELOPE_RECORD).read_bytes()),
                    "note": ("The envelope record stays byte-identical. Its gate pass is a pass of the proposed rule, "
                             "which was not adopted; its measurements stand as diagnostic evidence, and the "
                             "statements listed under corrections are corrected here.")},
        "decision": {
            "date": "2026-10-18",
            "by": "the maintainer",
            "relayed": ("quoted verbatim from the task text that carried the decision to this session; no other "
                        "statement of it exists in the repository"),
            "quote": DECISION_QUOTE,
            "effects": [
                "The gate's pass/fail criterion returns to the per-path floors of the two official F16 evaluations, "
                "with no slack.",
                "The four-graph envelope is printed as a diagnostic only: the compiled slacks and the leave-one-out "
                "assertion are removed, and the F16 graph bundles are optional for the gate.",
                "The gate's failure at cached decision 22 stays an open finding; no checkpoint support changes.",
            ],
        },
        "base_commit": ENVELOPE_COMMIT,
        "gate_commit": GATE_COMMIT,
        "code_head": CODE_HEAD,
        "documentation_commit": DOCS,
        "source_commit": DOCS,
        "head_note": (f"The gate code changed in {GATE_COMMIT[:7]} (tests/ and the envelope record's test), the "
                      f"derivation tool's docstring in {CODE_HEAD[:7]}, and documentation in {DOCS[:7]}. Every check "
                      f"ran from the clean worktree at {DOCS[:7]} (dirty counts changes outside this campaign "
                      "directory). Its library code (src/, Cargo.toml, Cargo.lock) equals 2cf83d7's and fa8426e's; "
                      "the commit after it adds only this record, its campaign directory, and its Python test."),
        "campaign_directory": CAMPAIGN,
        "host": ("Apple M3 Max, 96 GiB unified memory, macOS 27.0.1 (26A434), shared with desktop applications and "
                 "other sessions; external volumes /Volumes/bag of holding and /Volumes/T9 mounted throughout"),
        "harness": ("run_check.py runs each check once from the clean worktree at the documentation commit and "
                    "refuses a dirty or moved checkout; the bundle-only run reads the external oracle, precision, "
                    "and F16 graph bundles without the checkpoint or Metal, so it ran without the heavy-model lock. "
                    "No real-checkpoint run or benchmark ran."),
        "rule": {
            "statement": ("Every Burn F16 row of a path needs exact top-1 and a cosine to its F32 truth row at least "
                          "the path's floor: the worst cosine against F32 that the MPS/F16 oracle or the precision "
                          "bundle's CPU/F16 rows (both eager attention) reach on that path, with no slack. Every row "
                          "is measured before any is judged."),
            "floors": floors,
            "floor_rows": {kind: verdict_paths[kind]["floor_row"] for kind in floors},
            "previous_rule_commit": "fa8426e5dc8cbcb269c29e211b119ea0acd61054",
            "evidence": [evidence(f"{CAMPAIGN}/{BUNDLE_LOG}", "Nemotron official F16 floors (the gate's rule)",
                                  "test result: ok. 1 passed")],
        },
        "verdict": {
            "status": "fails: an open finding",
            "basis": "inferred",
            "basis_note": ("Inferred, not measured at the new source: the restored rule applied to Burn's rows "
                           "measured at 2cf83d7 (the envelope record's gate run), whose library code the new source "
                           "compiles, against the floors the bundle-only run measured at the new source. The "
                           "restored gate was not run on the checkpoint. The same criterion failed, measured, at "
                           "f9f78b5 with the same cosine at cached decision 22 (half-precision envelope record)."),
            "failing_rows": failing,
            "paths": verdict_paths,
            "top1": "32/32 on the teacher and cached paths and 1/1 on the prompt row at 2cf83d7",
            "evidence": [evidence(GATE_LOG, "Cached: 32 rows, minimum cosine", "Teacher: 32 rows, minimum cosine",
                                  "Prompt: 1 rows, minimum cosine")],
        },
        "diagnostic": {
            "status": "printed, not adopted",
            "note": ("The four-graph envelope, its leave-one-out ratios, and the registered method's slack, as the "
                     "bundle-only run printed them at the new source (equal to the envelope round's derivation), "
                     "with Burn's worst row of each path as a multiple of the envelope from the 2cf83d7 run."),
            "paths": diagnostic_paths,
            "evidence": [evidence(f"{CAMPAIGN}/{BUNDLE_LOG}", "Prompt: envelope 4.014610e-2",
                                  "Teacher: envelope 1.056400e-1", "Cached: envelope 5.248829e-2",
                                  "Cached: official mps-sdpa worst delta"),
                         evidence(GATE_LOG, "(delta 5.370318e-2, 1.0231 of the envelope")],
        },
        "corrections": [
            {
                "id": "teacher-decision-30-official-cosines",
                "corrects": [f"{ENVELOPE_RECORD} findings[teacher-decision-30].summary",
                             f"{ENVELOPE_RECORD} awaiting_maintainer[2]",
                             "NEMOTRON_H_ACCEPTANCE.md before 603ec5c"],
                "statement": "every official F16 graph is near 0.99996",
                "correction": ("At teacher decision 30 the official F16 graphs reach 0.999958 (CPU eager), 0.999346 "
                               "(CPU SDPA), 0.999968 (MPS eager), and 0.999954 (MPS SDPA) from F32, and the MPS eager "
                               "control 0.999957: CPU SDPA's deficit 1 - cosine is 14 to 21 times the other three graphs'. Burn's 0.996039 "
                               "has a deficit six times CPU SDPA's and 85 to 125 times the other three graphs'."),
                "teacher_decision_30_cosines": row30,
                "burn_teacher_decision_30_cosine": burn["teacher"]["minimum_cosine"],
            },
            {
                "id": "teacher-spike-is-not-attention-path-alone",
                "corrects": [f"{ENVELOPE_RECORD} findings[teacher-spike-is-attention-path].summary",
                             f"{ENVELOPE_CAMPAIGN}/derivation/assessment.txt limit 2"],
                "statement": "the teacher decision-18 spike follows the attention path, not the device",
                "correction": ("The spike is absent only from CPU eager. CPU SDPA, which differs from CPU eager only in "
                               "the attention implementation, has it, and both MPS graphs have it with eager or SDPA "
                               "attention, so neither the attention implementation nor the device alone accounts "
                               "for it. The cosines are measured; no attribution is."),
                "teacher_decision_18_cosines": row18,
            },
            {
                "id": "hidden-trace-minimum",
                "corrects": [f"{ENVELOPE_RECORD} findings[closer-to-the-f16-oracle].summary",
                             "NEMOTRON_H_ACCEPTANCE.md before 603ec5c"],
                "statement": "hidden traces at least 0.999995",
                "correction": ("The hidden traces' minimum cosine to the F16 oracle at 2cf83d7 is 0.999994958685 "
                               "(first recurrent decode, boundary 27), which rounds up to 0.999995; they are at least "
                               "0.99999495."),
                "hidden_minimum_cosines": hidden,
                "evidence": [fragment(GATE_LOG, r"cached_decode_hidden_minimum_cosine=[0-9.]+@\d+")],
            },
            {
                "id": "mps-sdpa-resident-budget",
                "corrects": ["NEMOTRON_H_ACCEPTANCE.md reproduction commands before 603ec5c"],
                "statement": "the MPS graph with --device mps --attn-implementation sdpa and no budget",
                "correction": ("The MPS SDPA graph ran with --max-resident-gib 16 and an offload directory, as the "
                               "CPU SDPA graph and the MPS eager control did."),
                "evidence": [fragment(f"{ENVELOPE_CAMPAIGN}/generator/logs/mps-sdpa.log",
                                      r'"--device", "mps", "--attn-implementation", "sdpa", .*"--max-resident-gib", "16"')],
            },
            {
                "id": "gate-pass-superseded",
                "corrects": [f"{ENVELOPE_RECORD} passed, findings[gate-passes-on-the-rounded-slack], "
                             "decisions[nemotron-envelope-rule]"],
                "statement": "the gate passes under the envelope rule; the rule awaits review",
                "correction": ("The rule was reviewed and not adopted; the gate's criterion is the two-graph floors, "
                               "under which it fails at cached decision 22."),
            },
            {
                "id": "generator-docstring-not-corrected",
                "corrects": ["scripts/generate_nemotron_h_f16_graph_reference.py module docstring"],
                "statement": "Its bound is the envelope of the official float16 graphs",
                "correction": ("Not corrected in place: the two graph bundles and the strict reader pin the "
                               "generator's SHA-256 (81f5bd77...), so its bytes must not change. The gate's bound is "
                               "the two-graph floors; the derivation tool's docstring was corrected in "
                               f"{CODE_HEAD[:7]}."),
            },
        ],
        "runs": [{
            "name": "bundles.reports_official_f16_floors_and_four_graph_envelope",
            "test": "reports_official_f16_floors_and_four_graph_envelope",
            "kind": "bundle-only",
            "heavy_model_lock": False,
            "commit": DOCS,
            "exit": int(bundle_log.splitlines()[-1].removeprefix("== exit=")),
            "result": "pass" if "test result: ok. 1 passed" in bundle_log else "fail",
            "expected": "pass",
            "log": BUNDLE_LOG,
        }],
        "source_validation": {"commit": DOCS, "checks": checks,
                              "passed": all(check["exit"] == 0 for check in checks)},
        "not_run": [
            ("The Nemotron-H Metal/F16 official-precision gate on the 30B checkpoint at the new source: the library "
             "code equals 2cf83d7's, whose run measured every row; the change judges and prints those rows "
             "differently, which the unit tests and the bundle-only run cover; the task forbade a 30B rerun unless a "
             "code change requires it."),
            "Every other real-checkpoint gate, benchmark, and smoke: no library code changed.",
            "check_release.py, CUDA and ROCm runs (Clippy compiled their features), coverage, and Verus.",
            "Official F32 hidden traces and real-weight MTP parity for Nemotron-H (still open).",
        ],
        "open": [
            "The Nemotron-H official-precision failure at cached decision 22 (0.998557984 against 0.998635047).",
            "Burn's teacher decision 30 (0.996039 from F32, where the official F16 graphs reach 0.999346 to 0.999968).",
            ("Whether more official graphs (a regenerated oracle on the current macOS, other experts "
             "implementations) or a repeat-based slack could ever justify a wider bound; none is adopted."),
        ],
        "historical_records": {name: sha256((measurements / name).read_bytes()) for name in HISTORICAL},
    }
    record["files"] = {path.relative_to(campaign).as_posix(): sha256(path.read_bytes())
                       for path in sorted(campaign.rglob("*")) if path.is_file()}
    (measurements / RECORD).write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
