"""Validate the October 18 Qwen3.5-0.8B benchmark re-baseline record without a GPU.

The record is published with the crate; its campaign directory
``acceptance/measurements/qwen3.5-0.8b-metal-rebaseline-20261018/`` (raw
reports, per-process logs and host records, the plan, the build logs, and
the harness copy) is repository-only, so every read of it goes through
:func:`repository_evidence.repository_evidence`: in the extracted package
those tests skip, and in a Git checkout a missing file fails them. The tests
recompute every sample, the steady state, the reference and control blocks,
and every comparison from the raw reports with the crate's own
``scripts/qwen35_benchmark_protocol.py``; check that the plan follows the
protocol and that every process was admitted under it; and require the
superseded September 10 records and the earlier rounds' records to keep
their bytes.
"""

import functools
import hashlib
import importlib.util
import json
import re
import unittest
from pathlib import Path

from repository_evidence import repository_evidence


ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS = ROOT / "acceptance/measurements"
RECORD = MEASUREMENTS / "qwen3.5-0.8b-metal-rebaseline-20261018.json"
SEPTEMBER = MEASUREMENTS / "small-model-performance-20260910"
SPEC = importlib.util.spec_from_file_location("qwen35_benchmark_protocol", ROOT / "scripts/qwen35_benchmark_protocol.py")
assert SPEC is not None and SPEC.loader is not None
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)
#: The maintainer's decision, quoted exactly.
REASON = ("re-baseline the Qwen3.5-0.8B benchmarks on the current host (macOS 27.0.1, same Mac15,11), "
          "and report cold first token as a diagnostic rather than a pass/fail gate.")
#: Source commits of the two measured binaries.
SOURCES = {"head": "fa8426e5dc8cbcb269c29e211b119ea0acd61054",
           "september": "c70057e82510e085bc057910a99886c1ce551567"}
#: Records the re-baseline supersedes or follows, which must keep their bytes.
HISTORICAL = frozenset({
    "v0_1-release-20260910.json",
    "small-model-performance-20260910/manifest.json",
    "small-model-performance-20260910/qwen-controlled-current-summary.json",
    "small-model-performance-20260910/qwen-public-summary.json",
    "half-precision-review-20261018.json",
    "half-precision-envelope-20261018.json",
    "half-precision-followups-20261017.json",
})


def sha256(data):
    """Hexadecimal SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


class Qwen35RebaselineRecordTests(unittest.TestCase):
    """Tie every decision, sample, statistic, and identity of the re-baseline record to its evidence."""

    @classmethod
    def setUpClass(cls):
        """Read the published record; campaign files are read on demand."""
        cls.record = json.loads(RECORD.read_text())
        cls.campaign = MEASUREMENTS / cls.record["campaign_directory"]

    def evidence(self, relative):
        """Bytes of a campaign file named relative to the campaign directory, checked against ``files``."""
        data = repository_evidence(self.campaign / relative).read_bytes()
        self.assertEqual(sha256(data), self.record["files"][relative], relative)
        return data

    @functools.lru_cache(maxsize=None)
    def json_evidence(self, relative):
        """An authenticated campaign JSON file."""
        return json.loads(self.evidence(relative))

    @functools.lru_cache(maxsize=None)
    def summary(self):
        """The protocol's summary of the committed campaign, recomputed from its raw reports."""
        for relative in self.record["files"]:
            if relative.startswith("campaign/"):
                self.evidence(relative)
        samples = PROTOCOL.load_campaign(repository_evidence(self.campaign / "campaign/plan.json").parent)
        summary = PROTOCOL.summarize(samples)
        summary["series"] = PROTOCOL.campaign_series(samples)
        return summary

    def test_record_is_an_authorized_rebaseline_not_a_support_declaration(self):
        """The maintainer's decision is quoted, cold first token is ungated, and no support claim changes."""
        record = self.record
        self.assertEqual(record["schema_version"], 1)
        self.assertFalse(record["advertised_supported"])
        self.assertFalse(record["support_claims_changed"])
        self.assertEqual((record["repository"], record["revision"], record["profile"]),
                         ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "qwen3.5-0.8b-metal-f16"))
        policy = record["performance_policy"]
        self.assertIs(policy["user_authorized_rebaseline"], True)
        self.assertEqual(policy["authorization_date"], "2026-10-18")
        self.assertEqual(policy["quoted_reason"], REASON)
        self.assertIs(policy["cold_first_token_gated"], False)
        self.assertEqual(policy["diagnostic_metrics"], ["cold_first_token"])
        self.assertEqual(policy["cold_first_token_reported_as"], ["median", "range"])
        self.assertEqual((policy["future_alert_percent"], policy["future_failure_percent_after_controlled_rerun"]),
                         (5, 10))
        self.assertEqual(policy["gated_metrics"],
                         {kind: PROTOCOL.gated_metric_names(kind) for kind in PROTOCOL.BENCHMARKS})
        for names in policy["gated_metrics"].values():
            self.assertNotIn("cold_first_token", names)
        self.assertIs(policy["historical_measurements_modified"], False)

    def test_supersedes_the_september_baseline_without_editing_it(self):
        """The superseded release record and summaries are named by their current, unchanged digests."""
        supersedes = self.record["performance_policy"]["supersedes"]
        release_bytes = (MEASUREMENTS / supersedes["record"]).read_bytes()
        self.assertEqual(sha256(release_bytes), supersedes["record_sha256"])
        release = json.loads(release_bytes)["performance_policy"]
        self.assertEqual(release["qwen_model_loop_baseline"], supersedes["qwen_model_loop_baseline"])
        self.assertEqual(release["qwen_model_loop_baseline_sha256"], supersedes["qwen_model_loop_baseline_sha256"])
        self.assertEqual(release["qwen_public_coordinator_baseline"], supersedes["qwen_public_coordinator_baseline"])
        manifest = json.loads((SEPTEMBER / "manifest.json").read_text())["records"]
        for key in ("qwen_model_loop_baseline", "qwen_public_coordinator_baseline"):
            data = (MEASUREMENTS / supersedes[key]).read_bytes()
            self.assertEqual(sha256(data), supersedes[f"{key}_sha256"])
            self.assertEqual(manifest[Path(supersedes[key]).name]["sha256"], supersedes[f"{key}_sha256"])

    def test_files_cover_the_campaign_directory(self):
        """``files`` lists exactly the campaign directory's files, each with its digest."""
        root = repository_evidence(self.campaign / "campaign/plan.json").parents[1]
        present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        self.assertEqual(present, set(self.record["files"]))
        for relative in sorted(present):
            with self.subTest(file=relative):
                self.evidence(relative)

    def test_plan_follows_the_protocol(self):
        """The plan is the protocol's alternating, counterbalanced plan, extended by whole blocks from 8 rounds."""
        plan = self.json_evidence("campaign/plan.json")
        first = self.json_evidence("campaign/plan-8-rounds.json")
        self.assertEqual(first["rounds"], PROTOCOL.MINIMUM_ROUNDS)
        self.assertEqual(plan["samples"][:len(first["samples"])], first["samples"])
        self.assertEqual({key: value for key, value in first.items() if key not in ("rounds", "samples")},
                         {key: value for key, value in plan.items() if key not in ("rounds", "samples")})
        self.assertEqual(plan["rounds"] % PROTOCOL.BLOCK_ROUNDS, 0)
        self.assertLessEqual(plan["rounds"], PROTOCOL.MAXIMUM_ROUNDS)
        self.assertEqual(plan["script_sha256"], sha256(self.evidence("harness/qwen35_benchmark_protocol.py")))
        self.assertEqual(plan["script_sha256"], self.record["protocol"]["campaign_script_sha256"])
        binaries = [{key: binary[key] for key in ("label", "source", "binary", "sha256")} for binary in plan["binaries"]]
        self.assertEqual([binary["label"] for binary in binaries], ["head", "september"])
        artifacts = plan["samples"][0]["artifacts"]
        expected = PROTOCOL.make_plan(plan["rounds"], binaries, artifacts["TRANSFORMERS_QWEN35_DIR"],
                                      artifacts["TRANSFORMERS_QWEN35_REFERENCE_DIR"])
        self.assertEqual(plan["samples"], expected)
        self.assertGreaterEqual(plan["rounds"], PROTOCOL.MINIMUM_ROUNDS)
        self.assertEqual(set(PROTOCOL.position_counts(plan["samples"]).values()), {plan["rounds"] // 4})
        self.assertEqual(plan["constants"]["cooldown_seconds"], PROTOCOL.COOLDOWN_SECONDS)
        self.assertEqual(plan["constants"]["load_threshold_1m"], PROTOCOL.LOAD_THRESHOLD)
        self.assertEqual(plan["constants"]["steady_drift"], PROTOCOL.STEADY_DRIFT)
        for recorded, planned in zip(self.record["binaries"], plan["binaries"]):
            self.assertEqual((recorded["path"], recorded["sha256"]), (planned["binary"], planned["sha256"]))

    def test_binaries_were_built_from_clean_trees_with_rustc_1_92(self):
        """Both executables came from clean checkouts of their commits with rustc 1.92.0 and the profile's features."""
        for binary in self.record["binaries"]:
            directory = Path(binary["build_log"]).parent.as_posix()
            with self.subTest(label=binary["label"]):
                preflight = self.evidence(f"{directory}/preflight.txt").decode()
                result = self.evidence(f"{directory}/result.txt").decode().splitlines()
                self.assertIn(f"== commit {SOURCES[binary['label']]}", preflight.splitlines())
                self.assertEqual(binary["source_commit"], SOURCES[binary["label"]])
                self.assertIn("== dirty 0", preflight.splitlines())
                self.assertIn("rustc 1.92.0 (ded5c06cf 2025-12-08)", preflight.splitlines())
                self.assertIn("== exit 0", " ".join(result[:1]))
                self.assertIn(f"== sha256 {binary['sha256']}", result)
                self.assertIn(f"== commit_after {SOURCES[binary['label']]}", result)
                self.assertIn("== dirty_after 0", result)
                self.assertIn("== artifact transformers compiled", result)
                command = self.evidence(f"{directory}/command.txt").decode()
                self.assertIn("std\\,metal\\,tokenizers\\,chat-templates", command)
                self.assertIn("+1.92.0", command)

    def test_every_sample_was_admitted_and_recorded_under_the_protocol(self):
        """Cooldown, load threshold, AC power, digests, exit status, source identity, and log lines hold for every run."""
        plan = self.json_evidence("campaign/plan.json")
        rows = self.record["samples"]
        self.assertEqual(len(rows), len(plan["samples"]))
        for row, planned in zip(rows, plan["samples"]):
            with self.subTest(sample=row["index"]):
                host = self.json_evidence(row["host_record"])
                report = self.json_evidence(row["report"])
                self.assertEqual(self.record["files"][row["report"]], row["report_sha256"])
                self.assertEqual(host["sample"], planned)
                self.assertEqual(host["exit"], 0)
                self.assertEqual(host["binary_sha256_before"], planned["sha256"])
                self.assertEqual(host["binary_sha256_after"], planned["sha256"])
                self.assertEqual(report["provenance"]["benchmark_executable_sha256"], planned["sha256"])
                self.assertEqual(report["provenance"]["crate_commit"]["value"], SOURCES[row["label"]])
                self.assertEqual(report["provenance"]["crate_worktree_status"]["value"], "")
                self.assertEqual(row["crate_commit"], SOURCES[row["label"]])
                self.assertGreaterEqual(host["admission"]["cooldown_seconds"], PROTOCOL.COOLDOWN_SECONDS)
                self.assertLessEqual(host["admission"]["admitting_load_1m"], PROTOCOL.LOAD_THRESHOLD)
                self.assertEqual(row["admission"], host["admission"])
                self.assertTrue(host["host_before"]["power"].startswith("Now drawing from 'AC Power'"))
                self.assertEqual(row["metal_cache_growth_bytes"], host["metal_cache_growth"]["growth_bytes"])
                self.assertEqual(host["metal_cache_growth"],
                                 PROTOCOL.cache_growth(host["host_before"]["metal_cache"],
                                                       host["host_after"]["metal_cache"]))
                metrics = PROTOCOL.run_metrics(report, row["benchmark"])
                self.assertEqual(row["gated"], metrics["gated"])
                self.assertEqual(row["cold_first_token_seconds"], metrics["diagnostic"]["cold_first_token"])
                self.assertEqual(row["tokens"], PROTOCOL.token_identity(report, row["benchmark"]))
                log = self.evidence(row["log"]).decode().splitlines()
                self.assertIn(f"== binary {planned['binary']} sha256 {planned['sha256']}", log)
                self.assertIn(f"test {planned['test']} ... ok", log)
                self.assertTrue(any(line.startswith("test result: ok. 1 passed; 0 failed;") for line in log))
                self.assertTrue(any(line.startswith("== exit=0 ") for line in log))
                self.assertIs(host["clean"], True)
                self.assertLessEqual(host["in_run_load_1m_maximum"], PROTOCOL.IN_RUN_LOAD_LIMIT)
                self.assertLessEqual(host["admission"]["admitting_load_5m"], PROTOCOL.SUSTAINED_LOAD_THRESHOLD)
                self.assertIn(f"== in-run load_1m maximum {host['in_run_load_1m_maximum']} over "
                              f"{len(host['in_run_load_1m'])} observations; limit {PROTOCOL.IN_RUN_LOAD_LIMIT}; clean", log)
        cells = {}
        for row in rows:
            cells.setdefault((row["label"], row["benchmark"]), []).append(row["round"])
        self.assertEqual(set(cells), {(label, kind) for label in SOURCES for kind in PROTOCOL.BENCHMARKS})
        for rounds in cells.values():
            self.assertEqual(sorted(rounds), list(range(1, plan["rounds"] + 1)))

    def test_contaminated_and_aborted_attempts_are_kept_and_unused(self):
        """Every contaminated attempt exceeded the in-run limit and was rerun; the aborted attempt is unused."""
        execution = self.record["execution"]
        attempts = execution["contaminated_attempts"]
        named = {Path(relative).name for relative in self.record["files"] if ".contaminated-" in relative}
        self.assertEqual(len(named), 3 * len(attempts))
        clean = {(row["round"], row["position"]) for row in self.record["samples"]}
        for attempt in attempts:
            with self.subTest(attempt=attempt["host_record"]):
                host = self.json_evidence(attempt["host_record"])
                self.assertIs(host["clean"], False)
                self.assertEqual(host["exit"], 0)
                self.assertGreater(host["in_run_load_1m_maximum"], PROTOCOL.IN_RUN_LOAD_LIMIT)
                self.assertEqual(attempt["in_run_load_1m_maximum"], host["in_run_load_1m_maximum"])
                self.assertIn("CONTAMINATED", self.evidence(attempt["log"]).decode())
                self.evidence(attempt["report"])
                self.assertIn((attempt["round"], attempt["position"]), clean)
        self.assertEqual(execution["processes"], len(self.record["samples"]) + len(attempts))
        aborted = self.record["aborted_attempt"]
        self.assertIs(aborted["used"], False)
        plan = self.json_evidence(f"{aborted['directory']}/plan.json")
        self.assertEqual(plan["script_sha256"], aborted["protocol_script_sha256"])
        self.assertEqual(sha256(self.evidence(f"{aborted['directory']}/qwen35_benchmark_protocol.py")),
                         aborted["protocol_script_sha256"])
        hosts = [relative for relative in self.record["files"]
                 if relative.startswith(f"{aborted['directory']}/host/")]
        self.assertEqual(len(hosts), aborted["samples"])
        self.assertNotIn(aborted["directory"], json.dumps(self.record["samples"]))

    def test_execution_summary_is_recomputed(self):
        """The admission, load, timing, and reference-check summary follows from the host records and reports."""
        execution = self.record["execution"]
        hosts = [self.json_evidence(relative) for relative in self.record["files"]
                 if relative.startswith("campaign/host/")]
        clean = [host for host in hosts if host["clean"]]
        self.assertEqual(execution["processes"], len(hosts))
        self.assertEqual(execution["clean_samples"], len(clean))
        self.assertEqual(execution["first_sample_started_utc"], min(host["host_before"]["utc"] for host in hosts))
        self.assertEqual(execution["last_sample_ended_utc"], max(host["host_after"]["utc"] for host in hosts))
        self.assertEqual(execution["maximum_in_run_load_1m_of_clean_samples"],
                         max(host["in_run_load_1m_maximum"] for host in clean))
        self.assertGreaterEqual(execution["minimum_cooldown_seconds"], PROTOCOL.COOLDOWN_SECONDS)
        self.assertEqual(execution["power"], ["Now drawing from 'AC Power'"])
        for label, checks in execution["in_run_reference_checks"].items():
            with self.subTest(label=label):
                self.assertGreaterEqual(checks["minimum_cached_reference_cosine"], 0.999)
                self.assertGreaterEqual(checks["minimum_teacher_reference_cosine"], 0.999)
                self.assertEqual(checks["reference_cached_decisions"], [32])
                self.assertEqual(len(checks["prompt_reference_cosine"]), 1)
        early = [sample for sample in PROTOCOL.load_campaign(self.campaign / "campaign") if sample["round"] <= 8]
        self.assertEqual(execution["steady_state_after_8_rounds"], PROTOCOL.summarize(early)["steady_state"])
        self.assertIsNone(execution["steady_state_after_8_rounds"]["first_round"])

    def test_steady_state_reference_and_control_are_recomputed(self):
        """The steady window, the head reference, and the September control follow from the raw reports."""
        summary = self.summary()
        self.assertEqual(self.record["steady_state"], summary["steady_state"])
        self.assertIsNotNone(summary["steady_state"]["first_round"])
        self.assertLessEqual(summary["steady_state"]["maximum_absolute_drift"], PROTOCOL.STEADY_DRIFT)
        self.assertEqual(self.record["reference"], PROTOCOL.reference_block(summary, "head"))
        self.assertEqual(self.record["september_rebuild"], PROTOCOL.reference_block(summary, "september"))
        for kind in PROTOCOL.BENCHMARKS:
            for label in ("head", "september"):
                self.assertEqual(self.record["cold_first_token"][kind][label],
                                 summary["cells"][f"{label}/{kind}"]["diagnostic"]["cold_first_token"])

    def test_reference_judges_itself_ok_and_never_gates_cold(self):
        """Judging the head against its own reference passes; cold first token is only reported."""
        verdict = PROTOCOL.judge(self.record["reference"], self.summary(), "head", "september")
        for name, item in verdict["metrics"].items():
            with self.subTest(metric=name):
                if name.endswith("cold_first_token"):
                    self.assertEqual(item["status"], "diagnostic")
                    self.assertIs(item["gated"], False)
                else:
                    self.assertEqual(item["ratio"], 1.0)
                    self.assertEqual(item["status"], "ok")

    def test_comparisons_are_recomputed(self):
        """Head over the September rebuild, and both over the approved September 10 values, follow from the blocks."""
        manifest = json.loads((SEPTEMBER / "manifest.json").read_text())["records"]
        for kind, name in (("baseline", "qwen-controlled-current-summary.json"), ("public", "qwen-public-summary.json")):
            data = (SEPTEMBER / name).read_bytes()
            self.assertEqual(sha256(data), manifest[name]["sha256"])
            approved = PROTOCOL.run_metrics(json.loads(data)["benchmark"], kind)
            self.assertEqual(self.record["september_10_approved"][kind]["tokens"],
                             PROTOCOL.token_identity(json.loads(data)["benchmark"], kind))
            self.assertEqual(self.record["reference"]["benchmarks"][kind]["tokens"],
                             self.record["september_10_approved"][kind]["tokens"])
            for metric, row in self.record["comparisons"][kind].items():
                with self.subTest(kind=kind, metric=metric):
                    if metric == "cold_first_token":
                        head = self.record["reference"]["benchmarks"][kind]["diagnostic"][metric]["median"]
                        base = self.record["september_rebuild"]["benchmarks"][kind]["diagnostic"][metric]["median"]
                        september = approved["diagnostic"][metric]
                    else:
                        head = self.record["reference"]["benchmarks"][kind]["gated"][metric]["median"]
                        base = self.record["september_rebuild"]["benchmarks"][kind]["gated"][metric]["median"]
                        september = approved["gated"][metric]["median"]
                    self.assertEqual((row["head"], row["september_rebuild"], row["september_10_approved"]),
                                     (head, base, september))
                    self.assertEqual(row["head_over_september_rebuild"], head / base)
                    self.assertEqual(row["head_over_september_10"], head / september)
                    self.assertEqual(row["september_rebuild_over_september_10"], base / september)

    def test_every_quote_is_in_its_log(self):
        """Every quote in the record is part of a line of the campaign file it cites."""
        for entry in self.record.get("quotes", []):
            lines = self.evidence(entry["log"]).decode().splitlines()
            for quote in entry["quotes"]:
                with self.subTest(log=entry["log"], quote=quote[:60]):
                    self.assertGreaterEqual(len(quote), 20)
                    self.assertTrue(any(quote in line for line in lines), quote)

    def test_historical_records_keep_their_bytes(self):
        """The superseded and earlier records keep the bytes they had when the re-baseline was made."""
        self.assertEqual(set(self.record["historical_records"]), HISTORICAL)
        for name, digest in self.record["historical_records"].items():
            with self.subTest(record=name):
                self.assertEqual(sha256((MEASUREMENTS / name).read_bytes()), digest)


if __name__ == "__main__":
    unittest.main()
