"""Validate the October 16 reference-precision follow-up record without checkpoints or a GPU.

The record itself is published with the crate; its logs and auxiliary files
live in a repository-only campaign directory, so every read of them goes
through :func:`repository_evidence.repository_evidence`. In the extracted
package those tests skip; in a Git checkout a missing file still fails them.
"""

import functools
import hashlib
import json
from pathlib import Path
import re
import unittest

from repository_evidence import repository_evidence


ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS = ROOT / "acceptance/measurements"
RECORD = MEASUREMENTS / "reference-precision-followups-20261016.json"


class ReferencePrecisionFollowupsRecordTests(unittest.TestCase):
    """Tie every summarized result to the committed evidence and keep the record non-promoting."""

    @classmethod
    def setUpClass(cls):
        """Read only the published record; its logs and auxiliary files are read on demand."""
        cls.record = json.loads(RECORD.read_text())

    @functools.cached_property
    def log(self):
        """The campaign log, which only a Git checkout carries; the published crate skips."""
        return repository_evidence(MEASUREMENTS / self.record["log"]).read_text()

    def log_of(self, entry):
        """Returns the log a source or run names, defaulting to the campaign log."""
        path = entry.get("log", self.record["log"])
        if path == self.record["log"]:
            return self.log
        return repository_evidence(MEASUREMENTS / path).read_text()

    def campaign_runs(self):
        """Runs whose output was appended to the campaign log."""
        return [run for run in self.record["runs"] if run.get("log", self.record["log"]) == self.record["log"]]

    def separately_logged_runs(self):
        """Runs recorded in a log of their own beside the campaign log."""
        return [run for run in self.record["runs"] if run.get("log", self.record["log"]) != self.record["log"]]

    def test_record_is_evidence_not_a_support_declaration(self):
        """The record names its exact sources and advertises nothing."""
        record = self.record
        self.assertEqual(record["schema_version"], 1)
        self.assertFalse(record["advertised_supported"])
        self.assertTrue(record["passed"])
        self.assertRegex(record["cargo_lock_sha256"], r"^[0-9a-f]{64}$")
        sources = record["sources"]
        self.assertEqual(sources[0]["commit"], record["source_commit"])
        self.assertFalse(sources[0].get("dirty", False))
        for source in sources:
            with self.subTest(source=source["commit"]):
                self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")
                dirty = bool(source.get("dirty", False))
                log = self.log_of(source)
                self.assertIn(f"== source {source['commit']} dirty={int(dirty)}", log)
                if dirty:
                    # A dirty source keeps its own log, which names every uncommitted file by digest.
                    self.assertNotEqual(source["log"], record["log"])
                    self.assertRegex(log, r"(?m)^== uncommitted \S+ sha256 [0-9a-f]{64}$")
        commits = {source["commit"] for source in sources}
        self.assertLessEqual({run["source_commit"] for run in record["runs"]}, commits)

    def test_log_and_auxiliary_files_match_their_digests(self):
        """Every log and auxiliary evidence file is authenticated by its recorded SHA-256."""
        digest = hashlib.sha256(repository_evidence(MEASUREMENTS / self.record["log"]).read_bytes()).hexdigest()
        self.assertEqual(digest, self.record["log_sha256"])
        for run in self.separately_logged_runs():
            with self.subTest(log=run["log"]):
                data = repository_evidence(MEASUREMENTS / run["log"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), run["log_sha256"])
        for auxiliary in self.record["auxiliary_files"]:
            with self.subTest(path=auxiliary["path"]):
                data = repository_evidence(MEASUREMENTS / auxiliary["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), auxiliary["sha256"])

    def test_every_process_is_accounted_for(self):
        """Each process in the campaign log is a listed run or a diagnosed superseded attempt."""
        runs = self.campaign_runs()
        aborted = self.record["aborted_attempts"]
        exits = re.findall(r"== exit=(-?\d+) ", self.log)
        log_tests = re.findall(r"^test (\S+) \.\.\. ", self.log, flags=re.MULTILINE)
        killed = re.findall(r"WATCHDOG_KILLED: (.+)$", self.log, flags=re.MULTILINE)
        self.assertEqual(killed, [a["watchdog_reason"] for a in aborted if "watchdog_reason" in a])
        for attempt in aborted:
            with self.subTest(attempt=attempt["log_label"]):
                self.assertNotEqual(attempt["exit"], 0)
                self.assertTrue(attempt["diagnosis"])
                self.assertIn(f"== {attempt['log_label']}\n", self.log)
        listed = [test for run in runs if run["kind"] == "test" for test in run["tests"]]
        self.assertEqual(sorted(log_tests), sorted(listed + [a["test"] for a in aborted if "test" in a]))
        other = [run for run in runs if run["kind"] != "test"]
        aborted_other = [a for a in aborted if "test" not in a]
        self.assertEqual(len(exits), len(log_tests) + len(other) + len(aborted_other))
        self.assertTrue(all(run["failed"] == 0 for run in self.record["runs"]))
        self.assertEqual(
            sorted(int(code) for code in exits if code != "0"), sorted(a["exit"] for a in aborted)
        )
        passed_tests = sum(run["passed"] for run in runs if run["kind"] == "test")
        self.assertEqual(
            len(re.findall(r"^test result: ok\. 1 passed", self.log, flags=re.MULTILINE)), passed_tests
        )

    def test_separately_logged_runs_are_their_logs_only_process(self):
        """A run with its own log is that log's one process, from its own source, with its exact command."""
        runs = self.separately_logged_runs()
        self.assertTrue(runs)
        sources = {source["commit"]: source for source in self.record["sources"]}
        for run in runs:
            with self.subTest(run=run["name"]):
                log = self.log_of(run)
                self.assertEqual(sources[run["source_commit"]].get("log"), run["log"])
                self.assertEqual(re.findall(r"^== source (\S+) ", log, flags=re.MULTILINE), [run["source_commit"]])
                self.assertIn(f"== {run['log_label']}\n", log)
                self.assertIn(f"== command {run['command']}\n", log)
                self.assertEqual(re.findall(r"^== exit=(-?\d+) ", log, flags=re.MULTILINE), ["0"])
                self.assertEqual(re.findall(r"^test (\S+) \.\.\. ", log, flags=re.MULTILINE), run["tests"])
                self.assertEqual(
                    len(re.findall(r"^test result: ok\. 1 passed", log, flags=re.MULTILINE)), run["passed"]
                )
                self.assertNotIn("test result: FAILED", log)

    def test_every_summary_is_quoted_from_the_log(self):
        """Each run's measurement strings occur verbatim in the log that recorded it."""
        for run in self.record["runs"]:
            with self.subTest(run=run["name"]):
                self.assertRegex(run["revision"], r"^[0-9a-f]{40}$")
                self.assertEqual(run["passed"] + run["failed"], max(len(run["tests"]), 1))
                self.assertTrue(run["measurements"])
                log = self.log_of(run)
                for measurement in run["measurements"]:
                    self.assertIn(measurement, log)

    def test_bundle_comparison_calibrates_the_bound_without_counting_as_a_gate(self):
        """The official BF16-against-F32 comparison names both bundles, evaluates no Burn model, and fits 2.7/0.15."""
        comparisons = [run for run in self.record["runs"] if run["kind"] == "reference_comparison"]
        self.assertEqual([run["name"] for run in comparisons], ["gpt-oss-20b-transformers-bf16-vs-f32-bundles"])
        run, = comparisons
        self.assertIs(run["burn_gate"], False)
        self.assertNotIn("reference", run)
        gated = {
            other["reference"]["bundle"]: other["reference"]["reference_json_sha256"]
            for other in self.record["runs"]
            if "reference_json_sha256" in other.get("reference", {})
        }
        created = {
            f"{bundle['location']}/{bundle['path']}": bundle["manifest_sha256"]
            for bundle in self.record["created_bundles"]
        }
        references = run["references"]
        self.assertEqual(sorted(references), ["bfloat16", "float32"])
        self.assertEqual(created[references["float32"]["bundle"]], references["float32"]["reference_json_sha256"])
        log = self.log_of(run)
        for reference in references.values():
            with self.subTest(bundle=reference["bundle"]):
                # The same bundles, by manifest digest, as the Burn gates that the bound governs.
                self.assertEqual(gated[reference["bundle"]], reference["reference_json_sha256"])
                self.assertIn(f"{reference['bundle']}: reference.json SHA-256 {reference['reference_json_sha256']}", log)
        prompts = [
            re.search(r"^prompt \d+: \d+ tokens, max \|Δ\| = (\S+) at .*, mean \|Δ\| = (\S+)$", measurement)
            for measurement in run["measurements"]
            if measurement.startswith("prompt ")
        ]
        self.assertEqual(len(prompts), 4)
        largest = max(float(match.group(1)) for match in prompts)
        largest_mean = max(float(match.group(2)) for match in prompts)
        # The documented figures (2.665 and 0.108) and the bounds they support (2.7 and 0.15).
        self.assertEqual(round(largest, 3), 2.665)
        self.assertEqual(round(largest_mean, 3), 0.108)
        self.assertLess(largest, 2.7)
        self.assertLess(largest_mean, 0.15)

    def test_half_precision_bundle_runs_are_informational(self):
        """Burn runs against the BF16 bundle compare two roundings and are not presented as gate passes."""
        bf16 = [run for run in self.record["runs"] if run.get("reference", {}).get("bundle", "").endswith("-bf16-reference.bundle")]
        self.assertEqual(
            sorted(run["name"] for run in bf16),
            ["gpt-oss-20b-full-bf16-metal-f16", "gpt-oss-20b-full-bf16-metal-f16-greedy"],
        )
        for run in bf16:
            with self.subTest(run=run["name"]):
                self.assertIn("informational", run["gate_role"])
                self.assertIn("not a gate pass", run["gate_role"])

    def test_new_bundles_carry_their_identities(self):
        """The generated F32 bundles name every file by SHA-256, and the log carries those digests."""
        for bundle in self.record["created_bundles"]:
            with self.subTest(bundle=bundle["path"]):
                self.assertRegex(bundle["manifest_sha256"], r"^[0-9a-f]{64}$")
                for name, digest in bundle["files_sha256"].items():
                    self.assertRegex(digest, r"^[0-9a-f]{64}$")
                    self.assertIn(f"{digest}  {bundle['path']}/{name}", self.log)

    def test_qwen38_comparison_alternates_and_reports_each_round(self):
        """The study alternates base with current, then with the fix, and quotes each round."""
        study = self.record["qwen38_memory_and_timing"]
        names = [round_["variant"] for round_ in study["rounds"]]
        self.assertEqual(
            names,
            ["base", "current", "base", "current", "first-bad-commit", "base", "fixed", "base", "fixed"],
        )
        for round_ in study["rounds"]:
            with self.subTest(round=round_["log_label"]):
                self.assertIn(f"== {round_['log_label']}\n", self.log)
                self.assertRegex(round_["log_label"], r"load averages [0-9.]+ [0-9.]+ [0-9.]+ at start")
                self.assertIn(f"stateless_seconds={round_['stateless_seconds']:.3f}", self.log)
                self.assertIn(f"{round_['peak_memory_footprint_bytes']}  peak memory footprint", self.log)

        def peaks(variant):
            """Peak footprints of one variant's rounds, in round order."""
            return [r["peak_memory_footprint_bytes"] for r in study["rounds"] if r["variant"] == variant]

        # Ordering claims of the summary: first bad > current > base > fixed, in every round.
        self.assertGreater(min(peaks("first-bad-commit")), max(peaks("current")))
        self.assertGreater(min(peaks("current")), max(peaks("base")))
        self.assertGreater(min(peaks("base")), max(peaks("fixed")))
        digests = {r["variant"]: r["stateless_logits_fnv1a64"] for r in study["rounds"]}
        self.assertEqual(digests["base"], "a3ae49dc10b7fbea")
        self.assertEqual(digests["current"], digests["fixed"])

    def test_muse_audit_comparison_alternates_and_the_fix_is_lower(self):
        """The Muse streaming audit alternates pre-fix and fixed binaries; every fixed peak is lower."""
        rounds = [run for run in self.record["runs"] if run["name"].startswith("muse-glimmer-30b-audit-round-")]
        self.assertEqual(
            [re.search(r"round-(\d+)-(.+)$", run["name"]).groups() for run in rounds],
            [("1", "pre-fix"), ("1", "fixed"), ("2", "pre-fix"), ("2", "fixed")],
        )

        def peaks(variant):
            """Peak footprints of one variant's audit rounds, in round order."""
            return [
                run["peak_memory_footprint_bytes"]["loads_pinned_muse_glimmer_f16_and_audits_streaming"]
                for run in rounds
                if run["name"].endswith(f"-{variant}")
            ]

        for run in rounds:
            with self.subTest(run=run["name"]):
                self.assertIn(f"== {run['log_label']}\n", self.log)
        self.assertGreater(min(peaks("pre-fix")), max(peaks("fixed")))

    def test_binaries_carry_identities(self):
        """Every binary is named by digest and source commit."""
        for binary in self.record["binaries"]:
            with self.subTest(binary=binary["path"]):
                self.assertRegex(binary["sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(binary["source_commit"], r"^[0-9a-f]{7,40}$")


if __name__ == "__main__":
    unittest.main()
