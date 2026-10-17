"""Validate the October 16 reference-precision follow-up record without checkpoints or a GPU."""

import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS = ROOT / "acceptance/measurements"
RECORD = MEASUREMENTS / "reference-precision-followups-20261016.json"


class ReferencePrecisionFollowupsRecordTests(unittest.TestCase):
    """Tie every summarized result to the committed evidence and keep the record non-promoting."""

    @classmethod
    def setUpClass(cls):
        """Read only the committed record, its log, and its auxiliary files."""
        cls.record = json.loads(RECORD.read_text())
        cls.log = (MEASUREMENTS / cls.record["log"]).read_text()

    def test_record_is_evidence_not_a_support_declaration(self):
        """The record names its exact sources and advertises nothing."""
        record = self.record
        self.assertEqual(record["schema_version"], 1)
        self.assertFalse(record["advertised_supported"])
        self.assertTrue(record["passed"])
        self.assertRegex(record["cargo_lock_sha256"], r"^[0-9a-f]{64}$")
        commits = [source["commit"] for source in record["sources"]]
        self.assertEqual(commits[0], record["source_commit"])
        for commit in commits:
            with self.subTest(source=commit):
                self.assertRegex(commit, r"^[0-9a-f]{40}$")
                self.assertIn(f"== source {commit} dirty=0", self.log)
        self.assertLessEqual({run["source_commit"] for run in record["runs"]}, set(commits))

    def test_log_and_auxiliary_files_match_their_digests(self):
        """The log and every auxiliary evidence file are authenticated by their recorded SHA-256."""
        digest = hashlib.sha256((MEASUREMENTS / self.record["log"]).read_bytes()).hexdigest()
        self.assertEqual(digest, self.record["log_sha256"])
        for auxiliary in self.record["auxiliary_files"]:
            with self.subTest(path=auxiliary["path"]):
                data = (MEASUREMENTS / auxiliary["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), auxiliary["sha256"])

    def test_every_process_is_accounted_for(self):
        """Each process in the log is a listed run or a diagnosed superseded attempt."""
        runs = self.record["runs"]
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
        self.assertTrue(all(run["failed"] == 0 for run in runs))
        self.assertEqual(
            sorted(int(code) for code in exits if code != "0"), sorted(a["exit"] for a in aborted)
        )
        passed_tests = sum(run["passed"] for run in runs if run["kind"] == "test")
        self.assertEqual(
            len(re.findall(r"^test result: ok\. 1 passed", self.log, flags=re.MULTILINE)), passed_tests
        )

    def test_every_summary_is_quoted_from_the_log(self):
        """Each run's measurement strings occur verbatim in the committed log."""
        for run in self.record["runs"]:
            with self.subTest(run=run["name"]):
                self.assertRegex(run["revision"], r"^[0-9a-f]{40}$")
                self.assertEqual(run["passed"] + run["failed"], max(len(run["tests"]), 1))
                self.assertTrue(run["measurements"])
                for measurement in run["measurements"]:
                    self.assertIn(measurement, self.log)

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
