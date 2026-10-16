"""Validate the October 16 large-checkpoint replay record without checkpoints or a GPU."""

import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS = ROOT / "acceptance/measurements"
RECORD = MEASUREMENTS / "large-checkpoint-replay-20261016.json"


class LargeCheckpointReplayRecordTests(unittest.TestCase):
    """Tie every summarized gate to the committed log and keep the record non-promoting."""

    @classmethod
    def setUpClass(cls):
        """Read only the committed record and its log, never the external model paths."""
        cls.record = json.loads(RECORD.read_text())
        cls.log = (MEASUREMENTS / cls.record["log"]).read_text()

    def test_record_is_evidence_not_a_support_declaration(self):
        """The replay names its exact sources and advertises nothing."""
        record = self.record
        self.assertEqual(record["schema_version"], 1)
        self.assertFalse(record["advertised_supported"])
        self.assertRegex(record["cargo_lock_sha256"], r"^[0-9a-f]{64}$")
        commits = [source["commit"] for source in record["sources"]]
        self.assertEqual(commits[0], record["source_commit"])
        for commit in commits:
            with self.subTest(source=commit):
                self.assertRegex(commit, r"^[0-9a-f]{40}$")
                self.assertIn(f"== source {commit} dirty=0", self.log)
        self.assertEqual({run["source_commit"] for run in record["runs"]}, set(commits))

    def test_overall_verdict_matches_the_runs(self):
        """The record passes exactly when every listed gate passed, and says which failed."""
        runs = self.record["runs"]
        self.assertEqual(self.record["passed"], all(run["failed"] == 0 for run in runs))
        for run in runs:
            with self.subTest(run=run["name"]):
                self.assertEqual(len(run.get("failures", [])), run["failed"])
                if run["failed"]:
                    self.assertTrue(run["failure_diagnosis"]["status"])
                    for failure in run["failures"]:
                        self.assertIn(f"thread '{failure['test']}'", self.log)

    def test_log_identity_matches_the_committed_bytes(self):
        """The recorded SHA-256 authenticates the exact log that the summaries quote."""
        digest = hashlib.sha256((MEASUREMENTS / self.record["log"]).read_bytes()).hexdigest()
        self.assertEqual(digest, self.record["log_sha256"])

    def test_every_process_is_accounted_for(self):
        """Each process in the log is a listed gate or a diagnosed superseded attempt."""
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
        self.assertEqual(
            sorted(log_tests), sorted(listed + [a["test"] for a in aborted if "test" in a])
        )
        other = [run for run in runs if run["kind"] != "test"]
        for run in other:
            with self.subTest(run=run["name"]):
                self.assertIn(f"== {run['log_label']}\n", self.log)
        aborted_other = [a for a in aborted if "test" not in a]
        self.assertEqual(len(exits), len(log_tests) + len(other) + len(aborted_other))
        passed_tests = sum(run["passed"] for run in runs if run["kind"] == "test")
        self.assertEqual(
            len(re.findall(r"^test result: ok\. 1 passed", self.log, flags=re.MULTILINE)),
            passed_tests,
        )
        failures = [failure for run in runs for failure in run.get("failures", [])]
        self.assertEqual(
            sorted(int(code) for code in exits if code != "0"),
            sorted([a["exit"] for a in aborted] + [f["exit"] for f in failures]),
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

    def test_public_example_reproduces_the_recorded_text(self):
        """The Qwen3.5 example's stdout digest equals the September public-example record."""
        example, = [run for run in self.record["runs"] if run["kind"] == "example"]
        stdout = example["stdout"]
        self.assertIn(f"{stdout}\n", self.log)
        digest = hashlib.sha256(f"{stdout}\n".encode()).hexdigest()
        self.assertEqual(digest, example["stdout_sha256"])
        previous = json.loads((MEASUREMENTS / "public-examples-20260909.json").read_text())
        previous_run, = [
            run for run in previous["runs"] if run["repository"] == example["repository"]
        ]
        self.assertEqual(previous_run["stdout_sha256"], example["stdout_sha256"])

    def test_claimed_matches_with_earlier_records_hold(self):
        """Every value said to equal an earlier record appears in that record and in the log."""
        for run in self.record["runs"]:
            for match in run.get("matches_earlier_record", []):
                with self.subTest(run=run["name"], value=match["value"]):
                    self.assertIn(match["value"], (ROOT / match["record"]).read_text())
                    self.assertIn(match["value"], self.log)

    def test_binaries_and_artifacts_carry_identities(self):
        """Test binaries and new Burnpacks are named by digest and exact size."""
        commits = {source["commit"] for source in self.record["sources"]}
        for binary in self.record["binaries"]:
            with self.subTest(binary=binary["path"]):
                self.assertRegex(binary["sha256"], r"^[0-9a-f]{64}$")
                self.assertIn(binary["source_commit"], commits)
        for artifact in self.record["created_artifacts"]:
            with self.subTest(artifact=artifact["path"]):
                self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")
                self.assertIn(f"artifact_bytes={artifact['bytes']}", self.log)
                self.assertIn(f"{artifact['sha256']}  {artifact['path']}", self.log)


if __name__ == "__main__":
    unittest.main()
