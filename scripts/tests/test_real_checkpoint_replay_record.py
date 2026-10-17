"""Validate the October 16 real-checkpoint replay record without checkpoints or a GPU."""

import functools
import hashlib
import json
from pathlib import Path
import re
import unittest

from repository_evidence import repository_evidence


MEASUREMENTS = Path(__file__).resolve().parents[2] / "acceptance/measurements"
RECORD = MEASUREMENTS / "real-checkpoint-replay-20261016.json"


class RealCheckpointReplayRecordTests(unittest.TestCase):
    """Tie every summarized run to the committed log and keep the record non-promoting."""

    @classmethod
    def setUpClass(cls):
        """Read only the committed record, never the external model paths."""
        cls.record = json.loads(RECORD.read_text())

    @functools.cached_property
    def log(self):
        """The committed replay log, which only a Git checkout carries."""
        return repository_evidence(MEASUREMENTS / self.record["log"]).read_text()

    def test_record_is_evidence_not_a_support_declaration(self):
        """The replay passes, names its exact source, and advertises nothing."""
        record = self.record
        self.assertEqual(record["schema_version"], 1)
        self.assertFalse(record["advertised_supported"])
        self.assertTrue(record["passed"])
        self.assertRegex(record["source_commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(record["cargo_lock_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn(f"== source {record['source_commit']} dirty=0", self.log)

    def test_log_identity_matches_the_committed_bytes(self):
        """The recorded SHA-256 authenticates the exact log that the summaries quote."""
        digest = hashlib.sha256(repository_evidence(MEASUREMENTS / self.record["log"]).read_bytes()).hexdigest()
        self.assertEqual(digest, self.record["log_sha256"])

    def test_every_run_passed_each_listed_test_once(self):
        """Each run's tests appear in the log, and its pass count covers all of them."""
        log_tests = re.findall(r"^test (\S+) \.\.\. ", self.log, flags=re.MULTILINE)
        self.assertEqual(len(re.findall(r"test result: ok\. 1 passed", self.log)), len(log_tests))
        self.assertNotIn("test result: FAILED", self.log)
        self.assertNotIn("panicked", self.log)
        listed = 0
        for run in self.record["runs"]:
            with self.subTest(run=run["name"]):
                self.assertTrue(run["tests"])
                self.assertEqual(run["passed"], len(run["tests"]))
                self.assertEqual(run["failed"], 0)
                self.assertRegex(run["revision"], r"^[0-9a-f]{40}$")
                for measurement in run["measurements"]:
                    self.assertIn(measurement, self.log)
                listed += len(run["tests"])
        self.assertEqual(listed, len(log_tests))

    def test_reference_bundles_carry_their_identities(self):
        """Locally generated bundles name their dtype, config digest, and Transformers version."""
        bundled = [run for run in self.record["runs"] if "reference_json_sha256" in run.get("reference", {})]
        self.assertEqual(
            {run["name"] for run in bundled},
            {"smollm-135m", "qwen3-4b", "smollm3-3b", "qwen2-7b-f64", "llama-3.1-8b",
             "gemma-3-4b-it", "gpt-oss-20b-1layer", "gpt-oss-20b-4layer"},
        )
        for run in bundled:
            with self.subTest(run=run["name"]):
                reference = run["reference"]
                self.assertRegex(reference["reference_json_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(reference["config_sha256"], r"^[0-9a-f]{64}$")
                self.assertIn(reference["dtype"], {"float32", "float64", "bfloat16"})
                self.assertEqual(reference["transformers"], "5.2.0")
        qwen2, = [run for run in self.record["runs"] if run["name"] == "qwen2-7b-f64"]
        self.assertEqual(qwen2["reference"]["dtype"], "float64")
        self.assertIn("TRANSFORMERS_LLAMA_FAMILY_TOLERANCE=2.5e-3", qwen2["features"])


if __name__ == "__main__":
    unittest.main()
