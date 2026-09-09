"""Authenticate compact public-example evidence without loading any weights."""

import hashlib
import json
from pathlib import Path
import unittest


CRATE = Path(__file__).resolve().parents[2]


class PublicExampleRecordTests(unittest.TestCase):
    """Keep functional execution separate from source promotion and timing claims."""

    def test_identities_commands_and_output_hashes(self):
        """Each example belongs to the pinned local checkpoint and retained source."""
        record = json.loads((CRATE / "acceptance/measurements/public-examples-20260909.json").read_text())
        self.assertFalse(record["standardized_benchmark"])
        self.assertFalse(record["final_candidate_replay"])
        self.assertFalse(record["network_access"])
        self.assertEqual(record["cargo_lock_sha256"], hashlib.sha256((CRATE / "Cargo.lock").read_bytes()).hexdigest())
        self.assertEqual(len(record["runs"]), 2)
        for run in record["runs"]:
            with self.subTest(example=run["example"]):
                primary = json.loads((CRATE / "acceptance" / run["primary_acceptance"]).read_text())
                self.assertEqual(run["repository"], primary["repository"])
                self.assertEqual(run["revision"], primary["revision"])
                self.assertEqual(run["exit_code"], 0)
                self.assertEqual(run["example_sha256"], hashlib.sha256((CRATE / run["example"]).read_bytes()).hexdigest())
                self.assertEqual(run["stdout_sha256"], hashlib.sha256(run["stdout"].encode()).hexdigest())
                self.assertIn("--offline", run["command"])
                self.assertIn("--locked", run["command"])
                self.assertIn("--release", run["command"])
                self.assertEqual(run["command"][1], "+stable")
                self.assertTrue(run["stdout"].endswith("\n"))
                self.assertGreater(run["process_wall_seconds"], run["compile_seconds"])


if __name__ == "__main__":
    unittest.main()
