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

    def test_integrated_functional_replay_is_authenticated_and_scoped(self):
        """Four numerical/API tests and two examples cannot imply a release promotion."""
        relative = "measurements/small-model-functional-replay-6226170-20260909.json"
        raw = (CRATE / "acceptance" / relative).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "d1741745f8450414192e7d6d2e0de6cff171d98440e787bd8d88afdb84f85524")
        record = json.loads(raw)
        self.assertEqual(record["scope"], "functional_replay_not_benchmark")
        self.assertEqual(record["source_commit"], "6226170f25a4760d59ba1b391d86e3bce46a1bb7")
        self.assertTrue(record["source_worktree_clean"])
        self.assertEqual(record["source_edits"], [])
        self.assertFalse(record["summary"]["performance_claims"])
        self.assertEqual(record["summary"]["passed_test_cases"], 4)
        self.assertEqual(record["summary"]["public_examples_passed"], 2)
        self.assertEqual(record["toolchain"]["profile"], "release")
        self.assertIn("1.92.0", record["toolchain"]["rustc"])
        for filename in (
            "qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17.json",
            "flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.json",
        ):
            primary = json.loads((CRATE / "acceptance" / filename).read_text())
            self.assertEqual(primary["integrated_functional_replay_record"], relative)
        runs = {run["name"]: run for run in record["runs"]}
        self.assertEqual(len(runs), 5)
        for run in runs.values():
            self.assertEqual(run["exit_code"], 0)
            self.assertIn("--locked --offline --release", run["command"])
            for field in ("log_sha256", "executable_sha256"):
                self.assertRegex(run[field], r"^[0-9a-f]{64}$")
        qwen = runs["qwen_official_inference"]["results"]
        self.assertEqual(qwen["exact_cached_and_stateless_official_greedy_tokens"], 32)
        self.assertGreaterEqual(qwen["minimum_teacher_official_full_vocabulary_cosine"], 0.999)
        flan = runs["flan_official_metal_oracle"]["results"]
        self.assertEqual(flan["exact_greedy_official_tokens"], 32)
        self.assertEqual(flan["tolerance_failures_1e_4"], 0)
        self.assertEqual(len(flan["vector_details"]), 7)
        for vector in flan["vector_details"]:
            self.assertEqual(vector["failures_1e_4"], 0)
            self.assertGreaterEqual(vector["cosine"], 0.999999)
        for name in ("flan_public_example", "qwen_public_example"):
            output = runs[name]["results"]
            self.assertEqual(
                hashlib.sha256((output["output"] + "\n").encode()).hexdigest(),
                output["output_utf8_line_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
