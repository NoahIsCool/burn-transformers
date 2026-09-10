"""Validate retained small-model replay evidence without weights or GPU access."""

import hashlib
import json
from pathlib import Path
import re
import unittest


MEASUREMENTS = Path(__file__).resolve().parents[2] / "acceptance/measurements"


class ParameterReloadReplayRecordTests(unittest.TestCase):
    """Keep native success, environment failure and exact vector hashes distinct."""

    @classmethod
    def setUpClass(cls):
        """Read only the compact committed records and terminal output captures."""

        cls.record = json.loads((MEASUREMENTS / "small-model-parameter-reload-replay-20260910.json").read_text())
        cls.prior = json.loads((MEASUREMENTS / "small-model-post-json-validation-replay-20260910.json").read_text())

    def test_source_scope_and_serialized_commands(self):
        """A non-regression replay is not a release or benchmark promotion."""

        record = self.record
        self.assertEqual(record["source_commit"], "f5482eaa9ff3e65f575bedb47e12e1bb1d455436")
        self.assertEqual(record["runtime_source_tree_git_oid"], "ff1966773e6c141e73dfec052e93a0f3f8453d47")
        self.assertTrue(record["passed"])
        self.assertFalse(record["advertised_supported"])
        self.assertEqual(record["cargo_lock_sha256"], self.prior["cargo_lock_sha256"])
        self.assertEqual(record["common_arguments"], self.prior["common_arguments"])
        for current, prior in zip(record["runs"], self.prior["runs"], strict=True):
            self.assertEqual(current["environment"], prior["environment"])
            self.assertEqual(current["argument_suffix"], prior["argument_suffix"])
            self.assertIn("--test-threads=1", current["argument_suffix"])
            self.assertEqual(current["revision"], prior["revision"])
            self.assertEqual(current["exit_code"], 0)
            self.assertRegex(current["executable_sha256"], r"^[0-9a-f]{64}$")

    def test_raw_outputs_are_authenticated_and_failure_is_not_counted_as_pass(self):
        """A denied GPU adapter remains a failed attempt, not omitted parity data."""

        for run in self.record["runs"] + self.record["failed_attempts"]:
            relative = Path(run["log"])
            self.assertFalse(relative.is_absolute())
            self.assertNotIn("..", relative.parts)
            data = (MEASUREMENTS / relative).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), run["log_sha256"])
            log = data.decode()
            if run["exit_code"] == 0:
                self.assertIn(f"test result: ok. {run['passed_test_cases']} passed; 0 failed", log)
                self.assertNotIn("panicked", log)
            else:
                self.assertEqual(run["exit_code"], 101)
                self.assertEqual(run["passed_test_cases"], 0)
                self.assertEqual(run["failed_test_cases"], 2)
                self.assertIn("test result: FAILED. 0 passed; 2 failed", log)
                self.assertIn("No possible adapter available", log)

    def test_reported_parity_agrees_with_retained_outputs_and_prior_vectors(self):
        """Recompute vector/hash comparisons instead of trusting a summary label."""

        qwen, flan = self.record["runs"]
        qwen_log = (MEASUREMENTS / qwen["log"]).read_text()
        self.assertIn("q4: tensors=186, top1=13, cached_cosine=1.000000000", qwen_log)
        self.assertEqual(qwen["q4"], self.prior["runs"][0]["q4"])
        self.assertEqual(qwen["f16"], self.prior["runs"][0]["f16"])
        for key in ("prompt_official_cosine", "minimum_cached_official_cosine", "minimum_teacher_official_cosine"):
            self.assertIn(f"{qwen['f16'][key]:.9f}", qwen_log)
        pattern = r"Metal/F32 (\w+): cosine=([0-9.]+) max_abs=([0-9.]+) failures_1e-4=(\d+) actual_f32le_sha256=([0-9a-f]{64})"
        rows = re.findall(pattern, (MEASUREMENTS / flan["log"]).read_text())
        self.assertEqual(len(rows), 7)
        self.assertEqual({row[0]: row[4] for row in rows}, self.prior["runs"][1]["vectors_f32le_sha256"])
        self.assertEqual(min(float(row[1]) for row in rows), flan["minimum_full_vector_cosine"])
        self.assertEqual(max(float(row[2]) for row in rows), flan["maximum_absolute_error"])
        self.assertEqual(sum(int(row[3]) for row in rows), flan["failures_at_1e_4"])


if __name__ == "__main__":
    unittest.main()
