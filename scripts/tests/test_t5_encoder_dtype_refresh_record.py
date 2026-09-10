"""Validate the compact dtype-fix replay without a checkpoint or GPU."""

import hashlib
import json
import math
from pathlib import Path
import re
import unittest


MEASUREMENTS = Path(__file__).resolve().parents[2] / "acceptance/measurements"


class T5EncoderDtypeRefreshTests(unittest.TestCase):
    """Distinguish numerical acceptance, recorder failure and incomplete coverage."""

    @classmethod
    def setUpClass(cls):
        """Read only committed compact measurements, never their external paths."""
        cls.record = json.loads((MEASUREMENTS / "t5-encoder-dtype-refresh-20260910.json").read_text())

    def test_replay_identities_and_numerical_results(self):
        """Embedded native logs authenticate every independently passing vector."""
        record = self.record
        self.assertFalse(record["advertised_supported"])
        self.assertFalse(record["release_gate_promotion"])
        pattern = r"(?:Flex|Metal)/F32 (\w+): cosine=([\d.]+) max_abs=([\d.]+) failures_1e-4=(\d+) actual_f32le_sha256=([a-f0-9]{64})"
        for run in record["successful_replays"] + record["failed_recorder_attempts"]:
            with self.subTest(run=run["name"]):
                self.assertEqual(run["source_before"], run["source_after"])
                self.assertEqual(run["source_before"]["commit"], record["source_commit"])
                self.assertEqual(run["source_before"]["transformer_worktree_status"], "")
                self.assertIn("--offline", run["build_command"])
                self.assertIn("--locked", run["build_command"])
                self.assertIn("--test-threads=1", run["argv"])
                for key in ("stdout", "stderr"):
                    data = run[key]["text"].encode()
                    self.assertEqual(len(data), run[key]["identity"]["bytes"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), run[key]["identity"]["sha256"])
                self.assertIn("test result: ok. 1 passed; 0 failed; 0 ignored", run["stdout"]["text"])
                rows = re.findall(pattern, run["stderr"]["text"])
                self.assertEqual(len(rows), 7)
                self.assertEqual({row[0] for row in rows}, {"encoder", "decoder", "first", "cached", "teacher", "padded", "boundary"})
                for row, retained in zip(rows, run["vectors"], strict=True):
                    label, cosine, error, failures, digest = row
                    self.assertEqual(retained, {"label": label, "cosine": float(cosine),
                                               "maximum_absolute_error": float(error),
                                               "tolerance_failures": int(failures),
                                               "actual_f32le_sha256": digest})
                    self.assertTrue(math.isfinite(float(cosine)) and math.isfinite(float(error)))
                    self.assertGreaterEqual(float(cosine), 0.999)
                    self.assertEqual(int(failures), 0)

    def test_failed_recorder_is_not_promoted_to_a_successful_run(self):
        """The host retry preserves the binary and does not rewrite the sandbox failure."""
        success = self.record["successful_replays"]
        self.assertEqual([run["backend"] for run in success], ["flex/F32", "metal/F32"])
        self.assertEqual([run["process_exit_code"] for run in success], [0, 0])
        failed, = self.record["failed_recorder_attempts"]
        self.assertEqual(failed["process_exit_code"], 1)
        self.assertEqual(failed["native_tests"], {"passed": 1, "failed": 0, "ignored": 0})
        self.assertIn("time: sysctl kern.clockrate: Operation not permitted", failed["stderr"]["text"])
        self.assertEqual(failed["executable_sha256"], success[0]["executable_sha256"])
        self.assertFalse(self.record["independent_replay_review"]["bit_identical_logits_claimed"])
        self.assertEqual(self.record["reference_target"]["exact_greedy_decisions"], 32)
        self.assertEqual(self.record["reference_target"]["exact_beam_ids"], [644, 4598, 229, 9685, 5, 1])

    def test_coverage_arithmetic_retains_both_failed_thresholds(self):
        """New runtime guards stay in the denominator; the exclusion policy is unchanged."""
        coverage = self.record["coverage"]
        parent = json.loads((MEASUREMENTS / coverage["parent_record"]).read_text())
        self.assertEqual(coverage["policy"], parent["policy"])
        self.assertFalse(coverage["exclusion_policy_changed"])
        self.assertTrue(coverage["whole_test_file_exclusions_qualified"])
        self.assertFalse(coverage["production_only_measurement"])
        self.assertFalse(coverage["production_only_exclusions_qualified"])
        self.assertEqual(coverage["tests"], {"passed": 1140, "failed": 0, "ignored": 29})
        self.assertEqual(coverage["unit_tests"], {"passed": 1038, "failed": 0, "ignored": 3, "identities": 1041})
        self.assertEqual(coverage["reported_files"], 168)
        self.assertEqual(coverage["excluded_files"], 79)
        for scope in ("crate", "shared"):
            values = coverage["physical_whole_file_diagnostic"][scope]
            previous = parent["physical_whole_file_diagnostic"][scope]
            self.assertAlmostEqual(values["percent"], values["covered"] * 100 / values["lines"])
            self.assertLess(values["covered"] * 100, values["lines"] * values["minimum_percent"])
            self.assertFalse(values["meets_threshold"])
            self.assertEqual(coverage["change_since_parent"][scope],
                             {key: values[key] - previous[key] for key in ("lines", "covered")})
        self.assertEqual(sum(coverage["independent_reauthentication"]["mapped_runtime_line_changes"].values()), 29)
        self.assertEqual(coverage["independent_reauthentication"]["new_excluded_file"],
                         "src/models/custom/state_contracts_tests.rs")


if __name__ == "__main__":
    unittest.main()
