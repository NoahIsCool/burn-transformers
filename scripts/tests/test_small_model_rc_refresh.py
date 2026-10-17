"""Validate source-bound RC correctness evidence and reject invalid timing promotion."""

import hashlib
import json
import math
from pathlib import Path
import re
import struct
import unittest

from repository_evidence import repository_evidence


MEASUREMENTS = Path(__file__).resolve().parents[2] / "acceptance/measurements"


class SmallModelRcRefreshTests(unittest.TestCase):
    """Authenticate retained native outputs without loading model artifacts."""

    @classmethod
    def setUpClass(cls):
        """Read the compact summary and the prior public-example output identities."""
        cls.record = json.loads((MEASUREMENTS / "small-model-rc-refresh-20260910.json").read_text())
        cls.examples = json.loads((MEASUREMENTS / "public-examples-20260909.json").read_text())

    def execution(self, relative):
        """Resolve a checked relative evidence path and verify every native log hash.

        The execution records and their logs are repository-only evidence, so
        a test that needs them is skipped in the published crate.
        """
        path = Path(relative)
        self.assertFalse(path.is_absolute())
        self.assertNotIn("..", path.parts)
        record = json.loads(repository_evidence(MEASUREMENTS / path).read_text())
        self.assertEqual(record["source_before"], self.record["source"])
        self.assertEqual(record["source_before"], record["source_after"])
        self.assertEqual(record["executable_sha256_before"], record["executable_sha256_after"])
        self.assertRegex(record["executable_sha256_before"], r"^[0-9a-f]{64}$")
        self.assertEqual(set(record["logs"]), {"stdout.log", "stderr.log"})
        for filename, identity in record["logs"].items():
            self.assertIn(filename, ("stdout.log", "stderr.log"))
            data = repository_evidence(MEASUREMENTS / path.parent / filename).read_bytes()
            self.assertEqual(len(data), identity["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), identity["sha256"])
        return record, MEASUREMENTS / path.parent

    def test_native_success_does_not_promote_release_or_timings(self):
        """All four real tests and both examples succeeded at the exact candidate tree."""
        self.assertTrue(self.record["passed"])
        self.assertFalse(self.record["advertised_supported"])
        self.assertEqual(self.record["source"]["commit"], "50c80a246bc52706acdb44f8dea116dd48a21467")
        self.assertEqual(self.record["source"]["runtime_src_tree"], "b4d4a96ca38fdb709bce07b6fd384d497e523f98")
        self.assertEqual(len(self.record["runs"]), 5)
        self.assertEqual(sum(run["passed_test_cases"] or 0 for run in self.record["runs"]), 4)
        for run in self.record["runs"]:
            execution, directory = self.execution(run["execution_record"])
            self.assertEqual(execution["process_exit_code"], 0)
            self.assertEqual(run["process_exit_code"], 0)
            self.assertFalse(run["accepted_for_performance"])
            if run["passed_test_cases"] is not None:
                self.assertIn(f"test result: ok. {run['passed_test_cases']} passed; 0 failed", (directory / "stdout.log").read_text())
                self.assertIn("--test-threads=1", execution["argv"])

    def test_flex_full_vectors_match_native_rows(self):
        """Every native F32 comparison, including padding and boundary, meets tolerance."""
        run = next(run for run in self.record["runs"] if run["name"] == "flan-flex-oracle")
        _, directory = self.execution(run["execution_record"])
        pattern = r"Flex/F32 (\w+): cosine=([0-9.]+) max_abs=([0-9.]+) failures_1e-4=(\d+) actual_f32le_sha256=([0-9a-f]{64})"
        rows = re.findall(pattern, (directory / "stderr.log").read_text())
        self.assertEqual(len(rows), 7)
        self.assertEqual({row[0] for row in rows}, {"encoder", "decoder", "first", "cached", "teacher", "padded", "boundary"})
        for row, saved in zip(rows, self.record["flan_vectors"], strict=True):
            self.assertEqual(saved, dict(name=row[0], cosine=float(row[1]), maximum_absolute_error=float(row[2]), failures_at_1e_4=int(row[3]), actual_f32le_sha256=row[4]))
            self.assertGreaterEqual(saved["cosine"], 0.999)
            self.assertEqual(saved["failures_at_1e_4"], 0)

    def test_build_metadata_keeps_benchmark_and_correctness_features_distinct(self):
        """The public timing attempt used Metal alone; correctness also enabled Flex."""
        self.assertEqual(self.record["compiler"]["rustc_release"], "1.92.0")
        self.assertEqual(self.record["compiler"]["rustc_commit"], "ded5c06cf21d2b93bffd5d884aa6e96934ee4234")
        self.assertEqual(self.record["compiler"]["llvm_version"], "21.1.3")
        self.assertEqual(self.record["compiler"]["burn_version"], "0.21.0")
        expected = ["std", "metal", "flex", "tokenizers", "chat-templates"]
        for run in self.record["runs"]:
            self.assertEqual(run["features"], expected)
        for index, build in enumerate(self.record["builds"]):
            features = expected if index < 2 else ["std", "metal", "tokenizers", "chat-templates"]
            self.assertEqual(build["features"], features)
            command = build["command"]
            self.assertEqual(command[command.index("--features") + 1].split(","), features)
            for flag in ("--locked", "--offline", "--release", "--no-default-features"):
                self.assertIn(flag, command)
            self.assertEqual(build["process_exit_code"], 0)

    def test_stress_is_populated_and_not_quality_or_timing(self):
        """The record proves 32K real history, not merely an allocated capacity."""
        run = next(run for run in self.record["runs"] if run["name"] == "qwen-q4-32k")
        _, directory = self.execution(run["execution_record"])
        line = next(line for line in (directory / "stderr.log").read_text().splitlines() if line.startswith("QWEN35_Q4_32K_STRESS "))
        stress = json.loads(line.split(" ", 1)[1])
        self.assertEqual(stress, self.record["qwen_q4_populated_stress"])
        self.assertEqual((stress["context_tokens"], stress["prefill_tokens"]), (32768, 32767))
        self.assertEqual(len(stress["chunk_seconds"]), 128)
        self.assertTrue(stress["overflow_atomic"])
        self.assertTrue(stress["reset_matches_stateless"])
        self.assertFalse(stress["quality_gate"])
        self.assertFalse(stress["standardized_benchmark"])
        self.assertEqual(stress["cache_payload_bytes"], stress["context_tokens"] * stress["kv_bytes_per_token"] + stress["recurrent_fixed_bytes"])

    def test_public_examples_keep_prior_exact_text(self):
        """Both unchanged public entry points retain their previously recorded stdout."""
        prior = {"qwen-example": self.examples["runs"][1], "flan-example": self.examples["runs"][0]}
        for run in self.record["runs"]:
            if run["name"] in prior:
                _, directory = self.execution(run["execution_record"])
                output = (directory / "stdout.log").read_bytes()
                self.assertEqual(output.decode(), prior[run["name"]]["stdout"])
                self.assertEqual(hashlib.sha256(output).hexdigest(), prior[run["name"]]["stdout_sha256"])

    def test_public_coordinator_correctness_survives_rejected_power_metadata(self):
        """Keep thirteen exact continuations but never accept the misleading AC label."""
        public = self.record["public_coordinator"]
        execution, _ = self.execution(public["execution_record"])
        raw = repository_evidence(MEASUREMENTS / public["raw_report"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), public["raw_report_sha256"])
        self.assertEqual(public["raw_report_sha256"], execution["benchmark"]["sha256"])
        report = json.loads(raw)
        self.assertFalse(public["accepted_for_performance"])
        self.assertTrue(public["correctness_passed"])
        self.assertIn("incorrectly declared AC power", public["rejection_reason"])
        self.assertIn("Battery Power", public["observed_power_before"])
        self.assertFalse(public["physical_power_transition_established"])
        self.assertEqual(public["power_probe_contexts"], {"before": "host, outside sandbox", "after": "sandboxed"})
        self.assertFalse(public["subsequent_attempt"]["model_inference_started"])
        self.assertEqual(public["subsequent_attempt"]["process_exit_code"], 1)
        natural = report["natural_greedy"]
        self.assertEqual(natural["mode"], "greedy_public_generate_causal_coordinator")
        self.assertIsNone(natural["first_token_latency"])
        self.assertIsNone(natural["model_only_time"])
        samples = natural["per_repetition_outputs"]
        self.assertEqual([sample["partition"] for sample in samples], ["warmup"] * 3 + ["measured"] * 10)
        for sample in samples:
            output = sample["output"]
            self.assertEqual(output["token_ids"], natural["expected_token_ids"])
            self.assertEqual(output["generated_length"], 32)
            self.assertEqual(output["finish_reason"], "length")
            self.assertEqual(len(output["log_probabilities"]), 32)
            self.assertTrue(all(math.isfinite(value) for value in output["log_probabilities"]))
            digest = hashlib.sha256(b"".join(struct.pack("<I", token) for token in output["token_ids"])).hexdigest()
            self.assertEqual(digest, public["tokens_u32le_sha256"])
            self.assertEqual(digest, output["tokens_u32le_sha256"])


if __name__ == "__main__":
    unittest.main()
