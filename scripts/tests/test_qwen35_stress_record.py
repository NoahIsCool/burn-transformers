"""Check the arithmetic and provenance links of the committed 32K stress evidence.

These inexpensive checks detect inconsistent reporting, not inference errors.
Only the opt-in Metal test can establish execution of the authenticated model.
"""

import hashlib
import json
from pathlib import Path
import struct
import unittest


ACCEPTANCE = Path(__file__).resolve().parents[2] / "acceptance"
RECORD = "measurements/qwen3.5-0.8b-q4-32k-stress-20260909.json"


class QwenStressRecordTests(unittest.TestCase):
    """Keep immutable stress observations internally reproducible and scoped."""

    def setUp(self):
        """Read the small source-controlled evidence, never external artifacts."""
        self.record = json.loads((ACCEPTANCE / RECORD).read_text())

    def test_input_hash_and_actual_populated_length(self):
        """Reconstruct every stress token and check the complete canonical hash."""
        metrics = self.record["metrics"]
        pattern = self.record["input"]["pattern_ids"]
        context = metrics["context_tokens"]
        self.assertEqual(context, 32768)
        tokens = (pattern * ((context + len(pattern) - 1) // len(pattern)))[:context]
        payload = b"".join(struct.pack("<q", token) for token in tokens)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), metrics["input_i64_le_sha256"])
        self.assertEqual(metrics["prefill_tokens"] + 1, context)
        calls = (metrics["prefill_tokens"] + metrics["chunk_tokens"] - 1) // metrics["chunk_tokens"]
        self.assertEqual(calls, self.record["input"]["prefill_calls"])
        self.assertEqual(len(metrics["chunk_seconds"]), calls)
        self.assertTrue(all(value > 0 for value in metrics["chunk_seconds"]))
        self.assertLessEqual(sum(metrics["chunk_seconds"]), metrics["prefill_including_logit_validation_seconds"])

    def test_cache_geometry_and_memory_scope(self):
        """Validate physical tensor bytes without inventing measured device peaks."""
        metrics = self.record["metrics"]
        kv_per_token = 6 * 2 * 2 * 256 * 2
        fixed = 18 * (16 * 128 * 128 * 4 + 6144 * 3 * 2)
        self.assertEqual(metrics["kv_bytes_per_token"], kv_per_token)
        self.assertEqual(metrics["recurrent_fixed_bytes"], fixed)
        self.assertEqual(metrics["cache_payload_bytes"], kv_per_token * 32768 + fixed)
        self.assertIsNone(self.record["memory"]["idle_baseline_bytes"])
        self.assertFalse(self.record["memory"]["phase_specific_peaks_measured"])
        self.assertGreater(self.record["memory"]["peak_process_memory_footprint_bytes"], metrics["cache_payload_bytes"])

    def test_provenance_links_and_non_quality_scope(self):
        """Require a linked execution pass, keeping unmeasured quality gates open."""
        checkpoint = self.record["checkpoint"]
        source = json.loads((ACCEPTANCE / checkpoint["source_record"]).read_text())
        self.assertEqual(source["repository"], checkpoint["repository"])
        self.assertEqual(source["revision"], checkpoint["revision"])
        self.assertEqual(source["quantized_32k_execution_record"], RECORD)
        self.assertTrue(source["checks"]["quantized_long_context_32k_validated"])
        self.assertTrue(self.record["stress_passed"])
        self.assertTrue(self.record["companion_envelope"]["passed"])
        self.assertFalse(self.record["advertised_supported"])
        self.assertFalse(self.record["metrics"]["quality_gate"])
        self.assertFalse(self.record["metrics"]["standardized_benchmark"])
        self.assertEqual(self.record["timing_protocol"]["warmups"], 0)
        self.assertEqual(self.record["timing_protocol"]["repetitions"], 1)


if __name__ == "__main__":
    unittest.main()
