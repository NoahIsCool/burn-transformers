"""Verify committed benchmark arithmetic, hashes, provenance and scope offline."""

import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import unittest


ACCEPTANCE = Path(__file__).resolve().parents[2] / "acceptance"
MEASUREMENTS = ACCEPTANCE / "measurements"


class FlanBenchmarkRecordTests(unittest.TestCase):
    """Validate reporting independently from the expensive inference benchmark."""

    prefill_field = "encoder_decoder_prefill"

    def setUp(self):
        """Authenticate the retained raw report before inspecting its contents."""
        self.run = json.loads((MEASUREMENTS / "flan-t5-small-metal-benchmark-20260909-run.json").read_text())
        raw = (MEASUREMENTS / self.run["raw_report"]["file"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), self.run["raw_report"]["sha256"])
        self.report = json.loads(raw)

    def test_primary_checkpoint_identity_and_scope(self):
        """A measured run cannot silently become a supported checkpoint or per-phase peak."""
        primary = json.loads((ACCEPTANCE / self.run["primary_acceptance"]).read_text())
        provenance = self.report["provenance"]
        self.assertEqual(primary["repository"], provenance["repository"])
        self.assertEqual(primary["revision"], provenance["revision"])
        self.assertEqual(primary["sha256"], provenance["artifact_sha256"])
        self.assertEqual(self.run["source_commit"], provenance["crate_commit"]["value"])
        self.assertEqual(provenance["crate_worktree_status"]["value"], "")
        self.assertFalse(self.run["advertised_supported"])
        self.assertFalse(self.run["external_memory"]["phase_specific"])
        self.assertIsNone(self.report["memory"]["phase_peak_unified_bytes"])
        self.assertEqual(len(self.report["strict_load_report"]["applied"]), 190)
        self.assertFalse(self.report["strict_load_report"]["missing"])
        self.assertFalse(self.report["strict_load_report"]["unexpected"])

    def test_raw_samples_order_statistics_and_rates(self):
        """Recompute all warm summaries without copying the Rust implementation."""
        self.assertEqual(self.report["protocol"]["warmup_repetitions"], 3)
        self.assertEqual(self.report["protocol"]["measured_repetitions"], 10)
        self.assertEqual(self.report["protocol"]["decode_steps"], 128)
        self.assertEqual([series["source_length"] for series in self.report["warm_series"]], [32, 128, 512])
        for series in self.report["warm_series"]:
            for name in (self.prefill_field, "first_token_latency", "decode_model_only", "decode_including_greedy_readback"):
                phase = series[name]
                samples = phase["samples_seconds"]
                self.assertEqual(len(samples), 10)
                self.assertTrue(all(math.isfinite(value) and value > 0 for value in samples))
                self.assertAlmostEqual(phase["median_seconds"], statistics.median(samples), places=12)
                self.assertEqual(phase["p95_seconds"], max(samples))
                self.assertEqual(phase["minimum_seconds"], min(samples))
                self.assertEqual(phase["maximum_seconds"], max(samples))
                if phase["valid_tokens_per_sample"] is not None:
                    for seconds, rate in zip(samples, phase["tokens_per_second"], strict=True):
                        self.assertAlmostEqual(rate * seconds, phase["valid_tokens_per_sample"], places=9)
            for model, inclusive in zip(series["decode_model_only"]["samples_seconds"], series["decode_including_greedy_readback"]["samples_seconds"], strict=True):
                self.assertGreaterEqual(inclusive, model)

    def test_canonical_ids_and_cache_accounting(self):
        """Reconstruct token hashes and the physical T5 cache payload factors."""
        for series in self.report["warm_series"]:
            for key, hash_key in (("source_token_ids", "source_tokens_u32le_sha256"), ("predicted_token_ids", "predicted_tokens_u32le_sha256")):
                canonical = b"".join(struct.pack("<I", token) for token in series[key])
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), series[hash_key])
            self.assertEqual(series["per_repetition_prediction_hashes"], [series["predicted_tokens_u32le_sha256"]] * 10)
            cache = series["cache"]
            per_token = 8 * 2 * 6 * 64 * 4
            self.assertEqual(per_token, cache["self_kv_bytes_per_token_per_sequence"])
            self.assertEqual(cache["self_kv_allocated_bytes"], per_token * 129)
            self.assertEqual(cache["cross_kv_bytes"], per_token * series["source_length"])
            self.assertEqual(cache["retained_encoder_hidden_bytes"], 512 * 4 * series["source_length"])
            self.assertEqual(cache["total_tensor_payload_bytes"], cache["self_kv_allocated_bytes"] + cache["cross_kv_bytes"] + cache["retained_encoder_hidden_bytes"])


class QwenBenchmarkRecordTests(FlanBenchmarkRecordTests):
    """Apply the same measurement rules to Qwen's distinct hybrid cache."""

    prefill_field = "prefill"

    def setUp(self):
        """Verify the exact original Qwen report bytes before parsing evidence."""
        self.run = json.loads((MEASUREMENTS / "qwen3.5-0.8b-metal-benchmark-20260909-run.json").read_text())
        raw = (MEASUREMENTS / self.run["raw_report"]["file"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), self.run["raw_report"]["sha256"])
        self.report = json.loads(raw)

    def test_primary_checkpoint_identity_and_scope(self):
        """Retain all source identities and distinguish optional MTP from vision."""
        primary = json.loads((ACCEPTANCE / self.run["primary_acceptance"]).read_text())
        provenance = self.report["provenance"]
        self.assertEqual(primary["repository"], provenance["repository"])
        self.assertEqual(primary["revision"], provenance["revision"])
        self.assertEqual(primary["sha256"], {entry["relative_path"]: entry["sha256"] for entry in provenance["checkpoint_artifacts"]})
        self.assertEqual(self.run["source_commit"], provenance["crate_commit"]["value"])
        self.assertEqual(provenance["crate_worktree_status"]["value"], "")
        report = self.report["strict_load_report"]
        self.assertEqual(len(report["applied"]), 321)
        self.assertEqual(len(report["aliased"]), 1)
        self.assertEqual(len(report["dtype_converted"]), 320)
        self.assertEqual(len(report["allowlisted_ignored"]), 168)
        self.assertEqual(sum(name.startswith("mtp.") for name in report["allowlisted_ignored"]), 15)
        self.assertFalse(report["missing"])
        self.assertFalse(report["unexpected"])
        self.assertFalse(self.run["advertised_supported"])
        self.assertFalse(self.run["external_memory"]["phase_specific"])
        self.assertFalse(self.run["protocol"]["long_prefill_series_executed"])
        self.assertEqual(self.run["natural_greedy"]["mode"], "greedy_model_api_loop")
        self.assertGreaterEqual(self.report["cold_first_token"]["minimum_cached_reference_cosine"], 0.999)
        self.assertGreaterEqual(self.report["cold_first_token"]["minimum_teacher_reference_cosine"], 0.999)

    def test_canonical_ids_and_cache_accounting(self):
        """Reconstruct prediction hashes and F16 KV/F32 recurrent physical bytes."""
        for series in self.report["warm_series"]:
            for key, hash_key in (("source_token_ids", "source_tokens_u32le_sha256"), ("predicted_token_ids", "predicted_tokens_u32le_sha256")):
                canonical = b"".join(struct.pack("<I", token) for token in series[key])
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), series[hash_key])
            self.assertEqual(series["per_repetition_prediction_hashes"], [series["predicted_tokens_u32le_sha256"]] * 10)
            self.assertEqual(len(series["predicted_token_ids"]), 129)
            self.assertEqual(series["cache_capacity"], series["source_length"] + 128)
            cache = series["cache"]["analytical"]
            kv = 6 * 2 * 2 * 256 * 2 * series["cache_capacity"]
            matrix = 18 * 16 * 128 * 128 * 4
            convolution = 18 * 6144 * 3 * 2
            self.assertEqual(cache["kv_allocated_bytes"], kv)
            self.assertEqual(cache["recurrent_matrix_bytes"], matrix)
            self.assertEqual(cache["convolution_history_bytes"], convolution)
            self.assertEqual(cache["total_tensor_payload_bytes"], kv + matrix + convolution)
            self.assertEqual(series["cache"]["observed_kv_bytes"], kv)
            self.assertEqual(series["cache"]["observed_recurrent_matrix_bytes"], matrix)
            self.assertEqual(series["cache"]["observed_convolution_history_bytes"], convolution)


if __name__ == "__main__":
    unittest.main()
