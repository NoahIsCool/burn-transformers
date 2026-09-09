"""Authenticate separate, sampled process-memory evidence without models or GPU."""

import collections
import hashlib
import json
import math
from pathlib import Path
import struct
import unittest


MEASUREMENTS = Path(__file__).resolve().parents[2] / "acceptance" / "measurements"
SUMMARY = "small-model-metal-memory-20260909.json"
PREFLIGHT = "small-model-metal-memory-20260909-source-preflight.json"
COMMIT = "6226170f25a4760d59ba1b391d86e3bce46a1bb7"
MODELS = ("qwen3.5-0.8b", "flan-t5-small")
METRICS = ("physical_footprint_bytes", "resident_bytes")


def file_identity(name):
    """Hash the retained immutable JSON bytes, not a reserialization of them."""

    data = (MEASUREMENTS / name).read_bytes()
    return {"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read_json(name):
    """Read a compact committed record; the large raw sample trace stays external."""

    return json.loads((MEASUREMENTS / name).read_text())


def phase_extremum(phase, metric):
    """Keep a selected maximum with its own idle baseline and observation bracket."""

    observed = phase["observed"]
    baseline = phase["idle_baseline"]
    return {
        "phase_id": phase["id"],
        "sampled_maximum_bytes": observed["sampled_maximum_bytes"][metric],
        "idle_median_bytes": baseline["baseline_bytes"][metric]["median"],
        "sampled_maximum_minus_idle_median_bytes": phase["sampled_maximum_minus_idle_median_bytes"][metric],
        "idle_start_ns": baseline["start_ns"],
        "idle_end_ns": baseline["end_ns"],
        "observed_start_ns": observed["start_ns"],
        "observed_end_ns": observed["end_ns"],
        "successful_samples": observed["successful_samples"],
        "maximum_successful_sampling_or_edge_gap_ns": observed["maximum_successful_sampling_or_edge_gap_ns"],
    }


def phase_groups(report):
    """Partition occurrence order per name; first three repetitions are warmups."""

    by_name = collections.defaultdict(list)
    for phase in report["phases"]:
        by_name[phase["name"]].append(phase)
    result = []
    for name, phases in by_name.items():
        if name in ("strict_load", "cold_prefill"):
            groups = [("single", phases, 0)]
        else:
            groups = [("warmup", phases[:3], 0), ("measured", phases[3:], 3)]
        for role, selected, first in groups:
            result.append({
                "phase_name": name,
                "role": role,
                "occurrence_indices_zero_based": list(range(first, first + len(selected))),
                "phase_ids": [phase["id"] for phase in selected],
                "successful_active_samples": sum(phase["observed"]["successful_samples"] for phase in selected),
                "successful_idle_samples": sum(phase["idle_baseline"]["successful_samples"] for phase in selected),
                "probe_errors": sum(phase[stage]["probe_errors"] for phase in selected for stage in ("idle_baseline", "observed")),
                "maximum_active_sampling_or_edge_gap_ns": max(phase["observed"]["maximum_successful_sampling_or_edge_gap_ns"] for phase in selected),
                "maximum_idle_sampling_or_edge_gap_ns": max(phase["idle_baseline"]["maximum_successful_sampling_or_edge_gap_ns"] for phase in selected),
                "maxima": {
                    metric: phase_extremum(max(selected, key=lambda phase: phase["observed"]["sampled_maximum_bytes"][metric]), metric)
                    for metric in METRICS
                },
            })
    return result


class SmallModelMemoryRecordTests(unittest.TestCase):
    """Historical memory claims remain traceable without rerunning the benchmark."""

    @classmethod
    def setUpClass(cls):
        """Read the two small reports once; no raw traces or checkpoints are needed."""

        cls.summary = read_json(SUMMARY)
        cls.preflight = read_json(PREFLIGHT)
        cls.models = {}
        for name in MODELS:
            stem = f"{name}-metal-memory-20260909"
            cls.models[name] = (read_json(f"{stem}-supervisor.json"), read_json(f"{stem}-run.json"))

    def test_exact_retained_bytes_are_authenticated(self):
        """Preserve the supervisor and benchmark output bytes from the real runs."""

        expected = {
            "qwen3.5-0.8b": ("f9e68ae922a10f92e620d8885506964b0a6405e0e400af5215a0d4c9dc8bcb34", "8bf1218997e6bc8187a292afe95db1fc8dfdb9d251ddc4df9eb14c45d43dd4a0"),
            "flan-t5-small": ("d5b7affbb419592c7d0bc6cbae2a64884c4aa26dae2c733d42cd2619485294df", "4a5824067a9247e5971868254c9b5f9e854b6972aa37530c979b3a061776a989"),
        }
        self.assertEqual(self.summary["source_preflight"], file_identity(PREFLIGHT))
        for name, record in self.summary["models"].items():
            for field, digest in zip(("supervisor_report", "instrumented_benchmark_report"), expected[name], strict=True):
                self.assertEqual(record[field], file_identity(record[field]["file"]))
                self.assertEqual(record[field]["sha256"], digest)

    def test_exact_source_and_execution_protocol(self):
        """Both instrumented binaries came from the same clean integration revision."""

        self.assertEqual(self.preflight["integration_commit"], COMMIT)
        self.assertEqual(self.preflight["runtime_source_tree_git_oid"], "11d33a072124025fba5048f8a49d58a23aa057a6")
        self.assertEqual(self.summary["source_commit"], COMMIT)
        for name, (report, run) in self.models.items():
            record = self.summary["models"][name]
            provenance = run["provenance"]
            self.assertEqual(provenance["crate_commit"], {"error": None, "value": COMMIT})
            self.assertEqual(provenance["crate_worktree_status"], {"error": None, "value": ""})
            self.assertEqual(provenance["cargo_features"], self.preflight["features"])
            self.assertIn("rustc 1.92.0", provenance["rustc"]["value"])
            self.assertEqual(provenance["profile"], "release (debug assertions disabled)")
            self.assertEqual(record["provenance"], provenance)
            self.assertEqual(record["exact_command"], provenance["environment"]["command"])
            self.assertEqual(record["supervised_command_argv"], report["command"])
            executable = Path(report["command"][0]).name
            self.assertEqual(provenance["benchmark_executable_sha256"], self.preflight["benchmark_executable_sha256"][executable])
            for source, field in (("Cargo.lock", "cargo_lock_sha256"), ("tests/support/benchmark_metrics.rs", "metric_helper_sha256")):
                self.assertEqual(provenance[field], self.preflight["sha256"][source])
            source = "tests/qwen35_benchmarks.rs" if name.startswith("qwen") else "tests/small_model_benchmarks.rs"
            self.assertEqual(provenance["benchmark_source_sha256"], self.preflight["sha256"][source])
            self.assertEqual(report["script_sha256"], self.preflight["sha256"]["scripts/profile_process_memory.py"])
            self.assertEqual(run["protocol"]["warmup_repetitions"], 3)
            self.assertEqual(run["protocol"]["measured_repetitions"], 10)
            self.assertEqual(run["protocol"]["decode_steps"], 128)
            self.assertEqual([series["source_length"] for series in run["warm_series"]], [32, 128, 512])
            self.assertFalse(run["strict_load_report"]["missing"])
            self.assertFalse(run["strict_load_report"]["unexpected"])

    def test_sampling_claims_and_lifetime_identity_are_honest(self):
        """A requested cadence is not an exact cadence, peak, or host/device split."""

        self.assertFalse(self.summary["timing_comparable_to_uninstrumented_series"])
        self.assertFalse(self.summary["sampled_maxima_are_exact_unsampled_peaks"])
        self.assertFalse(self.summary["advertised_supported"])
        self.assertIsNone(self.summary["host_device_memory_split"])
        expected_counts = {"qwen3.5-0.8b": 205985, "flan-t5-small": 87349}
        for name, (report, run) in self.models.items():
            record = self.summary["models"][name]
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["child_exit_code"], 0)
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["validation_failures"], [])
            self.assertIsNone(report["unfinished_phase"])
            self.assertEqual(report["sample_count"], expected_counts[name])
            self.assertEqual(record["whole_observation_window"], report["whole_observation_window"])
            self.assertEqual(record["identity"], report["identity"])
            self.assertGreater(report["identity"]["pid"], 0)
            self.assertGreater(report["identity"]["process_start_abstime"], 0)
            self.assertEqual(report["requested_interval_ns"], 1_000_000)
            self.assertEqual(report["requested_idle_baseline_ns"], 10_000_000)
            self.assertEqual(record["raw_trace"]["sha256"], report["raw_samples"]["sha256"])
            self.assertEqual(record["raw_trace"]["bytes"], report["raw_samples"]["bytes"])
            self.assertTrue(record["raw_trace"]["streaming_identity_count_and_hash_revalidated"])
            self.assertFalse(report["sampled_maxima_are_exact_unsampled_peaks"])
            self.assertFalse(report["timing_comparable_to_uninstrumented_series"])
            self.assertIsNone(report["host_device_memory_split"])
            self.assertTrue(report["kernel_lifetime_counter_is_not_a_phase_peak"])
            self.assertTrue(run["process_memory_instrumentation_enabled"])
            self.assertFalse(run["timing_comparable_to_uninstrumented_series"])
            self.assertGreaterEqual(report["observed_kernel_reported_lifetime_max_physical_footprint_bytes"], report["whole_observation_window"]["sampled_maximum_bytes"]["physical_footprint_bytes"])

    def test_phase_brackets_baselines_and_signed_deltas(self):
        """Each phase owns its measured idle baseline and conservative IPC bracket."""

        for report, _ in self.models.values():
            self.assertEqual(report["phase_count"], 93)
            self.assertEqual([phase["id"] for phase in report["phases"]], list(range(93)))
            previous_end = report["whole_observation_window"]["start_ns"]
            for phase in report["phases"]:
                self.assertTrue(phase["valid"])
                for stage in ("idle_baseline", "observed"):
                    window = phase[stage]
                    self.assertGreaterEqual(window["start_ns"], previous_end)
                    self.assertEqual(window["duration_ns"], window["end_ns"] - window["start_ns"])
                    self.assertGreater(window["duration_ns"], 0)
                    self.assertEqual(window["probe_errors"], 0)
                    self.assertFalse(window["exact_unsampled_peak"])
                    self.assertGreaterEqual(window["successful_samples"], 5 if stage == "idle_baseline" else 1)
                    first, last = window["first_successful_sample_ns"], window["last_successful_sample_ns"]
                    self.assertLessEqual(window["start_ns"], first)
                    self.assertLessEqual(last, window["end_ns"])
                    gap = window["maximum_successful_sampling_or_edge_gap_ns"]
                    self.assertGreaterEqual(gap, max(first - window["start_ns"], window["end_ns"] - last))
                    if window["successful_samples"] > 1:
                        mean = (last - first) / (window["successful_samples"] - 1)
                        self.assertAlmostEqual(window["mean_successful_sample_interval_ns"], mean, places=6)
                        self.assertGreaterEqual(gap, mean)
                    previous_end = window["end_ns"]
                for metric in METRICS:
                    baseline = phase["idle_baseline"]["baseline_bytes"][metric]
                    self.assertLessEqual(baseline["minimum"], baseline["median"])
                    self.assertLessEqual(baseline["median"], baseline["maximum"])
                    delta = phase["observed"]["sampled_maximum_bytes"][metric] - baseline["median"]
                    self.assertTrue(math.isfinite(delta))
                    self.assertEqual(delta, phase["sampled_maximum_minus_idle_median_bytes"][metric])
            self.assertLessEqual(previous_end, report["whole_observation_window"]["end_ns"])
        cold = self.models["flan-t5-small"][0]["phases"][1]
        self.assertLess(cold["sampled_maximum_minus_idle_median_bytes"]["physical_footprint_bytes"], 0)

    def test_warmup_partition_and_summary_use_same_phase_baselines(self):
        """No warmup maximum or independent baseline can leak into measured maxima."""

        for name, (report, _) in self.models.items():
            groups = self.summary["models"][name]["phase_groups"]
            self.assertEqual(groups, phase_groups(report))
            self.assertEqual(len(groups), 16)
            self.assertEqual(sorted(identity for group in groups for identity in group["phase_ids"]), list(range(93)))
            for group in groups:
                expected = {"single": [0], "warmup": list(range(3)), "measured": list(range(3, 13))}[group["role"]]
                self.assertEqual(group["occurrence_indices_zero_based"], expected)
                self.assertEqual(len(group["phase_ids"]), len(expected))
                self.assertEqual(group["probe_errors"], 0)

    def test_deterministic_outputs_and_analytical_cache_bytes(self):
        """Retain canonical complete token hashes and distinct physical cache factors."""

        for name, (_, run) in self.models.items():
            for series in run["warm_series"]:
                for ids, key in ((series["source_token_ids"], "source_tokens_u32le_sha256"), (series["predicted_token_ids"], "predicted_tokens_u32le_sha256")):
                    digest = hashlib.sha256(b"".join(struct.pack("<I", token) for token in ids)).hexdigest()
                    self.assertEqual(digest, series[key])
                self.assertEqual(series["per_repetition_prediction_hashes"], [series["predicted_tokens_u32le_sha256"]] * 10)
                if name.startswith("qwen"):
                    cache = series["cache"]["analytical"]
                    kv = 6 * 2 * 2 * 256 * 2 * (series["source_length"] + 128)
                    matrix, convolution = 18 * 16 * 128 * 128 * 4, 18 * 6144 * 3 * 2
                    self.assertEqual(cache["kv_allocated_bytes"], kv)
                    self.assertEqual(cache["recurrent_matrix_bytes"], matrix)
                    self.assertEqual(cache["convolution_history_bytes"], convolution)
                    self.assertEqual(cache["total_tensor_payload_bytes"], kv + matrix + convolution)
                else:
                    cache = series["cache"]
                    per_token = 8 * 2 * 6 * 64 * 4
                    self.assertEqual(cache["self_kv_allocated_bytes"], per_token * 129)
                    self.assertEqual(cache["cross_kv_bytes"], per_token * series["source_length"])
                    self.assertEqual(cache["retained_encoder_hidden_bytes"], 512 * 4 * series["source_length"])
            natural = run["natural_greedy"]
            digest = hashlib.sha256(b"".join(struct.pack("<I", token) for token in natural["token_ids"])).hexdigest()
            self.assertEqual(natural["tokens_u32le_sha256"], digest)


if __name__ == "__main__":
    unittest.main()
