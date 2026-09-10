"""Check compact September 10 measurements without running model inference.

The retained files are derived summaries, not copies of the large raw reports.
Their manifest hashes bind the committed bytes; each execution's report hash
separately identifies its original local report. Neither identity is a support
declaration or proof of an exact unsampled process-memory peak.
"""

import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import unittest


MEASUREMENTS = Path(__file__).resolve().parents[2] / "acceptance" / "measurements"
ROOT = MEASUREMENTS / "small-model-performance-20260910"


def read_record(name):
    """Read one fixed, committed JSON fixture without following external paths."""
    return json.loads((ROOT / name).read_text())


def timing_nodes(value):
    """Find retained native timing summaries regardless of model-specific nesting."""
    if isinstance(value, dict):
        if "samples_seconds" in value and "median_seconds" in value:
            yield value
        for item in value.values():
            yield from timing_nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from timing_nodes(item)


class SmallModelRefreshRecords(unittest.TestCase):
    """Preserve measured arithmetic and the narrower interpretation of each run."""

    def test_manifest_binds_every_compact_record(self):
        """An omitted or edited summary cannot silently retain the old manifest."""
        manifest = read_record("manifest.json")
        self.assertFalse(manifest["advertised_supported"])
        self.assertFalse(manifest["release_gate_promotion"])
        self.assertEqual(set(manifest["records"]), {
            path.name for path in ROOT.glob("*.json") if path.name != "manifest.json"
        })
        for name, identity in manifest["records"].items():
            payload = (ROOT / name).read_bytes()
            self.assertEqual(len(payload), identity["bytes"], name)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), identity["sha256"], name)
        self.assertEqual(manifest["qwen_historical_timing_gate"], "confirmed_failure_unresolved")

    def test_all_native_timing_summaries_recompute(self):
        """Retained samples reproduce medians, nearest-rank p95 and per-sample rates."""
        count = 0
        for path in ROOT.glob("*-summary.json"):
            report = json.loads(path.read_text())
            for phase in timing_nodes(report.get("benchmark", {})):
                with self.subTest(file=path.name):
                    samples = phase["samples_seconds"]
                    self.assertEqual(len(samples), 10)
                    self.assertTrue(all(math.isfinite(value) and value > 0 for value in samples))
                    self.assertAlmostEqual(phase["median_seconds"], statistics.median(samples), places=12)
                    self.assertEqual(phase["p95_seconds"], max(samples))
                    self.assertEqual(phase["minimum_seconds"], min(samples))
                    self.assertEqual(phase["maximum_seconds"], max(samples))
                    if phase.get("valid_tokens_per_sample") is not None:
                        for seconds, rate in zip(samples, phase["tokens_per_second"], strict=True):
                            self.assertAlmostEqual(seconds * rate, phase["valid_tokens_per_sample"], places=9)
                    count += 1
        self.assertGreater(count, 90)

    def test_build_run_and_correctness_identities_agree(self):
        """Each run is source-stable, binary-stable, successful and strictly loaded."""
        names = ("qwen-public", "qwen-short", "qwen-short-rerun", "flan-short",
                 "qwen-memory", "flan-memory", "qwen-baseline",
                 "qwen-controlled-current", "flan-flex")
        for name in names:
            with self.subTest(run=name):
                record = read_record(name + "-summary.json")
                execution, report = record["execution"], record["benchmark"]
                self.assertEqual(execution["process_exit_code"], 0)
                self.assertEqual(execution["source_before"], execution["source_after"])
                self.assertEqual(execution["executable_sha256_before"], execution["executable_sha256_after"])
                provenance = report["provenance"]
                self.assertEqual(execution["source_before"]["commit"], provenance["crate_commit"]["value"])
                self.assertEqual(execution["executable_sha256_before"], provenance["benchmark_executable_sha256"])
                source = execution["source_before"]
                status_key = "transformer_worktree_status" if "transformer_worktree_status" in source else "worktree_status"
                self.assertEqual(source[status_key], "")
                if "crate_worktree_status" in provenance:
                    self.assertEqual(provenance["crate_worktree_status"]["value"], "")
                self.assertTrue(execution["timing_power_endpoints_valid"])
                self.assertFalse(execution["power"]["continuous_monitoring"])
                self.assertEqual(report["strict_load_report"]["missing"], 0)
                self.assertEqual(report["strict_load_report"]["unexpected"], 0)
                natural = report.get("natural_greedy")
                if natural and natural.get("token_ids") is not None:
                    payload = b"".join(struct.pack("<I", token) for token in natural["token_ids"])
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), natural["tokens_u32le_sha256"])
                for series in report.get("warm_series", []):
                    self.assertEqual(series["per_repetition_prediction_hashes"],
                                     [series["predicted_tokens_u32le_sha256"]] * 10)

    def test_public_coordinator_retains_all_thirteen_outputs(self):
        """The public boundary has no separate TTFT or model-only timing claim."""
        record = read_record("qwen-public-summary.json")
        natural = record["benchmark"]["natural_greedy"]
        self.assertEqual(natural["mode"], "greedy_public_generate_causal_coordinator")
        self.assertIsNone(natural["first_token_latency"])
        self.assertIsNone(natural["model_only_time"])
        rows = natural["per_repetition_outputs"]
        self.assertEqual(len(rows), 13)
        self.assertEqual([row["partition"] for row in rows], ["warmup"] * 3 + ["measured"] * 10)
        for row in rows:
            output = row["output"]
            payload = b"".join(struct.pack("<I", token) for token in output["token_ids"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), output["tokens_u32le_sha256"])
            self.assertEqual(output["tokens_u32le_sha256"], "39ca0198c6fcae83b36801500e037213bd15de2c59b1a490dc3a35ebfa11e131")
            self.assertEqual(output["generated_length"], 32)
            self.assertEqual(len(output["log_probabilities"]), 32)
            self.assertTrue(all(math.isfinite(value) for value in output["log_probabilities"]))
            self.assertEqual(output["finish_reason"], "length")

    def test_controlled_comparison_does_not_waive_historical_failure(self):
        """Recompute direction-sensitive latency/throughput deltas from both runs."""
        comparison = read_record("qwen-controlled-historical-comparison.json")
        self.assertTrue(comparison["historical_failure"]["greater_than_ten_percent_confirmed"])
        self.assertFalse(comparison["historical_failure"]["waived"])
        self.assertFalse(comparison["historical_failure"]["cause_established"])
        self.assertIsNone(comparison["performance_policy_decision"])
        before = read_record(comparison["controlled_baseline_summary"])["benchmark"]
        after = read_record(comparison["controlled_current_summary"])["benchmark"]
        observed = comparison["controlled_comparison"]
        alerts = []
        deltas = []
        for left, right, row in zip(before["warm_series"], after["warm_series"], observed["rows"], strict=True):
            self.assertEqual(left["source_length"], right["source_length"])
            self.assertEqual(row["source_tokens"], right["source_length"])
            for metric in row["median_deltas_percent"]:
                old = left[metric]["median_seconds"]
                new = right[metric]["median_seconds"]
                # Throughput is inversely proportional to elapsed time.
                delta = 100 * ((old / new if metric.startswith("decode_") else new / old) - 1)
                self.assertAlmostEqual(row["median_deltas_percent"][metric], delta, places=11)
                deltas.append(abs(delta))
                tail_delta = 100 * (right[metric]["p95_seconds"] / left[metric]["p95_seconds"] - 1)
                self.assertAlmostEqual(row["p95_elapsed_deltas_percent"][metric], tail_delta, places=11)
                if tail_delta > 5:
                    alerts.append((row["source_tokens"], metric))
        self.assertAlmostEqual(max(deltas), observed["largest_absolute_warm_median_change_percent"], places=11)
        self.assertEqual(alerts, [(row["source_tokens"], row["metric"]) for row in observed["p95_five_percent_alerts"]])
        self.assertEqual(len(alerts), 3)
        flan = read_record("flan-flex-summary.json")["benchmark"]
        manifest_flan = read_record("manifest.json")["results"]["flan_flex"]
        self.assertTrue(manifest_flan["first_cpu_baseline"])
        for series, rate in zip(flan["warm_series"], manifest_flan["model_tokens_per_second_using_median_seconds"], strict=True):
            self.assertAlmostEqual(rate, 128 / series["decode_model_only"]["median_seconds"], places=9)

    def test_phase_memory_keeps_signed_deltas_and_non_exact_scope(self):
        """Sampling above idle is distinguished from lifetime and unsampled peaks."""
        for model in ("qwen", "flan"):
            report = read_record(model + "-memory-phase-summary.json")
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["child_exit_code"], 0)
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["validation_failures"], [])
            self.assertFalse(report["sampled_maxima_are_exact_unsampled_peaks"])
            self.assertFalse(report["timing_comparable_to_uninstrumented_series"])
            self.assertIsNone(report["host_device_memory_split"])
            self.assertEqual(report["phase_count"], 93)
            phase_ids = []
            for group in report["phase_groups"]:
                phase_ids.extend(group["phase_ids"])
                self.assertEqual(group["active_probe_errors"], 0)
                self.assertEqual(group["idle_probe_errors"], 0)
                for field in ("largest_absolute_samples", "largest_signed_deltas"):
                    for maximum in group[field].values():
                        self.assertIn(maximum["phase_id"], group["phase_ids"])
                        self.assertEqual(maximum["sampled_maximum_minus_idle_median_bytes"],
                                         maximum["sampled_maximum_bytes"] - maximum["baseline_median_bytes"])
            self.assertEqual(sorted(phase_ids), list(range(93)))


class WholeFileCoverageRecords(unittest.TestCase):
    """Partial exclusions cannot be renamed into a passing production gate."""

    def test_whole_file_scope_and_exact_thresholds(self):
        """Both versioned records retain their actual denominator and false flags."""
        for version, excluded in ((1, 39), (2, 78)):
            record = json.loads((MEASUREMENTS / f"whole-test-file-coverage-v{version}-20260910.json").read_text())
            self.assertTrue(record["whole_test_file_exclusions_qualified"])
            for name in ("advertised_supported", "production_only_exclusions_qualified",
                         "production_only_measurement", "release_gate_promotion",
                         "compiler_dependency_closure_verified"):
                self.assertFalse(record[name])
            self.assertEqual(record["excluded_file_count"], excluded)
            self.assertEqual(len(record["excluded_files"]), excluded)
            for kind in ("legacy_native_aggregate", "physical_before_exclusions", "physical_whole_file_diagnostic"):
                for group in ("crate", "shared"):
                    summary = record[kind][group]
                    self.assertAlmostEqual(summary["percent"], 100 * summary["covered"] / summary["lines"], places=12)
                    self.assertEqual(summary["meets_threshold"],
                                     100 * summary["covered"] >= summary["minimum_percent"] * summary["lines"])
            for group in ("crate", "shared"):
                before = record["physical_before_exclusions"][group]
                after = record["physical_whole_file_diagnostic"][group]
                files = [row for row in record["excluded_files"]
                         if group == "crate" or not row["path"].startswith("src/models/")]
                self.assertEqual(before["lines"] - after["lines"], sum(row["mapped_lines"] for row in files))
                self.assertEqual(before["covered"] - after["covered"], sum(row["covered_lines"] for row in files))
                if version == 2:
                    self.assertFalse(after["meets_threshold"])
