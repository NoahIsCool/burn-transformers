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
import re
import shlex
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
        summary = read_record("manifest.json")["results"]["qwen_public"]
        self.assertEqual(summary["total_call_median_seconds"], natural["total_time"]["median_seconds"])
        self.assertEqual(summary["total_call_p95_seconds"], natural["total_time"]["p95_seconds"])
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
            manifest_memory = read_record("manifest.json")["results"]["memory"]
            self.assertEqual(manifest_memory[model + "_observed_kernel_lifetime_high_water_bytes"],
                             report["observed_kernel_reported_lifetime_max_physical_footprint_bytes"])
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


class HistoricalWarmInferenceAudit(unittest.TestCase):
    """Authenticate the diagnostic arithmetic without touching live host artifacts."""

    @classmethod
    def setUpClass(cls):
        """Load only committed fixtures; historical absolute paths are evidence labels."""
        cls.audit = json.loads((MEASUREMENTS / "qwen-historical-warm-inference-audit-20260910.json").read_text())
        cls.original = json.loads((MEASUREMENTS / "qwen3.5-0.8b-metal-benchmark-20260909.json").read_text())
        cls.original_run = json.loads((MEASUREMENTS / "qwen3.5-0.8b-metal-benchmark-20260909-run.json").read_text())
        cls.rebuilt = read_record("qwen-baseline-summary.json")
        cls.current = read_record("qwen-controlled-current-summary.json")

    def test_committed_and_raw_record_identities_are_cross_bound(self):
        """Derived observations stay bound to exact committed sources and raw hashes."""
        expected = {
            "qwen3.5-0.8b-metal-benchmark-20260909.json",
            "qwen3.5-0.8b-metal-benchmark-20260909-run.json",
            *("small-model-performance-20260910/" + name for name in (
                "qwen-baseline-summary.json", "qwen-controlled-current-summary.json",
                "qwen-controlled-historical-comparison.json", "historical-source-build.json",
                "metal-build.json")),
        }
        records = self.audit["committed_record_identities"]
        self.assertEqual(set(records), expected)
        for name in expected:
            payload = (MEASUREMENTS / name).read_bytes()
            self.assertEqual(records[name], {
                "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            })
        raw = self.audit["raw_record_identities"]
        self.assertEqual(raw["original_report"]["sha256"], self.original_run["raw_report"]["sha256"])
        self.assertEqual(raw["original_report"]["bytes"], records["qwen3.5-0.8b-metal-benchmark-20260909.json"]["bytes"])
        for label, record in (("rebuilt", self.rebuilt), ("current", self.current)):
            for kind, filename in (("report", "benchmark.json"), ("execution", "execution.json"),
                                   ("runtime_controls", "runtime-controls.json")):
                self.assertEqual(raw[label + "_" + kind], record["evidence"]["records"][filename])
            self.assertEqual(raw[label + "_build"], record["build"]["record"])
        # The committed build copies are exact originals, not newly synthesized builds.
        for label, filename in (("rebuilt", "historical-source-build.json"), ("current", "metal-build.json")):
            self.assertEqual({key: raw[label + "_build"][key] for key in ("bytes", "sha256")},
                             records["small-model-performance-20260910/" + filename])

    def test_native_boundaries_protocol_and_work_are_matched(self):
        """A native-only process counter comparison cannot silently become Cargo timing."""
        boundary = self.audit["source_and_boundary_matches"]
        rebuilt = self.rebuilt["benchmark"]
        self.assertFalse(boundary["cargo_build_inside_timed_process"])
        self.assertTrue(boundary["original_vs_rebuilt_protocol_identical"])
        self.assertEqual(self.original["protocol"], rebuilt["protocol"])
        protocol = self.original["protocol"]
        self.assertEqual(boundary["warmup_repetitions_per_shape"], protocol["warmup_repetitions"])
        self.assertEqual(boundary["measured_repetitions_per_shape"], protocol["measured_repetitions"])
        self.assertEqual(boundary["forced_decode_steps"], protocol["decode_steps"])
        self.assertEqual(boundary["historical_commit"], self.original["provenance"]["crate_commit"]["value"])
        self.assertEqual(boundary["controlled_current_commit"], self.current["execution"]["source_before"]["commit"])
        for key, value in boundary["original_and_rebuilt_benchmark_sources"].items():
            self.assertEqual(value, self.original["provenance"][key], key)
            self.assertEqual(value, rebuilt["provenance"][key], key)
        differing = [key for key, value in self.original["provenance"].items()
                     if value != rebuilt["provenance"][key]]
        self.assertEqual(differing, boundary["original_vs_rebuilt_provenance_differing_keys"])
        self.assertEqual(differing, ["benchmark_executable_sha256", "environment"])
        for name, report in (("original", self.original), ("rebuilt_historical", rebuilt)):
            process = self.audit["processes"][name]
            self.assertEqual(process["command"], report["provenance"]["environment"]["command"])
            self.assertEqual(process["benchmark_executable_sha256"], report["provenance"]["benchmark_executable_sha256"])
            self.assertEqual(process["benchmark_source_commit"], boundary["historical_commit"])
            command = shlex.split(process["command"])
            position = command.index("/usr/bin/time")
            self.assertEqual(command[position + 1], "-l")
            self.assertIn("/target/release/deps/qwen35_benchmarks-", command[position + 2])
            self.assertEqual(command[position + 3:position + 8],
                             [boundary["test_filter"], *boundary["test_arguments"]])
            self.assertIn("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 15 filtered out;",
                          process["native_stdout"]["utf8"])
        self.assertEqual((boundary["native_test_count"], boundary["native_filtered_test_count"]), (1, 15))
        self.assertEqual(boundary["cold_reference_rows"], self.original["cold_first_token"]["reference_cached_decisions"])
        self.assertEqual(boundary["natural_generated_tokens"], len(self.original["natural_greedy"]["token_ids"]))
        self.assertEqual((boundary["natural_warmup_repetitions"], boundary["natural_measured_repetitions"]), (3, 10))
        self.assertEqual(self.original["natural_greedy"]["tokens_u32le_sha256"], boundary["original_and_rebuilt_natural_hash"])
        self.assertEqual(rebuilt["natural_greedy"]["tokens_u32le_sha256"], boundary["original_and_rebuilt_natural_hash"])
        self.assertEqual(boundary["source_lengths"], [row["source_length"] for row in rebuilt["warm_series"]])
        for left, right in zip(self.original["warm_series"], rebuilt["warm_series"], strict=True):
            for key in boundary["per_shape_work_fields_identical"]:
                if key in ("source_token_ids", "predicted_token_ids"):
                    # The compact record omits large vectors but retains their exact hash.
                    hash_key = "source_tokens_u32le_sha256" if key == "source_token_ids" else "predicted_tokens_u32le_sha256"
                    payload = b"".join(struct.pack("<I", token) for token in left[key])
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), right[hash_key])
                else:
                    self.assertEqual(left[key], right[key], key)

    def test_embedded_native_logs_authenticate_and_reproduce_process_counters(self):
        """Retired instructions and CPU totals come from exact whole-process logs."""
        labels = {
            "instructions_retired": "instructions retired", "cycles_elapsed": "cycles elapsed",
            "involuntary_context_switches": "involuntary context switches", "swaps": "swaps",
            "peak_memory_footprint_bytes": "peak memory footprint",
        }
        for name, process in self.audit["processes"].items():
            for key in ("timing_log", "native_stdout"):
                payload = process[key]["utf8"].encode()
                self.assertEqual(len(payload), process[key]["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), process[key]["sha256"])
            log = process["timing_log"]["utf8"]
            wall, user, system = map(float, re.search(
                r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", log).groups())
            expected = {"wall_seconds": wall, "user_seconds": user, "system_seconds": system,
                        "cpu_seconds": user + system,
                        **{key: int(re.search(r"(\d+)\s+" + label, log)[1]) for key, label in labels.items()}}
            self.assertEqual(process["counters"], expected, name)
            self.assertEqual(expected["swaps"], 0)
        original = self.audit["processes"]["original"]["timing_log"]
        self.assertEqual(original["sha256"], self.original_run["external_memory"]["timing_log_sha256"])
        rebuilt = self.audit["processes"]["rebuilt_historical"]
        for embedded, filename in (("timing_log", "stderr.log"), ("native_stdout", "stdout.log")):
            self.assertEqual({key: rebuilt[embedded][key] for key in ("sha256", "bytes")},
                             self.rebuilt["execution"]["logs"][filename])

    def test_disjoint_phase_sums_and_contributions_recompute_from_native_samples(self):
        """Nested prefill/decode intervals are never counted twice as wall time."""
        accounting = self.audit["phase_accounting"]
        for name, report in (("original", self.original), ("rebuilt_historical", self.rebuilt["benchmark"])):
            expected = {key: sum(sum(row[key]["samples_seconds"]) for row in report["warm_series"])
                        for key in ("prefill", "first_token_latency", "decode_model_only", "decode_including_greedy_readback")}
            expected["natural_greedy"] = sum(report["natural_greedy"]["total_time"]["samples_seconds"])
            expected["nonoverlapping_measured_inference"] = (expected["first_token_latency"]
                + expected["decode_including_greedy_readback"] + expected["natural_greedy"])
            expected["decode_outside_model_intervals"] = expected["decode_including_greedy_readback"] - expected["decode_model_only"]
            for key in ("artifact_authentication_seconds_separate_from_load", "post_run_artifact_authentication_seconds", "strict_load_seconds"):
                expected[key] = report[key]
            expected["cold_first_token"] = report["cold_first_token"]["seconds"]
            self.assertEqual(set(accounting[name]), set(expected))
            for key, value in expected.items():
                self.assertAlmostEqual(accounting[name][key], value, places=12, msg=key)
        before = self.audit["processes"]["original"]["counters"]
        after = self.audit["processes"]["rebuilt_historical"]["counters"]
        wall_delta = after["wall_seconds"] - before["wall_seconds"]
        self.assertAlmostEqual(accounting["total_process_wall_delta_seconds"], wall_delta, places=12)
        for key, delta in accounting["delta_seconds"].items():
            self.assertAlmostEqual(delta, accounting["rebuilt_historical"][key] - accounting["original"][key], places=12)
        for key, delta in accounting["counter_delta_percent"].items():
            self.assertAlmostEqual(delta, 100 * (after[key] / before[key] - 1), places=12)
        for field, phase in (("measured_inference_share_of_wall_delta_percent", "nonoverlapping_measured_inference"),
                             ("measured_model_decode_share_of_wall_delta_percent", "decode_model_only")):
            self.assertAlmostEqual(accounting[field], 100 * accounting["delta_seconds"][phase] / wall_delta, places=12)
        self.assertAlmostEqual(accounting["wall_delta_outside_retained_measured_inference_seconds"],
                               wall_delta - accounting["delta_seconds"]["nonoverlapping_measured_inference"], places=12)
        self.assertLess(accounting["delta_seconds"]["cold_first_token"], 0)
        self.assertIn("not pure GPU kernel time", accounting["conventions"]["decode_model_only"])

    def test_observations_and_follow_up_do_not_establish_cause_or_waive_failure(self):
        """Host discovery and a proposed diagnostic are not new benchmark evidence."""
        audit = self.audit
        for flag in ("advertised_supported", "release_gate_promotion", "new_benchmark_executed",
                     "cause_established", "historical_failure_waived"):
            self.assertFalse(audit[flag])
        self.assertEqual(audit["historical_timing_gate"], "confirmed_failure_unresolved")
        self.assertIsNone(audit["performance_policy_decision"])
        comparison = read_record("qwen-controlled-historical-comparison.json")
        self.assertEqual(audit["controlled_pair"]["largest_absolute_warm_median_change_percent"],
                         comparison["controlled_comparison"]["largest_absolute_warm_median_change_percent"])
        self.assertEqual(audit["controlled_pair"]["p95_five_percent_alert_count"],
                         len(comparison["controlled_comparison"]["p95_five_percent_alerts"]))
        observation = audit["binary_path_observation"]
        self.assertFalse(observation["matches_original"])
        self.assertEqual(observation["original_recorded_sha256"], self.original["provenance"]["benchmark_executable_sha256"])
        self.assertNotEqual(observation["sha256"], observation["original_recorded_sha256"])
        self.assertEqual(observation["observed_utc"], audit["recorded_utc"])
        self.assertIn(observation["path"], self.original["provenance"]["environment"]["command"])
        discovery = audit["profiler_discovery"]
        self.assertEqual(discovery["active_developer_directory"]["stdout"].strip(), "/Library/Developer/CommandLineTools")
        self.assertNotEqual(discovery["xctrace_lookup"]["exit_code"], 0)
        self.assertNotEqual(discovery["xctrace_shim"]["exit_code"], 0)
        self.assertIn("requires Xcode", discovery["xctrace_shim"]["stderr"])
        self.assertFalse(discovery["full_metal_timeline_available_through_active_developer_directory"])
        self.assertFalse(discovery["system_configuration_changed"])
        self.assertTrue(discovery["powermetrics"]["sampling_not_started"])
        self.assertTrue(discovery["powermetrics"]["sampling_privileges_not_tested"])
        follow_up = audit["minimal_follow_up_protocol"]
        self.assertFalse(follow_up["executed"])
        self.assertTrue(follow_up["diagnostic_only"])
        self.assertFalse(follow_up["waives_historical_failure"])
        self.assertIn("Do not substitute the memory-supervisor ACK", follow_up["phase_alignment"])
        self.assertIn("not counters for the timed decode intervals", audit["limitations"][0])


class WholeFileCoverageRecords(unittest.TestCase):
    """Partial exclusions cannot be renamed into a passing production gate."""

    def test_whole_file_scope_and_exact_thresholds(self):
        """Both versioned records retain their actual denominator and false flags."""
        for version, excluded in ((1, 39), (2, 78), (3, 78)):
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
                if version >= 2:
                    self.assertFalse(after["meets_threshold"])

    def test_contract_tests_improve_the_unchanged_diagnostic_denominator(self):
        """A passing replay retains the rejected attempt and narrow audit scope."""
        before = json.loads((MEASUREMENTS / "whole-test-file-coverage-v2-20260910.json").read_text())
        after = json.loads((MEASUREMENTS / "whole-test-file-coverage-v3-20260910.json").read_text())
        for group, increase in (("crate", 548), ("shared", 37)):
            old = before["physical_whole_file_diagnostic"][group]
            new = after["physical_whole_file_diagnostic"][group]
            self.assertEqual(new["lines"], old["lines"])
            self.assertEqual(new["covered"] - old["covered"], increase)
        self.assertEqual(after["tests"], {
            "passed": 1075, "failed": 0, "ignored": 29,
            "unit_identities": 976, "unit_passed": 973, "unit_ignored": 3,
        })
        self.assertEqual(after["compiler_inputs"], before["compiler_inputs"])
        self.assertEqual({row["path"] for row in after["excluded_files"]},
                         {row["path"] for row in before["excluded_files"]})
        audit = after["structural_test_audit"]
        self.assertTrue(audit["structural_eligibility"])
        self.assertEqual(len(set(audit["inspected_production_scope_files"])), 101)
        self.assertEqual(audit["blockers"], [])
        self.assertFalse(audit["macro_expansion_resolved"])
        self.assertFalse(audit["production_measurement_qualified"])
        failed = after["preceding_failed_attempt"]
        self.assertEqual(failed["native_exit_code"], 1)
        self.assertEqual(failed["unit_tests"], {"passed": 971, "failed": 1, "ignored": 3})
        for flag in ("coverage_adopted", "runtime_changed", "tolerances_changed"):
            self.assertFalse(failed[flag])
        self.assertEqual(failed["fix_validation"]["focused_parallel_runs"], 100)
        self.assertEqual(failed["fix_validation"]["full_parallel_runs"], 10)
        self.assertEqual(failed["fix_validation"]["full_serial_runs"], 1)

    def test_checkpoint_contract_refresh_keeps_scope_and_parent_identities(self):
        """The compact follow-up links exact test changes without a scope waiver."""
        record = json.loads((MEASUREMENTS / "checkpoint-contract-coverage-20260910.json").read_text())
        parent = json.loads((MEASUREMENTS / record["parent_record"]).read_text())
        self.assertEqual(record["policy"], parent["policy"])
        self.assertEqual(record["compiler_inputs"], parent["compiler_inputs"])
        for flag in ("advertised_supported", "release_gate_promotion", "production_only_measurement",
                     "production_only_exclusions_qualified", "compiler_dependency_closure_verified",
                     "exclusion_policy_changed", "production_sources_changed"):
            self.assertFalse(record[flag])
        paths = sorted(row["path"] for row in parent["excluded_files"])
        digest = hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(record["files"]["excluded_paths_sha256"], digest)
        self.assertEqual(record["files"]["excluded_count"], 78)
        self.assertEqual(record["files"]["retained_count"], 89)
        self.assertEqual(record["files"]["reported_count"], 167)
        old_hashes = {row["path"]: row["source_sha256"] for row in parent["excluded_files"]}
        self.assertEqual({row["path"] for row in record["test_source_changes"]}, {
            "src/checkpoint/tests.rs", "src/checkpoint/runtime/tests.rs",
            "src/checkpoint/quantization/tests.rs", "src/checkpoint/quantization/burnpack_tests.rs",
            "src/loaded_tests.rs",
        })
        for change in record["test_source_changes"]:
            self.assertEqual(change["parent_sha256"], old_hashes[change["path"]])
            self.assertNotEqual(change["source_sha256"], change["parent_sha256"])
        added = sum(row["tests_added"] for row in record["test_milestones"])
        self.assertEqual(added, 24)
        self.assertEqual(record["tests"]["unit_identities"], parent["tests"]["unit_identities"] + added)
        self.assertEqual(record["tests"]["passed"], 1099)
        self.assertEqual(record["tests"]["failed"], 0)
        self.assertEqual(record["tests"]["ignored"], 29)
        for kind in ("legacy_native_aggregate", "physical_before_exclusions", "physical_whole_file_diagnostic"):
            for group in ("crate", "shared"):
                count = record[kind][group]
                self.assertAlmostEqual(count["percent"], 100 * count["covered"] / count["lines"], places=12)
                self.assertEqual(count["meets_threshold"], 100 * count["covered"] >= count["minimum_percent"] * count["lines"])
        for group in ("crate", "shared"):
            current = record["physical_whole_file_diagnostic"][group]
            old = parent["physical_whole_file_diagnostic"][group]
            self.assertFalse(current["meets_threshold"])
            self.assertEqual(current["lines"], old["lines"])
            self.assertEqual(current["covered"] - old["covered"], 86)
            self.assertEqual(record["coverage_change_since_parent"][group], {"mapped_lines": 0, "covered_lines": 86})
        audit = record["structural_test_audit"]
        self.assertTrue(audit["structural_eligibility"])
        self.assertEqual(audit["inspected_production_file_count"], 101)
        self.assertEqual(audit["blockers"], [])
        self.assertFalse(audit["macro_expansion_resolved"])
        self.assertFalse(audit["production_measurement_qualified"])
